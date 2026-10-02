"""Host driver for the Triton twin of the multi-block flood fill.

Public API (identical to chapters/ch03_gpu_1blob_nblock/flood_fill.py):
    flood_fill(img, seed_x, seed_y, threads_per_block=256, blocks=None,
               bare=False, connectivity=4, radius=1,
               probe_layout="thread") -> MultiFloodFillResult
    max_blocks(threads_per_block=256, bare=False, connectivity=4, radius=1,
               probe_layout="thread") -> int

Same variants, validation, result fields and timing decomposition as the
Numba driver; see that module for what each option means. The differences
are the backend's:

- threads_per_block must also be a power of 2 (32, 64, 128, 256 or 512):
  a Triton program's lane count is a tl.arange length and its warp count
  must be a power of 2. Numba's other values raise ValueError here.
- blocks=None resolves to the largest grid whose programs are all
  co-resident (runtime.occupancy.max_coresident_programs, the twin of
  Numba's max_cooperative_grid_blocks), queried per compiled kernel and
  per threads_per_block. Triton's register counts differ from Numba's, so
  this number can differ from the Numba twin's; pin blocks to compare.
- Every launch is cooperative (launch_cooperative_grid=True), so an
  oversized grid is refused by the driver exactly as in Numba.
- Device memory is CuPy's; a grid barrier counter (int32[1]) is allocated
  with the other buffers and zeroed with the H2D copies.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import cupy as cp

from ....shared.bandwidth import model_bytes as _model_bytes, model_gb_s as _model_gb_s
from ...runtime import device_info, max_coresident_programs, sync, t
from .kernels import (
    multi_block_global_kernel, multi_block_global_bare_kernel,
    multi_block_global8_kernel, multi_block_global8_bare_kernel,
    multi_block_global8r2_kernel, multi_block_global8r2_bare_kernel,
    multi_block_global8wc_kernel, multi_block_global8wc_bare_kernel,
    NUM_COUNTERS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS, INTERIOR,
    BS_PROCESSED, BS_SMID,
)

# (bare, connectivity, radius, probe_layout) -> kernel, as in Numba.
_KERNELS = {
    (False, 4, 1, "thread"): multi_block_global_kernel,
    (True, 4, 1, "thread"): multi_block_global_bare_kernel,
    (False, 8, 1, "thread"): multi_block_global8_kernel,
    (True, 8, 1, "thread"): multi_block_global8_bare_kernel,
    (False, 8, 2, "thread"): multi_block_global8r2_kernel,
    (True, 8, 2, "thread"): multi_block_global8r2_bare_kernel,
    (False, 8, 1, "warp"): multi_block_global8wc_kernel,
    (True, 8, 1, "warp"): multi_block_global8wc_bare_kernel,
}

# Cap on the recorded per-level trace (1D int32 -> 8 MB max)
LEVEL_TRACE_CAPACITY = 2 ** 21

# Lane counts a Triton program can have (power-of-2 tl.arange and num_warps)
POW2_TPB = (32, 64, 128, 256, 512)


@dataclass
class MultiFloodFillResult:
    img: np.ndarray        # (width, height, 3) uint8, blob recolored blue
    visited: np.ndarray    # (width, height) int32, 1 where reached
    depth: np.ndarray      # (width, height) int32, BFS level per pixel
    owner: np.ndarray      # (width, height) int16, processing program id, -1
                           # unreached; empty for bare runs
    levels: int
    filled: int
    peak_level: int        # largest single frontier
    peak_occupancy: int    # max queue entries alive across two adjacent levels
    threads_per_block: int
    blocks: int            # resolved launch size (None -> co-resident max)
    bare: bool
    connectivity: int      # 4 (Manhattan waves) or 8 (Chebyshev waves)
    radius: int            # 1, or 2 for the guarded radius-2 twin
    probe_layout: str      # "thread" or "warp" (4 entries x 8 dirs per warp)
    # Work / balance
    processed: int
    cas_attempts: int
    interior: int          # radius-2 only: guard passes (else 0)
    processed_per_block: np.ndarray = field(repr=False)  # (blocks,) int64
    balance_min_max_pct: float  # 100 * min / max of per-block processed
    balance_cv_pct: float       # 100 * std / mean
    # Placement (observed via %smid, not assumed; -1s for bare)
    sm_ids: list
    distinct_sms: int
    # Per-level trace (empty for bare)
    level_sizes: np.ndarray = field(repr=False)
    level_trace_truncated: bool
    # Timing decomposition (ms)
    alloc_ms: float
    h2d_ms: float
    kernel_ms: float
    d2h_ms: float
    total_ms: float
    # Derived utilization metrics (zeroed for bare)
    thread_util_pct: float
    warp_engagement_pct: float
    lane_efficiency_pct: float
    grid_occupancy_pct: float   # 100 * blocks*tpb / (SMs * threads-per-SM)
    sm_utilization_pct: float   # 100 * distinct observed SMs / SM count
    discovery_redundancy: float
    neighbor_check_efficiency_pct: float
    # Derived bandwidth model (zeroed for bare; see shared/bandwidth.py)
    model_bytes: int
    model_gb_s: float


# (key, tpb) -> CompiledKernel of the warm-up launch; (key, tpb) -> capacity
_warmed = {}
_coop_cache = {}


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _launch(kernel_fn, blocks, tpb, instrumented, d_img, d_visited, d_depth,
            d_owner, d_queue, d_q_state, d_counters, d_stats, d_trace, d_bar,
            width, height):
    """One cooperative launch; returns the CompiledKernel."""
    common = dict(BLOCK=tpb, num_warps=tpb // 32, launch_cooperative_grid=True)
    if instrumented:
        return kernel_fn[(blocks,)](
            t(d_img), t(d_visited), t(d_depth), t(d_owner), t(d_queue),
            t(d_q_state), t(d_counters), t(d_stats), t(d_trace), t(d_bar),
            width, height, d_queue.shape[0], d_trace.shape[0], **common)
    return kernel_fn[(blocks,)](
        t(d_img), t(d_visited), t(d_depth), t(d_queue), t(d_q_state),
        t(d_counters), t(d_bar), width, height, d_queue.shape[0], **common)


def _tiny_args():
    """A single red pixel at (1, 1) of an 8x8 white image: terminates under
    any program count (no discoveries), so it is safe for 1-program
    compile launches."""
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    visited = np.zeros((8, 8), dtype=np.int32)
    visited[1, 1] = 1
    queue = np.zeros(64, dtype=np.int32)
    queue[0] = 1 * 8 + 1
    return (cp.asarray(tiny), cp.asarray(visited),
            cp.full((8, 8), -1, dtype=cp.int32),
            cp.zeros(NUM_COUNTERS, dtype=cp.int64), cp.asarray(queue),
            cp.asarray(np.array([1], dtype=np.int32)))


def _warmup(bare, connectivity=4, radius=1, probe_layout="thread",
            threads_per_block=256):
    """Compile each kernel once per threads_per_block, off the clock.

    Numba compiles one binary per kernel and warms it with a [1, 32]
    launch; a Triton program's lane count is a compile-time constant, so
    the twin warms the exact (kernel, tpb) binary the timed launch uses.
    The launch returns the CompiledKernel the occupancy query needs.
    """
    key = (bare, connectivity, radius, probe_layout)
    if (key, threads_per_block) in _warmed:
        return _warmed[(key, threads_per_block)]
    d_img, d_visited, d_depth, d_counters, d_queue, d_q = _tiny_args()
    d_bar = cp.zeros(1, dtype=cp.int32)
    kernel_fn = _KERNELS[key]
    if bare:
        compiled = _launch(kernel_fn, 1, threads_per_block, False, d_img,
                           d_visited, d_depth, None, d_queue, d_q,
                           d_counters, None, None, d_bar, 8, 8)
    else:
        d_owner = cp.full((8, 8), -1, dtype=cp.int16)
        d_stats = cp.zeros((1, 2), dtype=cp.int64)
        d_trace = cp.zeros(4, dtype=cp.int32)
        compiled = _launch(kernel_fn, 1, threads_per_block, True, d_img,
                           d_visited, d_depth, d_owner, d_queue, d_q,
                           d_counters, d_stats, d_trace, d_bar, 8, 8)
    sync()
    _warmed[(key, threads_per_block)] = compiled
    return compiled


def _coop_max_blocks(key, tpb):
    if (key, tpb) not in _coop_cache:
        _coop_cache[(key, tpb)] = max_coresident_programs(
            _warmed[(key, tpb)])
    return _coop_cache[(key, tpb)]


def kernel_info(threads_per_block=256, bare=False, connectivity=4, radius=1,
                probe_layout="thread"):
    """Registers, spills, shared bytes and warps of the compiled twin
    (runtime.occupancy.kernel_resources); compiles it on first call."""
    from ...runtime import kernel_resources

    return kernel_resources(_warmup(bare, connectivity, radius, probe_layout,
                                    threads_per_block))


def max_blocks(threads_per_block=256, bare=False, connectivity=4, radius=1,
               probe_layout="thread"):
    """The largest co-resident grid this GPU can host at threads_per_block
    (what blocks=None resolves to). Compiles the kernel on first call.
    Queried per kernel and per tpb, never assumed equal across twins."""
    _warmup(bare, connectivity, radius, probe_layout, threads_per_block)
    return _coop_max_blocks((bare, connectivity, radius, probe_layout),
                            threads_per_block)


def flood_fill(img_host, seed_x, seed_y, threads_per_block=256, blocks=None,
               bare=False, connectivity=4, radius=1, probe_layout="thread"):
    """Flood-fill the red blob containing (seed_x, seed_y) with N programs.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for bad inputs and RuntimeError if the GPU
    cannot host the requested cooperative launch (or if a structural
    tripwire fires, which would indicate a kernel bug).
    """
    if img_host.ndim != 3 or img_host.shape[2] != 3 or img_host.dtype != np.uint8:
        raise ValueError("img must be a (width, height, 3) uint8 array")
    width, height = img_host.shape[0], img_host.shape[1]
    if width * height >= 2 ** 31:
        raise ValueError("image too large for int32 linear pixel indices")
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError(f"seed ({seed_x}, {seed_y}) outside {width}x{height} image")
    if not _is_red(img_host, seed_x, seed_y):
        raise ValueError(f"seed pixel ({seed_x}, {seed_y}) is not red - nothing to fill")
    if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 512):
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 512], "
            f"got {threads_per_block}")
    if threads_per_block not in POW2_TPB:
        raise ValueError(
            f"threads_per_block must be a power of 2 for the Triton twin "
            f"(a program's lane count is a tl.arange length and num_warps "
            f"must be a power of 2): one of {POW2_TPB}, got "
            f"{threads_per_block}")
    if blocks is not None:
        if not isinstance(blocks, (int, np.integer)) or isinstance(blocks, bool):
            raise ValueError(f"blocks must be an int or None, got {blocks!r}")
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    if radius not in (1, 2):
        raise ValueError(f"radius must be 1 or 2, got {radius!r}")
    if radius == 2 and connectivity != 8:
        raise ValueError(
            "radius=2 requires connectivity=8 (the guarded ring-2 twin is "
            "built on the 8-connectivity kernel)")
    if probe_layout not in ("thread", "warp"):
        raise ValueError(
            f"probe_layout must be 'thread' or 'warp', got {probe_layout!r}")
    if probe_layout == "warp" and (connectivity != 8 or radius != 1):
        raise ValueError(
            "probe_layout='warp' requires connectivity=8 and radius=1 (the "
            "warp-cooperative twin exists only for the plain conn8 kernel)")

    _warmup(bare, connectivity, radius, probe_layout, threads_per_block)
    dev = device_info()
    sm_count = dev.sm_count

    key = (bare, connectivity, radius, probe_layout)
    kernel_fn = _KERNELS[key]
    coop_max = _coop_max_blocks(key, threads_per_block)
    if coop_max < 1:
        raise RuntimeError(
            f"this GPU cannot host even one cooperative block of this "
            f"kernel at {threads_per_block} threads (register pressure); "
            f"try a smaller threads_per_block")
    if blocks is None:
        launch_blocks = coop_max
    elif blocks > coop_max:
        raise RuntimeError(
            f"blocks={blocks} exceeds this GPU's cooperative-launch capacity "
            f"of {coop_max} blocks at {threads_per_block} threads "
            f"(grid.sync would deadlock)")
    else:
        launch_blocks = int(blocks)

    seed_lin = np.array([seed_x * height + seed_y], dtype=np.int32)
    trace_capacity = min(width * height, LEVEL_TRACE_CAPACITY)
    instrumented = not bare

    t_total0 = time.perf_counter()

    visited_host = np.zeros((width, height), dtype=np.int32)
    visited_host[seed_x, seed_y] = 1
    depth_host = np.full((width, height), -1, dtype=np.int32)
    counters_host = np.zeros(NUM_COUNTERS, dtype=np.int64)
    d_img = cp.empty(img_host.shape, dtype=cp.uint8)
    d_visited = cp.empty(visited_host.shape, dtype=cp.int32)
    d_depth = cp.empty(depth_host.shape, dtype=cp.int32)
    d_counters = cp.empty(NUM_COUNTERS, dtype=cp.int64)
    d_queue = cp.empty(width * height, dtype=cp.int32)
    d_bar = cp.empty(1, dtype=cp.int32)
    d_owner = d_stats = d_trace = None
    if instrumented:
        owner_host = np.full((width, height), -1, dtype=np.int16)
        d_owner = cp.empty(owner_host.shape, dtype=cp.int16)
        stats_host = np.zeros((launch_blocks, 2), dtype=np.int64)
        stats_host[:, BS_SMID] = -1
        d_stats = cp.empty(stats_host.shape, dtype=cp.int64)
        d_trace = cp.empty(trace_capacity, dtype=cp.int32)
    sync()
    t_h2d0 = time.perf_counter()

    d_img.set(img_host)
    d_visited.set(visited_host)
    d_depth.set(depth_host)
    d_counters.set(counters_host)
    d_queue[:1].set(seed_lin)
    d_q_state = cp.asarray(np.array([1], dtype=np.int32))
    d_bar.set(np.zeros(1, dtype=np.int32))
    if instrumented:
        d_owner.set(owner_host)
        d_stats.set(stats_host)
    sync()
    t_kernel0 = time.perf_counter()

    _launch(kernel_fn, launch_blocks, threads_per_block, instrumented, d_img,
            d_visited, d_depth, d_owner, d_queue, d_q_state, d_counters,
            d_stats, d_trace, d_bar, width, height)
    sync()
    t_d2h0 = time.perf_counter()

    counters = d_counters.get()
    if counters[OVERFLOW]:
        raise RuntimeError(
            "structural tripwire fired - this indicates a kernel bug: the "
            "queue cannot legitimately overflow")
    levels = int(counters[LEVELS])
    truncated = instrumented and levels > trace_capacity
    img_out = d_img.get()
    visited_out = d_visited.get()
    depth_out = d_depth.get()
    if instrumented:
        owner_out = d_owner.get()
        stats = d_stats.get()
        sizes = d_trace[:min(levels, trace_capacity)].get()
    else:
        owner_out = np.zeros((0, 0), dtype=np.int16)
        stats = np.zeros((launch_blocks, 2), dtype=np.int64)
        stats[:, BS_SMID] = -1
        sizes = np.zeros(0, dtype=np.int32)
    t_end = time.perf_counter()

    filled = int(counters[FILLED])
    processed = int(counters[PROCESSED])
    cas_attempts = int(counters[CAS_ATTEMPTS])
    interior = int(counters[INTERIOR])
    ppb = stats[:, BS_PROCESSED].copy()
    sm_ids = [int(s) for s in stats[:, BS_SMID]]
    distinct_sms = len({s for s in sm_ids if s >= 0})
    p_max = int(ppb.max()) if ppb.size else 0
    p_min = int(ppb.min()) if ppb.size else 0
    p_mean = float(ppb.mean()) if ppb.size else 0.0
    kernel_ms = (t_d2h0 - t_kernel0) * 1000
    grid_threads = launch_blocks * threads_per_block
    warps = grid_threads // 32
    ats = int(counters[ACTIVE_THREAD_SUM])
    aws = int(counters[ACTIVE_WARP_SUM])
    # Exact probe count: radius-2 pixels probe 8 always + 16 when interior.
    eff_probes = (8 * processed + 16 * interior if radius == 2
                  else connectivity * processed)
    mbytes = (_model_bytes(processed, cas_attempts, filled, True,
                           n_dirs=connectivity, probe_reads=eff_probes)
              if instrumented else 0)

    return MultiFloodFillResult(
        img=img_out,
        visited=visited_out,
        depth=depth_out,
        owner=owner_out,
        levels=levels,
        filled=filled,
        peak_level=int(counters[PEAK_LEVEL]),
        peak_occupancy=int(counters[PEAK_OCC]),
        threads_per_block=threads_per_block,
        blocks=launch_blocks,
        bare=bare,
        connectivity=connectivity,
        radius=radius,
        probe_layout=probe_layout,
        processed=processed,
        cas_attempts=cas_attempts,
        interior=interior,
        processed_per_block=ppb,
        balance_min_max_pct=(100.0 * p_min / p_max if p_max > 0 else 0.0),
        balance_cv_pct=(100.0 * float(ppb.std()) / p_mean
                        if p_mean > 0 else 0.0),
        sm_ids=sm_ids,
        distinct_sms=distinct_sms,
        level_sizes=sizes,
        level_trace_truncated=truncated,
        alloc_ms=(t_h2d0 - t_total0) * 1000,
        h2d_ms=(t_kernel0 - t_h2d0) * 1000,
        kernel_ms=kernel_ms,
        d2h_ms=(t_end - t_d2h0) * 1000,
        total_ms=(t_end - t_total0) * 1000,
        thread_util_pct=(100.0 * ats / (levels * grid_threads)
                         if instrumented and levels else 0.0),
        warp_engagement_pct=(100.0 * aws / (levels * warps)
                             if instrumented and levels else 0.0),
        lane_efficiency_pct=(100.0 * ats / (32 * aws)
                             if instrumented and aws else 0.0),
        grid_occupancy_pct=100.0 * grid_threads
        / (sm_count * dev.max_threads_per_sm),
        sm_utilization_pct=100.0 * distinct_sms / sm_count,
        discovery_redundancy=(cas_attempts / max(filled - 1, 1)
                              if instrumented else 0.0),
        neighbor_check_efficiency_pct=(100.0 * filled / eff_probes
                                       if instrumented and processed else 0.0),
        model_bytes=mbytes,
        model_gb_s=_model_gb_s(mbytes, kernel_ms),
    )
