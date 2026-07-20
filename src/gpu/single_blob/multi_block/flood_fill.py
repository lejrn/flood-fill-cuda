"""Host driver for the multi-block flood fill.

Public API:
    flood_fill(img, seed_x, seed_y, threads_per_block=256, blocks=None,
               bare=False, connectivity=4, radius=1,
               probe_layout="thread") -> MultiFloodFillResult

connectivity=4 explores right/down/left/up (Manhattan-distance waves,
diamond fronts); connectivity=8 adds the diagonals (Chebyshev waves,
square fronts — fewer, wider levels for the same blob). The two are
different algorithms: fills agree only on scenes without diagonal-only
gaps, and depth/levels always differ.

radius=2 (requires connectivity=8) selects the guarded radius-2 twin:
ring-1 is always probed first with the unchanged protocol; only pixels
whose entire ring-1 is in-bounds blob material additionally probe the 16
ring-2 cells. Fill set identical to plain conn8 (the guard keeps every
jump inside true 8-connectivity); depth/levels are its own — roughly
half the levels on solid blobs. Instrumented results report `interior`
(guard passes; ring-2 probes == 16 * interior).

probe_layout="warp" (requires connectivity=8, radius=1) selects the
warp-cooperative twin: each warp takes 4 queue entries and assigns each
lane one (entry, direction) pair, so all 32 probes issue in one round
instead of 8 lockstep loop iterations. Results are bit-identical to the
plain conn8 twin (same BFS graph — only work distribution changes).

blocks=None launches the maximum cooperative grid the GPU can host for the
chosen threads_per_block (queried from the compiled kernel, never hardcoded
— oversizing deadlocks grid.sync). An explicit block count is validated
against the same limit; blocks=1 is legal (grid.sync degenerates to a
block-wide barrier) and serves as the equivalence anchor in tests.

bare=True selects the uninstrumented twin for measuring instrumentation
overhead; such results carry timing, filled and levels but zeroed
work/balance/bandwidth metrics.

Every instrumented result records the %smid each block observed itself
running on (placement is reported, not assumed) and a derived bytes-moved
bandwidth model (see bandwidth.py for what that figure does and does not
mean).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from bandwidth import model_bytes as _model_bytes, model_gb_s as _model_gb_s
from kernels import (
    multi_block_global_kernel, multi_block_global_bare_kernel,
    multi_block_global8_kernel, multi_block_global8_bare_kernel,
    multi_block_global8r2_kernel, multi_block_global8r2_bare_kernel,
    multi_block_global8wc_kernel, multi_block_global8wc_bare_kernel,
    NUM_COUNTERS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS, INTERIOR,
    BS_PROCESSED, BS_SMID,
)
from numba import cuda

# (bare, connectivity, radius, probe_layout) -> kernel. All variants are
# twins of the 4-conn/conn8 pair; the baselines are never touched by the
# experiments.
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


@dataclass
class MultiFloodFillResult:
    img: np.ndarray        # (width, height, 3) uint8, blob recolored blue
    visited: np.ndarray    # (width, height) int32, 1 where reached
    depth: np.ndarray      # (width, height) int32, BFS level per pixel
    owner: np.ndarray      # (width, height) int16, processing block id, -1
                           # unreached; empty for bare runs
    levels: int
    filled: int
    peak_level: int        # largest single frontier
    peak_occupancy: int    # max queue entries alive across two adjacent levels
    threads_per_block: int
    blocks: int            # resolved launch size (None -> cooperative max)
    bare: bool
    connectivity: int      # 4 (Manhattan waves) or 8 (Chebyshev waves)
    radius: int            # 1, or 2 for the guarded radius-2 twin
    probe_layout: str      # "thread" (each thread loops its 8 dirs) or
                           # "warp" (4 entries x 8 dirs across 32 lanes)
    # Work / balance
    processed: int
    cas_attempts: int
    interior: int          # radius-2 only: guard passes (else 0)
    processed_per_block: np.ndarray = field(repr=False)  # (blocks,) int64
    balance_min_max_pct: float  # 100 * min / max of per-block processed
    balance_cv_pct: float       # 100 * std / mean — N-way imbalance in one number
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
    # Derived bandwidth model (zeroed for bare; see bandwidth.py)
    model_bytes: int
    model_gb_s: float


_warmed = set()
_coop_cache = {}


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _tiny_args():
    """A single red pixel at (1, 1) of an 8x8 white image: terminates under
    any block count (no discoveries), so it is safe for 1-block compile
    launches."""
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    d_img = cuda.to_device(tiny)
    d_visited = cuda.to_device(np.zeros((8, 8), dtype=np.int32))
    d_visited[1:2, 1:2].copy_to_device(np.ones((1, 1), dtype=np.int32))
    d_depth = cuda.to_device(np.full((8, 8), -1, dtype=np.int32))
    d_counters = cuda.to_device(np.zeros(NUM_COUNTERS, dtype=np.int64))
    d_queue = cuda.device_array(64, dtype=np.int32)
    d_queue[:1].copy_to_device(np.array([1 * 8 + 1], dtype=np.int32))
    d_q = cuda.to_device(np.array([1], dtype=np.int32))
    return d_img, d_visited, d_depth, d_counters, d_queue, d_q


def _warmup(bare, connectivity=4, radius=1, probe_layout="thread"):
    """JIT-compile (and NVRTC-link) each kernel once, off the clock. The
    instrumented warmup also exercises the tuple-indexed int64 atomic on
    block_stats so any numba regression fails here, not mid-benchmark."""
    key = (bare, connectivity, radius, probe_layout)
    if key in _warmed:
        return
    d_img, d_visited, d_depth, d_counters, d_queue, d_q = _tiny_args()
    kernel_fn = _KERNELS[key]
    if bare:
        kernel_fn[1, 32](d_img, d_visited, d_depth, d_queue, d_q, d_counters)
    else:
        d_owner = cuda.to_device(np.full((8, 8), -1, dtype=np.int16))
        d_stats = cuda.to_device(np.zeros((1, 2), dtype=np.int64))
        d_trace = cuda.device_array(4, dtype=np.int32)
        kernel_fn[1, 32](
            d_img, d_visited, d_depth, d_owner, d_queue, d_q, d_counters,
            d_stats, d_trace)
    cuda.synchronize()
    _warmed.add(key)


def _coop_max_blocks(kernel_fn, tpb):
    key = (id(kernel_fn), tpb)
    if key not in _coop_cache:
        overload = next(iter(kernel_fn.overloads.values()))
        _coop_cache[key] = overload.max_cooperative_grid_blocks(tpb)
    return _coop_cache[key]


def max_blocks(threads_per_block=256, bare=False, connectivity=4, radius=1,
               probe_layout="thread"):
    """The largest cooperative grid this GPU can host at threads_per_block
    (what blocks=None resolves to). Compiles the kernel on first call.
    Queried per kernel — no twin's capacity is ever assumed equal to
    another's (a register-count difference would change it)."""
    _warmup(bare, connectivity, radius, probe_layout)
    return _coop_max_blocks(
        _KERNELS[(bare, connectivity, radius, probe_layout)],
        threads_per_block)


def flood_fill(img_host, seed_x, seed_y, threads_per_block=256, blocks=None,
               bare=False, connectivity=4, radius=1, probe_layout="thread"):
    """Flood-fill the red blob containing (seed_x, seed_y) with N blocks.

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
        raise ValueError(f"seed pixel ({seed_x}, {seed_y}) is not red — nothing to fill")
    # 512 cap (vs the single-block package's 1024): the kernel's ~104
    # registers per thread mean a 1024-thread block would need more than
    # the SM's entire 64K register file and could never be resident.
    if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 512):
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 512], "
            f"got {threads_per_block}")
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

    _warmup(bare, connectivity, radius, probe_layout)
    device = cuda.get_current_device()
    sm_count = int(device.MULTIPROCESSOR_COUNT)

    kernel_fn = _KERNELS[(bare, connectivity, radius, probe_layout)]
    coop_max = _coop_max_blocks(kernel_fn, threads_per_block)
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
    d_img = cuda.device_array_like(img_host)
    d_visited = cuda.device_array_like(visited_host)
    d_depth = cuda.device_array_like(depth_host)
    d_counters = cuda.device_array_like(counters_host)
    d_queue = cuda.device_array(width * height, dtype=np.int32)
    if instrumented:
        owner_host = np.full((width, height), -1, dtype=np.int16)
        d_owner = cuda.device_array_like(owner_host)
        stats_host = np.zeros((launch_blocks, 2), dtype=np.int64)
        stats_host[:, BS_SMID] = -1
        d_stats = cuda.device_array_like(stats_host)
        d_trace = cuda.device_array(trace_capacity, dtype=np.int32)
    cuda.synchronize()
    t_h2d0 = time.perf_counter()

    d_img.copy_to_device(img_host)
    d_visited.copy_to_device(visited_host)
    d_depth.copy_to_device(depth_host)
    d_counters.copy_to_device(counters_host)
    d_queue[:1].copy_to_device(seed_lin)
    d_q_state = cuda.to_device(np.array([1], dtype=np.int32))
    if instrumented:
        d_owner.copy_to_device(owner_host)
        d_stats.copy_to_device(stats_host)
    cuda.synchronize()
    t_kernel0 = time.perf_counter()

    if instrumented:
        kernel_fn[launch_blocks, threads_per_block](
            d_img, d_visited, d_depth, d_owner, d_queue, d_q_state,
            d_counters, d_stats, d_trace)
    else:
        kernel_fn[launch_blocks, threads_per_block](
            d_img, d_visited, d_depth, d_queue, d_q_state, d_counters)
    cuda.synchronize()
    t_d2h0 = time.perf_counter()

    counters = d_counters.copy_to_host()
    if counters[OVERFLOW]:
        raise RuntimeError(
            "structural tripwire fired — this indicates a kernel bug: the "
            "queue cannot legitimately overflow")
    levels = int(counters[LEVELS])
    truncated = instrumented and levels > trace_capacity
    img_out = d_img.copy_to_host()
    visited_out = d_visited.copy_to_host()
    depth_out = d_depth.copy_to_host()
    if instrumented:
        owner_out = d_owner.copy_to_host()
        stats = d_stats.copy_to_host()
        sizes = d_trace[:min(levels, trace_capacity)].copy_to_host()
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
        / (sm_count * device.MAX_THREADS_PER_MULTIPROCESSOR),
        sm_utilization_pct=100.0 * distinct_sms / sm_count,
        discovery_redundancy=(cas_attempts / max(filled - 1, 1)
                              if instrumented else 0.0),
        neighbor_check_efficiency_pct=(100.0 * filled / eff_probes
                                       if instrumented and processed else 0.0),
        model_bytes=mbytes,
        model_gb_s=_model_gb_s(mbytes, kernel_ms),
    )
