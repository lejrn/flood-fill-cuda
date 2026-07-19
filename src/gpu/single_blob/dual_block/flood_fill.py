"""Host driver for the dual-block flood fill.

Public API:
    flood_fill(img, seed_x, seed_y, threads_per_block=256,
               kernel="split", bare=False, placement=None)
        -> DualFloodFillResult

kernel="split"    domain decomposition (per-block shared ring + spill,
                  cross-seam inboxes)
kernel="global"   one shared global queue, grid-strided
kernel="dirsplit" partition by discovery direction (double-ended buffer)
kernel="pinned"   placement experiment: requires placement="same_sm" (48x768
                  occupancy-forced launch, the two blocks sharing one SM do
                  the work) or placement="spread" (2x768, natural placement);
                  threads_per_block must be 768; minimal instrumentation.

bare=True selects the uninstrumented twin (split/global/dirsplit only) for
measuring instrumentation overhead; such results carry timing, filled and
levels but zeroed work/utilization metrics.

Cooperative sizing follows persistent/: the compiled kernel is queried for
max_cooperative_grid_blocks (never hardcoded — oversizing deadlocks
grid.sync). Every result records the %smid each worker block observed
itself running on: placement is reported, not assumed.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from kernels import (
    dual_block_global_kernel, dual_block_split_kernel,
    dual_block_dirsplit_kernel, dual_block_pinned_kernel,
    dual_block_global_bare_kernel, dual_block_split_bare_kernel,
    dual_block_dirsplit_bare_kernel,
    RING_CAPACITY, NUM_COUNTERS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS,
    SPILLED, PEAK_SPILL_WINDOW,
    PROCESSED_B0, PROCESSED_B1, SPILLED_B0, SPILLED_B1,
    INBOX_TO_B0, INBOX_TO_B1, SMID_B0, SMID_B1,
)
from numba import cuda

# Cap on the recorded per-level trace (2 rows of int32 -> 16 MB max).
LEVEL_TRACE_CAPACITY = 2 ** 21

PINNED_TPB = 768  # floor(1536 / 768) = 2 blocks/SM: the occupancy-forcing size


@dataclass
class DualFloodFillResult:
    img: np.ndarray        # (width, height, 3) uint8, blob recolored blue
    visited: np.ndarray    # (width, height) int32, 1 where reached
    depth: np.ndarray      # (width, height) int32, BFS level per pixel
    levels: int
    filled: int
    peak_level: int        # largest single global frontier
    peak_occupancy: int    # max queue entries alive across two adjacent levels
    ring_capacity: int
    threads_per_block: int
    kernel: str            # "split" | "global" | "dirsplit" | "pinned"
    bare: bool
    placement: str | None  # pinned only: "same_sm" | "spread"
    blocks: int            # worker blocks (always 2; pinned same_sm launches 48)
    # Work / balance
    processed: int
    cas_attempts: int
    processed_b0: int
    processed_b1: int
    balance_pct: float     # 100 * min / max of per-block processed
    # Split-kernel tiers
    spilled: int
    spilled_b0: int
    spilled_b1: int
    peak_spill_window: int
    inbox_to_b0: int
    inbox_to_b1: int
    inbox_pct: float       # cross-seam handoffs as % of filled
    # Placement (observed, not assumed; -1 when not recorded)
    sm_id_b0: int
    sm_id_b1: int
    same_sm: bool
    # Per-level traces (empty for bare/pinned)
    level_sizes: np.ndarray = field(repr=False)          # global (rows summed)
    level_sizes_per_block: np.ndarray = field(repr=False)  # (2, levels)
    level_trace_truncated: bool
    # Timing decomposition (ms)
    alloc_ms: float
    h2d_ms: float
    kernel_ms: float
    d2h_ms: float
    total_ms: float
    # Derived utilization metrics (zeroed for bare/pinned)
    thread_util_pct: float
    warp_engagement_pct: float
    lane_efficiency_pct: float
    occupancy_pct: float
    sm_utilization_pct: float
    discovery_redundancy: float
    neighbor_check_efficiency_pct: float


_KERNELS = {
    ("split", False): dual_block_split_kernel,
    ("split", True): dual_block_split_bare_kernel,
    ("global", False): dual_block_global_kernel,
    ("global", True): dual_block_global_bare_kernel,
    ("dirsplit", False): dual_block_dirsplit_kernel,
    ("dirsplit", True): dual_block_dirsplit_bare_kernel,
}

_warmed = set()
_coop_cache = {}


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _tiny_args():
    """A single red pixel at (1, 1) of an 8x8 white image: terminates under
    any kernel and any block count (no discoveries, no seam crossing), so
    it is safe even for the 1-block compile launches."""
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    d_img = cuda.to_device(tiny)
    d_visited = cuda.to_device(np.zeros((8, 8), dtype=np.int32))
    d_visited[1:2, 1:2].copy_to_device(np.ones((1, 1), dtype=np.int32))
    d_depth = cuda.to_device(np.full((8, 8), -1, dtype=np.int32))
    d_counters = cuda.to_device(np.zeros(NUM_COUNTERS, dtype=np.int64))
    return d_img, d_visited, d_depth, d_counters


def _warmup(kernel, bare, placement=None):
    """JIT-compile (and NVRTC-link) each kernel once, off the clock."""
    key = (kernel, bare)
    if key in _warmed:
        return
    d_img, d_visited, d_depth, d_counters = _tiny_args()
    d_trace = cuda.device_array((2, 4), dtype=np.int32)
    seed_lin = np.array([1 * 8 + 1], dtype=np.int32)
    if kernel == "split":
        d_spill0 = cuda.device_array(32, dtype=np.int32)
        d_spill1 = cuda.device_array(32, dtype=np.int32)
        d_inbox0 = cuda.device_array(8, dtype=np.int32)
        d_inbox1 = cuda.device_array(8, dtype=np.int32)
        d_g = cuda.to_device(np.zeros(6, dtype=np.int32))
        args = (d_img, d_visited, d_depth, 1, 1, d_spill0, d_spill1,
                d_inbox0, d_inbox1, d_g, d_counters)
        fn = _KERNELS[(kernel, bare)]
        fn[1, 32](*args) if bare else fn[1, 32](*args, d_trace)
    elif kernel in ("global", "dirsplit"):
        d_queue = cuda.device_array(64, dtype=np.int32)
        d_queue[:1].copy_to_device(seed_lin)
        init = [1] if kernel == "global" else [1, 0]
        d_q = cuda.to_device(np.array(init, dtype=np.int32))
        args = (d_img, d_visited, d_depth, d_queue, d_q, d_counters)
        fn = _KERNELS[(kernel, bare)]
        fn[1, 32](*args) if bare else fn[1, 32](*args, d_trace)
    else:  # pinned — compile via a 2-block spread run (1 block would spin)
        d_queue = cuda.device_array(64, dtype=np.int32)
        d_queue[:1].copy_to_device(seed_lin)
        d_q = cuda.to_device(np.array([1], dtype=np.int32))
        d_bar = cuda.to_device(np.zeros(2, dtype=np.int32))
        d_pin = cuda.to_device(np.array([1, -1, 0], dtype=np.int32))
        dual_block_pinned_kernel[2, PINNED_TPB](
            d_img, d_visited, d_depth, d_queue, d_q, d_bar, d_pin, d_counters)
    cuda.synchronize()
    _warmed.add(key)


def _coop_max_blocks(kernel_fn, tpb):
    key = (id(kernel_fn), tpb)
    if key not in _coop_cache:
        overload = next(iter(kernel_fn.overloads.values()))
        _coop_cache[key] = overload.max_cooperative_grid_blocks(tpb)
    return _coop_cache[key]


def flood_fill(img_host, seed_x, seed_y, threads_per_block=256,
               kernel="split", bare=False, placement=None):
    """Flood-fill the red blob containing (seed_x, seed_y) with two blocks.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for bad inputs and RuntimeError if the GPU
    cannot host the cooperative launch (or if a structural tripwire fires,
    which would indicate a kernel bug).
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
    if kernel not in ("split", "global", "dirsplit", "pinned"):
        raise ValueError(
            f'kernel must be "split", "global", "dirsplit" or "pinned", '
            f'got {kernel!r}')
    if kernel == "pinned":
        if placement not in ("same_sm", "spread"):
            raise ValueError(
                'kernel="pinned" requires placement="same_sm" or "spread"')
        if threads_per_block != PINNED_TPB:
            raise ValueError(
                f"the pinned experiment requires threads_per_block == "
                f"{PINNED_TPB}: floor(1536/{PINNED_TPB}) = 2 blocks per SM is "
                f"what forces every SM to host exactly two blocks")
        if bare:
            raise ValueError("the pinned kernel is minimal by design; "
                             "bare=True does not apply")
    else:
        if placement is not None:
            raise ValueError("placement only applies to kernel=\"pinned\"")
        # 512 cap (vs the single-block package's 1024): the dual kernels'
        # coordination state costs ~104-159 registers per thread, so a
        # 1024-thread block would need more than the SM's entire 64K
        # register file and could never be resident.
        if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 512):
            raise ValueError(
                f"threads_per_block must be a multiple of 32 in [32, 512], "
                f"got {threads_per_block}")

    _warmup(kernel, bare, placement)
    device = cuda.get_current_device()
    sm_count = int(device.MULTIPROCESSOR_COUNT)

    if kernel == "pinned":
        kernel_fn = dual_block_pinned_kernel
        launch_blocks = 2 * sm_count if placement == "same_sm" else 2
    else:
        kernel_fn = _KERNELS[(kernel, bare)]
        launch_blocks = 2
    max_blocks = _coop_max_blocks(kernel_fn, threads_per_block)
    if max_blocks < launch_blocks:
        raise RuntimeError(
            f"this GPU supports at most {max_blocks} cooperative blocks of "
            f"{threads_per_block} threads for kernel {kernel!r}; "
            f"{launch_blocks} required")

    seed_lin = np.array([seed_x * height + seed_y], dtype=np.int32)
    half = width // 2
    trace_capacity = min(width * height, LEVEL_TRACE_CAPACITY)
    instrumented = not bare and kernel != "pinned"

    t_total0 = time.perf_counter()

    visited_host = np.zeros((width, height), dtype=np.int32)
    visited_host[seed_x, seed_y] = 1
    depth_host = np.full((width, height), -1, dtype=np.int32)
    counters_host = np.zeros(NUM_COUNTERS, dtype=np.int64)
    d_img = cuda.device_array_like(img_host)
    d_visited = cuda.device_array_like(visited_host)
    d_depth = cuda.device_array_like(depth_host)
    d_counters = cuda.device_array_like(counters_host)
    if kernel == "split":
        d_spill0 = cuda.device_array(max(half * height, 1), dtype=np.int32)
        d_spill1 = cuda.device_array(max((width - half) * height, 1),
                                     dtype=np.int32)
        d_inbox0 = cuda.device_array(max(height, 1), dtype=np.int32)
        d_inbox1 = cuda.device_array(max(height, 1), dtype=np.int32)
    else:
        d_queue = cuda.device_array(width * height, dtype=np.int32)
    if instrumented:
        d_trace = cuda.device_array((2, trace_capacity), dtype=np.int32)
    cuda.synchronize()
    t_h2d0 = time.perf_counter()

    d_img.copy_to_device(img_host)
    d_visited.copy_to_device(visited_host)
    d_depth.copy_to_device(depth_host)
    d_counters.copy_to_device(counters_host)
    if kernel == "split":
        d_g_state = cuda.to_device(np.zeros(6, dtype=np.int32))
    elif kernel == "global":
        d_queue[:1].copy_to_device(seed_lin)
        d_q_state = cuda.to_device(np.array([1], dtype=np.int32))
    elif kernel == "dirsplit":
        d_queue[:1].copy_to_device(seed_lin)
        d_q_state = cuda.to_device(np.array([1, 0], dtype=np.int32))
    else:  # pinned
        d_queue[:1].copy_to_device(seed_lin)
        d_q_state = cuda.to_device(np.array([1], dtype=np.int32))
        d_barrier = cuda.to_device(np.zeros(2, dtype=np.int32))
        mode = 0 if placement == "same_sm" else 1
        d_pin = cuda.to_device(np.array([mode, -1, 0], dtype=np.int32))
    cuda.synchronize()
    t_kernel0 = time.perf_counter()

    if kernel == "split":
        args = (d_img, d_visited, d_depth, seed_x, seed_y, d_spill0, d_spill1,
                d_inbox0, d_inbox1, d_g_state, d_counters)
        if instrumented:
            kernel_fn[2, threads_per_block](*args, d_trace)
        else:
            kernel_fn[2, threads_per_block](*args)
    elif kernel in ("global", "dirsplit"):
        args = (d_img, d_visited, d_depth, d_queue, d_q_state, d_counters)
        if instrumented:
            kernel_fn[2, threads_per_block](*args, d_trace)
        else:
            kernel_fn[2, threads_per_block](*args)
    else:  # pinned
        kernel_fn[launch_blocks, threads_per_block](
            d_img, d_visited, d_depth, d_queue, d_q_state, d_barrier, d_pin,
            d_counters)
    cuda.synchronize()
    t_d2h0 = time.perf_counter()

    counters = d_counters.copy_to_host()
    if counters[OVERFLOW]:
        raise RuntimeError(
            "structural tripwire fired — this indicates a kernel bug: no "
            "queue in this package can legitimately overflow")
    levels = int(counters[LEVELS])
    truncated = instrumented and levels > trace_capacity
    img_out = d_img.copy_to_host()
    visited_out = d_visited.copy_to_host()
    depth_out = d_depth.copy_to_host()
    if instrumented:
        n = min(levels, trace_capacity)
        row0 = d_trace[0, :n].copy_to_host()
        row1 = d_trace[1, :n].copy_to_host()
        per_block = np.stack([row0, row1])
        sizes = row0 + row1
    else:
        per_block = np.zeros((2, 0), dtype=np.int32)
        sizes = np.zeros(0, dtype=np.int32)
    t_end = time.perf_counter()

    filled = int(counters[FILLED])
    processed = int(counters[PROCESSED])
    cas_attempts = int(counters[CAS_ATTEMPTS])
    p0 = int(counters[PROCESSED_B0])
    p1 = int(counters[PROCESSED_B1])
    balance = 100.0 * min(p0, p1) / max(p0, p1) if max(p0, p1) > 0 else 0.0
    inb0 = int(counters[INBOX_TO_B0])
    inb1 = int(counters[INBOX_TO_B1])
    sm0 = int(counters[SMID_B0]) if (instrumented or kernel == "pinned") else -1
    sm1 = int(counters[SMID_B1]) if (instrumented or kernel == "pinned") else -1
    warps = 2 * threads_per_block // 32
    ats = int(counters[ACTIVE_THREAD_SUM])
    aws = int(counters[ACTIVE_WARP_SUM])

    return DualFloodFillResult(
        img=img_out,
        visited=visited_out,
        depth=depth_out,
        levels=levels,
        filled=filled,
        peak_level=int(counters[PEAK_LEVEL]),
        peak_occupancy=int(counters[PEAK_OCC]),
        ring_capacity=RING_CAPACITY,
        threads_per_block=threads_per_block,
        kernel=kernel,
        bare=bare,
        placement=placement,
        blocks=2,
        processed=processed,
        cas_attempts=cas_attempts,
        processed_b0=p0,
        processed_b1=p1,
        balance_pct=balance,
        spilled=int(counters[SPILLED]),
        spilled_b0=int(counters[SPILLED_B0]),
        spilled_b1=int(counters[SPILLED_B1]),
        peak_spill_window=int(counters[PEAK_SPILL_WINDOW]),
        inbox_to_b0=inb0,
        inbox_to_b1=inb1,
        inbox_pct=100.0 * (inb0 + inb1) / filled if filled else 0.0,
        sm_id_b0=sm0,
        sm_id_b1=sm1,
        same_sm=(sm0 >= 0 and sm0 == sm1),
        level_sizes=sizes,
        level_sizes_per_block=per_block,
        level_trace_truncated=truncated,
        alloc_ms=(t_h2d0 - t_total0) * 1000,
        h2d_ms=(t_kernel0 - t_h2d0) * 1000,
        kernel_ms=(t_d2h0 - t_kernel0) * 1000,
        d2h_ms=(t_end - t_d2h0) * 1000,
        total_ms=(t_end - t_total0) * 1000,
        thread_util_pct=(100.0 * ats / (levels * 2 * threads_per_block)
                         if instrumented and levels else 0.0),
        warp_engagement_pct=(100.0 * aws / (levels * warps)
                             if instrumented and levels else 0.0),
        lane_efficiency_pct=(100.0 * ats / (32 * aws)
                             if instrumented and aws else 0.0),
        occupancy_pct=100.0 * threads_per_block
        / device.MAX_THREADS_PER_MULTIPROCESSOR,
        sm_utilization_pct=100.0 * (1 if (kernel == "pinned"
                                          and placement == "same_sm")
                                    else 2) / sm_count,
        discovery_redundancy=(cas_attempts / max(filled - 1, 1)
                              if instrumented else 0.0),
        neighbor_check_efficiency_pct=(100.0 * filled / (4 * processed)
                                       if instrumented and processed else 0.0),
    )
