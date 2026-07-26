"""
Host driver for the single-block shared-memory-ring flood fill.

Public API: flood_fill(img, seed_x, seed_y, threads_per_block=256,
                       variant="ring") -> FloodFillResult

variant="ring"  — v1: pure shared-memory ring; raises RuntimeError when a
                  scene's frontier exceeds the 8192-slot capacity.
variant="spill" — v2: two-tier frontier (shared ring + global spill tier
                  sized width*height, so it cannot overflow) with
                  warp-aggregated enqueue. Any blob that fits in memory
                  completes; ring-sized scenes just never touch the tier.

Handles validation, allocation, the single 1-block kernel launch, the
overflow tripwire, and a full timing decomposition (alloc / H2D / kernel /
D2H / total) using the repo's perf_counter + cuda.synchronize convention.
CUDA events would be the tool of choice if kernel times dropped under
~100 us; at the ms scale measured here, host-perceived wall time (which
includes launch overhead — part of the honest single-block story) is the
more meaningful number.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .kernels import (
    single_block_bfs_kernel, single_block_bfs_spill_kernel,
    RING_CAPACITY, NUM_COUNTERS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS,
    SPILLED, PEAK_SPILL_WINDOW,
)
from numba import cuda

# Cap on the recorded per-level frontier trace (int32 entries -> 8 MB max).
# Aggregate counters remain exact when a pathological scene exceeds it.
LEVEL_TRACE_CAPACITY = 2 ** 21


@dataclass
class FloodFillResult:
    img: np.ndarray        # (width, height, 3) uint8, blob recolored blue
    visited: np.ndarray    # (width, height) int32, 1 where reached
    depth: np.ndarray      # (width, height) int32, BFS level per pixel, -1 elsewhere
    levels: int            # BFS levels processed
    filled: int            # pixels reached
    peak_level: int        # largest single frontier
    peak_occupancy: int    # max queue occupancy: ring for v1, both tiers for v2
    ring_capacity: int     # = RING_CAPACITY, for reporting
    threads_per_block: int
    variant: str           # "ring" (v1) or "spill" (v2 two-tier)
    spilled: int           # total pixels routed to the global tier (0 for ring)
    peak_spill_window: int  # largest single-level spill count (0 for ring)
    # Per-level frontier sizes (len == levels unless truncated)
    level_sizes: np.ndarray = field(repr=False)
    level_trace_truncated: bool
    # Work-efficiency counters
    processed: int         # pixels dequeued (== filled: exactly-once processing)
    cas_attempts: int      # visited-CAS ops tried across all threads
    # Timing decomposition (ms)
    alloc_ms: float        # device allocation + host-side array prep
    h2d_ms: float          # host -> device copies
    kernel_ms: float       # the single kernel launch
    d2h_ms: float          # device -> host copies
    total_ms: float        # everything above, wall clock
    # Derived utilization metrics (percentages unless noted)
    thread_util_pct: float       # avg fraction of block threads with work per level
    warp_engagement_pct: float   # avg fraction of block warps with >=1 active thread
    lane_efficiency_pct: float   # how full the engaged warps were
    occupancy_pct: float         # theoretical: tpb / max threads per SM
    sm_utilization_pct: float    # constant by design: 1 block / SM count
    discovery_redundancy: float  # cas_attempts / (filled - 1); 1.0 = zero duplicated discovery
    neighbor_check_efficiency_pct: float  # filled / (4 * processed)


_warmed_up = {"ring": False, "spill": False}


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _warmup(variant):
    """JIT-compile the kernel on a tiny scene so timings never include compile."""
    if _warmed_up[variant]:
        return
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[4, 4] = (255, 0, 0)
    args = (
        cuda.to_device(tiny),
        cuda.to_device(np.zeros((8, 8), dtype=np.int32)),
        cuda.to_device(np.full((8, 8), -1, dtype=np.int32)),
        4, 4,
        cuda.to_device(np.zeros(NUM_COUNTERS, dtype=np.int64)),
        cuda.device_array(4, dtype=np.int32),
    )
    if variant == "ring":
        single_block_bfs_kernel[1, 32](*args)
    else:
        single_block_bfs_spill_kernel[1, 32](
            *args[:5], cuda.device_array(64, dtype=np.int32), *args[5:])
    cuda.synchronize()
    _warmed_up[variant] = True


def flood_fill(img_host, seed_x, seed_y, threads_per_block=256, variant="ring"):
    """Flood-fill the red blob containing (seed_x, seed_y) with one CUDA block.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for bad inputs. variant="ring" raises
    RuntimeError if the scene's peak frontier overflows the shared-memory
    ring; variant="spill" completes any scene (the global tier absorbs the
    excess) at the cost of a width*height int32 spill allocation.
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
    if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 1024):
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 1024], "
            f"got {threads_per_block}")
    if variant not in ("ring", "spill"):
        raise ValueError(f'variant must be "ring" or "spill", got {variant!r}')

    _warmup(variant)
    device = cuda.get_current_device()

    trace_capacity = min(width * height, LEVEL_TRACE_CAPACITY)

    t_total0 = time.perf_counter()

    visited_host = np.zeros((width, height), dtype=np.int32)
    depth_host = np.full((width, height), -1, dtype=np.int32)
    counters_host = np.zeros(NUM_COUNTERS, dtype=np.int64)
    d_img = cuda.device_array_like(img_host)
    d_visited = cuda.device_array_like(visited_host)
    d_depth = cuda.device_array_like(depth_host)
    d_counters = cuda.device_array_like(counters_host)
    d_level_sizes = cuda.device_array(trace_capacity, dtype=np.int32)
    if variant == "spill":
        # width*height entries make spill overflow structurally impossible
        # (every pixel is enqueued at most once). The allocation cost is part
        # of the honest v2 story and lands in alloc_ms.
        d_spill = cuda.device_array(width * height, dtype=np.int32)
    cuda.synchronize()
    t_h2d0 = time.perf_counter()

    d_img.copy_to_device(img_host)
    d_visited.copy_to_device(visited_host)
    d_depth.copy_to_device(depth_host)
    d_counters.copy_to_device(counters_host)
    cuda.synchronize()
    t_kernel0 = time.perf_counter()

    if variant == "ring":
        single_block_bfs_kernel[1, threads_per_block](
            d_img, d_visited, d_depth, seed_x, seed_y, d_counters,
            d_level_sizes)
    else:
        single_block_bfs_spill_kernel[1, threads_per_block](
            d_img, d_visited, d_depth, seed_x, seed_y, d_spill, d_counters,
            d_level_sizes)
    cuda.synchronize()
    t_d2h0 = time.perf_counter()

    counters = d_counters.copy_to_host()
    if variant == "ring" and counters[OVERFLOW]:
        raise RuntimeError(
            f"shared-memory ring overflow: this scene's frontier needs more than "
            f"the {RING_CAPACITY} slots available "
            f"(largest completed-level occupancy: {int(counters[PEAK_OCC])}). "
            f"Rule of thumb: a center-seeded solid square of side W needs ~4W "
            f"slots, corner-seeded ~2W. Rerun with variant=\"spill\" (the v2 "
            f"two-tier kernel), or use the multi-block/persistent "
            f"implementations.")
    levels = int(counters[LEVELS])
    truncated = levels > trace_capacity
    img_out = d_img.copy_to_host()
    visited_out = d_visited.copy_to_host()
    depth_out = d_depth.copy_to_host()
    level_sizes = d_level_sizes[:min(levels, trace_capacity)].copy_to_host()
    t_end = time.perf_counter()

    filled = int(counters[FILLED])
    processed = int(counters[PROCESSED])
    cas_attempts = int(counters[CAS_ATTEMPTS])
    warps = threads_per_block // 32
    active_thread_sum = int(counters[ACTIVE_THREAD_SUM])
    active_warp_sum = int(counters[ACTIVE_WARP_SUM])

    return FloodFillResult(
        img=img_out,
        visited=visited_out,
        depth=depth_out,
        levels=levels,
        filled=filled,
        peak_level=int(counters[PEAK_LEVEL]),
        peak_occupancy=int(counters[PEAK_OCC]),
        ring_capacity=RING_CAPACITY,
        threads_per_block=threads_per_block,
        variant=variant,
        spilled=int(counters[SPILLED]),
        peak_spill_window=int(counters[PEAK_SPILL_WINDOW]),
        level_sizes=level_sizes,
        level_trace_truncated=truncated,
        processed=processed,
        cas_attempts=cas_attempts,
        alloc_ms=(t_h2d0 - t_total0) * 1000,
        h2d_ms=(t_kernel0 - t_h2d0) * 1000,
        kernel_ms=(t_d2h0 - t_kernel0) * 1000,
        d2h_ms=(t_end - t_d2h0) * 1000,
        total_ms=(t_end - t_total0) * 1000,
        thread_util_pct=100.0 * active_thread_sum / (levels * threads_per_block),
        warp_engagement_pct=100.0 * active_warp_sum / (levels * warps),
        lane_efficiency_pct=100.0 * active_thread_sum / (32 * active_warp_sum),
        occupancy_pct=100.0 * threads_per_block / device.MAX_THREADS_PER_MULTIPROCESSOR,
        sm_utilization_pct=100.0 / device.MULTIPROCESSOR_COUNT,
        discovery_redundancy=cas_attempts / max(filled - 1, 1),
        neighbor_check_efficiency_pct=100.0 * filled / (4 * processed),
    )
