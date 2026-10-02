"""
Host driver for the Triton twin of the single-block flood fill.

Public API: flood_fill(img, seed_x, seed_y, threads_per_block=256,
                       variant="ring", enqueue="lane") -> FloodFillResult

Same contract as chapters/ch01_gpu_1blob_1block/flood_fill.py:

variant="ring"  - v1: the 8192-slot ring; raises RuntimeError when a scene's
                  frontier exceeds the ring capacity.
variant="spill" - v2: two-tier frontier (ring + global spill tier sized
                  width*height, so it cannot overflow).

enqueue (Triton only, v2 only) picks the spill kernel's enqueue form:
"lane" (default) issues one atomic per winning lane, which ptxas
warp-aggregates into the machine code of Numba's _warp_enqueue_two_tier;
"program" is the first translation, aggregated over the whole program with
tl.cumsum/tl.sum and extra CTA barriers, kept so its cost stays measurable.
Both give identical results and counters. v1 has one form only (one ticket
atomic per winning lane, as in Numba), so variant="ring" accepts only
enqueue="lane".

Same validation in the same order (plus the Triton-only rules, checked
last: threads_per_block must be a power of 2, because num_warps =
threads_per_block // 32 and tl.arange lengths must be powers of 2; then the
enqueue value), the same accepted seed types (NumPy integers are converted
to int before the launch), the same one-program launch, the same overflow
tripwire, and the same timing decomposition (alloc / H2D / kernel / D2H /
total with time.perf_counter and a device synchronize closing each phase).
CuPy arrays
replace Numba device arrays. The ring and its scalars, shared memory in
Numba, are a per-call global scratch allocated in the alloc phase.

The result is the Numba driver's FloodFillResult itself, so every field
name and meaning is shared.
"""

import operator
import time

import cupy as cp
import numpy as np

from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.flood_fill import (
    FloodFillResult, LEVEL_TRACE_CAPACITY,
)
from flood_fill_cuda.triton_twins.runtime import device_info, sync, t

from .kernels import (
    single_block_bfs_kernel, single_block_bfs_spill_kernel,
    ENQ_MODES, RING_CAPACITY, NUM_COUNTERS, STATE_SLOTS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS,
    SPILLED, PEAK_SPILL_WINDOW,
)

__all__ = ["flood_fill", "FloodFillResult", "LEVEL_TRACE_CAPACITY"]

# Compiles are per (kernel, BLOCK, num_warps, ENQ); the scalar args are
# do_not_specialize, so one warm-up per (variant, tpb, enqueue) covers every
# scene. Maps that triple -> the CompiledKernel the warm-up launch returned.
_warmed_up = {}


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _launch(variant, enqueue, threads_per_block, d_img, d_visited, d_depth,
            seed_x, seed_y, d_spill, d_counters, d_level_sizes, d_ring,
            d_state):
    """The single one-program launch (grid (1,), BLOCK lanes)."""
    width, height = d_img.shape[0], d_img.shape[1]
    trace_cap = d_level_sizes.shape[0]
    opts = dict(BLOCK=threads_per_block, num_warps=threads_per_block // 32)
    if variant == "ring":
        return single_block_bfs_kernel[(1,)](
            t(d_img), t(d_visited), t(d_depth), seed_x, seed_y,
            t(d_counters), t(d_level_sizes), t(d_ring), t(d_state),
            width, height, trace_cap, **opts)
    return single_block_bfs_spill_kernel[(1,)](
        t(d_img), t(d_visited), t(d_depth), seed_x, seed_y, t(d_spill),
        t(d_counters), t(d_level_sizes), t(d_ring), t(d_state),
        width, height, trace_cap, ENQ=enqueue, **opts)


def _check_enqueue(variant, enqueue):
    """The Triton-only enqueue rules, shared by flood_fill and
    compiled_kernel (so neither caches a ring kernel under "program")."""
    if enqueue not in ENQ_MODES:
        raise ValueError(
            f'enqueue must be "lane" or "program", got {enqueue!r}')
    if variant == "ring" and enqueue != "lane":
        raise ValueError(
            'enqueue="program" exists for variant="spill" only: the v1 ring '
            'kernel issues one ticket atomic per winning lane, as Numba does')


def _warmup(variant, threads_per_block, enqueue="lane"):
    """Compile the kernel on a tiny scene so timings never include compile.

    No enqueue check here: the public entry points check first, and the
    tests reach the kernel's own static_assert through this function."""
    key = (variant, threads_per_block, enqueue)
    if key in _warmed_up:
        return
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[4, 4] = (255, 0, 0)
    compiled = _launch(variant, enqueue, threads_per_block,
                       cp.asarray(tiny),
                       cp.zeros((8, 8), dtype=cp.int32),
                       cp.full((8, 8), -1, dtype=cp.int32),
                       4, 4,
                       cp.empty(64, dtype=cp.int32),
                       cp.zeros(NUM_COUNTERS, dtype=cp.int64),
                       cp.empty(4, dtype=cp.int32),
                       cp.empty(RING_CAPACITY, dtype=cp.int32),
                       cp.empty(STATE_SLOTS, dtype=cp.int32))
    sync()
    _warmed_up[key] = compiled


def compiled_kernel(variant, threads_per_block, enqueue="lane"):
    """The CompiledKernel for (variant, tpb, enqueue), for
    runtime.kernel_resources() and the SASS check. Raises ValueError for
    the enqueue values flood_fill refuses."""
    _check_enqueue(variant, enqueue)
    _warmup(variant, threads_per_block, enqueue)
    return _warmed_up[(variant, threads_per_block, enqueue)]


def flood_fill(img_host, seed_x, seed_y, threads_per_block=256, variant="ring",
               enqueue="lane"):
    """Flood-fill the red blob containing (seed_x, seed_y) with one program.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for bad inputs. variant="ring" raises
    RuntimeError if the scene's peak frontier overflows the 8192-slot ring;
    variant="spill" completes any scene (the global tier absorbs the
    excess) at the cost of a width*height int32 spill allocation.
    enqueue="program" runs v2 with the first translation's
    program-aggregated enqueue (see the module docstring).
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
    if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 1024):
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 1024], "
            f"got {threads_per_block}")
    if variant not in ("ring", "spill"):
        raise ValueError(f'variant must be "ring" or "spill", got {variant!r}')
    # The Triton-only rule comes last, so every check shared with Numba
    # fires in Numba's order.
    if threads_per_block & (threads_per_block - 1):
        raise ValueError(
            f"threads_per_block must be a power of 2 for the Triton twin "
            f"(num_warps = threads_per_block // 32 and tl.arange lengths must "
            f"be powers of 2), got {threads_per_block}")
    _check_enqueue(variant, enqueue)
    # Numba types NumPy integer seeds (np.int32, np.int64, ...) as kernel
    # args; Triton's launcher cannot specialize NumPy scalars. The seeds
    # already indexed img_host above, so they are integers.
    seed_x, seed_y = operator.index(seed_x), operator.index(seed_y)

    _warmup(variant, threads_per_block, enqueue)
    device = device_info()

    trace_capacity = min(width * height, LEVEL_TRACE_CAPACITY)

    t_total0 = time.perf_counter()

    visited_host = np.zeros((width, height), dtype=np.int32)
    depth_host = np.full((width, height), -1, dtype=np.int32)
    counters_host = np.zeros(NUM_COUNTERS, dtype=np.int64)
    d_img = cp.empty(img_host.shape, dtype=img_host.dtype)
    d_visited = cp.empty(visited_host.shape, dtype=visited_host.dtype)
    d_depth = cp.empty(depth_host.shape, dtype=depth_host.dtype)
    d_counters = cp.empty(counters_host.shape, dtype=counters_host.dtype)
    d_level_sizes = cp.empty(trace_capacity, dtype=np.int32)
    # The twin of the kernel's shared memory: ring + its scalars.
    d_ring = cp.empty(RING_CAPACITY, dtype=np.int32)
    d_state = cp.empty(STATE_SLOTS, dtype=np.int32)
    d_spill = None
    if variant == "spill":
        # width*height entries make spill overflow structurally impossible
        # (every pixel is enqueued at most once). The allocation cost is part
        # of the honest v2 story and lands in alloc_ms.
        d_spill = cp.empty(width * height, dtype=np.int32)
    sync()
    t_h2d0 = time.perf_counter()

    d_img.set(np.ascontiguousarray(img_host))
    d_visited.set(visited_host)
    d_depth.set(depth_host)
    d_counters.set(counters_host)
    sync()
    t_kernel0 = time.perf_counter()

    _launch(variant, enqueue, threads_per_block, d_img, d_visited, d_depth,
            seed_x, seed_y, d_spill, d_counters, d_level_sizes, d_ring,
            d_state)
    sync()
    t_d2h0 = time.perf_counter()

    counters = d_counters.get()
    if variant == "ring" and counters[OVERFLOW]:
        raise RuntimeError(
            f"ring overflow: this scene's frontier needs more than "
            f"the {RING_CAPACITY} slots available "
            f"(largest completed-level occupancy: {int(counters[PEAK_OCC])}). "
            f"Rule of thumb: a center-seeded solid square of side W needs ~4W "
            f"slots, corner-seeded ~2W. Rerun with variant=\"spill\" (the v2 "
            f"two-tier kernel), or use the multi-block/persistent "
            f"implementations.")
    levels = int(counters[LEVELS])
    truncated = levels > trace_capacity
    img_out = d_img.get()
    visited_out = d_visited.get()
    depth_out = d_depth.get()
    level_sizes = d_level_sizes[:min(levels, trace_capacity)].get()
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
        occupancy_pct=100.0 * threads_per_block / device.max_threads_per_sm,
        sm_utilization_pct=100.0 / device.sm_count,
        discovery_redundancy=cas_attempts / max(filled - 1, 1),
        neighbor_check_efficiency_pct=100.0 * filled / (4 * processed),
    )
