"""Host driver for the Triton twin of the dual-block flood fill.

Public API (same as the Numba chapter):
    flood_fill(img, seed_x, seed_y, threads_per_block=256,
               kernel="split", bare=False, placement=None)
        -> DualFloodFillResult

kernel="split"    domain decomposition (per-program ring + spill, cross-seam
                  inboxes)
kernel="global"   one shared global queue, grid-strided
kernel="dirsplit" partition by discovery direction (double-ended buffer)
kernel="pinned"   placement experiment: requires placement="same_sm" (an
                  occupancy-forced launch, the two programs sharing one SM do
                  the work) or placement="spread" (2 programs, natural
                  placement); threads_per_block must be PINNED_TPB (512 here:
                  Numba's 768 is not a power of 2); minimal instrumentation.

bare=True selects the uninstrumented specialization (split/global/dirsplit
only); such results carry timing, filled and levels but zeroed
work/utilization metrics.

Differences from the Numba driver, all forced by Triton:
- threads_per_block must also be a power of 2 (num_warps = tpb // 32 and
  tl.arange lengths are powers of 2): 96, 160, ... raise ValueError.
- Each (kernel, bare, tpb) is a separate compile (TPB is a constexpr), so
  the warm-up is keyed on tpb too.
- grid.sync has no buffer; its Triton twin needs a zeroed int32 counter.
  The split kernel's shared ring lives in a (2, 8192) global scratch array.
  Both are allocated with the other device arrays and zeroed / left
  uninitialized exactly where Numba's shared memory would be.
"""

import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from flood_fill_cuda.triton_twins.runtime import (
    device_info, kernel_resources, max_coresident_programs, programs_per_sm,
    sync, t,
)

from .kernels import (
    dual_block_global_kernel, dual_block_split_kernel,
    dual_block_dirsplit_kernel, dual_block_pinned_kernel,
    RING_CAPACITY, NUM_COUNTERS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS,
    SPILLED, PEAK_SPILL_WINDOW,
    PROCESSED_B0, PROCESSED_B1, SPILLED_B0, SPILLED_B1,
    INBOX_TO_B0, INBOX_TO_B1, SMID_B0, SMID_B1,
)

# Cap on the recorded per-level trace (2 rows of int32 -> 16 MB max).
LEVEL_TRACE_CAPACITY = 2 ** 21

# Numba uses 768 (floor(1536/768) = 2 blocks/SM). 768 lanes is not a power
# of 2, so the twin pins 512-lane programs and caps registers so that two
# fit per SM (65536 / (2 * 512) = 64): the nearest faithful experiment.
PINNED_TPB = 512
PINNED_MAXNREG = 64
PINNED_WORKERS = 2

# Numba compiles split with max_registers=120 (instrumented and bare twin).
SPLIT_MAXNREG = 120


@dataclass
class DualFloodFillResult:
    img: np.ndarray        # (width, height, 3) uint8, blob recolored blue
    visited: np.ndarray    # (width, height) int32, 1 where reached
    depth: np.ndarray      # (width, height) int32, BFS level per pixel
    owner: np.ndarray      # (width, height) int8, processing program id, -1
                           # unreached; empty for bare/pinned runs
    levels: int
    filled: int
    peak_level: int        # largest single global frontier
    peak_occupancy: int    # max queue entries alive across two adjacent levels
    ring_capacity: int
    threads_per_block: int
    kernel: str            # "split" | "global" | "dirsplit" | "pinned"
    bare: bool
    placement: str | None  # pinned only: "same_sm" | "spread"
    blocks: int            # worker programs (always 2)
    # Work / balance
    processed: int
    cas_attempts: int
    processed_b0: int
    processed_b1: int
    balance_pct: float     # 100 * min / max of per-program processed
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
    "split": dual_block_split_kernel,
    "global": dual_block_global_kernel,
    "dirsplit": dual_block_dirsplit_kernel,
    "pinned": dual_block_pinned_kernel,
}

_warmed = {}       # (kernel, bare, tpb) -> CompiledKernel of the warm-up launch
_coop_cache = {}   # (kernel, bare, tpb) -> max co-resident programs


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _launch_options(kernel, tpb):
    """Compile/launch options shared by the warm-up and the timed launch, so
    the timed launch always hits the warm-up's compiled specialization."""
    opts = {"num_warps": tpb // 32, "num_stages": 1,
            "launch_cooperative_grid": True}
    if kernel == "split":
        opts["maxnreg"] = SPLIT_MAXNREG
    elif kernel == "pinned":
        opts["maxnreg"] = PINNED_MAXNREG
    return opts


def _tiny_args():
    """A single red pixel at (1, 1) of an 8x8 white image: terminates under
    any kernel and any program count (no discoveries, no seam crossing)."""
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    d_img = cp.asarray(tiny)
    d_visited = cp.zeros((8, 8), dtype=cp.int32)
    d_visited[1, 1] = 1
    d_depth = cp.full((8, 8), -1, dtype=cp.int32)
    d_counters = cp.zeros(int(NUM_COUNTERS), dtype=cp.int64)
    return d_img, d_visited, d_depth, d_counters


def _warmup(kernel, bare, tpb):
    """Compile each (kernel, bare, tpb) specialization once, off the clock.

    The warm-up launches 2 programs with the timed launch's exact options
    (cooperative, num_warps, num_stages, maxnreg) and the same argument
    kinds, so no compile can land inside kernel_ms. Returns the
    CompiledKernel (its registers size the cooperative grid)."""
    key = (kernel, bare, tpb)
    if key in _warmed:
        return _warmed[key]
    d_img, d_visited, d_depth, d_counters = _tiny_args()
    d_trace = cp.empty((2, 4), dtype=cp.int32)
    d_owner = cp.full((8, 8), -1, dtype=cp.int8)
    d_bar = cp.zeros(1, dtype=cp.int32)
    opts = _launch_options(kernel, tpb)
    instrumented = not bare
    if kernel == "split":
        d_spill0 = cp.empty(32, dtype=cp.int32)
        d_spill1 = cp.empty(32, dtype=cp.int32)
        d_inbox0 = cp.empty(8, dtype=cp.int32)
        d_inbox1 = cp.empty(8, dtype=cp.int32)
        d_g = cp.zeros(6, dtype=cp.int32)
        d_ring = cp.empty((2, int(RING_CAPACITY)), dtype=cp.int32)
        compiled = dual_block_split_kernel[(2,)](
            t(d_img), t(d_visited), t(d_depth), t(d_owner), 1, 1,
            t(d_spill0), t(d_spill1), t(d_inbox0), t(d_inbox1), t(d_g),
            t(d_counters), t(d_trace), t(d_ring), t(d_bar), 8, 8, 8, 4,
            TPB=tpb, INSTRUMENTED=instrumented, **opts)
    elif kernel in ("global", "dirsplit"):
        d_queue = cp.empty(64, dtype=cp.int32)
        d_queue[0] = 1 * 8 + 1
        init = [1] if kernel == "global" else [1, 0]
        d_q = cp.asarray(np.array(init, dtype=np.int32))
        compiled = _KERNELS[kernel][(2,)](
            t(d_img), t(d_visited), t(d_depth), t(d_owner), t(d_queue),
            t(d_q), t(d_counters), t(d_trace), t(d_bar), 8, 8, 4,
            TPB=tpb, INSTRUMENTED=instrumented, **opts)
    else:  # pinned: compile via a 2-program spread run
        d_queue = cp.empty(64, dtype=cp.int32)
        d_queue[0] = 1 * 8 + 1
        d_q = cp.asarray(np.array([1], dtype=np.int32))
        d_barrier = cp.zeros(2, dtype=cp.int32)
        d_pin = cp.asarray(np.array([1, -1, 0], dtype=np.int32))
        compiled = dual_block_pinned_kernel[(2,)](
            t(d_img), t(d_visited), t(d_depth), t(d_queue), t(d_q),
            t(d_barrier), t(d_pin), t(d_counters), 8, 8, TPB=tpb, **opts)
    sync()
    counters = d_counters.get()
    if counters[OVERFLOW] or counters[FILLED] != 1:
        raise RuntimeError(f"warm-up of the {kernel!r} twin produced a wrong "
                           f"result: counters {counters}")
    _warmed[key] = compiled
    return compiled


def compiled_kernel(kernel, bare=False, threads_per_block=256):
    """The CompiledKernel behind a configuration (warming it up if needed):
    for kernel_resources() and the occupancy calculator."""
    return _warmup(kernel, bare, threads_per_block)


def _coop_max_blocks(kernel, bare, tpb):
    key = (kernel, bare, tpb)
    if key not in _coop_cache:
        _coop_cache[key] = max_coresident_programs(_warmup(kernel, bare, tpb))
    return _coop_cache[key]


def launch_grid(kernel, placement=None, threads_per_block=256, bare=False):
    """Programs a configuration launches (2, or C * sm_count for pinned
    same_sm, C = resident programs per SM of the compiled pinned kernel)."""
    if kernel == "pinned" and placement == "same_sm":
        per_sm = programs_per_sm(_warmup(kernel, bare, threads_per_block))
        return per_sm * device_info().sm_count
    return 2


def flood_fill(img_host, seed_x, seed_y, threads_per_block=256,
               kernel="split", bare=False, placement=None):
    """Flood-fill the red blob containing (seed_x, seed_y) with two programs.

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
        raise ValueError(f"seed pixel ({seed_x}, {seed_y}) is not red - nothing to fill")
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
                f"{PINNED_TPB}: 512-lane programs capped at {PINNED_MAXNREG} "
                f"registers fit 2 per SM, which is what pins the two "
                f"workers to one SM (Numba's 768 is not a power of 2)")
        if bare:
            raise ValueError("the pinned kernel is minimal by design; "
                             "bare=True does not apply")
    else:
        if placement is not None:
            raise ValueError("placement only applies to kernel=\"pinned\"")
        # Same 512 cap as Numba (the dual kernels' coordination state costs
        # too many registers for 1024-thread blocks), plus Triton's rule.
        if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 512):
            raise ValueError(
                f"threads_per_block must be a multiple of 32 in [32, 512], "
                f"got {threads_per_block}")
        if threads_per_block & (threads_per_block - 1):
            raise ValueError(
                f"threads_per_block must be a power of 2 in the Triton twin "
                f"(num_warps = threads_per_block // 32 and tl.arange lengths "
                f"are powers of 2), got {threads_per_block}")

    _warmup(kernel, bare, threads_per_block)
    sm_count = device_info().sm_count
    launch_blocks = launch_grid(kernel, placement, threads_per_block, bare)
    if kernel == "pinned" and placement == "same_sm" and \
            launch_blocks < PINNED_WORKERS * sm_count:
        raise RuntimeError(
            f"the pinned kernel fits {launch_blocks // sm_count} program(s) "
            f"of {threads_per_block} lanes per SM; at least "
            f"{PINNED_WORKERS} are needed to put both workers on one SM")
    max_blocks = _coop_max_blocks(kernel, bare, threads_per_block)
    if max_blocks < launch_blocks:
        raise RuntimeError(
            f"this GPU supports at most {max_blocks} cooperative blocks of "
            f"{threads_per_block} threads for kernel {kernel!r}; "
            f"{launch_blocks} required")
    opts = _launch_options(kernel, threads_per_block)

    seed_lin = np.array([seed_x * height + seed_y], dtype=np.int32)
    half = width // 2
    trace_capacity = min(width * height, LEVEL_TRACE_CAPACITY)
    instrumented = not bare and kernel != "pinned"

    t_total0 = time.perf_counter()

    visited_host = np.zeros((width, height), dtype=np.int32)
    visited_host[seed_x, seed_y] = 1
    depth_host = np.full((width, height), -1, dtype=np.int32)
    counters_host = np.zeros(int(NUM_COUNTERS), dtype=np.int64)
    d_img = cp.empty_like(img_host)
    d_visited = cp.empty_like(visited_host)
    d_depth = cp.empty_like(depth_host)
    d_counters = cp.empty_like(counters_host)
    d_bar = cp.empty(1, dtype=cp.int32)   # grid_sync's arrival counter
    if kernel == "split":
        d_spill0 = cp.empty(max(half * height, 1), dtype=cp.int32)
        d_spill1 = cp.empty(max((width - half) * height, 1), dtype=cp.int32)
        d_inbox0 = cp.empty(max(height, 1), dtype=cp.int32)
        d_inbox1 = cp.empty(max(height, 1), dtype=cp.int32)
        # the two programs' rings (Numba: one shared array per block)
        d_ring = cp.empty((2, int(RING_CAPACITY)), dtype=cp.int32)
    else:
        d_queue = cp.empty(width * height, dtype=cp.int32)
    if instrumented:
        d_trace = cp.empty((2, trace_capacity), dtype=cp.int32)
        owner_host = np.full((width, height), -1, dtype=np.int8)
        d_owner = cp.empty_like(owner_host)
    else:
        d_trace = cp.empty((2, 1), dtype=cp.int32)   # unused by bare twins
        d_owner = cp.empty(1, dtype=cp.int8)
    sync()
    t_h2d0 = time.perf_counter()

    d_img.set(img_host)
    d_visited.set(visited_host)
    d_depth.set(depth_host)
    d_counters.set(counters_host)
    d_bar.fill(0)
    if instrumented:
        d_owner.set(owner_host)
    if kernel == "split":
        d_g_state = cp.asarray(np.zeros(6, dtype=np.int32))
    elif kernel == "global":
        d_queue[:1].set(seed_lin)
        d_q_state = cp.asarray(np.array([1], dtype=np.int32))
    elif kernel == "dirsplit":
        d_queue[:1].set(seed_lin)
        d_q_state = cp.asarray(np.array([1, 0], dtype=np.int32))
    else:  # pinned
        d_queue[:1].set(seed_lin)
        d_q_state = cp.asarray(np.array([1], dtype=np.int32))
        d_barrier = cp.asarray(np.zeros(2, dtype=np.int32))
        mode = 0 if placement == "same_sm" else 1
        d_pin = cp.asarray(np.array([mode, -1, 0], dtype=np.int32))
    sync()
    t_kernel0 = time.perf_counter()

    if kernel == "split":
        dual_block_split_kernel[(2,)](
            t(d_img), t(d_visited), t(d_depth), t(d_owner), seed_x, seed_y,
            t(d_spill0), t(d_spill1), t(d_inbox0), t(d_inbox1),
            t(d_g_state), t(d_counters), t(d_trace), t(d_ring), t(d_bar),
            width, height, max(height, 1), trace_capacity,
            TPB=threads_per_block, INSTRUMENTED=instrumented, **opts)
    elif kernel in ("global", "dirsplit"):
        _KERNELS[kernel][(2,)](
            t(d_img), t(d_visited), t(d_depth), t(d_owner), t(d_queue),
            t(d_q_state), t(d_counters), t(d_trace), t(d_bar), width, height,
            trace_capacity, TPB=threads_per_block, INSTRUMENTED=instrumented,
            **opts)
    else:  # pinned
        dual_block_pinned_kernel[(launch_blocks,)](
            t(d_img), t(d_visited), t(d_depth), t(d_queue), t(d_q_state),
            t(d_barrier), t(d_pin), t(d_counters), width, height,
            TPB=threads_per_block, **opts)
    sync()
    t_d2h0 = time.perf_counter()

    counters = d_counters.get()
    if counters[OVERFLOW]:
        raise RuntimeError(
            "structural tripwire fired - this indicates a kernel bug: no "
            "queue in this package can legitimately overflow")
    levels = int(counters[LEVELS])
    truncated = instrumented and levels > trace_capacity
    img_out = d_img.get()
    visited_out = d_visited.get()
    depth_out = d_depth.get()
    owner_out = (d_owner.get() if instrumented
                 else np.zeros((0, 0), dtype=np.int8))
    if instrumented:
        n = min(levels, trace_capacity)
        row0 = d_trace[0, :n].get()
        row1 = d_trace[1, :n].get()
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
        owner=owner_out,
        levels=levels,
        filled=filled,
        peak_level=int(counters[PEAK_LEVEL]),
        peak_occupancy=int(counters[PEAK_OCC]),
        ring_capacity=int(RING_CAPACITY),
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
        / device_info().max_threads_per_sm,
        sm_utilization_pct=100.0 * (1 if (kernel == "pinned"
                                          and placement == "same_sm")
                                    else 2) / sm_count,
        discovery_redundancy=(cas_attempts / max(filled - 1, 1)
                              if instrumented else 0.0),
        neighbor_check_efficiency_pct=(100.0 * filled / (4 * processed)
                                       if instrumented and processed else 0.0),
    )


__all__ = [
    "DualFloodFillResult", "flood_fill", "compiled_kernel", "launch_grid",
    "kernel_resources", "LEVEL_TRACE_CAPACITY", "PINNED_TPB",
    "PINNED_MAXNREG", "SPLIT_MAXNREG",
]
