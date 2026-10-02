"""Host driver for the Triton twin of the dual-blob flood fill.

Public API (identical to the Numba chapter's flood_fill.py):
    flood_fill(img, seeds, mode="multisource", threads_per_block=256,
               blocks=None, bare=False, connectivity=4, entry_format="lin",
               radius=1)
        -> DualBlobResult
    max_blocks(threads_per_block=256, bare=False, connectivity=4,
               entry_format="lin", radius=1)

Same modes, same validation (messages included), same result fields and
the same timing brackets (time.perf_counter around runtime.sync()). What
changes is the runtime underneath:

- CuPy arrays replace Numba device arrays; kernels launch through the
  CuPy-backed Triton driver (runtime.t / runtime.sync).
- Every launch is cooperative (launch_cooperative_grid=True), the twin of
  Numba's implicit cooperative launch for a kernel with grid.sync. Each
  launch gets its own zeroed int64 barrier counter for grid_sync.
- threads_per_block must also be a power of 2 (tl.arange and num_warps),
  so Numba's 96, 160, ... are refused with a ValueError that says so.
- Triton compiles per BLOCK, so warm-up is per (variant, tpb), not one
  [1, 32] launch per variant; max_blocks() and flood_fill() both warm the
  exact binary they size or launch, so no compile lands in a timed window.
- streams mode runs the two cooperative launches on two CuPy streams. It
  stays opt-in in the tests for the same reason as in Numba (concurrent
  cooperative grids can wedge), and an explicit grid whose PAIR would not
  fit the cooperative capacity is refused here (Numba only checks each
  launch on its own).

seeds is a list of exactly two (x, y) red pixels in two DIFFERENT
connected components; seeds[0]'s blob comes back blue, seeds[1]'s green.
"""

import os

# shared.bandwidth imports numba: the binding choice must come first
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from ....shared.bandwidth import (
    model_bytes as _model_bytes, model_gb_s as _model_gb_s,
)
from ...runtime import kernel_resources, max_coresident_programs, sync, t
from .kernels import (
    dual_blob_lin_kernel, dual_blob_lin_bare_kernel,
    dual_blob_lin8_kernel, dual_blob_lin8_bare_kernel,
    dual_blob_lin8r2_kernel, dual_blob_lin8r2_bare_kernel,
    dual_blob_xy_kernel, dual_blob_xy_bare_kernel,
    dual_blob_xy8_kernel, dual_blob_xy8_bare_kernel,
    PALETTE_HOST,
    XY_FIELD_BITS, XY_LBL_SHIFT, XY_MAX_DIM,
    NUM_COUNTERS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS, INTERIOR,
    BS_PROCESSED, BS_SMID,
)

MODES = ("sequential", "streams", "multisource")
ENTRY_FORMATS = ("lin", "xy")

# (entry_format, bare, connectivity, radius) -> kernel, as in Numba
_KERNELS = {
    ("lin", False, 4, 1): dual_blob_lin_kernel,
    ("lin", True, 4, 1): dual_blob_lin_bare_kernel,
    ("lin", False, 8, 1): dual_blob_lin8_kernel,
    ("lin", True, 8, 1): dual_blob_lin8_bare_kernel,
    ("lin", False, 8, 2): dual_blob_lin8r2_kernel,
    ("lin", True, 8, 2): dual_blob_lin8r2_bare_kernel,
    ("xy", False, 4, 1): dual_blob_xy_kernel,
    ("xy", True, 4, 1): dual_blob_xy_bare_kernel,
    ("xy", False, 8, 1): dual_blob_xy8_kernel,
    ("xy", True, 8, 1): dual_blob_xy8_bare_kernel,
}

# Cap on the recorded per-level trace (1D int32 -> 8 MB max), per launch
LEVEL_TRACE_CAPACITY = 2 ** 21

# Lane counts a Triton program can have (power-of-2 tl.arange and num_warps)
POW2_TPB = (32, 64, 128, 256, 512)

# Grid barrier counter: 2 arrivals per program per level. int64 cannot
# wrap in practice, like Numba's grid.sync.
BAR_DTYPE = np.int64


def _pack(x, y, lbl, height, entry_format):
    if entry_format == "xy":
        return (lbl << XY_LBL_SHIFT) | (x << XY_FIELD_BITS) | y
    return ((x * height + y) << 1) | lbl


@dataclass
class LaunchStats:
    """Per-cooperative-launch statistics (one per blob for the two-launch
    modes; a single combined one for multisource, label=-1)."""
    label: int             # blob this launch flooded; -1 = both (multisource)
    blocks: int
    filled: int
    levels: int
    peak_level: int        # largest single frontier
    peak_occupancy: int    # max queue entries alive across adjacent levels
    processed: int
    cas_attempts: int
    interior: int          # radius-2 only: guard passes (else 0)
    kernel_ms: float       # device time (CUDA events under streams)
    thread_util_pct: float
    processed_per_block: np.ndarray = field(repr=False)
    sm_ids: list = field(repr=False)
    distinct_sms: int = 0
    level_sizes: np.ndarray = field(repr=False, default=None)
    level_trace_truncated: bool = False


@dataclass
class DualBlobResult:
    img: np.ndarray        # (w, h, 3) uint8: blob 0 blue, blob 1 green
    visited: np.ndarray    # (w, h) int32, 1 where reached
    depth: np.ndarray      # (w, h) int32, BFS level per pixel (per blob)
    owner: np.ndarray      # (w, h) int16, program id per pixel (per launch);
                           # empty for bare
    label: np.ndarray      # (w, h) int8: 0 blue blob, 1 green blob, -1,
                           # derived on host from img+visited
    mode: str
    seeds: list
    threads_per_block: int
    blocks: int            # per-launch grid size actually used
    bare: bool
    connectivity: int
    radius: int            # 1, or 2 for the guarded radius-2 twins
    entry_format: str
    # Combined outcome
    filled: int
    filled_a: int
    filled_b: int
    levels: int            # sequential/streams: max(a, b); multisource: kernel
    levels_a: int
    levels_b: int
    processed: int
    cas_attempts: int
    interior: int          # radius-2 only: guard passes summed over launches
    # Timing (ms). kernel_ms: sequential = tA + tB; streams = wall clock
    # around both launches; multisource = the single launch.
    kernel_ms: float
    kernel_a_ms: float     # 0.0 for multisource (indivisible)
    kernel_b_ms: float
    overlap_ratio: float   # streams only: (tA + tB) / wall; 0.0 otherwise
    alloc_ms: float
    h2d_ms: float
    d2h_ms: float
    total_ms: float
    # Derived bandwidth model over combined counters and kernel_ms
    model_bytes: int
    model_gb_s: float
    launches: list = field(repr=False, default=None)  # of LaunchStats


# (key, tpb) -> CompiledKernel of the warm-up launch; (key, tpb) -> capacity
_warmed = {}
_coop_cache = {}
_streams_ok = None      # None = unprobed; True/False after first warmup
_streams_err = ""


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _check_tpb(threads_per_block):
    # 512 cap kept from the Numba driver (a 1024-thread block cannot be
    # resident with these kernels' register use)
    if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 512):
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 512], "
            f"got {threads_per_block}")
    if threads_per_block not in POW2_TPB:
        raise ValueError(
            f"threads_per_block must be a power of 2 for the Triton twin "
            f"(tl.arange lengths and num_warps are powers of 2): 32, 64, "
            f"128, 256 or 512, got {threads_per_block}")


def _launch(kernel_fn, blocks, tpb, instrumented, d_img, d_visited, d_depth,
            d_owner, d_queue, d_q_state, d_counters, d_stats, d_trace, d_bar,
            width, height, n_seeds):
    """One cooperative launch on the current CuPy stream; returns the
    CompiledKernel (its registers size the cooperative grid)."""
    common = dict(BLOCK=tpb, num_warps=tpb // 32, num_stages=1,
                  launch_cooperative_grid=True)
    if instrumented:
        return kernel_fn[(blocks,)](
            t(d_img), t(d_visited), t(d_depth), t(d_owner), t(d_queue),
            t(d_q_state), t(d_counters), t(d_stats), t(d_trace), n_seeds,
            t(d_bar), width, height, d_queue.shape[0], d_trace.shape[0],
            **common)
    return kernel_fn[(blocks,)](
        t(d_img), t(d_visited), t(d_depth), t(d_queue), t(d_q_state),
        t(d_counters), n_seeds, t(d_bar), width, height, d_queue.shape[0],
        **common)


def _tiny_args(entry_format):
    """Two isolated red pixels of an 8x8 white image (labels 0 and 1): no
    discoveries, so a compile launch terminates under any program count
    and exercises the multi-seed init (q_state=[2])."""
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    tiny[5, 5] = (255, 0, 0)
    d_img = cp.asarray(tiny)
    visited = np.zeros((8, 8), dtype=np.int32)
    visited[1, 1] = 1
    visited[5, 5] = 1
    d_visited = cp.asarray(visited)
    d_depth = cp.asarray(np.full((8, 8), -1, dtype=np.int32))
    d_counters = cp.zeros(NUM_COUNTERS, dtype=cp.int64)
    d_queue = cp.empty(64, dtype=cp.int32)
    seeds = np.array([_pack(1, 1, 0, 8, entry_format),
                      _pack(5, 5, 1, 8, entry_format)], dtype=np.int32)
    d_queue[:2].set(seeds)
    d_q = cp.asarray(np.array([2], dtype=np.int32))
    return d_img, d_visited, d_depth, d_counters, d_queue, d_q


def _tiny_launch(key, tpb):
    """The [1 program x tpb lanes] launch on the tiny image (n_seeds=2)."""
    entry_format, bare = key[0], key[1]
    d_img, d_visited, d_depth, d_counters, d_queue, d_q = \
        _tiny_args(entry_format)
    d_bar = cp.zeros(1, dtype=BAR_DTYPE)
    if bare:
        d_owner = d_stats = d_trace = None
    else:
        d_owner = cp.asarray(np.full((8, 8), -1, dtype=np.int16))
        d_stats = cp.zeros((1, 2), dtype=cp.int64)
        d_trace = cp.empty(4, dtype=cp.int32)
    return _launch(_KERNELS[key], 1, tpb, not bare, d_img, d_visited,
                   d_depth, d_owner, d_queue, d_q, d_counters, d_stats,
                   d_trace, d_bar, 8, 8, 2)


def _warmup(entry_format, bare, connectivity=4, radius=1,
            threads_per_block=256):
    """Compile each (kernel, tpb) once, off the clock, and return its
    CompiledKernel. Numba compiles one binary per kernel and warms it with
    a [1, 32] launch; a Triton program's lane count is a compile-time
    constant, so the twin warms the exact binary the timed launch uses.
    The first instrumented warmup also probes whether a cooperative
    launch on a non-default stream works (mode="streams" needs it); the
    outcome is recorded, never assumed."""
    global _streams_ok, _streams_err
    key = (entry_format, bare, connectivity, radius)
    wkey = (key, threads_per_block)
    if wkey not in _warmed:
        _warmed[wkey] = _tiny_launch(key, threads_per_block)
        sync()
    if _streams_ok is None and not bare:
        try:
            s = cp.cuda.Stream()
            with s:
                _tiny_launch((entry_format, False, connectivity, radius),
                             threads_per_block)
            s.synchronize()
            _streams_ok = True
        except Exception as exc:  # record, don't crash: streams mode raises
            _streams_ok = False
            _streams_err = f"{type(exc).__name__}: {exc}"
    return _warmed[wkey]


def _coop_max_blocks(key, tpb):
    if (key, tpb) not in _coop_cache:
        _coop_cache[(key, tpb)] = max_coresident_programs(_warmed[(key, tpb)])
    return _coop_cache[(key, tpb)]


def kernel_info(threads_per_block=256, bare=False, connectivity=4,
                entry_format="lin", radius=1):
    """Registers, spills, shared bytes and warps of the compiled twin
    (runtime.occupancy.kernel_resources); compiles it on first call.
    Twin-only helper (the compare script records it per row)."""
    _check_tpb(threads_per_block)
    return kernel_resources(_warmup(entry_format, bare, connectivity, radius,
                                    threads_per_block))


def max_blocks(threads_per_block=256, bare=False, connectivity=4,
               entry_format="lin", radius=1):
    """The largest cooperative grid this GPU can host at threads_per_block
    for ONE launch of that exact twin (what blocks=None resolves to
    outside streams mode). Queried per compiled binary, never assumed
    equal across twins or to the Numba kernels' capacity."""
    _check_tpb(threads_per_block)
    _warmup(entry_format, bare, connectivity, radius, threads_per_block)
    return _coop_max_blocks((entry_format, bare, connectivity, radius),
                            threads_per_block)


def flood_fill(img_host, seeds, mode="multisource", threads_per_block=256,
               blocks=None, bare=False, connectivity=4, entry_format="lin",
               radius=1):
    """Flood-fill two disconnected red blobs, blob 0 blue / blob 1 green.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for bad inputs, RuntimeError if the GPU
    cannot host the requested cooperative launch (or a structural tripwire
    fires), and NotImplementedError for mode="streams" if the driver
    rejected a cooperative launch on a non-default stream (probed at
    warmup, recorded in the error message).
    """
    if img_host.ndim != 3 or img_host.shape[2] != 3 or img_host.dtype != np.uint8:
        raise ValueError("img must be a (width, height, 3) uint8 array")
    width, height = img_host.shape[0], img_host.shape[1]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if entry_format not in ENTRY_FORMATS:
        raise ValueError(
            f"entry_format must be one of {ENTRY_FORMATS}, got {entry_format!r}")
    if entry_format == "lin" and width * height >= 2 ** 30:
        raise ValueError(
            "image too large for lin-format entries: width*height must be "
            "< 2**30 (the label bit occupies one bit of the int32)")
    if entry_format == "xy" and (width > XY_MAX_DIM or height > XY_MAX_DIM):
        raise ValueError(
            f"image too large for xy-format entries: each dimension must "
            f"be <= {XY_MAX_DIM}")
    try:
        (ax, ay), (bx, by) = ((int(x), int(y)) for x, y in seeds)
    except (TypeError, ValueError):
        raise ValueError(
            f"seeds must be exactly two (x, y) pairs, got {seeds!r}")
    seeds = [(ax, ay), (bx, by)]
    for sx, sy in seeds:
        if not (0 <= sx < width and 0 <= sy < height):
            raise ValueError(
                f"seed ({sx}, {sy}) outside {width}x{height} image")
        if not _is_red(img_host, sx, sy):
            raise ValueError(
                f"seed pixel ({sx}, {sy}) is not red - nothing to fill")
    if seeds[0] == seeds[1]:
        raise ValueError(f"seeds must be two distinct pixels, got {seeds}")
    _check_tpb(threads_per_block)
    if blocks is not None:
        if not isinstance(blocks, (int, np.integer)) or isinstance(blocks, bool):
            raise ValueError(f"blocks must be an int or None, got {blocks!r}")
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    if radius not in (1, 2):
        raise ValueError(f"radius must be 1 or 2, got {radius!r}")
    if radius == 2 and (connectivity != 8 or entry_format != "lin"):
        raise ValueError(
            "radius=2 requires connectivity=8 and entry_format='lin' (the "
            "guarded ring-2 twins exist only for the lin conn8 kernels)")

    _warmup(entry_format, bare, connectivity, radius, threads_per_block)
    if mode == "streams" and not _streams_ok:
        raise NotImplementedError(
            "mode='streams' unavailable: this Triton/driver rejected a "
            f"cooperative launch on a non-default stream ({_streams_err})")

    key = (entry_format, bare, connectivity, radius)
    coop_max = _coop_max_blocks(key, threads_per_block)
    if blocks is None:
        # Streams default: a THIRD of capacity per launch (the Numba
        # driver's probed margin: two concurrent cooperative grids near
        # capacity can wedge forever at the grid barrier).
        launch_blocks = max(coop_max // 3, 1) if mode == "streams" else coop_max
    elif blocks > coop_max:
        raise RuntimeError(
            f"blocks={blocks} exceeds this GPU's cooperative-launch capacity "
            f"of {coop_max} blocks at {threads_per_block} threads "
            f"(grid.sync would deadlock)")
    elif mode == "streams" and 2 * blocks > coop_max:
        # Twin-only rail: the two grids run at once, so the PAIR must fit.
        raise RuntimeError(
            f"blocks={blocks} per stream: the pair ({2 * blocks}) exceeds "
            f"this GPU's cooperative-launch capacity of {coop_max} blocks at "
            f"{threads_per_block} threads (concurrent grid barriers would "
            f"deadlock)")
    else:
        launch_blocks = int(blocks)

    n_launches = 1 if mode == "multisource" else 2
    trace_capacity = min(width * height, LEVEL_TRACE_CAPACITY)
    instrumented = not bare

    t_total0 = time.perf_counter()

    visited_host = np.zeros((width, height), dtype=np.int32)
    for sx, sy in seeds:
        visited_host[sx, sy] = 1
    depth_host = np.full((width, height), -1, dtype=np.int32)
    d_img = cp.empty(img_host.shape, dtype=cp.uint8)
    d_visited = cp.empty(visited_host.shape, dtype=cp.int32)
    d_depth = cp.empty(depth_host.shape, dtype=cp.int32)
    d_queues = [cp.empty(width * height, dtype=cp.int32)
                for _ in range(n_launches)]
    d_counters = [cp.empty(NUM_COUNTERS, dtype=cp.int64)
                  for _ in range(n_launches)]
    d_bars = [cp.empty(1, dtype=BAR_DTYPE) for _ in range(n_launches)]
    if instrumented:
        owner_host = np.full((width, height), -1, dtype=np.int16)
        d_owner = cp.empty(owner_host.shape, dtype=cp.int16)
        stats_host = np.zeros((launch_blocks, 2), dtype=np.int64)
        stats_host[:, BS_SMID] = -1
        d_stats = [cp.empty(stats_host.shape, dtype=cp.int64)
                   for _ in range(n_launches)]
        d_traces = [cp.empty(trace_capacity, dtype=cp.int32)
                    for _ in range(n_launches)]
    else:
        d_owner = None
        d_stats = d_traces = [None] * n_launches
    sync()
    t_h2d0 = time.perf_counter()

    d_img.set(img_host)
    d_visited.set(visited_host)
    d_depth.set(depth_host)
    zeros = np.zeros(NUM_COUNTERS, dtype=np.int64)
    for d_c in d_counters:
        d_c.set(zeros)
    for d_b in d_bars:
        d_b.set(np.zeros(1, dtype=BAR_DTYPE))
    if mode == "multisource":
        packed = np.array(
            [_pack(ax, ay, 0, height, entry_format),
             _pack(bx, by, 1, height, entry_format)], dtype=np.int32)
        d_queues[0][:2].set(packed)
        d_q_states = [cp.asarray(np.array([2], dtype=np.int32))]
    else:
        d_q_states = []
        for i, (sx, sy) in enumerate(seeds):
            packed = np.array([_pack(sx, sy, i, height, entry_format)],
                              dtype=np.int32)
            d_queues[i][:1].set(packed)
            d_q_states.append(cp.asarray(np.array([1], dtype=np.int32)))
    if instrumented:
        d_owner.set(owner_host)
        for d_s in d_stats:
            d_s.set(stats_host)
    sync()
    t_kernel0 = time.perf_counter()

    # The seed count is a launch-uniform kernel ARGUMENT (reading the
    # initial rear from q_state inside the kernel races with the first
    # enqueues and deadlocks the grid barrier: Numba kernels.py, Finding 1).
    n_seeds = 2 if mode == "multisource" else 1

    kernel_fn = _KERNELS[key]

    def _go(i):
        _launch(kernel_fn, launch_blocks, threads_per_block, instrumented,
                d_img, d_visited, d_depth, d_owner, d_queues[i],
                d_q_states[i], d_counters[i], d_stats[i], d_traces[i],
                d_bars[i], width, height, n_seeds)

    overlap_ratio = 0.0
    if mode == "sequential":
        per_launch_ms = []
        for i in range(2):
            t0 = time.perf_counter()
            _go(i)
            sync()
            per_launch_ms.append((time.perf_counter() - t0) * 1000)
        kernel_ms = sum(per_launch_ms)
    elif mode == "streams":
        streams = [cp.cuda.Stream(), cp.cuda.Stream()]
        events = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(2)]
        t0 = time.perf_counter()
        for i in range(2):  # issue BOTH before syncing either
            with streams[i]:
                events[i][0].record(streams[i])
                _go(i)
                events[i][1].record(streams[i])
        for s in streams:
            s.synchronize()
        kernel_ms = (time.perf_counter() - t0) * 1000  # wall around both
        per_launch_ms = [cp.cuda.get_elapsed_time(e0, e1)
                         for e0, e1 in events]
        overlap_ratio = (sum(per_launch_ms) / kernel_ms
                         if kernel_ms > 0 else 0.0)
    else:  # multisource
        t0 = time.perf_counter()
        _go(0)
        sync()
        kernel_ms = (time.perf_counter() - t0) * 1000
        per_launch_ms = []
    t_d2h0 = time.perf_counter()

    counters = [d_c.get() for d_c in d_counters]
    for c in counters:
        if c[OVERFLOW]:
            raise RuntimeError(
                "structural tripwire fired - this indicates a kernel bug: "
                "the queue cannot legitimately overflow")
    img_out = d_img.get()
    visited_out = d_visited.get()
    depth_out = d_depth.get()
    if instrumented:
        owner_out = d_owner.get()
        stats = [d_s.get() for d_s in d_stats]
        traces = [d_traces[i][:min(int(counters[i][LEVELS]), trace_capacity)]
                  .get() for i in range(n_launches)]
    else:
        owner_out = np.zeros((0, 0), dtype=np.int16)
        stats = [None] * n_launches
        traces = [np.zeros(0, dtype=np.int32)] * n_launches

    # The painted image IS the label map: recover it from the colors.
    label_out = np.full((width, height), -1, dtype=np.int8)
    label_out[np.all(img_out == PALETTE_HOST[0], axis=2)
              & (visited_out == 1)] = 0
    label_out[np.all(img_out == PALETTE_HOST[1], axis=2)
              & (visited_out == 1)] = 1
    t_end = time.perf_counter()

    grid_threads = launch_blocks * threads_per_block
    launches = []
    for i in range(n_launches):
        c = counters[i]
        levels_i = int(c[LEVELS])
        if instrumented:
            ppb = stats[i][:, BS_PROCESSED].copy()
            sm_ids = [int(s) for s in stats[i][:, BS_SMID]]
            distinct = len({s for s in sm_ids if s >= 0})
            util = (100.0 * int(c[ACTIVE_THREAD_SUM])
                    / (levels_i * grid_threads) if levels_i else 0.0)
        else:
            ppb = np.zeros(0, dtype=np.int64)
            sm_ids, distinct, util = [], 0, 0.0
        launches.append(LaunchStats(
            label=(-1 if mode == "multisource" else i),
            blocks=launch_blocks,
            filled=int(c[FILLED]),
            levels=levels_i,
            peak_level=int(c[PEAK_LEVEL]),
            peak_occupancy=int(c[PEAK_OCC]),
            processed=int(c[PROCESSED]),
            cas_attempts=int(c[CAS_ATTEMPTS]),
            interior=int(c[INTERIOR]),
            kernel_ms=(per_launch_ms[i] if per_launch_ms else kernel_ms),
            thread_util_pct=util,
            processed_per_block=ppb,
            sm_ids=sm_ids,
            distinct_sms=distinct,
            level_sizes=traces[i],
            level_trace_truncated=(instrumented
                                   and levels_i > trace_capacity),
        ))

    filled_a = int((label_out == 0).sum())
    filled_b = int((label_out == 1).sum())
    if mode == "multisource":
        levels_a = int(depth_out[label_out == 0].max()) + 1 if filled_a else 0
        levels_b = int(depth_out[label_out == 1].max()) + 1 if filled_b else 0
        levels = launches[0].levels
    else:
        levels_a = launches[0].levels
        levels_b = launches[1].levels
        levels = max(levels_a, levels_b)
    filled = sum(l.filled for l in launches)
    processed = sum(l.processed for l in launches)
    cas_attempts = sum(l.cas_attempts for l in launches)
    interior = sum(l.interior for l in launches)
    kernel_a_ms = per_launch_ms[0] if per_launch_ms else 0.0
    kernel_b_ms = per_launch_ms[1] if per_launch_ms else 0.0
    # Exact probe count: radius-2 pixels probe 8 always + 16 when interior.
    probe_reads = (8 * processed + 16 * interior if radius == 2
                   else processed * connectivity)
    mbytes = (_model_bytes(processed, cas_attempts, filled, True,
                           n_dirs=connectivity, probe_reads=probe_reads)
              if instrumented else 0)

    return DualBlobResult(
        img=img_out,
        visited=visited_out,
        depth=depth_out,
        owner=owner_out,
        label=label_out,
        mode=mode,
        seeds=seeds,
        threads_per_block=threads_per_block,
        blocks=launch_blocks,
        bare=bare,
        connectivity=connectivity,
        radius=radius,
        entry_format=entry_format,
        filled=filled,
        filled_a=filled_a,
        filled_b=filled_b,
        levels=levels,
        levels_a=levels_a,
        levels_b=levels_b,
        processed=processed,
        cas_attempts=cas_attempts,
        interior=interior,
        kernel_ms=kernel_ms,
        kernel_a_ms=kernel_a_ms,
        kernel_b_ms=kernel_b_ms,
        overlap_ratio=overlap_ratio,
        alloc_ms=(t_h2d0 - t_total0) * 1000,
        h2d_ms=(t_kernel0 - t_h2d0) * 1000,
        d2h_ms=(t_end - t_d2h0) * 1000,
        total_ms=(t_end - t_total0) * 1000,
        model_bytes=mbytes,
        model_gb_s=_model_gb_s(mbytes, kernel_ms),
        launches=launches,
    )
