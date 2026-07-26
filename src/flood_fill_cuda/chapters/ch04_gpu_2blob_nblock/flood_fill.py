"""Host driver for the dual-blob flood fill.

Public API:
    flood_fill(img, seeds, mode="multisource", threads_per_block=256,
               blocks=None, bare=False, connectivity=4, entry_format="lin",
               radius=1)
        -> DualBlobResult
    max_blocks(threads_per_block=256, bare=False, connectivity=4,
               entry_format="lin", radius=1)

radius=2 (requires connectivity=8 and entry_format="lin") selects the
guarded radius-2 twins ported from multi_block: ring-1 probed first with
the unchanged protocol; only pixels whose entire ring-1 is in-bounds blob
material also probe the 16 ring-2 cells, inheriting the dequeuer's label.
Fill set and label map identical to plain conn8; levels roughly halve on
solid blobs. Instrumented results report `interior` (guard passes).

seeds is a list of exactly two (x, y) red pixels lying in two DIFFERENT
connected components (the caller's contract — reference.py's merged oracle
verifies it in tests). seeds[0]'s blob comes back blue, seeds[1]'s green,
painted in-kernel.

Three modes flood the two blobs "in parallel" three different ways:

- "sequential": two single-seed cooperative launches back-to-back, each
  with its own queue/counters. Wall cost ~ tA + tB. The baseline.
- "streams": the same two launches issued on two CUDA streams, each at a
  THIRD of the cooperative capacity by default (a probed safety margin:
  two concurrent cooperative grids near device capacity wedge forever at
  grid.sync on this GPU — 88+88 of 192 deadlocked, 80+80 survived).
  Whether the driver actually co-schedules two cooperative grids is
  MEASURED, not assumed: overlap_ratio = (tA + tB) / wall reads ~1.0 when
  the launches serialized and approaches 2.0 when they truly overlapped.
  Per-launch times come from CUDA events (host timers cannot attribute
  overlapped work).
- "multisource": both seeds pre-loaded into ONE shared queue; a single
  cooperative launch floods both blobs simultaneously, labels riding in
  the queue entries. Wall cost ~ max(tA, tB) plus whatever wider
  frontiers buy in utilization.

entry_format picks the labeled-queue encoding ("lin" or "xy" — see
kernels.py for the head-to-head bet between them).

blocks=None launches the maximum cooperative grid per launch (halved for
streams so both grids can be resident together); explicit blocks are
validated against the same per-launch limit. bare=True selects the
uninstrumented twins (timing/filled/levels only; zeroed work metrics).

All device buffers that both launches touch (img, visited, depth, owner)
are SHARED — the blobs are disjoint pixel sets, so the two launches never
contend — while each launch owns its queue, rear counter, counters,
block_stats and trace ("its own queue", made real).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ...shared.bandwidth import model_bytes as _model_bytes, model_gb_s as _model_gb_s
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
from numba import cuda

MODES = ("sequential", "streams", "multisource")
ENTRY_FORMATS = ("lin", "xy")

# (entry_format, bare, connectivity, radius) -> kernel
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
    owner: np.ndarray      # (w, h) int16, block id per pixel (per launch);
                           # empty for bare
    label: np.ndarray      # (w, h) int8: 0 blue blob, 1 green blob, -1 —
                           # derived on host from img+visited (the painted
                           # image IS the label map)
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


_warmed = set()
_coop_cache = {}
_streams_ok = None      # None = unprobed; True/False after first warmup
_streams_err = ""


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _tiny_args(entry_format):
    """Two isolated red pixels of an 8x8 white image (labels 0 and 1): no
    discoveries, so a compile launch terminates under any block count and
    exercises the multi-seed init (q_state=[2])."""
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    tiny[5, 5] = (255, 0, 0)
    d_img = cuda.to_device(tiny)
    visited = np.zeros((8, 8), dtype=np.int32)
    visited[1, 1] = 1
    visited[5, 5] = 1
    d_visited = cuda.to_device(visited)
    d_depth = cuda.to_device(np.full((8, 8), -1, dtype=np.int32))
    d_counters = cuda.to_device(np.zeros(NUM_COUNTERS, dtype=np.int64))
    d_queue = cuda.device_array(64, dtype=np.int32)
    seeds = np.array([_pack(1, 1, 0, 8, entry_format),
                      _pack(5, 5, 1, 8, entry_format)], dtype=np.int32)
    d_queue[:2].copy_to_device(seeds)
    d_q = cuda.to_device(np.array([2], dtype=np.int32))
    return d_img, d_visited, d_depth, d_counters, d_queue, d_q


def _warmup(entry_format, bare, connectivity=4, radius=1):
    """JIT-compile (and NVRTC-link) each kernel once, off the clock. The
    first warmup also probes whether this Numba/driver accepts a
    cooperative launch on a non-default stream (mode="streams" needs it);
    the outcome is recorded, never assumed."""
    global _streams_ok, _streams_err
    key = (entry_format, bare, connectivity, radius)
    if key not in _warmed:
        d_img, d_visited, d_depth, d_counters, d_queue, d_q = \
            _tiny_args(entry_format)
        kernel_fn = _KERNELS[key]
        if bare:
            kernel_fn[1, 32](d_img, d_visited, d_depth, d_queue, d_q,
                             d_counters, 2)
        else:
            d_owner = cuda.to_device(np.full((8, 8), -1, dtype=np.int16))
            d_stats = cuda.to_device(np.zeros((1, 2), dtype=np.int64))
            d_trace = cuda.device_array(4, dtype=np.int32)
            kernel_fn[1, 32](
                d_img, d_visited, d_depth, d_owner, d_queue, d_q,
                d_counters, d_stats, d_trace, 2)
        cuda.synchronize()
        _warmed.add(key)
    if _streams_ok is None and not bare:
        try:
            d_img, d_visited, d_depth, d_counters, d_queue, d_q = \
                _tiny_args(entry_format)
            d_owner = cuda.to_device(np.full((8, 8), -1, dtype=np.int16))
            d_stats = cuda.to_device(np.zeros((1, 2), dtype=np.int64))
            d_trace = cuda.device_array(4, dtype=np.int32)
            s = cuda.stream()
            _KERNELS[(entry_format, False, connectivity, radius)][1, 32, s](
                d_img, d_visited, d_depth, d_owner, d_queue, d_q,
                d_counters, d_stats, d_trace, 2)
            s.synchronize()
            _streams_ok = True
        except Exception as exc:  # record, don't crash — streams mode raises
            _streams_ok = False
            _streams_err = f"{type(exc).__name__}: {exc}"


def _coop_max_blocks(kernel_fn, tpb):
    key = (id(kernel_fn), tpb)
    if key not in _coop_cache:
        overload = next(iter(kernel_fn.overloads.values()))
        _coop_cache[key] = overload.max_cooperative_grid_blocks(tpb)
    return _coop_cache[key]


def max_blocks(threads_per_block=256, bare=False, connectivity=4,
               entry_format="lin", radius=1):
    """The largest cooperative grid this GPU can host at threads_per_block
    for ONE launch (what blocks=None resolves to outside streams mode;
    streams mode defaults each of its two launches to half of this).
    Queried per kernel — never assumed equal across formats/twins."""
    _warmup(entry_format, bare, connectivity, radius)
    return _coop_max_blocks(
        _KERNELS[(entry_format, bare, connectivity, radius)],
        threads_per_block)


def flood_fill(img_host, seeds, mode="multisource", threads_per_block=256,
               blocks=None, bare=False, connectivity=4, entry_format="lin",
               radius=1):
    """Flood-fill two disconnected red blobs, blob 0 blue / blob 1 green.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for bad inputs, RuntimeError if the GPU
    cannot host the requested cooperative launch (or a structural tripwire
    fires), and NotImplementedError for mode="streams" if this driver
    rejects cooperative launches on non-default streams (probed at warmup,
    recorded in the error message).
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
                f"seed pixel ({sx}, {sy}) is not red — nothing to fill")
    if seeds[0] == seeds[1]:
        raise ValueError(f"seeds must be two distinct pixels, got {seeds}")
    # 512 cap: ~104 regs/thread — a 1024-thread block could never be resident
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
    if radius == 2 and (connectivity != 8 or entry_format != "lin"):
        raise ValueError(
            "radius=2 requires connectivity=8 and entry_format='lin' (the "
            "guarded ring-2 twins exist only for the lin conn8 kernels)")

    _warmup(entry_format, bare, connectivity, radius)
    if mode == "streams" and not _streams_ok:
        raise NotImplementedError(
            "mode='streams' unavailable: this Numba/driver rejected a "
            f"cooperative launch on a non-default stream ({_streams_err})")

    kernel_fn = _KERNELS[(entry_format, bare, connectivity, radius)]
    coop_max = _coop_max_blocks(kernel_fn, threads_per_block)
    if blocks is None:
        # Streams default: a THIRD of capacity per launch, not half. Probed
        # on this GPU (tpb=64, capacity 192): pairs at 48+48 and 80+80 ran;
        # 88+88 wedged forever at grid.sync — two concurrent cooperative
        # grids anywhere near capacity can interleave block placement so
        # neither grid ever has all its blocks resident. 1/3 each keeps the
        # pair at 2/3 of capacity, well below the observed cliff.
        launch_blocks = max(coop_max // 3, 1) if mode == "streams" else coop_max
    elif blocks > coop_max:
        raise RuntimeError(
            f"blocks={blocks} exceeds this GPU's cooperative-launch capacity "
            f"of {coop_max} blocks at {threads_per_block} threads "
            f"(grid.sync would deadlock)")
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
    d_img = cuda.device_array_like(img_host)
    d_visited = cuda.device_array_like(visited_host)
    d_depth = cuda.device_array_like(depth_host)
    d_queues = [cuda.device_array(width * height, dtype=np.int32)
                for _ in range(n_launches)]
    d_counters = [cuda.device_array(NUM_COUNTERS, dtype=np.int64)
                  for _ in range(n_launches)]
    if instrumented:
        owner_host = np.full((width, height), -1, dtype=np.int16)
        d_owner = cuda.device_array_like(owner_host)
        stats_host = np.zeros((launch_blocks, 2), dtype=np.int64)
        stats_host[:, BS_SMID] = -1
        d_stats = [cuda.device_array_like(stats_host)
                   for _ in range(n_launches)]
        d_traces = [cuda.device_array(trace_capacity, dtype=np.int32)
                    for _ in range(n_launches)]
    cuda.synchronize()
    t_h2d0 = time.perf_counter()

    d_img.copy_to_device(img_host)
    d_visited.copy_to_device(visited_host)
    d_depth.copy_to_device(depth_host)
    zeros = np.zeros(NUM_COUNTERS, dtype=np.int64)
    for d_c in d_counters:
        d_c.copy_to_device(zeros)
    if mode == "multisource":
        packed = np.array(
            [_pack(ax, ay, 0, height, entry_format),
             _pack(bx, by, 1, height, entry_format)], dtype=np.int32)
        d_queues[0][:2].copy_to_device(packed)
        d_q_states = [cuda.to_device(np.array([2], dtype=np.int32))]
    else:
        d_q_states = []
        for i, (sx, sy) in enumerate(seeds):
            packed = np.array([_pack(sx, sy, i, height, entry_format)],
                              dtype=np.int32)
            d_queues[i][:1].copy_to_device(packed)
            d_q_states.append(cuda.to_device(np.array([1], dtype=np.int32)))
    if instrumented:
        d_owner.copy_to_device(owner_host)
        for d_s in d_stats:
            d_s.copy_to_device(stats_host)
    cuda.synchronize()
    t_kernel0 = time.perf_counter()

    # The seed count is a launch-uniform kernel PARAMETER (reading the
    # initial rear from q_state inside the kernel races with the first
    # enqueues and deadlocks grid.sync — see kernels.py module doc).
    n_seeds = 2 if mode == "multisource" else 1

    def _args(i):
        if instrumented:
            return (d_img, d_visited, d_depth, d_owner, d_queues[i],
                    d_q_states[i], d_counters[i], d_stats[i], d_traces[i],
                    n_seeds)
        return (d_img, d_visited, d_depth, d_queues[i], d_q_states[i],
                d_counters[i], n_seeds)

    overlap_ratio = 0.0
    if mode == "sequential":
        per_launch_ms = []
        for i in range(2):
            t0 = time.perf_counter()
            kernel_fn[launch_blocks, threads_per_block](*_args(i))
            cuda.synchronize()
            per_launch_ms.append((time.perf_counter() - t0) * 1000)
        kernel_ms = sum(per_launch_ms)
    elif mode == "streams":
        streams = [cuda.stream(), cuda.stream()]
        events = [(cuda.event(timing=True), cuda.event(timing=True))
                  for _ in range(2)]
        t0 = time.perf_counter()
        for i in range(2):  # issue BOTH before syncing either
            events[i][0].record(streams[i])
            kernel_fn[launch_blocks, threads_per_block, streams[i]](*_args(i))
            events[i][1].record(streams[i])
        for s in streams:
            s.synchronize()
        kernel_ms = (time.perf_counter() - t0) * 1000  # wall around both
        per_launch_ms = [cuda.event_elapsed_time(e0, e1)
                         for e0, e1 in events]
        overlap_ratio = (sum(per_launch_ms) / kernel_ms
                         if kernel_ms > 0 else 0.0)
    else:  # multisource
        t0 = time.perf_counter()
        kernel_fn[launch_blocks, threads_per_block](*_args(0))
        cuda.synchronize()
        kernel_ms = (time.perf_counter() - t0) * 1000
        per_launch_ms = []
    t_d2h0 = time.perf_counter()

    counters = [d_c.copy_to_host() for d_c in d_counters]
    for c in counters:
        if c[OVERFLOW]:
            raise RuntimeError(
                "structural tripwire fired — this indicates a kernel bug: "
                "the queue cannot legitimately overflow")
    img_out = d_img.copy_to_host()
    visited_out = d_visited.copy_to_host()
    depth_out = d_depth.copy_to_host()
    if instrumented:
        owner_out = d_owner.copy_to_host()
        stats = [d_s.copy_to_host() for d_s in d_stats]
        traces = [d_traces[i][:min(int(counters[i][LEVELS]), trace_capacity)]
                  .copy_to_host() for i in range(n_launches)]
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
