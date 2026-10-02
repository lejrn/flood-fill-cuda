"""Host driver for the Triton twin of the run-table recolor.

Public API (the Numba driver's, unchanged):
    recolor(img_host, **kw) -> RunRecolorResult        one-shot, allocates
    RunRecolor(width, height, ...)                     reusable buffers
        .pack(img_dev)                                 RGB -> 1 bit/px
        .run(img_dev, contract=...) -> (names, events)

Same contracts ("rgb" includes the pack, "mask" does not), defaults,
validation messages, result dataclass and timing decomposition as
chapters/ch06_gpu_nblob_runs/recolor.py, which explains all of them.
Only the runtime underneath changes:

    cuda.device_array / to_device   cupy.empty / cupy.asarray
    cuda.event(timing=True)         cupy.cuda.Event(), on the null stream
    cuda.event_elapsed_time         cupy.cuda.get_elapsed_time
    cuda.synchronize()              runtime.sync()
    kernel[blocks, tpb](...)        kernel[(blocks,)](..., num_warps=tpb // 32)

Device arrays are CuPy arrays; a Numba device array is accepted too
(viewed zero-copy through __cuda_array_interface__). They must be
C-contiguous: the kernels address them through a raw pointer.

Block size. tl.arange lengths and num_warps must be powers of 2, so the
threads-per-block of `grid` must be 32, 64, ..., 1024. A multiple of 32
that Numba would accept (96, 160, ...) raises ValueError here.

Compiles. Triton compiles one kernel per constexpr combination: the
block size and, for merge and flatten, INSTRUMENTED. `_warmup(tpb)`
compiles all of them for one block size, and `recolor()` calls it for
the block size it is about to use, so no compile lands on the clock.
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from flood_fill_cuda.triton_twins.runtime import sync, t

from .kernels import (
    pack_kernel, unpack_kernel, count_kernel, row_scan_kernel,
    emit_kernel, merge_rows_kernel, flatten_kernel,
    paint_kernel, label_kernel,
    PALETTE_HOST, SCAN_TPB,
    NUM_COUNTERS, RUN_OVERFLOW, N_RUNS, N_BLOBS,
    UNION_ATTEMPTS, UNION_DONE,
)
# The configuration and the bandwidth model are the Numba driver's own
# objects, so the two backends cannot drift apart on either.
from flood_fill_cuda.chapters.ch06_gpu_nblob_runs.recolor import (  # noqa: F401
    CONTRACTS, DEFAULT_GRID, PACK_ROW_BLOCKS, PHASE_BLOCKS,
    PIXELS_PER_RUN_SLOT, MIN_RUN_CAPACITY, MODEL_NOTE, model_bytes_ch06,
)

__all__ = ["recolor", "RunRecolor", "RunRecolorResult", "CONTRACTS",
           "model_bytes_ch06", "MODEL_NOTE"]

SCAN_WARPS = SCAN_TPB // 32


@dataclass
class RunRecolorResult:
    img: np.ndarray        # (w, h, 3) uint8 painted palette[label % 6];
                           # empty unless copy_img=True
    label: np.ndarray      # (w, h) int32 canonical label map, -1 off-blob;
                           # empty unless emit_label=True
    contract: str
    threads_per_block: int
    blocks: int
    # What the GPU discovered
    n_runs: int            # maximal red runs in the image
    n_blobs: int           # surviving union-find roots
    seeds: list            # canonical seed (x, y) per blob, sorted by
                           # label; empty unless emit_seeds=True
    union_attempts: int    # vertical run adjacencies probed
    union_done: int        # successful links == n_runs - n_blobs
    red_px: int            # red pixels (== painted pixels)
    # Timing (ms)
    phase_ms: dict         # per-kernel CUDA-event times
    kernel_ms: float       # the contract's launches, nothing else
    label_ms: float        # label-map materialisation, when asked for
    alloc_ms: float
    h2d_ms: float
    d2h_ms: float
    total_ms: float
    # Derived bandwidth model
    model_bytes: int
    model_gb_s: float
    run_capacity: int = 0
    mask: np.ndarray = field(repr=False, default=None)


def _check_tpb(tpb):
    """Block sizes Triton can express: num_warps = tpb // 32 and the
    tl.arange lengths (tpb lanes, tpb // 32 warps) must be powers of 2."""
    if tpb < 32 or tpb > 1024 or tpb & (tpb - 1):
        raise ValueError(
            f"threads per block must be a power of 2 from 32 to 1024 in "
            f"the Triton twin (num_warps = tpb // 32 and tl.arange lengths "
            f"must be powers of 2), got {tpb}")


def _device(arr):
    """A CuPy view of a device array (CuPy, or anything exposing
    __cuda_array_interface__, such as a Numba device array)."""
    if isinstance(arr, np.ndarray):
        raise TypeError("expected a device array, got a numpy array")
    a = arr if isinstance(arr, cp.ndarray) else cp.asarray(arr)
    if not a.flags.c_contiguous:
        raise ValueError("device arrays must be C-contiguous (the Triton "
                         "kernels address them through a raw pointer)")
    return a


class RunRecolor:
    """Reusable device buffers + the six-launch pipeline.

    width/height are the image dimensions in the repo's img[x, y]
    convention (x is the first axis; y is contiguous in memory).
    """

    def __init__(self, width, height, run_capacity=None, grid=DEFAULT_GRID):
        if width < 1 or height < 1:
            raise ValueError(f"bad shape {width}x{height}")
        n = width * height
        if n >= 2 ** 31:
            raise ValueError(
                "image too large: width*height must be < 2**31 (labels "
                "are int32 linear indices)")
        self.width = int(width)
        self.height = int(height)
        self.words_per_row = (self.height + 31) // 32
        self.grid = tuple(grid)
        self.run_capacity = int(
            run_capacity if run_capacity is not None
            else max(MIN_RUN_CAPACITY, n // PIXELS_PER_RUN_SLOT))

        self.mask = cp.empty((self.width, self.words_per_row),
                             dtype=cp.uint32)
        self.row_count = cp.empty(self.width, dtype=cp.int32)
        self.row_off = cp.empty(self.width + 1, dtype=cp.int32)
        self.counters = cp.zeros(NUM_COUNTERS, dtype=cp.int64)

        self.run_x = cp.empty(self.run_capacity, dtype=cp.int32)
        self.run_y0 = cp.empty(self.run_capacity, dtype=cp.int32)
        self.run_y1 = cp.empty(self.run_capacity, dtype=cp.int32)
        self.parent = cp.empty(self.run_capacity, dtype=cp.int32)
        self.run_label = cp.empty(self.run_capacity, dtype=cp.int32)
        self.palette = cp.asarray(PALETTE_HOST)
        tpb = self.grid[1]
        if tpb % 32:
            raise ValueError(f"threads per block must be a multiple of 32, "
                             f"got {tpb}")
        _check_tpb(tpb)
        self.warps = tpb // 32
        # Same 2D pack grid as Numba: words along axis 0, rows along
        # axis 1 capped at PACK_ROW_BLOCKS and strided.
        self.pack_grid = ((self.words_per_row * 32 + tpb - 1) // tpb,
                          min(self.width, PACK_ROW_BLOCKS)), tpb
        self.phase_grid = {k: (v, tpb) for k, v in PHASE_BLOCKS.items()}

        # Pointer arguments wrapped once: the launch path stays at
        # Triton's per-launch floor.
        self._mask = t(self.mask)
        self._row_count = t(self.row_count)
        self._row_off = t(self.row_off)
        self._counters = t(self.counters)
        self._run_x = t(self.run_x)
        self._run_y0 = t(self.run_y0)
        self._run_y1 = t(self.run_y1)
        self._parent = t(self.parent)
        self._run_label = t(self.run_label)
        self._palette = t(self.palette)
        # The CompiledKernel of each kernel's latest launch, for
        # occupancy.kernel_resources (registers, spills, shared bytes).
        self.compiled = {}

    # ------------------------------------------------------------ pack
    def _pack(self, img):
        (gx, gy), _ = self.pack_grid
        self.compiled["pack"] = pack_kernel[(gx, gy)](
            t(img), self._mask, img.shape[0], img.shape[1],
            self.words_per_row, WPB=self.warps, num_warps=self.warps)

    def pack(self, img_dev):
        """RGB -> packed mask, on the current stream (untimed helper)."""
        self._pack(_device(img_dev))

    def unpack_to(self, img_dev):
        """Packed mask -> pure red-on-white RGB (inverse of pack)."""
        img = _device(img_dev)
        (gx, gy), _ = self.pack_grid
        self.compiled["unpack"] = unpack_kernel[(gx, gy)](
            self._mask, t(img), self.width, self.height,
            self.words_per_row, WPB=self.warps, num_warps=self.warps)

    # ------------------------------------------------------------- run
    def run(self, img_dev, contract="rgb", instrumented=True):
        """Launch the six-kernel pipeline; returns (phase_names, events).

        The RAW entry point: it does not synchronize and it does not
        raise on a run-table overflow (read counters[RUN_OVERFLOW]
        afterwards, or use recolor()). An overflow is memory-safe.
        """
        if contract not in CONTRACTS:
            raise ValueError(f"contract must be one of {CONTRACTS}")
        img = _device(img_dev)
        nw = self.warps
        instrumented = bool(instrumented)

        names = []
        events = [cp.cuda.Event()]
        events[0].record()

        if contract == "rgb":
            self._pack(img)
            names.append("pack")
            events.append(cp.cuda.Event())
            events[-1].record()

        self.compiled["count"] = count_kernel[(self.phase_grid["count"][0],)](
            self._mask, self._row_count, self.width, self.words_per_row,
            WPB=nw, num_warps=nw)
        names.append("count")
        events.append(cp.cuda.Event())
        events[-1].record()

        self.compiled["scan"] = row_scan_kernel[(1,)](
            self._row_count, self._row_off, self._counters, self.width,
            self.run_capacity, BLOCK=SCAN_TPB, num_warps=SCAN_WARPS)
        names.append("scan")
        events.append(cp.cuda.Event())
        events[-1].record()

        self.compiled["emit"] = emit_kernel[(self.phase_grid["emit"][0],)](
            self._mask, self._row_off, self._run_x, self._run_y0,
            self._run_y1, self._parent, self._counters, self.width,
            self.words_per_row, self.run_capacity, WPB=nw, num_warps=nw)
        names.append("emit")
        events.append(cp.cuda.Event())
        events[-1].record()

        self.compiled["merge"] = merge_rows_kernel[
            (self.phase_grid["merge"][0],)](
            self._run_x, self._run_y0, self._run_y1, self._row_off,
            self._parent, self._counters, self.run_capacity, self.width,
            INSTRUMENTED=instrumented, BLOCK=nw * 32, num_warps=nw)
        names.append("merge")
        events.append(cp.cuda.Event())
        events[-1].record()

        self.compiled["flatten"] = flatten_kernel[
            (self.phase_grid["flatten"][0],)](
            self._parent, self._run_x, self._run_y0, self._run_label,
            self.height, self._counters,
            INSTRUMENTED=instrumented, BLOCK=nw * 32, num_warps=nw)
        names.append("flatten")
        events.append(cp.cuda.Event())
        events[-1].record()

        self.compiled["paint"] = paint_kernel[(self.phase_grid["paint"][0],)](
            t(img), self._run_x, self._run_y0, self._run_y1,
            self._run_label, self._counters, self._palette, img.shape[1],
            WPB=nw, num_warps=nw)
        names.append("paint")
        events.append(cp.cuda.Event())
        events[-1].record()

        return names, events

    def emit_label_map(self, label_dev):
        label = _device(label_dev)
        self.compiled["label"] = label_kernel[(self.phase_grid["paint"][0],)](
            t(label), self._run_x, self._run_y0, self._run_y1,
            self._run_label, self._counters, label.shape[1],
            WPB=self.warps, num_warps=self.warps)


_warmed = set()


def _warmup(tpb=DEFAULT_GRID[1]):
    """Compile every kernel once for block size `tpb`, off the clock, on
    a tiny scene: both contracts, both INSTRUMENTED variants, the label
    map and unpack. (Numba compiles once for every block size; Triton's
    block size is a compile-time constant, hence the argument.)"""
    if tpb in _warmed:
        return
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    tiny[5, 5] = (255, 0, 0)
    rc = RunRecolor(8, 8, run_capacity=64, grid=(2, tpb))
    dev = cp.asarray(tiny)
    for contract in CONTRACTS:
        for instrumented in (True, False):
            rc.pack(dev)
            rc.run(dev, contract=contract, instrumented=instrumented)
    lab = cp.empty((8, 8), dtype=cp.int32)
    rc.emit_label_map(lab)
    rc.unpack_to(dev)
    sync()
    _warmed.add(tpb)


def recolor(img_host, contract="rgb", run_capacity=None, grid=DEFAULT_GRID,
            emit_label=False, emit_seeds=True, copy_img=True,
            instrumented=True, engine=None):
    """Discover, label and recolor every red blob: one-shot convenience.

    img_host: (width, height, 3) uint8, not modified (a painted copy is
    returned when copy_img). Blank images are valid (n_blobs=0).

    contract: "rgb" times the pack pass too; "mask" assumes a packed
    binary input and times only the run pipeline (the pack still runs,
    off the clock, to build it).

    engine: reuse an existing RunRecolor (must match the shape); None
    builds one. Reuse is how the benchmark measures steady state.
    """
    if (img_host.ndim != 3 or img_host.shape[2] != 3
            or img_host.dtype != np.uint8):
        raise ValueError("img must be a (width, height, 3) uint8 array")
    if contract not in CONTRACTS:
        raise ValueError(f"contract must be one of {CONTRACTS}")
    _warmup((engine.grid if engine is not None else tuple(grid))[1])

    width, height = img_host.shape[0], img_host.shape[1]
    t_total0 = time.perf_counter()
    rc = engine if engine is not None else RunRecolor(
        width, height, run_capacity=run_capacity, grid=grid)
    if (rc.width, rc.height) != (width, height):
        raise ValueError(
            f"engine is {rc.width}x{rc.height}, image is {width}x{height}")

    # Same overflow policy as the Numba driver: a caller-owned engine
    # raises, the one-shot path resizes once and reruns from the
    # untouched host image.
    for attempt in (0, 1):
        img_dev = cp.asarray(np.ascontiguousarray(img_host))
        if contract == "mask":
            rc.pack(img_dev)      # build the packed input, OFF the clock
        sync()
        t_kernel0 = time.perf_counter()

        names, events = rc.run(img_dev, contract=contract,
                               instrumented=instrumented)
        sync()

        counters = rc.counters.get()
        if not counters[RUN_OVERFLOW]:
            break
        needed = int(counters[N_RUNS])
        if engine is not None or attempt:
            raise RuntimeError(
                f"run table overflowed capacity {rc.run_capacity}: this "
                f"scene has {needed} runs. Rebuild with "
                f"run_capacity={needed} (worst case is "
                f"width*ceil(height/2) = {width * ((height + 1) // 2)}).")
        rc = RunRecolor(width, height, run_capacity=needed + needed // 8,
                        grid=grid)

    phase_ms = {n: cp.cuda.get_elapsed_time(events[i], events[i + 1])
                for i, n in enumerate(names)}
    kernel_ms = cp.cuda.get_elapsed_time(events[0], events[-1])
    n_runs = int(counters[N_RUNS])
    n_blobs = int(counters[N_BLOBS])

    label_ms = 0.0
    label_out = np.zeros((0, 0), dtype=np.int32)
    if emit_label:
        label_dev = cp.asarray(np.full((width, height), -1, dtype=np.int32))
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        rc.emit_label_map(label_dev)
        e1.record()
        e1.synchronize()
        label_ms = cp.cuda.get_elapsed_time(e0, e1)
        label_out = label_dev.get()

    t_d2h0 = time.perf_counter()
    img_out = img_dev.get() if copy_img else np.zeros((0, 0, 0),
                                                      dtype=np.uint8)
    seeds = []
    if emit_seeds and n_runs:
        parent = rc.parent[:n_runs].get()
        rx = rc.run_x[:n_runs].get()
        ry = rc.run_y0[:n_runs].get()
        roots = np.flatnonzero(parent == np.arange(n_runs, dtype=np.int32))
        order = np.argsort(rx[roots].astype(np.int64) * height + ry[roots])
        seeds = [(int(rx[roots[i]]), int(ry[roots[i]])) for i in order]
        if not instrumented:
            n_blobs = int(roots.size)
    t_end = time.perf_counter()

    red_px = int(np.count_nonzero(
        (img_host[..., 0] == 255) & (img_host[..., 1] == 0)
        & (img_host[..., 2] == 0)))
    mbytes = model_bytes_ch06(contract, width * height, red_px, n_runs,
                              int(counters[UNION_ATTEMPTS]))

    return RunRecolorResult(
        img=img_out,
        label=label_out,
        contract=contract,
        threads_per_block=rc.grid[1],
        blocks=rc.grid[0],
        n_runs=n_runs,
        n_blobs=n_blobs,
        seeds=seeds,
        union_attempts=int(counters[UNION_ATTEMPTS]),
        union_done=int(counters[UNION_DONE]),
        red_px=red_px,
        phase_ms=phase_ms,
        kernel_ms=kernel_ms,
        label_ms=label_ms,
        alloc_ms=(t_kernel0 - t_total0) * 1000,
        h2d_ms=0.0,                     # folded into alloc, as in ch05
        d2h_ms=(t_end - t_d2h0) * 1000,
        total_ms=(t_end - t_total0) * 1000,
        model_bytes=mbytes,
        model_gb_s=mbytes / (kernel_ms * 1e6) if kernel_ms > 0 else 0.0,
        run_capacity=rc.run_capacity,
    )
