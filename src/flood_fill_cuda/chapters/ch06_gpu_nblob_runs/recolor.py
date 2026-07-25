"""Host driver for the run-table recolor.

Public API:
    recolor(img_host, **kw) -> RunRecolorResult        one-shot, allocates
    RunRecolor(width, height, ...)                     reusable buffers
        .pack(img_dev)                                 RGB -> 1 bit/px
        .run(img_dev, contract=...) -> RunRecolorResult

THE headline API change of this chapter: there are two CONTRACTS, and
saying which one a number belongs to is half the result.

    contract="rgb"    the ch05 contract. A (width, height, 3) uint8
                      image is already on the device; every red blob is
                      recolored IN PLACE. Includes the pack pass.
    contract="mask"   the image is already a packed 1-bit mask (the
                      natural form of a binary segmentation), and the
                      paint writes into the RGB image. Excludes pack.

The distinction is not bookkeeping — it is the chapter's finding. At 81
Mpx the RGB image is 243 MB and the mask is 10.15 MB; on a 190 GB/s
device, merely READING the RGB costs 1.28 ms, so no algorithm of any
kind recolors from RGB in under a millisecond on this hardware. The
sub-millisecond claim belongs to the mask contract and is stated as
such, with the RGB number reported beside it every time.

Timing. Every phase is bracketed by CUDA events, so `phase_ms` sums to
`kernel_ms` up to event overhead and the phases are attributable
individually. `kernel_ms` covers ONLY the launches of the selected
contract: no allocation, no host<->device copies, no label map. The
optional outputs (label map, seeds, painted image back on the host) are
timed separately and never folded in — a 324 MB label map would be four
times the pipeline it is meant to describe.

Buffers are allocated once per RunRecolor and reused, so the benchmark
measures the steady state of a pipeline, which is what a real caller
(video frames, tiles of a slide scan) actually pays.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from numba import cuda

from .kernels import (
    pack_kernel, unpack_kernel, count_kernel, row_scan_kernel,
    emit_kernel, merge_rows_kernel, flatten_kernel,
    paint_kernel, label_kernel,
    PALETTE_HOST, N_PALETTE, SCAN_TPB,
    NUM_COUNTERS, RUN_OVERFLOW, N_RUNS, N_BLOBS,
    UNION_ATTEMPTS, UNION_DONE,
)

__all__ = ["recolor", "RunRecolor", "RunRecolorResult", "CONTRACTS",
           "model_bytes_ch06", "MODEL_NOTE"]

CONTRACTS = ("rgb", "mask")

# Grid-stride launch shape for the big kernels. Deliberately a plain,
# far-oversubscribed grid: this chapter has no cooperative launch, so
# nothing constrains it to a residency cap (ch05's whole register saga).
DEFAULT_GRID = (2048, 256)

# gridDim.y cap for the 2D pack/unpack grid — see RunRecolor.__init__
PACK_ROW_BLOCKS = 64

# Per-phase launch shapes, swept on a HOT GPU (see the chapter README's
# tuning table). They differ by up to 1.6x from a single shared grid,
# and the reason is per-phase: count/emit are warp-per-row and want few,
# long-lived warps; merge wants enough parallelism to hide the
# union-find pointer chase but not so much that the atomicMin traffic
# collides (512 blocks beat 2048 by 1.6x); paint is scatter-bound and
# flat from 512 to 2048.
PHASE_BLOCKS = {
    "count": 256,
    "emit": 512,
    "merge": 512,
    "flatten": 512,
    "paint": 1024,
}

# Run-table capacity heuristic when the caller gives none: one slot per
# 16 pixels, with a floor. `input_blobs.png` uses 539k of the 5.06M that
# buys; the floor exists because the ratio is meaningless on small
# images (an 80x80 noise scene has 1,343 runs against a 400-slot
# estimate). The RUN_OVERFLOW tripwire names the exact number needed
# when a scene is finer-grained than the heuristic — the worst case is
# width*ceil(height/2), a 1-px checkerboard.
PIXELS_PER_RUN_SLOT = 16
MIN_RUN_CAPACITY = 8192

MODEL_NOTE = (
    "ch06 model_bytes counts the traffic each phase must move by "
    "construction, not by counter sampling — the pipeline has no "
    "data-dependent probe loops. pack: n*3 read + n/8 write. count: "
    "n/8 read. emit: n/8 read + runs*12 write. merge: runs*(12 read + "
    "union RMW 8 each). flatten: runs*8. paint: runs*12 read + "
    "red_px*3 write. The dominant terms are exactly two: the "
    "full-resolution read (n*3 in the rgb contract, n/8 in the mask "
    "contract) and the paint write (red_px*3). Derived lower-bound "
    "model as in every prior chapter; 32 B sectors inflate, L2 "
    "deflates; compare only against the measured copy peak."
)


def model_bytes_ch06(contract, n_pixels, red_px, n_runs, union_attempts):
    """Algorithmic bytes moved by the timed phases of `contract`."""
    mask_bytes = (n_pixels + 7) // 8
    total = 0
    if contract == "rgb":
        total += n_pixels * 3 + mask_bytes          # pack
    total += mask_bytes                             # count
    total += mask_bytes + n_runs * 16               # emit (+ parent iota)
    total += n_runs * 12 + union_attempts * 8       # merge
    total += n_runs * 8                             # flatten
    total += n_runs * 12 + red_px * 3               # paint
    return total


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


class RunRecolor:
    """Reusable device buffers + the six-launch pipeline.

    width/height are the image dimensions in the repo's img[x, y]
    convention (x is the first axis; y is contiguous in memory, which is
    why runs are y-intervals — a run is a contiguous byte span).
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

        self.mask = cuda.device_array((self.width, self.words_per_row),
                                      dtype=np.uint32)
        self.row_count = cuda.device_array(self.width, dtype=np.int32)
        self.row_off = cuda.device_array(self.width + 1, dtype=np.int32)
        self.counters = cuda.to_device(np.zeros(NUM_COUNTERS, dtype=np.int64))

        self.run_x = cuda.device_array(self.run_capacity, dtype=np.int32)
        self.run_y0 = cuda.device_array(self.run_capacity, dtype=np.int32)
        self.run_y1 = cuda.device_array(self.run_capacity, dtype=np.int32)
        self.parent = cuda.device_array(self.run_capacity, dtype=np.int32)
        self.run_label = cuda.device_array(self.run_capacity, dtype=np.int32)
        self.palette = cuda.to_device(PALETTE_HOST)
        tpb = self.grid[1]
        if tpb % 32:
            raise ValueError(f"threads per block must be a multiple of 32, "
                             f"got {tpb}")
        # 2D pack grid: word along x, row along y. gridDim.y is CAPPED
        # (PACK_ROW_BLOCKS) rather than set to `width`, and the kernel
        # strides rows past it. One block per row looks natural and is a
        # trap: at 9,000 rows it launches 324,000 blocks of 256 threads
        # that each do one row's worth of work, and block dispatch alone
        # then costs more than the memory traffic — the same read
        # measures 3.71 ms that way and 1.12 ms (217 GB/s, the device's
        # read peak) with the cap on.
        self.pack_grid = ((self.words_per_row * 32 + tpb - 1) // tpb,
                          min(self.width, PACK_ROW_BLOCKS)), tpb
        self.phase_grid = {k: (v, tpb) for k, v in PHASE_BLOCKS.items()}

    # ------------------------------------------------------------ pack
    def pack(self, img_dev):
        """RGB -> packed mask, on the current stream (untimed helper)."""
        pack_kernel[self.pack_grid](img_dev, self.mask)

    def unpack_to(self, img_dev):
        """Packed mask -> pure red-on-white RGB (inverse of pack)."""
        unpack_kernel[self.pack_grid](self.mask, img_dev, self.height)

    # ------------------------------------------------------------- run
    def run(self, img_dev, contract="rgb", instrumented=True):
        """Launch the six-kernel pipeline; returns (phase_names, events).

        The RAW entry point: it does not synchronize (the caller owns the
        clock boundaries) and it does not raise on a run-table overflow.
        Callers that are not benchmarks should check
        `counters[RUN_OVERFLOW]` afterwards, or use `recolor()`, which
        checks it and resizes. An overflow is memory-safe — every
        downstream loop is clamped to the capacity — but the answer is
        wrong, so it must not go unread.
        """
        if contract not in CONTRACTS:
            raise ValueError(f"contract must be one of {CONTRACTS}")
        # No counter clear here: row_scan_kernel resets everything the
        # later phases accumulate into, so the pipeline is exactly six
        # launches and not one host round trip.

        names = []
        events = [cuda.event(timing=True)]
        events[0].record()

        if contract == "rgb":
            pack_kernel[self.pack_grid](img_dev, self.mask)
            names.append("pack")
            events.append(cuda.event(timing=True))
            events[-1].record()

        count_kernel[self.phase_grid["count"]](self.mask, self.row_count)
        names.append("count")
        events.append(cuda.event(timing=True))
        events[-1].record()

        row_scan_kernel[1, SCAN_TPB](self.row_count, self.row_off,
                                     self.counters, self.run_capacity)
        names.append("scan")
        events.append(cuda.event(timing=True))
        events[-1].record()

        emit_kernel[self.phase_grid["emit"]](self.mask, self.row_off, self.run_x,
                                   self.run_y0, self.run_y1, self.parent,
                                   self.counters)
        names.append("emit")
        events.append(cuda.event(timing=True))
        events[-1].record()

        merge_rows_kernel[self.phase_grid["merge"]](self.run_x, self.run_y0, self.run_y1,
                                     self.row_off, self.parent, self.counters,
                                     instrumented)
        names.append("merge")
        events.append(cuda.event(timing=True))
        events[-1].record()

        flatten_kernel[self.phase_grid["flatten"]](self.parent, self.run_x, self.run_y0,
                                  self.run_label, self.height, self.counters,
                                  instrumented)
        names.append("flatten")
        events.append(cuda.event(timing=True))
        events[-1].record()

        paint_kernel[self.phase_grid["paint"]](img_dev, self.run_x, self.run_y0,
                                self.run_y1, self.run_label, self.counters,
                                self.palette)
        names.append("paint")
        events.append(cuda.event(timing=True))
        events[-1].record()

        return names, events

    def emit_label_map(self, label_dev):
        label_kernel[self.phase_grid["paint"]](label_dev, self.run_x, self.run_y0,
                                self.run_y1, self.run_label, self.counters)


_warmed = False


def _warmup():
    """JIT-compile every kernel once, off the clock, on a tiny scene."""
    global _warmed
    if _warmed:
        return
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    tiny[5, 5] = (255, 0, 0)
    rc = RunRecolor(8, 8, run_capacity=64, grid=(2, 32))
    dev = cuda.to_device(tiny)
    for contract in CONTRACTS:
        rc.pack(dev)
        rc.run(dev, contract=contract)
    lab = cuda.device_array((8, 8), dtype=np.int32)
    rc.emit_label_map(lab)
    rc.unpack_to(dev)
    cuda.synchronize()
    _warmed = True


def recolor(img_host, contract="rgb", run_capacity=None, grid=DEFAULT_GRID,
            emit_label=False, emit_seeds=True, copy_img=True,
            instrumented=True, engine=None):
    """Discover, label and recolor every red blob — one-shot convenience.

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
    _warmup()

    width, height = img_host.shape[0], img_host.shape[1]
    t_total0 = time.perf_counter()
    rc = engine if engine is not None else RunRecolor(
        width, height, run_capacity=run_capacity, grid=grid)
    if (rc.width, rc.height) != (width, height):
        raise ValueError(
            f"engine is {rc.width}x{rc.height}, image is {width}x{height}")

    # The run table is sized by a heuristic, so the first call on an
    # unusually fine-grained scene can overflow it. The tripwire reports
    # the exact count it needed, so a caller-owned engine raises (its
    # buffers are the caller's), and the one-shot path resizes and reruns
    # from the untouched host image.
    for attempt in (0, 1):
        img_dev = cuda.to_device(img_host)
        if contract == "mask":
            rc.pack(img_dev)      # build the packed input, OFF the clock
        cuda.synchronize()
        t_kernel0 = time.perf_counter()

        names, events = rc.run(img_dev, contract=contract,
                               instrumented=instrumented)
        cuda.synchronize()

        counters = rc.counters.copy_to_host()
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

    phase_ms = {n: cuda.event_elapsed_time(events[i], events[i + 1])
                for i, n in enumerate(names)}
    kernel_ms = cuda.event_elapsed_time(events[0], events[-1])
    n_runs = int(counters[N_RUNS])
    n_blobs = int(counters[N_BLOBS])

    label_ms = 0.0
    label_out = np.zeros((0, 0), dtype=np.int32)
    if emit_label:
        label_dev = cuda.to_device(np.full((width, height), -1,
                                           dtype=np.int32))
        e0, e1 = cuda.event(timing=True), cuda.event(timing=True)
        e0.record()
        rc.emit_label_map(label_dev)
        e1.record()
        e1.synchronize()
        label_ms = cuda.event_elapsed_time(e0, e1)
        label_out = label_dev.copy_to_host()

    t_d2h0 = time.perf_counter()
    img_out = img_dev.copy_to_host() if copy_img else np.zeros((0, 0, 0),
                                                               dtype=np.uint8)
    seeds = []
    if emit_seeds and n_runs:
        parent = rc.parent[:n_runs].copy_to_host()
        rx = rc.run_x[:n_runs].copy_to_host()
        ry = rc.run_y0[:n_runs].copy_to_host()
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
