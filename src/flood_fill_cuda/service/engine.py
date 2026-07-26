"""GPU/CPU engine wrapper for the paint service.

Two jobs, two kernels:

- `run_fill` — ch03's single-seed cooperative flood fill and its ch01 CPU
  oracle. Seeded, one blob, returns a real BFS depth map. This is what the
  paint page's CPU/GPU toggle compares.
- `discover` — ch06's run-table connected components. Seedless, every blob
  at once, no depth map because there is no BFS. Backs /api/scan (DEEP
  FIELD, and the paint page's RUNS mode).

Three things a plain call to `ch03.flood_fill.flood_fill` doesn't give a web
handler for free, all handled here:

- **Seed placement.** The browser sends a painted mask (screen-orientation,
  (height, width) bool) plus where the pointer was released; the kernel
  needs one red pixel to start from, so that release point is snapped to
  the nearest actual mask pixel (an off-stroke release, or a request that
  omits it, still needs a valid seed — the mask's center of mass is the
  fallback, similarly snapped for rings/"C" strokes whose centroid falls
  in the hole).
- **Orientation + encoding.** The kernel is (width, height, 3) uint8 indexed
  img[x, y] — the transpose of the mask's (height, width) row-major layout
  (see ch04/benchmarks/wavefront.py's to_image, which transposes the same
  way on the way out). depth comes back int32 (width, height), -1 unreached;
  the wire format the browser wants is uint16 (height, width), 0 = not part
  of the blob, else depth+1 (0 is reserved so "unfilled" and "filled at
  level 0" are distinguishable in an unsigned encoding).
- **CPU mode.** ch01's `cpu_flood_fill` is the exact 4-connectivity oracle
  ch03's kernel is tested against (bit-identical depth maps — see
  ch03/test_correctness.py's `test_matches_cpu_reference`), so routing a
  request through it instead of the GPU kernel gives a true side-by-side
  timing comparison on the same BFS, not just a different algorithm.
- **Amplified-scale timing.** A Full HD brush stroke tops out at maybe a
  few hundred thousand pixels — nowhere near the multi-megapixel scale
  where the GPU's cooperative-launch overhead actually pays for itself
  (see ch03/ch04's own benchmarks). So every fill runs *twice*: once on
  the mask exactly as painted (this is what becomes the response's
  `depth_u16`/`levels`/`filled` — unchanged visual size, sent to the
  browser as-is), and once on the same blob shape nearest-neighbor
  upscaled to ~`AMPLIFY_FACTOR`x the pixel count (capped at
  `MAX_AMPLIFIED_PIXELS`), purely to get a realistic `kernel_ms`/
  `total_ms`/pixel-count at the scale that actually favors the GPU. The
  visible shape never changes size; only the reported/paced timing does.

os.environ.setdefault must run before numba is imported anywhere in this
process — see ch03/kernels.py's identical line. It is set here too,
defensively, in case this module is ever imported before that one.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import time
from dataclasses import dataclass

import numpy as np

from ..chapters.ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
from ..chapters.ch03_gpu_1blob_nblock.flood_fill import flood_fill
from ..chapters.ch05_gpu_nblob_nblock.cpu_oracle import cpu_fill_canonical
from ..chapters.ch06_gpu_nblob_runs.recolor import RunRecolor
from ..chapters.ch06_gpu_nblob_runs.kernels import (
    N_RUNS, N_BLOBS, UNION_DONE, RUN_OVERFLOW)

MAX_PIXELS = 4_000_000       # generous headroom over any realistic stroke bbox
THREADS_PER_BLOCK = 256
CONNECTIVITY = 4              # diamond wavefronts — the more legible animation
DEPTH_CLAMP = 65535           # uint16 ceiling; only a >65k-level stroke clips
MAX_TRACKS = 65_535           # dense track ids are uint16; 0 = background
MODES = ("cpu", "gpu")

# Sweep buckets for the /api/scan reveal. NOT a timeline — see
# `_sweep_field`. Enough steps to look continuous, few enough that the
# browser's per-bucket index lists stay cheap to build.
SWEEP_STEPS = 240

AMPLIFY_FACTOR = 250          # target pixel-count multiplier for the timing run
# Separate, higher ceiling than MAX_PIXELS (which guards the *input* upload)
# -- this bounds the synthetic amplified mask, landing in the same
# multi-megapixel range ch03/ch04's own benchmarks already validated.
MAX_AMPLIFIED_PIXELS = 16_000_000


class MaskTooLargeError(ValueError):
    """Raised instead of plain ValueError so the API layer can map it to
    413 instead of 400."""


@dataclass
class FillOutcome:
    depth_u16: np.ndarray   # (height, width) uint16, 0=unfilled else depth+1
    width: int
    height: int
    levels: int
    filled: int
    seed_x: int
    seed_y: int
    mode: str
    kernel_ms: float        # real elapsed compute time of the AMPLIFIED run
                             # (see module docstring) -- what the browser
                             # paces its replay animation to
    total_ms: float
    amplified_filled: int   # pixel count actually reached by the amplified
                             # run -- the honest "at scale" number to show,
                             # distinct from `filled` (the small/real mask)


def _resolve_seed(mask_hw, x=None, y=None):
    """Nearest mask pixel to (x, y), in (x, y) = (col, row) terms. Defaults
    to the mask's center of mass when x/y are omitted. Always returns an
    actual mask pixel — including when the requested point isn't one (the
    pointer released just outside the stroke, or a ring/C-shape's centroid
    falling in the hole)."""
    ys_arr, xs_arr = np.nonzero(mask_hw)
    if x is None or y is None:
        x, y = xs_arr.mean(), ys_arr.mean()
    d2 = (xs_arr - x) ** 2 + (ys_arr - y) ** 2
    k = int(np.argmin(d2))
    return int(xs_arr[k]), int(ys_arr[k])


def _amplify_mask(mask_hw, factor, max_pixels):
    """Nearest-neighbor upscale of a boolean mask to ~factor x its pixel
    count, capped at max_pixels. Returns (big_mask_hw, scale); scale is
    1.0 (mask returned unchanged) if the mask is already at/over the cap.
    Index-mapping nearest neighbor keeps the blob's shape crisp and works
    exactly for non-integer scales, unlike a naive per-axis np.repeat."""
    height, width = mask_hw.shape
    current = height * width
    capped = min(current * factor, max_pixels)
    if capped <= current:
        return mask_hw, 1.0
    scale = (capped / current) ** 0.5
    new_h = max(height, round(height * scale))
    new_w = max(width, round(width * scale))
    row_idx = np.minimum((np.arange(new_h) / scale).astype(np.int64), height - 1)
    col_idx = np.minimum((np.arange(new_w) / scale).astype(np.int64), width - 1)
    big_mask = mask_hw[row_idx][:, col_idx]
    return big_mask, scale


def _run_engine(img, seed_x, seed_y, mode):
    """Dispatch one BFS run to the GPU kernel or the CPU oracle. Returns
    (depth_xy, levels, filled, kernel_ms, total_ms)."""
    if mode == "gpu":
        result = flood_fill(img, seed_x, seed_y,
                             threads_per_block=THREADS_PER_BLOCK, blocks=None,
                             bare=True, connectivity=CONNECTIVITY)
        return (result.depth, result.levels, result.filled,
                result.kernel_ms, result.total_ms)
    t0 = time.perf_counter()
    _visited, depth_xy, levels, filled = cpu_flood_fill(img, seed_x, seed_y)
    kernel_ms = (time.perf_counter() - t0) * 1000
    return depth_xy, levels, filled, kernel_ms, kernel_ms


def _build_image(mask_hw):
    """(height,width) mask -> (width,height,3) uint8 kernel image; pure
    red on white, the transpose the kernel expects (img[x, y])."""
    height, width = mask_hw.shape
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    img[mask_hw.T] = (255, 0, 0)
    return img


def run_fill(mask_hw, mode="gpu", seed_x=None, seed_y=None):
    """Flood-fill a painted mask, on the GPU kernel or the CPU oracle.

    mask_hw: (height, width) bool — True where the user painted.
    mode: "gpu" (ch03's cooperative kernel) or "cpu" (ch01's sequential
    njit oracle) — same BFS, same connectivity, bit-identical depth maps
    on a given mask; only the engine (and therefore the timing) differs.
    seed_x, seed_y: where the fill starts from — typically where the user
    released the pointer. Snapped to the nearest actual mask pixel, so an
    off-stroke release still works. Both default to the mask's center of
    mass when omitted.
    Raises ValueError (empty mask, bad mode) or MaskTooLargeError (over
    MAX_PIXELS); both are the caller's cue to answer 400 / 413. A
    kernel-side RuntimeError (GPU cooperative-capacity exhaustion,
    tripwire) propagates unchanged — the caller's cue to answer 500.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mask_hw.ndim != 2 or mask_hw.dtype != bool:
        raise ValueError("mask must be a 2-D boolean array")
    height, width = mask_hw.shape
    if width * height > MAX_PIXELS:
        raise MaskTooLargeError(
            f"mask is {width}x{height} ({width * height} px), over the "
            f"{MAX_PIXELS} px cap")
    if not mask_hw.any():
        raise ValueError("mask is empty — nothing to fill")

    seed_x, seed_y = _resolve_seed(mask_hw, seed_x, seed_y)

    img = _build_image(mask_hw)
    depth_xy, levels, filled, _small_kernel_ms, _small_total_ms = _run_engine(
        img, seed_x, seed_y, mode)

    # Amplified-scale timing run (see module docstring): same blob shape,
    # upscaled, purely so kernel_ms/total_ms/pixel-count are honest at the
    # scale where the GPU's advantage is real. The visible depth map above
    # (small, real) is what's actually returned to the browser.
    big_mask, scale = _amplify_mask(mask_hw, AMPLIFY_FACTOR, MAX_AMPLIFIED_PIXELS)
    if scale > 1.0:
        big_seed_x, big_seed_y = _resolve_seed(
            big_mask, seed_x * scale, seed_y * scale)
        big_img = _build_image(big_mask)
        _big_depth, _big_levels, amplified_filled, kernel_ms, total_ms = (
            _run_engine(big_img, big_seed_x, big_seed_y, mode))
    else:
        amplified_filled, kernel_ms, total_ms = filled, _small_kernel_ms, _small_total_ms

    depth_hw = depth_xy.T   # (width,height) -> (height,width)
    encoded = np.where(depth_hw < 0, 0, np.minimum(depth_hw + 1, DEPTH_CLAMP))
    depth_u16 = np.ascontiguousarray(encoded.astype('<u2'))

    return FillOutcome(
        depth_u16=depth_u16,
        width=width,
        height=height,
        levels=levels,
        filled=filled,
        seed_x=seed_x,
        seed_y=seed_y,
        mode=mode,
        kernel_ms=kernel_ms,
        total_ms=total_ms,
        amplified_filled=amplified_filled,
    )


@dataclass
class ScanOutcome:
    """One seedless discovery pass over the whole canvas (ch06)."""
    sweep_u16: np.ndarray   # (height, width) uint16, 0=background else
                             # 1..steps — SCAN ORDER, not a timeline
    track_u16: np.ndarray   # (height, width) uint16, 0=background else 1..N
    seeds: np.ndarray       # (n_blobs, 2) int32 (x, y), row i = track i+1
    width: int
    height: int
    steps: int
    n_blobs: int
    filled: int
    n_runs: int
    unions: int
    kernel_ms: float
    total_ms: float
    phase_ms: dict
    amplified_filled: int   # 0 when no amplified run was asked for
    amplified_runs: int
    amplified_kernel_ms: float
    cold: bool              # first launch after an idle GPU — the clock
                             # had not spun up, so kernel_ms is an
                             # over-estimate. See COLD_AFTER_S.


def _dense_tracks(label_xy, height):
    """Canonical labels (huge linear indices, -1 background) -> dense
    uint16 track ids 1..N in canonical order, plus each track's seed:
    the root IS the lex-min pixel's linear index, so it decodes straight
    into one red pixel per blob — ordering matches the ids by
    construction."""
    roots = np.unique(label_xy[label_xy >= 0])
    if len(roots) > MAX_TRACKS:
        raise MaskTooLargeError(
            f"{len(roots)} blobs exceed the {MAX_TRACKS}-track wire cap")
    track_xy = np.zeros(label_xy.shape, dtype=np.int64)
    mask = label_xy >= 0
    track_xy[mask] = np.searchsorted(roots, label_xy[mask]) + 1
    seeds = np.stack([roots // height, roots % height],
                     axis=1).astype(np.int32) if len(roots) else \
        np.empty((0, 2), dtype=np.int32)
    return track_xy, seeds


# ch06 keeps its buffers between calls by design (see its RunRecolor
# docstring: the benchmark measures the steady state of a pipeline, which
# is what a real caller pays). DEEP FIELD rescans the same-sized field
# over and over, so caching one engine per canvas size turns every scan
# after the first into pure kernel time. Only ever touched from the
# single-threaded GPU executor, so it needs no lock.
_ch06_engines = {}
_CH06_CACHE_MAX = 4

# Seconds of GPU idleness after which the next launch is reported COLD.
# Chapter 6's measurement lesson, arriving in production: this GPU drops
# to ~700 MHz (of 3105) when nothing is running and does not climb for
# kernels that finish in a millisecond. Measured through this very
# endpoint, the first scan after a pause reads 37 ms for the same
# amplified run that reads 1.1 ms back-to-back. Rather than hide that
# behind a keep-warm spin loop — burning the GPU to flatter a number —
# the outcome carries a `cold` flag and the UI says so.
COLD_AFTER_S = 5.0
_last_gpu_call = 0.0


def _ch06_engine(width, height, n_runs_hint):
    key = (width, height)
    engine = _ch06_engines.get(key)
    if engine is not None and engine.run_capacity >= n_runs_hint:
        return engine
    if len(_ch06_engines) >= _CH06_CACHE_MAX and key not in _ch06_engines:
        _ch06_engines.pop(next(iter(_ch06_engines)))
    engine = RunRecolor(width, height,
                        run_capacity=max(8192, int(n_runs_hint * 1.25)))
    _ch06_engines[key] = engine
    return engine


def _host_run_count(mask_hw):
    """Runs the ch06 table will need, counted on the host with numpy.

    ch06 sizes its run table from a heuristic (one slot per 16 px) and
    trips a tripwire if a scene is finer-grained than that — which a
    painted canvas of thin strokes easily is. Counting first turns a
    would-be 500 into a correctly-sized buffer. Runs lie along the
    kernel's contiguous axis, which is the mask's COLUMN direction (the
    kernel image is the mask transposed).
    """
    m = mask_hw.astype(np.int8)
    return int((np.diff(m, axis=0, prepend=0) == 1).sum())


def _sweep_field(label_xy, width, height, steps):
    """Per-pixel sweep bucket, 0 = background, else 1..steps.

    HONEST LABEL: this is the order the kernel SCANS in, not an order it
    happens in. ch05's depth map was a real BFS timeline — level 3 truly
    came after level 2. ch06 has no temporal structure at all: every row
    is counted, emitted and merged at once, and the whole thing is over
    in about a millisecond. So the reveal animates the one ordering the
    algorithm does have — its row-major scan — and nothing here should be
    read as "this pixel was found later than that one".

    The kernel image is the mask transposed, so a kernel row is a canvas
    COLUMN: the sweep runs left to right across the picture.
    """
    lit = (label_xy >= 0).T                     # -> (height, width)
    col = np.arange(width, dtype=np.int64)
    bucket = (col * steps) // max(width, 1) + 1   # 1..steps, per column
    field = np.where(lit, bucket[None, :], 0)
    return np.ascontiguousarray(field.astype('<u2'))


def _run_ch06(mask_hw):
    """One ch06 pass over a mask. Returns (label_xy, counters, kernel_ms,
    total_ms, phase_ms)."""
    from numba import cuda
    global _last_gpu_call
    cold = (time.perf_counter() - _last_gpu_call) > COLD_AFTER_S
    height, width = mask_hw.shape
    img = _build_image(mask_hw)
    engine = _ch06_engine(width, height, _host_run_count(mask_hw))

    t0 = time.perf_counter()
    dev = cuda.to_device(img)
    names, events = engine.run(dev, contract="rgb")
    cuda.synchronize()
    kernel_ms = cuda.event_elapsed_time(events[0], events[-1])
    counters = engine.counters.copy_to_host()
    if counters[RUN_OVERFLOW]:
        raise RuntimeError(
            f"ch06 run table overflowed ({counters[N_RUNS]} runs) — the "
            f"host pre-count under-estimated it")
    label_dev = cuda.to_device(np.full((width, height), -1, dtype=np.int32))
    engine.emit_label_map(label_dev)
    cuda.synchronize()
    label_xy = label_dev.copy_to_host()
    total_ms = (time.perf_counter() - t0) * 1000.0
    phase_ms = {n: cuda.event_elapsed_time(events[i], events[i + 1])
                for i, n in enumerate(names)}
    _last_gpu_call = time.perf_counter()
    return label_xy, counters, kernel_ms, total_ms, phase_ms, cold


def discover(mask_hw, amplify=False):
    """Seedless multi-blob discovery over the whole canvas — ch06.

    No seeds, no mode: the GPU decomposes the canvas into runs, unions
    the ones that touch, and hands back one canonical label per blob.
    An empty mask is VALID (n_blobs=0), the same contract ch05 set.

    What changed from ch05 (which this replaced): there is no `depth`
    map and no `levels`, because there is no BFS. The reveal field is a
    scan-order sweep instead — see `_sweep_field`. Canonical labels are
    unchanged, so `track` ids and `seeds` mean exactly what they did.

    amplify: also run the same shape upscaled ~AMPLIFY_FACTOR x, purely
    to report an honest kernel time at a scale that isn't launch-bound.
    A painted stroke is ~100k px, where ch06's six launches dominate; the
    paint page's RUNS mode asks for this so its number is comparable with
    the CPU/GPU modes, which have always reported the amplified run.
    """
    if mask_hw.ndim != 2 or mask_hw.dtype != bool:
        raise ValueError("mask must be a 2-D boolean array")
    height, width = mask_hw.shape
    if width * height > MAX_PIXELS:
        raise MaskTooLargeError(
            f"canvas is {width}x{height} ({width * height} px), over the "
            f"{MAX_PIXELS} px cap")

    label_xy, counters, kernel_ms, total_ms, phase_ms, cold = \
        _run_ch06(mask_hw)

    track_xy, seeds = _dense_tracks(label_xy, height)
    steps = max(1, min(SWEEP_STEPS, width))
    sweep_u16 = _sweep_field(label_xy, width, height, steps)
    track_u16 = np.ascontiguousarray(track_xy.T.astype('<u2'))

    amplified_filled = amplified_runs = 0
    amplified_kernel_ms = 0.0
    if amplify:
        big_mask, scale = _amplify_mask(mask_hw, AMPLIFY_FACTOR,
                                        MAX_AMPLIFIED_PIXELS)
        if scale > 1.0:
            (_lbl, big_counters, amplified_kernel_ms,
             _t, _p, _c) = _run_ch06(big_mask)
            amplified_filled = int(big_mask.sum())
            amplified_runs = int(big_counters[N_RUNS])

    return ScanOutcome(
        sweep_u16=sweep_u16, track_u16=track_u16, seeds=seeds,
        width=width, height=height, steps=steps,
        n_blobs=int(counters[N_BLOBS]),
        filled=int((label_xy >= 0).sum()),
        n_runs=int(counters[N_RUNS]),
        unions=int(counters[UNION_DONE]),
        kernel_ms=float(kernel_ms), total_ms=float(total_ms),
        phase_ms=phase_ms,
        amplified_filled=amplified_filled,
        amplified_runs=amplified_runs,
        amplified_kernel_ms=float(amplified_kernel_ms),
        cold=bool(cold),
    )


def njit_reference_ms(mask_hw):
    """The honest CPU bar for the race: the @njit seedless reference
    (CCL + canonical fill) on the same canvas, wall-clocked. Compiled at
    warmup, so this is pure run time."""
    img = _build_image(mask_hw)
    t0 = time.perf_counter()
    cpu_fill_canonical(img)
    return (time.perf_counter() - t0) * 1000.0


def warmup():
    """Pay the first-call JIT-compile cost of every engine once, off the
    request path: numba CUDA's kernel compile + cooperative-launch-capacity
    query for GPU mode, and @njit(cache=True)'s compile for CPU mode."""
    tiny = np.zeros((8, 8), dtype=bool)
    tiny[4, 4] = True
    run_fill(tiny, mode="gpu")
    run_fill(tiny, mode="cpu")
    discover(tiny, amplify=True)
    njit_reference_ms(tiny)
