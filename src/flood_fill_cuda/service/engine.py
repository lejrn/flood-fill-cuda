"""GPU/CPU engine wrapper around ch03's single-seed flood fill and its ch01
CPU oracle, for the paint service.

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

MAX_PIXELS = 4_000_000       # generous headroom over any realistic stroke bbox
THREADS_PER_BLOCK = 256
CONNECTIVITY = 4              # diamond wavefronts — the more legible animation
DEPTH_CLAMP = 65535           # uint16 ceiling; only a >65k-level stroke clips
MODES = ("cpu", "gpu")

AMPLIFY_FACTOR = 50           # target pixel-count multiplier for the timing run
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


def warmup():
    """Pay the first-call JIT-compile cost of both engines once, off the
    request path: numba CUDA's kernel compile + cooperative-launch-capacity
    query for GPU mode, and @njit(cache=True)'s compile for CPU mode."""
    tiny = np.zeros((8, 8), dtype=bool)
    tiny[4, 4] = True
    run_fill(tiny, mode="gpu")
    run_fill(tiny, mode="cpu")
