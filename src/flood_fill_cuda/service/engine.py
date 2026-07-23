"""GPU engine wrapper around ch03's single-seed flood fill, for the paint
service.

Two things a plain call to `ch03.flood_fill.flood_fill` doesn't give a web
handler for free, both handled here:

- **Seed placement.** The browser doesn't pick a seed — it sends a painted
  mask (screen-orientation, (height, width) bool). The kernel needs one red
  pixel to start from, so the seed is the mask's center of mass, snapped to
  the nearest actual mask pixel (a ring/"C" stroke's centroid can fall in
  the hole).
- **Orientation + encoding.** The kernel is (width, height, 3) uint8 indexed
  img[x, y] — the transpose of the mask's (height, width) row-major layout
  (see ch04/benchmarks/wavefront.py's to_image, which transposes the same
  way on the way out). depth comes back int32 (width, height), -1 unreached;
  the wire format the browser wants is uint16 (height, width), 0 = not part
  of the blob, else depth+1 (0 is reserved so "unfilled" and "filled at
  level 0" are distinguishable in an unsigned encoding).

os.environ.setdefault must run before numba is imported anywhere in this
process — see ch03/kernels.py's identical line. It is set here too,
defensively, in case this module is ever imported before that one.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

from dataclasses import dataclass

import numpy as np

from ..chapters.ch03_gpu_1blob_nblock.flood_fill import flood_fill

MAX_PIXELS = 4_000_000       # generous headroom over any realistic stroke bbox
THREADS_PER_BLOCK = 256
CONNECTIVITY = 4              # diamond wavefronts — the more legible animation
DEPTH_CLAMP = 65535           # uint16 ceiling; only a >65k-level stroke clips


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
    kernel_ms: float
    total_ms: float


def _centroid_seed(mask_hw):
    """Nearest mask pixel to the mask's center of mass, in (x, y) = (col,
    row) terms. Always returns an actual mask pixel — including when the
    centroid itself isn't one (rings, C-shapes, concave strokes)."""
    ys, xs = np.nonzero(mask_hw)
    cx, cy = xs.mean(), ys.mean()
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    k = int(np.argmin(d2))
    return int(xs[k]), int(ys[k])


def run_fill(mask_hw):
    """Flood-fill a painted mask from its center of mass.

    mask_hw: (height, width) bool — True where the user painted.
    Raises ValueError (empty mask) or MaskTooLargeError (over MAX_PIXELS);
    both are the caller's cue to answer 400 / 413. A kernel-side
    RuntimeError (cooperative-capacity exhaustion, tripwire) propagates
    unchanged — the caller's cue to answer 500.
    """
    if mask_hw.ndim != 2 or mask_hw.dtype != bool:
        raise ValueError("mask must be a 2-D boolean array")
    height, width = mask_hw.shape
    if width * height > MAX_PIXELS:
        raise MaskTooLargeError(
            f"mask is {width}x{height} ({width * height} px), over the "
            f"{MAX_PIXELS} px cap")
    if not mask_hw.any():
        raise ValueError("mask is empty — nothing to fill")

    seed_x, seed_y = _centroid_seed(mask_hw)

    img = np.full((width, height, 3), 255, dtype=np.uint8)
    img[mask_hw.T] = (255, 0, 0)

    result = flood_fill(img, seed_x, seed_y,
                         threads_per_block=THREADS_PER_BLOCK, blocks=None,
                         bare=True, connectivity=CONNECTIVITY)

    depth_hw = result.depth.T   # (width,height) -> (height,width)
    encoded = np.where(depth_hw < 0, 0, np.minimum(depth_hw + 1, DEPTH_CLAMP))
    depth_u16 = np.ascontiguousarray(encoded.astype('<u2'))

    return FillOutcome(
        depth_u16=depth_u16,
        width=width,
        height=height,
        levels=result.levels,
        filled=result.filled,
        seed_x=seed_x,
        seed_y=seed_y,
        kernel_ms=result.kernel_ms,
        total_ms=result.total_ms,
    )


def warmup():
    """Pay the first-call JIT + cooperative-launch-capacity query cost once,
    off the request path. Exercises the exact kernel variant (bare, conn=4)
    real requests use."""
    tiny = np.zeros((8, 8), dtype=bool)
    tiny[4, 4] = True
    run_fill(tiny)
