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
    kernel_ms: float        # real elapsed compute time: GPU kernel, or the
                             # CPU BFS loop — this is what the browser paces
                             # its replay animation to, so it plays at the
                             # engine's actual speed
    total_ms: float


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

    img = np.full((width, height, 3), 255, dtype=np.uint8)
    img[mask_hw.T] = (255, 0, 0)

    if mode == "gpu":
        result = flood_fill(img, seed_x, seed_y,
                             threads_per_block=THREADS_PER_BLOCK, blocks=None,
                             bare=True, connectivity=CONNECTIVITY)
        depth_xy, levels, filled = result.depth, result.levels, result.filled
        kernel_ms, total_ms = result.kernel_ms, result.total_ms
    else:
        t0 = time.perf_counter()
        _visited, depth_xy, levels, filled = cpu_flood_fill(img, seed_x, seed_y)
        kernel_ms = (time.perf_counter() - t0) * 1000
        total_ms = kernel_ms   # no separate H2D/D2H phases on the CPU path

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
    )


def warmup():
    """Pay the first-call JIT-compile cost of both engines once, off the
    request path: numba CUDA's kernel compile + cooperative-launch-capacity
    query for GPU mode, and @njit(cache=True)'s compile for CPU mode."""
    tiny = np.zeros((8, 8), dtype=bool)
    tiny[4, 4] = True
    run_fill(tiny, mode="gpu")
    run_fill(tiny, mode="cpu")
