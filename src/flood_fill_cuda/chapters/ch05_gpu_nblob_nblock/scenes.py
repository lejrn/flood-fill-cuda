"""Seedless test scenes for the seed-discovery stage.

Every prior chapter's scene hands back the seed(s) alongside the image;
this chapter's premise is that nobody knows the seeds — so the builders
return (img, n_blobs) and nothing else. n_blobs is the ground-truth
component count (8-connectivity), either by construction or, for the
random scene, computed by the CPU oracle at build time.

The two-blob builders delegate to ch04's generators (same geometry, same
gap >= 2 contract) and drop the seeds; the single-blob builders delegate
to shared/scenes.py the same way. New to this chapter:

- u_shape_scene: ONE blob with TWO lex-corner candidates — the smallest
  scene where a local scan cannot tell it found the same blob twice.
- comb_scene: ONE blob with `teeth` candidates (default 80 > 64) — proof
  that the retired 64-label queue format really is gone.
- blob_grid_scene: nx*ny disjoint squares — the N-blob workhorse.
- random_blobs_scene: sub-percolation red noise — hundreds of components
  of every shape at once.

All images are (width, height, 3) uint8, indexed img[x, y], blobs pure
red (255, 0, 0) on white.
"""

import numpy as np

from ..ch04_gpu_2blob_nblock import scenes as _two
from ...shared import scenes as _one

RED = _one.RED
WHITE = _one.WHITE


def _blank(width, height):
    return np.full((width, height, 3), 255, dtype=np.uint8)


# ---------------------------------------------------- adapted two-blob scenes

def two_squares_scene(width, height, blob_w, blob_h, gap=4):
    img, _ = _two.two_squares_scene(width, height, blob_w, blob_h, gap)
    return img, 2


def two_disks_scene(width, height, radius, gap=4):
    img, _ = _two.two_disks_scene(width, height, radius, gap)
    return img, 2


def asym_squares_scene(width, height, big_side, small_side, gap=4):
    img, _ = _two.asym_squares_scene(width, height, big_side, small_side, gap)
    return img, 2


def two_pixels_scene(width, height):
    img, _ = _two.two_pixels_scene(width, height)
    return img, 2


# --------------------------------------------------- adapted one-blob scenes

def square_scene(width, height, blob_w, blob_h, corner=False):
    img, _, _ = _one.square_scene(width, height, blob_w, blob_h, corner)
    return img, 1


def disk_scene(width, height, radius):
    img, _, _ = _one.disk_scene(width, height, radius)
    return img, 1


def serpentine_scene(width, height):
    img, _, _ = _one.serpentine_scene(width, height)
    return img, 1


def single_pixel_scene(width, height):
    img, _, _ = _one.single_pixel_scene(width, height)
    return img, 1


def full_red_scene(width, height):
    img, _, _ = _one.full_red_scene(width, height)
    return img, 1


def blank_scene(width, height):
    """Zero blobs — the discovery kernels must terminate on nothing."""
    return _blank(width, height), 0


# --------------------------------------------------------------- new scenes

def u_shape_scene(width, height, arm_len=None, arm_w=None, gap=None):
    """One U-shaped blob: two arms toward low x, bridged at high x.

    Both arm tips sit at the same minimal x with white between them, so
    the candidate rule (no red lex-predecessor) fires TWICE on one blob —
    the smallest scene that forces the seed-merge variant to merge.
    """
    arm_len = arm_len if arm_len is not None else max(width // 2, 2)
    arm_w = arm_w if arm_w is not None else max(width // 8, 1)
    gap = gap if gap is not None else max(height // 3, 2)
    x0 = (width - arm_len - arm_w) // 2
    y0 = (height - 2 * arm_w - gap) // 2
    if x0 < 0 or y0 < 0:
        raise ValueError(
            f"U with arm_len {arm_len}, arm_w {arm_w}, gap {gap} does not "
            f"fit in {width}x{height}")
    img = _blank(width, height)
    ya, yb = y0, y0 + arm_w + gap
    img[x0:x0 + arm_len, ya:ya + arm_w] = RED            # arm A
    img[x0:x0 + arm_len, yb:yb + arm_w] = RED            # arm B
    img[x0 + arm_len:x0 + arm_len + arm_w,
        ya:yb + arm_w] = RED                             # bridge
    return img, 1


def comb_scene(width, height, teeth=80, tooth_len=8, spine_w=2):
    """One comb-shaped blob: `teeth` 1-px teeth toward low x, joined by a
    spine at high x. Every tooth tip is a candidate — teeth=80 puts the
    provisional-label count past the retired 64-label queue format.
    """
    span = 2 * teeth - 1          # teeth at every other y column
    x0 = (width - tooth_len - spine_w) // 2
    y0 = (height - span) // 2
    if teeth < 1 or x0 < 0 or y0 < 0:
        raise ValueError(
            f"comb with {teeth} teeth (span {span}) does not fit "
            f"in {width}x{height}")
    img = _blank(width, height)
    for i in range(teeth):
        img[x0:x0 + tooth_len, y0 + 2 * i] = RED
    img[x0 + tooth_len:x0 + tooth_len + spine_w, y0:y0 + span] = RED
    return img, 1


def blob_grid_scene(width, height, nx, ny, blob_side, gap=4):
    """nx*ny equal solid squares on a regular grid — the N-blob workhorse
    (10x10 = 100 blobs also outgrows the retired 64-label format)."""
    if gap < 2:
        raise ValueError(f"gap must be >= 2 white pixels, got {gap}")
    span_x = nx * blob_side + (nx - 1) * gap
    span_y = ny * blob_side + (ny - 1) * gap
    if nx < 1 or ny < 1 or span_x > width or span_y > height:
        raise ValueError(
            f"{nx}x{ny} grid of {blob_side}px blobs with gap {gap} does "
            f"not fit in {width}x{height}")
    img = _blank(width, height)
    x0 = (width - span_x) // 2
    y0 = (height - span_y) // 2
    for i in range(nx):
        for j in range(ny):
            xs = x0 + i * (blob_side + gap)
            ys = y0 + j * (blob_side + gap)
            img[xs:xs + blob_side, ys:ys + blob_side] = RED
    return img, nx * ny


def random_blobs_scene(width, height, density=0.3, rng_seed=0):
    """Sub-percolation red noise: hundreds of components of every shape.

    Below ~0.5 density (8-connectivity percolation) the noise shatters
    into many small islands. n_blobs is not knowable by construction —
    the CPU oracle counts the components at build time.
    """
    from .cpu_oracle import cpu_label_components
    rng = np.random.default_rng(rng_seed)
    img = _blank(width, height)
    img[rng.random((width, height)) < density] = RED
    _, n_blobs = cpu_label_components(img)
    return img, int(n_blobs)


def png_scene(path):
    """Load an external PNG as a seedless scene, img[x, y] pure red on
    white: red-dominant pixels (R >= 128, G < 128, B < 128) snap to RED,
    everything else to WHITE. Returns (img, None) — foreign inputs carry
    no ground-truth blob count; the benchmark's cross-config crosscheck
    stands in for it.
    """
    from PIL import Image
    arr = np.array(Image.open(path).convert("RGB"))        # [y, x, 3]
    red = ((arr[..., 0] >= 128) & (arr[..., 1] < 128)
           & (arr[..., 2] < 128))
    img = np.full(arr.shape, 255, dtype=np.uint8)
    img[red] = RED
    return np.ascontiguousarray(np.transpose(img, (1, 0, 2))), None
