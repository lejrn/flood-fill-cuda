"""Two-blob test scenes.

All scenes are (width, height, 3) uint8 images indexed img[x, y] (repo
convention), with blobs drawn in pure red (255, 0, 0) on a white
background. Each generator returns (img, seeds) where seeds is a list of
exactly two (x, y) tuples, one per blob, each guaranteed red. seeds[0]'s
blob paints blue, seeds[1]'s paints green.

Every builder takes an explicit `gap` — the minimum number of white pixels
between the two blobs — and rejects gap < 2. One white pixel already
disconnects two blobs even at 8-connectivity (diagonal adjacency needs
|dx| <= 1 and |dy| <= 1); requiring two adds a margin so no scene is ever
one careless off-by-one away from being a single blob. reference.py's
merged oracle re-checks disjointness at runtime.
"""

import numpy as np

RED = np.array([255, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)


def _blank(width, height):
    return np.full((width, height, 3), 255, dtype=np.uint8)


def _check_gap(gap):
    if gap < 2:
        raise ValueError(f"gap must be >= 2 white pixels, got {gap}")


def two_squares_scene(width, height, blob_w, blob_h, gap=4):
    """Two equal solid rectangles side by side along x, centered as a group.

    Exactly `gap` white columns separate them.
    """
    _check_gap(gap)
    total_w = 2 * blob_w + gap
    if total_w > width or blob_h > height:
        raise ValueError(
            f"two {blob_w}x{blob_h} blobs with gap {gap} do not fit "
            f"in {width}x{height}")
    img = _blank(width, height)
    x0 = (width - total_w) // 2
    y0 = (height - blob_h) // 2
    x1 = x0 + blob_w + gap
    img[x0:x0 + blob_w, y0:y0 + blob_h] = RED
    img[x1:x1 + blob_w, y0:y0 + blob_h] = RED
    seeds = [(x0 + blob_w // 2, y0 + blob_h // 2),
             (x1 + blob_w // 2, y0 + blob_h // 2)]
    return img, seeds


def two_disks_scene(width, height, radius, gap=4):
    """Two equal solid disks stacked along y, centered as a group.

    The closest approach (along the shared center column) has exactly
    `gap` white rows.
    """
    _check_gap(gap)
    span = 4 * radius + gap + 3  # two (2r+1)-px disks + gap white rows
    if span > height or 2 * radius + 1 > width:
        raise ValueError(
            f"two r={radius} disks with gap {gap} do not fit "
            f"in {width}x{height}")
    img = _blank(width, height)
    cx = width // 2
    cy0 = (height - span) // 2 + radius
    cy1 = cy0 + 2 * radius + gap + 1
    xs, ys = np.ogrid[:width, :height]
    img[(xs - cx) ** 2 + (ys - cy0) ** 2 <= radius ** 2] = RED
    img[(xs - cx) ** 2 + (ys - cy1) ** 2 <= radius ** 2] = RED
    return img, [(cx, cy0), (cx, cy1)]


def asym_squares_scene(width, height, big_side, small_side, gap=4):
    """A big square and a small square side by side along x, both
    vertically centered — the max-vs-sum story scene: sequential pays
    t_big + t_small while multisource pays ~max = t_big.
    """
    _check_gap(gap)
    if big_side < small_side:
        raise ValueError("big_side must be >= small_side")
    total_w = big_side + gap + small_side
    if total_w > width or big_side > height:
        raise ValueError(
            f"{big_side}+{small_side} blobs with gap {gap} do not fit "
            f"in {width}x{height}")
    img = _blank(width, height)
    x0 = (width - total_w) // 2
    yb = (height - big_side) // 2
    xs_ = x0 + big_side + gap
    ys_ = (height - small_side) // 2
    img[x0:x0 + big_side, yb:yb + big_side] = RED
    img[xs_:xs_ + small_side, ys_:ys_ + small_side] = RED
    seeds = [(x0 + big_side // 2, yb + big_side // 2),
             (xs_ + small_side // 2, ys_ + small_side // 2)]
    return img, seeds


def two_pixels_scene(width, height):
    """Two single red pixels — the smallest possible pair of blobs
    (level 0 is the whole story; barrier/launch overhead dominates)."""
    if width < 7 or height < 3:
        raise ValueError("image too small for two separated pixels")
    img = _blank(width, height)
    a = (width // 3, height // 2)
    b = (2 * width // 3, height // 2)
    img[a] = RED
    img[b] = RED
    return img, [a, b]
