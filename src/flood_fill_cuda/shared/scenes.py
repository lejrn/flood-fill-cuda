"""
Test scene generators.

All scenes are (width, height, 3) uint8 images indexed img[x, y] (matching the
repo convention), with blobs drawn in pure red (255, 0, 0) on a white
background. Each generator returns (img, seed_x, seed_y) with the seed
guaranteed to be on a red pixel.
"""

import numpy as np

RED = np.array([255, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)


def square_scene(width, height, blob_w, blob_h, corner=False):
    """Solid rectangular blob, centered by default or at the (0,0) corner."""
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    if corner:
        x0, y0 = 0, 0
    else:
        x0, y0 = (width - blob_w) // 2, (height - blob_h) // 2
    img[x0:x0 + blob_w, y0:y0 + blob_h] = RED
    return img, x0 + blob_w // 2, y0 + blob_h // 2


def disk_scene(width, height, radius):
    """Solid disk blob centered in the image."""
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    xs, ys = np.ogrid[:width, :height]
    img[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = RED
    return img, cx, cy


def serpentine_scene(width, height):
    """Thin snake: every other row is red, joined at alternating ends.

    One connected component whose geodesic diameter is ~(width/2)*height —
    the worst case for level-synchronous BFS (thousands of tiny frontiers).
    """
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    for x in range(0, width, 2):
        img[x, :] = RED
    # Connectors through the odd rows, alternating ends
    for x in range(1, width, 2):
        y = height - 1 if (x // 2) % 2 == 0 else 0
        img[x, y] = RED
    return img, 0, 0


def random_scene(width, height, density, rng_seed=0):
    """Red noise at the given density; seed on the red pixel nearest center.

    Above ~0.5 density (8-connectivity percolation), the seed's component is
    usually a giant blob; below it, a small island. Either way the CPU
    reference from the same seed defines the expected fill.
    """
    rng = np.random.default_rng(rng_seed)
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    red_mask = rng.random((width, height)) < density
    img[red_mask] = RED
    red_coords = np.argwhere(red_mask)
    if red_coords.size == 0:
        raise ValueError("density too low: no red pixels generated")
    center = np.array([width // 2, height // 2])
    nearest = red_coords[np.argmin(((red_coords - center) ** 2).sum(axis=1))]
    return img, int(nearest[0]), int(nearest[1])


def single_pixel_scene(width, height):
    """Exactly one red pixel."""
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    img[width // 2, height // 2] = RED
    return img, width // 2, height // 2


def full_red_scene(width, height):
    """Entire image is one blob; BFS depth spans the whole diagonal."""
    img = np.zeros((width, height, 3), dtype=np.uint8)
    img[:, :] = RED
    return img, 0, 0
