"""Scene generators: single_block_shared's set plus three seam-aware scenes.

The base scenes are loaded by file path (see reference.py for why). The new
scenes exist because the split kernel divides the image at the vertical seam
x = width//2, and the interesting behaviors — inbox traffic, ownership
imbalance, ping-ponging wavefronts — need scenes built around that seam.
Note the base serpentine's red lines run at constant x, PARALLEL to the
seam: its wave crosses the seam exactly once, which is why the transposed
seam_serpentine below exists.
"""
import os

import numpy as np

from reference import load_by_path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SBS = os.path.abspath(os.path.join(_HERE, os.pardir, "single_block_shared"))

_scenes = load_by_path("_sbs_scenes", os.path.join(_SBS, "scenes.py"))

RED = _scenes.RED
WHITE = _scenes.WHITE
square_scene = _scenes.square_scene
disk_scene = _scenes.disk_scene
serpentine_scene = _scenes.serpentine_scene
random_scene = _scenes.random_scene
single_pixel_scene = _scenes.single_pixel_scene
full_red_scene = _scenes.full_red_scene
corner_seeded_square_scene = _scenes.corner_seeded_square_scene
overflow_scene = _scenes.overflow_scene


def seam_serpentine_scene(width, height):
    """Transposed serpentine: every red line spans the full width.

    Lines at even y run across all x — each one crosses the split kernel's
    seam — joined by single connector pixels at alternating ends, so the
    BFS wave ping-pongs across the seam once per line. Worst case for the
    split kernel's inbox choreography (and a stress test for it).
    """
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    for y in range(0, height, 2):
        img[:, y] = RED
    for y in range(1, height, 2):
        x = width - 1 if (y // 2) % 2 == 0 else 0
        img[x, y] = RED
    return img, 0, 0


def seam_seeded_scene(width, height, blob_w, blob_h):
    """Centered solid blob seeded exactly on the seam column x = width//2.

    That column belongs to block 1, so the very first expansion pushes
    pixels of column width//2 - 1 into block 0's inbox — immediate
    cross-seam traffic from level 0.
    """
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    x0 = (width - blob_w) // 2
    y0 = (height - blob_h) // 2
    img[x0:x0 + blob_w, y0:y0 + blob_h] = RED
    return img, width // 2, height // 2


def offcenter_blob_scene(width, height, blob):
    """Solid blob entirely inside block 0's half (x < width//2).

    The split kernel's worst case: block 1 owns no red pixel and processes
    exactly zero work (balance 0%). The dirsplit kernel's showcase: its
    direction partition still splits this blob ~50/50.
    """
    half = width // 2
    x0 = 2
    if x0 + blob > half - 1:
        raise ValueError("blob does not fit strictly inside the left half")
    y0 = (height - blob) // 2
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    img[x0:x0 + blob, y0:y0 + blob] = RED
    return img, x0 + blob // 2, y0 + blob // 2
