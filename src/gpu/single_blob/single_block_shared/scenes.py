"""
Test scenes for the single-block shared-memory kernel.

Re-exports the generators from persistent/scenes.py (loaded by file path via
importlib — adding persistent/ to sys.path would shadow this package's own
kernels/flood_fill/scenes modules, which share basenames), plus two scenes
specific to the shared-ring capacity story:

- corner_seeded_square_scene: seed at (0, 0). persistent's
  square_scene(corner=True) places the *blob* at the corner but still seeds
  at the blob center, which halves nothing — corner seeding is what caps the
  peak frontier at ~2W instead of ~4W.
- overflow_scene: guaranteed to overflow the 8192-entry shared ring.
"""

import importlib.util
import os

import numpy as np

_PERSISTENT_SCENES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "persistent", "scenes.py")
_spec = importlib.util.spec_from_file_location("_persistent_scenes", _PERSISTENT_SCENES)
_persistent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_persistent)

RED = _persistent.RED
WHITE = _persistent.WHITE
square_scene = _persistent.square_scene
disk_scene = _persistent.disk_scene
serpentine_scene = _persistent.serpentine_scene
random_scene = _persistent.random_scene
single_pixel_scene = _persistent.single_pixel_scene
full_red_scene = _persistent.full_red_scene


def corner_seeded_square_scene(width, height, blob_w, blob_h):
    """Solid red blob at the (0, 0) corner, seeded AT the corner.

    Corner seeding halves the peak ring occupancy vs center seeding
    (anti-diagonal frontiers of ~min(blob_w, blob_h) instead of Manhattan
    diamonds of ~2*side), which is what lets a corner-seeded 4000x4000
    square fit the 8192-entry shared ring while a center-seeded one cannot.
    """
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    img[:blob_w, :blob_h] = RED
    return img, 0, 0


def overflow_scene():
    """Full-bleed center-seeded 2600x2600 square: guaranteed ring overflow.

    4-connected BFS from the center grows Manhattan-diamond frontiers of
    ~4r pixels. Ring occupancy peaks at two adjacent levels ~8r+4, which
    crosses RING_CAPACITY=8192 at r~1024 — well before the diamond reaches
    the walls at r=1300 — so the tripwire fires deterministically and fast.
    """
    img = np.zeros((2600, 2600, 3), dtype=np.uint8)
    img[:, :] = RED
    return img, 1300, 1300
