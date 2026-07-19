"""Scene generators: single_block_shared's set, re-exported.

The base scenes are loaded by file path (see reference.py for why). The
dual-block stage's three seam-aware scenes are deliberately NOT carried
forward: they existed to exercise the split kernel's seam at x = width//2,
and this stage's only partitioning is by queue index — no seam exists.
"""
import os

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
