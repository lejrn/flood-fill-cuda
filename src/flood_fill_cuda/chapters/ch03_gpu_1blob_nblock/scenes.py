"""Scene generators: ch01_gpu_1blob_1block's set, re-exported.

The dual-block stage's three seam-aware scenes are deliberately NOT carried
forward: they existed to exercise the split kernel's seam at x = width//2,
and this stage's only partitioning is by queue index — no seam exists.
"""

from ..ch01_gpu_1blob_1block.scenes import (
    RED, WHITE, square_scene, disk_scene, serpentine_scene, random_scene,
    single_pixel_scene, full_red_scene, corner_seeded_square_scene,
    overflow_scene,
)
