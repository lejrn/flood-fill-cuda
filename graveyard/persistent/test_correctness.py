"""
Correctness tests: GPU persistent-kernel flood fill vs CPU reference BFS.

The GPU result must match the CPU reference exactly:
- visited mask (which pixels were filled)
- depth map (the BFS level of every pixel — catches level-mixing races that
  a visited-only comparison would miss)
- level count and filled count

Run: uv run pytest src/gpu/single_blob/persistent/test_correctness.py -v
"""

import numpy as np
import pytest

from flood_fill import flood_fill
from reference import cpu_flood_fill
import scenes


def assert_matches_reference(img, seed_x, seed_y, **gpu_kwargs):
    ref_visited, ref_depth, ref_levels, ref_filled = cpu_flood_fill(img, seed_x, seed_y)
    result = flood_fill(img, seed_x, seed_y, **gpu_kwargs)

    np.testing.assert_array_equal(result.visited, ref_visited)
    np.testing.assert_array_equal(result.depth, ref_depth)
    assert result.levels == ref_levels
    assert result.filled == ref_filled
    # Every reached pixel must actually have been recolored (no longer red
    # where it was red before, unless the gradient reproduced red — it can't).
    filled_mask = result.visited == 1
    still_red = ((result.img[:, :, 0] == 255)
                 & (result.img[:, :, 1] == 0)
                 & (result.img[:, :, 2] == 0))
    assert not (filled_mask & still_red).any()
    return result


SCENES = {
    "square_64": lambda: scenes.square_scene(64, 64, 32, 32),
    "square_nonsquare_img": lambda: scenes.square_scene(200, 130, 100, 70),
    "square_at_corner": lambda: scenes.square_scene(128, 128, 50, 50, corner=True),
    "square_full_bleed": lambda: scenes.square_scene(96, 96, 96, 96),
    "disk_101": lambda: scenes.disk_scene(101, 101, 40),
    "serpentine_128": lambda: scenes.serpentine_scene(128, 128),
    "serpentine_nonsquare": lambda: scenes.serpentine_scene(64, 200),
    "random_subcritical": lambda: scenes.random_scene(200, 200, 0.35, rng_seed=7),
    "random_supercritical": lambda: scenes.random_scene(200, 200, 0.6, rng_seed=7),
    "single_pixel": lambda: scenes.single_pixel_scene(64, 64),
    "full_red_128": lambda: scenes.full_red_scene(128, 128),
}


@pytest.mark.parametrize("name", SCENES.keys())
def test_matches_cpu_reference(name):
    img, sx, sy = SCENES[name]()
    assert_matches_reference(img, sx, sy)


@pytest.mark.parametrize("tpb", [128, 256, 512])
def test_block_size_invariance(tpb):
    img, sx, sy = scenes.random_scene(256, 256, 0.55, rng_seed=3)
    assert_matches_reference(img, sx, sy, threads_per_block=tpb)


def test_single_block_grid():
    """The kernel must also be correct when parallelism is minimal."""
    img, sx, sy = scenes.disk_scene(80, 80, 30)
    assert_matches_reference(img, sx, sy, threads_per_block=128, blocks=1)


def test_deterministic_across_runs():
    """Queue order is nondeterministic; visited/depth/counts must not be."""
    img, sx, sy = scenes.random_scene(200, 200, 0.6, rng_seed=11)
    a = flood_fill(img, sx, sy)
    b = flood_fill(img, sx, sy)
    np.testing.assert_array_equal(a.visited, b.visited)
    np.testing.assert_array_equal(a.depth, b.depth)
    assert a.levels == b.levels and a.filled == b.filled


def test_input_not_modified():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    before = img.copy()
    flood_fill(img, sx, sy)
    np.testing.assert_array_equal(img, before)


def test_unreached_pixels_untouched():
    """Pixels outside the seed's component keep their original color."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[10:30, 10:30] = scenes.RED   # component A (seeded)
    img[60:80, 60:80] = scenes.RED   # component B (disconnected)
    result = assert_matches_reference(img, 20, 20)
    np.testing.assert_array_equal(result.img[60:80, 60:80], img[60:80, 60:80])
    assert result.visited[60:80, 60:80].sum() == 0


def test_seed_not_red_raises():
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="not red"):
        flood_fill(img, 0, 0)


def test_seed_out_of_bounds_raises():
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="outside"):
        flood_fill(img, 64, 0)


def test_oversized_grid_raises():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="cooperative"):
        flood_fill(img, sx, sy, blocks=10_000)


def test_depth_is_true_bfs_distance():
    """Spot-check the depth semantics on a known geometry: on a full-red
    image seeded at the corner, 8-connected BFS depth is the Chebyshev
    distance max(x, y)."""
    img, sx, sy = scenes.full_red_scene(64, 64)
    result = flood_fill(img, sx, sy)
    xs, ys = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    np.testing.assert_array_equal(result.depth, np.maximum(xs, ys))
    assert result.levels == 64
