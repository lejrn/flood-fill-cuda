"""Correctness tests: GPU seed discovery vs the CPU oracles.

Nobody passes seeds in this chapter, so the oracle defines the expected
answer (see cpu_oracle.py): canonical label = min linear index per blob,
canonical seed = that pixel. Both GPU variants — seed_merge (candidate
scan + colliding waves + atomicMin union) and ccl_fill (union-find CCL,
then multisource fill) — must reproduce it exactly, each against its own
depth semantics, and agree with EACH OTHER on labels and seeds
everywhere: that agreement is the chapter's central claim.

Run:

    uv run pytest src/flood_fill_cuda/chapters/ch05_gpu_nblob_nblock/test_correctness.py -v
"""

import numpy as np
import pytest

from . import scenes
from .cpu_oracle import (
    cpu_candidates, cpu_label_components,
    cpu_fill_canonical, cpu_fill_from_candidates,
)
from ...shared.cpu_oracle import cpu_flood_fill_8

# Small enough to run the whole matrix quickly, big enough that every
# scene has interior, and (for the GPU tests) that multiple blocks and
# levels genuinely interleave.
SCENES = {
    "two_squares": lambda: scenes.two_squares_scene(96, 64, 30, 30, gap=4),
    "two_disks": lambda: scenes.two_disks_scene(64, 128, 20, gap=4),
    "asym_squares": lambda: scenes.asym_squares_scene(128, 96, 60, 20, gap=4),
    "two_pixels": lambda: scenes.two_pixels_scene(64, 64),
    "min_gap": lambda: scenes.two_squares_scene(64, 64, 24, 24, gap=2),
    "square": lambda: scenes.square_scene(64, 64, 30, 30),
    "disk": lambda: scenes.disk_scene(64, 64, 20),
    "u_shape": lambda: scenes.u_shape_scene(64, 64),
    "comb": lambda: scenes.comb_scene(48, 176, teeth=80),
    "blob_grid": lambda: scenes.blob_grid_scene(128, 96, 4, 3, 20, gap=4),
    "single_pixel": lambda: scenes.single_pixel_scene(32, 32),
    "full_red": lambda: scenes.full_red_scene(48, 48),
}


def _red_mask(img):
    return ((img[:, :, 0] == 255) & (img[:, :, 1] == 0)
            & (img[:, :, 2] == 0))


def _canonical_seed_mask(img):
    """1 where a pixel is its own component's minimum linear index."""
    height = img.shape[1]
    label, _ = cpu_label_components(img)
    lin = (np.arange(img.shape[0])[:, None] * height
           + np.arange(height)[None, :]).astype(np.int32)
    return (label == lin) & (label >= 0)


# ------------------------------------------------------------ scene contract

@pytest.mark.parametrize("name", SCENES.keys())
def test_scene_blob_count(name):
    """Each builder's declared n_blobs matches the CCL oracle's count."""
    img, n_blobs = SCENES[name]()
    _, counted = cpu_label_components(img)
    assert counted == n_blobs


def test_blank_scene_has_no_blobs():
    img, n_blobs = scenes.blank_scene(32, 32)
    assert n_blobs == 0
    assert not _red_mask(img).any()
    assert cpu_candidates(img).sum() == 0


def test_random_blobs_scene_counts_itself():
    img, n_blobs = scenes.random_blobs_scene(128, 128, density=0.3,
                                             rng_seed=7)
    _, counted = cpu_label_components(img)
    assert counted == n_blobs
    assert n_blobs > 50  # genuinely many-blob at this density and size


def test_scene_builders_validate():
    with pytest.raises(ValueError, match="gap"):
        scenes.blob_grid_scene(128, 96, 4, 3, 20, gap=1)
    with pytest.raises(ValueError, match="fit"):
        scenes.blob_grid_scene(32, 32, 10, 10, 20)
    with pytest.raises(ValueError, match="comb"):
        scenes.comb_scene(48, 64, teeth=80)


# -------------------------------------------------------- candidate scan rule

@pytest.mark.parametrize("name", SCENES.keys())
def test_canonical_seed_is_always_a_candidate(name):
    """The lemma both variants stand on: every blob's lex-min pixel
    passes the local candidate rule, so min(candidates) == min(blob) and
    the union-find root IS the canonical label."""
    img, n_blobs = SCENES[name]()
    candidates = cpu_candidates(img)
    seed_mask = _canonical_seed_mask(img)
    assert seed_mask.sum() == n_blobs
    assert (candidates[seed_mask] == 1).all()


@pytest.mark.parametrize("name", SCENES.keys())
def test_candidates_form_an_independent_set(name):
    """No two candidates are 8-adjacent — of any adjacent pair, the
    lex-smaller sits in a predecessor slot of the other."""
    cand = cpu_candidates(SCENES[name]()[0]).astype(bool)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = cand[max(dx, 0):cand.shape[0] + min(dx, 0),
                 max(dy, 0):cand.shape[1] + min(dy, 0)]
        b = cand[max(-dx, 0):cand.shape[0] + min(-dx, 0),
                 max(-dy, 0):cand.shape[1] + min(-dy, 0)]
        assert not (a & b).any()


def test_u_shape_has_two_candidates_one_blob():
    """The scene that forces the merge: a local scan finds both arm tips
    and cannot know they are the same blob."""
    img, n_blobs = scenes.u_shape_scene(64, 64)
    assert n_blobs == 1
    assert cpu_candidates(img).sum() == 2


def test_comb_has_more_candidates_than_the_retired_label_format():
    img, n_blobs = scenes.comb_scene(48, 176, teeth=80)
    assert n_blobs == 1
    assert cpu_candidates(img).sum() == 80  # one per tooth tip, > 64


def test_rectangles_have_one_candidate_per_blob():
    """Only axis-aligned rectangles guarantee a single candidate. Even a
    CONVEX blob multi-triggers the local rule: a rasterized disk's upper
    arc is a staircase, and every step wide enough to clear the previous
    row's diagonal is another lex corner."""
    for name in ("two_squares", "blob_grid", "square", "single_pixel",
                 "full_red"):
        img, n_blobs = SCENES[name]()
        assert cpu_candidates(img).sum() == n_blobs, name
    img, n_blobs = SCENES["two_disks"]()
    per_disk = cpu_candidates(img).sum() / n_blobs
    assert per_disk > 1  # the staircase effect, deliberately not hidden


# ------------------------------------------------------------- oracle fills

@pytest.mark.parametrize("name", SCENES.keys())
def test_oracles_cover_exactly_the_red_pixels(name):
    img, _ = SCENES[name]()
    red = _red_mask(img)
    for fill in (cpu_fill_canonical, cpu_fill_from_candidates):
        visited, depth, label, levels, filled = fill(img)
        np.testing.assert_array_equal(visited.astype(bool), red)
        assert filled == int(red.sum())
        assert ((depth >= 0) == red).all()
        assert ((label >= 0) == red).all()
        assert levels == (int(depth.max()) + 1 if red.any() else 0)


@pytest.mark.parametrize("name", SCENES.keys())
def test_oracle_labels_agree_across_semantics(name):
    """Both depth semantics sit on the SAME canonical label map."""
    img, _ = SCENES[name]()
    label_ccl, _ = cpu_label_components(img)
    _, _, label_a, _, _ = cpu_fill_from_candidates(img)
    _, _, label_b, _, _ = cpu_fill_canonical(img)
    np.testing.assert_array_equal(label_a, label_ccl)
    np.testing.assert_array_equal(label_b, label_ccl)


def test_canonical_depth_matches_per_seed_single_source_fills():
    """Cross-check against the battle-tested shared oracle: running the
    single-seed conn8 fill from each canonical seed and merging must
    reproduce cpu_fill_canonical exactly (blobs are disjoint)."""
    img, _ = SCENES["blob_grid"]()
    visited, depth, label, levels, filled = cpu_fill_canonical(img)
    seed_mask = _canonical_seed_mask(img)
    merged_depth = np.full_like(depth, -1)
    total = 0
    per_blob_levels = []
    for sx, sy in np.argwhere(seed_mask):
        v, d, lv, f = cpu_flood_fill_8(img, sx, sy)
        merged_depth[v == 1] = d[v == 1]
        per_blob_levels.append(lv)
        total += f
    np.testing.assert_array_equal(depth, merged_depth)
    assert filled == total
    assert levels == max(per_blob_levels)


def test_candidate_depth_is_nearest_candidate_distance():
    """On the U, the two waves split the blob: every pixel's depth is the
    min over per-candidate single-source distances."""
    img, _ = SCENES["u_shape"]()
    _, depth, _, _, _ = cpu_fill_from_candidates(img)
    dists = []
    for sx, sy in np.argwhere(cpu_candidates(img)):
        v, d, _, _ = cpu_flood_fill_8(img, sx, sy)
        dists.append(np.where(v == 1, d, np.iinfo(np.int32).max))
    expected = np.minimum.reduce(dists)
    reached = expected != np.iinfo(np.int32).max
    np.testing.assert_array_equal(depth[reached], expected[reached])
    assert (depth[~reached] == -1).all()
    # and the merge genuinely shortens the clock vs the canonical seed
    _, _, _, levels_canon, _ = cpu_fill_canonical(img)
    _, _, _, levels_cand, _ = cpu_fill_from_candidates(img)
    assert levels_cand < levels_canon


def test_oracle_blank_image():
    img, _ = scenes.blank_scene(32, 32)
    for fill in (cpu_fill_canonical, cpu_fill_from_candidates):
        visited, depth, label, levels, filled = fill(img)
        assert filled == 0 and levels == 0
        assert not visited.any()
        assert (depth == -1).all() and (label == -1).all()
