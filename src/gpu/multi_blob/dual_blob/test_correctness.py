"""Correctness tests: dual-blob GPU flood fill vs the merged CPU oracle.

Every mode (sequential / streams / multisource), both entry formats
(lin / xy), both connectivities (4 / 8) and the bare twins must produce
the identical result: visited/depth/label maps equal to the per-seed
oracles merged, blob 0 painted exactly blue, blob 1 exactly green,
everything else untouched.

Streams-mode tests pin explicit small block counts: two concurrent
cooperative grids near the device's co-residency capacity can wedge (see
flood_fill.py), and probing that cliff is benchmark.py's job, not a
correctness concern.

Run per-directory (this file shares its basename with sibling packages
and none has an __init__.py):

    uv run pytest src/gpu/multi_blob/dual_blob/test_correctness.py -v
"""

import os

import numpy as np
import pytest

import bandwidth
from flood_fill import flood_fill, max_blocks, MODES
from reference import cpu_flood_fill_two
import scenes

BLUE = np.array([0, 0, 255], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)

STREAMS_TEST_BLOCKS = 8   # small pair; still not a guarantee — see below

# Streams-mode tests are OPT-IN (DUAL_BLOB_STREAMS=1). Two concurrent
# cooperative grids on this device can wedge forever at grid.sync, and
# the wedge is nondeterministic: standalone pairs at 48+48 and 80+80
# blocks ran fine while 88+88 hung, and even a tiny 8+8 pair hung when
# run after other flood_fill calls in the same process — it cost this
# suite 82 minutes of dead spin before being isolated. A test that can
# hang the runner forever is worse than an unrun test; the mechanism's
# real measurement is the fresh-process probe documented in README.md.
RUN_STREAMS = os.environ.get("DUAL_BLOB_STREAMS") == "1"
TEST_MODES = MODES if RUN_STREAMS else tuple(m for m in MODES
                                             if m != "streams")
requires_streams = pytest.mark.skipif(
    not RUN_STREAMS,
    reason="streams mode can deadlock the GPU; set DUAL_BLOB_STREAMS=1 to run")


def _kw(mode, kw):
    """Streams-mode calls get a safe explicit block count unless the test
    pins its own."""
    if mode == "streams" and "blocks" not in kw:
        kw = dict(kw, blocks=STREAMS_TEST_BLOCKS)
    return kw


def assert_matches_reference(img, seeds, mode, connectivity=4, **gpu_kwargs):
    (ref_v, ref_d, ref_l, ref_levels, ref_filled,
     (la, fa), (lb, fb)) = cpu_flood_fill_two(img, seeds, connectivity)
    result = flood_fill(img, seeds, mode=mode, connectivity=connectivity,
                        **_kw(mode, gpu_kwargs))

    np.testing.assert_array_equal(result.visited, ref_v)
    np.testing.assert_array_equal(result.depth, ref_d)
    np.testing.assert_array_equal(result.label, ref_l)
    assert result.levels == ref_levels
    assert result.filled == ref_filled
    assert result.filled_a == fa and result.filled_b == fb
    assert result.levels_a == la and result.levels_b == lb
    assert (result.img[ref_l == 0] == BLUE).all()
    assert (result.img[ref_l == 1] == GREEN).all()
    untouched = ref_v == 0
    np.testing.assert_array_equal(result.img[untouched], img[untouched])
    return result


SCENES = {
    "two_squares": lambda: scenes.two_squares_scene(96, 64, 30, 30, gap=4),
    "two_disks": lambda: scenes.two_disks_scene(64, 128, 20, gap=4),
    "asym_squares": lambda: scenes.asym_squares_scene(128, 96, 60, 20, gap=4),
    "two_pixels": lambda: scenes.two_pixels_scene(64, 64),
    "min_gap": lambda: scenes.two_squares_scene(64, 64, 24, 24, gap=2),
}


@pytest.mark.parametrize("mode", TEST_MODES)
@pytest.mark.parametrize("name", SCENES.keys())
def test_matches_reference(name, mode):
    img, seeds = SCENES[name]()
    assert_matches_reference(img, seeds, mode)


@pytest.mark.parametrize("mode", TEST_MODES)
def test_xy_format_matches_reference_and_lin(mode):
    """The xy entry format must be bit-for-bit output-equivalent to lin —
    only the in-queue encoding differs."""
    img, seeds = SCENES["two_squares"]()
    r_xy = assert_matches_reference(img, seeds, mode, entry_format="xy")
    r_lin = flood_fill(img, seeds, mode=mode, **_kw(mode, {}))
    np.testing.assert_array_equal(r_xy.img, r_lin.img)
    np.testing.assert_array_equal(r_xy.depth, r_lin.depth)
    np.testing.assert_array_equal(r_xy.label, r_lin.label)


def test_mode_equivalence():
    """The modes are the same algorithm run different ways: identical
    img/visited/depth/label. (owner is excluded — which block claims a
    pixel is race- and mode-dependent.) Covers streams only when
    DUAL_BLOB_STREAMS=1."""
    img, seeds = SCENES["asym_squares"]()
    results = [flood_fill(img, seeds, mode=m, **_kw(m, {}))
               for m in TEST_MODES]
    for other in results[1:]:
        np.testing.assert_array_equal(results[0].img, other.img)
        np.testing.assert_array_equal(results[0].visited, other.visited)
        np.testing.assert_array_equal(results[0].depth, other.depth)
        np.testing.assert_array_equal(results[0].label, other.label)


@pytest.mark.parametrize("blocks", [1, 3, None])
@pytest.mark.parametrize("tpb", [64, 256])
def test_block_count_and_tpb_invariance(blocks, tpb):
    img, seeds = SCENES["two_squares"]()
    assert_matches_reference(img, seeds, "multisource",
                             threads_per_block=tpb, blocks=blocks)


def test_multisource_levels_is_max_of_blobs():
    """One shared clock: the combined level count is the slower blob's,
    not the sum — the whole point of the multisource bet."""
    img, seeds = SCENES["asym_squares"]()
    r = flood_fill(img, seeds, mode="multisource")
    assert r.levels == max(r.levels_a, r.levels_b)
    assert r.levels_a > r.levels_b  # asym: big blob is blob 0


def test_two_pixels_single_level():
    img, seeds = SCENES["two_pixels"]()
    r = flood_fill(img, seeds, mode="multisource")
    assert r.filled == 2 and r.levels == 1
    assert r.filled_a == 1 and r.filled_b == 1


def test_input_not_modified():
    img, seeds = SCENES["two_squares"]()
    before = img.copy()
    flood_fill(img, seeds, mode="multisource")
    np.testing.assert_array_equal(img, before)


def test_deterministic_across_runs():
    img, seeds = SCENES["asym_squares"]()
    a = flood_fill(img, seeds, mode="multisource")
    b = flood_fill(img, seeds, mode="multisource")
    np.testing.assert_array_equal(a.img, b.img)
    np.testing.assert_array_equal(a.depth, b.depth)
    assert a.filled == b.filled and a.levels == b.levels


# -------------------------------------------------------------- accounting

def test_accounting_multisource():
    """filled splits exactly into the two blobs, the label census agrees,
    and every pixel is processed exactly once."""
    img, seeds = SCENES["asym_squares"]()
    r = flood_fill(img, seeds, mode="multisource")
    assert r.filled_a + r.filled_b == r.filled
    assert (r.label == 0).sum() == r.filled_a
    assert (r.label == 1).sum() == r.filled_b
    assert r.processed == r.filled
    assert len(r.launches) == 1
    assert r.launches[0].processed_per_block.sum() == r.processed
    assert r.filled - 2 <= r.cas_attempts <= 4 * r.filled


def test_accounting_sequential():
    """Two launches, each owning exactly its blob's work."""
    img, seeds = SCENES["asym_squares"]()
    r = flood_fill(img, seeds, mode="sequential")
    assert len(r.launches) == 2
    assert r.launches[0].filled == r.filled_a
    assert r.launches[1].filled == r.filled_b
    assert r.launches[0].levels == r.levels_a
    assert r.launches[1].levels == r.levels_b
    assert r.processed == r.filled
    assert r.kernel_ms == pytest.approx(r.kernel_a_ms + r.kernel_b_ms)


def test_owner_census_multisource():
    img, seeds = SCENES["two_disks"]()
    r = flood_fill(img, seeds, mode="multisource", blocks=4)
    reached = r.visited == 1
    owners = r.owner[reached]
    assert owners.min() >= 0 and owners.max() < r.blocks
    assert (r.owner[~reached] == -1).all()
    census = np.bincount(owners.astype(np.int64), minlength=r.blocks)
    np.testing.assert_array_equal(census, r.launches[0].processed_per_block)


def test_label_cannot_mix():
    """Every blue pixel belongs to seed 0's component, every green pixel
    to seed 1's — even at the minimum legal gap."""
    img, seeds = SCENES["min_gap"]()
    r = flood_fill(img, seeds, mode="multisource")
    blue_mask = np.all(r.img == BLUE, axis=2)
    green_mask = np.all(r.img == GREEN, axis=2)
    assert not np.any(blue_mask & green_mask)
    assert blue_mask.sum() == r.filled_a
    assert green_mask.sum() == r.filled_b


# ----------------------------------------------------------------- streams

@requires_streams
def test_streams_reports_overlap_and_per_launch_times():
    img, seeds = SCENES["two_squares"]()
    r = flood_fill(img, seeds, mode="streams", blocks=STREAMS_TEST_BLOCKS)
    assert r.kernel_a_ms > 0 and r.kernel_b_ms > 0
    assert r.overlap_ratio > 0
    assert r.blocks == STREAMS_TEST_BLOCKS


@requires_streams
def test_streams_default_blocks_is_conservative():
    """blocks=None under streams must resolve below half the cooperative
    capacity: two grids at exactly half each can fail to co-schedule and
    wedge at grid.sync (observed on this GPU — see flood_fill.py).

    Deliberately run at tpb=512, where capacity is smallest, so the pair
    this test actually launches stays tiny (~8+8 blocks). Exercising the
    default at a small tpb would launch ~64+64 concurrent cooperative
    blocks — squarely in the nondeterministic wedge zone that hung this
    suite for 82 minutes once already."""
    img, seeds = SCENES["two_squares"]()
    coop = max_blocks(threads_per_block=512)
    r = flood_fill(img, seeds, mode="streams", threads_per_block=512)
    assert 1 <= r.blocks < coop // 2


def test_non_streams_modes_report_no_overlap():
    img, seeds = SCENES["two_squares"]()
    for mode in ("sequential", "multisource"):
        r = flood_fill(img, seeds, mode=mode)
        assert r.overlap_ratio == 0.0


# --------------------------------------------------------------- bare twins

@pytest.mark.parametrize("mode", TEST_MODES)
def test_bare_twin_matches_reference(mode):
    img, seeds = SCENES["two_squares"]()
    assert_matches_reference(img, seeds, mode, bare=True)


def test_bare_twin_reports_no_instrumentation():
    img, seeds = SCENES["two_squares"]()
    r = flood_fill(img, seeds, mode="multisource", bare=True)
    assert r.bare
    assert r.processed == 0 and r.cas_attempts == 0
    assert r.owner.size == 0
    assert r.model_bytes == 0 and r.model_gb_s == 0.0
    assert r.filled > 0 and r.levels > 0


# ------------------------------------------------------- 8-connectivity twins

@pytest.mark.parametrize("mode", TEST_MODES)
@pytest.mark.parametrize("name", ["two_squares", "asym_squares", "min_gap"])
def test_matches_reference_8(name, mode):
    img, seeds = SCENES[name]()
    assert_matches_reference(img, seeds, mode, connectivity=8)


def test_conn8_fewer_levels_same_fill():
    img, seeds = SCENES["two_squares"]()
    r4 = flood_fill(img, seeds, mode="multisource", connectivity=4)
    r8 = flood_fill(img, seeds, mode="multisource", connectivity=8)
    np.testing.assert_array_equal(r4.visited, r8.visited)
    np.testing.assert_array_equal(r4.label, r8.label)
    assert r4.filled == r8.filled
    assert r8.levels < r4.levels


def test_xy_format_8():
    img, seeds = SCENES["two_squares"]()
    assert_matches_reference(img, seeds, "multisource", connectivity=8,
                             entry_format="xy")


# ------------------------------------------------------------ bandwidth model

def test_model_bytes_consistency():
    """Labeling adds ZERO bytes to the model: the result's figure must be
    exactly the single-blob formula applied to the combined counters."""
    img, seeds = SCENES["two_disks"]()
    r = flood_fill(img, seeds, mode="multisource")
    expected = bandwidth.model_bytes(r.processed, r.cas_attempts,
                                     r.filled, instrumented=True)
    assert r.model_bytes == expected
    assert r.model_gb_s > 0


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("bad_seeds", [
    [(5, 5)],                          # one seed
    [(5, 5), (6, 6), (7, 7)],          # three seeds
    [(5, 5), (5, 5)],                  # duplicate
    "nonsense",
])
def test_rejects_bad_seed_lists(bad_seeds):
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError):
        flood_fill(img, bad_seeds, mode="multisource")


def test_rejects_non_red_seed():
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="not red"):
        flood_fill(img, [seeds[0], (0, 0)], mode="multisource")


def test_rejects_out_of_bounds_seed():
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="outside"):
        flood_fill(img, [seeds[0], (96, 0)], mode="multisource")


def test_rejects_bad_mode():
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="mode"):
        flood_fill(img, seeds, mode="parallel")


def test_rejects_bad_entry_format():
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="entry_format"):
        flood_fill(img, seeds, entry_format="packed")


def test_rejects_xy_oversized_dimension():
    img = np.full((8200, 4, 3), 255, dtype=np.uint8)
    img[1, 1] = scenes.RED
    img[8000, 1] = scenes.RED
    with pytest.raises(ValueError, match="xy-format"):
        flood_fill(img, [(1, 1), (8000, 1)], entry_format="xy")


@pytest.mark.parametrize("tpb", [100, 0, 1024])
def test_rejects_bad_threads_per_block(tpb):
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="threads_per_block"):
        flood_fill(img, seeds, threads_per_block=tpb)


@pytest.mark.parametrize("blocks", [0, -1])
def test_rejects_bad_block_count(blocks):
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="blocks"):
        flood_fill(img, seeds, blocks=blocks)


def test_rejects_blocks_beyond_cooperative_capacity():
    img, seeds = SCENES["two_squares"]()
    coop_max = max_blocks(threads_per_block=256)
    with pytest.raises(RuntimeError, match="cooperative"):
        flood_fill(img, seeds, blocks=coop_max + 1)


@pytest.mark.parametrize("conn", [5, 0, "8"])
def test_rejects_bad_connectivity(conn):
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="connectivity"):
        flood_fill(img, seeds, connectivity=conn)


# ------------------------------------------------------------- scene contract

def test_scenes_reject_small_gap():
    with pytest.raises(ValueError, match="gap"):
        scenes.two_squares_scene(96, 64, 30, 30, gap=1)


def test_scene_seeds_are_red_and_disjoint():
    for name, builder in SCENES.items():
        img, seeds = builder()
        (ax, ay), (bx, by) = seeds
        assert (img[ax, ay] == scenes.RED).all()
        assert (img[bx, by] == scenes.RED).all()
        # the merged oracle raises if the components touch
        cpu_flood_fill_two(img, seeds, connectivity=8)
