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
from .flood_fill import flood_fill, max_blocks, VARIANTS, model_bytes_ch05
from .kernels import PALETTE_HOST, N_PALETTE
from ...shared.cpu_oracle import cpu_flood_fill_8

# Each variant's exact depth semantics (labels/visited agree across all)
_ORACLES = {
    "ccl_fill": cpu_fill_canonical,
    "seed_merge": cpu_fill_from_candidates,
}

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


# ============================================================ GPU variants

def assert_matches_oracle(img, variant, **gpu_kwargs):
    """The full contract: visited/depth/label/levels/filled equal the
    variant's oracle, every visited pixel wears its label's palette row,
    the background is untouched, and the discovered seeds are exactly
    one canonical (lex-min) pixel per blob."""
    ref_v, ref_d, ref_l, ref_levels, ref_filled = _ORACLES[variant](img)
    result = flood_fill(img, variant=variant, **gpu_kwargs)

    np.testing.assert_array_equal(result.visited, ref_v)
    np.testing.assert_array_equal(result.label, ref_l)
    np.testing.assert_array_equal(result.depth, ref_d)
    assert result.levels == ref_levels
    assert result.filled == ref_filled

    vis = ref_v.astype(bool)
    np.testing.assert_array_equal(result.img[vis],
                                  PALETTE_HOST[ref_l[vis] % N_PALETTE])
    np.testing.assert_array_equal(result.img[~vis], img[~vis])

    expected_labels = np.unique(ref_l[vis])
    height = img.shape[1]
    assert result.n_blobs == expected_labels.size
    assert result.seeds == [(int(l) // height, int(l) % height)
                            for l in expected_labels]
    return result


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", SCENES.keys())
def test_gpu_matches_oracle(name, variant):
    img, n_blobs = SCENES[name]()
    r = assert_matches_oracle(img, variant)
    assert r.n_blobs == n_blobs


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("blocks", [1, 3, None])
@pytest.mark.parametrize("tpb", [64, 256])
def test_block_count_and_tpb_invariance(variant, blocks, tpb):
    for name in ("two_squares", "u_shape"):
        img, _ = SCENES[name]()
        assert_matches_oracle(img, variant,
                              threads_per_block=tpb, blocks=blocks)


@pytest.mark.parametrize("variant", VARIANTS)
def test_gpu_many_blobs_exceed_retired_label_format(variant):
    """100 disjoint blobs and a random shatter of hundreds — both far
    past ch04's 64-label entry format cap."""
    img, n_blobs = scenes.blob_grid_scene(256, 256, 10, 10, 20, gap=4)
    assert n_blobs == 100
    r = assert_matches_oracle(img, variant)
    assert r.n_blobs == 100
    img, n_blobs = scenes.random_blobs_scene(128, 128, density=0.3,
                                             rng_seed=7)
    r = assert_matches_oracle(img, variant)
    assert r.n_blobs == n_blobs > 50


@pytest.mark.parametrize("variant", VARIANTS)
def test_gpu_blank_image(variant):
    """A blank image is a valid question with answer zero — the kernel
    must terminate with an empty queue (the fence-sandwich rear read is
    exercised with rear == 0)."""
    img, _ = scenes.blank_scene(64, 64)
    r = flood_fill(img, variant=variant)
    assert r.n_blobs == 0 and r.filled == 0 and r.levels == 0
    assert r.seeds == []
    assert not r.visited.any()
    np.testing.assert_array_equal(r.img, img)


@pytest.mark.parametrize("variant", VARIANTS)
def test_gpu_serpentine(variant):
    """The barrier-bound worst case still discovers and fills exactly."""
    img, _ = scenes.serpentine_scene(48, 48)
    assert_matches_oracle(img, variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_input_not_modified(variant):
    img, _ = SCENES["two_squares"]()
    before = img.copy()
    flood_fill(img, variant=variant)
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize("variant", VARIANTS)
def test_deterministic_across_runs(variant):
    """Label, depth and paint are pure functions of the image — the
    races decide who claims and who merges, never the outcome."""
    img, _ = SCENES["u_shape"]()
    a = flood_fill(img, variant=variant)
    b = flood_fill(img, variant=variant)
    np.testing.assert_array_equal(a.img, b.img)
    np.testing.assert_array_equal(a.depth, b.depth)
    np.testing.assert_array_equal(a.label, b.label)
    assert a.seeds == b.seeds


# ------------------------------------------------------------ cross-variant

@pytest.mark.parametrize("name", SCENES.keys())
def test_variants_agree_on_labels_and_seeds(name):
    """The chapter's central claim: opposite strategies, identical
    canonical answer. Depth may differ (nearest-candidate vs canonical
    seed) — labels, seeds and coverage may not."""
    img, _ = SCENES[name]()
    a = flood_fill(img, variant="seed_merge")
    b = flood_fill(img, variant="ccl_fill")
    np.testing.assert_array_equal(a.label, b.label)
    np.testing.assert_array_equal(a.visited, b.visited)
    assert a.seeds == b.seeds
    assert a.n_blobs == b.n_blobs
    assert a.filled == b.filled


# ------------------------------------------------------- union accounting

def test_seed_merge_candidate_accounting():
    """The scan finds exactly the oracle's candidates; every one starts
    a level-0 wave; each effective union retires exactly one of them."""
    for name in ("two_squares", "two_disks", "u_shape", "comb",
                 "blob_grid", "disk"):
        img, _ = SCENES[name]()
        r = flood_fill(img, variant="seed_merge")
        n_candidates = int(cpu_candidates(img).sum())
        assert r.candidates == n_candidates, name
        assert (r.depth == 0).sum() == n_candidates, name
        np.testing.assert_array_equal((r.depth == 0).astype(np.int32),
                                      cpu_candidates(img))
        assert r.union_done == r.candidates - r.n_blobs, name
        assert r.union_attempts >= r.union_done


def test_seed_merge_forced_merges():
    """u_shape and comb are the scenes where the local scan CANNOT nail
    uniqueness — the merge path must actually run."""
    for name, n_cand in (("u_shape", 2), ("comb", 80)):
        img, _ = SCENES[name]()
        r = flood_fill(img, variant="seed_merge")
        assert r.n_blobs == 1
        assert r.candidates == n_cand
        assert r.union_done == n_cand - 1
        # one surviving label: the canonical (lex-min) one
        assert np.unique(r.label[r.visited == 1]).size == 1


def test_seed_merge_prov_label_snapshot():
    """The instrumented twin preserves the pre-merge picture: every
    filled pixel's provisional label is a candidate of its OWN blob, and
    on multi-candidate blobs more than one provisional wave survives to
    the snapshot."""
    img, _ = SCENES["u_shape"]()
    height = img.shape[1]
    r = flood_fill(img, variant="seed_merge")
    vis = r.visited == 1
    assert (r.prov_label[~vis] == -1).all()
    provs = np.unique(r.prov_label[vis])
    cand = cpu_candidates(img)
    for p in provs:
        px, py = int(p) // height, int(p) % height
        assert cand[px, py] == 1              # a real candidate...
        assert r.label[px, py] == r.label[vis][0]  # ...of this blob
    assert provs.size == 2  # both arms' waves reached the snapshot


def test_ccl_union_accounting():
    """Every red pixel starts as a root; each effective link retires
    exactly one — so union_done == filled - n_blobs, structurally."""
    for name in ("two_squares", "u_shape", "comb", "blob_grid", "disk"):
        img, _ = SCENES[name]()
        r = flood_fill(img, variant="ccl_fill")
        assert r.union_done == r.filled - r.n_blobs, name
        assert r.union_attempts >= r.union_done
        assert r.candidates == r.n_blobs, name  # queue held one seed/blob


def test_ccl_seed_is_the_unique_depth0_pixel_per_blob():
    img, _ = SCENES["blob_grid"]()
    r = flood_fill(img, variant="ccl_fill")
    for sx, sy in r.seeds:
        blob = r.label == r.label[sx, sy]
        assert r.depth[sx, sy] == 0
        assert (r.depth[blob] == 0).sum() == 1
        # and it is the blob's lex-min pixel
        xs, ys = np.nonzero(blob)
        k = np.lexsort((ys, xs))[0]
        assert (xs[k], ys[k]) == (sx, sy)


# ------------------------------------------------------------- accounting

@pytest.mark.parametrize("variant", VARIANTS)
def test_accounting(variant):
    img, _ = SCENES["blob_grid"]()
    r = flood_fill(img, variant=variant)
    red = int(_red_mask(img).sum())
    assert r.filled == red
    assert r.processed == r.filled
    assert r.processed_per_block.sum() == r.processed
    assert r.level_sizes.sum() == r.filled  # every entry dequeued once
    assert r.filled <= r.cas_attempts <= 8 * r.filled


@pytest.mark.parametrize("variant", VARIANTS)
def test_owner_census(variant):
    img, _ = SCENES["two_disks"]()
    r = flood_fill(img, variant=variant, blocks=4)
    reached = r.visited == 1
    owners = r.owner[reached]
    assert owners.min() >= 0 and owners.max() < r.blocks
    assert (r.owner[~reached] == -1).all()
    census = np.bincount(owners.astype(np.int64), minlength=r.blocks)
    np.testing.assert_array_equal(census, r.processed_per_block)


@pytest.mark.parametrize("variant", VARIANTS)
def test_model_bytes_consistency(variant):
    """The label/discovery traffic is PRICED now — the result's figure
    must be exactly the ch05 formula over the kernel's own counters."""
    img, _ = SCENES["two_disks"]()
    r = flood_fill(img, variant=variant)
    n = img.shape[0] * img.shape[1]
    expected = model_bytes_ch05(variant, n, r.filled, r.processed,
                                r.cas_attempts, r.union_attempts,
                                r.candidates, instrumented=True)
    assert r.model_bytes == expected
    assert r.model_gb_s > 0


# -------------------------------------------------------- lattice seeding

def assert_lat_matches_oracle(img, lattice, **gpu_kwargs):
    """v2 contract: exact against the lattice-widened oracle, canonical
    seeds unchanged from the corner rule."""
    ref_v, ref_d, ref_l, ref_levels, ref_filled = \
        cpu_fill_from_candidates(img, lattice)
    r = flood_fill(img, variant="seed_merge", lattice=lattice, **gpu_kwargs)
    np.testing.assert_array_equal(r.visited, ref_v)
    np.testing.assert_array_equal(r.label, ref_l)
    np.testing.assert_array_equal(r.depth, ref_d)
    assert r.levels == ref_levels
    assert r.filled == ref_filled
    vis = ref_v.astype(bool)
    np.testing.assert_array_equal(r.img[vis],
                                  PALETTE_HOST[ref_l[vis] % N_PALETTE])
    np.testing.assert_array_equal(r.img[~vis], img[~vis])
    assert r.lattice == lattice
    return r


def test_oracle_lattice_widens_the_candidate_set():
    img, n_blobs = SCENES["two_squares"]()
    corner = cpu_candidates(img)
    for S in (1, 5, 32):
        lat = cpu_candidates(img, S)
        assert (lat[corner == 1] == 1).all()      # superset of corner
        assert lat.sum() >= corner.sum()
    # S=1: every red pixel is a candidate
    np.testing.assert_array_equal(cpu_candidates(img, 1).astype(bool),
                                  _red_mask(img))
    # the lemma survives every stride: lex-min pixels stay candidates
    seed_mask = _canonical_seed_mask(img)
    for S in (1, 5, 32):
        assert (cpu_candidates(img, S)[seed_mask] == 1).all()


@pytest.mark.parametrize("lattice", [0, 1, 5, 32])
@pytest.mark.parametrize("name", ["two_squares", "two_disks", "u_shape",
                                  "comb", "blob_grid", "serpentine_like"])
def test_lattice_matches_oracle(name, lattice):
    if name == "serpentine_like":
        img, _ = scenes.serpentine_scene(48, 48)
    else:
        img, _ = SCENES[name]()
    assert_lat_matches_oracle(img, lattice)


def test_lattice_zero_is_bit_exact_with_v1():
    """S=0 isolates the v2 deltas that must be output-neutral (the
    compression pass and the rule refactor): identical everything."""
    for name in ("two_squares", "u_shape", "comb"):
        img, _ = SCENES[name]()
        v1 = flood_fill(img, variant="seed_merge")
        v2 = flood_fill(img, variant="seed_merge", lattice=0)
        np.testing.assert_array_equal(v1.img, v2.img)
        np.testing.assert_array_equal(v1.label, v2.label)
        np.testing.assert_array_equal(v1.depth, v2.depth)
        np.testing.assert_array_equal(v1.visited, v2.visited)
        assert v1.seeds == v2.seeds
        assert v1.candidates == v2.candidates


@pytest.mark.parametrize("lattice", [1, 5, 32])
def test_lattice_labels_and_seeds_are_stride_invariant(lattice):
    """The chapter's central claim, extended: seeding density changes the
    clock, never the answer."""
    for name in ("two_squares", "u_shape", "random_like"):
        if name == "random_like":
            img, _ = scenes.random_blobs_scene(96, 96, density=0.3,
                                               rng_seed=3)
        else:
            img, _ = SCENES[name]()
        r = flood_fill(img, variant="seed_merge", lattice=lattice)
        c = flood_fill(img, variant="ccl_fill")
        np.testing.assert_array_equal(r.label, c.label)
        assert r.seeds == c.seeds
        assert r.n_blobs == c.n_blobs


def test_lattice_one_degenerates_to_one_level():
    """S=1: every red pixel is its own wave — the ccl-like boundary."""
    img, n_blobs = SCENES["two_squares"]()
    r = flood_fill(img, variant="seed_merge", lattice=1)
    red = int(_red_mask(img).sum())
    assert r.candidates == red
    assert r.levels == 1
    assert (r.depth[r.visited == 1] == 0).all()
    assert r.union_done == r.candidates - n_blobs


def test_lattice_shortens_the_clock():
    img, _ = SCENES["two_squares"]()
    v1 = flood_fill(img, variant="seed_merge")
    v2 = flood_fill(img, variant="seed_merge", lattice=8)
    assert v2.levels < v1.levels
    assert v2.candidates > v1.candidates


def test_lattice_phase_keys_include_compress():
    img, _ = SCENES["u_shape"]()
    r = flood_fill(img, variant="seed_merge", lattice=8)
    assert tuple(r.phase_ms) == ("init", "scan", "fill", "compress",
                                 "flatten")
    assert all(v >= 0 for v in r.phase_ms.values())


def test_lattice_bare_twin_parity():
    img, _ = SCENES["blob_grid"]()
    inst = flood_fill(img, variant="seed_merge", lattice=8)
    bare = flood_fill(img, variant="seed_merge", lattice=8, bare=True)
    np.testing.assert_array_equal(inst.img, bare.img)
    np.testing.assert_array_equal(inst.label, bare.label)
    np.testing.assert_array_equal(inst.depth, bare.depth)
    assert inst.levels == bare.levels and inst.filled == bare.filled
    assert bare.phase_ms == {}


def test_lattice_validation():
    img, _ = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="lattice"):
        flood_fill(img, variant="ccl_fill", lattice=8)
    with pytest.raises(ValueError, match="lattice"):
        flood_fill(img, variant="seed_merge", lattice=-1)
    with pytest.raises(ValueError, match="lattice"):
        flood_fill(img, variant="seed_merge", lattice=2.5)


# ------------------------------------------------------------ phase timing

def test_phase_timing_reported():
    """tid-0 %globaltimer stamps at the phase barriers: variant-specific
    keys, non-negative wall times that nest inside the host-timed kernel
    wall (generous bound — this is a structure test, not a race with the
    clock)."""
    img, _ = SCENES["u_shape"]()
    expected = {"seed_merge": ("init", "scan", "fill", "flatten"),
                "ccl_fill": ("init", "union_merge", "flatten_seed", "fill")}
    for variant in VARIANTS:
        r = flood_fill(img, variant=variant)
        assert tuple(r.phase_ms) == expected[variant]
        assert all(v >= 0 for v in r.phase_ms.values())
        total = sum(r.phase_ms.values())
        assert 0 < total <= r.kernel_ms * 1.5 + 0.5
        # the fill dominates this scene in both variants
        assert r.phase_ms["fill"] > 0


def test_union_cycles_follow_the_collisions():
    """seed_merge's in-flight union clock only ticks where waves collide:
    zero on a scene with no cross-wave contact, positive on the U.
    ccl_fill never accumulates cycles — its unions ARE the union_merge
    phase, timed by wall clock instead."""
    img_sq, _ = SCENES["two_squares"]()
    r = flood_fill(img_sq, variant="seed_merge")
    assert r.union_attempts == 0
    assert r.union_cycles == 0 and r.union_thread_ms == 0.0

    img_u, _ = SCENES["u_shape"]()
    r = flood_fill(img_u, variant="seed_merge")
    assert r.union_attempts > 0
    assert r.union_cycles > 0 and r.union_thread_ms > 0.0

    r = flood_fill(img_u, variant="ccl_fill")
    assert r.union_cycles == 0
    assert r.phase_ms["union_merge"] >= 0


def test_bare_twin_reports_no_phase_timing():
    img, _ = SCENES["u_shape"]()
    for variant in VARIANTS:
        r = flood_fill(img, variant=variant, bare=True)
        assert r.phase_ms == {}
        assert r.union_cycles == 0 and r.union_thread_ms == 0.0


# --------------------------------------------------------------- bare twins

@pytest.mark.parametrize("variant", VARIANTS)
def test_bare_twin_matches_oracle(variant):
    img, _ = SCENES["two_squares"]()
    assert_matches_oracle(img, variant, bare=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bare_twin_reports_no_instrumentation(variant):
    img, _ = SCENES["two_squares"]()
    r = flood_fill(img, variant=variant, bare=True)
    assert r.bare
    assert r.processed == 0 and r.cas_attempts == 0
    assert r.candidates == 0 and r.union_attempts == 0 and r.union_done == 0
    assert r.owner.size == 0
    assert r.prov_label.size == 0
    assert r.model_bytes == 0 and r.model_gb_s == 0.0
    assert r.filled > 0 and r.levels > 0
    assert r.n_blobs == 2 and len(r.seeds) == 2  # label map still full


# ------------------------------------------------------------------ validation

def test_rejects_bad_variant():
    img, _ = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="variant"):
        flood_fill(img, variant="magic")


def test_rejects_bad_image():
    with pytest.raises(ValueError, match="uint8"):
        flood_fill(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        flood_fill(np.zeros((8, 8, 3), dtype=np.int32))


@pytest.mark.parametrize("tpb", [100, 0, 1024])
def test_rejects_bad_threads_per_block(tpb):
    img, _ = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="threads_per_block"):
        flood_fill(img, threads_per_block=tpb)


@pytest.mark.parametrize("blocks", [0, -1, 1.5])
def test_rejects_bad_block_count(blocks):
    img, _ = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="blocks"):
        flood_fill(img, blocks=blocks)


def test_rejects_blocks_beyond_cooperative_capacity():
    img, _ = SCENES["two_squares"]()
    coop_max = max_blocks(threads_per_block=256)
    with pytest.raises(RuntimeError, match="cooperative"):
        flood_fill(img, blocks=coop_max + 1)
