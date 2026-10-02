"""Correctness tests for the Triton twin of the seed-discovery fill.

Test for test the GPU half of chapters/ch05_gpu_nblob_nblock/
test_correctness.py, same names, scenes, parameters and assertions,
against the same CPU oracles (imported from the Numba chapter, never
copied). The oracle-only tests there (scene contract, candidate lemma,
oracle self-consistency) exercise no GPU code and are not repeated here.

The cross-backend section (test_cross_backend_*) runs the Numba driver
and the twin on the same images and asserts EXACT equality of every
deterministic output: img, visited, depth, label, n_blobs, seeds, filled,
levels and the deterministic counters, for every kernel (seed_merge,
ccl_fill, the lattice fused / bare / r128 / split builds, both interior
rules, the discovery-only phase kernels). Schedule-dependent values
(owner, prov_label at seams, smid, union_cycles, ccl_fill cas_attempts,
seed_merge union_attempts) are never compared.

Run:

    .venv/bin/python -m pytest -p no:cacheprovider \
        src/flood_fill_cuda/triton_twins/chapters/ch05_gpu_nblob_nblock -v
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import numpy as np
import pytest

from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock import scenes
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock import (
    flood_fill as numba_driver,
)
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.cpu_oracle import (
    cpu_candidates, cpu_fill_canonical, cpu_fill_from_candidates,
)
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.kernels import (
    PALETTE_HOST, N_PALETTE,
)
# The Numba suite's parametrization tables and mask helpers, shared so the
# two suites cannot drift apart
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.test_correctness import (
    SCENES, _ORACLES, _red_mask,
)
from flood_fill_cuda.triton_twins.chapters.ch05_gpu_nblob_nblock import (
    flood_fill as twin_driver,
)
from flood_fill_cuda.triton_twins.chapters.ch05_gpu_nblob_nblock.flood_fill import (
    VARIANTS, discovery_only, flood_fill, max_blocks, model_bytes_ch05,
)


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
    """100 disjoint blobs and a random shatter of hundreds - both far
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
    """A blank image is a valid question with answer zero - the kernel
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
    """Label, depth and paint are pure functions of the image - the
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
    seed) - labels, seeds and coverage may not."""
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
    uniqueness - the merge path must actually run."""
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
    exactly one - so union_done == filled - n_blobs, structurally."""
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
    """The label/discovery traffic is PRICED now - the result's figure
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
    """S=1: every red pixel is its own wave - the ccl-like boundary."""
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


# --------------------------------------------- interior rule + builds

@pytest.mark.parametrize("lattice", [1, 5, 32])
@pytest.mark.parametrize("name", ["two_squares", "two_disks", "u_shape",
                                  "single_pixel", "serpentine_like"])
def test_interior_matches_oracle(name, lattice):
    if name == "serpentine_like":
        img, _ = scenes.serpentine_scene(48, 48)
    else:
        img, _ = SCENES[name]()
    ref_v, ref_d, ref_l, ref_levels, ref_filled = \
        cpu_fill_from_candidates(img, lattice, 1)
    r = flood_fill(img, variant="seed_merge", lattice=lattice,
                   interior=True)
    np.testing.assert_array_equal(r.visited, ref_v)
    np.testing.assert_array_equal(r.label, ref_l)
    np.testing.assert_array_equal(r.depth, ref_d)
    assert r.levels == ref_levels and r.filled == ref_filled
    assert r.interior


@pytest.mark.parametrize("build", ["r128", "split"])
def test_builds_are_bit_exact_with_fused(build):
    """The register experiments change occupancy, never the answer."""
    for name, kw in (("two_squares", {}), ("u_shape", {}),
                     ("comb", {}), ("blob_grid", {"interior": True})):
        img, _ = SCENES[name]()
        fused = flood_fill(img, variant="seed_merge", lattice=16, **kw)
        other = flood_fill(img, variant="seed_merge", lattice=16,
                           build=build, **kw)
        np.testing.assert_array_equal(fused.img, other.img)
        np.testing.assert_array_equal(fused.label, other.label)
        np.testing.assert_array_equal(fused.depth, other.depth)
        np.testing.assert_array_equal(fused.visited, other.visited)
        # prov_label is deliberately NOT compared exactly: it records
        # which wave won each claim race, and at equidistant seam pixels
        # the winner is timing-dependent - different grids shuffle it.
        # Structural check instead: prov covers exactly the visited set.
        np.testing.assert_array_equal(other.prov_label >= 0,
                                      other.visited.astype(bool))
        assert fused.seeds == other.seeds
        assert fused.candidates == other.candidates
        assert other.build == build


def test_split_build_phase_keys_match_fused():
    img, _ = SCENES["u_shape"]()
    r = flood_fill(img, variant="seed_merge", lattice=16, build="split")
    assert tuple(r.phase_ms) == ("init", "scan", "fill", "compress",
                                 "flatten")
    assert r.phase_ms["compress"] >= 0 and r.phase_ms["flatten"] >= 0


def test_build_and_interior_validation():
    img, _ = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="build"):
        flood_fill(img, variant="seed_merge", lattice=16, build="turbo")
    with pytest.raises(ValueError, match="build"):
        flood_fill(img, variant="seed_merge", build="split")  # no lattice
    with pytest.raises(ValueError, match="instrumented-only"):
        flood_fill(img, variant="seed_merge", lattice=16, build="split",
                   bare=True)
    with pytest.raises(ValueError, match="interior"):
        flood_fill(img, variant="seed_merge", interior=True)  # no lattice
    with pytest.raises(ValueError, match="interior"):
        flood_fill(img, variant="seed_merge", lattice=0, interior=True)


# ------------------------------------------------------------ phase timing

def test_phase_timing_reported():
    """tid-0 %globaltimer stamps at the phase barriers: variant-specific
    keys, non-negative wall times that nest inside the host-timed kernel
    wall (generous bound - this is a structure test, not a race with the
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
    ccl_fill never accumulates cycles - its unions ARE the union_merge
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


# ---------------------------------------------------- twin-only validation

@pytest.mark.parametrize("tpb", [96, 160, 480])
def test_twin_rejects_non_power_of_two_threads_per_block(tpb):
    """Numba accepts any multiple of 32 in [32, 512]; a Triton program's
    lane count must be a power of 2, and the error says so."""
    img, _ = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="threads_per_block must be a "
                                         "power of 2"):
        flood_fill(img, threads_per_block=tpb)
    with pytest.raises(ValueError, match="power of 2"):
        discovery_only(img, threads_per_block=tpb)


@pytest.mark.parametrize("kw", [dict(variant="seed_merge"),
                                dict(variant="ccl_fill", bare=True),
                                dict(variant="seed_merge", lattice=8,
                                     build="split")])
def test_twin_rejects_int32_grid_stride_overflow(kw):
    """The twin's grid-stride indices are int32 (Numba's are int64), so
    an image Numba would accept (width*height < 2**31) is refused when its
    last stride would wrap. A zero-copy broadcast image: the guard runs
    before any buffer is allocated."""
    img = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8),
                          (2 ** 31 - 1, 1, 3))
    with pytest.raises(ValueError, match="int32"):
        flood_fill(img, **kw)
    with pytest.raises(ValueError, match="int32"):
        discovery_only(img, variant=kw["variant"])


def _n_compiled(fn):
    """Specializations Triton holds for a @triton.jit kernel."""
    return sum(len(cache[0]) for cache in fn.device_caches.values())


@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("variant", VARIANTS)
def test_twin_scene_size_never_recompiles(variant, bare):
    """Every runtime int is do_not_specialize, so after the warm-up no
    scene size (1x1, odd, multiples of 16) can put a compile inside the
    timed kernel_ms window."""
    flood_fill(SCENES["two_squares"]()[0], variant=variant, bare=bare)
    fn = twin_driver._KERNELS[(variant, bare)].fn
    before = _n_compiled(fn)
    one = np.full((1, 1, 3), 255, dtype=np.uint8)
    one[0, 0] = (255, 0, 0)
    for img in (one, scenes.square_scene(37, 23, 5, 7)[0],
                scenes.square_scene(64, 48, 16, 16)[0],
                scenes.disk_scene(33, 65, 12)[0]):
        assert_matches_oracle(img, variant, bare=bare)
    assert _n_compiled(fn) == before


@pytest.mark.parametrize("variant", VARIANTS)
def test_twin_bare_compiles_without_instrumentation(variant):
    """INSTR=False removes the observer code at compile time, like the
    separate Numba *_bare_kernel: no timer, %smid or %clock64 reads."""
    inst = twin_driver.compiled_kernel(variant, 256, bare=False).asm["ptx"]
    bare = twin_driver.compiled_kernel(variant, 256, bare=True).asm["ptx"]
    assert "%globaltimer" in inst and "%smid" in inst
    assert ("%clock64" in inst) == (variant == "seed_merge")
    for reg in ("%globaltimer", "%smid", "%clock64"):
        assert reg not in bare, reg


# lattice twins: kernel key -> (fn, extra plain kernels of the build)
_LAT_FNS = {
    "fused": (twin_driver.seed_merge_lat_kernel, ()),
    "r128": (twin_driver.seed_merge_lat_kernel, ()),
    "split": (twin_driver.seed_merge_lat_core_kernel,
              (twin_driver.lat_compress_kernel,
               twin_driver.lat_finish_kernel)),
}


@pytest.mark.parametrize("build", ["fused", "r128", "split"])
def test_twin_lattice_stride_never_recompiles(build):
    """lat_stride and lat_interior are do_not_specialize too: after the
    warm-up, no stride (1 and multiples of 16 included), interior rule or
    scene size compiles again - tuning.py's sweep stays off the
    compiler."""
    flood_fill(SCENES["two_squares"]()[0], variant="seed_merge", lattice=4,
               build=build)
    fn, extra = _LAT_FNS[build]
    before = [_n_compiled(f) for f in (fn,) + extra]
    for img in (SCENES["u_shape"]()[0], scenes.square_scene(37, 23, 5, 7)[0],
                scenes.disk_scene(33, 65, 12)[0]):
        for lattice in (0, 1, 16, 32, 256):
            for interior in ((False, True) if lattice else (False,)):
                ref_v, ref_d, ref_l, ref_levels, _ = cpu_fill_from_candidates(
                    img, lattice, int(interior))
                r = flood_fill(img, variant="seed_merge", lattice=lattice,
                               interior=interior, build=build)
                np.testing.assert_array_equal(r.label, ref_l)
                np.testing.assert_array_equal(r.depth, ref_d)
                assert r.levels == ref_levels
    assert [_n_compiled(f) for f in (fn,) + extra] == before


@pytest.mark.parametrize("tpb", [128, 256])
def test_twin_r128_honours_the_register_cap(tpb):
    """r128 is the fused body compiled with maxnreg=128 (Numba's
    max_registers=128): the cap holds, and r128 is a separate compile of
    the same kernel (its capacity is queried on its own compiled object)."""
    fused = twin_driver.kernel_info("seed_merge", tpb, lattice=4,
                                    build="fused")
    r128 = twin_driver.kernel_info("seed_merge", tpb, lattice=4,
                                   build="r128")
    assert r128["n_regs"] <= 128
    assert r128["coop_max_blocks"] >= 1 and fused["coop_max_blocks"] >= 1
    assert (twin_driver.compiled_kernel("seed_merge", tpb, lattice=4,
                                        build="r128")
            is not twin_driver.compiled_kernel("seed_merge", tpb, lattice=4,
                                               build="fused"))


def test_twin_lattice_bare_compiles_without_instrumentation():
    """The lattice bare twin drops the observer code like the Numba
    seed_merge_lat_bare_kernel; the split core keeps it (instrumented
    only, like Numba), the plain cleanup kernels never had any."""
    asm = {key: twin_driver.compiled_kernel("seed_merge", 256, bare=bare,
                                            lattice=4,
                                            build=build).asm["ptx"]
           for key, bare, build in (("inst", False, "fused"),
                                    ("bare", True, "fused"),
                                    ("core", False, "split"))}
    for reg in ("%globaltimer", "%smid", "%clock64"):
        assert reg in asm["inst"] and reg in asm["core"], reg
        assert reg not in asm["bare"], reg
    for fn in (twin_driver.lat_compress_kernel,
               twin_driver.lat_finish_kernel):
        for cache in fn.device_caches.values():
            for compiled in cache[0].values():
                for reg in ("%globaltimer", "%smid", "%clock64"):
                    assert reg not in compiled.asm["ptx"], reg


# ============================================== cross-backend (Numba == Triton)

def _cross_scene(name):
    if name == "random":
        return scenes.random_blobs_scene(128, 128, density=0.3,
                                         rng_seed=7)[0]
    if name == "serpentine":
        return scenes.serpentine_scene(48, 48)[0]
    return SCENES[name]()[0]


_CROSS_SCENES = ("two_squares", "two_disks", "u_shape", "comb", "blob_grid",
                 "full_red", "random", "serpentine")


def assert_backends_agree(a, b, pinned=False):
    """a = Numba result, b = Triton result of the same call. Asserts
    exact equality of everything that is a pure function of the image
    (and, when pinned, of blocks + threads_per_block)."""
    for name in ("img", "visited", "depth", "label"):
        np.testing.assert_array_equal(getattr(b, name), getattr(a, name),
                                      err_msg=name)
    for name in ("variant", "lattice", "interior", "build",
                 "threads_per_block", "bare", "n_blobs", "seeds", "filled",
                 "levels"):
        assert getattr(b, name) == getattr(a, name), name
    assert tuple(b.phase_ms) == tuple(a.phase_ms)
    if a.bare:
        for name in ("processed", "cas_attempts", "candidates",
                     "union_attempts", "union_done", "union_cycles",
                     "model_bytes"):
            assert getattr(b, name) == getattr(a, name) == 0, name
        assert b.owner.size == a.owner.size == 0
        assert b.prov_label.size == a.prov_label.size == 0
    else:
        # level sizes are race-free, so every level statistic is too
        for name in ("candidates", "union_done", "processed", "peak_level",
                     "peak_occupancy", "level_trace_truncated"):
            assert getattr(b, name) == getattr(a, name), name
        np.testing.assert_array_equal(b.level_sizes, a.level_sizes)
        if a.variant == "seed_merge":
            # paint is deferred, so img is constant during the fill: every
            # dequeued pixel probes the same red neighbours on both sides
            assert b.cas_attempts == a.cas_attempts
            np.testing.assert_array_equal(b.prov_label >= 0,
                                          a.prov_label >= 0)
        else:
            # one attempt per (red pixel, red lex-predecessor) pair
            assert b.union_attempts == a.union_attempts
    if pinned:
        # same blocks and lanes: the grid-stride assignment of queue slots
        # to blocks depends only on the (race-free) level sizes
        assert b.blocks == a.blocks
        assert b.thread_util_pct == a.thread_util_pct
        if not a.bare:
            np.testing.assert_array_equal(b.processed_per_block,
                                          a.processed_per_block)


@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", _CROSS_SCENES)
def test_cross_backend_outputs_equal(name, variant, bare):
    img = _cross_scene(name)
    a = numba_driver.flood_fill(img, variant=variant, bare=bare)
    b = flood_fill(img, variant=variant, bare=bare)
    assert_backends_agree(a, b)


@pytest.mark.parametrize("tpb, blocks", [(64, 4), (256, 3), (32, 7)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_cross_backend_pinned_grid(variant, tpb, blocks):
    """With blocks and threads_per_block pinned identical, the per-block
    work split and the thread utilization match too."""
    for name in ("u_shape", "random"):
        img = _cross_scene(name)
        a = numba_driver.flood_fill(img, variant=variant,
                                    threads_per_block=tpb, blocks=blocks)
        b = flood_fill(img, variant=variant, threads_per_block=tpb,
                       blocks=blocks)
        assert_backends_agree(a, b, pinned=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_cross_backend_model_bytes_where_deterministic(variant):
    """model_bytes contains seed_merge's union_attempts and ccl_fill's
    cas_attempts (both schedule-dependent), except on seed_merge scenes
    with no wave collisions: there it is identical across backends."""
    img = _cross_scene("two_squares")
    a = numba_driver.flood_fill(img, variant=variant)
    b = flood_fill(img, variant=variant)
    if variant == "seed_merge":
        assert a.union_attempts == b.union_attempts == 0
        assert b.model_bytes == a.model_bytes
    else:
        n = img.shape[0] * img.shape[1]
        assert b.model_bytes == model_bytes_ch05(
            variant, n, b.filled, b.processed, b.cas_attempts,
            b.union_attempts, b.candidates, instrumented=True)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", ("two_disks", "comb", "random",
                                  "serpentine"))
def test_cross_backend_discovery_only_candidates(name, variant):
    """The benchmark phase kernels discover the same queue: the candidate
    count (seed_merge) or n_blobs (ccl_fill)."""
    img = _cross_scene(name)
    _, cand_numba = numba_driver.discovery_only(img, variant=variant)
    _, cand_triton = discovery_only(img, variant=variant)
    assert cand_triton == cand_numba
    if variant == "seed_merge":
        assert cand_triton == int(cpu_candidates(img).sum())


# (lattice, interior, build, bare): every lattice kernel of the Numba
# driver, both P1 rules, the degenerate strides
_LAT_CONFIGS = (
    (0, False, "fused", False), (1, False, "fused", False),
    (5, False, "fused", False), (8, True, "fused", False),
    (32, False, "fused", False),
    (8, False, "fused", True), (1, True, "fused", True),
    (1, False, "r128", False), (8, True, "r128", False),
    (16, False, "split", False), (1, True, "split", False),
)
_LAT_CROSS_SCENES = ("two_disks", "u_shape", "comb", "random", "serpentine")


@pytest.mark.parametrize("lattice, interior, build, bare", _LAT_CONFIGS)
@pytest.mark.parametrize("name", _LAT_CROSS_SCENES)
def test_cross_backend_lattice_outputs_equal(name, lattice, interior, build,
                                             bare):
    img = _cross_scene(name)
    kw = dict(variant="seed_merge", lattice=lattice, interior=interior,
              build=build, bare=bare)
    a = numba_driver.flood_fill(img, **kw)
    b = flood_fill(img, **kw)
    assert_backends_agree(a, b)


@pytest.mark.parametrize("tpb, blocks", [(64, 4), (256, 3), (32, 7)])
@pytest.mark.parametrize("build", ["fused", "r128", "split"])
def test_cross_backend_lattice_pinned_grid(build, tpb, blocks):
    """Pinned blocks and lanes: the lattice builds split the work per
    block exactly like Numba too."""
    for name in ("u_shape", "random"):
        img = _cross_scene(name)
        kw = dict(variant="seed_merge", lattice=8, build=build,
                  threads_per_block=tpb, blocks=blocks)
        a = numba_driver.flood_fill(img, **kw)
        b = flood_fill(img, **kw)
        assert_backends_agree(a, b, pinned=True)


@pytest.mark.parametrize("build", ["fused", "split"])
def test_cross_backend_lattice_one_prov_label_exact(build):
    """lattice=1 without interior: every red pixel is its own candidate,
    so prov_label is each pixel's own index - deterministic, so equal."""
    img = _cross_scene("random")
    a = numba_driver.flood_fill(img, variant="seed_merge", lattice=1,
                                build=build)
    b = flood_fill(img, variant="seed_merge", lattice=1, build=build)
    np.testing.assert_array_equal(b.prov_label, a.prov_label)
    lin = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[:2])
    vis = b.visited == 1
    np.testing.assert_array_equal(b.prov_label[vis], lin[vis])


def test_cross_backend_lattice_capacity_story():
    """Numba's register story (fused over the 128-register line, so fewer
    cooperative blocks than r128 and split) is a property of Numba's
    compiler. Each backend's capacity is its own; both are queried per
    compiled kernel, every build's blocks=None launch is the capacity it
    reports, and at those (possibly different) grids the two backends
    still produce the same deterministic outputs."""
    img = _cross_scene("u_shape")
    for build in ("fused", "r128", "split"):
        results = []
        for drv in (numba_driver, twin_driver):
            cap = drv.max_blocks("seed_merge", 256, lattice=4, build=build)
            r = drv.flood_fill(img, variant="seed_merge", lattice=4,
                               build=build)
            assert r.blocks == cap >= 1, (drv.__name__, build)
            results.append(r)
        assert_backends_agree(*results)


def test_cross_backend_image_layout_rules():
    """Both drivers take a C- or F-ordered image (and agree on it), and
    both refuse a non-contiguous view with cuda.to_device's ValueError
    (the twin applies Numba's own check before its C-order upload)."""
    img = _cross_scene("u_shape")
    fortran = np.asfortranarray(img)
    for variant in VARIANTS:
        assert_backends_agree(numba_driver.flood_fill(fortran, variant=variant),
                              flood_fill(fortran, variant=variant))
    strided = np.repeat(img, 2, axis=0)[::2]
    transposed = np.ascontiguousarray(img.transpose(1, 0, 2)).transpose(1, 0, 2)
    for view in (strided, transposed):
        assert not (view.flags.c_contiguous or view.flags.f_contiguous)
        for fn in (numba_driver.flood_fill, flood_fill,
                   numba_driver.discovery_only, discovery_only):
            with pytest.raises(ValueError, match="non-contiguous"):
                fn(view)


def test_cross_backend_result_fields_are_the_numba_dataclass():
    """Same dataclass, so the same field names and meanings."""
    img = _cross_scene("u_shape")
    r = flood_fill(img)
    assert type(r) is numba_driver.SeedDiscoveryResult
