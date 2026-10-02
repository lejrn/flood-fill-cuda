"""Correctness tests: the Triton dual-blob twin vs the merged CPU oracle,
and vs the Numba chapter.

The first part mirrors the Numba chapter's test_correctness.py test for
test (same names, scenes, parameters and assertions, against the same
oracle and scene builders, imported from the Numba chapter). Two Numba
tests are not repeated because they never touch a GPU:
test_scenes_reject_small_gap and test_scene_seeds_are_red_and_disjoint
check the shared scene builders and the CPU oracle only.

The twin-only tests (test_twin_*) cover what Numba has no counterpart
for: the power-of-2 lane rule, the streams pair rail, no recompiles inside
a timed launch, the 64-bit grid barrier and bare binaries without the
instrumentation code.

The cross-backend part (test_cross_backend_*) runs the Numba kernels and
their Triton twins on the same scene at the same pinned grid and requires
the deterministic outputs to be equal: images, visited/depth/label maps,
every counter whose value does not depend on scheduling, the per-level
trace and the per-program work census. Never compared: owner maps (except
at one program), sm_ids, timings, and cas_attempts at 8-conn / radius 2
(same-level diagonal neighbors may or may not be painted yet when probed).

Streams-mode tests stay OPT-IN (DUAL_BLOB_STREAMS=1), exactly as in Numba:
two concurrent cooperative grids can wedge the GPU. The direct
streams-vs-streams comparison is separately opt-in
(DUAL_BLOB_STREAMS_NUMBA=1): it runs Numba's pair in a fresh process under
a hard timeout.

Run:

    .venv/bin/python -m pytest -p no:cacheprovider \
        src/flood_fill_cuda/triton_twins/chapters/ch04_gpu_2blob_nblock/test_correctness.py
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import numpy as np
import pytest

from flood_fill_cuda.shared import bandwidth
from flood_fill_cuda.chapters.ch04_gpu_2blob_nblock import scenes
from flood_fill_cuda.chapters.ch04_gpu_2blob_nblock.cpu_oracle import (
    cpu_flood_fill_two,
)
from .flood_fill import flood_fill, max_blocks, MODES
from . import flood_fill as twin_ff, kernels as twin_kernels

BLUE = np.array([0, 0, 255], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)

STREAMS_TEST_BLOCKS = 8   # small pair; still not a guarantee - see below

# Streams-mode tests are OPT-IN (DUAL_BLOB_STREAMS=1), for the Numba
# suite's reason: two concurrent cooperative grids can wedge forever at
# the grid barrier, nondeterministically (it cost the Numba suite 82
# minutes of dead spin once). A test that can hang the runner forever is
# worse than an unrun test.
RUN_STREAMS = os.environ.get("DUAL_BLOB_STREAMS") == "1"
TEST_MODES = MODES if RUN_STREAMS else tuple(m for m in MODES
                                             if m != "streams")
requires_streams = pytest.mark.skipif(
    not RUN_STREAMS,
    reason="streams mode can deadlock the GPU; set DUAL_BLOB_STREAMS=1 to run")

# Numba's OWN streams pair, for a direct streams-vs-streams comparison,
# runs only in a fresh process under a hard timeout: the Numba README's
# fresh-process probe (an 8+8 pair ran there, while the same pair wedged
# after other fills in one process). Separately opt-in, because a wedge
# still costs the timeout.
RUN_NUMBA_STREAMS = os.environ.get("DUAL_BLOB_STREAMS_NUMBA") == "1"
requires_numba_streams = pytest.mark.skipif(
    not RUN_NUMBA_STREAMS,
    reason="Numba's streams pair can wedge the GPU; set "
           "DUAL_BLOB_STREAMS_NUMBA=1 to run it in a fresh process under a "
           "hard timeout")
NUMBA_STREAMS_TIMEOUT_S = 120


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
    """The xy entry format must be bit-for-bit output-equivalent to lin:
    only the in-queue encoding differs."""
    img, seeds = SCENES["two_squares"]()
    r_xy = assert_matches_reference(img, seeds, mode, entry_format="xy")
    r_lin = flood_fill(img, seeds, mode=mode, **_kw(mode, {}))
    np.testing.assert_array_equal(r_xy.img, r_lin.img)
    np.testing.assert_array_equal(r_xy.depth, r_lin.depth)
    np.testing.assert_array_equal(r_xy.label, r_lin.label)


def test_mode_equivalence():
    """The modes are the same algorithm run different ways: identical
    img/visited/depth/label (owner excluded: which program claims a pixel
    is race- and mode-dependent). Covers streams only when
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
    """blocks=None launches the full co-resident grid: the case where a
    wrong capacity would hang a spin barrier instead of raising."""
    img, seeds = SCENES["two_squares"]()
    assert_matches_reference(img, seeds, "multisource",
                             threads_per_block=tpb, blocks=blocks)


def test_multisource_levels_is_max_of_blobs():
    """One shared clock: the combined level count is the slower blob's,
    not the sum."""
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
    to seed 1's, even at the minimum legal gap."""
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
    capacity. Run at tpb=512, where capacity is smallest. (The Triton
    twin's capacity is larger than Numba's, so this pair is ~24+24
    programs here, not ~8+8.)"""
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


# ------------------------------------------------- radius-2 twins (guarded)

# The guarded radius-2 twins share conn8's fill set AND label map, but not
# its depth/levels, so these assert visited/label/filled/recolor.


def assert_r2_fill_and_labels(img, seeds, mode, **gpu_kwargs):
    (ref_v, _, ref_l, _, ref_filled,
     (_, fa), (_, fb)) = cpu_flood_fill_two(img, seeds, 8)
    r = flood_fill(img, seeds, mode=mode, connectivity=8, radius=2,
                   **_kw(mode, gpu_kwargs))
    np.testing.assert_array_equal(r.visited, ref_v)
    np.testing.assert_array_equal(r.label, ref_l)
    assert r.filled == ref_filled
    assert r.filled_a == fa and r.filled_b == fb
    assert (r.img[ref_l == 0] == BLUE).all()
    assert (r.img[ref_l == 1] == GREEN).all()
    untouched = ref_v == 0
    np.testing.assert_array_equal(r.img[untouched], img[untouched])
    return r


@pytest.mark.parametrize("mode", TEST_MODES)
@pytest.mark.parametrize("name", ["two_squares", "asym_squares", "min_gap"])
def test_r2_fill_and_labels(name, mode):
    img, seeds = SCENES[name]()
    assert_r2_fill_and_labels(img, seeds, mode)


def test_r2_labels_isolated_across_1px_gap():
    """Two blobs one white row apart (tighter than the scene builders'
    >= 2 px contract, so built by hand). An unguarded ring-2 would jump
    the gap and smear one blob's label onto the other; the guard must
    not."""
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    img[10:20, :] = scenes.RED   # blob A
    img[21:31, :] = scenes.RED   # blob B, one white row away
    seeds = [(15, 30), (25, 30)]
    r = flood_fill(img, seeds, mode="multisource", connectivity=8, radius=2)
    assert (r.label[10:20, :] == 0).all()
    assert (r.label[21:31, :] == 1).all()
    r8 = flood_fill(img, seeds, mode="multisource", connectivity=8)
    np.testing.assert_array_equal(r.visited, r8.visited)
    np.testing.assert_array_equal(r.label, r8.label)


def test_r2_fewer_levels_and_interior():
    """Supergraph BFS: fewer levels, no pixel reached later than conn8;
    the static guard makes interior an exact census: two 30x30 solid
    squares have 28x28 interior cores each."""
    img, seeds = SCENES["two_squares"]()
    r2 = flood_fill(img, seeds, mode="multisource", connectivity=8, radius=2)
    r8 = flood_fill(img, seeds, mode="multisource", connectivity=8)
    assert r2.levels < r8.levels
    reached = r8.visited == 1
    assert (r2.depth[reached] <= r8.depth[reached]).all()
    assert r2.interior == 2 * 28 * 28
    assert r2.filled - 2 <= r2.cas_attempts <= 24 * r2.filled
    assert r2.processed == r2.filled


def test_r2_bare_twin_parity():
    img, seeds = SCENES["two_squares"]()
    inst = flood_fill(img, seeds, mode="multisource", connectivity=8,
                      radius=2)
    bare = flood_fill(img, seeds, mode="multisource", connectivity=8,
                      radius=2, bare=True)
    np.testing.assert_array_equal(inst.visited, bare.visited)
    np.testing.assert_array_equal(inst.depth, bare.depth)
    np.testing.assert_array_equal(inst.label, bare.label)
    assert inst.levels == bare.levels and inst.filled == bare.filled


def test_r2_model_bytes_consistency():
    """r2's model uses the exact probe count 8*processed + 16*interior."""
    img, seeds = SCENES["two_squares"]()
    r = flood_fill(img, seeds, mode="multisource", connectivity=8, radius=2)
    expected = bandwidth.model_bytes(
        r.processed, r.cas_attempts, r.filled, instrumented=True, n_dirs=8,
        probe_reads=8 * r.processed + 16 * r.interior)
    assert r.model_bytes == expected


@pytest.mark.parametrize("kwargs", [
    {"connectivity": 4, "radius": 2},
    {"connectivity": 8, "radius": 2, "entry_format": "xy"},
    {"connectivity": 8, "radius": 3},
])
def test_rejects_bad_radius(kwargs):
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="radius"):
        flood_fill(img, seeds, **kwargs)


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


# ------------------------------------------------- twin-only rules (Triton)

@pytest.mark.parametrize("tpb", [96, 160, 480])
def test_twin_rejects_non_power_of_two_threads_per_block(tpb):
    """Numba accepts any multiple of 32 in [32, 512]; Triton needs powers
    of 2 (tl.arange, num_warps), and the error says so."""
    img, seeds = SCENES["two_squares"]()
    with pytest.raises(ValueError, match="threads_per_block.*power of 2"):
        flood_fill(img, seeds, threads_per_block=tpb)


def test_twin_streams_rejects_pair_beyond_capacity():
    """The twin refuses an explicit streams grid whose PAIR cannot be
    co-resident. Raised before any launch, so it is safe to run without
    DUAL_BLOB_STREAMS."""
    img, seeds = SCENES["two_squares"]()
    coop = max_blocks(threads_per_block=256)
    with pytest.raises(RuntimeError, match="cooperative"):
        flood_fill(img, seeds, mode="streams", blocks=coop // 2 + 1)


@pytest.mark.parametrize("tpb", [32, 128, 512])
@pytest.mark.parametrize("key", list(twin_ff._KERNELS),
                         ids=lambda k: twin_ff._KERNELS[k].__name__)
def test_twin_no_recompile_inside_the_timed_launch(key, tpb):
    """The warm-up compiles the exact binary the timed launch uses: image
    sizes of every divisibility class, and n_seeds 1 (sequential) or 2
    (multisource), reuse the one specialization per (kernel, tpb), so no
    compile can land inside kernel_ms."""
    entry_format, bare, connectivity, radius = key
    kw = dict(bare=bare, connectivity=connectivity,
              entry_format=entry_format, radius=radius)
    kernel = twin_ff._KERNELS[key]

    def compiled():
        return sum(len(c[0]) for c in kernel.device_caches.values())

    flood_fill(*SCENES["two_squares"](), threads_per_block=tpb, **kw)
    before = compiled()
    assert before >= 1
    for w, h in [(101, 37), (96, 160), (33, 1), (200, 130)]:
        img, seeds = _stripes(w, h)
        for mode in ("sequential", "multisource"):
            flood_fill(img, seeds, mode=mode, threads_per_block=tpb,
                       blocks=3, **kw)
    assert compiled() == before


@pytest.mark.parametrize("key", list(twin_ff._KERNELS),
                         ids=lambda k: twin_ff._KERNELS[k].__name__)
def test_twin_grid_barrier_is_64_bit(key):
    """The barrier counter and its target are int64: both arrivals per
    level are 64-bit release adds and the spin is a 64-bit acquire load,
    so 2 * levels * programs arrivals cannot wrap."""
    import re

    assert twin_ff.BAR_DTYPE == np.int64
    ptx = twin_ff._warmup(*key, threads_per_block=256).asm["ptx"]
    assert len(re.findall(r"atom\.global\.gpu\.release\.add\.u64", ptx)) == 2
    assert not re.search(r"release\.add\.[us]32", ptx)
    spins = re.findall(r"ld\.global\.gpu\.acquire\.(\w+)", ptx)
    assert spins and set(spins) == {"b64"}


@pytest.mark.parametrize("key", list(twin_ff._KERNELS),
                         ids=lambda k: twin_ff._KERNELS[k].__name__)
def test_twin_bare_kernels_compile_without_instrumentation(key):
    """INSTR is a constexpr: a bare twin's binary has no %smid read and no
    int16 owner store, and its only 64-bit relaxed atomic adds are none
    (the instrumented twins flush their int64 counters with them)."""
    import re

    ptx = twin_ff._warmup(*key, threads_per_block=256).asm["ptx"]
    bare = key[1]
    has_smid = "%smid" in ptx
    has_owner_store = re.search(r"st\.global(\.\w+)*\.b16", ptx) is not None
    counter_atomics = re.findall(r"atom\.global\.gpu\.relaxed\.add\.u64", ptx)
    if bare:
        assert not has_smid and not has_owner_store and not counter_atomics
    else:
        assert has_smid and has_owner_store and counter_atomics


# ====================================================== Numba vs Triton
#
# Same scene, same variant, same (blocks, tpb): every deterministic output
# must be identical. Which lane handles queue index i depends only on
# (i - front) and the stride, so with the grid pinned even the per-program
# work census and thread utilisation agree exactly. Never compared: owner
# maps (except at one program), sm_ids, timings, and cas_attempts at
# 8-conn / radius 2 (same-level diagonal neighbors may or may not be
# painted yet when probed).

from flood_fill_cuda.chapters.ch04_gpu_2blob_nblock import (  # noqa: E402
    flood_fill as numba_ff, kernels as numba_kernels,
)

VARIANTS = list(twin_ff._KERNELS)
VARIANT_IDS = [twin_ff._KERNELS[k].__name__ for k in VARIANTS]
INSTRUMENTED = [k for k in VARIANTS if not k[1]]
INSTRUMENTED_IDS = [twin_ff._KERNELS[k].__name__ for k in INSTRUMENTED]


def _variant_kw(variant):
    entry_format, bare, connectivity, radius = variant
    return dict(bare=bare, connectivity=connectivity,
                entry_format=entry_format, radius=radius)


def _pinned(blocks, tpb, kw):
    """The requested grid, capped at both backends' capacity."""
    return min(blocks, numba_ff.max_blocks(threads_per_block=tpb, **kw),
               max_blocks(threads_per_block=tpb, **kw))


def _stripes(w, h, along_x=True):
    """An all-red w x h image cut in two by one white line (a column at
    x = w // 2, or a row at y = h // 2), with the seeds in opposite
    corners. The line is 1 px wide: tighter than the scene builders'
    >= 2 px gap, still two components at 8-connectivity."""
    img = np.empty((w, h, 3), dtype=np.uint8)
    img[:, :] = scenes.RED
    if along_x:
        img[w // 2, :] = (255, 255, 255)
    else:
        img[:, h // 2] = (255, 255, 255)
    return img, [(0, 0), (w - 1, h - 1)]


def _cas_is_deterministic(r):
    """4-conn radius-1: each component is bipartite with one seed, so a
    neighbor one level closer is already painted (a barrier earlier) and
    one level further is still red: the CAS count is fixed by the depth
    map. 8-conn and radius 2 race on same-level diagonals."""
    return r.connectivity == 4 and r.radius == 1


LAUNCH_FIELDS = ("label", "blocks", "filled", "levels", "peak_level",
                 "peak_occupancy", "processed", "interior", "thread_util_pct",
                 "level_trace_truncated")


def assert_same_launches(rn, rt):
    assert len(rt.launches) == len(rn.launches)
    for ln, lt in zip(rn.launches, rt.launches):
        for name in LAUNCH_FIELDS:
            assert getattr(lt, name) == getattr(ln, name), name
        np.testing.assert_array_equal(lt.processed_per_block,
                                      ln.processed_per_block)
        np.testing.assert_array_equal(lt.level_sizes, ln.level_sizes)
        assert lt.processed_per_block.dtype == ln.processed_per_block.dtype
        assert lt.level_sizes.dtype == ln.level_sizes.dtype
        if _cas_is_deterministic(rn) or rn.bare:
            assert lt.cas_attempts == ln.cas_attempts
        if rn.bare:
            assert lt.sm_ids == ln.sm_ids == [] and lt.distinct_sms == 0


def assert_same_deterministic(rn, rt):
    """Every deterministic output of a Numba result equals the Triton one."""
    for name in ("img", "visited", "depth", "label"):
        a, b = getattr(rn, name), getattr(rt, name)
        np.testing.assert_array_equal(b, a, err_msg=name)
        assert b.dtype == a.dtype, name
    for name in ("mode", "seeds", "threads_per_block", "blocks", "bare",
                 "connectivity", "radius", "entry_format", "filled",
                 "filled_a", "filled_b", "levels", "levels_a", "levels_b",
                 "processed", "interior", "overlap_ratio"):
        assert getattr(rt, name) == getattr(rn, name), name
    assert rt.owner.shape == rn.owner.shape
    assert rt.owner.dtype == rn.owner.dtype
    if _cas_is_deterministic(rn) or rn.bare:
        assert rt.cas_attempts == rn.cas_attempts
        assert rt.model_bytes == rn.model_bytes
    assert_same_launches(rn, rt)
    if not rn.bare:
        # the per-program owner census (not the map) is deterministic, and
        # every unreached pixel keeps owner -1
        # (summed over the launches: they share the owner map and the grid)
        reached = rn.visited == 1
        ppb = sum(l.processed_per_block for l in rt.launches)
        for r in (rn, rt):
            assert (r.owner[~reached] == -1).all()
            np.testing.assert_array_equal(
                np.bincount(r.owner[reached].astype(np.int64),
                            minlength=rn.blocks), ppb)


def _both(img, seeds, **kw):
    return (numba_ff.flood_fill(img, seeds, **kw),
            flood_fill(img, seeds, **kw))


@pytest.mark.parametrize("mode", ["sequential", "multisource"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_cross_backend_every_variant_matches_numba(variant, mode):
    """All ten twins, both non-streams modes, a pinned small grid."""
    img, seeds = SCENES["asym_squares"]()
    rn, rt = _both(img, seeds, mode=mode, threads_per_block=64, blocks=3,
                   **_variant_kw(variant))
    assert_same_deterministic(rn, rt)


@pytest.mark.parametrize("grid", [(5, 32), (7, 128), (2, 512)],
                         ids=lambda g: f"{g[0]}x{g[1]}")
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_cross_backend_every_lane_count_matches_numba(variant, grid):
    """The other lane counts the twin accepts (tpb=32 is the one-warp
    program), multisource on the disks."""
    kw = _variant_kw(variant)
    blocks = _pinned(grid[0], grid[1], kw)
    img, seeds = SCENES["two_disks"]()
    rn, rt = _both(img, seeds, mode="multisource", threads_per_block=grid[1],
                   blocks=blocks, **kw)
    assert_same_deterministic(rn, rt)


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_cross_backend_full_shared_grid_matches_numba(variant):
    """The largest grid both backends can host for this twin at tpb=256
    (Numba's own blocks=None default whenever Triton's capacity is the
    larger one)."""
    kw = _variant_kw(variant)
    blocks = _pinned(10 ** 6, 256, kw)
    img, seeds = SCENES["two_disks"]()
    rn, rt = _both(img, seeds, mode="multisource", threads_per_block=256,
                   blocks=blocks, **kw)
    assert_same_deterministic(rn, rt)


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_cross_backend_at_scale_matches_numba(variant):
    """Two 0.5M-pixel disks at the chapter's benchmark grid (48 x 256,
    capped at both capacities): many programs racing on wide frontiers."""
    kw = _variant_kw(variant)
    img, seeds = scenes.two_disks_scene(1000, 2000, 400, gap=8)
    blocks = _pinned(48, 256, kw)
    rn, rt = _both(img, seeds, mode="multisource", threads_per_block=256,
                   blocks=blocks, **kw)
    assert_same_deterministic(rn, rt)


@pytest.mark.parametrize("mode", ["sequential", "multisource"])
@pytest.mark.parametrize("variant", INSTRUMENTED, ids=INSTRUMENTED_IDS)
def test_cross_backend_owner_map_at_one_block(variant, mode):
    """With one program every pixel is owned by block 0 on both sides, so
    even the spatial owner map is deterministic."""
    img, seeds = SCENES["asym_squares"]()
    rn, rt = _both(img, seeds, mode=mode, blocks=1, **_variant_kw(variant))
    assert_same_deterministic(rn, rt)
    np.testing.assert_array_equal(rt.owner, rn.owner)


@pytest.mark.parametrize("name", ["two_pixels", "min_gap"])
def test_cross_backend_edge_scenes_match_numba(name):
    """The one-level scene and the minimum-gap scene, at 8-conn radius 2
    (the guard's corner cases) and at 4-conn."""
    img, seeds = SCENES[name]()
    for kw in ({"connectivity": 4}, {"connectivity": 8, "radius": 2}):
        rn, rt = _both(img, seeds, mode="multisource", threads_per_block=32,
                       blocks=2, **kw)
        assert_same_deterministic(rn, rt)


# Degenerate shapes cut by a 1 px white line, seeds in opposite corners:
# 1-wide and 1-tall images (every neighbor on one axis out of bounds), a
# 3x1 image, odd rectangles. (width, height, cut along x)
EDGE_CASES = [(1, 257, False), (257, 1, True), (3, 1, True),
              (17, 33, True), (33, 17, False)]


@pytest.mark.parametrize("grid", [(5, 32), (2, 512)],
                         ids=lambda g: f"{g[0]}x{g[1]}")
@pytest.mark.parametrize("case", EDGE_CASES,
                         ids=lambda c: f"{c[0]}x{c[1]}")
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_cross_backend_degenerate_shapes_and_corner_seeds(variant, case,
                                                          grid):
    w, h, along_x = case
    img, seeds = _stripes(w, h, along_x)
    kw = _variant_kw(variant)
    blocks = _pinned(grid[0], grid[1], kw)
    rn, rt = _both(img, seeds, mode="multisource", threads_per_block=grid[1],
                   blocks=blocks, **kw)
    assert rt.filled == w * h - (h if along_x else w)
    assert_same_deterministic(rn, rt)


def _conn4_cas_from_depth(depth):
    """sum over reached u of #{4-neighbors v with depth(v) == depth(u)+1}."""
    total = 0
    reached = depth >= 0
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.full_like(depth, -2)
        src = depth[max(sx, 0):depth.shape[0] + min(sx, 0),
                    max(sy, 0):depth.shape[1] + min(sy, 0)]
        nb[max(-sx, 0):depth.shape[0] + min(-sx, 0),
           max(-sy, 0):depth.shape[1] + min(-sy, 0)] = src
        total += int(np.count_nonzero(reached & (nb == depth + 1)))
    return total


@pytest.mark.parametrize("grid", ["8x256", "own_max_x32"])
@pytest.mark.parametrize("mode", ["sequential", "multisource"])
@pytest.mark.parametrize("entry_format", ["lin", "xy"])
def test_cross_backend_conn4_cas_attempts_equal_depth_formula(entry_format,
                                                              mode, grid):
    """The canary for barrier visibility: at 4-conn, both backends must
    attempt exactly the claims the oracle's depth map predicts. A stale
    read of a pixel painted before the barrier would add attempts.
    own_max_x32 runs each backend at its own full residency of one-warp
    programs: the most barrier arrivals per level."""
    img, seeds = SCENES["two_disks"]()
    _, ref_d, *_ = cpu_flood_fill_two(img, seeds, 4)
    expected = _conn4_cas_from_depth(ref_d)
    if grid == "8x256":
        blocks, tpb = 8, 256
    else:
        blocks, tpb = None, 32
    rn, rt = _both(img, seeds, mode=mode, entry_format=entry_format,
                   threads_per_block=tpb, blocks=blocks)
    if blocks is None:
        assert rt.blocks == max_blocks(threads_per_block=32,
                                       entry_format=entry_format)
    np.testing.assert_array_equal(rt.depth, ref_d)
    assert rt.cas_attempts == expected
    assert rn.cas_attempts == expected


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_cross_backend_capacity_is_reported_per_backend(variant):
    """blocks=None resolves to each backend's own co-resident maximum (they
    differ: the register counts differ); both are honoured, and one more
    block is refused with the same RuntimeError."""
    kw = _variant_kw(variant)
    img, seeds = SCENES["two_squares"]()
    cap_t = max_blocks(threads_per_block=256, **kw)
    cap_n = numba_ff.max_blocks(threads_per_block=256, **kw)
    assert flood_fill(img, seeds, **kw).blocks == cap_t
    assert numba_ff.flood_fill(img, seeds, **kw).blocks == cap_n
    with pytest.raises(RuntimeError, match="cooperative-launch capacity"):
        flood_fill(img, seeds, blocks=cap_t + 1, **kw)


def test_cross_backend_constants_match_numba():
    """Same tables, palette, field layout, counter slots, trace cap and
    kernel names."""
    for name in ("DX_HOST", "DY_HOST", "DX8_HOST", "DY8_HOST", "DX_R2_HOST",
                 "DY_R2_HOST", "PALETTE_HOST"):
        np.testing.assert_array_equal(getattr(twin_kernels, name),
                                      getattr(numba_kernels, name))
    for name in ("XY_FIELD_BITS", "XY_FIELD_MASK", "XY_LBL_SHIFT",
                 "XY_MAX_DIM", "FILLED", "LEVELS", "OVERFLOW", "PEAK_LEVEL",
                 "PEAK_OCC", "ACTIVE_THREAD_SUM", "ACTIVE_WARP_SUM",
                 "PROCESSED", "CAS_ATTEMPTS", "INTERIOR", "NUM_COUNTERS",
                 "BS_PROCESSED", "BS_SMID", "Q_REAR"):
        assert getattr(twin_kernels, name) == getattr(numba_kernels, name)
    # the device-side tuples are the host tables
    assert tuple(twin_kernels.DX8.value) == tuple(numba_kernels.DX8_HOST)
    assert tuple(twin_kernels.DY_R2.value) == tuple(numba_kernels.DY_R2_HOST)
    assert tuple(twin_kernels.PAL1.value) == tuple(numba_kernels.PALETTE_HOST[1])
    assert twin_ff.MODES == numba_ff.MODES
    assert twin_ff.ENTRY_FORMATS == numba_ff.ENTRY_FORMATS
    assert twin_ff.LEVEL_TRACE_CAPACITY == numba_ff.LEVEL_TRACE_CAPACITY
    assert set(twin_ff._KERNELS) == set(numba_ff._KERNELS)
    for key, kernel in twin_ff._KERNELS.items():
        assert kernel.__name__ == numba_ff._KERNELS[key].__name__
    for x, y, lbl in [(3, 5, 0), (8191, 8191, 1), (100, 0, 1)]:
        for fmt in ("lin", "xy"):
            assert (twin_ff._pack(x, y, lbl, 8192, fmt)
                    == numba_ff._pack(x, y, lbl, 8192, fmt))


@pytest.mark.parametrize("args, kwargs", [
    (("not-an-image",), {}),
    ((), {"mode": "parallel"}),
    ((), {"entry_format": "packed"}),
    ((), {"seeds": [(5, 5)]}),
    ((), {"seeds": "nonsense"}),
    ((), {"seeds": "dup"}),
    ((), {"seeds": "not_red"}),
    ((), {"seeds": "outside"}),
    ((), {"threads_per_block": 100}),
    ((), {"blocks": 0}),
    ((), {"blocks": True}),
    ((), {"connectivity": "8"}),
    ((), {"radius": 3}),
    ((), {"connectivity": 4, "radius": 2}),
], ids=["img", "mode", "entry_format", "one_seed", "nonsense", "dup",
        "not_red", "outside", "tpb", "blocks0", "blocks_bool", "conn_str",
        "radius3", "radius2_conn4"])
def test_cross_backend_validation_messages_match_numba(args, kwargs):
    """Same exception type and message as the Numba driver (dashes aside:
    the twin writes "-" where the Numba messages use an em dash)."""
    img, seeds = SCENES["two_squares"]()
    if args:
        img = np.zeros((4, 4), dtype=np.uint8)
    special = {"dup": [seeds[0], seeds[0]], "not_red": [seeds[0], (0, 0)],
               "outside": [seeds[0], (96, 0)]}
    if "seeds" in kwargs:
        s = kwargs.pop("seeds")
        seeds = special.get(s, s) if isinstance(s, str) else s
    errors = []
    for ff in (numba_ff.flood_fill, flood_fill):
        with pytest.raises(Exception) as exc:
            ff(img, seeds, **kwargs)
        errors.append((exc.type, str(exc.value).replace("\u2014", "-")))
    assert errors[0] == errors[1]


@requires_streams
def test_cross_backend_streams_matches_numba():
    """Triton's streams mode against Numba's sequential mode at the same
    grid: the same two single-seed launches, so every deterministic output
    and per-launch statistic agrees. Numba's own streams mode is not run
    here: its concurrent cooperative pair can wedge after other launches
    in the same process (see the Numba test file)."""
    img, seeds = SCENES["two_squares"]()
    rn = numba_ff.flood_fill(img, seeds, mode="sequential",
                             blocks=STREAMS_TEST_BLOCKS)
    rt = flood_fill(img, seeds, mode="streams", blocks=STREAMS_TEST_BLOCKS)
    for name in ("img", "visited", "depth", "label"):
        np.testing.assert_array_equal(getattr(rt, name), getattr(rn, name))
    for name in ("blocks", "filled", "filled_a", "filled_b", "levels",
                 "levels_a", "levels_b", "processed", "cas_attempts",
                 "model_bytes"):
        assert getattr(rt, name) == getattr(rn, name), name
    assert_same_launches(rn, rt)
    assert rt.overlap_ratio > 0


# The child: Numba's streams mode on a saved scene, its result pickled
# back (DualBlobResult and LaunchStats hold only host data).
_NUMBA_STREAMS_CHILD = """
import os, pickle, sys
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
import numpy as np
from flood_fill_cuda.chapters.ch04_gpu_2blob_nblock.flood_fill import (
    flood_fill)
src, dst, blocks = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = np.load(src)
seeds = [tuple(int(v) for v in s) for s in data["seeds"]]
r = flood_fill(data["img"], seeds, mode="streams", blocks=blocks)
with open(dst, "wb") as f:
    pickle.dump(r, f)
"""


@requires_numba_streams
def test_cross_backend_streams_matches_numba_streams(tmp_path):
    """Both backends' streams mode, directly: Numba's concurrent pair runs
    in a fresh process under a hard timeout (the Numba README's
    fresh-process probe), Triton's here, at the same 8+8 grid. Every
    deterministic output and per-launch statistic agrees; the overlap
    ratio is timing and only has to be positive on both. A Numba wedge
    (the timeout) skips with the reason: it is the documented Numba
    behaviour, not a twin result."""
    import pickle
    import subprocess
    import sys

    img, seeds = SCENES["two_squares"]()
    # the twin first, so a Numba wedge cannot leave the GPU busy under it
    rt = flood_fill(img, seeds, mode="streams", blocks=STREAMS_TEST_BLOCKS)

    src, dst = tmp_path / "scene.npz", tmp_path / "numba_streams.pkl"
    np.savez(src, img=img, seeds=np.asarray(seeds, dtype=np.int64))
    env = dict(os.environ, NUMBA_CUDA_USE_NVIDIA_BINDING="1")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _NUMBA_STREAMS_CHILD, str(src), str(dst),
             str(STREAMS_TEST_BLOCKS)],
            capture_output=True, text=True, env=env,
            timeout=NUMBA_STREAMS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pytest.skip(f"Numba's {STREAMS_TEST_BLOCKS}+{STREAMS_TEST_BLOCKS} "
                    f"streams pair did not finish in "
                    f"{NUMBA_STREAMS_TIMEOUT_S} s and was killed: the wedge "
                    f"the Numba README documents (Finding 2)")
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open(dst, "rb") as f:
        rn = pickle.load(f)

    assert rn.mode == rt.mode == "streams"
    for name in ("img", "visited", "depth", "label"):
        np.testing.assert_array_equal(getattr(rt, name), getattr(rn, name),
                                      err_msg=name)
    for name in ("blocks", "filled", "filled_a", "filled_b", "levels",
                 "levels_a", "levels_b", "processed", "cas_attempts",
                 "model_bytes"):
        assert getattr(rt, name) == getattr(rn, name), name
    assert_same_launches(rn, rt)
    assert rn.overlap_ratio > 0 and rt.overlap_ratio > 0
