"""Correctness tests: the Triton twin of the dual-block flood fill.

The first part is the Numba chapter's test file, test for test, with the
same names, scenes, parameters and assertions against the same CPU oracle:
every kernel must match the reference exactly (visited mask, depth map,
level and filled counts), recolor reached pixels solid blue, and leave
everything else untouched. The only adaptation: the pinned experiment runs
at the twin's PINNED_TPB (512) instead of Numba's 768, which is not a power
of 2.

The second part (test_cross_backend_*) runs the Numba chapter and the twin
on the same inputs and asserts that every deterministic output is
identical.

The third part (test_enqueue_*, test_lane_enqueue_*) runs both enqueue
translations of every kernel (enqueue="lane", the default, and "program",
the first translation) against Numba and the CPU oracle, and checks in
the SASS that ptxas warp-aggregates the per-lane atomics.

Run:

    .venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch02_gpu_1blob_2block/test_correctness.py -v
"""

import os
import re
import shutil
import subprocess

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import numpy as np
import pytest

from flood_fill_cuda.chapters.ch02_gpu_1blob_2block import scenes
from flood_fill_cuda.chapters.ch02_gpu_1blob_2block.cpu_oracle import cpu_flood_fill

from .flood_fill import PINNED_TPB, flood_fill

BLUE = np.array([0, 0, 255], dtype=np.uint8)

KERNELS = ["split", "global", "dirsplit"]


def assert_matches_reference(img, seed_x, seed_y, **gpu_kwargs):
    ref_visited, ref_depth, ref_levels, ref_filled = cpu_flood_fill(img, seed_x, seed_y)
    result = flood_fill(img, seed_x, seed_y, **gpu_kwargs)

    np.testing.assert_array_equal(result.visited, ref_visited)
    np.testing.assert_array_equal(result.depth, ref_depth)
    assert result.levels == ref_levels
    assert result.filled == ref_filled
    filled_mask = result.visited == 1
    assert (result.img[filled_mask] == BLUE).all()
    np.testing.assert_array_equal(result.img[~filled_mask], img[~filled_mask])
    return result


SCENES = {
    "square_64": lambda: scenes.square_scene(64, 64, 32, 32),
    "square_nonsquare_img": lambda: scenes.square_scene(200, 130, 100, 70),
    "square_at_corner": lambda: scenes.square_scene(128, 128, 50, 50, corner=True),
    "corner_seeded_square": lambda: scenes.corner_seeded_square_scene(128, 128, 64, 64),
    "square_full_bleed": lambda: scenes.square_scene(96, 96, 96, 96),
    "disk_101": lambda: scenes.disk_scene(101, 101, 40),
    "serpentine_128": lambda: scenes.serpentine_scene(128, 128),
    "serpentine_nonsquare": lambda: scenes.serpentine_scene(64, 200),
    "random_subcritical": lambda: scenes.random_scene(200, 200, 0.35, rng_seed=7),
    "random_supercritical": lambda: scenes.random_scene(200, 200, 0.65, rng_seed=7),
    "single_pixel": lambda: scenes.single_pixel_scene(64, 64),
    "full_red_128": lambda: scenes.full_red_scene(128, 128),
    # seam-aware additions
    "seam_seeded": lambda: scenes.seam_seeded_scene(128, 128, 60, 60),
    "seam_serpentine_128": lambda: scenes.seam_serpentine_scene(128, 128),
    "offcenter_blob": lambda: scenes.offcenter_blob_scene(128, 128, 40),
}


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("name", SCENES.keys())
def test_matches_cpu_reference(name, kernel):
    img, sx, sy = SCENES[name]()
    assert_matches_reference(img, sx, sy, kernel=kernel)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("tpb", [64, 256, 512])
def test_block_size_invariance(tpb, kernel):
    img, sx, sy = scenes.random_scene(256, 256, 0.65, rng_seed=3)
    assert_matches_reference(img, sx, sy, threads_per_block=tpb, kernel=kernel)


@pytest.mark.parametrize("kernel", KERNELS)
def test_deterministic_across_runs(kernel):
    """Queue order - and for split/dirsplit even which program claims a
    pixel - is race-dependent, so per-program/inbox counts may differ
    between runs; visited/depth/levels/filled must not."""
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=11)
    a = flood_fill(img, sx, sy, kernel=kernel)
    b = flood_fill(img, sx, sy, kernel=kernel)
    np.testing.assert_array_equal(a.visited, b.visited)
    np.testing.assert_array_equal(a.depth, b.depth)
    assert a.levels == b.levels and a.filled == b.filled


@pytest.mark.parametrize("name", ["seam_seeded", "random_supercritical",
                                  "serpentine_128"])
def test_kernels_agree(name):
    """All three partitionings are the same BFS: identical results."""
    img, sx, sy = SCENES[name]()
    results = [flood_fill(img, sx, sy, kernel=k) for k in KERNELS]
    for r in results[1:]:
        np.testing.assert_array_equal(results[0].visited, r.visited)
        np.testing.assert_array_equal(results[0].depth, r.depth)
        assert results[0].filled == r.filled and results[0].levels == r.levels


def test_input_not_modified():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    before = img.copy()
    flood_fill(img, sx, sy)
    np.testing.assert_array_equal(img, before)


def test_unreached_pixels_untouched():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[10:30, 10:30] = scenes.RED   # component A (seeded)
    img[60:80, 60:80] = scenes.RED   # component B (disconnected)
    result = assert_matches_reference(img, 20, 20)
    np.testing.assert_array_equal(result.img[60:80, 60:80], img[60:80, 60:80])
    assert result.visited[60:80, 60:80].sum() == 0


@pytest.mark.parametrize("kernel", KERNELS)
def test_only_red_pixels_ever_visited(kernel):
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=5)
    originally_red = (img == scenes.RED).all(axis=2)
    result = flood_fill(img, sx, sy, kernel=kernel)
    assert (originally_red[result.visited == 1]).all()


@pytest.mark.parametrize("kernel", KERNELS)
def test_depth_is_true_bfs_distance(kernel):
    """On a full-red image seeded at the corner, 4-connected BFS depth is
    the Manhattan distance x + y - including across the seam, which proves
    the cross-program handoff preserves level-exactness."""
    img, sx, sy = scenes.full_red_scene(64, 64)
    result = flood_fill(img, sx, sy, kernel=kernel)
    xs, ys = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    np.testing.assert_array_equal(result.depth, xs + ys)
    assert result.levels == 127


@pytest.mark.parametrize("kernel", KERNELS)
def test_exactly_once_processing(kernel):
    """No double work anywhere - including through inboxes and the
    double-ended buffer: every filled pixel is dequeued exactly once, and
    the per-program counts account for all of it."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    _, _, _, ref_filled = cpu_flood_fill(img, sx, sy)
    result = flood_fill(img, sx, sy, kernel=kernel)
    assert result.processed == result.filled == ref_filled
    assert result.processed_b0 + result.processed_b1 == result.processed
    assert result.filled - 1 <= result.cas_attempts <= 4 * result.filled


@pytest.mark.parametrize("kernel", KERNELS)
def test_frontier_trace_consistency(kernel):
    """The global per-level trace (sum of both programs' rows) must account
    for every pixel and agree with the depth map and the in-kernel
    utilization accumulators."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    tpb = 128
    result = flood_fill(img, sx, sy, threads_per_block=tpb, kernel=kernel)
    sizes = result.level_sizes.astype(np.int64)
    per_block = result.level_sizes_per_block.astype(np.int64)

    assert not result.level_trace_truncated
    assert len(sizes) == result.levels
    assert sizes[0] == 1
    assert sizes.sum() == result.filled
    assert sizes.max() == result.peak_level
    np.testing.assert_array_equal(per_block.sum(axis=0), sizes)
    # split and dirsplit count per program (own window vs its tpb lanes);
    # global's grid-stride assigns work by GLOBAL lane id, so its counts
    # are grid-wide: min(level_size, 2*tpb) contiguous busy lanes.
    if kernel == "global":
        active = np.minimum(sizes, 2 * tpb)
    else:
        active = np.minimum(per_block, tpb)
    expected_thread_util = 100.0 * active.sum() / (result.levels * 2 * tpb)
    engaged = (active + 31) // 32
    expected_warp_engagement = 100.0 * engaged.sum() / (result.levels * 2 * (tpb // 32))
    assert result.thread_util_pct == pytest.approx(expected_thread_util)
    assert result.warp_engagement_pct == pytest.approx(expected_warp_engagement)
    depth_counts = np.bincount(result.depth[result.depth >= 0].ravel(),
                               minlength=result.levels)
    np.testing.assert_array_equal(depth_counts, sizes)


# ------------------------------------------------------- split-kernel specifics

def test_split_balance_on_seam_symmetric_scene():
    """A centered blob is mirror-symmetric about the seam: both halves get
    the same pixel count, so per-program processed must match exactly."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    r = flood_fill(img, sx, sy, kernel="split")
    assert r.processed_b0 + r.processed_b1 == r.filled
    assert r.balance_pct >= 99.9
    img, sx, sy = scenes.disk_scene(256, 256, 100)
    r = flood_fill(img, sx, sy, kernel="split")
    assert r.balance_pct >= 95.0


def test_split_offcenter_starves_block1():
    """A blob entirely in the left half: program 1 owns no red pixel and
    must process exactly zero work."""
    img, sx, sy = scenes.offcenter_blob_scene(256, 256, 60)
    r = flood_fill(img, sx, sy, kernel="split")
    assert r.processed_b1 == 0
    assert r.balance_pct == 0.0
    assert r.inbox_to_b0 == r.inbox_to_b1 == 0


def test_split_seam_seeded_uses_inbox_immediately():
    img, sx, sy = scenes.seam_seeded_scene(128, 128, 60, 60)
    r = flood_fill(img, sx, sy, kernel="split")
    assert r.inbox_to_b0 >= 1  # seed column belongs to program 1


@pytest.mark.parametrize("scene", ["seam_seeded", "seam_serpentine_128"])
def test_split_inbox_structural_bound(scene):
    """Cross-seam handoffs can only be pixels of the single column adjacent
    to the seam, each claimed once: inbox counts can never exceed height."""
    img, sx, sy = SCENES[scene]()
    height = img.shape[1]
    r = flood_fill(img, sx, sy, kernel="split")
    assert 0 <= r.inbox_to_b0 <= height
    assert 0 <= r.inbox_to_b1 <= height
    if scene == "seam_serpentine_128":
        assert r.inbox_to_b0 >= 1 and r.inbox_to_b1 >= 1


def test_split_two_rings_absorb_the_v1_tripwire_scene():
    """The 2600^2 full-bleed scene fits the split kernel's two 8192-slot
    rings outright: per-half peak occupancy ~2*2600 < 8192."""
    img, sx, sy = scenes.overflow_scene()
    r = assert_matches_reference(img, sx, sy, kernel="split")
    assert r.spilled_b0 == r.spilled_b1 == 0
    assert r.peak_spill_window == 0


def test_split_spills_when_a_half_overflows():
    """4600^2 full-bleed center-seeded: per-half two-level occupancy
    ~2*4600 = 9200 > 8192, so BOTH halves must use their spill tiers and
    the result must stay reference-exact through them."""
    img, sx, sy = scenes.square_scene(4600, 4600, 4600, 4600)
    r = assert_matches_reference(img, sx, sy, kernel="split")
    assert r.spilled_b0 > 0 and r.spilled_b1 > 0
    assert r.spilled == r.spilled_b0 + r.spilled_b1


# ---------------------------------------------------- dirsplit-kernel specifics

def test_dirsplit_balance_is_spatially_agnostic():
    """The direction partition splits ~50/50 wherever the blob sits - even
    entirely inside one half, where the split kernel's balance is 0%."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    assert flood_fill(img, sx, sy, kernel="dirsplit").balance_pct >= 90.0
    img, sx, sy = scenes.offcenter_blob_scene(256, 256, 60)
    assert flood_fill(img, sx, sy, kernel="dirsplit").balance_pct >= 90.0


def test_dirsplit_starves_on_direction_degenerate_shape():
    """Per-program TOTALS on the serpentine come out ~50/50, yet at any
    given level the single-pixel frontier sits in exactly one queue: the
    per-program trace shows the programs almost never work together."""
    img, sx, sy = scenes.serpentine_scene(256, 256)
    r = flood_fill(img, sx, sy, kernel="dirsplit")
    assert r.balance_pct > 90.0  # totals look great...
    per_block = r.level_sizes_per_block
    both_active = ((per_block[0] > 0) & (per_block[1] > 0)).mean()
    assert both_active < 0.05  # ...but the programs almost never work together


@pytest.mark.parametrize("kernel", KERNELS)
def test_owner_map(kernel):
    """The per-pixel owner map must be complete and consistent: every
    reached pixel owned by exactly one program, nothing else touched, and
    the per-pixel census must reproduce the balance counters exactly. For
    split, ownership IS the spatial partition."""
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=13)
    r = flood_fill(img, sx, sy, kernel=kernel)
    reached = r.visited == 1
    assert np.isin(r.owner[reached], (0, 1)).all()
    assert (r.owner[~reached] == -1).all()
    counts = np.bincount(r.owner[reached].astype(np.int64), minlength=2)
    assert counts[0] == r.processed_b0
    assert counts[1] == r.processed_b1
    if kernel == "split":
        half = img.shape[0] // 2
        xs = np.arange(img.shape[0])[:, None]
        np.testing.assert_array_equal(r.owner[reached],
                                      (xs >= half).astype(np.int8)
                                      .repeat(img.shape[1], axis=1)[reached])


def test_owner_map_absent_on_bare():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    r = flood_fill(img, sx, sy, bare=True)
    assert r.owner.size == 0


# ------------------------------------------------------------------ bare twins

@pytest.mark.parametrize("kernel", KERNELS)
def test_bare_twin_matches_reference(kernel):
    """The uninstrumented specializations must be exactly as correct - only
    counters and traces are stripped, never algorithm."""
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=9)
    r = assert_matches_reference(img, sx, sy, kernel=kernel, bare=True)
    assert r.bare
    assert r.processed == 0  # instrumentation genuinely absent


# --------------------------------------------------------- pinned (placement)
# Adapted: threads_per_block=PINNED_TPB (512) instead of Numba's 768.

@pytest.mark.parametrize("scene", ["square_64", "random_supercritical"])
def test_pinned_same_sm_is_reference_exact_and_co_resident(scene):
    img, sx, sy = SCENES[scene]()
    r = assert_matches_reference(img, sx, sy, kernel="pinned",
                                 threads_per_block=PINNED_TPB,
                                 placement="same_sm")
    assert r.sm_id_b0 == r.sm_id_b1 >= 0
    assert r.same_sm


@pytest.mark.parametrize("scene", ["square_64", "random_supercritical"])
def test_pinned_spread_is_reference_exact(scene):
    """Natural placement: smids are recorded (the scheduler is expected to
    spread, but is not obligated - report, don't hard-assert inequality)."""
    img, sx, sy = SCENES[scene]()
    r = assert_matches_reference(img, sx, sy, kernel="pinned",
                                 threads_per_block=PINNED_TPB,
                                 placement="spread")
    assert r.sm_id_b0 >= 0 and r.sm_id_b1 >= 0


# ------------------------------------------------------------------ validation

def test_seed_not_red_raises():
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="not red"):
        flood_fill(img, 0, 0)


def test_seed_out_of_bounds_raises():
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="outside"):
        flood_fill(img, 64, 0)


@pytest.mark.parametrize("tpb", [100, 2048, 0, 1024])
def test_bad_threads_per_block_raises(tpb):
    """1024 is deliberately rejected, as in the Numba chapter."""
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="threads_per_block"):
        flood_fill(img, sx, sy, threads_per_block=tpb)


def test_bad_kernel_name_raises():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="kernel"):
        flood_fill(img, sx, sy, kernel="turbo")


def test_pinned_requires_placement_and_tpb():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="placement"):
        flood_fill(img, sx, sy, kernel="pinned", threads_per_block=PINNED_TPB)
    with pytest.raises(ValueError, match=str(PINNED_TPB)):
        flood_fill(img, sx, sy, kernel="pinned", threads_per_block=256,
                   placement="same_sm")
    with pytest.raises(ValueError, match="placement"):
        flood_fill(img, sx, sy, kernel="split", placement="same_sm")


# ============================================================ cross-backend
# Numba chapter vs Triton twin on identical inputs. Only deterministic
# outputs are compared (see README): queue order, which program wins a
# multi-parent claim (split inbox/spill counts, dirsplit per-program split,
# global's per-pixel owner), and %smid are schedule-dependent.
#
# cas_attempts IS deterministic here: the 4-connected grid is bipartite, so
# a depth-L pixel always sees its depth L-1 neighbors already blue (behind a
# barrier) and its depth L+1 neighbors still red. Every directed edge
# p -> q with depth(q) = depth(p) + 1 is exactly one claim attempt, on both
# backends. A mismatch would mean a stale read slipped past a barrier.

CROSS_SCENES = {
    "seam_seeded": SCENES["seam_seeded"],
    "random_supercritical": lambda: scenes.random_scene(256, 256, 0.65, rng_seed=3),
    "serpentine_128": SCENES["serpentine_128"],
    "offcenter_blob": SCENES["offcenter_blob"],
    "full_red_128": SCENES["full_red_128"],
    "seam_serpentine_128": SCENES["seam_serpentine_128"],
}


@pytest.fixture(scope="module")
def numba_ff():
    from flood_fill_cuda.chapters.ch02_gpu_1blob_2block import flood_fill as nff
    return nff


def _assert_same_bfs(n, t):
    np.testing.assert_array_equal(t.img, n.img)
    np.testing.assert_array_equal(t.visited, n.visited)
    np.testing.assert_array_equal(t.depth, n.depth)
    assert (t.levels, t.filled) == (n.levels, n.filled)


def _assert_same_deterministic(n, t, kernel, bare):
    _assert_same_bfs(n, t)
    for name in ("peak_level", "peak_occupancy", "processed", "cas_attempts",
                 "level_trace_truncated", "ring_capacity", "blocks",
                 "threads_per_block", "occupancy_pct", "sm_utilization_pct",
                 "discovery_redundancy", "neighbor_check_efficiency_pct"):
        assert getattr(t, name) == getattr(n, name), name
    np.testing.assert_array_equal(t.level_sizes, n.level_sizes)
    assert t.owner.shape == n.owner.shape
    if bare:
        assert t.processed == t.cas_attempts == 0
        assert t.sm_id_b0 == t.sm_id_b1 == -1
        assert t.level_sizes_per_block.shape == (2, 0)
        return
    if kernel in ("split", "global"):
        # split: ownership is spatial; global: item -> program is positional
        for name in ("processed_b0", "processed_b1", "balance_pct",
                     "thread_util_pct", "warp_engagement_pct",
                     "lane_efficiency_pct"):
            assert getattr(t, name) == getattr(n, name), name
        np.testing.assert_array_equal(t.level_sizes_per_block,
                                      n.level_sizes_per_block)
    if kernel == "split":
        np.testing.assert_array_equal(t.owner, n.owner)
        # structural bounds only: the seam race moves pixels between tiers
        height = t.img.shape[1]
        assert 0 <= t.inbox_to_b0 <= height and 0 <= t.inbox_to_b1 <= height
        assert t.spilled == t.spilled_b0 + t.spilled_b1
    else:
        # global's per-pixel owner is race-dependent, its census is not
        reached = t.visited == 1
        assert np.isin(t.owner[reached], (0, 1)).all()
        assert (t.owner[~reached] == -1).all()
        if kernel == "global":
            np.testing.assert_array_equal(
                np.bincount(t.owner[reached].astype(np.int64), minlength=2),
                np.bincount(n.owner[reached].astype(np.int64), minlength=2))
    assert t.processed_b0 + t.processed_b1 == t.processed


def test_cross_backend_constants_match(numba_ff):
    """Same counter slots, state slots and capacities as the Numba chapter."""
    from flood_fill_cuda.chapters.ch02_gpu_1blob_2block import kernels as nk
    from . import flood_fill as tff
    from . import kernels as tk
    names = ["RING_CAPACITY", "RING_MASK", "FILLED", "LEVELS", "OVERFLOW",
             "PEAK_LEVEL", "PEAK_OCC", "ACTIVE_THREAD_SUM", "ACTIVE_WARP_SUM",
             "PROCESSED", "CAS_ATTEMPTS", "SPILLED", "PEAK_SPILL_WINDOW",
             "PROCESSED_B0", "PROCESSED_B1", "SPILLED_B0", "SPILLED_B1",
             "INBOX_TO_B0", "INBOX_TO_B1", "SMID_B0", "SMID_B1",
             "NUM_COUNTERS", "Q_REAR", "Q_REAR0", "Q_REAR1", "G_INBOX_REAR0",
             "G_INBOX_REAR1", "G_PUB_RING0", "G_PUB_SPILL0", "G_PUB_RING1",
             "G_PUB_SPILL1", "P_MODE", "P_CHOSEN_SMID", "P_WORKER_COUNT",
             "BAR_ARRIVE", "BAR_GEN"]
    for name in names:
        assert int(getattr(tk, name)) == int(getattr(nk, name)), name
    assert tff.LEVEL_TRACE_CAPACITY == numba_ff.LEVEL_TRACE_CAPACITY
    import dataclasses
    assert ([f.name for f in dataclasses.fields(tff.DualFloodFillResult)]
            == [f.name for f in dataclasses.fields(numba_ff.DualFloodFillResult)])


@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("name", CROSS_SCENES.keys())
def test_cross_backend_matches_numba(name, kernel, bare, numba_ff):
    img, sx, sy = CROSS_SCENES[name]()
    n = numba_ff.flood_fill(img, sx, sy, kernel=kernel, bare=bare)
    t = flood_fill(img, sx, sy, kernel=kernel, bare=bare)
    _assert_same_deterministic(n, t, kernel, bare)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("tpb", [32, 64, 128, 512])
def test_cross_backend_tpb_sweep_matches_numba(tpb, kernel, numba_ff):
    """The per-program metrics that depend on tpb (positional work split of
    global, utilization accumulators) agree at every power-of-2 tpb."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    n = numba_ff.flood_fill(img, sx, sy, threads_per_block=tpb, kernel=kernel)
    t = flood_fill(img, sx, sy, threads_per_block=tpb, kernel=kernel)
    _assert_same_deterministic(n, t, kernel, False)


def test_cross_backend_split_spill_tiers_match_numba(numba_ff):
    """overflow_scene: no seam race can push a pixel into a spill tier (the
    rings never fill), so both backends report zero spill and the same
    exact BFS through the same two rings."""
    img, sx, sy = scenes.overflow_scene()
    n = numba_ff.flood_fill(img, sx, sy, kernel="split")
    t = flood_fill(img, sx, sy, kernel="split")
    _assert_same_deterministic(n, t, "split", False)
    assert (t.spilled, t.peak_spill_window) == (n.spilled, n.peak_spill_window) == (0, 0)


@pytest.mark.parametrize("placement", ["same_sm", "spread"])
@pytest.mark.parametrize("scene", ["square_64", "random_supercritical",
                                   "seam_serpentine_128"])
def test_cross_backend_pinned_matches_numba(scene, placement, numba_ff):
    """Numba pins 2 x 768 threads, the twin 2 x 512 lanes: the BFS output
    does not depend on the worker width, so it must be identical."""
    img, sx, sy = SCENES[scene]()
    n = numba_ff.flood_fill(img, sx, sy, kernel="pinned",
                            threads_per_block=numba_ff.PINNED_TPB,
                            placement=placement)
    t = flood_fill(img, sx, sy, kernel="pinned", threads_per_block=PINNED_TPB,
                   placement=placement)
    _assert_same_bfs(n, t)
    assert t.owner.size == n.owner.size == 0
    assert (t.processed, t.cas_attempts) == (n.processed, n.cas_attempts) == (0, 0)
    if placement == "same_sm":
        assert t.same_sm and n.same_sm


def test_cross_backend_non_power_of_2_tpb_rejected_by_twin_only(numba_ff):
    """Numba accepts any multiple of 32 up to 512; the twin also needs a
    power of 2 and says so."""
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    n = numba_ff.flood_fill(img, sx, sy, threads_per_block=96)
    assert n.filled == 400
    with pytest.raises(ValueError, match="threads_per_block must be a power of 2"):
        flood_fill(img, sx, sy, threads_per_block=96)


# ------------------------------------------- input layouts and seed types
# Every scene builder returns a C-contiguous image and Python-int seeds, so
# the tests above never vary either. The twin's kernels address img as a
# flat C-order buffer; the driver must make any other layout contiguous.

INPUT_VARIANTS = [(k, b) for k in KERNELS for b in (False, True)] + [("pinned", False)]


def _run_both(numba_ff, img, sx, sy, kernel, bare, numba_img=None):
    """Numba on numba_img (default img) and the twin on img, same config.
    pinned runs its spread placement (2 workers, no occupancy forcing)."""
    if kernel == "pinned":
        nkw = dict(kernel="pinned", threads_per_block=numba_ff.PINNED_TPB,
                   placement="spread")
        tkw = dict(kernel="pinned", threads_per_block=PINNED_TPB,
                   placement="spread")
    else:
        nkw = tkw = dict(kernel=kernel, bare=bare)
    n = numba_ff.flood_fill(img if numba_img is None else numba_img, sx, sy,
                            **nkw)
    t = flood_fill(img, sx, sy, **tkw)
    return n, t


def _assert_same_as_numba_and_reference(img, sx, sy, n, t, kernel, bare):
    if kernel == "pinned":
        _assert_same_bfs(n, t)
    else:
        _assert_same_deterministic(n, t, kernel, bare)
    ref_visited, ref_depth, ref_levels, ref_filled = cpu_flood_fill(img, sx, sy)
    np.testing.assert_array_equal(t.visited, ref_visited)
    np.testing.assert_array_equal(t.depth, ref_depth)
    assert (t.levels, t.filled) == (ref_levels, ref_filled)
    filled_mask = t.visited == 1
    assert (t.img[filled_mask] == BLUE).all()
    np.testing.assert_array_equal(t.img[~filled_mask], img[~filled_mask])


@pytest.mark.parametrize("kernel,bare", INPUT_VARIANTS)
def test_cross_backend_fortran_order_input(kernel, bare, numba_ff):
    """A Fortran-ordered image: Numba's kernels index it through strides,
    the twin's driver copies it to a C-order device buffer. Same result,
    input untouched."""
    img_c, sx, sy = scenes.random_scene(97, 61, 0.65, rng_seed=21)
    img = np.asfortranarray(img_c)
    assert img.flags.f_contiguous and not img.flags.c_contiguous
    n, t = _run_both(numba_ff, img, sx, sy, kernel, bare)
    _assert_same_as_numba_and_reference(img, sx, sy, n, t, kernel, bare)
    np.testing.assert_array_equal(img, img_c)


@pytest.mark.parametrize("kernel", KERNELS)
def test_cross_backend_strided_input(kernel, numba_ff):
    """A strided view (every other row of a larger buffer): both backends
    copy it to a contiguous device buffer."""
    img_c, sx, sy = scenes.random_scene(97, 61, 0.65, rng_seed=21)
    buf = np.zeros((2 * 97, 61, 3), dtype=np.uint8)
    buf[::2] = img_c
    img = buf[::2]
    assert not img.flags.c_contiguous and not img.flags.f_contiguous
    n, t = _run_both(numba_ff, img, sx, sy, kernel, False)
    _assert_same_as_numba_and_reference(img, sx, sy, n, t, kernel, False)


@pytest.mark.parametrize("kernel", KERNELS)
def test_cross_backend_permuted_axis_input(kernel, numba_ff):
    """A permuted-axis view (a transposed buffer, neither C nor F order):
    Numba's copy_to_device rejects it; the twin copies it to C order and
    returns what Numba returns for the contiguous copy (a documented
    superset)."""
    img_c, sx, sy = scenes.random_scene(97, 61, 0.65, rng_seed=21)
    img = np.ascontiguousarray(img_c.transpose(1, 0, 2)).transpose(1, 0, 2)
    assert not img.flags.c_contiguous and not img.flags.f_contiguous
    with pytest.raises(ValueError, match="non-contiguous"):
        numba_ff.flood_fill(img, sx, sy, kernel=kernel)
    n, t = _run_both(numba_ff, img, sx, sy, kernel, False, numba_img=img_c)
    _assert_same_as_numba_and_reference(img, sx, sy, n, t, kernel, False)


@pytest.mark.parametrize("kernel,bare", INPUT_VARIANTS)
def test_cross_backend_numpy_int_seeds(kernel, bare, numba_ff):
    """NumPy integer seeds (np.argwhere gives np.int64): Numba types them
    as kernel args; the twin converts them, so the split kernel's scalar
    seed args never reach Triton's launcher as NumPy scalars."""
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    ax, ay = np.argwhere((img == scenes.RED).all(axis=2))[0]
    assert isinstance(ax, np.int64)
    for cast in (np.int64, np.int32, np.uint16):
        sx, sy = cast(ax), cast(ay)
        n, t = _run_both(numba_ff, img, sx, sy, kernel, bare)
        _assert_same_as_numba_and_reference(img, sx, sy, n, t, kernel, bare)
        assert t.filled == 400


# ================================================== enqueue translations
# Every kernel has two enqueue translations (the ENQ constexpr, the
# driver's enqueue=): "lane" (default: one masked atomic per claiming lane,
# warp-aggregated by ptxas) and "program" (the first translation:
# program-wide tl.sum/tl.cumsum, one atomic per program). Only queue order
# may differ between them; every deterministic output must equal Numba's
# and the CPU oracle's in both.

ENQUEUE_MODES = ["lane", "program"]
ENQ_SCENES = ["seam_seeded", "random_supercritical", "seam_serpentine_128",
              "full_red_128"]


@pytest.mark.parametrize("enqueue", ENQUEUE_MODES)
@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("name", ENQ_SCENES)
def test_enqueue_modes_match_numba_and_reference(name, kernel, bare, enqueue,
                                                 numba_ff):
    img, sx, sy = CROSS_SCENES[name]()
    n = numba_ff.flood_fill(img, sx, sy, kernel=kernel, bare=bare)
    t = flood_fill(img, sx, sy, kernel=kernel, bare=bare, enqueue=enqueue)
    _assert_same_as_numba_and_reference(img, sx, sy, n, t, kernel, bare)


@pytest.mark.parametrize("enqueue", ENQUEUE_MODES)
@pytest.mark.parametrize("placement", ["same_sm", "spread"])
@pytest.mark.parametrize("scene", ["random_supercritical",
                                   "seam_serpentine_128"])
def test_enqueue_modes_pinned_match_numba_and_reference(scene, placement,
                                                        enqueue, numba_ff):
    img, sx, sy = SCENES[scene]()
    n = numba_ff.flood_fill(img, sx, sy, kernel="pinned",
                            threads_per_block=numba_ff.PINNED_TPB,
                            placement=placement)
    t = flood_fill(img, sx, sy, kernel="pinned", threads_per_block=PINNED_TPB,
                   placement=placement, enqueue=enqueue)
    _assert_same_as_numba_and_reference(img, sx, sy, n, t, "pinned", False)
    if placement == "same_sm":
        assert t.same_sm


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("tpb", [32, 512])
def test_enqueue_modes_agree_at_tpb_extremes(tpb, kernel, numba_ff):
    """One warp per program (32) and 16 warps per program (512): the lane
    path's per-warp slabs and the program path's single slab give the same
    deterministic outputs as Numba."""
    img, sx, sy = scenes.random_scene(256, 256, 0.65, rng_seed=17)
    n = numba_ff.flood_fill(img, sx, sy, threads_per_block=tpb, kernel=kernel)
    for enqueue in ENQUEUE_MODES:
        t = flood_fill(img, sx, sy, threads_per_block=tpb, kernel=kernel,
                       enqueue=enqueue)
        _assert_same_as_numba_and_reference(img, sx, sy, n, t, kernel, False)


def test_enqueue_modes_spill_tiers_match_numba(numba_ff):
    """A blob in the left half only, big enough to overflow program 0's
    8192-slot ring (two-level occupancy ~9,400). No seam race can move a
    pixel between tiers, so the spill counters are deterministic: Numba,
    lane and program must report the same nonzero spill, the same peak
    spill window, and stay reference-exact through the spill tier. Under
    enqueue="lane" the ring rear overshoots by the spilled tickets every
    spilling level and is clamped back at the level end."""
    img, sx, sy = scenes.offcenter_blob_scene(4800, 2400, 2350)
    n = numba_ff.flood_fill(img, sx, sy, kernel="split")
    assert n.spilled_b0 > 0 and n.spilled_b1 == 0
    for enqueue in ENQUEUE_MODES:
        t = flood_fill(img, sx, sy, kernel="split", enqueue=enqueue)
        _assert_same_as_numba_and_reference(img, sx, sy, n, t, "split", False)
        for name in ("spilled", "spilled_b0", "spilled_b1",
                     "peak_spill_window", "inbox_to_b0", "inbox_to_b1"):
            assert getattr(t, name) == getattr(n, name), (enqueue, name)
        del t


def test_enqueue_bad_mode_raises():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="enqueue"):
        flood_fill(img, sx, sy, enqueue="warp")


# ---------------------------------------------------------- SASS evidence
# The point of enqueue="lane": ptxas compiles a masked per-lane
# atomic_add on a uniform address into a warp-aggregated atomic (vote of
# the claiming lanes, popcount, ONE leader ATOMG.E.ADD, SHFL.IDX of its
# result, lanemask popcount for the rank), the pattern Numba's
# _warp_enqueue_* writes by hand, and with no CTA barrier. If a Triton or
# ptxas upgrade stops doing that, these tests say so.

def _find_nvdisasm():
    """PATH first, then the copy Triton ships beside its ptxas."""
    import triton
    bundled = os.path.join(os.path.dirname(triton.__file__), "backends",
                           "nvidia", "bin", "nvdisasm")
    return shutil.which("nvdisasm") or bundled


NVDISASM = _find_nvdisasm()

# per-lane append sites per kernel: 4 directions x (global queue | ring
# ticket + spill ticket + inbox), pinned shares global's 4
LANE_APPEND_SITES = {"global": 4, "dirsplit": 4, "split": 12, "pinned": 4}


def _sass_instructions(cubin, path):
    """nvdisasm needs a seekable file, so the cubin goes to path first."""
    path.write_bytes(cubin)
    out = subprocess.run([NVDISASM, "-c", str(path)], capture_output=True,
                         check=True).stdout.decode()
    return [m.group(1) for m in
            (re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
             for line in out.splitlines()) if m]


def _warp_aggregated_adds(ins, back=8, fwd=8):
    """Leader-only ATOMG.E.ADD with a lane vote and a popcount before it
    and a SHFL.IDX broadcast of its result after it."""
    n = 0
    for i, s in enumerate(ins):
        if s.startswith("@") and "ATOMG.E.ADD" in s:
            before, after = ins[max(0, i - back):i], ins[i + 1:i + 1 + fwd]
            if (any("VOTE" in b and ".ANY" in b for b in before)
                    and any("POPC" in b for b in before)
                    and any("SHFL.IDX" in a for a in after)):
                n += 1
    return n


@pytest.mark.skipif(not os.path.exists(NVDISASM), reason="nvdisasm not found")
@pytest.mark.parametrize("kernel,bare", [("global", False), ("global", True),
                                         ("dirsplit", False),
                                         ("dirsplit", True), ("split", False),
                                         ("split", True), ("pinned", False)])
def test_lane_enqueue_is_warp_aggregated_in_sass(kernel, bare, tmp_path):
    from .flood_fill import compiled_kernel
    tpb = PINNED_TPB if kernel == "pinned" else 256
    lane = _sass_instructions(
        compiled_kernel(kernel, bare, tpb, "lane").asm["cubin"],
        tmp_path / "lane.cubin")
    program = _sass_instructions(
        compiled_kernel(kernel, bare, tpb, "program").asm["cubin"],
        tmp_path / "program.cubin")
    # every per-lane append site is one leader atomic per warp (pinned's
    # pair-barrier and rank atomics may add aggregated adds of their own)
    assert _warp_aggregated_adds(lane) >= LANE_APPEND_SITES[kernel]
    assert sum("ATOMG.E.ADD" in s for s in lane) == \
        _warp_aggregated_adds(lane), "an add atomic is not warp-aggregated"
    # the first translation's scans are CTA barriers the lane path lacks
    bars = lambda ins: sum("BAR.SYNC" in s for s in ins)
    assert bars(program) >= bars(lane) + 20
