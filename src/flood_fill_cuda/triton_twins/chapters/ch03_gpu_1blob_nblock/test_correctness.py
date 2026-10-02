"""Correctness tests: the Triton twin vs the same CPU references.

A test-for-test mirror of chapters/ch03_gpu_1blob_nblock/test_correctness.py
(same names, scenes, parameters and assertions; only the implementation
under test changes), followed by a cross-backend section that runs the
Numba original and this twin on identical inputs and grids and asserts
their deterministic outputs are identical.

The kernel must match the reference exactly (visited mask, depth map -
which catches level-mixing races a visited-only check would miss - level
and filled counts) at EVERY block count, recolor reached pixels solid blue,
and leave everything else untouched.

Run:

    uv run pytest src/flood_fill_cuda/triton_twins/chapters/ch03_gpu_1blob_nblock/test_correctness.py -v
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import numpy as np
import pytest

from flood_fill_cuda.shared import bandwidth
from flood_fill_cuda.triton_twins.chapters.ch03_gpu_1blob_nblock.flood_fill import (
    flood_fill, max_blocks,
)
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock.cpu_oracle import (
    cpu_flood_fill, cpu_flood_fill_8,
)
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import scenes

BLUE = np.array([0, 0, 255], dtype=np.uint8)


def assert_matches_reference(img, seed_x, seed_y, connectivity=4, **gpu_kwargs):
    ref = cpu_flood_fill_8 if connectivity == 8 else cpu_flood_fill
    ref_visited, ref_depth, ref_levels, ref_filled = ref(img, seed_x, seed_y)
    result = flood_fill(img, seed_x, seed_y, connectivity=connectivity,
                        **gpu_kwargs)

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
}


@pytest.mark.parametrize("name", SCENES.keys())
def test_matches_cpu_reference(name):
    """Default config = the maximum cooperative grid this GPU can host."""
    img, sx, sy = SCENES[name]()
    assert_matches_reference(img, sx, sy)


@pytest.mark.parametrize("blocks", [1, 2, 3, 8, None])
@pytest.mark.parametrize("tpb", [64, 256, 512])
def test_block_count_and_tpb_invariance(blocks, tpb):
    """The BFS result must not depend on grid shape - including blocks=1
    (grid.sync degenerates) and the deliberately odd blocks=3."""
    img, sx, sy = scenes.random_scene(256, 256, 0.65, rng_seed=3)
    assert_matches_reference(img, sx, sy, threads_per_block=tpb, blocks=blocks)


def test_deterministic_across_runs():
    """Queue order and which block claims a pixel are race-dependent, so
    per-block counts may differ between runs; visited/depth/levels/filled
    must not."""
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=11)
    a = flood_fill(img, sx, sy)
    b = flood_fill(img, sx, sy)
    np.testing.assert_array_equal(a.visited, b.visited)
    np.testing.assert_array_equal(a.depth, b.depth)
    assert a.levels == b.levels and a.filled == b.filled


def test_blocks_1_equals_blocks_max():
    """One block and the full cooperative grid compute the identical BFS."""
    img, sx, sy = scenes.disk_scene(256, 256, 100)
    one = flood_fill(img, sx, sy, blocks=1)
    full = flood_fill(img, sx, sy, blocks=None)
    assert one.blocks == 1 and full.blocks > 1
    np.testing.assert_array_equal(one.visited, full.visited)
    np.testing.assert_array_equal(one.depth, full.depth)
    assert one.filled == full.filled and one.levels == full.levels


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


def test_only_red_pixels_ever_visited():
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=5)
    originally_red = (img == scenes.RED).all(axis=2)
    result = flood_fill(img, sx, sy)
    assert (originally_red[result.visited == 1]).all()


@pytest.mark.parametrize("blocks", [2, None])
def test_depth_is_true_bfs_distance(blocks):
    """On a full-red image seeded at the corner, 4-connected BFS depth is
    the Manhattan distance x + y - at any block count, which proves the
    grid-wide level barrier preserves level-exactness."""
    img, sx, sy = scenes.full_red_scene(64, 64)
    result = flood_fill(img, sx, sy, blocks=blocks)
    xs, ys = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    np.testing.assert_array_equal(result.depth, xs + ys)
    assert result.levels == 127


def test_exactly_once_processing():
    """Every filled pixel is dequeued exactly once, and the per-block counts
    account for all of it."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    _, _, _, ref_filled = cpu_flood_fill(img, sx, sy)
    result = flood_fill(img, sx, sy)
    assert result.processed == result.filled == ref_filled
    assert result.processed_per_block.sum() == result.processed
    assert result.filled - 1 <= result.cas_attempts <= 4 * result.filled


def test_frontier_trace_consistency():
    """The grid-wide per-level trace must account for every pixel and agree
    with the depth map and the in-kernel utilization accumulators. The
    grid-stride kernel assigns work by GLOBAL thread id, so active threads
    per level are min(level_size, blocks*tpb) contiguous busy threads."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    tpb = 128
    result = flood_fill(img, sx, sy, threads_per_block=tpb, blocks=4)
    sizes = result.level_sizes.astype(np.int64)

    assert not result.level_trace_truncated
    assert len(sizes) == result.levels
    assert sizes[0] == 1
    assert sizes.sum() == result.filled
    assert sizes.max() == result.peak_level
    grid_threads = result.blocks * tpb
    active = np.minimum(sizes, grid_threads)
    expected_thread_util = 100.0 * active.sum() / (result.levels * grid_threads)
    engaged = (active + 31) // 32
    expected_warp_engagement = (100.0 * engaged.sum()
                                / (result.levels * (grid_threads // 32)))
    assert result.thread_util_pct == pytest.approx(expected_thread_util)
    assert result.warp_engagement_pct == pytest.approx(expected_warp_engagement)
    depth_counts = np.bincount(result.depth[result.depth >= 0].ravel(),
                               minlength=result.levels)
    np.testing.assert_array_equal(depth_counts, sizes)


def test_owner_census():
    """owner[x,y] = claiming block, everywhere reached; -1 elsewhere; and
    the per-pixel census must agree exactly with the per-block counters."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    result = flood_fill(img, sx, sy, blocks=4)
    assert result.owner.dtype == np.int16
    reached = result.visited == 1
    owners = result.owner[reached]
    assert owners.min() >= 0 and owners.max() < result.blocks
    assert (result.owner[~reached] == -1).all()
    census = np.bincount(owners.astype(np.int64), minlength=result.blocks)
    np.testing.assert_array_equal(census, result.processed_per_block)


def test_all_blocks_participate():
    """On a scene whose frontiers dwarf the grid, no block may sit idle -
    the grid-stride partition guarantees work reaches every block. (The
    corner-seeded full-red diagonal peaks at width pixels, so the grid must
    stay below that: 8 x 64 = 512 threads < 1024-pixel peak frontiers.)"""
    img, sx, sy = scenes.full_red_scene(1024, 1024)
    result = flood_fill(img, sx, sy, blocks=8, threads_per_block=64)
    assert result.blocks == 8
    assert (result.processed_per_block > 0).all()


def test_smids_recorded():
    """%smid is observed per block. Report, don't assume: assert the values
    are recorded and plausible, never a particular placement."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    result = flood_fill(img, sx, sy, blocks=8)
    assert len(result.sm_ids) == 8
    assert all(s >= 0 for s in result.sm_ids)
    assert 1 <= result.distinct_sms <= result.blocks
    assert result.sm_utilization_pct > 0


# ----------------------------------------------------------------- bare twin

def test_bare_twin_matches_reference():
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=9)
    assert_matches_reference(img, sx, sy, bare=True)


def test_bare_twin_reports_no_instrumentation():
    img, sx, sy = scenes.square_scene(128, 128, 64, 64)
    result = flood_fill(img, sx, sy, bare=True)
    assert result.bare
    assert result.processed == 0 and result.cas_attempts == 0
    assert result.owner.size == 0
    assert result.level_sizes.size == 0
    assert all(s == -1 for s in result.sm_ids)
    assert result.model_bytes == 0 and result.model_gb_s == 0.0
    assert result.filled > 0 and result.levels > 0


# ------------------------------------------------------------ bandwidth model

def test_model_bytes_consistency():
    """The result's bandwidth figure must be exactly the documented formula
    applied to its own counters."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    result = flood_fill(img, sx, sy)
    expected = bandwidth.model_bytes(result.processed, result.cas_attempts,
                                     result.filled, instrumented=True)
    assert result.model_bytes == expected
    assert result.model_gb_s > 0
    assert result.model_gb_s == pytest.approx(
        bandwidth.model_gb_s(expected, result.kernel_ms))


# ------------------------------------------------------- 8-connectivity twins

# The 8-conn kernels are different algorithms with different results: depth
# is Chebyshev distance, and random scenes percolate at ~0.407 instead of
# the 4-conn threshold - hence their own density bracket (0.30 / 0.60).
SCENES8 = dict(SCENES)
SCENES8["random_subcritical"] = lambda: scenes.random_scene(200, 200, 0.30,
                                                            rng_seed=7)
SCENES8["random_supercritical"] = lambda: scenes.random_scene(200, 200, 0.60,
                                                              rng_seed=7)


@pytest.mark.parametrize("name", SCENES8.keys())
def test_matches_cpu_reference_8(name):
    img, sx, sy = SCENES8[name]()
    assert_matches_reference(img, sx, sy, connectivity=8)


@pytest.mark.parametrize("blocks", [2, None])
def test_depth_is_chebyshev_distance(blocks):
    """On a full-red image seeded at the corner, 8-connected BFS depth is
    the Chebyshev distance max(x, y) - square waves, not diamonds - and a
    64x64 image fills in 64 levels instead of the 4-conn 127."""
    img, sx, sy = scenes.full_red_scene(64, 64)
    result = flood_fill(img, sx, sy, blocks=blocks, connectivity=8)
    xs, ys = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    np.testing.assert_array_equal(result.depth, np.maximum(xs, ys))
    assert result.levels == 64


@pytest.mark.parametrize("blocks", [1, 3, None])
@pytest.mark.parametrize("tpb", [64, 512])
def test_block_count_and_tpb_invariance_8(blocks, tpb):
    img, sx, sy = scenes.random_scene(256, 256, 0.60, rng_seed=3)
    assert_matches_reference(img, sx, sy, connectivity=8,
                             threads_per_block=tpb, blocks=blocks)


def test_exactly_once_processing_8():
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    _, _, _, ref_filled = cpu_flood_fill_8(img, sx, sy)
    result = flood_fill(img, sx, sy, connectivity=8)
    assert result.processed == result.filled == ref_filled
    assert result.processed_per_block.sum() == result.processed
    assert result.filled - 1 <= result.cas_attempts <= 8 * result.filled


def test_owner_census_8():
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    result = flood_fill(img, sx, sy, blocks=4, connectivity=8)
    reached = result.visited == 1
    census = np.bincount(result.owner[reached].astype(np.int64),
                         minlength=result.blocks)
    np.testing.assert_array_equal(census, result.processed_per_block)


def test_bare_twin_matches_reference_8():
    img, sx, sy = scenes.random_scene(200, 200, 0.60, rng_seed=9)
    assert_matches_reference(img, sx, sy, connectivity=8, bare=True)


@pytest.mark.parametrize("name", ["square_64", "disk_101", "serpentine_128"])
def test_4_vs_8_on_solid_scenes(name):
    """On scenes without diagonal-only gaps the two connectivities fill the
    IDENTICAL pixel set - only the timeline differs: 8-conn reaches every
    pixel at least as early (Chebyshev <= Manhattan) in strictly fewer
    levels (serpentine: <=, its 1-px corridors only save U-turn corners)."""
    img, sx, sy = SCENES[name]()
    r4 = flood_fill(img, sx, sy, connectivity=4)
    r8 = flood_fill(img, sx, sy, connectivity=8)
    np.testing.assert_array_equal(r4.visited, r8.visited)
    assert r4.filled == r8.filled
    if name == "serpentine_128":
        assert r8.levels <= r4.levels
    else:
        assert r8.levels < r4.levels
    reached = r4.visited == 1
    assert (r8.depth[reached] <= r4.depth[reached]).all()


def test_model_bytes_consistency_8():
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    result = flood_fill(img, sx, sy, connectivity=8)
    expected = bandwidth.model_bytes(result.processed, result.cas_attempts,
                                     result.filled, instrumented=True,
                                     n_dirs=8)
    assert result.model_bytes == expected
    assert result.model_gb_s > 0


@pytest.mark.parametrize("conn", [5, 0, "8"])
def test_rejects_bad_connectivity(conn):
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="connectivity"):
        flood_fill(img, sx, sy, connectivity=conn)


# ------------------------------------------------------ radius-2 twins (guarded)

# The guarded radius-2 kernel shares conn8's FILL SET exactly (the guard
# keeps every ring-2 jump inside true 8-connectivity) but not its
# depth/levels: BFS on a supergraph advances Chebyshev distance 2 through
# interior, so assert_matches_reference (which checks depth) is unusable.
# The guard is static - "all 8 ring-1 in-bounds and originally red" - so
# depth/levels/interior are still fully deterministic and grid-invariant.


def assert_r2_fill_set(img, seed_x, seed_y, **gpu_kwargs):
    ref_visited, _, _, ref_filled = cpu_flood_fill_8(img, seed_x, seed_y)
    result = flood_fill(img, seed_x, seed_y, connectivity=8, radius=2,
                        **gpu_kwargs)
    np.testing.assert_array_equal(result.visited, ref_visited)
    assert result.filled == ref_filled
    filled_mask = result.visited == 1
    assert (result.img[filled_mask] == BLUE).all()
    np.testing.assert_array_equal(result.img[~filled_mask], img[~filled_mask])
    return result


@pytest.mark.parametrize("name", SCENES8.keys())
def test_r2_fill_set_matches_reference(name):
    """Random scenes are the best available 1-px-gap fuzzer for the guard."""
    img, sx, sy = SCENES8[name]()
    assert_r2_fill_set(img, sx, sy)


def test_r2_never_leaks_across_gap():
    """THE guard test: two components at Chebyshev distance 2 (one white
    row between two red stripes). An unguarded 5x5 neighborhood would jump
    the gap; the guard must not - every stripe-edge pixel has a non-blob
    ring-1 neighbor, so no pixel whose ring-2 reaches the far stripe ever
    fires its second ring."""
    img = np.full((60, 60, 3), 255, dtype=np.uint8)
    img[10:20, :] = scenes.RED   # stripe A (seeded)
    img[21:31, :] = scenes.RED   # stripe B, one white row away
    result = flood_fill(img, 15, 30, connectivity=8, radius=2)
    assert result.visited[10:20, :].all()
    assert result.visited[21:31, :].sum() == 0
    np.testing.assert_array_equal(result.img[21:31, :], img[21:31, :])


def test_r2_serpentine_guard_never_fires():
    """1-px corridors: no pixel has an all-red ring-1, so interior == 0 and
    the supergraph degenerates to plain conn8 - fill set AND depth
    identical. This is the pure-guard-overhead scene of the benchmark."""
    img, sx, sy = scenes.serpentine_scene(128, 128)
    r2 = flood_fill(img, sx, sy, connectivity=8, radius=2)
    r8 = flood_fill(img, sx, sy, connectivity=8)
    assert r2.interior == 0
    np.testing.assert_array_equal(r2.visited, r8.visited)
    np.testing.assert_array_equal(r2.depth, r8.depth)
    assert r2.levels == r8.levels


def test_r2_depth_dominance_and_fewer_levels():
    """Supergraph BFS: no pixel is reached later than conn8 reaches it, and
    a solid blob takes strictly fewer levels (~half)."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    r2 = flood_fill(img, sx, sy, connectivity=8, radius=2)
    r8 = flood_fill(img, sx, sy, connectivity=8)
    reached = r8.visited == 1
    assert (r2.depth[reached] <= r8.depth[reached]).all()
    assert r2.levels < r8.levels


def test_r2_interior_census_full_red():
    """The guard is static, so on a full-red image interior == the count of
    non-border pixels, exactly."""
    img, sx, sy = scenes.full_red_scene(96, 96)
    result = flood_fill(img, sx, sy, connectivity=8, radius=2)
    assert result.interior == 94 * 94
    assert result.levels < 96  # conn8 would take 96


def test_r2_exactly_once_processing():
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    _, _, _, ref_filled = cpu_flood_fill_8(img, sx, sy)
    result = flood_fill(img, sx, sy, connectivity=8, radius=2)
    assert result.processed == result.filled == ref_filled
    assert result.processed_per_block.sum() == result.processed
    assert result.filled - 1 <= result.cas_attempts <= 24 * result.filled
    assert 0 <= result.interior <= result.processed


@pytest.mark.parametrize("blocks", [1, 3, None])
@pytest.mark.parametrize("tpb", [64, 512])
def test_r2_block_count_and_tpb_invariance(blocks, tpb):
    """Depth/levels/interior are grid-invariant too (static guard), so full
    determinism across shapes is asserted against a baseline run."""
    img, sx, sy = scenes.random_scene(256, 256, 0.60, rng_seed=3)
    a = flood_fill(img, sx, sy, connectivity=8, radius=2)
    b = flood_fill(img, sx, sy, connectivity=8, radius=2,
                   threads_per_block=tpb, blocks=blocks)
    np.testing.assert_array_equal(a.visited, b.visited)
    np.testing.assert_array_equal(a.depth, b.depth)
    assert a.levels == b.levels and a.filled == b.filled
    assert a.interior == b.interior


def test_r2_bare_twin_parity():
    """The bare twin runs the identical supergraph BFS - visited/depth/
    levels/filled equality against the instrumented run."""
    img, sx, sy = scenes.random_scene(200, 200, 0.60, rng_seed=9)
    inst = flood_fill(img, sx, sy, connectivity=8, radius=2)
    bare = flood_fill(img, sx, sy, connectivity=8, radius=2, bare=True)
    np.testing.assert_array_equal(inst.visited, bare.visited)
    np.testing.assert_array_equal(inst.depth, bare.depth)
    assert inst.levels == bare.levels and inst.filled == bare.filled


def test_r2_model_bytes_consistency():
    """r2's model uses the exact probe count 8*processed + 16*interior."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    result = flood_fill(img, sx, sy, connectivity=8, radius=2)
    expected = bandwidth.model_bytes(
        result.processed, result.cas_attempts, result.filled,
        instrumented=True, n_dirs=8,
        probe_reads=8 * result.processed + 16 * result.interior)
    assert result.model_bytes == expected
    assert result.model_gb_s > 0


@pytest.mark.parametrize("kwargs", [
    {"connectivity": 4, "radius": 2},
    {"connectivity": 8, "radius": 3},
    {"connectivity": 8, "radius": 0},
])
def test_rejects_bad_radius(kwargs):
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="radius"):
        flood_fill(img, sx, sy, **kwargs)


# ------------------------------------------------- warp-cooperative twins

# Same BFS graph as conn8 - only the work distribution changes (4 entries
# x 8 directions across a warp's 32 lanes), so results must be
# bit-identical to conn8's and the full reference assertion applies
# unchanged, depth and levels included.


@pytest.mark.parametrize("name", SCENES8.keys())
def test_wc_matches_cpu_reference(name):
    img, sx, sy = SCENES8[name]()
    assert_matches_reference(img, sx, sy, connectivity=8,
                             probe_layout="warp")


def test_wc_equals_conn8_gpu_result():
    img, sx, sy = scenes.random_scene(200, 200, 0.60, rng_seed=13)
    wc = flood_fill(img, sx, sy, connectivity=8, probe_layout="warp")
    r8 = flood_fill(img, sx, sy, connectivity=8)
    np.testing.assert_array_equal(wc.visited, r8.visited)
    np.testing.assert_array_equal(wc.depth, r8.depth)
    assert wc.levels == r8.levels and wc.filled == r8.filled


@pytest.mark.parametrize("blocks", [1, 3, None])
@pytest.mark.parametrize("tpb", [64, 512])
def test_wc_block_count_and_tpb_invariance(blocks, tpb):
    """Partial final chunks (rear not a multiple of 4) are exercised by any
    random scene; odd grids by blocks=3."""
    img, sx, sy = scenes.random_scene(256, 256, 0.60, rng_seed=3)
    assert_matches_reference(img, sx, sy, connectivity=8,
                             probe_layout="warp",
                             threads_per_block=tpb, blocks=blocks)


def test_wc_exactly_once_processing():
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    _, _, _, ref_filled = cpu_flood_fill_8(img, sx, sy)
    result = flood_fill(img, sx, sy, connectivity=8, probe_layout="warp")
    assert result.processed == result.filled == ref_filled
    assert result.processed_per_block.sum() == result.processed
    assert result.filled - 1 <= result.cas_attempts <= 8 * result.filled


def test_wc_bare_twin_matches_reference():
    img, sx, sy = scenes.random_scene(200, 200, 0.60, rng_seed=9)
    assert_matches_reference(img, sx, sy, connectivity=8,
                             probe_layout="warp", bare=True)


@pytest.mark.parametrize("kwargs", [
    {"connectivity": 4, "probe_layout": "warp"},
    {"connectivity": 8, "radius": 2, "probe_layout": "warp"},
    {"connectivity": 8, "probe_layout": "wide"},
])
def test_rejects_bad_probe_layout(kwargs):
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="probe_layout"):
        flood_fill(img, sx, sy, **kwargs)


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("tpb", [100, 0, 1024, 2048])
def test_rejects_bad_threads_per_block(tpb):
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="threads_per_block"):
        flood_fill(img, sx, sy, threads_per_block=tpb)


@pytest.mark.parametrize("blocks", [0, -1])
def test_rejects_bad_block_count(blocks):
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="blocks"):
        flood_fill(img, sx, sy, blocks=blocks)


def test_rejects_blocks_beyond_cooperative_capacity():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    coop_max = max_blocks(threads_per_block=256)
    with pytest.raises(RuntimeError, match="cooperative"):
        flood_fill(img, sx, sy, blocks=coop_max + 1)


def test_rejects_non_red_seed():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="not red"):
        flood_fill(img, 0, 0)


def test_rejects_out_of_bounds_seed():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="outside"):
        flood_fill(img, 64, 0)


# ------------------------------------------------------------ twin-only rule

@pytest.mark.parametrize("tpb", [96, 160, 480])
def test_twin_rejects_non_power_of_2_threads_per_block(tpb):
    """Numba accepts any multiple of 32 up to 512; a Triton program's lane
    count must be a power of 2, and the error names that rule."""
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="threads_per_block must be a power of 2"):
        flood_fill(img, sx, sy, threads_per_block=tpb)


# ------------------------------------------------------------- cross-backend
#
# The Numba original and the twin on identical inputs AND identical grids
# (explicit blocks and tpb: the two backends' co-resident capacities
# differ, so blocks=None would compare different launches). Asserted
# equal: every output the algorithm fixes regardless of scheduling. Never
# compared: cas_attempts outside conn4 (same-level 8-neighbors race with
# painting), the owner map's spatial layout (queue order), sm_ids, timings.

from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import kernels as numba_kernels
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock.flood_fill import (
    flood_fill as numba_flood_fill, max_blocks as numba_max_blocks,
)
from flood_fill_cuda.triton_twins.chapters.ch03_gpu_1blob_nblock import (
    kernels as twin_kernels,
)

VARIANTS = {
    "conn4": {},
    "conn4_bare": {"bare": True},
    "conn8": {"connectivity": 8},
    "conn8_bare": {"connectivity": 8, "bare": True},
    "r2": {"connectivity": 8, "radius": 2},
    "r2_bare": {"connectivity": 8, "radius": 2, "bare": True},
    "wc": {"connectivity": 8, "probe_layout": "warp"},
    "wc_bare": {"connectivity": 8, "probe_layout": "warp", "bare": True},
}
INSTRUMENTED = [v for v, kw in VARIANTS.items() if not kw.get("bare")]
CROSS_SCENES = ["square_nonsquare_img", "disk_101", "serpentine_nonsquare",
                "random_supercritical", "full_red_128"]
CROSS_GRIDS = [(3, 64), (8, 256)]  # (blocks, tpb): odd grid, wide grid


def _scene(name, kw):
    return (SCENES8 if kw.get("connectivity") == 8 else SCENES)[name]()


def _pinned(blocks, tpb, kw):
    """The requested grid, capped at both backends' capacity."""
    return min(blocks, numba_max_blocks(threads_per_block=tpb, **kw),
               max_blocks(threads_per_block=tpb, **kw))


def _adjacent_pairs_4(mask):
    """4-adjacent pixel pairs inside a mask: conn4's exact cas_attempts."""
    return int((mask[1:, :] & mask[:-1, :]).sum()
               + (mask[:, 1:] & mask[:, :-1]).sum())


def assert_same_deterministic_outputs(n, t, kw):
    np.testing.assert_array_equal(t.img, n.img)
    np.testing.assert_array_equal(t.visited, n.visited)
    np.testing.assert_array_equal(t.depth, n.depth)
    for f in ("levels", "filled", "blocks", "threads_per_block", "bare",
              "connectivity", "radius", "probe_layout", "processed",
              "interior", "peak_level", "peak_occupancy",
              "level_trace_truncated", "grid_occupancy_pct",
              "thread_util_pct", "warp_engagement_pct",
              "lane_efficiency_pct", "balance_min_max_pct",
              "balance_cv_pct", "neighbor_check_efficiency_pct"):
        assert getattr(t, f) == pytest.approx(getattr(n, f)), f
    np.testing.assert_array_equal(t.level_sizes, n.level_sizes)
    np.testing.assert_array_equal(t.processed_per_block, n.processed_per_block)
    assert t.owner.shape == n.owner.shape and t.owner.dtype == n.owner.dtype
    if kw.get("bare"):
        assert t.processed == t.cas_attempts == t.model_bytes == 0
        assert t.sm_ids == n.sm_ids == [-1] * t.blocks
    else:
        # the owner census is fixed by the grid; the spatial map is not
        reached = t.visited == 1
        for r in (n, t):
            census = np.bincount(r.owner[reached].astype(np.int64),
                                 minlength=r.blocks)
            np.testing.assert_array_equal(census, t.processed_per_block)
            assert (r.owner[~reached] == -1).all()
    if kw.get("connectivity", 4) == 4 and not kw.get("bare"):
        # bipartite grid graph: exactly one probe per edge of the component
        assert t.cas_attempts == n.cas_attempts
        assert t.model_bytes == n.model_bytes
        assert t.discovery_redundancy == n.discovery_redundancy


def test_cross_backend_slots_and_tables_match():
    for name in ("FILLED", "LEVELS", "OVERFLOW", "PEAK_LEVEL", "PEAK_OCC",
                 "ACTIVE_THREAD_SUM", "ACTIVE_WARP_SUM", "PROCESSED",
                 "CAS_ATTEMPTS", "INTERIOR", "NUM_COUNTERS", "BS_PROCESSED",
                 "BS_SMID", "Q_REAR"):
        assert getattr(twin_kernels, name) == getattr(numba_kernels, name)
    for name in ("DX_HOST", "DY_HOST", "DX8_HOST", "DY8_HOST", "DX_R2_HOST",
                 "DY_R2_HOST"):
        np.testing.assert_array_equal(getattr(twin_kernels, name),
                                      getattr(numba_kernels, name))
    assert tuple(twin_kernels.DX_R2.value) == tuple(numba_kernels.DX_R2_HOST)
    assert tuple(twin_kernels.DY8.value) == tuple(numba_kernels.DY8_HOST)


@pytest.mark.parametrize("grid", CROSS_GRIDS, ids=lambda g: f"{g[0]}x{g[1]}")
@pytest.mark.parametrize("scene", CROSS_SCENES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_cross_backend_identical_outputs(variant, scene, grid):
    kw = VARIANTS[variant]
    img, sx, sy = _scene(scene, kw)
    blocks = _pinned(grid[0], grid[1], kw)
    n = numba_flood_fill(img, sx, sy, threads_per_block=grid[1],
                         blocks=blocks, **kw)
    t = flood_fill(img, sx, sy, threads_per_block=grid[1], blocks=blocks,
                   **kw)
    assert_same_deterministic_outputs(n, t, kw)


@pytest.mark.parametrize("variant", VARIANTS)
def test_cross_backend_identical_outputs_at_scale(variant):
    """A 1M-pixel disk at the chapter's benchmark grid (48 x 256, capped at
    both capacities): many programs racing on wide frontiers."""
    kw = VARIANTS[variant]
    img, sx, sy = scenes.disk_scene(1001, 1001, 480)
    blocks = _pinned(48, 256, kw)
    n = numba_flood_fill(img, sx, sy, blocks=blocks, **kw)
    t = flood_fill(img, sx, sy, blocks=blocks, **kw)
    assert_same_deterministic_outputs(n, t, kw)


@pytest.mark.parametrize("variant", INSTRUMENTED)
def test_cross_backend_owner_map_at_one_block(variant):
    """With one program every pixel is owned by block 0 on both sides, so
    even the spatial owner map is deterministic."""
    kw = VARIANTS[variant]
    img, sx, sy = _scene("random_supercritical", kw)
    n = numba_flood_fill(img, sx, sy, blocks=1, **kw)
    t = flood_fill(img, sx, sy, blocks=1, **kw)
    assert_same_deterministic_outputs(n, t, kw)
    np.testing.assert_array_equal(t.owner, n.owner)


@pytest.mark.parametrize("scene", ["square_64", "disk_101", "full_red_128",
                                   "random_supercritical"])
def test_cross_backend_conn4_cas_attempts_is_edge_count(scene):
    """conn4 probes each edge of the filled component exactly once, so a
    stale red read after the grid barrier (a memory-ordering bug) would
    show up as an extra attempt on either backend."""
    img, sx, sy = SCENES[scene]()
    blocks = _pinned(48, 256, {})
    n = numba_flood_fill(img, sx, sy, blocks=blocks)
    t = flood_fill(img, sx, sy, blocks=blocks)
    edges = _adjacent_pairs_4(n.visited == 1)
    assert n.cas_attempts == t.cas_attempts == edges


@pytest.mark.parametrize("variant", VARIANTS)
def test_cross_backend_capacity_is_reported_per_backend(variant):
    """blocks=None resolves to each backend's own co-resident maximum (they
    differ: the register counts differ); both must be honoured and refuse
    one more block with the same RuntimeError."""
    kw = VARIANTS[variant]
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    cap_t = max_blocks(threads_per_block=256, **kw)
    cap_n = numba_max_blocks(threads_per_block=256, **kw)
    assert flood_fill(img, sx, sy, **kw).blocks == cap_t
    assert numba_flood_fill(img, sx, sy, **kw).blocks == cap_n
    with pytest.raises(RuntimeError, match="cooperative"):
        flood_fill(img, sx, sy, blocks=cap_t + 1, **kw)


def test_twin_no_recompile_inside_the_timed_launch():
    """The warm-up compiles the exact binary the timed launch uses: image
    sizes of every divisibility class reuse the one specialization per
    (kernel, tpb), so no compile can land inside kernel_ms."""
    from flood_fill_cuda.triton_twins.chapters.ch03_gpu_1blob_nblock.flood_fill import (
        _KERNELS,
    )
    kernel = _KERNELS[(False, 4, 1, "thread")]

    def compiled():
        return sum(len(c[0]) for c in kernel.device_caches.values())

    flood_fill(*scenes.square_scene(64, 64, 20, 20), threads_per_block=128)
    before = compiled()
    assert before >= 1
    for w, h in [(101, 37), (96, 160), (33, 1), (200, 130)]:
        img = np.full((w, h, 3), 255, dtype=np.uint8)
        img[:, :] = scenes.RED
        flood_fill(img, 0, 0, threads_per_block=128, blocks=3)
    assert compiled() == before
