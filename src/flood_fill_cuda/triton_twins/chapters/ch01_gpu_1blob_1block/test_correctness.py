"""
Correctness tests: Triton twin of the single-block flood fill vs CPU reference.

Test for test the same file as chapters/ch01_gpu_1blob_1block/
test_correctness.py (same names, scenes, parameters and assertions against
the same @njit oracle), run against the Triton flood_fill. The GPU result
must match the 4-connectivity CPU reference exactly:
- visited mask (which pixels were filled)
- depth map (the BFS level of every pixel - catches level-mixing races that
  a visited-only comparison would miss)
- level count and filled count

The test_cross_backend_* section at the end runs both backends on the same
input and requires every deterministic output to be identical: the arrays,
the BFS shape counters, the spill counters, cas_attempts (deterministic
here: the 4-connected grid is bipartite, so every claim attempt is one edge
of the filled component) and the derived percentages. Only the *_ms
timings differ.

Run:

    .venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch01_gpu_1blob_1block/test_correctness.py -v
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import dataclasses
import re

import numpy as np
import pytest

from .flood_fill import flood_fill
from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
from flood_fill_cuda.chapters.ch01_gpu_1blob_1block import scenes

BLUE = np.array([0, 0, 255], dtype=np.uint8)


def assert_matches_reference(img, seed_x, seed_y, **gpu_kwargs):
    ref_visited, ref_depth, ref_levels, ref_filled = cpu_flood_fill(img, seed_x, seed_y)
    result = flood_fill(img, seed_x, seed_y, **gpu_kwargs)

    np.testing.assert_array_equal(result.visited, ref_visited)
    np.testing.assert_array_equal(result.depth, ref_depth)
    assert result.levels == ref_levels
    assert result.filled == ref_filled
    # Every reached pixel must be recolored solid blue; everything else keeps
    # its original color.
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
    # 4-connectivity percolation threshold is ~0.593 (vs ~0.5 for 8-conn),
    # so supercritical here means density 0.65.
    "random_subcritical": lambda: scenes.random_scene(200, 200, 0.35, rng_seed=7),
    "random_supercritical": lambda: scenes.random_scene(200, 200, 0.65, rng_seed=7),
    "single_pixel": lambda: scenes.single_pixel_scene(64, 64),
    "full_red_128": lambda: scenes.full_red_scene(128, 128),
}


VARIANTS = ["ring", "spill"]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", SCENES.keys())
def test_matches_cpu_reference(name, variant):
    img, sx, sy = SCENES[name]()
    assert_matches_reference(img, sx, sy, variant=variant)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("tpb", [64, 256, 1024])
def test_block_size_invariance(tpb, variant):
    img, sx, sy = scenes.random_scene(256, 256, 0.65, rng_seed=3)
    assert_matches_reference(img, sx, sy, threads_per_block=tpb,
                             variant=variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_deterministic_across_runs(variant):
    """Queue order is nondeterministic; visited/depth/counts must not be."""
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=11)
    a = flood_fill(img, sx, sy, variant=variant)
    b = flood_fill(img, sx, sy, variant=variant)
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


def test_only_red_pixels_ever_visited():
    """Regression for the old single_block.py flaw: CAS before the red check
    marked non-red pixels visited. visited must imply originally-red."""
    img, sx, sy = scenes.random_scene(200, 200, 0.65, rng_seed=5)
    originally_red = (img == scenes.RED).all(axis=2)
    result = flood_fill(img, sx, sy)
    assert (originally_red[result.visited == 1]).all()


def test_depth_is_true_bfs_distance():
    """Spot-check depth semantics on a known geometry: on a full-red image
    seeded at the corner, 4-connected BFS depth is the Manhattan distance
    x + y (the 4-conn analogue of persistent/'s Chebyshev check)."""
    img, sx, sy = scenes.full_red_scene(64, 64)
    result = flood_fill(img, sx, sy)
    xs, ys = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    np.testing.assert_array_equal(result.depth, xs + ys)
    assert result.levels == 127


def test_frontier_trace_consistency():
    """The per-level trace must account for every filled pixel and agree
    with the in-kernel utilization accumulators."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    tpb = 128
    result = flood_fill(img, sx, sy, threads_per_block=tpb)
    sizes = result.level_sizes.astype(np.int64)

    assert not result.level_trace_truncated
    assert len(sizes) == result.levels
    assert sizes[0] == 1  # the seed
    assert sizes.sum() == result.filled
    assert sizes.max() == result.peak_level
    # Recompute the utilization aggregates the kernel tracked in registers.
    active = np.minimum(sizes, tpb)
    expected_thread_util = 100.0 * active.sum() / (result.levels * tpb)
    engaged_warps = (active + 31) // 32
    expected_warp_engagement = 100.0 * engaged_warps.sum() / (result.levels * (tpb // 32))
    assert result.thread_util_pct == pytest.approx(expected_thread_util)
    assert result.warp_engagement_pct == pytest.approx(expected_warp_engagement)
    # Depth map and trace must tell the same story per level.
    depth_counts = np.bincount(result.depth[result.depth >= 0].ravel(),
                               minlength=result.levels)
    np.testing.assert_array_equal(depth_counts, sizes)


def test_exactly_once_processing_and_work_efficiency():
    """No double work: every filled pixel is dequeued exactly once, and the
    duplicated-discovery counter stays within its structural bounds."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    _, _, _, ref_filled = cpu_flood_fill(img, sx, sy)
    result = flood_fill(img, sx, sy)
    assert result.processed == result.filled == ref_filled
    # Each non-seed pixel was claimed by exactly one winning CAS, and each
    # processed pixel attempts at most 4 CAS ops.
    assert result.filled - 1 <= result.cas_attempts <= 4 * result.filled
    assert result.discovery_redundancy >= 1.0


def test_serpentine_starves_threads():
    """The utilization metrics must expose the serpentine's ~1-pixel
    frontiers vs the square's wide ones."""
    sq_img, sq_x, sq_y = scenes.square_scene(256, 256, 128, 128)
    sp_img, sp_x, sp_y = scenes.serpentine_scene(128, 128)
    sq = flood_fill(sq_img, sq_x, sq_y, threads_per_block=128)
    sp = flood_fill(sp_img, sp_x, sp_y, threads_per_block=128)
    assert sp.thread_util_pct < 5.0 < sq.thread_util_pct


def test_seed_not_red_raises():
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="not red"):
        flood_fill(img, 0, 0)


def test_seed_out_of_bounds_raises():
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="outside"):
        flood_fill(img, 64, 0)


@pytest.mark.parametrize("tpb", [100, 2048, 0])
def test_bad_threads_per_block_raises(tpb):
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="threads_per_block"):
        flood_fill(img, sx, sy, threads_per_block=tpb)


def test_overflow_tripwire_raises():
    """A center-seeded 2600^2 full-bleed square's frontier (~8r+4 ring
    occupancy) crosses the 8192-slot capacity at r~1024 and must abort
    loudly instead of returning a partial fill."""
    img, sx, sy = scenes.overflow_scene()
    with pytest.raises(RuntimeError, match="overflow"):
        flood_fill(img, sx, sy)


def test_bad_variant_raises():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="variant"):
        flood_fill(img, sx, sy, variant="turbo")


def test_spill_untouched_when_scene_fits():
    """On a ring-sized scene the v2 kernel must behave exactly like v1: the
    spill tier stays empty and the results are identical."""
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    ring = flood_fill(img, sx, sy, variant="ring")
    spill = flood_fill(img, sx, sy, variant="spill")
    np.testing.assert_array_equal(ring.visited, spill.visited)
    np.testing.assert_array_equal(ring.depth, spill.depth)
    assert ring.levels == spill.levels and ring.filled == spill.filled
    assert ring.peak_occupancy == spill.peak_occupancy
    assert spill.spilled == 0
    assert spill.peak_spill_window == 0


def test_spill_completes_the_tripwire_scene():
    """The exact scene that trips v1's ring must complete on v2 with a
    reference-exact result, spilled work accounted, and every guarantee the
    ring kernel makes (exactly-once processing, trace consistency) intact."""
    img, sx, sy = scenes.overflow_scene()
    ref_visited, ref_depth, ref_levels, ref_filled = cpu_flood_fill(img, sx, sy)
    result = flood_fill(img, sx, sy, variant="spill")

    np.testing.assert_array_equal(result.visited, ref_visited)
    np.testing.assert_array_equal(result.depth, ref_depth)
    assert result.levels == ref_levels
    assert result.filled == ref_filled
    # The scene overflows the ring by design, so the tier must have been used
    # and the two tiers together must account for every filled pixel.
    assert result.spilled > 0
    assert result.peak_spill_window > 0
    assert result.peak_occupancy > result.ring_capacity
    # Exactly-once processing holds through the spill path too.
    assert result.processed == result.filled == ref_filled
    assert result.filled - 1 <= result.cas_attempts <= 4 * result.filled
    # The fused two-tier frontier trace still accounts for every pixel.
    sizes = result.level_sizes.astype(np.int64)
    assert not result.level_trace_truncated
    assert sizes.sum() == result.filled
    assert sizes.max() == result.peak_level
    depth_counts = np.bincount(result.depth[result.depth >= 0].ravel(),
                               minlength=result.levels)
    np.testing.assert_array_equal(depth_counts, sizes)


# ---------------------------------------------------------------------------
# Cross-backend: the Triton twin and the Numba original on the same input.
# ---------------------------------------------------------------------------

from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.flood_fill import (  # noqa: E402
    flood_fill as numba_flood_fill,
)

# Schedule-free fields are compared; only the wall-clock timings may differ.
TIMING_FIELDS = {"alloc_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms"}


def assert_same_as_numba(tri, nb):
    for f in dataclasses.fields(nb):
        if f.name in TIMING_FIELDS:
            continue
        a, b = getattr(tri, f.name), getattr(nb, f.name)
        if isinstance(b, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, f.name
            np.testing.assert_array_equal(a, b, err_msg=f.name)
        else:
            assert a == b, f"{f.name}: triton {a!r} != numba {b!r}"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", SCENES.keys())
def test_cross_backend_matches_numba(name, variant):
    img, sx, sy = SCENES[name]()
    assert_same_as_numba(flood_fill(img, sx, sy, variant=variant),
                         numba_flood_fill(img, sx, sy, variant=variant))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("tpb", [32, 64, 128, 256, 512, 1024])
def test_cross_backend_block_size_invariance(tpb, variant):
    """Every power-of-2 block size, including the utilization counters
    (they depend on tpb, so both sides run the same tpb)."""
    img, sx, sy = scenes.random_scene(256, 256, 0.65, rng_seed=3)
    assert_same_as_numba(
        flood_fill(img, sx, sy, threads_per_block=tpb, variant=variant),
        numba_flood_fill(img, sx, sy, threads_per_block=tpb, variant=variant))


def test_cross_backend_spill_tripwire_scene():
    """The 6.76M px scene that trips v1: identical spill counts, peak
    occupancy, trace and arrays on both backends."""
    img, sx, sy = scenes.overflow_scene()
    tri = flood_fill(img, sx, sy, variant="spill")
    assert tri.spilled > 0
    nb = numba_flood_fill(img, sx, sy, variant="spill")
    assert_same_as_numba(tri, nb)


def test_cross_backend_overflow_tripwire():
    """Both ring kernels trip, and report the same largest completed-level
    occupancy (the abort level is a function of the BFS layer sizes)."""
    img, sx, sy = scenes.overflow_scene()
    occupancy = []
    for ff in (flood_fill, numba_flood_fill):
        with pytest.raises(RuntimeError, match="overflow") as info:
            ff(img, sx, sy, variant="ring")
        occupancy.append(re.search(r"occupancy: (\d+)", str(info.value)).group(1))
    assert occupancy[0] == occupancy[1]


def test_cross_backend_tpb_768_numba_accepts_triton_rejects():
    """768 threads is a valid Numba block but not a Triton program
    (num_warps must be a power of 2): the twin refuses it by name."""
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    nb = numba_flood_fill(img, sx, sy, threads_per_block=768, variant="spill")
    assert nb.threads_per_block == 768
    with pytest.raises(ValueError, match="threads_per_block must be a power of 2"):
        flood_fill(img, sx, sy, threads_per_block=768, variant="spill")


def test_cross_backend_no_recompile_inside_timing():
    """After the warm-up, scenes of other sizes and seeds reuse the compiled
    kernel, so no Triton compile ever lands inside kernel_ms."""
    from flood_fill_cuda.triton_twins.runtime import bridge  # noqa: F401
    from triton.runtime.driver import driver
    from .kernels import single_block_bfs_kernel, single_block_bfs_spill_kernel

    dev = driver.active.get_current_device()
    sizes = {}
    for variant, kernel in (("ring", single_block_bfs_kernel),
                            ("spill", single_block_bfs_spill_kernel)):
        flood_fill(*scenes.square_scene(64, 64, 32, 32), variant=variant)
        before = len(kernel.device_caches[dev][0])
        for build in (lambda: scenes.square_scene(97, 33, 51, 17),
                      lambda: scenes.serpentine_scene(48, 80),
                      lambda: scenes.full_red_scene(129, 1)):
            flood_fill(*build(), variant=variant)
        sizes[variant] = (before, len(kernel.device_caches[dev][0]))
    assert all(b == a for b, a in sizes.values()), sizes
