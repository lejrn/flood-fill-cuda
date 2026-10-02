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

The test_enqueue_* section runs both v2 enqueue forms (enqueue="lane",
the default, and enqueue="program", the first translation) against the
CPU oracle and Numba: the chapter's scenes, the spill scene, and
center-seeded squares at the ring/spill window edge (0, 2, 14 and 2,702
spilled pixels) at 32, 256 and 1024 threads. It also pins every default
to "lane" (at the launch itself) and checks the codegen claim: the
default kernels carry Numba's CTA barrier count and every enqueue atomic
is warp-aggregated by ptxas in the SASS. test_compare_marks_repeated_cells
checks which compare.py rows carry duplicate_of.

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


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("seed_type", [np.int32, np.int64])
def test_cross_backend_numpy_int_seeds(seed_type, variant):
    """Seeds taken from NumPy (np.argwhere, arrays) work on both backends:
    Numba types them as kernel args, the twin converts them to int."""
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    tri = flood_fill(img, seed_type(sx), seed_type(sy), variant=variant)
    nb = numba_flood_fill(img, seed_type(sx), seed_type(sy), variant=variant)
    assert tri.filled == 400
    assert_same_as_numba(tri, nb)


def test_cross_backend_check_order_matches_numba():
    """With two bad inputs, the twin raises the same error as Numba: the
    checks they share run in Numba's order, the power-of-2 rule last."""
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    for ff in (flood_fill, numba_flood_fill):
        with pytest.raises(ValueError, match="variant must be"):
            ff(img, sx, sy, threads_per_block=96, variant="turbo")


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


# ---------------------------------------------------------------------------
# v2 enqueue forms: enqueue="lane" (default, warp-aggregated by ptxas like
# Numba's _warp_enqueue_two_tier) and enqueue="program" (the first
# translation). Both must give outputs identical to Numba and to the CPU
# oracle, and the same deterministic counters.
# ---------------------------------------------------------------------------

import functools  # noqa: E402
import os.path  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

from .kernels import ENQ_MODES  # noqa: E402

ENQ_SCENES = ["square_64", "square_nonsquare_img", "corner_seeded_square",
              "disk_101", "serpentine_nonsquare", "random_supercritical",
              "single_pixel", "full_red_128"]


@pytest.mark.parametrize("enqueue", ENQ_MODES)
@pytest.mark.parametrize("name", ENQ_SCENES)
def test_enqueue_modes_match_reference_and_numba(name, enqueue):
    img, sx, sy = SCENES[name]()
    tri = assert_matches_reference(img, sx, sy, variant="spill",
                                   enqueue=enqueue)
    assert_same_as_numba(tri, numba_flood_fill(img, sx, sy, variant="spill"))


@pytest.mark.parametrize("enqueue", ENQ_MODES)
@pytest.mark.parametrize("tpb", [32, 128, 1024])
def test_enqueue_modes_block_size_invariance(tpb, enqueue):
    img, sx, sy = scenes.random_scene(256, 256, 0.65, rng_seed=3)
    tri = assert_matches_reference(img, sx, sy, threads_per_block=tpb,
                                   variant="spill", enqueue=enqueue)
    assert_same_as_numba(tri, numba_flood_fill(img, sx, sy,
                                               threads_per_block=tpb,
                                               variant="spill"))


@functools.lru_cache(maxsize=1)
def _spill_scene_references():
    """The 6.76M px scene that overflows the ring: CPU oracle and Numba v2,
    computed once for both enqueue forms."""
    img, sx, sy = scenes.overflow_scene()
    return (img, sx, sy, cpu_flood_fill(img, sx, sy),
            numba_flood_fill(img, sx, sy, variant="spill"))


@pytest.mark.parametrize("enqueue", ENQ_MODES)
def test_enqueue_modes_spill_scene(enqueue):
    """The spill tier in use: slabs straddle the ring window, the ring rear
    overshoots and is clamped, and 304,702 pixels spill. Both forms must
    match the oracle and every Numba counter (spilled, peak_spill_window,
    peak_occupancy, processed, cas_attempts, the level trace)."""
    img, sx, sy, (ref_visited, ref_depth, ref_levels, ref_filled), nb = \
        _spill_scene_references()
    tri = flood_fill(img, sx, sy, variant="spill", enqueue=enqueue)
    np.testing.assert_array_equal(tri.visited, ref_visited)
    np.testing.assert_array_equal(tri.depth, ref_depth)
    assert tri.levels == ref_levels and tri.filled == ref_filled
    assert tri.spilled > 0 and tri.peak_occupancy > tri.ring_capacity
    assert tri.processed == tri.filled
    assert_same_as_numba(tri, nb)


def test_enqueue_modes_agree_with_each_other():
    """Same counters from both forms on a scene with every level width."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    lane = flood_fill(img, sx, sy, variant="spill", enqueue="lane")
    prog = flood_fill(img, sx, sy, variant="spill", enqueue="program")
    assert_same_as_numba(lane, prog)


def test_enqueue_default_is_lane(monkeypatch):
    """Every default is the lane form, and a call that names no enqueue
    launches the spill kernel with ENQ="lane" (seen at the launch itself,
    so flipping any one default to "program" fails here)."""
    import inspect
    import sys
    from .kernels import single_block_bfs_spill_kernel

    ff_mod = sys.modules[flood_fill.__module__]
    for fn in (ff_mod.flood_fill, ff_mod.compiled_kernel, ff_mod._warmup):
        assert inspect.signature(fn).parameters["enqueue"].default == "lane"
    kernel_sig = inspect.signature(single_block_bfs_spill_kernel.fn)
    assert kernel_sig.parameters["ENQ"].default == "lane"
    assert ENQ_MODES[0] == "lane"

    launches = []
    real_launch = ff_mod._launch

    def spy(variant, enqueue, *args):
        launches.append((variant, enqueue))
        return real_launch(variant, enqueue, *args)

    monkeypatch.setattr(ff_mod, "_launch", spy)
    flood_fill(*scenes.square_scene(64, 64, 20, 20), variant="spill")
    assert launches and set(launches) == {("spill", "lane")}, launches


def test_bad_enqueue_raises():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="enqueue must be"):
        flood_fill(img, sx, sy, variant="spill", enqueue="warp")


def test_enqueue_program_on_ring_raises():
    """v1 has one form only (one ticket per winning lane, as in Numba)."""
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match='variant="spill" only'):
        flood_fill(img, sx, sy, variant="ring", enqueue="program")


def test_compiled_kernel_refuses_what_flood_fill_refuses():
    """compiled_kernel never caches a ring kernel under "program", nor any
    kernel under an unknown enqueue value."""
    from .flood_fill import _warmed_up, compiled_kernel
    with pytest.raises(ValueError, match='variant="spill" only'):
        compiled_kernel("ring", 256, "program")
    with pytest.raises(ValueError, match="enqueue must be"):
        compiled_kernel("spill", 256, "warp")
    assert ("ring", 256, "program") not in _warmed_up
    assert ("spill", 256, "warp") not in _warmed_up


def test_kernel_static_assert_rejects_unknown_enqueue():
    """Below the host checks, the kernel's own tl.static_assert on ENQ
    refuses to compile an unknown form."""
    from triton.compiler.errors import CompileTimeAssertionFailure
    from .flood_fill import _warmup
    with pytest.raises(CompileTimeAssertionFailure):
        _warmup("spill", 256, "warp")


def _full_red_center(side):
    """Full-bleed red square of the given side, seeded at the center. Near
    side 2049 the two-level ring occupancy (~4 * side) crosses the 8192-slot
    window, so a few more pixels of side move the scene from "fills the
    ring exactly" to a handful of spills to thousands."""
    img = np.zeros((side, side, 3), dtype=np.uint8)
    img[:, :] = scenes.RED
    return img, side // 2, side // 2


# side -> (spilled, peak_occupancy) of Numba v2 at every block size.
BOUNDARY_SCENES = {
    2049: (0, 8192),      # peak occupancy exactly RING_CAPACITY: no spill
    2050: (2, 8194),      # 2 tickets past the window
    2052: (14, 8202),     # slabs straddle the window edge
    2100: (2702, 8394),   # light spill, still cheap (4.4M px)
}


@functools.lru_cache(maxsize=1)
def _boundary_oracle(side):
    img, sx, sy = _full_red_center(side)
    return img, sx, sy, cpu_flood_fill(img, sx, sy)


@functools.lru_cache(maxsize=1)
def _boundary_numba(side, tpb):
    img, sx, sy, _ = _boundary_oracle(side)
    return numba_flood_fill(img, sx, sy, threads_per_block=tpb,
                            variant="spill")


# enqueue varies fastest, then tpb, then side: each one-slot cache is
# computed once per (side) and once per (side, tpb).
@pytest.mark.parametrize("enqueue", ENQ_MODES)
@pytest.mark.parametrize("tpb", [32, 256, 1024])
@pytest.mark.parametrize("side", list(BOUNDARY_SCENES))
def test_enqueue_modes_ring_spill_boundary(side, tpb, enqueue):
    """The ring/spill split at the window edge, at the smallest and largest
    block sizes too (1 warp, where every slab straddles inside one warp, and
    32 warps): both forms match the oracle and every Numba counter."""
    img, sx, sy, (ref_visited, ref_depth, ref_levels, ref_filled) = \
        _boundary_oracle(side)
    tri = flood_fill(img, sx, sy, threads_per_block=tpb, variant="spill",
                     enqueue=enqueue)
    np.testing.assert_array_equal(tri.visited, ref_visited)
    np.testing.assert_array_equal(tri.depth, ref_depth)
    assert tri.levels == ref_levels and tri.filled == ref_filled
    assert tri.processed == tri.filled
    assert (tri.spilled, tri.peak_occupancy) == BOUNDARY_SCENES[side]
    assert_same_as_numba(tri, _boundary_numba(side, tpb))


@pytest.mark.parametrize("tpb", [32, 1024])
def test_enqueue_lane_spill_repeat_runs_identical(tpb):
    """Per-lane tickets land in a schedule-dependent order, but every
    deterministic output of a spilling run repeats exactly."""
    img, sx, sy = _full_red_center(2052)
    a = flood_fill(img, sx, sy, threads_per_block=tpb, variant="spill")
    b = flood_fill(img, sx, sy, threads_per_block=tpb, variant="spill")
    assert a.spilled == 14
    assert_same_as_numba(a, b)


def test_enqueue_program_no_recompile_inside_timing():
    """The first translation, like the default, compiles once per (tpb,
    ENQ): other scene sizes and seeds reuse it."""
    from flood_fill_cuda.triton_twins.runtime import bridge  # noqa: F401
    from triton.runtime.driver import driver
    from .kernels import single_block_bfs_spill_kernel

    cache = single_block_bfs_spill_kernel.device_caches[
        driver.active.get_current_device()][0]
    flood_fill(*scenes.square_scene(64, 64, 32, 32), variant="spill",
               enqueue="program")
    before = len(cache)
    for build in (lambda: scenes.square_scene(97, 33, 51, 17),
                  lambda: scenes.serpentine_scene(48, 80),
                  lambda: scenes.full_red_scene(129, 1)):
        flood_fill(*build(), variant="spill", enqueue="program")
    assert len(cache) == before


def _nvdisasm():
    import triton
    bundled = os.path.join(os.path.dirname(triton.__file__), "backends",
                           "nvidia", "bin", "nvdisasm")
    for path in (bundled, shutil.which("nvdisasm"),
                 "/usr/local/cuda/bin/nvdisasm"):
        if path and os.path.exists(path):
            return path
    return None


def _sass(compiled):
    tool = _nvdisasm()
    if tool is None:
        pytest.skip("no nvdisasm to read the SASS")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "k.cubin")
        with open(path, "wb") as f:
            f.write(compiled.asm["cubin"])
        out = subprocess.run([tool, "-c", path], capture_output=True,
                             text=True, check=True).stdout
    return [ln.split("*/", 1)[1].split(";")[0].strip()
            for ln in out.splitlines() if re.match(r"\s+/\*[0-9a-f]{4}\*/", ln)]


# (variant, enqueue) -> (CTA barriers = Numba's syncthreads count,
#                        enqueue atomics = directions x tiers)
CODEGEN = {("ring", "lane"): (3, 4), ("spill", "lane"): (4, 8)}


@pytest.mark.parametrize("variant,enqueue", list(CODEGEN))
def test_lane_enqueue_is_warp_aggregated_like_numba(variant, enqueue):
    """The default kernels carry exactly Numba's barriers (1 + 2 per level
    for v1, 1 + 3 for v2) and every enqueue atomic is warp-aggregated by
    ptxas: VOTEU.ANY (active mask), POPC (count), one predicated leader
    ATOMG.E.ADD, SHFL.IDX (base broadcast), the same idiom Numba's
    activemask/popc/ffs/shfl_sync enqueue compiles to."""
    from .flood_fill import compiled_kernel
    ck = compiled_kernel(variant, 256, enqueue)
    bars, atomics = CODEGEN[(variant, enqueue)]
    assert len(re.findall(r"\bbar\.sync\b", ck.asm["ptx"])) == bars
    sass = _sass(ck)
    assert sum(i.startswith("BAR.SYNC") for i in sass) == bars
    sites = [k for k, i in enumerate(sass)
             if re.match(r"@!?U?P\d ATOMG\.E\.ADD", i)]
    unpredicated = [i for i in sass if i.startswith("ATOMG.E.ADD")]
    assert len(sites) == atomics and not unpredicated, (sites, unpredicated)
    for k in sites:
        before, after = sass[max(0, k - 8):k], sass[k + 1:k + 9]
        assert any(i.startswith("VOTEU.ANY") for i in before), sass[k - 8:k + 9]
        assert any(i.startswith("POPC") for i in before), sass[k - 8:k + 9]
        assert any(i.startswith("SHFL.IDX") for i in after), sass[k - 8:k + 9]


def test_program_enqueue_adds_cta_barriers():
    """The first translation's tl.cumsum / tl.sum / scalar atomic cost CTA
    barriers per direction that Numba never had."""
    from .flood_fill import compiled_kernel
    ptx = compiled_kernel("spill", 256, "program").asm["ptx"]
    assert len(re.findall(r"\bbar\.sync\b", ptx)) > 4 * 4


@pytest.mark.parametrize("quick", [True, False])
def test_compare_marks_repeated_cells(quick):
    """compare.py tags exactly the like-for-like rows that repeat an earlier
    experiment's cell (per_lane enqueue rows, the 256-thread sweep row) with
    duplicate_of, never a first_translation row; skipping the tagged rows
    leaves every comparable cell measured once. Builds the cases only."""
    from . import compare as cmp
    cases = cmp.build_cases(quick=quick)
    enq_scenes = cmp.QUICK_ENQ_SCENES if quick else cmp.ENQ_SCENES
    sweep_scene, sweep_tpbs = (cmp.QUICK_SWEEP if quick
                               else (cmp.SWEEP_SCENE, cmp.TPB_SWEEP))
    expected = {("enqueue", s, "spill", cmp.TPB): "scenes"
                for s in enq_scenes}
    if cmp.TPB in sweep_tpbs:
        expected[("tpb_sweep", sweep_scene, "ring", cmp.TPB)] = "scenes"
    tagged = {(c.experiment, c.scene, c.config["variant"],
               c.config["threads_per_block"]): c.extra["duplicate_of"]
              for c in cases if "duplicate_of" in c.extra}
    assert tagged == expected
    assert not any(c.extra.get("first_translation") and
                   "duplicate_of" in c.extra for c in cases)
    cells = [cmp.repeated_cell(c) for c in cases
             if c.comparable and "duplicate_of" not in c.extra]
    assert len(cells) == len(set(cells))
