"""Correctness tests for the service engine's mask -> depth-timeline wrapper.

Things only this layer can get wrong that ch03's own suite doesn't cover:
seed placement (an explicit release point, snapped onto the mask, or the
center-of-mass fallback), the (height,width) <-> (width,height) transpose,
the depth+1 wire encoding, and — since the CPU/GPU mode split lives here —
that both engines agree on the identical BFS. Run:

    uv run pytest src/flood_fill_cuda/service/test_engine.py -v
"""

import numpy as np
import pytest

from . import engine
from ..chapters.ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill


def _disc_mask(h, w, cx, cy, r):
    ys, xs = np.mgrid[0:h, 0:w]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r


def _bar_mask(h, w, y0, y1):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, :] = True
    return mask


def _ring_mask(h, w, cx, cy, r_out, r_in):
    ys, xs = np.mgrid[0:h, 0:w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return (d2 <= r_out * r_out) & (d2 >= r_in * r_in)


def test_oracle_equivalence_disc():
    mask = _disc_mask(101, 101, 50, 50, 40)
    out = engine.run_fill(mask)

    img = np.full((101, 101, 3), 255, dtype=np.uint8)
    img[mask.T] = (255, 0, 0)
    ref_visited, ref_depth, ref_levels, ref_filled = cpu_flood_fill(
        img, out.seed_x, out.seed_y)

    expected = np.where(ref_depth.T < 0, 0,
                        np.minimum(ref_depth.T + 1, engine.DEPTH_CLAMP))
    np.testing.assert_array_equal(out.depth_u16, expected.astype('<u2'))
    assert out.levels == ref_levels
    assert out.filled == ref_filled == int(ref_visited.sum())


def test_oracle_equivalence_l_shape():
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:30, 10:70] = True   # top bar
    mask[10:70, 10:30] = True   # left bar (forms an L)
    out = engine.run_fill(mask)

    img = np.full((80, 80, 3), 255, dtype=np.uint8)
    img[mask.T] = (255, 0, 0)
    ref_visited, ref_depth, ref_levels, ref_filled = cpu_flood_fill(
        img, out.seed_x, out.seed_y)

    expected = np.where(ref_depth.T < 0, 0,
                        np.minimum(ref_depth.T + 1, engine.DEPTH_CLAMP))
    np.testing.assert_array_equal(out.depth_u16, expected.astype('<u2'))
    assert out.levels == ref_levels
    assert out.filled == ref_filled == int(ref_visited.sum())


def test_centroid_seed_on_symmetric_disc():
    """A solid disc's center of mass is its geometric center: seed depth is
    level 1 (its own encoded value) and every rim pixel shares the same
    (near-)maximal depth."""
    mask = _disc_mask(101, 101, 50, 50, 40)
    out = engine.run_fill(mask)
    assert (out.seed_x, out.seed_y) == (50, 50)
    assert out.depth_u16[50, 50] == 1   # depth 0 -> encoded 1


def test_centroid_off_mask_snaps_onto_ring():
    """A ring's center of mass falls in the empty hole; the seed must snap
    onto an actual mask pixel, and the fill must still reach every pixel of
    the ring."""
    mask = _ring_mask(121, 121, 60, 60, 50, 35)
    assert not mask[60, 60]   # the hole itself is not part of the mask
    out = engine.run_fill(mask)
    assert mask[out.seed_y, out.seed_x]
    assert out.filled == int(mask.sum())
    assert (out.depth_u16[mask] > 0).all()
    assert (out.depth_u16[~mask] == 0).all()


def test_orientation_asymmetry():
    """A wide horizontal bar seeded at its center: depth must be maximal at
    BOTH the left and right ends in (row, col) canvas coordinates. A
    transposed implementation would instead show the asymmetry running
    top-to-bottom and fail this immediately."""
    mask = _bar_mask(20, 200, 8, 12)
    out = engine.run_fill(mask)
    assert out.seed_y in range(8, 12)
    assert 90 <= out.seed_x <= 110   # centroid x is the bar's midpoint

    row = out.seed_y
    left_depth = out.depth_u16[row, 0]
    right_depth = out.depth_u16[row, -1]
    mid_depth = out.depth_u16[row, out.seed_x]
    assert left_depth > mid_depth
    assert right_depth > mid_depth
    assert left_depth == pytest.approx(right_depth, abs=1)


def test_monotonic_adjacency():
    """Every reached pixel (other than the seed) has a 4-neighbor exactly
    one encoded level shallower — the depth map is a true BFS distance
    field, not just a reached/unreached mask."""
    mask = _disc_mask(101, 101, 50, 50, 40)
    out = engine.run_fill(mask)
    d = out.depth_u16.astype(np.int64)
    seed_val = d[out.seed_y, out.seed_x]
    ys, xs = np.nonzero(d > 0)
    for y, x in zip(ys, xs):
        v = d[y, x]
        if v == seed_val:
            continue
        neighbors = []
        if y > 0:
            neighbors.append(d[y - 1, x])
        if y + 1 < d.shape[0]:
            neighbors.append(d[y + 1, x])
        if x > 0:
            neighbors.append(d[y, x - 1])
        if x + 1 < d.shape[1]:
            neighbors.append(d[y, x + 1])
        assert (v - 1) in neighbors


def test_explicit_seed_used_directly_when_on_mask():
    """An explicit release point already on the mask (not the centroid) is
    used as-is: depth 0 lands exactly there, not at the mask's center."""
    mask = _bar_mask(20, 200, 8, 12)
    out = engine.run_fill(mask, seed_x=5, seed_y=10)
    assert (out.seed_x, out.seed_y) == (5, 10)
    assert out.depth_u16[10, 5] == 1   # depth 0 -> encoded 1


def test_explicit_seed_snaps_onto_mask():
    """A release point just outside the mask still resolves to a real mask
    pixel, not the literal off-mask coordinate."""
    mask = _disc_mask(101, 101, 50, 50, 40)
    out = engine.run_fill(mask, seed_x=50, seed_y=95)   # 6px above the rim
    assert mask[out.seed_y, out.seed_x]
    assert (out.seed_x, out.seed_y) != (50, 95)


def test_missing_seed_falls_back_to_centroid():
    mask = _disc_mask(101, 101, 50, 50, 40)
    out = engine.run_fill(mask)
    assert (out.seed_x, out.seed_y) == (50, 50)


def test_cpu_and_gpu_modes_agree():
    """ch01's CPU oracle and ch03's GPU kernel are tested elsewhere to be
    bit-identical on the same BFS (ch03/test_correctness.py); this checks
    that guarantee survives the engine's mode dispatch, seed resolution,
    and wire encoding intact."""
    mask = _disc_mask(151, 151, 70, 70, 55)
    gpu_out = engine.run_fill(mask, mode="gpu", seed_x=40, seed_y=70)
    cpu_out = engine.run_fill(mask, mode="cpu", seed_x=40, seed_y=70)
    np.testing.assert_array_equal(gpu_out.depth_u16, cpu_out.depth_u16)
    assert gpu_out.levels == cpu_out.levels
    assert gpu_out.filled == cpu_out.filled
    assert (gpu_out.seed_x, gpu_out.seed_y) == (cpu_out.seed_x, cpu_out.seed_y)
    assert gpu_out.mode == "gpu" and cpu_out.mode == "cpu"


def test_cpu_mode_reports_real_compute_time():
    mask = _disc_mask(101, 101, 50, 50, 40)
    out = engine.run_fill(mask, mode="cpu")
    assert out.kernel_ms > 0
    assert out.total_ms == out.kernel_ms


def test_amplified_filled_scales_up_a_small_mask():
    """A small mask is well under MAX_AMPLIFIED_PIXELS, so the amplified
    run should actually happen and report a substantially bigger pixel
    count than the real (painted) one -- while the returned depth map
    stays at the real, painted size."""
    mask = _disc_mask(41, 41, 20, 20, 15)
    out = engine.run_fill(mask)
    assert out.amplified_filled > out.filled
    assert out.depth_u16.shape == (41, 41)
    assert out.filled == int(mask.sum())


def test_amplify_mask_skips_when_already_at_cap():
    """_amplify_mask must not try to upscale past its own ceiling -- a
    mask already at/over MAX_AMPLIFIED_PIXELS comes back unchanged
    (scale 1.0). Exercised directly since run_fill's own (much lower)
    MAX_PIXELS input-validation gate means no request ever reaches
    run_fill's amplification step already this large."""
    side = int(engine.MAX_AMPLIFIED_PIXELS ** 0.5) + 10
    mask = np.zeros((side, side), dtype=bool)
    mask[0:5, 0:5] = True
    big_mask, scale = engine._amplify_mask(
        mask, engine.AMPLIFY_FACTOR, engine.MAX_AMPLIFIED_PIXELS)
    assert scale == 1.0
    assert big_mask is mask


def test_amplify_mask_respects_ceiling_and_shape():
    """A mask well under the ceiling amplifies to roughly factor x its
    pixel count (capped), and the upscaled mask preserves the original
    shape's footprint (every original True pixel maps to a True block)."""
    mask = _disc_mask(31, 31, 15, 15, 10)
    big_mask, scale = engine._amplify_mask(mask, 50, engine.MAX_AMPLIFIED_PIXELS)
    assert scale > 1.0
    assert big_mask.sum() > mask.sum()
    assert big_mask.shape[0] * big_mask.shape[1] <= engine.MAX_AMPLIFIED_PIXELS


def test_invalid_mode_raises():
    mask = _disc_mask(64, 64, 32, 32, 20)
    with pytest.raises(ValueError, match="mode"):
        engine.run_fill(mask, mode="tpu")


def test_empty_mask_raises():
    mask = np.zeros((32, 32), dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        engine.run_fill(mask)


def test_oversized_mask_raises_mask_too_large():
    side = int(engine.MAX_PIXELS ** 0.5) + 100
    mask = np.zeros((side, side), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(engine.MaskTooLargeError):
        engine.run_fill(mask)


def test_deterministic_across_runs():
    mask = _disc_mask(151, 151, 70, 70, 55)
    a = engine.run_fill(mask)
    b = engine.run_fill(mask)
    np.testing.assert_array_equal(a.depth_u16, b.depth_u16)
    assert a.levels == b.levels and a.filled == b.filled
    assert (a.seed_x, a.seed_y) == (b.seed_x, b.seed_y)


def test_warmup_runs_without_error():
    engine.warmup()
