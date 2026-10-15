"""Correctness tests: tile-BFS flood fill vs the CPU references.

There is no depth map in this chapter, so the contract is the fill set
and the painted image: every reached pixel (0, 0, 255), every other
pixel untouched, the reached set equal to the @njit oracle's visited
set (ch01's for 4-connectivity, shared/'s for 8). The bit helpers the
closure is built from are tested on their own against plain-Python
references first, so a failure further down can be localised.

Run:

    uv run pytest src/flood_fill_cuda/chapters/ch07_gpu_1blob_tiles/test_correctness.py -v
"""

import os

# Before numba is imported: ch03's host code reads device attributes that
# only the NVIDIA binding provides (and ctypes + CUDA 12.9 segfaults).
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
import pytest
from numba import cuda

from . import kernels
from .flood_fill import flood_fill, max_blocks, TileFill
from ..ch03_gpu_1blob_nblock.cpu_oracle import cpu_flood_fill, cpu_flood_fill_8
from ..ch03_gpu_1blob_nblock import scenes
from ..ch03_gpu_1blob_nblock.flood_fill import flood_fill as ch03_flood_fill

BLUE = np.array([0, 0, 255], dtype=np.uint8)
RED = scenes.RED


def expected_image(img, seed_x, seed_y, connectivity):
    ref = cpu_flood_fill_8 if connectivity == 8 else cpu_flood_fill
    visited, _, _, filled = ref(img, seed_x, seed_y)
    out = img.copy()
    out[visited == 1] = BLUE
    return out, visited, filled


def assert_matches_reference(img, seed_x, seed_y, connectivity, **kw):
    exp_img, exp_visited, exp_filled = expected_image(img, seed_x, seed_y,
                                                      connectivity)
    result = flood_fill(img, seed_x, seed_y, connectivity=connectivity, **kw)
    np.testing.assert_array_equal(result.visited, exp_visited)
    np.testing.assert_array_equal(result.img, exp_img)
    assert result.filled == exp_filled
    return result


# --------------------------------------------------------- bit helpers

@cuda.jit
def _bits_probe(f_in, m_in, out_row, out_col, out_diag):
    """One warp per 32 consecutive words: lane i holds word warp*32 + i."""
    i = cuda.grid(1)
    lane = cuda.laneid
    f = f_in[i]
    m = m_in[i]
    out_row[i] = kernels._row_fill(f, m)
    out_col[i] = kernels._col_fill(f, m, lane)
    out_diag[i] = kernels._diag_step(f, m, lane)


def _runs(word):
    """Maximal runs of 1-bits of a 32-bit int as (lo, hi) inclusive."""
    runs = []
    b = 0
    while b < 32:
        if (word >> b) & 1:
            lo = b
            while b + 1 < 32 and (word >> (b + 1)) & 1:
                b += 1
            runs.append((lo, b))
        b += 1
    return runs


def ref_row_fill(f, m):
    out = 0
    for lo, hi in _runs(m):
        span = ((1 << (hi - lo + 1)) - 1) << lo
        if f & span:
            out |= span
    return out | f


def ref_col_fill(fs, ms):
    """fs, ms: 32 ints (one per lane). Per bit column, every vertical run
    of m that holds an f bit."""
    out = list(fs)
    for b in range(32):
        col_m = sum(((ms[i] >> b) & 1) << i for i in range(32))
        col_f = sum(((fs[i] >> b) & 1) << i for i in range(32))
        for lo, hi in _runs(col_m):
            span = ((1 << (hi - lo + 1)) - 1) << lo
            if col_f & span:
                for i in range(lo, hi + 1):
                    out[i] |= 1 << b
    return out


def ref_diag(fs, ms):
    mask32 = 0xFFFFFFFF

    def dil(v):
        return (v | (v << 1) | (v >> 1)) & mask32

    out = []
    for i in range(32):
        nb = 0
        if i > 0:
            nb |= dil(fs[i - 1])
        if i < 31:
            nb |= dil(fs[i + 1])
        out.append(fs[i] | (ms[i] & nb))
    return out


def _random_words(rng, n, density):
    """n uint32 words with roughly `density` of their bits set."""
    bits = rng.random((n, 32)) < density
    return (bits * (1 << np.arange(32, dtype=np.uint64))).sum(axis=1).astype(np.uint32)


@pytest.mark.parametrize("density", [0.3, 0.6, 0.85, 1.0])
def test_bit_helpers_match_python_reference(density):
    rng = np.random.default_rng(int(density * 100))
    n_warps = 48
    n = 32 * n_warps
    m = _random_words(rng, n, density)
    f = m & _random_words(rng, n, 0.05)          # f is always a subset of m
    d_row = cuda.device_array(n, np.uint32)
    d_col = cuda.device_array(n, np.uint32)
    d_diag = cuda.device_array(n, np.uint32)
    _bits_probe[n_warps, 32](cuda.to_device(f), cuda.to_device(m),
                             d_row, d_col, d_diag)
    row = d_row.copy_to_host()
    col = d_col.copy_to_host()
    diag = d_diag.copy_to_host()
    for i in range(n):
        assert int(row[i]) == ref_row_fill(int(f[i]), int(m[i])), i
    for w in range(n_warps):
        sl = slice(32 * w, 32 * w + 32)
        fs = [int(v) for v in f[sl]]
        ms = [int(v) for v in m[sl]]
        assert [int(v) for v in col[sl]] == ref_col_fill(fs, ms), w
        assert [int(v) for v in diag[sl]] == ref_diag(fs, ms), w
    # every helper keeps f inside m and only ever adds bits
    for out in (row, col, diag):
        assert np.all((out & ~m) == 0)
        assert np.all((out & f) == f)


def test_pack_zeroes_bits_past_height():
    """ch06's pack_kernel, reused here: bits past `height` in the last
    word of a row must be zero, or a closure could walk into them."""
    for height in (1, 31, 33, 70, 95):
        img = np.zeros((37, height, 3), dtype=np.uint8)
        img[:, :] = RED
        eng = TileFill(37, height, connectivity=4)
        eng.pack(cuda.to_device(img))
        mask = eng.mask.copy_to_host()
        wpr = (height + 31) // 32
        assert mask.shape == (37, wpr)
        tail = height - 32 * (wpr - 1)
        assert np.all(mask[:, :wpr - 1] == 0xFFFFFFFF)
        assert np.all(mask[:, wpr - 1] == np.uint32((1 << tail) - 1))


# ------------------------------------------- scenes vs the CPU oracles

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


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("name", SCENES.keys())
def test_ch03_scenes(name, connectivity):
    img, sx, sy = SCENES[name]()
    assert_matches_reference(img, sx, sy, connectivity)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("shape", [(33, 65), (100, 77), (31, 31), (65, 33),
                                   (1, 50), (50, 1), (97, 161)])
def test_sizes_not_multiples_of_32(shape, connectivity):
    """Partial tiles in x (lanes past the image) and in y (pad bits)."""
    w, h = shape
    full = np.zeros((w, h, 3), dtype=np.uint8)
    full[:, :] = RED
    assert_matches_reference(full, w // 2, h // 2, connectivity)
    if w > 2 and h > 2:
        img, sx, sy = scenes.random_scene(w, h, 0.62, rng_seed=w * h)
        assert_matches_reference(img, sx, sy, connectivity)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_seeds_at_corners_and_edges(connectivity):
    w, h = 70, 45
    full = np.zeros((w, h, 3), dtype=np.uint8)
    full[:, :] = RED
    noise, _, _ = scenes.random_scene(w, h, 0.6, rng_seed=21)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1)]
    for sx, sy in seeds:
        assert_matches_reference(full, sx, sy, connectivity)
        if (noise[sx, sy] == RED).all():
            assert_matches_reference(noise, sx, sy, connectivity)


def _staircase(n, anti=False):
    img = np.full((n, n, 3), 255, dtype=np.uint8)
    for i in range(n):
        img[i, n - 1 - i if anti else i] = RED
    return img


@pytest.mark.parametrize("anti", [False, True])
def test_diagonal_staircase_crosses_tile_corners(anti):
    """A 1-px diagonal steps from (31, 31) to (32, 32): the only link is a
    corner-to-corner export into the diagonal tile. 8-connected it is one
    blob; 4-connected every pixel is its own."""
    n = 100
    img = _staircase(n, anti)
    sy = n - 1 if anti else 0
    r8 = assert_matches_reference(img, 0, sy, 8)
    assert r8.filled == n
    r4 = assert_matches_reference(img, 0, sy, 4)
    assert r4.filled == 1


@pytest.mark.parametrize("connectivity", [4, 8])
def test_checkerboard(connectivity):
    n = 70
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    img = np.full((n, n, 3), 255, dtype=np.uint8)
    img[(xs + ys) % 2 == 0] = RED
    r = assert_matches_reference(img, 10, 10, connectivity)
    assert r.filled == (n * n // 2 if connectivity == 8 else 1)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_serpentine_transposed(connectivity):
    """ch03's serpentine runs along y (inside words); its transpose runs
    along x (across lanes and tile rows)."""
    img, sx, sy = scenes.serpentine_scene(96, 130)
    img_t = np.ascontiguousarray(img.transpose(1, 0, 2))
    assert_matches_reference(img_t, sy, sx, connectivity)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("density", [0.3, 0.45, 0.55, 0.6, 0.75])
def test_random_noise_many_seeds(density, connectivity):
    img, _, _ = scenes.random_scene(150, 170, density, rng_seed=int(density * 1000))
    red = np.argwhere((img == RED).all(axis=2))
    rng = np.random.default_rng(int(density * 1000) + connectivity)
    for k in rng.choice(len(red), size=8, replace=False):
        sx, sy = (int(v) for v in red[k])
        assert_matches_reference(img, sx, sy, connectivity)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_single_pixel(connectivity):
    img, sx, sy = scenes.single_pixel_scene(40, 40)
    r = assert_matches_reference(img, sx, sy, connectivity)
    assert r.filled == 1 and r.rounds == 1 and r.tile_visits == 1


@pytest.mark.parametrize("connectivity", [4, 8])
def test_non_red_seed_raises_like_ch03(connectivity):
    img, _, _ = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="not red"):
        flood_fill(img, 0, 0, connectivity=connectivity)
    with pytest.raises(ValueError, match="not red"):
        ch03_flood_fill(img, 0, 0, connectivity=connectivity)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (20, 12), (31, 32)])
def test_image_smaller_than_one_tile(shape, connectivity):
    w, h = shape
    full = np.zeros((w, h, 3), dtype=np.uint8)
    full[:, :] = RED
    assert_matches_reference(full, w - 1, 0, connectivity)
    if w * h > 4:
        img, sx, sy = scenes.random_scene(w, h, 0.6, rng_seed=w + 100 * h)
        assert_matches_reference(img, sx, sy, connectivity)


# ------------------------------------------------------ the launch shape

@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("blocks", [1, 3, None])
@pytest.mark.parametrize("tpb", [32, 64, 256])
def test_grid_shape_invariance(tpb, blocks, connectivity):
    """The fill must not depend on the grid - including one block (every
    round runs on 1-8 warps) and an odd block count."""
    img, sx, sy = scenes.random_scene(256, 230, 0.62, rng_seed=3)
    assert_matches_reference(img, sx, sy, connectivity,
                             threads_per_block=tpb, blocks=blocks)


@pytest.mark.parametrize("max_registers", [None, 64])
def test_register_cap_invariance(max_registers):
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    assert_matches_reference(img, sx, sy, 8, max_registers=max_registers)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_bare_twin_matches_reference(connectivity):
    img, sx, sy = scenes.random_scene(200, 200, 0.62, rng_seed=9)
    r = assert_matches_reference(img, sx, sy, connectivity, bare=True)
    assert r.bare and r.rounds >= 1
    assert r.tile_visits == 0 and r.or_atomics == 0  # counters stripped


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("name", ["disk_101", "serpentine_128",
                                  "random_supercritical", "full_red_128"])
def test_painted_image_equals_ch03_gpu(name, connectivity):
    """The headline contract, checked against ch03's kernel itself."""
    img, sx, sy = SCENES[name]()
    ours = flood_fill(img, sx, sy, connectivity=connectivity)
    theirs = ch03_flood_fill(img, sx, sy, connectivity=connectivity)
    np.testing.assert_array_equal(ours.img, theirs.img)
    assert ours.filled == theirs.filled


def test_input_not_modified():
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    before = img.copy()
    flood_fill(img, sx, sy, connectivity=8)
    np.testing.assert_array_equal(img, before)


def test_unreached_pixels_untouched():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[10:30, 10:30] = RED   # component A (seeded)
    img[60:80, 60:80] = RED   # component B, disconnected
    r = assert_matches_reference(img, 20, 20, 8)
    np.testing.assert_array_equal(r.img[60:80, 60:80], img[60:80, 60:80])


def test_counters_are_consistent():
    """Full-red 128x128 from a corner: 4x4 tiles, so the tile frontier
    needs at least 4 rounds (Chebyshev, 8-conn) or 7 (Manhattan, 4-conn);
    every tile is visited at least once; every visit runs >= 1 closure
    iteration."""
    img, sx, sy = scenes.full_red_scene(128, 128)
    for conn, min_rounds in ((8, 4), (4, 7)):
        r = flood_fill(img, sx, sy, connectivity=conn)
        assert r.filled == 128 * 128
        assert r.n_tiles == 16
        assert r.rounds >= min_rounds
        assert r.tile_visits >= 16
        assert r.closure_iters >= r.tile_visits
        assert r.or_atomics > 0 and r.stamp_atomics >= 15
        assert set(r.phase_ms) == {"pack", "init", "rounds", "paint"}


@pytest.mark.parametrize("kwargs, exc", [
    ({"connectivity": 6}, ValueError),
    ({"threads_per_block": 48}, ValueError),
    ({"threads_per_block": 2048}, ValueError),
    ({"blocks": 0}, ValueError),
    ({"blocks": 10 ** 6}, RuntimeError),
])
def test_rejects_bad_arguments(kwargs, exc):
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(exc):
        flood_fill(img, sx, sy, **kwargs)


def test_cooperative_capacity_is_positive():
    for conn in (4, 8):
        assert max_blocks(256, conn) >= 1
