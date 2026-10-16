"""Correctness tests: tile-BFS flood fill vs the CPU references.

There is no depth map in this chapter, so the contract is the fill set
and the painted image: every reached pixel (0, 0, 255), every other
pixel untouched, the reached set equal to the @njit oracle's visited
set (ch01's for 4-connectivity, shared/'s for 8). The bit helpers the
closure is built from are tested on their own against plain-Python
references first, so a failure further down can be localised.

Every test runs once per switch mode (MODES: the rounds kernel's
one_sync / fused_pack / skip_full / fused_paint switches, set through
flood_fill.DEFAULT_SWITCHES), so the whole suite checks each path; the
tests at the end target the switches themselves.

Run:

    uv run pytest src/flood_fill_cuda/chapters/ch07_gpu_1blob_tiles/test_correctness.py -v
"""

import os
import subprocess
import sys

# Before numba is imported: ch03's host code reads device attributes that
# only the NVIDIA binding provides (and ctypes + CUDA 12.9 segfaults).
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
import pytest
from numba import cuda

from numba import uint64

from . import kernels
from . import flood_fill as ff
from .flood_fill import flood_fill, max_blocks, plain_shape, TileFill
from ..ch03_gpu_1blob_nblock.cpu_oracle import cpu_flood_fill, cpu_flood_fill_8
from ..ch03_gpu_1blob_nblock import scenes
from ..ch03_gpu_1blob_nblock.flood_fill import flood_fill as ch03_flood_fill
from .benchmarks import worst_cases

BLUE = np.array([0, 0, 255], dtype=np.uint8)
RED = scenes.RED

# switch modes the whole module runs under (unset switches are off)
MODES = {
    "old": {},
    "one_sync": {"one_sync": True},
    "fused_pack": {"fused_pack": True},
    "skip_full": {"skip_full": True},
    "one_sync+fused_pack": {"one_sync": True, "fused_pack": True},
    "one_sync+fused_pack+skip_full": {"one_sync": True, "fused_pack": True,
                                      "skip_full": True},
    "all+fused_paint": {"one_sync": True, "fused_pack": True,
                        "skip_full": True, "fused_paint": True},
}


@pytest.fixture(autouse=True, scope="module", params=list(MODES))
def switch_mode(request):
    """Run every test of the module with these switches as the defaults
    of flood_fill() and TileFill (module scope: grouped by mode)."""
    saved = dict(ff.DEFAULT_SWITCHES)
    ff.DEFAULT_SWITCHES.update({k: False for k in ff.SWITCHES})
    ff.DEFAULT_SWITCHES.update(MODES[request.param])
    yield request.param
    ff.DEFAULT_SWITCHES.clear()
    ff.DEFAULT_SWITCHES.update(saved)


def only_in(mode, *wanted):
    """Skip a mode-independent test in all but the listed modes."""
    if mode not in wanted:
        pytest.skip(f"switch-independent: runs in mode {wanted[0]} only")


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


@pytest.mark.parametrize("width, wpr", [(1, 1), (64, 1), (1 << 20, 1),
                                        (5000, 2), (5000, 7), (5000, 8),
                                        (5000, 9), (8000, 250), (70, 3000)])
def test_plain_shape_covers_every_word(width, wpr):
    """pack/paint launch: whole warps, every word of a row covered along
    x, gridDim.y legal and no larger than the image."""
    (gx, gy), tpb = plain_shape(width, wpr)
    assert tpb % 32 == 0 and 32 <= tpb <= 256
    assert gx * tpb >= 32 * wpr > (gx - 1) * tpb
    assert 1 <= gy <= min(width, 65535)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("shape", [(20000, 1), (9000, 32), (5000, 64),
                                   (3000, 200), (1500, 300)])
def test_thin_images(shape, connectivity):
    """Thin images get a different pack/paint grid (narrow blocks, many
    rows in flight). Full red from the middle, then noise."""
    w, h = shape
    full = np.zeros((w, h, 3), dtype=np.uint8)
    full[:, :] = RED
    assert_matches_reference(full, w // 2, h // 2, connectivity)
    img, sx, sy = scenes.random_scene(w, h, 0.62, rng_seed=w + h)
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


WORST_CASES = {
    "zz_serpentine": lambda: worst_cases.zz_serpentine_scene(256, 256),
    "zz_lattice": lambda: worst_cases.zz_lattice_scene(512, 512),
    "maze": lambda: worst_cases.maze_scene(cells=48),
    "thin_h32_blob": lambda: worst_cases.thin_h32_blob_scene(1 << 13),
}


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("name", WORST_CASES.keys())
def test_worst_case_scenes(name, connectivity):
    """The benchmark's worst cases, small. On the zigzags every pixel step
    crosses a tile border, so at 8-conn ch07 needs about one round per
    level; 4-conn fills only the seed there."""
    img, sx, sy = WORST_CASES[name]()
    r = assert_matches_reference(img, sx, sy, connectivity)
    red = int((img == RED).all(axis=2).sum())
    if connectivity == 8:
        assert r.filled == red        # one blob: ch06's check relies on it
    if name == "zz_serpentine":
        if connectivity == 8:
            levels = cpu_flood_fill_8(img, sx, sy)[2]
            assert r.rounds >= 0.9 * levels
        else:
            assert r.filled == 1


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


NARROW_SEED_DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32,
                      np.int64]


@pytest.mark.parametrize("dtype", NARROW_SEED_DTYPES)
def test_seed_args_ignore_the_seed_dtype(dtype):
    """Under NumPy 2 a narrow NumPy scalar keeps its dtype through
    `tx << 16`, `1 << (y % 32)` and `x * wpr + ty`, so an int16 seed
    once named tile (0, ty) and a uint8 seed set no bit at all. The queue
    entry, seed word, seed bit and stamp index must not depend on the
    integer type the caller happened to hold."""
    eng = TileFill(100, 100, connectivity=8, _skip_warmup=True)
    want = eng.seed_args(40, 40)
    assert eng.seed_args(dtype(40), dtype(40)) == want
    assert want[3] == (1 << kernels.TY_BITS) | 1      # tile (1, 1)
    if np.iinfo(dtype).max >= 30000:
        # x * wpr + ty past 2**15: the seed word / stamp index wrap too
        wide = TileFill(30001, 64, connectivity=8, _skip_warmup=True)
        assert wide.seed_args(dtype(30000), dtype(40)) == \
            wide.seed_args(30000, 40)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("dtype", NARROW_SEED_DTYPES)
def test_numpy_integer_seeds_fill_the_blob(dtype, connectivity):
    """Regression: an np.int16 seed at (40, 40) filled 1 px where ch03
    and the oracle fill the 3,600-px square."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[10:70, 10:70] = RED
    exp_img, exp_visited, exp_filled = expected_image(img, 40, 40,
                                                      connectivity)
    r = flood_fill(img, dtype(40), dtype(40), connectivity=connectivity)
    assert r.filled == exp_filled == 3600
    np.testing.assert_array_equal(r.visited, exp_visited)
    np.testing.assert_array_equal(r.img, exp_img)


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


def _layout_scene(name):
    """(C-ordered image, seed) for the memory-layout tests."""
    if name == "stripes":
        img = np.zeros((99, 64, 3), dtype=np.uint8)   # rows 0, 3, 6, ... red
        img[0::3, :] = RED
        return img, 0, 0
    return scenes.disk_scene(150, 130, 50)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("name", ["disk", "stripes", "chw_T"])
def test_fortran_ordered_input(name, connectivity):
    """Regression: flood_fill() accepts any (W, H, 3) uint8 array, and an
    F-contiguous one kept its F order on the device. `chw.T` of a
    channel-first (3, H, W) image is exactly such an array. The fused
    pack read the device bytes as C-ordered (x, y, c): it filled 0 of the
    disk's 7,845 px, and on the stripes it filled 2,112 px instead of 64
    and painted 2,112 black pixels."""
    img, sx, sy = _layout_scene("disk" if name == "chw_T" else name)
    if name == "chw_T":
        f_img = np.ascontiguousarray(img.transpose(2, 1, 0)).T
    else:
        f_img = np.asfortranarray(img)
    assert f_img.flags.f_contiguous and not f_img.flags.c_contiguous
    np.testing.assert_array_equal(f_img, img)
    before = f_img.copy(order="K")
    assert_matches_reference(f_img, sx, sy, connectivity)
    np.testing.assert_array_equal(f_img, before)      # input untouched


def test_run_on_a_fortran_ordered_device_array(switch_mode):
    """TileFill.run() takes a device array as it is. The separate pack and
    paint index img[x, y, c] through its strides, so F order works; the
    fused pack reads the raw buffer as C-ordered (x, y, c) words, so it
    must refuse F order instead of filling the wrong pixels."""
    img, sx, sy = _layout_scene("disk")
    exp_img, _, _ = expected_image(img, sx, sy, 8)
    d = cuda.to_device(np.asfortranarray(img))
    assert d.is_f_contiguous() and not d.is_c_contiguous()
    eng = TileFill(img.shape[0], img.shape[1], connectivity=8)
    if eng.fused_pack:
        with pytest.raises(ValueError, match="C-contiguous"):
            eng.run(d, sx, sy)
    else:
        eng.run(d, sx, sy)
        np.testing.assert_array_equal(d.copy_to_host(), exp_img)


@pytest.mark.parametrize("height", [128, 129, 130, 131])
def test_run_on_an_offset_device_view(height, switch_mode):
    """Regression: TileFill.run() on d_big[1:], a C-contiguous view that
    starts 3 * height bytes into its allocation. The fused pack loads and
    stores the image as uint32 words; with 3 * height % 4 != 0 every word
    was misaligned and the kernel died with CUDA_ERROR_MISALIGNED_ADDRESS
    (poisoning the context). Such a view is now refused; the separate
    pack and paint fill it exactly, at every offset."""
    w = 150
    img, sx, sy = scenes.disk_scene(w, height, 50)
    exp_img, _, _ = expected_image(img, sx, sy, 8)
    big = np.zeros((w + 1, height, 3), dtype=np.uint8)
    big[1:] = img
    d = cuda.to_device(big)[1:]
    assert d.is_c_contiguous()
    offset = d.__cuda_array_interface__["data"][0] % 4
    assert offset == 3 * height % 4
    eng = TileFill(w, height, connectivity=8)
    if eng.fused_pack and offset:
        with pytest.raises(ValueError, match="4-byte aligned"):
            eng.run(d, sx, sy)
    else:
        eng.run(d, sx, sy)
        np.testing.assert_array_equal(d.copy_to_host(), exp_img)


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


# ------------------------------------------------------------ switches

def _near_red_image(w, h, rng, p_red=0.5):
    """RGB noise of red, near-red ((254,0,0), (255,1,0), (255,0,1),
    (255,0,255)), white, blue and random colours: every byte the pack's
    word test looks at is exercised."""
    palette = np.array([[255, 0, 0], [254, 0, 0], [255, 1, 0], [255, 0, 1],
                        [255, 0, 255], [255, 255, 255], [0, 0, 255],
                        [0, 0, 0]], dtype=np.uint8)
    pick = rng.integers(1, len(palette), size=(w, h))
    pick[rng.random((w, h)) < p_red] = 0
    img = palette[pick]
    rnd = rng.random((w, h)) < 0.05
    img[rnd] = rng.integers(0, 256, size=(int(rnd.sum()), 3), dtype=np.uint8)
    return np.ascontiguousarray(img)


@cuda.jit
def _pack_probe(img32, aligned, width, height, wpr, n32, tail, out):
    """kernels._pack_tile on every tile (one warp per tile)."""
    lane = cuda.laneid
    t = cuda.grid(1) >> 5
    tiles_x = (width + 31) // 32
    if t < tiles_x * wpr:                       # warp-uniform
        tx = t // wpr
        ty = t - tx * wpr
        h34 = (uint64(height) * kernels.Q3) >> kernels.Q2
        n32m1 = uint64(max(n32 - 1, 0))
        m = kernels._pack_tile(img32, aligned, tx * 32, ty, width, height,
                               h34, uint64(n32), n32m1, tail, lane)
        x = tx * 32 + lane
        if x < width:
            out[x * wpr + ty] = m


@pytest.mark.parametrize("shape", [
    (64, 64), (100, 96), (37, 128),              # rows 4-byte aligned
    (33, 65), (97, 161), (50, 1), (31, 31),      # height % 4 = 1, 1, 1, 3
    (70, 66), (5, 7), (40, 34), (3, 99),         # height % 4 = 2, 3, 2, 3
    (1, 1), (1, 2), (2, 1), (1, 5), (7, 33)])    # 3-15 bytes, tail words
def test_fused_pack_equals_ch06_pack(shape, switch_mode):
    """The fused pack's word test (three XOR-ed words per 4 pixels, a
    funnel shift and the image's partial last word when rows are not
    4-byte aligned) against ch06's pack_kernel, on near-red noise."""
    only_in(switch_mode, "old")
    w, h = shape
    img = _near_red_image(w, h, np.random.default_rng(w * 1000 + h))
    eng = TileFill(w, h, connectivity=8)
    d_img = cuda.to_device(img)
    eng.pack(d_img)
    want = eng.mask.copy_to_host()
    n = w * h * 3
    n32 = n // 4
    flat = img.reshape(n)
    tail = np.uint32(sum(int(b) << (8 * k)
                         for k, b in enumerate(flat[4 * n32:])))
    img32 = (d_img.reshape(n)[:4 * n32].view(np.uint32) if n32
             else cuda.device_array(1, np.uint32))
    wpr = (h + 31) // 32
    out = cuda.to_device(np.zeros(w * wpr, np.uint32))
    tiles = ((w + 31) // 32) * wpr
    _pack_probe[tiles, 32](img32, h % 4 == 0, w, h, wpr, n32, tail, out)
    np.testing.assert_array_equal(out.copy_to_host().reshape(w, wpr), want)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("blocks, tpb", [(1, 32), (1, 64), (2, 32), (2, 64)])
@pytest.mark.parametrize("name", ["serpentine", "zz_serpentine", "maze",
                                  "random"])
def test_tiny_grids_many_rounds(name, blocks, tpb, connectivity):
    """One and two blocks: every round's warps share one or two SMs, and
    on the serpentines a round holds one tile, so the barrier (one or
    two grid.sync per round) is crossed thousands of times."""
    img, sx, sy = {
        "serpentine": lambda: scenes.serpentine_scene(128, 160),
        "zz_serpentine": lambda: worst_cases.zz_serpentine_scene(256, 256),
        "maze": lambda: worst_cases.maze_scene(cells=40),
        "random": lambda: scenes.random_scene(200, 230, 0.62, rng_seed=11),
    }[name]()
    assert_matches_reference(img, sx, sy, connectivity,
                             threads_per_block=tpb, blocks=blocks)


@pytest.mark.parametrize("blocks", [1, 2, None])
@pytest.mark.parametrize("name", ["serpentine", "zz_serpentine"])
def test_one_sync_keeps_the_rounds(name, blocks, switch_mode):
    """A one-tile front has a fixed schedule, so one grid.sync per round
    must take exactly the rounds (and visits) of two."""
    only_in(switch_mode, "old")
    img, sx, sy = (scenes.serpentine_scene(128, 160) if name == "serpentine"
                   else worst_cases.zz_serpentine_scene(256, 256))
    two = flood_fill(img, sx, sy, connectivity=8, blocks=blocks,
                     one_sync=False)
    one = assert_matches_reference(img, sx, sy, 8, blocks=blocks,
                                   one_sync=True)
    assert one.one_sync and not two.one_sync
    assert one.rounds == two.rounds > 250
    assert one.tile_visits == two.tile_visits


@pytest.mark.parametrize("connectivity", [4, 8])
def test_fused_pack_reads_only_reached_tiles(connectivity):
    """A 21x13 blob in a 4096 x 2048 image (8,192 tiles), and the thin
    strip: only the tiles the blob covers are packed. Neither blob reaches
    a tile border of a white tile, so no candidate lands outside them."""
    img = np.full((4096, 2048, 3), 255, dtype=np.uint8)
    img[1010:1031, 700:713] = RED     # x across 1023|1024, y across 703|704
    r = assert_matches_reference(img, 1020, 705, connectivity,
                                 fused_pack=True)
    assert r.fused_pack and r.filled == 21 * 13
    assert r.packed_tiles == 4
    img, sx, sy = worst_cases.thin_h32_blob_scene(1 << 15)
    r = assert_matches_reference(img, sx, sy, connectivity, fused_pack=True)
    assert r.packed_tiles == 2
    r = flood_fill(img, sx, sy, connectivity=connectivity, fused_pack=False,
                   fused_paint=False)
    assert r.packed_tiles == 0


@pytest.mark.parametrize("connectivity", [4, 8])
def test_near_red_neighbours_are_never_painted(connectivity):
    """Candidates (unfiltered border bits into unpacked tiles) land on
    near-red pixels here: the blob ends on tile borders (x 64 and 255,
    y 32 and 223) and a near-red ring sits just across them, so every
    border export of the blob's edge tiles is a near-red candidate. None
    may be painted."""
    rng = np.random.default_rng(5)
    img = _near_red_image(300, 260, rng, p_red=0.0)
    img[63:257, 31:225] = (254, 0, 0)                  # the ring ...
    img[64:256, 32:224] = RED                          # ... around the blob
    img[100:200, 96:98] = (255, 0, 1)                  # a near-red wall
    r = assert_matches_reference(img, 150, 120, connectivity)
    changed = np.any(r.img != img, axis=2)
    red = np.all(img == RED, axis=2)
    assert not np.any(changed & ~red)


@pytest.mark.parametrize("name", ["full_red_300", "random_supercritical",
                                  "disk_301"])
def test_repeated_runs_are_identical(name):
    """Races (same-round exports, packing while another warp exports,
    skipped exports) change the work, never the result: 12 runs on
    reused buffers, at the default grid and at one block."""
    img, sx, sy = {
        "full_red_300": lambda: scenes.full_red_scene(300, 300),
        "random_supercritical": lambda: scenes.random_scene(
            256, 256, 0.62, rng_seed=17),
        "disk_301": lambda: scenes.disk_scene(301, 301, 140),
    }[name]()
    exp_img, _, _ = expected_image(img, sx, sy, 8)
    d0 = cuda.to_device(img)
    for blocks in (None, 1):
        eng = TileFill(img.shape[0], img.shape[1], connectivity=8,
                       blocks=blocks)
        d = cuda.device_array_like(img)
        for _ in range(12):
            d.copy_to_device(d0)
            eng.run(d, sx, sy)
            np.testing.assert_array_equal(d.copy_to_host(), exp_img)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("shape", [(512, 512), (300, 200), (97, 161)])
def test_skip_full_visits_each_full_tile_once(shape, connectivity,
                                              switch_mode):
    """Fused pack + skip-full on an all-red image: every tile is full, so
    every export into an already packed tile is dropped and each tile is
    visited exactly once (its full bit is visible from the round after
    its pack on). Without skip-full, 8-conn races re-visit tiles. (At
    4-conn the tile graph is bipartite: no two tiles of a round touch,
    so there is nothing to skip and nothing to re-visit.)"""
    only_in(switch_mode, "old")
    w, h = shape
    full = np.zeros((w, h, 3), dtype=np.uint8)
    full[:, :] = RED
    for sx, sy in ((0, 0), (w // 2, h // 2), (w - 1, h // 3)):
        r = assert_matches_reference(full, sx, sy, connectivity,
                                     fused_pack=True, skip_full=True,
                                     one_sync=True)
        assert r.tile_visits == r.n_tiles == r.packed_tiles
        if r.n_tiles > 4 and connectivity == 8:
            assert r.skipped_exports > 0


# A child process: the jittered build can deadlock, and a hung cooperative
# kernel would hang the test session with it. argv: width, height, radius,
# connectivity, one_sync, runs.
_JITTER_CHILD = r"""
import os, sys, warnings
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
warnings.filterwarnings("ignore")
from numba import cuda
from flood_fill_cuda.chapters.ch07_gpu_1blob_tiles import flood_fill as ff
from flood_fill_cuda.chapters.ch07_gpu_1blob_tiles import kernels as k7
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import scenes
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock.cpu_oracle import (
    cpu_flood_fill, cpu_flood_fill_8)
w, h, rad, conn, one_sync, runs = (int(v) for v in sys.argv[1:7])
img, sx, sy = scenes.disk_scene(w, h, rad)
vis = (cpu_flood_fill_8 if conn == 8 else cpu_flood_fill)(img, sx, sy)[0]
want = img.copy()
want[vis == 1] = (0, 0, 255)
eng = ff.TileFill(w, h, connectivity=conn, one_sync=bool(one_sync),
                  fused_pack=False, skip_full=True, fused_paint=False)
eng.kernel_fn = k7._make_rounds_kernel(conn == 8, True, None,
                                       one_sync=bool(one_sync),
                                       skip_full=True, jitter=True)
d0 = cuda.to_device(img)
d = cuda.device_array_like(img)
for k in range(runs):
    d.copy_to_device(d0)
    eng.run(d, sx, sy)
    cuda.synchronize()
    print("run", k, "ok" if (d.copy_to_host() == want).all() else "WRONG",
          flush=True)
print("COMPLETED", flush=True)
"""
JITTER_RUNS = 20
JITTER_TIMEOUT_S = 120     # a passing child takes about 10 s


@pytest.mark.parametrize("one_sync", [False, True])
@pytest.mark.parametrize("scene", [(301, 301, 140, 8), (559, 232, 100, 4)])
def test_skip_full_vote_survives_schedule_jitter(scene, one_sync,
                                                 switch_mode):
    """Regression: skip_full without the fused pack voted
    `st >= rnd and all_sync(FULL, ...)`. The stamp st is a plain load that
    other warps raise in the same round, so the lanes of one warp may
    disagree on it, and then only some of them reach a FULL-mask vote.
    That is undefined behaviour; it held only while the warp happened to
    run in lockstep. The jittered build (thread 0 sleeps 1.5-2.7 us at the
    top of two rounds in three, a delay the CUDA model allows) deadlocked
    on both scenes, with one and with two grid.sync per round."""
    only_in(switch_mode, "old")
    args = [sys.executable, "-c", _JITTER_CHILD,
            *(str(v) for v in (*scene, int(one_sync), JITTER_RUNS))]
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=JITTER_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
        pytest.fail(f"deadlock: no completion within {JITTER_TIMEOUT_S} s "
                    f"(last output: {(out or '').splitlines()[-1:]})")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "COMPLETED" in r.stdout
    assert r.stdout.count(" ok") == JITTER_RUNS, r.stdout


def test_counters_of_the_switches(switch_mode):
    """packed_tiles counts the fused pack's reads: every touched tile, at
    most every tile; 0 without it. skipped_exports is 0 without
    skip-full."""
    img, sx, sy = scenes.disk_scene(301, 301, 140)
    r = assert_matches_reference(img, sx, sy, 8)
    eng = TileFill(301, 301, connectivity=8)
    d = cuda.to_device(img)
    eng.run(d, sx, sy)
    touched = eng.tiles_touched()
    if r.fused_pack:
        assert touched <= r.packed_tiles <= r.n_tiles
    else:
        assert r.packed_tiles == 0
    if not r.skip_full:
        assert r.skipped_exports == 0


def test_switch_validation(switch_mode):
    only_in(switch_mode, "old")
    img, sx, sy = scenes.square_scene(64, 64, 20, 20)
    with pytest.raises(ValueError, match="fused_pack"):
        flood_fill(img, sx, sy, fused_pack=False, fused_paint=True)
    with pytest.raises(TypeError, match="unknown switch"):
        TileFill(64, 64, connectivity=8, fused_pak=True)
