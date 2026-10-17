"""Correctness tests: BFS on every CPU core vs the one-core references.

The bit helpers the tile variants are built from are tested first,
against plain-Python references, so a failure further down can be
localised. Then each variant against its oracle:

    pixel BFS    the @njit level-synchronous BFS (ch01's at 4-conn,
                 shared/'s at 8-conn): the same visited set, the same
                 fill count and the same LEVEL count, the painted image
    tile BFS     the same oracles: visited set, fill count, painted image
                 (no per-pixel depth: a round is a step of the tile graph)
    tile labels  ch05's cpu_label_components (8-conn) and ch08's
                 cpu_label (4-conn): the canonical label map, the blob
                 count, and the palette paint ch05 and ch08 use

Thread counts 1, 2 and 16 (numba.set_num_threads) run the same scenes,
and repeated runs must give identical outputs.

Run:

    uv run pytest src/flood_fill_cuda/chapters/ch09_cpu_nthreads/test_correctness.py -v
"""

import numba
import numpy as np
import pytest

from . import kernels as K
from . import flood_fill as ff
from ..ch03_gpu_1blob_nblock.cpu_oracle import cpu_flood_fill, cpu_flood_fill_8
from ..ch03_gpu_1blob_nblock import scenes
from ..ch07_gpu_1blob_tiles.benchmarks import worst_cases
from ..ch08_gpu_nblob_tiles import scenes as scenes8

BLUE = np.array([0, 0, 255], dtype=np.uint8)
RED = scenes.RED
THREADS = (1, 2, 16)
FILL_METHODS = ("pixel", "tile")

RNG_SEEDS = (0, 1, 2, 3)


# ------------------------------------------------------------ references

def _py_row_fill(s, m):
    """Runs of m (a Python int) holding a bit of s."""
    out = 0
    b = 0
    while b < 64:
        if (m >> b) & 1:
            lo = b
            while b + 1 < 64 and (m >> (b + 1)) & 1:
                b += 1
            run = ((1 << (b + 1)) - 1) ^ ((1 << lo) - 1)
            if s & run:
                out |= run
        b += 1
    return out


def _py_close(f, m, conn8):
    """Pixel BFS inside the 64 x 64 tile: the pieces of m holding a bit
    of f. f, m: lists of 64 Python ints."""
    seen = [0] * 64
    stack = []
    for i in range(64):
        for b in range(64):
            if (f[i] >> b) & 1:
                seen[i] |= 1 << b
                stack.append((i, b))
    while stack:
        i, b = stack.pop()
        for di in (-1, 0, 1):
            for db in (-1, 0, 1):
                if di == 0 and db == 0:
                    continue
                if not conn8 and di != 0 and db != 0:
                    continue
                ni, nb = i + di, b + db
                if 0 <= ni < 64 and 0 <= nb < 64:
                    if (m[ni] >> nb) & 1 and not (seen[ni] >> nb) & 1:
                        seen[ni] |= 1 << nb
                        stack.append((ni, nb))
    return seen


def _rand_words(rng, n, density):
    bits = rng.random((n, 64)) < density
    return [int(sum(1 << b for b in range(64) if row[b])) for row in bits]


def _u64(words):
    return np.array(words, dtype=np.uint64)


# ------------------------------------------------------------ bit helpers

@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_row_fill_matches_reference(seed):
    rng = np.random.default_rng(seed)
    for density in (0.1, 0.5, 0.8, 0.97):
        ms = _rand_words(rng, 200, density)
        for m in ms:
            # seeds: a random subset of m, sometimes one bit, sometimes many
            pick = int(rng.integers(0, 1 << 63, dtype=np.int64)) \
                | (int(rng.integers(0, 2)) << 63)
            for s in (m & pick, m & (pick >> 17), m & -m if m else 0, m):
                got = int(K.row_fill(np.uint64(s), np.uint64(m)))
                assert got == _py_row_fill(s, m), (hex(s), hex(m))


def test_row_fill_edges():
    full = (1 << 64) - 1
    cases = [(0, 0), (0, full), (1, full), (1 << 63, full),
             (1 << 63, 1 << 63), (1, 1), (1 << 20, 0xFFFFF << 10),
             ((1 << 63) | 1, full ^ (1 << 32)),
             (1 << 40, full ^ (1 << 32) ^ (1 << 50))]
    for s, m in cases:
        assert int(K.row_fill(np.uint64(s), np.uint64(m))) == \
            _py_row_fill(s, m), (hex(s), hex(m))


def test_popcount_lowest_bit_dilate():
    rng = np.random.default_rng(5)
    for _ in range(500):
        x = int(rng.integers(0, 1 << 63, dtype=np.int64)) \
            | (int(rng.integers(0, 2)) << 63)
        assert K.popcount(np.uint64(x)) == bin(x).count("1")
        if x:
            assert K.lowest_bit(np.uint64(x)) == (x & -x).bit_length() - 1
        d = (x | (x << 1) | (x >> 1)) & ((1 << 64) - 1)
        assert int(K.dilate(np.uint64(x))) == d
        assert int(K.spread(np.uint64(x), True)) == d
        assert int(K.spread(np.uint64(x), False)) == x
    for i in range(64):
        assert int(K.bit(i)) == 1 << i
        assert K.lowest_bit(np.uint64(1 << i)) == i


def test_columns():
    rng = np.random.default_rng(6)
    words = _rand_words(rng, 64, 0.5)
    f = _u64(words)
    lo = sum(((w & 1) << i) for i, w in enumerate(words))
    hi = sum((((w >> 63) & 1) << i) for i, w in enumerate(words))
    assert int(K.col_low(f, 64)) == lo
    assert int(K.col_high(f, 64)) == hi
    assert int(K.col_low(f, 10)) == lo & ((1 << 10) - 1)


@pytest.mark.parametrize("conn8", [False, True])
@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_close_from_matches_pixel_bfs(conn8, seed):
    """A seed bit closed with the passes equals a pixel BFS in the tile,
    on noise around the percolation threshold and on sparse/dense
    noise."""
    rng = np.random.default_rng(seed)
    for density in (0.3, 0.45, 0.55, 0.6, 0.75, 0.95):
        mw = _rand_words(rng, 64, density)
        m = _u64(mw)
        for _ in range(6):
            i = int(rng.integers(0, 64))
            if mw[i] == 0:
                continue
            bits = [b for b in range(64) if (mw[i] >> b) & 1]
            b = bits[int(rng.integers(0, len(bits)))]
            f = np.zeros(64, np.uint64)
            passes, lo, hi = K.close_from(f, m, conn8, i, np.uint64(1 << b))
            seed_f = [0] * 64
            seed_f[i] = 1 << b
            want = _py_close(seed_f, mw, conn8)
            assert [int(w) for w in f] == want
            rows = [r for r in range(64) if want[r]]
            assert (lo, hi) == (rows[0], rows[-1])
            assert passes >= 1


@pytest.mark.parametrize("conn8", [False, True])
def test_close_tile_after_new_seeds(conn8):
    """A closed f plus new row-closed seeds on flagged rows closes to the
    pieces of the union, from either end (the dirty-row protocol)."""
    rng = np.random.default_rng(11)
    for trial in range(60):
        density = (0.35, 0.5, 0.6, 0.9)[trial % 4]
        mw = _rand_words(rng, 64, density)
        m = _u64(mw)
        # a closed start: one piece
        f = np.zeros(64, np.uint64)
        start = [r for r in range(64) if mw[r]]
        r0 = start[int(rng.integers(0, len(start)))]
        _, lo, hi = K.close_from(f, m, conn8, r0, np.uint64(mw[r0] & -mw[r0]))
        before = [int(w) for w in f]
        # new seeds: border rows (like inboxes) and a random row
        dirty = 0
        new = [0] * 64
        for r in (0, 63, int(rng.integers(0, 64))):
            if mw[r]:
                bits = [b for b in range(64) if (mw[r] >> b) & 1]
                b = bits[int(rng.integers(0, len(bits)))]
                new[r] |= 1 << b
        for r in range(64):
            if new[r] & ~before[r]:
                f[r] = K.row_fill(np.uint64(before[r] | new[r]),
                                  np.uint64(mw[r]))
                dirty |= 1 << r
                if lo > hi:
                    lo = hi = r
                lo, hi = min(lo, r), max(hi, r)
        K.close_tile(f, m, conn8, np.uint64(dirty), lo, hi)
        want = _py_close([before[r] | new[r] for r in range(64)], mw, conn8)
        assert [int(w) for w in f] == want


def test_close_tile_serpentine_needs_many_passes():
    """A vertical serpentine inside one tile turns at both ends: the
    passes alternate until it is filled (and only then stop)."""
    mw = [0] * 64
    for i in range(0, 64, 2):
        mw[i] = (1 << 64) - 1
    for i in range(1, 64, 2):
        mw[i] = 1 << 63 if (i // 2) % 2 == 0 else 1
    m = _u64(mw)
    f = np.zeros(64, np.uint64)
    passes, lo, hi = K.close_from(f, m, True, 0, np.uint64(1))
    assert [int(w) for w in f] == mw
    assert (lo, hi) == (0, 63)
    # one down pass covers it all (every turn is downward), one proves
    assert passes <= 2
    f = np.zeros(64, np.uint64)
    passes, _, _ = K.close_from(f, m, True, 62, np.uint64(1))
    assert [int(w) for w in f] == mw


# ------------------------------------------------------------ one blob

def expected_fill(img, sx, sy, connectivity):
    ref = cpu_flood_fill_8 if connectivity == 8 else cpu_flood_fill
    visited, _, levels, filled = ref(img, sx, sy)
    out = img.copy()
    out[visited == 1] = BLUE
    return out, visited, int(levels), int(filled)


def _white(w, h):
    return np.full((w, h, 3), 255, dtype=np.uint8)


def checkerboard_scene(w, h):
    img = _white(w, h)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    img[(xs + ys) % 2 == 0] = RED
    return img, 0, 0


def tile_border_zigzag_scene(w=320, h=300, tile=64):
    """Zigzags along y on every x tile border (rows 64k+63 | 64k+64) and
    along x on every y border, joined: rounds == levels for a 64-tile
    BFS (ch07's zz_lattice at this chapter's tile size)."""
    img = _white(w, h)
    ys = np.arange(h)
    for k in range(w // tile):
        if tile * k + tile < w:
            img[tile * k + tile - 1 + (ys % 2), ys] = RED
    xs = np.arange(w)
    for j in range(h // tile):
        if tile * j + tile < h:
            img[xs, tile * j + tile - 1 + (xs % 2)] = RED
    red = np.argwhere((img[..., 0] == 255) & (img[..., 1] == 0))
    return img, int(red[0][0]), int(red[0][1])


def corner_pixels_scene(w=193, h=130, tile=64):
    """Single pixels and 2x2 blocks on the 64-tile corners, joined only
    diagonally across the corner, plus the image corners."""
    img = _white(w, h)
    for x in range(tile, w, tile):
        for y in range(tile, h, tile):
            img[x - 1, y - 1] = RED
            img[x, y] = RED
    for x, y in ((0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)):
        img[x, y] = RED
    return img, tile - 1, tile - 1


def _noise(w, h, density, seed):
    return scenes.random_scene(w, h, density, rng_seed=seed)


ONE_BLOB = {
    "square_off64": lambda: scenes.square_scene(300, 200, 150, 101),
    "corner_square": lambda: scenes.square_scene(130, 70, 130, 70,
                                                 corner=True),
    "disk_257": lambda: scenes.disk_scene(257, 257, 100),
    "serpentine_64": lambda: scenes.serpentine_scene(64, 64),
    "serpentine_100x37": lambda: scenes.serpentine_scene(100, 37),
    "full_red_130x70": lambda: scenes.full_red_scene(130, 70),
    "single_px": lambda: scenes.single_pixel_scene(65, 129),
    "one_by_one": lambda: scenes.full_red_scene(1, 1),
    "one_row": lambda: scenes.full_red_scene(1, 200),
    "one_col": lambda: scenes.full_red_scene(200, 1),
    "noise_0.40": lambda: _noise(200, 170, 0.40, 1),
    "noise_0.45": lambda: _noise(200, 170, 0.45, 2),
    "noise_0.59": lambda: _noise(160, 200, 0.59, 3),
    "noise_0.62": lambda: _noise(160, 200, 0.62, 4),
    "checkerboard": lambda: checkerboard_scene(130, 67),
    "spiral": lambda: (lambda im: (im, int(np.argwhere(
        im[..., 1] == 0)[0][0]), int(np.argwhere(im[..., 1] == 0)[0][1])))(
            scenes8.spiral_scene(128, 2)),
    "maze_63": lambda: worst_cases.maze_scene(cells=32, rng_seed=3),
    "zz_serp": lambda: worst_cases.zz_serpentine_scene(256, 200),
    "tile_zigzags": tile_border_zigzag_scene,
    "tile_corners": corner_pixels_scene,
}


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("method", FILL_METHODS)
@pytest.mark.parametrize("name", list(ONE_BLOB))
def test_fill_matches_oracle(name, method, connectivity, threads):
    img, sx, sy = ONE_BLOB[name]()
    want_img, want_vis, want_levels, want_filled = expected_fill(
        img, sx, sy, connectivity)
    res = ff.flood_fill(img, sx, sy, connectivity=connectivity,
                        method=method, threads=threads)
    np.testing.assert_array_equal(res.visited, want_vis)
    np.testing.assert_array_equal(res.img, want_img)
    assert res.filled == want_filled
    assert res.threads == threads
    if method == "pixel":
        assert res.steps == want_levels
    else:
        # a round moves the wave at least one pixel level
        assert 1 <= res.steps <= want_levels


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("name", ["full_red_512", "noise_0.62_big",
                                  "disk_400"])
def test_pixel_bfs_forced_races(name, connectivity):
    """Every level split into as many chunks as allowed (serial_below=1,
    min_chunk=1, 16 threads): races between chunks are as frequent as
    they get, and the output must not move."""
    img, sx, sy = {
        "full_red_512": lambda: scenes.full_red_scene(512, 512),
        "noise_0.62_big": lambda: _noise(500, 400, 0.62, 9),
        "disk_400": lambda: scenes.disk_scene(400, 400, 190),
    }[name]()
    if name == "full_red_512":
        sx, sy = 256, 256
    want_img, want_vis, want_levels, want_filled = expected_fill(
        img, sx, sy, connectivity)
    eng = ff.PixelBFS(img.shape[0], img.shape[1], connectivity=connectivity,
                      threads=16, serial_below=1, min_chunk=1)
    for _ in range(3):
        work = img.copy()
        st = eng.run(work, sx, sy)
        np.testing.assert_array_equal(eng.visited(), want_vis == 1)
        np.testing.assert_array_equal(work, want_img)
        assert st["filled"] == want_filled
        assert st["levels"] == want_levels
        assert st["parallel_levels"] > 0
        # duplicates never multiply: at most one copy per racing chunk
        assert st["duplicates"] < want_filled


def test_pixel_bfs_repeatable_and_reusable():
    """One engine, many runs, alternating seeds and thread counts: every
    output equals the oracle's (the buffers are reset on the clock)."""
    img, _, _ = scenes.square_scene(300, 260, 200, 150)
    img[0:20, 0:20] = RED
    seeds = [(150, 130), (5, 5), (150, 130)]
    eng = ff.PixelBFS(300, 260, connectivity=8, serial_below=1, min_chunk=8)
    for i in range(6):
        sx, sy = seeds[i % 3]
        want_img, want_vis, want_levels, want_filled = expected_fill(
            img, sx, sy, 8)
        work = img.copy()
        with ff.num_threads(THREADS[i % 3]):
            st = eng.run(work, sx, sy)
        np.testing.assert_array_equal(work, want_img)
        assert (st["levels"], st["filled"]) == (want_levels, want_filled)


@pytest.mark.parametrize("method", FILL_METHODS)
def test_non_red_seed_raises(method):
    img, _, _ = scenes.square_scene(64, 64, 10, 10)
    with pytest.raises(ValueError, match="not red"):
        ff.flood_fill(img, 0, 0, method=method)
    blank = _white(40, 30)
    with pytest.raises(ValueError, match="not red"):
        ff.flood_fill(blank, 3, 3, method=method)
    with pytest.raises(ValueError, match="outside"):
        ff.flood_fill(img, 64, 0, method=method)


@pytest.mark.parametrize("method", FILL_METHODS)
def test_bad_inputs(method):
    img, sx, sy = scenes.square_scene(64, 64, 10, 10)
    with pytest.raises(ValueError):
        ff.flood_fill(img, sx, sy, connectivity=6, method=method)
    with pytest.raises(ValueError):
        ff.flood_fill(img, sx, sy, method="queue")
    with pytest.raises(ValueError):
        ff.flood_fill(img[..., :2], sx, sy, method=method)
    with pytest.raises(ValueError):
        ff.flood_fill(img, sx, sy, method=method,
                      threads=numba.config.NUMBA_NUM_THREADS + 1)
    eng = ff.ENGINES[method](64, 64)
    with pytest.raises(ValueError):
        eng.run(np.full((64, 65, 3), 255, np.uint8), 0, 0)


def test_input_layouts_and_seed_dtypes():
    img, sx, sy = scenes.disk_scene(150, 120, 50)
    want_img, _, _, _ = expected_fill(img, sx, sy, 8)
    f_order = np.asfortranarray(img)
    res = ff.flood_fill(f_order, np.int16(sx), np.uint8(sy), method="pixel")
    np.testing.assert_array_equal(res.img, want_img)
    np.testing.assert_array_equal(img, scenes.disk_scene(150, 120, 50)[0])


# ------------------------------------------------------------ tile BFS

def _red(img):
    return (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)


@pytest.mark.parametrize("shape", [(1, 1), (64, 64), (65, 63), (130, 200),
                                   (3, 300), (300, 3)])
def test_pack_tiles_and_columns(shape):
    """The tile-major mask equals the red mask bit for bit (colours that
    share bytes with red do not count), padding rows and bits are 0, and
    the mask columns are the tiles' first and last bits."""
    w, h = shape
    rng = np.random.default_rng(w * 1000 + h)
    img = rng.integers(0, 256, (w, h, 3)).astype(np.uint8)
    pick = rng.random((w, h))
    img[pick < 0.4] = RED
    img[(pick >= 0.4) & (pick < 0.5)] = (255, 0, 1)
    img[(pick >= 0.5) & (pick < 0.6)] = (255, 1, 0)
    img[(pick >= 0.6) & (pick < 0.7)] = (254, 0, 0)
    eng = ff.TileBFS(w, h)
    eng.mask[:] = np.uint64(0xDEADBEEFDEADBEEF)   # poison: all rewritten
    K.pack_tiles(img, eng.mask, eng.wpr)
    K.tile_columns(eng.mask, eng.colm)
    got = np.empty((w, h), np.bool_)
    K.unpack_tiles(eng.mask, eng.wpr, got)
    np.testing.assert_array_equal(got, _red(img))
    assert K.popcount_words(eng.mask) == int(_red(img).sum())
    for t in range(eng.n_tiles):
        rows = [int(v) for v in eng.mask[t]]
        assert int(eng.colm[t, 0]) == sum((r & 1) << i
                                          for i, r in enumerate(rows))
        assert int(eng.colm[t, 1]) == sum(((r >> 63) & 1) << i
                                          for i, r in enumerate(rows))


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("name", ["full_red_512", "noise_0.62_big",
                                  "disk_400", "maze_255", "tile_zigzags"])
def test_tile_bfs_every_round_parallel(name, connectivity):
    """serial_below=1: every round of two tiles or more is a prange, at
    16 threads; the same engine three times."""
    img, sx, sy = {
        "full_red_512": lambda: (scenes.full_red_scene(512, 512)[0], 256,
                                 256),
        "noise_0.62_big": lambda: _noise(500, 400, 0.62, 9),
        "disk_400": lambda: scenes.disk_scene(400, 400, 190),
        "maze_255": lambda: worst_cases.maze_scene(cells=128, rng_seed=5),
        "tile_zigzags": lambda: tile_border_zigzag_scene(700, 520),
    }[name]()
    want_img, want_vis, want_levels, want_filled = expected_fill(
        img, sx, sy, connectivity)
    eng = ff.TileBFS(img.shape[0], img.shape[1], connectivity=connectivity,
                     threads=16, serial_below=1)
    for _ in range(3):
        work = img.copy()
        st = eng.run(work, sx, sy)
        np.testing.assert_array_equal(eng.visited(), want_vis == 1)
        np.testing.assert_array_equal(work, want_img)
        assert eng.filled() == want_filled
        assert st["peak_round_tiles"] < 2 or st["parallel_rounds"] >= 1
        assert st["tile_visits"] >= st["idle_visits"]


def test_tile_bfs_rounds_on_a_solid_square():
    """A center-seeded full square: the 8-conn tile graph's BFS from the
    seed tile reaches the farthest tile in max(|dtx|, |dty|) rounds, plus
    one round for the last sends to land; every tile is closed once and
    the bounces are filtered (few idle visits)."""
    img, _, _ = scenes.full_red_scene(640, 640)
    eng = ff.TileBFS(640, 640, connectivity=8)
    st = eng.run(img, 320, 320)
    assert eng.filled() == 640 * 640
    # seed tile (5, 5) of 10 x 10: farthest tile 5 steps away
    assert st["rounds"] in (6, 7)
    assert st["tile_visits"] - st["idle_visits"] == 100
    assert st["idle_visits"] <= 2 * 100


def test_tile_bfs_reuse_across_seeds_and_threads():
    img, _, _ = scenes.square_scene(300, 260, 200, 150)
    img[0:20, 0:20] = RED
    img[290:300, 250:260] = RED
    seeds = [(150, 130), (5, 5), (295, 255), (150, 130)]
    eng = ff.TileBFS(300, 260, connectivity=8, serial_below=1)
    for i in range(8):
        sx, sy = seeds[i % 4]
        want_img, want_vis, _, want_filled = expected_fill(img, sx, sy, 8)
        work = img.copy()
        with ff.num_threads(THREADS[i % 3]):
            eng.run(work, sx, sy)
        np.testing.assert_array_equal(work, want_img)
        assert eng.filled() == want_filled


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("connectivity", [4, 8])
def test_tile_bfs_fuzz(seed, connectivity):
    """Random rectangles, disks and noise patches on odd image sizes,
    seeded on random red pixels: both methods equal the oracle."""
    rng = np.random.default_rng(100 + seed)
    w, h = int(rng.integers(1, 400)), int(rng.integers(1, 400))
    img = _white(w, h)
    for _ in range(int(rng.integers(1, 12))):
        kind = rng.integers(0, 3)
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        a, b = int(rng.integers(1, 150)), int(rng.integers(1, 150))
        if kind == 0:
            img[x0:x0 + a, y0:y0 + b] = RED
        elif kind == 1:
            xs, ys = np.ogrid[:w, :h]
            img[(xs - x0) ** 2 + (ys - y0) ** 2 <= a * a // 4] = RED
        else:
            patch = rng.random((min(a, w - x0), min(b, h - y0))) < 0.6
            sub = img[x0:x0 + a, y0:y0 + b]
            sub[patch] = RED
    red = np.argwhere(_red(img))
    if len(red) == 0:
        img[w // 2, h // 2] = RED
        red = np.argwhere(_red(img))
    for k in range(3):
        sx, sy = (int(v) for v in red[int(rng.integers(0, len(red)))])
        want_img, want_vis, want_levels, want_filled = expected_fill(
            img, sx, sy, connectivity)
        for method in FILL_METHODS:
            res = ff.flood_fill(img, sx, sy, connectivity=connectivity,
                                method=method, threads=THREADS[k])
            np.testing.assert_array_equal(res.img, want_img)
            assert res.filled == want_filled
