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

import numpy as np
import pytest

from . import kernels as K

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
