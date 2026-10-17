"""BFS on every CPU core: the GPU chapters' ideas, run with prange.

Chapters 3, 7 and 8 were compared with ONE CPU core. This chapter asks
the fair question: what do ALL the cores do with the same work and the
same ideas? Three variants, each an @njit(parallel=True) loop over
chunks with prange:

    pixel BFS    ch03 on the CPU: a level-synchronous BFS whose frontier
                 is split across threads
    tile BFS     ch07 on the CPU: 64 x 64 tiles closed with word
                 operations, a frontier of tiles, one barrier per round
    tile labels  ch08 on the CPU: local pieces per tile, seeds, waves
                 from piece to piece, unions applied at the end

NO ATOMICS
Numba has no CPU atomics. Every parallel loop here is a prange over
CHUNKS, and every chunk owns the memory it writes: its own slice of an
output buffer, its own counters, its own tiles. Where two chunks can
still touch one location (a pixel claimed from two sides, a piece
claimed by two waves), the location only ever receives whole aligned
values (one byte, or one 4-byte int32), the value read back after the
barrier is one of the values written, and the protocol is built so
that ANY of them gives the same final answer. Each variant's comment
says why.

THE TILE (tile BFS and tile labels)
A tile is 64 consecutive rows (x0 .. x0+63) times one 64-bit word
(64 consecutive y). Tiles are stored tile-major: mask[t, i] is row
x0+i of tile t = tx * wpr + ty, so a tile is 512 contiguous bytes
(eight cache lines). Bit b of a word is y = 64 * ty + b. Rows past the
image (x >= width) and bits past it (y >= height) are 0.

THE CLOSURE (bit helpers below)
    row fill     the runs of a word's mask that hold a seed bit: one
                 carry (an addition) fills toward the high bits, a
                 Kogge-Stone occluded fill (six shift steps) toward
                 the low bits
    column step  row i takes m[i] & spread(f[i-1]); spread is the word
                 itself (4-conn) or its dilation f | f << 1 | f >> 1
                 (8-conn: the diagonals)
    passes       a top-down pass, then bottom-up, alternating; a pass
                 runs only when the previous one changed a row whose
                 backward neighbour can still grow (checked per row)
"""

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros

TILE = 64

# Every word value and every shift count is uint64. Numba turns a
# uint64 mixed with an int64 into float64, so no bare Python int is ever
# combined with a word.
U0 = np.uint64(0)
U1 = np.uint64(1)
ONES = np.uint64(0xFFFFFFFFFFFFFFFF)
S1 = np.uint64(1)
S2 = np.uint64(2)
S4 = np.uint64(4)
S8 = np.uint64(8)
S16 = np.uint64(16)
S32 = np.uint64(32)
S56 = np.uint64(56)
S63 = np.uint64(63)
TOP = np.uint64(1 << 63)
LOW_HALF = np.uint64(0x00000000FFFFFFFF)
_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


# ------------------------------------------------------------ bit helpers

@njit(cache=True)
def bit(i):
    """The word with only bit i set (0 <= i < 64)."""
    return U1 << np.uint64(i)


@njit(cache=True)
def popcount(x):
    """Set bits of a uint64 (SWAR)."""
    x = x - ((x >> S1) & _M1)
    x = (x & _M2) + ((x >> S2) & _M2)
    x = (x + (x >> S4)) & _M4
    return int((x * _H01) >> S56)


@njit(cache=True)
def lowest_bit(x):
    """Index of the lowest set bit of a non-zero uint64."""
    return int(trailing_zeros(x))


@njit(cache=True)
def dilate(f):
    """f and its two neighbours along the word (8-conn spread)."""
    return f | (f << S1) | (f >> S1)


@njit(cache=True)
def spread(f, conn8):
    """The bits of the next row a word f touches: f itself at 4-conn,
    its dilation at 8-conn."""
    if conn8:
        return f | (f << S1) | (f >> S1)
    return f


@njit(cache=True)
def row_fill(s, m):
    """The runs of m that hold a bit of s (s must be a subset of m).

    Toward the high bits by one carry: for a run [a, b] of m whose lowest
    seed bit is c, adding s to m clears bits c..b (except higher seed
    bits, which the final | s restores), sets bits a..c-1 and carries
    into bit b+1, which is 0 in m (a gap), so the carry stops there and
    no run disturbs another. (m + s) ^ m, masked by m, is then exactly
    c..b without the higher seeds. Toward the low bits by a Kogge-Stone
    occluded fill: six shift steps double the reach, and the propagator
    word p keeps a bit only while the whole stretch it has jumped is in
    m.
    """
    g = (((m + s) ^ m) & m) | s
    p = m
    g |= p & (g >> S1)
    p &= p >> S1
    g |= p & (g >> S2)
    p &= p >> S2
    g |= p & (g >> S4)
    p &= p >> S4
    g |= p & (g >> S8)
    p &= p >> S8
    g |= p & (g >> S16)
    p &= p >> S16
    g |= p & (g >> S32)
    return g


@njit(cache=True)
def col_low(f, n):
    """bit i = bit 0 of f[i], for the first n rows (the tile's column at
    y0, its west border)."""
    c = U0
    for i in range(n):
        c |= (f[i] & U1) << np.uint64(i)
    return c


@njit(cache=True)
def col_high(f, n):
    """bit i = bit 63 of f[i], for the first n rows (the east border)."""
    c = U0
    for i in range(n):
        c |= (f[i] >> S63) << np.uint64(i)
    return c


# ---------------------------------------------------------- the closure

@njit(cache=True)
def _pass_down(f, m, conn8, dirty, lo, hi):
    """One top-down pass from row lo. Row i takes the runs of m[i] that
    touch f[i-1] (spread). Rows past hi held no bit of f, so the pass
    stops at the first such row that stays empty.

    Returns (need_up, new hi). need_up: a row that changed in this pass
    (or was flagged in `dirty`) touches bits of m in the row above that
    f lacks, so an upward pass is needed. Row i-1 is final for this pass
    when row i is processed, so the check is exact."""
    need = False
    prev = U0
    if lo > 0:
        prev = f[lo - 1]
    new_hi = hi
    for i in range(lo, TILE):
        mi = m[i]
        fi = f[i]
        if mi == U0:
            if i > hi:
                break
            prev = U0
            continue
        s = fi | (mi & spread(prev, conn8))
        changed = ((dirty >> np.uint64(i)) & U1) != U0
        if s != fi:
            fi = row_fill(s, mi)
            f[i] = fi
            changed = True
        if fi != U0 and i > new_hi:
            new_hi = i
        if changed and i > 0:
            if (m[i - 1] & spread(fi, conn8) & ~f[i - 1]) != U0:
                need = True
        if fi == U0 and i > hi:
            break
        prev = fi
    return need, new_hi


@njit(cache=True)
def _pass_up(f, m, conn8, dirty, lo, hi):
    """The bottom-up mirror of _pass_down, from row hi. Returns
    (need_down, new lo)."""
    need = False
    prev = U0
    if hi < TILE - 1:
        prev = f[hi + 1]
    new_lo = lo
    for i in range(hi, -1, -1):
        mi = m[i]
        fi = f[i]
        if mi == U0:
            if i < lo:
                break
            prev = U0
            continue
        s = fi | (mi & spread(prev, conn8))
        changed = ((dirty >> np.uint64(i)) & U1) != U0
        if s != fi:
            fi = row_fill(s, mi)
            f[i] = fi
            changed = True
        if fi != U0 and i < new_lo:
            new_lo = i
        if changed and i < TILE - 1:
            if (m[i + 1] & spread(fi, conn8) & ~f[i + 1]) != U0:
                need = True
        if fi == U0 and i < lo:
            break
        prev = fi
    return need, new_lo


@njit(cache=True)
def close_tile(f, m, conn8, dirty, lo, hi):
    """Close f inside m: afterwards f is the union of the connected
    pieces of m (inside the tile) that hold a bit of f.

    Preconditions: f is a subset of m, every row of f is row-closed
    (a union of whole runs of its m row), and f was closed except at the
    rows flagged in `dirty` (the rows that took new bits since). lo..hi
    is the row extent of f (lo > hi when f is empty). Why it ends
    closed: the conditions to hold are, for every pair of rows,
    f[i+1] >= m[i+1] & spread(f[i]) (forward) and f[i] >= m[i] &
    spread(f[i+1]) (backward). A pass in one direction leaves its own
    direction's condition true everywhere (each row is processed after
    its predecessor is final), and the other direction's can only fail
    next to a row that grew; each grown row is checked when it is
    final, so a pass that reports no need leaves both true.

    Returns (passes, lo, hi)."""
    if lo > hi:
        return 0, lo, hi
    # start where the new bits are: down if any flagged row is in the
    # top half (or none is flagged), up otherwise
    down = (dirty & LOW_HALF) != U0 or dirty == U0
    passes = 0
    need = True
    while need:
        if down:
            need, hi = _pass_down(f, m, conn8, dirty, lo, hi)
        else:
            need, lo = _pass_up(f, m, conn8, dirty, lo, hi)
        dirty = U0
        passes += 1
        down = not down
    return passes, lo, hi


@njit(cache=True)
def close_from(f, m, conn8, i, s):
    """Seed f (all zero) with bits s of row i (a subset of m[i]) and
    close it. Returns (passes, lo, hi)."""
    f[i] = row_fill(s, m[i])
    return close_tile(f, m, conn8, bit(i), i, i)
