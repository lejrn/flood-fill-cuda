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
from numba import njit, prange
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
    return np.int64((x * _H01) >> S56)


@njit(cache=True)
def lowest_bit(x):
    """Index of the lowest set bit of a non-zero uint64."""
    return np.int64(trailing_zeros(x))


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


# ------------------------------------------------------------- shared

@njit(cache=True)
def is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


@njit(parallel=True, cache=True)
def reset_u8(a):
    """a[...] = 0 for a C-contiguous (W, H) uint8 array, rows in
    parallel."""
    for x in prange(a.shape[0]):
        for y in range(a.shape[1]):
            a[x, y] = 0


@njit(parallel=True, cache=True)
def paint_visited(img, visited):
    """Reached pixels (visited != 0) become (0, 0, 255), rows in
    parallel; every row is written by one thread."""
    for x in prange(img.shape[0]):
        for y in range(img.shape[1]):
            if visited[x, y] != 0:
                img[x, y, 0] = 0
                img[x, y, 1] = 0
                img[x, y, 2] = 255


def chunk_bounds(n, nch, c):
    """Chunk c of n entries split into nch contiguous chunks."""
    return n * c // nch, n * (c + 1) // nch


# ------------------------------------------------------------- pixel BFS
#
# ch03 on the CPU: a level-synchronous BFS. The frontier is an array of
# entries (z << 8) | stamp, z = x * height + y. A level splits it into
# nch contiguous chunks; chunk c (one prange iteration, so one thread)
# expands its slice and writes what it discovers into its own region of
# `raw`, [K * a, K * b) for its slice [a, b) (K = 4 or 8 neighbours, so
# the region cannot overflow). After the prange (the barrier) the
# regions are concatenated into the next frontier, in parallel, at
# offsets from a serial prefix sum over the nch counts.
#
# The visited flag is one uint8 per pixel, 0 = not reached, written
# WITHOUT atomics: chunk c stores its stamp c + 1 (1..255) when it
# claims a pixel. Why the fill set is still exact:
#
# (1) Nothing outside the blob is reached. A pixel is flagged only by
#     the expansion of a frontier pixel next to it, and only if it is
#     red; by induction from the seed, every flagged pixel is in the
#     seed's component.
# (2) Nothing inside is missed. A flag goes from 0 to a stamp and never
#     back during the fill, and a one-byte store is one instruction, so
#     any read returns 0 or a stamp some chunk stored. The chunk that
#     stores a stamp appends the pixel to its own region in the same
#     iteration, and every region is copied into the next frontier
#     after the barrier. So when an expansion reads a non-zero flag on
#     a red neighbour, that neighbour is (or was) in some frontier
#     already. Every red neighbour of every expanded pixel is therefore
#     expanded at some level, and the component is closed under that.
# (3) Every pixel is expanded exactly once. Two chunks can both read 0
#     and both flag and append the same pixel: the race. After the
#     barrier its byte holds exactly one of their stamps (the last
#     store wins). A frontier entry is expanded only if the byte equals
#     its own stamp, so one copy is expanded and the others are dropped
#     (counted as duplicates). One chunk never appends a pixel twice:
#     its second read sees its own store. So a duplicate never claims
#     anything and duplicates cannot grow from level to level; the
#     frontier holds at most one copy per racing chunk.
# (4) Levels are exact. A pixel is flagged only while its first
#     neighbour is being expanded, all at one level L, so it enters the
#     frontier of level L + 1: the same partition into levels as the
#     sequential oracle's. The level count and the expanded count (the
#     fill count) equal the oracle's.
#
# Stamps are unique within a level (nch <= 255), and a pixel is flagged
# in one level only (the barrier makes the level's stores visible to
# the next), so stamps can be reused from level to level.

MAX_CHUNKS = 255

# neighbour orders as the oracles (ch01's 4-conn, shared/'s 8-conn)
DX4 = np.array([1, 0, -1, 0], dtype=np.int64)
DY4 = np.array([0, 1, 0, -1], dtype=np.int64)
DX8 = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
DY8 = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit(cache=True)
def _expand_pixels(img, visited, front, a, b, raw, base, stamp, conn8):
    """Expand frontier entries [a, b) as chunk `stamp` - 1: appends to
    raw[base:]. Returns (appended, expanded, dropped duplicates)."""
    width, height = img.shape[0], img.shape[1]
    nd = 8 if conn8 else 4
    st = np.uint8(stamp)
    cnt = 0
    expanded = 0
    dropped = 0
    for j in range(a, b):
        e = front[j]
        z = e >> 8
        x = z // height
        y = z - x * height
        if visited[x, y] != (e & 255):
            dropped += 1
            continue
        expanded += 1
        for d in range(nd):
            if conn8:
                nx = x + DX8[d]
                ny = y + DY8[d]
            else:
                nx = x + DX4[d]
                ny = y + DY4[d]
            if 0 <= nx < width and 0 <= ny < height and visited[nx, ny] == 0:
                if (img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0
                        and img[nx, ny, 2] == 0):
                    visited[nx, ny] = st
                    raw[base + cnt] = ((nx * height + ny) << 8) | stamp
                    cnt += 1
    return cnt, expanded, dropped


@njit(parallel=True, cache=True)
def pixel_bfs(img, visited, seed_x, seed_y, conn8, front, raw, min_chunk,
              serial_below, n_threads):
    """Level-synchronous BFS from (seed_x, seed_y), a red pixel; visited
    must be all 0. Levels with fewer than serial_below entries run on
    the calling thread (a prange costs a fork and a barrier); larger
    ones use up to 4 chunks per thread (n_threads: numba's current
    setting, passed in so the function can be cached), at least
    min_chunk entries each.

    Returns (levels, filled, duplicates, peak frontier, parallel levels,
    front, raw): front and raw may have been grown."""
    height = img.shape[1]
    k = 8 if conn8 else 4
    max_chunks = min(MAX_CHUNKS, 4 * max(n_threads, 1))
    counts = np.zeros(max_chunks, np.int64)
    expd = np.zeros(max_chunks, np.int64)
    drop = np.zeros(max_chunks, np.int64)
    offs = np.zeros(max_chunks + 1, np.int64)
    visited[seed_x, seed_y] = 1
    front[0] = ((seed_x * height + seed_y) << 8) | 1
    n = 1
    levels = 0
    filled = 0
    dups = 0
    peak = 0
    par_levels = 0
    while n > 0:
        levels += 1
        if n > peak:
            peak = n
        nch = n // max(min_chunk, 1)
        if nch > max_chunks:
            nch = max_chunks
        if n < serial_below or nch < 2:
            nch = 1
        if raw.size < k * n:
            raw = np.empty(2 * k * n, np.int64)
        if nch == 1:
            r = _expand_pixels(img, visited, front, 0, n, raw, 0, 1, conn8)
            counts[0] = r[0]
            expd[0] = r[1]
            drop[0] = r[2]
        else:
            par_levels += 1
            for c in prange(nch):
                a = n * c // nch
                b = n * (c + 1) // nch
                r = _expand_pixels(img, visited, front, a, b, raw, k * a,
                                   c + 1, conn8)
                counts[c] = r[0]
                expd[c] = r[1]
                drop[c] = r[2]
        total = 0
        for c in range(nch):
            offs[c] = total
            total += counts[c]
            filled += expd[c]
            dups += drop[c]
        if front.size < total:
            front = np.empty(2 * total, np.int64)
        if nch == 1:
            for j in range(total):
                front[j] = raw[j]
        else:
            for c in prange(nch):
                a = n * c // nch
                src = k * a
                dst = offs[c]
                for j in range(counts[c]):
                    front[dst + j] = raw[src + j]
        n = total
    return levels, filled, dups, peak, par_levels, front, raw


# ---------------------------------------------------------------- tiles

@njit(parallel=True, cache=True)
def pack_tiles(img, mask, wpr):
    """RGB -> 1 bit per pixel, tile-major: mask[tx * wpr + ty, x % 64]
    bit b <=> img[x, 64 * ty + b] is exactly (255, 0, 0). Rows in
    parallel: every word is written by the one thread that owns its row.
    The padding rows of the last tile row (x >= width) are zeroed."""
    width, height = img.shape[0], img.shape[1]
    for x in prange(width):
        base = (x >> 6) * wpr
        i = x & 63
        for ty in range(wpr):
            y0 = ty * TILE
            y1 = min(height, y0 + TILE)
            w = U0
            for y in range(y0, y1):
                r = (img[x, y, 0] == 255) & (img[x, y, 1] == 0) \
                    & (img[x, y, 2] == 0)
                w |= np.uint64(r) << np.uint64(y - y0)
            mask[base + ty, i] = w
    pad0 = width & 63
    if pad0 != 0:
        base = (width >> 6) * wpr
        for ty in range(wpr):
            for i in range(pad0, TILE):
                mask[base + ty, i] = U0


@njit(parallel=True, cache=True)
def tile_columns(mask, colm):
    """colm[t, 0] = the tile's west mask column (bit i = bit 0 of row i),
    colm[t, 1] its east column (bit 63). One tile per iteration."""
    for t in prange(mask.shape[0]):
        colm[t, 0] = col_low(mask[t], TILE)
        colm[t, 1] = col_high(mask[t], TILE)


@njit(parallel=True, cache=True)
def unpack_tiles(words, wpr, out):
    """out[x, y] = bit of tile-major words (bool (W, H)), rows in
    parallel (off the clock: the tests' and the checks' view)."""
    width, height = out.shape[0], out.shape[1]
    for x in prange(width):
        base = (x >> 6) * wpr
        i = x & 63
        for y in range(height):
            out[x, y] = ((words[base + (y >> 6), i]
                          >> np.uint64(y & 63)) & U1) != U0


@njit(parallel=True, cache=True)
def paint_tiles(img, reach, wpr):
    """Reached pixels become (0, 0, 255): rows in parallel, a full word
    painted in one stretch, a partial one bit by bit, an empty one
    skipped."""
    width = img.shape[0]
    for x in prange(width):
        base = (x >> 6) * wpr
        i = x & 63
        for ty in range(wpr):
            w = reach[base + ty, i]
            if w == U0:
                continue
            y0 = ty * TILE
            if w == ONES:
                for y in range(y0, y0 + TILE):
                    img[x, y, 0] = 0
                    img[x, y, 1] = 0
                    img[x, y, 2] = 255
            else:
                while w != U0:
                    y = y0 + lowest_bit(w)
                    w &= w - U1
                    img[x, y, 0] = 0
                    img[x, y, 1] = 0
                    img[x, y, 2] = 255


@njit(parallel=True, cache=True)
def popcount_words(words):
    """Set bits of a 2-D uint64 array (a parallel reduction)."""
    total = 0
    for t in prange(words.shape[0]):
        s = 0
        for i in range(words.shape[1]):
            s += popcount(words[t, i])
        total += s
    return total


# --------------------------------------------------------------- tile BFS
#
# ch07 on the CPU. A round is a prange over the frontier's tiles, cut
# into contiguous chunks. Visiting a tile: OR in what its neighbours
# sent, close it, send its borders. The barrier at the end of the
# prange ends the round; the next frontier is built serially from the
# chunks' lists of targets, deduplicated by a per-tile round stamp
# (frontiers of tiles are small: a ring of a few hundred tiles).
#
# NO RACE ON A BORDER. A tile's words (reach[t]) are read and written
# only by the thread visiting it, and a tile is visited at most once per
# round (the frontier is deduplicated). Neighbours talk only through
# inboxes: inbox[buf, t, d] holds the raw border of t's neighbour in
# direction d (its facing row word, column word or corner bit). Each
# (target, side) slot has exactly ONE writer, the neighbour on that
# side, and the inboxes are double-buffered by round parity: in round r
# a tile reads and clears inbox[r & 1] (its own slots only) and writes
# inbox[(r + 1) & 1] of its neighbours. So in one round a slot is either
# read by its owner or written by its one source, never both, and the
# barrier orders the write before the read a round later. A tile that
# writes a slot appends the target to its chunk's list, so every written
# slot is read (and cleared) in the next round.
#
# WHAT IS SENT. The raw border (the source's own reached bits), not the
# target's bits: the target spreads it (8-conn) and masks it itself.
# known[t, d] keeps the latest raw border each neighbour sent t, and
# sent[t, d] what t last sent each neighbour; both are owned by t. A
# tile sends to a neighbour only if (a) some red bit of the neighbour
# next to its reached border is not known to be reached there (known is
# a subset of the neighbour's reached bits, as they only grow), and
# (b) the border changed since the last send (the neighbour absorbed the
# last one in full). Without (a) every visit would bounce a full border
# back to where it came from; without (b) a stale `known` would make a
# tile resend the same border on every visit. Neither filter can drop a
# needed bit: a dropped candidate is either reached already (a) or was
# already delivered (b).
#
# WHY THE FILL IS EXACT. Every bit set is red and joined to a set bit
# (row fill, column step, inbox seeds), so nothing outside the blob is
# reached. The rounds end when no tile sent anything: then every tile is
# closed, and every red pixel next to a reached border pixel in another
# tile is reached (else the source would have sent). A set closed like
# that which holds the seed is the whole component.

# neighbour tile in direction d: N, S, W, E, NW, NE, SW, SE
DTX = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int64)
DTY = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int64)
OPP = np.array([1, 0, 3, 2, 7, 6, 5, 4], dtype=np.int64)
N_, S_, W_, E_, NW_, NE_, SW_, SE_ = 0, 1, 2, 3, 4, 5, 6, 7

# per-chunk counters of the tile BFS
TS_VISITS = 0       # tiles visited (frontier entries)
TS_IDLE = 1         # visits whose inboxes held no new bit (no closure)
TS_PASSES = 2       # closure passes, summed
TS_SENDS = 3        # inbox slots written
N_TS = 4


@njit(cache=True)
def _seed_row(f, m, i, s, dirty, lo, hi):
    """OR new bits s (a subset of m[i]) into row i, row-filled; returns
    (dirty, lo, hi)."""
    new = s & ~f[i]
    if new != U0:
        f[i] = row_fill(f[i] | new, m[i])
        dirty |= bit(i)
        if i < lo:
            lo = i
        if i > hi:
            hi = i
    return dirty, lo, hi


@njit(cache=True)
def _seed_column(f, m, s, b, dirty, lo, hi):
    """For every set bit i of s: bit b of row i (a red pixel) is new."""
    one = bit(b)
    while s != U0:
        i = lowest_bit(s)
        s &= s - U1
        dirty, lo, hi = _seed_row(f, m, i, one, dirty, lo, hi)
    return dirty, lo, hi


@njit(cache=True)
def _merge_inbox(t, f, m, colm, inbox, cur, known, conn8, dirty, lo, hi):
    """Read and clear tile t's inbox slots of buffer cur; seed the bits
    next to each neighbour's raw border. Returns (dirty, lo, hi)."""
    nd = 8 if conn8 else 4
    for d in range(nd):
        r = inbox[cur, t, d]
        if r == U0:
            continue
        inbox[cur, t, d] = U0
        known[t, d] |= r
        if d == N_:
            dirty, lo, hi = _seed_row(f, m, 0, spread(r, conn8) & m[0],
                                      dirty, lo, hi)
        elif d == S_:
            dirty, lo, hi = _seed_row(f, m, TILE - 1,
                                      spread(r, conn8) & m[TILE - 1],
                                      dirty, lo, hi)
        elif d == W_:
            dirty, lo, hi = _seed_column(f, m, spread(r, conn8) & colm[t, 0],
                                         0, dirty, lo, hi)
        elif d == E_:
            dirty, lo, hi = _seed_column(f, m, spread(r, conn8) & colm[t, 1],
                                         TILE - 1, dirty, lo, hi)
        elif d == NW_:
            dirty, lo, hi = _seed_row(f, m, 0, m[0] & U1, dirty, lo, hi)
        elif d == NE_:
            dirty, lo, hi = _seed_row(f, m, 0, m[0] & TOP, dirty, lo, hi)
        elif d == SW_:
            dirty, lo, hi = _seed_row(f, m, TILE - 1, m[TILE - 1] & U1,
                                      dirty, lo, hi)
        else:
            dirty, lo, hi = _seed_row(f, m, TILE - 1, m[TILE - 1] & TOP,
                                      dirty, lo, hi)
    return dirty, lo, hi


@njit(cache=True)
def _send_borders(t, tx, ty, tw, wpr, f, mask, colm, inbox, nxt, known,
                  sent, conn8, out, base):
    """Write t's raw borders into the neighbours' inboxes of buffer nxt
    (filters (a) and (b) above) and list the targets at out[base:].
    Returns the number of targets."""
    row0 = f[0]
    row63 = f[TILE - 1]
    cl = col_low(f, TILE)
    ch = col_high(f, TILE)
    nd = 8 if conn8 else 4
    cnt = 0
    for d in range(nd):
        ntx = tx + DTX[d]
        nty = ty + DTY[d]
        if ntx < 0 or ntx >= tw or nty < 0 or nty >= wpr:
            continue
        x = ntx * wpr + nty
        if d == N_:
            raw = row0
            cand = spread(row0, conn8) & mask[x, TILE - 1]
        elif d == S_:
            raw = row63
            cand = spread(row63, conn8) & mask[x, 0]
        elif d == W_:
            raw = cl
            cand = spread(cl, conn8) & colm[x, 1]
        elif d == E_:
            raw = ch
            cand = spread(ch, conn8) & colm[x, 0]
        elif d == NW_:
            raw = row0 & U1
            cand = raw & (mask[x, TILE - 1] >> S63)
        elif d == NE_:
            raw = row0 >> S63
            cand = raw & mask[x, TILE - 1]
        elif d == SW_:
            raw = row63 & U1
            cand = raw & (mask[x, 0] >> S63)
        else:
            raw = row63 >> S63
            cand = raw & mask[x, 0]
        if (cand & ~known[t, d]) == U0 or raw == sent[t, d]:
            continue
        sent[t, d] = raw
        inbox[nxt, x, OPP[d]] = raw
        out[base + cnt] = x
        cnt += 1
    return cnt


@njit(cache=True)
def _visit_tiles(front, a, b, out, base, tw, wpr, mask, colm, reach, inbox,
                 cur, known, sent, tdirty, tlo, thi, conn8, st, c):
    """Visit the frontier tiles [a, b) as chunk c: merge, close, send.
    Returns the number of targets listed at out[base:]."""
    cnt = 0
    for j in range(a, b):
        t = front[j]
        tx = t // wpr
        ty = t - tx * wpr
        f = reach[t]
        m = mask[t]
        dirty = tdirty[t]
        tdirty[t] = U0
        lo = np.int64(tlo[t])
        hi = np.int64(thi[t])
        dirty, lo, hi = _merge_inbox(t, f, m, colm, inbox, cur, known, conn8,
                                     dirty, lo, hi)
        st[c, TS_VISITS] += 1
        if dirty == U0:
            st[c, TS_IDLE] += 1
            continue
        passes, lo, hi = close_tile(f, m, conn8, dirty, lo, hi)
        tlo[t] = lo
        thi[t] = hi
        st[c, TS_PASSES] += passes
        k = _send_borders(t, tx, ty, tw, wpr, f, mask, colm, inbox, cur ^ 1,
                          known, sent, conn8, out, base + cnt)
        st[c, TS_SENDS] += k
        cnt += k
    return cnt


@njit(parallel=True, cache=True)
def reset_tile_state(reach, inbox, known, sent, tdirty, tlo, thi, queued):
    """Every per-tile buffer of the tile BFS back to its start, tiles in
    parallel (on the clock, as ch07's init kernel)."""
    for t in prange(reach.shape[0]):
        for i in range(TILE):
            reach[t, i] = U0
        for d in range(8):
            inbox[0, t, d] = U0
            inbox[1, t, d] = U0
            known[t, d] = U0
            sent[t, d] = U0
        tdirty[t] = U0
        tlo[t] = TILE
        thi[t] = -1
        queued[t] = 0


@njit(parallel=True, cache=True)
def tile_bfs(mask, colm, reach, inbox, known, sent, tdirty, tlo, thi,
             queued, front, out, conn8, tw, wpr, seed_x, seed_y,
             serial_below, n_threads):
    """Rounds of the tile BFS from (seed_x, seed_y), a red pixel, on
    reset state. A round with fewer than serial_below tiles runs on the
    calling thread. Returns (rounds, peak frontier, parallel rounds,
    counters summed over chunks: visits, idle visits, passes, sends)."""
    max_chunks = min(MAX_CHUNKS, 4 * max(n_threads, 1))
    counts = np.zeros(max_chunks, np.int64)
    st = np.zeros((max_chunks, N_TS), np.int64)
    t0 = (seed_x >> 6) * wpr + (seed_y >> 6)
    i0 = seed_x & 63
    reach[t0, i0] = row_fill(bit(seed_y & 63), mask[t0, i0])
    tdirty[t0] = bit(i0)
    tlo[t0] = i0
    thi[t0] = i0
    front[0] = t0
    queued[t0] = 1
    n = 1
    rounds = 0
    peak = 0
    par = 0
    while n > 0:
        rounds += 1
        if n > peak:
            peak = n
        cur = rounds & 1
        nch = min(max_chunks, n)
        if n < serial_below:
            nch = 1
        if nch == 1:
            counts[0] = _visit_tiles(front, 0, n, out, 0, tw, wpr, mask, colm,
                                     reach, inbox, cur, known, sent, tdirty,
                                     tlo, thi, conn8, st, 0)
        else:
            par += 1
            for c in prange(nch):
                a = n * c // nch
                b = n * (c + 1) // nch
                counts[c] = _visit_tiles(front, a, b, out, 8 * a, tw, wpr,
                                         mask, colm, reach, inbox, cur,
                                         known, sent, tdirty, tlo, thi,
                                         conn8, st, c)
        # the next frontier: the targets, each once (serial)
        m = 0
        nxt_stamp = rounds + 1
        for c in range(nch):
            base = 8 * (n * c // nch)
            for j in range(counts[c]):
                t = out[base + j]
                if queued[t] != nxt_stamp:
                    queued[t] = nxt_stamp
                    front[m] = t
                    m += 1
        n = m
    tot = np.zeros(N_TS, np.int64)
    for c in range(max_chunks):
        for k in range(N_TS):
            tot[k] += st[c, k]
    return rounds, peak, par, tot
