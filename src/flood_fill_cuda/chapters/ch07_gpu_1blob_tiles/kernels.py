"""Tile BFS: one blob, a frontier of 32x32 tiles closed in registers.

Chapter 3's best kernel (warp-coop, 8-conn) fills a 64 Mpx square in
88.5 ms, and its bill is structural: 4,001 BFS levels, two grid.sync()
each, and 3.16 visited-CAS attempts per filled pixel. This chapter keeps
the BFS shape - a frontier spreading out from one seed, one grid-wide
barrier pair per step - but changes what one step is. A step is no
longer "every frontier pixel probes its neighbours"; it is "every
frontier TILE closes itself completely, in registers, then tells its
neighbours what crossed the border".

THE TILE
A tile is 32 consecutive rows (x0 .. x0+31) times one mask word (32
consecutive y). One warp owns one tile per visit, and lane i holds row
x0+i as two uint32 registers:

    m   the packed red mask word   (ch06's pack_kernel; bit b = y 32*ty+b)
    f   the reached word           (a subset of m, always)

Rows past the image (x0+i >= width) hold m = f = 0, so every lane runs
every instruction and the warp intrinsics stay converged.

CLOSURE (no shared or global memory inside the loop)
    row fill      along y inside the word: Kogge-Stone occluded fill,
                  both directions (five shift steps each)
    diagonal      8-connectivity only: one step to rows x-1 and x+1 of
                  the dilated word f | f<<1 | f>>1
    column fill   along x across the 32 lanes: the same Kogge-Stone,
                  with shfl_up/shfl_down by 1, 2, 4, 8, 16
repeated while any lane's f changed (any_sync). A solid tile closes in
two iterations (one that fills, one that proves nothing changed).

EXPORTS
After closure the warp ORs its border bits into the neighbouring tiles'
reached words. Every lane owns at most three targets: its own row's
word ty-1 (bit 0 -> bit 31) and word ty+1 (bit 31 -> bit 0), and lanes
0-2 / 29-31 one word each of the side rows x0-1 / x0+32 (word ty, and
for 8-connectivity the corner words ty-1 and ty+1, fed from lane 0's /
lane 31's word by a broadcast shuffle). For 8-connectivity the bits are
dilated first (rows x-1 and x+1 for the vertical exports, y-1 and y+1
for the side ones). Only bits that are red in the target and not yet
reached are sent: plain loads filter first, the atomic OR runs only on
what survives, and a target whose atomic actually added a bit is
enqueued for the next round. The targets do not depend on the closure,
so their filter words are loaded together with the tile's own, before
it. The owner writes its own word back with an atomic OR too, never a
plain store: a neighbour may be OR-ing into the same word in the same
round, and a store would erase those bits.

ROUNDS
The ch03 skeleton, unchanged: a global queue (of tiles now), the
warp-aggregated enqueue copied from ch03, a cooperative launch, a
grid-stride over the round's entries (one warp per entry), and two
grid.sync() per round. A per-tile stamp, raised with atomic.max, keeps a
tile out of a round's queue after its first enqueue, so ping-pong queues
of n_tiles entries can never overflow.

SWITCHES (each off by default in get_rounds_kernel, each an ablation;
flood_fill.DEFAULT_SWITCHES turns on one_sync and fused_pack; the
correctness arguments are in _make_rounds_kernel's docstring)
    one_sync     one grid.sync per round: three rotating queue counters
    fused_pack   no pack launch: a tile's RGB is read the round it is
                 first enqueued, so tiles the wave never reaches are never
                 read; paint is then paint_tiles_kernel
    skip_full    exports into a full tile already queued are dropped

WHAT IS GIVEN UP
There is no per-pixel depth: a round is a step of the TILE graph, and a
pixel's round says nothing exact about its geodesic distance. The
counters keep rounds, tile visits, closure iterations and the atomics
issued; the fill set and the painted image are exact.
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda, int32, int64, uint32, uint64

# Re-used verbatim from Chapter 6: RGB -> 1 bit/px. Imported, not copied,
# so the mask layout cannot drift between the two chapters.
from ..ch06_gpu_nblob_runs.kernels import pack_kernel  # noqa: F401

# Every bit value and every constant is uint32. Numba promotes a
# uint32 operand mixed with an int64 literal to int64 (and uint64 with
# int64 to float64), so no bare Python int is ever combined with a word.
FULL = np.uint32(0xFFFFFFFF)
U0 = np.uint32(0)
U1 = np.uint32(1)
S1 = np.uint32(1)
S2 = np.uint32(2)
S4 = np.uint32(4)
S8 = np.uint32(8)
S16 = np.uint32(16)
S31 = np.uint32(31)

S3 = np.uint32(3)
# uint64 constants: Numba turns uint64 mixed with an int64 into float64
Q0 = np.uint64(0)
Q1 = np.uint64(1)
Q2 = np.uint64(2)
Q3 = np.uint64(3)
Q32 = np.uint64(32)
S12 = np.uint32(12)
S20 = np.uint32(20)
S24 = np.uint32(24)
S28 = np.uint32(28)
NIB = np.uint32(0xF)

TILE = 32
# A queue entry packs a tile as (tx << 16) | ty, so decoding is two
# shifts and a mask instead of a 64-bit division by words_per_row.
TY_BITS = 16
TY_MASK = (1 << TY_BITS) - 1
# tx < 2**15 (flood_fill.MAX_TILES_X), so bit 31 of an entry is free. It
# marks a tile's FIRST entry ever: the seed's (set by the host for the
# fused-pack and skip-full kernels), and in the skip-full kernel without
# the fused pack every first enqueue - the exporter whose atomic.max on
# the stamp returned -1 is the only one that can see -1, and the round it
# enqueues for holds the tile's first visit.
TX_MASK = (1 << (31 - TY_BITS)) - 1
FIRST_ENTRY = -(1 << 31)

# Per-tile info word of the fused-pack kernel: 2 * (round in which the
# tile was packed) + (1 if every in-image pixel of the tile is red, set
# only by the skip-full kernel). Before the pack it holds NOT_PACKED,
# which is even (no "full" bit) and larger than 2 * any round, so
# "packed before round r" is ti < 2 * r. The skip-full kernel without the
# fused pack uses only bit 0 (init 0).
NOT_PACKED = 0x7FFFFFFE

# Slots in the int64 counters array
ROUNDS = 0          # tile-BFS rounds (one or two grid.sync each)
TILE_VISITS = 1     # warps that closed a tile (= tiles ever enqueued + 1)
CLOSURE_ITERS = 2   # closure loop trips, summed over visits
OR_ATOMICS = 3      # atomic.or_ on `reached` (own word + exports)
STAMP_ATOMICS = 4   # atomic.max on the per-tile stamp (enqueue requests)
PEAK_ROUND = 5      # largest round (tiles)
OVERFLOW = 6        # defensive tripwire: structurally unreachable
PACKED_TILES = 7    # fused pack: tiles whose RGB the rounds kernel read
SKIPPED_EXPORTS = 8  # skip-full: export words dropped into full tiles
NUM_COUNTERS = 9

# Red (255, 0, 0) as 12 bytes = 4 pixels, read as three little-endian
# words: FF 00 00 FF | 00 00 FF 00 | 00 FF 00 00.
RED_W0 = np.uint32(0xFF0000FF)
RED_W1 = np.uint32(0x00FF0000)
RED_W2 = np.uint32(0x0000FF00)
# the bytes of pixel 0..3 inside the three XOR-ed words
PX0_W0 = np.uint32(0x00FFFFFF)
PX1_W0 = np.uint32(0xFF000000)
PX1_W1 = np.uint32(0x0000FFFF)
PX2_W1 = np.uint32(0xFFFF0000)
PX2_W2 = np.uint32(0x000000FF)
PX3_W2 = np.uint32(0xFFFFFF00)
# Blue (0, 0, 255), 4 pixels: 00 00 FF 00 | 00 FF 00 00 | FF 00 00 FF
BLUE_W0 = np.uint32(0x00FF0000)
BLUE_W1 = np.uint32(0x0000FF00)
BLUE_W2 = np.uint32(0xFF0000FF)


# --------------------------------------------------------------- bit helpers

@cuda.jit(device=True, inline=True)
def _up(v, k):
    """shfl_up of a uint32 word: lane i gets lane i-k's word.

    Numba's shuffle intrinsic has no uint32 overload (int32, int64,
    float32 and float64 only), and a uint32 argument converts SAFELY to
    int64 - so a plain call shuffles a 64-bit value (two SHFLs) and
    hands back an int64 that drags the rest of the arithmetic to 64
    bits. Shuffling the int32 bit pattern and casting back is one SHFL
    and stays 32-bit.
    """
    return uint32(cuda.shfl_up_sync(FULL, int32(v), k))


@cuda.jit(device=True, inline=True)
def _down(v, k):
    """shfl_down of a uint32 word (see _up): lane i gets lane i+k's."""
    return uint32(cuda.shfl_down_sync(FULL, int32(v), k))


@cuda.jit(device=True, inline=True)
def _row_fill(f, m):
    """Close f inside m along the word (y), both directions.

    Kogge-Stone occluded fill: after step k, `p` holds the bits whose
    2k-wide window toward bit 0 is all red, so `g |= p & (g << k)` moves
    every reached bit k places through red pixels only. g starts as f, a
    subset of m, and only ever gains bits of p (a subset of m), so the
    result stays inside m.
    """
    g = f
    p = m
    g = uint32(g | (p & (g << S1)))
    p = uint32(p & (p << S1))
    g = uint32(g | (p & (g << S2)))
    p = uint32(p & (p << S2))
    g = uint32(g | (p & (g << S4)))
    p = uint32(p & (p << S4))
    g = uint32(g | (p & (g << S8)))
    p = uint32(p & (p << S8))
    g = uint32(g | (p & (g << S16)))

    h = f
    q = m
    h = uint32(h | (q & (h >> S1)))
    q = uint32(q & (q >> S1))
    h = uint32(h | (q & (h >> S2)))
    q = uint32(q & (q >> S2))
    h = uint32(h | (q & (h >> S4)))
    q = uint32(q & (q >> S4))
    h = uint32(h | (q & (h >> S8)))
    q = uint32(q & (q >> S8))
    h = uint32(h | (q & (h >> S16)))
    return uint32(g | h)


@cuda.jit(device=True, inline=True)
def _col_step_up(f, p, lane, k):
    """One Kogge-Stone step from lane-k to lane (toward higher lanes)."""
    fu = _up(f, k)
    pu = _up(p, k)
    if lane >= k:
        f = uint32(f | (p & fu))
        p = uint32(p & pu)
    return f, p


@cuda.jit(device=True, inline=True)
def _col_step_down(f, p, lane, k):
    """One Kogge-Stone step from lane+k to lane (toward lower lanes)."""
    fd = _down(f, k)
    pd = _down(p, k)
    if lane + k < 32:
        f = uint32(f | (p & fd))
        p = uint32(p & pd)
    return f, p


@cuda.jit(device=True, inline=True)
def _col_fill(f, m, lane):
    """Close f inside m along x (across the warp's 32 lanes), both ways.

    Every lane must call it (shuffles). Lanes below k receive nothing
    from above, which is the tile's top edge; the windows they leave
    unclamped are never used, because lane j only reads its own `p` at a
    step k with j >= k, and that window lies entirely inside the tile.

    The syncwarp() is a 15x speed-up, measured: without it this function
    costs 4.4 us per call from a lone warp, with it 0.29 us (same
    results). Inside a loop ptxas cannot prove the warp converged, so it
    guards every shuffle with a runtime convergence test and a slow path
    (a call to __cuda_sm70_shflsync_up_p); after the loop back-edge the
    warp tests as diverged and all twenty shuffles take the slow path.
    An explicit syncwarp() reconverges it, and the fast SHFL runs.
    """
    cuda.syncwarp()
    p = m
    f, p = _col_step_up(f, p, lane, 1)
    f, p = _col_step_up(f, p, lane, 2)
    f, p = _col_step_up(f, p, lane, 4)
    f, p = _col_step_up(f, p, lane, 8)
    f, p = _col_step_up(f, p, lane, 16)
    p = m
    f, p = _col_step_down(f, p, lane, 1)
    f, p = _col_step_down(f, p, lane, 2)
    f, p = _col_step_down(f, p, lane, 4)
    f, p = _col_step_down(f, p, lane, 8)
    f, p = _col_step_down(f, p, lane, 16)
    return f


@cuda.jit(device=True, inline=True)
def _dilate(f):
    """f plus its two in-word y neighbours (the bits that cross the word
    edge belong to the next word and are exported, not kept)."""
    return uint32(f | (f << S1) | (f >> S1))


@cuda.jit(device=True, inline=True)
def _diag_step(f, m, lane):
    """One 8-connectivity step: rows x-1 and x+1 of the dilated word.

    Lane 0 gets nothing from above and lane 31 nothing from below - those
    rows belong to the neighbouring tiles and arrive as exports.
    (syncwarp: see _col_fill.)
    """
    cuda.syncwarp()
    d = _dilate(f)
    du = _up(d, 1)
    dd = _down(d, 1)
    if lane == 0:
        du = U0
    if lane == 31:
        dd = U0
    return uint32(f | (m & (du | dd)))


# ------------------------------------------------------------- enqueue

@cuda.jit(device=True, inline=True)
def _warp_enqueue_global(queues, off, cap, state, rear_slot, item, counters):
    """Warp-aggregated append on a global rear counter (one atomic per
    warp). Copied from ch03's helper of the same name; the only change is
    that the queue is the slice [off, off + cap) of a flat ping-pong array.

    The bound check is a defensive tripwire only: the stamp lets a tile
    into a round's queue at most once, so a round holds <= cap = n_tiles.
    """
    mask = cuda.activemask()
    count = cuda.popc(mask)
    rank = cuda.popc(mask & cuda.lanemask_lt())
    leader = cuda.ffs(mask) - 1  # ffs is 1-based; mask always has a bit set
    base = 0
    if cuda.laneid == leader:
        base = cuda.atomic.add(state, rear_slot, count)
    base = cuda.shfl_sync(mask, base, leader)
    idx = base + rank
    if idx < cap:
        queues[uint32(off + idx)] = item
    else:
        counters[OVERFLOW] = 1  # unreachable by the stamp argument


@cuda.jit(device=True, inline=True)
def _bcast(v, src_lane):
    """Lane src_lane's uint32 word, to every lane (int32 trick of _up)."""
    return uint32(cuda.shfl_sync(FULL, int32(v), src_lane))


# ------------------------------------------------------- fused pack

@cuda.jit(device=True, inline=True)
def _red4(w0, w1, w2):
    """Four pixels from 12 bytes held as three little-endian words, as a
    nibble (bit j = pixel j is exactly (255, 0, 0)). XOR with the red
    pattern zeroes a red pixel's three bytes."""
    d0 = uint32(w0 ^ RED_W0)
    d1 = uint32(w1 ^ RED_W1)
    d2 = uint32(w2 ^ RED_W2)
    return uint32(uint32((d0 & PX0_W0) == U0)
                  | (uint32(uint32((d0 & PX1_W0) | (d1 & PX1_W1)) == U0)
                     << S1)
                  | (uint32(uint32((d1 & PX2_W1) | (d2 & PX2_W2)) == U0)
                     << S2)
                  | (uint32((d2 & PX3_W2) == U0) << S3))


@cuda.jit(device=True, inline=True)
def _funnel(lo, hi, sh):
    """The word starting sh bits into lo (sh in 0, 8, 16, 24)."""
    return uint32(((uint64(hi) << Q32) | uint64(lo)) >> sh)


@cuda.jit(device=True, inline=True)
def _ld_word(img32, i, n32, n32m1, tail):
    """img32[i] for i < n32; the image's last, partial word (`tail`, built
    from the 0-3 bytes past 4 * n32) for i >= n32. Bytes past the image
    are only ever pixels past `height`, which the caller masks. All
    uint64 (n32m1 = max(n32 - 1, 0); with n32 = 0, img32 is a 1-element
    dummy and every word is the tail)."""
    v = img32[min(i, n32m1)]
    return v if i < n32 else tail


@cuda.jit(device=True, inline=True)
def _ld4(img32, p, n32, n32m1, tail):
    """The four words p .. p+3, each through _ld_word."""
    return (_ld_word(img32, p, n32, n32m1, tail),
            _ld_word(img32, p + Q1, n32, n32m1, tail),
            _ld_word(img32, p + Q2, n32, n32m1, tail),
            _ld_word(img32, p + Q3, n32, n32m1, tail))


@cuda.jit(device=True, inline=True)
def _ok(x, y, width, height):
    """Group (x, y .. y+3) starts inside the image."""
    return x < width and y < height


@cuda.jit(device=True, inline=True)
def _wi(x, y, width, height, h34):
    """uint32-word index of pixel (x, y) when rows are 4-byte aligned
    (height % 4 == 0 and y % 4 == 0: (x * height + y) * 3 / 4 is exact),
    0 for a group outside the image (its load still runs and its nibble
    is dropped: a guarded load cannot be hoisted above its guard).
    Unsigned, so Numba emits no negative-index wraparound."""
    if x < width and y < height:
        return uint64(x) * h34 + uint64(3 * (y >> 2))
    return Q0


@cuda.jit(device=True, inline=True)
def _bi(x, y, width, height):
    """Byte offset of pixel (x, y), 0 for a group outside the image."""
    if x < width and y < height:
        return (uint64(x) * uint64(height) + uint64(y)) * Q3
    return Q0


@cuda.jit(device=True, inline=True)
def _nib_a(w0, w1, w2, ok):
    """Aligned group nibble, dropped outside the image (aligned rows: a
    group is all in or all out)."""
    return _red4(w0, w1, w2) if ok else U0


@cuda.jit(device=True, inline=True)
def _nib_u(w0, w1, w2, w3, b, y, height, ok):
    """Unaligned group nibble from the four words around byte b: funnel
    shift, then drop pixels past `height` (the next row's bytes)."""
    sh = (b & Q3) << Q3
    n = _red4(_funnel(w0, w1, sh), _funnel(w1, w2, sh), _funnel(w2, w3, sh))
    keep = uint32((U1 << uint32(min(4, height - y))) - U1) if ok else U0
    return uint32(n & keep)


@cuda.jit(device=True, inline=True)
def _gather_nib(nibs, src, sh, q):
    """Nibble k (bits sh..sh+3) of lane src's word, moved to bits 4q.."""
    return uint32(((_bcast(nibs, src) >> sh) & NIB) << uint32(4 * q))


@cuda.jit(device=True, inline=True)
def _gather_rows(nibs, lane):
    """Lane L holds batch k's nibble at bits 4k (row 4k + (L >> 3), pixels
    4q .. 4q+3 with q = L & 7). Row i = 4k + j was read by lanes 8j ..
    8j+7, so eight shuffles from lanes 8j + q gather its word."""
    cuda.syncwarp()  # fast-path shuffles: see _col_fill
    src = (lane & 3) << 3
    sh = uint32((lane >> 2) << 2)
    m = _gather_nib(nibs, src, sh, 0)
    m = uint32(m | _gather_nib(nibs, src + 1, sh, 1))
    m = uint32(m | _gather_nib(nibs, src + 2, sh, 2))
    m = uint32(m | _gather_nib(nibs, src + 3, sh, 3))
    m = uint32(m | _gather_nib(nibs, src + 4, sh, 4))
    m = uint32(m | _gather_nib(nibs, src + 5, sh, 5))
    m = uint32(m | _gather_nib(nibs, src + 6, sh, 6))
    m = uint32(m | _gather_nib(nibs, src + 7, sh, 7))
    return m


@cuda.jit(device=True, inline=True)
def _pack_tile(img32, aligned, x0, ty, width, height, h34, n32, n32m1,
               tail, lane):
    """RGB -> this lane's mask word (row x0 + lane, word ty), the same word
    ch06's pack_kernel writes. Every lane must call it (shuffles).

    The tile is 32 rows x 96 bytes. Batch k (0..7) covers rows 4k..4k+3:
    lane L reads 12 contiguous bytes, pixels 32ty + 4q .. +3 (q = L & 7)
    of row 4k + (L >> 3), so one load instruction touches 4 row segments.
    (The plan's layout, lane i reading its own row's 96 bytes, touches 32
    rows per instruction; alone at the rounds kernel's grid it packed
    sq_8000 in 1.78 ms against 0.90 for this one.)

    Every load runs unguarded (outside the image it reads word 0 and its
    nibble is dropped), and all of them come before a syncwarp(): no load
    moves across a warp barrier, so all 24 words (32 when rows are not
    4-byte aligned) are in flight together and the pack costs one
    dependent memory trip. Left to itself, ptxas consumed each batch's
    loads before issuing the next batch's (up to 8 trips per pack).
    _gather_rows then turns the eight nibbles into row words."""
    q = lane & 7
    x = x0 + (lane >> 3)
    y = ty * 32 + 4 * q
    nibs = U0
    if aligned:                                     # warp-uniform
        i0 = _wi(x, y, width, height, h34)
        i1 = _wi(x + 4, y, width, height, h34)
        i2 = _wi(x + 8, y, width, height, h34)
        i3 = _wi(x + 12, y, width, height, h34)
        i4 = _wi(x + 16, y, width, height, h34)
        i5 = _wi(x + 20, y, width, height, h34)
        i6 = _wi(x + 24, y, width, height, h34)
        i7 = _wi(x + 28, y, width, height, h34)
        a0, b0, c0 = img32[i0], img32[i0 + Q1], img32[i0 + Q2]
        a1, b1, c1 = img32[i1], img32[i1 + Q1], img32[i1 + Q2]
        a2, b2, c2 = img32[i2], img32[i2 + Q1], img32[i2 + Q2]
        a3, b3, c3 = img32[i3], img32[i3 + Q1], img32[i3 + Q2]
        a4, b4, c4 = img32[i4], img32[i4 + Q1], img32[i4 + Q2]
        a5, b5, c5 = img32[i5], img32[i5 + Q1], img32[i5 + Q2]
        a6, b6, c6 = img32[i6], img32[i6 + Q1], img32[i6 + Q2]
        a7, b7, c7 = img32[i7], img32[i7 + Q1], img32[i7 + Q2]
        cuda.syncwarp()   # every load above is issued before any use
        nibs = uint32(
            _nib_a(a0, b0, c0, _ok(x, y, width, height))
            | (_nib_a(a1, b1, c1, _ok(x + 4, y, width, height)) << S4)
            | (_nib_a(a2, b2, c2, _ok(x + 8, y, width, height)) << S8)
            | (_nib_a(a3, b3, c3, _ok(x + 12, y, width, height)) << S12)
            | (_nib_a(a4, b4, c4, _ok(x + 16, y, width, height)) << S16)
            | (_nib_a(a5, b5, c5, _ok(x + 20, y, width, height)) << S20)
            | (_nib_a(a6, b6, c6, _ok(x + 24, y, width, height)) << S24)
            | (_nib_a(a7, b7, c7, _ok(x + 28, y, width, height)) << S28))
    else:
        o0 = _bi(x, y, width, height)
        o1 = _bi(x + 4, y, width, height)
        o2 = _bi(x + 8, y, width, height)
        o3 = _bi(x + 12, y, width, height)
        o4 = _bi(x + 16, y, width, height)
        o5 = _bi(x + 20, y, width, height)
        o6 = _bi(x + 24, y, width, height)
        o7 = _bi(x + 28, y, width, height)
        # four words from byte o >> 2. In the image's last rows a group
        # whose pixels run past `height` reads past the image's end, so
        # every one of them may be the partial last word (or beyond it):
        # _ld_word clamps each (tiny images have no whole word at all)
        p0, p1, p2, p3 = o0 >> Q2, o1 >> Q2, o2 >> Q2, o3 >> Q2
        p4, p5, p6, p7 = o4 >> Q2, o5 >> Q2, o6 >> Q2, o7 >> Q2
        a0, b0, c0, d0 = _ld4(img32, p0, n32, n32m1, tail)
        a1, b1, c1, d1 = _ld4(img32, p1, n32, n32m1, tail)
        a2, b2, c2, d2 = _ld4(img32, p2, n32, n32m1, tail)
        a3, b3, c3, d3 = _ld4(img32, p3, n32, n32m1, tail)
        a4, b4, c4, d4 = _ld4(img32, p4, n32, n32m1, tail)
        a5, b5, c5, d5 = _ld4(img32, p5, n32, n32m1, tail)
        a6, b6, c6, d6 = _ld4(img32, p6, n32, n32m1, tail)
        a7, b7, c7, d7 = _ld4(img32, p7, n32, n32m1, tail)
        cuda.syncwarp()   # every load above is issued before any use
        nibs = uint32(
            _nib_u(a0, b0, c0, d0, o0, y, height, _ok(x, y, width, height))
            | (_nib_u(a1, b1, c1, d1, o1, y, height,
                      _ok(x + 4, y, width, height)) << S4)
            | (_nib_u(a2, b2, c2, d2, o2, y, height,
                      _ok(x + 8, y, width, height)) << S8)
            | (_nib_u(a3, b3, c3, d3, o3, y, height,
                      _ok(x + 12, y, width, height)) << S12)
            | (_nib_u(a4, b4, c4, d4, o4, y, height,
                      _ok(x + 16, y, width, height)) << S16)
            | (_nib_u(a5, b5, c5, d5, o5, y, height,
                      _ok(x + 20, y, width, height)) << S20)
            | (_nib_u(a6, b6, c6, d6, o6, y, height,
                      _ok(x + 24, y, width, height)) << S24)
            | (_nib_u(a7, b7, c7, d7, o7, y, height,
                      _ok(x + 28, y, width, height)) << S28))
    return _gather_rows(nibs, lane)


@cuda.jit(device=True, inline=True)
def _pack_store(mask, tinfo, img32, aligned, tx, ty, width, height, wpr,
                h34, n32, n32m1, tail, last_full, two_r, with_full, lane):
    """Pack tile (tx, ty), store its mask words and then its info word
    2r + full (full only when with_full: every in-image pixel red).
    Every lane must call it. Returns this lane's mask word."""
    x0 = tx * TILE
    m = _pack_tile(img32, aligned, x0, ty, width, height, h34, n32, n32m1,
                   tail, lane)
    x = x0 + lane
    if x < width:
        mask[uint32(x * wpr + ty)] = m
    tval = two_r
    if with_full:
        exp = last_full if ty + 1 == wpr else FULL
        if cuda.all_sync(FULL, x >= width or m == exp):
            tval = int32(two_r + 1)
    if lane == 0:
        tinfo[uint32(tx * wpr + ty)] = tval
    return m


@cuda.jit(device=True, inline=True)
def _paint_rows(img8, img32, aligned, x0, ty, height, h34, bits, lane,
                blue):
    """Paint the set bits of a tile blue, one row at a time: lane i holds
    row x0+i's word `bits` (inside the image: a subset of the mask). Each
    row is the 96 contiguous bytes of 32 lanes: 24 word stores of the
    blue pattern (`blue`, by lane % 3) when the whole word is set and
    rows are 4-byte aligned, byte stores otherwise. Every lane must call
    it (shuffles)."""
    cnt = min(TILE, height - ty * TILE)      # pixels in this word
    whole = FULL
    if cnt < TILE:
        whole = uint32((U1 << uint32(cnt)) - U1)
    n_words = (3 * cnt) >> 2                 # aligned: exact
    y = ty * TILE + lane
    cuda.syncwarp()
    rows = cuda.ballot_sync(FULL, bits != U0)
    while rows != U0:                                    # warp-uniform
        i = cuda.ffs(rows) - 1
        rows = uint32(rows & (rows - U1))
        cuda.syncwarp()  # fast-path shuffle: see _col_fill
        b = _bcast(bits, i)
        if aligned and b == whole:                       # warp-uniform
            if lane < n_words:
                img32[uint64(x0 + i) * h34 + uint64(24 * ty + lane)] = blue
        elif (b >> uint32(lane)) & U1:
            bi = (uint64(x0 + i) * uint64(height) + uint64(y)) * Q3
            img8[bi] = 0
            img8[bi + Q1] = 0
            img8[bi + Q2] = 255


# --------------------------------------------------------------- kernels

def _make_rounds_kernel(conn8, instrumented, max_registers=None,
                        one_sync=False, fused_pack=False, skip_full=False,
                        fused_paint=False):
    """Build the cooperative rounds kernel for one connectivity.

    Every flag is a compile-time constant (a closure variable), so the
    4-conn kernel carries no diagonal code, the bare twin no counters,
    and with every switch off the kernel runs the chapter's original
    algorithm - the twin discipline of ch03, without hand-copied bodies.
    (Not the original binary: the shared code was reorganised for the
    switches, and the linked 8-conn kernel now takes 145 registers, 135
    bare, against the 143 and 129 below.)

    Registers, measured. Every array here is flat and indexed with a
    uint32 (Numba then emits no negative-index wraparound), the side and
    corner exports are spread over lanes 0-2 and 29-31 so each lane owns
    at most three targets, and the counters are uint32. The LINKED 8-conn
    kernel still reports 143 registers/thread: ptxas compiles the same
    PTX on its own (`-c`, nothing linked) to 87 with zero spills, and the
    other ~56 arrive with linking the external grid.sync() call from
    libcudadevrt. A link-time cap (`max_registers`) trades them for
    spills and more resident warps; it is a parameter so the benchmark
    can price that trade. The fused-pack kernels link at 242-249
    registers (ptxas alone: 144), so they seat 48 x 128 = 192 warps, not
    288. Caps, in one interleaved A/B: 168 (288 warps, 32 bytes of
    spills) tied the disk and sq_4000c and lost 6-7% on the serpentines
    and 37% on sq_8000; 128 (384 warps, 152 bytes of spills) won 9% on
    sq_8000 but lost 10-12% on sq_4000c and the disk and 25-90% on the
    serpentines. So no cap.

    ONE grid.sync PER ROUND (one_sync). Three queues and three counters
    rotate. In round r the current queue is c = r % 3, appends go to
    n = (r+1) % 3, and thread 0 resets the counter of z = (r+2) % 3
    before the round's only barrier B_r. After B_r every thread reads
    q_state[n], the size of round r+1. Follow one counter k through its
    cycle (rounds r, r+1, r+2 with k = (r+1) % 3):
        round r     appends to k (atomic add)           - before B_r
        round r+1   every thread reads k, once, right after B_r; the
                    warps read queue k's entries         - before B_r+1
        round r+2   thread 0 stores k = 0                 - before B_r+2
        round r+3   appends to k again                    - after B_r+2
    - Every read of k (round r+1) happens before its reader arrives at
      B_r+1, and the reset happens after thread 0 leaves B_r+1, so no
      reader can see the reset.
    - The reset is thread 0's store before B_r+2, and every append of
      round r+3 is issued after its thread leaves B_r+2. grid.sync() is
      a grid-wide fence plus barrier, so no thread can append before the
      reset is visible, and no append is overwritten by it.
    - Queue k's entries are read in round r+1 and rewritten from round
      r+3 on: two barriers apart.
    - Between B_r and B_r+1 nobody writes k (its appends ended at B_r,
      its reset waits for round r+2), so every thread reads the same
      n_cur and takes the same branch of `while n_cur > 0`. Round 0's
      n_cur = 1 is a constant. Hence every thread runs the same number
      of rounds and of barriers, and grid.sync() cannot deadlock.
    With one_sync off, the original ritual runs: two queues, and B_r is
    two barriers with the read and thread 0's reset between them.

    FUSED PACK (fused_pack). There is no pack launch: the rounds kernel
    reads a tile's RGB (_pack_tile) in the round the tile is first
    enqueued, one round before its first visit, so tiles the wave never
    reaches are never read. The packer is the warp whose atomic.max on
    the tile's stamp returned -1 (exactly one warp can see -1); it packs
    after its own visit's exports, stores the mask words and then
    tinfo[t] = 2r (+ the full bit). The seed's tile, which nobody
    enqueues, is packed by its own visit in round 0 (FIRST entry).
    (The first version packed in the owner's first visit instead. That
    put the pack between the owner's read of reached and its write-back,
    the window in which same-round exports re-enqueue a tile, and visits
    on sq_8000 rose from 98K to 104K; packed ahead, they fall to 99.8K,
    and the next round's exporters see the tile's mask and full bit.)
    `reached` may now hold bits that are not red ("candidates"): an
    export into a tile whose mask is not known yet cannot be filtered.
    The protocol, for an exporter E in round r with target tile T:
        E loads tinfo[T] with its filter words (before its closure).
        tinfo[T] < 2r   T was packed in an earlier round: its mask words
                        were stored before that round's barrier, so E's
                        loads of them (all after B_r-1) see the true
                        mask. Filter as always: mask & ~reached.
        otherwise       T is unpacked, or being packed in this very
                        round (tinfo is r's value or not visible yet):
                        send the border bits filtered by ~reached only.
    The owner masks on every visit: f0 = reached & m, the closure runs
    inside m, and the write-back ORs f (a subset of m). Paint paints
    reached & mask (paint_tiles_kernel), and the host's fill set is the
    same AND. A tile is always packed before its first visit: it is
    packed in the round it is first enqueued (or, the seed, by that
    visit itself), and visited in the next. Why the fill is exact:
    - Nothing non-red is painted: paint ANDs with the mask the packer
      built from the image (a tile's mask words are written once, by its
      packer; nothing writes RGB before paint).
    - Nothing outside the component is painted: every bit of reached is
      the seed, a bit of some f (closed from f0 inside m), or a border
      neighbour of a bit of some f. ANDed with the mask, each is a red
      pixel joined to the seed.
    - No bit is lost. An export whose atomic adds a bit to reached
      enqueues T (or finds the stamp at r+1 already), so T is visited in
      round r+1 and reads the bit after B_r: if it is red it is in f0.
      An export that adds nothing found the bit already set by someone
      who was responsible for it by the same rule (an exporter that
      enqueued, or an owner whose closure contained it and exported its
      border). The race "E sees T unpacked while T is being packed in
      round r": E's bits go to reached unfiltered, and whether or not
      they land before the packer stores the mask, T's visit (round r+1
      at the earliest, after B_r) masks them with the finished mask. E
      never uses a mask word unless tinfo[T] says it was stored before
      this round's barrier.
    - Candidates cost work, not correctness: a candidate in a white tile
      costs that tile one visit that closes nothing (from the round after
      its pack on, T's mask filters every export into it).

    SKIP EXPORTS INTO FULL TILES (skip_full). On solid blobs a third of
    the visits re-close a tile that was already closed: an export landed
    after the owner's read of reached in the same round, and re-enqueued
    the tile. Call T "full" when every in-image pixel of T is red.
    - A full tile's first visit fills all of it. That visit was enqueued
      by an atomic that added a bit to T's reached and stays visible to
      it: the bit itself if inside the image (red: T is full), or else -
      a bit past `height` comes only from a dilated side export - the
      red bit it was dilated from, sent in the same word or already
      reached. The visit starts from a red bit, and a full tile is one
      component at either connectivity.
    - So once T's first visit has run, or is queued for this round or
      the next, every export into T is redundant: its bits are inside T,
      which that visit fills (if it runs now or later) or has filled
      (then E's bits are reached already). E may drop it.
    - The full bit tells E exactly that. With fused_pack it is bit 0 of
      tinfo[T], stored when T is packed: T was then first enqueued, for
      the next round. Without fused_pack it is bit 0 of tinfo[T] set by
      T's owner during its first visit (FIRST entry), and for the up and
      down targets also the vote over the 32 target words the warp
      already loaded - the mask says nothing about a visit, so the vote
      needs T's stamp >= r as well (T queued for round r or r+1, where
      the first argument applies to that visit).
    A stale read of tinfo or of a stamp is an older value: no full bit,
    or a smaller stamp. It can only cancel a skip, never cause one.
    """

    jit_kw = {} if max_registers is None else {"max_registers": max_registers}
    use_first = fused_pack or skip_full

    @cuda.jit(**jit_kw)
    def rounds_kernel(mask, reached, stamp, queues, q_state, counters,
                      width, wpr, n_tiles, img8, img32, n32, height,
                      aligned, tinfo, painted):
        """Host contract: launch [blocks, tpb] cooperatively (grid.sync
        inside); init_kernel has run. mask and reached are flat
        (width * wpr) uint32, stamp flat (tiles_x * wpr) int32, queues
        flat (2 or 3 * n_tiles) int32, q_state 3 int32. img8 is the flat
        RGB image and img32 the uint32 view of its first 4 * n32 bytes
        (aligned: height % 4 == 0); tinfo is n_tiles int32 (fused pack or
        skip-full) - otherwise these are unused 1-element arrays (and
        n32 = 0). n_cur = 1 is a constant, not a read of q_state (ch04's
        deadlock lesson: a launch value is never read from mutable memory
        without a barrier)."""
        grid = cuda.cg.this_grid()
        lane = cuda.laneid
        tid = cuda.grid(1)
        warp_gid = tid >> 5
        n_warps = cuda.gridsize(1) >> 5
        left = lane < 16              # lanes 0-15 face x0-1, 16-31 x0+32
        sel = lane if left else 31 - lane  # 0 side, 1 side-up, 2 side-down
        h34 = (uint64(height) * Q3) >> Q2    # uint32 words per RGB row
        n32u = uint64(n32)
        n32m1 = uint64(max(n32 - 1, 0))
        tail = U0      # the image's last 1-3 bytes, when 4 does not divide
        if fused_pack:
            for k in range(img8.shape[0] - n32 * 4):
                tail = uint32(tail | (uint32(img8[n32 * 4 + k])
                                      << uint32(8 * k)))
        blue = BLUE_W0                    # fused paint: this lane's word
        if lane % 3 == 1:
            blue = BLUE_W1
        elif lane % 3 == 2:
            blue = BLUE_W2
        last_full = FULL                  # an all-red last word of a row
        if height & 31 != 0:
            last_full = uint32((U1 << uint32(height & 31)) - U1)

        cur = 0
        n_cur = 1
        rnd = 0
        peak = 1
        c_visits = U0
        c_iters = U0
        c_or = U0
        c_stamp = U0
        c_pack = U0
        c_skip = U0

        while n_cur > 0:
            if one_sync:
                nxt = cur + 1
                if nxt == 3:
                    nxt = 0
                if tid == 0:
                    zap = nxt + 1
                    if zap == 3:
                        zap = 0
                    q_state[zap] = 0  # read in round r-1, refilled in r+1
            else:
                nxt = 1 - cur
            next_stamp = int32(rnd + 1)
            two_r = int32(2 * rnd)
            if n_cur > peak:
                peak = n_cur

            for e in range(warp_gid, n_cur, n_warps):  # warp-uniform
                entry = queues[uint32(cur * n_tiles + e)]
                if use_first:
                    first = entry < 0
                    tx = (entry >> TY_BITS) & TX_MASK
                else:
                    first = False
                    tx = entry >> TY_BITS
                ty = entry & TY_MASK
                x0 = tx * TILE
                x = x0 + lane
                inb = x < width
                li = x * wpr + ty
                t_own = tx * wpr + ty
                # ---- loads. The export targets are fixed by the tile and
                # the lane, so their words are loaded together with the
                # tile's own, BEFORE the closure: one dependent round trip
                # fewer per visit. Only bits red in the target and not yet
                # reached may be sent; the reached half of that filter is
                # read early, but bits only ever get set, so an old read is
                # a subset of the truth and costs at most an unneeded
                # atomic, never a lost bit.
                ok_up = inb and ty > 0
                ok_dn = inb and ty + 1 < wpr
                ex = x0 - 1 if left else x0 + 32
                ttx = tx - 1 if left else tx + 1
                ew = ty
                ok_e = (x0 > 0) if left else (x0 + 32 < width)
                if conn8:
                    if sel == 1:
                        ew = ty - 1
                        ok_e = ok_e and ty > 0
                    elif sel == 2:
                        ew = ty + 1
                        ok_e = ok_e and ty + 1 < wpr
                    elif sel != 0:
                        ok_e = False
                else:
                    ok_e = ok_e and sel == 0
                le = ex * wpr + ew
                t_e = ttx * wpr + ew
                m = U0
                r0 = U0
                mu = U0
                md = U0
                me = U0
                ru = U0
                rd = U0
                re_ = U0
                p0 = U0
                if inb:
                    r0 = reached[uint32(li)]
                    if not (fused_pack and first):  # (that one packs m)
                        m = mask[uint32(li)]
                    if fused_paint:
                        p0 = painted[uint32(li)]
                if ok_up:
                    mu = mask[uint32(li - 1)]
                    ru = reached[uint32(li - 1)]
                if ok_dn:
                    md = mask[uint32(li + 1)]
                    rd = reached[uint32(li + 1)]
                if ok_e:
                    me = mask[uint32(le)]
                    re_ = reached[uint32(le)]
                ti_up = int32(NOT_PACKED)
                ti_dn = int32(NOT_PACKED)
                ti_e = int32(NOT_PACKED)
                if fused_pack or skip_full:
                    if ty > 0:
                        ti_up = tinfo[uint32(t_own - 1)]
                    if ty + 1 < wpr:
                        ti_dn = tinfo[uint32(t_own + 1)]
                    if ok_e:
                        ti_e = tinfo[uint32(t_e)]
                st_up = int32(-1)
                st_dn = int32(-1)
                if skip_full and not fused_pack:
                    # the voted up / down targets need their stamps (the
                    # full bit alone decides everywhere else: docstring)
                    if ty > 0:
                        st_up = stamp[uint32(t_own - 1)]
                    if ty + 1 < wpr:
                        st_dn = stamp[uint32(t_own + 1)]

                # ---- fused pack: only the seed's entry is FIRST (every
                # other tile was packed ahead, by its first enqueuer)
                if fused_pack:
                    if first:                     # warp-uniform
                        m = _pack_store(mask, tinfo, img32, aligned, tx, ty,
                                        width, height, wpr, h34, n32u, n32m1,
                                        tail, last_full, two_r, skip_full,
                                        lane)
                        if instrumented:
                            c_pack = uint32(c_pack + U1)
                    # an unpacked target's mask word is not known yet: send
                    # every border bit not yet reached (see the docstring)
                    if ok_up and ti_up >= two_r:
                        mu = FULL
                    if ok_dn and ti_dn >= two_r:
                        md = FULL
                    if ok_e and ti_e >= two_r:
                        me = FULL
                    f0 = uint32(r0 & m)
                else:
                    if skip_full and first:       # warp-uniform
                        exp_own = last_full if ty + 1 == wpr else FULL
                        if cuda.all_sync(FULL, (not inb) or m == exp_own):
                            if lane == 0:
                                tinfo[uint32(t_own)] = 1
                    f0 = r0
                a_up = uint32(mu & ~ru)
                a_dn = uint32(md & ~rd)
                a_e = uint32(me & ~re_)

                # ---- closure, registers only
                f = f0
                it = U0
                while True:
                    old = f
                    f = _row_fill(f, m)
                    if conn8:
                        f = _diag_step(f, m, lane)
                    f = _col_fill(f, m, lane)
                    it = uint32(it + U1)
                    if not cuda.any_sync(FULL, f != old):
                        break
                if instrumented:
                    c_visits = uint32(c_visits + U1)
                    c_iters = uint32(c_iters + it)

                # ---- export bits (every lane runs the shuffles)
                cuda.syncwarp()  # fast-path shuffles: see _col_fill
                if conn8:
                    fu = _up(f, 1)
                    fd = _down(f, 1)
                    if lane == 0:
                        fu = U0
                    if lane == 31:
                        fd = U0
                    v = uint32(f | fu | fd)
                    # ONE shuffle with a per-lane source, run by all 32
                    # lanes. `_bcast(f, 0) if left else _bcast(f, 31)`
                    # compiled to two full-mask shuffles in divergent
                    # branches (legal PTX on sm_70+, undefined on sm_6x)
                    # and cost 1-5% of the rounds kernel.
                    src = _bcast(f, 0 if left else 31)
                    if sel == 0:
                        ce = _dilate(src)               # side row, word ty
                    elif sel == 1:
                        ce = uint32((src & U1) << S31)  # corner, word ty-1
                    else:
                        ce = uint32(src >> S31)         # corner, word ty+1
                else:
                    v = f
                    ce = f            # only lanes 0 and 31 export sideways
                c_up = uint32(((v & U1) << S31) & a_up)  # bit 0 -> 31 of ty-1
                c_dn = uint32((v >> S31) & a_dn)         # bit 31 -> 0 of ty+1
                c_e = uint32(ce & a_e)

                if skip_full:
                    # full bit set: the target's first visit has run or is
                    # queued (docstring), so no stamp is needed
                    skip_up = (ti_up & 1) != 0
                    skip_dn = (ti_dn & 1) != 0
                    if not fused_pack:
                        # the warp holds all 32 words of the up / down
                        # target (one per row); rows past the image vote
                        # yes. Fullness from the mask says nothing about
                        # a visit, so these also need stamp >= r.
                        exp_dn = last_full if ty + 2 == wpr else FULL
                        skip_up = (st_up >= rnd and cuda.all_sync(
                            FULL, (not inb) or mu == FULL))
                        skip_dn = (st_dn >= rnd and cuda.all_sync(
                            FULL, (not inb) or md == exp_dn))
                    if skip_up and c_up != U0:
                        c_up = U0
                        if instrumented:
                            c_skip = uint32(c_skip + U1)
                    if skip_dn and c_dn != U0:
                        c_dn = U0
                        if instrumented:
                            c_skip = uint32(c_skip + U1)
                    if (ti_e & 1) != 0 and c_e != U0:
                        c_e = U0
                        if instrumented:
                            c_skip = uint32(c_skip + U1)

                # ---- atomics: the owner's own word, then the exports.
                # (f != f0 implies inb: out-of-image lanes hold m = 0.)
                # All three exports are issued before any old value is
                # used, so their round trips overlap instead of chaining
                # (a skipped export keeps o = 0, and c & ~0 = c = 0).
                if f != f0:
                    cuda.atomic.or_(reached, uint32(li), f)
                o_up = U0
                o_dn = U0
                o_e = U0
                if c_up != U0:
                    o_up = cuda.atomic.or_(reached, uint32(li - 1), c_up)
                if c_dn != U0:
                    o_dn = cuda.atomic.or_(reached, uint32(li + 1), c_dn)
                if c_e != U0:
                    o_e = cuda.atomic.or_(reached, uint32(le), c_e)
                add_up = uint32(c_up & ~o_up) != U0
                add_dn = uint32(c_dn & ~o_dn) != U0
                add_e = uint32(c_e & ~o_e) != U0
                if instrumented:
                    c_or = uint32(c_or + uint32(f != f0) + uint32(c_up != U0)
                                  + uint32(c_dn != U0) + uint32(c_e != U0))

                # ---- enqueue: one lane per target tile. Side lanes own
                # theirs; lane 3 speaks for ty-1 and lane 4 for ty+1.
                cuda.syncwarp()  # reconverge after the atomics' branches
                b_up = cuda.ballot_sync(FULL, add_up)
                b_dn = cuda.ballot_sync(FULL, add_dn)
                want = add_e
                tty = ew
                if lane == 3:
                    want = b_up != U0
                    ttx = tx
                    tty = ty - 1
                elif lane == 4:
                    want = b_dn != U0
                    ttx = tx
                    tty = ty + 1
                ahead = False
                if want:
                    if instrumented:
                        c_stamp = uint32(c_stamp + U1)
                    old_st = cuda.atomic.max(stamp, uint32(ttx * wpr + tty),
                                             next_stamp)
                    if old_st < next_stamp:
                        item = (ttx << TY_BITS) | tty
                        if fused_pack:
                            ahead = old_st < 0   # this warp packs it, below
                        elif skip_full and old_st < 0:
                            item = item | FIRST_ENTRY   # its first visit
                        _warp_enqueue_global(queues, nxt * n_tiles, n_tiles,
                                             q_state, nxt, item, counters)

                # ---- fused pack, ahead: a tile is packed in the round it
                # is first enqueued, by the warp whose atomic.max saw -1
                # (one warp per tile), after this visit's own exports. Its
                # first visit then carries no pack, and from that round on
                # every exporter sees its mask (and its full bit).
                if fused_pack:
                    cuda.syncwarp()
                    pend = cuda.ballot_sync(FULL, ahead)
                    while pend != U0:                    # warp-uniform
                        src = cuda.ffs(pend) - 1
                        pend = uint32(pend & (pend - U1))
                        cuda.syncwarp()  # fast-path shuffles: see _col_fill
                        px = cuda.shfl_sync(FULL, int32(ttx), src)
                        py = cuda.shfl_sync(FULL, int32(tty), src)
                        _pack_store(mask, tinfo, img32, aligned, px, py,
                                    width, height, wpr, h34, n32u, n32m1,
                                    tail, last_full, two_r, skip_full, lane)
                        if instrumented:
                            c_pack = uint32(c_pack + U1)

                # ---- fused paint: the owner paints what this visit added
                # to its closed set (painted is written only by owners)
                if fused_paint:
                    new = uint32(f & ~p0)
                    _paint_rows(img8, img32, aligned, x0, ty, height, h34,
                                new, lane, blue)
                    if new != U0:
                        painted[uint32(li)] = f

            if one_sync:
                grid.sync()  # B_r: every export, enqueue and reset is visible
                n_next = q_state[nxt]
            else:
                grid.sync()  # every export and enqueue of the round is visible
                n_next = q_state[nxt]
                if tid == 0:
                    q_state[cur] = 0  # nobody reads it again; next round fills
                grid.sync()  # everyone has n_next; next round's atomics begin
            cur = nxt
            n_cur = n_next
            rnd += 1

        if instrumented:
            if lane == 0:
                cuda.atomic.add(counters, TILE_VISITS, c_visits)
                cuda.atomic.add(counters, CLOSURE_ITERS, c_iters)
                cuda.atomic.add(counters, PACKED_TILES, c_pack)
            cuda.atomic.add(counters, OR_ATOMICS, c_or)
            cuda.atomic.add(counters, STAMP_ATOMICS, c_stamp)
            cuda.atomic.add(counters, SKIPPED_EXPORTS, c_skip)
        if tid == 0:
            counters[ROUNDS] = rnd
            counters[PEAK_ROUND] = peak

    return rounds_kernel


_ROUNDS_CACHE = {}


def get_rounds_kernel(connectivity, bare=False, max_registers=None,
                      one_sync=False, fused_pack=False, skip_full=False,
                      fused_paint=False):
    """The rounds kernel for (connectivity, bare, register cap, switches),
    built once per key. A cap is a link-time option, so each cap is its
    own compiled kernel."""
    if fused_paint and not fused_pack:
        raise ValueError("fused_paint needs fused_pack (a tile's RGB must "
                         "be read before anything paints it)")
    key = (connectivity, bool(bare), max_registers, bool(one_sync),
           bool(fused_pack), bool(skip_full), bool(fused_paint))
    if key not in _ROUNDS_CACHE:
        _ROUNDS_CACHE[key] = _make_rounds_kernel(
            connectivity == 8, not bare, max_registers, one_sync=one_sync,
            fused_pack=fused_pack, skip_full=skip_full,
            fused_paint=fused_paint)
    return _ROUNDS_CACHE[key]


@cuda.jit
def init_kernel(reached_flat, stamp_flat, queues, q_state, counters,
                seed_word, seed_bits, seed_tile, seed_entry, tinfo,
                tinfo_init, painted):
    """Per-run state, written on the device (no host copy on the clock -
    ch06's lesson: a numba H2D copy from pageable memory is synchronous).

    reached = 0 except the seed bit; stamp = -1 except the seed tile's 0;
    queue 0 holds the seed tile; q_state = [1, 0, 0]; counters zeroed;
    tinfo = tinfo_init (NOT_PACKED for the fused pack, 0 for skip-full,
    a 1-element dummy otherwise); painted = 0 (fused paint; else a
    dummy). Written in one pass with no
    read-modify-write, so no ordering issue.
    """
    i0 = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i0, reached_flat.shape[0], stride):
        if i == seed_word:
            reached_flat[i] = seed_bits
        else:
            reached_flat[i] = U0
    for i in range(i0, stamp_flat.shape[0], stride):
        if i == seed_tile:
            stamp_flat[i] = 0
        else:
            stamp_flat[i] = -1
    for i in range(i0, tinfo.shape[0], stride):
        tinfo[i] = tinfo_init
    for i in range(i0, painted.shape[0], stride):
        painted[i] = U0
    if i0 == 0:
        queues[0] = seed_entry
        q_state[0] = 1
        q_state[1] = 0
        q_state[2] = 0
        for c in range(counters.shape[0]):
            counters[c] = 0


@cuda.jit
def paint_kernel(reached, img):
    """reached -> RGB: every reached pixel becomes (0, 0, 255), as ch03
    paints it; everything else is left untouched.

    The same 2D grid as ch06's pack_kernel (one warp per word, rows
    strided along gridDim.y), so lanes write 32 consecutive pixels and
    the stores coalesce. A zero word is skipped warp-uniformly. reached
    is a subset of the packed mask, whose bits past `height` are zero, so
    a set bit is always inside the image.
    """
    width = img.shape[0]
    words_per_row = reached.shape[1]
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    w = t >> 5              # warp-uniform: blockDim.x is a multiple of 32
    lane = t & 31
    if w >= words_per_row:
        return
    y = w * TILE + lane
    for x in range(cuda.blockIdx.y, width, cuda.gridDim.y):
        bits = reached[x, w]
        if bits != U0:
            if (bits >> uint32(lane)) & U1:
                img[x, y, 0] = 0
                img[x, y, 1] = 0
                img[x, y, 2] = 255


@cuda.jit
def paint_tiles_kernel(reached, mask, tinfo, img8, img32, aligned, width,
                       height, wpr, n_tiles):
    """The fused-pack paint: (reached & mask) -> (0, 0, 255), one warp per
    TILE, grid-stride. With the pack fused, `reached` may hold candidate
    bits that are not red, so the AND with the mask is what makes the
    paint exact; mask words exist only for packed tiles, and a tile that
    was never packed (tinfo NOT_PACKED) has no reached bit at all (every
    bit that lands in a tile enqueues it, and its visit packs it), so it
    is skipped after one load - the thin strip touches 2 of 16,384 tiles.

    Lane i loads row x0+i's two words once, before any store, so no load
    waits behind a store it might alias (the plain paint_kernel loads one
    row per iteration between its stores). Then the reached rows are
    painted one at a time, all 32 lanes writing the row's 96 contiguous
    bytes: as 24 word stores of the blue pattern when the whole word is
    reached and rows are 4-byte aligned, as byte stores otherwise.
    """
    lane = cuda.laneid
    warp = cuda.grid(1) >> 5
    n_warps = cuda.gridsize(1) >> 5
    h34 = (uint64(height) * Q3) >> Q2
    blue = BLUE_W0
    if lane % 3 == 1:
        blue = BLUE_W1
    elif lane % 3 == 2:
        blue = BLUE_W2
    for t in range(warp, n_tiles, n_warps):            # warp-uniform
        if tinfo[t] == NOT_PACKED:
            continue
        tx = t // wpr
        ty = t - tx * wpr
        x0 = tx * TILE
        x = x0 + lane
        bits = U0
        if x < width:
            li = x * wpr + ty
            bits = reached[li]
            if bits != U0:
                bits = uint32(bits & mask[li])
        _paint_rows(img8, img32, aligned, x0, ty, height, h34, bits, lane,
                    blue)


@cuda.jit
def gpu_sleep_kernel(n_naps):
    """A one-thread spacer for timing: launched before a timed sequence
    so every later launch is already queued when the first event fires,
    which keeps Python's launch overhead out of the event interval."""
    if cuda.grid(1) == 0:
        for _ in range(n_naps):
            cuda.nanosleep(100000)
