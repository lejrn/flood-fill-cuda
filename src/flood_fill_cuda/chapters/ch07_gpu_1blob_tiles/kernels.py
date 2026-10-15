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
from numba import cuda, int32, uint32

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

TILE = 32
# A queue entry packs a tile as (tx << 16) | ty, so decoding is two
# shifts and a mask instead of a 64-bit division by words_per_row.
TY_BITS = 16
TY_MASK = (1 << TY_BITS) - 1

# Slots in the int64 counters array
ROUNDS = 0          # tile-BFS rounds (two grid.sync each)
TILE_VISITS = 1     # warps that closed a tile (= tiles ever enqueued + 1)
CLOSURE_ITERS = 2   # closure loop trips, summed over visits
OR_ATOMICS = 3      # atomic.or_ on `reached` (own word + exports)
STAMP_ATOMICS = 4   # atomic.max on the per-tile stamp (enqueue requests)
PEAK_ROUND = 5      # largest round (tiles)
OVERFLOW = 6        # defensive tripwire: structurally unreachable
NUM_COUNTERS = 7


# --------------------------------------------------------------- bit helpers

@cuda.jit(device=True, inline=True)
def _up(v, k):
    """shfl_up of a uint32 word: lane i gets lane i-k's word.

    Numba's shuffle intrinsic has int32 and int64 overloads only, and a
    uint32 argument converts SAFELY to int64 - so a plain call shuffles
    a 64-bit value (two SHFLs) and hands back an int64 that drags the
    rest of the arithmetic to 64 bits. Shuffling the int32 bit pattern
    and casting back is one SHFL and stays 32-bit.
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


# --------------------------------------------------------------- kernels

def _make_rounds_kernel(conn8, instrumented, max_registers=None):
    """Build the cooperative rounds kernel for one connectivity.

    Both flags are compile-time constants (closure variables), so the
    4-conn kernel carries no diagonal code and the bare twin no counters
    - the twin discipline of ch03, without hand-copied bodies.

    Registers, measured. Every array here is flat and indexed with a
    uint32 (Numba then emits no negative-index wraparound), the side and
    corner exports are spread over lanes 0-2 and 29-31 so each lane owns
    at most three targets, and the counters are uint32. The LINKED 8-conn
    kernel still reports 143 registers/thread: ptxas compiles the same
    PTX on its own (`-c`, nothing linked) to 87 with zero spills, and the
    other ~56 arrive with linking the external grid.sync() call from
    libcudadevrt. A link-time cap (`max_registers`) trades them for
    spills and more resident warps; it is a parameter so the benchmark
    can price that trade.
    """

    jit_kw = {} if max_registers is None else {"max_registers": max_registers}

    @cuda.jit(**jit_kw)
    def rounds_kernel(mask, reached, stamp, queues, q_state, counters,
                      width, wpr, n_tiles):
        """Host contract: launch [blocks, tpb] cooperatively (grid.sync
        inside); init_kernel has run. mask and reached are flat
        (width * wpr) uint32, stamp flat (tiles_x * wpr) int32, queues
        flat (2 * n_tiles) int32 (round r reads half r & 1). n_cur = 1 is
        a constant, not a read of q_state (ch04's deadlock lesson: a
        launch value is never read from mutable memory without a
        barrier)."""
        grid = cuda.cg.this_grid()
        lane = cuda.laneid
        tid = cuda.grid(1)
        warp_gid = tid >> 5
        n_warps = cuda.gridsize(1) >> 5
        left = lane < 16              # lanes 0-15 face x0-1, 16-31 x0+32
        sel = lane if left else 31 - lane  # 0 side, 1 side-up, 2 side-down

        cur = 0
        n_cur = 1
        rnd = 0
        peak = 1
        c_visits = U0
        c_iters = U0
        c_or = U0
        c_stamp = U0

        while n_cur > 0:
            nxt = 1 - cur
            next_stamp = int32(rnd + 1)
            if n_cur > peak:
                peak = n_cur

            for e in range(warp_gid, n_cur, n_warps):  # warp-uniform
                entry = queues[uint32(cur * n_tiles + e)]
                tx = entry >> TY_BITS
                ty = entry & TY_MASK
                x0 = tx * TILE
                x = x0 + lane
                inb = x < width
                li = x * wpr + ty
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
                m = U0
                f0 = U0
                a_up = U0
                a_dn = U0
                a_e = U0
                if inb:
                    m = mask[uint32(li)]
                    f0 = reached[uint32(li)]
                if ok_up:
                    a_up = uint32(mask[uint32(li - 1)]
                                  & ~reached[uint32(li - 1)])
                if ok_dn:
                    a_dn = uint32(mask[uint32(li + 1)]
                                  & ~reached[uint32(li + 1)])
                if ok_e:
                    a_e = uint32(mask[uint32(le)] & ~reached[uint32(le)])

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
                ttx = tx - 1 if left else tx + 1
                tty = ew
                if lane == 3:
                    want = b_up != U0
                    ttx = tx
                    tty = ty - 1
                elif lane == 4:
                    want = b_dn != U0
                    ttx = tx
                    tty = ty + 1
                if want:
                    if instrumented:
                        c_stamp = uint32(c_stamp + U1)
                    if cuda.atomic.max(stamp, uint32(ttx * wpr + tty),
                                       next_stamp) < next_stamp:
                        _warp_enqueue_global(queues, nxt * n_tiles, n_tiles,
                                             q_state, nxt,
                                             (ttx << TY_BITS) | tty,
                                             counters)

            grid.sync()  # every export and enqueue of this round is visible
            n_next = q_state[nxt]
            if tid == 0:
                q_state[cur] = 0  # nobody reads it again; next round fills it
            grid.sync()  # everyone has n_next; next round's atomics may begin
            cur = nxt
            n_cur = n_next
            rnd += 1

        if instrumented:
            if lane == 0:
                cuda.atomic.add(counters, TILE_VISITS, c_visits)
                cuda.atomic.add(counters, CLOSURE_ITERS, c_iters)
            cuda.atomic.add(counters, OR_ATOMICS, c_or)
            cuda.atomic.add(counters, STAMP_ATOMICS, c_stamp)
        if tid == 0:
            counters[ROUNDS] = rnd
            counters[PEAK_ROUND] = peak

    return rounds_kernel


_ROUNDS_CACHE = {}


def get_rounds_kernel(connectivity, bare=False, max_registers=None):
    """The rounds kernel for (connectivity, bare, register cap), built
    once per key. A cap is a link-time option, so each cap is its own
    compiled kernel."""
    key = (connectivity, bool(bare), max_registers)
    if key not in _ROUNDS_CACHE:
        _ROUNDS_CACHE[key] = _make_rounds_kernel(connectivity == 8,
                                                 not bare, max_registers)
    return _ROUNDS_CACHE[key]


@cuda.jit
def init_kernel(reached_flat, stamp_flat, queues, q_state, counters,
                seed_word, seed_bits, seed_tile, seed_entry):
    """Per-run state, written on the device (no host copy on the clock -
    ch06's lesson: a numba H2D copy from pageable memory is synchronous).

    reached = 0 except the seed bit; stamp = -1 except the seed tile's 0;
    queue 0 holds the seed tile; q_state = [1, 0]; counters zeroed.
    Written in one pass with no read-modify-write, so no ordering issue.
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
    if i0 == 0:
        queues[0] = seed_entry
        q_state[0] = 1
        q_state[1] = 0
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
def gpu_sleep_kernel(n_naps):
    """A one-thread spacer for timing: launched before a timed sequence
    so every later launch is already queued when the first event fires,
    which keeps Python's launch overhead out of the event interval."""
    if cuda.grid(1) == 0:
        for _ in range(n_naps):
            cuda.nanosleep(100000)
