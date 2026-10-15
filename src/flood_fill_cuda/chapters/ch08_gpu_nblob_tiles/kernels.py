"""Seeded waves over tile components: many blobs, BFS-like, tile sized.

Chapter 7 filled ONE blob by a BFS over 32x32 tiles closed in
registers. This chapter asks whether the same tiles can label MANY
blobs, and keeps the method BFS-like: labels travel as frontier waves,
one grid-wide barrier per round.

THE NODE IS A TILE COMPONENT, NOT A TILE
Inside one tile a blob can fall apart into several local components
("comps"): a U that leaves the tile and comes back gives two comps in
that tile. ch07 opened such a tile once per arriving wave (98,195
openings on 62,500 tiles in sq_8000, ch07 session
tiles_20261015T114440Z). Here every comp is a node of its own, and
every node is claimed exactly once.

THE TILE (ch07's, unchanged)
32 consecutive rows x0 .. x0+31 times one mask word (32 consecutive y).
Lane i of a warp holds row x0+i as a uint32 word; bit j is y = 32*ty+j.
Rows past the image hold 0, so every lane runs every instruction.

THE FIVE LAUNCHES
    K1 pack_tiles   RGB -> 1 bit/px, read tile by tile (rgb contract)
    K0 init         zero the queue state and the counters
    K2 label        one warp per tile: close every comp in registers,
                    record each ring comp (side words, anchor, dirs),
                    seed it or leave it unclaimed, queue crossing seeds
    K3 waves        cooperative: rounds of claims by CAS over the comp
                    graph, collisions recorded; then one union pass,
                    then two flatten passes to the canonical label
    K4 paint        one warp per tile: palette[label % 6] on red

A RING COMP touches the 124-pixel frame of its tile, so it may have
neighbours in other tiles; it gets a 32-byte record at id 64*t + b,
where b is its ordinal among the tile's ring comps (b < 62, proven in
the README). An INTERIOR comp touches no frame pixel, so it is a whole
blob: its label is its own anchor and it gets no record.

THE LABEL
A comp's anchor is its minimum linear index lin = x*height + y. A
blob's label is the minimum anchor over its comps, which is ch05's and
ch06's canonical label exactly (not just an equivalent partition).

Record layout (int32, 8 words per id):
    0 N   top row word        1 S   bottom row word
    2 Wc  left column word    3 Ec  right column word
    4 anchor                  5 dirs (8 direction bits)
    6 label (-1 unclaimed; a seed id during the waves; the canonical
      label after flatten)
    7 rmin (min anchor of the class, valid at roots)

Direction bits: 0 up (tx-1), 1 down (tx+1), 2 left (ty-1), 3 right
(ty+1), 4 UL, 5 UR, 6 DL, 7 DR. PRED = {up, left, UL, UR} point at
lex-smaller tiles, SUCC = {down, right, DL, DR} at lex-larger ones.
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda, int32, uint32

# ch07's closure and warp helpers, imported so the tile closure cannot
# drift between the two chapters.
from ..ch07_gpu_1blob_tiles.kernels import (  # noqa: F401
    _up, _down, _bcast, _row_fill, _col_fill, _diag_step, _dilate,
    FULL, U0, U1, S1, S31, gpu_sleep_kernel)
# ch06's union-find, verbatim (the atomicMin protocol; see README risk 8:
# design 3's race came from "optimising" the line that compares the
# atomicMin result) and ch06's pack (the reference K1 must equal).
from ..ch06_gpu_nblob_runs.kernels import (  # noqa: F401
    pack_kernel, _find, _union)
from ..ch05_gpu_nblob_nblock.kernels import (  # noqa: F401
    PALETTE_HOST, N_PALETTE, SMID_CU, get_globaltimer)

TILE = 32
IDS_PER_TILE = 64          # record ids per tile: 64*t + b, b < 62
REC = 8                    # int32 words per record (one 32-byte sector)
R_N, R_S, R_W, R_E, R_ANCHOR, R_DIRS, R_LABEL, R_RMIN = range(8)

PRED = 0x35                # up, left, UL, UR
SUCC = 0xCA                # down, right, DL, DR
EDGE = np.uint32(0x80000001)
NEG1 = np.int32(-1)
HDR_RING_MASK = 63
INDEX_BYTES = 128          # index twin: a uint8 ordinal per frame pixel,
                           # 32 per side (N, S, Wc, Ec) per tile
HDR_TOTAL_SHIFT = 8

# Direction tables packed into ints (no device arrays needed):
# dx = +1 for {down, DL, DR}, -1 for {up, UL, UR}; dy likewise.
DXP = 0xC2
DXM = 0x31
DYP = 0xA8
DYM = 0x54
# my side / the neighbour's facing side, 2 bits per direction
# (0 = N, 1 = S, 2 = Wc, 3 = Ec)
MY_SIDE = (0 | 1 << 2 | 2 << 4 | 3 << 6 | 0 << 8 | 0 << 10 | 1 << 12
           | 1 << 14)
FACING = (1 | 0 << 2 | 3 << 4 | 2 << 6 | 1 << 8 | 1 << 10 | 0 << 12
          | 0 << 14)

# q_state (int32[8])
Q_SEEDS = 0                # crossing seeds appended by K2 (the first rear)
Q_C0 = 1                   # 1, 2, 3: rotating per-round append counters
Q_COLL = 4                 # collision list length
Q_VALVE = 5                # valve appends
N_QSTATE = 8

# counters (int64[48]); the bare twins keep only what the result needs
COUNTER_NAMES = (
    # K2 label
    "tiles_touched", "comps", "ring_comps", "interior_comps",
    "max_comps_tile", "max_ring_tile", "close_iters", "exit_last",
    "exit_frontier", "seeds", "seeds_lattice", "seeds_queued",
    "seed_append_atomics",
    # K3 waves
    "rounds", "tasks", "dense_tasks", "ksteps", "sig_tests", "cas",
    "claims",
    "collisions", "coll_inline", "pred_skips", "append_atomics",
    "coll_append_atomics", "union_calls", "union_done",
    "peak_frontier", "valve_round", "valve_seeds",
    # flatten
    "roots", "rmin_atomics",
    # K4 paint
    "paint_tiles", "paint_uniform", "paint_single_interior",
    "paint_reclosed_tiles", "paint_reclose_iters", "rows_painted",
    # debug twins
    "ordinal_overflow", "anchor_mismatch", "unlabelled",
)
NUM_COUNTERS = 48
assert len(COUNTER_NAMES) <= NUM_COUNTERS
C = {name: i for i, name in enumerate(COUNTER_NAMES)}
TILES_TOUCHED = C["tiles_touched"]
COMPS = C["comps"]
RING_COMPS = C["ring_comps"]
INTERIOR_COMPS = C["interior_comps"]
MAX_COMPS_TILE = C["max_comps_tile"]
MAX_RING_TILE = C["max_ring_tile"]
CLOSE_ITERS = C["close_iters"]
EXIT_LAST = C["exit_last"]
EXIT_FRONTIER = C["exit_frontier"]
SEEDS = C["seeds"]
SEEDS_LATTICE = C["seeds_lattice"]
SEEDS_QUEUED = C["seeds_queued"]
SEED_APPEND_ATOMICS = C["seed_append_atomics"]
ROUNDS = C["rounds"]
TASKS = C["tasks"]
DENSE_TASKS = C["dense_tasks"]
KSTEPS = C["ksteps"]
SIG_TESTS = C["sig_tests"]
CAS = C["cas"]
CLAIMS = C["claims"]
COLLISIONS = C["collisions"]
COLL_INLINE = C["coll_inline"]
PRED_SKIPS = C["pred_skips"]
APPEND_ATOMICS = C["append_atomics"]
COLL_APPEND_ATOMICS = C["coll_append_atomics"]
UNION_CALLS = C["union_calls"]
UNION_DONE = C["union_done"]
PEAK_FRONTIER = C["peak_frontier"]
VALVE_ROUND = C["valve_round"]
VALVE_SEEDS = C["valve_seeds"]
ROOTS = C["roots"]
RMIN_ATOMICS = C["rmin_atomics"]
PAINT_TILES = C["paint_tiles"]
PAINT_UNIFORM = C["paint_uniform"]
PAINT_SINGLE_INTERIOR = C["paint_single_interior"]
PAINT_RECLOSED_TILES = C["paint_reclosed_tiles"]
PAINT_RECLOSE_ITERS = C["paint_reclose_iters"]
ROWS_PAINTED = C["rows_painted"]
ORDINAL_OVERFLOW = C["ordinal_overflow"]
ANCHOR_MISMATCH = C["anchor_mismatch"]
UNLABELLED = C["unlabelled"]

# instrumented waves: %globaltimer stamps. Slot r is the end of round
# r for r < ST_TAIL; slot ST_TAIL holds the end of the last round when
# there are ST_TAIL rounds or more (so the union split never includes
# rounds); then the fixed slots of the union and flatten passes.
N_STAMPS = 256
ST_TAIL = 252
ST_UNION = 253
ST_FLAT1 = 254
ST_FLAT2 = 255


# ------------------------------------------------------------ helpers

@cuda.jit(device=True, inline=True)
def _first_pixel(rem, lane):
    """The lowest remaining pixel in (lane, bit) order: returns (f, a, bit)
    with f its single bit on lane a. It is the min lin of what remains,
    so the comp grown from it has it as its anchor."""
    cuda.syncwarp()
    nz = uint32(cuda.ballot_sync(FULL, rem != U0))
    a = int32(cuda.ffs(nz) - 1)
    wa = _bcast(rem, a)
    bit = int32(cuda.ffs(wa) - 1)
    f = U0
    if lane == a:
        f = uint32(U1 << uint32(bit))
    return f, a, bit


@cuda.jit(device=True, inline=True)
def _close(f, rem, lane, conn8):
    """Close f inside rem (ch07's row fill, diagonal step, column fill)
    with two exits instead of ch07's "nothing changed" proving trip:

      last comp   f covers everything that remains (all_sync)
      frontier    no remaining pixel outside f is adjacent to f

    Either one proves f closed without another full pass. Returns
    (f, iterations, 1 if the last-comp exit fired else 0)."""
    it = 0
    last = 0
    while True:
        f = _row_fill(f, rem)
        if conn8:
            f = _diag_step(f, rem, lane)
        f = _col_fill(f, rem, lane)
        it += 1
        if cuda.all_sync(FULL, f == rem):
            last = 1
            break
        cuda.syncwarp()
        # lane 0's _up and lane 31's _down return their own word, which
        # the in-row dilation already holds, so no clamp is needed
        if conn8:
            d = _dilate(f)
            g = uint32(d | _up(d, 1) | _down(d, 1))
        else:
            g = uint32(f | (f << S1) | (f >> S1) | _up(f, 1) | _down(f, 1))
        if not cuda.any_sync(FULL, uint32(g & rem & ~f) != U0):
            break
    return f, it, last


@cuda.jit(device=True, inline=True)
def _close_exits(f, rem, lane, conn8, exits):
    """_close with a compile-time choice of exits, for experiment E2:
    exits 2 = last comp + frontier (the default, == _close),
    1 = frontier only, 0 = ch07's loop (stop when a trip changed
    nothing). Returns (f, iterations, 1 if the last-comp exit fired)."""
    if exits == 2:
        return _close(f, rem, lane, conn8)
    it = 0
    while True:
        old = f
        f = _row_fill(f, rem)
        if conn8:
            f = _diag_step(f, rem, lane)
        f = _col_fill(f, rem, lane)
        it += 1
        if exits == 1:
            cuda.syncwarp()
            if conn8:
                d = _dilate(f)
                g = uint32(d | _up(d, 1) | _down(d, 1))
            else:
                g = uint32(f | (f << S1) | (f >> S1) | _up(f, 1)
                           | _down(f, 1))
            if not cuda.any_sync(FULL, uint32(g & rem & ~f) != U0):
                break
        else:
            if not cuda.any_sync(FULL, f != old):
                break
    return f, it, 0


@cuda.jit(device=True, inline=True)
def _is_ring(f, lane):
    """Does the comp f touch the tile frame (lane 0, lane 31, bit 0 or
    bit 31)? Warp-uniform."""
    on = f != U0 and (lane == 0 or lane == 31 or uint32(f & EDGE) != U0)
    return cuda.any_sync(FULL, on)


@cuda.jit(device=True, inline=True)
def _warp_slot(pred, lane, state, slot):
    """Warp-aggregated append at a CONVERGENT point with the FULL mask:
    one ballot, one atomic by the leader, a broadcast of the base.
    Returns (position for lanes with pred, 1 if an atomic was issued)."""
    cuda.syncwarp()
    bal = uint32(cuda.ballot_sync(FULL, pred))
    pos = int32(0)
    n_atomic = 0
    if bal != U0:
        cnt = int32(cuda.popc(bal))
        rank = int32(cuda.popc(bal & uint32(cuda.lanemask_lt())))
        leader = int32(cuda.ffs(bal) - 1)
        base = int32(0)
        if lane == leader:
            base = cuda.atomic.add(state, slot, cnt)
        base = cuda.shfl_sync(FULL, base, leader)
        pos = base + rank
        n_atomic = 1
    return pos, n_atomic


@cuda.jit(device=True, inline=True)
def _claim(hit, c2, d2, lab, succ, lane, rec, parent, qc, ql, qd, ca, cb,
           q_state, slot, rear, coll_cap):
    """One K3 step's claims, at a CONVERGENT point (every lane calls it):
    a CAS on comp c2's label for lanes with a hit, the warp-aggregated
    queue append of the winners, and the collision record (SUCC side
    only) of a hit on another wave's comp. Past the collision list's
    soft cap the pair is unioned at once (ch06's _union is lock-free).

    Returns (claimed, collided, inline, union done, pred skip, append
    atomic, collision append atomic)."""
    old = int32(0)
    if hit:
        old = cuda.atomic.cas(rec, uint32(REC * c2 + R_LABEL), NEG1, lab)
    claimed = hit and old == -1
    other = hit and old != -1 and old != lab
    collided = other and succ
    pos, na = _warp_slot(claimed, lane, q_state, slot)
    if claimed:
        p = rear + pos
        qc[uint32(p)] = c2
        ql[uint32(p)] = lab
        qd[uint32(p)] = d2
    cpos, cna = _warp_slot(collided, lane, q_state, Q_COLL)
    inline = 0
    done = 0
    if collided:
        if cpos < coll_cap:
            ca[uint32(cpos)] = lab
            cb[uint32(cpos)] = old
        else:
            done = _union(parent, lab, old)
            inline = 1
    return (claimed, collided, inline, done, other and not succ, na, cna)


@cuda.jit(device=True, inline=True)
def _warp_sum(v):
    """int64 warp sum, valid on lane 0."""
    v += cuda.shfl_down_sync(FULL, v, 16)
    v += cuda.shfl_down_sync(FULL, v, 8)
    v += cuda.shfl_down_sync(FULL, v, 4)
    v += cuda.shfl_down_sync(FULL, v, 2)
    v += cuda.shfl_down_sync(FULL, v, 1)
    return v


@cuda.jit(device=True, inline=True)
def _flush(lane, cnt, id0, rN, rS, rW, rE, ranchor, side, wl, wr,
           lat_hit, rec, parent, qc, ql, qd, q_state, conn8):
    """Write one chunk of up to 32 ring records (lane k < cnt holds comp
    id0 + k) and queue the crossing seeds.

    The facing words are reduced to warp-uniform values here, at the
    flush, not before the comp loop: the loads were issued at tile
    start and their latency hides behind the closures.
    Returns (seed, lattice-only seed, queued, append atomics) per lane.
    """
    cuda.syncwarp()
    upw = _bcast(side, 1)                 # row x0-1, word ty
    dnw = _bcast(side, 30)                # row x0+32, word ty
    lft = uint32(cuda.ballot_sync(FULL, (wl >> S31) != U0))  # (x0+i, 32ty-1)
    rgt = uint32(cuda.ballot_sync(FULL, (wr & U1) != U0))    # (x0+i, 32ty+32)
    if conn8:
        ulb = _bcast(side, 0) >> S31      # (x0-1, 32ty-1)
        urb = _bcast(side, 2) & U1        # (x0-1, 32ty+32)
        dlb = _bcast(side, 29) >> S31     # (x0+32, 32ty-1)
        drb = _bcast(side, 31) & U1       # (x0+32, 32ty+32)
        dirs = (int32(uint32(_dilate(rN) & upw) != U0)
                | (int32(uint32(_dilate(rS) & dnw) != U0) << 1)
                | (int32(uint32(_dilate(rW) & lft) != U0) << 2)
                | (int32(uint32(_dilate(rE) & rgt) != U0) << 3)
                | (int32(uint32(rN & U1 & ulb) != U0) << 4)
                | (int32(uint32((rN >> S31) & urb) != U0) << 5)
                | (int32(uint32(rS & U1 & dlb) != U0) << 6)
                | (int32(uint32((rS >> S31) & drb) != U0) << 7))
    else:
        dirs = (int32(uint32(rN & upw) != U0)
                | (int32(uint32(rS & dnw) != U0) << 1)
                | (int32(uint32(rW & lft) != U0) << 2)
                | (int32(uint32(rE & rgt) != U0) << 3))
    valid = lane < cnt
    cid = id0 + lane
    pred_free = (dirs & PRED) == 0
    seed = valid and (pred_free or lat_hit)
    if valid:
        w = uint32(REC * cid)
        rec[w] = int32(rN)
        rec[w + 1] = int32(rS)
        rec[w + 2] = int32(rW)
        rec[w + 3] = int32(rE)
        rec[w + 4] = ranchor
        rec[w + 5] = dirs
        rec[w + 6] = cid if seed else NEG1
        rec[w + 7] = ranchor
        if seed:
            parent[uint32(cid)] = cid
    queue = seed and dirs != 0
    pos, n_atomic = _warp_slot(queue, lane, q_state, Q_SEEDS)
    if queue:
        qc[uint32(pos)] = cid
        ql[uint32(pos)] = cid
        qd[uint32(pos)] = dirs
    return (int32(seed), int32(seed and not pred_free), int32(queue),
            n_atomic)


# ------------------------------------------------------------- K1 pack

@cuda.jit(device=True, inline=True)
def _red_px(img, x, y, ok):
    """Independent loads ANDed (ch06's lesson: `and` chains them)."""
    r = img[x, y, 0]
    g = img[x, y, 1]
    b = img[x, y, 2]
    return ok & (r == 255) & (g == 0) & (b == 0)


@cuda.jit(device=True, inline=True)
def _red8(img, xb, yc, oky, wm):
    """Bit i = pixel (xb+i, yc) is red, for 8 rows: 24 independent byte
    loads per lane, every index clamped so every load is unconditional,
    validity ANDed in afterwards."""
    a = U0
    x = xb
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm)))
    x = xb + 1
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm))) << uint32(1)
    x = xb + 2
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm))) << uint32(2)
    x = xb + 3
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm))) << uint32(3)
    x = xb + 4
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm))) << uint32(4)
    x = xb + 5
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm))) << uint32(5)
    x = xb + 6
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm))) << uint32(6)
    x = xb + 7
    a |= uint32(_red_px(img, min(x, wm), yc, oky & (x <= wm))) << uint32(7)
    return a


@cuda.jit(device=True, inline=True)
def _ballot8(acc, lane, rb, m):
    """Lane rb+i keeps the ballot of bit i: the word of row rb+i."""
    cuda.syncwarp()
    for i in range(8):
        bits = uint32(cuda.ballot_sync(FULL, ((acc >> uint32(i)) & U1)
                                       != U0))
        if lane == rb + i:
            m = bits
    return m


def _make_pack_kernel(variant):
    """K1, grafted from design 3's tile-read pack, with a ROW-MAJOR
    store so the mask equals ch06's pack_kernel bit for bit.

    One warp owns one word column ty (lane j owns y = 32*ty + j) and
    walks tx = blockIdx.y, +gridDim.y. Each tile is read as 4 batches
    of 8 rows: 32 lanes read 96 contiguous bytes of one image row per
    load instruction, the pattern that took design 3's pack from 168
    to 195 GB/s. 8 ballots per batch turn the votes into row words.

    variant "direct": lane r stores row x0+r's word. The block's warps
    hold consecutive ty of the same tx, so a block covers a 32-byte
    sector of each row; partial writes merge in L2.
    variant "smem": the block stages a [32][8] word tile in shared
    memory and stores whole row sectors (8 words per row, 4 rows per
    warp instruction).
    variant "tilemajor": design 3's store, lane r writes word
    (t * 32 + r) of a flat tile-major array. A different layout that no
    later kernel reads: experiment E1's reference arm only.
    """
    smem = variant == "smem"
    tilemajor = variant == "tilemajor"

    @cuda.jit
    def pack_tiles(img, mask):
        width = img.shape[0]
        height = img.shape[1]
        wpr = (height + 31) // 32      # == mask.shape[1] when row-major
        tiles_x = (width + 31) // 32
        g = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ty = g >> 5
        lane = cuda.threadIdx.x & 31
        if not smem:
            if ty >= wpr:
                return              # warp-uniform
        y = ty * 32 + lane
        oky = y < height
        yc = min(y, height - 1)
        wm = width - 1
        if smem:
            sh = cuda.shared.array(256, dtype=uint32)
            wib = cuda.threadIdx.x >> 5
            ty0 = cuda.blockIdx.x * (cuda.blockDim.x >> 5)
            nwb = cuda.blockDim.x >> 5
        for tx in range(cuda.blockIdx.y, tiles_x, cuda.gridDim.y):
            x0 = tx * 32
            m = U0
            a0 = _red8(img, x0, yc, oky, wm)
            a1 = _red8(img, x0 + 8, yc, oky, wm)
            m = _ballot8(a0, lane, 0, m)
            m = _ballot8(a1, lane, 8, m)
            a2 = _red8(img, x0 + 16, yc, oky, wm)
            a3 = _red8(img, x0 + 24, yc, oky, wm)
            m = _ballot8(a2, lane, 16, m)
            m = _ballot8(a3, lane, 24, m)
            if smem:
                sh[lane * 8 + wib] = m
                cuda.syncthreads()
                for idx in range(cuda.threadIdx.x, 256, cuda.blockDim.x):
                    r = idx >> 3
                    wi = idx & 7
                    tw = ty0 + wi
                    if wi < nwb and tw < wpr and x0 + r < width:
                        mask[x0 + r, tw] = sh[idx]
                cuda.syncthreads()
            elif tilemajor:
                mask[(tx * wpr + ty) * 32 + lane, 0] = m
            else:
                if x0 + lane < width:
                    mask[x0 + lane, ty] = m

    return pack_tiles


_PACK_CACHE = {}


def get_pack_kernel(variant="direct"):
    if variant not in ("direct", "smem", "tilemajor"):
        raise ValueError(f"pack variant must be 'direct', 'smem' or "
                         f"'tilemajor', got {variant!r}")
    if variant not in _PACK_CACHE:
        _PACK_CACHE[variant] = _make_pack_kernel(variant)
    return _PACK_CACHE[variant]


def pack_shape(width, words_per_row):
    """((gridDim.x, gridDim.y), blockDim.x) for pack_tiles: a block is at
    most 8 warps (8 word columns), gridDim.y one block row per tile row
    (capped at 65535; the kernel strides past it)."""
    tpb = 32 * min(8, words_per_row)
    gx = (words_per_row * 32 + tpb - 1) // tpb
    tiles_x = (width + TILE - 1) // TILE
    return (gx, min(tiles_x, 65535)), tpb


# ------------------------------------------------------------- K0 init

@cuda.jit
def init_kernel(q_state, counters):
    """Zero the queue state and the counters. Nothing else needs a reset:
    every record, header, queue entry and parent slot a later kernel
    reads is written earlier in the same run."""
    i = cuda.threadIdx.x
    if i < q_state.shape[0]:
        q_state[i] = 0
    for c in range(i, counters.shape[0], cuda.blockDim.x):
        counters[c] = 0


# ------------------------------------------------------------ K2 label

def _make_label_kernel(conn8, instrumented, debug, max_registers=None,
                       index=False, exits=2):
    """K2: one warp per tile, grid-stride over every tile.

    Compile-time flags (closure variables): conn8, instrumented, debug,
    index (also write the frame-pixel ordinals and the tile's frame words
    for K3's index lookup), exits (E2: 2 both closure exits, 1 frontier
    only, 0 ch07's loop). The bare twin keeps only the interior-comp
    count (n_blobs needs it).
    """
    jit_kw = {} if max_registers is None else {"max_registers": max_registers}

    @cuda.jit(**jit_kw)
    def label_kernel(mask, hdr, rec, parent, qc, ql, qd, q_state, counters,
                     idx, frame, width, height, wpr, n_tiles, lattice):
        lane = cuda.laneid
        warp = cuda.grid(1) >> 5
        n_warps = cuda.gridsize(1) >> 5
        c_int = 0
        c_touched = 0
        c_comps = 0
        c_ring = 0
        c_max_comps = 0
        c_max_ring = 0
        c_iters = 0
        c_last = 0
        c_front = 0
        c_seeds = 0
        c_lat = 0
        c_queued = 0
        c_app = 0
        c_ovf = 0

        # software prefetch of the next tile's own word (proto3)
        t = warp
        m_next = U0
        if t < n_tiles:
            txn = int32(uint32(t) // uint32(wpr))
            xn = txn * 32 + lane
            if xn < width:
                m_next = mask[uint32(xn * wpr + (t - txn * wpr))]
        while t < n_tiles:
            tc = t
            m = m_next
            t += n_warps
            m_next = U0
            if t < n_tiles:
                txn = int32(uint32(t) // uint32(wpr))
                xn = txn * 32 + lane
                if xn < width:
                    m_next = mask[uint32(xn * wpr + (t - txn * wpr))]
            if not cuda.any_sync(FULL, m != U0):
                if lane == 0:
                    hdr[uint32(tc)] = 0
                continue
            tx = int32(uint32(tc) // uint32(wpr))
            ty = tc - tx * wpr
            x0 = tx * 32
            x = x0 + lane
            li = x * wpr + ty
            # ---- facing loads, issued now, consumed only at the flush
            wl = U0
            wr = U0
            side = U0
            if x < width:
                if ty > 0:
                    wl = mask[uint32(li - 1)]
                if ty + 1 < wpr:
                    wr = mask[uint32(li + 1)]
            if lane < 3:
                sw = ty - 1 + lane
                if tx > 0 and sw >= 0 and sw < wpr and (conn8 or lane == 1):
                    side = mask[uint32((x0 - 1) * wpr + sw)]
            elif lane >= 29:
                sw = ty + lane - 30
                if (x0 + 32 < width and sw >= 0 and sw < wpr
                        and (conn8 or lane == 30)):
                    side = mask[uint32((x0 + 32) * wpr + sw)]
            lat_hit = lattice > 0 and (tx + ty) % lattice == 0

            # ---- comp loop, warp-uniform
            rem = m
            b = 0
            n_total = 0
            rN = U0
            rS = U0
            rW = U0
            rE = U0
            ranchor = int32(0)
            id_base = IDS_PER_TILE * tc
            while cuda.any_sync(FULL, rem != U0):
                f, a, bit = _first_pixel(rem, lane)
                anchor = int32((x0 + a) * height + ty * 32 + bit)
                f, it, last = _close_exits(f, rem, lane, conn8, exits)
                rem = uint32(rem & ~f)
                n_total += 1
                if instrumented:
                    c_iters += it
                    c_last += last
                    c_front += 1 - last
                if _is_ring(f, lane):
                    cuda.syncwarp()
                    sN = _bcast(f, 0)
                    sS = _bcast(f, 31)
                    sW = uint32(cuda.ballot_sync(FULL, (f & U1) != U0))
                    sE = uint32(cuda.ballot_sync(FULL, (f >> S31) != U0))
                    if index and b >= 1:
                        # lane j records the ordinal of every frame pixel
                        # j this comp owns, per side (N, S, Wc, Ec). Only
                        # tiles with 2+ ring comps get an index: K3 takes a
                        # single ring comp directly. So comp 0's entries
                        # are written when comp 1 appears (lane 0 still
                        # holds comp 0's side words then).
                        ib = uint32(INDEX_BYTES * tc + lane)
                        if b == 1:
                            cuda.syncwarp()
                            zN = _bcast(rN, 0)
                            zS = _bcast(rS, 0)
                            zW = _bcast(rW, 0)
                            zE = _bcast(rE, 0)
                            b0 = np.uint8(0)
                            if (zN >> uint32(lane)) & U1:
                                idx[ib] = b0
                            if (zS >> uint32(lane)) & U1:
                                idx[ib + 32] = b0
                            if (zW >> uint32(lane)) & U1:
                                idx[ib + 64] = b0
                            if (zE >> uint32(lane)) & U1:
                                idx[ib + 96] = b0
                        bv = np.uint8(b)
                        if (sN >> uint32(lane)) & U1:
                            idx[ib] = bv
                        if (sS >> uint32(lane)) & U1:
                            idx[ib + 32] = bv
                        if (sW >> uint32(lane)) & U1:
                            idx[ib + 64] = bv
                        if (sE >> uint32(lane)) & U1:
                            idx[ib + 96] = bv
                    if lane == (b & 31):
                        rN = sN
                        rS = sS
                        rW = sW
                        rE = sE
                        ranchor = anchor
                    b += 1
                    if (b & 31) == 0:       # only tiles with 32+ ring comps
                        s, sl, q, ap = _flush(
                            lane, 32, id_base + b - 32, rN, rS, rW, rE,
                            ranchor, side, wl, wr, lat_hit, rec, parent,
                            qc, ql, qd, q_state, conn8)
                        if instrumented:
                            c_seeds += s
                            c_lat += sl
                            c_queued += q
                            c_app += ap
                else:
                    c_int += 1
            if (b & 31) != 0:
                s, sl, q, ap = _flush(
                    lane, b & 31, id_base + (b & ~31), rN, rS, rW, rE,
                    ranchor, side, wl, wr, lat_hit, rec, parent, qc, ql,
                    qd, q_state, conn8)
                if instrumented:
                    c_seeds += s
                    c_lat += sl
                    c_queued += q
                    c_app += ap
            if index and b >= 2:
                cuda.syncwarp()
                fN = _bcast(m, 0)
                fS = _bcast(m, 31)
                fW = uint32(cuda.ballot_sync(FULL, (m & U1) != U0))
                fE = uint32(cuda.ballot_sync(FULL, (m >> S31) != U0))
                if lane == 0:
                    fb = uint32(4 * tc)
                    frame[fb] = fN
                    frame[fb + 1] = fS
                    frame[fb + 2] = fW
                    frame[fb + 3] = fE
            if lane == 0:
                hdr[uint32(tc)] = b | (n_total << HDR_TOTAL_SHIFT)
            if instrumented:
                c_touched += 1
                c_comps += n_total
                c_ring += b
                c_max_comps = max(c_max_comps, n_total)
                c_max_ring = max(c_max_ring, b)
            if debug:
                if b > 62:
                    c_ovf += 1

        # ---- one atomic per warp per counter
        if lane == 0 and c_int:
            cuda.atomic.add(counters, INTERIOR_COMPS, c_int)
        if instrumented:
            seeds = _warp_sum(np.int64(c_seeds))
            lat = _warp_sum(np.int64(c_lat))
            queued = _warp_sum(np.int64(c_queued))
            if lane == 0:
                cuda.atomic.add(counters, TILES_TOUCHED, c_touched)
                cuda.atomic.add(counters, COMPS, c_comps)
                cuda.atomic.add(counters, RING_COMPS, c_ring)
                cuda.atomic.max(counters, MAX_COMPS_TILE, c_max_comps)
                cuda.atomic.max(counters, MAX_RING_TILE, c_max_ring)
                cuda.atomic.add(counters, CLOSE_ITERS, c_iters)
                cuda.atomic.add(counters, EXIT_LAST, c_last)
                cuda.atomic.add(counters, EXIT_FRONTIER, c_front)
                cuda.atomic.add(counters, SEEDS, seeds)
                cuda.atomic.add(counters, SEEDS_LATTICE, lat)
                cuda.atomic.add(counters, SEEDS_QUEUED, queued)
                cuda.atomic.add(counters, SEED_APPEND_ATOMICS, c_app)
        if debug:
            if lane == 0 and c_ovf:
                cuda.atomic.add(counters, ORDINAL_OVERFLOW, c_ovf)

    return label_kernel


# ------------------------------------------------------------ K3 waves

def _make_waves_kernel(conn8, instrumented, debug, index=False,
                       hybrid=False, max_registers=None):
    """K3: cooperative. Rounds of claims over the comp graph, one
    grid.sync per round (three rotating append counters), the escalation
    valve, one union pass over the recorded collisions, two flatten
    passes. Launched with the cooperative maximum of THIS twin.

    Neighbour lookup (compile-time `index`):
      scan   test every ring comp of the neighbour tile against the
             wanted bits: n_ring(t2) k-steps per task
      index  hit = wanted bits & the neighbour's frame word; per step,
             the lowest hit bit's ordinal (K2's byte index) names the
             comp, and all of that comp's bits leave `hit`: one step per
             DISTINCT adjacent comp, at the price of a dependent chain
             (frame word, ordinal, record)
      hybrid scan for a neighbour tile with at most dense_min ring
             comps; above that the whole warp takes the task: lane k
             tests ring comp k (one coalesced step per 32 comps), one
             dense task after another. The k-steps of the sparse lanes
             stay bounded by dense_min, and K2 is the scan's.

    dense_min is a runtime argument; the scan and index twins ignore
    it."""
    link = [SMID_CU] if instrumented else []
    jit_kw = {} if max_registers is None else {"max_registers": max_registers}

    @cuda.jit(link=link, **jit_kw)
    def waves_kernel(hdr, rec, parent, qc, ql, qd, ca, cb, q_state,
                     counters, bounds, stamps, idx, frame, wpr, n_tiles,
                     valve_round, coll_cap, dense_min):
        grid = cuda.cg.this_grid()
        tid = cuda.grid(1)
        stride = cuda.gridsize(1)
        lane = cuda.laneid
        n_bounds = bounds.shape[0]
        c_tasks = 0
        c_dense = 0
        c_ksteps = 0
        c_tests = 0
        c_cas = 0
        c_claims = 0
        c_coll = 0
        c_inline = 0
        c_pskip = 0
        c_app = 0
        c_capp = 0
        c_ucalls = 0
        c_udone = 0
        c_rmin = 0
        c_roots = 0
        c_unl = 0

        front = 0
        rear = q_state[Q_SEEDS]         # final: written by K2's launch
        r = 1
        peak = rear
        if instrumented and tid == 0:
            stamps[0] = get_globaltimer()
            bounds[0] = rear
        while front < rear:
            slot = Q_C0 + r % 3
            if rear - front > peak:
                peak = rear - front
            n_tasks = (rear - front) << 3
            n_round = (n_tasks + 31) & ~31   # whole warps: uniform trips
            for i in range(tid, n_round, stride):
                valid = i < n_tasks
                e = front + (i >> 3)
                d = i & 7                     # 8 lanes share one entry
                dirs = int32(0)
                c = int32(0)
                lab = int32(0)
                if valid:
                    dirs = qd[uint32(e)]
                    c = qc[uint32(e)]
                    lab = ql[uint32(e)]
                valid = valid and ((dirs >> d) & 1) != 0
                n2 = 0
                t2 = 0
                want = U0
                hits = U0
                fs = (FACING >> (2 * d)) & 3
                succ = ((SUCC >> d) & 1) != 0
                if valid:
                    dx = ((DXP >> d) & 1) - ((DXM >> d) & 1)
                    dy = ((DYP >> d) & 1) - ((DYM >> d) & 1)
                    t2 = (c >> 6) + dx * wpr + dy
                    own = uint32(rec[uint32(REC * c + ((MY_SIDE >> (2 * d))
                                                       & 3))])
                    if d < 4:
                        want = _dilate(own) if conn8 else own
                    elif d == 4 or d == 6:
                        want = uint32((own & U1) << S31)
                    else:
                        want = uint32(own >> S31)
                    n2 = hdr[uint32(t2)] & HDR_RING_MASK
                    if index:
                        # both loads independent; the frame word is only
                        # written (and only used) for tiles with 2+ ring
                        # comps, a single ring comp is ordinal 0
                        fw = frame[uint32(4 * t2 + fs)]
                        if n2 == 1:
                            hits = U1
                        else:
                            hits = uint32(want & fw)
                dense = False
                if hybrid:
                    dense = valid and n2 > dense_min
                if instrumented and valid:
                    c_tasks += 1
                    c_dense += int32(dense)
                    if not index and not dense:
                        c_tests += n2
                k = 0
                if index:
                    active = hits != U0
                else:
                    active = k < n2 and not dense
                while cuda.any_sync(FULL, active):    # uniform k-steps
                    hit = False
                    c2 = IDS_PER_TILE * t2 + k
                    d2 = int32(0)
                    if index:
                        if hits != U0:
                            if n2 == 1:
                                c2 = IDS_PER_TILE * t2
                                hits = U0
                            else:
                                j = cuda.ffs(hits) - 1
                                b2 = int32(idx[uint32(INDEX_BYTES * t2
                                                      + 32 * fs + j)])
                                c2 = IDS_PER_TILE * t2 + b2
                                s2 = uint32(rec[uint32(REC * c2 + fs)])
                                hits = uint32(hits & ~s2)
                            hit = True
                            d2 = rec[uint32(REC * c2 + R_DIRS)]
                            if instrumented:
                                c_tests += 1
                    else:
                        if k < n2:
                            s2 = uint32(rec[uint32(REC * c2 + fs)])
                            hit = uint32(s2 & want) != U0
                            if hit:
                                d2 = rec[uint32(REC * c2 + R_DIRS)]
                    # (written out, not through _claim: the scan twin
                    # keeps 80 registers and 3 cooperative blocks per SM)
                    old = int32(0)
                    if hit:
                        old = cuda.atomic.cas(rec, uint32(REC * c2 + R_LABEL),
                                              NEG1, lab)
                    claimed = hit and old == -1
                    collided = hit and old != -1 and old != lab and succ
                    pos, na = _warp_slot(claimed, lane, q_state, slot)
                    if claimed:
                        p = rear + pos
                        qc[uint32(p)] = c2
                        ql[uint32(p)] = lab
                        qd[uint32(p)] = d2
                    cpos, cna = _warp_slot(collided, lane, q_state, Q_COLL)
                    if collided:
                        if cpos < coll_cap:
                            ca[uint32(cpos)] = lab
                            cb[uint32(cpos)] = old
                        else:
                            # past the soft cap: union now (ch06's _union
                            # is lock-free and needs no warp sync)
                            done = _union(parent, lab, old)
                            if instrumented:
                                c_udone += done
                                c_ucalls += 1
                            c_inline += 1
                    if instrumented:
                        c_cas += int32(hit)
                        c_claims += int32(claimed)
                        c_coll += int32(collided)
                        c_pskip += int32(hit and old != -1 and old != lab
                                         and not succ)
                        if lane == 0:
                            c_ksteps += 1          # warp k-steps
                            c_app += na
                            c_capp += cna
                    k += 1
                    if index:
                        active = hits != U0
                    else:
                        active = k < n2 and not dense
                if hybrid:
                    # dense tasks, one at a time, by the whole warp: lane
                    # j tests ring comp base + j of the leader's tile
                    pend = dense
                    while cuda.any_sync(FULL, pend):
                        cuda.syncwarp()
                        bal = uint32(cuda.ballot_sync(FULL, pend))
                        ld = int32(cuda.ffs(bal) - 1)
                        if lane == ld:
                            pend = False
                        t2l = cuda.shfl_sync(FULL, int32(t2), ld)
                        wl = _bcast(want, ld)
                        fsl = cuda.shfl_sync(FULL, int32(fs), ld)
                        labl = cuda.shfl_sync(FULL, lab, ld)
                        n2l = cuda.shfl_sync(FULL, int32(n2), ld)
                        sl = cuda.shfl_sync(FULL, int32(succ), ld) != 0
                        base = 0
                        while base < n2l:              # uniform
                            kk = base + lane
                            c2 = IDS_PER_TILE * t2l + kk
                            hit = False
                            d2 = int32(0)
                            if kk < n2l:
                                s2 = uint32(rec[uint32(REC * c2 + fsl)])
                                hit = uint32(s2 & wl) != U0
                                if hit:
                                    d2 = rec[uint32(REC * c2 + R_DIRS)]
                            claimed, collided, inl, done, pskip, na, cna = \
                                _claim(hit, c2, d2, labl, sl, lane, rec,
                                       parent, qc, ql, qd, ca, cb, q_state,
                                       slot, rear, coll_cap)
                            c_inline += inl
                            if instrumented:
                                c_tests += int32(kk < n2l)
                                c_udone += done
                                c_ucalls += inl
                                c_cas += int32(hit)
                                c_claims += int32(claimed)
                                c_coll += int32(collided)
                                c_pskip += int32(pskip)
                                if lane == 0:
                                    c_ksteps += 1
                                    c_app += na
                                    c_capp += cna
                            base += 32
            grid.sync()
            n_new = q_state[Q_C0 + r % 3]
            if tid == 0:
                q_state[Q_C0 + (r + 2) % 3] = 0   # read by all before this
            if n_new > 0 and r == valve_round:
                # escalation valve: self-seed every ring comp that no wave
                # has reached; no claim runs now, so plain stores suffice
                for tt in range(tid, n_tiles, stride):
                    nr = hdr[uint32(tt)] & HDR_RING_MASK
                    for kk in range(nr):
                        cid = IDS_PER_TILE * tt + kk
                        w = uint32(REC * cid)
                        if rec[w + R_LABEL] == -1:
                            rec[w + R_LABEL] = cid
                            rec[w + R_RMIN] = rec[w + R_ANCHOR]
                            parent[uint32(cid)] = cid
                            dv = rec[w + R_DIRS]
                            if dv != 0:
                                p = rear + n_new + cuda.atomic.add(
                                    q_state, Q_VALVE, 1)
                                qc[uint32(p)] = cid
                                ql[uint32(p)] = cid
                                qd[uint32(p)] = dv
                grid.sync()
                n_valve = q_state[Q_VALVE]
                if tid == 0:
                    counters[VALVE_ROUND] = r
                    counters[VALVE_SEEDS] = n_valve
                n_new += n_valve
            front = rear
            rear += n_new
            if instrumented and tid == 0:
                stamps[min(r, ST_TAIL)] = get_globaltimer()
                if r < n_bounds:
                    bounds[r] = rear
            r += 1

        # ---- union pass over the recorded collisions
        n_coll = min(q_state[Q_COLL], coll_cap)
        for j in range(tid, n_coll, stride):
            c_udone += _union(parent, ca[uint32(j)], cb[uint32(j)])
            if instrumented:
                c_ucalls += 1
        grid.sync()
        if instrumented and tid == 0:
            stamps[ST_UNION] = get_globaltimer()
        # ---- flatten 1: root per ring comp, min anchor into the root
        for tt in range(tid, n_tiles, stride):
            nr = hdr[uint32(tt)] & HDR_RING_MASK
            for kk in range(nr):
                cid = IDS_PER_TILE * tt + kk
                w = uint32(REC * cid)
                lab = rec[w + R_LABEL]
                if debug:
                    if lab < 0:
                        c_unl += 1
                        continue
                root = _find(parent, lab)
                rec[w + R_LABEL] = root
                if root == cid:
                    c_roots += 1
                else:
                    cuda.atomic.min(rec, uint32(REC * root + R_RMIN),
                                    rec[w + R_ANCHOR])
                    if instrumented:
                        c_rmin += 1
        grid.sync()
        if instrumented and tid == 0:
            stamps[ST_FLAT1] = get_globaltimer()
        # ---- flatten 2: the canonical label = the root's rmin
        for tt in range(tid, n_tiles, stride):
            nr = hdr[uint32(tt)] & HDR_RING_MASK
            for kk in range(nr):
                w = uint32(REC * (IDS_PER_TILE * tt + kk))
                root = rec[w + R_LABEL]
                if debug:
                    if root < 0:
                        continue
                rec[w + R_LABEL] = rec[uint32(REC * root + R_RMIN)]
        if instrumented and tid == 0:
            stamps[ST_FLAT2] = get_globaltimer()

        # ---- counters: warp-reduced, one atomic per warp
        roots = _warp_sum(np.int64(c_roots))
        inline = _warp_sum(np.int64(c_inline))
        if lane == 0:
            if roots:
                cuda.atomic.add(counters, ROOTS, roots)
            if inline:
                cuda.atomic.add(counters, COLL_INLINE, inline)
        if instrumented:
            v_tasks = _warp_sum(np.int64(c_tasks))
            v_dense = _warp_sum(np.int64(c_dense))
            v_ksteps = _warp_sum(np.int64(c_ksteps))
            v_tests = _warp_sum(np.int64(c_tests))
            v_cas = _warp_sum(np.int64(c_cas))
            v_claims = _warp_sum(np.int64(c_claims))
            v_coll = _warp_sum(np.int64(c_coll))
            v_pskip = _warp_sum(np.int64(c_pskip))
            v_ucalls = _warp_sum(np.int64(c_ucalls))
            v_udone = _warp_sum(np.int64(c_udone))
            v_rmin = _warp_sum(np.int64(c_rmin))
            if lane == 0:
                cuda.atomic.add(counters, TASKS, v_tasks)
                cuda.atomic.add(counters, DENSE_TASKS, v_dense)
                cuda.atomic.add(counters, KSTEPS, v_ksteps)
                cuda.atomic.add(counters, SIG_TESTS, v_tests)
                cuda.atomic.add(counters, CAS, v_cas)
                cuda.atomic.add(counters, CLAIMS, v_claims)
                cuda.atomic.add(counters, COLLISIONS, v_coll)
                cuda.atomic.add(counters, PRED_SKIPS, v_pskip)
                cuda.atomic.add(counters, APPEND_ATOMICS, c_app)
                cuda.atomic.add(counters, COLL_APPEND_ATOMICS, c_capp)
                cuda.atomic.add(counters, UNION_CALLS, v_ucalls)
                cuda.atomic.add(counters, UNION_DONE, v_udone)
                cuda.atomic.add(counters, RMIN_ATOMICS, v_rmin)
        if debug:
            unl = _warp_sum(np.int64(c_unl))
            if lane == 0 and unl:
                cuda.atomic.add(counters, UNLABELLED, unl)
        if tid == 0:
            counters[ROUNDS] = r - 1
            if instrumented:
                counters[PEAK_FRONTIER] = peak

    return waves_kernel


# ------------------------------------------------------------ K4 paint

MODE_RGB = 0       # palette[label % 6] on red
MODE_LABEL = 1     # int32 label map (off the clock)
MODE_SEEDS = 2     # the anchor of every comp whose label equals it


@cuda.jit(device=True, inline=True)
def _paint_rgb(g, x0, y, lane, img, p0, p1, p2):
    """Paint the fill g: per set row r, the 32 lanes write the 96
    contiguous bytes of (x0+r, 32*ty .. 32*ty+31). Returns rows."""
    cuda.syncwarp()
    rows = uint32(cuda.ballot_sync(FULL, g != U0))
    n = 0
    while rows != U0:
        cuda.syncwarp()
        r = int32(cuda.ffs(rows) - 1)
        rows = uint32(rows & (rows - U1))
        w = _bcast(g, r)
        if ((w >> uint32(lane)) & U1) != U0:
            img[x0 + r, y, 0] = p0
            img[x0 + r, y, 1] = p1
            img[x0 + r, y, 2] = p2
        n += 1
    return n


@cuda.jit(device=True, inline=True)
def _paint_label(g, x0, y, lane, label_map, lab):
    cuda.syncwarp()
    rows = uint32(cuda.ballot_sync(FULL, g != U0))
    n = 0
    while rows != U0:
        cuda.syncwarp()
        r = int32(cuda.ffs(rows) - 1)
        rows = uint32(rows & (rows - U1))
        w = _bcast(g, r)
        if ((w >> uint32(lane)) & U1) != U0:
            label_map[x0 + r, y] = lab
        n += 1
    return n


def _make_paint_kernel(conn8, mode, fast, instrumented, debug):
    """K4 and its off-clock twins: one warp per tile, grid-stride over
    every tile in t order.

    mode 0 (paint) has two fast paths (compile-time `fast`):
      colour-uniform  every comp is a ring comp and all their labels
                      give one palette colour: paint the whole mask word
                      in that colour, no closure
      single interior one interior comp and no ring comp: its anchor is
                      the lowest pixel, found by two ffs
    Everything else re-closes the tile with K2's comp loop, in the same
    order, so ring comp b takes record 64*t + b.
    """

    @cuda.jit
    def paint_kernel(mask, hdr, rec, img, label_map, seeds, seed_n,
                     counters, width, height, wpr, n_tiles):
        palette = cuda.const.array_like(PALETTE_HOST)
        lane = cuda.laneid
        warp = cuda.grid(1) >> 5
        n_warps = cuda.gridsize(1) >> 5
        c_tiles = 0
        c_uni = 0
        c_single = 0
        c_recl = 0
        c_iters = 0
        c_rows = 0
        c_mis = 0
        for t in range(warp, n_tiles, n_warps):
            h = hdr[uint32(t)]
            if h == 0:
                continue
            nr = h & HDR_RING_MASK
            nt = h >> HDR_TOTAL_SHIFT
            tx = int32(uint32(t) // uint32(wpr))
            ty = t - tx * wpr
            x0 = tx * 32
            x = x0 + lane
            m = U0
            if x < width:
                m = mask[uint32(x * wpr + ty)]
            y = ty * 32 + lane
            id_base = IDS_PER_TILE * t
            lab0 = int32(0)
            lab1 = int32(0)
            if lane < nr:
                lab0 = rec[uint32(REC * (id_base + lane) + R_LABEL)]
            if lane + 32 < nr:
                lab1 = rec[uint32(REC * (id_base + 32 + lane) + R_LABEL)]
            if instrumented:
                c_tiles += 1
            if mode == MODE_RGB and fast:
                if nt == nr:
                    bits = int32(0)
                    if lane < nr:
                        bits = int32(1) << (lab0 % N_PALETTE)
                    if lane + 32 < nr:
                        bits |= int32(1) << (lab1 % N_PALETTE)
                    bits |= cuda.shfl_xor_sync(FULL, bits, 16)
                    bits |= cuda.shfl_xor_sync(FULL, bits, 8)
                    bits |= cuda.shfl_xor_sync(FULL, bits, 4)
                    bits |= cuda.shfl_xor_sync(FULL, bits, 2)
                    bits |= cuda.shfl_xor_sync(FULL, bits, 1)
                    if (bits & (bits - 1)) == 0:
                        col = int32(cuda.ffs(bits) - 1)
                        n = _paint_rgb(m, x0, y, lane, img, palette[col, 0],
                                       palette[col, 1], palette[col, 2])
                        if instrumented:
                            c_uni += 1
                            c_rows += n
                        continue
                elif nt == 1:
                    f, a, bit = _first_pixel(m, lane)
                    anchor = (x0 + a) * height + ty * 32 + bit
                    col = anchor % N_PALETTE
                    n = _paint_rgb(m, x0, y, lane, img, palette[col, 0],
                                   palette[col, 1], palette[col, 2])
                    if instrumented:
                        c_single += 1
                        c_rows += n
                    continue
            # ---- re-close, in K2's order
            if instrumented:
                c_recl += 1
            rem = m
            b = 0
            while cuda.any_sync(FULL, rem != U0):
                f, a, bit = _first_pixel(rem, lane)
                anchor = int32((x0 + a) * height + ty * 32 + bit)
                f, it, last = _close(f, rem, lane, conn8)
                rem = uint32(rem & ~f)
                if instrumented:
                    c_iters += it
                lab = anchor
                if _is_ring(f, lane):
                    cuda.syncwarp()
                    if b < 32:
                        lab = cuda.shfl_sync(FULL, lab0, int32(b))
                    else:
                        lab = cuda.shfl_sync(FULL, lab1, int32(b - 32))
                    if debug:
                        ra = rec[uint32(REC * (id_base + b) + R_ANCHOR)]
                        if lane == 0 and ra != anchor:
                            c_mis += 1
                    b += 1
                if mode == MODE_RGB:
                    col = lab % N_PALETTE
                    n = _paint_rgb(f, x0, y, lane, img, palette[col, 0],
                                   palette[col, 1], palette[col, 2])
                    if instrumented:
                        c_rows += n
                elif mode == MODE_LABEL:
                    _paint_label(f, x0, y, lane, label_map, lab)
                else:
                    if lab == anchor and lane == 0:
                        p = cuda.atomic.add(seed_n, 0, 1)
                        if p < seeds.shape[0]:
                            seeds[p] = anchor
        if instrumented and lane == 0:
            cuda.atomic.add(counters, PAINT_TILES, c_tiles)
            cuda.atomic.add(counters, PAINT_UNIFORM, c_uni)
            cuda.atomic.add(counters, PAINT_SINGLE_INTERIOR, c_single)
            cuda.atomic.add(counters, PAINT_RECLOSED_TILES, c_recl)
            cuda.atomic.add(counters, PAINT_RECLOSE_ITERS, c_iters)
            cuda.atomic.add(counters, ROWS_PAINTED, c_rows)
        if debug and lane == 0 and c_mis:
            cuda.atomic.add(counters, ANCHOR_MISMATCH, c_mis)

    return paint_kernel


# --------------------------------------------------------- factories

_LABEL_CACHE = {}
_WAVES_CACHE = {}
_PAINT_CACHE = {}


def get_label_kernel(connectivity=8, instrumented=False, debug=False,
                     max_registers=None, index=False, exits=2):
    key = (connectivity, bool(instrumented), bool(debug), max_registers,
           bool(index), int(exits))
    if key not in _LABEL_CACHE:
        _LABEL_CACHE[key] = _make_label_kernel(
            connectivity == 8, bool(instrumented), bool(debug), max_registers,
            bool(index), int(exits))
    return _LABEL_CACHE[key]


def get_waves_kernel(connectivity=8, instrumented=False, debug=False,
                     index=False, hybrid=False, max_registers=None):
    if index and hybrid:
        raise ValueError("index and hybrid are separate lookups")
    key = (connectivity, bool(instrumented), bool(debug), bool(index),
           bool(hybrid), max_registers)
    if key not in _WAVES_CACHE:
        _WAVES_CACHE[key] = _make_waves_kernel(
            connectivity == 8, bool(instrumented), bool(debug), bool(index),
            bool(hybrid), max_registers)
    return _WAVES_CACHE[key]


def get_paint_kernel(connectivity=8, mode=MODE_RGB, fast=True,
                     instrumented=False, debug=False):
    key = (connectivity, mode, bool(fast), bool(instrumented), bool(debug))
    if key not in _PAINT_CACHE:
        _PAINT_CACHE[key] = _make_paint_kernel(
            connectivity == 8, mode, bool(fast), bool(instrumented),
            bool(debug))
    return _PAINT_CACHE[key]


@cuda.jit
def fill_i32(a, v):
    """a[...] = v over a flat int32 view (label map init, buffer poison)."""
    for i in range(cuda.grid(1), a.shape[0], cuda.gridsize(1)):
        a[i] = v
