"""Triton twins of the dual-blob BFS kernels: labeled queue entries,
per-blob colors.

Same algorithm as chapters/ch04_gpu_2blob_nblock/kernels.py, kernel for
kernel: ch03's cooperative global-queue BFS with two changes. The seed
count arrives as a launch argument (``rear = n_seeds``), and the blob label
rides inside the int32 queue entry, so each dequeued pixel is painted with
its own blob's color (label 0 blue, label 1 green). Two entry formats are
twin families:

    lin   entry = (x*height + y) << 1 | label   (int64 div/mod decode)
    xy    entry = label << 26 | x << 13 | y     (shift-and-mask decode)

The ten Numba kernels ({lin, xy} x {instrumented, bare} x {4, 8}-conn, plus
the guarded radius-2 pair, lin 8-conn only) keep their names and argument
lists here. Each is a thin @triton.jit wrapper around one body,
``_dual_blob``, whose constexpr flags (FMT, CONN, RADIUS, INSTR) pick the
variant. A constexpr ``if`` is resolved while compiling, so every Numba
kernel has its own binary and the bare twins carry none of the
instrumentation code (no owner map, no per-lane counters, no level trace,
no %smid read).

Mapping (one Numba block of T threads = one program of T lanes, num_warps =
T // 32, lane i plays thread i):

- grid.sync()              -> runtime.device.grid_sync on a per-launch int64
                              counter (monotonic epochs, two per level as in
                              Numba); the host launches with
                              launch_cooperative_grid=True, so an oversized
                              grid is refused by the driver as in Numba
- cuda.atomic.cas(visited, 0, 1) == 0
                           -> masked tl.atomic_xchg(visited, 1), old == 0:
                              the same exactly-once claim on a 0/1 flag,
                              relaxed like Numba's atom.cas (Triton's CAS
                              takes no mask)
- _warp_enqueue_global     -> _block_enqueue_global: aggregated per program
                              (tl.sum + tl.cumsum + one atomic), because
                              Triton has no activemask/popc/shfl
- per-thread int64 counters + exit atomics
                           -> per-lane int64 tensors + the same per-lane
                              atomics at exit
- cuda.const.array_like    -> constexpr tuples indexed by tl.static_range
                              (direction tables) or selected by label
                              (palette)
- get_smid (smid.cu)       -> runtime.device.read_smid (inline PTX)

Everything load-bearing is the Numba kernels' own: the claim protocol
(bounds -> is_red -> claim visited -> enqueue), two barriers per level,
the seed count as a launch argument (never read from q_state at kernel
start: see the Numba kernels.py docstring), every counter slot and its
meaning, and the no-overflow argument with its tripwire.
"""

import numpy as np
import triton
import triton.language as tl

from ...runtime.device import grid_sync, read_smid

# 4-connectivity offsets (right, down, left, up), as in the Numba chapter
DX_HOST = np.array([1, 0, -1, 0], dtype=np.int32)
DY_HOST = np.array([0, 1, 0, -1], dtype=np.int32)

# 8-connectivity offsets (E SE S SW W NW N NE)
DX8_HOST = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
DY8_HOST = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# Ring-2 offsets: the 16 cells at Chebyshev distance exactly 2, clockwise
# from E. Probed only when all 8 ring-1 neighbors are in-bounds blob
# material (the guard keeps every jump inside the dequeuer's component).
DX_R2_HOST = np.array([2, 2, 2, 1, 0, -1, -2, -2,
                       -2, -2, -2, -1, 0, 1, 2, 2], dtype=np.int32)
DY_R2_HOST = np.array([0, 1, 2, 2, 2, 2, 2, 1,
                       0, -1, -2, -2, -2, -2, -2, -1], dtype=np.int32)

# Per-blob fill colors, indexed by label: 0 -> blue, 1 -> green.
# Neither may be RED (255,0,0): painted pixels must stop matching _is_red.
PALETTE_HOST = np.array([[0, 0, 255],
                         [0, 255, 0]], dtype=np.uint8)

# xy-family field layout: entry = label << 26 | x << 13 | y
XY_FIELD_BITS = 13
XY_FIELD_MASK = (1 << XY_FIELD_BITS) - 1   # 0x1FFF
XY_LBL_SHIFT = 2 * XY_FIELD_BITS           # 26
XY_MAX_DIM = 1 << XY_FIELD_BITS            # 8192

# Slots in the device-side int64 counters array (identical to Numba's)
FILLED = 0
LEVELS = 1
OVERFLOW = 2            # defensive tripwire: structurally unreachable
PEAK_LEVEL = 3          # largest single frontier
PEAK_OCC = 4            # max queue entries alive across two adjacent levels
ACTIVE_THREAD_SUM = 5   # sum over levels of min(level_size, grid threads)
ACTIVE_WARP_SUM = 6     # sum over levels of ceil(min(level_size, grid)/32)
PROCESSED = 7           # pixels dequeued/recolored (== FILLED iff exactly-once)
CAS_ATTEMPTS = 8        # visited claims tried
INTERIOR = 9            # radius-2 twins: pixels whose ring-1 was all blob
NUM_COUNTERS = 10

# Columns of the (blocks, 2) int64 block_stats array
BS_PROCESSED = 0        # pixels this program dequeued -> N-way load balance
BS_SMID = 1             # %smid this program observed itself running on

# q_state slots
Q_REAR = 0

FMT_LIN = 0
FMT_XY = 1

# Device-side twins of the const arrays: compile-time tuples, so every
# offset and color folds into an immediate.
DX4 = tl.constexpr(tuple(int(v) for v in DX_HOST))
DY4 = tl.constexpr(tuple(int(v) for v in DY_HOST))
DX8 = tl.constexpr(tuple(int(v) for v in DX8_HOST))
DY8 = tl.constexpr(tuple(int(v) for v in DY8_HOST))
DX_R2 = tl.constexpr(tuple(int(v) for v in DX_R2_HOST))
DY_R2 = tl.constexpr(tuple(int(v) for v in DY_R2_HOST))
PAL0 = tl.constexpr(tuple(int(v) for v in PALETTE_HOST[0]))
PAL1 = tl.constexpr(tuple(int(v) for v in PALETTE_HOST[1]))

# The same slots as compile-time constants for the device code.
_FILLED = tl.constexpr(FILLED)
_LEVELS = tl.constexpr(LEVELS)
_OVERFLOW = tl.constexpr(OVERFLOW)
_PEAK_LEVEL = tl.constexpr(PEAK_LEVEL)
_PEAK_OCC = tl.constexpr(PEAK_OCC)
_ACTIVE_THREAD_SUM = tl.constexpr(ACTIVE_THREAD_SUM)
_ACTIVE_WARP_SUM = tl.constexpr(ACTIVE_WARP_SUM)
_PROCESSED = tl.constexpr(PROCESSED)
_CAS_ATTEMPTS = tl.constexpr(CAS_ATTEMPTS)
_INTERIOR = tl.constexpr(INTERIOR)
_BS_PROCESSED = tl.constexpr(BS_PROCESSED)
_BS_SMID = tl.constexpr(BS_SMID)
_Q_REAR = tl.constexpr(Q_REAR)
_XY_FIELD_BITS = tl.constexpr(XY_FIELD_BITS)
_XY_FIELD_MASK = tl.constexpr(XY_FIELD_MASK)
_XY_LBL_SHIFT = tl.constexpr(XY_LBL_SHIFT)

# Runtime ints that vary between calls: never specialize on them (n_seeds
# is 1 or 2, and a new ==1 or divisibility-by-16 class of a size would
# otherwise recompile inside a timed launch).
_DNS = ["n_seeds", "width", "height", "qcap", "trace_cap"]
_DNS_BARE = ["n_seeds", "width", "height", "qcap"]


@triton.jit
def _is_red(img_ptr, pix, inb):
    """img[x, y] == (255, 0, 0) on the lanes in ``inb``. The chained masks
    are the CUDA short-circuit: a channel is loaded only if every earlier
    channel matched."""
    p = img_ptr + pix * 3
    r = tl.load(p, mask=inb, other=0)
    m_r = inb & (r == 255)
    g = tl.load(p + 1, mask=m_r, other=1)
    m_g = m_r & (g == 0)
    b = tl.load(p + 2, mask=m_g, other=1)
    return m_g & (b == 0)


@triton.jit
def _block_enqueue_global(queue_ptr, q_state_ptr, counters_ptr, qcap, item,
                          won):
    """Program-aggregated append on a global rear counter (one atomic per
    program, the twin of the warp-aggregated helper).

    The winning lanes are ranked by an exclusive prefix sum; one relaxed
    atomic reserves a slab of ``count`` slots and its result reaches every
    lane. The scan stays unconditional (Triton 3.7 miscompiles a scan
    inside an ``if``); only the atomic is masked, and a program with no
    winner issues none. The bound check is the same defensive tripwire.
    """
    w = won.to(tl.int32)
    rank = tl.cumsum(w, 0) - w
    count = tl.sum(w, 0)
    base = tl.atomic_add(q_state_ptr + _Q_REAR, count, mask=count > 0,
                         sem="relaxed", scope="gpu")
    idx = base + rank
    tl.store(queue_ptr + idx, item, mask=won & (idx < qcap))
    # unreachable by the structural argument
    tl.store(counters_ptr + _OVERFLOW + rank * 0,
             (rank * 0 + 1).to(tl.int64), mask=won & (idx >= qcap))


@triton.jit
def _probe(img_ptr, visited_ptr, queue_ptr, q_state_ptr, counters_ptr, qcap,
           nx, ny, width64, height64, lbl, active, FMT: tl.constexpr):
    """One neighbor per lane: bounds -> is_red -> claim -> enqueue.

    Returns (inb, red, npix): the in-bounds mask, the lanes that found the
    neighbor red (each tried one claim, the CAS_ATTEMPTS unit) and the
    neighbor's linear index.
    """
    inb = active & (nx >= 0) & (nx < width64) & (ny >= 0) & (ny < height64)
    npix = nx * height64 + ny
    red = _is_red(img_ptr, npix, inb)
    old = tl.atomic_xchg(visited_ptr + npix, 1, mask=red,
                         sem="relaxed", scope="gpu")
    won = red & (old == 0)
    if FMT == 0:
        item = ((npix << 1) | lbl.to(tl.int64)).to(tl.int32)
    else:
        item = ((lbl.to(tl.int64) << _XY_LBL_SHIFT)
                | (nx << _XY_FIELD_BITS) | ny).to(tl.int32)
    _block_enqueue_global(queue_ptr, q_state_ptr, counters_ptr, qcap, item,
                          won)
    return inb, red, npix


@triton.jit
def _dual_blob(img_ptr, visited_ptr, depth_ptr, owner_ptr, queue_ptr,
               q_state_ptr, counters_ptr, block_stats_ptr, level_sizes_ptr,
               n_seeds, bar_ptr, width, height, qcap, trace_cap,
               FMT: tl.constexpr, CONN: tl.constexpr, RADIUS: tl.constexpr,
               INSTR: tl.constexpr, BLOCK: tl.constexpr):
    """One shared global queue of label-packed entries; every program
    grid-strides each level window.

    Host contract (the Numba kernels' own): launch (blocks,) programs of
    BLOCK lanes cooperatively; visited[seed]=1 for every seed,
    queue[0:n] = the n packed seed entries, q_state=[n], n_seeds=n,
    depth=-1, counters and block_stats zeroed (block_stats[:, 1] = -1),
    owner=-1, bar=int64[0]. Two barriers per level: #1 makes the level's
    enqueues and rear visible grid-wide, #2 guarantees everyone has read
    the new rear before any lane's next-level atomics.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    stride = nprog * BLOCK  # cuda.gridsize(1)
    width64 = width.to(tl.int64)
    height64 = height.to(tl.int64)

    front = 0
    rear = n_seeds
    level = 0
    # Barrier epochs (2 per level). Counter and target are int64, so
    # 2 * levels * programs arrivals cannot wrap, like grid.sync.
    epoch = tl.full((), 0, tl.int64)
    if INSTR:
        peak_level = 1
        peak_occ = 1
        active_thread_sum = tl.full((), 0, tl.int64)
        active_warp_sum = tl.full((), 0, tl.int64)
        my_processed = tl.zeros([BLOCK], tl.int64)
        my_cas_attempts = tl.zeros([BLOCK], tl.int64)
        if RADIUS == 2:
            my_interior = tl.zeros([BLOCK], tl.int64)

    while front < rear:
        if INSTR:
            level_size = rear - front
            peak_level = tl.maximum(peak_level, level_size)
            active = tl.minimum(level_size, stride)
            active_thread_sum += active
            active_warp_sum += (active + 31) // 32
            # tid == 0 records the grid-wide frontier trace
            tl.store(level_sizes_ptr + level, level_size,
                     mask=(pid == 0) & (level < trace_cap))

        for base in range(front + pid * BLOCK, rear, stride):
            i = base + lane
            valid = i < rear
            entry = tl.load(queue_ptr + i, mask=valid, other=0)
            if FMT == 0:
                lbl = entry & 1
                pixel = (entry >> 1).to(tl.int64)
                x = pixel // height64  # 64-bit div/mod, as in Numba
                y = pixel % height64
            else:
                lbl = (entry >> _XY_LBL_SHIFT) & 0x3F
                x = ((entry >> _XY_FIELD_BITS) & _XY_FIELD_MASK).to(tl.int64)
                y = (entry & _XY_FIELD_MASK).to(tl.int64)
            pix = x * height64 + y

            p = img_ptr + pix * 3
            for c in tl.static_range(3):
                color = tl.where(lbl == 0, PAL0[c], PAL1[c]).to(tl.uint8)
                tl.store(p + c, color, mask=valid)
            tl.store(depth_ptr + pix, level, mask=valid)
            if INSTR:
                tl.store(owner_ptr + pix, pid.to(tl.int16), mask=valid)
                my_processed += valid.to(tl.int64)

            if RADIUS == 2:
                interior = valid
            for d in tl.static_range(CONN):
                if CONN == 4:
                    nx = x + DX4[d]
                    ny = y + DY4[d]
                else:
                    nx = x + DX8[d]
                    ny = y + DY8[d]
                inb, red, npix = _probe(
                    img_ptr, visited_ptr, queue_ptr, q_state_ptr,
                    counters_ptr, qcap, nx, ny, width64, height64, lbl,
                    valid, FMT)
                if INSTR:
                    my_cas_attempts += red.to(tl.int64)
                if RADIUS == 2:
                    # in bounds and not red: blob material only if already
                    # claimed (visited == 1); out of bounds: edge pixels
                    # never jump
                    seen = tl.load(visited_ptr + npix, mask=inb & ~red,
                                   other=1)
                    interior = interior & inb & (red | (seen != 0))

            if RADIUS == 2:
                if INSTR:
                    my_interior += interior.to(tl.int64)
                # Numba's divergent `if interior:` skips ring 2 per warp;
                # here per program, as a 0/1-trip loop so the enqueue's
                # scan is never inside an if.
                any_interior = tl.max(interior.to(tl.int32), axis=0)
                for _ring2 in range(0, any_interior):
                    for d in tl.static_range(16):
                        # bounds still required: ring-1 in bounds does not
                        # imply ring-2 in bounds (x == 1 -> ring-2 at -1)
                        inb2, red2, npix2 = _probe(
                            img_ptr, visited_ptr, queue_ptr, q_state_ptr,
                            counters_ptr, qcap, x + DX_R2[d], y + DY_R2[d],
                            width64, height64, lbl, interior, FMT)
                        if INSTR:
                            my_cas_attempts += red2.to(tl.int64)

        epoch += 1
        grid_sync(bar_ptr, epoch * nprog)  # enqueues + final rear visible
        new_rear = tl.load(q_state_ptr + _Q_REAR)
        epoch += 1
        grid_sync(bar_ptr, epoch * nprog)  # everyone has read new_rear

        level += 1
        if INSTR:
            peak_occ = tl.maximum(peak_occ, new_rear - front)
        front = rear
        rear = new_rear

    if INSTR:
        # every lane adds its own count, as every Numba thread does
        same = lane * 0
        tl.atomic_add(counters_ptr + _PROCESSED + same, my_processed,
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters_ptr + _CAS_ATTEMPTS + same, my_cas_attempts,
                      sem="relaxed", scope="gpu")
        if RADIUS == 2:
            tl.atomic_add(counters_ptr + _INTERIOR + same, my_interior,
                          sem="relaxed", scope="gpu")
        tl.atomic_add(block_stats_ptr + pid * 2 + _BS_PROCESSED + same,
                      my_processed, sem="relaxed", scope="gpu")
        # threadIdx.x == 0
        tl.store(block_stats_ptr + pid * 2 + _BS_SMID,
                 read_smid(pid).to(tl.int64))
    if pid == 0:
        # tid == 0: all grid-uniform values
        tl.store(counters_ptr + _FILLED,
                 tl.load(q_state_ptr + _Q_REAR).to(tl.int64))
        tl.store(counters_ptr + _LEVELS, level.to(tl.int64))
        if INSTR:
            tl.store(counters_ptr + _PEAK_LEVEL, peak_level.to(tl.int64))
            tl.store(counters_ptr + _PEAK_OCC, peak_occ.to(tl.int64))
            tl.store(counters_ptr + _ACTIVE_THREAD_SUM, active_thread_sum)
            tl.store(counters_ptr + _ACTIVE_WARP_SUM, active_warp_sum)


# Instrumented kernels take Numba's (img, visited, depth, owner, queue,
# q_state, counters, block_stats, level_sizes, n_seeds), plus the grid
# barrier counter and the sizes Numba reads from the arrays' shapes. Bare
# twins take Numba's (img, visited, depth, queue, q_state, counters,
# n_seeds) plus the same extras; their unused instrumentation pointers are
# filled with `counters`, and the code reading them is compiled out.

# ===================================================== lin entry family


@triton.jit(do_not_specialize=_DNS)
def dual_blob_lin_kernel(img, visited, depth, owner, queue, q_state,
                         counters, block_stats, level_sizes, n_seeds, bar,
                         width, height, qcap, trace_cap,
                         BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, owner, queue, q_state, counters,
               block_stats, level_sizes, n_seeds, bar, width, height, qcap,
               trace_cap, 0, 4, 1, True, BLOCK)


@triton.jit(do_not_specialize=_DNS_BARE)
def dual_blob_lin_bare_kernel(img, visited, depth, queue, q_state, counters,
                              n_seeds, bar, width, height, qcap,
                              BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, counters, queue, q_state, counters,
               counters, counters, n_seeds, bar, width, height, qcap, 0,
               0, 4, 1, False, BLOCK)


@triton.jit(do_not_specialize=_DNS)
def dual_blob_lin8_kernel(img, visited, depth, owner, queue, q_state,
                          counters, block_stats, level_sizes, n_seeds, bar,
                          width, height, qcap, trace_cap,
                          BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, owner, queue, q_state, counters,
               block_stats, level_sizes, n_seeds, bar, width, height, qcap,
               trace_cap, 0, 8, 1, True, BLOCK)


@triton.jit(do_not_specialize=_DNS_BARE)
def dual_blob_lin8_bare_kernel(img, visited, depth, queue, q_state, counters,
                               n_seeds, bar, width, height, qcap,
                               BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, counters, queue, q_state, counters,
               counters, counters, n_seeds, bar, width, height, qcap, 0,
               0, 8, 1, False, BLOCK)


# ------------------------------------- radius-2 twins (guarded, lin only)


@triton.jit(do_not_specialize=_DNS)
def dual_blob_lin8r2_kernel(img, visited, depth, owner, queue, q_state,
                            counters, block_stats, level_sizes, n_seeds, bar,
                            width, height, qcap, trace_cap,
                            BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, owner, queue, q_state, counters,
               block_stats, level_sizes, n_seeds, bar, width, height, qcap,
               trace_cap, 0, 8, 2, True, BLOCK)


@triton.jit(do_not_specialize=_DNS_BARE)
def dual_blob_lin8r2_bare_kernel(img, visited, depth, queue, q_state,
                                 counters, n_seeds, bar, width, height, qcap,
                                 BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, counters, queue, q_state, counters,
               counters, counters, n_seeds, bar, width, height, qcap, 0,
               0, 8, 2, False, BLOCK)


# ====================================================== xy entry family


@triton.jit(do_not_specialize=_DNS)
def dual_blob_xy_kernel(img, visited, depth, owner, queue, q_state,
                        counters, block_stats, level_sizes, n_seeds, bar,
                        width, height, qcap, trace_cap,
                        BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, owner, queue, q_state, counters,
               block_stats, level_sizes, n_seeds, bar, width, height, qcap,
               trace_cap, 1, 4, 1, True, BLOCK)


@triton.jit(do_not_specialize=_DNS_BARE)
def dual_blob_xy_bare_kernel(img, visited, depth, queue, q_state, counters,
                             n_seeds, bar, width, height, qcap,
                             BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, counters, queue, q_state, counters,
               counters, counters, n_seeds, bar, width, height, qcap, 0,
               1, 4, 1, False, BLOCK)


@triton.jit(do_not_specialize=_DNS)
def dual_blob_xy8_kernel(img, visited, depth, owner, queue, q_state,
                         counters, block_stats, level_sizes, n_seeds, bar,
                         width, height, qcap, trace_cap,
                         BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, owner, queue, q_state, counters,
               block_stats, level_sizes, n_seeds, bar, width, height, qcap,
               trace_cap, 1, 8, 1, True, BLOCK)


@triton.jit(do_not_specialize=_DNS_BARE)
def dual_blob_xy8_bare_kernel(img, visited, depth, queue, q_state, counters,
                              n_seeds, bar, width, height, qcap,
                              BLOCK: tl.constexpr):
    _dual_blob(img, visited, depth, counters, queue, q_state, counters,
               counters, counters, n_seeds, bar, width, height, qcap, 0,
               1, 8, 1, False, BLOCK)
