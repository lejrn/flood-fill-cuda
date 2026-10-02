"""Triton twin of ch03's multi-block BFS kernels: N programs, one global queue.

Same algorithm as chapters/ch03_gpu_1blob_nblock/kernels.py, kernel for
kernel: one monotonic global int32 queue, every program grid-strides each
level window, one claim protocol (bounds -> is_red -> atomic claim of
visited -> enqueue), two grid barriers per level. The eight Numba kernels
(4-conn, 8-conn, guarded radius-2 and warp-cooperative probing, each
instrumented and bare) keep their names here. Each is a thin @triton.jit
wrapper around one body, _bfs, whose constexpr flags pick the variant, so
every Numba kernel compiles to its own binary and the bare twins compile
with no instrumentation code at all.

Mapping (one Numba block of T threads = one program of T lanes, num_warps =
T // 32, lane i plays thread i):

- grid.sync()              -> runtime.device.grid_sync (cooperative launch,
                              monotonic int64 counter, two per level as in
                              Numba)
- cuda.atomic.cas(visited, 0, 1)
                           -> masked tl.atomic_xchg(visited, 1): the same
                              exactly-once claim on a 0/1 flag (old == 0
                              wins), relaxed like Numba's atom.cas
- _warp_enqueue_global     -> _lane_enqueue_global (ENQ="lane", the
                              default): one relaxed tl.atomic_add per
                              claiming lane on the rear, which ptxas
                              warp-aggregates (VOTEU.ANY, UPOPC, one leader
                              ATOMG, SHFL.IDX): the SASS pattern of Numba's
                              activemask/popc/leader/shfl helper.
                              ENQ="program" keeps the first translation,
                              _block_enqueue_global (tl.sum + tl.cumsum +
                              one atomic per program), whose CTA barriers
                              Numba never had; it stays for measurement.
- per-thread counters      -> per-lane int32 accumulators, reduced once at
                              exit (one atomic per program instead of one
                              per thread)
- cuda.const.array_like    -> constexpr tuples indexed by tl.static_range
- get_smid (smid.cu)       -> runtime.device.read_smid (inline PTX)

Structural no-overflow argument (unchanged): the queue holds width*height
slots and every pixel is claimed at most once before enqueue, so OVERFLOW
stays a defensive tripwire.
"""

import numpy as np
import triton
import triton.language as tl

from ...runtime.device import grid_sync, read_smid

# Host-side tables, identical to the Numba chapter's (tests check this).
DX_HOST = np.array([1, 0, -1, 0], dtype=np.int32)
DY_HOST = np.array([0, 1, 0, -1], dtype=np.int32)
DX8_HOST = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
DY8_HOST = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
DX_R2_HOST = np.array([2, 2, 2, 1, 0, -1, -2, -2,
                       -2, -2, -2, -1, 0, 1, 2, 2], dtype=np.int32)
DY_R2_HOST = np.array([0, 1, 2, 2, 2, 2, 2, 1,
                       0, -1, -2, -2, -2, -2, -2, -1], dtype=np.int32)

# Device-side twins of the const arrays: compile-time tuples, indexed by a
# tl.static_range variable, so every offset folds into an immediate.
DX4 = tl.constexpr(tuple(int(v) for v in DX_HOST))
DY4 = tl.constexpr(tuple(int(v) for v in DY_HOST))
DX8 = tl.constexpr(tuple(int(v) for v in DX8_HOST))
DY8 = tl.constexpr(tuple(int(v) for v in DY8_HOST))
DX_R2 = tl.constexpr(tuple(int(v) for v in DX_R2_HOST))
DY_R2 = tl.constexpr(tuple(int(v) for v in DY_R2_HOST))

# Slots in the device-side int64 counters array (same indices as Numba).
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
BS_PROCESSED = 0
BS_SMID = 1

# q_state slots
Q_REAR = 0

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

# Enqueue modes: the ENQ constexpr of every kernel (one binary per mode).
#   "lane"     the default: one relaxed atomic_add per claiming lane on the
#              rear counter; ptxas warp-aggregates it, so the SASS matches
#              Numba's _warp_enqueue_global (no CTA barrier).
#   "program"  the first translation: tl.sum + tl.cumsum over the program,
#              one atomic per program per direction (CTA barriers and
#              shared-memory round trips in SASS). Kept to measure its cost.
ENQ_LANE = "lane"
ENQ_PROGRAM = "program"
ENQ_MODES = (ENQ_LANE, ENQ_PROGRAM)
DEFAULT_ENQ = ENQ_LANE

# Runtime ints that vary between calls: never specialize on them (a new
# divisibility-by-16 or ==1 class would otherwise recompile mid-benchmark).
_DNS = ["width", "height", "qcap", "trace_cap"]


@triton.jit
def _is_red(img_ptr, lin, mask):
    """Per lane: is pixel ``lin`` red (255, 0, 0)? Three byte loads,
    short-circuited lane by lane like Numba's ``and`` chain."""
    p = img_ptr + lin.to(tl.int64) * 3
    c0 = tl.load(p, mask=mask, other=0)
    m1 = mask & (c0 == 255)
    c1 = tl.load(p + 1, mask=m1, other=1)
    m2 = m1 & (c1 == 0)
    c2 = tl.load(p + 2, mask=m2, other=1)
    return m2 & (c2 == 0)


@triton.jit
def _claim(visited_ptr, lin, red):
    """Exactly-once claim of visited[lin] on the ``red`` lanes. A masked
    exchange to 1 on a 0/1 flag returns 0 to exactly one claimant, the same
    guarantee as Numba's cas(visited, 0, 1); relaxed, like atom.cas."""
    old = tl.atomic_xchg(visited_ptr + lin, 1, mask=red, sem="relaxed",
                         scope="gpu")
    return red & (old == 0)


@triton.jit
def _lane_enqueue_global(queue_ptr, q_state_ptr, counters_ptr, item, claimed,
                         qcap):
    """Per-lane append on the global rear counter (ENQ="lane", the default).

    Twin of _warp_enqueue_global: every claiming lane takes its own ticket
    with one relaxed atomic_add of 1 on the rear and writes its item at the
    slot the atomic returned. The address is the same for the whole warp,
    so ptxas compiles the add into a warp-aggregated atomic: VOTEU.ANY of
    the active lanes, UPOPC for the count, one ATOMG by the leader lane,
    SHFL.IDX of the base and each lane's rank among the active lanes. That
    is the machine code Numba's activemask/popc/leader/shfl helper produces,
    with no CTA barrier. The bound check is the same defensive tripwire
    (a rear past qcap writes nothing out of bounds).
    """
    zero = item * 0
    slot = tl.atomic_add(q_state_ptr + _Q_REAR + zero, 1, mask=claimed,
                         sem="relaxed", scope="gpu")
    tl.store(queue_ptr + slot, item, mask=claimed & (slot < qcap))
    # unreachable by the structural argument
    tl.store(counters_ptr + _OVERFLOW + zero, (zero + 1).to(tl.int64),
             mask=claimed & (slot >= qcap))


@triton.jit
def _block_enqueue_global(queue_ptr, q_state_ptr, counters_ptr, item, claimed,
                          qcap):
    """Program-aggregated append on the global rear counter (ENQ="program").

    The first translation of _warp_enqueue_global, kept to measure its
    cost: the claiming lanes count themselves, one atomic reserves a slab
    for the whole program, and each lane writes at base + its rank among
    the claimants. tl.sum and tl.cumsum across 8 warps go through shared
    memory with CTA barriers, which Numba's warp helper never has. The scan
    stays outside the branch (Triton 3.7 miscompiles scans inside an if).
    The bound check is the same defensive tripwire as Numba's.
    """
    c = claimed.to(tl.int32)
    count = tl.sum(c, axis=0)
    rank = tl.cumsum(c, axis=0) - c
    if count > 0:
        base = tl.atomic_add(q_state_ptr + _Q_REAR, count, sem="relaxed",
                             scope="gpu")
        idx = base + rank
        tl.store(queue_ptr + idx, item, mask=claimed & (idx < qcap))
        # unreachable by the structural argument
        tl.store(counters_ptr + _OVERFLOW + idx * 0,
                 (idx * 0 + 1).to(tl.int64), mask=claimed & (idx >= qcap))


@triton.jit
def _enqueue_global(queue_ptr, q_state_ptr, counters_ptr, item, claimed,
                    qcap, ENQ: tl.constexpr):
    """Append the ``claimed`` lanes' items: per lane (the default) or
    aggregated over the program (the first translation)."""
    tl.static_assert((ENQ == "lane") | (ENQ == "program"),
                     "ENQ must be 'lane' or 'program'")
    if ENQ == "program":
        _block_enqueue_global(queue_ptr, q_state_ptr, counters_ptr, item,
                              claimed, qcap)
    else:
        _lane_enqueue_global(queue_ptr, q_state_ptr, counters_ptr, item,
                             claimed, qcap)


@triton.jit
def _bfs(img_ptr, visited_ptr, depth_ptr, owner_ptr, queue_ptr, q_state_ptr,
         counters_ptr, block_stats_ptr, level_sizes_ptr, bar_ptr,
         width, height, qcap, trace_cap,
         CONN: tl.constexpr, RADIUS2: tl.constexpr, WARP_COOP: tl.constexpr,
         INSTRUMENTED: tl.constexpr, BLOCK: tl.constexpr,
         ENQ: tl.constexpr):
    """The level-synchronous BFS every ch03 kernel runs.

    Host contract (as in Numba): launch [blocks] programs of BLOCK lanes,
    cooperatively; visited[seed]=1, queue[0]=seed linear index, q_state=[1],
    depth=-1, counters and block_stats zeroed (BS_SMID column -1), owner=-1,
    bar=int64[0]. Two grid barriers per level: #1 makes the level's enqueues and
    rear visible grid-wide, #2 guarantees every program has read the new
    rear before any next-level atomic. ENQ picks the enqueue (see
    ENQ_MODES); everything else is the same code in both modes.
    """
    bx = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    stride = nprog * BLOCK  # cuda.gridsize(1)

    if WARP_COOP:
        # lane = tid & 31 (BLOCK is a multiple of 32); k = lane >> 3 picks
        # one of the warp's 4 entries, d = lane & 7 one of its 8 directions.
        d_lane = lanes & 7
        dxl = tl.zeros([BLOCK], tl.int32)
        dyl = tl.zeros([BLOCK], tl.int32)
        for d in tl.static_range(8):
            dxl = tl.where(d_lane == d, DX8[d], dxl)
            dyl = tl.where(d_lane == d, DY8[d], dyl)
        n_warps = stride // 32
        WARPS: tl.constexpr = BLOCK // 32

    front = 0
    rear = 1
    level = 0
    # Barrier epochs (2 per level). The counter and the target are int64:
    # 2 * levels * programs arrivals can pass 2**31 on a long serpentine at
    # a wide grid, and grid.sync has no such limit. epoch itself stays int32
    # (levels < width*height < 2**31 / 15 for any image that fits in VRAM).
    epoch = 0
    if INSTRUMENTED:
        peak_level = 1
        peak_occ = 1
        active_thread_sum = tl.full((), 0, tl.int64)
        active_warp_sum = tl.full((), 0, tl.int64)
        my_processed = tl.zeros([BLOCK], tl.int32)
        my_cas_attempts = tl.zeros([BLOCK], tl.int32)
        if RADIUS2:
            my_interior = tl.zeros([BLOCK], tl.int32)

    while front < rear:
        if INSTRUMENTED:
            level_size = rear - front
            peak_level = tl.maximum(peak_level, level_size)
            if WARP_COOP:
                active = tl.minimum(level_size.to(tl.int64) * 8,
                                    stride.to(tl.int64))  # 8 lanes per entry
            else:
                active = tl.minimum(level_size, stride).to(tl.int64)
            active_thread_sum += active
            active_warp_sum += (active + 31) // 32
            # tid == 0 records the grid-wide frontier trace
            tl.store(level_sizes_ptr + level, level_size,
                     mask=(bx == 0) & (level < trace_cap))

        if WARP_COOP:
            # Warp w of the grid owns chunks front + w*4 + j*n_warps*4; this
            # program's warps are bx*WARPS .. bx*WARPS + WARPS-1, and
            # (lane >> 5)*4 + k == lanes >> 3.
            for cb in range(front + bx * WARPS * 4, rear, n_warps * 4):
                idx = cb + (lanes >> 3)
                valid = idx < rear  # partial final chunk
                pixel = tl.load(queue_ptr + idx, mask=valid, other=0)
                x = pixel // height
                y = pixel % height

                painter = valid & (d_lane == 0)  # one lane per entry paints
                p = img_ptr + pixel.to(tl.int64) * 3
                tl.store(p, 0, mask=painter)
                tl.store(p + 1, 0, mask=painter)
                tl.store(p + 2, 255, mask=painter)
                tl.store(depth_ptr + pixel, level, mask=painter)
                if INSTRUMENTED:
                    tl.store(owner_ptr + pixel, bx.to(tl.int16), mask=painter)
                    my_processed += painter.to(tl.int32)

                nx = x + dxl
                ny = y + dyl
                inb = (valid & (nx >= 0) & (nx < width)
                       & (ny >= 0) & (ny < height))
                nlin = nx * height + ny
                red = _is_red(img_ptr, nlin, inb)
                if INSTRUMENTED:
                    my_cas_attempts += red.to(tl.int32)
                claimed = _claim(visited_ptr, nlin, red)
                _enqueue_global(queue_ptr, q_state_ptr, counters_ptr, nlin,
                                claimed, qcap, ENQ)
        else:
            for base in range(front + bx * BLOCK, rear, stride):
                i = base + lanes
                m = i < rear
                pixel = tl.load(queue_ptr + i, mask=m, other=0)
                x = pixel // height
                y = pixel % height

                p = img_ptr + pixel.to(tl.int64) * 3
                tl.store(p, 0, mask=m)
                tl.store(p + 1, 0, mask=m)
                tl.store(p + 2, 255, mask=m)
                tl.store(depth_ptr + pixel, level, mask=m)
                if INSTRUMENTED:
                    tl.store(owner_ptr + pixel, bx.to(tl.int16), mask=m)
                    my_processed += m.to(tl.int32)

                if RADIUS2:
                    interior = m
                for d in tl.static_range(CONN):
                    if CONN == 4:
                        nx = x + DX4[d]
                        ny = y + DY4[d]
                    else:
                        nx = x + DX8[d]
                        ny = y + DY8[d]
                    inb = (m & (nx >= 0) & (nx < width)
                           & (ny >= 0) & (ny < height))
                    nlin = nx * height + ny
                    red = _is_red(img_ptr, nlin, inb)
                    if INSTRUMENTED:
                        my_cas_attempts += red.to(tl.int32)
                    claimed = _claim(visited_ptr, nlin, red)
                    _enqueue_global(queue_ptr, q_state_ptr, counters_ptr,
                                    nlin, claimed, qcap, ENQ)
                    if RADIUS2:
                        # in bounds and not red: blob material only if it
                        # was already claimed (visited == 1); out of bounds
                        # or never red ends the guard
                        seen = tl.load(visited_ptr + nlin, mask=inb & ~red,
                                       other=1)
                        interior = interior & inb & (red | (seen != 0))

                if RADIUS2:
                    if INSTRUMENTED:
                        my_interior += interior.to(tl.int32)
                    # Numba's divergent `if interior:` skips ring 2 per warp;
                    # here per program, as a 0/1-trip loop so the program
                    # enqueue's scan is never inside an if. Same skip in both
                    # ENQ modes, so the switch measures the enqueue alone.
                    any_interior = tl.max(interior.to(tl.int32), axis=0)
                    for _ring2 in range(0, any_interior):
                        for d in tl.static_range(16):
                            nx = x + DX_R2[d]
                            ny = y + DY_R2[d]
                            # ring-1 in bounds does not imply ring-2 in
                            # bounds (x == 1 -> ring-2 at -1)
                            ok = (interior & (nx >= 0) & (nx < width)
                                  & (ny >= 0) & (ny < height))
                            nlin = nx * height + ny
                            red = _is_red(img_ptr, nlin, ok)
                            if INSTRUMENTED:
                                my_cas_attempts += red.to(tl.int32)
                            claimed = _claim(visited_ptr, nlin, red)
                            _enqueue_global(queue_ptr, q_state_ptr,
                                            counters_ptr, nlin, claimed,
                                            qcap, ENQ)

        epoch += 1
        grid_sync(bar_ptr, epoch.to(tl.int64) * nprog)  # enqueues + rear visible
        new_rear = tl.load(q_state_ptr + _Q_REAR)
        epoch += 1
        grid_sync(bar_ptr, epoch.to(tl.int64) * nprog)  # everyone read new_rear

        level += 1
        if INSTRUMENTED:
            peak_occ = tl.maximum(peak_occ, new_rear - front)
        front = rear
        rear = new_rear

    if INSTRUMENTED:
        processed = tl.sum(my_processed.to(tl.int64), axis=0)
        tl.atomic_add(counters_ptr + _PROCESSED, processed, sem="relaxed",
                      scope="gpu")
        tl.atomic_add(counters_ptr + _CAS_ATTEMPTS,
                      tl.sum(my_cas_attempts.to(tl.int64), axis=0),
                      sem="relaxed", scope="gpu")
        if RADIUS2:
            tl.atomic_add(counters_ptr + _INTERIOR,
                          tl.sum(my_interior.to(tl.int64), axis=0),
                          sem="relaxed", scope="gpu")
        tl.atomic_add(block_stats_ptr + bx * 2 + _BS_PROCESSED, processed,
                      sem="relaxed", scope="gpu")
        tl.store(block_stats_ptr + bx * 2 + _BS_SMID,
                 read_smid(bx).to(tl.int64))
    if bx == 0:
        # tid == 0: all grid-uniform register values
        tl.store(counters_ptr + _FILLED,
                 tl.load(q_state_ptr + _Q_REAR).to(tl.int64))
        tl.store(counters_ptr + _LEVELS, level.to(tl.int64))
        if INSTRUMENTED:
            tl.store(counters_ptr + _PEAK_LEVEL, peak_level.to(tl.int64))
            tl.store(counters_ptr + _PEAK_OCC, peak_occ.to(tl.int64))
            tl.store(counters_ptr + _ACTIVE_THREAD_SUM, active_thread_sum)
            tl.store(counters_ptr + _ACTIVE_WARP_SUM, active_warp_sum)


# ------------------------------------------------------- 4-connectivity pair
# Instrumented kernels take (img, visited, depth, owner, queue, q_state,
# counters, block_stats, level_sizes) like Numba, plus the grid barrier
# counter and the sizes Numba reads from the arrays' shapes.


@triton.jit(do_not_specialize=_DNS)
def multi_block_global_kernel(img, visited, depth, owner, queue, q_state,
                              counters, block_stats, level_sizes, bar,
                              width, height, qcap, trace_cap,
                              BLOCK: tl.constexpr,
                              ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, owner, queue, q_state, counters, block_stats,
         level_sizes, bar, width, height, qcap, trace_cap,
         4, False, False, True, BLOCK, ENQ)


# Bare twins: (img, visited, depth, queue, q_state, counters) like Numba.
# The unused instrumentation pointers are filled with `counters` and the
# code reading them is compiled out (INSTRUMENTED=False).


@triton.jit(do_not_specialize=_DNS[:3])
def multi_block_global_bare_kernel(img, visited, depth, queue, q_state,
                                   counters, bar, width, height, qcap,
                                   BLOCK: tl.constexpr,
                                   ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, counters, queue, q_state, counters, counters,
         counters, bar, width, height, qcap, 0,
         4, False, False, False, BLOCK, ENQ)


# ------------------------------------------------------- 8-connectivity pair


@triton.jit(do_not_specialize=_DNS)
def multi_block_global8_kernel(img, visited, depth, owner, queue, q_state,
                               counters, block_stats, level_sizes, bar,
                               width, height, qcap, trace_cap,
                               BLOCK: tl.constexpr,
                               ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, owner, queue, q_state, counters, block_stats,
         level_sizes, bar, width, height, qcap, trace_cap,
         8, False, False, True, BLOCK, ENQ)


@triton.jit(do_not_specialize=_DNS[:3])
def multi_block_global8_bare_kernel(img, visited, depth, queue, q_state,
                                    counters, bar, width, height, qcap,
                                    BLOCK: tl.constexpr,
                                    ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, counters, queue, q_state, counters, counters,
         counters, bar, width, height, qcap, 0,
         8, False, False, False, BLOCK, ENQ)


# ------------------------------------------------ radius-2 pair (guarded)


@triton.jit(do_not_specialize=_DNS)
def multi_block_global8r2_kernel(img, visited, depth, owner, queue, q_state,
                                 counters, block_stats, level_sizes, bar,
                                 width, height, qcap, trace_cap,
                                 BLOCK: tl.constexpr,
                                 ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, owner, queue, q_state, counters, block_stats,
         level_sizes, bar, width, height, qcap, trace_cap,
         8, True, False, True, BLOCK, ENQ)


@triton.jit(do_not_specialize=_DNS[:3])
def multi_block_global8r2_bare_kernel(img, visited, depth, queue, q_state,
                                      counters, bar, width, height, qcap,
                                      BLOCK: tl.constexpr,
                                      ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, counters, queue, q_state, counters, counters,
         counters, bar, width, height, qcap, 0,
         8, True, False, False, BLOCK, ENQ)


# ------------------------------------- warp-cooperative probing pair


@triton.jit(do_not_specialize=_DNS)
def multi_block_global8wc_kernel(img, visited, depth, owner, queue, q_state,
                                 counters, block_stats, level_sizes, bar,
                                 width, height, qcap, trace_cap,
                                 BLOCK: tl.constexpr,
                                 ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, owner, queue, q_state, counters, block_stats,
         level_sizes, bar, width, height, qcap, trace_cap,
         8, False, True, True, BLOCK, ENQ)


@triton.jit(do_not_specialize=_DNS[:3])
def multi_block_global8wc_bare_kernel(img, visited, depth, queue, q_state,
                                      counters, bar, width, height, qcap,
                                      BLOCK: tl.constexpr,
                                      ENQ: tl.constexpr = DEFAULT_ENQ):
    _bfs(img, visited, depth, counters, queue, q_state, counters, counters,
         counters, bar, width, height, qcap, 0,
         8, False, True, False, BLOCK, ENQ)
