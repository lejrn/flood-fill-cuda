"""Triton twins of the dual-block BFS kernels: 1 block -> 2 programs.

Same four kernels as the Numba chapter, same claim protocol (bounds ->
is_red -> claim visited -> enqueue), same level-synchronous structure, same
queues and counters. One Numba block of T threads is one Triton program of
T lanes (num_warps = T // 32); the grid of 2 blocks is a grid of 2 programs.

- "global"   one monotonic global queue, both programs grid-stride each
             level window; two grid_sync per level.
- "split"    domain decomposition at x = width//2: each program runs the
             two-tier machinery (8192-slot ring + own spill tier) on its
             half; cross-seam claims go to the OTHER program's inbox.
             1 cta_sync + 2 grid_sync per level.
- "dirsplit" right/up claims -> queue 0 (program 0), down/left claims ->
             queue 1 (program 1); one width*height buffer filled from both
             ends.
- "pinned"   the placement experiment: global-queue BFS run by two worker
             programs, either forced onto ONE SM (occupancy-forced launch +
             %smid self-identification) or spread by the scheduler, with the
             chapter's hand-rolled pair barrier.

What changes in the translation (see README for the full mapping table):

- grid.sync() -> runtime.device.grid_sync on a host-zeroed counter, inside
  a cooperative launch (launch_cooperative_grid=True), same placement.
- Warp-aggregated appends -> program-aggregated appends: tl.cumsum ranks
  the claiming lanes, one atomic per program per call site reserves the
  slab. Tickets keep their meaning; contiguity is per program, not per warp.
- The split kernel's shared-memory ring has no Triton equivalent (no
  user-addressable shared memory): it becomes a per-program 8192-slot
  region of global scratch with the same virtual tickets, mask, clamp and
  spill-over rule. Its shared rears become program registers: a program
  is the only producer of its own ring, so the shared atomics reduce to
  register adds.
- The 0/1 visited claim is a masked tl.atomic_xchg(.., 1): old == 0 wins,
  exactly the CAS(0 -> 1) outcome, without dummy traffic for masked lanes.
- Per-thread register counters (my_processed, my_cas_attempts) -> per-lane
  [TPB] accumulators; Numba's per-thread exit atomics on the int64 counters
  -> one atomic per program carrying the program's sum (same totals).
- INSTRUMENTED (constexpr) merges each kernel with its bare twin; the bare
  specialization is compiled without any counter or trace code.
"""

import triton
import triton.language as tl

from flood_fill_cuda.triton_twins.runtime.device import (
    cta_sync, grid_sync, load_acquire, read_smid,
)

RING_CAPACITY = tl.constexpr(8192)
RING_MASK = tl.constexpr(8192 - 1)

# Slots in the device-side int64 counters array (same indices as Numba)
FILLED = tl.constexpr(0)
LEVELS = tl.constexpr(1)
OVERFLOW = tl.constexpr(2)            # defensive tripwire: structurally unreachable
PEAK_LEVEL = tl.constexpr(3)          # largest single global frontier
PEAK_OCC = tl.constexpr(4)            # max queue entries alive across two adjacent levels
ACTIVE_THREAD_SUM = tl.constexpr(5)   # sum over levels+blocks of min(own_size, tpb)
ACTIVE_WARP_SUM = tl.constexpr(6)     # sum over levels+blocks of ceil(min(own_size, tpb)/32)
PROCESSED = tl.constexpr(7)           # pixels dequeued/recolored
CAS_ATTEMPTS = tl.constexpr(8)        # visited claims tried
SPILLED = tl.constexpr(9)             # split: total pixels sent to both spill tiers
PEAK_SPILL_WINDOW = tl.constexpr(10)  # split: largest single-level spill count
PROCESSED_B0 = tl.constexpr(11)       # per-program work counts -> load balance
PROCESSED_B1 = tl.constexpr(12)
SPILLED_B0 = tl.constexpr(13)
SPILLED_B1 = tl.constexpr(14)
INBOX_TO_B0 = tl.constexpr(15)        # split: pixels handed across the seam to program 0
INBOX_TO_B1 = tl.constexpr(16)
SMID_B0 = tl.constexpr(17)            # %smid each program observed itself running on
SMID_B1 = tl.constexpr(18)
NUM_COUNTERS = tl.constexpr(19)

# q_state slots
Q_REAR = tl.constexpr(0)              # "global"/"pinned": the single queue's rear
Q_REAR0 = tl.constexpr(0)             # "dirsplit": forward queue (right/up claims)
Q_REAR1 = tl.constexpr(1)             # "dirsplit": backward queue (down/left claims)

# g_state slots ("split")
G_INBOX_REAR0 = tl.constexpr(0)       # rear of the inbox TO program 0 (bumped by 1)
G_INBOX_REAR1 = tl.constexpr(1)       # rear of the inbox TO program 1 (bumped by 0)
G_PUB_RING0 = tl.constexpr(2)         # program 0's published next-level ring count
G_PUB_SPILL0 = tl.constexpr(3)        # program 0's published next-level spill count
G_PUB_RING1 = tl.constexpr(4)
G_PUB_SPILL1 = tl.constexpr(5)

# pin_state slots ("pinned")
P_MODE = tl.constexpr(0)              # host-set, immutable: 0 = same_sm, 1 = spread
P_CHOSEN_SMID = tl.constexpr(1)       # -1 until the first program CASes its smid in
P_WORKER_COUNT = tl.constexpr(2)      # worker-rank dispenser

# barrier_state slots ("pinned")
BAR_ARRIVE = tl.constexpr(0)
BAR_GEN = tl.constexpr(1)


@triton.jit
def _is_red(img, lin, mask):
    """Red test with Numba's short-circuit: channel 1 is read only where
    channel 0 is 255, channel 2 only where channel 1 is 0.

    Offsets are int64 (lin * 3 can pass 2**31 on very large images)."""
    off = lin.to(tl.int64) * 3
    r = tl.load(img + off, mask=mask, other=0)
    m1 = mask & (r == 255)
    g = tl.load(img + off + 1, mask=m1, other=1)
    m2 = m1 & (g == 0)
    b = tl.load(img + off + 2, mask=m2, other=1)
    return m2 & (b == 0)


@triton.jit
def _recolor(img, lin, mask):
    off = lin.to(tl.int64) * 3
    tl.store(img + off, 0, mask=mask)
    tl.store(img + off + 1, 0, mask=mask)
    tl.store(img + off + 2, 255, mask=mask)


@triton.jit
def _neighbor(x, y, active, width, height, D: tl.constexpr):
    """Neighbor d in the chapter's order: right(1,0), down(0,1),
    left(-1,0), up(0,-1). Returns (linear index, in-bounds mask)."""
    dx: tl.constexpr = 1 if D == 0 else (-1 if D == 2 else 0)
    dy: tl.constexpr = 1 if D == 1 else (-1 if D == 3 else 0)
    nx = x + dx
    ny = y + dy
    inb = active & (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
    return nx * height + ny, nx, inb


@triton.jit
def _claim(visited, nlin, m):
    """Claim a 0/1 visited flag for the lanes in m; True where this lane won.

    Numba: cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0. A masked exchange
    to 1 has the same exactly-once outcome on a 0/1 flag (old == 0 wins)."""
    old = tl.atomic_xchg(visited + nlin, 1, mask=m, sem="relaxed", scope="gpu")
    return m & (old == 0)


@triton.jit
def _enqueue_global(arr, cap, state, rear_slot, item, won, counters,
                    BACKWARD: tl.constexpr):
    """Program-aggregated append on a global rear counter.

    Twin of _warp_enqueue_global: the claiming lanes are ranked with a
    prefix sum and one atomic per program reserves the slab (Numba: one per
    warp). backward=True writes slot cap-1-idx (dirsplit's double-ended
    buffer). The bound check is a defensive tripwire only.
    """
    w = won.to(tl.int32)
    cnt = tl.sum(w, axis=0)
    rank = tl.cumsum(w, axis=0) - w  # scans stay outside conditionals
    if cnt > 0:
        base = tl.atomic_add(state + rear_slot, cnt, sem="relaxed", scope="gpu")
        idx = base + rank
        ok = won & (idx < cap)
        if BACKWARD:
            tl.store(arr + (cap - 1 - idx), item, mask=ok)
        else:
            tl.store(arr + idx, item, mask=ok)
        # unreachable by the structural arguments
        tl.store(counters + OVERFLOW + rank * 0, 1, mask=won & (idx >= cap))


@triton.jit
def _enqueue_two_tier(ring, spill, s_rear, s_spill_rear, front, item, won):
    """Own-half append: ring fast path, own spill tier past it.

    Twin of _warp_enqueue_two_tier. The program reserves cnt virtual
    tickets [s_rear, s_rear + cnt); tickets inside [front, front + 8192)
    take ring slots (ticket & 8191), the rest spill. Because tickets are
    contiguous in rank order, the spilling lanes are exactly the top ranks,
    so the second aggregation (Numba's re-ballot of the else-branch lanes)
    is the closed form rank - room. Returns the new (s_rear, s_spill_rear).
    """
    w = won.to(tl.int32)
    cnt = tl.sum(w, axis=0)
    rank = tl.cumsum(w, axis=0) - w
    room = tl.maximum(front + RING_CAPACITY - s_rear, 0)
    to_ring = won & (rank < room)
    to_spill = won & (rank >= room)
    tl.store(ring + ((s_rear + rank) & RING_MASK), item, mask=to_ring)
    tl.store(spill + s_spill_rear + (rank - room), item, mask=to_spill)
    return s_rear + cnt, s_spill_rear + tl.maximum(cnt - room, 0)


@triton.jit
def _pair_barrier(barrier_state, n_workers):
    """The chapter's hand-rolled sense-reversing barrier, for co-resident
    worker programs (grid_sync would wait for every launched program).

    Scalar atomics run on one thread of the program, which matches Numba's
    "thread 0 does it, then syncthreads". threadfence() becomes the
    atomics' ordering:
    - the generation snapshot is relaxed, as in Numba (the arrival's
      release keeps it before the arrival);
    - the arrival is acq_rel: it publishes this program's writes, and the
      last arriver acquires everyone's;
    - the last arriver's relaxed reset and releasing generation bump run on
      the same thread, so the release orders the reset (Numba: store,
      threadfence, atomic add);
    - the spin reads acquire: the read that sees the new generation is the
      acquire Numba gets from its threadfence after the spin. A relaxed
      spin plus one trailing acquire read costs one more CTA-wide broadcast
      of a scalar per barrier here, measured up to 5% slower.
    """
    cta_sync()
    gen = tl.atomic_add(barrier_state + BAR_GEN, 0, sem="relaxed", scope="gpu")
    arrived = tl.atomic_add(barrier_state + BAR_ARRIVE, 1, sem="acq_rel",
                            scope="gpu")
    if arrived == n_workers - 1:
        tl.atomic_xchg(barrier_state + BAR_ARRIVE, 0, sem="relaxed", scope="gpu")
        tl.atomic_add(barrier_state + BAR_GEN, 1, sem="release", scope="gpu")
    else:
        g = load_acquire(barrier_state + BAR_GEN)
        while g == gen:
            g = load_acquire(barrier_state + BAR_GEN)
    cta_sync()


@triton.jit(do_not_specialize=["width", "height", "trace_cap"])
def dual_block_global_kernel(img, visited, depth, owner, queue, q_state,
                             counters, level_sizes, bar, width, height,
                             trace_cap, TPB: tl.constexpr,
                             INSTRUMENTED: tl.constexpr):
    """One shared global queue, both programs grid-stride each level window.

    Host contract: launch (2,) cooperative with num_warps = TPB // 32;
    visited[seed]=1, queue[0]=seed linear index, q_state=[1], depth=-1,
    counters and bar zeroed. Two grid_sync per level: #1 makes the level's
    enqueues and rear visible grid-wide, #2 guarantees everyone has read
    the new rear before any next-level atomic. INSTRUMENTED=False is the
    bare twin (FILLED/LEVELS/OVERFLOW only; owner and level_sizes unused).
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    stride = nprog * TPB
    lanes = tl.arange(0, TPB)
    cap = width * height

    front = 0
    rear = 1
    level = 0
    epoch = 0
    peak_level = 1
    peak_occ = 1
    active_thread_sum = tl.full((), 0, tl.int64)
    active_warp_sum = tl.full((), 0, tl.int64)
    # per-lane counters: lane i's registers are Numba thread i's
    # my_processed / my_cas_attempts (int32 per lane, summed as int64 at exit)
    my_processed = tl.zeros([TPB], tl.int32)
    my_cas_attempts = tl.zeros([TPB], tl.int32)

    while front < rear:
        if INSTRUMENTED:
            level_size = rear - front
            peak_level = tl.maximum(peak_level, level_size)
            active = tl.minimum(level_size, stride)
            active_thread_sum += active
            active_warp_sum += (active + 31) // 32
            if level < trace_cap:
                # row 0 carries the trace; row 1 stays a clean zero row
                tl.store(level_sizes + pid * trace_cap + level,
                         tl.where(pid == 0, level_size, 0))

        for start in range(front + pid * TPB, rear, stride):
            i = start + lanes
            act = i < rear
            pixel = tl.load(queue + i, mask=act, other=0)
            x = pixel // height
            y = pixel % height

            _recolor(img, pixel, act)
            tl.store(depth + pixel, level, mask=act)
            if INSTRUMENTED:
                tl.store(owner + pixel, pid.to(tl.int8), mask=act)
                my_processed += act.to(tl.int32)

            for d in tl.static_range(4):
                nlin, nx, inb = _neighbor(x, y, act, width, height, d)
                m = _is_red(img, nlin, inb)
                if INSTRUMENTED:
                    my_cas_attempts += m.to(tl.int32)
                won = _claim(visited, nlin, m)
                _enqueue_global(queue, cap, q_state, Q_REAR, nlin, won,
                                counters, False)

        epoch += 1
        grid_sync(bar, epoch * nprog)  # enqueues + final rear visible
        new_rear = tl.load(q_state + Q_REAR)
        epoch += 1
        grid_sync(bar, epoch * nprog)  # everyone has read new_rear

        level += 1
        if INSTRUMENTED:
            peak_occ = tl.maximum(peak_occ, new_rear - front)
        front = rear
        rear = new_rear

    if INSTRUMENTED:
        processed = tl.sum(my_processed.to(tl.int64), axis=0)
        tl.atomic_add(counters + PROCESSED, processed, sem="relaxed",
                      scope="gpu")
        tl.atomic_add(counters + PROCESSED_B0 + pid, processed,
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters + CAS_ATTEMPTS,
                      tl.sum(my_cas_attempts.to(tl.int64), axis=0),
                      sem="relaxed", scope="gpu")
        tl.store(counters + SMID_B0 + pid, read_smid(pid).to(tl.int64))
    if pid == 0:
        # all grid-uniform register values
        tl.store(counters + FILLED, tl.load(q_state + Q_REAR).to(tl.int64))
        tl.store(counters + LEVELS, level.to(tl.int64))
        if INSTRUMENTED:
            tl.store(counters + PEAK_LEVEL, peak_level.to(tl.int64))
            tl.store(counters + PEAK_OCC, peak_occ.to(tl.int64))
            tl.store(counters + ACTIVE_THREAD_SUM, active_thread_sum)
            tl.store(counters + ACTIVE_WARP_SUM, active_warp_sum)


@triton.jit(do_not_specialize=["seed_x", "seed_y", "width", "height",
                               "inbox_cap", "trace_cap"])
def dual_block_split_kernel(img, visited, depth, owner, seed_x, seed_y,
                            spill0, spill1, inbox0, inbox1, g_state,
                            counters, level_sizes, ring, bar, width, height,
                            inbox_cap, trace_cap, TPB: tl.constexpr,
                            INSTRUMENTED: tl.constexpr):
    """Domain decomposition: program 0 owns x < width//2, program 1 the rest.

    Each program runs the two-tier machinery on its half: its 8192-slot
    ring (a private region of ``ring``, (2, 8192) int32 global scratch, in
    place of Numba's shared array) plus its own spill tier. Cross-seam
    claims go to the OTHER program's inbox.

    Host contract: launch (2,) cooperative; visited[seed]=1; g_state, bar
    and counters zeroed; spill_b sized to half b, inboxes sized height;
    depth=-1. The seed-owning program seeds its own ring in the prologue.

    Level boundary = 1 cta_sync + 2 grid_sync: the program clamps its ring
    rear (retracting tickets that went to the spill tier) and publishes its
    next-level ring/spill counts; grid_sync #1 makes enqueues, inbox rears
    and both pubs visible; every program reads the six g_state values;
    grid_sync #2 orders those reads before the next level's atomics.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, TPB)
    half = width // 2
    my_ring = ring + pid * RING_CAPACITY

    if pid == 0:
        my_spill = spill0
        my_inbox = inbox0
        their_inbox = inbox1
    else:
        my_spill = spill1
        my_inbox = inbox1
        their_inbox = inbox0
    their_rear_slot = G_INBOX_REAR1 - pid  # G_INBOX_REAR1 for 0, 0 for 1

    seed_owner = tl.where(seed_x < half, 0, 1)
    if pid == seed_owner:
        tl.store(my_ring, seed_x * height + seed_y)
    cta_sync()  # the seed slot is visible to every lane of the program

    # the program-private shared rears of the Numba kernel
    s_rear = (pid == seed_owner).to(tl.int32)
    s_spill_rear = 0

    sf = 0                          # own ring window (virtual tickets)
    sr = s_rear
    gf = 0                          # own spill window
    gr = 0
    inf_ = 0                        # own inbox window
    inr = 0
    oth_inr = 0                     # other program's inbox rear (shadow copy)
    total = 1                       # GLOBAL level size, grid-uniform
    level = 0
    epoch = 0
    peak_level = 1
    peak_occ = 1
    peak_spill_window = 0
    active_thread_sum = tl.full((), 0, tl.int64)
    active_warp_sum = tl.full((), 0, tl.int64)
    # per-lane counters: lane i's registers are Numba thread i's
    # my_processed / my_cas_attempts (int32 per lane, summed as int64 at exit)
    my_processed = tl.zeros([TPB], tl.int32)
    my_cas_attempts = tl.zeros([TPB], tl.int32)

    while total > 0:
        n_ring = sr - sf
        n_spill = gr - gf
        own_size = n_ring + n_spill + (inr - inf_)
        if INSTRUMENTED:
            peak_level = tl.maximum(peak_level, total)
            active = tl.minimum(own_size, TPB)
            active_thread_sum += active
            active_warp_sum += (active + 31) // 32
            if level < trace_cap:
                tl.store(level_sizes + pid * trace_cap + level, own_size)

        # Fused own window [ring | spill | inbox]: one flat index space.
        for base in range(0, own_size, TPB):
            i = base + lanes
            act = i < own_size
            m_ring = act & (i < n_ring)
            m_spill = act & (i >= n_ring) & (i < n_ring + n_spill)
            m_inbox = act & (i >= n_ring + n_spill)
            p_ring = tl.load(my_ring + ((sf + i) & RING_MASK), mask=m_ring,
                             other=0)
            p_spill = tl.load(my_spill + (gf + (i - n_ring)), mask=m_spill,
                              other=0)
            p_inbox = tl.load(my_inbox + (inf_ + (i - n_ring - n_spill)),
                              mask=m_inbox, other=0)
            pixel = tl.where(m_ring, p_ring, tl.where(m_spill, p_spill, p_inbox))
            x = pixel // height
            y = pixel % height

            _recolor(img, pixel, act)
            tl.store(depth + pixel, level, mask=act)
            if INSTRUMENTED:
                tl.store(owner + pixel, pid.to(tl.int8), mask=act)
                my_processed += act.to(tl.int32)

            for d in tl.static_range(4):
                nlin, nx, inb = _neighbor(x, y, act, width, height, d)
                m = _is_red(img, nlin, inb)
                if INSTRUMENTED:
                    my_cas_attempts += m.to(tl.int32)
                won = _claim(visited, nlin, m)
                own_side = (nx < half) == (pid == 0)
                s_rear, s_spill_rear = _enqueue_two_tier(
                    my_ring, my_spill, s_rear, s_spill_rear, sf, nlin,
                    won & own_side)
                _enqueue_global(their_inbox, inbox_cap, g_state,
                                their_rear_slot, nlin, won & (~own_side),
                                counters, False)

        cta_sync()  # Numba: own shared atomics final; thread 0 may read them
        sr_eff = tl.minimum(s_rear, sf + RING_CAPACITY)
        s_rear = sr_eff  # retract tickets that went to the spill tier
        tl.store(g_state + G_PUB_RING0 + 2 * pid, sr_eff - sr)
        tl.store(g_state + G_PUB_SPILL0 + 2 * pid, s_spill_rear - gr)
        epoch += 1
        grid_sync(bar, epoch * nprog)  # SYNC 1: enqueues, rears, pubs visible

        pr0 = tl.load(g_state + G_PUB_RING0)
        ps0 = tl.load(g_state + G_PUB_SPILL0)
        pr1 = tl.load(g_state + G_PUB_RING1)
        ps1 = tl.load(g_state + G_PUB_SPILL1)
        ir0 = tl.load(g_state + G_INBOX_REAR0)
        ir1 = tl.load(g_state + G_INBOX_REAR1)
        epoch += 1
        grid_sync(bar, epoch * nprog)  # SYNC 2: reads done before atomics

        my_ir = tl.where(pid == 0, ir0, ir1)
        oth_ir = tl.where(pid == 0, ir1, ir0)
        pown_r = tl.where(pid == 0, pr0, pr1)
        pown_s = tl.where(pid == 0, ps0, ps1)
        total_next = pr0 + ps0 + pr1 + ps1 + (my_ir - inr) + (oth_ir - oth_inr)
        if INSTRUMENTED:
            peak_spill_window = tl.maximum(peak_spill_window, ps0 + ps1)
            # all six windows across two adjacent levels
            peak_occ = tl.maximum(peak_occ, total + total_next)
        sf = sr
        sr += pown_r
        gf = gr
        gr += pown_s
        inf_ = inr
        inr = my_ir
        oth_inr = oth_ir
        total = total_next
        level += 1

    my_inbox_total = tl.load(g_state + G_INBOX_REAR0 + pid)
    if INSTRUMENTED:
        processed = tl.sum(my_processed.to(tl.int64), axis=0)
        tl.atomic_add(counters + PROCESSED, processed, sem="relaxed",
                      scope="gpu")
        tl.atomic_add(counters + PROCESSED_B0 + pid, processed,
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters + CAS_ATTEMPTS,
                      tl.sum(my_cas_attempts.to(tl.int64), axis=0),
                      sem="relaxed", scope="gpu")
        tl.store(counters + SMID_B0 + pid, read_smid(pid).to(tl.int64))
    # own-tier totals: clamped ring rear counts the ring-stored tickets,
    # spill rear the spilled ones, inbox rear the handed-over ones
    tl.atomic_add(counters + FILLED,
                  (s_rear + s_spill_rear + my_inbox_total).to(tl.int64),
                  sem="relaxed", scope="gpu")
    if INSTRUMENTED:
        tl.atomic_add(counters + SPILLED, s_spill_rear.to(tl.int64),
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters + ACTIVE_THREAD_SUM, active_thread_sum,
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters + ACTIVE_WARP_SUM, active_warp_sum,
                      sem="relaxed", scope="gpu")
        tl.store(counters + SPILLED_B0 + pid, s_spill_rear.to(tl.int64))
        tl.store(counters + INBOX_TO_B0 + pid, my_inbox_total.to(tl.int64))
    if pid == 0:
        tl.store(counters + LEVELS, level.to(tl.int64))
        if INSTRUMENTED:
            tl.store(counters + PEAK_LEVEL, peak_level.to(tl.int64))
            tl.store(counters + PEAK_OCC, peak_occ.to(tl.int64))
            tl.store(counters + PEAK_SPILL_WINDOW,
                     peak_spill_window.to(tl.int64))


@triton.jit(do_not_specialize=["width", "height", "trace_cap"])
def dual_block_dirsplit_kernel(img, visited, depth, owner, queue, q_state,
                               counters, level_sizes, bar, width, height,
                               trace_cap, TPB: tl.constexpr,
                               INSTRUMENTED: tl.constexpr):
    """Partition by discovery direction: right/up claims -> queue 0
    (program 0), down/left claims -> queue 1 (program 1).

    One width*height buffer filled from both ends: queue 0 appends forward
    from slot 0, queue 1 backward from the last slot (ticket i -> slot
    N-1-i). Total appends are claim-bounded by width*height, so the ends
    never collide.

    Host contract: launch (2,) cooperative; visited[seed]=1, queue[0]=seed,
    q_state=[1, 0], depth=-1, counters and bar zeroed.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, TPB)
    cap = width * height

    f0 = 0
    r0 = 1
    f1 = 0
    r1 = 0
    level = 0
    epoch = 0
    peak_level = 1
    peak_occ = 1
    active_thread_sum = tl.full((), 0, tl.int64)
    active_warp_sum = tl.full((), 0, tl.int64)
    # per-lane counters: lane i's registers are Numba thread i's
    # my_processed / my_cas_attempts (int32 per lane, summed as int64 at exit)
    my_processed = tl.zeros([TPB], tl.int32)
    my_cas_attempts = tl.zeros([TPB], tl.int32)

    while (r0 - f0) + (r1 - f1) > 0:
        total = (r0 - f0) + (r1 - f1)
        my_front = tl.where(pid == 0, f0, f1)
        my_rear = tl.where(pid == 0, r0, r1)
        own_size = my_rear - my_front
        if INSTRUMENTED:
            peak_level = tl.maximum(peak_level, total)
            active = tl.minimum(own_size, TPB)
            active_thread_sum += active
            active_warp_sum += (active + 31) // 32
            if level < trace_cap:
                tl.store(level_sizes + pid * trace_cap + level, own_size)

        for base in range(0, own_size, TPB):
            i = base + lanes
            act = i < own_size
            tk = my_front + i
            slot = tl.where(pid == 0, tk, cap - 1 - tk)
            pixel = tl.load(queue + slot, mask=act, other=0)
            x = pixel // height
            y = pixel % height

            _recolor(img, pixel, act)
            tl.store(depth + pixel, level, mask=act)
            if INSTRUMENTED:
                tl.store(owner + pixel, pid.to(tl.int8), mask=act)
                my_processed += act.to(tl.int32)

            for d in tl.static_range(4):
                nlin, nx, inb = _neighbor(x, y, act, width, height, d)
                m = _is_red(img, nlin, inb)
                if INSTRUMENTED:
                    my_cas_attempts += m.to(tl.int32)
                won = _claim(visited, nlin, m)
                if d == 0 or d == 3:  # right/up -> queue 0
                    _enqueue_global(queue, cap, q_state, Q_REAR0, nlin, won,
                                    counters, False)
                else:                 # down/left -> queue 1
                    _enqueue_global(queue, cap, q_state, Q_REAR1, nlin, won,
                                    counters, True)

        epoch += 1
        grid_sync(bar, epoch * nprog)  # both queues' enqueues + rears visible
        nr0 = tl.load(q_state + Q_REAR0)
        nr1 = tl.load(q_state + Q_REAR1)
        epoch += 1
        grid_sync(bar, epoch * nprog)  # everyone has read them

        level += 1
        if INSTRUMENTED:
            peak_occ = tl.maximum(peak_occ, (nr0 - f0) + (nr1 - f1))
        f0 = r0
        r0 = nr0
        f1 = r1
        r1 = nr1

    if INSTRUMENTED:
        processed = tl.sum(my_processed.to(tl.int64), axis=0)
        tl.atomic_add(counters + PROCESSED, processed, sem="relaxed",
                      scope="gpu")
        tl.atomic_add(counters + PROCESSED_B0 + pid, processed,
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters + CAS_ATTEMPTS,
                      tl.sum(my_cas_attempts.to(tl.int64), axis=0),
                      sem="relaxed", scope="gpu")
        tl.store(counters + SMID_B0 + pid, read_smid(pid).to(tl.int64))
        tl.atomic_add(counters + ACTIVE_THREAD_SUM, active_thread_sum,
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters + ACTIVE_WARP_SUM, active_warp_sum,
                      sem="relaxed", scope="gpu")
    if pid == 0:
        tl.store(counters + FILLED,
                 (tl.load(q_state + Q_REAR0)
                  + tl.load(q_state + Q_REAR1)).to(tl.int64))
        tl.store(counters + LEVELS, level.to(tl.int64))
        if INSTRUMENTED:
            tl.store(counters + PEAK_LEVEL, peak_level.to(tl.int64))
            tl.store(counters + PEAK_OCC, peak_occ.to(tl.int64))


@triton.jit(do_not_specialize=["width", "height"])
def dual_block_pinned_kernel(img, visited, depth, queue, q_state,
                             barrier_state, pin_state, counters, width,
                             height, TPB: tl.constexpr):
    """The placement experiment: global-queue BFS, hand-rolled pair barrier.

    pin_state[P_MODE] == 0 (same_sm): launched with C * sm_count programs,
    where C is the kernel's resident programs per SM, so every SM hosts
    exactly C (the cooperative launch proves co-residency; this kernel
    never calls grid_sync). Each program reads %smid; the first CASes its
    smid into pin_state; the first two programs matching it take worker
    ranks 0/1 and run the whole BFS sharing one SM; the others exit.

    pin_state[P_MODE] == 1 (spread): launched (2,); programs 0/1 are the
    workers, placed by the scheduler (the recorded smids show where).

    Instrumentation is minimal by design (FILLED/LEVELS/SMID).
    """
    pid = tl.program_id(0)
    lanes = tl.arange(0, TPB)
    cap = width * height

    if tl.load(pin_state + P_MODE) == 0:
        my_smid = read_smid(pid)
        tl.atomic_cas(pin_state + P_CHOSEN_SMID, -1, my_smid, sem="relaxed",
                      scope="gpu")
        # the CAS made the slot stable either way; an atomic read gives
        # every program the winning smid
        chosen = tl.atomic_add(pin_state + P_CHOSEN_SMID, 0, sem="relaxed",
                               scope="gpu")
        if my_smid == chosen:
            rank = tl.atomic_add(pin_state + P_WORKER_COUNT, 1, sem="relaxed",
                                 scope="gpu")
        else:
            rank = -1
    else:
        rank = pid
    cta_sync()

    # Numba: "if rank < 0: return". With C > 2 programs on the chosen SM
    # the ranks past 1 also leave (Numba's 768-thread launch has C == 2).
    if (rank >= 0) & (rank < 2):
        tl.store(counters + SMID_B0 + rank, read_smid(pid).to(tl.int64))
        stride = 2 * TPB
        front = 0
        rear = 1
        level = 0
        while front < rear:
            for start in range(front + rank * TPB, rear, stride):
                i = start + lanes
                act = i < rear
                pixel = tl.load(queue + i, mask=act, other=0)
                x = pixel // height
                y = pixel % height
                _recolor(img, pixel, act)
                tl.store(depth + pixel, level, mask=act)
                for d in tl.static_range(4):
                    nlin, nx, inb = _neighbor(x, y, act, width, height, d)
                    m = _is_red(img, nlin, inb)
                    won = _claim(visited, nlin, m)
                    _enqueue_global(queue, cap, q_state, Q_REAR, nlin, won,
                                    counters, False)

            _pair_barrier(barrier_state, 2)  # enqueues + rear visible
            new_rear = tl.load(q_state + Q_REAR)
            _pair_barrier(barrier_state, 2)  # both have read it

            level += 1
            front = rear
            rear = new_rear

        if rank == 0:
            tl.store(counters + FILLED, tl.load(q_state + Q_REAR).to(tl.int64))
            tl.store(counters + LEVELS, level.to(tl.int64))
