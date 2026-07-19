"""Dual-block BFS flood fill: 1 block -> 2 blocks, three partitionings + pinning.

Five cooperative-era kernels share one claim protocol (bounds -> is_red ->
atomic.cas(visited) -> enqueue) and one level-synchronous structure; they
differ ONLY in how two blocks share the frontier and how they meet at the
level barrier:

- "global"   one monotonic global queue, both blocks grid-stride each level
             window; two grid.sync() per level (persistent/'s design at
             4-connectivity, with this package's counters).
- "split"    domain decomposition: block b owns half the image and runs the
             single-block v2 two-tier machinery (shared ring + spill) on its
             half; discoveries across the seam x = width//2 are appended to
             the OTHER block's global-memory inbox. 1 syncthreads + 2
             grid.sync per level.
- "dirsplit" partition by discovery direction: pixels claimed via right/up
             edges go to queue 0 (block 0), via down/left to queue 1
             (block 1). One width*height buffer filled from BOTH ends -- q0
             appends forward, q1 backward -- and since every pixel is
             CAS-claimed at most once, total appends <= width*height and the
             two ends can never collide.
- "pinned"   the placement experiment: the global-queue BFS with the level
             barrier replaced by a hand-rolled 2-block barrier, plus %smid
             self-identification so the two workers can be forced onto the
             SAME SM (launch 48 blocks x 768 threads: the occupancy limit
             floor(1536/768) = 2 means every SM hosts exactly two; the two
             blocks sharing the chosen SM work, the other 46 exit).
- bare twins of split/global/dirsplit: identical BFS with all per-level and
  per-thread instrumentation stripped, to MEASURE (not assume) the
  instrumentation overhead.

Cross-block safety facts the kernels rely on (see README for the argument):
shared memory is block-private even between co-resident blocks, so ALL
cross-block data rides global memory; queue/inbox slots are written once and
read fresh (no stale-L1 hazard); a stale red read of a recolored pixel just
loses the CAS; grid.sync / the pair barrier order everything else.

Structural no-overflow arguments: global/pinned queue and the dirsplit
double-ended buffer hold <= width*height CAS-claimed entries; each split
inbox receives only pixels of the single column adjacent to the seam, each
claimed once -> <= height entries; split spill tiers hold only own-half
pixels -> <= half-area entries. OVERFLOW is a defensive tripwire on all of
them; the host raising on it means a kernel bug, not a capacity limit.
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda, int32

_HERE = os.path.dirname(os.path.abspath(__file__))
SMID_CU = os.path.join(_HERE, "smid.cu")

RING_CAPACITY = 8192
RING_MASK = RING_CAPACITY - 1

# 4-connectivity neighbor offsets (matches reference.py: right, down, left, up)
DX_HOST = np.array([1, 0, -1, 0], dtype=np.int32)
DY_HOST = np.array([0, 1, 0, -1], dtype=np.int32)

# Slots in the device-side int64 counters array (0-10 match single_block_shared)
FILLED = 0
LEVELS = 1
OVERFLOW = 2            # defensive tripwire: structurally unreachable
PEAK_LEVEL = 3          # largest single global frontier
PEAK_OCC = 4            # max queue entries alive across two adjacent levels
ACTIVE_THREAD_SUM = 5   # sum over levels+blocks of min(own_size, tpb)
ACTIVE_WARP_SUM = 6     # sum over levels+blocks of ceil(min(own_size, tpb)/32)
PROCESSED = 7           # pixels dequeued/recolored (== FILLED iff exactly-once)
CAS_ATTEMPTS = 8        # visited-CAS ops tried
SPILLED = 9             # split: total pixels sent to both spill tiers
PEAK_SPILL_WINDOW = 10  # split: largest single-level spill count (both halves)
PROCESSED_B0 = 11       # per-block work counts -> load balance
PROCESSED_B1 = 12
SPILLED_B0 = 13
SPILLED_B1 = 14
INBOX_TO_B0 = 15        # split: pixels handed across the seam to block 0
INBOX_TO_B1 = 16
SMID_B0 = 17            # %smid each block observed itself running on
SMID_B1 = 18
NUM_COUNTERS = 19

# q_state slots
Q_REAR = 0              # "global"/"pinned": the single queue's rear
Q_REAR0 = 0             # "dirsplit": forward queue (right/up claims)
Q_REAR1 = 1             # "dirsplit": backward queue (down/left claims)

# g_state slots ("split")
G_INBOX_REAR0 = 0       # rear of the inbox TO block 0 (bumped by block 1)
G_INBOX_REAR1 = 1       # rear of the inbox TO block 1 (bumped by block 0)
G_PUB_RING0 = 2         # block 0's published next-level ring count
G_PUB_SPILL0 = 3        # block 0's published next-level spill count
G_PUB_RING1 = 4
G_PUB_SPILL1 = 5

# pin_state slots ("pinned")
P_MODE = 0              # host-set, immutable: 0 = same_sm, 1 = spread
P_CHOSEN_SMID = 1       # -1 until the first block CASes its smid in
P_WORKER_COUNT = 2      # worker-rank dispenser

# barrier_state slots ("pinned")
BAR_ARRIVE = 0
BAR_GEN = 1

# %smid reader linked from smid.cu (verified working in this environment)
get_smid = cuda.declare_device('get_smid', 'uint32()')


@cuda.jit(device=True, inline=True)
def _is_red(img, x, y):
    """Check if pixel is red (255, 0, 0)."""
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


@cuda.jit(device=True, inline=True)
def _warp_enqueue_two_tier(ring, spill, s_rear, s_spill_rear, front, item):
    """Own-half enqueue: shared ring fast path, own spill tier past it.

    Verbatim port of single_block_shared v2: all lanes arriving together
    aggregate (activemask/popc/shfl), the lowest lane reserves a ticket slab
    with one shared atomic; tickets inside [front, front + RING_CAPACITY)
    take ring slots (front frozen per level -> distinct slots); the lanes
    whose tickets straddle past the window are exactly the ones active in
    the else branch, so they aggregate again into the spill tier.
    """
    mask = cuda.activemask()
    count = cuda.popc(mask)
    rank = cuda.popc(mask & cuda.lanemask_lt())
    leader = cuda.ffs(mask) - 1  # ffs is 1-based; mask always has a bit set
    base = 0
    if cuda.laneid == leader:
        base = cuda.atomic.add(s_rear, 0, count)
    base = cuda.shfl_sync(mask, base, leader)
    ticket = base + rank
    if ticket - front < RING_CAPACITY:
        ring[ticket & RING_MASK] = item
    else:
        mask2 = cuda.activemask()
        count2 = cuda.popc(mask2)
        rank2 = cuda.popc(mask2 & cuda.lanemask_lt())
        leader2 = cuda.ffs(mask2) - 1
        gbase = 0
        if cuda.laneid == leader2:
            gbase = cuda.atomic.add(s_spill_rear, 0, count2)
        gbase = cuda.shfl_sync(mask2, gbase, leader2)
        spill[gbase + rank2] = item


@cuda.jit(device=True, inline=True)
def _warp_enqueue_global(arr, state, rear_slot, item, backward, counters):
    """Warp-aggregated append on a global rear counter (one atomic per warp).

    backward=True writes slot arr.shape[0]-1-idx: the dirsplit kernel fills
    one buffer from both ends (total appends <= capacity, so the ends can
    never collide). The bound check is a defensive tripwire only.
    """
    mask = cuda.activemask()
    count = cuda.popc(mask)
    rank = cuda.popc(mask & cuda.lanemask_lt())
    leader = cuda.ffs(mask) - 1
    base = 0
    if cuda.laneid == leader:
        base = cuda.atomic.add(state, rear_slot, count)
    base = cuda.shfl_sync(mask, base, leader)
    idx = base + rank
    if idx < arr.shape[0]:
        if backward:
            arr[arr.shape[0] - 1 - idx] = item
        else:
            arr[idx] = item
    else:
        counters[OVERFLOW] = 1  # unreachable by the structural arguments


@cuda.jit(device=True)
def _pair_barrier(barrier_state, n_workers):
    """grid.sync() built by hand for co-resident worker blocks.

    Sense-reversing barrier through global memory: after the block-local
    syncthreads (all of this block's stores are issued), thread 0 snapshots
    the generation, then atomically arrives; the last arriver resets the
    arrival count, threadfence()s (its ordered view of everyone's prior
    stores becomes device-visible), and bumps the generation; everyone else
    spins on the generation with ATOMIC reads (plain global reads may be
    register-cached and would spin forever). Deadlock-free only because the
    workers are guaranteed co-resident from launch — exactly the guarantee
    cooperative launches formalize, which is why this is normally grid.sync.
    """
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        gen = cuda.atomic.add(barrier_state, BAR_GEN, 0)  # atomic read
        if cuda.atomic.add(barrier_state, BAR_ARRIVE, 1) == n_workers - 1:
            barrier_state[BAR_ARRIVE] = 0
            cuda.threadfence()
            cuda.atomic.add(barrier_state, BAR_GEN, 1)  # release
        else:
            while cuda.atomic.add(barrier_state, BAR_GEN, 0) == gen:
                pass
        cuda.threadfence()
    cuda.syncthreads()


@cuda.jit(link=[SMID_CU])
def dual_block_global_kernel(img, visited, depth, owner, queue, q_state,
                             counters, level_sizes):
    """One shared global queue, both blocks grid-stride each level window.

    Host contract: launch [2, tpb] (cooperative — grid.sync inside);
    visited[seed]=1, queue[0]=seed linear index, q_state=[1], depth=-1,
    counters zeroed. Two grid.sync() per level for persistent/'s reasons:
    #1 makes the level's enqueues and rear visible grid-wide, #2 guarantees
    everyone has read the new rear before any thread's next-level atomics.
    """
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    front = 0
    rear = 1
    level = 0
    peak_level = 1
    peak_occ = 1
    active_thread_sum = 0
    active_warp_sum = 0
    my_processed = 0
    my_cas_attempts = 0

    while front < rear:
        level_size = rear - front
        if level_size > peak_level:
            peak_level = level_size
        active = min(level_size, stride)
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if cuda.threadIdx.x == 0 and level < level_sizes.shape[1]:
            # row 0 carries the trace; row 1 stays a clean zero row
            level_sizes[bx, level] = level_size if bx == 0 else 0

        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map (wavefront viz)
            my_processed += 1

            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, False, counters)

        grid.sync()  # enqueues + final rear for this level visible grid-wide
        new_rear = q_state[Q_REAR]
        grid.sync()  # everyone has read new_rear; next level's atomics may begin

        level += 1
        occ = new_rear - front
        if occ > peak_occ:
            peak_occ = occ
        front = rear
        rear = new_rear

    cuda.atomic.add(counters, PROCESSED, my_processed)
    cuda.atomic.add(counters, PROCESSED_B0 + bx, my_processed)
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    if cuda.threadIdx.x == 0:
        counters[SMID_B0 + bx] = get_smid()
    if tid == 0:
        # all grid-uniform register values
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum


# max_registers=120: the split kernel's natural allocation (159/thread)
# would bar 512-thread blocks from residency (512*159 > 64K registers);
# 120 keeps every supported tpb resident. The bare twin gets the same
# cap so the instrumentation-overhead comparison stays register-fair.
@cuda.jit(link=[SMID_CU], max_registers=120)
def dual_block_split_kernel(img, visited, depth, owner, seed_x, seed_y,
                            spill0, spill1, inbox0, inbox1, g_state,
                            counters, level_sizes):
    """Domain decomposition: block 0 owns x < width//2, block 1 the rest.

    Each block runs the single-block v2 two-tier machinery on its half:
    private shared ring + own spill tier. Cross-seam discoveries (possible
    only along the edge x = width//2 - 1 <-> width//2 in 4-connectivity) are
    appended to the OTHER block's inbox in global memory.

    Host contract: launch [2, tpb] cooperative; visited[seed]=1 (host-set —
    sidesteps cross-block init visibility); g_state zeroed; spill_b sized to
    block b's half, inboxes sized height; depth=-1; counters zeroed. The
    seed-owning block seeds its own ring in the prologue.

    Level boundary = 1 syncthreads + 2 grid.sync: after processing, the
    block-local syncthreads finalizes this block's shared atomics; thread 0
    clamps its shared rear (retracting tickets that diverted to the spill
    tier — race-free because NO other thread ever reads s_rear; unlike v2,
    every thread takes its next windows from the published g_state counts)
    and publishes next-level ring/spill counts; grid.sync #1 makes enqueues,
    inbox rears and both blocks' pubs visible; all threads read the six
    g_state values; grid.sync #2 orders those reads before the next level's
    atomics. The loop predicate (total work over all six windows) is
    grid-uniform, so both blocks always execute identical barrier sequences.
    """
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    ltid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    width = img.shape[0]
    height = img.shape[1]
    half = width // 2

    ring = cuda.shared.array(RING_CAPACITY, int32)
    s_rear = cuda.shared.array(1, int32)
    s_spill_rear = cuda.shared.array(1, int32)

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    if bx == 0:
        my_spill = spill0
        my_inbox = inbox0
        their_inbox = inbox1
        their_rear_slot = G_INBOX_REAR1
    else:
        my_spill = spill1
        my_inbox = inbox1
        their_inbox = inbox0
        their_rear_slot = G_INBOX_REAR0

    seed_owner = 0 if seed_x < half else 1
    if ltid == 0:
        s_spill_rear[0] = 0
        if bx == seed_owner:
            ring[0] = seed_x * height + seed_y
            s_rear[0] = 1
        else:
            s_rear[0] = 0
    cuda.syncthreads()

    sf = 0                          # own ring window (virtual tickets)
    sr = 1 if bx == seed_owner else 0
    gf = 0                          # own spill window
    gr = 0
    inf_ = 0                        # own inbox window
    inr = 0
    oth_inr = 0                     # other block's inbox rear (shadow copy)
    total = 1                       # GLOBAL level size — grid-uniform
    level = 0
    peak_level = 1
    peak_occ = 1
    peak_spill_window = 0
    active_thread_sum = 0
    active_warp_sum = 0
    my_processed = 0
    my_cas_attempts = 0

    while total > 0:
        n_ring = sr - sf
        n_spill = gr - gf
        own_size = n_ring + n_spill + (inr - inf_)
        if total > peak_level:
            peak_level = total
        active = min(own_size, nthreads)
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if ltid == 0 and level < level_sizes.shape[1]:
            level_sizes[bx, level] = own_size

        # Fused own window [ring | spill | inbox]: one flat index space so
        # no thread idles just because one segment happens to be empty.
        for i in range(ltid, own_size, nthreads):
            if i < n_ring:
                pixel = ring[(sf + i) & RING_MASK]
            elif i < n_ring + n_spill:
                pixel = my_spill[gf + (i - n_ring)]
            else:
                pixel = my_inbox[inf_ + (i - n_ring - n_spill)]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map (wavefront viz)
            my_processed += 1

            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        item = nx * height + ny
                        if (nx < half) == (bx == 0):
                            _warp_enqueue_two_tier(ring, my_spill, s_rear,
                                                   s_spill_rear, sf, item)
                        else:
                            _warp_enqueue_global(their_inbox, g_state,
                                                 their_rear_slot, item,
                                                 False, counters)

        cuda.syncthreads()  # own shared atomics final; thread 0 may read them
        if ltid == 0:
            sr_raw = s_rear[0]
            gr_new = s_spill_rear[0]
            sr_eff = min(sr_raw, sf + RING_CAPACITY)
            if sr_eff != sr_raw:
                s_rear[0] = sr_eff  # retract tickets that went to the spill
            g_state[G_PUB_RING0 + 2 * bx] = sr_eff - sr
            g_state[G_PUB_SPILL0 + 2 * bx] = gr_new - gr
        grid.sync()  # SYNC 1: enqueues, inbox rears, both pubs visible

        pr0 = g_state[G_PUB_RING0]
        ps0 = g_state[G_PUB_SPILL0]
        pr1 = g_state[G_PUB_RING1]
        ps1 = g_state[G_PUB_SPILL1]
        ir0 = g_state[G_INBOX_REAR0]
        ir1 = g_state[G_INBOX_REAR1]
        grid.sync()  # SYNC 2: reads done before next level's atomics

        if bx == 0:
            my_ir = ir0
            oth_ir = ir1
            pown_r = pr0
            pown_s = ps0
        else:
            my_ir = ir1
            oth_ir = ir0
            pown_r = pr1
            pown_s = ps1
        total_next = pr0 + ps0 + pr1 + ps1 + (my_ir - inr) + (oth_ir - oth_inr)
        if ps0 + ps1 > peak_spill_window:
            peak_spill_window = ps0 + ps1
        occ = total + total_next  # all six windows across two adjacent levels
        if occ > peak_occ:
            peak_occ = occ
        sf = sr
        sr += pown_r
        gf = gr
        gr += pown_s
        inf_ = inr
        inr = my_ir
        oth_inr = oth_ir
        total = total_next
        level += 1

    cuda.atomic.add(counters, PROCESSED, my_processed)
    cuda.atomic.add(counters, PROCESSED_B0 + bx, my_processed)
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    if ltid == 0:
        counters[SMID_B0 + bx] = get_smid()
        # own-tier totals: clamped ring rear counts exactly the ring-stored
        # tickets, spill rear the spilled ones, inbox rear the handed-over
        # ones — every claimed pixel lands in exactly one of the three.
        cuda.atomic.add(counters, FILLED,
                        s_rear[0] + s_spill_rear[0]
                        + g_state[G_INBOX_REAR0 + bx])
        cuda.atomic.add(counters, SPILLED, s_spill_rear[0])
        cuda.atomic.add(counters, ACTIVE_THREAD_SUM, active_thread_sum)
        cuda.atomic.add(counters, ACTIVE_WARP_SUM, active_warp_sum)
        counters[SPILLED_B0 + bx] = s_spill_rear[0]
        counters[INBOX_TO_B0 + bx] = g_state[G_INBOX_REAR0 + bx]
        if bx == 0:
            counters[LEVELS] = level
            counters[PEAK_LEVEL] = peak_level
            counters[PEAK_OCC] = peak_occ
            counters[PEAK_SPILL_WINDOW] = peak_spill_window


@cuda.jit(link=[SMID_CU])
def dual_block_dirsplit_kernel(img, visited, depth, owner, queue, q_state,
                               counters, level_sizes):
    """Partition by discovery direction: right/up claims -> queue 0 (block 0),
    down/left claims -> queue 1 (block 1).

    One width*height buffer filled from both ends: queue 0 appends forward
    from slot 0, queue 1 backward from the last slot (ticket i -> slot
    N-1-i). Total appends are CAS-bounded by width*height, so the two ends
    can structurally never collide. Which queue a pixel lands in depends on
    which direction's claimer won the CAS — race-dependent, so per-queue
    counts are nondeterministic while visited/depth/filled stay exact.

    Host contract: launch [2, tpb] cooperative; visited[seed]=1,
    queue[0]=seed, q_state=[1,0], depth=-1, counters zeroed.
    """
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    ltid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    width = img.shape[0]
    height = img.shape[1]
    cap = queue.shape[0]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    f0 = 0
    r0 = 1
    f1 = 0
    r1 = 0
    level = 0
    peak_level = 1
    peak_occ = 1
    active_thread_sum = 0
    active_warp_sum = 0
    my_processed = 0
    my_cas_attempts = 0

    while (r0 - f0) + (r1 - f1) > 0:
        total = (r0 - f0) + (r1 - f1)
        if bx == 0:
            my_front = f0
            my_rear = r0
        else:
            my_front = f1
            my_rear = r1
        own_size = my_rear - my_front
        if total > peak_level:
            peak_level = total
        active = min(own_size, nthreads)
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if ltid == 0 and level < level_sizes.shape[1]:
            level_sizes[bx, level] = own_size

        for i in range(ltid, own_size, nthreads):
            t = my_front + i
            if bx == 0:
                pixel = queue[t]
            else:
                pixel = queue[cap - 1 - t]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map (wavefront viz)
            my_processed += 1

            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        item = nx * height + ny
                        if d == 0 or d == 3:  # right/up -> queue 0
                            _warp_enqueue_global(queue, q_state, Q_REAR0,
                                                 item, False, counters)
                        else:                 # down/left -> queue 1
                            _warp_enqueue_global(queue, q_state, Q_REAR1,
                                                 item, True, counters)

        grid.sync()  # both queues' enqueues + rears visible grid-wide
        nr0 = q_state[Q_REAR0]
        nr1 = q_state[Q_REAR1]
        grid.sync()  # everyone has read them; next level's atomics may begin

        level += 1
        occ = (nr0 - f0) + (nr1 - f1)
        if occ > peak_occ:
            peak_occ = occ
        f0 = r0
        r0 = nr0
        f1 = r1
        r1 = nr1

    cuda.atomic.add(counters, PROCESSED, my_processed)
    cuda.atomic.add(counters, PROCESSED_B0 + bx, my_processed)
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    if ltid == 0:
        counters[SMID_B0 + bx] = get_smid()
        cuda.atomic.add(counters, ACTIVE_THREAD_SUM, active_thread_sum)
        cuda.atomic.add(counters, ACTIVE_WARP_SUM, active_warp_sum)
        if bx == 0:
            counters[FILLED] = q_state[Q_REAR0] + q_state[Q_REAR1]
            counters[LEVELS] = level
            counters[PEAK_LEVEL] = peak_level
            counters[PEAK_OCC] = peak_occ


# max_registers=40: two 768-thread blocks per SM need <= 65536/1536 = 42.67
# registers per thread, and the hardware allocates in per-warp granules of
# 8, so 42 rounds up to 48 and drops cooperative occupancy to 1 block/SM;
# 40 is the largest granule-aligned count that keeps 2 blocks resident.
@cuda.jit(link=[SMID_CU], max_registers=40)
def dual_block_pinned_kernel(img, visited, depth, queue, q_state,
                             barrier_state, pin_state, counters):
    """The placement experiment: global-queue BFS, hand-rolled pair barrier.

    pin_state[P_MODE] == 0 (same_sm): launched [48, 768]. floor(1536/768)=2
    blocks fit per SM, and 48 = 24 SMs x 2, so every SM hosts exactly two
    blocks (cooperative launch guarantees simultaneous residency; this
    kernel never calls grid.sync, it only uses this_grid() for the
    guarantee). Each block reads %smid; the first block CASes its smid into
    pin_state; the two blocks matching it take worker ranks 0/1 and run the
    whole BFS provably sharing one SM; the other 46 blocks exit.

    pin_state[P_MODE] == 1 (spread): launched [2, 768]; blocks 0/1 are the
    workers, placed naturally by the scheduler (recorded smids prove the
    spread). Same code path, same barrier — placement is the only variable.

    Instrumentation is minimal by design (FILLED/LEVELS/SMID): this kernel
    exists to measure placement, not to be observed in detail.
    """
    grid = cuda.cg.this_grid()  # noqa: F841 — forces a cooperative launch
    bx = cuda.blockIdx.x
    ltid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    s_rank = cuda.shared.array(1, int32)

    if ltid == 0:
        if pin_state[P_MODE] == 0:
            my_smid = int32(get_smid())
            cuda.atomic.cas(pin_state, P_CHOSEN_SMID, -1, my_smid)
            # CAS made the slot stable either way; an atomic read gives
            # every block the winning smid.
            chosen = cuda.atomic.add(pin_state, P_CHOSEN_SMID, 0)
            if my_smid == chosen:
                s_rank[0] = cuda.atomic.add(pin_state, P_WORKER_COUNT, 1)
            else:
                s_rank[0] = -1
        else:
            s_rank[0] = bx
    cuda.syncthreads()
    rank = s_rank[0]
    if rank < 0:
        return  # not a worker: free this SM slot immediately

    if ltid == 0:
        counters[SMID_B0 + rank] = get_smid()

    wtid = rank * nthreads + ltid
    stride = 2 * nthreads

    front = 0
    rear = 1
    level = 0

    while front < rear:
        for i in range(front + wtid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level

            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, False, counters)

        _pair_barrier(barrier_state, 2)  # enqueues + rear visible to the pair
        new_rear = q_state[Q_REAR]
        _pair_barrier(barrier_state, 2)  # both have read it; atomics may resume

        level += 1
        front = rear
        rear = new_rear

    if rank == 0 and ltid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# --------------------------------------------------------------- bare twins
# Identical BFS with all per-level/per-thread instrumentation stripped (only
# exit-time FILLED/LEVELS remain — two stores, zero steady-state cost) to
# measure the instrumented kernels' observer overhead. They share every
# device helper above so the algorithm cannot drift from the instrumented
# versions; only counter/trace lines differ.


@cuda.jit
def dual_block_global_bare_kernel(img, visited, depth, queue, q_state,
                                  counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    front = 0
    rear = 1
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, False, counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


@cuda.jit(max_registers=120)
def dual_block_split_bare_kernel(img, visited, depth, seed_x, seed_y,
                                 spill0, spill1, inbox0, inbox1, g_state,
                                 counters):
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    ltid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    width = img.shape[0]
    height = img.shape[1]
    half = width // 2

    ring = cuda.shared.array(RING_CAPACITY, int32)
    s_rear = cuda.shared.array(1, int32)
    s_spill_rear = cuda.shared.array(1, int32)

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    if bx == 0:
        my_spill = spill0
        my_inbox = inbox0
        their_inbox = inbox1
        their_rear_slot = G_INBOX_REAR1
    else:
        my_spill = spill1
        my_inbox = inbox1
        their_inbox = inbox0
        their_rear_slot = G_INBOX_REAR0

    seed_owner = 0 if seed_x < half else 1
    if ltid == 0:
        s_spill_rear[0] = 0
        if bx == seed_owner:
            ring[0] = seed_x * height + seed_y
            s_rear[0] = 1
        else:
            s_rear[0] = 0
    cuda.syncthreads()

    sf = 0
    sr = 1 if bx == seed_owner else 0
    gf = 0
    gr = 0
    inf_ = 0
    inr = 0
    oth_inr = 0
    total = 1
    level = 0

    while total > 0:
        n_ring = sr - sf
        n_spill = gr - gf
        own_size = n_ring + n_spill + (inr - inf_)

        for i in range(ltid, own_size, nthreads):
            if i < n_ring:
                pixel = ring[(sf + i) & RING_MASK]
            elif i < n_ring + n_spill:
                pixel = my_spill[gf + (i - n_ring)]
            else:
                pixel = my_inbox[inf_ + (i - n_ring - n_spill)]
            x = pixel // height
            y = pixel % height
            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        item = nx * height + ny
                        if (nx < half) == (bx == 0):
                            _warp_enqueue_two_tier(ring, my_spill, s_rear,
                                                   s_spill_rear, sf, item)
                        else:
                            _warp_enqueue_global(their_inbox, g_state,
                                                 their_rear_slot, item,
                                                 False, counters)

        cuda.syncthreads()
        if ltid == 0:
            sr_raw = s_rear[0]
            gr_new = s_spill_rear[0]
            sr_eff = min(sr_raw, sf + RING_CAPACITY)
            if sr_eff != sr_raw:
                s_rear[0] = sr_eff
            g_state[G_PUB_RING0 + 2 * bx] = sr_eff - sr
            g_state[G_PUB_SPILL0 + 2 * bx] = gr_new - gr
        grid.sync()

        pr0 = g_state[G_PUB_RING0]
        ps0 = g_state[G_PUB_SPILL0]
        pr1 = g_state[G_PUB_RING1]
        ps1 = g_state[G_PUB_SPILL1]
        ir0 = g_state[G_INBOX_REAR0]
        ir1 = g_state[G_INBOX_REAR1]
        grid.sync()

        if bx == 0:
            my_ir = ir0
            oth_ir = ir1
            pown_r = pr0
            pown_s = ps0
        else:
            my_ir = ir1
            oth_ir = ir0
            pown_r = pr1
            pown_s = ps1
        total = (pr0 + ps0 + pr1 + ps1
                 + (my_ir - inr) + (oth_ir - oth_inr))
        sf = sr
        sr += pown_r
        gf = gr
        gr += pown_s
        inf_ = inr
        inr = my_ir
        oth_inr = oth_ir
        level += 1

    if ltid == 0:
        cuda.atomic.add(counters, FILLED,
                        s_rear[0] + s_spill_rear[0]
                        + g_state[G_INBOX_REAR0 + bx])
        if bx == 0:
            counters[LEVELS] = level


@cuda.jit
def dual_block_dirsplit_bare_kernel(img, visited, depth, queue, q_state,
                                    counters):
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    ltid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    width = img.shape[0]
    height = img.shape[1]
    cap = queue.shape[0]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    f0 = 0
    r0 = 1
    f1 = 0
    r1 = 0
    level = 0

    while (r0 - f0) + (r1 - f1) > 0:
        if bx == 0:
            my_front = f0
            my_rear = r0
        else:
            my_front = f1
            my_rear = r1
        own_size = my_rear - my_front

        for i in range(ltid, own_size, nthreads):
            t = my_front + i
            if bx == 0:
                pixel = queue[t]
            else:
                pixel = queue[cap - 1 - t]
            x = pixel // height
            y = pixel % height
            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        item = nx * height + ny
                        if d == 0 or d == 3:
                            _warp_enqueue_global(queue, q_state, Q_REAR0,
                                                 item, False, counters)
                        else:
                            _warp_enqueue_global(queue, q_state, Q_REAR1,
                                                 item, True, counters)

        grid.sync()
        nr0 = q_state[Q_REAR0]
        nr1 = q_state[Q_REAR1]
        grid.sync()

        level += 1
        f0 = r0
        r0 = nr0
        f1 = r1
        r1 = nr1

    if bx == 0 and ltid == 0:
        counters[FILLED] = q_state[Q_REAR0] + q_state[Q_REAR1]
        counters[LEVELS] = level
