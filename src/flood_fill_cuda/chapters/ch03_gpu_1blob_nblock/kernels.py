"""Multi-block BFS flood fill: 2 blocks -> N blocks, one global queue.

This stage scales the dual-block stage's measured winner — the global-memory
queue kernel — to any cooperative grid size. The algorithm is unchanged from
dual_block's "global" kernel: one monotonic global queue, every block
grid-strides each level window, one shared claim protocol (bounds -> is_red
-> atomic.cas(visited) -> enqueue), two grid.sync() per level. What changes
is only the instrumentation, which must stop assuming two blocks:

- per-block work counts and %smid observations move from fixed counter slots
  (PROCESSED_B0/B1, SMID_B0/B1) to a (blocks, 2) int64 block_stats array;
- the per-level frontier trace collapses from a (2, cap) per-block array to
  a single grid-wide 1D trace: at hundreds of cooperative blocks a per-block
  per-level trace would cost gigabytes (48 x 2^21 x 4 B = 400 MB; 576 blocks
  -> 4.8 GB), and the owner map already answers "who filled what";
- owner widens int8 -> int16: coop capacity at tpb=64 is 24 blocks/SM x
  24 SMs = 576 blocks, past int8's 127.

Structural no-overflow argument (unchanged from dual_block): the queue holds
width*height int32 slots, every pixel is CAS-claimed at most once before
enqueue, so total appends <= filled <= width*height. OVERFLOW stays a
defensive tripwire; the host raising on it means a kernel bug, not capacity.
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
SMID_CU = os.path.join(_HERE, "smid.cu")

# 4-connectivity neighbor offsets (matches reference.py: right, down, left, up)
DX_HOST = np.array([1, 0, -1, 0], dtype=np.int32)
DY_HOST = np.array([0, 1, 0, -1], dtype=np.int32)

# 8-connectivity offsets (matches persistent/reference.py: E SE S SW W NW N NE).
# The conn8 kernels below are verbatim twins of the 4-conn pair rather than a
# single kernel with a runtime n_dirs loop: a dynamic loop bound would change
# the 4-conn baseline's codegen and perturb every number already published.
DX8_HOST = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
DY8_HOST = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# Ring-2 offsets: the 16 cells at Chebyshev distance exactly 2 (the outer
# ring of the 5x5 neighborhood), clockwise from E. Probed by the radius-2
# twins ONLY when all 8 ring-1 neighbors are in-bounds blob material — the
# guard that keeps every jump inside true 8-connectivity (each ring-2 cell
# is 8-adjacent to a ring-1 cell: clamp each coordinate toward 0 by one).
DX_R2_HOST = np.array([2, 2, 2, 1, 0, -1, -2, -2,
                       -2, -2, -2, -1, 0, 1, 2, 2], dtype=np.int32)
DY_R2_HOST = np.array([0, 1, 2, 2, 2, 2, 2, 1,
                       0, -1, -2, -2, -2, -2, -2, -1], dtype=np.int32)

# Slots in the device-side int64 counters array (grid-wide scalars only;
# 0-8 match dual_block — the per-block slots 11+ moved to block_stats.
# Slot 9 (INTERIOR) is written only by the radius-2 twins; all other
# kernels leave the host-zeroed value untouched.)
FILLED = 0
LEVELS = 1
OVERFLOW = 2            # defensive tripwire: structurally unreachable
PEAK_LEVEL = 3          # largest single frontier
PEAK_OCC = 4            # max queue entries alive across two adjacent levels
ACTIVE_THREAD_SUM = 5   # sum over levels of min(level_size, grid threads)
ACTIVE_WARP_SUM = 6     # sum over levels of ceil(min(level_size, grid)/32)
PROCESSED = 7           # pixels dequeued/recolored (== FILLED iff exactly-once)
CAS_ATTEMPTS = 8        # visited-CAS ops tried
INTERIOR = 9            # radius-2 twins: pixels whose ring-1 was all blob
NUM_COUNTERS = 10

# Columns of the (blocks, 2) int64 block_stats array
BS_PROCESSED = 0        # pixels this block dequeued -> N-way load balance
BS_SMID = 1             # %smid this block observed itself running on

# q_state slots
Q_REAR = 0

# %smid reader linked from smid.cu
get_smid = cuda.declare_device('get_smid', 'uint32()')


@cuda.jit(device=True, inline=True)
def _is_red(img, x, y):
    """Check if pixel is red (255, 0, 0)."""
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


@cuda.jit(device=True, inline=True)
def _warp_enqueue_global(arr, state, rear_slot, item, counters):
    """Warp-aggregated append on a global rear counter (one atomic per warp).

    All lanes arriving together aggregate (activemask/popc/shfl); the lowest
    lane reserves a ticket slab with one global atomic. The bound check is a
    defensive tripwire only (see the structural argument in the module doc).
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
    if idx < arr.shape[0]:
        arr[idx] = item
    else:
        counters[OVERFLOW] = 1  # unreachable by the structural argument


@cuda.jit(link=[SMID_CU])
def multi_block_global_kernel(img, visited, depth, owner, queue, q_state,
                              counters, block_stats, level_sizes):
    """One shared global queue, every block grid-strides each level window.

    Host contract: launch [blocks, tpb] (cooperative — grid.sync inside);
    visited[seed]=1, queue[0]=seed linear index, q_state=[1], depth=-1,
    counters and block_stats zeroed, owner=-1. Two grid.sync() per level:
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
        if tid == 0 and level < level_sizes.shape[0]:
            level_sizes[level] = level_size

        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map
            my_processed += 1

            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)

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
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    cuda.atomic.add(block_stats, (bx, BS_PROCESSED), my_processed)
    if cuda.threadIdx.x == 0:
        block_stats[bx, BS_SMID] = get_smid()
    if tid == 0:
        # all grid-uniform register values
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum


# --------------------------------------------------------------- bare twin
# Identical BFS with all per-level/per-thread instrumentation stripped (only
# exit-time FILLED/LEVELS remain — two stores, zero steady-state cost) to
# measure the instrumented kernel's observer overhead. It shares every
# device helper above so the algorithm cannot drift from the instrumented
# version; only counter/trace lines differ.


@cuda.jit
def multi_block_global_bare_kernel(img, visited, depth, queue, q_state,
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
                                             nx * height + ny, counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# ------------------------------------------------ 8-connectivity twins
# Verbatim copies of the two kernels above with exactly three tokens
# changed each (name, DX8/DY8 const arrays, range(8)) — see the offsets
# comment at the top for why they are twins and not a runtime n_dirs
# parameter. Depth becomes Chebyshev distance (square waves) instead of
# Manhattan (diamond waves): fewer, wider levels for the same blob.


@cuda.jit(link=[SMID_CU])
def multi_block_global8_kernel(img, visited, depth, owner, queue, q_state,
                               counters, block_stats, level_sizes):
    """8-connectivity variant of multi_block_global_kernel; same host
    contract, same instrumentation, diagonal neighbors included."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)

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
        if tid == 0 and level < level_sizes.shape[0]:
            level_sizes[level] = level_size

        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map
            my_processed += 1

            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)

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
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    cuda.atomic.add(block_stats, (bx, BS_PROCESSED), my_processed)
    if cuda.threadIdx.x == 0:
        block_stats[bx, BS_SMID] = get_smid()
    if tid == 0:
        # all grid-uniform register values
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum


@cuda.jit
def multi_block_global8_bare_kernel(img, visited, depth, queue, q_state,
                                    counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)

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
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# ------------------------------------------------ radius-2 twins (guarded)
# 8-connectivity plus a GUARDED second ring: the 8 ring-1 neighbors are
# probed first, unconditionally, with the unchanged claim protocol —
# coverage never depends on jumps. Only if all 8 are in-bounds blob
# material (red, or visited==1 for already-claimed pixels whose paint may
# be racing) does the thread also probe the 16 ring-2 cells. The guard is
# equivalent to "all 8 ring-1 in-bounds AND red in the ORIGINAL image"
# (unclaimed pixels are never painted, so not-red + visited==0 means
# never-red; claimed pixels were red by CAS precondition), which makes the
# fill set provably identical to the conn8 twins' — only depth/levels
# change meaning (BFS on a supergraph: each level advances Chebyshev
# distance 2 through interior, so levels roughly halve on solid blobs).
# The bet: half the grid.sync barriers, paid for with up to 3x the probe
# traffic. INTERIOR counts guard passes — the exact ring-2 probe count is
# 16 * interior.


@cuda.jit(link=[SMID_CU])
def multi_block_global8r2_kernel(img, visited, depth, owner, queue, q_state,
                                 counters, block_stats, level_sizes):
    """Guarded radius-2 variant of multi_block_global8_kernel; same host
    contract, same instrumentation, plus counters[INTERIOR]."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    dx2 = cuda.const.array_like(DX_R2_HOST)
    dy2 = cuda.const.array_like(DY_R2_HOST)

    front = 0
    rear = 1
    level = 0
    peak_level = 1
    peak_occ = 1
    active_thread_sum = 0
    active_warp_sum = 0
    my_processed = 0
    my_cas_attempts = 0
    my_interior = 0

    while front < rear:
        level_size = rear - front
        if level_size > peak_level:
            peak_level = level_size
        active = min(level_size, stride)
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if tid == 0 and level < level_sizes.shape[0]:
            level_sizes[level] = level_size

        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map
            my_processed += 1

            interior = True
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height:
                    if _is_red(img, nx, ny):
                        my_cas_attempts += 1
                        if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                            _warp_enqueue_global(queue, q_state, Q_REAR,
                                                 nx * height + ny, counters)
                    elif visited[nx, ny] == 0:
                        interior = False  # never red: not blob material
                else:
                    interior = False      # edge pixels never jump

            if interior:
                my_interior += 1
                for d in range(16):
                    nx = x + dx2[d]
                    ny = y + dy2[d]
                    # bounds still required: ring-1 in-bounds does not
                    # imply ring-2 in-bounds (x==1 -> ring-2 at -1)
                    if (0 <= nx < width and 0 <= ny < height
                            and _is_red(img, nx, ny)):
                        my_cas_attempts += 1
                        if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                            _warp_enqueue_global(queue, q_state, Q_REAR,
                                                 nx * height + ny, counters)

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
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    cuda.atomic.add(counters, INTERIOR, my_interior)
    cuda.atomic.add(block_stats, (bx, BS_PROCESSED), my_processed)
    if cuda.threadIdx.x == 0:
        block_stats[bx, BS_SMID] = get_smid()
    if tid == 0:
        # all grid-uniform register values
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum


@cuda.jit
def multi_block_global8r2_bare_kernel(img, visited, depth, queue, q_state,
                                      counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    dx2 = cuda.const.array_like(DX_R2_HOST)
    dy2 = cuda.const.array_like(DY_R2_HOST)

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
            interior = True
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height:
                    if _is_red(img, nx, ny):
                        if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                            _warp_enqueue_global(queue, q_state, Q_REAR,
                                                 nx * height + ny, counters)
                    elif visited[nx, ny] == 0:
                        interior = False
                else:
                    interior = False
            if interior:
                for d in range(16):
                    nx = x + dx2[d]
                    ny = y + dy2[d]
                    if (0 <= nx < width and 0 <= ny < height
                            and _is_red(img, nx, ny)):
                        if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                            _warp_enqueue_global(queue, q_state, Q_REAR,
                                                 nx * height + ny, counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# ------------------------------------- warp-cooperative probing twins
# Same BFS graph as the conn8 twins — fill set, depth AND levels must come
# out bit-identical; only the WORK DISTRIBUTION changes. Each warp takes 4
# queue entries per chunk and assigns each lane one (entry, direction)
# pair (k = lane>>3, d = lane&7), so all 32 neighbor probes for 4 pixels
# issue in ONE round instead of 8 lockstep loop iterations. On wide
# frontiers this is ~a wash (same loads in flight, reshuffled); on NARROW
# frontiers it converts 8 serial probe round-trips per level into 1 and
# lifts lane occupancy (a 4-pixel level: 4/32 lanes busy x 8 rounds ->
# 32/32 busy x 1 round). The 8 lanes sharing an entry re-read it (hardware
# broadcast — one transaction) and redundantly decode x, y: accepted ALU
# cost, part of the bet being measured.


@cuda.jit(link=[SMID_CU])
def multi_block_global8wc_kernel(img, visited, depth, owner, queue, q_state,
                                 counters, block_stats, level_sizes):
    """Warp-cooperative variant of multi_block_global8_kernel; same host
    contract, same instrumentation. active-thread accounting scales
    level_size by 8 (each entry occupies 8 lanes)."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)

    lane = cuda.laneid
    warp_gid = tid // 32
    n_warps = stride // 32
    k = lane >> 3   # which of this warp's 4 entries
    d = lane & 7    # which of that entry's 8 directions

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
        active = min(level_size * 8, stride)  # 8 lanes per entry
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if tid == 0 and level < level_sizes.shape[0]:
            level_sizes[level] = level_size

        for base in range(front + warp_gid * 4, rear, n_warps * 4):
            idx = base + k
            if idx < rear:  # partial final chunk
                pixel = queue[idx]  # 8 lanes, same address: broadcast read
                x = pixel // height
                y = pixel % height

                if d == 0:  # one lane per entry paints/records
                    img[x, y, 0] = 0
                    img[x, y, 1] = 0
                    img[x, y, 2] = 255
                    depth[x, y] = level
                    owner[x, y] = bx
                    my_processed += 1

                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)

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
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    cuda.atomic.add(block_stats, (bx, BS_PROCESSED), my_processed)
    if cuda.threadIdx.x == 0:
        block_stats[bx, BS_SMID] = get_smid()
    if tid == 0:
        # all grid-uniform register values
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum


@cuda.jit
def multi_block_global8wc_bare_kernel(img, visited, depth, queue, q_state,
                                      counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)

    lane = cuda.laneid
    warp_gid = tid // 32
    n_warps = stride // 32
    k = lane >> 3
    d = lane & 7

    front = 0
    rear = 1
    level = 0

    while front < rear:
        for base in range(front + warp_gid * 4, rear, n_warps * 4):
            idx = base + k
            if idx < rear:
                pixel = queue[idx]
                x = pixel // height
                y = pixel % height
                if d == 0:
                    img[x, y, 0] = 0
                    img[x, y, 1] = 0
                    img[x, y, 2] = 255
                    depth[x, y] = level
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
