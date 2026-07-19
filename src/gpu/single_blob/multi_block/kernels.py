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

# Slots in the device-side int64 counters array (grid-wide scalars only;
# 0-8 match dual_block — the per-block slots 11+ moved to block_stats)
FILLED = 0
LEVELS = 1
OVERFLOW = 2            # defensive tripwire: structurally unreachable
PEAK_LEVEL = 3          # largest single frontier
PEAK_OCC = 4            # max queue entries alive across two adjacent levels
ACTIVE_THREAD_SUM = 5   # sum over levels of min(level_size, grid threads)
ACTIVE_WARP_SUM = 6     # sum over levels of ceil(min(level_size, grid)/32)
PROCESSED = 7           # pixels dequeued/recolored (== FILLED iff exactly-once)
CAS_ATTEMPTS = 8        # visited-CAS ops tried
NUM_COUNTERS = 9

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
