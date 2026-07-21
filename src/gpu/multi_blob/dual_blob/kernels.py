"""Dual-blob BFS flood fill: labeled queue entries, per-blob colors.

The image now holds TWO disconnected red blobs, and each must come back a
different color (blob 0 -> blue, blob 1 -> green), painted by the kernel
itself. Every kernel here is a copy of the multi_block stage's proven
global-queue loop; the whole stage differs from that baseline in exactly
two ways:

1. init: `rear = 1` -> `rear = n_seeds` (a new trailing scalar argument) —
   the old kernels baked in "the queue starts with exactly one seed"; now
   the host passes the pre-loaded seed count (2 for the multisource mode,
   1 for the per-blob modes). The count must arrive as a launch-uniform
   PARAMETER, not be read from q_state at kernel start: blocks do not
   start in lockstep, so an early block can process its seed and enqueue
   (mutating q_state's rear) before a late block performs its initial
   read — threads then disagree on the level-0 window, diverge in loop
   trip count, and deadlock at grid.sync. Measured, not hypothetical:
   the first draft of this stage read q_state here and hung the GPU on
   its first non-trivial launch. peak_level/peak_occ start values and
   the level_sizes[0] trace all derive from rear-front in the first loop
   iteration, so no other init changes.
2. queue entries carry a blob label, and the paint site becomes a const
   2x3 palette lookup (blue/green) instead of a hardcoded blue triple.
   Neighbors inherit the dequeuer's label at enqueue, so each wave carries
   its blob id outward like a surname. Labels cannot mix: a label only
   travels by inheritance, and the scenes keep >= 2 px of white between
   blobs, which no wave can cross even at 8-connectivity.

The label rides INSIDE the int32 entry — no label array, no extra bytes on
the hottest data structure. Two entry formats coexist as twin families,
because "how should a labeled entry be encoded?" is itself a measured bet:

  lin family:  entry = (x*height + y) << 1 | label
      One spare bit. Decode pays the baseline's integer div/mod
      (`// height`, `% height` — no hardware int divide on GPU, ~20-40
      cycles each). Minimal one-bit diff vs the published multi_block
      kernel, so labeling cost benchmarks cleanly against it.
      Host constraint: width*height < 2**30.

  xy family:   entry = label << 26 | x << 13 | y
      Coordinates stored as 13-bit fields: decode is shifts+masks
      (~3 cycles), the div/mod disappears entirely, and 6 label bits
      allow up to 64 blobs. Costs comparability with the published
      kernel (two changes at once) and caps dimensions at 8192.
      Host constraint: width <= 8192 and height <= 8192.

  The bet: on big solid scenes the divide's latency hides behind warp
  parallelism (wash); where the GPU is ALU/latency-bound the xy family
  should pull ahead. benchmark.py measures the decode tax head-to-head.

Everything load-bearing is untouched from multi_block: the claim protocol
(bounds -> is_red -> atomic.cas(visited) -> warp-aggregated enqueue), two
grid.sync() per level, every counter's meaning (FILLED still == pixels ==
queue entries), and the structural no-overflow argument (each pixel is
CAS-claimed at most once before enqueue, so total appends <= filled <=
width*height regardless of how many seeds the host pre-loads).
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
# Twins, not a runtime n_dirs loop — same reasoning as multi_block: a dynamic
# loop bound would change the 4-conn codegen and decouple it from the numbers
# already published for the single-blob family.
DX8_HOST = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
DY8_HOST = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# Ring-2 offsets: the 16 cells at Chebyshev distance exactly 2, clockwise
# from E — probed by the radius-2 twins only when all 8 ring-1 neighbors
# are in-bounds blob material (see multi_block's radius-2 twins: the guard
# keeps every jump inside true 8-connectivity, so labels stay correct too —
# a ring-2 claim is 8-connected to the dequeuer's own component).
DX_R2_HOST = np.array([2, 2, 2, 1, 0, -1, -2, -2,
                       -2, -2, -2, -1, 0, 1, 2, 2], dtype=np.int32)
DY_R2_HOST = np.array([0, 1, 2, 2, 2, 2, 2, 1,
                       0, -1, -2, -2, -2, -2, -2, -1], dtype=np.int32)

# Per-blob fill colors, indexed by label: 0 -> blue, 1 -> green.
# Neither may be RED (255,0,0) — painted pixels must stop matching _is_red.
PALETTE_HOST = np.array([[0, 0, 255],
                         [0, 255, 0]], dtype=np.uint8)

# xy-family field layout: entry = label << 26 | x << 13 | y
XY_FIELD_BITS = 13
XY_FIELD_MASK = (1 << XY_FIELD_BITS) - 1   # 0x1FFF
XY_LBL_SHIFT = 2 * XY_FIELD_BITS           # 26
XY_MAX_DIM = 1 << XY_FIELD_BITS            # 8192

# Slots in the device-side int64 counters array (identical to multi_block)
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


# ===================================================== lin entry family
# entry = (x*height + y) << 1 | label — the baseline-faithful format.


@cuda.jit(link=[SMID_CU])
def dual_blob_lin_kernel(img, visited, depth, owner, queue, q_state,
                         counters, block_stats, level_sizes, n_seeds):
    """One shared global queue of label-packed entries; every block
    grid-strides each level window.

    Host contract: launch [blocks, tpb] (cooperative — grid.sync inside);
    visited[seed]=1 for every seed, queue[0:n] = the n packed seed entries,
    q_state=[n], n_seeds=n, depth=-1, counters and block_stats zeroed,
    owner=-1. Two
    grid.sync() per level: #1 makes the level's enqueues and rear visible
    grid-wide, #2 guarantees everyone has read the new rear before any
    thread's next-level atomics.
    """
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
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
            entry = queue[i]
            lbl = entry & 1
            pixel = entry >> 1
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
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
                                             ((nx * height + ny) << 1) | lbl,
                                             counters)

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


# Bare twin: all per-level/per-thread instrumentation stripped (only
# exit-time FILLED/LEVELS remain) to measure observer overhead. The palette
# paint stays: per-blob color is the algorithm here, not instrumentation.
@cuda.jit
def dual_blob_lin_bare_kernel(img, visited, depth, queue, q_state,
                              counters, n_seeds):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            entry = queue[i]
            lbl = entry & 1
            pixel = entry >> 1
            x = pixel // height
            y = pixel % height
            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
            depth[x, y] = level
            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             ((nx * height + ny) << 1) | lbl,
                                             counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


@cuda.jit(link=[SMID_CU])
def dual_blob_lin8_kernel(img, visited, depth, owner, queue, q_state,
                          counters, block_stats, level_sizes, n_seeds):
    """8-connectivity twin of dual_blob_lin_kernel; same host contract,
    same instrumentation, diagonal neighbors included."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
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
            entry = queue[i]
            lbl = entry & 1
            pixel = entry >> 1
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
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
                                             ((nx * height + ny) << 1) | lbl,
                                             counters)

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
def dual_blob_lin8_bare_kernel(img, visited, depth, queue, q_state,
                               counters, n_seeds):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            entry = queue[i]
            lbl = entry & 1
            pixel = entry >> 1
            x = pixel // height
            y = pixel % height
            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
            depth[x, y] = level
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             ((nx * height + ny) << 1) | lbl,
                                             counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# ------------------------------------- radius-2 twins (guarded, lin only)
# The multi_block radius-2 experiment ported to labeled entries: ring-1
# probed first, unconditionally; only pixels whose entire ring-1 is
# in-bounds blob material (red, or visited==1 for claimed pixels whose
# paint may be racing) also probe the 16 ring-2 cells. The guard keeps
# every jump inside the dequeuer's own 8-connected component, so the
# inherited label is exactly as correct as at ring-1 — fill set AND label
# map provably identical to the lin8 twins'; only depth/levels change
# meaning (levels roughly halve on solid blobs). lin family only: the xy
# bet measured a wash, so the experiment stays on the baseline encoding.


@cuda.jit(link=[SMID_CU])
def dual_blob_lin8r2_kernel(img, visited, depth, owner, queue, q_state,
                            counters, block_stats, level_sizes, n_seeds):
    """Guarded radius-2 variant of dual_blob_lin8_kernel; same host
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
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
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
            entry = queue[i]
            lbl = entry & 1
            pixel = entry >> 1
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
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
                            _warp_enqueue_global(
                                queue, q_state, Q_REAR,
                                ((nx * height + ny) << 1) | lbl,
                                counters)
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
                            _warp_enqueue_global(
                                queue, q_state, Q_REAR,
                                ((nx * height + ny) << 1) | lbl,
                                counters)

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
def dual_blob_lin8r2_bare_kernel(img, visited, depth, queue, q_state,
                                 counters, n_seeds):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    dx2 = cuda.const.array_like(DX_R2_HOST)
    dy2 = cuda.const.array_like(DY_R2_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            entry = queue[i]
            lbl = entry & 1
            pixel = entry >> 1
            x = pixel // height
            y = pixel % height
            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
            depth[x, y] = level
            interior = True
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height:
                    if _is_red(img, nx, ny):
                        if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                            _warp_enqueue_global(
                                queue, q_state, Q_REAR,
                                ((nx * height + ny) << 1) | lbl,
                                counters)
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
                            _warp_enqueue_global(
                                queue, q_state, Q_REAR,
                                ((nx * height + ny) << 1) | lbl,
                                counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# ====================================================== xy entry family
# entry = label << 26 | x << 13 | y — coordinates as bit-fields, decode is
# shifts+masks and the integer div/mod disappears. Verbatim twins of the
# lin family above; only the pack/unpack lines differ.


@cuda.jit(link=[SMID_CU])
def dual_blob_xy_kernel(img, visited, depth, owner, queue, q_state,
                        counters, block_stats, level_sizes, n_seeds):
    """xy-format twin of dual_blob_lin_kernel; same host contract, entries
    carry (label, x, y) as bit-fields. Host guarantees dims <= 8192."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
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
            entry = queue[i]
            lbl = (entry >> 26) & 0x3F
            x = (entry >> 13) & 0x1FFF
            y = entry & 0x1FFF

            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
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
                                             (lbl << 26) | (nx << 13) | ny,
                                             counters)

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
def dual_blob_xy_bare_kernel(img, visited, depth, queue, q_state,
                             counters, n_seeds):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            entry = queue[i]
            lbl = (entry >> 26) & 0x3F
            x = (entry >> 13) & 0x1FFF
            y = entry & 0x1FFF
            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
            depth[x, y] = level
            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             (lbl << 26) | (nx << 13) | ny,
                                             counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


@cuda.jit(link=[SMID_CU])
def dual_blob_xy8_kernel(img, visited, depth, owner, queue, q_state,
                         counters, block_stats, level_sizes, n_seeds):
    """8-connectivity twin of dual_blob_xy_kernel."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
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
            entry = queue[i]
            lbl = (entry >> 26) & 0x3F
            x = (entry >> 13) & 0x1FFF
            y = entry & 0x1FFF

            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
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
                                             (lbl << 26) | (nx << 13) | ny,
                                             counters)

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
def dual_blob_xy8_bare_kernel(img, visited, depth, queue, q_state,
                              counters, n_seeds):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    front = 0
    rear = n_seeds
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            entry = queue[i]
            lbl = (entry >> 26) & 0x3F
            x = (entry >> 13) & 0x1FFF
            y = entry & 0x1FFF
            img[x, y, 0] = palette[lbl, 0]
            img[x, y, 1] = palette[lbl, 1]
            img[x, y, 2] = palette[lbl, 2]
            depth[x, y] = level
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             (lbl << 26) | (nx << 13) | ny,
                                             counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
