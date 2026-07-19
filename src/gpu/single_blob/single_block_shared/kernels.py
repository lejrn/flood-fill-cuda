"""
Single-block BFS flood fill with the frontier queue in shared memory.

One CUDA block runs the whole level-synchronous BFS. The frontier lives in a
ring buffer in shared memory (~1-30 cycle access); the image, visited mask,
depth map, and counters live in global memory. Levels are separated by two
cuda.syncthreads() calls — the block-local analogue of the persistent
kernel's two grid.sync() calls, and for the same reason: the first makes the
level's enqueues and final rear visible, the second guarantees every thread
has read the new rear before any thread's atomics for the next level begin.

Ring-buffer safety argument
---------------------------
front, rear, and enqueue tickets are monotonically increasing *virtual*
indices (never themselves wrapped; slots are addressed as ticket & RING_MASK).
front is frozen for the duration of a level, so every ticket that passes the
occupancy check lies in the window [front, front + RING_CAPACITY), and any
two such tickets map to distinct slots. Live readable entries [front, rear)
sit inside the same window, so a write can never land on a slot still being
read, and a full ring never overwrites live entries — a failing enqueue
writes nothing and sets the overflow tripwire instead. Total tickets are
bounded by the CAS-claimed pixel count <= width*height < 2^31, so int32
never overflows. After a tripwire the queue contents are incomplete, so the
kernel aborts at the next level boundary (a uniform branch — every thread
reads the same shared flag between the two syncs) and the host raises.

Flaws of the old single_block.py this kernel fixes
--------------------------------------------------
- Non-wrapping queue (rear only grew, capacity 6000) -> ring buffer whose
  requirement is O(peak frontier), not O(blob area).
- Silent pixel drop when full -> overflow tripwire + host RuntimeError.
- atomic CAS on visited *before* the is-red check (permanently marked
  non-red pixels visited) -> bounds, then red check, then CAS.
- Thread-id debug pattern written into the blue channel -> solid blue.
- Hardcoded 64 threads -> threads_per_block is a driver parameter.

Capacity: RING_CAPACITY = 8192 int32 linear indices = 32 KB, the largest
power of two under the 48 KB static shared-memory limit (16384 would need
64 KB); power-of-two enables ticket & RING_MASK instead of modulo. Peak ring
occupancy spans two adjacent BFS levels: ~4W for a center-seeded solid WxW
square (fits W <= ~2048), ~2W corner-seeded (fits W <= ~4096), O(width) for
serpentines. v2 (not implemented): spill-to-global overflow removes the cap.
"""

import os

# Must be set before numba is imported — CUDA 12.9 + Numba ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda, int32

RING_CAPACITY = 8192
RING_MASK = RING_CAPACITY - 1

# 4-connectivity neighbor offsets (matches reference.py: right, down, left, up)
DX_HOST = np.array([1, 0, -1, 0], dtype=np.int32)
DY_HOST = np.array([0, 1, 0, -1], dtype=np.int32)

# Slots in the device-side int64 counters array
FILLED = 0             # total pixels enqueued (= filled when no overflow)
LEVELS = 1             # BFS levels processed
OVERFLOW = 2           # tripwire: an enqueue found the ring full
PEAK_LEVEL = 3         # largest single frontier (rear - front at level start)
PEAK_OCC = 4           # max ring occupancy (new_rear - front at level end)
ACTIVE_THREAD_SUM = 5  # sum over levels of min(level_size, threads)
ACTIVE_WARP_SUM = 6    # sum over levels of ceil(min(level_size, threads) / 32)
PROCESSED = 7          # pixels dequeued/recolored (== FILLED iff exactly-once)
CAS_ATTEMPTS = 8       # visited-CAS ops tried (attempts - wins = redundant work)
NUM_COUNTERS = 9


@cuda.jit(device=True, inline=True)
def _is_red(img, x, y):
    """Check if pixel is red (255, 0, 0)."""
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


@cuda.jit
def single_block_bfs_kernel(img, visited, depth, seed_x, seed_y, counters,
                            level_sizes):
    """Whole 4-connected BFS runs inside this one single-block launch.

    Host contract: launch with exactly 1 block; visited all zeros, depth all
    -1, counters all zeros. The kernel seeds itself (thread 0). level_sizes
    records each level's frontier size while level < level_sizes.shape[0]
    (aggregate counters stay exact past that; host detects truncation as
    levels > level_sizes.shape[0]).
    """
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    width = img.shape[0]
    height = img.shape[1]

    ring = cuda.shared.array(RING_CAPACITY, int32)
    s_rear = cuda.shared.array(1, int32)
    s_overflow = cuda.shared.array(1, int32)

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    if tid == 0:
        ring[0] = seed_x * height + seed_y
        s_rear[0] = 1
        s_overflow[0] = 0
        visited[seed_x, seed_y] = 1
    cuda.syncthreads()

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
        # Uniform per-level accounting, kept in registers (every thread
        # computes the same values; only thread 0's copy is written at exit).
        level_size = rear - front
        if level_size > peak_level:
            peak_level = level_size
        active = min(level_size, nthreads)
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if tid == 0 and level < level_sizes.shape[0]:
            level_sizes[level] = level_size

        # Block-stride partition of this level's segment [front, rear):
        # thread tid is active iff tid < level_size.
        for i in range(front + tid, rear, nthreads):
            pixel = ring[i & RING_MASK]
            x = pixel // height
            y = pixel % height

            img[x, y, 0] = 0
            img[x, y, 1] = 0
            img[x, y, 2] = 255
            depth[x, y] = level
            my_processed += 1

            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    # Exactly-once claim: only the CAS winner may enqueue.
                    # (Recoloring can't confuse the red check: only claimed
                    # pixels are written, so an unvisited pixel's color is
                    # stable, and a stale red read just loses the CAS.)
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        ticket = cuda.atomic.add(s_rear, 0, 1)
                        if ticket - front < RING_CAPACITY:
                            ring[ticket & RING_MASK] = nx * height + ny
                        else:
                            s_overflow[0] = 1

        cuda.syncthreads()  # all enqueues + final s_rear for this level visible
        new_rear = s_rear[0]
        overflowed = s_overflow[0]
        cuda.syncthreads()  # everyone has read new_rear; next level's atomics may begin

        level += 1
        if overflowed:
            break  # uniform value -> uniform exit; queue is incomplete, host raises
        occ = new_rear - front
        if occ > peak_occ:
            peak_occ = occ
        front = rear
        rear = new_rear

    cuda.atomic.add(counters, PROCESSED, my_processed)
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    if tid == 0:
        counters[FILLED] = s_rear[0]
        counters[LEVELS] = level
        counters[OVERFLOW] = s_overflow[0]
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum
