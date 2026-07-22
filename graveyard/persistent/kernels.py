"""
Persistent cooperative-kernel BFS flood fill.

One kernel launch for the whole fill. Blocks synchronize between BFS levels
with cooperative-groups grid.sync() instead of kernel-launch boundaries, so
there are zero host round-trips during the fill.

Design (see PARALLEL_BFS_DESIGN_ANALYSIS.md at the repo root):
- Single history queue of linear pixel indices. Each BFS level occupies a
  consecutive segment [front, rear); enqueues append after rear. Because a
  pixel can only be enqueued by the thread that wins the atomic CAS on its
  visited cell, every pixel enters the queue at most once, so a capacity of
  width*height can never overflow.
- Two grid.sync() per level: the first makes all of this level's enqueues
  visible, the second guarantees every thread has read the new rear before
  any thread starts appending for the following level. Without the second
  sync, threads would disagree on the level boundary and the grid-stride
  partition would skip queue entries.
- Enqueue uses one atomicAdd per warp (ballot/shuffle aggregation) instead
  of one per discovered pixel.
"""

import os

# Must be set before numba is imported — CUDA 12.9 + Numba ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda

# 8-connectivity neighbor offsets (matches the multi-blocks implementation)
DX_HOST = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
DY_HOST = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# Slots in the device-side counters array
REAR = 0       # queue rear: total pixels ever enqueued
LEVELS = 1     # number of BFS levels processed (written at exit)
OVERFLOW = 2   # tripwire: set if an enqueue ever lands past capacity


@cuda.jit(device=True, inline=True)
def _is_red(img, x, y):
    """Check if pixel is red (255, 0, 0)."""
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


@cuda.jit(device=True, inline=True)
def _warp_enqueue(queue, counters, item, capacity):
    """Append item to the queue with one atomicAdd per warp.

    All lanes that reach this call aggregate: the lowest active lane reserves
    a slab of slots for the whole group with a single atomicAdd, broadcasts
    the base offset, and each lane writes at base + its rank. Safe under
    divergence: whatever subset of a warp arrives here together forms the
    active mask, and any straggler subsets simply aggregate separately.
    """
    mask = cuda.activemask()
    count = cuda.popc(mask)
    rank = cuda.popc(mask & cuda.lanemask_lt())
    leader = cuda.ffs(mask) - 1  # ffs is 1-based; mask always has a bit set
    base = 0
    if cuda.laneid == leader:
        base = cuda.atomic.add(counters, REAR, count)
    base = cuda.shfl_sync(mask, base, leader)
    idx = base + rank
    if idx < capacity:
        queue[idx] = item
    else:
        counters[OVERFLOW] = 1  # unreachable when capacity == width*height


@cuda.jit
def persistent_flood_fill_kernel(img, visited, depth, queue, counters):
    """Whole BFS runs inside this one cooperative launch.

    Host contract before launch:
      visited[seed] == 1, queue[0] == seed linear index,
      counters == [1, 0, 0], depth filled with -1.
    Grid must not exceed max_cooperative_grid_blocks or grid.sync deadlocks.
    """
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    dx = cuda.const.array_like(DX_HOST)
    dy = cuda.const.array_like(DY_HOST)

    width = img.shape[0]
    height = img.shape[1]
    capacity = queue.shape[0]

    front = 0
    rear = 1  # the seed, pre-enqueued by the host
    level = 0

    while front < rear:
        # Process this level's segment of the queue. front/rear are uniform
        # across the grid, so the grid-stride partition covers [front, rear)
        # exactly once.
        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            # Spatial gradient recolor + BFS depth for wavefront visualization
            img[x, y, 0] = (x * 255) // width
            img[x, y, 1] = (y * 255) // height
            img[x, y, 2] = 255 - ((x * 255) // width)
            depth[x, y] = level

            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    # Exactly-once claim: only the CAS winner may enqueue.
                    # (Recoloring can't confuse this: only claimed pixels are
                    # ever written, so an unvisited pixel's color is stable.)
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue(queue, counters, nx * height + ny, capacity)
        grid.sync()  # all enqueues for this level are now visible

        new_rear = counters[REAR]
        grid.sync()  # everyone has read new_rear; appending may resume

        front = rear
        rear = new_rear
        if rear > capacity:
            rear = capacity  # defensive; unreachable when capacity == width*height
        level += 1

    if tid == 0:
        counters[LEVELS] = level
