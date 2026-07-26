"""
CPU reference flood fill: sequential level-synchronous BFS, 8-connectivity.

This is the ground truth the GPU implementation is tested against. It mirrors
the GPU kernel's semantics exactly: same neighbor order, same level structure,
same "enqueue once via visited" rule — so visited masks, depth maps, level
counts, and fill counts must all match bit-for-bit.
"""

import numpy as np
from numba import njit

# Same 8-connectivity offsets as the GPU kernel
_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)


@njit(cache=True)
def cpu_flood_fill_8(img, seed_x, seed_y):
    """Level-synchronous BFS from (seed_x, seed_y) over red pixels.

    Returns (visited, depth, levels, filled):
      visited: int32 (width, height), 1 where reached
      depth:   int32 (width, height), BFS level per reached pixel, -1 elsewhere
      levels:  number of BFS levels processed
      filled:  number of pixels reached
    """
    width, height = img.shape[0], img.shape[1]
    visited = np.zeros((width, height), dtype=np.int32)
    depth = np.full((width, height), -1, dtype=np.int32)
    queue = np.empty(width * height, dtype=np.int64)

    visited[seed_x, seed_y] = 1
    queue[0] = seed_x * height + seed_y
    front, rear = 0, 1
    level = 0

    while front < rear:
        level_end = rear
        for i in range(front, level_end):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            depth[x, y] = level
            for d in range(8):
                nx = x + _DX[d]
                ny = y + _DY[d]
                if 0 <= nx < width and 0 <= ny < height and visited[nx, ny] == 0:
                    if img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0 and img[nx, ny, 2] == 0:
                        visited[nx, ny] = 1
                        queue[rear] = nx * height + ny
                        rear += 1
        front = level_end
        level += 1

    return visited, depth, level, rear
