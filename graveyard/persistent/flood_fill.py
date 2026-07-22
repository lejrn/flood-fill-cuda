"""
Host driver for the persistent cooperative-kernel flood fill.

Public API: flood_fill(img, seed_x, seed_y) -> FloodFillResult

Handles validation, device allocation, cooperative grid sizing (queried from
the compiled kernel — never hardcoded, oversizing deadlocks grid.sync), the
single kernel launch, and result collection.
"""

import time
from dataclasses import dataclass

import numpy as np

from kernels import persistent_flood_fill_kernel, REAR, LEVELS, OVERFLOW
from numba import cuda


@dataclass
class FloodFillResult:
    img: np.ndarray        # (width, height, 3) uint8, blob recolored with gradient
    visited: np.ndarray    # (width, height) int32, 1 where reached
    depth: np.ndarray      # (width, height) int32, BFS level per pixel, -1 elsewhere
    levels: int            # BFS levels processed (geodesic eccentricity of seed + 1)
    filled: int            # pixels reached
    blocks: int            # cooperative grid size used
    threads_per_block: int
    kernel_ms: float       # kernel execution time (excludes transfers)
    total_ms: float        # including host<->device transfers


_warmed_up = set()  # threads_per_block values already compiled/queried
_max_blocks_cache = {}


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _max_cooperative_blocks(threads_per_block):
    """Compile the kernel (via a tiny 1-block run) and query how many blocks
    can be co-resident for a cooperative launch at this block size."""
    if threads_per_block in _max_blocks_cache:
        return _max_blocks_cache[threads_per_block]

    tiny_img = np.full((4, 4, 3), 255, dtype=np.uint8)
    tiny_img[1, 1] = (255, 0, 0)
    tiny_visited = np.zeros((4, 4), dtype=np.int32)
    tiny_visited[1, 1] = 1
    d_img = cuda.to_device(tiny_img)
    d_visited = cuda.to_device(tiny_visited)
    d_depth = cuda.to_device(np.full((4, 4), -1, dtype=np.int32))
    d_queue = cuda.to_device(np.array([1 * 4 + 1] + [0] * 15, dtype=np.int32))
    d_counters = cuda.to_device(np.array([1, 0, 0], dtype=np.int32))
    # A 1-block cooperative launch is always legal, so it is safe pre-query.
    persistent_flood_fill_kernel[1, threads_per_block](
        d_img, d_visited, d_depth, d_queue, d_counters)
    cuda.synchronize()

    kernel = next(iter(persistent_flood_fill_kernel.overloads.values()))
    max_blocks = kernel.max_cooperative_grid_blocks(threads_per_block)
    _max_blocks_cache[threads_per_block] = max_blocks
    return max_blocks


def flood_fill(img_host, seed_x, seed_y, threads_per_block=256, blocks=None):
    """Flood-fill the red blob containing (seed_x, seed_y) on the GPU.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for an out-of-bounds or non-red seed.
    """
    if img_host.ndim != 3 or img_host.shape[2] != 3 or img_host.dtype != np.uint8:
        raise ValueError("img must be a (width, height, 3) uint8 array")
    width, height = img_host.shape[0], img_host.shape[1]
    if width * height >= 2 ** 31:
        raise ValueError("image too large for int32 linear pixel indices")
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError(f"seed ({seed_x}, {seed_y}) outside {width}x{height} image")
    if not _is_red(img_host, seed_x, seed_y):
        raise ValueError(f"seed pixel ({seed_x}, {seed_y}) is not red — nothing to fill")

    max_blocks = _max_cooperative_blocks(threads_per_block)
    if blocks is None:
        blocks = max_blocks
    elif blocks > max_blocks:
        raise ValueError(
            f"{blocks} blocks exceeds cooperative-launch capacity "
            f"({max_blocks} at {threads_per_block} threads/block)")

    t_total0 = time.perf_counter()

    visited_host = np.zeros((width, height), dtype=np.int32)
    visited_host[seed_x, seed_y] = 1

    d_img = cuda.to_device(img_host)
    d_visited = cuda.to_device(visited_host)
    d_depth = cuda.to_device(np.full((width, height), -1, dtype=np.int32))
    # Capacity = width*height: every pixel is enqueued at most once (visited
    # CAS), so overflow is impossible by construction.
    d_queue = cuda.device_array(width * height, dtype=np.int32)
    cuda.to_device(np.array([seed_x * height + seed_y], dtype=np.int32),
                   to=d_queue[:1])
    d_counters = cuda.to_device(np.array([1, 0, 0], dtype=np.int32))

    cuda.synchronize()
    t_kernel0 = time.perf_counter()
    persistent_flood_fill_kernel[blocks, threads_per_block](
        d_img, d_visited, d_depth, d_queue, d_counters)
    cuda.synchronize()
    kernel_ms = (time.perf_counter() - t_kernel0) * 1000

    counters = d_counters.copy_to_host()
    if counters[OVERFLOW]:
        raise RuntimeError(
            "queue overflow tripwire fired — this indicates a kernel bug, "
            "capacity width*height cannot legitimately overflow")

    img_out = d_img.copy_to_host()
    visited_out = d_visited.copy_to_host()
    depth_out = d_depth.copy_to_host()
    total_ms = (time.perf_counter() - t_total0) * 1000

    return FloodFillResult(
        img=img_out,
        visited=visited_out,
        depth=depth_out,
        levels=int(counters[LEVELS]),
        filled=int(counters[REAR]),
        blocks=blocks,
        threads_per_block=threads_per_block,
        kernel_ms=kernel_ms,
        total_ms=total_ms,
    )
