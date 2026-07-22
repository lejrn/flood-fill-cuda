"""
Tests for kernels_fixed.py — gradient-colored multi-block flood fill.

Skips gracefully when no CUDA device is available.
"""

import os
import numpy as np
import pytest

import subprocess
import sys

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

def _cuda_functional():
    """Return True only if a CUDA device array can actually be created."""
    try:
        from numba import cuda
        if not cuda.is_available():
            return False
    except Exception:
        return False
    # is_available() can return True even when the driver segfaults on first use
    # (common in WSL2 without a physical GPU). Probe with a subprocess so that a
    # crash there does not kill the test process.
    env = {**os.environ, 'NUMBA_CUDA_USE_NVIDIA_BINDING': '1'}
    probe = subprocess.run(
        [sys.executable, "-c",
         "from numba import cuda; import numpy as np; cuda.to_device(np.zeros(1, dtype=np.int32))"],
        timeout=15,
        capture_output=True,
        env=env,
    )
    return probe.returncode == 0

CUDA_AVAILABLE = _cuda_functional()

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device available")


def _run_fill(width=256, height=256, blob_size=128):
    """Helper: create a small scene, run gradient flood fill, return host arrays."""
    from kernels_fixed import run_multi_iteration_flood_fill, reset_global_queue
    from debug_logging import create_global_queue_arrays

    img_host = np.full((width, height, 3), 255, dtype=np.uint8)
    bx0 = width // 2 - blob_size // 2
    bx1 = bx0 + blob_size
    by0 = height // 2 - blob_size // 2
    by1 = by0 + blob_size
    img_host[bx0:bx1, by0:by1] = [255, 0, 0]

    start_x, start_y = width // 2, height // 2

    img = cuda.to_device(img_host)
    visited = cuda.device_array((width, height), dtype=np.int32)
    visited[:] = 0

    gqx, gqy, gqf, gqr = create_global_queue_arrays()
    reset_global_queue(gqf, gqr)

    blocks = 8
    threads = 64
    debug_blocks = cuda.device_array(blocks, dtype=np.int32)
    debug_threads = cuda.device_array(blocks * threads, dtype=np.int32)
    debug_warps = cuda.device_array(blocks * 2, dtype=np.int32)
    debug_pixels = cuda.device_array(1, dtype=np.int32)
    for arr in (debug_blocks, debug_threads, debug_warps, debug_pixels):
        arr[:] = 0

    new_color = np.array([0, 0, 255], dtype=np.uint8)
    new_color_gpu = cuda.to_device(new_color)

    run_multi_iteration_flood_fill(
        img, visited, start_x, start_y, width, height, new_color_gpu,
        gqx, gqy, gqf, gqr,
        debug_blocks, debug_threads, debug_warps, debug_pixels,
        blocks_per_grid=blocks, threads_per_block=threads,
    )

    return img.copy_to_host(), visited.copy_to_host()


def test_no_red_pixels_remain():
    """After fill, no originally-red pixels should still be red."""
    img, _ = _run_fill()
    red_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    assert not red_mask.any(), f"{red_mask.sum()} red pixels still present after fill"


def test_gradient_produces_varied_colors():
    """Filled pixels must not be a single uniform color — gradient must vary."""
    img, visited = _run_fill()
    filled = visited == 1

    r_vals = np.unique(img[:, :, 0][filled])
    g_vals = np.unique(img[:, :, 1][filled])
    b_vals = np.unique(img[:, :, 2][filled])

    assert len(r_vals) > 1, "Red channel is constant — spatial gradient not applied"
    assert len(g_vals) > 1, "Green channel is constant — spatial gradient not applied"
    assert len(b_vals) > 1, "Blue channel is constant — spatial gradient not applied"


def test_visited_covers_blob():
    """All blob pixels must be marked visited."""
    width, height, blob_size = 256, 256, 128
    _, visited = _run_fill(width, height, blob_size)

    bx0 = width // 2 - blob_size // 2
    bx1 = bx0 + blob_size
    by0 = height // 2 - blob_size // 2
    by1 = by0 + blob_size

    blob_visited = visited[bx0:bx1, by0:by1]
    unvisited = (blob_visited == 0).sum()
    assert unvisited == 0, f"{unvisited} blob pixels were not visited"


def test_non_blob_pixels_unchanged():
    """White background pixels outside the blob must remain white."""
    width, height, blob_size = 256, 256, 128
    img, visited = _run_fill(width, height, blob_size)
    unfilled = visited == 0

    non_white = ((img[:, :, 0][unfilled] != 255) |
                 (img[:, :, 1][unfilled] != 255) |
                 (img[:, :, 2][unfilled] != 255))
    assert not non_white.any(), "Background pixels were modified"
