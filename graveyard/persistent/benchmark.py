"""
Benchmark: persistent cooperative kernel vs kernel-per-level (multi-blocks)
vs CPU sequential BFS.

Scenes:
- 2000x2000 image, 1000x1000 blob  (the old benchmark scene, 1M red pixels)
- 8000x8000 image, 4000x4000 blob  (the old large scene, 16M red pixels)
- 256x256 serpentine               (thin-blob killer case, ~33k BFS levels)

All three implementations are cross-checked to fill the same pixel count.
Old-implementation timings include its debug instrumentation and per-level
host round-trips — that is its honest current cost.

Run: uv run python src/gpu/single_blob/persistent/benchmark.py
"""

import os
import sys
import io
import gc
import time
from contextlib import redirect_stdout

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Append (not insert) so multi-blocks' empty kernels.py placeholder cannot
# shadow this directory's kernels module.
sys.path.append(os.path.join(_HERE, '..', 'multi-blocks'))

from flood_fill import flood_fill                        # noqa: E402
from reference import cpu_flood_fill                     # noqa: E402
import scenes                                            # noqa: E402
from kernels_fixed import run_multi_iteration_flood_fill  # noqa: E402

# Old implementation's validated optimum (new_optimal_config_validation.json)
OLD_BLOCKS, OLD_THREADS, OLD_CHUNK = 24, 128, 32
OLD_QUEUE_CAPACITY = 16_000_000


def run_new(img, sx, sy, repeats=3):
    best = None
    for _ in range(repeats):
        result = flood_fill(img, sx, sy)
        if best is None or result.kernel_ms < best.kernel_ms:
            best = result
        gc.collect()
    return best.kernel_ms, best.filled, best.levels, best


def run_cpu(img, sx, sy, repeats=3):
    best_ms, filled = None, 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, _, _, filled = cpu_flood_fill(img, sx, sy)
        ms = (time.perf_counter() - t0) * 1000
        best_ms = ms if best_ms is None else min(best_ms, ms)
    return best_ms, filled


def run_old(img, sx, sy, max_iterations=200_000):
    width, height = img.shape[0], img.shape[1]
    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(np.zeros((width, height), dtype=np.int32))
    d_color = cuda.to_device(np.array([0, 0, 255], dtype=np.uint8))
    d_qx = cuda.device_array(OLD_QUEUE_CAPACITY, dtype=np.int32)
    d_qy = cuda.device_array(OLD_QUEUE_CAPACITY, dtype=np.int32)
    d_front = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_rear = cuda.to_device(np.zeros(1, dtype=np.int32))
    dbg_block = cuda.to_device(np.zeros(OLD_BLOCKS, dtype=np.int32))
    dbg_thread = cuda.to_device(np.zeros(OLD_BLOCKS * OLD_THREADS, dtype=np.int32))
    dbg_warp = cuda.to_device(np.zeros(OLD_BLOCKS * 2, dtype=np.int32))
    dbg_pixels = cuda.to_device(np.zeros(1, dtype=np.int32))

    t0 = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        iterations = run_multi_iteration_flood_fill(
            d_img, d_visited, sx, sy, width, height, d_color,
            d_qx, d_qy, d_front, d_rear,
            dbg_block, dbg_thread, dbg_warp, dbg_pixels,
            blocks_per_grid=OLD_BLOCKS, threads_per_block=OLD_THREADS,
            chunk_size=OLD_CHUNK, max_iterations=max_iterations)
    ms = (time.perf_counter() - t0) * 1000
    filled = int(d_visited.copy_to_host().sum())

    del d_img, d_visited, d_qx, d_qy, dbg_thread
    gc.collect()
    return ms, filled, iterations


def bench_scene(name, img, sx, sy, skip_old=False):
    print(f"\n=== {name} ({img.shape[0]}x{img.shape[1]}) ===")

    cpu_ms, cpu_filled = run_cpu(img, sx, sy)
    print(f"  CPU sequential BFS : {cpu_ms:10.2f} ms   filled={cpu_filled:,}")

    new_ms, new_filled, new_levels, _ = run_new(img, sx, sy)
    agree = "OK" if new_filled == cpu_filled else "MISMATCH!"
    print(f"  GPU persistent     : {new_ms:10.2f} ms   filled={new_filled:,} "
          f"levels={new_levels:,}  [{agree}]  speedup vs CPU: {cpu_ms / new_ms:.1f}x")

    if not skip_old:
        old_ms, old_filled, old_iters = run_old(img, sx, sy)
        agree = "OK" if old_filled == cpu_filled else "MISMATCH!"
        print(f"  GPU kernel-per-lvl : {old_ms:10.2f} ms   filled={old_filled:,} "
              f"iters={old_iters:,}  [{agree}]  persistent is {old_ms / new_ms:.1f}x faster")
    gc.collect()


def main():
    # Warm up all JITs on a tiny scene so compile time is excluded
    img, sx, sy = scenes.square_scene(64, 64, 32, 32)
    run_cpu(img, sx, sy, repeats=1)
    run_new(img, sx, sy, repeats=1)
    run_old(img, sx, sy)

    img, sx, sy = scenes.square_scene(2000, 2000, 1000, 1000)
    bench_scene("compact blob, 1M red px", img, sx, sy)

    img, sx, sy = scenes.serpentine_scene(256, 256)
    bench_scene("serpentine (thin killer case)", img, sx, sy)

    img, sx, sy = scenes.square_scene(8000, 8000, 4000, 4000)
    bench_scene("large compact blob, 16M red px", img, sx, sy)


if __name__ == '__main__':
    main()
