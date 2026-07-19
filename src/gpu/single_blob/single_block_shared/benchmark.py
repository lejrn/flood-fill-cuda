"""
Benchmark: single-block shared-ring GPU flood fill vs two CPU baselines.

Implementations compared (all 4-connectivity, exact-red match, same work):
  1. pure-Python deque BFS (self-contained below — the classic sequential
     algorithm; src/cpu/sequential.py is not imported because it runs a
     file-loading job at import time and uses a threshold red test)
  2. @njit compiled level-synchronous BFS (reference.py — the honest CPU bar)
  3. GPU single-block shared-memory ring kernel

The GPU uses ONE block = 1 of the RTX 4060's 24 SMs, by design. The point of
this stage is the technique and truthful numbers, not beating @njit; expect
the GPU to lose on serpentine (frontier starves the block) and to be judged
per-scene elsewhere. Speedups below 1.0x are printed as-is.

Run:  uv run python src/gpu/single_blob/single_block_shared/benchmark.py
      add --slow to include the pure-Python baseline on the 16M-pixel scene

Writes JSON (with per-level frontier traces) and CSV to benchmark_results/
next to this script.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import json
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from flood_fill import flood_fill
from reference import cpu_flood_fill
import scenes
from numba import cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "benchmark_results")

GPU_REPEATS = 5
NJIT_REPEATS = 5
TPB_SWEEP = [64, 128, 256, 512, 1024]

# Pure-Python run count by fill size: it is 100-1000x slower than the others,
# so large blobs get one run and the 16M-pixel scene is opt-in via --slow.
PURE_MEDIAN_LIMIT = 300_000   # <= this many filled pixels: median of 3
PURE_SINGLE_LIMIT = 2_000_000  # <= this: single run; above: only with --slow


def pure_python_bfs(img, seed_x, seed_y):
    """Classic sequential BFS flood fill; recolors a copy, returns filled."""
    out = img.copy()
    width, height = out.shape[0], out.shape[1]
    visited = np.zeros((width, height), dtype=np.uint8)
    visited[seed_x, seed_y] = 1
    queue = deque([(seed_x, seed_y)])
    filled = 0
    while queue:
        x, y = queue.popleft()
        out[x, y, 0] = 0
        out[x, y, 1] = 0
        out[x, y, 2] = 255
        filled += 1
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[nx, ny]:
                p = out[nx, ny]
                if p[0] == 255 and p[1] == 0 and p[2] == 0:
                    visited[nx, ny] = 1
                    queue.append((nx, ny))
    return filled


SCENES = [
    # (name, builder, note)
    ("sq_256_center", lambda: scenes.square_scene(256, 256, 128, 128), "128^2 blob"),
    ("sq_512_center", lambda: scenes.square_scene(512, 512, 256, 256), "256^2 blob"),
    ("sq_1024_center", lambda: scenes.square_scene(1024, 1024, 512, 512), "512^2 blob"),
    ("sq_2000_center", lambda: scenes.square_scene(2000, 2000, 1000, 1000), "1000^2 blob"),
    ("sq_4000_corner", lambda: scenes.full_red_scene(4000, 4000),
     "16M px, corner seed — peak occupancy ~7999 proves the 8192 capacity math"),
    ("serpentine_256", lambda: scenes.serpentine_scene(256, 256),
     "worst case: ~1-wide frontier starves the block"),
    ("disk_1024", lambda: scenes.disk_scene(1024, 1024, 480), "radius-480 disk"),
]


def _median_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def bench_scene(name, builder, note, slow):
    img, sx, sy = builder()
    width, height = img.shape[0], img.shape[1]

    # GPU: median of GPU_REPEATS; keep the last result for metrics/counters.
    gpu_kernel_times, gpu_total_times = [], []
    result = None
    for _ in range(GPU_REPEATS):
        result = flood_fill(img, sx, sy)
        gpu_kernel_times.append(result.kernel_ms)
        gpu_total_times.append(result.total_ms)
    gpu_kernel_ms = statistics.median(gpu_kernel_times)
    gpu_total_ms = statistics.median(gpu_total_times)

    # @njit reference (read-only on img, no copies needed)
    ref_visited, _, ref_levels, ref_filled = cpu_flood_fill(img, sx, sy)
    njit_ms = _median_ms(lambda: cpu_flood_fill(img, sx, sy), NJIT_REPEATS)

    # pure Python, scaled to blob size
    if ref_filled <= PURE_MEDIAN_LIMIT:
        pure_runs = 3
    elif ref_filled <= PURE_SINGLE_LIMIT or slow:
        pure_runs = 1
    else:
        pure_runs = 0
    pure_ms = _median_ms(lambda: pure_python_bfs(img, sx, sy), pure_runs) \
        if pure_runs else None
    pure_filled = pure_python_bfs(img, sx, sy) if pure_runs else None

    filled_ok = (result.filled == ref_filled
                 and (pure_filled is None or pure_filled == ref_filled))

    row = {
        "scene": name,
        "note": note,
        "width": width, "height": height,
        "filled": result.filled,
        "levels": result.levels,
        "peak_frontier": result.peak_level,
        "peak_occupancy": result.peak_occupancy,
        "ring_capacity": result.ring_capacity,
        "threads_per_block": result.threads_per_block,
        "gpu_kernel_ms": gpu_kernel_ms,
        "gpu_total_ms": gpu_total_ms,
        "gpu_alloc_ms": result.alloc_ms,
        "gpu_h2d_ms": result.h2d_ms,
        "gpu_d2h_ms": result.d2h_ms,
        "gpu_mpx_s_kernel": result.filled / gpu_kernel_ms / 1000,
        "gpu_mpx_s_total": result.filled / gpu_total_ms / 1000,
        "njit_ms": njit_ms,
        "njit_mpx_s": ref_filled / njit_ms / 1000,
        "pure_ms": pure_ms,
        "speedup_kernel_vs_njit": njit_ms / gpu_kernel_ms,
        "speedup_total_vs_njit": njit_ms / gpu_total_ms,
        "speedup_total_vs_pure": (pure_ms / gpu_total_ms) if pure_ms else None,
        "thread_util_pct": result.thread_util_pct,
        "warp_engagement_pct": result.warp_engagement_pct,
        "lane_efficiency_pct": result.lane_efficiency_pct,
        "occupancy_pct": result.occupancy_pct,
        "sm_utilization_pct": result.sm_utilization_pct,
        "discovery_redundancy": result.discovery_redundancy,
        "neighbor_check_efficiency_pct": result.neighbor_check_efficiency_pct,
        "filled_crosscheck": "OK" if filled_ok else "MISMATCH",
        "level_sizes": result.level_sizes.tolist(),
    }

    check = "[OK]" if filled_ok else "[MISMATCH!]"
    pure_s = f"{pure_ms:10.1f}" if pure_ms else "   skipped"
    pure_x = f"{row['speedup_total_vs_pure']:7.1f}x" if pure_ms else "       -"
    print(f"{name:16s} {result.filled:>10,d} {result.levels:>6d} "
          f"{result.peak_occupancy:>6d} {gpu_kernel_ms:9.2f} {gpu_total_ms:9.2f} "
          f"{row['gpu_mpx_s_kernel']:8.2f} {njit_ms:8.2f} "
          f"{row['speedup_kernel_vs_njit']:6.2f}x {pure_s} {pure_x} {check}")
    return row


def tpb_sweep():
    print("\nthreads-per-block sweep on sq_2000_center (median kernel ms of "
          f"{GPU_REPEATS}):")
    img, sx, sy = scenes.square_scene(2000, 2000, 1000, 1000)
    print(f"{'tpb':>6s} {'kernel_ms':>10s} {'Mpx/s':>8s} {'thread_util%':>13s} "
          f"{'occupancy%':>11s}")
    rows = []
    for tpb in TPB_SWEEP:
        times = []
        result = None
        for _ in range(GPU_REPEATS):
            result = flood_fill(img, sx, sy, threads_per_block=tpb)
            times.append(result.kernel_ms)
        ms = statistics.median(times)
        rows.append({
            "tpb": tpb,
            "kernel_ms": ms,
            "mpx_s": result.filled / ms / 1000,
            "thread_util_pct": result.thread_util_pct,
            "occupancy_pct": result.occupancy_pct,
        })
        print(f"{tpb:>6d} {ms:>10.2f} {rows[-1]['mpx_s']:>8.2f} "
              f"{result.thread_util_pct:>13.1f} {result.occupancy_pct:>11.1f}")
    return rows


def main():
    slow = "--slow" in sys.argv
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    print(f"Device: {dev_name.strip()} ({device.MULTIPROCESSOR_COUNT} SMs; "
          f"this kernel uses 1 block = 1 SM = "
          f"{100.0 / device.MULTIPROCESSOR_COUNT:.1f}% of them by design)")

    print("Warming up JITs on a tiny scene...")
    warm_img, wx, wy = scenes.square_scene(64, 64, 32, 32)
    flood_fill(warm_img, wx, wy)
    cpu_flood_fill(warm_img, wx, wy)
    pure_python_bfs(warm_img, wx, wy)

    print(f"\n{'scene':16s} {'filled':>10s} {'levels':>6s} {'p.occ':>6s} "
          f"{'gpu k.ms':>9s} {'gpu t.ms':>9s} {'Mpx/s(k)':>8s} {'njit ms':>8s} "
          f"{'vs njit':>7s} {'pure ms':>10s} {'vs pure':>8s}")
    rows = [bench_scene(name, builder, note, slow) for name, builder, note in SCENES]
    sweep_rows = tpb_sweep()

    print("\nNotes: 'vs njit' = njit_ms / gpu_kernel_ms (kernel-only; <1x means "
          "the CPU wins). SM utilization is a constant 1/24 for this stage; "
          "occupancy is theoretical (tpb / max threads per SM). See the JSON "
          "for full timing decomposition and per-level frontier traces.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "config": {"gpu_repeats": GPU_REPEATS, "njit_repeats": NJIT_REPEATS,
                   "slow": slow},
        "scenes": rows,
        "tpb_sweep": sweep_rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"single_block_shared_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, f"single_block_shared_{stamp}.csv")
    csv_rows = [{k: v for k, v in row.items() if k != "level_sizes"}
                for row in rows]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
