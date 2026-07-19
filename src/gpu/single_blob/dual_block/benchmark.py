"""Benchmark: dual-block partitionings vs single-block v2 vs @njit CPU.

Per scene (median of GPU_REPEATS):
  - @njit CPU reference (the honest sequential bar)
  - single-block v2 spill kernel (cross-imported from ../single_block_shared
    — the "what does the 2nd block buy?" baseline)
  - the three dual kernels, instrumented AND bare (identical BFS without
    counters/traces), so the instrumentation overhead is measured
Then two focused sections:
  - tpb sweep {64..512} x three kernels on sq_2000_center (2x512 = 1,024
    total threads, matching the single-block sweep's top config)
  - the placement experiment: single-block v2 at 768/1024 threads vs the
    pinned pair on ONE SM (2x768 = 1,536 resident threads, occupancy-forced)
    vs the same pair spread by the scheduler across two SMs — with the
    observed %smid values recorded as proof of placement

Run:  uv run python src/gpu/single_blob/dual_block/benchmark.py
Writes JSON (with per-level global and per-block traces) and CSV to
benchmark_results/ next to this script.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

from flood_fill import flood_fill
from reference import cpu_flood_fill, load_by_path, _SBS
import scenes
from numba import cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "benchmark_results")

GPU_REPEATS = 5
NJIT_REPEATS = 5
KERNELS = ["split", "global", "dirsplit"]
TPB_SWEEP = [64, 128, 256, 512]
PLACEMENT_SCENES = ["sq_2000_center", "sq_6000_center"]


def load_single_block_shared():
    """Load ../single_block_shared/flood_fill.py despite the module-name
    collision: its `from kernels import ...` consults sys.modules first,
    which is already bound to THIS package's kernels — so temporarily remap
    the name while executing it."""
    sbs_kernels = load_by_path("_sbs_kernels", os.path.join(_SBS, "kernels.py"))
    saved = sys.modules.get("kernels")
    sys.modules["kernels"] = sbs_kernels
    try:
        return load_by_path("_sbs_flood_fill", os.path.join(_SBS, "flood_fill.py"))
    finally:
        if saved is not None:
            sys.modules["kernels"] = saved
        else:
            sys.modules.pop("kernels", None)


sbs = load_single_block_shared()

SCENES = [
    ("sq_256_center", lambda: scenes.square_scene(256, 256, 128, 128),
     "small: per-level barrier overhead dominates"),
    ("sq_1024_center", lambda: scenes.square_scene(1024, 1024, 512, 512),
     "medium"),
    ("sq_2000_center", lambda: scenes.square_scene(2000, 2000, 1000, 1000),
     "1M px"),
    ("sq_4000_corner", lambda: scenes.full_red_scene(4000, 4000),
     "16M px, corner seed"),
    ("serpentine_256", lambda: scenes.serpentine_scene(256, 256),
     "seam-parallel snake: worst case for every parallel design"),
    ("seam_serpentine_256", lambda: scenes.seam_serpentine_scene(256, 256),
     "seam-crossing snake: the split kernel's anti-pattern"),
    ("offcenter_2000", lambda: scenes.offcenter_blob_scene(2000, 2000, 900),
     "blob entirely in one half: split's 0% balance vs dirsplit's ~50/50"),
    ("sq_2600_full_center", scenes.overflow_scene,
     "v1's tripwire scene: two split rings absorb it with zero spill"),
    ("sq_4600_full_center", lambda: scenes.square_scene(4600, 4600, 4600, 4600),
     "21M px: both split halves must spill"),
    ("sq_5000_center", lambda: scenes.square_scene(5000, 5000, 5000, 5000),
     "25M px"),
    ("sq_6000_center", lambda: scenes.square_scene(6000, 6000, 6000, 6000),
     "36M px — host RAM is this laptop's ceiling"),
]


def _median_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _run_dual(img, sx, sy, kernel, bare=False, tpb=256):
    kernel_times, total_times = [], []
    result = None
    for _ in range(GPU_REPEATS):
        result = flood_fill(img, sx, sy, threads_per_block=tpb,
                            kernel=kernel, bare=bare)
        kernel_times.append(result.kernel_ms)
        total_times.append(result.total_ms)
    return result, statistics.median(kernel_times), statistics.median(total_times)


def bench_scene(name, builder, note):
    img, sx, sy = builder()
    width, height = img.shape[0], img.shape[1]

    # single-block v2 baseline (spill variant completes every scene)
    v2_kernel_times, v2_total_times = [], []
    v2 = None
    for _ in range(GPU_REPEATS):
        v2 = sbs.flood_fill(img, sx, sy, variant="spill")
        v2_kernel_times.append(v2.kernel_ms)
        v2_total_times.append(v2.total_ms)
    v2_kernel_ms = statistics.median(v2_kernel_times)
    v2_total_ms = statistics.median(v2_total_times)
    v2_filled = v2.filled
    del v2

    row = {
        "scene": name, "note": note, "width": width, "height": height,
        "v2_kernel_ms": v2_kernel_ms, "v2_total_ms": v2_total_ms,
    }
    traces = {}

    filled_ref = None
    for kernel in KERNELS:
        r, k_ms, t_ms = _run_dual(img, sx, sy, kernel)
        row[f"{kernel}_kernel_ms"] = k_ms
        row[f"{kernel}_total_ms"] = t_ms
        row[f"{kernel}_mpx_s_kernel"] = r.filled / k_ms / 1000
        row[f"{kernel}_balance_pct"] = r.balance_pct
        row[f"{kernel}_speedup_vs_v2"] = v2_kernel_ms / k_ms
        row[f"{kernel}_sm_ids"] = [r.sm_id_b0, r.sm_id_b1]
        if kernel == "split":
            row["split_inbox_to_b0"] = r.inbox_to_b0
            row["split_inbox_to_b1"] = r.inbox_to_b1
            row["split_spilled_b0"] = r.spilled_b0
            row["split_spilled_b1"] = r.spilled_b1
        if filled_ref is None:
            filled_ref = r.filled
            row["filled"] = r.filled
            row["levels"] = r.levels
            row["peak_frontier"] = r.peak_level
            traces["level_sizes"] = r.level_sizes.tolist()
        assert r.filled == filled_ref
        traces[f"{kernel}_per_block"] = r.level_sizes_per_block.tolist()
        del r

        _, bare_ms, _ = _run_dual(img, sx, sy, kernel, bare=True)
        row[f"{kernel}_bare_kernel_ms"] = bare_ms
        row[f"{kernel}_instrumentation_overhead_pct"] = \
            100.0 * (k_ms - bare_ms) / bare_ms

    # @njit reference last (transient memory peak) + cross-check
    _, _, _, ref_filled = cpu_flood_fill(img, sx, sy)
    njit_ms = _median_ms(lambda: cpu_flood_fill(img, sx, sy), NJIT_REPEATS)
    row["njit_ms"] = njit_ms
    row["njit_mpx_s"] = ref_filled / njit_ms / 1000
    for kernel in KERNELS:
        row[f"{kernel}_speedup_vs_njit"] = njit_ms / row[f"{kernel}_kernel_ms"]
    row["speedup_v2_vs_njit"] = njit_ms / v2_kernel_ms
    ok = filled_ref == ref_filled == v2_filled
    row["filled_crosscheck"] = "OK" if ok else "MISMATCH"
    row.update(traces)

    best = min(KERNELS, key=lambda k: row[f"{k}_kernel_ms"])
    check = "[OK]" if ok else "[MISMATCH!]"
    print(f"{name:20s} {row['filled']:>11,d} {njit_ms:9.2f} {v2_kernel_ms:8.2f} "
          f"{row['split_kernel_ms']:8.2f} {row['global_kernel_ms']:8.2f} "
          f"{row['dirsplit_kernel_ms']:8.2f} "
          f"{best:8s} {row[f'{best}_speedup_vs_v2']:6.2f}x "
          f"{row['split_balance_pct']:5.0f}% {row['dirsplit_balance_pct']:5.0f}% {check}")
    return row


def tpb_sweep():
    print(f"\nthreads-per-block sweep on sq_2000_center "
          f"(median kernel ms of {GPU_REPEATS}; 2 blocks each):")
    img, sx, sy = scenes.square_scene(2000, 2000, 1000, 1000)
    print(f"{'tpb':>6s}" + "".join(f" {k + ' ms':>12s} {k + ' Mpx/s':>12s}"
                                   for k in KERNELS))
    rows = []
    for tpb in TPB_SWEEP:
        line = f"{tpb:>6d}"
        for kernel in KERNELS:
            r, ms, _ = _run_dual(img, sx, sy, kernel, tpb=tpb)
            rows.append({"kernel": kernel, "tpb": tpb, "kernel_ms": ms,
                         "mpx_s": r.filled / ms / 1000,
                         "thread_util_pct": r.thread_util_pct,
                         "balance_pct": r.balance_pct})
            line += f" {ms:>12.2f} {rows[-1]['mpx_s']:>12.2f}"
            del r
        print(line)
    return rows


def placement_experiment():
    """Same BFS, one variable — where the threads live — at a time."""
    print("\nplacement experiment (median kernel ms of "
          f"{GPU_REPEATS}; smids observed per run):")
    print(f"{'scene':20s} {'config':22s} {'kernel_ms':>10s} {'Mpx/s':>8s} "
          f"{'smids':>10s}")
    rows = []
    lookup = dict((n, b) for n, b, _ in SCENES)
    for sname in PLACEMENT_SCENES:
        img, sx, sy = lookup[sname]()

        for tpb in (768, 1024):
            times = []
            r = None
            for _ in range(GPU_REPEATS):
                r = sbs.flood_fill(img, sx, sy, threads_per_block=tpb,
                                   variant="spill")
                times.append(r.kernel_ms)
            ms = statistics.median(times)
            rows.append({"scene": sname, "config": f"v2 1x{tpb} (1 SM)",
                         "kernel_ms": ms, "mpx_s": r.filled / ms / 1000,
                         "sm_ids": None})
            print(f"{sname:20s} {rows[-1]['config']:22s} {ms:>10.2f} "
                  f"{rows[-1]['mpx_s']:>8.2f} {'—':>10s}")
            del r

        for placement, label in (("same_sm", "pinned 2x768 same SM"),
                                 ("spread", "pinned 2x768 spread")):
            times = []
            r = None
            for _ in range(GPU_REPEATS):
                r = flood_fill(img, sx, sy, kernel="pinned",
                               threads_per_block=768, placement=placement)
                times.append(r.kernel_ms)
            ms = statistics.median(times)
            rows.append({"scene": sname, "config": label, "kernel_ms": ms,
                         "mpx_s": r.filled / ms / 1000,
                         "sm_ids": [r.sm_id_b0, r.sm_id_b1]})
            print(f"{sname:20s} {label:22s} {ms:>10.2f} "
                  f"{rows[-1]['mpx_s']:>8.2f} "
                  f"{str(tuple(rows[-1]['sm_ids'])):>10s}")
            del r
    return rows


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    print(f"Device: {dev_name.strip()} ({device.MULTIPROCESSOR_COUNT} SMs; "
          f"dual kernels use 2 blocks; placement observed via %smid)")

    print("Warming up JITs (incl. NVRTC-linked %smid helper)...")
    warm_img, wx, wy = scenes.square_scene(64, 64, 32, 32)
    for kernel in KERNELS:
        flood_fill(warm_img, wx, wy, kernel=kernel)
        flood_fill(warm_img, wx, wy, kernel=kernel, bare=True)
    flood_fill(warm_img, wx, wy, kernel="pinned", threads_per_block=768,
               placement="spread")
    sbs.flood_fill(warm_img, wx, wy, variant="spill")
    cpu_flood_fill(warm_img, wx, wy)

    print(f"\n{'scene':20s} {'filled':>11s} {'njit ms':>9s} {'v2 ms':>8s} "
          f"{'split':>8s} {'global':>8s} {'dirspl':>8s} {'best':8s} "
          f"{'vs v2':>7s} {'s.bal':>6s} {'d.bal':>6s}")
    rows = [bench_scene(name, builder, note) for name, builder, note in SCENES]
    sweep_rows = tpb_sweep()
    placement_rows = placement_experiment()

    print("\nNotes: all GPU times are kernel-only medians. 'vs v2' = best "
          "dual kernel vs the single-block v2 spill kernel (>1x means the "
          "second block paid for itself). s.bal/d.bal = split/dirsplit "
          "per-block balance; remember the serpentine result — totals can "
          "look balanced while the per-level traces (in the JSON) show the "
          "blocks never working at the same time. Instrumentation overhead "
          "columns compare each kernel against its bare twin.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "config": {"gpu_repeats": GPU_REPEATS, "njit_repeats": NJIT_REPEATS},
        "scenes": rows,
        "tpb_sweep": sweep_rows,
        "placement": placement_rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"dual_block_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    trace_keys = ("level_sizes", "split_per_block", "global_per_block",
                  "dirsplit_per_block")
    csv_rows = [{k: v for k, v in row.items() if k not in trace_keys}
                for row in rows]
    csv_path = os.path.join(RESULTS_DIR, f"dual_block_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
