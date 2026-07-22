"""Benchmark: N-block flood fill vs dual-global vs single-block v2 vs @njit.

Session header: a MEASURED device-to-device copy peak (bandwidth.py) — the
operational reference every derived bandwidth figure is expressed against.

Section 1, scene suite (median of GPU_REPEATS, blocks=None -> cooperative
max, tpb=256): @njit CPU reference, single-block v2 spill (cross-imported
from ch01_gpu_1blob_1block), dual-block global (cross-imported from
ch02_gpu_1blob_2block — the 2-block ancestor, re-measured fresh), and the
multi-block kernel instrumented AND bare. Every instrumented row carries
model_gb_s (derived bytes-moved / kernel time) and % of the measured peak.
The final scene is a guarded 8000^2 / 64M px stretch run: 10000^2 (the L2
hypothesis scale) needs ~3 GB of host arrays and this laptop's free RAM
cannot host it — the Chapter 1 host-RAM ceiling bites long before the GPU.

Section 2, the centerpiece: a blocks x tpb sweep ("many SMs and blocks and
warps configurations") on a big square, a big disk, and the serpentine.
Combos beyond the queried cooperative maximum are recorded as skipped, and
"max" is resolved per tpb (the capacity varies 8x across the tpb axis —
itself a register-pressure finding).

The 8-direction experiment rides along: every scene row carries conn8_*
columns (the 8-conn twin re-measured head-to-head, bare twin pinned to the
same grid, filled-parity asserted — valid because every suite scene is
solid/corridor), and the sweep runs the full blocks x tpb grid at 8-conn
on the two big blobs, rows tagged by connectivity. Session budget
~25-35 min.

Run:  uv run python -m flood_fill_cuda.chapters.ch03_gpu_1blob_nblock.benchmarks.benchmark
Writes JSON (with per-level traces and per-block stats) plus two CSVs to
results/ch03_gpu_1blob_nblock/benchmark_results/ (centralized, not next to
this script).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import gc
import json
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

from . import bandwidth
from ..flood_fill import flood_fill, max_blocks
from ..cpu_oracle import cpu_flood_fill
from ...ch01_gpu_1blob_1block.flood_fill import flood_fill as sbs_flood_fill
from ...ch02_gpu_1blob_2block.flood_fill import flood_fill as dual_flood_fill
from .. import scenes
from ....shared import results_paths
from numba import cuda

RESULTS_DIR = results_paths.results_dir("ch03_gpu_1blob_nblock", "benchmark_results")

GPU_REPEATS = 5
NJIT_REPEATS = 5
SWEEP_REPEATS = 3
# tpb=32 is the block-count ceiling: ~104 regs/thread cap every tpb at
# 12,288 resident threads (512/SM), so the smallest legal block size
# yields the most blocks — 384 cooperative blocks (16/SM).
TPB_SWEEP = [32, 64, 128, 256, 512]
BLOCKS_SWEEP = [1, 2, 4, 8, 16, 32, 48, 96, 128, 192, "max"]
SWEEP_SCENES = ["sq_4000_corner", "disk_4001_r1900", "serpentine_256"]
SWEEP_SCENES_CONN8 = ["sq_4000_corner", "disk_4001_r1900"]

SCENES = [
    ("sq_256_center", lambda: scenes.square_scene(256, 256, 128, 128),
     "small: per-level barrier overhead dominates"),
    ("sq_1024_center", lambda: scenes.square_scene(1024, 1024, 512, 512),
     "medium"),
    ("sq_2000_center", lambda: scenes.square_scene(2000, 2000, 1000, 1000),
     "1M px"),
    ("sq_4000_corner", lambda: scenes.full_red_scene(4000, 4000),
     "16M px, corner seed"),
    ("disk_2001_r950", lambda: scenes.disk_scene(2001, 2001, 950),
     "2.8M px disk"),
    ("disk_4001_r1900", lambda: scenes.disk_scene(4001, 4001, 1900),
     "11.3M px disk"),
    ("serpentine_256", lambda: scenes.serpentine_scene(256, 256),
     "32,896 one-pixel levels: the barrier-cost meter"),
    ("sq_4600_full_center", lambda: scenes.square_scene(4600, 4600, 4600, 4600),
     "21M px"),
    ("sq_5000_center", lambda: scenes.square_scene(5000, 5000, 5000, 5000),
     "25M px"),
    ("sq_6000_center", lambda: scenes.square_scene(6000, 6000, 6000, 6000),
     "36M px — the dual stage's largest"),
    ("sq_8000_center", lambda: scenes.square_scene(8000, 8000, 8000, 8000),
     "64M px stretch: guarded — host RAM, not the GPU, is the ceiling"),
]
GUARDED_SCENES = {"sq_8000_center"}


def _median_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _run_multi(img, sx, sy, tpb=256, blocks=None, bare=False, connectivity=4,
               repeats=GPU_REPEATS):
    kernel_times, total_times = [], []
    result = None
    for _ in range(repeats):
        result = flood_fill(img, sx, sy, threads_per_block=tpb,
                            blocks=blocks, bare=bare,
                            connectivity=connectivity)
        kernel_times.append(result.kernel_ms)
        total_times.append(result.total_ms)
    return result, statistics.median(kernel_times), statistics.median(total_times)


def bench_scene(name, builder, note, peak_gb_s):
    img, sx, sy = builder()
    width, height = img.shape[0], img.shape[1]

    # single-block v2 baseline (spill variant completes every scene)
    v2_kernel_times = []
    v2 = None
    for _ in range(GPU_REPEATS):
        v2 = sbs_flood_fill(img, sx, sy, variant="spill")
        v2_kernel_times.append(v2.kernel_ms)
    v2_kernel_ms = statistics.median(v2_kernel_times)
    v2_filled = v2.filled
    del v2
    gc.collect()

    # dual-block global baseline (the 2-block ancestor, re-measured fresh)
    dualg_kernel_times = []
    dg = None
    for _ in range(GPU_REPEATS):
        dg = dual_flood_fill(img, sx, sy, kernel="global")
        dualg_kernel_times.append(dg.kernel_ms)
    dualg_kernel_ms = statistics.median(dualg_kernel_times)
    dualg_filled = dg.filled
    del dg
    gc.collect()

    # the multi-block kernel, instrumented then bare
    r, multi_kernel_ms, multi_total_ms = _run_multi(img, sx, sy)
    row = {
        "scene": name, "note": note, "width": width, "height": height,
        "filled": r.filled, "levels": r.levels, "peak_frontier": r.peak_level,
        "v2_kernel_ms": v2_kernel_ms,
        "dual_global_kernel_ms": dualg_kernel_ms,
        "multi_kernel_ms": multi_kernel_ms,
        "multi_total_ms": multi_total_ms,
        "multi_blocks": r.blocks,
        "multi_tpb": r.threads_per_block,
        "multi_mpx_s_kernel": r.filled / multi_kernel_ms / 1000,
        "multi_speedup_vs_v2": v2_kernel_ms / multi_kernel_ms,
        "multi_speedup_vs_dual": dualg_kernel_ms / multi_kernel_ms,
        "multi_distinct_sms": r.distinct_sms,
        "multi_balance_cv_pct": r.balance_cv_pct,
        "multi_balance_min_max_pct": r.balance_min_max_pct,
        "multi_thread_util_pct": r.thread_util_pct,
        "multi_grid_occupancy_pct": r.grid_occupancy_pct,
        "multi_model_bytes": r.model_bytes,
        "multi_model_gb_s": r.model_gb_s,
        "multi_pct_of_peak": 100.0 * r.model_gb_s / peak_gb_s,
    }
    filled_ref = r.filled
    traces = {
        "level_sizes": r.level_sizes.tolist(),
        "processed_per_block": r.processed_per_block.tolist(),
        "sm_ids": r.sm_ids,
    }
    del r
    gc.collect()

    # Pin the bare twin to the SAME grid: its lower register count would
    # otherwise resolve blocks=None to a larger cooperative max (72 vs 48
    # at tpb=256), and the overhead comparison must vary only the
    # instrumentation, never the launch shape.
    _, bare_ms, _ = _run_multi(img, sx, sy, bare=True,
                               blocks=row["multi_blocks"])
    row["multi_bare_kernel_ms"] = bare_ms
    row["multi_instrumentation_overhead_pct"] = \
        100.0 * (multi_kernel_ms - bare_ms) / bare_ms

    # The 8-direction twin, head-to-head on the same scene. Valid because
    # every suite scene is solid/corridor (no diagonal-only gaps), so the
    # fill SET is identical under 4/8-conn — only depth/levels differ.
    r8, conn8_ms, _ = _run_multi(img, sx, sy, connectivity=8)
    row.update({
        "conn8_kernel_ms": conn8_ms,
        "conn8_blocks": r8.blocks,
        "conn8_levels": r8.levels,
        "conn8_peak_frontier": r8.peak_level,
        "conn8_mpx_s": r8.filled / conn8_ms / 1000,
        "conn8_vs_conn4": multi_kernel_ms / conn8_ms,  # >1 = 8-conn faster
        "conn8_thread_util_pct": r8.thread_util_pct,
        "conn8_cas_attempts": r8.cas_attempts,
        "conn8_model_bytes": r8.model_bytes,
        "conn8_model_gb_s": r8.model_gb_s,
        "conn8_pct_of_peak": 100.0 * r8.model_gb_s / peak_gb_s,
    })
    conn8_filled = r8.filled
    traces["conn8_level_sizes"] = r8.level_sizes.tolist()
    del r8
    gc.collect()
    _, bare8_ms, _ = _run_multi(img, sx, sy, connectivity=8, bare=True,
                                blocks=row["conn8_blocks"])
    row["conn8_bare_kernel_ms"] = bare8_ms
    row["conn8_overhead_pct"] = 100.0 * (conn8_ms - bare8_ms) / bare8_ms

    # @njit reference last (transient memory peak) + cross-check
    _, _, _, ref_filled = cpu_flood_fill(img, sx, sy)
    njit_ms = _median_ms(lambda: cpu_flood_fill(img, sx, sy), NJIT_REPEATS)
    row["njit_ms"] = njit_ms
    row["njit_mpx_s"] = ref_filled / njit_ms / 1000
    row["multi_speedup_vs_njit"] = njit_ms / multi_kernel_ms
    row["speedup_v2_vs_njit"] = njit_ms / v2_kernel_ms
    ok = (filled_ref == ref_filled == v2_filled == dualg_filled
          == conn8_filled)
    row["filled_crosscheck"] = "OK" if ok else "MISMATCH"
    row.update(traces)

    check = "[OK]" if ok else "[MISMATCH!]"
    print(f"{name:20s} {row['filled']:>11,d} {njit_ms:9.2f} {v2_kernel_ms:8.2f} "
          f"{dualg_kernel_ms:8.2f} {multi_kernel_ms:8.2f} {bare_ms:8.2f} "
          f"{conn8_ms:8.2f} {row['conn8_vs_conn4']:5.2f}x "
          f"{row['multi_speedup_vs_v2']:6.2f}x {row['multi_blocks']:>4d} "
          f"{row['multi_model_gb_s']:7.1f} {row['multi_pct_of_peak']:5.1f}% {check}")
    return row


def block_tpb_sweep(peak_gb_s):
    """The centerpiece: how do runtime and modeled bandwidth respond to
    blocks (SM coverage) and threads per block (warps per block)? Run at
    4-conn on all sweep scenes, then at 8-conn on the two big blobs (the
    serpentine is skipped there: prediction says levels barely change, so
    the 8-conn run would only re-measure probe cost). Same-session grids
    make the 4-vs-8 comparison same-clock fair."""
    rows = []
    lookup = dict((n, b) for n, b, _ in SCENES)
    passes = ([(s, 4) for s in SWEEP_SCENES]
              + [(s, 8) for s in SWEEP_SCENES_CONN8])
    for sname, conn in passes:
        img, sx, sy = lookup[sname]()
        results = {}
        for tpb in TPB_SWEEP:
            coop = max_blocks(threads_per_block=tpb, connectivity=conn)
            seen = set()
            for b in BLOCKS_SWEEP:
                n = coop if b == "max" else b
                if n in seen:
                    continue
                seen.add(n)
                if n > coop:
                    rows.append({"scene": sname, "connectivity": conn,
                                 "tpb": tpb, "blocks": n,
                                 "skipped": f"exceeds coop max ({coop})"})
                    continue
                r, ms, _ = _run_multi(img, sx, sy, tpb=tpb, blocks=n,
                                      connectivity=conn,
                                      repeats=SWEEP_REPEATS)
                rows.append({
                    "scene": sname, "connectivity": conn,
                    "tpb": tpb, "blocks": n,
                    "is_coop_max": n == coop,
                    "kernel_ms": ms,
                    "mpx_s": r.filled / ms / 1000,
                    "model_gb_s": bandwidth.model_gb_s(r.model_bytes, ms),
                    "pct_of_peak": 100.0 * bandwidth.model_gb_s(
                        r.model_bytes, ms) / peak_gb_s,
                    "balance_cv_pct": r.balance_cv_pct,
                    "distinct_sms": r.distinct_sms,
                    "thread_util_pct": r.thread_util_pct,
                    "grid_occupancy_pct": r.grid_occupancy_pct,
                })
                results[(tpb, n)] = rows[-1]
                del r
        gc.collect()

        # console grid: blocks down, tpb across, Mpx/s ("-" = beyond coop max)
        all_blocks = sorted({n for (t, n) in results})
        print(f"\n{sname} ({conn}-conn): Mpx/s by blocks x tpb "
              f"(median of {SWEEP_REPEATS}; * = coop max for that tpb)")
        print(f"{'blocks':>8s}" + "".join(f" {'tpb ' + str(t):>12s}"
                                          for t in TPB_SWEEP))
        for n in all_blocks:
            line = f"{n:>8d}"
            for t in TPB_SWEEP:
                cell = results.get((t, n))
                if cell is None:
                    line += f" {'-':>12s}"
                else:
                    star = "*" if cell["is_coop_max"] else ""
                    line += f" {cell['mpx_s']:>11.1f}{star or ' '}"
            print(line)
        del img
        gc.collect()
    return rows


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    sm_count = int(device.MULTIPROCESSOR_COUNT)
    print(f"Device: {dev_name.strip()} ({sm_count} SMs; blocks=None -> "
          f"cooperative max, observed via %smid)")

    print("Measuring D2D copy peak (the reference for all model GB/s)...")
    peak = bandwidth.measure_peak_bandwidth()
    peak_gb_s = peak["gb_s"]
    gc.collect()
    try:  # hand the two 256 MB probe buffers back before the big scenes
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass
    print(f"  measured peak: {peak_gb_s:.1f} GB/s "
          f"(runs: {', '.join(f'{r:.0f}' for r in peak['runs_gb_s'])})")

    print("Warming up JITs (incl. NVRTC-linked %smid helper)...")
    warm_img, wx, wy = scenes.square_scene(64, 64, 32, 32)
    flood_fill(warm_img, wx, wy)
    flood_fill(warm_img, wx, wy, bare=True)
    flood_fill(warm_img, wx, wy, connectivity=8)
    flood_fill(warm_img, wx, wy, connectivity=8, bare=True)
    sbs_flood_fill(warm_img, wx, wy, variant="spill")
    dual_flood_fill(warm_img, wx, wy, kernel="global")
    cpu_flood_fill(warm_img, wx, wy)

    coop_by_tpb = {tpb: max_blocks(threads_per_block=tpb) for tpb in TPB_SWEEP}
    coop_by_tpb_conn8 = {tpb: max_blocks(threads_per_block=tpb, connectivity=8)
                         for tpb in TPB_SWEEP}
    print("Cooperative capacity by tpb (instrumented kernel): "
          + ", ".join(f"{t}->{n}" for t, n in coop_by_tpb.items()))
    print("Cooperative capacity by tpb (8-conn twin):         "
          + ", ".join(f"{t}->{n}" for t, n in coop_by_tpb_conn8.items()))

    print(f"\n{'scene':20s} {'filled':>11s} {'njit ms':>9s} {'v2 ms':>8s} "
          f"{'dual ms':>8s} {'multi':>8s} {'bare':>8s} {'conn8':>8s} "
          f"{'4v8':>6s} {'vs v2':>7s} {'blks':>4s} {'GB/s':>7s} {'%peak':>6s}")
    rows = []
    for name, builder, note in SCENES:
        if name in GUARDED_SCENES:
            try:
                rows.append(bench_scene(name, builder, note, peak_gb_s))
            except MemoryError:
                rows.append({"scene": name, "note": note,
                             "skipped": "host MemoryError — the Chapter 1 "
                                        "host-RAM ceiling, not the GPU"})
                print(f"{name:20s} SKIPPED (host MemoryError)")
            gc.collect()
        else:
            rows.append(bench_scene(name, builder, note, peak_gb_s))
    sweep_rows = block_tpb_sweep(peak_gb_s)

    print("\nNotes: all GPU times are kernel-only medians. 'vs v2' = "
          "multi-block (cooperative max) vs the single-block v2 spill "
          "kernel. GB/s figures are the DERIVED bytes-moved model (see "
          "bandwidth.MODEL_NOTE) against the measured copy peak of "
          f"{peak_gb_s:.0f} GB/s — a lower bound on traffic, not an ncu "
          "measurement. Sweep cells marked * ran at that tpb's cooperative "
          "maximum.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": sm_count,
        "measured_peak_gb_s": peak_gb_s,
        "measured_peak_runs_gb_s": peak["runs_gb_s"],
        "bandwidth_model": bandwidth.MODEL_NOTE,
        "coop_max_by_tpb": coop_by_tpb,
        "coop_max_by_tpb_conn8": coop_by_tpb_conn8,
        "connectivity_experiment": (
            "8-conn twin kernels benchmarked head-to-head: conn8_* scene "
            "columns and connectivity-tagged sweep rows; predictions in "
            "README §The 8-direction experiment"),
        "config": {"gpu_repeats": GPU_REPEATS, "njit_repeats": NJIT_REPEATS,
                   "sweep_repeats": SWEEP_REPEATS,
                   "blocks_sweep": [str(b) for b in BLOCKS_SWEEP],
                   "tpb_sweep": TPB_SWEEP,
                   "sweep_scenes_conn8": SWEEP_SCENES_CONN8},
        "scenes": rows,
        "block_tpb_sweep": sweep_rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"multi_block_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    trace_keys = ("level_sizes", "conn8_level_sizes", "processed_per_block",
                  "sm_ids")
    csv_rows = [{k: v for k, v in row.items() if k not in trace_keys}
                for row in rows if "skipped" not in row]
    scenes_csv = os.path.join(RESULTS_DIR, f"multi_block_{stamp}_scenes.csv")
    with open(scenes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    sweep_fields = ["scene", "connectivity", "tpb", "blocks", "is_coop_max",
                    "kernel_ms", "mpx_s", "model_gb_s", "pct_of_peak",
                    "balance_cv_pct", "distinct_sms", "thread_util_pct",
                    "grid_occupancy_pct", "skipped"]
    sweep_csv = os.path.join(RESULTS_DIR, f"multi_block_{stamp}_sweep.csv")
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_fields)
        writer.writeheader()
        writer.writerows(sweep_rows)

    print(f"\nResults written to:\n  {json_path}\n  {scenes_csv}\n  {sweep_csv}")


if __name__ == "__main__":
    main()
