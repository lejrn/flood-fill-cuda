"""Benchmark: the per-barrier work experiments, head-to-head.

Times conn4 / conn8 / radius-2 / warp-coop (and the bare twins of the
three conn8-family kernels) in ONE interleaved round-robin per scene —
every config timed once per round, within-round order alternating. Not a
stylistic choice: this GPU's per-run spread reaches ~73% of the median
over a session (dual_blob's Finding 3), so sequential A-then-B timing
produces sign-flipping artifacts. Only interleaving makes the A/B ratios
(r2_vs_conn8, wc_vs_conn8) trustworthy.

All timed configs run at ONE pinned grid — the minimum cooperative
capacity across the kernels at tpb=256 — because the twins' capacities
are NOT equal (wc compiles to fewer registers and can host a bigger
grid at low tpb); ratios must never compare unequal grids. Each
kernel's own capacity is recorded in the JSON.

Memory discipline: only scalars are harvested from each result and the
result object is dropped inside the loop — retaining a sq_8000 result
(~0.8 GB of host arrays) per config would blow the Chapter 1 host-RAM
ceiling.

The @njit 8-conn oracle provides njit_ms context and the filled
cross-check (valid for every config: the radius-2 guard provably
preserves the conn8 fill set, and wc is bit-identical by construction).

Run:  uv run python -m flood_fill_cuda.chapters.ch03_gpu_1blob_nblock.benchmarks.benchmark_connectivity_and_barrier_work
Writes JSON + CSV (still the neighbors_* prefix — the dashboard already
globs for that name) to results/ch03_gpu_1blob_nblock/benchmark_results/
(centralized, not next to this script).
Budget ~15-25 min (7 kernel JITs + 35 fills per scene, 6 scenes).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import gc
import json
import statistics
import time
from datetime import datetime, timezone

import numpy as np

from . import bandwidth
from ..flood_fill import flood_fill, max_blocks
from ..cpu_oracle import cpu_flood_fill_8
from .. import scenes
from ....shared import results_paths
from numba import cuda

RESULTS_DIR = results_paths.results_dir("ch03_gpu_1blob_nblock", "benchmark_results")

GPU_REPEATS = 5
NJIT_REPEATS = 3
TPB = 256

SCENES = [
    ("sq_2000_center", lambda: scenes.square_scene(2000, 2000, 1000, 1000),
     "1M px: mid-size, where conn8's utilization was lowest"),
    ("sq_4000_corner", lambda: scenes.full_red_scene(4000, 4000),
     "16M px, corner seed"),
    ("disk_4001_r1900", lambda: scenes.disk_scene(4001, 4001, 1900),
     "11.3M px disk"),
    ("serpentine_256", lambda: scenes.serpentine_scene(256, 256),
     "32,896 one-pixel levels: wc's headline scene, r2's losing one"),
    ("sq_6000_center", lambda: scenes.square_scene(6000, 6000, 6000, 6000),
     "36M px"),
    ("sq_8000_center", lambda: scenes.square_scene(8000, 8000, 8000, 8000),
     "64M px stretch: guarded — host RAM, not the GPU, is the ceiling"),
]
GUARDED_SCENES = {"sq_8000_center"}

# name -> flood_fill kwargs beyond (img, sx, sy, tpb, blocks)
CONFIGS = {
    "conn4": {"connectivity": 4},
    "conn8": {"connectivity": 8},
    "r2": {"connectivity": 8, "radius": 2},
    "wc": {"connectivity": 8, "probe_layout": "warp"},
    "conn8_bare": {"connectivity": 8, "bare": True},
    "r2_bare": {"connectivity": 8, "radius": 2, "bare": True},
    "wc_bare": {"connectivity": 8, "probe_layout": "warp", "bare": True},
}


def _median(values):
    return statistics.median(values)


def _run_round_robin(img, sx, sy, blocks, repeats=GPU_REPEATS):
    """Every config timed once per round, order alternating per round.
    Harvests scalars only; each result object is dropped inside the loop
    (a 64M px result holds ~0.8 GB of host arrays)."""
    names = list(CONFIGS)
    samples = {n: [] for n in names}
    meta = {}
    for i in range(repeats):
        order = names if i % 2 == 0 else list(reversed(names))
        for n in order:
            r = flood_fill(img, sx, sy, threads_per_block=TPB,
                           blocks=blocks, **CONFIGS[n])
            samples[n].append(r.kernel_ms)
            if n not in meta:  # deterministic per config — harvest once
                meta[n] = {
                    "filled": r.filled,
                    "levels": r.levels,
                    "interior": r.interior,
                    "cas_attempts": r.cas_attempts,
                    "thread_util_pct": r.thread_util_pct,
                    "model_bytes": r.model_bytes,
                }
            del r
        gc.collect()
    stats = {}
    for n in names:
        k = samples[n]
        stats[n] = {
            "ms": _median(k),
            "ms_min": min(k),
            "ms_max": max(k),
            "ms_stdev": statistics.stdev(k) if len(k) > 1 else 0.0,
        }
    return stats, meta


def bench_scene(name, builder, note, blocks, peak_gb_s):
    img, sx, sy = builder()
    width, height = img.shape[0], img.shape[1]

    stats, meta = _run_round_robin(img, sx, sy, blocks)

    ms = {n: stats[n]["ms"] for n in CONFIGS}
    row = {
        "scene": name, "note": note, "width": width, "height": height,
        "filled": meta["conn8"]["filled"],
        "blocks": blocks, "tpb": TPB,
    }
    for n in CONFIGS:
        row[f"{n}_kernel_ms"] = stats[n]["ms"]
        row[f"{n}_kernel_ms_min"] = stats[n]["ms_min"]
        row[f"{n}_kernel_ms_max"] = stats[n]["ms_max"]
        row[f"{n}_kernel_ms_stdev"] = stats[n]["ms_stdev"]

    # The two experiment ratios (>1 = the variant beats conn8), median and
    # best-vs-best forms — agreement between them is the drift signal.
    row.update({
        "conn8_vs_conn4": ms["conn4"] / ms["conn8"],
        "r2_vs_conn8": ms["conn8"] / ms["r2"],
        "r2_vs_conn8_min": stats["conn8"]["ms_min"] / stats["r2"]["ms_min"],
        "wc_vs_conn8": ms["conn8"] / ms["wc"],
        "wc_vs_conn8_min": stats["conn8"]["ms_min"] / stats["wc"]["ms_min"],
        "conn8_levels": meta["conn8"]["levels"],
        "r2_levels": meta["r2"]["levels"],
        "wc_levels": meta["wc"]["levels"],
        "r2_interior": meta["r2"]["interior"],
        "r2_interior_pct": (100.0 * meta["r2"]["interior"]
                            / meta["r2"]["filled"]),
        "conn8_cas_attempts": meta["conn8"]["cas_attempts"],
        "r2_cas_attempts": meta["r2"]["cas_attempts"],
        "conn8_thread_util_pct": meta["conn8"]["thread_util_pct"],
        "r2_thread_util_pct": meta["r2"]["thread_util_pct"],
        "wc_thread_util_pct": meta["wc"]["thread_util_pct"],
        "conn8_overhead_pct": 100.0 * (ms["conn8"] - ms["conn8_bare"])
        / ms["conn8_bare"],
        "r2_overhead_pct": 100.0 * (ms["r2"] - ms["r2_bare"])
        / ms["r2_bare"],
        "wc_overhead_pct": 100.0 * (ms["wc"] - ms["wc_bare"])
        / ms["wc_bare"],
    })
    for n in ("conn8", "r2", "wc"):
        gb_s = bandwidth.model_gb_s(meta[n]["model_bytes"], ms[n])
        row[f"{n}_model_gb_s"] = gb_s
        row[f"{n}_pct_of_peak"] = 100.0 * gb_s / peak_gb_s

    # -- @njit 8-conn oracle: njit_ms context + the filled cross-check.
    #    Valid for every config: r2's guard provably preserves the conn8
    #    fill set; wc is bit-identical by construction.
    oracle_filled = None
    njit_times = []
    for _ in range(NJIT_REPEATS):
        t0 = time.perf_counter()
        _, _, _, oracle_filled = cpu_flood_fill_8(img, sx, sy)
        njit_times.append((time.perf_counter() - t0) * 1000)
    njit_ms = _median(njit_times)
    row["njit_ms"] = njit_ms
    row["r2_speedup_vs_njit"] = njit_ms / ms["r2"]
    ok = (oracle_filled == meta["conn8"]["filled"] == meta["r2"]["filled"]
          == meta["wc"]["filled"] == meta["conn4"]["filled"]
          == meta["conn8_bare"]["filled"] == meta["r2_bare"]["filled"]
          == meta["wc_bare"]["filled"])
    row["filled_crosscheck"] = "OK" if ok else "MISMATCH"

    check = "[OK]" if ok else "[MISMATCH!]"
    print(f"{name:18s} {row['filled']:>11,d} {ms['conn4']:8.2f} "
          f"{ms['conn8']:8.2f} {ms['r2']:8.2f} {ms['wc']:8.2f} "
          f"{row['r2_vs_conn8']:6.2f}x {row['r2_vs_conn8_min']:6.2f}x "
          f"{row['wc_vs_conn8']:6.2f}x {row['wc_vs_conn8_min']:6.2f}x "
          f"{row['r2_interior_pct']:5.1f}% {check}")
    del img
    gc.collect()
    return row


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    sm_count = int(device.MULTIPROCESSOR_COUNT)
    print(f"Device: {dev_name.strip()} ({sm_count} SMs; tpb={TPB})")

    print("Measuring D2D copy peak (the reference for all model GB/s)...")
    peak = bandwidth.measure_peak_bandwidth()
    peak_gb_s = peak["gb_s"]
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass
    print(f"  measured peak: {peak_gb_s:.1f} GB/s")

    print("Warming up JITs (7 kernels + oracle)...")
    warm_img, wx, wy = scenes.square_scene(64, 64, 32, 32)
    for kw in CONFIGS.values():
        flood_fill(warm_img, wx, wy, **kw)
    cpu_flood_fill_8(warm_img, wx, wy)

    # Pin ALL timed configs to the minimum cooperative capacity across the
    # kernels — capacities are measured per twin, never assumed equal (wc
    # compiles leaner and can host more blocks at low tpb).
    caps = {n: max_blocks(threads_per_block=TPB, **kw)
            for n, kw in CONFIGS.items()}
    pin_blocks = min(caps.values())
    print("Cooperative capacity @tpb=256 per kernel: "
          + ", ".join(f"{n}={c}" for n, c in caps.items())
          + f" -> all configs pinned to {pin_blocks} blocks")

    print(f"\n{'scene':18s} {'filled':>11s} {'conn4':>8s} {'conn8':>8s} "
          f"{'r2':>8s} {'wc':>8s} {'r2/8':>7s} {'(min)':>7s} "
          f"{'wc/8':>7s} {'(min)':>7s} {'int%':>6s}")
    rows = []
    for name, builder, note in SCENES:
        if name in GUARDED_SCENES:
            try:
                rows.append(bench_scene(name, builder, note, pin_blocks,
                                        peak_gb_s))
            except MemoryError:
                rows.append({"scene": name, "note": note,
                             "skipped": "host MemoryError — the Chapter 1 "
                                        "host-RAM ceiling, not the GPU"})
                print(f"{name:18s} SKIPPED (host MemoryError)")
            gc.collect()
        else:
            rows.append(bench_scene(name, builder, note, pin_blocks,
                                    peak_gb_s))

    print("\nNotes: all GPU times are kernel-only medians over an "
          "INTERLEAVED round-robin (every config timed once per round, "
          "order alternating). 'r2/8' and 'wc/8' are conn8_ms/variant_ms "
          "(>1 = the variant faster); '(min)' is the best-vs-best form — "
          "agreement between median and min ratios is the signal that a "
          "difference is real, not drift. 'int%' = interior pixels (guard "
          "passes) as % of filled. All configs pinned to one common grid; "
          "GB/s figures are the DERIVED bytes-moved model against the "
          f"measured {peak_gb_s:.0f} GB/s copy peak.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": sm_count,
        "tpb": TPB,
        "measured_peak_gb_s": peak_gb_s,
        "bandwidth_model": bandwidth.MODEL_NOTE,
        "coop_max_by_config": caps,
        "pinned_blocks": pin_blocks,
        "experiment": (
            "per-barrier work: radius-2 (guarded ring-2 probes — half the "
            "levels, ~3x the probes; fill set provably identical to conn8) "
            "and warp-coop (4 entries x 8 dirs per warp — one probe round "
            "per chunk instead of 8; results bit-identical to conn8) vs "
            "the conn4/conn8 baselines, one interleaved round-robin per "
            "scene, all configs at one pinned grid"),
        "config": {"gpu_repeats": GPU_REPEATS, "njit_repeats": NJIT_REPEATS},
        "scenes": rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"neighbors_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, f"neighbors_{stamp}_scenes.csv")
    fieldnames = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
