"""Benchmark: three ways to flood two blobs, head-to-head.

Session header: a MEASURED device-to-device copy peak (bandwidth.py), the
reference every derived bandwidth figure is expressed against.

Per scene (median of GPU_REPEATS, tpb=256, instrumented lin kernels unless
stated):

- seq_full     two back-to-back launches at the full cooperative max each
               (the baseline: tA + tB, and tA/tB feed ideal_max)
- seq_half     the same two launches at HALF capacity each — the control
               for how much a smaller grid alone costs
- multi        one multisource launch, both seeds in one shared queue
               (the width bet: ~max(tA, tB))
- multi_xy     the same multisource launch with xy-format entries — the
               decode-tax experiment (kills the per-pixel div/mod at
               equal traffic)
- bare twins   multisource lin AND xy, pinned to the same grid, for
               observer overhead and the purest decode-tax figure
- mb_a         the SINGLE-blob multi_block kernel cross-imported and run
               on blob A alone — prices the label-packing arithmetic
               against the published baseline (packing tax)
- conn8        sequential + multisource re-run at 8-connectivity

mode="streams" is deliberately ABSENT from this benchmark. Two concurrent
cooperative grids wedge nondeterministically at grid.sync on this device
(a hang, not a slow run — it cost the test suite 82 minutes of dead spin),
and in every pair that did complete the overlap ratio came out at ~0.67 to
1.02: serialized, no benefit. Measuring it inline would risk the whole
session for a mechanism already known to lose. Its proper experiment is
the fresh-process, watchdog-guarded probe whose table lives in README.md.

The @njit oracle fills both blobs back-to-back on the CPU (the honest CPU
baseline for a two-blob job) and cross-checks every filled count.

Run:  uv run python src/gpu/multi_blob/dual_blob/benchmark.py
Writes JSON + CSV to benchmark_results/ next to this script.
Budget ~15-25 min (dominated by 6 kernel compiles and the 16M px scenes).
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

import bandwidth
from flood_fill import flood_fill, max_blocks
from reference import cpu_flood_fill, cpu_flood_fill_two, load_by_path
import scenes
from numba import cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "benchmark_results")
_MULTI_BLOCK = os.path.abspath(os.path.join(
    _HERE, os.pardir, os.pardir, "single_blob", "multi_block"))

GPU_REPEATS = 5
NJIT_REPEATS = 3
TPB = 256


def _load_multi_block():
    """Load the single-blob multi_block stage's flood_fill.py despite the
    module-name collision (same technique as multi_block itself uses for
    its siblings): temporarily remap 'kernels' and 'bandwidth' in
    sys.modules while executing it."""
    mb_kernels = load_by_path("_mb_kernels",
                              os.path.join(_MULTI_BLOCK, "kernels.py"))
    mb_bandwidth = load_by_path("_mb_bandwidth2",
                                os.path.join(_MULTI_BLOCK, "bandwidth.py"))
    saved = {}
    for name, mod in (("kernels", mb_kernels), ("bandwidth", mb_bandwidth)):
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    try:
        return load_by_path("_mb_flood_fill",
                            os.path.join(_MULTI_BLOCK, "flood_fill.py"))
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)


mb = _load_multi_block()

SCENES = [
    ("two_sq_300", lambda: scenes.two_squares_scene(700, 400, 300, 300,
                                                    gap=8),
     "small pair: launch/barrier overhead dominates"),
    ("two_sq_2800", lambda: scenes.two_squares_scene(6000, 3200, 2800, 2800,
                                                     gap=16),
     "2 x 7.8M px equal squares"),
    ("two_disks_r1400", lambda: scenes.two_disks_scene(3000, 6000, 1400,
                                                       gap=16),
     "2 x 6.2M px equal disks"),
    ("asym_4000_800", lambda: scenes.asym_squares_scene(5200, 4400, 4000,
                                                        800, gap=16),
     "16M + 0.6M px: the max-vs-sum story"),
]


def _median(values):
    return statistics.median(values)


def _run(img, seeds, repeats=GPU_REPEATS, **kw):
    """Median kernel_ms over repeats; returns (last_result, med_kernel_ms,
    med_a_ms, med_b_ms, med_overlap)."""
    ks, aks, bks, ovs = [], [], [], []
    r = None
    for _ in range(repeats):
        r = flood_fill(img, seeds, threads_per_block=TPB, **kw)
        ks.append(r.kernel_ms)
        aks.append(r.kernel_a_ms)
        bks.append(r.kernel_b_ms)
        ovs.append(r.overlap_ratio)
    return r, _median(ks), _median(aks), _median(bks), _median(ovs)


def bench_scene(name, builder, note, peak_gb_s):
    img, seeds = builder()
    width, height = img.shape[0], img.shape[1]
    coop = max_blocks(threads_per_block=TPB)

    # -- sequential, full capacity per launch (the baseline)
    r, seq_ms, seq_a, seq_b, _ = _run(img, seeds, mode="sequential")
    ideal_max = max(seq_a, seq_b)
    row = {
        "scene": name, "note": note, "width": width, "height": height,
        "filled": r.filled, "filled_a": r.filled_a, "filled_b": r.filled_b,
        "levels_a": r.levels_a, "levels_b": r.levels_b,
        "seq_ms": seq_ms, "seq_a_ms": seq_a, "seq_b_ms": seq_b,
        "seq_blocks": r.blocks,
        "seq_mpx_s": r.filled / seq_ms / 1000,
        "ideal_max_ms": ideal_max,
    }
    filled_ref = r.filled
    del r
    gc.collect()

    # -- sequential at half capacity (the streams control)
    _, seq_half_ms, _, _, _ = _run(img, seeds, mode="sequential",
                                   blocks=coop // 2)
    row["seq_half_ms"] = seq_half_ms
    row["seq_half_blocks"] = coop // 2

    # -- multisource, full capacity (the width bet)
    r, mu_ms, _, _, _ = _run(img, seeds, mode="multisource")
    row.update({
        "multi_ms": mu_ms,
        "multi_blocks": r.blocks,
        "levels_multi": r.levels,
        "peak_frontier": r.launches[0].peak_level,
        "multi_mpx_s": r.filled / mu_ms / 1000,
        "speedup_multi_vs_seq": seq_ms / mu_ms,
        "multi_vs_ideal": mu_ms / ideal_max,  # ~1.0 = the max(tA,tB) dream
        "multi_thread_util_pct": r.launches[0].thread_util_pct,
        "multi_model_bytes": r.model_bytes,
        "multi_model_gb_s": r.model_gb_s,
        "multi_pct_of_peak": 100.0 * r.model_gb_s / peak_gb_s,
    })
    multi_filled = r.filled
    multi_blocks = r.blocks
    traces = {"level_sizes_multi": r.launches[0].level_sizes.tolist()}
    del r
    gc.collect()

    # -- the decode-tax experiment: identical launch, xy entries
    _, xy_ms, _, _, _ = _run(img, seeds, mode="multisource",
                             entry_format="xy", blocks=multi_blocks)
    row["multi_xy_ms"] = xy_ms
    row["xy_vs_lin"] = mu_ms / xy_ms          # >1 = xy (no div/mod) faster

    # -- bare twins pinned to the same grid (observer overhead + the
    #    purest decode-tax figure)
    _, bare_ms, _, _, _ = _run(img, seeds, mode="multisource", bare=True,
                               blocks=multi_blocks)
    _, bare_xy_ms, _, _, _ = _run(img, seeds, mode="multisource", bare=True,
                                  entry_format="xy", blocks=multi_blocks)
    row["multi_bare_ms"] = bare_ms
    row["multi_overhead_pct"] = 100.0 * (mu_ms - bare_ms) / bare_ms
    row["multi_bare_xy_ms"] = bare_xy_ms
    row["xy_vs_lin_bare"] = bare_ms / bare_xy_ms

    # -- packing tax: the published single-blob kernel on blob A alone
    (ax, ay) = seeds[0]
    mb_times = []
    for _ in range(GPU_REPEATS):
        mb_r = mb.flood_fill(img, ax, ay, threads_per_block=TPB)
        mb_times.append(mb_r.kernel_ms)
    mb_a_ms = _median(mb_times)
    row["mb_a_ms"] = mb_a_ms
    row["packing_tax_pct"] = 100.0 * (seq_a - mb_a_ms) / mb_a_ms
    del mb_r
    gc.collect()

    # -- 8-connectivity: sequential + multisource
    _, seq8_ms, _, _, _ = _run(img, seeds, mode="sequential", connectivity=8)
    r8, mu8_ms, _, _, _ = _run(img, seeds, mode="multisource", connectivity=8)
    row.update({
        "conn8_seq_ms": seq8_ms,
        "conn8_multi_ms": mu8_ms,
        "conn8_levels_multi": r8.levels,
        "conn8_multi_vs_conn4": mu_ms / mu8_ms,   # >1 = 8-conn faster
        "conn8_speedup_multi_vs_seq": seq8_ms / mu8_ms,
    })
    conn8_filled = r8.filled
    del r8
    gc.collect()

    # -- CPU oracle: both blobs back-to-back (the honest two-blob CPU
    #    baseline), plus the filled cross-check across every path
    _, _, _, _, oracle_filled, _, _ = cpu_flood_fill_two(img, seeds)
    njit_times = []
    for _ in range(NJIT_REPEATS):
        t0 = time.perf_counter()
        cpu_flood_fill(img, seeds[0][0], seeds[0][1])
        cpu_flood_fill(img, seeds[1][0], seeds[1][1])
        njit_times.append((time.perf_counter() - t0) * 1000)
    njit_ms = _median(njit_times)
    row["njit_ms"] = njit_ms
    row["speedup_multi_vs_njit"] = njit_ms / mu_ms
    ok = filled_ref == oracle_filled == multi_filled == conn8_filled
    row["filled_crosscheck"] = "OK" if ok else "MISMATCH"
    row.update(traces)

    check = "[OK]" if ok else "[MISMATCH!]"
    print(f"{name:16s} {row['filled']:>11,d} {njit_ms:9.2f} {seq_ms:8.2f} "
          f"{seq_half_ms:8.2f} {mu_ms:8.2f} "
          f"{row['speedup_multi_vs_seq']:6.2f}x {row['multi_vs_ideal']:6.2f} "
          f"{row['xy_vs_lin']:6.2f}x {check}")
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

    print("Warming up JITs (5 kernels + sibling + oracle)...")
    warm_img, warm_seeds = scenes.two_squares_scene(64, 64, 20, 20, gap=4)
    for kw in ({}, {"bare": True}, {"entry_format": "xy"},
               {"entry_format": "xy", "bare": True}, {"connectivity": 8}):
        flood_fill(warm_img, warm_seeds, **{"mode": "multisource", **kw})
    mb.flood_fill(warm_img, *warm_seeds[0])
    cpu_flood_fill_two(warm_img, warm_seeds)

    coop = max_blocks(threads_per_block=TPB)
    coop_xy = max_blocks(threads_per_block=TPB, entry_format="xy")
    print(f"Cooperative capacity @tpb={TPB}: lin={coop}, xy={coop_xy}")

    print(f"\n{'scene':16s} {'filled':>11s} {'njit ms':>9s} {'seq':>8s} "
          f"{'seq/2':>8s} {'multi':>8s} {'mu/seq':>7s} {'/ideal':>6s} "
          f"{'xy':>7s}")
    rows = [bench_scene(name, builder, note, peak_gb_s)
            for name, builder, note in SCENES]

    print("\nNotes: all GPU times are kernel-only medians. '/ideal' = "
          "multi_ms / max(seq tA, tB): 1.0 is the perfect multisource "
          "outcome (one shared clock instead of two). 'xy' = lin_ms/xy_ms: "
          ">1 means the div/mod-free entry format is faster. GB/s figures "
          "are the DERIVED bytes-moved model against the measured "
          f"{peak_gb_s:.0f} GB/s copy peak. mode='streams' is excluded by "
          "design — see the module docstring and README.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": sm_count,
        "tpb": TPB,
        "measured_peak_gb_s": peak_gb_s,
        "bandwidth_model": bandwidth.MODEL_NOTE,
        "coop_max_lin": coop,
        "coop_max_xy": coop_xy,
        "modes_experiment": (
            "sequential vs multisource on two-blob scenes. mode='streams' "
            "excluded: two concurrent cooperative grids wedge "
            "nondeterministically at grid.sync on this device and showed "
            "overlap 0.67-1.02 (serialized) when they did complete — see "
            "the fresh-process probe table in README. Entry-format "
            "experiment: lin (div/mod decode) vs xy (bit-field decode) at "
            "identical traffic"),
        "config": {"gpu_repeats": GPU_REPEATS, "njit_repeats": NJIT_REPEATS},
        "scenes": rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"dual_blob_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    trace_keys = ("level_sizes_multi",)
    csv_rows = [{k: v for k, v in row.items() if k not in trace_keys}
                for row in rows]
    scenes_csv = os.path.join(RESULTS_DIR, f"dual_blob_{stamp}_scenes.csv")
    with open(scenes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults written to:\n  {json_path}\n  {scenes_csv}")


if __name__ == "__main__":
    main()
