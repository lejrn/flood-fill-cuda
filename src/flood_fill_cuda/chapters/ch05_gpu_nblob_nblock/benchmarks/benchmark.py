"""Benchmark: two ways to discover your own seeds, head-to-head.

Session header: a MEASURED device-to-device copy peak (bandwidth.py), the
reference every derived bandwidth figure is expressed against.

Per scene (median of GPU_REPEATS, tpb=256, everything in ONE interleaved
round-robin — see _run_round_robin):

- merge        seed_merge: candidate scan + colliding waves + in-flight
               atomicMin unions (discovery rides inside the fill)
- ccl          ccl_fill: union-find CCL prepass, then one seed per blob
               feeds the ch03-style multisource fill
- *_bare       both uninstrumented twins (observer overhead)
- scan / cclp  the standalone discovery-phase kernels (seed_merge P0-P1 /
               ccl_fill P0-P2) on fresh buffers; fill ~= fused - phase is
               the attribution, reported with the subtraction caveat
- ch04_multi   (two-blob scenes only) ch04's multisource conn8 kernel
               with HOST-GIVEN seeds, cross-imported — the discovery tax:
               what "being told" costs vs "finding out". Caveat: a
               different module/signature, so it cannot share the ch05
               drivers' buffers; it IS interleaved in the same rounds.

The @njit CPU baseline is the honest discovery-included one:
cpu_fill_canonical = sequential CCL + canonical multisource fill.

Run:  uv run python -m flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.benchmarks.benchmark
Writes JSON + CSV to results/ch05_gpu_nblob_nblock/benchmark_results/
(centralized, not next to this script).
Budget ~15-25 min (dominated by 6 kernel compiles and the 16M px scenes).
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
from ..flood_fill import flood_fill, max_blocks, discovery_only
from ..cpu_oracle import cpu_fill_canonical
from ...ch04_gpu_2blob_nblock.flood_fill import flood_fill as ch04_flood_fill
from ...ch04_gpu_2blob_nblock import scenes as ch04_scenes
from .. import scenes
from ....shared import results_paths
from numba import cuda

RESULTS_DIR = results_paths.results_dir("ch05_gpu_nblob_nblock",
                                        "benchmark_results")

GPU_REPEATS = 5
NJIT_REPEATS = 3
TPB = 256

# Two-blob scenes come from the ch04 builders so their host-given seeds
# exist for the discovery-tax comparison; ch05-only scenes have no seeds
# to give (which is the whole point).
SCENES = [
    ("two_sq_2800",
     lambda: ch04_scenes.two_squares_scene(6000, 3200, 2800, 2800, gap=16),
     "2 x 7.8M px equal squares (ch04-comparable)"),
    ("two_disks_r1400",
     lambda: ch04_scenes.two_disks_scene(3000, 6000, 1400, gap=16),
     "2 x 6.2M px equal disks (ch04-comparable)"),
    ("asym_4000_800",
     lambda: ch04_scenes.asym_squares_scene(5200, 4400, 4000, 800, gap=16),
     "16M + 0.6M px (ch04-comparable)"),
    ("blob_grid_100",
     lambda: (scenes.blob_grid_scene(4000, 4000, 10, 10, 360, gap=40)[0],
              None),
     "100 x 130k px blobs — the N ch04's entry format could not hold"),
    ("random_4000",
     lambda: (scenes.random_blobs_scene(4000, 4000, density=0.3,
                                        rng_seed=0)[0], None),
     "sub-percolation noise: ~hundreds of thousands of tiny blobs"),
    ("comb_2000",
     lambda: (scenes.comb_scene(3000, 4001, teeth=2000, tooth_len=2400,
                                spine_w=8)[0], None),
     "ONE blob, 2000 candidates — the union-merge stress test"),
    ("serpentine_256",
     lambda: (scenes.serpentine_scene(256, 256)[0], None),
     "geodesic worst case: ~33k levels of barrier"),
]


def _median(values):
    return statistics.median(values)


def _run_round_robin(runners, repeats=GPU_REPEATS):
    """Time every runner INTERLEAVED, one round at a time, alternating the
    within-round order each round — the correctness requirement for every
    A/B ratio on this GPU (per-run drift up to 73% of the median produced
    sign-flipping artifacts under sequential A-then-B timing; see ch04).

    runners: {name: callable -> (kernel_ms, result_or_None)}. Returns
    {name: (last_result, stats_dict)}.
    """
    names = list(runners)
    samples = {n: {"k": [], "ph": [], "uth": []} for n in names}
    last = {}
    for i in range(repeats):
        order = names if i % 2 == 0 else list(reversed(names))
        for n in order:
            ms, r = runners[n]()
            samples[n]["k"].append(ms)
            ph = getattr(r, "phase_ms", None)
            if ph:
                samples[n]["ph"].append(dict(ph))
                samples[n]["uth"].append(r.union_thread_ms)
            last[n] = r
        gc.collect()
    out = {}
    for n in names:
        k = samples[n]["k"]
        ph_rounds = samples[n]["ph"]
        phase_med = ({key: _median([p[key] for p in ph_rounds])
                      for key in ph_rounds[0]} if ph_rounds else {})
        out[n] = (last[n], {
            "ms": _median(k),
            "ms_min": min(k),
            "ms_max": max(k),
            "ms_stdev": statistics.stdev(k) if len(k) > 1 else 0.0,
            "phase_ms": phase_med,
            "union_thread_ms": (_median(samples[n]["uth"])
                                if samples[n]["uth"] else 0.0),
        })
    return out


def bench_scene(name, builder, note, peak_gb_s):
    img, ch04_seeds = builder()
    width, height = img.shape[0], img.shape[1]

    runners = {
        "merge": lambda: _fused(img, "seed_merge", False),
        "ccl": lambda: _fused(img, "ccl_fill", False),
        "merge_bare": lambda: _fused(img, "seed_merge", True),
        "ccl_bare": lambda: _fused(img, "ccl_fill", True),
        "scan": lambda: discovery_only(img, "seed_merge",
                                       threads_per_block=TPB),
        "cclp": lambda: discovery_only(img, "ccl_fill",
                                       threads_per_block=TPB),
    }
    if ch04_seeds is not None:
        runners["ch04_multi"] = lambda: _ch04(img, ch04_seeds)

    runs = _run_round_robin(runners)

    rm, ms_merge = runs["merge"]
    rc, ms_ccl = runs["ccl"]
    merge_ms, ccl_ms = ms_merge["ms"], ms_ccl["ms"]
    scan_ms = runs["scan"][1]["ms"]
    cclp_ms = runs["cclp"][1]["ms"]
    merge_bare_ms = runs["merge_bare"][1]["ms"]
    ccl_bare_ms = runs["ccl_bare"][1]["ms"]

    row = {
        "scene": name, "note": note, "width": width, "height": height,
        "filled": rm.filled, "n_blobs": rm.n_blobs,
        "candidates": rm.candidates,
        "unions_merge": rm.union_done,
        "unions_ccl": rc.union_done,
        "levels_merge": rm.levels, "levels_ccl": rc.levels,
        "blocks_merge": rm.blocks, "blocks_ccl": rc.blocks,
        "merge_ms": merge_ms,
        "merge_ms_min": ms_merge["ms_min"],
        "merge_ms_max": ms_merge["ms_max"],
        "merge_ms_stdev": ms_merge["ms_stdev"],
        "ccl_ms": ccl_ms,
        "ccl_ms_min": ms_ccl["ms_min"],
        "ccl_ms_max": ms_ccl["ms_max"],
        "ccl_ms_stdev": ms_ccl["ms_stdev"],
        # >1 = seed_merge (discovery inside the fill) faster
        "merge_vs_ccl": ccl_ms / merge_ms,
        "merge_vs_ccl_min": ms_ccl["ms_min"] / ms_merge["ms_min"],
        "merge_mpx_s": rm.filled / merge_ms / 1000,
        "ccl_mpx_s": rc.filled / ccl_ms / 1000,
        # phase attribution (subtraction of separate launches — caveat)
        "scan_ms": scan_ms,
        "cclp_ms": cclp_ms,
        "merge_fill_est_ms": merge_ms - scan_ms,
        "ccl_fill_est_ms": ccl_ms - cclp_ms,
        # IN-KERNEL phase attribution: tid-0 %globaltimer stamps at the
        # phase barriers (medians over the same interleaved rounds) — no
        # separate-launch subtraction caveat, this is the fused kernel
        # timing itself
        "merge_init_dev_ms": ms_merge["phase_ms"].get("init"),
        "merge_scan_dev_ms": ms_merge["phase_ms"].get("scan"),
        "merge_fill_dev_ms": ms_merge["phase_ms"].get("fill"),
        "merge_flatten_dev_ms": ms_merge["phase_ms"].get("flatten"),
        # aggregate thread-time inside in-flight _union calls (clock64
        # cycle sum / base clock; concurrent, so NOT wall time)
        "merge_union_thread_ms": ms_merge["union_thread_ms"],
        "ccl_init_dev_ms": ms_ccl["phase_ms"].get("init"),
        "ccl_union_dev_ms": ms_ccl["phase_ms"].get("union_merge"),
        "ccl_flatten_seed_dev_ms": ms_ccl["phase_ms"].get("flatten_seed"),
        "ccl_fill_dev_ms": ms_ccl["phase_ms"].get("fill"),
        # observer overhead
        "merge_bare_ms": merge_bare_ms,
        "merge_overhead_pct": 100.0 * (merge_ms - merge_bare_ms)
                              / merge_bare_ms,
        "ccl_bare_ms": ccl_bare_ms,
        "ccl_overhead_pct": 100.0 * (ccl_ms - ccl_bare_ms) / ccl_bare_ms,
        # bandwidth model
        "merge_model_gb_s": rm.model_gb_s,
        "merge_pct_of_peak": 100.0 * rm.model_gb_s / peak_gb_s,
        "ccl_model_gb_s": rc.model_gb_s,
        "ccl_pct_of_peak": 100.0 * rc.model_gb_s / peak_gb_s,
        "merge_thread_util_pct": rm.thread_util_pct,
        "ccl_thread_util_pct": rc.thread_util_pct,
    }
    traces = {"level_sizes_merge": rm.level_sizes.tolist()}

    # -- discovery tax: what knowing the seeds was worth
    if ch04_seeds is not None:
        r4, ms4 = runs["ch04_multi"]
        ch04_ms = ms4["ms"]
        row.update({
            "ch04_multi_ms": ch04_ms,
            # >1 = discovery costs that much over being told the seeds
            "discovery_tax_merge": merge_ms / ch04_ms,
            "discovery_tax_ccl": ccl_ms / ch04_ms,
        })
        del r4
    filled_ref = rm.filled
    n_blobs_ref = rm.n_blobs
    ccl_filled = rc.filled
    ccl_n_blobs = rc.n_blobs
    del rm, rc, runs
    gc.collect()

    # -- CPU oracle: sequential CCL + canonical fill (discovery INCLUDED —
    #    the honest baseline for a find-your-own-seeds job)
    njit_times = []
    oracle = None
    for _ in range(NJIT_REPEATS):
        t0 = time.perf_counter()
        oracle = cpu_fill_canonical(img)
        njit_times.append((time.perf_counter() - t0) * 1000)
    njit_ms = _median(njit_times)
    oracle_filled = int(oracle[4])
    row["njit_ms"] = njit_ms
    row["speedup_merge_vs_njit"] = njit_ms / merge_ms
    row["speedup_ccl_vs_njit"] = njit_ms / ccl_ms
    ok = (filled_ref == oracle_filled == ccl_filled
          and n_blobs_ref == ccl_n_blobs)
    row["crosscheck"] = "OK" if ok else "MISMATCH"
    row.update(traces)
    del oracle
    gc.collect()

    check = "[OK]" if ok else "[MISMATCH!]"
    print(f"{name:16s} {row['filled']:>11,d} {row['n_blobs']:>7,d} "
          f"{row['candidates']:>7,d} {njit_ms:9.2f} {merge_ms:8.2f} "
          f"{ccl_ms:8.2f} {row['merge_vs_ccl']:6.2f}x "
          f"{row['merge_vs_ccl_min']:6.2f}x "
          f"{row.get('discovery_tax_merge', float('nan')):6.2f} {check}")
    return row


def _fused(img, variant, bare):
    r = flood_fill(img, variant=variant, threads_per_block=TPB, bare=bare)
    return r.kernel_ms, r


def _ch04(img, seeds):
    r = ch04_flood_fill(img, seeds, mode="multisource", connectivity=8,
                        threads_per_block=TPB)
    return r.kernel_ms, r


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

    print("Warming up JITs (6 ch05 kernels + ch04 sibling + oracle)...")
    warm_img, _ = scenes.two_squares_scene(64, 64, 20, 20, gap=4)
    for variant in ("seed_merge", "ccl_fill"):
        for bare in (False, True):
            flood_fill(warm_img, variant=variant, bare=bare)
        discovery_only(warm_img, variant)
    warm4_img, warm4_seeds = ch04_scenes.two_squares_scene(64, 64, 20, 20,
                                                           gap=4)
    ch04_flood_fill(warm4_img, warm4_seeds, mode="multisource",
                    connectivity=8)
    cpu_fill_canonical(warm_img)

    coop = {v: max_blocks(variant=v, threads_per_block=TPB)
            for v in ("seed_merge", "ccl_fill")}
    print(f"Cooperative capacity @tpb={TPB}: "
          f"merge={coop['seed_merge']}, ccl={coop['ccl_fill']}")

    print(f"\n{'scene':16s} {'filled':>11s} {'blobs':>7s} {'cand':>7s} "
          f"{'njit ms':>9s} {'merge':>8s} {'ccl':>8s} {'ccl/mg':>7s} "
          f"{'(min)':>7s} {'tax':>6s}")
    rows = [bench_scene(name, builder, note, peak_gb_s)
            for name, builder, note in SCENES]

    print(f"\nIn-kernel phase attribution (device %globaltimer stamps, "
          f"median ms; u-thr = aggregate thread-ms inside in-flight "
          f"unions, concurrent not wall):")
    print(f"{'scene':16s} | {'mg init':>8s} {'scan':>7s} {'fill':>9s} "
          f"{'flatten':>8s} {'u-thr':>8s} | {'ccl init':>8s} {'union':>9s} "
          f"{'flat+seed':>9s} {'fill':>9s}")
    for r in rows:
        print(f"{r['scene']:16s} | {r['merge_init_dev_ms']:8.2f} "
              f"{r['merge_scan_dev_ms']:7.2f} {r['merge_fill_dev_ms']:9.2f} "
              f"{r['merge_flatten_dev_ms']:8.2f} "
              f"{r['merge_union_thread_ms']:8.3f} | "
              f"{r['ccl_init_dev_ms']:8.2f} {r['ccl_union_dev_ms']:9.2f} "
              f"{r['ccl_flatten_seed_dev_ms']:9.2f} "
              f"{r['ccl_fill_dev_ms']:9.2f}")

    print("\nNotes: all GPU times are kernel-only medians over an "
          "INTERLEAVED round-robin (every config timed once per round, "
          "order alternating; this GPU's per-run spread demands it). "
          "'ccl/mg' = ccl_ms/merge_ms: >1 means seed_merge (discovery "
          "riding inside the fill) beat the CCL prepass; '(min)' is the "
          "best-vs-best form — agreement between them is the signal a "
          "difference is real. 'tax' = merge_ms / ch04-with-given-seeds: "
          "the price of finding out vs being told (two-blob scenes only). "
          "scan/cclp are the standalone discovery phases; fused - phase "
          "approximates fill time but subtracts across separate launches. "
          "GB/s figures are the DERIVED ch05 model (discovery + label_map "
          f"terms included) against the measured {peak_gb_s:.0f} GB/s "
          "copy peak.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": sm_count,
        "tpb": TPB,
        "measured_peak_gb_s": peak_gb_s,
        "bandwidth_model": bandwidth.MODEL_NOTE,
        "coop_max_merge": coop["seed_merge"],
        "coop_max_ccl": coop["ccl_fill"],
        "variants_experiment": (
            "seed_merge (candidate scan + colliding waves + in-flight "
            "atomicMin unions) vs ccl_fill (union-find CCL prepass + "
            "canonical-seed multisource fill), both discovering every "
            "seed on-GPU in one cooperative launch. ch04_multi rows are "
            "the given-seeds multisource conn8 kernel for the discovery "
            "tax; njit is sequential CCL + canonical fill (discovery "
            "included)"),
        "config": {"gpu_repeats": GPU_REPEATS, "njit_repeats": NJIT_REPEATS},
        "scenes": rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"seed_discovery_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    trace_keys = ("level_sizes_merge",)
    csv_rows = [{k: v for k, v in row.items() if k not in trace_keys}
                for row in rows]
    scenes_csv = os.path.join(RESULTS_DIR,
                              f"seed_discovery_{stamp}_scenes.csv")
    with open(scenes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults written to:\n  {json_path}\n  {scenes_csv}")


if __name__ == "__main__":
    main()
