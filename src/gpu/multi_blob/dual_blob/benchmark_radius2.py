"""Benchmark: the radius-2 experiment on two blobs, head-to-head.

Times {seq8, multi8, seq8r2, multi8r2} — the conn8 baselines and their
guarded radius-2 twins, both mechanisms — in ONE interleaved round-robin
per scene (Finding 3: sequential A-then-B timing on this drifting GPU
produces sign-flipping artifacts; only interleaving makes the ratios
trustworthy). All configs run at one pinned grid: the minimum
cooperative capacity across the lin8 and lin8r2 kernels at tpb=256.

Two questions, answered per scene:
1. does the ring-2 bet lose here the way it lost in multi_block
   (r2_multi_vs_conn8, r2_seq_vs_conn8 — >1 = the r2 twin faster)?
2. does the multisource-vs-sequential win survive the ring-2 tax
   (r2_speedup_multi_vs_seq vs conn8_speedup_multi_vs_seq)?

Scalars only are harvested from each result; the filled cross-check runs
against the 8-conn merged oracle (valid for r2: the guard provably
preserves the conn8 fill set).

Run:  uv run python src/gpu/multi_blob/dual_blob/benchmark_radius2.py
Writes JSON + CSV to benchmark_results/ next to this script.
Budget ~5-10 min.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import gc
import json
import statistics
from datetime import datetime, timezone

from flood_fill import flood_fill, max_blocks
from reference import cpu_flood_fill_two
import scenes
from numba import cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "benchmark_results")

GPU_REPEATS = 5
TPB = 256

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

CONFIGS = {
    "seq8": {"mode": "sequential", "connectivity": 8},
    "multi8": {"mode": "multisource", "connectivity": 8},
    "seq8r2": {"mode": "sequential", "connectivity": 8, "radius": 2},
    "multi8r2": {"mode": "multisource", "connectivity": 8, "radius": 2},
}


def _median(values):
    return statistics.median(values)


def _run_round_robin(img, seeds, blocks, repeats=GPU_REPEATS):
    """Every config timed once per round, order alternating per round.
    Scalars only; each result is dropped inside the loop."""
    names = list(CONFIGS)
    samples = {n: [] for n in names}
    meta = {}
    for i in range(repeats):
        order = names if i % 2 == 0 else list(reversed(names))
        for n in order:
            r = flood_fill(img, seeds, threads_per_block=TPB, blocks=blocks,
                           **CONFIGS[n])
            samples[n].append(r.kernel_ms)
            if n not in meta:  # deterministic per config — harvest once
                meta[n] = {"filled": r.filled, "levels": r.levels,
                           "interior": r.interior}
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


def bench_scene(name, builder, note, blocks):
    img, seeds = builder()
    width, height = img.shape[0], img.shape[1]

    stats, meta = _run_round_robin(img, seeds, blocks)
    ms = {n: stats[n]["ms"] for n in CONFIGS}

    row = {
        "scene": name, "note": note, "width": width, "height": height,
        "filled": meta["multi8"]["filled"],
        "blocks": blocks, "tpb": TPB,
    }
    for n in CONFIGS:
        row[f"{n}_ms"] = stats[n]["ms"]
        row[f"{n}_ms_min"] = stats[n]["ms_min"]
        row[f"{n}_ms_max"] = stats[n]["ms_max"]
        row[f"{n}_ms_stdev"] = stats[n]["ms_stdev"]
    row.update({
        # >1 = the radius-2 twin faster than its conn8 baseline
        "r2_multi_vs_conn8": ms["multi8"] / ms["multi8r2"],
        "r2_multi_vs_conn8_min": (stats["multi8"]["ms_min"]
                                  / stats["multi8r2"]["ms_min"]),
        "r2_seq_vs_conn8": ms["seq8"] / ms["seq8r2"],
        "r2_seq_vs_conn8_min": (stats["seq8"]["ms_min"]
                                / stats["seq8r2"]["ms_min"]),
        # does the multisource win survive the ring-2 tax?
        "conn8_speedup_multi_vs_seq": ms["seq8"] / ms["multi8"],
        "r2_speedup_multi_vs_seq": ms["seq8r2"] / ms["multi8r2"],
        "r2_speedup_multi_vs_seq_min": (stats["seq8r2"]["ms_min"]
                                        / stats["multi8r2"]["ms_min"]),
        "conn8_levels_multi": meta["multi8"]["levels"],
        "r2_levels_multi": meta["multi8r2"]["levels"],
        "r2_interior": meta["multi8r2"]["interior"],
        "r2_interior_pct": (100.0 * meta["multi8r2"]["interior"]
                            / meta["multi8r2"]["filled"]),
    })

    # 8-conn merged oracle: the filled cross-check for every config (the
    # guard provably preserves the conn8 fill set)
    _, _, _, _, oracle_filled, _, _ = cpu_flood_fill_two(img, seeds, 8)
    ok = (oracle_filled == meta["multi8"]["filled"]
          == meta["multi8r2"]["filled"] == meta["seq8"]["filled"]
          == meta["seq8r2"]["filled"])
    row["filled_crosscheck"] = "OK" if ok else "MISMATCH"

    check = "[OK]" if ok else "[MISMATCH!]"
    print(f"{name:16s} {row['filled']:>11,d} {ms['seq8']:8.2f} "
          f"{ms['multi8']:8.2f} {ms['seq8r2']:8.2f} {ms['multi8r2']:8.2f} "
          f"{row['r2_multi_vs_conn8']:6.2f}x "
          f"{row['r2_multi_vs_conn8_min']:6.2f}x "
          f"{row['r2_speedup_multi_vs_seq']:6.2f}x {check}")
    del img
    gc.collect()
    return row


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    sm_count = int(device.MULTIPROCESSOR_COUNT)
    print(f"Device: {dev_name.strip()} ({sm_count} SMs; tpb={TPB})")

    print("Warming up JITs (2 kernels + oracle)...")
    warm_img, warm_seeds = scenes.two_squares_scene(64, 64, 20, 20, gap=4)
    flood_fill(warm_img, warm_seeds, mode="multisource", connectivity=8)
    flood_fill(warm_img, warm_seeds, mode="multisource", connectivity=8,
               radius=2)
    cpu_flood_fill_two(warm_img, warm_seeds, 8)

    caps = {"lin8": max_blocks(threads_per_block=TPB, connectivity=8),
            "lin8r2": max_blocks(threads_per_block=TPB, connectivity=8,
                                 radius=2)}
    pin_blocks = min(caps.values())
    print(f"Cooperative capacity @tpb={TPB}: lin8={caps['lin8']}, "
          f"lin8r2={caps['lin8r2']} -> all configs pinned to {pin_blocks}")

    print(f"\n{'scene':16s} {'filled':>11s} {'seq8':>8s} {'multi8':>8s} "
          f"{'seq8r2':>8s} {'mu8r2':>8s} {'r2/8':>7s} {'(min)':>7s} "
          f"{'mu/seq':>7s}")
    rows = [bench_scene(name, builder, note, pin_blocks)
            for name, builder, note in SCENES]

    print("\nNotes: all GPU times are kernel-only medians over an "
          "INTERLEAVED round-robin (every config timed once per round, "
          "order alternating — Finding 3). 'r2/8' = multi8_ms/multi8r2_ms "
          "(>1 = the ring-2 twin faster), '(min)' the best-vs-best form; "
          "agreement between them is the signal a difference is real. "
          "'mu/seq' = the multisource speedup WITH ring-2 on both sides — "
          "compare against conn8_speedup_multi_vs_seq in the JSON to see "
          "whether the multisource win survives the ring-2 tax. All "
          "configs pinned to one grid.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": sm_count,
        "tpb": TPB,
        "coop_max_by_kernel": caps,
        "pinned_blocks": pin_blocks,
        "experiment": (
            "radius-2 (guarded ring-2, labels inherited) vs conn8, both "
            "mechanisms (sequential and multisource), one interleaved "
            "round-robin per scene at one pinned grid — the multi_block "
            "per-barrier experiment ported to two blobs"),
        "config": {"gpu_repeats": GPU_REPEATS},
        "scenes": rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"dual_blob_radius2_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR,
                            f"dual_blob_radius2_{stamp}_scenes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
