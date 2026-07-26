"""Seeding-density sweep: how many seeds should discovery plant?

The hypothesis under test (raised while reading the phase timings): the
corner rule plants ONE seed on a solid rectangle, so the fill clock runs
O(blob diameter) barrier pairs, while the flatten costs ~2%. Densify the
seeds — corner rule PLUS every red pixel on an S x S lattice — and the
clock should collapse toward O(S); unions rise but each is one write,
and the compressed flatten stays tiny by construction.

Per scene, ONE interleaved round-robin (house methodology) over:

    v1        corner rule only (the published seed_merge)
    S0        v2 with the lattice off — isolates the compression pass
    S1        every red pixel a wave: levels=1, ALL connectivity through
              collisions (the ccl-like boundary, expected to lose)
    S4..S256  the density curve
    ccl       ccl_fill as the up-front reference

Run:  uv run python -m flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.benchmarks.seeding
Writes JSON + CSV to results/ch05_gpu_nblob_nblock/benchmark_results/.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import gc
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from . import bandwidth
from .benchmark import SCENES, TPB, _run_round_robin
from ..flood_fill import flood_fill, max_blocks
from .. import scenes as _scenes
from ....shared import results_paths
from numba import cuda

RESULTS_DIR = results_paths.results_dir("ch05_gpu_nblob_nblock",
                                        "benchmark_results")

STRIDES = (0, 1, 4, 16, 64, 256)


def _cfg_runner(img, lattice):
    if lattice == "v1":
        return lambda: _run(img, "seed_merge", None)
    if lattice == "ccl":
        return lambda: _run(img, "ccl_fill", None)
    return lambda: _run(img, "seed_merge", lattice)


def _run(img, variant, lattice):
    r = flood_fill(img, variant=variant, threads_per_block=TPB,
                   lattice=lattice)
    # Slim the result to scalars IMMEDIATELY: the round-robin keeps the
    # last result per config, and 8 configs x ~500 MB of host arrays on
    # the 16M px scenes is a host-OOM kill (measured: exit 137). Only
    # one full result may ever be alive.
    slim = SimpleNamespace(
        phase_ms=r.phase_ms, union_thread_ms=r.union_thread_ms,
        levels=r.levels, candidates=r.candidates,
        union_done=r.union_done, filled=r.filled, n_blobs=r.n_blobs)
    ms = r.kernel_ms
    del r
    return ms, slim


def bench_scene(name, builder, note):
    img = builder()[0]

    order = ["v1"] + ["S" + str(s) for s in STRIDES] + ["ccl"]
    runners = {}
    runners["v1"] = _cfg_runner(img, "v1")
    for s in STRIDES:
        runners["S" + str(s)] = _cfg_runner(img, s)
    runners["ccl"] = _cfg_runner(img, "ccl")

    runs = _run_round_robin(runners)

    row = {"scene": name, "note": note,
           "width": img.shape[0], "height": img.shape[1]}
    configs = {}
    for cfg in order:
        r, st = runs[cfg]
        configs[cfg] = {
            "ms": st["ms"], "ms_min": st["ms_min"], "ms_max": st["ms_max"],
            "ms_stdev": st["ms_stdev"],
            "phase_ms": st["phase_ms"],
            "union_thread_ms": st["union_thread_ms"],
            "levels": r.levels,
            "candidates": r.candidates,
            "unions": r.union_done,
            "filled": r.filled,
            "n_blobs": r.n_blobs,
        }
    row["configs"] = configs

    # derived: the best discovery-in-flight config (v1 or any stride)
    merge_cfgs = ["v1"] + ["S" + str(s) for s in STRIDES]
    best = min(merge_cfgs, key=lambda c: configs[c]["ms"])
    row.update({
        "best_cfg": best,
        "best_ms": configs[best]["ms"],
        "best_vs_v1": configs["v1"]["ms"] / configs[best]["ms"],
        "best_vs_ccl": configs["ccl"]["ms"] / configs[best]["ms"],
    })

    filled_set = {c["filled"] for c in configs.values()}
    blob_set = {c["n_blobs"] for c in configs.values()}
    row["crosscheck"] = ("OK" if len(filled_set) == 1 and len(blob_set) == 1
                         else "MISMATCH")

    print(f"\n{name}  ({note};  filled={configs['v1']['filled']:,d}, "
          f"blobs={configs['v1']['n_blobs']:,d})  "
          f"[{row['crosscheck']}]")
    print(f"  {'config':>6s} {'ms':>9s} {'(min)':>9s} {'levels':>8s} "
          f"{'cand':>10s} {'unions':>10s} {'fill ms':>8s} {'cmp+flat':>9s}")
    for cfg in order:
        c = configs[cfg]
        ph = c["phase_ms"]
        fill = ph.get("fill", 0.0)
        cleanup = ph.get("compress", 0.0) + ph.get("flatten", 0.0) \
            + ph.get("flatten_seed", 0.0)
        star = " *" if cfg == best else ""
        print(f"  {cfg:>6s} {c['ms']:9.2f} {c['ms_min']:9.2f} "
              f"{c['levels']:8,d} {c['candidates']:10,d} "
              f"{c['unions']:10,d} {fill:8.2f} {cleanup:9.2f}{star}")
    print(f"  best {best}: {row['best_vs_v1']:.2f}x vs v1, "
          f"{row['best_vs_ccl']:.2f}x vs ccl", flush=True)
    del runs
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass
    return row


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    print(f"Device: {dev_name.strip()} "
          f"({int(device.MULTIPROCESSOR_COUNT)} SMs; tpb={TPB})")

    print("Measuring D2D copy peak...")
    peak = bandwidth.measure_peak_bandwidth()
    print(f"  measured peak: {peak['gb_s']:.1f} GB/s")
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass

    print("Warming up JITs (v1 + lat + ccl kernels + oracle scenes)...")
    warm_img, _ = _scenes.two_squares_scene(64, 64, 20, 20, gap=4)
    flood_fill(warm_img, variant="seed_merge")
    flood_fill(warm_img, variant="seed_merge", lattice=4)
    flood_fill(warm_img, variant="ccl_fill")

    coop = {
        "v1": max_blocks(variant="seed_merge", threads_per_block=TPB),
        "lat": max_blocks(variant="seed_merge", threads_per_block=TPB,
                          lattice=0),
        "ccl": max_blocks(variant="ccl_fill", threads_per_block=TPB),
    }
    print(f"Cooperative capacity @tpb={TPB}: v1={coop['v1']}, "
          f"lat={coop['lat']}, ccl={coop['ccl']}")

    rows = [bench_scene(name, builder, note)
            for name, builder, note in SCENES]

    print("\nSweep summary (median ms; * = scene's best in-flight config):")
    hdr = ["scene"] + ["v1"] + ["S" + str(s) for s in STRIDES] + ["ccl"]
    print("  " + "".join(f"{h:>12s}" for h in hdr))
    for row in rows:
        cells = [f"{row['scene']:>12s}"]
        for cfg in hdr[1:]:
            c = row["configs"][cfg]
            mark = "*" if cfg == row["best_cfg"] else " "
            cells.append(f"{c['ms']:11.2f}{mark}")
        print("  " + "".join(cells))

    print("\nNotes: all times kernel-only medians over an interleaved "
          "round-robin. S0 differs from v1 only by the compression pass "
          "(and the modulo test); S1 seeds every red pixel — the "
          "ccl-like boundary; larger S traces the density curve. "
          "'fill ms' and 'cmp+flat' are in-kernel device phases; "
          "ccl's 'fill ms' excludes its union-merge prepass (see its "
          "phase columns in the JSON).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "tpb": TPB,
        "measured_peak_gb_s": peak["gb_s"],
        "strides": list(STRIDES),
        "coop_max": coop,
        "experiment": (
            "seeding-density sweep: corner-rule v1 vs lattice-seeded v2 "
            "(corner + every red pixel at x%S==0 and y%S==0, with a "
            "parent-compression pass before the flatten) vs ccl_fill. "
            "Canonical labels/seeds are stride-invariant; only the level "
            "clock, union volume and phase profile move."),
        "scenes": rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"seeding_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_rows = []
    for row in rows:
        for cfg, c in row["configs"].items():
            flat = {"scene": row["scene"], "config": cfg}
            for k, v in c.items():
                if k == "phase_ms":
                    for pk, pv in v.items():
                        flat["phase_" + pk + "_ms"] = pv
                else:
                    flat[k] = v
            csv_rows.append(flat)
    fieldnames = sorted({k for r in csv_rows for k in r},
                        key=lambda k: (k not in ("scene", "config"), k))
    csv_path = os.path.join(RESULTS_DIR, f"seeding_{stamp}_configs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
