"""Tuning cross-product: builds × seeding rules × strides, all at once.

Three questions raced in one session (user-requested full cross-product):

1. RULE — plain lattice (corner OR every S-th red pixel) vs interior
   lattice (lattice hits must have all 8 neighbors red: seeds inside the
   mass, off the blob edges — e.g. off the disks' staircase boundary).
2. BUILD — the register experiment. The fused lat kernel spends 129
   regs/thread, ONE over the 128 line that decides 2-blocks-per-SM, so
   it runs 24 cooperative blocks vs v1's 48. r128 caps the compiler at
   128 (lands at 122, spills for occupancy); split keeps only P0-P2
   cooperative (114 regs, like v1) and finishes with two plain
   full-occupancy kernels.
3. STRIDE — the fine curve {1,4,8,16,32,64,128,256}: locate the optimum
   the factor-4 sweep bracketed at S16, and test whether the S64 dip
   survives at S32/S128.

Per scene ONE interleaved round-robin over 53 configs: v1, ccl, and per
build {fused, r128, split}: S0 + {lattice, interior} × 8 strides.
Results are slimmed to scalars inside the runner (the OOM lesson) and
the JSON is REWRITTEN after every scene, so a crash loses nothing.

Run:  PYTHONUNBUFFERED=1 uv run python -m flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.benchmarks.tuning
Writes JSON + CSV to results/ch05_gpu_nblob_nblock/benchmark_results/.
Budget ~35-50 min (53 configs x 5 rounds x 7 scenes).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import gc
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from .benchmark import SCENES, TPB, _run_round_robin
from ..flood_fill import (flood_fill, max_blocks, regs_per_thread,
                          _KERNELS)
from .. import scenes as _scenes
from ....shared import results_paths
from numba import cuda

RESULTS_DIR = results_paths.results_dir("ch05_gpu_nblob_nblock",
                                        "benchmark_results")

STRIDES = (1, 4, 8, 16, 32, 64, 128, 256)
BUILDS = ("fused", "r128", "split")
RULES = ("L", "I")     # L = plain lattice, I = interior lattice


def _configs():
    """[(name, flood_fill kwargs)] — 53 per scene."""
    cfgs = [("v1", dict(variant="seed_merge")),
            ("ccl", dict(variant="ccl_fill"))]
    for b in BUILDS:
        cfgs.append((b + "_0", dict(variant="seed_merge", lattice=0,
                                    build=b)))
        for s in STRIDES:
            cfgs.append((f"{b}_L{s}",
                         dict(variant="seed_merge", lattice=s, build=b)))
            cfgs.append((f"{b}_I{s}",
                         dict(variant="seed_merge", lattice=s,
                              interior=True, build=b)))
    return cfgs


def _run(img, kwargs):
    r = flood_fill(img, threads_per_block=TPB, **kwargs)
    slim = SimpleNamespace(
        phase_ms=r.phase_ms, union_thread_ms=r.union_thread_ms,
        levels=r.levels, candidates=r.candidates,
        union_done=r.union_done, filled=r.filled, n_blobs=r.n_blobs)
    ms = r.kernel_ms
    del r
    return ms, slim


def bench_scene(name, builder, note):
    img = builder()[0]
    cfgs = _configs()
    runners = {n: (lambda kw=kw: _run(img, kw)) for n, kw in cfgs}
    runs = _run_round_robin(runners)

    configs = {}
    for cfg_name, _ in cfgs:
        r, st = runs[cfg_name]
        configs[cfg_name] = {
            "ms": st["ms"], "ms_min": st["ms_min"], "ms_max": st["ms_max"],
            "ms_stdev": st["ms_stdev"], "phase_ms": st["phase_ms"],
            "union_thread_ms": st["union_thread_ms"],
            "levels": r.levels, "candidates": r.candidates,
            "unions": r.union_done, "filled": r.filled,
            "n_blobs": r.n_blobs,
        }

    merge_names = [n for n, _ in cfgs if n != "ccl"]
    best = min(merge_names, key=lambda n: configs[n]["ms"])
    row = {
        "scene": name, "note": note,
        "width": img.shape[0], "height": img.shape[1],
        "configs": configs,
        "best_cfg": best,
        "best_ms": configs[best]["ms"],
        "best_vs_v1": configs["v1"]["ms"] / configs[best]["ms"],
        "best_vs_ccl": configs["ccl"]["ms"] / configs[best]["ms"],
        "best_per_build": {
            b: min((n for n in merge_names if n.startswith(b + "_")),
                   key=lambda n: configs[n]["ms"]) for b in BUILDS},
        "crosscheck": ("OK" if len({c["filled"] for c in configs.values()})
                       == 1 and len({c["n_blobs"]
                                     for c in configs.values()}) == 1
                       else "MISMATCH"),
    }

    # console: builds x rules matrix over the stride axis
    print(f"\n{name}  ({note};  filled={configs['v1']['filled']:,d}, "
          f"blobs={configs['v1']['n_blobs']:,d})  [{row['crosscheck']}]")
    print(f"  refs: v1={configs['v1']['ms']:.2f} ms  "
          f"ccl={configs['ccl']['ms']:.2f} ms")
    cols = ["S0"] + [f"S{s}" for s in STRIDES]
    print("  " + f"{'':10s}" + "".join(f"{c:>9s}" for c in cols))
    for b in BUILDS:
        for rule in RULES:
            cells = []
            for c in cols:
                if c == "S0":
                    key = b + "_0" if rule == "L" else None
                else:
                    key = f"{b}_{rule}{c[1:]}"
                if key is None:
                    cells.append(f"{'—':>9s}")
                else:
                    mark = "*" if key == best else " "
                    cells.append(f"{configs[key]['ms']:8.2f}{mark}")
            print(f"  {b + '/' + rule:10s}" + "".join(cells))
    print(f"  best {best}: {configs[best]['ms']:.2f} ms — "
          f"{row['best_vs_v1']:.2f}x vs v1, "
          f"{row['best_vs_ccl']:.2f}x vs ccl")

    del runs
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass
    return row


def _dump(payload, json_path):
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    print(f"Device: {dev_name.strip()} "
          f"({int(device.MULTIPROCESSOR_COUNT)} SMs; tpb={TPB})")

    print("Warming up JITs (v1, lat builds, ccl)...")
    warm_img, _ = _scenes.two_squares_scene(64, 64, 20, 20, gap=4)
    flood_fill(warm_img, variant="seed_merge")
    flood_fill(warm_img, variant="ccl_fill")
    for b in BUILDS:
        flood_fill(warm_img, variant="seed_merge", lattice=4, build=b)
        flood_fill(warm_img, variant="seed_merge", lattice=4, build=b,
                   interior=True)

    # the register story, measured per build
    builds_info = {}
    for key, kw in (("v1", dict()),
                    ("fused", dict(lattice=4, build="fused")),
                    ("r128", dict(lattice=4, build="r128")),
                    ("split", dict(lattice=4, build="split"))):
        kname = {"v1": "seed_merge", "fused": "seed_merge_lat",
                 "r128": "seed_merge_lat_r128",
                 "split": "seed_merge_lat_core"}[key]
        builds_info[key] = {
            "regs": regs_per_thread(_KERNELS[(kname, False)]),
            "coop_256": max_blocks(threads_per_block=256, **kw),
            "coop_128": max_blocks(threads_per_block=128, **kw),
        }
        print(f"  {key:6s}: {builds_info[key]['regs']} regs/thread, "
              f"coop blocks {builds_info[key]['coop_256']} @tpb256 / "
              f"{builds_info[key]['coop_128']} @tpb128")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"tuning_{stamp}.json")
    payload = {
        "device": dev_name.strip(),
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "tpb": TPB,
        "strides": list(STRIDES),
        "builds": builds_info,
        "experiment": (
            "full cross-product: builds {fused 24-block, r128 capped, "
            "split core+plain-cleanup} x rules {L plain lattice, "
            "I interior lattice} x strides {1..256} vs v1 and ccl_fill. "
            "Canonical labels invariant across every cell; only the "
            "clock, the union volume and the occupancy move."),
        "scenes": [],
    }
    _dump(payload, json_path)

    for name, builder, note in SCENES:
        payload["scenes"].append(bench_scene(name, builder, note))
        _dump(payload, json_path)   # crash-safe: rewritten per scene

    rows = payload["scenes"]
    print("\nSummary (best in-flight config per scene):")
    print(f"  {'scene':16s} {'best':>12s} {'ms':>9s} {'vs v1':>7s} "
          f"{'vs ccl':>7s}   per-build best")
    for r in rows:
        pb = "  ".join(f"{b}:{r['best_per_build'][b].split('_', 1)[1]}"
                       f"={r['configs'][r['best_per_build'][b]]['ms']:.1f}"
                       for b in BUILDS)
        print(f"  {r['scene']:16s} {r['best_cfg']:>12s} "
              f"{r['best_ms']:9.2f} {r['best_vs_v1']:6.2f}x "
              f"{r['best_vs_ccl']:6.2f}x   {pb}")

    csv_rows = []
    for r in rows:
        for cfg, c in r["configs"].items():
            flat = {"scene": r["scene"], "config": cfg}
            for k, v in c.items():
                if k == "phase_ms":
                    for pk, pv in v.items():
                        flat["phase_" + pk + "_ms"] = pv
                else:
                    flat[k] = v
            csv_rows.append(flat)
    fieldnames = sorted({k for cr in csv_rows for k in cr},
                        key=lambda k: (k not in ("scene", "config"), k))
    csv_path = os.path.join(RESULTS_DIR, f"tuning_{stamp}_configs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
