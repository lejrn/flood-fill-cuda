"""CPU vs GPU on real drone frames, in one session.

The explainer video opens on a drone light show: video/assets/drones_mask
holds 240 binary masks (576 x 1024, one per camera frame at 30 fps, made
by video/assets/make_drone_frames.py from the downloaded footage). The
grand table (bench.py) never timed a drone frame; its CPU bars are on
test pictures. This script times the grand table's own cells on the drone
masks, in ONE session, next to two grand-table rows re-timed in the same
session (random_1000, png_blocks), so drones and many-blob test pictures
can be compared without mixing sessions:

- pure_python: bench.pure_python_bfs, one seed per blob handed in
  (exactly the grand table's pure_python cell on N-blob rows).
- njit: oracle5.cpu_fill_canonical, label + fill with the seeds found on
  the CPU (the grand table's @njit cell on N-blob rows).
- ch05_split_L8: the grand table's ch05 runner (kernel ms on the host
  clock around launch + sync, median of 5).
- ch06_rgb / ch06_mask: ch06's own protocol (CUDA events, 9 interleaved
  rounds over both contracts, after one clock spin-up per image size).

Fairness: all GPU cells run first, back to back after the spin-up, with
no CPU work in between (the GPU drops its clock when idle); then all CPU
cells, in the same process. The SM clock and the host CPU are recorded.

A mask becomes an image the way ch05_scenes.png_scene does it: white mask
pixels become pure red on white, transposed to img[x, y]. Each mask's
SHA-1 is recorded, since the masks themselves are gitignored.

Crosscheck per image: ch05 and ch06 must paint exactly the red pixel
count, ch06 must leave no red pixel and count the same blobs as the CPU
oracle, and pure Python must fill the red pixel count.

Output: results/overview/benchmark_results/drones_<stamp>.json. The
`drones_` prefix keeps it out of the `overview_*.json` globs.

Run:  PYTHONUNBUFFERED=1 uv run python -m flood_fill_cuda.overview.bench_drones [--step 1]
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import gc
import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone

import numpy as np
from numba import cuda

from .bench import RESULTS_DIR, ROWS, COLS, _cell_gpu, _cell_njit, _cell_pure, _component_seeds
from ..chapters.ch05_gpu_nblob_nblock.scenes import RED
from ..chapters.ch06_gpu_nblob_runs.recolor import RunRecolor, _warmup as ch06_warmup
from ..chapters.ch06_gpu_nblob_runs.benchmarks.benchmark import (
    _bench_ch06, _spin_up, _clocks, _host_run_count, ROUNDS, SPIN_SECONDS)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3))
MASK_DIR = os.path.join(REPO, "video", "assets", "drones_mask")
CALIBRATION = ("random_1000", "png_blocks")
CH05_KEY = "ch05_split_L8"


def mask_image(path):
    """A drone mask as a seedless scene: img[x, y], blobs pure red on white."""
    from PIL import Image
    m = np.array(Image.open(path).convert("L"))              # [y, x]
    img = np.full(m.shape + (3,), 255, dtype=np.uint8)
    img[m > 127] = RED
    return np.ascontiguousarray(np.transpose(img, (1, 0, 2)))


def red_count(img):
    return int(np.count_nonzero(
        (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)))


def sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def summary(values):
    v = sorted(values)
    return {"median_ms": statistics.median(v), "p90_ms": float(np.percentile(v, 90)),
            "max_ms": v[-1], "min_ms": v[0], "n": len(v)}


def gpu_cells(items):
    """ch05 and ch06 on every image, grouped by size, one spin-up per size."""
    runner = next(r for k, _, _, _, r in COLS if k == CH05_KEY)
    by_size = {}
    for it in items:
        by_size.setdefault(it["img"].shape[:2], []).append(it)
    for (w, h), group in by_size.items():
        capacity = max(8192, int(max(_host_run_count(it["img"]) for it in group) * 1.05))
        engine = RunRecolor(w, h, run_capacity=capacity)
        try:
            first = cuda.to_device(group[0]["img"])
            engine.pack(first)
            cuda.synchronize()
            _spin_up(engine, first)
            del first
            for it in group:
                img = it["img"]
                ctx = {"kind": "n", "img": img}
                it["cells"][CH05_KEY] = _cell_gpu(runner, ctx)
                pristine = cuda.to_device(img)
                dev = cuda.to_device(img)
                engine.pack(dev)
                cuda.synchronize()
                clocks = _clocks()
                stats = _bench_ch06(engine, dev, pristine)
                info = stats.pop("info")
                out = dev.copy_to_host()
                painted = int(np.count_nonzero(np.any(out != img, axis=2)))
                for c in ("rgb", "mask"):
                    s = stats[c]
                    it["cells"][f"ch06_{c}"] = {
                        "ms": s["median_ms"], "ms_min": s["min_ms"], "ms_max": s["max_ms"],
                        "label_only_ms": s["label_only_ms"], "filled": painted, "skip": None}
                it["ch06_blobs"] = int(info["n_blobs"])
                it["ch06_runs"] = int(info["n_runs"])
                it["ch06_still_red"] = red_count(out)
                it["clocks_during_ch06"] = clocks
                del pristine, dev, out
                print(f"  gpu {it['key']:>14s}: ch05 {it['cells'][CH05_KEY].get('ms', float('nan')):7.3f} ms | "
                      f"ch06 rgb {it['cells']['ch06_rgb']['ms']:.3f} mask {it['cells']['ch06_mask']['ms']:.3f} ms",
                      flush=True)
        finally:
            del engine
            gc.collect()
            try:
                cuda.current_context().deallocations.clear()
            except AttributeError:
                pass


def cpu_cells(items):
    for it in items:
        ctx = {"kind": "n", "img": it["img"], "seeds": it["seeds"]}
        it["cells"]["njit"] = _cell_njit(ctx)
        it["cells"]["pure_python"] = _cell_pure(ctx, it["red_px"])
        print(f"  cpu {it['key']:>14s}: njit {it['cells']['njit']['ms']:8.3f} ms | "
              f"pure {it['cells']['pure_python'].get('ms', float('nan')):9.1f} ms", flush=True)


def crosscheck(it):
    c = it["cells"]
    ok = (c[CH05_KEY].get("filled") == it["red_px"]
          and c["ch06_rgb"]["filled"] == it["red_px"]
          and it["ch06_still_red"] == 0
          and it["ch06_blobs"] == it["n_blobs"]
          and c["pure_python"].get("filled") == it["red_px"])
    return "OK" if ok else "MISMATCH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1, help="time every STEP-th drone frame")
    args = ap.parse_args()

    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    print(f"Device: {dev_name.strip()} ({int(device.MULTIPROCESSOR_COUNT)} SMs)")
    print(f"Idle clocks: {_clocks()}")

    items = []
    masks = sorted(f for f in os.listdir(MASK_DIR) if f.startswith("frame_") and f.endswith(".png"))
    for name in masks[::args.step]:
        path = os.path.join(MASK_DIR, name)
        img = mask_image(path)
        items.append({"key": name[:-4], "kind": "drone", "img": img, "sha1": sha1(path)})
    rows = {k: b for k, _f, _n, _e, b in ROWS}
    for key in CALIBRATION:
        if key in rows:
            items.append({"key": key, "kind": "calibration", "img": rows[key]()["img"]})
    for it in items:
        it["seeds"] = _component_seeds(it["img"])
        it["n_blobs"] = len(it["seeds"])
        it["red_px"] = red_count(it["img"])
        it["width"], it["height"] = int(it["img"].shape[0]), int(it["img"].shape[1])
        it["cells"] = {}
    print(f"{sum(it['kind'] == 'drone' for it in items)} drone frames, calibration rows: "
          f"{[it['key'] for it in items if it['kind'] == 'calibration']}")

    print("Warming up (ch05 runner, ch06 kernels, @njit oracle)...", flush=True)
    runner = next(r for k, _, _, _, r in COLS if k == CH05_KEY)
    runner({"kind": "n", "img": items[0]["img"]})
    ch06_warmup()
    _cell_njit({"kind": "n", "img": items[0]["img"]})

    print("GPU cells (one clock spin-up per image size, then back to back):", flush=True)
    gpu_cells(items)
    print("CPU cells:", flush=True)
    cpu_cells(items)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"drones_{stamp}.json")
    out_rows = []
    for it in items:
        row = {k: it[k] for k in ("key", "kind", "width", "height", "red_px", "n_blobs", "cells",
                                  "ch06_blobs", "ch06_runs", "ch06_still_red", "clocks_during_ch06")}
        if "sha1" in it:
            row["mask_sha1"] = it["sha1"]
        row["crosscheck"] = crosscheck(it)
        out_rows.append(row)
    drones = [r for r in out_rows if r["kind"] == "drone"]
    cols = ("pure_python", "njit", CH05_KEY, "ch06_rgb", "ch06_mask")
    payload = {
        "schema": "drones_cpu_vs_gpu",
        "device": dev_name.strip(),
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "host": {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count()},
        "idle_clocks": _clocks(),
        "spin_seconds": SPIN_SECONDS,
        "rounds_ch06": ROUNDS,
        "mask_dir": "video/assets/drones_mask",
        "step": args.step,
        "experiment": (
            "the grand table's cells (pure_python with seeds handed in, @njit label + fill, "
            "ch05 split_L8 kernel ms, ch06 rgb / mask CUDA events) on real drone masks and on "
            "two grand-table rows, all in one session; GPU cells first after one clock spin-up "
            "per image size, then CPU cells; per-frame medians, summarised over frames"),
        "drone_summary": {c: summary([r["cells"][c]["ms"] for r in drones if r["cells"][c].get("ms") is not None])
                          for c in cols},
        "drone_blobs": {"median": statistics.median([r["n_blobs"] for r in drones]),
                        "min": min(r["n_blobs"] for r in drones),
                        "max": max(r["n_blobs"] for r in drones)},
        "rows": out_rows,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    bad = [r["key"] for r in out_rows if r["crosscheck"] != "OK"]
    print(f"\nResults written to {path}")
    for c, s in payload["drone_summary"].items():
        print(f"  drones {c:>14s}: median {s['median_ms']:9.3f} ms, p90 {s['p90_ms']:9.3f}, max {s['max_ms']:9.3f}")
    for r in out_rows:
        if r["kind"] == "calibration":
            print(f"  {r['key']:>14s}: " + ", ".join(f"{c} {r['cells'][c].get('ms', float('nan')):.3f}" for c in cols))
    print("crosscheck:", "all OK" if not bad else f"MISMATCH on {bad}")


if __name__ == "__main__":
    main()
