"""Benchmark: runs vs pixels, head to head with Chapter 5.

The question this chapter was opened to answer is a specific one — what
does recoloring `images/input/input_blobs.png` (81 Mpx, 13.45M red px,
2,522 blobs) cost if it is bounded only by memory? — so the benchmark
is built around that image and reports FOUR numbers for it, because
"the runtime" is not one number:

  ch05 best        the previous chapter's winning config on this image
                   (seed_merge, lattice stride 8, split build)
  ch06 rgb         same contract as ch05: RGB in, recolored in place
  ch06 mask        the input is already a packed 1-bit mask
  ch06 label       the labeling alone — every blob discovered and given
                   its canonical label, nothing painted

...against three MEASURED machine limits, so every number can be read
against what the hardware can do at all:

  copy peak        saturating D2D copy (shared/bandwidth.py)
  read peak        streaming read of this very image
  write peak       streaming write of the same span

THE MEASUREMENT LESSON OF THIS CHAPTER, and the reason for `_spin_up`:
this laptop GPU idles at 1470 MHz of a 3105 MHz maximum and does not
raise its clock for short kernels separated by host syncs. Timed cold,
the same pipeline reads 2.27 ms; after a few seconds of back-to-back
launches it reads 1.37 ms and keeps falling. Every earlier chapter's
"the spread is 73% of the median" noise complaint has this underneath
it. Nothing here is timed until the clock has been driven up and
reported in the JSON, and the achieved clock is recorded alongside the
numbers so a future session can tell whether it is comparing like with
like.

Run:  uv run python -m flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks.benchmark
Writes runs_<stamp>.json (+ a flat CSV) to
results/ch06_gpu_nblob_runs/benchmark_results/. Budget ~3-6 min.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import gc
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from numba import cuda

from ..recolor import (RunRecolor, CONTRACTS, model_bytes_ch06, MODEL_NOTE,
                       PHASE_BLOCKS, PACK_ROW_BLOCKS, _warmup)
from ..kernels import N_RUNS, N_BLOBS, UNION_ATTEMPTS, RUN_OVERFLOW
from ...ch05_gpu_nblob_nblock import scenes as _scenes
from ....shared import results_paths
from ....shared.bandwidth import measure_peak_bandwidth

RESULTS_DIR = results_paths.results_dir("ch06_gpu_nblob_runs",
                                        "benchmark_results")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          *[".."] * 5))
PNG_INPUTS = [os.path.join(_REPO_ROOT, "images", "input", f)
              for f in ("input_blobs.png", "input_blocks.png")]

ROUNDS = 9              # per contract, interleaved
SPIN_SECONDS = 8.0      # clock ramp before anything is timed

# ch05's winning config on input_blobs.png, from that chapter's own
# png_tuning cross-product (53 configs/scene): the split build of the
# lattice-seeded seed_merge kernel at stride 8.
#
# CAVEAT, stated once and meant everywhere: this is ch05's best config
# ON THE HEADLINE IMAGE. It is carried unchanged to the other scenes,
# where ch05's own tuning would have picked a different stride (S1 for
# the serpentine, S1+interior for disks). So the non-PNG head-to-heads
# are "ch06 vs ch05-at-the-headline-config", not "ch06 vs ch05 at its
# per-scene best" — read them as indicative, and read ch05's own
# tuning table for that chapter's ceiling.
CH05_CFG = dict(variant="seed_merge", lattice=8, build="split")
CH05_LABEL = "split_L8"
CH05_ROUNDS = 3         # 52 ms a run; three is enough and honest


def _clocks():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,"
             "temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, mx, pw, tp = [x.strip() for x in out.split(",")]
        return {"sm_mhz": int(float(sm)), "max_sm_mhz": int(float(mx)),
                "power_w": float(pw), "temp_c": int(float(tp))}
    except Exception:
        return {}


def _spin_up(engine, dev, seconds=SPIN_SECONDS):
    """Drive the GPU into a boost clock before timing anything.

    Back-to-back launches with no host sync inside the inner loop — a
    sync per launch is exactly what keeps the clock down.
    """
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(50):
            engine.run(dev, contract="mask")
        cuda.synchronize()


def _measure_read_write_peaks(nbytes):
    """Streaming read and write peaks over `nbytes` — the two limits the
    pipeline's two ends are actually up against (a copy peak is neither)."""
    n = nbytes // 4
    buf = cuda.device_array(n, dtype=np.uint32)
    # One sink slot per thread, written UNCONDITIONALLY. Parking the
    # accumulator behind `if acc == <impossible>` lets the optimiser
    # delete the loads outright — the first version of this probe
    # reported 8,738 GB/s, which is the number a read that never
    # happened produces.
    out = cuda.device_array(2048 * 256, dtype=np.int64)

    @cuda.jit
    def _read(src, sink):
        acc = np.int64(0)
        for i in range(cuda.grid(1), src.shape[0], cuda.gridsize(1)):
            acc += src[i]
        sink[cuda.grid(1)] = acc

    @cuda.jit
    def _write(dst, v):
        for i in range(cuda.grid(1), dst.shape[0], cuda.gridsize(1)):
            dst[i] = v

    def _time(fn, args):
        fn[(2048, 256)](*args)
        cuda.synchronize()
        ts = []
        for _ in range(9):
            e0, e1 = cuda.event(timing=True), cuda.event(timing=True)
            e0.record()
            fn[(2048, 256)](*args)
            e1.record()
            e1.synchronize()
            ts.append(cuda.event_elapsed_time(e0, e1))
        return statistics.median(ts)

    r_ms = _time(_read, (buf, out))
    w_ms = _time(_write, (buf, np.uint32(7)))
    del buf, out
    return (nbytes / (r_ms * 1e6), nbytes / (w_ms * 1e6))


def _phase_times(names, events):
    return {n: cuda.event_elapsed_time(events[i], events[i + 1])
            for i, n in enumerate(names)}


def _bench_ch06(engine, dev, pristine, rounds=ROUNDS):
    """Interleaved rounds over both contracts, order reversed on odd
    rounds (ch04's timing rule). Returns per-contract stats."""
    acc = {c: {"total": [], "phases": []} for c in CONTRACTS}
    info = {}
    for r in range(rounds):
        order = list(CONTRACTS) if r % 2 == 0 else list(reversed(CONTRACTS))
        for contract in order:
            dev.copy_to_device(pristine)
            if contract == "mask":
                engine.pack(dev)          # build the packed input, off clock
            cuda.synchronize()
            names, events = engine.run(dev, contract=contract)
            cuda.synchronize()
            acc[contract]["total"].append(
                cuda.event_elapsed_time(events[0], events[-1]))
            acc[contract]["phases"].append(_phase_times(names, events))
            counters = engine.counters.copy_to_host()
            if counters[RUN_OVERFLOW]:
                raise RuntimeError("run table overflowed during benchmark")
            info = {"n_runs": int(counters[N_RUNS]),
                    "n_blobs": int(counters[N_BLOBS]),
                    "union_attempts": int(counters[UNION_ATTEMPTS])}

    out = {}
    for contract, a in acc.items():
        keys = a["phases"][0].keys()
        phase_med = {k: statistics.median(p[k] for p in a["phases"])
                     for k in keys}
        # the labeling tier: everything except the paint
        label_ms = sum(v for k, v in phase_med.items() if k != "paint")
        out[contract] = {
            "median_ms": statistics.median(a["total"]),
            "min_ms": min(a["total"]),
            "max_ms": max(a["total"]),
            "phase_ms": phase_med,
            "label_only_ms": label_ms,
        }
    out["info"] = info
    return out


def _bench_ch05(img, rounds=CH05_ROUNDS):
    """The previous chapter's best config on the same image, same
    process, same session — cross-imported exactly as ch05 cross-imported
    ch04. Returns None if it cannot run (e.g. VRAM)."""
    try:
        from ...ch05_gpu_nblob_nblock.flood_fill import flood_fill
        ts, blobs = [], None
        for _ in range(rounds):
            res = flood_fill(img, **CH05_CFG)
            ts.append(res.kernel_ms)
            blobs = res.n_blobs
            del res
            gc.collect()
        return {"config": CH05_LABEL, "median_ms": statistics.median(ts),
                "min_ms": min(ts), "n_blobs": blobs}
    except Exception as exc:                        # pragma: no cover
        return {"config": CH05_LABEL, "error": f"{type(exc).__name__}: {exc}"}


def _host_run_count(img):
    """The image's run count, by numpy, on the host. Used only to size
    the run table before timing — the default heuristic is one slot per
    16 px and sub-percolation noise blows straight through it (a 16 Mpx
    0.3-density scene has 3.4M runs against a 1M estimate). Sizing it
    here keeps the tripwire meaningful instead of routine."""
    red = ((img[..., 0] == 255) & (img[..., 1] == 0)
           & (img[..., 2] == 0)).astype(np.int8)
    return int((np.diff(red, axis=1, prepend=0) == 1).sum())


def _scene_row(name, img, note, run_ch05, peaks):
    width, height = img.shape[0], img.shape[1]
    n = width * height
    red_px = int(np.count_nonzero(
        (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)))
    capacity = max(8192, int(_host_run_count(img) * 1.05))

    ch05 = _bench_ch05(img) if run_ch05 else None
    gc.collect()

    engine = RunRecolor(width, height, run_capacity=capacity)
    pristine = cuda.to_device(img)
    dev = cuda.to_device(img)
    engine.pack(dev)
    cuda.synchronize()
    _spin_up(engine, dev)
    clocks = _clocks()
    stats = _bench_ch06(engine, dev, pristine)
    info = stats.pop("info")

    row = {
        "scene": name, "note": note, "width": width, "height": height,
        "n_pixels": n, "red_px": red_px,
        "n_runs": info["n_runs"], "n_blobs": info["n_blobs"],
        "union_attempts": info["union_attempts"],
        "mean_run_px": (red_px / info["n_runs"]) if info["n_runs"] else 0.0,
        "run_capacity": engine.run_capacity,
        "clocks_during_run": clocks,
        "ch06": {c: stats[c] for c in CONTRACTS},
        "ch05": ch05,
    }
    for c in CONTRACTS:
        mb = model_bytes_ch06(c, n, red_px, info["n_runs"],
                              info["union_attempts"])
        row["ch06"][c]["model_bytes"] = int(mb)
        row["ch06"][c]["model_gb_s"] = mb / (stats[c]["median_ms"] * 1e6)
    # the floor arithmetic: what the two ends alone cost at measured peak
    row["floor_ms"] = {
        "rgb_read": n * 3 / (peaks["read_gb_s"] * 1e6),
        "mask_read": ((n + 7) // 8) / (peaks["read_gb_s"] * 1e6),
        "paint_write": red_px * 3 / (peaks["write_gb_s"] * 1e6),
    }
    if ch05 and "median_ms" in ch05:
        for c in CONTRACTS:
            row["ch06"][c]["speedup_vs_ch05"] = (
                ch05["median_ms"] / stats[c]["median_ms"])

    del engine, pristine, dev
    gc.collect()
    return row


def _synthetic_scenes():
    return [
        ("blob_grid_100",
         lambda: _scenes.blob_grid_scene(4000, 4000, 10, 10, 360, gap=40)[0],
         "100 x 130k px solid squares — few, long runs"),
        ("random_4000",
         lambda: _scenes.random_blobs_scene(4000, 4000, 0.30, 0)[0],
         "sub-percolation noise — the run table's worst case"),
        ("disk_r2000",
         lambda: _scenes.disk_scene(4200, 4200, 2000)[0],
         "one 12.6M px disk — the geodesic monster that cost ch03 its levels"),
        ("serpentine_2048",
         lambda: _scenes.serpentine_scene(2048, 2048)[0],
         "one 2M px snake — 32,641 BFS levels for ch03, one merge pass here"),
    ]


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    dev_name = dev_name.strip()
    print(f"Device: {dev_name} ({int(device.MULTIPROCESSOR_COUNT)} SMs)")
    print(f"Idle clocks: {_clocks()}")

    print("Warming up JITs...")
    _warmup()

    peak = measure_peak_bandwidth()
    read_gb_s, write_gb_s = _measure_read_write_peaks(256 * 2 ** 20)
    print(f"Measured peaks: copy {peak['gb_s']:.0f} GB/s, "
          f"read {read_gb_s:.0f} GB/s, write {write_gb_s:.0f} GB/s")
    peaks = {"copy_gb_s": peak["gb_s"], "read_gb_s": read_gb_s,
             "write_gb_s": write_gb_s}

    payload = {
        "device": dev_name,
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "rounds": ROUNDS,
        "spin_seconds": SPIN_SECONDS,
        "idle_clocks": _clocks(),
        "measured_peak_gb_s": peak["gb_s"],
        "measured_read_gb_s": read_gb_s,
        "measured_write_gb_s": write_gb_s,
        "phase_blocks": PHASE_BLOCKS,
        "pack_row_blocks": PACK_ROW_BLOCKS,
        "ch05_config": CH05_LABEL,
        "bandwidth_model": MODEL_NOTE,
        "experiment": (
            "run-table connected components vs ch05's BFS/union-find over "
            "pixels, on the same images, with the RGB and packed-mask "
            "contracts reported separately and the measured read/write "
            "peaks alongside. Timed only after a clock spin-up."),
        "scenes": [],
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"runs_{stamp}.json")

    todo = []
    for path in PNG_INPUTS:
        if os.path.exists(path):
            name = os.path.splitext(os.path.basename(path))[0]
            img, _ = _scenes.png_scene(path)
            todo.append((name, img,
                         f"external PNG, {img.shape[0]}x{img.shape[1]}", True))
        else:
            print(f"skipping missing {path}")
    for name, build, note in _synthetic_scenes():
        todo.append((name, build(), note, True))

    for name, img, note, run_ch05 in todo:
        print(f"\n{name}: {img.shape[0]}x{img.shape[1]}"
              f"{' (+ ch05 head-to-head)' if run_ch05 else ''}")
        row = _scene_row(name, img, note, run_ch05, peaks)
        payload["scenes"].append(row)
        with open(json_path, "w") as f:            # crash-safe, per scene
            json.dump(payload, f, indent=2)
        c5 = row["ch05"]
        c5s = (f"{c5['median_ms']:.2f} ms" if c5 and "median_ms" in c5
               else "—")
        print(f"   runs={row['n_runs']:,} blobs={row['n_blobs']:,} "
              f"mean_run={row['mean_run_px']:.1f} px")
        print(f"   ch05 {c5s:>12s} | ch06 rgb "
              f"{row['ch06']['rgb']['median_ms']:.3f} ms | mask "
              f"{row['ch06']['mask']['median_ms']:.3f} ms | label "
              f"{row['ch06']['mask']['label_only_ms']:.3f} ms")
        if "speedup_vs_ch05" in row["ch06"]["rgb"]:
            print(f"   speedup vs ch05: rgb "
                  f"{row['ch06']['rgb']['speedup_vs_ch05']:.1f}x, mask "
                  f"{row['ch06']['mask']['speedup_vs_ch05']:.1f}x")

    csv_path = json_path.replace(".json", "_scenes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "width", "height", "n_pixels", "red_px",
                    "n_runs", "n_blobs", "mean_run_px", "ch05_ms",
                    "ch06_rgb_ms", "ch06_mask_ms", "ch06_label_ms",
                    "speedup_rgb", "speedup_mask"])
        for r in payload["scenes"]:
            c5 = r["ch05"] or {}
            w.writerow([
                r["scene"], r["width"], r["height"], r["n_pixels"],
                r["red_px"], r["n_runs"], r["n_blobs"],
                f"{r['mean_run_px']:.2f}",
                f"{c5.get('median_ms', float('nan')):.3f}",
                f"{r['ch06']['rgb']['median_ms']:.4f}",
                f"{r['ch06']['mask']['median_ms']:.4f}",
                f"{r['ch06']['mask']['label_only_ms']:.4f}",
                f"{r['ch06']['rgb'].get('speedup_vs_ch05', float('nan')):.2f}",
                f"{r['ch06']['mask'].get('speedup_vs_ch05', float('nan')):.2f}",
            ])

    print(f"\nResults written to {json_path}")
    print(f"                   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
