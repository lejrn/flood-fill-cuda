"""Where are the 1 ms and 0.5 ms marks?

The chapter's target was "recolor `input_blobs.png` in 1 ms, or even
0.5 ms". On the full 9000x9000 image the answer is no, and the reason is
not the algorithm — it is arithmetic that no algorithm escapes:

    243 MB of RGB to read   / 167 GB/s measured  = 1.45 ms
    40.4 MB of red px to write / 169 GB/s        = 0.24 ms
    10.15 MB of packed mask to read              = 0.06 ms

So the RGB contract cannot go below ~1.7 ms at this size, and the
packed-mask contract's floor is the paint. The useful question is
therefore not "did we hit 1 ms" but **at what size do we hit it**, at
this image's own blob statistics — which is what this sweep measures.

Method: take CENTERED CROPS of the real image, not resized copies. A
crop preserves what actually drives the pipeline — run length (24.9 px
mean), red fraction (16.6%), blob size distribution — where a resize
would change all three and quietly answer a different question. Each
crop is measured with the same spun-up clock and interleaved rounds as
the main benchmark, and the crossings are reported by linear
interpolation between the bracketing crops.

Run:  uv run python -m flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks.scaling
Writes scaling_<stamp>.json to
results/ch06_gpu_nblob_runs/benchmark_results/.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import json
import statistics
import sys
from datetime import datetime, timezone

import numpy as np
from numba import cuda

from .benchmark import (_clocks, _spin_up, _measure_read_write_peaks,
                        _phase_times, _host_run_count, PNG_INPUTS)
from ..recolor import RunRecolor, CONTRACTS, _warmup
from ..kernels import N_RUNS, N_BLOBS, RUN_OVERFLOW
from ...ch05_gpu_nblob_nblock import scenes as _scenes
from ....shared import results_paths
from ....shared.bandwidth import measure_peak_bandwidth

RESULTS_DIR = results_paths.results_dir("ch06_gpu_nblob_runs",
                                        "benchmark_results")

SIDES = (1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000)
ROUNDS = 9
TARGETS = (1.0, 0.5)


def _crop(img, side):
    """Centered side x side crop — preserves run length, red fraction and
    blob size distribution, which a resize would not."""
    w, h = img.shape[0], img.shape[1]
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return np.ascontiguousarray(img[x0:x0 + side, y0:y0 + side])


def _measure(img):
    width, height = img.shape[0], img.shape[1]
    capacity = max(8192, int(_host_run_count(img) * 1.05))
    engine = RunRecolor(width, height, run_capacity=capacity)
    pristine = cuda.to_device(img)
    dev = cuda.to_device(img)
    engine.pack(dev)
    cuda.synchronize()
    _spin_up(engine, dev, seconds=4.0)

    acc = {c: {"total": [], "phases": []} for c in CONTRACTS}
    for r in range(ROUNDS):
        order = list(CONTRACTS) if r % 2 == 0 else list(reversed(CONTRACTS))
        for contract in order:
            dev.copy_to_device(pristine)
            if contract == "mask":
                engine.pack(dev)
            cuda.synchronize()
            names, events = engine.run(dev, contract=contract)
            cuda.synchronize()
            acc[contract]["total"].append(
                cuda.event_elapsed_time(events[0], events[-1]))
            acc[contract]["phases"].append(_phase_times(names, events))

    counters = engine.counters.copy_to_host()
    if counters[RUN_OVERFLOW]:
        raise RuntimeError("run table overflowed during scaling sweep")
    red_px = int(np.count_nonzero(
        (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)))
    out = {
        "side": width, "n_pixels": width * height, "red_px": red_px,
        "n_runs": int(counters[N_RUNS]), "n_blobs": int(counters[N_BLOBS]),
        "clocks": _clocks(),
    }
    for contract, a in acc.items():
        keys = a["phases"][0].keys()
        phase_med = {k: statistics.median(p[k] for p in a["phases"])
                     for k in keys}
        out[contract] = {
            "median_ms": statistics.median(a["total"]),
            "min_ms": min(a["total"]),
            "phase_ms": phase_med,
            "label_only_ms": sum(v for k, v in phase_med.items()
                                 if k != "paint"),
        }
    del engine, pristine, dev
    return out


def _crossings(rows, key, sub):
    """Where each target ms is crossed, by linear interpolation in
    megapixels between the bracketing crops."""
    pts = [(r["n_pixels"] / 1e6, r[key][sub] if sub != "label_only_ms"
            else r[key]["label_only_ms"]) for r in rows]
    pts.sort()
    out = {}
    for target in TARGETS:
        hit = None
        for (m0, t0), (m1, t1) in zip(pts, pts[1:]):
            if t0 <= target <= t1:
                frac = 0.0 if t1 == t0 else (target - t0) / (t1 - t0)
                hit = m0 + frac * (m1 - m0)
                break
        if hit is None and pts and pts[-1][1] <= target:
            hit = float("inf")          # never crossed within the sweep
        out[f"{target}ms_at_mpx"] = hit
    return out


def main():
    path = next((p for p in PNG_INPUTS
                 if p.endswith("input_blobs.png") and os.path.exists(p)), None)
    if path is None:
        print("images/input/input_blobs.png not found")
        return 1

    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    print(f"Device: {dev_name.strip()}")
    _warmup()
    peak = measure_peak_bandwidth()
    read_gb_s, write_gb_s = _measure_read_write_peaks(256 * 2 ** 20)
    print(f"Peaks: copy {peak['gb_s']:.0f}, read {read_gb_s:.0f}, "
          f"write {write_gb_s:.0f} GB/s")

    full, _ = _scenes.png_scene(path)
    rows = []
    for side in SIDES:
        if side > min(full.shape[0], full.shape[1]):
            continue
        row = _measure(_crop(full, side))
        rows.append(row)
        print(f"  {side:5d}² = {row['n_pixels']/1e6:5.1f} Mpx  "
              f"runs={row['n_runs']:>9,}  blobs={row['n_blobs']:>6,}  "
              f"rgb {row['rgb']['median_ms']:6.3f}  "
              f"mask {row['mask']['median_ms']:6.3f}  "
              f"label {row['mask']['label_only_ms']:6.3f} ms")

    payload = {
        "device": dev_name.strip(),
        "source": os.path.relpath(path, os.path.abspath(
            os.path.join(os.path.dirname(__file__), *[".."] * 5))),
        "method": ("centered crops of the real image (not resizes) — run "
                   "length, red fraction and blob size distribution are "
                   "preserved, which is what the pipeline's cost depends on"),
        "rounds": ROUNDS,
        "measured_peak_gb_s": peak["gb_s"],
        "measured_read_gb_s": read_gb_s,
        "measured_write_gb_s": write_gb_s,
        "targets_ms": list(TARGETS),
        "crossings": {
            "rgb": _crossings(rows, "rgb", "median_ms"),
            "mask": _crossings(rows, "mask", "median_ms"),
            "label_only": _crossings(rows, "mask", "label_only_ms"),
        },
        "rows": rows,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"scaling_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nTarget crossings (megapixels of this image's own statistics):")
    for tier, cross in payload["crossings"].items():
        parts = []
        for target in TARGETS:
            v = cross[f"{target}ms_at_mpx"]
            parts.append(f"{target} ms at "
                         + ("never in sweep" if v is None else
                            "always" if v == float("inf") else
                            f"{v:.1f} Mpx"))
        print(f"  {tier:11s} {'; '.join(parts)}")
    print(f"\nResults written to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
