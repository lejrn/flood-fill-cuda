"""The tuning cross-product on external PNG inputs.

Same 53-config interleaved round-robin as tuning.py (v1, ccl, and
{fused, r128, split} x {plain lattice, interior} x strides 1..256),
but on images loaded from disk instead of the built-in scenes — the
first time the chapter's kernels meet inputs that were never designed
for them. png_scene snaps red-dominant pixels to pure RED; correctness
rides on the crosscheck field (all 53 configs must agree on the filled
pixel count and the blob count).

Default inputs: images/input/input_blobs.png and input_blocks.png at
the repo root (both gitignored — the benchmark skips whatever is
absent). Pass explicit paths as argv to run others.

Run:  PYTHONUNBUFFERED=1 uv run python -m flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.benchmarks.png_inputs [png ...]
Writes png_tuning_<stamp>.json to results/ch05_gpu_nblob_nblock/
benchmark_results/ (the png_ prefix keeps it out of the dashboard's
canonical tuning_*.json glob; its own card reads png_tuning_*.json).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import json
import sys
from datetime import datetime, timezone

from numba import cuda

from . import tuning
from .benchmark import TPB
from ..flood_fill import flood_fill
from .. import scenes as _scenes
from ....shared import results_paths

RESULTS_DIR = results_paths.results_dir("ch05_gpu_nblob_nblock",
                                        "benchmark_results")
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[".."] * 5))
DEFAULT_INPUTS = [os.path.join(_REPO_ROOT, "images", "input", f)
                  for f in ("input_blobs.png", "input_blocks.png")]


def main():
    candidates = [os.path.abspath(p) for p in sys.argv[1:]] or DEFAULT_INPUTS
    paths = []
    for p in candidates:
        if os.path.exists(p):
            paths.append(p)
        else:
            print(f"skipping missing {p}")
    if not paths:
        print("no inputs found")
        return 1

    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    print(f"Device: {dev_name.strip()} "
          f"({int(device.MULTIPROCESSOR_COUNT)} SMs; tpb={TPB})")

    print("Warming up JITs (v1, lat builds, ccl)...")
    warm_img, _ = _scenes.two_squares_scene(64, 64, 20, 20, gap=4)
    flood_fill(warm_img, variant="seed_merge")
    flood_fill(warm_img, variant="ccl_fill")
    for b in tuning.BUILDS:
        flood_fill(warm_img, variant="seed_merge", lattice=4, build=b)
        flood_fill(warm_img, variant="seed_merge", lattice=4, build=b,
                   interior=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"png_tuning_{stamp}.json")
    payload = {
        "device": dev_name.strip(),
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "tpb": TPB,
        "strides": list(tuning.STRIDES),
        "experiment": (
            "the tuning cross-product on external PNG inputs — red "
            "snapped to pure RED by png_scene, no ground-truth blob "
            "count, correctness carried by the cross-config crosscheck"),
        "scenes": [],
    }

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img, _ = _scenes.png_scene(path)
        note = f"external PNG, {img.shape[0]}x{img.shape[1]}"
        row = tuning.bench_scene(name, lambda img=img: (img, None), note)
        row["source"] = os.path.relpath(path, _REPO_ROOT)
        payload["scenes"].append(row)
        with open(json_path, "w") as f:      # crash-safe, per scene
            json.dump(payload, f, indent=2)

    print(f"\nResults written to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
