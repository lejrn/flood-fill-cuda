"""
Numba vs Triton on the scan-based blob discovery prototype.

The experiment's only measurement is main_small_example's: the time of
process_image_small_example (img, visited and found_flag to the device, one
launch of scan_image_small_example[1, 100], visited and found_flag back) on
one 100x100 scene with a 20x20 red square. The cases time those same steps
on both backends:

- small_example: scan_image_small_example on the experiment's own scene
                 (setup_scene_small_example, seeds 0 and 1). The square
                 always covers a patch corner, so the scan stops after one
                 step: the time is launch and transfer overhead.
- early_stop:    the same kernel on scenes where the stop comes later: a
                 one-pixel spot found mid-scan, and an all-white image (no
                 red: every thread scans its whole patch).
- simple_scan:   simple_scan_kernel (defined, never launched by the Numba
                 host) on the seed-0 scene and on the white image.

Both backends run the experiment's configuration: one block of 100 threads
(Triton: one program of 128 lanes, 100 active, num_warps 4). kernel_ms is
the launch + synchronize; total_ms adds the transfers. The Numba host's
print inside process_image_small_example is left out on both sides.

The outputs are nondeterministic by construction (tick order, the race to
the first find), so same() compares only what is not: whether red was
found, the found pixel where only one thread can find it (spot) or nobody
does (white), and the painted coverage of full scans. The timings are
launch-bound (a 100x100 image, one block): record them, do not headline
them.

Run (writes results/triton_twins/scan_multi_blob/compare_<UTC>.json):

    .venv/bin/python -m flood_fill_cuda.triton_twins.experiments.scan_multi_blob.compare
    .venv/bin/python -m flood_fill_cuda.triton_twins.experiments.scan_multi_blob.compare --quick

--quick is a smoke test (one case per experiment, 1 repeat, no JSON).
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import time
from types import SimpleNamespace

import numpy as np

from flood_fill_cuda.experiments.scan_multi_blob import scan_only as numba_scan
from flood_fill_cuda.triton_twins.compare.harness import (
    Case, arrays_equal, run_cases,
)
from flood_fill_cuda.triton_twins.runtime import kernel_resources

from .scan_only import (
    BLOCK, NUM_WARPS, THREADS_PER_BLOCK, blank_scene, compiled_kernel,
    run_scan, setup_scene_small_example,
)

CHAPTER = "scan_multi_blob"
DEFAULT_REPEATS = 50
SIDE = 100


def spot_scene(x0, y0, size):
    """The white scene with a small red square inside one patch."""
    img, visited, width, height, found_flag = blank_scene()
    img[x0:x0 + size, y0:y0 + size] = (255, 0, 0)
    return img, visited, width, height, found_flag


SCENES = {
    "square20_seed0": lambda: setup_scene_small_example(0),
    "square20_seed1": lambda: setup_scene_small_example(1),
    "spot1_at_4_95": lambda: spot_scene(4, 95, 1),
    "white": blank_scene,
}


def numba_kernel(kernel):
    return (numba_scan.scan_image_small_example if kernel == "scan"
            else numba_scan.simple_scan_kernel)


def numba_run(kernel, scene):
    """process_image_small_example's steps on Numba (without its print),
    with run_scan's brackets."""
    from numba import cuda

    img, visited, width, height, found_flag = scene
    t0 = time.perf_counter()
    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(visited)
    d_found_flag = cuda.to_device(found_flag)
    cuda.synchronize()
    t1 = time.perf_counter()
    numba_kernel(kernel)[1, THREADS_PER_BLOCK](d_img, d_visited, width,
                                                height, d_found_flag)
    cuda.synchronize()
    t2 = time.perf_counter()
    result_visited = d_visited.copy_to_host()
    result_found = d_found_flag.copy_to_host()
    t3 = time.perf_counter()
    return SimpleNamespace(
        kernel=kernel, visited=result_visited, found_flag=result_found,
        h2d_ms=(t1 - t0) * 1000, kernel_ms=(t2 - t1) * 1000,
        d2h_ms=(t3 - t2) * 1000, total_ms=(t3 - t0) * 1000)


def same_for(kernel, scene_name, img):
    """The schedule-free outputs of this case."""
    reds = (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)
    full_scan = kernel == "simple" or scene_name == "white"
    single_finder = scene_name.startswith("spot") or scene_name == "white"

    def same(nb, tri):
        if int(nb.found_flag[0]) != int(tri.found_flag[0]):
            return False, (f"found: numba {nb.found_flag[0]} vs triton "
                           f"{tri.found_flag[0]}")
        for name, r in (("numba", nb), ("triton", tri)):
            if r.found_flag[0] and not reds[r.found_flag[1], r.found_flag[2]]:
                return False, f"{name}: found pixel {tuple(r.found_flag[1:])} is not red"
        pairs = {}
        if single_finder:
            pairs["found_flag"] = (nb.found_flag, tri.found_flag)
        if full_scan:
            pairs["painted"] = (nb.visited.any(axis=2), tri.visited.any(axis=2))
        return arrays_equal(**pairs)
    return same


def _one(values):
    values = values.values() if isinstance(values, dict) else [values]
    values = sorted(set(int(v) for v in values))
    return values[0] if len(values) == 1 else values


def numba_resources(kernel):
    k = numba_kernel(kernel)
    return {"regs_per_thread": _one(k.get_regs_per_thread()),
            "shared_bytes": _one(k.get_shared_mem_per_block()),
            "local_bytes_per_thread": _one(k.get_local_mem_per_thread())}


def info_for(kernel):
    def info(nb, tri):
        return {
            "found": int(nb.found_flag[0]),
            # schedule-dependent, recorded for reading only
            "painted_numba": int(nb.visited.any(axis=2).sum()),
            "painted_triton": int(tri.visited.any(axis=2).sum()),
            "ticks_triton": tri.clock,
            "numba": {"grid": [1, THREADS_PER_BLOCK],
                      **numba_resources(kernel),
                      "h2d_ms": nb.h2d_ms, "d2h_ms": nb.d2h_ms},
            "triton": {"grid": [1], "BLOCK": BLOCK, "active_lanes":
                       THREADS_PER_BLOCK, "num_warps": NUM_WARPS,
                       **kernel_resources(compiled_kernel(kernel)),
                       "h2d_ms": tri.h2d_ms, "d2h_ms": tri.d2h_ms},
        }
    return info


PLAN = [  # (experiment, kernel, scene, note)
    ("small_example", "scan", "square20_seed0",
     "main_small_example's scene: found at the first step"),
    ("small_example", "scan", "square20_seed1",
     "main_small_example's scene: found at the first step"),
    ("early_stop", "scan", "spot1_at_4_95",
     "one red pixel, found at its thread's 50th step"),
    ("early_stop", "scan", "white",
     "no red: every thread scans its whole patch (100 steps)"),
    ("simple_scan", "simple", "square20_seed0",
     "no early stop: 100 steps per thread"),
    ("simple_scan", "simple", "white", "no early stop, no red"),
]
QUICK_PLAN = [PLAN[0], PLAN[2], PLAN[4]]


def build_cases(quick=False):
    cases = []
    for experiment, kernel, scene_name, note in (QUICK_PLAN if quick else PLAN):
        scene = SCENES[scene_name]()
        cases.append(Case(
            experiment=experiment, scene=scene_name,
            config={"kernel": ("scan_image_small_example" if kernel == "scan"
                               else "simple_scan_kernel"),
                    "threads_per_block": THREADS_PER_BLOCK, "blocks": 1},
            run_numba=lambda k=kernel, s=scene: numba_run(k, s),
            run_triton=lambda k=kernel, s=scene: run_scan(*s, kernel=k),
            same=same_for(kernel, scene_name, scene[0]),
            pixels=SIDE * SIDE, info=info_for(kernel), notes=note))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: one case per experiment, 1 repeat, no JSON")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per case (default {DEFAULT_REPEATS})")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write the comparison JSON")
    args = ap.parse_args(argv)

    repeats = args.repeats or (1 if args.quick else DEFAULT_REPEATS)
    meta = {
        "source_benchmark": ("experiments/scan_multi_blob/scan_only.py "
                             "main_small_example (timeit around "
                             "process_image_small_example)"),
        "quick": args.quick,
        "caps": [],  # 100x100 scenes, nothing to cap
        "notes": [
            "Grid is 1 block of 100 threads (Numba [1, 100]) = 1 program of "
            "128 lanes, 100 active (Triton (1,), num_warps 4) on both sides.",
            "Numba's stop flag and clock are shared memory; the twin's are a "
            "global scratch allocated outside the timed brackets. One clock "
            "atomic per lane per step on both sides (shared-memory atomics "
            "in Numba, L2 atomics in Triton).",
            "Launch-bound: one block on a 100x100 image. The numbers show "
            "per-launch overhead more than kernel speed.",
            "Outputs are nondeterministic by construction; same() checks "
            "only the schedule-free parts (see compare.py).",
        ],
    }
    return run_cases(CHAPTER, build_cases(quick=args.quick), repeats=repeats,
                     meta=meta, write=not (args.quick or args.no_write),
                     spin_seconds=0 if args.quick else 8.0)


if __name__ == "__main__":
    main()
