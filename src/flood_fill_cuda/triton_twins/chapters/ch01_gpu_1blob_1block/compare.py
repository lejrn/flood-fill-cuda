"""
Numba vs Triton on Chapter 1's own benchmark (single-block BFS flood fill).

The cases mirror chapters/ch01_gpu_1blob_1block/benchmarks/benchmark.py:

- scenes:        the benchmark's 11 scenes (the very SCENES list, imported),
                 both kernels ("ring" v1, "spill" v2) at 256 threads per block.
- ring_tripwire: the 4 scenes whose frontier overflows the 8192-slot ring.
                 The Numba benchmark records "trip" there; here both
                 backends must raise, with the same largest completed-level
                 occupancy. kernel_ms and total_ms are BOTH the wall time of
                 the whole call up to the raise (alloc + H2D + the kernel
                 running until its first overflowing level + counters D2H).
- tpb_sweep:     the benchmark's threads-per-block sweep (64..1024) on
                 sq_2000_center with the default "ring" kernel.

Both backends always run the same configuration: one block = one program,
the same threads per block (Triton num_warps = tpb // 32). The CPU
baselines of the Numba benchmark (@njit, pure Python) are not re-run: they
do not depend on the GPU backend.

same() compares every deterministic output: img, visited, depth,
level_sizes and every counter and derived percentage (cas_attempts,
spilled and peak_spill_window are schedule-free in this chapter; see the
twin's test_correctness.py). Only the *_ms timings differ.

Run (writes results/triton_twins/ch01_gpu_1blob_1block/compare_<UTC>.json):

    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare --quick

--quick is a smoke test (small scenes plus the tripwire scene, 1 repeat,
no JSON). --only NAME[,NAME] keeps only the named scenes.
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import dataclasses
import gc
import re
import time
from types import SimpleNamespace

import numpy as np

from flood_fill_cuda.chapters.ch01_gpu_1blob_1block import kernels as nb_kernels
from flood_fill_cuda.chapters.ch01_gpu_1blob_1block import scenes
from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks import (
    benchmark as nb_bench,
)
from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.flood_fill import (
    flood_fill as numba_flood_fill,
)
from flood_fill_cuda.triton_twins.compare.harness import Case, run_cases
from flood_fill_cuda.triton_twins.runtime import kernel_resources

from .flood_fill import compiled_kernel, flood_fill as triton_flood_fill

CHAPTER = "ch01_gpu_1blob_1block"
TPB = 256                       # the benchmark's default for both variants
TPB_SWEEP = nb_bench.TPB_SWEEP  # [64, 128, 256, 512, 1024]
SWEEP_SCENE = "sq_2000_center"
DEFAULT_REPEATS = nb_bench.GPU_REPEATS  # 5

# The benchmark's scenes beyond the ring's capacity (v1 trips there).
RING_TRIP_SCENES = {"sq_2600_full_center", "sq_4000_center",
                    "sq_5000_center", "sq_6000_center"}

# Image sizes of the benchmark scenes, for the rows' "pixels" field.
SCENE_DIMS = {
    "sq_256_center": (256, 256), "sq_512_center": (512, 512),
    "sq_1024_center": (1024, 1024), "sq_2000_center": (2000, 2000),
    "sq_4000_corner": (4000, 4000), "serpentine_256": (256, 256),
    "disk_1024": (1024, 1024), "sq_2600_full_center": (2600, 2600),
    "sq_4000_center": (4000, 4000), "sq_5000_center": (5000, 5000),
    "sq_6000_center": (6000, 6000),
}

QUICK_SCENES = [
    ("sq_256_center", lambda: scenes.square_scene(256, 256, 128, 128),
     "128^2 blob"),
    ("serpentine_64", lambda: scenes.serpentine_scene(64, 64),
     "small serpentine: one-pixel frontiers"),
    ("disk_128", lambda: scenes.disk_scene(128, 128, 60), "radius-60 disk"),
    ("sq_2600_full_center", scenes.overflow_scene,
     "v1's tripwire scene; the spill tier completes it"),
]
QUICK_DIMS = {"serpentine_64": (64, 64), "disk_128": (128, 128)}
QUICK_TRIP_SCENES = {"sq_2600_full_center"}
QUICK_SWEEP = ("sq_256_center", [64, 1024])

TIMING_FIELDS = {"alloc_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms"}


class SceneCache:
    """Builds one scene at a time: the ring and spill cases of a scene run
    back to back, so a one-slot cache builds each image once and never holds
    two big ones (sq_6000_center alone is 108 MB of input)."""

    def __init__(self, table):
        self.builders = {name: builder for name, builder, _ in table}
        self.name = None
        self.value = None

    def get(self, name):
        if name != self.name:
            self.name, self.value = None, None
            gc.collect()
            self.value = self.builders[name]()
            self.name = name
        return self.value


def same_result(nb, tri):
    """Every deterministic field equal; the timings are ignored."""
    for f in dataclasses.fields(nb):
        if f.name in TIMING_FIELDS:
            continue
        a, b = getattr(nb, f.name), getattr(tri, f.name)
        if isinstance(a, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False, f"{f.name}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
            if not np.array_equal(a, b):
                return False, f"{f.name}: {int(np.count_nonzero(a != b))} elements differ"
        elif a != b:
            return False, f"{f.name}: numba {a!r} vs triton {b!r}"
    return True, ""


def same_trip(nb, tri):
    if nb.peak_occupancy != tri.peak_occupancy:
        return False, (f"tripwire occupancy: numba {nb.peak_occupancy} vs "
                       f"triton {tri.peak_occupancy}")
    return True, ""


def numba_resources(variant):
    """Registers / static shared memory of the Numba kernel (one compile for
    every block size: Numba passes no launch bounds)."""
    kernel = (nb_kernels.single_block_bfs_kernel if variant == "ring"
              else nb_kernels.single_block_bfs_spill_kernel)

    def one(values):
        values = values.values() if isinstance(values, dict) else [values]
        values = sorted(set(int(v) for v in values))
        return values[0] if len(values) == 1 else values

    return {"regs_per_thread": one(kernel.get_regs_per_thread()),
            "shared_bytes": one(kernel.get_shared_mem_per_block()),
            "local_bytes_per_thread": one(kernel.get_local_mem_per_thread())}


def result_info(variant, tpb):
    def info(nb, tri):
        return {
            "filled": nb.filled, "levels": nb.levels,
            "peak_level": nb.peak_level, "peak_occupancy": nb.peak_occupancy,
            "spilled": nb.spilled, "peak_spill_window": nb.peak_spill_window,
            "processed": nb.processed, "cas_attempts": nb.cas_attempts,
            "thread_util_pct": nb.thread_util_pct,
            "numba": {"grid": [1, tpb], **numba_resources(variant),
                      "alloc_ms": nb.alloc_ms, "h2d_ms": nb.h2d_ms,
                      "d2h_ms": nb.d2h_ms},
            "triton": {"grid": [1], "BLOCK": tpb, "num_warps": tpb // 32,
                       **kernel_resources(compiled_kernel(variant, tpb)),
                       "alloc_ms": tri.alloc_ms, "h2d_ms": tri.h2d_ms,
                       "d2h_ms": tri.d2h_ms},
        }
    return info


def trip_info(nb, tri):
    return {"tripped": True, "peak_occupancy": nb.peak_occupancy,
            "numba": {"grid": [1, TPB], **numba_resources("ring")},
            "triton": {"grid": [1], "BLOCK": TPB, "num_warps": TPB // 32,
                       **kernel_resources(compiled_kernel("ring", TPB))}}


def runner(ff, cache, name, variant, tpb):
    def run():
        img, sx, sy = cache.get(name)
        return ff(img, sx, sy, threads_per_block=tpb, variant=variant)
    return run


def trip_runner(ff, cache, name):
    """Wall time of a ring call until its overflow RuntimeError."""
    def run():
        img, sx, sy = cache.get(name)
        t0 = time.perf_counter()
        try:
            ff(img, sx, sy, threads_per_block=TPB, variant="ring")
        except RuntimeError as exc:
            ms = (time.perf_counter() - t0) * 1000
            occ = int(re.search(r"occupancy: (\d+)", str(exc)).group(1))
            return SimpleNamespace(kernel_ms=ms, total_ms=ms, tripped=True,
                                   peak_occupancy=occ)
        raise AssertionError(f"{name}: the ring kernel did not trip")
    return run


def build_cases(quick=False, only=None):
    table = QUICK_SCENES if quick else nb_bench.SCENES
    dims = {**SCENE_DIMS, **QUICK_DIMS}
    trips = QUICK_TRIP_SCENES if quick else RING_TRIP_SCENES
    sweep_scene, sweep_tpbs = QUICK_SWEEP if quick else (SWEEP_SCENE, TPB_SWEEP)
    cache = SceneCache(table)
    cases = []

    def keep(name):
        return only is None or name in only

    for name, _, note in table:
        if not keep(name):
            continue
        w, h = dims[name]
        note = note.replace("\u2014", "-")
        if name in trips:
            cases.append(Case(
                experiment="ring_tripwire", scene=name,
                config={"variant": "ring", "threads_per_block": TPB},
                run_numba=trip_runner(numba_flood_fill, cache, name),
                run_triton=trip_runner(triton_flood_fill, cache, name),
                same=same_trip, pixels=w * h, info=trip_info,
                notes=(note + ". v1 trips: kernel_ms = total_ms = wall time "
                       "of the call up to the overflow RuntimeError.")))
        for variant in ("spill",) if name in trips else ("ring", "spill"):
            cases.append(Case(
                experiment="scenes", scene=name,
                config={"variant": variant, "threads_per_block": TPB},
                run_numba=runner(numba_flood_fill, cache, name, variant, TPB),
                run_triton=runner(triton_flood_fill, cache, name, variant, TPB),
                same=same_result, pixels=w * h,
                info=result_info(variant, TPB), notes=note))

    if keep(sweep_scene):
        w, h = dims[sweep_scene]
        for tpb in sweep_tpbs:
            cases.append(Case(
                experiment="tpb_sweep", scene=sweep_scene,
                config={"variant": "ring", "threads_per_block": tpb},
                run_numba=runner(numba_flood_fill, cache, sweep_scene, "ring", tpb),
                run_triton=runner(triton_flood_fill, cache, sweep_scene, "ring", tpb),
                same=same_result, pixels=w * h,
                info=result_info("ring", tpb),
                notes="the benchmark's threads-per-block sweep (ring kernel)"))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: small scenes, 1 repeat, no JSON")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per case (default {DEFAULT_REPEATS})")
    ap.add_argument("--only", default=None,
                    help="comma-separated scene names to keep")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write the comparison JSON")
    args = ap.parse_args(argv)

    only = set(args.only.split(",")) if args.only else None
    repeats = args.repeats or (1 if args.quick else DEFAULT_REPEATS)
    cases = build_cases(quick=args.quick, only=only)
    meta = {
        "source_benchmark": "chapters/ch01_gpu_1blob_1block/benchmarks/benchmark.py",
        "quick": args.quick,
        "threads_per_block": TPB,
        "tpb_sweep": {"scene": QUICK_SWEEP[0] if args.quick else SWEEP_SCENE,
                      "tpbs": QUICK_SWEEP[1] if args.quick else TPB_SWEEP,
                      "variant": "ring"},
        "ring_trip_scenes": sorted(QUICK_TRIP_SCENES if args.quick
                                   else RING_TRIP_SCENES),
        "only": sorted(only) if only else None,
        # Every benchmark scene and sweep point runs at full size; the
        # pure-Python and @njit CPU baselines are not part of this comparison.
        # The largest case (sq_6000_center: tripwire + spill, both backends,
        # two results alive during same()) peaked at 1.56 GB host RSS.
        "caps": [],
        "measured_peak_host_rss_gb_sq_6000_center": 1.56,
        "notes": [
            "Grid is 1 block (Numba [1, tpb]) = 1 program (Triton (1,), "
            "num_warps = tpb // 32) on both sides.",
            "The Triton ring (8192 int32) and its rear/overflow/spill-rear "
            "scalars live in global scratch (Triton has no user-addressable "
            "shared memory); Numba's live in shared memory.",
            "v1 ticket atomics: one per winning lane on both sides (Numba: "
            "shared-memory atomics; Triton: L2 atomics). v2: Numba aggregates "
            "per warp, Triton per program (one atomic per tier per chunk "
            "per direction).",
            "alloc_ms: CuPy's memory pool serves the Triton side's "
            "allocations after the first call; Numba allocates fresh.",
        ],
    }
    doc = run_cases(CHAPTER, cases, repeats=repeats, meta=meta,
                    write=not (args.quick or args.no_write),
                    spin_seconds=0 if args.quick else 8.0)
    return doc


if __name__ == "__main__":
    main()
