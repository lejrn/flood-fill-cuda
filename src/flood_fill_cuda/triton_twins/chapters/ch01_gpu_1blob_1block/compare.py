"""
Numba vs Triton on Chapter 1's own benchmark (single-block BFS flood fill).

The cases mirror chapters/ch01_gpu_1blob_1block/benchmarks/benchmark.py:

- scenes:        the benchmark's 11 scenes (the very SCENES list, imported),
                 both kernels ("ring" v1, "spill" v2) at 256 threads per block.
- ring_tripwire: the 4 scenes whose frontier overflows the 8192-slot ring.
                 The Numba benchmark records "trip" there; here both
                 backends must raise, with the same largest completed-level
                 occupancy. kernel_ms is the driver's own kernel bracket
                 (launch + synchronize, the kernel running until it aborts
                 after its first overflowing level), read from the driver's
                 perf_counter stamps; total_ms is the driver's bracket from
                 its first stamp to the RuntimeError.
- tpb_sweep:     the benchmark's threads-per-block sweep (64..1024) on
                 sq_2000_center with the default "ring" kernel.
- enqueue:       the v2 "spill" kernel's two enqueue forms on representative
                 scenes (one-pixel frontiers, a ring-sized square, the big
                 corner square, the spill scenes), two rows per scene:
                 enqueue="lane" (config label "per_lane", the default: one
                 atomic per winning lane, warp-aggregated by ptxas into
                 Numba's SASS idiom) and enqueue="program" (config label
                 "first_translation": the first translation's program-wide
                 tl.cumsum/tl.sum enqueue, about 9 extra CTA barriers per
                 direction). Numba runs its one v2 kernel in both rows.
                 The lane rows repeat the scenes experiment's spill rows;
                 the program rows measure what the first translation cost.
                 Program rows are comparable=False and carry
                 first_translation=true, as in ch02-ch04: they stay out of
                 the like-for-like averages and extremes of summary.py and
                 figures.py.

Both backends always run the same configuration: one block = one program,
the same threads per block (Triton num_warps = tpb // 32). The CPU
baselines of the Numba benchmark (@njit, pure Python) are not re-run: they
do not depend on the GPU backend.

same() compares every deterministic output: img, visited, depth,
level_sizes and every counter and derived percentage (cas_attempts,
spilled and peak_spill_window are schedule-free in this chapter; see the
twin's test_correctness.py). Only the *_ms timings differ.

Compare speedup_kernel, not speedup_total: total_ms adds allocation and
copies, which go through each stack's host library (CuPy's pool and
.set/.get vs Numba's fresh cuMemAlloc and copy_to_device). Each row's
phases_ms keeps those per backend: the warm-up sample and the medians of
the timed rounds.

Run (writes results/triton_twins/ch01_gpu_1blob_1block/compare_<UTC>.json):

    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare --quick

--quick is a smoke test (1 repeat, no JSON): three small scenes plus the
2600x2600 tripwire scene. That one is not tiny, but overflowing the
8192-slot ring needs a frontier above 8192 pixels, so it is the only quick
case that exercises the spill tier and the tripwire (about 3 s, 0.8 GB
host RAM). --only NAME[,NAME] keeps only the named scenes.
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import dataclasses
import gc
import re
import statistics
import sys
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
from .kernels import ENQ_MODES

CHAPTER = "ch01_gpu_1blob_1block"
TPB = 256                       # the benchmark's default for both variants
TPB_SWEEP = nb_bench.TPB_SWEEP  # [64, 128, 256, 512, 1024]
SWEEP_SCENE = "sq_2000_center"
DEFAULT_REPEATS = nb_bench.GPU_REPEATS  # 5 (the harness rounds odd counts up)

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
    # Not tiny: the smallest benchmark scene that overflows the ring, kept
    # because it is the only quick case for the spill tier and the tripwire.
    ("sq_2600_full_center", scenes.overflow_scene,
     "v1's tripwire scene; the spill tier completes it"),
]
QUICK_DIMS = {"serpentine_64": (64, 64), "disk_128": (128, 128)}
QUICK_TRIP_SCENES = {"sq_2600_full_center"}
QUICK_SWEEP = ("sq_256_center", [64, 1024])

# The enqueue experiment's scenes: one-pixel frontiers, a ring-sized wide
# frontier, the biggest scene that fits the ring, light and heavy spill.
ENQ_SCENES = ["serpentine_256", "sq_2000_center", "sq_4000_corner",
              "sq_2600_full_center", "sq_6000_center"]
QUICK_ENQ_SCENES = ["serpentine_64", "sq_2600_full_center"]
# The labels ch02-ch04 use; only the lane form is the twin's default.
ENQ_LABELS = {"lane": "per_lane", "program": "first_translation"}
DEFAULT_ENQ = "lane"
assert set(ENQ_LABELS) == set(ENQ_MODES) and ENQ_MODES[0] == DEFAULT_ENQ

TIMING_FIELDS = {"alloc_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms"}
HOST_PHASES = ("alloc_ms", "h2d_ms", "d2h_ms")

TRIP_TIMING = ("kernel_ms = the driver's kernel bracket (launch + "
               "synchronize) up to the overflow abort; total_ms = the "
               "driver's first stamp to the RuntimeError; d2h_ms = the "
               "counters copy and the raise")


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


class PhaseLog:
    """The host phases (alloc / H2D / D2H) of every call of one backend in
    one case.

    The harness calls each runner once to warm up, right after it releases
    both memory pools (so CuPy's allocation there is a cold cudaMalloc),
    then once per timed round. The warm-up sample and the medians of the
    timed rounds are kept apart. The harness copies Case.extra into the row
    shallowly, so ``out``, updated in place after every call, reaches the
    JSON with the final medians."""

    def __init__(self):
        self.samples = {p: [] for p in HOST_PHASES}
        self.calls = 0
        self.out = {}

    def record(self, res):
        values = {p: round(float(getattr(res, p)), 6) for p in HOST_PHASES}
        if self.calls == 0:
            self.out["warmup"] = values
        else:
            for p, v in values.items():
                self.samples[p].append(v)
            self.out["timed_median"] = {p: statistics.median(v)
                                        for p, v in self.samples.items()}
            self.out["timed_runs"] = self.calls
        self.calls += 1
        return res


class _StampClock:
    """Stands in for the ``time`` module inside a driver module and records
    every perf_counter() stamp the driver takes. Both drivers stamp once per
    bracket point (total start, H2D start, kernel start, D2H start, end), so
    on the overflow path, which raises after the counters copy and before
    the end stamp, the 4 stamps are the driver's own alloc / H2D / kernel
    brackets. No driver file is edited: the module attribute is swapped for
    the duration of one call."""

    def __init__(self):
        self.stamps = []

    def perf_counter(self):
        now = time.perf_counter()
        self.stamps.append(now)
        return now

    def __getattr__(self, name):
        return getattr(time, name)


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


def bar_syncs(ptx):
    """CTA barriers (bar.sync / barrier.sync) in a kernel's PTX."""
    return len(re.findall(r"\b(?:bar|barrier)\.sync\b", ptx))


def numba_bar_syncs(variant):
    kernel = (nb_kernels.single_block_bfs_kernel if variant == "ring"
              else nb_kernels.single_block_bfs_spill_kernel)
    counts = sorted({bar_syncs(p) for p in kernel.inspect_asm().values()})
    return counts[0] if len(counts) == 1 else counts


def result_info(variant, tpb, enqueue=None):
    def info(nb, tri):
        if enqueue is None:
            extra_nb, extra_tri = {}, {}
        else:
            ck = compiled_kernel(variant, tpb, enqueue)
            extra_nb = {"bar_sync_in_ptx": numba_bar_syncs(variant)}
            extra_tri = {"enqueue": enqueue,
                         "bar_sync_in_ptx": bar_syncs(ck.asm["ptx"])}
        return {
            "filled": nb.filled, "levels": nb.levels,
            "peak_level": nb.peak_level, "peak_occupancy": nb.peak_occupancy,
            "spilled": nb.spilled, "peak_spill_window": nb.peak_spill_window,
            "processed": nb.processed, "cas_attempts": nb.cas_attempts,
            "thread_util_pct": nb.thread_util_pct,
            "numba": {"grid": [1, tpb], **numba_resources(variant),
                      **extra_nb},
            "triton": {"grid": [1], "BLOCK": tpb, "num_warps": tpb // 32,
                       **kernel_resources(compiled_kernel(
                           variant, tpb, enqueue or "lane")),
                       **extra_tri},
        }
    return info


def trip_info(nb, tri):
    return {"tripped": True, "peak_occupancy": nb.peak_occupancy,
            "numba": {"grid": [1, TPB], **numba_resources("ring")},
            "triton": {"grid": [1], "BLOCK": TPB, "num_warps": TPB // 32,
                       **kernel_resources(compiled_kernel("ring", TPB))}}


def runner(ff, cache, name, variant, tpb, log, **kwargs):
    def run():
        img, sx, sy = cache.get(name)
        return log.record(ff(img, sx, sy, threads_per_block=tpb,
                             variant=variant, **kwargs))
    return run


def trip_runner(ff, cache, name, log):
    """A ring call up to its overflow RuntimeError, timed by the driver's own
    bracket stamps (see _StampClock)."""
    module = sys.modules[ff.__module__]

    def run():
        img, sx, sy = cache.get(name)
        clock = _StampClock()
        real_time = module.time
        module.time = clock
        try:
            ff(img, sx, sy, threads_per_block=TPB, variant="ring")
        except RuntimeError as exc:
            t_raise = time.perf_counter()
            message = str(exc)
        else:
            raise AssertionError(f"{name}: the ring kernel did not trip")
        finally:
            module.time = real_time
        s = clock.stamps
        if len(s) != 4:
            raise AssertionError(
                f"{name}: expected the driver's 4 bracket stamps before the "
                f"overflow raise, got {len(s)}")
        occ = int(re.search(r"occupancy: (\d+)", message).group(1))
        return log.record(SimpleNamespace(
            alloc_ms=(s[1] - s[0]) * 1000, h2d_ms=(s[2] - s[1]) * 1000,
            kernel_ms=(s[3] - s[2]) * 1000, d2h_ms=(t_raise - s[3]) * 1000,
            total_ms=(t_raise - s[0]) * 1000, tripped=True,
            peak_occupancy=occ))
    return run


def phase_logs():
    """One PhaseLog per backend, and the row's extra that publishes them."""
    logs = {"numba": PhaseLog(), "triton": PhaseLog()}
    return logs, {"phases_ms": {k: v.out for k, v in logs.items()}}


def build_cases(quick=False, only=None):
    table = QUICK_SCENES if quick else nb_bench.SCENES
    dims = {**SCENE_DIMS, **QUICK_DIMS}
    trips = QUICK_TRIP_SCENES if quick else RING_TRIP_SCENES
    sweep_scene, sweep_tpbs = QUICK_SWEEP if quick else (SWEEP_SCENE, TPB_SWEEP)
    enq_scenes = QUICK_ENQ_SCENES if quick else ENQ_SCENES
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
            logs, extra = phase_logs()
            cases.append(Case(
                experiment="ring_tripwire", scene=name,
                config={"variant": "ring", "threads_per_block": TPB},
                run_numba=trip_runner(numba_flood_fill, cache, name,
                                      logs["numba"]),
                run_triton=trip_runner(triton_flood_fill, cache, name,
                                       logs["triton"]),
                same=same_trip, pixels=w * h, info=trip_info,
                notes=note + ". v1 trips on both backends.",
                extra={"timing": TRIP_TIMING, **extra}))
        for variant in ("spill",) if name in trips else ("ring", "spill"):
            logs, extra = phase_logs()
            cases.append(Case(
                experiment="scenes", scene=name,
                config={"variant": variant, "threads_per_block": TPB},
                run_numba=runner(numba_flood_fill, cache, name, variant, TPB,
                                 logs["numba"]),
                run_triton=runner(triton_flood_fill, cache, name, variant,
                                  TPB, logs["triton"]),
                same=same_result, pixels=w * h,
                info=result_info(variant, TPB), notes=note, extra=extra))
        # Right after the scene's own cases, so the one-slot cache holds it.
        if name in enq_scenes:
            for enqueue in ENQ_MODES:
                logs, extra = phase_logs()
                label = ENQ_LABELS[enqueue]
                default = enqueue == DEFAULT_ENQ
                if not default:
                    extra["first_translation"] = True
                cases.append(Case(
                    experiment="enqueue", scene=name,
                    config={"variant": "spill", "threads_per_block": TPB,
                            "enqueue": enqueue, "label": label},
                    run_numba=runner(numba_flood_fill, cache, name, "spill",
                                     TPB, logs["numba"]),
                    run_triton=runner(triton_flood_fill, cache, name,
                                      "spill", TPB, logs["triton"],
                                      enqueue=enqueue),
                    same=same_result, pixels=w * h,
                    info=result_info("spill", TPB, enqueue),
                    notes=(f"{note}. v2 enqueue={enqueue!r} ({label}); "
                           f"Numba runs its warp-aggregated v2 kernel"
                           + ("" if default else
                              "; first translation, superseded by the "
                              "per-lane default: not in the like-for-like "
                              "averages")),
                    extra=extra, comparable=default))

    if keep(sweep_scene):
        w, h = dims[sweep_scene]
        for tpb in sweep_tpbs:
            logs, extra = phase_logs()
            cases.append(Case(
                experiment="tpb_sweep", scene=sweep_scene,
                config={"variant": "ring", "threads_per_block": tpb},
                run_numba=runner(numba_flood_fill, cache, sweep_scene, "ring",
                                 tpb, logs["numba"]),
                run_triton=runner(triton_flood_fill, cache, sweep_scene,
                                  "ring", tpb, logs["triton"]),
                same=same_result, pixels=w * h,
                info=result_info("ring", tpb),
                notes="the benchmark's threads-per-block sweep (ring kernel)",
                extra=extra))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: small scenes plus the tripwire scene, "
                         "1 repeat, no JSON")
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
        "enqueue": {"scenes": QUICK_ENQ_SCENES if args.quick
                    else ENQ_SCENES,
                    "variant": "spill", "labels": dict(ENQ_LABELS),
                    "default": DEFAULT_ENQ},
        "triton_enqueue_default": DEFAULT_ENQ,
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
            "v1 ticket atomics: one per winning lane in the source on both "
            "sides; ptxas warp-aggregates both (one ATOMS per warp in Numba, "
            "one ATOMG per warp in Triton), so the executed atomic counts "
            "match.",
            "v2 enqueue: Numba aggregates per warp by hand (activemask, "
            "popc, ffs, shfl_sync). The Triton default (enqueue=\"lane\") "
            "issues one atomic per winning lane per tier, which ptxas "
            "warp-aggregates into the same SASS idiom (VOTEU.ANY, POPC, one "
            "leader ATOMG, SHFL.IDX) with no CTA barrier: both v2 kernels "
            "carry Numba's 4 bar.sync. The scenes rows run this default.",
            "enqueue rows: two per scene. label \"per_lane\" "
            "(enqueue=\"lane\") is the default and repeats the scenes "
            "experiment's spill row. label \"first_translation\" "
            "(enqueue=\"program\") is the first translation, aggregated "
            "over the program with tl.cumsum/tl.sum (40 bar.sync in the "
            "kernel instead of 4); it is kept to measure that choice. "
            "Those rows are comparable=false and carry "
            "first_translation=true, so the like-for-like averages and "
            "extremes leave them out. info.*.bar_sync_in_ptx counts the "
            "CTA barriers.",
            "Use speedup_kernel for Numba vs Triton. total_ms and "
            "speedup_total add host-library costs that differ by stack: "
            "CuPy's pool and lighter .set/.get calls vs Numba's fresh "
            "cuMemAlloc and copy_to_device. On small scenes speedup_total "
            "exceeds speedup_kernel and can flip its direction (spill rows).",
            "phases_ms (per row, per backend): alloc/h2d/d2h of the warm-up "
            "call (cold: the harness releases both pools before each case) "
            "and the medians of the timed rounds.",
            "ring_tripwire rows: " + TRIP_TIMING + ". Taken from the "
            "drivers' own perf_counter stamps, so kernel_ms is the same "
            "bracket as in the other rows.",
        ],
    }
    if args.quick:
        meta["notes"].append(
            "--quick keeps the 2600x2600 sq_2600_full_center: overflowing "
            "the 8192-slot ring needs a frontier above 8192 pixels, so it "
            "is the only quick case for the spill tier and the tripwire.")
    doc = run_cases(CHAPTER, cases, repeats=repeats, meta=meta,
                    write=not (args.quick or args.no_write),
                    spin_seconds=0 if args.quick else 8.0)
    return doc


if __name__ == "__main__":
    main()
