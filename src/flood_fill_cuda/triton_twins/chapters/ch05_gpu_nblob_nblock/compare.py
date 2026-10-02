"""Numba vs Triton for chapter 5, on the chapter's own benchmark experiments.

Five experiments, built from the Numba benchmarks' own scene lists,
strides and build names (imported, so they cannot drift). Every case runs
at tpb = benchmark.TPB = 256 (num_warps 8) on both backends.

benchmark              benchmarks/benchmark.py: every SCENES scene with merge,
                       ccl, merge_bare, ccl_bare (flood_fill) and scan, cclp
                       (discovery_only), each pinned to min(Numba capacity,
                       Triton capacity) of its own kernel: the same grid on
                       both backends.
benchmark_blocks_none  The Numba benchmark's literal launch, blocks=None, for
                       every flood_fill runner whose capacity differs between
                       the backends (merge's does not, so its pinned row is
                       also its blocks=None row). Both resolved grids are in
                       the row's info.
seeding                benchmarks/seeding.py: the fused lattice twin at every
                       seeding.STRIDES stride, pinned to min(capacities) of
                       the fused lattice kernel (Numba's own blocks=None
                       grid). Its v1 and ccl columns are the benchmark rows
                       merge and ccl.
tuning                 benchmarks/tuning.py, a 6-config subset of its 53 (see
                       caps): fused_L8, fused_I8 and, per experiment build
                       (r128, split), b_L1 and b_L8, blocks=None as
                       tuning.py launches them (the register experiment is
                       about each build's own capacity). Its v1 and ccl are
                       the benchmark rows.
png                    benchmarks/png_inputs.py: v1, ccl and the tuning subset,
                       blocks=None, on images/input/input_blocks.png (whole)
                       and input_blobs.png (cropped, see caps).

same() compares only what the algorithm fixes regardless of scheduling:
img, visited, depth, label, n_blobs, filled, levels, and for instrumented
runs the level trace, candidates, union_done, processed, peak_level,
peak_occupancy, plus cas_attempts (seed_merge family: paint is deferred)
or union_attempts (ccl_fill: one per red lex-predecessor pair); with a
pinned grid also blocks, thread_util_pct and processed_per_block. The
discovery probes compare their candidate count.

Each run's result is slimmed inside the runner to those arrays and
scalars (owner, prov_label, seeds are dropped), so the harness can hold a
Numba and a Triton result at once on the 23 M px scene within the
host-RAM budget. The per-run kernel_ms, in-kernel phase_ms and
union_thread_ms are kept in each row's "runs" (entry 0 is the warm-up).
A discovery probe's total_ms is the wall time of the discovery_only call.

Budget: 148 cases by default (benchmark 42, benchmark_blocks_none 18,
seeding 36, tuning 36, png 16). From single-run timings of every config
on the full-size scenes, about 16 minutes of GPU time at 5 repeats; peak
host RSS 2.03 GB measured on asym_4000_800 with both slim results alive.

Run:
    python -m flood_fill_cuda.triton_twins.chapters.ch05_gpu_nblob_nblock.compare [--quick] [--repeats N] [--experiments a,b]
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import gc
import time
import warnings
from types import SimpleNamespace

import numpy as np

from ....chapters.ch05_gpu_nblob_nblock import flood_fill as numba_ff
from ....chapters.ch05_gpu_nblob_nblock import kernels as numba_kernels
from ....chapters.ch05_gpu_nblob_nblock import scenes
from ....chapters.ch05_gpu_nblob_nblock.benchmarks import benchmark as nb_bench
from ....chapters.ch05_gpu_nblob_nblock.benchmarks import png_inputs as nb_png
from ....chapters.ch05_gpu_nblob_nblock.benchmarks import seeding as nb_seeding
from ....chapters.ch05_gpu_nblob_nblock.benchmarks import tuning as nb_tuning
from ....shared import bandwidth
from ...compare.harness import Case, arrays_equal, run_cases, spin_up
from ...runtime import kernel_resources
from ...runtime.bandwidth import measure_peak_bandwidth as triton_peak
from . import flood_fill as triton_ff

CHAPTER = "ch05_gpu_nblob_nblock"
DEFAULT_REPEATS = nb_bench.GPU_REPEATS  # 5, all three timing scripts
TPB = nb_bench.TPB
EXPERIMENTS = ("benchmark", "benchmark_blocks_none", "seeding", "tuning",
               "png")

# The benchmark's runners: name -> ("ff", flood_fill kwargs) or
# ("probe", discovery_only variant)
BENCH_RUNNERS = {
    "merge": ("ff", {"variant": "seed_merge"}),
    "ccl": ("ff", {"variant": "ccl_fill"}),
    "merge_bare": ("ff", {"variant": "seed_merge", "bare": True}),
    "ccl_bare": ("ff", {"variant": "ccl_fill", "bare": True}),
    "scan": ("probe", "seed_merge"),
    "cclp": ("probe", "ccl_fill"),
}

# tuning.py's 53 configs, by its own names; the subset this script runs.
# fused_L1 is left out: its Numba launch is seeding's S1 row (both run
# Numba's 24-block fused grid). The interior rule is a runtime int of the
# same kernel body in every build, so fused_I8 carries it.
TUNING_CONFIGS = dict(nb_tuning._configs())
TUNING_SUBSET = ["fused_L8", "fused_I8"] + [
    f"{b}_L{s}" for b in nb_tuning.BUILDS if b != "fused" for s in (1, 8)]
assert set(TUNING_SUBSET) <= set(TUNING_CONFIGS)
PNG_CONFIGS = ["v1", "ccl"] + TUNING_SUBSET

# ------------------------------------------------------------------- caps

# The largest scene (22.9 M px) runs in the benchmark experiment only
BENCHMARK_ONLY_SCENES = {"asym_4000_800"}
PNG_CROP = {"input_blobs.png": (4500, 4500)}
CAPS = [
    "tuning: 6 of tuning.py's 53 configs per scene (fused L8 and I8; r128 "
    "and split L1 and L8). The stride axis of the fused build is the "
    "seeding experiment (S0..S256), the interior rule rides on fused_I8, "
    "v1 and ccl are the benchmark rows. 53 configs x 7 scenes x 2 backends "
    "would take about 1.5 hours",
    "asym_4000_800 (22.9 M px, the largest scene; two solid squares like "
    "two_sq_2800) runs in benchmark only, not in benchmark_blocks_none, "
    "seeding or tuning: GPU-time budget",
    "benchmark_blocks_none runs the flood_fill runners only; the discovery "
    "probes (scan, cclp) are compared at a pinned grid in benchmark",
    "png: the same 6 tuning configs plus v1 and ccl (png_inputs.py runs all "
    "53)",
    "png: input_blobs.png (9000 x 9000, 81 M px) is cropped to its top-left "
    "4500 x 4500 quadrant (png columns and rows 0-4499; 655 blobs, 3.4 M "
    "red px): a full-size flood_fill result is ~1.7 GB of host arrays "
    "(~1 GB slimmed), so two kept results plus one in flight pass the "
    "2.5 GB host-RAM budget, and each backend needs ~2.4 GB of device "
    "buffers. input_blocks.png runs whole",
]
METHOD_NOTES = [
    "every experiment runs at the harness's repeats (default 5, the Numba "
    "scripts' GPU_REPEATS); the Numba scripts interleave all configs of a "
    "scene per round, the harness interleaves the two backends per case",
    "seeding and the benchmark pin each kernel to min(Numba, Triton "
    "capacity); tuning and png launch blocks=None like tuning.py and "
    "png_inputs.py, so the fused lattice build runs Numba's 24-block grid "
    "against Triton's own capacity (both in each row's info)",
    "results are slimmed inside the runners (owner, prov_label, seeds "
    "dropped); per-run phase_ms are in each row's runs",
]
SCOPE = (
    "benchmark.py also times ch04's given-seeds multisource kernel "
    "(ch04_multi, another unit's comparison) and the @njit CPU baseline "
    "(no GPU backend); neither is run here. wavefront.py (GIF renderer) and "
    "visualize.py (dashboard) have no timing to compare.")

# --------------------------------------------------------------- quick mode

QUICK_SCENES = [
    ("two_sq_96", lambda: (scenes.two_squares_scene(96, 64, 30, 30, gap=4)[0],
                           None), "tiny"),
    ("comb_48", lambda: (scenes.comb_scene(48, 176, teeth=80)[0], None),
     "tiny"),
    ("serpentine_32", lambda: (scenes.serpentine_scene(32, 32)[0], None),
     "tiny"),
]
QUICK_PNG_CROP = (128, 128)


class SceneSlot:
    """Holds one scene at a time: cases are grouped by scene, so a scene is
    built when its first case runs and dropped when the next one starts."""

    def __init__(self, builders):
        self.builders = builders
        self.name = None
        self.img = None
        self.shapes = {}

    def get(self, name):
        if name != self.name:
            self.release()
            self.img = np.ascontiguousarray(self.builders[name]()[0])
            self.name = name
            self.shapes[name] = self.img.shape[:2]
        return self.img

    def pixels(self, name):
        """width * height, building the scene once if it was never seen."""
        if name not in self.shapes:
            self.get(name)
        w, h = self.shapes[name]
        return int(w * h)

    def release(self):
        self.img = self.name = None
        gc.collect()


def _png_builder(path, crop):
    """scenes.png_scene on the whole PNG, or on its top-left crop[0] x
    crop[1] corner (img[x, y]: x = PNG column, y = PNG row). The crop is
    cut by PIL first and handed to png_scene as an in-memory BMP, so the
    full 81 M px image is never converted (about 1 GB less host RAM)."""
    def build():
        if crop is None:
            return scenes.png_scene(path)
        import io
        from PIL import Image
        with Image.open(path) as im:
            part = im.crop((0, 0, crop[0], crop[1]))
        buf = io.BytesIO()
        part.save(buf, format="BMP")
        del part
        buf.seek(0)
        return scenes.png_scene(buf)
    return build


# ---------------------------------------------------------- per-backend facts

def _one(value):
    """Numba returns {signature: value}; a kernel here has one signature."""
    if isinstance(value, dict):
        values = sorted({int(v) for v in value.values()})
        return values[0] if len(values) == 1 else values
    return int(value)


def _numba_resources(kernel_fn):
    return {"n_regs": numba_ff.regs_per_thread(kernel_fn),
            "local_bytes": _one(kernel_fn.get_local_mem_per_thread())}


_tiny = None


def _warm_probe(variant):
    """discovery_only compiles its phase kernel on first use (both)."""
    global _tiny
    if _tiny is None:
        _tiny = scenes.two_squares_scene(64, 64, 20, 20, gap=4)[0]
    numba_ff.discovery_only(_tiny, variant, threads_per_block=TPB)
    triton_ff.discovery_only(_tiny, variant, threads_per_block=TPB)


_facts = {}


def facts(kind, spec):
    """Capacities and compiled resources of one config on both backends
    (compiles both kernels if needed). Cached per config."""
    key = (kind, repr(spec))
    if key in _facts:
        return _facts[key]
    if kind == "probe":
        _warm_probe(spec)
        nk = numba_ff._PHASE_KERNELS[spec]
        tkey = ("phase", spec, TPB)
        caps = {"numba": numba_ff._coop_max_blocks(nk, TPB),
                "triton": triton_ff._coop_max_blocks(
                    tkey, triton_ff._compiled[tkey])}
        res = {"numba_resources": _numba_resources(nk),
               "triton_resources": kernel_resources(
                   triton_ff._compiled[tkey])}
    else:
        # the kernel does not depend on the interior rule (a runtime int)
        kw = {k: v for k, v in spec.items() if k != "interior"}
        caps = {"numba": numba_ff.max_blocks(threads_per_block=TPB, **kw),
                "triton": triton_ff.max_blocks(threads_per_block=TPB, **kw)}
        key_n = numba_ff._kernel_key(kw["variant"], kw.get("lattice"),
                                     kw.get("build", "fused"))
        nres = _numba_resources(
            numba_ff._KERNELS[(key_n, kw.get("bare", False))])
        if key_n == "seed_merge_lat_core":
            nres["cleanup"] = {
                "lat_compress": _numba_resources(
                    numba_kernels.lat_compress_kernel),
                "lat_finish": _numba_resources(numba_kernels.lat_finish_kernel)}
            nres["cleanup_grid"] = list(numba_ff._PLAIN_GRID)
        res = {"numba_resources": nres,
               "triton_resources": triton_ff.kernel_info(
                   threads_per_block=TPB, **kw)}
    _facts[key] = (caps, res)
    return caps, res


# ------------------------------------------------------- runners and checks

_SCALARS = ("variant", "lattice", "interior", "build", "bare", "blocks",
            "n_blobs", "filled", "levels", "candidates", "union_attempts",
            "union_done", "processed", "cas_attempts", "peak_level",
            "peak_occupancy", "level_trace_truncated", "thread_util_pct",
            "model_bytes", "model_gb_s", "union_thread_ms", "distinct_sms")


def _slim(r, log):
    """Keep what same() and info() read; drop the rest of the result."""
    log.append({"kernel_ms": float(r.kernel_ms),
                "phase_ms": dict(r.phase_ms),
                "union_thread_ms": float(r.union_thread_ms)})
    return SimpleNamespace(
        kernel_ms=float(r.kernel_ms), total_ms=float(r.total_ms),
        img=r.img, visited=r.visited.astype(np.bool_), depth=r.depth,
        label=r.label, level_sizes=r.level_sizes,
        processed_per_block=r.processed_per_block,
        **{f: getattr(r, f) for f in _SCALARS})


def _ff_runner(drv, slot, scene, kw, blocks, log):
    def run():
        img = slot.get(scene)
        r = drv.flood_fill(img, threads_per_block=TPB, blocks=blocks, **kw)
        s = _slim(r, log)
        del r
        return s
    return run


def _probe_runner(drv, slot, scene, variant, blocks, log):
    def run():
        img = slot.get(scene)
        t0 = time.perf_counter()
        ms, cand = drv.discovery_only(img, variant, threads_per_block=TPB,
                                      blocks=blocks)
        total = (time.perf_counter() - t0) * 1000
        log.append({"kernel_ms": float(ms)})
        return SimpleNamespace(kernel_ms=float(ms), total_ms=total,
                               candidates=int(cand))
    return run


def make_same(kw, pinned):
    """same(numba, triton): the deterministic outputs only."""
    bare = kw.get("bare", False)
    merge_family = kw["variant"] == "seed_merge"

    def same(n, t):
        ok, detail = arrays_equal(img=(n.img, t.img),
                                  visited=(n.visited, t.visited),
                                  depth=(n.depth, t.depth),
                                  label=(n.label, t.label))
        if not ok:
            return ok, detail
        fields = ["variant", "lattice", "interior", "build", "bare",
                  "n_blobs", "filled", "levels"]
        if not bare:
            pairs = {"level_sizes": (n.level_sizes, t.level_sizes)}
            if pinned:
                pairs["processed_per_block"] = (n.processed_per_block,
                                                t.processed_per_block)
            ok, detail = arrays_equal(**pairs)
            if not ok:
                return ok, detail
            fields += ["candidates", "union_done", "processed", "peak_level",
                       "peak_occupancy", "level_trace_truncated",
                       "cas_attempts" if merge_family else "union_attempts"]
            if pinned:
                fields.append("thread_util_pct")
        if pinned:
            fields.append("blocks")
        for f in fields:
            a, b = getattr(n, f), getattr(t, f)
            if a != b:
                return False, f"{f}: numba {a} vs triton {b}"
        return True, ""

    return same


def same_probe(n, t):
    if n.candidates != t.candidates:
        return False, f"candidates: numba {n.candidates} vs triton {t.candidates}"
    return True, ""


def ff_info(n, t):
    """Per-case facts from the warm-up results."""
    return {
        "numba_blocks": n.blocks, "triton_blocks": t.blocks,
        "filled": t.filled, "n_blobs": t.n_blobs, "levels": t.levels,
        "candidates": t.candidates, "union_done": t.union_done,
        "peak_level": t.peak_level, "peak_occupancy": t.peak_occupancy,
        "numba_union_attempts": n.union_attempts,
        "triton_union_attempts": t.union_attempts,
        "numba_cas_attempts": n.cas_attempts,
        "triton_cas_attempts": t.cas_attempts,
        "numba_model_bytes": n.model_bytes,
        "triton_model_bytes": t.model_bytes,
        "numba_thread_util_pct": n.thread_util_pct,
        "triton_thread_util_pct": t.thread_util_pct,
        "numba_distinct_sms": n.distinct_sms,
        "triton_distinct_sms": t.distinct_sms,
    }


def probe_info(n, t):
    return {"candidates": t.candidates}


def make_case(experiment, slot, scene, name, kind, spec, pinned, notes=""):
    """One cell: same scene, same kwargs, same tpb on both backends; the
    same explicit grid when pinned, else blocks=None on both."""
    caps, res = facts(kind, spec)
    blocks = min(caps.values()) if pinned else None
    runs = {"numba": [], "triton": []}
    if kind == "probe":
        rn = _probe_runner(numba_ff, slot, scene, spec, blocks, runs["numba"])
        rt = _probe_runner(triton_ff, slot, scene, spec, blocks,
                           runs["triton"])
        same, info = same_probe, probe_info
        config = {"runner": name, "probe": spec}
    else:
        rn = _ff_runner(numba_ff, slot, scene, spec, blocks, runs["numba"])
        rt = _ff_runner(triton_ff, slot, scene, spec, blocks, runs["triton"])
        same, info = make_same(spec, pinned), ff_info
        config = {"runner": name, **spec}
    config.update({"tpb": TPB, "num_warps": TPB // 32,
                   "blocks": "None" if blocks is None else int(blocks)})
    extra = {"caps": caps, **res, "runs": runs}
    return Case(experiment=experiment, scene=scene, config=config,
                run_numba=rn, run_triton=rt, same=same,
                pixels=slot.pixels(scene), info=info, notes=notes,
                extra=extra)


# -------------------------------------------------------------- experiments

def _tuning_spec(cfg_name):
    if cfg_name == "v1":
        return {"variant": "seed_merge"}
    if cfg_name == "ccl":
        return {"variant": "ccl_fill"}
    kw = dict(TUNING_CONFIGS[cfg_name])
    if kw.get("build") == "fused":
        kw.pop("build")  # the driver default, as seeding.py passes it
    return kw


def benchmark_cases(slot, scene_list):
    return [make_case("benchmark", slot, sname, name, kind, spec, True,
                      notes=note)
            for sname, _, note in scene_list
            for name, (kind, spec) in BENCH_RUNNERS.items()]


def blocks_none_cases(slot, scene_list):
    differ = [name for name, (kind, spec) in BENCH_RUNNERS.items()
              if kind == "ff" and len(set(facts(kind, spec)[0].values())) > 1]
    cases = [make_case("benchmark_blocks_none", slot, sname, name,
                       *BENCH_RUNNERS[name], False, notes=note)
             for sname, _, note in scene_list for name in differ]
    return cases, differ


def seeding_cases(slot, scene_list):
    return [make_case("seeding", slot, sname, f"S{s}", "ff",
                      {"variant": "seed_merge", "lattice": s}, True,
                      notes=note)
            for sname, _, note in scene_list for s in nb_seeding.STRIDES]


def tuning_cases(slot, scene_list, experiment, cfg_names):
    return [make_case(experiment, slot, sname, cfg, "ff", _tuning_spec(cfg),
                      False, notes=note)
            for sname, _, note in scene_list for cfg in cfg_names]


def png_scenes(quick):
    found, missing = [], []
    for path in nb_png.DEFAULT_INPUTS:
        base = os.path.basename(path)
        if not os.path.exists(path):
            missing.append(path)
            continue
        crop = QUICK_PNG_CROP if quick else PNG_CROP.get(base)
        stem = os.path.splitext(base)[0]
        name = stem if crop is None else f"{stem}_crop{crop[0]}x{crop[1]}"
        note = (f"external PNG {base}"
                + ("" if crop is None else f", top-left {crop[0]}x{crop[1]}"))
        found.append((name, _png_builder(path, crop), note))
    return found, missing


def build(quick, experiments):
    """All cases plus the meta block, in experiment then scene order."""
    scene_list = QUICK_SCENES if quick else nb_bench.SCENES
    sweep_list = [s for s in scene_list if s[0] not in BENCHMARK_ONLY_SCENES]
    pngs, missing = png_scenes(quick)
    builders = {n: b for n, b, _ in scene_list + pngs}
    slot = SceneSlot(builders)

    cases, meta_exp = [], {}
    if "benchmark" in experiments:
        cases += benchmark_cases(slot, scene_list)
    if "benchmark_blocks_none" in experiments:
        bn, differ = blocks_none_cases(slot, sweep_list)
        cases += bn
        meta_exp["benchmark_blocks_none"] = {"runners": differ}
    if "seeding" in experiments:
        cases += seeding_cases(slot, sweep_list)
        meta_exp["seeding"] = {"strides": list(nb_seeding.STRIDES)}
    if "tuning" in experiments:
        cases += tuning_cases(slot, sweep_list, "tuning", TUNING_SUBSET)
        meta_exp["tuning"] = {
            "configs": TUNING_SUBSET,
            "as_benchmark_rows": ["v1", "ccl"],
            "omitted": [c for c in TUNING_CONFIGS
                        if c not in TUNING_SUBSET + ["v1", "ccl"]]}
    if "png" in experiments:
        cases += tuning_cases(slot, pngs, "png", PNG_CONFIGS)
        meta_exp["png"] = {"configs": PNG_CONFIGS,
                           "scenes": [n for n, _, _ in pngs],
                           "missing_inputs": missing}
    slot.release()  # the cases rebuild each scene when they run

    capacity = {}
    for name, (kind, spec) in BENCH_RUNNERS.items():
        capacity[name] = facts(kind, spec)[0]
    for b in nb_tuning.BUILDS:
        capacity[f"lattice_{b}"] = facts("ff", _tuning_spec(f"{b}_L8"))[0]
    meta = {
        "caps": [] if quick else CAPS,
        "method_notes": METHOD_NOTES,
        "quick": quick,
        "scope": SCOPE,
        "experiments": list(experiments),
        "experiment_detail": meta_exp,
        "tpb": TPB,
        "coop_max_at_tpb": capacity,
        "scenes": {n: note for n, _, note in scene_list + pngs},
        "bandwidth_model": numba_ff.MODEL_NOTE,
        "derived": ("model GB/s = info.<backend>_model_bytes / (median "
                    "kernel_ms * 1e6); % of peak against meta.peak_gb_s of "
                    "the same backend's copy probe. Phase medians: from "
                    "runs[backend][1:].phase_ms"),
        "pixels": "scene width * height; info.filled is the red pixel count",
    }
    return cases, meta


def measure_peaks(quick):
    """Both copy probes, back to back, after the GPU is at boost."""
    n_bytes = 16 * 2 ** 20 if quick else 256 * 2 ** 20
    repeats = 2 if quick else 10
    nb = bandwidth.measure_peak_bandwidth(n_bytes=n_bytes, repeats=repeats)
    tr = triton_peak(n_bytes=n_bytes, repeats=repeats)
    gc.collect()
    try:
        from numba import cuda
        cuda.current_context().deallocations.clear()
    except Exception:
        pass
    return {"numba": nb, "triton": tr}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="tiny scenes, 1 repeat, no JSON (smoke test)")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS),
                    help="comma-separated subset of " + ",".join(EXPERIMENTS))
    args = ap.parse_args(argv)
    experiments = [e for e in args.experiments.split(",") if e]
    bad = set(experiments) - set(EXPERIMENTS)
    if bad:
        ap.error(f"unknown experiments {sorted(bad)}")
    repeats = args.repeats or (1 if args.quick else DEFAULT_REPEATS)
    spin = 0.0 if args.quick else 8.0

    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

    if spin:
        spin_up(spin)
    peaks = measure_peaks(args.quick)
    print(f"copy peak: numba {peaks['numba']['gb_s']:.1f} GB/s | "
          f"triton {peaks['triton']['gb_s']:.1f} GB/s")
    cases, meta = build(args.quick, experiments)
    meta["peak_gb_s"] = {k: v["gb_s"] for k, v in peaks.items()}
    meta["peak_runs_gb_s"] = {k: v["runs_gb_s"] for k, v in peaks.items()}
    print(f"{len(cases)} cases, repeats={repeats}, capacities @tpb={TPB}: "
          + ", ".join(f"{k} {v['numba']}/{v['triton']}"
                      for k, v in meta["coop_max_at_tpb"].items()))
    doc = run_cases(CHAPTER, cases, repeats=repeats, meta=meta,
                    write=not args.quick, spin_seconds=spin)
    bad = [r for r in doc["rows"]
           if "error" in r or not r.get("outputs_equal", False)]
    print(f"{len(doc['rows'])} rows, {len(bad)} with an error or a mismatch")
    return doc


if __name__ == "__main__":
    main()
