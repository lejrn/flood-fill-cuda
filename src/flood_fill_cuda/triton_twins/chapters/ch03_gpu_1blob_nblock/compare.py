"""Numba vs Triton for chapter 3, on the chapter's own benchmark experiments.

Five experiments, the first four built from the Numba benchmarks' own
scene and config lists (imported, so they cannot drift):

suite              benchmarks/benchmark.py section 1: every suite scene at
                   tpb=256 with conn4, conn4 bare, conn8 and conn8 bare.
                   Pinned to one grid per variant, min(Numba capacity,
                   Triton capacity) of the instrumented kernel; the bare
                   twin runs on the instrumented kernel's grid, as in the
                   Numba benchmark.
suite_blocks_none  The same scenes with blocks=None on both sides, conn4 and
                   conn8: each backend at its own co-resident maximum, the
                   launch a caller gets by default. Both resolved grids are
                   in config["resolved_blocks"]; they differ (48 vs 144 or
                   120 at tpb=256), so these rows are comparable=False and
                   stay out of the summary averages.
sweep              benchmark.py section 2: TPB_SWEEP x BLOCKS_SWEEP on the
                   sweep scenes at conn4 and conn8. "max" resolves per tpb
                   to the common grid min(both capacities), which is
                   Numba's maximum but not Triton's (is_common_max, and
                   is_backend_max per backend); cells beyond either
                   capacity are listed in meta["skipped_cells"], like the
                   Numba sweep's skipped rows.
barrier_work       benchmarks/benchmark_connectivity_and_barrier_work.py:
                   conn4, conn8, r2, wc and the three conn8-family bare
                   twins at tpb=256, all pinned to the minimum capacity over
                   the seven kernels and both backends.
enqueue            The twin's two enqueue settings against the same Numba
                   kernel, two rows per cell: config.enqueue "lane" (label
                   per_lane, the default every other experiment runs) and
                   "program" (label first_translation, the program-
                   aggregated enqueue of the first translation). All eight
                   variants on sq_2000_center, disk_4001_r1900 and
                   serpentine_256 at the minimum capacity over variants,
                   backends and settings, plus the first run's worst sweep
                   cells (one 512-lane program on disk_4001_r1900, conn4
                   and conn8). first_translation rows are comparable=False
                   with config.label first_translation (as in ch01, ch02
                   and ch04) and carry first_translation=true (as in ch01
                   and ch02; ch04 has the label only): they measure the
                   first translation's cost and stay out of the
                   like-for-like averages. In the default
                   run all 26 per_lane rows repeat a cell of suite (12),
                   barrier_work (12) or sweep (2); those carry
                   duplicate_of=<experiment>, so a unit-wide average can
                   count each cell once.

Every case runs the same configuration on both backends (same tpb =
num_warps * 32, same explicit program count except in suite_blocks_none)
through compare/harness.py. same() compares only outputs the algorithm
fixes regardless of scheduling (img, visited, depth, levels, filled, the
level trace, processed, interior, peaks; per-block counts when the grids
match; cas_attempts for conn4 only).

speedup_kernel is the metric to read. speedup_total mostly compares
CuPy's caching pool with Numba's per-array cuMemAlloc. kernel_ms includes
each runtime's Python launch path, which matters only for the sub-2 ms
scenes.

Both copy peaks (Numba and Triton
probes) are measured after a spin-up and stored in meta; model_bytes per
backend is in each row's info, so model GB/s = model_bytes /
(median kernel_ms * 1e6).

Cap (recorded in meta["caps"]): the 64M-px scene is dropped, which keeps
the default run under 2 GB of host RAM (1.84 GB peak measured). Every
other scene, variant and sweep cell is the Numba benchmarks' own. One run
of every case per backend took 2.3 minutes, so the default (warm-up + 4
rounds; repeats are kept even so each backend runs first equally often)
is about 12 minutes of GPU time, plus about 70 s for the enqueue
experiment (52 cases, measured).

Run:
    python -m flood_fill_cuda.triton_twins.chapters.ch03_gpu_1blob_nblock.compare [--quick] [--repeats N]
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import gc

from ....chapters.ch03_gpu_1blob_nblock import flood_fill as numba_ff
from ....chapters.ch03_gpu_1blob_nblock import scenes
from ....chapters.ch03_gpu_1blob_nblock.benchmarks import benchmark as nb_bench
from ....chapters.ch03_gpu_1blob_nblock.benchmarks import (
    benchmark_connectivity_and_barrier_work as nb_barrier,
)
from ....shared import bandwidth
from ...compare.harness import Case, arrays_equal, run_cases, spin_up
from ...runtime.bandwidth import measure_peak_bandwidth as triton_peak
from . import flood_fill as triton_ff
from .kernels import DEFAULT_ENQ, ENQ_MODES

CHAPTER = "ch03_gpu_1blob_nblock"
# Even, so each backend runs first in half the rounds: the second run of a
# round is measurably slower on the 16M-px scenes (total_ms most of all),
# and an odd count would put Triton second more often. The harness rounds
# an odd --repeats up; 4 keeps the default run near 12 minutes.
DEFAULT_REPEATS = 4

# flood_fill kwargs per variant, named as in the Numba benchmarks
VARIANTS = {
    "conn4": {"connectivity": 4},
    "conn4_bare": {"connectivity": 4, "bare": True},
    "conn8": {"connectivity": 8},
    "conn8_bare": {"connectivity": 8, "bare": True},
    "r2": {"connectivity": 8, "radius": 2},
    "r2_bare": {"connectivity": 8, "radius": 2, "bare": True},
    "wc": {"connectivity": 8, "probe_layout": "warp"},
    "wc_bare": {"connectivity": 8, "probe_layout": "warp", "bare": True},
}
SUITE_VARIANTS = [("conn4", "conn4"), ("conn4_bare", "conn4"),
                  ("conn8", "conn8"), ("conn8_bare", "conn8")]  # (run, grid of)
assert set(nb_barrier.CONFIGS) <= set(VARIANTS)
for _name, _kw in nb_barrier.CONFIGS.items():
    assert {**{"connectivity": 4}, **_kw} == VARIANTS[_name], _name

# ------------------------------------------------------------------- caps

DROPPED_SCENES = {"sq_8000_center"}
CAPS = [
    "sq_8000_center dropped from suite, suite_blocks_none and barrier_work: "
    "the harness holds a Numba and a Triton result at once (~0.8 GB of "
    "host arrays each at 64M px), past the 2.5 GB host-RAM budget",
]
METHOD_NOTES = [
    "every experiment runs at the harness's repeats (default 4; both Numba "
    "benchmarks use GPU_REPEATS=5 and the Numba sweep SWEEP_REPEATS=3). "
    "Repeats must be even (an odd --repeats is rounded up) so each backend "
    "runs first in half the rounds: the second run of a round is slower on "
    "the large scenes, total_ms most of all",
    "the Numba barrier-work benchmark interleaves its 7 configs per round; "
    "the harness interleaves the two backends per case instead",
    "speedup_kernel is the Numba-vs-Triton metric. kernel_ms includes each "
    "runtime's Python launch path (Numba's is 13-52 us slower per launch on "
    "this machine), which matters only for the sub-2 ms scenes. "
    "speedup_total mostly compares host memory management (CuPy's caching "
    "pool vs Numba's cuMemAlloc per array), not the kernels",
    "suite_blocks_none rows run each backend at its own co-resident maximum "
    "(different grids, see config.resolved_blocks), so they are marked "
    "comparable=false and stay out of the summary averages",
    "sweep 'max' is the common grid min(Numba cap, Triton cap): "
    "extra.is_common_max marks it, extra.is_backend_max says which backend "
    "(if either) is at its own capacity there",
    "every row's config.enqueue names the twin's enqueue mode: 'lane' (the "
    "default) everywhere except the enqueue experiment's first_translation "
    "rows",
    "the enqueue experiment's per_lane rows that repeat a cell of suite, "
    "sweep or barrier_work (same scene, variant, tpb, blocks and enqueue) "
    "carry duplicate_of=<experiment>. They stay comparable for the "
    "enqueue experiment's own average; a unit-wide average that keeps "
    "them counts those cells twice",
]
SCOPE = (
    "benchmark.py also times ch01 v2, ch02 dual-global and the @njit "
    "oracle on every scene; those are other units' comparisons and are not "
    "run here. The Numba-only wavefront.py (GIF renderer) and visualize.py "
    "(dashboard) have no timing to compare.")

# --------------------------------------------------------------- quick mode

QUICK_SUITE = [
    ("sq_256_center", lambda: scenes.square_scene(256, 256, 128, 128), "tiny"),
    ("disk_201_r90", lambda: scenes.disk_scene(201, 201, 90), "tiny"),
    ("serpentine_64", lambda: scenes.serpentine_scene(64, 64), "tiny"),
]
QUICK_SWEEP = ["sq_256_center", "serpentine_64"]
QUICK_SWEEP_CONN8 = ["sq_256_center"]
QUICK_TPB = [32, 256]
QUICK_BLOCKS = [1, 8, "max"]

# ------------------------------------------------------ enqueue experiment
# The twin's enqueue in both settings against the same Numba run: per lane
# (the default, warp-aggregated by ptxas like Numba's helper) and per
# program (the first translation). Every variant on three of the barrier
# benchmark's scenes at its pinned grid, plus the sweep's worst cells of
# the first run (one 512-lane program on the 11.3M-px disk, x0.57-0.58).

ENQ_LABELS = {"lane": "per_lane", "program": "first_translation"}
assert set(ENQ_LABELS) == set(ENQ_MODES)
ENQUEUE_SCENES = ["sq_2000_center", "disk_4001_r1900", "serpentine_256"]
assert set(ENQUEUE_SCENES) <= {n for n, _, _ in nb_barrier.SCENES}
# (scene, variant, tpb, blocks)
ENQUEUE_LOW_GRID = [("disk_4001_r1900", "conn4", 512, 1),
                    ("disk_4001_r1900", "conn8", 512, 1)]
QUICK_ENQUEUE_SCENES = ["sq_256_center", "serpentine_64"]
QUICK_ENQUEUE_LOW_GRID = [("sq_256_center", "conn4", 512, 1),
                          ("sq_256_center", "conn8", 512, 1)]
ENQUEUE_NOTE = (
    "enqueue: two rows per (scene, variant, grid), config.enqueue 'lane' "
    "(label per_lane: the twin's default, one relaxed atomic per claiming "
    "lane, which ptxas warp-aggregates into the SASS of Numba's "
    "_warp_enqueue_global) and 'program' (label first_translation: tl.sum + "
    "tl.cumsum over the program and one atomic per program per direction, "
    "7 CTA barriers per enqueue site in SASS). Both rows time the same "
    "Numba kernel. first_translation rows are comparable=false with "
    "config.label first_translation (as in ch01, ch02 and ch04) and carry "
    "first_translation=true (as in ch01 and ch02; ch04 has the label "
    "only), so the like-for-like averages describe the default twin only; "
    "their speedups are the measured cost of the first translation, and an "
    "'own default' average over all measured rows must drop them too. "
    "per_lane rows whose cell another experiment already measures carry "
    "duplicate_of=<experiment>; a unit-wide average should skip them")


class SceneSlot:
    """Holds one scene at a time: cases are grouped by scene, so a scene is
    built when its first case runs and dropped when the next one starts."""

    def __init__(self, builders):
        self.builders = builders
        self.name = None
        self.scene = None
        self.shapes = {}

    def get(self, name):
        if name != self.name:
            self.release()
            self.scene = self.builders[name]()
            self.name = name
            self.shapes[name] = self.scene[0].shape[:2]
        return self.scene

    def pixels(self, name):
        """width * height, building the scene once if it was never seen."""
        if name not in self.shapes:
            self.get(name)
        w, h = self.shapes[name]
        return int(w * h)

    def release(self):
        self.scene = self.name = None
        gc.collect()


# ---------------------------------------------------------- per-backend facts

def caps(name, tpb, enqueue=DEFAULT_ENQ):
    """Co-resident capacity of one variant at one tpb, per backend (the
    Triton one for the given enqueue mode)."""
    kw = VARIANTS[name]
    return {"numba": numba_ff.max_blocks(threads_per_block=tpb, **kw),
            "triton": triton_ff.max_blocks(threads_per_block=tpb,
                                           enqueue=enqueue, **kw)}


def resources(name, tpb, enqueue=DEFAULT_ENQ):
    """Registers and friends of both compiled kernels (compiles if needed)."""
    kw = VARIANTS[name]
    key = (kw.get("bare", False), kw.get("connectivity", 4),
           kw.get("radius", 1), kw.get("probe_layout", "thread"))
    numba_ff.max_blocks(threads_per_block=tpb, **kw)
    numba_kernel = numba_ff._KERNELS[key]
    return {
        "triton_resources": triton_ff.kernel_info(threads_per_block=tpb,
                                                  enqueue=enqueue, **kw),
        "numba_resources": {
            "n_regs": _one(numba_kernel.get_regs_per_thread()),
            "shared_bytes": _one(numba_kernel.get_shared_mem_per_block()),
            "local_bytes": _one(numba_kernel.get_local_mem_per_thread()),
        },
    }


def _one(value):
    """Numba returns {signature: value}; a kernel here has one signature."""
    if isinstance(value, dict):
        values = sorted({int(v) for v in value.values()})
        return values[0] if len(values) == 1 else values
    return int(value)


def make_same(name, pinned):
    """same(numba, triton): the deterministic outputs only."""
    kw = VARIANTS[name]
    instrumented = not kw.get("bare", False)
    conn4 = kw.get("connectivity", 4) == 4

    def same(n, t):
        pairs = {"img": (n.img, t.img), "visited": (n.visited, t.visited),
                 "depth": (n.depth, t.depth)}
        if instrumented:
            pairs["level_sizes"] = (n.level_sizes, t.level_sizes)
        if pinned:
            pairs["processed_per_block"] = (n.processed_per_block,
                                            t.processed_per_block)
        ok, detail = arrays_equal(**pairs)
        if not ok:
            return ok, detail
        fields = ["levels", "filled", "processed", "interior", "peak_level",
                  "peak_occupancy", "level_trace_truncated"]
        if pinned:
            fields += ["blocks", "thread_util_pct", "warp_engagement_pct"]
        if conn4 and instrumented:
            fields.append("cas_attempts")  # exact: one probe per edge
        for f in fields:
            a, b = getattr(n, f), getattr(t, f)
            if a != b:
                return False, f"{f}: numba {a} vs triton {b}"
        return True, ""

    return same


def info(n, t):
    """Per-case facts from the warm-up results. Unprefixed fields are the
    deterministic ones same() checks; the rest are per backend."""
    return {
        "filled": t.filled, "levels": t.levels, "peak_level": t.peak_level,
        "interior": t.interior,
        "numba_thread_util_pct": n.thread_util_pct,
        "triton_thread_util_pct": t.thread_util_pct,
        "numba_blocks": n.blocks, "triton_blocks": t.blocks,
        "numba_cas_attempts": n.cas_attempts,
        "triton_cas_attempts": t.cas_attempts,
        "numba_model_bytes": n.model_bytes,
        "triton_model_bytes": t.model_bytes,
        "numba_balance_cv_pct": n.balance_cv_pct,
        "triton_balance_cv_pct": t.balance_cv_pct,
        "numba_distinct_sms": n.distinct_sms,
        "triton_distinct_sms": t.distinct_sms,
    }


def make_case(experiment, slot, scene, name, tpb, blocks, notes="",
              grid_of=None, extra=None, enqueue=DEFAULT_ENQ):
    """One cell: same scene, same kwargs, same grid on both backends.
    blocks=None lets each backend resolve its own grid: both resolved sizes
    go into the config, and the row is comparable only if they agree.
    enqueue picks the Triton twin's enqueue mode (config["enqueue"]); a
    "program" row is the first translation: comparable=False and label
    first_translation (as in ch01, ch02 and ch04), plus
    first_translation=true (as in ch01 and ch02; ch04 has the label only;
    see ENQUEUE_NOTE)."""
    kw = {**VARIANTS[name], "threads_per_block": tpb, "blocks": blocks}
    c = caps(name, tpb, enqueue)

    def run_numba():
        img, sx, sy = slot.get(scene)
        return numba_ff.flood_fill(img, sx, sy, **kw)

    def run_triton():
        img, sx, sy = slot.get(scene)
        return triton_ff.flood_fill(img, sx, sy, enqueue=enqueue, **kw)

    config = {"variant": name, **VARIANTS[name], "tpb": tpb,
              "num_warps": tpb // 32,
              "blocks": "None" if blocks is None else int(blocks),
              "enqueue": enqueue, "label": ENQ_LABELS[enqueue]}
    if blocks is None:
        resolved = dict(c)
    else:
        resolved = {"numba": int(blocks), "triton": int(blocks)}
    config["resolved_blocks"] = resolved
    equal_grid = resolved["numba"] == resolved["triton"]
    row_extra = {"caps": c, **resources(name, tpb, enqueue), **(extra or {})}
    if grid_of:
        row_extra["grid_of"] = grid_of
    if enqueue != DEFAULT_ENQ:
        # the filter key ch01 and ch02 use too (config.label is the other)
        row_extra["first_translation"] = True
    return Case(experiment=experiment, scene=scene, config=config,
                run_numba=run_numba, run_triton=run_triton,
                same=make_same(name, pinned=equal_grid),
                pixels=slot.pixels(scene), info=info,
                notes=notes, extra=row_extra,
                comparable=equal_grid and enqueue == DEFAULT_ENQ)


# -------------------------------------------------------------- experiments

def suite_cases(slot, scene_list, tpb):
    out = []
    for sname, _, note in scene_list:
        for name, grid_of in SUITE_VARIANTS:
            c = caps(grid_of, tpb)
            pin = min(c["numba"], c["triton"])
            out.append(make_case("suite", slot, sname, name, tpb, pin,
                                 notes=note, grid_of=grid_of))
    return out


def blocks_none_cases(slot, scene_list, tpb):
    return [make_case("suite_blocks_none", slot, sname, name, tpb, None,
                      notes=note)
            for sname, _, note in scene_list for name in ("conn4", "conn8")]


def sweep_cases(slot, passes, tpbs, blocks_axis, skipped):
    out = []
    for sname, conn in passes:
        name = "conn4" if conn == 4 else "conn8"
        for tpb in tpbs:
            c = caps(name, tpb)
            cap = min(c["numba"], c["triton"])
            seen = set()
            for b in blocks_axis:
                n = cap if b == "max" else b
                if n in seen:
                    continue
                seen.add(n)
                if n > cap:
                    skipped.append({"scene": sname, "connectivity": conn,
                                    "tpb": tpb, "blocks": n,
                                    "numba_cap": c["numba"],
                                    "triton_cap": c["triton"]})
                    continue
                # "max" is the common grid: Numba's capacity, at most
                # Triton's (which is often 1.5-3x larger)
                out.append(make_case(
                    "sweep", slot, sname, name, tpb, n,
                    extra={"is_common_max": n == cap,
                           "is_backend_max": {k: n == v
                                              for k, v in c.items()}}))
    return out


def barrier_cases(slot, scene_list, tpb):
    names = list(nb_barrier.CONFIGS)
    all_caps = {n: caps(n, tpb) for n in names}
    pin = min(min(c.values()) for c in all_caps.values())
    out = [make_case("barrier_work", slot, sname, name, tpb, pin, notes=note)
           for sname, _, note in scene_list for name in names]
    return out, all_caps, pin


def enqueue_cases(slot, scene_list, low_grid, tpb, notes):
    """Both enqueue settings of every variant at one pinned grid (the
    minimum capacity over the variants, both backends and both settings),
    then the low-grid cells. Scene-grouped, lane row first."""
    names = list(VARIANTS)
    all_caps = {n: {e: caps(n, tpb, e) for e in ENQ_MODES} for n in names}
    pin = min(v for c in all_caps.values() for e in c.values()
              for v in e.values())
    out = []
    order = list(scene_list) + [s for s, _, _, _ in low_grid
                                if s not in scene_list]
    for sname in dict.fromkeys(order):
        if sname in scene_list:
            for name in names:
                for enq in ENQ_MODES:
                    out.append(make_case("enqueue", slot, sname, name, tpb,
                                         pin, notes=notes.get(sname, ""),
                                         enqueue=enq))
        for s, name, ltpb, blocks in low_grid:
            if s != sname:
                continue
            for enq in ENQ_MODES:
                out.append(make_case(
                    "enqueue", slot, sname, name, ltpb, blocks,
                    notes="the first run's worst sweep cell: one program",
                    enqueue=enq))
    return out, all_caps, pin


def _cell(case):
    cfg = case.config
    return (case.scene, cfg["variant"], cfg["tpb"], cfg["blocks"],
            cfg["enqueue"])


def mark_repeated_cells(earlier, enqueue_rows):
    """Tag each per_lane enqueue row whose cell (scene, variant, tpb,
    explicit blocks, enqueue) an earlier experiment already measures with
    duplicate_of=<that experiment>. Those rows stay comparable (the
    enqueue experiment's own average needs them), but a unit-wide average
    should skip them to weigh each cell once. Returns the tagged count."""
    seen = {}
    for c in earlier:
        if c.config["blocks"] != "None":
            seen.setdefault(_cell(c), c.experiment)
    tagged = 0
    for c in enqueue_rows:
        if c.config["enqueue"] == DEFAULT_ENQ and _cell(c) in seen:
            c.extra["duplicate_of"] = seen[_cell(c)]
            tagged += 1
    return tagged


def build(quick):
    """All cases plus the meta block, in scene-grouped order."""
    if quick:
        suite_list = QUICK_SUITE
        sweep_passes = ([(s, 4) for s in QUICK_SWEEP]
                        + [(s, 8) for s in QUICK_SWEEP_CONN8])
        tpbs, blocks_axis = QUICK_TPB, QUICK_BLOCKS
        barrier_list = QUICK_SUITE
        enq_scenes, enq_low = QUICK_ENQUEUE_SCENES, QUICK_ENQUEUE_LOW_GRID
    else:
        suite_list = [s for s in nb_bench.SCENES if s[0] not in DROPPED_SCENES]
        sweep_passes = ([(s, 4) for s in nb_bench.SWEEP_SCENES]
                        + [(s, 8) for s in nb_bench.SWEEP_SCENES_CONN8])
        tpbs, blocks_axis = nb_bench.TPB_SWEEP, nb_bench.BLOCKS_SWEEP
        barrier_list = [s for s in nb_barrier.SCENES
                        if s[0] not in DROPPED_SCENES]
        enq_scenes, enq_low = ENQUEUE_SCENES, ENQUEUE_LOW_GRID

    builders = {n: b for n, b, _ in suite_list + barrier_list}
    for n, b, _ in nb_bench.SCENES:
        builders.setdefault(n, b)
    slot = SceneSlot(builders)

    tpb = nb_barrier.TPB  # 256, the suite's tpb too
    skipped = []
    cases = suite_cases(slot, suite_list, tpb)
    cases += blocks_none_cases(slot, suite_list, tpb)
    cases += sweep_cases(slot, sweep_passes, tpbs, blocks_axis, skipped)
    bcases, bcaps, bpin = barrier_cases(slot, barrier_list, tpb)
    cases += bcases
    ecases, ecaps, epin = enqueue_cases(
        slot, enq_scenes, enq_low, tpb,
        {n: note for n, _, note in suite_list + barrier_list})
    repeated = mark_repeated_cells(cases, ecases)
    cases += ecases
    slot.release()  # the cases rebuild each scene when they run

    meta = {
        "caps": [] if quick else CAPS,
        "method_notes": METHOD_NOTES + [ENQUEUE_NOTE],
        "quick": quick,
        "scope": SCOPE,
        "skipped_cells": skipped,
        "coop_max_by_tpb": {
            "conn4": {t: caps("conn4", t) for t in tpbs},
            "conn8": {t: caps("conn8", t) for t in tpbs},
        },
        "barrier_work": {"tpb": tpb, "pinned_blocks": bpin,
                         "coop_max_by_config": bcaps},
        "enqueue": {"tpb": tpb, "pinned_blocks": epin,
                    "scenes": enq_scenes,
                    "low_grid_cells": [
                        {"scene": s, "variant": v, "tpb": t, "blocks": b}
                        for s, v, t, b in enq_low],
                    "coop_max_by_config_and_enqueue": ecaps,
                    "labels": ENQ_LABELS, "default": DEFAULT_ENQ,
                    "per_lane_rows_repeating_a_cell": repeated,
                    "note": ENQUEUE_NOTE},
        "triton_enqueue_default": DEFAULT_ENQ,
        "suite_tpb": tpb,
        "sweep": {"tpb_sweep": tpbs,
                  "blocks_sweep": [str(b) for b in blocks_axis],
                  "passes": [{"scene": s, "connectivity": c}
                             for s, c in sweep_passes]},
        "bandwidth_model": bandwidth.MODEL_NOTE,
        "derived": ("model GB/s = info.<backend>_model_bytes / (median "
                    "kernel_ms * 1e6); % of peak against meta.peak_gb_s of "
                    "the same backend's copy probe"),
        "pixels": "scene width * height; info.filled is the blob size",
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
    args = ap.parse_args(argv)
    repeats = args.repeats or (1 if args.quick else DEFAULT_REPEATS)
    repeats += repeats % 2  # what the harness runs: even, see METHOD_NOTES
    spin = 0.0 if args.quick else 8.0

    if spin:
        spin_up(spin)
    peaks = measure_peaks(args.quick)
    print(f"copy peak: numba {peaks['numba']['gb_s']:.1f} GB/s | "
          f"triton {peaks['triton']['gb_s']:.1f} GB/s")
    cases, meta = build(args.quick)
    meta["peak_gb_s"] = {k: v["gb_s"] for k, v in peaks.items()}
    meta["peak_runs_gb_s"] = {k: v["runs_gb_s"] for k, v in peaks.items()}
    print(f"{len(cases)} cases, repeats={repeats}, "
          f"{len(meta['skipped_cells'])} sweep cells beyond a capacity")
    doc = run_cases(CHAPTER, cases, repeats=repeats, meta=meta,
                    write=not args.quick, spin_seconds=spin)
    bad = [r for r in doc["rows"]
           if "error" in r or not r.get("outputs_equal", False)]
    print(f"{len(doc['rows'])} rows, {len(bad)} with an error or a mismatch")
    return doc


if __name__ == "__main__":
    main()
