"""Numba vs Triton for chapter 4, on the chapter's own benchmark experiments.

Seven experiments on the Numba benchmarks' own four scenes (imported, so
they cannot drift), all at tpb=256. Every Triton run uses the default
per-lane enqueue (ENQ="lane") except the program rows of the enqueue
experiment; each row's config records its "enqueue":

modes        benchmarks/benchmark.py's round-robin: seq, seq_half, multi,
             multi_xy, bare, bare_xy, seq8, multi8. Each config runs on its
             own kernel's grid, as the Numba benchmark does (blocks=None,
             each kernel's own capacity), pinned on both backends to
             min(Numba capacity, Triton capacity) of that kernel, which is
             Numba's own capacity for every ch04 kernel. seq_half runs at
             half the lin kernel's pinned grid (the Numba coop // 2).
seq_a, seq_b the seq config again, reporting launch A's or launch B's
             kernel time (the Numba benchmark's seq_a_ms / seq_b_ms, which
             feed ideal_max and the packing tax). total_ms is the whole
             two-launch call.
packing_tax  benchmark.py's mb_a: ch03's single-blob kernel on blob A
             alone (the Numba ch03 kernel vs its Triton twin), pinned to
             min of both capacities. packing tax = seq_a / mb_a - 1 per
             backend.
blocks_none  multi and multi8 with blocks=None on both sides: each backend
             at its own co-resident maximum, the launch a caller gets by
             default. Both resolved grids are in config["resolved_blocks"];
             they differ, so these rows are comparable=False and stay out
             of the summary averages.
radius2      benchmarks/benchmark_radius2_barrier_work.py: seq8, multi8,
             seq8r2, multi8r2, all pinned to the minimum capacity over the
             lin8 and lin8r2 kernels and both backends (Numba pins over its
             two kernels).
enqueue      the cost of the first translation. multi, bare_xy, multi8 and
             multi8r2 (one config per kernel family: lin 4-conn, xy bare,
             8-conn, radius 2) on every scene, Numba vs Triton twice: the
             default per-lane enqueue (config.enqueue="lane",
             label="per_lane") and the program-aggregated enqueue of the
             first translation (config.enqueue="program",
             label="first_translation"). Both rows of a pair run at one grid,
             min(Numba cap, Triton lane cap, Triton program cap), which is
             Numba's own grid. Each row's triton_ptx_bar_sync is the CTA
             barrier count of the binary it ran (equal to the SASS
             BAR.SYNC count; see sass.py).

Every case runs the same configuration on both backends (same tpb =
num_warps * 32, same explicit program count except in blocks_none) through
compare/harness.py. same() compares only outputs the algorithm fixes
regardless of scheduling: img, visited, depth, label, filled (and per
blob), levels (and per blob), processed, interior, the per-launch peaks
and level traces; per-program counts and thread utilisation when the grids
match; cas_attempts and model_bytes at 4-conn radius 1 only.

speedup_kernel (median over median) is the metric to read, next to
speedup_kernel_min (best run over best run), which this script adds to
every row: benchmark.py's "(min)" column. Single runs on this GPU can be
1.4-2.8x off the median on either backend, so a row's difference is real
only when the two ratios agree; on opposite sides of 1 the row is noise.
speedup_total mostly compares CuPy's caching pool with Numba's per-array
cuMemAlloc. kernel_ms includes each runtime's Python launch path (twice
for the sequential configs).

The radius-2 rows carry one structural difference: Numba skips ring 2
per warp, the twin per program (8 warps at tpb=256), so the Triton
seq8r2 / multi8r2 times include masked ring-2 probe work Numba skips, and
the gate's program-wide tl.max adds 3 CTA barriers per tile (see the
README's Deviations).

Not run here, by design: mode="streams" (the Numba benchmark excludes it:
two concurrent cooperative grids can wedge the GPU), and the @njit oracle
timings and filled cross-check (CPU only; same() checks the two backends
against each other on every run instead).

Both copy peaks (Numba and Triton probes) are measured after a spin-up and
stored in meta; model_bytes per backend is in each row's info, so model
GB/s = model_bytes / (median kernel_ms * 1e6).

Caps: none. Every scene, config and grid is the Numba benchmarks' own.
The harness holds one Numba and one Triton result at a time (about 0.3 GB
each at the 22.9M-px asym scene); a probe of the seq, multi and mb_a cases
on that scene peaked at 1.5 GB RSS. Each run spends about 0.9 s in the
drivers at the 18-23M-px scenes, so the default (100 cases, warm-up + 6
rounds) is about 18 minutes of GPU time.

Run:
    python -m flood_fill_cuda.triton_twins.chapters.ch04_gpu_2blob_nblock.compare [--quick] [--repeats N]
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import gc
import json

from ....chapters.ch03_gpu_1blob_nblock import flood_fill as numba_mb
from ....chapters.ch04_gpu_2blob_nblock import flood_fill as numba_ff
from ....chapters.ch04_gpu_2blob_nblock import scenes
from ....chapters.ch04_gpu_2blob_nblock.benchmarks import benchmark as nb_bench
from ....chapters.ch04_gpu_2blob_nblock.benchmarks import (
    benchmark_radius2_barrier_work as nb_r2,
)
from ....shared import bandwidth
from ...compare.harness import Case, arrays_equal, run_cases, spin_up
from ...runtime.bandwidth import measure_peak_bandwidth as triton_peak
from ..ch03_gpu_1blob_nblock import flood_fill as triton_mb
from . import flood_fill as triton_ff
from . import sass as twin_sass
from .kernels import ENQ_LANE, ENQ_PROGRAM

CHAPTER = "ch04_gpu_2blob_nblock"
TPB = nb_bench.TPB  # 256, also nb_r2.TPB
assert nb_r2.TPB == TPB
# The Numba benchmarks use GPU_REPEATS=5; the harness needs an even count
# so each backend runs first in half the rounds, so 5 rounds up to 6.
DEFAULT_REPEATS = nb_bench.GPU_REPEATS + nb_bench.GPU_REPEATS % 2

# benchmark.py's round-robin configs (defined inside its bench_scene)
MODE_CONFIGS = {
    "seq": {"mode": "sequential"},
    "seq_half": {"mode": "sequential"},  # blocks = lin capacity // 2
    "multi": {"mode": "multisource"},
    "multi_xy": {"mode": "multisource", "entry_format": "xy"},
    "bare": {"mode": "multisource", "bare": True},
    "bare_xy": {"mode": "multisource", "bare": True, "entry_format": "xy"},
    "seq8": {"mode": "sequential", "connectivity": 8},
    "multi8": {"mode": "multisource", "connectivity": 8},
}
R2_CONFIGS = nb_r2.CONFIGS  # seq8, multi8, seq8r2, multi8r2
for _name in ("seq8", "multi8"):
    assert R2_CONFIGS[_name] == MODE_CONFIGS[_name], _name

# The enqueue experiment: one config per kernel family, both ENQ settings
ENQ_CONFIGS = {"multi": MODE_CONFIGS["multi"],
               "bare_xy": MODE_CONFIGS["bare_xy"],
               "multi8": MODE_CONFIGS["multi8"],
               "multi8r2": R2_CONFIGS["multi8r2"]}
ENQ_LABELS = {ENQ_LANE: "per_lane", ENQ_PROGRAM: "first_translation"}

CAPS = []
METHOD_NOTES = [
    "every experiment runs at the harness's repeats (default 6: the Numba "
    "benchmarks' GPU_REPEATS=5 rounded up to even, so each backend runs "
    "first in half the rounds)",
    "the Numba benchmarks interleave their configs per round; the harness "
    "interleaves the two backends per case instead",
    "modes rows run each config on its own kernel's grid, like the Numba "
    "benchmark (blocks=None there: lin 48, xy 48, bare 72, bare_xy 72, "
    "lin8 48 at tpb=256), pinned to min(Numba cap, Triton cap) on both "
    "backends; the Numba benchmark docstring says the bare twins are "
    "pinned to the instrumented grid, its code does not pin them",
    "seq_a/seq_b rows time the full sequential call but report one "
    "launch's kernel_ms (perf_counter + synchronize around that launch, as "
    "the drivers measure it); total_ms is the whole call",
    "speedup_kernel is the Numba-vs-Triton metric. kernel_ms includes each "
    "runtime's Python launch path (twice for sequential configs), which "
    "matters only on two_sq_300 (2-6 ms kernels). speedup_total mostly "
    "compares host memory management (CuPy's caching pool vs Numba's "
    "cuMemAlloc per array), not the kernels",
    "blocks_none rows run each backend at its own co-resident maximum "
    "(different grids, see config.resolved_blocks), so they are marked "
    "comparable=false and stay out of the summary averages",
    "speedup_kernel_min = numba.kernel_ms.min / triton.kernel_ms.min, the "
    "best-vs-best ratio (benchmark.py's '(min)' column), added to every "
    "row after the harness returns. Single runs reach 1.4-2.8x the median "
    "on either backend, so a row's speedup_kernel is a real difference "
    "only when speedup_kernel_min agrees with it; ratios on opposite sides "
    "of 1 mean the row is noise",
    "radius2 rows: Numba skips ring 2 per warp (divergent `if interior:`), "
    "the twin per program (256 lanes = 8 warps): when any lane is "
    "interior every warp runs the 16 masked ring-2 probes, and the gate "
    "itself is a program-wide tl.max (3 CTA barriers per tile). Outputs "
    "and counters are unchanged, but the Triton seq8r2/multi8r2 times "
    "include masked probe work Numba skips, so r2_multi_vs_conn8 is not a "
    "pure algorithm-vs-algorithm ratio on the Triton side",
    "Triton runs use the per-lane enqueue (ENQ='lane'): one relaxed atomic "
    "per winning lane, which ptxas compiles warp-aggregated (VOTEU.ANY, "
    "POPC, one leader ATOMG.ADD, SHFL.IDX), the SASS of Numba's "
    "_warp_enqueue_global. The enqueue experiment adds, per scene and "
    "config, a row with the first translation's program-aggregated "
    "enqueue (ENQ='program', label first_translation: tl.sum + tl.cumsum, "
    "7 BAR.SYNC per enqueue site), at the same grid, so the cost of that "
    "translation choice is measured in the same harness. Those rows "
    "measure an alternative twin, not the default one",
]
SCOPE = (
    "benchmark.py also times the @njit two-blob oracle and cross-checks "
    "filled against it; that is CPU-only and not a Numba-vs-Triton "
    "comparison, so it is not run (same() checks the backends against each "
    "other on every run). mode='streams' is excluded, as in the Numba "
    "benchmark. wavefront.py (GIF renderer) and visualize.py (dashboard) "
    "have no timing to compare.")

# --------------------------------------------------------------- quick mode

QUICK_SCENES = [
    ("two_sq_60", lambda: scenes.two_squares_scene(140, 80, 60, 60, gap=8),
     "tiny pair"),
    ("two_disks_r40", lambda: scenes.two_disks_scene(100, 200, 40, gap=8),
     "tiny disks"),
    ("asym_80_16", lambda: scenes.asym_squares_scene(104, 88, 80, 16, gap=8),
     "tiny asym"),
]


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


class LaunchView:
    """A sequential result whose kernel_ms is ONE launch's time (the
    harness records whatever kernel_ms says); every other attribute is the
    full result's."""

    def __init__(self, result, which):
        self._r = result
        self.kernel_ms = result.kernel_a_ms if which == "a" else result.kernel_b_ms
        self.total_ms = result.total_ms

    def __getattr__(self, name):
        return getattr(self._r, name)


# ---------------------------------------------------------- per-backend facts

def _key(kw):
    return (kw.get("entry_format", "lin"), kw.get("bare", False),
            kw.get("connectivity", 4), kw.get("radius", 1))


def _kernel_kw(kw):
    """The flood_fill kwargs that pick the kernel (mode excluded)."""
    return {k: v for k, v in kw.items() if k != "mode"}


def caps(kw, enqueue=ENQ_LANE):
    """Co-resident capacity of one ch04 kernel at TPB, per backend (the
    Triton binary compiled with ``enqueue``)."""
    k = _kernel_kw(kw)
    return {"numba": numba_ff.max_blocks(threads_per_block=TPB, **k),
            "triton": triton_ff.max_blocks(threads_per_block=TPB,
                                           enqueue=enqueue, **k)}


def _one(value):
    """Numba returns {signature: value}; a kernel here has one signature."""
    if isinstance(value, dict):
        values = sorted({int(v) for v in value.values()})
        return values[0] if len(values) == 1 else values
    return int(value)


def resources(kw, enqueue=ENQ_LANE):
    """Registers and friends of both compiled ch04 kernels, plus the CTA
    barrier count of the Triton binary (its PTX; equal to SASS BAR.SYNC)."""
    k = _kernel_kw(kw)
    numba_ff.max_blocks(threads_per_block=TPB, **k)  # compiles if needed
    nk = numba_ff._KERNELS[_key(kw)]
    ptx = triton_ff._warmup(*_key(kw), TPB, enqueue).asm["ptx"]
    return {
        "kernel": nk.__name__,
        "triton_resources": triton_ff.kernel_info(threads_per_block=TPB,
                                                  enqueue=enqueue, **k),
        "triton_ptx_bar_sync": twin_sass.ptx_barriers(ptx),
        "numba_resources": {
            "n_regs": _one(nk.get_regs_per_thread()),
            "shared_bytes": _one(nk.get_shared_mem_per_block()),
            "local_bytes": _one(nk.get_local_mem_per_thread()),
        },
    }


def make_same(kw, pinned):
    """same(numba, triton) for a ch04 result: deterministic outputs only."""
    instrumented = not kw.get("bare", False)
    conn4_r1 = kw.get("connectivity", 4) == 4 and kw.get("radius", 1) == 1

    def same(n, t):
        pairs = {"img": (n.img, t.img), "visited": (n.visited, t.visited),
                 "depth": (n.depth, t.depth), "label": (n.label, t.label)}
        for i, (ln, lt) in enumerate(zip(n.launches, t.launches)):
            pairs[f"level_sizes[{i}]"] = (ln.level_sizes, lt.level_sizes)
            if pinned:
                pairs[f"processed_per_block[{i}]"] = (
                    ln.processed_per_block, lt.processed_per_block)
        ok, detail = arrays_equal(**pairs)
        if not ok:
            return ok, detail
        fields = ["filled", "filled_a", "filled_b", "levels", "levels_a",
                  "levels_b", "processed", "interior"]
        if pinned:
            fields.append("blocks")
        if conn4_r1 and instrumented:
            fields += ["cas_attempts", "model_bytes"]  # one probe per edge
        for f in fields:
            a, b = getattr(n, f), getattr(t, f)
            if a != b:
                return False, f"{f}: numba {a} vs triton {b}"
        if len(n.launches) != len(t.launches):
            return False, "launch count differs"
        lfields = ["filled", "levels", "peak_level", "peak_occupancy",
                   "processed", "interior", "level_trace_truncated"]
        if pinned:
            lfields.append("thread_util_pct")
        for i, (ln, lt) in enumerate(zip(n.launches, t.launches)):
            for f in lfields:
                a, b = getattr(ln, f), getattr(lt, f)
                if a != b:
                    return False, f"launches[{i}].{f}: numba {a} vs triton {b}"
        return True, ""

    return same


def info(n, t):
    """Per-case facts from the warm-up results. Unprefixed fields are the
    deterministic ones same() checks; the rest are per backend."""
    out = {
        "filled": t.filled, "filled_a": t.filled_a, "filled_b": t.filled_b,
        "levels": t.levels, "levels_a": t.levels_a, "levels_b": t.levels_b,
        "interior": t.interior,
        "peak_frontier": max(l.peak_level for l in t.launches),
    }
    for name, r in (("numba", n), ("triton", t)):
        out[f"{name}_blocks"] = r.blocks
        out[f"{name}_cas_attempts"] = r.cas_attempts
        out[f"{name}_model_bytes"] = r.model_bytes
        out[f"{name}_thread_util_pct"] = [l.thread_util_pct
                                          for l in r.launches]
        out[f"{name}_distinct_sms"] = [l.distinct_sms for l in r.launches]
    return out


def make_case(experiment, slot, scene, name, kw, blocks, notes="",
              grid_of=None, which=None, extra=None, enqueue=ENQ_LANE,
              label=None):
    """One ch04 cell: same scene, same kwargs, same grid on both backends.
    blocks=None lets each backend resolve its own grid: both resolved sizes
    go into the config, and the row is comparable only if they agree.
    ``enqueue`` is the Triton side's ENQ (Numba has one enqueue)."""
    c = caps(kw, enqueue)

    def wrap(r):
        return LaunchView(r, which) if which else r

    def run_numba():
        img, seeds = slot.get(scene)
        return wrap(numba_ff.flood_fill(img, seeds, threads_per_block=TPB,
                                        blocks=blocks, **kw))

    def run_triton():
        img, seeds = slot.get(scene)
        return wrap(triton_ff.flood_fill(img, seeds, threads_per_block=TPB,
                                         blocks=blocks, enqueue=enqueue,
                                         **kw))

    config = {"config": name, **kw, "tpb": TPB, "num_warps": TPB // 32,
              "blocks": "None" if blocks is None else int(blocks),
              "enqueue": enqueue}
    if label:
        config["label"] = label
    if which:
        config["launch"] = which
    if blocks is None:
        resolved = dict(c)
    else:
        resolved = {"numba": int(blocks), "triton": int(blocks)}
    config["resolved_blocks"] = resolved
    equal_grid = resolved["numba"] == resolved["triton"]
    row_extra = {"caps": c, **resources(kw, enqueue), **(extra or {})}
    if grid_of:
        row_extra["grid_of"] = grid_of
    return Case(experiment=experiment, scene=scene, config=config,
                run_numba=run_numba, run_triton=run_triton,
                same=make_same(kw, pinned=equal_grid),
                pixels=slot.pixels(scene), info=info, notes=notes,
                extra=row_extra, comparable=equal_grid)


# ------------------------------------------- packing tax: ch03 on blob A

def mb_caps():
    return {"numba": numba_mb.max_blocks(threads_per_block=TPB),
            "triton": triton_mb.max_blocks(threads_per_block=TPB)}


def mb_same(n, t):
    ok, detail = arrays_equal(img=(n.img, t.img), visited=(n.visited, t.visited),
                              depth=(n.depth, t.depth),
                              level_sizes=(n.level_sizes, t.level_sizes),
                              processed_per_block=(n.processed_per_block,
                                                   t.processed_per_block))
    if not ok:
        return ok, detail
    for f in ("levels", "filled", "processed", "cas_attempts", "peak_level",
              "peak_occupancy", "blocks", "thread_util_pct"):
        a, b = getattr(n, f), getattr(t, f)
        if a != b:
            return False, f"{f}: numba {a} vs triton {b}"
    return True, ""


def mb_info(n, t):
    return {"filled": t.filled, "levels": t.levels,
            "numba_model_bytes": n.model_bytes,
            "triton_model_bytes": t.model_bytes}


def make_mb_case(slot, scene, blocks, notes=""):
    """ch03's flood_fill on blob A (seeds[0]) of the two-blob scene."""
    c = mb_caps()

    def run_numba():
        img, seeds = slot.get(scene)
        return numba_mb.flood_fill(img, *seeds[0], threads_per_block=TPB,
                                   blocks=blocks)

    def run_triton():
        img, seeds = slot.get(scene)
        return triton_mb.flood_fill(img, *seeds[0], threads_per_block=TPB,
                                    blocks=blocks)

    nk = numba_mb._KERNELS[(False, 4, 1, "thread")]
    numba_mb.max_blocks(threads_per_block=TPB)
    config = {"config": "mb_a", "kernel": "ch03 conn4 instrumented, blob A",
              "tpb": TPB, "num_warps": TPB // 32, "blocks": int(blocks),
              "resolved_blocks": {"numba": int(blocks),
                                  "triton": int(blocks)}}
    extra = {"caps": c,
             "kernel": nk.__name__,
             "triton_resources": triton_mb.kernel_info(threads_per_block=TPB),
             "numba_resources": {
                 "n_regs": _one(nk.get_regs_per_thread()),
                 "shared_bytes": _one(nk.get_shared_mem_per_block()),
                 "local_bytes": _one(nk.get_local_mem_per_thread())}}
    return Case(experiment="packing_tax", scene=scene, config=config,
                run_numba=run_numba, run_triton=run_triton, same=mb_same,
                pixels=slot.pixels(scene), info=mb_info, notes=notes,
                extra=extra)


# -------------------------------------------------------------- experiments

def pinned(kw):
    c = caps(kw)
    return min(c.values())


def enq_pinned(kw):
    """One grid for both rows of an enqueue pair: min over Numba and both
    Triton binaries (Numba's own grid for every ch04 kernel)."""
    return min(min(caps(kw, e).values()) for e in ENQ_LABELS)


def enqueue_cases(slot, scene, note):
    """The enqueue experiment's cases for one scene: per config, the
    per-lane row, then the first translation's program row."""
    out = []
    for name, kw in ENQ_CONFIGS.items():
        blocks = enq_pinned(kw)
        for enq, label in ENQ_LABELS.items():
            out.append(make_case(
                "enqueue", slot, scene, name, kw, blocks, notes=note,
                grid_of="min(numba, triton lane, triton program)",
                enqueue=enq, label=label))
    return out


def scene_cases(slot, scene, note, r2_pin, mb_pin):
    """Every experiment's cases for one scene, in Numba benchmark order."""
    out = []
    lin_pin = pinned(MODE_CONFIGS["seq"])
    for name, kw in MODE_CONFIGS.items():
        if name == "seq_half":
            blocks, grid_of = lin_pin // 2, "lin // 2"
        else:
            blocks, grid_of = pinned(kw), triton_ff._KERNELS[_key(kw)].__name__
        out.append(make_case("modes", slot, scene, name, kw, blocks,
                             notes=note, grid_of=grid_of))
    for which in ("a", "b"):
        out.append(make_case(f"seq_{which}", slot, scene, "seq",
                             MODE_CONFIGS["seq"], lin_pin, notes=note,
                             which=which))
    out.append(make_mb_case(slot, scene, mb_pin, notes=note))
    for name in ("multi", "multi8"):
        out.append(make_case("blocks_none", slot, scene, name,
                             MODE_CONFIGS[name], None, notes=note))
    for name, kw in R2_CONFIGS.items():
        out.append(make_case("radius2", slot, scene, name, kw, r2_pin,
                             notes=note))
    out += enqueue_cases(slot, scene, note)
    return out


def build(quick):
    """All cases plus the meta block, in scene-grouped order."""
    scene_list = QUICK_SCENES if quick else nb_bench.SCENES
    assert [s[0] for s in nb_r2.SCENES] == [s[0] for s in nb_bench.SCENES]
    slot = SceneSlot({n: b for n, b, _ in scene_list})

    r2_caps = {"lin8": caps({"connectivity": 8}),
               "lin8r2": caps({"connectivity": 8, "radius": 2})}
    r2_pin = min(min(c.values()) for c in r2_caps.values())
    mb_c = mb_caps()
    mb_pin = min(mb_c.values())
    cases = []
    for sname, _, note in scene_list:
        cases += scene_cases(slot, sname, note, r2_pin, mb_pin)
    slot.release()  # the cases rebuild each scene when they run

    coop = {}
    for name, kw in {**MODE_CONFIGS, **R2_CONFIGS}.items():
        coop[triton_ff._KERNELS[_key(kw)].__name__] = caps(kw)
    meta = {
        "caps": [] if quick else CAPS,
        "method_notes": METHOD_NOTES,
        "quick": quick,
        "scope": SCOPE,
        "tpb": TPB,
        "coop_max_by_kernel": coop,
        "modes": {"configs": MODE_CONFIGS,
                  "seq_half_blocks": pinned(MODE_CONFIGS["seq"]) // 2},
        "radius2": {"pinned_blocks": r2_pin, "coop_max_by_kernel": r2_caps},
        "packing_tax": {"pinned_blocks": mb_pin, "coop_max": mb_c},
        "enqueue": enqueue_meta(),
        "bandwidth_model": bandwidth.MODEL_NOTE,
        "derived": (
            "per backend, from the kernel_ms medians: speedup_multi_vs_seq = "
            "seq / multi; ideal_max = max(seq_a, seq_b); multi_vs_ideal = "
            "multi / ideal_max; xy_vs_lin = multi / multi_xy; "
            "multi_overhead_pct = 100 * (multi - bare) / bare; "
            "packing_tax_pct = 100 * (seq_a - mb_a) / mb_a; "
            "r2_multi_vs_conn8 = multi8 / multi8r2 (radius2 rows); "
            "first-translation cost = triton kernel_ms of the program row / "
            "triton kernel_ms of the per_lane row of the same enqueue pair "
            "(enqueue rows; each row also times Numba itself, so each has "
            "its own speedup_kernel against Numba); model "
            "GB/s = info.<backend>_model_bytes / (kernel_ms * 1e6), against "
            "meta.peak_gb_s of the same backend's copy probe"),
        "pixels": "scene width * height; info.filled is both blobs' size",
    }
    return cases, meta


def enqueue_meta():
    """Per enqueue config: the pinned grid, and per Triton binary its
    capacity, registers and CTA barrier count."""
    out = {"labels": ENQ_LABELS, "configs": ENQ_CONFIGS, "by_config": {}}
    for name, kw in ENQ_CONFIGS.items():
        k = _kernel_kw(kw)
        per = {}
        for enq in ENQ_LABELS:
            ck = triton_ff._warmup(*_key(kw), TPB, enq)
            per[enq] = {
                "cap": triton_ff.max_blocks(threads_per_block=TPB,
                                            enqueue=enq, **k),
                "n_regs": triton_ff.kernel_info(threads_per_block=TPB,
                                                enqueue=enq, **k)["n_regs"],
                "ptx_bar_sync": twin_sass.ptx_barriers(ck.asm["ptx"]),
            }
        out["by_config"][name] = {
            "kernel": triton_ff._KERNELS[_key(kw)].__name__,
            "pinned_blocks": enq_pinned(kw),
            "numba_cap": numba_ff.max_blocks(threads_per_block=TPB, **k),
            "triton": per}
    return out


def measure_peaks(quick):
    """Both copy probes, back to back, after the GPU is at boost.

    Always 256 MiB per buffer: at 16 MiB both buffers fit the 32 MB L2 and
    the copy is short enough for Numba's slower Python launch to land
    inside its event bracket, which made the quick peaks look like a 4x
    Triton advantage. --quick only cuts the copy count (and has no
    spin-up), so its peaks are a smoke test of the probes, not a figure.
    """
    n_bytes = 256 * 2 ** 20
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
    label = (" (quick: no spin-up, 2 copies; a smoke test, not a peak)"
             if args.quick else "")
    print(f"copy peak: numba {peaks['numba']['gb_s']:.1f} GB/s | "
          f"triton {peaks['triton']['gb_s']:.1f} GB/s{label}")
    cases, meta = build(args.quick)
    meta["peak_gb_s"] = {k: v["gb_s"] for k, v in peaks.items()}
    meta["peak_runs_gb_s"] = {k: v["runs_gb_s"] for k, v in peaks.items()}
    print(f"{len(cases)} cases, repeats={repeats}")
    doc = run_cases(CHAPTER, cases, repeats=repeats, meta=meta,
                    write=not args.quick, spin_seconds=spin)
    add_min_ratios(doc)
    print_ratios(doc)
    bad = [r for r in doc["rows"]
           if "error" in r or not r.get("outputs_equal", False)]
    print(f"{len(doc['rows'])} rows, {len(bad)} with an error or a mismatch")
    return doc


def add_min_ratios(doc):
    """Add speedup_kernel_min (best run vs best run, benchmark.py's '(min)'
    column) to every measured row, and rewrite the JSON the harness wrote
    so the file carries it too."""
    for row in doc["rows"]:
        if "error" in row:
            continue
        row["speedup_kernel_min"] = (row["numba"]["kernel_ms"]["min"]
                                     / row["triton"]["kernel_ms"]["min"])
    path = doc.pop("path", None)
    if path:
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
        doc["path"] = path


def print_ratios(doc):
    """Both kernel ratios per row; 'split' marks rows whose median and
    best-vs-best ratios fall on opposite sides of 1 (noise, not a
    difference)."""
    print("\nspeedup_kernel (median) vs speedup_kernel_min (best-vs-best), "
          ">1: Triton faster")
    for row in doc["rows"]:
        name = (f"{row['experiment']:12s} {row['scene']:16s} "
                f"{row['config'].get('config', ''):9s}")
        if "error" in row:
            print(f"  {name} error")
            continue
        med, best = row["speedup_kernel"], row["speedup_kernel_min"]
        mark = "split" if (med - 1) * (best - 1) < 0 else ""
        print(f"  {name} x{med:6.3f}  (min) x{best:6.3f}  {mark}")


if __name__ == "__main__":
    main()
