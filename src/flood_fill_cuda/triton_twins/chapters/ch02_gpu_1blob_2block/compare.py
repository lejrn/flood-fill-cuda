"""Numba vs Triton on chapter 2's own benchmark (benchmarks/benchmark.py).

Cases mirror the chapter's benchmark, with the scene list, sweep axes and
placement scenes imported from it:

  scenes     every SCENES row: the single-block v2 baseline (chapter 1's
             "spill" kernel at tpb 256, cross-imported as the benchmark
             does) and {split, global, dirsplit} x {instrumented, bare}
             at tpb 256, 2 blocks / 2 programs
  tpb_sweep  TPB_SWEEP {64, 128, 256, 512} x 3 kernels on sq_2000_center
  placement  on PLACEMENT_SCENES, the benchmark's configurations:
             - v2 1x1024 (1 SM), both backends at 1024;
             - v2 1x768 has no Triton twin (768 is not a power of 2): both
               backends run 1x512 instead, the twin's pinned worker width;
             - pinned same SM and spread as the chapter runs them: Numba
               2 x 768 threads, the twin 2 x 512 lanes. Not like-for-like,
               so these rows carry comparable=False;
             - pinned spread at 2 x 512 on both backends: the matched row.

Not mirrored: the benchmark's @njit rows (CPU, no GPU backend).

Every run is reduced to its deterministic outputs right after the driver
returns: scalars plus BLAKE2 digests of img / visited / depth (and the
traces / owner map where they are deterministic). same() compares those, so
equality stays exact while no two 36M-pixel results are alive at once.

Run:
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch02_gpu_1blob_2block.compare [--quick] [--repeats N]
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import contextlib
import dataclasses
import hashlib
from dataclasses import dataclass, field

import numpy as np

from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.flood_fill import (
    flood_fill as v2_numba_flood_fill,
)
from flood_fill_cuda.chapters.ch02_gpu_1blob_2block import flood_fill as nff
from flood_fill_cuda.chapters.ch02_gpu_1blob_2block import scenes
from flood_fill_cuda.chapters.ch02_gpu_1blob_2block.benchmarks import (
    benchmark as nbench,
)
from flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare import (
    numba_resources as v2_numba_resources,
)
from flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.flood_fill import (
    compiled_kernel as v2_compiled_kernel,
    flood_fill as v2_triton_flood_fill,
)
from flood_fill_cuda.triton_twins.compare.harness import run_cases, Case
from flood_fill_cuda.triton_twins.runtime import device_info, kernel_resources

from . import flood_fill as tff

CHAPTER = "ch02_gpu_1blob_2block"
TPB = 256  # the benchmark's default tpb for the scene rows
V2_VARIANT = "spill"  # the benchmark's single-block baseline
NUMBA_PINNED_TPB = nff.PINNED_TPB  # 768
V2_PLACEMENT_TPBS = (1024, tff.PINNED_TPB)  # benchmark: (768, 1024)
TIMING_FIELDS = {"alloc_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms"}

QUICK_SCENES = [
    ("sq_128_center", lambda: scenes.square_scene(128, 128, 64, 64),
     "quick smoke scene"),
    ("seam_serpentine_64", lambda: scenes.seam_serpentine_scene(64, 64),
     "quick smoke scene"),
]


@dataclass
class Slim:
    """What a run leaves behind: timings, deterministic outputs, and the
    schedule-dependent observations reported (never compared) in info."""
    kernel_ms: float
    total_ms: float
    alloc_ms: float
    h2d_ms: float
    d2h_ms: float
    det: dict = field(default_factory=dict)
    obs: dict = field(default_factory=dict)


def _digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.blake2b(digest_size=16)
    h.update(str((a.dtype.str, a.shape)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _ascii_dashes(text):
    return text.replace("\u2014", "-").replace("\u2013", "-")


def _slim(r, kernel, bare):
    det = {
        "img": _digest(r.img),
        "visited": _digest(r.visited),
        "depth": _digest(r.depth),
        "levels": r.levels,
        "filled": r.filled,
    }
    instrumented = not bare and kernel != "pinned"
    if instrumented:
        det.update(
            level_sizes=_digest(r.level_sizes.astype(np.int64)),
            level_trace_truncated=r.level_trace_truncated,
            peak_level=r.peak_level,
            peak_occupancy=r.peak_occupancy,
            processed=r.processed,
            # deterministic on the bipartite 4-connected grid (see README)
            cas_attempts=r.cas_attempts,
        )
        if kernel in ("split", "global"):
            det.update(
                processed_b0=r.processed_b0,
                processed_b1=r.processed_b1,
                level_sizes_per_block=_digest(
                    r.level_sizes_per_block.astype(np.int64)),
                thread_util_pct=r.thread_util_pct,
                warp_engagement_pct=r.warp_engagement_pct,
                lane_efficiency_pct=r.lane_efficiency_pct,
            )
        if kernel == "split":
            det["owner"] = _digest(r.owner)
        elif kernel == "global":
            reached = r.visited == 1
            det["owner_census"] = np.bincount(
                r.owner[reached].astype(np.int64), minlength=2).tolist()
    obs = {
        "sm_ids": [r.sm_id_b0, r.sm_id_b1],
        "processed_b0": r.processed_b0,
        "processed_b1": r.processed_b1,
        "balance_pct": r.balance_pct,
        "peak_level": r.peak_level,
    }
    if kernel == "split" and instrumented:
        obs.update(spilled_b0=r.spilled_b0, spilled_b1=r.spilled_b1,
                   peak_spill_window=r.peak_spill_window,
                   inbox_to_b0=r.inbox_to_b0, inbox_to_b1=r.inbox_to_b1)
    return Slim(r.kernel_ms, r.total_ms, r.alloc_ms, r.h2d_ms, r.d2h_ms,
                det, obs)


def _slim_v2(r):
    """Chapter 1's result: every field is deterministic (ch01 twin's
    compare compares them all), arrays by digest."""
    det = {}
    for f in dataclasses.fields(r):
        if f.name in TIMING_FIELDS:
            continue
        v = getattr(r, f.name)
        det[f.name] = _digest(v) if isinstance(v, np.ndarray) else v
    return Slim(r.kernel_ms, r.total_ms, r.alloc_ms, r.h2d_ms, r.d2h_ms, det)


def _same(n, t):
    for key in n.det:
        if n.det[key] != t.det.get(key):
            return False, f"{key}: numba {n.det[key]!r} vs triton {t.det.get(key)!r}"
    return True, ""


class _SceneCache:
    """One scene image alive at a time: consecutive cases share it, and a
    new scene releases the previous one (host RAM is 6 GB)."""

    def __init__(self):
        self.name = None
        self.value = None

    def get(self, name, builder):
        if self.name != name:
            self.name, self.value = None, None
            self.value = builder()
            self.name = name
        return self.value


_cache = _SceneCache()
_pixels = {}


def _scene_pixels(name, builder):
    if name not in _pixels:
        img, _, _ = _cache.get(name, builder)
        _pixels[name] = int(img.shape[0] * img.shape[1])
    return _pixels[name]


@contextlib.contextmanager
def _numba_pinned_tpb(tpb):
    """Run Numba's pinned experiment at another block size. Its driver
    checks threads_per_block against the module constant PINNED_TPB (and
    warms up with it), so the constant is swapped for the call and
    restored. Valid for placement="spread" only (see meta)."""
    old = nff.PINNED_TPB
    nff.PINNED_TPB = tpb
    try:
        yield
    finally:
        nff.PINNED_TPB = old


def _numba_resources(kernel, bare, tpb):
    fn = (nff.dual_block_pinned_kernel if kernel == "pinned"
          else nff._KERNELS[(kernel, bare)])
    try:
        ov = next(iter(fn.overloads.values()))
        return {"regs_per_thread": int(ov.regs_per_thread),
                "local_mem_per_thread": int(ov.local_mem_per_thread),
                "shared_mem_per_block": int(ov.shared_mem_per_block),
                "max_cooperative_blocks": int(
                    ov.max_cooperative_grid_blocks(tpb))}
    except Exception as exc:  # recorded, not hidden
        return {"error": f"{type(exc).__name__}: {exc}"}


def _make_case(experiment, scene_name, builder, note, kernel, bare,
               tpb_numba, tpb_triton, placement=None, label=None):
    numba_pinned_override = (kernel == "pinned"
                             and tpb_numba != NUMBA_PINNED_TPB)
    matched = tpb_numba == tpb_triton

    def run_numba():
        img, sx, sy = _cache.get(scene_name, builder)
        ctx = (_numba_pinned_tpb(tpb_numba) if numba_pinned_override
               else contextlib.nullcontext())
        with ctx:
            r = nff.flood_fill(img, sx, sy, threads_per_block=tpb_numba,
                               kernel=kernel, bare=bare, placement=placement)
        return _slim(r, kernel, bare)

    def run_triton():
        img, sx, sy = _cache.get(scene_name, builder)
        return _slim(tff.flood_fill(img, sx, sy, threads_per_block=tpb_triton,
                                    kernel=kernel, bare=bare,
                                    placement=placement), kernel, bare)

    def info(n, t):
        sm = device_info().sm_count
        numba_grid = 2 * sm if (kernel == "pinned"
                                and placement == "same_sm") else 2
        return {
            "filled": n.det["filled"],
            "levels": n.det["levels"],
            "numba": {**n.obs, "alloc_ms": n.alloc_ms, "h2d_ms": n.h2d_ms,
                      "d2h_ms": n.d2h_ms, "grid": numba_grid,
                      "tpb": tpb_numba,
                      "resources": _numba_resources(kernel, bare, tpb_numba)},
            "triton": {**t.obs, "alloc_ms": t.alloc_ms, "h2d_ms": t.h2d_ms,
                       "d2h_ms": t.d2h_ms,
                       "grid": tff.launch_grid(kernel, placement, tpb_triton,
                                               bare),
                       "tpb": tpb_triton,
                       "num_warps": tpb_triton // 32,
                       "resources": kernel_resources(
                           tff.compiled_kernel(kernel, bare, tpb_triton))},
        }

    config = {"kernel": kernel, "bare": bare, "tpb": tpb_numba,
              "placement": placement}
    if label:
        config["label"] = label
    notes = _ascii_dashes(note)
    if not matched:
        config["tpb_triton"] = tpb_triton
        notes = (f"{notes}; Numba pins 2 x {tpb_numba} threads, the twin "
                 f"2 x {tpb_triton} lanes (768 is not a power of 2): not "
                 f"like-for-like, speedup is not a backend ratio")
    elif numba_pinned_override:
        notes = (f"{notes}; matched: Numba's pinned kernel at 2 x "
                 f"{tpb_numba} (PINNED_TPB swapped for the call)")
    return Case(experiment=experiment, scene=scene_name, config=config,
                run_numba=run_numba, run_triton=run_triton, same=_same,
                pixels=_scene_pixels(scene_name, builder), info=info,
                notes=notes, extra={"config_matched": matched},
                comparable=matched)


def _make_v2_case(experiment, scene_name, builder, note, tpb, label=None):
    """Chapter 1's single-block spill kernel on both backends (the
    benchmark cross-imports it as the "what does the 2nd block buy?"
    baseline)."""

    def run_numba():
        img, sx, sy = _cache.get(scene_name, builder)
        return _slim_v2(v2_numba_flood_fill(img, sx, sy, threads_per_block=tpb,
                                            variant=V2_VARIANT))

    def run_triton():
        img, sx, sy = _cache.get(scene_name, builder)
        return _slim_v2(v2_triton_flood_fill(img, sx, sy,
                                             threads_per_block=tpb,
                                             variant=V2_VARIANT))

    def info(n, t):
        return {
            "filled": n.det["filled"],
            "levels": n.det["levels"],
            "numba": {"alloc_ms": n.alloc_ms, "h2d_ms": n.h2d_ms,
                      "d2h_ms": n.d2h_ms, "grid": 1, "tpb": tpb,
                      "resources": v2_numba_resources(V2_VARIANT)},
            "triton": {"alloc_ms": t.alloc_ms, "h2d_ms": t.h2d_ms,
                       "d2h_ms": t.d2h_ms, "grid": 1, "tpb": tpb,
                       "num_warps": tpb // 32,
                       "resources": kernel_resources(
                           v2_compiled_kernel(V2_VARIANT, tpb))},
        }

    config = {"kernel": "v2", "variant": V2_VARIANT, "blocks": 1,
              "tpb": tpb}
    if label:
        config["label"] = label
    return Case(experiment=experiment, scene=scene_name, config=config,
                run_numba=run_numba, run_triton=run_triton, same=_same,
                pixels=_scene_pixels(scene_name, builder), info=info,
                notes=_ascii_dashes(note) + "; chapter 1's v2 spill kernel",
                extra={"config_matched": True})


def build_cases(quick=False):
    scene_rows = QUICK_SCENES if quick else nbench.SCENES
    lookup = {name: (builder, note) for name, builder, note in nbench.SCENES}
    cases = []
    for name, builder, note in scene_rows:
        cases.append(_make_v2_case("scenes", name, builder, note, TPB))
        for kernel in nbench.KERNELS:
            for bare in (False, True):
                cases.append(_make_case("scenes", name, builder, note, kernel,
                                        bare, TPB, TPB))
    sweep_name = "sq_128_center" if quick else "sq_2000_center"
    sweep_builder, sweep_note = ((QUICK_SCENES[0][1], QUICK_SCENES[0][2])
                                 if quick else lookup[sweep_name])
    for tpb in nbench.TPB_SWEEP:
        for kernel in nbench.KERNELS:
            cases.append(_make_case("tpb_sweep", sweep_name, sweep_builder,
                                    sweep_note, kernel, False, tpb, tpb))
    placement_scenes = ([(QUICK_SCENES[0][0],) + QUICK_SCENES[0][1:]] if quick
                        else [(n,) + lookup[n] for n in nbench.PLACEMENT_SCENES])
    for name, builder, note in placement_scenes:
        for tpb in V2_PLACEMENT_TPBS:
            label = f"v2 1x{tpb} (1 SM)"
            if tpb != 1024:
                label += " [stands in for v2 1x768]"
            cases.append(_make_v2_case("placement", name, builder, note, tpb,
                                       label))
        for placement, where in (("same_sm", "same SM"), ("spread", "spread")):
            cases.append(_make_case(
                "placement", name, builder, note, "pinned", False,
                NUMBA_PINNED_TPB, tff.PINNED_TPB, placement,
                f"pinned 2x{NUMBA_PINNED_TPB} {where} (twin 2x{tff.PINNED_TPB})"))
        cases.append(_make_case(
            "placement", name, builder, note, "pinned", False,
            tff.PINNED_TPB, tff.PINNED_TPB, "spread",
            f"pinned 2x{tff.PINNED_TPB} spread (matched)"))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="tiny scenes, 1 repeat, no JSON (smoke test only)")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per case (default "
                         f"{nbench.GPU_REPEATS}, the benchmark's GPU_REPEATS)")
    args = ap.parse_args(argv)
    repeats = args.repeats or (1 if args.quick else nbench.GPU_REPEATS)
    cases = build_cases(quick=args.quick)
    meta = {
        "mirrors": "chapters/ch02_gpu_1blob_2block/benchmarks/benchmark.py",
        "experiments": {
            "scenes": "SCENES x (v2 single-block spill at tpb 256, plus "
                      "{split, global, dirsplit} x {instrumented, bare} at "
                      "tpb 256)",
            "tpb_sweep": "TPB_SWEEP x 3 kernels on sq_2000_center",
            "placement": "on PLACEMENT_SCENES: v2 1x1024, v2 1x512 (for "
                         "1x768), pinned same_sm / spread as the chapter "
                         "runs them (Numba 768, twin 512), pinned spread "
                         "matched at 2 x 512",
        },
        "not_mirrored": ["@njit CPU rows (no GPU backend to compare)"],
        "placement_matching": [
            "Rows labeled 'pinned 2x768 ... (twin 2x512)' compare 1,536 "
            "Numba threads with 1,024 Triton lanes: comparable=false, "
            "config_matched=false, their speedup is not a backend ratio.",
            "'pinned 2x512 spread (matched)': Numba's pinned kernel run at "
            "512 threads (PINNED_TPB swapped for the call) vs the twin at "
            "512 lanes: the like-for-like spread row.",
            "same_sm has no matched row: Numba's pinned kernel at 2 x 512 is "
            "not a valid same_sm experiment. At max_registers 40, three "
            "512-thread blocks fit per SM, so the 2 * sm_count launch does "
            "not put exactly two on the chosen SM, and the kernel has no "
            "rank < 2 guard: a third block would join the BFS and break "
            "the 2-worker barrier.",
            "v2 1x768 has no Triton twin (not a power of 2): both backends "
            "run 1x512 instead, the twin's pinned worker width.",
        ],
        "caps": [] if not args.quick else [
            "quick: scenes replaced by sq_128_center and seam_serpentine_64, "
            "tpb sweep and placement on sq_128_center"],
        "budget": "no scene capped: the largest case (sq_6000_center) "
                  "measured at ~1.45 GB host RSS; default mode ~8 min of "
                  "GPU time",
        "timing": "speedup_kernel is the backend comparison. total_ms adds "
                  "the host stacks: CuPy's pooled allocation (warm after "
                  "a case's first run) vs Numba's cuMemAlloc on every run, "
                  "and CuPy .set/.get vs Numba copies. On small scenes "
                  "speedup_total mostly measures that (quick sq_128_center "
                  "split: alloc 0.62 ms Numba vs 0.13 ms Triton).",
        "deviations": [
            "placement: Numba's experiment is 2 x 768 threads at "
            "max_registers 40, the twin's 2 x 512 lanes at maxnreg 64; both "
            "put exactly 2 programs per SM in same_sm mode",
            "split: the 8192-slot ring is shared memory in Numba, global "
            "scratch in Triton",
            "the twin's driver allocates an int32 grid_sync counter "
            "(split/global/dirsplit), zeroed in the h2d bracket",
        ],
        "same": "dual kernels: img, visited, depth, levels, filled; "
                "instrumented: level_sizes, peak_level, peak_occupancy, "
                "processed, cas_attempts; split/global: per-program "
                "processed, trace and utilization; split: owner map; "
                "global: owner census. v2: every result field. Arrays "
                "compared by BLAKE2 digest.",
        "triton_tpb_rule": "num_warps = tpb // 32",
    }
    run_cases(CHAPTER, cases, repeats=repeats, meta=meta,
              write=not args.quick, spin_seconds=0 if args.quick else 8.0)


if __name__ == "__main__":
    main()
