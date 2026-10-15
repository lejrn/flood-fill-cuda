"""Experiments E1-E10 of the ch08 design (section 14), one session.

Every experiment is an interleaved A/B with the benchmark's policy: 9
rounds, order reversed on odd rounds; before every run the image is
restored, the mask packed if the arm needs it, a 96 MB read flushes the
L2, a ~2 ms GPU sleep kernel is queued, and the arm is enqueued with
CUDA events. A 1 s GPU-bound respin precedes each experiment on each
scene, and the clock sampler's window is kept per experiment.

  E1  pack store        ch06 pack_kernel | pack_tiles direct | smem |
                        tile-major (design 3's store, reference only)
  E2  closure exits     ch07 loop | + frontier exit | + last-comp exit
  E3  K2 registers/grid max_registers None/96/80 x grid 384/768/1536
  E4  neighbour lookup  scan | index
  E5  paint             fast path on/off x 128/256 threads
  E6  seed dial         L = 0, 32, 16, 8, 4, 2, 1 (+ counters)
  E7  union timing      deferred | inline (coll_capacity 0)
  E8  counter cost      bare | instrumented (waves phase)
  E9  valve round       128, 64, 32, 16, off, at L8 (+ rounds, valve
                        seeds); the rule looks only at scenes where some
                        arm fires the valve (elsewhere the arms run the
                        same code); R8 records the rounds the main scenes
                        need at L8, which a valve must stay above
  E10 waves launch      256 | 128 threads per block
  E4b lookup x K2       follow-up: scan at E3's pick vs index at three
                        K2 register/grid settings (index spills at 80)
  E13 lookup x K3 regs  scan | index | hybrid (dense_min 4, 8, 16), each
                        at the compiler's registers and at an 80 cap
                        (80 keeps 3 cooperative blocks of 256 per SM)

The decision rules are applied at the end and written to the JSON.

Run:
  uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.experiments --images DIR
Writes experiments_<stamp>.json to results/ch08_gpu_nblob_tiles/
benchmark_results/.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import gc
import json
import statistics
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from .. import flood_fill as ff
from .. import scenes as S8
from ..kernels import (gpu_sleep_kernel, get_pack_kernel, pack_shape,
                       get_label_kernel, get_paint_kernel, get_waves_kernel,
                       MODE_RGB)
from ...ch05_gpu_nblob_nblock import scenes as S5
from ...ch06_gpu_nblob_runs.recolor import RunRecolor
from ...ch06_gpu_nblob_runs.kernels import pack_kernel as ch06_pack
from ...ch07_gpu_1blob_tiles.benchmarks import worst_cases as W7
from . import benchmark as B

warnings.simplefilter("ignore", NumbaPerformanceWarning)

ROUNDS = 9


def _summ(samples):
    keys = [k for k in samples[0] if k != "total"]
    return {"total_ms": statistics.median(s["total"] for s in samples),
            "total_ms_min": min(s["total"] for s in samples),
            "phase_ms": {k: statistics.median(s[k] for s in samples)
                         for k in keys}}


def interleave(arms, d_img, d_pristine, sampler):
    """arms: {name: (prep or None, run)}; run(d_img) -> (names, events).
    Returns {name: summary} plus per-round ratios to the first arm."""
    flush, sink = B._SPIN["flush"], B._SPIN["sink"]
    names_ = list(arms)
    samples = {n: [] for n in names_}
    for n in names_:                 # untimed: compile and warm each arm
        prep, run = arms[n]
        d_img.copy_to_device(d_pristine)
        if prep is not None:
            prep(d_img)
        run(d_img)
        cuda.synchronize()
    t0 = time.time()
    for i in range(ROUNDS):
        order = names_ if i % 2 == 0 else list(reversed(names_))
        for n in order:
            prep, run = arms[n]
            d_img.copy_to_device(d_pristine)
            if prep is not None:
                prep(d_img)
            B._l2_flush[512, 256](flush, sink)
            cuda.synchronize()
            gpu_sleep_kernel[1, 32](B.SLEEP_NAPS)
            ph, events = run(d_img)
            events[-1].synchronize()
            out = {p: cuda.event_elapsed_time(events[j], events[j + 1])
                   for j, p in enumerate(ph)}
            out["total"] = cuda.event_elapsed_time(events[0], events[-1])
            samples[n].append(out)
    t1 = time.time()
    res = {n: _summ(samples[n]) for n in names_}
    base = names_[0]
    for n in names_[1:]:
        r = [a["total"] / b["total"]
             for a, b in zip(samples[n], samples[base])]
        res[n]["ratio_to_first_median"] = statistics.median(r)
        res[n]["ratio_to_first_range"] = [min(r), max(r)]
    res["_clock"] = sampler.window(t0, t1)
    return res


def _respin(eng, d_img):
    B._spin(eng, d_img, B.SCENE_SPIN_SECONDS)


def _ch08_arm(eng, contract="rgb", attrs=None, **run_kw):
    """An arm on a (possibly shared) engine. attrs: engine attributes the
    arm sets before each run (a kernel twin, a launch shape), so arms
    that differ only in one kernel share one set of 81 Mpx buffers."""
    prep = eng.pack if contract == "mask" else None

    def run(d):
        for k_, v_ in (attrs or {}).items():
            setattr(eng, k_, v_)
        return eng.run(d, contract, **run_kw)
    return (prep, run)


def _single(kernel_call):
    """Wrap one launch as a one-phase run."""
    def run(d):
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        kernel_call(d)
        e1.record()
        return ["kernel"], [e0, e1]
    return run


def _instrumented_counters(img, **kw):
    """One instrumented run off the clock: the counters of a config."""
    eng = ff.TileWaves(img.shape[0], img.shape[1], instrumented=True, **kw)
    d = cuda.to_device(img)
    eng.run(d, "rgb")
    cuda.synchronize()
    c = eng.read_counters()
    del eng, d
    B._release()
    return {k: c[k] for k in ("rounds", "seeds", "seeds_lattice", "claims",
                              "union_calls", "union_done", "collisions",
                              "coll_inline", "roots", "valve_round",
                              "valve_seeds", "sig_tests", "ksteps", "tasks",
                              "dense_tasks", "max_ring_tile",
                              "close_iters", "exit_last", "exit_frontier",
                              "paint_uniform", "paint_reclosed_tiles",
                              "paint_single_interior")}


# ------------------------------------------------------------ the E's

def e1_pack(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    wpr = (h + 31) // 32
    r6 = RunRecolor(w, h, run_capacity=8192)
    mask = cuda.device_array((w, wpr), np.uint32)
    n_tiles = ((w + 31) // 32) * wpr
    tm = cuda.device_array((n_tiles * 32, 1), np.uint32)
    grid, tpb = pack_shape(w, wpr)
    arms = {
        "ch06_pack": (None, _single(lambda d: ch06_pack[r6.pack_grid](
            d, mask))),
        "tiles_direct": (None, _single(lambda d: get_pack_kernel(
            "direct")[grid, tpb](d, mask))),
        "tiles_smem": (None, _single(lambda d: get_pack_kernel(
            "smem")[grid, tpb](d, mask))),
        "tiles_tilemajor_ref": (None, _single(lambda d: get_pack_kernel(
            "tilemajor")[grid, tpb](d, tm))),
    }
    eng = ff.TileWaves(w, h)
    _respin(eng, d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    for n in arms:
        res[n]["gb_s"] = 3 * w * h / (res[n]["total_ms"] * 1e6)
    del r6, mask, tm, eng
    return res


def e2_exits(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h)
    arms = {f"exits_{e}": _ch08_arm(eng, "mask", attrs={
        "label_k": get_label_kernel(8, False, False, None, False, e)})
        for e in (0, 1, 2)}
    _respin(eng, d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    for e in (0, 1, 2):
        res[f"exits_{e}"]["counters"] = _instrumented_counters(
            img, label_exits=e)
    return res


def e3_label(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h)
    arms = {}
    for regs in (None, 96, 80):
        for g in (384, 768, 1536):
            arms[f"regs{regs or 'auto'}_g{g}"] = _ch08_arm(eng, "mask", attrs={
                "label_k": get_label_kernel(8, False, False, regs),
                "label_grid": (g, 128)})
    _respin(eng, d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    res["_resources"] = {str(r): ff.kernel_resources(
        8, label_max_registers=r)["label"] for r in (None, 96, 80)}
    return res


def e4_lookup(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    engs = {"scan": ff.TileWaves(w, h),
            "index": ff.TileWaves(w, h, lookup="index")}
    arms = {n: _ch08_arm(e) for n, e in engs.items()}
    _respin(engs["scan"], d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    for n in engs:
        res[n]["counters"] = _instrumented_counters(img, lookup=n)
    return res


def e5_paint(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h)
    arms = {}
    for fast in (True, False):
        for tpb in (128, 256):
            arms[f"fast{int(fast)}_tpb{tpb}"] = _ch08_arm(eng, "mask", attrs={
                "paint_k": get_paint_kernel(8, MODE_RGB, fast, False, False),
                "paint_grid": (768 * 128 // tpb, tpb)})
    _respin(eng, d_img)
    return interleave(arms, d_img, d_pr, sampler)


def e6_dial(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h)
    arms = {f"L{L}": _ch08_arm(eng, lattice=L)
            for L in (8, 0, 32, 16, 4, 2, 1)}
    _respin(eng, d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    for L in (8, 0, 32, 16, 4, 2, 1):
        res[f"L{L}"]["counters"] = _instrumented_counters(img, lattice=L)
    return res


def e7_union(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h)
    cap = eng.coll_capacity
    arms = {"deferred": _ch08_arm(eng, attrs={"coll_capacity": cap}),
            "inline": _ch08_arm(eng, attrs={"coll_capacity": 0})}
    _respin(eng, d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    eng.coll_capacity = cap
    return res


def e8_counters(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    engs = {"bare": ff.TileWaves(w, h),
            "instrumented": ff.TileWaves(w, h, instrumented=True)}
    arms = {n: _ch08_arm(e, "mask") for n, e in engs.items()}
    _respin(engs["bare"], d_img)
    return interleave(arms, d_img, d_pr, sampler)


E9_VALVES = (128, 64, 32, 16, 0)


def e9_valve(img, d_img, d_pr, sampler):
    """The valve at the default lattice and lookup (L8)."""
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h)
    arms = {f"valve{v or 'off'}": _ch08_arm(eng, valve_round=v)
            for v in E9_VALVES}
    _respin(eng, d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    for v in E9_VALVES:
        res[f"valve{v or 'off'}"]["counters"] = _instrumented_counters(
            img, valve_round=v)
    return res


def r8_rounds(img, d_img, d_pr, sampler):
    """Counters only (no timing): the rounds a scene needs at L8."""
    return {"_clock": {}, "L8": {"counters": _instrumented_counters(
        img, valve_round=0)}}


E13_ARMS = {   # name: (lookup, dense_min, K3 max_registers)
    "scan": ("scan", 8, None),
    "scan_r80": ("scan", 8, 80),
    "index": ("index", 8, None),
    "index_r80": ("index", 8, 80),
    "hybrid8": ("hybrid", 8, None),
    "hybrid8_r80": ("hybrid", 8, 80),
    "hybrid4_r80": ("hybrid", 4, 80),
    "hybrid16_r80": ("hybrid", 16, 80),
}


def e13_lookup(img, d_img, d_pr, sampler):
    """K3's neighbour lookup and register cap. One engine per K2 (scan
    and hybrid share the scan K2; index writes its byte index), the K3
    twin, its cooperative block count and dense_min set per arm."""
    w, h = img.shape[:2]
    engs = {"scan": ff.TileWaves(w, h, lookup="hybrid"),
            "index": ff.TileWaves(w, h, lookup="index")}
    arms = {}
    meta = {}
    for name, (lk, dm, regs) in E13_ARMS.items():
        kern = get_waves_kernel(8, False, False, lk == "index",
                                lk == "hybrid", regs)
        blocks = ff.max_blocks(256, 8, False, False, lk, regs)
        eng = engs["index" if lk == "index" else "scan"]
        arms[name] = _ch08_arm(eng, attrs={"waves_k": kern,
                                           "waves_blocks": blocks,
                                           "dense_min": dm})
        meta[name] = {"blocks": blocks, "resources": ff.kernel_resources(
            8, lookup=lk, waves_max_registers=regs)["waves"]}
    _respin(engs["scan"], d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    for name, (lk, dm, regs) in E13_ARMS.items():
        res[name].update(meta[name])
        res[name]["counters"] = _instrumented_counters(
            img, lookup=lk, dense_min=dm)
    return res


def e10_waves(img, d_img, d_pr, sampler):
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h)
    blocks = {tpb: ff.max_blocks(tpb, 8, False, False) for tpb in (256, 128)}
    arms = {f"tpb{tpb}": _ch08_arm(eng, "mask", attrs={
        "waves_tpb": tpb, "waves_blocks": blocks[tpb]}) for tpb in (256, 128)}
    _respin(eng, d_img)
    res = interleave(arms, d_img, d_pr, sampler)
    for tpb in (256, 128):
        res[f"tpb{tpb}"]["blocks"] = blocks[tpb]
    eng.waves_tpb, eng.waves_blocks = 256, blocks[256]
    return res


def e4b_lookup_k2(img, d_img, d_pr, sampler):
    """Follow-up to E3 x E4: E3 was run on the scan twin and picked an
    80-register cap at a 1536-block grid; the index twin's K2 spills 16
    bytes at that cap. Full rgb pipelines: scan at E3's pick against
    index at three K2 settings."""
    w, h = img.shape[:2]
    es = ff.TileWaves(w, h, lookup="scan")
    ei = ff.TileWaves(w, h, lookup="index")
    arms = {"scan_r80_g1536": _ch08_arm(es, attrs={
        "label_k": get_label_kernel(8, False, False, 80, False),
        "label_grid": (1536, 128)})}
    for regs, g in ((None, 768), (None, 1536), (80, 1536)):
        arms[f"index_r{regs or 'auto'}_g{g}"] = _ch08_arm(ei, attrs={
            "label_k": get_label_kernel(8, False, False, regs, True),
            "label_grid": (g, 128)})
    _respin(es, d_img)
    return interleave(arms, d_img, d_pr, sampler)


EXPERIMENTS = {"E1": e1_pack, "E2": e2_exits, "E3": e3_label,
               "E4": e4_lookup, "E5": e5_paint, "E6": e6_dial,
               "E7": e7_union, "E8": e8_counters, "E9": e9_valve,
               "E10": e10_waves, "E4b": e4b_lookup_k2, "E13": e13_lookup,
               "R8": r8_rounds}


def plan(pngs):
    """(scene, builder, [experiments]) - each image is built once."""
    blobs = lambda: S5.png_scene(pngs["input_blobs"])[0]  # noqa: E731

    def crop(side):
        def build():
            full = blobs()
            c0 = (full.shape[0] - side) // 2
            return np.ascontiguousarray(full[c0:c0 + side, c0:c0 + side])
        return build
    return [
        ("input_blobs", blobs,
         ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E10", "E4b",
          "E13", "R8"]),
        ("input_blocks", lambda: S5.png_scene(pngs["input_blocks"])[0],
         ["E2", "E3", "E4", "E5", "E6", "E8", "E4b", "E13", "R8"]),
        ("blob_grid_100",
         lambda: S5.blob_grid_scene(4000, 4000, 10, 10, 360, gap=40)[0],
         ["E1", "E5", "E6", "E7", "E8", "R8"]),
        ("random_4000",
         lambda: S5.random_blobs_scene(4000, 4000, 0.30, 0)[0],
         ["E3", "E4", "E6", "E4b", "E13", "R8"]),
        ("disk_r2000", lambda: S5.disk_scene(4200, 4200, 2000)[0],
         ["E1", "E6", "E7", "E10", "E13", "R8"]),
        ("serpentine_2048", lambda: S5.serpentine_scene(2048, 2048)[0],
         ["E2", "E4", "E6", "E9", "E10", "E4b", "E13", "R8"]),
        ("crop_2000", crop(2000), ["E1"]),
        ("crop_4000", crop(4000), ["E1"]),
        ("crop_6000", crop(6000), ["E1"]),
        ("zz_serp_1k", lambda: W7.zz_serpentine_scene()[0], ["E9", "E13"]),
        ("diag_snake_159", lambda: S8.diag_snake_scene(159), ["E9"]),
        ("maze_1k", lambda: W7.maze_scene()[0], ["E9", "E13"]),
        ("stair_decoy_19", lambda: S8.stair_scene(19), ["E9", "E13"]),
        ("stair_plain_19", lambda: S8.stair_scene(19, decoys=False),
         ["E9"]),
        ("zz_lattice_4k", lambda: W7.zz_lattice_scene()[0], ["E13"]),
    ]


# ----------------------------------------------------------- decisions

def _t(res, exp, scene, arm, phase=None):
    r = res[exp][scene][arm]
    return r["phase_ms"][phase] if phase else r["total_ms"]


def decide(res):
    out = {}
    # E1: the faster row-major arm on input_blobs (the others reported)
    if "E1" in res:
        sc = "input_blobs"
        d, s = _t(res, "E1", sc, "tiles_direct"), _t(res, "E1", sc,
                                                      "tiles_smem")
        out["E1_pack_variant"] = "direct" if d <= s else "smem"
    # E2: keep both exits if label time does not rise anywhere
    if "E2" in res:
        rise = [sc for sc in res["E2"]
                if _t(res, "E2", sc, "exits_2", "label")
                > _t(res, "E2", sc, "exits_0", "label")]
        out["E2_keep_both_exits"] = not rise
        out["E2_scenes_where_label_rose"] = rise
    # E3: fastest on input_blobs without losing over 3% elsewhere
    if "E3" in res:
        arms = [a for a in res["E3"]["input_blobs"] if not a.startswith("_")]
        ranked = sorted(arms, key=lambda a: _t(res, "E3", "input_blobs", a,
                                               "label"))
        pick = "regsauto_g768"
        for a in ranked:
            ok = all(_t(res, "E3", sc, a, "label")
                     <= 1.03 * _t(res, "E3", sc, "regsauto_g768", "label")
                     for sc in res["E3"] if sc != "input_blobs")
            if ok:
                pick = a
                break
        out["E3_label_config"] = pick
    # E4: index if input_blobs moves under +1% and 2 of 3 others gain 10%.
    # "Moves" is read with the design's pair statistic, the median of the
    # per-round ratios (13.4). Session 155420Z first decided with the
    # ratio of the medians (1.006, "index"); the per-round median was
    # 1.011, and the follow-up E4b gave 1.007 vs 1.013, where 0.025 ms of
    # pack noise (the same kernel in both arms) moved the ratio of
    # medians. The stricter reading is the rule now.
    if "E4" in res:
        blobs = res["E4"]["input_blobs"]["index"]["ratio_to_first_median"]
        gains = {sc: 1 - _t(res, "E4", sc, "index") / _t(res, "E4", sc,
                                                           "scan")
                 for sc in res["E4"] if sc != "input_blobs"}
        out["E4_input_blobs_index_over_scan"] = blobs
        out["E4_gains_elsewhere"] = gains
        out["E4_lookup"] = ("index" if blobs < 1.01 and
                            sum(g >= 0.10 for g in gains.values()) >= 2
                            else "scan")
    # E4b: E4's rule between scan at E3's pick and the best index arm
    if "E4b" in res:
        idx_arms = [a for a in res["E4b"]["input_blobs"]
                    if a.startswith("index")]
        best = min(idx_arms, key=lambda a: _t(res, "E4b", "input_blobs", a))
        base = "scan_r80_g1536"
        blobs = res["E4b"]["input_blobs"][best]["ratio_to_first_median"]
        gains = {sc: 1 - _t(res, "E4b", sc, best) / _t(res, "E4b", sc, base)
                 for sc in res["E4b"] if sc != "input_blobs"}
        out["E4b_best_index_arm"] = best
        out["E4b_input_blobs_index_over_scan"] = blobs
        out["E4b_gains_elsewhere"] = gains
        out["E4b_choice"] = (best if blobs < 1.01 and
                             sum(g >= 0.10 for g in gains.values()) >= 2
                             else base)
    # E5: fastest paint on input_blobs; a tie within 1% keeps the default
    if "E5" in res:
        arms = [a for a in res["E5"]["input_blobs"] if not a.startswith("_")]
        best = min(arms, key=lambda a: _t(res, "E5", "input_blobs", a,
                                          "paint"))
        dflt = "fast1_tpb128"
        out["E5_paint"] = (dflt if _t(res, "E5", "input_blobs", dflt, "paint")
                           <= 1.01 * _t(res, "E5", "input_blobs", best,
                                        "paint") else best)
    # E6: the largest L within 3% of L8 on every scene, else L8
    if "E6" in res:
        pick = 8
        for L in (32, 16):
            if all(_t(res, "E6", sc, f"L{L}") <= 1.03 * _t(res, "E6", sc,
                                                           "L8")
                   for sc in res["E6"]):
                pick = L
                break
        out["E6_lattice"] = pick
    # E7: deferred unless inline wins on input_blobs
    if "E7" in res:
        out["E7_unions"] = ("inline" if _t(res, "E7", "input_blobs",
                                           "inline")
                            < _t(res, "E7", "input_blobs", "deferred")
                            else "deferred")
    # E9: 128 unless a smaller value never fires on the main scenes at
    # L8 (R8's rounds), is never slower where some arm fires, and is
    # faster on at least one such scene. A scene where no arm fires runs
    # the same code in every arm, so its medians differ by noise only;
    # session 155420Z kept 128 on such a tie (maze_1k, 0.001 ms).
    if "E9" in res and "R8" in res:
        max_rounds = max(res["R8"][sc]["L8"]["counters"]["rounds"]
                         for sc in res["R8"])
        firing = [sc for sc in res["E9"]
                  if any(res["E9"][sc][a]["counters"]["valve_round"] > 0
                         for a in res["E9"][sc] if not a.startswith("_"))]
        pick = 128
        for v in (16, 32, 64):
            if max_rounds >= v:
                continue
            ts = [(_t(res, "E9", sc, f"valve{v}"),
                   _t(res, "E9", sc, "valve128")) for sc in firing]
            if ts and all(a <= b for a, b in ts) and any(a < b
                                                         for a, b in ts):
                pick = v
                break
        out["E9_valve_round"] = pick
        out["E9_firing_scenes"] = firing
        out["E9_max_rounds_main_scenes_at_L8"] = max_rounds
    # E13: E4's rule against the scan at the compiler's registers: an arm
    # qualifies if input_blobs moves under +1% (per-round median) and at
    # least 2 other scenes gain 10%; the qualifier with the lowest
    # geometric mean ratio over the other scenes wins
    if "E13" in res:
        others = [sc for sc in res["E13"] if sc != "input_blobs"]
        score = {}
        for a in E13_ARMS:
            if a == "scan":
                continue
            blobs = res["E13"]["input_blobs"][a]["ratio_to_first_median"]
            r = [_t(res, "E13", sc, a) / _t(res, "E13", sc, "scan")
                 for sc in others]
            gm = float(np.exp(np.mean(np.log(r))))
            score[a] = {"input_blobs": blobs, "geomean_others": gm,
                        "gains_10pct": sum(x <= 0.90 for x in r)}
        ok = [a for a, v in score.items()
              if v["input_blobs"] < 1.01 and v["gains_10pct"] >= 2]
        out["E13_scores"] = score
        out["E13_lookup_arm"] = (min(ok, key=lambda a:
                                     score[a]["geomean_others"])
                                 if ok else "scan")
    # E10: fastest waves on input_blobs
    if "E10" in res:
        out["E10_waves_tpb"] = min(("tpb256", "tpb128"), key=lambda a: _t(
            res, "E10", "input_blobs", a, "waves"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", default=os.environ.get("FLOOD_FILL_IMAGES"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="experiments to run (default: all)")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    pngs = B._resolve_pngs(args.images)
    print(B._free_m())
    for lk in ff.LOOKUPS:
        ff._warmup(8, False, False, lookup=lk)
        ff._warmup(8, True, False, lookup=lk)
    flush = cuda.to_device(np.zeros(B.L2_FLUSH_BYTES // 4, np.uint32))
    sink = cuda.device_array(1, np.uint32)
    B._SPIN["flush"], B._SPIN["sink"] = flush, sink
    sampler = B.ClockSampler()
    spin = np.zeros((2048, 2048, 3), np.uint8)
    spin[..., 0] = 255
    d_spin = cuda.to_device(spin)
    seng = ff.TileWaves(2048, 2048)
    B._spin(seng, d_spin, B.SPIN_SECONDS)
    del seng, d_spin
    print(f"clocks: {B._clocks()}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(B.RESULTS_DIR, f"experiments_{stamp}.json")
    res = {}
    payload = {"policy": {"rounds": ROUNDS, "sleep_naps": B.SLEEP_NAPS,
                          "l2_flush_bytes": B.L2_FLUSH_BYTES,
                          "respin_s": B.SCENE_SPIN_SECONDS},
               "png_md5": {n: B._md5(p) for n, p in pngs.items()},
               "results": res}
    for scene, build, exps in plan(pngs):
        if args.scenes and scene not in args.scenes:
            continue
        exps = [e for e in exps if args.only is None or e in args.only]
        if not exps:
            continue
        img = build()
        d_pr = cuda.to_device(img)
        d_img = cuda.device_array_like(img)
        for e in exps:
            t0 = time.time()
            r = EXPERIMENTS[e](img, d_img, d_pr, sampler)
            res.setdefault(e, {})[scene] = r
            line = " ".join(
                f"{a} {v['total_ms']:.3f}" + (
                    f"({v['ratio_to_first_median']:.3f})"
                    if "ratio_to_first_median" in v else "")
                if "total_ms" in v else
                f"{a} rounds {v['counters']['rounds']}"
                for a, v in r.items() if not a.startswith("_"))
            print(f"{e:4s} {scene:16s} {line} | sm "
                  f"{r['_clock'].get('clk_sm_min', '?')} mem "
                  f"{r['_clock'].get('clk_mem_min', '?')} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
            B._release()
        del img, d_pr, d_img
        B._release()
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)
    try:
        payload["decisions"] = decide(res)
    except KeyError as exc:          # a partial run (--only / --scenes)
        payload["decisions"] = {"skipped": f"missing {exc}"}
    payload["clock_trace"] = sampler.trace()
    sampler.stop()
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(json.dumps(payload["decisions"], indent=1))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
