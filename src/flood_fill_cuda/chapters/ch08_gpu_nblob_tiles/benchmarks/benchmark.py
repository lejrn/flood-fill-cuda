"""Benchmark: seeded tile waves (ch08) against ch06's runs and ch05.

The question: can ch07's tiles label many blobs, and how close do they
get to ch06 on input_blobs? One session, every side timed the same way
in the same interleaved rounds (ch07's policy, plus the fixes of the
ch08 fairness review):

  every GPU run   image restored from a pristine device copy; for the
                  mask and label contracts the pack runs here, off the
                  clock; a 96 MB read flushes the 32 MB L2; synchronize;
                  a ~2 ms GPU sleep kernel is queued so every timed
                  launch is in the stream before the first event fires
  totals          ONE event before the first launch and ONE after the
                  last. Per-phase events put 16 commands behind the sync
                  for ch06 and 12 for ch08, and past ~13 the driver
                  stalls 40-85 us inside the window: an artefact that
                  only ch06 paid on scenes under ~8 Mpx (review F2)
  phases          a separate pass: each phase in its own run, with its
                  two events around that one launch (the earlier phases
                  run first, untimed in between)
  rounds          9 for the totals (5 for the phases), every config
                  once per round, order reversed on odd rounds; medians,
                  minimums, max, stdev, per-round ratios for the pairs
  ch06's merge    swept per scene (bare, 512 to 8192 blocks, the merge
                  phase alone, 5 rounds) and timed at its best. ch06
                  picked 512 with its instrumented merge, whose counter
                  atomics make big grids slow; bare, 2048 is 0.12 ms
                  faster on input_blobs (review F1). ch06_m512 keeps the
                  shipped shape for reference
  clocks          8 s spin before the scenes, a 1 s GPU-bound respin
                  before each scene's rounds, a background nvidia-smi
                  sampler (100 ms) whose window over the GPU rounds is
                  kept. ch05 runs AFTER the GPU rounds, 0.5 s respin
                  before each call, with its own clock window

Configs (contract: rgb = RGB in, recoloured RGB out; mask = packed mask
in, RGB painted; label = packed mask in, every comp or run labelled,
nothing painted):
  ch08_L8         rgb, lattice 8, the default lookup and valve: headline
  ch08_L8_mask    mask contract
  ch08_L8_label   labelling only: init, label, waves
  ch08_L0         rgb, S2 seeds alone and the valve off: pure waves
  ch08_L16        rgb, lattice 16: a dial point
  ch08_L8_scan    rgb, the scan lookup (the previous default)
  ch08_L8_index   rgb, the index lookup
  ch08_L8_voff    rgb, the valve off
  ch06            ch06's run pipeline at its best merge grid, rgb
  ch06_mask       the same, mask contract
  ch06_label      labelling only: count, scan, emit, merge, flatten
  ch06_tp         ch06 with ch08's pack_tiles in place of its pack: the
                  like-for-like pack, so the pack gain and the algorithmic
                  gain are reported apart
  ch06_m512       ch06 rgb at its shipped merge grid (512 blocks)
  ch05 split_L8   ch05's best config on input_blobs (seed_merge, lattice
                  8, split build), by its own host clock (launch plus
                  synchronize), after the GPU rounds

Before the timed rounds, an untimed pass checks every config: each
painted image must equal ch06's own RunRecolor.run (all three chapters
paint palette[canonical label % 6]); ch05's too. An instrumented ch08
engine runs each ch08 config once more for the counters and the
per-round frontier.

Scenes: input_blobs and input_blocks (PNGs from --images DIR or
FLOOD_FILL_IMAGES; a missing file is an error; md5 recorded), then
ch06's four synthetic scenes; a crop sweep of input_blobs; worst cases
(ch07's zz_serp_1k, maze_1k and zz_lattice_4k, the lattice-avoiding
diagonal snake, and four scenes of the ch08 perf review: the decoy
staircase, zz_serp_4k, a dot lattice with 256 comps per tile, a 1-px
noise strip). The @njit oracle is timed last, after every GPU scene.

Run:
  uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.benchmark --images DIR
Writes waves_<stamp>.json + waves_<stamp>_scenes.csv to
results/ch08_gpu_nblob_tiles/benchmark_results/.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import csv
import gc
import hashlib
import json
import statistics
import subprocess
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from .. import flood_fill as ff
from .. import scenes as S8
from ..kernels import gpu_sleep_kernel, get_pack_kernel, pack_shape
from ..cpu_oracle import cpu_label_components
from ...ch05_gpu_nblob_nblock import scenes as S5
from ...ch06_gpu_nblob_runs.recolor import (
    RunRecolor, _warmup as ch06_warmup, PHASE_BLOCKS as CH06_PHASE_BLOCKS,
    PACK_ROW_BLOCKS as CH06_PACK_ROW_BLOCKS)
from ...ch06_gpu_nblob_runs import kernels as k6
from ...ch06_gpu_nblob_runs.benchmarks.benchmark import (
    _measure_read_write_peaks, CH05_CFG, CH05_LABEL)
from ...ch07_gpu_1blob_tiles.benchmarks.benchmark import (
    ClockSampler, _l2_flush, _clocks, L2_FLUSH_BYTES, SLEEP_NAPS)
from ...ch07_gpu_1blob_tiles.benchmarks import worst_cases as W7
from ....shared import results_paths
from ....shared.bandwidth import measure_peak_bandwidth

warnings.simplefilter("ignore", NumbaPerformanceWarning)

RESULTS_DIR = results_paths.results_dir("ch08_gpu_nblob_tiles",
                                        "benchmark_results")
GPU_REPEATS = 9
PHASE_REPEATS = 5
SWEEP_REPEATS = 5
MERGE_SWEEP = (512, 1024, 2048, 4096, 8192)
CH06_SHIPPED_MERGE = CH06_PHASE_BLOCKS["merge"]
CH05_SLOW_S = 2.0          # a first ch05 call slower than this: 3 samples
CH05_FEW = 3
E2E_REPEATS = 3
ORACLE_REPEATS = 3
SPIN_SECONDS = 8.0
SCENE_SPIN_SECONDS = 1.0
CH05_RESPIN_SECONDS = 0.5
PACK_VARIANT = "direct"    # ch08 and ch06_tp (E1 picks it)
PNG_MD5 = {"input_blobs": "47639eb37f973b5092ec2c4db86f45a4",
           "input_blocks": "68f1fda7f2eba3d73b87e1cc11155061"}

LOOKUP = ff.DEFAULT_LOOKUP       # the headline's neighbour lookup (E13)
VALVE = ff.DEFAULT_VALVE_ROUND   # the default valve round (E9)
# name -> (side, contract, lattice, valve_round, lookup)
CONFIGS = {
    "ch08_L8": ("ch08", "rgb", 8, VALVE, LOOKUP),
    "ch08_L8_mask": ("ch08", "mask", 8, VALVE, LOOKUP),
    "ch08_L8_label": ("ch08", "label", 8, VALVE, LOOKUP),
    "ch08_L0": ("ch08", "rgb", 0, 0, LOOKUP),
    "ch08_L16": ("ch08", "rgb", 16, VALVE, LOOKUP),
    "ch08_L8_scan": ("ch08", "rgb", 8, VALVE, "scan"),
    "ch08_L8_index": ("ch08", "rgb", 8, VALVE, "index"),
    "ch08_L8_voff": ("ch08", "rgb", 8, 0, LOOKUP),
    "ch06": ("ch06", "rgb", None, None, None),
    "ch06_mask": ("ch06", "mask", None, None, None),
    "ch06_label": ("ch06", "label", None, None, None),
    "ch06_tp": ("ch06_tp", "rgb", None, None, None),
    "ch06_m512": ("ch06_m512", "rgb", None, None, None),
}
MAIN_CONFIGS = ["ch08_L8", "ch08_L8_mask", "ch08_L8_label", "ch08_L0",
                "ch08_L16", "ch08_L8_scan", "ch08_L8_index", "ch06",
                "ch06_mask", "ch06_label", "ch06_tp", "ch06_m512"]
CROP_CONFIGS = ["ch08_L8", "ch08_L8_label", "ch06", "ch06_label",
                "ch06_tp"]
WORST_CONFIGS = ["ch08_L8", "ch08_L8_label", "ch08_L8_voff", "ch08_L0",
                 "ch08_L8_scan", "ch08_L8_index", "ch06", "ch06_label"]
PHASE_CONFIGS = ["ch08_L8", "ch06"]     # the per-phase pass
HEADLINE = "ch08_L8"
# per-round ratios a / b (< 1: a is faster)
PAIRS = {
    "L8_over_ch06": ("ch08_L8", "ch06"),
    "L8_over_ch06_tp": ("ch08_L8", "ch06_tp"),
    "L8_over_ch06_m512": ("ch08_L8", "ch06_m512"),
    "mask_over_ch06_mask": ("ch08_L8_mask", "ch06_mask"),
    "label_over_ch06_label": ("ch08_L8_label", "ch06_label"),
    "ch06_tp_over_ch06": ("ch06_tp", "ch06"),
    "ch06_over_ch06_m512": ("ch06", "ch06_m512"),
    "L0_over_L8": ("ch08_L0", "ch08_L8"),
    "L16_over_L8": ("ch08_L16", "ch08_L8"),
    "scan_over_L8": ("ch08_L8_scan", "ch08_L8"),
    "index_over_L8": ("ch08_L8_index", "ch08_L8"),
    "voff_over_L8": ("ch08_L8_voff", "ch08_L8"),
}
CROP_SIDES = (1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000)


# ------------------------------------------------------------ helpers

def _free_m():
    try:
        return subprocess.run(["free", "-m"], capture_output=True,
                              text=True, timeout=10).stdout
    except Exception:
        return ""


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _host_run_count(img):
    red = ((img[..., 0] == 255) & (img[..., 1] == 0)
           & (img[..., 2] == 0)).astype(np.int8)
    return int((np.diff(red, axis=1, prepend=0) == 1).sum())


def _ratio_stats(a, b):
    r = [x / y for x, y in zip(a, b)]
    return statistics.median(r), min(r), max(r)


_SPIN = {}


def _spin(eng, d_img, seconds, contract="rgb"):
    """Hold the GPU busy for `seconds`: 96 MB L2-flush reads (~0.45 ms
    each) interleaved with pipeline runs, 10 of each per host sync.

    A smoke session showed why the GPU-bound part matters: a respin of
    back-to-back runs on a 1 Mpx scene is launch-bound (5 launches of
    ~0.1 ms each), the GPU idles between them, and the scene was timed
    at 585-790 MHz SM and 810 MHz memory (P8)."""
    flush, sink = _SPIN["flush"], _SPIN["sink"]
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            _l2_flush[512, 256](flush, sink)
            eng.run(d_img, contract)
        cuda.synchronize()


def _derived(c):
    """Counters the report puts next to the times."""
    return {
        "links": c["union_done"],
        "atomics_excl_union": (c["seed_append_atomics"] + c["cas"]
                               + c["append_atomics"]
                               + c["coll_append_atomics"]
                               + c["rmin_atomics"]),
    }


def _time_stages(stages, phase=None):
    """Enqueue `stages` ((name, launch) pairs) and time them with two
    CUDA events: around the whole list, or around stage `phase` alone
    (the stages before it run first, the ones after it are not run).
    Returns ms after synchronizing on the last event."""
    e0 = cuda.event(timing=True)
    e1 = cuda.event(timing=True)
    if phase is None:
        e0.record()
        for _, launch in stages:
            launch()
    else:
        for _, launch in stages[:phase]:
            launch()
        e0.record()
        stages[phase][1]()
    e1.record()
    e1.synchronize()
    return cuda.event_elapsed_time(e0, e1)


class Ch06Side:
    """ch06's RunRecolor buffers, launched as a stage list.

    The launches are RunRecolor.run's, bare (instrumented=False), with
    the same arguments and per-phase grids; the merge grid can be set
    per scene. side "ch06_tp" packs with ch08's pack_tiles (which writes
    ch06's mask bit for bit) and runs ch06's mask pipeline. Every scene's
    untimed pass checks each stage list's paint against RunRecolor.run
    itself, so the copy cannot drift silently."""

    def __init__(self, width, height, run_capacity):
        self.r6 = RunRecolor(width, height, run_capacity=run_capacity)
        self.tp = get_pack_kernel(PACK_VARIANT)
        self.tp_grid, self.tp_tpb = pack_shape(width, self.r6.words_per_row)
        self.merge_blocks = CH06_SHIPPED_MERGE

    def pack(self, d_img):
        self.r6.pack(d_img)

    def stages(self, d_img, side, contract, merge_blocks=None):
        r6 = self.r6
        g = dict(r6.phase_grid)
        mb = merge_blocks or (CH06_SHIPPED_MERGE if side == "ch06_m512"
                              else self.merge_blocks)
        g["merge"] = (mb, g["merge"][1])
        out = []
        if side == "ch06_tp":
            out.append(("pack", lambda: self.tp[self.tp_grid, self.tp_tpb](
                d_img, r6.mask)))
        elif contract == "rgb":
            out.append(("pack", lambda: k6.pack_kernel[r6.pack_grid](
                d_img, r6.mask)))
        out += [
            ("count", lambda: k6.count_kernel[g["count"]](
                r6.mask, r6.row_count)),
            ("scan", lambda: k6.row_scan_kernel[1, k6.SCAN_TPB](
                r6.row_count, r6.row_off, r6.counters, r6.run_capacity)),
            ("emit", lambda: k6.emit_kernel[g["emit"]](
                r6.mask, r6.row_off, r6.run_x, r6.run_y0, r6.run_y1,
                r6.parent, r6.counters)),
            ("merge", lambda: k6.merge_rows_kernel[g["merge"]](
                r6.run_x, r6.run_y0, r6.run_y1, r6.row_off, r6.parent,
                r6.counters, False)),
            ("flatten", lambda: k6.flatten_kernel[g["flatten"]](
                r6.parent, r6.run_x, r6.run_y0, r6.run_label, r6.height,
                r6.counters, False)),
        ]
        if contract != "label":
            out.append(("paint", lambda: k6.paint_kernel[g["paint"]](
                d_img, r6.run_x, r6.run_y0, r6.run_y1, r6.run_label,
                r6.counters, r6.palette)))
        return out

    def counters(self):
        c = self.r6.counters.copy_to_host()
        if c[k6.RUN_OVERFLOW]:
            raise RuntimeError("ch06 run table overflowed")
        return {"runs": int(c[k6.N_RUNS])}


def _release():
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass


def _ch05_call(img):
    from ...ch05_gpu_nblob_nblock.flood_fill import flood_fill
    t0 = time.perf_counter()
    res = flood_fill(img, **CH05_CFG)
    wall = time.perf_counter() - t0
    return res, wall


# ------------------------------------------------------------- scenes

def bench_scene(name, img, note, cfg_names, sampler, flush, sink,
                run_ch05=False, e2e=True, group="main"):
    width, height = img.shape[0], img.shape[1]
    t_scene0 = time.time()
    d_pristine = cuda.to_device(img)
    d_img = cuda.device_array_like(img)
    lookups = sorted({CONFIGS[c][4] for c in cfg_names
                      if CONFIGS[c][0] == "ch08"})
    engs = {lk: ff.TileWaves(width, height, pack_variant=PACK_VARIANT,
                             lookup=lk) for lk in lookups}
    eng = engs[LOOKUP]
    n_runs_host = _host_run_count(img)
    side6 = Ch06Side(width, height, max(8192, int(n_runs_host * 1.05)))
    phase_cfgs = [c for c in PHASE_CONFIGS if c in cfg_names]
    has_ch06 = any(CONFIGS[c][0].startswith("ch06") for c in cfg_names)

    def stages_of(cfg):
        side, contract, lat, valve, lk = CONFIGS[cfg]
        if side == "ch08":
            return engs[lk].stages(
                d_img, "rgb" if contract == "rgb" else "mask", lattice=lat,
                valve_round=valve, paint=contract != "label")
        return side6.stages(d_img, side, contract)

    def prepare(cfg):
        """Restore, pack off the clock if the contract says so, flush,
        sync, queue the sleep kernel."""
        side, contract, lat, valve, lk = CONFIGS[cfg]
        d_img.copy_to_device(d_pristine)
        if contract in ("mask", "label"):
            if side == "ch08":
                engs[lk].pack(d_img)
            else:
                side6.pack(d_img)
        _l2_flush[512, 256](flush, sink)
        cuda.synchronize()
        gpu_sleep_kernel[1, 32](SLEEP_NAPS)

    def run_total(cfg):
        prepare(cfg)
        return _time_stages(stages_of(cfg))

    # ---- untimed pass: correctness and counters
    check = {}
    meta = {}
    d_img.copy_to_device(d_pristine)
    names6, ev6 = side6.r6.run(d_img, contract="rgb", instrumented=False)
    ev6[-1].synchronize()
    ref = d_img.copy_to_host()         # ch06's own pipeline, shipped shape
    meta["ch06"] = side6.counters()
    for cfg in cfg_names:
        run_total(cfg)
        if CONFIGS[cfg][1] == "label":
            check[cfg] = True          # nothing painted; its mask twin is
            continue                   # checked
        check[cfg] = bool(np.array_equal(d_img.copy_to_host(), ref))
        if cfg.startswith("ch06"):
            check[cfg] = check[cfg] and (side6.counters()["runs"]
                                         == n_runs_host)
    engis = {lk: ff.TileWaves(width, height, instrumented=True,
                              pack_variant=PACK_VARIANT, lookup=lk)
             for lk in lookups}
    for cfg in cfg_names:
        side, contract, lat, valve, lk = CONFIGS[cfg]
        if side != "ch08" or contract == "label":
            continue
        engi = engis[lk]
        d_img.copy_to_device(d_pristine)
        if contract == "mask":
            engi.pack(d_img)
        engi.run(d_img, contract, lattice=lat, valve_round=valve)
        cuda.synchronize()
        c = engi.read_counters()
        st = engi.round_stats()
        ok = bool(np.array_equal(d_img.copy_to_host(), ref))
        check[cfg] = check[cfg] and ok
        meta[cfg] = {**c, **_derived(c), "frontier_comps":
                     st["frontier_comps"], "frontier_tiles":
                     st["frontier_tiles"], "tile_visits": st["tile_visits"],
                     "round_us_instrumented": st["round_us"],
                     "union_us_instrumented": st["union_us"],
                     "flatten_us_instrumented": st["flatten1_us"]
                     + st["flatten2_us"],
                     "waves_us_instrumented": st["waves_us_total"]}
    del engis
    _release()
    ch05 = None
    if run_ch05:
        try:
            res5, wall = _ch05_call(img)
            ch05 = {"config": CH05_LABEL, "first_call_s": wall,
                    "n_blobs": int(res5.n_blobs),
                    "paint_matches_ch06": bool(np.array_equal(res5.img, ref)),
                    "samples": [], "n_samples": (CH05_FEW if wall > CH05_SLOW_S
                                                 else GPU_REPEATS)}
            del res5
        except Exception as exc:
            ch05 = {"config": CH05_LABEL,
                    "error": f"{type(exc).__name__}: {exc}"}
        gc.collect()
    del ref
    gc.collect()

    # ---- ch06's merge grid, swept bare (the merge phase alone)
    _spin(eng, d_img, SCENE_SPIN_SECONDS)
    sweep = {}
    if has_ch06:
        mi = [n for n, _ in side6.stages(d_img, "ch06", "label")].index(
            "merge")
        times = {mb: [] for mb in MERGE_SWEEP}
        for i in range(SWEEP_REPEATS):
            order = MERGE_SWEEP if i % 2 == 0 else MERGE_SWEEP[::-1]
            for mb in order:
                prepare("ch06_label")
                times[mb].append(_time_stages(side6.stages(
                    d_img, "ch06", "label", merge_blocks=mb), phase=mi))
        sweep = {mb: statistics.median(v) for mb, v in times.items()}
        side6.merge_blocks = min(sweep, key=sweep.get)

    # ---- timed rounds, interleaved, two events per run
    samples = {c: [] for c in cfg_names}
    t0 = time.time()
    for i in range(GPU_REPEATS):
        order = cfg_names if i % 2 == 0 else list(reversed(cfg_names))
        for cfg in order:
            samples[cfg].append(run_total(cfg))
    t1 = time.time()

    # ---- per-phase pass: each phase in its own run
    phases = {c: {} for c in phase_cfgs}
    for i in range(PHASE_REPEATS):
        order = phase_cfgs if i % 2 == 0 else list(reversed(phase_cfgs))
        for cfg in order:
            names = [n for n, _ in stages_of(cfg)]
            for j, ph in enumerate(names):
                prepare(cfg)
                phases[cfg].setdefault(ph, []).append(
                    _time_stages(stages_of(cfg), phase=j))
    t2 = time.time()

    # ---- ch05 after the GPU rounds, a respin before every call
    t5a = time.time()
    if ch05 and "error" not in ch05:
        while len(ch05["samples"]) < ch05["n_samples"]:
            _spin(eng, d_img, CH05_RESPIN_SECONDS)
            res5, _ = _ch05_call(img)
            ch05["samples"].append(res5.kernel_ms)
            del res5
            gc.collect()
    t5b = time.time()

    row = {"scene": name, "group": group, "note": note, "width": width,
           "height": height, "red_px": int(np.count_nonzero(
               (img[..., 0] == 255) & (img[..., 1] == 0)
               & (img[..., 2] == 0))),
           "runs": n_runs_host, "tiles": eng.n_tiles,
           "configs_timed": list(cfg_names),
           "timed_window_s": [round(t0 - sampler.t0, 2),
                              round(t1 - sampler.t0, 2)],
           "phase_window_s": [round(t1 - sampler.t0, 2),
                              round(t2 - sampler.t0, 2)]}
    row.update(sampler.window(t0, t1))
    row["phase_pass_clock"] = sampler.window(t1, t2)
    if sweep:
        row["ch06_merge_sweep_ms"] = {str(k): v for k, v in sweep.items()}
        row["ch06_merge_blocks"] = side6.merge_blocks
    for cfg in cfg_names:
        tot = samples[cfg]
        row[f"{cfg}_ms"] = statistics.median(tot)
        row[f"{cfg}_ms_min"] = min(tot)
        row[f"{cfg}_ms_max"] = max(tot)
        row[f"{cfg}_ms_stdev"] = statistics.stdev(tot)
        row[f"{cfg}_paint_matches_ch06"] = check[cfg]
        if cfg in meta and cfg != "ch06":
            for kk, v in meta[cfg].items():
                row[f"{cfg}_{kk}"] = v
    for cfg in phase_cfgs:
        for ph, v in phases[cfg].items():
            row[f"{cfg}_phase_{ph}_ms"] = statistics.median(v)
        row[f"{cfg}_phase_sum_ms"] = sum(statistics.median(v)
                                         for v in phases[cfg].values())
    row["ch06_runs_match_host"] = meta["ch06"]["runs"] == n_runs_host
    for key, (a, b) in PAIRS.items():
        if a in cfg_names and b in cfg_names:
            med, lo, hi = _ratio_stats(samples[a], samples[b])
            row[f"ratio_{key}_median"] = med
            row[f"ratio_{key}_min"] = lo
            row[f"ratio_{key}_max"] = hi
    if ch05:
        row["ch05"] = {kk: v for kk, v in ch05.items()}
        row["ch05_clock"] = sampler.window(t5a, t5b)
        if ch05.get("samples"):
            row["ch05_ms"] = statistics.median(ch05["samples"])
            row["ch05_ms_min"] = min(ch05["samples"])
            row["ch05_over_ch08_L8"] = row["ch05_ms"] / row[f"{HEADLINE}_ms"]
    if e2e and HEADLINE in cfg_names:
        host_out = np.empty_like(img)
        e2 = []
        for _ in range(E2E_REPEATS):
            cuda.synchronize()
            ts = time.perf_counter()
            d_img.copy_to_device(img)
            eng.run(d_img, "rgb")
            d_img.copy_to_host(host_out)
            e2.append((time.perf_counter() - ts) * 1000)
        row[f"{HEADLINE}_e2e_ms"] = statistics.median(e2)
        row[f"{HEADLINE}_e2e_ms_min"] = min(e2)
        del host_out
    row["all_paint_match"] = all(check.values()) and (
        not ch05 or ch05.get("paint_matches_ch06", True))
    row["scene_seconds"] = round(time.time() - t_scene0, 1)

    line = f"{name:16s} "
    for cfg in cfg_names:
        line += f"{cfg.replace('ch08_', '')} {row[f'{cfg}_ms']:.3f} "
    for key in ("L8_over_ch06", "label_over_ch06_label"):
        if f"ratio_{key}_median" in row:
            line += f"| {key} {row[f'ratio_{key}_median']:.3f} "
    if row.get("ch05_ms"):
        line += f"| ch05 {row['ch05_ms']:.2f} "
    if sweep:
        line += f"| merge {side6.merge_blocks} "
    line += (f"| sm {row.get('clk_sm_min', '?')}-"
             f"{row.get('clk_sm_median', '?')} mem "
             f"{row.get('clk_mem_min', '?')} "
             f"{'OK' if row['all_paint_match'] else 'MISMATCH'}")
    print(line, flush=True)

    del eng, engs, side6, d_img, d_pristine
    _release()
    return row


def _resolve_pngs(images_dir):
    if not images_dir:
        raise SystemExit("give --images DIR or set FLOOD_FILL_IMAGES: the "
                         "PNGs are git-ignored and absent from worktrees")
    out = {}
    for name in ("input_blobs", "input_blocks"):
        path = os.path.join(images_dir, f"{name}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        out[name] = path
    return out


def main_scenes(pngs):
    """(name, builder, note, run_ch05)"""
    return [
        ("input_blobs", lambda: S5.png_scene(pngs["input_blobs"])[0],
         "9000x9000 PNG, 2,522 blobs: the headline", True),
        ("input_blocks", lambda: S5.png_scene(pngs["input_blocks"])[0],
         "1000x1000 PNG, 21,618 blobs, 24 comps per tile", True),
        ("blob_grid_100",
         lambda: S5.blob_grid_scene(4000, 4000, 10, 10, 360, gap=40)[0],
         "100 squares of 129,600 px", True),
        ("random_4000",
         lambda: S5.random_blobs_scene(4000, 4000, 0.30, 0)[0],
         "density 0.30 noise, 755,577 blobs, 58 comps per tile", True),
        ("disk_r2000", lambda: S5.disk_scene(4200, 4200, 2000)[0],
         "one 12.6M px disk", True),
        ("serpentine_2048", lambda: S5.serpentine_scene(2048, 2048)[0],
         "one 2M px snake, 16 comps per tile", True),
    ]


def worst_scenes():
    return [
        ("zz_serp_1k", lambda: W7.zz_serpentine_scene()[0],
         "ch07's zigzag serpentine on the x tile borders"),
        ("maze_1k", lambda: W7.maze_scene()[0],
         "perfect maze, 1-px corridors, 1023x1023"),
        ("diag_snake_159", lambda: S8.diag_snake_scene(159),
         "lattice-avoiding anti-diagonal through tile corners: 160 comps "
         "in a chain, one seed (5120x5120)"),
        ("zz_lattice_4k", lambda: W7.zz_lattice_scene()[0],
         "zigzag on every tile border, 4096x4096"),
        # from the ch08 perf review
        ("stair_decoy_19", lambda: S8.stair_scene(19),
         "lattice-avoiding staircase, one seed, 133-comp chain, ~45-58 "
         "isolated decoys on every path tile's frame (608x4288)"),
        ("zz_serp_4k", lambda: W7.zz_serpentine_scene(4096, 4096)[0],
         "ch07's zigzag serpentine at 4096x4096: 32 ring comps per tile"),
        ("dots_4k", lambda: S8.dot_lattice_scene((128, 128)),
         "red at even (x, y): 256 one-pixel comps per tile, 4096x4096"),
        ("strip_1px_x", lambda: S8.thin_strip_scene(1 << 22, 1, 0, 0.5, 3),
         "4194304 x 1 noise strip, density 0.5: 32 px per tile"),
    ]


def oracle_pass(rows, builders):
    """The @njit oracle (ch05's cpu_label_components) per main scene,
    after every GPU scene."""
    by = {r["scene"]: r for r in rows}
    for name, build in builders:
        if name not in by:
            continue
        img = build()
        reps = 1 if img.shape[0] * img.shape[1] > 30_000_000 else \
            ORACLE_REPEATS
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _, n = cpu_label_components(img)
            ts.append((time.perf_counter() - t0) * 1000)
            gc.collect()
        row = by[name]
        row["njit_oracle_ms"] = statistics.median(ts)
        row["njit_oracle_blobs"] = int(n)
        row[f"{HEADLINE}_speedup_vs_njit"] = (row["njit_oracle_ms"]
                                              / row[f"{HEADLINE}_ms"])
        print(f"  {name:20s} njit oracle {row['njit_oracle_ms']:9.1f} ms "
              f"({n} blobs)", flush=True)
        del img
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", default=os.environ.get("FLOOD_FILL_IMAGES"))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="main scenes to run (default: all)")
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--no-worst", action="store_true")
    ap.add_argument("--worst", nargs="*", default=None,
                    help="worst-case scenes to run (default: all)")
    ap.add_argument("--no-ch05", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    pngs = _resolve_pngs(args.images)
    md5 = {n: _md5(p) for n, p in pngs.items()}
    for n, h in md5.items():
        if h != PNG_MD5[n]:
            print(f"WARNING: {n} md5 {h} differs from {PNG_MD5[n]}")
    free0 = _free_m()
    print(free0)
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    sm_count = int(device.MULTIPROCESSOR_COUNT)
    print(f"Device: {dev_name.strip()} ({sm_count} SMs)")

    # ---- warm every kernel off the clock
    for inst in (False, True):
        for lk in ff.LOOKUPS:
            ff._warmup(8, inst, False, "auto", True, PACK_VARIANT, lk)
    ch06_warmup()
    warm = S5.random_blobs_scene(96, 96, 0.3, 1)[0]
    side = Ch06Side(96, 96, 8192)
    dw = cuda.to_device(warm)
    for sd, ct in (("ch06", "rgb"), ("ch06_tp", "rgb"), ("ch06", "label")):
        for mb in MERGE_SWEEP:
            _time_stages(side.stages(dw, sd, ct, merge_blocks=mb))
    cuda.synchronize()
    del side, dw
    if not args.no_ch05:
        _ch05_call(warm)
    cpu_label_components(warm)
    flush = cuda.to_device(np.zeros(L2_FLUSH_BYTES // 4, np.uint32))
    sink = cuda.device_array(1, np.uint32)
    _l2_flush[512, 256](flush, sink)
    cuda.synchronize()
    _SPIN["flush"], _SPIN["sink"] = flush, sink
    resources = {}
    for lk in ff.LOOKUPS:
        resources[lk] = {
            "bare": ff.kernel_resources(8, False, False, lookup=lk),
            "instrumented": ff.kernel_resources(8, True, False, lookup=lk),
            "waves_coop_max_blocks_256": ff.max_blocks(256, 8, False, False,
                                                       lk),
            "waves_coop_max_blocks_256_instrumented": ff.max_blocks(
                256, 8, True, False, lk)}
    print(f"  resources: {resources}")

    sampler = ClockSampler()
    print(f"Spinning the clock up for {SPIN_SECONDS:.0f} s...")
    spin_img = np.full((2048, 2048, 3), 0, np.uint8)
    spin_img[..., 0] = 255
    d_spin = cuda.to_device(spin_img)
    spin_eng = ff.TileWaves(2048, 2048, pack_variant=PACK_VARIANT)
    _spin(spin_eng, d_spin, SPIN_SECONDS)
    del spin_eng, d_spin
    peak = measure_peak_bandwidth()
    read_gb_s, write_gb_s = _measure_read_write_peaks(256 * 2 ** 20)
    peaks = {"copy_gb_s": peak["gb_s"], "read_gb_s": read_gb_s,
             "write_gb_s": write_gb_s}
    print(f"  peaks: {peaks}")
    clocks_start = _clocks()
    print(f"  clocks: {clocks_start}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"waves_{stamp}.json")
    payload = {
        "device": dev_name.strip(), "sm_count": sm_count,
        "experiment": (
            "ch08 seeded waves over tile components vs ch06 run CCL (rgb, "
            "mask, labelling only, ch06 with ch08's pack, ch06 at its "
            "shipped merge grid) and ch05 split_L8, one interleaved "
            "session; totals with two events, phases in a separate pass, "
            "ch06's merge grid swept per scene"),
        "headline_config": HEADLINE,
        "configs": {k: dict(zip(("side", "contract", "lattice",
                                 "valve_round", "lookup"), v))
                    for k, v in CONFIGS.items()},
        "ch08_defaults": {"label_grid": ff.LABEL_GRID,
                          "label_max_registers": ff.LABEL_MAX_REGISTERS,
                          "waves_max_registers": ff.WAVES_MAX_REGISTERS,
                          "paint_grid": ff.PAINT_GRID,
                          "waves_tpb": ff.WAVES_TPB,
                          "lookup": LOOKUP,
                          "dense_min": ff.DEFAULT_DENSE_MIN,
                          "valve_round": VALVE,
                          "lattice": ff.DEFAULT_LATTICE,
                          "pack_variant": PACK_VARIANT,
                          "coll_capacity": "16 * n_tiles"},
        "ch06_launch": {"phase_blocks": dict(CH06_PHASE_BLOCKS),
                        "merge_sweep": list(MERGE_SWEEP),
                        "merge": "per scene: the sweep's fastest bare "
                                 "merge phase; ch06_m512 keeps 512",
                        "pack_row_blocks": CH06_PACK_ROW_BLOCKS,
                        "instrumented": False,
                        "run_capacity": "max(8192, 1.05 * host runs)"},
        "ch05_config": {"label": CH05_LABEL, **CH05_CFG},
        "resources": resources,
        "png_md5": md5, "png_paths": pngs,
        "policy": {"gpu_repeats": GPU_REPEATS,
                   "phase_repeats": PHASE_REPEATS,
                   "merge_sweep_repeats": SWEEP_REPEATS,
                   "spin_s": SPIN_SECONDS,
                   "scene_spin_s": SCENE_SPIN_SECONDS,
                   "ch05_respin_s": CH05_RESPIN_SECONDS,
                   "sleep_naps": SLEEP_NAPS,
                   "l2_flush_bytes": L2_FLUSH_BYTES,
                   "ch05_samples": f"{GPU_REPEATS}, or {CH05_FEW} if the "
                                   f"first call takes > {CH05_SLOW_S} s",
                   "e2e_repeats": E2E_REPEATS},
        "timing_note": (
            "*_ms = one event before the config's first launch to one "
            "after its last (no events in between). ch08 rgb: pack, init, "
            "label, waves, paint; mask: the same minus pack (pack runs "
            "before the L2 flush, off the clock); label: init, label, "
            "waves. ch06 rgb: pack, count, scan, emit, merge, flatten, "
            "paint; label: count to flatten; ch06_tp: ch08's pack_tiles "
            "then ch06's mask pipeline. <cfg>_phase_<phase>_ms: the phase pass, "
            "each phase alone in its own run (events around it only). "
            "ch05_ms = ch05's own host clock (launch + synchronize), "
            "median of its samples, taken after the GPU rounds. Every "
            "GPU run: restore, pack if mask/label, 96 MB L2 flush, sync, "
            "sleep kernel, events. e2e = pageable H2D + pipeline + D2H, "
            "perf_counter."),
        "free_m_start": free0, "peaks": peaks, "clocks_start": clocks_start,
        "scenes": [],
    }

    def save():
        if args.no_save:
            return
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

    print("\nmain scenes")
    rows = payload["scenes"]
    mains = [s for s in main_scenes(pngs)
             if args.scenes is None or s[0] in args.scenes]
    for name, build, note, ch05 in mains:
        img = build()
        rows.append(bench_scene(name, img, note, MAIN_CONFIGS, sampler,
                                flush, sink, run_ch05=ch05 and not
                                args.no_ch05))
        del img
        gc.collect()
        save()

    if not args.no_crops:
        print("\ncrop sweep of input_blobs")
        full = S5.png_scene(pngs["input_blobs"])[0]
        for side_px in CROP_SIDES:
            c0 = (full.shape[0] - side_px) // 2
            c1 = (full.shape[1] - side_px) // 2
            crop = np.ascontiguousarray(full[c0:c0 + side_px,
                                             c1:c1 + side_px])
            rows.append(bench_scene(f"crop_{side_px}", crop,
                                    f"centred {side_px}^2 crop of "
                                    f"input_blobs", CROP_CONFIGS, sampler,
                                    flush, sink, e2e=False, group="crop"))
            del crop
            gc.collect()
            save()
        del full
        gc.collect()

    if not args.no_worst:
        print("\nworst cases")
        for name, build, note in worst_scenes():
            if args.worst is not None and name not in args.worst:
                continue
            img = build()
            rows.append(bench_scene(name, img, note, WORST_CONFIGS, sampler,
                                    flush, sink, run_ch05=not args.no_ch05,
                                    e2e=False, group="worst"))
            del img
            gc.collect()
            save()

    payload["clocks_end"] = _clocks()
    print("\n@njit oracle, after all GPU scenes:")
    oracle_pass(rows, [(n, b) for n, b, _, _ in mains])
    sampler.stop()
    payload["clock_trace"] = sampler.trace()
    payload["free_m_end"] = _free_m()
    if args.no_save:
        return
    save()
    csv_path = json_path.replace(".json", "_scenes.csv")
    fields = []
    for r in rows:
        for kk in r:
            if kk not in fields:
                fields.append(kk)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({kk: (json.dumps(v) if isinstance(v, (list, dict))
                             else v) for kk, v in r.items()})
    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
