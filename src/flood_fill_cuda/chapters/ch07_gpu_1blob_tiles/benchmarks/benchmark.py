"""Benchmark: tile BFS (ch07) head to head with ch03's best BFS kernels.

One session, one interleaved round-robin per scene: every config timed
once per round, the order reversed on odd rounds (ch03's connectivity
benchmark policy - this laptop GPU drifts too much for A-then-B timing),
GPU_REPEATS = 5 rounds, medians and minimums reported. Before the timed
rounds each config runs once untimed, and that pass is the correctness
check: its painted image must equal ch03's for the same connectivity.

The ch03 side is NOT timed through ch03's flood_fill(): that clock is
perf_counter around a launch and a host sync, and it allocates ~1 GB of
host arrays per call at 64 Mpx. Instead its three kernels are launched
here on reusable device buffers, reset on the device between runs, and
timed with CUDA events like ch07. Both sides get the same treatment: a
short GPU sleep kernel is queued first, so every timed launch is already
in the stream when the first event fires and Python's launch overhead
stays outside the interval. ch03 runs at the grid its own session used
(48 x 256: the cooperative capacity of all three kernels at tpb 256).

What each ch07 number covers: `pack` (ch06's RGB -> 1 bit/px), `init`
(the per-run state, on the device), `rounds` (the cooperative tile
BFS) and `paint` (reached -> blue); `total` is the first event to the
last. ch03's kernel_ms covers its one launch only; its own buffer
initialisation (visited, depth and owner: 640 MB of host copies at 64
Mpx) is not on its clock. So the totals compared below favour ch03.

The GPU clock is spun up first (ch06's lesson: this GPU idles at 1470
of 3105 MHz and does not boost for short kernels), and nvidia-smi's
reading is recorded at the start and the end. Then the micro-probes in
probes.py run (barrier ritual per grid, column fill with and without
syncwarp, dependent load/atomic latency, standalone vs linked
registers) and land in the same JSON, so every number the README argues
from comes from one session.

Run:  uv run python -m flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks.benchmark
Writes tiles_<stamp>.json + tiles_<stamp>_scenes.csv to
results/ch07_gpu_1blob_tiles/benchmark_results/. Budget ~1-2 min.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import csv
import gc
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
from . import probes
from ..kernels import gpu_sleep_kernel
from ...ch03_gpu_1blob_nblock import scenes
from ...ch03_gpu_1blob_nblock import kernels as k3
from ....shared import results_paths

warnings.simplefilter("ignore", NumbaPerformanceWarning)

RESULTS_DIR = results_paths.results_dir("ch07_gpu_1blob_tiles",
                                        "benchmark_results")

GPU_REPEATS = 5         # ch03's connectivity benchmark policy
SPIN_SECONDS = 8.0      # ch06's clock ramp
RESPIN_SECONDS = 4.0    # again after the probes, before the scenes
SLEEP_NAPS = 20         # ~2 ms of GPU sleep ahead of every timed sequence
CH03_TPB = 256

SCENES = [
    ("sq_2000_center", lambda: scenes.square_scene(2000, 2000, 1000, 1000),
     "1M px blob in a 4M px image"),
    ("sq_4000_corner", lambda: scenes.full_red_scene(4000, 4000),
     "16M px, corner seed"),
    ("disk_4001_r1900", lambda: scenes.disk_scene(4001, 4001, 1900),
     "11.3M px disk"),
    ("serpentine_256", lambda: scenes.serpentine_scene(256, 256),
     "ch03's serpentine: 32,641 one-pixel levels at 8-conn"),
    ("sq_6000_center", lambda: scenes.square_scene(6000, 6000, 6000, 6000),
     "36M px"),
    ("sq_8000_center", lambda: scenes.square_scene(8000, 8000, 8000, 8000),
     "64M px: ch03 wc's 88.5 ms headline scene"),
]

# ch03 kernels, all at 48 x 256.
CH03 = {
    "ch03_conn4": (4, k3.multi_block_global_kernel),
    "ch03_conn8": (8, k3.multi_block_global8_kernel),
    "ch03_wc": (8, k3.multi_block_global8wc_kernel),
}

# ch07 configs: name -> TileFill kwargs. w2/w4/w8 are warps per block,
# uncapped: 144 x 64, 72 x 128 and 24 x 256 (the registers seat 288,
# 288 and 192 warps). w4_cap96 buys a bigger grid (480 warps) with a
# link-time register cap and pays in spills; w4_bare drops the counters;
# c4_w4 is the 4-connectivity twin.
CH07 = {
    "ch07_w2": dict(connectivity=8, threads_per_block=64, max_registers=None),
    "ch07_w4": dict(connectivity=8, threads_per_block=128, max_registers=None),
    "ch07_w8": dict(connectivity=8, threads_per_block=256, max_registers=None),
    "ch07_w4_cap96": dict(connectivity=8, threads_per_block=128,
                          max_registers=96),
    "ch07_w4_bare": dict(connectivity=8, threads_per_block=128,
                         max_registers=None, bare=True),
    "ch07_c4_w4": dict(connectivity=4, threads_per_block=128,
                       max_registers=None),
}
CONN = {**{n: c for n, (c, _) in CH03.items()},
        **{n: kw["connectivity"] for n, kw in CH07.items()}}
HEADLINE = "ch07_w4"      # the default flood_fill() configuration


# ------------------------------------------------------------ helpers

@cuda.jit
def _fill(a, v):
    for i in range(cuda.grid(1), a.shape[0], cuda.gridsize(1)):
        a[i] = v


@cuda.jit
def _ch03_seed(visited_flat, queue, q_state, counters, seed_lin):
    if cuda.grid(1) == 0:
        visited_flat[seed_lin] = 1
        queue[0] = seed_lin
        q_state[0] = 1
        for c in range(counters.shape[0]):
            counters[c] = 0


def _clocks():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,"
             "temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, mx, pw, tp = [x.strip() for x in out.split(",")]
        return {"sm_mhz": int(float(sm)), "max_sm_mhz": int(float(mx)),
                "power_w": float(pw), "temp_c": int(float(tp))}
    except Exception:
        return {}


def _one(d):
    return int(next(iter(d.values()))) if isinstance(d, dict) else int(d)


class Ch03Runner:
    """ch03's kernels on reusable device buffers, reset on the device."""

    def __init__(self, width, height, blocks):
        n = width * height
        self.height = height
        self.blocks = blocks
        self.visited = cuda.device_array((width, height), np.int32)
        self.depth = cuda.device_array((width, height), np.int32)
        self.owner = cuda.device_array((width, height), np.int16)
        self.queue = cuda.device_array(n, np.int32)
        self.q_state = cuda.device_array(1, np.int32)
        self.counters = cuda.device_array(k3.NUM_COUNTERS, np.int64)
        self.stats = cuda.device_array((blocks, 2), np.int64)
        self.trace = cuda.device_array(min(n, 2 ** 21), np.int32)
        self._flat = (self.visited.reshape(n), self.depth.reshape(n),
                      self.owner.reshape(n), self.stats.reshape(blocks * 2))

    def reset(self, seed_x, seed_y):
        v, d, o, s = self._flat
        _fill[512, 256](v, np.int32(0))
        _fill[512, 256](d, np.int32(-1))
        _fill[512, 256](o, np.int16(-1))
        _fill[4, 256](s, np.int64(0))
        _ch03_seed[1, 32](v, self.queue, self.q_state, self.counters,
                          np.int64(seed_x * self.height + seed_y))

    def run(self, kernel, d_img):
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        kernel[self.blocks, CH03_TPB](
            d_img, self.visited, self.depth, self.owner, self.queue,
            self.q_state, self.counters, self.stats, self.trace)
        e1.record()
        return e0, e1

    def read(self):
        c = self.counters.copy_to_host()
        return {"levels": int(c[k3.LEVELS]), "filled": int(c[k3.FILLED]),
                "cas_attempts": int(c[k3.CAS_ATTEMPTS])}


def _spin_up(seconds=SPIN_SECONDS):
    """Back-to-back ch07 fills on a mid-size scene, no host sync inside
    the inner loop, until the clock has had `seconds` to ramp."""
    img, sx, sy = scenes.full_red_scene(2048, 2048)
    d_img = cuda.to_device(img)
    eng = ff.TileFill(2048, 2048, **CH07[HEADLINE])
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            eng.run(d_img, sx, sy)
        cuda.synchronize()
    del eng, d_img


# ------------------------------------------------------------- scenes

def bench_scene(name, builder, note, ch03_blocks):
    img, sx, sy = builder()
    width, height = img.shape[0], img.shape[1]
    d_pristine = cuda.to_device(img)
    d_img = cuda.device_array_like(img)
    r3 = Ch03Runner(width, height, ch03_blocks)
    engines = {n: ff.TileFill(width, height, **kw) for n, kw in CH07.items()}
    names = list(CH03) + list(CH07)

    def run_once(n):
        """Reset, queue the sleep, enqueue the timed work; returns a
        callable that yields the timings once the stream is done."""
        d_img.copy_to_device(d_pristine)
        if n in CH03:
            r3.reset(sx, sy)
        cuda.synchronize()
        gpu_sleep_kernel[1, 32](SLEEP_NAPS)
        if n in CH03:
            e0, e1 = r3.run(CH03[n][1], d_img)
            e1.synchronize()
            return {"total": cuda.event_elapsed_time(e0, e1)}
        _, events = engines[n].run(d_img, sx, sy)
        events[-1].synchronize()
        phases, total = ff.phase_times(events)
        return {"total": total, **phases}

    # -- untimed pass: correctness (painted image vs ch03's) + counters
    refs = {}
    meta = {}
    check = {}
    for n in names:
        run_once(n)
        out = d_img.copy_to_host()
        conn = CONN[n]
        if n in CH03:
            meta[n] = r3.read()
            if conn not in refs:
                refs[conn] = out
        else:
            meta[n] = engines[n].read_counters()
            meta[n]["filled"] = engines[n].filled()
            meta[n]["tiles_touched"] = engines[n].tiles_touched()
        check[n] = bool(np.array_equal(out, refs[conn]))
        del out
    del refs
    gc.collect()

    # -- timed rounds, interleaved, order reversed on odd rounds
    samples = {n: [] for n in names}
    for i in range(GPU_REPEATS):
        order = names if i % 2 == 0 else list(reversed(names))
        for n in order:
            samples[n].append(run_once(n))

    row = {"scene": name, "note": note, "width": width, "height": height,
           "seed": [int(sx), int(sy)],
           "filled": meta["ch03_conn8"]["filled"],
           "tiles": engines[HEADLINE].n_tiles}
    for n in names:
        tot = [s["total"] for s in samples[n]]
        row[f"{n}_ms"] = statistics.median(tot)
        row[f"{n}_ms_min"] = min(tot)
        row[f"{n}_ms_max"] = max(tot)
        row[f"{n}_ms_stdev"] = statistics.stdev(tot)
        if n in CH07:
            for ph in ff.PHASES:
                row[f"{n}_{ph}_ms"] = statistics.median(
                    s[ph] for s in samples[n])
        for k, v in meta[n].items():
            row[f"{n}_{k}"] = v
        row[f"{n}_paint_matches_ch03"] = check[n]

    wc = "ch03_wc"
    for n in CH07:
        if CONN[n] == 8:
            row[f"{n}_speedup_vs_wc"] = row[f"{wc}_ms"] / row[f"{n}_ms"]
            row[f"{n}_speedup_vs_wc_min"] = (row[f"{wc}_ms_min"]
                                             / row[f"{n}_ms_min"])
            row[f"{n}_rounds_speedup_vs_wc"] = (row[f"{wc}_ms"]
                                                / row[f"{n}_rounds_ms"])
    row["ch07_c4_w4_speedup_vs_conn4"] = (row["ch03_conn4_ms"]
                                          / row["ch07_c4_w4_ms"])
    row["ch07_c4_w4_speedup_vs_conn4_min"] = (row["ch03_conn4_ms_min"]
                                              / row["ch07_c4_w4_ms_min"])
    lv8 = meta["ch03_wc"]["levels"]
    row["levels8_per_round"] = lv8 / meta[HEADLINE]["rounds"]
    row["levels4_per_round"] = (meta["ch03_conn4"]["levels"]
                                / meta["ch07_c4_w4"]["rounds"])
    h = meta[HEADLINE]
    row["visits_per_tile_touched"] = h["tile_visits"] / h["tiles_touched"]
    row["or_atomics_per_px"] = h["or_atomics"] / row["filled"]
    row["ch03_cas_per_px"] = meta["ch03_wc"]["cas_attempts"] / row["filled"]
    row["closure_iters_per_visit"] = h["closure_iters"] / h["tile_visits"]
    hp = row[f"{HEADLINE}_pack_ms"]
    hpaint = row[f"{HEADLINE}_paint_ms"]
    row["pack_gb_s"] = 3 * width * height / hp / 1e6
    row["paint_gb_s"] = 3 * row["filled"] / hpaint / 1e6
    row["all_paint_match"] = all(check.values())

    print(f"{name:16s} {row['filled']:>11,d} wc {row['ch03_wc_ms']:8.2f} "
          f"w4 {row['ch07_w4_ms']:7.3f} (rounds {row['ch07_w4_rounds_ms']:6.3f}) "
          f"x{row['ch07_w4_speedup_vs_wc']:6.1f} "
          f"lv {lv8:>6,d} -> rounds {h['rounds']:>4d} "
          f"{'OK' if row['all_paint_match'] else 'MISMATCH'}", flush=True)

    del engines, r3, d_img, d_pristine, img
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="run only these scenes (default: all)")
    ap.add_argument("--no-save", action="store_true",
                    help="print only; write no JSON/CSV (smoke runs)")
    args = ap.parse_args()
    scene_list = [s for s in SCENES
                  if args.scenes is None or s[0] in args.scenes]

    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    sm_count = int(device.MULTIPROCESSOR_COUNT)
    print(f"Device: {dev_name.strip()} ({sm_count} SMs)")

    # -- compile everything, record registers and capacities
    warm, wx, wy = scenes.square_scene(64, 64, 32, 32)
    from ...ch03_gpu_1blob_nblock.flood_fill import (
        max_blocks as ch03_max_blocks)
    ch03_caps = {n: ch03_max_blocks(threads_per_block=CH03_TPB,
                                    connectivity=c,
                                    probe_layout="warp" if n == "ch03_wc"
                                    else "thread")
                 for n, (c, _) in CH03.items()}
    ch03_blocks = min(ch03_caps.values())
    config_meta = {}
    for n, (c, kern) in CH03.items():
        config_meta[n] = {"connectivity": c, "tpb": CH03_TPB,
                          "blocks": ch03_blocks,
                          "coop_max_blocks": ch03_caps[n],
                          "regs_per_thread": _one(kern.get_regs_per_thread()),
                          "local_bytes_per_thread": _one(
                              kern.get_local_mem_per_thread())}
    for n, kw in CH07.items():
        eng = ff.TileFill(64, 64, **kw)
        bare = kw.get("bare", False)
        config_meta[n] = {
            "connectivity": kw["connectivity"],
            "tpb": kw["threads_per_block"],
            "blocks": eng.blocks,
            "warps": eng.blocks * kw["threads_per_block"] // 32,
            "max_registers": kw["max_registers"],
            "bare": bare,
            "coop_max_blocks": eng.coop_max_blocks,
            "regs_per_thread": ff.regs_per_thread(
                kw["connectivity"], bare, kw["max_registers"]),
            "local_bytes_per_thread": ff.local_bytes_per_thread(
                kw["connectivity"], bare, kw["max_registers"]),
        }
        del eng
    for n, m in config_meta.items():
        print(f"  {n:14s} " + " ".join(f"{k}={v}" for k, v in m.items()))

    print(f"Spinning the clock up for {SPIN_SECONDS:.0f} s...")
    _spin_up()
    clocks_start = _clocks()
    print(f"  clocks: {clocks_start}")

    print("Probes (barrier ritual, column fill, latency, registers)...")
    grids = sorted({(m["blocks"], m["tpb"]) for m in config_meta.values()})
    probe_res = probes.run_all(grids)
    for k, v in probe_res.items():
        print(f"  {k}: {v}")
    # The probes are single-warp kernels the clock relaxes under; without
    # a second ramp the first scene ran ~1.5-3x slow on BOTH sides.
    _spin_up(RESPIN_SECONDS)
    clocks_scenes = _clocks()
    print(f"  clocks before the scenes: {clocks_scenes}")

    print(f"\n{'scene':16s} {'filled':>11s}")
    rows = []
    for name, builder, note in scene_list:
        rows.append(bench_scene(name, builder, note, ch03_blocks))
    clocks_end = _clocks()
    if args.no_save:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "device": dev_name.strip(),
        "sm_count": sm_count,
        "experiment": (
            "tile BFS (ch07: 32x32 tiles closed in registers, rounds of "
            "tiles, two grid.sync per round) vs ch03's conn4 / conn8 / "
            "warp-coop BFS kernels, one interleaved round-robin per "
            "scene, CUDA-event timings on both sides"),
        "headline_config": HEADLINE,
        "configs": config_meta,
        "config": {"gpu_repeats": GPU_REPEATS, "spin_seconds": SPIN_SECONDS,
                   "respin_seconds": RESPIN_SECONDS,
                   "sleep_naps": SLEEP_NAPS, "ch03_tpb": CH03_TPB,
                   "ch03_blocks": ch03_blocks},
        "timing_note": (
            "ch07 *_ms = pack + init + rounds + paint (first event to "
            "last). ch03 *_ms = its one kernel launch; its buffer "
            "initialisation is off the clock. Both sides: a GPU sleep "
            "kernel is queued ahead so host launch overhead is excluded."),
        "probes": probe_res,
        "clocks_start": clocks_start,
        "clocks_before_scenes": clocks_scenes,
        "clocks_end": clocks_end,
        "scenes": rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"tiles_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, f"tiles_{stamp}_scenes.csv")
    fieldnames = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                             for k, v in row.items()})

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
