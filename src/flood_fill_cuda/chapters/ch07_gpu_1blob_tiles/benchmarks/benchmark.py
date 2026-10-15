"""Benchmark: tile BFS (ch07) head to head with ch03's BFS kernels.

One session, one interleaved round-robin per scene: every config timed
once per round, the order reversed on odd rounds (ch03's connectivity
benchmark policy - this laptop GPU drifts too much for A-then-B timing).
GPU_REPEATS = 9 rounds; medians, minimums and, for the headline pairs,
the median and range of the per-round ratio are reported. Before the
timed rounds each config runs once untimed, and that pass is the
correctness check: its painted image must equal ch03's for the same
connectivity.

The ch03 side is NOT timed through ch03's flood_fill(): that clock is
perf_counter around a launch and a host sync, and it allocates ~1 GB of
host arrays per call at 64 Mpx. Instead its kernels are launched here on
reusable device buffers, reset on the device between runs (visited,
depth, owner, stats fills plus the seed), and timed with CUDA events
like ch07. Both sides get the same treatment before every run: the
image is restored from a pristine device copy, a 96 MB buffer is READ
so neither side finds its input in the 32 MB L2 (the restore copy would
otherwise leave a 12 MB image warm for ch07's pack), and a short GPU
sleep kernel is queued so every timed launch is already in the stream
when the first event fires.

Which ch03 grid is fair? None is ch03's best everywhere, so several are
timed and the README says which one it quotes:
  ch03_wc            the instrumented warp-coop kernel at ch03's own
                     default and twin-comparison pin, 48 x 256 (the
                     cooperative maximum at tpb 256). Its counters pair
                     with ch07's.
  ch03_wc_bare       the uninstrumented twin at ITS default grid, 72 x 256
                     (it fits more blocks). The bare-vs-bare headline is
                     ch07_w4_bare against this.
  ch03_wc_120x128 /  other grids of the instrumented kernel; with
  ch03_wc_240x64 /   ch03_wc they make the per-scene best-grid column.
  ch03_wc_1x128      (1 x 128 only on the narrow scenes, where ch03's own
                     sweep found one block fastest; it would take seconds
                     on the big blobs.)
  ch03_conn4         4-conn, re-pinned to 48 x 128, which beat 48 x 256
                     on every scene of the fairness audit.
  ch03_conn8         the thread-layout 8-conn kernel at its default.
ch07's best-grid column takes the best of w2/w4/w8/cap96 and, on the
narrow scenes, the one-block grids 1 x 32 and 1 x 64.

What each ch07 number covers: `pack` (ch06's RGB -> 1 bit/px), `init`
(the per-run state, on the device), `rounds` (the cooperative tile
BFS) and `paint` (reached -> blue); `total` is the first event to the
last. ch03's kernel_ms covers its one launch: its device reset is off
its clock, and is timed separately (ch03_reset_ms; the part its painted
image needs, visited + seed, is ch03_reset_visited_ms). The ch07
end-to-end time (pageable H2D of the image + pipeline + D2H of the
painted image, reused buffers) is measured after the timed rounds.

Clocks. The GPU clock is spun up first (ch06's lesson: this GPU idles
low and does not boost for short kernels), and again for 1 s before each
scene. A background `nvidia-smi -lms 100` sampler records the SM and
memory clocks and the P-state for the whole session; each scene row
carries what they were during its timed rounds. The micro-probes in
probes.py run after the spin-up and land in the same JSON.

The @njit 8-conn CPU oracle (shared/cpu_oracle.py: allocates visited,
depth and a queue, computes depth, does not paint) is timed LAST, after
every GPU scene, rebuilding each scene: run between scenes, its seconds
of CPU-only work let the GPU drop to P4-P8 before the next scene's timed
rounds (session 0942's disk outlier).

Run:  uv run python -m flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks.benchmark
Writes tiles_<stamp>.json + tiles_<stamp>_scenes.csv to
results/ch07_gpu_1blob_tiles/benchmark_results/. Budget ~4-6 min.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import csv
import gc
import json
import statistics
import subprocess
import threading
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from numba import cuda, uint32
from numba.core.errors import NumbaPerformanceWarning

from .. import flood_fill as ff
from . import probes
from ..kernels import gpu_sleep_kernel
from ...ch03_gpu_1blob_nblock import scenes
from ...ch03_gpu_1blob_nblock import kernels as k3
from ...ch03_gpu_1blob_nblock import flood_fill as ch03_ff
from ...ch03_gpu_1blob_nblock.cpu_oracle import cpu_flood_fill_8
from ....shared import results_paths

warnings.simplefilter("ignore", NumbaPerformanceWarning)

RESULTS_DIR = results_paths.results_dir("ch07_gpu_1blob_tiles",
                                        "benchmark_results")

GPU_REPEATS = 9         # 5 left the median 5-40% apart between sessions
NJIT_REPEATS = 3        # the @njit 8-conn oracle, for context
E2E_REPEATS = 3         # ch07 end to end (H2D + pipeline + D2H)
SPIN_SECONDS = 8.0      # ch06's clock ramp
RESPIN_SECONDS = 4.0    # again after the probes, before the scenes
SCENE_SPIN_SECONDS = 1.0  # and before every scene's untimed pass
SLEEP_NAPS = 20         # ~2 ms of GPU sleep ahead of every timed sequence
L2_FLUSH_BYTES = 96 << 20   # 3x the 32 MB L2, read before every run
SMI_PERIOD_MS = 100

# (name, builder, note, narrow). narrow: a one-tile front (ch07's peak
# round is one tile), where one-block grids are timed too.
SCENES = [
    ("sq_2000_center", lambda: scenes.square_scene(2000, 2000, 1000, 1000),
     "1M px blob in a 4M px image", False),
    ("sq_4000_corner", lambda: scenes.full_red_scene(4000, 4000),
     "16M px, corner seed", False),
    ("disk_4001_r1900", lambda: scenes.disk_scene(4001, 4001, 1900),
     "11.3M px disk", False),
    ("serpentine_256", lambda: scenes.serpentine_scene(256, 256),
     "ch03's serpentine: 32,641 one-pixel levels at 8-conn", True),
    ("sq_6000_center", lambda: scenes.square_scene(6000, 6000, 6000, 6000),
     "36M px", False),
    ("sq_8000_center", lambda: scenes.square_scene(8000, 8000, 8000, 8000),
     "64M px: ch03 wc's 88.5 ms headline scene", False),
]

# ch03 configs: name -> (layout, connectivity, bare, tpb, blocks, narrow
# only). blocks=None is the kernel's own cooperative maximum at that tpb,
# which is what ch03's flood_fill() launches by default.
CH03 = {
    "ch03_conn4": ("thread", 4, False, 128, 48, False),
    "ch03_conn8": ("thread", 8, False, 256, None, False),
    "ch03_wc": ("warp", 8, False, 256, None, False),
    "ch03_wc_120x128": ("warp", 8, False, 128, 120, False),
    "ch03_wc_240x64": ("warp", 8, False, 64, 240, False),
    "ch03_wc_1x128": ("warp", 8, False, 128, 1, True),
    "ch03_wc_bare": ("warp", 8, True, 256, None, False),
}
CH03_WC_GRIDS = ["ch03_wc", "ch03_wc_120x128", "ch03_wc_240x64",
                 "ch03_wc_1x128"]

# ch07 configs: name -> TileFill kwargs. w2/w4/w8 are warps per block,
# uncapped: 144 x 64, 72 x 128 and 24 x 256 (the registers seat 288,
# 288 and 192 warps). w4_cap96 buys a bigger grid (480 warps) with a
# link-time register cap and pays in spills; w4_bare drops the counters;
# c4_w4 is the 4-connectivity twin; 1x32 / 1x64 are one-block grids for
# the narrow scenes.
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
    "ch07_1x32": dict(connectivity=8, threads_per_block=32, blocks=1,
                      max_registers=None),
    "ch07_1x64": dict(connectivity=8, threads_per_block=64, blocks=1,
                      max_registers=None),
}
CH07_NARROW_ONLY = {"ch07_1x32", "ch07_1x64"}
CH07_GRIDS = ["ch07_w2", "ch07_w4", "ch07_w8", "ch07_w4_cap96",
              "ch07_1x32", "ch07_1x64"]
CONN = {**{n: spec[1] for n, spec in CH03.items()},
        **{n: kw["connectivity"] for n, kw in CH07.items()}}
HEADLINE = "ch07_w4"      # the default flood_fill() configuration
# per-round ratios (numerator / denominator: > 1 = ch07 faster)
PAIRS = {
    "w4_vs_wc": ("ch03_wc", "ch07_w4"),
    "w4_bare_vs_wc_bare": ("ch03_wc_bare", "ch07_w4_bare"),
    "c4_w4_vs_conn4": ("ch03_conn4", "ch07_c4_w4"),
}


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


@cuda.jit
def _l2_flush(buf, sink):
    """Read every word of `buf` (a read: no dirty lines are left behind).
    The store is never taken (buf is zero) but keeps the loads alive."""
    acc = uint32(0)
    for i in range(cuda.grid(1), buf.shape[0], cuda.gridsize(1)):
        acc ^= buf[i]
    if acc == uint32(0x9E3779B9):
        sink[0] = acc


def _clocks():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,clocks.max.sm,"
             "pstate,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, mem, mx, ps, pw, tp = [x.strip() for x in out.split(",")]
        return {"sm_mhz": int(float(sm)), "mem_mhz": int(float(mem)),
                "max_sm_mhz": int(float(mx)), "pstate": ps,
                "power_w": float(pw), "temp_c": int(float(tp))}
    except Exception:
        return {}


class ClockSampler:
    """`nvidia-smi -lms` in the background: (t, sm MHz, mem MHz, P-state)
    every SMI_PERIOD_MS, so a scene's row can say what the clocks were
    while it was timed (a single query after the fact cannot). t is
    nvidia-smi's own timestamp (its stdout into a pipe may arrive in
    bursts), on the host's wall clock like time.time()."""

    def __init__(self, period_ms=SMI_PERIOD_MS):
        self.samples = []
        self.t0 = time.time()
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=timestamp,clocks.sm,clocks.mem,pstate",
                 "--format=csv,noheader,nounits", f"-lms={period_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
        except Exception:
            self.proc = None
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                ts, sm, mem, ps = [x.strip() for x in line.split(",")]
                t = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
                self.samples.append((t, int(float(sm)), int(float(mem)), ps))
            except ValueError:
                continue

    def window(self, t0, t1):
        """Clock summary over [t0, t1] (time.time() values)."""
        s = [x for x in list(self.samples) if t0 <= x[0] <= t1]
        if not s:
            return {"clk_samples": 0}
        sm = [x[1] for x in s]
        mem = [x[2] for x in s]
        return {"clk_samples": len(s),
                "clk_sm_min": min(sm), "clk_sm_median": statistics.median(sm),
                "clk_mem_min": min(mem),
                "clk_mem_median": statistics.median(mem),
                "clk_pstates": sorted({x[3] for x in s})}

    def trace(self):
        """[[seconds since start, sm MHz, mem MHz, P-state], ...]"""
        return [[round(x[0] - self.t0, 2), x[1], x[2], x[3]]
                for x in list(self.samples)]

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.thread.join(timeout=5)


def _one(d):
    return int(next(iter(d.values()))) if isinstance(d, dict) else int(d)


def _ch03_kernel(layout, conn, bare):
    return ch03_ff._KERNELS[(bare, conn, 1, layout)]


class Ch03Runner:
    """ch03's kernels on reusable device buffers, reset on the device.
    One buffer set serves every ch03 config of a scene (stats is sized
    for the largest grid)."""

    def __init__(self, width, height, max_blocks):
        n = width * height
        self.height = height
        self.visited = cuda.device_array((width, height), np.int32)
        self.depth = cuda.device_array((width, height), np.int32)
        self.owner = cuda.device_array((width, height), np.int16)
        self.queue = cuda.device_array(n, np.int32)
        self.q_state = cuda.device_array(1, np.int32)
        self.counters = cuda.device_array(k3.NUM_COUNTERS, np.int64)
        self.stats = cuda.device_array((max_blocks, 2), np.int64)
        self.trace = cuda.device_array(min(n, 2 ** 21), np.int32)
        self._flat = (self.visited.reshape(n), self.depth.reshape(n),
                      self.owner.reshape(n),
                      self.stats.reshape(max_blocks * 2))

    def reset(self, seed_x, seed_y):
        """Enqueue the reset; returns 3 events: start, visited + seed done
        (all the painted image needs), everything done."""
        v, d, o, s = self._flat
        ev = [cuda.event(timing=True) for _ in range(3)]
        ev[0].record()
        _fill[512, 256](v, np.int32(0))
        _ch03_seed[1, 32](v, self.queue, self.q_state, self.counters,
                          np.int64(int(seed_x) * self.height + int(seed_y)))
        ev[1].record()
        _fill[512, 256](d, np.int32(-1))
        _fill[512, 256](o, np.int16(-1))
        _fill[4, 256](s, np.int64(0))
        ev[2].record()
        return ev

    def run(self, kernel, bare, blocks, tpb, d_img):
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        if bare:
            kernel[blocks, tpb](d_img, self.visited, self.depth, self.queue,
                                self.q_state, self.counters)
        else:
            kernel[blocks, tpb](
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


def _ratio_stats(num, den):
    r = [a / b for a, b in zip(num, den)]
    return statistics.median(r), min(r), max(r)


# ------------------------------------------------------------- scenes

def bench_scene(name, builder, note, narrow, ch03_launch, flush, sampler):
    img, sx, sy = builder()
    width, height = img.shape[0], img.shape[1]
    d_pristine = cuda.to_device(img)
    d_img = cuda.device_array_like(img)
    ch03_names = [n for n in CH03 if narrow or not CH03[n][5]]
    ch07_names = [n for n in CH07 if narrow or n not in CH07_NARROW_ONLY]
    r3 = Ch03Runner(width, height,
                    max(ch03_launch[n]["blocks"] for n in ch03_names))
    engines = {n: ff.TileFill(width, height, **CH07[n]) for n in ch07_names}
    names = ch03_names + ch07_names
    sink = cuda.device_array(1, np.uint32)

    def run_once(n):
        """Restore, reset (ch03), flush L2, queue the sleep, enqueue the
        timed work; returns the timings once the stream is done."""
        d_img.copy_to_device(d_pristine)
        rev = r3.reset(sx, sy) if n in CH03 else None
        _l2_flush[512, 256](flush, sink)
        cuda.synchronize()
        gpu_sleep_kernel[1, 32](SLEEP_NAPS)
        if n in CH03:
            c = ch03_launch[n]
            e0, e1 = r3.run(c["kernel"], c["bare"], c["blocks"], c["tpb"],
                            d_img)
            e1.synchronize()
            return {"total": cuda.event_elapsed_time(e0, e1),
                    "reset": cuda.event_elapsed_time(rev[0], rev[2]),
                    "reset_visited": cuda.event_elapsed_time(rev[0], rev[1])}
        _, events = engines[n].run(d_img, sx, sy)
        events[-1].synchronize()
        phases, total = ff.phase_times(events)
        return {"total": total, **phases}

    _spin_up(SCENE_SPIN_SECONDS)

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
    ref8 = refs[8]
    del refs
    gc.collect()

    # -- timed rounds, interleaved, order reversed on odd rounds
    samples = {n: [] for n in names}
    t_timed0 = time.time()
    for i in range(GPU_REPEATS):
        order = names if i % 2 == 0 else list(reversed(names))
        for n in order:
            samples[n].append(run_once(n))
    t_timed1 = time.time()

    # -- ch07 end to end: pageable H2D + pipeline + D2H, reused buffers
    host_out = np.empty_like(img)
    e2e = []
    for _ in range(E2E_REPEATS):
        cuda.synchronize()
        t0 = time.perf_counter()
        d_img.copy_to_device(img)
        engines[HEADLINE].run(d_img, sx, sy)
        d_img.copy_to_host(host_out)
        e2e.append((time.perf_counter() - t0) * 1000)
    e2e_ok = bool(np.array_equal(host_out, ref8))
    del host_out, ref8

    row = {"scene": name, "note": note, "width": width, "height": height,
           "seed": [int(sx), int(sy)], "narrow": narrow,
           "filled": meta["ch03_conn8"]["filled"],
           "tiles": engines[HEADLINE].n_tiles,
           "configs_timed": names}
    row.update(sampler.window(t_timed0, t_timed1))
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
    row["ch03_reset_ms"] = statistics.median(
        s["reset"] for s in samples["ch03_wc"])
    row["ch03_reset_visited_ms"] = statistics.median(
        s["reset_visited"] for s in samples["ch03_wc"])

    wc = "ch03_wc"
    for n in ch07_names:
        if CONN[n] == 8:
            row[f"{n}_speedup_vs_wc"] = row[f"{wc}_ms"] / row[f"{n}_ms"]
            row[f"{n}_speedup_vs_wc_min"] = (row[f"{wc}_ms_min"]
                                             / row[f"{n}_ms_min"])
            row[f"{n}_rounds_speedup_vs_wc"] = (row[f"{wc}_ms"]
                                                / row[f"{n}_rounds_ms"])
    row["ch07_w4_bare_speedup_vs_wc_bare"] = (row["ch03_wc_bare_ms"]
                                              / row["ch07_w4_bare_ms"])
    row["ch07_w4_bare_speedup_vs_wc_bare_min"] = (
        row["ch03_wc_bare_ms_min"] / row["ch07_w4_bare_ms_min"])
    row["ch07_c4_w4_speedup_vs_conn4"] = (row["ch03_conn4_ms"]
                                          / row["ch07_c4_w4_ms"])
    row["ch07_c4_w4_speedup_vs_conn4_min"] = (row["ch03_conn4_ms_min"]
                                              / row["ch07_c4_w4_ms_min"])
    for key, (num, den) in PAIRS.items():
        med, lo, hi = _ratio_stats([s["total"] for s in samples[num]],
                                   [s["total"] for s in samples[den]])
        row[f"paired_{key}_median"] = med
        row[f"paired_{key}_min"] = lo
        row[f"paired_{key}_max"] = hi
    # per-scene best grid, each side over its own grids (medians)
    b3 = min((n for n in CH03_WC_GRIDS if n in names),
             key=lambda n: row[f"{n}_ms"])
    b7 = min((n for n in CH07_GRIDS if n in names),
             key=lambda n: row[f"{n}_ms"])
    row["ch03_wc_best_config"] = b3
    row["ch03_wc_best_ms"] = row[f"{b3}_ms"]
    row["ch07_best_config"] = b7
    row["ch07_best_ms"] = row[f"{b7}_ms"]
    row["best_vs_best_speedup"] = row["ch03_wc_best_ms"] / row["ch07_best_ms"]

    row[f"{HEADLINE}_e2e_ms"] = statistics.median(e2e)
    row[f"{HEADLINE}_e2e_ms_min"] = min(e2e)
    row[f"{HEADLINE}_e2e_matches_ch03"] = e2e_ok

    lv8 = meta["ch03_wc"]["levels"]
    row["levels8_per_round"] = lv8 / meta[HEADLINE]["rounds"]
    row["levels4_per_round"] = (meta["ch03_conn4"]["levels"]
                                / meta["ch07_c4_w4"]["rounds"])
    h = meta[HEADLINE]
    row["visits_per_tile_touched"] = h["tile_visits"] / h["tiles_touched"]
    row["or_atomics_per_px"] = h["or_atomics"] / row["filled"]
    row["stamp_atomics_per_px"] = h["stamp_atomics"] / row["filled"]
    row["ch03_cas_per_px"] = meta["ch03_wc"]["cas_attempts"] / row["filled"]
    row["closure_iters_per_visit"] = h["closure_iters"] / h["tile_visits"]
    row["rounds_us_per_round"] = (1000 * row[f"{HEADLINE}_rounds_ms"]
                                  / h["rounds"])
    row["ch03_wc_us_per_level"] = 1000 * row["ch03_wc_ms"] / lv8
    hp = row[f"{HEADLINE}_pack_ms"]
    hpaint = row[f"{HEADLINE}_paint_ms"]
    row["pack_gb_s"] = 3 * width * height / hp / 1e6
    row["paint_gb_s"] = 3 * row["filled"] / hpaint / 1e6
    row["all_paint_match"] = all(check.values()) and e2e_ok

    print(f"{name:16s} {row['filled']:>11,d} "
          f"wc {row['ch03_wc_ms']:8.2f} w4 {row['ch07_w4_ms']:7.3f} "
          f"x{row['ch07_w4_speedup_vs_wc']:6.1f} | "
          f"bare {row['ch03_wc_bare_ms']:8.2f} / {row['ch07_w4_bare_ms']:7.3f} "
          f"x{row['ch07_w4_bare_speedup_vs_wc_bare']:6.1f} | "
          f"best {b3}/{b7} x{row['best_vs_best_speedup']:6.1f} | "
          f"lv {lv8:>6,d} -> rounds {h['rounds']:>5d} "
          f"sm {row.get('clk_sm_min', '?')}-{row.get('clk_sm_median', '?')} "
          f"mem {row.get('clk_mem_min', '?')} "
          f"{'OK' if row['all_paint_match'] else 'MISMATCH'}", flush=True)

    del engines, r3, d_img, d_pristine, img, sink
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass
    return row


def njit_pass(scene_list, rows):
    """The @njit 8-conn oracle per scene, after ALL GPU work (see the
    module docstring), rebuilding each scene's image."""
    by_name = {r["scene"]: r for r in rows}
    for name, builder, _note, _narrow in scene_list:
        row = by_name[name]
        img, sx, sy = builder()
        times = []
        oracle_filled = None
        for _ in range(NJIT_REPEATS):
            t0 = time.perf_counter()
            _, _, _, oracle_filled = cpu_flood_fill_8(img, sx, sy)
            times.append((time.perf_counter() - t0) * 1000)
            gc.collect()
        row["njit8_ms"] = statistics.median(times)
        row["njit8_ms_min"] = min(times)
        row["njit8_filled_matches"] = bool(
            oracle_filled == row["filled"] == row[f"{HEADLINE}_filled"])
        row[f"{HEADLINE}_speedup_vs_njit8"] = (row["njit8_ms"]
                                               / row[f"{HEADLINE}_ms"])
        row[f"{HEADLINE}_e2e_speedup_vs_njit8"] = (
            row["njit8_ms"] / row[f"{HEADLINE}_e2e_ms"])
        row["ch03_wc_speedup_vs_njit8"] = row["njit8_ms"] / row["ch03_wc_ms"]
        row["all_paint_match"] = (row["all_paint_match"]
                                  and row["njit8_filled_matches"])
        print(f"  {name:16s} njit {row['njit8_ms']:9.2f} ms "
              f"(kernel-only ch07 x{row[f'{HEADLINE}_speedup_vs_njit8']:7.1f}, "
              f"end to end x{row[f'{HEADLINE}_e2e_speedup_vs_njit8']:6.1f}) "
              f"{'OK' if row['njit8_filled_matches'] else 'MISMATCH'}",
              flush=True)
        del img
        gc.collect()


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
    config_meta = {}
    ch03_launch = {}
    for n, (layout, conn, bare, tpb, blocks, narrow) in CH03.items():
        kern = _ch03_kernel(layout, conn, bare)
        cap = ch03_ff.max_blocks(threads_per_block=tpb, bare=bare,
                                 connectivity=conn, probe_layout=layout)
        b = cap if blocks is None else blocks
        if b > cap:
            raise RuntimeError(f"{n}: {b} blocks exceed the cooperative "
                               f"capacity {cap} at tpb {tpb}")
        ch03_launch[n] = {"kernel": kern, "bare": bare, "blocks": b,
                          "tpb": tpb}
        config_meta[n] = {"connectivity": conn, "layout": layout,
                          "bare": bare, "tpb": tpb, "blocks": b,
                          "warps": b * tpb // 32,
                          "coop_max_blocks": cap,
                          "default_grid": blocks is None,
                          "narrow_scenes_only": narrow,
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
            "narrow_scenes_only": n in CH07_NARROW_ONLY,
            "regs_per_thread": ff.regs_per_thread(
                kw["connectivity"], bare, kw["max_registers"]),
            "local_bytes_per_thread": ff.local_bytes_per_thread(
                kw["connectivity"], bare, kw["max_registers"]),
        }
        del eng
    for n, m in config_meta.items():
        print(f"  {n:16s} " + " ".join(f"{k}={v}" for k, v in m.items()))
    flush = cuda.to_device(np.zeros(L2_FLUSH_BYTES // 4, np.uint32))
    _l2_flush[512, 256](flush, cuda.device_array(1, np.uint32))
    probes.warm()
    cpu_flood_fill_8(warm, wx, wy)   # compile the oracle off the clock
    cuda.synchronize()

    sampler = ClockSampler()
    print(f"Spinning the clock up for {SPIN_SECONDS:.0f} s...")
    _spin_up()
    clocks_start = _clocks()
    print(f"  clocks: {clocks_start}")

    print("Probes (latency, column fill, barrier ritual, registers)...")
    grids = sorted({(m["blocks"], m["tpb"]) for m in config_meta.values()})
    probe_res = probes.run_all(grids)
    for k, v in probe_res.items():
        print(f"  {k}: {v}")
    # The single-warp probes let the clock relax; ramp it again.
    _spin_up(RESPIN_SECONDS)
    clocks_scenes = _clocks()
    print(f"  clocks before the scenes: {clocks_scenes}")

    print(f"\n{'scene':16s} {'filled':>11s}")
    rows = []
    for name, builder, note, narrow in scene_list:
        rows.append(bench_scene(name, builder, note, narrow, ch03_launch,
                                flush, sampler))
    clocks_end = _clocks()
    print("\n@njit 8-conn oracle, after all GPU scenes:")
    njit_pass(scene_list, rows)
    sampler.stop()
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
            "warp-coop BFS kernels (instrumented and bare, several grids), "
            "one interleaved round-robin per scene, CUDA-event timings on "
            "every side"),
        "headline_config": HEADLINE,
        "configs": config_meta,
        "config": {"gpu_repeats": GPU_REPEATS, "njit_repeats": NJIT_REPEATS,
                   "e2e_repeats": E2E_REPEATS,
                   "spin_seconds": SPIN_SECONDS,
                   "respin_seconds": RESPIN_SECONDS,
                   "scene_spin_seconds": SCENE_SPIN_SECONDS,
                   "sleep_naps": SLEEP_NAPS,
                   "l2_flush_bytes": L2_FLUSH_BYTES,
                   "smi_period_ms": SMI_PERIOD_MS},
        "timing_note": (
            "ch07 *_ms = pack + init + rounds + paint (first event to "
            "last). ch03 *_ms = its one kernel launch; its device reset "
            "is off the clock and timed separately (ch03_reset_ms). Every "
            "run: image restored from a device copy, 96 MB L2 flush read, "
            "GPU sleep kernel queued ahead so host launch overhead is "
            "excluded. ch07_w4_e2e_ms = pageable H2D + pipeline + D2H of "
            "the painted image (perf_counter, reused buffers). njit8_ms = "
            "the @njit oracle end to end (allocations, depth map, no "
            "paint), timed after all GPU scenes."),
        "probes": probe_res,
        "clocks_start": clocks_start,
        "clocks_before_scenes": clocks_scenes,
        "clocks_end": clocks_end,
        "clock_trace": sampler.trace(),
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
