"""Benchmark: tile BFS (ch07) head to head with ch03's BFS and ch06's CCL.

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
timed, and "ch03's best" is per scene the fastest of ALL its warp-coop
grids, bare and instrumented (the fairness audit: the bare kernel at
48 x 256 or 1 x 128 beat every instrumented grid on the narrow scenes,
and the bare one at its own 72 x 256 won the blobs):
  ch03_wc            the instrumented warp-coop kernel at ch03's own
                     default and twin-comparison pin, 48 x 256 (the
                     cooperative maximum at tpb 256). Its counters pair
                     with ch07's.
  ch03_wc_bare       the uninstrumented twin at ITS default grid, 72 x 256
                     (it fits more blocks).
  ch03_wc_bare_48x256, ch03_wc_bare_1x128 (narrow scenes only)
                     the bare twin at the instrumented kernel's grids.
  ch03_wc_120x128 /  other grids of the instrumented kernel.
  ch03_wc_240x64 /
  ch03_wc_1x128      (1 x 128 only on the narrow scenes, where ch03's own
                     sweep found one block fastest; it would take seconds
                     on the big blobs.)
  ch03_conn4         4-conn, re-pinned to 48 x 128, which beat 48 x 256
                     on every scene of the fairness audit.
  ch03_conn8         the thread-layout 8-conn kernel at its default.
Every ch03 kernel pays two grid.sync per level; ch03 has no one-sync
twin, so on the barrier-bound scenes part of the gap to ch07's default
is that ritual (the README prices it from the probes).

`ch06_runs_bare` is the CCL reference: ch06's run-table pipeline (pack,
count, scan, emit, merge, flatten, paint: union-find over horizontal
runs, 8-connected, no seed, no BFS) on its own reusable buffers, in
ch06's rgb contract (RGB on the device in, painted RGB out), with its
counters off (instrumented=False), timed with the same restore, flush,
sleep, events and interleaving. `ch06_runs` is the instrumented twin
(its counters: runs, blobs, union attempts), timed too: the audit found
it 4-18% slower than the bare one, which is the twin the bare ch07
configs pair with. It labels every red blob; on these one-blob scenes
that is the same job. It paints palette[label % 6], not blue, so its
check is the set of pixels it CHANGED, which must equal the set ch03's
8-conn fill changed. Its run table is sized to the scene's run count
when ch06's n/16 heuristic is too small (the maze): ch06's recolor()
would resize and rerun.

Scenes: ch03's six (the README's), four worst cases from the chapter's
perf-adversary review (worst_cases.py): a serpentine of zigzags on the
tile borders (one tile per round), a zigzag lattice on every border
(rounds == levels with a wide front), a 1-px maze, and a 524288 x 32
strip with a 32x32 blob (pack/paint dominated); a 5,000-px disk in a
64 Mpx image (pack/paint dominated, square); and a 2000 x 2000 square
seeded in its corner (a front narrower than the grid for most rounds,
where the fused pack lost most).

The switches (flood_fill.SWITCHES) are timed as an ablation in the same
rounds, every config pinning every switch. Each switch is judged by its
own increment on top of the ones adopted before it (the adoption rule:
adopt only if it loses nothing measurable):
  ch07_w4_bare          the old default's bare twin (all switches off)
  ch07_w4_bare_1s       one_sync: one grid.sync per round
  ch07_w4_bare_1s_tp    + tile_paint: paint_tiles_kernel after the
                        separate pack, untouched tiles skipped
  ch07_w4_bare_1s_fp    one_sync + fused_pack (the RGB read inside the
                        rounds; implies the tile paint)
  ch07_w4_bare_1s_tp_sk one_sync + tile_paint + skip_full
NEW is the bare twin whose switches are flood_fill.DEFAULT_SWITCHES (the
adopted default) and HEADLINE its instrumented twin: what flood_fill()
runs. Instrumented twins (ch07_w4, ch07_w4_1s_tp, ...) give the counters
(visits, packed tiles, skipped exports). ch07_c4_w4 / ch07_c4_w4_1s_tp
are the 4-conn twins of the old and the new default (flood_fill()'s
default connectivity is 4). On the narrow scenes, the old and the new
default also run at one block (ch07_1x32_bare, ch07_1x32_bare_1s_tp).
Paired per-round ratios (PAIRS) set every increment against the config
below it, NEW against the old default, ch03's best, ch03_wc_bare and
ch06's bare twin, and the fused pack against the same references. The
per-round samples of every config are saved in the JSON.

What each ch07 number covers: `pack` (ch06's RGB -> 1 bit/px), `init`
(the per-run state, on the device), `rounds` (the cooperative tile
BFS) and `paint` (reached -> blue); `total` is the first event to the
last. ch03's kernel_ms covers its one launch: its device reset is off
its clock, and is timed separately (ch03_reset_ms; the part its painted
image needs, visited + seed, is ch03_reset_visited_ms). The ch07
end-to-end time (pageable H2D of the image + pipeline + D2H of the
painted image, reused buffers) of the old and the new default
(instrumented, as flood_fill() runs them) is measured after the timed
rounds.

Clocks. The GPU clock is spun up first (ch06's lesson: this GPU idles
low and does not boost for short kernels), and again for 1 s right
before each scene's timed rounds (the untimed pass before them is
seconds of host-side image compares at 64 Mpx). A background `nvidia-smi -lms 100` sampler records the SM and
memory clocks and the P-state for the whole session; each scene row
carries what they were during its timed rounds (timed_window_s, in the
trace's seconds). The micro-probes in
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
from ...ch06_gpu_nblob_runs import kernels as k6
from ...ch06_gpu_nblob_runs.recolor import (
    RunRecolor, PHASE_BLOCKS as CH06_PHASE_BLOCKS,
    PACK_ROW_BLOCKS as CH06_PACK_ROW_BLOCKS, MIN_RUN_CAPACITY,
    PIXELS_PER_RUN_SLOT, DEFAULT_GRID as CH06_GRID, _warmup as ch06_warmup)
from . import worst_cases
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
    # worst cases (worst_cases.py)
    ("zz_serp_1k", worst_cases.zz_serpentine_scene,
     "serpentine of zigzags on the x tile borders, 1024x1024: one tile "
     "per round, rounds == levels", True),
    ("zz_lattice_4k", worst_cases.zz_lattice_scene,
     "zigzag on every tile border, 4096x4096: rounds == levels with a "
     "wide front", False),
    ("maze_1k", worst_cases.maze_scene,
     "perfect maze, 1-px corridors, 1023x1023", False),
    ("thin_h32_blob", worst_cases.thin_h32_blob_scene,
     "524288x32 strip (16.8M px), 32x32 blob: pack/paint dominated", False),
    ("tiny_blob_8k", worst_cases.tiny_blob_scene,
     "5,025 px disk (r=40) in an 8000x8000 image (64M px): pack/paint "
     "dominated", False),
    ("sq_2000_corner", lambda: scenes.full_red_scene(2000, 2000),
     "4M px, corner seed: a front narrower than the grid", False),
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
    "ch03_wc_bare_48x256": ("warp", 8, True, 256, 48, False),
    "ch03_wc_bare_1x128": ("warp", 8, True, 128, 1, True),
}
# ch03's best, per scene: the fastest of all its warp-coop grids
CH03_WC_GRIDS = ["ch03_wc", "ch03_wc_120x128", "ch03_wc_240x64",
                 "ch03_wc_1x128", "ch03_wc_bare", "ch03_wc_bare_48x256",
                 "ch03_wc_bare_1x128"]

# ch07 configs: name -> TileFill kwargs, every switch pinned (so a change
# of flood_fill.DEFAULT_SWITCHES cannot move a config). 4 warps per block
# (72 x 128 for the separate-pack kernels, 48 x 128 for the fused pack's
# 242-249 registers); the grids w2/w8 and a register cap were timed in
# session 20261015T114440Z and lost to w4.
OLD_SW = {"one_sync": False, "fused_pack": False, "skip_full": False,
          "fused_paint": False, "tile_paint": False}
SW_SETS = {
    "": OLD_SW,
    "_1s": {**OLD_SW, "one_sync": True},
    "_1s_tp": {**OLD_SW, "one_sync": True, "tile_paint": True},
    "_1s_fp": {**OLD_SW, "one_sync": True, "fused_pack": True},
    "_1s_tp_sk": {**OLD_SW, "one_sync": True, "tile_paint": True,
                  "skip_full": True},
}
# the adopted default's suffix: the set equal to flood_fill's defaults
# (its own set if no listed one is: ch08's benchmark imports this module)
_DEFAULT_SW = {k: ff.DEFAULT_SWITCHES[k] for k in ff.SWITCHES}
DEFAULT_KEY = next((k for k, v in SW_SETS.items() if v == _DEFAULT_SW),
                   None)
if DEFAULT_KEY is None:
    DEFAULT_KEY = "_default"
    SW_SETS[DEFAULT_KEY] = _DEFAULT_SW


def _cfg(tpb, blocks=None, cap=None, conn=8, bare=False, sw=OLD_SW):
    kw = dict(connectivity=conn, threads_per_block=tpb, max_registers=cap,
              bare=bare, **sw)
    if blocks is not None:
        kw["blocks"] = blocks
    return kw


CH07 = {
    "ch07_w4": _cfg(128),
    "ch07_w4_bare": _cfg(128, bare=True),
    **{f"ch07_w4_bare{k}": _cfg(128, bare=True, sw=v)
       for k, v in SW_SETS.items() if k},
    # instrumented twins: the counters of each family (and timed)
    **{f"ch07_w4{k}": _cfg(128, sw=v)
       for k, v in SW_SETS.items() if k and k != "_1s"},
    "ch07_c4_w4": _cfg(128, conn=4),
    f"ch07_c4_w4{DEFAULT_KEY}": _cfg(128, conn=4, sw=SW_SETS[DEFAULT_KEY]),
    "ch07_1x32_bare": _cfg(32, blocks=1, bare=True),
    f"ch07_1x32_bare{DEFAULT_KEY}": _cfg(32, blocks=1, bare=True,
                                         sw=SW_SETS[DEFAULT_KEY]),
}
CH07_NARROW_ONLY = {"ch07_1x32_bare", f"ch07_1x32_bare{DEFAULT_KEY}"}
OLD_BARE = "ch07_w4_bare"     # the old default's bare twin
OLD_I = "ch07_w4"             # the old default, instrumented
NEW = f"ch07_w4_bare{DEFAULT_KEY}"    # the adopted default, bare
HEADLINE = f"ch07_w4{DEFAULT_KEY}"    # ... instrumented: flood_fill()
NEW_1X32 = f"ch07_1x32_bare{DEFAULT_KEY}"
FP = "ch07_w4_bare_1s_fp"     # the fused pack, opt-in
# ch06's run-table CCL, rgb contract, its own launch shapes.
CH06 = "ch06_runs_bare"       # counters off: pairs with the bare twins
CH06_I = "ch06_runs"          # instrumented: the counters
CH06_NAMES = (CH06_I, CH06)
CH06_PHASES = ("pack", "count", "scan", "emit", "merge", "flatten", "paint")
CH06_CORE = ("count", "scan", "emit", "merge", "flatten")
CONN = {**{n: spec[1] for n, spec in CH03.items()},
        **{n: kw["connectivity"] for n, kw in CH07.items()},
        CH06: 8, CH06_I: 8}
# per-round ratios (numerator / denominator: > 1 = the denominator,
# the second config, is faster). "ch03_best" is resolved per scene.
PAIRS = {
    # the switch ablation: each increment on top of the one before
    "1s_vs_old": (OLD_BARE, "ch07_w4_bare_1s"),
    "tp_on_1s": ("ch07_w4_bare_1s", "ch07_w4_bare_1s_tp"),
    "fp_on_1s_tp": ("ch07_w4_bare_1s_tp", FP),
    "fp_on_1s": ("ch07_w4_bare_1s", FP),
    "sk_on_1s_tp": ("ch07_w4_bare_1s_tp", "ch07_w4_bare_1s_tp_sk"),
    "default_vs_old": (OLD_BARE, NEW),
    "default_i_vs_old_i": (OLD_I, HEADLINE),
    "fp_vs_old": (OLD_BARE, FP),
    # against the other chapters
    "old_vs_wc_bare": ("ch03_wc_bare", OLD_BARE),
    "default_vs_wc_bare": ("ch03_wc_bare", NEW),
    "default_vs_ch03_best": ("ch03_best", NEW),
    "fp_vs_ch03_best": ("ch03_best", FP),
    "default_vs_ch06": (CH06, NEW),
    "fp_vs_ch06": (CH06, FP),
    "old_vs_ch06": (CH06, OLD_BARE),
    "ch06_bare_vs_instrumented": (CH06_I, CH06),
    "c4_default_vs_conn4": ("ch03_conn4", f"ch07_c4_w4{DEFAULT_KEY}"),
    "c4_default_vs_c4_old": ("ch07_c4_w4", f"ch07_c4_w4{DEFAULT_KEY}"),
}
NARROW_PAIRS = {
    "default_1x32_vs_default": (NEW, NEW_1X32),
    "default_1x32_vs_ch03_best": ("ch03_best", NEW_1X32),
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


def _run_count(img):
    """Maximal red runs along y (ch06's unit), counted on the host."""
    red = (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)
    return int(np.count_nonzero(red[:, 0])
               + np.count_nonzero(red[:, 1:] & ~red[:, :-1]))


def _ratio_stats(num, den):
    r = [a / b for a, b in zip(num, den)]
    return statistics.median(r), min(r), max(r)


# ------------------------------------------------------------- scenes

def _work(meta, row, name, filled, lv8):
    """Per-pixel and per-round work of one instrumented ch07 config."""
    h = meta[name]
    return {
        f"{name}_levels8_per_round": lv8 / h["rounds"],
        f"{name}_visits_per_tile_touched": (h["tile_visits"]
                                            / h["tiles_touched"]),
        f"{name}_or_atomics_per_px": h["or_atomics"] / filled,
        f"{name}_stamp_atomics_per_px": h["stamp_atomics"] / filled,
        f"{name}_closure_iters_per_visit": (h["closure_iters"]
                                            / h["tile_visits"]),
        f"{name}_rounds_us_per_round": (1000 * row[f"{name}_rounds_ms"]
                                        / h["rounds"]),
    }


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
    n_runs_host = _run_count(img)
    run_cap = max(MIN_RUN_CAPACITY, width * height // PIXELS_PER_RUN_SLOT,
                  n_runs_host)
    r6 = RunRecolor(width, height, run_capacity=run_cap)
    names = ch03_names + ch07_names + list(CH06_NAMES)
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
        if n in CH06_NAMES:
            ph, events = r6.run(d_img, contract="rgb",
                                instrumented=(n == CH06_I))
            events[-1].synchronize()
            out = {p: cuda.event_elapsed_time(events[i], events[i + 1])
                   for i, p in enumerate(ph)}
            out["total"] = cuda.event_elapsed_time(events[0], events[-1])
            return out
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
        elif n in CH06_NAMES:
            c6 = r6.counters.copy_to_host()
            if c6[k6.RUN_OVERFLOW]:
                raise RuntimeError(f"ch06 run table overflowed on {name}")
            # ch06 paints palette colours (none of them red): compare the
            # SET of pixels it changed with the set ch03's 8-conn changed
            changed = np.any(out != img, axis=2)
            ref_changed = np.any(refs[8] != img, axis=2)
            meta[n] = {"runs": int(c6[k6.N_RUNS]),
                       "run_capacity": run_cap,
                       "filled": int(np.count_nonzero(changed))}
            if n == CH06_I:       # counted only by the instrumented twin
                meta[n]["blobs"] = int(c6[k6.N_BLOBS])
                meta[n]["union_attempts"] = int(c6[k6.UNION_ATTEMPTS])
            check[n] = bool(np.array_equal(changed, ref_changed)
                            and meta[n]["runs"] == n_runs_host)
            del out, changed, ref_changed
            continue
        else:
            meta[n] = engines[n].read_counters()
            meta[n]["filled"] = engines[n].filled()
            meta[n]["tiles_touched"] = engines[n].tiles_touched()
        check[n] = bool(np.array_equal(out, refs[conn]))
        del out
    ref8 = refs[8]
    del refs
    gc.collect()

    # -- timed rounds, interleaved, order reversed on odd rounds. The
    #    untimed pass is mostly host work at 36-64 Mpx (image copies and
    #    compares: seconds of GPU idle), so the clock is ramped again here.
    _spin_up(SCENE_SPIN_SECONDS)
    samples = {n: [] for n in names}
    t_timed0 = time.time()
    for i in range(GPU_REPEATS):
        order = names if i % 2 == 0 else list(reversed(names))
        for n in order:
            samples[n].append(run_once(n))
    t_timed1 = time.time()

    # -- ch07 end to end: pageable H2D + pipeline + D2H, reused buffers,
    #    the old and the new default as flood_fill() runs them
    host_out = np.empty_like(img)
    e2e = {}
    e2e_ok = {}
    for n in (OLD_I, HEADLINE):
        e2e[n] = []
        for _ in range(E2E_REPEATS):
            cuda.synchronize()
            t0 = time.perf_counter()
            d_img.copy_to_device(img)
            engines[n].run(d_img, sx, sy)
            d_img.copy_to_host(host_out)
            e2e[n].append((time.perf_counter() - t0) * 1000)
        e2e_ok[n] = bool(np.array_equal(host_out, ref8))
    del host_out, ref8

    row = {"scene": name, "note": note, "width": width, "height": height,
           "seed": [int(sx), int(sy)], "narrow": narrow,
           "filled": meta["ch03_conn8"]["filled"],
           "tiles": engines[HEADLINE].n_tiles,
           "configs_timed": names}
    row["timed_window_s"] = [round(t_timed0 - sampler.t0, 2),
                             round(t_timed1 - sampler.t0, 2)]
    row.update(sampler.window(t_timed0, t_timed1))
    for n in names:
        tot = [s["total"] for s in samples[n]]
        row[f"{n}_ms"] = statistics.median(tot)
        row[f"{n}_ms_min"] = min(tot)
        row[f"{n}_ms_max"] = max(tot)
        row[f"{n}_ms_stdev"] = statistics.stdev(tot)
        phases = ff.PHASES if n in CH07 else (CH06_PHASES if n in CH06_NAMES
                                              else ())
        for ph in phases:
            row[f"{n}_{ph}_ms"] = statistics.median(
                s[ph] for s in samples[n])
        for k, v in meta[n].items():
            row[f"{n}_{k}"] = v
        if n in CH06_NAMES:
            row[f"{n}_fillset_matches_ch03"] = check[n]
        else:
            row[f"{n}_paint_matches_ch03"] = check[n]
    row["ch03_reset_ms"] = statistics.median(
        s["reset"] for s in samples["ch03_wc"])
    row["ch03_reset_visited_ms"] = statistics.median(
        s["reset_visited"] for s in samples["ch03_wc"])

    # ch03's best: the fastest of all its warp-coop grids (bare and
    # instrumented); the default's best: w4, or one block on the narrow
    # scenes
    b3 = min((n for n in CH03_WC_GRIDS if n in names),
             key=lambda n: row[f"{n}_ms"])
    row["ch03_best_config"] = b3
    row["ch03_best_ms"] = row[f"{b3}_ms"]
    b7 = (min((NEW, NEW_1X32), key=lambda n: row[f"{n}_ms"]) if narrow
          else NEW)
    row["default_best_config"] = b7
    row["default_best_ms"] = row[f"{b7}_ms"]
    row["default_best_vs_ch03_best"] = row["ch03_best_ms"] / row[f"{b7}_ms"]

    def resolve(n):
        return b3 if n == "ch03_best" else n

    pairs = dict(PAIRS)
    if narrow:
        pairs.update(NARROW_PAIRS)
    pairs["default_best_vs_ch03_best"] = ("ch03_best", b7)
    for key, (num, den) in pairs.items():
        med, lo, hi = _ratio_stats(
            [s["total"] for s in samples[resolve(num)]],
            [s["total"] for s in samples[resolve(den)]])
        row[f"paired_{key}_median"] = med
        row[f"paired_{key}_min"] = lo
        row[f"paired_{key}_max"] = hi
        # rounds in which the denominator lost (ratio below 1)
        row[f"paired_{key}_rounds_below_1"] = sum(
            a < b for a, b in zip(
                [s["total"] for s in samples[resolve(num)]],
                [s["total"] for s in samples[resolve(den)]]))

    # medians against medians (> 1: the ch07 config is faster)
    wc = "ch03_wc"
    for n in ch07_names:
        if CONN[n] == 8:
            row[f"{n}_speedup_vs_wc"] = row[f"{wc}_ms"] / row[f"{n}_ms"]
            row[f"{n}_speedup_vs_ch03_best"] = (row["ch03_best_ms"]
                                                / row[f"{n}_ms"])
            row[f"{n}_speedup_vs_wc_bare"] = (row["ch03_wc_bare_ms"]
                                              / row[f"{n}_ms"])
            row[f"{n}_speedup_vs_ch06"] = row[f"{CH06}_ms"] / row[f"{n}_ms"]
            row[f"{n}_speedup_vs_old"] = (row[f"{OLD_BARE}_ms"]
                                          / row[f"{n}_ms"])
        else:
            row[f"{n}_speedup_vs_conn4"] = (row["ch03_conn4_ms"]
                                            / row[f"{n}_ms"])
    for n in CH06_NAMES:
        row[f"{n}_speedup_vs_wc"] = row[f"{wc}_ms"] / row[f"{n}_ms"]
    # the connectivity work alone, without the shared ends: ch06's
    # count..flatten vs ch07's init + rounds (both start from the same
    # packed mask, made by the same pack kernel); bare twins
    row[f"{CH06}_core_ms"] = sum(row[f"{CH06}_{p}_ms"] for p in CH06_CORE)
    for n in (OLD_BARE, NEW):
        row[f"{n}_core_ms"] = row[f"{n}_init_ms"] + row[f"{n}_rounds_ms"]
        row[f"{n}_core_speedup_vs_ch06"] = (row[f"{CH06}_core_ms"]
                                            / row[f"{n}_core_ms"])

    for n in (OLD_I, HEADLINE):
        row[f"{n}_e2e_ms"] = statistics.median(e2e[n])
        row[f"{n}_e2e_ms_min"] = min(e2e[n])
        row[f"{n}_e2e_matches_ch03"] = e2e_ok[n]

    lv8 = meta["ch03_wc"]["levels"]
    row["levels8"] = lv8
    row["levels4"] = meta["ch03_conn4"]["levels"]
    for n in ch07_names:
        if n in meta and not CH07[n]["bare"] and CONN[n] == 8:
            row.update(_work(meta, row, n, row["filled"], lv8))
    row["ch03_cas_per_px"] = meta["ch03_wc"]["cas_attempts"] / row["filled"]
    row["ch03_wc_us_per_level"] = 1000 * row["ch03_wc_ms"] / lv8
    for n in (OLD_BARE, NEW):
        row[f"{n}_pack_gb_s"] = (3 * width * height / row[f"{n}_pack_ms"]
                                 / 1e6)
        row[f"{n}_paint_gb_s"] = 3 * row["filled"] / row[f"{n}_paint_ms"] / 1e6
    row["all_paint_match"] = all(check.values()) and all(e2e_ok.values())
    # every per-round sample, for ratios the row does not carry (JSON only)
    row["samples"] = {n: {k: [s[k] for s in samples[n]]
                          for k in samples[n][0]} for n in names}

    def pr(key):
        return f"x{row[f'paired_{key}_median']:5.2f}"

    print(f"{name:16s} {row['filled']:>11,d} "
          f"old {row[f'{OLD_BARE}_ms']:8.3f} new {row[f'{NEW}_ms']:8.3f} "
          f"{pr('default_vs_old')} | 1s {pr('1s_vs_old')} "
          f"tp {pr('tp_on_1s')} fp {pr('fp_on_1s_tp')} "
          f"sk {pr('sk_on_1s_tp')} | ch03 best {b3} "
          f"{row['ch03_best_ms']:8.2f} {pr('default_vs_ch03_best')} | "
          f"ch06 {row[f'{CH06}_ms']:7.3f} {pr('default_vs_ch06')} | "
          f"lv {lv8:>6,d} -> rounds {meta[HEADLINE]['rounds']:>5d} "
          f"sm {row.get('clk_sm_min', '?')}-{row.get('clk_sm_median', '?')} "
          f"mem {row.get('clk_mem_min', '?')} "
          f"{'OK' if row['all_paint_match'] else 'MISMATCH'}", flush=True)

    del engines, r3, r6, d_img, d_pristine, img, sink
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
        for n in (OLD_I, HEADLINE):
            row[f"{n}_e2e_speedup_vs_njit8"] = (row["njit8_ms"]
                                                / row[f"{n}_e2e_ms"])
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
        sw = {k: kw[k] for k in ff.SWITCHES}
        config_meta[n] = {
            "connectivity": kw["connectivity"],
            "tpb": kw["threads_per_block"],
            "blocks": eng.blocks,
            "warps": eng.blocks * kw["threads_per_block"] // 32,
            "max_registers": kw["max_registers"],
            "bare": bare,
            **sw,
            "coop_max_blocks": eng.coop_max_blocks,
            "narrow_scenes_only": n in CH07_NARROW_ONLY,
            "regs_per_thread": ff.regs_per_thread(
                kw["connectivity"], bare, kw["max_registers"], **sw),
            "local_bytes_per_thread": ff.local_bytes_per_thread(
                kw["connectivity"], bare, kw["max_registers"], **sw),
        }
        del eng
    for n, m in config_meta.items():
        print(f"  {n:16s} " + " ".join(f"{k}={v}" for k, v in m.items()))
    ch06_warmup()       # JIT ch06's kernels off the clock
    for n in CH06_NAMES:
        config_meta[n] = {"connectivity": 8, "contract": "rgb",
                          "instrumented": n == CH06_I,
                          "tpb": CH06_GRID[1],
                          "phase_blocks": dict(CH06_PHASE_BLOCKS),
                          "pack_row_blocks": CH06_PACK_ROW_BLOCKS,
                          "run_capacity": "max(8192, n/16, runs in the "
                                          "scene)"}
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
    grids = sorted({(m["blocks"], m["tpb"]) for m in config_meta.values()
                    if "blocks" in m})        # ch06 is not cooperative
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
            "tiles): its switches (one_sync, tile_paint, fused_pack, "
            "skip_full) timed as an ablation, each increment on top of "
            "the one before, against the old default (two grid.sync per "
            "round, separate pack, plain paint), vs ch03's conn4 / conn8 / "
            "warp-coop BFS kernels (instrumented and bare, several grids; "
            "ch03's best = the fastest of all its warp-coop grids) and "
            "ch06's run-table CCL (bare, and instrumented for its "
            "counters), one interleaved round-robin per scene, CUDA-event "
            "timings on every side, per-round samples saved"),
        "default_switches": dict(ff.DEFAULT_SWITCHES),
        "new_config": NEW,
        "headline_config": HEADLINE,
        "old_config_bare": OLD_BARE,
        "old_config": OLD_I,
        "fused_pack_config": FP,
        "ch06_config": CH06,
        "pairs": {k: list(v) for k, v in {**PAIRS, **NARROW_PAIRS}.items()},
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
            "last; with fused_pack the pack interval is empty and the RGB "
            "is read inside rounds, with fused_paint so is paint). "
            "ch03 *_ms = its one kernel launch; its device reset "
            "is off the clock and timed separately (ch03_reset_ms). "
            "ch06_runs *_ms = pack + count + scan + emit + merge + flatten "
            "+ paint, rgb contract (ch06_runs_bare: counters off). Every "
            "run: image restored from a device copy, 96 MB L2 flush read, "
            "GPU sleep kernel queued ahead so host launch overhead is "
            "excluded. <config>_e2e_ms = pageable H2D + pipeline + D2H of "
            "the painted image (perf_counter, reused buffers), for the old "
            "and the new default, instrumented. paired_<key>_* = median, "
            "min and max of the per-round ratio num/den (> 1: den is "
            "faster) and the rounds below 1. njit8_ms = "
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
            if k not in fieldnames and k != "samples":   # JSON only
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                             for k, v in row.items() if k != "samples"})

    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
