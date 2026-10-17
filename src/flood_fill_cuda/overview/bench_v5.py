"""v5: one session for every CPU-vs-GPU number the chain cut's v5 shows.

The chain cut v5 drops ch06 and ends on ch07 (one blob, a frontier of
32x32 tiles) and ch08 (many blobs, waves over tile components). Its
scoreboard divides CPU times by GPU times, so every ratio it shows must
come from ONE session, on one day's clocks. This script is that session.

Scenes (the v4 scenes, so v4's rows can be redrawn):

  sq_256           one small blob: a 128 x 128 square in a 256 x 256 image
                   (bench.py's row, v4's scoreboard row 1)
  sq_8000_center   one huge blob: 64 Mpx, all red, centre seed (ch03's
                   scene, v4's act 1 and ch07's headline)
  input_blobs      the 81 Mpx frame, images/input/input_blobs.png, 2,522
                   blobs (md5 checked against ch08's)
  random_1000      v4's count row: 1 Mpx noise of density 0.3, seed 0
  drone frames     the 240 masks of video/assets/drones_mask (576 x 1024),
                   converted as bench_drones.mask_image does; their SHA-1
                   are checked against v4's drones session

GPU configs (every one timed by CUDA events, see the policy below):

  ch07             ch07's default: one_sync + tile_paint, 8-conn, the
                   bare (counter-free) twin at its own grid, 72 x 128
                   (the twin ch07's README headline times)
  ch07_fused_pack  the same plus fused_pack (opt-in), 48 x 128
  ch08             ch08's default engine, TileWaves(w, h): rgb contract,
                   lattice 8, the dense-tile path, bare
  ch05_fused_L8    ch05 seed_merge, lattice 8, fused build, instrumented:
                   v4's fastest GPU cell on sq_256
  ch05_split_L8    ch05 seed_merge, lattice 8, split build: v4's 51 ms
                   on the 81 Mpx frame
  ch03_conn8       ch03's 8-conn thread-layout kernel, instrumented, its
                   default grid (v4's "8 ways", 25.8x)
  ch03_conn4       ch03's 4-conn kernel, instrumented, its default grid
                   (v4's "48 blocks")
  ch03_wc          ch03's warp-coop 8-conn kernel, instrumented, 48 x 256
  ch03_wc_bare     its bare twin at its own grid, 72 x 256 (the ~70 ms
                   ch07's README compares against)
  ch02_global      ch02's two blocks on a global queue, instrumented
                   (v4's "2 blocks")
  ch01_spill       ch01's one block, shared ring + spill (v4's "1 block")

  sq_256: ch07, ch07_fused_pack, ch05_fused_L8
  sq_8000_center: ch07, ch03_conn8, ch03_conn4, ch03_wc, ch03_wc_bare,
                  ch02_global, ch01_spill
  input_blobs, random_1000: ch08, ch05_split_L8
  drone frames: ch08

CPU cells (the @njit oracles, one core, compiled before any timing):

  njit        v4's own baseline for that scene: ch01's cpu_flood_fill
              (4-conn BFS, visited + depth, no paint) on the one-blob
              scenes, ch05's cpu_fill_canonical (labels, then a BFS from
              each blob's canonical seed) on the many-blob scenes and the
              drone frames
  njit8       one-blob scenes: the 8-conn twin (shared cpu_flood_fill_8),
              the GPU's connectivity (the same fill set on these squares)
  njit_label  many-blob scenes and drones: cpu_label_components alone
              (labels only, the cheaper like-for-like for labelling)

Timing policy (ch07's, with ch08's two-event rule):

  every GPU run   the image is restored from a pristine device copy; the
                  sides that keep state between runs (ch01, ch02, ch03,
                  ch05) reset it on the device, off the clock; a 96 MB
                  read flushes the 32 MB L2; synchronize; a ~2 ms GPU
                  sleep kernel is queued so every timed launch is in the
                  stream before the first event fires
  events          one CUDA event before the first launch and one after
                  the last (ch07's TileFill.run also records its three
                  phase boundaries; its 7 commands stay below the ~13
                  where ch08 saw the driver stall)
  rounds          9 per scene, every config once per round, the order
                  reversed on odd rounds; medians, min, max, stdev and the
                  per-round samples; paired per-round ratios between GPU
                  configs. Drone frames: 9 rounds over all 240 frames,
                  the frame order reversed on odd rounds
  ch05            in the rounds like every other side. ch08's benchmark
                  rebuilt ch05's buffers on the host per sample and ran it
                  after the rounds; here its buffers are reset on the
                  device (img restored, visited 0, depth / label / prov /
                  owner -1, the small state copied from its first build),
                  so no host work separates it from the others
  ch07, ch08      their init kernels are on their own clocks; ch01-ch03
                  and ch05 resets are off theirs (as in ch07's benchmark)
  clocks          8 s spin-up before the first scene, a 1 s GPU-bound
                  respin right before each scene's timed rounds (after
                  the untimed checks, which are host work), a background
                  nvidia-smi sampler (100 ms) whose window over each
                  scene's timed rounds is kept, plus the whole trace
  CPU             all @njit cells run last, after every GPU scene (a CPU
                  pass between scenes lets the GPU clock fall), and they
                  are interleaved too: one warm call per cell (its result
                  is checked), then 5 rounds over every cell of every
                  scene and every drone frame, the order reversed on odd
                  rounds. A cell under 0.4 s repeats within a round (at
                  most 101 calls). The host is shared and its speed
                  drifts (other processes, turbo, scheduling): each round
                  records the load average and the machine's busy
                  fraction, each cell its per-round medians. Each CPU
                  round, and each GPU scene's timed rounds, first waits
                  (up to 600 s / 120 s) until the rest of the machine is
                  at most 10% busy, and records what it saw
  end to end      context only: ch07 and ch08 with a pageable host-to-GPU
                  copy of the image, the pipeline, and the copy back
                  (perf_counter, 5 repeats, reused buffers)

Checks (untimed pass before the rounds, every config): the painted image
must equal one built from the CPU result. Blue (0, 0, 255) on the CPU
fill set for ch01, ch02, ch03 and ch07 (4-conn sides against the 4-conn
CPU BFS, 8-conn sides against the 8-conn one); palette[label % 6] on
red, from ch05's canonical CPU labels, for ch05 and ch08. ch08's label
map (off the clock) must equal the CPU labels and its blob count the
CPU's; the fill counts the kernels report must equal the CPU's. The CPU
pass checks its own fill and blob counts against the same references.

Output: results/overview/benchmark_results/v5_<stamp>.json and
v5_<stamp>.csv. Files are recorded by name only (with md5 / SHA-1).

Run:  PYTHONUNBUFFERED=1 uv run python -m flood_fill_cuda.overview.bench_v5
Smoke: ... bench_v5 --no-save --rounds 2 --drone-step 24 --scenes sq_256
Budget: about 6-10 min (compiles, 8 s spin, ~2 s per sq_8000 round).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import csv
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import time
import warnings
from datetime import datetime, timezone

import numba
import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from ..shared import scenes as shared_scenes
from ..shared.cpu_oracle import cpu_flood_fill_8
from ..chapters.ch01_gpu_1blob_1block import flood_fill as ch01_ff
from ..chapters.ch01_gpu_1blob_1block import kernels as k1
from ..chapters.ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
from ..chapters.ch02_gpu_1blob_2block import flood_fill as ch02_ff
from ..chapters.ch02_gpu_1blob_2block import kernels as k2
from ..chapters.ch03_gpu_1blob_nblock import scenes as ch03_scenes
from ..chapters.ch03_gpu_1blob_nblock import flood_fill as ch03_ff
from ..chapters.ch05_gpu_nblob_nblock import flood_fill as F5
from ..chapters.ch05_gpu_nblob_nblock import kernels as k5
from ..chapters.ch05_gpu_nblob_nblock import scenes as ch05_scenes
from ..chapters.ch05_gpu_nblob_nblock.cpu_oracle import (
    cpu_fill_canonical, cpu_label_components)
from ..chapters.ch07_gpu_1blob_tiles import flood_fill as ff7
from ..chapters.ch07_gpu_1blob_tiles.kernels import gpu_sleep_kernel
from ..chapters.ch07_gpu_1blob_tiles.benchmarks.benchmark import (
    ClockSampler, Ch03Runner, _ch03_kernel, _ch03_seed, _fill, _l2_flush,
    _clocks, _spin_up, _one, L2_FLUSH_BYTES, SLEEP_NAPS, SPIN_SECONDS,
    SCENE_SPIN_SECONDS, GPU_REPEATS)
from ..chapters.ch08_gpu_nblob_tiles import flood_fill as ff8
from ..shared import results_paths

warnings.simplefilter("ignore", NumbaPerformanceWarning)

RESULTS_DIR = results_paths.results_dir("overview", "benchmark_results")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    *[".."] * 3))
PNG_NAME = "input_blobs.png"
PNG_PATH = os.path.join(REPO, "images", "input", PNG_NAME)
PNG_MD5 = "47639eb37f973b5092ec2c4db86f45a4"     # ch08 benchmark's PNG_MD5
MASK_DIR = os.path.join(REPO, "video", "assets", "drones_mask")
V4_DRONES = "drones_20261014T023648Z.json"       # v4's drone session

E2E_REPEATS = 5
QUIET_BUSY_PCT = 10.0       # the rest of the host, at most (1.6 of 16 CPUs)
QUIET_WAIT_CPU_S = 600      # before each CPU round, at most
QUIET_WAIT_GPU_S = 120      # before each GPU scene's timed rounds, at most
CPU_ROUNDS = 5              # interleaved rounds over every CPU cell
CPU_ROUND_S = 0.4           # a shorter cell repeats within a round
CPU_MAX_CALLS = 101         # calls per cell per round, at most
TPB = 256                   # ch01-ch05's block size (their defaults)
RED = np.array([255, 0, 0], np.uint8)
BLUE = np.array([0, 0, 255], np.uint8)

# ch07 configs: TileFill kwargs, every switch pinned
CH07_DEFAULT = dict(connectivity=8, threads_per_block=128, bare=True,
                    **{k: ff7.DEFAULT_SWITCHES[k] for k in ff7.SWITCHES})
CH07_FP = {**CH07_DEFAULT, "fused_pack": True}
# ch03 configs: (layout, connectivity, bare, tpb, blocks or None = coop max)
CH03 = {
    "ch03_conn8": ("thread", 8, False, TPB, None),
    "ch03_conn4": ("thread", 4, False, TPB, None),
    "ch03_wc": ("warp", 8, False, TPB, None),
    "ch03_wc_bare": ("warp", 8, True, TPB, None),
}

# scene -> GPU configs, in round order
PLAN = {
    "sq_256": ["ch07", "ch07_fused_pack", "ch05_fused_L8"],
    "sq_8000_center": ["ch07", "ch03_conn8", "ch03_conn4", "ch03_wc",
                       "ch03_wc_bare", "ch02_global", "ch01_spill"],
    "input_blobs": ["ch08", "ch05_split_L8"],
    "random_1000": ["ch08", "ch05_split_L8"],
}
SCENE_NOTES = {
    "sq_256": "one small blob: 128 x 128 square in 256 x 256, centre seed "
              "(v4 scoreboard row 1)",
    "sq_8000_center": "one huge blob: 8000 x 8000 all red, centre seed "
                      "(v4 act 1, ch07's headline)",
    "input_blobs": "the 81 Mpx frame, 9000 x 9000 (v4 intro and act 2)",
    "random_1000": "1000 x 1000 noise, density 0.3, seed 0 (v4 count row)",
}
# v4's baseline per scene kind, and the like-for-like twin
CPU_FNS = {
    "one": (("njit", "cpu_flood_fill (ch01, 4-conn BFS: v4's baseline)"),
            ("njit8", "cpu_flood_fill_8 (shared, 8-conn BFS)")),
    "many": (("njit", "cpu_fill_canonical (ch05: labels + BFS from each "
                      "canonical seed: v4's baseline)"),
             ("njit_label", "cpu_label_components (ch05: labels only)")),
}
# GPU-vs-GPU per-round ratios worth keeping (num / den: > 1 = den faster)
PAIRS = {
    "sq_256": [("ch05_fused_L8", "ch07"), ("ch07", "ch07_fused_pack")],
    "sq_8000_center": [("ch03_conn8", "ch07"), ("ch03_wc", "ch07"),
                       ("ch03_wc_bare", "ch07"), ("ch03_conn4", "ch07"),
                       ("ch01_spill", "ch02_global"),
                       ("ch02_global", "ch03_conn4"),
                       ("ch03_conn4", "ch03_conn8")],
    "input_blobs": [("ch05_split_L8", "ch08")],
    "random_1000": [("ch05_split_L8", "ch08")],
}


# ------------------------------------------------------------- helpers

def _release():
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass


def _stats(vals):
    v = list(vals)
    return {"ms": statistics.median(v), "ms_min": min(v), "ms_max": max(v),
            "ms_stdev": statistics.stdev(v) if len(v) > 1 else 0.0,
            "n": len(v)}


def _ratio_stats(num, den):
    r = [a / b for a, b in zip(num, den)]
    return {"median": statistics.median(r), "min": min(r), "max": max(r),
            "rounds_below_1": sum(x < 1 for x in r)}


def _red_mask(img):
    return (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)


def clock_window(sampler, t0, t1):
    """The sampler's clocks over [t0, t1]. A window shorter than the
    sampler's period may hold no sample; then it is widened, 0.25 s at a
    time up to 2 s, and the padding is recorded (the second before t0 is
    the GPU-bound respin)."""
    pad = 0.0
    while True:
        w = sampler.window(t0 - pad, t1 + pad)
        if w.get("clk_samples") or pad >= 2.0:
            return {**w, "clk_window_padding_s": pad}
        pad += 0.25


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def mask_image(path):
    """A drone mask as a seedless scene, img[x, y], blobs pure red on
    white: bench_drones.mask_image, verbatim (kept here so this script
    needs no ch06 import)."""
    from PIL import Image
    m = np.array(Image.open(path).convert("L"))              # [y, x]
    img = np.full(m.shape + (3,), 255, dtype=np.uint8)
    img[m > 127] = RED
    return np.ascontiguousarray(np.transpose(img, (1, 0, 2)))


def blue_image(img, visited):
    out = img.copy()
    out[visited.astype(bool)] = BLUE
    return out


def palette_image(img, label):
    out = img.copy()
    m = label >= 0
    out[m] = k5.PALETTE_HOST[label[m] % k5.N_PALETTE]
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10, cwd=REPO).stdout.strip()
    except Exception:
        return ""


def _host_info():
    model = platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu": model, "logical_cpus": os.cpu_count()}


def _driver():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------- sides

class Ctx:
    """One scene on the device: the pristine image and the work image
    every side but ch05 paints."""

    def __init__(self, img, seed=None):
        self.img = img
        self.w, self.h = int(img.shape[0]), int(img.shape[1])
        self.seed = seed
        self.d_pristine = cuda.to_device(img)
        self.d_img = cuda.device_array_like(img)

    def restore(self):
        self.d_img.copy_to_device(self.d_pristine)


class Ch07Side:
    paints = "blue"
    chapter = "ch07"

    def __init__(self, name, ctx, cfg):
        self.name, self.ctx = name, ctx
        self.conn = cfg["connectivity"]
        self.eng = ff7.TileFill(ctx.w, ctx.h, **cfg)
        sw = {k: cfg[k] for k in ff7.SWITCHES}
        self.meta = {"chapter": "ch07", "connectivity": self.conn,
                     "bare": cfg["bare"], **sw,
                     "blocks": self.eng.blocks,
                     "tpb": cfg["threads_per_block"],
                     "regs_per_thread": ff7.regs_per_thread(
                         self.conn, cfg["bare"], None, **sw)}

    def prepare(self):
        self.ctx.restore()

    def enqueue(self):
        sx, sy = self.ctx.seed
        return self.eng.run(self.ctx.d_img, sx, sy)

    def launch(self):
        _, events = self.enqueue()
        events[-1].synchronize()
        phases, total = ff7.phase_times(events)
        return {"total": total, **phases}

    def out(self):
        return self.ctx.d_img

    def info(self):
        c = self.eng.read_counters()
        return {"rounds": c["rounds"], "filled": self.eng.filled()}


class Ch08Side:
    paints = "palette"
    chapter = "ch08"

    def __init__(self, name, ctx):
        self.name, self.ctx = name, ctx
        self.conn = 8
        self.eng = ff8.TileWaves(ctx.w, ctx.h)
        self.meta = {"chapter": "ch08", "connectivity": 8, "contract": "rgb",
                     "engine": "TileWaves defaults", "lookup":
                     ff8.DEFAULT_LOOKUP, "dense": ff8.DEFAULT_DENSE,
                     "lattice": ff8.DEFAULT_LATTICE, "instrumented": False,
                     "waves_blocks": self.eng.waves_blocks,
                     "waves_tpb": self.eng.waves_tpb}

    def prepare(self):
        self.ctx.restore()

    def enqueue(self):
        return self.eng.run(self.ctx.d_img, "rgb")

    def launch(self):
        stages = self.eng.stages(self.ctx.d_img, "rgb")
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        for _, go in stages:
            go()
        e1.record()
        e1.synchronize()
        return {"total": cuda.event_elapsed_time(e0, e1)}

    def out(self):
        return self.ctx.d_img

    def info(self):
        c = self.eng.read_counters()        # the bare twin counts these
        return {"rounds": c["rounds"], "n_blobs": c["n_blobs"]}

    def label_map(self):
        lab = cuda.device_array((self.ctx.w, self.ctx.h), np.int32)
        self.eng.emit_label_map(lab)
        out = lab.copy_to_host()
        del lab
        return out


class Ch03Side:
    paints = "blue"
    chapter = "ch03"

    def __init__(self, name, ctx, runner, launch):
        self.name, self.ctx, self.runner = name, ctx, runner
        self.l = launch
        self.conn = launch["conn"]
        self.meta = {"chapter": "ch03", "layout": launch["layout"],
                     "connectivity": self.conn, "bare": launch["bare"],
                     "blocks": launch["blocks"], "tpb": launch["tpb"],
                     "regs_per_thread": _one(
                         launch["kernel"].get_regs_per_thread())}

    def prepare(self):
        self.ctx.restore()
        self.runner.reset(*self.ctx.seed)

    def launch(self):
        L = self.l
        e0, e1 = self.runner.run(L["kernel"], L["bare"], L["blocks"],
                                 L["tpb"], self.ctx.d_img)
        e1.synchronize()
        return {"total": cuda.event_elapsed_time(e0, e1)}

    def out(self):
        return self.ctx.d_img

    def info(self):
        return self.runner.read()


class Ch01Side:
    """ch01's v2 kernel (shared ring + spill), one block of 256, on
    reusable buffers: what ch01's flood_fill(variant="spill") launches."""
    paints = "blue"
    chapter = "ch01"

    def __init__(self, name, ctx):
        self.name, self.ctx = name, ctx
        self.conn = 4
        ch01_ff._warmup("spill")
        n = ctx.w * ctx.h
        self.visited = cuda.device_array((ctx.w, ctx.h), np.int32)
        self.depth = cuda.device_array((ctx.w, ctx.h), np.int32)
        self.spill = cuda.device_array(n, np.int32)
        self.counters = cuda.device_array(k1.NUM_COUNTERS, np.int64)
        self.trace = cuda.device_array(min(n, ch01_ff.LEVEL_TRACE_CAPACITY),
                                       np.int32)
        self._flat = (self.visited.reshape(n), self.depth.reshape(n))
        self.meta = {"chapter": "ch01", "variant": "spill",
                     "connectivity": 4, "blocks": 1, "tpb": TPB}

    def prepare(self):
        self.ctx.restore()
        v, d = self._flat
        _fill[512, 256](v, np.int32(0))
        _fill[512, 256](d, np.int32(-1))
        _fill[1, 32](self.counters, np.int64(0))

    def launch(self):
        sx, sy = self.ctx.seed
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        k1.single_block_bfs_spill_kernel[1, TPB](
            self.ctx.d_img, self.visited, self.depth, int(sx), int(sy),
            self.spill, self.counters, self.trace)
        e1.record()
        e1.synchronize()
        return {"total": cuda.event_elapsed_time(e0, e1)}

    def out(self):
        return self.ctx.d_img

    def info(self):
        c = self.counters.copy_to_host()
        return {"levels": int(c[k1.LEVELS]), "filled": int(c[k1.FILLED]),
                "spilled": int(c[k1.SPILLED])}


class Ch02Side:
    """ch02's global-queue kernel, two blocks of 256, instrumented, on
    reusable buffers: what ch02's flood_fill(kernel="global") launches."""
    paints = "blue"
    chapter = "ch02"

    def __init__(self, name, ctx):
        self.name, self.ctx = name, ctx
        self.conn = 4
        ch02_ff._warmup("global", False)
        n = ctx.w * ctx.h
        self.visited = cuda.device_array((ctx.w, ctx.h), np.int32)
        self.depth = cuda.device_array((ctx.w, ctx.h), np.int32)
        self.owner = cuda.device_array((ctx.w, ctx.h), np.int8)
        self.queue = cuda.device_array(n, np.int32)
        self.q_state = cuda.device_array(1, np.int32)
        self.counters = cuda.device_array(k2.NUM_COUNTERS, np.int64)
        self.trace = cuda.device_array(
            (2, min(n, ch02_ff.LEVEL_TRACE_CAPACITY)), np.int32)
        self._flat = (self.visited.reshape(n), self.depth.reshape(n),
                      self.owner.reshape(n))
        self.meta = {"chapter": "ch02", "kernel": "global",
                     "connectivity": 4, "blocks": 2, "tpb": TPB,
                     "instrumented": True}

    def prepare(self):
        self.ctx.restore()
        v, d, o = self._flat
        sx, sy = self.ctx.seed
        _fill[512, 256](v, np.int32(0))
        _fill[512, 256](d, np.int32(-1))
        _fill[512, 256](o, np.int8(-1))
        # visited[seed] = 1, queue[0] = seed, q_state = [1], counters 0
        _ch03_seed[1, 32](v, self.queue, self.q_state, self.counters,
                          np.int64(int(sx) * self.ctx.h + int(sy)))

    def launch(self):
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        k2.dual_block_global_kernel[2, TPB](
            self.ctx.d_img, self.visited, self.depth, self.owner, self.queue,
            self.q_state, self.counters, self.trace)
        e1.record()
        e1.synchronize()
        return {"total": cuda.event_elapsed_time(e0, e1)}

    def out(self):
        return self.ctx.d_img

    def info(self):
        c = self.counters.copy_to_host()
        if c[k2.OVERFLOW]:
            raise RuntimeError("ch02 tripwire fired")
        return {"levels": int(c[k2.LEVELS]), "filled": int(c[k2.FILLED])}


class Ch05Side:
    """ch05 seed_merge with a lattice (fused or split build), the
    instrumented kernel ch05's flood_fill launches, at its cooperative
    maximum of 256-thread blocks. Its buffers come from ch05's own
    _device_buffers once and are reset on the device before every run;
    the events bracket what ch05's kernel_ms brackets (the core and, for
    the split build, its two plain kernels)."""
    paints = "palette"
    chapter = "ch05"

    def __init__(self, name, ctx, lattice, build):
        self.name, self.ctx = name, ctx
        self.conn = 8
        self.lattice, self.build = lattice, build
        self.key = F5._kernel_key("seed_merge", lattice, build)
        F5._warmup(self.key, False)
        self.kern = F5._KERNELS[(self.key, False)]
        self.blocks = F5._coop_max_blocks(self.kern, TPB)
        n = ctx.w * ctx.h
        self.bufs = F5._device_buffers(ctx.img, self.key, True, self.blocks,
                                       min(n, F5.LEVEL_TRACE_CAPACITY))
        self.small = {k: self.bufs[k].copy_to_host()
                      for k in ("q_state", "counters", "stats", "phase")}
        self.flat = {k: self.bufs[k].reshape(n)
                     for k in ("visited", "depth", "label", "owner", "prov")}
        self.args = F5._kernel_args(self.key, False, self.bufs, lattice,
                                    False)
        self.meta = {"chapter": "ch05", "variant": "seed_merge",
                     "lattice": lattice, "build": build,
                     "instrumented": True, "connectivity": 8,
                     "blocks": self.blocks, "tpb": TPB}

    def prepare(self):
        b, f = self.bufs, self.flat
        b["img"].copy_to_device(self.ctx.d_pristine)
        _fill[512, 256](f["visited"], np.int32(0))
        _fill[512, 256](f["depth"], np.int32(-1))
        _fill[512, 256](f["label"], np.int32(-1))
        _fill[512, 256](f["prov"], np.int32(-1))
        _fill[512, 256](f["owner"], np.int16(-1))
        for k, v in self.small.items():
            b[k].copy_to_device(v)

    def launch(self):
        b = self.bufs
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        self.kern[self.blocks, TPB](*self.args)
        if self.key == "seed_merge_lat_core":
            F5.lat_compress_kernel[F5._PLAIN_GRID](b["parent"])
            F5.lat_finish_kernel[F5._PLAIN_GRID](b["img"], b["parent"],
                                                 b["label"], b["prov"])
        e1.record()
        e1.synchronize()
        return {"total": cuda.event_elapsed_time(e0, e1)}

    def out(self):
        return self.bufs["img"]

    def info(self):
        c = self.bufs["counters"].copy_to_host()
        if c[k5.OVERFLOW]:
            raise RuntimeError("ch05 tripwire fired")
        return {"levels": int(c[k5.LEVELS]), "filled": int(c[k5.FILLED])}


def build_sides(names, ctx):
    sides = []
    ch03_names = [n for n in names if n in CH03]
    runner, launches = None, {}
    if ch03_names:
        for n in ch03_names:
            layout, conn, bare, tpb, blocks = CH03[n]
            kern = _ch03_kernel(layout, conn, bare)
            cap = ch03_ff.max_blocks(threads_per_block=tpb, bare=bare,
                                     connectivity=conn, probe_layout=layout)
            launches[n] = {"kernel": kern, "layout": layout, "conn": conn,
                           "bare": bare, "tpb": tpb,
                           "blocks": cap if blocks is None else blocks}
        runner = Ch03Runner(ctx.w, ctx.h,
                            max(L["blocks"] for L in launches.values()))
    for n in names:
        if n == "ch07":
            sides.append(Ch07Side(n, ctx, CH07_DEFAULT))
        elif n == "ch07_fused_pack":
            sides.append(Ch07Side(n, ctx, CH07_FP))
        elif n == "ch08":
            sides.append(Ch08Side(n, ctx))
        elif n == "ch05_fused_L8":
            sides.append(Ch05Side(n, ctx, 8, "fused"))
        elif n == "ch05_split_L8":
            sides.append(Ch05Side(n, ctx, 8, "split"))
        elif n in CH03:
            sides.append(Ch03Side(n, ctx, runner, launches[n]))
        elif n == "ch02_global":
            sides.append(Ch02Side(n, ctx))
        elif n == "ch01_spill":
            sides.append(Ch01Side(n, ctx))
        else:
            raise KeyError(n)
    return sides


def run_once(side, flush, sink):
    side.prepare()
    _l2_flush[512, 256](flush, sink)
    cuda.synchronize()
    gpu_sleep_kernel[1, 32](SLEEP_NAPS)
    return side.launch()


def e2e(side, img, expected, repeats=E2E_REPEATS):
    """Context: pageable H2D + pipeline + D2H, perf_counter."""
    out = np.empty_like(img)
    times = []
    for _ in range(repeats):
        cuda.synchronize()
        t0 = time.perf_counter()
        side.ctx.d_img.copy_to_device(img)
        side.enqueue()
        side.ctx.d_img.copy_to_host(out)
        times.append((time.perf_counter() - t0) * 1000)
    ok = bool(np.array_equal(out, expected))
    del out
    return {**_stats(times), "matches_cpu": ok}


# ------------------------------------------------------------ references

def reference(kind, img, seed, need_palette):
    """The CPU results the GPU outputs are checked against (untimed)."""
    ref = {"red_px": int(np.count_nonzero(_red_mask(img)))}
    if kind == "one":
        sx, sy = seed
        v4, _, lv4, f4 = cpu_flood_fill(img, sx, sy)
        v8, _, lv8, f8 = cpu_flood_fill_8(img, sx, sy)
        v4 = v4.astype(bool)
        v8 = v8.astype(bool)
        same = bool(np.array_equal(v4, v8))
        ref.update(filled4=int(f4), filled8=int(f8), levels4=int(lv4),
                   levels8=int(lv8), same_fill_4_8=same,
                   visited8=v8)
        ref["blue"] = {4: blue_image(img, v4)}
        ref["blue"][8] = ref["blue"][4] if same else blue_image(img, v8)
        del v4
        if need_palette:
            label, nb = cpu_label_components(img)
            ref["palette"] = palette_image(img, label)
            ref["n_blobs"] = int(nb)
            del label
    else:
        label, nb = cpu_label_components(img)
        ref.update(label=label, n_blobs=int(nb),
                   palette=palette_image(img, label))
    return ref


def check_side(side, ref, kind, img):
    """Compare one side's painted image (and counters) with the CPU."""
    out = side.out().copy_to_host()
    info = side.info()
    chk = {}
    if side.paints == "blue":
        chk["image_matches_cpu"] = bool(np.array_equal(out,
                                                       ref["blue"][side.conn]))
        want = ref["filled4"] if side.conn == 4 else ref["filled8"]
        if "filled" in info and not (side.chapter == "ch03"
                                     and side.meta.get("bare")):
            chk["filled_matches_cpu"] = info["filled"] == want
    else:
        chk["image_matches_cpu"] = bool(np.array_equal(out, ref["palette"]))
        if kind == "one":
            changed = np.any(out != img, axis=2)
            chk["fill_set_matches_cpu"] = bool(np.array_equal(
                changed, ref["visited8"]))
            del changed
        if side.chapter == "ch05":
            want = ref["filled8"] if kind == "one" else ref["red_px"]
            chk["filled_matches_cpu"] = info["filled"] == want
        if side.chapter == "ch08":
            chk["n_blobs_matches_cpu"] = info["n_blobs"] == ref["n_blobs"]
            lab = side.label_map()
            chk["labels_match_cpu"] = bool(np.array_equal(lab, ref["label"]))
            del lab
    del out
    chk["ok"] = all(chk.values())
    return info, chk


# ---------------------------------------------------------------- scenes

def bench_scene(name, kind, img, seed, sampler, flush, sink, rounds):
    names = PLAN[name]
    print(f"\n{name}: {img.shape[0]} x {img.shape[1]}, configs {names}",
          flush=True)
    t_ref = time.perf_counter()
    ref = reference(kind, img, seed,
                    need_palette=any(n.startswith("ch05") for n in names))
    print(f"  CPU reference (untimed) in "
          f"{time.perf_counter() - t_ref:.1f} s: red {ref['red_px']:,}"
          + (f", blobs {ref['n_blobs']:,}" if "n_blobs" in ref else "")
          + (f", filled4 {ref['filled4']:,} filled8 {ref['filled8']:,}"
             if kind == "one" else ""), flush=True)
    ctx = Ctx(img, seed)
    sides = build_sides(names, ctx)

    # -- untimed pass: the checks
    infos, checks = {}, {}
    for s in sides:
        run_once(s, flush, sink)
        infos[s.name], checks[s.name] = check_side(s, ref, kind, img)
        print(f"  check {s.name:16s} {'OK' if checks[s.name]['ok'] else 'MISMATCH'} "
              f"{infos[s.name]}", flush=True)

    # -- timed rounds, interleaved
    gate = wait_quiet(QUIET_WAIT_GPU_S)
    _spin_up(SCENE_SPIN_SECONDS)
    samples = {s.name: [] for s in sides}
    t0 = time.time()
    for i in range(rounds):
        order = sides if i % 2 == 0 else sides[::-1]
        for s in order:
            samples[s.name].append(run_once(s, flush, sink))
    t1 = time.time()
    clk = clock_window(sampler, t0, t1)

    # ch05 keeps its own image: its last timed run is checked too
    for s in sides:
        if s.chapter == "ch05":
            out = s.out().copy_to_host()
            checks[s.name]["last_timed_run_matches_cpu"] = bool(
                np.array_equal(out, ref["palette"]))
            checks[s.name]["ok"] = all(v for k, v in checks[s.name].items()
                                       if k != "ok")
            del out

    # -- end to end (context), the tile chapter's default
    e2e_cells = {}
    for s in sides:
        if s.name in ("ch07", "ch08"):
            exp = ref["blue"][8] if kind == "one" else ref["palette"]
            e2e_cells[s.name] = e2e(s, img, exp)

    cells = {}
    for s in sides:
        tot = [x["total"] for x in samples[s.name]]
        cell = {**_stats(tot), "side": "gpu", "chapter": s.chapter,
                "config": s.meta, "info": infos[s.name],
                "check": checks[s.name],
                "samples": {k: [x[k] for x in samples[s.name]]
                            for k in samples[s.name][0]}}
        extra = [k for k in samples[s.name][0] if k != "total"]
        if extra:
            cell["phase_ms"] = {k: statistics.median(
                x[k] for x in samples[s.name]) for k in extra}
        if s.name in e2e_cells:
            cell["e2e"] = e2e_cells[s.name]
        cells[s.name] = cell
    pairs = {}
    for num, den in PAIRS.get(name, []):
        if num in samples and den in samples:
            pairs[f"{num}/{den}"] = _ratio_stats(
                [x["total"] for x in samples[num]],
                [x["total"] for x in samples[den]])
    row = {"scene": name, "kind": kind, "note": SCENE_NOTES[name],
           "width": ctx.w, "height": ctx.h, "red_px": ref["red_px"],
           "configs": names, "cells": cells, "paired": pairs,
           "timed_window_s": [round(t0 - sampler.t0, 2),
                              round(t1 - sampler.t0, 2)],
           "host_loadavg": list(os.getloadavg()), "quiet_gate": gate,
           **clk}
    if kind == "one":
        row.update(seed=[int(seed[0]), int(seed[1])],
                   cpu_filled4=ref["filled4"], cpu_filled8=ref["filled8"],
                   cpu_levels4=ref["levels4"], cpu_levels8=ref["levels8"],
                   same_fill_4_8=ref["same_fill_4_8"])
    else:
        row["n_blobs"] = ref["n_blobs"]
    row["all_checks_ok"] = all(c["ok"] for c in checks.values()) and all(
        c["matches_cpu"] for c in e2e_cells.values())

    for s in sides:
        c = cells[s.name]
        print(f"  {s.name:16s} {c['ms']:10.3f} ms  [{c['ms_min']:.3f}, "
              f"{c['ms_max']:.3f}]", flush=True)
    for k, v in pairs.items():
        print(f"  paired {k}: x{v['median']:.2f} [{v['min']:.2f}, "
              f"{v['max']:.2f}]", flush=True)
    print(f"  sm {clk.get('clk_sm_min', '?')}-{clk.get('clk_sm_median', '?')} "
          f"MHz, mem median {clk.get('clk_mem_median', '?')}, "
          f"{'all OK' if row['all_checks_ok'] else 'CHECK FAILED'}",
          flush=True)

    del sides, ctx, ref
    _release()
    return row


def bench_drones(frames, sampler, flush, sink, rounds):
    """ch08 on every drone frame: one engine, 9 rounds over all frames."""
    w, h = frames[0]["img"].shape[:2]
    print(f"\ndrones: {len(frames)} frames of {w} x {h}", flush=True)
    ctx = Ctx(frames[0]["img"])
    side = Ch08Side("ch08", ctx)
    pristine = [cuda.to_device(f["img"]) for f in frames]
    ok_all = True
    for f, dp in zip(frames, pristine):
        label, nb = cpu_label_components(f["img"])
        f["n_blobs"] = int(nb)
        f["red_px"] = int(np.count_nonzero(_red_mask(f["img"])))
        ref = {"label": label, "n_blobs": int(nb),
               "palette": palette_image(f["img"], label)}
        ctx.d_pristine = dp
        run_once(side, flush, sink)
        f["info"], f["check"] = check_side(side, ref, "many", f["img"])
        ok_all &= f["check"]["ok"]
        del label, ref
    print(f"  checks: {'all OK' if ok_all else 'MISMATCH'}", flush=True)

    gate = wait_quiet(QUIET_WAIT_GPU_S)
    _spin_up(SCENE_SPIN_SECONDS)
    samples = [[] for _ in frames]
    idx = list(range(len(frames)))
    t0 = time.time()
    for i in range(rounds):
        for j in (idx if i % 2 == 0 else idx[::-1]):
            ctx.d_pristine = pristine[j]
            samples[j].append(run_once(side, flush, sink)["total"])
    t1 = time.time()
    clk = clock_window(sampler, t0, t1)

    e2e_ok = True
    for j, f in enumerate(frames):
        label, _ = cpu_label_components(f["img"])
        f["e2e"] = e2e(side, f["img"], palette_image(f["img"], label))
        e2e_ok &= f["e2e"]["matches_cpu"]
        f["gpu"] = _stats(samples[j])
        f["gpu_samples"] = samples[j]
        del label
    meta = side.meta
    del side, ctx, pristine
    _release()
    print(f"  ch08 per-frame medians: median "
          f"{statistics.median(f['gpu']['ms'] for f in frames):.3f} ms, "
          f"max {max(f['gpu']['ms'] for f in frames):.3f}; worst run "
          f"{max(f['gpu']['ms_max'] for f in frames):.3f}; sm "
          f"{clk.get('clk_sm_min', '?')}-{clk.get('clk_sm_median', '?')}",
          flush=True)
    return {"config": meta, "checks_ok": bool(ok_all),
            "e2e_matches_cpu": bool(e2e_ok), "quiet_gate": gate,
            "host_loadavg": list(os.getloadavg()),
            "timed_window_s": [round(t0 - sampler.t0, 2),
                               round(t1 - sampler.t0, 2)], **clk}


# ------------------------------------------------------------------ CPU

def _cpu_busy():
    """(busy, total) jiffies of the whole machine, from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:9]]
        return sum(v) - v[3] - v[4], sum(v)
    except (OSError, ValueError):
        return 0, 0


def wait_quiet(max_wait_s):
    """Wait until the rest of the machine is nearly idle: its busy share
    over 1 s (this process sleeps meanwhile) at most QUIET_BUSY_PCT, or
    max_wait_s passed. Other jobs share this host (renders, other
    sessions); under a load of 3-4 the 8-conn @njit BFS on sq_8000 read
    5.4 s in a smoke run and 1.6 s on a quiet host. Returns what it saw."""
    t0 = time.time()
    while True:
        b0, j0 = _cpu_busy()
        time.sleep(1.0)
        b1, j1 = _cpu_busy()
        busy = 100.0 * (b1 - b0) / (j1 - j0) if j1 > j0 else 0.0
        waited = time.time() - t0
        if busy <= QUIET_BUSY_PCT or waited >= max_wait_s:
            return {"quiet": busy <= QUIET_BUSY_PCT,
                    "busy_pct": round(busy, 1), "waited_s": round(waited, 1)}


def _cpu_cells(rows, scenes, frames):
    """Every CPU cell as a dict: where its result goes, its call, and the
    check of the warm call's result against the CPU reference."""
    cells = []
    for row, (name, kind, img, seed) in zip(rows, scenes):
        if kind == "one":
            sx, sy = seed
            cells += [
                dict(row=row, key="njit", fn_name=CPU_FNS["one"][0][1],
                     fn=(lambda i=img, x=sx, y=sy: cpu_flood_fill(i, x, y)),
                     check=(lambda r, w=row: {"filled": int(r[3]),
                                              "ok": int(r[3]) == w["cpu_filled4"]})),
                dict(row=row, key="njit8", fn_name=CPU_FNS["one"][1][1],
                     fn=(lambda i=img, x=sx, y=sy: cpu_flood_fill_8(i, x, y)),
                     check=(lambda r, w=row: {"filled": int(r[3]),
                                              "ok": int(r[3]) == w["cpu_filled8"]}))]
        else:
            cells += [
                dict(row=row, key="njit", fn_name=CPU_FNS["many"][0][1],
                     fn=(lambda i=img: cpu_fill_canonical(i)),
                     check=(lambda r, w=row: {"filled": int(r[4]),
                                              "ok": int(r[4]) == w["red_px"]})),
                dict(row=row, key="njit_label", fn_name=CPU_FNS["many"][1][1],
                     fn=(lambda i=img: cpu_label_components(i)),
                     check=(lambda r, w=row: {"n_blobs": int(r[1]),
                                              "ok": int(r[1]) == w["n_blobs"]}))]
    for f in frames:
        cells += [
            dict(frame=f, key="njit", calls=1,
                 fn=(lambda i=f["img"]: cpu_fill_canonical(i)),
                 check=(lambda r, g=f: {"filled": int(r[4]),
                                        "ok": int(r[4]) == g["red_px"]})),
            dict(frame=f, key="njit_label", calls=1,
                 fn=(lambda i=f["img"]: cpu_label_components(i)),
                 check=(lambda r, g=f: {"n_blobs": int(r[1]),
                                        "ok": int(r[1]) == g["n_blobs"]}))]
    return cells


def cpu_pass(rows, scenes, frames):
    """All @njit cells, interleaved like the GPU configs: one warm call
    per cell (its result is checked), then CPU_ROUNDS rounds over every
    cell, the order reversed on odd rounds, so a slow stretch of the host
    (other processes, turbo, scheduling) hits every scene alike. A cell
    shorter than CPU_ROUND_S runs several calls per round. Each round
    records the load average and the machine's busy fraction."""
    cells = _cpu_cells(rows, scenes, frames)
    for c in cells:
        t0 = time.perf_counter()
        res = c["fn"]()
        first = time.perf_counter() - t0
        c["result"] = c.pop("check")(res)
        del res
        c.setdefault("calls", max(1, min(CPU_MAX_CALLS,
                                         int(CPU_ROUND_S / max(first, 1e-6)))))
        c["times"] = []
        c["round_medians"] = []
    gc.collect()
    rounds = []
    for r in range(CPU_ROUNDS):
        gate = wait_quiet(QUIET_WAIT_CPU_S)
        b0, t0j = _cpu_busy()
        w0 = time.time()
        for c in (cells if r % 2 == 0 else cells[::-1]):
            ts = []
            for _ in range(c["calls"]):
                t0 = time.perf_counter()
                c["fn"]()
                ts.append((time.perf_counter() - t0) * 1000)
            c["times"] += ts
            c["round_medians"].append(statistics.median(ts))
        b1, t1j = _cpu_busy()
        rounds.append({"round": r, "seconds": round(time.time() - w0, 2),
                       "loadavg": list(os.getloadavg()),
                       "machine_busy_pct": (round(100 * (b1 - b0)
                                                  / (t1j - t0j), 1)
                                            if t1j > t0j else None),
                       "quiet_gate": gate})
        print(f"  CPU round {r}: {rounds[-1]['seconds']:.1f} s, load "
              f"{rounds[-1]['loadavg'][0]:.2f}, machine busy "
              f"{rounds[-1]['machine_busy_pct']}% (gate {gate})", flush=True)

    for c in cells:
        cell = {**_stats(c["times"]), "side": "cpu",
                "calls_per_round": c["calls"],
                "round_medians": c["round_medians"], **c["result"]}
        cell["check_ok"] = cell.pop("ok")
        if "row" in c:
            c["row"]["cells"][c["key"]] = {**cell, "fn": c["fn_name"]}
        else:
            c["frame"][c["key"]] = cell
    for row in rows:
        cells_r = {k: v for k, v in row["cells"].items() if v["side"] == "cpu"}
        base = cells_r["njit"]["ms"]
        twin = cells_r.get("njit8", cells_r.get("njit_label"))["ms"]
        row["cpu_over_gpu"] = {n: base / row["cells"][n]["ms"]
                               for n in row["configs"]}
        row["cpu_twin_over_gpu"] = {n: twin / row["cells"][n]["ms"]
                                    for n in row["configs"]}
        row["all_checks_ok"] = row["all_checks_ok"] and all(
            c["check_ok"] for c in cells_r.values())
        print(f"  {row['scene']:16s} " + "  ".join(
            f"{k} {c['ms']:.3f} ms (n={c['n']})" for k, c in cells_r.items())
            + "  " + ", ".join(f"{n} x{v:.1f}"
                               for n, v in row["cpu_over_gpu"].items()),
            flush=True)
    return rounds


def _summary(values):
    v = sorted(values)
    return {"median_ms": statistics.median(v),
            "p90_ms": float(np.percentile(v, 90)), "max_ms": v[-1],
            "min_ms": v[0], "n": len(v)}


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="only these scenes (and 'drones'); default all")
    ap.add_argument("--rounds", type=int, default=GPU_REPEATS)
    ap.add_argument("--drone-step", type=int, default=1,
                    help="every STEP-th drone frame (smoke runs)")
    ap.add_argument("--no-save", action="store_true",
                    help="print only, write nothing")
    ap.add_argument("--out-dir", default=None,
                    help="write here instead of the results folder")
    args = ap.parse_args()
    want = (lambda s: args.scenes is None or s in args.scenes)

    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    dev_name = dev_name.replace("\x00", "").strip()
    sm_count = int(device.MULTIPROCESSOR_COUNT)
    print(f"Device: {dev_name} ({sm_count} SMs)", flush=True)
    clocks_idle = _clocks()

    # -- inputs
    scenes = []
    if want("sq_256"):
        img, sx, sy = shared_scenes.square_scene(256, 256, 128, 128)
        scenes.append(("sq_256", "one", img, (sx, sy)))
    if want("sq_8000_center"):
        img, sx, sy = ch03_scenes.square_scene(8000, 8000, 8000, 8000)
        scenes.append(("sq_8000_center", "one", img, (sx, sy)))
    png_md5 = None
    if want("input_blobs"):
        png_md5 = _md5(PNG_PATH)
        if png_md5 != PNG_MD5:
            raise RuntimeError(f"{PNG_NAME}: md5 {png_md5}, expected {PNG_MD5}")
        img, _ = ch05_scenes.png_scene(PNG_PATH)
        scenes.append(("input_blobs", "many", img, None))
    if want("random_1000"):
        img, _ = ch05_scenes.random_blobs_scene(1000, 1000, density=0.3,
                                                rng_seed=0)
        scenes.append(("random_1000", "many", img, None))
    frames = []
    if want("drones"):
        v4 = {}
        try:
            with open(os.path.join(RESULTS_DIR, V4_DRONES)) as f:
                v4 = {r["key"]: r.get("mask_sha1")
                      for r in json.load(f)["rows"] if r["kind"] == "drone"}
        except OSError:
            pass
        names = sorted(n for n in os.listdir(MASK_DIR)
                       if n.startswith("frame_") and n.endswith(".png"))
        for n in names[::args.drone_step]:
            p = os.path.join(MASK_DIR, n)
            frames.append({"key": n[:-4], "file": n, "sha1": _sha1(p),
                           "img": mask_image(p)})
        v4_same = sum(v4.get(f["key"]) == f["sha1"] for f in frames)
        print(f"{len(frames)} drone frames; SHA-1 equal to v4's session: "
              f"{v4_same}/{len(frames)}", flush=True)

    # -- compile everything off the clock
    print("Warming up (JIT of every side, the @njit oracles)...", flush=True)
    t_w = time.perf_counter()
    warm, wx, wy = shared_scenes.square_scene(64, 64, 32, 32)
    wctx = Ctx(warm, (wx, wy))
    flush = cuda.to_device(np.zeros(L2_FLUSH_BYTES // 4, np.uint32))
    sink = cuda.device_array(1, np.uint32)
    _l2_flush[512, 256](flush, sink)
    all_names = sorted({n for s in scenes for n in PLAN[s[0]]}
                       | ({"ch08"} if frames else set()))
    for s in build_sides(all_names, wctx):
        run_once(s, flush, sink)
        s.info()
    gpu_sleep_kernel[1, 32](1)
    cuda.synchronize()
    del wctx
    _release()
    cpu_flood_fill(warm, wx, wy)
    cpu_flood_fill_8(warm, wx, wy)
    cpu_fill_canonical(warm)
    cpu_label_components(warm)
    print(f"  warm in {time.perf_counter() - t_w:.0f} s", flush=True)

    sampler = ClockSampler()
    print(f"Spinning the clock up for {SPIN_SECONDS:.0f} s...", flush=True)
    _spin_up(SPIN_SECONDS)
    clocks_start = _clocks()
    print(f"  clocks: {clocks_start}", flush=True)

    rows = []
    for name, kind, img, seed in scenes:
        rows.append(bench_scene(name, kind, img, seed, sampler, flush, sink,
                                args.rounds))
    drones = None
    if frames:
        drones = bench_drones(frames, sampler, flush, sink, args.rounds)
    clocks_gpu_end = _clocks()

    print("\n@njit CPU cells, after every GPU scene, interleaved:",
          flush=True)
    t_cpu0 = time.time()
    cpu_rounds = cpu_pass(rows, scenes, frames)
    t_cpu1 = time.time()
    sampler.stop()

    if frames:
        g = [f["gpu"]["ms"] for f in frames]
        c = [f["njit"]["ms"] for f in frames]
        cl = [f["njit_label"]["ms"] for f in frames]
        drones["summary"] = {
            "ch08": _summary(g),
            "ch08_worst_run_ms": max(f["gpu"]["ms_max"] for f in frames),
            "ch08_e2e": _summary([f["e2e"]["ms"] for f in frames]),
            "njit": _summary(c),
            "njit_label": _summary(cl),
            "njit_over_ch08_median": statistics.median(c) / statistics.median(g),
            "njit_over_ch08_per_frame": _summary(
                [a / b for a, b in zip(c, g)]),
            "blobs": {"median": statistics.median(f["n_blobs"] for f in frames),
                      "min": min(f["n_blobs"] for f in frames),
                      "max": max(f["n_blobs"] for f in frames)},
        }
        drones["cpu_checks_ok"] = all(f["njit"]["check_ok"]
                                      and f["njit_label"]["check_ok"]
                                      for f in frames)
        drones["frames"] = [
            {"key": f["key"], "file": f["file"], "mask_sha1": f["sha1"],
             "width": int(f["img"].shape[0]), "height": int(f["img"].shape[1]),
             "red_px": f["red_px"], "n_blobs": f["n_blobs"],
             "ch08": f["gpu"], "ch08_samples": f["gpu_samples"],
             "ch08_info": f["info"], "ch08_check": f["check"],
             "ch08_e2e": f["e2e"], "njit": f["njit"],
             "njit_label": f["njit_label"]} for f in frames]
        drones["frames_sha1_equal_to_v4"] = int(v4_same)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = f"v5_{stamp}"
    payload = {
        "schema": "v5_cpu_vs_gpu",
        "session": session,
        "experiment": (
            "every CPU-vs-GPU number of the chain cut v5 in one session: "
            "one small blob (sq_256), one huge blob (sq_8000_center), the "
            "81 Mpx frame (input_blobs.png), v4's count row (random_1000) "
            "and the 240 drone frames; ch07 / ch08 defaults against v4's "
            "@njit baselines, with ch05 and ch01-ch03 (v4's cells) as "
            "references; GPU configs interleaved per scene, CUDA events, "
            "CPU last"),
        "device": dev_name,
        "sm_count": sm_count,
        "driver": _driver(),
        "host": _host_info(),
        "versions": {"python": platform.python_version(),
                     "numba": numba.__version__, "numpy": np.__version__},
        "git_head": _git("rev-parse", "--short", "HEAD"),
        "script_matches_head": _git("status", "--porcelain", "--",
                                    "src/flood_fill_cuda/overview/"
                                    "bench_v5.py") == "",
        "config": {"gpu_rounds": args.rounds, "e2e_repeats": E2E_REPEATS,
                   "cpu_rounds": CPU_ROUNDS, "cpu_round_s": CPU_ROUND_S,
                   "cpu_max_calls_per_round": CPU_MAX_CALLS,
                   "drone_step": args.drone_step,
                   "spin_seconds": SPIN_SECONDS,
                   "scene_spin_seconds": SCENE_SPIN_SECONDS,
                   "sleep_naps": SLEEP_NAPS,
                   "l2_flush_bytes": L2_FLUSH_BYTES},
        "timing_note": (
            "GPU *_ms: one CUDA event before the first launch and one after "
            "the last, kernel-only (the image is already on the GPU and "
            "stays there). ch07 and ch08 time their init on the clock; "
            "ch01, ch02, ch03 and ch05 reset their buffers on the device "
            "off the clock. Every run: image restored, 96 MB L2 flush, sync, "
            f"~2 ms sleep kernel queued. {args.rounds} interleaved rounds per "
            "scene, order "
            "reversed on odd rounds. e2e = pageable H2D + pipeline + D2H "
            "(perf_counter), context only. CPU *_ms: @njit, one core, end to "
            "end of the call (allocations included, no paint), after all GPU "
            "work, in 5 interleaved rounds over every cell (median over all "
            "calls; round_medians show drift; cpu_rounds the host load). "
            "cpu_over_gpu = njit (v4's baseline) / GPU median; "
            "cpu_twin_over_gpu uses njit8 (one blob) or njit_label (many)."),
        "inputs": {"png": PNG_NAME if png_md5 else None, "png_md5": png_md5,
                   "drone_masks": len(frames),
                   "v4_drone_session": V4_DRONES if frames else None},
        "clocks_idle": clocks_idle,
        "clocks_start": clocks_start,
        "clocks_gpu_end": clocks_gpu_end,
        "cpu_window_s": [round(t_cpu0 - sampler.t0, 2),
                         round(t_cpu1 - sampler.t0, 2)],
        "cpu_rounds": cpu_rounds,
        "clock_trace": sampler.trace(),
        "scenes": rows,
        "drones": drones,
    }
    ok = all(r["all_checks_ok"] for r in rows) and (
        drones is None or (drones["checks_ok"] and drones["cpu_checks_ok"]
                           and drones["e2e_matches_cpu"]))
    payload["all_checks_ok"] = ok

    print(f"\nSession {session}  checks: {'all OK' if ok else 'FAILED'}")
    print(f"{'scene':16s} {'config':16s} {'ms':>10s} {'min':>9s} "
          f"{'max':>9s} {'njit/gpu':>9s}")
    for r in rows:
        for k, c in r["cells"].items():
            x = r["cpu_over_gpu"].get(k)
            print(f"{r['scene']:16s} {k:16s} {c['ms']:10.3f} "
                  f"{c['ms_min']:9.3f} {c['ms_max']:9.3f} "
                  + (f"{x:9.1f}" if x else f"{'':9s}"))
    if drones:
        s = drones["summary"]
        print(f"drones: ch08 median {s['ch08']['median_ms']:.3f} ms, max "
              f"{s['ch08']['max_ms']:.3f}; njit median "
              f"{s['njit']['median_ms']:.3f}, max {s['njit']['max_ms']:.3f}; "
              f"x{s['njit_over_ch08_median']:.2f}")
    if args.no_save:
        return 0 if ok else 1

    out_dir = args.out_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{session}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    csv_path = os.path.join(out_dir, f"{session}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "side", "config", "ms", "ms_min", "ms_max",
                    "n", "njit_over", "twin_over", "check", "detail"])
        for r in rows:
            for k, c in r["cells"].items():
                gpu = c["side"] == "gpu"
                w.writerow([
                    r["scene"], c["side"], k, f"{c['ms']:.4f}",
                    f"{c['ms_min']:.4f}", f"{c['ms_max']:.4f}", c["n"],
                    f"{r['cpu_over_gpu'][k]:.3f}" if gpu else "",
                    f"{r['cpu_twin_over_gpu'][k]:.3f}" if gpu else "",
                    ("OK" if (c["check"]["ok"] if gpu else c["check_ok"])
                     else "MISMATCH"),
                    json.dumps(c["info"]) if gpu else c["fn"]])
        if drones:
            s = drones["summary"]
            for key, side in (("ch08", "gpu"), ("njit", "cpu"),
                              ("njit_label", "cpu")):
                d = s[key]
                w.writerow([
                    "drones (per-frame medians)", side, key,
                    f"{d['median_ms']:.4f}", f"{d['min_ms']:.4f}",
                    f"{d['max_ms']:.4f}", d["n"],
                    f"{s['njit_over_ch08_median']:.3f}" if side == "gpu"
                    else "", "",
                    "OK" if (drones["checks_ok"] if side == "gpu"
                             else drones["cpu_checks_ok"]) else "MISMATCH",
                    f"p90 {d['p90_ms']:.4f}"])
    print(f"\nResults written: {os.path.basename(json_path)}, "
          f"{os.path.basename(csv_path)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
