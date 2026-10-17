"""v5: one session for every CPU-vs-GPU number the chain cut's v5 shows.

The chain cut v5 (acts 1-2 on the pixel BFS, act 3 on 32 x 32 tiles)
divides CPU times by GPU times on screen, so every ratio it shows must
come from ONE session, on one day's clocks. This script is that session.

The user's rules (2026-10-17), which this session implements:

  one session     every CPU-vs-GPU number comes from it, July's included
                  (act 2's two estimates are re-estimated here)
  like for like   the CPU does exactly the work of the GPU it is compared
                  with: the same connectivity (a 4-conn GPU fill against
                  the 4-conn @njit BFS, an 8-conn one against the 8-conn
                  BFS) and the same output (many blobs: labels only,
                  cpu_label_components, no depth). Every GPU cell names
                  its CPU twin, and every ratio is taken against it
  no dense path   no GPU number from ch08's dense-tile path (the run
                  labelling inside tiles, K2b / K4b): ch08's two dense
                  lists (q_state[Q_DENSE], q_state[Q_PDENSE]) are read
                  after every checked run and must be empty
  full clocks     every timed block runs at full clocks or again (below)

Scenes:

  sq_256           one small blob: a 128 x 128 square in a 256 x 256
                   image, centre seed (r02's race, r14's row 1)
  sq_8000_center   one huge blob: 64 Mpx, all red, centre seed (act 1,
                   r11, row 2)
  input_blobs      the 81 Mpx frame, images/input/input_blobs.png, 2,522
                   blobs, md5 checked against ch08's (i1, act 2, r13,
                   row 3)
  drone frames     the 240 masks of video/assets/drones_mask (576 x 1024),
                   converted as bench_drones.mask_image does; their SHA-1
                   are checked against v4's drones session (i1, r14)

GPU configs, each with its CPU twin (cpu_bfs4 = ch01's cpu_flood_fill,
the 4-conn BFS; cpu_bfs8 = the shared cpu_flood_fill_8; cpu_label =
ch05's cpu_label_components, 8-conn labels only):

  sq_256 (two blocks)
    ch07              ch07's default: one_sync + tile_paint, 8-conn, the
                      bare twin at its own grid (72 x 128)     cpu_bfs8
    ch01_ring         ch01's one block of 256, the shared-memory ring
                      (v4's ch01_ring cell): r02               cpu_bfs4
    ch05_fused_L8     reference only (v4's "GPU so far"): ch05
                      seed_merge, lattice 8, fused build       cpu_bfs8
  sq_8000_center (one block)
    ch01_spill        one block, shared ring + spill: r03b     cpu_bfs4
    ch02_global       two blocks, global queue: r04            cpu_bfs4
    ch03_conn4        ch03, 48 x 256, 4-conn: r05              cpu_bfs4
    ch03_conn8        ch03 at its 8-conn cap: r06's diagonals  cpu_bfs8
    ch07              the tile BFS: r11                        cpu_bfs8
    ch03_wc_bare      reference: the tuned pixel BFS (warp
                      layout, bare, 72 x 256)                  cpu_bfs8
  input_blobs (four blocks, every cell against cpu_label)
    ch08              ch08's default engine, TileWaves(w, h): r13
    ch05_split_L8     ch05 seed_merge, lattice 8, split build: r09
    est_ch03_conn8    r07, one launch per blob: ch03's instrumented 8-conn
                      kernel at its cap, ESTIMATED as v4 did
    est_ch04_multi8   r08, two blobs per launch with tags: ch04's
                      multisource kernel, 8-conn, ESTIMATED as v4 did
    loop_ch03_conn8   the same two loops MEASURED in full (a check on the
    loop_ch04_multi8  estimates): every blob (pair) in turn, one seed
                      kernel and one cooperative launch each
  drone frames (one block)
    ch08_nodense      ch08's default engine with its dense thresholds
                      (ccl_min, paint_ccl_min) out of reach: every tile
                      takes the piece loop in registers, the same kernels
                      as the frame's engine; the dense lists are checked
                      empty on every frame                    cpu_label
                      An untimed census of the default engine's dense
                      tiles on every frame is kept for the record

The estimates (v4's method, overview/bench.py _cell_gpu_est): one seed
per blob, each blob's canonical (lex-min) pixel, from the CPU labels; a
linspace sample of 6 calls (v4's EST_SAMPLE_HUGE, images past 20 Mpx;
for ch04 the calls are consecutive pairs of seeds); the median per-call
kernel time times the call count (2,522 and 1,261). v4 timed each
sampled call once by the host clock through the chapters' flood_fill().
Here every sampled call is timed like every GPU cell (events, below) in
every round, and the estimate of each round is a sample; v4's host clock
is recorded too, on the same calls (host_clock_est_ms). ch04 runs at
8-conn, the twin's connectivity: v4's cell used ch04's default 4-conn.

Timing policy (ch07's, with ch08's two-event rule):

  every GPU run   the image is restored from a pristine device copy; the
                  sides that keep state between runs (ch01-ch05 and the
                  loops) reset it on the device, off the clock; two reads
                  of a 96 MB buffer flush the 32 MB L2; synchronize; a
                  ~1.5 ms GPU sleep kernel is queued so every timed
                  launch is in the stream before the first event fires.
                  (v5's first session read once and slept 2 ms: on light
                  blocks the memory clock then fell to P3, 7,001 MHz, in
                  about half the samples; see _gap)
  events          one CUDA event before the first launch and one after
                  the last (ch07's TileFill.run also records its phase
                  boundaries). An estimate side times each sampled call
                  this way, its own flush and sleep before each
  blocks          a scene's configs are timed in blocks: the configs of a
                  block run once per round, the order reversed on odd
                  rounds; at least 9 rounds, more for a short block, so
                  its timed window spans at least 1.5 s (about 10 clock
                  samples). Medians, min, max, stdev and every sample;
                  paired per-round ratios inside a block, ratios of
                  medians across blocks. ch05 has its own block (it hits
                  the laptop's 40 W cap on the frame), so do the
                  estimates and the loops
  clocks          a background nvidia-smi sampler (100 ms: SM and memory
                  clocks, P-state, power, clock event reasons). Before its
                  timed rounds a block runs its own protocol untimed for
                  4 s (the ramp). A block is at full clocks when every
                  sample of its timed window is P0, the memory clock is
                  never below its maximum (8,001 MHz) and the SM median
                  is at least 2,040 MHz (the rounds settle at 2,070 under
                  the cap). Otherwise the block runs again, ramp and
                  rounds, up to 4 attempts, each with more light load
                  between its runs (0, 0.1, 0.25, 0.5 s of the gap's own
                  pattern, _filler; the drones: after each pass over the
                  frames): a heavy block's runs stay as they are, but its
                  average power falls under the laptop's 40 W cap, which
                  held ch05 on the frame in P3 in every back-to-back
                  attempt of a smoke run. Every attempt's clocks and
                  medians are kept, and the first attempt at full clocks
                  is used (else the last, flagged). ch05's own kernel on
                  the frame draws more than the cap: a probe read 48.9 to
                  51.3 ms at spacings of 0 to 1 s, the software power cap
                  active in every window, so spacing fixes its memory
                  P-state, not the SM dips inside its runs
  ch07, ch08      their init kernels are on their own clocks; ch01-ch03,
                  ch05 resets are off theirs (as in ch07's benchmark)
  CPU             all @njit cells run last, after every GPU block (a CPU
                  pass between scenes lets the GPU clock fall), and they
                  are interleaved too: one warm call per cell (its result
                  is checked), then 5 rounds over every cell of every
                  scene and every drone frame, the order reversed on odd
                  rounds. A cell under 0.4 s repeats within a round (at
                  most 101 calls). Each CPU round, and each GPU block's
                  attempt, first waits (up to 600 s / 120 s) until the
                  rest of the machine is at most 10% busy, and records
                  what it saw
  end to end      context only: ch07 (sq_256, sq_8000_center), ch08
                  (input_blobs) and ch08_nodense (every drone frame) with
                  a pageable host-to-GPU copy of the image, the pipeline,
                  and the copy back (perf_counter, 5 repeats, reused
                  buffers)

Checks (an untimed pass before each block, every config; and the last
timed run of each block, whose output the work image still holds, plus
ch05's own buffer): the painted image must equal one built from
the CPU result. Blue (0, 0, 255) on the CPU fill set for ch01, ch02,
ch03 and ch07 (4-conn sides against the 4-conn CPU BFS, 8-conn sides
against the 8-conn one); palette[label % 6] on red, from the CPU labels,
for ch05 and ch08; blue on the blob (ch03) or blue and green on the pair
(ch04) for every sampled estimate call, and on every blob for the
loops. ch08's label map (off the clock) must equal the CPU labels, its
blob count the CPU's and its dense lists must be empty; the fill counts
the kernels report must equal the CPU's. The CPU pass checks its own
fill and blob counts against the same references.

Output: results/overview/benchmark_results/v5_<stamp>.json and
v5_<stamp>.csv. Files are recorded by name only (with md5 / SHA-1). The
JSON's "story" lists every CPU-vs-GPU number of the v5 script with its
cells, times and ratio.

Run:  PYTHONUNBUFFERED=1 uv run python -m flood_fill_cuda.overview.bench_v5
Smoke: ... bench_v5 --no-save --drone-step 24 --scenes sq_256 drones
Budget: about 6-10 min (compiles, 4 s ramps per block, ~2 s per
sq_8000 round, the loops ~4 s per round, about 1.5 min of CPU cells).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import threading
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
from ..chapters.ch04_gpu_2blob_nblock import flood_fill as ff4
from ..chapters.ch04_gpu_2blob_nblock import kernels as k4
from ..chapters.ch04_gpu_2blob_nblock import scenes as ch04_scenes
from ..chapters.ch05_gpu_nblob_nblock import flood_fill as F5
from ..chapters.ch05_gpu_nblob_nblock import kernels as k5
from ..chapters.ch05_gpu_nblob_nblock import scenes as ch05_scenes
from ..chapters.ch05_gpu_nblob_nblock.cpu_oracle import (
    cpu_fill_canonical, cpu_label_components)
from ..chapters.ch07_gpu_1blob_tiles import flood_fill as ff7
from ..chapters.ch07_gpu_1blob_tiles.kernels import gpu_sleep_kernel
from ..chapters.ch07_gpu_1blob_tiles.benchmarks.benchmark import (
    Ch03Runner, _ch03_kernel, _ch03_seed, _fill, _l2_flush, _clocks, _one,
    L2_FLUSH_BYTES, GPU_REPEATS)
from ..chapters.ch08_gpu_nblob_tiles import flood_fill as ff8
from ..chapters.ch08_gpu_nblob_tiles.kernels import Q_DENSE, Q_PDENSE
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
V4_OVERVIEW = "overview_20260724T234423Z.json"   # v4's estimates (July)

E2E_REPEATS = 5
WARM_SECONDS = 4.0          # untimed rounds of the block before its timed ones
MIN_TIMED_S = 1.5           # a block's timed window, at least (wall)
MAX_ROUNDS = 300            # rounds per block, at most
MAX_ATTEMPTS = 4            # timed attempts per block, until full clocks
SPACING_S = (0.0, 0.1, 0.25, 0.5)   # light filler between runs, per attempt
SM_MEDIAN_MIN = 2040        # MHz: the block's SM median, at least
SMI_PERIOD_MS = 100
QUIET_BUSY_PCT = 10.0       # the rest of the host, at most (1.6 of 16 CPUs)
QUIET_WAIT_CPU_S = 600      # before each CPU round, at most
QUIET_WAIT_GPU_S = 120      # before each GPU block attempt, at most
CPU_ROUNDS = 5              # interleaved rounds over every CPU cell
CPU_ROUND_S = 0.4           # a shorter cell repeats within a round
CPU_MAX_CALLS = 101         # calls per cell per round, at most
EST_SAMPLE = 6              # v4's EST_SAMPLE_HUGE (images past 20 Mpx)
HOST_CLOCK_REPEATS = 3      # v4's host clock, per sampled call
LOOP_ROUNDS = 3             # the measured loops take seconds per round
NO_DENSE = 1 << 30          # a ccl_min no tile reaches
FLUSH_READS = 2             # 96 MB L2-flush reads before each timed run
GAP_NAPS = 15               # then ~1.5 ms of GPU sleep (see _gap)
TPB = 256                   # ch01-ch05's block size (their defaults)
RED = np.array([255, 0, 0], np.uint8)
BLUE = np.array([0, 0, 255], np.uint8)

# NVML's clock event reasons (nvidia-smi clocks_event_reasons.active)
EVENT_REASONS = {0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap",
                 0x8: "hw_slowdown", 0x10: "sync_boost",
                 0x20: "sw_thermal", 0x40: "hw_thermal",
                 0x80: "hw_power_brake", 0x100: "display_clock"}

# ch07 configs: TileFill kwargs, every switch pinned
CH07_DEFAULT = dict(connectivity=8, threads_per_block=128, bare=True,
                    **{k: ff7.DEFAULT_SWITCHES[k] for k in ff7.SWITCHES})
# ch08 engines: TileWaves kwargs
CH08_DEFAULT = {}
CH08_NODENSE = dict(ccl_min=NO_DENSE, paint_ccl_min=NO_DENSE)
# ch03 configs: (layout, connectivity, bare, tpb, blocks or None = coop max)
CH03 = {
    "ch03_conn8": ("thread", 8, False, TPB, None),
    "ch03_conn4": ("thread", 4, False, TPB, None),
    "ch03_wc_bare": ("warp", 8, True, TPB, None),
}

# scene -> blocks: (block, configs in round order, minimum rounds)
BLOCKS = {
    "sq_256": [("interleaved", ["ch07", "ch01_ring"], GPU_REPEATS),
               ("ch05", ["ch05_fused_L8"], GPU_REPEATS)],
    "sq_8000_center": [("interleaved",
                        ["ch07", "ch03_conn8", "ch03_conn4", "ch03_wc_bare",
                         "ch02_global", "ch01_spill"], GPU_REPEATS)],
    "input_blobs": [("waves", ["ch08"], GPU_REPEATS),
                    ("ch05", ["ch05_split_L8"], GPU_REPEATS),
                    ("estimates", ["est_ch03_conn8", "est_ch04_multi8"],
                     GPU_REPEATS),
                    ("loops", ["loop_ch03_conn8", "loop_ch04_multi8"],
                     LOOP_ROUNDS)],
}
SCENE_NOTES = {
    "sq_256": "one small blob: 128 x 128 square in 256 x 256, centre seed",
    "sq_8000_center": "one huge blob: 8000 x 8000 all red, centre seed",
    "input_blobs": "the 81 Mpx frame, 9000 x 9000, 2,522 blobs",
}
# the GPU config's CPU twin: the same connectivity, the same output
TWIN = {
    "ch01_ring": "cpu_bfs4", "ch01_spill": "cpu_bfs4",
    "ch02_global": "cpu_bfs4", "ch03_conn4": "cpu_bfs4",
    "ch03_conn8": "cpu_bfs8", "ch03_wc_bare": "cpu_bfs8",
    "ch07": "cpu_bfs8", "ch05_fused_L8": "cpu_bfs8",
    "ch08": "cpu_label", "ch05_split_L8": "cpu_label",
    "est_ch03_conn8": "cpu_label", "est_ch04_multi8": "cpu_label",
    "loop_ch03_conn8": "cpu_label", "loop_ch04_multi8": "cpu_label",
    "ch08_nodense": "cpu_label",
}
REFERENCE_ONLY = {"ch05_fused_L8", "ch03_wc_bare", "loop_ch03_conn8",
                  "loop_ch04_multi8"}
CPU_FNS = {
    "cpu_bfs4": "cpu_flood_fill (ch01: 4-conn BFS, visited + depth)",
    "cpu_bfs8": "cpu_flood_fill_8 (shared: 8-conn BFS, visited + depth)",
    "cpu_label": "cpu_label_components (ch05: 8-conn labels only)",
    "cpu_fill_canonical": ("cpu_fill_canonical (ch05: labels + a BFS from "
                           "each canonical seed, with depth): v4's "
                           "baseline, reference only"),
}
CPU_CELLS = {"one": ("cpu_bfs4", "cpu_bfs8"),
             "frame": ("cpu_label", "cpu_fill_canonical")}
E2E = {"sq_256": "ch07", "sq_8000_center": "ch07", "input_blobs": "ch08"}
# GPU-vs-GPU ratios worth keeping (num / den: > 1 = den faster); paired
# per round inside a block, a ratio of medians across blocks
PAIRS = {
    "sq_256": [("ch01_ring", "ch07"), ("ch05_fused_L8", "ch07")],
    "sq_8000_center": [("ch03_conn8", "ch07"), ("ch03_wc_bare", "ch07"),
                       ("ch03_conn4", "ch07"), ("ch01_spill", "ch02_global"),
                       ("ch02_global", "ch03_conn4"),
                       ("ch03_conn4", "ch03_conn8")],
    "input_blobs": [("ch05_split_L8", "ch08"), ("est_ch03_conn8", "ch08"),
                    ("est_ch04_multi8", "ch08"),
                    ("est_ch03_conn8", "est_ch04_multi8"),
                    ("est_ch04_multi8", "ch05_split_L8"),
                    ("loop_ch03_conn8", "est_ch03_conn8"),
                    ("loop_ch04_multi8", "est_ch04_multi8")],
}
# every CPU-vs-GPU number of the v5 script: (key, beat, scene, GPU cell,
# what is said). The CPU side is the GPU cell's twin.
STORY = [
    ("small_pixel_bfs", "r02", "sq_256", "ch01_ring",
     "one block, shared-memory ring, against the 4-conn CPU BFS: the CPU "
     "still wins the small blob"),
    ("big_1block", "r03b", "sq_8000_center", "ch01_spill",
     "one block (spill) against the 4-conn CPU BFS"),
    ("big_2blocks", "r04", "sq_8000_center", "ch02_global",
     "two blocks, global queue, against the 4-conn CPU BFS"),
    ("big_48blocks", "r05", "sq_8000_center", "ch03_conn4",
     "48 blocks against the 4-conn CPU BFS"),
    ("big_8ways", "r06", "sq_8000_center", "ch03_conn8",
     "diagonals (8-conn) against the 8-conn CPU BFS: row 2"),
    ("frame_one_launch_per_blob", "r07", "input_blobs", "est_ch03_conn8",
     "one launch per blob (estimate) against the CPU labels"),
    ("frame_two_per_launch", "r08", "input_blobs", "est_ch04_multi8",
     "two blobs per launch with tags (estimate) against the CPU labels"),
    ("frame_seeds", "r09", "input_blobs", "ch05_split_L8",
     "the GPU finds its own seeds against the CPU labels: row 3"),
    ("big_tiles", "r11", "sq_8000_center", "ch07",
     "the tile BFS against the 8-conn CPU BFS: row 2"),
    ("frame_waves", "r13", "input_blobs", "ch08",
     "the tile waves against the CPU labels: row 3"),
    ("small_tiles", "r14", "sq_256", "ch07",
     "the tile BFS on the small blob against the 8-conn CPU BFS: row 1"),
]


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
            "rounds_below_1": sum(x < 1 for x in r), "n": len(r)}


def _red_mask(img):
    return (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)


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
    needs no ch06 import of its own)."""
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


def _smi(query):
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _reasons(mask):
    return [n for b, n in EVENT_REASONS.items() if mask & b] + (
        [f"other_0x{mask & ~sum(EVENT_REASONS):x}"]
        if mask & ~sum(EVENT_REASONS) else [])


# --------------------------------------------------------------- clocks

class Sampler:
    """`nvidia-smi -lms` in the background: (t, SM MHz, memory MHz,
    P-state, power W, clock event reasons) every SMI_PERIOD_MS. t is
    nvidia-smi's own timestamp, on the host's wall clock like
    time.time(); a line reached the pipe 25-45 ms after its timestamp in
    a probe, so a block's window is read once a later sample is in."""

    QUERY = ("timestamp,clocks.sm,clocks.mem,pstate,power.draw,"
             "clocks_event_reasons.active")

    def __init__(self, period_ms=SMI_PERIOD_MS):
        self.samples = []
        self.t0 = time.time()
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}",
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
                ts, sm, mem, ps, pw, rs = [x.strip() for x in line.split(",")]
                t = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
                try:
                    pw = float(pw)
                except ValueError:
                    pw = None
                try:
                    rs = int(rs, 16)
                except ValueError:
                    rs = None
                self.samples.append((t, int(float(sm)), int(float(mem)), ps,
                                     pw, rs))
            except ValueError:
                continue

    def wait_past(self, t, timeout=5.0):
        """Until a sample later than t is in (or timeout)."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.samples and self.samples[-1][0] > t:
                return True
            time.sleep(0.05)
        return False

    def window(self, t0, t1):
        """Clock summary over [t0, t1] (time.time() values)."""
        s = [x for x in list(self.samples) if t0 <= x[0] <= t1]
        if not s:
            return {"clk_samples": 0}
        sm = [x[1] for x in s]
        mem = [x[2] for x in s]
        pw = [x[4] for x in s if x[4] is not None]
        rs = 0
        for x in s:
            rs |= x[5] or 0
        out = {"clk_samples": len(s),
               "clk_sm_min": min(sm), "clk_sm_median": statistics.median(sm),
               "clk_mem_min": min(mem),
               "clk_mem_median": statistics.median(mem),
               "clk_pstates": sorted({x[3] for x in s}),
               "clk_p0_share": sum(x[3] == "P0" for x in s) / len(s),
               "clk_event_reasons": _reasons(rs)}
        if pw:
            out.update(power_w_median=statistics.median(pw),
                       power_w_max=max(pw))
        return out

    def trace(self):
        """[[seconds since start, sm MHz, mem MHz, P-state, W, reasons], ...]"""
        return [[round(x[0] - self.t0, 2), x[1], x[2], x[3], x[4],
                 None if x[5] is None else f"0x{x[5]:x}"]
                for x in list(self.samples)]

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.thread.join(timeout=5)


def clock_window(sampler, t0, t1):
    """The sampler's clocks over [t0, t1]. A window shorter than the
    sampler's period may hold no sample; then it is widened, 0.25 s at a
    time up to 2 s, and the padding is recorded."""
    sampler.wait_past(t1)
    pad = 0.0
    while True:
        w = sampler.window(t0 - pad, t1 + pad)
        if w.get("clk_samples") or pad >= 2.0:
            return {**w, "clk_window_padding_s": pad}
        pad += 0.25


MEM_FULL_MHZ = None         # the card's maximum memory clock, read in main()


def full_clocks(w):
    """Every sample P0, the memory clock never below its maximum, the SM
    median at least SM_MEDIAN_MIN."""
    return bool(w.get("clk_samples") and w["clk_pstates"] == ["P0"]
                and w["clk_mem_min"] >= MEM_FULL_MHZ
                and w["clk_sm_median"] >= SM_MEDIAN_MIN
                and w.get("clk_window_padding_s", 0.0) == 0.0)


def _filler(flush, sink, seconds):
    """Light load between the runs of a retried block: the gap's own
    pattern (two flush reads, a ~1.5 ms sleep, a synchronize) for
    `seconds`. It held P0 at ~24 W in the probe (see _gap). A heavy block
    (ch05 on the 81 Mpx frame: 43-45 W, the software power cap, the
    memory clock in P3 in every back-to-back attempt) keeps its runs as
    they are, but its average power falls under the laptop's cap."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        _gap(flush, sink)
        cuda.synchronize()


def timed_attempts(label, warm_fn, round_fn, rounds_min, sampler):
    """A block's timed rounds, run again until its clocks are full (at
    most MAX_ATTEMPTS); attempt a puts SPACING_S[a] seconds of light load
    (_filler) between the block's runs, its ramp included.
    warm_fn(seconds, spacing) -> (rounds done, seconds); round_fn(i,
    spacing) -> that round's samples. The round count is fixed by the
    first attempt's ramp: at least rounds_min, enough to span
    MIN_TIMED_S, at most MAX_ROUNDS. Returns (the used attempt's rounds,
    every attempt's record, the used attempt's index)."""
    attempts, rounds, data = [], None, None
    for a in range(MAX_ATTEMPTS):
        spacing = SPACING_S[min(a, len(SPACING_S) - 1)]
        gate = wait_quiet(QUIET_WAIT_GPU_S)
        n_w, s_w = warm_fn(WARM_SECONDS, spacing)
        if rounds is None:
            per_round = s_w / max(n_w, 1)
            rounds = max(rounds_min, min(MAX_ROUNDS,
                                         math.ceil(MIN_TIMED_S / per_round)))
        t0 = time.time()
        data = [round_fn(i, spacing) for i in range(rounds)]
        t1 = time.time()
        w = clock_window(sampler, t0, t1)
        ok = full_clocks(w)
        attempts.append({"attempt": a, "spacing_s": spacing,
                         "rounds": rounds, "quiet_gate": gate,
                         "timed_window_s": [round(t0 - sampler.t0, 2),
                                            round(t1 - sampler.t0, 2)],
                         **w, "full_clocks": ok,
                         "medians_ms": _attempt_medians(data)})
        print(f"  {label} attempt {a} (spacing {spacing} s): {rounds} "
              f"rounds in {t1 - t0:.1f} s, "
              f"sm {w.get('clk_sm_min', '?')}-{w.get('clk_sm_median', '?')}, "
              f"mem {w.get('clk_mem_min', '?')}-"
              f"{w.get('clk_mem_median', '?')}, {w.get('clk_pstates')}, "
              f"{w.get('power_w_median', '?')} W, "
              f"{w.get('clk_event_reasons')}: "
              f"{'full clocks' if ok else 'NOT full clocks'}", flush=True)
        if ok:
            break
    return data, attempts, len(attempts) - 1


def _attempt_medians(data):
    """Per config, the median total of one attempt (every attempt keeps
    its own, so a discarded attempt's numbers stay on the record)."""
    if not data:
        return {}
    if isinstance(data[0], dict):
        return {k: statistics.median(r[k]["total"] for r in data)
                for k in data[0]}
    return {"per_frame_median_of_medians": statistics.median(
        statistics.median(r[j] for r in data) for j in range(len(data[0])))}


# --------------------------------------------------------------- sides

class Ctx:
    """One scene on the device: the pristine image and the work image
    every side but ch05 paints. `cache` holds buffers sides share."""

    def __init__(self, img, seed=None):
        self.img = img
        self.w, self.h = int(img.shape[0]), int(img.shape[1])
        self.seed = seed
        self.d_pristine = cuda.to_device(img)
        self.d_img = cuda.device_array_like(img)
        self.cache = {}

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
    """ch08's TileWaves, bare, rgb contract. Its dense lists are read
    off the clock: the tiles K2 left to K2b and K4 to K4b."""
    paints = "palette"
    chapter = "ch08"

    def __init__(self, name, ctx, kw):
        self.name, self.ctx = name, ctx
        self.conn = 8
        self.eng = ff8.TileWaves(ctx.w, ctx.h, **kw)
        e = self.eng
        self.meta = {"chapter": "ch08", "connectivity": 8, "contract": "rgb",
                     "engine": ("TileWaves defaults" if not kw else
                                f"TileWaves defaults but {sorted(kw)}"),
                     "dense": e.dense, "ccl_min": e.ccl_min,
                     "paint_ccl_min": e.paint_ccl_min, "lookup": e.lookup,
                     "lattice": e.lattice, "instrumented": False,
                     "waves_blocks": e.waves_blocks,
                     "waves_tpb": e.waves_tpb}

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
        q = self.eng.read_q_state()
        return {"rounds": c["rounds"], "n_blobs": c["n_blobs"],
                "dense_label_tiles": int(q[Q_DENSE]),
                "dense_paint_tiles": int(q[Q_PDENSE])}

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
    """ch01's one block of 256 on reusable buffers: the v1 kernel (pure
    shared-memory ring, variant "ring") or the v2 kernel (shared ring +
    global spill, variant "spill"), what ch01's flood_fill launches."""
    paints = "blue"
    chapter = "ch01"

    def __init__(self, name, ctx, variant):
        self.name, self.ctx, self.variant = name, ctx, variant
        self.conn = 4
        ch01_ff._warmup(variant)
        n = ctx.w * ctx.h
        self.visited = cuda.device_array((ctx.w, ctx.h), np.int32)
        self.depth = cuda.device_array((ctx.w, ctx.h), np.int32)
        self.spill = (cuda.device_array(n, np.int32) if variant == "spill"
                      else None)
        self.counters = cuda.device_array(k1.NUM_COUNTERS, np.int64)
        self.trace = cuda.device_array(min(n, ch01_ff.LEVEL_TRACE_CAPACITY),
                                       np.int32)
        self._flat = (self.visited.reshape(n), self.depth.reshape(n))
        self.meta = {"chapter": "ch01", "variant": variant,
                     "connectivity": 4, "blocks": 1, "tpb": TPB}
        if variant == "ring":
            self.meta["ring_capacity"] = k1.RING_CAPACITY

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
        if self.variant == "ring":
            k1.single_block_bfs_kernel[1, TPB](
                self.ctx.d_img, self.visited, self.depth, int(sx), int(sy),
                self.counters, self.trace)
        else:
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
        if c[k1.OVERFLOW]:
            raise RuntimeError(f"ch01 {self.variant}: ring overflow")
        out = {"levels": int(c[k1.LEVELS]), "filled": int(c[k1.FILLED]),
               "peak_occupancy": int(c[k1.PEAK_OCC])}
        if self.variant == "spill":
            out["spilled"] = int(c[k1.SPILLED])
        return out


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


@cuda.jit
def _ch04_seed(visited_flat, queue, q_state, counters, lin_a, lin_b, pk_a,
               pk_b):
    """ch04 multisource's host init, on the device: both seeds visited,
    queued with their tags (lin format), q_state = [2], counters 0."""
    if cuda.grid(1) == 0:
        visited_flat[lin_a] = 1
        visited_flat[lin_b] = 1
        queue[0] = pk_a
        queue[1] = pk_b
        q_state[0] = 2
        for c in range(counters.shape[0]):
            counters[c] = 0


class Ch04Kernel:
    """ch04's multisource kernel (lin entries, 8-conn, instrumented) at
    its cooperative maximum of 256-thread blocks, on the scene's shared
    ch03 buffers (visited, depth, owner, queue: the same shapes and
    types) plus its own counters, stats and trace."""

    def __init__(self, ctx, runner):
        ff4._warmup("lin", False, 8, 1)
        self.kern = ff4._KERNELS[("lin", False, 8, 1)]
        self.blocks = ff4.max_blocks(threads_per_block=TPB, bare=False,
                                     connectivity=8)
        n = ctx.w * ctx.h
        self.ctx, self.r = ctx, runner
        self.q_state = cuda.device_array(1, np.int32)
        self.counters = cuda.device_array(k4.NUM_COUNTERS, np.int64)
        st = np.zeros((self.blocks, 2), np.int64)
        st[:, k4.BS_SMID] = -1
        self.stats_host = st
        self.stats = cuda.to_device(st)
        self.trace = cuda.device_array(min(n, ff4.LEVEL_TRACE_CAPACITY),
                                       np.int32)

    def seed(self, a, b):
        h = self.ctx.h
        la, lb = a[0] * h + a[1], b[0] * h + b[1]
        _ch04_seed[1, 32](self.r.visited.reshape(self.ctx.w * h),
                          self.r.queue, self.q_state, self.counters,
                          np.int64(la), np.int64(lb),
                          np.int32((la << 1) | 0), np.int32((lb << 1) | 1))

    def launch(self):
        r = self.r
        self.kern[self.blocks, TPB](
            self.ctx.d_img, r.visited, r.depth, r.owner, r.queue,
            self.q_state, self.counters, self.stats, self.trace, 2)

    def read(self):
        c = self.counters.copy_to_host()
        if c[k4.OVERFLOW]:
            raise RuntimeError("ch04 tripwire fired")
        return {"levels": int(c[k4.LEVELS]), "filled": int(c[k4.FILLED])}


def _scene_runner(ctx, blocks):
    """One Ch03Runner per scene, shared by the estimate and loop sides."""
    if "runner" not in ctx.cache:
        ctx.cache["runner"] = Ch03Runner(ctx.w, ctx.h, blocks)
    return ctx.cache["runner"]


class _BlobLoop:
    """The common part of the estimate and loop sides: the calls (one
    seed for ch03, a pair of consecutive seeds for ch04, as v4's
    _cell_gpu_est), the kernels and the resets."""
    chapter = "est"

    def __init__(self, name, ctx, kind):
        self.name, self.ctx, self.kind = name, ctx, kind
        self.conn = 8
        seeds = ctx.cache["seeds"]
        n = len(seeds)
        if kind == "ch03":
            self.k3 = _ch03_kernel("thread", 8, False)
            self.blocks = ch03_ff.max_blocks(threads_per_block=TPB,
                                             bare=False, connectivity=8,
                                             probe_layout="thread")
            self.r = _scene_runner(ctx, self.blocks)
            self.calls = [(s,) for s in seeds]
            self.n_calls = n
            self.meta = {"chapter": "ch03", "layout": "thread",
                         "connectivity": 8, "bare": False,
                         "blocks": self.blocks, "tpb": TPB,
                         "per_call": "one blob, its canonical seed"}
        else:
            if n % 2:
                raise ValueError("ch04 pairs need an even blob count")
            r = _scene_runner(ctx, ch03_ff.max_blocks(
                threads_per_block=TPB, bare=False, connectivity=8,
                probe_layout="thread"))
            self.r = r
            self.k4 = Ch04Kernel(ctx, r)
            self.blocks = self.k4.blocks
            self.calls = [(seeds[i], seeds[i + 1])
                          for i in range(0, n - 1, 2)]
            self.n_calls = (n + 1) // 2
            self.meta = {"chapter": "ch04", "mode": "multisource",
                         "entry_format": "lin", "connectivity": 8,
                         "bare": False, "blocks": self.blocks, "tpb": TPB,
                         "per_call": ("two blobs, consecutive canonical "
                                      "seeds, tags 0 and 1")}
        self.meta["calls"] = self.n_calls

    def _reset_all(self):
        """The whole image's state: visited 0, depth and owner -1, stats."""
        r = self.r
        v, d, o, s = r._flat
        _fill[512, 256](v, np.int32(0))
        _fill[512, 256](d, np.int32(-1))
        _fill[512, 256](o, np.int16(-1))
        _fill[4, 256](s, np.int64(0))
        if self.kind == "ch04":
            self.k4.stats.copy_to_device(self.k4.stats_host)

    def _seed(self, call):
        if self.kind == "ch03":
            (sx, sy), = call
            _ch03_seed[1, 32](self.r._flat[0], self.r.queue, self.r.q_state,
                              self.r.counters,
                              np.int64(int(sx) * self.ctx.h + int(sy)))
        else:
            self.k4.seed(*call)

    def _launch(self):
        if self.kind == "ch03":
            r = self.r
            self.k3[self.blocks, TPB](
                self.ctx.d_img, r.visited, r.depth, r.owner, r.queue,
                r.q_state, r.counters, r.stats, r.trace)
        else:
            self.k4.launch()

    def _filled(self):
        if self.kind == "ch03":
            return self.r.read()["filled"]
        return self.k4.read()["filled"]

    def _colors(self):
        """The fill colour of each blob of a call."""
        if self.kind == "ch03":
            return (BLUE,)
        return tuple(k4.PALETTE_HOST[:2])


class EstSide(_BlobLoop):
    """v4's estimate: the sampled calls, each timed alone (restore, full
    reset, its seed, flush, sleep, two events around its launch); a
    round's sample is the median call times the call count."""
    paints = "est"

    def __init__(self, name, ctx, kind):
        super().__init__(name, ctx, kind)
        idx = np.unique(np.linspace(0, len(self.calls) - 1,
                                    min(EST_SAMPLE, len(self.calls)))
                        .astype(int))
        self.sample_idx = [int(i) for i in idx]
        self.sample = [self.calls[i] for i in idx]
        self.meta.update(estimate="median sampled call x calls (v4)",
                         sample=len(self.sample),
                         sample_call_index=self.sample_idx)

    def _prepare_call(self, call):
        self.ctx.restore()
        self._reset_all()
        self._seed(call)

    def run_timed(self, flush, sink):
        per = []
        for call in self.sample:
            self._prepare_call(call)
            _gap(flush, sink)
            e0 = cuda.event(timing=True)
            e1 = cuda.event(timing=True)
            e0.record()
            self._launch()
            e1.record()
            e1.synchronize()
            per.append(cuda.event_elapsed_time(e0, e1))
        return {"total": statistics.median(per) * self.n_calls,
                "per_call": per}

    def host_clock(self, flush, sink):
        """v4's clock on the same calls: perf_counter around the launch
        and a synchronize (what the chapters' flood_fill() report),
        median of HOST_CLOCK_REPEATS per call."""
        per = []
        for call in self.sample:
            ts = []
            for _ in range(HOST_CLOCK_REPEATS):
                self._prepare_call(call)
                for _ in range(FLUSH_READS):
                    _l2_flush[512, 256](flush, sink)
                cuda.synchronize()
                t0 = time.perf_counter()
                self._launch()
                cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            per.append(statistics.median(ts))
        return per

    def check(self, ref, img):
        """Every sampled call once, untimed: its painted image must be the
        pristine one with its blob(s) recoloured, its fill count the
        blob(s)' size from the CPU labels."""
        label, h = ref["label"], self.ctx.h
        ok_img, ok_fill, sizes, filled, levels = True, True, [], [], []
        for call in self.sample:
            self._prepare_call(call)
            self._launch()
            cuda.synchronize()
            out = self.ctx.d_img.copy_to_host()
            exp = img.copy()
            size = 0
            for (sx, sy), col in zip(call, self._colors()):
                m = label == int(sx) * h + int(sy)
                exp[m] = col
                size += int(np.count_nonzero(m))
                del m
            f = self._filled()
            ok_img &= bool(np.array_equal(out, exp))
            ok_fill &= f == size
            sizes.append(size)
            filled.append(f)
            levels.append((self.r.read() if self.kind == "ch03"
                           else self.k4.read())["levels"])
            del out, exp
        info = {"sample_blob_px": sizes, "sample_filled": filled,
                "sample_levels": levels}
        chk = {"image_matches_cpu": ok_img, "filled_matches_cpu": ok_fill}
        chk["ok"] = all(chk.values())
        return info, chk


class LoopSide(_BlobLoop):
    """The full loop, measured: restore and reset once (off the clock),
    then every call in turn, one seed kernel and one cooperative launch,
    between two events (after the flush and the sleep)."""
    paints = "loop"

    def run_timed(self, flush, sink):
        self.ctx.restore()
        self._reset_all()
        _gap(flush, sink)
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        for call in self.calls:
            self._seed(call)
            self._launch()
        e1.record()
        e1.synchronize()
        return {"total": cuda.event_elapsed_time(e0, e1)}

    def expected(self, ref, img):
        exp = img.copy()
        red = _red_mask(img)
        if self.kind == "ch03":
            exp[red] = BLUE
        else:
            pos = np.searchsorted(ref["roots"], ref["label"][red])
            exp[red] = k4.PALETTE_HOST[pos % 2]
        return exp

    def check_out(self, ref, img):
        out = self.ctx.d_img.copy_to_host()
        ok = bool(np.array_equal(out, self.expected(ref, img)))
        del out
        return ok

    def check(self, ref, img, flush, sink):
        run = self.run_timed(flush, sink)
        chk = {"image_matches_cpu": self.check_out(ref, img)}
        chk["ok"] = all(chk.values())
        return {"untimed_ms": run["total"]}, chk


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
        elif n == "ch08":
            sides.append(Ch08Side(n, ctx, CH08_DEFAULT))
        elif n == "ch08_nodense":
            sides.append(Ch08Side(n, ctx, CH08_NODENSE))
        elif n == "ch05_fused_L8":
            sides.append(Ch05Side(n, ctx, 8, "fused"))
        elif n == "ch05_split_L8":
            sides.append(Ch05Side(n, ctx, 8, "split"))
        elif n in CH03:
            sides.append(Ch03Side(n, ctx, runner, launches[n]))
        elif n == "ch02_global":
            sides.append(Ch02Side(n, ctx))
        elif n == "ch01_spill":
            sides.append(Ch01Side(n, ctx, "spill"))
        elif n == "ch01_ring":
            sides.append(Ch01Side(n, ctx, "ring"))
        elif n.startswith("est_"):
            sides.append(EstSide(n, ctx, "ch03" if "ch03" in n else "ch04"))
        elif n.startswith("loop_"):
            sides.append(LoopSide(n, ctx, "ch03" if "ch03" in n else "ch04"))
        else:
            raise KeyError(n)
    return sides


def _gap(flush, sink):
    """The gap before every timed launch: FLUSH_READS reads of the 96 MB
    buffer (each one evicts the 32 MB L2), a synchronize, then a ~1.5 ms
    GPU sleep, so every timed launch is in the stream before the first
    event fires. A probe on sq_256 (ch07 + ch01_ring, 3 s each): one read
    and a 2 ms sleep left the memory clock in P3 at 7,001 MHz for about
    half the samples (18-19 W); two reads and 1.5 ms held P0 at 8,001
    (24 W) with the same kernel times (ch07 0.043 vs 0.044 ms); five
    reads and no sleep held P0 at 39 W but put the launch on the clock
    (ch07 0.24 ms)."""
    for _ in range(FLUSH_READS):
        _l2_flush[512, 256](flush, sink)
    cuda.synchronize()
    gpu_sleep_kernel[1, 32](GAP_NAPS)


def run_once(side, flush, sink):
    if hasattr(side, "run_timed"):
        return side.run_timed(flush, sink)
    side.prepare()
    _gap(flush, sink)
    return side.launch()


def warm_block(sides, flush, sink, seconds, spacing=0.0):
    """The clock ramp: the block's own protocol, untimed, for `seconds`
    (at least one round), with the attempt's spacing. A spin of L2-flush
    reads drew 40 W, hit the laptop's 40 W cap and left the next short
    rounds at ~1,650 MHz; ch07's launch-bound spin idled at ~1,650-1,700.
    The scenes' own rounds settle at 2,070 MHz under the cap (both
    measured on this machine). Returns (rounds, seconds)."""
    t0 = time.time()
    n = 0
    while True:
        for s in sides:
            run_once(s, flush, sink)
            if spacing:
                _filler(flush, sink, spacing)
        n += 1
        if time.time() - t0 >= seconds:
            return n, time.time() - t0


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
        roots = np.unique(label[label >= 0])
        h = img.shape[1]
        ref.update(label=label, n_blobs=int(nb), roots=roots,
                   seeds=[(int(r) // h, int(r) % h) for r in roots],
                   palette=palette_image(img, label))
        if len(roots) != nb:
            raise RuntimeError("canonical roots != blob count")
    return ref


def check_side(side, ref, kind, img, flush=None, sink=None):
    """Compare one side's painted image (and counters) with the CPU."""
    if side.paints == "est":
        return side.check(ref, img)
    if side.paints == "loop":
        return side.check(ref, img, flush, sink)
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
            chk["no_dense_tiles"] = (info["dense_label_tiles"] == 0
                                     and info["dense_paint_tiles"] == 0)
            lab = side.label_map()
            chk["labels_match_cpu"] = bool(np.array_equal(lab, ref["label"]))
            del lab
    del out
    chk["ok"] = all(chk.values())
    return info, chk


def last_run_ok(side, ref, img):
    """The last timed run's output, checked like the untimed pass."""
    if side.paints == "est":
        return None                 # each call restores: checked untimed
    if side.paints == "loop":
        return side.check_out(ref, img)
    out = side.out().copy_to_host()
    want = (ref["blue"][side.conn] if side.paints == "blue"
            else ref["palette"])
    ok = bool(np.array_equal(out, want))
    del out
    return ok


# ---------------------------------------------------------------- scenes

def bench_scene(name, kind, img, seed, sampler, flush, sink, rounds_min):
    plan = BLOCKS[name]
    names = [n for _, ns, _ in plan for n in ns]
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
    if kind == "frame":
        ctx.cache["seeds"] = ref["seeds"]

    cells, blocks, block_of = {}, {}, {}
    samples_all = {}
    e2e_cells = {}
    for label, bnames, rmin in plan:
        rmin = max(rmin, rounds_min) if rmin == GPU_REPEATS else rmin
        sides = build_sides(bnames, ctx)
        # -- untimed pass: the checks
        infos, checks = {}, {}
        for s in sides:
            if s.paints not in ("est", "loop"):
                run_once(s, flush, sink)
            infos[s.name], checks[s.name] = check_side(s, ref, kind, img,
                                                       flush, sink)
            print(f"  check {s.name:18s} "
                  f"{'OK' if checks[s.name]['ok'] else 'MISMATCH'} "
                  f"{infos[s.name]}", flush=True)

        def round_fn(i, spacing, sides=sides):
            out = {}
            for s in (sides if i % 2 == 0 else sides[::-1]):
                out[s.name] = run_once(s, flush, sink)
                if spacing:
                    _filler(flush, sink, spacing)
            return out

        def warm_fn(sec, spacing, sides=sides):
            return warm_block(sides, flush, sink, sec, spacing)

        data, attempts, used = timed_attempts(f"{name}/{label}", warm_fn,
                                              round_fn, rmin, sampler)
        del round_fn, warm_fn       # they hold the block's device buffers
        blk = {"configs": bnames, "attempts": attempts, "used_attempt": used,
               **{k: v for k, v in attempts[used].items()
                  if k.startswith("clk") or k.startswith("power")
                  or k in ("timed_window_s", "rounds", "full_clocks")}}
        blocks[label] = blk
        # the work image holds the output of the block's last run only
        # (ch05 paints its own buffer)
        last = (sides if (len(data) - 1) % 2 == 0 else sides[::-1])[-1]
        for s in sides:
            block_of[s.name] = label
            samples_all[s.name] = [r[s.name] for r in data]
            if s is last or s.chapter == "ch05":
                lr = last_run_ok(s, ref, img)
                if lr is not None:
                    checks[s.name]["last_timed_run_matches_cpu"] = lr
                    checks[s.name]["ok"] = all(
                        v for k, v in checks[s.name].items() if k != "ok")
        # -- v4's host clock on the sampled estimate calls
        host = {s.name: s.host_clock(flush, sink) for s in sides
                if s.paints == "est"}
        # -- end to end (context)
        for s in sides:
            if E2E.get(name) == s.name:
                exp = ref["blue"][8] if kind == "one" else ref["palette"]
                e2e_cells[s.name] = e2e(s, img, exp)
        for s in sides:
            smp = samples_all[s.name]
            tot = [x["total"] for x in smp]
            cell = {**_stats(tot), "side": "gpu", "chapter": s.chapter
                    if s.chapter != "est" else s.meta["chapter"],
                    "block": label, "config": s.meta,
                    "reference_only": s.name in REFERENCE_ONLY,
                    "info": infos[s.name], "check": checks[s.name],
                    "samples": tot}
            extra = [k for k in smp[0] if k != "total"]
            if s.paints == "est":
                per = [x["per_call"] for x in smp]
                blob_med = [statistics.median(p[j] for p in per)
                            for j in range(len(per[0]))]
                cell.update(
                    est=True, calls=s.n_calls,
                    per_call_samples=per, per_call_median_ms=blob_med,
                    est_from_per_call_medians_ms=(
                        statistics.median(blob_med) * s.n_calls),
                    est_mean_based_ms=(statistics.mean(blob_med)
                                       * s.n_calls),
                    host_clock_per_call_ms=host[s.name],
                    host_clock_est_ms=(statistics.median(host[s.name])
                                       * s.n_calls))
            elif extra:
                cell["phase_ms"] = {k: statistics.median(
                    x[k] for x in smp) for k in extra}
                cell["phase_samples"] = {k: [x[k] for x in smp]
                                         for k in extra}
            if s.name in e2e_cells:
                cell["e2e"] = e2e_cells[s.name]
            cells[s.name] = cell
            c = cell
            print(f"  {s.name:18s} {c['ms']:10.3f} ms  [{c['ms_min']:.3f}, "
                  f"{c['ms_max']:.3f}] n={c['n']}"
                  + (f"  host-clock est {c['host_clock_est_ms']:.1f}"
                     if c.get("est") else ""), flush=True)
        del sides, data
        _release()

    pairs, unpaired = {}, {}
    for num, den in PAIRS.get(name, []):
        if num in cells and den in cells:
            if block_of[num] == block_of[den]:
                pairs[f"{num}/{den}"] = _ratio_stats(
                    cells[num]["samples"], cells[den]["samples"])
            else:
                unpaired[f"{num}/{den}"] = (cells[num]["ms"]
                                            / cells[den]["ms"])
    row = {"scene": name, "kind": kind, "note": SCENE_NOTES[name],
           "width": ctx.w, "height": ctx.h, "red_px": ref["red_px"],
           "configs": names, "cells": cells, "paired": pairs,
           "ratio_of_medians": unpaired, "blocks": blocks,
           "host_loadavg": list(os.getloadavg())}
    if kind == "one":
        row.update(seed=[int(seed[0]), int(seed[1])],
                   cpu_filled4=ref["filled4"], cpu_filled8=ref["filled8"],
                   cpu_levels4=ref["levels4"], cpu_levels8=ref["levels8"],
                   same_fill_4_8=ref["same_fill_4_8"])
    else:
        row["n_blobs"] = ref["n_blobs"]
    row["all_checks_ok"] = all(c["check"]["ok"] for c in cells.values()) \
        and all(c["matches_cpu"] for c in e2e_cells.values())
    for k, c in cells.items():
        bad = [x for x, v in c["check"].items() if v is False]
        if bad:
            print(f"  FAILED {k}: {bad}", flush=True)
    for k, c in e2e_cells.items():
        if not c["matches_cpu"]:
            print(f"  FAILED {k}: e2e output", flush=True)
    row["all_blocks_full_clocks"] = all(b["full_clocks"]
                                        for b in blocks.values())
    for k, v in pairs.items():
        print(f"  paired {k}: x{v['median']:.3f} [{v['min']:.3f}, "
              f"{v['max']:.3f}]", flush=True)
    for k, v in unpaired.items():
        print(f"  ratio of medians {k}: x{v:.3f}", flush=True)
    print(f"  {'all OK' if row['all_checks_ok'] else 'CHECK FAILED'}; "
          f"{'all blocks at full clocks' if row['all_blocks_full_clocks'] else 'NOT ALL BLOCKS AT FULL CLOCKS'}",
          flush=True)

    del ctx, ref
    _release()
    return row


def bench_drones(frames, sampler, flush, sink, rounds_min):
    """ch08 without its dense path on every drone frame: one engine, the
    timed rounds over all frames (the frame order reversed on odd
    rounds). First an untimed census of the default engine's dense tiles
    (the instrumented twin), then the checks of the timed engine."""
    w, h = frames[0]["img"].shape[:2]
    print(f"\ndrones: {len(frames)} frames of {w} x {h}", flush=True)
    ctx = Ctx(frames[0]["img"])
    pristine = [cuda.to_device(f["img"]) for f in frames]

    census = ff8.TileWaves(w, h, instrumented=True)
    for f, dp in zip(frames, pristine):
        ctx.d_img.copy_to_device(dp)
        _, ev = census.run(ctx.d_img, "rgb")
        ev[-1].synchronize()
        c = census.read_counters()
        f["census"] = {k: int(c[k]) for k in ("tiles_touched", "ccl_tiles",
                                               "paint_tiles",
                                               "paint_ccl_tiles")}
    del census
    _release()
    cen = {k: sum(f["census"][k] for f in frames)
           for k in frames[0]["census"]}
    cen["frames_with_dense_tiles"] = sum(
        (f["census"]["ccl_tiles"] + f["census"]["paint_ccl_tiles"]) > 0
        for f in frames)
    print(f"  default engine's dense tiles (census, untimed): {cen}",
          flush=True)

    side = Ch08Side("ch08_nodense", ctx, CH08_NODENSE)
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

    idx = list(range(len(frames)))

    def warm_fn(seconds, spacing):
        t0, j = time.time(), 0
        while True:
            ctx.d_pristine = pristine[j % len(frames)]
            run_once(side, flush, sink)
            j += 1
            if spacing and j % len(frames) == 0:
                _filler(flush, sink, spacing)
            if time.time() - t0 >= seconds and j >= len(frames):
                return j / len(frames), time.time() - t0

    def round_fn(i, spacing):
        """One pass over every frame; a retried attempt's spacing goes
        after the pass, not after each frame."""
        out = [None] * len(frames)
        for j in (idx if i % 2 == 0 else idx[::-1]):
            ctx.d_pristine = pristine[j]
            out[j] = run_once(side, flush, sink)["total"]
        if spacing:
            _filler(flush, sink, spacing)
        return out

    data, attempts, used = timed_attempts("drones", warm_fn, round_fn,
                                          rounds_min, sampler)
    samples = [[r[j] for r in data] for j in idx]

    # the last timed frame's output (frame 0 on an odd round count's
    # reversed order is last; check whichever ran last)
    last = idx[-1] if (len(data) - 1) % 2 == 0 else idx[0]
    lab_last, _ = cpu_label_components(frames[last]["img"])
    last_ok = bool(np.array_equal(ctx.d_img.copy_to_host(),
                                  palette_image(frames[last]["img"],
                                                lab_last)))
    del lab_last

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
    blk = {k: v for k, v in attempts[used].items()
           if k.startswith("clk") or k.startswith("power")
           or k in ("timed_window_s", "rounds", "full_clocks")}
    print(f"  ch08_nodense per-frame medians: median "
          f"{statistics.median(f['gpu']['ms'] for f in frames):.4f} ms, "
          f"max {max(f['gpu']['ms'] for f in frames):.4f}; worst run "
          f"{max(f['gpu']['ms_max'] for f in frames):.4f}", flush=True)
    return {"config": meta, "checks_ok": bool(ok_all),
            "last_timed_run_matches_cpu": last_ok,
            "e2e_matches_cpu": bool(e2e_ok),
            "default_engine_dense_census": cen,
            "attempts": attempts, "used_attempt": used,
            "host_loadavg": list(os.getloadavg()), **blk}


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
                dict(row=row, key="cpu_bfs4",
                     fn=(lambda i=img, x=sx, y=sy: cpu_flood_fill(i, x, y)),
                     check=(lambda r, w=row: {
                         "filled": int(r[3]),
                         "ok": int(r[3]) == w["cpu_filled4"]})),
                dict(row=row, key="cpu_bfs8",
                     fn=(lambda i=img, x=sx, y=sy: cpu_flood_fill_8(i, x, y)),
                     check=(lambda r, w=row: {
                         "filled": int(r[3]),
                         "ok": int(r[3]) == w["cpu_filled8"]}))]
        else:
            cells += [
                dict(row=row, key="cpu_label",
                     fn=(lambda i=img: cpu_label_components(i)),
                     check=(lambda r, w=row: {
                         "n_blobs": int(r[1]),
                         "ok": int(r[1]) == w["n_blobs"]})),
                dict(row=row, key="cpu_fill_canonical",
                     fn=(lambda i=img: cpu_fill_canonical(i)),
                     check=(lambda r, w=row: {
                         "filled": int(r[4]),
                         "ok": int(r[4]) == w["red_px"]}))]
    for f in frames:
        cells.append(dict(frame=f, key="cpu_label", calls=1,
                          fn=(lambda i=f["img"]: cpu_label_components(i)),
                          check=(lambda r, g=f: {
                              "n_blobs": int(r[1]),
                              "ok": int(r[1]) == g["n_blobs"]})))
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
            c["row"]["cells"][c["key"]] = {**cell, "fn": CPU_FNS[c["key"]]}
        else:
            c["frame"][c["key"]] = cell
    for row in rows:
        cells_r = {k: v for k, v in row["cells"].items() if v["side"] == "cpu"}
        for n in row["configs"]:
            g = row["cells"][n]
            twin = cells_r[TWIN[n]]
            g["twin"] = _twin(TWIN[n], twin, g)
            if "cpu_fill_canonical" in cells_r:
                g["x_vs_v4_baseline"] = (cells_r["cpu_fill_canonical"]["ms"]
                                         / g["ms"])
        row["all_checks_ok"] = row["all_checks_ok"] and all(
            c["check_ok"] for c in cells_r.values())
        print(f"  {row['scene']:16s} " + "  ".join(
            f"{k} {c['ms']:.3f} ms (n={c['n']})" for k, c in cells_r.items())
            + "  " + ", ".join(f"{n} x{row['cells'][n]['twin']['x']:.2f}"
                               for n in row["configs"]),
            flush=True)
    return rounds


def _twin(key, cpu, gpu):
    """A GPU cell's CPU twin and the ratio (> 1: the GPU is faster); the
    range uses the CPU's round medians, the GPU median fixed. With the
    copies too, where the cell has an end-to-end time."""
    out = {"cpu": key, "fn": CPU_FNS[key], "cpu_ms": cpu["ms"],
           "x": cpu["ms"] / gpu["ms"],
           "x_range_cpu_rounds": [min(cpu["round_medians"]) / gpu["ms"],
                                  max(cpu["round_medians"]) / gpu["ms"]]}
    if "e2e" in gpu:
        out["x_e2e"] = cpu["ms"] / gpu["e2e"]["ms"]
    return out


def _summary(values):
    v = sorted(values)
    return {"median_ms": statistics.median(v),
            "p90_ms": float(np.percentile(v, 90)), "max_ms": v[-1],
            "min_ms": v[0], "n": len(v)}


def story(rows, drones):
    """Every CPU-vs-GPU number of the v5 script, flat."""
    by = {r["scene"]: r for r in rows}
    out = []
    for key, beat, scene, cfg, what in STORY:
        if scene not in by or cfg not in by[scene]["cells"]:
            continue
        g = by[scene]["cells"][cfg]
        t = g["twin"]
        e = {"key": key, "beat": beat, "scene": scene, "gpu": cfg,
             "gpu_ms": g["ms"], "estimate": bool(g.get("est")),
             "cpu": t["cpu"], "cpu_ms": t["cpu_ms"], "x": t["x"],
             "x_range_cpu_rounds": t["x_range_cpu_rounds"],
             "winner": "GPU" if t["x"] > 1 else "CPU",
             "factor": t["x"] if t["x"] > 1 else 1 / t["x"],
             "gpu_block_full_clocks": by[scene]["blocks"][g["block"]][
                 "full_clocks"], "what": what}
        if "e2e" in g:
            e.update(gpu_e2e_ms=g["e2e"]["ms"], x_e2e=t["x_e2e"],
                     winner_e2e="GPU" if t["x_e2e"] > 1 else "CPU")
        out.append(e)
    if "input_blobs" in by:
        c = by["input_blobs"]["cells"]
        out.append({"key": "frame_cpu", "beat": "i1", "scene": "input_blobs",
                    "cpu": "cpu_label", "cpu_ms": c["cpu_label"]["ms"],
                    "what": "the CPU on the 81 Mpx frame (labels only)"})
    if drones:
        s = drones["summary"]
        out.append({"key": "drones", "beat": "i1, r14", "scene": "drones",
                    "gpu": "ch08_nodense", "gpu_ms": s["gpu"]["median_ms"],
                    "gpu_max_ms": s["gpu"]["max_ms"],
                    "gpu_worst_run_ms": s["gpu_worst_run_ms"],
                    "gpu_e2e_ms": s["gpu_e2e"]["median_ms"],
                    "gpu_e2e_max_ms": s["gpu_e2e"]["max_ms"],
                    "cpu": "cpu_label", "cpu_ms": s["cpu_label"]["median_ms"],
                    "cpu_max_ms": s["cpu_label"]["max_ms"],
                    "x": s["cpu_over_gpu_median"],
                    "gpu_block_full_clocks": drones["full_clocks"],
                    "what": ("per-frame medians over 240 frames, the tile "
                             "waves without the dense path, against the "
                             "CPU labels; budget 33 ms")})
    return out


# ----------------------------------------------------------------- main

def main():
    global MEM_FULL_MHZ
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="only these scenes (and 'drones'); default all")
    ap.add_argument("--rounds", type=int, default=GPU_REPEATS,
                    help="rounds per block, at least")
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
    mx = _smi("clocks.max.sm,clocks.max.mem").split(",")
    max_sm = int(float(mx[0])) if len(mx) == 2 else None
    MEM_FULL_MHZ = int(float(mx[1])) if len(mx) == 2 else 8001
    print(f"Device: {dev_name} ({sm_count} SMs); max clocks sm {max_sm}, "
          f"mem {MEM_FULL_MHZ} MHz", flush=True)
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
        scenes.append(("input_blobs", "frame", img, None))
    frames, v4_same = [], 0
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
    warm, wseeds = ch04_scenes.two_squares_scene(96, 64, 24, 24, gap=8)
    wx, wy = (int(v) for v in wseeds[0])
    wctx = Ctx(warm, (wx, wy))
    wref = reference("frame", warm, None, False)
    wctx.cache["seeds"] = wref["seeds"]              # two blobs: one pair
    flush = cuda.to_device(np.zeros(L2_FLUSH_BYTES // 4, np.uint32))
    sink = cuda.device_array(1, np.uint32)
    _l2_flush[512, 256](flush, sink)
    all_names = sorted({n for s in scenes for _, ns, _ in BLOCKS[s[0]]
                        for n in ns} | ({"ch08_nodense"} if frames else set()))
    for s in build_sides(all_names, wctx):
        run_once(s, flush, sink)
        if s.paints not in ("est", "loop"):
            s.info()
    gpu_sleep_kernel[1, 32](1)
    cuda.synchronize()
    del wctx, wref
    _release()
    cpu_flood_fill(warm, wx, wy)
    cpu_flood_fill_8(warm, wx, wy)
    cpu_fill_canonical(warm)
    cpu_label_components(warm)
    print(f"  warm in {time.perf_counter() - t_w:.0f} s", flush=True)

    sampler = Sampler()
    clocks_start = _clocks()

    rows = []
    for name, kind, img, seed in scenes:
        rows.append(bench_scene(name, kind, img, seed, sampler, flush, sink,
                                args.rounds))
    drones = None
    if frames:
        drones = bench_drones(frames, sampler, flush, sink, args.rounds)
    clocks_gpu_end = _clocks()

    print("\n@njit CPU cells, after every GPU block, interleaved:",
          flush=True)
    t_cpu0 = time.time()
    cpu_rounds = cpu_pass(rows, scenes, frames)
    t_cpu1 = time.time()
    sampler.stop()

    if frames:
        g = [f["gpu"]["ms"] for f in frames]
        cl = [f["cpu_label"]["ms"] for f in frames]
        drones["summary"] = {
            "gpu": _summary(g),
            "gpu_worst_run_ms": max(f["gpu"]["ms_max"] for f in frames),
            "gpu_e2e": _summary([f["e2e"]["ms"] for f in frames]),
            "cpu_label": _summary(cl),
            "cpu_over_gpu_median": statistics.median(cl) / statistics.median(g),
            "cpu_over_gpu_per_frame": _summary(
                [a / b for a, b in zip(cl, g)]),
            "cpu_over_gpu_e2e_median": (statistics.median(cl) / statistics
                                        .median(f["e2e"]["ms"]
                                                for f in frames)),
            "blobs": {"median": statistics.median(f["n_blobs"] for f in frames),
                      "min": min(f["n_blobs"] for f in frames),
                      "max": max(f["n_blobs"] for f in frames)},
        }
        drones["cpu_checks_ok"] = all(f["cpu_label"]["check_ok"]
                                      for f in frames)
        drones["frames"] = [
            {"key": f["key"], "file": f["file"], "mask_sha1": f["sha1"],
             "width": int(f["img"].shape[0]), "height": int(f["img"].shape[1]),
             "red_px": f["red_px"], "n_blobs": f["n_blobs"],
             "gpu": f["gpu"], "gpu_samples": f["gpu_samples"],
             "gpu_info": f["info"], "gpu_check": f["check"],
             "gpu_e2e": f["e2e"], "cpu_label": f["cpu_label"],
             "default_engine_census": f["census"]} for f in frames]
        drones["frames_sha1_equal_to_v4"] = int(v4_same)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = f"v5_{stamp}"
    st = story(rows, drones)
    payload = {
        "schema": "v5_cpu_vs_gpu_like_for_like",
        "session": session,
        "experiment": (
            "every CPU-vs-GPU number of the chain cut v5 in one session, "
            "each GPU cell against its CPU twin (same connectivity, same "
            "output): one small blob (sq_256), one huge blob "
            "(sq_8000_center), the 81 Mpx frame (input_blobs.png, act 2's "
            "estimates re-made with v4's method) and the 240 drone frames "
            "(ch08 without its dense path); every block at full clocks or "
            "run again; CUDA events, CPU last"),
        "rules": {
            "one_session": True,
            "cpu_twin": TWIN,
            "cpu_fns": CPU_FNS,
            "no_dense_tile_path": ("ch08's dense lists checked empty on "
                                   "every checked run: the frame's default "
                                   "engine and the drones' ch08_nodense"),
            "full_clocks": {"pstates": ["P0"], "mem_min_mhz": MEM_FULL_MHZ,
                            "sm_median_min_mhz": SM_MEDIAN_MIN,
                            "max_attempts": MAX_ATTEMPTS},
        },
        "device": dev_name,
        "sm_count": sm_count,
        "max_clocks": {"sm_mhz": max_sm, "mem_mhz": MEM_FULL_MHZ},
        "driver": _smi("driver_version"),
        "host": _host_info(),
        "versions": {"python": platform.python_version(),
                     "numba": numba.__version__, "numpy": np.__version__},
        "git_head": _git("rev-parse", "--short", "HEAD"),
        "script_matches_head": _git("status", "--porcelain", "--",
                                    "src/flood_fill_cuda/overview/"
                                    "bench_v5.py") == "",
        "config": {"gpu_rounds_min": args.rounds,
                   "min_timed_s": MIN_TIMED_S, "max_rounds": MAX_ROUNDS,
                   "e2e_repeats": E2E_REPEATS,
                   "cpu_rounds": CPU_ROUNDS, "cpu_round_s": CPU_ROUND_S,
                   "cpu_max_calls_per_round": CPU_MAX_CALLS,
                   "est_sample": EST_SAMPLE,
                   "host_clock_repeats": HOST_CLOCK_REPEATS,
                   "loop_rounds": LOOP_ROUNDS,
                   "drone_step": args.drone_step,
                   "warm_seconds": WARM_SECONDS,
                   "flush_reads": FLUSH_READS,
                   "gap_naps": GAP_NAPS,
                   "l2_flush_bytes": L2_FLUSH_BYTES,
                   "smi_period_ms": SMI_PERIOD_MS},
        "timing_note": (
            "GPU *_ms: one CUDA event before the first launch and one after "
            "the last, kernel-only (the image is already on the GPU and "
            "stays there). ch07 and ch08 time their init on the clock; "
            "ch01-ch05 and the loops reset their buffers on the device off "
            "the clock. Every run: image restored, two 96 MB L2-flush "
            "reads, sync, ~1.5 ms sleep kernel queued. Blocks of "
            "interleaved rounds (order "
            "reversed on odd rounds), at least the minimum, enough to span "
            "min_timed_s; a block not at full clocks runs again. est_* = "
            "v4's estimate: median sampled call x calls, each sampled call "
            "timed alone; host_clock_est_ms is v4's clock on the same calls. "
            "e2e = pageable H2D + pipeline + D2H (perf_counter), context "
            "only. CPU *_ms: @njit, one core, end to end of the call "
            "(allocations included, no paint), after all GPU work, in 5 "
            "interleaved rounds over every cell (median over all calls; "
            "round_medians show drift; cpu_rounds the host load). "
            "twin.x = CPU twin / GPU median (> 1: the GPU is faster)."),
        "inputs": {"png": PNG_NAME if png_md5 else None, "png_md5": png_md5,
                   "drone_masks": len(frames),
                   "v4_drone_session": V4_DRONES if frames else None,
                   "v4_estimates_session": V4_OVERVIEW},
        "clocks_idle": clocks_idle,
        "clocks_start": clocks_start,
        "clocks_gpu_end": clocks_gpu_end,
        "cpu_window_s": [round(t_cpu0 - sampler.t0, 2),
                         round(t_cpu1 - sampler.t0, 2)],
        "cpu_rounds": cpu_rounds,
        "clock_trace": sampler.trace(),
        "story": st,
        "scenes": rows,
        "drones": drones,
    }
    ok = all(r["all_checks_ok"] for r in rows) and (
        drones is None or (drones["checks_ok"] and drones["cpu_checks_ok"]
                           and drones["e2e_matches_cpu"]
                           and drones["last_timed_run_matches_cpu"]))
    full = all(r["all_blocks_full_clocks"] for r in rows) and (
        drones is None or drones["full_clocks"])
    payload["all_checks_ok"] = ok
    payload["all_blocks_full_clocks"] = full

    print(f"\nSession {session}  checks: {'all OK' if ok else 'FAILED'}; "
          f"clocks: {'all full' if full else 'NOT ALL FULL'}")
    print(f"{'scene':16s} {'config':18s} {'ms':>11s} {'cpu twin':>12s} "
          f"{'cpu ms':>10s} {'x':>9s}")
    for r in rows:
        for k, c in r["cells"].items():
            if c["side"] != "gpu":
                continue
            t = c["twin"]
            print(f"{r['scene']:16s} {k:18s} {c['ms']:11.4f} "
                  f"{t['cpu']:>12s} {t['cpu_ms']:10.3f} {t['x']:9.3f}"
                  + ("  (ref)" if c["reference_only"] else ""))
    if drones:
        s = drones["summary"]
        print(f"drones: ch08_nodense median {s['gpu']['median_ms']:.4f} ms, "
              f"max {s['gpu']['max_ms']:.4f}; cpu_label median "
              f"{s['cpu_label']['median_ms']:.3f}, max "
              f"{s['cpu_label']['max_ms']:.3f}; "
              f"x{s['cpu_over_gpu_median']:.2f}")
    print("\nstory:")
    for e in st:
        print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float)
                                     else v) for k, v in e.items()
                                 if k not in ("what",)}))
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
                    "n", "cpu_twin", "twin_ms", "x_twin", "reference_only",
                    "full_clocks", "check", "detail"])
        for r in rows:
            for k, c in r["cells"].items():
                gpu = c["side"] == "gpu"
                w.writerow([
                    r["scene"], c["side"], k, f"{c['ms']:.4f}",
                    f"{c['ms_min']:.4f}", f"{c['ms_max']:.4f}", c["n"],
                    c["twin"]["cpu"] if gpu else "",
                    f"{c['twin']['cpu_ms']:.4f}" if gpu else "",
                    f"{c['twin']['x']:.4f}" if gpu else "",
                    c.get("reference_only", ""),
                    r["blocks"][c["block"]]["full_clocks"] if gpu else "",
                    ("OK" if (c["check"]["ok"] if gpu else c["check_ok"])
                     else "MISMATCH"),
                    json.dumps(c["info"]) if gpu else c["fn"]])
        if drones:
            s = drones["summary"]
            for key, side, d in (("ch08_nodense", "gpu", s["gpu"]),
                                 ("cpu_label", "cpu", s["cpu_label"])):
                w.writerow([
                    "drones (per-frame medians)", side, key,
                    f"{d['median_ms']:.4f}", f"{d['min_ms']:.4f}",
                    f"{d['max_ms']:.4f}", d["n"],
                    "cpu_label" if side == "gpu" else "",
                    f"{s['cpu_label']['median_ms']:.4f}" if side == "gpu"
                    else "",
                    f"{s['cpu_over_gpu_median']:.4f}" if side == "gpu"
                    else "", False,
                    drones["full_clocks"] if side == "gpu" else "",
                    "OK" if (drones["checks_ok"] if side == "gpu"
                             else drones["cpu_checks_ok"]) else "MISMATCH",
                    f"p90 {d['p90_ms']:.4f}"])
    print(f"\nResults written: {os.path.basename(json_path)}, "
          f"{os.path.basename(csv_path)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
