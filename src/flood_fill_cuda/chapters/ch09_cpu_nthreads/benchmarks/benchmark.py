"""ch09: every CPU core against one core and the GPU, in one session.

The question this session answers: the video compares the GPU with ONE
CPU core doing the same work. What do ALL the cores do, with the same
work and the same ideas (ch09's pixel BFS, tile BFS and tile labels),
and where does the GPU stand against that?

Scenes (bench_v5's, plus the serpentine):

  sq_256           a 128 x 128 square in a 256 x 256 image, centre seed
  sq_8000_center   8000 x 8000, all red, centre seed (64 Mpx)
  serpentine_256   ch03's serpentine, 256 x 256: 32,641 one-pixel levels
                   at 8-conn, a one-tile front
                   (each of these at 8-conn and at 4-conn)
  input_blobs      the 81 Mpx frame (md5 checked), 2,522 blobs, 8-conn
  drones           the 240 masks of video/assets/drones_mask, if present
                   (--drones-dir), 8-conn labels per frame

CPU cells (one blob; "on" = reset on the clock, like ch07 and ch08):

  ch09_pixel_t{n}        PixelBFS, paint=True (fill + paint), n threads
  ch09_tile_t{n}         TileBFS, paint=True (pack + rounds + paint)
  ch09_pixel_fill_t{n}   the same without the paint (n = 1 and 16 only)
  ch09_tile_fill_t{n}    the same without the paint (n = 1 and 16 only)
  one_fill               ONE core: the oracle's loop without its depth
                         map, buffers kept, reset on the clock (v5's
                         cpu_fill8 twin, at this cell's connectivity)
  one_fill_paint         the same plus a serial paint: the one-core twin
                         of the full pipelines above (like for like)
  one_bfs                ONE core with the depth map, reset before the
                         clock (v5's cpu_bfs twin): reference
  oracle_bfs             the oracle as called (fresh arrays): reference

CPU cells (many blobs):

  ch09_labels_t{n}       TileLabels, lattice 8, paint=True
  ch09_labels_nopaint_t{n}   labels only (n = 1 and 16)
  ch09_labels_L0_t16     lattice 0 (no lattice seeds), paint=True
  one_label              ONE core: ch05's labels-only loop, buffers
                         kept, reset on the clock (v5's cpu_label)
  one_label_paint        the same plus a serial palette paint (like for
                         like with the paint=True cells)
  oracle_label           cpu_label_components as called: reference

GPU cells (CUDA events, the image restored and L2 flushed off the
clock, bench_v5's gap before every timed launch):

  ch03_conn{4,8}   ch03's level-synchronous pixel BFS (ch07's benchmark
                   configs, bench_v5's twins): the same idea as ch09's
                   pixel BFS. Its visited reset, seed and kernel are on
                   the clock (kernel_only: the kernel alone, bench_v5's
                   clock); the depth, owner and stats resets are not
  ch03_wc_bare     ch03's uninstrumented warp kernel, 8-conn: reference
  ch07_c{4,8}      ch07's default TileFill (one_sync + tile_paint),
                   bare, 128 threads per block, at its own grid: the
                   same idea as ch09's tile BFS
  ch08_old         ch08's first version (flood_fill.OLD_PATH), rgb: the
                   same idea as ch09's tile labels (no dense-tile path)
  ch08             ch08's default engine (the dense-tile path), rgb

Cold cells (scenes of at most COLD_MAX_BYTES, the drone frames too):

  <key>_cold    the headline cells (ch09 at 1 thread and at the most
                threads, painting, and the one-core like-for-like twin)
                again, with the caches flushed before every call
                (flush_caches: a 64 MB read on every thread, then on
                the calling thread, then the pool parked; at most
                COLD_MAX_CALLS calls a round). The GPU's L2 is flushed
                before each of its timed launches, while the hot cells'
                image and buffers sit in the 24 MB L3: on the small
                scenes the summary gives both (cache "hot", "cold")

CPU cells (dispatch_1x1, dispatch_1x1_labels: CPU only):

  the cells above on a 1 x 1 image: the fixed cost of one call at each
  thread count (the checks, the thread count, every njit dispatch, the
  stats dict, the forks), which the CPU clock carries and the GPU's
  events do not

GPU context (perf_counter, after each timed block, E2E_REPEATS each):

  e2e         a pageable host-to-GPU copy of the image, the pipeline,
              the copy back (bench_v5's e2e; the CPU's data is already
              in host memory). Its output is checked
  host_wall   launch to synchronize with the image already on the GPU
              (no gap): what the host waits for
  Factors: gpu_x (events), gpu_x_e2e, gpu_x_host_wall

Policy:

  one session     GPU blocks first (a CPU pass between them lets the GPU
                  clock fall), then CPU_COOLDOWN_S of idle, then the CPU
                  passes, scene by scene
  idle gate       before every CPU round and every GPU attempt, wait (up
                  to QUIET_WAIT_*_S) until the REST of the machine
                  (/proc/stat's busy time minus this process's own CPU
                  time, all its threads) is at most QUIET_BUSY_PCT_
                  PARALLEL (3%) busy over one second for rounds with
                  cells of 8 threads or more, QUIET_BUSY_PCT (10%) for
                  the GPU attempts. The same share is measured over
                  every CPU round and every cell's batch; a round above
                  its threshold is run again (up to MAX_ROUND_REDOS
                  times, the discarded attempts recorded), then kept and
                  flagged "loaded". WSL2 shows no steal time: load on
                  the Windows side is invisible to the gate, so
  probe           every CPU round starts with a fixed integer chain on
                  one thread and the same chain on every thread at once,
                  timed, each time also relative to the session's first
                  probe: Windows-side load or a throttled package shows
                  as a ratio above 1 (recorded, not a redo criterion)
  runtime         Numba's threading layer is pinned (omp) before the
                  first parallel call. Niceness, CPU affinity, the
                  OpenMP and Numba variables, the chunk size, the
                  kernel, the OpenMP runtime and the topology the guest
                  sees are recorded. A timed run stops at nice > 0, on a
                  CPU affinity narrower than the pool, or with an OpenMP
                  wait, spin, bind or schedule variable set (flags turn
                  these into recorded notes). Threads are not bound:
                  under WSL2 a binding pins vCPUs, not P- or E-cores
  JIT warm-up     every engine and twin runs at every thread count on a
                  small scene first (compiles, thread pool start)
  settle          before every cell's batch, and before every drone
                  call with fewer threads than the call before it, wait
                  (off the clock) until the previous call's OpenMP
                  workers have parked: libgomp's idle workers spin for
                  about 10 ms after a parallel call by default, beside
                  whatever runs next (a one-core twin after a 16-thread
                  cell would share the package with 15 spinning
                  threads). The wait policy itself is left at its
                  default, so a cell's own back-to-back calls keep it
  CPU rounds      one untimed checked call per cell, one timed first
                  call (sets the calls per round: a cell under
                  CPU_ROUND_S repeats, at most CPU_MAX_CALLS), then
                  CPU_ROUNDS rounds over every cell of the scene, the
                  order reversed on odd rounds. Before every call the
                  work image is restored from a pristine copy, off the
                  clock, for every cell alike (one-core twins too)
  GPU blocks      bench_v5's: a 4 s ramp of the block's own runs, then
                  rounds (order reversed on odd rounds) spanning at
                  least MIN_TIMED_S; a block is at full clocks if every
                  nvidia-smi sample of its window is P0, the memory clock
                  at its maximum and the SM median at least
                  SM_MEDIAN_MIN; otherwise it runs again (up to
                  MAX_ATTEMPTS, with more light load between runs)
  stats           medians, min, max, stdev, every sample; per-round
                  medians; speedups and CPU-vs-GPU factors with their
                  range over the CPU rounds. gpu_x is against the GPU
                  cell with the same idea (gpu_cell: pixel - ch03_conn
                  {c}, tile - ch07_c{c}, labels - ch08_old); gpu_x_other
                  against every other GPU cell of that connectivity,
                  kept apart

Checks (every CPU output): EVERY timed call of a CPU cell is checked
right after it, off the clock and on one thread (a check must not wake
the thread pool between calls): the work image must equal the expected
one (blue on the fill set, palette[label % 6] on red, or the pristine
image for a cell that does not paint), fill counts and levels (pixel
BFS) the oracle's, the tile BFS's rounds the first call's, label maps
(ch09's off-clock map, the twins' arrays) and blob counts the oracle's,
and ch09's wave invariants must hold (every ring piece expanded once,
claims = ring pieces - seeds + dropped entries). A cell's untimed call
and its last timed call also compare the visited sets. A cell with
any mismatch is reported (calls_checked, call_mismatches, where) and
fails all_checks_ok. The drone pass checks every call in full. GPU
cells check their untimed run and their last timed run (ch08's label
map too): a host check between timed GPU runs would let the clock
fall.

Output: results/ch09_cpu_nthreads/benchmark_results/ch09_<stamp>.json
and ch09_<stamp>.csv (files named, never paths). The drones get a
summary (per frame, then median, p90, min and max over the frames:
ch09 against the one-core twin, paired per round, and against the GPU
by events and end to end) and their own CSV lines (ms: the median of
the per-frame medians). Every sample is kept, per frame for the
drones.

Run (on a quiet machine, never under nice; the GPU part takes the GPU
lock if one is used on this host):

    PYTHONUNBUFFERED=1 uv run python -m \\
        flood_fill_cuda.chapters.ch09_cpu_nthreads.benchmarks.benchmark

Smoke (small sizes, no gate, nothing written):

    ... benchmark --smoke --no-save [--drones-dir DIR] [--no-gpu]

Budget: about 15-25 min. Peak host memory about 3.5 GB (one scene's
buffers at a time: the oracles' arrays and the one-core twins' buffers
of the 64 Mpx square and of the 81 Mpx frame are the largest).
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
import resource
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

import numba
import numpy as np
from numba import njit, prange

from ....shared import results_paths
from ....shared import scenes as shared_scenes
from ...ch03_gpu_1blob_nblock.cpu_oracle import cpu_flood_fill, cpu_flood_fill_8
from ...ch05_gpu_nblob_nblock import scenes as ch05_scenes
from ...ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components
from .. import flood_fill as ff
from .. import kernels as K

CHAPTER = "ch09_cpu_nthreads"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    *[".."] * 5))
PNG_NAME = "input_blobs.png"
PNG_PATH = os.path.join(REPO, "images", "input", PNG_NAME)
PNG_MD5 = "47639eb37f973b5092ec2c4db86f45a4"     # ch08's benchmark PNG_MD5
MASK_DIR = os.path.join(REPO, "video", "assets", "drones_mask")

THREADS = (1, 2, 4, 8, 16)
NOPAINT_THREADS = (1, 16)
CPU_ROUNDS = 6             # even: as many rounds forward as reversed
CPU_ROUND_S = 0.4
CPU_MAX_CALLS = 101
CPU_COOLDOWN_S = 60
QUIET_BUSY_PCT = 10.0       # the rest of the host, at most (1.6 of 16):
                            # GPU attempts, rounds of fewer than 8 threads
QUIET_BUSY_PCT_PARALLEL = 3.0   # the same for rounds with 8+ threads: one
                                # other runnable thread stalls a barrier
MAX_ROUND_REDOS = 2
PROBE_N = 8_000_000         # the calibration chain (about 10 ms per core)
COLD_MAX_BYTES = 8 << 20    # scenes whose image is at most this (it and
                            # the buffers fit the 24 MB L3) get cold cells
COLD_MAX_CALLS = 15         # a cold call costs a flush (about 30 ms)
FLUSH_BYTES = 64 << 20      # read to evict the L2s and the 24 MB L3
QUIET_WAIT_CPU_S = 600
QUIET_WAIT_GPU_S = 120
GPU_ROUNDS = 9
WARM_SECONDS = 4.0
MIN_TIMED_S = 1.5
MAX_ROUNDS = 300
MAX_ATTEMPTS = 4
SPACING_S = (0.0, 0.1, 0.25, 0.5)
SM_MEDIAN_MIN = 2040
SMI_PERIOD_MS = 100
FLUSH_READS = 2
GAP_NAPS = 15
L2_FLUSH_BYTES = 96 << 20
RED = np.array([255, 0, 0], np.uint8)
BLUE = np.array([0, 0, 255], np.uint8)

EVENT_REASONS = {0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap",
                 0x8: "hw_slowdown", 0x10: "sync_boost",
                 0x20: "sw_thermal", 0x40: "hw_thermal",
                 0x80: "hw_power_brake", 0x100: "display_clock"}

# scenes without GPU cells: the dispatch-overhead cells (1 x 1)
CPU_ONLY_SCENES = ("dispatch",)
E2E_REPEATS = 5

# the GPU cell with the same idea as each ch09 variant
SAME_IDEA_GPU = {"pixel": "ch03_conn{conn}", "tile": "ch07_c{conn}",
                 "labels": "ch08_old"}

# which one-core cell is each ch09 / GPU cell's like-for-like twin
TWIN_FULL = {"one": "one_fill_paint", "many": "one_label_paint"}
TWIN_NOPAINT = {"one": "one_fill", "many": "one_label"}


# ------------------------------------------------------------ one core

@njit(cache=True)
def _reset_i32(a, value):
    flat = a.reshape(a.size)
    for i in range(flat.size):
        flat[i] = value


@njit(cache=True)
def fill_into(img, seed_x, seed_y, visited, queue, conn8):
    """The oracles' BFS loop (ch01's 4-conn, shared/'s 8-conn, their
    neighbour orders) without the depth map: visited only."""
    width, height = img.shape[0], img.shape[1]
    nd = 8 if conn8 else 4
    visited[seed_x, seed_y] = 1
    queue[0] = seed_x * height + seed_y
    front, rear = 0, 1
    level = 0
    while front < rear:
        level_end = rear
        for i in range(front, level_end):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            for d in range(nd):
                if conn8:
                    nx = x + K.DX8[d]
                    ny = y + K.DY8[d]
                else:
                    nx = x + K.DX4[d]
                    ny = y + K.DY4[d]
                if 0 <= nx < width and 0 <= ny < height and visited[nx, ny] == 0:
                    if img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0 and img[nx, ny, 2] == 0:
                        visited[nx, ny] = 1
                        queue[rear] = nx * height + ny
                        rear += 1
        front = level_end
        level += 1
    return level, rear


@njit(cache=True)
def bfs_into(img, seed_x, seed_y, visited, depth, queue, conn8):
    """The same loop with the oracles' depth map."""
    width, height = img.shape[0], img.shape[1]
    nd = 8 if conn8 else 4
    visited[seed_x, seed_y] = 1
    queue[0] = seed_x * height + seed_y
    front, rear = 0, 1
    level = 0
    while front < rear:
        level_end = rear
        for i in range(front, level_end):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            depth[x, y] = level
            for d in range(nd):
                if conn8:
                    nx = x + K.DX8[d]
                    ny = y + K.DY8[d]
                else:
                    nx = x + K.DX4[d]
                    ny = y + K.DY4[d]
                if 0 <= nx < width and 0 <= ny < height and visited[nx, ny] == 0:
                    if img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0 and img[nx, ny, 2] == 0:
                        visited[nx, ny] = 1
                        queue[rear] = nx * height + ny
                        rear += 1
        front = level_end
        level += 1
    return level, rear


@njit(cache=True)
def label_into(img, label, queue):
    """ch05's cpu_label_components loop (8-conn, canonical labels) on
    given buffers; label must be -1 everywhere. Returns n_blobs."""
    width, height = img.shape[0], img.shape[1]
    n_blobs = 0
    for x in range(width):
        for y in range(height):
            if not (img[x, y, 0] == 255 and img[x, y, 1] == 0
                    and img[x, y, 2] == 0) or label[x, y] >= 0:
                continue
            n_blobs += 1
            root = x * height + y
            label[x, y] = root
            queue[0] = root
            front, rear = 0, 1
            while front < rear:
                pixel = queue[front]
                front += 1
                px = pixel // height
                py = pixel % height
                for d in range(8):
                    nx = px + K.DX8[d]
                    ny = py + K.DY8[d]
                    if (0 <= nx < width and 0 <= ny < height
                            and label[nx, ny] < 0
                            and img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0
                            and img[nx, ny, 2] == 0):
                        label[nx, ny] = root
                        queue[rear] = nx * height + ny
                        rear += 1
    return n_blobs


@njit(cache=True)
def paint_blue_serial(img, visited):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if visited[x, y] != 0:
                img[x, y, 0] = 0
                img[x, y, 1] = 0
                img[x, y, 2] = 255


@njit(cache=True)
def paint_palette_serial(img, label):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            lb = label[x, y]
            if lb >= 0:
                c = lb % K.N_PALETTE
                img[x, y, 0] = K.PALETTE[c, 0]
                img[x, y, 1] = K.PALETTE[c, 1]
                img[x, y, 2] = K.PALETTE[c, 2]


@njit(cache=True)
def _count_differences(a, b):
    """Positions where two flat arrays of one dtype and size differ (no
    early exit, so the loop vectorises)."""
    n = 0
    for i in range(a.size):
        if a[i] != b[i]:
            n += 1
    return n


def same(a, b):
    """a and b have one shape, one dtype and equal values (no temporary
    arrays: the checks after every timed call stay cheap)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return _count_differences(np.ascontiguousarray(a).reshape(-1),
                              np.ascontiguousarray(b).reshape(-1)) == 0


@njit(cache=True)
def _read_serial(a):
    s = np.uint64(0)
    for i in range(a.size):
        s += a[i]
    return s


@njit(parallel=True, cache=True)
def _read_parallel(a):
    s = np.uint64(0)
    for i in prange(a.size):
        s += a[i]
    return s


_FLUSH = {}


def flush_caches():
    """Evict the CPU caches, off the clock (a cold call's prep): a read
    of FLUSH_BYTES on every thread (every core's L2, the L3), then the
    same read on this thread (its own core and the L3 again, long
    enough for the pool's spin to end), then settle_workers(). The GPU
    side's twin is its L2 flush before every timed launch."""
    if "buf" not in _FLUSH:
        _FLUSH["buf"] = np.ones(FLUSH_BYTES // 8, np.uint64)
    buf = _FLUSH["buf"]
    with ff.num_threads(numba.config.NUMBA_NUM_THREADS):
        _read_parallel(buf)
    _read_serial(buf)
    settle_workers()


class OneCore:
    """The one-core twins of one image shape, buffers allocated once."""

    def __init__(self, width, height, kind):
        self.kind = kind
        self.queue = np.zeros(width * height, np.int64)
        if kind == "one":
            self.visited = np.zeros((width, height), np.int32)
            self.depth = np.full((width, height), -1, np.int32)
        else:
            self.label = np.full((width, height), -1, np.int32)

    def fill(self, img, sx, sy, conn8, paint):
        _reset_i32(self.visited, 0)                 # on the clock
        res = fill_into(img, sx, sy, self.visited, self.queue, conn8)
        if paint:
            paint_blue_serial(img, self.visited)
        return res

    def bfs_prep(self):
        _reset_i32(self.visited, 0)                 # before the clock
        _reset_i32(self.depth, -1)

    def bfs(self, img, sx, sy, conn8):
        return bfs_into(img, sx, sy, self.visited, self.depth, self.queue,
                        conn8)

    def labels(self, img, paint):
        _reset_i32(self.label, -1)                  # on the clock
        n = label_into(img, self.label, self.queue)
        if paint:
            paint_palette_serial(img, self.label)
        return n


# ------------------------------------------------------------ helpers

def _stats(vals):
    v = list(vals)
    return {"ms": statistics.median(v), "ms_min": min(v), "ms_max": max(v),
            "ms_stdev": statistics.stdev(v) if len(v) > 1 else 0.0,
            "n": len(v)}


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
    white (bench_drones.mask_image / bench_v5's)."""
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
    out[m] = K.PALETTE[label[m] % K.N_PALETTE]
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10, cwd=REPO).stdout.strip()
    except Exception:
        return ""


def _host_info():
    """The CPU as this (possibly virtual) machine sees it. Under WSL2 the
    guest sees vCPUs: a synthetic 8 cores x 2 threads, scheduled by
    Windows over the real P- and E-cores, with no cpufreq and no hybrid
    PMUs, so threads are not bound to cores (OMP_PROC_BIND unset)."""
    model = platform.machine()
    flags, cores = set(), set()
    phys = core = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "model name" and model == platform.machine():
                    model = v
                elif k == "flags" and not flags:
                    flags = set(v.split())
                elif k == "physical id":
                    phys = v
                elif k == "core id":
                    core = v
                elif not k and phys is not None:
                    cores.add((phys, core))
                    phys = core = None
        if phys is not None:
            cores.add((phys, core))
    except OSError:
        pass
    logical = os.cpu_count()
    return {"cpu": model, "logical_cpus": logical,
            "hypervisor": "hypervisor" in flags,
            "cores_seen": len(cores) or None,
            "threads_per_core_seen": (logical // len(cores)
                                      if cores and logical else None),
            "cpufreq_visible": os.path.isdir(
                "/sys/devices/system/cpu/cpu0/cpufreq"),
            "hybrid_pmus_visible": (os.path.isdir("/sys/devices/cpu_core")
                                    or os.path.isdir("/sys/devices/cpu_atom")),
            "numba": numba.__version__,
            "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
            "python": platform.python_version()}


# OpenMP settings that change how the thread pool waits, binds or
# schedules: recorded always, and a timed run refuses them unless
# --allow-openmp-env (the session measures libgomp's defaults)
OPENMP_SKEW_ENV = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_PROC_BIND",
                   "OMP_PLACES", "OMP_SCHEDULE", "OMP_DYNAMIC",
                   "KMP_BLOCKTIME")
RUNTIME_ENV_PREFIXES = ("OMP_", "GOMP_", "KMP_", "NUMBA_")
RUNTIME_LIBS = ("libgomp", "libiomp", "libomp", "libtbb")


def _runtime_libs():
    """The OpenMP / TBB libraries this process has loaded: basenames and
    the GCC version string found in each (paths are never recorded)."""
    import re
    found = {}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.split()[-1] if len(line.split()) >= 6 else ""
                base = os.path.basename(path)
                if base.startswith(RUNTIME_LIBS) and base not in found:
                    found[base] = path
    except OSError:
        return {}
    out = {}
    for base, path in sorted(found.items()):
        version = None
        try:
            with open(path, "rb") as f:
                m = re.search(rb"GCC: \(([^)]{0,80})\) ([0-9][0-9.]*)",
                              f.read())
            if m:
                version = (f"GCC {m.group(2).decode()} "
                           f"({m.group(1).decode(errors='replace')})")
        except OSError:
            pass
        if version is None and base.startswith("libgomp"):
            version = _dpkg_version("libgomp1")
        out[base] = version if version and "/" not in version else None
    return out


def _dpkg_version(package):
    """The Debian package's version, or None (no dpkg, no package)."""
    try:
        r = subprocess.run(["dpkg-query", "-W", "-f=${Version}", package],
                           capture_output=True, text=True, timeout=10)
        v = r.stdout.strip()
        return f"{package} {v}" if r.returncode == 0 and v else None
    except (OSError, subprocess.SubprocessError):
        return None


def runtime_info():
    """How the CPU side is scheduled in this process (recorded in the
    session): Numba's threading layer and chunk size, the niceness, the
    CPU affinity, the OpenMP and Numba variables (a value that holds a
    path is replaced), the kernel and the OpenMP runtime loaded."""
    try:
        layer = numba.threading_layer()
    except ValueError:
        layer = "not started"
    env = {}
    for k, v in sorted(os.environ.items()):
        if k.startswith(RUNTIME_ENV_PREFIXES):
            env[k] = "<path>" if "/" in v else v
    for k in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_PROC_BIND",
              "OMP_PLACES"):
        env.setdefault(k, "unset")
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count()
    release = platform.release()
    return {"threading_layer": layer,
            "parallel_chunksize": numba.get_parallel_chunksize(),
            "niceness": os.getpriority(os.PRIO_PROCESS, 0),
            "affinity_cpus": affinity,
            "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
            "kernel": release, "wsl": "microsoft" in release.lower(),
            "openmp_env": env, "runtime_libs": _runtime_libs()}


def check_runtime(args, info):
    """Settings that would bias the all-core cells against the one-core
    ones. A timed run stops on any of them unless a flag allows it; a
    smoke run, or an allowed one, gets notes (recorded). Returns the
    notes."""
    notes, stop = [], []
    if info["niceness"] > 0:
        msg = (f"the process runs at nice {info['niceness']}: a 16-thread "
               "barrier waits for its slowest thread, and a low-priority "
               "thread that shares a CPU loses far more than a lone one")
        (notes if args.smoke or args.allow_nice else stop).append(msg)
    if info["affinity_cpus"] < info["numba_num_threads"]:
        msg = (f"the CPU affinity has {info['affinity_cpus']} CPUs for "
               f"{info['numba_num_threads']} threads")
        (notes if args.smoke or args.allow_nice else stop).append(msg)
    preset = {k: v for k, v in info["openmp_env"].items()
              if k in OPENMP_SKEW_ENV and v != "unset"}
    if preset:
        msg = (f"OpenMP settings are preset ({preset}); the session "
               "measures libgomp's defaults")
        (notes if args.smoke or args.allow_openmp_env else stop).append(msg)
    if stop:
        raise SystemExit(
            "Not a fair timed run: " + "; ".join(stop) + ". Rerun without "
            "nice and with those variables unset, or pass --allow-nice / "
            "--allow-openmp-env to record them as notes.")
    return notes


def _cpu_busy():
    """(busy, total) jiffies of the whole machine, from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:9]]
        return sum(v) - v[3] - v[4], sum(v)
    except (OSError, ValueError):
        return 0, 0


SETTLE_WINDOW_S = 0.005
SETTLE_QUIET_NS = 250_000        # 5% of one CPU over the window
SETTLE_MAX_S = 0.5
SETTLE_FALLBACK_S = 0.05


def _pool_cpu_ns():
    """CPU time (ns) of this process's threads other than the calling
    one (the OpenMP workers), from /proc/self/task/*/schedstat; None if
    the kernel does not provide it."""
    me = threading.get_native_id()
    total = 0
    try:
        for tid in os.listdir("/proc/self/task"):
            if int(tid) == me:
                continue
            try:
                with open(f"/proc/self/task/{tid}/schedstat") as f:
                    total += int(f.read().split()[0])
            except (FileNotFoundError, ProcessLookupError):
                continue          # a thread that just exited
    except (OSError, ValueError, IndexError):
        return None
    return total


def settle_workers(max_s=SETTLE_MAX_S):
    """Wait until the thread pool is parked: after a parallel call,
    libgomp's idle workers spin (about 10 ms by default) before they
    sleep, beside whatever runs next. Polls the workers' CPU time until
    a window passes with almost none. Off the clock."""
    t0 = time.perf_counter()
    prev = _pool_cpu_ns()
    if prev is None:
        time.sleep(SETTLE_FALLBACK_S)
        return {"settled": None,
                "waited_ms": (time.perf_counter() - t0) * 1000}
    while True:
        time.sleep(SETTLE_WINDOW_S)
        cur = _pool_cpu_ns()
        if cur is None or cur - prev <= SETTLE_QUIET_NS:
            return {"settled": cur is not None,
                    "waited_ms": (time.perf_counter() - t0) * 1000}
        if time.perf_counter() - t0 >= max_s:
            return {"settled": False,
                    "waited_ms": (time.perf_counter() - t0) * 1000}
        prev = cur


CLK_TCK = os.sysconf("SC_CLK_TCK")


def _own_cpu_s():
    """CPU seconds of this process, every thread (its OpenMP workers
    too)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


class LoadMeter:
    """The machine's busy share over a window, and the share of the REST
    of the machine: /proc/stat's busy jiffies (all CPUs) minus this
    process's own CPU time, over all CPUs' jiffies. WSL2 reports no
    steal time, so load on the Windows side is invisible here: the
    calibration probe is what shows it."""

    def __init__(self, kind="batch"):
        self.kind = kind
        self.b0, self.j0 = _cpu_busy()
        self.o0 = _own_cpu_s()

    def read(self):
        b1, j1 = _cpu_busy()
        o1 = _own_cpu_s()
        dj = j1 - self.j0
        if dj <= 0:
            return {"others_pct": None, "machine_pct": None}
        busy = b1 - self.b0
        own = (o1 - self.o0) * CLK_TCK
        return {"machine_pct": round(100.0 * busy / dj, 1),
                "others_pct": round(max(0.0, 100.0 * (busy - own) / dj), 1)}


def gate_threshold(max_threads):
    """The others' share a round may see: strict when it has cells of 8
    threads or more (a barrier waits for any preempted thread)."""
    return QUIET_BUSY_PCT_PARALLEL if max_threads >= 8 else QUIET_BUSY_PCT


def wait_quiet(max_wait_s, enabled=True, threshold=QUIET_BUSY_PCT):
    """Until the rest of the machine (not this process) is at most
    `threshold` busy over 1 s, or max_wait_s passed. Returns what it
    saw."""
    if not enabled:
        return {"quiet": None, "others_pct": None, "waited_s": 0.0,
                "gate": "off"}
    t0 = time.time()
    while True:
        meter = LoadMeter("gate")
        time.sleep(1.0)
        load = meter.read()
        others = load["others_pct"] or 0.0
        waited = time.time() - t0
        if others <= threshold or waited >= max_wait_s:
            return {"quiet": others <= threshold, "others_pct": others,
                    "machine_pct": load["machine_pct"],
                    "threshold_pct": threshold, "waited_s": round(waited, 1)}


@njit(cache=True)
def _probe_chain(n, seed):
    """A fixed dependent chain of n xorshift64 steps (shifts and xors do
    not reassociate, so the compiler cannot shorten it)."""
    x = np.uint64(seed) | np.uint64(1)
    for _ in range(n):
        x ^= x >> np.uint64(12)
        x ^= x << np.uint64(25)
        x ^= x >> np.uint64(27)
    return x


@njit(parallel=True, cache=True)
def _probe_all(n, nt):
    """The chain on every thread at once, each from its own seed (so no
    call is loop-invariant)."""
    out = np.empty(nt, np.uint64)
    for c in prange(nt):
        out[c] = _probe_chain(n, c + 1)
    return out


class Calibration:
    """A fixed amount of work at the start of every round: the chain on
    one thread, then on every thread at once. Each time is also given
    relative to the session's first probe: Windows-side load (which the
    gate cannot see) or a throttled package shows as a ratio above 1."""

    def __init__(self, n=PROBE_N):
        self.n = n
        self.base = None

    def warm(self, nt):
        """One untimed probe (compiled, the clock ramped): the next probe
        is the session's baseline."""
        self.probe(nt)
        self.base = None

    def probe(self, nt):
        t0 = time.perf_counter()
        _probe_chain(self.n, 1)
        serial = (time.perf_counter() - t0) * 1000
        with ff.num_threads(nt):
            t0 = time.perf_counter()
            _probe_all(self.n, nt)
            par = (time.perf_counter() - t0) * 1000
        settle_workers()
        if self.base is None:
            self.base = (serial, par)
        return {"threads": nt, "serial_ms": round(serial, 3),
                "parallel_ms": round(par, 3),
                "serial_rel": round(serial / self.base[0], 3),
                "parallel_rel": round(par / self.base[1], 3)}


# ---------------------------------------------------------------- scenes

def build_scenes(args):
    """[(name, kind, img, seed or None, connectivities)]; kind "one"
    (one blob, a seed) or "many" (labels)."""
    want = (lambda s: args.scenes is None or s in args.scenes)
    out = []
    if want("sq_256"):
        img, sx, sy = shared_scenes.square_scene(256, 256, 128, 128)
        out.append(("sq_256", "one", img, (sx, sy), (8, 4)))
    if want("sq_8000_center"):
        n = args.sq_size
        img, sx, sy = shared_scenes.square_scene(n, n, n, n)
        name = "sq_8000_center" if n == 8000 else f"sq_{n}_center"
        out.append((name, "one", img, (sx, sy), (8, 4)))
    if want("serpentine_256"):
        img, sx, sy = shared_scenes.serpentine_scene(256, 256)
        out.append(("serpentine_256", "one", img, (sx, sy), (8, 4)))
    if want("input_blobs"):
        md5 = _md5(PNG_PATH)
        if md5 != PNG_MD5:
            raise RuntimeError(f"{PNG_NAME}: md5 {md5}, expected {PNG_MD5}")
        img, _ = ch05_scenes.png_scene(PNG_PATH)
        name = "input_blobs"
        if args.crop:
            w, h = args.crop
            img = np.ascontiguousarray(img[:w, :h])
            name = f"input_blobs_crop_{w}x{h}"
        out.append((name, "many", img, None, (8,)))
    if want("dispatch"):
        # the fixed cost of a call (CPU only): checks, the thread count,
        # every njit dispatch, the stats, the forks
        one = np.full((1, 1, 3), 255, np.uint8)
        one[0, 0] = RED
        out.append(("dispatch_1x1", "one", one, (0, 0), (8,)))
        out.append(("dispatch_1x1_labels", "many", one.copy(), None, (8,)))
    return [(n, k, np.ascontiguousarray(im), sd, c)
            for n, k, im, sd, c in out]


def load_drones(args):
    if not (args.scenes is None or "drones" in args.scenes):
        return []
    d = args.drones_dir or MASK_DIR
    if not os.path.isdir(d):
        print("drones: no mask folder, skipped", flush=True)
        return []
    names = sorted(n for n in os.listdir(d)
                   if n.startswith("frame_") and n.endswith(".png"))
    frames = []
    for n in names[::args.drone_step]:
        p = os.path.join(d, n)
        frames.append({"key": n[:-4], "file": n, "sha1": _sha1(p),
                       "img": mask_image(p)})
    print(f"drones: {len(frames)} frames", flush=True)
    return frames


def reference(kind, img, seed, conns):
    """The oracle results the outputs are checked against (untimed)."""
    ref = {"red_px": int(np.count_nonzero(
        (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)))}
    if kind == "one":
        sx, sy = seed
        for conn in conns:
            fn = cpu_flood_fill_8 if conn == 8 else cpu_flood_fill
            v, _, lv, f = fn(img, sx, sy)
            ref[conn] = {"visited": v.astype(bool), "levels": int(lv),
                         "filled": int(f), "blue": blue_image(img, v)}
    else:
        lab, nb = cpu_label_components(img)
        ref.update(label=lab, n_blobs=int(nb),
                   palette=palette_image(img, lab))
    return ref


# ------------------------------------------------------------ CPU cells

def cpu_cells(name, kind, img, seed, conns, ref, args, work, one):
    """Every CPU cell of a scene: {key, fn (timed), prep (untimed),
    verify(full) (the output of the last call, untimed), meta}.

    verify(full=False) is the per-call check, run after EVERY timed call
    (off the clock, on one thread): the work image against the expected
    one (the painted image, or the pristine one for a cell that does
    not paint), the counts against the oracle's, and for ch09's labels
    the label map and the wave invariants. verify(full=True) adds the
    visited sets (the untimed call and the last timed call)."""
    pristine = img
    cells = []

    def restore():
        np.copyto(work, pristine)

    def cell(key, fn, verify, prep=None, **meta):
        def pre(prep=prep):
            restore()
            if prep is not None:
                prep()
        cells.append({"key": key, "fn": fn, "prep": pre, "verify": verify,
                      "meta": meta})

    if kind == "one":
        sx, sy = seed
        for conn in conns:
            r = ref[conn]
            c8 = conn == 8
            sfx = f"_c{conn}"
            for method, Eng in (("pixel", ff.PixelBFS), ("tile", ff.TileBFS)):
                # one engine per method and connectivity (the cells run one
                # at a time): its thread count is set before every call
                eng = Eng(img.shape[0], img.shape[1], connectivity=conn)
                for nt in args.threads:
                    for paint in (True, False):
                        if not paint and nt not in NOPAINT_THREADS:
                            continue
                        key = (f"ch09_{method}_t{nt}{sfx}" if paint
                               else f"ch09_{method}_fill_t{nt}{sfx}")
                        last = {}

                        def fn(eng=eng, paint=paint, last=last, nt=nt):
                            eng.threads = nt
                            last["st"] = eng.run(work, sx, sy, paint=paint)

                        def verify(full=True, eng=eng, paint=paint, r=r,
                                   last=last, method=method):
                            st = last["st"]
                            ok = {"image": same(work, r["blue"] if paint
                                                else pristine)}
                            if method == "pixel":
                                ok["filled"] = st["filled"] == r["filled"]
                                ok["levels"] = st["levels"] == r["levels"]
                            else:
                                # tile BFS counters do not depend on the
                                # threads: every call has the first's
                                ok["filled"] = eng.filled() == r["filled"]
                                last.setdefault("rounds", st["rounds"])
                                ok["rounds"] = st["rounds"] == last["rounds"]
                            if full:
                                ok["visited"] = bool(np.array_equal(
                                    eng.visited(), r["visited"]))
                            return {"ok": all(ok.values()), **ok,
                                    "work": st}
                        cell(key, fn, verify, variant=method, threads=nt,
                             paint=paint, connectivity=conn, side="ch09")
            for paint in (False, True):
                key = f"one_fill_paint{sfx}" if paint else f"one_fill{sfx}"
                last = {}

                def fn(paint=paint, c8=c8, last=last):
                    last["res"] = one.fill(work, sx, sy, c8, paint)

                def verify(full=True, paint=paint, r=r, last=last):
                    lv, f = last["res"]
                    ok = {"image": same(work, r["blue"] if paint
                                        else pristine),
                          "filled": int(f) == r["filled"],
                          "levels": int(lv) == r["levels"]}
                    if full:
                        ok["visited"] = bool(np.array_equal(
                            one.visited != 0, r["visited"]))
                    return {"ok": all(ok.values()), **ok}
                cell(key, fn, verify, variant="one_core", threads=1,
                     paint=paint, connectivity=conn, side="one_core",
                     reset="on the clock")
            last = {}

            def fn_bfs(c8=c8, last=last):
                last["res"] = one.bfs(work, sx, sy, c8)

            def verify_bfs(full=True, r=r, last=last):
                lv, f = last["res"]
                ok = {"image": same(work, pristine),
                      "filled": int(f) == r["filled"],
                      "levels": int(lv) == r["levels"]}
                if full:
                    ok["visited"] = bool(np.array_equal(one.visited != 0,
                                                        r["visited"]))
                return {"ok": all(ok.values()), **ok}
            cell(f"one_bfs{sfx}", fn_bfs, verify_bfs, prep=one.bfs_prep,
                 variant="one_core_depth", threads=1, paint=False,
                 connectivity=conn, side="one_core", reset="before the clock",
                 reference_only=True)
            last = {}
            oracle = cpu_flood_fill_8 if c8 else cpu_flood_fill

            def fn_or(oracle=oracle, last=last):
                last["res"] = oracle(work, sx, sy)

            def verify_or(full=True, r=r, last=last):
                v, _, lv, f = last["res"]
                ok = {"image": same(work, pristine),
                      "filled": int(f) == r["filled"],
                      "levels": int(lv) == r["levels"]}
                if full:
                    ok["visited"] = bool(np.array_equal(v != 0,
                                                        r["visited"]))
                return {"ok": all(ok.values()), **ok}
            cell(f"oracle_bfs{sfx}", fn_or, verify_or, variant="oracle",
                 threads=1, paint=False, connectivity=conn, side="oracle",
                 reference_only=True)
    else:
        variants = [(nt, True, 8) for nt in args.threads]
        variants += [(nt, False, 8) for nt in NOPAINT_THREADS
                     if nt in args.threads]
        variants.append((max(args.threads), True, 0))   # no lattice seeds
        engs = {}
        for nt, paint, lat in variants:
            if lat not in engs:
                engs[lat] = ff.TileLabels(img.shape[0], img.shape[1],
                                          lattice=lat)
            eng = engs[lat]
            if lat == 0:
                key = f"ch09_labels_L0_t{nt}"
            elif paint:
                key = f"ch09_labels_t{nt}"
            else:
                key = f"ch09_labels_nopaint_t{nt}"
            last = {}

            def fn(eng=eng, paint=paint, last=last, nt=nt):
                eng.threads = nt
                last["st"] = eng.run(work, paint=paint)

            def verify(full=True, eng=eng, paint=paint, last=last):
                st = last["st"]
                # seeds depend on the lattice only: every call has the
                # first's; every ring piece is expanded once
                last.setdefault("seeds", st["seeds"])
                ok = {"image": same(work, ref["palette"] if paint
                                    else pristine),
                      "labels": same(eng.label_map(), ref["label"]),
                      "n_blobs": st["n_blobs"] == ref["n_blobs"],
                      "seeds": st["seeds"] == last["seeds"],
                      "expanded": st["expanded"] == st["ring_pieces"],
                      "claims": st["claims"] == (st["ring_pieces"]
                                                 - st["seeds"]
                                                 + st["dropped"])}
                return {"ok": all(ok.values()), **ok, "work": st}
            cell(key, fn, verify, variant="labels", threads=nt, paint=paint,
                 lattice=lat, connectivity=8, side="ch09")
        for paint in (False, True):
            key = "one_label_paint" if paint else "one_label"
            last = {}

            def fn(paint=paint, last=last):
                last["n"] = one.labels(work, paint)

            def verify(full=True, paint=paint, last=last):
                ok = {"image": same(work, ref["palette"] if paint
                                    else pristine),
                      "labels": same(one.label, ref["label"]),
                      "n_blobs": int(last["n"]) == ref["n_blobs"]}
                return {"ok": all(ok.values()), **ok}
            cell(key, fn, verify, variant="one_core", threads=1, paint=paint,
                 connectivity=8, side="one_core", reset="on the clock")
        last = {}

        def fn_or(last=last):
            last["res"] = cpu_label_components(work)

        def verify_or(full=True, last=last):
            lab, nb = last["res"]
            ok = {"image": same(work, pristine),
                  "labels": same(lab, ref["label"]),
                  "n_blobs": int(nb) == ref["n_blobs"]}
            return {"ok": all(ok.values()), **ok}
        cell("oracle_label", fn_or, verify_or, variant="oracle", threads=1,
             paint=False, connectivity=8, side="oracle", reference_only=True)
    for c in cells:
        c["meta"]["cache"] = "hot"
    if img.nbytes <= COLD_MAX_BYTES and not name.startswith(CPU_ONLY_SCENES):
        cells += cold_cells(cells, kind, conns, args)
    return cells


def cold_cells(cells, kind, conns, args):
    """Cold twins of the headline cells (ch09 at 1 thread and at the most
    threads, painting, and the one-core like-for-like twin): the same
    call and checks, with the caches flushed before every call."""
    nts = sorted({1, max(args.threads)} & set(args.threads))
    if kind == "one":
        want = [f"ch09_{m}_t{nt}_c{c}" for c in conns for m in ("pixel",
                                                               "tile")
                for nt in nts] + [f"one_fill_paint_c{c}" for c in conns]
    else:
        want = [f"ch09_labels_t{nt}" for nt in nts] + ["one_label_paint"]
    out = []
    for c in cells:
        if c["key"] not in want:
            continue

        def prep(hot=c["prep"]):
            hot()
            flush_caches()
        out.append({"key": c["key"] + "_cold", "fn": c["fn"], "prep": prep,
                    "verify": c["verify"], "max_calls": COLD_MAX_CALLS,
                    "meta": {**c["meta"], "cache": "cold"}})
    return out


def check_call(c, where):
    """The per-call check of a timed call, off the clock and on one
    thread (a check must not wake the thread pool between calls)."""
    with ff.num_threads(1):
        ok = c["verify"](False)["ok"]
    c["calls_checked"] += 1
    if not ok:
        c["call_mismatches"] += 1
        c["mismatch_at"].append(where)


def settle(c):
    """Before a cell's batch: the previous cell's workers parked. A
    timeout is counted (batches_started_unsettled)."""
    if settle_workers()["settled"] is False:
        c["unsettled"] += 1


def cpu_pass(cells, args):
    """One untimed checked call per cell, a timed first call (it sets
    the calls per round), then the interleaved rounds. Every timed call
    is checked (check_call); the untimed call and the last timed call
    get the full check. Returns the rounds' records."""
    for c in cells:
        c["prep"]()
        c["fn"]()
        with ff.num_threads(1):
            c["check"] = c["verify"]()
        c["calls_checked"], c["call_mismatches"] = 0, 0
        c["mismatch_at"] = []
        c["unsettled"] = 0
        c["prep"]()
        settle(c)
        t0 = time.perf_counter()
        c["fn"]()
        first = time.perf_counter() - t0
        check_call(c, "first timed call")
        c["calls"] = max(1, min(args.max_calls,
                                c.get("max_calls", args.max_calls),
                                int(args.round_s / max(first, 1e-6))))
        c["times"], c["round_medians"] = [], []
    gc.collect()
    calib = _calibration(args)
    max_nt = max(c["meta"].get("threads", 1) for c in cells)
    threshold = gate_threshold(max_nt)
    for c in cells:
        c["others_pct_batches"] = []
    rounds = []
    for r in range(args.cpu_rounds):
        final = r == args.cpu_rounds - 1
        order = cells if r % 2 == 0 else cells[::-1]
        discarded = []
        for attempt in range(MAX_ROUND_REDOS + 1):
            gate = wait_quiet(QUIET_WAIT_CPU_S, args.gate, threshold)
            probe = calib.probe(max_nt)
            meter = LoadMeter("round")
            w0 = time.time()
            got = {}
            for c in order:
                ts = []
                c["prep"]()
                settle(c)
                bm = LoadMeter("batch")
                for i in range(c["calls"]):
                    if i:
                        c["prep"]()
                    t0 = time.perf_counter()
                    c["fn"]()
                    ts.append((time.perf_counter() - t0) * 1000)
                    check_call(c, f"round {r} attempt {attempt} call {i}")
                got[c["key"]] = (ts, bm.read()["others_pct"])
                if final:
                    with ff.num_threads(1):
                        c["check_last"] = c["verify"]()
            load = meter.read()
            rec = {"round": r, "attempt": attempt,
                   "seconds": round(time.time() - w0, 2),
                   "loadavg": list(os.getloadavg()),
                   "machine_busy_pct": load["machine_pct"],
                   "others_busy_pct": load["others_pct"],
                   "threshold_pct": threshold, "quiet_gate": gate,
                   "probe": probe}
            loaded = (load["others_pct"] is not None
                      and load["others_pct"] > threshold)
            print(f"    CPU round {r}.{attempt}: {rec['seconds']:.1f} s, "
                  f"others {load['others_pct']}% (machine "
                  f"{load['machine_pct']}%), probe x{probe['serial_rel']} / "
                  f"x{probe['parallel_rel']}"
                  f"{', LOADED' if loaded else ''}", flush=True)
            if loaded and args.gate and attempt < MAX_ROUND_REDOS:
                discarded.append(rec)
                continue
            break
        for c in cells:
            ts, others = got[c["key"]]
            c["times"] += ts
            c["round_medians"].append(statistics.median(ts))
            c["others_pct_batches"].append(others)
        rec["loaded"] = loaded
        rec["discarded_attempts"] = discarded
        rounds.append(rec)
    return rounds


def _calibration(args):
    """The session's calibration (its first probe is the baseline)."""
    if getattr(args, "calib", None) is None:
        args.calib = Calibration()
    return args.calib


def cpu_record(c):
    out = {**_stats(c["times"]), **c["meta"], "calls_per_round": c["calls"],
           "round_medians": c["round_medians"], "samples": c["times"]}
    chk, last = dict(c["check"]), dict(c["check_last"])
    out["work"] = chk.pop("work", None)
    last.pop("work", None)
    out["check"] = chk
    out["check_last_timed"] = last
    out["calls_checked"] = c["calls_checked"]
    out["call_mismatches"] = c["call_mismatches"]
    out["batches_started_unsettled"] = c["unsettled"]
    out["others_pct_batches"] = c["others_pct_batches"]
    out["mismatch_at"] = c["mismatch_at"][:20]
    out["checks_ok"] = bool(chk["ok"] and last["ok"]
                            and c["call_mismatches"] == 0)
    return out


# ------------------------------------------------------------------ GPU

class Sampler:
    """`nvidia-smi -lms` in the background (bench_v5's): (t, SM MHz,
    memory MHz, P-state, power W, clock event reasons)."""

    QUERY = ("timestamp,clocks.sm,clocks.mem,pstate,power.draw,"
             "clocks_event_reasons.active")

    def __init__(self, period_ms=SMI_PERIOD_MS):
        self.samples = []
        self.t0 = time.time()
        self.proc = None
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}",
                 "--format=csv,noheader,nounits", f"-lms={period_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
        except Exception:
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

    def window(self, t0, t1):
        if self.proc is None:
            return {"clk_samples": 0}
        t_end = time.time() + 5.0
        while time.time() < t_end and not (self.samples
                                            and self.samples[-1][0] > t1):
            time.sleep(0.05)
        pad = 0.0
        while True:
            s = [x for x in list(self.samples)
                 if t0 - pad <= x[0] <= t1 + pad]
            if s or pad >= 2.0:
                break
            pad += 0.25
        if not s:
            return {"clk_samples": 0}
        rs = 0
        for x in s:
            rs |= x[5] or 0
        pw = [x[4] for x in s if x[4] is not None]
        out = {"clk_samples": len(s), "clk_window_padding_s": pad,
               "clk_sm_min": min(x[1] for x in s),
               "clk_sm_median": statistics.median(x[1] for x in s),
               "clk_mem_min": min(x[2] for x in s),
               "clk_pstates": sorted({x[3] for x in s}),
               "clk_event_reasons": [n for b, n in EVENT_REASONS.items()
                                     if rs & b]}
        if pw:
            out["power_w_median"] = statistics.median(pw)
        return out

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


def _smi(query):
    """nvidia-smi --query-gpu=<query> for GPU 0: the fields as strings,
    or None (no nvidia-smi, an error)."""
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError):
        return None


def _num(x, kind=int):
    try:
        return kind(float(x)) if kind is int else kind(x)
    except (TypeError, ValueError):
        return None


def gpu_identity(gpu):
    """Which GPU and driver made the GPU numbers (saved in the session)."""
    out = {"device": gpu.device, "sm_count": gpu.sm_count,
           "compute_capability": ".".join(str(v) for v in gpu.cc)}
    v = _smi("driver_version,clocks.max.sm,clocks.max.mem")
    if v:
        out.update(driver=v[0], max_sm_mhz=_num(v[1]),
                   max_mem_mhz=_num(v[2]))
    return out


def gpu_clocks():
    """A snapshot of the GPU's clocks, P-state, temperature and power."""
    v = _smi("clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw")
    if not v:
        return {}
    return {"sm_mhz": _num(v[0]), "mem_mhz": _num(v[1]), "pstate": v[2],
            "temperature_c": _num(v[3]), "power_w": _num(v[4], float)}


class Gpu:
    """The GPU side: ch03's, ch07's and ch08's engines, the L2 flush, the
    gap."""

    def __init__(self):
        from numba import cuda
        from ...ch03_gpu_1blob_nblock import flood_fill as ff3
        from ...ch07_gpu_1blob_tiles import flood_fill as ff7
        from ...ch07_gpu_1blob_tiles.kernels import gpu_sleep_kernel
        from ...ch07_gpu_1blob_tiles.benchmarks import benchmark as b7
        from ...ch08_gpu_nblob_tiles import flood_fill as ff8
        self.cuda, self.ff3, self.ff7, self.ff8 = cuda, ff3, ff7, ff8
        self.b7 = b7
        self.sleep, self.l2 = gpu_sleep_kernel, b7._l2_flush
        self.flush = cuda.to_device(np.zeros(L2_FLUSH_BYTES // 4, np.uint32))
        self.sink = cuda.device_array(1, np.uint32)
        dev = cuda.get_current_device()
        name = dev.name.decode() if isinstance(dev.name, bytes) else str(
            dev.name)
        self.device = name.replace("\x00", "").strip()
        self.sm_count = int(dev.MULTIPROCESSOR_COUNT)
        self.cc = tuple(dev.compute_capability)
        self.mem_full_mhz = 8001
        mx = _smi("clocks.max.mem")
        if mx:
            try:
                self.mem_full_mhz = int(float(mx[0]))
            except ValueError:
                pass

    def gap(self):
        for _ in range(FLUSH_READS):
            self.l2[512, 256](self.flush, self.sink)
        self.cuda.synchronize()
        self.sleep[1, 32](GAP_NAPS)

    def filler(self, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:
            self.gap()
            self.cuda.synchronize()

    def ch03_launch(self, key):
        """ch03's kernel and grid for one of ch07's benchmark configs
        (bench_v5's: blocks None = the kernel's cooperative maximum)."""
        layout, conn, bare, tpb, blocks, _ = self.b7.CH03[key]
        cap = self.ff3.max_blocks(threads_per_block=tpb, bare=bare,
                                  connectivity=conn, probe_layout=layout)
        return {"kernel": self.b7._ch03_kernel(layout, conn, bare),
                "layout": layout, "conn": conn, "bare": bare, "tpb": tpb,
                "blocks": cap if blocks is None else blocks}

    def full_clocks(self, w):
        return bool(w.get("clk_samples") and w["clk_pstates"] == ["P0"]
                    and w["clk_mem_min"] >= self.mem_full_mhz
                    and w["clk_sm_median"] >= SM_MEDIAN_MIN
                    and w.get("clk_window_padding_s", 0.0) == 0.0)


class GpuCell:
    """One GPU config on one image: restore (off the clock), the gap,
    then the timed pipeline between CUDA events.

    ch03 (runner, launch): ch03's pixel BFS on ch07's benchmark runner.
    Its timed span is the visited reset, the seed and the kernel (the
    painted image needs no more; the depth, owner and stats resets run
    between them, off the clock), like ch07's and ch09's on-clock
    resets; last_kernel_ms keeps the kernel alone (bench_v5's clock).
    The ch03 cells of a scene share one runner: its counters belong to
    the last ch03 run, so they are checked right after a cell's own
    untimed run; each cell's painted image (its own work image) is
    checked after its last timed run too."""

    def __init__(self, gpu, key, img, seed=None, conn=8, ch08_kw=None,
                 ch03=None):
        cuda = gpu.cuda
        self.gpu, self.key, self.seed, self.conn = gpu, key, seed, conn
        self.w, self.h = img.shape[0], img.shape[1]
        self.d_pristine = cuda.to_device(img)
        self.d_img = cuda.device_array_like(img)
        self.last_kernel_ms = None
        if ch03 is not None:
            self.runner, self.launch = ch03
            L = self.launch
            self.conn = L["conn"]
            self.meta = {"chapter": "ch03", "connectivity": L["conn"],
                         "layout": L["layout"], "bare": L["bare"],
                         "tpb": L["tpb"], "blocks": L["blocks"],
                         "timed": "visited reset + seed + kernel"}
        elif ch08_kw is None:
            cfg = dict(connectivity=conn, threads_per_block=128, bare=True,
                       **{k: gpu.ff7.DEFAULT_SWITCHES[k]
                          for k in gpu.ff7.SWITCHES})
            self.eng = gpu.ff7.TileFill(self.w, self.h, **cfg)
            self.meta = {"chapter": "ch07", "connectivity": conn,
                         "engine": "TileFill default switches, bare",
                         "tpb": 128, "blocks": self.eng.blocks}
        else:
            self.eng = gpu.ff8.TileWaves(self.w, self.h, **ch08_kw)
            self.meta = {"chapter": "ch08", "connectivity": 8,
                         "contract": "rgb",
                         "engine": ("first version (flood_fill.OLD_PATH)"
                                    if ch08_kw else "default engine")}

    def run_timed(self):
        cuda = self.gpu.cuda
        self.d_img.copy_to_device(self.d_pristine)
        self.gpu.gap()
        ev = self.enqueue()
        ev[-1].synchronize()
        if self.meta["chapter"] == "ch03":
            self.last_kernel_ms = cuda.event_elapsed_time(ev[3], ev[4])
            return cuda.event_elapsed_time(ev[0], ev[1]) + self.last_kernel_ms
        if self.meta["chapter"] == "ch07":
            return self.gpu.ff7.phase_times(ev)[1]
        return cuda.event_elapsed_time(ev[0], ev[1])

    def enqueue(self):
        """The pipeline on d_img, enqueued (not waited for). Returns its
        events: ch03 reset (3) + kernel (2), ch07 its own, ch08 two."""
        cuda = self.gpu.cuda
        if self.meta["chapter"] == "ch03":
            ev = self.runner.reset(*self.seed)
            L = self.launch
            return list(ev) + list(self.runner.run(
                L["kernel"], L["bare"], L["blocks"], L["tpb"], self.d_img))
        if self.meta["chapter"] == "ch07":
            return self.eng.run(self.d_img, *self.seed)[1]
        stages = self.eng.stages(self.d_img, "rgb")
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        for _, go in stages:
            go()
        e1.record()
        return [e0, e1]

    def e2e(self, img, expected, repeats=E2E_REPEATS):
        """Context, perf_counter: end to end (a pageable host-to-GPU copy
        of img, the pipeline, the copy back: bench_v5's e2e) and the
        host wall time (the image already on the GPU: launch to
        synchronize, no gap). Reused buffers."""
        cuda = self.gpu.cuda
        out = np.empty_like(img)
        e2e, host = [], []
        for _ in range(repeats):
            cuda.synchronize()
            t0 = time.perf_counter()
            self.d_img.copy_to_device(img)
            self.enqueue()
            self.d_img.copy_to_host(out)
            e2e.append((time.perf_counter() - t0) * 1000)
        ok = same(out, expected)
        for _ in range(repeats):
            self.d_img.copy_to_device(self.d_pristine)
            cuda.synchronize()
            t0 = time.perf_counter()
            self.enqueue()
            cuda.synchronize()
            host.append((time.perf_counter() - t0) * 1000)
        return {**_stats(e2e), "samples": e2e, "matches_cpu": ok,
                "host_wall": _stats(host)}

    def verify(self, ref, counters=True):
        out = self.d_img.copy_to_host()
        if self.meta["chapter"] == "ch03":
            r = ref[self.conn]
            ok = {"image": same(out, r["blue"])}
            info = {}
            if counters:
                info = self.runner.read()
                ok["filled"] = info["filled"] == r["filled"]
        elif self.meta["chapter"] == "ch07":
            r = ref[self.conn]
            ok = {"image": bool(np.array_equal(out, r["blue"])),
                  "filled": self.eng.filled() == r["filled"]}
            info = {"rounds": self.eng.read_counters()["rounds"]}
        else:
            lab = self.gpu.cuda.device_array((self.w, self.h), np.int32)
            self.eng.emit_label_map(lab)
            c = self.eng.read_counters()
            ok = {"image": bool(np.array_equal(out, ref["palette"])),
                  "labels": bool(np.array_equal(lab.copy_to_host(),
                                                ref["label"])),
                  "n_blobs": c["n_blobs"] == ref["n_blobs"]}
            info = {"rounds": c["rounds"], "n_blobs": c["n_blobs"]}
            del lab
        return {"ok": all(ok.values()), **ok, "info": info}


# ch03 cells per connectivity: the same idea as ch09's pixel BFS
# (bench_v5's twins), and the uninstrumented warp kernel as a reference
CH03_CELLS = {4: ("ch03_conn4",), 8: ("ch03_conn8", "ch03_wc_bare")}


def timed_block(gpu, sampler, label, run_round, warm_round, args):
    """bench_v5's attempts: a ramp, then rounds spanning MIN_TIMED_S,
    again (with light load between runs) until full clocks."""
    attempts, n_rounds, data = [], None, None
    for a in range(args.max_attempts):
        spacing = SPACING_S[min(a, len(SPACING_S) - 1)]
        gate = wait_quiet(QUIET_WAIT_GPU_S, args.gate)
        t0 = time.time()
        n_w = 0
        while True:
            warm_round(spacing)
            n_w += 1
            if time.time() - t0 >= args.warm_seconds:
                break
        if n_rounds is None:
            per = (time.time() - t0) / max(n_w, 1)
            n_rounds = max(args.gpu_rounds,
                           min(MAX_ROUNDS, math.ceil(MIN_TIMED_S / per)))
        t1 = time.time()
        data = [run_round(i, spacing) for i in range(n_rounds)]
        t2 = time.time()
        w = sampler.window(t1, t2)
        ok = gpu.full_clocks(w)
        attempts.append({"attempt": a, "spacing_s": spacing,
                         "rounds": n_rounds, "quiet_gate": gate, **w,
                         "full_clocks": ok})
        print(f"    {label} attempt {a}: {n_rounds} rounds, sm "
              f"{w.get('clk_sm_min', '?')}-{w.get('clk_sm_median', '?')}, "
              f"mem {w.get('clk_mem_min', '?')}, {w.get('clk_pstates')}: "
              f"{'full clocks' if ok else 'NOT full clocks'}", flush=True)
        if ok:
            break
    return data, attempts


def gpu_scene(gpu, sampler, name, kind, img, seed, conns, ref, args):
    """The GPU cells of one scene, timed as one block."""
    if kind == "one":
        cells = [GpuCell(gpu, f"ch07_c{c}", img, seed, conn=c) for c in conns]
        keys3 = [k for c in conns for k in CH03_CELLS[c]]
        launches = {k: gpu.ch03_launch(k) for k in keys3}
        runner = gpu.b7.Ch03Runner(img.shape[0], img.shape[1],
                                   max(L["blocks"] for L in launches.values()))
        cells += [GpuCell(gpu, k, img, seed, ch03=(runner, launches[k]))
                  for k in keys3]
    else:
        cells = [GpuCell(gpu, "ch08_old", img, ch08_kw=dict(gpu.ff8.OLD_PATH)),
                 GpuCell(gpu, "ch08", img, ch08_kw={})]
    checks = {}
    for c in cells:
        c.run_timed()
        checks[c.key] = c.verify(ref)

    def run_round(i, spacing):
        out = {}
        for c in (cells if i % 2 == 0 else cells[::-1]):
            out[c.key] = c.run_timed()
            if c.last_kernel_ms is not None:
                out[c.key + ":kernel"] = c.last_kernel_ms
            if spacing:
                gpu.filler(spacing)
        return out

    def warm_round(spacing):
        run_round(0, spacing)

    data, attempts = timed_block(gpu, sampler, name, run_round, warm_round,
                                 args)
    res = {}
    for c in cells:
        samples = [r[c.key] for r in data]
        # the last timed run's output (ch03: its image; the shared
        # runner's counters were checked after its own untimed run)
        last = c.verify(ref, counters=c.meta["chapter"] != "ch03")
        res[c.key] = {**_stats(samples), "samples": samples, **c.meta,
                      "side": "gpu", "check": checks[c.key],
                      "check_last_timed": last,
                      "checks_ok": bool(checks[c.key]["ok"] and last["ok"]),
                      "full_clocks": attempts[-1]["full_clocks"]}
        if c.last_kernel_ms is not None:
            res[c.key]["kernel_only"] = _stats(
                [r[c.key + ":kernel"] for r in data])
    for c in cells:
        # context, after the timed block (its own last output checked)
        exp = (ref[c.conn]["blue"] if kind == "one" else ref["palette"])
        res[c.key]["e2e"] = c.e2e(img, exp)
        res[c.key]["checks_ok"] &= res[c.key]["e2e"]["matches_cpu"]
    block = {"attempts": attempts}
    del cells
    gpu.cuda.current_context().deallocations.clear()
    return res, block


def gpu_drones(gpu, sampler, frames, args):
    """ch08_old and ch08 on every frame: rounds over all frames."""
    w, h = frames[0]["img"].shape[:2]
    cells = [GpuCell(gpu, "ch08_old", frames[0]["img"],
                     ch08_kw=dict(gpu.ff8.OLD_PATH)),
             GpuCell(gpu, "ch08", frames[0]["img"], ch08_kw={})]
    pristine = [gpu.cuda.to_device(f["img"]) for f in frames]
    ok_all = True
    for f, dp in zip(frames, pristine):
        for c in cells:
            c.d_pristine = dp
            c.run_timed()
            chk = c.verify(f["ref"])
            f.setdefault("gpu_check", {})[c.key] = chk["ok"]
            ok_all &= chk["ok"]
    idx = list(range(len(frames)))

    def run_round(i, spacing):
        out = {c.key: [None] * len(frames) for c in cells}
        for j in (idx if i % 2 == 0 else idx[::-1]):
            for c in cells:
                c.d_pristine = pristine[j]
                out[c.key][j] = c.run_timed()
        if spacing:
            gpu.filler(spacing)
        return out

    def warm_round(spacing):
        run_round(0, spacing)

    data, attempts = timed_block(gpu, sampler, "drones", run_round,
                                 warm_round, args)
    last_j = idx[-1] if (len(data) - 1) % 2 == 0 else idx[0]
    last_ok = all(c.verify(frames[last_j]["ref"])["ok"] for c in cells)
    # context, after the timed block: end to end and host wall per frame
    e2e = {c.key: [] for c in cells}
    host = {c.key: [] for c in cells}
    for j, f in enumerate(frames):
        for c in cells:
            c.d_pristine = pristine[j]
            r = c.e2e(f["img"], f["ref"]["palette"])
            e2e[c.key].append(r["ms"])
            host[c.key].append(r["host_wall"]["ms"])
            ok_all &= r["matches_cpu"]
    res = {}
    for c in cells:
        per = [statistics.median(r[c.key][j] for r in data) for j in idx]
        res[c.key] = {"per_frame_median_ms": per,
                      "median_of_frame_medians_ms": statistics.median(per),
                      "max_frame_median_ms": max(per),
                      "per_frame_samples_ms": [[r[c.key][j] for r in data]
                                               for j in idx],
                      "per_frame_e2e_median_ms": e2e[c.key],
                      "per_frame_host_wall_median_ms": host[c.key],
                      **c.meta, "side": "gpu",
                      "checks_ok": bool(ok_all and last_ok),
                      "full_clocks": attempts[-1]["full_clocks"]}
    del cells, pristine
    gpu.cuda.current_context().deallocations.clear()
    return res, {"attempts": attempts}


# --------------------------------------------------------------- drones CPU

def cpu_drones(frames, args):
    """Per frame: the one-core twins and ch09's labels at every thread
    count, rounds over all frames (order reversed on odd rounds). The
    headline keys also run cold (_cold: the caches flushed first)."""
    w, h = frames[0]["img"].shape[:2]
    work = np.empty_like(frames[0]["img"])
    one = OneCore(w, h, "many")
    eng = ff.TileLabels(w, h)
    engs = {f"ch09_labels_t{nt}": nt for nt in args.threads}
    keys = list(engs) + ["one_label", "one_label_paint"]
    if work.nbytes <= COLD_MAX_BYTES:
        keys += [f"ch09_labels_t{max(args.threads)}_cold",
                 "one_label_paint_cold"]

    state = {"prev_nt": 1, "unsettled": 0}

    def call(key, f):
        cold = key.endswith("_cold")
        base = key[:-len("_cold")] if cold else key
        np.copyto(work, f["img"])
        nt = engs.get(base, 1)
        if cold:
            flush_caches()                 # it settles the pool too
            state["prev_nt"] = 1
        if nt < state["prev_nt"]:
            # the workers of a call with more threads park first
            if settle_workers()["settled"] is False:
                state["unsettled"] += 1
        state["prev_nt"] = nt
        if base in engs:
            eng.threads = nt
        t0 = time.perf_counter()
        if base in engs:
            st = eng.run(work)
            n = st["n_blobs"]
        else:
            n = one.labels(work, base == "one_label_paint")
        ms = (time.perf_counter() - t0) * 1000
        with ff.num_threads(1):            # checks never wake the pool
            ok = int(n) == f["ref"]["n_blobs"]
            ok &= same(work, f["img"] if base == "one_label"
                       else f["ref"]["palette"])
            if base in engs:
                ok &= same(eng.label_map(), f["ref"]["label"])
                ok &= st["expanded"] == st["ring_pieces"]
                ok &= st["claims"] == (st["ring_pieces"] - st["seeds"]
                                       + st["dropped"])
            else:
                ok &= same(one.label, f["ref"]["label"])
        return ms, bool(ok)

    ok_all = True
    for f in frames:                      # untimed check pass
        for k in keys:
            ok_all &= call(k, f)[1]
    samples = {k: [[] for _ in frames] for k in keys}
    rounds = []
    idx = list(range(len(frames)))
    calib = _calibration(args)
    max_nt = max(args.threads)
    threshold = gate_threshold(max_nt)
    for r in range(args.cpu_rounds):
        discarded = []
        for attempt in range(MAX_ROUND_REDOS + 1):
            gate = wait_quiet(QUIET_WAIT_CPU_S, args.gate, threshold)
            probe = calib.probe(max_nt)
            state["prev_nt"] = max_nt     # the probe used every thread
            meter = LoadMeter("round")
            w0 = time.time()
            got = {k: [None] * len(frames) for k in keys}
            for j in (idx if r % 2 == 0 else idx[::-1]):
                for k in (keys if r % 2 == 0 else keys[::-1]):
                    ms, ok = call(k, frames[j])
                    got[k][j] = ms
                    ok_all &= ok
            load = meter.read()
            loaded = (load["others_pct"] is not None
                      and load["others_pct"] > threshold)
            rec = {"round": r, "attempt": attempt,
                   "seconds": round(time.time() - w0, 2),
                   "loadavg": list(os.getloadavg()),
                   "machine_busy_pct": load["machine_pct"],
                   "others_busy_pct": load["others_pct"],
                   "threshold_pct": threshold, "quiet_gate": gate,
                   "probe": probe,
                   "calls_started_unsettled": state["unsettled"]}
            state["unsettled"] = 0
            print(f"    drones round {r}.{attempt}: {rec['seconds']:.1f} s, "
                  f"others {load['others_pct']}%"
                  f"{', LOADED' if loaded else ''}", flush=True)
            if loaded and args.gate and attempt < MAX_ROUND_REDOS:
                discarded.append(rec)
                continue
            break
        for k in keys:
            for j in idx:
                samples[k][j].append(got[k][j])
        rec["loaded"] = loaded
        rec["discarded_attempts"] = discarded
        rounds.append(rec)
    out = {}
    for k in keys:
        per = [statistics.median(s) for s in samples[k]]
        base = k[:-len("_cold")] if k.endswith("_cold") else k
        out[k] = {"per_frame_median_ms": per,
                  "median_of_frame_medians_ms": statistics.median(per),
                  "max_frame_median_ms": max(per),
                  "per_frame_samples_ms": samples[k],
                  "side": "ch09" if base in engs else "one_core",
                  "variant": "labels" if base in engs else "one_core",
                  "threads": engs.get(base, 1),
                  "paint": base != "one_label",
                  "cache": "cold" if k.endswith("_cold") else "hot"}
    return out, rounds, bool(ok_all)


# ---------------------------------------------------------------- summary

def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _paired(a, b):
    """The median over rounds of a's round median over b's."""
    return statistics.median(x / y for x, y in zip(a["round_medians"],
                                                   b["round_medians"]))


def summarize(row, kind, threads):
    """Scaling and the like-for-like factors of one scene (and one
    connectivity for one-blob scenes). Per variant: every thread count
    (speedup over 1 thread, per-vCPU efficiency, the factor over the
    one-core twin, over the GPU), the no-paint cells, and the steps:
    one core's twin -> ch09 on one thread (x_method) -> ch09 on every
    thread (x_cores) -> the GPU with the same idea (x_gpu)."""
    cells = row["cells"]
    out = {}
    variants = ((("pixel", "ch09_pixel_t{}", "ch09_pixel_fill_t{}"),
                 ("tile", "ch09_tile_t{}", "ch09_tile_fill_t{}"))
                if kind == "one" else
                (("labels", "ch09_labels_t{}", "ch09_labels_nopaint_t{}"),))
    for conn in row["connectivities"]:
        sfx = f"_c{conn}" if kind == "one" else ""
        twin = cells.get(TWIN_FULL[kind] + sfx)
        twin_np = cells.get(TWIN_NOPAINT[kind] + sfx)
        gpus = {k: g for k, g in row["gpu"].items()
                if g.get("connectivity", 8) == conn}
        for name, fmt, fmt_np in variants:
            base = cells.get(fmt.format(1) + sfx)
            gkey = SAME_IDEA_GPU[name].format(conn=conn)
            gpu = gpus.get(gkey)
            sc = {}
            for nt in threads:
                c = cells.get(fmt.format(nt) + sfx)
                if c is None:
                    continue
                e = {"ms": c["ms"]}
                if base:
                    rr = [a / b for a, b in zip(base["round_medians"],
                                                c["round_medians"])]
                    e.update(speedup_vs_t1=statistics.median(rr),
                             speedup_range=[min(rr), max(rr)],
                             # per vCPU: WSL's 16 logical CPUs are not 16
                             # equal cores (6 P-cores with 2 threads each
                             # and 8 E-cores, scheduled by Windows)
                             per_vcpu_efficiency=statistics.median(rr) / nt)
                if twin:
                    rr = [a / b for a, b in zip(twin["round_medians"],
                                                c["round_medians"])]
                    e.update(x_vs_one_core=statistics.median(rr),
                             x_vs_one_core_range=[min(rr), max(rr)])
                if gpu:
                    # the same idea on the GPU (its cell named)
                    e.update(gpu_cell=gkey, gpu_x=c["ms"] / gpu["ms"],
                             gpu_x_range_cpu_rounds=[
                                 min(c["round_medians"]) / gpu["ms"],
                                 max(c["round_medians"]) / gpu["ms"]])
                    if "e2e" in gpu:
                        # the data starts in host memory: the GPU's copies
                        # in and out, and its launch costs, count too
                        e.update(gpu_x_e2e=c["ms"] / gpu["e2e"]["ms"],
                                 gpu_x_host_wall=(
                                     c["ms"] / gpu["e2e"]["host_wall"]["ms"]))
                e["cache"] = "hot"
                cold = cells.get(fmt.format(nt) + sfx + "_cold")
                twin_cold = cells.get(TWIN_FULL[kind] + sfx + "_cold")
                if cold:
                    # caches flushed before every call, like the GPU's
                    # L2 before every timed launch
                    co = {"ms": cold["ms"]}
                    if twin_cold:
                        rr = [a / b for a, b in zip(twin_cold["round_medians"],
                                                    cold["round_medians"])]
                        co.update(x_vs_one_core=statistics.median(rr),
                                  x_vs_one_core_range=[min(rr), max(rr)])
                    if gpu:
                        co.update(gpu_cell=gkey,
                                  gpu_x=cold["ms"] / gpu["ms"])
                    e["cold"] = co
                other = {k: c["ms"] / g["ms"] for k, g in gpus.items()
                         if k != gkey}
                if other:
                    # against the other GPU cells (another idea, or the
                    # GPU's best): kept apart from gpu_x
                    e["gpu_x_other"] = other
                sc[nt] = e
            for nt in NOPAINT_THREADS:
                c = cells.get(fmt_np.format(nt) + sfx)
                if c is not None and twin_np:
                    rr = [a / b for a, b in zip(twin_np["round_medians"],
                                                c["round_medians"])]
                    sc[f"no_paint_t{nt}"] = {
                        "ms": c["ms"], "x_vs_one_core_no_paint":
                            statistics.median(rr)}
            top = cells.get(fmt.format(max(threads)) + sfx)
            if base and top and twin:
                # one core's twin -> ch09 on one thread (the method) ->
                # ch09 on every thread (the cores) -> the GPU (same idea)
                st = {"one_core_ms": twin["ms"],
                      "x_method": _paired(twin, base),
                      "ch09_t1_ms": base["ms"],
                      "x_cores": _paired(base, top),
                      "cores": max(threads), "ch09_ms": top["ms"]}
                if gpu:
                    st.update(gpu_cell=gkey, x_gpu=top["ms"] / gpu["ms"],
                              gpu_ms=gpu["ms"])
                sc["steps"] = st
            out[f"{name}{sfx}"] = sc
        if kind == "many":
            nt = max(threads)
            l0, l8 = cells.get(f"ch09_labels_L0_t{nt}"), cells.get(
                f"ch09_labels_t{nt}")
            if l0 and l8:
                rr = [a / b for a, b in zip(l0["round_medians"],
                                            l8["round_medians"])]
                out["lattice_0_over_8"] = {
                    "threads": nt, "ratio": statistics.median(rr),
                    "range": [min(rr), max(rr)],
                    "rounds": [l0["work"]["rounds"], l8["work"]["rounds"]],
                    "meetings": [l0["work"]["meetings"],
                                 l8["work"]["meetings"]]}
        if gpus and twin:
            out[f"gpu{sfx}"] = {
                "one_core_like_for_like": twin["ms"],
                "x_gpu_vs_one_core": {k: twin["ms"] / g["ms"]
                                      for k, g in gpus.items()},
                "gpu_ms": {k: g["ms"] for k, g in gpus.items()}}
    return out


def _spread(vals):
    """Median, p90 (nearest rank), min, max and count of per-frame
    values."""
    v = sorted(vals)
    return {"median": statistics.median(v),
            "p90": v[max(0, math.ceil(0.9 * len(v)) - 1)],
            "min": v[0], "max": v[-1], "n": len(v)}


def summarize_drones(drones):
    """Per drone frame, then over the frames (median, p90, min, max):
    each CPU key against the one-core like-for-like twin (ch09 keys,
    paired per round), against ch08_old (the same idea) by events and
    end to end, and against the other GPU cells, kept apart."""
    cpu, gpu = drones.get("cpu") or {}, drones.get("gpu") or {}
    same_gpu = gpu.get(SAME_IDEA_GPU["labels"])
    out = {}
    for k, c in cpu.items():
        e = {"threads": c.get("threads"), "cache": c.get("cache", "hot")}
        if c.get("side") == "ch09":
            twin = cpu.get(TWIN_FULL["many"] + ("_cold" if k.endswith("_cold")
                                                else ""))
            if twin:
                e["x_vs_one_core"] = _spread([
                    statistics.median(a / b for a, b in zip(ts, cs))
                    for ts, cs in zip(twin["per_frame_samples_ms"],
                                      c["per_frame_samples_ms"])])
        per = c["per_frame_median_ms"]
        if same_gpu:
            e["gpu_cell"] = SAME_IDEA_GPU["labels"]
            e["gpu_x"] = _spread([a / b for a, b in zip(
                per, same_gpu["per_frame_median_ms"])])
            if same_gpu.get("per_frame_e2e_median_ms"):
                e["gpu_x_e2e"] = _spread([a / b for a, b in zip(
                    per, same_gpu["per_frame_e2e_median_ms"])])
        other = {g: _spread([a / b for a, b in zip(
                     per, gc["per_frame_median_ms"])])
                 for g, gc in gpu.items() if g != SAME_IDEA_GPU["labels"]}
        if other:
            e["gpu_x_other"] = other
        out[k] = e
    return out


CSV_FIELDS = ["scene", "cell", "side", "variant", "threads", "connectivity",
              "paint", "cache", "ms", "ms_min", "ms_max", "n", "checks_ok"]


def write_csv(path, rows, drones):
    """One line per cell. Drone lines: ms is the median of the per-frame
    medians, ms_min and ms_max the least and the largest per-frame
    median, n the number of frames."""
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(CSV_FIELDS)
        for r in rows:
            for k, c in list(r["cells"].items()) + list(r["gpu"].items()):
                wr.writerow([r["scene"], k, c.get("side"), c.get("variant"),
                             c.get("threads"), c.get("connectivity"),
                             c.get("paint"), c.get("cache"), c["ms"],
                             c["ms_min"], c["ms_max"], c["n"],
                             c["checks_ok"]])
        for side in ("cpu", "gpu"):
            for k, c in ((drones or {}).get(side) or {}).items():
                per = c["per_frame_median_ms"]
                ok = (drones.get("cpu_checks_ok") if side == "cpu"
                      else c.get("checks_ok"))
                wr.writerow(["drones", k, c.get("side"), c.get("variant"),
                             c.get("threads"), 8, c.get("paint"),
                             c.get("cache"), statistics.median(per),
                             min(per), max(per), len(per), ok])


# ------------------------------------------------------------------ main

def warm_up(args):
    """Compile and start the thread pool: every engine and twin at every
    thread count on small scenes."""
    img, sx, sy = shared_scenes.square_scene(200, 150, 90, 70)
    many, _ = ch05_scenes.random_blobs_scene(300, 200, 0.3, 1)
    for conn in (4, 8):
        for nt in args.threads:
            for Eng in (ff.PixelBFS, ff.TileBFS):
                e = Eng(200, 150, connectivity=conn, threads=nt)
                for paint in (True, False):
                    e.run(img.copy(), sx, sy, paint=paint)
                e.visited()
    for nt in args.threads:
        e = ff.TileLabels(300, 200, threads=nt)
        e.run(many.copy())
        e.label_map()
    one = OneCore(200, 150, "one")
    for c8 in (True, False):
        one.fill(img.copy(), sx, sy, c8, True)
        one.bfs_prep()
        one.bfs(img.copy(), sx, sy, c8)
    onel = OneCore(300, 200, "many")
    onel.labels(many.copy(), True)
    cpu_flood_fill(img, sx, sy)
    cpu_flood_fill_8(img, sx, sy)
    cpu_label_components(many)
    same(img, img)
    same(onel.label, onel.label)
    _probe_chain(10, 1)
    _read_serial(np.ones(8, np.uint64))
    _read_parallel(np.ones(8, np.uint64))
    for nt in args.threads:
        with ff.num_threads(nt):
            _probe_all(10, nt)


def run_gpu_blocks(rows, scenes, frames, args):
    """Every GPU block (scenes, then the drones), then the cooldown.
    The nvidia-smi sampler lives only as long as the GPU blocks: it is
    stopped before the cooldown, so it never runs beside a CPU pass.
    Fills rows[i]["gpu"]; returns (gpu_blocks, drones, the GPU's
    identity and its clocks at the start and the end)."""
    gpu_blocks, drones = {}, None
    gpu = Gpu()
    info = {**gpu_identity(gpu), "clocks_start": gpu_clocks()}
    sampler = Sampler()
    try:
        print(f"GPU: {gpu.device} ({gpu.sm_count} SMs)", flush=True)
        for row, (name, kind, img, seed, conns) in zip(rows, scenes):
            if name.startswith(CPU_ONLY_SCENES):
                continue
            print(f"  GPU {name}", flush=True)
            ref = reference(kind, img, seed, conns)
            row["gpu"], gpu_blocks[name] = gpu_scene(
                gpu, sampler, name, kind, img, seed, conns, ref, args)
            for k, c in row["gpu"].items():
                print(f"    {k}: {c['ms']:.4f} ms, checks "
                      f"{'OK' if c['checks_ok'] else 'MISMATCH'}", flush=True)
            del ref
            gc.collect()
        if frames:
            print("  GPU drones", flush=True)
            dg, gpu_blocks["drones"] = gpu_drones(gpu, sampler, frames, args)
            drones = {"gpu": dg}
    finally:
        sampler.stop()
    info["clocks_end"] = gpu_clocks()
    if args.cooldown:
        print(f"Cooldown {args.cooldown:.0f} s before the CPU passes",
              flush=True)
        time.sleep(args.cooldown)
    return gpu_blocks, drones, info


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="only these (sq_256 sq_8000_center serpentine_256 "
                         "input_blobs drones)")
    ap.add_argument("--threads", nargs="*", type=int, default=list(THREADS))
    ap.add_argument("--sq-size", type=int, default=8000)
    ap.add_argument("--crop", nargs=2, type=int, default=None,
                    help="label a W x H crop of input_blobs instead")
    ap.add_argument("--drones-dir", default=None)
    ap.add_argument("--drone-step", type=int, default=1)
    ap.add_argument("--cpu-rounds", type=int, default=CPU_ROUNDS)
    ap.add_argument("--round-s", type=float, default=CPU_ROUND_S)
    ap.add_argument("--max-calls", type=int, default=CPU_MAX_CALLS)
    ap.add_argument("--gpu-rounds", type=int, default=GPU_ROUNDS)
    ap.add_argument("--warm-seconds", type=float, default=WARM_SECONDS)
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    ap.add_argument("--cooldown", type=float, default=CPU_COOLDOWN_S)
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="small sizes, short rounds, no gate (a harness "
                         "check: its timings mean nothing)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threading-layer", default="omp",
                    help="pinned before the first parallel call (the "
                         "default layer would switch to TBB if a newer TBB "
                         "appeared)")
    ap.add_argument("--allow-nice", action="store_true",
                    help="run timed even at nice > 0 or with a CPU affinity "
                         "narrower than the thread pool (recorded as a "
                         "note)")
    ap.add_argument("--allow-openmp-env", action="store_true",
                    help="run timed even with OMP_WAIT_POLICY, "
                         "GOMP_SPINCOUNT, OMP_PROC_BIND ... set (recorded)")
    return ap


def parse_args(argv=None):
    """The command line, with --smoke's small sizes applied."""
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.sq_size = min(args.sq_size, 1000)
        args.crop = args.crop or (1024, 1024)
        args.drone_step = max(args.drone_step, 60)
        args.threads = [t for t in args.threads if t in (1, 2, 16)]
        args.cpu_rounds, args.round_s, args.max_calls = 1, 0.02, 3
        args.gpu_rounds, args.warm_seconds, args.max_attempts = 2, 0.3, 1
        args.cooldown, args.gate = 0.0, False
    return args


def select_threads(threads, limit):
    """The thread counts to run, sorted, each once. Stops the run (before
    anything is warmed up) if one is below 1 or above `limit`, the
    threads Numba can use: NUMBA_NUM_THREADS is fixed when the process
    starts, from its CPU affinity mask (or the variable itself), so a
    narrower mask would otherwise drop the 16-thread cells silently."""
    want = sorted(set(int(t) for t in threads))
    bad = [t for t in want if not 1 <= t <= limit]
    if bad:
        raise SystemExit(
            f"threads {bad} are not available: Numba can use 1-{limit} "
            f"threads in this process (NUMBA_NUM_THREADS, set at start-up "
            f"from the CPU affinity mask; check taskset and the "
            f"NUMBA_NUM_THREADS variable). Nothing was run.")
    return want


def main(argv=None):
    args = parse_args(argv)
    args.threads = select_threads(args.threads,
                                  numba.config.NUMBA_NUM_THREADS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"ch09 session {stamp}: threads {args.threads}"
          f"{' (SMOKE: timings mean nothing)' if args.smoke else ''}",
          flush=True)

    runtime_notes = check_runtime(args, runtime_info())
    for n in runtime_notes:
        print(f"NOTE: {n}", flush=True)
    numba.config.THREADING_LAYER = args.threading_layer
    scenes = build_scenes(args)
    frames = load_drones(args)
    print("Warming up (JIT, thread pool)...", flush=True)
    t_w = time.perf_counter()
    warm_up(args)
    _calibration(args).warm(max(args.threads))
    runtime = runtime_info()
    if runtime["threading_layer"] != args.threading_layer:
        raise SystemExit(f"threading layer {runtime['threading_layer']}, "
                         f"asked for {args.threading_layer}")
    print(f"  done in {time.perf_counter() - t_w:.1f} s, threading layer "
          f"{runtime['threading_layer']}, libs {runtime['runtime_libs']}",
          flush=True)

    rows = []
    for name, kind, img, seed, conns in scenes:
        rows.append({"scene": name, "kind": kind,
                     "shape": [int(img.shape[0]), int(img.shape[1])],
                     "seed": None if seed is None else [int(v) for v in seed],
                     "connectivities": list(conns), "cells": {}, "gpu": {}})
    for f in frames:
        lab, nb = cpu_label_components(f["img"])
        f["ref"] = {"label": lab, "n_blobs": int(nb),
                    "palette": palette_image(f["img"], lab)}

    # -- GPU blocks first
    gpu_blocks, drones, gpu_info = {}, None, None
    if not args.no_gpu:
        gpu_blocks, drones, gpu_info = run_gpu_blocks(rows, scenes, frames,
                                                      args)

    # -- CPU passes, scene by scene
    cpu_rounds = {}
    for row, (name, kind, img, seed, conns) in zip(rows, scenes):
        print(f"  CPU {name}", flush=True)
        ref = reference(kind, img, seed, conns)
        work = np.empty_like(img)
        one = OneCore(img.shape[0], img.shape[1], kind)
        cells = cpu_cells(name, kind, img, seed, conns, ref, args, work, one)
        cpu_rounds[name] = cpu_pass(cells, args)
        for c in cells:
            row["cells"][c["key"]] = cpu_record(c)
        row["all_checks_ok"] = all(c["checks_ok"] for c in
                                   list(row["cells"].values())
                                   + list(row["gpu"].values()))
        row["summary"] = summarize(row, kind, args.threads)
        for c in cells:
            rc = row["cells"][c["key"]]
            print(f"    {c['key']:26s} {rc['ms']:10.3f} ms  "
                  f"{'OK' if rc['checks_ok'] else 'MISMATCH'}", flush=True)
        del cells, one, work, ref
        gc.collect()
    if frames:
        print("  CPU drones", flush=True)
        dc, dr, dok = cpu_drones(frames, args)
        drones = drones or {}
        drones.update(cpu=dc, cpu_rounds=dr, cpu_checks_ok=dok,
                      frames=[{"key": f["key"], "sha1": f["sha1"],
                               "n_blobs": f["ref"]["n_blobs"]}
                              for f in frames])
        drones["summary"] = summarize_drones(drones)
        print(f"    checks {'OK' if dok else 'MISMATCH'}", flush=True)

    ok = all(r["all_checks_ok"] for r in rows) and (
        drones is None or (drones.get("cpu_checks_ok", True) and all(
            c["checks_ok"] for c in drones.get("gpu", {}).values())))
    print(f"\nAll checked outputs {'match' if ok else 'DO NOT match'} the "
          "oracles (every CPU call; GPU: the untimed and last timed runs)",
          flush=True)
    session = {
        "session": f"ch09_{stamp}", "smoke": bool(args.smoke),
        "host": _host_info(),
        "threading_layer": numba.threading_layer(),
        "runtime": {**runtime,
                    "requested_threading_layer": args.threading_layer,
                    "notes": runtime_notes},
        "gpu": gpu_info,
        "git": {"commit": _git("rev-parse", "--short", "HEAD"),
                "dirty": bool(_git("status", "--porcelain"))},
        "settings": {
            "threads": args.threads, "cpu_rounds": args.cpu_rounds,
            "round_s": args.round_s, "max_calls": args.max_calls,
            "gpu_rounds_min": args.gpu_rounds, "gate": args.gate,
            "quiet_busy_pct": QUIET_BUSY_PCT, "cooldown_s": args.cooldown,
            "pixel_serial_below": ff.PIXEL_SERIAL_BELOW,
            "pixel_min_chunk": ff.PIXEL_MIN_CHUNK,
            "tile_serial_below": ff.TILE_SERIAL_BELOW,
            "label_serial_below": ff.LABEL_SERIAL_BELOW,
            "lattice": ff.DEFAULT_LATTICE, "tile": K.TILE,
            "png": PNG_NAME, "png_md5": PNG_MD5},
        "rows": rows, "cpu_rounds": cpu_rounds, "gpu_blocks": gpu_blocks,
        "drones": drones, "all_checks_ok": bool(ok)}
    if args.no_save:
        return
    out_dir = args.out_dir or results_paths.results_dir(CHAPTER,
                                                        "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, f"ch09_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump(session, f, indent=1, default=_json_default)
    cpath = os.path.join(out_dir, f"ch09_{stamp}.csv")
    write_csv(cpath, rows, drones)
    print(f"wrote {os.path.basename(jpath)} and {os.path.basename(cpath)}",
          flush=True)


if __name__ == "__main__":
    main()
