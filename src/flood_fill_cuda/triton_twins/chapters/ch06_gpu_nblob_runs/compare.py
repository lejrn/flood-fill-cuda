"""Numba vs Triton on chapter 6's own benchmarks.

Mirrors the chapter's two benchmark scripts, with identical configs on
both sides: DEFAULT_GRID (256 threads per block = 8 warps per program),
PHASE_BLOCKS, PACK_ROW_BLOCKS, the single 1024-lane scan, and the run
capacity sized from the host run count exactly as the benchmark sizes
it (max(8192, runs * 1.05)).

  benchmark      benchmarks/benchmark.py: input_blobs.png, input_blocks.png
                 and the four synthetic scenes (blob_grid_100, random_4000,
                 disk_r2000, serpentine_2048), each in both contracts
  scaling        benchmarks/scaling.py: centered crops of input_blobs.png,
                 1000 to 9000 px a side, both contracts
  lane_schedule  the twin's two spellings of merge and flatten against the
                 same Numba pipeline, two rows per cell (config.lane_sched):
                 "independent" (label lane_independent, the default every
                 other experiment runs) and "lockstep" (label
                 first_translation: the lockstep loops of the first
                 translation). The cells are the benchmark scenes in the
                 "mask" contract, where merge and flatten weigh most (the
                 two contracts run the same merge and flatten).
                 first_translation rows are comparable=False and carry
                 first_translation=true (as in ch01-ch05): they measure the
                 first translation's cost and stay out of the averages.
                 lane_independent rows repeat a benchmark cell and carry
                 duplicate_of="benchmark" (when that experiment runs).

The twin runs its default lane schedule ("independent", see kernels.py)
everywhere except the lane_schedule experiment's first_translation rows.

One case is one scene in one contract (and, in lane_schedule, one
schedule). Each backend runs the pipeline
the way _bench_ch06 does: restore the pristine image (device to
device), pack off the clock for the "mask" contract, synchronize,
run(), synchronize. Each scene is packed before its clock spin, as
benchmark.py packs before _spin_up.

What the numbers mean:

  kernel_ms     the CUDA-event span from before the first launch to
                after the last: the chapter's headline number. The
                events sit on the stream, so when the HOST is the
                bottleneck (small scenes: 7 launches at ~25 us each in
                Triton, ~57 us in Numba) the GPU waits for every launch
                and the span INCLUDES the Python launch enqueue. Only
                when the GPU is the bottleneck does the span hide it.
  total_ms      host wall time of run() + synchronize: the span plus
                the host work before the first event executes and the
                sync return.
  enqueue_ms    host time for run() to return (all launches queued).
  gpu_fraction  gpu_only kernel_ms / kernel_ms: the share of the span
                that is GPU work.
  launch_bound  per backend: enqueue_ms >= 0.8 * kernel_ms or
                gpu_fraction < 0.8, i.e. a fifth or more of the span is
                the GPU waiting for the host. On such rows
                speedup_kernel and speedup_phase compare Python launch
                paths as much as kernels.
  gpu_only      the same pipeline run once more per call with every
                launch queued behind a QUEUE_US device spin, so the
                event span is GPU work only (a run counts only if its
                first event had not executed when run() returned).
                speedup_gpu_kernel / speedup_gpu_phase compare kernels.

Per-phase medians (and label_only_ms, every phase but paint) are added
to each row from the same runs, with model_gb_s per backend, the
floor arithmetic (floor_ms) at each backend's own read/write peaks, and,
for the scaling sweep, the 1.0 / 0.5 ms crossings per backend
(meta["scaling_crossings"], scaling._crossings unchanged).

Outputs are compared on the device after every run: the counters, the
painted image, the packed mask, row offsets, the run table and labels up
to N_RUNS_USED, and the root set. Everything else about the method (the
backend order flipping every round, warm-ups, clocks, versions) is
compare/harness.py's.

Not mirrored, and recorded in meta["caps"]: the ch05 head-to-head
column (a Numba-only baseline outside this comparison, about 55 ms a
run on 81 Mpx), figures.py and visualize.py (no kernels of their own;
figures' one GPU call is recolor()), and overview/bench_ch06.py (the
overview phase).

Run:
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare [--quick] [--repeats N] [--experiments a,b]

--quick is a smoke test: small scenes, one round, no JSON. --experiments
runs a subset (default: all three).
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import gc
import json
import resource
import statistics
import time
from functools import lru_cache
from types import SimpleNamespace

import cupy as cp
import numpy as np
import triton
import triton.language as tl
from numba import cuda

from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock import scenes as _scenes
from flood_fill_cuda.chapters.ch06_gpu_nblob_runs import recolor as nb_recolor
from flood_fill_cuda.chapters.ch06_gpu_nblob_runs import kernels as nb_kernels
from flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks import (
    benchmark as nb_benchmark, scaling as nb_scaling,
)
from flood_fill_cuda.shared.bandwidth import (
    measure_peak_bandwidth as numba_copy_peak,
)
from flood_fill_cuda.shared.results_paths import results_dir
from flood_fill_cuda.triton_twins.compare.harness import Case, run_cases
from flood_fill_cuda.triton_twins.runtime import kernel_resources, sync, t
from flood_fill_cuda.triton_twins.runtime.bandwidth import (
    measure_peak_bandwidth as triton_copy_peak,
)
from flood_fill_cuda.triton_twins.runtime.device import read_globaltimer

from . import recolor as tr_recolor
from .kernels import (
    N_BLOBS, N_RUNS, N_RUNS_USED, RUN_OVERFLOW, SCAN_TPB, UNION_ATTEMPTS,
    UNION_DONE,
)

CHAPTER = "ch06_gpu_nblob_runs"
EXPERIMENTS = ("benchmark", "scaling", "lane_schedule")
ROUNDS = nb_benchmark.ROUNDS            # 9, as in benchmark.py and scaling.py
SCENE_SPIN_SECONDS = 3.0                # per scene, both backends alternating
PEAK_BYTES = 256 * 2 ** 20
QUEUE_US = 3000                         # device spin ahead of a GPU-only run
LAUNCH_BOUND = 0.8                      # enqueue_ms / kernel_ms threshold
DEFAULT_SCHED = tr_recolor.DEFAULT_LANE_SCHEDULE
LANE_LABELS = {"independent": "lane_independent",
               "lockstep": "first_translation"}
LANE_CONTRACTS = ("mask",)


# ------------------------------------------------------- GPU-only timing

@triton.jit(do_not_specialize=["ns"])
def _queue_kernel(ns):
    """Hold the stream for `ns` nanoseconds: one program spinning on
    %globaltimer. Everything enqueued behind it on the (legacy, shared)
    null stream waits, so a run() enqueued meanwhile is fully queued
    before its first event executes."""
    t0 = read_globaltimer(tl.program_id(0))
    now = t0
    while now - t0 < ns:
        now = read_globaltimer(tl.program_id(0))


def _queue(us):
    _queue_kernel[(1,)](us * 1000, num_warps=1)


# ------------------------------------------------ read / write peak probes
# Triton twins of benchmark._measure_read_write_peaks' two kernels: the
# same [2048 x 256] grid, grid-stride loops, an int64 accumulator, and an
# UNCONDITIONAL per-lane sink store so the reads cannot be optimised away.

@triton.jit(do_not_specialize=["n"])
def _read_kernel(src_ptr, sink_ptr, n, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int64)
    for base in range(tl.program_id(0) * BLOCK, n, tl.num_programs(0) * BLOCK):
        offs = base + lane
        acc += tl.load(src_ptr + offs, mask=offs < n, other=0).to(tl.int64)
    tl.store(sink_ptr + tl.program_id(0) * BLOCK + lane, acc)


@triton.jit(do_not_specialize=["n"])
def _write_kernel(dst_ptr, v, n, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    val = tl.full([BLOCK], v, tl.uint32)
    for base in range(tl.program_id(0) * BLOCK, n, tl.num_programs(0) * BLOCK):
        offs = base + lane
        tl.store(dst_ptr + offs, val, mask=offs < n)


def measure_read_write_peaks_triton(nbytes, repeats=9):
    """(read_gb_s, write_gb_s) over `nbytes`, median of `repeats` timed
    launches after one warm-up, CUDA-event timing: the Numba probe's
    method, in Triton."""
    n = nbytes // 4
    buf = cp.empty(n, dtype=cp.uint32)
    out = cp.empty(2048 * 256, dtype=cp.int64)

    def _time(launch):
        launch()
        sync()
        ts = []
        for _ in range(repeats):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            launch()
            e1.record()
            e1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(e0, e1))
        return statistics.median(ts)

    r_ms = _time(lambda: _read_kernel[(2048,)](t(buf), t(out), n, BLOCK=256,
                                               num_warps=8))
    w_ms = _time(lambda: _write_kernel[(2048,)](t(buf), 7, n, BLOCK=256,
                                                num_warps=8))
    del buf, out
    cp.get_default_memory_pool().free_all_blocks()
    return nbytes / (r_ms * 1e6), nbytes / (w_ms * 1e6)


# ----------------------------------------------------------- the two sides

_VIEWS = ("mask", "row_count", "row_off", "run_x", "run_y0", "run_y1",
          "parent", "run_label")


class _NumbaSide:
    """The Numba engine on one scene, run as _bench_ch06 runs it."""

    name = "numba"

    def __init__(self, img, capacity):
        self.engine = nb_recolor.RunRecolor(img.shape[0], img.shape[1],
                                            run_capacity=capacity)
        self.pristine = cuda.to_device(img)
        self.dev = cuda.to_device(img)
        # zero-copy CuPy views, for comparing outputs on the device
        self.views = {k: cp.asarray(getattr(self.engine, k)) for k in _VIEWS}
        self.views["img"] = cp.asarray(self.dev)

    def restore(self):
        self.dev.copy_to_device(self.pristine)

    def pack(self):
        self.engine.pack(self.dev)

    def run(self, contract):
        return self.engine.run(self.dev, contract=contract)

    sync = staticmethod(cuda.synchronize)
    elapsed = staticmethod(cuda.event_elapsed_time)

    @staticmethod
    def done(event):
        return event.query()

    def counters(self):
        return self.engine.counters.copy_to_host()

    def resources(self):
        regs = {}
        for name, k in (("pack", nb_kernels.pack_kernel),
                        ("count", nb_kernels.count_kernel),
                        ("scan", nb_kernels.row_scan_kernel),
                        ("emit", nb_kernels.emit_kernel),
                        ("merge", nb_kernels.merge_rows_kernel),
                        ("flatten", nb_kernels.flatten_kernel),
                        ("paint", nb_kernels.paint_kernel)):
            try:
                r = k.get_regs_per_thread()
                regs[name] = (int(r) if not isinstance(r, dict)
                              else max(int(v) for v in r.values()))
            except Exception as exc:  # recorded, not hidden
                regs[name] = f"{type(exc).__name__}: {exc}"
        return {"registers_per_thread": regs}


class _TritonSide:
    """The Triton twin's engine on the same scene, run the same way, in
    one lane schedule. img: the host image, or another side's pristine
    device copy (copied, never shared)."""

    name = "triton"

    def __init__(self, img, capacity, lane_schedule=DEFAULT_SCHED):
        self.lane_schedule = lane_schedule
        self.engine = tr_recolor.RunRecolor(img.shape[0], img.shape[1],
                                            run_capacity=capacity,
                                            lane_schedule=lane_schedule)
        self.pristine = cp.array(img)
        self.dev = cp.array(img)
        self.views = {k: getattr(self.engine, k) for k in _VIEWS}
        self.views["img"] = self.dev

    def restore(self):
        cp.copyto(self.dev, self.pristine)

    def pack(self):
        self.engine.pack(self.dev)

    def run(self, contract):
        return self.engine.run(self.dev, contract=contract)

    sync = staticmethod(sync)
    elapsed = staticmethod(cp.cuda.get_elapsed_time)

    @staticmethod
    def done(event):
        return event.done

    def counters(self):
        return self.engine.counters.get()

    def resources(self):
        return {name: kernel_resources(k)
                for name, k in self.engine.compiled.items()}


def _prepare(side, contract):
    side.restore()
    if contract == "mask":
        side.pack()                     # the packed input, off the clock
    side.sync()


def _phases(side, names, events):
    return {n: side.elapsed(events[i], events[i + 1])
            for i, n in enumerate(names)}


def _gpu_only_run(side, contract):
    """The pipeline with every launch queued behind a QUEUE_US device
    spin: its event span is GPU work only. `valid` is False when the
    first event had already executed by the time run() returned (the
    host was slower than the spin), and then the run is not used."""
    _prepare(side, contract)
    _queue(QUEUE_US)
    names, events = side.run(contract)
    valid = not side.done(events[0])
    side.sync()
    return (valid, side.elapsed(events[0], events[-1]),
            _phases(side, names, events), side.counters())


def _measure(side, contract, samples):
    """One call of the harness: a GPU-only run (recorded in `samples`),
    then the timed run the harness reads (kernel_ms, total_ms), whose
    outputs same() compares."""
    q_valid, q_kernel, q_phase, q_counters = _gpu_only_run(side, contract)
    _prepare(side, contract)
    t0 = time.perf_counter()
    names, events = side.run(contract)
    t_enq = time.perf_counter()
    side.sync()
    t1 = time.perf_counter()
    counters = side.counters()
    if counters[RUN_OVERFLOW]:
        raise RuntimeError("run table overflowed during benchmark")
    if not np.array_equal(counters, q_counters):
        raise RuntimeError(f"GPU-only run counters {q_counters.tolist()} != "
                           f"timed run counters {counters.tolist()}")
    phase_ms = _phases(side, names, events)
    kernel_ms = side.elapsed(events[0], events[-1])
    samples.append({"phase_ms": phase_ms, "kernel_ms": kernel_ms,
                    "enqueue_ms": (t_enq - t0) * 1000,
                    "gpu_valid": q_valid, "gpu_kernel_ms": q_kernel,
                    "gpu_phase_ms": q_phase})
    return SimpleNamespace(kernel_ms=kernel_ms, total_ms=(t1 - t0) * 1000,
                           phase_ms=phase_ms, counters=counters, side=side)


def _same(rn, rt):
    """Deterministic outputs only: never parent[] of non-root runs."""
    if not np.array_equal(rn.counters, rt.counters):
        return False, (f"counters: {rn.counters.tolist()} vs "
                       f"{rt.counters.tolist()}")
    n = int(rn.counters[N_RUNS_USED])
    a, b = rn.side.views, rt.side.views
    for k in ("img", "mask", "row_count", "row_off"):
        if not bool(cp.array_equal(a[k], b[k])):
            return False, f"{k} differs"
    for k in ("run_x", "run_y0", "run_y1", "run_label"):
        if not bool(cp.array_equal(a[k][:n], b[k][:n])):
            return False, f"{k}[:{n}] differs"
    ids = cp.arange(n, dtype=cp.int32)
    if not bool(cp.array_equal(a["parent"][:n] == ids,
                               b["parent"][:n] == ids)):
        return False, "root set differs"
    return True, ""


# ------------------------------------------------------------------ scenes

@lru_cache(maxsize=1)
def _png(path):
    """The decoded PNG, one at a time: input_blobs.png (243 MB) stays
    cached while its crops run, and is dropped for any other PNG."""
    img, _ = _scenes.png_scene(path)
    img.setflags(write=False)
    return img


def _png_path(name):
    return next((p for p in nb_benchmark.PNG_INPUTS
                 if os.path.basename(p) == name and os.path.exists(p)), None)


def _scene_list(quick):
    """(experiment, scene, builder, note) in run order. Builders are
    lazy: one scene is resident at a time."""
    out = []
    blobs = _png_path("input_blobs.png")
    blocks = _png_path("input_blocks.png")
    if quick:
        if blocks:
            out.append(("benchmark", "input_blocks", lambda: _png(blocks),
                        "external PNG, 1000x1000"))
        out += [
            ("benchmark", "blob_grid_100_small",
             lambda: _scenes.blob_grid_scene(400, 400, 10, 10, 30, gap=4)[0],
             "quick stand-in for blob_grid_100"),
            ("benchmark", "random_300",
             lambda: _scenes.random_blobs_scene(300, 300, 0.30, 0)[0],
             "quick stand-in for random_4000"),
            ("benchmark", "disk_r200",
             lambda: _scenes.disk_scene(420, 420, 200)[0],
             "quick stand-in for disk_r2000"),
            ("benchmark", "serpentine_256",
             lambda: _scenes.serpentine_scene(256, 256)[0],
             "quick stand-in for serpentine_2048"),
        ]
        src = (lambda: _png(blocks)) if blocks else (
            lambda: _scenes.random_blobs_scene(1000, 1000, 0.2, 1)[0])
        for side in (250, 500):
            out.append(("scaling", f"crop_{side}",
                        lambda s=side: nb_scaling._crop(src(), s),
                        "quick: centered crop of input_blocks.png"))
        return out

    for path in (blobs, blocks):
        if path:
            name = os.path.splitext(os.path.basename(path))[0]
            out.append(("benchmark", name, lambda p=path: _png(p),
                        "external PNG"))
    for name, build, note in nb_benchmark._synthetic_scenes():
        out.append(("benchmark", name, build, note))
    if blobs:
        for side in nb_scaling.SIDES:
            out.append(("scaling", f"crop_{side}",
                        lambda s=side: nb_scaling._crop(_png(blobs), s),
                        "centered crop of input_blobs.png"))
    return out


class _Pair:
    """Both backends on the current scene. Built at the first call for a
    scene, released when the next scene starts. The twin in a lane
    schedule other than the default is built at its first use
    (triton(sched))."""

    def __init__(self, spin_seconds):
        self.key = None
        self.cur = None
        self.spin_seconds = spin_seconds

    def release(self):
        self.key = None
        self.cur = None
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cuda.current_context().deallocations.clear()

    def get(self, key, build):
        if self.key != key:
            self.release()
            img = build()
            capacity = max(8192, int(nb_benchmark._host_run_count(img)
                                     * 1.05))
            red_px = int(np.count_nonzero(
                (img[..., 0] == 255) & (img[..., 1] == 0)
                & (img[..., 2] == 0)))
            self.cur = SimpleNamespace(
                width=img.shape[0], height=img.shape[1], red_px=red_px,
                capacity=capacity, numba=_NumbaSide(img, capacity),
                triton=_TritonSide(img, capacity), other_scheds={})
            del img
            self.key = key
            # Pack before the spin, as benchmark._scene_row does: the
            # spin runs the "mask" contract, and an unpacked mask is
            # uninitialised memory (garbage runs, padding bits set).
            for side in (self.cur.numba, self.cur.triton):
                side.restore()
                side.pack()
                side.sync()
            self._spin()
        return self.cur

    def triton(self, sched):
        """The twin side in lane schedule `sched` on the current scene."""
        if sched == DEFAULT_SCHED:
            return self.cur.triton
        side = self.cur.other_scheds.get(sched)
        if side is None:
            side = _TritonSide(self.cur.triton.pristine, self.cur.capacity,
                               lane_schedule=sched)
            side.restore()
            side.pack()
            side.sync()
            self.cur.other_scheds[sched] = side
        return side

    def _spin(self):
        """The chapter's clock rule, per scene: back-to-back pipeline
        runs with no host sync inside a batch, both backends."""
        end = time.perf_counter() + self.spin_seconds
        nbs, trs = self.cur.numba, self.cur.triton
        while time.perf_counter() < end:
            for _ in range(10):
                nbs.run("mask")
                trs.run("mask")
            sync()


def _phase_medians(dicts):
    phase = {k: statistics.median(d[k] for d in dicts) for k in dicts[0]}
    return phase, sum(v for k, v in phase.items() if k != "paint")


def _summary(samples):
    """Per-backend medians over the timed calls: phases, enqueue, and
    the GPU-only runs that were valid."""
    if not samples:
        return None
    med = statistics.median
    phase, label_only = _phase_medians([s["phase_ms"] for s in samples])
    span = med(s["kernel_ms"] for s in samples)
    enqueue = med(s["enqueue_ms"] for s in samples)
    out = {"phase_ms": phase, "label_only_ms": label_only,
           "enqueue_ms": enqueue}
    ok = [s for s in samples if s["gpu_valid"]]
    gpu = {"valid_runs": len(ok), "runs": len(samples)}
    launch_bound = enqueue >= LAUNCH_BOUND * span
    if ok:
        g_phase, g_label = _phase_medians([s["gpu_phase_ms"] for s in ok])
        gpu.update(kernel_ms=med(s["gpu_kernel_ms"] for s in ok),
                   phase_ms=g_phase, label_only_ms=g_label)
        out["gpu_fraction"] = gpu["kernel_ms"] / span
        launch_bound = launch_bound or out["gpu_fraction"] < LAUNCH_BOUND
    out["launch_bound"] = launch_bound
    out["gpu_only"] = gpu
    return out


def _ratio(a, b):
    return a / b if a is not None and b else None


def _finish_row(row, smp, peaks):
    """Everything compare.py adds to a harness row (see the module doc)."""
    info = row.get("info")
    if info:
        row["pixels"] = info["width"] * info["height"]
    if "error" in row:
        return
    for backend in ("numba", "triton"):
        s = _summary(smp[backend][1:])          # sample 0 is the warm-up
        if s:
            row[backend].update(s)
        b = row[backend]
        b["model_gb_s"] = info["model_bytes"] / (
            b["kernel_ms"]["median"] * 1e6)
        g = b.get("gpu_only", {})
        if g.get("kernel_ms"):
            g["model_gb_s"] = info["model_bytes"] / (g["kernel_ms"] * 1e6)
    nb, tr = row["numba"], row["triton"]
    row["launch_bound"] = bool(nb.get("launch_bound")
                               or tr.get("launch_bound"))
    if nb.get("phase_ms") and tr.get("phase_ms"):
        row["speedup_phase"] = {k: _ratio(nb["phase_ms"][k], tr["phase_ms"][k])
                                for k in nb["phase_ms"]}
    gn, gt = nb.get("gpu_only", {}), tr.get("gpu_only", {})
    if gn.get("kernel_ms") and gt.get("kernel_ms"):
        row["speedup_gpu_kernel"] = gn["kernel_ms"] / gt["kernel_ms"]
        row["speedup_gpu_phase"] = {
            k: _ratio(gn["phase_ms"][k], gt["phase_ms"][k])
            for k in gn["phase_ms"]}
    # benchmark._scene_row's floor arithmetic, at each backend's peaks
    n, red_px = row["pixels"], info["red_px"]
    row["floor_ms"] = {
        backend: {
            "rgb_read": n * 3 / (peaks[backend]["read_gb_s"] * 1e6),
            "mask_read": ((n + 7) // 8) / (peaks[backend]["read_gb_s"] * 1e6),
            "paint_write": red_px * 3 / (peaks[backend]["write_gb_s"] * 1e6),
        } for backend in ("numba", "triton")}


def _scaling_crossings(rows):
    """scaling.py's 1.0 / 0.5 ms crossings (its own _crossings, linear
    in megapixels between bracketing crops), per backend, from the event
    span medians (the chapter's number) and from the GPU-only medians."""
    by_scene = {}
    for row in rows:
        if row["experiment"] == "scaling" and "error" not in row:
            by_scene.setdefault(row["scene"], {})[
                row["config"]["contract"]] = row
    tiers = {
        "event_span": lambda b: (b["kernel_ms"]["median"],
                                 b.get("label_only_ms")),
        "gpu_only": lambda b: (b.get("gpu_only", {}).get("kernel_ms"),
                               b.get("gpu_only", {}).get("label_only_ms")),
    }
    out = {"targets_ms": list(nb_scaling.TARGETS)}
    for tier, get in tiers.items():
        out[tier] = {}
        for backend in ("numba", "triton"):
            pts = []
            for by_c in by_scene.values():
                if set(by_c) != set(nb_recolor.CONTRACTS):
                    continue
                entry = {"n_pixels": by_c["rgb"]["pixels"]}
                for c in nb_recolor.CONTRACTS:
                    ms, label_ms = get(by_c[c][backend])
                    if ms is None or label_ms is None:
                        break
                    entry[c] = {"median_ms": ms, "label_only_ms": label_ms}
                else:
                    pts.append(entry)
            if pts:
                out[tier][backend] = {
                    "rgb": nb_scaling._crossings(pts, "rgb", "median_ms"),
                    "mask": nb_scaling._crossings(pts, "mask", "median_ms"),
                    "label_only": nb_scaling._crossings(pts, "mask",
                                                        "label_only_ms"),
                }
    return out


def _info_fn(pair, contract, sched):
    """The case's info(): the scene's facts from the warm-up results, the
    grids, and both backends' kernel resources (the twin's in `sched`)."""
    def info(rn, rt):
        cur = pair.cur
        tside = pair.triton(sched)
        c = rn.counters
        n_px = cur.width * cur.height
        tpb = tside.engine.grid[1]
        return {
            "width": cur.width, "height": cur.height,
            "red_px": cur.red_px, "n_runs": int(c[N_RUNS]),
            "n_blobs": int(c[N_BLOBS]),
            "union_attempts": int(c[UNION_ATTEMPTS]),
            "union_done": int(c[UNION_DONE]),
            "mean_run_px": (cur.red_px / int(c[N_RUNS])
                            if c[N_RUNS] else 0.0),
            "run_capacity": cur.capacity,
            "model_bytes": int(nb_recolor.model_bytes_ch06(
                contract, n_px, cur.red_px, int(c[N_RUNS]),
                int(c[UNION_ATTEMPTS]))),
            "lane_schedule": tside.lane_schedule,
            "grid": {
                "threads_per_block": tpb,
                "num_warps": tpb // 32,
                "pack": {"numba": list(cur.numba.engine.pack_grid[0]),
                         "triton": list(tside.engine.pack_grid[0])},
                "phase_blocks": {
                    "numba": {k: v[0] for k, v in
                              cur.numba.engine.phase_grid.items()},
                    "triton": {k: v[0] for k, v in
                               tside.engine.phase_grid.items()}},
                "scan": {"numba": [1, nb_kernels.SCAN_TPB],
                         "triton": [1, SCAN_TPB]},
            },
            "numba_resources": cur.numba.resources(),
            "triton_kernel_resources": tside.resources(),
        }
    return info


def _case(pair, experiment, scene, build, note, contract, sched, config):
    key = (experiment, scene)
    smp = {"numba": [], "triton": []}

    def run_numba():
        return _measure(pair.get(key, build).numba, contract, smp["numba"])

    def run_triton():
        pair.get(key, build)
        return _measure(pair.triton(sched), contract, smp["triton"])

    extra = {}
    if sched != DEFAULT_SCHED:
        extra["first_translation"] = True
    return Case(experiment=experiment, scene=scene, config=config,
                run_numba=run_numba, run_triton=run_triton, same=_same,
                pixels=0, info=_info_fn(pair, contract, sched), notes=note,
                extra=extra, comparable=sched == DEFAULT_SCHED), smp


def _cell_key(case):
    cfg = {k: v for k, v in case.config.items()
           if k not in ("lane_sched", "label")}
    return case.scene, json.dumps(cfg, sort_keys=True)


def mark_repeated_cells(earlier, lane_cases):
    """Tag each lane_independent row whose cell (scene and config, the
    schedule keys aside) an earlier experiment already measures with
    duplicate_of=<that experiment>, so a unit-wide average counts each
    cell once. Returns the tagged count."""
    seen = {}
    for c in earlier:
        seen.setdefault(_cell_key(c), c.experiment)
    tagged = 0
    for c in lane_cases:
        if c.config["lane_sched"] == DEFAULT_SCHED:
            hit = seen.get(_cell_key(c))
            if hit is not None:
                c.extra["duplicate_of"] = hit
                tagged += 1
    return tagged


def build_cases(quick, spin_seconds, experiments=EXPERIMENTS):
    """(cases, samples, pair, lane_meta) in experiment then scene order.
    lane_schedule takes the benchmark experiment's scenes, the default
    schedule's row first."""
    pair = _Pair(spin_seconds)
    cases, samples, lane = [], [], []
    base = {"tpb": nb_recolor.DEFAULT_GRID[1],
            "grid": list(nb_recolor.DEFAULT_GRID)}
    scene_list = _scene_list(quick)
    for experiment, scene, build, note in scene_list:
        if experiment not in experiments:
            continue
        for contract in nb_recolor.CONTRACTS:
            case, smp = _case(pair, experiment, scene, build, note, contract,
                              DEFAULT_SCHED, {"contract": contract, **base})
            cases.append(case)
            samples.append(smp)
    if "lane_schedule" in experiments:
        order = (DEFAULT_SCHED,) + tuple(
            x for x in tr_recolor.LANE_SCHEDULES if x != DEFAULT_SCHED)
        for experiment, scene, build, note in scene_list:
            if experiment != "benchmark":
                continue
            for contract in LANE_CONTRACTS:
                for sched in order:
                    config = {"contract": contract, **base,
                              "lane_sched": sched,
                              "label": LANE_LABELS[sched]}
                    case, smp = _case(pair, "lane_schedule", scene, build,
                                      note, contract, sched, config)
                    lane.append(case)
                    samples.append(smp)
    tagged = mark_repeated_cells(cases, lane)
    lane_meta = None
    if "lane_schedule" in experiments:
        lane_meta = {
            "schedules": list(tr_recolor.LANE_SCHEDULES),
            "default": DEFAULT_SCHED,
            "contracts": list(LANE_CONTRACTS),
            "cells": [c.scene for c in lane
                      if c.config["lane_sched"] == DEFAULT_SCHED],
            "duplicate_rows": tagged}
    return cases + lane, samples, pair, lane_meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="small scenes, 1 round, no JSON (smoke test)")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per case (default {ROUNDS})")
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS),
                    help=f"comma-separated subset of {EXPERIMENTS}")
    args = ap.parse_args(argv)
    experiments = tuple(e for e in args.experiments.split(",") if e)
    bad = [e for e in experiments if e not in EXPERIMENTS]
    if bad:
        ap.error(f"unknown experiments {bad}; choose from {EXPERIMENTS}")
    quick = args.quick
    repeats = args.repeats or (1 if quick else ROUNDS)
    spin = 0.0 if quick else SCENE_SPIN_SECONDS

    print("Warming up both backends...")
    nb_recolor._warmup()
    tr_recolor._warmup(nb_recolor.DEFAULT_GRID[1])
    if "lane_schedule" in experiments:
        for sched in tr_recolor.LANE_SCHEDULES:
            tr_recolor._warmup(nb_recolor.DEFAULT_GRID[1], sched)
    _queue(1)                           # compile the GPU-only spin
    sync()

    peak_bytes = 16 * 2 ** 20 if quick else PEAK_BYTES
    peak_reps = 3 if quick else 10
    nb_copy = numba_copy_peak(n_bytes=peak_bytes, repeats=peak_reps)["gb_s"]
    tr_copy = triton_copy_peak(n_bytes=peak_bytes, repeats=peak_reps)["gb_s"]
    nb_rw = nb_benchmark._measure_read_write_peaks(peak_bytes)
    tr_rw = measure_read_write_peaks_triton(peak_bytes)
    peaks = {
        "n_bytes": peak_bytes,
        "numba": {"copy_gb_s": nb_copy, "read_gb_s": nb_rw[0],
                  "write_gb_s": nb_rw[1]},
        "triton": {"copy_gb_s": tr_copy, "read_gb_s": tr_rw[0],
                   "write_gb_s": tr_rw[1]},
    }
    print(f"Peaks GB/s (copy/read/write): numba {nb_copy:.0f}/{nb_rw[0]:.0f}/"
          f"{nb_rw[1]:.0f}  triton {tr_copy:.0f}/{tr_rw[0]:.0f}/{tr_rw[1]:.0f}")

    cases, samples, pair, lane_meta = build_cases(quick, spin, experiments)
    caps = [
        "ch05 head-to-head column of benchmark.py not run: a Numba-only "
        "baseline (split_L8, ~55 ms/run at 81 Mpx), outside Numba-vs-Triton",
        f"per-scene clock spin {spin:g} s with both backends alternating "
        f"(benchmark.py spins 8 s, scaling.py 4 s, per scene, one backend)",
        "one case per (scene, contract); the harness interleaves the two "
        "backends every round instead of the two contracts",
        "no scene-size caps: the largest scene is 9000x9000 (243 MB RGB). "
        "One scene is resident on the device at a time; on the host the "
        "decoded input_blobs.png (243 MB) also stays cached while its "
        "crops run (one decoded PNG at a time). Peak host RSS is about "
        "2.1-2.3 GB (meta.peak_host_rss_mb has this run's value)",
        "no scaling.py JSON of its own: its crossings are "
        "meta.scaling_crossings, per backend",
        "figures.py / visualize.py not mirrored (no kernels of their own)",
        "overview/bench_ch06.py not mirrored (overview phase)",
        "lane_schedule: the benchmark scenes in the mask contract only "
        "(merge and flatten are the same launches in both contracts)",
    ]
    if quick:
        caps.insert(0, "QUICK smoke run: small stand-in scenes, 1 round, "
                       "no spin, 16 MiB peak probes; not a measurement")
    meta = {
        "mirrors": ["chapters/ch06_gpu_nblob_runs/benchmarks/benchmark.py",
                    "chapters/ch06_gpu_nblob_runs/benchmarks/scaling.py"],
        "quick": quick,
        "experiments": list(experiments),
        "lane_schedule_default": DEFAULT_SCHED,
        "lane_schedule_note": (
            "the twin runs its default lane schedule ('independent') "
            "everywhere except the lane_schedule experiment's "
            "first_translation rows (lane_sched 'lockstep', "
            "comparable=false, first_translation=true). Its "
            "lane_independent rows that repeat a benchmark cell carry "
            "duplicate_of=<experiment>"),
        "config": {
            "grid": list(nb_recolor.DEFAULT_GRID),
            "threads_per_block": nb_recolor.DEFAULT_GRID[1],
            "phase_blocks": dict(nb_recolor.PHASE_BLOCKS),
            "pack_row_blocks": nb_recolor.PACK_ROW_BLOCKS,
            "scan_tpb": SCAN_TPB,
            "run_capacity_rule": "max(8192, int(host_run_count * 1.05))",
            "instrumented": True,
        },
        "timing": (
            "kernel_ms = CUDA-event span of run() (first launch to last), "
            "the chapter's number. The events are on the stream, so when "
            "the host is the bottleneck the span INCLUDES the Python "
            "launch enqueue (the GPU waits for each launch); it hides the "
            "enqueue only when the GPU is the bottleneck. total_ms = "
            "perf_counter around run() + synchronize: the span plus the "
            "host work before the first event executes and the sync "
            "return. enqueue_ms = host time for run() to return. "
            "gpu_fraction = gpu_only kernel_ms / kernel_ms. launch_bound "
            "= enqueue_ms >= 0.8 * kernel_ms or gpu_fraction < 0.8 (per "
            "backend; the row flag is either): there, speedup_kernel and "
            "speedup_phase measure Python launch paths as much as "
            "kernels. "
            "gpu_only = the same pipeline, once per call, with every "
            "launch queued behind a device spin of queue_us, so its event "
            "span is GPU work only; a run counts only if its first event "
            "had not executed when run() returned. speedup_gpu_kernel and "
            "speedup_gpu_phase compare kernels. phase_ms = per-phase event "
            "medians over the timed rounds (the warm-up call excluded)."),
        "queue_us": QUEUE_US,
        "launch_bound_threshold": LAUNCH_BOUND,
        "scene_spin_seconds": spin,
        "peaks": peaks,
        "bandwidth_model": nb_recolor.MODEL_NOTE,
        "caps": caps,
    }

    doc = run_cases(CHAPTER, cases, repeats, meta=meta, write=False,
                    spin_seconds=0.0 if quick else 8.0)
    pair.release()

    _png.cache_clear()
    for row, smp in zip(doc["rows"], samples):
        _finish_row(row, smp, peaks)
    doc["meta"]["scaling_crossings"] = _scaling_crossings(doc["rows"])
    if lane_meta is not None:
        doc["meta"]["lane_schedule"] = lane_meta
    doc["meta"]["peak_host_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

    for row in doc["rows"]:
        if "error" in row:
            continue
        g = row.get("speedup_gpu_kernel")
        gp = row.get("speedup_gpu_phase") or {}
        sched = row["config"].get("lane_sched")
        print(f"  {row['experiment']:13s} {row['scene']:18s} "
              f"{row['config']['contract']:4s}"
              + (f" {sched:11s}" if sched else "")
              + f" span x{row['speedup_kernel']:.2f}"
              f"  gpu-only " + (f"x{g:.2f}" if g else "n/a")
              + "".join(f" {k} x{gp[k]:.2f}" for k in ("merge", "flatten")
                        if gp.get(k))
              + ("  (launch-bound)" if row["launch_bound"] else ""))

    if not quick:
        path = os.path.join(results_dir("triton_twins", CHAPTER),
                            f"compare_{doc['created_utc']}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {path}")

    bad = [r for r in doc["rows"]
           if "error" in r or not r.get("outputs_equal", False)]
    print(f"{len(doc['rows'])} cases, {len(bad)} with errors or mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
