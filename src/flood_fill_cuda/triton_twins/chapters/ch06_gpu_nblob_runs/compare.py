"""Numba vs Triton on chapter 6's own benchmarks.

Mirrors the chapter's two benchmark scripts, with identical configs on
both sides: DEFAULT_GRID (256 threads per block = 8 warps per program),
PHASE_BLOCKS, PACK_ROW_BLOCKS, the single 1024-lane scan, and the run
capacity sized from the host run count exactly as the benchmark sizes
it (max(8192, runs * 1.05)).

  benchmark   benchmarks/benchmark.py: input_blobs.png, input_blocks.png
              and the four synthetic scenes (blob_grid_100, random_4000,
              disk_r2000, serpentine_2048), each in both contracts
  scaling     benchmarks/scaling.py: centered crops of input_blobs.png,
              1000 to 9000 px a side, both contracts

One case is one scene in one contract. Each backend runs the pipeline
the way _bench_ch06 does: restore the pristine image (device to
device), pack off the clock for the "mask" contract, synchronize,
run(), synchronize. kernel_ms is the CUDA-event span from before the
first launch to after the last (the chapter's headline number);
total_ms is the host wall time of run() + synchronize, so it adds the
Python launch enqueue that the event span hides when the GPU outruns
the host. Per-phase medians (and label_only_ms, every phase but paint)
are added to each row, from the same timed runs.

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
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare [--quick] [--repeats N]

--quick is a smoke test: small scenes, one round, no JSON.
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import gc
import json
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

from . import recolor as tr_recolor
from .kernels import (
    N_BLOBS, N_RUNS, N_RUNS_USED, RUN_OVERFLOW, SCAN_TPB, UNION_ATTEMPTS,
    UNION_DONE,
)

CHAPTER = "ch06_gpu_nblob_runs"
ROUNDS = nb_benchmark.ROUNDS            # 9, as in benchmark.py and scaling.py
SCENE_SPIN_SECONDS = 3.0                # per scene, both backends alternating
PEAK_BYTES = 256 * 2 ** 20


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
    """The Triton twin's engine on the same scene, run the same way."""

    name = "triton"

    def __init__(self, img, capacity):
        self.engine = tr_recolor.RunRecolor(img.shape[0], img.shape[1],
                                            run_capacity=capacity)
        self.pristine = cp.asarray(img)
        self.dev = cp.asarray(img)
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

    def counters(self):
        return self.engine.counters.get()

    def resources(self):
        return {name: kernel_resources(k)
                for name, k in self.engine.compiled.items()}


def _measure(side, contract, samples):
    """One timed pipeline run of one backend (the harness's callable)."""
    side.restore()
    if contract == "mask":
        side.pack()                     # the packed input, off the clock
    side.sync()
    t0 = time.perf_counter()
    names, events = side.run(contract)
    side.sync()
    t1 = time.perf_counter()
    counters = side.counters()
    if counters[RUN_OVERFLOW]:
        raise RuntimeError("run table overflowed during benchmark")
    phase_ms = {n: side.elapsed(events[i], events[i + 1])
                for i, n in enumerate(names)}
    samples.append(phase_ms)
    return SimpleNamespace(kernel_ms=side.elapsed(events[0], events[-1]),
                           total_ms=(t1 - t0) * 1000, phase_ms=phase_ms,
                           counters=counters, side=side)


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

@lru_cache(maxsize=None)
def _png(path):
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
    scene, released when the next scene starts."""

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
                triton=_TritonSide(img, capacity))
            del img
            self.key = key
            self._spin()
        return self.cur

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


def _summary(samples):
    if not samples:
        return None
    phase = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
    return {"phase_ms": phase,
            "label_only_ms": sum(v for k, v in phase.items() if k != "paint")}


def build_cases(quick, spin_seconds):
    pair = _Pair(spin_seconds)
    cases, samples = [], []
    for experiment, scene, build, note in _scene_list(quick):
        key = (experiment, scene)
        for contract in nb_recolor.CONTRACTS:
            smp = {"numba": [], "triton": []}

            def run_numba(key=key, build=build, contract=contract, smp=smp):
                return _measure(pair.get(key, build).numba, contract,
                                smp["numba"])

            def run_triton(key=key, build=build, contract=contract, smp=smp):
                return _measure(pair.get(key, build).triton, contract,
                                smp["triton"])

            def info(rn, rt, contract=contract):
                cur = pair.cur
                c = rn.counters
                n_px = cur.width * cur.height
                tpb = cur.triton.engine.grid[1]
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
                    "grid": {
                        "threads_per_block": tpb,
                        "num_warps": tpb // 32,
                        "pack": {"numba": list(cur.numba.engine.pack_grid[0]),
                                 "triton": list(cur.triton.engine.pack_grid[0])},
                        "phase_blocks": {
                            "numba": {k: v[0] for k, v in
                                      cur.numba.engine.phase_grid.items()},
                            "triton": {k: v[0] for k, v in
                                       cur.triton.engine.phase_grid.items()}},
                        "scan": {"numba": [1, nb_kernels.SCAN_TPB],
                                 "triton": [1, SCAN_TPB]},
                    },
                    "numba_resources": cur.numba.resources(),
                    "triton_kernel_resources": cur.triton.resources(),
                }

            cases.append(Case(
                experiment=experiment, scene=scene,
                config={"contract": contract,
                        "tpb": nb_recolor.DEFAULT_GRID[1],
                        "grid": list(nb_recolor.DEFAULT_GRID)},
                run_numba=run_numba, run_triton=run_triton, same=_same,
                pixels=0, info=info, notes=note))
            samples.append(smp)
    return cases, samples, pair


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="small scenes, 1 round, no JSON (smoke test)")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per case (default {ROUNDS})")
    args = ap.parse_args(argv)
    quick = args.quick
    repeats = args.repeats or (1 if quick else ROUNDS)
    spin = 0.0 if quick else SCENE_SPIN_SECONDS

    print("Warming up both backends...")
    nb_recolor._warmup()
    tr_recolor._warmup(nb_recolor.DEFAULT_GRID[1])

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

    cases, samples, pair = build_cases(quick, spin)
    caps = [
        "ch05 head-to-head column of benchmark.py not run: a Numba-only "
        "baseline (split_L8, ~55 ms/run at 81 Mpx), outside Numba-vs-Triton",
        f"per-scene clock spin {spin:g} s with both backends alternating "
        f"(benchmark.py spins 8 s, scaling.py 4 s, per scene, one backend)",
        "one case per (scene, contract); the harness interleaves the two "
        "backends every round instead of the two contracts",
        "no scene-size caps: the largest scene is 9000x9000 (243 MB RGB), "
        "one scene resident at a time on host and device",
        "figures.py / visualize.py not mirrored (no kernels of their own)",
        "overview/bench_ch06.py not mirrored (overview phase)",
    ]
    if quick:
        caps.insert(0, "QUICK smoke run: small stand-in scenes, 1 round, "
                       "no spin, 16 MiB peak probes; not a measurement")
    meta = {
        "mirrors": ["chapters/ch06_gpu_nblob_runs/benchmarks/benchmark.py",
                    "chapters/ch06_gpu_nblob_runs/benchmarks/scaling.py"],
        "quick": quick,
        "config": {
            "grid": list(nb_recolor.DEFAULT_GRID),
            "threads_per_block": nb_recolor.DEFAULT_GRID[1],
            "phase_blocks": dict(nb_recolor.PHASE_BLOCKS),
            "pack_row_blocks": nb_recolor.PACK_ROW_BLOCKS,
            "scan_tpb": SCAN_TPB,
            "run_capacity_rule": "max(8192, int(host_run_count * 1.05))",
            "instrumented": True,
        },
        "timing": ("kernel_ms = CUDA-event span of run() (first launch to "
                   "last); total_ms = perf_counter around run() + "
                   "synchronize (adds host launch enqueue); phase_ms = "
                   "per-phase event medians over the timed rounds"),
        "scene_spin_seconds": spin,
        "peaks": peaks,
        "bandwidth_model": nb_recolor.MODEL_NOTE,
        "caps": caps,
    }

    doc = run_cases(CHAPTER, cases, repeats, meta=meta, write=False,
                    spin_seconds=0.0 if quick else 8.0)
    pair.release()

    # Per-phase medians from the timed rounds (sample 0 is the warm-up),
    # and the pixel count of the lazily built scene.
    for row, smp in zip(doc["rows"], samples):
        if "info" in row:
            row["pixels"] = row["info"]["width"] * row["info"]["height"]
        if "error" in row:
            continue
        for backend in ("numba", "triton"):
            s = _summary(smp[backend][1:])
            if s:
                row[backend].update(s)
        if row["numba"].get("phase_ms") and row["triton"].get("phase_ms"):
            row["speedup_phase"] = {
                k: (row["numba"]["phase_ms"][k] / row["triton"]["phase_ms"][k]
                    if row["triton"]["phase_ms"][k] > 0 else None)
                for k in row["numba"]["phase_ms"]}

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
