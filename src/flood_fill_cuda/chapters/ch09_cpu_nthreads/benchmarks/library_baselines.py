"""ch09, the libraries: OpenCV and CuPy beside this project, in one session.

Every "x times faster" in chapters 1-9 is against code written here (an
@njit BFS on one core, then ch09's cores). This session asks what a
reader asks first: how do the standard libraries do on the same images,
measured the same way, in the same process?

Scenes (8-connectivity, every method labels every blob):

  input_blobs    the 81 Mpx frame (md5 checked), 2,522 blobs
  sq_8000        8000 x 8000, all red: one 64 Mpx blob
  random_4000    4000 x 4000 noise, density 0.30: 755,577 blobs (ch05's
                 and ch08's scene)
  input_blocks   1000 x 1000, 21,618 blobs

Three contracts, kept apart (a number is only compared inside one):

  labelmap       the method's own mask format in, a full int32 label map
                 out. The libraries' natural contract.
  rgb_labelmap   the RGB image in, an int32 label map out.
  recolor        the project's published contracts, for reference:
                 "rgb" (RGB in, every blob recoloured in place) and
                 "mask" (the packed mask in). No library cell.

CPU cells (host memory in, host memory out, perf_counter):

  opencv_labelmap_t{1,N}      cv2.connectedComponents(uint8 mask, 8,
                              CV_32S), 1 thread and every thread
  opencv_rgb_labelmap_t{1,N}  cv2.inRange(exact red) + the same
  ch09_rgb_labelmap_t{1,N}    ch09's TileLabels.run(paint=False) +
                              label_map(): its input is the RGB image
  ch09_recolor_t{1,N}         TileLabels.run(paint=True), as published
  oracle_labelmap             cpu_label_components as called: the one-
                              core baseline of chapters 1-8

GPU cells (device memory in, device memory out):

  cupy_labelmap       cupyx.scipy.ndimage.label(bool mask, 3x3)
  cupy_rgb_labelmap   the exact-red test on the device + the same
                      (both: perf_counter between two synchronizes, after
                      the gap; CuPy's launches are on its clock)
  ch08_labelmap       ch08's init + label + waves on the packed mask,
                      then its label-map twin (fill -1, map kernels)
  ch08_rgb_labelmap   the same from the RGB image (pack on the clock)
  ch06_labelmap       ch06's pipeline without its paint, plus the fill
                      and its label-map kernel: the sum of two event
                      spans (the paint runs between them, untimed)
  ch08_rgb, ch08_mask, ch06_rgb, ch06_mask   recolor, as published
                      (all: CUDA events, ch09's gap before every run)

GPU context (perf_counter, after the timed block):

  e2e   ch08_rgb_labelmap from a pageable host image to a host label map
        (copy in, pipeline, copy back): what a caller with the image in
        host memory waits for, the libraries' CPU cells' own situation

Policy: ch09's (benchmark.py). GPU blocks first, each with a ramp and a
full-clocks check (nvidia-smi samples); a cooldown; then the CPU passes
with the idle gate before every round, the thread pools settled before
every cell, CPU_ROUNDS rounds, the order reversed on odd rounds. A CPU
round during which the rest of the machine was busier than the gate's
threshold is run again (up to MAX_ROUND_REDOS times), then kept and
flagged; the session records all_cpu_rounds_quiet.

Checks: every cell's output is compared with cpu_label_components after
its untimed run and after its last timed run. The project's label maps
must equal the oracle's exactly; a library's labels must give the same
partition and the same count; recoloured images must equal the palette
image.

Output: results/ch09_cpu_nthreads/benchmark_results/libs_<stamp>.json

Run (a quiet machine, no nice):

    uv run python -m flood_fill_cuda.chapters.ch09_cpu_nthreads.benchmarks.library_baselines
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import gc
import json
import math
import resource
import time
from datetime import datetime, timezone

import cv2
import numba
import numpy as np

from ....shared import results_paths
from ...ch05_gpu_nblob_nblock import scenes as ch05_scenes
from ...ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components
from .. import flood_fill as ff
from . import benchmark as B

CHAPTER = "ch09_cpu_nthreads"
PNG_DIR = os.path.join(B.REPO, "images", "input")
PNG_MD5 = {"input_blobs": "47639eb37f973b5092ec2c4db86f45a4",
           "input_blocks": "68f1fda7f2eba3d73b87e1cc11155061"}
SCENES = ("input_blobs", "sq_8000", "random_4000", "input_blocks")
RED = (255, 0, 0)


# ------------------------------------------------------------ helpers

def red_mask(img):
    return ((img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0))


def same_partition(lab, bg, ref, n_ref):
    """Does `lab` (background value bg) split the red pixels exactly as
    the oracle's canonical map `ref` (-1 off blob, n_ref blobs)?"""
    red = ref >= 0
    if not np.array_equal(lab != bg, red):
        return False
    a = lab[red].astype(np.int64)
    b = ref[red].astype(np.int64)
    if a.size == 0:
        return n_ref == 0
    pairs = np.unique(a * (int(b.max()) + 1) + b).size
    return pairs == np.unique(a).size == n_ref


def build_scenes(args):
    """[(name, img)], img (W, H, 3) uint8, pure red on white."""
    out = []
    for name in SCENES:
        if args.scenes and name not in args.scenes:
            continue
        if name in PNG_MD5:
            path = os.path.join(PNG_DIR, f"{name}.png")
            md5 = B._md5(path)
            if md5 != PNG_MD5[name]:
                raise RuntimeError(f"{name}.png: md5 {md5}, expected "
                                   f"{PNG_MD5[name]}")
            img, _ = ch05_scenes.png_scene(path)
        elif name == "sq_8000":
            n = args.sq_size
            img = np.empty((n, n, 3), np.uint8)
            img[...] = RED
        else:
            n = args.noise_size
            img = ch05_scenes.random_blobs_scene(n, n, 0.30, 0)[0]
        if args.crop:
            img = img[:args.crop, :args.crop]
        out.append((name, np.ascontiguousarray(img)))
    return out


def reference(img):
    label, n = cpu_label_components(img)
    return {"label": label, "n_blobs": int(n),
            "palette": B.palette_image(img, label)}


# ---------------------------------------------------------- CPU cells

def cpu_cells(img, ref, threads):
    """[{key, contract, threads, call, check}]: call() is the timed
    span and returns its output; check(out) compares it with the
    oracle."""
    w, h = img.shape[0], img.shape[1]
    mask = np.ascontiguousarray(red_mask(img).astype(np.uint8))
    work = np.empty_like(img)
    engines = {t: ff.TileLabels(w, h, connectivity=8, threads=t)
               for t in threads}
    cells = []

    def add(key, contract, t, call, check, prep=None):
        cells.append({"key": key, "contract": contract, "threads": t,
                      "call": call, "check": check, "prep": prep})

    def cv_check(out):
        n, lab = out
        return bool(n - 1 == ref["n_blobs"] and same_partition(
            lab, 0, ref["label"], ref["n_blobs"]))

    def map_check(out):
        return bool(B.same(out, ref["label"]))

    def restore():
        np.copyto(work, img)

    for t in threads:
        def cv_labels(t=t):
            return cv2.connectedComponents(mask, connectivity=8,
                                           ltype=cv2.CV_32S)

        def cv_rgb(t=t):
            m = cv2.inRange(img, RED, RED)
            return cv2.connectedComponents(m, connectivity=8,
                                           ltype=cv2.CV_32S)

        def ch09_map(t=t):
            with ff.num_threads(t):
                engines[t].run(img, paint=False)
                return engines[t].label_map()

        def ch09_recolor(t=t):
            engines[t].run(work, paint=True)
            return work

        cv_prep = (lambda t=t: cv2.setNumThreads(t))
        add(f"opencv_labelmap_t{t}", "labelmap", t, cv_labels, cv_check,
            cv_prep)
        add(f"opencv_rgb_labelmap_t{t}", "rgb_labelmap", t, cv_rgb,
            cv_check, cv_prep)
        add(f"ch09_rgb_labelmap_t{t}", "rgb_labelmap", t, ch09_map,
            map_check)
        add(f"ch09_recolor_t{t}", "recolor", t, ch09_recolor,
            lambda out: bool(B.same(out, ref["palette"])), restore)
    add("oracle_labelmap", "rgb_labelmap", 1,
        lambda: cpu_label_components(img)[0], map_check)
    return cells


def cpu_pass(cells, args):
    """ch09's rounds: one untimed checked call, one timed first call
    (sets the calls per round), then the rounds, the order reversed on
    odd rounds. Before every round the idle gate; the pools settled and
    the cell's own preparation (thread count, image restore) off the
    clock before every call. The rest of the machine is metered over
    every round: a round above the gate's threshold is run again (up to
    MAX_ROUND_REDOS times, the discarded attempts recorded), then kept
    and flagged "loaded". Returns the rounds' records."""
    max_t = max(c["threads"] for c in cells)
    threshold = B.gate_threshold(max_t)

    def one_call(c):
        if c["prep"]:
            c["prep"]()
        t0 = time.perf_counter()
        out = c["call"]()
        return (time.perf_counter() - t0) * 1000, out

    for c in cells:
        B.settle_workers()
        _, out = one_call(c)
        c["checks_ok"] = c["check"](out)
        del out
        ms, out = one_call(c)
        del out
        c["calls"] = max(1, min(args.max_calls,
                                math.ceil(args.round_s * 1000 / ms)))
        c["samples"], c["round_ms"] = [], []
    rounds = []
    for r in range(args.cpu_rounds):
        final = r == args.cpu_rounds - 1
        discarded = []
        for attempt in range(B.MAX_ROUND_REDOS + 1):
            gate = B.wait_quiet(B.QUIET_WAIT_CPU_S, args.gate, threshold)
            meter = B.LoadMeter("round")
            w0 = time.time()
            got, ok = {}, {}
            for c in (cells if r % 2 == 0 else cells[::-1]):
                B.settle_workers()
                batch, out = [], None
                for _ in range(c["calls"]):
                    del out
                    ms, out = one_call(c)
                    batch.append(ms)
                if final:
                    ok[c["key"]] = bool(c["check"](out))
                del out
                got[c["key"]] = batch
            load = meter.read()
            rec = {"round": r, "attempt": attempt,
                   "seconds": round(time.time() - w0, 2),
                   "machine_busy_pct": load["machine_pct"],
                   "others_busy_pct": load["others_pct"],
                   "threshold_pct": threshold, "quiet_gate": gate}
            loaded = (load["others_pct"] is not None
                      and load["others_pct"] > threshold)
            print(f"    CPU round {r}.{attempt}: {rec['seconds']:.1f} s, "
                  f"others {load['others_pct']}%"
                  f"{', LOADED' if loaded else ''}", flush=True)
            if loaded and args.gate and attempt < B.MAX_ROUND_REDOS:
                discarded.append(rec)
                continue
            break
        for c in cells:
            c["samples"] += got[c["key"]]
            c["round_ms"].append(float(np.median(got[c["key"]])))
            if final:
                c["checks_ok"] = bool(c["checks_ok"] and ok[c["key"]])
        rec["loaded"] = bool(loaded)
        rec["discarded_attempts"] = discarded
        rounds.append(rec)
    cv2.setNumThreads(max_t)
    return rounds


def cpu_record(c):
    return {**B._stats(c["round_ms"]), "round_ms": c["round_ms"],
            "samples": c["samples"], "calls_per_round": c["calls"],
            "side": "cpu", "contract": c["contract"],
            "threads": c["threads"], "clock": "perf_counter",
            "checks_ok": c["checks_ok"]}


# ---------------------------------------------------------- GPU cells

class GpuSide:
    """One scene on the GPU: the image, ch06's and ch08's engines, CuPy's
    arrays, and one run per cell key."""

    def __init__(self, gpu, img, ref):
        import cupy as cp
        from cupyx.scipy import ndimage as cnd
        from ...ch06_gpu_nblob_runs import recolor as r6
        from ...ch06_gpu_nblob_runs.kernels import N_RUNS, RUN_OVERFLOW
        from ...ch08_gpu_nblob_tiles.kernels import fill_i32

        cuda = gpu.cuda
        self.gpu, self.cuda, self.cp, self.cnd = gpu, cuda, cp, cnd
        self.ref, self.fill = ref, fill_i32
        self.w, self.h = img.shape[0], img.shape[1]
        self.d_pristine = cuda.to_device(img)
        self.d_img = cuda.device_array_like(img)
        self.d_label = cuda.device_array((self.w, self.h), np.int32)
        self.d_label_flat = self.d_label.reshape(self.w * self.h)
        # ch08: the default engine; its mask is packed once (the rgb
        # contract's own pack rewrites the same words)
        self.e8 = gpu.ff8.TileWaves(self.w, self.h)
        self.e8.pack(self.d_pristine)
        # ch06: a run table sized for this scene (the one-shot path would
        # resize; an engine reports the count it needed)
        self.e6 = r6.RunRecolor(self.w, self.h)
        self.e6.pack(self.d_pristine)
        self.d_img.copy_to_device(self.d_pristine)
        self.e6.run(self.d_img, contract="mask")
        cuda.synchronize()
        c = self.e6.counters.copy_to_host()
        if c[RUN_OVERFLOW]:
            need = int(c[N_RUNS])
            self.e6 = r6.RunRecolor(self.w, self.h,
                                    run_capacity=need + need // 8)
            self.e6.pack(self.d_pristine)
        # CuPy: its own arrays (a bool mask, the RGB image)
        self.c_mask = cp.asarray(red_mask(img))
        self.c_img = cp.asarray(img)
        self.s8 = cp.ones((3, 3), dtype=bool)
        self.c_out = None
        self.keys = ["cupy_labelmap", "cupy_rgb_labelmap", "ch08_labelmap",
                     "ch08_rgb_labelmap", "ch06_labelmap", "ch08_rgb",
                     "ch08_mask", "ch06_rgb", "ch06_mask"]
        self.contract = {k: ("recolor" if k.endswith(("_rgb", "_mask"))
                             else "rgb_labelmap" if "rgb_" in k
                             else "labelmap") for k in self.keys}

    def release(self):
        self.c_mask = self.c_img = self.c_out = None
        self.cp.get_default_memory_pool().free_all_blocks()
        self.cuda.current_context().deallocations.clear()

    # ---- one timed run per key (ms)
    def run(self, key):
        cuda, e8, e6 = self.cuda, self.e8, self.e6
        self.d_img.copy_to_device(self.d_pristine)      # off the clock
        self.gpu.gap()
        if key.startswith("cupy"):
            cp = self.cp
            cuda.synchronize()        # the gap's sleep is not CuPy's time
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            if key == "cupy_rgb_labelmap":
                im = self.c_img
                m = (im[..., 0] == 255) & (im[..., 1] == 0) & (im[..., 2] == 0)
            else:
                m = self.c_mask
            self.c_out = self.cnd.label(m, structure=self.s8)
            cp.cuda.Stream.null.synchronize()
            return (time.perf_counter() - t0) * 1000
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        if key.startswith("ch08"):
            contract = "rgb" if "rgb" in key else "mask"
            labelmap = key.endswith("labelmap")
            e0.record()
            for _, go in e8.stages(self.d_img, contract, paint=not labelmap):
                go()
            if labelmap:
                e8.emit_label_map(self.d_label)
            e1.record()
            e1.synchronize()
            return cuda.event_elapsed_time(e0, e1)
        contract = "rgb" if key == "ch06_rgb" else "mask"
        names, ev = e6.run(self.d_img, contract=contract)
        if key != "ch06_labelmap":
            ev[-1].synchronize()
            return cuda.event_elapsed_time(ev[0], ev[-1])
        e0.record()
        self.fill[256, 256](self.d_label_flat, np.int32(-1))
        e6.emit_label_map(self.d_label)
        e1.record()
        e1.synchronize()
        return (cuda.event_elapsed_time(ev[0], ev[names.index("paint")])
                + cuda.event_elapsed_time(e0, e1))

    def verify(self, key):
        """The last run of `key` against the oracle."""
        ref = self.ref
        if key.startswith("cupy"):
            lab, n = self.c_out
            return bool(int(n) == ref["n_blobs"] and same_partition(
                self.cp.asnumpy(lab), 0, ref["label"], ref["n_blobs"]))
        if key.endswith("labelmap"):
            return bool(B.same(self.d_label.copy_to_host(), ref["label"]))
        return bool(np.array_equal(self.d_img.copy_to_host(),
                                   ref["palette"]))

    def e2e(self, img, repeats=B.E2E_REPEATS):
        """A pageable host image to a host label map, through ch08."""
        cuda = self.cuda
        out = np.empty((self.w, self.h), np.int32)
        ms = []
        for _ in range(repeats):
            cuda.synchronize()
            t0 = time.perf_counter()
            self.d_img.copy_to_device(img)
            for _, go in self.e8.stages(self.d_img, "rgb", paint=False):
                go()
            self.e8.emit_label_map(self.d_label)
            self.d_label.copy_to_host(out)
            ms.append((time.perf_counter() - t0) * 1000)
        return {**B._stats(ms), "samples": ms, "cell": "ch08_rgb_labelmap",
                "matches_cpu": bool(B.same(out, self.ref["label"]))}


def gpu_scene(gpu, sampler, name, img, ref, args):
    side = GpuSide(gpu, img, ref)
    checks = {}
    for k in side.keys:
        side.run(k)
        checks[k] = side.verify(k)

    def run_round(i, spacing):
        out = {}
        for k in (side.keys if i % 2 == 0 else side.keys[::-1]):
            out[k] = side.run(k)
            if spacing:
                gpu.filler(spacing)
        return out

    data, attempts = B.timed_block(gpu, sampler, name, run_round,
                                   lambda spacing: run_round(0, spacing),
                                   args)
    res = {}
    for k in side.keys:
        samples = [r[k] for r in data]
        side.run(k)
        last = side.verify(k)
        res[k] = {**B._stats(samples), "samples": samples, "side": "gpu",
                  "contract": side.contract[k],
                  "clock": ("perf_counter between synchronizes"
                            if k.startswith("cupy") else "CUDA events"),
                  "checks_ok": bool(checks[k] and last),
                  "full_clocks": attempts[-1]["full_clocks"]}
    e2e = side.e2e(img)
    side.release()
    del side
    gc.collect()
    return res, {"attempts": attempts}, e2e


# ---------------------------------------------------------- summary

def summarize(cells):
    """Per contract: every cell's median, and how many times slower it is
    than the contract's fastest cell."""
    out = {}
    for contract in ("labelmap", "rgb_labelmap", "recolor"):
        rows = {k: c["ms"] for k, c in cells.items()
                if c["contract"] == contract}
        if rows:
            best = min(rows.values())
            out[contract] = {k: {"ms": v, "x_fastest": v / best}
                             for k, v in sorted(rows.items(),
                                                key=lambda kv: kv[1])}
    return out


def lib_versions():
    info = cv2.getBuildInformation()
    par = next((ln.split(":", 1)[1].strip() for ln in info.splitlines()
                if "Parallel framework" in ln), None)
    out = {"opencv": cv2.__version__, "opencv_parallel_framework": par,
           "opencv_threads": cv2.getNumThreads(),
           "numpy": np.__version__, "numba": numba.__version__}
    try:
        import cupy
        out["cupy"] = cupy.__version__
    except ImportError:
        out["cupy"] = None
    return out


# -------------------------------------------------------------- main

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", nargs="*", default=None, choices=SCENES)
    ap.add_argument("--sq-size", type=int, default=8000)
    ap.add_argument("--noise-size", type=int, default=4000)
    ap.add_argument("--crop", type=int, default=None,
                    help="use the top-left N x N of every scene")
    ap.add_argument("--threads", nargs="*", type=int, default=None,
                    help="default: 1 and every thread Numba can use")
    ap.add_argument("--cpu-rounds", type=int, default=B.CPU_ROUNDS)
    ap.add_argument("--round-s", type=float, default=B.CPU_ROUND_S)
    ap.add_argument("--max-calls", type=int, default=25)
    ap.add_argument("--gpu-rounds", type=int, default=B.GPU_ROUNDS)
    ap.add_argument("--warm-seconds", type=float, default=B.WARM_SECONDS)
    ap.add_argument("--max-attempts", type=int, default=B.MAX_ATTEMPTS)
    ap.add_argument("--cooldown", type=float, default=B.CPU_COOLDOWN_S)
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="small crops, short rounds, no gate (a harness "
                         "check: its timings mean nothing)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threading-layer", default="omp")
    ap.add_argument("--allow-nice", action="store_true")
    ap.add_argument("--allow-openmp-env", action="store_true")
    return ap


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.crop = args.crop or 512
        args.sq_size = min(args.sq_size, 512)
        args.noise_size = min(args.noise_size, 512)
        args.cpu_rounds, args.round_s, args.max_calls = 1, 0.02, 2
        args.gpu_rounds, args.warm_seconds, args.max_attempts = 2, 0.3, 1
        args.cooldown, args.gate = 0.0, False
    if not args.threads:
        args.threads = sorted({1, numba.config.NUMBA_NUM_THREADS})
    args.threads = B.select_threads(args.threads,
                                    numba.config.NUMBA_NUM_THREADS)
    return args


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"library baselines, session libs_{stamp}: threads {args.threads}"
          f"{' (SMOKE: timings mean nothing)' if args.smoke else ''}",
          flush=True)
    notes = B.check_runtime(args, B.runtime_info())
    for n in notes:
        print(f"NOTE: {n}", flush=True)
    numba.config.THREADING_LAYER = args.threading_layer
    scenes = build_scenes(args)
    # JIT and both thread pools, on a small image
    tiny = ch05_scenes.random_blobs_scene(96, 96, 0.3, 1)[0]
    warm = cpu_cells(tiny, reference(tiny), args.threads)
    for c in warm:
        if c["prep"]:
            c["prep"]()
        c["call"]()
    del warm
    runtime = B.runtime_info()

    rows = [{"scene": name, "shape": [int(i.shape[0]), int(i.shape[1])],
             "cells": {}} for name, i in scenes]
    gpu_info, gpu_blocks = None, {}
    if not args.no_gpu:
        gpu = B.Gpu()
        gpu_info = {**B.gpu_identity(gpu), "clocks_start": B.gpu_clocks()}
        sampler = B.Sampler()
        try:
            print(f"GPU: {gpu.device} ({gpu.sm_count} SMs)", flush=True)
            for row, (name, img) in zip(rows, scenes):
                print(f"  GPU {name}", flush=True)
                ref = reference(img)
                row["n_blobs"] = ref["n_blobs"]
                res, gpu_blocks[name], row["e2e"] = gpu_scene(
                    gpu, sampler, name, img, ref, args)
                row["cells"].update(res)
                for k, c in res.items():
                    print(f"    {k:22s} {c['ms']:10.4f} ms  "
                          f"{'OK' if c['checks_ok'] else 'MISMATCH'}",
                          flush=True)
                del ref
                gc.collect()
        finally:
            sampler.stop()
        gpu_info["clocks_end"] = B.gpu_clocks()
        if args.cooldown:
            print(f"Cooldown {args.cooldown:.0f} s before the CPU passes",
                  flush=True)
            time.sleep(args.cooldown)

    cpu_rounds = {}
    for row, (name, img) in zip(rows, scenes):
        print(f"  CPU {name}", flush=True)
        ref = reference(img)
        row["n_blobs"] = ref["n_blobs"]
        cells = cpu_cells(img, ref, args.threads)
        cpu_rounds[name] = cpu_pass(cells, args)
        row["loaded_rounds"] = sum(r["loaded"] for r in cpu_rounds[name])
        for c in cells:
            row["cells"][c["key"]] = cpu_record(c)
            print(f"    {c['key']:26s} {row['cells'][c['key']]['ms']:10.3f}"
                  f" ms  {'OK' if c['checks_ok'] else 'MISMATCH'}",
                  flush=True)
        row["summary"] = summarize(row["cells"])
        row["all_checks_ok"] = all(c["checks_ok"]
                                   for c in row["cells"].values()) and (
            "e2e" not in row or row["e2e"]["matches_cpu"])
        del cells, ref
        gc.collect()

    ok = all(r["all_checks_ok"] for r in rows)
    print(f"\nAll checked outputs {'match' if ok else 'DO NOT match'} the "
          "oracle", flush=True)
    loaded = {r["scene"]: r["loaded_rounds"] for r in rows
              if r.get("loaded_rounds")}
    if loaded:
        print(f"LOADED CPU rounds kept (the machine was not quiet): {loaded}",
              flush=True)
    session = {
        "session": f"libs_{stamp}", "smoke": bool(args.smoke),
        "host": B._host_info(), "libraries": lib_versions(),
        "runtime": {**runtime, "notes": notes}, "gpu": gpu_info,
        "git": {"commit": B._git("rev-parse", "--short", "HEAD"),
                "dirty": bool(B._git("status", "--porcelain"))},
        "settings": {"threads": args.threads, "cpu_rounds": args.cpu_rounds,
                     "round_s": args.round_s, "max_calls": args.max_calls,
                     "gpu_rounds_min": args.gpu_rounds, "gate": args.gate,
                     "cooldown_s": args.cooldown, "connectivity": 8,
                     "crop": args.crop, "png_md5": PNG_MD5},
        "rows": rows, "cpu_rounds": cpu_rounds, "gpu_blocks": gpu_blocks,
        "all_checks_ok": bool(ok),
        "all_cpu_rounds_quiet": all(r.get("loaded_rounds", 0) == 0
                                    for r in rows),
        "all_gpu_blocks_full_clocks": all(
            b["attempts"][-1]["full_clocks"] for b in gpu_blocks.values()),
        "peak_rss_mb": round(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024)}
    if args.no_save:
        return session
    out_dir = args.out_dir or results_paths.results_dir(CHAPTER,
                                                        "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"libs_{stamp}.json")
    with open(path, "w") as f:
        json.dump(session, f, indent=1, default=B._json_default)
    print(f"wrote {os.path.basename(path)}", flush=True)
    return session


if __name__ == "__main__":
    main()
