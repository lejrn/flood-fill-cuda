"""Numba vs Triton on the grand table: every overview row x every GPU column.

The twin of overview/bench.py (chapters 1-5) and overview/bench_ch06.py
(chapter 6). Rows, scene builders, seed helpers, column definitions,
sample sizes and skips are imported from those two files, never copied,
so the grid cannot drift. Each Numba column runner gets a Triton
counterpart that calls the twin driver with the identical arguments.
One Case per row x column goes through compare/harness.py (warm-up,
alternating rounds, outputs compared on every run), and the JSON lands in
results/triton_twins/overview/compare_<UTC>.json, which
compare/summary.py rolls up as one more unit.

Cell semantics, kept from bench.py and applied the same way on both
sides:

  measured   the column's job is the row's job: one driver call.
  loop       a one-blob kernel (ch01-ch03) on a two-blob row: one call
             per blob, kernel_ms and total_ms summed (MEASURED).
  est        a one-blob kernel on an N-blob row: the same k-blob sample
             as bench.py (16 blobs, 6 past 20 Mpx), median per-call ms x
             the call count a full loop needs (one per blob). ch04 on an
             N-blob row samples blob PAIRS, one call per pair. est=True
             is recorded in the row; a full loop would take hours.
  skip       bench.py's own skips, on both sides: "na" (ch04 needs
             exactly two blobs, a one-blob row is outside its input
             space), STATIC_SKIPS (ch04 streams: two persistent kernels
             at once can wedge the GPU, and bench.py skips it as
             "unsupported"), and bench.py's typed runtime refusals (a
             RuntimeError on the first call: "overflow" for the ch01
             ring, else "error:<type>"; ch06 uses bench_ch06's rule).
             A runtime skip is decided by the Numba side, as bench.py
             decides it; the Triton side is then probed once and what it
             did is recorded next to the reason. Skips never become
             harness rows: they are listed in meta["skipped_cells"].

Same arguments, so the same launch a caller gets: ch03, ch04 and ch05
columns run blocks=None, and each backend resolves its own cooperative
grid. Both resolved grids (and block sizes) are taken from the warm-up
results into config["resolved_blocks"] / ["resolved_tpb"]; a row whose
two grids or block sizes differ is comparable=False and stays out of the
summary's averages, as in the chapter compares. The one argument that
cannot be identical is ch02_pinned: Numba pins 2 x 768 threads, the twin
refuses non-power-of-2 blocks and runs its PINNED_TPB (2 x 512), as the
ch02 compare does (comparable=False).

Outputs: every call's deterministic outputs are reduced to sha1 digests
(with dtype and shape) right after the call, and the arrays are dropped.
A run returns a LIGHT result (kernel_ms, total_ms, digests, scalars), so
the harness never holds two full outputs (one ch05 result at 81 Mpx is
about 1.7 GB of host arrays). The fields digested per chapter are the
ones each chapter compare's same() treats as deterministic, minus the
grid-dependent ones (blocks, per-block counts, thread utilisation),
because the grids may differ here. On rows above 20 Mpx both memory pools
are emptied before every call (6 GB host, shared VRAM), so total_ms there
includes cudaMalloc on both sides; kernel_ms is untouched.

Timing is the drivers' own: kernel_ms is the perf_counter + synchronize
bracket for ch01-ch05 and the CUDA-event span of run() for ch06 (pack off
the clock for the mask contract, restore first: bench_ch06's protocol).
bench_ch06's per-row 8 s clock spin is replaced by the harness spin (8 s)
plus ROW_SPIN_SECONDS before every later row.

Not compared: the two CPU columns (pure_python, njit) have no GPU
backend to pair, and rows whose input PNG is missing (bench.py adds
them only when present) are absent from both tables.

Budget: the default run (17 rows x 20 columns: 303 cases and 37 static
skips, 6 repeats, bench.py's GPU_REPEATS=5 rounded up to even) is
estimated before it starts from the Numba overview JSONs' per-cell ms
plus a per-call host-cost model (allocations, transfers, sha1, the
drivers' host checks) measured on this laptop. It comes to about 57
minutes, most of it the est cells of random_4000 (13 min) and png_blobs
(28 min: every sampled call moves the whole 81 Mpx image, and ch04's
driver spends about 3 s of host work per call there). If the estimate
exceeds --budget-min (60), the heaviest cells drop to 2 repeats until it
fits, and meta["reduced_repeats"] lists them. --estimate-only prints the
plan per row without measuring.

Memory: peak host RSS measured 3.5 GB on png_blobs (the 243 MB scene, a
ch05 result of about 1.7 GB inside the Numba or Triton driver before it
is digested, the component-seed labelling, and the CUDA, CuPy and Triton
runtimes). Nothing else memory-heavy should run beside it.

Run:
    python -m flood_fill_cuda.triton_twins.compare.overview [--quick] [--repeats N] [--rows a,b] [--cols x,y]
        [--budget-min M] [--estimate-only] [--no-write]

--quick runs the small rows only (at most 1 Mpx), 1 repeat (2 after
rounding), no spin, no JSON. The default writes the JSON after every
row (meta.complete stays false until the last one), so a crash keeps
the rows already measured.
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import dataclasses
import gc
import hashlib
import json
import resource
import statistics
import time
import warnings
from types import SimpleNamespace

import numpy as np
from numba.core.errors import NumbaPerformanceWarning

from flood_fill_cuda.chapters.ch06_gpu_nblob_runs.kernels import (
    N_BLOBS, N_RUNS, N_RUNS_USED, RUN_OVERFLOW, UNION_ATTEMPTS,
)
from flood_fill_cuda.overview import bench
from flood_fill_cuda.overview import bench_ch06
from flood_fill_cuda.shared import results_paths
from flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block import (
    flood_fill as tff1,
)
from flood_fill_cuda.triton_twins.chapters.ch02_gpu_1blob_2block import (
    flood_fill as tff2,
)
from flood_fill_cuda.triton_twins.chapters.ch03_gpu_1blob_nblock import (
    flood_fill as tff3,
)
from flood_fill_cuda.triton_twins.chapters.ch04_gpu_2blob_nblock import (
    flood_fill as tff4,
)
from flood_fill_cuda.triton_twins.chapters.ch05_gpu_nblob_nblock import (
    flood_fill as tff5,
)
from flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare import (
    _NumbaSide, _TritonSide, _prepare,
)
from flood_fill_cuda.triton_twins.compare.harness import (
    Case, free_device_memory, run_cases,
)

UNIT = "overview"
TPB = bench.TPB
DEFAULT_REPEATS = bench.GPU_REPEATS + bench.GPU_REPEATS % 2   # 5 -> 6
MIN_REPEATS = 2              # what a budget-reduced cell still runs
BUDGET_MIN = 60.0            # default-mode GPU-time budget, minutes
SPIN_SECONDS = 8.0           # harness spin before the first row
ROW_SPIN_SECONDS = 2.0       # before every later row (scene builds idle it)
BIG_ROW_PX = 20_000_000      # empty both pools before every call above this
TIMING_FIELDS = {"alloc_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms"}

# Small rows for --quick: every kind and every cell mode, at most 1 Mpx.
QUICK_ROWS = ("sq_256", "disk_256", "serp_128", "serp_256", "comb_24",
              "two_sq_300", "random_1000", "png_blocks")

# ----------------------------------------------------------------- columns
# The Numba side of every chapter 1-5 column IS bench.py's runner; the
# Triton side calls the twin driver with the identical arguments.
NUMBA_COLS = {key: SimpleNamespace(group=group, label=label, kinds=kinds,
                                   runner=runner)
              for key, group, label, kinds, runner in bench.COLS}

TRITON_RUNNERS = {
    "ch01_ring": lambda c: tff1.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB, variant="ring"),
    "ch01_spill": lambda c: tff1.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB, variant="spill"),
    "ch02_split": lambda c: tff2.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB, kernel="split"),
    "ch02_global": lambda c: tff2.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB, kernel="global"),
    "ch02_dirsplit": lambda c: tff2.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB,
        kernel="dirsplit"),
    # the one argument that differs: see DEVIATIONS
    "ch02_pinned": lambda c: tff2.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=tff2.PINNED_TPB,
        kernel="pinned", placement="spread"),
    "ch03_conn4": lambda c: tff3.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB),
    "ch03_conn8": lambda c: tff3.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB, connectivity=8),
    "ch03_conn8_r2": lambda c: tff3.flood_fill(
        c["img"], c["sx"], c["sy"], threads_per_block=TPB, connectivity=8,
        radius=2),
    "ch04_seq": lambda c: tff4.flood_fill(
        c["img"], c["seeds"], mode="sequential", threads_per_block=TPB),
    "ch04_multi": lambda c: tff4.flood_fill(
        c["img"], c["seeds"], mode="multisource", threads_per_block=TPB),
    "ch05_merge": lambda c: tff5.flood_fill(
        c["img"], variant="seed_merge", threads_per_block=TPB),
    "ch05_ccl": lambda c: tff5.flood_fill(
        c["img"], variant="ccl_fill", threads_per_block=TPB),
    "ch05_fused_L8": lambda c: tff5.flood_fill(
        c["img"], variant="seed_merge", lattice=8, build="fused",
        threads_per_block=TPB),
    "ch05_r128_L8": lambda c: tff5.flood_fill(
        c["img"], variant="seed_merge", lattice=8, build="r128",
        threads_per_block=TPB),
    "ch05_split_L8": lambda c: tff5.flood_fill(
        c["img"], variant="seed_merge", lattice=8, build="split",
        threads_per_block=TPB),
    "ch05_split_I1": lambda c: tff5.flood_fill(
        c["img"], variant="seed_merge", lattice=1, interior=True,
        build="split", threads_per_block=TPB),
}

# Columns with no Triton runner: skipped on both sides, with the reason.
SKIPPED_COLUMNS = {
    key: (f"{reason}: bench.py STATIC_SKIPS. ch04 streams runs two "
          "persistent cooperative kernels at once, which can wedge the GPU "
          "(it deadlocks after ch03's cooperative kernels in one process); "
          "bench.py never runs it, so neither side runs it here")
    for key, reason in bench.STATIC_SKIPS.items()
}

DEVIATIONS = {
    "ch02_pinned": (
        "Numba pins 2 x 768-thread blocks (bench.py's argument); the twin "
        f"accepts only power-of-2 blocks and runs 2 x {tff2.PINNED_TPB} "
        "lanes (its PINNED_TPB), as the ch02 compare does. Outputs are "
        "compared; the row is comparable=false"),
}

# Chapter 6: bench_ch06.COLS, one contract each, on both engines.
CH06_COLS = {key: SimpleNamespace(group=group, label=label,
                                  kinds=("one", "two", "n"),
                                  contract=key[len("ch06_"):])
             for key, group, label in bench_ch06.COLS}

GPU_COLUMNS = list(NUMBA_COLS) + list(CH06_COLS)   # bench order, then ch06

NOT_COMPARED = {
    key: (f"{label}: a CPU bar ({group}); no GPU backend to pair")
    for key, group, label, _ in bench.CPU_COLS
}


def _family(col):
    return col[:4]                                   # "ch01" ... "ch06"


def _column_info(col):
    c = NUMBA_COLS.get(col) or CH06_COLS[col]
    return {"group": c.group, "label": c.label}


# ------------------------------------------------------------- the plan
_KIND_OF_BUILDER = {"_one": "one", "_one_seedless": "one", "_two": "two",
                    "_n": "n"}


def row_kind(build):
    """A ROWS builder's kind ("one", "two", "n"), without building it:
    bench.py makes every builder with _one / _one_seedless / _two / _n."""
    return _KIND_OF_BUILDER[build.__qualname__.split(".")[0]]


def cell_mode(col, kind):
    """bench_row's dispatch, as (mode, skip reason): mode is "measured",
    "loop", "est", "est_pair" or "skip"."""
    if col in CH06_COLS:
        return "measured", None
    if col in bench.STATIC_SKIPS:
        return "skip", bench.STATIC_SKIPS[col]
    kinds = NUMBA_COLS[col].kinds
    if kind in kinds:
        return "measured", None
    if kind == "two" and kinds == ("one",):
        return "loop", None
    if kind == "n" and kinds == ("one",):
        return "est", None
    if kind == "n" and kinds == ("two",):
        return "est_pair", None
    return "skip", "na"


def bench_skip_reason(exc):
    """bench.py's typed refusal for a column's first call, or None when
    bench.py would not have treated the exception as a skip. The order
    is bench.py's: NotImplementedError is a RuntimeError, so the
    RuntimeError branch sees it first there too."""
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return ("overflow" if any(w in msg for w in
                                  ("overflow", "capacity", "ring"))
                else f"error:{type(exc).__name__}")
    if isinstance(exc, NotImplementedError):
        return "unsupported"
    return None


def skip_reason(col, exc):
    if col in CH06_COLS:
        return bench_ch06._skip_reason(exc)      # it skips on any Exception
    return bench_skip_reason(exc)


# ------------------------------------------------------------- the scenes
def build_row(build, kind):
    """bench_row's context: the builder's dict, plus (N-blob rows) one
    seed per component from bench._component_seeds."""
    ctx = build()
    if ctx["kind"] != kind:
        raise RuntimeError(f"builder kind {ctx['kind']} != planned {kind}")
    if ctx["kind"] == "n":
        ctx["seeds"] = bench._component_seeds(ctx["img"])
        ctx["n_blobs"] = len(ctx["seeds"])
    return ctx


def loop_subs(ctx):
    """bench._cell_gpu_loop's calls: one one-blob job per seed."""
    return [{"kind": "one", "img": ctx["img"], "sx": sx, "sy": sy}
            for sx, sy in ctx["seeds"]]


def est_subs(ctx, pair=False):
    """bench._cell_gpu_est's sample and call count: (subs, n_calls)."""
    seeds = ctx["seeds"]
    n = len(seeds)
    w, h = ctx["img"].shape[0], ctx["img"].shape[1]
    k = bench.EST_SAMPLE_HUGE if w * h > 20_000_000 else bench.EST_SAMPLE
    if pair:
        n_calls = (n + 1) // 2
        pairs = [(seeds[i], seeds[i + 1]) for i in range(0, n - 1, 2)]
        idx = np.linspace(0, len(pairs) - 1,
                          min(k, len(pairs))).astype(int)
        subs = [{"kind": "two", "img": ctx["img"], "seeds": list(pairs[i])}
                for i in np.unique(idx)]
    else:
        n_calls = n
        idx = np.linspace(0, n - 1, min(k, n)).astype(int)
        subs = [{"kind": "one", "img": ctx["img"],
                 "sx": seeds[i][0], "sy": seeds[i][1]}
                for i in np.unique(idx)]
    return subs, n_calls


# ------------------------------------------------------ light results
def digest(a):
    """sha1 of an array's bytes, tagged with its dtype and shape."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha1()
    h.update(a.reshape(-1).view(np.uint8))
    return f"sha1:{h.hexdigest()}:{a.dtype.str}:{list(a.shape)}"


def _scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    return v


def _take(r, arrays=(), scalars=(), int64=()):
    det = {k: digest(getattr(r, k)) for k in arrays}
    # small traces: dtype-normalised (the backends may pick int32 / int64)
    det.update({k: digest(np.asarray(getattr(r, k)).astype(np.int64))
                for k in int64})
    det.update({k: _scalar(getattr(r, k)) for k in scalars})
    return det


def _det_ch01(r):
    """ch01 compare's same_result: every non-timing field (one block, so
    all of them are deterministic)."""
    det = {}
    for f in dataclasses.fields(r):
        if f.name in TIMING_FIELDS:
            continue
        v = getattr(r, f.name)
        det[f.name] = digest(v) if isinstance(v, np.ndarray) else _scalar(v)
    return det


def _det_ch02(r):
    """ch02 compare's _slim det (instrumented split / global / dirsplit;
    pinned is uninstrumented)."""
    det = _take(r, arrays=("img", "visited", "depth"),
                scalars=("levels", "filled"))
    if r.kernel != "pinned" and not r.bare:
        det.update(_take(r, int64=("level_sizes",), scalars=(
            "level_trace_truncated", "peak_level", "peak_occupancy",
            "processed", "cas_attempts")))
        if r.kernel in ("split", "global"):
            det.update(_take(r, int64=("level_sizes_per_block",), scalars=(
                "processed_b0", "processed_b1", "thread_util_pct",
                "warp_engagement_pct", "lane_efficiency_pct")))
        if r.kernel == "split":
            det["owner"] = digest(r.owner)
        elif r.kernel == "global":
            reached = r.visited == 1
            det["owner_census"] = np.bincount(
                r.owner[reached].astype(np.int64), minlength=2).tolist()
    return det


def _det_ch03(r):
    """ch03 compare's same() at an unpinned grid."""
    det = _take(r, arrays=("img", "visited", "depth"), scalars=(
        "levels", "filled", "processed", "interior", "peak_level",
        "peak_occupancy", "level_trace_truncated"))
    if not r.bare:
        det.update(_take(r, int64=("level_sizes",)))
        if r.connectivity == 4:
            det["cas_attempts"] = _scalar(r.cas_attempts)
    return det


def _det_ch04(r):
    """ch04 compare's same() at an unpinned grid."""
    det = _take(r, arrays=("img", "visited", "depth", "label"), scalars=(
        "filled", "filled_a", "filled_b", "levels", "levels_a", "levels_b",
        "processed", "interior"))
    if not r.bare and r.connectivity == 4 and r.radius == 1:
        det.update(_take(r, scalars=("cas_attempts", "model_bytes")))
    det["launches"] = [
        _take(ln, int64=("level_sizes",), scalars=(
            "filled", "levels", "peak_level", "peak_occupancy", "processed",
            "interior", "level_trace_truncated"))
        for ln in (r.launches or [])]
    return det


def _det_ch05(r):
    """ch05 compare's same() at an unpinned grid."""
    det = _take(r, arrays=("img", "visited", "depth", "label"), scalars=(
        "variant", "lattice", "interior", "build", "bare", "n_blobs",
        "filled", "levels"))
    if not r.bare:
        det.update(_take(r, int64=("level_sizes",), scalars=(
            "candidates", "union_done", "processed", "peak_level",
            "peak_occupancy", "level_trace_truncated",
            "cas_attempts" if r.variant == "seed_merge"
            else "union_attempts")))
    return det


_DET = {"ch01": _det_ch01, "ch02": _det_ch02, "ch03": _det_ch03,
        "ch04": _det_ch04, "ch05": _det_ch05}


def light(r, family):
    """What a driver call leaves behind: its two timings, the digests and
    scalars of its deterministic outputs, and its launch shape."""
    return SimpleNamespace(
        kernel_ms=float(r.kernel_ms), total_ms=float(r.total_ms),
        det=_DET[family](r),
        obs={"blocks": int(getattr(r, "blocks", 1)),
             "tpb": int(r.threads_per_block)})


def _call(runner, family, ctx, big):
    """One driver call reduced to its light result; the full result is
    dropped before the next call can allocate."""
    if big:
        gc.collect()
        free_device_memory()
    r = runner(ctx)
    out = light(r, family)
    del r
    return out


# ------------------------------------------------------------ chapter 6
def _ch06_side(ctx, backend):
    """The row's engine on one backend, built at its first call and kept
    for both ch06 cells of the row (released with the row)."""
    sides = ctx.setdefault("_ch06_sides", {})
    if backend not in sides:
        if "_ch06_capacity" not in ctx:
            # bench_ch06's sizing: max(8192, host run count * 1.05)
            ctx["_ch06_capacity"] = max(8192, int(
                bench_ch06._host_run_count(ctx["img"]) * 1.05))
        cls = _NumbaSide if backend == "numba" else _TritonSide
        sides[backend] = cls(ctx["img"], ctx["_ch06_capacity"])
    return sides[backend]


def _ch06_runner(backend, contract):
    """bench_ch06's protocol for one call: restore the pristine image,
    pack off the clock for "mask", synchronize, run(), synchronize.
    kernel_ms is the CUDA-event span, total_ms the host wall time."""
    import cupy as cp

    def run(ctx):
        side = _ch06_side(ctx, backend)
        _prepare(side, contract)
        t0 = time.perf_counter()
        names, events = side.run(contract)
        side.sync()
        t1 = time.perf_counter()
        kernel_ms = float(side.elapsed(events[0], events[-1]))
        counters = side.counters()
        if counters[RUN_OVERFLOW]:
            raise RuntimeError("run table overflowed during benchmark")
        n = int(counters[N_RUNS_USED])
        v = side.views
        det = {"counters": [int(x) for x in counters]}
        for k in ("img", "mask", "row_count", "row_off"):
            det[k] = digest(cp.asnumpy(v[k]))
        for k in ("run_x", "run_y0", "run_y1", "run_label"):
            det[f"{k}[:n]"] = digest(cp.asnumpy(v[k][:n]))
        ids = cp.arange(n, dtype=v["parent"].dtype)
        det["root_set"] = digest(cp.asnumpy(v["parent"][:n] == ids))
        grid = list(side.engine.grid)
        return SimpleNamespace(kernel_ms=kernel_ms,
                               total_ms=(t1 - t0) * 1000.0, det=det,
                               obs={"blocks": int(grid[0]),
                                    "tpb": int(grid[1])})
    return run


def _ch06_call(runner, _family, ctx, big):
    if big:
        gc.collect()
        free_device_memory()
    return runner(ctx)


# ------------------------------------------------------------- one cell
def run_cell(mode, runner, family, ctx, subs, n_calls, big):
    """One backend's whole job for one cell, as a light result."""
    call = _ch06_call if family == "ch06" else _call
    if mode == "measured":
        return call(runner, family, ctx, big)
    parts = [call(runner, family, sub, big) for sub in subs]
    if mode == "loop":
        kernel = sum(p.kernel_ms for p in parts)
        total = sum(p.total_ms for p in parts)
    else:   # est / est_pair: median per call x calls a full loop needs
        kernel = statistics.median(p.kernel_ms for p in parts) * n_calls
        total = statistics.median(p.total_ms for p in parts) * n_calls
    return SimpleNamespace(kernel_ms=kernel, total_ms=total,
                           det={"calls": [p.det for p in parts]},
                           obs=parts[0].obs)


def same(n, t):
    """Digests and scalars equal, field by field; detail names the first
    difference."""
    return _same(n.det, t.det, "")


def _same(a, b, path):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return False, f"{path or 'det'}: fields differ"
        for k in a:
            ok, d = _same(a[k], b[k], f"{path}.{k}" if path else k)
            if not ok:
                return ok, d
        return True, ""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, d = _same(x, y, f"{path}[{i}]")
            if not ok:
                return ok, d
        return True, ""
    if a != b:
        return False, f"{path}: numba {a!r} vs triton {b!r}"
    return True, ""


def _ascii(text):
    return str(text).replace("\u2014", "-").replace("\u2013", "-")


def make_case(row_key, note, col, mode, ctx, wall):
    """One grand-table cell as a harness Case. `wall` collects each side's
    wall seconds (warm-up and digests included) for the row record."""
    family = _family(col)
    if family == "ch06":
        contract = CH06_COLS[col].contract
        n_run = _ch06_runner("numba", contract)
        t_run = _ch06_runner("triton", contract)
    else:
        n_run = NUMBA_COLS[col].runner
        t_run = TRITON_RUNNERS[col]
    subs, n_calls = None, 1
    if mode == "loop":
        subs = loop_subs(ctx)
        n_calls = len(subs)
    elif mode in ("est", "est_pair"):
        subs, n_calls = est_subs(ctx, pair=mode == "est_pair")
    w, h = int(ctx["img"].shape[0]), int(ctx["img"].shape[1])
    big = w * h > BIG_ROW_PX
    state = {"numba_calls": 0}
    wall.update(numba=0.0, triton=0.0)

    def side(name, runner):
        def run():
            t0 = time.perf_counter()
            try:
                return run_cell(mode, runner, family, ctx, subs, n_calls, big)
            finally:
                wall[name] += time.perf_counter() - t0
        return run

    run_numba_raw = side("numba", n_run)
    run_triton = side("triton", t_run)

    def run_numba():
        state["numba_calls"] += 1
        try:
            return run_numba_raw()
        except Exception as exc:
            reason = skip_reason(col, exc)
            if state["numba_calls"] == 1 and reason is not None:
                # bench.py's skip, decided by the Numba side on the
                # first call; probe the Triton side once for the record
                state["skip"] = reason
                state["numba"] = _ascii(f"{type(exc).__name__}: {exc}")[:300]
                try:
                    run_triton()
                    state["triton"] = "ran (skipped anyway: bench.py's rule)"
                except Exception as exc2:
                    t_reason = skip_reason(col, exc2)
                    state["triton"] = _ascii(
                        f"{t_reason or 'not a typed skip'}: "
                        f"{type(exc2).__name__}: {exc2}")[:300]
            raise

    def info(rn, rt):
        out = {"numba_blocks": rn.obs["blocks"],
               "triton_blocks": rt.obs["blocks"],
               "numba_tpb": rn.obs["tpb"], "triton_tpb": rt.obs["tpb"]}
        det = rn.det
        if family == "ch06":
            c = det["counters"]
            out.update(n_runs=c[N_RUNS], n_blobs=c[N_BLOBS],
                       union_attempts=c[UNION_ATTEMPTS],
                       run_capacity=ctx.get("_ch06_capacity"))
        elif mode == "measured":
            out.update(filled=det.get("filled"), levels=det.get("levels"))
            if "n_blobs" in det:
                out["n_blobs"] = det["n_blobs"]
        elif mode == "loop":
            out["filled"] = sum(d["filled"] for d in det["calls"])
        return out

    cfg = {"column": col, "cell": mode,
           "tpb": TPB if col != "ch02_pinned" else
           {"numba": bench.ff2.PINNED_TPB, "triton": tff2.PINNED_TPB}}
    extra = {"row": row_key, "family": family, **_column_info(col),
             "est": mode in ("est", "est_pair")}
    if family == "ch06":
        # bench_ch06's blob check: 1 / 2 on one- / two-blob rows, else
        # the row's component count
        extra["expected_blobs"] = (bench_ch06.EXPECTED_BLOBS[ctx["kind"]]
                                   or ctx.get("n_blobs"))
    if subs is not None:
        extra["calls"] = n_calls
        if extra["est"]:
            extra["sample"] = len(subs)
    notes = note
    if col in DEVIATIONS:
        notes = f"{note}. {DEVIATIONS[col]}"
    case = Case(experiment=col, scene=row_key, config=cfg,
                run_numba=run_numba, run_triton=run_triton, same=same,
                pixels=w * h, info=info, notes=notes, extra=extra,
                comparable=col not in DEVIATIONS)
    return case, state


def finish_row(row, state, wall, col):
    """Harness row -> (row or None, skip record or None)."""
    if "error" in row and state.get("skip"):
        return None, {"row": row["scene"], "column": col,
                      "reason": state["skip"], "source": "runtime",
                      "numba": state.get("numba"),
                      "triton": state.get("triton")}
    row["wall_s"] = {k: round(v, 3) for k, v in wall.items()}
    info = row.get("info")
    if info:
        resolved = {"numba": info["numba_blocks"],
                    "triton": info["triton_blocks"]}
        tpbs = {"numba": info["numba_tpb"], "triton": info["triton_tpb"]}
        row["config"]["resolved_blocks"] = resolved
        row["config"]["resolved_tpb"] = tpbs
        row["comparable"] = (col not in DEVIATIONS
                             and resolved["numba"] == resolved["triton"]
                             and tpbs["numba"] == tpbs["triton"])
    if "error" in row:
        row["error"] = _ascii(row["error"])
    return row, None


# -------------------------------------------------------------- estimate
# Per-call host cost the Numba JSONs do not hold: allocations, the H2D
# image, the D2H outputs, the sha1 of the digested arrays, and (ch04) the
# driver's own host-side checks. Measured per side on this laptop (RTX
# 4060 Laptop, i9-13900H, WSL2) on random_1000, two_sq_2800 and
# png_blobs: ch01-ch03 9-12 ns/px, ch04 46-47, ch05 25-59 (the upper end
# under memory pressure at 81 Mpx), ch06 3-9; rows above BIG_ROW_PX also
# pay cudaMalloc on every call.
OVERHEAD_BASE_MS = 3.0
OVERHEAD_NS_PER_PX = {"ch01": 12.0, "ch02": 12.0, "ch03": 12.0,
                      "ch04": 47.0, "ch05": 40.0, "ch06": 6.0}
CH06_SETUP_NS_PER_PX = 10.0  # both engines and device images, once a row
# Triton kernel_ms / Numba kernel_ms assumed for the estimate (measured
# 1.0-1.6 on ch01-ch04 and the ch05 lattice builds; ccl_fill and the
# interior split build reach 4-6x, hence ch05's larger factor)
TRITON_KERNEL_FACTOR = {"ch05": 2.5}
TRITON_KERNEL_FACTOR_DEFAULT = 1.5
UNKNOWN_CALL_MS = 100.0      # a cell with no Numba reference
COMPILE_S = 30.0             # first use of every column (measured ~10 s)
ROW_BUILD_S = {"one": 1.0, "two": 2.0, "n": 4.0}
PNG_BUILD_S = 25.0           # decode + component seeds, 9000 x 9000


def load_reference():
    """{row: {"width", "height", "kind", "n_blobs", "cells"}} from the
    newest overview_*.json and ch06_overview_*.json (Numba runs)."""
    ref, sources = {}, []
    path = results_paths.newest_optional("overview_*.json",
                                         bench.RESULTS_DIR)
    if path:
        sources.append(os.path.basename(path))
        with open(path) as f:
            for r in json.load(f)["rows"]:
                ref[r["row"]] = {"width": r["width"], "height": r["height"],
                                 "kind": r["kind"],
                                 "n_blobs": r.get("n_blobs"),
                                 "cells": dict(r["cells"])}
    path = results_paths.newest_optional("ch06_overview_*.json",
                                         bench.RESULTS_DIR)
    if path:
        sources.append(os.path.basename(path))
        with open(path) as f:
            for r in json.load(f)["rows"]:
                entry = ref.setdefault(r["row"], {
                    "width": r["width"], "height": r["height"],
                    "kind": r["kind"], "n_blobs": r.get("n_blobs"),
                    "cells": {}})
                entry["cells"].update(r["cells"])
    return ref, sources


def _calls_per_run(mode, ref_row, px):
    if mode == "loop":
        return 2
    if mode in ("est", "est_pair"):
        k = bench.EST_SAMPLE_HUGE if px > 20_000_000 else bench.EST_SAMPLE
        n = ref_row.get("n_blobs") or k
        return min(k, n // 2 if mode == "est_pair" else n)
    return 1


def estimate_cell(row_key, col, mode, kind, repeats, ref):
    """Seconds one case should take, and its seconds per extra round."""
    if mode == "skip":
        return 0.0, 0.0
    ref_row = ref.get(row_key) or {}
    px = (ref_row.get("width", 4000) * ref_row.get("height", 4000))
    family = _family(col)
    calls = _calls_per_run(mode, ref_row, px)
    cell = (ref_row.get("cells") or {}).get(col) or {}
    overhead = (OVERHEAD_BASE_MS
                + px * OVERHEAD_NS_PER_PX[family] * 1e-6) / 1000.0
    if cell.get("skip"):
        # a typed refusal: the Numba warm call fails, the Triton probe runs
        return 2 * calls * overhead, 0.0
    ms = cell.get("ms")
    if ms is None:
        per_call = UNKNOWN_CALL_MS
    elif cell.get("est") or mode == "loop":
        per_call = ms / max(1, cell.get("calls", 1))
    else:
        per_call = ms
    factor = TRITON_KERNEL_FACTOR.get(family, TRITON_KERNEL_FACTOR_DEFAULT)
    run_n = calls * (per_call / 1000.0 + overhead)
    run_t = calls * (per_call * factor / 1000.0 + overhead)
    per_round = run_n + run_t
    return (1 + repeats) * per_round, per_round


def plan_budget(plan, repeats, ref, budget_s):
    """Estimate every case; past the budget, drop the heaviest cells to
    MIN_REPEATS until it fits. Returns (repeats per cell, estimate,
    reduced cells)."""
    per_cell = {}
    fixed = COMPILE_S
    for row_key, kind, build, cols in plan:
        fixed += (PNG_BUILD_S if row_key.startswith("png")
                  else ROW_BUILD_S[kind]) + ROW_SPIN_SECONDS
        ref_row = ref.get(row_key) or {}
        px = ref_row.get("width", 4000) * ref_row.get("height", 4000)
        if any(c in CH06_COLS and m != "skip" for c, m, _ in cols):
            fixed += 2 * px * CH06_SETUP_NS_PER_PX * 1e-9
        for col, mode, _ in cols:
            total, per_round = estimate_cell(row_key, col, mode, kind,
                                             repeats, ref)
            per_cell[(row_key, col)] = [total, per_round, repeats]
    est = fixed + sum(v[0] for v in per_cell.values())
    reduced = []
    if budget_s and est > budget_s and repeats > MIN_REPEATS:
        for key, v in sorted(per_cell.items(), key=lambda kv: -kv[1][1]):
            if est <= budget_s:
                break
            saved = (repeats - MIN_REPEATS) * v[1]
            if saved <= 0:
                continue
            v[0] -= saved
            v[2] = MIN_REPEATS
            est -= saved
            reduced.append({"row": key[0], "column": key[1],
                            "repeats": MIN_REPEATS,
                            "estimated_round_s": round(v[1], 2)})
    by_row = {}
    for (row_key, _), v in per_cell.items():
        by_row[row_key] = by_row.get(row_key, 0.0) + v[0]
    estimate = {
        "total_s": round(est, 1), "budget_s": budget_s,
        "fixed_s": round(fixed, 1),
        "by_row_s": {k: round(v, 1) for k, v in by_row.items()},
        "model": (
            "per case: (1 + repeats) x (Numba run + Triton run); a run is "
            "calls x (reference per-call kernel ms + host cost), with the "
            "Triton kernel assumed TRITON_KERNEL_FACTOR x Numba's; host "
            "cost per call = OVERHEAD_BASE_MS + pixels x "
            "OVERHEAD_NS_PER_PX[chapter] (allocations, transfers, sha1, "
            "driver host checks); plus COMPILE_S, scene builds, row spins "
            "and the ch06 engine setup"),
        "constants": {"overhead_base_ms": OVERHEAD_BASE_MS,
                      "overhead_ns_per_px": OVERHEAD_NS_PER_PX,
                      "ch06_setup_ns_per_px": CH06_SETUP_NS_PER_PX,
                      "triton_kernel_factor": {
                          **{f: TRITON_KERNEL_FACTOR_DEFAULT for f in
                             ("ch01", "ch02", "ch03", "ch04", "ch06")},
                          **TRITON_KERNEL_FACTOR},
                      "compile_s": COMPILE_S},
    }
    reps = {k: v[2] for k, v in per_cell.items()}
    return reps, estimate, reduced


# ------------------------------------------------------------------ main
def select(rows_arg, cols_arg, quick):
    all_rows = [r[0] for r in bench.ROWS]
    if rows_arg:
        rows = [r.strip() for r in rows_arg.split(",") if r.strip()]
        bad = [r for r in rows if r not in all_rows]
        if bad:
            raise SystemExit(f"unknown rows {bad}; known: {all_rows}")
    elif quick:
        rows = [r for r in all_rows if r in QUICK_ROWS]
    else:
        rows = all_rows
    known_cols = GPU_COLUMNS
    if cols_arg:
        cols = [c.strip() for c in cols_arg.split(",") if c.strip()]
        bad = [c for c in cols if c not in known_cols]
        if bad:
            raise SystemExit(f"unknown columns {bad}; known: {known_cols}")
    else:
        cols = list(known_cols)
    cols = [c for c in known_cols if c in cols]           # table order
    return [r for r in bench.ROWS if r[0] in rows], cols


def build_plan(rows, cols):
    """[(row, kind, build, [(col, mode, skip reason)])], plus the static
    skip records."""
    plan, skipped = [], []
    for key, _family_name, _note, _est, build in rows:
        kind = row_kind(build)
        entries = []
        for col in cols:
            mode, reason = cell_mode(col, kind)
            if mode == "skip":
                why = (SKIPPED_COLUMNS.get(col) or
                       "ch04's kernel takes exactly two blobs in two "
                       "components; a one-blob row is outside its input "
                       "space (bench.py 'na')")
                skipped.append({"row": key, "column": col, "reason": reason,
                                "source": "static", "why": why})
            entries.append((col, mode, reason))
        plan.append((key, kind, build, entries))
    return plan, skipped


def crosscheck(rows):
    """bench.py's per-row check: every completed, non-estimated ch01-ch05
    cell fills the same pixel count; bench_ch06's: the ch06 blob count is
    1 / 2 on one- / two-blob rows and the component count on N-blob rows
    (its painted-pixel count is covered by the img digest instead)."""
    fills, blobs = {}, {}
    for r in rows:
        info = r.get("info") or {}
        if "error" in r or r.get("est"):
            continue
        if info.get("filled") is not None:
            fills.setdefault(r["scene"], set()).add(info["filled"])
        if r.get("expected_blobs") is not None and "n_blobs" in info:
            blobs.setdefault(r["scene"], []).append(
                info["n_blobs"] == r["expected_blobs"])
    out = {}
    for scene in sorted(set(fills) | set(blobs)):
        f = fills.get(scene, set())
        ok = len(f) <= 1 and all(blobs.get(scene, []))
        out[scene] = {"status": "OK" if ok else "MISMATCH",
                      "filled": sorted(f),
                      "ch06_blobs_ok": all(blobs.get(scene, []))}
    return out


def _rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="small rows, 1 repeat, no spin, no JSON")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per cell (default {DEFAULT_REPEATS})")
    ap.add_argument("--rows", default=None, help="comma-separated row keys")
    ap.add_argument("--cols", default=None,
                    help="comma-separated column keys")
    ap.add_argument("--budget-min", type=float, default=BUDGET_MIN,
                    help="GPU-time budget in minutes (0: no limit; "
                         "ignored with --quick)")
    ap.add_argument("--no-write", action="store_true",
                    help="measure but write no JSON")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the plan and its time estimate, measure "
                         "nothing")
    args = ap.parse_args(argv)
    # the 1-, 2- and 48-block launches are the experiment, not a mistake
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    quick = args.quick
    repeats = args.repeats or (1 if quick else DEFAULT_REPEATS)
    repeats += repeats % 2                  # what the harness runs
    write = not (quick or args.no_write)
    spin = 0.0 if quick else SPIN_SECONDS
    row_spin = 0.0 if quick else ROW_SPIN_SECONDS

    rows, cols = select(args.rows, args.cols, quick)
    plan, skipped = build_plan(rows, cols)
    ref, ref_sources = load_reference()
    budget_s = 0.0 if quick else args.budget_min * 60.0
    reps, estimate, reduced = plan_budget(plan, repeats, ref, budget_s)
    estimate["reference"] = ref_sources
    n_cases = sum(1 for *_, entries in plan
                  for _, mode, _ in entries if mode != "skip")
    print(f"overview twin: {len(plan)} rows x {len(cols)} columns, "
          f"{n_cases} cases, {len(skipped)} static skips, "
          f"repeats={repeats}")
    print(f"estimated GPU time: {estimate['total_s'] / 60:.1f} min "
          f"(budget {budget_s / 60:.0f} min; "
          f"{len(reduced)} cells reduced to {MIN_REPEATS} repeats)")
    if args.estimate_only:
        for k, v in estimate["by_row_s"].items():
            print(f"  {k:16s} {v / 60:6.1f} min")
        for r in reduced:
            print(f"  reduced: {r['row']} {r['column']} "
                  f"({r['estimated_round_s']:.1f} s per round)")
        return {"rows": [], "meta": {"estimate": estimate,
                                     "reduced_repeats": reduced,
                                     "skipped_cells": skipped}}

    caps = []
    if reduced:
        caps.append(f"{len(reduced)} heaviest cells run {MIN_REPEATS} "
                    f"repeats instead of {repeats} to fit the "
                    f"{budget_s / 60:.0f}-minute budget "
                    "(meta.reduced_repeats)")
    if quick:
        caps.insert(0, "QUICK smoke run: small rows, 2 rounds, no spin; "
                       "not a measurement")
    missing_png = [k for k, f in (("png_blocks", "input_blocks.png"),
                                  ("png_blobs", "input_blobs.png"))
                   if k not in [r[0] for r in bench.ROWS]]
    meta = {
        "mirrors": ["overview/bench.py", "overview/bench_ch06.py"],
        "quick": quick, "complete": False,
        "rows": [p[0] for p in plan], "columns": {
            c: {**_column_info(c), "family": _family(c),
                "triton": ("skipped" if c in SKIPPED_COLUMNS else
                           DEVIATIONS.get(c, "identical arguments"))}
            for c in cols},
        "cell_rules": (
            "measured: one call; loop: one-blob kernel on a two-blob row, "
            "one call per blob, summed; est: one-blob kernel on an N-blob "
            "row, median per-call ms over bench.py's k-blob sample "
            f"(k={bench.EST_SAMPLE}, {bench.EST_SAMPLE_HUGE} past 20 Mpx) "
            "x the blob count; est_pair: ch04 on an N-blob row, the same "
            "over blob pairs x the pair count. Both backends run the same "
            "sample. est rows carry est=true, calls and sample"),
        "outputs": (
            "deterministic outputs reduced to sha1 digests (dtype, shape) "
            "right after each call, per chapter compare's same() minus "
            "grid-dependent fields; loop and est cells compare every call"),
        "grids": (
            "bench.py's arguments: blocks=None for ch03-ch05, so each "
            "backend runs its own cooperative grid; config.resolved_blocks "
            "and resolved_tpb come from the warm-up results, and rows where "
            "they differ are comparable=false"),
        "timing": (
            "ch01-ch05: the drivers' kernel_ms (perf_counter + "
            "synchronize) and total_ms; ch06: the CUDA-event span of run() "
            "and the host wall time of run() + synchronize, after restore "
            "and (mask) pack off the clock. Rows above "
            f"{BIG_ROW_PX // 1_000_000} Mpx empty both memory pools before "
            "every call, so total_ms there includes cudaMalloc on both "
            "sides"),
        "spin": {"first_s": spin, "per_row_s": row_spin},
        "skipped_cells": skipped,
        "not_compared": {**NOT_COMPARED, **{
            k: "row absent: its input PNG is missing (bench.py adds it only "
               "when present)" for k in missing_png}},
        "deviations": dict(DEVIATIONS),
        "est_sample": {"k": bench.EST_SAMPLE, "k_huge": bench.EST_SAMPLE_HUGE},
        "estimate": estimate, "reduced_repeats": reduced, "caps": caps,
    }

    stamp = None
    path = None
    doc = None
    t_start = time.perf_counter()
    first = True
    for row_key, kind, build, entries in plan:
        live = [(col, mode) for col, mode, _ in entries if mode != "skip"]
        if not live:
            continue
        note = next(r[2] for r in bench.ROWS if r[0] == row_key)
        print(f"\n{row_key} ({note}): building the scene...", flush=True)
        ctx = build_row(build, kind)
        groups = []                       # consecutive cells, same repeats
        for col, mode in live:
            r = reps[(row_key, col)]
            wall = {}
            case, state = make_case(row_key, note, col, mode, ctx, wall)
            if groups and groups[-1][0] == r:
                groups[-1][1].append((case, state, wall, col))
            else:
                groups.append((r, [(case, state, wall, col)]))
        for r, members in groups:
            s = spin if first else row_spin
            first = False
            d = run_cases(UNIT, [m[0] for m in members], repeats=r,
                          meta=meta, write=False, spin_seconds=s,
                          log=lambda m, k=row_key: print(f"  {k} {m}",
                                                         flush=True))
            measured = d["rows"]
            if doc is None:
                doc = d
                doc["rows"] = []
                doc["repeats"] = repeats
                stamp = doc["created_utc"]
            for row, (case, state, wall, col) in zip(measured, members):
                if r != repeats:
                    row["repeats"] = r
                kept, skip = finish_row(row, state, wall, col)
                if skip:
                    skipped.append(skip)
                    print(f"  {row_key} {col}: skip ({skip['reason']}); "
                          f"triton: {skip['triton']}")
                else:
                    doc["rows"].append(kept)
        ctx.clear()
        del ctx, groups
        gc.collect()
        free_device_memory()
        if doc is not None and write:      # crash-safe per row
            meta["peak_host_rss_mb"] = _rss_mb()
            meta["elapsed_s"] = round(time.perf_counter() - t_start, 1)
            meta["crosscheck"] = crosscheck(doc["rows"])
            path = os.path.join(results_paths.results_dir("triton_twins",
                                                          UNIT),
                                f"compare_{stamp}.json")
            with open(path, "w") as f:
                json.dump(doc, f, indent=1)

    if doc is None:
        print("nothing to measure: every selected cell is a skip")
        return {"rows": [], "meta": meta}
    meta["complete"] = True
    meta["peak_host_rss_mb"] = _rss_mb()
    meta["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    meta["crosscheck"] = crosscheck(doc["rows"])
    doc["meta"] = meta
    if write:
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
        doc["path"] = path
        print(f"wrote {path}")

    print(f"\n{len(doc['rows'])} rows measured, {len(skipped)} skipped "
          f"cells, {meta['elapsed_s'] / 60:.1f} min, peak RSS "
          f"{meta['peak_host_rss_mb']} MB")
    bad = [r for r in doc["rows"]
           if "error" in r or not r.get("outputs_equal", False)]
    for r in bad:
        print(f"  NOT OK {r['scene']} {r['experiment']}: "
              f"{r.get('error') or r.get('mismatch_detail')}")
    print(f"{len(bad)} rows with an error or a mismatch")
    return doc


if __name__ == "__main__":
    _doc = main()
    raise SystemExit(1 if any("error" in r or not r.get("outputs_equal")
                              for r in _doc["rows"]) else 0)
