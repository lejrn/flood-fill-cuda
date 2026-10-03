"""Numba vs Triton on the grand table: every overview row x every GPU column.

The twin of overview/bench.py (chapters 1-5) and overview/bench_ch06.py
(chapter 6). Rows, scene builders, seed helpers, column runners, the
est sample, the per-blob loop and the typed skip rule are bench.py's
own code, called by import (the sample and the loop are read off
bench._cell_gpu_est / _cell_gpu_loop driven with a recording runner,
the skip reason off bench._cell_gpu driven with a raising one), so the
grid cannot drift. Each Numba column runner gets a Triton counterpart
that calls the twin driver with the same arguments (CALL_KW, checked
against bench.py's runners by test_overview). One Case per row x column
goes through compare/harness.py (warm-up, alternating rounds, outputs
compared on every run), and the JSON lands in
results/triton_twins/overview/compare_<UTC>.json, which
compare/summary.py rolls up as one more unit.

Cell semantics, kept from bench.py and applied the same way on both
sides:

  measured   the column's job is the row's job: one driver call.
  loop       a one-blob kernel (ch01-ch03) on a two-blob row: one call
             per blob, kernel_ms and total_ms summed (MEASURED).
  est        a one-blob kernel on an N-blob row: bench.py's k-blob
             sample (16 blobs, 6 past 20 Mpx), median per-call ms x the
             call count a full loop needs (one per blob). ch04 on an
             N-blob row samples blob PAIRS, one call per pair. est=True
             is recorded in the row, with calls, sample and per_call_ms;
             a full loop would take hours.
  skip       bench.py's own skips, on both sides: "na" (ch04 needs
             exactly two blobs, a one-blob row is outside its input
             space), STATIC_SKIPS (ch04 streams: two persistent kernels
             at once can wedge the GPU, and bench.py skips it as
             "unsupported"), and bench.py's typed runtime refusals on
             the first call ("overflow" for the ch01 ring, else
             "error:<type>"; ch06 uses bench_ch06's rule). A runtime
             skip is decided by the Numba side, as bench.py decides it,
             and only for an exception the driver (or the ch06 engine)
             raised: a failure in this file's own digest or bookkeeping
             code is an error row. The Triton side is then probed once
             and what it did is recorded next to the reason. Skips never
             become harness rows: they are listed in meta.skipped_cells,
             and an "error:" skip makes the run NOT OK.

Grids. bench.py launches the ch03, ch04 and ch05 columns with
blocks=None: each backend's own co-resident maximum, which differs (at
tpb 256 here: conn4 48 vs 144, conn8 48 vs 120, ch04 48 vs 144, ccl
48 vs 72, fused_L8 24 vs 48). Both sides here run the common grid
min(Numba capacity, Triton capacity), as the ch03 compare's suite does.
On this GPU that minimum is always Numba's, so the Numba cell keeps
bench.py's own launch (config.numba_is_bench_launch) and the twin gets
an explicit blocks= with the same grid. The caps and the pinned grid are
in each row's config and in meta.grids; resolved grids and block sizes
come from the warm-up results, and a row whose two launches still differ
is comparable=false.

ch02_pinned: Numba pins 2 x 768 threads (bench.py's argument); the twin
accepts only power-of-2 blocks and runs its PINNED_TPB (2 x 512), as the
ch02 compare does. That row is kept as bench.py's cell and is
comparable=false. ch02_pinned_matched is the like-for-like row beside
it: Numba's pinned kernel at 2 x 512 (its PINNED_TPB swapped for the
call, the ch02 compare's matched case) vs the twin at 2 x 512.

Outputs: every call's deterministic outputs are reduced to sha1 digests
(with dtype and shape) right after the call, and the arrays are dropped.
A run returns a LIGHT result (kernel_ms, total_ms, digests, scalars), so
the harness never holds two full outputs (one ch05 result at 81 Mpx is
about 1.7 GB of host arrays). The fields digested per chapter are the
ones each chapter compare's same() treats as deterministic, minus the
per-block ones. ch06 cells also count, on the device, the painted pixels
(any channel differs from the pristine image) and the red pixels left.

Crosscheck, per row, printed as bench.py prints it: every completed,
non-estimated cell (ch06 included) fills the same pixel count; and
bench_ch06's absolute checks on the ch06 cells: painted == the scene's
red pixel count, no red pixel left, and the blob count is 1 / 2 on one- /
two-blob rows and the component count on N-blob rows. A MISMATCH makes
the run NOT OK.

Timing is the drivers' own: kernel_ms is the perf_counter + synchronize
bracket for ch01-ch05 and the CUDA-event span of run() for ch06 (pack off
the clock for the mask contract, restore first: bench_ch06's protocol).
ch06 cells run bench_ch06.ROUNDS (9, rounded to 10 by the harness);
the others bench.py's GPU_REPEATS (5, rounded to 6). Memory pools: every
row keeps CuPy's pool warm within a case (the harness frees both pools
between cases), so total_ms means the same thing on every row. On the
two rows above 20 Mpx, Python garbage is collected and Numba's deferred
frees are flushed before every call (Numba never pools, so its timings
do not change). VRAM allows it: measured on asym_4000_800 and scaled to
81 Mpx, the worst case (ch05 split_I1, both sides' buffers alive) needs
about 3.7 GB above a 1.1 GB baseline, of 8 GB.

Clocks: the harness spins 8 s before the first row; every later row
spins ROW_SPIN_SECONDS, and the ch05 and ch06 cells of a row form their
own groups behind a PHASE_SPIN_SECONDS spin, since the est cells before
them are mostly host work and let the clock sag. Each row records
clocks_before (nvidia-smi before its warm-up) and clocks_after.

Not compared: the two CPU columns (pure_python, njit) have no GPU
backend to pair, and rows whose input PNG is missing (bench.py adds
them only when present) are absent from both tables.

Budget: the default run (17 rows x 21 columns, 6 repeats, ch06 10) is
estimated before it starts from the Numba overview JSONs' per-cell ms
plus a per-call host-cost model (allocations, transfers, sha1, the
drivers' host checks) measured on this laptop. If the estimate exceeds
--budget-min (60), the heaviest cells drop to 2 repeats until it fits,
and meta.reduced_repeats lists them. --estimate-only prints the plan per
row without measuring.

Memory: peak host RSS measured 3.5 GB on png_blobs (the 243 MB scene, a
ch05 result of about 1.7 GB inside the Numba or Triton driver before it
is digested, the component-seed labelling, and the CUDA, CuPy and Triton
runtimes). Before building a row above 20 Mpx (its size read from the
Numba reference JSON), MemAvailable is read from /proc/meminfo; below
HOST_BYTES_PER_PX x pixels + HOST_MARGIN_MB the row's cells become typed
"host-memory" skips (and a cap says so) instead of risking an OOM kill
or a swap thrash. --no-mem-check turns that off.

Crash safety: after every row the document so far goes to
results/triton_twins/overview/partial.json (meta.complete=false, plus an
INCOMPLETE cap), a name summary.py never globs. compare_<UTC>.json is
written only when the run completes, so a killed run never becomes the
overview unit; partial.json then holds a small marker naming it.

Run:
    python -m flood_fill_cuda.triton_twins.compare.overview [--quick] [--repeats N] [--rows a,b] [--cols x,y]
        [--budget-min M] [--estimate-only] [--no-write] [--no-mem-check]

--quick runs the small rows only (at most 1 Mpx), 1 repeat (2 after
rounding), no spin, no JSON. The exit status is 1 when any row has an
error or unequal outputs, a crosscheck fails, or a skip is "error:".
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import inspect
import json
import resource
import statistics
import time
import warnings
from types import SimpleNamespace

import numpy as np
from numba import cuda
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
    Case, free_device_memory, gpu_clocks, run_cases,
)

UNIT = "overview"
TPB = bench.TPB
DEFAULT_REPEATS = bench.GPU_REPEATS + bench.GPU_REPEATS % 2   # 5 -> 6
CH06_REPEATS = bench_ch06.ROUNDS + bench_ch06.ROUNDS % 2      # 9 -> 10
MIN_REPEATS = 2              # what a budget-reduced cell still runs
BUDGET_MIN = 60.0            # default-mode GPU-time budget, minutes
SPIN_SECONDS = 8.0           # harness spin before the first row
ROW_SPIN_SECONDS = 2.0       # before every later row (scene builds idle it)
PHASE_SPIN_SECONDS = 3.0     # before a row's ch05 group and its ch06 group
BIG_ROW_PX = 20_000_000      # above this: collect garbage before each call
TIMING_FIELDS = {"alloc_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms"}
# Host-memory preflight for rows above BIG_ROW_PX. png_blobs (81 Mpx)
# peaked at 3.53 GB RSS; the --quick rows peak at 0.8 GB with every
# runtime loaded and every kernel compiled: about 34 bytes per pixel on
# top of that baseline (3.1 GB needed for png_blobs with the margin).
HOST_BYTES_PER_PX = 34
HOST_MARGIN_MB = 512
PARTIAL_NAME = "partial.json"

# Small rows for --quick: every kind and every cell mode, at most 1 Mpx.
QUICK_ROWS = ("sq_256", "disk_256", "serp_128", "serp_256", "comb_24",
              "two_sq_300", "random_1000", "png_blocks")

# ----------------------------------------------------------------- columns
# The Numba side of every chapter 1-5 column IS bench.py's runner.
NUMBA_COLS = {key: SimpleNamespace(group=group, label=label, kinds=kinds,
                                   runner=runner)
              for key, group, label, kinds, runner in bench.COLS}

_NUMBA_FF = {"ch01": bench.ff1, "ch02": bench.ff2, "ch03": bench.ff3,
             "ch04": bench.ff4, "ch05": bench.ff5}
_TRITON_FF = {"ch01": tff1, "ch02": tff2, "ch03": tff3, "ch04": tff4,
              "ch05": tff5}

# The keyword arguments bench.py's runner passes for each column, besides
# the image, the seed(s) and threads_per_block. test_overview records
# bench.py's runners and the twins' calls and checks they match.
CALL_KW = {
    "ch01_ring": {"variant": "ring"},
    "ch01_spill": {"variant": "spill"},
    "ch02_split": {"kernel": "split"},
    "ch02_global": {"kernel": "global"},
    "ch02_dirsplit": {"kernel": "dirsplit"},
    "ch02_pinned": {"kernel": "pinned", "placement": "spread"},
    "ch03_conn4": {},
    "ch03_conn8": {"connectivity": 8},
    "ch03_conn8_r2": {"connectivity": 8, "radius": 2},
    "ch04_seq": {"mode": "sequential"},
    "ch04_multi": {"mode": "multisource"},
    "ch05_merge": {"variant": "seed_merge"},
    "ch05_ccl": {"variant": "ccl_fill"},
    "ch05_fused_L8": {"variant": "seed_merge", "lattice": 8,
                      "build": "fused"},
    "ch05_r128_L8": {"variant": "seed_merge", "lattice": 8,
                     "build": "r128"},
    "ch05_split_L8": {"variant": "seed_merge", "lattice": 8,
                      "build": "split"},
    "ch05_split_I1": {"variant": "seed_merge", "lattice": 1,
                      "interior": True, "build": "split"},
}


def _family(col):
    return col[:4]                                   # "ch01" ... "ch06"


def driver_runner(mod, col, tpb, blocks=None):
    """runner(ctx) for one backend's driver with bench.py's arguments for
    `col`; blocks pins the cooperative grid (ch03-ch05)."""
    family = _family(col)
    kw = dict(CALL_KW[col])
    if blocks is not None:
        kw["blocks"] = int(blocks)
    if family in ("ch01", "ch02", "ch03"):
        return lambda c: mod.flood_fill(c["img"], c["sx"], c["sy"],
                                        threads_per_block=tpb, **kw)
    if family == "ch04":
        return lambda c: mod.flood_fill(c["img"], c["seeds"],
                                        threads_per_block=tpb, **kw)
    return lambda c: mod.flood_fill(c["img"], threads_per_block=tpb, **kw)


# The twins with bench.py's arguments (blocks=None). ch02_pinned is the one
# argument that cannot match: see DEVIATIONS.
TRITON_RUNNERS = {
    col: driver_runner(_TRITON_FF[_family(col)], col,
                       tff2.PINNED_TPB if col == "ch02_pinned" else TPB)
    for col in CALL_KW}

# Cooperative-grid columns: both sides run min(Numba cap, Triton cap).
GRID_COLS = tuple(c for c in CALL_KW if _family(c) in ("ch03", "ch04",
                                                       "ch05"))
_CAPS = {}

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
        "compared; the row is comparable=false (ch02_pinned_matched is "
        "the like-for-like row)"),
}

MATCHED = "ch02_pinned_matched"
MATCHED_OF = {MATCHED: "ch02_pinned"}
MATCHED_NOTE = (
    f"matched: Numba's pinned kernel at 2 x {tff2.PINNED_TPB} (its "
    "PINNED_TPB swapped for the call, the ch02 compare's matched case) vs "
    f"the twin at 2 x {tff2.PINNED_TPB}; not a bench.py cell, the "
    "comparable row beside ch02_pinned")
GRID_NOTE = (
    "grid pinned to min(Numba cap, Triton cap) on both sides "
    "(config.blocks); bench.py's blocks=None gives each backend its own cap")

# Chapter 6: bench_ch06.COLS, one contract each, on both engines.
CH06_COLS = {key: SimpleNamespace(group=group, label=label,
                                  kinds=("one", "two", "n"),
                                  contract=key[len("ch06_"):])
             for key, group, label in bench_ch06.COLS}

GPU_COLUMNS = list(NUMBA_COLS) + list(CH06_COLS)   # bench order, then ch06
TABLE_COLUMNS = []                                 # + the matched row
for _c in GPU_COLUMNS:
    TABLE_COLUMNS.append(_c)
    if _c == MATCHED_OF[MATCHED]:
        TABLE_COLUMNS.append(MATCHED)

NOT_COMPARED = {
    key: (f"{label}: a CPU bar ({group}); no GPU backend to pair")
    for key, group, label, _ in bench.CPU_COLS
}


def _column_info(col):
    if col == MATCHED:
        c = NUMBA_COLS[MATCHED_OF[col]]
        return {"group": c.group,
                "label": f"pinned (spread, tpb {tff2.PINNED_TPB} both)"}
    c = NUMBA_COLS.get(col) or CH06_COLS[col]
    return {"group": c.group, "label": c.label}


def grid_caps(col):
    """Co-resident capacity of a grid column's kernel at TPB on each
    backend (what blocks=None resolves to), and the grid both run."""
    if col not in _CAPS:
        fam = _family(col)
        caps = {}
        for name, mod in (("numba", _NUMBA_FF[fam]),
                          ("triton", _TRITON_FF[fam])):
            params = inspect.signature(mod.max_blocks).parameters
            kw = {k: v for k, v in CALL_KW[col].items() if k in params}
            caps[name] = int(mod.max_blocks(threads_per_block=TPB, **kw))
        _CAPS[col] = {"caps": caps, "pinned": min(caps.values()),
                      "numba_is_bench_launch":
                          min(caps.values()) == caps["numba"]}
    return _CAPS[col]


@contextlib.contextmanager
def _numba_pinned_tpb(tpb):
    """Numba's pinned driver checks threads_per_block against its module
    constant PINNED_TPB (and warms up with it); the kernel itself does not
    read it, so the constant is swapped for the call and restored."""
    old = bench.ff2.PINNED_TPB
    bench.ff2.PINNED_TPB = tpb
    try:
        yield
    finally:
        bench.ff2.PINNED_TPB = old


def runners(col):
    """(Numba runner, Triton runner, grid record or None) for a ch01-ch05
    column, read at case time (so tests can patch the tables)."""
    fam = _family(col)
    if col == MATCHED:
        base = driver_runner(bench.ff2, MATCHED_OF[col], tff2.PINNED_TPB)

        def numba_matched(c):
            with _numba_pinned_tpb(tff2.PINNED_TPB):
                return base(c)
        return numba_matched, TRITON_RUNNERS[MATCHED_OF[col]], None
    if col not in GRID_COLS:
        return NUMBA_COLS[col].runner, TRITON_RUNNERS[col], None
    grid = grid_caps(col)
    pin = grid["pinned"]
    if grid["numba_is_bench_launch"]:
        n_run = NUMBA_COLS[col].runner       # blocks=None resolves to pin
    else:
        n_run = driver_runner(_NUMBA_FF[fam], col, TPB, blocks=pin)
    t_run = driver_runner(_TRITON_FF[fam], col, TPB, blocks=pin)
    return n_run, t_run, grid


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
    col = MATCHED_OF.get(col, col)
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
    bench.py would not treat the exception as a skip: bench._cell_gpu
    itself is driven with a runner that raises `exc`. (Its except
    clauses are the same in _cell_gpu_loop and _cell_gpu_est.)"""
    def raising(_ctx):
        raise exc
    try:
        return bench._cell_gpu(raising, None)["skip"]
    except Exception as got:
        if got is exc:
            return None                     # bench.py would have crashed
        raise


def skip_reason(col, exc):
    if col in CH06_COLS:
        return bench_ch06._skip_reason(exc)      # it skips on any Exception
    return bench_skip_reason(exc)


_DRIVER_MARK = "_overview_from_driver"


def _mark_driver(exc):
    """Tag an exception as raised by a driver or engine call, the only
    kind a runtime skip may classify."""
    try:
        setattr(exc, _DRIVER_MARK, True)
    except Exception:
        pass


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


class _Recorded(Exception):
    """Ends a recording pass. Not a RuntimeError, so bench.py's typed
    except clauses let it through."""


def _recorded_subs(cell_fn, ctx, **kw):
    """The sub-contexts bench.py's cell function calls its runner with,
    in order, one pass (a repeated sub object ends the recording), and
    the cell it returned (None when the recording ended it)."""
    seen = []

    def runner(sub):
        if any(s is sub for s in seen):
            raise _Recorded
        seen.append(sub)
        return SimpleNamespace(kernel_ms=1.0, total_ms=1.0, filled=0,
                               levels=0)
    try:
        cell = cell_fn(runner, ctx, **kw)
    except _Recorded:
        cell = None
    return seen, cell


def loop_subs(ctx):
    """bench._cell_gpu_loop's calls: one one-blob job per seed."""
    return _recorded_subs(bench._cell_gpu_loop, ctx)[0]


def est_subs(ctx, pair=False):
    """bench._cell_gpu_est's sample and call count: (subs, n_calls)."""
    subs, cell = _recorded_subs(bench._cell_gpu_est, ctx, pair=pair)
    return subs, cell["calls"]


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
    """ch03 compare's same() without the per-block fields."""
    det = _take(r, arrays=("img", "visited", "depth"), scalars=(
        "levels", "filled", "processed", "interior", "peak_level",
        "peak_occupancy", "level_trace_truncated"))
    if not r.bare:
        det.update(_take(r, int64=("level_sizes",)))
        if r.connectivity == 4:
            det["cas_attempts"] = _scalar(r.cas_attempts)
    return det


def _det_ch04(r):
    """ch04 compare's same() without the per-block fields."""
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
    """ch05 compare's same() without the per-block fields."""
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


def _settle(big):
    """Before every call on a row above BIG_ROW_PX: collect the previous
    call's garbage and flush Numba's deferred frees (Numba never pools,
    so its timings do not change). CuPy's pool stays warm, as it does on
    every smaller row; the harness empties both pools between cases."""
    if big:
        gc.collect()
        try:
            cuda.current_context().deallocations.clear()
        except Exception:
            pass


def _call(runner, family, ctx, big):
    """One driver call reduced to its light result; the full result is
    dropped before the next call can allocate."""
    _settle(big)
    try:
        r = runner(ctx)
    except Exception as exc:
        _mark_driver(exc)
        raise
    out = light(r, family)
    del r
    return out


# ------------------------------------------------------------ chapter 6
_PAINT_SRC = r"""
extern "C" __global__
void paint_counts(const unsigned char* img, const unsigned char* pristine,
                  long long n, unsigned long long* out) {
    unsigned long long painted = 0, red = 0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const unsigned char* p = img + 3 * i;
        const unsigned char* q = pristine + 3 * i;
        painted += (p[0] != q[0]) | (p[1] != q[1]) | (p[2] != q[2]);
        red += (p[0] == 255) & (p[1] == 0) & (p[2] == 0);
    }
    for (int o = 16; o > 0; o >>= 1) {
        painted += __shfl_down_sync(0xffffffffu, painted, o);
        red += __shfl_down_sync(0xffffffffu, red, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&out[0], painted);
        atomicAdd(&out[1], red);
    }
}
"""
_PAINT_KERNEL = []


def paint_counts(img, pristine):
    """On the device: (pixels where any channel differs from pristine,
    pure red (255, 0, 0) pixels left in img), bench_ch06's two counts."""
    import cupy as cp

    if not _PAINT_KERNEL:
        _PAINT_KERNEL.append(cp.RawKernel(_PAINT_SRC, "paint_counts"))
    img = cp.ascontiguousarray(cp.asarray(img))
    pristine = cp.ascontiguousarray(cp.asarray(pristine))
    if (img.dtype != cp.uint8 or img.ndim != 3 or img.shape[2] != 3
            or img.shape != pristine.shape or pristine.dtype != cp.uint8):
        raise ValueError(f"paint_counts wants two (w, h, 3) uint8 images, "
                         f"got {img.shape} {img.dtype} and {pristine.shape} "
                         f"{pristine.dtype}")
    n = int(img.shape[0] * img.shape[1])
    out = cp.zeros(2, dtype=cp.uint64)
    blocks = max(1, min(1024, (n + 255) // 256))
    _PAINT_KERNEL[0]((blocks,), (256,), (img, pristine, np.int64(n), out))
    painted, red = (int(x) for x in out.get())
    return painted, red


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
        try:                              # the engine's part: may skip
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
        except Exception as exc:
            _mark_driver(exc)
            raise
        # this file's bookkeeping: an exception here is an error row
        n = int(counters[N_RUNS_USED])
        v = side.views
        det = {"counters": [int(x) for x in counters]}
        for k in ("img", "mask", "row_count", "row_off"):
            det[k] = digest(cp.asnumpy(v[k]))
        for k in ("run_x", "run_y0", "run_y1", "run_label"):
            det[f"{k}[:n]"] = digest(cp.asnumpy(v[k][:n]))
        ids = cp.arange(n, dtype=v["parent"].dtype)
        det["root_set"] = digest(cp.asnumpy(v["parent"][:n] == ids))
        det["painted"], det["still_red"] = paint_counts(v["img"],
                                                         side.pristine)
        if "_red_px" not in ctx:
            ctx["_red_px"] = paint_counts(side.pristine, side.pristine)[1]
        grid = list(side.engine.grid)
        return SimpleNamespace(kernel_ms=kernel_ms,
                               total_ms=(t1 - t0) * 1000.0, det=det,
                               obs={"blocks": int(grid[0]),
                                    "tpb": int(grid[1])})
    return run


def _ch06_call(runner, _family, ctx, big):
    _settle(big)
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
    grid = None
    if family == "ch06":
        contract = CH06_COLS[col].contract
        n_run = _ch06_runner("numba", contract)
        t_run = _ch06_runner("triton", contract)
    else:
        n_run, t_run, grid = runners(col)
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
        if state["numba_calls"] == 1:
            state["clocks_before"] = gpu_clocks()
        try:
            return run_numba_raw()
        except Exception as exc:
            reason = (skip_reason(col, exc)
                      if getattr(exc, _DRIVER_MARK, False) else None)
            if state["numba_calls"] == 1 and reason is not None:
                # bench.py's skip, decided by the Numba side on the
                # first call; probe the Triton side once for the record
                state["skip"] = reason
                state["numba"] = _ascii(f"{type(exc).__name__}: {exc}")[:300]
                try:
                    run_triton()
                    state["triton"] = "ran (skipped anyway: bench.py's rule)"
                except Exception as exc2:
                    t_reason = (skip_reason(col, exc2)
                                if getattr(exc2, _DRIVER_MARK, False)
                                else None)
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
                       run_capacity=ctx.get("_ch06_capacity"),
                       filled=det["painted"], still_red=det["still_red"],
                       red_px=ctx.get("_red_px"))
        elif mode == "measured":
            out.update(filled=det.get("filled"), levels=det.get("levels"))
            if "n_blobs" in det:
                out["n_blobs"] = det["n_blobs"]
        elif mode == "loop":
            out["filled"] = sum(d["filled"] for d in det["calls"])
        return out

    if col == "ch02_pinned":
        tpb = {"numba": bench.ff2.PINNED_TPB, "triton": tff2.PINNED_TPB}
    elif col == MATCHED:
        tpb = tff2.PINNED_TPB
    else:
        tpb = TPB
    cfg = {"column": col, "cell": mode, "tpb": tpb}
    if grid is not None:
        cfg.update(blocks=grid["pinned"], caps=dict(grid["caps"]),
                   numba_is_bench_launch=grid["numba_is_bench_launch"])
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
    elif col == MATCHED:
        notes = f"{note}. {MATCHED_NOTE}"
    elif grid is not None:
        notes = f"{note}. {GRID_NOTE}"
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
    if state.get("clocks_before") is not None:
        row["clocks_before"] = state["clocks_before"]
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
    if row.get("est") and "numba" in row and row.get("calls"):
        # the projection's per-call basis, for readers of single cells
        calls = row["calls"]
        row["per_call_ms"] = {
            s: {"kernel": row[s]["kernel_ms"]["median"] / calls,
                "total": row[s]["total_ms"]["median"] / calls}
            for s in ("numba", "triton")}
    if "error" in row:
        row["error"] = _ascii(row["error"])
    return row, None


# -------------------------------------------------------------- estimate
# Per-call host cost the Numba JSONs do not hold: allocations, the H2D
# image, the D2H outputs, the sha1 of the digested arrays, and (ch04) the
# driver's own host-side checks. Measured per side on this laptop (RTX
# 4060 Laptop, i9-13900H, WSL2) on random_1000, two_sq_2800 and
# png_blobs: ch01-ch03 9-12 ns/px, ch04 46-47, ch05 25-59 (the upper end
# under memory pressure at 81 Mpx), ch06 3-9.
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
CASE_FIXED_S = 0.2           # two nvidia-smi queries per case


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


def _ref_pixels(ref, row_key):
    r = ref.get(row_key) or {}
    if r.get("width") and r.get("height"):
        return int(r["width"]) * int(r["height"])
    return None


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
    cell = (ref_row.get("cells") or {}).get(MATCHED_OF.get(col, col)) or {}
    overhead = (OVERHEAD_BASE_MS
                + px * OVERHEAD_NS_PER_PX[family] * 1e-6) / 1000.0
    if cell.get("skip"):
        # a typed refusal: the Numba warm call fails, the Triton probe runs
        return 2 * calls * overhead + CASE_FIXED_S, 0.0
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
    return (1 + repeats) * per_round + CASE_FIXED_S, per_round


def default_repeats(col, repeats, explicit):
    """bench_ch06's round count for the ch06 cells, bench.py's for the
    rest; an explicit --repeats (or --quick) applies to every cell."""
    if explicit or _family(col) != "ch06":
        return repeats
    return CH06_REPEATS


def plan_budget(plan, reps_of, ref, budget_s, spins=(0.0, 0.0)):
    """Estimate every case; past the budget, drop the heaviest cells to
    MIN_REPEATS until it fits. reps_of(col) gives a cell's repeats and
    spins is (per row, per ch05 / ch06 group). Returns (repeats per
    cell, estimate, reduced cells)."""
    per_cell = {}
    fixed = COMPILE_S
    row_spin, phase_spin = spins
    for row_key, kind, build, cols in plan:
        live = [c for c, m, _ in cols if m != "skip"]
        if not live:
            continue
        fixed += (PNG_BUILD_S if row_key.startswith("png")
                  else ROW_BUILD_S[kind]) + row_spin
        fixed += phase_spin * len({_family(c) for c in live}
                                  & {"ch05", "ch06"})
        ref_row = ref.get(row_key) or {}
        px = ref_row.get("width", 4000) * ref_row.get("height", 4000)
        if any(c in CH06_COLS for c in live):
            fixed += 2 * px * CH06_SETUP_NS_PER_PX * 1e-9
        for col, mode, _ in cols:
            r = reps_of(col)
            total, per_round = estimate_cell(row_key, col, mode, kind, r,
                                             ref)
            per_cell[(row_key, col)] = [total, per_round, r]
    est = fixed + sum(v[0] for v in per_cell.values())
    reduced = []
    if budget_s and est > budget_s:
        for key, v in sorted(per_cell.items(), key=lambda kv: -kv[1][1]):
            if est <= budget_s:
                break
            saved = (v[2] - MIN_REPEATS) * v[1]
            if saved <= 0:
                continue
            v[0] -= saved
            reduced.append({"row": key[0], "column": key[1],
                            "repeats": MIN_REPEATS, "from": v[2],
                            "estimated_round_s": round(v[1], 2)})
            v[2] = MIN_REPEATS
            est -= saved
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
            "driver host checks); plus CASE_FIXED_S per case, COMPILE_S, "
            "scene builds, row and phase spins and the ch06 engine setup"),
        "constants": {"overhead_base_ms": OVERHEAD_BASE_MS,
                      "overhead_ns_per_px": OVERHEAD_NS_PER_PX,
                      "ch06_setup_ns_per_px": CH06_SETUP_NS_PER_PX,
                      "triton_kernel_factor": {
                          **{f: TRITON_KERNEL_FACTOR_DEFAULT for f in
                             ("ch01", "ch02", "ch03", "ch04", "ch06")},
                          **TRITON_KERNEL_FACTOR},
                      "compile_s": COMPILE_S,
                      "case_fixed_s": CASE_FIXED_S},
    }
    reps = {k: v[2] for k, v in per_cell.items()}
    return reps, estimate, reduced


# ---------------------------------------------------------- host memory
def mem_available_mb():
    """MemAvailable from /proc/meminfo, in MB (None if unreadable)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def host_need_mb(px):
    return int(px * HOST_BYTES_PER_PX / 2 ** 20) + HOST_MARGIN_MB


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
    known_cols = TABLE_COLUMNS
    if cols_arg:
        cols = [c.strip() for c in cols_arg.split(",") if c.strip()]
        bad = [c for c in cols if c not in known_cols]
        if bad:
            raise SystemExit(f"unknown columns {bad}; known: {known_cols}")
        if MATCHED_OF[MATCHED] in cols:
            cols.append(MATCHED)         # the comparable row comes along
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
                why = (SKIPPED_COLUMNS.get(MATCHED_OF.get(col, col)) or
                       "ch04's kernel takes exactly two blobs in two "
                       "components; a one-blob row is outside its input "
                       "space (bench.py 'na')")
                skipped.append({"row": key, "column": col, "reason": reason,
                                "source": "static", "why": why})
            entries.append((col, mode, reason))
        plan.append((key, kind, build, entries))
    return plan, skipped


def _phase(col):
    fam = _family(col)
    return fam if fam in ("ch05", "ch06") else "ch01-ch04"


def plan_groups(cells, row_spin, phase_spin):
    """A row's cells [(case, state, wall, col, repeats)], in table order,
    as harness groups: consecutive cells with the same repeats and phase
    (ch01-ch04, ch05, ch06). The first group of a row spins row_spin
    (phase_spin if it is a ch05 / ch06 group, when longer); a group that
    starts the ch05 or the ch06 cells spins phase_spin."""
    groups = []
    for item in cells:
        col, r = item[3], item[4]
        ph = _phase(col)
        if groups and groups[-1]["repeats"] == r and groups[-1]["phase"] == ph:
            groups[-1]["members"].append(item)
            continue
        if not groups:
            spin = max(row_spin, phase_spin if ph != "ch01-ch04" else 0.0)
        elif groups[-1]["phase"] != ph:
            spin = phase_spin
        else:
            spin = 0.0
        groups.append({"repeats": r, "phase": ph, "spin": spin,
                       "members": [item]})
    return groups


def crosscheck(rows):
    """Per row: bench.py's check (every completed, non-estimated cell,
    ch06 included, fills the same pixel count) and bench_ch06's absolute
    checks on the ch06 cells (painted == red pixels, none left, blob
    count 1 / 2 on one- / two-blob rows and the component count on N-blob
    rows). {row: {"status", "filled", "failed_checks"}}."""
    fills, checks = {}, {}
    for r in rows:
        info = r.get("info") or {}
        if "error" in r or r.get("est"):
            continue
        scene = r["scene"]
        if info.get("filled") is not None:
            fills.setdefault(scene, set()).add(info["filled"])
        c = checks.setdefault(scene, [])
        exp = r["experiment"]
        if r.get("expected_blobs") is not None and "n_blobs" in info:
            c.append((f"{exp}.n_blobs", info["n_blobs"]
                      == r["expected_blobs"]))
        if "still_red" in info:
            c.append((f"{exp}.still_red", info["still_red"] == 0))
            if info.get("red_px") is not None:
                c.append((f"{exp}.painted_eq_red_px",
                          info["filled"] == info["red_px"]))
    out = {}
    for scene in sorted(set(fills) | set(checks)):
        f = fills.get(scene, set())
        failed = [name for name, ok in checks.get(scene, []) if not ok]
        ok = len(f) <= 1 and not failed
        out[scene] = {"status": "OK" if ok else "MISMATCH",
                      "filled": sorted(f), "failed_checks": failed}
    return out


def problems(doc):
    """Every reason a run is NOT OK: error rows, unequal outputs, failed
    crosschecks, and skips bench.py would report as "error:<type>"."""
    out = []
    for r in doc.get("rows", []):
        if "error" in r or not r.get("outputs_equal", False):
            out.append(f"{r['scene']} {r['experiment']}: "
                       f"{r.get('error') or r.get('mismatch_detail')}")
    meta = doc.get("meta") or {}
    for scene, c in (meta.get("crosscheck") or {}).items():
        if c["status"] != "OK":
            out.append(f"{scene} crosscheck {c['status']}: filled "
                       f"{c['filled']}, failed {c['failed_checks']}")
    for s in meta.get("skipped_cells") or []:
        if str(s.get("reason", "")).startswith("error:"):
            out.append(f"{s['row']} {s['column']}: skip {s['reason']} "
                       f"(numba: {s.get('numba')}; triton: "
                       f"{s.get('triton')})")
    return out


def _rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def _write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="small rows, 1 repeat, no spin, no JSON")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per cell (default {DEFAULT_REPEATS},"
                         f" ch06 {CH06_REPEATS}; given, it applies to all)")
    ap.add_argument("--rows", default=None, help="comma-separated row keys")
    ap.add_argument("--cols", default=None,
                    help="comma-separated column keys (ch02_pinned brings "
                         f"{MATCHED} along)")
    ap.add_argument("--budget-min", type=float, default=BUDGET_MIN,
                    help="GPU-time budget in minutes (0: no limit; "
                         "ignored with --quick)")
    ap.add_argument("--no-write", action="store_true",
                    help="measure but write no JSON")
    ap.add_argument("--no-mem-check", action="store_true",
                    help="build rows above 20 Mpx whatever MemAvailable is")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the plan and its time estimate, measure "
                         "nothing")
    args = ap.parse_args(argv)
    # the 1-, 2- and 48-block launches are the experiment, not a mistake
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    quick = args.quick
    explicit = quick or args.repeats is not None
    repeats = args.repeats or (1 if quick else DEFAULT_REPEATS)
    repeats += repeats % 2                  # what the harness runs
    write = not (quick or args.no_write)
    spin = 0.0 if quick else SPIN_SECONDS
    row_spin = 0.0 if quick else ROW_SPIN_SECONDS
    phase_spin = 0.0 if quick else PHASE_SPIN_SECONDS
    mem_check = not args.no_mem_check

    def reps_of(col):
        return default_repeats(col, repeats, explicit)

    rows, cols = select(args.rows, args.cols, quick)
    plan, skipped = build_plan(rows, cols)
    ref, ref_sources = load_reference()
    budget_s = 0.0 if quick else args.budget_min * 60.0
    reps, estimate, reduced = plan_budget(plan, reps_of, ref, budget_s,
                                          spins=(row_spin, phase_spin))
    estimate["reference"] = ref_sources
    n_cases = sum(1 for *_, entries in plan
                  for _, mode, _ in entries if mode != "skip")
    print(f"overview twin: {len(plan)} rows x {len(cols)} columns, "
          f"{n_cases} cases, {len(skipped)} static skips, "
          f"repeats={repeats}"
          + ("" if explicit else f" (ch06 {CH06_REPEATS})"))
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
                    f"repeats instead of their default to fit the "
                    f"{budget_s / 60:.0f}-minute budget "
                    "(meta.reduced_repeats)")
    if quick:
        caps.insert(0, "QUICK smoke run: small rows, 2 rounds, no spin; "
                       "not a measurement")
    missing_png = [k for k, f in (("png_blocks", "input_blocks.png"),
                                  ("png_blobs", "input_blobs.png"))
                   if k not in [r[0] for r in bench.ROWS]]
    grid_cols = [c for c in cols if c in GRID_COLS]
    meta = {
        "mirrors": ["overview/bench.py", "overview/bench_ch06.py"],
        "quick": quick, "complete": False,
        "rows": [p[0] for p in plan], "columns": {
            c: {**_column_info(c), "family": _family(c),
                "triton": ("skipped" if c in SKIPPED_COLUMNS else
                           DEVIATIONS.get(c) or
                           (MATCHED_NOTE if c == MATCHED else
                            "identical arguments, plus blocks= the pinned "
                            "grid" if c in GRID_COLS else
                            "identical arguments"))}
            for c in cols},
        "cell_rules": (
            "measured: one call; loop: one-blob kernel on a two-blob row, "
            "one call per blob, summed; est: one-blob kernel on an N-blob "
            "row, median per-call ms over bench.py's k-blob sample "
            f"(k={bench.EST_SAMPLE}, {bench.EST_SAMPLE_HUGE} past 20 Mpx) "
            "x the blob count; est_pair: ch04 on an N-blob row, the same "
            "over blob pairs x the pair count. Samples, loops and skip "
            "reasons come from bench.py's own cell functions. Both "
            "backends run the same sample. est rows carry est=true, calls, "
            "sample and per_call_ms (the projected kernel_ms / total_ms "
            "medians divided by calls)"),
        "outputs": (
            "deterministic outputs reduced to sha1 digests (dtype, shape) "
            "right after each call, per chapter compare's same() minus "
            "per-block fields; loop and est cells compare every call; "
            "ch06 also counts painted and still-red pixels on the device"),
        "grids": {
            "rule": (
                "ch03-ch05: both backends run min(Numba cap, Triton cap) "
                "at tpb 256, the ch03 compare's suite rule. bench.py "
                "launches blocks=None (each backend's own cap); where "
                "Numba's cap is the minimum (numba_is_bench_launch) the "
                "Numba cell is bench.py's own launch and the twin gets "
                "blocks= the same grid. config.resolved_blocks / "
                "resolved_tpb come from the warm-up results, and rows "
                "where they differ are comparable=false"),
            "columns": {c: grid_caps(c) for c in grid_cols},
        },
        "timing": (
            "ch01-ch05: the drivers' kernel_ms (perf_counter + "
            "synchronize) and total_ms; ch06: the CUDA-event span of run() "
            "and the host wall time of run() + synchronize, after restore "
            "and (mask) pack off the clock. CuPy's pool stays warm within "
            "a case on every row (the harness frees both pools between "
            f"cases); above {BIG_ROW_PX // 1_000_000} Mpx garbage is "
            "collected and Numba's deferred frees flushed before each "
            f"call. ch06 cells run {CH06_REPEATS} rounds "
            f"(bench_ch06.ROUNDS={bench_ch06.ROUNDS}, made even) unless "
            "--repeats is given"),
        "spin": {"first_s": spin, "per_row_s": row_spin,
                 "before_ch05_and_ch06_groups_s": phase_spin,
                 "note": ("bench_ch06 spins 8 s with the ch06 engine per "
                          "row; here a CuPy copy spin of "
                          f"{phase_spin:g} s runs right before each row's "
                          "ch05 and ch06 groups instead. clocks_before / "
                          "clocks_after are on every row")},
        "host_memory": {"bytes_per_px": HOST_BYTES_PER_PX,
                        "margin_mb": HOST_MARGIN_MB,
                        "checked": mem_check, "checks": []},
        "skipped_cells": skipped,
        "not_compared": {**NOT_COMPARED, **{
            k: "row absent: its input PNG is missing (bench.py adds it only "
               "when present)" for k in missing_png}},
        "deviations": {**DEVIATIONS, MATCHED: MATCHED_NOTE},
        "est_sample": {"k": bench.EST_SAMPLE, "k_huge": bench.EST_SAMPLE_HUGE},
        "estimate": estimate, "reduced_repeats": reduced, "caps": caps,
    }

    out_dir = (results_paths.results_dir("triton_twins", UNIT) if write
               else None)
    partial_path = os.path.join(out_dir, PARTIAL_NAME) if write else None
    live_rows = [p[0] for p in plan
                 if any(m != "skip" for _, m, _ in p[3])]
    stamp = None
    doc = None
    t_start = time.perf_counter()
    first = True
    done = 0
    for row_key, kind, build, entries in plan:
        live = [(col, mode) for col, mode, _ in entries if mode != "skip"]
        if not live:
            continue
        note = next(r[2] for r in bench.ROWS if r[0] == row_key)
        px = _ref_pixels(ref, row_key)
        if mem_check and px and px > BIG_ROW_PX:
            avail, need = mem_available_mb(), host_need_mb(px)
            meta["host_memory"]["checks"].append(
                {"row": row_key, "available_mb": avail, "needed_mb": need})
            if avail is not None and avail < need:
                why = (f"MemAvailable {avail} MB < {need} MB needed "
                       f"({HOST_BYTES_PER_PX} B/px x {px / 1e6:.1f} Mpx + "
                       f"{HOST_MARGIN_MB} MB)")
                for col, _ in live:
                    skipped.append({"row": row_key, "column": col,
                                    "reason": "host-memory",
                                    "source": "preflight", "why": why})
                caps.append(f"{row_key} not measured: {why}")
                print(f"\n{row_key}: SKIPPED, {why}", flush=True)
                continue
        print(f"\n{row_key} ({note}): building the scene...", flush=True)
        ctx = build_row(build, kind)
        cells = []
        for col, mode in live:
            wall = {}
            case, state = make_case(row_key, note, col, mode, ctx, wall)
            cells.append((case, state, wall, col, reps[(row_key, col)]))
        groups = plan_groups(cells, row_spin, phase_spin)
        cells.clear()
        for g in groups:
            s = spin if first else g["spin"]
            first = False
            members = g["members"]
            d = run_cases(UNIT, [m[0] for m in members],
                          repeats=g["repeats"], meta=meta, write=False,
                          spin_seconds=s,
                          log=lambda m, k=row_key: print(f"  {k} {m}",
                                                         flush=True))
            measured = d["rows"]         # before doc (maybe d) is reset
            if doc is None:
                doc = d
                doc["rows"] = []
                doc["repeats"] = repeats
                stamp = doc["created_utc"]
            for row, (case, state, wall, col, r) in zip(measured, members):
                if r != repeats:
                    row["repeats"] = r + r % 2
                kept, skip = finish_row(row, state, wall, col)
                if skip:
                    skipped.append(skip)
                    print(f"  {row_key} {col}: skip ({skip['reason']}); "
                          f"triton: {skip['triton']}")
                else:
                    doc["rows"].append(kept)
        # release the row before the next build: the cases' closures and
        # est subs hold its image
        groups.clear()
        g = members = measured = d = case = state = wall = None
        ctx.clear()
        del ctx
        gc.collect()
        free_device_memory()
        done += 1
        if doc is not None:
            cc = crosscheck([r for r in doc["rows"]
                             if r["scene"] == row_key]).get(row_key)
            if cc:
                failed = (f" failed={cc['failed_checks']}"
                          if cc["failed_checks"] else "")
                print(f"  {row_key} [{cc['status']}] filled={cc['filled']}"
                      f"{failed}", flush=True)
        if doc is not None and write:      # crash-safe per row
            meta["peak_host_rss_mb"] = _rss_mb()
            meta["elapsed_s"] = round(time.perf_counter() - t_start, 1)
            meta["crosscheck"] = crosscheck(doc["rows"])
            meta["caps"] = caps + [
                f"INCOMPLETE: {done} of {len(live_rows)} rows measured "
                "(meta.complete=false)"]
            _write_json(partial_path, doc)

    if doc is None:
        print("nothing to measure: every selected cell is a skip")
        return {"rows": [], "meta": meta}
    meta["complete"] = True
    meta["caps"] = caps
    meta["peak_host_rss_mb"] = _rss_mb()
    meta["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    meta["crosscheck"] = crosscheck(doc["rows"])
    doc["meta"] = meta
    if write:
        path = os.path.join(out_dir, f"compare_{stamp}.json")
        _write_json(path, doc)
        _write_json(partial_path, {"complete": True,
                                   "final": os.path.basename(path)})
        doc["path"] = path
        print(f"wrote {path}")

    print(f"\n{len(doc['rows'])} rows measured, {len(skipped)} skipped "
          f"cells, {meta['elapsed_s'] / 60:.1f} min, peak RSS "
          f"{meta['peak_host_rss_mb']} MB")
    bad = problems(doc)
    for p in bad:
        print(f"  NOT OK {p}")
    print(f"{len(bad)} problems (error or mismatch rows, failed "
          "crosschecks, error: skips)")
    return doc


if __name__ == "__main__":
    _doc = main()
    raise SystemExit(1 if problems(_doc) else 0)
