"""
Numba vs Triton, measured the same way for every chapter.

Each chapter's ``compare.py`` builds a list of ``Case`` objects (one scene,
one configuration, one callable per backend) and hands them to
``run_cases``. The harness owns the method, so every chapter's numbers mean
the same thing:

1. Warm both backends on the case (JIT and Triton compiles stay out of the
   timings), and check that their outputs are identical.
2. Run ``repeats`` rounds. Each round runs both backends back to back, and
   the order flips every round (N T, T N, N T, ...). This laptop's clocks
   drift by up to 8% between sessions, so timing one backend's runs
   after the other's would mostly measure the drift.
3. Re-check output equality on every timed run, not only the first.
4. Report the median and min of the drivers' own ``kernel_ms`` and
   ``total_ms``: the repo's perf_counter + synchronize convention, which
   includes launch overhead on both sides.

Before the first case the GPU is spun up to its boost clock (the ch06
benchmark's rule: an idle laptop GPU sits at a third of its clock), and
every row records the SM clock, power and temperature nvidia-smi saw
right after it. Between cases both runtimes' memory pools are released:
VRAM is shared by Numba and CuPy, and the host has 6 GB.

``speedup`` is numba_ms / triton_ms: above 1 means Triton is faster.

The JSON lands in results/triton_twins/<chapter>/compare_<UTC stamp>.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from flood_fill_cuda.shared.results_paths import results_dir

SCHEMA_VERSION = 1


@dataclass
class Case:
    """One measured cell: a scene, a configuration, and both backends.

    run_numba / run_triton take no arguments and return the backend's
    result object (anything with ``kernel_ms`` and ``total_ms``).
    same(numba_result, triton_result) returns (equal, detail); detail is a
    short string naming the first mismatch, or "" when equal.
    """

    experiment: str
    scene: str
    config: dict
    run_numba: Callable[[], Any]
    run_triton: Callable[[], Any]
    same: Callable[[Any, Any], tuple[bool, str]]
    pixels: int = 0
    # Extra per-case facts for the JSON (blob counts, levels, ...), taken
    # from the warm-up results: info(numba_result, triton_result) -> dict.
    info: Callable[[Any, Any], dict] | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _stats(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": [round(v, 6) for v in values],
    }


def _versions():
    import cupy
    import numba
    import triton
    import cupy.cuda.runtime as rt

    drv = rt.driverGetVersion()
    return {
        "python": platform.python_version(),
        "numba": numba.__version__,
        "triton": triton.__version__,
        "cupy": cupy.__version__,
        "cuda_driver": f"{drv // 1000}.{drv % 1000 // 10}",
    }


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return None


def gpu_clocks():
    """SM clock, its max, power and temperature, as nvidia-smi reports."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,"
             "temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, sm_max, power, temp = (v.strip() for v in out.split(","))
        return {"sm_mhz": int(sm), "max_sm_mhz": int(sm_max),
                "power_w": float(power), "temp_c": int(temp)}
    except Exception:
        return None


def free_device_memory():
    """Release CuPy's pool and Numba's deferred frees between cases."""
    import cupy

    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    try:
        from numba import cuda

        cuda.current_context().deallocations.clear()
    except Exception:
        pass


def spin_up(seconds: float = 8.0):
    """Keep the GPU busy with back-to-back copies until it reaches boost."""
    import cupy

    a = cupy.empty(64 * 2 ** 20 // 8, dtype=cupy.int64)
    b = cupy.empty_like(a)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(20):
            b[...] = a
        cupy.cuda.Device().synchronize()
    del a, b
    free_device_memory()


def run_case(case: Case, repeats: int) -> dict:
    """Measure one case. Never raises for a backend failure: the row records
    the error instead, so one bad cell cannot sink a whole sweep."""
    row = {
        "experiment": case.experiment,
        "scene": case.scene,
        "pixels": case.pixels,
        "config": case.config,
        "notes": case.notes,
        **case.extra,
    }
    try:
        rn = case.run_numba()
        rt = case.run_triton()
    except Exception as exc:  # recorded, not hidden
        row["error"] = f"warm-up failed: {type(exc).__name__}: {exc}"
        return row
    equal, detail = case.same(rn, rt)
    if case.info is not None:
        row["info"] = case.info(rn, rt)
    del rn, rt

    times = {"numba": {"kernel": [], "total": []},
             "triton": {"kernel": [], "total": []}}
    runners = {"numba": case.run_numba, "triton": case.run_triton}
    mismatches = 0 if equal else 1
    try:
        for r in range(repeats):
            order = ("numba", "triton") if r % 2 == 0 else ("triton", "numba")
            got = {}
            for name in order:
                res = runners[name]()
                times[name]["kernel"].append(float(res.kernel_ms))
                times[name]["total"].append(float(res.total_ms))
                got[name] = res
            ok, d = case.same(got["numba"], got["triton"])
            if not ok:
                mismatches += 1
                detail = detail or d
            del got
    except Exception as exc:  # recorded, not hidden
        row["error"] = f"timed round failed: {type(exc).__name__}: {exc}"
        return row
    row["clocks_after"] = gpu_clocks()

    for name in ("numba", "triton"):
        row[name] = {"kernel_ms": _stats(times[name]["kernel"]),
                     "total_ms": _stats(times[name]["total"])}
    row["speedup_kernel"] = (row["numba"]["kernel_ms"]["median"]
                             / row["triton"]["kernel_ms"]["median"])
    row["speedup_total"] = (row["numba"]["total_ms"]["median"]
                            / row["triton"]["total_ms"]["median"])
    row["outputs_equal"] = mismatches == 0
    row["mismatched_runs"] = mismatches
    if detail:
        row["mismatch_detail"] = detail
    return row


def run_cases(chapter: str, cases: list[Case], repeats: int,
              meta: dict | None = None, write: bool = True,
              log: Callable[[str], None] = print,
              spin_seconds: float = 8.0) -> dict:
    """Measure every case in order and write the comparison JSON."""
    from flood_fill_cuda.triton_twins.runtime import device_info

    dev = device_info()
    clocks_before = gpu_clocks()
    if spin_seconds > 0:
        spin_up(spin_seconds)
    rows = []
    for i, case in enumerate(cases, 1):
        row = run_case(case, repeats)
        free_device_memory()
        rows.append(row)
        if "error" in row:
            log(f"[{i}/{len(cases)}] {case.experiment} {case.scene} "
                f"{case.config}: ERROR {row['error']}")
        else:
            log(f"[{i}/{len(cases)}] {case.experiment} {case.scene} "
                f"{case.config}: numba {row['numba']['kernel_ms']['median']:.3f} ms"
                f" | triton {row['triton']['kernel_ms']['median']:.3f} ms"
                f" | x{row['speedup_kernel']:.2f}"
                f" | equal={row['outputs_equal']}")

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    doc = {
        "schema": SCHEMA_VERSION,
        "chapter": chapter,
        "created_utc": stamp,
        "gpu": {"name": dev.name, "sm_count": dev.sm_count,
                "max_threads_per_sm": dev.max_threads_per_sm},
        "versions": _versions(),
        "git_commit": _git_commit(),
        "clocks_idle": clocks_before,
        "spin_up_seconds": spin_seconds,
        "repeats": repeats,
        "method": (
            "Both backends warmed per case; repeats rounds with the run order "
            "flipping each round (N,T then T,N); outputs compared on every "
            "run. kernel_ms/total_ms are each driver's own perf_counter + "
            "synchronize measurements, launch overhead included. "
            "speedup = numba_ms / triton_ms (>1: Triton faster)."),
        "meta": meta or {},
        "rows": rows,
    }
    if write:
        path = os.path.join(results_dir("triton_twins", chapter),
                            f"compare_{stamp}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
        doc["path"] = path
        log(f"wrote {path}")
    return doc


def arrays_equal(**pairs) -> tuple[bool, str]:
    """same() helper: arrays_equal(img=(a, b), depth=(c, d), ...)."""
    import numpy as np

    for name, (a, b) in pairs.items():
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return False, f"{name}: shape {a.shape} vs {b.shape}"
        if not np.array_equal(a, b):
            n = int(np.count_nonzero(a != b))
            return False, f"{name}: {n} elements differ"
    return True, ""
