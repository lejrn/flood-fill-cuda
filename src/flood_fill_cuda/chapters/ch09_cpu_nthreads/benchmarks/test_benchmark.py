"""Tests of the ch09 benchmark harness (no GPU, no timings).

The harness is what turns ch09's racy kernels into published numbers,
so its checks, its gate and its summaries are tested here on tiny
scenes. Nothing here measures anything: every timing these tests take
is thrown away.

Run:

    nice -n 10 uv run pytest src/flood_fill_cuda/chapters/ch09_cpu_nthreads/benchmarks/
"""

import numpy as np
import pytest

from ...ch03_gpu_1blob_nblock import scenes
from ...ch05_gpu_nblob_nblock import scenes as scenes5
from . import benchmark as B


def _args(*argv):
    """Harness settings for a test: no gate, small rounds."""
    args = B.parse_args(["--no-gate", "--no-gpu", *argv])
    args.threads = [1, 2]
    return args


def _scene(kind):
    if kind == "one":
        img, sx, sy = scenes.square_scene(96, 80, 60, 50)
        return img, (sx, sy), (8,)
    img, _ = scenes5.random_blobs_scene(150, 110, 0.3, 1)
    return img, None, (8,)


def _cells(kind, args):
    img, seed, conns = _scene(kind)
    ref = B.reference(kind, img, seed, conns)
    work = np.empty_like(img)
    one = B.OneCore(img.shape[0], img.shape[1], kind)
    return B.cpu_cells("t", kind, img, seed, conns, ref, args, work, one), work


@pytest.mark.parametrize("kind,key", [
    ("one", "ch09_pixel_t2_c8"), ("one", "ch09_tile_t2_c8"),
    ("one", "ch09_tile_fill_t1_c8"), ("one", "one_fill_c8"),
    ("many", "ch09_labels_t2"), ("many", "one_label")])
def test_every_timed_output_is_checked(kind, key):
    """One timed call in the middle of round 0 writes a wrong pixel:
    the cell must fail its checks, and no other cell may."""
    args = _args("--cpu-rounds", "2", "--max-calls", "3", "--round-s", "5")
    cells, work = _cells(kind, args)
    cell = next(c for c in cells if c["key"] == key)
    calls = {"n": 0}
    real = cell["fn"]

    def corrupt_fourth_call():
        # call 1 untimed, call 2 the first timed one, calls 3-5 round 0
        real()
        calls["n"] += 1
        if calls["n"] == 4:
            work[0, 0] = (1, 2, 3)
    cell["fn"] = corrupt_fourth_call
    B.cpu_pass(cells, args)
    rec = {c["key"]: B.cpu_record(c) for c in cells}
    assert cell["calls"] == 3 and calls["n"] == 8
    assert rec[key]["checks_ok"] is False
    assert rec[key]["call_mismatches"] == 1
    assert all(r["checks_ok"] for k, r in rec.items() if k != key)
    assert all(r["calls_checked"] == 7 for r in rec.values())


def test_unavailable_thread_counts_stop_the_run():
    """A process that starts with a narrower CPU affinity sees fewer
    threads (NUMBA_NUM_THREADS is fixed at start-up): the run must stop
    before warming up, not drop the 16-thread cells and go on."""
    with pytest.raises(SystemExit, match="16"):
        B.select_threads([1, 2, 4, 8, 16], limit=8)
    with pytest.raises(SystemExit):
        B.select_threads([0, 1], limit=16)
    assert B.select_threads([16, 1, 2, 2], limit=16) == [1, 2, 16]


def _leaves(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from _leaves(v)
    elif isinstance(d, (list, tuple)):
        for v in d:
            yield from _leaves(v)
    else:
        yield d


def test_runtime_is_recorded_without_paths():
    """The session records how the CPU side was scheduled: the layer,
    the chunk size, niceness, affinity, the OpenMP variables, the
    kernel and the OpenMP runtime, and the CPU topology the guest sees.
    No value may hold a path (results are committed)."""
    info = B.runtime_info()
    for k in ("threading_layer", "parallel_chunksize", "niceness",
              "affinity_cpus", "numba_num_threads", "kernel", "wsl",
              "openmp_env", "runtime_libs"):
        assert k in info, k
    for k in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_PROC_BIND",
              "OMP_PLACES"):
        assert k in info["openmp_env"], k
    host = B._host_info()
    for k in ("hypervisor", "cores_seen", "threads_per_core_seen",
              "cpufreq_visible"):
        assert k in host, k
    assert not any("/" in str(v) for v in _leaves(info))
    assert not any("/" in str(v) for v in _leaves(host))


def test_timed_run_refuses_a_skewed_runtime():
    """nice, a narrowed affinity or a preset OpenMP wait policy would
    bias the 16-thread cells against the one-core ones: a timed run
    stops unless a flag allows it; a smoke run only notes it."""
    base = {"niceness": 0, "affinity_cpus": 16, "numba_num_threads": 16,
            "openmp_env": {"OMP_WAIT_POLICY": "unset"}}
    args = B.parse_args([])
    assert B.check_runtime(args, base) == []
    with pytest.raises(SystemExit, match="nice"):
        B.check_runtime(args, {**base, "niceness": 10})
    with pytest.raises(SystemExit, match="affinity"):
        B.check_runtime(args, {**base, "affinity_cpus": 8})
    with pytest.raises(SystemExit, match="OMP_WAIT_POLICY"):
        B.check_runtime(args, {**base,
                               "openmp_env": {"OMP_WAIT_POLICY": "PASSIVE"}})
    allowed = B.parse_args(["--allow-nice", "--allow-openmp-env"])
    notes = B.check_runtime(allowed, {
        **base, "niceness": 10,
        "openmp_env": {"OMP_WAIT_POLICY": "PASSIVE"}})
    assert len(notes) == 2
    smoke = B.parse_args(["--smoke"])
    assert len(B.check_runtime(smoke, {**base, "niceness": 10})) == 1
