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


def test_settle_waits_until_the_pool_is_parked():
    """After a 16-thread call libgomp's idle workers spin before they
    sleep: settle_workers() returns once they stopped using CPU."""
    if B._pool_cpu_ns() is None:
        pytest.skip("no per-thread schedstat on this kernel")
    img, sx, sy = scenes.square_scene(256, 256, 128, 128)
    from .. import flood_fill as ff
    eng = ff.TileBFS(256, 256, threads=16)
    for _ in range(3):
        eng.run(img.copy(), sx, sy)
        got = B.settle_workers()
        assert got["settled"] is True, got
        a = B._pool_cpu_ns()
        B.time.sleep(0.02)
        # parked: well under one CPU's worth over 20 ms
        assert B._pool_cpu_ns() - a < 5_000_000


def test_every_cell_batch_starts_on_a_parked_pool(monkeypatch):
    """A one-core cell must not start next to the workers of the
    16-thread cell before it: cpu_pass settles before every batch."""
    args = _args("--cpu-rounds", "2", "--max-calls", "2", "--round-s", "5")
    cells, _ = _cells("one", args)
    seen = []
    monkeypatch.setattr(B, "settle_workers",
                        lambda **kw: seen.append(1) or {"settled": True})
    B.cpu_pass(cells, args)
    # the first timed call, and every batch of every round
    assert len(seen) >= len(cells) * (1 + args.cpu_rounds)


class _FakeProc:
    """/proc/stat and this process's CPU time, advanced one second per
    read pair: `others` and `own` CPUs busy out of 16."""

    def __init__(self, others, own):
        self.others, self.own = others, own
        self.busy = self.total = 0
        self.own_s = 0.0
        self.n = 0

    def cpu_busy(self):
        self.n += 1
        if self.n % 2 == 0:              # the second read of a window
            self.busy += int(100 * (self.others + self.own))
            self.total += 1600
            self.own_s += self.own
        return self.busy, self.total

    def own_cpu_s(self):
        return self.own_s


@pytest.mark.parametrize("others,own,threshold,quiet", [
    (0.2, 0.0, 3.0, True),
    (1.5, 0.0, 3.0, False),       # 9.4%: passed the old 10% gate
    (1.5, 0.0, 10.0, True),
    (0.2, 12.0, 3.0, True),       # this process's own threads don't count
])
def test_gate_counts_only_the_rest_of_the_machine(monkeypatch, others, own,
                                                  threshold, quiet):
    fake = _FakeProc(others, own)
    monkeypatch.setattr(B, "_cpu_busy", fake.cpu_busy)
    monkeypatch.setattr(B, "_own_cpu_s", fake.own_cpu_s)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    got = B.wait_quiet(0, True, threshold)
    assert got["quiet"] is quiet
    assert got["others_pct"] == pytest.approx(100 * others / 16, abs=0.1)


def test_cpu_rounds_hold_8_threads_to_the_strict_gate():
    assert B.gate_threshold(16) == B.QUIET_BUSY_PCT_PARALLEL <= 3.0
    assert B.gate_threshold(1) == B.QUIET_BUSY_PCT


def test_a_loaded_round_is_run_again(monkeypatch):
    """A round whose others' share went over the threshold is discarded
    and run again (up to MAX_ROUND_REDOS times), so its samples never
    reach the medians; the discarded attempts are recorded."""
    args = _args("--cpu-rounds", "2", "--max-calls", "2", "--round-s", "5")
    args.gate = True
    cells, _ = _cells("one", args)
    script = [20.0, 0.0, 0.0]           # round 0 loaded once, then quiet

    class Meter:
        def __init__(self, kind="batch"):
            self.kind = kind

        def read(self):
            pct = script.pop(0) if self.kind == "round" else 0.0
            return {"others_pct": pct, "machine_pct": pct}
    monkeypatch.setattr(B, "LoadMeter", Meter)
    monkeypatch.setattr(B, "wait_quiet", lambda *a, **k: {"quiet": True})
    rounds = B.cpu_pass(cells, args)
    assert [len(r["discarded_attempts"]) for r in rounds] == [1, 0]
    assert not any(r["loaded"] for r in rounds)
    for c in cells:
        assert len(c["times"]) == 2 * c["calls"]
        assert len(c["round_medians"]) == 2
    # every round starts with the probe, relative to the session's first
    assert all(r["probe"]["serial_ms"] > 0 for r in rounds)
    assert "parallel_rel" in rounds[-1]["probe"]


def test_gpu_part_stops_its_sampler(monkeypatch):
    """nvidia-smi samples about 7 times a second and wakes a reader
    thread each time: it must be stopped when the GPU blocks end, not
    run through the CPU passes, where each sample preempts a worker of
    a 16-thread cell."""
    events = []

    class FakeGpu:
        device, sm_count, cc = "fake", 1, (8, 9)

    class FakeSampler:
        def stop(self):
            events.append("stop")
    monkeypatch.setattr(B, "Gpu", FakeGpu)
    monkeypatch.setattr(B, "Sampler", FakeSampler)
    monkeypatch.setattr(B, "gpu_scene", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(B, "reference", lambda *a, **k: {})
    monkeypatch.setattr(B, "_smi", lambda q: None)
    monkeypatch.setattr(B.time, "sleep", lambda s: events.append("cooldown"))
    args = _args("--cooldown", "1")
    args.no_gpu = False
    img, seed, conns = _scene("one")
    B.run_gpu_blocks([{"scene": "t", "gpu": {}}],
                     [("t", "one", img, seed, conns)], [], args)
    assert events == ["stop", "cooldown"]


def test_session_names_the_gpu(monkeypatch):
    """The GPU numbers must say which GPU and driver made them: the
    device, its SMs, the driver, the max clocks, and the clocks at the
    start and the end of the GPU blocks."""
    answers = {
        "driver_version,clocks.max.sm,clocks.max.mem": ["610.62", "3105",
                                                        "8001"],
        "clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw": [
            "210", "405", "P8", "41", "3.2"]}
    monkeypatch.setattr(B, "_smi", lambda q: answers[q])

    class FakeGpu:
        device, sm_count, cc = "Some GPU", 24, (8, 9)
    info = B.gpu_identity(FakeGpu())
    assert info == {"device": "Some GPU", "sm_count": 24,
                    "compute_capability": "8.9", "driver": "610.62",
                    "max_sm_mhz": 3105, "max_mem_mhz": 8001}
    clk = B.gpu_clocks()
    assert clk == {"sm_mhz": 210, "mem_mhz": 405, "pstate": "P8",
                   "temperature_c": 41, "power_w": 3.2}
    monkeypatch.setattr(B, "_smi", lambda q: None)
    assert B.gpu_clocks() == {}


def _cell(ms, rounds=2, **meta):
    return {"ms": ms, "round_medians": [ms] * rounds, **meta}


def test_each_variant_is_paired_with_its_own_gpu_chapter():
    """Same work, same idea: the pixel BFS against ch03's pixel BFS at
    the same connectivity, the tile BFS against ch07, the labels against
    ch08's first version. Every factor names its GPU cell, and the
    factors against the other GPU cells are kept apart."""
    row = {"connectivities": [8, 4], "cells": {
        "ch09_pixel_t1_c8": _cell(400.0), "ch09_pixel_t16_c8": _cell(200.0),
        "ch09_tile_t1_c8": _cell(100.0), "ch09_tile_t16_c8": _cell(31.0),
        "one_fill_paint_c8": _cell(1400.0), "one_fill_c8": _cell(1390.0)},
        "gpu": {"ch07_c8": {"ms": 3.1, "connectivity": 8},
                "ch03_conn8": {"ms": 96.9, "connectivity": 8},
                "ch03_wc_bare": {"ms": 70.9, "connectivity": 8},
                "ch07_c4": {"ms": 3.3, "connectivity": 4},
                "ch03_conn4": {"ms": 133.3, "connectivity": 4}}}
    sm = B.summarize(row, "one", [1, 16])
    px, tl = sm["pixel_c8"][16], sm["tile_c8"][16]
    assert px["gpu_cell"] == "ch03_conn8"
    assert px["gpu_x"] == pytest.approx(200.0 / 96.9)
    assert px["gpu_x_other"] == {"ch07_c8": pytest.approx(200.0 / 3.1),
                                 "ch03_wc_bare": pytest.approx(200 / 70.9)}
    assert tl["gpu_cell"] == "ch07_c8"
    assert tl["gpu_x"] == pytest.approx(31.0 / 3.1)
    many = {"connectivities": [8], "cells": {
        "ch09_labels_t16": _cell(20.0), "one_label_paint": _cell(300.0)},
        "gpu": {"ch08_old": {"ms": 2.2, "connectivity": 8},
                "ch08": {"ms": 1.9, "connectivity": 8}}}
    lb = B.summarize(many, "many", [16])["labels"][16]
    assert lb["gpu_cell"] == "ch08_old"
    assert lb["gpu_x_other"] == {"ch08": pytest.approx(20.0 / 1.9)}


def test_probe_chain_is_real_work():
    """The calibration probe must cost the time of its steps, or its
    ratios mean nothing: a multiply-add chain was reassociated (0.1 ns a
    step) and, in the prange, ran 1M steps in 0.014 ms. A xorshift step
    is six dependent operations: at least about 1 ns on any current
    core, so 0.5 ns is a safe floor."""
    import time
    from .. import flood_fill as ff

    def best(fn, *a):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn(*a)
            ts.append(time.perf_counter() - t0)
        return min(ts)
    n = 2_000_000
    B._probe_chain(10, 1)
    assert best(B._probe_chain, n, 1) > 0.5e-9 * n
    with ff.num_threads(2):
        B._probe_all(10, 2)
        assert best(B._probe_all, n, 2) > 0.5e-9 * n
