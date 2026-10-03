"""The overview twin: coverage of bench.py's columns, parity of its cell
rules and arguments with bench.py's own, the skip, crosscheck and
crash-safety paths, and one tiny row end to end on both backends.

Fast by design (under a minute, most of it JIT): the parity tests drive
bench.py's own functions with fake runners or recorded drivers, the
runtime-skip tests run one column of the 96 x 64 comb row with patched
runners, and the end-to-end tests run every column on the comb row and
two columns of the 700 x 400 two-squares row (the per-blob loop and
ch04).
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import json
from types import SimpleNamespace

import numpy as np
import pytest

from flood_fill_cuda.overview import bench, bench_ch06
from flood_fill_cuda.triton_twins.compare import overview as ov

# the 1-, 2- and 48-block launches are the experiment
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning")


def _rows(*keys):
    return {r[0]: r for r in bench.ROWS if r[0] in keys}


# ------------------------------------------------------------- coverage

def test_every_numba_column_has_a_twin_or_a_skip_reason():
    for key, *_ in bench.COLS:
        if key in ov.TRITON_RUNNERS:
            continue
        assert ov.SKIPPED_COLUMNS.get(key), f"{key}: no twin, no reason"
    for key, *_ in bench_ch06.COLS:
        assert key in ov.CH06_COLS, f"{key}: ch06 column not mirrored"
    # nothing invented: every twin answers a bench.py column
    assert set(ov.TRITON_RUNNERS) <= {c[0] for c in bench.COLS}
    assert ov.GPU_COLUMNS == ([c[0] for c in bench.COLS]
                              + [c[0] for c in bench_ch06.COLS])
    # the one extra row is the matched ch02_pinned, right after it
    extra = [c for c in ov.TABLE_COLUMNS if c not in ov.GPU_COLUMNS]
    assert extra == [ov.MATCHED]
    i = ov.TABLE_COLUMNS.index("ch02_pinned")
    assert ov.TABLE_COLUMNS[i + 1] == ov.MATCHED
    # the CPU bars are recorded as not compared, with the reason
    for key, *_ in bench.CPU_COLS:
        assert ov.NOT_COMPARED.get(key)


def test_streams_stays_skipped_like_bench():
    assert "ch04_streams" in bench.STATIC_SKIPS
    assert "ch04_streams" not in ov.TRITON_RUNNERS
    for kind in ("one", "two", "n"):
        assert ov.cell_mode("ch04_streams", kind) == (
            "skip", bench.STATIC_SKIPS["ch04_streams"])


def test_every_row_kind_is_known_without_building():
    for key, _f, _n, _e, build in bench.ROWS:
        assert ov.row_kind(build) in ("one", "two", "n"), key
    for key, (_k, _f, _n, _e, build) in _rows("comb_24",
                                              "two_sq_300").items():
        assert build()["kind"] == ov.row_kind(build)


# ------------------------------------------- parity with bench.py's rules

def test_runners_pass_bench_arguments(monkeypatch):
    """Both drivers patched with recorders: every twin runner passes the
    arguments bench.py's runner passes (ch02_pinned's block size aside),
    and a pinned-grid runner adds blocks= and nothing else."""
    calls = []

    def recorder(tag):
        def flood_fill(*args, **kw):
            calls.append((tag, args, dict(kw)))
        return flood_fill

    for fam in ("ch01", "ch02", "ch03", "ch04", "ch05"):
        monkeypatch.setattr(ov._NUMBA_FF[fam], "flood_fill",
                            recorder("numba"))
        monkeypatch.setattr(ov._TRITON_FF[fam], "flood_fill",
                            recorder("triton"))
    ctx = {"img": "IMG", "sx": 3, "sy": 4, "seeds": "SEEDS"}
    assert set(ov.CALL_KW) == set(ov.NUMBA_COLS) - set(bench.STATIC_SKIPS)
    for col in ov.CALL_KW:
        calls.clear()
        ov.NUMBA_COLS[col].runner(ctx)
        ov.TRITON_RUNNERS[col](ctx)
        (_, n_args, n_kw), (_, t_args, t_kw) = calls
        assert n_args == t_args, col
        n_tpb = n_kw.pop("threads_per_block")
        t_tpb = t_kw.pop("threads_per_block")
        assert n_kw == t_kw == ov.CALL_KW[col], col
        if col == "ch02_pinned":
            assert (n_tpb, t_tpb) == (768, ov.tff2.PINNED_TPB)
        else:
            assert n_tpb == t_tpb == bench.TPB, col
    calls.clear()
    ov.driver_runner(ov._TRITON_FF["ch05"], "ch05_ccl", bench.TPB,
                     blocks=48)(ctx)
    assert calls[0][2] == {**ov.CALL_KW["ch05_ccl"],
                           "threads_per_block": bench.TPB, "blocks": 48}


def test_cell_dispatch_matches_bench_row(monkeypatch):
    """bench.bench_row with its cell functions replaced by recorders: the
    mode it picks for every column must be the mode cell_mode() picks."""
    def tag(mode):
        def cell(*args, **kw):
            m = "est_pair" if kw.get("pair") else mode
            return {"skip": None, "ms": 1.0, "mode": m,
                    "est": m.startswith("est")}
        return cell

    monkeypatch.setattr(bench, "_cell_gpu", tag("measured"))
    monkeypatch.setattr(bench, "_cell_gpu_loop", tag("loop"))
    monkeypatch.setattr(bench, "_cell_gpu_est", tag("est"))
    monkeypatch.setattr(bench, "_cell_pure", lambda *a: {"skip": "cpu"})
    monkeypatch.setattr(bench, "_cell_njit", lambda *a: {"skip": "cpu"})
    for key, family, note, est, build in _rows(
            "comb_24", "two_sq_300", "random_1000").values():
        row = bench.bench_row(key, family, note, est, build)
        kind = ov.row_kind(build)
        assert row["kind"] == kind
        for col, *_ in bench.COLS:
            mode, reason = ov.cell_mode(col, kind)
            cell = row["cells"][col]
            if mode == "skip":
                assert cell == {"skip": reason}, (key, col)
            else:
                assert cell["mode"] == mode, (key, col)
        assert ov.cell_mode(ov.MATCHED, kind) == ov.cell_mode(
            "ch02_pinned", kind)


def _fake_ctx(n_seeds, width=300, height=200):
    img = np.broadcast_to(np.zeros(3, np.uint8), (width, height, 3))
    seeds = [(i % width, i // width) for i in range(n_seeds)]
    return {"kind": "n", "img": img, "seeds": seeds, "n_blobs": n_seeds}


@pytest.mark.parametrize("n_seeds,pair,width,height", [
    (40, False, 300, 200), (40, True, 300, 200), (7, False, 300, 200),
    (5, True, 300, 200), (3000, False, 5000, 4001),
    (3000, True, 5000, 4001)])
def test_est_sample_matches_bench(n_seeds, pair, width, height):
    """est_subs() reads bench._cell_gpu_est's sample and call count,
    including the smaller sample past 20 Mpx (a regression guard)."""
    ctx = _fake_ctx(n_seeds, width, height)
    seen = []

    def runner(sub):
        seen.append(sub)
        return SimpleNamespace(kernel_ms=1.0)

    cell = bench._cell_gpu_est(runner, ctx, pair=pair)
    subs, n_calls = ov.est_subs(ctx, pair=pair)
    assert subs == seen
    assert n_calls == cell["calls"] == cell["ms"]
    assert len(subs) == cell["sample"]


def test_loop_calls_match_bench():
    ctx = _fake_ctx(2)
    ctx["kind"] = "two"
    seen = []

    def runner(sub):
        seen.append(sub)
        return SimpleNamespace(kernel_ms=1.0, filled=1)

    bench._cell_gpu_loop(runner, ctx)
    assert ov.loop_subs(ctx) == seen[:2]


def _raising(exc):
    def run(_ctx):
        raise exc
    return run


def test_typed_skips_follow_bench():
    assert ov.bench_skip_reason(RuntimeError("ring overflow: x")) == "overflow"
    assert ov.bench_skip_reason(RuntimeError("cooperative capacity")) == (
        "overflow")
    assert ov.bench_skip_reason(RuntimeError("tripwire")) == (
        "error:RuntimeError")
    # bench.py's RuntimeError branch comes first, so it sees these too
    assert ov.bench_skip_reason(NotImplementedError("x")) == (
        "error:NotImplementedError")
    assert ov.bench_skip_reason(ValueError("bad seed")) is None
    assert ov.skip_reason("ch06_rgb", RuntimeError("run table OVERFLOW")) == (
        "overflow")
    # bench.py's three cell functions classify alike (bench_skip_reason
    # reads _cell_gpu's)
    for exc in (RuntimeError("ring overflow"), RuntimeError("x"),
                NotImplementedError("y")):
        loop = bench._cell_gpu_loop(_raising(exc), _fake_ctx(2))
        est = bench._cell_gpu_est(_raising(exc), _fake_ctx(3))
        assert loop["skip"] == est["skip"] == ov.bench_skip_reason(exc)


# ---------------------------------------------------- light results

def _fake_side(kernel_ms, digests):
    it = iter(digests)

    def run(ctx):
        return SimpleNamespace(kernel_ms=kernel_ms, total_ms=kernel_ms + 1,
                               det={"img": next(it)},
                               obs={"blocks": 1, "tpb": 256})
    return run


def test_loop_and_est_fold_calls_and_compare_every_call():
    subs = [{}, {}, {}]
    loop = ov.run_cell("loop", _fake_side(2.0, "abc"), "ch06", {}, subs,
                       3, False)
    assert loop.kernel_ms == 6.0 and loop.total_ms == 9.0
    est = ov.run_cell("est", _fake_side(2.0, "abc"), "ch06", {}, subs,
                      1000, False)
    assert est.kernel_ms == 2000.0 and est.total_ms == 3000.0
    other = ov.run_cell("est", _fake_side(1.0, "abd"), "ch06", {}, subs,
                        1000, False)
    ok, detail = ov.same(est, other)
    assert not ok and detail == "calls[2].img: numba 'c' vs triton 'd'"
    assert ov.same(est, est) == (True, "")


def test_digest_tags_dtype_and_shape():
    a = np.arange(6, dtype=np.int32)
    assert ov.digest(a) == ov.digest(a.copy())
    assert ov.digest(a) != ov.digest(a.astype(np.int64))
    assert ov.digest(a) != ov.digest(a.reshape(2, 3))
    assert ov.digest(np.zeros((0,), np.int8)).startswith("sha1:")


def _only_plain(value):
    if isinstance(value, dict):
        return all(_only_plain(v) for v in value.values())
    if isinstance(value, list):
        return all(_only_plain(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def test_light_results_hold_no_arrays():
    """Every chapter's light result is digests and scalars only, and the
    two backends' digests agree on a tiny scene."""
    ctx = ov.build_row(_rows("two_sq_300")["two_sq_300"][4], "two")
    one = {"kind": "one", "img": ctx["img"], "sx": ctx["seeds"][0][0],
           "sy": ctx["seeds"][0][1]}
    for col in ("ch01_spill", "ch02_split", "ch03_conn4", "ch04_multi",
                "ch05_merge", "ch06_mask"):
        fam = ov._family(col)
        sub = one if fam in ("ch01", "ch02", "ch03") else ctx
        if fam == "ch06":
            n_run = ov._ch06_runner("numba", "mask")
            t_run = ov._ch06_runner("triton", "mask")
        else:
            n_run, t_run = ov.NUMBA_COLS[col].runner, ov.TRITON_RUNNERS[col]
        rn = ov.run_cell("measured", n_run, fam, sub, None, 1, False)
        rt = ov.run_cell("measured", t_run, fam, sub, None, 1, False)
        assert _only_plain(rn.det) and _only_plain(rt.det), col
        assert ov.same(rn, rt) == (True, ""), col
        if fam == "ch06":
            red = bench_ch06._red_count(ctx["img"])
            assert rn.det["painted"] == red and rn.det["still_red"] == 0
            assert ctx["_red_px"] == red
    ctx.clear()


def test_est_rows_carry_per_call_ms():
    row = {"scene": "r", "experiment": "ch01_spill", "config": {},
           "est": True, "calls": 100,
           "info": {"numba_blocks": 1, "triton_blocks": 1,
                    "numba_tpb": 256, "triton_tpb": 256},
           "numba": {"kernel_ms": {"median": 200.0},
                     "total_ms": {"median": 300.0}},
           "triton": {"kernel_ms": {"median": 100.0},
                      "total_ms": {"median": 150.0}}}
    kept, skip = ov.finish_row(row, {"clocks_before": {"sm_mhz": 1}},
                               {"numba": 1.0, "triton": 2.0}, "ch01_spill")
    assert skip is None and kept["comparable"] is True
    assert kept["per_call_ms"] == {"numba": {"kernel": 2.0, "total": 3.0},
                                   "triton": {"kernel": 1.0, "total": 1.5}}
    assert kept["clocks_before"] == {"sm_mhz": 1}


# --------------------------------------------- crosscheck, report, groups

def test_crosscheck_and_problems():
    merge = {"scene": "s", "experiment": "ch05_merge", "outputs_equal": True,
             "info": {"filled": 10, "n_blobs": 1}}
    rgb = {"scene": "s", "experiment": "ch06_rgb", "outputs_equal": True,
           "expected_blobs": 1,
           "info": {"filled": 10, "still_red": 0, "red_px": 10,
                    "n_blobs": 1}}
    est = {"scene": "s", "experiment": "ch01_spill", "outputs_equal": True,
           "est": True, "info": {"filled": 3}}
    assert ov.crosscheck([merge, rgb, est])["s"] == {
        "status": "OK", "filled": [10], "failed_checks": []}
    cases = {
        "ch06_rgb.still_red": {"still_red": 2},
        "ch06_rgb.painted_eq_red_px": {"red_px": 11},
        "ch06_rgb.n_blobs": {"n_blobs": 2},
    }
    for name, change in cases.items():
        bad = dict(rgb, info={**rgb["info"], **change})
        cc = ov.crosscheck([merge, bad])["s"]
        assert cc["status"] == "MISMATCH" and cc["failed_checks"] == [name]
    fewer = dict(merge, info={"filled": 9})
    assert ov.crosscheck([fewer, rgb])["s"]["status"] == "MISMATCH"

    bad = dict(rgb, info={**rgb["info"], "still_red": 2})
    doc = {"rows": [merge, bad],
           "meta": {"crosscheck": ov.crosscheck([merge, bad]),
                    "skipped_cells": [
                        {"row": "s", "column": "a",
                         "reason": "error:KeyError"},
                        {"row": "s", "column": "b", "reason": "overflow"}]}}
    found = ov.problems(doc)
    assert len(found) == 2
    assert "crosscheck MISMATCH" in found[0] and "error:KeyError" in found[1]


def test_groups_spin_before_ch05_and_ch06():
    def cells(spec):
        return [(None, None, None, c, r) for c, r in spec]

    g = ov.plan_groups(cells([("ch01_spill", 6), ("ch04_multi", 6),
                              ("ch05_merge", 6), ("ch05_ccl", 6),
                              ("ch06_rgb", 10), ("ch06_mask", 10)]),
                       2.0, 3.0)
    assert [(x["phase"], x["spin"], x["repeats"], len(x["members"]))
            for x in g] == [("ch01-ch04", 2.0, 6, 2), ("ch05", 3.0, 6, 2),
                            ("ch06", 3.0, 10, 2)]
    # a budget-reduced cell splits its phase without a spin; a row that
    # starts at ch05 spins the longer of the two
    g = ov.plan_groups(cells([("ch01_spill", 6), ("ch04_multi", 2)]),
                       2.0, 3.0)
    assert [x["spin"] for x in g] == [2.0, 0.0]
    g = ov.plan_groups(cells([("ch05_merge", 6)]), 2.0, 3.0)
    assert [x["spin"] for x in g] == [3.0]


def test_ch06_repeats_follow_bench_ch06():
    assert ov.default_repeats("ch06_rgb", 6, False) == ov.CH06_REPEATS
    assert ov.CH06_REPEATS == bench_ch06.ROUNDS + bench_ch06.ROUNDS % 2
    assert ov.default_repeats("ch06_rgb", 2, True) == 2
    assert ov.default_repeats("ch05_merge", 6, False) == 6


# ------------------------------------------------- runtime skip paths

def test_runtime_skip_is_decided_by_numba(monkeypatch):
    """bench.py's typed refusal on the Numba side's first call: a skip
    record (not a row), with the Triton side's probe recorded."""
    boom = _raising(RuntimeError("ring overflow: capacity 4096"))
    monkeypatch.setattr(ov.NUMBA_COLS["ch01_ring"], "runner", boom)
    monkeypatch.setitem(ov.TRITON_RUNNERS, "ch01_ring", boom)
    doc = ov.main(["--quick", "--rows", "comb_24", "--cols", "ch01_ring"])
    assert doc["rows"] == []
    (skip,) = [s for s in doc["meta"]["skipped_cells"]
               if s["source"] == "runtime"]
    assert (skip["row"], skip["column"], skip["reason"]) == (
        "comb_24", "ch01_ring", "overflow")
    assert skip["triton"].startswith("overflow: RuntimeError: ring overflow")
    assert ov.problems(doc) == []


def test_runtime_error_skip_is_not_ok(monkeypatch):
    monkeypatch.setattr(ov.NUMBA_COLS["ch01_ring"], "runner",
                        _raising(RuntimeError("tripwire")))
    doc = ov.main(["--quick", "--rows", "comb_24", "--cols", "ch01_ring"])
    (skip,) = doc["meta"]["skipped_cells"]
    assert skip["reason"] == "error:RuntimeError"
    assert skip["triton"] == "ran (skipped anyway: bench.py's rule)"
    assert len(ov.problems(doc)) == 1


def test_triton_only_refusal_is_an_error_row(monkeypatch):
    monkeypatch.setitem(ov.TRITON_RUNNERS, "ch01_ring",
                        _raising(RuntimeError("ring overflow")))
    doc = ov.main(["--quick", "--rows", "comb_24", "--cols", "ch01_ring"])
    (row,) = doc["rows"]
    assert row["error"].startswith("warm-up failed: RuntimeError")
    assert doc["meta"]["skipped_cells"] == []
    assert len(ov.problems(doc)) == 1


def test_bookkeeping_failure_is_an_error_not_a_skip(monkeypatch):
    """An exception in this file's digest code (not the driver's or the
    engine's) must never become a skip, for ch01-ch05 or for ch06."""
    def broken(_r):
        raise KeyError("no such field")
    monkeypatch.setitem(ov._DET, "ch01", broken)

    def broken_counts(*_a):
        raise AttributeError("no such view")
    monkeypatch.setattr(ov, "paint_counts", broken_counts)
    doc = ov.main(["--quick", "--rows", "comb_24",
                   "--cols", "ch01_spill,ch06_mask"])
    by = {r["experiment"]: r for r in doc["rows"]}
    assert by["ch01_spill"]["error"].startswith("warm-up failed: KeyError")
    assert by["ch06_mask"]["error"].startswith(
        "warm-up failed: AttributeError")
    assert doc["meta"]["skipped_cells"] == []
    assert len(ov.problems(doc)) == 2


def test_host_memory_preflight_skips_big_rows(monkeypatch):
    ref, _ = ov.load_reference()
    if not ov._ref_pixels(ref, "asym_4000_800"):
        pytest.skip("no Numba overview reference JSON to size the row")
    monkeypatch.setattr(ov, "mem_available_mb", lambda: 1)
    doc = ov.main(["--quick", "--rows", "asym_4000_800",
                   "--cols", "ch01_spill"])
    assert doc["rows"] == []
    (skip,) = doc["meta"]["skipped_cells"]
    assert (skip["reason"], skip["source"]) == ("host-memory", "preflight")
    assert any(c.startswith("asym_4000_800 not measured")
               for c in doc["meta"]["caps"])


def test_partial_json_then_final(monkeypatch, tmp_path):
    """Per-row snapshots go to partial.json with an INCOMPLETE cap;
    compare_<stamp>.json appears only at completion, without it."""
    monkeypatch.setattr(ov.results_paths, "results_dir",
                        lambda *a: str(tmp_path))
    for k in ("SPIN_SECONDS", "ROW_SPIN_SECONDS", "PHASE_SPIN_SECONDS"):
        monkeypatch.setattr(ov, k, 0.0)
    snaps = []
    real = ov._write_json

    def spy(path, doc):
        snaps.append((os.path.basename(path), json.loads(json.dumps(doc))))
        real(path, doc)
    monkeypatch.setattr(ov, "_write_json", spy)
    doc = ov.main(["--rows", "sq_256,comb_24", "--cols", "ch01_spill",
                   "--repeats", "1", "--budget-min", "0"])
    final = f"compare_{doc['created_utc']}.json"
    assert [n for n, _ in snaps] == ["partial.json", "partial.json", final,
                                     "partial.json"]
    first = snaps[0][1]["meta"]
    assert first["complete"] is False
    assert first["caps"][-1].startswith("INCOMPLETE: 1 of 2 rows")
    written = json.loads((tmp_path / final).read_text())
    assert written["meta"]["complete"] is True
    assert not any("INCOMPLETE" in c for c in written["meta"]["caps"])
    assert len(written["rows"]) == 2
    assert json.loads((tmp_path / "partial.json").read_text()) == {
        "complete": True, "final": final}


# ---------------------------------------------------------- end to end

def test_tiny_row_end_to_end():
    """comb_24 through main() on every column, both backends: every cell
    measured with equal digests, bench.py's three skips recorded, every
    grid pinned to the common capacity, ch06's absolute checks met."""
    doc = ov.main(["--quick", "--rows", "comb_24"])
    rows = doc["rows"]
    json.dumps(doc)                                  # the JSON it would write
    skips = doc["meta"]["skipped_cells"]
    assert len(rows) + len(skips) == len(ov.TABLE_COLUMNS)
    assert {(s["row"], s["column"], s["reason"]) for s in skips} == {
        ("comb_24", "ch04_seq", "na"), ("comb_24", "ch04_multi", "na"),
        ("comb_24", "ch04_streams", "unsupported")}
    for r in rows:
        assert "error" not in r, (r["experiment"], r["error"])
        assert r["outputs_equal"], (r["experiment"],
                                    r.get("mismatch_detail"))
        assert r["config"]["resolved_blocks"]["numba"] >= 1
    by = {r["experiment"]: r for r in rows}
    assert by["ch02_pinned"]["comparable"] is False
    for col, r in by.items():
        if col != "ch02_pinned":
            assert r["comparable"] is True, col
    for col in ov.GRID_COLS:
        if col in by:
            cfg = by[col]["config"]
            assert cfg["blocks"] == min(cfg["caps"].values()), col
            assert cfg["resolved_blocks"] == {"numba": cfg["blocks"],
                                              "triton": cfg["blocks"]}, col
    assert by[ov.MATCHED]["config"]["resolved_tpb"] == {
        "numba": ov.tff2.PINNED_TPB, "triton": ov.tff2.PINNED_TPB}
    for col in ov.CH06_COLS:
        info = by[col]["info"]
        assert info["still_red"] == 0 and info["filled"] == info["red_px"]
    assert doc["meta"]["crosscheck"]["comb_24"]["status"] == "OK"
    assert ov.problems(doc) == []


def test_loop_cell_end_to_end():
    """A one-blob kernel on the two-blob row: one call per blob, summed;
    ch04 pinned to the common grid, so comparable."""
    doc = ov.main(["--quick", "--rows", "two_sq_300",
                   "--cols", "ch01_spill,ch04_multi"])
    by = {r["experiment"]: r for r in doc["rows"]}
    loop = by["ch01_spill"]
    assert loop["config"]["cell"] == "loop" and loop["calls"] == 2
    assert loop["outputs_equal"] and by["ch04_multi"]["outputs_equal"]
    assert loop["info"]["filled"] == by["ch04_multi"]["info"]["filled"]
    assert by["ch04_multi"]["comparable"] is True
