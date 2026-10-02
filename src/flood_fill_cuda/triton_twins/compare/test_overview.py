"""The overview twin: coverage of bench.py's columns, parity of its cell
rules with bench.py's own, and one tiny row end to end on both backends.

Fast by design (well under a minute, most of it JIT): the parity tests
drive bench.py's own functions with fake runners, and only the last three
tests launch kernels: every column on the 96 x 64 comb row, and two
columns of the 700 x 400 two-squares row for the per-blob loop and ch04.
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


def _fake_ctx(n_seeds, width=300, height=200):
    img = np.broadcast_to(np.zeros(3, np.uint8), (width, height, 3))
    seeds = [(i % width, i // width) for i in range(n_seeds)]
    return {"kind": "n", "img": img, "seeds": seeds, "n_blobs": n_seeds}


@pytest.mark.parametrize("n_seeds,pair,width,height", [
    (40, False, 300, 200), (40, True, 300, 200), (7, False, 300, 200),
    (5, True, 300, 200), (3000, False, 5000, 4001),
    (3000, True, 5000, 4001)])
def test_est_sample_matches_bench(n_seeds, pair, width, height):
    """est_subs() picks bench._cell_gpu_est's sample and call count,
    including the smaller sample past 20 Mpx."""
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
    ctx.clear()


# ---------------------------------------------------------- end to end

def test_tiny_row_end_to_end():
    """comb_24 through main() on every column, both backends: every cell
    measured with equal digests, bench.py's three skips recorded."""
    doc = ov.main(["--quick", "--rows", "comb_24"])
    rows = doc["rows"]
    json.dumps(doc)                                  # the JSON it would write
    skips = doc["meta"]["skipped_cells"]
    assert len(rows) + len(skips) == len(ov.GPU_COLUMNS)
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
    assert by["ch01_ring"]["comparable"] is True
    assert doc["meta"]["crosscheck"]["comb_24"]["status"] == "OK"


def test_loop_cell_end_to_end():
    """A one-blob kernel on the two-blob row: one call per blob, summed."""
    doc = ov.main(["--quick", "--rows", "two_sq_300",
                   "--cols", "ch01_spill,ch04_multi"])
    by = {r["experiment"]: r for r in doc["rows"]}
    loop = by["ch01_spill"]
    assert loop["config"]["cell"] == "loop" and loop["calls"] == 2
    assert loop["outputs_equal"] and by["ch04_multi"]["outputs_equal"]
    assert loop["info"]["filled"] == by["ch04_multi"]["info"]["filled"]
