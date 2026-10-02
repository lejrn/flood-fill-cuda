"""The comparison harness's bookkeeping, checked with fake backends."""

from types import SimpleNamespace

from flood_fill_cuda.triton_twins.compare.harness import (
    Case, arrays_equal, run_case, run_cases,
)


def _backend(ms, log, name, out):
    def run():
        log.append(name)
        return SimpleNamespace(kernel_ms=ms, total_ms=ms + 1, out=out)
    return run


def _same(a, b):
    return arrays_equal(out=(a.out, b.out))


def test_order_flips_every_round_and_speedup_is_numba_over_triton():
    log = []
    case = Case("e", "s", {"k": 1}, _backend(4.0, log, "N", [1]),
                _backend(2.0, log, "T", [1]), _same)
    row = run_case(case, repeats=4)
    # warm-up N T, then rounds N T | T N | N T | T N
    assert log == ["N", "T", "N", "T", "T", "N", "N", "T", "T", "N"]
    assert row["speedup_kernel"] == 2.0
    assert row["numba"]["total_ms"]["median"] == 5.0
    assert row["outputs_equal"] and row["mismatched_runs"] == 0


def test_mismatch_is_counted_and_described():
    log = []
    case = Case("e", "s", {}, _backend(1.0, log, "N", [1, 2]),
                _backend(1.0, log, "T", [1, 3]), _same)
    row = run_case(case, repeats=2)
    assert not row["outputs_equal"]
    assert row["mismatched_runs"] == 3  # warm-up + 2 rounds
    assert "1 elements differ" in row["mismatch_detail"]


def test_backend_error_is_recorded_not_raised():
    def boom():
        raise RuntimeError("tripwire")
    case = Case("e", "s", {}, boom, boom, _same)
    doc = run_cases("chXX", [case], repeats=1, write=False, log=lambda m: None)
    assert "tripwire" in doc["rows"][0]["error"]
    assert doc["versions"]["triton"]
