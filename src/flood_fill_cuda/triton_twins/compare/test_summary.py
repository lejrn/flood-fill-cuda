"""The summary's arithmetic, on a hand-made results tree."""

import json
import os

import pytest

from flood_fill_cuda.triton_twins.compare.summary import (
    geomean, newest_per_unit, summarize,
)


def _row(exp, scene, n_ms, t_ms, equal=True):
    return {"experiment": exp, "scene": scene, "config": {}, "pixels": 1,
            "numba": {"kernel_ms": {"median": n_ms}, "total_ms": {"median": n_ms}},
            "triton": {"kernel_ms": {"median": t_ms}, "total_ms": {"median": t_ms}},
            "speedup_kernel": n_ms / t_ms, "speedup_total": n_ms / t_ms,
            "outputs_equal": equal}


def _write(root, unit, stamp, rows):
    d = os.path.join(root, unit)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"compare_{stamp}.json"), "w") as f:
        json.dump({"created_utc": stamp, "repeats": 3, "rows": rows}, f)


def test_geomean_cancels_symmetric_ratios():
    assert geomean([2.0, 0.5]) == pytest.approx(1.0)
    assert geomean([]) is None


def test_newest_file_wins_and_extremes_are_named(tmp_path):
    root = str(tmp_path)
    _write(root, "ch01", "20260101T000000Z", [_row("a", "old", 1, 1)])
    _write(root, "ch01", "20260102T000000Z",
           [_row("scenes", "sq", 4.0, 2.0), _row("scenes", "serp", 1.0, 2.0),
            {"experiment": "scenes", "scene": "bad", "config": {},
             "error": "warm-up failed"}])
    _write(root, "ch02", "20260101T000000Z", [_row("x", "y", 3, 3, equal=False)])
    assert newest_per_unit(root)["ch01"].endswith("20260102T000000Z.json")
    doc = summarize(root)
    ch01 = doc["units"]["ch01"]
    assert ch01["rows"] == 3 and len(ch01["errors"]) == 1
    assert ch01["geomean_speedup_kernel"] == pytest.approx(1.0)
    assert ch01["best_for_triton"]["scene"] == "sq"
    assert ch01["best_for_numba"]["scene"] == "serp"
    assert ch01["triton_faster_rows"] == 1
    assert not doc["units"]["ch02"]["all_outputs_equal"]
    assert doc["overall"]["rows"] == 3
