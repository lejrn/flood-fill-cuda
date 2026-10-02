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
    assert not doc["overall"]["all_outputs_equal"]
    # units weigh equally: ch01 geomean 1.0, ch02 1.0
    assert doc["overall"]["geomean_of_unit_geomeans_kernel"] == pytest.approx(1.0)


def test_not_comparable_rows_are_counted_but_not_averaged(tmp_path):
    root = str(tmp_path)
    odd = _row("placement", "pinned", 1.0, 10.0)
    odd["comparable"] = False
    _write(root, "ch02", "20260101T000000Z", [_row("s", "a", 2.0, 1.0), odd])
    u = summarize(root)["units"]["ch02"]
    assert u["geomean_speedup_kernel"] == pytest.approx(2.0)
    assert u["not_comparable_rows"] == 1
    assert u["best_for_numba"]["scene"] == "a"
    assert u["own_default_geomean_speedup_kernel"] == pytest.approx((2.0 * 0.1) ** 0.5)


def test_ablation_and_duplicate_rows_stay_out_of_every_average(tmp_path):
    root = str(tmp_path)
    lane = _row("enqueue", "sq", 4.0, 2.0)
    lane["config"] = {"tpb": 256, "enqueue": "lane", "label": "per_lane"}
    prog = _row("enqueue", "sq", 4.0, 8.0)
    prog["config"] = {"tpb": 256, "enqueue": "program", "label": "first_translation"}
    prog["comparable"] = False
    prog["first_translation"] = True
    dup = _row("enqueue", "sq", 4.0, 2.0)
    dup["duplicate_of"] = "scenes"
    main = _row("scenes", "sq", 4.0, 2.0)
    _write(root, "ch01", "20260101T000000Z", [main, lane, prog, dup])
    u = summarize(root)["units"]["ch01"]
    assert u["geomean_speedup_kernel"] == pytest.approx(2.0)
    assert "own_default_geomean_speedup_kernel" not in u  # nothing left over
    assert u["ablation_rows"] == 1 and u["duplicate_rows"] == 1
    ab = u["first_translation_ablations"]["enqueue"]
    assert ab["paired"] == 1
    assert ab["geomean_speedup_kernel_first_translation"] == pytest.approx(0.5)
    assert ab["geomean_speedup_kernel_default"] == pytest.approx(2.0)
    assert ab["geomean_triton_gain"] == pytest.approx(4.0)


def test_two_ablations_in_one_unit_are_reported_apart(tmp_path):
    root = str(tmp_path)
    rows = []
    for exp, key, first, default, t_first in (
            ("enqueue", "enqueue", "program", "lane", 8.0),
            ("lane_schedule", "lane_sched", "lockstep", "independent", 40.0)):
        d = _row(exp, "sq", 4.0, 2.0)
        d["config"] = {"tpb": 256, key: default}
        d["duplicate_of"] = "benchmark"
        f = _row(exp, "sq", 4.0, t_first)
        f["config"] = {"tpb": 256, key: first, "label": "first_translation"}
        f["comparable"] = False
        f["first_translation"] = True
        rows += [d, f]
    rows.append(_row("benchmark", "sq", 4.0, 2.0))
    _write(root, "ch05", "20260101T000000Z", rows)
    ab = summarize(root)["units"]["ch05"]["first_translation_ablations"]
    assert set(ab) == {"enqueue", "lane_schedule"}
    assert ab["enqueue"]["geomean_triton_gain"] == pytest.approx(4.0)
    assert ab["lane_schedule"]["geomean_triton_gain"] == pytest.approx(20.0)
