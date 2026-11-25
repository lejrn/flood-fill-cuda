"""The library-baselines session's harness, CPU only (its GPU side needs
the device and is exercised by a --smoke run).

Run:

    uv run pytest src/flood_fill_cuda/chapters/ch09_cpu_nthreads/benchmarks/test_library_baselines.py -v
"""

import numpy as np
import pytest

from . import library_baselines as lb
from ...ch05_gpu_nblob_nblock import scenes as ch05_scenes
from ...ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components


def _noise(seed=3):
    return ch05_scenes.random_blobs_scene(70, 90, 0.35, seed)[0]


def test_same_partition_accepts_a_relabelling_and_nothing_else():
    img = _noise()
    ref, n = cpu_label_components(img)
    ids = np.unique(ref[ref >= 0])
    assert n == ids.size >= 3
    # the same components under other names (1..n, background 0)
    other = np.zeros(ref.shape, np.int32)
    for i, v in enumerate(ids[::-1]):
        other[ref == v] = i + 1
    assert lb.same_partition(other, 0, ref, n)
    merged = other.copy()
    merged[merged == 2] = 1                     # two blobs under one name
    assert not lb.same_partition(merged, 0, ref, n)
    split = other.copy()
    x, y = np.argwhere(ref == ids[np.argmax(
        [(ref == v).sum() for v in ids])])[0]
    split[x, y] = n + 1                         # one pixel of a blob apart
    assert not lb.same_partition(split, 0, ref, n)
    missing = other.copy()
    missing[x, y] = 0                           # a red pixel left unlabelled
    assert not lb.same_partition(missing, 0, ref, n)


def test_every_cpu_cell_matches_the_oracle_and_names_its_contract():
    img = _noise(5)
    ref = lb.reference(img)
    cells = lb.cpu_cells(img, ref, [1, 2])
    keys = [c["key"] for c in cells]
    assert len(keys) == len(set(keys))
    assert {c["contract"] for c in cells} == {"labelmap", "rgb_labelmap",
                                              "recolor"}
    for c in cells:
        if c["prep"]:
            c["prep"]()
        assert c["check"](c["call"]()), c["key"]


def test_a_wrong_output_fails_its_check():
    img = _noise(7)
    ref = lb.reference(img)
    cells = {c["key"]: c for c in lb.cpu_cells(img, ref, [1])}
    n, lab = cells["opencv_labelmap_t1"]["call"]()
    lab[lab == 2] = 1
    assert not cells["opencv_labelmap_t1"]["check"]((n, lab))
    out = cells["oracle_labelmap"]["call"]()
    out[out >= 0] = out[out >= 0].min()
    assert not cells["oracle_labelmap"]["check"](out)


def test_smoke_session_on_the_cpu_checks_every_output():
    session = lb.main(["--smoke", "--no-gpu", "--no-save", "--threads", "1",
                       "2", "--scenes", "random_4000", "sq_8000"])
    assert session["all_checks_ok"] and session["smoke"]
    for row in session["rows"]:
        assert row["all_checks_ok"]
        assert set(row["summary"]) == {"labelmap", "rgb_labelmap", "recolor"}
        for contract in row["summary"].values():
            assert min(c["x_fastest"] for c in contract.values()) == 1.0


def test_summary_compares_only_inside_a_contract():
    cells = {"a": {"ms": 2.0, "contract": "labelmap"},
             "b": {"ms": 8.0, "contract": "labelmap"},
             "c": {"ms": 1.0, "contract": "recolor"}}
    s = lb.summarize(cells)
    assert s["labelmap"]["b"]["x_fastest"] == pytest.approx(4.0)
    assert s["recolor"]["c"]["x_fastest"] == 1.0 and "rgb_labelmap" not in s
