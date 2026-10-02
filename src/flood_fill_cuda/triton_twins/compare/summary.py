"""
Roll every chapter's newest comparison JSON into one summary.

Reads results/triton_twins/<unit>/compare_*.json (newest per unit) and
writes results/triton_twins/summary.json: per unit and per experiment, the
geometric-mean speedup (numba_ms / triton_ms, so above 1 means Triton is
faster) on kernel_ms and total_ms, the extremes with their scene and
config, and whether every row's outputs were identical.

The geometric mean is the right average for ratios: a 2x win and a 2x
loss cancel to 1.0, where an arithmetic mean would call it 1.25.

Run:  python -m flood_fill_cuda.triton_twins.compare.summary
"""

import glob
import json
import math
import os

from flood_fill_cuda.shared.results_paths import RESULTS_ROOT

TWINS_ROOT = os.path.join(RESULTS_ROOT, "triton_twins")


def newest_per_unit(root=TWINS_ROOT):
    """{unit: path of its newest compare_*.json}, units sorted by name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "compare_*.json"))):
        out[os.path.basename(os.path.dirname(path))] = path  # sorted: last wins
    return dict(sorted(out.items()))


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _extreme(rows, key, pick):
    row = pick(rows, key=lambda r: r[key])
    return {"speedup": row[key], "experiment": row["experiment"],
            "scene": row["scene"], "config": row["config"],
            "numba_ms": row["numba"]["kernel_ms"]["median"],
            "triton_ms": row["triton"]["kernel_ms"]["median"]}


def summarize_rows(rows):
    measured = [r for r in rows if "error" not in r]
    # Not like-for-like rows are reported but never averaged.
    ok = [r for r in measured if r.get("comparable", True)]
    out = {
        "rows": len(rows),
        "errors": [{"experiment": r["experiment"], "scene": r["scene"],
                    "config": r["config"], "error": r["error"]}
                   for r in rows if "error" in r],
        "not_comparable_rows": len(measured) - len(ok),
        "all_outputs_equal": all(r.get("outputs_equal") for r in measured),
        "unequal_rows": [{"experiment": r["experiment"], "scene": r["scene"],
                          "config": r["config"],
                          "detail": r.get("mismatch_detail", "")}
                         for r in measured if not r.get("outputs_equal")],
    }
    if ok:
        out["geomean_speedup_kernel"] = geomean(r["speedup_kernel"] for r in ok)
        out["geomean_speedup_total"] = geomean(r["speedup_total"] for r in ok)
        out["triton_faster_rows"] = sum(r["speedup_kernel"] > 1 for r in ok)
        out["best_for_triton"] = _extreme(ok, "speedup_kernel", max)
        out["best_for_numba"] = _extreme(ok, "speedup_kernel", min)
    return out


def summarize(root=TWINS_ROOT):
    units = {}
    for unit, path in newest_per_unit(root).items():
        with open(path) as f:
            doc = json.load(f)
        rows = doc["rows"]
        experiments = {}
        for r in rows:
            experiments.setdefault(r["experiment"], []).append(r)
        units[unit] = {
            "source": os.path.relpath(path, RESULTS_ROOT),
            "created_utc": doc["created_utc"],
            "git_commit": doc.get("git_commit"),
            "repeats": doc["repeats"],
            "caps": doc.get("meta", {}).get("caps"),
            **summarize_rows(rows),
            "experiments": {k: summarize_rows(v)
                            for k, v in experiments.items()},
        }
    every = [r for u, p in newest_per_unit(root).items()
             for r in json.load(open(p))["rows"] if "error" not in r]
    return {
        "speedup_definition": "numba_ms / triton_ms; above 1 = Triton faster",
        "units": units,
        "overall": summarize_rows(every) if every else {},
    }


def main():
    doc = summarize()
    path = os.path.join(TWINS_ROOT, "summary.json")
    os.makedirs(TWINS_ROOT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    for unit, u in doc["units"].items():
        g = u.get("geomean_speedup_kernel")
        print(f"{unit:28s} rows {u['rows']:4d}  kernel x{g:.2f}" if g else
              f"{unit:28s} rows {u['rows']:4d}  (no timed rows)",
              "" if u["all_outputs_equal"] else "  OUTPUTS DIFFER")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
