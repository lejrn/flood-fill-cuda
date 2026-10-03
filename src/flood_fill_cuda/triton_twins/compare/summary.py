"""
Roll every chapter's newest comparison JSON into one summary.

Reads results/triton_twins/<unit>/compare_*.json (newest per unit) and
writes results/triton_twins/summary.json: per unit and per experiment, the
geometric-mean speedup (numba_ms / triton_ms, so above 1 means Triton is
faster) on kernel_ms and total_ms, the extremes with their scene and
config, and whether every row's outputs were identical.

The geometric mean is the right average for ratios: a 2x win and a 2x
loss cancel to 1.0, where an arithmetic mean would call it 1.25.

Two kinds of rows never enter any average: rows marked
first_translation=true (the twins' first translation of a construct,
kept as an ablation: they are summarized on their own, paired with the
default row they differ from) and rows marked duplicate_of=<experiment>
(a cell another experiment already measures).

"overall" weighs every unit equally (the geometric mean of the units'
geometric means), so the grand table's 300-odd cells cannot drown out a
chapter with 30. Rows marked comparable=false are reported per unit as
"own default" figures (each backend at its own default grid) and never
enter the like-for-like averages.

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
            "triton_ms": row["triton"]["kernel_ms"]["median"],
            # projected from a per-blob sample (grand table), not one call
            "est": bool(row.get("est") or row.get("config", {}).get("est"))}


ABLATION_KEYS = ("enqueue", "label", "schedule", "lane_sched", "LANE_SCHED",
                 "lane_schedule", "duplicate_of")


def _is_ablation(r):
    return bool(r.get("first_translation"))


def _is_duplicate(r):
    # ch04 records the marker inside config (as a list), the others on the row
    return bool(r.get("duplicate_of") or r.get("config", {}).get("duplicate_of"))


def _pair_key(r):
    cfg = {k: v for k, v in r.get("config", {}).items()
           if k not in ABLATION_KEYS}
    return (r["experiment"], r["scene"], json.dumps(cfg, sort_keys=True))


def ablation(rows, experiment):
    """One experiment's first-translation rows against the default rows
    they differ from: same experiment, scene and config apart from the
    ablated setting. Partners may be duplicate_of rows."""
    measured = [r for r in rows if "error" not in r]
    first = [r for r in measured
             if _is_ablation(r) and r["experiment"] == experiment]
    default = {}
    for r in measured:
        if not _is_ablation(r):
            default.setdefault(_pair_key(r), r)
    pairs = [(f, default.get(_pair_key(f))) for f in first]
    pairs = [(f, d) for f, d in pairs if d is not None]
    return {
        "first_translation_rows": len(first),
        "paired": len(pairs),
        "geomean_speedup_kernel_first_translation":
            geomean(f["speedup_kernel"] for f, _ in pairs),
        "geomean_speedup_kernel_default":
            geomean(d["speedup_kernel"] for _, d in pairs),
        "geomean_triton_gain": geomean(
            f["triton"]["kernel_ms"]["median"] / d["triton"]["kernel_ms"]["median"]
            for f, d in pairs),
    }


def ablations(rows):
    """{experiment: ablation} for every experiment that holds
    first-translation rows; one entry per translation choice, never
    pooled (ch05 ablates both its enqueue and its union-find schedule)."""
    exps = sorted({r["experiment"] for r in rows
                   if "error" not in r and _is_ablation(r)})
    return {e: ablation(rows, e) for e in exps} or None


def summarize_rows(rows):
    measured = [r for r in rows if "error" not in r
                and not _is_ablation(r) and not _is_duplicate(r)]
    # Not like-for-like rows are reported but never averaged.
    ok = [r for r in measured if r.get("comparable", True)]
    out = {
        "rows": len(rows),
        "ablation_rows": sum(_is_ablation(r) for r in rows),
        "duplicate_rows": sum(_is_duplicate(r) for r in rows),
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
    if len(measured) > len(ok):
        out["own_default_geomean_speedup_kernel"] = geomean(
            r["speedup_kernel"] for r in measured)
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
            "first_translation_ablations": ablations(rows),
            "experiments": {k: summarize_rows(v)
                            for k, v in experiments.items()},
        }
    per_unit = [u["geomean_speedup_kernel"] for u in units.values()
                if u.get("geomean_speedup_kernel")]
    per_unit_total = [u["geomean_speedup_total"] for u in units.values()
                      if u.get("geomean_speedup_total")]
    return {
        "speedup_definition": "numba_ms / triton_ms; above 1 = Triton faster",
        "units": units,
        "overall": {
            "units": len(units),
            "geomean_of_unit_geomeans_kernel": geomean(per_unit),
            "geomean_of_unit_geomeans_total": geomean(per_unit_total),
            "all_outputs_equal": all(u["all_outputs_equal"]
                                     for u in units.values()),
        },
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
