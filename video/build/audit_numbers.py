"""Audit every number the video prints against the committed JSON files.

A second, independent code path: plain dict lookups on the JSON, no use
of `scenes.panes.data` for the values, only for the formatting. Prints
each on-screen string next to its source and exits 1 on any mismatch.

The Triton stage is audited the same way: every twin cell recomputed from
the grand table's rows, and the headline (overall ratio, rows faster,
ch05) from the per-unit compare JSONs, not from summary.json.

Usage (from video/):  uv run build/audit_numbers.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO))
from scenes.panes import data  # noqa: E402  (formatting + the loader under test)

RESULTS = VIDEO.parent / "src" / "flood_fill_cuda" / "results"
OV_DIR = RESULTS / "overview" / "benchmark_results"
CH06_DIR = RESULTS / "ch06_gpu_nblob_runs" / "benchmark_results"
TWINS_DIR = RESULTS / "triton_twins"
SCRIPT = VIDEO / "narration" / "script.md"

STAGE_VARIANTS = {
    "ch01": ["ch01_ring", "ch01_spill"],
    "ch02": ["ch02_split", "ch02_global", "ch02_dirsplit", "ch02_pinned"],
    "ch03_conn4": ["ch03_conn4"],
    "ch03_conn8": ["ch03_conn8"],
    "ch04": ["ch04_multi"],
    "ch05": ["ch05_merge", "ch05_ccl", "ch05_fused_L8", "ch05_r128_L8", "ch05_split_L8", "ch05_split_I1"],
    "ch06": ["ch06_mask"],
}


# the grand table's like-for-like columns per stage (ch02 pinned runs 2 x 512 on both sides)
TWIN_VARIANTS = {
    "ch01": ["ch01_ring", "ch01_spill"],
    "ch02": ["ch02_split", "ch02_global", "ch02_dirsplit", "ch02_pinned_matched"],
    "ch03_conn4": ["ch03_conn4"],
    "ch03_conn8": ["ch03_conn8"],
    "ch04": ["ch04_multi"],
    "ch05": ["ch05_merge", "ch05_ccl", "ch05_fused_L8", "ch05_r128_L8", "ch05_split_L8", "ch05_split_I1"],
    "ch06": ["ch06_mask"],
}
NUMBER_WORDS = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen"}


def newest(folder, pattern, exclude=None):
    paths = sorted(glob.glob(str(folder / pattern)))
    if exclude:
        paths = [p for p in paths if exclude not in os.path.basename(p)]
    return paths[-1]


def like(r) -> bool:
    """Averaged rows: no error, not a first-translation ablation, not a repeat, like-for-like."""
    return ("error" not in r and not r.get("first_translation")
            and not r.get("duplicate_of") and not r.get("config", {}).get("duplicate_of")
            and r.get("comparable", True))


def kernel_ratio(r) -> float:
    return r["numba"]["kernel_ms"]["median"] / r["triton"]["kernel_ms"]["median"]


def audit_twins() -> int:
    bad = 0
    grand_path = newest(TWINS_DIR / "overview", "compare_*.json")
    grand = json.load(open(grand_path))
    tw = data.load_twins()
    if tw.source != os.path.basename(grand_path):
        bad += 1
        print(f"loader read {tw.source}, the newest grand table is {os.path.basename(grand_path)}")
    print(f"\nTriton stage: {os.path.basename(grand_path)}, every unit's newest compare_*.json")
    print("matrix (numba_ms / triton_ms, each side's fastest like-for-like variant) - independent vs loader")
    rows = {}
    for r in grand["rows"]:
        rows.setdefault((r["row"], r["config"]["column"]), []).append(r)
    for row in grand["meta"]["rows"]:
        line = f"{row:14s}"
        for stage, variants in TWIN_VARIANTS.items():
            cands = [r for v in variants for r in rows.get((row, v), []) if like(r)]
            measured = [r for r in cands if not r.get("est")]
            c = tw.cell(row, stage)
            if measured:
                ratio = (min(r["numba"]["kernel_ms"]["median"] for r in measured)
                         / min(r["triton"]["kernel_ms"]["median"] for r in measured))
                txt = data.fmt_twin(ratio) + ("*" if ratio > 1 else " ")
                ok = c.ratio is not None and abs(c.ratio - ratio) < 1e-9 and c.glows == (ratio > 1)
            elif cands:
                txt, ok = "~est ", c.kind == "est"
            else:
                txt, ok = " n/a ", c.kind == "na"
            if not ok:
                bad += 1
                txt += "!!"
            line += f" | {stage[-5:]:>5s} {txt:>7s}"
        print(line)
    if tw.cell("png_blobs", "cpu").kind != "na":
        bad += 1
        print("the CPU column has no twin, but its cell is not n/a")

    unit_means, timed, n_like, n_faster, equal = {}, 0, 0, 0, True
    for path in sorted(glob.glob(str(TWINS_DIR / "*" / "compare_*.json"))):
        unit = os.path.basename(os.path.dirname(path))
        unit_means[unit] = path                      # sorted: the newest wins
    for unit, path in sorted(unit_means.items()):
        doc = json.load(open(path))
        ok_rows = [r for r in doc["rows"] if like(r)]
        timed += len(doc["rows"])
        n_like += len(ok_rows)
        n_faster += sum(kernel_ratio(r) > 1 for r in ok_rows)
        equal &= all(r.get("outputs_equal") for r in ok_rows)
        unit_means[unit] = math.exp(sum(math.log(kernel_ratio(r)) for r in ok_rows) / len(ok_rows))
    overall = math.exp(sum(math.log(v) for v in unit_means.values()) / len(unit_means))
    ch05 = unit_means["ch05_gpu_nblob_nblock"]
    print("\nTriton headline strings")
    checks = [
        ("overall", f"{data.fmt_twin(overall)}×", f"{data.fmt_twin(tw.overall)}×", abs(overall - tw.overall) < 1e-6),
        ("units", str(len(unit_means)), str(tw.units), len(unit_means) == tw.units),
        ("faster", f"faster in {n_faster:,} of {n_like:,} cases",
         f"faster in {tw.rows_faster:,} of {tw.rows_like:,} cases",
         (n_faster, n_like) == (tw.rows_faster, tw.rows_like)),
        ("timed", f"{timed:,}", f"{tw.rows_timed:,}", timed == tw.rows_timed),
        ("ch05", f"chapter 5 still slower: {data.fmt_twin(ch05)}×",
         f"chapter 5 still slower: {data.fmt_twin(tw.ch05)}×", abs(ch05 - tw.ch05) < 1e-6 and ch05 < 1),
        ("outputs", "identical" if equal else "DIFFER", "identical" if tw.all_equal else "DIFFER",
         equal and tw.all_equal),
    ]
    for name, want, got, ok in checks:
        bad += 0 if ok and want == got else 1
        print(f"  {name:8s} {want:>32s}   {'OK' if ok and want == got else 'MISMATCH (loader: ' + got + ')'}")
    pct = round((overall - 1) * 100)
    spoken = f"{NUMBER_WORDS.get(pct, str(pct))} percent faster overall"
    said = spoken in SCRIPT.read_text(encoding="utf-8")
    bad += 0 if said else 1
    print(f"  narration '{spoken}' (x{overall:.4f}) in script.md: {'OK' if said else 'MISSING'}")
    return bad


def main() -> int:
    bad = 0
    ov_path = newest(OV_DIR, "overview_*.json", exclude="ch06")
    c6_path = newest(OV_DIR, "ch06_overview_*.json")
    runs_path = newest(CH06_DIR, "runs_*.json")
    ov = json.load(open(ov_path))
    c6 = {r["row"]: r for r in json.load(open(c6_path))["rows"]}
    runs = json.load(open(runs_path))
    bench = data.load_bench()
    assert bench.source_overview == os.path.basename(ov_path), bench.source_overview
    assert bench.source_ch06 == os.path.basename(c6_path), bench.source_ch06

    print(f"sources: {os.path.basename(ov_path)}, {os.path.basename(c6_path)}, {os.path.basename(runs_path)}")
    print(f"sm_count {ov['sm_count']} -> pane title 'the GPU · {bench.sm_count} SMs'")
    if ov["sm_count"] != bench.sm_count:
        bad += 1

    print("\nmatrix (row: CPU ms | stage: chosen ms [glow]) - independent vs loader")
    for row in ov["rows"]:
        key = row["row"]
        cells = dict(row["cells"])
        cells.update(c6[key]["cells"])
        njit = cells["njit"]["ms"]
        brow = next(r for r in bench.rows if r.key == key)
        line = f"{key:14s} CPU {data.fmt_cell_ms(njit):>5s}"
        if abs(brow.njit_ms - njit) > 1e-9:
            bad += 1
            line += "  <- njit MISMATCH"
        for stage, variants in STAGE_VARIANTS.items():
            measured = [(cells[v]["ms"], v) for v in variants
                        if v in cells and cells[v].get("skip") is None and not cells[v].get("est")]
            est = [(cells[v]["ms"], v) for v in variants
                   if v in cells and cells[v].get("skip") is None and cells[v].get("est")]
            bcell = brow.cells[stage]
            if measured:
                ms, v = min(measured)
                glow = ms < njit
                txt = data.fmt_cell_ms(ms) + ("*" if glow else " ")
                ok = (bcell.ms is not None and abs(bcell.ms - ms) < 1e-9 and bcell.variant == v
                      and bcell.glows == glow and not bcell.est)
            elif est:
                ms, v = min(est)
                txt = "~est "
                ok = bcell.est and abs(bcell.ms - ms) < 1e-9 and not bcell.glows
            else:
                txt = " n/a "
                ok = bcell.kind == "na"
            if not ok:
                bad += 1
                txt += "!!"
            line += f" | {stage[-5:]:>5s} {txt:>7s}"
        print(line)

    print("\nheadline strings")
    png = next(r for r in ov["rows"] if r["row"] == "png_blobs")["cells"]
    sc = next(s for s in runs["scenes"] if s["scene"] == "input_blobs")
    h = data.headline()
    checks = [
        ("pure Python", png["pure_python"]["ms"], h.pure_ms, data.fmt_ms),
        ("@njit", png["njit"]["ms"], h.njit_ms, data.fmt_ms),
        ("ch05", sc["ch05"]["median_ms"], h.ch05_ms, data.fmt_ms),
        ("ch06 mask", sc["ch06"]["mask"]["median_ms"], h.ch06_mask_ms, data.fmt_ms),
        ("ch06 rgb", sc["ch06"]["rgb"]["median_ms"], h.ch06_rgb_ms, data.fmt_ms),
        ("blobs", sc["n_blobs"], h.n_blobs, lambda x: f"{x:,}"),
        ("red px", sc["red_px"], h.red_px, lambda x: f"{x:,}"),
        ("runs", sc["n_runs"], h.n_runs, lambda x: f"{x:,}"),
        ("pixels", sc["n_pixels"], h.n_pixels, lambda x: f"{x:,}"),
    ]
    print(f"  frame budget  {1000 / 30:14.1f} ms at 30 fps -> shown '33 ms'; "
          f"{sc['n_pixels'] / 1e6:.0f} Mpx per frame")
    for name, raw, got, fmt in checks:
        ok = abs(float(raw) - float(got)) < 1e-9
        bad += 0 if ok else 1
        print(f"  {name:12s} {fmt(raw):>14s}   {'OK' if ok else 'MISMATCH'}")
    ratio = png["pure_python"]["ms"] / sc["ch06"]["mask"]["median_ms"]
    runs_ratio = sc["red_px"] / sc["n_runs"]
    print(f"  ratio        {ratio:14,.0f}   shown '{data.fmt_ratio_round(ratio)}' "
          f"(loader '{data.fmt_ratio_round(h.ratio)}')")
    print(f"  px per run   {runs_ratio:14.1f}   shown '{runs_ratio:.0f}× fewer'")
    if data.fmt_ratio_round(ratio) != data.fmt_ratio_round(h.ratio):
        bad += 1
    bad += audit_twins()
    print(f"\n{'ALL OK' if not bad else f'{bad} MISMATCHES'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
