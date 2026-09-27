"""Audit every number the video prints against the committed JSON files.

A second, independent code path: plain dict lookups on the JSON, no use
of `scenes.panes.data` for the values, only for the formatting. Prints
each on-screen string next to its source and exits 1 on any mismatch.

Usage (from video/):  uv run build/audit_numbers.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEO))
from scenes.panes import data  # noqa: E402  (formatting + the loader under test)

RESULTS = VIDEO.parent / "src" / "flood_fill_cuda" / "results"
OV_DIR = RESULTS / "overview" / "benchmark_results"
CH06_DIR = RESULTS / "ch06_gpu_nblob_runs" / "benchmark_results"

STAGE_VARIANTS = {
    "ch01": ["ch01_ring", "ch01_spill"],
    "ch02": ["ch02_split", "ch02_global", "ch02_dirsplit", "ch02_pinned"],
    "ch03_conn4": ["ch03_conn4"],
    "ch03_conn8": ["ch03_conn8"],
    "ch04": ["ch04_multi"],
    "ch05": ["ch05_merge", "ch05_ccl", "ch05_fused_L8", "ch05_r128_L8", "ch05_split_L8", "ch05_split_I1"],
    "ch06": ["ch06_mask"],
}


def newest(folder, pattern, exclude=None):
    paths = sorted(glob.glob(str(folder / pattern)))
    if exclude:
        paths = [p for p in paths if exclude not in os.path.basename(p)]
    return paths[-1]


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

    print("\nmatrix (row: CPU ms | stage: chosen ms [glow]) — independent vs loader")
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
    ]
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
    print(f"\n{'ALL OK' if not bad else f'{bad} MISMATCHES'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
