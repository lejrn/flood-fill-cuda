"""The page's two bar charts, drawn from committed results.

chain_bfs.svg: every BFS stage on input_blobs.png. The repo's own chart
(results/ch06_gpu_nblob_runs/figures/chain.svg) ends on chapter 6, which
is not a BFS. The page tells the BFS story, so its chart ends on chapter
8's tile waves and shows chapter 6 as a reference bar under them.

libs.svg: the project beside OpenCV and CuPy on the same image, one
contract (RGB in, label map out), with the copies shown as their own bar.

Same drawing code and colours as the repo's chart (the helpers of
ch06/benchmarks/figures.py), so build_figures.py themes them the same way.

Data, all committed:

    results/overview/benchmark_results/overview_*.json (the newest)
        row png_blobs: pure Python, @njit, ch01-ch05. The ch01-ch04
        cells are estimates (a sample of launches, scaled).
    results/ch08_gpu_nblob_tiles/benchmark_results/dense_20261016T054922Z.json
        scene input_blobs: ch08 (default engine) and ch06, one session,
        RGB in and recoloured in place, CUDA events.
    results/ch09_cpu_nthreads/benchmark_results/libs_20261125T140827Z.json
        scene input_blobs: OpenCV, CuPy, ch08, ch09 and the one-core BFS.

Outputs: project-page/tools/chain_bfs.svg and libs.svg (committed;
build_figures.py inlines them). Run from the repository root:

    .venv/bin/python project-page/tools/make_chain_bfs.py
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks import figures as F

PAGE = Path(__file__).resolve().parents[1]
REPO = PAGE.parent
RESULTS = REPO / "src" / "flood_fill_cuda" / "results"
OUT = Path(__file__).resolve().parent / "chain_bfs.svg"
DENSE = RESULTS / "ch08_gpu_nblob_tiles" / "benchmark_results" / "dense_20261016T054922Z.json"
LIBS = RESULTS / "ch09_cpu_nthreads" / "benchmark_results" / "libs_20261125T140827Z.json"


def draw(bars, title: str, alt: str, out: Path, lab_w: int = 150) -> None:
    """bars: (label, tag, ms, colour, is_estimate), top to bottom, log axis."""
    lo = 1.0
    hi = 10 ** math.ceil(math.log10(max(b[2] for b in bars)))
    W, row_h, pad_r = 720, 30, 96
    H = len(bars) * row_h + 72
    span = W - lab_w - pad_r

    def x_of(v: float) -> float:
        return lab_w + (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * span

    body = [F._text(0, 16, title, 13, F.INK_STRONG, weight="600")]
    t = lo
    while t <= hi:
        x = x_of(t)
        body.append(f'<line x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{len(bars) * row_h + 32}" '
                    f'stroke="{F.RULE}" stroke-width="1" stroke-opacity="0.22"/>')
        body.append(F._text(x, len(bars) * row_h + 48, f"{t:g}" if t < 1000 else f"{t / 1000:g}s", 10,
                            F.INK, anchor="middle", mono=True))
        t *= 10
    body.append(F._text(lab_w + span / 2, len(bars) * row_h + 64, "milliseconds (log)", 10.5, F.INK,
                        anchor="middle"))

    for i, (label, tag, ms, colour, est) in enumerate(bars):
        y = 32 + i * row_h
        body.append(F._text(lab_w - 10, y + 15, label, 11.5, F.INK, anchor="end"))
        bw = max(2.0, x_of(ms) - lab_w)
        body.append(f'<rect x="{lab_w}" y="{y + 3}" width="{bw:.1f}" height="16" rx="2.5" fill="{colour}"/>')
        txt = f"{ms:,.0f} ms" if ms >= 100 else f"{ms:.2f} ms"
        if est:
            txt = f"~{txt} (est.)"
        body.append(F._text(lab_w + bw + 8, y + 16, txt, 10.5, F.INK_STRONG, mono=True))
        body.append(F._text(W - 4, y + 16, tag, 10, F.INK, anchor="end"))

    out.write_text(F._svg(W, H, "\n".join(body), alt), encoding="utf-8")
    print(f"{out.relative_to(REPO)}: {len(bars)} bars")
    for label, tag, ms, _, est in bars:
        print(f"  {label:28s} {ms:12,.3f} ms{'  (estimate)' if est else ''}  [{tag}]")


def chain() -> None:
    overview_path = sorted(glob.glob(str(RESULTS / "overview" / "benchmark_results" / "overview_*.json")))[-1]
    overview = json.loads(Path(overview_path).read_text())
    row = next(r for r in overview["rows"] if r["row"] == "png_blobs")
    dense = json.loads(DENSE.read_text())
    scene = next(s for s in dense["scenes"] if s["scene"] == "input_blobs")

    bars = []
    for label, key, tag in F.CHAIN:
        cell = row["cells"].get(key)
        ms = cell.get("ms") if isinstance(cell, dict) else cell
        if ms:
            est = isinstance(cell, dict) and bool(cell.get("est"))
            bars.append((label, tag, float(ms), F.C_CH05 if "ch" in key else F.RULE, est))
    bars.append(("ch08 · N blobs, tiles", "tile waves", float(scene["ch08_L8_ms"]), F.C_MASK, False))
    bars.append(("ch06 · runs (reference)", "not a BFS", float(scene["ch06_ms"]), F.C_RGB, False))
    draw(bars, "Every stage on one image - input_blobs.png, 81 Mpx, 2,522 blobs",
         "Every BFS stage on input_blobs.png", OUT)


def libs() -> None:
    session = json.loads(LIBS.read_text())
    row = next(r for r in session["rows"] if r["scene"] == "input_blobs")
    ms = {k: float(v["ms"]) for k, v in row["cells"].items()}
    assert row["e2e"]["cell"] == "ch08_rgb_labelmap"
    bars = [
        ("one-core BFS (reference)", "CPU", ms["oracle_labelmap"], F.RULE, False),
        ("OpenCV · 1 thread", "CPU", ms["opencv_rgb_labelmap_t1"], F.RULE, False),
        ("OpenCV · 16 threads", "CPU", ms["opencv_rgb_labelmap_t16"], F.RULE, False),
        ("ch08 · with the copies", "GPU, host to host", float(row["e2e"]["ms"]), F.C_RGB, False),
        ("ch09 · 16 threads", "CPU", ms["ch09_rgb_labelmap_t16"], F.C_LABEL, False),
        ("CuPy", "GPU", ms["cupy_rgb_labelmap"], F.RULE, False),
        ("ch08 · tile waves", "GPU", ms["ch08_rgb_labelmap"], F.C_MASK, False),
    ]
    draw(bars, "RGB in, label map out - input_blobs.png, 81 Mpx, one session",
         "The project beside OpenCV and CuPy on input_blobs.png", OUT.with_name("libs.svg"), lab_w=168)


if __name__ == "__main__":
    chain()
    libs()
