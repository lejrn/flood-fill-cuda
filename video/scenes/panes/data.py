"""Benchmark data for the video, read from the committed JSON files.

Rules:
- Every millisecond on screen is copied from a committed benchmark JSON;
  the only derived number is `speedup = njit_ms / ms`.
- The matrix rows are the 17 rows of the overview benchmark
  (`results/overview/benchmark_results/overview_*.json`); its ch06 column
  comes from `ch06_overview_*.json` in the same folder (a separate
  session, produced by `flood_fill_cuda.overview.bench_ch06`).
- A stage column is one representative per chapter: the fastest MEASURED
  variant of that chapter on that row. If only estimated cells exist
  (`est: true`, one launch per blob on N-blob rows) the cell is estimated
  and never glows. If nothing ran, the cell carries the skip reason.
- The headline numbers (real image) come from the overview `png_blobs`
  row (pure Python, @njit) and from ch06's own session
  `runs_20260725T161448Z.json` (ch05 58.51 ms, ch06 1.46 / 2.96 ms), the
  session the READMEs quote.

No manim import here, so `uv run python scenes/panes/data.py` prints the
matrix as text for auditing before anything is rendered.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parents[2]
REPO = VIDEO_DIR.parent
RESULTS = REPO / "src" / "flood_fill_cuda" / "results"
OVERVIEW_DIR = RESULTS / "overview" / "benchmark_results"
CH06_DIR = RESULTS / "ch06_gpu_nblob_runs" / "benchmark_results"


# ------------------------------------------------------------------ specs
@dataclass(frozen=True)
class ColSpec:
    key: str
    header: tuple[str, str]        # two short header lines
    variants: tuple[str, ...]      # JSON cell keys this column may draw from
    tag: str                       # the strip tag / narration name
    final: bool = False            # teal instead of blue


COLUMNS: tuple[ColSpec, ...] = (
    ColSpec("cpu", ("CPU", "@njit"), ("njit",), "CPU"),
    ColSpec("ch01", ("1", "block"), ("ch01_ring", "ch01_spill"), "1 block"),
    ColSpec("ch02", ("2", "blocks"),
            ("ch02_split", "ch02_global", "ch02_dirsplit", "ch02_pinned"), "2 blocks"),
    ColSpec("ch03_conn4", ("N", "blocks"), ("ch03_conn4",), "N blocks"),
    ColSpec("ch03_conn8", ("8-", "conn"), ("ch03_conn8",), "8-conn"),
    ColSpec("ch04", ("2", "blobs"), ("ch04_multi",), "2 blobs"),
    ColSpec("ch05", ("N", "blobs"),
            ("ch05_merge", "ch05_ccl", "ch05_fused_L8", "ch05_r128_L8",
             "ch05_split_L8", "ch05_split_I1"), "N blobs"),
    ColSpec("ch06", ("runs", ""), ("ch06_mask",), "runs", final=True),
)
COL_INDEX = {c.key: i for i, c in enumerate(COLUMNS)}

# row key -> (short display name, glyph kind)
ROW_LABELS = {
    "sq_256": ("sq 256", "square"),
    "sq_1024": ("sq 1024", "square"),
    "sq_4000": ("sq 4000", "square"),
    "disk_256": ("disk 256", "disk"),
    "disk_1024": ("disk 1024", "disk"),
    "disk_4000": ("disk 4000", "disk"),
    "serp_128": ("snake 128", "snake"),
    "serp_256": ("snake 256", "snake"),
    "comb_24": ("comb 24", "comb"),
    "comb_2000": ("comb 2000", "comb"),
    "two_sq_300": ("2 sq 300", "two_sq"),
    "two_sq_2800": ("2 sq 2800", "two_sq"),
    "asym_4000_800": ("asym 4000", "asym"),
    "random_1000": ("noise 1000", "noise"),
    "random_4000": ("noise 4000", "noise"),
    "png_blobs": ("png blobs", "picture"),
    "png_blocks": ("png blocks", "picture"),
}

SKIP_LABELS = {"na": "n/a", "overflow": "overflow", "capped": "capped",
               "unsupported": "n/s", "oom": "oom", "missing": "…"}


# ------------------------------------------------------------------ model
@dataclass(frozen=True)
class Cell:
    col: str
    ms: float | None
    est: bool
    skip: str | None
    variant: str | None
    speedup: float | None          # njit_ms / ms, None when no ms
    alt_ms: float | None = None    # ch06: the rgb contract, for captions

    @property
    def kind(self) -> str:
        """cpu | fast | slow | est | na"""
        if self.col == "cpu":
            return "cpu"
        if self.skip is not None or self.ms is None:
            return "na"
        if self.est:
            return "est"
        return "fast" if self.speedup is not None and self.speedup > 1 else "slow"

    @property
    def glows(self) -> bool:
        return self.kind == "fast"


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    glyph: str
    family: str
    note: str
    width: int
    height: int
    kind: str                      # one | two | n
    n_blobs: int | None
    njit_ms: float
    pure_ms: float | None
    cells: dict                    # col key -> Cell
    crosscheck: str
    ch06_crosscheck: str | None


@dataclass(frozen=True)
class Bench:
    rows: tuple
    device: str
    sm_count: int
    source_overview: str
    source_ch06: str | None

    def cell(self, row_key: str, col_key: str) -> Cell:
        return next(r for r in self.rows if r.key == row_key).cells[col_key]

    def column(self, col_key: str) -> list:
        return [r.cells[col_key] for r in self.rows]


@dataclass(frozen=True)
class Headline:
    pure_ms: float
    njit_ms: float
    ch05_ms: float
    ch06_mask_ms: float
    ch06_rgb_ms: float
    n_blobs: int
    red_px: int
    n_runs: int
    n_pixels: int
    sources: dict

    @property
    def ratio(self) -> float:
        return self.pure_ms / self.ch06_mask_ms


# ------------------------------------------------------------------ loading
def _newest(folder: Path, pattern: str, exclude: str | None = None) -> Path | None:
    paths = sorted(glob.glob(str(folder / pattern)))
    if exclude:
        paths = [p for p in paths if exclude not in os.path.basename(p)]
    return Path(paths[-1]) if paths else None


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(cells: dict, variants: tuple[str, ...], col: str, njit_ms: float,
          alt: dict | None = None) -> Cell:
    """The chapter's representative on this row: fastest measured variant,
    else fastest estimated variant, else the first skip reason."""
    cands = [(v, cells[v]) for v in variants if v in cells]
    measured = [(v, c) for v, c in cands
                if c.get("skip") is None and not c.get("est") and c.get("ms") is not None]
    est = [(v, c) for v, c in cands
           if c.get("skip") is None and c.get("est") and c.get("ms") is not None]
    if measured:
        v, c = min(measured, key=lambda vc: vc[1]["ms"])
        return Cell(col, float(c["ms"]), False, None, v, njit_ms / float(c["ms"]),
                    alt_ms=(float(alt["ms"]) if alt and alt.get("skip") is None else None))
    if est:
        v, c = min(est, key=lambda vc: vc[1]["ms"])
        return Cell(col, float(c["ms"]), True, None, v, njit_ms / float(c["ms"]))
    skip = cands[0][1].get("skip", "na") if cands else "missing"
    return Cell(col, None, False, skip or "na", None, None)


@lru_cache(maxsize=1)
def load_bench() -> Bench:
    ov_path = _newest(OVERVIEW_DIR, "overview_*.json", exclude="ch06")
    if ov_path is None:
        raise FileNotFoundError(f"no overview_*.json under {OVERVIEW_DIR}")
    ov = _load(ov_path)
    ch06_path = _newest(OVERVIEW_DIR, "ch06_overview_*.json")
    ch06_rows = {}
    if ch06_path is not None:
        ch06_rows = {r["row"]: r for r in _load(ch06_path)["rows"]}

    rows = []
    for r in ov["rows"]:
        cells_json = dict(r["cells"])
        c6 = ch06_rows.get(r["row"])
        if c6 is not None:
            cells_json.update(c6["cells"])
        njit_ms = float(cells_json["njit"]["ms"])
        pure = cells_json.get("pure_python", {})
        pure_ms = float(pure["ms"]) if pure.get("skip") is None and "ms" in pure else None
        cells = {}
        for col in COLUMNS:
            if col.key == "cpu":
                cells[col.key] = Cell("cpu", njit_ms, False, None, "njit", 1.0)
            elif col.key == "ch06":
                cells[col.key] = _pick(cells_json, col.variants, col.key, njit_ms,
                                       alt=cells_json.get("ch06_rgb"))
            else:
                cells[col.key] = _pick(cells_json, col.variants, col.key, njit_ms)
        name, glyph = ROW_LABELS.get(r["row"], (r["row"], "square"))
        rows.append(Row(
            key=r["row"], name=name, glyph=glyph, family=r["family"], note=r["note"],
            width=int(r["width"]), height=int(r["height"]), kind=r["kind"],
            n_blobs=r.get("n_blobs"), njit_ms=njit_ms, pure_ms=pure_ms,
            cells=cells, crosscheck=r.get("crosscheck", "?"),
            ch06_crosscheck=(c6.get("crosscheck") if c6 else None)))
    return Bench(rows=tuple(rows), device=str(ov["device"]).rstrip("\x00").strip(),
                 sm_count=int(ov["sm_count"]), source_overview=ov_path.name,
                 source_ch06=(ch06_path.name if ch06_path else None))


@lru_cache(maxsize=1)
def headline() -> Headline:
    b = load_bench()
    png = next(r for r in b.rows if r.key == "png_blobs")
    runs_path = _newest(CH06_DIR, "runs_*.json")
    if runs_path is None:
        raise FileNotFoundError(f"no runs_*.json under {CH06_DIR}")
    runs = _load(runs_path)
    scene = next(s for s in runs["scenes"] if s["scene"] == "input_blobs")
    return Headline(
        pure_ms=float(png.pure_ms), njit_ms=float(png.njit_ms),
        ch05_ms=float(scene["ch05"]["median_ms"]),
        ch06_mask_ms=float(scene["ch06"]["mask"]["median_ms"]),
        ch06_rgb_ms=float(scene["ch06"]["rgb"]["median_ms"]),
        n_blobs=int(scene["n_blobs"]), red_px=int(scene["red_px"]),
        n_runs=int(scene["n_runs"]), n_pixels=int(scene["n_pixels"]),
        sources={"pure_ms": f"{b.source_overview}:png_blobs.pure_python",
                 "njit_ms": f"{b.source_overview}:png_blobs.njit",
                 "ch05_ms": f"{runs_path.name}:input_blobs.ch05",
                 "ch06_mask_ms": f"{runs_path.name}:input_blobs.ch06.mask",
                 "ch06_rgb_ms": f"{runs_path.name}:input_blobs.ch06.rgb"})


# ------------------------------------------------------------------ formatting
def fmt_cell_ms(ms: float) -> str:
    """At most 4 characters, for a matrix cell: 24s, 1.3s, 269, 22.9, 0.15."""
    if ms >= 10_000:
        return f"{ms / 1000:.0f}s"
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 10:
        return f"{ms:.1f}"
    return f"{ms:.2f}"


def fmt_ms(ms: float) -> str:
    """24083 -> '24,083 ms', 1346 -> '1,346 ms', 58.51 -> '58.51 ms', 1.46 -> '1.46 ms'."""
    if ms >= 100:
        return f"{ms:,.0f} ms"
    return f"{ms:.2f} ms"


def fmt_speedup(x: float, est: bool = False) -> str:
    if x >= 100:
        s = f"{x:,.0f}×"
    elif x >= 10:
        s = f"{x:.0f}×"
    elif x >= 1:
        s = f"{x:.1f}×"
    else:
        s = f"{x:.2f}×"
    return ("~" + s) if est else s


def fmt_ratio_round(x: float) -> str:
    """16551 -> '16,000×' (two significant figures, floor)."""
    mag = 10 ** (int(math.log10(x)) - 1)
    return f"{int(x // mag) * mag:,}×"


def glow_opacity(speedup: float) -> float:
    """1x -> 0.25, 10x -> 0.62, 100x and above -> 1.0."""
    return 0.25 + 0.75 * min(1.0, max(0.0, math.log10(speedup)) / 2.0)


# ------------------------------------------------------------------ CLI audit
def main() -> int:
    b = load_bench()
    h = headline()
    bad = []
    print(f"overview: {b.source_overview}   ch06: {b.source_ch06 or 'MISSING'}")
    print(f"device: {b.device} ({b.sm_count} SMs)\n")
    head = f"{'row':12s} {'kind':4s} {'size':11s} {'blobs':>7s} | {'pure':>9s} {'njit':>8s} |"
    head += "".join(f" {c.key:>13s}" for c in COLUMNS[1:])
    print(head)
    for r in b.rows:
        line = (f"{r.key:12s} {r.kind:4s} {r.width:5d}x{r.height:<5d} "
                f"{(r.n_blobs or ''):>7} | "
                f"{(fmt_ms(r.pure_ms) if r.pure_ms else 'capped'):>9s} "
                f"{fmt_ms(r.njit_ms):>8s} |")
        for c in COLUMNS[1:]:
            cell = r.cells[c.key]
            if cell.kind == "na":
                txt = SKIP_LABELS.get(cell.skip, cell.skip)
            else:
                mark = "*" if cell.glows else ("~" if cell.est else " ")
                txt = f"{mark}{fmt_cell_ms(cell.ms):>4s} {fmt_speedup(cell.speedup, cell.est):>6s}"
            line += f" {txt:>13s}"
        flags = []
        if r.crosscheck != "OK":
            flags.append(f"overview {r.crosscheck}")
            bad.append(r.key)
        if r.ch06_crosscheck not in (None, "OK"):
            flags.append(f"ch06 {r.ch06_crosscheck}")
            bad.append(r.key)
        for c in COLUMNS[1:]:
            cell = r.cells[c.key]
            if cell.kind in ("fast", "slow") and 0.8 <= cell.speedup <= 1.25:
                flags.append(f"{c.key} borderline {cell.speedup:.2f}x")
        print(line + ("   <- " + "; ".join(flags) if flags else ""))
    print("\n  * = glows (faster than @njit)   ~ = estimated, one launch per blob (never glows)")
    print("  cell = chosen variant's ms and speedup vs @njit; column = fastest measured variant of the chapter\n")

    for c in COLUMNS[1:]:
        n_glow = sum(1 for r in b.rows if r.cells[c.key].glows)
        chosen = sorted({r.cells[c.key].variant for r in b.rows if r.cells[c.key].variant})
        print(f"  {c.key:11s} glows on {n_glow:2d}/17 rows   variants used: {', '.join(chosen)}")

    png = next(r for r in b.rows if r.key == "png_blobs")
    m = png.cells["ch06"]
    print(f"\nheadline (real image, {h.n_blobs:,} blobs, {h.red_px:,} red px, {h.n_runs:,} runs):")
    for k, v in h.sources.items():
        print(f"  {k:13s} {fmt_ms(getattr(h, k)):>12s}   <- {v}")
    print(f"  ratio         {h.ratio:,.0f}x -> shown as {fmt_ratio_round(h.ratio)}")
    if m.ms is not None:
        diff = abs(m.ms - h.ch06_mask_ms) / h.ch06_mask_ms
        print(f"  matrix png_blobs ch06 mask = {m.ms:.3f} ms ({b.source_ch06}) vs headline "
              f"{h.ch06_mask_ms:.3f} ms: {diff * 100:.0f}% apart"
              + ("   <- more than 10%" if diff > 0.10 else ""))
    if bad:
        print(f"\nMISMATCH rows: {sorted(set(bad))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
