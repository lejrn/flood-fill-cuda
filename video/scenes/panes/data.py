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
- The Triton stage reads the twins' grand table
  (`results/triton_twins/overview/compare_*.json`): Numba and Triton
  re-timed on the same 17 shapes in one session. A twin cell is
  `numba_ms / triton_ms` (above 1: Triton faster), each side at its own
  fastest like-for-like variant of the chapter. Its headline (overall
  ratio, rows faster, ch05) comes from `results/triton_twins/summary.json`
  and the per-unit compare JSONs it names.

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
TWINS_DIR = RESULTS / "triton_twins"


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

# matrix column -> the grand table's columns for that chapter, like-for-like
# only. ch02_pinned (2 x 768 threads) has no twin: Triton runs power-of-2
# blocks, so its like-for-like column is ch02_pinned_matched (2 x 512 both).
TWIN_VARIANTS = {
    "ch01": ("ch01_ring", "ch01_spill"),
    "ch02": ("ch02_split", "ch02_global", "ch02_dirsplit", "ch02_pinned_matched"),
    "ch03_conn4": ("ch03_conn4",),
    "ch03_conn8": ("ch03_conn8",),
    "ch04": ("ch04_multi",),
    "ch05": ("ch05_merge", "ch05_ccl", "ch05_fused_L8", "ch05_r128_L8",
             "ch05_split_L8", "ch05_split_I1"),
    "ch06": ("ch06_mask",),
}

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


@dataclass(frozen=True)
class TwinCell:
    """One matrix cell of the Triton stage: Numba against Triton, same session."""
    col: str
    numba_ms: float | None
    triton_ms: float | None
    est: bool
    numba_variant: str | None
    triton_variant: str | None

    @property
    def ratio(self) -> float | None:
        """numba_ms / triton_ms: above 1, Triton is faster."""
        if self.numba_ms is None or self.triton_ms is None:
            return None
        return self.numba_ms / self.triton_ms

    @property
    def kind(self) -> str:
        """fast (Triton faster) | slow | est | na. The CPU has no twin: na."""
        if self.ratio is None:
            return "est" if self.est else "na"
        return "fast" if self.ratio > 1 else "slow"

    @property
    def glows(self) -> bool:
        return self.kind == "fast"


@dataclass(frozen=True)
class Twins:
    cells: dict                    # (row key, col key) -> TwinCell
    source: str                    # the grand table's compare JSON
    summary_source: str
    overall: float                 # geometric mean of the unit means, kernel time
    units: int
    unit_kernel: dict              # unit -> geometric mean, kernel time
    rows_timed: int
    rows_like: int                 # like-for-like: the rows every mean uses
    rows_faster: int               # like-for-like rows where Triton is faster
    all_equal: bool                # outputs identical on every timed run

    def cell(self, row_key: str, col_key: str) -> TwinCell:
        return self.cells[(row_key, col_key)]

    @property
    def ch05(self) -> float:
        return self.unit_kernel["ch05_gpu_nblob_nblock"]


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


def _like(r: dict) -> bool:
    """A like-for-like row, as triton_twins/compare/summary.py counts them."""
    cfg = r.get("config", {})
    return ("error" not in r and not r.get("first_translation")
            and not (r.get("duplicate_of") or cfg.get("duplicate_of"))
            and r.get("comparable", True))


def _twin_cell(by: dict, row: str, col: str) -> TwinCell:
    """Each backend's fastest measured like-for-like variant of the chapter;
    estimated (dashed) when only per-blob estimates ran; else n/a."""
    if col == "cpu":
        return TwinCell(col, None, None, False, None, None)
    cands = [by[(row, v)] for v in TWIN_VARIANTS[col] if (row, v) in by]
    cands = [r for r in cands if _like(r)]
    measured = [r for r in cands if not r.get("est")]
    if not measured:
        return TwinCell(col, None, None, bool(cands), None, None)
    n = min(measured, key=lambda r: r["numba"]["kernel_ms"]["median"])
    t = min(measured, key=lambda r: r["triton"]["kernel_ms"]["median"])
    return TwinCell(col, float(n["numba"]["kernel_ms"]["median"]),
                    float(t["triton"]["kernel_ms"]["median"]), False,
                    n["config"]["column"], t["config"]["column"])


@lru_cache(maxsize=1)
def load_twins() -> Twins:
    grand_path = _newest(TWINS_DIR / "overview", "compare_*.json")
    summary_path = TWINS_DIR / "summary.json"
    if grand_path is None or not summary_path.exists():
        raise FileNotFoundError(f"no grand table or summary.json under {TWINS_DIR}")
    grand, summary = _load(grand_path), _load(summary_path)
    rel = grand_path.relative_to(RESULTS).as_posix()
    if summary["units"]["overview"]["source"] != rel:
        raise ValueError(f"summary.json was built from {summary['units']['overview']['source']}, "
                         f"not {rel}: re-run triton_twins.compare.summary")
    by = {}
    for r in grand["rows"]:
        key = (r["row"], r["config"]["column"])
        if key in by:
            raise ValueError(f"two grand-table rows for {key}")
        by[key] = r
    bench = load_bench()
    cells = {(row.key, c.key): _twin_cell(by, row.key, c.key)
             for row in bench.rows for c in COLUMNS}
    timed = like = faster = 0
    for u in summary["units"].values():
        rows = _load(RESULTS / u["source"])["rows"]
        ok = [r for r in rows if _like(r)]
        timed += len(rows)
        like += len(ok)
        faster += sum(r["speedup_kernel"] > 1 for r in ok)
    return Twins(
        cells=cells, source=grand_path.name, summary_source=summary_path.name,
        overall=float(summary["overall"]["geomean_of_unit_geomeans_kernel"]),
        units=int(summary["overall"]["units"]),
        unit_kernel={k: float(u["geomean_speedup_kernel"]) for k, u in summary["units"].items()},
        rows_timed=timed, rows_like=like, rows_faster=faster,
        all_equal=bool(summary["overall"]["all_outputs_equal"]))


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


def fmt_twin(ratio: float) -> str:
    """A twin cell, 4 characters: 1.16, 0.73, 2.41."""
    return f"{ratio:.2f}" if ratio < 9.995 else f"{ratio:.1f}"


def twin_glow_opacity(ratio: float) -> float:
    """The gaps are small, so the scale is too: x1 -> 0.25, x2.5 and above -> 1.0."""
    return 0.25 + 0.75 * min(1.0, max(0.0, math.log(ratio) / math.log(2.5)))


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

    tw = load_twins()
    print(f"\nTriton stage: {tw.source} + {tw.summary_source}   cell = numba_ms / triton_ms, "
          f"each side's fastest like-for-like variant")
    print(f"{'row':12s}" + "".join(f" {c.key:>11s}" for c in COLUMNS[1:]))
    for r in b.rows:
        line = f"{r.key:12s}"
        for c in COLUMNS[1:]:
            t, cell = tw.cell(r.key, c.key), r.cells[c.key]
            if t.kind in ("fast", "slow"):
                same = "" if t.numba_variant == t.triton_variant else "/"
                txt = f"{'*' if t.glows else ' '}{fmt_twin(t.ratio)}{same}"
            else:
                txt = "~est" if t.kind == "est" else "n/a"
            # the stage flips the ms matrix in place: the same cells must carry a number
            ms_kind = {"fast": "num", "slow": "num"}.get(cell.kind, cell.kind)
            tw_kind = {"fast": "num", "slow": "num"}.get(t.kind, t.kind)
            if ms_kind != tw_kind:
                txt += "!!"
                bad.append(r.key)
            line += f" {txt:>11s}"
        print(line)
    print("  * = Triton faster (glows)   / = Numba's and Triton's fastest variants differ"
          "   !! = not the ms matrix's cell kind")
    shown = [tw.cell(r.key, c.key) for r in b.rows for c in COLUMNS[1:]]
    shown = [t for t in shown if t.ratio is not None]
    print(f"  on screen: {sum(t.glows for t in shown)} of {len(shown)} cells faster in Triton")
    print(f"  headline: {fmt_twin(tw.overall)}× overall ({tw.units} units), Triton faster in "
          f"{tw.rows_faster:,} of {tw.rows_like:,} like-for-like rows ({tw.rows_timed:,} timed), "
          f"ch05 {fmt_twin(tw.ch05)}×, outputs identical: {tw.all_equal}")
    if not tw.all_equal:
        bad.append("twins outputs")
    if bad:
        print(f"\nMISMATCH rows: {sorted(set(bad))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
