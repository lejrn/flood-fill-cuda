"""
Generate the combined benchmark dashboard: single blob (1 block -> 2
blocks -> N blocks -> 8 directions) and dual blob (2 blobs, N blocks).

Organized by DOMAIN, not by benchmark session:

  1. Single blob
     1.1 Single block        (single_block_shared: v1 ring, v2 spill)
     1.2 Dual blocks          (dual_block: split, global, dirsplit,
                               placement/pinning, balance, tpb sweep)
     1.3 Dual blocks vs N blocks (multi_block: 4-conn runtime/speedup/
                               sweep/bandwidth)
     1.4 4 vs 8 connectivity  (multi_block: the 8-direction twin kernels)
     1.5 More work per barrier (multi_block: radius-2 + warp-coop twins,
                               the neighbors_* benchmark)
  2. Dual blob                (dual_blob: sequential vs streams vs
                               multisource, lin vs xy entry format,
                               4 vs 8 connectivity on both mechanisms,
                               radius-2 on both mechanisms)

Every subsection reports its OWN speedup multiplier from its own
benchmark session; 1.3 also shows those multipliers chained together
into one total (single block -> N blocks) via `chain_strip()` — see that
function's docstring for why the chain is computed from ONE file
(multi_block's own JSON, which re-measures every predecessor kernel
fresh in the same session) rather than cross-multiplying separate
sessions' numbers.

Usage:
    uv run python -m flood_fill_cuda.chapters.ch02_gpu_1blob_2block.benchmarks.visualize [dual.json]

Output: results/ch02_gpu_1blob_2block/benchmark_results/dual_block_benchmark.html
(overwritten per run — the timestamped JSON/CSV remain the durable record).
"""
import json
import math
import os
import sys

from ....shared import results_paths
from ....shared import viz
from ...ch03_gpu_1blob_nblock.benchmarks import visualize as mb_viz

RESULTS_DIR = results_paths.results_dir("ch02_gpu_1blob_2block", "benchmark_results")
SBS_RESULTS_DIR = results_paths.results_dir("ch01_gpu_1blob_1block", "benchmark_results")
DB_RESULTS_DIR = results_paths.results_dir("ch04_gpu_2blob_nblock", "benchmark_results")


_newest = results_paths.newest
_newest_optional = results_paths.newest_optional


DUAL_PATH = sys.argv[1] if len(sys.argv) > 1 else _newest("dual_block_*.json",
                                                          RESULTS_DIR)
SBS_PATH = _newest("single_block_shared_*.json", SBS_RESULTS_DIR)
DB_PATH = _newest_optional("dual_blob_*.json", DB_RESULTS_DIR,
                           exclude="radius2")
# The per-barrier work experiments (radius-2 + warp-coop) write their own
# JSONs; both are optional (graceful degradation, same contract as DB).
DBR2_PATH = _newest_optional("dual_blob_radius2_*.json", DB_RESULTS_DIR)
OUT_PATH = os.path.join(RESULTS_DIR, "dual_block_benchmark.html")

with open(DUAL_PATH) as f:
    DUAL = json.load(f)
with open(SBS_PATH) as f:
    SBS = json.load(f)
DB = None
if DB_PATH:
    with open(DB_PATH) as f:
        DB = json.load(f)
DBR2 = None
if DBR2_PATH:
    with open(DBR2_PATH) as f:
        DBR2 = json.load(f)

ROWS = DUAL["scenes"]
SWEEP = DUAL["tpb_sweep"]
PLACEMENT = DUAL["placement"]
SBS_ROWS = SBS["scenes"]
SBS_SWEEP = SBS["tpb_sweep"]
# The sweep grid was extended to run at both connectivities on two scenes;
# every consumer must pick one explicitly or it draws a tangled mix.

DB_ROWS = DB["scenes"] if DB else []
DB_PEAK = DB["measured_peak_gb_s"] if DB else 0.0
HAS_DUALBLOB = bool(DB_ROWS)

DBR2_ROWS = DBR2["scenes"] if DBR2 else []
HAS_DB_R2 = HAS_DUALBLOB and bool(DBR2_ROWS)

KERNELS = ["split", "global", "dirsplit"]
# Entity -> color slot, constant across every dual chart: s1 = single-block
# v2, s2 = @njit CPU, s3 = split, s4 = global, s5 = dirsplit.
KCLS = {"split": "s3", "global": "s4", "dirsplit": "s5"}

SCENE_LABELS = viz.SCENE_LABELS
SBS_LABELS = {
    "sq_256_center": "square 256² · center",
    "sq_512_center": "square 512² · center",
    "sq_1024_center": "square 1024² · center",
    "sq_2000_center": "square 2000² · center",
    "sq_4000_corner": "square 4000² · corner",
    "serpentine_256": "serpentine 256²",
    "disk_1024": "disk r=480",
    "sq_2600_full_center": "square 2600² full · center",
    "sq_4000_center": "square 4000² · center",
    "sq_5000_center": "square 5000² · center",
    "sq_6000_center": "square 6000² · center",
}


label = viz.label


fmt_ms = viz.fmt_ms


fmt_int = viz.fmt_int


# ------------------------------------------------------------ shared geometry
W, ROW_H, GUT_L, GUT_R = viz.W, viz.ROW_H, viz.GUT_L, viz.GUT_R


decimate = viz.decimate


log_dot_plot = viz.log_dot_plot


# ------------------------------------------------------------------ sections
def runtime_chart():
    series = [(lambda r: r["njit_ms"], "@njit CPU", "s2"),
              (lambda r: r["v2_kernel_ms"], "single-block v2", "s1"),
              (lambda r: r["split_kernel_ms"], "split", "s3"),
              (lambda r: r["global_kernel_ms"], "global", "s4"),
              (lambda r: r["dirsplit_kernel_ms"], "dirsplit", "s5")]
    return log_dot_plot(ROWS, series, "Runtime per scene, log scale",
                        lambda r: label(r["scene"]))


def speedup_chart():
    lo, hi = 0.0, max(r[f"{k}_speedup_vs_v2"] for r in ROWS
                      for k in KERNELS) * 1.12
    n = len(ROWS)
    h = n * ROW_H + 34
    span = W - GUT_L - GUT_R

    def x_of(v):
        return GUT_L + (v - lo) / (hi - lo) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" '
             f'aria-label="Speedup vs the single-block v2 kernel">']
    tick = 0.0
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                     f'class="tick" text-anchor="middle">{tick:g}×</text>')
        tick += 0.5
    px = x_of(1.0)
    parts.append(f'<line x1="{px:.1f}" y1="4" x2="{px:.1f}" '
                 f'y2="{n * ROW_H}" class="satline"/>')
    parts.append(f'<text x="{px + 6:.1f}" y="14" class="anno" '
                 f'text-anchor="start">v2 parity — right of this line, the '
                 f'2nd block paid for itself</text>')
    for i, r in enumerate(ROWS):
        cy = i * ROW_H + ROW_H / 2
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - GUT_R}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label(r["scene"])}</text>')
        for si, k in enumerate(KERNELS):
            v = r[f"{k}_speedup_vs_v2"]
            dy = (si - 1) * 6.5
            tip = (f"{label(r['scene'])} — {k}: {v:.2f}× vs single-block v2 "
                   f"({fmt_ms(r[f'{k}_kernel_ms'])} vs "
                   f"{fmt_ms(r['v2_kernel_ms'])} ms)")
            parts.append(f'<circle cx="{x_of(v):.1f}" cy="{cy + dy:.1f}" '
                         f'r="5" class="dot {KCLS[k]}" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def placement_chart():
    """Two per-scene panels (own ms scale each), 4 bars per panel."""
    scenes = []
    for row in PLACEMENT:
        if row["scene"] not in scenes:
            scenes.append(row["scene"])
    html = []
    for sname in scenes:
        rows = [r for r in PLACEMENT if r["scene"] == sname]
        w, h, pl, pr = 420, 176, 132, 118
        bar_h, row_h = 20, 34
        top = 30
        vmax = max(r["kernel_ms"] for r in rows) * 1.05
        span = w - pl - pr
        svg = [f'<svg viewBox="0 0 {w} {h}" role="img" '
               f'aria-label="Placement experiment, {label(sname)}">']
        svg.append(f'<text x="{pl}" y="14" class="paneltitle">'
                   f'{label(sname)}</text>')
        best = min(r["kernel_ms"] for r in rows)
        for i, r in enumerate(rows):
            y = top + i * row_h + (row_h - bar_h) / 2
            cy = y + bar_h / 2 + 4
            bw = r["kernel_ms"] / vmax * span
            cfg = (r["config"].replace("v2 1x", "v2 1×")
                   .replace("pinned 2x768 ", "pair "))
            smid = ("SM " + ",".join(str(s) for s in r["sm_ids"])
                    if r["sm_ids"] else "1 SM")
            tip = (f"{label(sname)} — {r['config']}: "
                   f"{fmt_ms(r['kernel_ms'])} ms · {r['mpx_s']:.0f} Mpx/s · "
                   f"observed {smid}")
            svg.append(f'<text x="{pl - 8}" y="{cy:.1f}" class="rowlab" '
                       f'text-anchor="end">{cfg}</text>')
            svg.append(f'<rect x="{pl}" y="{y:.1f}" width="{bw:.1f}" '
                       f'height="{bar_h}" rx="4" class="seg s1" '
                       f'data-tip="{tip}"/>')
            weight = ' font-weight="650"' if r["kernel_ms"] == best else ""
            svg.append(f'<text x="{pl + bw + 6:.1f}" y="{cy:.1f}" '
                       f'class="rowval"{weight}>{fmt_ms(r["kernel_ms"])} ms · '
                       f'{smid}</text>')
        svg.append("</svg>")
        html.append('<div class="panel">' + "\n".join(svg) + "</div>")
    return '<div class="panels">' + "\n".join(html) + "</div>"


# (kernel, scene, title, level_cap): the serpentine panel zooms to its first
# 2,048 levels — on the full 32,896-level axis the alternating staircase
# collapses into one diagonal and the taking-turns story disappears.
BALANCE_PANELS = [
    ("split", "offcenter_2000", "split — block 1 owns nothing", None),
    ("dirsplit", "offcenter_2000", "dirsplit — same blob, both blocks climb",
     None),
    ("dirsplit", "serpentine_256",
     "dirsplit — taking turns (first 2,048 levels)", 2048),
    ("split", "sq_2000_center", "split — seam-symmetric mirror", None),
]


def balance_panels():
    html, js = [], {}
    lookup = {r["scene"]: r for r in ROWS}
    for kernel, sname, title, cap in BALANCE_PANELS:
        r = lookup[sname]
        per_block = r[f"{kernel}_per_block"]
        c0, c1 = [], []
        t0 = t1 = 0
        for a, b in zip(per_block[0], per_block[1]):
            t0 += a
            t1 += b
            c0.append(t0)
            c1.append(t1)
        if cap is not None:
            c0, c1 = c0[:cap], c1[:cap]
            t0, t1 = c0[-1], c1[-1]
        xs, y0 = decimate(c0)
        _, y1 = decimate(c1)  # same bins: monotone, so max == last-of-bin
        y1 = [c1[i] for i in xs]
        n_levels = len(c0)
        peak = max(t0, t1, 1)
        w, h, pl, pr, pt, pb = 420, 176, 62, 64, 26, 26
        px = lambda x: pl + x / max(n_levels - 1, 1) * (w - pl - pr)
        py = lambda v: pt + (1 - v / peak) * (h - pt - pb)
        key = f"{kernel}_{sname}"
        svg = [f'<svg viewBox="0 0 {w} {h}" class="panel-svg" id="b_{key}" '
               f'role="img" aria-label="Cumulative per-block work, {title}">']
        for frac in (0, 0.5, 1):
            yy = pt + frac * (h - pt - pb)
            v = peak * (1 - frac)
            svg.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w - pr}" '
                       f'y2="{yy:.1f}" class="grid"/>')
            svg.append(f'<text x="{pl - 6}" y="{yy + 4:.1f}" class="tick" '
                       f'text-anchor="end">{fmt_int(round(v))}</text>')
        svg.append(f'<text x="{pl}" y="14" class="paneltitle">{title}</text>')
        svg.append(f'<text x="{w - pr}" y="{h - 8}" class="tick" '
                   f'text-anchor="end">BFS level →</text>')
        pts0 = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, y0))
        pts1 = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, y1))
        svg.append(f'<polyline points="{pts1}" class="trace4"/>')
        svg.append(f'<polyline points="{pts0}" class="trace"/>')
        # direct labels at line ends, pushed apart if the finals collide
        ly0, ly1 = py(t0), py(t1)
        if abs(ly0 - ly1) < 14:
            mid = (ly0 + ly1) / 2
            ly0, ly1 = mid - 7, mid + 7
        svg.append(f'<text x="{w - pr + 4}" y="{ly0 + 4:.1f}" class="anno" '
                   f'text-anchor="start">b0</text>')
        svg.append(f'<text x="{w - pr + 4}" y="{ly1 + 4:.1f}" class="anno" '
                   f'text-anchor="start">b1</text>')
        svg.append(f'<line class="xhair" id="bxh_{key}" x1="0" x2="0" '
                   f'y1="{pt}" y2="{h - pb}" visibility="hidden"/>')
        svg.append(f'<circle class="xdot2" id="bxd1_{key}" r="4" visibility="hidden"/>')
        svg.append(f'<circle class="xdot" id="bxd0_{key}" r="4" visibility="hidden"/>')
        svg.append(f'<rect x="{pl}" y="{pt}" width="{w - pl - pr}" '
                   f'height="{h - pt - pb}" fill="transparent" '
                   f'class="bhover-capture" data-panel="{key}"/>')
        svg.append("</svg>")
        html.append('<div class="panel">' + "\n".join(svg) + "</div>")
        js[key] = {"xs": xs, "c0": y0, "c1": y1, "w": w, "h": h, "pl": pl,
                   "pr": pr, "pt": pt, "pb": pb, "peak": peak, "n": n_levels}
    return '<div class="panels">' + "\n".join(html) + "</div>", json.dumps(js)


def sweep_chart():
    """Mpx/s vs tpb: three dual kernels (2 blocks) + the single-block sweep."""
    tpbs = [64, 128, 256, 512, 1024]
    w, h, pl, pr, pt, pb = 860, 260, 60, 150, 22, 40
    ymax = max([r["mpx_s"] for r in SWEEP]
               + [r["mpx_s"] for r in SBS_SWEEP]) * 1.12
    xstep = (w - pl - pr) / (len(tpbs) - 1)

    def x_of(tpb):
        return pl + tpbs.index(tpb) * xstep

    def y_of(v):
        return pt + (1 - v / ymax) * (h - pt - pb)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" '
             f'aria-label="Throughput vs threads per block, all kernels">']
    for frac in (0, 0.5, 1):
        yy = pt + frac * (h - pt - pb)
        parts.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w - pr}" '
                     f'y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pl - 6}" y="{yy + 4:.1f}" class="tick" '
                     f'text-anchor="end">{round(ymax * (1 - frac)):g}</text>')
    parts.append(f'<text x="{pl - 40}" y="{pt - 6}" class="tick">Mpx/s</text>')
    for tpb in tpbs:
        parts.append(f'<text x="{x_of(tpb):.1f}" y="{h - pb + 16}" '
                     f'class="tick" text-anchor="middle">{tpb}</text>')
    parts.append(f'<text x="{(pl + w - pr) / 2:.1f}" y="{h - 4}" class="tick" '
                 f'text-anchor="middle">threads per block · square 2000² '
                 f'scene</text>')

    lines = []
    for k in KERNELS:
        pts = [(r["tpb"], r["mpx_s"]) for r in SWEEP if r["kernel"] == k]
        lines.append((f"{k} (2 blocks)", KCLS[k], pts,
                      lambda r, k=k: f"{k} 2×{r[0]}: {r[1]:.1f} Mpx/s"))
    sbs_pts = [(r["tpb"], r["mpx_s"]) for r in SBS_SWEEP]
    lines.append(("single block (v1 ring)", "s1", sbs_pts,
                  lambda r: f"single block 1×{r[0]}: {r[1]:.1f} Mpx/s"))

    ends = []
    for name, cls, pts, tipfn in lines:
        poly = " ".join(f"{x_of(t):.1f},{y_of(v):.1f}" for t, v in pts)
        parts.append(f'<polyline points="{poly}" class="line {cls}l"/>')
        for t, v in pts:
            parts.append(f'<circle cx="{x_of(t):.1f}" cy="{y_of(v):.1f}" '
                         f'r="4.5" class="dot {cls}" '
                         f'data-tip="{tipfn((t, v))}"/>')
        ends.append([y_of(pts[-1][1]), x_of(pts[-1][0]), name])
    # direct end labels, pushed apart top-down
    ends.sort()
    prev = -99.0
    for ey, ex, name in ends:
        ey = max(ey, prev + 13)
        prev = ey
        parts.append(f'<text x="{ex + 8:.1f}" y="{ey + 4:.1f}" class="anno" '
                     f'text-anchor="start">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def overhead_chart():
    vals = [r[f"{k}_instrumentation_overhead_pct"] for r in ROWS for k in KERNELS]
    lo, hi = min(min(vals), 0) - 0.5, max(vals) * 1.15
    n = len(ROWS)
    h = n * ROW_H + 34
    span = W - GUT_L - GUT_R

    def x_of(v):
        return GUT_L + (v - lo) / (hi - lo) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" '
             f'aria-label="Instrumentation overhead vs bare twins">']
    tick = 0
    while tick <= hi:
        x = x_of(tick)
        cls = "satline" if tick == 0 else "grid"
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="{cls}"/>')
        parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                     f'class="tick" text-anchor="middle">{tick:g}%</text>')
        tick += 3
    for i, r in enumerate(ROWS):
        cy = i * ROW_H + ROW_H / 2
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - GUT_R}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label(r["scene"])}</text>')
        for si, k in enumerate(KERNELS):
            v = r[f"{k}_instrumentation_overhead_pct"]
            dy = (si - 1) * 6.5
            tip = (f"{label(r['scene'])} — {k}: {v:+.1f}% vs its bare twin "
                   f"({fmt_ms(r[f'{k}_kernel_ms'])} vs "
                   f"{fmt_ms(r[f'{k}_bare_kernel_ms'])} ms)")
            parts.append(f'<circle cx="{x_of(v):.1f}" cy="{cy + dy:.1f}" '
                         f'r="5" class="dot {KCLS[k]}" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "\n".join(parts)




def sbs_chart():
    series = [(lambda r: r["njit_ms"], "@njit CPU", "s2"),
              (lambda r: r.get("pure_ms"), "pure Python", "s3"),
              (lambda r: r.get("gpu_kernel_ms"), "v1 ring kernel", "s4"),
              (lambda r: r["spill_kernel_ms"], "v2 spill kernel", "s1")]
    return log_dot_plot(SBS_ROWS, series,
                        "Single-block stage runtime per scene, log scale",
                        lambda r: SBS_LABELS.get(r["scene"], r["scene"]))


def dual_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>levels</th>"
            "<th>@njit ms</th><th>v2 ms</th><th>split ms</th>"
            "<th>global ms</th><th>dirsplit ms</th><th>best vs v2</th>"
            "<th>split bal %</th><th>dirsplit bal %</th>"
            "<th>inbox b0/b1</th><th>spilled b0/b1</th>"
            "<th>ovh% s/g/d</th></tr>")
    body = []
    for r in ROWS:
        best = max(r[f"{k}_speedup_vs_v2"] for k in KERNELS)
        body.append(
            "<tr>"
            f"<td>{label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_int(r['levels'])}</td>"
            f"<td>{fmt_ms(r['njit_ms'])}</td>"
            f"<td>{fmt_ms(r['v2_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['split_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['global_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['dirsplit_kernel_ms'])}</td>"
            f"<td>{best:.2f}×</td>"
            f"<td>{r['split_balance_pct']:.0f}</td>"
            f"<td>{r['dirsplit_balance_pct']:.0f}</td>"
            f"<td>{fmt_int(r['split_inbox_to_b0'])}/{fmt_int(r['split_inbox_to_b1'])}</td>"
            f"<td>{fmt_int(r['split_spilled_b0'])}/{fmt_int(r['split_spilled_b1'])}</td>"
            f"<td>{r['split_instrumentation_overhead_pct']:.1f}/"
            f"{r['global_instrumentation_overhead_pct']:.1f}/"
            f"{r['dirsplit_instrumentation_overhead_pct']:.1f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


def sbs_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>levels</th>"
            "<th>ring ms</th><th>spill ms</th><th>spilled px</th>"
            "<th>@njit ms</th><th>pure ms</th></tr>")
    body = []
    for r in SBS_ROWS:
        body.append(
            "<tr>"
            f"<td>{SBS_LABELS.get(r['scene'], r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_int(r['levels'])}</td>"
            f"<td>{fmt_ms(r.get('gpu_kernel_ms'))}</td>"
            f"<td>{fmt_ms(r['spill_kernel_ms'])}</td>"
            f"<td>{fmt_int(r['spilled_px'])}</td>"
            f"<td>{fmt_ms(r['njit_ms'])}</td>"
            f"<td>{fmt_ms(r.get('pure_ms'))}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


# --------------------------------------------------------- dual-blob stage
DB_LABELS = {
    "two_sq_300": "two squares 300² (small)",
    "two_sq_2800": "two squares 2800²",
    "two_disks_r1400": "two disks r=1400",
    "asym_4000_800": "asymmetric 4000²+800²",
}


def db_label(name):
    return DB_LABELS.get(name, name)


def db_runtime_chart():
    series = [(lambda r: r["njit_ms"], "@njit CPU (2 blobs)", "s2"),
              (lambda r: r["seq_ms"], "sequential", "s7"),
              (lambda r: r["multi_ms"], "multisource", "s8")]
    return log_dot_plot(DB_ROWS, series,
                        "Dual-blob runtime per scene, log scale",
                        lambda r: db_label(r["scene"]))


def db_speedup_chart():
    """Sequential vs multisource kernel time per scene, log axis, dumbbell
    pairs — two SOLID dots (s7, s8): sequential and multisource are two
    distinct mechanisms, not variants of one entity, so unlike conn8_chart
    neither dot is hollow (hollow is reserved for true variants, e.g. lin
    vs xy encoding of the SAME multisource kernel — see db_table)."""
    rows = DB_ROWS
    gut_r = 92
    vals = [v for r in rows for v in (r["seq_ms"], r["multi_ms"])]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * ROW_H + 34
    span = W - GUT_L - gut_r

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Sequential '
             f'vs multisource kernel time per scene">']
    tick = lo
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        if x < W - gut_r - 70:
            parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                         f'class="tick" text-anchor="middle">{tick:g}</text>')
        tick *= 10
    parts.append(f'<text x="{W - gut_r}" y="{n * ROW_H + 18}" class="tick" '
                 f'text-anchor="end">ms (log)</text>')
    for i, r in enumerate(rows):
        cy = i * ROW_H + ROW_H / 2
        v_seq, v_mu = r["seq_ms"], r["multi_ms"]
        x_seq, x_mu = x_of(v_seq), x_of(v_mu)
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - gut_r}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{db_label(r["scene"])}</text>')
        parts.append(f'<line x1="{x_seq:.1f}" y1="{cy:.1f}" x2="{x_mu:.1f}" '
                     f'y2="{cy:.1f}" class="pairline-db"/>')
        tip_seq = f"{db_label(r['scene'])} — sequential: {fmt_ms(v_seq)} ms"
        tip_mu = (f"{db_label(r['scene'])} — multisource: {fmt_ms(v_mu)} ms "
                 f"(min {fmt_ms(r['multi_ms_min'])})")
        parts.append(f'<circle cx="{x_seq:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s7" data-tip="{tip_seq}"/>')
        parts.append(f'<circle cx="{x_mu:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s8" data-tip="{tip_mu}"/>')
        ratio = r["speedup_multi_vs_seq"]
        parts.append(f'<text x="{W - gut_r + 8:.1f}" y="{cy + 4:.1f}" '
                     f'class="rowval" font-weight="650">{ratio:.2f}×</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def db_conn8_chart():
    """Per-scene 4-conn vs 8-conn kernel time for BOTH mechanisms, log
    axis. Two dumbbells per row — sequential (s7) above, multisource (s8)
    below — each with solid dot = the 4-conn baseline, hollow dot = the
    verbatim 8-conn twin (one entity at two variants, the same convention
    as the §1.4 dumbbell; the entity color stays with the mechanism).
    Gutter: each dumbbell's own 4→8 ratio, >1 = the 8-conn twin faster."""
    rows = DB_ROWS
    gut_r = 92
    row_h = ROW_H + 18          # two dumbbells per row need the headroom
    vals = [v for r in rows for v in (r["seq_ms"], r["conn8_seq_ms"],
                                      r["multi_ms"], r["conn8_multi_ms"])]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * row_h + 34
    span = W - GUT_L - gut_r

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="4-conn vs '
             f'8-conn kernel time per scene, sequential and multisource">']
    tick = lo
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * row_h}" class="grid"/>')
        if x < W - gut_r - 70:
            parts.append(f'<text x="{x:.1f}" y="{n * row_h + 18}" '
                         f'class="tick" text-anchor="middle">{tick:g}</text>')
        tick *= 10
    parts.append(f'<text x="{W - gut_r}" y="{n * row_h + 18}" class="tick" '
                 f'text-anchor="end">ms (log)</text>')
    for i, r in enumerate(rows):
        cy = i * row_h + row_h / 2
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - gut_r}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{db_label(r["scene"])}</text>')
        pairs = ((-9, "s7", "pairline-s7", "dot-o7", "sequential",
                  r["seq_ms"], r["conn8_seq_ms"], None, None),
                 (9, "s8", "pairline-s8", "dot-o8", "multisource",
                  r["multi_ms"], r["conn8_multi_ms"],
                  r["levels_multi"], r["conn8_levels_multi"]))
        for dy_off, cls, pl_cls, o_cls, mech, v4, v8, lv4, lv8 in pairs:
            yy = cy + dy_off
            x4, x8 = x_of(v4), x_of(v8)
            parts.append(f'<line x1="{x4:.1f}" y1="{yy:.1f}" x2="{x8:.1f}" '
                         f'y2="{yy:.1f}" class="{pl_cls}"/>')
            lvl4 = f", {fmt_int(lv4)} levels" if lv4 else ""
            lvl8 = f", {fmt_int(lv8)} levels" if lv8 else ""
            tip4 = (f"{db_label(r['scene'])} — {mech} 4-conn: "
                    f"{fmt_ms(v4)} ms{lvl4}")
            tip8 = (f"{db_label(r['scene'])} — {mech} 8-conn: "
                    f"{fmt_ms(v8)} ms{lvl8}")
            parts.append(f'<circle cx="{x4:.1f}" cy="{yy:.1f}" r="5" '
                         f'class="dot {cls}" data-tip="{tip4}"/>')
            parts.append(f'<circle cx="{x8:.1f}" cy="{yy:.1f}" r="5" '
                         f'class="{o_cls}" data-tip="{tip8}"/>')
            ratio = v4 / v8
            if abs(ratio - 1) < 0.005:
                rtxt, weight = "≈1.00×", ""
            elif ratio >= 1:
                rtxt, weight = f"{ratio:.2f}×", ' font-weight="650"'
            else:
                rtxt, weight = f"{ratio:.2f}× slower", ""
            parts.append(f'<text x="{W - gut_r + 8:.1f}" y="{yy + 4:.1f}" '
                         f'class="rowval"{weight}>{rtxt}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def db_r2_chart():
    """Per-scene conn8 vs radius-2 kernel time for BOTH mechanisms, log
    axis — db_conn8_chart's two-dumbbells-per-row geometry over the
    dual_blob_radius2 JSON. Solid dot = the 8-conn baseline, hollow = its
    guarded ring-2 twin (one entity at two variants, per mechanism);
    entity color stays with the mechanism (sequential s7, multisource
    s8). Gutter: each dumbbell's own ratio, >1 = the ring-2 twin faster."""
    rows = DBR2_ROWS
    gut_r = 108
    row_h = ROW_H + 18
    vals = [v for r in rows
            for v in (r["seq8_ms"], r["seq8r2_ms"],
                      r["multi8_ms"], r["multi8r2_ms"])]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * row_h + 34
    span = W - GUT_L - gut_r

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="8-conn vs '
             f'radius-2 kernel time per scene, sequential and multisource">']
    tick = lo
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * row_h}" class="grid"/>')
        if x < W - gut_r - 70:
            parts.append(f'<text x="{x:.1f}" y="{n * row_h + 18}" '
                         f'class="tick" text-anchor="middle">{tick:g}</text>')
        tick *= 10
    parts.append(f'<text x="{W - gut_r}" y="{n * row_h + 18}" class="tick" '
                 f'text-anchor="end">ms (log)</text>')
    for i, r in enumerate(rows):
        cy = i * row_h + row_h / 2
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - gut_r}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{db_label(r["scene"])}</text>')
        pairs = ((-9, "s7", "pairline-s7", "dot-o7", "sequential",
                  r["seq8_ms"], r["seq8r2_ms"],
                  r["r2_seq_vs_conn8"], r["r2_seq_vs_conn8_min"]),
                 (9, "s8", "pairline-s8", "dot-o8", "multisource",
                  r["multi8_ms"], r["multi8r2_ms"],
                  r["r2_multi_vs_conn8"], r["r2_multi_vs_conn8_min"]))
        for dy_off, cls, pl_cls, o_cls, mech, v8, vr, ratio, rmin in pairs:
            yy = cy + dy_off
            x8, xr = x_of(v8), x_of(vr)
            parts.append(f'<line x1="{x8:.1f}" y1="{yy:.1f}" x2="{xr:.1f}" '
                         f'y2="{yy:.1f}" class="{pl_cls}"/>')
            tip8 = (f"{db_label(r['scene'])} — {mech} 8-conn: "
                    f"{fmt_ms(v8)} ms")
            tipr = (f"{db_label(r['scene'])} — {mech} radius-2: "
                    f"{fmt_ms(vr)} ms (best-vs-best ratio {rmin:.2f}×)")
            parts.append(f'<circle cx="{x8:.1f}" cy="{yy:.1f}" r="5" '
                         f'class="dot {cls}" data-tip="{tip8}"/>')
            parts.append(f'<circle cx="{xr:.1f}" cy="{yy:.1f}" r="5" '
                         f'class="{o_cls}" data-tip="{tipr}"/>')
            if abs(ratio - 1) < 0.005:
                rtxt, weight = "≈1.00×", ""
            elif ratio >= 1:
                rtxt, weight = f"{ratio:.2f}×", ' font-weight="650"'
            else:
                rtxt, weight = f"{ratio:.2f}× slower", ""
            parts.append(f'<text x="{W - gut_r + 8:.1f}" y="{yy + 4:.1f}" '
                         f'class="rowval"{weight}>{rtxt}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def db_r2_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>seq8 ms</th>"
            "<th>multi8 ms</th><th>seq8-r2 ms</th><th>multi8-r2 ms</th>"
            "<th>r2 multi vs 8</th><th>(min)</th>"
            "<th>mu/seq 8→r2</th><th>levels 8→r2</th>"
            "<th>interior %</th></tr>")
    body = []
    for r in DBR2_ROWS:
        body.append(
            "<tr>"
            f"<td>{db_label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_ms(r['seq8_ms'])}</td>"
            f"<td>{fmt_ms(r['multi8_ms'])}</td>"
            f"<td>{fmt_ms(r['seq8r2_ms'])}</td>"
            f"<td>{fmt_ms(r['multi8r2_ms'])}</td>"
            f"<td>{r['r2_multi_vs_conn8']:.2f}×</td>"
            f"<td>{r['r2_multi_vs_conn8_min']:.2f}×</td>"
            f"<td>{r['conn8_speedup_multi_vs_seq']:.2f}→"
            f"{r['r2_speedup_multi_vs_seq']:.2f}×</td>"
            f"<td>{fmt_int(r['conn8_levels_multi'])}→"
            f"{fmt_int(r['r2_levels_multi'])}</td>"
            f"<td>{r['r2_interior_pct']:.1f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


def db_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>@njit ms</th>"
            "<th>seq ms</th><th>seq/2 ms</th><th>multi ms</th>"
            "<th>multi min</th><th>vs seq</th><th>vs seq (min)</th>"
            "<th>xy ms</th><th>xy/lin</th><th>pack tax %</th>"
            "<th>8-conn seq ms</th><th>8-conn multi ms</th>"
            "<th>8 vs 4-conn</th><th>8-conn mu/seq</th>"
            "<th>GB/s</th><th>% peak</th></tr>")
    body = []
    for r in DB_ROWS:
        body.append(
            "<tr>"
            f"<td>{db_label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_ms(r['njit_ms'])}</td>"
            f"<td>{fmt_ms(r['seq_ms'])}</td>"
            f"<td>{fmt_ms(r['seq_half_ms'])}</td>"
            f"<td>{fmt_ms(r['multi_ms'])}</td>"
            f"<td>{fmt_ms(r['multi_ms_min'])}</td>"
            f"<td>{r['speedup_multi_vs_seq']:.2f}×</td>"
            f"<td>{r['speedup_multi_vs_seq_min']:.2f}×</td>"
            f"<td>{fmt_ms(r['multi_xy_ms'])}</td>"
            f"<td>{r['xy_vs_lin']:.2f}×</td>"
            f"<td>{r['packing_tax_pct']:+.1f}</td>"
            f"<td>{fmt_ms(r['conn8_seq_ms'])}</td>"
            f"<td>{fmt_ms(r['conn8_multi_ms'])}</td>"
            f"<td>{r['conn8_multi_vs_conn4']:.2f}×</td>"
            f"<td>{r['conn8_speedup_multi_vs_seq']:.2f}×</td>"
            f"<td>{r['multi_model_gb_s']:.1f}</td>"
            f"<td>{r['multi_pct_of_peak']:.1f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


if HAS_DUALBLOB:
    _db_biggest = max(DB_ROWS, key=lambda r: r["filled"])
    _db_best_spd = max(DB_ROWS, key=lambda r: r["speedup_multi_vs_seq"])
    _db8_best = max(DB_ROWS, key=lambda r: r["conn8_multi_vs_conn4"])
    _db8_wash = min(DB_ROWS, key=lambda r: r["conn8_multi_vs_conn4"])
    _db8_spd_lo = min(r["conn8_speedup_multi_vs_seq"] for r in DB_ROWS)
    _db8_spd_hi = max(r["conn8_speedup_multi_vs_seq"] for r in DB_ROWS)
    _db8_lvl = max(DB_ROWS, key=lambda r: r["levels_multi"])
    _db8_seq_back = min(DB_ROWS, key=lambda r: r["seq_ms"] / r["conn8_seq_ms"])
    _db8_back_ratio = _db8_seq_back["seq_ms"] / _db8_seq_back["conn8_seq_ms"]
    _db8_back_note = ""
    if _db8_back_ratio < 0.9:
        _db8_back_note = (
            f" One dumbbell points backwards — "
            f"{db_label(_db8_seq_back['scene'])}, sequential "
            f"({_db8_back_ratio:.2f}×): the same scene whose 4-conn numbers "
            f"already showed this session's worst drift (median vs min "
            f"mu/seq: {_db8_seq_back['speedup_multi_vs_seq']:.2f}× vs "
            f"{_db8_seq_back['speedup_multi_vs_seq_min']:.2f}×), so treat "
            f"that single reversal with the house median-vs-min caution.")
    _db_tax_lo = min(r["packing_tax_pct"] for r in DB_ROWS)
    _db_tax_hi = max(r["packing_tax_pct"] for r in DB_ROWS)
    DB_TILES = [
        (f"{fmt_ms(_db_biggest['multi_ms'])} ms",
         f"biggest scene · {db_label(_db_biggest['scene'])} "
         f"({fmt_int(_db_biggest['filled'])} px) · "
         f"{_db_biggest['speedup_multi_vs_njit']:.1f}× vs @njit"),
        (f"{_db_best_spd['speedup_multi_vs_seq']:.2f}× "
         f"(min {_db_best_spd['speedup_multi_vs_seq_min']:.2f}×)",
         f"multisource's best payoff vs sequential · "
         f"{db_label(_db_best_spd['scene'])}"),
        (f"{_db_tax_lo:+.1f}% to {_db_tax_hi:+.1f}%",
         "packing tax vs the published single-blob kernel — straddles "
         "zero: unmeasurable, not zero-and-hidden (see Finding 3)"),
    ]
    db_tiles_html = "".join(
        f'<div class="tile"><div class="tile-v">{v}</div>'
        f'<div class="tile-l">{l}</div></div>' for v, l in DB_TILES)
else:
    db_tiles_html = ""


# -------------------------------------------------------------- stat tiles
_big = max(ROWS, key=lambda r: r["filled"])
_best_v2 = max(max(r[f"{k}_speedup_vs_v2"] for k in KERNELS) for r in ROWS)
_pl = {(r["scene"], r["config"]): r for r in PLACEMENT}
_same = _pl[("sq_6000_center", "pinned 2x768 same SM")]["kernel_ms"]
_spread = _pl[("sq_6000_center", "pinned 2x768 spread")]["kernel_ms"]
_serp = next(r for r in ROWS if r["scene"] == "serpentine_256")
_sync_us = _serp["global_kernel_ms"] / (_serp["levels"] * 2) * 1000
_inb_max = max(max(r["split_inbox_to_b0"], r["split_inbox_to_b1"])
               for r in ROWS if "serpentine" not in r["scene"]
               and r["scene"] != "sq_4000_corner")
TILES = [
    (f"{_best_v2:.2f}× vs 1 block",
     f"the 2nd block's best payoff · {label(_big['scene'])} "
     f"({fmt_int(_big['filled'])} px)"),
    (f"{_same / _spread:.2f}× spread vs same-SM",
     "placement verdict · 2 SMs beat 1 full SM — parallelism over concurrency"),
    (f"{_sync_us:.1f} µs per grid.sync",
     f"measured via the serpentine's {fmt_int(_serp['levels'])} levels × 2 "
     f"barriers"),
    (f"≤ {_inb_max} px inbox traffic",
     "cross-seam handoffs on center-seeded scenes · structural bound: height"),
]
tiles_html = "".join(
    f'<div class="tile"><div class="tile-v">{v}</div>'
    f'<div class="tile-l">{l}</div></div>' for v, l in TILES)



PROJECT_TILES = list(mb_viz.TILES)
if HAS_DUALBLOB:
    PROJECT_TILES.append(
        (f"{_db_best_spd['speedup_multi_vs_seq']:.2f}×",
         f"two blobs, one pass · multisource vs sequential · "
         f"{db_label(_db_best_spd['scene'])} — see §2",
         "dual-blob stage"))
project_tiles_html = "".join(
    f'<div class="tile"><div class="tile-v">{v}</div>'
    f'<div class="tile-l">{l}</div><div class="tile-src">{s}</div></div>'
    for v, l, s in PROJECT_TILES)

# --------------------------------------------------- chained speedups
# Each subsection reports its OWN stage's speedup from ITS OWN benchmark
# session (1.1/1.2 use single_block_shared's/dual_block's own JSON). 1.3
# additionally chains njit -> v2 -> dual-global -> N-blocks together using
# multi_block's OWN JSON, which re-measures v2 and dual-global fresh in
# the SAME session as its N-block numbers — one self-consistent product,
# never a cross-session multiply. (Three multi_block_*.json sessions on
# disk disagree slightly on njit/kernel timings from normal GPU clock
# drift; only the newest file's own numbers ever get multiplied here.)
_sbs_v2_36m = next(r for r in SBS_ROWS if r["scene"] == "sq_6000_center")
_dual_36m = next(r for r in ROWS if r["scene"] == "sq_6000_center")

CHAIN_NOTE = viz.CHAIN_NOTE

CHAIN_11 = [("CPU (@njit)", _sbs_v2_36m["njit_ms"]),
           ("v2 spill kernel", _sbs_v2_36m["spill_kernel_ms"])]
CHAIN_12 = [("CPU (@njit)", _dual_36m["njit_ms"]),
           ("single-block v2", _dual_36m["v2_kernel_ms"]),
           ("dual global", _dual_36m["global_kernel_ms"])]

_db_2800 = next((r for r in DB_ROWS if r["scene"] == "two_sq_2800"), None)
CHAIN_DB = ([("CPU (@njit, 2 blobs)", _db_2800["njit_ms"]),
            ("sequential", _db_2800["seq_ms"]),
            ("multisource", _db_2800["multi_ms"])]
           if _db_2800 else None)

balance_html, balance_js = balance_panels()

CSS = viz.CSS

JS = viz.TOOLTIP_JS + """const BPANELS = __BPANELS__;
document.querySelectorAll('.bhover-capture').forEach(el => {
  const name = el.dataset.panel, p = BPANELS[name];
  const svg = document.getElementById('b_' + name);
  const xh = document.getElementById('bxh_' + name);
  const d0 = document.getElementById('bxd0_' + name);
  const d1 = document.getElementById('bxd1_' + name);
  const toX = lv => p.pl + lv / Math.max(p.n - 1, 1) * (p.w - p.pl - p.pr);
  const toY = v => p.pt + (1 - v / p.peak) * (p.h - p.pt - p.pb);
  el.addEventListener('mousemove', ev => {
    const box = svg.getBoundingClientRect();
    const sx = (ev.clientX - box.left) / box.width * p.w;
    let lo = 0, hi = p.xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (toX(p.xs[mid]) < sx) lo = mid; else hi = mid;
    }
    const i = (sx - toX(p.xs[lo]) < toX(p.xs[hi]) - sx) ? lo : hi;
    const cx = toX(p.xs[i]);
    xh.setAttribute('x1', cx); xh.setAttribute('x2', cx);
    xh.setAttribute('visibility', 'visible');
    d1.setAttribute('cx', cx); d1.setAttribute('cy', toY(p.c1[i]));
    d1.setAttribute('visibility', 'visible');
    d0.setAttribute('cx', cx); d0.setAttribute('cy', toY(p.c0[i]));
    d0.setAttribute('visibility', 'visible');
    showTip('level ' + p.xs[i].toLocaleString() + ' \\u2014 block 0: ' +
            p.c0[i].toLocaleString() + ' px \\u00b7 block 1: ' +
            p.c1[i].toLocaleString() + ' px (cumulative)', ev);
  });
  el.addEventListener('mouseleave', () => {
    xh.setAttribute('visibility', 'hidden');
    d0.setAttribute('visibility', 'hidden');
    d1.setAttribute('visibility', 'hidden');
    hideTip();
  });
});
""".replace("__BPANELS__", balance_js)


legend = viz.legend


chain_strip = viz.chain_strip


LEG5 = legend([("@njit CPU", "s2"), ("single-block v2", "s1"),
               ("split", "s3"), ("global", "s4"), ("dirsplit", "s5")])
LEG3 = legend([("split", "s3"), ("global", "s4"), ("dirsplit", "s5")])
LEG_BAL = legend([("block 0", "s1"), ("block 1", "s4")])
LEG_SBS = legend([("@njit CPU", "s2"), ("pure Python", "s3"),
                  ("v1 ring kernel", "s4"), ("v2 spill kernel", "s1")])
LEG_DB_R2 = ('<div class="legend">'
             '<span><i class="chip" style="background:var(--s7)"></i>'
             'sequential 8-conn</span>'
             '<span><i class="chip chip-o" '
             'style="border-color:var(--s7)"></i>sequential radius-2</span>'
             '<span><i class="chip" style="background:var(--s8)"></i>'
             'multisource 8-conn</span>'
             '<span><i class="chip chip-o" '
             'style="border-color:var(--s8)"></i>multisource radius-2</span>'
             '</div>')
LEG_DB = legend([("@njit CPU (2 blobs)", "s2"), ("sequential", "s7"),
                 ("multisource", "s8")])
LEG_DB_CONN8 = ('<div class="legend">'
                '<span><i class="chip" style="background:var(--s7)"></i>'
                'sequential 4-conn</span>'
                '<span><i class="chip chip-o" '
                'style="border-color:var(--s7)"></i>sequential 8-conn</span>'
                '<span><i class="chip" style="background:var(--s8)"></i>'
                'multisource 4-conn</span>'
                '<span><i class="chip chip-o" '
                'style="border-color:var(--s8)"></i>multisource 8-conn</span>'
                '</div>')

# Conditional zone-3 blocks: only render where the experiment applies.
_db_r2_card = ""
if HAS_DB_R2:
    _dbr2_big = max(DBR2_ROWS, key=lambda r: r["filled"])
    _db_r2_card = f"""
<div class="card">
<h2>Radius-2 on two blobs — the same bet, labeled</h2>
<p class="note">The §1.5 verdict ports cleanly: the guarded ring-2 twin
(hollow) loses to its 8-conn baseline (solid) for both mechanisms, while
the multisource-vs-sequential win survives the ring-2 tax where the data
is clean (mu/seq column in the table). Ring-2 claims inherit the
dequeuer's label — provably safe (the guard keeps every jump inside the
dequeuer's own component; tested across a 1-px gap, tighter than the
scene contract). The 2800² pair is the session's drift casualty: its
median and min mu/seq forms disagree in sign — see the stage README.</p>
{LEG_DB_R2}
{db_r2_chart()}
<details><summary>Radius-2 per-scene table</summary>
<div class="tablewrap">{db_r2_table()}</div></details>
</div>
"""

_db_section = ""
if HAS_DUALBLOB:
    _db_wavefront_note = (
        '<p class="note"><img src="../../ch04_gpu_2blob_nblock/wavefront/'
        'asym384_b8_t32_multisource.gif" alt="Multisource wavefront on an '
        'asymmetric blob pair: the small (green) blob finishes early and '
        'stays light while the large (blue) blob keeps darkening — one '
        'shared clock, not two." style="max-width:340px; border-radius:8px; '
        'display:block; margin:10px 0;">One shared clock: the small (green) '
        'blob finishes early and stays light while the large (blue) blob '
        'keeps darkening — max(tA, tB), not tA + tB. Sequential-replay '
        'contrast GIF and the full write-up live in the stage '
        '<code>README.md</code>.</p>')
    _db_section = f"""
<h2 class="domain-h">2. Dual blob</h2>
<p class="sub">Two disconnected red blobs, labeled and recolored
in-kernel (blob 0 blue, blob 1 green) by the SAME N-block cooperative
kernel family as §1 — the label rides in spare bits of the queue entry,
so it costs zero extra bytes. Three mechanisms tested: sequential (two
launches), streams (two CUDA streams — excluded below, see the stage
README's Finding 2: concurrent cooperative launches wedge
nondeterministically on this GPU), and multisource (both seeds in one
shared queue, one launch). Each mechanism also has a verbatim
8-direction twin — compared against its 4-direction baseline at the end
of this section.</p>

<div class="tiles">{db_tiles_html}</div>

<div class="card">
<h2>The chain — CPU to multisource, two blobs at once</h2>
<p class="note">Its own chain, not multiplied into §1.3's total: this
solves a different problem (filling TWO blobs) with a different CPU
baseline. Sequential's baseline already embeds §1's per-blob N-block
speedup (near-zero labeling tax); multisource adds a further ~1.6–2× on
top, specifically for running two blobs at once.</p>
{chain_strip(CHAIN_DB, footnote=CHAIN_NOTE, total_cls="s8") if CHAIN_DB else ""}
</div>

<div class="card">
<h2>Runtime per scene</h2>
<p class="note">Log scale. Sequential pays tA + tB; multisource shares
one clock — levels become max(a, b) instead of a + b, and every level is
twice as wide. Hover any dot; exact numbers in the table below.</p>
{LEG_DB}
{db_runtime_chart()}
</div>

<div class="card">
<h2>What multisource buys — speedup vs sequential</h2>
<p class="note">Two solid dots, not a hollow pair: sequential and
multisource are different mechanisms, not variants of one entity (unlike
the 4-conn/8-conn dumbbell in §1.4). On the asymmetric pair, multisource
fills BOTH blobs faster than sequential filled the big one alone
(multi_vs_ideal &lt; 1).</p>
{db_speedup_chart()}
</div>

{_db_wavefront_note}

<div class="card">
<h2>4 vs 8 directions — the same two blobs</h2>
<p class="note">The §1.4 experiment repeated on the labeled two-blob
kernels: every mechanism has a verbatim 8-direction twin (diagonals
included, label inheritance untouched), measured in the same interleaved
round-robin as everything above. Solid dot = 4-conn, hollow = the 8-conn
twin, one dumbbell per mechanism per scene. Diagonals cut BFS depth to
the Chebyshev distance ({fmt_int(_db8_lvl['levels_multi'])} →
{fmt_int(_db8_lvl['conn8_levels_multi'])} levels on
{db_label(_db8_lvl['scene'])}) and mostly buy time — up to
{_db8_best['conn8_multi_vs_conn4']:.2f}× for multisource, at
{db_label(_db8_best['scene'])} — though at {db_label(_db8_wash['scene'])}
the multisource twin reads as a wash
({_db8_wash['conn8_multi_vs_conn4']:.2f}×) even as its sequential twin
improves. The headline survives the extra directions: at 8-conn,
multisource still beats its own 8-conn sequential baseline
{_db8_spd_lo:.2f}–{_db8_spd_hi:.2f}× (the "8-conn mu/seq" table
column).{_db8_back_note}</p>
{LEG_DB_CONN8}
{db_conn8_chart()}
</div>

{_db_r2_card}

<div class="card">
<h2>All numbers — dual-blob stage</h2>
<p class="note">The lin-vs-xy entry-format bet (does killing the
per-pixel integer divide matter?) reads as a wash in the xy/lin column —
confirmed by an interleaved controlled A/B in the stage README after an
uncontrolled first pass wrongly read it as a 28% loss (see that README's
Finding 3 on timing methodology). The packing-tax verdict (which
straddles zero — unmeasurable, not "no tax") is a column here rather
than its own chart, matching how this project already treats
small/inconclusive findings elsewhere.</p>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{db_table()}</div></details>
</div>
"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flood fill — 1 → 2 → N blocks → 8 directions → 2 blobs, benchmarked</title>
<style>{CSS}</style></head>
<body><div class="viz-root">
<h1>BFS flood fill — 1 block → 2 blocks → N blocks → 8 directions →
2 blobs, benchmarked</h1>
<div class="sub">{DUAL['device']} · {DUAL['sm_count']} SMs · organized by
domain: single blob (§1.1 one block → §1.2 two blocks → §1.3 N blocks →
§1.4 4-vs-8 connectivity → §1.5 per-barrier work experiments), then dual
blob (§2). Every subsection reports
its own speedup from its own benchmark session; §1.3 chains them into one
multiplier from single block to N blocks. 4-connectivity throughout
except where 8-direction twins are charted explicitly · placement
observed via %smid</div>

<div class="tiles">{project_tiles_html}</div>

<h2 class="domain-h" style="margin-top:8px">1. Single blob</h2>

<h3 class="subsection-h">1.1 Single block</h3>
<div class="card">
<h2>Runtime — v1 ring vs v2 spill vs @njit</h2>
<p class="note">Log scale — each decade gridline is 10×. v1 (the pure
shared-memory ring) is fastest until the frontier outgrows it; v2 (the
spill tier) is what every later stage is measured against. Pure Python
was skipped above 2M px.</p>
{LEG_SBS}
{sbs_chart()}
<details><summary>Single-block per-scene results table</summary>
<div class="tablewrap">{sbs_table()}</div></details>
</div>

<div class="card">
<h2>What one GPU block buys — chained from the CPU</h2>
<p class="note">v2's own speedup vs @njit, from single_block_shared's own
benchmark session, at its biggest scene (36M px).</p>
{chain_strip(CHAIN_11, footnote=CHAIN_NOTE, total_cls="s1")}
</div>

<h3 class="subsection-h">1.2 Dual blocks — split, global, dirsplit</h3>
<div class="tiles">{tiles_html}</div>

<div class="card">
<h2>Runtime per scene</h2>
<p class="note">Log scale — each decade gridline is 10×. The three dual
kernels cluster tightly; their lead over the single-block v2 (blue) grows
with blob size, and all of them lose to v2 on the serpentines, where two
grid.sync barriers per ~1-pixel level dominate. Hover any dot; exact
numbers in the table below.</p>
{LEG5}
{runtime_chart()}
</div>

<div class="card">
<h2>What the second block buys — speedup vs single-block v2</h2>
<p class="note">Kernel-only speedup per partitioning. Right of the dashed
line, two blocks beat one; the payoff grows from 1.6× at 1M px to 2.1× at
36M px. Left of it: the small scenes (barrier tax) and the serpentines
(≈1.9 µs × 2 barriers × tens of thousands of levels).</p>
{LEG3}
{speedup_chart()}
</div>

<div class="card">
<h2>The placement experiment — where the threads live</h2>
<p class="note">Same BFS, four configurations; the observed %smid pair on
each bar proves the placement. Filling ONE SM to its full 1,536-thread
residency (only possible with two blocks — a single block caps at 1,024)
buys ~12% at 36M px; spreading the same pair across two SMs buys another
1.68×. Concurrency helps; parallelism wins.</p>
{placement_chart()}
</div>

<div class="card">
<h2>Balance over time — cumulative work per block</h2>
<p class="note">Cumulative pixels processed by each block, level by level.
Top-left: the split kernel's failure mode (block 1 owns nothing). Top-right:
dirsplit on the same blob — spatial agnosticism, both blocks climb. Bottom-left:
dirsplit's own failure — the aggregate totals end 98.5% "balanced," but the
alternating staircase shows the blocks taking turns, never working
together. Bottom-right: the ideal — a seam-symmetric blob mirrors
perfectly. Hover for exact values.</p>
{LEG_BAL}
{balance_html}
</div>

<div class="card">
<h2>Throughput vs threads per block — both stages</h2>
<p class="note">The dual kernels stop at 512 (their coordination state
costs ~104–159 registers/thread; a 1,024-thread block cannot be resident),
so their rightmost points are 2×512 = 1,024 total threads — directly
comparable to the single-block line's 1×1,024 end, and ~15% above it.
Single-block sweep numbers come from that stage's own benchmark run.</p>
{sweep_chart()}
</div>

<div class="card">
<h2>Instrumentation overhead — measured, not assumed</h2>
<p class="note">Each instrumented kernel vs its bare twin (identical BFS,
counters and traces stripped). global and dirsplit sit at 0–3%
(indistinguishable from run noise at scale, occasionally negative); split
pays 1.4–8.7% for its per-level publish/clamp choreography, worst on tiny
scenes where the fixed cost has nothing to amortize against.</p>
{LEG3}
{overhead_chart()}
</div>

<div class="card">
<h2>All numbers — dual-block stage</h2>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{dual_table()}</div></details>
</div>

<div class="card">
<h2>What two blocks buy, chained from the CPU</h2>
<p class="note">Same idea as §1.1's chip, one link longer — dual_block's
own session, its biggest scene (36M px).</p>
{chain_strip(CHAIN_12, footnote=CHAIN_NOTE, total_cls="s4")}
</div>

{mb_viz.SECTION_1_3}

{mb_viz.SECTION_1_4}
{mb_viz.SECTION_1_5}
{_db_section}
<div id="tooltip"></div>
</div>
<script>{JS}</script>
</body></html>
"""

with open(OUT_PATH, "w") as f:
    f.write(html)
_db_part = f" + {os.path.basename(DB_PATH)}" if DB_PATH else " (no dual_blob JSON — §2 omitted)"
_nb_part = (f" + {os.path.basename(mb_viz.NB_PATH)}" if mb_viz.NB_PATH
            else " (no neighbors JSON — §1.5 omitted)")
_dbr2_part = (f" + {os.path.basename(DBR2_PATH)}" if DBR2_PATH
              else " (no dual_blob_radius2 JSON — §2 r2 card omitted)")
print(f"rendered {os.path.basename(mb_viz.MB_PATH)} + {os.path.basename(DUAL_PATH)}"
      f" + {os.path.basename(SBS_PATH)}{_nb_part}{_db_part}{_dbr2_part}"
      f" -> {OUT_PATH} ({len(html):,} bytes)")
