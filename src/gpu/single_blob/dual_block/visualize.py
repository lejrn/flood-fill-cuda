"""
Generate the combined multi-block + dual-block + single-block dashboard.

Renders, newest stage first, from the newest multi_block, dual_block (or
argv[1]) and single_block_shared benchmark JSONs: the N-block stage (stat
tiles, 4-series runtime, speedup vs three baselines, the blocks x tpb
sweep panels, the bandwidth chart against the measured copy peak), the
dual-block stage (5-series runtime, speedup vs v2, the placement
experiment with observed %smid annotations, balance-over-time small
multiples, the merged tpb sweep, instrumentation overhead), the appended
single-block stage, and full results tables — self-contained HTML, hover
tooltips, light/dark.

Usage:
    uv run python src/gpu/single_blob/dual_block/visualize.py [dual.json]

Output: benchmark_results/dual_block_benchmark.html (overwritten per run —
the timestamped JSON/CSV remain the durable record).
"""
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "benchmark_results")
SBS_RESULTS_DIR = os.path.join(_HERE, os.pardir, "single_block_shared",
                               "benchmark_results")
MB_RESULTS_DIR = os.path.join(_HERE, os.pardir, "multi_block",
                              "benchmark_results")


def _newest(pattern, folder):
    candidates = sorted(glob.glob(os.path.join(folder, pattern)))
    if not candidates:
        sys.exit(f"no benchmark JSON matching {pattern} — run the benchmark first")
    return candidates[-1]


DUAL_PATH = sys.argv[1] if len(sys.argv) > 1 else _newest("dual_block_*.json",
                                                          RESULTS_DIR)
SBS_PATH = _newest("single_block_shared_*.json", SBS_RESULTS_DIR)
MB_PATH = _newest("multi_block_*.json", MB_RESULTS_DIR)
OUT_PATH = os.path.join(RESULTS_DIR, "dual_block_benchmark.html")

with open(DUAL_PATH) as f:
    DUAL = json.load(f)
with open(SBS_PATH) as f:
    SBS = json.load(f)
with open(MB_PATH) as f:
    MB = json.load(f)

ROWS = DUAL["scenes"]
SWEEP = DUAL["tpb_sweep"]
PLACEMENT = DUAL["placement"]
SBS_ROWS = SBS["scenes"]
SBS_SWEEP = SBS["tpb_sweep"]
MB_ROWS = [r for r in MB["scenes"] if "skipped" not in r]
MB_SWEEP = [r for r in MB["block_tpb_sweep"] if "skipped" not in r]
MB_PEAK = MB["measured_peak_gb_s"]
MB_TPBS = MB["config"]["tpb_sweep"]

KERNELS = ["split", "global", "dirsplit"]
# Entity -> color slot, constant across every dual chart: s1 = single-block
# v2, s2 = @njit CPU, s3 = split, s4 = global, s5 = dirsplit.
KCLS = {"split": "s3", "global": "s4", "dirsplit": "s5"}

SCENE_LABELS = {
    "sq_256_center": "square 256² · center",
    "sq_1024_center": "square 1024² · center",
    "sq_2000_center": "square 2000² · center",
    "sq_4000_corner": "square 4000² · corner",
    "serpentine_256": "serpentine 256²",
    "seam_serpentine_256": "seam serpentine 256²",
    "offcenter_2000": "off-center blob 2000²",
    "sq_2600_full_center": "square 2600² full · center",
    "sq_4600_full_center": "square 4600² full · center",
    "sq_5000_center": "square 5000² · center",
    "sq_6000_center": "square 6000² · center",
    # multi-block stage additions
    "disk_2001_r950": "disk r=950",
    "disk_4001_r1900": "disk r=1900",
    "sq_8000_center": "square 8000² · center",
}
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


def label(name):
    return SCENE_LABELS.get(name, name)


def fmt_ms(v):
    if v is None:
        return "—"
    if v < 1:
        return f"{v:.2f}"
    if v < 10:
        return f"{v:.1f}"
    return f"{v:,.0f}"


def fmt_int(v):
    return f"{v:,}"


# ------------------------------------------------------------ shared geometry
W, ROW_H, GUT_L, GUT_R = 860, 36, 214, 30


def decimate(values, max_pts=600):
    n = len(values)
    if n <= max_pts:
        return list(range(n)), list(values)
    bin_size = math.ceil(n / max_pts)
    xs, ys = [], []
    for start in range(0, n, bin_size):
        chunk = values[start:start + bin_size]
        j = max(range(len(chunk)), key=chunk.__getitem__)
        xs.append(start + j)
        ys.append(chunk[j])
    return xs, ys


def log_dot_plot(rows, series, aria, row_label):
    """Rows x log-ms dot plot; series = [(value_fn, name, cls), ...]."""
    vals = [v for r in rows for fn, _, _ in series for v in [fn(r)]
            if v is not None]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * ROW_H + 34
    span = W - GUT_L - GUT_R

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{aria}">']
    tick = lo
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        if x < W - GUT_R - 70:
            parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                         f'class="tick" text-anchor="middle">{tick:g}</text>')
        tick *= 10
    parts.append(f'<text x="{W - GUT_R}" y="{n * ROW_H + 18}" class="tick" '
                 f'text-anchor="end">ms (log)</text>')
    for i, r in enumerate(rows):
        cy = i * ROW_H + ROW_H / 2
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - GUT_R}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{row_label(r)}</text>')
        for si, (fn, name, cls) in enumerate(series):
            v = fn(r)
            if v is None:
                continue
            # dodge: fixed per-series vertical offset within the row, so
            # near-identical values sit side by side instead of occluding
            dy = (si - (len(series) - 1) / 2) * 6.5
            tip = f"{row_label(r)} — {name}: {fmt_ms(v)} ms"
            parts.append(f'<circle cx="{x_of(v):.1f}" cy="{cy + dy:.1f}" '
                         f'r="5" class="dot {cls}" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


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


# ------------------------------------------------------- multi-block stage
def mb_runtime_chart():
    series = [(lambda r: r["njit_ms"], "@njit CPU", "s2"),
              (lambda r: r["v2_kernel_ms"], "single-block v2", "s1"),
              (lambda r: r["dual_global_kernel_ms"], "dual global (2 blocks)",
               "s4"),
              (lambda r: r["multi_kernel_ms"],
               f"multi ({MB_ROWS[0]['multi_blocks']} blocks)", "s6")]
    return log_dot_plot(MB_ROWS, series,
                        "N-block stage runtime per scene, log scale",
                        lambda r: label(r["scene"]))


def mb_speedup_chart():
    """Multi-block speedup vs each baseline entity, log scale (0.09x-15x)."""
    keys = [("multi_speedup_vs_v2", "vs single-block v2", "s1",
             "v2_kernel_ms"),
            ("multi_speedup_vs_dual", "vs dual global", "s4",
             "dual_global_kernel_ms"),
            ("multi_speedup_vs_njit", "vs @njit CPU", "s2", "njit_ms")]
    vals = [r[k] for r in MB_ROWS for k, _, _, _ in keys]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(MB_ROWS)
    h = n * ROW_H + 34
    span = W - GUT_L - GUT_R

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" '
             f'aria-label="N-block speedup vs three baselines, log scale">']
    tick = lo
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                     f'class="tick" text-anchor="middle">{tick:g}×</text>')
        tick *= 10
    px = x_of(1.0)
    parts.append(f'<line x1="{px:.1f}" y1="4" x2="{px:.1f}" '
                 f'y2="{n * ROW_H}" class="satline"/>')
    for i, r in enumerate(MB_ROWS):
        cy = i * ROW_H + ROW_H / 2
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - GUT_R}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label(r["scene"])}</text>')
        for si, (k, name, cls, base_key) in enumerate(keys):
            v = r[k]
            dy = (si - 1) * 6.5
            tip = (f"{label(r['scene'])} — {v:.2f}× {name} "
                   f"({fmt_ms(r['multi_kernel_ms'])} vs "
                   f"{fmt_ms(r[base_key])} ms)")
            parts.append(f'<circle cx="{x_of(v):.1f}" cy="{cy + dy:.1f}" '
                         f'r="5" class="dot {cls}" data-tip="{tip}"/>')
    # annotation painted last so the top row's dots cannot overprint it
    parts.append(f'<text x="{px - 6:.1f}" y="14" class="anno" '
                 f'text-anchor="end">right of this line, the N blocks beat '
                 f'that baseline — parity</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# (scene, metric, panel title): the serpentine panel plots kernel ms — its
# Mpx/s is a flat ~0.2 and the story is "no configuration helps".
MB_SWEEP_PANELS = [
    ("sq_4000_corner", "mpx_s", "square 4000² corner (16M px) — Mpx/s"),
    ("disk_4001_r1900", "mpx_s", "disk r=1900 (11.3M px) — Mpx/s"),
    ("serpentine_256", "kernel_ms", "serpentine 256² — kernel ms"),
]


def mb_sweep_panels():
    """Per-scene panels: metric vs blocks (log2 x), one line per tpb."""
    html = []
    for sname, metric, title in MB_SWEEP_PANELS:
        rows = [r for r in MB_SWEEP if r["scene"] == sname]
        blocks_all = sorted({r["blocks"] for r in rows})
        bmax = blocks_all[-1]
        vmax = max(r[metric] for r in rows) * 1.14
        w, h, pl, pr, pt, pb = 420, 230, 56, 64, 26, 34
        span = w - pl - pr

        def x_of(b):
            return pl + math.log2(b) / math.log2(bmax) * span

        def y_of(v):
            return pt + (1 - v / vmax) * (h - pt - pb)

        svg = [f'<svg viewBox="0 0 {w} {h}" role="img" '
               f'aria-label="Blocks × tpb sweep, {title}">']
        svg.append(f'<text x="{pl}" y="14" class="paneltitle">{title}</text>')
        for frac in (0, 0.5, 1):
            yy = pt + frac * (h - pt - pb)
            svg.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w - pr}" '
                       f'y2="{yy:.1f}" class="grid"/>')
            svg.append(f'<text x="{pl - 6}" y="{yy + 4:.1f}" class="tick" '
                       f'text-anchor="end">{round(vmax * (1 - frac)):g}</text>')
        for b in (1, 4, 16, 48, bmax):
            svg.append(f'<text x="{x_of(b):.1f}" y="{h - pb + 15}" '
                       f'class="tick" text-anchor="middle">{b}</text>')
        svg.append(f'<text x="{(pl + w - pr) / 2:.1f}" y="{h - 6}" '
                   f'class="tick" text-anchor="middle">blocks (log₂)</text>')

        ends = []
        for ti, tpb in enumerate(MB_TPBS):
            pts = sorted(((r["blocks"], r) for r in rows if r["tpb"] == tpb))
            poly = " ".join(f"{x_of(b):.1f},{y_of(r[metric]):.1f}"
                            for b, r in pts)
            svg.append(f'<polyline points="{poly}" class="line m{ti + 1}l"/>')
            for b, r in pts:
                unit = "Mpx/s" if metric == "mpx_s" else "ms"
                star = (" · coop max for this tpb"
                        if r.get("is_coop_max") else "")
                tip = (f"{b}×{tpb}: {r[metric]:.1f} {unit} · "
                       f"{r['model_gb_s']:.1f} GB/s "
                       f"({r['pct_of_peak']:.1f}% of peak){star}")
                svg.append(f'<circle cx="{x_of(b):.1f}" '
                           f'cy="{y_of(r[metric]):.1f}" r="4" '
                           f'class="dot m{ti + 1}" data-tip="{tip}"/>')
            last_b, last_r = pts[-1]
            ends.append([y_of(last_r[metric]), x_of(last_b), f"tpb {tpb}"])
        ends.sort()
        prev = -99.0
        for ey, ex, name in ends:
            ey = max(ey, prev + 12)
            prev = ey
            svg.append(f'<text x="{w - pr + 6}" y="{ey + 4:.1f}" class="anno" '
                       f'text-anchor="start">{name}</text>')
        svg.append("</svg>")
        html.append('<div class="panel">' + "\n".join(svg) + "</div>")
    return '<div class="panels">' + "\n".join(html) + "</div>"


def mb_bandwidth_chart():
    """Model GB/s per scene against the measured copy peak — the gap IS the
    finding (or the model's sector-blindness; ncu decides)."""
    hi = MB_PEAK * 1.04
    n = len(MB_ROWS)
    bar_h = 14
    h = n * ROW_H + 34
    span = W - GUT_L - GUT_R

    def x_of(v):
        return GUT_L + v / hi * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" '
             f'aria-label="Modeled bandwidth per scene vs measured peak">']
    tick = 0
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                     f'class="tick" text-anchor="middle">{tick:g}</text>')
        tick += 50
    px = x_of(MB_PEAK)
    parts.append(f'<line x1="{px:.1f}" y1="4" x2="{px:.1f}" '
                 f'y2="{n * ROW_H}" class="satline"/>')
    parts.append(f'<text x="{px - 6:.1f}" y="14" class="anno" '
                 f'text-anchor="end">measured D2D copy peak '
                 f'{MB_PEAK:.0f} GB/s</text>')
    for i, r in enumerate(MB_ROWS):
        y = i * ROW_H + (ROW_H - bar_h) / 2
        cy = i * ROW_H + ROW_H / 2
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label(r["scene"])}</text>')
        v = r["multi_model_gb_s"]
        bw = max(v / hi * span, 1.5)
        tip = (f"{label(r['scene'])} — model {v:.1f} GB/s = "
               f"{r['multi_pct_of_peak']:.1f}% of the measured peak · "
               f"{r['multi_mpx_s_kernel']:.0f} Mpx/s · lower-bound model")
        parts.append(f'<rect x="{GUT_L}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{bar_h}" rx="4" class="seg s6" '
                     f'data-tip="{tip}"/>')
        note = (f"{v:.1f} GB/s · {r['multi_pct_of_peak']:.1f}%"
                if v >= 0.05 else "≈0 — barrier-bound")
        parts.append(f'<text x="{GUT_L + bw + 6:.1f}" y="{cy + 4:.1f}" '
                     f'class="rowval">{note}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def mb_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>levels</th>"
            "<th>@njit ms</th><th>v2 ms</th><th>dual ms</th>"
            "<th>multi ms</th><th>bare ms</th><th>ovh %</th>"
            "<th>vs v2</th><th>vs @njit</th><th>grid</th><th>SMs</th>"
            "<th>bal CV %</th><th>GB/s</th><th>% peak</th></tr>")
    body = []
    for r in MB_ROWS:
        body.append(
            "<tr>"
            f"<td>{label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_int(r['levels'])}</td>"
            f"<td>{fmt_ms(r['njit_ms'])}</td>"
            f"<td>{fmt_ms(r['v2_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['dual_global_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['multi_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['multi_bare_kernel_ms'])}</td>"
            f"<td>{r['multi_instrumentation_overhead_pct']:+.1f}</td>"
            f"<td>{r['multi_speedup_vs_v2']:.2f}×</td>"
            f"<td>{r['multi_speedup_vs_njit']:.2f}×</td>"
            f"<td>{r['multi_blocks']}×{r['multi_tpb']}</td>"
            f"<td>{r['multi_distinct_sms']}</td>"
            f"<td>{r['multi_balance_cv_pct']:.0f}</td>"
            f"<td>{r['multi_model_gb_s']:.1f}</td>"
            f"<td>{r['multi_pct_of_peak']:.1f}</td>"
            "</tr>")
    skipped = [r for r in MB["scenes"] if "skipped" in r]
    foot = "".join(f'<tr><td>{label(r["scene"])}</td>'
                   f'<td colspan="15">skipped — {r["skipped"]}</td></tr>'
                   for r in skipped)
    return f"<table>{head}{''.join(body)}{foot}</table>"


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

# ------------------------------------------------- multi-block stat tiles
_mb_best_v2 = max(MB_ROWS, key=lambda r: r["multi_speedup_vs_v2"])
_mb_best_njit = max(MB_ROWS, key=lambda r: r["multi_speedup_vs_njit"])
_mb_best_cell = max(MB_SWEEP, key=lambda r: r["mpx_s"])
_mb_peak_pct = max(r["multi_pct_of_peak"] for r in MB_ROWS)
MB_TILES = [
    (f"{_mb_best_v2['multi_speedup_vs_v2']:.1f}× vs 1 block",
     f"the N blocks' best payoff · {label(_mb_best_v2['scene'])} "
     f"({fmt_int(_mb_best_v2['filled'])} px)"),
    (f"{_mb_best_njit['multi_speedup_vs_njit']:.1f}× vs @njit",
     f"{label(_mb_best_njit['scene'])} in "
     f"{fmt_ms(_mb_best_njit['multi_kernel_ms'])} ms vs "
     f"{fmt_ms(_mb_best_njit['njit_ms'])} ms on the CPU"),
    (f"{_mb_best_cell['mpx_s']:.0f} Mpx/s plateau",
     f"scaling stops at ~8–16 blocks · best cell "
     f"{_mb_best_cell['blocks']}×{_mb_best_cell['tpb']} "
     f"({label(_mb_best_cell['scene'])})"),
    (f"{MB_PEAK:.0f} GB/s measured peak",
     f"modeled traffic plateaus at ≈{_mb_peak_pct:.0f}% of it — a lower "
     f"bound; ncu is the arbiter"),
]
mb_tiles_html = "".join(
    f'<div class="tile"><div class="tile-v">{v}</div>'
    f'<div class="tile-l">{l}</div></div>' for v, l in MB_TILES)

balance_html, balance_js = balance_panels()

CSS = """
:root { color-scheme: light dark; }
body {
  margin: 0; padding: 24px 28px 40px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--ink);
}
.viz-root {
  --page: #f9f9f7; --surface-1: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #008300; --s3: #e87ba4; --s4: #eda100; --s5: #1baf7a;
  --s6: #8257d8;
  --m1: #b9a3ec; --m2: #9678db; --m3: #7350c9; --m4: #50309f;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #008300; --s3: #d55181; --s4: #c98500; --s5: #199e70;
    --s6: #9678db;
    --m1: #cbbcf2; --m2: #ab93e6; --m3: #8f6cd8; --m4: #7350c9;
  }
}
:root[data-theme="dark"] .viz-root {
  --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
  --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #008300; --s3: #d55181; --s4: #c98500; --s5: #199e70;
  --s6: #9678db;
  --m1: #cbbcf2; --m2: #ab93e6; --m3: #8f6cd8; --m4: #7350c9;
}
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--ink-2); font-size: 13px; margin-bottom: 20px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px; margin-bottom: 18px;
}
.card h2 { font-size: 14px; margin: 0 0 2px; }
.card .note { font-size: 12px; color: var(--ink-2); margin: 0 0 12px; }
svg { width: 100%; height: auto; display: block; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
         gap: 12px; margin-bottom: 18px; }
.tile { background: var(--surface-1); border: 1px solid var(--border);
        border-radius: 10px; padding: 14px 16px; }
.tile-v { font-size: 21px; font-weight: 650; }
.tile-l { font-size: 12px; color: var(--ink-2); margin-top: 4px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.rowline { stroke: var(--grid); stroke-width: 1; stroke-dasharray: 2 3; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.rowlab { fill: var(--ink-2); font-size: 12px; }
.rowval { fill: var(--ink-2); font-size: 11.5px; font-variant-numeric: tabular-nums; }
.paneltitle { fill: var(--ink); font-size: 12px; font-weight: 600; }
.anno { fill: var(--ink-2); font-size: 11px; }
.dot { stroke: var(--surface-1); stroke-width: 2; }
.seg { }
.s1 { fill: var(--s1); } .s2 { fill: var(--s2); } .s3 { fill: var(--s3); }
.s4 { fill: var(--s4); } .s5 { fill: var(--s5); } .s6 { fill: var(--s6); }
.m1 { fill: var(--m1); } .m2 { fill: var(--m2); }
.m3 { fill: var(--m3); } .m4 { fill: var(--m4); }
.line { fill: none; stroke-width: 2; }
.s1l { stroke: var(--s1); } .s3l { stroke: var(--s3); }
.s4l { stroke: var(--s4); } .s5l { stroke: var(--s5); }
.m1l { stroke: var(--m1); } .m2l { stroke: var(--m2); }
.m3l { stroke: var(--m3); } .m4l { stroke: var(--m4); }
.trace { fill: none; stroke: var(--s1); stroke-width: 2; }
.trace4 { fill: none; stroke: var(--s4); stroke-width: 2; }
.satline { stroke: var(--baseline); stroke-width: 1; stroke-dasharray: 3 3; }
.xhair { stroke: var(--baseline); stroke-width: 1; }
.xdot { fill: var(--s1); stroke: var(--surface-1); stroke-width: 2; }
.xdot2 { fill: var(--s4); stroke: var(--surface-1); stroke-width: 2; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px;
          color: var(--ink-2); margin-bottom: 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.chip { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
          gap: 14px; }
#tooltip {
  position: fixed; pointer-events: none; visibility: hidden; z-index: 10;
  background: var(--ink); color: var(--page); font-size: 12px;
  padding: 6px 9px; border-radius: 6px; max-width: 340px;
}
details { margin-top: 4px; }
summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12px; margin-top: 10px; width: 100%; }
th, td { text-align: right; padding: 5px 9px; border-bottom: 1px solid var(--grid);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
"""

JS = """
const tooltip = document.getElementById('tooltip');
function showTip(text, ev) {
  tooltip.textContent = text;
  tooltip.style.visibility = 'visible';
  const pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  const r = tooltip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
  tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
}
function hideTip() { tooltip.style.visibility = 'hidden'; }
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', ev => showTip(el.dataset.tip, ev));
  el.addEventListener('mouseleave', hideTip);
});
const BPANELS = __BPANELS__;
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


def legend(entries):
    return ('<div class="legend">'
            + "".join(f'<span><i class="chip" style="background:var(--{c})">'
                      f'</i>{n}</span>' for n, c in entries)
            + '</div>')


LEG5 = legend([("@njit CPU", "s2"), ("single-block v2", "s1"),
               ("split", "s3"), ("global", "s4"), ("dirsplit", "s5")])
LEG3 = legend([("split", "s3"), ("global", "s4"), ("dirsplit", "s5")])
LEG_BAL = legend([("block 0", "s1"), ("block 1", "s4")])
LEG_SBS = legend([("@njit CPU", "s2"), ("pure Python", "s3"),
                  ("v1 ring kernel", "s4"), ("v2 spill kernel", "s1")])
LEG_MB = legend([("@njit CPU", "s2"), ("single-block v2", "s1"),
                 ("dual global (2 blocks)", "s4"),
                 (f"multi ({MB_ROWS[0]['multi_blocks']} blocks)", "s6")])
LEG_MB_SPD = legend([("vs single-block v2", "s1"), ("vs dual global", "s4"),
                     ("vs @njit CPU", "s2")])
LEG_MB_TPB = legend([(f"tpb {t}", f"m{i + 1}")
                     for i, t in enumerate(MB_TPBS)])

html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flood fill — 1 → 2 → N blocks, benchmarked</title>
<style>{CSS}</style></head>
<body><div class="viz-root">
<h1>BFS flood fill — 1 block → 2 blocks → N blocks, benchmarked</h1>
<div class="sub">{DUAL['device']} · {DUAL['sm_count']} SMs · newest stage
first: the N-block global-queue kernel (blocks=None → cooperative max),
then the dual-block partitionings, then the single-block stage ·
4-connectivity · tpb=256 unless noted · placement observed via %smid</div>

<div class="tiles">{mb_tiles_html}</div>

<div class="card">
<h2>N blocks — runtime per scene</h2>
<p class="note">Log scale — each decade gridline is 10×. The lineage on
one chart: CPU (green) → one block (blue) → two blocks (yellow) → the
cooperative maximum, {MB_ROWS[0]['multi_blocks']} blocks at tpb=256
(violet). The gap widens with blob size to
{_mb_best_v2['multi_speedup_vs_v2']:.1f}× vs one block; on the serpentine
the ordering inverts — more blocks means a costlier barrier and nothing to
feed. Hover any dot; exact numbers in the table below.</p>
{LEG_MB}
{mb_runtime_chart()}
</div>

<div class="card">
<h2>What N blocks buy — speedup vs each baseline</h2>
<p class="note">Log scale. Each dot color is the baseline being compared
against. Right of the parity line the N blocks win; the payoff vs the CPU
reaches {_mb_best_njit['multi_speedup_vs_njit']:.1f}× at 64M px. Left of
it: the tiny scene (barrier tax beats 16K pixels) and the serpentine
(0.0007× vs the CPU — 65,792 grid-wide barriers at ~2 µs each ARE the
runtime).</p>
{LEG_MB_SPD}
{mb_speedup_chart()}
</div>

<div class="card">
<h2>The centerpiece — blocks × threads-per-block sweep</h2>
<p class="note">Throughput vs block count (log₂ axis), one line per
threads-per-block; line ends mark each tpb's cooperative-capacity limit
(registers: ~104/thread cap every configuration at 12,288 total threads =
512 per SM). Near-linear scaling to ~8–16 blocks, then the plateau: 96 or
192 blocks move nothing, and tpb=512 anti-scales past ~8 blocks — at
equal thread counts many small blocks beat few big ones (best cell:
{_mb_best_cell['blocks']}×{_mb_best_cell['tpb']}). The serpentine panel is
in kernel ms: flat everywhere — no configuration helps a shape that
starves every block between barriers.</p>
{LEG_MB_TPB}
{mb_sweep_panels()}
</div>

<div class="card">
<h2>Bandwidth — the modeled traffic vs the measured ceiling</h2>
<p class="note">Violet bars: algorithmic bytes moved (from each run's
exactly-once counters, ≈61 B/pixel) ÷ kernel time. The dashed line is the
MEASURED device-to-device copy peak — the honest ceiling, not a spec
sheet. The plateau tops out at ≈{_mb_peak_pct:.0f}% of it <i>by a
lower-bound model</i>: 32 B DRAM sectors can inflate the real traffic of
scattered 3–4 B accesses several-fold, which would put the true figure
near the ceiling — consistent with bandwidth saturation, but only ncu can
close that attribution gap.</p>
{mb_bandwidth_chart()}
</div>

<div class="card">
<h2>All numbers — N-block stage</h2>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{mb_table()}</div></details>
</div>

<div class="sub" style="margin-top:26px">Below: the dual-block stage
(previous chapter — how two blocks should share one BFS, and where they
should live), then the single-block stage.</div>

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
<h2>Appended: the single-block stage (previous chapter)</h2>
<p class="note">The stage these dual kernels are measured against — from
its own benchmark run. The v2 spill kernel (blue here and everywhere
above) is the baseline of the speedup chart; v1 ring is the pure
shared-memory kernel that trips on oversized frontiers. Pure Python was
skipped above 2M px.</p>
{LEG_SBS}
{sbs_chart()}
<details><summary>Single-block per-scene results table</summary>
<div class="tablewrap">{sbs_table()}</div></details>
</div>

<div class="card">
<h2>All numbers — dual-block stage</h2>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{dual_table()}</div></details>
</div>

<div id="tooltip"></div>
</div>
<script>{JS}</script>
</body></html>
"""

with open(OUT_PATH, "w") as f:
    f.write(html)
print(f"rendered {os.path.basename(MB_PATH)} + {os.path.basename(DUAL_PATH)}"
      f" + {os.path.basename(SBS_PATH)} -> {OUT_PATH} ({len(html):,} bytes)")
