"""
Generate a self-contained HTML dashboard from benchmark.py's JSON output.

Renders the runtime comparison (log-scale dot plot, v1 ring and v2 spill
kernels vs the CPU baselines), the peak-queue-occupancy-vs-ring-capacity
chart, the GPU end-to-end time decomposition (100% stacked), the per-level
frontier traces, the per-level warp/thread activity traces (lanes busy vs
lanes woken), the threads-per-block sweep, and the full results table —
with hover tooltips and light/dark theming, no external dependencies.

Usage:
    uv run python -m flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.visualize [results.json]

With no argument, the newest single_block_shared_*.json in
results/ch01_gpu_1blob_1block/benchmark_results/ is used. Output:
results/ch01_gpu_1blob_1block/benchmark_results/single_block_benchmark.html
(overwritten on each run — the timestamped JSON/CSV remain the durable
record).
"""
import glob
import json
import math
import os
import sys

from ....shared import results_paths
from ....shared.viz import (
    fmt_ms, fmt_int, decimate, legend, chain_strip, CHAIN_NOTE, log_dot_plot,
)

RESULTS_DIR = results_paths.results_dir("ch01_gpu_1blob_1block", "benchmark_results")

# sys.argv is only this module's own override when it's the script being
# run directly -- when imported as a dependency (by ch02's renderer or the
# dashboard assembler), argv belongs to whatever positional override THEY
# accept (e.g. ch02's own dual_block.json path), not to this chapter.
if __name__ == "__main__" and len(sys.argv) > 1:
    JSON_PATH = sys.argv[1]
else:
    candidates = sorted(glob.glob(
        os.path.join(RESULTS_DIR, "single_block_shared_*.json")))
    if not candidates:
        sys.exit("no benchmark JSON found — run benchmark.py first")
    JSON_PATH = candidates[-1]
OUT_PATH = os.path.join(RESULTS_DIR, "single_block_benchmark.html")

with open(JSON_PATH) as f:
    DATA = json.load(f)

SCENE_LABELS = {
    "sq_256_center": "square 256² · center seed",
    "sq_512_center": "square 512² · center seed",
    "sq_1024_center": "square 1024² · center seed",
    "sq_2000_center": "square 2000² · center seed",
    "sq_4000_corner": "square 4000² · corner seed",
    "serpentine_256": "serpentine 256²",
    "disk_1024": "disk r=480",
    "sq_2600_full_center": "square 2600² full · center seed",
    "sq_4000_center": "square 4000² · center seed",
    "sq_5000_center": "square 5000² · center seed",
    "sq_6000_center": "square 6000² · center seed",
}
ROWS = DATA["scenes"]
SWEEP = DATA["tpb_sweep"]


def scene_label(name):
    return SCENE_LABELS.get(name, name)


# ---------------------------------------------------------------- dot plot
DOT_W, ROW_H, GUT_L, GUT_R, AX_H = 860, 40, 200, 30, 26
DOT_SERIES = [("gpu_kernel_ms", "GPU v1 ring kernel", "s1"),
              ("spill_kernel_ms", "GPU v2 spill kernel", "s4"),
              ("njit_ms", "@njit CPU", "s2"),
              ("pure_ms", "pure Python", "s3")]
LOG_MIN = 0.1
_max_ms = max(row[key] for row in ROWS for key, _, _ in DOT_SERIES
              if row.get(key))
LOG_MAX = 10 ** math.ceil(math.log10(_max_ms))


def log_x(ms):
    frac = (math.log10(ms) - math.log10(LOG_MIN)) / (
        math.log10(LOG_MAX) - math.log10(LOG_MIN))
    return GUT_L + frac * (DOT_W - GUT_L - GUT_R)


def dot_plot():
    n = len(ROWS)
    h = n * ROW_H + AX_H + 8
    parts = [f'<svg viewBox="0 0 {DOT_W} {h}" role="img" '
             f'aria-label="Runtime per scene, log scale">']
    # gridlines + ticks at decades (no text on the last tick — the axis-unit
    # label owns the right edge)
    tick = LOG_MIN
    while tick <= LOG_MAX:
        x = log_x(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        if x < DOT_W - GUT_R - 70:
            parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                         f'class="tick" text-anchor="middle">{tick:g}</text>')
        tick *= 10
    parts.append(f'<text x="{DOT_W - GUT_R}" y="{n * ROW_H + 18}" class="tick" '
                 f'text-anchor="end">ms (log)</text>')
    for i, row in enumerate(ROWS):
        cy = i * ROW_H + ROW_H / 2
        label = scene_label(row["scene"])
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{DOT_W - GUT_R}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label}</text>')
        for key, name, cls in DOT_SERIES:
            v = row.get(key)
            if v is None:
                continue
            x = log_x(v)
            tip = (f"{label} — {name}: {fmt_ms(v)} ms "
                   f"({fmt_int(row['filled'])} px)")
            parts.append(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="5.5" '
                         f'class="dot {cls}" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------- stacked decomposition
STK_W = 860


def stacked():
    n = len(ROWS)
    bar_h, row_h = 20, 40
    h = n * row_h + 30
    gut_l = 230  # wider than the dot plots: labels carry a " · v2" suffix
    # kernel reuses the GPU-blue slot from the runtime chart (color follows
    # the entity); this exact adjacency order was validated in both modes.
    segs = [("alloc_ms", "alloc", "s2", "w"),
            ("h2d_ms", "H2D copy", "s3", "k"),
            ("kernel_ms", "kernel", "s1", "w"),
            ("d2h_ms", "D2H copy", "s4", "k")]
    parts = [f'<svg viewBox="0 0 {STK_W} {h}" role="img" '
             f'aria-label="GPU time decomposition, share of total">',
             '<defs>']
    span = STK_W - gut_l - 90
    for i in range(n):
        y = i * row_h + (row_h - bar_h) / 2
        parts.append(f'<clipPath id="rc{i}"><rect x="{gut_l}" y="{y:.1f}" '
                     f'width="{span}" height="{bar_h}" rx="4"/></clipPath>')
    parts.append('</defs>')
    for i, row in enumerate(ROWS):
        label = scene_label(row["scene"])
        # Ring decomposition where v1 completed; spill decomposition where
        # only the v2 kernel could run the scene.
        if row.get("gpu_total_ms"):
            prefix, total = "gpu_", row["gpu_total_ms"]
        else:
            prefix, total = "spill_", row["spill_total_ms"]
            label += " · v2"
        y = i * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2 + 4
        parts.append(f'<text x="{gut_l - 10}" y="{cy:.1f}" class="rowlab" '
                     f'text-anchor="end">{label}</text>')
        x = gut_l
        parts.append(f'<g clip-path="url(#rc{i})">')
        for key, name, cls, ink in segs:
            v = row[prefix + key]
            w = v / total * span
            pct = v / total * 100
            tip = f"{label} — {name}: {fmt_ms(v)} ms ({pct:.0f}%)"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w - 2, 0.5):.1f}" '
                         f'height="{bar_h}" class="seg {cls}" data-tip="{tip}"/>')
            if w > 60:
                parts.append(f'<text x="{x + w / 2:.1f}" y="{cy:.1f}" '
                             f'class="seglab {ink}" text-anchor="middle">{pct:.0f}%</text>')
            x += w
        parts.append('</g>')
        parts.append(f'<text x="{gut_l + span + 8}" y="{cy:.1f}" class="rowval">'
                     f'{fmt_ms(total)} ms</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------- occupancy vs ring capacity
def occ_chart():
    """Log-scale dot per scene: peak queue occupancy vs the 8192-slot ring.

    Dots left of the dashed capacity line fit v1's pure-shared ring; dots
    right of it are only possible because the v2 spill tier absorbed the
    excess — the tooltip carries the spilled-pixel count.
    """
    n = len(ROWS)
    h = n * ROW_H + AX_H + 8
    cap = ROWS[0]["ring_capacity"]
    omax = max(r["peak_occupancy"] for r in ROWS)
    lo, hi = 1.0, 10 ** math.ceil(math.log10(omax))
    span = DOT_W - GUT_L - GUT_R

    def ox(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {DOT_W} {h}" role="img" '
             f'aria-label="Peak queue occupancy per scene vs ring capacity">']
    tick = lo
    while tick <= hi:
        x = ox(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        if x < DOT_W - GUT_R - 110:
            parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" '
                         f'class="tick" text-anchor="middle">{fmt_int(int(tick))}</text>')
        tick *= 10
    parts.append(f'<text x="{DOT_W - GUT_R}" y="{n * ROW_H + 18}" class="tick" '
                 f'text-anchor="end">queue slots (log)</text>')
    cx = ox(cap)
    parts.append(f'<line x1="{cx:.1f}" y1="4" x2="{cx:.1f}" '
                 f'y2="{n * ROW_H}" class="satline"/>')
    parts.append(f'<text x="{cx + 6:.1f}" y="16" class="anno" '
                 f'text-anchor="start">shared ring capacity {fmt_int(cap)}</text>')
    for i, row in enumerate(ROWS):
        cy = i * ROW_H + ROW_H / 2
        label = scene_label(row["scene"])
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{DOT_W - GUT_R}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label}</text>')
        occ = row["peak_occupancy"]
        spilled = row.get("spilled_px", 0)
        if spilled:
            tip = (f"{label} — peak occupancy {fmt_int(occ)} of {fmt_int(cap)} "
                   f"ring slots: v1 trips; v2 spilled {fmt_int(spilled)} px "
                   f"({row['spill_pct']:.0f}% of the blob)")
        else:
            tip = (f"{label} — peak occupancy {fmt_int(occ)} of {fmt_int(cap)}: "
                   f"fits the shared ring, spill tier untouched")
        parts.append(f'<circle cx="{ox(occ):.1f}" cy="{cy:.1f}" r="5.5" '
                     f'class="dot s1" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------ frontier panels
PANEL_SCENES = ["sq_2000_center", "sq_4000_corner", "disk_1024",
                "serpentine_256", "sq_6000_center"]


def panels():
    html, js = [], {}
    for row in ROWS:
        if row["scene"] not in PANEL_SCENES:
            continue
        name = row["scene"]
        label = scene_label(name)
        sizes = row["level_sizes"]
        xs, ys = decimate(sizes)
        w, h, pl, pr, pt, pb = 420, 170, 52, 14, 26, 26
        peak = max(ys)
        n_levels = row["levels"]
        px = lambda x: pl + x / max(n_levels - 1, 1) * (w - pl - pr)
        py = lambda y: pt + (1 - y / peak) * (h - pt - pb)
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        peak_i = ys.index(peak)
        peak_x, peak_y = px(xs[peak_i]), py(peak)
        svg = [f'<svg viewBox="0 0 {w} {h}" class="panel-svg" id="p_{name}" '
               f'role="img" aria-label="Frontier size per BFS level, {label}">']
        for frac in (0, 0.5, 1):
            yy = pt + frac * (h - pt - pb)
            v = peak * (1 - frac)
            val = fmt_int(round(v)) if abs(v - round(v)) < 1e-9 else f"{v:g}"
            svg.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w - pr}" '
                       f'y2="{yy:.1f}" class="grid"/>')
            svg.append(f'<text x="{pl - 6}" y="{yy + 4:.1f}" class="tick" '
                       f'text-anchor="end">{val}</text>')
        svg.append(f'<text x="{pl}" y="14" class="paneltitle">{label} '
                   f'— {fmt_int(n_levels)} levels</text>')
        svg.append(f'<text x="{w - pr}" y="{h - 8}" class="tick" '
                   f'text-anchor="end">BFS level →</text>')
        svg.append(f'<polyline points="{pts}" class="trace"/>')
        if peak > min(ys):  # a flat trace has no peak worth marking
            svg.append(f'<circle cx="{peak_x:.1f}" cy="{peak_y:.1f}" r="4" '
                       f'class="dot s1"/>')
            anchor = "start" if peak_x < w * 0.6 else "end"
            dx = 8 if anchor == "start" else -8
            svg.append(f'<text x="{peak_x + dx:.1f}" y="{max(peak_y - 8, pt + 4):.1f}" '
                       f'class="anno" text-anchor="{anchor}">peak {fmt_int(peak)}</text>')
        svg.append(f'<line class="xhair" id="xh_{name}" x1="0" x2="0" '
                   f'y1="{pt}" y2="{h - pb}" visibility="hidden"/>')
        svg.append(f'<circle class="xdot" id="xd_{name}" r="4" visibility="hidden"/>')
        svg.append(f'<rect x="{pl}" y="{pt}" width="{w - pl - pr}" '
                   f'height="{h - pt - pb}" fill="transparent" '
                   f'class="hover-capture" data-panel="{name}"/>')
        svg.append("</svg>")
        html.append('<div class="panel">' + "\n".join(svg) + "</div>")
        js[name] = {"xs": xs, "ys": ys, "w": w, "h": h, "pl": pl, "pr": pr,
                    "pt": pt, "pb": pb, "peak": peak, "n": n_levels}
    return "\n".join(html), json.dumps(js)


# ------------------------------------------------ warp/thread activity panels
def activity_panels():
    """Per-level lanes-busy vs lanes-woken traces, one panel per PANEL_SCENE.

    Both series share the lane unit (0..tpb) so one axis carries both:
    busy = min(frontier, tpb) threads have a pixel; woken = engaged warps
    x 32 lanes are burning an issue slot because a warp with any active
    thread runs all 32 lanes. The vertical gap between them is pure waste.
    """
    html, js = [], {}
    for row in ROWS:
        if row["scene"] not in PANEL_SCENES:
            continue
        name = row["scene"]
        label = scene_label(name)
        cap = row["threads_per_block"]
        n_warps = cap // 32
        sizes = row["level_sizes"]
        xs, sampled = decimate(sizes)
        busy = [min(s, cap) for s in sampled]
        woken = [-(-min(s, cap) // 32) * 32 for s in sampled]
        n_levels = row["levels"]
        sat = next((i for i, s in enumerate(sizes) if s >= cap), None)
        peak_busy = min(max(sizes), cap)
        w, h, pl, pr, pt, pb = 420, 170, 52, 46, 26, 26
        px = lambda x: pl + x / max(n_levels - 1, 1) * (w - pl - pr)
        py = lambda v: pt + (1 - v / cap) * (h - pt - pb)
        svg = [f'<svg viewBox="0 0 {w} {h}" class="panel-svg" id="a_{name}" '
               f'role="img" aria-label="Lanes busy and lanes woken per BFS '
               f'level, {label}">']
        for frac in (0, 0.5, 1):
            yy = pt + frac * (h - pt - pb)
            lanes = round(cap * (1 - frac))
            svg.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w - pr}" '
                       f'y2="{yy:.1f}" class="grid"/>')
            svg.append(f'<text x="{pl - 6}" y="{yy + 4:.1f}" class="tick" '
                       f'text-anchor="end">{lanes}</text>')
            svg.append(f'<text x="{w - pr + 6}" y="{yy + 4:.1f}" class="tick" '
                       f'text-anchor="start">{lanes // 32}</text>')
        svg.append(f'<text x="{w - pr + 6}" y="{pt - 6}" class="tick" '
                   f'text-anchor="start">warps</text>')
        svg.append(f'<text x="{pl}" y="14" class="paneltitle">{label}</text>')
        svg.append(f'<text x="{w - pr}" y="{h - 8}" class="tick" '
                   f'text-anchor="end">BFS level →</text>')
        if sat is not None:
            sx = px(sat)
            svg.append(f'<line x1="{sx:.1f}" y1="{pt}" x2="{sx:.1f}" '
                       f'y2="{h - pb}" class="satline"/>')
            svg.append(f'<text x="{sx + 6:.1f}" y="{pt + 12}" class="anno" '
                       f'text-anchor="start">all {n_warps} warps have work '
                       f'from level {fmt_int(sat)}</text>')
        else:
            svg.append(f'<text x="{pl + 4}" y="{pt + 12}" class="anno" '
                       f'text-anchor="start">peak {fmt_int(peak_busy)} of '
                       f'{cap} lanes busy — {-(-peak_busy // 32)} of '
                       f'{n_warps} warps</text>')
        wok_pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, woken))
        busy_pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, busy))
        svg.append(f'<polyline points="{wok_pts}" class="trace4"/>')
        svg.append(f'<polyline points="{busy_pts}" class="trace"/>')
        busy_med = sorted(busy)[len(busy) // 2]
        wok_med = sorted(woken)[len(woken) // 2]
        if py(busy_med) - py(wok_med) >= 12:  # visibly separated → direct labels
            lx = pl + 0.4 * (w - pl - pr)
            svg.append(f'<text x="{lx:.1f}" y="{py(wok_med) - 6:.1f}" '
                       f'class="anno" text-anchor="middle">lanes woken</text>')
            svg.append(f'<text x="{lx:.1f}" y="{py(busy_med) + 14:.1f}" '
                       f'class="anno" text-anchor="middle">lanes busy</text>')
        svg.append(f'<line class="xhair" id="axh_{name}" x1="0" x2="0" '
                   f'y1="{pt}" y2="{h - pb}" visibility="hidden"/>')
        svg.append(f'<circle class="xdot2" id="axd2_{name}" r="4" visibility="hidden"/>')
        svg.append(f'<circle class="xdot" id="axd1_{name}" r="4" visibility="hidden"/>')
        svg.append(f'<rect x="{pl}" y="{pt}" width="{w - pl - pr}" '
                   f'height="{h - pt - pb}" fill="transparent" '
                   f'class="ahover-capture" data-panel="{name}"/>')
        svg.append("</svg>")
        html.append('<div class="panel">' + "\n".join(svg) + "</div>")
        js[name] = {"xs": xs, "fr": sampled, "busy": busy, "wok": woken,
                    "w": w, "h": h, "pl": pl, "pr": pr, "pt": pt, "pb": pb,
                    "cap": cap, "n": n_levels}
    return "\n".join(html), json.dumps(js)


# --------------------------------------------------------------- tpb sweep
def sweep_chart():
    w, h, pl, pr, pt, pb = 560, 220, 56, 16, 20, 40
    peak = max(r["mpx_s"] for r in SWEEP)
    top = math.ceil(peak / 10) * 10
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" '
             f'aria-label="Throughput by threads per block">']
    for frac in (0, 0.5, 1):
        yy = pt + frac * (h - pt - pb)
        val = fmt_int(round(top * (1 - frac)))
        parts.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w - pr}" '
                     f'y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pl - 6}" y="{yy + 4:.1f}" class="tick" '
                     f'text-anchor="end">{val}</text>')
    parts.append(f'<text x="{pl - 40}" y="{pt - 6}" class="tick">Mpx/s</text>')
    n = len(SWEEP)
    slot = (w - pl - pr) / n
    bar_w = min(slot * 0.55, 56)
    for i, r in enumerate(SWEEP):
        x = pl + i * slot + (slot - bar_w) / 2
        bh = r["mpx_s"] / top * (h - pt - pb)
        y = h - pb - bh
        tip = (f"tpb {r['tpb']}: {r['mpx_s']:.1f} Mpx/s · "
               f"kernel {fmt_ms(r['kernel_ms'])} ms · "
               f"thread util {r['thread_util_pct']:.0f}% · "
               f"occupancy {r['occupancy_pct']:.0f}%")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                     f'height="{bh:.1f}" rx="4" class="seg s1" data-tip="{tip}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" '
                     f'class="anno" text-anchor="middle">{r["mpx_s"]:.0f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{h - pb + 16}" '
                     f'class="tick" text-anchor="middle">{r["tpb"]}</text>')
    parts.append(f'<text x="{(pl + w - pr) / 2:.1f}" y="{h - 4}" class="tick" '
                 f'text-anchor="middle">threads per block · square 2000² scene</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------------ table
def fmt_x(v):
    return f"{v:.2f}×" if v is not None else "—"


def table():
    head = ("<tr><th>scene</th><th>filled px</th><th>levels</th>"
            "<th>peak occ.</th><th>ring ms</th><th>spill ms</th>"
            "<th>spilled px</th><th>@njit ms</th><th>pure ms</th>"
            "<th>ring vs @njit</th><th>spill vs @njit</th>"
            "<th>thread util %</th><th>disc. redund.</th></tr>")
    body = []
    for r in ROWS:
        body.append(
            "<tr>"
            f"<td>{scene_label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_int(r['levels'])}</td>"
            f"<td>{fmt_int(r['peak_occupancy'])}</td>"
            f"<td>{fmt_ms(r['gpu_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['spill_kernel_ms'])}</td>"
            f"<td>{fmt_int(r['spilled_px'])}</td>"
            f"<td>{fmt_ms(r['njit_ms'])}</td>"
            f"<td>{fmt_ms(r['pure_ms'])}</td>"
            f"<td>{fmt_x(r['speedup_kernel_vs_njit'])}</td>"
            f"<td>{fmt_x(r['speedup_spill_vs_njit'])}</td>"
            f"<td>{r['thread_util_pct']:.1f}</td>"
            f"<td>{r['discovery_redundancy']:.2f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


# -------------------------------------------------------------- stat tiles
def _best_kernel_mpx(r):
    return max(r.get("gpu_mpx_s_kernel") or 0.0, r["spill_mpx_s_kernel"])


def _best_speedup(r):
    return max(r.get("speedup_kernel_vs_njit") or 0.0, r["speedup_spill_vs_njit"])


best_thr = max(ROWS, key=_best_kernel_mpx)
best_speed = max(ROWS, key=_best_speedup)
worst_speed = min(ROWS, key=_best_speedup)
big_row = max(ROWS, key=lambda r: r["filled"])
TILES = [
    (f"{_best_kernel_mpx(best_thr):.1f} Mpx/s",
     f"peak kernel throughput · {scene_label(best_thr['scene'])}"),
    (f"{_best_speedup(best_speed):.2f}× vs @njit",
     f"best GPU win · {scene_label(best_speed['scene'])} "
     f"({fmt_int(best_speed['filled'])} px, v2 spill kernel)"),
    (f"{_best_speedup(worst_speed):.3f}× vs @njit",
     f"worst case · {scene_label(worst_speed['scene'])}: "
     f"{fmt_int(worst_speed['levels'])} tiny frontiers"),
    (f"{fmt_int(big_row['filled'])} px",
     f"biggest blob · impossible for v1 (needs {fmt_int(big_row['peak_occupancy'])} "
     f"queue slots) — v2 spilled {fmt_int(big_row['spilled_px'])} px"),
]
tiles_html = "".join(
    f'<div class="tile"><div class="tile-v">{v}</div>'
    f'<div class="tile-l">{l}</div></div>' for v, l in TILES)

panels_html, panels_js = panels()
act_html, act_js = activity_panels()

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
  --s1: #2a78d6; --s2: #008300; --s3: #e87ba4; --s4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #008300; --s3: #d55181; --s4: #c98500;
  }
}
:root[data-theme="dark"] .viz-root {
  --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
  --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #008300; --s3: #d55181; --s4: #c98500;
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
.tile-v { font-size: 22px; font-weight: 650; }
.tile-l { font-size: 12px; color: var(--ink-2); margin-top: 4px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.rowline { stroke: var(--grid); stroke-width: 1; stroke-dasharray: 2 3; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.rowlab { fill: var(--ink-2); font-size: 12px; }
.rowval { fill: var(--ink-2); font-size: 11.5px; font-variant-numeric: tabular-nums; }
.seglab { font-size: 10.5px; }
.seglab.w { fill: #ffffff; }
.seglab.k { fill: #0b0b0b; }
.paneltitle { fill: var(--ink); font-size: 12px; font-weight: 600; }
.anno { fill: var(--ink-2); font-size: 11px; }
.dot { stroke: var(--surface-1); stroke-width: 2; }
.s1 { fill: var(--s1); } .s2 { fill: var(--s2); }
.s3 { fill: var(--s3); } .s4 { fill: var(--s4); }
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
  padding: 6px 9px; border-radius: 6px; max-width: 320px;
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
const PANELS = __PANELS__;
document.querySelectorAll('.hover-capture').forEach(el => {
  const name = el.dataset.panel, p = PANELS[name];
  const svg = document.getElementById('p_' + name);
  const xh = document.getElementById('xh_' + name);
  const xd = document.getElementById('xd_' + name);
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
    const cx = toX(p.xs[i]), cy = toY(p.ys[i]);
    xh.setAttribute('x1', cx); xh.setAttribute('x2', cx);
    xh.setAttribute('visibility', 'visible');
    xd.setAttribute('cx', cx); xd.setAttribute('cy', cy);
    xd.setAttribute('visibility', 'visible');
    showTip('level ' + p.xs[i].toLocaleString() + ' \\u2014 frontier ' +
            p.ys[i].toLocaleString() + ' px', ev);
  });
  el.addEventListener('mouseleave', () => {
    xh.setAttribute('visibility', 'hidden');
    xd.setAttribute('visibility', 'hidden');
    hideTip();
  });
});
const APANELS = __APANELS__;
document.querySelectorAll('.ahover-capture').forEach(el => {
  const name = el.dataset.panel, p = APANELS[name];
  const svg = document.getElementById('a_' + name);
  const xh = document.getElementById('axh_' + name);
  const d1 = document.getElementById('axd1_' + name);
  const d2 = document.getElementById('axd2_' + name);
  const toX = lv => p.pl + lv / Math.max(p.n - 1, 1) * (p.w - p.pl - p.pr);
  const toY = v => p.pt + (1 - v / p.cap) * (p.h - p.pt - p.pb);
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
    d2.setAttribute('cx', cx); d2.setAttribute('cy', toY(p.wok[i]));
    d2.setAttribute('visibility', 'visible');
    d1.setAttribute('cx', cx); d1.setAttribute('cy', toY(p.busy[i]));
    d1.setAttribute('visibility', 'visible');
    showTip('level ' + p.xs[i].toLocaleString() + ' \\u2014 frontier ' +
            p.fr[i].toLocaleString() + ' px \\u00b7 ' + p.busy[i] + ' of ' +
            p.cap + ' lanes busy \\u00b7 ' + (p.wok[i] / 32) + ' of ' +
            (p.cap / 32) + ' warps woken', ev);
  });
  el.addEventListener('mouseleave', () => {
    xh.setAttribute('visibility', 'hidden');
    d1.setAttribute('visibility', 'hidden');
    d2.setAttribute('visibility', 'hidden');
    hideTip();
  });
});
""".replace("__PANELS__", panels_js).replace("__APANELS__", act_js)

legend3 = ('<div class="legend">'
           '<span><i class="chip" style="background:var(--s1)"></i>GPU v1 ring kernel (median of 5)</span>'
           '<span><i class="chip" style="background:var(--s4)"></i>GPU v2 spill kernel</span>'
           '<span><i class="chip" style="background:var(--s2)"></i>@njit CPU</span>'
           '<span><i class="chip" style="background:var(--s3)"></i>pure Python</span>'
           '</div>')
legend4 = ('<div class="legend">'
           '<span><i class="chip" style="background:var(--s2)"></i>alloc</span>'
           '<span><i class="chip" style="background:var(--s3)"></i>H2D copy</span>'
           '<span><i class="chip" style="background:var(--s1)"></i>kernel</span>'
           '<span><i class="chip" style="background:var(--s4)"></i>D2H copy</span>'
           '</div>')
legend_act = ('<div class="legend">'
              '<span><i class="chip" style="background:var(--s1)"></i>'
              'lanes busy — threads with a pixel to process</span>'
              '<span><i class="chip" style="background:var(--s4)"></i>'
              'lanes woken — engaged warps × 32</span>'
              '</div>')


# ---------------------------------------------------------- dashboard contract
# Exports consumed by the assembled cross-chapter dashboard: SBS_ROWS/SWEEP
# are this chapter's own ROWS/SWEEP under the name the dashboard expects;
# sbs_chart/sbs_table are a simpler, dashboard-styled rendering of the same
# data as dot_plot()/table() above (kept separate -- this chapter's own
# standalone page keeps its richer charts unchanged).
SBS_ROWS = ROWS
SBS_SWEEP = SWEEP

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


def sbs_chart():
    series = [(lambda r: r["njit_ms"], "@njit CPU", "s2"),
              (lambda r: r.get("pure_ms"), "pure Python", "s3"),
              (lambda r: r.get("gpu_kernel_ms"), "v1 ring kernel", "s4"),
              (lambda r: r["spill_kernel_ms"], "v2 spill kernel", "s1")]
    return log_dot_plot(SBS_ROWS, series,
                        "Single-block stage runtime per scene, log scale",
                        lambda r: SBS_LABELS.get(r["scene"], r["scene"]))


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


_sbs_v2_36m = next(r for r in SBS_ROWS if r["scene"] == "sq_6000_center")

CHAIN_11 = [("CPU (@njit)", _sbs_v2_36m["njit_ms"]),
           ("v2 spill kernel", _sbs_v2_36m["spill_kernel_ms"])]


LEG_SBS = legend([("@njit CPU", "s2"), ("pure Python", "s3"),
                  ("v1 ring kernel", "s4"), ("v2 spill kernel", "s1")])


SECTION_1_1 = f"""<h3 class="subsection-h">1.1 Single block</h3>
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
</div>"""

# This chapter contributes no project tile.
TILES = []

def main():
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Single-block flood fill — benchmark</title>
<style>{CSS}</style></head>
<body><div class="viz-root">
<h1>Single-block shared-memory BFS flood fill — benchmark</h1>
<div class="sub">{DATA['device']} · {DATA['sm_count']} SMs · one block = 1 SM
(4.2% of the GPU) by design · 4-connectivity · tpb=256 unless noted ·
v1 "ring" = pure shared-memory queue, v2 "spill" = two-tier queue
(shared ring + global spill) with warp-aggregated enqueue</div>

<div class="tiles">{tiles_html}</div>

<div class="card">
<h2>Runtime per scene</h2>
<p class="note">Log scale — each decade gridline is 10×. Missing blue dots
are scenes v1 cannot run (ring overflow): only the v2 spill kernel has a
time there, and its lead over @njit grows with blob size. Pure Python was
skipped above 2M px.</p>
{legend3}
{dot_plot()}
</div>

<div class="card">
<h2>Peak queue occupancy vs the shared ring</h2>
<p class="note">How much frontier each scene's BFS actually queued
(spanning two adjacent levels), against the 8192-slot shared ring. Scenes
right of the dashed line are exactly the ones where v1 trips its overflow
tripwire; v2 routes the excess through the global spill tier instead —
hover for spilled-pixel counts.</p>
{occ_chart()}
</div>

<div class="card">
<h2>Where the GPU's end-to-end time goes</h2>
<p class="note">Share of total wall time per scene (100% = the value at the
right). Rows marked "· v2" show the spill kernel (v1 cannot run those
scenes); others show the v1 ring kernel. On small scenes allocation +
transfers dominate — the kernel is not the bottleneck until the image is
large.</p>
{legend4}
{stacked()}
</div>

<div class="card">
<h2>Frontier size over BFS levels</h2>
<p class="note">The number of pixels available for parallel work at each
step. Wide diamonds (squares, disk) keep 256 threads busy; the serpentine's
~1&#8209;pixel frontier starves them — hover for exact values.</p>
<div class="panels">{panels_html}</div>
</div>

<div class="card">
<h2>Warp &amp; thread activity over BFS levels</h2>
<p class="note">Allocation never changes: the single block's 256 threads =
8 warps sit resident on their SM from launch to exit. What varies per level
is how many have work. Blue is lanes with a pixel to process
(min(frontier,&nbsp;256)); yellow is lanes woken because their warp has at
least one active thread (engaged warps&nbsp;×&nbsp;32) — a warp always runs
all 32 lanes together. The vertical gap between yellow and blue is lanes
burning issue slots with nothing to do. Squares and the disk saturate all
8 warps within the first levels (dashed marker) and the lines fuse at the
256 ceiling; the serpentine wakes one full warp for a 1-pixel frontier, so
31 of 32 lanes idle for all 32,896 levels. Right axis: warps. Hover for
exact values.</p>
{legend_act}
<div class="panels">{act_html}</div>
</div>

<div class="card">
<h2>Throughput vs threads per block</h2>
<p class="note">Kernel-only Mpx/s on the square 2000² scene (median of
5). Wide frontiers reward bigger blocks; hover shows thread utilization and
theoretical occupancy.</p>
{sweep_chart()}
</div>

<div class="card">
<h2>All numbers</h2>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{table()}</div></details>
</div>

<div id="tooltip"></div>
</div>
<script>{JS}</script>
</body></html>
"""

    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"rendered {os.path.basename(JSON_PATH)} -> {OUT_PATH} ({len(html):,} bytes)")



if __name__ == "__main__":
    main()
