"""
Chapter 3 dashboard renderer: N-block runtime/speedup/sweep/bandwidth
(section 1.3), the 4-vs-8 connectivity experiment (1.4), and the
per-barrier work experiments -- radius-2 and warp-coop (1.5).

Extracted from the dashboard monolith
(chapters/ch02_gpu_1blob_2block/benchmarks/visualize.py). Self-contained:
loads its own multi_block_*.json (required) and neighbors_*.json
(optional), and exposes ready-to-embed section HTML (SECTION_1_3,
SECTION_1_4, SECTION_1_5) plus project-tile tuples (TILES) for the
assembled dashboard.
"""

import json
import math

from ....shared import results_paths
from ....shared.viz import (
    W, ROW_H, GUT_L, GUT_R, fmt_ms, fmt_int, log_dot_plot, legend,
    chain_strip, CHAIN_NOTE, label,
)

RESULTS_DIR = results_paths.results_dir("ch03_gpu_1blob_nblock", "benchmark_results")

MB_PATH = results_paths.newest("multi_block_*.json", RESULTS_DIR)
NB_PATH = results_paths.newest_optional("neighbors_*.json", RESULTS_DIR)

with open(MB_PATH) as f:
    MB = json.load(f)
NB = None
if NB_PATH:
    with open(NB_PATH) as f:
        NB = json.load(f)

MB_ROWS = [r for r in MB["scenes"] if "skipped" not in r]
MB_SWEEP = [r for r in MB["block_tpb_sweep"] if "skipped" not in r]
# The sweep grid was extended to run at both connectivities on two scenes;
# every consumer must pick one explicitly or it draws a tangled mix.
MB_SWEEP4 = [r for r in MB_SWEEP if r.get("connectivity", 4) == 4]
MB_SWEEP8 = [r for r in MB_SWEEP if r.get("connectivity", 4) == 8]
MB_PEAK = MB["measured_peak_gb_s"]
MB_TPBS = MB["config"]["tpb_sweep"]
HAS_CONN8 = any(r.get("conn8_kernel_ms") is not None for r in MB_ROWS)

NB_ROWS = ([r for r in NB["scenes"] if "skipped" not in r] if NB else [])
HAS_NEIGHBORS = bool(NB_ROWS)


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


def mb_sweep_panels(connectivity=4):
    """Per-scene panels: metric vs blocks (log2 x), one line per tpb.

    Sources MB_SWEEP4 or MB_SWEEP8 explicitly — the sweep grid runs at
    both connectivities on two scenes, so drawing from the unsplit list
    tangles two different experiments into one line-set. A scene absent
    at this connectivity (the serpentine has no 8-conn sweep) is skipped.
    """
    src = MB_SWEEP4 if connectivity == 4 else MB_SWEEP8
    html = []
    for sname, metric, base_title in MB_SWEEP_PANELS:
        rows = [r for r in src if r["scene"] == sname]
        if not rows:
            continue
        title = base_title + (" — 8-conn" if connectivity == 8 else "")
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
        for b in (1, 4, 16, 48, 128, bmax):
            if b > bmax:
                continue
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


def mb_bandwidth_chart(show_conn8=True):
    """Model GB/s per scene against the measured copy peak — the gap IS the
    finding (or the model's sector-blindness; ncu decides). Paired bars
    where the 8-conn twin was also measured: 4-conn solid, 8-conn a
    lighter step of the same ramp — one entity, two variants.

    show_conn8=False renders 4-conn bars only (used in 1.3, which is
    scoped to 4-connectivity); the default renders the paired comparison
    (used in 1.4, the connectivity experiment)."""
    pair_mode = show_conn8 and HAS_CONN8
    hi = MB_PEAK * 1.04
    n = len(MB_ROWS)
    bar_h = 10
    row_h = 44 if pair_mode else ROW_H
    h = n * row_h + 34
    span = W - GUT_L - GUT_R

    def x_of(v):
        return GUT_L + v / hi * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" '
             f'aria-label="Modeled bandwidth per scene vs measured peak">']
    tick = 0
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * row_h}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{n * row_h + 18}" '
                     f'class="tick" text-anchor="middle">{tick:g}</text>')
        tick += 50
    px = x_of(MB_PEAK)
    parts.append(f'<line x1="{px:.1f}" y1="4" x2="{px:.1f}" '
                 f'y2="{n * row_h}" class="satline"/>')
    parts.append(f'<text x="{px - 6:.1f}" y="14" class="anno" '
                 f'text-anchor="end">measured D2D copy peak '
                 f'{MB_PEAK:.0f} GB/s</text>')
    for i, r in enumerate(MB_ROWS):
        cy = i * row_h + row_h / 2
        has8 = pair_mode and r.get("conn8_model_gb_s") is not None
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label(r["scene"])}</text>')
        v4 = r["multi_model_gb_s"]
        y4 = (cy - bar_h - 1) if has8 else (cy - bar_h / 2)
        bw4 = max(v4 / hi * span, 1.5)
        tip4 = (f"{label(r['scene'])} — 4-conn model {v4:.1f} GB/s = "
               f"{r['multi_pct_of_peak']:.1f}% of the measured peak · "
               f"{r['multi_mpx_s_kernel']:.0f} Mpx/s · lower-bound model")
        parts.append(f'<rect x="{GUT_L}" y="{y4:.1f}" width="{bw4:.1f}" '
                     f'height="{bar_h}" rx="3" class="seg s6" '
                     f'data-tip="{tip4}"/>')
        note4 = (f"{v4:.1f} GB/s · {r['multi_pct_of_peak']:.1f}%"
                if v4 >= 0.05 else "≈0 — barrier-bound")
        parts.append(f'<text x="{GUT_L + bw4 + 6:.1f}" y="{y4 + bar_h - 1:.1f}" '
                     f'class="rowval">{note4}</text>')
        if has8:
            v8 = r["conn8_model_gb_s"]
            y8 = cy + 1
            bw8 = max(v8 / hi * span, 1.5)
            tip8 = (f"{label(r['scene'])} — 8-conn model {v8:.1f} GB/s = "
                   f"{r['conn8_pct_of_peak']:.1f}% of the measured peak · "
                   f"{r['conn8_mpx_s']:.0f} Mpx/s · lower-bound model")
            parts.append(f'<rect x="{GUT_L}" y="{y8:.1f}" width="{bw8:.1f}" '
                         f'height="{bar_h}" rx="3" class="seg m2" '
                         f'data-tip="{tip8}"/>')
            note8 = (f"{v8:.1f} GB/s · {r['conn8_pct_of_peak']:.1f}%"
                    if v8 >= 0.05 else "≈0 — barrier-bound")
            parts.append(f'<text x="{GUT_L + bw8 + 6:.1f}" '
                         f'y="{y8 + bar_h - 1:.1f}" class="rowval">'
                         f'{note8}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def conn8_chart():
    """Per-scene 4-conn vs 8-conn kernel time, log axis, dumbbell pairs.

    One entity (the multi-block kernel) at two variants: solid dot = the
    4-conn baseline, hollow dot = the 8-conn twin, connected so the
    direction of the gap (and the serpentine's inversion) reads at a
    glance; the ratio gutter gives the exact number."""
    rows = [r for r in MB_ROWS if r.get("conn8_kernel_ms") is not None]
    gut_r = 92
    vals = [v for r in rows
            for v in (r["multi_kernel_ms"], r["conn8_kernel_ms"])]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * ROW_H + 34
    span = W - GUT_L - gut_r

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="4-conn vs '
             f'8-conn kernel time per scene">']
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
        v4, v8 = r["multi_kernel_ms"], r["conn8_kernel_ms"]
        x4, x8 = x_of(v4), x_of(v8)
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - gut_r}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label(r["scene"])}</text>')
        parts.append(f'<line x1="{x4:.1f}" y1="{cy:.1f}" x2="{x8:.1f}" '
                     f'y2="{cy:.1f}" class="pairline"/>')
        tip4 = (f"{label(r['scene'])} — 4-conn: {fmt_ms(v4)} ms, "
               f"{fmt_int(r['levels'])} levels")
        tip8 = (f"{label(r['scene'])} — 8-conn: {fmt_ms(v8)} ms, "
               f"{fmt_int(r['conn8_levels'])} levels")
        parts.append(f'<circle cx="{x4:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s6" data-tip="{tip4}"/>')
        parts.append(f'<circle cx="{x8:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot-o" data-tip="{tip8}"/>')
        ratio = r["conn8_vs_conn4"]
        rtxt = f"{ratio:.2f}×" if ratio >= 1 else f"{ratio:.2f}× slower"
        weight = ' font-weight="650"' if ratio >= 1 else ""
        parts.append(f'<text x="{W - gut_r + 8:.1f}" y="{cy + 4:.1f}" '
                     f'class="rowval"{weight}>{rtxt}</text>')
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


def conn8_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>4-conn ms</th>"
            "<th>8-conn ms</th><th>bare ms</th><th>ovh %</th>"
            "<th>vs 4-conn</th><th>levels 4→8</th><th>util % 4→8</th>"
            "<th>peak frontier</th><th>Mpx/s</th><th>GB/s</th>"
            "<th>% peak</th></tr>")
    body = []
    for r in MB_ROWS:
        if r.get("conn8_kernel_ms") is None:
            continue
        body.append(
            "<tr>"
            f"<td>{label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_ms(r['multi_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['conn8_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['conn8_bare_kernel_ms'])}</td>"
            f"<td>{r['conn8_overhead_pct']:+.1f}</td>"
            f"<td>{r['conn8_vs_conn4']:.2f}×</td>"
            f"<td>{fmt_int(r['levels'])}→{fmt_int(r['conn8_levels'])}</td>"
            f"<td>{r['multi_thread_util_pct']:.0f}→"
            f"{r['conn8_thread_util_pct']:.0f}</td>"
            f"<td>{fmt_int(r['conn8_peak_frontier'])}</td>"
            f"<td>{r['conn8_mpx_s']:.1f}</td>"
            f"<td>{r['conn8_model_gb_s']:.1f}</td>"
            f"<td>{r['conn8_pct_of_peak']:.1f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


def nb_variant_chart(var_key, ratio_key, ratio_min_key, var_label,
                     var_tip=None):
    """Per-scene conn8 baseline vs ONE per-barrier variant (radius-2 or
    warp-coop), log axis, dumbbell pairs — conn8_chart's geometry over
    the neighbors JSON. Same entity at two variants: solid dot = the
    conn8 baseline, hollow = the variant; the gutter gives the exact
    median ratio (>1 = the variant faster), with the best-vs-best form
    in the hover tooltip (agreement between the two = the drift signal)."""
    rows = NB_ROWS
    gut_r = 108
    vals = [v for r in rows for v in (r["conn8_kernel_ms"], r[var_key])]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * ROW_H + 34
    span = W - GUT_L - gut_r

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="8-conn '
             f'baseline vs {var_label} kernel time per scene">']
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
        v8, vv = r["conn8_kernel_ms"], r[var_key]
        x8, xv = x_of(v8), x_of(vv)
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - gut_r}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{label(r["scene"])}</text>')
        parts.append(f'<line x1="{x8:.1f}" y1="{cy:.1f}" x2="{xv:.1f}" '
                     f'y2="{cy:.1f}" class="pairline"/>')
        tip8 = (f"{label(r['scene'])} — 8-conn baseline: {fmt_ms(v8)} ms, "
                f"{fmt_int(r['conn8_levels'])} levels")
        extra = var_tip(r) if var_tip else ""
        tipv = (f"{label(r['scene'])} — {var_label}: {fmt_ms(vv)} ms{extra} "
                f"(best-vs-best ratio {r[ratio_min_key]:.2f}×)")
        parts.append(f'<circle cx="{x8:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s6" data-tip="{tip8}"/>')
        parts.append(f'<circle cx="{xv:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot-o" data-tip="{tipv}"/>')
        ratio = r[ratio_key]
        if abs(ratio - 1) < 0.005:
            rtxt, weight = "≈1.00×", ""
        elif ratio >= 1:
            rtxt, weight = f"{ratio:.2f}×", ' font-weight="650"'
        else:
            rtxt, weight = f"{ratio:.2f}× slower", ""
        parts.append(f'<text x="{W - gut_r + 8:.1f}" y="{cy + 4:.1f}" '
                     f'class="rowval"{weight}>{rtxt}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def nb_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>conn4 ms</th>"
            "<th>conn8 ms</th><th>r2 ms</th><th>wc ms</th>"
            "<th>r2 vs 8</th><th>(min)</th><th>wc vs 8</th><th>(min)</th>"
            "<th>levels 8→r2</th><th>interior %</th>"
            "<th>util 8/r2/wc %</th></tr>")
    body = []
    for r in NB_ROWS:
        body.append(
            "<tr>"
            f"<td>{label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_ms(r['conn4_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['conn8_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['r2_kernel_ms'])}</td>"
            f"<td>{fmt_ms(r['wc_kernel_ms'])}</td>"
            f"<td>{r['r2_vs_conn8']:.2f}×</td>"
            f"<td>{r['r2_vs_conn8_min']:.2f}×</td>"
            f"<td>{r['wc_vs_conn8']:.2f}×</td>"
            f"<td>{r['wc_vs_conn8_min']:.2f}×</td>"
            f"<td>{fmt_int(r['conn8_levels'])}→{fmt_int(r['r2_levels'])}</td>"
            f"<td>{r['r2_interior_pct']:.1f}</td>"
            f"<td>{r['conn8_thread_util_pct']:.0f}/"
            f"{r['r2_thread_util_pct']:.0f}/"
            f"{r['wc_thread_util_pct']:.0f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


# ------------------------------------------------- multi-block stat tiles
_mb_best_v2 = max(MB_ROWS, key=lambda r: r["multi_speedup_vs_v2"])
_mb_best_njit = max(MB_ROWS, key=lambda r: r["multi_speedup_vs_njit"])
_mb_best_cell = max(MB_SWEEP4, key=lambda r: r["mpx_s"])  # 4-conn only
_mb_best_cell8 = max(MB_SWEEP8, key=lambda r: r["mpx_s"]) if MB_SWEEP8 else None
_mb_peak_pct = max(r["multi_pct_of_peak"] for r in MB_ROWS)
_mb_conn8_rows = [r for r in MB_ROWS if r.get("conn8_kernel_ms") is not None]
_best_conn8 = (max(_mb_conn8_rows, key=lambda r: r["conn8_vs_conn4"])
              if _mb_conn8_rows else None)
_peak_pct8 = (max(r["conn8_pct_of_peak"] for r in _mb_conn8_rows)
             if _mb_conn8_rows else None)
_sq8000 = next((r for r in MB_ROWS if r["scene"] == "sq_8000_center"), None)
_serp8 = next((r for r in _mb_conn8_rows if r["scene"] == "serpentine_256"),
              None)

# --------------------------------------------------------- project tiles
_fastest = max(MB_SWEEP, key=lambda r: r["mpx_s"])  # across both conn's
_biggest = max(MB_ROWS, key=lambda r: r["filled"])
if _biggest.get("conn8_kernel_ms") is not None:
    _big_ms, _big_conn = _biggest["conn8_kernel_ms"], "8-conn"
else:
    _big_ms, _big_conn = _biggest["multi_kernel_ms"], "4-conn"
_big_speedup = _biggest["njit_ms"] / _big_ms
_cap_threads = max(int(t) * n for t, n in MB["coop_max_by_tpb"].items())
_cap_blocks = max(MB["coop_max_by_tpb"].values())
_mb_serp = next(r for r in MB_ROWS if r["scene"] == "serpentine_256")

# -- §1.5 per-barrier work experiments (neighbors JSON, optional)
if HAS_NEIGHBORS:
    _nb_big = max(NB_ROWS, key=lambda r: r["filled"])
    _nb_serp = next((r for r in NB_ROWS
                     if r["scene"] == "serpentine_256"), None)
    _nb_r2_lo = min(r["r2_vs_conn8"] for r in NB_ROWS)
    _nb_r2_hi = max(r["r2_vs_conn8"] for r in NB_ROWS)
    _nb_wc_lo = min(r["wc_vs_conn8"] for r in NB_ROWS)
    _nb_wc_hi = max(r["wc_vs_conn8"] for r in NB_ROWS)
    _nb_wc_mpx = _nb_big["filled"] / _nb_big["wc_kernel_ms"] / 1000
    # Its own single-session chain; the CPU baseline is the 8-CONN oracle
    # (slower than §1.3's 4-conn @njit) — never multiplied into §1.3.
    CHAIN_15 = [("CPU (@njit, 8-conn)", _nb_big["njit_ms"]),
                ("8-conn N blocks", _nb_big["conn8_kernel_ms"]),
                ("warp-coop", _nb_big["wc_kernel_ms"])]


# The 5 base project tiles (always present) + the conditional
# per-barrier-work tile -- concatenated by the assembler with every
# other chapter's tiles into one PROJECT_TILES list, registry order.
TILES = [
    (f"{_fastest['mpx_s']:.0f} Mpx/s",
     f"fastest sweep cell · {label(_fastest['scene'])} · "
     f"{_fastest['blocks']}×{_fastest['tpb']} · "
     f"{_fastest.get('connectivity', 4)}-conn",
     "blocks × tpb sweep"),
    (f"{_big_speedup:.1f}× chained",
     f"1 block → N blocks{' (+8-conn)' if _big_conn == '8-conn' else ''} · "
     f"{label(_biggest['scene'])} ({fmt_int(_biggest['filled'])} px) · "
     f"{fmt_ms(_big_ms)} ms — see §1.3 for the step-by-step multipliers",
     "N-block scene suite"),
    (f"{_cap_threads:,} threads",
     f"cooperative capacity wall · {_cap_blocks} blocks max · ~104 "
     f"regs/thread caps every grid shape at 512 threads/SM",
     "N-block capacity table"),
    (f"{MB_PEAK:.0f} GB/s",
     f"measured D2D copy peak · model share ≈{_mb_peak_pct:.0f}% (4-conn)"
     + (f", ≈{_peak_pct8:.0f}% (8-conn)" if _peak_pct8 else "")
     + " — a lower bound",
     "bandwidth instrumentation"),
    (f"{fmt_ms(_mb_serp['njit_ms'])} vs {fmt_ms(_mb_serp['v2_kernel_ms'])} ms",
     "standing worst case · serpentine 256² · CPU vs the best GPU design "
     "(single-block v2) — every parallel kernel loses here",
     "N-block scene suite"),
]
if HAS_NEIGHBORS:
    TILES.append(
        (f"{_nb_wc_mpx:.0f} Mpx/s",
         f"fastest whole-scene fill · warp-coop 8-conn · "
         f"{label(_nb_big['scene'])} ({fmt_int(_nb_big['filled'])} px) in "
         f"{fmt_ms(_nb_big['wc_kernel_ms'])} ms — see §1.5",
         "per-barrier experiments"))


LEG_MB = legend([("@njit CPU", "s2"), ("single-block v2", "s1"),
                 ("dual global (2 blocks)", "s4"),
                 (f"multi ({MB_ROWS[0]['multi_blocks']} blocks)", "s6")])
LEG_MB_SPD = legend([("vs single-block v2", "s1"), ("vs dual global", "s4"),
                     ("vs @njit CPU", "s2")])
LEG_MB_TPB = legend([(f"tpb {t}", f"m{i + 1}")
                     for i, t in enumerate(MB_TPBS)])
LEG_CONN8 = ('<div class="legend">'
             '<span><i class="chip" style="background:var(--s6)"></i>'
             '4-conn</span>'
             '<span><i class="chip chip-o"></i>8-conn</span>'
             '</div>')
LEG_MB_BW = LEG_CONN8 if HAS_CONN8 else ""
LEG_MB_BW_4 = legend([("4-conn model GB/s", "s6")])
LEG_NB_R2 = ('<div class="legend">'
             '<span><i class="chip" style="background:var(--s6)"></i>'
             '8-conn baseline</span>'
             '<span><i class="chip chip-o"></i>radius-2 twin</span>'
             '</div>')
LEG_NB_WC = ('<div class="legend">'
             '<span><i class="chip" style="background:var(--s6)"></i>'
             '8-conn baseline</span>'
             '<span><i class="chip chip-o"></i>warp-coop twin</span>'
             '</div>')

_mb_16m = next(r for r in MB_ROWS if r["scene"] == "sq_4000_corner")

CHAIN_13_16M = [("CPU (@njit)", _mb_16m["njit_ms"]),
               ("single-block v2", _mb_16m["v2_kernel_ms"]),
               ("dual global", _mb_16m["dual_global_kernel_ms"]),
               ("N blocks", _mb_16m["multi_kernel_ms"])]
CHAIN_13_64M = ([("CPU (@njit)", _sq8000["njit_ms"]),
                ("single-block v2", _sq8000["v2_kernel_ms"]),
                ("dual global", _sq8000["dual_global_kernel_ms"]),
                ("N blocks", _sq8000["multi_kernel_ms"])]
               if _sq8000 else None)
CHAIN_14_16M = (CHAIN_13_16M + [("8-connectivity", _mb_16m["conn8_kernel_ms"])]
               if _mb_16m.get("conn8_kernel_ms") is not None else None)
CHAIN_14_64M = (CHAIN_13_64M
               + [("8-connectivity", _sq8000["conn8_kernel_ms"])]
               if CHAIN_13_64M and _sq8000.get("conn8_kernel_ms") is not None
               else None)


_sweep8_pointer = (" The same sweep at 8 directions is charted next."
                  if MB_SWEEP8 else "")
_sweep8_card = ""
if MB_SWEEP8:
    _sweep8_card = f"""
<div class="card">
<h2>The same sweep, 8 directions</h2>
<p class="note">Identical grid, identical scenes, connectivity=8. Same
capacity ceilings (registers don't care how many neighbors a thread
probes) but a higher plateau, and the record cell moves to
{_mb_best_cell8['blocks']}×{_mb_best_cell8['tpb']} —
{_mb_best_cell8['mpx_s']:.0f} Mpx/s, the project's fastest fill.</p>
{LEG_MB_TPB}
{mb_sweep_panels(8)}
</div>
"""

_bw_conn8_note = ""
if _peak_pct8:
    _bw_conn8_note = (
        f" The 8-direction twins (lighter bars) reach ≈{_peak_pct8:.0f}% of "
        f"peak at LOWER wall-clock time — proof the 4-conn plateau was "
        f"never a hard DRAM wall; the ceiling's real cause shifts toward "
        f"occupancy/latency structure, sharpening what ncu must arbitrate.")

_conn8_card = ""
_conn8_table_block = ""
if HAS_CONN8:
    _conn8_card = f"""
<div class="card">
<h2>8 directions — the same scenes, twin kernels</h2>
<p class="note">One entity, two variants: solid dot = the 4-conn
baseline, hollow dot = the verbatim 8-conn twin, connected so the
direction of the gap reads at a glance. The width bet wins nearly
everywhere ({_best_conn8['conn8_vs_conn4']:.2f}× best, at
{label(_best_conn8['scene'])}) — levels halve, utilization roughly
doubles — except the serpentine, whose dumbbell points the other way
({_serp8['conn8_vs_conn4']:.2f}×): pure probe cost with no wider levels
to pay for it.</p>
{LEG_CONN8}
{conn8_chart()}
</div>
"""
    _conn8_table_block = f"""
<details><summary>8-connectivity per-scene results table</summary>
<div class="tablewrap">{conn8_table()}</div></details>
"""

_nb_section = ""
if HAS_NEIGHBORS:
    _nb_serp_note = ""
    if _nb_serp:
        _nb_serp_note = (
            f" The serpentine tells both stories at once: radius-2's guard "
            f"never fires there (interior = 0, "
            f"{_nb_serp['r2_vs_conn8']:.2f}× — ironically its best scene: "
            f"pure guard overhead beats paying for jumps), while warp-coop "
            f"hits {_nb_serp['wc_vs_conn8']:.2f}× and takes the scene "
            f"below its own 4-conn baseline "
            f"({fmt_ms(_nb_serp['wc_kernel_ms'])} vs "
            f"{fmt_ms(_nb_serp['conn4_kernel_ms'])} ms) for the first "
            f"time in the project.")
    _nb_section = f"""
<h3 class="subsection-h">1.5 More work per barrier — radius-2 and
warp-coop</h3>

<div class="card">
<h2>Two bets on the same question</h2>
<p class="note">Can a BFS level do more before paying its two grid.sync
barriers? <b>Radius-2</b> probes MORE: a guarded second ring (jumps only
where all 8 ring-1 neighbors are blob, keeping the fill exactly
8-connected) — levels halve, ~3× the probes. <b>Warp-coop</b> probes
SMARTER: 4 queue entries × 8 directions spread across a warp's 32 lanes
— identical work, one probe round per chunk instead of 8, results
bit-identical to the baseline. One interleaved round-robin, one pinned
grid, and the verdicts point in opposite directions: radius-2 LOSES
every scene ({_nb_r2_lo:.2f}–{_nb_r2_hi:.2f}×, worst at the biggest — by
64M px the lanes were already 81% fed, so tripled probe traffic is pure
bill), warp-coop WINS every scene
({_nb_wc_lo:.2f}–{_nb_wc_hi:.2f}×).{_nb_serp_note} Predictions-vs-verdicts
table and post-mortem in the multi_block README.</p>
{chain_strip(CHAIN_15,
             title=f"{label(_nb_big['scene'])} "
                   f"({fmt_int(_nb_big['filled'])} px), one session — "
                   f"the fastest whole-scene rate measured: "
                   f"{_nb_wc_mpx:.0f} Mpx/s",
             footnote="The CPU baseline here is the 8-connectivity oracle "
                      "(a different, slower baseline than §1.3's 4-conn "
                      "@njit), and every step is measured within the "
                      "neighbors benchmark's own session — this chain is "
                      "not multiplied into §1.3's.",
             total_cls="s6")}
</div>

<div class="card">
<h2>Radius-2 vs the 8-conn baseline</h2>
<p class="note">Solid dot = 8-conn baseline, hollow = the guarded ring-2
twin. The mechanism worked perfectly — levels halve to the pixel,
interior ≈100%, fill sets identical — and the economics still lose:
the width lever that powered §1.4's win was already exhausted, so the
extra probes and ~3× CAS attempts buy nothing. Its modeled bandwidth is
a project record that loses wall-clock: work-efficiency loss, not
machine inefficiency. Hover a hollow dot for the best-vs-best ratio
(agreement with the median = the drift signal).</p>
{LEG_NB_R2}
{nb_variant_chart("r2_kernel_ms", "r2_vs_conn8", "r2_vs_conn8_min",
                  "radius-2",
                  var_tip=lambda r: (f", {fmt_int(r['r2_levels'])} levels, "
                                     f"{r['r2_interior_pct']:.0f}% interior"))}
</div>

<div class="card">
<h2>Warp-coop vs the 8-conn baseline</h2>
<p class="note">Same BFS graph, same probes, same atomics — only the
work DISTRIBUTION changes, and every scene speeds up: there were idle
lanes to harvest everywhere (even the 64M px square idled 19% of lanes
at 8-conn; warp-coop closes it to 2%). The one drift caveat: the disk's
median and min ratios disagree (hover the hollow dot) — read that row
cautiously.</p>
{LEG_NB_WC}
{nb_variant_chart("wc_kernel_ms", "wc_vs_conn8", "wc_vs_conn8_min",
                  "warp-coop")}
</div>

<div class="card">
<h2>All numbers — per-barrier experiments</h2>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{nb_table()}</div></details>
</div>
"""



SECTION_1_3 = f"""<h3 class="subsection-h">1.3 Dual blocks vs N blocks</h3>
<div class="card">
<h2>The chain — single block to N blocks</h2>
<p class="note">Each arrow multiplies the one before it — the same
per-stage numbers as §1.1/§1.2 above and the N-block speedup below,
chained end to end from ONE benchmark session (multi_block's own JSON,
which re-measures the single-block and dual-block kernels fresh
alongside its own). This is the headline the earlier sections build
toward.</p>
{chain_strip(CHAIN_13_16M, title="16M px — square 4000² corner", total_cls="s6")}
{chain_strip(CHAIN_13_64M, title="64M px — square 8000² center, the project's biggest scene", footnote=CHAIN_NOTE, total_cls="s6") if CHAIN_13_64M else ""}
</div>

<div class="card">
<h2>N blocks — runtime per scene</h2>
<p class="note">Log scale — each decade gridline is 10×. The lineage on
one chart: CPU (green) → one block (blue) → two blocks (yellow) → the
cooperative maximum, {MB_ROWS[0]['multi_blocks']} blocks at tpb=256
(violet). The gap widens with blob size to
{_mb_best_v2['multi_speedup_vs_v2']:.1f}× vs one block; on the serpentine
the ordering inverts — more blocks means a costlier barrier and nothing to
feed. 4-conn kernels — the 8-direction twins are charted in §1.4. Hover
any dot; exact numbers in the table below.</p>
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
runtime). 4-conn kernels — the 8-direction twins are charted in §1.4.</p>
{LEG_MB_SPD}
{mb_speedup_chart()}
</div>

<div class="card">
<h2>The centerpiece — blocks × threads-per-block sweep (4-conn)</h2>
<p class="note">Throughput vs block count (log₂ axis), one line per
threads-per-block; line ends mark each tpb's cooperative-capacity limit
(registers: ~104/thread cap every configuration at 12,288 total threads =
512 per SM; tpb=32 reaches the 384-block ceiling). Near-linear scaling to
~8–16 blocks, then a plateau — and past it a decline: the capacity ends
(384×32, 192×64) run measurably slower, every extra block being another
barrier arrival. The best cell
({_mb_best_cell['blocks']}×{_mb_best_cell['tpb']}) obeys the rule: the
smallest tpb whose grid still covers the peak frontier in one stride
pass, with the block count maxed so the same frontier spreads across
more SMs. The serpentine panel is in kernel ms — no configuration helps
a shape that starves every block between barriers; it only gets worse as
the barrier population grows. The same sweep at 8 directions is in
§1.4.</p>
{LEG_MB_TPB}
{mb_sweep_panels(4)}
</div>

<div class="card">
<h2>Bandwidth — the modeled traffic vs the measured ceiling (4-conn)</h2>
<p class="note">Bars: algorithmic bytes moved (from each run's
exactly-once counters) ÷ kernel time. The dashed line is the MEASURED
device-to-device copy peak — the honest ceiling, not a spec sheet. The
4-conn plateau tops out at ≈{_mb_peak_pct:.0f}% of it <i>by a lower-bound
model</i> (≈61 B/pixel): 32 B DRAM sectors can inflate the real traffic
of scattered 3–4 B accesses. The 8-conn comparison is in §1.4.</p>
{LEG_MB_BW_4}
{mb_bandwidth_chart(show_conn8=False)}
</div>

<div class="card">
<h2>All numbers — N-block stage (4-conn)</h2>
<details open><summary>Per-scene results table (4-conn)</summary>
<div class="tablewrap">{mb_table()}</div></details>
</div>"""

SECTION_1_4 = f"""<h3 class="subsection-h">1.4 4 vs 8 connectivity</h3>
<div class="card">
<h2>The bonus branch — 8-connectivity on top of the chain</h2>
<p class="note">§1.3's chain above uses 4-connectivity throughout.
Swapping the N-block kernel for its verbatim 8-connectivity twin (same
capacity ceilings — registers don't care how many neighbors a thread
probes — but fewer, wider levels) adds one more multiplier on top.</p>
{chain_strip(CHAIN_14_16M, title="16M px", total_cls="s6") if CHAIN_14_16M else ""}
{chain_strip(CHAIN_14_64M, title="64M px", footnote=CHAIN_NOTE, total_cls="s6") if CHAIN_14_64M else ""}
</div>
{_conn8_card}
{_sweep8_card}
<div class="card">
<h2>Bandwidth — 4-conn vs 8-conn, paired</h2>
<p class="note">Same bars as §1.3's bandwidth card, with the 8-conn twin
overlaid (lighter step of the same ramp — one entity, two
variants).{_bw_conn8_note}</p>
{LEG_MB_BW}
{mb_bandwidth_chart(show_conn8=True)}
</div>
<div class="card">
<h2>All numbers — 8-connectivity</h2>
{_conn8_table_block}
</div>"""

SECTION_1_5 = _nb_section
