"""
Chapter 2 dashboard renderer: dual-block (2 blocks) split/global/dirsplit
kernels -- runtime, speedup vs the single-block v2 baseline, thread
placement, per-block balance over time, a throughput sweep paired with
the single-block stage, instrumentation overhead, and the CPU -> 2 blocks
chained speedup (section 1.2).

Extracted from the dashboard monolith. Self-contained: loads its own
dual_block_*.json (an explicit path can be passed via sys.argv[1],
forwarded by the dashboard assembler when given), and exposes
ready-to-embed section HTML (SECTION_1_2), the balance-panel hover JS
(JS), and the device/SM identifiers (DEVICE, SM_COUNT) the assembled
dashboard's subtitle needs.
"""
import json
import sys

from ....shared import results_paths
from ....shared import viz
from ...ch01_gpu_1blob_1block.benchmarks import visualize as sbs_viz

RESULTS_DIR = results_paths.results_dir("ch02_gpu_1blob_2block", "benchmark_results")

DUAL_PATH = sys.argv[1] if len(sys.argv) > 1 else results_paths.newest(
    "dual_block_*.json", RESULTS_DIR)

with open(DUAL_PATH) as f:
    DUAL = json.load(f)

DEVICE = DUAL["device"]
SM_COUNT = DUAL["sm_count"]

ROWS = DUAL["scenes"]
SWEEP = DUAL["tpb_sweep"]
PLACEMENT = DUAL["placement"]
SBS_SWEEP = sbs_viz.SWEEP


KERNELS = ["split", "global", "dirsplit"]
# Entity -> color slot, constant across every dual chart: s1 = single-block
# v2, s2 = @njit CPU, s3 = split, s4 = global, s5 = dirsplit.
KCLS = {"split": "s3", "global": "s4", "dirsplit": "s5"}


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

# This chapter's own tiles are subsection-local (embedded above via
# tiles_html), not part of the assembled dashboard's top project-tiles
# strip -- it contributes no project tile.
TILES = []



# --------------------------------------------------- chained speedups
# Each subsection reports its OWN stage's speedup from ITS OWN benchmark
# session (1.1/1.2 use single_block_shared's/dual_block's own JSON). 1.3
# additionally chains njit -> v2 -> dual-global -> N-blocks together using
# multi_block's OWN JSON, which re-measures v2 and dual-global fresh in
# the SAME session as its N-block numbers — one self-consistent product,
# never a cross-session multiply. (Three multi_block_*.json sessions on
# disk disagree slightly on njit/kernel timings from normal GPU clock
# drift; only the newest file's own numbers ever get multiplied here.)
_dual_36m = next(r for r in ROWS if r["scene"] == "sq_6000_center")


CHAIN_NOTE = viz.CHAIN_NOTE


CHAIN_12 = [("CPU (@njit)", _dual_36m["njit_ms"]),
           ("single-block v2", _dual_36m["v2_kernel_ms"]),
           ("dual global", _dual_36m["global_kernel_ms"])]



balance_html, balance_js = balance_panels()


JS = """const BPANELS = __BPANELS__;
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



SECTION_1_2 = f"""<h3 class="subsection-h">1.2 Dual blocks — split, global, dirsplit</h3>
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
</div>"""
