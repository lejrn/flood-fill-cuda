"""
Shared presentation core for the project's benchmark dashboards.

Extracted verbatim from the dashboard monolith: formatting helpers, the
generic log-scale dot-plot engine, the legend/chain-strip components, the
design system (CSS) and the tooltip JS engine, plus the scene-name map
shared by the ch02 and ch03 renderers. Every chapter's benchmarks/visualize.py
and the top-level dashboard/assemble.py import from here rather than
duplicating any of it.

Color-slot contract (consistent across every chart in the project):
  s1 = single-block v2 spill kernel     s2 = @njit CPU reference
  s3 = split (dual-block)               s4 = global (dual-block) / dual-global baseline
  s5 = dirsplit (dual-block)            s6 = multi-block (N blocks) / project accent
  s7 = dual-blob sequential             s8 = dual-blob multisource
  m1..m5 = a 5-step monochrome ramp used by tpb-sweep legends
"""

import math

# ------------------------------------------------------------- formatting
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


# --------------------------------------------------------- legend / chain strip
def legend(entries):
    return ('<div class="legend">'
            + "".join(f'<span><i class="chip" style="background:var(--{c})">'
                      f'</i>{n}</span>' for n, c in entries)
            + '</div>')


def chain_strip(pairs, title="", footnote="", total_cls="s6"):
    """A row of connected chips: baseline -> ...intermediate stages... ->
    final, each step's multiplier computed from consecutive ms values
    (prev/curr), ending in a bold TOTAL chip (first/last — the same
    number the per-step multipliers telescope to, since this is nothing
    more than a product of ratios rendered one factor at a time).

    pairs: [(label, ms_value), ...] ordered slowest/baseline -> fastest.
    Not a chart — plain HTML/CSS, wraps on narrow viewports.
    """
    parts = ['<div class="chain">']
    if title:
        parts.append(f'<div class="chain-title">{title}</div>')
    parts.append('<div class="chain-row">')
    for i, (label_, ms) in enumerate(pairs):
        parts.append(f'<div class="chain-step"><div class="chain-lbl">'
                     f'{label_}</div><div class="chain-val">'
                     f'{fmt_ms(ms)} ms</div></div>')
        if i < len(pairs) - 1:
            mult = ms / pairs[i + 1][1]
            parts.append(f'<div class="chain-op">×{mult:.2f}</div>')
    total = pairs[0][1] / pairs[-1][1]
    parts.append(f'<div class="chain-op">=</div>'
                 f'<div class="chain-total" style="background:var(--{total_cls})">'
                 f'{total:.2f}×</div>')
    parts.append('</div>')
    if footnote:
        parts.append(f'<div class="chain-note">{footnote}</div>')
    parts.append('</div>')
    return "\n".join(parts)


CHAIN_NOTE = (
    "Each step is measured within ONE benchmark session (never multiplied "
    "across separate stages' own standalone runs), so this may differ "
    "~1–3% from each stage's own headline number above — normal GPU clock "
    "drift between sessions, not an error in the arithmetic.")


# ------------------------------------------------------------- scene name map
# Shared by ch02 (dual-block) and ch03 (multi-block) renderers; ch01 and ch04
# use their own SBS_LABELS / DB_LABELS (different scene sets).
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


def label(name):
    return SCENE_LABELS.get(name, name)


# ------------------------------------------------------------------- design system
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
  --s6: #8257d8; --s7: #d6484a; --s8: #0d94ac;
  --m1: #c4b1ef; --m2: #a488e2; --m3: #8257d8; --m4: #633cb8; --m5: #452683;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #008300; --s3: #d55181; --s4: #c98500; --s5: #199e70;
    --s6: #9678db; --s7: #e2585a; --s8: #189aad;
    --m1: #d6caf5; --m2: #b8a2ea; --m3: #9678db; --m4: #7d5bd0; --m5: #6743bb;
  }
}
:root[data-theme="dark"] .viz-root {
  --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
  --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #008300; --s3: #d55181; --s4: #c98500; --s5: #199e70;
  --s6: #9678db; --s7: #e2585a; --s8: #189aad;
  --m1: #d6caf5; --m2: #b8a2ea; --m3: #9678db; --m4: #7d5bd0; --m5: #6743bb;
}
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--ink-2); font-size: 13px; margin-bottom: 20px; }
.domain-h { font-size: 21px; font-weight: 700; margin: 30px 0 6px;
            padding-top: 14px; border-top: 2px solid var(--border); }
.domain-h:first-of-type { border-top: none; padding-top: 0; }
.subsection-h { font-size: 15.5px; font-weight: 650; color: var(--ink);
                margin: 20px 0 2px; }
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
.s7 { fill: var(--s7); } .s8 { fill: var(--s8); }
.m1 { fill: var(--m1); } .m2 { fill: var(--m2); }
.m3 { fill: var(--m3); } .m4 { fill: var(--m4); } .m5 { fill: var(--m5); }
.line { fill: none; stroke-width: 2; }
.s1l { stroke: var(--s1); } .s3l { stroke: var(--s3); }
.s4l { stroke: var(--s4); } .s5l { stroke: var(--s5); }
.s7l { stroke: var(--s7); } .s8l { stroke: var(--s8); }
.m1l { stroke: var(--m1); } .m2l { stroke: var(--m2); }
.m3l { stroke: var(--m3); } .m4l { stroke: var(--m4); }
.m5l { stroke: var(--m5); }
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
.tile-src { font-size: 11px; color: var(--muted); margin-top: 6px; }
.timeline { position: relative; padding-left: 30px; margin: 6px 0 26px; }
.timeline::before {
  content: ""; position: absolute; left: 9px; top: 8px; bottom: 8px;
  width: 2px; background: var(--grid); border-radius: 1px;
}
.tl-node { position: relative; margin-bottom: 14px; }
.tl-dot {
  position: absolute; left: -27px; top: 16px; width: 12px; height: 12px;
  border-radius: 50%; background: var(--tl, var(--s1));
  border: 2px solid var(--page); box-shadow: 0 0 0 1px var(--border);
}
.tl-card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px;
  display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap;
}
.tl-main { flex: 1 1 340px; min-width: 0; }
.tl-badge { font-size: 11px; color: var(--muted); text-transform: uppercase;
            letter-spacing: 0.02em; }
.tl-title { font-size: 13.5px; font-weight: 650; margin: 2px 0 4px; }
.tl-overview { font-size: 12px; color: var(--ink-2); margin: 0 0 8px; }
.tl-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.tl-chip { border: 1px solid var(--border); border-radius: 6px;
           padding: 3px 9px; font-size: 12px; color: var(--ink-2); }
.tl-chip b { color: var(--ink); font-variant-numeric: tabular-nums; }
.tl-mini { flex: 0 0 130px; }
.dot-o { fill: var(--surface-1); stroke: var(--s6); stroke-width: 2; }
.pairline { stroke: var(--s6); stroke-width: 1.5; opacity: 0.45; }
.pairline-db { stroke: var(--muted); stroke-width: 1.5; opacity: 0.5; }
.dot-o7 { fill: var(--surface-1); stroke: var(--s7); stroke-width: 2; }
.dot-o8 { fill: var(--surface-1); stroke: var(--s8); stroke-width: 2; }
.pairline-s7 { stroke: var(--s7); stroke-width: 1.5; opacity: 0.45; }
.pairline-s8 { stroke: var(--s8); stroke-width: 1.5; opacity: 0.45; }
.chip-o { background: transparent; border: 2px solid var(--s6);
          box-sizing: border-box; }
.chain { margin: 4px 0 16px; }
.chain-title { font-size: 12px; font-weight: 600; color: var(--ink-2);
               margin-bottom: 6px; }
.chain-row { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.chain-step { background: var(--surface-1); border: 1px solid var(--border);
              border-radius: 8px; padding: 6px 11px; min-width: 92px; }
.chain-lbl { font-size: 11px; color: var(--ink-2); }
.chain-val { font-size: 13px; font-weight: 650; font-variant-numeric: tabular-nums; }
.chain-op { font-size: 13px; color: var(--muted); font-weight: 600;
            padding: 0 2px; }
.chain-total { background: var(--s6); color: #fff; border-radius: 8px;
               padding: 8px 14px; font-size: 15px; font-weight: 700;
               font-variant-numeric: tabular-nums; }
.chain-note { font-size: 11px; color: var(--muted); margin-top: 6px; }
"""


TOOLTIP_JS = """
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
"""
