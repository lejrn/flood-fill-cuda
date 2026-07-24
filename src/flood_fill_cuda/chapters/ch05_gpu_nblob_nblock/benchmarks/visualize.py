"""
Chapter 5 dashboard renderer: the seed-discovery stage (section 3) --
seed_merge (candidate scan + colliding waves + in-flight unions) vs
ccl_fill (union-find CCL prepass + canonical-seed fill), the discovery
tax against ch04's given-seeds kernel, and the phase attribution.

Self-contained: loads its own seed_discovery_*.json (optional -- this
stage postdates the others, and a dashboard run predating it should
still render, just omitting section 3). Exposes SECTION_3 (empty string
if no data) and TILES (0 or 1 project tiles).
"""

import json
import math
import os

from ....shared import results_paths
from ....shared.viz import (
    W, ROW_H, GUT_L, GUT_R, fmt_ms, fmt_int, legend, log_dot_plot,
)

RESULTS_DIR = results_paths.results_dir("ch05_gpu_nblob_nblock",
                                        "benchmark_results")
_WAVEFRONT_DIR = results_paths.results_dir("ch05_gpu_nblob_nblock",
                                           "wavefront")
# Relative to the CONSUMING page's own output dir (the assembled
# dashboard's results/dashboard/) -- computed, not hardcoded.
_CONSUMER_OUT_DIR = results_paths.results_dir("dashboard")
_WAVEFRONT_RELPATH = os.path.relpath(_WAVEFRONT_DIR, _CONSUMER_OUT_DIR)

SD_PATH = results_paths.newest_optional("seed_discovery_*.json", RESULTS_DIR)

SD = None
if SD_PATH:
    with open(SD_PATH) as f:
        SD = json.load(f)

SD_ROWS = SD["scenes"] if SD else []
SD_PEAK = SD["measured_peak_gb_s"] if SD else 0.0
HAS_SEED = bool(SD_ROWS)

# The seeding-density sweep postdates the main session's JSON — optional,
# same contract: absent file, absent card.
SW_PATH = results_paths.newest_optional("seeding_*.json", RESULTS_DIR)
SW = None
if SW_PATH:
    with open(SW_PATH) as f:
        SW = json.load(f)
SW_ROWS = SW["scenes"] if SW else []
HAS_SWEEP = HAS_SEED and bool(SW_ROWS)

# The tuning cross-product (builds x rules x strides) — optional, same
# contract again.
TU_PATH = results_paths.newest_optional("tuning_*.json", RESULTS_DIR)
TU = None
if TU_PATH:
    with open(TU_PATH) as f:
        TU = json.load(f)
TU_ROWS = TU["scenes"] if TU else []
HAS_TUNING = HAS_SEED and bool(TU_ROWS)

SD_LABELS = {
    "two_sq_2800": "two squares 2800²",
    "two_disks_r1400": "two disks r=1400",
    "asym_4000_800": "asymmetric 4000²+800²",
    "blob_grid_100": "grid of 100 blobs",
    "random_4000": "random noise 4000²",
    "comb_2000": "comb, 2000 teeth",
    "serpentine_256": "serpentine 256²",
}


def sd_label(name):
    return SD_LABELS.get(name, name)


def sd_runtime_chart():
    series = [(lambda r: r["njit_ms"], "@njit CPU (CCL + fill)", "s2"),
              (lambda r: r["merge_ms"], "seed_merge", "s7"),
              (lambda r: r["ccl_ms"], "ccl_fill", "s8")]
    return log_dot_plot(SD_ROWS, series,
                        "Seed-discovery runtime per scene, log scale",
                        lambda r: sd_label(r["scene"]))


def sd_ab_chart():
    """seed_merge vs ccl_fill kernel time per scene, log axis, dumbbell
    pairs -- two SOLID dots (s7, s8): two distinct discovery mechanisms,
    not variants of one entity (hollow stays reserved for true twins).
    Gutter: ccl_ms / merge_ms, >1 = seed_merge (discovery inside the
    fill) faster."""
    rows = SD_ROWS
    gut_r = 92
    vals = [v for r in rows for v in (r["merge_ms"], r["ccl_ms"])]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * ROW_H + 34
    span = W - GUT_L - gut_r

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="seed_merge '
             f'vs ccl_fill kernel time per scene">']
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
        v_mg, v_cc = r["merge_ms"], r["ccl_ms"]
        x_mg, x_cc = x_of(v_mg), x_of(v_cc)
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - gut_r}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{sd_label(r["scene"])}</text>')
        parts.append(f'<line x1="{x_mg:.1f}" y1="{cy:.1f}" x2="{x_cc:.1f}" '
                     f'y2="{cy:.1f}" class="pairline-db"/>')
        tip_mg = (f"{sd_label(r['scene'])} — seed_merge: {fmt_ms(v_mg)} ms "
                  f"({fmt_int(r['candidates'])} candidates, "
                  f"{fmt_int(r['unions_merge'])} unions)")
        tip_cc = (f"{sd_label(r['scene'])} — ccl_fill: {fmt_ms(v_cc)} ms "
                  f"({fmt_int(r['unions_ccl'])} unions)")
        parts.append(f'<circle cx="{x_mg:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s7" data-tip="{tip_mg}"/>')
        parts.append(f'<circle cx="{x_cc:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s8" data-tip="{tip_cc}"/>')
        ratio = r["merge_vs_ccl"]
        if ratio >= 1:
            rtxt, weight = f"{ratio:.2f}×", ' font-weight="650"'
        else:
            rtxt, weight = f"{1 / ratio:.2f}× ccl", ' font-weight="650"'
        parts.append(f'<text x="{W - gut_r + 8:.1f}" y="{cy + 4:.1f}" '
                     f'class="rowval"{weight}>{rtxt}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def sd_table():
    head = ("<tr><th>scene</th><th>filled px</th><th>blobs</th>"
            "<th>candidates</th><th>@njit ms</th>"
            "<th>merge ms</th><th>ccl ms</th><th>ccl/merge</th>"
            "<th>(min)</th><th>scan ms</th><th>flatten ms</th>"
            "<th>ccl union ms</th>"
            "<th>tax merge</th><th>tax ccl</th>"
            "<th>vs @njit</th><th>GB/s (merge)</th><th>% peak</th></tr>")
    body = []
    for r in SD_ROWS:
        tax_m = r.get("discovery_tax_merge")
        tax_c = r.get("discovery_tax_ccl")
        # in-kernel device stamps where the JSON has them; the standalone
        # phase-kernel times as the pre-instrumentation fallback
        scan = r.get("merge_scan_dev_ms", r["scan_ms"])
        flat = r.get("merge_flatten_dev_ms")
        cclu = r.get("ccl_union_dev_ms", r["cclp_ms"])
        body.append(
            "<tr>"
            f"<td>{sd_label(r['scene'])}</td>"
            f"<td>{fmt_int(r['filled'])}</td>"
            f"<td>{fmt_int(r['n_blobs'])}</td>"
            f"<td>{fmt_int(r['candidates'])}</td>"
            f"<td>{fmt_ms(r['njit_ms'])}</td>"
            f"<td>{fmt_ms(r['merge_ms'])}</td>"
            f"<td>{fmt_ms(r['ccl_ms'])}</td>"
            f"<td>{r['merge_vs_ccl']:.2f}×</td>"
            f"<td>{r['merge_vs_ccl_min']:.2f}×</td>"
            f"<td>{fmt_ms(scan)}</td>"
            f"<td>{fmt_ms(flat) if flat is not None else '—'}</td>"
            f"<td>{fmt_ms(cclu)}</td>"
            f"<td>{f'{tax_m:.2f}×' if tax_m else '—'}</td>"
            f"<td>{f'{tax_c:.2f}×' if tax_c else '—'}</td>"
            f"<td>{max(r['speedup_merge_vs_njit'], r['speedup_ccl_vs_njit']):.1f}×</td>"
            f"<td>{r['merge_model_gb_s']:.1f}</td>"
            f"<td>{r['merge_pct_of_peak']:.1f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


# ------------------------------------------------- experiment chart engine
# One fixed slot per scene, shared by every curve chart in this section,
# so a scene keeps its color from card to card.
_SCENE_SLOTS = {
    "two_sq_2800": "s1", "two_disks_r1400": "s8", "asym_4000_800": "s6",
    "blob_grid_100": "s5", "random_4000": "s3", "comb_2000": "s4",
    "serpentine_256": "s7",
}

_CURVE_H = 230          # plot-area height; x labels ride below
_CURVE_GUT_L = 64       # y-axis labels only, no row labels


def _curve_chart(x_labels, series, aria, y_ticks, y_fmt, ref=None,
                 ref_label=None, log_y=True):
    """Categorical-x line chart: series = [(name, slot, points)] with
    points aligned to x_labels, each point (value | None, tip). ref draws
    an emphasized dashed baseline (the 'this equals v1' line)."""
    span = W - _CURVE_GUT_L - GUT_R
    n = len(x_labels)
    xs = [_CURVE_GUT_L + span * (i + 0.5) / n for i in range(n)]
    vals = [v for _, _, pts in series for v, _ in pts if v is not None]
    lo, hi = min(vals), max(vals)
    if ref is not None:
        lo, hi = min(lo, ref), max(hi, ref)
    top = 10
    h = _CURVE_H + top + 28

    if log_y:
        pad = 0.05 * (math.log10(hi / lo) or 1.0)
        llo, lhi = math.log10(lo) - pad, math.log10(hi) + pad

        def y_of(v):
            return top + (lhi - math.log10(v)) / (lhi - llo) * _CURVE_H
    else:
        pad = (hi - lo) * 0.06 or 1.0
        flo, fhi = lo - pad, hi + pad

        def y_of(v):
            return top + (fhi - v) / (fhi - flo) * _CURVE_H

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{aria}">']
    for x, xl in zip(xs, x_labels):
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                     f'y2="{top + _CURVE_H}" class="rowline"/>')
        parts.append(f'<text x="{x:.1f}" y="{h - 6}" class="tick" '
                     f'text-anchor="middle">{xl}</text>')
    for t in y_ticks:
        if not lo <= t <= hi or t == ref:
            continue
        y = y_of(t)
        parts.append(f'<line x1="{_CURVE_GUT_L}" y1="{y:.1f}" '
                     f'x2="{W - GUT_R}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{_CURVE_GUT_L - 8}" y="{y + 4:.1f}" '
                     f'class="tick" text-anchor="end">{y_fmt(t)}</text>')
    if ref is not None:
        y = y_of(ref)
        parts.append(f'<line x1="{_CURVE_GUT_L}" y1="{y:.1f}" '
                     f'x2="{W - GUT_R}" y2="{y:.1f}" class="satline"/>')
        parts.append(f'<text x="{_CURVE_GUT_L - 8}" y="{y + 4:.1f}" '
                     f'class="tick" text-anchor="end" font-weight="650">'
                     f'{ref_label or y_fmt(ref)}</text>')
    for name, slot, pts in series:
        coords = " ".join(f"{x:.1f},{y_of(v):.1f}"
                          for x, (v, _) in zip(xs, pts) if v is not None)
        parts.append(f'<polyline points="{coords}" class="line {slot}l"/>')
        for x, (v, tip) in zip(xs, pts):
            if v is None:
                continue
            parts.append(f'<circle cx="{x:.1f}" cy="{y_of(v):.1f}" r="4" '
                         f'class="dot {slot}" data-tip="{tip}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _scene_legend(rows):
    return legend([(sd_label(r["scene"]),
                    _SCENE_SLOTS.get(r["scene"], "s1")) for r in rows])


def sw_curve_chart():
    """Speedup over v1 across the seeding strides, one line per scene."""
    order = ["v1"] + ["S" + str(s) for s in SW["strides"]]
    series = []
    for r in SW_ROWS:
        v1 = r["configs"]["v1"]["ms"]
        pts = []
        for cfg in order:
            c = r["configs"][cfg]
            sp = v1 / c["ms"]
            ph = c.get("phase_ms") or {}
            tip = (f"{sd_label(r['scene'])} — {cfg}: {fmt_ms(c['ms'])} ms "
                   f"({sp:.2f}× vs v1) · {fmt_int(c['levels'])} levels · "
                   f"{fmt_int(c['candidates'])} candidates · fill "
                   f"{fmt_ms(ph.get('fill'))} / flatten "
                   f"{fmt_ms(ph.get('flatten'))} ms")
            pts.append((sp, tip))
        series.append((sd_label(r["scene"]),
                       _SCENE_SLOTS.get(r["scene"], "s1"), pts))
    return _curve_chart(order, series,
                        "Speedup over v1 across seeding strides, per scene",
                        (0.25, 0.5, 2, 4), lambda t: f"{t:g}×",
                        ref=1.0, ref_label="1× = v1")


def sw_table():
    order = ["v1"] + ["S" + str(s) for s in SW["strides"]] + ["ccl"]
    head = ("<tr><th>scene</th>"
            + "".join(f"<th>{c} ms</th>" for c in order)
            + "<th>best</th><th>vs v1</th><th>vs ccl</th></tr>")
    body = []
    for r in SW_ROWS:
        cells = [f"<td>{sd_label(r['scene'])}</td>"]
        for cfg in order:
            c = r["configs"][cfg]
            v = fmt_ms(c["ms"])
            if cfg == r["best_cfg"]:
                v = f"<b>{v}</b>"
            cells.append(f'<td title="{fmt_int(c["levels"])} levels, '
                         f'{fmt_int(c["candidates"])} candidates">{v}</td>')
        cells.append(f"<td>{r['best_cfg']}</td>")
        cells.append(f"<td>{r['best_vs_v1']:.2f}×</td>")
        cells.append(f"<td>{r['best_vs_ccl']:.2f}×</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


_sw_card = ""
if HAS_SWEEP:
    _sw_best_gain = max(SW_ROWS, key=lambda r: r["best_vs_v1"])
    _sw_wins = sum(1 for r in SW_ROWS if r["best_cfg"] != "v1")
    _sw_card = f"""
<div class="card">
<h2>Seeding density — how many seeds should discovery plant?</h2>
<p class="note">The corner rule plants one seed per solid rectangle, so
the fill clock runs O(blob diameter). The lattice twin ADDS a seed at
every red pixel on an S×S grid (canonical labels provably unchanged for
every S) and compresses the parent chains before the repaint. S=0 is
the compression-only control; S=1 seeds every red pixel — the ccl-like
boundary where all connectivity flows through collisions. One line per
scene, y = speedup over v1 (log scale), the dashed line is v1 itself:
above it densified seeding wins, below it the extra unions cost more
than the shorter fill saves. Densified seeding beat the corner rule on
{_sw_wins} of {len(SW_ROWS)} scenes — best case
{_sw_best_gain['best_vs_v1']:.2f}× over v1 ({_sw_best_gain['best_cfg']}
on {sd_label(_sw_best_gain['scene'])}). Two shapes to notice: the rise
into S16 and the sag at S64 — the cross-product below chases both with
a finer stride curve. Hover any dot for ms, levels, candidates and the
fill/flatten phase split.</p>
{_scene_legend(SW_ROWS)}
{sw_curve_chart()}
<details><summary>All numbers — stride sweep (this run: 24-block lattice
build, before the register fixes below)</summary>
<div class="tablewrap">{sw_table()}</div></details>
</div>
"""

def tu_table():
    builds = ("fused", "r128", "split")
    head = ("<tr><th>scene</th><th>v1 ms</th><th>ccl ms</th>"
            + "".join(f"<th>{b} best</th>" for b in builds)
            + "<th>overall best</th><th>vs v1</th><th>vs ccl</th></tr>")
    body = []
    for r in TU_ROWS:
        cells = [f"<td>{sd_label(r['scene'])}</td>",
                 f"<td>{fmt_ms(r['configs']['v1']['ms'])}</td>",
                 f"<td>{fmt_ms(r['configs']['ccl']['ms'])}</td>"]
        for b in builds:
            n = r["best_per_build"][b]
            c = r["configs"][n]
            label = n.split("_", 1)[1]
            v = f"{label}: {fmt_ms(c['ms'])}"
            if n == r["best_cfg"]:
                v = f"<b>{v}</b>"
            cells.append(f'<td title="{fmt_int(c["levels"])} levels, '
                         f'{fmt_int(c["candidates"])} candidates">{v}</td>')
        cells.append(f"<td>{r['best_cfg']}</td>")
        cells.append(f"<td>{r['best_vs_v1']:.2f}×</td>")
        cells.append(f"<td>{r['best_vs_ccl']:.2f}×</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


# ------------------------------------------ the cross-product's four views
_TU_BUILD_ORDER = ("v1", "fused", "r128", "split")
# builds share the monochrome ramp (one kernel family, build varied);
# v1 keeps seed_merge's s7 red from the charts above
_TU_BUILD_SLOTS = {"v1": "s7", "fused": "m1", "r128": "m3", "split": "m5"}
_TU_BUILD_LABELS = {
    "v1": "v1 (corner rule)", "fused": "fused lattice",
    "r128": "r128 — compiler cap", "split": "split — core + plain cleanup"}
_TU_BUILD_VERDICTS = {
    "v1": "the reference",
    "fused": "the handicapped one — ONE register over the line",
    "r128": "works — and 122 is BELOW the cap: the compiler had room all "
            "along, it just didn't try",
    "split": "works equally well — the cooperative core alone never "
             "needed the extra registers"}


def tu_regs_chart():
    """Registers per thread, bar per build, with the 128-register line
    that decides two-blocks-per-SM drawn through them."""
    binfo = TU["builds"]
    gut_r = 118
    span = W - GUT_L - gut_r
    x_max = max([140] + [v.get("regs", 0) + 8 for v in binfo.values()
                         if v.get("regs")])

    def x_of(v):
        return GUT_L + v / x_max * span

    n = len(_TU_BUILD_ORDER)
    h = n * ROW_H + 56
    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Registers '
             f'per thread and cooperative blocks per build">']
    for t in (0, 32, 64, 96):
        x = x_of(t)
        parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" '
                     f'y2="{n * ROW_H}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{n * ROW_H + 18}" class="tick" '
                     f'text-anchor="middle">{t}</text>')
    x128 = x_of(128)
    parts.append(f'<line x1="{x128:.1f}" y1="0" x2="{x128:.1f}" '
                 f'y2="{n * ROW_H}" class="xhair"/>')
    parts.append(f'<text x="{x128:.1f}" y="{n * ROW_H + 18}" class="tick" '
                 f'text-anchor="middle" font-weight="650">128</text>')
    parts.append(f'<text x="{x128:.1f}" y="{n * ROW_H + 36}" class="tick" '
                 f'text-anchor="middle">the 2-blocks-per-SM line: 65,536 '
                 f'regs ÷ 256 threads ÷ 2 blocks</text>')
    for i, b in enumerate(_TU_BUILD_ORDER):
        info = binfo.get(b, {})
        regs, blocks = info.get("regs"), info.get("coop_256")
        cy = i * ROW_H + ROW_H / 2
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{_TU_BUILD_LABELS[b]}</text>')
        if regs is None:
            continue
        tip = (f"{_TU_BUILD_LABELS[b]}: {regs} regs/thread → {blocks} "
               f"cooperative blocks at tpb 256 "
               f"({info.get('coop_128', '?')} at tpb 128)")
        parts.append(f'<rect x="{GUT_L}" y="{cy - 9:.1f}" '
                     f'width="{x_of(regs) - GUT_L:.1f}" height="18" rx="3" '
                     f'class="{_TU_BUILD_SLOTS[b]}" data-tip="{tip}"/>')
        parts.append(f'<text x="{x_of(regs) + 6:.1f}" y="{cy + 4:.1f}" '
                     f'class="rowval" font-weight="650">{regs}</text>')
        w = ' font-weight="650"' if blocks and blocks < 48 else ""
        parts.append(f'<text x="{W - gut_r + 42}" y="{cy + 4:.1f}" '
                     f'class="rowval"{w}>→ {blocks} blocks</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def tu_regs_table():
    head = ("<tr><th>build</th><th>regs/thread</th>"
            "<th>coop blocks (tpb 256)</th><th>verdict</th></tr>")
    body = []
    for b in _TU_BUILD_ORDER:
        info = TU["builds"].get(b, {})
        body.append(
            "<tr>"
            f"<td>{_TU_BUILD_LABELS[b]}</td>"
            f"<td>{info.get('regs', '—')}</td>"
            f"<td>{info.get('coop_256', '—')}</td>"
            f"<td>{_TU_BUILD_VERDICTS[b]}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _tu_rule_best(r, rule):
    """(config name, config) with the lowest ms among one rule's cells,
    any build, any stride. The *_0 controls carry no lattice and belong
    to neither rule."""
    tag = f"_{rule}"
    best = None
    for name, c in r["configs"].items():
        if tag not in name:
            continue
        if best is None or c["ms"] < best[1]["ms"]:
            best = (name, c)
    return best


def tu_rule_chart():
    """Best plain-lattice cell vs best interior cell per scene, log-ms
    dumbbells. Gutter: L/I ratio — above 1 the interior rule wins."""
    rows = TU_ROWS
    gut_r = 92
    pairs = [(_tu_rule_best(r, "L"), _tu_rule_best(r, "I")) for r in rows]
    vals = [c["ms"] for pr in pairs for _, c in pr]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = max(vals) * 1.3
    n = len(rows)
    h = n * ROW_H + 34
    span = W - GUT_L - gut_r

    def x_of(v):
        return GUT_L + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Best '
             f'plain-lattice vs best interior config per scene">']
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
    for i, (r, ((ln, lc), (iname, ic))) in enumerate(zip(rows, pairs)):
        cy = i * ROW_H + ROW_H / 2
        x_l, x_i = x_of(lc["ms"]), x_of(ic["ms"])
        parts.append(f'<line x1="{GUT_L}" y1="{cy:.1f}" x2="{W - gut_r}" '
                     f'y2="{cy:.1f}" class="rowline"/>')
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{sd_label(r["scene"])}</text>')
        parts.append(f'<line x1="{x_l:.1f}" y1="{cy:.1f}" x2="{x_i:.1f}" '
                     f'y2="{cy:.1f}" class="pairline-db"/>')
        tip_l = (f"{sd_label(r['scene'])} — best plain lattice {ln}: "
                 f"{fmt_ms(lc['ms'])} ms · {fmt_int(lc['candidates'])} "
                 f"candidates · {fmt_int(lc['levels'])} levels")
        tip_i = (f"{sd_label(r['scene'])} — best interior {iname}: "
                 f"{fmt_ms(ic['ms'])} ms · {fmt_int(ic['candidates'])} "
                 f"candidates · {fmt_int(ic['levels'])} levels")
        parts.append(f'<circle cx="{x_l:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s4" data-tip="{tip_l}"/>')
        parts.append(f'<circle cx="{x_i:.1f}" cy="{cy:.1f}" r="5" '
                     f'class="dot s5" data-tip="{tip_i}"/>')
        ratio = lc["ms"] / ic["ms"]
        if ratio >= 1.02:
            rtxt, weight = f"{ratio:.2f}× I", ' font-weight="650"'
        elif ratio <= 0.98:
            rtxt, weight = f"{1 / ratio:.2f}× L", ""
        else:
            rtxt, weight = "even", ""
        parts.append(f'<text x="{W - gut_r + 8:.1f}" y="{cy + 4:.1f}" '
                     f'class="rowval"{weight}>{rtxt}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def tu_stride_chart():
    """Speedup over v1 across the fine stride curve, split build (48
    blocks), plain lattice rule — one line per scene."""
    order = ["S" + str(s) for s in TU["strides"]]
    series = []
    for r in TU_ROWS:
        v1 = r["configs"]["v1"]["ms"]
        pts = []
        for s in TU["strides"]:
            c = r["configs"].get(f"split_L{s}")
            if c is None:
                pts.append((None, ""))
                continue
            sp = v1 / c["ms"]
            tip = (f"{sd_label(r['scene'])} — split_L{s}: "
                   f"{fmt_ms(c['ms'])} ms ({sp:.2f}× vs v1) · "
                   f"{fmt_int(c['levels'])} levels · "
                   f"{fmt_int(c['candidates'])} candidates")
            pts.append((sp, tip))
        series.append((sd_label(r["scene"]),
                       _SCENE_SLOTS.get(r["scene"], "s1"), pts))
    return _curve_chart(order, series,
                        "Speedup over v1 across strides, split build",
                        (0.25, 0.5, 2, 4), lambda t: f"{t:g}×",
                        ref=1.0, ref_label="1× = v1")


def tu_build_chart():
    """The occupancy gap on one solid scene: raw ms across strides for
    the three builds, plain lattice rule, dashed line = v1."""
    r = next((x for x in TU_ROWS if x["scene"].startswith("two_sq")),
             TU_ROWS[0])
    order = ["S" + str(s) for s in TU["strides"]]
    v1 = r["configs"]["v1"]["ms"]
    series = []
    for b in ("fused", "r128", "split"):
        pts = []
        for s in TU["strides"]:
            c = r["configs"].get(f"{b}_L{s}")
            if c is None:
                pts.append((None, ""))
                continue
            blocks = TU["builds"].get(b, {}).get("coop_256", "?")
            tip = (f"{sd_label(r['scene'])} — {b}_L{s}: "
                   f"{fmt_ms(c['ms'])} ms ({blocks} blocks) · "
                   f"{fmt_int(c['levels'])} levels")
            pts.append((c["ms"], tip))
        series.append((_TU_BUILD_LABELS[b], _TU_BUILD_SLOTS[b], pts))
    all_ms = [v for _, _, pts in series for v, _ in pts if v is not None]
    lo, hi = min(all_ms + [v1]), max(all_ms + [v1])
    step = 20 if hi - lo > 60 else 10
    ticks = tuple(range(int(lo // step) * step,
                        int(hi // step) * step + step + 1, step))
    return _curve_chart(order, series,
                        f"Build comparison on {sd_label(r['scene'])}, "
                        f"ms across strides",
                        ticks, lambda t: f"{t:g}",
                        ref=v1, ref_label=f"v1: {fmt_ms(v1)}", log_y=False)


def tu_best_chart():
    series = [
        (lambda r: r["configs"]["v1"]["ms"], "v1 (corner rule)", "s7"),
        (lambda r: r["configs"]["ccl"]["ms"], "ccl_fill", "s8"),
        (lambda r: r["best_ms"], "best tuned config", "s6"),
    ]
    return log_dot_plot(TU_ROWS, series,
                        "v1 vs ccl_fill vs best tuned config per scene, "
                        "log scale",
                        lambda r: sd_label(r["scene"]))


_tu_regs_card = _tu_rule_card = _tu_stride_card = _tu_best_card = ""
if HAS_TUNING:
    _tu_fused = TU["builds"].get("fused", {})
    _tu_best_gain = max(TU_ROWS, key=lambda r: r["best_vs_v1"])
    _tu_disks = next((r for r in TU_ROWS
                      if r["scene"].startswith("two_disks")), None)

    _tu_regs_card = f"""
<div class="card">
<h2>One register cost half the GPU — the builds</h2>
<p class="note">The register file decides cooperative occupancy: 65,536
registers per SM ÷ 256 threads ÷ 2 blocks = 128 registers per thread,
exactly. The fused lattice kernel compiled to
{_tu_fused.get('regs', '?')} — ONE over the line — and dropped from 48
to {_tu_fused.get('coop_256', '?')} co-resident blocks: half the GPU
lost to a single register. Both fixes work and land within noise of
each other: r128 caps the compiler at 128 (it settles at
{TU['builds'].get('r128', {}).get('regs', '?')}, BELOW the cap — there
was room all along), and split keeps only the fill core cooperative
(back to {TU['builds'].get('split', {}).get('regs', '?')} regs, the
compress/flatten cleanup runs as plain full-occupancy kernels). Either
one beats the fused build by ~25–40% on the big solid scenes: occupancy
was the whole bottleneck, which fix you pick barely matters.</p>
{tu_regs_chart()}
<div class="tablewrap">{tu_regs_table()}</div>
</div>
"""

    _tu_rule_pitch = ""
    if _tu_disks:
        _dl = _tu_rule_best(_tu_disks, "L")
        _di = _tu_rule_best(_tu_disks, "I")
        _tu_rule_pitch = (
            f" On the disks the interior rule is worth "
            f"{_dl[1]['ms'] / _di[1]['ms']:.2f}× over the best plain "
            f"lattice ({_di[0]}: {fmt_ms(_di[1]['ms'])} ms vs {_dl[0]}: "
            f"{fmt_ms(_dl[1]['ms'])} ms) and "
            f"{_tu_disks['configs']['v1']['ms'] / _di[1]['ms']:.2f}× over "
            f"v1.")
    _tu_rule_card = f"""
<div class="card">
<h2>The interior rule — seeds inside the mass</h2>
<p class="note">The plain lattice (L) seeds every S-th red pixel — on a
rasterized disk that includes the staircase edge, where seeds buy
little. The interior rule (I) keeps a lattice seed only if all 8 of its
neighbors are red: seeds land inside the mass and the noisy boundary
stays seedless (the corner rule still runs underneath as the coverage
net, so single pixels and 1-px snakes are never lost). Amber dot = best
L cell, green dot = best I cell, any build and stride; gutter = the
ratio.{_tu_rule_pitch} And the serpentine is the safety net made
visible: a 1-px snake has NO interior pixels, so every I cell measures
identical to its L twin — "even".</p>
{legend([("best plain lattice (L)", "s4"), ("best interior (I)", "s5")])}
{tu_rule_chart()}
<p class="note">
<img src="{_WAVEFRONT_RELPATH}/disk192_corner_final.gif"
 alt="Corner-seeded disk replay: one wave sweeps radially through 140
 levels, light to dark." style="max-width:280px; border-radius:8px;
 margin:10px 12px 10px 0; vertical-align:top;">
<img src="{_WAVEFRONT_RELPATH}/disk192_interior_final.gif"
 alt="Interior-S1 replay of the same disk: the whole mass lights at
 level 0, only the staircase edge fills late — 3 levels total."
 style="max-width:280px; border-radius:8px; margin:10px 0;
 vertical-align:top;">
<br>The same disk twice. Left, corner seeding: one wave, 140 levels of
light→dark ramp. Right, interior S=1: every 8-neighbors-red pixel is a
seed, the whole mass is level 0 and only the staircase edge fills late
— 3 levels total.</p>
</div>
"""

    _tu_stride_card = f"""
<div class="card">
<h2>The stride curve — S8, and the S≥32 mystery</h2>
<p class="note">The fine curve with occupancy restored (split build,
plain lattice, one line per scene, speedup over v1 on a log axis). With
48 blocks back, the optimum moved from the coarse sweep's S16 to S8 on
every solid scene. And the old "S64 dip" turns out to be a broad slow
zone: EVERY S≥32 config sinks toward or below the v1 line even though
it runs 15–90× fewer levels. Fewer levels means fewer grid-wide
barriers — if less synchronization measures slower, the loss is in the
memory system, not the sync. That is now the clearest question this
chapter owes the ncu profiler.</p>
{_scene_legend(TU_ROWS)}
{tu_stride_chart()}
<p class="note">The occupancy gap itself, on the two-squares scene: raw
ms per stride for the three builds (light→dark = fused 24 blocks, r128,
split 48 blocks), dashed line = v1. r128 and split run near-tied and
~25–40% under the fused build at every useful stride.</p>
{legend([(_TU_BUILD_LABELS["fused"], "m1"), (_TU_BUILD_LABELS["r128"], "m3"),
         (_TU_BUILD_LABELS["split"], "m5")])}
{tu_build_chart()}
</div>
"""

    _tu_best_card = f"""
<div class="card">
<h2>Best per scene — and the recipe that falls out</h2>
<p class="note">Every scene's best tuned cell against the two
references. Best case: {_tu_best_gain['best_vs_v1']:.2f}× over v1
({_tu_best_gain['best_cfg']} on
{sd_label(_tu_best_gain['scene'])}). The practical recipe: a 48-block
build at S8 for big solid shapes · interior S1 for staircase-edged
shapes · S1 for thin snakes · plain v1 for dense noise. The winners
stay variants for now — promoting a default (or choosing the stride
automatically from a cheap image statistic) is Chapter 6 material.
Full per-cell data below and in the tuning JSON/CSV (53 configs × 7
scenes).</p>
{legend([("v1 (corner rule)", "s7"), ("ccl_fill", "s8"),
         ("best tuned config", "s6")])}
{tu_best_chart()}
<details><summary>Best per build and scene — the cross-product
table</summary>
<div class="tablewrap">{tu_table()}</div></details>
</div>
"""

if HAS_SEED:
    _sd_most_blobs = max(SD_ROWS, key=lambda r: r["n_blobs"])
    _sd_biggest = max(SD_ROWS, key=lambda r: r["filled"])
    _sd_merge_wins = sum(1 for r in SD_ROWS if r["merge_vs_ccl"] > 1)
    _sd_taxes = [(r, r["discovery_tax_merge"]) for r in SD_ROWS
                 if r.get("discovery_tax_merge")]
    _sd_best_variant = ("seed_merge" if _sd_merge_wins > len(SD_ROWS) / 2
                        else "ccl_fill")
    SD_TILES = [
        (f"{fmt_int(_sd_most_blobs['n_blobs'])} blobs",
         f"discovered, labeled and filled in ONE launch · "
         f"{sd_label(_sd_most_blobs['scene'])} · zero seeds given"),
        (f"{_sd_merge_wins}/{len(SD_ROWS)}",
         "scenes where seed_merge (discovery riding inside the fill) "
         "beat the ccl_fill prepass"),
    ]
    if _sd_taxes:
        _tax_lo = min(t for _, t in _sd_taxes)
        _tax_hi = max(t for _, t in _sd_taxes)
        SD_TILES.append(
            (f"{_tax_lo:.2f}×–{_tax_hi:.2f}×",
             "discovery tax: kernel time vs ch04's given-seeds "
             "multisource (8-conn) on the two-blob scenes"))
    if HAS_TUNING:
        _t_fregs = TU["builds"].get("fused", {}).get("regs")
        if _t_fregs and _t_fregs > 128:
            SD_TILES.append(
                (f"{_t_fregs} vs 128",
                 "one register over the two-blocks-per-SM line cost the "
                 "lattice kernel half the GPU — fixed two ways in the "
                 "cross-product cards"))
        _t_disks = next((r for r in TU_ROWS
                         if r["scene"].startswith("two_disks")), None)
        if _t_disks:
            SD_TILES.append(
                (f"{_t_disks['best_vs_v1']:.2f}×",
                 f"the disks with {_t_disks['best_cfg']} — interior "
                 "seeding floods the mass at level 0, the staircase "
                 "edge stays seedless"))
    sd_tiles_html = "".join(
        f'<div class="tile"><div class="tile-v">{v}</div>'
        f'<div class="tile-l">{l}</div></div>' for v, l in SD_TILES)
else:
    sd_tiles_html = ""


# The conditional project tiles this chapter contributes.
TILES = []
if HAS_SEED:
    TILES.append(
        (f"{fmt_int(_sd_most_blobs['n_blobs'])} blobs",
         "no seeds given: discovered, labeled and filled in one "
         "cooperative launch — see §3",
         "seed-discovery stage"))
if HAS_TUNING:
    _t_fregs = TU["builds"].get("fused", {}).get("regs")
    if _t_fregs and _t_fregs > 128:
        TILES.append(
            (f"{_t_fregs} vs 128",
             "one register cost half the GPU — found and fixed in the "
             "tuning cross-product, see §3",
             "tuning cross-product"))


LEG_SD = legend([("@njit CPU (CCL + fill)", "s2"), ("seed_merge", "s7"),
                 ("ccl_fill", "s8")])

_sd_section = ""
if HAS_SEED:
    _sd_wavefront_note = (
        '<p class="note">'
        '<img src="' + _WAVEFRONT_RELPATH + '/u192_merge_prov.gif" '
        'alt="Provisional-label replay on a U-shaped blob: two candidate '
        'waves in different colors race from the arm tips and collide at '
        'the bridge." style="max-width:300px; border-radius:8px; '
        'margin:10px 12px 10px 0; vertical-align:top;">'
        '<img src="' + _WAVEFRONT_RELPATH + '/u192_merge_final.gif" '
        'alt="Final-label replay of the same run: one color — the union '
        'erased the seam." style="max-width:300px; border-radius:8px; '
        'margin:10px 0; vertical-align:top;">'
        '<br>The merge made visible: the same seed_merge run replayed '
        'twice. Left, each pixel hued by its PROVISIONAL label — the two '
        'candidate waves race from the arm tips and slam together at the '
        'bridge, where the CAS loser fires the union. Right, the same '
        'depth clock hued by the CANONICAL label: one blob, no seam. '
        'More replays (24-tooth comb, 100-blob grid, the 1,858-blob '
        'random shatter) in <code>results/ch05_gpu_nblob_nblock/'
        'wavefront/</code>.</p>')
    _sd_section = f"""
<h2 class="domain-h">3. Seed discovery</h2>
<p class="sub">Nobody passes seeds anymore: the GPU is handed an image
and must find every blob itself — one canonical seed and label per blob
(its minimum linear index), any number of blobs, one cooperative launch,
8-connectivity. Two strategies compared head-to-head: seed_merge floods
from every locally-detectable candidate and merges colliding waves'
labels in flight (atomicMin union-find); ccl_fill solves connectivity
first with a data-independent union-find pass, then fills from exactly
one seed per blob. Both retire ch04's in-entry label format (64-blob
cap) for a per-pixel label map — priced, not free — and both drop the
n_seeds parameter: the GPU-produced seed count is read behind a
grid.sync fence sandwich, the refined form of ch04's deadlock lesson.</p>

<div class="tiles">{sd_tiles_html}</div>

<div class="card">
<h2>Runtime per scene — discovery included</h2>
<p class="note">Log scale. The @njit baseline runs the same job
sequentially (CCL + canonical fill, discovery included). Hover any dot;
exact numbers in the table below.</p>
{LEG_SD}
{sd_runtime_chart()}
</div>

<div class="card">
<h2>Where should connectivity be solved — in flight or up front?</h2>
<p class="note">Two solid dots per scene: two distinct mechanisms, not
variants of one entity. Gutter: ccl/merge — above 1.00× the colliding-
waves bet wins, below it the CCL prepass does. The interesting spread is
scene shape: many tiny blobs (candidates ≈ blobs) reward the prepass;
few big blobs reward waves that discover while they fill.</p>
{sd_ab_chart()}
</div>

{_sw_card}

{_tu_regs_card}

{_tu_rule_card}

{_tu_stride_card}

{_tu_best_card}

{_sd_wavefront_note}

<div class="card">
<h2>All numbers — seed-discovery stage</h2>
<p class="note">'scan', 'flatten' and 'ccl union' are IN-KERNEL phase
wall times: tid-0 %globaltimer stamps at the grid.sync phase
boundaries, medians over the same interleaved rounds (no
separate-launch subtraction caveat). The discovery-tax columns compare
against ch04's multisource conn8 kernel WITH host-given seeds on the
ch04-comparable scenes — the price of finding out vs being told. GB/s
is the ch05 derived model (discovery + label-map terms included, see
the benchmark JSON's bandwidth_model note) against the measured
{SD_PEAK:.0f} GB/s copy peak.</p>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{sd_table()}</div></details>
</div>
"""

SECTION_3 = _sd_section
