"""
Chapter 4 dashboard renderer: the dual-blob stage (section 2) -- sequential
vs streams vs multisource, lin vs xy entry format, 4 vs 8 connectivity on
both mechanisms, and radius-2 on both mechanisms.

Extracted from the dashboard monolith
(chapters/ch02_gpu_1blob_2block/benchmarks/visualize.py). Self-contained:
loads its own dual_blob_*.json and dual_blob_radius2_*.json (both
optional -- this stage postdates the others, and a dashboard run
predating it should still render, just omitting section 2). Exposes
SECTION_2 (empty string if no data) and TILES (0 or 1 project tiles).
"""

import math
import os

from ....shared import results_paths
from ....shared.viz import (
    W, ROW_H, GUT_L, fmt_ms, fmt_int, legend, chain_strip,
    CHAIN_NOTE, log_dot_plot,
)

RESULTS_DIR = results_paths.results_dir("ch04_gpu_2blob_nblock", "benchmark_results")
_WAVEFRONT_DIR = results_paths.results_dir("ch04_gpu_2blob_nblock", "wavefront")
# Relative to the CONSUMING page's own output dir. This pass renders into
# ch02's benchmark_results/ (the dashboard is not split out yet); computed,
# not hardcoded, so it stays correct once that changes.
_CONSUMER_OUT_DIR = results_paths.results_dir("ch02_gpu_1blob_2block", "benchmark_results")
_WAVEFRONT_RELPATH = os.path.relpath(_WAVEFRONT_DIR, _CONSUMER_OUT_DIR)

DB_PATH = results_paths.newest_optional("dual_blob_*.json", RESULTS_DIR,
                                        exclude="radius2")
# The per-barrier work experiment (radius-2) writes its own JSON; optional,
# same contract as DB_PATH.
DBR2_PATH = results_paths.newest_optional("dual_blob_radius2_*.json", RESULTS_DIR)

import json

DB = None
if DB_PATH:
    with open(DB_PATH) as f:
        DB = json.load(f)
DBR2 = None
if DBR2_PATH:
    with open(DBR2_PATH) as f:
        DBR2 = json.load(f)

DB_ROWS = DB["scenes"] if DB else []
DB_PEAK = DB["measured_peak_gb_s"] if DB else 0.0
HAS_DUALBLOB = bool(DB_ROWS)

DBR2_ROWS = DBR2["scenes"] if DBR2 else []
HAS_DB_R2 = HAS_DUALBLOB and bool(DBR2_ROWS)

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


# The single conditional project tile this chapter contributes.
TILES = []
if HAS_DUALBLOB:
    TILES.append(
        (f"{_db_best_spd['speedup_multi_vs_seq']:.2f}×",
         f"two blobs, one pass · multisource vs sequential · "
         f"{db_label(_db_best_spd['scene'])} — see §2",
         "dual-blob stage"))


_db_2800 = next((r for r in DB_ROWS if r["scene"] == "two_sq_2800"), None)
CHAIN_DB = ([("CPU (@njit, 2 blobs)", _db_2800["njit_ms"]),
            ("sequential", _db_2800["seq_ms"]),
            ("multisource", _db_2800["multi_ms"])]
           if _db_2800 else None)


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
        '<p class="note"><img src="' + _WAVEFRONT_RELPATH + '/'
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

SECTION_2 = _db_section
