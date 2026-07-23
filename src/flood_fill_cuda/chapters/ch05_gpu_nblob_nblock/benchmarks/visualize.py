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
    W, ROW_H, GUT_L, fmt_ms, fmt_int, legend, log_dot_plot,
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
            "<th>(min)</th><th>scan ms</th><th>ccl-pass ms</th>"
            "<th>tax merge</th><th>tax ccl</th>"
            "<th>vs @njit</th><th>GB/s (merge)</th><th>% peak</th></tr>")
    body = []
    for r in SD_ROWS:
        tax_m = r.get("discovery_tax_merge")
        tax_c = r.get("discovery_tax_ccl")
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
            f"<td>{fmt_ms(r['scan_ms'])}</td>"
            f"<td>{fmt_ms(r['cclp_ms'])}</td>"
            f"<td>{f'{tax_m:.2f}×' if tax_m else '—'}</td>"
            f"<td>{f'{tax_c:.2f}×' if tax_c else '—'}</td>"
            f"<td>{max(r['speedup_merge_vs_njit'], r['speedup_ccl_vs_njit']):.1f}×</td>"
            f"<td>{r['merge_model_gb_s']:.1f}</td>"
            f"<td>{r['merge_pct_of_peak']:.1f}</td>"
            "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


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
    sd_tiles_html = "".join(
        f'<div class="tile"><div class="tile-v">{v}</div>'
        f'<div class="tile-l">{l}</div></div>' for v, l in SD_TILES)
else:
    sd_tiles_html = ""


# The single conditional project tile this chapter contributes.
TILES = []
if HAS_SEED:
    TILES.append(
        (f"{fmt_int(_sd_most_blobs['n_blobs'])} blobs",
         "no seeds given: discovered, labeled and filled in one "
         "cooperative launch — see §3",
         "seed-discovery stage"))


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

{_sd_wavefront_note}

<div class="card">
<h2>All numbers — seed-discovery stage</h2>
<p class="note">'scan' and 'ccl-pass' are the standalone discovery
phases (fused − phase ≈ fill, with the separate-launch subtraction
caveat). The discovery-tax columns compare against ch04's multisource
conn8 kernel WITH host-given seeds on the ch04-comparable scenes — the
price of finding out vs being told. GB/s is the ch05 derived model
(discovery + label-map terms included, see the benchmark JSON's
bandwidth_model note) against the measured {SD_PEAK:.0f} GB/s copy
peak.</p>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{sd_table()}</div></details>
</div>
"""

SECTION_3 = _sd_section
