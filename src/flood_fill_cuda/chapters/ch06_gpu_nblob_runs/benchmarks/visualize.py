"""Chapter 6 dashboard renderer: runs instead of pixels (section 4).

Three cards: the head-to-head against ch05 on the same images, the
phase breakdown against the measured machine floors (which is where the
chapter's argument actually lives), and the full table.

Self-contained and optional, same contract as every other chapter's
renderer: no runs_*.json means SECTION_4 is the empty string and the
dashboard renders without it.
"""

import json
import os

from ....shared import results_paths
from ....shared.viz import (
    W, ROW_H, GUT_L, GUT_R, fmt_ms, fmt_int, legend, log_dot_plot,
)

RESULTS_DIR = results_paths.results_dir("ch06_gpu_nblob_runs",
                                        "benchmark_results")
RN_PATH = results_paths.newest_optional("runs_*.json", RESULTS_DIR)

RN = None
if RN_PATH:
    with open(RN_PATH) as f:
        RN = json.load(f)

ROWS = RN["scenes"] if RN else []
HAS = bool(ROWS)
PEAK = RN["measured_peak_gb_s"] if RN else 0.0
READ_PEAK = RN.get("measured_read_gb_s", 0.0) if RN else 0.0
WRITE_PEAK = RN.get("measured_write_gb_s", 0.0) if RN else 0.0

PHASES = ("pack", "count", "scan", "emit", "merge", "flatten", "paint")
PHASE_CLS = {"pack": "s1", "count": "s2", "scan": "s3", "emit": "s4",
             "merge": "s5", "flatten": "s6", "paint": "s7"}

LEG = legend([("ch05 BFS + union-find over pixels", "s1"),
              ("ch06 runs — RGB in, recolored in place", "s2"),
              ("ch06 runs — packed 1-bit mask in", "s3"),
              ("ch06 labeling only (nothing painted)", "s4")])


def _ch05_ms(r):
    c = r.get("ch05") or {}
    return c.get("median_ms")


def _headline():
    for r in ROWS:
        if r["scene"] == "input_blobs":
            return r
    return ROWS[0] if ROWS else None


def _runtime_chart():
    return log_dot_plot(
        ROWS,
        [(_ch05_ms, "ch05 best (split_L8)", "s1"),
         (lambda r: r["ch06"]["rgb"]["median_ms"], "ch06 rgb contract", "s2"),
         (lambda r: r["ch06"]["mask"]["median_ms"], "ch06 mask contract", "s3"),
         (lambda r: r["ch06"]["mask"]["label_only_ms"], "ch06 labeling only",
          "s4")],
        "Runtime per scene, ch05 versus ch06, log scale",
        lambda r: r["scene"])


def _phase_bars():
    """Stacked phase bars for the rgb contract, one row per scene, with
    the measured read+write floor drawn as a tick — the point being how
    much of the bar is the two ends and how little is the algorithm."""
    if not ROWS:
        return ""
    hi = max(sum(r["ch06"]["rgb"]["phase_ms"].get(p, 0.0) for p in PHASES)
             for r in ROWS) * 1.08
    n = len(ROWS)
    h = n * ROW_H + 34
    span = W - GUT_L - GUT_R
    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" '
             f'aria-label="Per-phase time breakdown, rgb contract">']
    for i, r in enumerate(ROWS):
        cy = i * ROW_H + ROW_H / 2
        ph = r["ch06"]["rgb"]["phase_ms"]
        x = GUT_L
        parts.append(f'<text x="{GUT_L - 10}" y="{cy + 4:.1f}" class="rowlab" '
                     f'text-anchor="end">{r["scene"]}</text>')
        for p in PHASES:
            v = ph.get(p, 0.0)
            if v <= 0:
                continue
            wpx = v / hi * span
            tip = (f"{r['scene']} — {p}: {fmt_ms(v)} ms "
                   f"({v / sum(ph.values()) * 100:.0f}%)")
            parts.append(
                f'<rect x="{x:.1f}" y="{cy - 9:.1f}" width="{max(wpx, 0.6):.1f}" '
                f'height="18" class="dot {PHASE_CLS[p]}" data-tip="{tip}"/>')
            x += wpx
        floor = (r["floor_ms"]["rgb_read"] + r["floor_ms"]["paint_write"])
        fx = GUT_L + floor / hi * span
        parts.append(
            f'<line x1="{fx:.1f}" y1="{cy - 13:.1f}" x2="{fx:.1f}" '
            f'y2="{cy + 13:.1f}" class="grid" stroke-dasharray="3,2" '
            f'data-tip="{r["scene"]} — machine floor (read RGB + write red '
            f'px at measured peak): {fmt_ms(floor)} ms"/>')
    parts.append(f'<text x="{W - GUT_R}" y="{n * ROW_H + 18}" class="tick" '
                 f'text-anchor="end">ms — dashed tick = measured '
                 f'read+write floor</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _table():
    head = ("<tr><th>scene</th><th>px</th><th>red px</th><th>runs</th>"
            "<th>blobs</th><th>mean run</th><th>ch05 ms</th>"
            "<th>ch06 rgb</th><th>ch06 mask</th><th>label only</th>"
            "<th>×rgb</th><th>×mask</th><th>rgb GB/s</th>"
            "<th>floor ms</th></tr>")
    body = []
    for r in ROWS:
        c5 = _ch05_ms(r)
        rgb, mask = r["ch06"]["rgb"], r["ch06"]["mask"]
        floor = r["floor_ms"]["rgb_read"] + r["floor_ms"]["paint_write"]
        body.append(
            f"<tr><td>{r['scene']}</td><td>{fmt_int(r['n_pixels'])}</td>"
            f"<td>{fmt_int(r['red_px'])}</td><td>{fmt_int(r['n_runs'])}</td>"
            f"<td>{fmt_int(r['n_blobs'])}</td>"
            f"<td>{r['mean_run_px']:.1f}</td>"
            f"<td>{fmt_ms(c5) if c5 else '—'}</td>"
            f"<td>{fmt_ms(rgb['median_ms'])}</td>"
            f"<td>{fmt_ms(mask['median_ms'])}</td>"
            f"<td>{fmt_ms(mask['label_only_ms'])}</td>"
            f"<td>{rgb.get('speedup_vs_ch05', 0):.1f}×</td>"
            f"<td>{mask.get('speedup_vs_ch05', 0):.1f}×</td>"
            f"<td>{rgb['model_gb_s']:.0f}</td>"
            f"<td>{fmt_ms(floor)}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


TILES = []
_section = ""

if HAS:
    _h = _headline()
    _c5 = _ch05_ms(_h) or 0.0
    TILES = [
        (f"{_h['ch06']['mask']['speedup_vs_ch05']:.0f}×",
         "runs vs pixels, same image",
         f"ch06 · {fmt_ms(_c5)} → {fmt_ms(_h['ch06']['mask']['median_ms'])} ms"),
        (fmt_ms(_h["ch06"]["mask"]["label_only_ms"]) + " ms",
         f"{fmt_int(_h['n_blobs'])} blobs labeled, {fmt_int(_h['n_pixels'])} px",
         "ch06 · discovery + canonical labels, packed input"),
    ]

    _section = f"""
<h2 class="domain-h">4. Runs, not pixels — the bandwidth floor</h2>

<p class="note" style="max-width:900px">Chapters 1–5 all move PIXELS: a
BFS frontier is a list of pixel indices, a union-find pass unions per
pixel adjacency, the label map is one int32 per pixel. This stage asks
what the same job costs when the unit is the <em>run</em> — a maximal
red span inside one row. <code>input_blobs.png</code> has 81,000,000
pixels, 13,451,960 of them red, and only <strong>539,207 runs</strong>:
150× fewer items than red pixels, and every fact the job needs
(connectivity, canonical labels, the spans to paint) is a fact about
runs. Connectivity becomes a union-find over 539k items and the clock
collapses onto the two ends that must touch pixels at all — one read
and one write. Canonical labels are unchanged from ch05 (a blob's label
is still its minimum linear index), so the same CPU oracle judges both,
pixel for pixel.</p>

<p class="note" style="max-width:900px"><strong>Two contracts, always
reported together.</strong> The RGB image is 243 MB and the packed
1-bit mask is 10.15 MB; at the measured read peak of
{READ_PEAK:.0f} GB/s, merely reading the RGB costs ~1.3 ms, so no
algorithm recolors from RGB in under a millisecond on this hardware.
The sub-millisecond numbers belong to the mask contract and say so.</p>

<div class="tiles">{"".join(
    f'<div class="tile"><div class="tile-v">{v}</div>'
    f'<div class="tile-l">{l}</div><div class="tile-src">{s}</div></div>'
    for v, l, s in TILES)}</div>

<div class="card">
<h2>Runtime per scene — ch05's pixels versus ch06's runs</h2>
<p class="note">Log scale, same images, same session, interleaved
rounds. ch05 runs its winning config on the headline image
(<code>split_L8</code>) everywhere, which is not its per-scene best —
read the non-PNG rows as indicative. Hover any dot.</p>
{LEG}
{_runtime_chart()}
</div>

<div class="card">
<h2>Where the time goes — and how close the floor is</h2>
<p class="note">The rgb contract, phase by phase. The dashed tick is the
MEASURED machine floor for that scene: read the RGB once at
{READ_PEAK:.0f} GB/s plus write the red pixels once at
{WRITE_PEAK:.0f} GB/s. Two phases are the bar — <code>pack</code> (the
only full-resolution read) and <code>paint</code> (the only write).
Everything between them — count, scan, emit, merge, flatten, i.e. the
entire connected-components problem — is the thin middle.</p>
{legend([(p, PHASE_CLS[p]) for p in PHASES])}
{_phase_bars()}
</div>

<div class="card">
<h2>All numbers — the run-table stage</h2>
<p class="note">'label only' is the pipeline minus the paint: every blob
discovered and given its canonical label, nothing recolored. GB/s is the
ch06 derived model (see the benchmark JSON's bandwidth_model note)
against a measured {PEAK:.0f} GB/s copy peak. 'floor ms' is
read-RGB + write-red-pixels at the measured read/write peaks — the
number no implementation of this contract can go below.</p>
<details open><summary>Per-scene results table</summary>
<div class="tablewrap">{_table()}</div></details>
</div>
"""

SECTION_4 = _section
