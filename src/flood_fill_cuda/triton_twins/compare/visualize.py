"""Triton twins dashboard renderer: Numba vs Triton (section 5).

Same contract as every chapter's renderer: computed at import from the
committed JSON, optional (no results/triton_twins/<unit>/compare_*.json
means SECTION_5 is the empty string and the dashboard renders without
it), and it grows by itself: a new unit or a new compare run shows up
the next time the dashboard is assembled.

Cards: the headline tiles; every like-for-like row of every unit as one
strip per unit; the per-unit table; the two translation choices
(first translation vs the twin as shipped, on the same paired cells);
the grand table cell by cell; the run history of every unit; and, folded
away per unit, a strip per experiment plus every row.

One colour rule for the whole section: speedup = numba_ms / triton_ms,
blue (--s1) where Triton is faster, red (--s7) where Numba is, and the
deeper the colour the bigger the gap. Rows that are not like-for-like,
first-translation ablations and repeated cells are never averaged; the
summary module owns those rules and this one only draws them.
"""

import glob
import html
import json
import math
import os

from ...shared.viz import fmt_ms, fmt_int
from .figures import UNITS
from .summary import (
    TWINS_ROOT, ablations, geomean, newest_per_unit, summarize,
    summarize_rows, _is_ablation, _is_duplicate,
)

NAMES = dict(UNITS)
W = 860
TOP_STRIP_LABEL_W = 170

# config keys worth showing in a tooltip, in this order
CFG_KEYS = ("column", "variant", "kernel", "mode", "contract", "runner",
            "probe", "connectivity", "radius", "lattice", "interior",
            "build", "entry_format", "tpb", "threads_per_block", "blocks",
            "enqueue", "lane_sched", "lane_schedule", "bare", "placement")


def _esc(s):
    return html.escape(str(s), quote=True)


def _cfg_short(cfg):
    parts = []
    for k in CFG_KEYS:
        if k in cfg and cfg[k] not in (None, False, ""):
            v = cfg[k]
            if isinstance(v, dict):
                v = "/".join(f"{a}:{b}" for a, b in v.items())
            parts.append(f"{k}={v}" if k not in ("column", "variant", "kernel",
                                                  "mode", "runner") else str(v))
    return " ".join(parts[:6])


def _counted(r):
    """The rows the summary averages."""
    return ("error" not in r and r.get("comparable", True)
            and not _is_ablation(r) and not _is_duplicate(r)
            and r.get("speedup_kernel"))


def _x(r):
    return f"x{r:.2f}"


def _tip(r):
    n = r["numba"]["kernel_ms"]["median"]
    t = r["triton"]["kernel_ms"]["median"]
    est = r.get("est") or r.get("config", {}).get("est")
    return (f"{r['experiment']} | {r['scene']} | {_cfg_short(r.get('config', {}))}"
            f" || numba {fmt_ms(n)} ms, triton {fmt_ms(t)} ms, "
            f"{_x(r['speedup_kernel'])}"
            f"{' (estimated per blob)' if est else ''}")


def _pol(s):
    return "var(--s1)" if s > 1 else "var(--s7)"


def _strength(s, full=2.0):
    return min(abs(math.log2(s)) / full, 1.0)


# ------------------------------------------------------------ strip chart

def _strips(groups, aria, label_w=TOP_STRIP_LABEL_W):
    """groups = [(label, sublabel, rows)]: one log2 strip per group."""
    groups = [(a, b, rows) for a, b, rows in groups if rows]
    if not groups:
        return ""
    vals = [r["speedup_kernel"] for _, _, rows in groups for r in rows]
    lo = min(0.5, 2.0 ** math.floor(math.log2(max(min(vals), 1 / 64))))
    hi = max(2.0, 2.0 ** math.ceil(math.log2(min(max(vals), 64))))
    row_h, pad_r, top = 34, 64, 8
    n = len(groups)
    h = top + n * row_h + 30
    span = W - label_w - pad_r

    def x_of(v):
        v = min(max(v, lo), hi)
        return label_w + (math.log2(v) - math.log2(lo)) / (
            math.log2(hi) - math.log2(lo)) * span

    p = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{_esc(aria)}">']
    bottom = top + n * row_h
    t = lo
    while t <= hi * 1.0001:
        x = x_of(t)
        cls = "tw-one" if abs(t - 1) < 1e-9 else "grid"
        p.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                 f'class="{cls}"/>')
        p.append(f'<text x="{x:.1f}" y="{bottom + 16}" class="tick" '
                 f'text-anchor="middle">x{t:g}</text>')
        t *= 2
    for i, (lab, sub, rows) in enumerate(groups):
        cy = top + i * row_h + row_h / 2
        p.append(f'<line x1="{label_w}" y1="{cy:.1f}" x2="{W - pad_r}" '
                 f'y2="{cy:.1f}" class="rowline"/>')
        p.append(f'<text x="{label_w - 10}" y="{cy + 1:.1f}" class="rowlab" '
                 f'text-anchor="end">{_esc(lab)}</text>')
        p.append(f'<text x="{label_w - 10}" y="{cy + 13:.1f}" class="tick" '
                 f'text-anchor="end">{_esc(sub)}</text>')
        for j, r in enumerate(sorted(rows, key=lambda r: r["speedup_kernel"])):
            s = r["speedup_kernel"]
            dy = ((j * 7) % 9 - 4) * 1.6
            p.append(f'<circle cx="{x_of(s):.1f}" cy="{cy + dy:.1f}" r="3.6" '
                     f'class="tw-dot" style="fill:{_pol(s)}" '
                     f'data-tip="{_esc(_tip(r))}"/>')
        g = geomean(r["speedup_kernel"] for r in rows)
        p.append(f'<rect x="{x_of(g) - 1.5:.1f}" y="{cy - 11:.1f}" width="3" '
                 f'height="22" rx="1.5" class="tw-mean" '
                 f'data-tip="{_esc(lab)}: geometric mean {_x(g)} over '
                 f'{len(rows)} rows"/>')
        p.append(f'<text x="{W - pad_r + 8}" y="{cy + 4:.1f}" class="tw-val">'
                 f'{_x(g)}</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------ dumbbell

def _dumbbell(items):
    """items = [(label, paired, first, final)]."""
    if not items:
        return ""
    vals = [v for _, _, a, b in items for v in (a, b)]
    lo = min(0.125, 2.0 ** math.floor(math.log2(min(vals))))
    hi = max(2.0, 2.0 ** math.ceil(math.log2(max(vals))))
    label_w, row_h, pad_r, top = 230, 30, 130, 8
    h = top + len(items) * row_h + 30
    span = W - label_w - pad_r

    def x_of(v):
        return label_w + (math.log2(v) - math.log2(lo)) / (
            math.log2(hi) - math.log2(lo)) * span

    p = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="First '
         f'translation versus refined twin">']
    bottom = top + len(items) * row_h
    t = lo
    while t <= hi * 1.0001:
        x = x_of(t)
        cls = "tw-one" if abs(t - 1) < 1e-9 else "grid"
        p.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                 f'class="{cls}"/>')
        p.append(f'<text x="{x:.1f}" y="{bottom + 16}" class="tick" '
                 f'text-anchor="middle">x{t:g}</text>')
        t *= 2
    for i, (lab, n, first, final) in enumerate(items):
        cy = top + i * row_h + row_h / 2
        x0, x1 = x_of(first), x_of(final)
        p.append(f'<text x="{label_w - 10}" y="{cy + 4:.1f}" class="rowlab" '
                 f'text-anchor="end">{_esc(lab)}</text>')
        p.append(f'<line x1="{x0:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" '
                 f'y2="{cy:.1f}" class="pairline-db"/>')
        p.append(f'<circle cx="{x0:.1f}" cy="{cy:.1f}" r="5.5" '
                 f'class="tw-hollow" style="stroke:{_pol(first)}" '
                 f'data-tip="{_esc(lab)}: first translation {_x(first)} over '
                 f'{n} paired cells"/>')
        p.append(f'<circle cx="{x1:.1f}" cy="{cy:.1f}" r="6" class="tw-dot" '
                 f'style="fill:{_pol(final)}" data-tip="{_esc(lab)}: twin as '
                 f'shipped {_x(final)} over the same {n} cells"/>')
        p.append(f'<text x="{W - pad_r + 8}" y="{cy + 4:.1f}" class="tw-val">'
                 f'{_x(first)} → {_x(final)}</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------ grand table

def _chapter(c):
    return int(c[2:4]) if c[:2] == "ch" and c[2:4].isdigit() else 99


def _heatmap(doc):
    rows = doc["rows"]
    scenes, cols = [], []
    for r in rows:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])
        if r["experiment"] not in cols:
            cols.append(r["experiment"])
    skipped = doc.get("meta", {}).get("skipped_cells", [])
    for s in skipped:
        if s.get("row") not in scenes:
            scenes.append(s["row"])
        if s.get("column") not in cols:
            cols.append(s["column"])
    cols = sorted(cols, key=lambda c: (_chapter(c), cols.index(c)))
    cell = {(r["scene"], r["experiment"]): r for r in rows if "error" not in r}
    skip = {(s["row"], s["column"]): s.get("reason", "") for s in skipped}
    cw, chh, lab_w, top = 36, 22, 104, 118
    w = lab_w + cw * len(cols) + 60  # room for the last rotated labels
    h = top + chh * len(scenes) + 6
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Grand table, '
         f'Numba versus Triton per cell">']
    for j, c in enumerate(cols):
        x = lab_w + j * cw + cw / 2
        p.append(f'<text x="{x:.1f}" y="{top - 6}" class="tick" '
                 f'transform="rotate(-55 {x:.1f} {top - 6})">{_esc(c)}</text>')
    for i, sc in enumerate(scenes):
        y = top + i * chh
        p.append(f'<text x="{lab_w - 6}" y="{y + chh / 2 + 4}" class="rowlab" '
                 f'text-anchor="end">{_esc(sc)}</text>')
        for j, c in enumerate(cols):
            x = lab_w + j * cw
            if (sc, c) in cell:
                r = cell[(sc, c)]
                s = r["speedup_kernel"]
                op = 0.08 + 0.72 * _strength(s)
                dash = ("" if r.get("comparable", True)
                        else ' stroke="var(--muted)" stroke-dasharray="2 2"')
                est = r.get("est") or r.get("config", {}).get("est")
                tip = _tip(r) + ("" if r.get("comparable", True)
                                 else " | not like-for-like")
                p.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" '
                         f'height="{chh - 2}" rx="3" style="fill:{_pol(s)};'
                         f'fill-opacity:{op:.2f}"{dash} data-tip="{_esc(tip)}"/>')
                lab = f"{s:.2f}" if s < 9.95 else f"{s:.0f}"
                p.append(f'<text x="{x + cw / 2}" y="{y + chh / 2 + 3.5}" '
                         f'class="tw-cell" text-anchor="middle">'
                         f'{lab}{"*" if est else ""}</text>')
            elif (sc, c) in skip:
                p.append(f'<text x="{x + cw / 2}" y="{y + chh / 2 + 3.5}" '
                         f'class="tw-cell tw-skip" text-anchor="middle" '
                         f'data-tip="{_esc(sc)} | {_esc(c)}: skipped on both '
                         f'sides ({_esc(skip[(sc, c)])})">-</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------ tables

def _unit_table(units):
    head = ("<tr><th>unit</th><th>rows</th><th>like-for-like</th>"
            "<th>outputs equal</th><th>Triton faster</th><th>kernel</th>"
            "<th>total</th><th>best row</th><th>worst row</th></tr>")
    body = []
    for key, u in units.items():
        lfl = u["rows"] - u["ablation_rows"] - u["duplicate_rows"] - u[
            "not_comparable_rows"] - len(u["errors"])
        bt, bn = u.get("best_for_triton"), u.get("best_for_numba")
        body.append(
            f"<tr><td>{_esc(NAMES.get(key, key))}</td><td>{fmt_int(u['rows'])}</td>"
            f"<td>{fmt_int(lfl)}</td>"
            f"<td>{'yes' if u['all_outputs_equal'] else '<b>NO</b>'}</td>"
            f"<td>{u.get('triton_faster_rows', 0)}</td>"
            f"<td><b>{_x(u['geomean_speedup_kernel'])}</b></td>"
            f"<td>{_x(u['geomean_speedup_total'])}</td>"
            f"<td data-tip=\"{_esc(bt['experiment'])} | {_esc(bt['scene'])}\">"
            f"{_x(bt['speedup'])}</td>"
            f"<td data-tip=\"{_esc(bn['experiment'])} | {_esc(bn['scene'])}\">"
            f"{_x(bn['speedup'])}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _rows_table(rows):
    head = ("<tr><th>experiment</th><th>scene</th><th>config</th>"
            "<th>numba ms</th><th>triton ms</th><th>kernel</th><th>total</th>"
            "<th>equal</th><th>counted</th></tr>")
    body = []
    for r in rows:
        if "error" in r:
            body.append(f"<tr><td>{_esc(r['experiment'])}</td>"
                        f"<td>{_esc(r['scene'])}</td><td colspan=7>"
                        f"{_esc(r['error'][:120])}</td></tr>")
            continue
        why = ("yes" if _counted(r) else "first translation"
               if _is_ablation(r) else "repeat" if _is_duplicate(r)
               else "not like-for-like")
        body.append(
            f"<tr><td>{_esc(r['experiment'])}</td><td>{_esc(r['scene'])}</td>"
            f"<td style='text-align:left'>{_esc(_cfg_short(r.get('config', {})))}</td>"
            f"<td>{fmt_ms(r['numba']['kernel_ms']['median'])}</td>"
            f"<td>{fmt_ms(r['triton']['kernel_ms']['median'])}</td>"
            f"<td style='color:{_pol(r['speedup_kernel'])}'>"
            f"{_x(r['speedup_kernel'])}</td>"
            f"<td>{_x(r['speedup_total'])}</td>"
            f"<td>{'yes' if r.get('outputs_equal') else '<b>NO</b>'}</td>"
            f"<td>{why}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _history():
    """Every compare run of every unit: how the numbers moved over time."""
    head = ("<tr><th>unit</th><th>run (UTC)</th><th>commit</th><th>rows</th>"
            "<th>kernel</th><th>total</th><th>outputs equal</th></tr>")
    body = []
    for key, name in UNITS:
        paths = sorted(glob.glob(os.path.join(TWINS_ROOT, key, "compare_*.json")))
        for i, path in enumerate(paths):
            with open(path) as f:
                doc = json.load(f)
            s = summarize_rows(doc["rows"])
            g, gt = s.get("geomean_speedup_kernel"), s.get("geomean_speedup_total")
            newest = i == len(paths) - 1
            body.append(
                f"<tr><td>{_esc(name) if i == 0 else ''}</td>"
                f"<td>{_esc(doc['created_utc'])}{' (current)' if newest else ''}</td>"
                f"<td><code>{_esc(doc.get('git_commit') or '')}</code></td>"
                f"<td>{fmt_int(len(doc['rows']))}</td>"
                f"<td>{_x(g) if g else '-'}</td><td>{_x(gt) if gt else '-'}</td>"
                f"<td>{'yes' if s['all_outputs_equal'] else '<b>NO</b>'}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


# ------------------------------------------------------------ assemble

STYLE = """
<style>
.tw-dot { stroke: var(--surface-1); stroke-width: 1; fill-opacity: 0.7; }
.tw-hollow { fill: var(--surface-1); stroke-width: 2.2; }
.tw-mean { fill: var(--ink-2); }
.tw-one { stroke: var(--muted); stroke-width: 1.5; }
.tw-val { font-size: 12px; font-weight: 650; fill: var(--ink);
          font-variant-numeric: tabular-nums; }
.tw-cell { font-size: 9.5px; fill: var(--ink); font-variant-numeric: tabular-nums;
           pointer-events: none; }
.tw-skip { fill: var(--muted); pointer-events: auto; }
.tw-wide { overflow-x: auto; }
</style>
"""

TILES = []
SECTION_5 = ""

_SUM = summarize() if newest_per_unit() else None
if _SUM and _SUM["units"]:
    _units = {k: _SUM["units"][k] for k, _ in UNITS if k in _SUM["units"]}
    _units.update({k: v for k, v in _SUM["units"].items() if k not in _units})
    _docs = {}
    for _k, _p in newest_per_unit().items():
        with open(_p) as _f:
            _docs[_k] = json.load(_f)
    _all = [r for d in _docs.values() for r in d["rows"]]
    _counted_rows = [r for r in _all if _counted(r)]
    _ov = _SUM["overall"]
    _first = []  # the oldest run per unit, for the first-translation tile
    for _k in _units:
        _paths = sorted(glob.glob(os.path.join(TWINS_ROOT, _k, "compare_*.json")))
        if len(_paths) > 1:
            with open(_paths[0]) as _f:
                _g = summarize_rows(json.load(_f)["rows"]).get(
                    "geomean_speedup_kernel")
            if _g:
                _first.append((_k, _g, _units[_k]["geomean_speedup_kernel"]))

    TILES = [(_x(_ov["geomean_of_unit_geomeans_kernel"]),
              "Triton vs Numba, kernel time",
              f"triton_twins · {len(_units)} units, "
              f"{fmt_int(len(_all))} rows, outputs identical")]
    _tiles = [
        (_x(_ov["geomean_of_unit_geomeans_kernel"]),
         "kernel time, Numba / Triton",
         "geometric mean of the unit means; launch path included"),
        (fmt_int(len(_all)),
         "rows timed" + (", outputs identical on every run"
                         if _ov["all_outputs_equal"] else ", OUTPUTS DIFFER"),
         f"{fmt_int(len(_counted_rows))} like-for-like, "
         f"{sum(r['speedup_kernel'] > 1 for r in _counted_rows)} faster in Triton"),
        (_x(_ov["geomean_of_unit_geomeans_total"]),
         "total time, Numba / Triton",
         "alloc + copies + kernel; mostly CuPy's pool, not kernels"),
    ]
    if _first:
        _tiles.append((f"{_x(geomean(a for _, a, _ in _first))} → "
                       f"{_x(geomean(b for _, _, b in _first))}",
                       "first translation → twin as shipped",
                       f"the {len(_first)} units measured twice, oldest vs "
                       f"newest run"))
    if "overview" in _units:
        _o = _units["overview"]
        _tiles.append((_x(_o["geomean_speedup_kernel"]),
                       "the grand table, cell by cell",
                       f"{_o.get('triton_faster_rows', 0)} cells faster in "
                       f"Triton"))

    _abl = []
    for _k, _d in _docs.items():
        for _e, _a in (ablations(_d["rows"]) or {}).items():
            if _a and _a["paired"]:
                _abl.append((f"{NAMES.get(_k, _k)}, {_e.replace('_', ' ')}",
                             _a["paired"],
                             _a["geomean_speedup_kernel_first_translation"],
                             _a["geomean_speedup_kernel_default"]))

    _per_unit = []
    for _k, _u in _units.items():
        if _k == "overview":
            continue
        _rows = _docs[_k]["rows"]
        _exps = []
        for _r in _rows:
            if _r["experiment"] not in _exps:
                _exps.append(_r["experiment"])
        _groups = [(_e, f"{sum(1 for r in _rows if r['experiment'] == _e and _counted(r))} rows",
                    [r for r in _rows if r["experiment"] == _e and _counted(r)])
                   for _e in _exps]
        _per_unit.append(f"""
<details><summary>{_esc(NAMES.get(_k, _k))}: {_x(_u['geomean_speedup_kernel'])}
over {sum(1 for r in _rows if _counted(r))} like-for-like rows
({_esc(os.path.basename(newest_per_unit()[_k]))})</summary>
{_strips(_groups, f"{_k} per experiment", label_w=220)}
<div class="tablewrap">{_rows_table(_rows)}</div>
</details>""")

    _grand = ""
    if "overview" in _docs:
        _grand = f"""
<div class="card">
<h2>The grand table, cell by cell</h2>
<p class="note">The overview's scenes (rows) and every GPU column of the
Numba grand table, each cell measured on both backends in one session.
The value is numba_ms / triton_ms; blue means Triton is faster, red means
Numba is, deeper means a bigger gap (full at 4x). * = estimated per blob
with the same sample on both sides; dashed = not like-for-like; - =
skipped on both sides, as the Numba table skips it. Hover a cell for the
two timings.</p>
<div class="tw-wide">{_heatmap(_docs["overview"])}</div>
</div>"""

    SECTION_5 = f"""
{STYLE}
<h2 class="domain-h">5. Numba vs Triton - the same chapters, twice</h2>

<p class="note" style="max-width:900px">Every Numba kernel of chapters 0-6
and the scan experiment has a Triton twin: the same algorithm, the same
host API and the same tests against the same CPU oracles
(<code>src/flood_fill_cuda/triton_twins/</code>). Each unit's
<code>compare.py</code> times both backends on the same cells, swapping
which one runs first every round, and checks that their deterministic
outputs are identical on every run. <strong>Speedup =
numba_ms / triton_ms</strong>: above x1, Triton is faster. kernel_ms is
each driver's own bracket, the Python launch path included. The full
write-up is <code>triton_twins/README.md</code>.</p>

<div class="tiles">{"".join(
    f'<div class="tile"><div class="tile-v">{v}</div>'
    f'<div class="tile-l">{l}</div><div class="tile-src">{s}</div></div>'
    for v, l, s in _tiles)}</div>

<div class="card">
<h2>Every like-for-like row, per unit</h2>
<p class="note">One dot per row, the bar is the unit's geometric mean
(the number on the right). Rows that are not like-for-like (different
thread counts or grids), first-translation ablations and repeated cells
are left out here, exactly as in the averages. Hover a dot for the row.</p>
{_strips([(NAMES.get(k, k), f"{sum(1 for r in _docs[k]['rows'] if _counted(r))} rows",
           [r for r in _docs[k]['rows'] if _counted(r)]) for k in _units],
         "Numba versus Triton, every like-for-like row")}
<div class="tablewrap">{_unit_table(_units)}</div>
</div>

<div class="card">
<h2>The two translation choices</h2>
<p class="note">The first faithful translation lost on average. Two
choices made most of the difference: a program-wide
<code>tl.cumsum</code> enqueue (about 7 CTA barriers per direction;
a per-lane <code>tl.atomic_add</code> compiles to Numba's own
warp-aggregated pattern) and lockstep union-find loops (per-lane state
machines are the closer twin of SIMT threads). Both first translations
stay behind a switch and are measured here on the same cells as the
twin as shipped: hollow = first translation, filled = as shipped.</p>
{_dumbbell(_abl)}
</div>

{_grand}

<div class="card">
<h2>Run history</h2>
<p class="note">Every comparison run per unit, oldest first; the
dashboard always charts the current (newest) one. Each run is one
session, so runs differ by clock drift as well as by code.</p>
<div class="tablewrap">{_history()}</div>
</div>

<div class="card">
<h2>Every row, per unit</h2>
<p class="note">Per experiment, the like-for-like rows; below each chart,
every row of the unit with its two timings and why it does or does not
count toward the averages.</p>
{"".join(_per_unit)}
</div>
"""
