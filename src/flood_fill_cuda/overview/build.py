"""Build site/index.html — the project's one-page overview.

Abstract, the evolution chain, a glossary of every metric the project
measures, and the grand table: every chapter's every variant (columns)
on one common shape x scale grid (rows), cells = median kernel ms from
the newest overview_*.json. Fully self-contained: inline CSS (the
project design system from shared/viz.py), no scripts, no external
requests, native title tooltips.

Run:  uv run python -m flood_fill_cuda.overview.build
"""

import os
import re
from datetime import date

import json

from ..shared import results_paths
from ..shared.viz import CSS, fmt_ms, fmt_int

RESULTS_DIR = results_paths.results_dir("overview", "benchmark_results")
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_PATH = os.path.join(_REPO_ROOT, "site", "index.html")

SKIP_LABELS = {"na": "—", "overflow": "overflow", "capped": "capped",
               "unsupported": "n/s"}
SKIP_TIPS = {
    "na": "ch04's kernel takes exactly two blobs in two components — "
          "a one-blob scene is outside its input space",
    "overflow": "ch01's shared-memory ring exceeded its 8192-slot "
                "frontier capacity",
    "capped": "pure Python past the 20M red-pixel cap",
    "unsupported": "ch04 streams deadlocks after ch03's cooperative "
                   "kernels in the same process — documented, skipped",
}

EXTRA_CSS = """
.hero p { max-width: 68ch; }
.chain { display: flex; flex-wrap: wrap; align-items: center;
  gap: 6px; margin: 18px 0 6px; }
.chain .stage { border: 1px solid var(--grid); border-radius: 8px;
  padding: 6px 12px; font-weight: 650; white-space: nowrap; }
.chain .why { color: var(--muted); font-size: 11.5px; max-width: 110px;
  text-align: center; line-height: 1.25; }
.chain .arrow { color: var(--muted); }
dl.gloss { display: grid; grid-template-columns: max-content 1fr;
  gap: 6px 18px; margin: 8px 0; }
dl.gloss dt { font-weight: 650; white-space: nowrap; }
dl.gloss dd { margin: 0; color: var(--ink-2); max-width: 90ch; }
.grand th, .grand td { font-size: 12px; padding: 4px 8px; }
.grand td:first-child, .grand th:first-child { position: sticky; left: 0;
  background: var(--surface-1); z-index: 2; text-align: left;
  white-space: nowrap; }
.grand thead th { text-align: right; }
.grand thead .grp { text-align: center; border-bottom: 1px solid
  var(--grid); }
.grand .skip { color: var(--muted); }
.grand .best { font-weight: 700; }
.grand .est { color: var(--ink-2); font-style: italic; }
"""


def _load():
    path = results_paths.newest_optional("overview_*.json", RESULTS_DIR)
    if not path:
        raise SystemExit("no overview_*.json — run "
                         "`python -m flood_fill_cuda.overview.bench` first")
    with open(path) as f:
        return json.load(f), os.path.basename(path)


def _version():
    with open(os.path.join(_REPO_ROOT, "pyproject.toml")) as f:
        m = re.search(r'^version = "([^"]+)"', f.read(), re.M)
    return m.group(1) if m else "?"


def chain_html():
    stages = ["CPU BFS", "1 block", "2 blocks", "N blocks", "2 blobs",
              "N blobs"]
    whys = ["one core is serial", "one SM is ~4% of the GPU",
            "2 SMs are 8%", "one blob is one BFS", "who finds the seeds?"]
    parts = ['<div class="chain">']
    for i, s in enumerate(stages):
        parts.append(f'<span class="stage">{s}</span>')
        if i < len(whys):
            parts.append(f'<span class="why">{whys[i]}</span>'
                         '<span class="arrow">→</span>')
    parts.append("</div>")
    return "".join(parts)


GLOSSARY = [
    ("kernel ms", "Wall time of the GPU kernel(s) only — launch to "
     "cuda.synchronize — excluding host↔device copies and allocation. "
     "Every cell in the table. 'total ms' (not shown) would add those "
     "transfers."),
    ("median of N", "Each GPU cell is the median of 5 timed runs after "
     "one untimed warmup (JIT compilation and caches settle first); "
     "@njit cells are median of 3; pure Python is median of 3 up to "
     "300k px, a single run above."),
    ("blob / component", "A maximal group of red pixels connected to "
     "each other — the thing one flood fill fills."),
    ("seed", "A starting pixel handed to the kernel. Chapters 1–4 must "
     "be told their seeds; chapter 5's whole point is finding them on "
     "the GPU."),
    ("connectivity 4 / 8", "Which pixels count as neighbors: the 4 "
     "edge-sharing ones, or those plus the 4 diagonals. Chapters 1–2 "
     "are 4-conn, chapter 3 measures both, chapter 5 is 8-conn. Every "
     "scene in this table forms the same components either way, so the "
     "cells compare the same job."),
    ("radius 2", "A ch03 experiment: probe neighbors up to 2 pixels "
     "away, halving the level count on solid shapes at the cost of "
     "more probes per pixel."),
    ("level", "One BFS wavefront step: all pixels at the same distance "
     "from a seed. The GPU processes a whole level in parallel, then "
     "synchronizes; thin shapes (serpentine) have huge level counts "
     "and are the universal worst case."),
    ("cooperative launch", "A CUDA launch mode where every block is "
     "resident on the GPU at once so the whole grid can synchronize "
     "mid-kernel (grid.sync) — the barrier between BFS levels. Its "
     "price: a hard cap on blocks."),
    ("occupancy / coop blocks", "How many blocks the GPU can host "
     "simultaneously. At 256 threads/block the register file allows 2 "
     "blocks per SM only if a kernel stays ≤128 registers/thread — "
     "the line behind the ch05 'one register cost half the GPU' story "
     "(fused build: 129 regs → 24 blocks; r128/split builds → 48)."),
    ("union-find / canonical label", "Chapter 5 bookkeeping: when two "
     "flood waves from different provisional seeds collide, an "
     "atomicMin union-find merges their labels; the minimum linear "
     "index of each blob survives as its one canonical label."),
    ("seed_merge vs ccl_fill", "The two ch05 strategies: discover "
     "seeds locally and merge colliding waves in flight, versus solve "
     "all connectivity up front (CCL) and fill from exactly one seed "
     "per blob."),
    ("lattice stride S", "ch05 tuning: besides corner candidates, "
     "plant a seed at every S-th red pixel on an S×S grid. More seeds "
     "→ fewer levels → shorter fill, until union traffic wins. S8 "
     "proved optimal on big solid shapes."),
    ("interior rule (I)", "A lattice seed only counts if all 8 of its "
     "neighbors are red — seeds land inside the mass, not on noisy "
     "staircase edges. Wins on rasterized disks."),
    ("per-blob loop", "A one-blob kernel (ch01–ch03) doing a multi-blob "
     "job runs one call per blob. On two-blob rows the cell is the "
     "MEASURED sum of both calls — what using that stage would really "
     "cost. Kernel time only; each call would also pay allocation and "
     "copies."),
    ("≈ estimated", "On N-blob rows a full per-blob loop would take "
     "hours (755k calls), so ch01–ch04 cells are estimates: median "
     "per-call kernel ms over a k-blob sample × the number of calls "
     "(one per blob; one per PAIR for ch04). Italic with '≈'; never "
     "eligible to be a row winner and excluded from the crosscheck."),
    ("— / overflow / capped / n/s", "'—': the job cannot be expressed "
     "at all — ch04's kernel takes exactly two blobs in two distinct "
     "components, so one-blob scenes are outside its input space. "
     "'overflow': ch01's shared-memory ring exceeded its 8192-slot "
     "capacity. 'capped': pure Python past the 20M red-px cap. 'n/s': "
     "ch04's streams mode deadlocks after ch03's cooperative kernels "
     "have run in the same process — a real cross-chapter finding, "
     "documented in the bench source."),
]


def glossary_html():
    items = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARY)
    return f'<dl class="gloss">{items}</dl>'


def grand_table(data):
    cols = data["columns"]
    # group header with colspans
    groups = []
    for c in cols:
        if groups and groups[-1][0] == c["group"]:
            groups[-1][1] += 1
        else:
            groups.append([c["group"], 1])
    head1 = ('<tr><th rowspan="2">scene</th>'
             + "".join(f'<th class="grp" colspan="{n}">{g}</th>'
                       for g, n in groups) + "</tr>")
    head2 = ("<tr>" + "".join(f'<th>{c["label"]}</th>' for c in cols)
             + "</tr>")

    body = []
    for r in data["rows"]:
        label = f"{r['family']} · {r['note']}"
        cells = [f'<td title="{r["width"]}×{r["height"]}">{label}</td>']
        for c in cols:
            cell = r["cells"].get(c["key"], {"skip": "na"})
            skip = cell.get("skip")
            if skip is not None:
                mark = SKIP_LABELS.get(skip, "err")
                tip = SKIP_TIPS.get(skip, "")
                t = f' title="{tip}"' if tip else ""
                cells.append(f'<td class="skip"{t}>{mark}</td>')
                continue
            if cell.get("est"):
                tip = (f"ESTIMATED: {fmt_int(cell['calls'])} calls (one "
                       f"per blob{' pair' if 'ch04' in c['key'] else ''}) "
                       f"× median per-call kernel ms from a "
                       f"{cell['sample']}-blob sample. Never a row "
                       f"winner.")
                cells.append(f'<td class="est" title="{tip}">'
                             f'≈{fmt_ms(cell["ms"])}</td>')
                continue
            tip = (f"median {cell['ms']:.3f} ms "
                   f"(min {cell['ms_min']:.3f}, max {cell['ms_max']:.3f})")
            if cell.get("calls"):
                tip += (f" · MEASURED sum of {cell['calls']} per-blob "
                        f"calls")
            if "filled" in cell:
                tip += f" · {fmt_int(cell['filled'])} px filled"
            if "levels" in cell:
                tip += f" · {fmt_int(cell['levels'])} levels"
            cls = ' class="best"' if c["key"] == r.get("best") else ""
            cells.append(f'<td{cls} title="{tip}">{fmt_ms(cell["ms"])}'
                         "</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tablewrap"><table class="grand">'
            f"<thead>{head1}{head2}</thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def main():
    data, json_name = _load()
    n_rows = len(data["rows"])
    n_cols = len(data["columns"])
    ok = sum(1 for r in data["rows"] if r["crosscheck"] == "OK")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flood-fill-cuda — the evolution of a flood fill</title>
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body><div class="viz-root">

<div class="hero">
<h1>flood-fill-cuda</h1>
<p class="sub">The same job — find the red blobs in a white image and
fill them — solved over and over, from a pure-Python BFS to a
cooperative CUDA kernel that discovers, labels and fills every blob
with <b>zero seeds given</b>. Each chapter inherits the previous one's
problems and answers them with a measured experiment; every kernel is
validated pixel-for-pixel against a compiled CPU oracle.</p>
{chain_html()}
<p class="sub">Highlights along the way: 755,000 blobs discovered and
filled in ~25 ms; a level-synchronous BFS whose worst enemy is a 1-px
serpentine; a cooperative-launch deadlock lesson that became a fence
protocol; and a kernel that lost half the GPU to a single register —
129 where the two-blocks-per-SM line is exactly 128.</p>
</div>

<div class="card">
<h2>Glossary — how to read the numbers</h2>
{glossary_html()}
</div>

<div class="card">
<h2>The grand table — every approach × every shape</h2>
<p class="note">Rows: {n_rows} scenes (shape × scale, including two
external PNG inputs). Columns: {n_cols} approaches in chapter order.
Cells: median kernel ms on {data['device']} ({data['sm_count']} SMs,
tpb={data['tpb']}), hover any cell for details. <b>Bold = fastest
measured GPU approach for that row.</b> Every approach attempts every
job: one-blob kernels run one call per blob on multi-blob scenes —
measured sums on the two-blob rows, italic ≈estimates on the N-blob
rows (a full 755k-call loop would take hours; see the glossary).
Crosscheck: {ok}/{n_rows} rows with every completed measured approach
agreeing on the exact filled pixel count.</p>
{grand_table(data)}
</div>

<p class="note">Source data: <code>{json_name}</code> · rebuilt with
<code>python -m flood_fill_cuda.overview.bench</code> then
<code>python -m flood_fill_cuda.overview.build</code> · v{_version()}
· {date.today().isoformat()} · full interactive dashboard:
<code>src/flood_fill_cuda/results/dashboard/project_dashboard.html</code>
· <a href="https://github.com/lejrn/flood-fill-cuda">github.com/lejrn/flood-fill-cuda</a></p>

</div></body>
</html>
"""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(page)
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH):,} bytes)")


if __name__ == "__main__":
    main()
