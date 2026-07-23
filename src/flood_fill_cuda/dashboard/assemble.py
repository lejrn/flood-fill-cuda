"""
Assembles the whole-project benchmark dashboard from each chapter's own
renderer module: single blob (1 block -> 2 blocks -> N blocks -> 8
directions) and dual blob (2 blobs, N blocks).

Organized by DOMAIN, not by benchmark session:

  1. Single blob
     1.1 Single block        (single_block_shared: v1 ring, v2 spill)
     1.2 Dual blocks          (dual_block: split, global, dirsplit,
                               placement/pinning, balance, tpb sweep)
     1.3 Dual blocks vs N blocks (multi_block: 4-conn runtime/speedup/
                               sweep/bandwidth)
     1.4 4 vs 8 connectivity  (multi_block: the 8-direction twin kernels)
     1.5 More work per barrier (multi_block: radius-2 + warp-coop twins,
                               the neighbors_* benchmark)
  2. Dual blob                (dual_blob: sequential vs streams vs
                               multisource, lin vs xy entry format,
                               4 vs 8 connectivity on both mechanisms,
                               radius-2 on both mechanisms)
  3. Seed discovery           (seed_discovery: seed_merge colliding
                               waves vs ccl_fill union-find prepass,
                               N blobs, zero host-provided seeds,
                               discovery tax vs given-seeds ch04)

Every subsection reports its OWN speedup multiplier from its own
benchmark session; 1.3 also shows those multipliers chained together
into one total (single block -> N blocks) via `chain_strip()` -- see that
function's docstring for why the chain is computed from ONE file
(multi_block's own JSON, which re-measures every predecessor kernel
fresh in the same session) rather than cross-multiplying separate
sessions' numbers.

Each chapter owns its own renderer (chapters/<chapter_id>/benchmarks/
visualize.py, listed in registry.CHAPTERS); this module owns only what's
genuinely cross-chapter: the page frame, the project-tiles summary strip,
section ordering, and the combined tooltip + balance-panel JS.

Usage:
    uv run python -m flood_fill_cuda.dashboard [dual.json]

Output: results/dashboard/project_dashboard.html
(overwritten per run -- the timestamped JSON/CSV remain the durable record).
"""
import os

from ..shared import results_paths
from ..shared import viz
from . import registry

ch01_viz, ch02_viz, ch03_viz, ch04_viz, ch05_viz = registry.CHAPTERS

OUT_DIR = results_paths.results_dir("dashboard")
OUT_PATH = os.path.join(OUT_DIR, "project_dashboard.html")

PROJECT_TILES = [t for ch in registry.CHAPTERS for t in getattr(ch, "TILES", [])]
project_tiles_html = "".join(
    f'<div class="tile"><div class="tile-v">{v}</div>'
    f'<div class="tile-l">{l}</div><div class="tile-src">{s}</div></div>'
    for v, l, s in PROJECT_TILES)

CSS = viz.CSS
JS = viz.TOOLTIP_JS + ch02_viz.JS


def main():
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flood fill — 1 → 2 → N blocks → 8 directions → 2 blobs → N blobs
found by the GPU, benchmarked</title>
<style>{CSS}</style></head>
<body><div class="viz-root">
<h1>BFS flood fill — 1 block → 2 blocks → N blocks → 8 directions →
2 blobs → N blobs found by the GPU, benchmarked</h1>
<div class="sub">{ch02_viz.DEVICE} · {ch02_viz.SM_COUNT} SMs · organized by
domain: single blob (§1.1 one block → §1.2 two blocks → §1.3 N blocks →
§1.4 4-vs-8 connectivity → §1.5 per-barrier work experiments), dual
blob (§2), then seed discovery (§3: no host-provided seeds — the GPU
finds, labels and fills every blob itself). Every subsection reports
its own speedup from its own benchmark session; §1.3 chains them into one
multiplier from single block to N blocks. 4-connectivity throughout
except where 8-direction twins are charted explicitly · placement
observed via %smid</div>

<div class="tiles">{project_tiles_html}</div>

<h2 class="domain-h" style="margin-top:8px">1. Single blob</h2>

{ch01_viz.SECTION_1_1}

{ch02_viz.SECTION_1_2}

{ch03_viz.SECTION_1_3}

{ch03_viz.SECTION_1_4}
{ch03_viz.SECTION_1_5}
{ch04_viz.SECTION_2}
{ch05_viz.SECTION_3}
<div id="tooltip"></div>
</div>
<script>{JS}</script>
</body></html>
"""

    with open(OUT_PATH, "w") as f:
        f.write(html)
    _db_part = (f" + {os.path.basename(ch04_viz.DB_PATH)}" if ch04_viz.DB_PATH
                else " (no dual_blob JSON — §2 omitted)")
    _nb_part = (f" + {os.path.basename(ch03_viz.NB_PATH)}" if ch03_viz.NB_PATH
                else " (no neighbors JSON — §1.5 omitted)")
    _dbr2_part = (f" + {os.path.basename(ch04_viz.DBR2_PATH)}" if ch04_viz.DBR2_PATH
                  else " (no dual_blob_radius2 JSON — §2 r2 card omitted)")
    _sd_part = (f" + {os.path.basename(ch05_viz.SD_PATH)}" if ch05_viz.SD_PATH
                else " (no seed_discovery JSON — §3 omitted)")
    print(f"rendered {os.path.basename(ch03_viz.MB_PATH)} + "
          f"{os.path.basename(ch02_viz.DUAL_PATH)} + "
          f"{os.path.basename(ch01_viz.JSON_PATH)}{_nb_part}{_db_part}{_dbr2_part}"
          f"{_sd_part} -> {OUT_PATH} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
