"""Inline the committed ch06 figures into index.html, re-themed for the page.

The SVGs in results/ch06_gpu_nblob_runs/figures/ are drawn for GitHub's
dark theme: light grey text on a transparent background. On a white page
that text almost vanishes. This script keeps every coordinate, bar width
and label exactly as committed and only swaps the seven hard-coded colours
for CSS classes, so the page's light and dark tokens decide them.

It also replaces em and en dashes in the figure text with plain hyphens.

Each figure lands in index.html between a pair of markers:

    <!-- figure:chain:start --> ... <!-- figure:chain:end -->

Run it again whenever a figure is regenerated:

    python project-page/tools/build_figures.py
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1]
REPO = PAGE.parent
FIGURES = REPO / "src/flood_fill_cuda/results/ch06_gpu_nblob_runs/figures"
INDEX = PAGE / "index.html"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# Hard-coded colour -> role. The page CSS gives each role a light and a
# dark value (.fig-<role> for fills, .fig-<role>-s for strokes).
ROLES = {
    "#adbac7": "ink",  # titles and values
    "#7d8590": "muted",  # axis ticks, row labels, notes
    "#8b949e": "grid",  # grid lines, the neutral "all pixels" bar
    "#2aa198": "teal",  # ch06
    "#d99a2b": "amber",  # ch06 rgb-in series
    "#e2645a": "red",  # red pixels, ch01-ch05 bars
    "#4c8fd6": "blue",  # blobs bar, labeling-only series
}

NAMES = ["chain", "runs_vs_pixels", "speedup", "scaling"]

# Text lines the page replaces with its own <figcaption>. chain.svg's
# footnote runs past the 720 px viewBox, and the page caption says the
# same thing with the session and estimate caveats added.
DROP_TEXT = {
    "chain": ["pure Python \u2192 ch06"],
}


def themed_svg(name: str) -> str:
    tree = ET.parse(FIGURES / f"{name}.svg")
    root = tree.getroot()
    unknown: set[str] = set()

    for el in root.iter():
        classes = el.get("class", "").split()
        for attr, suffix in (("fill", ""), ("stroke", "-s")):
            colour = el.get(attr)
            if colour is None or not colour.startswith("#"):
                continue
            role = ROLES.get(colour.lower())
            if role is None:
                unknown.add(colour)
                continue
            del el.attrib[attr]
            classes.append(f"fig-{role}{suffix}")
        if classes:
            el.set("class", " ".join(classes))
        if el.text:
            el.text = el.text.replace("\u2014", "-").replace("\u2013", "-")

    if unknown:
        sys.exit(f"{name}.svg: unmapped colours {sorted(unknown)}; add them to ROLES")

    drops = DROP_TEXT.get(name, [])
    if drops:
        for parent in list(root.iter()):
            for child in list(parent):
                text = child.text or ""
                if child.tag == f"{{{SVG_NS}}}text" and any(text.startswith(d) for d in drops):
                    parent.remove(child)
        # Trim the empty band the dropped line leaves at the bottom.
        lowest = max(
            float(t.get("y", 0)) for t in root.iter(f"{{{SVG_NS}}}text")
        )
        x0, y0, w, h = root.get("viewBox").split()
        root.set("viewBox", f"{x0} {y0} {w} {min(float(h), lowest + 8):g}")

    # Scale with the column: keep the viewBox, drop the fixed size.
    root.attrib.pop("width", None)
    root.attrib.pop("height", None)
    root.set("class", "figure-svg")
    root.set("focusable", "false")

    title = root.find(f"{{{SVG_NS}}}title")
    if title is not None:
        title.set("id", f"fig-{name}-title")
        root.set("aria-labelledby", f"fig-{name}-title")
        root.attrib.pop("aria-label", None)

    text = ET.tostring(root, encoding="unicode")
    # ElementTree writes the default namespace on the root; inline SVG in
    # HTML does not need it, but it is harmless, so it stays.
    return text


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for name in NAMES:
        pattern = re.compile(
            rf"(<!-- figure:{name}:start -->)(.*?)(<!-- figure:{name}:end -->)",
            re.S,
        )
        if not pattern.search(html):
            sys.exit(f"index.html has no markers for figure '{name}'")
        svg = themed_svg(name)
        html = pattern.sub(lambda m: f"{m.group(1)}\n{svg}\n{m.group(3)}", html)
        print(f"inlined {name}.svg ({len(svg):,} chars)")
    INDEX.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
