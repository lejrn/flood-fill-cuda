"""Inline the committed ch06 figures into index.html, re-themed for the page.

The SVGs in results/ch06_gpu_nblob_runs/figures/ are drawn for GitHub's
dark theme: light grey text on a transparent background. On a white page
that text almost vanishes. This script keeps every bar, dot, line and
number exactly as committed. It changes only what the page needs:

- the seven hard-coded colors become CSS classes, so the page's light and
  dark tokens decide them;
- em and en dashes in the figure text become plain hyphens;
- a few lines of text are dropped, relabeled or moved (see EDITS below),
  each because the page caption already says it or because it collided
  with another label;
- the fixed width and height go, ids and an accessible title and
  description are added, and the viewBox is trimmed when a dropped line
  leaves an empty band at the bottom.

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
TEXT = f"{{{SVG_NS}}}text"
ET.register_namespace("", SVG_NS)

# Hard-coded color -> role. The page CSS gives each role a light and a
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

# Text edits per figure. Matching is on a <text> element's content
# (drop: its start; relabel and move: the whole text). No edit touches a
# number.
EDITS = {
    "chain": {
        # The footnote runs past the 720-unit viewBox; the page caption
        # says the same with the estimate and session caveats added.
        "drop": ["pure Python → ch06"],
        # The two ch06 rows share one label; name the contract instead.
        "relabel": [("ch06 · N runs", ["ch06 · rgb in", "ch06 · packed mask"])],
    },
    "runs_vs_pixels": {
        # Repeats the paragraph above it, rounded down to 13.4 million.
        "drop": ["13.4 million red pixels"],
    },
    "scaling": {
        # Repeats the page text, and sits on the x-axis title's baseline.
        "drop": ["packed mask stays under"],
        # The y-axis unit sat 8 units under the title; give it room.
        "move": [("ms", {"y": "48"})],
    },
}


def _dash_free(text: str) -> str:
    return text.replace("\u2014", "-").replace("\u2013", "-")


def themed_svg(name: str) -> str:
    tree = ET.parse(FIGURES / f"{name}.svg")
    root = tree.getroot()
    unknown: set[str] = set()

    for el in root.iter():
        classes = el.get("class", "").split()
        for attr, suffix in (("fill", ""), ("stroke", "-s")):
            color = el.get(attr)
            if color is None or not color.startswith("#"):
                continue
            role = ROLES.get(color.lower())
            if role is None:
                unknown.add(color)
                continue
            del el.attrib[attr]
            classes.append(f"fig-{role}{suffix}")
        if classes:
            el.set("class", " ".join(classes))
        if el.text:
            el.text = _dash_free(el.text)

    if unknown:
        sys.exit(f"{name}.svg: unmapped colors {sorted(unknown)}; add them to ROLES")

    edits = EDITS.get(name, {})
    drops = edits.get("drop", [])
    if drops:
        for parent in list(root.iter()):
            for child in list(parent):
                text = child.text or ""
                if child.tag == TEXT and any(text.startswith(d) for d in drops):
                    parent.remove(child)
        # Trim the empty band a dropped bottom line leaves behind.
        lowest = max(float(t.get("y", 0)) for t in root.iter(TEXT))
        x0, y0, w, h = root.get("viewBox").split()
        root.set("viewBox", f"{x0} {y0} {w} {min(float(h), lowest + 8):g}")

    for old, new_labels in edits.get("relabel", []):
        hits = [t for t in root.iter(TEXT) if (t.text or "") == old]
        if len(hits) != len(new_labels):
            sys.exit(f"{name}.svg: expected {len(new_labels)} '{old}' labels, found {len(hits)}")
        for t, label in zip(hits, new_labels):
            t.text = label

    for text, attrs in edits.get("move", []):
        hits = [t for t in root.iter(TEXT) if (t.text or "") == text]
        if len(hits) != 1:
            sys.exit(f"{name}.svg: expected one '{text}' label, found {len(hits)}")
        for key, value in attrs.items():
            hits[0].set(key, value)

    # Scale with the column: keep the viewBox, drop the fixed size.
    root.attrib.pop("width", None)
    root.attrib.pop("height", None)
    root.set("class", "figure-svg")
    root.set("focusable", "false")

    # role="img" hides the text nodes, so spell every label and value out
    # in a <desc>, in drawing order, for screen readers.
    title = root.find(f"{{{SVG_NS}}}title")
    if title is not None:
        title.set("id", f"fig-{name}-title")
        root.set("aria-labelledby", f"fig-{name}-title")
        root.attrib.pop("aria-label", None)
        words = [(t.text or "").strip() for t in root.iter(TEXT)]
        desc = ET.Element(f"{{{SVG_NS}}}desc", {"id": f"fig-{name}-desc"})
        desc.text = "; ".join(w for w in words if w)
        root.insert(list(root).index(title) + 1, desc)
        root.set("aria-describedby", f"fig-{name}-desc")

    return ET.tostring(root, encoding="unicode")


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
