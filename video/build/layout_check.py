"""Every pane state, measured without rendering: nothing may leave the frame
or touch another pane's content.

    uv run build/layout_check.py [--layout vertical]

Builds `build_state(k)` for every stage boundary and `build_live(k)` for the
middle of every stage, then checks each top-level mobject's bounding box
against the frame, and every piece of it (a Text, an image, a shape)
against the pieces of the other two panes. A caption may overhang its box
into empty space, as stage 6's does in landscape, but not into content.
Exit 1 on a finding. Cheap on memory (no frames are drawn), so it is the
first check after a geometry change; snapshots come after it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
EDGE = 0.05        # minimum distance from the frame edge, units
TOUCH = 0.02       # overlap between two panes' mobjects that still counts as clear


def bounds(m):
    """(x0, x1, y0, y1) of a mobject's drawn family, or None when it draws nothing."""
    if not m.family_members_with_points():
        return None
    return m.get_left()[0], m.get_right()[0], m.get_bottom()[1], m.get_top()[1]


def leaf_mobjects(m) -> list:
    """The pieces worth comparing: a Text as a whole (not its glyphs), an image,
    a shape. Groups are opened up, so an empty corner of a group never counts."""
    from manim import ImageMobject, Text

    if isinstance(m, (Text, ImageMobject)) or not m.submobjects:
        return [m]
    return [leaf for sub in m.submobjects for leaf in leaf_mobjects(sub)]


def overlap(a, b) -> tuple:
    """Smallest of the x and y overlaps of two boxes (<= 0 when apart)."""
    return min(a[1], b[1]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[2], b[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["landscape", "vertical"], default="landscape")
    args = ap.parse_args()
    # style.py sizes the frame at import time from VIDEO_LAYOUT
    os.environ["VIDEO_LAYOUT"] = args.layout
    sys.path.insert(0, str(VIDEO))

    from manim import config
    from scenes.panes import data
    from scenes.panes.config import STAGES
    from scenes.panes.geometry import pane_geometry
    from scenes.stage import build_live, build_state
    from scenes.style import layout

    L = layout()
    geo = pane_geometry(L)
    bench = data.load_bench()
    w, h = config.frame_width, config.frame_height
    boxes = {name: getattr(geo, name) for name in ("left", "middle", "right")}
    print(f"{args.layout}: frame {w:.2f} x {h:.2f}")
    for name, b in boxes.items():
        print(f"  {name:6s} x {b.x0:6.2f} .. {b.x1:6.2f}   y {b.y0:6.2f} .. {b.y1:6.2f}")

    states = [(f"state {k}", build_state(k, geo, bench)) for k in range(1, len(STAGES) + 1)]
    states += [(f"live {k}", build_live(k, geo, bench)) for k in range(len(STAGES))]
    findings = 0
    for tag, st in states:
        leaves = {}
        for name in boxes:
            pane = getattr(st, name)
            items = []
            for i, m in enumerate(pane.all() if pane is not None else []):
                bb = bounds(m)
                if bb is None:
                    continue
                what = f"{tag:9s} {name:6s} #{i} {type(m).__name__}"
                out = max(-w / 2 + EDGE - bb[0], bb[1] - w / 2 + EDGE, -h / 2 + EDGE - bb[2], bb[3] - h / 2 + EDGE)
                if out > 0:
                    findings += 1
                    print(f"OUT    {what}: {out:.2f} units past the frame edge")
                items += [(what, b) for sub in leaf_mobjects(m) if (b := bounds(sub)) is not None]
            leaves[name] = items
        names = list(leaves)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                for wa, ba in leaves[names[a]]:
                    for wb, bb in leaves[names[b]]:
                        ox, oy = overlap(ba, bb)
                        if ox > TOUCH and oy > TOUCH:
                            findings += 1
                            print(f"TOUCH  {wa} and {wb.split(maxsplit=2)[-1]}: {min(ox, oy):.2f} units")
    print(f"{len(states)} states, {findings} finding{'s' if findings != 1 else ''}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
