"""Every pane state, measured without rendering: nothing may leave the frame
or touch another pane's content.

    uv run build/layout_check.py [--layout vertical]
    uv run build/layout_check.py --cut chain [--layout vertical] [--look iso]

Builds `build_state(k)` for every stage boundary and `build_live(k)` for the
middle of every stage, then checks each top-level mobject's bounding box
against the frame, and every piece of it (a Text, an image, a shape)
against the pieces of the other two panes. A caption may overhang its box
into empty space, as stage 6's does in landscape, but not into content.
Exit 1 on a finding. Cheap on memory (no frames are drawn), so it is the
first check after a geometry change; snapshots come after it.

--cut chain checks the chain cut instead: every `chain_state(k)` (k = 2
to the last), each part grouped as floor, machine, caption, goal, strip,
inset or tag. Floors and the machine's faces are their real outlines
(convex polygons: the floor's tilted plane and slab, each box face), the
rest their pieces' boxes. Nothing may leave the frame, and no two groups
may touch, except that the inset and the act tag sit on the floor's
empty corners (so their boxes are checked against the floor's outline).
Then r10's panels open over their state (the SM close-up at G.sm_side and
the warp card at G.warp, beside it) and must touch nothing but the floor;
the r14 card with its tags and footnote reports the scale it fits at.
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


def _poly(points) -> "np.ndarray":
    import numpy as np

    return np.asarray(points, float)[:, :2]


def _box_poly(b) -> "np.ndarray":
    import numpy as np

    x0, x1, y0, y1 = b
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def sat_overlap(a, b) -> float:
    """How deep two convex polygons overlap (the smallest overlap along the
    edge normals of both; <= 0 when a separating axis exists)."""
    import numpy as np

    best = np.inf
    for P in (a, b):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            n = np.array([-e[1], e[0]])
            L = np.hypot(*n)
            if L < 1e-12:
                continue
            n /= L
            pa, pb = a @ n, b @ n
            best = min(best, min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
            if best <= 0:
                return best
    return best


def chain_pieces(group: str, m) -> list:
    """(what, convex polygon) for each drawn piece of a chain part: a face or
    a polygon by its own points, anything else by its box."""
    from manim import ImageMobject, Polygon, Text

    out = []
    for leaf in leaf_mobjects(m):
        if isinstance(leaf, Polygon) and leaf.has_points():
            pts = leaf.get_vertices()
            if len(pts) >= 3:
                out.append((f"{group} {type(leaf).__name__}", _poly(pts)))
                continue
        bb = bounds(leaf)
        if bb is not None and (isinstance(leaf, (Text, ImageMobject)) or bb[1] - bb[0] > 1e-6 or bb[3] - bb[2] > 1e-6):
            out.append((f"{group} {type(leaf).__name__}", _box_poly(bb)))
    return out


def floor_pieces(f) -> list:
    """A floor's real outline: its tilted plane and its slab's faces (its
    image's box is mostly transparent corners)."""
    from scenes.chain import floor as fl
    from scenes.chain.iso import plane_corners

    ul, ur, dl, dr = plane_corners(f.pl, -fl.HALF, -fl.HALF, fl.HALF, fl.HALF, 0.0)
    out = [("floor plane", _poly([ul, ur, dr, dl]))]
    out += [("floor slab", _poly(face.get_vertices())) for face in f.slab]
    return out


# groups that may share space: (a, b) -> False. The inset and the act tag sit
# on the floor's empty corners; a panel opens over the floor
CHAIN_ALLOWED = {frozenset(("panel", "floor"))}


def chain_main(args) -> int:
    from manim import config

    from scenes.chain import machine as mc
    from scenes.chain import overlays as ov
    from scenes.chain import state as st

    w, h = config.frame_width, config.frame_height
    L = st.layout()
    G = L.G
    print(f"chain {args.layout} ({os.environ['LOOK']}): frame {w:.2f} x {h:.2f}")
    findings = 0

    def pieces_of(s) -> dict:
        groups = {"floor": [p for f in s.floors for p in floor_pieces(f)],
                  "machine": [p for part in (s.machine.mobs() if s.machine else []) for p in chain_pieces("machine", part)],
                  "caption": chain_pieces("caption", s.cap) if s.cap is not None else []}
        for name, m in s.over.items():
            if m is not None:
                groups[name] = chain_pieces(name, m)
        return groups

    def check(tag: str, tops: list, groups: dict) -> int:
        bad = 0
        for what, m in tops:
            bb = bounds(m)
            if bb is None:
                continue
            out = max(-w / 2 + EDGE - bb[0], bb[1] - w / 2 + EDGE, -h / 2 + EDGE - bb[2], bb[3] - h / 2 + EDGE)
            if out > 0:
                bad += 1
                print(f"OUT    {tag:9s} {what}: {out:.2f} units past the frame edge")
        names = [n for n in groups if groups[n]]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if frozenset((a, b)) in CHAIN_ALLOWED:
                    continue
                worst = None
                for wa, pa in groups[a]:
                    for wb, pb in groups[b]:
                        d = sat_overlap(pa, pb)
                        if d > TOUCH and (worst is None or d > worst[0]):
                            worst = (d, wa, wb)
                if worst is not None:
                    bad += 1
                    print(f"TOUCH  {tag:9s} {worst[1]} and {worst[2]}: {worst[0]:.2f} units")
        return bad

    states = list(range(st.FIRST, st.LAST + 1))
    for k in states:
        s = st.chain_state(k, L)
        tops = [(f"#{i} {type(m).__name__}", m) for i, m in enumerate(s.mobs())]
        findings += check(f"state {k}", tops, pieces_of(s))

    # r10's panels, open over the state r10 starts on and the one it ends on
    (wc, wh), (sc, sh) = G.warp, G.sm_side
    card = mc.warp_closeup(wc, wh, st.a3()["cu_mask"])
    card.add(card.glyph("diag"))
    sm = mc.sm_closeup(L.machine_pl, (*sc, 0.0), sh, i=0, shelf="off")
    for k in (14, 15):
        s = st.chain_state(k, L)
        groups = pieces_of(s)
        groups["panel"] = chain_pieces("panel warp", card) + chain_pieces("panel sm", sm)
        findings += check(f"r10 k={k}", [("warp card", card), ("SM close-up", sm)], groups)
        # the two panels side by side must not cover each other
        gap = min(sat_overlap(a, b) for _, a in chain_pieces("w", card[0]) for _, b in chain_pieces("s", sm[0]))
        if gap > -0.05:
            findings += 1
            print(f"TOUCH  r10 k={k}  the warp card and the SM close-up: {gap:.2f} units")
    cb, sb = bounds(card), bounds(sm)
    print(f"  warp card x {cb[0]:.2f} .. {cb[1]:.2f}, y {cb[2]:.2f} .. {cb[3]:.2f};"
          f" SM close-up x {sb[0]:.2f} .. {sb[1]:.2f}, y {sb[2]:.2f} .. {sb[3]:.2f}")

    # the r14 card: the rows at both flip steps, tags and footnote; the scale it
    # fits at between the caption and the inset (as r11_reveal placed v4's)
    for flip in (1, 2):
        c = ov.score_card(G, st.score_rows("r14_reveal", flip), highlight=(0,), footnote=st.card_footnote())
        y0 = G.caption_c[1] + (0.6 if G.vertical else 0.5)
        y1 = (3.6 if G.vertical else h / 2 - 1.15)
        x_room = w - 0.3 if G.vertical else 6.8                  # 16:9: left of the boards
        scale = min(1.0, x_room / c.width, (y1 - y0) / c.height)
        print(f"  r14 card, flip {flip}: {c.width:.2f} x {c.height:.2f}, fits at scale {scale:.2f}")
        if scale < 0.7:
            findings += 1
            print(f"SMALL  r14 card, flip {flip}: scale {scale:.2f} (text under 70% of its size)")
    print(f"{len(states)} chain states + r10's panels + the r14 card, {findings} finding{'s' if findings != 1 else ''}")
    return 1 if findings else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["landscape", "vertical"], default="landscape")
    ap.add_argument("--cut", choices=["panes", "chain"], default="panes")
    ap.add_argument("--look", choices=["flat", "iso"], default="iso", help="chain cut only")
    args = ap.parse_args()
    # style.py sizes the frame at import time from VIDEO_LAYOUT
    os.environ["VIDEO_LAYOUT"] = args.layout
    os.environ["LOOK"] = args.look
    sys.path.insert(0, str(VIDEO))
    if args.cut == "chain":
        return chain_main(args)

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
