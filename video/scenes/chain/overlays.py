"""The flat overlays of the chain cut v2, drawn above the 2.5D stage.

- goal meter: the frame budget (33 ms, the 81 Mpx frame at 30 fps) on a
  log track, with a teal budget tick, a grey CPU tick and a needle for
  the real frame's current time;
- scoreboard: "CPU vs GPU" in three rows (one small blob, one huge blob,
  many blobs). Collapsed it is a strip of three pills; a scene shows the
  expanded card for a few seconds when a row fills;
- the "you are here" inset (one line) and the act tag (SIZE / COUNT).

Every builder is a pure function of its spec, so `state.chain_state(k)`
and a scene's last frame draw the same pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Group, ImageMobject, Line, Polygon, Rectangle, RoundedRectangle,
                   VGroup, config)
from manim.constants import RESAMPLING_ALGORITHMS

from scenes.chain.geo import Frame
from scenes.style import ASSETS, GREY, GRID, INK, INK_SOFT, RED_PX, TEAL, label

Z_OVER = 500.0
PANEL = "#0f141a"
DIM = "#59636e"


def _z(mob, z: float = Z_OVER):
    for m in mob.get_family():
        m.set_z_index(z)
    return mob


def lbl(text: str, size: int = 36, color: str = INK, mono: bool = False):
    """label(), but crisp when small: Pango spaces small Text badly (the
    word space all but vanishes under about size 24), so small text is set
    at 4x and scaled down."""
    if size >= 24:
        return label(text, size, color, mono=mono)
    return label(text, size * 4, color, mono=mono).scale(0.25)


def fmt_ms(ms: float) -> str:
    """1,346 ms / 58.5 ms / 2.96 ms: never rounded up past what is spoken."""
    if ms >= 100:
        return f"{ms:,.0f} ms"
    if ms >= 10:
        return f"{int(ms * 10) / 10:.1f} ms"
    return f"{ms:.2f} ms"


def fmt_x(x: float) -> str:
    return f"{x:.1f}×" if x < 10 else f"{x:.0f}×"


# ---- goal meter ----------------------------------------------------------

LO_MS, HI_MS = 0.5, 4000.0


@dataclass(frozen=True)
class GoalSpec:
    cpu_tick: bool = True
    needle_ms: float | None = None      # None: parked (hidden)
    needle_text: str = ""
    needle_color: str = INK              # neutral: amber means only "waiting"
    mask_mark: bool = False             # the faint "1.46 ms packed mask" mark


def _track_x(G: Frame, ms: float, x0: float, x1: float) -> float:
    a = (math.log10(ms) - math.log10(LO_MS)) / (math.log10(HI_MS) - math.log10(LO_MS))
    return x0 + min(max(a, 0.0), 1.0) * (x1 - x0)


def goal_meter(G: Frame, spec: GoalSpec, nums: dict) -> VGroup:
    w = G.goal_w
    h = 1.05
    box = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=GRID, stroke_width=2,
                           fill_color=PANEL, fill_opacity=1).move_to(G.goal_c)
    left = lbl("budget 33 ms", 26, INK, mono=True)
    left.move_to(box.get_left() + RIGHT * (0.2 + left.width / 2))
    x0 = left.get_right()[0] + 0.35
    x1 = box.get_right()[0] - 0.3
    y = G.goal_c[1] - 0.12
    track = Line([x0, y, 0], [x1, y, 0], stroke_color=GRID, stroke_width=3)
    g = VGroup(box, left, track)
    for ms in (1, 10, 100, 1000):           # decade marks; the named ticks carry the meaning
        x = _track_x(G, ms, x0, x1)
        g.add(Line([x, y - 0.05, 0], [x, y + 0.05, 0], stroke_color=GRID, stroke_width=2))
    xb = _track_x(G, 1000 / 30, x0, x1)
    g.add(Line([xb, y - 0.14, 0], [xb, y + 0.14, 0], stroke_color=TEAL, stroke_width=4))
    if spec.cpu_tick:
        xc = _track_x(G, nums["njit_ms"], x0, x1)
        g.add(Line([xc, y - 0.14, 0], [xc, y + 0.14, 0], stroke_color=GREY, stroke_width=4))
        g.add(lbl("CPU", 13, GREY, mono=True).move_to([xc, y + 0.25, 0]))
    if spec.mask_mark:
        xm = _track_x(G, nums["ch06_mask_ms"], x0, x1)
        g.add(Line([xm, y - 0.08, 0], [xm, y + 0.08, 0], stroke_color=TEAL, stroke_width=2, stroke_opacity=0.5))
    if spec.needle_ms is not None:
        xn = _track_x(G, spec.needle_ms, x0, x1)
        tri = Polygon([xn, y - 0.02, 0], [xn - 0.09, y - 0.2, 0], [xn + 0.09, y - 0.2, 0],
                      fill_color=spec.needle_color, fill_opacity=1, stroke_width=0)
        val = lbl(spec.needle_text, 14, spec.needle_color, mono=True).move_to([xn, y - 0.29, 0])
        if val.get_right()[0] > box.get_right()[0] - 0.05:
            val.shift(LEFT * (val.get_right()[0] - box.get_right()[0] + 0.05))
        g.add(tri, val)
    return _z(g)


# ---- scoreboard ----------------------------------------------------------

@dataclass(frozen=True)
class Row:
    what: str                  # "one small blob"
    cpu_ms: float | None = None
    gpu_ms: float | None = None
    winner: str | None = None  # "CPU" | "GPU" | None (not yet)
    factor: str = ""           # "2.4×"
    picture: str = ""          # "16 K px test blob"
    tag: str = ""              # "one session"


ROW_NAMES = ("small blob", "huge blob", "many blobs")


def _pill(text_top: str, text_bottom: str, color: str, w: float, h: float = 0.62, dashed: bool = False) -> VGroup:
    box = RoundedRectangle(corner_radius=0.1, width=w, height=h, fill_color=PANEL, fill_opacity=1,
                           stroke_color=color, stroke_width=2.5)
    top = lbl(text_top, 15, INK_SOFT)
    bot = lbl(text_bottom, 20, color, mono=True)
    t = VGroup(top, bot).arrange(DOWN, buff=0.04).move_to(box)
    for m in (top, bot):
        if m.width > w - 0.12:
            m.scale_to_fit_width(w - 0.12)
    return VGroup(box, t)


def score_strip(G: Frame, rows: tuple) -> VGroup:
    """rows: three entries, each a Row, "?" (promised, unknown) or None (absent)."""
    n = 3
    gap = 0.12
    w = (G.score_w - gap * (n - 1)) / n
    g = VGroup()
    for i, r in enumerate(rows):
        x = G.score_c[0] - G.score_w / 2 + w / 2 + i * (w + gap)
        if r is None:
            continue
        if r == "?":
            p = _pill(ROW_NAMES[i], "?", DIM, w)
        else:
            col = TEAL if r.winner == "GPU" else GREY
            p = _pill(ROW_NAMES[i], f"{r.winner} {r.factor}", col, w)
        p.move_to([x, G.score_c[1], 0])
        g.add(p)
    return _z(g, Z_OVER + 1)


def score_card(G: Frame, rows: tuple, highlight: int | None = None, footnote: str | None = None) -> VGroup:
    """The expanded scoreboard: picture | CPU | GPU | winner per row."""
    filled = [r for r in rows if isinstance(r, Row)]
    lines = VGroup()
    blank = Rectangle(width=0.01, height=0.01, stroke_width=0, fill_opacity=0)   # an empty cell with points
    head = VGroup(blank, lbl("CPU", 16, INK_SOFT), lbl("GPU", 16, INK_SOFT), lbl("winner", 16, INK_SOFT))
    lines.add(head)
    for i, r in enumerate(rows):
        if r is None:
            continue
        if r == "?":
            cells = VGroup(lbl(ROW_NAMES[i], 20, DIM), lbl("?", 20, DIM, mono=True),
                           lbl("?", 20, DIM, mono=True), lbl("?", 20, DIM, mono=True))
        else:
            col = TEAL if r.winner == "GPU" else GREY
            name = VGroup(lbl(r.what, 20, INK if i == highlight else INK_SOFT),
                          lbl(r.picture, 13, INK_SOFT)).arrange(DOWN, buff=0.03, aligned_edge=LEFT)
            cells = VGroup(name, lbl(fmt_ms(r.cpu_ms), 20, INK, mono=True),
                           lbl(fmt_ms(r.gpu_ms), 20, INK, mono=True),
                           lbl(f"{r.winner} {r.factor}", 22, col, mono=True))
        lines.add(cells)
    cols = [max(row[j].width for row in lines) for j in range(4)]
    xs = np.cumsum([0] + [c + 0.45 for c in cols[:-1]])
    for row in lines:
        for j, m in enumerate(row):
            m.move_to([xs[j], 0, 0], aligned_edge=LEFT)
    lines.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    for row in lines:
        for j, m in enumerate(row):
            m.set_x(xs[j] + m.width / 2)
    body = VGroup(lbl("CPU vs GPU", 22, INK), lines).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    if footnote:
        fn = lbl(footnote, 12, INK_SOFT, mono=True)
        body.add(fn)
        body.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    room = config.frame_width - 0.9
    if body.width > room:
        body.scale_to_fit_width(room)
    box = RoundedRectangle(corner_radius=0.16, width=body.width + 0.6, height=body.height + 0.5,
                           fill_color=PANEL, fill_opacity=0.97, stroke_color=GRID, stroke_width=2)
    box.move_to(G.card_c)
    body.move_to(box)
    return _z(VGroup(box, body), Z_OVER + 20)


# ---- small overlays ------------------------------------------------------

def act_tag(G: Frame, act: str, won: bool = False) -> VGroup:
    color = TEAL if won else INK
    text = lbl(act + (" ✓" if won else ""), 18, color, mono=True)
    box = RoundedRectangle(corner_radius=0.1, width=text.width + 0.35, height=text.height + 0.22,
                           fill_color=PANEL, fill_opacity=1, stroke_color=color, stroke_width=2)
    text.move_to(box)
    return _z(VGroup(box, text).move_to(G.tag_c))


DRONE_FRAME = 216
DRONE_BOX = (64, 104, 77, 116)          # x0, y0, x1, y1 in the 576 x 1024 frame


def inset(G: Frame, line: str) -> Group:
    """A thumbnail of the camera frame with a box on the chosen drone."""
    from PIL import Image

    im = np.asarray(Image.open(ASSETS / "drones_sky" / f"frame_{DRONE_FRAME:03d}.png").convert("RGBA"))
    img = ImageMobject(im)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    hgt = 1.25 if G.vertical else 1.6
    img.set_height(hgt)
    img.move_to(G.inset_c)
    H, W = im.shape[:2]
    x0, y0, x1, y1 = DRONE_BOX
    s = hgt / H
    ul = img.get_corner(UP + LEFT)
    cx = ul[0] + ((x0 + x1) / 2) * s
    cy = ul[1] - ((y0 + y1) / 2) * s
    ring = Rectangle(width=max((x1 - x0) * s, 0.08) + 0.1, height=max((y1 - y0) * s, 0.08) + 0.1,
                     stroke_color=RED_PX, stroke_width=2.5).move_to([cx, cy, 0])
    frame_box = Rectangle(width=img.width + 0.04, height=img.height + 0.04, stroke_color=GRID,
                          stroke_width=1.5).move_to(img)
    txt = lbl(line, 14, INK_SOFT, mono=True).next_to(frame_box, DOWN, buff=0.06)
    g = Group(img, frame_box, ring, txt)
    for m in g.get_family():
        m.set_z_index(Z_OVER - 5)
    return g
