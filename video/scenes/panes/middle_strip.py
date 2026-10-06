"""Middle pane: the current blob, big, and the strip of finished stages.

Every image goes through `load_image` (one resampling algorithm) and is
sized by the same helpers whether it is a replay frame, the still that
replaces it at a stage boundary, or the thumbnail it sweeps into. That is
what keeps the boundary frames pixel-identical.
"""
from __future__ import annotations

import numpy as np
from manim import (
    Group, ImageMobject, ReplacementTransform, RoundedRectangle, Square, VGroup,
    LEFT, RIGHT,
)
from manim.constants import RESAMPLING_ALGORITHMS

from scenes.style import ASSETS, GREY, GRID, INK, INK_SOFT, PURPLE, RED_PX, TEAL, is_vertical, label
from scenes.panes import data
from scenes.panes.geometry import Box

THUMB_W = 0.44
SLOT_PITCH = 0.50
BIG = 4.1
TAG_FONT, CAP_FONT, CAP2_FONT = 9, 20, 14
RESAMPLE = RESAMPLING_ALGORITHMS["bicubic"]

# stage 7's pixel row (from the old s05_runs): 36 cells, four red spans
N_CELLS = 36
SPANS = [(2, 6), (10, 18), (22, 24), (27, 35)]
ROW_CELL = 0.105


def slot_center(box: Box, i: int) -> np.ndarray:
    return box.at(box.x0 + 0.25 + i * SLOT_PITCH, box.y1 - 0.30)


def tag_point(box: Box, i: int) -> np.ndarray:
    return box.at(box.x0 + 0.25 + i * SLOT_PITCH, box.y1 - 0.64)


def big_center(box: Box, dy: float = 0.0) -> np.ndarray:
    return box.at(box.cx, box.cy - 0.12 + dy)


def load_image(source) -> ImageMobject:
    img = ImageMobject(str(source) if not isinstance(source, np.ndarray) else source)
    img.set_resampling_algorithm(RESAMPLE)
    return img


def fit(img: ImageMobject, w: float, h: float) -> ImageMobject:
    img.scale(min(w / img.width, h / img.height))
    return img


def big_image(box: Box, source, fit_wh=(BIG, BIG), dy: float = 0.0) -> ImageMobject:
    img = load_image(source)
    fit(img, *fit_wh)
    return img.move_to(big_center(box, dy))


def thumb(box: Box, i: int, source) -> ImageMobject:
    img = load_image(source)
    img.scale_to_fit_width(THUMB_W)
    return img.move_to(slot_center(box, i))


def tag(box: Box, i: int, text: str) -> VGroup:
    """The stage tag under a thumbnail, split at the first space into two lines."""
    parts = text.split(" ", 1)
    g = VGroup()
    for n, part in enumerate(parts):
        g.add(label(part, size=TAG_FONT, color=INK_SOFT)
              .move_to(tag_point(box, i) + np.array([0, -0.15 * n, 0])))
    return g


def captions(box: Box, line1: str | None, line2: str | None) -> VGroup:
    g = VGroup()
    if line1:
        g.add(label(line1, size=CAP_FONT, color=INK_SOFT).move_to(box.at(box.cx, box.y0 + 0.56)))
    if line2:
        g.add(label(line2, size=CAP2_FONT, color=GREY).move_to(box.at(box.cx, box.y0 + 0.24)))
    if is_vertical():
        # the pane sits at the frame edge here: no neighbour's margin to overhang into
        for m in g:
            if m.width > box.w:
                m.scale_to_fit_width(box.w)
    return g


def runs_row(box: Box, collapsed: bool):
    """Stage 7's pixel row. Returns (row, spans, bars): `row` holds grey
    segments and either the red spans (collapsed=False) or the teal bars
    (collapsed=True) at exactly the same places."""
    red = [any(a <= i <= b for a, b in SPANS) for i in range(N_CELLS)]
    cells = []
    for i in range(N_CELLS):
        c = Square(ROW_CELL, stroke_width=1.0, stroke_color=GRID)
        if red[i]:
            c.set_fill(RED_PX, 1.0)
        else:
            c.set_fill(INK_SOFT, 0.28)
        cells.append(c)
    VGroup(*cells).arrange(RIGHT, buff=0).move_to(box.at(box.cx, box.cy + 0.62))
    parts, spans, bars, i = [], [], [], 0
    while i < N_CELLS:
        j = i
        while j + 1 < N_CELLS and red[j + 1] == red[i]:
            j += 1
        seg = VGroup(*cells[i:j + 1])
        if red[i]:
            h = ROW_CELL * 0.7
            bar = RoundedRectangle(corner_radius=h / 2, width=seg.width, height=h,
                                   stroke_width=0, fill_color=TEAL, fill_opacity=1.0).move_to(seg)
            spans.append(seg)
            bars.append(bar)
            parts.append(bar if collapsed else seg)
        else:
            parts.append(seg)
        i = j + 1
    return VGroup(*parts), spans, bars


def triton_card(box: Box) -> VGroup:
    """The Triton stage's big area: the language pair, the overall ratio, how
    many like-for-like rows Triton wins, and the chapter where it still
    loses. Numbers from data.load_twins(); lines 1-2 and 3-4 enter apart."""
    tw = data.load_twins()
    lines = [
        label("Numba → Triton", size=26, color=INK, mono=True),
        label(f"{data.fmt_twin(tw.overall)}×", size=64, color=PURPLE, mono=True),
        label(f"faster in {tw.rows_faster:,} of {tw.rows_like:,} cases", size=16, color=INK_SOFT),
        label(f"chapter 5 still slower: {data.fmt_twin(tw.ch05)}×", size=14, color=GREY),
    ]
    for m, dy in zip(lines, (1.15, 0.2, -0.75, -1.08)):
        m.move_to(box.at(box.cx, box.cy + dy))
        if m.width > box.w - 0.2:
            m.scale_to_fit_width(box.w - 0.2)
    return VGroup(*lines)


class MiddlePane:
    """Strip of thumbs + tags, the big picture (image and optional extra), captions."""

    def __init__(self, box: Box, thumbs: list, big_source, caption1: str | None,
                 caption2: str | None, fit_wh=(BIG, BIG), dy: float = 0.0,
                 extra: str | None = None):
        self.box = box
        self.strip = Group(*[thumb(box, i, ASSETS / src) for i, (src, _) in enumerate(thumbs)])
        self.tags = VGroup(*[tag(box, i, t) for i, (_, t) in enumerate(thumbs)])
        self.caption = captions(box, caption1, caption2)
        self.big_image = big_image(box, ASSETS / big_source, fit_wh, dy) if big_source else None
        self.big_extra = None
        if extra == "runs_row":
            self.big_extra = runs_row(box, collapsed=True)[0]
        elif extra == "triton":
            self.big_extra = triton_card(box)

    def all(self) -> list:
        out = [self.tags, self.caption]
        if self.big_extra is not None:
            out.append(self.big_extra)
        out.append(self.strip)
        if self.big_image is not None:
            out.append(self.big_image)
        return out
