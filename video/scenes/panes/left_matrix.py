"""Left pane: the shapes x chapters matrix.

Rows are the 17 overview scenes, columns the stages. A cell glows when the
chapter's fastest measured variant beats the CPU @njit time on that row;
estimated cells are dashed and never glow; n/a cells are hollow. The CPU
column always prints its ms; a stage column prints ms only while it is
the current one, then keeps just its glow.
"""
from __future__ import annotations

import numpy as np
from manim import (
    Circle, DashedVMobject, Dot, Line, Rectangle, Square, VGroup, VMobject,
    LEFT, RIGHT,
)

from scenes.style import BLUE, GREY, INK, INK_SOFT, RED_PX, TEAL, label
from scenes.panes import data
from scenes.panes.geometry import Box

ROW_PITCH = 0.34
CELL_W, CELL_H = 0.42, 0.26
COL_PITCH = 0.49
TITLE_FONT, HEAD_FONT, NAME_FONT, NUM_FONT, LEGEND_FONT = 16, 9, 14, 14, 11


def row_glyph(kind: str) -> VMobject:
    """A small red glyph for the row's shape family, about 0.2 units."""
    fill = dict(stroke_width=0, fill_color=RED_PX, fill_opacity=1.0)
    line = dict(stroke_width=2.0, stroke_color=RED_PX)
    if kind == "square":
        return Square(0.17, **fill)
    if kind == "disk":
        return Circle(radius=0.095, **fill)
    if kind == "snake":
        pts = [[-0.1, 0.09], [0.1, 0.09], [0.1, 0.03], [-0.1, 0.03], [-0.1, -0.03],
               [0.1, -0.03], [0.1, -0.09], [-0.1, -0.09]]
        m = VMobject(**line)
        m.set_points_as_corners([np.array([x, y, 0.0]) for x, y in pts])
        return m
    if kind == "comb":
        g = VGroup(Line([-0.1, -0.09, 0], [0.1, -0.09, 0], **line))
        for x in (-0.09, -0.03, 0.03, 0.09):
            g.add(Line([x, -0.09, 0], [x, 0.09, 0], **line))
        return g
    if kind == "two_sq":
        return VGroup(Square(0.09, **fill).shift(LEFT * 0.06),
                      Square(0.09, **fill).shift(RIGHT * 0.06))
    if kind == "asym":
        return VGroup(Square(0.14, **fill).shift(LEFT * 0.045),
                      Square(0.06, **fill).shift(RIGHT * 0.075))
    if kind == "noise":
        pts = [(-0.08, 0.06), (0.02, 0.08), (0.08, 0.0), (-0.05, -0.03),
               (0.04, -0.07), (-0.09, -0.08)]
        return VGroup(*[Dot(np.array([x, y, 0.0]), radius=0.025, color=RED_PX) for x, y in pts])
    # picture
    frame = Rectangle(width=0.2, height=0.16, stroke_width=1.5, stroke_color=RED_PX, fill_opacity=0)
    return VGroup(frame, Square(0.06, **fill).shift(LEFT * 0.03 + np.array([0, -0.02, 0])))


def cell_mobject(cell: data.Cell, final: bool, centre: np.ndarray) -> VMobject:
    kw = dict(width=CELL_W, height=CELL_H)
    kind = cell.kind
    if kind == "cpu":
        m = Rectangle(**kw, stroke_width=0, fill_color=INK_SOFT, fill_opacity=0.10)
    elif kind == "fast":
        m = Rectangle(**kw, stroke_width=0, fill_color=TEAL if final else BLUE,
                      fill_opacity=data.glow_opacity(cell.speedup))
    elif kind == "slow":
        if cell.speedup < 0.1:
            m = Rectangle(**kw, stroke_width=0, fill_color=RED_PX, fill_opacity=0.22)
        else:
            m = Rectangle(**kw, stroke_width=0, fill_color=GREY, fill_opacity=0.15)
    elif kind == "est":
        base = Rectangle(**kw, stroke_width=1.0, stroke_color=GREY, fill_opacity=0)
        m = DashedVMobject(base, num_dashes=12, dashed_ratio=0.55, color=GREY)
    else:  # na
        m = Rectangle(**kw, stroke_width=0.8, stroke_color=GREY, stroke_opacity=0.5, fill_opacity=0)
    return m.move_to(centre)


def cell_number(cell: data.Cell, centre: np.ndarray) -> VMobject:
    if cell.kind in ("est", "na") or cell.ms is None:
        return VGroup()
    color = INK if cell.kind in ("fast", "cpu") else INK_SOFT
    return label(data.fmt_cell_ms(cell.ms), size=NUM_FONT, color=color, mono=True).move_to(centre)


class MatrixPane:
    def __init__(self, bench: data.Bench, box: Box, n_cols: int, current: int | None):
        self.bench, self.box = bench, box
        self.n_cols, self.current = n_cols, current
        self.x_glyph = box.x0 + 0.14
        self.x_name = box.x0 + 0.34
        self.x_col0 = box.x0 + 1.34 + CELL_W / 2
        self.y_title = box.y1 - 0.12
        self.y_head = box.y1 - 0.58
        self.y_row0 = box.y1 - 1.00
        self.y_legend = box.y0 + 0.16
        self.static = self._static()
        self.headers = [self.header(j) for j in range(n_cols)]
        self.cols = [self.col_group(j) for j in range(n_cols)]
        self.nums = [self.num_group(j) if self.shows_numbers(j) else VGroup()
                     for j in range(n_cols)]

    # ---- geometry
    def row_y(self, r: int) -> float:
        return self.y_row0 - r * ROW_PITCH

    def col_x(self, j: int) -> float:
        return self.x_col0 + j * COL_PITCH

    def cell_centre(self, r: int, j: int) -> np.ndarray:
        return self.box.at(self.col_x(j), self.row_y(r))

    def shows_numbers(self, j: int) -> bool:
        return j == 0 or j == self.current

    # ---- builders (pure: same input, same mobjects)
    def _static(self) -> VGroup:
        g = VGroup()
        title = label("ms on 17 shapes · CPU @njit vs each chapter", size=TITLE_FONT, color=INK_SOFT)
        title.move_to(self.box.at(self.box.x0, self.y_title), aligned_edge=LEFT)
        g.add(title)
        for r, row in enumerate(self.bench.rows):
            y = self.row_y(r)
            g.add(row_glyph(row.glyph).move_to(self.box.at(self.x_glyph, y)))
            g.add(label(row.name, size=NAME_FONT, color=INK_SOFT)
                  .move_to(self.box.at(self.x_name, y), aligned_edge=LEFT))
        l1 = label("glow = faster than the CPU · brighter = bigger win", size=LEGEND_FONT, color=GREY)
        l2 = label("dashed = estimated, one launch per blob · hollow = n/a", size=LEGEND_FONT, color=GREY)
        l1.move_to(self.box.at(self.box.x0, self.y_legend + 0.09), aligned_edge=LEFT)
        l2.move_to(self.box.at(self.box.x0, self.y_legend - 0.09), aligned_edge=LEFT)
        g.add(l1, l2)
        return g

    def header(self, j: int) -> VGroup:
        col = data.COLUMNS[j]
        x = self.col_x(j)
        g = VGroup()
        for i, line in enumerate(col.header):
            if line:
                g.add(label(line, size=HEAD_FONT, color=INK_SOFT)
                      .move_to(self.box.at(x, self.y_head + 0.08 - 0.15 * i)))
        return g

    def col_group(self, j: int) -> VGroup:
        col = data.COLUMNS[j]
        return VGroup(*[cell_mobject(row.cells[col.key], col.final, self.cell_centre(r, j))
                        for r, row in enumerate(self.bench.rows)])

    def num_group(self, j: int) -> VGroup:
        col = data.COLUMNS[j]
        return VGroup(*[cell_number(row.cells[col.key], self.cell_centre(r, j))
                        for r, row in enumerate(self.bench.rows)])

    def all(self) -> list:
        return [self.static, *self.headers, *self.cols, *self.nums]
