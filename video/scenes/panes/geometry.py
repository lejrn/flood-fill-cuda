"""Pane boxes for the three-pane layout.

Landscape: three columns side by side, each the full usable height.
Vertical (9:16, 8 x 14.22 units, see style.py): the blob and the GPU
share the top row, the matrix takes the bottom row. Every pane keeps its
landscape width, so text keeps its size; only the heights differ.
Everything in the panes is placed relative to its Box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MARGIN_X = 0.5
MARGIN_Y = 0.45
GAP = 0.26
WIDTHS = {"left": 5.3, "middle": 4.1, "right": 3.3}   # sums to 12.7 = 14.22 - 2*0.5 - 2*0.26

# vertical: the top row is 4.1 + 0.26 + 3.3 = 7.66 of the 8 units; the GPU
# pane needs 6.35 units of height, the matrix 7.0 (title, 17 rows, legend)
V_TOP_H = 6.4
V_BOTTOM_H = 7.0
V_GAP = 0.25
V_MARGIN_X = 0.12      # left edge of the top row; the GPU lines overhang their box on the right


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def at(self, x: float, y: float) -> np.ndarray:
        return np.array([x, y, 0.0])


@dataclass(frozen=True)
class Geometry:
    left: Box
    middle: Box
    right: Box


def pane_geometry(L: dict) -> Geometry:
    if L.get("vertical"):
        return stacked_geometry(L)
    w, h = L["w"], L["h"]
    usable = w - 2 * MARGIN_X - 2 * GAP
    scale = usable / sum(WIDTHS.values())
    y0, y1 = -h / 2 + MARGIN_Y, h / 2 - MARGIN_Y
    x = -w / 2 + MARGIN_X
    boxes = {}
    for name in ("left", "middle", "right"):
        bw = WIDTHS[name] * scale
        boxes[name] = Box(x, x + bw, y0, y1)
        x += bw + GAP
    return Geometry(**boxes)


def stacked_geometry(L: dict) -> Geometry:
    y1 = (V_TOP_H + V_GAP + V_BOTTOM_H) / 2       # both rows centred on the frame
    top_y0 = y1 - V_TOP_H
    x0 = -L["w"] / 2 + V_MARGIN_X
    middle = Box(x0, x0 + WIDTHS["middle"], top_y0, y1)
    right = Box(middle.x1 + GAP, middle.x1 + GAP + WIDTHS["right"], top_y0, y1)
    bottom_y1 = top_y0 - V_GAP
    left = Box(-WIDTHS["left"] / 2, WIDTHS["left"] / 2, bottom_y1 - V_BOTTOM_H, bottom_y1)
    return Geometry(left=left, middle=middle, right=right)
