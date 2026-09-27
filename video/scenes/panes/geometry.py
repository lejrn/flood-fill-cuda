"""Pane boxes for the three-pane layout.

Landscape only for now: three columns side by side, each the full usable
height. Everything in the panes is placed relative to its Box, so a
stacked (vertical) variant only needs another `pane_geometry`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MARGIN_X = 0.5
MARGIN_Y = 0.45
GAP = 0.26
WIDTHS = {"left": 5.3, "middle": 4.1, "right": 3.3}   # sums to 12.7 = 14.22 - 2*0.5 - 2*0.26


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
