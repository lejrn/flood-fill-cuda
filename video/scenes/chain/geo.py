"""Where things sit in the frame, per layout. Nothing else in the chain
cut uses frame coordinates.

9:16 (8 x 14.22), top to bottom: the goal meter, the scoreboard strip,
the floor (inset top left, act tag top right), the caption, the machine
(the CPU and GPU boards).
16:9 (14.22 x 8): the scoreboard top left, the goal meter top right,
the floor left, the machine right, the caption at the bottom centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import config

from scenes.style import is_vertical


@dataclass(frozen=True)
class Frame:
    floor_c: np.ndarray        # centre of the floor's box
    floor_wh: tuple            # its (w, h)
    machine_c: np.ndarray
    machine_wh: tuple
    caption_c: np.ndarray      # centre of the big caption word
    goal_c: np.ndarray         # centre of the goal meter
    goal_w: float              # its width
    score_c: np.ndarray        # centre of the collapsed scoreboard strip
    score_w: float
    card_c: np.ndarray         # centre of the expanded scoreboard card
    inset_c: np.ndarray        # centre of the "you are here" inset
    tag_c: np.ndarray          # centre of the act tag
    vertical: bool

    def split(self, side: str) -> tuple:
        """(centre, (w, h)) of the left or right half of the floor box."""
        w, h = self.floor_wh
        dx = -w / 4 if side == "left" else w / 4
        return self.floor_c + np.array([dx, 0, 0]), (w / 2 - 0.15, h * 0.8)


def frame() -> Frame:
    w, h = config.frame_width, config.frame_height
    if is_vertical():
        top = h / 2
        return Frame(
            floor_c=np.array([0.0, 2.05, 0]), floor_wh=(7.4, 5.0),
            machine_c=np.array([0.0, -4.42, 0]), machine_wh=(7.7, 5.1),
            caption_c=np.array([0.0, -1.15, 0]),
            goal_c=np.array([0.0, top - 0.62, 0]), goal_w=7.4,
            score_c=np.array([0.0, top - 1.55, 0]), score_w=7.4,
            card_c=np.array([0.0, 2.05, 0]),
            inset_c=np.array([-2.95, 4.5, 0]),
            tag_c=np.array([3.2, 4.3, 0]),
            vertical=True,
        )
    top = h / 2
    return Frame(
        floor_c=np.array([-3.25, -0.2, 0]), floor_wh=(6.3, 4.9),
        machine_c=np.array([3.6, -0.08, 0]), machine_wh=(6.8, 5.6),
        caption_c=np.array([0.0, -h / 2 + 0.6, 0]),
        goal_c=np.array([3.55, top - 0.55, 0]), goal_w=6.6,
        score_c=np.array([-3.5, top - 0.68, 0]), score_w=6.6,
        card_c=np.array([0.0, 0.2, 0]),
        inset_c=np.array([-6.05, 1.95, 0]),
        tag_c=np.array([-0.9, 2.3, 0]),
        vertical=False,
    )
