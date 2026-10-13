"""Where things sit in the frame, per layout. Nothing else in the chain
cut uses frame coordinates.

16:9 (14.22 x 8): the floor left, the machine right, the caption at the
bottom centre, the goal chip top right.
9:16 (8 x 14.22): the goal chip on top, the floor, the caption in the
middle of the frame, the machine below.
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
    goal_c: np.ndarray         # centre of the goal chip
    note_c: np.ndarray         # small notes under the floor (bars, counters)
    vertical: bool


def frame() -> Frame:
    w, h = config.frame_width, config.frame_height
    if is_vertical():
        return Frame(
            floor_c=np.array([0.0, 2.55, 0]), floor_wh=(7.2, 6.3),
            machine_c=np.array([0.0, -4.35, 0]), machine_wh=(7.4, 4.9),
            caption_c=np.array([0.0, -1.3, 0]),
            goal_c=np.array([0.0, h / 2 - 0.55, 0]),
            note_c=np.array([0.0, -0.2, 0]),
            vertical=True,
        )
    return Frame(
        floor_c=np.array([-3.45, 0.6, 0]), floor_wh=(6.2, 5.7),
        machine_c=np.array([3.55, 0.5, 0]), machine_wh=(6.6, 5.3),
        caption_c=np.array([0.0, -h / 2 + 0.72, 0]),
        goal_c=np.array([w / 2 - 1.75, h / 2 - 0.42, 0]),
        note_c=np.array([-3.45, -3.0, 0]),
        vertical=False,
    )
