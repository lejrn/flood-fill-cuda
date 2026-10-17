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
    # v4's call-out panels (chain_v4_proposal.md, "Where the new panels sit"),
    # each as (centre, (w, h)). Act 1's (D1) sit in 9:16 in the band between
    # the caption and the CPU board's back edge, in 16:9 above the CPU board.
    # The VRAM close-up (D2, act 2) sits under the launch pill (over the dark
    # CPU board), which the proposal's spot would cover.
    gauge: tuple = ()          # D1, the shelf gauge (r03_giant)
    cards: tuple = ()          # D1, the shelf cards: ((SM 0's centre, SM 2's centre), (w, h)), SM 0's lower (r04)
    vram: tuple = ()           # D2, the VRAM close-up (r08, r10_runs)
    side_card: tuple = ()      # the corner beside the floor's front tip, ((x0, y0), (x1, y1)): r05's cards
    # v5, r10: the warp card and the SM close-up it unfolds from, side by side
    # over the floor area, clear of the inset, the act tag and the caption;
    # each as (centre, height): the card's (machine.warp_closeup), the
    # close-up image's (machine.sm_closeup)
    warp: tuple = ()
    sm_side: tuple = ()

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
            gauge=((-2.62, -2.62), (2.7, 1.2)),
            cards=(((-2.70, -3.15), (-2.70, -2.25)), (2.3, 0.8)),
            vram=((0.0, 1.65), (6.8, 3.9)),           # over the floor area, under the inset and the act tag
            side_card=((-3.5, -0.3), (3.5, 3.6)),     # r05's sweep card, over the floor area
            warp=((1.3, 1.65), 3.9),                  # its bottom 0.2 clear of the caption band
            sm_side=((-2.55, 1.4), 2.9),
        )
    top = h / 2
    return Frame(
        floor_c=np.array([-3.25, -0.2, 0]), floor_wh=(6.3, 4.9),
        machine_c=np.array([3.6, -0.08, 0]), machine_wh=(6.8, 5.6),
        caption_c=np.array([0.0, -h / 2 + 0.6, 0]),
        goal_c=np.array([3.55, top - 0.58, 0]), goal_w=6.6,     # its top 0.05 inside the frame
        score_c=np.array([-3.5, top - 0.68, 0]), score_w=6.6,
        card_c=np.array([0.0, 0.2, 0]),
        inset_c=np.array([-6.05, 1.95, 0]),
        tag_c=np.array([-0.9, 2.3, 0]),
        vertical=False,
        gauge=((1.30, 2.05), (2.5, 1.2)),
        cards=(((1.20, 1.30), (1.20, 2.25)), (2.1, 0.85)),
        vram=((-2.75, -0.45), (5.1, 3.9)),       # over the floor area, clear of the inset
        side_card=((-5.3, -2.4), (-0.2, 1.5)),     # r05's sweep card, over the floor area
        warp=((-1.75, -0.55), 3.6),
        sm_side=((-4.7, -1.07), 2.4),            # its top clear of the inset's label
    )
