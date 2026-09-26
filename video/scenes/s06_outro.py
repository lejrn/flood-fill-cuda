"""Beat 06 `Outro`: the whole chain on one log-scale chart, then the closing line.

Phase 1 (0-4.2 s): nine horizontal bars, log10 time axis drawn by hand,
rows appearing top to bottom 0.35 s apart, then `16,000×` beside the
teal bar. Phase 2 (4.2-7.5 s): the chart shrinks to the top and dims,
two centred lines fade in, and the repo name sits bottom-right.
"""
from __future__ import annotations

import math

from manim import *
from scenes.style import *

# (row label, milliseconds, text shown after the bar, colour)
ROWS = [
    ("pure Python", 24083, "24,083 ms", GREY),
    ("@njit", 1346, "1,346", GREY),
    ("ch01", 1222, "~1,222", RED_PX),
    ("ch02", 1916, "~1,916", RED_PX),
    ("ch03", 2181, "~2,181", RED_PX),
    ("ch04", 1368, "~1,368", RED_PX),
    ("ch05", 58.51, "58.51", RED_PX),
    ("ch06 RGB", 2.96, "2.96", GOLD),
    ("ch06 mask", 1.46, "1.46", TEAL),
]
TICKS = [(1, "1 ms"), (10, "10"), (100, "100"), (1000, "1 s"), (10000, "10 s")]

X0 = -3.6       # x of 1 ms
K = 1.7         # units per decade: x(ms) = X0 + K * log10(ms)
PITCH = 0.55    # row pitch
BAR_H = 0.32
Y_TOP = 2.75    # centre of the first row


def x_of(ms: float) -> float:
    return X0 + K * math.log10(ms)


class Outro(BeatScene):
    beat = "06_outro"

    def construct(self) -> None:
        m = self.L["margin"]
        max_w = self.L["w"] - 2 * m
        max_h = self.L["h"] - 2 * m

        # ---- panel A: the chart (axis, faint decade lines, tick labels) ----
        y_axis = Y_TOP - (len(ROWS) - 1) * PITCH - 0.5
        y_grid_top = Y_TOP + 0.25
        grid = VGroup()
        ticks = VGroup()
        tick_labels = VGroup()
        for ms, txt in TICKS:
            x = x_of(ms)
            grid.add(Line([x, y_axis, 0], [x, y_grid_top, 0], stroke_width=1, color=GRID))
            ticks.add(Line([x, y_axis, 0], [x, y_axis - 0.12, 0], stroke_width=2, color=GREY))
            tick_labels.add(label(txt, size=22, color=GREY).move_to([x, y_axis - 0.36, 0]))
        axis = Line([X0, y_axis, 0], [x_of(40000), y_axis, 0], stroke_width=2, color=GREY)
        axes = VGroup(grid, axis, ticks, tick_labels)

        rows = VGroup()
        for i, (name, ms, txt, col) in enumerate(ROWS):
            y = Y_TOP - i * PITCH
            lab = caption(name).move_to([X0 - 0.3, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(
                width=x_of(ms) - X0, height=BAR_H,
                stroke_width=0, fill_color=col, fill_opacity=1,
            ).move_to([X0, y, 0], aligned_edge=LEFT)
            val = label(txt, size=28, color=col, mono=True).next_to(bar, RIGHT, buff=0.15)
            rows.add(VGroup(lab, bar, val))

        # ---- panel B: the one number ----
        big = label("16,000×", size=48, color=TEAL, mono=True)
        body = VGroup(axes, rows)
        if self.L["vertical"]:
            # portrait: chart across the top, the number centred under it
            body.scale_to_fit_width(max_w).move_to(self.L["top"], aligned_edge=UP)
            big.next_to(body, DOWN, buff=0.5)
            shrink = 0.6
        else:
            # landscape: the number slides in beside the teal bar
            big.next_to(rows[-1][2], RIGHT, buff=0.6)
            shrink = 0.42
        chart = VGroup(body, big)
        if self.L["vertical"]:
            chart.move_to(ORIGIN)
        if chart.width > max_w + 1e-6 or chart.height > max_h + 1e-6:
            chart.scale(min(max_w / chart.width, max_h / chart.height)).move_to(ORIGIN)

        # ---- phase 1: 0-4.2 s ----
        self.play(FadeIn(axes), run_time=0.2)
        row_anims = [
            AnimationGroup(
                FadeIn(lab, run_time=0.3),
                GrowFromEdge(bar, LEFT, run_time=0.4),
                FadeIn(val, run_time=0.3),
                lag_ratio=0.5,
            )
            for lab, bar, val in rows
        ]
        # each row's group lasts 0.65 s; a lag ratio of 0.35/0.65 starts a new row every 0.35 s
        self.play(LaggedStart(*row_anims, lag_ratio=0.35 / 0.65))   # 0.2-3.65 s
        self.play(FadeIn(big, shift=LEFT * 0.6), run_time=0.4)       # 3.65-4.05 s
        self.wait(0.15)                                              # hold to 4.2 s

        # ---- phase 2: 4.2-7.5 s ----
        self.play(
            chart.animate.scale(shrink).to_edge(UP, buff=m).set_opacity(0.3),
            run_time=0.7,
        )
        line1 = label("Not a smarter algorithm.", size=48).move_to([0, -0.35, 0])
        line2 = label("A better representation.", size=48, color=TEAL).move_to([0, -1.25, 0])
        for t in (line1, line2):
            if t.width > max_w:
                t.scale_to_fit_width(max_w)
        self.play(FadeIn(line1, shift=UP * 0.2), run_time=0.6)
        self.wait(0.4)
        self.play(FadeIn(line2, shift=UP * 0.2), run_time=0.6)
        name = label("flood-fill-cuda", size=24, color=INK_SOFT, mono=True).to_corner(DR, buff=m)
        self.play(FadeIn(name), run_time=0.4)

        self.finish()
