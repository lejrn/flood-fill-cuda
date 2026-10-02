"""Stage 8, outro: the strip and the matrix are complete; the one number."""
from __future__ import annotations

import numpy as np
from manim import FadeIn

from scenes.style import INK, INK_SOFT, TEAL, label
from scenes.stage import StageScene, fr


class Outro(StageScene):
    k = 8

    def stage(self) -> None:
        box = self.geo.middle
        ctx = self.ctx
        l1 = label(f"{ctx['pure']} → {ctx['mask']}", size=30, color=INK, mono=True)
        l2 = label(ctx["ratio"], size=64, color=TEAL, mono=True)
        l3 = label("flood-fill-cuda", size=20, color=INK_SOFT, mono=True)
        l1.move_to(box.at(box.cx, box.cy + 0.9))
        l2.move_to(box.at(box.cx, box.cy - 0.1))
        l3.move_to(box.at(box.cx, box.cy - 1.2))
        for m in (l1, l2, l3):
            if m.width > box.w - 0.2:
                m.scale_to_fit_width(box.w - 0.2)
        self.until(self.elapsed() + fr(6))
        self.play(FadeIn(l1, shift=np.array([0, 0.15, 0])), run_time=fr(9))
        self.until(self.elapsed() + fr(6))
        self.play(FadeIn(l2, shift=np.array([0, 0.15, 0])), run_time=fr(9))
        self.until(self.elapsed() + fr(12))
        self.play(FadeIn(l3), run_time=fr(6))
