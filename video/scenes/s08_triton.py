"""Stage 8, Triton: every chapter rebuilt in Triton; the matrix turns into
Numba / Triton (the transition flips it), then the headline ratio."""
from __future__ import annotations

import numpy as np
from manim import FadeIn

from scenes.panes.middle_strip import triton_card
from scenes.stage import StageScene, fr


class Triton(StageScene):
    k = 8

    def stage(self) -> None:
        lang, ratio, faster, ch05 = triton_card(self.geo.middle)
        up = np.array([0, 0.15, 0])
        t = self.target_seconds()
        self.play(FadeIn(lang, shift=up), run_time=fr(9))
        # "Triton is twelve percent" starts at 0.65-0.66 of the beat in both voices, "and still" at 0.83-0.86
        self.until(0.64 * t)
        self.play(FadeIn(ratio, shift=up), run_time=fr(9))
        self.until(0.82 * t)
        self.play(FadeIn(faster), FadeIn(ch05), run_time=fr(9))
