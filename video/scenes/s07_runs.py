"""Stage 7, runs: a pixel row collapses into runs, then the recoloured image."""
from __future__ import annotations

from manim import AnimationGroup, FadeIn, LaggedStart, ReplacementTransform

from scenes.panes.middle_strip import big_image, runs_row
from scenes.style import ASSETS
from scenes.stage import StageScene, fr


class Runs(StageScene):
    k = 7

    def first_caption(self) -> tuple:
        return "a red span is one run", "{red_px} red pixels → {n_runs} runs, {runs_ratio} fewer"

    def stage(self) -> None:
        box = self.geo.middle
        row, spans, _ = runs_row(box, collapsed=False)
        _, _, bars = runs_row(box, collapsed=True)
        self.play(FadeIn(row, lag_ratio=0.02), run_time=fr(9))
        self.until(self.elapsed() + fr(12))
        self.play(LaggedStart(*[ReplacementTransform(s, b) for s, b in zip(spans, bars)],
                              lag_ratio=0.18), run_time=fr(18))
        self.until(self.elapsed() + fr(9))
        still = big_image(box, ASSETS / self.cfg.thumb, self.cfg.big_fit, self.cfg.big_dy)
        self.swap_caption(self.cfg.caption, self.cfg.caption2, extra_in=[FadeIn(still)], frames=10)
        self.until(self.elapsed() + fr(15))
