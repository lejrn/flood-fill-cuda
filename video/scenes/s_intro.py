"""Intro: the problem the whole video optimises for.

Two narration beats in one clip (`intro_problem`, `intro_budget`; the
assembly places both wavs inside this clip). A defence camera watches a
clear sky; every frame must be labelled in real time. Four panels stream
the same simulated footage (`assets/make_drone_frames.py`): the camera,
the motion mask, the CPU labelling one frame while its stopwatch runs,
and the GPU labelling every frame. The panels are vertical because the
footage is a vertical Short. The clip fades to black; stage 0 then fades
the three panes in from black.
"""
from __future__ import annotations

import numpy as np
from manim import FadeIn, Rectangle, RoundedRectangle, ValueTracker, VGroup, config

from scenes.panes import data
from scenes.panes.geometry import Box
from scenes.panes.middle_strip import load_image
from scenes.stage import EPS, Replay, fr
from scenes.style import ASSETS, BG, GRID, INK, INK_SOFT, RED_PX, TEAL, BeatScene, Stopwatch, beat_seconds, label

# four vertical (9:16) panels in a row: the footage is a vertical Short
PW = 2.9
PH = PW * 16 / 9
COL_X = (-5.1, -1.7, 1.7, 5.1)
ROW_Y = 0.62
FPS_CAMERA = 30
CAP_FONT, TITLE_FONT, LINE_FONT = 16, 24, 20


class Intro(BeatScene):
    beat = "intro_problem"

    def target_seconds(self) -> float:
        return (beat_seconds("intro_problem", 14.0) + 0.4
                + beat_seconds("intro_budget", 15.0))

    def until(self, t: float) -> None:
        """Live wait (replays tick) to `t` seconds, whole frames."""
        fps = config.frame_rate
        n = round((t - self.elapsed()) * fps)
        if n >= 2:
            self.wait(n / fps - EPS, frozen_frame=False)

    # ---- pieces
    def panel(self, name: str, col: int, loop: bool = True) -> Replay:
        centre = np.array([COL_X[col], ROW_Y, 0.0])
        box = Box(centre[0] - PW / 2, centre[0] + PW / 2, centre[1] - PH / 2, centre[1] + PH / 2)
        return Replay(name, box, fit_wh=(PW, PH), center=centre, loop=loop)

    def frame_box(self, col: int) -> Rectangle:
        return Rectangle(width=PW + 0.04, height=PH + 0.04, stroke_width=1.5, stroke_color=GRID,
                         fill_opacity=0).move_to([COL_X[col], ROW_Y, 0])

    def caption(self, line1: str, line2: str, col: int, color=INK_SOFT) -> VGroup:
        y = ROW_Y - PH / 2 - 0.2
        g = VGroup(label(line1, size=CAP_FONT, color=color).move_to([COL_X[col], y, 0]),
                   label(line2, size=CAP_FONT - 3, color=INK_SOFT).move_to([COL_X[col], y - 0.24, 0]))
        for m in g:
            if m.width > PW:
                m.scale_to_fit_width(PW)
        return g

    def construct(self) -> None:
        h = data.headline()
        ms_frame = 1000 / FPS_CAMERA
        t_b = beat_seconds("intro_problem", 14.0) + 0.4      # start of the second beat

        title = label("the problem: find every drone in every frame, in real time",
                      size=TITLE_FONT, color=INK).move_to([0, 3.6, 0])

        # ---- phase A: camera, then the motion mask, then the frame budget
        sky = self.panel("drones_sky", 0)
        self.play(FadeIn(title), FadeIn(self.frame_box(0)), FadeIn(sky),
                  FadeIn(self.caption("camera", f"a drone show, {FPS_CAMERA} fps", 0)),
                  run_time=fr(9))
        sky.start(FPS_CAMERA)

        self.until(3.6)
        mask = self.panel("drones_mask", 1)
        mask.sync_to(sky)
        self.play(FadeIn(self.frame_box(1)), FadeIn(mask),
                  FadeIn(self.caption("filter", "the sky stripped away, blobs left", 1)),
                  run_time=fr(9))

        self.until(8.0)
        budget = label(f"{FPS_CAMERA} frames per second → one frame every {ms_frame:.0f} ms",
                       size=LINE_FONT, color=INK, mono=True).move_to([0, -3.05, 0])
        self.play(FadeIn(budget, shift=np.array([0, 0.15, 0])), run_time=fr(6))

        # ---- phase B: the CPU stuck on one frame, the GPU on every frame
        self.until(t_b)
        # the CPU has not finished frame 1: its output is still the bare mask
        cpu_img = load_image(ASSETS / "drones_mask" / "frame_000.png")
        cpu_img.scale(min(PW / cpu_img.width, PH / cpu_img.height)).move_to([COL_X[2], ROW_Y, 0])
        mpx = h.n_pixels / 1e6
        cpu_cap = self.caption("CPU · frame 1, still labelling", f"{mpx:.0f} Mpx per frame", 2, color=RED_PX)
        sw = Stopwatch("frame 1, so far", 0.0, size=30, color=RED_PX)
        sw.move_to([COL_X[2], ROW_Y, 0])
        backdrop = RoundedRectangle(corner_radius=0.12, width=2.6, height=1.15, stroke_width=0,
                                    fill_color=BG, fill_opacity=0.96).move_to(sw)
        clock = ValueTracker(0.0)

        def on_clock(m):
            ms = clock.get_value() * 1000.0
            if data.fmt_ms(ms) != m.value.text:
                m.set_ms(ms, RED_PX)

        self.play(FadeIn(self.frame_box(2)), FadeIn(cpu_img), FadeIn(cpu_cap), run_time=fr(9))
        self.play(FadeIn(backdrop), FadeIn(sw), run_time=fr(4))
        sw.add_updater(on_clock)
        clock.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(clock)

        self.until(t_b + 2.0)
        gpu = self.panel("drones_labels", 3)
        gpu.sync_to(sky)
        gpu_cap = self.caption("GPU · every frame", f"{data.fmt_ms(h.ch06_mask_ms)} per frame", 3,
                               color=TEAL)
        self.play(FadeIn(self.frame_box(3)), FadeIn(gpu), FadeIn(gpu_cap), run_time=fr(9))

        self.until(t_b + 5.0)
        verdict = VGroup(
            label(f"budget {ms_frame:.0f} ms", size=LINE_FONT, color=INK, mono=True),
            label(f"pure Python {data.fmt_ms(h.pure_ms)}", size=LINE_FONT, color=RED_PX, mono=True),
            label(f"@njit {data.fmt_ms(h.njit_ms)}", size=LINE_FONT, color=RED_PX, mono=True),
            label(f"GPU {data.fmt_ms(h.ch06_mask_ms)}", size=LINE_FONT, color=TEAL, mono=True),
        ).arrange(direction=np.array([1, 0, 0]), buff=0.55).move_to([0, -3.5, 0])
        self.play(FadeIn(verdict, lag_ratio=0.2), run_time=fr(9))

        # let the clocks run to the end of the narration, then fade out
        self.until(self.target_seconds() + 0.4 - 0.5)
        sw.remove_updater(on_clock)
        clock.clear_updaters()
        for r in (sky, mask, gpu):
            r.stop()
        self.finish(tail=0.4, fade=0.5, frozen=False)
