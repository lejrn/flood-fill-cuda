"""Intro: the problem the whole video optimises for.

Two narration beats in one clip (`intro_problem`, `intro_budget`; the
assembly places both wavs inside this clip). A defence camera watches a
clear sky; every frame must be labelled in real time. Four panels stream
the same simulated footage (`assets/make_drone_frames.py`): the camera,
the motion mask, the CPU labelling one frame while its stopwatch runs,
and the GPU labelling every frame. The panels are vertical because the
footage is a vertical Short: one row of four in landscape, two rows of
two in the 9:16 cut. The clip fades to black; stage 0 then fades the
three panes in from black.
"""
from __future__ import annotations

import json

import numpy as np
from manim import DOWN, Dot, FadeIn, Line, Rectangle, RoundedRectangle, ValueTracker, VGroup, VMobject, config

from scenes.panes import data
from scenes.panes.geometry import Box
from scenes.panes.middle_strip import load_image
from scenes.stage import EPS, Replay, fr
from scenes.style import (ASSETS, BG, GRID, INK, INK_SOFT, RED_PX, TEAL, BeatScene, Stopwatch, beat_seconds,
                          is_vertical, label)

# four vertical (9:16) panels: the footage is a vertical Short
PW = 2.9
PH = PW * 16 / 9
if is_vertical():
    # two rows of two; the frame is 8 x 14.22 units (style.py)
    CENTRES = ((-1.65, 3.07), (1.65, 3.07), (-1.65, -2.83), (1.65, -2.83))
    Y_TITLE, Y_BUDGET, Y_VERDICT = 6.59, 6.0, -6.49
else:
    CENTRES = tuple((x, 0.62) for x in (-5.1, -1.7, 1.7, 5.1))
    Y_TITLE, Y_BUDGET, Y_VERDICT = 3.6, -3.05, -3.5
TITLE = ("the problem: find every drone", "in every frame, in real time")
FPS_CAMERA = 30
CAP_FONT, TITLE_FONT, LINE_FONT = 16, 24, 20
COUNT_EVERY = 10          # frames between counter updates (3 Hz reads; 30 Hz dribbles)
SPARK_WINDOW = 90         # frames shown in the moving graph (3 s)


class LiveText(VGroup):
    """A mono line whose text is swapped from an updater. The Cairo renderer
    snapshots the family when a play starts, so the replaced glyphs are
    blanked instead of removed (the Stopwatch trick)."""

    def __init__(self, text: str, size: int, color: str, anchor: np.ndarray):
        super().__init__()
        self.size, self.color_, self.anchor = size, color, anchor
        self.value = label(text, size=size, color=color, mono=True).move_to(anchor, aligned_edge=np.array([-1, 0, 0]))
        self.add(self.value)

    def set(self, text: str) -> None:
        if text == self.value.text:
            return
        new = label(text, size=self.size, color=self.color_, mono=True)
        new.move_to(self.anchor, aligned_edge=np.array([-1, 0, 0]))
        old = self.value
        self.remove(old)
        self.value = new
        self.add(new)
        for m in old.get_family():
            m.points[:] = 0


class BlobCounter(VGroup):
    """Counter + sparkline of the blobs the kernel found, driven by a Replay's
    frame index: the number updates every COUNT_EVERY frames, the graph
    scrolls every frame over the last SPARK_WINDOW frames."""

    def __init__(self, replay: Replay, counts: list, centre: np.ndarray, width: float):
        super().__init__()
        self.replay, self.counts = replay, np.array(counts, dtype=float)
        # the graph spans the clip's own range (rounded to hundreds), not 0..max,
        # so the swing between frames is readable
        self.lo = float(np.floor(self.counts.min() * 0.9 / 100) * 100)
        self.hi = float(np.ceil(self.counts.max() * 1.05 / 100) * 100)
        x0, x1 = centre[0] - width / 2 + 0.42, centre[0] + width / 2 - 0.12
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = centre[1] - 0.38, centre[1] + 0.08
        self.backdrop = RoundedRectangle(corner_radius=0.08, width=width, height=1.02, stroke_width=0,
                                         fill_color=BG, fill_opacity=0.82).move_to(centre + np.array([0, -0.04, 0]))
        self.text = LiveText(self.fmt(0), 14, TEAL, np.array([centre[0] - width / 2 + 0.12, centre[1] + 0.32, 0]))
        self.base = Line([x0, self.y0, 0], [x1, self.y0, 0], stroke_width=1, color=GRID)
        self.top_line = Line([x0, self.y1, 0], [x1, self.y1, 0], stroke_width=1, color=GRID)
        ticks = VGroup(label(f"{int(self.hi):,}", size=9, color=INK_SOFT).move_to([x0 - 0.05, self.y1, 0], aligned_edge=np.array([1, 0, 0])),
                       label(f"{int(self.lo):,}", size=9, color=INK_SOFT).move_to([x0 - 0.05, self.y0, 0], aligned_edge=np.array([1, 0, 0])))
        self.line = VMobject(stroke_color=TEAL, stroke_width=2)
        self.dot = Dot(radius=0.035, color=TEAL)
        self.add(self.backdrop, self.base, self.top_line, ticks, self.line, self.dot, self.text)
        self.draw(self.replay.t * self.replay.fps)
        self.add_updater(lambda m, dt: m.draw(self.replay.t * self.replay.fps))

    def fmt(self, i: int) -> str:
        return f"{int(self.counts[i]):,} blobs in this frame"

    def draw(self, i: int) -> None:
        """`i` is the cumulative frame count; the footage loops, so the window
        indexes the counts cyclically and keeps scrolling through the wrap."""
        n = len(self.counts)
        i = int(i)
        cur = i % n
        self.text.set(self.fmt(cur - cur % COUNT_EVERY))
        lo = max(0, i - SPARK_WINDOW + 1)
        idx = np.arange(lo, i + 1)
        xs = self.x1 - (i - idx) * (self.x1 - self.x0) / (SPARK_WINDOW - 1)
        ys = self.y0 + (self.y1 - self.y0) * (self.counts[idx % n] - self.lo) / (self.hi - self.lo)
        pts = [np.array([x, y, 0.0]) for x, y in zip(xs, ys)]
        if len(pts) == 1:
            pts = pts * 2
        self.line.set_points_as_corners(pts)
        self.dot.move_to(pts[-1])


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
    def centre(self, col: int) -> np.ndarray:
        return np.array([*CENTRES[col], 0.0])

    def panel(self, name: str, col: int, loop: bool = True) -> Replay:
        centre = self.centre(col)
        box = Box(centre[0] - PW / 2, centre[0] + PW / 2, centre[1] - PH / 2, centre[1] + PH / 2)
        return Replay(name, box, fit_wh=(PW, PH), center=centre, loop=loop)

    def frame_box(self, col: int) -> Rectangle:
        return Rectangle(width=PW + 0.04, height=PH + 0.04, stroke_width=1.5, stroke_color=GRID,
                         fill_opacity=0).move_to(self.centre(col))

    def caption(self, line1: str, line2: str, col: int, color=INK_SOFT) -> VGroup:
        x, y = CENTRES[col][0], CENTRES[col][1] - PH / 2 - 0.2
        g = VGroup(label(line1, size=CAP_FONT, color=color).move_to([x, y, 0]),
                   label(line2, size=CAP_FONT - 3, color=INK_SOFT).move_to([x, y - 0.24, 0]))
        for m in g:
            if m.width > PW:
                m.scale_to_fit_width(PW)
        return g

    def construct(self) -> None:
        h = data.headline()
        ms_frame = 1000 / FPS_CAMERA
        t_b = beat_seconds("intro_problem", 14.0) + 0.4      # start of the second beat

        if is_vertical():
            title = VGroup(*[label(t, size=TITLE_FONT, color=INK) for t in TITLE]).arrange(DOWN, buff=0.1)
        else:
            title = label(" ".join(TITLE), size=TITLE_FONT, color=INK)
        title.move_to([0, Y_TITLE, 0])

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
                       size=LINE_FONT, color=INK, mono=True).move_to([0, Y_BUDGET, 0])
        self.play(FadeIn(budget, shift=np.array([0, 0.15, 0])), run_time=fr(6))

        # ---- phase B: the CPU stuck on one frame, the GPU on every frame
        self.until(t_b)
        # the CPU has not finished frame 1: its output is still the bare mask
        cpu_img = load_image(ASSETS / "drones_mask" / "frame_000.png")
        cpu_img.scale(min(PW / cpu_img.width, PH / cpu_img.height)).move_to(self.centre(2))
        mpx = h.n_pixels / 1e6
        cpu_cap = self.caption("CPU · frame 1, still labelling", f"{mpx:.0f} Mpx per frame", 2, color=RED_PX)
        sw = Stopwatch("frame 1, so far", 0.0, size=30, color=RED_PX)
        sw.move_to(self.centre(2))
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
        # the kernel's blob count per frame, from the frame set's metadata
        meta = json.loads((ASSETS / "drones_labels" / "meta.json").read_text(encoding="utf-8"))
        counter = BlobCounter(gpu, meta["blobs_per_frame"],
                              self.centre(3) + np.array([0, -PH / 2 + 0.62, 0]), PW - 0.16)
        # FadeIn is a Transform: it pairs glyphs once, at its start, so a count
        # that gains a digit mid-fade ("936" -> "1,104") would lose its last
        # glyphs. Hold the counter still for the fade, then let it run.
        counter.suspend_updating()
        self.play(FadeIn(counter), run_time=fr(6))
        counter.resume_updating()

        self.until(t_b + 5.0)
        verdict = VGroup(
            label(f"budget {ms_frame:.0f} ms", size=LINE_FONT, color=INK, mono=True),
            label(f"pure Python {data.fmt_ms(h.pure_ms)}", size=LINE_FONT, color=RED_PX, mono=True),
            label(f"@njit {data.fmt_ms(h.njit_ms)}", size=LINE_FONT, color=RED_PX, mono=True),
            label(f"GPU {data.fmt_ms(h.ch06_mask_ms)}", size=LINE_FONT, color=TEAL, mono=True),
        )
        if is_vertical():
            verdict.arrange_in_grid(2, 2, buff=(0.55, 0.14))
        else:
            verdict.arrange(direction=np.array([1, 0, 0]), buff=0.55)
        verdict.move_to([0, Y_VERDICT, 0])
        self.play(FadeIn(verdict, lag_ratio=0.2), run_time=fr(9))

        # let the clocks run to the end of the narration, then fade out
        self.until(self.target_seconds() + 0.4 - 0.5)
        sw.remove_updater(on_clock)
        clock.clear_updaters()
        counter.clear_updaters()
        for r in (sky, mask, gpu):
            r.stop()
        self.finish(tail=0.4, fade=0.5, frozen=False)
