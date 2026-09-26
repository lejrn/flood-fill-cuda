"""Beat 05 `Runs`: why move pixels at all?

Four phases in ~11 s, the screen is cleared between them:
1. one row of 36 pixels with four red spans
2. each red span collapses into one rounded teal run
3. linear bar chart: all pixels / red pixels / runs / blobs, then `25× fewer`
4. stopwatch ch05 58.51 ms (red) -> ch06 1.46 ms (teal), sub-caption and
   the seven-phase pipeline strip with the middle five bracketed `0.64 ms`
"""
from __future__ import annotations

from manim import *
from scenes.style import *

# ---- phase 1/2: one pixel row ----
N_CELLS = 36
SPANS = [(2, 6), (10, 18), (22, 24), (27, 35)]  # inclusive column ranges
CELL = 0.33

# ---- phase 3: bar chart (linear scale on purpose) ----
ROWS = [
    ("all pixels", 81_000_000, GREY),
    ("red pixels", 13_451_960, RED_PX),
    ("runs", 539_207, TEAL),
    ("blobs", 2_522, BLUE),
]
MIN_BAR_W = 0.08  # visibility floor for the two slivers

# ---- phase 4: ch06 pipeline ----
PHASES = ["pack", "count", "scan", "emit", "merge", "flatten", "paint"]


class Runs(BeatScene):
    beat = "05_runs"

    # ------------------------------------------------------------ helpers
    # Every run_time and wait below is a multiple of 0.2 s, which is a whole
    # number of frames at both 15 fps (-ql) and 30 fps (the assembly). The
    # renderer's clock adds exact run_times for cached plays but whole frames
    # for rendered ones, so fractional frame counts would make the padded
    # length depend on what happens to be cached.
    def until(self, t: float) -> None:
        """Wait until the scene clock reaches `t` seconds (whole frames)."""
        fps = config["frame_rate"]
        n = round((t - self.elapsed()) * fps)
        if n > 0:
            self.wait(n / fps)

    def clear_screen(self, run_time: float = 0.4) -> None:
        mobs = list(self.mobjects)
        if mobs:
            self.play(FadeOut(Group(*mobs)), run_time=run_time)

    def fit(self, group: Mobject, pad: float = 0.0) -> Mobject:
        """Scale `group` down (never up) to fit inside the margins."""
        max_w = self.L["w"] - 2 * self.L["margin"] - pad
        max_h = self.L["h"] - 2 * self.L["margin"] - pad
        s = min(1.0, max_w / group.width, max_h / group.height)
        if s < 1.0:
            group.scale(s)
        return group

    # ------------------------------------------------------------ construct
    def construct(self) -> None:
        self.phase_row()    # 0.0 - 4.4 s
        self.phase_chart()  # 4.4 - 8.4 s
        self.phase_clock()  # 8.4 - 10.6 s, then finish() pads to 11.56
        self.finish()

    # ------------------------------------------------------------ phases 1+2
    def phase_row(self) -> None:
        red = [any(a <= i <= b for a, b in SPANS) for i in range(N_CELLS)]
        cells = []
        for i in range(N_CELLS):
            c = Square(CELL, stroke_width=1.2, stroke_color=GRID)
            if red[i]:
                c.set_fill(RED_PX, 1.0)
            else:
                c.set_fill(INK_SOFT, 0.28)
            cells.append(c)
        VGroup(*cells).arrange(RIGHT, buff=0)

        # The row is a group of alternating grey segments and red spans, so a
        # span is a real member of the row's family and can be transformed
        # on its own (the scene restructures the row around it).
        parts, spans, i = [], [], 0
        while i < N_CELLS:
            j = i
            while j + 1 < N_CELLS and red[j + 1] == red[i]:
                j += 1
            seg = VGroup(*cells[i:j + 1])
            parts.append(seg)
            if red[i]:
                spans.append(seg)
            i = j + 1
        row = VGroup(*parts)
        self.fit(row, pad=0.4)
        row.move_to(UP * 0.4)

        cap = caption("one row of pixels").next_to(row, DOWN, buff=0.7)

        self.play(FadeIn(row, lag_ratio=0.02), run_time=0.8)
        self.play(FadeIn(cap, shift=UP * 0.15), run_time=0.4)
        self.until(2.0)

        # phase 2: each red span collapses into one rounded teal run
        cell = spans[0][0].width
        h = cell * 0.7
        runs = []
        for s in spans:
            bar = RoundedRectangle(
                corner_radius=h / 2, width=s.width, height=h,
                stroke_width=0, fill_color=TEAL, fill_opacity=1.0,
            ).move_to(s)
            runs.append(bar)
        cap2 = caption("a red span = one run").move_to(cap)

        # Caption swap is out-then-in (lag_ratio=1) so two different-width
        # captions never overlap mid-fade.
        self.play(
            LaggedStart(
                *[ReplacementTransform(s, b) for s, b in zip(spans, runs)],
                lag_ratio=0.18, run_time=1.4,
            ),
            AnimationGroup(
                FadeOut(cap, shift=UP * 0.12, run_time=0.4),
                FadeIn(cap2, shift=UP * 0.12, run_time=0.4),
                lag_ratio=1,
            ),
        )
        self.until(4.0)
        self.clear_screen()

    # ------------------------------------------------------------ phase 3
    def phase_chart(self) -> None:
        vertical = self.L["vertical"]
        # portrait: shorter bars, same text size; `25× fewer` goes under the chart
        x0, full_w = (-1.4, 2.4) if vertical else (-4.0, 7.3)
        bar_h = 0.6
        ys = [1.8, 0.6, -0.6, -1.8]
        total = ROWS[0][1]

        axis = Line(
            [x0, ys[0] + 0.55, 0], [x0, ys[-1] - 0.55, 0],
            stroke_width=1.5, color=GRID,
        )
        rows = []
        for (name, val, col), y in zip(ROWS, ys):
            w = max(MIN_BAR_W, full_w * val / total)
            bar = Rectangle(
                width=w, height=bar_h, stroke_width=0,
                fill_color=col, fill_opacity=1.0,
            ).move_to([x0 + w / 2, y, 0])
            lab = label(name, size=30)
            lab.move_to([x0 - 0.3 - lab.width / 2, y, 0])
            num = label(f"{val:,}", size=30, color=col, mono=True)
            num.next_to(bar, RIGHT, buff=0.18)
            rows.append((bar, lab, num))

        # `25× fewer` beside the runs row
        runs_bar, _, runs_num = rows[2]
        fewer = label("25× fewer", size=56, color=TEAL, mono=True)
        if vertical:
            fewer.next_to(VGroup(*[m for r in rows for m in r]), DOWN, buff=0.7)
        else:
            fewer.next_to(runs_num, RIGHT, buff=0.55)
            fewer.set_y(runs_bar.get_y())

        # No-op in landscape (the chart already sits inside the margins);
        # scales the whole chart down in a vertical frame.
        chart = VGroup(axis, *[m for r in rows for m in r], fewer)
        self.fit(chart, pad=0.4).move_to(ORIGIN)

        self.play(Create(axis), run_time=0.2)
        for bar, lab, num in rows:
            bar.save_state()
            bar.stretch_to_fit_width(0.01, about_edge=LEFT)
            self.play(FadeIn(lab), Restore(bar), run_time=0.4)
            self.play(FadeIn(num, shift=LEFT * 0.1), run_time=0.2)

        self.until(7.0)
        self.play(FadeIn(fewer, shift=LEFT * 0.3), run_time=0.4)
        self.until(8.0)
        self.clear_screen()

    # ------------------------------------------------------------ phase 4
    def pipeline_strip(self) -> VGroup:
        vertical = self.L["vertical"]
        # portrait: tighter boxes, no arrows, so the words stay legible
        bw, bh, gap = (0.95, 0.5, 0.14) if vertical else (1.22, 0.5, 0.42)
        boxes = VGroup(*[
            RoundedRectangle(
                corner_radius=0.1, width=bw, height=bh,
                stroke_width=1.5, stroke_color=GREY, fill_opacity=0,
            )
            for _ in PHASES
        ]).arrange(RIGHT, buff=gap)

        # One Text for all seven words so they share a baseline (a Text is
        # centred by its own bounding box, so "merge" would otherwise sit
        # higher than "scan"). Ligatures are off so "fl" in "flatten" is
        # two glyphs and spaces count as glyphs: one glyph per character.
        words = label(" ".join(PHASES), size=16 if vertical else 20, color=INK_SOFT, disable_ligatures=True)
        k = 0
        for box, name in zip(boxes, PHASES):
            words[k:k + len(name)].set_x(box.get_x())
            k += len(name) + 1
        words.set_y(boxes.get_y())

        arrows = VGroup()
        for a, b in [] if vertical else zip(boxes[:-1], boxes[1:]):
            arrows.add(Arrow(
                a.get_right(), b.get_left(), buff=0.06,
                stroke_width=2, tip_length=0.14, color=GREY,
            ))

        mid = VGroup(*boxes[1:6])
        y = boxes.get_bottom()[1] - 0.22
        xl, xr = mid.get_left()[0], mid.get_right()[0]
        line_kw = dict(stroke_width=1.5, color=GREY)
        bracket = VGroup(
            Line([xl, y, 0], [xr, y, 0], **line_kw),
            Line([xl, y, 0], [xl, y + 0.12, 0], **line_kw),
            Line([xr, y, 0], [xr, y + 0.12, 0], **line_kw),
        )
        ms = label("0.64 ms", size=22, color=INK_SOFT, mono=True)
        ms.next_to(bracket, DOWN, buff=0.12)
        return VGroup(boxes, words, arrows, bracket, ms)

    def phase_clock(self) -> None:
        sw1 = Stopwatch("ch05 · pixels", 58.51, color=RED_PX).move_to(UP * 1.5)
        sw2 = Stopwatch("ch06 · runs", 1.46, color=TEAL).move_to(UP * 1.5)
        sub = caption("1.46 ms packed mask · 2.96 ms from RGB", size=24)
        self.fit(sub).next_to(sw2, DOWN, buff=0.45)  # fit only matters when vertical
        strip = self.fit(self.pipeline_strip(), pad=0.4).move_to(DOWN * 1.7)

        self.play(FadeIn(sw1, shift=UP * 0.2), run_time=0.4)
        self.until(9.4)
        # Shared glyphs ("ch0", "·", "s", ".", "ms") slide into place, the
        # rest cross-fade: cleaner than a per-glyph morph.
        self.play(TransformMatchingShapes(sw1, sw2), run_time=0.6)
        self.play(
            FadeIn(sub, shift=UP * 0.1),
            FadeIn(strip, lag_ratio=0.04),
            run_time=0.6,
        )
