"""The chain cut's intro clip: beats i0_problem and i1_compare in one clip.

i0: the camera films a drone show, a filter turns each drone into a white
blob, and every frame has a 33 ms budget (the four panels of
`scenes/s_intro.py`: camera, filter, then two panels for i1).
i1: on a drone frame the compiled CPU code keeps up (a measured card:
CPU vs GPU against the 33 ms line, the GPU labelling every frame beside
it). Then "scale up": the card swaps to the 81 Mpx test picture, the drone
panels freeze and dim, the chain's goal meter comes in, and a CPU
stopwatch runs in real time past 33 ms to the @njit time, beside "GPU ?".
"Two places: size and count" brings the scoreboard strip; the clip ends on
exactly `chain_state(2)` (goal meter + strip), the first picture of
r00_zoom. The assembly places the i1 wav at len(i0) + 0.4 s.

Every number comes from `scenes.chain.numbers` (or from the assets).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, DashedLine, FadeIn, FadeOut, GrowFromEdge, Indicate, Rectangle,
                   RoundedRectangle, VGroup, config)

from narration.common import read_words
from scenes.chain import geo
from scenes.chain import numbers as nb
from scenes.chain import overlays as ov
from scenes.chain.state import chain_state, nums
from scenes.panes.geometry import Box
from scenes.panes.middle_strip import load_image
from scenes.s_intro import (CAP_FONT, FPS_CAMERA, LINE_FONT, PH, PW, TITLE, TITLE_FONT, BlobCounter)
from scenes.s_intro import CENTRES as CENTRES_V
from scenes.s_intro import Y_BUDGET as Y_BUDGET_V
from scenes.s_intro import Y_TITLE as Y_TITLE_V
from scenes.stage import EPS, Replay, fr
from scenes.style import (ASSETS, BG, GREY, GRID, INK, INK_SOFT, OUT, RED_PX, TEAL, BeatScene, beat_seconds,
                          is_vertical, label, voice)

GAP = 0.4               # silence between the two beats inside this clip (assembly)
DIM_TO = 0.4            # the drone panels' brightness after "Scale up"

if is_vertical():
    # s_intro's 2 x 2 grid fits under the goal meter as it is
    CENTRES, Y_TITLE, Y_BUDGET = CENTRES_V, Y_TITLE_V, Y_BUDGET_V
else:
    # s_intro's 1 x 4 row, lowered so its top clears the goal meter (top right)
    CENTRES = tuple((x, 0.22) for x in (-5.1, -1.7, 1.7, 5.1))
    Y_TITLE, Y_BUDGET = 3.6, -3.38


def ms_text(ms: float) -> str:
    return ov.fmt_ms(ms)


class Readout(VGroup):
    """A mono number swapped in place from an updater (s_intro's LiveText,
    aligned on the right edge and with a colour per value)."""

    def __init__(self, text: str, size: int, color: str, anchor: np.ndarray):
        super().__init__()
        self.size, self.anchor = size, anchor
        self.value = self._make(text, color)
        self.key = (text, color)
        self.add(self.value)

    def _make(self, text: str, color: str):
        return label(text, size=self.size, color=color, mono=True).move_to(self.anchor, aligned_edge=RIGHT)

    def set(self, text: str, color: str) -> None:
        if (text, color) == self.key:
            return
        new = self._make(text, color)
        old = self.value
        self.remove(old)
        self.value = new
        self.key = (text, color)
        self.add(new)
        for m in old.get_family():
            m.points[:] = 0


class MsCard(VGroup):
    """A small card of horizontal time bars on a linear axis 0 .. max_ms,
    with the 33 ms budget as a dashed teal line. rows: (name, ms or None);
    None draws no bar and a "?" value."""

    BAR_H = 0.13

    def __init__(self, centre: np.ndarray, width: float, max_ms: float, rows: list, budget_ms: float):
        super().__init__()
        h = 0.56 + 0.48 * len(rows)
        self.box = RoundedRectangle(corner_radius=0.08, width=width, height=h, stroke_color=GRID, stroke_width=1.5,
                                    fill_color=BG, fill_opacity=0.94).move_to(centre)
        self.x0 = self.box.get_left()[0] + 0.14
        self.x1 = self.box.get_right()[0] - 0.14
        self.max_ms = max_ms
        top = self.box.get_top()[1]
        xb = self.x(budget_ms)
        y_rows = [top - 0.2 - 0.48 * i for i in range(len(rows))]
        # the budget line crosses only the bar strips, never the row names;
        # its label sits under the last bar
        self.budget = VGroup(*[
            DashedLine([xb, y - 0.14, 0], [xb, y - 0.30, 0], dash_length=0.04, stroke_color=TEAL, stroke_width=2)
            for y in y_rows],
            label(f"{budget_ms:.0f} ms budget", 12, TEAL, mono=True))
        lab = self.budget[-1]
        lab.move_to([xb, y_rows[-1] - 0.46, 0])
        # keep the label inside the card (the line may sit near an edge)
        if lab.get_left()[0] < self.x0:
            lab.shift(RIGHT * (self.x0 - lab.get_left()[0]))
        if lab.get_right()[0] > self.x1:
            lab.shift(LEFT * (lab.get_right()[0] - self.x1))
        self.names, self.values, self.bars = VGroup(), [], []
        for (name, ms), y in zip(rows, y_rows):
            nm = label(name, 13, INK_SOFT).move_to([self.x0, y, 0], aligned_edge=LEFT)
            val = Readout("?" if ms is None else ms_text(ms), 13, INK_SOFT if ms is None else INK,
                          np.array([self.x1, y, 0]))
            self.names.add(nm)
            self.values.append(val)
            yb = y - 0.22
            bar = Rectangle(width=max(self.x(ms) - self.x0, 0.01) if ms is not None else 0.01, height=self.BAR_H,
                            stroke_width=0, fill_color=GREY, fill_opacity=1 if ms else 0)
            bar.move_to([self.x0 + bar.width / 2, yb, 0])
            self.bars.append(bar)
        self.add(self.box, *self.bars, self.budget, self.names, *self.values)

    def x(self, ms: float) -> float:
        return self.x0 + (self.x1 - self.x0) * min(max(ms / self.max_ms, 0.0), 1.0)

    def set_bar(self, i: int, ms: float, color: str) -> None:
        bar = self.bars[i]
        w = max(self.x(ms) - self.x0, 0.01)
        y = bar.get_center()[1]
        bar.stretch_to_fit_width(w)
        bar.move_to([self.x0 + w / 2, y, 0])
        bar.set_fill(color, opacity=1 if ms > 0 else 0)


class Intro(BeatScene):
    beat = "i0_problem"

    def setup(self) -> None:
        super().setup()
        self.words = read_words(OUT / voice())
        self.t_b = beat_seconds("i0_problem", 8.0) + GAP        # start of i1 in this clip
        self.num = nums()
        self.der = nb.derived()
        self.G = geo.frame()

    def target_seconds(self) -> float:
        return beat_seconds("i0_problem", 8.0) + GAP + beat_seconds("i1_compare", 12.5)

    # ---- time
    def cue(self, beat: str, word: str, n: int = 1, end: bool = False) -> float:
        """Clip time of the n-th `word` of `beat` (i1 is offset by len(i0) + 0.4 s)."""
        off = 0.0 if beat == "i0_problem" else self.t_b
        norm = lambda w: "".join(ch for ch in w.lower() if ch.isalnum() or ch in "'-")
        hits = [w for w in self.words.get(beat, []) if norm(w[0]) == norm(word)]
        if len(hits) < n:
            raise ValueError(f"{beat}: no occurrence {n} of {word!r}")
        return off + float(hits[n - 1][2] if end else hits[n - 1][1])

    def until(self, t: float) -> None:
        """Live wait (replays tick) to `t` seconds, whole frames."""
        fps = config.frame_rate
        n = round((t - self.elapsed()) * fps)
        if n >= 1:
            self.wait(n / fps - EPS, frozen_frame=False)

    # ---- pieces
    def centre(self, col: int) -> np.ndarray:
        return np.array([*CENTRES[col], 0.0])

    def panel(self, name: str, col: int, loop: bool = True) -> Replay:
        c = self.centre(col)
        box = Box(c[0] - PW / 2, c[0] + PW / 2, c[1] - PH / 2, c[1] + PH / 2)
        return Replay(name, box, fit_wh=(PW, PH), center=c, loop=loop)

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

    def dimmer(self, col: int) -> Rectangle:
        """A BG veil over a panel and its caption: what shows through is DIM_TO bright."""
        c = self.centre(col)
        r = Rectangle(width=PW + 0.2, height=PH + 0.75, stroke_width=0, fill_color=BG, fill_opacity=1 - DIM_TO)
        return r.move_to(c + np.array([0, -0.3, 0]))

    def card_centre(self, col: int, rows: int) -> np.ndarray:
        h = 0.56 + 0.48 * rows
        c = self.centre(col)
        return c + np.array([0, -PH / 2 + 0.1 + h / 2, 0])

    def write_snap_time(self) -> None:
        """<clip>.snap.json: build/chain_snapcheck.py then compares the frame
        before the swap to chain_state(2) with the frame after it."""
        fw = getattr(self.renderer, "file_writer", None)
        path = getattr(fw, "movie_file_path", None)
        if path:
            Path(str(path) + ".snap.json").write_text(json.dumps({"snap_s": self.elapsed()}), encoding="utf-8")

    # ---- the clip
    def construct(self) -> None:
        n, d = self.num, self.der
        budget = d["budget_ms"]
        cue = self.cue

        if is_vertical():
            title = VGroup(*[label(t, size=TITLE_FONT, color=INK) for t in TITLE]).arrange(DOWN, buff=0.1)
        else:
            title = label(" ".join(TITLE), size=TITLE_FONT, color=INK)
        title.move_to([0, Y_TITLE, 0])

        # ================= i0: camera, filter, budget =================
        sky = self.panel("drones_sky", 0)
        box0 = self.frame_box(0)
        cap0 = self.caption("camera", f"a drone show, {FPS_CAMERA} fps", 0)
        self.play(FadeIn(title), FadeIn(box0), FadeIn(sky), FadeIn(cap0), run_time=fr(9))
        sky.start(FPS_CAMERA)

        self.until(cue("i0_problem", "filter"))
        mask = self.panel("drones_mask", 1)
        mask.sync_to(sky)
        box1 = self.frame_box(1)
        cap1 = self.caption("filter", "the sky stripped away, blobs left", 1)
        self.play(FadeIn(box1), FadeIn(mask), FadeIn(cap1), run_time=fr(9))

        self.until(cue("i0_problem", "thirty-three"))
        budget_line = label(f"{FPS_CAMERA} frames per second -> one frame every {budget:.0f} ms",
                            size=LINE_FONT, color=INK, mono=True).move_to([0, Y_BUDGET, 0])
        if budget_line.width > config.frame_width - 0.4:
            budget_line.scale_to_fit_width(config.frame_width - 0.4)
        self.play(FadeIn(budget_line, shift=UP * 0.15), run_time=fr(6))

        # ================= i1: the drone frame card, the GPU panel =================
        self.until(cue("i1_compare", "On"))
        drone_img = load_image(ASSETS / "drones_mask" / "frame_000.png")
        drone_img.scale(min(PW / drone_img.width, PH / drone_img.height)).move_to(self.centre(2))
        box2 = self.frame_box(2)
        cap2 = self.caption("one drone frame", f"median of {int(n['drone_frames'])} frames, one session", 2)
        dcard = MsCard(self.card_centre(2, 2), PW - 0.12, budget * 1.2,
                       [("CPU (compiled)", n["drone_cpu_ms"]), ("GPU", n["drone_gpu_ms"])], budget)
        dbars = dcard.bars
        dcard.remove(*dbars, *dcard.values)        # bars and values arrive at "CPU"

        gpu = self.panel("drones_labels", 3)
        gpu.sync_to(sky)
        box3 = self.frame_box(3)
        with_px = gpu.frame(0).shape
        mpx = with_px[0] * with_px[1] / 1e6
        cap3 = self.caption("GPU, every frame", f"drone frames, {mpx:.2f} Mpx", 3, color=TEAL)
        self.play(FadeIn(box2), FadeIn(drone_img), FadeIn(cap2), FadeIn(dcard),
                  FadeIn(box3), FadeIn(gpu), FadeIn(cap3), run_time=fr(9))
        meta = json.loads((ASSETS / "drones_labels" / "meta.json").read_text(encoding="utf-8"))
        counter = BlobCounter(gpu, meta["blobs_per_frame"],
                              self.centre(3) + np.array([0, -PH / 2 + 0.62, 0]), PW - 0.16)
        # hold the counter still while it fades in (s_intro: FadeIn pairs glyphs once)
        counter.suspend_updating()
        self.play(FadeIn(counter), run_time=fr(5))
        counter.resume_updating()

        self.until(cue("i1_compare", "CPU"))
        dbars[0].set_fill(GREY, 1)
        dbars[1].set_fill(TEAL, 1)
        dcard.add(*dbars, *dcard.values)
        self.play(GrowFromEdge(dbars[0], LEFT), GrowFromEdge(dbars[1], LEFT),
                  FadeIn(dcard.values[0]), FadeIn(dcard.values[1]), run_time=fr(8))

        # ================= "Scale up": the 81 Mpx card =================
        self.until(cue("i1_compare", "Scale"))
        for r in (sky, mask, gpu):
            r.stop()
        counter.clear_updaters()
        dims = [self.dimmer(c) for c in (0, 1, 3)]
        self.play(FadeOut(drone_img), FadeOut(cap2), FadeOut(dcard), *[FadeIn(m) for m in dims],
                  run_time=fr(5))
        red = load_image(ASSETS / "ch05_input_blobs" / "frame_000.png")
        red.scale_to_fit_width(PW)
        red.move_to(self.centre(2) + np.array([0, PH / 2 - PW / 2, 0]))
        cap2b = self.caption(f"{n['n_pixels'] / 1e6:.0f} Mpx test frame, {int(n['n_blobs']):,} blobs",
                             "not a drone frame", 2)
        njit = n["njit_ms"]
        bcard = MsCard(self.card_centre(2, 2), PW - 0.12, njit * 1.1,
                       [("CPU (compiled)", 0.0), ("GPU", None)], budget)
        # the CPU value stays hidden until its stopwatch starts (a static
        # "0.00 ms" would read like a measured result)
        cpu_val = bcard.values[0]
        bcard.remove(cpu_val)
        self.play(FadeIn(red), FadeIn(cap2b), FadeIn(bcard), run_time=fr(7))

        # ================= "same thirty-three": the goal meter =================
        self.until(cue("i1_compare", "same"))
        self.play(FadeOut(title), FadeOut(budget_line), run_time=fr(4))
        full = ov.goal_meter(self.G, ov.GoalSpec(cpu_tick=True), n)
        base = ov.goal_meter(self.G, ov.GoalSpec(cpu_tick=False), n)
        tick = VGroup(*full.submobjects[len(base.submobjects):])
        self.until(cue("i1_compare", "thirty-three"))
        self.play(FadeIn(base, shift=DOWN * 0.15), run_time=fr(7))

        # ================= the CPU stopwatch, real time, to the @njit time =================
        t_stop = cue("i1_compare", "seconds")
        t_go = max(self.elapsed() + 0.1, t_stop - njit / 1000)

        def watch(m, dt, scene=self, card=bcard, t_go=t_go):
            ms = min(max((scene.elapsed() - t_go) * 1000.0, 0.0), njit)
            col = RED_PX if ms > budget else INK
            card.values[0].set(ms_text(ms), col)
            card.set_bar(0, ms, col)

        # the clock appears at zero just before it starts
        self.until(t_go - fr(2))
        bcard.add(cpu_val)
        self.play(FadeIn(cpu_val), run_time=fr(2))
        cpu_val.add_updater(watch)
        # the moment it passes the budget: the budget label flashes red
        self.until(t_go + budget / 1000 + fr(1))
        self.play(Indicate(bcard.budget[-1], color=RED_PX, scale_factor=1.2), run_time=fr(6))
        self.until(t_stop + 0.15)
        cpu_val.clear_updaters()
        watch(None, 0)
        # the stopwatch's result lands on the meter as the grey CPU tick
        self.play(FadeIn(tick), run_time=fr(6))

        # ================= "Where does the GPU win?" =================
        self.until(cue("i1_compare", "GPU"))
        # each pulses around its own centre, so both stay inside the card
        self.play(Indicate(bcard.names[1], color=TEAL, scale_factor=1.15),
                  Indicate(bcard.values[1], color=TEAL, scale_factor=1.3), run_time=fr(9))

        # ================= "Two places": clear the stage, then SIZE and COUNT =================
        self.until(cue("i1_compare", "Two"))
        keep = {id(base), id(tick)}
        gone = [m for m in self.mobjects if id(m) not in keep]
        self.play(*[FadeOut(m) for m in gone], run_time=0.5)
        for m in gone:
            self.remove(m)
        strip = ov.score_strip(self.G, (None, "?", "?"))
        pill_size, pill_count, head = strip.submobjects
        self.until(cue("i1_compare", "size"))
        self.play(FadeIn(head), FadeIn(pill_size, shift=DOWN * 0.1), run_time=fr(6))
        self.until(cue("i1_compare", "count"))
        self.play(FadeIn(pill_count, shift=DOWN * 0.1), run_time=fr(6))

        # ================= the seam: exactly chain_state(2) =================
        self.wait(1 / config.frame_rate + EPS, frozen_frame=True)
        if os.environ.get("CHAIN_DEBUG"):
            print(f"[intro] beat done at {self.elapsed():.2f} s, slot {self.target_seconds() + 0.4:.2f} s")
        self.write_snap_time()
        for m in list(self.mobjects):
            self.remove(m)
        self.add(*chain_state(2).mobs())
        self.finish(tail=0.4, fade=0)
