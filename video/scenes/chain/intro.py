"""The chain cut's intro clip: beats i0_problem and i1_compare in one clip (v5).

i0: four panels of the drone footage (`scenes/s_intro.py`'s grid): the
camera, the filter, then at "Count them all, every frame" the GPU
labelling every frame (ch08's first version, `assets/drones_labels`; a
blob counter under it), and at "thirty-three" the chain's goal meter
("budget 33 ms"), which stays for the whole cut. The footage starts at
frame START, so its one loop (frame 239 to 0) lands in the quiet between
the meter and the drone card.
i1: on a drone frame the compiled CPU code keeps up: a measured card
with the CPU's bar (labels only, `drone_cpu_ms`) against the 33 ms line
and the GPU as "?", as on the 81 Mpx card (v5: no kernel-time GPU win
on small frames before r02's "the CPU still wins"). "Scale up" freezes
and dims the drone panels; at "eighty-one" the card swaps to the 81 Mpx
test frame (transposed, as every frame floor from r07 on); "same
thirty-three" pulses the meter and the card's line together; a CPU
stopwatch runs in real time past 33 ms to the CPU's labels-only time
(`njit_ms`, 225 ms) beside "GPU ?", stopping at "second", where the
card's budget label pulses red (over budget); then its end flies onto
the meter as the grey CPU tick. "Two places" clears the stage
and brings the scoreboard strip's two open pills; the clip ends on
exactly `chain_state(2)` (goal meter + strip), the first picture of
r00_zoom, after a BREATH. The assembly places the i1 wav at len(i0) +
0.4 s.

Text: no title and no sub-labels; one short label per panel; the drone
card keeps its honesty note and the 81 Mpx card says it is a test frame.
Every number comes from `scenes.chain.numbers` (or the assets).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (DOWN, DR, LEFT, RIGHT, UP, UR, DashedLine, FadeIn, FadeOut, GrowFromEdge, Indicate, Line,
                   Rectangle, ReplacementTransform, RoundedRectangle, VGroup, config)

from narration.common import read_words
from scenes.chain import geo
from scenes.chain import numbers as nb
from scenes.chain import overlays as ov
from scenes.chain.base import BREATH
from scenes.chain.state import chain_state, nums, picture
from scenes.panes.geometry import Box
from scenes.panes.middle_strip import load_image
from scenes.s_intro import CAP_FONT, FPS_CAMERA, PH, PW, BlobCounter
from scenes.s_intro import CENTRES as CENTRES_V
from scenes.stage import EPS, Replay, fr
from scenes.style import (ASSETS, BG, GREY, GRID, INK, INK_SOFT, OUT, RED_PX, TEAL, BeatScene, beat_seconds,
                          is_vertical, voice)

GAP = 0.4               # silence between the two beats inside this clip (assembly)
DIM_TO = 0.4            # the drone panels' brightness after "Scale up"
START = 18              # the footage's first frame: its loop then falls at about 8.5 s, not at "On"
VALUE_SIZE = 16         # the cards' values (the names stay at 13)
CAMERA, FILTER, COUNT, CARD = 0, 1, 2, 3     # panel slots, in reading order

if is_vertical():
    # s_intro's 2 x 2 grid fits under the goal meter as it is
    CENTRES = CENTRES_V
else:
    # s_intro's 1 x 4 row, lowered so its top clears the goal meter (top right)
    # with a visible gap
    CENTRES = tuple((x, 0.0) for x in (-5.1, -1.7, 1.7, 5.1))


def ms_text(ms: float) -> str:
    return ov.fmt_ms(ms)


class Counter(BlobCounter):
    """s_intro's blob counter with its line cut to the number."""

    def fmt(self, i: int) -> str:
        return f"{int(self.counts[i]):,} blobs"


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
        return ov.lbl(text, self.size, color, mono=True).move_to(self.anchor, aligned_edge=RIGHT)

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
            ov.lbl(f"{budget_ms:.0f} ms", 12, TEAL, mono=True))
        lab = self.budget[-1]
        lab.move_to([xb, y_rows[-1] - 0.46, 0])
        # keep the label inside the card (the line may sit near an edge)
        if lab.get_left()[0] < self.x0:
            lab.shift(RIGHT * (self.x0 - lab.get_left()[0]))
        if lab.get_right()[0] > self.x1:
            lab.shift(LEFT * (lab.get_right()[0] - self.x1))
        self.names, self.values, self.bars = VGroup(), [], []
        for (name, ms), y in zip(rows, y_rows):
            nm = ov.lbl(name, 13, INK_SOFT).move_to([self.x0, y, 0], aligned_edge=LEFT)
            val = Readout("?" if ms is None else ms_text(ms), VALUE_SIZE, INK_SOFT if ms is None else INK,
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
        # v3: like every chain beat, the clip ends on a breath of silence
        return beat_seconds("i0_problem", 8.0) + GAP + beat_seconds("i1_compare", 12.5) + BREATH

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

    def caption(self, text: str, col: int, sub: str | None = None, color=INK_SOFT) -> VGroup:
        """A panel's label under it, and a small honesty note under that."""
        x, y = CENTRES[col][0], CENTRES[col][1] - PH / 2 - 0.2
        g = VGroup(ov.lbl(text, CAP_FONT, color).move_to([x, y, 0]))
        if sub:
            g.add(ov.lbl(sub, CAP_FONT - 3, INK_SOFT).move_to([x, y - 0.24, 0]))
        for m in g:
            if m.width > PW:
                m.scale_to_fit_width(PW)
        return g

    def dimmer(self, col: int) -> Rectangle:
        """A BG veil over a panel and its label: what shows through is DIM_TO bright."""
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

        # ================= i0: camera, filter =================
        sky = self.panel("drones_sky", CAMERA)
        sky.show(START)                     # fades in on the frame it starts from
        box0 = self.frame_box(CAMERA)
        cap0 = self.caption(f"camera, {FPS_CAMERA} fps", CAMERA)
        self.play(FadeIn(box0), FadeIn(sky), FadeIn(cap0), run_time=fr(9))
        sky.start(FPS_CAMERA)
        sky.t = START / FPS_CAMERA          # the panels synced to it and the counter follow
        sky.show(START)

        self.until(cue("i0_problem", "filter"))
        mask = self.panel("drones_mask", FILTER)
        mask.sync_to(sky)
        box1 = self.frame_box(FILTER)
        cap1 = self.caption("filter", FILTER)
        self.play(FadeIn(box1), FadeIn(mask), FadeIn(cap1), run_time=fr(9))
        mask.t = sky.t          # a play suspends the panel's clock: back in step with the camera

        # ================= "Count them all, every frame": the GPU labels them =================
        self.until(cue("i0_problem", "Count"))
        gpu = self.panel("drones_labels", COUNT)
        gpu.sync_to(sky)
        box2 = self.frame_box(COUNT)
        cap2 = self.caption("count", COUNT, color=TEAL)
        meta = json.loads((ASSETS / "drones_labels" / "meta.json").read_text(encoding="utf-8"))
        counter = Counter(gpu, meta["blobs_per_frame"],
                          self.centre(COUNT) + np.array([0, -PH / 2 + 0.62, 0]), PW - 0.16)
        # hold the counter still while it fades in (s_intro: FadeIn pairs glyphs
        # once); it comes with its panel, one text change
        counter.suspend_updating()
        self.play(FadeIn(box2), FadeIn(gpu), FadeIn(cap2), FadeIn(counter), run_time=fr(9))
        gpu.t = sky.t
        counter.resume_updating()

        # ================= "thirty-three milliseconds": the goal meter =================
        # the chain's meter from here on; its CPU tick comes from the stopwatch
        self.until(cue("i0_problem", "thirty-three"))
        meter = ov.goal_meter(self.G, ov.GoalSpec(cpu_tick=False), n)
        full = ov.goal_meter(self.G, ov.GoalSpec(cpu_tick=True), n)
        tick = VGroup(*full.submobjects[len(meter.submobjects):])      # the grey line, "CPU"
        self.play(FadeIn(meter, shift=DOWN * 0.15), run_time=fr(7))

        # ================= i1: the drone frame card =================
        self.until(cue("i1_compare", "On"))
        drone_img = load_image(ASSETS / "drones_mask" / "frame_000.png")
        drone_img.scale(min(PW / drone_img.width, PH / drone_img.height)).move_to(self.centre(CARD))
        box3 = self.frame_box(CARD)
        fh, fw = gpu.frame(0).shape[:2]
        cap3 = self.caption(f"{fh * fw / 1e6:.2f} Mpx", CARD, sub=f"{int(n['drone_frames'])} frames, one session")
        # v5: the CPU (labels only) against the budget, the GPU a "?" as on
        # the 81 Mpx card; no GPU drone time is shown anywhere in the cut
        dcard = MsCard(self.card_centre(CARD, 2), PW - 0.12, budget * 1.2,
                       [("CPU", n["drone_cpu_ms"]), ("GPU", None)], budget)
        cpu_bar, cpu_ms = dcard.bars[0], dcard.values[0]
        dcard.remove(cpu_bar, cpu_ms)              # the CPU's bar and value arrive at "CPU"
        self.play(FadeIn(box3), FadeIn(drone_img), FadeIn(cap3), FadeIn(dcard), run_time=fr(9))

        self.until(cue("i1_compare", "CPU"))
        cpu_bar.set_fill(GREY, 1)
        dcard.add(cpu_bar, cpu_ms)
        self.play(GrowFromEdge(cpu_bar, LEFT), FadeIn(cpu_ms), run_time=fr(8))

        # ================= "Scale up": the drone panels freeze and dim =================
        self.until(cue("i1_compare", "Scale"))
        for r in (sky, mask, gpu):
            r.stop()
        counter.clear_updaters()
        dims = [self.dimmer(c) for c in (CAMERA, FILTER, COUNT)]
        self.play(FadeOut(drone_img), FadeOut(cap3), FadeOut(dcard), *[FadeIn(m) for m in dims],
                  run_time=fr(6))

        # ================= "eighty-one megapixels": the test frame card =================
        self.until(cue("i1_compare", "eighty-one"))
        # the 900 px frame transposed (state.picture): the orientation of
        # every frame floor from r07 on
        red = load_image(picture("red"))
        red.scale_to_fit_width(PW)
        red.move_to(self.centre(CARD) + np.array([0, PH / 2 - PW / 2, 0]))
        cap3b = self.caption(f"{n['n_pixels'] / 1e6:.0f} Mpx", CARD, sub=f"test frame, {int(n['n_blobs']):,} blobs")
        njit = n["njit_ms"]
        bcard = MsCard(self.card_centre(CARD, 2), PW - 0.12, njit * 1.1,
                       [("CPU", 0.0), ("GPU", None)], budget)
        # the CPU value stays hidden until its stopwatch starts (a static
        # "0.00 ms" would read like a measured result)
        cpu_val = bcard.values[0]
        bcard.remove(cpu_val)
        self.play(FadeIn(red), FadeIn(cap3b), FadeIn(bcard), run_time=fr(7))

        # ================= "same thirty-three": one budget, on the meter and the card =================
        self.until(cue("i1_compare", "thirty-three"))
        self.play(Indicate(meter[1], color=TEAL, scale_factor=1.12),
                  Indicate(bcard.budget[-1], color=TEAL, scale_factor=1.3), run_time=fr(9))

        # ================= the CPU stopwatch, real time, to the CPU's labels-only time =================
        # it stops as "second" starts ("over a fifth of a second": 225 ms)
        t_stop = cue("i1_compare", "second")
        t_go = max(self.elapsed() + 0.1, t_stop - njit / 1000)

        def watch(m, dt, scene=self, card=bcard, t_go=t_go):
            ms = min(max((scene.elapsed() - t_go) * 1000.0, 0.0), njit)
            col = RED_PX if ms > budget else INK
            card.values[0].set(ms_text(ms), col)
            card.set_bar(0, ms, col)

        # the clock appears at zero just before it starts; it turns red as
        # it passes 33 ms
        self.until(t_go - fr(2))
        bcard.add(cpu_val)
        self.play(FadeIn(cpu_val), run_time=fr(2))
        cpu_val.add_updater(watch)
        # it stops at "second": the budget label pulses red, over budget
        self.until(t_stop)
        cpu_val.clear_updaters()
        # the last reading is the measured time, even when the whole-frame
        # wait lands a little before the cue
        end_col = RED_PX if njit > budget else INK
        bcard.values[0].set(ms_text(njit), end_col)
        bcard.set_bar(0, njit, end_col)
        self.play(Indicate(bcard.budget[-1], color=RED_PX, scale_factor=1.2), run_time=fr(6))
        # the bar's end flies up onto the meter and becomes the grey CPU tick
        bar = bcard.bars[0]
        mark = Line(bar.get_corner(UR) + UP * 0.05, bar.get_corner(DR) + DOWN * 0.05,
                    stroke_color=RED_PX, stroke_width=7)
        mark.set_z_index(ov.Z_OVER)
        self.play(ReplacementTransform(mark, tick[0]), run_time=fr(10))
        self.play(FadeIn(tick[1]), run_time=fr(4))

        # ================= "Where does the GPU win?" =================
        self.until(cue("i1_compare", "GPU"))
        # each pulses around its own centre, so both stay inside the card
        self.play(Indicate(bcard.names[1], color=TEAL, scale_factor=1.15),
                  Indicate(bcard.values[1], color=TEAL, scale_factor=1.3), run_time=fr(9))

        # ================= "Two places": clear the stage, then the two open pills =================
        self.until(cue("i1_compare", "Two"))
        keep = {id(meter), id(tick[0]), id(tick[1])}
        gone = [m for m in self.mobjects if id(m) not in keep]
        self.play(*[FadeOut(m) for m in gone], run_time=fr(6))
        for m in gone:
            self.remove(m)
        # both pills come in together (one text change); each pulses on its word
        pills = ov.score_strip(self.G, (None, "?", "?")).submobjects
        self.play(*[FadeIn(p, shift=DOWN * 0.1) for p in pills], run_time=fr(6))
        for p, word in zip(pills, ("size", "count")):
            self.until(cue("i1_compare", word))
            self.play(Indicate(p[1], color=INK, scale_factor=1.15), run_time=fr(8))

        # ================= the seam: exactly chain_state(2) =================
        self.wait(1 / config.frame_rate + EPS, frozen_frame=True)
        if os.environ.get("CHAIN_DEBUG"):
            print(f"[intro] beat done at {self.elapsed():.2f} s, slot {self.target_seconds() + 0.4:.2f} s")
        self.write_snap_time()
        for m in list(self.mobjects):
            self.remove(m)
        self.add(*chain_state(2).mobs())
        self.finish(tail=0.4, fade=0)
