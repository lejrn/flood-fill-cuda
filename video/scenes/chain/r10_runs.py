"""r10: stop moving pixels. One row of the many-blob toy: its red pixels
come in spans (runs). Every row becomes runs, 25x fewer items on the real
frame. Touching runs are linked (the smaller label wins), each run is
painted in one stroke. The queue leaves VRAM, the gate (the ring loop) and
the launch counter go; six plain kernels slide into the gap between the
boards, with 5 seams where the gate was, and light one by one while the
real frame turns teal. About three milliseconds: the needle crosses the
budget, then the scoreboard card opens on the count row.

Starts on chain_state(12), ends on chain_state(13).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, UP, FadeIn, FadeOut, SurroundingRectangle, config

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain import toy
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import (BLOB_HUES, TEAL, Z_OVER, boundary, carry_caption, label_colors, mix, picture,
                                score_rows, teal_picture, toys)
from scenes.style import GOLD, GREY

ROW = 12        # a middle row of the toy with three runs (the U's two arms and a blob)
# the linked groups' colours, less any that reads as amber (amber means waiting)
HUES = tuple(h for h in BLOB_HUES if np.abs(fl.rgb(h).astype(int) - fl.rgb(GOLD).astype(int)).max() > 48)


class Runs(ChainScene):
    k = 12

    def show(self, raster: np.ndarray) -> None:
        """Put one raster on the floor now (on the shared clock)."""
        self.floor_play(lambda p: 0, lambda _k: raster, fr(1))

    def say_last(self, at: float | None = None, frames: int = 5, extra=()) -> None:
        """base.say_last, with `extra` played alongside the caption's fade-in
        (the needle and the act tag land with the number they show)."""
        self.say(*carry_caption(self.k + 1), at=at, frames=frames, extra=extra)

    def relight(self, spec, frames: int = 2) -> None:
        """machine_to for a part that changes in place (a kernel lights): the
        new part fades in over the old one, then the old one goes, so the
        unchanged kernels never thin out mid-fade."""
        before = dict(self.machine.parts)
        anims = [a for a in self.machine_anims(spec, slide=0) if not isinstance(a, FadeOut)]
        self.play(*anims, run_time=fr(frames))
        self.remove(*[o for name, o in before.items() if self.machine.parts.get(name) is not o])

    def over_in(self, **news) -> tuple[list, list]:
        """over_anims, the same way: FadeIns of the new overlays drawn over the
        old ones, and the old ones to remove after the play, so the unchanged
        parts (the meter's track, the strip's filled pills) never thin out."""
        anims, old = [], []
        for name, new in news.items():
            if self.over.get(name) is not None:
                old.append(self.over[name])
            anims.append(FadeIn(new))
            self.over[name] = new
        return anims, old

    def card(self, rows: tuple, h: int):
        """The full scoreboard, its footnote naming each row's sessions, row h
        framed (teal: a GPU win, grey: a CPU win)."""
        foot = "; ".join(f"row {i + 1}: {r.tag}" for i, r in enumerate(rows))
        c = ov.score_card(self.G, rows, highlight=h, footnote=foot)
        line = c[1][1][1 + h]                      # box, body -> (title, lines, foot); lines[0] is the head
        col = TEAL if rows[h].winner == "GPU" else GREY
        frame = SurroundingRectangle(line, color=col, buff=0.09, corner_radius=0.08, stroke_width=2.5)
        frame.set_z_index(Z_OVER + 21)
        c.add(frame)
        # where r11 opens its card, so it keeps its place across the cut: 9:16
        # over the floor, 16:9 left of the boards (the six kernels stay in view)
        G, w = self.G, config.frame_width
        if G.vertical:
            x0, x1 = -w / 2 + 0.15, w / 2 - 0.15
        else:
            x0, x1 = -w / 2 + 0.2, self.machine.part("boards").get_left()[0] - 0.15
        c.scale(min(1.0, (x1 - x0) / c.width))
        c.set_x(min(max(G.floor_c[0], x0 + c.width / 2), x1 - c.width / 2))
        # and clear of what it would cut through: the inset and the act tag
        # above it (9:16: the inset's line sits on the card's top band), the caption below
        hi, lo = np.inf, self.cap.get_top()[1] + 0.15 if self.cap is not None else -np.inf
        for name in ("inset", "tag"):
            m = self.over.get(name)
            if m is not None and m.get_right()[0] > c.get_left()[0] and m.get_left()[0] < c.get_right()[0]:
                hi = min(hi, m.get_bottom()[1] - 0.12)
        if c.height > hi - lo:
            c.scale((hi - lo) / c.height)
        c.shift(DOWN * max(0.0, c.get_top()[1] - hi) + UP * max(0.0, lo - c.get_bottom()[1]))
        return c

    def beat_anim(self) -> None:
        t = toys()
        d = self.der
        many = t["many"]
        rs = toy.runs(many)
        boxes_of = lambda runs: [(x.r, x.c0, x.c1) for x in runs]
        red = label_colors(many, np.full(many.shape, -1))
        empty = fl.rgb(fl.EMPTY)
        teal = fl.rgb(TEAL)
        row_runs = [x for x in rs if x.r == ROW]
        end = boundary(self.k + 1)

        # "row": the toy comes back red, every row dimmed but one, outlined
        dim = red.copy()
        off = np.ones(many.shape[0], bool)
        off[ROW] = False
        dim[off] = mix(red[off], np.broadcast_to(empty, red[off].shape), 0.7)
        line = np.zeros(many.shape, bool)
        line[ROW] = many[ROW]
        self.until(self.cue("row"))
        self.floor_blend(self.pic(picture("labels")), self.toy(dim, ring=line), frames=6)

        # "spans": that row's red cells become teal spans
        col = dim.copy()
        for x in row_runs:
            col[x.r, x.c0:x.c1 + 1] = teal
        self.until(self.cue("spans"))
        self.show(self.toy(col, boxes=boxes_of(row_runs)))

        # then every row, top to bottom; the count is the real frame's
        allc = red.copy()
        for x in rs:
            allc[x.r, x.c0:x.c1 + 1] = teal

        def rows_done(r: int):
            c = dim.copy()                # each row lifts out of the dim as it turns teal
            c[:r + 1] = allc[:r + 1]
            c[ROW] = allc[ROW]
            return self.toy(c, boxes=boxes_of([x for x in rs if x.r <= r or x.r == ROW]))

        self.until(self.cue("spans") + fr(5))
        self.floor_play(lambda p: round(p * (many.shape[0] - 1)), rows_done, fr(8))
        self.say(f"{d['runs_x']:.0f}× fewer", "on the real frame", at=self.cue("spans", end=True))

        # "link touching spans": rows top to bottom, each run takes its group's
        # colour; where two groups touch, the smaller label wins (the U's right arm)
        parent = list(range(len(rs)))

        def find(a: int) -> int:
            while parent[a] != a:
                a = parent[a]
            return a

        _labels, pairs = toy.run_labels(many)
        steps = []                       # per row: the root of every run so far
        hue_of: dict = {}
        for r in range(many.shape[0]):
            for i, j in pairs:
                if rs[j].r == r:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            roots = {}
            for i, x in enumerate(rs):
                if x.r <= r:
                    root = find(i)
                    hue_of.setdefault(root, fl.rgb(HUES[len(hue_of) % len(HUES)]))
                    roots[i] = root
            steps.append(roots)

        def linked(r: int, paint: bool = False):
            c = allc.copy()
            for i, root in steps[r].items():
                x = rs[i]
                c[x.r, x.c0:x.c1 + 1] = hue_of[root]
            return self.toy(c, boxes=() if paint else boxes_of(rs))

        last = many.shape[0] - 1
        t0 = self.cue("touching")
        self.until(t0)
        self.floor_play(lambda p: round(p * last), linked, max(self.cue("pixels", after=t0) - t0, 0.6))
        # "not pixels": each run painted in one stroke
        painted = linked(last, paint=True)
        self.until(self.cue("pixels", after=t0, end=True) - fr(3))
        self.show(painted)

        # "No queue, no ring loop": the queue leaves VRAM, then the gate and the
        # launch counter go with the caption (one text change, not two 0.4 s
        # apart); no caption of their own, the parts say it
        m = self.machine.spec
        no_queue = replace(m, queue=False, tokens=())
        no_loop = replace(no_queue, gate=None, launches=None)
        self.until(self.cue("queue"))
        self.machine_to(no_queue, frames=5)
        self.until(self.cue("ring"))
        self.unsay(frames=5, extra=self.machine_anims(no_loop))

        # "six plain steps": the six kernels slide into the gap between the boards
        # (5 seams where the gate was), the first one lit, while the floor goes
        # back to the real frame's red pixels; each kernel then lights in turn
        # and the frame turns teal, a sixth at a time
        kern = replace(no_loop, kernels=True, seams=True)
        red_pic = self.pic(picture("red"))
        teal_pic = self.pic(teal_picture())
        tint = [red_pic if j == 0 else mix(red_pic, teal_pic, j / 5) for j in range(5)] + [teal_pic]
        self.until(self.cue("six"))
        self.floor_blend(painted, red_pic, frames=6)
        self.machine_to(replace(kern, kernel_lit=0), frames=6)
        t_about = self.cue("About")
        t1 = self.elapsed()
        step = min(max((t_about - fr(3) - t1) / 5, fr(3)), 0.4)
        for j in range(1, 6):
            self.until(t1 + (j - 1) * step)
            self.floor_blend(tint[j - 1], tint[j], frames=2)
            self.relight(replace(kern, kernel_lit=j), frames=2)
        self.until(t1 + 5 * step)
        self.relight(kern, frames=2)

        # "About three milliseconds": the needle swings past the budget and turns
        # teal, COUNT is won, the number lands as the last caption
        anims, old = self.over_in(goal=self.goal(end.goal), tag=ov.act_tag(self.G, *end.tag))
        self.say_last(at=t_about, frames=6, extra=anims)
        self.remove(*old)

        # then the scoreboard card opens on the count row (its footnote names the
        # sessions) and the strip's last pill fills; the card stays to the end of
        # the hold and closes here, not in base (whose fade's last step is never
        # drawn: the frame before the snap would keep a trace of it)
        rows = score_rows()
        card = self.card(rows, 2)
        anims, old = self.over_in(strip=self.strip(end.rows))
        self.until(t_about + MIN_DWELL)
        self.play(*anims, FadeIn(card, scale=0.96), run_time=fr(6))
        self.remove(*old)
        self.until(self.target_seconds() + 0.4 - fr(3) - fr(3))
        self.play(FadeOut(card), run_time=fr(3))
        assert self.machine.spec == end.machine, (self.machine.spec, end.machine)
