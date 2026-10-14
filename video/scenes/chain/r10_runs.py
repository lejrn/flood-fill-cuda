"""r10: stop moving pixels. One row of the many-blob toy: its red pixels
come in spans (runs). Every row becomes runs, 25x fewer items on the real
picture. Touching runs are linked (the smaller label wins), each run is
painted in one stroke. The queue, the gate (the ring loop) and the launch
counter come off; six plain kernels go on, with 5 seams where the gate
was. About three milliseconds: the needle crosses the budget, and the
scoreboard's count row fills.

Starts on chain_state(12), ends on chain_state(13).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, LEFT, FadeIn, FadeOut, RoundedRectangle, SurroundingRectangle, VectorizedPoint, VGroup, config

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain import toy
from scenes.chain.base import ChainScene, fr
from scenes.style import GREY, GRID, INK, INK_SOFT, label
from scenes.chain.state import BLOB_HUES, TEAL, boundary, label_colors, mix, picture, score_rows, teal_picture, toys

ROW = 12        # a middle row of the toy with three runs (the U's two arms and a blob)


def _score_card(G, rows: tuple, highlight: int | None = None, footnote: str | None = None) -> VGroup:
    """ov.score_card as r11 draws it: the head's empty first cell is a point
    (an empty Text has no points, so Manim cannot place it), a footnote line
    names each row's sessions, and the highlighted row gets a frame (teal: a
    GPU win, grey: a CPU win)."""
    lines = VGroup()
    lines.add(VGroup(VectorizedPoint(), label("CPU", 16, INK_SOFT), label("GPU", 16, INK_SOFT),
                     label("winner", 16, INK_SOFT)))
    for i, r in enumerate(rows):
        col = TEAL if r.winner == "GPU" else GREY
        name = VGroup(label(r.what, 20, INK if i == highlight else INK_SOFT),
                      label(r.picture, 13, INK_SOFT)).arrange(DOWN, buff=0.03, aligned_edge=LEFT)
        lines.add(VGroup(name, label(ov.fmt_ms(r.cpu_ms), 20, INK, mono=True),
                         label(ov.fmt_ms(r.gpu_ms), 20, INK, mono=True),
                         label(f"{r.winner} {r.factor}", 22, col, mono=True)))
    cols = [max(row[j].width for row in lines) for j in range(4)]
    xs = np.cumsum([0] + [c + 0.45 for c in cols[:-1]])
    lines.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    for row in lines:
        for j, m in enumerate(row):
            m.set_x(xs[j] + m.width / 2)
    body = VGroup(label("CPU vs GPU", 22, INK), lines).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    if footnote:
        body.add(label(footnote, 12, INK_SOFT, mono=True))
        body.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    room = config.frame_width - 0.9
    if body.width > room:
        body.scale_to_fit_width(room)
    box = RoundedRectangle(corner_radius=0.16, width=body.width + 0.6, height=body.height + 0.5,
                           fill_color=ov.PANEL, fill_opacity=0.97, stroke_color=GRID, stroke_width=2)
    box.move_to(G.card_c)
    body.move_to(box)
    g = VGroup(box, body)
    if highlight is not None:
        col = TEAL if rows[highlight].winner == "GPU" else GREY
        g.add(SurroundingRectangle(lines[1 + highlight], color=col, buff=0.09, corner_radius=0.08, stroke_width=2.5))
    for m in g.get_family():
        m.set_z_index(ov.Z_OVER + 20)
    return g


class Runs(ChainScene):
    k = 12

    def show(self, raster: np.ndarray) -> None:
        """Put one raster on the floor now (on the shared clock)."""
        self.floor_play(lambda p: 0, lambda _k: raster, fr(1))

    def beat_anim(self) -> None:
        t = toys()
        n, d = self.num, self.der
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

        # then every row, top to bottom
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
        self.say(f"{d['runs_x']:.0f}× fewer",
                 f"{n['n_pixels'] / 1e6:.0f} Mpx frame: {int(n['red_px']):,} px -> {int(n['n_runs']):,} runs",
                 at=self.cue("spans", end=True))

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
                    hue_of.setdefault(root, fl.rgb(BLOB_HUES[len(hue_of) % len(BLOB_HUES)]))
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
        self.until(self.cue("pixels", after=t0, end=True) - fr(3))
        self.show(linked(last, paint=True))

        # "No queue, no ring loop": the parts come off, one per word
        m = self.machine.spec
        no_queue = replace(m, queue=False, tokens=())
        no_loop = replace(no_queue, gate=None, launches=None)
        self.say("no queue", at=self.cue("No"), extra=self.machine_anims(no_queue))
        self.say("no ring loop", "no gate, no launch counter", at=self.cue("ring"),
                 extra=self.machine_anims(no_loop))

        # "six plain steps": the kernel strip slides on, its 5 seams where the gate was;
        # each kernel lights once, in order
        self.until(self.cue("six"))
        kern = replace(no_loop, kernels=True, seams=True)
        self.machine_to(replace(kern, kernel_lit=0), frames=5)
        self.floor_blend(linked(last, paint=True), self.pic(teal_picture()), frames=10)
        for j in range(1, 6):
            self.machine_to(replace(kern, kernel_lit=j), frames=2, slide=0)
        self.machine_to(kern, frames=2, slide=0)

        # "About three milliseconds": the needle crosses 33 ms and turns teal;
        # the count row fills on the expanded scoreboard, then it folds into the strip
        small, huge, many_row = score_rows()
        rows = (small, huge, many_row)
        foot = "; ".join(f"row {i + 1}: {r.tag}" for i, r in enumerate(rows))
        card = _score_card(self.G, rows, highlight=2, footnote=foot)
        self.extras = [card]
        self.until(self.cue("About"))
        self.over_to(goal=self.goal(end.goal), tag=ov.act_tag(self.G, *end.tag), frames=6,
                     extra=[FadeIn(card, scale=0.96)])
        self.bring_to_front(self.over["strip"])     # the strip's header sits on the meter's edge
        self.say_last(at=self.cue("milliseconds"))
        self.until(self.speech_end() + 0.9)
        self.over_to(strip=self.strip(end.rows), frames=5, extra=[FadeOut(card)])
        self.extras = []
        assert self.machine.spec == end.machine, (self.machine.spec, end.machine)
