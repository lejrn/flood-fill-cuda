"""r10: stop moving pixels. One row of the many-blob toy is marked on the
floor; in memory it is one unbroken line: the VRAM close-up (D2, form
"runs") grows out of the VRAM chips with that row as its image lane, red
spans and white gaps. Each span becomes one entry of the run table (five
int32 sub-cells): 25x fewer items on the real frame. Then a warp reads
the row: 32 neighbouring threads, 32 neighbouring pixels in one 96 B
packet, every byte lit (coalesced by design, never profiled), and the
spans it finds go down into the runs (ch06's pack and emit as one read
and one write). The panel folds back; the queue leaves VRAM, the gate
(the ring loop) and the launch counter go; six plain kernels slide into
the gap between the boards, with 5 seams where the gate was, and light
one by one while the real frame turns teal. About three milliseconds:
the needle crosses the budget, then the strip's last pill fills (r11
opens the full card).

Starts on chain_state(14), ends on chain_state(15).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (Create, FadeIn, FadeOut, FadeTransform, Group, LaggedStart, Line, Rectangle, ReplacementTransform,
                   VGroup)

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain import toy
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import TEAL, boundary, label_colors, mix, picture, teal_picture, toys
from scenes.style import INK, INK_SOFT

ROW = 12            # a middle row of the toy with three spans (the U's two arms and a blob)
LANE_PX = 40        # the panel's image lane: the row stretched to the pixels that fit (120 B across)
WARP = 32           # threads in a warp: 32 pixels, 96 B of the row in one packet
PANEL_H = 2.9       # the runs form's two lanes fill this; G.vram's full height would leave a third empty


def lane_row(cells, n: int = LANE_PX) -> list:
    """A toy row stretched to n memory pixels (nearest): the same spans, in order."""
    w = len(cells)
    return [bool(cells[i * w // n]) for i in range(n)]


class Runs(ChainScene):
    k = 14

    def show(self, raster: np.ndarray) -> None:
        """Put one raster on the floor now (on the shared clock)."""
        self.floor_play(lambda p: 0, lambda _k: raster, fr(1))

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

    def panel(self, row: list) -> mc.VramPanel:
        """The runs close-up at G.vram's centre and width, as tall as its lanes need."""
        (cx, cy), (w, h) = self.G.vram
        return mc.vram_closeup(self.Lay.machine_pl, (cx, cy), min(h, PANEL_H), form="runs", width=w, row=row)

    def beat_anim(self) -> None:
        t = toys()
        d = self.der
        many = t["many"]
        rs = toy.runs(many)
        row_runs = [x for x in rs if x.r == ROW]
        red = label_colors(many, np.full(many.shape, -1))
        teal = fl.rgb(TEAL)
        end = boundary(self.k + 1)
        pl = self.Lay.machine_pl
        home = mc.vram_point(pl)

        # "Look at one row": the toy comes back red, every row dimmed but one, outlined
        dim = red.copy()
        off = np.ones(many.shape[0], bool)
        off[ROW] = False
        dim[off] = mix(red[off], np.broadcast_to(fl.rgb(fl.EMPTY), red[off].shape), 0.7)
        line = np.zeros(many.shape, bool)
        line[ROW] = many[ROW]
        self.until(self.cue("Look"))
        self.floor_blend(self.pic(picture("labels")), self.toy(dim, ring=line), frames=6)

        # "in memory": the close-up grows out of the VRAM chips, its image lane
        # that row; the runs lane and the "1 warp" label wait for their words, and
        # the warp strip sits dim on the panel's SM edge (as in r08) until "Warps".
        # The panel opens over the floor, so the marked row gets all of "Look at
        # one row" first (about 0.9 s), and memory comes on "in memory"
        row = lane_row(many[ROW])
        p = self.panel(row)
        img, runs = p.lanes["image"], p.lanes["runs"]
        run_name = p.body[0][1]
        p.body[0].remove(run_name)
        p.body.remove(runs.cells, p.warp, p.warp_label)
        ghost = p.warp.copy().set_fill(mc.mix_hex(mc.WARP_FILL, mc.CARD_FILL, 0.6)).set_stroke(opacity=0.3)
        p.body.add(ghost)
        p.warp.set_z_index(p.z + 1.1)
        self.until(self.cue("in"))
        self.play(FadeIn(p, target_position=home, scale=0.08), run_time=fr(8))
        z = p.z + 2

        # "unbroken line": the lane underlined end to end
        x0, _ = img.span(0)
        _, x1 = img.span(len(img.cells) - 1)
        y_u = img.y - img.h / 2 - 0.07
        under = Line([x0, y_u, 0], [x1, y_u, 0], stroke_color=INK_SOFT, stroke_width=2).set_z_index(z)
        self.until(self.cue("unbroken"))
        self.play(Create(under), run_time=fr(8))

        # "spans": each red span outlined
        spans = mc.runs_of(row)
        boxes = VGroup()
        for a, b in spans:
            u0, u1 = img.span(a, b - a + 1)
            boxes.add(Rectangle(width=u1 - u0 + 0.03, height=img.h + 0.06, stroke_color=INK,
                                stroke_width=2.2).move_to([(u0 + u1) / 2, img.y, 0]))
        boxes.set_z_index(z)
        self.until(self.cue("spans"))
        self.play(FadeOut(under), FadeIn(boxes), run_time=fr(5))

        # "link touching spans": the runs lane opens under the image, and each
        # span, filled teal, goes down into one run entry (five int32 sub-cells);
        # the image itself is never changed. Under the panel the floor's row
        # turns to its teal spans too (seen when the panel folds).
        bars = VGroup()
        for a, b in spans:
            u0, u1 = img.span(a, b - a + 1)
            bars.add(Rectangle(width=u1 - u0 - 0.012, height=img.h, fill_color=TEAL, fill_opacity=1,
                               stroke_width=0).move_to([(u0 + u1) / 2, img.y, 0]))
        bars.set_z_index(z)
        entries = list(runs.cells[:len(spans)])
        for e in entries:
            e.set_z_index(z)
        empty = VGroup()                   # the entries' places, empty until their span lands
        for i in range(len(runs.cells)):
            u0, u1 = runs.span(i)
            empty.add(Rectangle(width=u1 - u0 - 0.012, height=runs.h, fill_color=mc.SLOT, fill_opacity=1,
                                stroke_width=0).move_to([(u0 + u1) / 2, runs.y, 0]))
        empty.set_z_index(p.z + 1)
        col = dim.copy()
        for x in row_runs:
            col[x.r, x.c0:x.c1 + 1] = teal
        self.until(self.cue("link"))
        self.show(self.toy(col, boxes=[(x.r, x.c0, x.c1) for x in row_runs]))
        self.play(FadeIn(run_name), FadeIn(empty), FadeIn(bars), run_time=fr(4))
        self.play(*[FadeTransform(bar, e) for bar, e in zip(bars, entries)], FadeOut(boxes),
                  run_time=fr(8))

        # "twenty-five times fewer": the count is the real frame's
        self.say(f"{d['runs_x']:.0f}× fewer", "runs than red pixels", at=self.cue("fewer"))

        # "Warps read the image": the warp strip lights, thread by thread, over
        # the 32 pixels it reads; "1 warp" at the second "row"
        t_w = self.cue("Warps")
        self.until(t_w)
        self.play(LaggedStart(*[FadeIn(c) for c in p.warp], lag_ratio=0.05), run_time=fr(6))
        self.until(self.cue("row", n=2, after=t_w))
        self.play(FadeIn(p.warp_label), run_time=fr(4))

        # "neighbouring threads": each thread's line down to its own pixel, side by
        # side; the lines pass behind the lane's name (in 16:9 the first one meets
        # its "e"), over a patch of the card's own fill
        top = img.y + img.h / 2
        threads = VGroup(*[Line(p.warp[k].get_bottom(), [p.warp[k].get_x(), top, 0], stroke_color=mc.WARP_EDGE,
                                stroke_width=1.2, stroke_opacity=0.7) for k in range(WARP)])
        threads.set_z_index(p.z + 1.5)
        name = p.body[0][0]
        patch = Rectangle(width=name.width + 0.04, height=name.height + 0.03, fill_color=mc.CARD_FILL,
                          fill_opacity=1, stroke_width=0).move_to(name).set_z_index(p.z + 1.6)
        name.set_z_index(p.z + 1.7)
        t_n = self.cue("neighbouring")
        self.until(t_n)
        self.add(patch)
        self.play(LaggedStart(*[Create(m) for m in threads], lag_ratio=0.08), run_time=fr(6))

        # "neighbouring bytes": the 32 pixels light as one packet (96 B, every
        # byte lit), which rises to the warp: one load. The caption lands with it.
        load = p.packet("image", p.warp_i0, WARP, INK)
        self.until(self.cue("neighbouring", after=t_n + 0.1))
        self.play(FadeIn(load), run_time=fr(3))
        self.say("coalesced", "by design, not profiled", at=self.cue("bytes"), frames=6,
                 extra=[load.animate.move_to([load.get_x(), p.warp_y, 0]), FadeOut(threads)])
        self.remove(patch)                 # the card's own fill: nothing shows when it goes

        # the spans it found go down into the runs lane: one store, 3 entries x 20 B
        store = p.packet("runs", 0, len(spans), TEAL)
        self.play(ReplacementTransform(load, store), run_time=fr(7))
        self.play(FadeOut(store), run_time=fr(3))

        # the panel folds back into the VRAM chips, just before "No queue"
        fold = Group(p, run_name, empty, *entries, p.warp, p.warp_label)
        self.until(max(self.cue("No") - fr(7), self.elapsed()))
        self.play(FadeOut(fold, target_position=home, scale=0.08), run_time=fr(7))

        # "No queue, no ring loop": the queue leaves VRAM, then the gate and the
        # launch counter go; the caption stays (one text change fewer)
        m = self.machine.spec
        no_queue = replace(m, queue=False, tokens=())
        no_loop = replace(no_queue, gate=None, launches=None)
        self.until(self.cue("queue"))
        self.machine_to(no_queue, frames=5)
        self.until(self.cue("ring"))
        self.machine_to(no_loop, frames=5)

        # "six plain steps": the six kernels slide into the gap between the boards
        # (5 seams where the gate was), the first one lit, while the floor goes
        # back to the real frame's red pixels; each kernel then lights in turn
        # and the frame turns teal, a sixth at a time
        kern = replace(no_loop, kernels=True, seams=True)
        marked = self.toy(col, boxes=[(x.r, x.c0, x.c1) for x in row_runs])
        red_pic = self.pic(picture("red"))
        teal_pic = self.pic(teal_picture())
        tint = [red_pic if j == 0 else mix(red_pic, teal_pic, j / 5) for j in range(5)] + [teal_pic]
        self.until(self.cue("six"))
        self.floor_blend(marked, red_pic, frames=6)
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
        # teal, COUNT is won and the strip's last pill fills (GPU 35x), all with
        # the last caption: one change, so r11's card (which opens on its first
        # words, with the sessions behind each row) comes a full dwell later
        anims, old = self.over_in(goal=self.goal(end.goal), tag=ov.act_tag(self.G, *end.tag),
                                  strip=self.strip(end.rows))
        self.say_last(at=t_about, frames=6, extra=anims)
        self.remove(*old)
        assert self.machine.spec == end.machine, (self.machine.spec, end.machine)
