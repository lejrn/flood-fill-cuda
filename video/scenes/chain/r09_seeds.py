"""r09: the GPU finds its own seeds. The tag printer and its rack leave;
on the many-blob toy every red pixel with no red neighbour before it in
scan order sparks a seed, then a stride-4 lattice adds more (ch05 runs a
stride-8 lattice on the real picture). Waves start from every seed at
once; where two meet, both keep the smaller label (the min unit). One fill
launch plus two cleanups: 3 launches, 51 ms on the real picture. But the
queue still carries every red pixel.

Starts on chain_state(11), ends on chain_state(12).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, LEFT, FadeIn, FadeOut, LaggedStart, RoundedRectangle, VGroup

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain import toy
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import BLOB_HUES, Z_MACHINE, Z_OVER, boundary, label_colors, picture, toys
from scenes.style import INK_SOFT, RED_PX, label

STRIDE = 4      # the toy's lattice; the real run (split_L8) uses 8 on a 9000 x 9000 picture


class Seeds(ChainScene):
    k = 11

    def show(self, raster: np.ndarray) -> None:
        """Put one raster on the floor now (on the shared clock)."""
        self.floor_play(lambda p: 0, lambda _k: raster, fr(1))

    def fill_tokens(self, spec: mc.Spec, frames: int) -> None:
        """The queue fills: the new tokens drop in one after another; the
        ones already there stay put (no cross-fade of the whole part)."""
        new = mc.build(spec, self.Lay.machine_pl, z0=Z_MACHINE)
        part = new.parts["queue_tokens"]
        old = self.machine.parts.get("queue_tokens")
        n_old, n_new = len(self.machine.spec.tokens), len(spec.tokens)
        per = len(part.submobjects) // n_new
        fresh = [VGroup(*[f.copy() for f in part.submobjects[j * per:(j + 1) * per]]) for j in range(n_old, n_new)]
        self.play(LaggedStart(*[FadeIn(g, shift=DOWN * 0.25) for g in fresh], lag_ratio=0.35),
                  run_time=fr(frames))
        self.remove(*fresh)
        if old is not None:
            self.remove(old)
        self.add(part)
        parts = dict(self.machine.parts)
        parts["queue_tokens"] = part
        self.machine = mc.Machine(spec, {k: parts[k] for k in new.parts})

    def side_note(self, text: str, part, side: str = "launches", color=INK_SOFT) -> VGroup:
        """A small mono note next to a machine part, above the machine (a transient
        extra: construct() fades it before the snap)."""
        t = label(text, 15, color, mono=True)
        box = RoundedRectangle(corner_radius=0.06, width=t.width + 0.2, height=t.height + 0.14,
                               fill_color=ov.PANEL, fill_opacity=0.92, stroke_width=0)
        t.move_to(box)
        m = VGroup(box, t)
        if side == "queue":                 # under the queue row, clear of the chip
            m.next_to(part, DOWN, buff=0.06).align_to(part, LEFT)
        else:                               # beside the launch counter, on the chip's edge
            m.next_to(part, LEFT, buff=0.06)
        for f in m.get_family():
            f.set_z_index(Z_OVER - 1)
        return m

    def beat_anim(self) -> None:
        t = toys()
        many = t["many"]
        cand = toy.candidates(many)
        lat = toy.lattice(many, STRIDE) & ~cand
        seeds = list(zip(*np.nonzero(cand | lat)))
        mf = toy.merge_fill(many, seeds, 8)
        order = sorted({int(v) for v in np.unique(mf.frames[0][mf.frames[0] >= 0])})
        # the labels that survive get distinct hues first: two blobs never end in one colour
        last_lab = mf.resolved[-1]
        fin = sorted(int(v) for v in np.unique(last_lab[last_lab >= 0]))
        rest = [v for v in order if v not in fin]
        hue = {v: fl.rgb(BLOB_HUES[i % len(BLOB_HUES)]) for i, v in enumerate(fin + rest)}
        assert len(fin) <= len(BLOB_HUES), fin
        red = label_colors(many, np.full(many.shape, -1))
        seed_dots = (cand | lat).astype(int) * 3
        end = boundary(self.k + 1)

        # "finds": the hand-fed parts leave (printer, rack, tags); the floor
        # becomes many blobs, all red
        m10 = self.machine.spec
        plain = replace(m10, printer=False, rack=False, tag_tokens=())
        self.until(self.cue("finds"))
        self.floor_blend(self.toy(label_colors(t["two"], t["two_lab"])), self.toy(red), frames=6)
        self.unsay(frames=6, extra=self.machine_anims(plain))

        # "seeds": sparks at every top-left corner (scan order), then the lattice
        cand_order = sorted(zip(*np.nonzero(cand)))
        lat_order = sorted(zip(*np.nonzero(lat)))

        def sparks(n: int):
            d = np.zeros(many.shape, int)
            for rc in (cand_order + lat_order)[:n]:
                d[rc] = 3
            return self.toy(red, dots=d)

        nc, nl = len(cand_order), len(lat_order)
        t_seeds = self.cue("seeds")
        self.until(t_seeds)
        self.floor_play(lambda p: max(1, round(p * nc)), sparks, fr(5))
        # the lattice sparks end before the waves start at "every" (one hook per floor at a time)
        t_lat = t_seeds + fr(6)
        dur_lat = max(min(fr(4), self.cue("every") - t_lat - fr(1)), fr(2))
        self.until(t_lat)
        self.floor_play(lambda p: nc + round(p * nl), sparks, dur_lat)

        # "every blob in one fill launch": waves from every seed at once; a
        # merge recolours both waves to the smaller label at once
        nf = len(mf.resolved)

        def waves(i: int):
            r = mf.resolved[i]
            col = red.copy()
            for v in np.unique(r[r >= 0]):
                col[r == v] = hue[int(v)]
            return self.toy(col, dots=seed_dots if i < nf - 1 else None)

        t0, t1 = self.cue("every"), self.cue("fifty-one")
        self.until(max(t0, t_lat + dur_lat + fr(1)))
        self.floor_play(lambda p: min(nf - 1, int(p * nf)), waves, max(t1 - t0 - 0.25, 0.8))
        # the min unit slides into the slot the rack left
        with_min = replace(plain, min_unit=True)
        self.say("min", "where waves meet, the smaller label wins", frames=5,
                 extra=self.machine_anims(with_min))
        # one fill launch + two cleanups: 3 launches
        # (a note, not a caption: say_last stays the last caption)
        self.until(self.cue("launch"))
        self.machine_to(replace(with_min, launches="3"), frames=4, slide=0)
        note = self.side_note("1 fill + 2 cleanup", self.machine.part("launches"))
        self.extras = [note]
        self.play(FadeIn(note), run_time=fr(3))

        # "fifty-one milliseconds": the real picture, every blob labelled; the needle stops at 51 ms
        self.until(self.cue("fifty-one"))
        self.floor_blend(waves(nf - 1), self.pic(picture("labels")), frames=8)
        self.over_to(goal=self.goal(end.goal), frames=6)
        self.bring_to_front(self.over["strip"])     # the strip's header sits on the meter's edge
        self.say_last(at=self.cue("milliseconds"))

        # "every red pixel still rides the queue": the queue fills up with red
        # a counter beside the queue shows the scale: every red pixel of the real picture
        self.until(self.cue("rides"))
        total = int(self.num["red_px"])
        steps = 8
        count = {"m": self.side_note("0 red px", self.machine.part("queue"), side="queue", color=RED_PX)}
        self.play(FadeOut(note), FadeIn(count["m"]), run_time=fr(2))

        def tick(p: float) -> None:
            k = round(p * steps)
            if k == count.get("k"):
                return
            count["k"] = k
            new = self.side_note(f"{round(total * k / steps):,} red px", self.machine.part("queue"), side="queue",
                                 color=RED_PX)
            count["m"].become(new)        # in place: a mobject added mid-play would not be drawn

        self.extras = [count["m"]]
        self.run(tick, fr(7))           # the total holds a few frames before the fade
        self.fill_tokens(end.machine, frames=9)
        assert self.machine.spec == end.machine, (self.machine.spec, end.machine)
