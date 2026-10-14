"""r09: the GPU finds its own seeds. The tags leave the queue and P-core 0
sends three launches over the link (the fill, then its two cleanups);
once the fill lands, the SMs spark: on the many-blob toy every red pixel
with no red neighbour before it in scan order sparks a seed, then a
stride-4 lattice adds more (ch05 runs a stride-8 lattice on the real
picture). The seeds are found on the GPU, inside the fill launch, not
handed over the link. Waves start from every seed at once; where two
meet, both keep the smaller label (the colours merge). The counter shows
3 launches, the needle stops at 51 ms. But the queue in VRAM still
carries every red pixel.

Starts on chain_state(11), ends on chain_state(12).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Dot, FadeIn, FadeOut, LaggedStart, RoundedRectangle, VGroup, VMobject,
                   config)

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain import toy
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import BLOB_HUES, Z_MACHINE, Z_OVER, boundary, label_colors, picture, toys
from scenes.style import GOLD, INK, RED_PX, label

STRIDE = 4      # the toy's lattice; the real run (split_L8) uses 8 on a 9000 x 9000 picture
Z_SPARK = Z_MACHINE + 90        # over the dies and the blocks, under the silkscreen and the launch counter
# the blob hues without the amber one: amber means waiting, only
HUES = tuple(h for h in BLOB_HUES if np.abs(fl.rgb(h).astype(int) - fl.rgb(GOLD).astype(int)).max() > 40)


class Seeds(ChainScene):
    k = 11

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
        self.machine = mc.Machine(spec, {k: parts[k] for k in new.parts}, dict(new.keys))

    def sm_sparks(self, seconds: float) -> list:
        """A white spark on every SM, each at its own moment: the blocks find
        the seeds themselves (in r08 they came over the link)."""
        pl = self.Lay.machine_pl
        phase = np.random.default_rng(9).uniform(0.0, 0.45, 24)
        dots = []
        for i in range(24):
            d = Dot(mc.sm_point(pl, i), radius=0.055, color=INK, fill_opacity=0)
            d.set_z_index(Z_SPARK)
            dots.append(d)
        self.add(*dots)

        def step(p: float) -> None:
            for d, ph in zip(dots, phase):
                q = min(max((p - ph) / 0.55, 0.0), 1.0)
                d.set_fill(opacity=float(np.sin(np.pi * q)))

        self.run(step, seconds)
        return dots

    def launch_stream(self, start: float, travel: float, gap: float, burst: float) -> tuple[list, list]:
        """Three launches leave P-core 0 for the GPU's GigaThread engine at
        scene time `start`, one after another (the host queues them without
        waiting): the fill, then its two cleanups, smaller. Each flares as it
        lands. Returns the dots and their landing times."""
        path = VMobject().set_points_as_corners(mc.link_path(self.Lay.machine_pl))
        dots = []
        for j in range(3):
            d = Dot(path.get_start(), radius=0.07 if j == 0 else 0.05, color=INK, fill_opacity=0)
            d.set_z_index(Z_SPARK)
            dots.append(d)
        base = [d.width for d in dots]
        self.add(*dots)
        total = 2 * gap + travel + burst

        def step(p: float) -> None:
            tt = p * total
            for j, d in enumerate(dots):
                s = (tt - j * gap) / travel
                if s < 1:
                    if s >= 0:
                        d.move_to(path.point_from_proportion(s))
                    d.set_fill(opacity=1.0 if s >= 0 else 0.0)
                else:
                    u = min((s - 1) * travel / burst, 1.0)
                    d.set_width(base[j] * (1 + 1.5 * u)).move_to(path.get_end())
                    d.set_fill(opacity=1.0 - u)

        self.run(step, total, start=start)
        return dots, [start + j * gap + travel for j in range(3)]

    def needle_slide(self, spec: ov.GoalSpec, frames: int = 6):
        """The needle slides to `spec`'s value: the old value label fades out,
        the triangle slides along the track. Returns the new value label,
        not yet shown (the caller fades it in with the caption). The meter's
        box and labels are the same pixels before and after."""
        old = self.over["goal"]
        new = self.goal(spec)
        self.play(FadeOut(old[-1]), run_time=fr(2))
        tri, val = new[-2], new[-1]
        new.remove(tri, val)
        dx = tri.get_center()[0] - old[-2].get_center()[0]
        tri.shift(LEFT * dx)
        self.remove(old)
        self.add(new, tri)
        self.play(tri.animate.shift(RIGHT * dx), run_time=fr(frames))
        self.remove(tri)
        new.add(tri)
        self.over["goal"] = new
        return new, val

    @staticmethod
    def note_mob(text: str) -> VGroup:
        t = label(text, 15, RED_PX, mono=True)
        box = RoundedRectangle(corner_radius=0.06, width=t.width + 0.2, height=t.height + 0.14,
                               fill_color=ov.PANEL, fill_opacity=0.92, stroke_width=0)
        t.move_to(box)
        return VGroup(box, t)

    def note_anchor(self, widest: str) -> np.ndarray:
        """Top-right corner for the counter note: just in front of the GPU
        board's front edge, below the queue's VRAM chips, clear of the
        silkscreen and the frame edge (the widest text decides the height,
        so the note never moves while it counts)."""
        pl = self.Lay.machine_pl
        a, b = mc.gpu_corner(pl, "front-left"), mc.gpu_corner(pl, "front-right")
        right = min(mc.slot_point(pl, mc.SLOTS - 1, z=0.0)[0] + 0.1, config.frame_width / 2 - 0.3)
        left = right - self.note_mob(widest).width

        def edge_y(x: float) -> float:
            return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])

        return np.array([right, min(edge_y(left), edge_y(right)) - 0.16, 0.0])

    def queue_note(self, text: str, anchor: np.ndarray) -> VGroup:
        """The counter note by the queue (a transient extra: construct()
        fades it before the snap)."""
        m = self.note_mob(text)
        m.move_to(anchor + np.array([-m.width / 2, -m.height / 2, 0.0]))
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
        hue = {v: fl.rgb(HUES[i % len(HUES)]) for i, v in enumerate(fin + rest)}
        assert len(fin) <= len(HUES), fin
        red = label_colors(many, np.full(many.shape, -1))
        seed_dots = (cand | lat).astype(int) * 3
        end = boundary(self.k + 1)
        pl = self.Lay.machine_pl

        # "finds": the tags lift off the queue and go, the caption with them;
        # the floor becomes many blobs, all red. P-core 0 lights and sends the
        # three launches (ch05's split build: init, scan and fill in one
        # launch, then compress and finish), so the seeds are found inside
        # the fill launch, after it lands
        plain = replace(self.machine.spec, tag_tokens=())
        core = mc.build(replace(plain, cpu="lit"), pl, z0=Z_MACHINE).parts["pcore"]
        t_finds = self.cue("finds")
        self.until(t_finds)
        tags = self.machine.parts.get("tags")
        self.machine_anims(plain)                    # the bookkeeping; the tags leave upward instead
        pulses, lands = self.launch_stream(t_finds + fr(4), travel=fr(5), gap=fr(3), burst=fr(3))
        self.floor_blend(self.toy(label_colors(t["two"], t["two_lab"])), self.toy(red), frames=6)
        self.unsay(frames=6, extra=[FadeIn(core)] + ([FadeOut(tags, shift=UP * 0.3)] if tags is not None else []))

        # "seeds": sparks at every top-left corner (scan order), then the
        # lattice; a spark is a white ring for a moment, then a seed dot.
        # The SMs spark with them.
        cand_order = sorted(zip(*np.nonzero(cand)))
        lat_order = sorted(zip(*np.nonzero(lat)))
        both = cand_order + lat_order
        nc, nl = len(cand_order), len(lat_order)

        def sparks(n: int):
            d = np.zeros(many.shape, int)
            ring = np.zeros(many.shape, bool)
            for j, rc in enumerate(both[:n]):
                d[rc] = 3
                ring[rc] = j >= n - 3
            return self.toy(red, dots=d, ring=ring)

        t_seeds, t_every = max(self.cue("seeds"), lands[0]), self.cue("every")      # never before the fill lands
        self.until(t_seeds)
        span = max(t_every - t_seeds - fr(2), fr(8))
        self.floor_play(lambda p: max(1, min(nc + nl + 3, round(p * (nc + nl + 3)))), sparks, span)
        sm_dots = self.sm_sparks(span)
        # the last launch has landed: the counter shows 3, the CPU's part is done
        self.until(lands[-1])
        self.machine_to(replace(plain, launches="3"), frames=3, slide=0, extra=[FadeOut(core)])

        # "every blob in one fill launch": waves start from every seed at
        # once; a merge recolours both waves to the smaller label
        nf = len(mf.resolved)

        def waves(i: int):
            r = mf.resolved[i]
            col = red.copy()
            for v in np.unique(r[r >= 0]):
                col[r == v] = hue[int(v)]
            last = i >= nf - 1
            front = (mf.frames[i] >= 0) & ((mf.frames[i - 1] < 0) if i else True)
            return self.toy(col, dots=None if last else seed_dots, ring=None if last else front)

        self.until(t_every)
        self.remove(*sm_dots, *pulses)
        t_fifty = self.cue("fifty-one")
        self.floor_play(lambda p: min(nf - 1, int(p * nf)), waves, max(t_fifty - self.elapsed() - fr(2), fr(10)))

        # "fifty-one milliseconds": the real picture, every blob labelled; the
        # needle slides to 51 ms, its value comes in, then the caption
        self.until(t_fifty)
        self.floor_blend(waves(nf - 1), self.pic(picture("labels")), frames=8)
        meter, val = self.needle_slide(end.goal)
        self.play(FadeIn(val), run_time=fr(3))
        self.remove(val)
        meter.add(val)
        self.say_last(at=self.cue("milliseconds"))

        # "every red pixel still rides the queue": a counter in front of the
        # queue's VRAM climbs to every red pixel of the real picture, and at
        # "rides" the queue fills up
        self.until(self.cue("pixel"))
        total = int(self.num["red_px"])
        steps = 10
        anchor = self.note_anchor(f"{total:,} red px")
        note = {"m": self.queue_note("0 red px", anchor)}
        self.play(FadeIn(note["m"]), run_time=fr(2))

        def tick(p: float) -> None:
            k = round(p * steps)
            if k == note.get("k"):
                return
            note["k"] = k
            # in place: a mobject added mid-play would not be drawn
            note["m"].become(self.queue_note(f"{round(total * k / steps):,} red px", anchor))

        self.extras = [note["m"]]
        self.run(tick, max(self.cue("queue") - self.elapsed(), fr(8)))
        self.until(self.cue("rides"))
        self.fill_tokens(end.machine, frames=9)
        # the total holds while "queue" is said, then goes before the snap
        self.until(min(self.speech_end() + 0.3, self.target_seconds() - 0.2))
        assert self.machine.spec == end.machine, (self.machine.spec, end.machine)
