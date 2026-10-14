"""r08: tags. Every queue entry carries its blob's tag, so two blobs share
one launch and finish in the rings of the bigger one (max, not sum): half
the launches, about CPU speed. But the rack holds only 64 tags, and every
seed is still handed in from outside.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, Dot, FadeIn, FadeOut, SurroundingRectangle

from scenes.chain import toy
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import Z_FLOOR, Z_OVER, boundary, filled_picture, label_colors, toys
from scenes.style import GOLD, INK


def raise_strip(scene) -> None:
    """The strip's header sits on the meter's edge. The state draws it over
    the meter (same z, added later); a raised z keeps it there while the
    meter changes, and draws the same pixels."""
    s = scene.over.get("strip")
    if s is not None:
        for m in s.get_family():
            m.set_z_index(Z_OVER + 1)


def needle_move(scene, spec, frames: int = 6) -> None:
    """Move the needle to `spec`'s value: the old value label fades out, the
    triangle slides along the track, the new label fades in. The meter's box
    and labels are the same pixels before and after, so they never blink."""
    raise_strip(scene)
    old = scene.over["goal"]
    new = scene.goal(spec)
    scene.play(FadeOut(old[-1]), run_time=fr(2))
    tri, val = new[-2], new[-1]
    new.remove(tri, val)
    dx = tri.get_center()[0] - old[-2].get_center()[0]
    tri.shift(-dx * np.array([1.0, 0, 0]))
    scene.remove(old)
    scene.add(new, tri)
    scene.play(tri.animate.shift(dx * np.array([1.0, 0, 0])), run_time=fr(frames))
    scene.play(FadeIn(val), run_time=fr(3))
    scene.remove(tri, val)
    new.add(tri, val)
    scene.over["goal"] = new


class Tags(ChainScene):
    k = 10

    def beat_anim(self) -> None:
        t = toys()
        end = boundary(self.k + 1)
        two, lab = t["two"], t["two_lab"]
        ids = sorted(int(v) for v in np.unique(lab[lab >= 0]))
        seeds = [toy.centre_seed(lab == v) for v in ids]
        multi = toy.bfs(two, seeds, 8).level
        rings = [int(toy.bfs(lab == v, [s], 8).level.max()) + 1 for v, s in zip(ids, seeds)]
        top = int(multi.max())
        full = label_colors(two, lab)
        red = label_colors(two, np.full(two.shape, -1))

        # "tag each queue entry": the printer and the tagged tokens slide in
        self.until(self.cue("tag"))
        self.machine_to(replace(end.machine, rack=False, launches=boundary(self.k).machine.launches), frames=7)
        self.say("tags", "a blob's tag in every queue entry", at=self.cue("entry"))

        # "two per launch": the floor becomes two blobs, the launch count halves
        self.until(self.cue("two"))
        self.floor_blend(self.pic(filled_picture()), self.toy(red), frames=5)
        self.machine_to(replace(end.machine, rack=False), frames=6, slide=0)

        # both blobs fill at once from their seeds, each in its own colour
        def render(i: int):
            col = red.copy()
            done = (multi >= 0) & (multi <= i)
            col[done] = full[done]
            return self.toy(col, dots=((multi == i) & (i < top)).astype(int))

        t0 = self.elapsed()
        self.floor_play(lambda p: round(p * top), render, max(self.cue("speed", end=True) - t0, 1.0))
        a, b = rings
        self.say("max, not sum", f"rings: max({a}, {b}) = {max(a, b)}, not {a + b}", at=self.cue("launch", end=True))
        # "about CPU speed": the needle settles on the CPU tick
        self.until(self.cue("CPU"))
        needle_move(self, end.goal)

        # "tags run out at sixty-four": the rack slides in, then the limit
        # flashes amber twice while "run out" is said
        self.until(self.cue("tags"))
        self.machine_to(end.machine, frames=6)
        flash = [SurroundingRectangle(self.machine.parts["rack"], color=GOLD, buff=0.06, stroke_width=5)]
        for f in flash:
            f.set_z_index(Z_OVER - 10)
            f.set_stroke(opacity=0)
        self.add(*flash)
        self.extras += flash
        t_run = self.cue("run")
        self.until(t_run)
        span = max(self.cue("sixty-four", end=True) - t_run, 0.6)
        self.run(lambda p: [f.set_stroke(opacity=float(np.sin(2 * np.pi * p) ** 2)) for f in flash], span)
        self.say_last(at=self.cue("sixty-four"))

        # "every seed is handed in": white seeds drop onto the blobs from above
        self.until(self.cue("every"))
        fl0 = self.floor
        drops = []
        for s in seeds:
            d = Dot(fl0.cell_centre(*s), radius=0.09, color=INK)
            d.set_z_index(Z_FLOOR + 5)
            drops.append(d)
        self.extras += drops
        self.play(*[FadeIn(d, shift=DOWN * 1.2) for d in drops], run_time=fr(6))
        self.until(self.cue("handed", end=True))     # they stay while "handed in" is said
