"""r08: tags. Every queue entry in VRAM carries its blob's tag, so two blobs
share one launch and finish in the rings of the bigger one (max, not sum):
half the launches, about CPU speed. But tags run out at 64, and every seed
still crosses the PCIe link from the CPU.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import Dot, FadeIn, FadeOut, VMobject

from scenes.chain import machine as mc
from scenes.chain import toy
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import Z_MACHINE, boundary, filled_picture, label_colors, toys
from scenes.style import INK

Z_SEED = Z_MACHINE + 90         # over the dies, under the silkscreen and the launch pill (as r07, r09)
LAND = 7                        # frames from say(at=t) to its caption landing (2 out, 5 in)


def needle_move(scene, spec, frames: int = 6) -> None:
    """Move the needle to `spec`'s value: the old value label fades out, the
    triangle slides along the track, the new label fades in. The meter's box
    and labels are the same pixels before and after, so they never blink."""
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


def along(pts: list, s: float) -> np.ndarray:
    """The point a fraction s (0..1) of the way along the polyline pts, by length."""
    seg = [np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])]
    d = min(max(s, 0.0), 1.0) * sum(seg)
    for a, b, n in zip(pts, pts[1:], seg):
        if d <= n:
            return a + (b - a) * (d / n if n else 0.0)
        d -= n
    return pts[-1]


class Tags(ChainScene):
    k = 10

    def beat_anim(self) -> None:
        t = toys()
        b1 = boundary(self.k + 1)
        start, end = boundary(self.k).machine, b1.machine
        two, lab = t["two"], t["two_lab"]
        ids = sorted(int(v) for v in np.unique(lab[lab >= 0]))
        seeds = [toy.centre_seed(lab == v) for v in ids]
        multi = toy.bfs(two, seeds, 8).level
        top = int(multi.max())
        full = label_colors(two, lab)
        red = label_colors(two, np.full(two.shape, -1))

        # "tag each queue entry": the entries land in the VRAM queue, the
        # floor becomes two blobs, and at "blob" each entry gets its blob's tag
        # while the launch counter halves (its own text change, apart from
        # the caption's)
        self.until(self.cue("tag"))
        self.machine_to(replace(start, tokens=end.tokens), frames=6)
        self.until(self.cue("entry"))
        self.floor_blend(self.pic(filled_picture()), self.toy(red), frames=8)
        self.until(self.cue("blob"))
        self.machine_to(end, frames=6)
        t_pill = self.elapsed()

        # "two per launch": both blobs fill at once from their seeds, each in
        # its own colour
        def render(i: int):
            col = red.copy()
            done = (multi >= 0) & (multi <= i)
            col[done] = full[done]
            return self.toy(col, dots=((multi == i) & (i < top)).astype(int))

        t_two = self.cue("two")
        self.until(t_two)
        self.floor_play(lambda p: round(p * top), render, max(self.cue("speed", end=True) - t_two, 1.0))
        self.say("max, not sum", at=max(self.cue("launch"), t_pill + 1.4 - fr(LAND)))

        # "about CPU speed": the needle settles by the CPU tick (its new label
        # never lands within MIN_DWELL of the caption)
        self.until(max(self.cue("CPU"), self._cap_t + 1.4 - fr(11)))
        needle_move(self, b1.goal)
        t_needle = self.elapsed()

        # "tags run out at sixty-four": the tag caps flash white twice
        glow = self.machine.parts["tags"].copy()
        for m in glow.get_family():
            if isinstance(m, VMobject):
                m.set_fill(INK, opacity=0).set_stroke(opacity=0)
        self.add(glow)
        self.extras.append(glow)
        t_run = self.cue("run")
        self.until(t_run)

        def flash(p: float) -> None:
            a = 0.8 * float(np.sin(2 * np.pi * p) ** 2)
            for m in glow.get_family():
                if isinstance(m, VMobject):
                    m.set_fill(opacity=a)

        self.run(flash, max(self.cue("sixty-four", end=True) - t_run, 0.6))
        self.say_last(at=max(self.cue("sixty-four"), t_needle + 1.4 - fr(LAND)))

        # "every seed is handed in": P-core 0 lights (our host code runs) and
        # seeds run from it over the PCIe link to the GPU, one after another
        core = mc.build(replace(end, cpu="lit"), self.Lay.machine_pl, z0=Z_MACHINE).parts["pcore"]
        self.extras.append(core)
        path = mc.link_path(self.Lay.machine_pl)
        t0, t1 = self.cue("every"), self.cue("in", end=True)
        n, travel = 6, 0.8
        gap = max((t1 - t0 - travel) / (n - 1), 0.08)
        dots = []
        for _ in range(n):
            d = Dot(path[0], radius=0.055, color=INK)
            d.set_z_index(Z_SEED)
            d.set_opacity(0)
            dots.append(d)
        self.add(*dots)
        self.extras += dots
        self.until(t0)

        def stream(p: float) -> None:
            tt = p * (t1 - t0)
            for j, d in enumerate(dots):
                s = (tt - j * gap) / travel
                d.move_to(along(path, s))
                d.set_opacity(1.0 if 0 <= s < 1 else 0.0)

        self.run(stream, t1 - t0)
        self.play(FadeIn(core), run_time=fr(3))
        self.until(t1)
