"""r04: a second block on a second SM. At "So" the carried caption leaves
and the square goes back to red: a fresh run. At "second" block 1 (a
green cube, its own colour while it works) drops onto SM 1, and SM 1
lights at "second SM". At "SM" SM 0's shared-memory queue and its fence
go: two blocks share one queue, in VRAM, and the chip-wide gate
(grid.sync) appears as a low rail at rest (it pays off in r05). Then the
square refills at twice r03's pace, each ring split between the two
blocks: block 0 (left half, blue dots) and block 1 (right half, green
dots); when it is done, block 1's cube turns blue like block 0's. "2×
again" (kernel time) at "twice". At "twenty-two" the 22 dark SMs ripple
idle, the carried "22 SMs idle" lands on "SMs", and at "dark" they go
dark again.

Starts on chain_state(6) and ends on chain_state(7).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, FadeIn, FadeOut, Group, ImageMobject, ManimColor

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.r03_giant import F15, plan_index, r03_plan
from scenes.chain.state import boundary, floor_raster, level_colors, toys
from scenes.style import BLUE, GREEN

DOT1 = GREEN            # block 1's colour: its dots on the floor, its cube on SM 1 while it works
DROP = 0.7              # how far above SM 1 block 1's cube starts its drop (frame units)
LIFT = 40.0             # block 1's cube drawn above the machine's top layer (SM 0's fence) as it drops past it
GLOW = 0.6              # seconds of the idle ripple over the 22 dark SMs
GLOW_W = 0.4            # each SM's share of that ripple (the rest is the lag across them)


def block_dots(n: int, ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ring's cells split between two blocks: left half, right half."""
    left = np.zeros((n, n), bool)
    left[:, : n // 2] = True
    return ring & left, ring & ~left


def paint_dots(raster: np.ndarray, cells: np.ndarray, hex_: str) -> np.ndarray:
    """Draw dots of colour hex_ on the given cells (as toy_raster draws its dots)."""
    rpx, n = raster.shape[0], cells.shape[0]
    idx, off, _ = fl._index_maps(rpx, n)
    d = cells[idx][:, idx]
    inside = off[:, None] ** 2 + off[None, :] ** 2 < 0.27 ** 2
    raster[d & inside, :3] = fl.rgb(hex_)
    return raster


def tinted(part: Group, src: str, dst: str, dz: float = 0.5) -> Group:
    """A copy of a machine part's box faces, each face's shade of src turned
    into the same shade of dst, drawn dz above the original."""
    s, d = ManimColor(src).to_rgb(), ManimColor(dst).to_rgb()
    j = int(np.argmax(s))
    g = part.copy()
    for face in g.submobjects:
        k = float(face.get_fill_color().to_rgb()[j]) / float(s[j])
        face.set_fill(ManimColor.from_rgb(tuple(float(v) * k for v in d)))
        face.set_z_index(face.z_index + dz)
    return g


def images(part: Group) -> list:
    return [m for m in part.get_family() if isinstance(m, ImageMobject)]


class TwoBlocks(ChainScene):
    k = 6

    def adopt(self, spec: mc.Spec, name: str):
        """Part `name` of spec's build becomes the machine's own, added
        behind everything: the old part, still on top, can then fade out
        over it (or be removed where the two are the same pixels)."""
        m = mc.build(spec, self.Lay.machine_pl, z0=100.0)
        old, new = self.machine.parts.get(name), m.part(name)
        self.bring_to_back(new)
        self.machine.parts[name] = new
        self.machine.keys[name] = m.keys[name]
        return old

    def beat_anim(self) -> None:
        t = toys()
        sq, lv = t["square"], t["square4"].level
        n = sq.shape[0]
        top = int(lv.max())
        f = self.floor
        b6, b7 = boundary(6), boundary(7)
        pl = self.Lay.machine_pl
        painted = floor_raster("square4", f.rpx)
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)))

        # one block's machine (shelf, fence) with SM 1 lit and both cubes: between b6 and b7
        both = replace(b6.machine, rooms=b7.machine.rooms, teams=b7.machine.teams)
        # block 1's cube in its own colour, over the machine's blue one (and over SM 0's fence, which
        # it passes in front of on its way down)
        green = tinted(mc.build(replace(b7.machine, teams=(0, 1) + (0,) * 22), pl).part("teams"), BLUE, DOT1,
                       dz=LIFT)
        # the 22 SMs that stay dark, lit idle for a moment (nearest SM 1 first)
        idle_m = mc.build(mc.with_rooms(b7.machine, "idle", range(2, 24)), pl)
        c1 = mc.sm_point(pl, 1)
        order = sorted(range(2, 24), key=lambda r: float(np.linalg.norm(mc.sm_point(pl, r) - c1)))
        idle = [idle_m.part(f"room{r}") for r in order]

        # ---- "So": a fresh run; the carried caption leaves, the square goes back to red
        self.until(self.cue("So"))
        self.floor_blend(painted, red, frames=5)
        self.unsay(frames=4)

        # ---- "second block": block 1 drops onto SM 1; SM 1 lights at "second SM"
        self.until(self.cue("second"))
        self.play(FadeIn(green, shift=DOWN * DROP), run_time=fr(8))
        self.remove(self.adopt(both, "teams"))          # cube 1, blue, under the green one
        self.extras = [green]
        self.until(self.cue("second", n=2))
        self.machine_to(both, frames=4, slide=0)

        # ---- "SM": the shelf and the fence go; the shared queue in VRAM, the gate at rest
        self.until(self.cue("SM"))
        self.machine_to(b7.machine, frames=7, extra=[FadeOut(self.adopt(b7.machine, "room0"))])

        # ---- the refill: r03's plan at twice the pace (every second frame of
        # it, without the fence waits: the fence is gone)
        plan = [i for i, _ in r03_plan()[::2]]
        total = len(plan) / F15

        def key(p: float):
            j = plan_index(p, total)
            return plan[j] if (j < len(plan) and p < 1) else "done"

        def render(i):
            if i == "done":
                return painted
            b0, b1 = block_dots(n, lv == i)
            out = self.toy(level_colors(sq, lv, upto=i, top=top), dots=b0.astype(int))
            return paint_dots(out, b1, DOT1)

        self.until(max(self.cue("SM", end=True), self.elapsed()))
        t_done = self.elapsed() + total
        self.floor_play(key, render, total)

        # one block's kernel time over two blocks': "kernel time", as r03's and r05's ratios say
        self.say(f"{round(self.der['big_2v1_x'])}× again", "kernel time", at=self.cue("twice"))

        # ---- "twenty-two SMs stay dark": they ripple idle, then go dark again
        def glow(p: float) -> None:
            for j, tile in enumerate(idle):
                a = (p - (1 - GLOW_W) * j / (len(idle) - 1)) / GLOW_W
                for im in images(tile):
                    im.set_opacity(min(max(a, 0.0), 1.0))

        glow(0.0)
        self.add(*idle)
        self.run(glow, GLOW, start=self.cue("twenty-two"))
        self.extras = [green, *idle]

        # block 1's work is done: its cube turns blue like block 0's
        self.until(t_done)
        self.play(FadeOut(green), run_time=fr(4))
        self.extras = list(idle)

        self.say_last(at=self.cue("SMs"))
        self.until(self.cue("dark"))
        self.play(FadeOut(Group(*idle)), run_time=fr(7))
        self.extras = []
