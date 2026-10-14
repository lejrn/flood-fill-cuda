"""r04: a second block on a second SM. Room 1 lights with its own team; the
shelf and the room fence go (one shared queue in global memory was
faster), the queue slides in along the front, and the chip-wide gate
appears at rest (it pays off in r05). The gauge reads 2 as room 1 lights.
The square refills at twice r03's pace, each ring split between the two
blocks: block 0 (left half, blue dots) and block 1 (right half, green
dots, and room 1's team tinted green while it works). At "twenty-two" the
22 idle rooms dim for a moment: twenty-two SMs stay dark.

Starts on chain_state(6) and ends on chain_state(7).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import FadeIn, FadeOut, VGroup

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.r03_giant import F15, plan_index, r03_plan
from scenes.chain.state import boundary, floor_raster, level_colors, toys
from scenes.style import GREEN

RESET = 5               # frames of the blend back to red
DOT1 = GREEN            # block 1's workers on the floor, as its team in room 1
DARK = "#0d1116"        # the idle rooms, dimmed for a moment at "twenty-two"


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


class TwoBlocks(ChainScene):
    k = 6

    def beat_anim(self) -> None:
        t = toys()
        sq, lv = t["square"], t["square4"].level
        n = sq.shape[0]
        top = int(lv.max())
        f = self.floor
        b7 = boundary(7)
        pl = self.Lay.machine_pl
        painted = floor_raster("square4", f.rpx)
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)))
        pa, ra = painted.astype(np.float32), red.astype(np.float32)

        # the refill: back to red, then r03's plan at twice the pace (every
        # second frame of it, without the fence waits: the fence is gone)
        plan = [i for i, _ in r03_plan()[::2]]
        frames = RESET + len(plan)
        total = frames / F15

        def key(p: float):
            j = plan_index(p, total)
            if p >= 1 or j >= frames:
                return ("done", 0)
            if j < RESET:
                return ("reset", j)
            return ("ring", plan[j - RESET])

        def render(k):
            kind, i = k
            if kind == "done":
                return painted
            if kind == "reset":
                return (pa + (ra - pa) * (i / RESET)).astype(np.uint8)
            b0, b1 = block_dots(n, lv == i)
            out = self.toy(level_colors(sq, lv, upto=i, top=top), dots=b0.astype(int))
            return paint_dots(out, b1, DOT1)

        # room 1's team in block 1's colour while the two blocks work
        team1 = mc.build(replace(b7.machine, teams=(0, 1) + (0,) * 22), pl, z0=100.5).part("teams")
        team1.set_fill(GREEN)
        # the 22 idle rooms, darker, for the dim at "twenty-two"
        idle = mc.build(mc.with_rooms(b7.machine, "off"), pl, z0=100.5)
        dim = VGroup(*[idle.part(f"room{r}") for r in range(2, 24)])
        dim.set_fill(DARK)

        self.until(self.cue("second"))
        self.unsay(frames=2)
        self.floor_play(key, render, total)
        # room 1 lights, the shelf and fence go, the queue and gate arrive, the gauge reads 2
        self.machine_to(b7.machine, frames=8,
                        extra=[FadeIn(team1), *self.over_anims("gauge", self.gauge(b7.gauge))])
        self.extras = [team1]

        self.say(f"{round(self.der['big_2v1_x'])}× again",
                 f"vs one block; {self.der['big_2blocks_x']:.1f}× the CPU", at=self.cue("twice"))

        # twenty-two SMs stay dark: the idle rooms dim, then come back as they were
        self.until(self.cue("twenty-two"))
        self.play(FadeIn(dim), FadeOut(team1), run_time=fr(5))
        self.extras = []
        self.until(self.cue("SMs", n=1, after=self.cue("twenty-two")) + 0.3)
        self.play(FadeOut(dim), run_time=fr(5))
        self.say_last(at=self.cue("dark"))
