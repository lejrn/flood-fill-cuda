"""c02: a team on the GPU paints the ring in one step, then waits for the
slowest at its room's fence (syncthreads). Twice the CPU, but one room
of twenty-four."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import FadeIn, FadeOut

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import L_MID, level_colors, toys


class Team(ChainScene):
    k = 2

    def beat_anim(self) -> None:
        t = toys()
        blob, lv = t["blob"], t["blob4"].level
        top = int(lv.max())
        one = mc.Spec(rooms=("lit",) + ("off",) * 23, teams=(1,) + (0,) * 23, shelf=True, room_fence="rest")
        self.until(self.cue("hire"))
        self.machine_to(one, frames=9)
        self.say("1 team", "256 workers in one room (an SM)", at=self.cue("team"))
        # ring by ring: paint (blue workers), then wait at the fence (amber)
        amber = mc.build(replace(one, room_fence="wait"), self.Lay.machine_pl, z0=100.0).part("room_fence")
        amber.set_opacity(0)
        self.add(amber)
        t0, t1 = self.cue("paint"), self.cue("Twice")
        rings = list(range(L_MID + 1, top + 1))
        nr = len(rings)

        def key(p: float):
            x = min(p * nr, nr - 1e-9)
            i = int(x)
            return i, (x - i) >= 0.55

        def render(k):
            i, waiting = k
            ring = lv == rings[i]
            nxt = (lv == rings[i] + 1) if waiting else None
            dots = ring.astype(int) * (2 if waiting else 1)
            return self.toy(level_colors(blob, lv, upto=rings[i]), dots=dots, ring=nxt)

        def flash(p: float) -> None:
            x = min(p * nr, nr - 1e-9)
            amber.set_opacity(1.0 if (x - int(x)) >= 0.55 and p < 1 else 0.0)

        self.until(t0)
        self.floor_play(key, render, t1 - t0)
        self.run(flash, t1 - t0)
        self.say("wait", "for the slowest, after every ring", at=self.cue("slowest"))
        self.until(t1)
        self.floor_show(self.toy(level_colors(blob, lv)))
        self.remove(amber)
        self.say(f"{round(self.der['c02_x'])}×", "the CPU, one 36 Mpx blob", at=t1)
        self.until(self.cue("twenty-four"))
        idle = mc.with_rooms(one, "idle", rooms=range(1, 24))
        self.machine_to(idle, frames=8, slide=0)
        self.say_last(at=self.cue("uses"))
