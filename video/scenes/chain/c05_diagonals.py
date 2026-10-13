"""c05: pass the gate less often. With diagonal steps the diamond wave
becomes a square: half the rings, so half the gates. 25x the CPU."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from scenes.chain.base import ChainScene, fr
from scenes.chain.c04_gate import amber_overlay
from scenes.chain.state import level_colors, machine_spec, toys


class Diagonals(ChainScene):
    k = 5

    def beat_anim(self) -> None:
        t = toys()
        sq, l4, l8 = t["square"], t["square4"].level, t["square8"].level
        n4, n8 = int(l4.max()) + 1, int(l8.max()) + 1
        spec = machine_spec(5)
        gate = amber_overlay(replace(spec, rooms=("lit",) * 24), self.Lay.machine_pl)
        self.add(gate)
        self.until(self.cue("gate"))
        self.play(gate.animate.set_opacity(1), run_time=fr(3))
        # the square with 4 neighbours: a diamond, n4 rings
        self.until(self.cue("so"))
        self.play(gate.animate.set_opacity(0), run_time=fr(3))
        self.remove(gate)
        snake = self.toy(level_colors(t["snake"], t["snake4"].level))
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)))
        self.floor_blend(snake, red, frames=4)
        self.settle()
        d0, d1 = self.elapsed(), self.cue("Step")
        self.floor_play(lambda p: round(p * (n4 - 1)),
                        lambda i: self.toy(level_colors(sq, l4, upto=i, top=n4 - 1), ring=(l4 == i + 1)),
                        max(d1 - d0, 0.6))
        self.say(f"{n4} rings", "4 neighbours: a diamond", at=self.cue("often"))
        # step diagonally too
        self.until(d1)
        self.machine_to(replace(spec, stencil=4), frames=4)
        self.until(self.cue("diagonally"))
        self.machine_to(replace(spec, stencil=8), frames=4, slide=0)
        # 8 neighbours: a square, in half the rings
        self.until(self.cue("diamond"))
        s1 = self.cue("half")
        self.floor_play(lambda p: round(p * (n8 - 1)),
                        lambda i: self.toy(level_colors(sq, l8, upto=i, top=n8 - 1), ring=(l8 == i + 1)),
                        max(s1 - self.elapsed(), 0.6))
        self.say(f"{n4} → {n8} rings", "half the rings, half the gates", at=s1)
        self.say_last(at=self.cue("Twenty-five"))
