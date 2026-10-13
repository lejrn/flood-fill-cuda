"""c03: the obvious fix, more teams. Two teams, then 48: eighteen times the
CPU on one big blob. Then the snake, where 48 teams lose to one."""
from __future__ import annotations

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene
from scenes.chain.state import ALL_LIT, level_colors, machine_spec, toys

import numpy as np


class MoreTeams(ChainScene):
    k = 3

    def beat_anim(self) -> None:
        t = toys()
        n, d = self.num, self.der
        self.until(self.cue("Two"))
        two = mc.Spec(rooms=("lit", "lit") + ("idle",) * 22, teams=(1, 1) + (0,) * 22)
        self.machine_to(two, frames=6)
        self.say(f"{round(d['c03_two_x'])}×", "two teams vs one, one 36 Mpx blob", at=self.cue("twice"))
        self.until(self.cue("Forty-eight"))
        self.machine_to(machine_spec(4), frames=8)
        self.say(f"{int(d['c03_n_x'])}×", f"the CPU: {int(n['c03_blocks'])} teams, one 64 Mpx blob",
                 at=self.cue("eighteen"))
        self.until(self.cue("snake"))
        blob = self.toy(level_colors(t["blob"], t["blob4"].level))
        snake = self.toy(level_colors(t["snake"], np.full(t["snake"].shape, -1)))
        self.floor_blend(blob, snake, frames=6)
        self.say_last(at=self.cue("lose"))
