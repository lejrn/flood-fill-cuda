"""r01: the CPU. One worker paints the drone one pixel per step, queueing
its red neighbours; its queue is the wave's edge. One core works while the
GPU's 24 rooms (SMs) sit dark.

Example scene for the v2 pattern: start on chain_state(k), cue every
change on a spoken word, end exactly on chain_state(k + 1) (base.construct
snaps to it), and say the carried caption last.
"""
from __future__ import annotations

import numpy as np

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import level_colors, toys


class Cpu(ChainScene):
    k = 3

    def beat_anim(self) -> None:
        t = toys()
        drone, fill = t["drone"], t["drone4"]
        order, qlen = fill.order, fill.queue_len
        total, top = len(order), int(fill.level.max())
        cpu = mc.Spec(cpu=True)
        # the board slides in: a CPU with its queue, beside a dark GPU chip
        self.until(self.cue("CPU"))
        self.machine_to(cpu, frames=9)

        # one worker walks the drone in queue order: slow for the first cells, then faster
        t0, t1 = self.cue("paint"), self.cue("while")

        def key(p: float) -> int:
            return min(total, int(np.ceil(total * p ** 1.6)))

        def render(n: int):
            lv = np.full(drone.shape, -1)
            for (r, c) in order[:n]:
                lv[r, c] = fill.level[r, c]
            dots = np.zeros(drone.shape, int)
            ring = np.zeros(drone.shape, bool)
            if 0 < n < total:
                dots[order[n - 1]] = 1
                for (r, c) in order[n:n + qlen[n - 1]]:
                    ring[r, c] = True
            return self.toy(level_colors(drone, lv, top=top), dots=dots, ring=ring)

        self.until(t0)
        self.floor_play(key, render, t1 - t0)
        self.say("1 pixel", "per step", at=self.cue("One"))
        self.say("1 core", f"our CPU code; the i9 has {int(self.num['cpu_cores'])}", at=self.cue("core"))
        # the 24 rooms light up for a moment, then go dark again: idle
        self.until(self.cue("twenty-four"))
        self.machine_to(mc.with_rooms(cpu, "idle"), frames=4, slide=0)
        self.machine_to(cpu, frames=6, slide=0)
        self.say_last(at=self.cue("SMs"))
