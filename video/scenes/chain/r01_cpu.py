"""r01: the CPU. One worker paints the drone one pixel per step, queueing
its red neighbours; its queue is the wave's edge. One P-core works while
the GPU's 24 SMs sit dark.

Start on chain_state(k), cue every change on a spoken word, end exactly on
chain_state(k + 1) (base.construct snaps to it), and say the carried
caption last.
"""
from __future__ import annotations

import numpy as np
from manim import AnimationGroup

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import level_colors, toys


class Cpu(ChainScene):
    k = 3

    def ripple(self, spec: mc.Spec, frames: int) -> None:
        """machine_to(spec), the SMs' changes swept across the die column by
        column, left to right (GPC 0 first)."""
        old = self.machine
        anims = self.machine_anims(spec, slide=0)
        name = {id(p): n for m in (old, self.machine) for n, p in m.parts.items()}
        cols: dict = {}
        for a in anims:
            n = name.get(id(a.mobject), "")
            x = round(mc.sm_rect(int(n[4:])).x, 2) if n[4:].isdigit() else -1.0
            cols.setdefault(x, []).append(a)
        sweep = [AnimationGroup(*cols[x]) for x in sorted(cols)]
        self.play(AnimationGroup(*sweep, lag_ratio=0.35), run_time=fr(frames))

    def beat_anim(self) -> None:
        t = toys()
        drone, fill = t["drone"], t["drone4"]
        order, qlen = fill.order, fill.queue_len
        total, top = len(order), int(fill.level.max())
        cpu = mc.Spec(cpu="lit")
        # both boards slide in; P-core 0 runs our CPU code, the GPU is dark
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
        # the sentence names three things, each kept the captions' least dwell (MIN_DWELL):
        # the first lands in the pause before "One", so the next two keep up
        self.say("1 pixel per step", at=self.cue("One") - 0.5)
        self.say(f"1 core of {int(self.num['cpu_cores'])}", at=self.cue("core") - fr(2))
        # the 24 SMs come up idle in a wave, left to right, then go dark again
        self.until(self.cue("twenty-four"))
        self.ripple(mc.with_rooms(cpu, "idle"), frames=12)
        self.say_last(at=self.cue("SMs") - fr(2))
        # they stay up under their caption and go dark as "idle" ends
        self.until(self.cue("idle", end=True) - 0.3)
        self.ripple(cpu, frames=8)
