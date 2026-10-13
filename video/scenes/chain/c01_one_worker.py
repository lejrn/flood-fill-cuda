"""c01: one worker. The CPU walks the blob in queue order; its queue is the
wave's edge. Then the picture stops mid-wave: that edge is one ring of
cells that do not depend on each other."""
from __future__ import annotations

import numpy as np

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import GOLD, L_MID, fmt_ms, level_colors, mid_wave, toys


class OneWorker(ChainScene):
    k = 1

    def beat_anim(self) -> None:
        t = toys()
        blob, fill = t["blob"], t["blob4"]
        order, qlen = fill.order, fill.queue_len
        total = len(order)
        top = int(fill.level.max())
        self.until(self.cue("Start"))
        self.machine_to(mc.Spec(cpu=True), frames=9)
        t0, t1 = self.cue("worker"), self.cue("Congratulations")
        self.say("1 worker", "one pixel at a time", at=t0)

        def key(p: float) -> int:
            return min(total, int(np.ceil(total * p ** 2)))

        def render(n: int):
            lv = np.full(blob.shape, -1)
            for (r, c) in order[:n]:
                lv[r, c] = fill.level[r, c]
            dots = np.zeros(blob.shape, int)
            ring = np.zeros(blob.shape, bool)
            if 0 < n < total:
                dots[order[n - 1]] = 1
                for (r, c) in order[n:n + qlen[n - 1]]:
                    ring[r, c] = True
            return self.toy(level_colors(blob, lv, top=top), dots=dots, ring=ring)

        self.floor_play(key, render, t1 - self.elapsed())
        self.say("filled", at=t1)
        self.until(self.cue("real"))
        self.goal_to((fmt_ms(self.num["njit_ms"]), GOLD))
        self.say("1.3 s", "real picture, CPU (@njit)", at=self.cue("picture"))
        self.until(self.cue("But"))
        lv = fill.level
        self.floor_blend(self.toy(level_colors(blob, lv)),
                         self.toy(level_colors(blob, lv, upto=L_MID)), frames=6)
        self.until(self.cue("ring"))
        self.floor_show(mid_wave(self.floor.rpx))
        self.say_last(at=self.cue("could"))
