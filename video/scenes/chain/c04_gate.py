"""c04: the hidden reason. After every ring all teams meet at one gate
across the chip (grid.sync) and wait for the last. On the snake a ring is
one pixel: one worker paints, everyone else waits, 65,792 times."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import VGroup

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import level_colors, machine_spec, toys


def amber_overlay(spec: mc.Spec, pl) -> VGroup:
    """The rooms, teams and gate of `spec` in their waiting colours, invisible."""
    w = mc.build(replace(mc.with_rooms(spec, "wait"), team_state="wait",
                         gate="wait" if spec.gate else None), pl, z0=100.0)
    g = VGroup(*[v for k, v in w.parts.items() if k.startswith("room") or k in ("teams", "gate")])
    g.set_opacity(0)
    return g


class Gate(ChainScene):
    k = 4

    def beat_anim(self) -> None:
        t = toys()
        sn, lv = t["snake"], t["snake4"].level
        top = int(lv.max())
        gated = replace(machine_spec(4), gate="rest")
        self.until(self.cue("gate"))
        self.machine_to(gated, frames=7)
        self.say("the gate", "grid.sync: every team on the chip", at=self.cue("across"))
        amber = amber_overlay(gated, self.Lay.machine_pl)
        self.add(amber)
        self.until(self.cue("wait"))
        self.play(amber.animate.set_opacity(1), run_time=fr(3))
        self.until(self.cue("last", end=True))
        self.play(amber.animate.set_opacity(0), run_time=fr(3))
        # the snake, one pixel per ring: a few slow rings, then the rest fast
        t0, t1 = self.cue("snake's"), self.cue("times", end=True)
        slow, period = 8, 0.3
        fast_t = max(t1 - t0 - slow * period, 0.5)

        def phase(p: float):
            tau = p * (t1 - t0)
            if tau < slow * period:
                i = int(tau / period)
                return i, (tau / period - i) >= 0.4
            q = min((tau - slow * period) / fast_t, 1.0)
            return min(top, slow + int(q * (top + 1 - slow))), None

        def render(k):
            i, waiting = k
            if p_done(i):
                return self.toy(level_colors(sn, lv))
            dots = (lv == i).astype(int) * (2 if waiting else 1)
            return self.toy(level_colors(sn, lv, upto=i if waiting is not False else i - 1, top=top), dots=dots)

        def p_done(i: int) -> bool:
            return i >= top

        def flash(p: float) -> None:
            i, waiting = phase(p)
            if p >= 1:
                amber.set_opacity(0)
            elif waiting is None:
                amber.set_opacity(0.85)
            else:
                amber.set_opacity(1.0 if waiting else 0.0)

        self.until(t0)
        self.floor_play(phase, render, t1 - t0)
        self.run(flash, t1 - t0)
        self.say("1 works", f"{int(self.der['c04_threads']) - 1:,} wait", at=self.cue("paints"))
        self.say_last(at=self.cue("Sixty-five"))
        self.settle()
        self.remove(amber)
