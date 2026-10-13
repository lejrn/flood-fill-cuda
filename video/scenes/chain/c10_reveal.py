"""c10: the reveal. 24 seconds to 1.46 ms, and the name, only now."""
from __future__ import annotations

from dataclasses import replace

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import TEAL, fmt_ms, machine_spec


class Reveal(ChainScene):
    k = 10

    def beat_anim(self) -> None:
        n, d = self.num, self.der
        self.say(f"{n['pure_ms'] / 1000:.0f} s → {fmt_ms(n['ch06_mask_ms'])}", "pure Python → runs on the GPU",
                 at=self.cue("Twenty-four"))
        self.say(f"{int(d['c10_x'] / 1000) * 1000:,}×", "faster than where you started", at=self.cue("Congratulations"))
        # the finished machine lights up once, in the final answer's teal
        glow = mc.build(replace(mc.with_rooms(machine_spec(10), "lit"), team_state="work"), self.Lay.machine_pl)
        rooms = [v for k, v in glow.parts.items() if k.startswith("room") or k == "teams"]
        for g in rooms:
            g.set_fill(TEAL).set_opacity(0)
            self.add(g)
        self.play(*[g.animate.set_opacity(0.9) for g in rooms], run_time=fr(5))
        self.play(*[g.animate.set_opacity(0) for g in rooms], run_time=fr(8))
        for g in rooms:
            self.remove(g)
        self.say("connected-component\nlabeller", "not a flood fill any more", at=self.cue("connected-component"))
