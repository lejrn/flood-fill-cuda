"""r03b: one SM of twenty-four (v3 r03's tail, its own beat in v4). Over
twice the CPU, but that's one SM of twenty-four.

For now a stand-in so the cut assembles: it starts on chain_state(6), says
the carried "1 of 24 SMs" at "SM" and snaps to chain_state(7). The beat's
own pictures ("2.46x" at "twice", the 23 other SMs rippling idle at
"twenty-four") come with the scene work.
"""
from __future__ import annotations

from scenes.chain.base import ChainScene


class OneSm(ChainScene):
    k = 6

    def beat_anim(self) -> None:
        self.say_last(at=self.cue("SM"))
