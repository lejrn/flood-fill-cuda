"""r05b: the snake (v3 r05's tail, its own beat in v4). On a thin snake 48
blocks lose to one: every ring the chip stops twice at a gate.

For now a stand-in so the cut assembles: it starts on chain_state(9),
blends the floor into the painted snake at "But" (as v3 r05 did), says
the carried "65,792 stops" at "each" and snaps to chain_state(10). The
beat's own pictures (the snake filling, the times panel, the gate
flashes, the idle cubes turning amber) come with the scene work.
"""
from __future__ import annotations

from scenes.chain.base import ChainScene
from scenes.chain.state import boundary, floor_raster


class Snake(ChainScene):
    k = 9

    def beat_anim(self) -> None:
        self.until(self.cue("But"))
        (a,), (b,) = boundary(self.k).floors, boundary(self.k + 1).floors
        self.floor_blend(floor_raster(a, self.floor.rpx), floor_raster(b, self.floor.rpx), frames=6)
        self.say_last(at=self.cue("each"))
