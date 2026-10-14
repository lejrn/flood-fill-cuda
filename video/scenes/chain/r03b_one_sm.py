"""r03b: one SM of twenty-four (v3 r03's tail, its own beat in v4). The
painted square holds: the one block on SM 0, its shared ring plus the
global spill, as r03_giant left it. At "twice" the speed-up lands ("2.46×",
kernel time); at "SM" the carried "1 of 24 SMs"; at "twenty-four" the 23
other SMs come up idle in a wave across the die, left to right, and go
dark again just after the word ends.

Starts on chain_state(6) and ends on chain_state(7): the same picture,
with the carried caption.
"""
from __future__ import annotations

from manim import AnimationGroup

from scenes.chain import machine as mc
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import boundary

UP_FRAMES, DARK_FRAMES = 9, 8       # the idle wave across the 23 dark SMs, and its way back
IDLE_HOLD = 0.15                    # seconds they stay up after "twenty-four" ends, so all 23 can be seen
TAIL = 0.4                          # finish()'s tail: the slot is target_seconds() + TAIL


class OneSm(ChainScene):
    k = 6

    def ripple(self, spec: mc.Spec, frames: int) -> None:
        """machine_to(spec), the SMs' changes swept across the die column by
        column, left to right (GPC 0 first), as r01 does."""
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
        one = boundary(self.k + 1).machine          # the machine holds: SM 0, its shelf, fence and spill

        # ---- "twice": the speed-up, in as the word starts (the carried "spill" goes out first)
        self.say(f"{self.der['big_1block_x']:.2f}×", "vs the CPU, kernel time", at=self.cue("twice") - fr(2))

        # ---- "one SM of twenty-four": the caption on "SM"; on "twenty-four" the
        # other 23 SMs come up idle in a wave, hold a moment after the word, and go
        # dark again (the wave back always ends a few frames inside the slot)
        self.say_last(at=self.cue("SM") - fr(2))
        self.until(self.cue("twenty-four"))
        self.ripple(mc.with_rooms(one, "idle", rooms=range(1, 24)), frames=UP_FRAMES)
        self.until(min(self.cue("twenty-four", end=True) + IDLE_HOLD,
                       self.target_seconds() + TAIL - fr(DARK_FRAMES + 3)))
        self.ripple(one, frames=DARK_FRAMES)
