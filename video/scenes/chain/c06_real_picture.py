"""c06: the twist. On the real picture every blob is its own launch, with
its own rings and gates, and the machine loses to the CPU."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from scenes.chain import floor as fl
from scenes.chain.base import ChainScene, fr
from scenes.chain.c04_gate import amber_overlay
from scenes.chain.state import (
    GOLD, LEVEL_HI, filled_picture, fmt_ms, level_colors, machine_spec, picture, picture_blobs, toys,
    zoom_crop, zoom_target,
)


class RealPicture(ChainScene):
    k = 6

    def beat_anim(self) -> None:
        t = toys()
        n = self.num
        red = picture("red")
        r, c, side = zoom_target()
        square = self.toy(level_colors(t["square"], t["square8"].level))
        crop = fl.picture_raster(zoom_crop(red, r, c, side), self.floor.rpx, nearest=True)
        steps = 20

        def zoom(i: int):
            p = i / steps
            if p <= 0.3:
                a = p / 0.3
                return (square.astype(np.float32) * (1 - a) + crop.astype(np.float32) * a).astype(np.uint8)
            q = ((p - 0.3) / 0.7) ** 0.5                     # ease out
            s = side * (900 / side) ** q
            return fl.picture_raster(zoom_crop(red, r + (450 - r) * q, c + (450 - c) * q, s), self.floor.rpx,
                                     nearest=q < 0.9)

        self.until(self.cue("feed"))
        self.floor_play(lambda p: round(p * steps), zoom, max(self.cue("loses") - self.elapsed(), 0.8))
        self.say("the real picture", f"{int(n['n_blobs']):,} blobs", at=self.cue("picture"))
        self.until(self.cue("loses"))
        self.goal_to(("~" + fmt_ms(n["c06_ms"]), GOLD))
        self.say("slower", f"~{fmt_ms(n['c06_ms'])} vs {fmt_ms(n['njit_ms'])} (estimated)", at=self.cue("CPU"))
        # one launch per blob, in scan order; the rooms wait between launches
        lab, sizes = picture_blobs()
        nb_ = len(sizes)
        hi = fl.rgb(LEVEL_HI)
        t0, t1 = self.cue("blobs"), self.cue("You")
        spec = machine_spec(6)
        amber = amber_overlay(replace(spec, gate=None), self.Lay.machine_pl)
        self.add(amber)

        def key(p: float) -> int:
            return min(nb_, int(nb_ * p))

        def render(i: int):
            if i >= nb_:
                return self.pic(filled_picture())
            img = red.copy()
            img[(lab >= 0) & (lab < i)] = hi
            return self.pic(img)

        self.until(t0)
        self.floor_play(key, render, t1 - t0)
        self.run(lambda p: amber.set_opacity(0.0 if p >= 1 else 0.55 + 0.3 * np.sin(2 * np.pi * 7 * p)), t1 - t0)
        self.until(self.cue("launch"))
        self.machine_to(machine_spec(7), frames=6)
        self.say(f"{int(n['c06_calls']):,} launches", "one per blob", at=self.cue("launch", end=True))
        self.say_last(at=self.cue("machine"))
        self.settle()
        self.remove(amber)
