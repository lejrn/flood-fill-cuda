"""c00: the job. The real picture, its numbers, the frame budget; then a
zoom into one blob, which becomes the floor's toy."""
from __future__ import annotations

from manim import FadeIn

from scenes.chain import floor as fl
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import (
    Z_FLOOR, level_colors, picture, toys, zoom_crop, zoom_target,
)
import numpy as np


class Job(ChainScene):
    k = 0

    def beat_anim(self) -> None:
        f = fl.Floor(self.Lay.floor_pl, z0=Z_FLOOR)
        red = picture("red")
        f.show_raster(fl.picture_raster(red, f.rpx))
        self.adopt_floor(f)
        self.play(FadeIn(f), run_time=fr(8))
        n = self.num
        self.say(f"{n['n_pixels'] / 1e6:.0f} Mpx", "the real picture", at=self.cue("Eighty-one"))
        self.say(f"{int(n['n_blobs']):,}", "red blobs", at=self.cue("two"))
        self.until(self.cue("colour"))
        self.goal_to(None)
        # zoom into one blob, then let it become the toy
        self.until(self.cue("camera"))
        r, c, side = zoom_target()
        t0 = self.elapsed()
        span = max(self.target_seconds() - 0.35 - t0, 0.6)
        toy_r = self.toy(level_colors(toys()["blob"], np.full(toys()["blob"].shape, -1)))
        steps = 24

        def key(p: float) -> int:
            return round(p * steps)

        def render(i: int):
            p = i / steps
            if p <= 0.7:
                q = (p / 0.7) ** 2                       # ease in
                s = 900 * (side / 900) ** q
                rr = 450 + (r - 450) * q
                cc = 450 + (c - 450) * q
                return fl.picture_raster(zoom_crop(red, rr, cc, s), f.rpx, nearest=True)
            a = (p - 0.7) / 0.3
            last = fl.picture_raster(zoom_crop(red, r, c, side), f.rpx, nearest=True)
            return (last.astype(np.float32) * (1 - a) + toy_r.astype(np.float32) * a).astype(np.uint8)

        self.floor_play(key, render, span)
        self.say_last(at=self.cue("thirty-three"))
