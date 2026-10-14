"""r00: freeze the footage and zoom into one drone until its pixels show.

The floor appears holding the camera frame (frame 216), a square crop
with the chosen drone near its top left. At "Freeze" a two-frame white
flash; the camera frame dissolves into the filter's mask and a thin red
box marks the drone; the "you are here" inset (one line) fades in while
they hold. At "Zoom" a nearest-neighbour push-in shrinks the crop from
the frame's full width to the toy's 18 px window, drifting so the drone
ends centred, and the last crop blends into the toy floor ('drone_red'):
white pixels become red cells, grid lines fade in. The caption (the
pixel count, no sub) comes at "pixels", at least MIN_DWELL after the
inset; nothing with text changes after it, so the next beat's boards
come in on a quiet screen.

The camera frame and the mask are light-on-dark, so `fl.picture_raster`
(made for red on white) is not used: `crop_raster` here samples a box
of the source with PIL and maps the mask's black to the empty floor.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain import toy
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import DRONE_N, Z_FLOOR, boundary, floor_raster
from scenes.style import ASSETS, RED_PX

from manim import FadeIn


@lru_cache(maxsize=None)
def _source(which: str) -> Image.Image:
    folder = {"camera": "drones_sky", "mask": "drones_mask"}[which]
    return Image.open(ASSETS / folder / f"frame_{toy.DRONE_FRAME:03d}.png").convert("RGB")


def crop_raster(which: str, x0: float, y0: float, side: float, rpx: int) -> np.ndarray:
    """(rpx, rpx, 4) RGBA: the source box [x0, x0 + side) x [y0, y0 + side)
    (pixel units, may be fractional), nearest-neighbour, so every source
    pixel is a sharp square. The mask's black becomes the empty floor."""
    im = _source(which).transform((rpx, rpx), Image.EXTENT, (x0, y0, x0 + side, y0 + side),
                                  resample=Image.NEAREST, fillcolor=(0, 0, 0))
    a = np.asarray(im).astype(np.float32)
    if which == "mask":
        w = a.mean(axis=2, keepdims=True) / 255.0
        a = fl.rgb(fl.EMPTY).astype(np.float32) * (1 - w) + 255.0 * w
    out = np.empty((rpx, rpx, 4), np.uint8)
    out[..., :3] = np.clip(a, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def box_outline(r: np.ndarray, x0: float, y0: float, side: float, alpha: float) -> np.ndarray:
    """Draw the drone's box (ov.DRONE_BOX, padded by 2 px) on raster r in red."""
    if alpha <= 0:
        return r
    rpx = r.shape[0]
    k = rpx / side
    bx0, by0, bx1, by1 = ov.DRONE_BOX
    X0, Y0 = int((bx0 - 2 - x0) * k), int((by0 - 2 - y0) * k)
    X1, Y1 = int((bx1 + 3 - x0) * k), int((by1 + 3 - y0) * k)
    t = max(2, rpx // 160)
    out = r.copy()
    red = fl.rgb(RED_PX).astype(np.float32)

    def paint(ys, xs):
        ys = slice(max(ys.start, 0), max(min(ys.stop, rpx), 0))
        xs = slice(max(xs.start, 0), max(min(xs.stop, rpx), 0))
        seg = out[ys, xs, :3].astype(np.float32)
        out[ys, xs, :3] = (seg * (1 - alpha) + red * alpha).astype(np.uint8)

    paint(slice(Y0 - t, Y0), slice(X0 - t, X1 + t))
    paint(slice(Y1, Y1 + t), slice(X0 - t, X1 + t))
    paint(slice(Y0, Y1), slice(X0 - t, X0))
    paint(slice(Y0, Y1), slice(X1, X1 + t))
    return out


class Zoom(ChainScene):
    k = 2

    def beat_anim(self) -> None:
        f = fl.Floor(self.Lay.floor_pl, z0=Z_FLOOR)
        rpx = f.rpx
        W, H = _source("camera").size                   # 576 x 1024
        cy, cx = toy.DRONE_CENTRE                       # (row, col)
        side0 = float(min(W, H))
        # the drone's place in the first crop (clamped inside the frame) ...
        ox0 = min(max(cx - side0 / 2, 0.0), W - side0)
        oy0 = min(max(cy - side0 / 2, 0.0), H - side0)
        fx0, fy0 = (cx - ox0) / side0, (cy - oy0) / side0
        side1 = float(DRONE_N)                          # ... and the toy's window, centred

        def view(q: float) -> tuple:
            """Crop box at push-in progress q: the side shrinks geometrically,
            the drone drifts from its spot in the frame to the centre."""
            s = side0 * (side1 / side0) ** q
            fx, fy = fx0 + (0.5 - fx0) * q, fy0 + (0.5 - fy0) * q
            return cx - fx * s, cy - fy * s, s

        first = crop_raster("camera", ox0, oy0, side0, rpx)
        f.show_raster(first)
        self.set_floors([f])
        self.play(FadeIn(f), run_time=fr(2))           # short, so the flash lands on "Freeze"

        # one clock for the whole floor: flash, dissolve, push-in, become the toy
        t_flash = max(self.cue("Freeze"), self.elapsed())
        t_dis = t_flash + 2 / 15
        t_mask = t_dis + 6 / 15                         # the mask and the box, held ...
        t_push = max(self.cue("Zoom"), t_mask)          # ... until "Zoom" starts the push-in
        t_toy = max(self.cue("show"), t_push + 1.4)    # the 18 px window by "show" ...
                                                        # ... and the red cells by its end
        t_end = t_toy + 7 / 15
        white = np.full_like(first, 255)
        mask0 = crop_raster("mask", ox0, oy0, side0, rpx)
        goal = floor_raster("drone_red", rpx)
        x1, y1, s1 = view(1.0)
        last = crop_raster("mask", x1, y1, s1, rpx).astype(np.float32)

        def ease(u: float) -> float:
            u = min(max(u, 0.0), 1.0)
            return u * u * (3 - 2 * u)

        def key(p: float):
            t = t0 + p * (t_end - t0)
            return int(round(t * 15)) if p < 1 else "end"

        def render(k):
            if k == "end":
                return goal
            t = k / 15
            if t < t_flash:
                return first
            if t < t_dis:
                return (first.astype(np.float32) * 0.25 + white.astype(np.float32) * 0.75).astype(np.uint8)
            if t < t_push:
                a = ease((t - t_dis) / (t_mask - t_dis))
                r = (first.astype(np.float32) * (1 - a) + mask0.astype(np.float32) * a).astype(np.uint8)
                return box_outline(r, ox0, oy0, side0, a)
            if t < t_toy:
                q = ease((t - t_push) / (t_toy - t_push))
                x, y, s = view(q)
                r = crop_raster("mask", x, y, s, rpx)
                return box_outline(r, x, y, s, 1.0 - min(1.0, q / 0.6))
            a = ease((t - t_toy) / (t_end - t_toy))
            return (last * (1 - a) + goal.astype(np.float32) * a).astype(np.uint8)

        t0 = self.elapsed()
        self.floor_play(key, render, t_end - t0)
        # the inset while the mask and its box hold, before "Zoom": its line is
        # text too, so it keeps MIN_DWELL clear of the caption, and the beat's
        # last text change stays clear of r01's boards (they slide in at "CPU")
        self.until(t_mask)
        b = boundary(self.k + 1)
        self.over_to(inset=ov.inset(self.G, b.inset), frames=6)
        self.say_last(at=max(self.cue("pixels"), self.elapsed() + MIN_DWELL))
