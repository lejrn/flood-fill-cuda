"""The floor: the toy picture (or the real one) where the work happens.

The floor is one image. A frame of it is an RGBA raster computed straight
at the output pixel size from per-cell colours (no resampling, so grid
lines stay one pixel wide). In the iso look the plane is tilted, which
Manim's Cairo camera cannot draw for images (it ignores shear), so the
raster is warped here with one affine transform and handed over upright.

`Floor.show(colors, dots, ring)` swaps the image's pixel_array in place,
like `stage.Replay`: the Cairo renderer snapshots the mobject family
when a play starts, so the image itself must stay the same object.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from manim import Group, ImageMobject, VGroup, config
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image

from scenes.chain.iso import Box, Placement, box_faces, plane_corners
from scenes.style import BG, BLUE, GOLD, GRID, INK, RED_PX

N = 24                      # toy grid size
HALF = 3.0                  # the floor plane is [-3, 3] x [-3, 3] at z = 0
SLAB = 0.18
FLOOR_LO, FLOOR_HI = (-HALF, -HALF, -SLAB), (HALF, HALF, 0.0)
EMPTY = "#161c24"           # an empty cell
SLAB_COL = "#11161d"


def rgb(hex_: str) -> np.ndarray:
    h = hex_.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def ppu() -> float:
    """Output pixels per frame unit."""
    return config.pixel_width / config.frame_width


def placement(P, centre, size) -> Placement:
    return Placement.fit(P, FLOOR_LO, FLOOR_HI, centre, size)


@lru_cache(maxsize=8)
def _index_maps(rpx: int, n: int):
    """Per output pixel: the cell index, and the offset from the cell centre in cells."""
    pos = (np.arange(rpx) + 0.5) * n / rpx
    idx = np.minimum(pos.astype(int), n - 1)
    off = pos - idx - 0.5
    edge = np.zeros(rpx, bool)
    edge[1:] = idx[1:] != idx[:-1]
    edge[0] = True
    edge[-1] = True
    return idx, off, edge


def toy_raster(colors: np.ndarray, rpx: int, dots: np.ndarray | None = None,
               ring: np.ndarray | None = None, boxes=()) -> np.ndarray:
    """(rpx, rpx, 4) uint8 from (n, n, 3) uint8 cell colours.

    dots: (n, n) int, 0 none, 1 a working worker (blue), 2 a waiting one (amber),
          3 a seed (white).
    ring: (n, n) bool, cells outlined in white (the wave's edge).
    boxes: (r, c0, c1) spans drawn as one bar: inner grid lines removed, a
           white border around the whole span (a ch06 run).
    """
    n = colors.shape[0]
    idx, off, edge = _index_maps(rpx, n)
    out = np.empty((rpx, rpx, 4), np.uint8)
    out[..., :3] = colors[idx][:, idx]
    out[..., 3] = 255
    grid = rgb(GRID)
    out[edge, :, :3] = grid
    out[:, edge, :3] = grid
    if ring is not None and ring.any():
        rr = ring[idx][:, idx]
        r_off = np.abs(off)
        near = (r_off[:, None] > 0.36) | (r_off[None, :] > 0.36)
        out[rr & near, :3] = rgb(INK)
    if boxes:
        cell = rpx / n
        white = rgb(INK)
        for r, c0, c1 in boxes:
            y0, y1 = int(round(r * cell)), int(round((r + 1) * cell))
            x0, x1 = int(round(c0 * cell)), int(round((c1 + 1) * cell))
            out[y0 + 1:y1 - 1, x0 + 1:x1 - 1, :3] = colors[r, c0]
            t = max(2, int(cell * 0.09))
            out[y0:y0 + t, x0:x1, :3] = white
            out[y1 - t:y1, x0:x1, :3] = white
            out[y0:y1, x0:x0 + t, :3] = white
            out[y0:y1, x1 - t:x1, :3] = white
    if dots is not None and dots.any():
        d = dots[idx][:, idx]
        dist = off[:, None] ** 2 + off[None, :] ** 2
        inside = dist < 0.27 ** 2
        out[(d == 1) & inside, :3] = rgb("#cfe3ff")
        out[(d == 2) & inside, :3] = rgb(GOLD)
        out[(d == 3) & (dist < 0.2 ** 2), :3] = rgb(INK)
    return out


def picture_raster(rgb_img: np.ndarray, rpx: int, nearest: bool = False) -> np.ndarray:
    """A real picture (white background) on the dark floor, resized to rpx
    (nearest: every source pixel a sharp square, for zooms)."""
    src = rgb_img.copy()
    red = (src[..., 0] > 200) & (src[..., 1] < 90) & (src[..., 2] < 90)
    src[red] = rgb(RED_PX)                       # the toy's red, not (255, 0, 0)
    im = Image.fromarray(src).resize((rpx, rpx), Image.NEAREST if nearest else Image.BILINEAR)
    a = np.asarray(im).astype(np.float32)
    white = a.min(axis=2) / 255.0                 # 1 on the white background
    bg = rgb(EMPTY).astype(np.float32)
    out = np.empty((rpx, rpx, 4), np.uint8)
    out[..., :3] = np.clip(a * (1 - white[..., None]) + bg * white[..., None], 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


@dataclass
class Warp:
    """Raster (rpx x rpx, the plane seen from straight above) -> the upright
    image Manim draws, plus where that image sits in the frame."""
    rpx: int
    size: tuple          # output (w, h) in pixels
    centre: np.ndarray   # frame centre of the output image
    coeffs: tuple | None # PIL affine coefficients, None = no warp needed

    @staticmethod
    def make(pl: Placement) -> "Warp":
        ul, ur, dl, _ = plane_corners(pl, -HALF, -HALF, HALF, HALF, 0.0)
        pts = np.array([ul, ur, dl, ur + dl - ul])[:, :2]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        k = ppu()
        w, h = int(np.ceil((hi[0] - lo[0]) * k)), int(np.ceil((hi[1] - lo[1]) * k))
        centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
        if pl.P.flat:
            return Warp(w, (w, w), centre, None)
        rpx = int(max(w, h) * 1.15)
        A = np.column_stack([(ur - ul)[:2] / rpx, (dl - ul)[:2] / rpx])
        Ai = np.linalg.inv(A)
        # output pixel (i, j) -> frame point -> raster (u, v)
        x0 = lo[0] - ul[0] + 0.5 / k
        y0 = hi[1] - ul[1] - 0.5 / k
        a, b = Ai[0, 0] / k, -Ai[0, 1] / k
        c = Ai[0, 0] * x0 + Ai[0, 1] * y0
        d, e = Ai[1, 0] / k, -Ai[1, 1] / k
        f = Ai[1, 0] * x0 + Ai[1, 1] * y0
        return Warp(rpx, (w, h), centre, (a, b, c, d, e, f))

    def apply(self, raster: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            return raster
        im = Image.fromarray(raster, "RGBA").transform(self.size, Image.AFFINE, self.coeffs,
                                                       resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))
        return np.asarray(im).copy()


class Floor(Group):
    """The floor's slab (iso only) and its image. `show_raster` swaps frames."""

    def __init__(self, pl: Placement, raster: np.ndarray | None = None, z0: float = 10.0):
        super().__init__()
        self.pl = pl
        self.warp = Warp.make(pl)
        self.slab = VGroup()
        if not pl.P.flat:
            b = Box(-HALF, -HALF, -SLAB, 2 * HALF, 2 * HALF, SLAB, SLAB_COL, layer=0)
            for i, face in enumerate(box_faces(b, pl)):
                face.set_z_index(z0 + i * 0.01)
                self.slab.add(face)
            self.add(self.slab)
        if raster is None:
            raster = toy_raster(np.tile(rgb(EMPTY), (N, N, 1)), self.warp.rpx)
        arr = self.warp.apply(raster)
        self.image = ImageMobject(arr)
        self.image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest" if pl.P.flat else "bilinear"])
        k = ppu()
        # +0.5 px: the camera truncates the drawn size, so this lands on exactly w x h
        self.image.stretch_to_fit_width((self.warp.size[0] + 0.5) / k)
        self.image.stretch_to_fit_height((self.warp.size[1] + 0.5) / k)
        self.image.move_to(self.warp.centre)
        self.image.set_z_index(z0 + 1)
        self.add(self.image)

    @property
    def rpx(self) -> int:
        return self.warp.rpx

    def show_raster(self, raster: np.ndarray) -> "Floor":
        src = self.warp.apply(raster)
        img = self.image
        img.orig_alpha_pixel_array = src[:, :, 3].copy()
        arr = src.copy()
        if img.stroke_opacity < 1:          # keep a fade in progress
            arr[:, :, 3] = (img.orig_alpha_pixel_array * img.stroke_opacity).astype(arr.dtype)
        img.pixel_array = arr
        return self

    def cell_centre(self, r: int, c: int, z: float = 0.0) -> np.ndarray:
        """Frame position of toy cell (r, c), for labels and arrows."""
        cell = 2 * HALF / N
        x = -HALF + (c + 0.5) * cell
        y = HALF - (r + 0.5) * cell
        return self.pl.to_frame([[x, y, z]])[0]

    def edge_point(self, which: str) -> np.ndarray:
        """Frame position of the middle of the floor's bottom or top edge."""
        y = -HALF if which == "bottom" else HALF
        return self.pl.to_frame([[0.0, y, 0.0]])[0]
