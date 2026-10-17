"""The floor: the toy picture (or the real one) where the work happens.

The floor is one image. A frame of it is an RGBA raster computed straight
at the output pixel size from per-cell colours (no resampling, so grid
lines stay one pixel wide). In the iso look the plane is tilted, which
Manim's Cairo camera cannot draw for images (it ignores shear), so the
raster is warped here with one affine transform and handed over upright.

`Floor.show(colors, dots, ring)` swaps the image's pixel_array in place,
like `stage.Replay`: the Cairo renderer snapshots the mobject family
when a play starts, so the image itself must stay the same object.

v5 (act 3) adds gridless rasters for real arrays (a 224 px tile context,
the 250 x 250 tiles of the square, the 384 px crop, the 900 px frame):
`Cells` says where an array sits on the floor, `cells_raster` upscales it
nearest with an optional faint tile grid, and the raster primitives draw
on top of it at the output's pixel size (outlines, discs, hatching,
segments, a border, masks). They change the raster in place and return
it, so a scene composes one frame as cells_raster(...) then primitives.
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
REACHED = "#5e89b9"         # act 3's reached pixel: the middle of the ring ramp (state.LEVEL_LO .. LEVEL_HI)
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


@lru_cache(maxsize=8)
def _fine_index(rpx: int, m: int) -> np.ndarray:
    """Per output pixel: the index on a fine grid of m points across."""
    return np.minimum(((np.arange(rpx) + 0.5) * m / rpx).astype(int), m - 1)


def toy_raster(colors: np.ndarray, rpx: int, dots: np.ndarray | None = None,
               ring: np.ndarray | None = None, boxes=(), fine: np.ndarray | None = None) -> np.ndarray:
    """(rpx, rpx, 4) uint8 from (n, n, 3) uint8 cell colours.

    fine: (n * k, n * k, 3) uint8: a k x k mosaic inside each cell (the owner
          maps: a dot per real pixel sampled); it replaces `colors` as the
          fill, the cell grid stays.

    dots: (n, n) int, 0 none, 1 a working worker (blue), 2 a waiting one (amber),
          3 a seed (white).
    ring: (n, n) bool, cells outlined in white (the wave's edge).
    boxes: (r, c0, c1) spans drawn as one bar: inner grid lines removed, a
           white border around the whole span (a ch06 run).
    """
    n = colors.shape[0]
    idx, off, edge = _index_maps(rpx, n)
    out = np.empty((rpx, rpx, 4), np.uint8)
    if fine is None:
        out[..., :3] = colors[idx][:, idx]
    else:
        fi = _fine_index(rpx, fine.shape[0])
        out[..., :3] = fine[fi][:, fi]
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


# ---- gridless rasters and raster primitives (v5, act 3) -------------------

@dataclass(frozen=True)
class Cells:
    """Where an array of cells sits on a floor raster (rpx x rpx): `shape`
    (rows, cols) laid into `box` = (top, left, bottom, right), fractions of
    the floor's side, row 0 at the floor's back edge (as toy_raster). The
    act-3 arrays are [x, y] and drop in as (row, col): a lane is a row."""
    rpx: int
    shape: tuple
    box: tuple = (0.0, 0.0, 1.0, 1.0)

    def maps(self) -> tuple:
        """(ri, ci): the cell index under each output row and column, -1
        outside the box."""
        return _cell_maps(self.rpx, int(self.shape[0]), int(self.shape[1]), tuple(float(v) for v in self.box))

    def px(self, r, c) -> tuple:
        """Output pixel (y, x) of the cell coordinate (r, c), in cells: (r + 0.5,
        c + 0.5) is cell (r, c)'s centre, (r, c) its top-left corner."""
        t, l, b, rr = self.box
        rows, cols = self.shape
        y = (t + np.asarray(r, float) / rows * (b - t)) * self.rpx
        x = (l + np.asarray(c, float) / cols * (rr - l)) * self.rpx
        return y, x

    def frac(self, r, c) -> tuple:
        """The same point as a fraction of the floor's side (Floor.frac_point)."""
        y, x = self.px(r, c)
        return y / self.rpx, x / self.rpx

    @property
    def cell_px(self) -> float:
        """Output pixels per cell (along the rows)."""
        return (self.box[2] - self.box[0]) * self.rpx / self.shape[0]


@lru_cache(maxsize=16)
def _cell_maps(rpx: int, rows: int, cols: int, box: tuple) -> tuple:
    t, l, b, r = box
    pos = (np.arange(rpx) + 0.5) / rpx

    def idx(n: int, a: float, z: float) -> np.ndarray:
        i = np.floor((pos - a) / (z - a) * n).astype(int)
        i[(pos < a) | (pos >= z)] = -1
        return np.minimum(i, n - 1)

    return idx(rows, t, b), idx(cols, l, r)


def _blend(raster: np.ndarray, sel: np.ndarray, color: str, alpha) -> None:
    """raster[sel] towards `color` by `alpha` (a number, or one value per selected pixel)."""
    if not np.any(sel):
        return
    c = rgb(color).astype(np.float32)
    a = np.asarray(alpha, np.float32)
    if a.ndim:
        a = a[..., None]
    px = raster[sel, :3].astype(np.float32)
    raster[sel, :3] = np.clip(px + (c - px) * a, 0, 255).astype(np.uint8)


def cells_raster(colors: np.ndarray, cells: Cells, bg: str = EMPTY, grid: int = 0, grid_color: str = GRID,
                 grid_alpha: float = 0.6) -> np.ndarray:
    """(rpx, rpx, 4) uint8: `colors` ((rows, cols, 3) uint8) upscaled nearest
    into cells.box, `bg` around it, alpha 255 everywhere (no keying: an
    array's light colours stay light). grid > 0: a faint 1 px line every
    `grid` cells (a tile grid: 32 on a pixel array), none on the box's edge."""
    ri, ci = cells.maps()
    out = np.empty((cells.rpx, cells.rpx, 4), np.uint8)
    out[..., :3] = rgb(bg)
    out[..., 3] = 255
    rs, cs = ri >= 0, ci >= 0
    out[np.ix_(rs, cs)] = np.concatenate(
        [colors[ri[rs]][:, ci[cs]], np.full((rs.sum(), cs.sum(), 1), 255, np.uint8)], axis=2)
    if grid:
        def lines(i: np.ndarray) -> np.ndarray:
            e = np.zeros(len(i), bool)
            e[1:] = (i[1:] != i[:-1]) & (i[1:] > 0) & (i[1:] % grid == 0)
            return e
        er, ec = lines(ri), lines(ci)
        sel = np.zeros((cells.rpx, cells.rpx), bool)
        sel[np.ix_(er, cs)] = True
        sel[np.ix_(rs, ec)] = True
        _blend(out, sel, grid_color, grid_alpha)
    return out


def paint(raster: np.ndarray, cells: Cells, mask: np.ndarray, color: str, alpha=1.0) -> np.ndarray:
    """Every output pixel whose cell is in `mask` ((rows, cols) bool) towards
    `color` by `alpha` (a number, or a (rows, cols) float array)."""
    ri, ci = cells.maps()
    rs, cs = ri >= 0, ci >= 0
    big = np.zeros((cells.rpx, cells.rpx), bool)
    big[np.ix_(rs, cs)] = mask[ri[rs]][:, ci[cs]]
    a = alpha
    if np.ndim(alpha):
        A = np.zeros((cells.rpx, cells.rpx), np.float32)
        A[np.ix_(rs, cs)] = np.asarray(alpha, np.float32)[ri[rs]][:, ci[cs]]
        a = A[big]
    _blend(raster, big, color, a)
    return raster


def outline(raster: np.ndarray, cells: Cells, ids: np.ndarray, color: str, width: int = 2,
            alpha=1.0, only=None, outer: bool = True) -> np.ndarray:
    """An outline `width` px wide just inside each region of `ids`
    ((rows, cols) int, -1 = none): a pixel is drawn when a pixel up to
    `width` away (4 directions) belongs to another id. only: the ids to
    outline (default every id >= 0). outer=False: only where two regions
    meet (cut lines between pieces), never against -1 (edges and holes)."""
    ri, ci = cells.maps()
    rs, cs = ri >= 0, ci >= 0
    C = np.full((cells.rpx, cells.rpx), -1, np.int64)
    C[np.ix_(rs, cs)] = ids[ri[rs]][:, ci[cs]]
    edge = np.zeros(C.shape, bool)

    def other(a, b):
        return (a != b) if outer else ((a != b) & (b >= 0))

    for d in range(1, max(1, int(width)) + 1):
        edge[d:] |= other(C[d:], C[:-d])
        edge[:-d] |= other(C[:-d], C[d:])
        edge[:, d:] |= other(C[:, d:], C[:, :-d])
        edge[:, :-d] |= other(C[:, :-d], C[:, d:])
    edge &= C >= 0
    if only is not None:
        edge &= np.isin(C, np.asarray(list(only)))
    _blend(raster, edge, color, alpha)
    return raster


def discs(raster: np.ndarray, cells: Cells, pts, radius: float, color: str, alpha=1.0) -> np.ndarray:
    """A filled disc of `radius` output pixels at each cell coordinate in
    `pts` ((n, 2) of (r, c); use r + 0.5, c + 0.5 for a cell's centre).
    alpha: a number or one per point; where discs overlap, the larger wins."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return raster
    y, x = cells.px(pts[:, 0], pts[:, 1])
    k = int(np.ceil(radius))
    dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
    yc, xc = np.floor(y).astype(int), np.floor(x).astype(int)
    # per stamp offset: inside the disc around the point's exact centre
    d2 = (yc[:, None] + dy.ravel()[None] + 0.5 - y[:, None]) ** 2 + (xc[:, None] + dx.ravel()[None] + 0.5 - x[:, None]) ** 2
    inside = d2 <= radius ** 2
    yy, xx = yc[:, None] + dy.ravel()[None], xc[:, None] + dx.ravel()[None]
    ok = inside & (yy >= 0) & (yy < cells.rpx) & (xx >= 0) & (xx < cells.rpx)
    a = np.broadcast_to(np.asarray(alpha, np.float32).reshape(-1, 1) if np.ndim(alpha) else
                        np.float32(alpha), ok.shape)
    A = np.zeros((cells.rpx, cells.rpx), np.float32)
    np.maximum.at(A, (yy[ok], xx[ok]), a[ok])
    sel = A > 0
    _blend(raster, sel, color, A[sel])
    return raster


def hatch(raster: np.ndarray, cells: Cells, mask: np.ndarray, color: str, alpha: float = 0.35,
          period: int = 9, width: int = 2) -> np.ndarray:
    """Faint diagonal stripes (down to the right, as the floor shows them)
    over every cell in `mask` ((rows, cols) bool; a tile mask works on its
    own Cells)."""
    ri, ci = cells.maps()
    rs, cs = ri >= 0, ci >= 0
    big = np.zeros((cells.rpx, cells.rpx), bool)
    big[np.ix_(rs, cs)] = mask[ri[rs]][:, ci[cs]]
    yy, xx = np.ogrid[:cells.rpx, :cells.rpx]
    big &= ((xx - yy) % period) < width
    _blend(raster, big, color, alpha)
    return raster


def segments(raster: np.ndarray, cells: Cells, segs, color: str, width: float = 2.0, alpha=1.0) -> np.ndarray:
    """Line segments `width` px wide between cell coordinates: segs (n, 4) of
    (r0, c0, r1, c1). alpha: a number or one per segment."""
    segs = np.asarray(segs, float).reshape(-1, 4)
    al = np.broadcast_to(np.asarray(alpha, np.float32), (len(segs),))
    h = width / 2
    A = np.zeros((cells.rpx, cells.rpx), np.float32)
    for (r0, c0, r1, c1), a in zip(segs, al):
        (y0, y1), (x0, x1) = cells.px(np.array([r0, r1]), np.array([c0, c1]))
        ya, yb = int(max(min(y0, y1) - h - 1, 0)), int(min(max(y0, y1) + h + 2, cells.rpx))
        xa, xb = int(max(min(x0, x1) - h - 1, 0)), int(min(max(x0, x1) + h + 2, cells.rpx))
        if ya >= yb or xa >= xb:
            continue
        py, px_ = np.mgrid[ya:yb, xa:xb] + 0.5
        vy, vx = y1 - y0, x1 - x0
        L2 = vy * vy + vx * vx
        t = np.clip(((py - y0) * vy + (px_ - x0) * vx) / L2, 0, 1) if L2 > 0 else 0.0
        d2 = (py - y0 - t * vy) ** 2 + (px_ - x0 - t * vx) ** 2
        sub = A[ya:yb, xa:xb]
        np.maximum(sub, np.where(d2 <= h * h, a, 0.0), out=sub)
    sel = A > 0
    _blend(raster, sel, color, A[sel])
    return raster


def border(raster: np.ndarray, cells: Cells, color: str, width: int = 2, alpha=1.0, pad: float = 0.0) -> np.ndarray:
    """A rectangle `width` px wide around cells.box, `pad` px outside it."""
    t, l, b, r = (np.array(cells.box) * cells.rpx).tolist()
    y0, x0, y1, x1 = (int(round(v)) for v in (t - pad - width, l - pad - width, b + pad, r + pad))
    sel = np.zeros((cells.rpx, cells.rpx), bool)
    Y0, X0 = max(y0, 0), max(x0, 0)
    Y1, X1 = min(y1 + width, cells.rpx), min(x1 + width, cells.rpx)
    sel[Y0:Y1, X0:X1] = True
    sel[max(y0 + width, 0):max(y1, 0), max(x0 + width, 0):max(x1, 0)] = False
    _blend(raster, sel, color, alpha)
    return raster


def key_picture(rgb_img: np.ndarray) -> np.ndarray:
    """A real picture's colours for the floor (as picture_raster draws them):
    its white background turned to the empty floor, its pure red to the
    toy's red; everything else kept."""
    src = rgb_img.astype(np.float32)
    red = (src[..., 0] > 200) & (src[..., 1] < 90) & (src[..., 2] < 90)
    src[red] = rgb(RED_PX)
    white = src.min(axis=2, keepdims=True) / 255.0
    return np.clip(src * (1 - white) + rgb(EMPTY).astype(np.float32) * white, 0, 255).astype(np.uint8)


def window_raster(rgb_img: np.ndarray, rpx: int, r0: float, c0: float, side: float, bg: str = EMPTY,
                  keyed: bool = False) -> np.ndarray:
    """(rpx, rpx, 4): the square window [r0, r0 + side) x [c0, c0 + side) of a
    picture (rows, cols), sampled nearest (every source pixel a sharp
    square), `bg` outside the picture: the frames of a zoom. keyed: the
    picture is already in floor colours (key_picture), else it is keyed here."""
    img = rgb_img if keyed else key_picture(rgb_img)
    H, W = img.shape[:2]
    pos = (np.arange(rpx) + 0.5) / rpx * side
    rr, cc = np.floor(r0 + pos).astype(int), np.floor(c0 + pos).astype(int)
    rs, cs = (rr >= 0) & (rr < H), (cc >= 0) & (cc < W)
    out = np.empty((rpx, rpx, 4), np.uint8)
    out[..., :3] = rgb(bg)
    out[..., 3] = 255
    out[np.ix_(rs, cs), :3] = img[rr[rs]][:, cc[cs]]
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

    def frac_point(self, fr_row: float, fr_col: float, z: float = 0.0) -> np.ndarray:
        """Frame position of a point given as fractions of the floor's side
        (row from the back edge, column from the left): Cells.frac's output."""
        x = -HALF + fr_col * 2 * HALF
        y = HALF - fr_row * 2 * HALF
        return self.pl.to_frame([[x, y, z]])[0]

    def edge_point(self, which: str) -> np.ndarray:
        """Frame position of the middle of the floor's bottom or top edge."""
        y = -HALF if which == "bottom" else HALF
        return self.pl.to_frame([[0.0, y, 0.0]])[0]
