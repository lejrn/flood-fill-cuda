"""One geometry, two looks: every part of the chain cut is a set of 3D
boxes, and the look is only the projection.

- `flat`: straight down. Only top faces show; it reads as a diagram.
- `iso`: the camera turned by `yaw` and raised to `elev` degrees. Top,
  front and side faces show, shaded, so the chip reads as an object.

World axes: x to the right, y away from the viewer, z up. The projection
is orthographic, so a rectangle on a plane stays a parallelogram and an
image on the floor needs only an affine warp (`plane_corners`).
Choose the look with the environment variable LOOK=flat|iso.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from manim import Polygon, VGroup, VMobject
from manim.utils.color import ManimColor

LOOKS = {"flat": (0.0, 90.0), "iso": (26.0, 40.0)}     # (yaw, elevation) in degrees


def look() -> str:
    name = os.environ.get("LOOK", "flat").lower()
    if name not in LOOKS:
        raise SystemExit(f"LOOK must be one of {sorted(LOOKS)}, not {name!r}")
    return name


@dataclass(frozen=True)
class Proj:
    yaw: float
    elev: float

    @property
    def M(self) -> np.ndarray:
        """3 x 2 matrix: world (x, y, z) -> screen (X, Y), before placement."""
        t, e = np.radians(self.yaw), np.radians(self.elev)
        return np.array([
            [np.cos(t), np.sin(t) * np.sin(e)],
            [-np.sin(t), np.cos(t) * np.sin(e)],
            [0.0, np.cos(e)],
        ])

    @property
    def view(self) -> np.ndarray:
        """Unit vector from the scene towards the camera, in world axes."""
        t, e = np.radians(self.yaw), np.radians(self.elev)
        vr = np.array([0.0, -np.cos(e), np.sin(e)])          # in the yawed frame
        rot = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])
        return rot @ vr

    def depth(self, p) -> float:
        """Larger = farther from the camera."""
        return -float(np.dot(np.asarray(p, float), self.view))

    @property
    def flat(self) -> bool:
        return self.elev >= 89.9


def proj(name: str | None = None) -> Proj:
    return Proj(*LOOKS[name or look()])


@dataclass(frozen=True)
class Box:
    """An axis-aligned box: min corner (x, y, z) and size (w, d, h)."""
    x: float
    y: float
    z: float
    w: float
    d: float
    h: float
    color: str
    layer: int = 1
    stroke: str | None = None
    stroke_width: float = 0.0
    opacity: float = 1.0

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.x + self.w / 2, self.y + self.d / 2, self.z + self.h / 2])


def _shade(color: str, k: float) -> str:
    c = ManimColor(color).to_rgb()
    return ManimColor.from_rgb(tuple(float(v) * k for v in c)).to_hex()


@dataclass
class Placement:
    """World -> frame: screen = origin + scale * (world @ M)."""
    P: Proj
    scale: float
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def to_frame(self, pts) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, float))
        xy = self.origin + self.scale * (pts @ self.P.M)
        return np.column_stack([xy, np.zeros(len(xy))])

    @staticmethod
    def fit(P: Proj, lo, hi, centre, size, pad: float = 0.0) -> "Placement":
        """Scale and place the world box [lo, hi] to fit `size` (w, h) around `centre`."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        xy = corners @ P.M
        span = xy.max(axis=0) - xy.min(axis=0)
        s = min((size[0] - pad) / span[0], (size[1] - pad) / span[1])
        mid = (xy.max(axis=0) + xy.min(axis=0)) / 2
        return Placement(P, s, np.asarray(centre[:2], float) - s * mid)


FACES = {
    # name: (normal, corner index order on the unit cube), shade factor
    "top": ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 1.0),
    "front": ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], 0.62),
    "back": ((0, 1, 0), [(1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)], 0.5),
    "right": ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], 0.78),
    "left": ((-1, 0, 0), [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)], 0.78),
}


def box_faces(b: Box, pl: Placement) -> list:
    out = []
    for name, (n, idx, k) in FACES.items():
        if float(np.dot(n, pl.P.view)) <= 1e-6:
            continue
        pts = np.array([[b.x + i * b.w, b.y + j * b.d, b.z + kz * b.h] for i, j, kz in idx])
        poly = Polygon(*pl.to_frame(pts), fill_color=_shade(b.color, k), fill_opacity=b.opacity,
                       stroke_color=b.stroke or _shade(b.color, k), stroke_width=b.stroke_width if b.stroke else 0)
        poly.set_stroke(opacity=b.opacity if b.stroke else 0)
        out.append(poly)
    return out


def draw(boxes, pl: Placement) -> VGroup:
    """Painter's order: by layer, then far to near within a layer."""
    order = sorted(boxes, key=lambda b: (b.layer, -pl.P.depth(b.centre)))
    g = VGroup()
    for b in order:
        g.add(*box_faces(b, pl))
    return g


def on_plane(mob: VMobject, pl: Placement, centre_xy, z: float, width: float | None = None,
             height: float | None = None) -> VMobject:
    """Lay a 2D mobject (built around the origin, in world units) flat on the
    plane at height z, centred on centre_xy. Text then reads as printed on
    the part; in the flat look this is a plain move."""
    if width is not None:
        mob.scale_to_fit_width(width)
    if height is not None:
        mob.scale(height / mob.height)
    mob.move_to(np.zeros(3))
    cx, cy = centre_xy

    def f(p):
        return pl.to_frame([[cx + p[0], cy + p[1], z]])[0]

    mob.apply_function(f)
    return mob


def plane_corners(pl: Placement, x0: float, y0: float, x1: float, y1: float, z: float) -> np.ndarray:
    """Frame coordinates of a rectangle on the plane z: (ul, ur, dl, dr) as
    seen in the world (ul = x0, y1 = back-left)."""
    return pl.to_frame([[x0, y1, z], [x1, y1, z], [x0, y0, z], [x1, y0, z]])
