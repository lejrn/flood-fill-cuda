"""A small isometric renderer: a grid of pixels drawn as boxes on a floor.

The explainer video's 3D look, for the page's clips. Every pixel of a
small scene is one box with a dark frame, so single pixels can be told
apart; filled pixels stand taller than unfilled ones, and a pixel that
was just filled pops up before it settles.

The camera is the video's (video/scenes/chain/iso.py): orthographic,
turned 26 degrees and raised 40. World axes: x right, y away from the
viewer, z up. An array is indexed [x, y] like the kernels' arrays, x the
image column and y the image row, and row 0 is drawn at the far side, so
the picture reads the way the image does.

Pure numpy and Pillow. Boxes are painted far to near.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

YAW, ELEV = 26.0, 40.0

BG = (11, 15, 20)              # the video's background
FLOOR = (22, 28, 36)           # an empty cell
SLAB = (17, 22, 29)
FRAME = (6, 8, 11)             # the dark frame round every pixel
RED_PX = (226, 100, 90)        # an unfilled red pixel
INK = (230, 237, 243)
INK_SOFT = (173, 186, 199)
POP = (236, 244, 255)          # a pixel in the step it is filled

FPS = 30


def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf"])
    for root in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
                 "/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype"):
        for name in names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    sys.exit(f"none of {names} found under /usr/share/fonts/truetype")


@dataclass
class Box:
    """An extra box on the floor, in cell units: footprint and height."""
    x: float
    y: float
    w: float
    d: float
    h: float
    colour: tuple


class Floor:
    """An nx by ny grid of cells, fitted into a width x height panel."""

    def __init__(self, nx: int, ny: int, width: int, height: int, max_h: float = 1.9,
                 ss: int = 2, pad: int = 18, frame_px: float = 1.0, slab: float = 0.8):
        self.nx, self.ny, self.w, self.h, self.ss = nx, ny, width, height, ss
        self.frame = max(1, round(frame_px * ss))
        self.slab = slab
        t, e = np.radians(YAW), np.radians(ELEV)
        # world (x, y, z) -> screen (X right, Y up)
        self.M = np.array([[np.cos(t), np.sin(t) * np.sin(e)],
                           [-np.sin(t), np.cos(t) * np.sin(e)],
                           [0.0, np.cos(e)]])
        rot = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])
        self.view = rot @ np.array([0.0, -np.cos(e), np.sin(e)])
        corners = np.array([[x, y, z] for x in (0, nx) for y in (0, ny) for z in (-slab, max_h)], float)
        scr = corners @ self.M
        lo, hi = scr.min(axis=0), scr.max(axis=0)
        W, H = width * ss, height * ss
        self.scale = min((W - 2 * pad * ss) / (hi[0] - lo[0]), (H - 2 * pad * ss) / (hi[1] - lo[1]))
        mid = (lo + hi) / 2
        self.off = np.array([W / 2 - mid[0] * self.scale, H / 2 + mid[1] * self.scale])
        # paint order: far cells first
        cx, cy = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="ij")
        depth = -(cx * self.view[0] + cy * self.view[1])
        self.order = np.argsort(-depth, axis=None, kind="stable")
        # which side faces look at the camera
        self.side_x = 1.0 if self.view[0] > 0 else 0.0       # the +x or the -x face
        self.side_y = 1.0 if self.view[1] > 0 else 0.0
        self._base = None                                    # the empty floor, drawn once

    def pt(self, x: float, y: float, z: float) -> tuple[float, float]:
        s = np.array([x, y, z]) @ self.M
        return (self.off[0] + s[0] * self.scale, self.off[1] - s[1] * self.scale)

    def _box(self, draw, x, y, w, d, h, colour, frame=True):
        c = np.asarray(colour, float)
        top = tuple(int(v) for v in c)
        outline = FRAME if frame else None
        if h > 1e-6:
            xs = x + w * self.side_x
            ys = y + d * self.side_y
            side = tuple(int(v) for v in c * 0.58)
            front = tuple(int(v) for v in c * 0.74)
            draw.polygon([self.pt(xs, y, 0), self.pt(xs, y + d, 0), self.pt(xs, y + d, h), self.pt(xs, y, h)],
                         fill=side, outline=outline, width=self.frame)
            draw.polygon([self.pt(x, ys, 0), self.pt(x + w, ys, 0), self.pt(x + w, ys, h), self.pt(x, ys, h)],
                         fill=front, outline=outline, width=self.frame)
        draw.polygon([self.pt(x, y, h), self.pt(x + w, y, h), self.pt(x + w, y + d, h), self.pt(x, y + d, h)],
                     fill=top, outline=outline, width=self.frame)

    def _empty_floor(self) -> Image.Image:
        """The slab and every cell as a flat framed square, at the supersampled size."""
        ss = self.ss
        img = Image.new("RGB", (self.w * ss, self.h * ss), BG)
        draw = ImageDraw.Draw(img)
        xs, ys = self.nx * self.side_x, self.ny * self.side_y
        s = self.slab
        draw.polygon([self.pt(xs, 0, -s), self.pt(xs, self.ny, -s), self.pt(xs, self.ny, 0), self.pt(xs, 0, 0)],
                     fill=tuple(int(v * 0.6) for v in SLAB))
        draw.polygon([self.pt(0, ys, -s), self.pt(self.nx, ys, -s), self.pt(self.nx, ys, 0), self.pt(0, ys, 0)],
                     fill=tuple(int(v * 0.8) for v in SLAB))
        for x in range(self.nx):
            for y in range(self.ny):
                self._box(draw, x, y, 1, 1, 0.0, FLOOR)
        return img

    def render(self, colours: np.ndarray, heights: np.ndarray, extra: list[Box] | None = None,
               thick: int = 0) -> Image.Image:
        """colours [x, y, 3] uint8, heights [x, y] in cells. A cell of height
        0 is the empty floor. thick: a heavier frame every `thick` cells
        (tile borders), drawn on the floor."""
        if self._base is None:
            self._base = self._empty_floor()
        img = self._base.copy()
        draw = ImageDraw.Draw(img)
        if thick:
            lw = self.frame * 2
            for i in range(0, self.nx + 1, thick):
                draw.line([self.pt(i, 0, 0), self.pt(i, self.ny, 0)], fill=FRAME, width=lw)
            for j in range(0, self.ny + 1, thick):
                draw.line([self.pt(0, j, 0), self.pt(self.nx, j, 0)], fill=FRAME, width=lw)
        ny = self.ny
        boxes = [(int(k), None) for k in self.order]
        for b in extra or []:
            # an extra box paints with the cell under its centre
            boxes.append((None, b))
        raised = heights > 0
        for k, _ in boxes[: len(self.order)]:
            x, yw = divmod(k, ny)
            y = ny - 1 - yw                      # world row yw is image row y
            if raised[x, y]:
                self._box(draw, x, yw, 1, 1, float(heights[x, y]), colours[x, y])
        for b in sorted(extra or [], key=lambda b: (b.x + b.w / 2) * self.view[0]
                        + (self.ny - b.y - b.d / 2) * self.view[1]):
            self._box(draw, b.x, self.ny - b.y - b.d, b.w, b.d, b.h, b.colour)
        return img.resize((self.w, self.h), Image.LANCZOS)


def label(img: Image.Image, title: str, note: str = "", at: tuple[int, int] = (18, 14),
          size: int = 22) -> Image.Image:
    """A bold title and a soft note after it, drawn onto the panel."""
    draw = ImageDraw.Draw(img)
    f1, f2 = _font(True, size), _font(False, size)
    draw.text(at, title, fill=INK, font=f1)
    if note:
        draw.text((at[0] + round(draw.textlength(title, font=f1)) + 12, at[1]), note, fill=INK_SOFT, font=f2)
    return img


def hstack(panels: list[Image.Image], gap: int = 0) -> Image.Image:
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (w, max(p.height for p in panels)), BG)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def lerp(a, b, t: float) -> np.ndarray:
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def wave_frame(floor: Floor, red: np.ndarray, when: np.ndarray, final: np.ndarray, t: float,
               base: float = 0.45, settled: float = 0.9, pop: float = 0.9, extra=None, thick: int = 0):
    """One frame of a fill. red [x, y] bool; when [x, y] the step a pixel is
    filled at (inf: never); final [x, y, 3] its colour once filled; t the
    clock in steps. A pixel pops in the step it is filled, then settles."""
    age = t - when
    filled = red & (age >= 0)
    colours = np.empty(red.shape + (3,), np.uint8)
    colours[:] = FLOOR
    colours[red] = RED_PX
    heights = np.where(red, base, 0.0)
    if filled.any():
        a = np.clip(age[filled] / 1.6, 0.0, 1.0)[:, None]
        colours[filled] = (np.asarray(POP, float) * (1 - a) + final[filled].astype(float) * a).astype(np.uint8)
        heights[filled] = settled + pop * (1 - a[:, 0]) ** 2
    return floor.render(colours, heights, extra=extra, thick=thick)


def encode(frames, out: Path, ffmpeg: str = "ffmpeg", crf: int = 23) -> None:
    """frames: an iterable of (PIL image, how many video frames to hold it)."""
    it = iter(frames)
    first, hold = next(it)
    w, h = first.size
    assert w % 2 == 0 and h % 2 == 0, (w, h)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-crf", str(crf), "-movflags", "+faststart",
           "-map_metadata", "-1", "-an", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    image = first
    while True:
        data = image.tobytes()
        for _ in range(hold):
            proc.stdin.write(data)
        n += hold
        try:
            image, hold = next(it)
        except StopIteration:
            break
    proc.stdin.close()
    if proc.wait():
        sys.exit(f"ffmpeg failed on {out}")
    print(f"{out}: {w}x{h}, {n} frames = {n / FPS:.2f} s, {out.stat().st_size:,} bytes")


def poster(image: Image.Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, quality=88, method=6)
    print(f"{out}: {image.width}x{image.height}, {out.stat().st_size:,} bytes")
