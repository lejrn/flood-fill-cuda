"""Shared look and timing for every beat scene.

Rules every scene follows:
- subclass `BeatScene`, set `beat = "<beat name from narration/script.md>"`
- build with positions relative to the frame edges (see `layout()`), so the
  same scene renders in 16:9 and 9:16 (`VIDEO_LAYOUT=vertical`)
- end `construct()` with `self.finish()`, which pads the scene to the
  narration length of that beat (`out/<voice>/timing.json`, `VOICE=kokoro`
  by default) plus the inter-beat gap, then fades out. Audio is muxed by
  `build/assemble.py`, never inside a scene
- no LaTeX: use `Text`, never `MathTex`/`Tex`
- colours only from this module
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Group,
    ImageMobject,
    Scene,
    Text,
    VGroup,
    config,
)
from manim.constants import RESAMPLING_ALGORITHMS

VIDEO_DIR = Path(__file__).resolve().parents[1]
ASSETS = VIDEO_DIR / "assets"
OUT = VIDEO_DIR / "out"

# ---- palette (matches the parent repo's chart colours) ----
BG = "#0b0f14"
INK = "#e6edf3"          # primary text
INK_SOFT = "#adbac7"     # captions
GREY = "#8b949e"         # axes, idle tiles
GRID = "#2d333b"         # grid lines
RED_PX = "#e2645a"       # unfilled red pixels, CPU / ch05 series
GOLD = "#d99a2b"         # ch06 RGB contract, "hot" phases
TEAL = "#2aa198"         # ch06 packed mask, the final answer
BLUE = "#4c8fd6"         # labeling only, GPU blocks
GREEN = "#3fb950"        # second block / second blob
PURPLE = "#a371f7"

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

FPS = 30


def is_vertical() -> bool:
    return os.environ.get("VIDEO_LAYOUT", "").lower() == "vertical"


# In 9:16 (rendered with `-r 1080,1920`) Manim keeps frame_width at 14.22, so a
# unit is only 76 px instead of 135 and everything shrinks. Make the frame 8
# units wide instead: same pixels per unit as landscape, just a taller canvas.
if is_vertical():
    config.frame_width = 8.0
    config.frame_height = 8.0 * 16 / 9


def voice() -> str:
    return os.environ.get("VOICE", "kokoro")


def beat_seconds(name: str, default: float = 6.0) -> float:
    """Length of this beat's narration, from out/<voice>/timing.json."""
    p = OUT / voice() / "timing.json"
    if not p.exists():
        return default
    for row in json.loads(p.read_text(encoding="utf-8"))["beats"]:
        if row["beat"] == name:
            return float(row["seconds"])
    return default


def beat_wav(name: str) -> Path | None:
    p = OUT / voice() / f"{name}.wav"
    return p if p.exists() else None


def layout() -> dict:
    """Frame geometry so scenes can place things relative to the edges."""
    w, h = config.frame_width, config.frame_height
    m = 0.6
    return {
        "w": w, "h": h, "margin": m,
        "top": UP * (h / 2 - m), "bottom": DOWN * (h / 2 - m),
        "left": LEFT * (w / 2 - m), "right": RIGHT * (w / 2 - m),
        "vertical": is_vertical(),
    }


def label(text: str, size: int = 36, color: str = INK, mono: bool = False, **kw) -> Text:
    return Text(text, font=MONO if mono else FONT, font_size=size, color=color, **kw)


def caption(text: str, size: int = 28) -> Text:
    return label(text, size=size, color=INK_SOFT)


def fmt_ms(ms: float) -> str:
    """24083 -> '24.1 s', 1346 -> '1,346 ms', 58.51 -> '58.51 ms', 1.46 -> '1.46 ms', 0 -> '0 ms'."""
    if ms == 0:
        return "0 ms"
    if ms >= 10_000:
        return f"{ms / 1000:.1f} s"
    if ms >= 100:
        return f"{ms:,.0f} ms"
    return f"{ms:.2f} ms"


class Stopwatch(VGroup):
    """A caption plus a monospace number. `set_ms(x, color)` swaps the number."""

    def __init__(self, title: str = "time", ms: float = 0.0, size: int = 44, color: str = INK):
        super().__init__()
        self.size = size
        self.title = caption(title, size=24)
        self.value = label(fmt_ms(ms), size=size, color=color, mono=True)
        self.value.next_to(self.title, DOWN, buff=0.15)
        self.add(self.title, self.value)

    def set_ms(self, ms: float, color: str | None = None) -> "Stopwatch":
        new = label(fmt_ms(ms), size=self.size, color=color or self.value.color, mono=True)
        new.move_to(self.value)
        old = self.value
        self.remove(old)
        self.value = new
        self.add(new)
        # Cairo snapshots the mobject family when a play starts, so a Text
        # swapped inside an updater is still drawn. Blank the old glyphs
        # (the same trick DecimalNumber.set_value uses).
        for m in old.get_family():
            m.points[:] = 0
        return self


class FrameSequence(Group):
    """Plays a PNG frame sequence from assets/<name>/ like a GIF.

    Add it to the scene, then `seq.start(fps)`; frames advance during any
    `self.play`/`self.wait`. `seq.duration(fps)` tells you how long a full
    pass takes so you can `self.wait(seq.duration(fps))`.
    """

    def __init__(self, name: str, height: float, pixelated: bool = True):
        super().__init__()
        folder = ASSETS / name
        paths = sorted(folder.glob("frame_*.png"))
        if not paths:
            raise FileNotFoundError(f"run assets/extract_gifs.py first: {folder}")
        # Frame store. These are never added to the scene: the Cairo renderer
        # snapshots the mobject family when a play/wait starts, so swapping
        # submobjects leaves the old frame drawn on top. Instead one
        # ImageMobject stays in the scene and its pixel_array is swapped.
        self.frames = [ImageMobject(str(p)) for p in paths]
        self.image = ImageMobject(self.frames[0].pixel_array.copy()).set_height(height)
        if pixelated:
            self.image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.idx = 0
        self.t = 0.0
        self.fps = 0.0
        self.hold_last = True
        self.add(self.image)

    def n(self) -> int:
        return len(self.frames)

    def duration(self, fps: float) -> float:
        return self.n() / fps

    def show(self, idx: int) -> "FrameSequence":
        idx = max(0, min(idx, self.n() - 1))
        if idx != self.idx:
            src = self.frames[idx].pixel_array
            arr = src.copy()
            img = self.image
            img.orig_alpha_pixel_array = src[:, :, 3].copy()
            if img.stroke_opacity < 1:  # keep a fade in progress
                arr[:, :, 3] = (img.orig_alpha_pixel_array * img.stroke_opacity).astype(arr.dtype)
            img.pixel_array = arr
            self.idx = idx
        return self

    def _tick(self, m, dt: float) -> None:
        self.t += dt
        self.show(int(self.t * self.fps))

    def start(self, fps: float, from_frame: int = 0) -> "FrameSequence":
        self.fps = fps
        self.t = from_frame / fps
        self.show(from_frame)
        self.add_updater(self._tick)
        return self

    def stop(self) -> "FrameSequence":
        self.remove_updater(self._tick)
        return self


class BeatScene(Scene):
    beat: str = ""

    def setup(self) -> None:
        self.camera.background_color = BG
        self.L = layout()
        # No add_sound here: Manim muxes with shortest=1 and would cut the
        # padded tail. build/assemble.py lays the narration over the concat.

    def elapsed(self) -> float:
        """Seconds of scene rendered so far (the renderer's own clock)."""
        return float(self.renderer.time)

    def target_seconds(self) -> float:
        return beat_seconds(self.beat)

    def finish(self, tail: float = 0.4, fade: float = 0.3, frozen: bool = True) -> None:
        """Pad to the narration length (+tail), fading everything out at the end.

        Scene length = beat_seconds + tail, which is exactly the slot the
        assembly step gives this beat (narration + inter-beat gap). The pad
        is a whole number of frames: a frozen wait renders int(d * fps)
        frames, a live one ceil(d * fps), so the epsilon leans each the
        right way and -ql (15 fps) and --fps 30 agree to the frame.
        """
        from manim import FadeOut

        fps = config.frame_rate
        total = self.target_seconds() + tail
        n = round((total - fade - self.elapsed()) * fps)
        if n > 0:
            if frozen:
                self.wait(n / fps + 1e-6, frozen_frame=True)
            else:
                self.wait(n / fps - 1e-6, frozen_frame=False)
        mobs = list(self.mobjects)
        if mobs and fade > 0:
            self.play(FadeOut(Group(*mobs)), run_time=fade)


def pixel_grid(n: int, cell: float, stroke: float = 1.0, color: str = GRID) -> VGroup:
    """n x n grid of squares, row-major, indexable as grid[row * n + col]."""
    from manim import Square

    cells = VGroup(*[Square(cell, stroke_width=stroke, stroke_color=color, fill_opacity=0) for _ in range(n * n)])
    cells.arrange_in_grid(n, n, buff=0)
    return cells


def manhattan_levels(n: int, seed: tuple[int, int]) -> np.ndarray:
    """BFS level (4-connectivity) of every cell in an n x n grid from `seed`."""
    r0, c0 = seed
    rr, cc = np.indices((n, n))
    return np.abs(rr - r0) + np.abs(cc - c0)


def chebyshev_levels(n: int, seed: tuple[int, int]) -> np.ndarray:
    """BFS level (8-connectivity)."""
    r0, c0 = seed
    rr, cc = np.indices((n, n))
    return np.maximum(np.abs(rr - r0), np.abs(cc - c0))
