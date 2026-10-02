"""One scene per stage, every scene rebuilt from the same pure state builder.

The three panes persist across the video, but the pipeline renders one
clip per beat and concatenates them. So:

- `build_state(k)` draws the picture "after stage k-1" from pure
  functions (fixed construction order, no updaters, no ValueTracker).
- Scene k adds that picture in `setup()`, so its first frame is exactly
  the last frame of clip k-1.
- Scene k animates its transition (sweep the previous blob up into the
  strip, bring in the new replay, GPU config and matrix column), plays
  its stage, then `snap()`s: everything is removed and `build_state(k+1)`
  is added, and the clip holds on that picture. The hold IS the next
  clip's first frame; `build/seam_check.py` verifies it.
- Whole-frame timing: a rendered play of `fr(n)` seconds lasts exactly
  n frames at 15 fps (2n at 30); a frozen hold of `n/15 + EPS` too.
  (`np.arange` rounds a play up, `int()` rounds a frozen wait down.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from manim import (
    AnimationGroup, FadeIn, FadeOut, Group, ImageMobject, LaggedStart, VGroup,
    config,
)

from scenes.style import ASSETS, BeatScene, beat_seconds
from scenes.panes import data
from scenes.panes.config import STAGES, Stage
from scenes.panes.geometry import Geometry, pane_geometry
from scenes.panes.left_matrix import MatrixPane
from scenes.panes.middle_strip import (
    MiddlePane, big_image, captions, load_image, slot_center, tag, THUMB_W, fit,
)
from scenes.panes.right_gpu import GpuPane

F = 1 / 15
EPS = 1e-6


def fr(n: int) -> float:
    """run_time of exactly n frames at 15 fps (2n at 30 fps) for a rendered play."""
    return n * F - EPS


def caption_ctx(bench: data.Bench) -> dict:
    h = data.headline()
    return {
        "pure": data.fmt_ms(h.pure_ms), "njit": data.fmt_ms(h.njit_ms),
        "ch05": data.fmt_ms(h.ch05_ms), "mask": data.fmt_ms(h.ch06_mask_ms),
        "rgb": data.fmt_ms(h.ch06_rgb_ms), "n_blobs": f"{h.n_blobs:,}",
        "ratio": data.fmt_ratio_round(h.ratio), "sm": bench.sm_count,
        "red_px": f"{h.red_px:,}", "n_runs": f"{h.n_runs:,}",
        "runs_ratio": f"{h.red_px / h.n_runs:.0f}×",
    }


def stage_captions(st: Stage, ctx: dict) -> tuple:
    c1 = st.caption.format(**ctx) if st.caption else None
    c2 = st.caption2.format(**ctx) if st.caption2 else None
    return c1, c2


@dataclass
class State:
    left: MatrixPane | None
    middle: MiddlePane | None
    right: GpuPane | None

    def all(self) -> list:
        out = []
        if self.left is not None:
            out += self.left.all()
        if self.right is not None:
            out += self.right.all()
        if self.middle is not None:
            out += self.middle.all()
        return out


def build_state(k: int, geo: Geometry, bench: data.Bench) -> State:
    """The picture after stage k-1 (k = 0: an empty frame)."""
    if k == 0:
        return State(None, None, None)
    prev = STAGES[k - 1]
    ctx = caption_ctx(bench)
    left = MatrixPane(bench, geo.left, n_cols=min(k, len(data.COLUMNS)), current=k - 1)
    right = GpuPane(geo.right, prev.gpu, sm_count=bench.sm_count)
    thumbs = [(STAGES[i].thumb, STAGES[i].tag) for i in range(k - 1)]
    c1, c2 = stage_captions(prev, ctx)
    middle = MiddlePane(geo.middle, thumbs, prev.thumb, c1, c2,
                        fit_wh=prev.big_fit, dy=prev.big_dy, extra=prev.extra)
    return State(left, middle, right)


def build_live(k: int, geo: Geometry, bench: data.Bench, frame: int = 48) -> State:
    """Layout aid only: the picture in the middle of stage k."""
    st = STAGES[k]
    ctx = caption_ctx(bench)
    n_cols = min(k + 1, len(data.COLUMNS)) if st.col else min(k, len(data.COLUMNS))
    left = MatrixPane(bench, geo.left, n_cols=n_cols, current=(k if st.col else k - 1))
    right = GpuPane(geo.right, st.gpu, sm_count=bench.sm_count)
    thumbs = [(STAGES[i].thumb, STAGES[i].tag) for i in range(k)]
    c1, c2 = stage_captions(st, ctx)
    src = f"{st.frames}/frame_{frame:03d}.png" if st.frames else st.thumb
    middle = MiddlePane(geo.middle, thumbs, src or None, c1, c2,
                        fit_wh=st.big_fit, dy=st.big_dy, extra=st.extra)
    return State(left, middle, right)


class Replay(Group):
    """Plays assets/<name>/frame_NNN.png at `fps` by swapping one image's
    pixel_array (the Cairo renderer snapshots the mobject family when a
    play starts, so submobjects cannot be swapped). Frames are read from
    disk when shown: the machine has little RAM and a 900 px set held as
    97 image mobjects is half a gigabyte."""

    def __init__(self, name: str, box, fit_wh=(4.1, 4.1), dy: float = 0.0,
                 center=None, loop: bool = False):
        super().__init__()
        folder = ASSETS / name
        self.paths = sorted(folder.glob("frame_*.png"))
        if not self.paths:
            raise FileNotFoundError(f"run assets/extract_gifs.py first: {folder}")
        self.image = big_image(box, self.paths[0], fit_wh, dy)
        if center is not None:
            self.image.move_to(center)
        self.loop = loop
        self.idx, self.t, self.fps = 0, 0.0, 0.0
        self.add(self.image)

    def n(self) -> int:
        return len(self.paths)

    def duration(self) -> float:
        return self.n() / self.fps

    def frame(self, idx: int) -> np.ndarray:
        """RGBA uint8 array of frame `idx`, exactly as ImageMobject loads it."""
        from PIL import Image

        with Image.open(self.paths[idx]) as im:
            return np.array(im.convert("RGBA"))

    def show(self, idx: int) -> "Replay":
        idx = idx % self.n() if self.loop else max(0, min(idx, self.n() - 1))
        if idx != self.idx:
            src = self.frame(idx)
            arr = src.copy()
            img = self.image
            img.orig_alpha_pixel_array = src[:, :, 3].copy()
            if img.stroke_opacity < 1:
                arr[:, :, 3] = (img.orig_alpha_pixel_array * img.stroke_opacity).astype(arr.dtype)
            img.pixel_array = arr
            self.idx = idx
        return self

    def _tick(self, m, dt: float) -> None:
        self.t += dt
        self.show(int(self.t * self.fps))

    def start(self, fps: float) -> "Replay":
        self.fps = fps
        self.t = 0.0
        self.show(0)
        self.add_updater(self._tick)
        return self

    def stop(self) -> "Replay":
        self.remove_updater(self._tick)
        if not self.loop:
            self.show(self.n() - 1)
        return self

    def sync_to(self, other: "Replay") -> "Replay":
        """Start ticking in lockstep with `other` (same fps, same frame)."""
        self.fps = other.fps
        self.t = other.t
        self.show(int(self.t * self.fps))
        self.add_updater(self._tick)
        return self


class StageScene(BeatScene):
    k: int = 0

    # ---- setup
    def setup(self) -> None:
        super().setup()
        self.geo = pane_geometry(self.L)
        self.bench = data.load_bench()
        self.ctx = caption_ctx(self.bench)
        self.cfg = STAGES[self.k]
        self.beat = self.cfg.beat
        self.state = build_state(self.k, self.geo, self.bench)
        self.add(*self.state.all())
        self.replay: Replay | None = None
        self.replay_t0 = 0.0
        self.caption = None

    def target_seconds(self) -> float:
        return beat_seconds(self.beat, default=self.cfg.target_s)

    # ---- timing
    def until(self, t: float, live: bool = False) -> None:
        """Advance the clip to `t` seconds (whole frames). `live` while a
        replay ticks: the wait must go through play_internal."""
        fps = config.frame_rate
        n = round((t - self.elapsed()) * fps)
        if n <= 0:
            return
        if live:
            if n / fps >= 1 / config.frame_rate * 1.5:
                self.wait(n / fps - EPS, frozen_frame=False)
        else:
            self.wait(n / fps + EPS, frozen_frame=True)

    # ---- pieces
    def new_replay(self) -> Replay | None:
        cfg = self.cfg
        if not cfg.frames:
            return None
        return Replay(cfg.frames, self.geo.middle, cfg.big_fit, cfg.big_dy)

    def new_caption(self, line1=None, line2=None) -> VGroup:
        """Caption group for this stage; `{placeholders}` come from `caption_ctx`."""
        if line1 is None and line2 is None:
            line1, line2 = stage_captions(self.cfg, self.ctx)
        line1 = line1.format(**self.ctx) if line1 else None
        line2 = line2.format(**self.ctx) if line2 else None
        return captions(self.geo.middle, line1, line2)

    def first_caption(self) -> tuple:
        """The caption shown when the stage begins (hooks may swap it later)."""
        return stage_captions(self.cfg, self.ctx)

    def swap_caption(self, line1, line2, extra_out=(), extra_in=(), frames: int = 6) -> None:
        """Fade the current caption out and a new one in (never in place)."""
        new = self.new_caption(line1, line2)
        self.play(FadeOut(self.caption), *extra_out, run_time=fr(frames // 2))
        self.play(FadeIn(new), *extra_in, run_time=fr(frames - frames // 2))
        self.caption = new

    def start_replay(self, replay: Replay, fps: float) -> None:
        replay.start(fps)
        self.replay = replay
        self.replay_t0 = self.elapsed()

    def wait_replay(self) -> None:
        if self.replay is not None:
            self.until(self.replay_t0 + self.replay.duration() + F, live=True)
            self.replay.stop()

    def reveal_column(self, j: int) -> None:
        left = self.state.left
        hdr, cells, nums = left.header(j), left.col_group(j), left.num_group(j)
        self.play(
            FadeIn(hdr, run_time=fr(6)),
            LaggedStart(*[FadeIn(VGroup(c, n)) for c, n in zip(cells, nums)], lag_ratio=0.05),
            run_time=fr(18),
        )

    # ---- the three phases
    def intro(self) -> None:
        """Stage 0: the panes appear, then the CPU column."""
        cfg = self.cfg
        left = MatrixPane(self.bench, self.geo.left, n_cols=0, current=None)
        right = GpuPane(self.geo.right, cfg.gpu, sm_count=self.bench.sm_count)
        middle = MiddlePane(self.geo.middle, [], None, None, None)
        self.state = State(left, middle, right)
        self.play(FadeIn(left.static), FadeIn(right.static), FadeIn(right.dynamic), run_time=fr(12))
        replay = self.new_replay()
        self.caption = self.new_caption(*self.first_caption())
        self.play(FadeIn(replay), FadeIn(self.caption), run_time=fr(9))
        self.start_replay(replay, cfg.replay_fps)
        self.reveal_column(0)

    def transition(self) -> None:
        """Stage k >= 1: sweep the previous blob up, bring in this stage."""
        k, cfg, prev = self.k, self.cfg, self.state
        box = self.geo.middle
        pcfg = STAGES[k - 1]
        outs = [FadeOut(prev.middle.caption)]
        if prev.middle.big_extra is not None:
            outs.append(FadeOut(prev.middle.big_extra))
        gpu_changes = cfg.gpu is not pcfg.gpu
        if gpu_changes:
            outs.append(FadeOut(prev.right.dynamic))
        self.play(
            prev.middle.big_image.animate.scale_to_fit_width(THUMB_W).move_to(slot_center(box, k - 1)),
            FadeIn(tag(box, k - 1, pcfg.tag)),
            *outs,
            run_time=fr(9),
        )
        ins = []
        replay = self.new_replay()
        if replay is not None:
            ins.append(FadeIn(replay))
        self.caption = None
        if cfg.caption:
            self.caption = self.new_caption(*self.first_caption())
            ins.append(FadeIn(self.caption))
        if gpu_changes:
            new_gpu = GpuPane(self.geo.right, cfg.gpu, sm_count=self.bench.sm_count)
            ins.append(FadeIn(new_gpu.dynamic))
        if ins:
            self.play(*ins, run_time=fr(9))
        if replay is not None:
            self.start_replay(replay, cfg.replay_fps)
        if cfg.col is not None:
            self.reveal_column(k)

    def stage(self) -> None:
        """Hook: the stage's own content. Default: let the replay finish."""
        self.wait_replay()

    def snap(self) -> None:
        """Replace everything with the pure picture of the next state."""
        if self.replay is not None:
            self.replay.stop()
        for m in list(self.mobjects):
            self.remove(m)
        self.state = build_state(self.k + 1, self.geo, self.bench)
        self.add(*self.state.all())

    def construct(self) -> None:
        if self.k == 0:
            self.intro()
        else:
            self.transition()
        self.stage()
        if self.k + 1 < len(STAGES):
            self.snap()
            self.finish(fade=0)
        else:
            self.finish(fade=0.5)
