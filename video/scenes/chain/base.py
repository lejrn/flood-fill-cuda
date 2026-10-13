"""ChainScene: one beat of the chain cut.

A scene starts on `chain_state(k)`, plays its beat on a timeline cued by
the spoken words, and ends with `snap()` to `chain_state(k + 1)`, held to
the narration's length + 0.4 s. Rules (as in the three-pane cut):

- Plays last whole frames (`fr(n)`: n frames at 15 fps, 2n at 30), holds
  go through `until()`.
- Text never changes in place: captions and the goal chip fade out and in.
- The machine changes part by part (`machine_to`): a new part slides in,
  a removed part fades out, a changed part cross-fades, an unchanged part
  is left alone. Every part is a top-level mobject.
- Floor animations and amber flashes run on one clock (`tick`), a hook
  called every rendered frame, so a wave's rings and the gate's flashes
  stay in step. While a hook runs, holds are live (updaters run). The
  clock is an updater on the floor itself: during a play the Cairo
  renderer caches every mobject that is neither animated nor updated, and
  the floor changes by swapping its pixels. The floor is added after the
  machine, so the machine stays cached while only the floor moves.
- Cues come from out/<VOICE>/words.json (`cue("gate")`), so the same
  scene fits every voice.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from manim import DOWN, LEFT, ORIGIN, RIGHT, UP, FadeIn, FadeOut, Mobject, VGroup, config

from narration.common import read_words
from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import numbers as nb
from scenes.chain.state import (
    BEATS, Z_OVER, chain_state, goal_chip, layout,
)
from scenes.style import INK, INK_SOFT, OUT, BeatScene, beat_seconds, label, voice

F = 1 / 15
EPS = 1e-6


def fr(n: int) -> float:
    """run_time of exactly n frames at 15 fps (2n at 30 fps)."""
    return n * F - EPS


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9'-]", "", w.lower())


def caption_group(G, text: str, sub: str | None = None) -> VGroup:
    """The big caption word (lines centred) and its small mono label: below it
    in 9:16, beside it in 16:9 unless that is too wide, then below and smaller."""
    size = 60 if G.vertical else 52
    lines = text.split("\n") if G.vertical else [text.replace("\n", " ")]
    big = VGroup(*[label(line, size, INK) for line in lines]).arrange(DOWN, buff=0.12)
    g = VGroup(big)
    room = config.frame_width - 0.8
    if sub:
        small = label(sub, 24, INK_SOFT, mono=True)
        g.add(small)
        if G.vertical:
            g.arrange(DOWN, buff=0.14)
        else:
            g.arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
            if g.width > room or big.height > 1.0:
                g.arrange(DOWN, buff=0.08)
                g.scale_to_fit_height(min(g.height, 1.15))
    if g.width > room:
        g.scale_to_fit_width(room)
    g.move_to(G.caption_c)
    if not G.vertical:          # keep the bottom band inside the frame
        g.shift(UP * max(0.0, (-config.frame_height / 2 + 0.12) - g.get_bottom()[1]))
    for m in g.get_family():
        m.set_z_index(Z_OVER + 2)
    return g


class ChainScene(BeatScene):
    k: int = 0

    # ---- setup
    def setup(self) -> None:
        super().setup()
        self.beat = BEATS[self.k]
        self.Lay = layout()
        self.G = self.Lay.G
        self.num = {key: v.value for key, v in nb.load().items()}
        self.der = nb.derived()
        self.words = read_words(OUT / voice()).get(self.beat, [])
        self.state = chain_state(self.k, self.Lay)
        self.add(*self.state.mobs())
        self.floor = self.state.floor
        self.machine = self.state.machine
        self.goal = self.state.goal
        self.cap: VGroup | None = self.state.cap      # carried over the cut
        self.extras: list = []          # transient mobjects to clear before snap
        self._hooks: list = []
        self._clock = None

        # a plain function, not a bound method: Manim deep-copies mobjects
        # (FadeIn keeps a starting copy), and deep-copying a bound method
        # copies the whole scene, thread locks included
        def tick(m, dt, scene=self):
            scene._tick(m, dt)

        self._tick_fn = tick
        self.adopt_floor(self.floor)

    def target_seconds(self) -> float:
        return beat_seconds(self.beat, default=10.0)

    # ---- words
    def cue(self, word: str, n: int = 1, end: bool = False, after: float = 0.0) -> float:
        """Start (or end) time of the n-th occurrence of `word` after `after` s."""
        want = _norm(word)
        hits = [w for w in self.words if _norm(w[0]) == want and w[1] >= after]
        if len(hits) < n:
            if not self.words:          # no words.json: spread cues by guess
                return min(self.elapsed() + 0.5, self.target_seconds())
            raise ValueError(f"{self.beat}: no occurrence {n} of {word!r} after {after}")
        return float(hits[n - 1][2] if end else hits[n - 1][1])

    # ---- time
    def until(self, t: float) -> None:
        fps = config.frame_rate
        n = round((t - self.elapsed()) * fps)
        if n <= 0:
            return
        if self._hooks:
            self.wait(n / fps - EPS, frozen_frame=False)
        else:
            self.wait(n / fps + EPS, frozen_frame=True)

    def adopt_floor(self, floor) -> None:
        """Move the clock to `floor` (or to an empty mobject while there is none)."""
        if self._clock is not None:
            self._clock.remove_updater(self._tick_fn)
            if self._clock is not self.floor and self._clock in self.mobjects:
                self.remove(self._clock)
        self.floor = floor
        self._clock = floor if floor is not None else Mobject()
        self._clock.add_updater(self._tick_fn)
        if floor is None:
            self.add(self._clock)

    def _tick(self, m, dt: float) -> None:
        # The renderer's clock, not a sum of dt: a play's time steps start at
        # t = 0, so summing dt loses one frame per play (and a one-frame wait
        # never advances at all). Inside an update, renderer.time is exactly
        # the time of the frame about to be drawn.
        now = self.elapsed()
        keep = []
        for h in self._hooks:
            if h(now) is not False:
                keep.append(h)
        self._hooks = keep

    def run(self, fn, seconds: float, start: float | None = None) -> None:
        """Call fn(p) every frame, p from 0 to 1 over `seconds` from now (or
        from scene time `start`); the last call has p = 1."""
        t0 = self.elapsed() if start is None else start

        def hook(t: float):
            p = (t - t0) / seconds if seconds > 0 else 1.0
            if p < 0:
                return True
            fn(min(p, 1.0))
            return p < 1.0

        if start is None:
            fn(0.0)
        self._hooks.append(hook)

    def settle(self) -> None:
        """Wait (live) until every hook has finished, then draw one frame.

        A play ends with one more update after its last drawn frame, so a
        hook can reach its end state there and never be drawn; the frozen
        frame makes the end state the frame before whatever comes next."""
        while self._hooks:
            self.wait(1 / config.frame_rate - EPS, frozen_frame=False)
        self.wait(1 / config.frame_rate + EPS, frozen_frame=True)

    # ---- floor
    def floor_play(self, keyfn, render, seconds: float) -> None:
        """Show render(keyfn(p)) on the floor as p goes 0 -> 1; renders only on a new key."""
        last = {"key": object()}
        f = self.floor

        def step(p: float) -> None:
            key = keyfn(p)
            if key != last["key"]:
                f.show_raster(render(key))
                last["key"] = key

        self.run(step, seconds)

    def floor_show(self, raster: np.ndarray) -> None:
        self.floor.show_raster(raster)

    def toy(self, colors, dots=None, ring=None, boxes=()) -> np.ndarray:
        return fl.toy_raster(colors, self.floor.rpx, dots=dots, ring=ring, boxes=boxes)

    def pic(self, rgb) -> np.ndarray:
        return fl.picture_raster(rgb, self.floor.rpx)

    def floor_blend(self, a: np.ndarray, b: np.ndarray, frames: int = 8) -> None:
        """Cross-fade the floor from raster a to raster b over `frames` (15 fps) frames."""
        af, bf = a.astype(np.float32), b.astype(np.float32)
        self.floor_play(lambda p: round(p * frames), lambda i: (af + (bf - af) * (i / frames)).astype(np.uint8),
                        seconds=fr(frames) + EPS)

    # ---- machine
    def machine_anims(self, spec: mc.Spec, slide: float = 0.3) -> list:
        """Animations from the current machine to `spec`, part by part."""
        new = mc.build(spec, self.Lay.machine_pl, z0=100.0)
        old = self.machine
        anims = []
        keep = {}
        names = list(dict.fromkeys(list(old.parts) if old else []) | dict.fromkeys(new.parts))
        for name in names:
            o = old.parts.get(name) if old else None
            n = new.parts.get(name)
            if o is not None and n is not None and _same(o, n):
                keep[name] = o
                continue
            if o is not None:
                anims.append(FadeOut(o))
            if n is not None:
                shift = DOWN * slide if (o is None and slide) else ORIGIN
                anims.append(FadeIn(n, shift=shift))
                keep[name] = n
        self.machine = mc.Machine(spec, {k: keep[k] for k in new.parts if k in keep})
        return anims

    def machine_to(self, spec: mc.Spec, frames: int = 6, slide: float = 0.3, extra=()) -> None:
        anims = self.machine_anims(spec, slide)
        if anims or extra:
            self.play(*anims, *extra, run_time=fr(frames))

    # ---- goal chip
    def goal_to(self, value, frames: int = 6, extra=()) -> None:
        new = goal_chip(self.G, value)
        anims = [FadeIn(new)]
        if self.goal is not None:
            anims.append(FadeOut(self.goal))
        self.play(*anims, *extra, run_time=fr(frames))
        self.goal = new

    # ---- captions
    def say(self, text: str, sub: str | None = None, at: float | None = None, frames: int = 5, extra=()) -> None:
        if at is not None:
            self.until(at)
        new = caption_group(self.G, text, sub)
        if self.cap is not None:        # out, then in: two captions never overlap
            self.play(FadeOut(self.cap), run_time=fr(2))
        self.play(FadeIn(new, scale=1.06), *extra, run_time=fr(frames))
        self.cap = new

    def say_last(self, at: float | None = None, frames: int = 5) -> None:
        """The beat's last caption: the one `chain_state(k + 1)` carries over the cut."""
        from scenes.chain.state import carry_caption

        cc = carry_caption(self.k + 1)
        if cc is None:
            raise ValueError(f"{self.beat}: no carried caption for state {self.k + 1}")
        self.say(*cc, at=at, frames=frames)

    def unsay(self, frames: int = 4, extra=()) -> None:
        anims = list(extra)
        if self.cap is not None:
            anims.append(FadeOut(self.cap))
            self.cap = None
        if anims:
            self.play(*anims, run_time=fr(frames))

    # ---- the end of a beat
    def snap(self) -> None:
        """Replace everything with the pure picture of the next state."""
        self._hooks = []
        for m in list(self.mobjects):
            self.remove(m)
        self._clock.remove_updater(self._tick_fn)
        self.state = chain_state(self.k + 1, self.Lay)
        self.add(*self.state.mobs())
        self._clock = None
        self.floor = self.state.floor

    def beat_anim(self) -> None:
        raise NotImplementedError

    def construct(self) -> None:
        self.beat_anim()
        self.settle()
        if os.environ.get("CHAIN_DEBUG"):
            print(f"[{self.beat}] beat done at {self.elapsed():.2f} s, slot {self.target_seconds() + 0.4:.2f} s")
        if self.k + 1 < len(BEATS):
            if self.extras:
                self.play(*[FadeOut(m) for m in self.extras], run_time=fr(3))
            self.write_snap_time()
            self.snap()
            self.finish(fade=0)
        else:
            self.finish(fade=0.5)

    def write_snap_time(self) -> None:
        """<clip>.snap.json beside the clip: build/seam_check.py --cut chain
        compares the frames on both sides of the snap."""
        fw = getattr(self.renderer, "file_writer", None)
        path = getattr(fw, "movie_file_path", None)
        if path:
            Path(str(path) + ".snap.json").write_text(json.dumps({"snap_s": self.elapsed()}), encoding="utf-8")


def _same(a: VGroup, b: VGroup) -> bool:
    """True when two parts draw the same thing (deterministic builds)."""
    fa, fb = a.family_members_with_points(), b.family_members_with_points()
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        if x.points.shape != y.points.shape or not np.allclose(x.points, y.points, atol=1e-6):
            return False
        if x.get_fill_color().to_hex() != y.get_fill_color().to_hex():
            return False
    return True
