"""r06: pass fewer gates. With 4 neighbours a square fills as a diamond;
with the 4 diagonals too, as a square, in half the rings: half the gates.
Act 1 closes: the scoreboard strip's second pill ("huge blob") fills for
the GPU and the SIZE tag gets its tick.

The stencil is an algorithm, not hardware: a small 3 x 3 glyph beside the
floor shows the neighbours one step reaches (4 arrows, then 8). It is an
extra and leaves with the last caption; the machine does not change.

Starts on chain_state(10) (the snake painted, "65,792 stops"), ends on
chain_state(11) (the square painted in 8 ways, row 2 filled, "25.8×").
"""
from __future__ import annotations

import numpy as np
from manim import Arrow, FadeIn, FadeOut, GrowArrow, Square, VGroup

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import LEVEL_LO, Z_OVER, carry_caption, level_colors, score_rows, toys
from scenes.style import BLUE, GRID, INK

SIDES = ((0, 1), (1, 0), (0, -1), (-1, 0))
DIAGONALS = ((1, 1), (1, -1), (-1, -1), (-1, 1))
GLYPH_AT = (-4.0, -0.9)         # world: just left of the floor's x = -3 edge


def stencil(pl) -> tuple[VGroup, VGroup, VGroup]:
    """The 3 x 3 stencil glyph, laid on the floor's plane beside it (the empty
    corner below the floor's left vertex, clear of the inset above and the
    caption below), so its arrows run along the floor's own rows and
    columns: (the grid, the centre and its 4 side neighbours with their
    arrows; the 4 diagonal cells; the 4 diagonal arrows)."""
    cell, step = 0.4, 0.46

    def box(dx: int, dy: int, color: str, z: float) -> Square:
        s = Square(cell, fill_color=color, fill_opacity=1, stroke_color=GRID, stroke_width=1.5)
        return s.move_to(step * np.array([dx, dy, 0.0])).set_z_index(z)

    def arrow(dx: int, dy: int) -> Arrow:
        d = np.array([dx, dy, 0.0])
        a = Arrow(0.05 * d / np.linalg.norm(d), step * d, buff=0, color=INK, stroke_width=3.5, tip_length=0.12,
                  max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=30)
        return a.set_z_index(Z_OVER - 4)

    grid = VGroup(*[box(dx, dy, fl.EMPTY, Z_OVER - 6) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    side = VGroup(grid, box(0, 0, LEVEL_LO, Z_OVER - 5), *[box(dx, dy, BLUE, Z_OVER - 5) for dx, dy in SIDES],
                  *[arrow(dx, dy) for dx, dy in SIDES])
    corners = VGroup(*[box(dx, dy, BLUE, Z_OVER - 5) for dx, dy in DIAGONALS])
    diag = VGroup(*[arrow(dx, dy) for dx, dy in DIAGONALS])
    cx, cy = GLYPH_AT
    for g in (side, corners, diag):         # built around the origin in world units
        g.apply_function(lambda p: pl.to_frame([[cx + p[0], cy + p[1], 0.0]])[0])
    return side, corners, diag


class Diagonals(ChainScene):
    k = 10

    def wave(self, m, lv, n: int):
        """keyfn and render for a fill ring by ring: key -1 is all red, n - 1 all painted."""
        def render(i: int):
            return self.toy(level_colors(m, lv, upto=i, top=n - 1), ring=(lv == i + 1) if i < n - 1 else None)
        return (lambda p: round(p * n) - 1), render

    def say_last(self, at: float | None = None, frames: int = 5, extra=()) -> None:
        """ChainScene.say_last, plus `extra` played with the caption's fade-in
        (drop this copy once base.py's say_last takes `extra`)."""
        self.say(*carry_caption(self.k + 1), at=at, frames=frames, extra=extra)

    def beat_anim(self) -> None:
        t = toys()
        sq, l4, l8 = t["square"], t["square4"].level, t["square8"].level
        n4, n8 = int(l4.max()) + 1, int(l8.max()) + 1
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)))
        side, corners, diag = stencil(self.Lay.floor_pl)

        # "So": the square again, all red, and the stencil beside it: one step
        # reaches the 4 side neighbours (the blend is a hook, so both run together)
        self.until(self.cue("So"))
        snake = self.toy(level_colors(t["snake"], t["snake4"].level))
        self.floor_blend(snake, red, frames=4)
        self.play(FadeIn(side, scale=0.9), run_time=fr(4))
        self.extras = [side]

        # with 4 neighbours it fills as a diamond, ring by ring; the snake's
        # stop count leaves once "gates" is said
        d0, d1 = self.cue("fewer"), self.cue("diagonally")
        key, render = self.wave(sq, l4, n4)
        self.until(d0)
        self.floor_play(key, render, d1 - d0)
        self.until(self.cue("gates", end=True))
        self.unsay()

        # "diagonally": the 4 diagonal neighbours join the stencil
        self.until(d1)
        self.play(FadeIn(corners), *[GrowArrow(a) for a in diag], run_time=fr(5))
        self.extras += [corners, diag]

        # "On a square": red again, then 8 neighbours fill it as a square
        self.until(self.cue("On"))
        full4 = self.toy(level_colors(sq, l4))
        self.floor_blend(full4, red, frames=3)
        s0, s1 = self.cue("square"), self.cue("rings", end=True)
        key, render = self.wave(sq, l8, n8)
        self.until(s0)
        self.floor_play(key, render, s1 - s0)
        self.say(f"{n4} → {n8} rings", at=self.cue("half"))

        # "half the gates": size is won. The pill, the tag's tick and the
        # result come in together (one text change), the stencil leaves. The
        # new strip fades in over the old one: its pills are opaque, so the
        # two that do not change never dip
        small, huge, _ = score_rows()
        old = self.over["strip"]
        self.over["strip"] = self.strip((small, huge, "?"))
        anims = [FadeIn(self.over["strip"])] + self.over_anims("tag", ov.act_tag(self.G, "SIZE", True))
        anims += [FadeOut(m) for m in self.extras]
        self.extras = []
        self.say_last(at=self.cue("gates", after=self.cue("half", n=2)), extra=anims)
        self.remove(old)
