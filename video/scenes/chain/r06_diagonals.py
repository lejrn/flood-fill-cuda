"""r06: pass fewer gates. With 4 neighbours a square fills as a diamond;
with the 4 diagonals too, as a square, in half the rings: half the gates.
Act 1 closes: the scoreboard's second row ("one huge blob") fills for the
GPU and the SIZE tag gets its tick.

Starts on chain_state(8) (the snake painted, no stencil badge), ends on
chain_state(9) (the square painted in 8 ways, stencil 8, row 2 filled).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, LEFT, FadeIn, FadeOut, RoundedRectangle, VGroup, config

from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import boundary, level_colors, score_rows, toys
from scenes.style import GREY, GRID, INK, INK_SOFT, TEAL, label


def score_card(G, rows: tuple, highlight: int | None = None):
    """overlays.score_card with one fix: its header's empty first cell
    (label("")) has no points, and set_x on it raises. Same layout and
    look otherwise; drop this copy once overlays.py skips empty cells."""
    lines = VGroup()
    lines.add(VGroup(label("", 16, INK_SOFT), label("CPU", 16, INK_SOFT), label("GPU", 16, INK_SOFT),
                     label("winner", 16, INK_SOFT)))
    for i, r in enumerate(rows):
        if r is None:
            continue
        if r == "?":
            cells = VGroup(label(ov.ROW_NAMES[i], 20, ov.DIM), label("?", 20, ov.DIM, mono=True),
                           label("?", 20, ov.DIM, mono=True), label("?", 20, ov.DIM, mono=True))
        else:
            col = TEAL if r.winner == "GPU" else GREY
            name = VGroup(label(r.what, 20, INK if i == highlight else INK_SOFT),
                          label(r.picture, 13, INK_SOFT)).arrange(DOWN, buff=0.03, aligned_edge=LEFT)
            cells = VGroup(name, label(ov.fmt_ms(r.cpu_ms), 20, INK, mono=True),
                           label(ov.fmt_ms(r.gpu_ms), 20, INK, mono=True),
                           label(f"{r.winner} {r.factor}", 22, col, mono=True))
        lines.add(cells)
    cols = [max(row[j].width for row in lines) for j in range(4)]
    xs = np.cumsum([0] + [c + 0.45 for c in cols[:-1]])
    lines.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    for row in lines:
        for j, m in enumerate(row):
            if len(m.get_all_points()):
                m.set_x(xs[j] + m.width / 2)
    body = VGroup(label("CPU vs GPU", 22, INK), lines).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    room = config.frame_width - 0.9
    if body.width > room:
        body.scale_to_fit_width(room)
    box = RoundedRectangle(corner_radius=0.16, width=body.width + 0.6, height=body.height + 0.5,
                           fill_color=ov.PANEL, fill_opacity=0.97, stroke_color=GRID, stroke_width=2)
    box.move_to(G.card_c)
    body.move_to(box)
    g = VGroup(box, body)
    for m in g.get_family():
        m.set_z_index(ov.Z_OVER + 20)
    return g


class Diagonals(ChainScene):
    k = 8

    def wave(self, m, lv, n: int):
        """keyfn and render for a fill ring by ring: key -1 is all red, n - 1 all painted."""
        def render(i: int):
            return self.toy(level_colors(m, lv, upto=i, top=n - 1), ring=(lv == i + 1) if i < n - 1 else None)
        return (lambda p: round(p * n) - 1), render

    def beat_anim(self) -> None:
        t = toys()
        sq, l4, l8 = t["square"], t["square4"].level, t["square8"].level
        n4, n8 = int(l4.max()) + 1, int(l8.max()) + 1
        spec = boundary(8).machine
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)))

        # the square again, with 4 neighbours: a diamond, ring by ring
        self.until(self.cue("So"))
        snake = self.toy(level_colors(t["snake"], t["snake4"].level))
        self.floor_blend(snake, red, frames=4)
        # the stencil badge (4 arrows) labels the diamond fill; the blend
        # is a hook, so both run together
        self.machine_to(replace(spec, stencil=4), frames=4)
        d0, d1 = self.cue("fewer"), self.cue("diagonally")
        key, render = self.wave(sq, l4, n4)
        self.until(d0)
        self.floor_play(key, render, d1 - d0)
        # the count once the diamond is done
        self.say(f"{n4} rings", "4 neighbours: a diamond", at=d1)

        # step diagonally too: the badge goes from 4 arrows to 8
        self.until(self.cue("too"))
        self.machine_to(replace(spec, stencil=8), frames=4, slide=0)

        # on a square: 8 neighbours, a square wave, half the rings
        self.until(self.cue("On"))
        full4 = self.toy(level_colors(sq, l4))
        self.floor_blend(full4, red, frames=3)
        s0, s1 = self.cue("square"), self.cue("rings", end=True)
        key, render = self.wave(sq, l8, n8)
        self.until(s0)
        self.floor_play(key, render, s1 - s0)
        n = self.num
        self.say(f"{n4} → {n8} rings",
                 f"8 neighbours; 64 Mpx: {int(n['big_rings4']):,} → {int(n['big_rings8']):,}",
                 at=self.cue("half"))

        # size is won: the scoreboard's second row fills
        small, huge, _ = score_rows()
        rows = (small, huge, "?")
        card = score_card(self.G, rows, highlight=1)
        g = self.cue("gates", after=self.cue("half", n=2))
        self.until(g)
        self.over_to(frames=6, strip=self.strip(rows), tag=ov.act_tag(self.G, "SIZE", True),
                     extra=[FadeIn(card, scale=0.96)])
        self.say_last()
        # the card stays up ~1.8 s, then folds away; faded here, not as an
        # extra: construct's extras fade never draws its last (empty) frame
        self.until(g + 1.8)
        self.play(FadeOut(card), run_time=fr(3))
