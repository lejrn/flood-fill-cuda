"""r06: pass fewer gates. With 4 neighbours a square fills as a diamond;
with the 4 diagonals too, as a square, in half the rings: half the gates.
Act 1 closes: the scoreboard strip's second pill ("huge blob") fills for
the GPU and the SIZE tag gets its tick.

The stencil is an algorithm, not hardware: a small 3 x 3 glyph beside the
floor shows the neighbours one step reaches (4 arrows, then 8). It is an
extra and leaves with the last caption.

v4 (D4, D6): both refills are the 48-block launch, so the floor paints its
real owner mosaic (the 4-way diamond `square_own48`, the 8-way square
`square_own48_8`), never one blue: in act 1 a hue is a block. The cubes
keep their hues and follow the busy rule ring by ring (`big_level_sizes`,
then `big8_level_sizes`): a block with no work in the ring waits at the
gate, amber (r05b named the gate). The diamond's small rings at both ends
keep most cubes amber; the square's only at its start, and it ends with
all 48 busy.

Starts on chain_state(10) (the snake painted, "65,792 stops"), ends on
chain_state(11) (the square's 8-way owner mosaic, row 2 filled, "25.8×").
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import Arrow, FadeIn, FadeOut, Group, GrowArrow, Square, VGroup, VMobject

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import numbers as nb
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import (LEVEL_LO, Z_OVER, floor_raster, level_colors, owner_colors, owner_fine,
                                score_rows, toys)
from scenes.style import BLUE, GRID, INK

SIDES = ((0, 1), (1, 0), (0, -1), (-1, 0))
DIAGONALS = ((1, 1), (1, -1), (-1, -1), (-1, 1))
GLYPH_AT = (-4.0, -0.9)         # world: just left of the floor's x = -3 edge
# A cube turns amber fast, so a fill's first rings (block 0's alone) read
# at once, and goes back to its hue slowly, so the blocks joining read as
# a wave rather than a flicker
ATTACK, RELEASE = 0.07, 0.2     # seconds for the full change, each way
LINGER = 0.35                   # seconds the last ring's waiting cubes stay amber after a fill ends


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


def set_alpha(g: Group, a: float) -> None:
    """Fill opacity of a machine part's box faces (a Group has no set_opacity)."""
    for m in g.get_family():
        if isinstance(m, VMobject):
            m.set_fill(opacity=a, family=False)


def cubes_by_block(teams: Group, pl, sm_ids) -> list[Group]:
    """The faces of a teams part, one group per block: block b stands on SM
    sm_ids[b], the lower block of a pair on the left (place_blocks). A face
    goes to the nearest SM centre, then to its side of that SM's middle
    along the board's x."""
    pts = np.array([mc.sm_point(pl, i)[:2] for i in range(24)])
    on: list[list[int]] = [[] for _ in range(24)]
    for b, sm in enumerate(sm_ids):
        on[sm].append(b)
    out = [Group() for _ in sm_ids]
    for face in teams.submobjects:
        c = face.get_center()[:2]
        sm = int(np.argmin(np.linalg.norm(pts - c, axis=1)))
        bs = on[sm]
        if len(bs) == 1:
            out[bs[0]].add(face)
            continue
        r = mc.sm_rect(sm)
        a, z = (pl.to_frame([[x, r.y + r.d / 2, mc.DIE_Z]])[0][:2] for x in (r.x, r.x1))
        out[bs[0] if np.dot(c - (a + z) / 2, z - a) < 0 else bs[1]].add(face)
    if not all(len(g) for g in out):
        raise ValueError("a block's cube has no faces")
    return out


class Diagonals(ChainScene):
    k = 10

    def wave(self, lv, n: int, launch: str):
        """keyfn and render for a fill ring by ring, painted with the owner
        mosaic of `launch` ("c4", "c8"): key -1 is all red, n - 1 all painted."""
        sq, own = toys()["square"], owner_fine(launch)

        def render(i: int):
            ring = (lv == i + 1) if i < n - 1 else None
            return fl.toy_raster(level_colors(sq, lv, upto=i, top=n - 1), self.floor.rpx, ring=ring,
                                 fine=owner_colors(sq, lv, own, upto=i, top=n - 1))
        return (lambda p: round(p * n) - 1), render

    def waiting_cubes(self, fills: list) -> None:
        """D6 on the board while the floor fills: an amber copy of every cube
        over the machine, each faded in while its block has no work in the
        current real ring. fills: (t0, t1, level_sizes) per fill; the real
        rings run evenly from t0 to t1, as the toy's do. After a fill the last
        ring's waiting cubes linger, then every cube is back in its hue (the
        rest picture of every boundary)."""
        n = self.num
        blocks, tpb = int(n["big_blocks"]), int(n["big_tpb"])
        pl = self.Lay.machine_pl
        wait = mc.build(replace(self.machine.spec, team_state="wait"), pl, z0=100.0, only=("teams",))
        amber = cubes_by_block(wait.parts["teams"], pl, [int(i) for i in n["big_sm_ids"]])
        for g in amber:
            set_alpha(g, 0.0)
        self.add(*amber)                    # after the floor: drawn every frame, over the machine's cubes
        busy = [[nb.busy_blocks(int(s), blocks, tpb) for s in sizes] for _, _, sizes in fills]
        alpha = np.zeros(blocks)
        last = {"t": self.elapsed()}
        end = fills[-1][1]

        def working(t: float) -> int:
            for (t0, t1, _), bz in zip(fills, busy):
                if t0 <= t <= t1 + LINGER:
                    p = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
                    return bz[round(p * (len(bz) - 1))]
            return blocks

        def hook(t: float):
            target = (np.arange(blocks) >= working(t)).astype(float)
            dt = max(t - last["t"], 0.0)
            last["t"] = t
            new = alpha + np.clip(target - alpha, -dt / RELEASE, dt / ATTACK)
            for b in np.nonzero(new != alpha)[0]:
                set_alpha(amber[b], float(new[b]))
            alpha[:] = new
            return t <= end or bool(alpha.any())

        self._hooks.append(hook)

    def beat_anim(self) -> None:
        t = toys()
        l4, l8 = t["square4"].level, t["square8"].level
        n4, n8 = int(l4.max()) + 1, int(l8.max()) + 1
        n = self.num
        rpx = self.floor.rpx
        red = self.toy(level_colors(t["square"], np.full(l4.shape, -1)))
        side, corners, diag = stencil(self.Lay.floor_pl)

        # the cubes wait at the gate on every ring their block has no work in
        d0, d1 = self.cue("fewer"), self.cue("diagonally")
        s0, s1 = self.cue("square"), self.cue("rings", end=True)
        self.waiting_cubes([(d0, d1, n["big_level_sizes"]), (s0, s1, n["big8_level_sizes"])])

        # "So": the square again, all red, and the stencil beside it: one step
        # reaches the 4 side neighbours (the blend is a hook, so both run together)
        self.until(self.cue("So"))
        self.floor_blend(floor_raster("snake", rpx), red, frames=4)
        self.play(FadeIn(side, scale=0.9), run_time=fr(4))
        self.extras = [side]

        # with 4 neighbours it fills as a diamond, ring by ring, every dot its
        # block's; the snake's stop count leaves once "gates" is said
        key, render = self.wave(l4, n4, "c4")
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
        self.floor_blend(floor_raster("square_own48", rpx), red, frames=3)
        key, render = self.wave(l8, n8, "c8")
        self.until(s0)
        self.floor_play(key, render, s1 - s0)
        self.say(f"{int(n['big_rings4']):,} → {int(n['big_rings8']):,} rings", at=self.cue("half"))

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
