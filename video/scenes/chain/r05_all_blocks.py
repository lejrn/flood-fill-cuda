"""r05: all blocks (v4). Every SM gets two blocks (48 x 256 threads) and
the 64 Mpx square goes 8.5x the CPU (v5: 1,195 vs 141 ms, against the
4-neighbour CPU on kept arrays; v4 said 18x). Each colour is one block:
the blocks take turns down one queue, so their pixels mix, and small
rings keep only a few of them busy. Then the launch-shape sweep, and the
default we kept.

At "So" the carried caption leaves and the square goes back to red (a
fresh run). At "every SM" the 22 dark SMs light in a ripple out from SMs
0 and 2. At "forty-eight" the 48 cubes land in the measured order (D5),
each in its block's hue: one on each even SM, then on each odd SM, then
the second cubes. At "Eight" ("Eight and a half times the CPU") the square
refills in one quick pass with the 48-block owner mosaic (D4), "8.5×" /
kernel time.

At "Each colour" the pass replays from the seed out: the mosaic dims to
a ghost, the rings light again under a white outline, and a cube is
bright only while its block has work in the ring (D6: the grid-stride
rule on the real level sizes; dim, not amber, until r05b names the gate).
Slowly over the small rings (rings 0-64 are block 0's alone) until
"busy" ("few busy" / on small rings), then fast through the widest band
(all 48 busy on 1,983 rings) and on, blocks dropping out from the top as
the rings shrink into the corners; the queue drains as the replay ends.
At "they share one queue" the blocks with work load from the one queue in
VRAM (red packets up to their SMs) and store down to its rear (a packet
in each block's hue), as r04's two did. At "pixels" a lens grows out of a
cell on the widest ring with a 64 x 64 crop of the real owner map (a
pixel a square; the cell is painted while it shows) and folds back into
it with the second caption.

At "threads" the sweep card grows out of the die (blocks x threads per
block, a grey ramp, the default 48 x 256 outlined) and every cube lights
again; "+15% at best" / 16 Mpx square at "fifteen"; at "kept" the default
cell pulses; the card folds back into the die as the words end.

Starts on chain_state(8) (SMs 0 and 2, the 2-block mosaic) and ends on
chain_state(9) (every SM, two cubes each in block hues, the 48-block
mosaic). The snake is r05b's.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, AnimationGroup, FadeIn, FadeOut, Group, LaggedStart, there_and_back

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import numbers as nb
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
from scenes.chain.iso import FACES, _shade
from scenes.chain.state import boundary, floor_raster, level_colors, owner_colors, owner_crop, owner_fine, toys
from scenes.style import RED_PX

RIPPLE = 14             # frames of the light ripple over the 22 dark SMs
LAND = 0.4              # seconds of one cube's drop (the drops overlap, in block order)
DROP = 0.22             # how far above its SM a cube starts (frame units)
QUICK = 1.4             # seconds of the quick refill pass at "Eight" (and a half)
SLOW_BUSY = 12          # the slow replay runs until this many blocks have work (a quarter: still few)
SLOW_TAIL = 0.35        # ... and holds the small rings this long past "busy", under the caption
GHOST = 0.6             # a cell the replay has not reached: its paint this far towards the empty floor
INTRO = 5               # frames for the floor and the cubes to go over to the replay
LENS = {True: ((2.62, 3.15), 0.8), False: ((-1.0, 0.9), 0.72)}   # (centre, radius): 9:16, 16:9;
# beside the floor's back-right edge, clear of the seed, the act tag and the machine
PACKET_B = 128          # a packet: one warp's 32 queue entries of 4 B, a dense read or write (as r04)
FRONT, REAR = (1, 2), (4, 5)   # r04's queue slots: the front (read) in VRAM chip 0, the rear (written) in chip 1
LAG = 0.08              # the last busy block's packets trail block 0's by this much of the cycle


def ripple_order(rooms, first=(0, 2)) -> list[int]:
    """The SMs by distance (on the die) from the first ones: a ripple out from them."""
    def c(i: int) -> np.ndarray:
        r = mc.sm_rect(i)
        return np.array([r.x + r.w / 2, r.y + r.d / 2])
    o = np.mean([c(i) for i in first], axis=0)
    return sorted(rooms, key=lambda i: (float(np.linalg.norm(c(i) - o)), i))


def cube_blocks(sm_ids) -> list[int]:
    """The block each cube is, in draw order (SM by SM, the lower block on the left)."""
    return [b for i in range(24) for b in sorted(b for b, s in enumerate(sm_ids) if s == i)]


def cube_faces(teams: Group, n: int) -> list[Group]:
    """A teams part's faces, one Group per cube, in draw order (every box
    has the same visible faces, so a cube is a run of them)."""
    faces = teams.submobjects
    k = len(faces) // n
    return [Group(*faces[c * k:(c + 1) * k]) for c in range(n)]


def crop_cell(key: str = "c4", crop: str = "ring") -> tuple[int, int]:
    """The toy floor cell an owner-map crop sits in (state.owner_fine's dot rule)."""
    own = nb.owner_map()
    g = own["grid"]
    k, o, stride = int(g["dots_per_cell"]), int(g["toy_offset"]), float(g["stride_px"])
    x, y = own[key]["crops"][crop]["center_xy"]          # the repo's [x, y] is the toy's [row, col]
    return o + int(x // stride) // k, o + int(y // stride) // k


class CubePaint:
    """Recolours the cubes in place, every frame if need be (the replay's
    busy states): each face gets its cube's colour, shaded as box_faces
    shades it; only cubes whose colour changed are touched."""

    def __init__(self, cubes: list[Group], pl) -> None:
        self.cubes = cubes
        self.shades = [k for n, _, k in FACES.values() if float(np.dot(n, pl.P.view)) > 1e-6]
        self.now: list = [None] * len(cubes)

    def set(self, colors: list[str]) -> None:
        for c, col in enumerate(colors):
            if col != self.now[c]:
                for face, k in zip(self.cubes[c].submobjects, self.shades):
                    face.set_fill(_shade(col, k), family=False)
                self.now[c] = col


class AllBlocks(ChainScene):
    k = 8

    def beat_anim(self) -> None:
        t = toys()
        sq, lv = t["square"], t["square4"].level
        top = int(lv.max())
        f = self.floor
        pl = self.Lay.machine_pl
        n = self.num
        b8, b9 = boundary(self.k), boundary(self.k + 1)
        ids = [int(i) for i in n["big_sm_ids"]]
        blocks, tpb = int(n["big_blocks"]), int(n["big_tpb"])
        levels = n["big_level_sizes"]
        last = len(levels) - 1
        ring_l = last / top                                   # real levels per toy ring (400)
        busy = [nb.busy_blocks(s, blocks, tpb) for s in levels]
        l_slow = next(L for L, b in enumerate(busy) if b > SLOW_BUSY) - 1

        # the floors: red, the 48-block mosaic (painted up to a ring), its ghost
        own = owner_fine("c4")
        kf = own.shape[0] // sq.shape[0]
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)))
        final = floor_raster("square_own48", f.rpx)
        full = owner_colors(sq, lv, own, top=top)
        ghost = (full.astype(np.float32) * (1 - GHOST) + fl.rgb(fl.EMPTY).astype(np.float32) * GHOST).astype(np.uint8)

        def mosaic(upto: int) -> np.ndarray:
            return fl.toy_raster(level_colors(sq, lv, upto=upto, top=top), f.rpx,
                                 fine=owner_colors(sq, lv, own, upto=upto, top=top))

        cell = crop_cell("c4", "ring")                        # the lens's cell, on the widest ring
        assert int(lv[cell]) == round(nb.owner_map()["c4"]["crops"]["ring"]["level"] / ring_l), cell

        def replay(r: int, lit: bool = False) -> np.ndarray:
            """Rings up to r painted, ring r outlined, the rest a ghost of its
            paint; with `lit`, the lens's cell painted too (the lens shows it)."""
            far = np.repeat(np.repeat(lv > r, kf, 0), kf, 1)
            if lit:
                far[cell[0] * kf:(cell[0] + 1) * kf, cell[1] * kf:(cell[1] + 1) * kf] = False
            return fl.toy_raster(level_colors(sq, lv), f.rpx, ring=(lv == r),
                                 fine=np.where(far[..., None], ghost, full))

        # ---- "So": a fresh run; the carried caption leaves, the square goes back to red
        self.until(self.cue("So"))
        self.floor_blend(floor_raster("square_own2", f.rpx), red, frames=5)
        self.unsay(frames=4)

        # ---- "every SM": the 22 dark SMs light, in a ripple out from SMs 0 and 2
        self.until(self.cue("every"))
        anims = self.machine_anims(mc.with_rooms(b8.machine, "lit"), slide=0)
        name = {id(p): k for k, p in self.machine.parts.items()}
        by_sm = {int(name[id(a.mobject)][4:]): a for a in anims if name.get(id(a.mobject), "")[4:].isdigit()}
        rest = [a for a in anims if a not in by_sm.values()]
        self.play(LaggedStart(*[by_sm[i] for i in ripple_order(by_sm)], lag_ratio=0.12), *rest,
                  run_time=fr(RIPPLE))

        # ---- "forty-eight blocks, two on each": the cubes land in the measured order
        # (block b on SM big_sm_ids[b]), each in its hue. Blocks 0 and 1 are already
        # there: their wide cubes narrow to make room for the second cubes
        t0, t1 = self.cue("forty-eight"), self.cue("each", end=True)
        self.until(t0)
        old = self.machine.parts["teams"]
        land = replace(b9.machine, tokens=b8.machine.tokens)     # the queue stays until the replay ends
        self.machine_anims(land)                                  # only the teams change; they land below
        teams = self.machine.parts["teams"]
        cubes = cube_faces(teams, blocks)
        order = cube_blocks(ids)
        of = {b: cubes[c] for c, b in enumerate(order)}
        drops = [FadeIn(of[b], shift=DOWN * DROP) for b in range(2, blocks)]
        span = max(t1 - t0, 1.0)
        self.play(LaggedStart(AnimationGroup(FadeOut(old), FadeIn(of[0]), FadeIn(of[1])), *drops,
                              lag_ratio=(span / LAND - 1) / len(drops)), run_time=span)
        self.remove(*cubes)
        self.add(teams)             # after the floor: redrawn every frame, so the replay can recolour it

        # ---- "Eight and a half times the CPU": the square refills in one quick
        # pass, block by block mixed ("forty-eight" is its own word: no clash)
        tq = self.cue("Eight")
        self.until(tq)
        self.floor_play(lambda p: min(top, int(p * (top + 1))), mosaic, QUICK)
        self.say(ov.fmt_x(self.der["big_48_x"]), "kernel time")

        # ---- "Each colour is one block": the pass replays from the seed out. A cube is
        # bright while its block has work in the ring, dim while it has none (D6). Slow
        # over the small rings until "busy", fast through the widest band and on
        paint = CubePaint(cubes, pl)
        hues = mc.block_hues(blocks)

        def colours(nbusy: int) -> list[str]:
            s = mc.place_blocks(b9.machine, ids, hues=hues, busy=mc.block_states(nbusy, blocks))
            return [mc.cube_color(s, i, j) for i in range(24) for j in range(s.teams[i])]

        work = colours(blocks)
        cache: dict[int, list[str]] = {}

        def cols(nbusy: int) -> list[str]:
            if nbusy not in cache:
                cache[nbusy] = colours(nbusy)
            return cache[nbusy]

        t_in = self.cue("Each", after=tq)                     # not the "each" of "two on each"
        t_s, t_f, t_e = t_in + fr(INTRO), self.cue("busy") + SLOW_TAIL, self.cue("threads") - 0.15
        total = t_e - t_in

        def level_at(p: float):
            """("in", a) while going over to the replay, else the real level shown."""
            tau = p * total
            if tau < t_s - t_in:
                return ("in", tau / (t_s - t_in))
            if tau < t_f - t_in:
                return l_slow * (tau - (t_s - t_in)) / (t_f - t_s)
            return min(last, l_slow + (last - l_slow) * (tau - (t_f - t_in)) / (t_e - t_f))

        # the lens's cell is painted from "pixels" until the lens has folded back into
        # it (the second caption's fade: from "busy" - 2 frames, 2 out and 5 in)
        t_lit = (self.cue("pixels"), self.cue("busy") + fr(5))

        def fkey(p: float):
            L = level_at(p)
            if p >= 1:
                return "done"
            if isinstance(L, tuple):
                return ("in", round(L[1] * INTRO))
            return ("ring", min(top, int(L // ring_l)), t_lit[0] <= t_in + p * total < t_lit[1])

        start = replay(0)

        def frender(key):
            if key == "done":
                return final
            if key[0] == "in":
                a = key[1] / INTRO
                return (final.astype(np.float32) * (1 - a) + start.astype(np.float32) * a).astype(np.uint8)
            return replay(key[1], lit=key[2])

        def cubes_at(p: float) -> None:
            L = level_at(p)
            if isinstance(L, tuple):
                paint.set([mc.mix_hex(a, b, L[1]) for a, b in zip(work, cols(busy[0]))])
            else:
                paint.set(cols(busy[int(L)]))

        # ---- "they share one queue": the blocks with work in the ring load from the one
        # queue in VRAM (red packets up to their SMs) and store new entries down to its
        # rear (a packet in each block's hue), block 0 first, as r04's two did
        t_q0, t_q1 = self.cue("they", after=t_in), self.cue("pixels") - 0.05
        L_q = level_at((t_q0 - t_in) / total)
        nq = busy[0] if isinstance(L_q, tuple) else busy[int(L_q)]
        legs, bars = [], []
        for b in range(nq):
            o = LAG * b / max(nq - 1, 1)
            legs += [(mc.vram_path(pl, ids[b], FRONT[b % 2])[::-1], o, 0.42 + o),
                     (mc.vram_path(pl, ids[b], REAR[b % 2]), 0.5 + o, 0.92 + o)]
            bars += [mc.packet(PACKET_B, RED_PX), mc.packet(PACKET_B, hues[b])]
        for bar in bars:
            bar.set_fill(opacity=0).set_stroke(opacity=0)
        self.add(*bars)                 # after the floor: redrawn every frame

        def traffic(p: float) -> None:
            for bar, (path, a, z) in zip(bars, legs):
                s = (p - a) / (z - a)
                on = p < 1 and 0.0 <= s < 1.0
                if on:
                    bar.move_to(mc.point_along(path, s))
                op = min(1.0, 6 * s, 6 * (1 - s)) if on else 0.0
                bar.set_fill(opacity=op).set_stroke(opacity=0.6 * op)

        self.until(t_in)
        self.floor_play(fkey, frender, total)
        self.run(cubes_at, total)
        self.run(traffic, t_q1 - t_q0, start=t_q0)

        # ---- "their pixels mix": a lens grows out of a cell on the widest ring, the
        # real owner map at full resolution; it folds back with the next caption
        tgt = f.cell_centre(*cell)
        (lc, lr) = LENS[self.G.vertical]
        lens = ov.owner_lens(owner_crop("c4", "ring"), lc, lr, target=tgt)
        self.until(self.cue("pixels"))
        self.play(FadeIn(lens, scale=0.15, target_position=tgt), run_time=fr(8))
        self.remove(*bars)

        # ---- "busy": few blocks busy on the small rings (the replay still shows them)
        self.say("few busy", "on small rings", at=self.cue("busy") - fr(2),
                 extra=[FadeOut(lens, scale=0.15, target_position=tgt)])

        # ---- the queue drains as the replay reaches the corners
        self.until(t_e - fr(5))
        self.machine_to(b9.machine, frames=5)

        # ---- "We also tried other threads per block and block counts": the sweep card
        # grows out of the die (the launch's blocks); every cube lights again
        die = np.mean([mc.sm_point(pl, i) for i in range(24)], axis=0)
        card = ov.sweep_card(self.G, n["sweep_grid_ms"], nb.SWEEP_ROWS, nb.SWEEP_TPB, (blocks, tpb),
                             n["sweep_best_ms"], nb.sweep_bin)
        outline = card[1][0][-1]                  # the default cell's outline (sweep_card adds it last)
        for m in card.get_family():               # under the caption, which it crosses as it grows
            m.set_z_index(ov.Z_OVER + 1)
        self.until(self.cue("threads"))
        low = cols(busy[last])
        self.run(lambda p: paint.set([mc.mix_hex(a, b, p) for a, b in zip(low, work)]), fr(9))
        self.play(FadeIn(card, scale=0.12, target_position=die), run_time=fr(9))

        # ---- "fifteen": the carried caption; "We kept": the default cell pulses
        self.say_last(at=self.cue("fifteen") - fr(2))
        self.until(self.cue("kept"))
        self.play(outline.animate(rate_func=there_and_back).scale(1.3).set_stroke(width=6), run_time=fr(12))

        # ---- the card folds back into the die as the words end
        self.until(self.cue("fifty-six", end=True) - 0.2)
        self.play(FadeOut(card, scale=0.12, target_position=die), run_time=fr(8))
