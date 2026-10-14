"""r04: a second block, then the whole queue in global memory.

At "So" the carried caption leaves and the square goes back to red: a
fresh run. At "second block" block 1 (its green cube) drops onto SM 0,
squeezed in beside block 0; at "second SM" SM 2 lights, outlined: where
it will go (D5, measured: SM 2, not SM 1). "Squeezed onto one SM": SM 0
is outlined; at "slower" both cubes dim (ch02's pinned pair, 1.7x
slower, spoken only). As the clause ends block 1 hops to SM 2 and both
cubes light again.

"private": the two shelf cards (D1) grow out of SM 0 (its queue region
full of red tokens) and SM 2 (empty, as its tile's shelf now shows).
"the whole queue moves": SM 0's tokens drain down its leader, into SM 0,
through the memory controller into VRAM, beside the spill; the cards
fold back, the shelves and the fence go, the queue's slots and its label
come up under the VRAM chips and the gate rail appears at rest.

"address": the refill, on r04's own plan (r03's sweep_plan, from
"address" to "fast"). The floor is the 2-block owner mosaic (D4): each
painted cell shows its real pixels' owners, blue and green mixed, never
two halves. Every ring loads from the VRAM queue up to SM 0 and SM 2
(red packets) and stores new entries down to its rear (a blue packet
from SM 0, a green one from SM 2). The first step stops at the last real
ring with one block's work (real rings 0-64): only block 0's seed dots
are painted, and block 1's cube stays dim until its block has work (D6).
"no slower" / "16 Mpx square" at the second "slower"; "2x again" /
"kernel time" at "twice". At "twenty-two" the 22 dark SMs ripple idle,
the carried "22 SMs idle" lands on "SMs", and at "dark" they go dark.

Starts on chain_state(7) and ends on chain_state(8).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, FadeIn, FadeOut, Group, ImageMobject, Polygon, ReplacementTransform, smooth

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import numbers as nb
from scenes.chain.base import ChainScene, _norm, fr
from scenes.chain.iso import plane_corners
from scenes.chain.r03_giant import F15, plan_index, sweep_plan
from scenes.chain.state import (OWNER_FLOORS, boundary, floor_raster, level_colors, owner_colors, owner_fine,
                                toys)
from scenes.style import INK, RED_PX

DROP = 0.7              # how far above SM 0 block 1's cube starts its drop (frame units)
LIFT = 40.0             # a moving cube drawn above the machine's top layer (SM 0's fence) as it passes it
HOP_ARC = -2.0          # block 1's hop from SM 0 to SM 2: a clockwise arc (up and over)
PULSE = 1.0             # seconds of an SM's white outline pulse
DRAIN = 1.4             # seconds of SM 0's tokens draining into VRAM
DRAIN_TRIP = 0.55       # each token's share of that (the rest is the lag across them)
CYCLE = 0.5             # seconds of a load-then-store cycle once the rings run fast
GLOW = 0.6              # seconds of the idle ripple over the 22 dark SMs
GLOW_W = 0.4            # each SM's share of that ripple (the rest is the lag across them)
PACKET_B = 128          # a packet: one warp's 32 queue entries of 4 B, a dense read or write
# Each packet's window in a load-then-store cycle (fractions): block 0 first,
# block 1 a little behind it (their paths share the memory controller).
WINDOWS = {"load0": (0.0, 0.42), "load1": (0.08, 0.5), "store0": (0.5, 0.92), "store1": (0.58, 1.0)}
# Queue slots the packets use: the front (being read) in VRAM chip 0, the
# rear (being written) in chip 1.
FRONT, REAR = (1, 2), (4, 5)
DRAIN_SLOTS = (3, 4, 5)  # where SM 0's drained tokens land: chip 1, beside the spill in chip 0


def r04_window(words: list) -> tuple[float, float]:
    """r04's refill: from "address" to just after "fast" starts. Words are
    matched like ChainScene.cue; a voice without them gets a 3 s refill."""
    def at(w: str) -> float | None:
        hits = [x for x in words if _norm(x[0]) == _norm(w)]
        return float(hits[0][1]) if hits else None
    t0, t1 = at("address"), at("fast")
    if t0 is None or t1 is None or t1 + 0.1 <= t0:
        t0 = t0 if t0 is not None else 0.0
        return t0, t0 + 3.0
    return t0, t1 + 0.1


def r04_plan(words: list, rings: int) -> list[int]:
    """The ring shown on each 15 fps frame of the refill: r03's sweep_plan
    (a slow opening, then one ring per step), stretched to fill the window.
    Two blocks sync at the gate after every ring, but here it stays at
    rest (r05b names it), so the plan's wait frames are each ring's tail."""
    t0, t1 = r04_window(words)
    frames = max(int((t1 - t0) * F15), rings)
    base = [i for i, _ in sweep_plan(rings, frames)]
    return [base[f * len(base) // frames] for f in range(frames)]


def cube_faces(m: mc.Machine) -> list[Group]:
    """A copy of each cube's faces in m's teams part, in build order (by SM,
    the left cube first). Every box of a build shows the same faces."""
    part, n = m.part("teams"), len(m.keys.get("teams", ()))
    k = len(part.submobjects) // max(n, 1)
    return [Group(*[f.copy() for f in part.submobjects[j * k:(j + 1) * k]]) for j in range(n)]


def lifted(g: Group, dz: float = LIFT) -> Group:
    for f in g.submobjects:
        f.set_z_index(f.z_index + dz)
    return g


def show_fill(g, a: float) -> None:
    """Fade a group of box faces (no stroke) or packets: a Group's set_opacity
    only sets an attribute, so each member's fill is set."""
    for m in g.family_members_with_points():
        m.set_fill(opacity=a)
        if m.get_stroke_width() > 0:
            m.set_stroke(opacity=0.6 * a)


def sm_outline(pl, i: int) -> Polygon:
    """A white outline round SM i on the die's top face (the new-part convention)."""
    r = mc.sm_rect(i)
    ul, ur, dl, dr = plane_corners(pl, r.x - 0.03, r.y - 0.03, r.x1 + 0.03, r.y1 + 0.03, mc.DIE_Z)
    p = Polygon(ul, ur, dr, dl, stroke_color=INK, stroke_width=3, fill_opacity=0)
    p.set_z_index(mc.Z_PACKET - 1)
    return p


def depth_fine(key: str) -> np.ndarray:
    """owner_fine's twin for the real depth (ring) of each sampled pixel, -1 off the square."""
    own = nb.owner_map()
    g = own["grid"]
    k, o, side = int(g["dots_per_cell"]), int(g["toy_offset"]), int(g["toy_side"])
    n = toys()["square"].shape[0]
    out = np.full((n * k, n * k), -1, np.int32)
    out[o * k:(o + side) * k, o * k:(o + side) * k] = own[key]["depth_samples"]
    return out


def images(part: Group) -> list:
    return [m for m in part.get_family() if isinstance(m, ImageMobject)]


class TwoBlocks(ChainScene):
    k = 7

    def cubes_to(self, spec: mc.Spec, moves, frames: int, extra=()) -> None:
        """machine_to(spec) with the cubes drawn by moves(old, new) -> anims,
        on copies of each cube; the new build's part replaces them after."""
        old_part = self.machine.part("teams")
        old = cube_faces(self.machine)
        anims = self.machine_anims(spec, slide=0)
        new_part = self.machine.part("teams")
        new = cube_faces(self.machine)
        anims = [a for a in anims if a.mobject is not old_part and a.mobject is not new_part]
        self.remove(old_part)
        self.add(*old)
        self.play(*anims, *moves(old, new), *extra, run_time=fr(frames))
        self.remove(*old, *new)
        self.add(new_part)

    def pulse(self, mob, seconds: float = PULSE) -> None:
        """mob's stroke up and back down over `seconds`, from now."""
        mob.set_stroke(opacity=0)
        self.add(mob)
        self.run(lambda p: mob.set_stroke(opacity=min(1.0, 2.5 * float(np.sin(np.pi * p)))), seconds)

    def beat_anim(self) -> None:
        t = toys()
        sq, lv = t["square"], t[OWNER_FLOORS["square_own2"][1]].level
        top = int(lv.max())
        f = self.floor
        b7, b8 = boundary(self.k), boundary(self.k + 1)
        pl = self.Lay.machine_pl
        n = self.num
        ids = tuple(int(i) for i in n["two_sm_ids"])             # (0, 2): where the blocks ran (D5)
        hues = mc.block_hues(len(ids))
        painted = floor_raster("square4", f.rpx)
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)))

        def blocks(spec, sms, busy=None):
            return mc.place_blocks(spec, sms, hues=hues[:len(sms)], busy=busy)

        # ---- "So": a fresh run; the carried caption leaves, the square goes back to red
        self.until(self.cue("So"))
        self.floor_blend(painted, red, frames=5)
        self.unsay(frames=4)

        # ---- "second block": block 1 drops onto SM 0, squeezed in beside block 0
        squeezed = blocks(b7.machine, (ids[0], ids[0]))
        self.until(max(self.cue("second"), self.elapsed()))
        self.cubes_to(squeezed, lambda o, w: [ReplacementTransform(o[0], w[0]),
                                              FadeIn(lifted(w[1]), shift=DOWN * DROP)], frames=8)

        # ---- "second SM": SM 2 lights, outlined: where block 1 will go
        squeezed = mc.with_rooms(squeezed, "lit", ids[1:])
        out2, out0 = sm_outline(pl, ids[1]), sm_outline(pl, ids[0])
        self.until(max(self.cue("second", n=2), self.elapsed()))
        self.pulse(out2)
        self.machine_to(squeezed, frames=5, slide=0)

        # ---- "Squeezed onto one SM, two blocks ran one point seven times slower"
        # (ch02's pinned pair, spoken only): SM 0 outlined, then both cubes dim
        self.until(max(self.cue("one"), self.elapsed()))
        self.pulse(out0)
        dimmed = blocks(squeezed, (ids[0], ids[0]), busy=("dim", "dim"))
        self.until(max(self.cue("slower"), self.elapsed()))
        self.cubes_to(dimmed, lambda o, w: [ReplacementTransform(o[j], w[j]) for j in range(2)], frames=4)

        # ... and as the clause ends block 1 hops to SM 2; both light again
        spread = blocks(squeezed, ids)
        self.until(max(self.cue("slower", end=True), self.elapsed()))
        self.cubes_to(spread, lambda o, w: [ReplacementTransform(o[0], w[0]),
                                            ReplacementTransform(lifted(o[1]), lifted(w[1]), path_arc=HOP_ARC)],
                      frames=8)
        self.remove(out0, out2)

        # ---- "private": each block's shared memory, a card grown out of its SM:
        # SM 0's region full of its queue, SM 2's empty (its tile's shelf too)
        (c0, c2), size = self.G.cards
        kw = dict(slots=int(n["ring_slots"]), shared_bytes=int(n["shared_kb"] * 1024))
        card0 = mc.shelf_card(pl, c0, size, ids[0], True, **kw)
        card2 = mc.shelf_card(pl, c2, size, ids[1], False, **kw)
        home0, home2 = mc.sm_point(pl, ids[0]), mc.sm_point(pl, ids[1])
        private = replace(spread, shelves=tuple("full" if i == ids[0] else "empty" if i == ids[1] else "off"
                                                for i in range(max(ids) + 1)))
        self.until(max(self.cue("private"), self.elapsed()))
        self.machine_to(private, frames=8, slide=0,
                        extra=[FadeIn(card0, target_position=home0, scale=0.08),
                               FadeIn(card2, target_position=home2, scale=0.08)])

        # ---- "the whole queue moves": SM 0's tokens drain down its leader, into
        # SM 0, then through the memory controller into VRAM, beside the spill
        # the tokens nearest the leader leave first, so none crosses another on its way out
        lead = card0.card[2].get_start()
        toks = sorted(card0.tokens, key=lambda tk: float(np.linalg.norm(tk.get_center() - lead)))
        paths = [[tk.get_center(), lead, *mc.vram_path(pl, ids[0], DRAIN_SLOTS[j % len(DRAIN_SLOTS)])]
                 for j, tk in enumerate(toks)]

        def drain(p: float) -> None:
            for j, (tk, path) in enumerate(zip(toks, paths)):
                s = (p - (1 - DRAIN_TRIP) * j / max(len(toks) - 1, 1)) / DRAIN_TRIP
                tk.move_to(mc.point_along(path, smooth(min(max(s, 0.0), 1.0))))

        self.until(max(self.cue("whole"), self.elapsed()))
        self.run(drain, DRAIN)
        self.until(self.elapsed() + DRAIN)
        # drained: the tokens stay on their slots in VRAM chip 1, the cards fold back without them
        drained = card0.tokens
        card0.body.remove(drained)
        self.add(drained)

        # ... the cards fold back into their SMs; the shelves and the fence go; the
        # queue's slots and label come up in VRAM and the gate rail appears at rest.
        # The queue's tokens are the same entries: the spill stays under them (it then
        # goes, no cross-fade) and the drained tokens hand over to them in chip 1
        spill = self.machine.part("spill")
        anims = [a for a in self.machine_anims(b8.machine, slide=0) if a.mobject is not spill]
        self.play(*anims, FadeOut(card0, target_position=home0, scale=0.08),
                  FadeOut(card2, target_position=home2, scale=0.08), FadeOut(drained), run_time=fr(8))
        self.remove(spill)

        # ---- "every new pixel's address": the refill on r04's own plan. The floor is the
        # owner map: each dot one real pixel, the block that painted it
        own = owner_fine(OWNER_FLOORS["square_own2"][0])
        k = own.shape[0] // sq.shape[0]
        lvf = np.repeat(np.repeat(lv, k, 0), k, 1)
        depth = depth_fine(OWNER_FLOORS["square_own2"][0])
        alone = int(self.der["block0_only_rings"]) - 1     # 64: the last real ring with one block's work
        first = (lvf == 0) & (depth > alone)                 # the seed cell's dots the first step leaves red
        # block 1 works in a step if the step paints any of its dots (none in the first: D6)
        works = []
        for i in range(top + 1):
            dots = (lvf == i) & (own == 1)
            works.append(bool((dots & ~first).any() if i == 0 else dots.any()))
        plan = r04_plan(self.words, top + 1)
        total = len(plan) / F15
        t0 = self.elapsed() if not self.words else max(r04_window(self.words)[0], self.elapsed())

        def key(p: float):
            j = plan_index(p, total)
            return plan[j] if (j < len(plan) and p < 1) else "done"

        def render(i):
            if i == "done":
                return floor_raster("square_own2", f.rpx)
            fine = owner_colors(sq, lv, own, upto=i, top=top)
            if i == 0:
                fine[first] = fl.rgb(RED_PX)
            return fl.toy_raster(level_colors(sq, lv, upto=i, top=top), f.rpx, ring=lv == i, fine=fine)

        # each step's span in seconds; the slow opening's rings get a load-then-store
        # cycle each, the fast rings a steady stream of cycles to the end
        start = {}
        for j, i in enumerate(plan):
            start.setdefault(i, j / F15)
        counts = [plan.count(i) for i in range(top + 1)]
        slow = next((i for i, c in enumerate(counts) if c <= 1.5 * min(counts)), top + 1)   # the slow opening's rings
        t_fast = start.get(slow, total)
        n_cyc = max(1, round((total - t_fast) / CYCLE))
        cyc = (total - t_fast) / n_cyc

        def phase(tt: float, ring: int) -> float:
            if ring < slow:
                end = start.get(ring + 1, total)
                return (tt - start[ring]) / max(end - start[ring], 1e-6)
            return ((tt - t_fast) / cyc) % 1.0 if tt < total else 1.0

        legs = {"load0": mc.vram_path(pl, ids[0], FRONT[0])[::-1], "load1": mc.vram_path(pl, ids[1], FRONT[1])[::-1],
                "store0": mc.vram_path(pl, ids[0], REAR[0]), "store1": mc.vram_path(pl, ids[1], REAR[1])}
        colors = {"load0": RED_PX, "load1": RED_PX, "store0": hues[0], "store1": hues[1]}
        packets = {name: mc.packet(PACKET_B, colors[name]) for name in legs}
        dim1 = lifted(cube_faces(mc.build(blocks(b8.machine, ids, busy=("work", "dim")), pl))[1], 0.05)
        for g in (*packets.values(), dim1):
            show_fill(g, 0.0)
        self.add(*packets.values(), dim1)

        def traffic(p: float) -> None:
            tt = p * total
            j = min(int(tt * F15), len(plan) - 1)
            ring = plan[j]
            u = phase(tt, ring) if p < 1 else 1.0
            show_fill(dim1, 1.0 if (p < 1 and not works[ring]) else 0.0)
            for name, bar in packets.items():
                a, b = WINDOWS[name]
                s = (u - a) / (b - a)
                on = p < 1 and 0.0 <= s < 1.0 and (name.endswith("0") or works[ring])
                if on:
                    bar.move_to(mc.point_along(legs[name], s))
                show_fill(bar, min(1.0, 6 * s, 6 * (1 - s)) if on else 0.0)

        self.until(t0)
        self.floor_play(key, render, total)
        self.run(traffic, total)

        # ---- "On big blobs, measured, that's no slower": one block, same session, the
        # 16 Mpx square, where the shared ring never spilled
        self.say("no slower", f"{round(n['q1_px'] / 1e6)} Mpx square", at=self.cue("slower", n=2))
        # one block's kernel time over two blocks': "kernel time", as r03b's and r05's ratios say
        self.say(f"{round(self.der['big_2v1_x'])}× again", "kernel time", at=self.cue("twice"))
        self.until(max(t0 + total, self.elapsed()))
        self.remove(*packets.values(), dim1)

        # ---- "twenty-two SMs stay dark": they ripple idle (nearest SM 2 first), then go dark again
        rest = [i for i in range(int(n["sm"])) if i not in ids]
        idle_m = mc.build(mc.with_rooms(b8.machine, "idle", rest), pl)
        c2p = mc.sm_point(pl, ids[1])
        order = sorted(rest, key=lambda r: float(np.linalg.norm(mc.sm_point(pl, r) - c2p)))
        idle = [idle_m.part(f"room{r}") for r in order]

        def glow(p: float) -> None:
            for j, tile in enumerate(idle):
                a = (p - (1 - GLOW_W) * j / (len(idle) - 1)) / GLOW_W
                for im in images(tile):
                    im.set_opacity(min(max(a, 0.0), 1.0))

        glow(0.0)
        self.add(*idle)
        self.run(glow, GLOW, start=self.cue("twenty-two"))
        self.extras = list(idle)

        self.say_last(at=self.cue("SMs"))
        self.until(max(self.cue("dark"), self.elapsed()))
        self.play(FadeOut(Group(*idle)), run_time=fr(7))
        self.extras = []
