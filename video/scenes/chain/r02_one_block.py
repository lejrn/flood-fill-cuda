"""r02: one block on one SM, its queue in the SM's shared memory. At "So"
the floor splits into two half floors, the same red drone on each; at
"onto" a block (the blue cube) drops onto SM 0, which lights; at "SM"
that SM grows out of the chip over the idle split floor, drawn large
like NVIDIA's SM figure (`sm_closeup`: "32" on each of its 4 partitions,
the shared-memory strip "full", the queue's red). At "queue" a few red
entries drop from the partitions into the strip; at "shared memory" the
strip gets the white outline pulse (a new part). Just before "paint" it
flies back into SM 0, whose shelf is "full" from then on (the queue
lives there). The caption "256 threads" and the race's floor labels
("CPU", "GPU", a step counter each) land together during that flight,
one text change: the floor labels at "So" would sit under a second of
the close-up's text. At "block" (v5: "workers, called threads, in one
block") the cube on SM 0 gets the white outline pulse: that cube is the
block. Text lands at least MIN_DWELL apart. The race in steps starts at
"paint", on one clock: the CPU paints one pixel per step (left), the
block one whole 4-way ring per step (right). The right side finishes at
its ring count; the left side then speeds up and finishes too.

Then the race in time, short, on a 16 K px test blob: a small card
between the strip and the split floor, the CPU's bar (its 4-neighbour
BFS) and the "1 block" bar (this very kernel, one block, its ring in
shared memory) on one scale, growing at one speed, so the CPU stops
first. "CPU wins" lands, the card folds into the strip's first pill
("CPU 4.3×"), the floor labels fade, and at "size" an outline marks the
next pill ("huge blob"), the next fix.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (DL, DOWN, LEFT, UP, FadeIn, FadeOut, GrowFromEdge, Group, LaggedStart, Rectangle, RoundedRectangle,
                   SurroundingRectangle, UpdateFromAlphaFunc, VGroup, VMobject, linear, rush_into, smooth,
                   there_and_back)

from scenes.chain import die as dz
from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, caption_group, fr
from scenes.chain.state import Z_FLOOR, Z_OVER, boundary, floor_raster, fmt_ms, level_colors, toys
from scenes.style import BLUE, GREY, GRID, INK, INK_SOFT, RED_PX, label

# The SM close-up: (centre of the SM picture, its height), the title and
# sub to its left. In 9:16 it sits in the band between the caption and
# the GPU board's back edge, left of SM 0; in 16:9, in the empty band
# above the CPU board. Both keep the CPU board (it races too) in view.
CLOSEUP = {True: 3.7, False: 3.5}          # the close-up's height, 9:16 and 16:9
ENTRIES = 4                                # queue entries that drop into the strip (schematic, no number)
CARD = "#0f141a"                           # the close-up's card: the gaps between entries
PULSE = 12                                 # frames of the block's outline pulse at "block"


class Counter(VGroup):
    """A mono line whose text changes from a floor hook. The Cairo renderer
    snapshots the family when a play starts, so the replaced glyphs are
    blanked instead of removed (scenes.s_intro.LiveText's trick)."""

    def __init__(self, text: str, size: int, color: str, centre: np.ndarray, z: float):
        super().__init__()
        self.size, self.color_, self.centre, self.z = size, color, centre, z
        self.value = self._make(text)
        self.add(self.value)
        self.add_updater(lambda m, dt: None)        # drawn every frame, never cached

    def _make(self, text: str):
        m = label(text, size=self.size, color=self.color_, mono=True).move_to(self.centre)
        for x in m.get_family():
            x.set_z_index(self.z)
        return m

    def set(self, text: str) -> None:
        if text == self.value.text:
            return
        old, new = self.value, self._make(text)
        self.remove(old)
        self.value = new
        self.add(new)
        for m in old.get_family():
            m.points[:] = 0


def _z(mob, z: float):
    for m in mob.get_family():
        m.set_z_index(z)
    return mob


def span(a: float, b: float):
    """A rate function that runs `smooth` over the fraction [a, b] of a play
    only: several changes in one play, each on its own word."""
    return lambda t: smooth(min(1.0, max(0.0, (t - a) / max(b - a, 1e-6))))


def sm_callout(L) -> Group:
    """mc.sm_closeup over the floor area (the split floors are still at step
    0): SM 0 drawn large, its 4 partitions ("32") and shared memory labelled,
    the strip "full" (the queue's red: the queue lives there)."""
    h = CLOSEUP[L.G.vertical]
    return mc.sm_closeup(L.machine_pl, L.G.floor_c, h, i=0, shelf="full")


def strip_rect(sm: Group) -> tuple[np.ndarray, float, float, float]:
    """The close-up's shared-memory strip in frame units: (centre, width,
    height, the free width left of its label). The same arithmetic as
    die._sm and die.sm_detail: inset 0.018; RT core 17%, texture 5%, the
    strip 7% of the inner depth, 0.008 between rows; the label 0.022 high."""
    from PIL import ImageFont

    img = sm[1]
    sx, sy = img.width / dz.SM_W, img.height / dz.SM_D
    w, d = dz.SM_W - 0.036, dz.SM_D - 0.036
    y0 = 0.018 + d * (0.17 + 0.05) + 2 * 0.008
    h = d * 0.07
    font = ImageFont.truetype(dz.FONT_DIR + "DejaVuSans.ttf", 1000)
    text_w = 0.022 * font.getlength("shared memory") / 1000
    c = img.get_corner(DL) + np.array([(0.018 + w / 2) * sx, (y0 + h / 2) * sy, 0.0])
    return c, w * sx, h * sy, (w - text_w) / 2 * sx


def strip_entries(sm: Group, z: float) -> VGroup:
    """ENTRIES queue entries in the strip, from its left end up to its label:
    red cells parted by the card's dark, so each reads as one entry on the
    red strip."""
    c, w, h, free = strip_rect(sm)
    pitch = (free - 0.22) / ENTRIES                 # a clear gap before the label
    x0 = c[0] - w / 2 + 0.04
    g = VGroup(*[Rectangle(width=pitch * 0.78, height=h * 0.78, fill_color=RED_PX, fill_opacity=1,
                           stroke_color=CARD, stroke_width=1.6).move_to([x0 + (k + 0.5) * pitch, c[1], 0])
                 for k in range(ENTRIES)])
    return _z(g, z)


def fall(entry, rise: float) -> UpdateFromAlphaFunc:
    """An entry appears `rise` above its place, over the partitions (a thread
    writes it), and falls into the strip, speeding up."""
    end = entry.get_center().copy()

    def step(m, a: float) -> None:
        m.move_to(end + UP * rise * (1 - rush_into(a)))
        m.set_fill(opacity=min(1.0, 4 * a))
        m.set_stroke(opacity=min(1.0, 4 * a))

    return UpdateFromAlphaFunc(entry, step, rate_func=linear)


def block_edge(cube: Group) -> VGroup:
    """The block's cube outlined in white (its faces' edges), hidden: the
    pulse at "block" fades it in and out."""
    faces = [f for f in cube.submobjects if isinstance(f, VMobject)]
    z = max(f.z_index for f in faces) + 0.5
    g = VGroup(*[f.copy().set_fill(opacity=0).set_stroke(INK, 2.5, opacity=0) for f in faces])
    return _z(g, z)


def strip_outline(sm: Group, z: float) -> Rectangle:
    """The white outline round the strip (the new-part pulse), hidden."""
    c, w, h, _ = strip_rect(sm)
    o = Rectangle(width=w + 0.08, height=h + 0.08, stroke_color=INK, stroke_width=3.5).move_to(c)
    o.set_stroke(opacity=0)
    return _z(o, z)


def time_card(n: dict, halves: list, ins, strip):
    """The race in time on the 16 K px test blob, like for like: the CPU's
    4-neighbour BFS and "1 block" (r02's own kernel: one block, its ring in
    shared memory) on one scale (the block's bar is full width). Placed
    between the inset and the floor's right edge, the split floor and the
    strip. Returns (card, bars)."""
    rows = [("CPU", n["small_cpu_ms"], GREY), ("1 block", n["small_gpu_ms"], BLUE)]
    vmax = n["small_gpu_ms"]
    bw = 1.6
    lines, bars = VGroup(), []
    for name, v, col in rows:
        bar = Rectangle(width=max(bw * v / vmax, 0.04), height=0.16, fill_color=col, fill_opacity=1, stroke_width=0)
        lines.add(VGroup(label(name, 16, INK_SOFT), bar, label(fmt_ms(v), 16, INK, mono=True)))
        bars.append((bar, v))
    wn = max(r[0].width for r in lines)
    for i, r in enumerate(lines):
        y = -i * 0.34
        r[0].move_to([wn / 2, y, 0])
        r[1].move_to([wn + 0.2 + r[1].width / 2, y, 0])
        r[2].move_to([wn + 0.2 + bw + 0.15 + r[2].width / 2, y, 0])
    body = VGroup(label("16 K px test blob", 18, INK), lines,
                  label("GPU: kernel time", 13, INK_SOFT, mono=True)).arrange(DOWN, buff=0.16)
    box = RoundedRectangle(corner_radius=0.12, width=body.width + 0.4, height=body.height + 0.3,
                           fill_color=ov.PANEL, fill_opacity=0.97, stroke_color=GRID, stroke_width=2)
    card = VGroup(box, body)
    body.move_to(box)
    x0 = ins.get_right()[0] if ins is not None else halves[0].image.get_left()[0]
    x1 = halves[1].image.get_right()[0]
    y0 = max(f.image.get_top()[1] for f in halves)
    y1 = strip.get_bottom()[1] if strip is not None else y0 + card.height + 0.4
    card.move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])
    return _z(card, Z_OVER + 5), bars


class OneBlock(ChainScene):
    k = 4

    def beat_anim(self) -> None:
        t = toys()
        drone, fill = t["drone"], t["drone4"]
        order, level = fill.order, fill.level
        total, top = len(order), int(level.max())
        rings = top + 1
        b4, b5 = boundary(self.k), boundary(self.k + 1)
        n = self.num
        zt = Z_OVER - 10

        # ---- "So": the drone on two half floors, reset to red; "CPU" and "GPU" under them
        old = self.floors[0]
        halves = [fl.Floor(self.Lay.split_pl[s], z0=Z_FLOOR) for s in ("left", "right")]
        red = level_colors(drone, np.full(drone.shape, -1))
        for f in halves:
            f.show_raster(fl.toy_raster(red, f.rpx))
        left, right = halves

        def under(f, dy):
            im = f.image
            return np.array([im.get_center()[0], im.get_bottom()[1] - dy, 0.0])

        names = _z(VGroup(label("CPU", 18, INK).move_to(under(left, 0.2)),
                          label("GPU", 18, INK).move_to(under(right, 0.2))), zt)
        cl = Counter("step 0", 16, INK_SOFT, under(left, 0.5), zt)
        cr = Counter("step 0", 16, INK_SOFT, under(right, 0.5), zt)
        notes = [names, cl, cr]

        # (the floor labels wait for the race: here they would sit under a
        # second of the close-up's text)
        self.until(self.cue("So"))
        anims = [FadeOut(old), FadeIn(left), FadeIn(right)]
        if self.cap is not None:
            anims.append(FadeOut(self.cap))
            self.cap = None
        self.set_floors(halves)
        self.play(*anims, run_time=fr(6))

        # ---- "onto": a block drops onto SM 0, which lights under it (no queue on its shelf yet)
        self.until(self.cue("onto"))
        lit = replace(b5.machine, shelves=())
        drop = self.machine_anims(replace(lit, rooms=b4.machine.rooms), slide=0.3)
        light = self.machine_anims(lit, slide=0)
        self.play(*drop, *light, run_time=fr(6))

        # ---- "SM": that SM grows out of the chip, drawn large, its shared-memory strip the queue's red
        sm = sm_callout(self.Lay)
        home = mc.sm_point(self.Lay.machine_pl, 0)
        t_sm = max(self.cue("SM"), self.elapsed())
        self.until(t_sm)
        self.play(FadeIn(sm, target_position=home, scale=0.08), run_time=fr(8))

        # ---- "queue": a few entries drop from the partitions into the strip
        z_sm = sm[1].z_index
        entries = strip_entries(sm, z_sm + 1)
        self.until(max(self.cue("queue"), self.elapsed()))
        self.play(LaggedStart(*[fall(e, 0.38) for e in entries], lag_ratio=0.3), run_time=fr(12))

        # ---- "shared memory": the strip's white outline pulse (a new part)
        ring = strip_outline(sm, z_sm + 2)
        self.add(ring)
        t_pulse = max(self.cue("shared"), self.elapsed())
        self.until(t_pulse)
        pulse = min(max(self.cue("memory", end=True) - t_pulse, fr(10)), fr(14))
        self.play(UpdateFromAlphaFunc(ring, lambda m, a: m.set_stroke(opacity=there_and_back(a))),
                  run_time=pulse)
        self.remove(ring)

        # ---- back into SM 0 just before "paint", where the race starts: SM 0's
        # shelf turns "full" as it lands. "256 threads" and the race's floor
        # labels land at "threads", in the same play: one text change. The
        # flight starts 6 frames before them, so in 9:16 the card has crossed
        # the caption band when the caption comes in.
        t_cap = max(self.cue("threads"), t_sm + MIN_DWELL)
        t0 = max(t_cap - fr(6), self.elapsed())
        self.until(t0)
        t0 = self.elapsed()
        frames = max(8, round((max(t_cap, t0) - t0) * 15) + 5)
        a_back = 8 / frames                             # the flight: the play's first 8 frames
        a_cap = 1 - 5 / frames                          # the text: its last 5
        cap = caption_group(self.G, f"{int(n['small_tpb'])} threads")
        shelf = self.machine_anims(b5.machine, slide=0)
        for anim in shelf:
            anim.set_rate_func(span(0.6 * a_back, a_back))
        self.play(FadeOut(Group(sm, entries), target_position=home, scale=0.08, rate_func=span(0.0, a_back)),
                  *shelf, FadeIn(cap, scale=1.06, rate_func=span(a_cap, 1.0)),
                  *[FadeIn(m, rate_func=span(a_cap, 1.0)) for m in notes],
                  run_time=fr(frames))
        self.cap, self._cap_t = cap, self.elapsed()

        # ---- "block": the cube on SM 0 gets the white outline pulse (the
        # new-part convention): that cube is the block the words define. A
        # hook, so the race still starts on "paint"
        edge = block_edge(self.machine.parts["teams"])
        self.add(edge)                  # after the floors: redrawn every frame
        self.run(lambda p: edge.set_stroke(opacity=there_and_back(p)), fr(PULSE),
                 start=max(self.cue("block"), self.elapsed()))

        # ---- the race in steps, on one clock
        t_r0 = max(self.cue("paint"), self.elapsed())
        t_r1 = max(self.cue("once"), t_r0 + 1.2)        # the right side's last ring
        dt = (t_r1 - t_r0) / (rings - 1)
        # the left side, sped up, done before the card (and never in under 0.4 s)
        t_l1 = max(t_r1 + 0.4, min(t_r1 + 0.9, self.cue("But") - fr(2)))
        gpu_done = floor_raster("drone_gpu", right.rpx)
        cpu_done = floor_raster("drone_cpu", left.rpx)

        def ring_at(p: float) -> int:
            tt = t_r0 + p * (t_l1 - t_r0)
            return min(rings - 1, int((tt - t_r0) / dt + 1e-6))

        def pixels_at(p: float) -> int:
            tt = t_r0 + p * (t_l1 - t_r0)
            if tt < t_r1:
                return int((tt - t_r0) / dt + 1e-6) + 1
            u = min(1.0, (tt - t_r1) / (t_l1 - t_r1))
            return min(total, rings + int(round((total - rings) * u ** 1.5)))

        def render_right(i: int):
            cr.set(f"step {i + 1}")
            if i == rings - 1:
                return gpu_done
            lv = np.where((level >= 0) & (level <= i), level, -1)
            ring = level == i
            return self.toy(level_colors(drone, lv, top=top), ring=ring, floor=right)

        def render_left(k: int):
            cl.set(f"step {k}")
            if k == total:
                return cpu_done
            lv = np.full(drone.shape, -1)
            for (r, c) in order[:k]:
                lv[r, c] = level[r, c]
            dots = np.zeros(drone.shape, int)
            dots[order[k - 1]] = 1
            return self.toy(level_colors(drone, lv, top=top), dots=dots, floor=left)

        self.until(t_r0)
        self.floor_play(ring_at, render_right, t_l1 - t_r0, floor=right)
        self.floor_play(pixels_at, render_left, t_l1 - t_r0, floor=left)

        # ---- "But": the race in time, on a 16 K px test blob: the CPU stops first
        card, bars = time_card(n, halves, self.over.get("inset"), self.over.get("strip"))
        for bar, _ in bars:
            bar.set_opacity(0)
        t_card = max(self.cue("But"), self.elapsed())
        self.until(t_card)
        self.play(FadeIn(card), run_time=fr(5))         # the bars stay hidden: they grow next
        for bar, _ in bars:
            bar.set_opacity(1)
        vmax = n["small_gpu_ms"]
        # one growth speed: the CPU bar stops at v / vmax of the run, the GPU bar at its end
        grow = [GrowFromEdge(bar, LEFT, rate_func=(lambda tt, f=v / vmax: min(1.0, tt / f)))
                for bar, v in bars]
        self.play(*grow, run_time=fr(12))

        # ---- "the CPU still wins": the carried caption; the card folds into the strip's first pill
        t_cap = max(self.cue("CPU"), t_card + MIN_DWELL)
        self.say_last(at=t_cap)
        strip = self.strip(b5.rows)
        pill = strip[0].copy()
        self.until(max(self.cue("wins") + 0.4, t_cap + MIN_DWELL))
        self.play(FadeOut(card, target_position=pill.get_center(), scale=0.3), FadeIn(pill),
                  *[FadeOut(m) for m in notes], run_time=fr(8))
        # the other pills did not change: swap in the whole strip without a cross-fade
        self.remove(self.over["strip"], pill)
        self.add(strip)
        self.over["strip"] = strip

        # ---- "So: size first": the next pill is the next fix
        o = _z(SurroundingRectangle(strip[1], color=INK, buff=0.05, corner_radius=0.12, stroke_width=2.5),
               Z_OVER + 2)
        self.until(max(self.cue("size"), self.elapsed()))
        self.play(FadeIn(o), run_time=fr(4))
        self.until(max(self.cue("first", end=True) + 0.2, self.elapsed() + 0.8))
        self.play(FadeOut(o), run_time=fr(4))
