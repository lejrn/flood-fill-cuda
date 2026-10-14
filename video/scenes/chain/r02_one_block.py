"""r02: one block on one SM. At "So" the floor splits into two half
floors, the same red drone on each, "CPU" and "GPU" under them with a
step counter each; at "onto" a block (the blue cube) drops onto SM 0,
which lights; at "SM" that SM grows out of the chip over the idle split
floor, drawn large like NVIDIA's SM figure (`sm_closeup`: 4 x 32 lanes,
shared memory), and flies back into SM 0 just before "paint". Text lands
at least MIN_DWELL apart. The race in steps starts at "paint", on one clock:
the CPU paints one pixel per step (left), the
block one whole 4-way ring per step (right). The right side finishes at
its ring count; the left side then speeds up and finishes too.

Then the race in time, short, on a 16 K px test blob: a small card
between the strip and the split floor, CPU and best GPU bars on one
scale growing at one speed, so the CPU stops first. "CPU wins" lands,
the card folds into the strip's first pill, the floor labels fade, and
at "size" an outline marks the next pill ("huge blob"), the next fix.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (DOWN, LEFT, FadeIn, FadeOut, GrowFromEdge, Group, Rectangle, RoundedRectangle,
                   SurroundingRectangle, VGroup)

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import Z_FLOOR, Z_OVER, boundary, floor_raster, fmt_ms, level_colors, toys
from scenes.style import BLUE, GREY, GRID, INK, INK_SOFT, label

# The SM close-up: (centre of the SM picture, its height), the title and
# sub to its left. In 9:16 it sits in the band between the caption and
# the GPU board's back edge, left of SM 0; in 16:9, in the empty band
# above the CPU board. Both keep the CPU board (it races too) in view.
CLOSEUP = {True: 3.7, False: 3.5}          # the close-up's height, 9:16 and 16:9


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


def sm_callout(L) -> Group:
    """mc.sm_closeup over the floor area (the split floors are still at step
    0): SM 0 drawn large, with its 4 x 32 lanes and shared memory labelled."""
    h = CLOSEUP[L.G.vertical]
    return mc.sm_closeup(L.machine_pl, L.G.floor_c, h, i=0, shelf=False)


def time_card(n: dict, halves: list, ins, strip):
    """The race in time on the 16 K px test blob: CPU and best GPU bars on one
    scale (the best GPU bar is full width). Placed between the inset and the
    floor's right edge, the split floor and the strip. Returns (card, bars)."""
    rows = [("CPU", n["small_cpu_ms"], GREY), ("best GPU", n["small_gpu_ms"], BLUE)]
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

        self.until(self.cue("So"))
        anims = [FadeOut(old), FadeIn(left), FadeIn(right), *[FadeIn(m) for m in notes]]
        if self.cap is not None:
            anims.append(FadeOut(self.cap))
            self.cap = None
        self.set_floors(halves)
        t_lab = self.elapsed()                          # text lands at least MIN_DWELL apart from here on
        self.play(*anims, run_time=fr(6))

        # ---- "onto": a block drops onto SM 0, which lights under it
        self.until(self.cue("onto"))
        drop = self.machine_anims(replace(b5.machine, rooms=b4.machine.rooms), slide=0.3)
        light = self.machine_anims(b5.machine, slide=0)
        self.play(*drop, *light, run_time=fr(6))

        # ---- "SM": that SM, drawn large, for about 2.5 s (its title waits out the floor labels)
        sm = sm_callout(self.Lay)
        home = mc.sm_point(self.Lay.machine_pl, 0)
        t_sm = max(self.cue("SM"), t_lab + MIN_DWELL, self.elapsed())
        self.until(t_sm)
        self.play(FadeIn(sm, target_position=home, scale=0.08), run_time=fr(8))
        self.say(f"{int(n['small_tpb'])} threads", at=max(self.cue("threads"), t_sm + MIN_DWELL))
        # back into SM 0 just before "paint", where the race starts
        self.until(max(self.cue("paint") - fr(8), t_sm + fr(8) + MIN_DWELL, self.elapsed()))
        self.play(FadeOut(sm, target_position=home, scale=0.08), run_time=fr(8))

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
