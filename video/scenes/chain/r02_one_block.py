"""r02: one block on one SM. The floor splits into two half floors, the
same red drone on each: the CPU paints one pixel per step (left), one
block of 256 threads paints one whole 4-way ring per step (right), on
one clock and counted in steps, not time. The right side finishes at
its ring count; the left side then speeds up and finishes too.

Then the race in time, on a 16 K px test blob (a small flat card): CPU
and best GPU bars on one scale grow at one speed, so the CPU stops first;
"1 block" is a dim text row, far off that scale. The carried caption
"CPU wins" lands, the floor labels fade, the scoreboard card opens (below
the inset) with row 1 filled, the strip becomes "CPU | ? | ?", and at
"size" an outline marks row 2, the next fix.
"""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, FadeIn, FadeOut, GrowFromEdge, Rectangle, RoundedRectangle,
                   SurroundingRectangle, VGroup, config)

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import Z_FLOOR, Z_OVER, boundary, floor_raster, fmt_ms, level_colors, score_rows, toys
from scenes.style import BLUE, GREY, INK, INK_SOFT, TEAL, label


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


def score_card(G, rows: tuple, highlight: int | None = None):
    """ov.score_card, with its empty header cell made a point-bearing
    placeholder (a `label("")` has no points, and `set_x` on it fails)."""
    lines = VGroup()
    blank = Rectangle(width=0.01, height=0.01, stroke_width=0, fill_opacity=0)
    lines.add(VGroup(blank, label("CPU", 16, INK_SOFT), label("GPU", 16, INK_SOFT), label("winner", 16, INK_SOFT)))
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
            cells = VGroup(name, label(fmt_ms(r.cpu_ms), 20, INK, mono=True),
                           label(fmt_ms(r.gpu_ms), 20, INK, mono=True),
                           label(f"{r.winner} {r.factor}", 22, col, mono=True))
        lines.add(cells)
    cols = [max(row[j].width for row in lines) for j in range(4)]
    xs = np.cumsum([0] + [c + 0.45 for c in cols[:-1]])
    lines.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    for row in lines:
        for j, m in enumerate(row):
            m.set_x(xs[j] + m.width / 2)
    body = VGroup(label("CPU vs GPU", 22, INK), lines).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    room = config.frame_width - 0.9
    if body.width > room:
        body.scale_to_fit_width(room)
    box = RoundedRectangle(corner_radius=0.16, width=body.width + 0.6, height=body.height + 0.5,
                           fill_color=ov.PANEL, fill_opacity=0.97, stroke_color=ov.GRID, stroke_width=2)
    box.move_to(G.card_c)
    body.move_to(box)
    return _z(VGroup(box, body), Z_OVER + 20)


class OneBlock(ChainScene):
    k = 4

    def beat_anim(self) -> None:
        t = toys()
        drone, fill = t["drone"], t["drone4"]
        order, level = fill.order, fill.level
        total, top = len(order), int(level.max())
        rings = top + 1
        b5 = boundary(self.k + 1)
        n, G = self.num, self.G
        zt = Z_OVER - 10

        # ---- split: the drone on two half floors, reset to red; room 0 lights
        old = self.floors[0]
        halves = [fl.Floor(self.Lay.split_pl[s], z0=Z_FLOOR) for s in ("left", "right")]
        red = level_colors(drone, np.full(drone.shape, -1))
        for f in halves:
            f.show_raster(fl.toy_raster(red, f.rpx))
        left, right = halves

        def under(f, dy):
            im = f.image
            return np.array([im.get_center()[0], im.get_bottom()[1] - dy, 0.0])

        names = VGroup(label("CPU: 1 pixel per step", 17, INK).move_to(under(left, 0.2)),
                       label("GPU, 1 block: 1 ring per step", 17, INK).move_to(under(right, 0.2)))
        for nm, f in zip(names, halves):
            if nm.width > f.image.width - 0.1:
                nm.scale_to_fit_width(f.image.width - 0.1)
        _z(names, zt)
        cl = Counter("step 0", 20, INK_SOFT, under(left, 0.52), zt)
        cr = Counter("step 0", 20, INK_SOFT, under(right, 0.52), zt)
        tag = _z(label("steps, not time", 15, INK_SOFT, mono=True), zt)
        tag.move_to((under(left, 0.84) + under(right, 0.84)) / 2)
        notes = [names, cl, cr, tag]

        self.until(self.cue("So"))
        anims = self.machine_anims(b5.machine)
        if self.cap is not None:
            anims.append(FadeOut(self.cap))
            self.cap = None
        self.set_floors(halves)
        self.play(*anims, FadeOut(old), FadeIn(left), FadeIn(right), *[FadeIn(m) for m in notes],
                  run_time=fr(9))

        # ---- the race in steps, on one clock
        t_r0 = max(self.cue("two"), self.elapsed())
        t_r1 = max(self.cue("once"), t_r0 + 1.2)        # the right side's last ring
        dt = (t_r1 - t_r0) / (rings - 1)
        t_l1 = min(t_r1 + 0.9, self.cue("But") - fr(2))  # the left side, sped up, done before the card
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
        self.say(f"{int(n['small_tpb'])} threads", "one block, one SM", at=self.cue("two") + 0.1)

        # ---- the race in time, on a 16 K px test blob: the CPU stops first.
        # The bars share one scale, CPU vs best GPU (the best GPU bar is full
        # width); "1 block" is far off that scale, so it is a dim text row.
        rows = [("CPU", n["small_cpu_ms"], GREY), ("best GPU", n["small_gpu_ms"], BLUE)]
        vmax = n["small_gpu_ms"]
        bw = 1.6
        lines, bars = VGroup(), []
        for name, v, col in rows:
            nm = label(name, 16, INK_SOFT)
            bar = Rectangle(width=max(bw * v / vmax, 0.04), height=0.16, fill_color=col, fill_opacity=1,
                            stroke_width=0)
            val = label(fmt_ms(v), 16, INK, mono=True)
            lines.add(VGroup(nm, bar, val))
            bars.append((bar, v))
        slow = VGroup(label("1 block", 16, ov.DIM), label(fmt_ms(n["small_ch01_ms"]), 16, ov.DIM, mono=True))
        lines.add(slow)
        wn = max(r[0].width for r in lines)
        for i, r in enumerate(lines):
            y = -i * 0.34
            r[0].move_to([wn / 2, y, 0])
            if len(r) == 3:
                r[1].move_to([wn + 0.2 + r[1].width / 2, y, 0])
                r[2].move_to([wn + 0.2 + bw + 0.15 + r[2].width / 2, y, 0])
            else:
                r[1].move_to([wn + 0.2 + r[1].width / 2, y, 0])
        head = VGroup(label("16 K px test blob", 18, INK),
                      label("one session (GPU: kernel only)", 14, INK_SOFT, mono=True)).arrange(DOWN, buff=0.05)
        body = VGroup(head, lines).arrange(DOWN, buff=0.2)
        box = RoundedRectangle(corner_radius=0.12, width=body.width + 0.4, height=body.height + 0.35,
                               fill_color=ov.PANEL, fill_opacity=0.97, stroke_color=ov.GRID, stroke_width=2)
        card = VGroup(box, body)
        body.move_to(box)
        mid = (left.image.get_center() + right.image.get_center()) / 2
        card.move_to(mid)
        ins = self.over.get("inset")
        if ins is not None and card.get_top()[1] > ins.get_bottom()[1]:   # clear of the inset's text
            card.shift(RIGHT * max(0.0, ins.get_right()[0] + 0.1 - card.get_left()[0]))
        _z(card, Z_OVER + 5)
        for bar, _ in bars:
            bar.set_opacity(0)

        self.until(self.cue("But"))
        self.play(FadeIn(card), run_time=fr(5))         # the bars stay hidden: they grow next
        for bar, _ in bars:
            bar.set_opacity(1)
        # one growth speed: the CPU bar stops at v / vmax of the run, the GPU bar at its end
        grow = [GrowFromEdge(bar, LEFT, rate_func=(lambda tt, f=v / vmax: min(1.0, tt / f)))
                for bar, v in bars]
        self.play(*grow, run_time=fr(14))

        # ---- the carried caption, then the scoreboard is born
        self.say_last(at=self.cue("CPU"))
        small = score_rows()[0]
        sc = score_card(G, (small, "?", "?"), highlight=0)
        if ins is not None:                             # below the "you are here" inset
            lim = ins.get_bottom()[1] - 0.1
            if sc.get_top()[1] > lim:
                sc.shift(DOWN * (sc.get_top()[1] - lim))
        self.until(self.cue("wins"))
        self.over_to(strip=self.strip(b5.rows),
                     extra=[FadeIn(sc), FadeOut(card), *[FadeOut(m) for m in notes]], frames=6)

        # ---- "So: size first": row 2 of the card is the next fix
        row2 = sc[1][1][2]
        o = _z(SurroundingRectangle(row2, color=INK, buff=0.06, corner_radius=0.06, stroke_width=2), Z_OVER + 21)
        self.until(max(self.cue("size"), self.elapsed()))
        self.play(FadeIn(o), run_time=fr(3))
        self.until(max(self.cue("first", end=True) + 0.3, self.elapsed() + 0.6))
        self.play(FadeOut(sc), FadeOut(o), run_time=fr(5))
