"""r05: all blocks. Every SM gets two blocks (48 x 256 threads) and the
64 Mpx square goes 18x the CPU. But on a thin snake 48 blocks lose to one:
every ring is one pixel, and after every ring the whole chip stops twice
at the gate (grid.sync), about two microseconds each.

Starts on chain_state(7) (two rooms lit, the square painted), ends on
chain_state(8) (all rooms lit, the snake painted, the gate at rest).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, LEFT, RIGHT, FadeIn, FadeOut, Rectangle, RoundedRectangle, VGroup

from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
from scenes.chain.state import boundary, level_colors, toys
from scenes.style import BLUE, GRID, INK, INK_SOFT, label


def amber_overlay(spec: mc.Spec, pl) -> VGroup:
    """The rooms, teams and gate of `spec` in their waiting colours, invisible.
    Built at the machine's own z, added later, so it draws over the parts."""
    w = mc.build(replace(mc.with_rooms(spec, "wait"), team_state="wait",
                         gate="wait" if spec.gate else None), pl, z0=100.0)
    g = VGroup(*[v for k, v in w.parts.items() if (k.startswith("room") and k[4:].isdigit())
                 or k in ("teams", "gate")])
    g.set_opacity(0)
    return g


def timing_chips(G, rows: list, tag: str) -> VGroup:
    """Three small flat chips, one per block count: a bar as long as its time.
    They sit in the empty corner beside the floor's bottom tip."""
    longest = max(ms for _, ms in rows)
    bar_w = 1.25
    lines = VGroup()
    for i, (name, ms) in enumerate(rows):
        col = INK if i == 0 else INK_SOFT
        txt = VGroup(label(name, 16, col), label(f"{ms:.0f} ms", 16, col, mono=True))
        txt[1].next_to(txt[0], RIGHT, buff=0.18)
        bar = Rectangle(width=bar_w * ms / longest, height=0.07, stroke_width=0,
                        fill_color=BLUE if i == 0 else INK_SOFT, fill_opacity=0.9)
        bar.next_to(txt, DOWN, buff=0.06, aligned_edge=LEFT)
        lines.add(VGroup(txt, bar))
    lines.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    # the times line up in one column
    xr = max(l[0][0].get_right()[0] for l in lines) + 0.18
    for l in lines:
        l[0][1].move_to([xr + l[0][1].width / 2, l[0][1].get_y(), 0])
    foot = label(tag, 13, INK_SOFT)
    body = VGroup(lines, foot).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    box = RoundedRectangle(corner_radius=0.1, width=body.width + 0.3, height=body.height + 0.24,
                           fill_color=ov.PANEL, fill_opacity=0.95, stroke_color=GRID, stroke_width=2)
    box.move_to(body)
    g = VGroup(box, body)
    # fit a box in the empty corner beside the floor's bottom tip, clear of the caption
    if G.vertical:
        (x0, y0), (x1, y1) = (1.6, -0.4), (3.9, 1.0)
    else:
        (x0, y0), (x1, y1) = (-7.0, -3.05), (-4.5, -1.45)
    g.scale(min(1.0, (x1 - x0) / g.width, (y1 - y0) / g.height))
    if G.vertical:
        g.move_to([x1 - g.width / 2, y0 + g.height / 2, 0])
    else:
        g.move_to([x0 + g.width / 2, y0 + g.height / 2, 0])
    for m in g.get_family():
        m.set_z_index(ov.Z_OVER + 4)
    return g


class AllBlocks(ChainScene):
    k = 7

    def beat_anim(self) -> None:
        t = toys()
        sq, l4 = t["square"], t["square4"].level
        sn, lv = t["snake"], t["snake4"].level
        n4 = int(l4.max()) + 1
        top = int(lv.max())
        full = boundary(8).machine

        n = self.num
        b = int(n["big_blocks"])

        # every SM: all 24 rooms light up with two teams each, the gauge fills;
        # the old "2 of 24" caption leaves with the old picture
        self.until(self.cue("every"))
        out = [FadeOut(self.cap)] if self.cap is not None else []
        self.cap = None
        self.over_to(frames=7, gauge=self.gauge(int(n["sm"]), "1/3 of each SM: registers"),
                     extra=self.machine_anims(full, slide=0) + out)

        # the square again: back to red, then every ring in one pass of the
        # whole chip, quick
        sq_red = self.toy(level_colors(sq, np.full(sq.shape, -1)))
        self.floor_blend(self.toy(level_colors(sq, l4)), sq_red, frames=3)
        self.until(self.elapsed() + fr(3) + 1e-3)
        r0, r1 = self.elapsed(), self.cue("eighteen")
        self.floor_play(lambda p: round(p * n4) - 1,
                        lambda i: self.toy(level_colors(sq, l4, upto=i, top=n4 - 1), ring=(l4 == i + 1)),
                        max(r1 - r0, 0.8))
        self.say(f"{int(self.der['big_48_x'])}×", "the CPU, one 64 Mpx blob, kernel time", at=self.cue("eighteen"))

        # the thin snake: 48 blocks lose to one
        self.until(self.cue("snake"))
        sq_done = self.toy(level_colors(sq, l4))
        sn_red = self.toy(level_colors(sn, np.full(sn.shape, -1)))
        self.floor_blend(sq_done, sn_red, frames=6)
        chips = timing_chips(self.G, [("1 block", n["snake_1block_ms"]), ("2 blocks", n["snake_2blocks_ms"]),
                                      (f"{b} blocks", n["snake_48blocks_ms"])],
                             "the 256 × 256 snake, GPU kernels")
        self.until(self.cue("lose"))
        self.play(FadeIn(chips, shift=DOWN * 0.15), run_time=fr(5))
        self.extras.append(chips)
        self.say(f"{b} lose to 1", "one thin snake, one pixel wide", at=self.cue("one", after=self.cue("lose")))

        # why: one pixel per ring, and two stops at the gate per ring
        amber = amber_overlay(full, self.Lay.machine_pl)
        self.add(amber)
        t0, t1 = self.cue("Why"), self.cue("each", end=True) - 0.2
        # slow rings: 2 gate stops per ring at < 3 flashes per second (a
        # photosensitivity limit); each stop eases in and out, capped at 0.85
        slow, period = 4, 0.72
        fast_t = max(t1 - t0 - slow * period, 0.5)
        # a ring: work, then wait / gap / wait / gap (fractions of the ring)
        cuts = (0.0, 0.3, 0.5, 0.6, 0.8, 1.0)
        names = ("work", "wait", "gap", "wait", "gap")

        def phase(p: float):
            """(ring, state): state is work | wait | gap in the slow rings, None when fast."""
            tau = p * (t1 - t0)
            if tau < slow * period:
                x = tau / period + 1e-6
                i, f = int(x), x - int(x)
                j = max(k for k in range(5) if f >= cuts[k])
                return i, names[j]
            q = min((tau - slow * period) / fast_t, 1.0)
            return min(top, slow + int(q * (top + 1 - slow))), None

        def glow(p: float) -> float:
            """Amber opacity in the slow rings: a trapezoid on each wait."""
            tau = p * (t1 - t0)
            f = (tau / period) % 1.0
            ramp = 0.06
            o = 0.0
            for a, z in ((cuts[1], cuts[2]), (cuts[3], cuts[4])):
                o = max(o, min(1.0, (f - a + ramp) / ramp, (z + ramp - f) / ramp))
            return 0.85 * max(o, 0.0)

        def render(key):
            i, st = key
            if i >= top:
                return self.toy(level_colors(sn, lv))
            if st == "work":
                dots = (lv == i).astype(int)
                return self.toy(level_colors(sn, lv, upto=i - 1, top=top), dots=dots)
            dots = (lv == i).astype(int) * (2 if st == "wait" else 1)
            return self.toy(level_colors(sn, lv, upto=i, top=top), dots=dots)

        def flash(p: float) -> None:
            i, st = phase(p)
            if p >= 1 or i >= top:
                amber.set_opacity(0)
            elif st is None:
                amber.set_opacity(0.7)
            else:
                amber.set_opacity(glow(p))

        self.until(t0)
        self.floor_play(phase, render, t1 - t0)
        self.run(flash, t1 - t0)
        # landscape: the inset's label sits on the snake's first cells, where
        # the slow fill creeps; it steps aside and comes back with the last caption
        hide = not self.G.vertical
        if hide:
            self.over_to(frames=4, inset=None)
        g = self.cue("gate")
        self.until(g)
        self.extras.remove(chips)
        self.say("the gate", "grid.sync, 2 per ring", extra=[FadeOut(chips)])
        if hide:        # the slow rings are over: the inset comes back
            self.until(t0 + slow * period)
            self.over_to(frames=4, inset=ov.inset(self.G, *boundary(8).inset))
        self.say_last(at=self.cue("each"))
        self.settle()
        self.remove(amber)
