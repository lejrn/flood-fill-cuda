"""r05: all blocks. Every SM gets two blocks (48 x 256 threads) and the
64 Mpx square goes 18x the CPU. But on a thin snake 48 blocks lose to one:
every ring is one pixel, and after every ring the whole chip stops twice
at the gate (grid.sync), about two microseconds each.

At "every SM" the 22 dark SMs light in a ripple out from SM 0 and 1; at
"forty-eight" two cubes land on every SM while the square refills in one
quick pass. At "But" the snake comes in with its three times (48 / 2 / 1
blocks, one small panel by the floor). From "Why" the snake fills one
ring at a time and the whole machine goes amber twice per ring (the gate
rises); a "grid.sync" tag sits by the GPU board from "gate"; then the
rings race to the end with the gate up.

Starts on chain_state(7) (two SMs lit, the square painted), ends on
chain_state(8) (all SMs lit, two blocks each, the snake painted, the gate
at rest).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (DOWN, LEFT, RIGHT, FadeIn, FadeOut, Group, ImageMobject, LaggedStart, Rectangle,
                   RoundedRectangle, VGroup, VMobject)

from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import ALL_LIT, boundary, level_colors, toys
from scenes.style import BLUE, GREY, GRID, INK, INK_SOFT, label

SLOW_RING = 0.95        # at least this many seconds per slow ring: 2 stops per ring stay under 3 flashes/s
STOP_RAMP = 0.05        # seconds for one slow stop to ease in (and out)
RAMP = 0.15             # seconds for the gate to come up as the fast rings start, and to go down at the end


def sm_order(first=(0, 1)) -> list[int]:
    """The SMs by distance from the first ones (on the die): a ripple out from them."""
    c = [np.array([mc.sm_rect(i).x + mc.sm_rect(i).w / 2, mc.sm_rect(i).y + mc.sm_rect(i).d / 2]) for i in range(24)]
    o = np.mean([c[i] for i in first], axis=0)
    return sorted(range(24), key=lambda i: (np.linalg.norm(c[i] - o), i))


def set_alpha(g: Group, a: float) -> None:
    """Opacity of a machine part (box faces and images); a Group has no set_opacity."""
    for m in g.get_family():
        if isinstance(m, ImageMobject):
            m.set_opacity(a)
        elif isinstance(m, VMobject):
            m.set_fill(opacity=a, family=False)


def amber_overlay(spec: mc.Spec, pl) -> Group:
    """The SMs, blocks and gate of `spec` waiting (amber, the gate up), invisible.
    Built at the machine's own z and added later, so it draws over the parts."""
    w = mc.build(replace(mc.with_rooms(spec, "wait"), team_state="wait", gate="wait"), pl, z0=100.0)
    g = Group(*[v for k, v in w.parts.items() if (k.startswith("room") and k[4:].isdigit())
                or k in ("teams", "gate")])
    set_alpha(g, 0.0)
    return g


def small(text: str, size: int, color, mono: bool = False):
    """A small label built at 4x and scaled down: Pango spaces small text badly
    (at size 15 "kernel time" reads "kerneltime" and letters touch)."""
    return label(text, size * 4, color, mono=mono).scale(0.25)


def cubes_by_sm(teams: Group, pl) -> list[Group]:
    """The faces of a teams part, grouped by the SM they stand on (nearest SM centre)."""
    pts = np.array([mc.sm_point(pl, i)[:2] for i in range(24)])
    out = [Group() for _ in range(24)]
    for face in teams.submobjects:
        d = np.linalg.norm(pts - face.get_center()[:2], axis=1)
        out[int(np.argmin(d))].add(face)
    return out


def times_panel(G, rows: list) -> VGroup:
    """The snake's three kernel times, one compact table: a count, a bar as long
    as its time, the time. The winner (1 block) is the bright row."""
    longest = max(ms for _, ms in rows)
    bar_w = 0.6
    lines = VGroup()
    for n, ms in rows:
        win = ms == min(m for _, m in rows)
        col = INK if win else INK_SOFT
        bar = Rectangle(width=bar_w * ms / longest, height=0.11, stroke_width=0,
                        fill_color=BLUE if win else GREY, fill_opacity=0.95 if win else 0.7)
        lines.add(VGroup(small(str(n), 20, col, mono=True), bar, small(f"{ms:.0f} ms", 20, col, mono=True)))
    lines.arrange(DOWN, buff=0.07)
    # three columns: counts right-aligned, bars from one x, times right-aligned
    xc = max(l[0].width for l in lines)
    xt = max(l[2].width for l in lines)
    for l in lines:
        y = l[0].get_y()
        l[0].move_to([xc - l[0].width / 2, y, 0])
        l[1].move_to([xc + 0.14 + l[1].width / 2, y, 0])
        l[2].move_to([xc + 0.14 + bar_w + 0.16 + xt - l[2].width / 2, y, 0])
    head = VGroup(small("blocks", 15, INK_SOFT), small("kernel time", 15, INK_SOFT))
    head[0].next_to(lines, np.array([0, 1, 0]), buff=0.07).align_to(lines, LEFT)
    head[1].move_to(head[0]).align_to(lines, RIGHT)
    body = VGroup(head, lines)
    box = RoundedRectangle(corner_radius=0.1, width=body.width + 0.22, height=body.height + 0.14,
                           fill_color=ov.PANEL, fill_opacity=0.95, stroke_color=GRID, stroke_width=2)
    box.move_to(body)
    g = VGroup(box, body)
    # 9:16: in the corner right of the floor's front tip, above the caption;
    # 16:9: in the corner left of it, below the floor's left edge
    if G.vertical:
        (x0, y0), (x1, y1) = (1.72, -0.62), (3.8, 0.46)
    else:
        (x0, y0), (x1, y1) = (-6.9, -3.85), (-4.55, -2.2)
    g.scale(min(1.0, (x1 - x0) / g.width, (y1 - y0) / g.height))
    if G.vertical:
        g.move_to([x1 - g.width / 2, y0 + g.height / 2, 0])
    else:
        g.move_to([x0 + g.width / 2, (y0 + y1) / 2, 0])
    for m in g.get_family():
        m.set_z_index(ov.Z_OVER + 4)
    return g


def sync_tag(G, pl) -> VGroup:
    """A small "grid.sync" tag above the GPU board's back-right corner (clear of
    the raised gate), its right edge on the goal meter's: the gate's name."""
    t = small("grid.sync", 18, INK_SOFT, mono=True)
    box = RoundedRectangle(corner_radius=0.08, width=t.width + 0.26, height=t.height + 0.18, fill_color=ov.PANEL,
                           fill_opacity=0.95, stroke_color=GRID, stroke_width=1.5)
    g = VGroup(box, t)
    t.move_to(box)
    c = mc.gpu_corner(pl, "back-right")
    g.move_to([G.goal_c[0] + G.goal_w / 2 - g.width / 2, c[1] + (0.12 if G.vertical else 0.42), 0])
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
        pl = self.Lay.machine_pl
        n = self.num
        b = int(n["big_blocks"])
        order = sm_order()

        # every SM: the 22 dark SMs light, a ripple out from SM 0 and 1; the
        # "22 SMs idle" caption leaves with the old picture, the square goes red
        self.until(self.cue("every"))
        sq_red = self.toy(level_colors(sq, np.full(sq.shape, -1)))
        self.floor_blend(self.toy(level_colors(sq, l4)), sq_red, frames=4)
        lit = {id(a.mobject): a for a in self.machine_anims(replace(self.machine.spec, rooms=ALL_LIT), slide=0)}
        ripple = [lit[k] for k in (id(self.machine.parts[f"room{i}"]) for i in order) if k in lit]
        self.unsay(frames=9, extra=[LaggedStart(*ripple, lag_ratio=0.12)])

        # forty-eight blocks: two cubes land on every SM (the old single cubes on
        # SM 0 and 1 make way), and the square fills in one quick pass
        self.until(self.cue("forty-eight"))
        old = self.machine.parts["teams"]
        self.machine_anims(full)                 # only the teams change; they land below
        new = self.machine.parts["teams"]
        per_sm = cubes_by_sm(new, pl)
        r0, r1 = self.elapsed(), self.cue("eighteen")
        self.floor_play(lambda p: round(p * n4) - 1,
                        lambda i: self.toy(level_colors(sq, l4, upto=i, top=n4 - 1), ring=(l4 == i + 1)),
                        max(r1 - r0, 0.8))
        land = [FadeIn(per_sm[i], shift=DOWN * 0.22) for i in order if len(per_sm[i])]
        self.play(FadeOut(old), LaggedStart(*land, lag_ratio=0.1), run_time=fr(9))
        self.remove(*per_sm)
        self.add(new)
        self.say(f"{int(self.der['big_48_x'])}×", "kernel time", at=r1)

        # the thin snake, with its three times: 48 blocks lose to one (the panel
        # comes with the snake, the caption on "one", never within MIN_DWELL of it)
        t_snake = self.cue("But")
        self.until(t_snake)
        sq_done = self.toy(level_colors(sq, l4))
        sn_red = self.toy(level_colors(sn, np.full(sn.shape, -1)))
        self.floor_blend(sq_done, sn_red, frames=6)
        panel = times_panel(self.G, [(b, n["snake_48blocks_ms"]), (2, n["snake_2blocks_ms"]),
                                     (1, n["snake_1block_ms"])])
        self.unsay(frames=5, extra=[FadeIn(panel, shift=DOWN * 0.1)])
        self.extras.append(panel)
        self.say(f"{b} lose to 1", at=max(self.cue("one", after=self.cue("lose")), t_snake + MIN_DWELL))

        # why: one pixel per ring, and two stops at the gate per ring. The slow
        # rings run from "Why" to "gate" (2 stops per ring at under 3 flashes per
        # second, a photosensitivity limit; each stop eases in and out, capped
        # at 0.85), then the rest race past with the gate up
        amber = amber_overlay(full, pl)
        self.add(amber)
        t0, ts, t1 = self.cue("Why"), self.cue("gate", end=True), self.cue("each", end=True) - 0.2
        slow = max(1, int((ts - t0) / SLOW_RING))
        period = (ts - t0) / slow
        slow_t = slow * period
        fast_t = max(t1 - t0 - slow_t, 0.5)
        # a ring: work, then wait / gap / wait / gap (fractions of the ring); the
        # gap between the two waits is long enough to read as two stops
        cuts = (0.0, 0.22, 0.42, 0.6, 0.8, 1.0)
        names = ("work", "wait", "gap", "wait", "gap")

        def phase(p: float):
            """(ring, state): state is work | wait | gap in the slow rings, None when fast."""
            tau = p * (slow_t + fast_t)
            if tau < slow_t:
                x = tau / period + 1e-6
                i, f = int(x), x - int(x)
                j = max(k for k in range(5) if f >= cuts[k])
                return i, names[j]
            q = min((tau - slow_t) / fast_t, 1.0)
            return min(top, slow + int(q * (top + 1 - slow))), None

        def glow(p: float) -> float:
            """Amber opacity: a trapezoid on each wait in the slow rings; up (0.7)
            while the rest race past, eased in and out over RAMP."""
            tau = p * (slow_t + fast_t)
            if tau >= slow_t:
                return 0.7 * max(0.0, min(1.0, (tau - slow_t) / RAMP, (slow_t + fast_t - tau) / RAMP))
            f = (tau / period) % 1.0
            ramp = STOP_RAMP / period           # inside each wait, so the gap stays dark
            o = 0.0
            for a, z in ((cuts[1], cuts[2]), (cuts[3], cuts[4])):
                o = max(o, min(1.0, (f - a) / ramp, (z - f) / ramp))
            return 0.85 * o

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
            set_alpha(amber, 0.0 if p >= 1 else glow(p))

        self.until(t0)
        self.floor_play(phase, render, slow_t + fast_t)
        self.run(flash, slow_t + fast_t)

        # at "gate": the panel goes, the gate's name comes in by the GPU board;
        # the tag leaves with the old caption, so the last caption is one text
        # change (not a second one when the extras fade before the snap)
        tg = self.cue("gate")
        self.until(tg)
        tag = sync_tag(self.G, pl)
        self.extras.remove(panel)
        self.play(FadeOut(panel), FadeIn(tag), run_time=fr(5))
        self.until(max(self.cue("each"), tg + MIN_DWELL))
        self.unsay(frames=2, extra=[FadeOut(tag)])
        self.say_last()
        self.settle()
        self.remove(amber)
