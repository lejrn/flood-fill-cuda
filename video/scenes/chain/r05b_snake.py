"""r05b: the snake (v3 r05's tail, its own beat in v4). On a thin snake 48
blocks lose to one: every ring is one pixel, and after every ring the
whole chip stops twice at the gate (grid.sync), about two microseconds
each.

At "But" the square gives way to the red snake and its three kernel times
come in (48 / 2 / 1 blocks, one small panel by the floor; the one-block
row is ch01's shared-ring kernel, so it reads "1, shared queue"). From
"Why" the snake fills one ring at a time. A ring is one pixel, so only
block 0 has work (D6: ceil(1 / 256) = 1 busy block): its cube on SM 0
stays bright and the 47 others go dim. Twice per ring the whole machine
goes amber and the gate rises. At "gate" the 47 idle cubes turn amber and
stay amber (they were waiting at the gate all along); block 0's turns
amber only at each flash. The panel gives way to a "grid.sync" tag; one
more slow ring shows the new colours, then the rest race to the end with
the gate up. The last caption lands on "each", under the line it
captions; as the race ends, just after, the gate comes down with its tag
and the cubes go back to their hues.

Starts on chain_state(9) (all SMs lit, 48 cubes in their hues, the
48-block owner mosaic), ends on chain_state(10) (the same machine, the
snake painted, the gate at rest).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (DOWN, RIGHT, FadeIn, FadeOut, Group, ImageMobject, Rectangle, RoundedRectangle, VGroup,
                   VMobject, config)

from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import boundary, floor_raster, level_colors, toys
from scenes.style import BLUE, GREY, GRID, INK, INK_SOFT

SLOW_RING = 0.95        # at least this many seconds per slow ring: 2 stops per ring stay under 3 flashes/s
STOP_RAMP = 0.05        # seconds for one slow stop to ease in (and out)
RAMP = 0.15             # seconds for the gate to come up as the fast rings start, and to go down at the end
TURN = 5                # frames for the cubes to change state (dim, then amber)
# MIN_DWELL in whole frames (1.8 s): until() rounds 1.75 s down a frame. A
# plain float: the ring hooks' length comes from it, and a numpy float makes
# a hook return np.False_, which base's `is not False` keeps forever
DWELL = float(np.ceil(MIN_DWELL * 15) / 15)


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


def times_panel(G, rows: list) -> VGroup:
    """The snake's three kernel times, one compact table: a count, a bar as long
    as its time, the time. The winner (1 block) is the bright row."""
    longest = max(ms for _, ms in rows)
    bar_w = 0.6
    lines = VGroup()
    for name, ms in rows:
        win = ms == min(m for _, m in rows)
        col = INK if win else INK_SOFT
        bar = Rectangle(width=bar_w * ms / longest, height=0.11, stroke_width=0,
                        fill_color=BLUE if win else GREY, fill_opacity=0.95 if win else 0.7)
        lines.add(VGroup(ov.lbl(name, 20, col, mono=True), bar, ov.lbl(f"{ms:.0f} ms", 20, col, mono=True)))
    lines.arrange(DOWN, buff=0.07)
    # three columns: counts right-aligned, bars from one x, times right-aligned
    xc = max(l[0].width for l in lines)
    xt = max(l[2].width for l in lines)
    for l in lines:
        y = l[0].get_y()
        l[0].move_to([xc - l[0].width / 2, y, 0])
        l[1].move_to([xc + 0.14 + l[1].width / 2, y, 0])
        l[2].move_to([xc + 0.14 + bar_w + 0.16 + xt - l[2].width / 2, y, 0])
    # the heads: "blocks" over the counts (the one-block row's note runs left of it)
    head = VGroup(ov.lbl("blocks", 15, INK_SOFT), ov.lbl("kernel time", 15, INK_SOFT))
    head[0].next_to(lines, np.array([0, 1, 0]), buff=0.07)
    head[0].move_to([xc - head[0].width / 2, head[0].get_y(), 0])
    head[1].move_to(head[0]).align_to(lines, RIGHT)
    body = VGroup(head, lines)
    box = RoundedRectangle(corner_radius=0.1, width=body.width + 0.22, height=body.height + 0.14,
                           fill_color=ov.PANEL, fill_opacity=0.95, stroke_color=GRID, stroke_width=2)
    box.move_to(body)
    g = VGroup(box, body)
    # 9:16: in the corner under the floor's right-front edge, above the caption;
    # 16:9: under the floor's front tip, left of the caption. Either corner
    # takes the panel near full size (v3's narrower corners shrank the
    # one-block row's note past reading); the old caption leaves first
    if G.vertical:
        (x0, y0), (x1, y1) = (0.9, -0.74), (3.88, 0.3)
    else:
        (x0, y0), (x1, y1) = (-6.95, -3.85), (-2.4, -2.45)
    g.scale(min(1.0, (x1 - x0) / g.width, (y1 - y0) / g.height))
    if G.vertical:
        g.move_to([x1 - g.width / 2, y0 + g.height / 2, 0])
    else:
        g.move_to([x0 + g.width / 2, y0 + g.height / 2, 0])
    for m in g.get_family():
        m.set_z_index(ov.Z_OVER + 4)
    return g


def sync_tag(G, pl) -> VGroup:
    """A small "grid.sync" tag above the GPU board's back-right corner (clear of
    the raised gate), its right edge on the goal meter's: the gate's name."""
    t = ov.lbl("grid.sync", 18, INK_SOFT, mono=True)
    box = RoundedRectangle(corner_radius=0.08, width=t.width + 0.26, height=t.height + 0.18, fill_color=ov.PANEL,
                           fill_opacity=0.95, stroke_color=GRID, stroke_width=1.5)
    g = VGroup(box, t)
    t.move_to(box)
    c = mc.gpu_corner(pl, "back-right")
    g.move_to([G.goal_c[0] + G.goal_w / 2 - g.width / 2, c[1] + (0.12 if G.vertical else 0.42), 0])
    for m in g.get_family():
        m.set_z_index(ov.Z_OVER + 4)
    return g


class Snake(ChainScene):
    k = 9

    def cube_anims(self, spec: mc.Spec, amber: Group) -> list:
        """The cross-fade of the cubes to `spec`'s states, the amber overlay kept
        on top: the new cubes go in under it before the play (a FadeIn adds its
        mobject last, and the last added draws on top at equal z)."""
        anims = self.machine_anims(spec, slide=0)
        new = self.machine.parts["teams"]
        self.remove(amber)
        self.add(new, amber)
        return anims

    def beat_anim(self) -> None:
        t = toys()
        sn, lv = t["snake"], t["snake4"].level
        top = int(lv.max())
        pl = self.Lay.machine_pl
        n = self.num
        b = int(n["big_blocks"])
        full = boundary(self.k + 1).machine

        # per block (D6): on the snake every ring is one pixel, so only block 0
        # has work; the other 47 are dim, then (from "gate") amber
        ids = [int(i) for i in n["big_sm_ids"]]
        hues = mc.block_hues(b)
        dim = mc.place_blocks(full, ids, hues=hues, busy=mc.block_states(1, b, "dim"))
        wait = mc.place_blocks(full, ids, hues=hues, busy=mc.block_states(1, b, "wait"))

        # the thin snake, with its three times: 48 blocks lose to one (the panel
        # comes with the snake, the caption on "one", DWELL after the panel is
        # fully in, as say() counts it)
        t_snake = self.cue("But")
        self.until(t_snake)
        sn_red = self.toy(level_colors(sn, np.full(sn.shape, -1)))
        self.floor_blend(floor_raster(boundary(self.k).floors[0], self.floor.rpx), sn_red, frames=6)
        panel = times_panel(self.G, [(str(b), n["snake_48blocks_ms"]), ("2", n["snake_2blocks_ms"]),
                                     ("1, shared queue", n["snake_1block_ms"])])
        self.unsay(frames=3)                    # out first: the 9:16 panel sits just over the caption
        self.play(FadeIn(panel, shift=DOWN * 0.1), run_time=fr(4))
        self.extras.append(panel)
        self.say(f"{b} lose to 1", at=max(self.cue("one", after=self.cue("lose")), self.elapsed() + DWELL))

        # why: one pixel per ring, and two stops at the gate per ring. The slow
        # rings run from "Why" (2 stops per ring at under 3 flashes per second,
        # a photosensitivity limit; each stop eases in and out, capped at
        # 0.85). A ring starts just after the cubes turn amber at "gate", so
        # one whole slow ring shows them; then the rest race past with the
        # gate up. The last caption lands on "each" (it captions that line);
        # the gate comes down when the race ends, its tag with it
        amber = amber_overlay(full, pl)
        self.add(amber)
        t0, tg = self.cue("Why"), self.cue("gate")
        t_amber = tg + fr(TURN)
        pre = max(1, int((t_amber - t0) / SLOW_RING))
        period = (t_amber - t0) / pre
        slow = pre + 1
        slow_t = slow * period

        def on_frame(t: float) -> float:
            """The frame until(t) lands on."""
            return round(t * config.frame_rate) / config.frame_rate

        # the old caption out (2 frames) just before "each", the last one in (5)
        # from it; the race runs at least 0.6 s after its slow rings, and no
        # sooner than the in-fade's last update, counted from the hooks' own
        # start frame, so settle() needs no extra live frame
        t_cap = self.cue("each") - fr(2)
        t1 = on_frame(t_cap) + fr(2) + fr(4)
        fast_t = max(t1 - on_frame(t0) - slow_t, 0.6)
        t_race = on_frame(t0) + slow_t + fast_t
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

        # the first ring is one pixel: block 0 works, the 47 others go dim
        self.until(t0)
        self.floor_play(phase, render, slow_t + fast_t)
        self.run(flash, slow_t + fast_t)
        self.play(*self.cube_anims(dim, amber), run_time=fr(TURN))

        # at "gate": the 47 idle cubes turn amber and stay amber; the panel goes,
        # the gate's name comes in by the GPU board
        self.until(tg)
        tag = sync_tag(self.G, pl)
        self.extras.remove(panel)
        self.play(*self.cube_anims(wait, amber), FadeOut(panel), FadeIn(tag), run_time=fr(TURN))

        # the last caption, on "each"; the tag stays while the gate is up
        self.until(t_cap)
        self.unsay(frames=2)
        self.say_last()
        # the race ends: the snake is done, the gate comes down with its name,
        # and every cube goes back to its hue (all work, chain_state(10))
        self.until(t_race - fr(TURN))
        self.play(*self.cube_anims(full, amber), FadeOut(tag), run_time=fr(TURN))
        self.settle()
        self.remove(amber)
