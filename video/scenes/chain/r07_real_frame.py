"""r07: the twist. Size is won, now count. The floor pulls back from the
64 Mpx test square to the 81 Mpx test frame, and the machine labels its
blobs one launch per blob, in scan order. The rooms drain amber between
launches, and the needle swings past the CPU tick: we built for size.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from manim import Circle, FadeIn, FadeOut, VGroup

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
from scenes.style import INK
from scenes.chain.state import LEVEL_HI, Z_OVER, boundary, filled_picture, level_colors, picture, picture_blobs, toys


@lru_cache(maxsize=None)
def zoom_target() -> tuple[float, float, float]:
    """(row, col, side) of a window around a mid-sized blob near the picture's
    centre: where the pull back starts."""
    lab, sizes = picture_blobs()
    H, W = lab.shape
    best = None
    for k, n in enumerate(sizes):
        if not 150 <= n <= 600:
            continue
        rs, cs = np.nonzero(lab == k)
        r, c = rs.mean(), cs.mean()
        d = (r - H / 2) ** 2 + (c - W / 2) ** 2
        side = max(rs.max() - rs.min(), cs.max() - cs.min()) * 1.5 + 6
        if best is None or d < best[0]:
            best = (d, r, c, side)
        if d < (H / 8) ** 2:
            break
    _, r, c, side = best
    return float(r), float(c), float(side)


def zoom_crop(img: np.ndarray, r: float, c: float, side: float) -> np.ndarray:
    """A square window of `img` around (r, c), clamped to the picture."""
    H, W = img.shape[:2]
    side = int(max(4, min(side, H)))
    r0 = int(min(max(r - side / 2, 0), H - side))
    c0 = int(min(max(c - side / 2, 0), W - side))
    return img[r0:r0 + side, c0:c0 + side]


def amber_overlay(spec: mc.Spec, pl) -> VGroup:
    """The rooms, teams and gate of `spec` in their waiting colours, invisible."""
    w = mc.build(replace(mc.with_rooms(spec, "wait"), team_state="wait",
                         gate="wait" if spec.gate else None), pl, z0=100.0)
    g = VGroup(*[v for k, v in w.parts.items() if k.startswith("room") or k in ("teams", "gate")])
    g.set_opacity(0)
    return g


def raise_strip(scene) -> None:
    """The strip's header sits on the meter's edge. The state draws it over
    the meter (same z, added later); a raised z keeps it there while the
    meter changes, and draws the same pixels."""
    s = scene.over.get("strip")
    if s is not None:
        for m in s.get_family():
            m.set_z_index(Z_OVER + 1)


def needle_in(scene, spec: ov.GoalSpec, from_ms: float, frames: int = 8) -> None:
    """Wake the parked needle: it appears at `from_ms`, swings to its value,
    then the value label fades in. Only the needle changes; the meter's box
    and labels are the same pixels before and after."""
    raise_strip(scene)
    old = scene.over["goal"]
    new = scene.goal(spec)
    tri, val = new[-2], new[-1]
    new.remove(tri, val)
    track = new[2]
    x0, x1 = track.get_start()[0], track.get_end()[0]
    dx = tri.get_center()[0] - ov._track_x(scene.G, from_ms, x0, x1)
    tri.shift(-dx * np.array([1.0, 0, 0]))
    scene.remove(old)
    scene.add(new)
    scene.play(FadeIn(tri), run_time=fr(2))
    scene.play(tri.animate.shift(dx * np.array([1.0, 0, 0])), run_time=fr(frames))
    scene.play(FadeIn(val), run_time=fr(3))
    scene.remove(tri, val)
    new.add(tri, val)
    scene.over["goal"] = new


class RealFrame(ChainScene):
    k = 9

    def beat_anim(self) -> None:
        t = toys()
        n = self.num
        end = boundary(self.k + 1)
        red = picture("red")
        r, c, side = zoom_target()
        H = red.shape[0]
        square = self.toy(level_colors(t["square"], t["square8"].level))
        crop = self.pic(zoom_crop(red, r, c, side), nearest=True)
        steps = 24

        def zoom(i: int):
            p = i / steps
            if p <= 0.25:                                    # the square dissolves into one blob up close
                a = p / 0.25
                return (square.astype(np.float32) * (1 - a) + crop.astype(np.float32) * a).astype(np.uint8)
            q = ((p - 0.25) / 0.75) ** 0.6                   # then the window widens to the whole frame
            if q >= 1:
                return self.pic(red)
            s = side * (H / side) ** q
            return self.pic(zoom_crop(red, r + (H / 2 - r) * q, c + (H / 2 - c) * q, s), nearest=q < 0.85)

        # "Now count": the act tag flips, the inset names a drone frame, the floor pulls back
        t_count, t_launch = self.cue("count"), self.cue("launch")
        self.until(t_count)
        self.floor_play(lambda p: round(p * steps), zoom, t_launch - t_count - fr(2))
        # out first, then in: the old tag and inset never sit under the new ones
        new_tag, new_inset = ov.act_tag(self.G, *end.tag), ov.inset(self.G, *end.inset)
        self.unsay(frames=3, extra=[FadeOut(self.over["tag"]), FadeOut(self.over["inset"])])
        self.play(FadeIn(new_tag), FadeIn(new_inset), run_time=fr(3))
        self.over["tag"], self.over["inset"] = new_tag, new_inset
        self.say(f"{int(n['n_blobs']):,} blobs", "81 Mpx test frame: size and count together",
                 at=self.cue("blobs"))

        # one launch per blob: the blobs fill in scan order, the rooms drain amber between launches
        lab, sizes = picture_blobs()
        nb_ = len(sizes)
        hi = fl.rgb(LEVEL_HI)
        done = filled_picture()
        t_fill_end = self.cue("not")

        # the first launches one at a time, one amber pulse each (rooms drain
        # between launches), each blob ringed; then the rest blur past
        dur = t_fill_end - t_launch
        slow, period = 3, 0.4
        t_slow = slow * period

        def key(p: float) -> int:
            tt = p * dur
            if tt < t_slow:
                return 1 + int(tt / period)
            q = min(1.0, (tt - t_slow) / max(dur - t_slow, 1e-6))
            return min(nb_, slow + int(np.ceil((nb_ - slow) * q ** 1.8)))

        def render(i: int):
            if i >= nb_:
                return self.pic(done)
            img = red.copy()
            img[(lab >= 0) & (lab < i)] = hi
            return self.pic(img)

        Hh, Ww = lab.shape
        rings = []
        for b in range(slow):
            rs, cs = np.nonzero(lab == b)
            x = -fl.HALF + (cs.mean() + 0.5) / Ww * 2 * fl.HALF
            y = fl.HALF - (rs.mean() + 0.5) / Hh * 2 * fl.HALF
            ring = Circle(radius=0.09, stroke_color=INK, stroke_width=3).move_to(self.floor.pl.to_frame([[x, y, 0.0]])[0])
            ring.set_z_index(Z_OVER - 20)
            ring.set_stroke(opacity=0)
            rings.append(ring)

        def ring_op(tt: float, b: int) -> float:
            t_on = b * period
            if tt < t_on:
                return 0.0
            return float(np.clip(1.0 - (tt - (t_slow + 0.3)) / 0.4, 0.0, 1.0))

        amber = amber_overlay(replace(end.machine, launches=None, stencil=0), self.Lay.machine_pl)
        self.add(amber, *rings)
        self.extras += [amber, *rings]
        self.until(t_launch)
        self.floor_play(key, render, dur)

        def pulse(p: float) -> None:
            tt = p * dur
            amber.set_opacity(0.0 if p >= 1 else 0.45 - 0.35 * np.cos(2 * np.pi * tt / period))
            for b, ring in enumerate(rings):
                ring.set_stroke(opacity=ring_op(tt, b))

        self.run(pulse, dur)
        self.machine_to(end.machine, frames=6)
        self.say(f"{int(n['est_ch03_calls']):,} launches", "one per blob, 81 Mpx test frame")
        # "Slower than the CPU": the needle wakes and swings past the CPU tick
        self.until(self.cue("Slower"))
        needle_in(self, end.goal, from_ms=1000 / 30)
        self.say_last(at=self.cue("built"))
