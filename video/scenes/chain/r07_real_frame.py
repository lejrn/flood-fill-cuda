"""r07: the twist. Size is won, now count. The floor pulls back from the
64 Mpx test square to the 81 Mpx test frame, and the CPU launches one
kernel per blob: each launch crosses the PCIe link to the GPU, its blob
fills, and the SMs wait amber for the next one. The launch counter climbs
to 2,522 and the needle swings past the CPU tick: we built for size.

The act changes here (D4): in act 1 a hue was a block, in act 2 a hue is
a blob. So on "count", while the floor pulls back, the 48 cubes drop
their block hues and turn plain blue.

v5: the frame is transposed at the source (one orientation for every
floor), and the estimate is re-made in the one session at 8 neighbours:
~1,462 ms, against the labels-only CPU's 202 ms (its label map reset
before the clock, as act 2's loops reset theirs): the CPU wins by 7.2x.
The meter's CPU tick is the same CPU with its reset on the clock (225
ms); the needle swings far past either.

Starts on chain_state(11) (the 8-way owner mosaic, cubes in block hues,
"11×"), ends on chain_state(12) (the filled frame, plain blue cubes,
2,522 launches, "~1,462 ms").
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from manim import DL, UR, Circle, Dot, FadeIn, FadeOut, Group, ImageMobject, VMobject

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import LEVEL_HI, Z_OVER, boundary, filled_picture, floor_raster, picture, picture_blobs
from scenes.style import INK, INK_SOFT, label

Z_LABEL = 195.0         # the machine's silkscreen and launch pill (z0 + 95)
Z_DOT = 190.0           # launches: above the boards, under the pill


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


def set_alpha(g, a: float) -> None:
    """Opacity of a group of images and faces (Group has no set_opacity)."""
    for m in g.get_family():
        if isinstance(m, ImageMobject):
            m.set_opacity(a)
        elif isinstance(m, VMobject):
            m.set_fill(opacity=a)


def amber_overlay(spec: mc.Spec, pl) -> Group:
    """The SMs and blocks of `spec` in their waiting colours, invisible: laid
    over the lit ones, its opacity says how long they wait for a launch."""
    w = mc.build(replace(mc.with_rooms(spec, "wait"), team_state="wait"), pl, z0=100.0)
    g = Group(*[v for k, v in w.parts.items() if (k.startswith("room") and k[4:].isdigit()) or k == "teams"])
    set_alpha(g, 0.0)
    return g


def along(pts, s: float) -> np.ndarray:
    """The point a fraction s of the way along the polyline pts (by length)."""
    p = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = float(np.clip(s, 0.0, 1.0)) * cum[-1]
    i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
    return p[i] + (p[i + 1] - p[i]) * (d - cum[i]) / max(seg[i], 1e-9)


class LaunchPill:
    """The machine's launch pill with a live count. The number sits where the
    final count sits (right-aligned, mono), so at the final count the pill
    is the machine's own pill, pixel for pixel."""

    def __init__(self, final: str, pl) -> None:
        self.g = mc.launch_chip(final, pl)              # VGroup(box, VGroup(number, "launches"))
        self.row = self.g[1]
        self.num, self.word = self.row[0], self.row[1]
        self.anchor = self.num.get_corner(UR)
        self.word_at = self.word.get_corner(DL)
        self.text, self.word_text = final, "launches"
        for m in self.g.get_family():
            m.set_z_index(Z_LABEL)

    def _swap(self, old, new):
        for m in new.get_family():
            m.set_z_index(Z_LABEL)
        self.row.remove(old)
        self.row.add(new)
        # Cairo snapshots the family when a play starts: blank the old glyphs
        # (the Stopwatch trick) instead of leaving them drawn
        for m in old.get_family():
            m.points[:] = 0
        return new

    def set(self, text: str) -> None:
        if text != self.text:
            new = label(text, 20, INK, mono=True).move_to(self.anchor, aligned_edge=UR)
            self.num, self.text = self._swap(self.num, new), text
        word = "launch" if text == "1" else "launches"
        if word != self.word_text:
            # built as launch_chip builds it, so "launches" ends as the machine's own glyphs
            new = label(word, 60, INK_SOFT).scale(0.25).move_to(self.word_at, aligned_edge=DL)
            self.word, self.word_text = self._swap(self.word, new), word


class RealFrame(ChainScene):
    k = 11

    def beat_anim(self) -> None:
        n = self.num
        start, end = boundary(self.k), boundary(self.k + 1)
        pl = self.Lay.machine_pl
        red = picture("red")
        r, c, side = zoom_target()
        H = red.shape[0]
        square = floor_raster(start.floors[0], self.floor.rpx)     # the 8-way owner mosaic on screen
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

        # "Now count": the old caption, tag and inset go, the COUNT tag and the
        # one-line inset come in (on "Now", so COUNT is up as "count" is said),
        # and the floor pulls back to the real frame while "twenty-five hundred
        # blobs in the big frame" is said
        t_now, t_l = self.cue("Now"), self.cue("launch")
        self.until(t_now)
        t_frame = min(self.cue("frame", end=True), t_l - fr(2))
        self.floor_play(lambda p: round(p * steps), zoom, t_frame - t_now)

        # "count": act 2 is one hue per blob, so the cubes drop their block
        # hues. Plain blue cubes (the end state's) fade in over the hued ones
        # as the word is said: a hook, so the pull back and the tag go on
        plain = replace(start.machine, team_hues=(), team_busy=())
        blue = mc.build(plain, pl, z0=100.0, only=("teams",))
        cubes = blue.parts["teams"]
        set_alpha(cubes, 0.0)
        self.add(cubes)
        t_count = self.cue("count")
        self.run(lambda p: set_alpha(cubes, p), fr(9), start=t_count)
        new_tag, new_inset = ov.act_tag(self.G, *end.tag), ov.inset(self.G, end.inset)
        self.unsay(frames=3, extra=[FadeOut(self.over["tag"]), FadeOut(self.over["inset"])])
        self.play(FadeIn(new_tag), FadeIn(new_inset), run_time=fr(3))
        self.over["tag"], self.over["inset"] = new_tag, new_inset
        t_text = self.elapsed()

        # the blue cubes are fully in: they become the machine's cubes and the
        # hued ones go at once (left under them, their soft edges tint the
        # blue ones). A frame after the fade's last step: the renderer drops
        # a removed part only when the next wait starts
        self.until(t_count + fr(9) + fr(1))
        self.remove(self.machine.parts["teams"])
        self.machine = mc.Machine(plain, {**self.machine.parts, "teams": cubes},
                                  {**self.machine.keys, "teams": blue.keys["teams"]})
        N = int(n["est_ch03_calls"])
        # every text change waits MIN_DWELL after the last one is fully in, as say() does
        self.say(f"{int(n['n_blobs']):,} blobs", at=max(self.cue("blobs"), t_text + MIN_DWELL))
        t_cap = self.elapsed()

        # "one launch each": P-core 0 lights (it runs the launch loop), and the
        # pill comes in with the first launch's arrival. The first launches go
        # one at a time: a dot over PCIe, its blob fills, the SMs drain amber
        # until the next. The rest stream past, and the needle rides the summed
        # launch time past the CPU tick as "CPU" is said
        # one at a time: count, period, travel. The cubes leave amber once per
        # launch, so P >= 0.5: at most twice a second (the cut's rule, v5)
        S, P, T = 3, 0.5, 0.3
        TS, DE = 0.5, 0.16                          # the stream: travel, gap between launches
        # the first launch leaves once the core is lit, and lands (the pill
        # comes in) no sooner than MIN_DWELL after the caption
        t_go = max(t_l + 0.2, t_cap + MIN_DWELL - T)
        t_b0 = t_go + S * P                         # the stream's first launch
        t_b1 = max(self.cue("CPU", end=True), t_b0 + TS + 0.8)   # its last arrival: the needle lands
        emits = [t_go + j * P for j in range(S)]
        emits += list(np.arange(t_b0, t_b1 - TS + 1e-6, DE))

        def count(tt: float) -> int:
            """Launches that have reached the GPU by scene time tt."""
            if tt < t_go + T:
                return 0
            if tt < t_b0 + TS:
                return min(S, 1 + int((tt - t_go - T) / P))
            q = min(1.0, (tt - t_b0 - TS) / (t_b1 - t_b0 - TS))
            x = 1 - (1 - q) ** 2                     # on the meter's log scale, slowing into the value
            return min(N, max(S, int(round(S * (N / S) ** x))))

        # the blobs fill in scan order, but the ringed first three are the first
        # big enough to see, apart, and clear of the inset and the tag
        lab, sizes = picture_blobs()
        nb_ = len(sizes)
        Hh, Ww = lab.shape
        hi = fl.rgb(LEVEL_HI)
        done = filled_picture()
        keep_out = [self.over["inset"], self.over["tag"]]

        def blob_point(b: int) -> np.ndarray:
            rs, cs = np.nonzero(lab == b)
            x = -fl.HALF + (cs.mean() + 0.5) / Ww * 2 * fl.HALF
            y = fl.HALF - (rs.mean() + 0.5) / Hh * 2 * fl.HALF
            return self.floor.pl.to_frame([[x, y, 0.0]])[0]

        def clear(b: int, pad: float = 0.45) -> bool:
            """Away from the floor's side corners, the inset and the tag."""
            cs = np.nonzero(lab == b)[1]
            pt = blob_point(b)
            return 0.15 * Ww < cs.mean() < 0.85 * Ww and all(
                not (m.get_left()[0] - pad < pt[0] < m.get_right()[0] + pad
                     and m.get_bottom()[1] - pad < pt[1] < m.get_top()[1] + pad) for m in keep_out)

        first = []
        for b in range(nb_):
            if sizes[b] >= 60 and clear(b) and all(
                    np.linalg.norm(blob_point(b) - blob_point(a)) > 0.6 for a in first):
                first.append(b)
                if len(first) == S:
                    break
        order = first + [b for b in range(nb_) if b not in first]
        rank = np.empty(nb_, np.int32)
        rank[order] = np.arange(nb_)
        rmap = np.where(lab >= 0, rank[np.maximum(lab, 0)], nb_)

        def blobs_done(k: int) -> int:
            """Picture blobs filled after k launches (the downscale has fewer blobs)."""
            return k if k <= S else S + round((nb_ - S) * (k - S) / (N - S))

        def render(i: int):
            if i >= nb_:
                return self.pic(done)
            img = red.copy()
            img[rmap < i] = hi
            return self.pic(img)

        rings = []
        for b in first:
            ring = Circle(radius=0.09, stroke_color=INK, stroke_width=3).move_to(blob_point(b))
            ring.set_z_index(Z_OVER - 20)
            ring.set_stroke(opacity=0)
            rings.append(ring)

        path = mc.link_path(pl)
        dots = [Dot(radius=0.05, color=INK) for _ in range(6)]
        for d in dots:
            d.set_z_index(Z_DOT)
            d.set_opacity(0)
        amber = amber_overlay(start.machine, pl)
        pill = LaunchPill(end.machine.launches, pl)
        pill.set("1")                               # it comes in with the first arrival

        # the needle wakes at the first arrival and rides the summed launch time
        old_goal = self.over["goal"]
        goal = self.goal(end.goal)
        tri, val = goal[-2], goal[-1]
        goal.remove(tri, val)
        track = goal[2]
        x0, x1 = track.get_start()[0], track.get_end()[0]
        est = n["est_ch03_ms"]
        tri.set_fill(opacity=0)

        def needle_x(k: int) -> float:
            return ov._track_x(self.G, max(est * k / N, ov.LO_MS), x0, x1)

        tri.shift((needle_x(1) - tri.get_center()[0]) * np.array([1.0, 0, 0]))

        A_WAIT, A_STREAM, RISE = 0.75, 0.5, 0.3

        def amber_at(tt: float) -> float:
            if tt >= t_b1:                           # the loop is done: back to lit
                return A_STREAM * max(0.0, 1 - (tt - t_b1) / 0.4)
            if tt >= t_b0 + TS:
                return A_STREAM
            arrived = [e + T for e in emits[:S] if e + T <= tt]
            if not arrived:                          # waiting for the first launch
                return A_WAIT * min(1.0, (tt - t_l) / RISE)
            return A_WAIT * min(1.0, (tt - arrived[-1]) / RISE)   # a launch lands: work, then wait

        def ring_op(tt: float, b: int) -> float:
            if tt < emits[b] + T:
                return 0.0
            return float(np.clip(1.0 - (tt - (t_b0 + TS)) / 0.4, 0.0, 1.0))

        span = t_b1 + 0.45 - t_l

        def trip(e: float) -> float:
            return T if e < t_b0 - 1e-6 else TS

        def tick(p: float) -> None:
            tt = t_l + p * span
            k = count(tt)
            set_alpha(amber, amber_at(tt))
            for b, ring in enumerate(rings):
                ring.set_stroke(opacity=ring_op(tt, b))
            for d in dots:
                d.set_opacity(0)
            for j, e in enumerate(emits):
                if e <= tt < e + trip(e):
                    dots[j % len(dots)].move_to(along(path, (tt - e) / trip(e))).set_opacity(1)
            if k:
                pill.set(f"{k:,}")
                tri.set_fill(opacity=min(1.0, (tt - t_go - T) / 0.15))
                tri.shift((needle_x(k) - tri.get_center()[0]) * np.array([1.0, 0, 0]))

        self.until(t_l)
        self.remove(old_goal)
        self.add(goal, tri, amber, *rings, *dots)
        self.over["goal"] = goal
        self.floor_play(lambda p: blobs_done(count(t_l + p * (t_b1 - t_l))), render, t_b1 - t_l)
        self.run(tick, span)
        self.play(*self.machine_anims(replace(plain, cpu="lit"), slide=0), run_time=fr(3))
        self.until(t_go + T)
        self.play(FadeIn(pill.g), run_time=fr(3))

        # the last launch lands: the CPU core goes dark, then the needle's value
        # comes in on the same frames as the carried caption (one text change)
        self.until(t_b1)
        self.play(*self.machine_anims(replace(end.machine, launches=None), slide=0), run_time=fr(4))
        self.machine = mc.Machine(end.machine, {**self.machine.parts, "launches": pill.g},
                                  {**self.machine.keys, "launches": ((end.machine.launches,),)})
        set_alpha(val, 0.0)
        self.add(val)
        self.run(lambda p: set_alpha(val, p), fr(5), start=self.elapsed() + fr(2))   # after the old caption's fade
        self.say_last()
        self.settle()
        self.remove(tri, val, amber, *rings, *dots)
        goal.add(tri, val)
