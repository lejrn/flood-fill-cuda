"""r10: the queue carries tiles, and one warp closes a tile in registers
(chain_v5_proposal.md, "r10_tiles"). It answers r09's last line, "every
red pixel still rides the queue".

- "So the queue carries tiles": the carried caption goes, the labelled
  frame turns back to red, and in one machine change the twelve red
  tokens become tile tokens (a 2 x 2 grid on each top) and the launch
  pill fades out.
- "not pixels: squares of thirty-two by thirty-two": the floor zooms into
  the frame's largest blob (one fixed point; the binary red frame sampled
  smoothly, so its 900 px pixels never show as blocks), its red easing
  from the picture floor's keyed red to the arrays' exact red. At the
  first "thirty-two" the close-up blob's 7 x 7 tile context lands on it:
  the real 224 px arrays, the 15 tiles of rounds 0-3 already blue, the
  centre tile outlined (`tile_ctx_open`). The blend brings in the detail
  the 900 px picture lacks (a tenth of it).
- "One warp, thirty-two threads that run in step, takes a tile": SM 0's
  close-up grows out of the chip as in r02; caption 1 lands with one
  partition's "32" lit; the warp card unfolds out of that partition with
  the tile's real mask (543 of 1,024 red) and its 17 entry bits on the
  left edge, where the round-3 tile beside it handed them over.
- "Each thread holds one row": the rows light top to bottom, then one
  row (the first with an entry bit) gets a bracket, kept through "the
  thread's own memory"; caption 2 at "pixel".
- "The row sits in registers": the card's frame and the lit partition's
  register-file strip pulse white together (the new-part convention: the
  two pictures of the same registers). At "shared memory" the close-up's
  shared-memory strip gets a soft outline for a moment: what the
  registers are closer than.
- "Fill along every row at once, step diagonally, fill along the
  columns": the recorded sweeps play, the first trip on its words with a
  two-headed glyph under the grid (no words): along the rows both ways,
  a cross of both diagonals, along the columns both ways, as ch07's
  kernels fill. Then each later trip starts on its own word under one
  steady loop arrow (an arrow switching five times a second would read
  as jitter): trip 2 on "and", trip 3 on "until" (it adds the last 2
  cells), trip 4 on "nothing" (it changes nothing). A tally mark beside
  the glyph lights as each trip starts. A sweep turns red cells blue at
  about the same brightness, so nothing flashes.
- "nothing changes": after trip 4 the 32 vote dots dim; then, during
  "changes", the card and the close-up fold back into SM 0, the 7
  round-4 tiles on the floor turn blue together, and each SM gets its
  third cube (72 blocks, 3 per SM): the two cubes slide apart during the
  flight, the third drops between them once the card is home. Caption 2
  goes as the fold starts; caption 3 comes in at once in 16:9 and once
  the flight has passed the caption band in 9:16 (where it crosses it).

Starts on chain_state(14), ends on chain_state(15).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (DOWN, PI, Arc, FadeIn, FadeOut, Group, Line, Rectangle, Transform, UpdateFromAlphaFunc, VGroup,
                   smooth, there_and_back)
from manim.mobject.geometry.tips import ArrowTip

from scenes.chain import die as dz
from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import state as st
from scenes.chain.base import ChainScene, caption_group, fr
from scenes.chain.r02_one_block import strip_rect
from scenes.style import INK, INK_SOFT, RED_PX

Q = 2                   # the partition the warp card unfolds from: right column, front row (nearest the card)
ZOOM_STEPS = 60         # distinct zoom frames (one per frame at 30 fps over ~2 s)
ZOOM_BLEND = 0.15       # the zoom's first share blends from the picture floor (resized by PIL) into its own sampling
BAND = 3.0              # "the rows light top to bottom": the lit band's half width, in rows
FAST_MIN = 0.13         # "and repeat": the closest two sweeps that change cells may be, s (2 frames at 15 fps)
TICK_IN = 2             # a trip's tally mark fades in over this many frames (15 fps)
VOTE_S = 0.35           # the vote after trip 4: the dots dim top to bottom over this long
FOLD = 14               # the fold's frames: the flight in 0-8, the two cubes slide apart in 0-5, the third drops in 8-13
WARP_N = mc.WARP_N


def _z(mob, z: float):
    for m in mob.get_family():
        m.set_z_index(z)
    return mob


def span(a: float, b: float):
    """A rate function that runs `smooth` over the fraction [a, b] of a play only."""
    return lambda t: smooth(min(1.0, max(0.0, (t - a) / max(b - a, 1e-6))))


def red_mask(rgb: np.ndarray) -> np.ndarray:
    """(H, W) float32: 1 on the red frame's red pixels (it is pure red on white)."""
    return ((rgb[..., 0] > 200) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)).astype(np.float32)


def soft_window(mask: np.ndarray, rpx: int, r0: float, c0: float, side: float) -> np.ndarray:
    """(rpx, rpx) float32 coverage of the window [r0, r0 + side) x [c0, c0 +
    side) of a 0/1 mask: sampled bilinearly, then cut at one half with an
    edge about one output pixel wide. So a zoom shows smooth blob outlines
    at any scale, never the picture's pixels as blocks; 0 off the picture."""
    H, W = mask.shape
    pos = (np.arange(rpx) + 0.5) / rpx * side - 0.5

    def taps(start: float, n: int):
        x = start + pos
        i = np.floor(x).astype(int)
        f = (x - i).astype(np.float32)
        ok0 = ((i >= 0) & (i < n)).astype(np.float32)
        ok1 = ((i + 1 >= 0) & (i + 1 < n)).astype(np.float32)
        return np.clip(i, 0, n - 1), np.clip(i + 1, 0, n - 1), f, ok0, ok1

    ia, ib, fy, ya, yb = taps(r0, H)
    rows = mask[ia] * (ya * (1 - fy))[:, None] + mask[ib] * (yb * fy)[:, None]      # (rpx, W)
    ja, jb, fx, xa, xb = taps(c0, W)
    cov = rows[:, ja] * (xa * (1 - fx))[None] + rows[:, jb] * (xb * fx)[None]       # (rpx, rpx)
    k = max(1.0, rpx / side)                                                        # output px per source px
    return np.clip((cov - 0.5) * k + 0.5, 0.0, 1.0)


def glyph_alpha(g, a: float) -> None:
    """Show a glyph at opacity `a`: its line by stroke only, its tip filled
    (set_opacity would fill the loop arrow's arc into a disc)."""
    for m in g.get_family():
        tip = isinstance(m, ArrowTip)
        m.set_stroke(opacity=a, family=False)
        m.set_fill(opacity=a if tip else 0.0, family=False)


def loop_glyph(card: mc.WarpCard, color: str = INK_SOFT):
    """"and repeat": a loop arrow in the card's glyph slot (no words), held
    while the later trips run, instead of a direction arrow that would
    switch five times a second."""
    r = card.glyph_h * 0.34
    arc = Arc(radius=r, start_angle=0.55 * PI, angle=-1.6 * PI, arc_center=card.glyph_c, color=color,
              stroke_width=4)
    arc.add_tip(tip_length=r * 0.75, tip_width=r * 0.75)
    return _z(arc, card.z + 1)


def tally(card: mc.WarpCard, n: int, color: str = INK_SOFT) -> VGroup:
    """n tally marks right of the glyph slot, one per trip (no words), all
    hidden: each trip lights its mark as it starts, so the trips can be
    counted before caption 3 says "4 trips"."""
    h = card.glyph_h
    x0, y = card.glyph_c[0] + 1.15 * h, card.glyph_c[1]       # clear of the widest glyph (the row's, 1.5 h)
    marks = VGroup(*[Line([x0 + i * 0.32 * h, y - 0.3 * h, 0], [x0 + i * 0.32 * h, y + 0.3 * h, 0], color=color,
                          stroke_width=3.5) for i in range(n)])
    marks.set_stroke(opacity=0)
    return _z(marks, card.z + 1)


def regs_rect(q: int) -> dz.R:
    """Partition q's register file in one SM (world units from its
    front-left corner): the strip under the partition's scheduler, by the
    arithmetic of die._sm, worked back from die.sm_parts().lanes[q]."""
    ln = dz.sm_parts().lanes[q]
    pd = (ln.d + 0.012) / (1 - 0.13 - 0.16 - 0.12)          # the partition's depth
    sd, rd, ld = pd * 0.13, pd * 0.16, pd * 0.12            # scheduler, registers, load/store
    top = ln.y - ld - 0.004 + pd
    return dz.R(ln.x, top - sd - rd - 0.004, ln.w / 0.6, rd)


def pulse_ring(target, color: str, width: float, pad: float, z: float) -> Rectangle:
    """An outline round `target`'s bounding box, hidden (stroke opacity 0)."""
    r = Rectangle(width=target.width + pad, height=target.height + pad, stroke_color=color,
                  stroke_width=width).move_to(target.get_center())
    r.set_stroke(opacity=0)
    return _z(r, z)


class Tiles(ChainScene):
    k = 14

    def pulse(self, *rings: Rectangle, seconds: float) -> None:
        """Fade `rings` in and out together over `seconds` (there and back), then remove them."""
        self.add(*rings)
        self.play(*[UpdateFromAlphaFunc(r, lambda m, a: m.set_stroke(opacity=there_and_back(a))) for r in rings],
                  run_time=seconds)
        self.remove(*rings)

    def third_cubes(self, spec: mc.Spec) -> tuple[list, FadeIn, callable]:
        """Animations from two cubes per SM to `spec`'s three: the two slide to
        the outer places (a morph of each face), the third drops in between.
        Returns (slides, drop, done); call done() after the play: it swaps in
        `spec`'s own cubes and makes `spec` the machine."""
        pl = self.Lay.machine_pl
        new = mc.build(spec, pl, z0=st.Z_MACHINE)
        old_t, new_t = self.machine.parts["teams"], new.parts["teams"]
        no, nn = old_t.submobjects, new_t.submobjects
        n_old, n_new = self.machine.spec.teams, spec.teams
        assert set(n_old) == {2} and set(n_new) == {3}, (n_old, n_new)
        nf = len(nn) // (24 * 3)
        assert nf * 24 * 3 == len(nn) and nf * 24 * 2 == len(no), (len(no), len(nn))
        anims, drop = [], []
        for i in range(24):
            for jo, jn in ((0, 0), (1, 2)):
                for f in range(nf):
                    anims.append(Transform(no[(i * 2 + jo) * nf + f], nn[(i * 3 + jn) * nf + f].copy()))
            drop += [m.copy() for m in nn[(i * 3 + 1) * nf:(i * 3 + 2) * nf]]
        landing = Group(*drop)

        def done() -> None:
            self.remove(old_t, landing)
            self.add(new_t)
            parts = dict(self.machine.parts)
            parts["teams"] = new_t
            assert set(new.parts) <= set(parts), (set(new.parts), set(parts))
            assert all(self.machine.keys[k] == new.keys[k] for k in new.parts if k != "teams")
            self.machine = mc.Machine(spec, {k: parts[k] for k in new.parts}, dict(new.keys))

        return anims, FadeIn(landing, shift=DOWN * 0.3), done

    def beat_anim(self) -> None:
        a = st.a3()
        n = self.num
        start, end = st.boundary(self.k), st.boundary(self.k + 1)
        pl, G = self.Lay.machine_pl, self.G
        rpx = self.floor.rpx
        sweeps = a["cu_sweeps"]
        trips = int(n["cu_trips"])
        kinds = ["entry"] + ["row", "diag", "col"] * trips
        assert len(sweeps) == len(kinds), (len(sweeps), kinds)
        tp = int(n["tile_px"])
        assert tp == WARP_N == st.TILE
        entry = mc.tile_bits(sweeps[0])
        row = int(np.flatnonzero(entry.any(axis=1))[0])      # the first thread with an entry bit (lane 13)

        # ---- "So the queue carries tiles": the caption goes, the labelled
        # frame turns back to red, the red tokens become tile tokens and the
        # launch pill fades out (one machine change)
        self.until(self.cue("So"))
        red_floor = st.floor_raster("frame_red", rpx)
        self.floor_blend(st.floor_raster("frame_labels", rpx), red_floor, frames=8)
        tile_spec = replace(start.machine, launches=None, token_kind="tile")
        self.unsay(frames=6, extra=self.machine_anims(tile_spec, slide=0))

        # ---- "not pixels: squares of thirty-two by thirty-two": a pure zoom
        # (one fixed point) into the largest blob, smooth outlines at every
        # scale; its red goes from the picture floor's keyed red to the
        # arrays' exact red on the way, so the tile context lands on the
        # same colours (only the detail changes)
        mask = red_mask(st.picture("red"))
        S = mask.shape[0]
        r_t, c_t, s_t = st.ctx_window()
        k_t = 1 - s_t / S
        fix = (r_t / k_t, c_t / k_t)
        red_f = red_floor.astype(np.float32)
        empty = fl.rgb(fl.EMPTY).astype(np.float32)
        red0 = fl.key_picture(fl.rgb(RED_PX)[None, None]).reshape(3).astype(np.float32)    # as picture_raster draws it
        red1 = fl.rgb(RED_PX).astype(np.float32)

        def zoom(i: int) -> np.ndarray:
            p = i / ZOOM_STEPS
            e = smooth(p)
            s = S * (s_t / S) ** e
            k = 1 - s / S
            cov = soft_window(mask, rpx, fix[0] * k, fix[1] * k, s)[..., None]
            red = red0 + (red1 - red0) * e
            rgb = empty + (red - empty) * cov
            if p < ZOOM_BLEND:          # from the picture floor (PIL's resize) into this sampling
                u = p / ZOOM_BLEND
                rgb = red_f[..., :3] * (1 - u) + rgb * u
            out = np.empty((rpx, rpx, 4), np.uint8)
            out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
            out[..., 3] = 255
            return out

        t_z0 = max(self.cue("tiles"), self.elapsed())
        t_32 = self.cue("thirty-two")
        self.until(t_z0)
        self.floor_play(lambda p: round(p * ZOOM_STEPS), zoom, t_32 - t_z0 - fr(1))
        self.until(t_32)
        opened = st.floor_raster("tile_ctx_open", rpx)
        self.floor_blend(zoom(ZOOM_STEPS), opened, frames=10)

        # ---- "One warp": SM 0's close-up grows out of the chip (as in r02)
        sm = mc.sm_closeup(pl, (*G.sm_side[0], 0.0), G.sm_side[1], i=0)
        home = mc.sm_point(pl, 0)
        t_one = max(self.cue("One"), self.elapsed())
        self.until(t_one)
        self.play(FadeIn(sm, target_position=home, scale=0.08), run_time=fr(8))

        # ---- "thirty-two threads": caption 1, with one partition's "32" lit
        light = mc.partition_light(sm, Q)
        self.say("1 tile, 1 warp", f"{tp} x {tp} pixels", at=self.cue("thirty-two", after=t_one), extra=[FadeIn(light)])

        # ---- "that run in step, takes a tile": the warp card unfolds out of
        # that partition, the tile's mask with its entry bits on the left edge
        card = mc.warp_closeup(G.warp[0], G.warp[1], a["cu_mask"], sweeps[0])
        self.until(max(self.cue("run"), self.elapsed()))
        self.play(FadeIn(card, target_position=mc.partition_point(sm, Q), scale=0.06), run_time=fr(10))

        # ---- "Each thread holds one row": the rows light top to bottom, then
        # one row gets its bracket
        t_each = self.cue("Each")
        t_row = self.cue("row", after=t_each)
        lanes = np.arange(WARP_N)

        def band(p: float) -> None:
            pos = -BAND + p * (WARP_N - 1 + 2 * BAND)
            card.show(sweeps[0], glow=np.clip(1 - np.abs(lanes - pos) / BAND, 0, 1))

        self.until(t_each)
        self.run(band, t_row - t_each)
        bracket = card.row_bracket(row)
        self.until(t_row)
        self.play(FadeIn(bracket), run_time=fr(4))

        # ---- "one bit per pixel": caption 2
        self.say("1 bit per pixel", "a row per thread", at=self.cue("pixel"))

        # ---- "The row sits in registers": the card's frame and the lit
        # partition's register file pulse together (a new part): the two
        # pictures of the same registers. The bracket stays on its row through
        # "the thread's own memory" and goes at "closer"
        t_reg = self.cue("registers")
        rc, rw, rh = mc.closeup_rect(sm, regs_rect(Q))
        regs = _z(Rectangle(width=rw + 0.04, height=rh + 0.02, stroke_color=INK, stroke_width=3.0).move_to(rc),
                  sm[1].z_index + 2)
        regs.set_stroke(opacity=0)
        self.until(t_reg)
        self.pulse(pulse_ring(card.frame, INK, 3.5, 0.1, card.z + 2), regs,
                   seconds=min(max(self.cue("registers", end=True) - t_reg, fr(10)), fr(14)))
        self.until(max(self.cue("closer", after=t_reg), self.elapsed()))
        self.play(FadeOut(bracket), run_time=fr(5))

        # ---- "closer than shared memory": a soft outline on the close-up's
        # shared-memory strip, for a moment (not a new part: no white)
        c, w, h, _ = strip_rect(sm)
        strip = _z(Rectangle(width=w + 0.06, height=h + 0.06, stroke_color=INK_SOFT, stroke_width=3.0).move_to(c),
                   sm[1].z_index + 2)
        strip.set_stroke(opacity=0)
        t_sh = self.cue("shared", after=t_reg)
        self.until(max(t_sh, self.elapsed()))
        self.pulse(strip, seconds=fr(16))

        # ---- "Fill along every row at once, step diagonally, fill along the
        # columns, and repeat until nothing changes": the recorded sweeps,
        # one trip (row, diagonal, column) per word. Trip 1 on its words,
        # each sweep with its two-way glyph; then each later trip starts on
        # its own word under one loop arrow: trip 2 on "and", trip 3 on
        # "until" (its row sweep adds the last 2 cells), trip 4 on "nothing"
        # (it changes nothing). A tally mark lights as each trip starts.
        # trip 3's row sweep (sweep 7) adds the last cells; nothing after it changes
        assert trips == 4, trips
        same = [np.array_equal(np.asarray(sweeps[i]), np.asarray(sweeps[7])) for i in range(8, len(sweeps))]
        assert all(same) and not np.array_equal(np.asarray(sweeps[7]), np.asarray(sweeps[6])), same
        t1 = self.cue("along", after=t_sh)
        t2 = self.cue("step", after=t1)
        t3 = self.cue("fill", after=t2)
        t_and = self.cue("and", after=t3)
        t_until = self.cue("until", after=t_and)
        t_none = self.cue("nothing", after=t_until)
        t_ch = self.cue("changes", after=t_none)
        starts = [t1, t_and, t_until, t_none]                # where each trip starts (its tally mark lights)
        d2 = (t_until - t_and) / 3                           # trip 2: three sweeps that all add cells
        d34 = (t_none - t_until) / 3                         # trips 3 and 4: only trip 3's first sweep adds any
        assert d2 >= FAST_MIN, f"{self.beat}: trip 2's sweeps {d2:.3f} s apart"
        times = [t1, t2, t3] + [t_and + j * d2 for j in range(3)] + [t + j * d34 for t in (t_until, t_none)
                                                                      for j in range(3)]
        assert len(times) == len(kinds) - 1 and all(np.diff(times) > 0), times
        glyphs = {kind: card.glyph(kind) for kind in ("row", "diag", "col")}
        glyphs["loop"] = loop_glyph(card)
        for g in glyphs.values():
            glyph_alpha(g, 0.0)
        marks = tally(card, trips)
        self.add(*glyphs.values(), marks)
        shown = {"k": 0}
        t_swept = times[-1] + fr(1)

        def sweep(p: float) -> None:
            t = t1 + p * (t_swept - t1)
            for j, m in enumerate(marks):
                m.set_stroke(opacity=float(np.clip((t - starts[j]) / fr(TICK_IN), 0.0, 1.0)))
            k = sum(1 for x in times if x <= t + 1e-6)
            if k == shown["k"]:
                return
            shown["k"] = k
            card.show(sweeps[k])
            on = kinds[k] if k <= 3 else "loop"
            for kind, g in glyphs.items():
                glyph_alpha(g, 1.0 if kind == on else 0.0)

        self.until(t1)
        self.run(sweep, t_swept - t1)

        # ---- "nothing changes": at the end of trip 4 the 32 threads vote
        # that nothing changed (the dots dim, top to bottom), seen whole
        # before the fold
        def vote(p: float) -> None:
            for i, d in enumerate(card.votes):
                q = np.clip((p - 0.5 * i / WARP_N) / 0.5, 0, 1)
                d.set_fill(opacity=1.0 - 0.75 * q)

        t_vote = times[-1] + d34
        self.run(vote, VOTE_S, start=t_vote)

        # ---- "changes": the card and the close-up fold back into SM 0, the 7
        # round-4 tiles turn blue together, and each SM gets its third cube:
        # the two cubes slide apart during the flight, the third drops in
        # once the card is home (so the chip is never busy under the card).
        # One play: caption 2 goes as the flight starts and caption 3
        # (carried over the cut) comes in; in 9:16 the flight crosses the
        # caption band, so there it comes in once the flight has passed (as
        # r02's does), in 16:9 at once
        self.until(max(self.elapsed(), t_ch, t_vote + VOTE_S + fr(1)))
        self.floor_blend(opened, st.floor_raster("tile_ctx", rpx), frames=10)
        slides, drop, done = self.third_cubes(end.machine)
        for anim in slides:
            anim.set_rate_func(span(0.0, 5 / FOLD))
        drop.set_rate_func(span(8 / FOLD, 13 / FOLD))
        cap = caption_group(G, *st.carry_caption(self.k + 1))
        cap_in = span(9 / FOLD, 1.0) if G.vertical else span(2 / FOLD, 7 / FOLD)
        self.play(FadeOut(Group(card, *glyphs.values(), marks, light, sm), target_position=home, scale=0.08,
                          rate_func=span(0.0, 8 / FOLD)),
                  FadeOut(self.cap, rate_func=span(0.0, 2 / FOLD)), *slides, drop,
                  FadeIn(cap, scale=1.06, rate_func=cap_in), run_time=fr(FOLD))
        self.cap, self._cap_t = cap, self.elapsed()
        done()
        assert self.machine.spec == end.machine, (self.machine.spec, end.machine)
