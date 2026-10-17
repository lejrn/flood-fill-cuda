"""r11 (v5, act 3): one gate per round. A closed tile hands on only its
border bits, into the tiles they really open; those join the next round's
queue, and the whole chip waits at the gate once per round. On the giant
square that turns 4,001 rings into 127 rounds (101 ms into 3.12 ms). But
that is one blob, from one seed.

On the close-up blob's 7 x 7 tile context (k = 15, rounds 0-4 blue):
- "border bits": the closed centre tile's reached border cells light
  (31 top, 26 left, 18 right, 2 bottom: state.border_bits_mask), and r10's
  carried "4 trips" goes, so the trips and the rounds never share the screen;
- "into the tiles around it": an arrow into each of the 4 tiles they
  really open (state.ctx_targets: above 32 px, right 18, above-left 1,
  above-right 1), each target outlined; no arrow where the tiles are open
  already or no reached pixel touches. The round's 12 tile tokens are
  consumed with the arrows;
- "those join the next round's queue": 4 tokens drop in, one per target;
- "One gate per round": the 4 tiles turn blue (round 5, exactly the 4
  targets), their 4 tokens go with them, and the gate flashes once; then
  the context holds still through "not two per ring" (no replay of later
  rounds).

"On the giant square": the inset's line reads "stress test" and the
context gives way to the 64 Mpx square at act 1's footprint, swelling out
of the middle, all red. Its rounds are the real ones (TL sq_first_open,
250 x 250 tiles, shaded light to dark by the round a tile first opened).
A lens (r05's, at r05's spot) grows out of the seed tile: 11 x 11 tiles
round it, each a sharp square in a grid, since a round's ring of tiles
is a few output pixels on the floor. Round 0 (the seed tile) and round 1
(the 8 round it) fill in it one at a time; round 0's gate flashes once,
round 1's gate rises into a steady amber hold while the other rounds
race (rounds 2-5 fill the lens), a small counter beside the floor runs
to 127 (the last round opens nothing: 126 rounds of tiles, then one that
changes nothing), easing in and out, and lands with caption 2, the gate
lowering as it lands. The queue refills to 12 tile tokens as the race
starts; the lens folds back into the seed tile as caption 2 comes in.
"from a hundred milliseconds": row 2's pill turns to GPU 340×, the
beat's one pill change, with no caption.

"But that's one blob": the caption and the counter go, the inset's line
returns to "drone frame", and the square shrinks onto the 81 Mpx frame's
one blob (state.one_blob_picture: the largest blob blue, the rest red),
a zoom about the one point that stays put, landing as a white ring round
the blob (state.one_blob_ring). Under "one seed" its seed (TL
closeup.seed) comes in as a white dot and pulses twice (1.25 a second, a
small element: the cut's photosensitivity rule). Ring and dot stay: k =
16 carries them (state.one_blob_raster), so "1 blob" / "from 1 seed"
always has its blob and its seed on screen.

Every text change keeps MIN_DWELL from the last one fully in (captions,
the inset's line and the pill, as r03, r07 and r09 count them).

Starts on chain_state(15) (tile_ctx, "4 trips"), ends on chain_state(16)
(frame_one_blob, row 2 GPU 340×, "1 blob" / "from 1 seed").
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import Arrow, Circle, FadeIn, FadeOut, GrowArrow, Group, ImageMobject, Line, VGroup, config, smooth
from manim.constants import RESAMPLING_ALGORITHMS

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain import state as st
from scenes.chain.base import EPS, MIN_DWELL, ChainScene, fr
from scenes.style import INK, INK_SOFT, RED_PX

# MIN_DWELL in whole frames (1.8 s): until() rounds 1.75 s down a frame. A
# plain float, so the hooks' times stay plain floats
DWELL = float(np.ceil(MIN_DWELL * 15) / 15)
HALF_FRAME = 0.5 / 15           # a time computed ahead may land this far off the frame grid
Z_ARROW = st.Z_FLOOR + 5        # over the floor, under the machine and every overlay
ARROW_SPAN = (0.32, 0.9)        # an arrow from this far along the centre-to-target line to this far
BITS_IN = 6                     # frames: the border bits light
TARGETS_IN = 6                  # frames: the arrows grow, the targets' outlines come in
BLUE_IN = 5                     # frames: the 4 targets turn blue (round 5)
SWELL = 9                       # frames: the context gives way to the square
INSET = 5                       # frames: the inset's line changes (2 out, 3 in)
LENS = {True: ((2.62, 3.15), 0.8), False: ((-1.0, 0.9), 0.72)}   # r05's lens spot (centre, radius): 9:16, 16:9
LENS_TILES = 11                 # the lens shows 11 x 11 tiles round the seed tile: rounds 0-5
LENS_IN = 8                     # frames: the lens grows out of the seed tile (r05's)
SLOW = 0.6                      # seconds per slow round on the square (rounds 0 and 1, then the race)
FLASH_AFTER = 0.15              # a round's gate this long after its tiles open: work, then the gate
CAP_IN = 7                      # frames: a caption replacing one (2 out, 5 in)
PULL = 14                       # frames: the square shrinks onto the frame's one blob
PULSE = 0.8                     # seconds per pulse of the blob's seed dot (1.25 a second)
PULSES = 2
DOT_IN = 0.2                    # seconds: the seed dot comes in as its first pulse starts


def boxed(box: tuple, s: float) -> tuple:
    """`box` (top, left, bottom, right) scaled by s about its centre."""
    t, l, b, r = box
    cy, cx, hy, hx = (t + b) / 2, (l + r) / 2, (b - t) / 2 * s, (r - l) / 2 * s
    return (cy - hy, cx - hx, cy + hy, cx + hx)


def onto_blob():
    """The square's box (floor fractions) as it shrinks onto the frame's one
    blob, e in 0..1: a zoom about the one point that stays put, from act 1's
    footprint to the box round the blob's ring (state.one_blob_ring)."""
    t, l, b, r = st.SQUARE_BOX
    c0, h0 = np.array([(t + b) / 2, (l + r) / 2]), (b - t) / 2
    side = st.one_blob_picture().shape[0]
    (cy, cx), rad = st.one_blob_ring()
    c1, h1 = np.array([cy, cx]) / side, rad / side
    k = h1 / h0
    fix = (c1 - k * c0) / (1 - k)

    def box(e: float) -> tuple:
        s = k ** e
        c, h = fix + (c0 - fix) * s, h0 * s
        return (c[0] - h, c[1] - h, c[0] + h, c[1] + h)

    return box


def over(out: np.ndarray, colors: np.ndarray, cells: fl.Cells, alpha: float) -> np.ndarray:
    """`colors` ((rows, cols, 3)) drawn nearest into cells.box over the floor
    raster `out`, by `alpha` (in place)."""
    if alpha <= 0:
        return out
    ri, ci = cells.maps()
    rs, cs = np.flatnonzero(ri >= 0), np.flatnonzero(ci >= 0)
    if not len(rs) or not len(cs):
        return out
    src = colors[ri[rs]][:, ci[cs]].astype(np.float32)
    dst = out[rs[:, None], cs, :3].astype(np.float32)
    out[rs[:, None], cs, :3] = np.clip(dst + (src - dst) * alpha, 0, 255).astype(np.uint8)
    return out


def plane_point(frac: tuple) -> np.ndarray:
    """A point given as fractions of the floor's side (Cells.frac) on the
    floor's own plane, in its world units (Floor.frac_point before to_frame)."""
    fr_row, fr_col = (float(v) for v in frac)
    return np.array([-fl.HALF + fr_col * 2 * fl.HALF, fl.HALF - fr_row * 2 * fl.HALF, 0.0])


def tile_arrows(pl, rpx: int) -> VGroup:
    """One arrow from the closed centre tile into each tile its border bits
    really open (state.ctx_targets), laid on the floor's plane so its tip
    foreshortens with the floor; white with a dark edge, over red and blue."""
    cells = st.ctx_tile_cells(rpx)
    ci, cj = st.ctx_centre()
    a = plane_point(cells.frac(ci + 0.5, cj + 0.5))
    out = VGroup()
    for (i, j), _ in st.ctx_targets():
        b = plane_point(cells.frac(i + 0.5, j + 0.5))
        s0, s1 = ARROW_SPAN
        arrow = Arrow(a + s0 * (b - a), a + s1 * (b - a), buff=0, color=INK, stroke_width=5, tip_length=0.17,
                      max_tip_length_to_length_ratio=0.4, max_stroke_width_to_length_ratio=40)
        arrow.set_stroke(fl.EMPTY, width=10, background=True)
        arrow.apply_function(lambda p: pl.to_frame([[p[0], p[1], 0.0]])[0])
        out.add(arrow)
    for m in out.get_family():
        m.set_z_index(Z_ARROW)
    return out


class RoundCounter(VGroup):
    """A small mono line ("round 12") whose number changes from a floor hook,
    its left edge fixed. The Cairo renderer snapshots the family when a play
    starts, so a replaced number is blanked, not removed (r02's Counter)."""

    SIZE = 18

    def __init__(self, text: str, anchor: np.ndarray, size: int = SIZE, z: float = ov.Z_OVER - 4):
        super().__init__()
        self.size, self.anchor, self.z = size, anchor, z
        self.text = text
        self.value = self._make(text)
        self.add(self.value)
        self.add_updater(lambda m, dt: None)        # drawn every frame, never cached

    def _make(self, text: str):
        m = ov.lbl(text, self.size, INK_SOFT, mono=True)
        m.move_to(self.anchor, aligned_edge=np.array([-1.0, 0.0, 0.0]))
        for x in m.get_family():
            x.set_z_index(self.z)
        return m

    def set(self, text: str) -> None:
        if text == self.text:
            return
        old, new = self.value, self._make(text)
        self.remove(old)
        self.value, self.text = new, text
        self.add(new)
        for m in old.get_family():
            m.points[:] = 0


class TileLens(Group):
    """r05's lens (ov.owner_lens) over the square's seed tile: LENS_TILES x
    LENS_TILES tiles at the output's own pixels, each a sharp square in a
    dark grid, a white ring, and a thin leader to a ring on the floor round
    the lens's window. show() swaps its pixels (the same size; never while
    it fades), so a round fills one ring of tiles in it. Drawn every frame
    (a no-op updater), as the counter."""

    def __init__(self, centre, radius: float, target: np.ndarray, target_r: float, z: float = ov.Z_OVER - 4):
        super().__init__()
        px = int(round(2 * radius * config.pixel_width / config.frame_width))
        pos = (np.arange(px) + 0.5) / px * LENS_TILES
        self.idx = np.minimum(np.floor(pos).astype(int), LENS_TILES - 1)
        gw = max(1.0, px / 150) * LENS_TILES / px / 2                 # half a grid line, in tiles
        f = pos - np.floor(pos)
        line = (f < gw) | (f > 1 - gw)
        self.grid = line[:, None] | line[None, :]
        yy, xx = np.mgrid[0:px, 0:px]
        self.alpha = np.where((xx + 0.5 - px / 2) ** 2 + (yy + 0.5 - px / 2) ** 2 <= (px / 2) ** 2,
                              255, 0).astype(np.uint8)
        self.img = ImageMobject(self._pixels(np.tile(fl.rgb(RED_PX), (LENS_TILES, LENS_TILES, 1))))
        self.img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.img.set_height(2 * radius)
        c = np.array([centre[0], centre[1], 0.0])
        self.img.move_to(c)
        t = np.array([target[0], target[1], 0.0])
        d = (t - c) / max(np.linalg.norm(t - c), 1e-9)
        self.add(self.img, Circle(radius=radius, stroke_color=INK, stroke_width=3).move_to(c),
                 Line(c + d * radius, t - d * target_r, stroke_color=INK, stroke_width=1.5),
                 Circle(radius=target_r, stroke_color=INK, stroke_width=2).move_to(t))
        for m in self.get_family():
            m.set_z_index(z)
        self.add_updater(lambda m, dt: None)

    def _pixels(self, colors: np.ndarray) -> np.ndarray:
        rgb = colors[self.idx][:, self.idx].astype(np.float32)
        rgb[self.grid] += (fl.rgb(fl.EMPTY).astype(np.float32) - rgb[self.grid]) * 0.6
        return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), self.alpha])

    def show(self, colors: np.ndarray) -> None:
        arr = self._pixels(colors)
        self.img.pixel_array = arr
        self.img.orig_alpha_pixel_array = arr[:, :, 3].copy()


class Rounds(ChainScene):
    k = 15

    # ---- the inset's line (only the line changes; the thumbnail stays, as r03)
    def inset_line(self, line: str, out_frames: int = 2, in_frames: int = 3, extra_out=()) -> float:
        """Out with the old line (and `extra_out`), then in with the new one;
        returns the time the new line is fully in."""
        old, new = self.over["inset"], ov.inset(self.G, line)
        self.play(FadeOut(old[-1]), *extra_out, run_time=fr(out_frames))
        self.play(FadeIn(new[-1]), run_time=fr(in_frames))
        self.remove(*old.submobjects, new[-1])
        self.add(new)
        self.over["inset"] = new
        return self.elapsed()

    def counter_anchor(self, longest: str) -> np.ndarray:
        """Left end of the round counter, clear of the caption, the inset, the
        act tag, the lens and the boards in both layouts. Iso: beside the
        floor, under its front-right edge; flat: in the floor's own margin by
        the square, `longest` ending at the square's right edge (under it in
        16:9; over it in 9:16, where the caption sits right under the floor)."""
        if self.Lay.P.flat:
            t, l, b, r = st.SQUARE_BOX
            w = ov.lbl(longest, RoundCounter.SIZE, INK_SOFT, mono=True).width
            row = t / 2 if self.G.vertical else (b + 1) / 2
            return self.floor.frac_point(row, r) - np.array([w, 0.0, 0.0])
        return self.floor.frac_point(1.0, 0.62) + np.array([0.0, -0.55, 0.0])

    def beat_anim(self) -> None:
        n = self.num
        rpx = self.floor.rpx
        cue = self.cue
        a3 = st.a3()

        # ---- the numbers on screen and in the picture
        rounds = int(n["big_tiles_rounds"])                 # 127
        rings = int(n["big_levels8_v5"])                    # 4,001 (today's session)
        fo = a3["sq_first_open"].astype(int)
        top = int(fo.max())                                 # 125: the last round that opens a tile
        assert top + 2 == rounds, "the square: 126 rounds open tiles, the 127th changes nothing"
        targets = st.ctx_targets()
        c4 = st.ctx_centre_round()                          # 4: the centre tile's round
        opened5 = {tuple(t) for t in st.ctx_round_tiles(c4 + 1)}
        assert opened5 == {t for t, _ in targets}, "round 5 opens exactly the border bits' targets"
        seeds = np.argwhere(fo == 0)
        assert len(seeds) == 1, "round 0 opens the seed tile alone"
        half = LENS_TILES // 2
        (si, sj) = (int(v) for v in seeds[0])
        win = (slice(si - half, si + half + 1), slice(sj - half, sj + half + 1))
        di, dj = np.abs(np.mgrid[-half:half + 1, -half:half + 1])
        assert (fo[win] == np.maximum(di, dj)).all(), "in the lens, round k is the ring k tiles out"

        # ---- the words
        t_bits, t_into = cue("border"), cue("into")
        t_join = cue("join")
        t_one, t_gate, t_round = cue("One"), cue("gate"), cue("round")
        t_on, t_27 = cue("On"), cue("twenty-seven")
        t_ms, t_but, t_seed = cue("milliseconds"), cue("But"), cue("seed")
        t_one_seed = cue("one", after=cue("from", after=t_but))

        # ---- the square's clock: the lens, two slow rounds, then the race
        t_lens = t_on + fr(INSET)                           # once the inset reads "stress test"
        t_r0 = t_lens + fr(LENS_IN) + 0.1                   # round 0 once the lens is in
        t_r1 = t_r0 + SLOW
        t_race = t_r1 + SLOW
        f0, f1 = t_r0 + FLASH_AFTER, t_r1 + FLASH_AFTER     # round 1's gate rises into the hold
        h1 = t_27 + fr(CAP_IN)                              # the counter lands as caption 2 is fully in
        f_ctx = max(t_gate, t_one + fr(BLUE_IN))            # the context's round 5: blue first, then the gate
        self.gate_rounds(flashes=[f_ctx, f0], hold=(f1, h1))

        def round_at(t: float) -> int:
            """The last round whose work is on the square at scene time t (-1: none)."""
            if t < t_r0:
                return -1
            if t < t_r1:
                return 0
            if t < t_race:
                return 1
            q = min((t - t_race) / (h1 - t_race), 1.0)
            return min(rounds - 1, 2 + int(smooth(q) * (rounds - 2)))

        # ---- the rasters
        cells = st.ctx_cells(rpx)
        segs = np.concatenate([st.tile_outline_segs(i, j) for (i, j), _ in targets])
        line_w = max(2.0, rpx / 450)

        def ctx(bits: float, outline: float) -> np.ndarray:
            out = st.tile_ctx_raster(rpx, upto=c4, border_bits=bits)
            if outline > 0:
                fl.segments(out, cells, segs, INK, width=line_w, alpha=outline)
            return out

        ctx5 = st.tile_ctx_raster(rpx, upto=c4 + 1)
        sq_shape = fo.shape
        sq_cells = st.square_cells(rpx)

        def square(upto: int) -> np.ndarray:
            return fl.cells_raster(st.square_tiles_colors(upto), sq_cells)

        # ---- "Only the tile's border bits go out": they light; r10's "4 trips" goes
        self.until(t_bits)
        self.floor_play(lambda p: round(p * BITS_IN), lambda i: ctx(i / BITS_IN, 0.0), fr(BITS_IN) + EPS)
        self.unsay(frames=4)

        # ---- "into the tiles around it": an arrow into each tile they open, each
        # outlined; the round's 12 tile tokens are consumed
        self.until(t_into)
        arrows = tile_arrows(self.Lay.floor_pl, rpx)
        spec = self.machine.spec
        self.floor_play(lambda p: round(p * TARGETS_IN), lambda i: ctx(1.0, 0.85 * i / TARGETS_IN),
                        fr(TARGETS_IN) + EPS)
        self.play(*[GrowArrow(a) for a in arrows], *self.machine_anims(replace(spec, tokens=())),
                  run_time=fr(TARGETS_IN))
        self.extras = [arrows]

        # ---- "those join the next round's queue": 4 tokens drop in, one per target
        self.until(t_join)
        self.machine_to(replace(spec, tokens=(RED_PX,) * len(targets)), frames=6, slide=0.3)

        # ---- "One gate per round": round 5 opens the 4 tiles (their tokens go), then the gate flashes once
        self.until(t_one)
        self.floor_blend(ctx(1.0, 0.85), ctx5, frames=BLUE_IN)
        self.extras = []
        self.play(FadeOut(arrows), *self.machine_anims(replace(spec, tokens=())), run_time=fr(BLUE_IN))
        self.say(f"{int(n['big_tiles_one_sync'])} gate", "per round", at=t_round)

        # ---- "On the giant square": the inset names it, the square swells out of the middle
        self.until(t_on)
        red = st.square_tiles_colors(-1)

        def swell(i: int) -> np.ndarray:
            p = smooth(i / SWELL)
            where = fl.Cells(rpx, sq_shape, boxed(st.SQUARE_BOX, 0.1 + 0.9 * p))
            return st.mix(ctx5, fl.cells_raster(red, where), p)

        self.floor_play(lambda p: round(p * SWELL), swell, fr(SWELL) + EPS)
        self.until(max(t_on, self._cap_t + DWELL - fr(INSET)))
        t_text = self.inset_line("stress test")
        assert t_text <= t_lens + HALF_FRAME, ("the inset came in late; the lens's clock assumes it", t_text, t_lens)

        # ---- the lens grows out of the seed tile: a round's ring of tiles is a
        # few output pixels on the floor, one tile is a square in the lens
        tgt = self.floor.frac_point(*sq_cells.frac(si + 0.5, sj + 0.5))
        reach = [np.linalg.norm(self.floor.frac_point(*sq_cells.frac(si + 0.5 + a, sj + 0.5 + b)) - tgt)
                 for a, b in ((LENS_TILES / 2, 0), (0, LENS_TILES / 2))]
        (lc, lr) = LENS[self.G.vertical]
        lens = TileLens(lc, lr, tgt, max(0.06, float(np.mean(reach))))
        self.until(t_lens)
        self.play(FadeIn(lens, scale=0.15, target_position=tgt), run_time=fr(LENS_IN))
        live = {"lens": True}

        # the real rounds on the square and in the lens: 0 and 1 slow, then
        # the race; the counter beside the floor
        counter = RoundCounter("round 1", self.counter_anchor(f"round {rounds}"))
        last = {"floor": None, "lens": None}

        def race(p: float) -> None:
            t = t_r0 + p * (h1 - t_r0)
            r = round_at(t)
            if r != last["floor"]:
                self.floor.show_raster(square(min(r, top)))
                last["floor"] = r
            k = min(r, half)
            if live["lens"] and k != last["lens"]:
                lens.show(st.square_tiles_colors(k)[win])
                last["lens"] = k
            counter.set(f"round {r + 1}")

        self.run(race, h1 - t_r0, start=t_r0)
        self.until(t_r0)
        self.play(FadeIn(counter), run_time=fr(3))
        self.extras = [counter]

        # the race: the queue refills with the round's tiles
        self.until(t_race)
        self.machine_to(st.act3_machine("tile"), frames=6)

        # ---- "a hundred twenty-seven rounds": caption 2 lands with the counter,
        # the lens folds back into the seed tile with it
        assert h1 - fr(CAP_IN) >= t_text + DWELL - fr(CAP_IN) - EPS, "caption 2 keeps MIN_DWELL from the inset"
        live["lens"] = False
        self.say(f"{rounds} rounds", f"was {rings:,} rings", at=h1 - fr(CAP_IN),
                 extra=[FadeOut(lens, scale=0.15, target_position=tgt)])
        assert abs(self._cap_t - h1) <= HALF_FRAME, ("caption 2 and the counter land apart", self._cap_t, h1)

        # ---- "from a hundred milliseconds to three": row 2's pill, no caption with it
        self.until(max(t_ms, self._cap_t + DWELL - fr(6)))
        old = self.over["strip"]
        self.over["strip"] = self.strip(st.score_rows(self.beat))
        self.play(FadeIn(self.over["strip"]), run_time=fr(6))
        self.remove(old)
        t_pill = self.elapsed()

        # ---- "But that's one blob": the square shrinks onto the frame's one
        # blob, landing as the ring round it; under "one seed" its seed comes
        # in as a dot and pulses twice. Ring and dot stay (k = 16 carries them)
        frame0 = st.one_blob_raster(rpx, ring=0.0, dot=0.0)
        frame = st.floor_raster("frame_one_blob", rpx)
        ringed = st.one_blob_raster(rpx, dot=0.0)
        full = st.square_tiles_colors()
        empty = frame0.copy()
        empty[..., :3] = fl.rgb(fl.EMPTY)
        box_at = onto_blob()
        t_pb = max(t_but - fr(3), t_pill + DWELL - fr(6))
        t_pulse = max(t_one_seed, t_pb + fr(PULL))
        t_end = t_pulse + PULSES * PULSE

        def pull(p: float) -> None:
            t = t_pb + p * (t_end - t_pb)
            if p >= 1:
                self.floor.show_raster(frame)
                return
            q = (t - t_pb) / (PULL / 15)
            if q < 1:
                out = st.mix(empty, frame0, smooth(min(q / 0.6, 1.0)))
                over(out, full, fl.Cells(rpx, sq_shape, box_at(smooth(q))), 1 - float(np.clip((q - 0.55) / 0.45, 0, 1)))
                self.floor.show_raster(st.one_blob_marks(out, ring=float(np.clip((q - 0.6) / 0.4, 0, 1)), dot=0.0))
                return
            x = (t - t_pulse) / PULSE
            if x < 0:
                self.floor.show_raster(ringed)
                return
            bump = float(np.sin(np.pi * x) ** 2) if x < PULSES else 0.0
            alpha = float(np.clip((t - t_pulse) / DOT_IN, 0, 1))
            self.floor.show_raster(st.one_blob_raster(rpx, dot=alpha, dot_scale=1 + 0.7 * bump))

        self.until(t_pb)
        self.run(pull, t_end - t_pb)
        self.extras = []
        gone = [FadeOut(counter)]
        if self.cap is not None:            # "127 rounds" leaves with the square
            gone.append(FadeOut(self.cap))
            self.cap = None
        t_text = self.inset_line("drone frame", out_frames=3, in_frames=3, extra_out=gone)

        # ---- "from one seed": the carried caption
        self.say_last(at=max(t_seed, t_text + DWELL - fr(5)))
