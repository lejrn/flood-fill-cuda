"""r11 (v5, act 3): one gate per round. A closed tile hands on only its
border bits, into the tiles they really open; those join the next round's
queue, and the whole chip waits at the gate once per round. On the giant
square that turns 4,001 rings into 127 rounds (101 ms into 3.12 ms). But
that is one blob, from one seed.

On the close-up blob's 7 x 7 tile context (k = 15, rounds 0-4 blue):
- "border bits": the closed centre tile's reached border cells light
  (31 top, 26 left, 18 right, 2 bottom: state.border_bits_mask);
- "into the tiles around it": an arrow into each of the 4 tiles they
  really open (state.ctx_targets: above 32 px, right 18, above-left 1,
  above-right 1), each target outlined; no arrow where the tiles are open
  already or no reached pixel touches;
- "join the next round's queue": the 12 tile tokens fade (consumed) and 4
  drop in;
- "One gate per round": the 4 tiles turn blue (round 5, exactly the 4
  targets) and the gate flashes once; then the context holds still
  through "not two per ring" (no replay of later rounds).

"On the giant square": the inset's line reads "stress test" and the
context gives way to the 64 Mpx square at act 1's footprint, swelling out
of the middle, all red. Its rounds are the real ones (TL sq_first_open,
250 x 250 tiles, shaded light to dark by the round a tile first opened):
round 0 (the seed tile, in a hollow white ring) and round 1 one at a time, one
gate flash each; then the gate holds at steady amber while the other
rounds race, a small counter beside the floor runs to 127 (the last round
opens nothing: 126 rounds of tiles, then one that changes nothing) and
the gate lowers as it lands. The queue refills to 12 tile tokens as the
race starts. "from a hundred milliseconds": row 2's pill turns to GPU
340×, the beat's one pill change, with no caption.

"But that's one blob": the caption and the counter go, the inset's line
returns to "drone frame", and the square shrinks away into the 81 Mpx
frame (state.one_blob_picture: the largest blob blue, the rest red). Its
seed (TL closeup.seed) is a white dot that pulses twice under "one seed"
(about 1.25 a second, a small element: the cut's photosensitivity rule)
and fades before the cut.

Every text change keeps MIN_DWELL from the last one fully in (captions,
the inset's line and the pill, as r03, r07 and r09 count them).

Starts on chain_state(15) (tile_ctx, "4 trips"), ends on chain_state(16)
(frame_one_blob, row 2 GPU 340×, "1 blob" / "from 1 seed").
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import Arrow, FadeIn, FadeOut, GrowArrow, VGroup, smooth

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain import state as st
from scenes.chain.base import EPS, MIN_DWELL, ChainScene, fr
from scenes.style import INK, INK_SOFT, RED_PX

# MIN_DWELL in whole frames (1.8 s): until() rounds 1.75 s down a frame. A
# plain float, so the hooks' times stay plain floats
DWELL = float(np.ceil(MIN_DWELL * 15) / 15)
Z_ARROW = st.Z_FLOOR + 5        # over the floor, under the machine and every overlay
ARROW_SPAN = (0.32, 0.9)        # an arrow from this far along the centre-to-target line to this far
BITS_IN = 6                     # frames: the border bits light
TARGETS_IN = 6                  # frames: the arrows grow, the targets' outlines come in
BLUE_IN = 5                     # frames: the 4 targets turn blue (round 5)
SWELL = 9                       # frames: the context gives way to the square
PULL = 10                       # frames: the square gives way to the frame
SLOW = 0.6                      # seconds per slow round on the square (one flash each, 0.6 s apart)
FLASH_AFTER = 0.15              # a round's flash this long after its tiles open: work, then the gate
RACE_AFTER = 0.55               # the race starts this long after the last slow flash (GATE_GAP is 0.5)
LAND_AFTER = 0.35               # the counter lands this long after "rounds" starts
PULSE = 0.8                     # seconds per pulse of the frame's seed dot (1.25 a second)
PULSES = 2
DOT_OUT = 0.25                  # seconds for the seed dot to fade after its last pulse


def boxed(box: tuple, s: float) -> tuple:
    """`box` (top, left, bottom, right) scaled by s about its centre."""
    t, l, b, r = box
    cy, cx, hy, hx = (t + b) / 2, (l + r) / 2, (b - t) / 2 * s, (r - l) / 2 * s
    return (cy - hy, cx - hx, cy + hy, cx + hx)


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
        act tag and the boards in both layouts. Iso: beside the floor, under
        its front-right edge; flat: in the floor's own margin under the
        square, `longest` ending at the square's right edge."""
        if self.Lay.P.flat:
            t, l, b, r = st.SQUARE_BOX
            w = ov.lbl(longest, RoundCounter.SIZE, INK_SOFT, mono=True).width
            return self.floor.frac_point((b + 1) / 2, r) - np.array([w, 0.0, 0.0])
        return self.floor.frac_point(1.0, 0.62) + np.array([0.0, -0.55, 0.0])

    def beat_anim(self) -> None:
        n = self.num
        rpx = self.floor.rpx
        cue = self.cue
        a3 = st.a3()

        # ---- the numbers on screen and in the picture
        rounds = int(n["big_tiles_rounds"])                 # 127
        rings = int(n["big_rings8"])                        # 4,001
        fo = a3["sq_first_open"].astype(int)
        top = int(fo.max())                                 # 125: the last round that opens a tile
        assert top + 2 == rounds, "the square: 126 rounds open tiles, the 127th changes nothing"
        targets = st.ctx_targets()
        c4 = st.ctx_centre_round()                          # 4: the centre tile's round
        opened5 = {tuple(t) for t in st.ctx_round_tiles(c4 + 1)}
        assert opened5 == {t for t, _ in targets}, "round 5 opens exactly the border bits' targets"

        # ---- the words
        t_bits, t_into = cue("border"), cue("into")
        t_join, t_queue = cue("join"), cue("queue")
        t_one, t_gate, t_round = cue("One"), cue("gate"), cue("round")
        t_on, t_27, t_rounds = cue("On"), cue("twenty-seven"), cue("rounds")
        t_ms, t_but, t_seed = cue("milliseconds"), cue("But"), cue("seed")
        t_one_seed = cue("one", after=cue("from", after=t_but))

        # ---- the square's clock: two slow rounds, then the race
        t_r0 = t_on + fr(SWELL) + 0.05                      # round 0 once the square is in
        t_r1 = t_r0 + SLOW
        f0, f1 = t_r0 + FLASH_AFTER, t_r1 + FLASH_AFTER
        h0 = f1 + RACE_AFTER
        h1 = t_rounds + LAND_AFTER
        f_ctx = max(t_gate, t_one + fr(BLUE_IN))            # the context's round 5: blue first, then the gate
        self.gate_rounds(flashes=[f_ctx, f0, f1], hold=(h0, h1))

        def round_at(t: float) -> int:
            """The last round whose work is on the square at scene time t (-1: none)."""
            if t < t_r0:
                return -1
            if t < t_r1:
                return 0
            if t < h0:
                return 1
            q = (t - h0) / (h1 - h0)
            return min(rounds - 1, 2 + int(q * (rounds - 2)))

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
        seed_rc = np.argwhere(fo == 0)[0] + 0.5             # the seed tile's centre (TL big seed (4000, 4000))
        dot_r = max(3.0, rpx / 190)
        ring_w = max(1.5, rpx / 700)

        def dot(out: np.ndarray, where: fl.Cells, rc, r: float, alpha: float) -> np.ndarray:
            if alpha > 0:
                fl.discs(out, where, [rc], r + ring_w, fl.EMPTY, alpha)
                fl.discs(out, where, [rc], r, INK, alpha)
            return out

        sq_cells = st.square_cells(rpx)
        mark_r = max(6.0, rpx / 70) / sq_cells.cell_px     # in tiles: rounds 0-2 fit inside (1, 3, 5 tiles)
        ang = np.linspace(0, 2 * np.pi, 33)
        mark = np.stack([seed_rc[0] + mark_r * np.cos(ang[:-1]), seed_rc[1] + mark_r * np.sin(ang[:-1]),
                         seed_rc[0] + mark_r * np.cos(ang[1:]), seed_rc[1] + mark_r * np.sin(ang[1:])], axis=1)

        def square(upto: int, s: float = 1.0, seed: float = 0.0) -> np.ndarray:
            """The square by round `upto`; seed: a white ring round the seed tile,
            hollow, so rounds 0 and 1 (1 and 9 tiles) show inside it."""
            where = fl.Cells(rpx, sq_shape, boxed(st.SQUARE_BOX, s))
            out = fl.cells_raster(st.square_tiles_colors(upto), where)
            if seed > 0:
                fl.segments(out, where, mark, INK, width=max(1.5, rpx / 450), alpha=seed)
            return out

        # ---- "Only the tile's border bits go out"
        self.until(t_bits)
        self.floor_play(lambda p: round(p * BITS_IN), lambda i: ctx(i / BITS_IN, 0.0), fr(BITS_IN) + EPS)

        # ---- "into the tiles around it": an arrow into each tile they open, each outlined
        self.until(t_into)
        arrows = tile_arrows(self.Lay.floor_pl, rpx)
        self.floor_play(lambda p: round(p * TARGETS_IN), lambda i: ctx(1.0, 0.85 * i / TARGETS_IN),
                        fr(TARGETS_IN) + EPS)
        self.play(*[GrowArrow(a) for a in arrows], run_time=fr(TARGETS_IN))
        self.extras = [arrows]

        # ---- "join the next round's queue": the 12 tile tokens are consumed, 4 drop in
        spec = self.machine.spec
        self.until(t_join)
        self.machine_to(replace(spec, tokens=()), frames=5)
        self.until(t_queue)
        self.machine_to(replace(spec, tokens=(RED_PX,) * len(targets)), frames=6, slide=0.3)

        # ---- "One gate per round": round 5 opens the 4 tiles, then the gate flashes once
        self.until(t_one)
        self.floor_blend(ctx(1.0, 0.85), ctx5, frames=BLUE_IN)
        self.extras = []
        self.play(FadeOut(arrows), run_time=fr(BLUE_IN))
        self.say(f"{int(n['big_tiles_one_sync'])} gate", "per round", at=t_round)

        # ---- "On the giant square": the inset names it, the square swells out of the middle
        self.until(t_on)
        red = st.square_tiles_colors(-1)

        def swell(i: int) -> np.ndarray:
            p = smooth(i / SWELL)
            where = fl.Cells(rpx, sq_shape, boxed(st.SQUARE_BOX, 0.1 + 0.9 * p))
            return st.mix(ctx5, fl.cells_raster(red, where), p)

        self.floor_play(lambda p: round(p * SWELL), swell, fr(SWELL) + EPS)
        self.until(max(t_on, self._cap_t + DWELL - fr(5)))
        t_text = self.inset_line("stress test")

        # the real rounds: 0 and 1 slow (one flash each), then the race, the
        # counter beside the floor; the seed's ring fades as the race starts
        counter = RoundCounter("round 1", self.counter_anchor(f"round {rounds}"))
        last = {"key": None}

        def race(p: float) -> None:
            t = t_r0 + p * (h1 - t_r0)
            r = round_at(t)
            seed = float(np.clip(1 - (t - h0) / 0.3, 0, 1))
            key = (r, round(seed * 8))
            if key != last["key"]:
                self.floor.show_raster(square(min(r, top), seed=key[1] / 8))
                last["key"] = key
            counter.set(f"round {r + 1}")

        self.run(race, h1 - t_r0, start=t_r0)
        self.until(t_r0)
        self.play(FadeIn(counter), run_time=fr(3))
        self.extras = [counter]

        # the race: the queue refills with the round's tiles
        self.until(h0)
        self.machine_to(st.act3_machine("tile"), frames=6)
        self.say(f"{rounds} rounds", f"was {rings:,} rings", at=max(t_27, t_text + DWELL - fr(7)))

        # ---- "from a hundred milliseconds to three": row 2's pill, no caption with it
        self.until(max(t_ms, self._cap_t + DWELL - fr(6)))
        old = self.over["strip"]
        self.over["strip"] = self.strip(st.score_rows(self.beat))
        self.play(FadeIn(self.over["strip"]), run_time=fr(6))
        self.remove(old)
        t_pill = self.elapsed()

        # ---- "But that's one blob": the square shrinks away into the frame,
        # its one blob blue; its seed's dot pulses under "one seed", then fades
        frame = st.floor_raster("frame_one_blob", rpx)
        full = st.square_tiles_colors()
        whole = fl.Cells(rpx, (1, 1))
        seed_fr = st.blob_seed_frac()
        t_pb = max(t_but - fr(3), t_pill + DWELL - fr(6))
        t_pulse = max(t_one_seed, t_pb + fr(PULL))
        t_end = t_pulse + PULSES * PULSE + DOT_OUT

        def pull(p: float) -> None:
            t = t_pb + p * (t_end - t_pb)
            q = min((t - t_pb) / fr(PULL), 1.0)
            if q < 1:
                s = smooth(q)
                where = fl.Cells(rpx, sq_shape, boxed(st.SQUARE_BOX, 1 - 0.65 * s))
                out = st.mix(fl.cells_raster(full, where), frame, s)
                self.floor.show_raster(dot(out, whole, seed_fr, dot_r, s))
                return
            if p >= 1:
                self.floor.show_raster(frame)
                return
            x = (t - t_pulse) / PULSE
            bump = float(np.sin(np.pi * x) ** 2) if 0 <= x < PULSES else 0.0
            alpha = 1.0 if x < PULSES else float(np.clip(1 - (t - t_pulse - PULSES * PULSE) / DOT_OUT, 0, 1))
            self.floor.show_raster(dot(frame.copy(), whole, seed_fr, dot_r * (1 + 0.7 * bump), alpha))

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
