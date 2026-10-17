"""r14: the reveal, the last beat (v5; from v4's r11_reveal). The full
scoreboard opens over the floor with its tags and its footnote of copy
times, and its rows are framed as they are named.

- "One huge blob, or many at once": the card opens, rows 2 and 3 framed
  in teal; the blocks land again on every SM, column by column.
- "the GPU": caption 1.
- "And one small blob?": the first step of row 1's flip. The CPU cell
  changes (0.121 to 0.192 ms, the 8-neighbour CPU without depth) with
  its tag; the GPU cell and the winner go to "?", and so does the strip's
  pill. The frame moves to row 1.
- "now": the second step. The GPU cell lands at 0.044 ms and the winner
  at "GPU 4.4×" (the pill too); the frame turns teal; caption 2.
- "copy": the footnote's first line (the small blob with its copy, and
  the CPU beside it) lights, on a band in the CPU's grey.
- "CPU": P-core 0 lights.
- "Every drone": the card closes onto the live drone footage (labels from
  the GPU, 30 fps), P-core 0 goes dark, the hidden overlays come back,
  caption 3 (the worst drone frame with the copies). The title card fades
  in over the drones for the last seconds; base.construct fades
  everything out (no snap: nothing follows).

Row 1 never changes shape: every card is the flip-2 card with row 1's
cells swapped in at the same left edges, so a "?" never moves a column.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from manim import DOWN, UP, AnimationGroup, FadeIn, FadeOut, RoundedRectangle, Text, VGroup, config
from PIL import Image

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import Z_MACHINE, Z_OVER, card_footnote, floor_raster, score_rows
from scenes.style import ASSETS, GREY, GRID, INK, INK_SOFT, TEAL, label

DRONE_FPS = 30
N_FRAMES = 240
TITLE = "flood fill, from pixels to tiles"
REPO = "flood-fill-cuda"
FRAME_BUFF = 0.08            # a row frame's margin round its cells


@lru_cache(maxsize=1)
def _footage_files() -> tuple:
    files = sorted((ASSETS / "drones_labels").glob("frame_*.png"))
    return tuple(files[:N_FRAMES])


@lru_cache(maxsize=4)          # floor_play asks for each frame once; keep only a few decoded
def drone_raster(i: int, rpx: int) -> np.ndarray:
    """Frame i of the labelled drone footage, whole (the frame that was timed):
    letterboxed on a square of the floor's empty colour, resized to the
    floor's raster, black sky lifted to the same empty colour."""
    files = _footage_files()
    im = Image.open(files[i % len(files)]).convert("RGB")
    w, h = im.size
    s = max(w, h)
    empty = fl.rgb(fl.EMPTY)
    sq = Image.new("RGB", (s, s), tuple(int(v) for v in empty))
    sq.paste(im, ((s - w) // 2, (s - h) // 2))
    sq = sq.resize((rpx, rpx), Image.BILINEAR)
    a = np.maximum(np.asarray(sq), empty[None, None, :])
    out = np.empty((rpx, rpx, 4), np.uint8)
    out[..., :3] = a
    out[..., 3] = 255
    return out


def _z(mob, z: float):
    for m in mob.get_family():
        m.set_z_index(z)
    return mob


class Reveal(ChainScene):
    k = 18

    # ---- the card
    def card(self, flip: int, lit: tuple) -> VGroup:
        """ov.score_card at row 1's flip step `flip` (score_rows), the names
        of the rows in `lit` bright, placed (place()). Built on the flip-2
        card, row 1's four cells swapped for this step's at the same left
        edges: the "?" cells are narrower than the times, and score_card
        would close the columns up round them."""
        G = self.G
        foot = card_footnote()
        c = ov.score_card(G, score_rows(self.beat, 2), highlight=lit, footnote=foot)
        if flip != 2:
            alt = ov.score_card(G, score_rows(self.beat, flip), highlight=lit, footnote=foot)
            row, new = c[1][1][1], alt[1][1][1]          # box, body -> (top, lines, foot); lines[0] is the head
            for j in range(4):
                m, ref = new[j], row[j]
                m.shift([ref.get_left()[0] - m.get_left()[0], ref.get_center()[1] - m.get_center()[1], 0])
            row.submobjects = list(new.submobjects)
        self.place(c)
        return c

    def place(self, c: VGroup) -> None:
        """Over the floor, clear of the caption. 9:16: under the inset. 16:9:
        left of the boards, so the machine stays in view (the inset and the
        act tag fade out under it, see behind()). Every card is the same
        size, so the first card's scale and centre serve them all."""
        if getattr(self, "_place", None) is None:
            G = self.G
            w, h = config.frame_width, config.frame_height
            y0 = G.caption_c[1] + (0.6 if G.vertical else 0.5)          # the caption's top, with a gap
            if G.vertical:
                x0, x1 = -w / 2 + 0.15, w / 2 - 0.15
                inset = self.over.get("inset")
                y1 = (inset.get_bottom()[1] if inset is not None else G.floor_c[1] + G.floor_wh[1] / 2) - 0.15
            else:
                x0 = -w / 2 + 0.2
                x1 = self.machine.part("boards").get_left()[0] - 0.15
                y1 = h / 2 - 1.15                                       # under the scoreboard strip
            s = min(1.0, (x1 - x0) / c.width, (y1 - y0) / c.height)
            cw, ch = c.width * s, c.height * s
            cx = min(max(G.floor_c[0], x0 + cw / 2), x1 - cw / 2)
            cy = min(max(G.floor_c[1], y0 + ch / 2), y1 - ch / 2)
            self._place = (s, np.array([cx, cy, 0.0]))
        s, centre = self._place
        c.scale(s)
        c.move_to(centre)

    def frame_round(self, c: VGroup, rows: tuple, color: str) -> RoundedRectangle:
        """A frame round the named rows of card c (teal: the GPU's, soft: not
        known yet), as wide as the table: a row of "?"s is framed like a
        full one. It is its own mobject: Scene.remove does not take a
        group's members, so a frame that fades alone is never inside the
        card."""
        lines = c[1][1]
        named = VGroup(*[lines[1 + h] for h in rows])
        x0, x1 = lines.get_left()[0] - FRAME_BUFF, lines.get_right()[0] + FRAME_BUFF
        y0, y1 = named.get_bottom()[1] - FRAME_BUFF, named.get_top()[1] + FRAME_BUFF
        f = RoundedRectangle(corner_radius=0.07, width=x1 - x0, height=y1 - y0, stroke_color=color, stroke_width=2.5,
                             fill_opacity=0)
        f.move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])
        return _z(f, Z_OVER + 21)

    def light_line(self, c: VGroup) -> VGroup:
        """The footnote's copy line lit: the line again in bright ink over a
        band in the CPU's grey (a band, not a frame: the footnote's two
        lines sit too close for a stroke between them)."""
        fn = c[1][2]
        line = fn if isinstance(fn, Text) else fn[0]
        gap = fn[0].get_bottom()[1] - fn[1].get_top()[1] if not isinstance(fn, Text) else 0.1
        pad = max(0.01, min(0.05, gap / 2))
        band = RoundedRectangle(corner_radius=0.03, width=line.width + 0.12, height=line.height + 2 * pad,
                                fill_color=GREY, fill_opacity=0.3, stroke_width=0).move_to(line)
        return VGroup(_z(band, Z_OVER + 20.5), _z(line.copy().set_color(INK), Z_OVER + 21))

    def behind(self, c: VGroup) -> list:
        """The overlays the card would cut through (the inset, the act tag).
        They fade out while it shows, so no sliver of them peeks out."""
        out = []
        for name in ("inset", "tag"):
            m = self.over.get(name)
            if m is None or not len(m.get_family()):
                continue
            if (m.get_bottom()[1] < c.get_top()[1] and m.get_top()[1] > c.get_bottom()[1]
                    and m.get_left()[0] < c.get_right()[0] and m.get_right()[0] > c.get_left()[0]):
                out.append(m)
        return out

    def swap_anims(self, card: VGroup, frame) -> list:
        """The new card fades in over the old one (the same text in the same
        place, so only what differs changes) as the frame moves (old out,
        new in). Call done_swap() after the play: the old card goes then,
        so the card never thins out."""
        anims = [FadeIn(card), FadeIn(frame), FadeOut(self.cur_frame)]
        self._old_card = self.cur_card
        self.cur_card, self.cur_frame = card, frame
        return anims

    def done_swap(self) -> None:
        self.remove(self._old_card)
        self._old_card = None

    # ---- the machine
    def reland(self, frames: int = 12) -> None:
        """The blocks land again on every SM, column by column, left to right
        (v4's sweep): per column, stand-in cubes drop in as stand-ins of the
        cubes there fade (both from the same spec, so the same faces); the
        machine's own cubes are back after the play, the same pixels."""
        spec = self.machine.spec
        pl = self.Lay.machine_pl
        own = self.machine.part("teams")
        col_x = [round(mc.sm_rect(i).x, 2) for i in range(24)]
        olds, news, cols = [], [], []
        for x in sorted(set(col_x)):
            teams = tuple(t if col_x[i] == x else 0 for i, t in enumerate(spec.teams))
            if not any(teams):
                continue
            s = replace(spec, teams=teams)
            o = mc.build(s, pl, z0=Z_MACHINE, only=("teams",)).part("teams")
            n = mc.build(s, pl, z0=Z_MACHINE, only=("teams",)).part("teams")
            olds.append(o)
            news.append(n)
            cols.append(AnimationGroup(FadeOut(o), FadeIn(n, shift=DOWN * 0.3)))
        self.remove(own)
        self.add(*olds)
        self.play(AnimationGroup(*cols, lag_ratio=0.35), run_time=fr(frames))
        self.remove(*olds, *news)
        self.add(own)

    # ---- the end
    def title_card(self) -> VGroup:
        t1 = label(TITLE, 40, INK)
        t2 = label(REPO, 30, TEAL, mono=True)
        txt = VGroup(t1, t2).arrange(DOWN, buff=0.18)
        room = 6.6 if self.G.vertical else 4.9          # 16:9: clear of the inset on the left
        if txt.width > room:
            txt.scale_to_fit_width(room)
        box = RoundedRectangle(corner_radius=0.16, width=txt.width + 0.6, height=txt.height + 0.5,
                               fill_color=ov.PANEL, fill_opacity=0.9, stroke_color=GRID, stroke_width=2)
        g = VGroup(box, txt)
        g.move_to(self.floor.image.get_center() + (0 if self.G.vertical else -0.35) * UP)
        txt.move_to(box)
        return _z(g, Z_OVER + 30)

    def beat_anim(self) -> None:
        n = self.num
        G = self.G
        end = self.target_seconds() + 0.4 - 0.5          # base fades out over the last 0.5 s
        m18 = self.machine.spec

        # "One huge blob, or many at once": the card opens (row 1 still as r02
        # left it), rows 2 and 3 framed in teal; the carried caption leaves
        # with the overlays the card would cut through
        self.until(self.cue("One"))
        self.cur_card = self.card(0, lit=(1, 2))
        self.cur_frame = self.frame_round(self.cur_card, (1, 2), TEAL)
        hidden = self.behind(self.cur_card)
        both = VGroup(self.cur_card, self.cur_frame)          # one scale centre while it opens
        self.unsay(frames=6, extra=[FadeIn(both, scale=0.96)] + [FadeOut(m) for m in hidden])
        self.remove(both)
        self.add(self.cur_card, self.cur_frame)

        # the blocks land again, column by column: the GPU's two rows
        self.until(max(self.cue("huge"), self.elapsed()))
        self.reland(frames=12)
        self.say("the GPU", "huge or many blobs", at=self.cue("GPU"))

        # "And one small blob?": row 1's first step. Only the CPU cell (and its
        # tag) and the "?"s change; the pill turns to "?" with it
        self.until(self.cue("And"))
        rows1 = score_rows(self.beat, 1)
        c1 = self.card(1, lit=(0,))
        anims = self.swap_anims(c1, self.frame_round(c1, (0,), INK_SOFT))
        self.play(*anims, *self.over_anims("strip", ov.score_strip(G, rows1)), run_time=fr(6))
        self.done_swap()

        # "In kernel time, the GPU now": the second step, the GPU's time and
        # the winner land, the frame and the pill turn teal; caption 2
        rows2 = score_rows(self.beat, 2)
        c2 = self.card(2, lit=(0,))
        anims = self.swap_anims(c2, self.frame_round(c2, (0,), TEAL))
        self.say("small blob too", "in kernel time", at=self.cue("now"), frames=6,
                 extra=anims + self.over_anims("strip", ov.score_strip(G, rows2)))
        self.done_swap()

        # "with the copy to the card": the footnote's copy line lights on a
        # band in the CPU's grey (the row's teal frame stays: both sides at once)
        self.until(self.cue("copy"))
        lit_line = self.light_line(self.cur_card)
        self.play(FadeIn(lit_line), run_time=fr(6))

        # "still the CPU": P-core 0 lights
        self.until(self.cue("CPU"))
        self.machine_to(replace(m18, cpu="lit"), frames=6, slide=0)

        # "Every drone, every frame." - one change: the card closes onto the
        # live drones as the caption turns, P-core 0 goes dark
        self.until(max(self.cue("Every"), self._cap_t + MIN_DWELL))
        t0 = self.elapsed()                  # the drones still end on time if the cue was reached late
        rpx = self.floor.rpx
        start = floor_raster("frame_labels", rpx).astype(np.float32)     # the floor as the beat began
        blend_n = 6
        seconds = fr(round((end - t0) * 15) - 1)       # on the frame grid: the last drone frame is the one before `end`

        def key(p: float):
            t = p * seconds
            return int(t * DRONE_FPS) % N_FRAMES, min(blend_n, int(t * 15))

        def render(k):
            i, b = k
            r = drone_raster(i, rpx)
            if b >= blend_n:
                return r
            a = b / blend_n
            return (start + (r.astype(np.float32) - start) * a).astype(np.uint8)

        self.floor_play(key, render, seconds)
        closing = [FadeOut(self.cur_card), FadeOut(self.cur_frame), FadeOut(lit_line)]
        self.say("every drone frame", f"with copies: {ov.fmt_ms(n['drone_e2e_max_ms'])}", frames=6,
                 extra=closing + [FadeIn(m) for m in hidden] + self.machine_anims(m18, slide=0))

        # the title card, never spoken, over the drones for the last seconds
        self.until(max(end - 2.0, self._cap_t + MIN_DWELL))
        self.play(FadeIn(self.title_card(), scale=1.04), run_time=fr(8))
