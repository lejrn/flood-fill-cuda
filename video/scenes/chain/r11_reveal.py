"""r11: the reveal, the last beat. The full scoreboard opens over the floor
with a one-line footnote of sources, and its rows are framed as they are
named. "The CPU": the small-blob row, and the machine shows it (P-core 0
works alone, the SMs go dark, the blocks leave). "Huge": one frame round
the size and count rows, the P-core rests and the blocks land as the SMs
light back up, column by column; then "the GPU". At "Every drone" the
card closes, the floor turns into the live drone footage (labels from
the GPU, 30 fps) and the caption turns to the worst drone frame's GPU
time, all in one change. A
flat title card fades in over the drones for the last seconds;
base.construct fades everything out (no snap: nothing follows).
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from manim import (DOWN, LEFT, UP, AnimationGroup, FadeIn, FadeOut, RoundedRectangle, SurroundingRectangle,
                   VectorizedPoint, VGroup, config)
from PIL import Image

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, fr
from scenes.chain.state import Z_OVER, floor_raster, score_rows
from scenes.style import ASSETS, GREY, GRID, INK, INK_SOFT, TEAL, label

DRONE_FPS = 30
N_FRAMES = 240


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


def score_card(G, rows: tuple, highlight: tuple = (), footnote: str | None = None) -> VGroup:
    """ov.score_card as r10 draws it, opaque: the head's empty first cell is a
    point (an empty Text has no points, so Manim cannot place it), and the
    names of the rows in `highlight` are bright."""
    lines = VGroup()
    head = VGroup(VectorizedPoint(), label("CPU", 16, INK_SOFT), label("GPU", 16, INK_SOFT),
                  label("winner", 16, INK_SOFT))
    lines.add(head)
    for i, r in enumerate(rows):
        col = TEAL if r.winner == "GPU" else GREY
        name = VGroup(label(r.what, 20, INK if i in highlight else INK_SOFT),
                      label(r.picture, 13, INK_SOFT)).arrange(DOWN, buff=0.03, aligned_edge=LEFT)
        lines.add(VGroup(name, label(ov.fmt_ms(r.cpu_ms), 20, INK, mono=True),
                         label(ov.fmt_ms(r.gpu_ms), 20, INK, mono=True),
                         label(f"{r.winner} {r.factor}", 22, col, mono=True)))
    cols = [max(row[j].width for row in lines) for j in range(4)]
    xs = np.cumsum([0] + [c + 0.45 for c in cols[:-1]])
    lines.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    for row in lines:
        for j, m in enumerate(row):
            m.set_x(xs[j] + m.width / 2)
    body = VGroup(label("CPU vs GPU", 22, INK), lines).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    if footnote:
        body.add(label(footnote, 12, INK_SOFT, mono=True))
        body.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    room = config.frame_width - 0.9
    if body.width > room:
        body.scale_to_fit_width(room)
    box = RoundedRectangle(corner_radius=0.16, width=body.width + 0.6, height=body.height + 0.5,
                           fill_color=ov.PANEL, fill_opacity=1.0, stroke_color=GRID, stroke_width=2)
    box.move_to(G.card_c)
    body.move_to(box)
    g = VGroup(box, body)
    for m in g.get_family():
        m.set_z_index(Z_OVER + 20)
    return g


class Reveal(ChainScene):
    k = 15

    def card(self, rows_lit: tuple) -> tuple:
        """The full scoreboard and a frame round the named rows (grey: the
        CPU's row, teal: the GPU's), as two mobjects: Scene.remove does not
        take a group's members, so a frame that fades alone is never inside
        the card."""
        rows = score_rows()
        foot = "; ".join(f"row {i + 1}: {r.tag}" for i, r in enumerate(rows))
        c = score_card(self.G, rows, highlight=rows_lit, footnote=foot)
        named = VGroup(*[c[1][1][1 + h] for h in rows_lit])     # box, body -> (title, lines, foot); lines[0] is the head
        col = TEAL if rows[rows_lit[0]].winner == "GPU" else GREY
        frame = SurroundingRectangle(named, color=col, buff=0.09, corner_radius=0.08, stroke_width=2.5)
        frame.set_z_index(Z_OVER + 21)
        self.place(VGroup(c, frame))
        return c, frame

    def place(self, c: VGroup) -> None:
        """Over the floor, clear of the caption. 9:16: under the inset. 16:9:
        left of the boards, so the machine stays in view (the inset fades out
        under it, see behind())."""
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
        c.scale(s)
        cx = min(max(G.floor_c[0], x0 + c.width / 2), x1 - c.width / 2)
        cy = min(max(G.floor_c[1], y0 + c.height / 2), y1 - c.height / 2)
        c.move_to([cx, cy, 0])

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

    def swap_card(self, new: tuple, frames: int = 4, extra=()) -> None:
        """The new card fades in over the old one (the same text in the same
        place) as the frame moves (old out, new in), then the old card goes:
        nothing changes in place and the card never thins out."""
        card, frame = new
        self.play(FadeIn(card), FadeIn(frame), FadeOut(self.cur_frame), *extra, run_time=fr(frames))
        self.remove(self.cur_card)
        self.cur_card, self.cur_frame = card, frame

    def ripple(self, spec: mc.Spec) -> tuple:
        """machine_anims(spec) with the SMs' changes swept across the die
        column by column, left to right (as r01 does); the other parts first.
        New blocks land with their column, so an SM never lights empty: each
        column's cubes are a stand-in from the same boards (the same faces),
        swapped for the machine's own part after the play (see land())."""
        old = self.machine
        anims = self.machine_anims(spec, slide=0)
        name = {id(p): n for m in (old, self.machine) for n, p in m.parts.items()}
        new_teams = self.machine.parts.get("teams") if "teams" not in old.parts else None
        col_x = [round(mc.sm_rect(i).x, 2) for i in range(24)]
        cols: dict = {}
        for a in anims:
            if new_teams is not None and a.mobject is new_teams:
                continue
            n = name.get(id(a.mobject), "")
            x = col_x[int(n[4:])] if n[4:].isdigit() else -1.0
            cols.setdefault(x, []).append(a)
        stand = []
        if new_teams is not None:
            for x in sorted(set(col_x)):
                teams = tuple(t if col_x[i] == x else 0 for i, t in enumerate(spec.teams))
                if any(teams):
                    cubes = mc.build(replace(spec, teams=teams), self.Lay.machine_pl, z0=100.0).part("teams")
                    cols.setdefault(x, []).append(FadeIn(cubes, shift=DOWN * 0.3))
                    stand.append(cubes)
        return AnimationGroup(*[AnimationGroup(*cols[x]) for x in sorted(cols)], lag_ratio=0.35), stand

    def land(self, stand: list) -> None:
        """The blocks' stand-ins out, the machine's own part in (no change on screen)."""
        if stand:
            self.remove(*stand)
            self.add(self.machine.part("teams"))

    def title_card(self) -> VGroup:
        t1 = label("connected-component labelling", 40, INK)
        t2 = label("flood-fill-cuda", 30, TEAL, mono=True)
        txt = VGroup(t1, t2).arrange(DOWN, buff=0.18)
        room = 6.6 if self.G.vertical else 4.9          # 16:9: clear of the inset on the left
        if txt.width > room:
            txt.scale_to_fit_width(room)
        box = RoundedRectangle(corner_radius=0.16, width=txt.width + 0.6, height=txt.height + 0.5,
                               fill_color=ov.PANEL, fill_opacity=0.9, stroke_color=GRID, stroke_width=2)
        g = VGroup(box, txt)
        g.move_to(self.floor.image.get_center() + (0 if self.G.vertical else -0.35) * UP)
        txt.move_to(box)
        for m in g.get_family():
            m.set_z_index(Z_OVER + 30)
        return g

    def beat_anim(self) -> None:
        n = self.num
        end = self.target_seconds() + 0.4 - 0.5          # base fades out over the last 0.5 s
        m13 = self.machine.spec

        # "One small blob": the full card opens on the CPU's row (the last
        # caption leaves with the overlays the card would cut through)
        self.until(self.cue("One"))
        t_open = self.elapsed()
        self.cur_card, self.cur_frame = self.card((0,))
        hidden = self.behind(self.cur_card)
        both = VGroup(self.cur_card, self.cur_frame)          # one scale centre while it opens
        self.unsay(frames=6, extra=[FadeIn(both, scale=0.96)] + [FadeOut(m) for m in hidden])
        self.remove(both)
        self.add(self.cur_card, self.cur_frame)

        # "the CPU": P-core 0 works alone, the SMs go dark, the blocks leave
        cpu_only = replace(m13, cpu="lit", rooms=("off",) * 24, teams=(0,) * 24)
        self.say("the CPU", "small blob", at=max(self.cue("CPU"), t_open + MIN_DWELL), frames=6,
                 extra=self.machine_anims(cpu_only, slide=0))

        # "One huge blob, or many at once": one frame round both GPU rows; the
        # CPU's caption leaves (after its MIN_DWELL), the P-core rests and the
        # blocks land as the SMs light up, column by column
        self.until(max(self.cue("huge"), self._cap_t + MIN_DWELL))
        t_swap = self.elapsed()               # the swap is a text change: the GPU caption waits its dwell
        cap, self.cap = self.cap, None
        sweep, stand = self.ripple(m13)
        self.swap_card(self.card((1, 2)), frames=8, extra=[FadeOut(cap), sweep])
        self.land(stand)
        self.say("the GPU", "huge or many blobs", at=max(self.cue("GPU"), t_swap + MIN_DWELL))

        # "Every drone, every frame." - one change: the card closes onto the
        # live drones as the caption turns (when "the GPU" has had its dwell)
        self.until(max(self.cue("Every"), self._cap_t + MIN_DWELL))
        t0 = self.elapsed()                  # the drones still end on time if the cue was reached late
        rpx = self.floor.rpx
        teal = floor_raster("frame_teal", rpx).astype(np.float32)     # the floor as the beat began
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
            return (teal + (r.astype(np.float32) - teal) * a).astype(np.uint8)

        self.floor_play(key, render, seconds)
        self.say("every drone frame", f"GPU max {ov.fmt_ms(n['drone_gpu_max_ms'])}",
                 extra=[FadeOut(self.cur_card), FadeOut(self.cur_frame)] + [FadeIn(m) for m in hidden])

        # the title card, never spoken, over the drones for the last seconds
        self.until(max(end - 1.5, self._cap_t + MIN_DWELL))
        self.play(FadeIn(self.title_card(), scale=1.04), run_time=fr(8))
