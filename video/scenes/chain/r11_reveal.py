"""r11: the reveal, the last beat. The full scoreboard opens over the board
with a one-line footnote of sources; its rows light up as they are named
("small": the CPU row, "huge": the size row, "many": the count row). At
"Every drone" the card closes and the floor turns into the live drone
footage (labels from the GPU, 30 fps), captioned with the measured drone
frame times. A flat title card fades in over the drones for the last
seconds; base.construct fades everything out (no snap: nothing follows).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, RoundedRectangle, SurroundingRectangle, VectorizedPoint, VGroup,
                   config)
from PIL import Image

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, fr
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


def score_card(G, rows: tuple, highlight: int | None = None, footnote: str | None = None) -> VGroup:
    """ov.score_card with one fix: the head's empty first cell is a point
    (an empty Text has no points, so Manim cannot place it)."""
    lines = VGroup()
    head = VGroup(VectorizedPoint(), label("CPU", 16, INK_SOFT), label("GPU", 16, INK_SOFT),
                  label("winner", 16, INK_SOFT))
    lines.add(head)
    for i, r in enumerate(rows):
        col = TEAL if r.winner == "GPU" else GREY
        name = VGroup(label(r.what, 20, INK if i == highlight else INK_SOFT),
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
    k = 13

    def card(self, h: int) -> VGroup:
        """The full scoreboard with row h framed (grey: the CPU's row, teal: the GPU's)."""
        rows = score_rows()
        foot = "; ".join(f"row {i + 1}: {r.tag}" for i, r in enumerate(rows))
        c = score_card(self.G, rows, highlight=h, footnote=foot)
        line = c[1][1][1 + h]                      # box, body -> (title, lines, foot); lines[0] is the head
        col = TEAL if rows[h].winner == "GPU" else GREY
        frame = SurroundingRectangle(line, color=col, buff=0.09, corner_radius=0.08, stroke_width=2.5)
        frame.set_z_index(Z_OVER + 21)
        c.add(frame)
        if not self.G.vertical:          # 16:9 has room: fill the band between the top overlays and the caption
            s = min(1.6, 5.2 / c.height, 10.0 / c.width)
            if s > 1:
                c.scale(s, about_point=self.G.card_c)
            # cover the min unit whole, not part of it (the tag and the gauge
            # under the card's top band fade out with it: see behind())
            m = self.machine.part("min") if self.machine else VGroup()
            if len(m.get_family()) and m.get_bottom()[1] < c.get_top()[1] and m.get_top()[1] > c.get_bottom()[1]:
                c.shift(RIGHT * max(0.0, m.get_right()[0] + 0.15 - c.get_right()[0]))
        return c

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

    def behind(self, c: VGroup) -> list:
        """The overlays the card would cut through (16:9: the act tag and the
        room gauge under its top band). They fade out while it shows, so no
        sliver of them peeks out or shows through."""
        out = []
        for name in ("tag", "gauge"):
            m = self.over.get(name)
            if m is None or not len(m.get_family()):
                continue
            if (m.get_bottom()[1] < c.get_top()[1] and m.get_top()[1] > c.get_bottom()[1]
                    and m.get_left()[0] < c.get_right()[0] and m.get_right()[0] > c.get_left()[0]):
                out.append(m)
        return out

    def swap_card(self, new: VGroup, frames: int = 4, extra=()) -> None:
        """The new card fades in over the old one, then the old one goes:
        the text never changes in place and the card never thins out."""
        self.play(FadeIn(new), *extra, run_time=fr(frames))
        self.remove(self.cur_card)
        self.cur_card = new

    def beat_anim(self) -> None:
        n, d = self.num, self.der
        end = self.target_seconds() + 0.4 - 0.5          # base fades out over the last 0.5 s

        # "One small blob: the CPU." - the full card opens on the CPU's row
        self.until(self.cue("One"))
        self.cur_card = self.card(0)
        hidden = self.behind(self.cur_card)
        self.say("the CPU", "one small blob", frames=5,
                 extra=[FadeIn(self.cur_card)] + [FadeOut(m) for m in hidden])

        # "One huge blob" - the size row; "or many at once" - the count row
        nxt = self.card(1)
        self.say("the GPU", "one huge blob, or many at once", at=self.cue("huge"), frames=5,
                 extra=[FadeIn(nxt)])
        self.remove(self.cur_card)
        self.cur_card = nxt
        self.until(self.cue("many"))
        self.swap_card(self.card(2), frames=4)

        # "Every drone, every frame." - the card closes onto the live drones
        t0 = self.cue("Every")
        self.until(t0)
        rpx = self.floor.rpx
        teal = floor_raster("frame_teal", rpx).astype(np.float32)     # the floor as the beat began
        blend_n = 6
        seconds = end - fr(1) - t0

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
        self.unsay(frames=2, extra=[FadeOut(self.cur_card)] + [FadeIn(m) for m in hidden])
        self.remove(self.cur_card)
        self.say("every drone frame",
                 f"median GPU {n['drone_gpu_ms']:.2f} ms (max {n['drone_gpu_max_ms']:.2f}), "
                 f"CPU {n['drone_cpu_ms']:.1f} ms; budget {int(d['budget_ms'])} ms",
                 frames=5)

        # the title card, never spoken, over the drones for the last seconds
        self.until(end - 1.5)
        self.play(FadeIn(self.title_card(), scale=1.04), run_time=fr(8))
