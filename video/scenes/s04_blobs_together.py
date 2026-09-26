"""Beat 04 `BlobsTogether`: the label rides inside the queue, colliding
waves merge in flight, and every blob floods in one launch.

Three phases, one clear picture each:
  1. (0-3.3 s)   queue of (x,y) entries carrying a coloured label chip,
                 beside the ch04 two-blob multi-source flood
  2. (3.3-7 s)   the ch05 U: two waves collide, then the seam is erased
  3. (7-end)     all 2,522 blobs of the real image flooding in one launch,
                 with the ch05 numbers on the right

Timing model: every play lasts a whole number of 1/15 s frames (whole
frames at 15, 30 and 60 fps too), nudged just below the frame boundary so
Manim's `np.arange` yields exactly that many frames. That keeps the
renderer clock (`elapsed()`) equal to the rendered length whether or not
partial movie files come from the cache, so `finish()` pads correctly.
"""
from __future__ import annotations

from manim import *
from scenes.style import *


# queue entries: (x,y) pair and the blob family whose label it carries.
# The coordinates are illustrative BFS neighbours around two seeds.
QUEUE_INITIAL = [("41,17", BLUE), ("73,29", GREEN), ("42,17", BLUE)]
QUEUE_PUSHES = [
    ("74,29", GREEN),
    ("40,17", BLUE),
    ("72,29", GREEN),
    ("41,18", BLUE),
    ("73,30", GREEN),
    ("41,16", BLUE),
    ("73,28", GREEN),
]

SLOT_W, SLOT_H, SLOT_BUFF = 0.95, 1.15, 0.10
N_SLOTS = 6

F = 1 / 15          # one low-quality frame; also 2 frames at 30 fps, 4 at 60
EPS = 1e-6


def fr(k: int) -> float:
    """Duration of exactly `k` low-quality frames (see module docstring)."""
    return k * F - EPS


class BlobsTogether(BeatScene):
    beat = "04_blobs_together"

    # ---- helpers -------------------------------------------------------
    def until(self, t: float) -> None:
        """Wait (frame-exact) until the renderer clock reaches `t` seconds.

        Single-frame gaps are skipped: `fr(1)` would sit below Manim's
        minimum run_time at 15 fps and get clamped (with a warning).
        """
        k = round((t - self.elapsed()) / F)
        if k >= 2:
            self.wait(fr(k), frozen_frame=False)

    @staticmethod
    def make_entry(xy: str, color: str) -> VGroup:
        text = label(xy, size=20, mono=True)
        chip = RoundedRectangle(
            corner_radius=0.06, width=0.5, height=0.2,
            stroke_width=0, fill_color=color, fill_opacity=1,
        )
        chip.next_to(text, DOWN, buff=0.14)
        return VGroup(text, chip)

    # ---- scene ---------------------------------------------------------
    def construct(self) -> None:
        L = self.L
        vertical = L["vertical"]

        # ================= phase 1: label rides inside the queue ==========
        slots = VGroup(*[
            RoundedRectangle(
                corner_radius=0.12, width=SLOT_W, height=SLOT_H,
                stroke_color=GREY, stroke_width=2, fill_opacity=0,
            )
            for _ in range(N_SLOTS)
        ]).arrange(RIGHT, buff=SLOT_BUFF)

        legend = label("label", size=18, color=GREY)
        legend.next_to(slots, LEFT, buff=0.2)
        queue = VGroup(legend, slots)

        cap_a = caption("label rides inside the queue")
        seq04 = FrameSequence("ch04_asym_multisource", height=3.6)
        cap_b = caption("two blobs, one launch", size=24)

        if vertical:
            queue.scale_to_fit_width(min(queue.width, L["w"] - 2 * L["margin"]))
            queue.move_to(UP * 3.2)
            cap_a.next_to(queue, DOWN, buff=0.35)
            seq04.move_to(DOWN * 1.6)
            cap_b.next_to(seq04, DOWN, buff=0.3)
        else:
            y_mid = 0.2
            # panel B (picture) on the right, panel A (queue) in the rest
            seq04.move_to(RIGHT * 3.6 + UP * y_mid)
            left_edge = -L["w"] / 2 + L["margin"]
            right_limit = seq04.get_left()[0]
            queue.move_to(RIGHT * (left_edge + right_limit) / 2 + UP * y_mid)
            cap_a.next_to(slots, DOWN, buff=0.4)
            cap_a.set_x(queue.get_x())
            cap_b.next_to(seq04, DOWN, buff=0.3)

        # entries follow whatever scale the queue ended up with (vertical
        # layouts shrink it to the frame width)
        entry_scale = slots[0].width / SLOT_W
        pitch = slots[1].get_x() - slots[0].get_x()

        entries: list[VGroup] = []
        for i, (xy, col) in enumerate(QUEUE_INITIAL):
            e = self.make_entry(xy, col).scale(entry_scale).move_to(slots[i])
            entries.append(e)
        # align the legend word with the chip row
        legend.set_y(entries[0][1].get_y())

        self.add(seq04)
        seq04.start(30)  # 97 frames at 30 fps = 3.23 s, then holds
        self.play(
            FadeIn(queue), FadeIn(cap_a), *[FadeIn(e) for e in entries],
            run_time=fr(6),
        )
        self.play(FadeIn(cap_b), run_time=fr(5))

        # new entries slide in from the right carrying their chip; once the
        # queue is full the front entry is popped and everything shifts left
        t_push = 12 * F  # 0.8 s
        for xy, col in QUEUE_PUSHES:
            self.until(t_push)
            anims = []
            if len(entries) == N_SLOTS:
                front = entries.pop(0)
                anims.append(FadeOut(front, scale=0.5))
                anims += [e.animate.shift(LEFT * pitch) for e in entries]
            new = self.make_entry(xy, col).scale(entry_scale).move_to(slots[len(entries)])
            anims.append(FadeIn(new, shift=LEFT * 0.7 * entry_scale))
            entries.append(new)
            self.play(*anims, run_time=fr(5))
            t_push += 5 * F

        # ================= phase 2: colliding waves merge ==================
        self.until(3.3)
        seq04.stop()
        phase1 = Group(queue, cap_a, seq04, cap_b, *entries)

        u_prov = FrameSequence("ch05_u_prov", height=5.0)
        u_final = FrameSequence("ch05_u_final", height=5.0).show(96)
        cap_u = caption("two waves, one label")
        if vertical:
            u_prov.move_to(UP * 0.4)
        else:
            u_prov.move_to(UP * 0.3)
        u_final.move_to(u_prov)
        cap_u.next_to(u_prov, DOWN, buff=0.4)

        # lagged fades: the new picture only appears once the old one is
        # three quarters gone, so the two never compete and there is no
        # black dip between them
        self.play(LaggedStart(FadeOut(phase1), FadeIn(u_prov), lag_ratio=0.75), run_time=fr(7))
        u_prov.start(45)  # 97 frames at 45 fps = 2.16 s
        t_u_start = self.elapsed()
        self.until(4.5)
        self.play(FadeIn(cap_u, shift=UP * 0.2), run_time=fr(5))
        # one extra frame so the integer frame index surely reaches 96
        self.until(t_u_start + u_prov.duration(45) + F)
        u_prov.stop()
        # the seam is erased: cross-fade to the one-colour final frame
        self.play(FadeIn(u_final), FadeOut(u_prov), run_time=fr(5))
        self.wait(fr(4), frozen_frame=False)  # hold the merged, one-label U

        # ================= phase 3: every blob in one launch ===============
        blobs = FrameSequence("ch05_input_blobs", height=5.8, pixelated=False)
        line1 = label("2,522 blobs · one launch", size=28, mono=True)
        line2 = label("755,577 blobs · 24.8 ms", size=28, mono=True)
        line2_a = VGroup(*line2[:12])   # "755,577 blobs"
        line2_b = VGroup(*line2[12:])   # "· 24.8 ms"
        cap_seeds = caption("no seeds given")

        if vertical:
            blobs.scale_to_fit_width(min(blobs.width, L["w"] - 2 * L["margin"]))
            blobs.move_to(UP * 2.4)
            lines = VGroup(line1, line2, cap_seeds).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
            lines.next_to(blobs, DOWN, buff=0.7)
            lines.set_x(0)
        else:
            # picture left of centre so the 5.9-wide mono lines stay inside
            # the right margin (line end at x ~ 6.3 < 6.51)
            blobs.move_to(LEFT * 3.2)
            x_text = blobs.get_right()[0] + 0.7
            line1.move_to(RIGHT * x_text + UP * 0.9, aligned_edge=LEFT)
            line2.move_to(RIGHT * x_text + UP * 0.0, aligned_edge=LEFT)
            cap_seeds.move_to(RIGHT * x_text + DOWN * 0.9, aligned_edge=LEFT)

        self.play(
            LaggedStart(FadeOut(Group(u_final, cap_u)), FadeIn(blobs), lag_ratio=0.75),
            run_time=fr(7),
        )
        blobs.start(30)  # 75 frames at 30 fps = 2.5 s, then holds

        self.until(7.1)
        self.play(FadeIn(line1, shift=LEFT * 0.2), run_time=fr(5))
        self.until(7.7)
        self.play(FadeIn(line2_a, shift=LEFT * 0.2), run_time=fr(5))
        self.until(8.7)
        self.play(FadeIn(cap_seeds, shift=LEFT * 0.2), run_time=fr(5))
        self.until(9.6)
        self.play(FadeIn(line2_b, shift=LEFT * 0.2), run_time=fr(5))

        self.until(10.0)
        blobs.stop()
        self.finish()
