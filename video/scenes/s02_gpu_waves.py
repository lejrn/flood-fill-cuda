"""Beat 02: a GPU floods in waves, one block per SM, all 24 SMs together."""
from __future__ import annotations

from manim import *
from scenes.style import *

# newest BFS ring: BLUE lifted toward INK so the frontier reads as "lit"
FRONTIER = ManimColor(BLUE).interpolate(ManimColor(INK), 0.45).to_hex()
HUES = [BLUE, GREEN, PURPLE, TEAL, GOLD]

TILE_IDLE = dict(fill_color=GREY, fill_opacity=0.28, stroke_color=GREY, stroke_width=1.2, stroke_opacity=0.7)


def fit(group, box_w: float, box_h: float) -> float:
    """Scale `group` down (never up) so it fits inside box_w x box_h."""
    s = min(1.0, box_w / group.width, box_h / group.height)
    if s < 1.0:
        group.scale(s)
    return s


class GpuWaves(BeatScene):
    beat = "02_gpu_waves"

    def until(self, t: float) -> None:
        rem = t - self.elapsed()
        if rem > 1 / 60:
            self.wait(rem)

    def construct(self) -> None:
        L = self.L
        m = L["margin"]
        if L["vertical"]:
            box = (L["w"] - 2 * m, L["h"] / 2 - 1.5 * m)
            centre_a, centre_b = UP * L["h"] / 4, DOWN * L["h"] / 4
        else:
            box = (L["w"] / 2 - 1.5 * m, L["h"] - 2 * m)
            centre_a, centre_b = LEFT * L["w"] / 4, RIGHT * L["w"] / 4

        # ---- panel A: 12x12 diamond blob, BFS rings ----
        n, cell = 12, 0.375
        grid = pixel_grid(n, cell)
        lv = manhattan_levels(n, (5, 5))
        rings = [
            VGroup(*[grid[r * n + c] for r in range(n) for c in range(n) if lv[r, c] == k])
            for k in range(5)
        ]
        for ring in rings:
            ring.set_fill(RED_PX, opacity=1)
        cap_a = caption("the whole frontier at once").next_to(grid, DOWN, buff=0.4)
        panel_a = VGroup(grid, cap_a)
        s_a = fit(panel_a, *box)
        panel_a.move_to(centre_a)

        seq = FrameSequence("ch03_square_conn4", height=4.5 * s_a)
        seq.move_to(grid.get_center())

        # ---- panel B: 24 SM tiles + numbers ----
        tiles = VGroup(*[
            RoundedRectangle(corner_radius=0.08, width=0.55, height=0.55, **TILE_IDLE)
            for _ in range(24)
        ])
        tiles.arrange_in_grid(rows=4, cols=6, buff=0.16)
        cap_b = [
            caption("24 SMs"),
            caption("1 block = 1 SM = 4%"),
            caption("2 blocks = 2 SMs = 8%"),
            caption("all 24 SMs"),
        ]
        cap_b[0].next_to(tiles, DOWN, buff=0.4)
        for c in cap_b[1:]:
            c.move_to(cap_b[0])
        big = label("20× the CPU", size=56, mono=True).next_to(cap_b[0], DOWN, buff=0.75)
        small = caption("64 Mpx in 100 ms", size=24).next_to(big, DOWN, buff=0.22)
        panel_b = VGroup(tiles, *cap_b, big, small)
        fit(panel_b, *box)
        panel_b.move_to(centre_b)

        # ---- phase 1 (0-2.5 s): rings fill together ----
        self.play(FadeIn(grid), FadeIn(cap_a), FadeIn(tiles), FadeIn(cap_b[0]), run_time=0.25)
        for k in range(5):
            self.until(0.25 + 0.25 * k)
            anims = [rings[k].animate.set_fill(FRONTIER, opacity=1)]
            if k > 0:
                anims.append(rings[k - 1].animate.set_fill(BLUE, opacity=1))
            self.play(*anims, run_time=0.25)
        self.play(rings[4].animate.set_fill(BLUE, opacity=1), run_time=0.25)

        # ---- 2.5 s: replay takes over panel A, tile 0 lights ----
        self.until(2.5)
        self.play(
            FadeOut(grid),
            FadeIn(seq),
            tiles[0].animate.set_fill(BLUE, opacity=1).set_stroke(BLUE, opacity=1),
            FadeOut(cap_b[0]),
            FadeIn(cap_b[1]),
            run_time=0.4,
        )
        seq.start(fps=30)

        # ---- 4.5 s: tile 1 ----
        self.until(4.5)
        self.play(
            tiles[1].animate.set_fill(GREEN, opacity=1).set_stroke(GREEN, opacity=1),
            FadeOut(cap_b[1]),
            FadeIn(cap_b[2]),
            run_time=0.35,
        )

        # ---- 6 s: all 24 tiles, then the number ----
        self.until(6.0)
        self.play(
            LaggedStart(
                *[
                    t.animate.set_fill(HUES[i % 5], opacity=1).set_stroke(HUES[i % 5], opacity=1)
                    for i, t in enumerate(tiles)
                ],
                lag_ratio=0.04,
            ),
            FadeOut(cap_b[2]),
            FadeIn(cap_b[3]),
            run_time=0.8,
        )
        self.until(6.9)
        self.play(FadeIn(big, shift=UP * 0.15), run_time=0.45)
        self.until(7.4)
        self.play(FadeIn(small), run_time=0.3)

        seq.stop()
        self.finish()
