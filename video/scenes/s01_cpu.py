"""Beat 01 `Cpu`: a CPU walks the blob one pixel at a time.

Panel A: 12 x 12 pixel grid with a diamond blob (Manhattan radius 4,
41 red cells). A small white cursor visits the red cells in BFS order,
one cell per 0.07 s, turning each visited cell BLUE.
Panel B: `pure Python` stopwatch counting up to 24,083 ms (shown as
`24.1 s`, RED_PX) during the walk, then `Numba @njit` with `1,346 ms`.
"""
from __future__ import annotations

from manim import *
from scenes.style import *

N = 12            # grid side
SEED = (5, 5)     # centre of the diamond blob
RADIUS = 4        # Manhattan radius -> 1 + 4 * (1 + 2 + 3 + 4) = 41 cells
STEP = 0.07       # seconds per visited pixel
PY_MS = 24_083    # pure Python BFS
NJIT_MS = 1_346   # Numba @njit BFS

T_WALK_START = 0.4     # grid and stopwatch are on screen
T_NUMBA = 3.7          # brief: Numba line in the 3.5-5 s window


def fit(group: Mobject, max_w: float, max_h: float) -> Mobject:
    """Scale `group` down (never up) so it fits inside max_w x max_h."""
    s = min(max_w / group.width, max_h / group.height, 1.0)
    if s < 1.0:
        group.scale(s)
    return group


class Cpu(BeatScene):
    beat = "01_cpu"

    def construct(self) -> None:
        L = self.L
        m = L["margin"]
        usable_w, usable_h = L["w"] - 2 * m, L["h"] - 2 * m

        # ---- panel A: pixel grid with a diamond blob ----
        levels = manhattan_levels(N, SEED)
        order = sorted(
            [(r, c) for r in range(N) for c in range(N) if levels[r, c] <= RADIUS],
            key=lambda rc: (int(levels[rc]), rc[0], rc[1]),
        )
        assert len(order) == 41, len(order)

        grid = pixel_grid(N, 0.42, stroke=1.5)
        red_cells = VGroup(*[grid[r * N + c] for r, c in order])
        for sq in red_cells:
            sq.set_fill(RED_PX, opacity=1.0)
        cap_a = caption("one pixel at a time")
        cap_a.next_to(grid, DOWN, buff=0.4)
        panel_a = VGroup(grid, cap_a)

        # ---- panel B: two stopwatches stacked ----
        sw_py = Stopwatch("pure Python", 0, size=52, color=RED_PX)
        sw_nb = Stopwatch("Numba @njit", NJIT_MS, size=52, color=INK)
        panel_b = VGroup(sw_py, sw_nb).arrange(DOWN, buff=1.0)

        # ---- layout: left/right in landscape, top/bottom in vertical ----
        if L["vertical"]:
            gap = 1.0
            fit(panel_a, usable_w, usable_h * 0.55)
            fit(panel_b, usable_w, usable_h - panel_a.height - gap)
            VGroup(panel_a, panel_b).arrange(DOWN, buff=gap).move_to(ORIGIN)
        else:
            col_w = usable_w / 2
            fit(panel_a, col_w - 0.4, usable_h)
            fit(panel_b, col_w - 0.4, usable_h)
            panel_a.move_to(L["left"] + RIGHT * col_w / 2)
            panel_b.move_to(L["right"] + LEFT * col_w / 2)
            panel_b.set_y(grid.get_center()[1])

        # ---- the cursor: a small white square, sized after layout ----
        cs = grid[0].width
        cursor = Square(cs * 0.6, stroke_width=0, fill_color=INK, fill_opacity=1.0)
        cursor.move_to(red_cells[0])

        # ---- 0.0-0.4 s: picture and the empty stopwatch appear ----
        self.play(FadeIn(panel_a), FadeIn(sw_py), run_time=T_WALK_START)

        # ---- 0.4-3.27 s: BFS walk, one cell per 0.07 s, counter alongside ----
        walk_t = len(order) * STEP  # 41 * 0.07 = 2.87 s
        tr = ValueTracker(0.0)
        last = len(order) - 1

        # The updaters sit on the mobjects that change (grid, stopwatch):
        # the Cairo renderer bakes every mobject without an updater into a
        # static background for the whole play, so changes made to it from
        # another mobject's updater would not show until the play ends.
        def on_grid(_m: Mobject) -> None:
            t = tr.get_value()
            k = min(int(t / STEP + 1e-6), last)
            for i in range(k + 1):
                red_cells[i].set_fill(BLUE, opacity=1.0)
            cursor.move_to(red_cells[k])

        def on_clock(_m: Mobject) -> None:
            ms = PY_MS * min(tr.get_value() / walk_t, 1.0)
            if fmt_ms(ms) != sw_py.value.text:
                old = sw_py.value
                sw_py.set_ms(ms, RED_PX)
                # Cairo keeps the play-start family in its "moving" list, so
                # the replaced Text would still be drawn underneath the new
                # one. Blank it (same trick as manim's DecimalNumber.set_value).
                for mob in old.get_family():
                    mob.points[:] = 0

        grid.add_updater(on_grid)
        sw_py.add_updater(on_clock)
        self.add(cursor)
        self.play(tr.animate.set_value(walk_t), run_time=walk_t, rate_func=linear)
        grid.remove_updater(on_grid)
        sw_py.remove_updater(on_clock)
        self.remove(tr)
        for sq in red_cells:
            sq.set_fill(BLUE, opacity=1.0)
        sw_py.set_ms(PY_MS, RED_PX)   # exactly "24.1 s"

        # ---- 3.27-3.52 s: the cursor leaves, the blob is done ----
        self.play(FadeOut(cursor), run_time=0.25)

        # ---- 3.7-4.3 s: Numba line ----
        if T_NUMBA > self.elapsed():
            self.wait(T_NUMBA - self.elapsed())
        self.play(FadeIn(sw_nb, shift=UP * 0.2), run_time=0.6)

        # ---- hold, then pad and fade ----
        self.finish()
