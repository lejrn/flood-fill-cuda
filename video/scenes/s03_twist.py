"""Beat 03 `Twist`: on the real image the per-blob GPU launches lose to the CPU.

Panel A: the real input image (red blobs on white).
Panel B: two proportional bars, CPU @njit 1,346 ms (INK) vs GPU ch03
~2,181 ms (RED_PX, longer), then the caption `one blob = one launch` and a
monospace counter that spins up to `2,522 launches`.
"""
from __future__ import annotations

from manim import *

from scenes.style import *

CPU_MS = 1346
GPU_MS = 2181          # estimate from the overview JSON, shown with a leading ~
LAUNCHES = 2522

BAR_MAX = 4.0          # length of the longest (GPU) bar, in scene units
BAR_H = 0.42
NUM_BUFF = 0.2         # gap between a bar tip and its number


class Twist(BeatScene):
    beat = "03_twist"

    def construct(self) -> None:
        vertical = self.L["vertical"]

        # ---- panel A: the real image -------------------------------------
        image = ImageMobject(str(ASSETS / "ch05_input_blobs" / "frame_000.png"))
        image.set_height(3.6 if vertical else 5.0)
        if vertical:
            image.move_to(self.L["top"] + DOWN * (image.height / 2))
        else:
            image.move_to(np.array([-3.8, 0.0, 0.0]))

        # ---- panel B geometry ---------------------------------------------
        if vertical:
            x0 = self.L["left"][0] + 0.3
            y_cpu_label, y_cpu_bar = image.get_bottom()[1] - 0.8, image.get_bottom()[1] - 1.35
            y_gpu_label, y_gpu_bar = y_cpu_bar - 0.8, y_cpu_bar - 1.35
            y_caption = y_gpu_bar - 1.1
            y_counter = y_caption - 0.9
        else:
            x0 = -0.5
            y_cpu_label, y_cpu_bar = 2.15, 1.55
            y_gpu_label, y_gpu_bar = 0.75, 0.15
            y_caption = -1.05
            y_counter = -1.95

        bar_max = BAR_MAX
        if vertical:
            # keep the GPU bar + its number inside the right margin
            bar_max = (self.L["right"][0] - x0) - NUM_BUFF - 2.7
        cpu_len = bar_max * CPU_MS / GPU_MS

        def bar(width: float, color: str, y: float) -> Rectangle:
            r = Rectangle(width=max(width, 0.01), height=BAR_H, stroke_width=0,
                          fill_color=color, fill_opacity=1.0)
            return r.move_to(np.array([x0, y, 0.0]), aligned_edge=LEFT)

        # CPU row
        cpu_label = label("CPU  @njit", size=30, color=INK)
        cpu_label.move_to(np.array([x0, y_cpu_label, 0.0]), aligned_edge=LEFT)
        cpu_bar = bar(cpu_len, INK, y_cpu_bar)
        cpu_num = label(fmt_ms(CPU_MS), size=34, color=INK, mono=True)
        cpu_num.next_to(cpu_bar, RIGHT, buff=NUM_BUFF)

        # GPU row (the bar grows in, the number rides at its tip).
        # Everything here is driven by ValueTrackers + always_redraw so the
        # end state is identical whether Manim renders the play frame by
        # frame or skips it as a partial-movie-file cache hit.
        gpu_label = label("GPU  ch03", size=30, color=RED_PX)
        gpu_label.move_to(np.array([x0, y_gpu_label, 0.0]), aligned_edge=LEFT)
        gpu_w = ValueTracker(0.02)
        gpu_op = ValueTracker(0.0)
        gpu_bar = always_redraw(lambda: bar(gpu_w.get_value(), RED_PX, y_gpu_bar))
        gpu_num = always_redraw(
            lambda: label("~" + fmt_ms(GPU_MS), size=34, color=RED_PX, mono=True)
            .set_opacity(gpu_op.get_value())
            .next_to(gpu_bar, RIGHT, buff=NUM_BUFF)
        )

        # ---- 0-0.5 s: image + CPU row --------------------------------------
        self.play(
            FadeIn(image),
            FadeIn(cpu_label),
            FadeIn(cpu_bar),
            FadeIn(cpu_num),
            run_time=0.5,
        )

        # ---- 0.5-2 s: the GPU bar grows past the CPU bar --------------------
        self.add(gpu_bar, gpu_num)

        def early(a: float) -> float:      # label + number fade in over the first 0.4 s
            return smooth(min(a / (0.4 / 1.5), 1.0))

        self.play(
            FadeIn(gpu_label, rate_func=early),
            gpu_op.animate(rate_func=early).set_value(1.0),
            gpu_w.animate(rate_func=smooth).set_value(bar_max),
            run_time=1.5,
        )
        # freeze the redraw so the final fade-out is not overwritten
        gpu_bar.clear_updaters()
        gpu_num.clear_updaters()

        # ---- 2.5 s: caption ------------------------------------------------
        self.wait(0.5)
        cap = caption("one blob = one launch")
        cap.move_to(np.array([x0, y_caption, 0.0]), aligned_edge=LEFT)
        self.play(FadeIn(cap, shift=UP * 0.15), run_time=0.4)

        # ---- 4 s: the launch counter spins to 2,522 -------------------------
        self.wait(1.1)
        counter_size = 40 if vertical else 48
        final = label(f"{LAUNCHES:,} launches", size=counter_size, color=INK, mono=True)
        final.move_to(np.array([x0, y_counter, 0.0]), aligned_edge=LEFT)
        anchor = np.array([final.get_right()[0], y_counter, 0.0])

        def counter_text(n: int) -> Text:
            t = label(f"{n:,} launches", size=counter_size, color=INK, mono=True)
            return t.move_to(anchor, aligned_edge=RIGHT)

        # One Text object mutated in place with `become`: the Cairo renderer
        # keeps drawing whatever family members existed when the play began,
        # so swapping submobjects in and out would leave a ghost on screen.
        count = ValueTracker(1)
        counter = counter_text(1)
        state = {"shown": 1, "since": 0.0}

        def tick(m: Text, dt: float) -> None:
            state["since"] += dt
            n = int(round(count.get_value()))
            if n == state["shown"] or state["since"] < 0.06:
                return
            m.become(counter_text(n))
            state["shown"] = n
            state["since"] = 0.0

        counter.add_updater(tick)
        self.add(counter)
        self.play(count.animate.set_value(LAUNCHES), run_time=1.6, rate_func=smooth)
        counter.clear_updaters()
        self.remove(counter)
        self.add(final)   # land exactly on 2,522

        self.finish()
