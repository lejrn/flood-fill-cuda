"""Beat 00 `Hook`: the real image, three numbers, a stopwatch at zero.

Panel A: `assets/ch05_input_blobs/frame_000.png` (red blobs on white),
height 5.5, slow Ken Burns scale 0.94 -> 1.0 across the beat.
Panel B: `81,000,000 px`, `2,522 red blobs`, `Stopwatch("time", 0)`,
fading in one after another in sync with the narration.
"""
from __future__ import annotations

from manim import *

from scenes.style import *

IMAGE = ASSETS / "ch05_input_blobs" / "frame_000.png"
IMAGE_H = 5.5
KB_START = 0.94  # Ken Burns start scale; ends at 1.0


class Hook(BeatScene):
    beat = "00_hook"

    # ---- helpers ----
    def _until(self, t: float) -> None:
        """Wait until scene time `t` (no-op if already past it)."""
        rem = t - self.elapsed()
        if rem > 1e-3:
            self.wait(rem)

    @staticmethod
    def _fit(mob: Mobject, w: float, h: float) -> Mobject:
        """Scale `mob` down (never up) so it fits inside w x h."""
        s = min(1.0, w / mob.width, h / mob.height)
        if s < 1.0:
            mob.scale(s)
        return mob

    def construct(self) -> None:
        L = self.L
        m = L["margin"]
        # The camera's frame is what is actually visible. Under `-r W,H`
        # (the vertical build) config and layout() keep 14.22 x 8 while the
        # Cairo camera keeps the width and grows the frame height to match
        # the pixel aspect ratio (14.22 x 25.28). In landscape both agree.
        fw, fh = self.camera.frame_width, self.camera.frame_height
        avail_w = fw - 2 * m
        avail_h = fh - 2 * m
        T = self.target_seconds()

        # ---- panel A: the real image, full size for layout ----
        im = ImageMobject(str(IMAGE)).set_height(IMAGE_H)

        # ---- panel B: three lines, left aligned ----
        line_px = label("81,000,000 px", size=56, mono=True)
        line_blobs = label("2,522 red blobs", size=44, color=RED_PX, mono=True)
        sw = Stopwatch("time", 0)
        sw.title.align_to(sw.value, LEFT)
        panel_b = VGroup(line_px, line_blobs, sw).arrange(DOWN, buff=0.55, aligned_edge=LEFT)

        # ---- arrange the two panels ----
        if L["vertical"]:
            # Stack, text flush with the image's left edge, then scale the
            # stack to fill ~80% of the width (capped by the height) so text
            # lands at about the same pixel size as in landscape.
            both = Group(im, panel_b).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
            both.scale(min(0.8 * avail_w / both.width, 0.8 * avail_h / both.height))
        else:
            both = Group(im, panel_b).arrange(RIGHT, buff=1.1)
            self._fit(both, avail_w, avail_h)
        both.move_to(ORIGIN)

        # ---- Ken Burns: scale 0.94 -> 1.0 about the image centre ----
        full_h = im.height
        centre = im.get_center().copy()
        im.set_height(full_h * KB_START).move_to(centre)

        def ken_burns(mob: Mobject, dt: float) -> None:
            f = min(max(self.elapsed() / T, 0.0), 1.0)
            mob.set_height(full_h * (KB_START + (1.0 - KB_START) * f)).move_to(centre)

        im.add_updater(ken_burns)

        # ---- 0-1.5 s: image, then the pixel count ----
        self.play(
            LaggedStart(
                FadeIn(im),
                FadeIn(line_px, shift=UP * 0.15),
                lag_ratio=0.5,
            ),
            run_time=1.2,
        )
        self._until(1.5)

        # ---- 1.5-3 s: the blob count ----
        self.play(FadeIn(line_blobs, shift=UP * 0.15), run_time=0.6)
        self._until(3.0)

        # ---- 3-4.8 s: the stopwatch at zero ----
        self.play(FadeIn(sw, shift=UP * 0.15), run_time=0.6)

        self.finish()
