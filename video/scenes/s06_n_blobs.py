"""Stage 6, N blobs: two waves collide in a U and merge, then the real image."""
from __future__ import annotations

from manim import FadeIn, FadeOut

from scenes.stage import Replay, StageScene, fr


class NBlobs(StageScene):
    k = 6

    def first_caption(self) -> tuple:
        return "colliding waves merge in flight", "two waves, one label, no seam"

    def stage(self) -> None:
        box = self.geo.middle
        self.wait_replay()                                    # u_prov, provisional labels
        prov = self.replay
        final = Replay("ch05_u_final", box)
        final.show(final.n() - 1)                             # one colour, seam erased
        self.play(FadeOut(prov), FadeIn(final), run_time=fr(6))
        self.until(self.elapsed() + fr(12))
        blobs = Replay("ch05_input_blobs", box)
        self.swap_caption(self.cfg.caption, self.cfg.caption2,
                          extra_out=[FadeOut(final)], extra_in=[FadeIn(blobs)], frames=8)
        self.start_replay(blobs, 20.0)
        self.wait_replay()
