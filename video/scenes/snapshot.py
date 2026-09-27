"""One PNG of a pane state, for layout work without animating.

    STAGE=4 VOICE=kokoro .venv/bin/python -m manim render -s -qh --disable_caching \
        --media_dir media/landscape scenes/snapshot.py Snapshot

STAGE=k draws the picture at the start of stage k (`build_state(k)`);
LIVE=1 draws the middle of stage k instead (`build_live`, replay frame 48).
"""
from __future__ import annotations

import os

from scenes.style import BeatScene
from scenes.panes import data
from scenes.panes.geometry import pane_geometry
from scenes.stage import build_live, build_state


class Snapshot(BeatScene):
    beat = "snapshot"

    def construct(self) -> None:
        k = int(os.environ.get("STAGE", "1"))
        geo = pane_geometry(self.L)
        bench = data.load_bench()
        state = build_live(k, geo, bench) if os.environ.get("LIVE") else build_state(k, geo, bench)
        self.add(*state.all())
        self.wait(1 / 15 + 1e-6, frozen_frame=True)
