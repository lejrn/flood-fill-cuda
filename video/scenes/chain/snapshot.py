"""One still of a chain state, for layout work and checks.

    STATE=7 LOOK=iso VIDEO_LAYOUT=vertical .venv/bin/python -m manim render -s -qh \
        --disable_caching -r 1080,1920 --media_dir media/chain_preview scenes/chain/snapshot.py Snapshot
"""
from __future__ import annotations

import os

from scenes.chain.state import chain_state
from scenes.style import BeatScene


class Snapshot(BeatScene):
    def construct(self) -> None:
        k = int(os.environ.get("STATE", "13"))
        self.add(*chain_state(k).mobs())
        self.wait(1 / 30)
