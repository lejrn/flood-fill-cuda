"""One still of a chain state, for layout work and checks.

    STATE=7 LOOK=iso VIDEO_LAYOUT=vertical .venv/bin/python -m manim render -s -qh \
        --disable_caching -r 1080,1920 --media_dir media/chain_preview scenes/chain/snapshot.py Snapshot

Optional, for the pieces a state does not show:
    FLOOR=<name>   the floor drawn as state.floor_raster(name) (one floor)
    PANEL=r10      r10's SM close-up (G.sm_side) and warp card (G.warp) open
    PANEL=r14      the r14 card (row 1 at its second flip step, tags, footnote)
"""
from __future__ import annotations

import os

from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain import state as st
from scenes.style import BeatScene


class Snapshot(BeatScene):
    def construct(self) -> None:
        k = int(os.environ.get("STATE", "13"))
        L = st.layout()
        s = st.chain_state(k, L)
        name = os.environ.get("FLOOR")
        if name:
            s.floors[0].show_raster(st.floor_raster(name, s.floors[0].rpx))
        self.add(*s.mobs())
        panel = os.environ.get("PANEL", "")
        G = L.G
        if panel == "r10":
            (wc, wh), (sc, sh) = G.warp, G.sm_side
            sm = mc.sm_closeup(L.machine_pl, (*sc, 0.0), sh, i=0)
            card = mc.warp_closeup(wc, wh, st.a3()["cu_mask"], st.a3()["cu_sweeps"][3])
            self.add(sm, mc.partition_light(sm, 0), card, card.row_bracket(13), card.glyph("diag"))
        elif panel == "r14":
            rows = st.score_rows("r14_reveal", 2)
            self.add(ov.score_card(G, rows, highlight=(0,), footnote=st.card_footnote()))
        self.wait(1 / 30)
