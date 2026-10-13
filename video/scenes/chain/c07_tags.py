"""c07: tags. Every queue entry carries its blob's label, so two blobs
share one launch and finish in the rings of the bigger one. But someone
still hands in every seed."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from scenes.chain import toy
from scenes.chain.base import ChainScene
from scenes.chain.state import (
    BLOB_HUES, GOLD, RED_PX, filled_picture, fmt_ms, label_colors, machine_spec, toys,
)
from scenes.chain import floor as fl


class Tags(ChainScene):
    k = 7

    def beat_anim(self) -> None:
        t = toys()
        n = self.num
        two, lab = t["two"], t["two_lab"]
        ids = sorted(int(v) for v in np.unique(lab[lab >= 0]))
        seeds = [toy.centre_seed(lab == v) for v in ids]
        multi = toy.bfs(two, seeds, 8).level
        rings = [int(toy.bfs(lab == v, [s], 8).level.max()) + 1 for v, s in zip(ids, seeds)]
        top = int(multi.max())
        end = machine_spec(8)
        self.until(self.cue("tag"))
        self.machine_to(replace(end, launches=machine_spec(7).launches), frames=7)
        self.say("tags", "a blob label in every queue entry", at=self.cue("whose"))
        self.until(self.cue("Two"))
        red = self.toy(label_colors(two, np.full(two.shape, -1)))
        self.floor_blend(self.pic(filled_picture()), red, frames=5)
        # both blobs fill at once, each in its own colour
        full = label_colors(two, lab)
        t0, t1 = self.cue("share"), self.cue("bigger", end=True)

        def render(i: int):
            col = label_colors(two, np.full(two.shape, -1))
            done = (multi >= 0) & (multi <= i)
            col[done] = full[done]
            return self.toy(col, dots=((multi == i) & (i < top)).astype(int))

        self.until(t0)
        self.floor_play(lambda p: round(p * top), render, t1 - t0)
        self.until(self.cue("launch"))
        self.machine_to(end, frames=5)
        self.goal_to(("~" + fmt_ms(n["c07_ms"]), GOLD))
        a, b = rings
        self.say("max, not sum", f"rings: max({a}, {b}) = {max(a, b)}, not {a + b}", at=self.cue("bigger"))
        # someone hands in the seeds
        self.settle()
        marks = np.zeros(two.shape, int)
        for s in seeds:
            marks[s] = 3
        self.until(self.cue("someone"))
        self.floor_show(self.toy(full, dots=marks))
        self.say_last(at=self.cue("hand"))
        self.until(self.cue("seed", end=True))
        self.floor_show(self.toy(full))
