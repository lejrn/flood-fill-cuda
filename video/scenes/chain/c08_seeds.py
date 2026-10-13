"""c08: the picture finds its own seeds (corners and a lattice), waves
start everywhere, and where two meet both keep the smaller tag (atomicMin).
One launch, 58 ms: still over one frame, because every red pixel still
rides the queue."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from scenes.chain import floor as fl
from scenes.chain import toy
from scenes.chain.base import ChainScene
from scenes.chain.state import (
    BLOB_HUES, GOLD, RED_PX, fmt_ms, label_colors, machine_spec, picture, toys,
)

STRIDE = 4      # the toy's lattice; the real run (split_L8) uses 8 on a 9000 x 9000 picture


class Seeds(ChainScene):
    k = 8

    def beat_anim(self) -> None:
        t = toys()
        n = self.num
        many = t["many"]
        cand = toy.candidates(many)
        lat = toy.lattice(many, STRIDE) & ~cand
        seeds = list(zip(*np.nonzero(cand | lat)))
        mf = toy.merge_fill(many, seeds, 8)
        order = sorted({int(v) for v in np.unique(mf.frames[0][mf.frames[0] >= 0])})
        hue = {v: fl.rgb(BLOB_HUES[i % len(BLOB_HUES)]) for i, v in enumerate(order)}
        spec = machine_spec(8)
        # the hand-fed parts go; the picture changes to many blobs
        self.until(self.cue("finds"))
        plain = replace(spec, printer=False, tag_tokens=(), tokens=())
        red = label_colors(many, np.full(many.shape, -1))
        self.floor_blend(self.toy(label_colors(t["two"], t["two_lab"])), self.toy(red), frames=6)
        self.machine_to(plain, frames=6)
        self.say("seeds", "found on the GPU", at=self.cue("seeds"))
        self.until(self.cue("top-left"))
        self.floor_show(self.toy(red, dots=cand.astype(int) * 3))
        self.until(self.cue("grid"))
        self.floor_show(self.toy(red, dots=(cand | lat).astype(int) * 3))
        # waves from every seed; a merge recolours to the smaller tag at once
        t0, t1 = self.cue("Where"), self.cue("Every")
        nf = len(mf.resolved)

        def render(i: int):
            r = mf.resolved[i]
            col = red.copy()
            for v in np.unique(r[r >= 0]):
                col[r == v] = hue[int(v)]
            return self.toy(col)

        self.until(t0)
        self.floor_play(lambda p: min(nf - 1, int(p * nf)), render, max(t1 - t0 - 0.2, 0.6))
        self.until(self.cue("smaller"))
        self.machine_to(replace(plain, min_unit=True), frames=6)
        self.say("min", "where waves meet, both keep the smaller tag", at=self.cue("tag"))
        self.settle()
        # the real picture, every blob in one launch
        self.until(self.cue("Every"))
        self.floor_blend(render(nf - 1), self.pic(picture("labels")), frames=8)
        self.machine_to(replace(plain, min_unit=True, launches="1"), frames=6)
        self.until(self.cue("fifty-eight"))
        self.goal_to((fmt_ms(n["ch05_ms"]), GOLD))
        self.say(fmt_ms(n["ch05_ms"]), "every blob, one launch", at=self.cue("milliseconds"))
        self.say(f"{int(n['ch05_ms'])} > 33", "still over one frame", at=self.cue("Still"))
        self.until(self.cue("every", after=self.cue("frame")))
        self.machine_to(machine_spec(9), frames=6, slide=0)
        self.say_last(at=self.cue("rides"))
