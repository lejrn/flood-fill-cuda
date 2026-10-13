"""c09: the loop closes. A row's red pixels come in spans (runs); link the
runs that touch, keep the smaller tag, paint each run in one stroke. The
queue, the gate and the launch counter come off the machine; six short
kernels go on. 1.46 ms."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from scenes.chain import floor as fl
from scenes.chain import toy
from scenes.chain.base import ChainScene
from scenes.chain.state import (
    BLOB_HUES, TEAL, fmt_ms, label_colors, machine_spec, picture, teal_picture, toys,
)


class Runs(ChainScene):
    k = 9

    def beat_anim(self) -> None:
        t = toys()
        n, d = self.num, self.der
        many = t["many"]
        rs = toy.runs(many)
        labels, _ = toy.run_labels(many)
        rows = {}
        for run in rs:
            rows.setdefault(run.r, []).append(run)
        row = max(rows, key=lambda r: (len(rows[r]), -r))
        red = label_colors(many, np.full(many.shape, -1))
        teal = fl.rgb(TEAL)
        # one row, then every row, as runs
        self.until(self.cue("row"))
        line = np.zeros(many.shape, bool)
        line[row] = many[row]
        self.floor_blend(self.pic(picture("labels")), self.toy(red, ring=line), frames=6)
        self.until(self.cue("spans"))
        col = red.copy()
        for run in rows[row]:
            col[run.r, run.c0:run.c1 + 1] = teal
        self.floor_show(self.toy(col, boxes=[(x.r, x.c0, x.c1) for x in rows[row]]))
        self.say("runs", "a red span in a row", at=self.cue("spans", end=True))
        self.until(self.cue("Thirteen"))
        allc = red.copy()
        for run in rs:
            allc[run.r, run.c0:run.c1 + 1] = teal
        boxes = [(x.r, x.c0, x.c1) for x in rs]
        self.floor_show(self.toy(allc, boxes=boxes))
        self.say(f"{d['c09_runs_x']:.0f}× fewer", f"{int(n['red_px']):,} px, {int(n['n_runs']):,} runs",
                 at=self.cue("spans", n=2))
        # the wave's parts come off; plain kernels go on
        self.until(self.cue("Stop"))
        bare = replace(machine_spec(9), queue=False, tokens=(), gate=None, launches=None)
        self.machine_to(bare, frames=7)
        self.until(self.cue("Link"))
        self.machine_to(replace(bare, kernels=True, kernel_lit=2), frames=7)
        self.until(self.cue("touch"))
        self.machine_to(replace(bare, kernels=True, kernel_lit=3), frames=3, slide=0)
        order = sorted(set(labels))
        hue = {v: fl.rgb(BLOB_HUES[i % len(BLOB_HUES)]) for i, v in enumerate(order)}
        lab_col = red.copy()
        for run, v in zip(rs, labels):
            lab_col[run.r, run.c0:run.c1 + 1] = hue[v]
        self.until(self.cue("smaller"))
        self.floor_show(self.toy(lab_col, boxes=boxes))
        self.machine_to(replace(bare, kernels=True, kernel_lit=4), frames=3, slide=0)
        self.until(self.cue("paint"))
        self.machine_to(replace(bare, kernels=True, kernel_lit=5), frames=3, slide=0)
        self.floor_show(self.toy(lab_col))
        self.say("no gate", "6 short kernels, one after another", at=self.cue("No"))
        self.until(self.cue("one", after=self.cue("gate")))
        self.floor_blend(self.toy(lab_col), self.pic(teal_picture()), frames=8)
        self.machine_to(machine_spec(10), frames=4, slide=0)
        self.goal_to((fmt_ms(n["ch06_mask_ms"]), TEAL))
        self.say_last(at=self.cue("milliseconds"))
