"""r03: one giant blob. The drone inflates into the 64 Mpx test square; one
block (one team in room 0) sweeps it ring by ring and waits at its room
fence (syncthreads) after every ring, amber. Over twice the CPU, but one
SM of twenty-four.

Starts on chain_state(5) (the split drones, CPU + room 0) and ends on
chain_state(6) (the square painted, room 0 with its shelf and fence).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import DOWN, LEFT, FadeIn, FadeOut, config

from narration.common import read_words
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import ChainScene, _norm, fr
from scenes.chain.state import Z_OVER, boundary, floor_raster, level_colors, make_floors, toys
from scenes.style import INK_SOFT, OUT, label, voice

# The sweep runs on whole frames (at 15 fps; every step lasts twice as many
# frames at 30 fps), so every fence flash is drawn and preview = final.
SLOW = 3                # the first rings run slowly, so the fence flash can be read
SLOW_STEP = (4, 2)      # ... paint frames, then wait frames at the fence
FALLBACK_STEP = (3, 1)  # when even the slow rings do not fit
F15 = 15                # the plan's frame rate


def sweep_window(words: list) -> tuple[float, float]:
    """r03's sweep: from "block" to just after "twenty-four" starts.
    Words are matched like ChainScene.cue (punctuation ignored); a voice
    without them gets a 3 s sweep."""
    def at(w: str, n: int = 1) -> float | None:
        hits = [x for x in words if _norm(x[0]) == _norm(w)]
        return float(hits[n - 1][1]) if len(hits) >= n else None
    t0, t4 = at("block"), at("twenty-four")
    if t0 is None or t4 is None or t4 + 0.1 <= t0:
        t0 = t0 if t0 is not None else 0.0
        return t0, t0 + 3.0
    return t0, t4 + 0.1


def sweep_plan(n: int, frames: int) -> list[tuple[int, bool]]:
    """(ring, waiting at the fence) for each 15 fps frame of the sweep.

    The first SLOW rings get a step each (paint frames, then wait frames);
    the rest go g rings per step: one paint frame per ring, then one wait
    frame. g is the smallest group that fits in `frames`."""
    for slow_p, slow_w in (SLOW_STEP, FALLBACK_STEP):
        ns = min(SLOW, n)
        for g in range(1, n + 1):
            plan: list[tuple[int, bool]] = []
            for i in range(ns):
                plan += [(i, False)] * slow_p + [(i, True)] * slow_w
            for a in range(ns, n, g):
                rings = list(range(a, min(a + g, n)))
                plan += [(i, False) for i in rings] + [(rings[-1], True)]
            if len(plan) <= frames:
                return plan
    # nothing fits: one frame per ring, no waits
    return [(i, False) for i in range(n)]


def plan_index(p: float, seconds: float) -> int:
    """The 15 fps plan index shown at progress p of a run of `seconds`."""
    fps = config.frame_rate
    n = round(p * seconds * fps)
    return int(n * F15 // round(fps))


def r03_plan() -> list[tuple[int, bool]]:
    """r03's sweep plan (for r04, which runs it at twice the pace)."""
    words = read_words(OUT / voice()).get("r03_giant", [])
    t0, t1 = sweep_window(words) if words else (0.0, 3.0)
    lv = toys()["square4"].level
    return sweep_plan(int(lv.max()) + 1, int((t1 - t0) * F15))


def amber_fence(spec: mc.Spec, pl):
    """Room 0's fence in its waiting colour, invisible (flashed by a hook)."""
    g = mc.build(replace(spec, room_fence="wait"), pl, z0=100.0).part("room_fence")
    g.set_opacity(0)
    return g


class Giant(ChainScene):
    k = 5

    def beat_anim(self) -> None:
        t = toys()
        sq, lv = t["square"], t["square4"].level
        top = int(lv.max())
        nrings = top + 1
        b6 = boundary(6)
        side = int(self.num["big_rings4"]) - 1          # 4-way rings from the centre of a side x side square
        mpx = side * side / 1e6

        # ---- the inflate: the drone becomes the giant test square
        self.until(self.cue("giant"))
        left, right = self.floors
        new = make_floors(("square4",), self.Lay)[0]
        red = self.toy(level_colors(sq, np.full(sq.shape, -1)), floor=new)
        new.show_raster(red)
        ratio = new.width / right.width
        note = label(f"test square {side} × {side} = {mpx:.0f} Mpx, drawn small", 16, INK_SOFT, mono=True)
        note.next_to(new, DOWN, buff=0.05)
        for m in note.get_family():
            m.set_z_index(Z_OVER)
        # out first, then in: "CPU wins" leaves before the note arrives
        self.unsay(frames=2)
        anims = [FadeOut(left, shift=LEFT * 1.2),
                 FadeOut(right, scale=ratio, target_position=new.get_center()),
                 FadeIn(new, scale=1 / ratio, target_position=right.get_center()),
                 FadeIn(note)]
        # the CPU leaves; room 0 gets its shelf and its fence (at rest)
        anims += self.machine_anims(b6.machine)
        anims += self.over_anims("inset", ov.inset(self.G, *b6.inset))
        anims += self.over_anims("tag", ov.act_tag(self.G, *b6.tag))
        self.play(*anims, run_time=fr(10))
        self.set_floors([new])
        amber = amber_fence(b6.machine, self.Lay.machine_pl)
        self.add(amber)                       # over the fence at rest, so it draws on top
        self.extras = [note, amber]

        # ---- one block sweeps it ring by ring; the room fence flashes after each ring.
        # The plan is in whole frames, so every wait is drawn (and preview = final).
        t0, t1 = sweep_window(self.words)
        sweep = t1 - t0
        plan = sweep_plan(nrings, int(sweep * F15))

        def key(p: float):
            j = plan_index(p, sweep)
            return plan[j] if (j < len(plan) and p < 1) else ("done", False)

        def render(k):
            i, waiting = k
            if i == "done":
                return floor_raster("square4", new.rpx)
            dots = (lv == i).astype(int) * (2 if waiting else 1)
            return self.toy(level_colors(sq, lv, upto=i, top=top), dots=dots)

        def flash(p: float) -> None:
            amber.set_opacity(1.0 if key(p)[1] else 0.0)

        self.until(t0)
        self.floor_play(key, render, sweep)
        self.run(flash, sweep)

        self.until(self.cue("twice"))
        if self.G.vertical:                   # in 9:16 the note sits where the caption goes
            self.play(FadeOut(note), run_time=fr(2))
            self.extras.remove(note)
        self.say(f"{self.der['big_1block_x']:.2f}×", f"the CPU, one {mpx:.0f} Mpx blob, kernel time")
        self.until(self.cue("SM"))
        self.over_to(gauge=self.gauge(1), frames=6)
        self.say_last(at=self.cue("twenty-four"))
