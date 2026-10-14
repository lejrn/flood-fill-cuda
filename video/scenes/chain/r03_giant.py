"""r03: one giant blob. The drone swells into the 64 Mpx test square and
the CPU core goes dark. One block (the cube on SM 0) sweeps it ring by
ring and waits at SM 0's fence (__syncthreads) after every ring: the
fence flashes amber, the ring's workers wait amber on the floor. Over
twice the CPU, but one SM of twenty-four: the other 23 come up idle in a
wave, then go dark again.

Starts on chain_state(5) (the split drones, P-core 0 and SM 0 lit) and
ends on chain_state(6) (the square painted, SM 0 with its shelf and its
fence at rest).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import LEFT, AnimationGroup, FadeIn, FadeOut, config, smooth

from narration.common import read_words
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, _norm, fr
from scenes.chain.state import boundary, floor_raster, level_colors, make_floors, toys
from scenes.style import OUT, voice

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


def _plan(n: int, slow: int, step: tuple[int, int], g: int) -> list[tuple[int, bool]]:
    """`slow` rings of `step` (paint, wait) frames, then g rings per step:
    one paint frame per ring, then one wait frame."""
    plan: list[tuple[int, bool]] = []
    for i in range(slow):
        plan += [(i, False)] * step[0] + [(i, True)] * step[1]
    for a in range(slow, n, g):
        rings = list(range(a, min(a + g, n)))
        plan += [(i, False) for i in rings] + [(rings[-1], True)]
    return plan


def sweep_plan(n: int, frames: int) -> list[tuple[int, bool]]:
    """(ring, waiting at the fence) for each 15 fps frame of the sweep.

    The block syncs after every ring, so a wait after every ring comes
    first: the slow opening shrinks (SLOW rings, then fewer, then one ring
    of a shorter step, then none) before rings are grouped. Grouped, the
    first SLOW rings get a step each and the rest go g rings per step, g
    the smallest that fits in `frames`."""
    shrink = [(s, SLOW_STEP) for s in range(min(SLOW, n), 0, -1)] + [(min(1, n), FALLBACK_STEP), (0, FALLBACK_STEP)]
    for slow, step in shrink:
        plan = _plan(n, slow, step, 1)
        if len(plan) <= frames:
            return plan
    for step in (SLOW_STEP, FALLBACK_STEP):
        for g in range(2, n + 1):
            plan = _plan(n, min(SLOW, n), step, g)
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
    """SM 0's fence in its waiting colour, a hair above the fence at rest
    (so add order does not matter), invisible until `show_amber` lights it."""
    g = mc.build(replace(spec, room_fence="wait"), pl, z0=100.05).part("room_fence")
    show_amber(g, False)
    return g


def show_amber(g, on: bool) -> None:
    """Light or hide the amber fence. A Group's set_opacity only sets an
    attribute, so each face's fill is set (the faces have no stroke)."""
    for m in g.family_members_with_points():
        m.set_fill(opacity=1.0 if on else 0.0)


class Giant(ChainScene):
    k = 5

    def ripple(self, spec: mc.Spec, frames: int) -> None:
        """machine_to(spec), the SMs' changes swept across the die column by
        column, left to right (GPC 0 first), as r01 does."""
        old = self.machine
        anims = self.machine_anims(spec, slide=0)
        name = {id(p): n for m in (old, self.machine) for n, p in m.parts.items()}
        cols: dict = {}
        for a in anims:
            n = name.get(id(a.mobject), "")
            x = round(mc.sm_rect(int(n[4:])).x, 2) if n[4:].isdigit() else -1.0
            cols.setdefault(x, []).append(a)
        sweep = [AnimationGroup(*cols[x]) for x in sorted(cols)]
        self.play(AnimationGroup(*sweep, lag_ratio=0.35), run_time=fr(frames))

    def beat_anim(self) -> None:
        t = toys()
        sq, lv = t["square"], t["square4"].level
        top = int(lv.max())
        nrings = top + 1
        b6 = boundary(6)
        pl = self.Lay.machine_pl
        # the swell's machine: P-core 0 dark, SM 0 as it was
        dark = replace(self.machine.spec, cpu="off")

        # ---- "giant": the drone swells into the test square, the CPU core goes dark,
        # the inset and the act tag name the new picture
        self.until(self.cue("giant"))
        left, right = self.floors
        new = make_floors(("square4",), self.Lay)[0]
        new.show_raster(self.toy(level_colors(sq, np.full(sq.shape, -1)), floor=new))
        ratio = new.width / right.width
        # only the inset's line changes (the thumbnail stays): out with the caption, then in.
        # The new line and the act tag are fully in by the swell's middle, and the next
        # text keeps MIN_DWELL from there (as say counts it: from fully in)
        swell, text_in = 10, 5
        old_inset, new_inset = self.over["inset"], ov.inset(self.G, b6.inset)
        self.unsay(frames=2, extra=[FadeOut(old_inset[-1])])
        t_text = self.elapsed() + fr(text_in)
        texts = [FadeIn(new_inset[-1])] + self.over_anims("tag", ov.act_tag(self.G, *b6.tag))
        for a in texts:
            a.rate_func = lambda x: smooth(min(1.0, x * swell / text_in))
        anims = [FadeOut(left, shift=LEFT * 1.2),
                 FadeOut(right, scale=ratio, target_position=new.get_center()),
                 FadeIn(new, scale=1 / ratio, target_position=right.get_center())]
        anims += self.machine_anims(dark)
        self.play(*anims, *texts, run_time=fr(swell))
        self.remove(*old_inset.submobjects, new_inset[-1])
        self.add(new_inset)
        self.over["inset"] = new_inset
        self.set_floors([new])

        # ---- "block": SM 0 gets its shelf and its fence; the block sweeps the square
        # ring by ring and waits at the fence after each one (amber: the fence, and the
        # ring's workers on the floor). The plan is in whole frames, so every wait is
        # drawn (and preview = final).
        t0, t1 = sweep_window(self.words)
        sweep = t1 - t0
        plan = sweep_plan(nrings, int(sweep * F15))
        amber = amber_fence(b6.machine, pl)

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
            show_amber(amber, key(p)[1])

        self.until(t0)
        self.add(amber)                       # after the floor: redrawn every frame of a play
        self.extras = [amber]
        self.floor_play(key, render, sweep)
        self.run(flash, sweep)
        # the fence fades in (no slide, so the amber copy fits it) by the first wait frame
        self.machine_to(b6.machine, frames=SLOW_STEP[0], slide=0)

        # ---- "twice": the speed-up, never within MIN_DWELL of the inset's new line being in
        self.say(f"{self.der['big_1block_x']:.2f}×", "vs the CPU, kernel time",
                 at=max(self.cue("twice"), t_text + MIN_DWELL))

        # ---- "one SM of twenty-four": the caption on "SM"; on "twenty-four" the other
        # 23 SMs come up idle in a wave, and go dark again as the word ends
        self.say_last(at=self.cue("SM"))
        self.until(self.cue("twenty-four"))
        self.ripple(mc.with_rooms(b6.machine, "idle", rooms=range(1, 24)), frames=10)
        self.until(self.cue("twenty-four", end=True))
        self.ripple(b6.machine, frames=8)
