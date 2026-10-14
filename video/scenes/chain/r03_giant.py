"""r03_giant: one giant blob, and its queue overflows shared memory (v4,
points 1 and 2). The drone swells into the 64 Mpx test square and the
CPU core goes dark. At "overflows" the shelf gauge (D1) grows out of SM 0:
its box is a block's default shared memory, the queue region two thirds
of it, the bar under it the queue's real load (two adjacent rings of the
64 Mpx square; toy ring i is real ring 400 i, and the bar moves through
the real rings in between, so it never jumps). The block sweeps the
square ring by ring and waits at SM 0's fence (__syncthreads) after each
ring: the fence flashes amber, the ring's workers wait amber on the floor.

While the bar is past the region (real rings 1,024 to 6,975, toy rings 3
to 17) red tokens pour out of the region's end, down the leader into
SM 0 and through the memory controller into VRAM chip 0's slots; on the
rings after, some come back up to SM 0 (the spill slice is read after the
ring's shared slots). Text: "48 KB" at "forty-eight" (the box pulses),
"125 KB" at "hundred", the carried "spill" at "spills". At "eight
gigabytes" the four VRAM chips pulse. The sweep ends with "gigabytes",
and the gauge folds back into SM 0.

Starts on chain_state(5) (the split drones, P-core 0 and SM 0 lit, its
shelf full) and ends on chain_state(6) (the square painted, SM 0 with its
shelf and its fence at rest, the spill in VRAM chip 0).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import LEFT, FadeIn, FadeOut, Polygon, Rectangle, config, smooth

from narration.common import read_words
from scenes.chain import machine as mc
from scenes.chain import overlays as ov
from scenes.chain.base import MIN_DWELL, ChainScene, _norm, fr
from scenes.chain.iso import plane_corners
from scenes.chain.state import boundary, floor_raster, level_colors, make_floors, toys
from scenes.style import INK, OUT, RED_PX, voice

# The sweep runs on whole frames (at 15 fps; every step lasts twice as many
# frames at 30 fps), so every fence flash is drawn and preview = final.
SLOW = 3                # sweep_plan: the first rings run slowly, so the fence flash can be read
SLOW_STEP = (4, 2)      # ... paint frames, then wait frames at the fence
FALLBACK_STEP = (3, 1)  # when even the slow rings do not fit
F15 = 15                # the plan's frame rate
WAIT = 2                # paced_plan: wait frames at the fence after each ring

GROW = 6                # frames: the gauge grows out of SM 0 (and folds back into it)
TOKEN = (0.11, 0.08)    # a travelling queue entry (frame units)
TOKEN_SPEED = 6.0       # frame units per second along its way
STORES_MAX = 4          # tokens poured on the ring with the largest overflow (at least one per ring)
PULSE = 0.9             # seconds of a white outline pulse (the new-part convention)


def sweep_window(words: list, first: str = "overflows", last: str = "gigabytes",
                 lead: float = 0.0) -> tuple[float, float]:
    """A sweep's window: from `lead` s after `first` starts to the end of
    `last`. Words are matched like ChainScene.cue (punctuation ignored); a
    voice without them gets a 3 s sweep."""
    def at(w: str, i: int) -> float | None:
        hits = [x for x in words if _norm(x[0]) == _norm(w)]
        return float(hits[0][i]) if hits else None
    t0, t1 = at(first, 1), at(last, 2)
    t0 = None if t0 is None else t0 + lead
    if t0 is None or t1 is None or t1 <= t0 + 0.1:
        t0 = t0 if t0 is not None else 0.0
        return t0, t0 + 3.0
    return t0, t1


def r03_window(words: list) -> tuple[float, float]:
    """r03's sweep: from the gauge being in (GROW frames after "overflows")
    to the end of "gigabytes"."""
    return sweep_window(words, "overflows", "gigabytes", lead=GROW / F15)


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
    """(ring, waiting at the fence) for each 15 fps frame of a sweep that
    must fit in `frames` (it may end sooner).

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


def paced_plan(n: int, frames: int, wait: int = WAIT) -> list[tuple[int, bool]]:
    """(ring, waiting at the fence) for each 15 fps frame of a sweep that
    fills `frames`: every ring gets an even share (within one frame), its
    last `wait` frames at the fence, so the rings keep one pace and a ring
    lands where the words put it. When a ring would get under wait + 2
    frames, sweep_plan's shorter sweep instead."""
    if frames < n * (wait + 2):
        return sweep_plan(n, frames)
    plan: list[tuple[int, bool]] = []
    for i in range(n):
        m = (i + 1) * frames // n - i * frames // n
        plan += [(i, False)] * (m - wait) + [(i, True)] * wait
    return plan


def plan_index(p: float, seconds: float) -> int:
    """The 15 fps plan index shown at progress p of a run of `seconds`."""
    fps = config.frame_rate
    n = round(p * seconds * fps)
    return int(n * F15 // round(fps))


def r03_plan() -> list[tuple[int, bool]]:
    """r03's sweep plan, timed from its words "overflows" to "gigabytes"."""
    words = read_words(OUT / voice()).get("r03_giant", [])
    t0, t1 = r03_window(words) if words else (0.0, 3.0)
    lv = toys()["square4"].level
    return paced_plan(int(lv.max()) + 1, int((t1 - t0) * F15))


def queue_load(levels) -> np.ndarray:
    """The queue's load on each real ring: the ring being read plus the ring
    being written (the last ring alone), in 4-byte slots."""
    s = np.asarray(levels, float)
    return s + np.append(s[1:], 0.0)


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


def chip_outlines(pl, z: float = 300.0) -> list:
    """The four VRAM chips' top faces, outlined in white (hidden until pulsed)."""
    out = []
    for r in mc.vram_rects():
        ul, ur, dl, dr = plane_corners(pl, r.x, r.y, r.x1, r.y1, 0.08)
        p = Polygon(ul, ur, dr, dl, stroke_color=INK, stroke_width=3.5, stroke_opacity=0, fill_opacity=0)
        p.set_z_index(z)
        out.append(p)
    return out


class Giant(ChainScene):
    k = 5

    def pulse(self, mobs: list, seconds: float = PULSE) -> None:
        """A white outline pulse, from now: the outlines' strokes rise and
        fall once (on the floor's clock, so it runs through any play)."""
        self.add(*mobs)

        def fn(p: float) -> None:
            for m in mobs:
                m.set_stroke(opacity=float(np.sin(np.pi * p)))

        self.run(fn, seconds)

    def beat_anim(self) -> None:
        t = toys()
        sq, lv = t["square"], t["square4"].level
        top = int(lv.max())
        nrings = top + 1
        b6 = boundary(6)
        pl = self.Lay.machine_pl
        n, d = self.num, self.der
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

        # ---- "overflows": the shelf gauge grows out of SM 0 to its place, on its leader.
        # Box: a block's shared memory (48 KB); region: the queue's 8,192 slots; bar: the
        # queue's real load, on the same byte scale up to the two-ring peak
        home = mc.sm_point(pl, 0)
        (gc, gs) = self.G.gauge
        gauge = mc.shelf_gauge(pl, gc, gs, slots=int(n["ring_slots"]), shared_bytes=round(n["shared_kb"] * 1024),
                               peak_slots=int(d["two_ring_peak"]))
        self.until(self.cue("overflows"))
        self.play(FadeIn(gauge, target_position=home, scale=0.08), run_time=fr(GROW))

        # ---- the sweep, from the gauge being in to the end of "gigabytes": one block paints
        # the square ring by ring and waits at SM 0's fence after each one (amber: the fence,
        # and the ring's workers on the floor). The plan is in whole frames, so every wait is
        # drawn (and preview = final).
        t0, t1 = r03_window(self.words)
        self.until(t0)
        t0 = self.elapsed()
        sweep = t1 - t0
        plan = paced_plan(nrings, int(sweep * F15))
        amber = amber_fence(b6.machine, pl)
        first = {}                                    # ring -> (its first plan frame, its frames, paint frames)
        for j, (i, waiting) in enumerate(plan):
            a, m, paint = first.get(i, (j, 0, 0))
            first[i] = (a, m + 1, paint + (0 if waiting else 1))

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

        # the bar: toy ring i is real ring 400 i at the middle of its frames, so the real
        # ring moves on every frame (and the ring the bar crosses the region on is toy ring 3)
        load = queue_load(n["big_level_sizes"])
        scale = (len(load) - 1) / top
        ring_slots = int(n["ring_slots"])

        def real_ring(p: float) -> float:
            q = p * sweep * F15
            j = min(int(q), len(plan) - 1)
            i = plan[j][0]
            a, m, _ = first[i]
            pos = min(i + (q - a) / m, nrings)
            return float(np.clip(scale * (pos - 0.5), 0, len(load) - 1))

        def bar(p: float) -> None:
            gauge.set_load(float(np.interp(real_ring(p), np.arange(len(load)), load)))

        # the overflow, per toy ring whose real ring overflows the region: stores pour out
        # of the region's end over the ring's paint frames, more as the overflow grows; on
        # the rings after, some come back up to SM 0 (after the ring's shared slots)
        over = {i: load[round(scale * i)] - ring_slots for i in range(nrings)}
        band = [i for i in range(nrings) if over[i] > 0]
        peak = max(over[i] for i in band)
        stores = {i: max(1, round(STORES_MAX * over[i] / peak)) for i in band}
        events = []                                   # (start s, path, seconds, slot, store?)

        def at_frame(f: float) -> float:
            return t0 + f / F15

        def way(pts) -> tuple:
            length = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))
            return pts, min(max(length / TOKEN_SPEED, 0.4), 1.0)

        k = 0
        for i in band:
            a, _, paint = first[i]
            for s in range(stores[i]):
                pts, sec = way(gauge.to_vram(pl, k % mc.SPILL_SLOTS))
                events.append((at_frame(a + paint * (s + 0.5) / stores[i]), pts, sec, k % mc.SPILL_SLOTS, True))
                k += 1
        landed = min(e[0] + e[2] for e in events)
        k = 0
        # each ring reads the ring before's spill slice, so the loads run one ring past the band
        for i in range(band[0] + 1, min(band[-1] + 2, nrings)):
            a, _, paint = first[i]
            for s in range(max(1, stores[i - 1] // 2)):
                ts = at_frame(a + paint / 2 + 1.5 * s)
                if ts < landed + fr(2):               # nothing to load before the first entry lands
                    continue
                pts, sec = way(list(reversed(mc.vram_path(pl, 0, k % mc.SPILL_SLOTS))))
                events.append((ts, pts, sec, k % mc.SPILL_SLOTS, False))
                k += 1
        tokens = []
        for _ in events:
            tok = Rectangle(width=TOKEN[0], height=TOKEN[1], fill_color=RED_PX, fill_opacity=0, stroke_width=0)
            tok.set_z_index(mc.Z_PACKET)
            tokens.append(tok)

        # VRAM chip 0's three spill slots, each shown as its first token lands: the end
        # state's own part (the same faces), so the cut is seamless
        spill = mc.build(b6.machine, pl, z0=100.0, only={"spill"})
        faces = spill.part("spill").submobjects
        per = len(faces) // mc.SPILL_SLOTS             # a slot and its token
        slot_faces = [faces[j * per:(j + 1) * per] for j in range(mc.SPILL_SLOTS)]
        lands = {}
        for ts, _, sec, slot, store in events:
            if store:
                lands[slot] = min(lands.get(slot, np.inf), ts + sec)
        for f in faces:
            f.set_fill(opacity=0)
        t_end = max(e[0] + e[2] for e in events)

        def pour(p: float) -> None:
            now = t0 + p * (t_end - t0)
            for tok, (ts, pts, sec, _, _) in zip(tokens, events):
                s = (now - ts) / sec
                if 0 <= s < 1:                        # gone on arrival (none left on SM 0 at the end)
                    tok.move_to(mc.point_along(pts, s))
                    tok.set_fill(opacity=1)
                else:
                    tok.set_fill(opacity=0)
            for j, fs in enumerate(slot_faces):
                a = min(max((now - lands[j] + fr(2)) / fr(3), 0.0), 1.0)
                for f in fs:
                    f.set_fill(opacity=a)

        # added after the floor: redrawn every frame of a play
        self.add(amber, spill.part("spill"), *tokens)
        self.extras = [amber, *tokens]
        self.floor_play(key, render, sweep)
        self.run(flash, sweep)
        self.run(bar, sweep)
        self.run(pour, t_end - t0)
        # the fence fades in (no slide, so the amber copy fits it) by the first wait frame
        self.machine_to(replace(b6.machine, spill=False), frames=SLOW_STEP[0], slide=0)
        self.machine.parts["spill"] = spill.part("spill")
        self.machine.keys["spill"] = spill.keys["spill"]
        self.machine.spec = b6.machine

        # ---- "forty-eight kilobytes": the box (a block's shared memory) pulses, caption 1
        box = gauge.box
        ring = Rectangle(width=box.width + 0.08, height=box.height + 0.08, stroke_color=INK, stroke_width=4,
                         stroke_opacity=0).move_to(box)
        ring.set_z_index(mc.Z_CALLOUT + 2)
        t48 = max(self.cue("forty-eight"), t_text + MIN_DWELL)
        self.until(t48)
        self.pulse([ring])
        self.say(f"{round(n['shared_kb'])} KB", "per block, by default", at=t48)

        # ---- "over a hundred": caption 2, the bar near its peak, far past the box
        self.say(f"{round(d['two_ring_kb'])} KB", "two rings at peak", at=self.cue("hundred"))

        # ---- "spills": the tokens are landing in VRAM chip 0's slots; the carried caption
        self.say_last(at=self.cue("spills"))

        # ---- "eight gigabytes": the four VRAM chips pulse
        self.until(self.cue("eight"))
        chips = chip_outlines(pl)
        self.pulse(chips)
        self.extras += [ring, *chips]

        # ---- the sweep ends with "gigabytes": the gauge folds back into SM 0
        self.until(t1)
        self.play(FadeOut(gauge, target_position=home, scale=0.08), run_time=fr(GROW))
