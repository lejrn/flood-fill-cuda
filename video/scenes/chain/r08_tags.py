"""r08: tags. "So tag each queue entry with its blob": the entries land in
the VRAM queue, the floor becomes the two blobs (red), and the VRAM close-up
(D2, form "tags") grows out of the four chips over the floor: lanes image,
visited, queue. At "blob" every entry gets its tag, a thin stripe, on the
board and on the panel. At "one spare bit" the followed entry opens into
its 32 bit cells, the last one the tag (ch04's `lin` entry: the pixel's
address shifted left, the label in bit 0); at "global memory" the panel's
frame pulses.

Then one step of the kernel, slowly: a warp loads 32 neighbouring entries
in one packet (128 B, a dense read); the other 31 lanes fade to faint
lines toward their own pixels, spread along the image lane, and one lane is
followed: it paints its pixel palette[tag] (a 3 B store), claims a
neighbour (its visited cell flips 0 -> 1) and appends the neighbour's entry
with the same tag at the queue's rear. "picks its colour: its label": the
tag cell and the painted pixel pulse together (no label array: the paint is
the label map). "One bit tells two blobs apart": the tag cell flips blue,
green, blue. The panel folds back into the chips; both blobs fill at once
from their seeds, 4-way like ch04's kernel ("max, not sum", the launch
counter halves in the same play); the needle settles by the CPU tick. But
two seeds per launch still come from P-core 0 over the PCIe link.

Starts on chain_state(12), ends on chain_state(13).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from manim import (DOWN, Dot, FadeIn, FadeOut, Indicate, Line, Rectangle, ReplacementTransform,
                   UpdateFromAlphaFunc, VGroup, there_and_back)

from scenes.chain import machine as mc
from scenes.chain import toy
from scenes.chain.base import ChainScene, MIN_DWELL, fr
from scenes.chain.state import Z_MACHINE, boundary, filled_picture, label_colors, toys
from scenes.style import BLUE, GREEN, GREY, INK, RED_PX

Z_SEED = Z_MACHINE + 90         # over the dies, under the silkscreen and the launch pill (as r07, r09)
LAND = 7                        # frames from say(at=t) to its caption landing (2 out, 5 in)
DWELL = np.ceil(MIN_DWELL * 15) / 15    # MIN_DWELL in whole frames (1.8 s): until() rounds 1.75 s down a frame

# The close-up's stretch of memory (schematic), one pixel a character, in
# memory order: "." off the blobs, "r" red, "R" red and queued (claimed, so
# visited), "b" / "g" painted blue / green by earlier rings (visited). Two
# blobs in one image, one queue. A neighbour in y is the next pixel in memory.
PIX = "..rrRbbbbbbbbRrr....rrRggggggRrr....rRbbbbbbRrr........"
PIX_COLOR = {".": mc.PIXEL_BG, "r": RED_PX, "R": RED_PX, "b": BLUE, "g": GREEN}
WINDOW = (4, 36)                # the level's entries [front, rear) in the queue lane
FOLLOW = 6                      # the followed warp lane: entry WINDOW[0] + 6, tag blue (the board's cycle)
FOLLOW_PX = 13                  # its pixel: queued, beside the blue core; pixel 14 is red and unclaimed
WARP_LIT = "#2c5a8f"            # the warp's lanes once they hold their entries


def tags_panel(L, tags) -> mc.VramPanel:
    """D2, form "tags", at G.vram: the image lane is PIX, visited follows
    it (claimed or painted), the queue window carries the board's tags."""
    (cx, cy), (w, h) = L.G.vram
    pixels = [PIX_COLOR[c] for c in PIX]
    visited = [1 if c in "Rbg" else 0 for c in PIX]
    return mc.vram_closeup(L.machine_pl, (cx, cy), h, "tags", width=w, pixels=pixels, visited=visited,
                           window=WINDOW, tags=tags)


def outline(m, z: float, pad: float = 0.05):
    """A white outline round m, hidden (the pulse fades it in and out)."""
    o = Rectangle(width=m.width + 2 * pad, height=m.height + 2 * pad, stroke_color=INK, stroke_width=2.5)
    o.move_to(m).set_stroke(opacity=0)
    o.set_z_index(z)
    return o


def needle_move(scene, spec, frames: int = 6) -> None:
    """Move the needle to `spec`'s value: the old value label fades out, the
    triangle slides along the track, the new label fades in. The meter's box
    and labels are the same pixels before and after, so they never blink."""
    old = scene.over["goal"]
    new = scene.goal(spec)
    scene.play(FadeOut(old[-1]), run_time=fr(2))
    tri, val = new[-2], new[-1]
    new.remove(tri, val)
    dx = tri.get_center()[0] - old[-2].get_center()[0]
    tri.shift(-dx * np.array([1.0, 0, 0]))
    scene.remove(old)
    scene.add(new, tri)
    scene.play(tri.animate.shift(dx * np.array([1.0, 0, 0])), run_time=fr(frames))
    scene.play(FadeIn(val), run_time=fr(3))
    scene.remove(tri, val)
    new.add(tri, val)
    scene.over["goal"] = new


class Tags(ChainScene):
    k = 12

    def adopt(self, panel, *mobs) -> None:
        """Make mobs part of the panel (they fold back with it). Same frame:
        z_index, not the add order, decides what is drawn on top."""
        self.remove(*mobs)
        panel.add(*mobs)

    def beat_anim(self) -> None:
        t = toys()
        b1 = boundary(self.k + 1)
        start, end = boundary(self.k).machine, b1.machine
        pl = self.Lay.machine_pl
        two, lab = t["two"], t["two_lab"]
        ids = sorted(int(v) for v in np.unique(lab[lab >= 0]))
        seeds = [toy.centre_seed(lab == v) for v in ids]
        multi = toy.bfs(two, seeds, 4).level
        top = int(multi.max())
        full = label_colors(two, lab)
        red = label_colors(two, np.full(two.shape, -1))
        tags = end.tag_tokens
        tag, other = tags[FOLLOW % len(tags)], GREEN      # palette[0] and palette[1]

        # "So": the entries land in the VRAM queue; the floor becomes the two blobs, red
        self.until(self.cue("So"))
        self.floor_blend(self.pic(filled_picture()), self.toy(red), frames=8)
        self.machine_to(replace(start, tokens=end.tokens), frames=8)

        # "queue entry": the close-up grows out of the four VRAM chips, over the
        # floor, its three lane names the beat's first text; its tag stripes wait
        panel = tags_panel(self.Lay, tags)
        stripes = panel.stripes
        panel.body.remove(stripes)
        home = mc.vram_point(pl)
        self.until(max(self.cue("queue"), self.elapsed()))
        self.play(FadeIn(panel, target_position=home, scale=0.08), run_time=fr(8))
        t_lanes = self.elapsed()

        # "blob": every entry gets its blob's tag, a thin stripe, on the board
        # and on the panel at once
        self.until(max(self.cue("blob"), self.elapsed()))
        self.machine_to(replace(start, tokens=end.tokens, tag_tokens=tags), frames=6, slide=0.15,
                        extra=[FadeIn(stripes, shift=DOWN * 0.08)])
        self.adopt(panel, stripes)

        # "one spare bit": the followed entry opens into its 32 bit cells
        # (31 address bits, the tag last), under the queue lane
        q = panel.lanes["queue"]
        e = WINDOW[0] + FOLLOW
        a, b = q.span(e)
        bits = panel.entry_bits(e, tag)
        n_tag = round(32 * 0.3)                   # the stripe's share of the drawn entry
        sw = (b - a - 0.012) / 32
        slivers = VGroup(*[Rectangle(width=sw, height=q.h, fill_color=tag if j >= 32 - n_tag else RED_PX,
                                     fill_opacity=1, stroke_width=0).move_to([a + (j + 0.5) * sw, q.y, 0])
                           for j in range(32)])
        top_b = bits.get_top()[1]
        wedge = VGroup(Line([a, q.y - q.h / 2, 0], [bits.get_left()[0], top_b, 0]),
                       Line([b, q.y - q.h / 2, 0], [bits.get_right()[0], top_b, 0]))
        wedge.set_stroke(GREY, 1.2, opacity=0.7)
        opened = Rectangle(width=b - a, height=q.h + 0.04, stroke_color=INK, stroke_width=1.6).move_to(q.centre(e))
        for m in [*slivers, *wedge, opened]:
            m.set_z_index(panel.z + 2.4)
        self.until(max(self.cue("one"), self.elapsed()))
        self.play(ReplacementTransform(slivers, bits), FadeIn(wedge), FadeIn(opened), run_time=fr(8))
        self.adopt(panel, bits, wedge, opened)
        self.say("1 spare bit", "per queue entry", at=max(self.cue("bit"), t_lanes + DWELL - fr(LAND)))

        # "in global memory": the panel's frame pulses (white, the new-part convention)
        ring = panel.card[1].copy().set_fill(opacity=0).set_stroke(INK, 4.0, opacity=0)
        ring.set_z_index(panel.z + 4)
        self.add(ring)
        t_glob = max(self.cue("global"), self.elapsed())
        self.until(t_glob)
        pulse_s = min(max(self.cue("memory") + 0.45 - t_glob, fr(10)), fr(15))     # over "global memory"
        self.play(UpdateFromAlphaFunc(ring, lambda m, al: m.set_stroke(opacity=there_and_back(al))), run_time=pulse_s)
        self.remove(ring)

        # ---- one step, slowly. The load: the warp reads the level's 32
        # neighbouring entries in one packet (128 B, every byte used)
        warp = panel.warp
        pk = panel.packet("queue", WINDOW[0], 32, INK)
        self.until(max(self.cue("memory") + 0.3, self.elapsed()))
        self.play(FadeIn(pk), run_time=fr(2))
        self.play(pk.animate.move_to([pk.get_center()[0], panel.warp_y, 0]), run_time=fr(8))
        self.play(FadeOut(pk), *[c.animate.set_fill(WARP_LIT) for c in warp], run_time=fr(3))

        # each lane's entry points at its own pixel, somewhere in the image:
        # 31 faint lines spread along the image lane (schematic places), and
        # the followed lane's line to its pixel; the other lanes dim
        img, vis = panel.lanes["image"], panel.lanes["visited"]
        y_img = img.y + img.h / 2
        x_lo, x_hi = img.span(0)[0] + 0.05, img.span(len(img.cells) - 1)[1] - 0.05
        xs = np.random.default_rng(8).permutation(np.linspace(x_lo, x_hi, 31))
        others = [k for k in range(32) if k != FOLLOW]
        faint = VGroup(*[Line(warp[k].get_bottom(), [x, y_img, 0]) for k, x in zip(others, xs)])
        faint.set_stroke(GREY, 1.0, opacity=0.32)
        pix = img.cells[FOLLOW_PX]
        src = warp[FOLLOW].get_bottom()
        lead = Line(src, [pix.get_center()[0], y_img, 0]).set_stroke(INK, 2.0)
        for m in [*faint, lead]:
            m.set_z_index(panel.z + 2)
        self.play(FadeIn(faint), FadeIn(lead), warp[FOLLOW].animate.set_stroke(INK, 2.2),
                  *[warp[k].animate.set_fill(opacity=0.35).set_stroke(opacity=0.35) for k in others],
                  run_time=fr(5))
        self.adopt(panel, faint, lead)

        def store(nbytes: int, color: str, to, cue_t: float, travel: int, land: list) -> None:
            """A packet (as long as its bytes) leaves the followed lane for `to`
            on its word; as it arrives, `land` plays."""
            p = mc.packet(nbytes * panel.unit, color, unit=1.0, height=0.12, z=panel.z + 3).move_to(src)
            self.until(max(cue_t, self.elapsed()))
            self.play(FadeIn(p), run_time=fr(2))
            self.play(p.animate.move_to(to), run_time=fr(travel))
            self.play(FadeOut(p), *land, run_time=fr(2))

        # "Each new pixel": the lane paints its pixel palette[tag] (3 B, RGB)
        store(mc.ELEM_BYTES["image"], tag, pix.get_center(), self.cue("each", 2), 6, [pix.animate.set_fill(tag)])
        # "inherits": it claims a red neighbour, the next pixel in memory: its visited cell flips 0 -> 1
        claim = vis.cells[FOLLOW_PX + 1]
        store(mc.ELEM_BYTES["visited"], INK, claim.get_center(), self.cue("inherits"), 6,
              [claim.animate.set_fill(mc.VISITED_ON)])
        # "the tag": the neighbour's entry, with the same tag bit, lands at the queue's rear
        slot = q.cells[WINDOW[1]]
        cell = Rectangle(width=slot.width, height=slot.height, fill_color=RED_PX, fill_opacity=1, stroke_width=0)
        mark = stripes[0].copy().set_fill(tag)
        entry = VGroup(cell, mark)
        mark.move_to([cell.get_right()[0] - mark.width / 2, cell.get_y(), 0])
        entry.move_to(src)
        cell.set_z_index(panel.z + 2)
        mark.set_z_index(panel.z + 2.2)
        self.until(max(self.cue("tag", after=self.cue("inherits")), self.elapsed()))
        self.play(FadeIn(entry), run_time=fr(2))
        self.play(entry.animate.move_to(slot.get_center() + (entry.get_center() - cell.get_center())), run_time=fr(7))
        self.adopt(panel, entry)

        # "picks its colour" and "its label": the tag cell and the painted pixel
        # pulse together. No label array: the paint is the label map.
        bit = bits[-1]

        def pulse(at: float) -> None:
            rings = [outline(bit, panel.z + 4), outline(pix, panel.z + 4)]
            self.add(*rings)
            self.until(max(at, self.elapsed()))
            self.play(Indicate(bit, color=bit.get_fill_color(), scale_factor=1.6),
                      Indicate(pix, color=pix.get_fill_color(), scale_factor=1.8),
                      *[UpdateFromAlphaFunc(r, lambda m, al: m.set_stroke(opacity=there_and_back(al))) for r in rings],
                      run_time=fr(10))
            self.remove(*rings)

        pulse(self.cue("picks"))
        pulse(self.cue("label"))

        # "One bit tells two blobs apart": the tag cell flips between its two
        # values, blue and green, and ends blue; then the panel folds back
        t_two = self.cue("two", after=self.cue("apart"))
        t_fold = max(t_two - fr(8), self.elapsed() + 1.0)
        t_one = max(self.cue("one", 2), self.elapsed())

        def flip(p: float) -> None:
            v = 0.5 - 0.5 * np.cos(2 * np.pi * 2 * p)             # 0 -> 1 -> 0, twice
            s = float(np.clip((v - 0.5) * 3 + 0.5, 0.0, 1.0))      # a bit: short changes, long holds
            bit.set_fill(mc.mix_hex(tag, other, s))

        self.until(t_one)
        self.run(flip, max(t_fold - t_one - fr(1), 0.8))
        self.until(t_fold)
        self.play(FadeOut(panel, target_position=home, scale=0.08), run_time=fr(8))

        # "two per launch": both blobs fill at once from their seeds, each in
        # its own colour, in the rings of the bigger one
        def render(i: int):
            col = red.copy()
            done = (multi >= 0) & (multi <= i)
            col[done] = full[done]
            return self.toy(col, dots=((multi == i) & (i < top)).astype(int))

        t_two = max(t_two, self.elapsed())
        self.until(t_two)
        self.floor_play(lambda p: round(p * top), render, max(self.cue("speed", end=True) - t_two, 1.0))
        # "launch": the caption, and the launch counter halves in the same play (one text change)
        self.say("max, not sum", at=max(self.cue("launch"), self._cap_t + DWELL - fr(LAND)),
                 extra=self.machine_anims(end, slide=0))

        # "about CPU speed": the needle settles by the CPU tick (its new label
        # never lands within MIN_DWELL of the caption)
        self.until(max(self.cue("CPU"), self._cap_t + DWELL - fr(11)))
        needle_move(self, b1.goal)
        t_needle = self.elapsed()

        # "But only two seeds, each handed in": P-core 0 lights (our host code
        # runs) and the seeds run from it over the PCIe link, a pair per
        # launch: two dots side by side, across the path
        core = mc.build(replace(end, cpu="lit"), pl, z0=Z_MACHINE).parts["pcore"]
        self.extras.append(core)
        path = mc.link_path(pl)
        launches, travel, apart = 3, 0.75, 0.065
        dots = []
        for _ in range(2 * launches):
            d = Dot(path[0], radius=0.05, color=INK)
            d.set_z_index(Z_SEED)
            d.set_opacity(0)
            dots.append(d)
        self.add(*dots)
        self.extras += dots
        self.until(max(self.cue("But"), self.elapsed()))
        self.play(FadeIn(core), run_time=fr(3))
        t0 = max(self.cue("two", after=self.cue("But")), self.elapsed())
        t1 = max(self.cue("in", end=True, after=self.cue("handed")), t0 + travel + 0.6)
        gap = (t1 - t0 - travel) / (launches - 1)

        def stream(p: float) -> None:
            tt = p * (t1 - t0)
            for j, d in enumerate(dots):
                s = (tt - (j // 2) * gap) / travel
                at = mc.point_along(path, s)
                ahead = mc.point_along(path, min(s + 0.02, 1.0)) - mc.point_along(path, max(s - 0.02, 0.0))
                side = np.array([-ahead[1], ahead[0], 0.0]) / max(np.linalg.norm(ahead), 1e-9)
                d.move_to(at + side * apart * (1 if j % 2 else -1))
                d.set_opacity(1.0 if 0 <= s < 1 else 0.0)

        self.run(stream, t1 - t0, start=t0)
        self.say_last(at=max(self.cue("seeds"), t_needle + DWELL - fr(LAND)))
        self.until(t1)
