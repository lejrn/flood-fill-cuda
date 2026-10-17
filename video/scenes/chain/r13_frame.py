"""r13: the big frame. It answers r12's carried caption ("min / smallest
label wins"): the same tiles, pieces and waves run on the whole 81 Mpx
frame.

"On the big frame": the crop's hues blend back to red and its border
fades; the crop gives way to the frame's own pixels at the same window,
and the floor pulls back to the whole red frame (one zoom about a fixed
point, geometric in the window's side).

"over eleven thousand seeds": the seeds fade in as small white dots, tile
by tile over the whole frame (state.frame_seed_dots, no rings at this
scale), and each seed's own piece takes its colour with its dot (round 0).

"six rounds": six gate flashes, GATE_GAP apart (the cubes and the gate
only); the claims spread out from the seeds, a round per flash (WV's
claim rounds at 900 px, state.frame_round_map). Only rounds 1-4 show at
900 px: the fifth claims 6 pieces and the sixth none, so the last two
flashes change nothing. The dots fade during them: every blob counted,
the floor is the labelled frame.

"Two point two milliseconds": the needle slides from 53.9 ms past the
33 ms line to 2.21 ms and turns teal; its value comes in with the carried
caption ("2.21 ms / kernel time, no copies"). "inside the budget": the act
tag turns "COUNT ✓" and row 3's pill turns from "GPU 3.8×" to "GPU 100×".

Starts on chain_state(17), ends on chain_state(18).
"""
from __future__ import annotations

import numpy as np
from manim import FadeIn, FadeOut, Transform, config

from scenes.chain import floor as fl
from scenes.chain import overlays as ov
from scenes.chain import state as st
from scenes.chain.base import GATE_GAP, ChainScene, fr

SEED_FADE = 1.0         # the dots fade in over at least this long (s)
SPREAD = 0.35           # a tile's share of the dots' fade (state.frame_seed_dots' default)
# phase lengths in seconds of whole 15 fps frames, so -ql and --fps 30 agree
HUE_S = 5 / 15          # the crop's hues back to red, its border out
SWAP_S = 3 / 15         # the crop (its own image) gives way to the frame's pixels at the same window
ZOOM_S = 0.8            # the pull back, at least
SMOOTH_S = 3 / 15       # the zoom's sharp pixels give way to the frame floor (bilinear)
ROUND_BLEND = 0.2       # s: a round's claims colour in
DOTS_OUT = 0.6          # s: the seed dots fade once the last visible round has landed


def ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


class BigFrame(ChainScene):
    k = 17

    def needle_slide(self, spec: ov.GoalSpec, frames: int = 7):
        """The needle slides to `spec`'s value and takes its colour: the old
        value fades out, the triangle slides along the track (and turns
        teal). Returns (meter, value label): the label is not yet shown, the
        caller fades it in with the caption. The meter's box, ticks and
        labels are the same pixels before and after."""
        old = self.over["goal"]
        new = self.goal(spec)
        tri, val = new[-2], new[-1]
        new.remove(tri, val)
        # the old meter leaves whole first: fading out one of its children
        # would split it into loose parts (Scene.remove), and the old box
        # would stay drawn under the new one (a brighter outline)
        old_val, moving = old[-1].copy(), old[-2].copy()
        self.remove(old)
        self.add(new, moving, old_val)
        self.play(FadeOut(old_val), run_time=fr(2))
        self.play(Transform(moving, tri), run_time=fr(frames))
        self.remove(moving)
        new.add(tri)
        self.over["goal"] = new
        return new, val

    def beat_anim(self) -> None:
        n = self.num
        end = st.boundary(self.k + 1)
        rpx = self.floor.rpx
        fps = config.frame_rate

        # ---- the floor's pictures, all at the floor's own size
        start = st.floor_raster("crop_labels", rpx)            # the state's floor
        crop_red = st.crop_raster(rpx, border=0.0)              # hues back to red, border gone
        keyed = fl.key_picture(st.picture("red"))
        r0, c0, s0 = st.crop_window()
        side = float(keyed.shape[0])

        def window(q: float) -> np.ndarray:
            """The zoom: the window's side grows geometrically from the crop's
            to the whole frame, about the one point that stays put."""
            s = s0 * (side / s0) ** q
            k = (side - s) / (side - s0)
            return fl.window_raster(keyed, rpx, r0 * k, c0 * k, s, keyed=True)

        red = fl.picture_raster(st.picture("red"), rpx)          # = floor_raster("frame_red")
        last = int(st.frame_round_map().max())                  # 4: the last round 900 px shows
        rounds = [fl.picture_raster(st.frame_rounds_picture(r), rpx) for r in range(last + 1)]
        labels = st.floor_raster("frame_labels", rpx)
        assert np.array_equal(rounds[last], labels), "the last round is not the labelled frame"

        # each output pixel's seed tile phase (the dots' fade-in order), so a
        # seed's own piece colours in with its dot; a tile without a seed
        # (an edge pixel filled from a neighbour) takes the middle
        tiles, phase = st.frame_seed_tiles()
        cells = st.frame_tile_cells(rpx)
        nt = cells.shape[0]
        ph = np.full((nt, nt), 0.5, np.float32)
        ph[tiles[:, 0], tiles[:, 1]] = phase
        ri, ci = cells.maps()
        ph_px = ph[np.clip(ri, 0, nt - 1)][:, np.clip(ci, 0, nt - 1)]
        sel0 = (red[..., :3] != rounds[0][..., :3]).any(axis=2)
        red_sel, r0_sel, ph_sel = red[sel0, :3].astype(np.float32), rounds[0][sel0, :3].astype(np.float32), ph_px[sel0]

        def seeds_raster(p: float) -> np.ndarray:
            out = red.copy()
            a = np.clip((p - ph_sel * (1 - SPREAD)) / SPREAD, 0, 1)[:, None]
            out[sel0, :3] = (red_sel + (r0_sel - red_sel) * a).astype(np.uint8)
            return out

        # ---- the timeline (scene seconds), from the word cues
        t_hue = 0.0
        t_swap = t_hue + HUE_S
        t_zoom = t_swap + SWAP_S
        t_six = self.cue("six")
        t_seed0 = max(self.cue("eleven"), t_zoom + ZOOM_S + SMOOTH_S)
        t_seed1 = max(t_six - 0.1, t_seed0 + SEED_FADE)
        t_smooth = t_seed0 - SMOOTH_S
        flashes = [max(t_six, t_seed1 + 0.1) + i * GATE_GAP for i in range(int(n["frame_rounds"]))]
        t_out0 = flashes[last] if len(flashes) > last else flashes[-1] + ROUND_BLEND
        t_out1 = t_out0 + DOTS_OUT
        assert t_smooth - t_zoom >= 0.75, "the zoom has no time"

        def key_at(t: float):
            """What the floor shows at scene time t, as a hashable key."""
            if t < t_swap:
                return ("hue", round((t - t_hue) * fps))
            if t < t_zoom:
                return ("swap", round((t - t_swap) * fps))
            if t < t_smooth:
                return ("zoom", round((t - t_zoom) * fps))
            if t < t_seed0:
                return ("smooth", round((t - t_smooth) * fps))
            if t < flashes[0]:
                return ("seeds", min(round((t - t_seed0) * fps), round((t_seed1 - t_seed0) * fps)))
            k = sum(1 for f in flashes if f <= t + 1e-9)
            blend = round(min((t - flashes[k - 1]) / ROUND_BLEND, 1.0) * ROUND_BLEND * fps) if k <= last else 0
            out = round(min(max((t - t_out0) / DOTS_OUT, 0.0), 1.0) * DOTS_OUT * fps)
            return ("rounds", min(k, last), blend, out)

        def render(key) -> np.ndarray:
            kind = key[0]
            if kind == "hue":
                return mix(start, crop_red, ease(key[1] / (HUE_S * fps)))
            if kind == "swap":
                return mix(crop_red, window(0.0), ease(key[1] / (SWAP_S * fps)))
            if kind == "zoom":
                return window(ease(key[1] / max((t_smooth - t_zoom) * fps, 1)))
            if kind == "smooth":
                return mix(window(1.0), red, ease(key[1] / (SMOOTH_S * fps)))
            if kind == "seeds":
                p = key[1] / max(round((t_seed1 - t_seed0) * fps), 1)
                return st.frame_seed_dots(seeds_raster(p), rpx, p, spread=SPREAD)
            _, k, blend, out = key
            steps = max(round(ROUND_BLEND * fps), 1)
            base = rounds[k] if k == 0 or blend >= steps else mix(rounds[k - 1], rounds[k], ease(blend / steps))
            u = out / max(round(DOTS_OUT * fps), 1)
            if u >= 1:
                return base
            return st.frame_seed_dots(base.copy(), rpx, 1.0, spread=SPREAD, alpha=0.9 * (1 - ease(u)))

        shown = {"key": None}
        total = t_out1 + 2 / fps

        def step(p: float) -> None:
            key = key_at(p * total)
            if key != shown["key"]:
                self.floor.show_raster(render(key))
                shown["key"] = key
            if p >= 1:
                self.floor.show_raster(labels)

        self.run(step, total)
        self.gate_rounds(flashes=flashes)

        # "six rounds": the first caption (the carried "min" goes)
        self.say(f"{int(n['full_seeds']):,} seeds", f"{int(n['frame_rounds'])} rounds, {int(n['n_blobs']):,} blobs",
                 at=self.cue("rounds"))

        # "Two point two milliseconds": the needle slides past the budget
        # line and turns teal; its value comes in with the carried caption
        self.until(max(self.cue("Two"), t_out1))
        meter, val = self.needle_slide(end.goal)
        self.say_last(at=self.cue("milliseconds"), extra=[FadeIn(val)])
        self.remove(val)
        meter.add(val)

        # "inside the budget": COUNT is won and row 3 jumps. The new strip
        # fades in over the old one: its pills are opaque, so the two that
        # do not change never dip
        self.until(self.cue("budget"))
        old = self.over["strip"]
        self.over["strip"] = self.strip(end.rows)
        anims = [FadeIn(self.over["strip"])] + self.over_anims("tag", ov.act_tag(self.G, *end.tag))
        self.play(*anims, run_time=fr(6))
        self.remove(old)
