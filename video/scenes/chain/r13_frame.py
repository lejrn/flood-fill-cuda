"""r13: the big frame. It answers r12's carried caption ("min / smallest
label wins"): the same tiles, pieces and waves run on the whole 81 Mpx
frame.

"On the big frame": the carried caption goes with the crop's hues, which
blend back to red as its border fades. Then the crop gives way to the
frame at the same window. The crop's pixels are the frame's (ch08 cut it
out of the frame), so only its faint tile grid goes, and the frame around
it comes in from the crop's edge outward: a blob the crop cut grows back
out of its cut edge. The floor pulls back to the whole red frame: one
zoom about a fixed point, geometric in the window's side.

The zoom is drawn from every pixel of the frame (frame_mask, from
assets/tiles/input_blobs_red.npz), never from the 900 px picture, whose
pixels would be squares 25 px wide at the start (a mosaic). Up close each
frame pixel is a sharp square, as the crop draws its cells; further out
each output pixel averages the frame pixels under it. The red drifts
from the crop's red to the frame floors' darker red on the way, and once
whole the frame blends into the 900 px floor (frame_red).

"over eleven thousand seeds": the seeds fade in as small white dots, tile
by tile over the whole frame (state.frame_seed_dots: one dot per seed
tile, 8,602 dots for the 11,609 seeds, no rings at this scale). Each
seed's own piece takes its colour with its dot (round 0). Caption
"11,609 seeds" / "6 rounds, 2,522 blobs", at "seeds".

"six rounds": six gate flashes, GATE_GAP apart (the cubes and the gate
only). The claims spread out from the seeds, a round per flash (WV's
claim rounds at 900 px, state.frame_round_map). Round 0's seed pieces
hold about a sixth of the 900 px blob pixels, round 1 brings it to about
three quarters and round 2 to 97%. Rounds 3 and 4 add the last 3%, and
the fifth and sixth flashes change nothing at 900 px. A claimed piece
takes its blob's final colour at once (frame_rounds_picture paints the
labelled frame), so no waves meet on screen: at 900 px a piece is a few
pixels, and r12 showed the meetings on the crop. The dots fade during the
last two flashes: every blob counted, the floor is the labelled frame.

"Two point two milliseconds": at "Two" the needle's old value goes. Once
the last flash is over, the needle slides from 53.9 ms past the 33 ms
line to 2.21 ms and turns teal. Its value comes in with the carried
caption ("2.21 ms" / "kernel time, no copies") at "milliseconds".

"inside the budget": out, then in. Row 3's old verdict and the act tag
fade out; then the tag comes back as "COUNT ✓" and row 3 reads
"GPU 100×" (the new strip fades in over the old one, whose other pills
are the same opaque pixels).

Starts on chain_state(17), ends on chain_state(18).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from manim import FadeIn, FadeOut, Transform, config
from PIL import Image

from scenes.chain import floor as fl
from scenes.chain import numbers as nb
from scenes.chain import overlays as ov
from scenes.chain import state as st
from scenes.chain.base import GATE_GAP, ChainScene, fr
from scenes.style import RED_PX

SEED_FADE = 1.0         # the dots fade in over at least this long (s)
SPREAD = 0.35           # a tile's share of the dots' fade (state.frame_seed_dots' default)
# phase lengths in seconds of whole 15 fps frames, so -ql and --fps 30 agree
HUE_S = 5 / 15          # the crop's hues back to red, its border out
SWAP_S = 3 / 15         # the crop's tile grid goes, the frame around it comes in from its edge
ZOOM_S = 0.8            # the pull back, at least
SMOOTH_S = 4 / 15       # the whole frame, drawn from every pixel, gives way to the 900 px floor
ROUND_BLEND = 0.2       # s: a round's claims colour in
DOTS_OUT = 0.6          # s: the seed dots fade once the last visible round has landed
FLASH_S = 0.08 + 0.12 + 0.22    # a gate flash's length (gate_rounds' default up, top and down)

MASK = nb.TILES / "input_blobs_red.npz"     # the frame's red mask, every pixel (assets/make_frame_mask.py)


def ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


@lru_cache(maxsize=1)
def frame_mask() -> tuple[np.ndarray, Image.Image]:
    """The whole frame's red mask, (9000, 9000) uint8 (255: red) in the
    kernels' [x, y] (state.picture's orientation), and the same pixels as
    a PIL image for area sampling."""
    z = np.load(MASK)
    m = np.unpackbits(z["red_bits"], axis=1, count=int(z["shape"][1])) * np.uint8(255)
    return m, Image.fromarray(m, "L")


def mask_window(rpx: int, r0: float, c0: float, side: float) -> np.ndarray:
    """(rpx, rpx) float32 in [0, 1]: how red each output pixel is, over the
    square window [r0, r0 + side) x [c0, c0 + side) of the frame (frame
    pixels). Up close (a frame pixel at least an output pixel) it samples
    nearest, every pixel a sharp square, as crop_raster draws the crop's
    cells; further out it averages the frame pixels under each output
    pixel (PIL's box filter)."""
    m, im = frame_mask()
    if side <= rpx:
        pos = (np.arange(rpx) + 0.5) / rpx * side
        rr = np.clip(np.floor(r0 + pos).astype(int), 0, m.shape[0] - 1)
        cc = np.clip(np.floor(c0 + pos).astype(int), 0, m.shape[1] - 1)
        return m[rr[:, None], cc].astype(np.float32) / 255
    out = im.resize((rpx, rpx), Image.BOX, box=(c0, r0, c0 + side, r0 + side))
    return np.asarray(out, np.float32) / 255


class BigFrame(ChainScene):
    k = 17

    def needle_slide(self, spec: ov.GoalSpec, slide_at: float, frames: int = 7):
        """The needle slides to `spec`'s value and takes its colour: the old
        value fades out now, the triangle slides along the track (and turns
        teal) at scene time `slide_at`. Returns (meter, value label): the
        label is not yet shown, the caller fades it in with the caption.
        The meter's box, ticks and labels are the same pixels before and
        after."""
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
        self.until(slide_at)
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
        r0, c0, s0 = st.crop_window()                           # in 900 px picture pixels
        side = st.SIDE_PX * st.TO_900                           # the 900 px picture's side
        to_px = st.SIDE_PX / side                               # picture pixel -> frame pixels
        empty = fl.rgb(fl.EMPTY).astype(np.float32)
        red_crop = fl.rgb(RED_PX).astype(np.float32)            # the crop's red (crop_colors)
        red_frame = fl.key_picture(np.array([[[255, 0, 0]]], np.uint8))[0, 0].astype(np.float32)

        def window(q: float) -> np.ndarray:
            """The zoom: the window's side grows geometrically from the crop's
            to the whole frame, about the one point that stays put. Drawn
            from every frame pixel; its red drifts from the crop's to the
            frame floors'."""
            s = s0 * (side / s0) ** q
            k = (side - s) / (side - s0)
            cov = mask_window(rpx, r0 * k * to_px, c0 * k * to_px, s * to_px)[..., None]
            red_q = red_crop + (red_frame - red_crop) * q
            out = np.empty((rpx, rpx, 4), np.uint8)
            out[..., :3] = (empty + (red_q - empty) * cov).astype(np.uint8)
            out[..., 3] = 255
            return out

        # the swap: inside the crop's box only the faint tile grid fades; the
        # frame around it comes in from the box's edge outward (a soft front),
        # so a blob the crop cut grows back out of its cut edge, never half
        # there along a straight seam
        w0 = window(0.0)
        cri, cci = st.crop_cells(rpx).maps()
        inside = (cri >= 0)[:, None] & (cci >= 0)[None, :]
        t_, l_, b_, r_ = (np.array(st.CROP_BOX) * rpx).tolist()
        pos = np.arange(rpx) + 0.5
        dy = np.maximum(np.maximum(t_ - pos, pos - b_), 0.0)
        dx = np.maximum(np.maximum(l_ - pos, pos - r_), 0.0)
        dist = np.hypot(dy[:, None], dx[None, :])               # output px outside the box
        feather = 0.03 * rpx
        reach = float(dist.max()) + feather

        def swap_raster(p: float) -> np.ndarray:
            a = np.where(inside, p, np.clip((p * reach - dist) / feather, 0.0, 1.0))[..., None]
            out = crop_red.astype(np.float32) * (1 - a) + w0.astype(np.float32) * a
            return out.astype(np.uint8)

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
                return swap_raster(ease(key[1] / (SWAP_S * fps)))
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

        # "On the big frame": the carried "min" goes with the crop's hues
        self.unsay(frames=round(HUE_S * 15))

        # "seeds": the first caption
        self.say(f"{int(n['full_seeds']):,} seeds", f"{int(n['frame_rounds'])} rounds, {int(n['n_blobs']):,} blobs",
                 at=self.cue("seeds"))

        # "Two point two milliseconds": at "Two" the old value goes; the
        # needle slides past the budget line (and turns teal) once the last
        # flash is over, and its value comes in with the carried caption
        self.until(max(self.cue("Two"), t_out1))
        meter, val = self.needle_slide(end.goal, slide_at=flashes[-1] + FLASH_S)
        self.say_last(at=self.cue("milliseconds"), extra=[FadeIn(val)])
        self.remove(val)
        meter.add(val)

        # "inside the budget": COUNT is won and row 3 jumps; out, then in.
        # Row 3's old verdict leaves from a stand-in copy of the strip (the
        # strip itself leaves whole, so no group splits; see needle_slide),
        # with the old tag; then the new strip fades in over the stand-in
        # (its pills are opaque, so the two that do not change never dip)
        # and the new tag comes in
        self.until(self.cue("budget"))
        old = self.over["strip"]
        stand = old.copy()
        verdict = stand[-1][1][1]                    # row 3's pill: (box, (name, verdict))
        stand[-1][1].remove(verdict)
        self.remove(old)
        self.add(stand, verdict)
        self.play(FadeOut(verdict), FadeOut(self.over["tag"]), run_time=fr(3))
        self.over["strip"] = self.strip(end.rows)
        self.over["tag"] = ov.act_tag(self.G, *end.tag)
        self.play(FadeIn(self.over["strip"]), FadeIn(self.over["tag"]), run_time=fr(5))
        self.remove(stand)
