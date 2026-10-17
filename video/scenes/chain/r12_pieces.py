"""r12: every tile finds its blob pieces, seeds send waves piece to piece,
and each blob keeps its smallest label (chain_v5_proposal.md, r12_pieces).

The picture is WV's crop: 384 x 384 px of the frame at x 4352, y 5632,
run by the kernels as an image of its own. It shares the frame's tile grid
and seed lattice (its 36 seeds are the frame's, tile for tile), but 6 of
its 8 blobs are cut by its edge, 7 of its tiles end on another round than
in the frame, and its labels are its own. So it is shown as its own small
image inside its own border (state.CROP_BOX), and nothing of it carries
back into the frame: r13 pulls back to the red frame.

- "So first, every tile": the carried caption goes, the one-blob frame
  turns red and the floor zooms into the crop's window (sharp pixels,
  about the one point both windows share). As it lands, the crop's own
  pixels come in inside a thin border and the rest of the frame fades
  out (the keyed picture's darker red steps to the arrays' red inside
  that blend).
- "pieces": the 24 SMs' outlines pulse once (finding the pieces is a
  launch of its own, one warp per tile, so the cubes, the waves' blocks,
  stay still), and all at once the cut lines between pieces and a dot on
  each piece's first pixel (its anchor) come in: 135 pieces, at most 4 in
  a tile. Caption "pieces" / "blobs cut by tiles".
- "Some pieces are seeds": 25 seeds spark (one white ring each, once) and
  take their wave's hue; then the lattice tiles get a faint hatching and
  the 11 seeds that are seeds only because their tile is on it spark.
  Caption "36 seeds" / "8 blobs", at "seeds".
- "each seed sends a wave": the queue's tokens become piece tokens. Six
  rounds, one gate flash each, GAP apart: a piece takes its wave's hue in
  its round (50, 38, 7, 3, 1, then none), just before its round's gate,
  and the round's meetings appear as faint white ticks across the tile
  borders (116 in all).
- "Where two waves meet": the ticks brighten together, once.
- "and at the end": the dots, hatching and ticks fade. At "smallest" every
  piece of a blob turns to its blob's hue at once (the hue of its
  smallest-label wave) and the cut lines go. Carried caption "min" /
  "smallest label wins", at "label".

Nothing flashes faster than twice a second: the gate keeps GATE_GAP, a
seed's ring shows once, the ticks brighten once.

Starts on chain_state(16) (the frame, its largest blob blue, "1 blob"),
ends on chain_state(17) (crop_labels, piece tokens, "min").
"""
from __future__ import annotations

import numpy as np
from manim import Polygon, config

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import state as st
from scenes.chain.base import ChainScene, fr
from scenes.chain.iso import plane_corners
from scenes.style import INK

PIECES = 0.85           # the cut lines between pieces, at full strength
ANCHORS = 0.95          # the dots on the pieces' first pixels
HATCH = 0.35            # the lattice tiles' hatching
TICK_LO, TICK_HI, TICK_NOTED = 0.45, 1.0, 0.7     # a meeting tick: as noted, brightened once, after
GAP = 0.55              # seconds between the rounds' gate flashes (base.GATE_GAP is the floor)
CLAIM_LEAD = 0.1        # a round's claims land this long before its gate flash
SPARK_LIFE = 0.4        # seconds of a seed's ring
PULSE = 1.0             # seconds of the SMs' outline pulse


def sm_outlines(pl) -> list:
    """A white outline round each SM on the die's top face, hidden until pulsed."""
    out = []
    for i in range(24):
        r = mc.sm_rect(i)
        ul, ur, dl, dr = plane_corners(pl, r.x - 0.03, r.y - 0.03, r.x1 + 0.03, r.y1 + 0.03, mc.DIE_Z)
        p = Polygon(ul, ur, dr, dl, stroke_color=INK, stroke_width=2.5, stroke_opacity=0, fill_opacity=0)
        p.set_z_index(mc.Z_PACKET - 1)
        out.append(p)
    return out


def rings(raster: np.ndarray, cells: fl.Cells, pts, radius, width: float, color: str, alpha) -> np.ndarray:
    """A ring (`width` px wide) of `radius` output pixels round each cell
    coordinate in `pts` ((n, 2) of (r, c)); radius and alpha: a number or
    one per point. In place; returns the raster."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return raster
    rad = np.broadcast_to(np.asarray(radius, float), (len(pts),))
    al = np.broadcast_to(np.asarray(alpha, np.float32), (len(pts),))
    y, x = cells.px(pts[:, 0], pts[:, 1])
    A = np.zeros((cells.rpx, cells.rpx), np.float32)
    for yc, xc, R, a in zip(y, x, rad, al):
        k = int(np.ceil(R + width))
        y0, y1 = max(int(yc) - k, 0), min(int(yc) + k + 1, cells.rpx)
        x0, x1 = max(int(xc) - k, 0), min(int(xc) + k + 1, cells.rpx)
        if y0 >= y1 or x0 >= x1 or a <= 0:
            continue
        py, px_ = np.mgrid[y0:y1, x0:x1] + 0.5
        d = np.sqrt((py - yc) ** 2 + (px_ - xc) ** 2)
        sub = A[y0:y1, x0:x1]
        np.maximum(sub, np.where(np.abs(d - R) <= width / 2, a, 0.0), out=sub)
    sel = A > 0
    fl._blend(raster, sel, color, A[sel])
    return raster


def piece_map(cells: fl.Cells) -> np.ndarray:
    """(rpx, rpx): the crop's piece under each output pixel (-1: none)."""
    comp = st.a3()["crop_comp"]
    ri, ci = cells.maps()
    rs, cs = ri >= 0, ci >= 0
    C = np.full((cells.rpx, cells.rpx), -1, np.int32)
    C[np.ix_(rs, cs)] = comp[ri[rs]][:, ci[cs]]
    return C


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """a towards b by t (exactly b at t = 1)."""
    if t >= 1:
        return b
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


def ramp(t: float, t0: float, t1: float) -> float:
    """0 before t0, 1 after t1, quantized to the frame rate (so a raster is
    only made again when the value changes)."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    n = max(1, round((t1 - t0) * config.frame_rate))
    return round((t - t0) / (t1 - t0) * n) / n


def ease(p: float) -> float:
    return p * p * (3 - 2 * p)


class Pieces(ChainScene):
    k = 16

    def beat_anim(self) -> None:
        a, n = st.a3(), self.num
        f = self.floor
        rpx = f.rpx
        end = st.boundary(self.k + 1)
        cells = st.crop_cells(rpx)
        cmap = piece_map(cells)
        red = st.picture("red")
        H = red.shape[0]

        seed = a["crop_comp_seed"].astype(bool)
        lat_only = a["crop_comp_seed_lattice_only"].astype(bool)
        rnd = a["crop_comp_round"].astype(int)
        rounds = int(n["crop_rounds"])                       # 6: the last claims nothing
        assert int(seed.sum()) == int(n["crop_seeds"]) and int(lat_only.sum()) == int(n["crop_seeds_lattice_only"])
        assert (rnd[seed] == 0).all() and int(rnd.max()) == rounds - 1, (rounds, int(rnd.max()))
        tiles = a["crop_comp_tile"].astype(int)
        ny = a["crop_tile_lattice"].shape[1]
        assert all(a["crop_tile_lattice"][divmod(int(tiles[k]), ny)] for k in np.nonzero(lat_only)[0])

        # ---- the cues
        t_pieces = self.cue("pieces")
        t_some, t_seeds, t_seeds_end = self.cue("Some"), self.cue("seeds"), self.cue("seeds", end=True)
        t_each, t_wave = self.cue("each"), self.cue("wave")
        t_meet, t_end, t_small, t_label = self.cue("meet"), self.cue("end"), self.cue("smallest"), self.cue("label")

        # ---- the timeline (scene seconds)
        z0 = fr(4)                                          # the one blob turns red, its pixels sharp
        z1 = z0 + 1.15                                      # the zoom into the crop's window
        l1 = z1 + 0.4                                       # the crop comes in, the frame goes
        p0 = max(l1, t_pieces)                              # cut lines and anchors, all at once
        p1 = p0 + 0.5
        # the seeds: first the 25 that are seeds by themselves, then the 11 of the lattice
        rng = np.random.default_rng(12)
        own = [int(k) for k in rng.permutation(np.nonzero(seed & ~lat_only)[0])]
        lat = [int(k) for k in rng.permutation(np.nonzero(lat_only)[0])]
        s1a, s1b = t_some, t_seeds - 0.15                   # the first group's rings start in here
        h0 = t_seeds                                        # the hatching
        h1 = h0 + 0.4
        s2a, s2b = h0 + 0.25, t_seeds_end - SPARK_LIFE      # the lattice group's rings start in here
        spark_at = {k: s1a + (s1b - s1a) * j / max(len(own) - 1, 1) for j, k in enumerate(own)}
        spark_at |= {k: s2a + (s2b - s2a) * j / max(len(lat) - 1, 1) for j, k in enumerate(lat)}
        sp0, sp1 = s1a, max(spark_at.values()) + SPARK_LIFE
        # the rounds: claims, then the round's gate
        claim_t = [t_wave + GAP * i for i in range(rounds)]   # round i + 1
        flashes = [t + CLAIM_LEAD for t in claim_t]
        b0 = t_meet                                         # the ticks brighten once
        b1, b2, b3 = b0 + 0.25, b0 + 0.55, b0 + 0.95
        e0 = t_end - 0.15                                   # the marks fade
        e1 = e0 + 0.5
        m0 = t_small                                        # every blob in one hue, at once: a short
        m1 = m0 + fr(3)                                     # cross-fade, so mixed hues barely show
        assert p1 <= sp0 and s2b > s2a and e0 > b3 and m0 > e1, (p1, sp0, s2a, s2b, b3, e0, e1, m0)
        assert flashes[-1] + 0.45 < t_meet, flashes

        # ---- the rasters
        start = st.floor_raster("frame_one_blob", rpx)
        r0, c0, side0 = 0.0, 0.0, float(H)
        r1, c1, side1 = st.crop_window()
        # the zoom keeps one point fixed: the point both windows show at the same place
        kr, kc = (r1 - r0) / (side0 - side1), (c1 - c0) / (side0 - side1)
        fix_r, fix_c = r0 + kr * side0, c0 + kc * side0

        def window(q: float) -> np.ndarray:
            side = side0 * (side1 / side0) ** q
            return fl.window_raster(red, rpx, fix_r - kr * side, fix_c - kc * side, side)

        sharp, landed = window(0.0), window(1.0)
        crop_red = st.crop_raster(rpx)
        cache: dict = {}

        def crop(**kw) -> np.ndarray:
            """st.crop_raster, the last few kept (a spark frame needs two)."""
            key = tuple(sorted(kw.items()))
            if key not in cache:
                if len(cache) >= 6:
                    cache.pop(next(iter(cache)))
                cache[key] = st.crop_raster(rpx, **kw)
            return cache[key]

        anchors_r = max(3.0, rpx / 300)                     # as crop_raster draws the anchors
        ring_w = max(2.0, rpx / 320)

        def sparks(t: float, hatch: float) -> np.ndarray:
            """The seeds sparked by t in their wave's hue (round 0), and a ring
            round each seed whose spark is on (once: it grows and fades)."""
            on = [k for k, t0 in spark_at.items() if t >= t0 + 0.35 * SPARK_LIFE]
            A = crop(upto=None, pieces=PIECES, anchors=ANCHORS, hatch=hatch)
            B = crop(upto=0, pieces=PIECES, anchors=ANCHORS, hatch=hatch)
            out = A.copy()
            if on:
                sel = np.isin(cmap, on)
                out[sel] = B[sel]
            live = [(k, (t - t0) / SPARK_LIFE) for k, t0 in spark_at.items() if 0 <= t - t0 < SPARK_LIFE]
            if live:
                ks = [k for k, _ in live]
                qs = np.array([q for _, q in live])
                rings(out, cells, st.piece_anchors()[ks], anchors_r + 2 + anchors_r * 5 * qs, ring_w, INK,
                      np.sin(np.pi * qs))
            return out

        def key_at(t: float):
            """What the floor shows at scene time t, as a hashable key."""
            if t < z0:
                return ("red", ramp(t, 0.0, z0))
            if t < z1:
                return ("zoom", ramp(t, z0, z1))
            if t < l1:
                return ("land", ramp(t, z1, l1))
            if t >= m0:
                return ("merge", ramp(t, m0, m1))
            keep = 1 - ramp(t, e0, e1)                       # "and at the end": the marks fade
            marks = ramp(t, p0, p1)
            hatch = round(HATCH * ramp(t, h0, h1) * keep, 4)
            if sp0 <= t < sp1:
                return ("sparks", round(t * config.frame_rate), hatch)
            r = sum(1 for c in claim_t if t >= c)            # rounds whose claims have landed
            upto = None if t < sp0 else min(r, rounds - 1)
            tick = TICK_LO + (TICK_HI - TICK_LO) * ramp(t, b0, b1) - (TICK_HI - TICK_NOTED) * ramp(t, b2, b3)
            tick = round(tick * keep, 4) if r else 0.0
            return ("crop", upto, marks, round(marks * keep, 4), hatch, tick, r)

        def render(key) -> np.ndarray:
            kind = key[0]
            if kind == "red":
                return mix(start, sharp, key[1])
            if kind == "zoom":
                return window(ease(key[1]))
            if kind == "land":
                return mix(landed, crop_red, key[1])
            if kind == "sparks":
                return sparks(key[1] / config.frame_rate, key[2])
            if kind == "crop":
                _, upto, pieces, anchors, hatch, tick, r = key
                return crop(upto=upto, pieces=PIECES * pieces, anchors=ANCHORS * anchors, hatch=hatch, ticks=tick,
                            ticks_upto=r)
            if kind == "merge":
                return mix(crop(upto=rounds - 1, pieces=PIECES), crop(final=True), key[1])
            raise KeyError(key)

        last = {"key": None}
        t_stop = m1 + fr(2)

        def step(p: float) -> None:
            key = key_at(p * t_stop)
            if key != last["key"]:
                f.show_raster(render(key))
                last["key"] = key

        self.run(step, t_stop, start=0.0)

        # ---- "So first, every tile": the carried caption goes as the zoom starts
        self.unsay(frames=4)

        # ---- "pieces": the SMs pulse once (their own launch), the caption
        pl = self.Lay.machine_pl
        outlines = sm_outlines(pl)
        self.until(p0)
        self.add(*outlines)

        def pulse(q: float) -> None:
            for m in outlines:
                m.set_stroke(opacity=float(np.sin(np.pi * q)) if q < 1 else 0.0)

        self.run(pulse, PULSE)
        self.say("pieces", "blobs cut by tiles", at=t_pieces)

        # ---- "seeds": the caption
        self.say(f"{int(n['crop_seeds'])} seeds", f"{int(n['crop_blobs'])} blobs", at=t_seeds)
        self.remove(*outlines)

        # ---- "each seed sends a wave": the tokens carry pieces, then six rounds
        self.until(t_each)
        self.machine_to(end.machine, frames=6, slide=0)
        self.gate_rounds(flashes=flashes)

        # ---- "label": the carried caption
        self.say_last(at=t_label)
        self.until(self.target_seconds() - fr(1))
        assert self.machine.spec == end.machine, (self.machine.spec, end.machine)
        assert last["key"] == ("merge", 1.0), last["key"]
