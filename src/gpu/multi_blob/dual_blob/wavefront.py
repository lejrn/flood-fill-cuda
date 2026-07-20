"""Wavefront visualization for two blobs: one clock, two color families.

No new GPU work — the kernel already recorded everything: depth[x,y] is
the level each pixel was filled at and label[x,y] (recovered from the
painted colors) says which blob it belongs to. Unlike the single-blob
stages, hue encodes the BLOB here (blue family for blob 0, green family
for blob 1), not the owning block — with two spatially separate blobs the
owner-speckle would only repeat Chapter 3's finding and bury this stage's
actual signal: WHICH BLOB IS STILL FLOODING.

Every frame is shaded on a single global clock (light -> dark with fill
tick over the WHOLE run, not per blob), which is what makes the two
timelines comparable:

- *_multisource.gif  both waves advance at once; on the asymmetric pair
  the green blob finishes early and stays LIGHT while the blue wave keeps
  darkening — max(tA, tB) made visible.
- *_sequential.gif   the same result arrays replayed on the sequential
  clock (blob B's ticks shifted by blob A's level count — legitimate
  because sequential mode provably produces the identical depth/label
  maps): blob A runs to completion first, and the green blob comes out
  mid-to-dark because it ran LATE — tA + tB made visible next to the
  multisource GIF of the same scene.

Both hues sit far outside the +-30 degree band around red: unfilled scene
pixels ARE red, and nothing may impersonate them.

Run:  uv run python src/gpu/multi_blob/dual_blob/wavefront.py
Writes GIFs + PNGs to wavefront/ next to this script.
"""

import colorsys
import os

import numpy as np
from PIL import Image

from flood_fill import flood_fill
import scenes

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "wavefront")

MAX_FRAMES = 96
FRAME_MS = 60
HOLD_MS = 1500

HUES = {0: 220.0 / 360.0,   # blob 0: blue family (kernel paints it blue)
        1: 130.0 / 360.0}   # blob 1: green family (kernel paints it green)


def _ramp_lut(hue, n=256):
    """n RGB rows for one blob: light -> dark with global fill tick. The
    lightest tints stay above the ramp (reserved for the frontier band)."""
    lut = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        light = 0.78 - 0.50 * i / (n - 1)          # 0.78 -> 0.28
        r, g, b = colorsys.hls_to_rgb(hue, light, 0.62)
        lut[i] = (round(r * 255), round(g * 255), round(b * 255))
    return lut


def _frontier_color(hue):
    r, g, b = colorsys.hls_to_rgb(hue, 0.90, 0.55)
    return np.array([round(r * 255), round(g * 255), round(b * 255)],
                    dtype=np.uint8)


LUTS = {lbl: _ramp_lut(h) for lbl, h in HUES.items()}
FRONTS = {lbl: _frontier_color(h) for lbl, h in HUES.items()}


def gradient_colors(time_map, label, ticks):
    """(W, H, 3) uint8: every filled pixel colored by (blob hue, global
    fill tick)."""
    frac = np.clip(time_map.astype(np.float64) / max(ticks - 1, 1), 0, 1)
    bins = (frac * 255).astype(np.int64)
    out = np.zeros(time_map.shape + (3,), dtype=np.uint8)
    for lbl in (0, 1):
        mask = label == lbl
        out[mask] = LUTS[lbl][bins[mask]]
    return out


def to_image(arr_xy3, upscale):
    """(W, H, 3) array-indexed [x, y] -> correctly oriented PIL image."""
    img = np.transpose(arr_xy3, (1, 0, 2))
    if upscale > 1:
        img = img.repeat(upscale, axis=0).repeat(upscale, axis=1)
    return Image.fromarray(img)


def render_timeline(img, time_map, label, ticks, stem, upscale,
                    max_frames=MAX_FRAMES, make_gif=True):
    """Render one timeline (a (W,H) tick map, -1 = never filled) as a
    static gradient PNG and optionally an animated GIF."""
    reached = time_map >= 0
    grad = gradient_colors(time_map, label, ticks)

    final = img.copy()
    final[reached] = grad[reached]
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    to_image(final, upscale).save(png_path)
    out = [f"{stem}.png ({os.path.getsize(png_path):,} B)"]

    if make_gif:
        nframes = min(max_frames, ticks)
        thresholds = np.unique(
            np.linspace(0, ticks - 1, nframes).round().astype(np.int64))
        frames, durations = [], []
        prev_t = -1
        for t in thresholds:
            frame = img.copy()
            filled = reached & (time_map <= t)
            frame[filled] = grad[filled]
            band = reached & (time_map > prev_t) & (time_map <= t)
            for lbl in (0, 1):
                frame[band & (label == lbl)] = FRONTS[lbl]
            frames.append(to_image(frame, upscale))
            durations.append(FRAME_MS)
            prev_t = t
        frames.append(to_image(final, upscale))
        durations.append(HOLD_MS)
        gif_path = os.path.join(OUT_DIR, f"{stem}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True)
        out.append(f"{stem}.gif ({len(frames)} frames, "
                   f"{os.path.getsize(gif_path):,} B)")
    print(f"  {'  '.join(out)}")


# (scene builder, blocks, tpb, stem, upscale, also sequential replay?)
COMBOS = [
    # the money shot: big + small blob. Multisource: green finishes early
    # and stays light. Sequential replay: green runs late and comes out
    # dark. Same pixels, two clocks.
    (lambda: scenes.asym_squares_scene(384, 256, 180, 60, gap=8),
     8, 32, "asym384_b8_t32", 2, True),
    # equal pair: both waves in lockstep, finishing together
    (lambda: scenes.two_squares_scene(320, 192, 120, 120, gap=8),
     8, 32, "twosq320_b8_t32", 2, False),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for builder, blocks, tpb, stem, upscale, seq_replay in COMBOS:
        img, seeds = builder()
        r = flood_fill(img, seeds, mode="multisource",
                       threads_per_block=tpb, blocks=blocks)
        print(f"{stem}: filled={r.filled:,d} (a={r.filled_a:,d} "
              f"b={r.filled_b:,d}) levels={r.levels} "
              f"(a={r.levels_a}, b={r.levels_b})")

        # multisource clock: tick = BFS level, both blobs at once
        render_timeline(img, r.depth, r.label, r.levels,
                        f"{stem}_multisource", upscale)

        if seq_replay:
            # sequential clock: blob A first (ticks 0..la-1), then blob B
            # (ticks la..la+lb-1) — same arrays, shifted timeline
            seq_time = r.depth.copy()
            seq_time[r.label == 1] += r.levels_a
            render_timeline(img, seq_time, r.label,
                            r.levels_a + r.levels_b,
                            f"{stem}_sequential", upscale)

    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
