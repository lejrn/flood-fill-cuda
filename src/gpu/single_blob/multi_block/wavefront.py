"""Wavefront visualization: the N-block BFS timeline from depth + owner.

No new GPU work — the kernel already records everything: depth[x,y] is the
level each pixel was filled at and owner[x,y] is the block that claimed it
(0..blocks-1, int16). This script replays them:

- Animated GIF per combo: the frontier sweeps as a light band; behind it
  every pixel keeps a color encoding WHEN it was filled (light -> dark
  with level) and WHO filled it (one hue per block, spaced around the
  color wheel by the golden angle so adjacent block ids never look alike).
- Static gradient PNG: the final first->last image on its own.

What the hues reveal at this stage: ownership is SPECKLE, not territory.
A block owns contiguous chunks of the queue window, but queue order is
warp-aggregated discovery order, which scrambles spatially within a few
levels — so the first rings around the seed keep visible per-block
coherence and then ownership decays into fine-grained rainbow noise. That
speckle is a real finding, not a rendering artifact: it is the spatial
scatter behind the bandwidth chapter's sector-inflation story (adjacent
pixels are written by unrelated warps). The combos also show when blocks
work at all: with the grid smaller than the frontier (8x32, 8x64) every
block appears; with the benchmark's own 48x256 config on a 256^2 scene a
~440 px frontier only ever feeds two 256-thread blocks (46 idle at every
barrier); and the serpentine's 1-pixel frontier hands block 0 every
single pixel. Block hues avoid the red band entirely — unfilled scene
pixels ARE red, and nothing may impersonate them.

Run:  uv run python src/gpu/single_blob/multi_block/wavefront.py
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

GOLDEN_ANGLE = 137.508  # degrees; spreads any block count evenly

# Hues live in [30°, 330°]: the ±30° band around red is excluded because
# the scenes' unfilled pixels are pure red — no block may impersonate them.
HUE_LO, HUE_SPAN = 30.0, 300.0


def _block_hue(b):
    return (HUE_LO + (b * GOLDEN_ANGLE) % HUE_SPAN) / 360.0


def _ramp_lut(hue, n=256):
    """n RGB rows for one block: light -> dark with fill level. The
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


# (scene builder, blocks, tpb, output stem, upscale, gif?, max frames)
COMBOS = [
    # 256 grid threads vs a ~440 px peak frontier: all 8 blocks work; the
    # first rings keep per-block coherence, then ownership speckles
    (lambda: scenes.square_scene(256, 256, 220, 220), 8, 32,
     "square256_b8_t32", 2, True, 96),
    # corner seed: the frontier grows 1 -> 256, so the blocks come online
    # one at a time as it outgrows the threads in front of them
    (lambda: scenes.full_red_scene(256, 256), 8, 32,
     "fullred256_b8_t32", 2, True, 96),
    # 512 threads vs rings up to ~1,400 px: all 8 blocks on a disk
    (lambda: scenes.disk_scene(512, 512, 240), 8, 64,
     "disk512_b8_t64", 1, True, 72),
    # the benchmark's own config on a small scene: only blocks 0-1 ever
    # appear — 46 of 48 blocks idle at every barrier (honest flip side)
    (lambda: scenes.square_scene(256, 256, 220, 220), 48, 256,
     "square256_b48_t256", 2, False, 0),
    # 1-pixel frontier: block 0 owns literally every pixel (starvation)
    (lambda: scenes.serpentine_scene(128, 128), 8, 32,
     "serpentine128_b8_t32", 3, False, 0),
]


def gradient_colors(depth, owner, levels, luts):
    """(W, H, 3) uint8: every filled pixel colored by (owner hue, level)."""
    frac = np.clip(depth.astype(np.float64) / max(levels - 1, 1), 0, 1)
    bins = (frac * 255).astype(np.int64)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    for o in np.unique(owner[owner >= 0]):
        mask = owner == o
        out[mask] = luts[int(o)][bins[mask]]
    return out


def to_image(arr_xy3, upscale):
    """(W, H, 3) array-indexed [x, y] -> correctly oriented PIL image."""
    img = np.transpose(arr_xy3, (1, 0, 2))
    if upscale > 1:
        img = img.repeat(upscale, axis=0).repeat(upscale, axis=1)
    return Image.fromarray(img)


def render(builder, blocks, tpb, stem, upscale, make_gif, max_frames):
    img, sx, sy = builder()
    r = flood_fill(img, sx, sy, threads_per_block=tpb, blocks=blocks)
    depth, owner, levels = r.depth, r.owner, r.levels
    luts = {b: _ramp_lut(_block_hue(b)) for b in range(blocks)}
    grad = gradient_colors(depth, owner, levels, luts)
    reached = depth >= 0
    active = int((r.processed_per_block > 0).sum())

    # static gradient: original scene with the timeline painted over the blob
    final = img.copy()
    final[reached] = grad[reached]
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    to_image(final, upscale).save(png_path)
    out = [f"{stem}.png ({os.path.getsize(png_path):,} B)"]

    if make_gif:
        front_cols = {b: _frontier_color(_block_hue(b))
                      for b in range(blocks)}
        nframes = min(max_frames or MAX_FRAMES, levels)
        thresholds = np.unique(
            np.linspace(0, levels - 1, nframes).round().astype(np.int64))
        frames, durations = [], []
        prev_t = -1
        for t in thresholds:
            frame = img.copy()
            filled = reached & (depth <= t)
            frame[filled] = grad[filled]
            band = reached & (depth > prev_t) & (depth <= t)
            for o in np.unique(owner[band]):
                frame[band & (owner == o)] = front_cols[int(o)]
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
    print(f"{blocks:>3d}x{tpb:<4d} ({active:>2d} of {blocks} blocks worked)  "
          f"{'  '.join(out)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for combo in COMBOS:
        render(*combo)
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
