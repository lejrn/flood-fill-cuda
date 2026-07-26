"""Wavefront visualization: the BFS timeline rendered from depth + owner.

No new GPU work — the kernels already record everything: depth[x,y] is the
level each pixel was filled at (the complete timeline) and owner[x,y] is
the block that processed it. This script replays them:

- Animated GIF per (kernel, scene): the frontier sweeps as a light band;
  behind it every pixel keeps a color encoding WHEN it was filled (light ->
  dark with level) and WHO filled it (block 0 = blues, block 1 = greens).
  The final frame is exactly the static gradient — the wave leaves the
  timeline behind as its trail.
- Static gradient PNG: that final first->last image on its own.

Because all three kernels compute the identical BFS, the timeline is the
same everywhere; the OWNER hues are what make the approaches distinct:
split shows two solid territories meeting at the seam, dirsplit shows two
direction-arcs chasing the wavefront, global shows an interleaved speckle.

Run:  uv run python -m flood_fill_cuda.chapters.ch02_gpu_1blob_2block.benchmarks.wavefront
Writes GIFs + PNGs to results/ch02_gpu_1blob_2block/wavefront/ (centralized,
not next to this script).
"""

import os

import numpy as np
from PIL import Image

from ..flood_fill import flood_fill
from .. import scenes
from ....shared import results_paths

OUT_DIR = results_paths.results_dir("ch02_gpu_1blob_2block", "wavefront")

MAX_FRAMES = 96
FRAME_MS = 60
HOLD_MS = 1500

# Sequential ramps (light -> dark by fill level), from the reference
# palette's blue ramp and a matching green ramp; the lightest tints are
# reserved for the frontier band so "just filled" never impersonates it.
BLUE_ANCHORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
GREEN_ANCHORS = ["#8fd98f", "#4dbb4d", "#1f9a1f", "#007300", "#003f00"]
FRONTIER = {0: "#cde2fb", 1: "#d9f4d9"}  # per-owner light tint


def _hex(c):
    return np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)],
                    dtype=np.float64)


def _ramp_lut(anchors, n=256):
    """n RGB rows interpolated through the anchor colors."""
    pts = np.array([_hex(a) for a in anchors])
    x = np.linspace(0, len(anchors) - 1, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(x, np.arange(len(anchors)),
                               pts[:, ch]).round().astype(np.uint8)
    return lut


LUTS = {0: _ramp_lut(BLUE_ANCHORS), 1: _ramp_lut(GREEN_ANCHORS)}


# (kernel, scene builder, output stem, upscale, gif?)
COMBOS = [
    ("split", lambda: scenes.square_scene(256, 256, 220, 220),
     "split_square256", 2, True),
    ("global", lambda: scenes.square_scene(256, 256, 220, 220),
     "global_square256", 2, True),
    ("dirsplit", lambda: scenes.square_scene(256, 256, 220, 220),
     "dirsplit_square256", 2, True),
    ("split", lambda: scenes.offcenter_blob_scene(256, 256, 100),
     "split_offcenter256", 2, True),
    ("dirsplit", lambda: scenes.serpentine_scene(128, 128),
     "dirsplit_serpentine128", 3, True),
    ("split", lambda: scenes.disk_scene(512, 512, 240),
     "split_disk512", 1, False),
    ("split", lambda: scenes.seam_serpentine_scene(256, 256),
     "split_seamserp256", 2, False),
]


def gradient_colors(depth, owner, levels):
    """(W, H, 3) uint8: every filled pixel colored by (owner ramp, level)."""
    frac = np.clip(depth.astype(np.float64) / max(levels - 1, 1), 0, 1)
    bins = (frac * 255).astype(np.int64)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    for o in (0, 1):
        mask = owner == o
        out[mask] = LUTS[o][bins[mask]]
    return out


def to_image(arr_xy3, upscale):
    """(W, H, 3) array-indexed [x, y] -> correctly oriented PIL image."""
    img = np.transpose(arr_xy3, (1, 0, 2))
    if upscale > 1:
        img = img.repeat(upscale, axis=0).repeat(upscale, axis=1)
    return Image.fromarray(img)


def render(kernel, builder, stem, upscale, make_gif):
    img, sx, sy = builder()
    r = flood_fill(img, sx, sy, kernel=kernel)
    depth, owner, levels = r.depth, r.owner, r.levels
    grad = gradient_colors(depth, owner, levels)
    reached = depth >= 0

    # static gradient: original scene with the timeline painted over the blob
    final = img.copy()
    final[reached] = grad[reached]
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    to_image(final, upscale).save(png_path)
    out = [f"{stem}.png ({os.path.getsize(png_path):,} B)"]

    if make_gif:
        front_cols = {o: _hex(FRONTIER[o]).astype(np.uint8) for o in (0, 1)}
        nframes = min(MAX_FRAMES, levels)
        thresholds = np.unique(
            np.linspace(0, levels - 1, nframes).round().astype(np.int64))
        frames, durations = [], []
        prev_t = -1
        for t in thresholds:
            frame = img.copy()
            filled = reached & (depth <= t)
            frame[filled] = grad[filled]
            band = reached & (depth > prev_t) & (depth <= t)
            for o in (0, 1):
                frame[band & (owner == o)] = front_cols[o]
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
    print(f"{kernel:9s} {'  '.join(out)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for combo in COMBOS:
        render(*combo)
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
