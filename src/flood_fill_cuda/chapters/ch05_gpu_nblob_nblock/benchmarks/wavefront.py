"""Wavefront visualization for seed discovery: the merge made visible.

No new GPU work — the kernels already recorded everything: depth[x,y] is
the level each pixel was filled at, label[x,y] the canonical blob, and
(seed_merge instrumented) prov_label[x,y] the PRE-merge label — which
candidate's wave actually got there first, preserved by the flatten
before it repaints.

Hue encodes the LABEL (rank-spread around the color wheel, excluding the
+-30 degree band around red — unfilled scene pixels ARE red and nothing
may impersonate them), light -> dark encodes the global level clock.

The money shot is the seed_merge pair on a multi-candidate blob:

- *_prov.gif   every pixel hued by its PROVISIONAL label: several waves
  in different colors race from the candidates and slam into each other
  mid-blob — the collisions where the unions happened.
- *_final.gif  the same depth clock hued by the CANONICAL label: one
  color per blob. Same pixels, same clock — the merge erased the seams.

Run:  uv run python -m flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.benchmarks.wavefront
Writes GIFs + PNGs to results/ch05_gpu_nblob_nblock/wavefront/
(centralized, not next to this script).
"""

import colorsys
import os

import numpy as np
from PIL import Image

from ..flood_fill import flood_fill
from .. import scenes
from ....shared import results_paths

OUT_DIR = results_paths.results_dir("ch05_gpu_nblob_nblock", "wavefront")

MAX_FRAMES = 96
FRAME_MS = 60
HOLD_MS = 1500

# Hues live in [30, 330] degrees — the +-30 band around red is reserved
# for the unfilled scene itself.
_HUE_LO, _HUE_SPAN = 30.0, 300.0
_GOLDEN = 137.508


def _hues_for(labels):
    """label value -> hue fraction, rank-spread. Few labels get an even
    spread; many get golden-angle steps so neighbors-by-rank stay far
    apart on the wheel. Hues quantize to 0.5-degree steps so the LUT
    cache below stays small at any label count."""
    uniq = sorted(int(l) for l in labels)
    n = len(uniq)
    hues = {}
    for i, l in enumerate(uniq):
        if n <= 8:
            deg = _HUE_LO + _HUE_SPAN * (i + 0.5) / n
        else:
            deg = _HUE_LO + (i * _GOLDEN) % _HUE_SPAN
        hues[l] = round(deg * 2) / 2 / 360.0
    return hues


def _ramp_lut(hue, n=256):
    """n RGB rows for one label: light -> dark with global fill tick. The
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


def color_maps(time_map, label, reached, ticks, labels, hues):
    """Two (W, H, 3) uint8 maps: the light->dark gradient color of every
    pixel and its frontier-flash color. Fully vectorized — per-label
    LUTs are stacked and fancy-indexed, so the cost is O(pixels)
    regardless of the label count (a 21k-blob input would take hours
    with a mask-per-label loop)."""
    cache = {}
    for h in hues.values():
        if h not in cache:
            cache[h] = (_ramp_lut(h), _frontier_color(h))
    lut_stack = np.stack([cache[hues[int(l)]][0] for l in labels])
    front_stack = np.stack([cache[hues[int(l)]][1] for l in labels])
    idx = np.searchsorted(labels, label).clip(max=len(labels) - 1)
    idx[~reached] = 0        # arbitrary — masked out by callers
    frac = np.clip(time_map.astype(np.float64) / max(ticks - 1, 1), 0, 1)
    bins = (frac * 255).astype(np.int64)
    return lut_stack[idx, bins], front_stack[idx]


def to_image(arr_xy3, upscale):
    """(W, H, 3) array-indexed [x, y] -> correctly oriented PIL image."""
    img = np.transpose(arr_xy3, (1, 0, 2))
    if upscale > 1:
        img = img.repeat(upscale, axis=0).repeat(upscale, axis=1)
    return Image.fromarray(img)


def render_timeline(img, time_map, label, ticks, stem, upscale,
                    max_frames=MAX_FRAMES, make_gif=True):
    """Render one timeline (a (W,H) tick map, -1 = never filled) as a
    static gradient PNG and optionally an animated GIF, hued per label."""
    reached = time_map >= 0
    labels = np.unique(label[reached])
    hues = _hues_for(labels)
    grad, front_colors = color_maps(time_map, label, reached, ticks,
                                    labels, hues)

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
            frame[band] = front_colors[band]
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


# (scene builder, variant, stem, upscale, render prov replay too?,
#  optional extra flood_fill kwargs, optional downscale — nearest-
#  neighbor ::d subsample of the FINISHED run before rendering, for
#  inputs too large to rasterize frame by frame)
COMBOS = [
    # THE money shot: one U, two candidates. prov = two waves in two
    # colors racing until they collide at the bridge; final = one color,
    # the union erased the seam.
    (lambda: scenes.u_shape_scene(192, 192, arm_len=110, arm_w=34, gap=60),
     "seed_merge", "u192_merge", 3, True),
    # 24 teeth, 24 waves, one comb — the merge at scale
    (lambda: scenes.comb_scene(96, 64, teeth=24, tooth_len=64, spine_w=6),
     "seed_merge", "comb96_merge", 4, True),
    # N blobs, N hues, one launch, no seeds given — both variants' clocks:
    # ccl waves ripple out from each lex-min corner, merge waves from
    # every local corner at once
    (lambda: scenes.blob_grid_scene(288, 216, 4, 3, 54, gap=18),
     "ccl_fill", "grid12_ccl", 2, False),
    (lambda: scenes.blob_grid_scene(288, 216, 4, 3, 54, gap=18),
     "seed_merge", "grid12_merge", 2, False),
    # the shatter: hundreds of blobs discovered and filled at once
    (lambda: scenes.random_blobs_scene(192, 192, density=0.3, rng_seed=7),
     "ccl_fill", "random192_ccl", 3, False),
    # the interior rule made visible: same disk twice. Corner seeding =
    # one wave sweeping radially through the level ramp; interior S1 =
    # every 8-neighbors-red pixel is a seed, the whole mass lights at
    # level 0 and only the staircase edge fills late.
    (lambda: scenes.disk_scene(192, 192, 80),
     "seed_merge", "disk192_corner", 3, False),
    (lambda: scenes.disk_scene(192, 192, 80),
     "seed_merge", "disk192_interior", 3, False,
     dict(lattice=1, interior=True)),
]

# External PNGs (gitignored — appended only when present). The 9000²
# blobs image is subsampled 10:1 for rendering; the run itself is full
# resolution.
_PNG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        *[".."] * 5, "images", "input")
for _stem, _ds in (("input_blobs", 10), ("input_blocks", 2)):
    _p = os.path.join(_PNG_DIR, _stem + ".png")
    if os.path.exists(_p):
        COMBOS.append((lambda p=_p: scenes.png_scene(p),
                       "seed_merge", _stem, 1, False, {}, _ds))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for builder, variant, stem, upscale, prov_replay, *rest in COMBOS:
        ff_kwargs = rest[0] if rest else {}
        downscale = rest[1] if len(rest) > 1 else 1
        img, _ = builder()
        r = flood_fill(img, variant=variant, **ff_kwargs)
        print(f"{stem}: blobs={r.n_blobs:,d} candidates={r.candidates:,d} "
              f"unions={r.union_done:,d} filled={r.filled:,d} "
              f"levels={r.levels}")

        img_r, depth_r, label_r, prov_r = img, r.depth, r.label, r.prov_label
        if downscale > 1:
            img_r = img_r[::downscale, ::downscale]
            depth_r = depth_r[::downscale, ::downscale]
            label_r = label_r[::downscale, ::downscale]
            if prov_r is not None:
                prov_r = prov_r[::downscale, ::downscale]
        render_timeline(img_r, depth_r, label_r, r.levels,
                        f"{stem}_final", upscale)
        if prov_replay:
            render_timeline(img_r, depth_r, prov_r, r.levels,
                            f"{stem}_prov", upscale)

    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
