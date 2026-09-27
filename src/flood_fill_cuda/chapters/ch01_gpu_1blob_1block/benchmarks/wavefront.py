"""Wavefront visualization for Chapter 1, and for the CPU walk it replaces.

No new GPU work: the ch01 kernel records depth[x,y], the BFS level each
pixel was filled at, and this script replays it. Two combos on the SAME
scene the ch02 and ch03 wavefront GIFs use (a 220x220 square in a 256x256
image, centre seed), so the three sit side by side as siblings:

- gpu_1block: the one-block, 256-thread kernel (spill variant). One BLUE
  ramp, light -> dark with level, the frontier band lighter. Owner is
  implicitly block 0 everywhere, which is the chapter's whole point.
- cpu_order: the same BFS visited ONE PIXEL AT A TIME in the CPU oracle's
  queue order. A grey ramp, light -> dark with visit index (grey = no
  block owns anything), the last TRAIL_PX visits lighter as the cursor's
  wake, and a small dark square on the pixel being visited. The CPU's
  frontier is a moving point on the ring; the GPU's is the whole ring.

Both GIFs have 96 frames plus a held final frame, 60 ms per frame, so a
viewer can play them in lockstep. The CPU order is checked against the
chapter's own @njit oracle (same neighbour order, same level structure):
depth must be non-decreasing along the visit order, and the GPU depth map
must equal the oracle's bit for bit.

Run:  uv run python -m flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.wavefront
Writes GIFs + PNGs to results/ch01_gpu_1blob_1block/wavefront/.
"""

import os
from collections import deque

import numpy as np
from PIL import Image

from ..flood_fill import flood_fill
from ..cpu_oracle import cpu_flood_fill
from .. import scenes
from ....shared import results_paths

OUT_DIR = results_paths.results_dir("ch01_gpu_1blob_1block", "wavefront")

MAX_FRAMES = 96
FRAME_MS = 60
HOLD_MS = 1500
TPB = 256

# Block 0's ramp, identical to ch02's BLUE_ANCHORS so the one-block GIF
# and the two-block GIF share a vocabulary: this hue means "block 0".
BLUE_ANCHORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
FRONTIER_BLUE = "#cde2fb"

# The CPU owns nothing in particular: an achromatic ramp.
GREY_ANCHORS = ["#b9c0c8", "#939ba5", "#6e7680", "#4c535c", "#2b3037"]
TRAIL_GREY = "#dde1e6"
CURSOR = "#0b0f14"
CURSOR_DOT = 7      # cursor square, in scene pixels (before upscale)
TRAIL_PX = 128      # visits drawn as the cursor's wake

# Same neighbour order as cpu_oracle.py and the kernel: right, down, left, up.
_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _hex(c):
    return np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)],
                    dtype=np.float64)


def _ramp_lut(anchors, n=256):
    """n RGB rows interpolated through the anchor colors (as in ch02)."""
    pts = np.array([_hex(a) for a in anchors])
    x = np.linspace(0, len(anchors) - 1, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(x, np.arange(len(anchors)),
                               pts[:, ch]).round().astype(np.uint8)
    return lut


LUT_BLUE = _ramp_lut(BLUE_ANCHORS)
LUT_GREY = _ramp_lut(GREY_ANCHORS)


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def cpu_visit_order(img, sx, sy):
    """order[x, y] = index at which the CPU BFS dequeues the pixel, -1
    elsewhere. Plain deque BFS in the oracle's neighbour order; the
    dequeue order is the enqueue order, so depth is non-decreasing."""
    width, height = img.shape[0], img.shape[1]
    order = np.full((width, height), -1, dtype=np.int64)
    visited = np.zeros((width, height), dtype=np.bool_)
    visited[sx, sy] = True
    q = deque([(sx, sy)])
    n = 0
    while q:
        x, y = q.popleft()
        order[x, y] = n
        n += 1
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and not visited[nx, ny]
                    and _is_red(img, nx, ny)):
                visited[nx, ny] = True
                q.append((nx, ny))
    return order, n


def _bins(values, top):
    frac = np.clip(values.astype(np.float64) / max(top, 1), 0, 1)
    return (frac * 255).astype(np.int64)


def to_image(arr_xy3, upscale):
    """(W, H, 3) array-indexed [x, y] -> correctly oriented PIL image."""
    img = np.transpose(arr_xy3, (1, 0, 2))
    if upscale > 1:
        img = img.repeat(upscale, axis=0).repeat(upscale, axis=1)
    return Image.fromarray(img)


def gpu_frames(img, depth, levels):
    """Frames of the one-block fill: thresholds over BFS level."""
    reached = depth >= 0
    grad = np.zeros(depth.shape + (3,), dtype=np.uint8)
    grad[reached] = LUT_BLUE[_bins(depth, levels - 1)[reached]]
    final = img.copy()
    final[reached] = grad[reached]
    front = _hex(FRONTIER_BLUE).astype(np.uint8)
    nframes = min(MAX_FRAMES, levels)
    thresholds = np.unique(
        np.linspace(0, levels - 1, nframes).round().astype(np.int64))
    frames = []
    prev_t = -1
    for t in thresholds:
        frame = img.copy()
        filled = reached & (depth <= t)
        frame[filled] = grad[filled]
        band = reached & (depth > prev_t) & (depth <= t)
        frame[band] = front
        frames.append(frame)
        prev_t = t
    return frames, final


def cpu_frames(img, order, n):
    """Frames of the CPU walk: thresholds over the visit index, with a
    lighter trail of the last TRAIL_PX visits and a cursor square."""
    reached = order >= 0
    grad = np.zeros(order.shape + (3,), dtype=np.uint8)
    grad[reached] = LUT_GREY[_bins(order, n - 1)[reached]]
    final = img.copy()
    final[reached] = grad[reached]
    trail = _hex(TRAIL_GREY).astype(np.uint8)
    cursor = _hex(CURSOR).astype(np.uint8)
    width, height = img.shape[0], img.shape[1]
    nframes = min(MAX_FRAMES, n)
    thresholds = np.unique(
        np.linspace(0, n - 1, nframes).round().astype(np.int64))
    frames = []
    for t in thresholds:
        frame = img.copy()
        filled = reached & (order <= t)
        frame[filled] = grad[filled]
        wake = reached & (order > t - TRAIL_PX) & (order <= t)
        frame[wake] = trail
        xs, ys = np.nonzero(order == t)
        cx, cy = int(xs[0]), int(ys[0])
        r = CURSOR_DOT // 2
        frame[max(0, cx - r):min(width, cx + r + 1),
              max(0, cy - r):min(height, cy + r + 1)] = cursor
        frames.append(frame)
    return frames, final


def _write(stem, frames, final, upscale, make_gif):
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    to_image(final, upscale).save(png_path)
    out = [f"{stem}.png ({os.path.getsize(png_path):,} B)"]
    if make_gif:
        pil = [to_image(f, upscale) for f in frames]
        pil.append(to_image(final, upscale))
        durations = [FRAME_MS] * len(frames) + [HOLD_MS]
        gif_path = os.path.join(OUT_DIR, f"{stem}.gif")
        pil[0].save(gif_path, save_all=True, append_images=pil[1:],
                    duration=durations, loop=0, optimize=True)
        out.append(f"{stem}.gif ({len(pil)} frames, "
                   f"{os.path.getsize(gif_path):,} B)")
    return out


def render(mode, builder, stem, upscale, make_gif):
    img, sx, sy = builder()
    _, depth_ref, levels_ref, filled_ref = cpu_flood_fill(img, sx, sy)
    if mode == "gpu_1block":
        r = flood_fill(img, sx, sy, threads_per_block=TPB, variant="spill")
        assert r.filled == filled_ref, (r.filled, filled_ref)
        assert np.array_equal(r.depth, depth_ref), "GPU depth != oracle depth"
        frames, final = gpu_frames(img, r.depth, r.levels)
        note = f"1x{TPB}, {r.levels} levels, {r.filled:,} px"
    elif mode == "cpu_order":
        order, n = cpu_visit_order(img, sx, sy)
        assert n == filled_ref, (n, filled_ref)
        idx = order[order >= 0]
        d = depth_ref[order >= 0]
        assert (np.diff(d[np.argsort(idx)]) >= 0).all(), \
            "visit order is not level-synchronous"
        frames, final = cpu_frames(img, order, n)
        note = f"{n:,} visits, {levels_ref} levels"
    else:
        raise ValueError(mode)
    out = _write(stem, frames, final, upscale, make_gif)
    print(f"{mode:10s} {note:32s} {'  '.join(out)}")


# (mode, scene builder, output stem, upscale, gif?)
COMBOS = [
    ("gpu_1block", lambda: scenes.square_scene(256, 256, 220, 220),
     "square256_b1_t256", 2, True),
    ("cpu_order", lambda: scenes.square_scene(256, 256, 220, 220),
     "square256_cpu_order", 2, True),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for combo in COMBOS:
        render(*combo)
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
