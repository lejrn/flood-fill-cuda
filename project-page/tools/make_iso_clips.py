"""The page's 3D clips: every step of the story, one box per pixel.

Drawn with tools/iso.py from recorded data only:

    tools/small_scenes.npz      the chapters' kernels on small scenes
                                (record_small.py, checked against the oracle)
    video/assets/tiles/*.npz    chapter 7's tile close-up and tile rounds,
                                chapter 8's pieces and waves on a crop

Outputs, under project-page/static/ (an .mp4 and a .webp poster each):

    iso_cpu_gpu     one CPU core beside one GPU block, on one step counter
    iso_blocks      8 blocks sharing one wave: 4 neighbours beside 8
    iso_merge       two waves meet in a U and become one blob
    iso_tile        one warp closing one 32 x 32 tile in registers
    iso_tiles_ab    a pixel BFS beside the tile BFS, on one step counter
    iso_waves       pieces, seeds and waves on a crop of the test image

Two things are drawn from a rule and not from a recording, and the page
says so beside them: the pixel BFS in iso_tiles_ab (on a solid square,
level k of an 8-neighbour BFS is the ring at distance k), and nothing
else. A "step" is one loop iteration of that method, never a time.

Run from the repository root (CPU only, about five minutes):

    FFMPEG=video/.venv/bin/ffmpeg .venv/bin/python project-page/tools/make_iso_clips.py [name ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iso  # noqa: E402

PAGE = Path(__file__).resolve().parents[1]
REPO = PAGE.parent
HERE = Path(__file__).resolve().parent
TILES = REPO / "video" / "assets" / "tiles"
VID = PAGE / "static" / "videos"
IMG = PAGE / "static" / "images"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

BLUE = (76, 143, 214)
AMBER = (217, 154, 43)
HUES = [BLUE, (63, 185, 80), (211, 106, 194), (153, 203, 77), (95, 201, 214), (163, 113, 247),
        (232, 163, 61), (120, 200, 160)]
PIECE_REDS = [(226, 100, 90), (176, 70, 64), (240, 142, 132), (140, 52, 48)]
PANE = (760, 600)


def newest(pattern: str) -> Path:
    hits = sorted(glob.glob(str(TILES / pattern)))
    if not hits:
        sys.exit(f"no {pattern} in {TILES}")
    return Path(hits[-1])


def ramp(colour, frac: np.ndarray) -> np.ndarray:
    """Light where frac is 0 (early), dark where it is 1 (late)."""
    c = np.asarray(colour, float)
    light = c + (255 - c) * 0.32
    dark = c * 0.62
    f = np.clip(frac, 0, 1)[..., None]
    return (light * (1 - f) + dark * f).astype(np.uint8)


def write(name: str, frames: list, poster_at: int) -> None:
    iso.poster(frames[poster_at][0], IMG / f"{name}.webp")
    iso.encode(frames, VID / f"{name}.mp4", ffmpeg=FFMPEG)


# ------------------------------------------------------------ the clips

def cpu_gpu() -> None:
    z = np.load(HERE / "small_scenes.npz")
    red, order, depth = z["blob_red"], z["blob_cpu_order"], z["blob_ch01_depth"]
    n = int(red.sum())
    levels = int(depth.max()) + 1
    fl = iso.Floor(*red.shape, *PANE)
    cpu_when = np.where(order >= 0, order.astype(float), np.inf)
    gpu_when = np.where(depth >= 0, depth.astype(float), np.inf)
    cpu_final = ramp(AMBER, order / max(n - 1, 1))
    gpu_final = ramp(BLUE, depth / max(levels - 1, 1))

    def frame(t: float):
        done_gpu = t >= levels
        left = iso.wave_frame(fl, red, cpu_when, cpu_final, t)
        right = iso.wave_frame(fl, red, gpu_when, gpu_final, t)
        cs = min(int(t) + 1, n)
        iso.label(left, "one CPU core", f"step {cs:,} of {n}" if cs < n else f"done: {n} steps")
        iso.label(right, "one GPU block", f"done: {levels} steps" if done_gpu
                  else f"step {min(int(t) + 1, levels)} of {levels}")
        return iso.hstack([left, right])

    frames = [(frame(0.0), 20)]
    t = 0.0
    while t < levels + 2:                       # one step every 6 video frames
        t += 0.5
        frames.append((frame(t), 3))
    poster_at = len(frames) - 6
    frames.append((frame(t), 25))
    while t < n + 3:                            # then let the CPU finish, faster
        t += 9.0
        frames.append((frame(t), 2))
    frames.append((frame(n + 10.0), 75))
    write("iso_cpu_gpu", frames, poster_at)


def blocks() -> None:
    z = np.load(HERE / "small_scenes.npz")
    facts = json.loads((HERE / "small_scenes.json").read_text())["facts"]["big"]
    red = z["big_red"]
    fl = iso.Floor(*red.shape, *PANE, frame_px=0.75)
    sides = []
    for depth, owner, tag in ((z["big_ch03_depth"], z["big_ch03_owner"], "4 neighbors"),
                              (z["big_ch03_depth8"], z["big_ch03_owner8"], "8 neighbors")):
        levels = int(depth.max()) + 1
        final = np.zeros(red.shape + (3,), np.uint8)
        for b in np.unique(owner[red]):
            m = red & (owner == b)
            final[m] = ramp(HUES[int(b) % len(HUES)], depth[m] / max(levels - 1, 1))
        sides.append((np.where(depth >= 0, depth.astype(float), np.inf), final, levels, tag))
    last = max(s[2] for s in sides)
    print("iso_blocks: blocks that worked:", facts["ch03"]["blocks_that_worked"],
          facts["ch03_conn8"]["blocks_that_worked"])

    def frame(t: float):
        panes = []
        for when, final, levels, tag in sides:
            p = iso.wave_frame(fl, red, when, final, t)
            iso.label(p, tag, f"done: {levels} levels" if t >= levels else f"level {int(t) + 1} of {levels}")
            panes.append(p)
        return iso.hstack(panes)

    frames = [(frame(0.0), 20)]
    t = 0.0
    while t < last + 2:
        t += 0.5
        frames.append((frame(t), 3))
    frames.append((frame(last + 10.0), 80))
    write("iso_blocks", frames, len(frames) // 2)


def merge() -> None:
    z = np.load(HERE / "small_scenes.npz")
    red, depth, prov, final_label = z["u_red"], z["u_depth"], z["u_prov_label"], z["u_label"]
    levels = int(depth.max()) + 1
    fl = iso.Floor(*red.shape, 960, 640)
    when = np.where(depth >= 0, depth.astype(float), np.inf)
    labels = sorted(int(v) for v in np.unique(prov[red]))
    hue = {l: HUES[i % len(HUES)] for i, l in enumerate(labels)}
    two = np.zeros(red.shape + (3,), np.uint8)
    for l in labels:
        m = red & (prov == l)
        two[m] = ramp(hue[l], depth[m] / max(levels - 1, 1))
    winner = int(np.unique(final_label[red])[0])
    assert len(np.unique(final_label[red])) == 1 and winner in hue, "the U must end as one blob"
    one = np.zeros(red.shape + (3,), np.uint8)
    one[red] = ramp(hue[winner], depth[red] / max(levels - 1, 1))

    def frame(t: float, blend: float = 0.0, note: str = ""):
        final = (two * (1 - blend) + one * blend).astype(np.uint8)
        p = iso.wave_frame(fl, red, when, final, t)
        return iso.label(p, "no seeds given", note)

    frames = [(frame(-1.0, note="the kernel finds two starting pixels"), 30)]
    t = 0.0
    while t < levels + 2:
        t += 0.5
        frames.append((frame(t, note=f"two waves, level {min(int(t) + 1, levels)} of {levels}"), 3))
    poster_at = len(frames) - 12
    frames.append((frame(t, note="the waves met: two labels on one blob"), 35))
    for i in range(1, 13):
        frames.append((frame(t, i / 12, "the smaller label wins"), 2))
    frames.append((frame(t, 1.0, "one blob, one label"), 80))
    write("iso_merge", frames, poster_at)


def tile() -> None:
    z = np.load(newest("ch07_tiles_*.npz"))

    def bits(words):                        # [lane, bit]
        return np.unpackbits(np.ascontiguousarray(words).view(np.uint8).reshape(32, 4), axis=1,
                             bitorder="little").astype(bool)

    # lane -> image row, bit -> image column: a thread's row is a screen row
    red = bits(z["cu_mask"]).T
    sweeps = [bits(w).T for w in z["cu_sweeps"]]
    counts = [int(s.sum()) for s in sweeps]
    assert all((s & ~red).sum() == 0 for s in sweeps) and counts[-1] == int(red.sum())
    when = np.full(red.shape, np.inf)
    for i in range(len(sweeps) - 1, -1, -1):
        when[sweeps[i]] = i
    names = ["entry: bits from the tile next door"] + ["fill along the rows", "step diagonally",
                                                       "fill along the columns"] * 4
    final = ramp(BLUE, np.where(np.isfinite(when), when, 0) / 7.0)
    fl = iso.Floor(32, 32, 960, 640)
    print("iso_tile: reached per sweep", counts)

    frames = [(iso.label(iso.wave_frame(fl, red, when, final, -1.0), "one tile, 32 x 32",
                         f"{counts[-1]} red pixels, one thread per row"), 40)]
    for i, name in enumerate(names):
        if i > 0 and counts[i] == counts[i - 1] and i > 7:
            break                           # nothing changes any more
        for k in range(6):
            p = iso.wave_frame(fl, red, when, final, i + k * 0.3)
            iso.label(p, name, f"{counts[i]} of {counts[-1]} reached")
            frames.append((p, 5 if k < 5 else 14))
    poster_at = len(frames) - 30
    p = iso.wave_frame(fl, red, when, final, len(names) + 5.0)
    frames.append((iso.label(p, "nothing changed: the tile is closed", f"{counts[-1]} of {counts[-1]}"), 80))
    write("iso_tile", frames, poster_at)


def tiles_ab() -> None:
    npz = newest("ch07_tiles_*.npz")
    first = np.load(npz)["sq_first_open"]
    meta = json.loads(npz.with_suffix(".json").read_text())
    levels = int(meta["square"]["oracle_levels"])
    half = 20                                   # a window of 41 x 41 tiles round the seed
    c = first.shape[0] // 2                     # the seed pixel (4000, 4000) opens tile (125, 125)
    win = first[c - half:c + half + 1, c - half:c + half + 1].astype(float)
    n = win.shape[0]
    red = np.ones((n, n), bool)
    fl = iso.Floor(n, n, *PANE)
    steps = int(win.max()) + 1
    final = ramp(BLUE, win / max(steps - 1, 1))
    never = np.full((n, n), np.inf)
    print(f"iso_tiles_ab: window {n} x {n} tiles = {n * 32} px, first-open rounds 0..{steps - 1}")

    def frame(t: float):
        k = min(int(t), steps - 1)
        side = (2 * k + 1) / 32.0               # the pixel wave, in tiles
        # the seed pixel is the first pixel of tile `half` in the window
        box = iso.Box(half + 0.5 / 32 - side / 2, half + 0.5 / 32 - side / 2, side, side, 1.5, iso.POP)
        left = iso.wave_frame(fl, red, never, final, t, extra=[box])
        right = iso.wave_frame(fl, red, win, final, t)
        iso.label(left, "pixel BFS", f"step {k + 1:,} of {levels:,}: {2 * k + 1} px wide")
        iso.label(right, "tile BFS", f"round {k + 1}: {(2 * k + 1) * 32:,} px wide")
        return iso.hstack([left, right])

    frames = [(frame(0.0), 24)]
    t = 0.0
    while t < steps + 1.5:
        t += 0.5
        frames.append((frame(t), 4))
    frames.append((frame(steps + 10.0), 85))
    write("iso_tiles_ab", frames, len(frames) * 2 // 3)


def waves() -> None:
    z = np.load(newest("ch08_waves_*.npz"))
    q = 4                                        # one box is a 4 x 4 pixel block
    red_px, comp_px = z["crop_red"], z["crop_comp"]
    n = red_px.shape[0] // q
    blocks_red = red_px.reshape(n, q, n, q).swapaxes(1, 2).reshape(n, n, q * q)
    blocks_comp = comp_px.reshape(n, q, n, q).swapaxes(1, 2).reshape(n, n, q * q)
    red = blocks_red.sum(axis=2) * 2 >= q * q    # a box is red if half its pixels are
    comp = np.full((n, n), -1, np.int64)
    for x, y in zip(*np.nonzero(red)):
        ids = blocks_comp[x, y][blocks_comp[x, y] >= 0]
        comp[x, y] = np.bincount(ids).argmax()
    is_seed, claim = z["crop_comp_seed"], z["crop_comp_round"]
    wave, label_, anchor = z["crop_comp_wave"], z["crop_comp_label"], z["crop_comp_anchor"]
    local = z["crop_comp_local"]
    last = int(claim.max())
    seeds = np.flatnonzero(is_seed)
    rank = {int(k): i for i, k in enumerate(seeds)}
    wave_rgb = np.array([HUES[rank[int(w)] % len(HUES)] for w in wave], float)
    # spread the seeds over more than eight colours: vary the lightness by rank
    for k in range(len(wave_rgb)):
        r = rank[int(wave[k])]
        wave_rgb[k] = wave_rgb[k] * (0.72 + 0.14 * (r // len(HUES) % 4))
    wave_rgb = np.clip(wave_rgb, 0, 255)
    owner = {int(l): int(np.flatnonzero((label_ == l) & (anchor == l))[0]) for l in np.unique(label_)}
    blob_rgb = np.array([wave_rgb[owner[int(l)]] for l in label_])
    c = np.maximum(comp, 0)
    when = np.where(red, claim[c].astype(float), np.inf)
    fl = iso.Floor(n, n, 960, 700, frame_px=0.6)
    n_pieces, n_seeds, n_blobs = len(is_seed), int(is_seed.sum()), len(owner)
    claimed = [int((claim == r).sum()) for r in range(1, last + 1)]
    print(f"iso_waves: {n} x {n} boxes, {n_pieces} pieces, {n_seeds} seeds, claimed {claimed}, {n_blobs} blobs")

    def plain(pieces: bool):
        colours = np.empty((n, n, 3), np.uint8)
        colours[:] = iso.FLOOR
        colours[red] = (np.array(PIECE_REDS)[local[c[red]] % len(PIECE_REDS)] if pieces else iso.RED_PX)
        return fl.render(colours, np.where(red, 0.45, 0.0), thick=32 // q)

    def frame(t: float, blend: float = 0.0):
        final = (wave_rgb[c] * (1 - blend) + blob_rgb[c] * blend).astype(np.uint8)
        return iso.wave_frame(fl, red, when, final, t, thick=32 // q)

    frames = [(iso.label(plain(False), "a crop of the test image", "384 px, drawn at a quarter"), 40),
              (iso.label(plain(True), f"{n_pieces} pieces", "each tile finds the blob pieces inside it"), 55)]
    for k in range(7):
        p = iso.label(frame(k * 0.25), f"{n_seeds} seeds", "each one starts a wave")
        frames.append((p, 4 if k < 6 else 30))
    done = n_seeds
    for r in range(1, last + 1):
        done += claimed[r - 1]
        for k in range(7):
            p = iso.label(frame(r + k * 0.25), f"round {r}", f"{done} of {n_pieces} pieces claimed")
            frames.append((p, 4 if k < 6 else 16))
        if r == 2:
            poster_at = len(frames) - 1
    t = last + 6.0
    for i in range(1, 11):
        frames.append((iso.label(frame(t, i / 10), "where waves met", "the smallest label wins"), 3))
    frames.append((iso.label(frame(t, 1.0), f"{n_blobs} blobs", "one label each"), 85))
    write("iso_waves", frames, poster_at)


CLIPS = {"cpu_gpu": cpu_gpu, "blocks": blocks, "merge": merge, "tile": tile, "tiles_ab": tiles_ab,
         "waves": waves}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(CLIPS):
        CLIPS[name]()
