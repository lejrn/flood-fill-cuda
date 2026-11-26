"""The tile clips of the project page: chapters 7 and 8, from recorded data.

Everything drawn here comes from video/assets/tiles/ (tracked in git): the
arrays the chapters' own GPU kernels recorded, checked against the CPU
oracles (see the README in that folder).

    ch07_tiles_*.npz   sq_first_open: the round each 32 x 32 tile of the
                       8000 x 8000 square was first visited
    ch08_waves_*.npz   a 384 px crop of input_blobs.png: its pieces, its
                       seeds, the round and the wave that claimed each piece,
                       the final labels

Outputs, under project-page/static/:

    videos/tiles_ab.mp4, images/tiles_ab.webp
        Two panes on one step counter. Right: the tile BFS, each tile
        coloured by the round it was first visited (recorded). Left: a pixel
        BFS after the same number of steps. The left pane is not a recording:
        on a solid square with 8 neighbours, level k is exactly the ring at
        distance k from the seed, so it is drawn from that rule. The square
        needs 4,001 such levels (the oracle's count, in the JSON).
    videos/waves.mp4, images/waves_poster.webp
        The crop: tiles, pieces, seeds, then one frame group per round, each
        piece in the colour of the wave that claimed it, then one colour per
        blob. Which wave claims a piece is a race: this is one recorded run.
    videos/carousel/tiles.mp4, images/carousel/tiles.webp
    videos/carousel/waves.mp4, images/carousel/waves.webp
        The same two, without the text strip, for the carousel.

Colours follow the other clips: red is unfilled, light is early, dark is
late, and separate waves take hues a golden angle apart.

Run from the repository root:

    FFMPEG=video/.venv/bin/ffmpeg .venv/bin/python project-page/tools/make_tile_clips.py
"""

from __future__ import annotations

import colorsys
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE = Path(__file__).resolve().parents[1]
REPO = PAGE.parent
TILES = REPO / "video" / "assets" / "tiles"
VID = PAGE / "static" / "videos"
IMG = PAGE / "static" / "images"
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FPS = 30

WHITE = np.array([255, 255, 255], np.uint8)
RED = np.array([255, 0, 0], np.uint8)
INK = (54, 54, 54)
MUTED = (107, 107, 107)
GRID = (150, 150, 150)
BLUE_HUE = 0.592            # the one-owner hue of the chapter 1 clips
PIECE_REDS = np.array([[255, 0, 0], [196, 0, 0], [255, 84, 84], [150, 0, 0]], np.uint8)
STRIP = 44                  # text strip under a pane, px


def newest(pattern: str) -> Path:
    hits = sorted(glob.glob(str(TILES / pattern)))
    if not hits:
        sys.exit(f"no {pattern} in {TILES}")
    return Path(hits[-1])


def find_font(names: list[str]) -> str:
    roots = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype"]
    for root in roots:
        for name in names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path
    sys.exit(f"none of {names} found under /usr/share/fonts/truetype")


FONT = find_font(["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf"])
FONT_BOLD = find_font(["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf"])


def depth_ramp(n: int, hue: float = BLUE_HUE) -> np.ndarray:
    """n colours, light (early) to dark (late)."""
    out = np.zeros((max(n, 1), 3), np.uint8)
    for i in range(max(n, 1)):
        light = 0.74 - 0.34 * (i / max(n - 1, 1))
        out[i] = [round(255 * c) for c in colorsys.hls_to_rgb(hue, light, 0.62)]
    return out


def wave_colour(i: int) -> np.ndarray:
    """A hue per wave, a golden angle apart; red's neighbourhood is skipped."""
    hue = 0.10 + 0.82 * ((i * 0.6180339887) % 1.0)
    light = (0.50, 0.62, 0.42)[i % 3]
    return np.array([round(255 * c) for c in colorsys.hls_to_rgb(hue, light, 0.66)], np.uint8)


def with_strip(pane: np.ndarray, parts: list[tuple[str, bool]]) -> np.ndarray:
    """The pane with a white text strip under it; parts are (text, bold)."""
    h, w, _ = pane.shape
    canvas = Image.new("RGB", (w, h + STRIP), (255, 255, 255))
    canvas.paste(Image.fromarray(pane), (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = 10
    for text, bold in parts:
        font = ImageFont.truetype(FONT_BOLD if bold else FONT, 22)
        draw.text((x, h + 8), text, fill=INK if bold else MUTED, font=font)
        x += round(draw.textlength(text, font=font)) + 10
    return np.asarray(canvas)


def encode(frames: list[tuple[np.ndarray, int]], out: Path, crf: int = 20) -> None:
    """frames: (image, how many video frames to hold it)."""
    h, w, _ = frames[0][0].shape
    assert w % 2 == 0 and h % 2 == 0, (w, h)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-crf", str(crf), "-movflags", "+faststart",
           "-map_metadata", "-1", "-an", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for image, hold in frames:
        data = np.ascontiguousarray(image).tobytes()
        for _ in range(hold):
            proc.stdin.write(data)
        n += hold
    proc.stdin.close()
    if proc.wait():
        sys.exit(f"ffmpeg failed on {out}")
    print(f"{out.relative_to(REPO)}: {w}x{h}, {n} frames = {n / FPS:.2f} s, "
          f"{out.stat().st_size:,} bytes")


def poster(image: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out, lossless=True, quality=100, method=6)
    print(f"{out.relative_to(REPO)}: {image.shape[1]}x{image.shape[0]}, {out.stat().st_size:,} bytes")


# ------------------------------------------------------------- chapter 7

def square_panes(first_open: np.ndarray, levels: int, cell: int, side: int):
    """Per step k: (pixel-BFS pane, tile-BFS pane), both side x side."""
    n = first_open.shape[0]                       # 250 tiles a side
    size = n * cell
    pad = (side - size) // 2
    rounds = int(first_open.max()) + 1
    ramp = depth_ramp(rounds)
    px_per_pixel = size / (n * 32)                # pane px per image px
    centre = pad + size / 2

    tiles_rgb = np.repeat(np.repeat(first_open.T, cell, axis=0), cell, axis=1)
    level_ramp = depth_ramp(levels)

    def tile_pane(k: int) -> np.ndarray:
        pane = np.full((side, side, 3), 255, np.uint8)
        inner = np.where((tiles_rgb <= k)[..., None], ramp[np.minimum(tiles_rgb, rounds - 1)], RED)
        pane[pad:pad + size, pad:pad + size] = inner
        return pane

    def pixel_pane(k: int) -> np.ndarray:
        pane = np.full((side, side, 3), 255, np.uint8)
        pane[pad:pad + size, pad:pad + size] = RED
        # level k of an 8-neighbour BFS on a solid square: every pixel within
        # Chebyshev distance k of the seed. (2k + 1) image pixels a side.
        half = (k + 0.5) * px_per_pixel
        lo, hi = round(centre - half), max(round(centre + half), round(centre - half) + 1)
        pane[lo:hi, lo:hi] = level_ramp[min(k, levels - 1)]
        return pane

    return rounds, pixel_pane, tile_pane


def chapter7() -> None:
    npz = newest("ch07_tiles_*.npz")
    meta = json.loads(npz.with_suffix(".json").read_text())
    levels = int(meta["square"]["oracle_levels"])
    first_open = np.load(npz)["sq_first_open"]
    side, gap = 768, 16
    rounds, pixel_pane, tile_pane = square_panes(first_open, levels, cell=3, side=side)
    total_rounds = int(meta["square"]["chapter_kernel"]["rounds"])
    print(f"ch07: {first_open.shape} tiles, {rounds} rounds open new tiles, "
          f"{total_rounds} rounds in all, {levels:,} pixel levels")

    def ab(k: int, done: bool = False) -> np.ndarray:
        covered = (2 * k + 1) ** 2 / (first_open.shape[0] * 32) ** 2
        left = with_strip(pixel_pane(k), [("pixel BFS", True),
                                          (f"step {k + 1:,} of {levels:,}", False),
                                          (f"{covered:.2%} filled", False)])
        tail = "done" if done else f"round {k + 1} of {total_rounds}"
        right = with_strip(tile_pane(k), [("tile BFS", True), (tail, False)])
        spacer = np.full((side + STRIP, gap, 3), 255, np.uint8)
        return np.concatenate([left, spacer, right], axis=1)

    frames = [(ab(0), 20)]
    frames += [(ab(k), 3) for k in range(1, rounds)]
    frames += [(ab(rounds - 1, done=True), 80)]
    encode(frames, VID / "tiles_ab.mp4")
    poster(ab(rounds * 2 // 3), IMG / "tiles_ab.webp")

    # carousel: the tile pane alone, 2 px a tile
    _, _, small = square_panes(first_open, levels, cell=2, side=512)
    frames = [(small(0), 20)] + [(small(k), 2) for k in range(1, rounds)] + [(small(rounds - 1), 70)]
    encode(frames, VID / "carousel" / "tiles.mp4")
    poster(small(rounds * 2 // 3), IMG / "carousel" / "tiles.webp")


# ------------------------------------------------------------- chapter 8

def chapter8() -> None:
    npz = newest("ch08_waves_*.npz")
    meta = json.loads(npz.with_suffix(".json").read_text())
    z = np.load(npz)
    red = z["crop_red"].T                      # [row, col] = PNG orientation
    comp = z["crop_comp"].T
    piece = z["crop_piece"].T
    is_seed = z["crop_comp_seed"]
    claim = z["crop_comp_round"]
    wave = z["crop_comp_wave"]
    label = z["crop_comp_label"]
    anchor = z["crop_comp_anchor"]
    n_pieces = len(is_seed)
    n_seeds = int(is_seed.sum())
    blobs = np.unique(label)
    last = int(claim.max())
    claimed = [int((claim == r).sum()) for r in range(1, last + 1)]
    assert red.shape == (384, 384) and (comp >= 0).sum() == red.sum()
    assert (claim >= 0).all(), "an interior piece: give it a colour rule"
    assert n_seeds + sum(claimed) == n_pieces
    print(f"ch08: {n_pieces} pieces, {n_seeds} seeds, claimed per round {claimed}, "
          f"{len(blobs)} blobs (JSON: {meta['crop'].get('blobs', '?')})")

    scale = 2
    side = red.shape[0] * scale
    seed_ids = np.flatnonzero(is_seed)
    seed_rank = {int(k): i for i, k in enumerate(seed_ids)}
    wave_rgb = np.stack([wave_colour(seed_rank[int(w)]) for w in wave])      # per piece
    # the final colour of a blob: the wave of the piece that holds the blob's
    # smallest pixel (its anchor is the label), "the smallest label wins"
    owner = {int(l): int(np.flatnonzero((label == l) & (anchor == l))[0]) for l in blobs}
    final_rgb = np.stack([wave_rgb[owner[int(l)]] for l in label])

    def up(image: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    def grid(image: np.ndarray) -> np.ndarray:
        out = image.copy()
        step = 32 * scale
        for i in range(step, side, step):
            out[i, :] = GRID
            out[:, i] = GRID
        return out

    def paint(per_piece: np.ndarray | None, shown: np.ndarray | None, pieces: bool) -> np.ndarray:
        """Red crop; pieces in alternating reds; shown pieces in per_piece colours."""
        image = np.full(red.shape + (3,), 255, np.uint8)
        image[red] = PIECE_REDS[piece[red] % len(PIECE_REDS)] if pieces else RED
        if per_piece is not None:
            on = red & shown[np.maximum(comp, 0)]
            image[on] = per_piece[comp[on]]
        return up(image)

    plain = paint(None, None, pieces=False)
    cut = grid(paint(None, None, pieces=True))
    steps = [(plain, 36, [("a 384 px crop", True), ("of the 81 Mpx frame", False)]),
             (cut, 50, [(f"{n_pieces} pieces", True), ("a blob, cut by 32 x 32 tiles", False)]),
             (grid(paint(wave_rgb, is_seed, True)), 46,
              [(f"{n_seeds} seeds", True), ("each starts a wave", False)])]
    done = 0
    for r in range(1, last + 1):
        done += claimed[r - 1]
        steps.append((grid(paint(wave_rgb, claim <= r, True)), 30,
                      [(f"round {r}", True), (f"{n_seeds + done} of {n_pieces} pieces claimed", False)]))
    steps.append((grid(paint(final_rgb, claim <= last, True)), 30,
                  [(f"{len(blobs)} blobs", True), ("where waves met, the smallest label won", False)]))
    steps.append((paint(final_rgb, claim <= last, False), 80,
                  [(f"{len(blobs)} blobs", True), ("where waves met, the smallest label won", False)]))

    encode([(with_strip(image, text), hold) for image, hold, text in steps], VID / "waves.mp4")
    poster(with_strip(steps[4][0], steps[4][2]), IMG / "waves_poster.webp")
    encode([(image, hold) for image, hold, _ in steps], VID / "carousel" / "waves.mp4")
    poster(steps[4][0], IMG / "carousel" / "waves.webp")


if __name__ == "__main__":
    chapter7()
    chapter8()
