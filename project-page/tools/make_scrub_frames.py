"""Frame sequences for the two wavefront scrub sliders on the project page.

Run from the repo root (the folder that holds src/ and project-page/):

    ${PYTHON:-python} project-page/tools/make_scrub_frames.py          # default
    ${PYTHON:-python} project-page/tools/make_scrub_frames.py --raw    # no colour lock

Needs PIL (with WebP) and numpy only. No GPU code, no ffmpeg. The inputs
are committed GIFs, so nothing comes from VIDEO_DIR.

A. The comb merge (main slider), from the ch05 wavefront GIFs
   comb96_merge_prov.gif and comb96_merge_final.gif (384x256 = a 96x64
   scene drawn at 4x). Each has 71 frames: frame i (0..69) shows every
   pixel with BFS depth <= i and flashes depth == i in a light tint, then
   frame 70 is the settled hold (same pixels as frame 69, flash removed,
   1.5 s). Writes, upscaled 2x with NEAREST to 768x512:

     static/scrub/comb/NNN.webp   prov frames 0..69, one per BFS level
     static/scrub/comb_start.webp prov frame 0: only the 24 candidates claimed
     static/scrub/comb_end.webp   final frame 70: the settled image, one label

   The settled prov frame 70 is dropped from the slider: it is not a BFS
   level, so keeping it would break "one step = one level".

   Colour lock. The prov GIF has 24 hues x 70 shades, more than a GIF
   palette holds, so PIL quantized every frame to its own adaptive
   palette. Pixels that are already filled shift colour from frame to
   frame (up to ~40 per channel), which shimmers on a scrub slider.
   The generator draws each filled pixel in one fixed colour once its
   flash is over, so by default this script takes every such pixel's
   colour from the settled frame 70 (one palette, the image the GIF
   holds on). Flash pixels and unfilled pixels come from frame i itself.
   Every colour still comes from an extracted GIF frame. --raw skips this.

B. The ch01 CPU vs GPU pair (one slider moves both), from
   square256_cpu_order.gif and square256_b1_t256.gif (512x512 = a 256x256
   scene at 2x, 97 frames). Frames 0..95 sample the run at i/95 of the way
   through: GPU frame i shows BFS depth <= round(i * 220 / 95), so 2 or 3
   levels per step; CPU frame i shows visits 0..round(i * 48399 / 95), so
   509 or 510 visits per step. Frame 96 is the settled hold (same pixels
   as frame 95, cursor, trail and flash removed); it adds no progress, so
   both sides drop it and keep 96 frames each. Kept at 512x512:

     static/scrub/ch01_cpu/NNN.webp
     static/scrub/ch01_gpu/NNN.webp

Frames come from PIL (im.seek(i); im.convert('RGB')), which composites
each GIF frame in full, never from raw GIF deltas. Output is lossless
WebP, method 6. The script asserts every assumption above, so a
regenerated GIF that breaks one fails loudly instead of shipping wrong
frames.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RESULTS = Path("src/flood_fill_cuda/results")
COMB_DIR = RESULTS / "ch05_gpu_nblob_nblock/wavefront"
CH01_DIR = RESULTS / "ch01_gpu_1blob_1block/wavefront"
OUT = Path("project-page/static/scrub")

FRAME_MS = 60
HOLD_MS = 1500
SIZE_BUDGET = 6 * 1024 * 1024  # bytes, all of static/scrub/

RED = (255, 0, 0)
CURSOR = (0x0B, 0x0F, 0x14)  # ch01 CPU cursor square

WEBP = dict(format="WEBP", lossless=True, quality=100, method=6, exact=True)


# ---- helpers -----------------------------------------------------------


def load_gif(path: Path) -> tuple[list[np.ndarray], list[int]]:
    """Every frame fully composited, as (H, W, 3) uint8, plus durations."""
    im = Image.open(path)
    frames, durations = [], []
    for i in range(im.n_frames):
        im.seek(i)
        durations.append(int(im.info.get("duration", 0)))
        frames.append(np.asarray(im.convert("RGB")).copy())
    return frames, durations


def is_colour(f: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    return (f[..., 0] == rgb[0]) & (f[..., 1] == rgb[1]) & (f[..., 2] == rgb[2])


def filled(f: np.ndarray) -> np.ndarray:
    """Blob pixels already reached: neither unfilled red nor white scene."""
    return ~is_colour(f, RED) & ~(f == 255).all(-1)


def first_filled(frames: list[np.ndarray]) -> np.ndarray:
    """Per pixel, the first frame index where it is filled (big if never)."""
    first = np.full(frames[0].shape[:2], len(frames), dtype=np.int32)
    for i in range(len(frames) - 1, -1, -1):
        first[filled(frames[i])] = i
    return first


def check(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"make_scrub_frames: check failed: {msg}")


def check_timing(name: str, frames, durations, n: int, size) -> None:
    check(len(frames) == n, f"{name}: {len(frames)} frames, expected {n}")
    check(frames[0].shape[1::-1] == size, f"{name}: size {frames[0].shape[1::-1]}, expected {size}")
    check(durations == [FRAME_MS] * (n - 1) + [HOLD_MS], f"{name}: unexpected frame durations")


def upscale(f: np.ndarray, k: int) -> Image.Image:
    im = Image.fromarray(f)
    if k == 1:
        return im
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


def save(f: np.ndarray, path: Path, k: int, written: list[Path]) -> None:
    img = upscale(f, k)
    img.save(path, **WEBP)
    back = np.asarray(Image.open(path).convert("RGB"))
    check(np.array_equal(back, np.asarray(img)), f"{path}: WebP round trip is not lossless")
    written.append(path)


def fresh_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.webp"):
        old.unlink()


# ---- A. comb merge -----------------------------------------------------


def make_comb(raw: bool, written: list[Path]) -> dict:
    prov, dp = load_gif(COMB_DIR / "comb96_merge_prov.gif")
    final, df = load_gif(COMB_DIR / "comb96_merge_final.gif")
    levels = len(prov) - 1  # last frame is the settled hold
    check_timing("comb prov", prov, dp, 71, (384, 256))
    check_timing("comb final", final, df, 71, (384, 256))

    # Frame-aligned: same unfilled-red and white pixels in every frame.
    for i in range(len(prov)):
        check(np.array_equal(is_colour(prov[i], RED), is_colour(final[i], RED)),
              f"comb frame {i}: prov and final cover different pixels")
        check(np.array_equal((prov[i] == 255).all(-1), (final[i] == 255).all(-1)),
              f"comb frame {i}: prov and final scenes differ")

    first = first_filled(final)
    counts = [int((first == i).sum()) // 16 for i in range(levels)]  # scene px per level
    check(all(c > 0 for c in counts), "comb: a BFS level adds no pixels")
    check(counts[0] == 24, f"comb: {counts[0]} candidates in frame 0, expected 24")
    check(np.array_equal(filled(prov[levels]), filled(prov[levels - 1])),
          "comb: settled frame covers different pixels than the last level")
    settled_diff = (final[levels] != final[levels - 1]).any(-1)
    check(np.array_equal(settled_diff, first == levels - 1),
          "comb final: settled frame changes more than the last level's flash")

    # The final GIF (one hue, few colours) was not quantized: no jitter.
    for i in range(1, len(final)):
        stable = first < i - 1
        check(np.array_equal(final[i][stable], final[i - 1][stable]),
              f"comb final: filled pixels change colour at frame {i}")

    settled = prov[levels]
    jitter = []
    fresh_dir(OUT / "comb")
    for i in range(levels):
        f = prov[i]
        stable = first < i
        if stable.any():
            jitter.append(int(np.abs(f[stable].astype(np.int16) - settled[stable]).max()))
        if not raw:
            f = f.copy()
            f[stable] = settled[stable]
        save(f, OUT / "comb" / f"{i:03d}.webp", 2, written)

    save(prov[0], OUT / "comb_start.webp", 2, written)
    save(final[levels], OUT / "comb_end.webp", 2, written)
    return {
        "frames": levels,
        "new_px_first_last": (counts[:3], counts[-3:]),
        "lock": "off (--raw)" if raw else "on",
        "max_shift_vs_gif": max(jitter) if jitter else 0,
    }


# ---- B. ch01 CPU vs GPU ------------------------------------------------


def make_ch01(written: list[Path]) -> dict:
    out = {}
    for side, stem in (("cpu", "square256_cpu_order"), ("gpu", "square256_b1_t256")):
        frames, durations = load_gif(CH01_DIR / f"{stem}.gif")
        check_timing(f"ch01 {side}", frames, durations, 97, (512, 512))
        last = len(frames) - 1  # settled hold
        # The hold adds no progress: frame 95 already has no unfilled red.
        check(not is_colour(frames[last - 1], RED).any(), f"ch01 {side}: frame {last - 1} still has red")
        check(not (frames[last] == frames[last - 1]).all(), f"ch01 {side}: hold is a pixel copy")
        # Filled pixels keep their colour once settled (no GIF quantization);
        # only the CPU cursor may pass over them.
        for i in range(1, last + 1):
            was = (frames[i - 1] == frames[last]).all(-1) & filled(frames[i - 1])
            now = (frames[i] == frames[last]).all(-1)
            if side == "cpu":
                now |= is_colour(frames[i], CURSOR)
            check(not (was & ~now).any(), f"ch01 {side}: settled pixels change at frame {i}")

        d = OUT / f"ch01_{side}"
        fresh_dir(d)
        for i in range(last):
            save(frames[i], d / f"{i:03d}.webp", 1, written)
        out[side] = last
        del frames
    check(out["cpu"] == out["gpu"], "ch01: CPU and GPU frame counts differ")

    # What one step means, from the generator's own formulas
    # (ch01 benchmarks/wavefront.py: thresholds = unique(round(linspace(0, top, 96)))).
    levels, visits = 221, 220 * 220
    tg = np.unique(np.linspace(0, levels - 1, 96).round().astype(int))
    tc = np.unique(np.linspace(0, visits - 1, 96).round().astype(int))
    check(len(tg) == len(tc) == out["cpu"], "ch01: threshold count does not match frame count")
    for key, t in (("gpu_step", tg), ("cpu_step", tc)):
        steps, n = np.unique(np.diff(t), return_counts=True)
        out[key] = {int(s): int(c) for s, c in zip(steps, n)}
    return out


# ---- main --------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", action="store_true",
                    help="write comb frames exactly as extracted, without the colour lock")
    args = ap.parse_args()

    if not (RESULTS.is_dir() and Path("project-page").is_dir()):
        sys.exit("make_scrub_frames: run from the repo root (it holds src/ and project-page/)")

    written: list[Path] = []
    comb = make_comb(args.raw, written)
    ch01 = make_ch01(written)

    sizes = [p.stat().st_size for p in written]
    seq = [p.stat().st_size for p in written if p.parent != OUT]
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"comb: {comb['frames']} frames (one BFS level each), colour lock {comb['lock']}, "
          f"max shift vs GIF frame {comb['max_shift_vs_gif']}")
    print(f"      scene px per level, first 3 / last 3: {comb['new_px_first_last']}")
    print(f"ch01: {ch01['cpu']} frames per side; GPU levels per step {ch01['gpu_step']}, "
          f"CPU visits per step {ch01['cpu_step']}")
    print(f"wrote {len(written)} files, {sum(sizes):,} B; median sequence frame {statistics.median(seq):,.0f} B")
    print(f"static/scrub total: {total:,} B (budget {SIZE_BUDGET:,} B)")
    check(total < SIZE_BUDGET, "static/scrub is over the size budget")


if __name__ == "__main__":
    main()
