"""The whole frame's red mask, every pixel: assets/tiles/input_blobs_red.npz.

r13 pulls back from r12's 384 px crop to the whole 9000 x 9000 frame. The
900 px frame pictures hold a tenth of the frame's side, so a zoom drawn
from them starts as a mosaic of squares 25 px wide (at 1080p). r13 draws
its zoom from this mask instead: sharp at every scale, and at the crop's
window it is the crop's own pixels.

Usage (from video/):
    uv run assets/make_frame_mask.py [--image PATH]

PATH defaults to the chapters' input, images/input/input_blobs.png at the
repo root (git-ignored there, so it is read once and the mask committed).

The script checks the image's md5 (the act-3 runs' input), that it holds
two colours only (pure red and white), and that ch08's crop
(crop_red at its origin) equals its window of the mask. It keeps the red
pixels (the chapters' red test), transposes them to the kernels' [x, y]
(state.picture's orientation) and packs 8 pixels a byte along y:

    red_bits    (9000, 1125) uint8   np.packbits(red_xy, axis=1)
    shape       (2,) int64           (9000, 9000)
    source_md5  str                  the image's md5
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TILES = HERE / "tiles"
OUT = TILES / "input_blobs_red.npz"
WV = TILES / "ch08_waves_20261017T160051Z"
DEFAULT_IMAGE = HERE.parents[1] / "images" / "input" / "input_blobs.png"
MD5 = "47639eb37f973b5092ec2c4db86f45a4"     # assets/tiles/README.md: the act-3 runs' input


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    args = ap.parse_args()

    md5 = hashlib.md5(args.image.read_bytes()).hexdigest()
    if md5 != MD5:
        raise SystemExit(f"md5 {md5} is not the act-3 input's {MD5}")
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(args.image).convert("RGB"))
    colours = np.unique(img.reshape(-1, 3), axis=0)
    if colours.tolist() != [[255, 0, 0], [255, 255, 255]]:
        raise SystemExit(f"expected pure red and white only, got {len(colours)} colours")
    red = (img[..., 0] > 200) & (img[..., 1] < 90) & (img[..., 2] < 90)
    red_xy = np.ascontiguousarray(red.T)                     # [x, y]: x the PNG column

    crop = json.loads(WV.with_suffix(".json").read_text(encoding="utf-8"))["crop"]
    (x0, y0), side = crop["origin"], int(crop["side"])
    crop_red = np.load(WV.with_suffix(".npz"))["crop_red"]
    if not np.array_equal(red_xy[x0:x0 + side, y0:y0 + side], crop_red):
        raise SystemExit("ch08's crop is not its window of the mask")

    bits = np.packbits(red_xy, axis=1)
    np.savez_compressed(OUT, red_bits=bits, shape=np.array(red_xy.shape, np.int64), source_md5=np.array(md5))
    print(f"{OUT.name}: {int(red_xy.sum()):,} red of {red_xy.size:,} px, "
          f"{OUT.stat().st_size / 1e6:.2f} MB; ch08's crop at ({x0}, {y0}) matches")


if __name__ == "__main__":
    main()
