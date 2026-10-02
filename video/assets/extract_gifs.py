"""Unpack the parent repo's wavefront GIFs into PNG frame sequences under assets/.

Usage (from video/):
    uv run assets/extract_gifs.py

Each GIF becomes assets/<name>/frame_NNN.png plus assets/<name>/meta.json
(frame count, native delay, size). assets/ is gitignored; rerun to rebuild.
An entry may be a dict {"path": ..., "crop": (x0, y0, x1, y1)} to keep only
a window of every frame (used for the right half of before_after).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageSequence

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "src" / "flood_fill_cuda" / "results"

GIFS = {
    # name: path under results/ (or {"path": ..., "crop": (x0, y0, x1, y1)})
    "ch00_cpu_square": "ch01_gpu_1blob_1block/wavefront/square256_cpu_order.gif",
    "ch01_square_1block": "ch01_gpu_1blob_1block/wavefront/square256_b1_t256.gif",
    "ch02_global_square": "ch02_gpu_1blob_2block/wavefront/global_square256.gif",
    "ch03_square_conn4": "ch03_gpu_1blob_nblock/wavefront/square256_b8_t32.gif",
    "ch03_square_conn8": "ch03_gpu_1blob_nblock/wavefront/square256_b8_t32_conn8.gif",
    "ch03_disk": "ch03_gpu_1blob_nblock/wavefront/disk512_b8_t64.gif",
    "ch04_asym_multisource": "ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_multisource.gif",
    "ch04_asym_sequential": "ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_sequential.gif",
    "ch05_u_prov": "ch05_gpu_nblob_nblock/wavefront/u192_merge_prov.gif",
    "ch05_u_final": "ch05_gpu_nblob_nblock/wavefront/u192_merge_final.gif",
    "ch05_input_blobs": "ch05_gpu_nblob_nblock/wavefront/input_blobs_final.gif",
    "ch06_before_after": "ch06_gpu_nblob_runs/figures/before_after.gif",
    # the recoloured half alone: the 1536x380 still is crop | 16 px divider | crop
    "ch06_after_half": {"path": "ch06_gpu_nblob_runs/figures/before_after.gif",
                        "crop": (776, 0, 1536, 380)},
}


def extract(name: str, spec) -> dict:
    rel = spec["path"] if isinstance(spec, dict) else spec
    crop = spec.get("crop") if isinstance(spec, dict) else None
    src = RESULTS / rel
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("frame_*.png"):
        old.unlink()
    with Image.open(src) as im:
        delays = []
        n = 0
        size = list(im.size)
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            rgba = frame.convert("RGBA")
            if crop:
                rgba = rgba.crop(crop)
                size = list(rgba.size)
            rgba.save(out / f"frame_{i:03d}.png")
            delays.append(frame.info.get("duration", 0))
            n = i + 1
        meta = {"source": rel, "frames": n, "size": size, "delay_ms": delays[0] if delays else 0}
        if crop:
            meta["crop"] = list(crop)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    for name, spec in GIFS.items():
        meta = extract(name, spec)
        print(f"{name:24s} {meta['frames']:3d} frames  {meta['size'][0]}x{meta['size'][1]}  {meta['delay_ms']} ms/frame")


if __name__ == "__main__":
    main()
