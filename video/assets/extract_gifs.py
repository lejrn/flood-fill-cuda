"""Unpack the parent repo's wavefront GIFs into PNG frame sequences under assets/.

Usage (from video/):
    uv run assets/extract_gifs.py

Each GIF becomes assets/<name>/frame_NNN.png plus assets/<name>/meta.json
(frame count, native delay, size). assets/ is gitignored; rerun to rebuild.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageSequence

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "src" / "flood_fill_cuda" / "results"

GIFS = {
    # name: path under results/
    "ch03_square_conn4": "ch03_gpu_1blob_nblock/wavefront/square256_b8_t32.gif",
    "ch03_square_conn8": "ch03_gpu_1blob_nblock/wavefront/square256_b8_t32_conn8.gif",
    "ch03_disk": "ch03_gpu_1blob_nblock/wavefront/disk512_b8_t64.gif",
    "ch04_asym_multisource": "ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_multisource.gif",
    "ch04_asym_sequential": "ch04_gpu_2blob_nblock/wavefront/asym384_b8_t32_sequential.gif",
    "ch05_u_prov": "ch05_gpu_nblob_nblock/wavefront/u192_merge_prov.gif",
    "ch05_u_final": "ch05_gpu_nblob_nblock/wavefront/u192_merge_final.gif",
    "ch05_input_blobs": "ch05_gpu_nblob_nblock/wavefront/input_blobs_final.gif",
    "ch06_before_after": "ch06_gpu_nblob_runs/figures/before_after.gif",
}


def extract(name: str, rel: str) -> dict:
    src = RESULTS / rel
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("frame_*.png"):
        old.unlink()
    with Image.open(src) as im:
        delays = []
        n = 0
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            frame.convert("RGBA").save(out / f"frame_{i:03d}.png")
            delays.append(frame.info.get("duration", 0))
            n = i + 1
        meta = {"source": rel, "frames": n, "size": list(im.size), "delay_ms": delays[0] if delays else 0}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    for name, rel in GIFS.items():
        meta = extract(name, rel)
        print(f"{name:24s} {meta['frames']:3d} frames  {meta['size'][0]}x{meta['size'][1]}  {meta['delay_ms']} ms/frame")


if __name__ == "__main__":
    main()
