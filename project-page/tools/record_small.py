"""Record the chapters' kernels on small scenes, for the page's 3D clips.

The page draws every pixel as a little box, so its scenes must be small
(about 36 pixels a side). The committed wavefront GIFs are 256 pixels a
side, far too fine for that. This script runs the same kernels on small
scenes and saves what they record: the level each pixel was filled at,
which block filled it, and the labels before and after the merge.

Nothing here is drawn or modelled. Every array comes from a kernel run on
this GPU (or, for the CPU order, from ch01's own deque BFS), and each GPU
depth map is checked against the CPU oracle before anything is written.

Outputs, next to this script (committed, so the clips rebuild without a
GPU):

    small_scenes.npz    the arrays, all indexed [x, y] like the kernels
    small_scenes.json   sizes, level counts, blocks that worked, checks

Run from the repository root (needs the GPU, about a minute of JIT):

    uv run python project-page/tools/record_small.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flood_fill_cuda.chapters.ch01_gpu_1blob_1block import flood_fill as ch01
from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.wavefront import cpu_visit_order
from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import flood_fill as ch03
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock import flood_fill as ch05
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components
from flood_fill_cuda.shared.cpu_oracle import cpu_flood_fill_8

HERE = Path(__file__).resolve().parent
RED = (255, 0, 0)
BLOCKS, TPB = 8, 32          # the grid of the committed ch03 wavefront GIFs


def image(mask: np.ndarray) -> np.ndarray:
    img = np.full(mask.shape + (3,), 255, np.uint8)
    img[mask] = RED
    return img


def blob_scene(n: int = 36):
    """One irregular blob: three discs joined, one bite taken out. The
    shape scales with n (36 is the base size)."""
    x, y = np.mgrid[0:n, 0:n]
    k = n / 36

    def disc(cx, cy, r):
        return (x - cx * k) ** 2 + (y - cy * k) ** 2 <= (r * k) ** 2

    mask = disc(18, 18, 11) | disc(10, 12, 6) | disc(26, 25, 6.5)
    mask &= ~disc(24, 10, 3.2)
    return image(mask), (n // 2, n // 2)


def u_scene():
    """A U: two arms joined by a bridge at the bottom. The two arm tips are
    the two candidates, and their waves meet in the bridge."""
    w, h = 36, 30
    mask = np.zeros((w, h), bool)
    mask[6:14, 4:26] = True          # left arm
    mask[22:30, 4:26] = True         # right arm
    mask[6:30, 20:26] = True         # the bridge
    return image(mask)


def main() -> None:
    out, facts, checks = {}, {}, {}

    img, (sx, sy) = blob_scene()
    visited, depth4, levels4, filled = cpu_flood_fill(img, sx, sy)
    _, depth8, levels8, _ = cpu_flood_fill_8(img, sx, sy)
    order, n = cpu_visit_order(img, sx, sy)
    out["blob_red"] = (img == RED).all(axis=2)
    out["blob_cpu_order"] = order.astype(np.int32)
    checks["cpu order covers the blob"] = bool(n == filled and ((order >= 0) == visited.astype(bool)).all())

    r1 = ch01.flood_fill(img, sx, sy, threads_per_block=256, variant="ring")
    out["blob_ch01_depth"] = r1.depth.astype(np.int32)
    checks["ch01 depth equals the oracle"] = bool((r1.depth == depth4).all())

    r3 = ch03.flood_fill(img, sx, sy, threads_per_block=TPB, blocks=BLOCKS, connectivity=4)
    out["blob_ch03_depth"] = r3.depth.astype(np.int32)
    out["blob_ch03_owner"] = r3.owner.astype(np.int16)
    checks["ch03 depth equals the oracle"] = bool((r3.depth == depth4).all())

    r8 = ch03.flood_fill(img, sx, sy, threads_per_block=TPB, blocks=BLOCKS, connectivity=8)
    out["blob_ch03_depth8"] = r8.depth.astype(np.int32)
    out["blob_ch03_owner8"] = r8.owner.astype(np.int16)
    checks["ch03 8-neighbour depth equals the oracle"] = bool((r8.depth == depth8).all())

    facts["blob"] = {
        "size": list(img.shape[:2]), "seed": [sx, sy], "red_pixels": int(filled),
        "levels_4": int(levels4), "levels_8": int(levels8),
        "ch01": {"threads_per_block": 256, "levels": int(r1.levels)},
        "ch03": {"blocks": BLOCKS, "threads_per_block": TPB, "levels": int(r3.levels),
                 "blocks_that_worked": int((r3.processed_per_block > 0).sum()),
                 "pixels_per_block": [int(v) for v in r3.processed_per_block]},
        "ch03_conn8": {"levels": int(r8.levels),
                       "blocks_that_worked": int((r8.processed_per_block > 0).sum())},
    }

    # The same blob at 64 pixels a side, for the many-block clip: on the
    # 36-pixel blob a ring never outgrows one or two blocks of 32 threads.
    big, (bx, by) = blob_scene(64)
    _, big4, big_levels4, big_filled = cpu_flood_fill(big, bx, by)
    _, big8, big_levels8, _ = cpu_flood_fill_8(big, bx, by)
    b3 = ch03.flood_fill(big, bx, by, threads_per_block=TPB, blocks=BLOCKS, connectivity=4)
    b8 = ch03.flood_fill(big, bx, by, threads_per_block=TPB, blocks=BLOCKS, connectivity=8)
    out["big_red"] = (big == RED).all(axis=2)
    out["big_ch03_depth"] = b3.depth.astype(np.int32)
    out["big_ch03_owner"] = b3.owner.astype(np.int16)
    out["big_ch03_depth8"] = b8.depth.astype(np.int32)
    out["big_ch03_owner8"] = b8.owner.astype(np.int16)
    checks["ch03 depth on the big blob equals the oracle"] = bool((b3.depth == big4).all())
    checks["ch03 8-neighbour depth on the big blob equals the oracle"] = bool((b8.depth == big8).all())
    facts["big"] = {
        "size": list(big.shape[:2]), "seed": [bx, by], "red_pixels": int(big_filled),
        "levels_4": int(big_levels4), "levels_8": int(big_levels8),
        "ch03": {"blocks": BLOCKS, "threads_per_block": TPB,
                 "blocks_that_worked": int((b3.processed_per_block > 0).sum()),
                 "pixels_per_block": [int(v) for v in b3.processed_per_block]},
        "ch03_conn8": {"blocks_that_worked": int((b8.processed_per_block > 0).sum()),
                       "pixels_per_block": [int(v) for v in b8.processed_per_block]},
    }

    u = u_scene()
    r5 = ch05.flood_fill(u, variant="seed_merge")
    oracle_label, oracle_blobs = cpu_label_components(u)
    out["u_red"] = (u == RED).all(axis=2)
    out["u_depth"] = r5.depth.astype(np.int32)
    out["u_label"] = r5.label.astype(np.int32)
    out["u_prov_label"] = r5.prov_label.astype(np.int32)
    checks["ch05 labels equal the oracle"] = bool((r5.label == oracle_label).all()
                                                  and r5.n_blobs == oracle_blobs)
    facts["u"] = {"size": list(u.shape[:2]), "red_pixels": int(out["u_red"].sum()),
                  "blobs": int(r5.n_blobs), "candidates": int(r5.candidates),
                  "unions": int(r5.union_done), "levels": int(r5.levels),
                  "provisional_labels": int(len(np.unique(r5.prov_label[out["u_red"]])))}

    if not all(checks.values()):
        raise SystemExit(f"a check failed, nothing written: {checks}")
    np.savez_compressed(HERE / "small_scenes.npz", **out)
    (HERE / "small_scenes.json").write_text(json.dumps({"facts": facts, "checks": checks}, indent=2) + "\n")
    print(json.dumps(facts, indent=1))
    print("checks:", checks)


if __name__ == "__main__":
    main()
