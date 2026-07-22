"""CPU references for the dual-blob stage.

The single-seed oracles are re-exported: 4-connectivity from ch01, 8-conn
from shared/.

The two-seed oracle needs NO new BFS: the blobs are disconnected, so a
pixel's distance to the nearest seed equals its distance to its own blob's
seed. Running the existing single-seed oracle once per seed on the ORIGINAL
image yields non-overlapping visited/depth maps that merge exactly into
what a correct multi-source labeled BFS must produce: combined levels =
max(levels_a, levels_b), filled = filled_a + filled_b, label = which run
reached the pixel. The disjointness assertion doubles as the scenes'
disconnection guard (it fails loudly if a "two blob" scene is secretly one
blob).
"""

import numpy as np

from ..ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
from ...shared.cpu_oracle import cpu_flood_fill_8


def cpu_flood_fill_two(img, seeds, connectivity=4):
    """Merged two-seed oracle over disconnected components.

    Returns (visited, depth, label, levels, filled, (levels_a, filled_a),
    (levels_b, filled_b)). label is int8: 0 for seeds[0]'s blob, 1 for
    seeds[1]'s, -1 unreached. Raises ValueError if the two seeds turn out
    to share a component (the scene is not two blobs).
    """
    fill = cpu_flood_fill_8 if connectivity == 8 else cpu_flood_fill
    (ax, ay), (bx, by) = seeds
    va, da, la, fa = fill(img, ax, ay)
    vb, db, lb, fb = fill(img, bx, by)
    if np.any((va == 1) & (vb == 1)):
        raise ValueError(
            "seeds share one connected component — the scene is not two "
            "disconnected blobs")
    visited = (va | vb).astype(np.int32)
    depth = np.where(va == 1, da, np.where(vb == 1, db, -1)).astype(np.int32)
    label = np.full(visited.shape, -1, dtype=np.int8)
    label[va == 1] = 0
    label[vb == 1] = 1
    return (visited, depth, label, max(la, lb), fa + fb,
            (la, fa), (lb, fb))
