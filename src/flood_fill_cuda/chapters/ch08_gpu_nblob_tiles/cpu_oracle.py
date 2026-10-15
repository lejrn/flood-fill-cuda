"""CPU oracles for ch08: canonical labels at 8- and 4-connectivity.

The 8-connectivity oracle is ch05's `cpu_label_components`, re-exported
unchanged: ch05, ch06 and ch08 are all judged by the same function.

ch06 is 8-connected only and the repo had no 4-connectivity many-blob
oracle, so `cpu_label` ports the scout's `global_ccl` (a BFS per blob in
lex order) and returns the same canonical form for either connectivity:
label = the blob's minimum linear index x*height + y, -1 off-blob. The
tests check it against `cpu_label_components` at 8-conn on every scene
before trusting it at 4-conn.
"""

import numpy as np
from numba import njit

from ..ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components  # noqa: F401


def red_mask(img):
    """(W, H) bool: pixels exactly (255, 0, 0)."""
    return ((img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0))


@njit(cache=True)
def _label_red(red, conn8):
    width, height = red.shape
    lab = np.full((width, height), -1, np.int32)
    n_red = 0
    for x in range(width):
        for y in range(height):
            if red[x, y]:
                n_red += 1
    q = np.empty(max(n_red, 1), np.int32)
    n_blobs = 0
    for x in range(width):
        for y in range(height):
            if not red[x, y] or lab[x, y] >= 0:
                continue
            # the first pixel met in x-major order is the blob's min lin
            canon = x * height + y
            lab[x, y] = canon
            q[0] = canon
            head = 0
            tail = 1
            while head < tail:
                p = q[head]
                head += 1
                px = p // height
                py = p - px * height
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        if not conn8 and dx != 0 and dy != 0:
                            continue
                        nx = px + dx
                        ny = py + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if red[nx, ny] and lab[nx, ny] < 0:
                                lab[nx, ny] = canon
                                q[tail] = nx * height + ny
                                tail += 1
            n_blobs += 1
    return lab, n_blobs


def cpu_label(img, connectivity=8):
    """(label, n_blobs) at 4- or 8-connectivity, canonical labels."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    lab, n = _label_red(np.ascontiguousarray(red_mask(img)),
                        connectivity == 8)
    return lab, int(n)
