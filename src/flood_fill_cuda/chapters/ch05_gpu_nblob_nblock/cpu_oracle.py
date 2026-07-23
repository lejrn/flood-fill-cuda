"""CPU references for the seed-discovery stage.

Nobody hands this chapter a seed, so the oracle must also DEFINE what the
right answer is. The canonicalization rule, shared by both GPU variants
and every reference here:

    canonical label of a blob = the minimum linear index (x*height + y)
    over its pixels; the canonical seed = that pixel (the blob's
    lexicographic-min pixel, x-major then y).

Deterministic, a pure function of the image, and exactly what union-by-
atomicMin over pixel indices converges to — so CPU and both GPU variants
must agree bit-for-bit.

Candidate rule (the seed-merge variant's local scan): a red pixel is a
candidate iff none of its four LEX-PREDECESSOR neighbors — (x-1, y-1),
(x-1, y), (x-1, y+1), (x, y-1), the 8-neighbors that precede it in
x-major order — is red. Two properties the tests lean on:

- Every blob's lex-min pixel is a candidate (all four predecessors lie
  lexicographically before it, so none can be in the blob — nor red at
  all, since any red predecessor would be 8-adjacent, hence in the blob).
  Therefore min(candidate indices) == min(blob indices): the union-find
  root over candidates IS the canonical label, no extra pass needed.
- No two candidates are 8-adjacent (of any adjacent pair, the lex-smaller
  occupies one of the other's predecessor slots), so the candidate count
  is bounded by an independent set — scene-shaped, not blob-counted.

Two depth semantics, one per GPU variant, both EXACT (level-synchronous
BFS depth is race-free: timing decides who claims a pixel, never at
which level):

- cpu_fill_canonical: distance from THE canonical seed (ccl_fill).
- cpu_fill_from_candidates: distance from the NEAREST candidate
  (seed_merge — every candidate starts a wave at level 0). Blobs are
  disjoint, so global nearest-candidate == own-blob nearest-candidate.
"""

import numpy as np
from numba import njit

# Same 8-connectivity offsets as the GPU kernels (E SE S SW W NW N NE)
_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# Lex-predecessor offsets: the 8-neighbors at a smaller linear index
_PDX = np.array([-1, -1, -1, 0], dtype=np.int32)
_PDY = np.array([-1, 0, 1, -1], dtype=np.int32)


@njit(cache=True)
def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


@njit(cache=True)
def cpu_candidates(img):
    """int32 (width, height) mask: 1 where the candidate rule fires."""
    width, height = img.shape[0], img.shape[1]
    mask = np.zeros((width, height), dtype=np.int32)
    for x in range(width):
        for y in range(height):
            if not _is_red(img, x, y):
                continue
            found = False
            for d in range(4):
                nx = x + _PDX[d]
                ny = y + _PDY[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    found = True
                    break
            if not found:
                mask[x, y] = 1
    return mask


@njit(cache=True)
def cpu_label_components(img):
    """8-connectivity CCL with canonical labels.

    Returns (label, n_blobs): label is int32 (width, height), -1 on
    background, else the component's minimum linear index. The lex-order
    scan guarantees the first unlabeled red pixel of a component is its
    lex-min pixel, whose linear index is the component minimum.
    """
    width, height = img.shape[0], img.shape[1]
    label = np.full((width, height), -1, dtype=np.int32)
    queue = np.empty(width * height, dtype=np.int64)
    n_blobs = 0
    for x in range(width):
        for y in range(height):
            if not _is_red(img, x, y) or label[x, y] >= 0:
                continue
            n_blobs += 1
            root = x * height + y
            label[x, y] = root
            queue[0] = root
            front, rear = 0, 1
            while front < rear:
                pixel = queue[front]
                front += 1
                px = pixel // height
                py = pixel % height
                for d in range(8):
                    nx = px + _DX[d]
                    ny = py + _DY[d]
                    if (0 <= nx < width and 0 <= ny < height
                            and label[nx, ny] < 0 and _is_red(img, nx, ny)):
                        label[nx, ny] = root
                        queue[rear] = nx * height + ny
                        rear += 1
    return label, n_blobs


@njit(cache=True)
def _multisource_fill(img, seed_mask):
    """Level-synchronous BFS from every seed_mask pixel at level 0.

    Mirrors the GPU loop's level semantics exactly: seeds are pre-visited
    and pre-queued, depth is stamped at dequeue, levels counts loop
    iterations (== max depth + 1 on non-empty input, 0 on blank).
    Returns (visited, depth, levels, filled).
    """
    width, height = img.shape[0], img.shape[1]
    visited = np.zeros((width, height), dtype=np.int32)
    depth = np.full((width, height), -1, dtype=np.int32)
    queue = np.empty(width * height, dtype=np.int64)
    rear = 0
    for x in range(width):
        for y in range(height):
            if seed_mask[x, y] != 0:
                visited[x, y] = 1
                queue[rear] = x * height + y
                rear += 1
    front = 0
    level = 0
    while front < rear:
        level_end = rear
        for i in range(front, level_end):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            depth[x, y] = level
            for d in range(8):
                nx = x + _DX[d]
                ny = y + _DY[d]
                if (0 <= nx < width and 0 <= ny < height
                        and visited[nx, ny] == 0 and _is_red(img, nx, ny)):
                    visited[nx, ny] = 1
                    queue[rear] = nx * height + ny
                    rear += 1
        front = level_end
        level += 1
    return visited, depth, level, rear


@njit(cache=True)
def cpu_fill_canonical(img):
    """ccl_fill oracle: one seed per blob — its canonical (lex-min) pixel.

    Returns (visited, depth, label, levels, filled); depth is the exact
    BFS distance from the blob's canonical seed, levels the shared clock
    (== the slowest blob's level count).
    """
    width, height = img.shape[0], img.shape[1]
    label, _ = cpu_label_components(img)
    seed_mask = np.zeros((width, height), dtype=np.int32)
    for x in range(width):
        for y in range(height):
            if label[x, y] == x * height + y:
                seed_mask[x, y] = 1
    visited, depth, levels, filled = _multisource_fill(img, seed_mask)
    return visited, depth, label, levels, filled


@njit(cache=True)
def cpu_fill_from_candidates(img):
    """seed_merge oracle: every candidate starts a wave at level 0.

    Returns (visited, depth, label, levels, filled); depth is the exact
    BFS distance to the NEAREST candidate of the pixel's own blob, label
    the canonical map (identical to cpu_fill_canonical's — merging must
    erase any trace of which candidate got there first).
    """
    label, _ = cpu_label_components(img)
    seed_mask = cpu_candidates(img)
    visited, depth, levels, filled = _multisource_fill(img, seed_mask)
    return visited, depth, label, levels, filled
