"""Single-blob scenes built to find where the tile BFS degrades.

From the chapter's perf-adversary review. Layout as everywhere in the
repo: img is (W, H, 3) uint8, img[x, y], x first (a memory row, a lane
inside a tile), y contiguous (a bit inside a mask word). Red is exactly
(255, 0, 0) on white. Every builder returns (img, seed_x, seed_y) with
the seed on red, and every red pixel of these scenes belongs to the
seeded blob (so ch06's all-blob CCL changes exactly the fill set).

  zz_serpentine   one path of zigzags ON the x tile borders: inside a
                  tile the path is isolated pixels, so every pixel step
                  is a tile-border export and rounds == levels, one tile
                  per round (the narrow-front worst case)
  zz_lattice      a zigzag on EVERY tile border: rounds == levels again,
                  but with a front of ~130 tiles per round (the
                  wide-front worst case)
  maze            a perfect maze with 1-px corridors: many revisits and
                  closure iterations per tile (tortuous but realistic)
  thin_h32_blob   W = 524288, H = 32: one mask word per row, the shape
                  that starved ch06's pack/paint launch
  tiny_blob       a 5,000-px disk in a 64 Mpx image: the separate pack
                  reads and paint scans the whole image for it (the
                  fused pack reads the ~16 tiles the disk covers)
"""

import numpy as np
from numba import njit

RED = np.array([255, 0, 0], dtype=np.uint8)


def _white(w, h):
    return np.full((w, h, 3), 255, dtype=np.uint8)


# A boundary zigzag along y sits on the x tile border between rows 32k+31
# (lane 31 of tile k) and 32k+32 (lane 0 of tile k+1): pixel
# (32k+31 + (y % 2), y). Consecutive pixels are diagonal neighbours
# (8-conn); pixels two apart share a row with a white gap between them.

def zz_serpentine_scene(width=1024, height=1024):
    """Strand k on x border 32k+31|32k+32 for y in [2, height-3]; strands
    k and k+1 joined by a run along x (inside tile k+1) at y = height-2
    (k even) or y = 1 (k odd). Seed at strand 0's free end. One path,
    8-connected only (4-conn fills just the seed)."""
    img = _white(width, height)
    n_strands = width // 32 - 1
    ys = np.arange(2, height - 2)
    for k in range(n_strands):
        img[32 * k + 31 + (ys % 2), ys] = RED
    for k in range(n_strands - 1):
        y = height - 2 if k % 2 == 0 else 1
        img[32 * k + 32: 32 * k + 64, y] = RED
    return img, 31, 2


def zz_lattice_scene(width=4096, height=4096):
    """Zigzags along y on every x border and along x on every y border,
    meeting at tile corners. The front is a diamond of tiles that each
    advance one pixel per round."""
    img = _white(width, height)
    ys = np.arange(height)
    for k in range(width // 32 - 1):
        img[32 * k + 31 + (ys % 2), ys] = RED
    xs = np.arange(width)
    for j in range(height // 32 - 1):
        img[xs, 32 * j + 31 + (xs % 2)] = RED
    return img, 32 * (width // 64 - 1) + 31, (height // 2) & ~1


@njit(cache=False)
def _maze_passages(ncx, ncy, seed):
    """Recursive backtracker on an ncx x ncy cell grid (deterministic for
    a seed). open_x[i, j]: cells (i, j)-(i+1, j) joined; open_y likewise
    along y."""
    np.random.seed(seed)
    seen = np.zeros((ncx, ncy), np.uint8)
    open_x = np.zeros((ncx, ncy), np.uint8)
    open_y = np.zeros((ncx, ncy), np.uint8)
    stack = np.empty((ncx * ncy, 2), np.int32)
    nbr = np.empty(4, np.int32)
    stack[0, 0] = 0
    stack[0, 1] = 0
    sp = 1
    seen[0, 0] = 1
    while sp > 0:
        cx = stack[sp - 1, 0]
        cy = stack[sp - 1, 1]
        n = 0
        if cx > 0 and seen[cx - 1, cy] == 0:
            nbr[n] = 0
            n += 1
        if cx < ncx - 1 and seen[cx + 1, cy] == 0:
            nbr[n] = 1
            n += 1
        if cy > 0 and seen[cx, cy - 1] == 0:
            nbr[n] = 2
            n += 1
        if cy < ncy - 1 and seen[cx, cy + 1] == 0:
            nbr[n] = 3
            n += 1
        if n == 0:
            sp -= 1
            continue
        d = nbr[np.random.randint(n)]
        if d == 0:
            open_x[cx - 1, cy] = 1
            nx, ny = cx - 1, cy
        elif d == 1:
            open_x[cx, cy] = 1
            nx, ny = cx + 1, cy
        elif d == 2:
            open_y[cx, cy - 1] = 1
            nx, ny = cx, cy - 1
        else:
            open_y[cx, cy] = 1
            nx, ny = cx, cy + 1
        seen[nx, ny] = 1
        stack[sp, 0] = nx
        stack[sp, 1] = ny
        sp += 1
    return open_x, open_y


def maze_scene(cells=512, rng_seed=7):
    """Perfect maze, 1-px corridors and 1-px walls: cell (i, j) is pixel
    (2i, 2j), passages at the odd coordinates. 8-conn adds no leaks (two
    diagonal corridor pixels always share a cell). Seed: the centre."""
    ox, oy = _maze_passages(cells, cells, rng_seed)
    n = 2 * cells - 1
    img = _white(n, n)
    img[0::2, 0::2] = RED
    ii, jj = np.nonzero(ox[:-1, :])
    img[2 * ii + 1, 2 * jj] = RED
    ii, jj = np.nonzero(oy[:, :-1])
    img[2 * ii, 2 * jj + 1] = RED
    c = 2 * (cells // 2)
    return img, c, c


def thin_h32_blob_scene(width=1 << 19, side=32):
    """H = 32 (one mask word per row), W = 524288: 16.8 Mpx with a 32x32
    blob at the centre. The fill is trivial; pack and paint are the
    bill."""
    img = _white(width, 32)
    c = width // 2
    img[c - side // 2: c + side // 2, :] = RED
    return img, c, 16


def tiny_blob_scene(width=8000, height=8000, radius=40, cx=3010, cy=5003):
    """A red disk of `radius` (5,025 px at 40) centred at (cx, cy) in a
    white width x height image (64 Mpx by default), off the tile grid.
    Seed: the centre."""
    img = _white(width, height)
    xs = np.arange(cx - radius, cx + radius + 1)
    ys = np.arange(cy - radius, cy + radius + 1)
    inside = ((xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2
              <= radius * radius)
    sub = img[cx - radius: cx + radius + 1, cy - radius: cy + radius + 1]
    sub[inside] = RED
    return img, cx, cy
