"""Scenes built to break a many-blob tile labeller.

Every builder returns a (W, H, 3) uint8 image, img[x, y], pure red
(255, 0, 0) on white. Tiles are ch07's: rows x0 .. x0+31 (lanes) times
one mask word y0 .. y0+31 (bits). ch05's and ch07's scenes are reused
by the tests and the benchmark; these are the cases new to ch08.

  corner_joined     two blobs that touch only through a tile-corner
                    pixel pair (one diagonal pair per call)
  all_eight         one comp with a red neighbour in all 8 tiles around
  frame_comb        one blob that enters a tile n times through its
                    frame: n ring comps in that tile (n up to 60 at
                    8-conn, 62 at 4-conn: the ordinal bound)
  dot_lattice       256 isolated pixels in a tile (8-conn's comp max);
  checker_tile      512 isolated pixels at 4-conn (4-conn's max)
  rmin_trap         the canonical comp is not a seed: its tile has a
                    left neighbour the blob reaches below the canonical
                    pixel, so flatten's min-anchor step must fix it
  lexmin_multi      the blob's lex-min tile holds two of its comps
  spiral            a square spiral: many comps of one blob per tile
  diag_snake        a lattice-avoiding chain through tile corners:
                    one seed, one comp per tile, rounds = its length
  tile_corners      single pixels and 2x2 blocks on tile corners, and
                    the four image corners
  thin_strip        a long strip of noise, 32 px across
fuzz builders (seeded): noise, rects_and_disks, walks, border_zigzags,
stitched_tiles.
"""

import numpy as np

RED = np.array([255, 0, 0], dtype=np.uint8)


def _white(w, h):
    return np.full((w, h, 3), 255, dtype=np.uint8)


def from_mask(red):
    img = _white(*red.shape)
    img[red.astype(bool)] = RED
    return img


# -------------------------------------------------------- adversarial

def corner_joined_scene(kind="main", solid=True):
    """96x96. kind "main": A in tile (0,0) holds (31,31), B in tile (1,1)
    holds (32,32): A's DR meets B's UL. kind "anti": A in tile (0,1)
    holds (31,32), B in tile (1,0) holds (32,31): DL meets UR. One blob
    at 8-conn, two at 4-conn."""
    img = _white(96, 96)
    if kind == "main":
        if solid:
            img[20:32, 20:32] = RED
            img[32:44, 32:44] = RED
        else:
            img[31, 31] = RED
            img[32, 32] = RED
    elif kind == "anti":
        if solid:
            img[20:32, 32:44] = RED
            img[32:44, 20:32] = RED
        else:
            img[31, 32] = RED
            img[32, 31] = RED
    else:
        raise ValueError(kind)
    return img


def all_eight_scene():
    """96x96: the centre tile's full frame is one comp, with a spoke into
    each of the 8 neighbour tiles (dirs == 0xFF)."""
    img = _white(96, 96)
    img[32, 32:64] = RED
    img[63, 32:64] = RED
    img[32:64, 32] = RED
    img[32:64, 63] = RED
    img[20:32, 48] = RED            # up
    img[64:76, 48] = RED            # down
    img[48, 20:32] = RED            # left
    img[48, 64:76] = RED            # right
    for i in range(1, 6):
        img[32 - i, 32 - i] = RED   # UL
        img[32 - i, 63 + i] = RED   # UR
        img[63 + i, 32 - i] = RED   # DL
        img[63 + i, 63 + i] = RED   # DR
    return img


def _comb_spine(img):
    """The closed square loop outside the centre tile that joins all
    teeth into one blob."""
    img[28, 28:68] = RED
    img[67, 28:68] = RED
    img[28:68, 28] = RED
    img[28:68, 67] = RED


def _teeth8():
    """60 tooth roots on the centre tile's frame, no two 8-adjacent, as
    (x, y, dx, dy): the frame pixel and the outward step."""
    out = []
    mids = list(range(34, 61, 2))
    for i in range(len(mids)):
        out.append((32, mids[i], -1, 0))      # top row, tooth up
        out.append((63, mids[i], 1, 0))       # bottom row, tooth down
        out.append((mids[i], 32, 0, -1))      # left column, tooth left
        out.append((mids[i], 63, 0, 1))       # right column, tooth right
    out += [(32, 32, -1, -1), (32, 63, -1, 1), (63, 32, 1, -1),
            (63, 63, 1, 1)]
    return out


def _teeth4():
    """62 tooth roots for 4-conn: every other pixel of the frame cycle."""
    cyc = ([(32, 32 + i) for i in range(32)]
           + [(32 + i, 63) for i in range(1, 32)]
           + [(63, 63 - i) for i in range(1, 32)]
           + [(63 - i, 32) for i in range(1, 31)])
    out = []
    for k, (x, y) in enumerate(cyc):
        if k % 2:
            continue
        if x == 32:
            out.append((x, y, -1, 0))
        elif y == 63:
            out.append((x, y, 0, 1))
        elif x == 63:
            out.append((x, y, 1, 0))
        else:
            out.append((x, y, 0, -1))
    return out


def frame_comb_scene(n, connectivity=8):
    """96x96: one blob entering the centre tile (rows 32-63, word 1)
    through n isolated frame pixels, so that tile holds n ring comps.
    n <= 60 at 8-conn, n <= 62 at 4-conn."""
    teeth = _teeth8() if connectivity == 8 else _teeth4()
    if n > len(teeth):
        raise ValueError(f"at most {len(teeth)} teeth at "
                         f"{connectivity}-conn, got {n}")
    img = _white(96, 96)
    _comb_spine(img)
    for x, y, dx, dy in teeth[:n]:
        for s in range(4):          # the root and 3 px outward
            img[x + s * dx, y + s * dy] = RED
    return img


def dot_lattice_scene(tiles=(1, 1)):
    """Isolated pixels at even (x, y): 256 comps per full tile at 8-conn."""
    w, h = 32 * tiles[0], 32 * tiles[1]
    img = _white(w, h)
    img[0::2, 0::2] = RED
    return img


def checker_tile_scene(tiles=(1, 1)):
    """(x + y) even: 512 isolated pixels per tile at 4-conn; one blob at
    8-conn."""
    w, h = 32 * tiles[0], 32 * tiles[1]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    return from_mask((xx + yy) % 2 == 0)


def rmin_trap_scene():
    """96x96: the blob's min-lin pixel (40, 40) sits in tile (1, 1); the
    blob runs down to row 50 and then left into tile (1, 0). The
    canonical comp has a PRED (left) neighbour, so it is not a seed; the
    seed is the comp in tile (1, 0), whose anchor (50, 20) is larger."""
    img = _white(96, 96)
    img[40:51, 40] = RED
    img[50, 20:41] = RED
    return img


def lexmin_multi_scene():
    """64x64: a U whose two arms start in tile (0, 0) and join in tile
    (1, 0): the lex-min tile holds two comps of the blob."""
    img = _white(64, 64)
    img[2:40, 5] = RED
    img[2:40, 20] = RED
    img[39, 5:21] = RED
    return img


def spiral_scene(size=128, step=2):
    """A 1-px square spiral with (step-1)-px gaps: one blob that re-enters
    most tiles many times. Segment lengths n, n, n, n-s, n-s, n-2s, ..."""
    img = _white(size, size)
    n = size - 1
    lengths = [n, n, n]
    k = 1
    while n - k * step > 0:
        lengths += [n - k * step, n - k * step]
        k += 1
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    x, y = 0, 0
    img[x, y] = RED
    for i, ln in enumerate(lengths):
        dx, dy = moves[i % 4]
        for _ in range(ln):
            x += dx
            y += dy
            img[x, y] = RED
    return img


def diag_snake_scene(m=45):
    """The anti-diagonal x + y = 32m + 31 through tile corners: tiles
    (0, m), (1, m-1), ... all have tx + ty = m, so for m % L != 0 no
    lattice tile is hit, and each tile's comp has its predecessor in a
    PRED (UR) tile: one seed, a chain of m+1 comps, m+1 rounds unless
    the valve fires. W = H = 32 (m + 1)."""
    n = 32 * (m + 1)
    img = _white(n, n)
    c = 32 * m + 31
    xs = np.arange(0, min(n, c + 1))
    ys = c - xs
    ok = ys < n
    img[xs[ok], ys[ok]] = RED
    return img


def tile_corners_scene(width=100, height=100, block=False):
    """A pixel at the top-left corner of every tile (isolated, dirs 0),
    or (block=True) a 2x2 block centred on every interior tile corner
    (one blob spanning four tiles, all diagonal directions), plus the
    four image corners."""
    img = _white(width, height)
    if block:
        for x in range(32, width, 32):
            for y in range(32, height, 32):
                img[x - 1:x + 1, y - 1:y + 1] = RED
    else:
        img[0::32, 0::32] = RED
    img[0, 0] = RED
    img[0, height - 1] = RED
    img[width - 1, 0] = RED
    img[width - 1, height - 1] = RED
    return img


def thin_strip_scene(long=1 << 19, across=32, axis=0, density=0.2,
                     rng_seed=0):
    """A long strip of noise: (long x 32) for axis 0, (32 x long) for
    axis 1 - the launch shapes that starve a fixed grid."""
    rng = np.random.default_rng(rng_seed)
    shape = (long, across) if axis == 0 else (across, long)
    return from_mask(rng.random(shape) < density)


# --------------------------------------------------------------- fuzz

def fuzz_noise(rng):
    w = int(rng.integers(1, 301))
    h = int(rng.integers(1, 301))
    d = float(rng.uniform(0.05, 0.6))
    return from_mask(rng.random((w, h)) < d)


def fuzz_rects_and_disks(rng):
    """Rectangles and disks with 1-px gaps and 1-px bridges: merges and
    near-merges through single pixels."""
    w = int(rng.integers(40, 260))
    h = int(rng.integers(40, 260))
    red = np.zeros((w, h), bool)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    for _ in range(int(rng.integers(3, 25))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            a, b = int(rng.integers(1, 40)), int(rng.integers(1, 40))
            red[cx:cx + a, cy:cy + b] = True
            if rng.random() < 0.5:        # carve a 1-px gap through it
                g = min(w - 1, cx + int(rng.integers(0, a)))
                red[g, cy:cy + b] = False
        else:
            r = int(rng.integers(1, 30))
            red |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    for _ in range(int(rng.integers(0, 20))):   # single-pixel bridges
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        red[x, y] = True
    return from_mask(red)


def fuzz_walks(rng):
    """Random 1-px 8-conn walks biased to cross tile borders."""
    w = int(rng.integers(32, 300))
    h = int(rng.integers(32, 300))
    red = np.zeros((w, h), bool)
    for _ in range(int(rng.integers(1, 6))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        for _ in range(int(rng.integers(50, 3000))):
            red[x, y] = True
            x = min(w - 1, max(0, x + int(rng.integers(-1, 2))))
            y = min(h - 1, max(0, y + int(rng.integers(-1, 2))))
            if rng.random() < 0.3:        # snap toward a tile border
                if rng.random() < 0.5:
                    x = min(w - 1, (x // 32) * 32 + 31 + int(rng.integers(0, 2)))
                else:
                    y = min(h - 1, (y // 32) * 32 + 31 + int(rng.integers(0, 2)))
    return from_mask(red)


def fuzz_border_zigzags(rng):
    """Zigzags on tile borders with random phase and direction: long
    comp paths, one pixel per comp."""
    w = 32 * int(rng.integers(2, 9)) + int(rng.integers(0, 32))
    h = 32 * int(rng.integers(2, 9)) + int(rng.integers(0, 32))
    red = np.zeros((w, h), bool)
    for _ in range(int(rng.integers(1, 6))):
        if rng.random() < 0.5 and w > 33:
            k = int(rng.integers(0, max(1, w // 32 - 1)))
            ph = int(rng.integers(0, 2))
            y0 = int(rng.integers(0, h))
            y1 = int(rng.integers(y0, h + 1))
            ys = np.arange(y0, y1)
            xs = 32 * k + 31 + ((ys + ph) % 2)
            ok = xs < w
            red[xs[ok], ys[ok]] = True
        elif h > 33:
            j = int(rng.integers(0, max(1, h // 32 - 1)))
            ph = int(rng.integers(0, 2))
            x0 = int(rng.integers(0, w))
            x1 = int(rng.integers(x0, w + 1))
            xs = np.arange(x0, x1)
            ys = 32 * j + 31 + ((xs + ph) % 2)
            ok = ys < h
            red[xs[ok], ys[ok]] = True
    return from_mask(red)


def fuzz_stitched_tiles(rng):
    """Per tile, one of: checkerboard, comb on the frame, full, empty,
    noise. Seams between unlike tiles make odd comps."""
    tx = int(rng.integers(1, 6))
    ty = int(rng.integers(1, 6))
    w = 32 * tx + int(rng.integers(0, 32))
    h = 32 * ty + int(rng.integers(0, 32))
    red = np.zeros((w, h), bool)
    xx, yy = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    checker = (xx + yy) % 2 == 0
    comb = np.zeros((32, 32), bool)
    comb[0, 0::2] = True
    comb[31, 1::2] = True
    comb[0::2, 0] = True
    comb[1::2, 31] = True
    comb[1:4, 0::4] = True
    for i in range((w + 31) // 32):
        for j in range((h + 31) // 32):
            kind = int(rng.integers(0, 5))
            if kind == 0:
                p = checker
            elif kind == 1:
                p = comb
            elif kind == 2:
                p = np.ones((32, 32), bool)
            elif kind == 3:
                continue
            else:
                p = rng.random((32, 32)) < float(rng.uniform(0.1, 0.6))
            a = red[32 * i:32 * i + 32, 32 * j:32 * j + 32]
            a |= p[:a.shape[0], :a.shape[1]]
    return from_mask(red)


FUZZ_FAMILIES = {
    "noise": fuzz_noise,
    "rects_and_disks": fuzz_rects_and_disks,
    "walks": fuzz_walks,
    "border_zigzags": fuzz_border_zigzags,
    "stitched_tiles": fuzz_stitched_tiles,
}
