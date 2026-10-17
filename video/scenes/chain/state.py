"""The picture between beats of the chain cut (v5): `chain_state(k)` is what
the screen shows when beat k starts (k = index into BEATS). The intro clip
voices beats 0 and 1, so the first chain state is k = 2. Scene k starts on
`chain_state(k)` and snaps to `chain_state(k + 1)` at its end, so its
last frame is the next clip's first frame (`build/seam_check.py --cut
chain` verifies this). The last beat (r14_reveal) fades out, so the last
state is k = 18.

Everything here is a pure function of k, the look and the layout; the
numbers come from `numbers.py`, act 3's arrays from `numbers.act3_arrays()`.

One orientation for the whole cut (chain_v5_proposal.md, act 3): the
kernels index [x, y] (x the PNG column) and the floors draw an array as
(row, col), so `picture()` returns the 900 px frame transposed; every
floor from r07 to r13 is then the same frame, and a lane is a screen row.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from manim import Group, VGroup

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import numbers as nb
from scenes.chain import overlays as ov
from scenes.chain import toy
from scenes.chain.geo import Frame, frame
from scenes.chain.iso import Proj, proj
from scenes.style import ASSETS, BLUE, GOLD, GREEN, INK, INK_SOFT, PURPLE, RED_PX, TEAL

BEATS = ("i0_problem", "i1_compare", "r00_zoom", "r01_cpu", "r02_one_block", "r03_giant", "r03b_one_sm",
         "r04_two_blocks", "r05_all_blocks", "r05b_snake", "r06_diagonals", "r07_real_frame", "r08_tags",
         "r09_seeds", "r10_tiles", "r11_rounds", "r12_pieces", "r13_frame", "r14_reveal")
FIRST = 2                                   # the first beat with its own chain clip
LAST = len(BEATS) - 1                       # the last state a scene starts on (r14 fades, no snap)

Z_FLOOR, Z_MACHINE, Z_OVER = 10.0, 100.0, ov.Z_OVER

# ---- colours of painted cells ----
LEVEL_LO, LEVEL_HI = "#9cc3ec", "#1f4f86"      # a wave's rings, first to last
REACHED = fl.REACHED                           # act 3: a reached pixel (the ramp's middle)
# act 2: one hue per blob. The fourth sits at 84 deg, at least 30 deg from
# amber (waiting) and from the other five (v3's "#e0a13a" was amber's twin).
BLOB_HUES = (BLUE, GREEN, "#d36ac2", "#99cb4d", "#5fc9d6", PURPLE)
fmt_ms = ov.fmt_ms
fmt_x = ov.fmt_x


def beat_index(beat) -> int:
    """A beat's index in BEATS (a name or an index)."""
    return BEATS.index(beat) if isinstance(beat, str) else int(beat)


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


def level_colors(m: np.ndarray, level: np.ndarray, upto: int | None = None, top: int | None = None) -> np.ndarray:
    """Cells: empty off the mask, red if not reached (by `upto`), else a ring
    shade (ring 0 light, ring `top` dark; default top = the deepest ring)."""
    n = m.shape[0]
    out = np.tile(fl.rgb(fl.EMPTY), (n, n, 1))
    out[m] = fl.rgb(RED_PX)
    top = max(int(level.max()) if top is None else top, 1)
    lo, hi = fl.rgb(LEVEL_LO), fl.rgb(LEVEL_HI)
    reached = (level >= 0) & (level <= (top if upto is None else upto))
    for r, c in zip(*np.nonzero(reached)):
        out[r, c] = mix(lo, hi, level[r, c] / top)
    return out


def owner_colors(m: np.ndarray, level: np.ndarray, owner: np.ndarray, upto: int | None = None,
                 top: int | None = None) -> np.ndarray:
    """level_colors on the fine grid (D4): a reached cell is a k x k mosaic of
    its real owner dots, each dot its block's ring ramp at the cell's ring
    (the hue is the block, the lightness the ring); unreached cells red,
    off the mask empty. owner: (n * k, n * k) block ids, -1 off the shape.
    Pass the result as toy_raster's `fine`."""
    n = m.shape[0]
    k = owner.shape[0] // n
    out = np.repeat(np.repeat(level_colors(m, np.full(m.shape, -1)), k, 0), k, 1)
    top = max(int(level.max()) if top is None else top, 1)
    reached = (level >= 0) & (level <= (top if upto is None else upto))
    sel = np.repeat(np.repeat(reached, k, 0), k, 1) & (owner >= 0)
    t = np.repeat(np.repeat(np.clip(level, 0, None) / top, k, 0), k, 1)[sel][:, None]
    lo, hi = _block_ramps(int(owner.max()) + 1)
    o = owner[sel]
    out[sel] = (lo[o] * (1 - t) + hi[o] * t).astype(np.uint8)
    return out


@lru_cache(maxsize=None)
def _block_ramps(blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Each block's ring ramp (first ring, last ring) as float RGB rows."""
    ramps = [mc.block_ramp(b, LEVEL_LO, LEVEL_HI) for b in range(blocks)]
    return (np.array([fl.rgb(a) for a, _ in ramps], np.float32),
            np.array([fl.rgb(b) for _, b in ramps], np.float32))


def label_colors(m: np.ndarray, lab: np.ndarray, order: list | None = None) -> np.ndarray:
    """Cells coloured by label: hue i for the i-th label in `order` (default: sorted)."""
    n = m.shape[0]
    out = np.tile(fl.rgb(fl.EMPTY), (n, n, 1))
    out[m] = fl.rgb(RED_PX)
    labels = order if order is not None else sorted(int(v) for v in np.unique(lab[lab >= 0]))
    for i, v in enumerate(labels):
        out[lab == v] = fl.rgb(BLOB_HUES[i % len(BLOB_HUES)])
    return out


# ---- the toys and pictures, fixed for the whole cut ----

DRONE_N = 18


@lru_cache(maxsize=None)
def toys() -> dict:
    drone = toy.drone(DRONE_N)
    sq = toy.square()
    sn = toy.snake()
    two = toy.two_blobs()
    many = toy.many_blobs()
    dseed = toy.centre_seed(drone)
    return {
        "drone": drone, "drone_seed": dseed,
        "drone4": toy.bfs(drone, [dseed], 4), "drone8": toy.bfs(drone, [dseed], 8),
        "square": sq, "square4": toy.bfs(sq, [toy.centre_seed(sq)], 4), "square8": toy.bfs(sq, [toy.centre_seed(sq)], 8),
        "snake": sn, "snake4": toy.bfs(sn, [toy.end_seed(sn)], 4),
        "two": two, "two_lab": toy.canonical(two),
        "many": many, "many_lab": toy.canonical(many),
    }


OWNER_FLOORS = {"square_own2": ("two", "square4"), "square_own48": ("c4", "square4"),
                "square_own48_8": ("c8", "square8")}     # floor: (owner-map launch, the toy fill it paints)


@lru_cache(maxsize=None)
def owner_fine(key: str) -> np.ndarray:
    """The owner map of launch `key` ("two", "c4", "c8") on the toy square's
    fine grid: (24 k, 24 k) block ids, -1 off the square, k dots per cell
    side. Dot (a, b) of toy cell (r, c) is the real pixel sampled at
    [k (r - 1) + a, k (c - 1) + b] (the repo's [x, y] is the toy's [row, col])."""
    own = nb.owner_map()
    g = own["grid"]
    k, o, side = int(g["dots_per_cell"]), int(g["toy_offset"]), int(g["toy_side"])
    n = toys()["square"].shape[0]
    sq = toys()["square"]
    assert int(g["toy_n"]) == n and sq[o:o + side, o:o + side].all() and int(sq.sum()) == side * side
    out = np.full((n * k, n * k), -1, np.int16)
    out[o * k:(o + side) * k, o * k:(o + side) * k] = own[key]["owner_samples"]
    return out


def owner_crop(key: str, crop: str = "ring") -> np.ndarray:
    """A full-resolution crop of launch `key`'s owner map (64 x 64; 192 x 192
    for "seed_wide"), each pixel its block's ring ramp at the crop's middle
    depth: the lens's picture (D4, Q4). (row, col) = the repo's [x, y]."""
    own = nb.owner_map()[key]
    ids = own[f"crop_{crop}"].astype(int)
    meta = own["crops"][crop]
    t = (meta["depth_min"] + meta["depth_max"]) / 2 / max(own["levels"] - 1, 1)
    lo, hi = _block_ramps(max(int(ids.max()) + 1, 2))
    return (lo[ids] * (1 - t) + hi[ids] * t).astype(np.uint8)


@lru_cache(maxsize=None)
def picture(which: str) -> np.ndarray:
    """The real picture at 900 x 900: 'red' (frame 0) or 'labels' (frame 74),
    transposed: (row, col) = the kernels' [x, y] / 10, so the act-3 arrays
    and every frame floor share one orientation (checked by
    numbers.floors()'s picture_iou_*)."""
    from PIL import Image

    name = {"red": "frame_000.png", "labels": "frame_074.png"}[which]
    img = np.asarray(Image.open(ASSETS / "ch05_input_blobs" / name).convert("RGB"))
    return np.ascontiguousarray(img.transpose(1, 0, 2))


@lru_cache(maxsize=None)
def picture_blobs() -> tuple[np.ndarray, list]:
    """8-connected blobs of the 900 x 900 picture: (label map, -1 = background,
    numbered in scan order) and each blob's pixel count. The downscale can
    join or split a few of the 2,522 blobs; the count on screen is the
    benchmark's, never this one."""
    from collections import deque

    mask = picture("labels").min(axis=2) < 250
    lab = np.full(mask.shape, -1, np.int32)
    H, W = mask.shape
    sizes = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0] >= 0:
            continue
        k = len(sizes)
        lab[r0, c0] = k
        q, n = deque([(r0, c0)]), 0
        while q:
            r, c = q.popleft()
            n += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and lab[rr, cc] < 0:
                        lab[rr, cc] = k
                        q.append((rr, cc))
        sizes.append(n)
    return lab, sizes


@lru_cache(maxsize=None)
def filled_picture() -> np.ndarray:
    """The real picture with every blob filled in the one-blob machine's blue."""
    p = picture("red").copy()
    red = (p[..., 0] > 200) & (p[..., 1] < 80) & (p[..., 2] < 80)
    lab = picture("labels")
    blob = lab.min(axis=2) < 250
    p[red | blob] = fl.rgb(LEVEL_HI)
    return p


@lru_cache(maxsize=None)
def teal_picture() -> np.ndarray:
    """ch05's label colours pulled towards teal: the same blobs, the final answer (v4)."""
    lab = picture("labels").astype(np.float32)
    blob = lab.min(axis=2) < 250
    out = lab.copy()
    out[blob] = lab[blob] * 0.45 + fl.rgb(TEAL).astype(np.float32) * 0.55
    return out.astype(np.uint8)


# ---- act 3: the tiles' pictures, from the kernels' own records (TL, WV) ----

TILE = 32                                   # a tile's side in pixels (numbers: tile_px)
SIDE_PX = 9000                              # the frame's side in pixels; the 900 px picture is a tenth
TO_900 = 900 / SIDE_PX
SQUARE_BOX = (1 / 24, 1 / 24, 22 / 24, 22 / 24)   # act 1's footprint: the toy square's cells 1-21 of 24
CROP_BOX = (0.08, 0.08, 0.92, 0.92)          # r12's crop: an image of its own, inside its own border
SQUARE_BANDS = 12                            # r11: the square's rounds, quantized, light to dark


def a3() -> dict:
    return nb.act3_arrays()


@lru_cache(maxsize=1)
def a3_meta() -> dict:
    """Where TL's context and WV's crop sit in the frame ([x, y] pixels)."""
    tl, wv = nb._load(nb.TL), nb._load(nb.WV)
    cu, cr = tl["closeup"], wv["crop"]
    return {"ctx_origin": tuple(cu["context_origin"]), "ctx_side": int(a3()["cu_ctx_red"].shape[0]),
            "tile": tuple(cu["tile"]), "seed": tuple(cu["seed"]) if "seed" in cu else None,
            "crop_origin": tuple(cr["origin"]), "crop_side": int(cr["side"])}


# -- r10, r11: the close-up blob's 7 x 7 tile context (224 x 224 px) --

def ctx_centre() -> tuple[int, int]:
    """The close-up tile's (i, j) among the context's 7 x 7 tiles (the centre)."""
    m = a3_meta()
    return (m["tile"][0] - m["ctx_origin"][0] // TILE, m["tile"][1] - m["ctx_origin"][1] // TILE)


def ctx_cells(rpx: int) -> fl.Cells:
    """The context fills the floor: 224 x 224 pixel cells."""
    return fl.Cells(rpx, a3()["cu_ctx_red"].shape)


def ctx_tile_cells(rpx: int) -> fl.Cells:
    """The context's 7 x 7 tiles on the same floor (for tile masks)."""
    return fl.Cells(rpx, a3()["cu_ctx_first_open"].shape)


def ctx_round_tiles(k: int) -> list[tuple[int, int]]:
    """The context tiles first opened in round k (ctx_tiles_in_centre_round: 7 in round 4)."""
    return [tuple(int(v) for v in t) for t in np.argwhere(a3()["cu_ctx_first_open"] == k)]


def tile_ctx_colors(upto: int = 4) -> np.ndarray:
    """(224, 224, 3): red pixels, the blob's fill blue (REACHED) in the tiles
    first opened by round `upto` (a tile's fill drawn at its first visit),
    the rest of the floor empty. r10 starts at upto = 3 (15 tiles blue) and
    closes round 4 (7 tiles) as it ends."""
    a = a3()
    red, fill, fo = a["cu_ctx_red"], a["cu_ctx_fill"], a["cu_ctx_first_open"]
    opened = (fo >= 0) & (fo <= upto)
    big = np.repeat(np.repeat(opened, TILE, 0), TILE, 1)
    out = np.tile(fl.rgb(fl.EMPTY), red.shape + (1,))
    out[red] = fl.rgb(RED_PX)
    out[fill & red & big] = fl.rgb(REACHED)
    return out


def tile_outline_segs(i: int, j: int) -> np.ndarray:
    """The four edges of context tile (i, j) as fl.segments in pixel cells."""
    r0, c0, r1, c1 = i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE
    return np.array([[r0, c0, r0, c1], [r1, c0, r1, c1], [r0, c0, r1, c0], [r0, c1, r1, c1]], float)


def tile_ctx_raster(rpx: int, upto: int = 4, outline: tuple | None = None, border_bits: float = 0.0,
                    grid_alpha: float = 0.5) -> np.ndarray:
    """r10-r11's floor: tile_ctx_colors with a faint grid every tile; outline:
    a tile (i, j) outlined in white (r10 starts with the centre outlined);
    border_bits: the closed tile's reached border cells lit white at this
    alpha (r11's "only the tile's border bits go out")."""
    cells = ctx_cells(rpx)
    out = fl.cells_raster(tile_ctx_colors(upto), cells, grid=TILE, grid_alpha=grid_alpha)
    if border_bits > 0:
        fl.paint(out, cells, border_bits_mask(), INK, border_bits)
    if outline is not None:
        fl.segments(out, cells, tile_outline_segs(*outline), INK, width=max(2.0, rpx / 450))
    return out


@lru_cache(maxsize=1)
def border_bits_mask() -> np.ndarray:
    """(224, 224) bool: the closed tile's reached cells on its border (lane 0
    and lane 31 rows, bit 0 and bit 31 columns: numbers.floors()'s
    cu_border_bits, 31 / 26 / 18 / 2)."""
    a = a3()
    reached = mc.tile_bits(a["cu_final_chapter"])
    edge = np.zeros_like(reached)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    ci, cj = ctx_centre()
    out = np.zeros(a["cu_ctx_red"].shape, bool)
    out[ci * TILE:(ci + 1) * TILE, cj * TILE:(cj + 1) * TILE] = reached & edge
    return out


@lru_cache(maxsize=1)
def ctx_targets() -> tuple:
    """The tiles the closed tile's border bits really open (first opened after
    its round): ((i, j), red pixels its reached border touches there), as
    numbers.floors() counts them (cu_targets_new: 4, 32 / 18 / 1 / 1 px)."""
    a = a3()
    fo, red = a["cu_ctx_first_open"], a["cu_ctx_red"]
    ci, cj = ctx_centre()
    R = np.zeros_like(red)
    R[ci * TILE:(ci + 1) * TILE, cj * TILE:(cj + 1) * TILE] = mc.tile_bits(a["cu_final_chapter"])
    D = np.zeros_like(red)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            D |= np.roll(np.roll(R, dx, 0), dy, 1)
    D &= red
    D[ci * TILE:(ci + 1) * TILE, cj * TILE:(cj + 1) * TILE] = False
    hit: dict = {}
    for p, q in np.argwhere(D):
        hit[(int(p) // TILE, int(q) // TILE)] = hit.get((int(p) // TILE, int(q) // TILE), 0) + 1
    k = int(fo[ci, cj])
    return tuple(sorted(((t, v) for t, v in hit.items() if fo[t] > k), key=lambda e: -e[1]))


# -- r11: the 64 Mpx square, 250 x 250 tiles, by the round each first opened --

def square_tiles_colors(upto: int | None = None) -> np.ndarray:
    """(250, 250, 3): a tile opened by round `upto` (default: all) shaded by
    its first-open round in SQUARE_BANDS bands, light (round 0) to dark
    (LEVEL_LO .. LEVEL_HI, act 1's ring ramp); the rest red."""
    fo = a3()["sq_first_open"].astype(int)
    top = max(int(fo.max()), 1)
    band = np.minimum(fo * SQUARE_BANDS // (top + 1), SQUARE_BANDS - 1)
    lo, hi = fl.rgb(LEVEL_LO).astype(np.float32), fl.rgb(LEVEL_HI).astype(np.float32)
    t = (band / (SQUARE_BANDS - 1))[..., None]
    out = (lo * (1 - t) + hi * t).astype(np.uint8)
    if upto is not None:
        out[fo > upto] = fl.rgb(RED_PX)
    return out


def square_cells(rpx: int) -> fl.Cells:
    return fl.Cells(rpx, a3()["sq_first_open"].shape, SQUARE_BOX)


def square_tiles_raster(rpx: int, upto: int | None = None) -> np.ndarray:
    """r11's square at act 1's footprint, gridless (a tile is ~4 px)."""
    return fl.cells_raster(square_tiles_colors(upto), square_cells(rpx))


# -- r11, r13: the frame --

@lru_cache(maxsize=1)
def largest_blob() -> int:
    """The 900 px picture's largest blob: the close-up blob (76,090 px, the
    frame's largest, cu_blob_rank 0); checked to hold the close-up's tile."""
    lab, sizes = picture_blobs()
    k = int(np.argmax(sizes))
    m = a3_meta()
    x, y = (m["tile"][0] * TILE + TILE // 2) * TO_900, (m["tile"][1] * TILE + TILE // 2) * TO_900
    near = lab[int(x) - 3:int(x) + 4, int(y) - 3:int(y) + 4]
    assert (near == k).any(), "the largest blob at 900 px is not the close-up's blob"
    return k


def blob_seed_frac() -> tuple[float, float]:
    """The close-up blob's one seed (TL closeup.seed, [x, y] pixels) as a
    fraction of a frame floor's side, for Floor.frac_point: r11's "from
    one seed" dot on frame_one_blob."""
    x, y = a3_meta()["seed"]
    return x / SIDE_PX, y / SIDE_PX


@lru_cache(maxsize=1)
def one_blob_picture() -> np.ndarray:
    """The red frame with its largest blob filled (REACHED): r11's "that's
    one blob, from one seed"."""
    p = picture("red").copy()
    lab, _ = picture_blobs()
    p[lab == largest_blob()] = fl.rgb(REACHED)
    return p


def frame_rounds_picture(upto: int) -> np.ndarray:
    """(900, 900, 3): r13's claims spreading out. A sample whose piece is
    claimed by round `upto` (WV full_round_900; seeds are round 0) takes its
    final label colour from the labelled picture; interior samples are
    labelled from the start; the rest stay as the red frame."""
    r9 = a3()["full_round_900"]
    red, lab = picture("red"), picture("labels")
    sel = ((r9 >= 0) & (r9 <= upto)) | (r9 == -2)
    sel &= lab.min(axis=2) < 250
    out = red.copy()
    out[sel] = lab[sel]
    return out


@lru_cache(maxsize=1)
def frame_seed_tiles() -> tuple[np.ndarray, np.ndarray]:
    """The frame's tiles holding seeds (WV full_tile_seeds, 282 x 282): their
    (tx, ty) and a fade-in phase in [0, 1) each (a fixed shuffle, so the
    dots come up scattered over the whole frame, tile by tile)."""
    s = a3()["full_tile_seeds"]
    t = np.argwhere(s > 0)
    phase = np.random.default_rng(17).permutation(len(t)) / len(t)
    return t, phase


def frame_tile_cells(rpx: int) -> fl.Cells:
    """The frame's 282 x 282 tiles on a frame floor (picture_raster fills the
    floor with 9,000 px; 282 tiles span 9,024, so the box runs past it)."""
    n = a3()["full_tile_seeds"].shape[0]
    f = n * TILE / SIDE_PX
    return fl.Cells(rpx, (n, n), (0.0, 0.0, f, f))


def frame_seed_dots(raster: np.ndarray, rpx: int, p: float = 1.0, spread: float = 0.35,
                    radius: float | None = None, alpha: float = 0.9) -> np.ndarray:
    """r13: a small white dot on every tile with a seed (8,000-odd dots for
    11,609 seeds), each fading in over `spread` of the fade as p goes 0 -> 1
    (no rings, nothing flashes). In place; returns the raster."""
    t, phase = frame_seed_tiles()
    a = np.clip((p - phase * (1 - spread)) / spread, 0, 1) * alpha
    keep = a > 0
    r = radius if radius is not None else max(1.2, rpx / 900)
    return fl.discs(raster, frame_tile_cells(rpx), t[keep] + 0.5, r, INK, a[keep])


def frame_window(origin_xy, side_px: float, box=(0.0, 0.0, 1.0, 1.0)) -> tuple[float, float, float]:
    """The window (r0, c0, side) of the 900 px picture that puts the frame's
    square [x0, x0 + side) x [y0, y0 + side) (pixels, [x, y]) at `box` on
    the floor: the end of a zoom (fl.window_raster)."""
    t, l, b, r = box
    side9 = side_px * TO_900 / (b - t)
    return origin_xy[0] * TO_900 - t * side9, origin_xy[1] * TO_900 - l * side9, side9


def ctx_window() -> tuple[float, float, float]:
    """r10's zoom target: the context's 224 px window, filling the floor."""
    m = a3_meta()
    return frame_window(m["ctx_origin"], m["ctx_side"])


def crop_window() -> tuple[float, float, float]:
    """r12's zoom target: the crop's 384 px window at CROP_BOX."""
    m = a3_meta()
    return frame_window(m["crop_origin"], m["crop_side"], CROP_BOX)


# -- r12: the 384 px crop, run as its own image (12 x 12 tiles, 135 pieces) --

def crop_cells(rpx: int) -> fl.Cells:
    return fl.Cells(rpx, a3()["crop_red"].shape, CROP_BOX)


def crop_tile_cells(rpx: int) -> fl.Cells:
    return fl.Cells(rpx, a3()["crop_tile_lattice"].shape, CROP_BOX)


def _greedy(nodes: list, edges: set, n_hues: int) -> dict:
    """A colouring of `nodes` (in order) so no edge joins two equal colours;
    each node takes the free colour used least so far, so the hues spread."""
    col: dict = {}
    count = [0] * n_hues
    for v in nodes:
        used = {col[u] for u in nodes if u in col and ((u, v) in edges or (v, u) in edges)}
        free = [c for c in range(n_hues) if c not in used]
        if not free:
            raise ValueError(f"no free hue for {v}: {len(used)} neighbours")
        col[v] = min(free, key=lambda c: (count[c], c))
        count[col[v]] += 1
    return col


@lru_cache(maxsize=1)
def crop_hues() -> tuple[dict, dict]:
    """(wave hue, blob hue), each a dict of RGB rows: wave (seed piece k) ->
    hue, blob (canonical label) -> hue. Waves that meet differ; each blob's
    smallest-label wave (its seed with the smallest anchor) differs from
    the other blobs' in the same or a neighbouring tile; a blob ends in the hue of its
    smallest-label wave, so "each blob keeps its smallest label" reads as
    that hue winning."""
    a = a3()
    wave, lab, tile = a["crop_comp_wave"], a["crop_comp_label"], a["crop_comp_tile"]
    seeds = [int(k) for k in np.nonzero(a["crop_comp_seed"])[0]]
    anchor = a["crop_comp_anchor"]
    blob_of_seed = {k: int(lab[k]) for k in seeds}
    winner = {}
    for k in seeds:
        b = blob_of_seed[k]
        if b not in winner or anchor[k] < anchor[winner[b]]:
            winner[b] = k
    edges = {(int(wave[t]), int(wave[o])) for t, o in zip(a["crop_meet_tester"], a["crop_meet_other"])}
    ny = a["crop_tile_lattice"].shape[1]
    tiles_of: dict = {}
    for k in range(len(lab)):
        tiles_of.setdefault(int(lab[k]), set()).add(divmod(int(tile[k]), ny))
    blobs = sorted(tiles_of)
    for i, x in enumerate(blobs):                  # blobs in the same or neighbouring tiles differ
        for y in blobs[i + 1:]:
            if any(abs(p - q) <= 1 and abs(u - v) <= 1 for p, u in tiles_of[x] for q, v in tiles_of[y]):
                edges.add((winner[x], winner[y]))
    order = sorted(winner.values()) + [k for k in seeds if k not in winner.values()]
    col = _greedy(order, edges, len(BLOB_HUES))
    hues = [fl.rgb(h) for h in BLOB_HUES]
    return {k: hues[c] for k, c in col.items()}, {b: hues[col[k]] for b, k in winner.items()}


def crop_colors(upto: int | None = None, final: bool = False) -> np.ndarray:
    """(384, 384, 3): the crop's red pixels; with `upto`, every piece claimed
    by round `upto` (seeds are round 0) in its wave's hue; final: every
    piece in its blob's hue (crop_labels)."""
    a = a3()
    comp, red = a["crop_comp"], a["crop_red"]
    out = np.tile(fl.rgb(fl.EMPTY), red.shape + (1,))
    out[red] = fl.rgb(RED_PX)
    wave_hue, blob_hue = crop_hues()
    if final:
        lab = a["crop_comp_label"]
        table = np.array([blob_hue[int(lab[k])] for k in range(len(lab))], np.uint8)
        out[comp >= 0] = table[comp[comp >= 0]]
        return out
    if upto is not None:
        rnd, wave = a["crop_comp_round"], a["crop_comp_wave"]
        on = (rnd >= 0) & (rnd <= upto)
        table = np.array([wave_hue[int(wave[k])] if on[k] else fl.rgb(RED_PX) for k in range(len(rnd))], np.uint8)
        out[comp >= 0] = table[comp[comp >= 0]]
    return out


@lru_cache(maxsize=1)
def piece_anchors() -> np.ndarray:
    """(135, 2): each piece's anchor (its first pixel, the minimum x * 384 + y)
    as a cell centre (row, col) on crop_cells, for fl.discs."""
    side = a3_meta()["crop_side"]
    x, y = np.divmod(a3()["crop_comp_anchor"].astype(np.int64), side)
    return np.stack([x + 0.5, y + 0.5], axis=1).astype(float)


TICK = 5.0                                  # half a meeting tick's length, in crop pixels


@lru_cache(maxsize=1)
def meeting_ticks() -> tuple[np.ndarray, np.ndarray]:
    """WV's 116 meeting edges as short ticks across the tile borders where
    the two pieces touch, a stitch between them: (segs (116, 4) of (r0,
    c0, r1, c1) in crop pixel cells, their rounds). On a shared tile edge
    the tick crosses it at the middle of the touching stretch; at a shared
    corner it crosses the corner along the other diagonal."""
    a = a3()
    comp = a["crop_comp"]
    side = comp.shape[0]
    ny = a["crop_tile_lattice"].shape[1]
    segs = []
    for t_, o_ in zip(a["crop_meet_tester"], a["crop_meet_other"]):
        A, B = comp == t_, comp == o_
        D = np.zeros_like(A)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                sh = np.zeros_like(A)
                sh[max(dx, 0):side + min(dx, 0), max(dy, 0):side + min(dy, 0)] = \
                    A[max(-dx, 0):side - max(dx, 0), max(-dy, 0):side - max(dy, 0)]
                D |= sh
        q = np.argwhere(D & B)                                  # B's pixels touching A
        txa, tya = divmod(int(a["crop_comp_tile"][t_]), ny)
        txb, tyb = divmod(int(a["crop_comp_tile"][o_]), ny)
        h = TICK
        if txa != txb and tya != tyb:                           # a corner: across the pieces' diagonal
            r, c = max(txa, txb) * TILE, max(tya, tyb) * TILE
            s_ = 1 if (txb - txa) * (tyb - tya) > 0 else -1
            d = h / np.sqrt(2)
            segs.append([r - d, c + s_ * d, r + d, c - s_ * d])
        elif txa != txb:                                        # a horizontal border (rows differ)
            r = max(txa, txb) * TILE
            m = (q[:, 1].min() + q[:, 1].max() + 1) / 2
            segs.append([r - h, m, r + h, m])
        else:                                                   # a vertical border (columns differ)
            c = max(tya, tyb) * TILE
            m = (q[:, 0].min() + q[:, 0].max() + 1) / 2
            segs.append([m, c - h, m, c + h])
    return np.array(segs, float), a["crop_meet_round"].astype(int)


def crop_raster(rpx: int, upto: int | None = None, final: bool = False, pieces: float = 0.0,
                anchors: float = 0.0, hatch: float = 0.0, ticks: float = 0.0, ticks_upto: int | None = None,
                border: float = 1.0, grid_alpha: float = 0.45) -> np.ndarray:
    """r12's floor: the crop inside its own border (CROP_BOX), a faint tile
    grid, crop_colors(upto, final); then, each at its alpha (0: off):
    pieces (dark cut lines where two pieces meet: a blob cut by tiles),
    anchors (a white dot, dark-ringed, on each piece's first pixel), hatch
    (the lattice tiles: seeds only because of the lattice), ticks (white
    stitches across the borders where two waves met, those recorded by
    round `ticks_upto`, default all)."""
    cells = crop_cells(rpx)
    out = fl.cells_raster(crop_colors(upto, final), cells, grid=TILE, grid_alpha=grid_alpha)
    if hatch > 0:
        fl.hatch(out, crop_tile_cells(rpx), a3()["crop_tile_lattice"], INK_SOFT, alpha=hatch,
                 period=max(8, round(rpx / 100)), width=max(3, round(rpx / 360)))
    if pieces > 0:
        fl.outline(out, cells, a3()["crop_comp"], fl.EMPTY, width=max(2, round(rpx / 500)), alpha=pieces,
                   outer=False)
    if anchors > 0:
        r = max(3.0, rpx / 300)
        fl.discs(out, cells, piece_anchors(), r + max(1.5, rpx / 900), fl.EMPTY, anchors)
        fl.discs(out, cells, piece_anchors(), r, INK, anchors)
    if ticks > 0:
        segs, rounds = meeting_ticks()
        keep = rounds <= (ticks_upto if ticks_upto is not None else int(rounds.max()))
        fl.segments(out, cells, segs[keep], INK, width=max(3.0, rpx / 330), alpha=ticks)
    if border > 0:
        fl.border(out, cells, INK_SOFT, width=max(2, round(rpx / 500)), alpha=border, pad=2)
    return out


def nums() -> dict:
    return {k: v.value for k, v in nb.load().items()}


# ---- the scoreboard rows ----

def score_rows(beat, flip: int = 2) -> tuple:
    """The three rows as they stand at the end of `beat` (a name in BEATS or
    its index), like for like (chain_v5_proposal.md, "The scoreboard"):
    row 1 from r02 (r02's one block against the 4-neighbour CPU), row 2
    from r06 (the diagonals against the 8-neighbour CPU with depth) and r11
    (the tiles against it without depth), row 3 from r09 (the seeds against
    the labels reset before the clock) and r13 (the waves against the
    labels reset on it). Before its beat a row is "?" (row 1: None, absent).
    Row 1 at r14 flips in two steps: flip 0 keeps r02's row, flip 1 shows
    the 8-neighbour CPU with the GPU and the winner "?", flip 2 the tiles'
    win. Every time and factor goes through card_ms / card_x."""
    k = beat_index(beat)
    n, d = nums(), nb.derived()
    i = BEATS.index
    small_pic = f"{round(n['small_px'] / 1024)} K px test blob"
    small = ov.Row("one small blob", n["small_cpu_ms"], n["small_gpu_ms"], "CPU", fmt_x(d["small_cpu_x"]),
                   small_pic, "CPU: 4 neighbours")
    small1 = ov.Row("one small blob", n["small_cpu8_ms"], None, None, "", small_pic, "CPU: 8 neighbours")
    small2 = ov.Row("one small blob", n["small_cpu8_ms"], n["small_tiles_ms"], "GPU", fmt_x(d["small_tiles_x"]),
                    small_pic, "CPU: 8 neighbours")
    huge_pic = f"{round(n['big_filled'] / 1e6)} Mpx test blob"
    huge6 = ov.Row("one huge blob", n["big_cpu8_ms"], n["big_8ways_ms"], "GPU", fmt_x(d["big_8ways_x"]),
                   huge_pic, "CPU: 8 neighbours")
    huge11 = ov.Row("one huge blob", n["big_cpu_fill_ms"], n["big_tiles_ms"], "GPU", fmt_x(d["big_tiles_x"]),
                    huge_pic, "CPU: 8 neighbours")
    many_pic = f"{round(n['n_pixels'] / 1e6)} Mpx frame, {int(n['n_blobs']):,} blobs"
    many9 = ov.Row("many blobs", n["count_cpu_pre_ms"], n["count_ch05_ms"], "GPU", fmt_x(d["count_ch05_x"]),
                   many_pic, "CPU: same labels")
    many13 = ov.Row("many blobs", n["count_cpu_ms"], n["count_gpu_ms"], "GPU", fmt_x(d["count_x"]),
                    many_pic, "CPU: same labels")
    r1 = None if k < i("r02_one_block") else small
    if k >= i("r14_reveal"):
        r1 = (small, small1, small2)[flip]
    r2 = huge11 if k >= i("r11_rounds") else huge6 if k >= i("r06_diagonals") else "?"
    r3 = many13 if k >= i("r13_frame") else many9 if k >= i("r09_seeds") else "?"
    return r1, r2, r3


def card_footnote() -> str:
    """The r14 card's footnote (never spoken): every row's GPU time with the
    copies, rounded (card_ms_note), and the small blob's CPU beside it."""
    n = nums()
    note = ov.fmt_ms_note
    cpu = fmt_ms(n["small_cpu8_ms"]).removesuffix(" ms")
    return (f"With the copies: small blob {note(n['small_tiles_e2e_ms'])} (CPU {cpu}), "
            f"huge blob {note(n['big_tiles_e2e_ms'])}, frame {note(n['frame_e2e_ms'])}.")


# ---- the boundary table ----

@dataclass(frozen=True)
class Boundary:
    floors: tuple = ()                 # ("drone_red",), ("drone_cpu", "drone_gpu") for a split
    machine: mc.Spec | None = None
    goal: ov.GoalSpec | None = None
    rows: tuple = (None, None, None)   # scoreboard: Row, "?" or None per row
    inset: str | None = None           # the inset's one line
    tag: tuple | None = None           # (act, won)
    cap: tuple | None = None           # (text, sub or None): the caption crossing the cut


ALL_LIT = ("lit",) * 24


def act3_machine(kind: str = "tile") -> mc.Spec:
    """Act 3's machine at rest: every SM lit with three blocks (72, the
    tiles' and the waves' cooperative grids), the queue full of `kind`
    tokens (tile, then piece), the gate at rest, no launch pill."""
    return mc.Spec(rooms=ALL_LIT, teams=(mc.MAX_TEAMS,) * 24, queue=True, gate="rest",
                   tokens=(RED_PX,) * mc.SLOTS, token_kind=kind)


def boundary(k: int) -> Boundary:
    """The state when beat k starts (FIRST <= k <= LAST). Captions are a big
    word and at most a short qualifier: the narration carries the rest."""
    if not 0 <= k <= LAST:
        raise ValueError(f"no chain state {k}: states run 0..{LAST} (r14_reveal fades out, nothing follows)")
    n, d = nums(), nb.derived()
    rows = score_rows(k - 1) if k >= 1 else (None, "?", "?")
    goal = ov.GoalSpec(cpu_tick=True)
    b = Boundary(goal=goal, rows=rows)
    if k <= 2:
        return b
    b = replace(b, floors=("drone_red",), inset="one drone", cap=(f"{int(toys()['drone'].sum())} pixels", None))
    if k == 3:
        return b
    b = replace(b, floors=("drone_cpu",), machine=mc.Spec(cpu="lit"),
                cap=(f"{int(n['sm'])} SMs", "streaming multiprocessors"))
    if k == 4:
        return b
    # r02: one block on SM 0, its queue in the SM's shared memory (shelf "full")
    one = mc.Spec(cpu="lit", rooms=("lit",) + ("off",) * 23, teams=(1,) + (0,) * 23, shelves=("full",))
    b = replace(b, floors=("drone_cpu", "drone_gpu"), machine=one, cap=("CPU wins", "one small blob"))
    if k == 5:
        return b
    # r03_giant: the giant square; the queue overflows the shared memory and
    # its overflow spills to VRAM (a few tokens in chip 0's slots, no label)
    giant = mc.Spec(rooms=("lit",) + ("off",) * 23, teams=(1,) + (0,) * 23, shelves=("full",), room_fence="rest",
                    spill=True)
    b = replace(b, floors=("square4",), machine=giant, inset="stress test", tag=("SIZE", False),
                cap=("spill", "to global memory"))
    if k == 6:
        return b
    # r03b: a little faster than the CPU, but one SM of twenty-four
    b = replace(b, cap=(f"1 of {int(n['sm'])} SMs", None))
    if k == 7:
        return b
    # r04: block 1 on SM 2 (measured), green; the whole queue in VRAM, the
    # gate at rest; the floor is who painted what (blue and green, mixed)
    two_ids = tuple(int(i) for i in n["two_sm_ids"])
    two = mc.place_blocks(mc.with_rooms(mc.Spec(queue=True, tokens=(RED_PX,) * 6, gate="rest"), "lit", two_ids),
                          two_ids, hues=mc.block_hues(len(two_ids)))
    b = replace(b, floors=("square_own2",), machine=two, cap=(f"{int(n['sm']) - 2} SMs idle", None))
    if k == 8:
        return b
    # r05: 48 blocks, two per SM in the measured order, each in its block's hue
    full = mc.Spec(rooms=ALL_LIT, teams=(2,) * 24, queue=True, gate="rest")
    hued = mc.place_blocks(full, [int(i) for i in n["big_sm_ids"]], hues=mc.block_hues(int(n["big_blocks"])))
    b = replace(b, floors=("square_own48",), machine=hued,
                cap=(f"+{round(100 * (d['sweep_x'] - 1))}% at best", f"{round(n['q1_px'] / 1e6)} Mpx square"))
    if k == 9:
        return b
    # r05b: the snake, 48 lose to 1; the gate
    b = replace(b, floors=("snake",), cap=(f"{int(d['snake_stops']):,} stops", "~2 µs each"))
    if k == 10:
        return b
    # r06: 8 ways, against the 8-neighbour CPU; the floor is the 8-way owner map
    b = replace(b, floors=("square_own48_8",), tag=("SIZE", True), cap=(fmt_x(d["big_8ways_x"]), "CPU: 8 neighbours too"))
    if k == 11:
        return b
    # act 2 (from r07's pull-back): one hue per blob, so the cubes are plain blue again
    m9 = replace(full, launches=f"{int(n['est_ch03_calls']):,}")
    b = replace(b, floors=("frame_filled",), machine=m9,
                goal=ov.GoalSpec(needle_ms=n["est_ch03_ms"], needle_text="~" + fmt_ms(n["est_ch03_ms"])),
                tag=("COUNT", False), inset="drone frame", cap=("~" + fmt_ms(n["est_ch03_ms"]), "estimate"))
    if k == 12:
        return b
    m10 = replace(m9, launches=f"{int(n['est_ch04_calls']):,}", tokens=(RED_PX,) * 6,
                  tag_tokens=(BLUE, GREEN, BLUE, BLUE, GREEN, BLUE))
    b = replace(b, floors=("two",), machine=m10,
                goal=ov.GoalSpec(needle_ms=n["est_ch04_ms"], needle_text="~" + fmt_ms(n["est_ch04_ms"])),
                cap=("2 seeds", "per launch"))
    if k == 13:
        return b
    # r09: the GPU finds its seeds, faster than the CPU (row 3), still over budget
    m11 = replace(full, launches="3", tokens=(RED_PX,) * mc.SLOTS)
    b = replace(b, floors=("frame_labels",), machine=m11,
                goal=ov.GoalSpec(needle_ms=n["ch05_ms"], needle_text=fmt_ms(n["ch05_ms"])),
                cap=(fmt_ms(n["ch05_ms"]), "still over 33 ms"))
    if k == 14:
        return b
    # r10: the queue carries tiles; three blocks per SM; the close-up blob's
    # tile context, rounds 0-4 closed
    b = replace(b, floors=("tile_ctx",), machine=act3_machine("tile"),
                cap=(f"{int(n['cu_trips'])} trips", "the last changes nothing"))
    if k == 15:
        return b
    # r11: one gate per round; row 2 jumps; but one blob, from one seed
    b = replace(b, floors=("frame_one_blob",), cap=("1 blob", "from 1 seed"))
    if k == 16:
        return b
    # r12: the crop as its own image, every blob in its smallest label's hue
    b = replace(b, floors=("crop_labels",), machine=act3_machine("piece"), cap=("min", "smallest label wins"))
    if k == 17:
        return b
    # r13: the frame, every blob counted, inside the budget (kernel time)
    return replace(b, floors=("frame_labels",),
                   goal=ov.GoalSpec(needle_ms=n["frame_ms"], needle_text=fmt_ms(n["frame_ms"]), needle_color=TEAL),
                   tag=("COUNT", True), cap=(fmt_ms(n["frame_ms"]), "kernel time, no copies"))


# ---- floor rasters ----

FLOOR_NAMES = ("drone_red", "drone_cpu", "drone_gpu", "square4", "square8", "snake", *OWNER_FLOORS, "two",
               "frame_filled", "frame_labels", "frame_red", "frame_teal", "tile_ctx", "tile_ctx_open",
               "square_tiles", "frame_one_blob", "frame_rounds", "crop_red", "crop_labels")


def floor_raster(name: str, rpx: int) -> np.ndarray:
    t = toys()
    if name == "drone_red":
        return fl.toy_raster(level_colors(t["drone"], np.full(t["drone"].shape, -1)), rpx)
    if name in ("drone_cpu", "drone_gpu"):
        return fl.toy_raster(level_colors(t["drone"], t["drone4"].level), rpx)
    if name == "square4":
        return fl.toy_raster(level_colors(t["square"], t["square4"].level), rpx)
    if name == "square8":
        return fl.toy_raster(level_colors(t["square"], t["square8"].level), rpx)
    if name == "snake":
        return fl.toy_raster(level_colors(t["snake"], t["snake4"].level), rpx)
    if name in OWNER_FLOORS:
        key, fill = OWNER_FLOORS[name]
        lv = t[fill].level
        return fl.toy_raster(level_colors(t["square"], lv), rpx, fine=owner_colors(t["square"], lv, owner_fine(key)))
    if name == "two":
        return fl.toy_raster(label_colors(t["two"], t["two_lab"]), rpx)
    if name == "frame_filled":
        return fl.picture_raster(filled_picture(), rpx)
    if name == "frame_labels":
        return fl.picture_raster(picture("labels"), rpx)
    if name == "frame_red":
        return fl.picture_raster(picture("red"), rpx)
    if name == "frame_teal":
        return fl.picture_raster(teal_picture(), rpx)
    # act 3
    if name == "tile_ctx":                     # k = 15: rounds 0-4 closed, faint grid, no outline
        return tile_ctx_raster(rpx, upto=ctx_centre_round())
    if name == "tile_ctx_open":                # r10's start: rounds 0-3 closed, the centre tile outlined
        return tile_ctx_raster(rpx, upto=ctx_centre_round() - 1, outline=ctx_centre())
    if name == "square_tiles":                 # r11: every round closed
        return square_tiles_raster(rpx)
    if name == "frame_one_blob":               # k = 16
        return fl.picture_raster(one_blob_picture(), rpx)
    if name == "frame_rounds":                 # r13: every claim (rounds 0-4 at 900 px)
        return fl.picture_raster(frame_rounds_picture(int(a3()["full_round_900"].max())), rpx)
    if name == "crop_red":                     # r12, after the zoom lands
        return crop_raster(rpx)
    if name == "crop_labels":                  # k = 17
        return crop_raster(rpx, final=True)
    raise KeyError(name)


def ctx_centre_round() -> int:
    """The round the close-up tile first opened in (4: ctx_centre_round)."""
    ci, cj = ctx_centre()
    return int(a3()["cu_ctx_first_open"][ci, cj])


# ---- the whole state ----

@dataclass
class State:
    floors: list = field(default_factory=list)
    machine: mc.Machine | None = None
    over: dict = field(default_factory=dict)       # goal, strip, inset, tag
    cap: VGroup | None = None

    def mobs(self) -> list:
        """Add order: machine, overlays, floors, caption. Drawing order is the
        z_index; the add order only decides what the renderer may cache during
        a play (everything before the first moving mobject), and floors move."""
        out = []
        if self.machine is not None:
            out += self.machine.mobs()
        out += [m for m in self.over.values() if m is not None]
        out += self.floors
        if self.cap is not None:
            out.append(self.cap)
        return out


@dataclass
class Layout:
    G: Frame
    P: Proj
    floor_pl: object
    machine_pl: object
    split_pl: dict


def layout() -> Layout:
    G = frame()
    P = proj()
    split = {s: fl.placement(P, *G.split(s)) for s in ("left", "right")}
    return Layout(G, P, fl.placement(P, G.floor_c, G.floor_wh), mc.placement(P, G.machine_c, G.machine_wh), split)


def make_floors(names: tuple, L: Layout) -> list:
    if not names:
        return []
    pls = [L.floor_pl] if len(names) == 1 else [L.split_pl["left"], L.split_pl["right"]]
    out = []
    for name, pl in zip(names, pls):
        f = fl.Floor(pl, z0=Z_FLOOR)
        f.show_raster(floor_raster(name, f.rpx))
        out.append(f)
    return out


def overlays(b: Boundary, L: Layout) -> dict:
    n = nums()
    G = L.G
    o = {"goal": ov.goal_meter(G, b.goal, n) if b.goal else None,
         "strip": ov.score_strip(G, b.rows) if any(r is not None for r in b.rows) else None,
         "inset": ov.inset(G, b.inset) if b.inset else None,
         "tag": ov.act_tag(G, *b.tag) if b.tag else None}
    return o


def chain_state(k: int, L: Layout | None = None) -> State:
    from scenes.chain.base import caption_group      # base imports this module

    L = L or layout()
    b = boundary(k)
    floors = make_floors(b.floors, L)
    m = mc.build(b.machine, L.machine_pl, z0=Z_MACHINE) if b.machine is not None else None
    cap = caption_group(L.G, *b.cap) if b.cap else None
    return State(floors, m, overlays(b, L), cap)


def carry_caption(k: int) -> tuple | None:
    """The caption crossing into beat k (the last caption of beat k - 1)."""
    return boundary(k).cap if FIRST <= k <= LAST else None
