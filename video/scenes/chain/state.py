"""The picture between beats: `chain_state(k)` is what the screen shows
after beat k-1 (k = 0: an empty frame). Scene k starts on it and snaps to
`chain_state(k + 1)` at its end, so its last frame is the next clip's
first frame (`build/seam_check.py --cut chain` verifies this).

Everything here is a pure function of k, the look and the layout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Group, RoundedRectangle, VGroup, config

from scenes.chain import floor as fl
from scenes.chain import machine as mc
from scenes.chain import numbers as nb
from scenes.chain import toy
from scenes.chain.geo import Frame, frame
from scenes.chain.iso import Proj, proj
from scenes.style import (
    ASSETS, BLUE, GOLD, GREEN, GRID, INK, INK_SOFT, PURPLE, RED_PX, TEAL, label,
)

BEATS = ("c00_job", "c01_one_worker", "c02_team", "c03_more_teams", "c04_gate", "c05_diagonals",
         "c06_real_picture", "c07_tags", "c08_seeds", "c09_runs", "c10_reveal")

Z_FLOOR, Z_MACHINE, Z_OVER = 10.0, 100.0, 500.0

# ---- colours of painted cells ----
LEVEL_LO, LEVEL_HI = "#9cc3ec", "#1f4f86"      # a wave's rings, first to last
BLOB_HUES = (BLUE, GREEN, "#d36ac2", "#e0a13a", "#5fc9d6", PURPLE)


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


def label_colors(m: np.ndarray, lab: np.ndarray, order: list | None = None) -> np.ndarray:
    """Cells coloured by label: hue i for the i-th label in `order` (default: sorted)."""
    n = m.shape[0]
    out = np.tile(fl.rgb(fl.EMPTY), (n, n, 1))
    out[m] = fl.rgb(RED_PX)
    labels = order if order is not None else sorted(int(v) for v in np.unique(lab[lab >= 0]))
    for i, v in enumerate(labels):
        out[lab == v] = fl.rgb(BLOB_HUES[i % len(BLOB_HUES)])
    return out


# ---- the toys, fixed for the whole cut ----

@lru_cache(maxsize=None)
def toys() -> dict:
    blob = toy.blob()
    sq = toy.square()
    sn = toy.snake()
    two = toy.two_blobs()
    many = toy.many_blobs()
    return {
        "blob": blob, "blob4": toy.bfs(blob, [toy.centre_seed(blob)], 4),
        "square": sq, "square4": toy.bfs(sq, [toy.centre_seed(sq)], 4), "square8": toy.bfs(sq, [toy.centre_seed(sq)], 8),
        "snake": sn, "snake4": toy.bfs(sn, [toy.end_seed(sn)], 4),
        "two": two, "two_lab": toy.canonical(two),
        "many": many, "many_lab": toy.canonical(many),
    }


@lru_cache(maxsize=None)
def picture(which: str) -> np.ndarray:
    """The real picture at 900 x 900: 'red' (frame 0) or 'labels' (frame 74)."""
    from PIL import Image

    name = {"red": "frame_000.png", "labels": "frame_074.png"}[which]
    return np.asarray(Image.open(ASSETS / "ch05_input_blobs" / name).convert("RGB"))


def floor_raster(k: int, rpx: int) -> np.ndarray:
    """The floor after beat k-1."""
    t = toys()
    if k <= 0:
        return fl.toy_raster(np.tile(fl.rgb(fl.EMPTY), (fl.N, fl.N, 1)), rpx)
    if k == 1:
        return fl.toy_raster(level_colors(t["blob"], np.full(t["blob"].shape, -1)), rpx)
    if k == 2:
        return mid_wave(rpx)
    if k == 3:
        return fl.toy_raster(level_colors(t["blob"], t["blob4"].level), rpx)
    if k == 4:
        return fl.toy_raster(level_colors(t["snake"], np.full(t["snake"].shape, -1)), rpx)
    if k == 5:
        return fl.toy_raster(level_colors(t["snake"], t["snake4"].level), rpx)
    if k == 6:
        return fl.toy_raster(level_colors(t["square"], t["square8"].level), rpx)
    if k == 7:
        return fl.picture_raster(filled_picture(), rpx)
    if k == 8:
        return fl.toy_raster(label_colors(t["two"], t["two_lab"]), rpx)
    if k == 9:
        return fl.picture_raster(picture("labels"), rpx)
    return fl.picture_raster(teal_picture(), rpx)


L_MID = 5     # c01 stops the serial walk after this ring; c02's team takes the next one


def mid_wave(rpx: int) -> np.ndarray:
    """The blob painted up to ring L_MID, the next ring outlined with a worker on each cell."""
    t = toys()
    lv = t["blob4"].level
    edge = lv == L_MID + 1
    return fl.toy_raster(level_colors(t["blob"], lv, upto=L_MID), rpx, dots=edge.astype(int), ring=edge)


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
def zoom_target() -> tuple[float, float, float]:
    """(row, col, side) of the window around a mid-sized blob near the centre:
    where c00 zooms in and c06 zooms out."""
    lab, sizes = picture_blobs()
    H, W = lab.shape
    best = None
    for k, n in enumerate(sizes):
        if not 150 <= n <= 600:
            continue
        rs, cs = np.nonzero(lab == k)
        r, c = rs.mean(), cs.mean()
        d = (r - H / 2) ** 2 + (c - W / 2) ** 2
        side = max(rs.max() - rs.min(), cs.max() - cs.min()) * 1.5 + 6
        if best is None or d < best[0]:
            best = (d, r, c, side)
        if d < (H / 8) ** 2:
            break
    _, r, c, side = best
    return float(r), float(c), float(side)


def zoom_crop(img: np.ndarray, r: float, c: float, side: float) -> np.ndarray:
    """A square window of `img` around (r, c), clamped to the picture."""
    H, W = img.shape[:2]
    side = int(max(4, min(side, H)))
    r0 = int(min(max(r - side / 2, 0), H - side))
    c0 = int(min(max(c - side / 2, 0), W - side))
    return img[r0:r0 + side, c0:c0 + side]


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
    """ch05's label colours pulled towards teal: the same blobs, the final answer."""
    lab = picture("labels").astype(np.float32)
    blob = lab.min(axis=2) < 250
    out = lab.copy()
    out[blob] = lab[blob] * 0.45 + fl.rgb(TEAL).astype(np.float32) * 0.55
    return out.astype(np.uint8)


# ---- the machine between beats ----

ALL_LIT = ("lit",) * 24


def machine_spec(k: int) -> mc.Spec | None:
    """The board after beat k-1 (None: no board yet)."""
    if k <= 1:
        return None
    if k == 2:
        return mc.Spec(cpu=True)
    if k == 3:
        rooms = ("lit",) + ("idle",) * 23
        return mc.Spec(rooms=rooms, teams=(1,) + (0,) * 23, shelf=True, room_fence="rest")
    base = mc.Spec(rooms=ALL_LIT, teams=(2,) * 24, queue=True)
    if k == 4:
        return base
    base = replace(base, gate="rest")
    if k == 5:
        return base
    base = replace(base, stencil=8)
    if k == 6:
        return base
    n = nb.load()
    if k == 7:
        return replace(base, launches=f"{int(n['c06_calls'].value):,}")
    if k == 8:
        return replace(base, launches=f"{int(n['c07_calls'].value):,}", printer=True,
                       tokens=(RED_PX,) * 6, tag_tokens=(BLUE, GREEN, BLUE, BLUE, GREEN, BLUE))
    if k == 9:
        return replace(base, launches="1", min_unit=True, tokens=(RED_PX,) * mc.SLOTS)
    return mc.Spec(rooms=ALL_LIT, teams=(2,) * 24, stencil=8, min_unit=True, kernels=True)


# ---- the goal chip ----

def goal_value(k: int) -> tuple[str, str] | None:
    """(text, colour) of the real picture's current time after beat k-1."""
    n = {key: v.value for key, v in nb.load().items()}
    if k <= 1:
        return None
    if k <= 6:
        return fmt_ms(n["njit_ms"]), GOLD
    if k == 7:
        return "~" + fmt_ms(n["c06_ms"]), GOLD
    if k == 8:
        return "~" + fmt_ms(n["c07_ms"]), GOLD
    if k == 9:
        return fmt_ms(n["ch05_ms"]), GOLD
    return fmt_ms(n["ch06_mask_ms"]), TEAL


def fmt_ms(ms: float) -> str:
    """1,346 ms / 58.5 ms / 1.46 ms: never rounded up past what is spoken."""
    if ms >= 100:
        return f"{ms:,.0f} ms"
    if ms >= 10:
        return f"{int(ms * 10) / 10:.1f} ms"
    return f"{ms:.2f} ms"


def goal_chip(G: Frame, value: tuple[str, str] | None) -> VGroup:
    """'one frame 33 ms' and, once known, 'real picture <value>'."""
    left = VGroup(label("one frame", 22, INK_SOFT), label("33 ms", 34, INK, mono=True)).arrange(DOWN, buff=0.06)
    parts = [left]
    if value is not None:
        right = VGroup(label("real picture", 22, INK_SOFT), label(value[0], 34, value[1], mono=True)).arrange(DOWN, buff=0.06)
        parts.append(right)
    row = VGroup(*parts).arrange(RIGHT, buff=0.45)
    box = RoundedRectangle(corner_radius=0.12, width=max(row.width + 0.5, 2.0), height=row.height + 0.3,
                           stroke_color=GRID, stroke_width=2, fill_color="#0f141a", fill_opacity=1)
    row.move_to(box)
    g = VGroup(box, row).move_to(G.goal_c)
    right = config.frame_width / 2 - 0.3          # never past the frame's right edge
    if g.get_right()[0] > right:
        g.shift(LEFT * (g.get_right()[0] - right))
    g.set_z_index(Z_OVER)
    for m in g.get_family():
        m.set_z_index(Z_OVER + 1)
    box.set_z_index(Z_OVER)
    return g


# ---- the caption carried over a cut ----

def carry_caption(k: int) -> tuple[str, str | None] | None:
    """The last caption of beat k-1, still on screen when beat k starts.
    Beats often end on their punchline, so the caption crosses the cut and
    the next beat's first caption replaces it. Each scene's last `say` is
    `say_last()`, which shows exactly this."""
    n = {key: v.value for key, v in nb.load().items()}
    d = nb.derived()
    table = {
        1: ("33 ms", "one camera frame at 30 fps"),
        2: ("1 ring", "every cell on it is independent"),
        3: (f"1 of {int(n['sm'])}", f"{d['c02_sm_pct']:.0f}% of the GPU"),
        4: (f"{int(n['c04_blocks'])} lose to 1", f"snake: {n['c04_n_ms']:.0f} ms vs {n['c04_one_ms']:.0f} ms"),
        5: (f"{int(d['c04_gates']):,} gates", "~2 µs each, the 256 × 256 snake"),
        6: (f"{int(d['c05_x'])}×", "the CPU, one 64 Mpx blob"),
        7: ("1 blob", "a machine built for one"),
        8: ("seeds?", "handed in from outside; tags hold 64 labels"),
        9: (f"{int(n['red_px']):,}", "red pixels through the queue"),
        10: (fmt_ms(n["ch06_mask_ms"]), f"{int(d['c09_x'])}× the waves, real picture"),
    }
    return table.get(k)


# ---- the whole state ----

@dataclass
class State:
    floor: fl.Floor | None
    machine: mc.Machine | None
    goal: VGroup | None
    cap: VGroup | None = None

    def mobs(self) -> list:
        """Add order: machine, goal, floor. Drawing order is the z_index; the
        add order only decides what the renderer may cache during a play
        (everything before the first moving mobject), and the floor moves."""
        out = []
        if self.machine is not None:
            out += self.machine.mobs()
        if self.goal is not None:
            out.append(self.goal)
        if self.floor is not None:
            out.append(self.floor)
        if self.cap is not None:
            out.append(self.cap)
        return out


@dataclass
class Layout:
    G: Frame
    P: Proj
    floor_pl: object
    machine_pl: object


def layout() -> Layout:
    G = frame()
    P = proj()
    return Layout(G, P, fl.placement(P, G.floor_c, G.floor_wh), mc.placement(P, G.machine_c, G.machine_wh))


def chain_state(k: int, L: Layout | None = None) -> State:
    L = L or layout()
    if k <= 0:
        return State(None, None, None)
    f = fl.Floor(L.floor_pl, z0=Z_FLOOR)
    f.show_raster(floor_raster(k, f.rpx))
    spec = machine_spec(k)
    m = mc.build(spec, L.machine_pl, z0=Z_MACHINE) if spec is not None else None
    g = goal_chip(L.G, goal_value(k)) if k < len(BEATS) else None
    from scenes.chain.base import caption_group      # base imports this module
    cc = carry_caption(k)
    cap = caption_group(L.G, *cc) if cc else None
    return State(f, m, g, cap)
