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
from manim import DOWN, LEFT, RIGHT, UP, Group, RoundedRectangle, VGroup

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


def level_colors(m: np.ndarray, level: np.ndarray, upto: int | None = None) -> np.ndarray:
    """Cells: empty off the mask, red if not reached (by `upto`), else a ring shade."""
    n = m.shape[0]
    out = np.tile(fl.rgb(fl.EMPTY), (n, n, 1))
    out[m] = fl.rgb(RED_PX)
    top = max(int(level.max()), 1)
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
    if k in (2, 3):
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
        return replace(base, launches="1", min_unit=True)
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
    if ms >= 100:
        return f"{ms:,.0f} ms"
    if ms >= 10:
        return f"{ms:.0f} ms"
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
    g.set_z_index(Z_OVER)
    for m in g.get_family():
        m.set_z_index(Z_OVER + 1)
    box.set_z_index(Z_OVER)
    return g


# ---- the whole state ----

@dataclass
class State:
    floor: fl.Floor | None
    machine: mc.Machine | None
    goal: VGroup | None

    def mobs(self) -> list:
        out = []
        if self.floor is not None:
            out.append(self.floor)
        if self.machine is not None:
            out.append(self.machine.group)
        if self.goal is not None:
            out.append(self.goal)
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
    return State(f, m, g)
