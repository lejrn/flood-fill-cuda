"""The picture between beats of the chain cut v2: `chain_state(k)` is what
the screen shows when beat k starts (k = index into BEATS). The intro clip
voices beats 0 and 1, so the first chain state is k = 2. Scene k starts on
`chain_state(k)` and snaps to `chain_state(k + 1)` at its end, so its
last frame is the next clip's first frame (`build/seam_check.py --cut
chain` verifies this).

Everything here is a pure function of k, the look and the layout; the
numbers come from `numbers.py`.
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
from scenes.style import ASSETS, BLUE, GOLD, GREEN, PURPLE, RED_PX, TEAL

BEATS = ("i0_problem", "i1_compare", "r00_zoom", "r01_cpu", "r02_one_block", "r03_giant",
         "r04_two_blocks", "r05_all_blocks", "r06_diagonals", "r07_real_frame", "r08_tags",
         "r09_seeds", "r10_runs", "r11_reveal")
FIRST = 2                                   # the first beat with its own chain clip

Z_FLOOR, Z_MACHINE, Z_OVER = 10.0, 100.0, ov.Z_OVER

# ---- colours of painted cells ----
LEVEL_LO, LEVEL_HI = "#9cc3ec", "#1f4f86"      # a wave's rings, first to last
BLOB_HUES = (BLUE, GREEN, "#d36ac2", "#e0a13a", "#5fc9d6", PURPLE)
fmt_ms = ov.fmt_ms


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


@lru_cache(maxsize=None)
def picture(which: str) -> np.ndarray:
    """The real picture at 900 x 900: 'red' (frame 0) or 'labels' (frame 74)."""
    from PIL import Image

    name = {"red": "frame_000.png", "labels": "frame_074.png"}[which]
    return np.asarray(Image.open(ASSETS / "ch05_input_blobs" / name).convert("RGB"))


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
    """ch05's label colours pulled towards teal: the same blobs, the final answer."""
    lab = picture("labels").astype(np.float32)
    blob = lab.min(axis=2) < 250
    out = lab.copy()
    out[blob] = lab[blob] * 0.45 + fl.rgb(TEAL).astype(np.float32) * 0.55
    return out.astype(np.uint8)


def nums() -> dict:
    return {k: v.value for k, v in nb.load().items()}


# ---- the scoreboard rows ----

def score_rows() -> tuple:
    n, d = nums(), nb.derived()
    small = ov.Row("one small blob", n["small_cpu_ms"], n["small_gpu_ms"], "CPU", f"{d['small_cpu_x']:.1f}×",
                   "16 K px test blob", "one session")
    huge = ov.Row("one huge blob", n["big_cpu_ms"], n["big_8ways_ms"], "GPU", f"{d['big_8ways_x']:.1f}×",
                  "64 Mpx test blob", "one session, kernel time")
    many = ov.Row("many blobs", n["count_cpu_ms"], n["count_gpu_ms"], "GPU", f"{d['count_x']:.0f}×",
                  f"1 Mpx, {int(n['count_blobs']):,} blobs", "two sessions")
    return small, huge, many


# ---- the boundary table ----

@dataclass(frozen=True)
class Boundary:
    floors: tuple = ()                 # ("drone_red",), ("drone_cpu", "drone_gpu") for a split
    machine: mc.Spec | None = None
    goal: ov.GoalSpec | None = None
    rows: tuple = (None, None, None)   # scoreboard: Row, "?" or None per row
    gauge: int | None = None           # SMs busy
    gauge_note: str = ""
    inset: tuple | None = None         # (line1, line2)
    tag: tuple | None = None           # (act, won)
    cap: tuple | None = None           # (text, sub): the caption crossing the cut


ALL_LIT = ("lit",) * 24


def boundary(k: int) -> Boundary:
    """The state when beat k starts (k >= FIRST)."""
    n, d = nums(), nb.derived()
    small, huge, many = score_rows()
    goal = ov.GoalSpec(cpu_tick=True)
    rows = (None, "?", "?")
    b = Boundary(goal=goal, rows=rows)
    if k <= 2:
        return b
    b = replace(b, floors=("drone_red",), inset=("you are here", "one drone, one frame"),
                cap=(f"{int(toys()['drone'].sum())} pixels", "one drone"))
    if k == 3:
        return b
    b = replace(b, floors=("drone_cpu",), machine=mc.Spec(cpu=True),
                cap=(f"{int(n['sm'])} SMs", "streaming multiprocessors"))
    if k == 4:
        return b
    one = mc.Spec(cpu=True, rooms=("lit",) + ("off",) * 23, teams=(1,) + (0,) * 23)
    b = replace(b, floors=("drone_cpu", "drone_gpu"), machine=one, rows=(small, "?", "?"),
                cap=("CPU wins", f"16 K px test blob: CPU {fmt_ms(n['small_cpu_ms'])}, "
                                 f"best GPU {fmt_ms(n['small_gpu_ms'])}"))
    if k == 5:
        return b
    b = replace(b, floors=("square4",),
                machine=mc.Spec(rooms=("lit",) + ("off",) * 23, teams=(1,) + (0,) * 23, shelf=True,
                                room_fence="rest"),
                gauge=1, inset=("not a drone:", "a stress test"), tag=("SIZE", False),
                cap=("1 of 24", "one block, one SM"))
    if k == 6:
        return b
    b = replace(b, machine=mc.Spec(rooms=("lit", "lit") + ("off",) * 22, teams=(1, 1) + (0,) * 22,
                                   queue=True, gate="rest"),
                gauge=2, cap=("22 dark", "22 of 24 SMs idle"))
    if k == 7:
        return b
    full = mc.Spec(rooms=ALL_LIT, teams=(2,) * 24, queue=True, gate="rest")
    b = replace(b, floors=("snake",), machine=full, gauge=24, gauge_note="1/3 of each SM: registers",
                cap=(f"{int(d['snake_stops']):,} stops", "~2 µs each, the 256 × 256 snake"))
    if k == 8:
        return b
    b = replace(b, floors=("square8",), machine=replace(full, stencil=8), rows=(small, huge, "?"),
                tag=("SIZE", True), cap=(f"{d['big_8ways_x']:.1f}×", "the CPU, one 64 Mpx blob, kernel time"))
    if k == 9:
        return b
    m9 = replace(full, stencil=8, launches=f"{int(n['est_ch03_calls']):,}")
    b = replace(b, floors=("frame_filled",), machine=m9,
                goal=ov.GoalSpec(needle_ms=n["est_ch03_ms"], needle_text="~" + fmt_ms(n["est_ch03_ms"])),
                tag=("COUNT", False), inset=("a drone frame:", f"~{int(n['drone_blobs_median']):,} blobs"),
                cap=("~" + fmt_ms(n["est_ch03_ms"]), "estimate: one launch per blob"))
    if k == 10:
        return b
    m10 = replace(m9, launches=f"{int(n['est_ch04_calls']):,}", printer=True, rack=True,
                  tokens=(RED_PX,) * 6, tag_tokens=(BLUE, GREEN, BLUE, BLUE, GREEN, BLUE))
    b = replace(b, floors=("two",), machine=m10,
                goal=ov.GoalSpec(needle_ms=n["est_ch04_ms"], needle_text="~" + fmt_ms(n["est_ch04_ms"])),
                cap=("64 tags", "6 spare bits per queue entry"))
    if k == 11:
        return b
    m11 = replace(full, stencil=8, launches="3", min_unit=True, tokens=(RED_PX,) * mc.SLOTS)
    b = replace(b, floors=("frame_labels",), machine=m11,
                goal=ov.GoalSpec(needle_ms=n["ch05_ms"], needle_text=fmt_ms(n["ch05_ms"])),
                cap=(fmt_ms(n["ch05_ms"]), "still over 33 ms"))
    if k == 12:
        return b
    m12 = mc.Spec(rooms=ALL_LIT, teams=(2,) * 24, stencil=8, min_unit=True, kernels=True, seams=True)
    return replace(b, floors=("frame_teal",), machine=m12,
                   goal=ov.GoalSpec(needle_ms=n["ch06_rgb_ms"], needle_text=fmt_ms(n["ch06_rgb_ms"]),
                                    needle_color=TEAL, mask_mark=True),
                   rows=(small, huge, many), tag=("COUNT", True),
                   cap=(fmt_ms(n["ch06_rgb_ms"]), f"from the RGB image; {fmt_ms(n['ch06_mask_ms'])} from a packed mask"))


# ---- floor rasters ----

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
    if name == "two":
        return fl.toy_raster(label_colors(t["two"], t["two_lab"]), rpx)
    if name == "frame_filled":
        return fl.picture_raster(filled_picture(), rpx)
    if name == "frame_labels":
        return fl.picture_raster(picture("labels"), rpx)
    if name == "frame_teal":
        return fl.picture_raster(teal_picture(), rpx)
    raise KeyError(name)


# ---- the whole state ----

@dataclass
class State:
    floors: list = field(default_factory=list)
    machine: mc.Machine | None = None
    over: dict = field(default_factory=dict)       # goal, strip, gauge, inset, tag
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
         "gauge": ov.room_gauge(G, b.gauge, int(n["sm"]), b.gauge_note) if b.gauge else None,
         "inset": ov.inset(G, *b.inset) if b.inset else None,
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
    return boundary(k).cap if k >= FIRST else None
