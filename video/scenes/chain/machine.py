"""The machine: a GPU board that gains a part with every fix.

`Spec` says which parts are on the board; `build(spec, pl)` draws it.
Every face's `z_index` comes from its own layer and depth only (never
from the other parts in the build), so a part kept from one build and a
part from the next one still draw in the right 3D order. `Machine.parts[name]` holds a part's faces and labels; a
scene adds every part as its own top-level mobject (`mobs()`), because
FadeOut on a member of a group restores it afterwards.

World layout (x right, y away from the viewer, z up), board 7.6 x 5.6:

    back   +----------------------------------+------------+
           | chip: 24 rooms (SMs), 6 x 4      | stencil    |
           |                                  | launches   |
           +----------------------------------+ min        |
    front  | CPU | queue conveyor / kernel strip           |
           +-----------------------------------------------+
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from manim import VGroup

from scenes.chain.iso import Box, Placement, box_faces, on_plane
from scenes.style import BLUE, FONT, GOLD, GREY, INK, INK_SOFT, MONO, RED_PX, TEAL, label

# ---- colours (built from style.py's palette) ----
BOARD = "#121820"
CHIP = "#1b222b"
ROOM_OFF = "#262d36"
ROOM_IDLE = "#424b57"
ROOM_LIT = "#22405f"
ROOM_WAIT = "#5a4519"
FENCE_REST = "#4d3c17"
SLOT = "#2a313a"
KERNEL = "#1d5f59"
CPU_BOX = "#39414c"

# ---- world geometry ----
BOARD_LO, BOARD_HI = (-3.8, -2.9, -0.12), (3.8, 2.7, 0.0)
CHIP_LO, CHIP_HI = (-3.5, -0.75, 0.0), (1.5, 2.45, 0.18)
ROOM, GAP = 0.66, 0.14
COLS, ROWS = 6, 4
ROOM_Z0, ROOM_H = 0.18, 0.10
STRIP_Y = (-2.55, -1.45)            # the front strip (CPU, queue, kernels)
SLOTS = 12
SLOT_X0, SLOT_W, SLOT_GAP = -2.8, 0.30, 0.06
SIDE_X = (1.95, 3.55)               # the right column of badges
EXTENT_HI_Z = 0.7                   # tallest part, for a fixed placement


def room_xy(i: int) -> tuple[float, float]:
    """Min corner of room i (row-major from the back-left)."""
    gw = COLS * ROOM + (COLS - 1) * GAP
    gd = ROWS * ROOM + (ROWS - 1) * GAP
    cx, cy = (CHIP_LO[0] + CHIP_HI[0]) / 2, (CHIP_LO[1] + CHIP_HI[1]) / 2
    col, row = i % COLS, i // COLS
    x = cx - gw / 2 + col * (ROOM + GAP)
    y = cy + gd / 2 - ROOM - row * (ROOM + GAP)
    return x, y


def slot_x(j: int) -> float:
    return SLOT_X0 + j * (SLOT_W + SLOT_GAP)


@dataclass(frozen=True)
class Spec:
    rooms: tuple = ("off",) * 24        # off | idle | lit | wait
    teams: tuple = (0,) * 24            # teams per room: 0, 1 or 2
    team_state: str = "work"            # work | wait (the dots' colour)
    cpu: bool = False
    cpu_state: str = "work"             # work | wait
    shelf: bool = False                 # the shared-memory ring in room 0
    room_fence: str | None = None       # None | rest | wait (syncthreads, room 0)
    queue: bool = False                 # the global queue along the front
    tokens: tuple = ()                  # token colours, slot 0 first
    tag_tokens: tuple = ()              # a tag colour per token (c07)
    gate: str | None = None             # None | rest | wait (grid.sync, the chip)
    stencil: int = 0                    # 0 | 4 | 8
    launches: str | None = None         # the counter's text
    printer: bool = False               # the tag printer (c07)
    min_unit: bool = False              # atomicMin (c08)
    kernels: bool = False               # the ch06 kernel strip (c09)
    kernel_lit: int = -1
    seams: bool = False                 # 5 thin seams between the kernels (what is left of the gate)
    rack: bool = False                  # the tag rack, "64 max" (v2 r08)


@dataclass
class Machine:
    spec: "Spec"
    parts: dict = field(default_factory=dict)       # name -> VGroup, each face in one part

    def part(self, name: str) -> VGroup:
        return self.parts.get(name, VGroup())

    def mobs(self) -> list:
        return list(self.parts.values())

    @property
    def group(self) -> VGroup:
        return VGroup(*self.parts.values())

    def rooms(self) -> VGroup:
        return VGroup(*[v for k, v in self.parts.items() if k.startswith("room") and k[4:].isdigit()])


def placement(P, centre, size) -> Placement:
    return Placement.fit(P, BOARD_LO, (BOARD_HI[0], BOARD_HI[1], EXTENT_HI_Z), centre, size)


def _boxes(spec: Spec) -> list[tuple[str, Box]]:
    """(part name, box) for everything on the board."""
    out: list[tuple[str, Box]] = []
    bx0, by0, bz0 = BOARD_LO
    bx1, by1, bz1 = BOARD_HI
    out.append(("board", Box(bx0, by0, bz0, bx1 - bx0, by1 - by0, bz1 - bz0, BOARD, layer=0)))
    cx0, cy0, cz0 = CHIP_LO
    cx1, cy1, cz1 = CHIP_HI
    out.append(("chip", Box(cx0, cy0, cz0, cx1 - cx0, cy1 - cy0, cz1 - cz0, CHIP, layer=1)))
    room_col = {"off": ROOM_OFF, "idle": ROOM_IDLE, "lit": ROOM_LIT, "wait": ROOM_WAIT}
    dot_col = {"work": BLUE, "wait": GOLD}[spec.team_state]
    for i in range(24):
        x, y = room_xy(i)
        out.append((f"room{i}", Box(x, y, ROOM_Z0, ROOM, ROOM, ROOM_H, room_col[spec.rooms[i]], layer=2)))
        n = spec.teams[i]
        if n:
            # 3 x 3 dots per team; two teams sit side by side
            cols = 3 * n
            pitch = (ROOM - 0.16) / cols
            for a in range(cols):
                for b in range(3):
                    dx = x + 0.08 + (a + 0.5) * pitch - 0.035
                    dy = y + 0.08 + (b + 0.5) * (ROOM - 0.16) / 3 - 0.035
                    out.append(("teams", Box(dx, dy, ROOM_Z0 + ROOM_H, 0.07, 0.07, 0.05, dot_col, layer=3)))
    if spec.shelf:
        # a ring of 8 slots in the back-right corner of room 0
        x, y = room_xy(0)
        for a in range(8):
            ang = 2 * np.pi * a / 8
            sx = x + ROOM - 0.17 + 0.11 * np.cos(ang) - 0.025
            sy = y + ROOM - 0.17 + 0.11 * np.sin(ang) - 0.025
            out.append(("shelf", Box(sx, sy, ROOM_Z0 + ROOM_H, 0.05, 0.05, 0.03, INK_SOFT, layer=3)))
    if spec.room_fence:
        x, y = room_xy(0)
        out += [("room_fence", b) for b in _fence(x - 0.05, y - 0.05, x + ROOM + 0.05, y + ROOM + 0.05,
                                                     ROOM_Z0, 0.22, spec.room_fence)]
    if spec.gate:
        out += [("gate", b) for b in _fence(cx0 - 0.04, cy0 - 0.04, cx1 + 0.04, cy1 + 0.04, 0.0, 0.42, spec.gate,
                                             thick=0.07)]
    sy0, sy1 = STRIP_Y
    if spec.cpu:
        col = BLUE if spec.cpu_state == "work" else GOLD
        out.append(("cpu", Box(-3.55, sy0 + 0.05, 0.0, 0.62, sy1 - sy0 - 0.1, 0.32, CPU_BOX, layer=2)))
        out.append(("cpu", Box(-3.29, (sy0 + sy1) / 2 - 0.05, 0.32, 0.1, 0.1, 0.06, col, layer=3)))
    if spec.queue or spec.cpu:
        part = "queue" if spec.queue else "cpu"
        for j in range(SLOTS):
            out.append((part, Box(slot_x(j), sy0 + 0.3, 0.0, SLOT_W, 0.5, 0.06, SLOT, layer=2)))
        for j, colr in enumerate(spec.tokens[:SLOTS]):
            out.append((part + "_tokens", Box(slot_x(j) + 0.05, sy0 + 0.38, 0.06, SLOT_W - 0.1, 0.34, 0.12, colr, layer=3)))
        for j, colr in enumerate(spec.tag_tokens[:SLOTS]):
            if colr:
                out.append(("tags", Box(slot_x(j) + 0.05, sy0 + 0.38, 0.18, SLOT_W - 0.1, 0.12, 0.08, colr, layer=4)))
    if spec.printer:
        out.append(("printer", Box(-3.55, sy0 + 0.1, 0.0, 0.62, sy1 - sy0 - 0.2, 0.5, "#3b3355", layer=2)))
    if spec.kernels:
        w, g = 0.74, 0.11
        x0 = -3.55
        for j in range(6):
            colr = TEAL if j == spec.kernel_lit else KERNEL
            out.append(("kernels", Box(x0 + j * (w + g), sy0 + 0.15, 0.0, w, sy1 - sy0 - 0.3, 0.22, colr, layer=2)))
        if spec.seams:
            for j in range(5):
                xs = x0 + (j + 1) * (w + g) - g / 2 - 0.015
                out.append(("seams", Box(xs, sy0 + 0.1, 0.0, 0.03, sy1 - sy0 - 0.2, 0.3, GREY, layer=3)))
    if spec.rack:
        # the rack never coexists with the min unit (tags leave when min arrives): same slot
        out.append(("rack", Box(SIDE_X[0] + 0.1, -1.35, 0.0, SIDE_X[1] - SIDE_X[0] - 0.2, 1.05, 0.1, "#3b3355", layer=2)))
    s0, s1 = SIDE_X
    if spec.stencil:
        out.append(("stencil", Box(s0, 1.3, 0.0, s1 - s0, 1.15, 0.08, CHIP, layer=2)))
        on = {(1, 1)} | {(1 + dr, 1 + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))}
        if spec.stencil == 8:
            on |= {(1 + dr, 1 + dc) for dr in (-1, 1) for dc in (-1, 1)}
        c = 0.24
        gx, gy = s0 + 0.22, 1.3 + (1.15 - 3 * c) / 2
        for r in range(3):
            for k in range(3):
                colr = BLUE if (r, k) == (1, 1) else ("#5d8fc0" if (r, k) in on else ROOM_OFF)
                h = 0.1 if (r, k) in on else 0.03
                out.append(("stencil", Box(gx + k * c + 0.02, gy + (2 - r) * c + 0.02, 0.08, c - 0.04, c - 0.04, h,
                                           colr, layer=3)))
    if spec.launches is not None:
        out.append(("launches", Box(s0, 0.0, 0.0, s1 - s0, 1.05, 0.08, CHIP, layer=2)))
    if spec.min_unit:
        out.append(("min", Box(s0 + 0.2, -1.3, 0.0, s1 - s0 - 0.4, 0.95, 0.38, "#3b3355", layer=2)))
    return out


def _fence(x0, y0, x1, y1, z0, h, state, thick=0.05) -> list[Box]:
    col = GOLD if state == "wait" else FENCE_REST
    t = thick
    return [
        Box(x0, y1 - t, z0, x1 - x0, t, h, col, layer=4),          # back
        Box(x0, y0, z0, t, y1 - y0, h, col, layer=4),              # left
        Box(x1 - t, y0, z0, t, y1 - y0, h, col, layer=4),          # right
        Box(x0, y0, z0, x1 - x0, t, h, col, layer=6),              # front, drawn last
    ]


def _labels(spec: Spec, pl: Placement) -> list[tuple[str, object, float]]:
    """(part, mobject, z for ordering) for text printed on parts."""
    out = []
    sy0, sy1 = STRIP_Y
    if spec.cpu:
        out.append(("cpu", on_plane(label("CPU", 30, INK), pl, (-3.24, sy1 - 0.2), 0.32, width=0.42), 0.4))
    if spec.queue:
        out.append(("queue", on_plane(label("queue", 26, INK_SOFT), pl, (slot_x(SLOTS - 1) + 0.72, sy0 + 0.55), 0.0,
                                      width=0.9), 0.1))
    if spec.printer:
        out.append(("printer", on_plane(label("tag", 30, INK), pl, (-3.24, (sy0 + sy1) / 2), 0.5, width=0.4), 0.6))
    if spec.kernels:
        w, g = 0.74, 0.11
        for j, name in enumerate(("count", "scan", "emit", "merge", "flatten", "paint")):
            # one text height for all six; "flatten" is the widest and sets it
            t = label(name, 26, INK)
            t.scale(0.68 / label("flatten", 26, INK).width)
            out.append(("kernels", on_plane(t, pl, (-3.55 + j * (w + g) + w / 2, (sy0 + sy1) / 2), 0.22), 0.3))
    s0, s1 = SIDE_X
    if spec.stencil:
        out.append(("stencil", on_plane(label(str(spec.stencil), 40, INK, mono=True), pl, (s1 - 0.33, 1.87), 0.08,
                                        width=0.22), 0.2))
    if spec.launches is not None:
        txt = VGroup(label(spec.launches, 34, INK, mono=True), label("launches", 24, INK_SOFT))
        txt.arrange(direction=np.array([0, -1, 0]), buff=0.08)
        out.append(("launches", on_plane(txt, pl, ((s0 + s1) / 2, 0.52), 0.08, width=1.2), 0.2))
    if spec.min_unit:
        out.append(("min", on_plane(label("min", 36, INK, mono=True), pl, ((s0 + s1) / 2, -0.83), 0.38, width=0.6), 0.5))
    if spec.rack:
        txt = VGroup(label("tags", 24, INK_SOFT), label("64 max", 34, INK, mono=True))
        txt.arrange(direction=np.array([0, -1, 0]), buff=0.06)
        out.append(("rack", on_plane(txt, pl, ((s0 + s1) / 2, -0.82), 0.1, width=1.0), 0.2))
    return out


def _depth_range(pl: Placement) -> tuple[float, float]:
    lo, hi = BOARD_LO, (BOARD_HI[0], BOARD_HI[1], EXTENT_HI_Z)
    ds = [pl.P.depth((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return min(ds), max(ds)


def build(spec: Spec, pl: Placement, z0: float = 100.0) -> Machine:
    """Draw the board for `spec`. A face's z_index is z0 + 10 * layer + its
    nearness within the board (0..9), and labels sit at z0 + 95."""
    boxes = _boxes(spec)
    dmin, dmax = _depth_range(pl)
    parts: dict[str, VGroup] = {}
    for name, _ in boxes:                        # part order = first appearance, stable
        parts.setdefault(name, VGroup())
    for name, b in boxes:
        near = 9.0 * (dmax - pl.P.depth(b.centre)) / (dmax - dmin)
        for j, face in enumerate(box_faces(b, pl)):
            face.set_z_index(z0 + 10 * b.layer + near + 0.001 * j)
            parts[name].add(face)
    for name, mob, z in _labels(spec, pl):
        for m in mob.get_family():
            m.set_z_index(z0 + 95 + z)
        parts.setdefault(name, VGroup()).add(mob)
    return Machine(spec, parts)


def with_rooms(spec: Spec, state: str, rooms=None) -> Spec:
    """A copy of spec with the given rooms (default: every room) set to state."""
    r = list(spec.rooms)
    for i in (range(24) if rooms is None else rooms):
        r[i] = state
    return replace(spec, rooms=tuple(r))
