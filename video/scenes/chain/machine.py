"""The machine: a CPU board and a GPU board side by side, drawn like the
vendors' block diagrams (`die.py`), with the parts each fix adds.

`Spec` says what is lit and what is on the boards; `build(spec, pl)`
draws it. Every part has a key (what it shows): `ChainScene.machine_to`
keeps a part whose key did not change, cross-fades one whose key did,
and slides a new one in. A face's or image's `z_index` comes from its own
layer and depth only, so parts from two builds still draw in 3D order.

World layout (x right, y away from the viewer, z up):

    +-- CPU board --+  link  +----------- GPU board ------------+
    |  i9-13900H    | ====== |  RTX 4060 Laptop (AD107)          |
    |  die          | kernels|  die: 3 GPCs, 24 SMs, L2           |
    |  RAM  RAM     | launch |  VRAM x 4 (the queue lives here)   |
    |  CPU label    |        |  GPU label                         |
    +---------------+        +-----------------------------------+
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from manim import DOWN, Group, ImageMobject, RoundedRectangle, VGroup, config
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image

from scenes.chain import die as dz
from scenes.chain.iso import Box, Placement, box_faces, on_plane, plane_corners
from scenes.style import BLUE, GOLD, GREY, INK, INK_SOFT, RED_PX, TEAL, label

# ---- colours ----
BOARD = "#10161c"
SUBSTRATE = "#1b2129"
CHIP_EDGE = "#2a313a"
MEM = "#0c1014"
SLOT = "#232a33"
KERNEL = "#1d5f59"
LINK = "#262d36"
FENCE_REST = "#3a4652"       # a fence at rest is neutral: amber means only "waiting"

# ---- world geometry ----
BM = 0.30                    # board margin around a die
FRONT = 0.42                 # the silkscreen strip at a board's front
CHIP_Y, CHIP_D = FRONT, 0.60  # RAM / VRAM row
DIE_Y = CHIP_Y + CHIP_D + 0.30
BOARD_D = DIE_Y + dz.GPU_D + BM
CPU_X = 0.0
CPU_BW = dz.CPU_W + 2 * BM
GAP = 0.90
GPU_X = CPU_BW + GAP
GPU_BW = dz.GPU_W + 2 * BM
BOARD_Z = -0.12
SUB_H, DIE_H = 0.06, 0.06
DIE_Z = SUB_H + DIE_H        # the chips' top faces
EXTENT_HI_Z = 0.6
WORLD_LO = (CPU_X, 0.0, BOARD_Z)
WORLD_HI = (GPU_X + GPU_BW, BOARD_D, EXTENT_HI_Z)

CPU_DIE = (CPU_X + BM, DIE_Y)                 # front-left corner of each die
GPU_DIE = (GPU_X + BM, DIE_Y)
SLOTS = 12                                    # queue slots, 3 per VRAM chip
N_VRAM, N_RAM = 4, 2
LINK_Y = (DIE_Y + dz.GPU_D - 1.25, DIE_Y + dz.GPU_D - 0.85)
KERNEL_W, KERNEL_D, KERNEL_GAP = 0.60, 0.40, 0.09


def sm_rect(i: int) -> dz.R:
    """SM i in world units (x, y of its front-left corner, size)."""
    return dz.gpu_plan().sms[i].shift(*GPU_DIE)


def pcore_rect(j: int) -> dz.R:
    return dz.cpu_plan().pcores[j].shift(*CPU_DIE)


def chip_rects(x0: float, w: float, n: int, gap: float = 0.22) -> list[dz.R]:
    cw = (w - (n - 1) * gap) / n
    return [dz.R(x0 + j * (cw + gap), CHIP_Y, cw, CHIP_D) for j in range(n)]


def vram_rects() -> list[dz.R]:
    return chip_rects(GPU_DIE[0], dz.GPU_W, N_VRAM)


def ram_rects() -> list[dz.R]:
    return chip_rects(CPU_DIE[0], dz.CPU_W, N_RAM)


def slot_rect(j: int) -> dz.R:
    chip = vram_rects()[j // 3]
    w = (chip.w - 0.16) / 3
    return dz.R(chip.x + 0.06 + (j % 3) * (w + 0.02), chip.y + 0.1, w - 0.02, chip.d - 0.2)


def kernel_rect(j: int) -> dz.R:
    x = CPU_BW + (GAP - KERNEL_W) / 2
    y = LINK_Y[0] - 0.25 - (j + 1) * KERNEL_D - j * KERNEL_GAP
    return dz.R(x, y, KERNEL_W, KERNEL_D)


@dataclass(frozen=True)
class Spec:
    cpu: str = "off"                    # P-core 0: off | lit (our CPU code runs on one core)
    rooms: tuple = ("off",) * 24        # SMs: off | idle | lit | wait
    teams: tuple = (0,) * 24            # blocks resident per SM: 0, 1 or 2
    team_state: str = "work"            # work | wait (the blocks' colour)
    shelf: bool = False                 # SM 0's shared memory holds the queue (one block)
    room_fence: str | None = None       # None | rest | wait: __syncthreads around SM 0
    queue: bool = False                 # the global queue, in VRAM
    tokens: tuple = ()                  # token colours, slot 0 first
    tag_tokens: tuple = ()              # a tag colour per token
    gate: str | None = None             # None | rest | wait: grid.sync around the GPU
    launches: str | None = None         # the launch counter on the link
    kernels: bool = False               # the six ch06 kernels, queued on the link
    kernel_lit: int = -1
    seams: bool = False                 # the kernel boundaries: all that is left of the gate


@dataclass
class Machine:
    spec: "Spec"
    parts: dict = field(default_factory=dict)       # name -> Group
    keys: dict = field(default_factory=dict)        # name -> what the part shows

    def part(self, name: str) -> Group:
        return self.parts.get(name, Group())

    def mobs(self) -> list:
        return list(self.parts.values())

    @property
    def group(self) -> Group:
        return Group(*self.parts.values())

    def rooms(self) -> Group:
        return Group(*[v for k, v in self.parts.items() if k.startswith("room") and k[4:].isdigit()])


def placement(P, centre, size) -> Placement:
    return Placement.fit(P, WORLD_LO, WORLD_HI, centre, size)


# ---- images on the plane -------------------------------------------------

def _ppu() -> float:
    return config.pixel_width / config.frame_width


def _pl_key(pl: Placement) -> tuple:
    return (pl.P.yaw, pl.P.elev, round(pl.scale, 6), tuple(np.round(pl.origin, 6)))


def raster_ppw(pl: Placement) -> float:
    """Raster pixels per world unit: a little above the output's."""
    return round(pl.scale * _ppu() * 1.3, 1)


_WARPS: dict = {}


def plane_image(raster: np.ndarray, pl: Placement, r: dz.R, z: float, key) -> ImageMobject:
    """`raster` (row 0 = the back edge) laid on the rectangle r at height z.
    Cairo ignores image shear, so the raster is warped here with one affine
    transform (as `floor.Warp` does) and drawn upright."""
    k = _ppu()
    ck = (key, r, z, _pl_key(pl))
    if ck not in _WARPS:
        ul, ur, dl, _ = plane_corners(pl, r.x, r.y, r.x1, r.y1, z)
        pts = np.array([ul, ur, dl, ur + dl - ul])[:, :2]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        w, h = int(np.ceil((hi[0] - lo[0]) * k)), int(np.ceil((hi[1] - lo[1]) * k))
        centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
        rh, rw = raster.shape[:2]
        if pl.P.flat:
            arr = np.asarray(Image.fromarray(raster, "RGBA").resize((w, h), Image.LANCZOS)).copy()
        else:
            A = np.column_stack([(ur - ul)[:2] / rw, (dl - ul)[:2] / rh])
            Ai = np.linalg.inv(A)
            x0 = lo[0] - ul[0] + 0.5 / k
            y0 = hi[1] - ul[1] - 0.5 / k
            co = (Ai[0, 0] / k, -Ai[0, 1] / k, Ai[0, 0] * x0 + Ai[0, 1] * y0,
                  Ai[1, 0] / k, -Ai[1, 1] / k, Ai[1, 0] * x0 + Ai[1, 1] * y0)
            arr = np.asarray(Image.fromarray(raster, "RGBA").transform(
                (w, h), Image.AFFINE, co, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))).copy()
        _WARPS[ck] = (arr, w, h, centre)
    arr, w, h, centre = _WARPS[ck]
    im = ImageMobject(arr)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    im.stretch_to_fit_width((w + 0.5) / k)
    im.stretch_to_fit_height((h + 0.5) / k)
    im.move_to(centre)
    return im


# ---- the parts -----------------------------------------------------------

def _boxes(spec: Spec) -> list[tuple[str, Box]]:
    """(part name, box) for every solid part."""
    out: list[tuple[str, Box]] = []
    for x0, w in ((CPU_X, CPU_BW), (GPU_X, GPU_BW)):
        out.append(("boards", Box(x0, 0.0, BOARD_Z, w, BOARD_D, -BOARD_Z, BOARD, layer=0)))
    out.append(("boards", Box(CPU_BW - 0.05, LINK_Y[0], BOARD_Z + 0.02, GAP + 0.1, LINK_Y[1] - LINK_Y[0], 0.04,
                              LINK, layer=0)))
    for (dx, dy), W in ((CPU_DIE, dz.CPU_W), (GPU_DIE, dz.GPU_W)):
        out.append(("boards", Box(dx - 0.12, dy - 0.12, 0.0, W + 0.24, dz.GPU_D + 0.24, SUB_H, SUBSTRATE, layer=1)))
        out.append(("boards", Box(dx, dy, SUB_H, W, dz.GPU_D, DIE_H, dz.DIE, layer=1)))
    for r in ram_rects() + vram_rects():
        out.append(("boards", Box(r.x, r.y, 0.0, r.w, r.d, 0.08, MEM, layer=1)))
    blue_or_gold = {"work": BLUE, "wait": GOLD}[spec.team_state]
    for i in range(24):
        n = spec.teams[i]
        if not n:
            continue
        r = sm_rect(i)
        bw = 0.20 if n == 1 else 0.14
        xs = [r.x + (r.w - bw) / 2] if n == 1 else [r.x + 0.04, r.x + r.w - 0.04 - bw]
        for x in xs:
            out.append(("teams", Box(x, r.y + 0.16, DIE_Z, bw, 0.24, 0.10, blue_or_gold, layer=4)))
    if spec.room_fence:
        r = sm_rect(0)
        out += [("room_fence", b) for b in _fence(r.x - 0.03, r.y - 0.03, r.x1 + 0.03, r.y1 + 0.03,
                                                     DIE_Z, 0.18, spec.room_fence)]
    if spec.gate:
        gx, gy = GPU_DIE
        # at rest the gate is a low rail; when the blocks wait at it, it rises, amber
        h = 0.30 if spec.gate == "wait" else 0.07
        out += [("gate", b) for b in _fence(gx - 0.16, gy - 0.16, gx + dz.GPU_W + 0.16, gy + dz.GPU_D + 0.16, 0.0,
                                             h, spec.gate, thick=0.04)]
    if spec.queue:
        for j in range(SLOTS):
            s = slot_rect(j)
            out.append(("queue", Box(s.x, s.y, 0.08, s.w, s.d, 0.02, SLOT, layer=2)))
        for j, colr in enumerate(spec.tokens[:SLOTS]):
            s = slot_rect(j).inset(0.025, 0.04)
            out.append(("queue_tokens", Box(s.x, s.y, 0.10, s.w, s.d, 0.10, colr, layer=4)))
        for j, colr in enumerate(spec.tag_tokens[:SLOTS]):
            if colr:
                s = slot_rect(j).inset(0.025, 0.04)
                out.append(("tags", Box(s.x, s.y + s.d * 0.55, 0.20, s.w, s.d * 0.45, 0.06, colr, layer=5)))
    if spec.kernels:
        for j in range(6):
            r = kernel_rect(j)
            colr = TEAL if j == spec.kernel_lit else KERNEL
            out.append(("kernels", Box(r.x, r.y, BOARD_Z + 0.02, r.w, r.d, 0.16, colr, layer=2)))
        if spec.seams:
            for j in range(5):
                r = kernel_rect(j)
                out.append(("seams", Box(r.x - 0.04, r.y - KERNEL_GAP / 2 - 0.012, BOARD_Z + 0.02, r.w + 0.08, 0.024,
                                         0.2, GREY, layer=3)))
    return out


def _fence(x0, y0, x1, y1, z0, h, state, thick=0.05) -> list[Box]:
    col = GOLD if state == "wait" else FENCE_REST
    t = thick
    return [
        Box(x0, y1 - t, z0, x1 - x0, t, h, col, layer=5),          # back
        Box(x0, y0, z0, t, y1 - y0, h, col, layer=5),              # left
        Box(x1 - t, y0, z0, t, y1 - y0, h, col, layer=5),          # right
        Box(x0, y0, z0, x1 - x0, t, h, col, layer=7),              # front, drawn last
    ]


def _images(spec: Spec, pl: Placement) -> list[tuple[str, object, int, tuple]]:
    """(part, image, layer, key) for the chips and their lit units."""
    ppw = raster_ppw(pl)
    cx, cy = CPU_DIE
    gx, gy = GPU_DIE
    out = [("cpu_die", plane_image(dz.cpu_raster(ppw), pl, dz.R(cx, cy, dz.CPU_W, dz.CPU_D), DIE_Z, ("cpu", ppw)),
            2, ()),
           ("gpu_die", plane_image(dz.gpu_raster(ppw), pl, dz.R(gx, gy, dz.GPU_W, dz.GPU_D), DIE_Z, ("gpu", ppw)),
            2, ())]
    if spec.cpu != "off":
        out.append(("pcore", plane_image(dz.pcore_tile(spec.cpu, ppw), pl, pcore_rect(0), DIE_Z,
                                         ("pcore", spec.cpu, ppw)), 3, (spec.cpu,)))
    for i, st in enumerate(spec.rooms):
        shelf = spec.shelf and i == 0
        if st == "off" and not shelf:
            continue
        st = "lit" if st == "off" else st
        out.append((f"room{i}", plane_image(dz.sm_tile(st, ppw, shelf), pl, sm_rect(i), DIE_Z, ("sm", st, shelf, ppw)),
                    3, (st, shelf)))
    return out


def _labels(spec: Spec, pl: Placement) -> list[tuple[str, object, tuple]]:
    """(part, mobject, key): silkscreen on the boards, the launch counter."""
    out = []
    for x0, name, model in ((CPU_X, "CPU", "i9-13900H"), (GPU_X, "GPU", "RTX 4060 Laptop")):
        t = VGroup(label(name, 40, INK), label(model, 26, INK_SOFT))
        t.arrange(direction=np.array([1, 0, 0]), buff=0.18, aligned_edge=DOWN)
        t.scale(0.2 / t.height)                  # world units, left-aligned on the front strip
        out.append(("labels", on_plane(t, pl, (x0 + BM + t.width / 2, FRONT / 2), 0.0), ()))
    if spec.queue:
        r = vram_rects()[-1]
        q = label("queue", 26, INK_SOFT)
        q.scale(0.16 / q.height)
        out.append(("queue", on_plane(q, pl, (r.x1 - q.width / 2, FRONT / 2), 0.0), ()))
    if spec.launches is not None:
        out.append(("launches", launch_chip(spec.launches, pl), (spec.launches,)))
    return out


def launch_chip(text: str, pl: Placement) -> VGroup:
    """The launch counter: a small pill over the PCIe link (launches cross it)."""
    t = VGroup(label(text, 20, INK, mono=True), label("launches", 15, INK_SOFT))
    t.arrange(np.array([1, 0, 0]), buff=0.1, aligned_edge=DOWN)
    box = RoundedRectangle(corner_radius=0.1, width=t.width + 0.3, height=t.height + 0.2, fill_color="#0f141a",
                           fill_opacity=0.95, stroke_color="#2d333b", stroke_width=1.5)
    p = pl.to_frame([[CPU_BW + GAP / 2, (LINK_Y[0] + LINK_Y[1]) / 2, 0.0]])[0]
    g = VGroup(box, t)
    t.move_to(box)
    g.move_to(p + np.array([0.0, 0.32, 0.0]))
    return g


def _depth_range(pl: Placement) -> tuple[float, float]:
    lo, hi = WORLD_LO, WORLD_HI
    ds = [pl.P.depth((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return min(ds), max(ds)


def _box_key(b: Box) -> tuple:
    return (b.x, b.y, b.z, b.w, b.d, b.h, b.color, b.layer)


def build(spec: Spec, pl: Placement, z0: float = 100.0) -> Machine:
    """Draw the boards for `spec`. A face's z_index is z0 + 10 * layer + its
    nearness within the world (0..9); images sit at their layer's top;
    labels at z0 + 95."""
    boxes = _boxes(spec)
    dmin, dmax = _depth_range(pl)
    parts: dict[str, Group] = {}
    keys: dict[str, list] = {}

    def add(name, mob, key):
        parts.setdefault(name, Group()).add(mob)
        keys.setdefault(name, []).append(key)

    for name, img, layer, key in _images(spec, pl):
        img.set_z_index(z0 + 10 * layer + 9.5)
        add(name, img, key)
    for name, b in boxes:
        near = 9.0 * (dmax - pl.P.depth(b.centre)) / (dmax - dmin)
        for j, face in enumerate(box_faces(b, pl)):
            face.set_z_index(z0 + 10 * b.layer + near + 0.001 * j)
            parts.setdefault(name, Group()).add(face)
        keys.setdefault(name, []).append(_box_key(b))
    for name, mob, key in _labels(spec, pl):
        for m in mob.get_family():
            m.set_z_index(z0 + 95)
        add(name, mob, key)
    order = ["boards", "cpu_die", "gpu_die"]
    names = order + [n for n in parts if n not in order]
    return Machine(spec, {n: parts[n] for n in names if n in parts},
                   {n: tuple(keys[n]) for n in names if n in parts})


def with_rooms(spec: Spec, state: str, rooms=None) -> Spec:
    """A copy of spec with the given SMs (default: every SM) set to state."""
    r = list(spec.rooms)
    for i in (range(24) if rooms is None else rooms):
        r[i] = state
    return replace(spec, rooms=tuple(r))


def sm_point(pl: Placement, i: int = 0) -> np.ndarray:
    """Frame position of SM i's centre on the chip's top face."""
    r = sm_rect(i)
    return pl.to_frame([[r.x + r.w / 2, r.y + r.d / 2, DIE_Z]])[0]


def sm_closeup(pl: Placement, centre, height: float, i: int = 0, shelf: bool = True, z: float = 480.0) -> Group:
    """One SM drawn large, flat, like NVIDIA's SM diagram, with a leader line
    to SM i on the chip: four partitions (one warp scheduler and 32 lanes
    each), the L1 / shared memory, texture units and the RT core."""
    from manim import Line, Rectangle

    k = _ppu()
    ppw = round(height * k / dz.SM_D * 1.3, 1)
    img = ImageMobject(dz.sm_tile("lit", ppw, shelf))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    img.set_height(height)
    img.move_to(np.asarray(centre, float))
    frame = Rectangle(width=img.width + 0.08, height=img.height + 0.08, stroke_color="#7fb2ea",
                      stroke_width=2).move_to(img)
    title = label("1 SM", 22, INK).next_to(frame, np.array([0, 1, 0]), buff=0.1)
    sub = label("4 × 32 lanes", 16, INK_SOFT).next_to(frame, DOWN, buff=0.1)
    target = sm_point(pl, i)
    corners = [frame.get_corner(np.array([a, b, 0])) for a in (-1, 1) for b in (-1, 1)]
    start = min(corners, key=lambda c: np.linalg.norm(c - target))
    lead = Line(start, target, stroke_color="#7fb2ea", stroke_width=2)
    g = Group(lead, img, frame, title, sub)
    for m in g.get_family():
        m.set_z_index(z)
    return g


def link_path(pl: Placement) -> list[np.ndarray]:
    """Frame points from the CPU's P-core 0, over the PCIe link, to the GPU's
    GigaThread engine: the way a kernel launch (or a seed) travels."""
    p0 = pcore_rect(0)
    gt = dz.gpu_plan().gigathread.shift(*GPU_DIE)
    ly = (LINK_Y[0] + LINK_Y[1]) / 2
    pts = [(p0.x + p0.w / 2, p0.y + p0.d / 2, DIE_Z), (CPU_BW - 0.05, ly, -0.06), (GPU_X + 0.05, ly, -0.06),
           (GPU_DIE[0] + 0.3, gt.y + gt.d / 2, DIE_Z), (gt.x + gt.w / 2, gt.y + gt.d / 2, DIE_Z)]
    return [pl.to_frame([p])[0] for p in pts]


def slot_point(pl: Placement, j: int, z: float = 0.2) -> np.ndarray:
    """Frame position of queue slot j (in VRAM)."""
    r = slot_rect(j)
    return pl.to_frame([[r.x + r.w / 2, r.y + r.d / 2, z]])[0]


def gpu_corner(pl: Placement, which: str = "front-right") -> np.ndarray:
    """Frame position of a corner of the GPU board, for notes beside it."""
    x = GPU_X + GPU_BW if "right" in which else GPU_X
    y = 0.0 if "front" in which else BOARD_D
    return pl.to_frame([[x, y, 0.0]])[0]
