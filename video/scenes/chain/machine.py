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

import colorsys
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from manim import DOWN, Group, ImageMobject, RoundedRectangle, VGroup, config
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image

from scenes.chain import die as dz
from scenes.chain.iso import Box, Placement, box_faces, on_plane, plane_corners
from scenes.style import BLUE, GOLD, GREEN, GREY, INK, INK_SOFT, RED_PX, TEAL, label

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
SPILL_SLOTS = 3                               # the spill: VRAM chip 0's slots
TAG_D = 0.15                                  # a tag stripe: this share of its token's depth (D7)


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
    team_state: str = "work"            # work | wait: wait (the gate flash) turns every cube amber
    team_hues: tuple = ()               # per SM, a hue per resident cube, left first (missing: BLUE)
    team_busy: tuple = ()               # per SM, per cube: work | dim | wait (missing: work)
    shelves: tuple = ()                 # per SM: off | full | empty, its shared memory (missing: off)
    room_fence: str | None = None       # None | rest | wait: __syncthreads around SM 0
    queue: bool = False                 # the global queue, in VRAM
    tokens: tuple = ()                  # token colours, slot 0 first
    tag_tokens: tuple = ()              # a tag colour per token: a thin stripe, one bit of the entry
    spill: bool = False                 # the shared ring's overflow: tokens in VRAM chip 0's slots
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


# ---- blocks: hues, placement, busy (D4, D5, D6) --------------------------

GOLDEN_ANGLE = 137.508              # degrees, as ch03's benchmarks/wavefront.py
# Hues no block may take (centre, half width in degrees): BLUE and GREEN
# (blocks 0 and 1), the red of unpainted pixels, the amber of waiting and
# the teal of the final answer.
HUE_BANDS = ((211.0, 12.0), (128.0, 12.0), (5.0, 20.0), (38.0, 32.0), (175.0, 14.0))
DIM_MIX = 0.62                      # a dim cube: its hue this far towards the die's colour


def _hls(hex_: str) -> tuple:
    h = hex_.lstrip("#")
    return colorsys.rgb_to_hls(*(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)))


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(min(max(v, 0.0), 1.0) * 255)):02x}" for v in rgb)


def mix_hex(a: str, b: str, t: float) -> str:
    """a, t of the way to b, in RGB."""
    ha, hb = a.lstrip("#"), b.lstrip("#")
    va = [int(ha[i:i + 2], 16) for i in (0, 2, 4)]
    vb = [int(hb[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(va, vb))


def _hue_ok(deg: float) -> bool:
    return all(min(abs(deg - c) % 360, 360 - abs(deg - c) % 360) > w for c, w in HUE_BANDS)


@lru_cache(maxsize=None)
def block_hue_degrees(n: int = 48) -> tuple:
    """Each block's hue in degrees: block 0 BLUE's, block 1 GREEN's, blocks
    2.. the golden-angle walk from BLUE's hue, skipping the HUE_BANDS."""
    out = [_hls(BLUE)[0] * 360, _hls(GREEN)[0] * 360]
    k = 0
    while len(out) < n:
        k += 1
        deg = (out[0] + k * GOLDEN_ANGLE) % 360
        if _hue_ok(deg):
            out.append(deg)
    return tuple(out[:n])


@lru_cache(maxsize=None)
def block_hues(n: int = 48) -> tuple:
    """A cube's colour per block: BLUE, GREEN, then the walk at BLUE's
    lightness and saturation."""
    _, l, s = _hls(BLUE)
    out = [BLUE, GREEN]
    for deg in block_hue_degrees(n)[2:]:
        out.append(_hex(colorsys.hls_to_rgb(deg / 360, l, s)))
    return tuple(out[:n])


def block_ramp(b: int, lo: str, hi: str) -> tuple[str, str]:
    """Block b's ring ramp (first ring, last ring): the blue ramp lo -> hi
    turned to the block's hue. Block 0 keeps lo and hi exactly, so the
    one-block floors are block 0's."""
    if b == 0:
        return lo, hi
    deg = block_hue_degrees(max(48, b + 1))[b] / 360
    (_, l0, s0), (_, l1, s1) = _hls(lo), _hls(hi)
    return _hex(colorsys.hls_to_rgb(deg, l0, s0)), _hex(colorsys.hls_to_rgb(deg, l1, s1))


def block_states(busy: int, blocks: int, idle: str = "dim") -> tuple:
    """Per block: work for blocks 0 .. busy - 1, `idle` (dim | wait) for the rest (D6)."""
    return tuple("work" if b < busy else idle for b in range(blocks))


def place_blocks(spec: "Spec", sm_ids, hues=None, busy=None) -> "Spec":
    """spec with block b's cube on SM sm_ids[b] (D5), in hues[b] (default:
    all BLUE) and state busy[b] (default: all work). On a shared SM the
    lower block stands on the left."""
    teams, th, tb = [0] * 24, [[] for _ in range(24)], [[] for _ in range(24)]
    for b, sm in enumerate(sm_ids):
        teams[sm] += 1
        th[sm].append(hues[b] if hues is not None else BLUE)
        tb[sm].append(busy[b] if busy is not None else "work")
    if max(teams) > 2:
        raise ValueError("at most two cubes per SM")
    return replace(spec, teams=tuple(teams), team_hues=tuple(map(tuple, th)) if hues is not None else (),
                   team_busy=tuple(map(tuple, tb)) if busy is not None else ())


def _at(t: tuple, i: int, j: int, default):
    return t[i][j] if i < len(t) and j < len(t[i]) else default


def cube_color(spec: "Spec", i: int, j: int) -> str:
    """Cube j on SM i: amber at the gate flash or while waiting, its hue
    while it works, its hue dimmed while it has no work in the ring."""
    busy = _at(spec.team_busy, i, j, "work")
    if spec.team_state == "wait" or busy == "wait":
        return GOLD
    hue = _at(spec.team_hues, i, j, BLUE)
    return mix_hex(hue, dz.DIE, DIM_MIX) if busy == "dim" else hue


def shelf_of(spec: "Spec", i: int) -> str:
    return dz.shelf_state(spec.shelves[i]) if i < len(spec.shelves) else "off"


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
    for i in range(24):
        n = spec.teams[i]
        if not n:
            continue
        r = sm_rect(i)
        bw = 0.20 if n == 1 else 0.14
        xs = [r.x + (r.w - bw) / 2] if n == 1 else [r.x + 0.04, r.x + r.w - 0.04 - bw]
        for j, x in enumerate(xs):
            out.append(("teams", Box(x, r.y + 0.16, DIE_Z, bw, 0.24, 0.10, cube_color(spec, i, j), layer=4)))
    if spec.room_fence:
        r = sm_rect(0)
        # like the gate: a low rail at rest (SM 0 stays visible), a wall when the threads wait
        h = 0.18 if spec.room_fence == "wait" else 0.05
        out += [("room_fence", b) for b in _fence(r.x - 0.03, r.y - 0.03, r.x1 + 0.03, r.y1 + 0.03,
                                                     DIE_Z, h, spec.room_fence)]
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
                # one bit of the entry: a thin band round the token's front end
                s = slot_rect(j).inset(0.025, 0.04)
                out.append(("tags", Box(s.x - 0.006, s.y - 0.006, 0.096, s.w + 0.012, s.d * TAG_D + 0.006, 0.11,
                                        colr, layer=5)))
    elif spec.spill:
        for j in range(SPILL_SLOTS):
            s = slot_rect(j)
            out.append(("spill", Box(s.x, s.y, 0.08, s.w, s.d, 0.02, SLOT, layer=2)))
            s = s.inset(0.025, 0.04)
            out.append(("spill", Box(s.x, s.y, 0.10, s.w, s.d, 0.10, RED_PX, layer=4)))
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
        shelf = shelf_of(spec, i)
        if st == "off" and shelf == "off":
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
        t.scale(0.27 / t.height)                 # world units, left-aligned on the front strip
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
    word = "launch" if text == "1" else "launches"
    t = VGroup(label(text, 20, INK, mono=True), label(word, 60, INK_SOFT).scale(0.25))
    t.arrange(np.array([1, 0, 0]), buff=0.1, aligned_edge=DOWN)
    box = RoundedRectangle(corner_radius=0.1, width=t.width + 0.3, height=t.height + 0.2, fill_color="#0f141a",
                           fill_opacity=0.95, stroke_color="#2d333b", stroke_width=1.5)
    p = pl.to_frame([[CPU_BW + GAP / 2, (LINK_Y[0] + LINK_Y[1]) / 2, 0.0]])[0]
    g = VGroup(box, t)
    t.move_to(box)
    g.move_to(p + np.array([0.0, 0.62, 0.0]))         # clear of the link: launches show crossing it
    return g


def _depth_range(pl: Placement) -> tuple[float, float]:
    lo, hi = WORLD_LO, WORLD_HI
    ds = [pl.P.depth((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return min(ds), max(ds)


def _box_key(b: Box) -> tuple:
    return (b.x, b.y, b.z, b.w, b.d, b.h, b.color, b.layer)


def build(spec: Spec, pl: Placement, z0: float = 100.0, only=None) -> Machine:
    """Draw the boards for `spec`. A face's z_index is z0 + 10 * layer + its
    nearness within the world (0..9); images sit at their layer's top;
    labels at z0 + 95. `only`: the part names to draw (default: all), for
    a scene that swaps one part every frame (the cubes' busy states)."""
    want = (lambda name: True) if only is None else (lambda name: name in only)
    boxes = [(name, b) for name, b in _boxes(spec) if want(name)]
    dmin, dmax = _depth_range(pl)
    parts: dict[str, Group] = {}
    keys: dict[str, list] = {}

    def add(name, mob, key):
        parts.setdefault(name, Group()).add(mob)
        keys.setdefault(name, []).append(key)

    if only is None or any(n in only for n in ("cpu_die", "gpu_die", "pcore")) or \
            any(n.startswith("room") for n in only):
        for name, img, layer, key in _images(spec, pl):
            if want(name):
                img.set_z_index(z0 + 10 * layer + 9.5)
                add(name, img, key)
    for name, b in boxes:
        near = 9.0 * (dmax - pl.P.depth(b.centre)) / (dmax - dmin)
        for j, face in enumerate(box_faces(b, pl)):
            face.set_z_index(z0 + 10 * b.layer + near + 0.001 * j)
            parts.setdefault(name, Group()).add(face)
        keys.setdefault(name, []).append(_box_key(b))
    if only is None or any(n in only for n in ("labels", "queue", "launches")):
        for name, mob, key in _labels(spec, pl):
            if want(name):
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


def sm_closeup(pl: Placement, centre, height: float, i: int = 0, shelf="off", z: float = 480.0) -> Group:
    """One SM drawn large, flat, like NVIDIA's SM figure (four partitions of
    32 lanes, the shared memory, texture units, the RT core), on a dark
    panel, titled "1 SM". Grow it out of `sm_point(pl, i)` and back into it
    (FadeIn / FadeOut with target_position) so it reads as that SM."""
    from manim import Rectangle

    k = _ppu()
    ppw = round(height * k / dz.SM_D * 1.3, 1)
    img = ImageMobject(dz.sm_detail(ppw, shelf))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    img.set_height(height)
    img.move_to(np.asarray(centre, float))
    frame = Rectangle(width=img.width + 0.08, height=img.height + 0.08, stroke_color="#7fb2ea",
                      stroke_width=2.5).move_to(img)
    title = label("1 SM", 26, INK).next_to(frame, np.array([0, 1, 0]), buff=0.12)
    body = Group(img, frame, title)
    box = RoundedRectangle(corner_radius=0.14, width=body.width + 0.4, height=body.height + 0.36,
                           fill_color="#0f141a", fill_opacity=0.97, stroke_color="#2d333b", stroke_width=1.5)
    box.move_to(body)
    g = Group(box, img, frame, title)
    for j, m in enumerate(g.get_family()):
        m.set_z_index(z + (0 if m is box else 1))
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


def vram_point(pl: Placement, z: float = 0.08) -> np.ndarray:
    """Frame position of the middle of the four VRAM chips (what the VRAM
    close-up grows out of and folds back into)."""
    v = vram_rects()
    return pl.to_frame([[(v[0].x + v[-1].x1) / 2, v[0].y + v[0].d / 2, z]])[0]


def point_along(pts: list, s: float) -> np.ndarray:
    """The point a fraction s (0..1) of the way along the polyline pts, by length."""
    seg = [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])]
    d = min(max(s, 0.0), 1.0) * sum(seg)
    for a, b, n in zip(pts, pts[1:], seg):
        if d <= n:
            return a + (b - a) * (d / n if n else 0.0)
        d -= n
    return pts[-1]


# ---- packets (D3): a load or a store between memory and an SM -------------

PACKET_H = 0.07              # frame units
BOARD_BYTE = 0.0022          # a packet's length on the board, frame units per byte (128 B: 0.28)
Z_PACKET = 486.0             # over the call-out cards, under the captions


def packet(nbytes: float, color: str = INK, unit: float = BOARD_BYTE, height: float = PACKET_H,
           z: float = Z_PACKET):
    """A bar as long as the bytes it moves (`unit` frame units per byte),
    fully lit: a dense read or write. Move it along a path (vram_path, a
    card's leader, a close-up's lane) up for a load, down for a store."""
    from manim import Rectangle

    bar = Rectangle(width=max(nbytes * unit, 0.01), height=height, fill_color=color, fill_opacity=1.0,
                    stroke_color=INK, stroke_width=0.8, stroke_opacity=0.6)
    bar.set_z_index(z)
    return bar


def _mc_for_slot(j: int) -> dz.R:
    """The memory controller in front of VRAM chip j // 3: chips 0, 1 on the
    left one, chips 2, 3 on the right one (front row of the die)."""
    mcs = dz.gpu_plan().mcs
    return (mcs[0] if j // 3 < 2 else mcs[2]).shift(*GPU_DIE)


def vram_path(pl: Placement, sm: int, slot: int) -> list[np.ndarray]:
    """Frame points from SM `sm` through its side's memory controller to
    queue slot `slot` in VRAM: the way a store travels (reverse it for a load)."""
    r, m, s = sm_rect(sm), _mc_for_slot(slot), slot_rect(slot)
    mx = m.x + m.w / 2
    pts = [(r.x + r.w / 2, r.y + r.d / 2, DIE_Z), (mx, r.y + r.d / 2, DIE_Z), (mx, m.y + 0.12, DIE_Z),
           (s.x + s.w / 2, s.y + s.d / 2, 0.2)]
    return [pl.to_frame([p])[0] for p in pts]


def path_mobject(pts: list):
    """A VMobject through the points, for MoveAlongPath."""
    from manim import VMobject

    return VMobject().set_points_as_corners([np.asarray(p, float) for p in pts])


# ---- call-outs: raised cards tied to a part of the board ------------------

CARD_FILL = "#0f141a"
CARD_EDGE = "#4a5562"
LEADER = "#8b949e"
Z_CALLOUT = 480.0            # sm_closeup's level: over the machine, under the overlays


def edge_point(centre, w: float, h: float, target) -> np.ndarray:
    """Where the line from a w x h box's centre towards `target` leaves the box."""
    c, t = np.asarray(centre, float), np.asarray(target, float)
    d = t - c
    sx = (w / 2) / abs(d[0]) if abs(d[0]) > 1e-9 else np.inf
    sy = (h / 2) / abs(d[1]) if abs(d[1]) > 1e-9 else np.inf
    return c + d * min(sx, sy, 1.0)


def callout(centre, size, target, z: float = Z_CALLOUT, side: str | None = None) -> Group:
    """A raised dark card (its shadow, its frame) and a leader from its edge
    to `target`, a frame point on the board, ending in a dot: it reads as a
    call-out from that part. The leader leaves the card where the line from
    its centre to the target crosses the edge, or from the middle of `side`
    ("right", "left", "top", "bottom"). Returns Group(shadow, box, leader, dot)."""
    from manim import Dot, Line

    c = np.array([centre[0], centre[1], 0.0])
    w, h = size
    mids = {"right": (w / 2, 0.0), "left": (-w / 2, 0.0), "top": (0.0, h / 2), "bottom": (0.0, -h / 2)}
    shadow = RoundedRectangle(corner_radius=0.1, width=w, height=h, fill_color="#000000", fill_opacity=0.5,
                              stroke_width=0).move_to(c + np.array([0.05, -0.06, 0.0]))
    box = RoundedRectangle(corner_radius=0.1, width=w, height=h, fill_color=CARD_FILL, fill_opacity=1.0,
                           stroke_color=CARD_EDGE, stroke_width=2).move_to(c)
    tgt = np.array([target[0], target[1], 0.0])
    start = c + np.array([*mids[side], 0.0]) if side else edge_point(c, w, h, tgt)
    leader = Line(start, tgt, stroke_color=LEADER, stroke_width=1.6)
    dot = Dot(tgt, radius=0.035, color=LEADER)
    for m, dz_ in ((shadow, -0.4), (leader, -0.3), (dot, -0.3), (box, 0.0)):
        m.set_z_index(z + dz_)
    return Group(shadow, box, leader, dot)


def _hatch(x0: float, y0: float, x1: float, y1: float, step: float, color: str, width: float):
    """Diagonal lines (up to the right) clipped to the rectangle."""
    from manim import Line

    g = VGroup()
    c = (x0 - y1) + step / 2                      # lines y = x - c
    while c < x1 - y0:
        pts = []
        for x in (x0, x1):                       # crossings with the vertical sides
            y = x - c
            if y0 <= y <= y1:
                pts.append((x, y))
        for y in (y0, y1):                       # and with the horizontal sides
            x = y + c
            if x0 < x < x1:
                pts.append((x, y))
        pts = sorted(set(pts))
        if len(pts) >= 2:
            g.add(Line([*pts[0], 0], [*pts[-1], 0], stroke_color=color, stroke_width=width))
        c += step
    return g


class ShelfCard(Group):
    """D1: one block's shared memory as a card tied to SM `sm` by its leader.

    The outlined box is the block's default shared memory (48 KB); inside
    it, outlined in the queue's red, the queue region (8,192 slots x 4 B,
    two thirds of the box: the largest power of two that fits); the last
    third hatched. No text: the captions and the narration name it.

    With `bar` (the r03_giant gauge) a bar under the box, on the same byte
    scale, is the queue's real load (`set_load(slots)`); the scale reaches
    `peak_slots`, so the box is the left part of the card. Without it (the
    r04 cards) the box fills the card, and `fill` is "full" (the region
    holds red tokens) or "empty" (outlined only).
    """

    def __init__(self, pl: Placement, centre, size, sm: int = 0, *, slots: int, shared_bytes: int,
                 bar: bool = False, peak_slots: int = 0, fill: str = "empty", side: str | None = None,
                 z: float = Z_CALLOUT):
        from manim import DashedLine, Line, Rectangle

        super().__init__()
        w, h = size
        cx, cy = float(centre[0]), float(centre[1])
        self.card = callout((cx, cy), size, sm_point(pl, sm), z, side)
        pad = 0.16
        inner = w - 2 * pad
        x_l = cx - inner / 2
        scale = max(peak_slots * 4 * 1.04, shared_bytes) if bar else shared_bytes
        self.unit = inner / scale                                  # frame units per byte
        bw, rw = shared_bytes * self.unit, slots * 4 * self.unit
        if bar:
            box_h, y_box, y_bar = h * 0.36, cy + h * 0.16, cy - h * 0.24
        else:
            box_h, y_box, y_bar = h - 2 * pad, cy, None
        inset = 0.035
        self.box = Rectangle(width=bw, height=box_h, stroke_color=INK_SOFT, stroke_width=2.2).move_to(
            [x_l + bw / 2, y_box, 0])
        self.region = Rectangle(width=rw - inset, height=box_h - 2 * inset, stroke_color=RED_PX,
                                stroke_width=2.2).move_to([x_l + inset + (rw - inset) / 2, y_box, 0])
        self.hatch = _hatch(x_l + rw + 0.02, y_box - box_h / 2 + 0.02, x_l + bw - 0.02, y_box + box_h / 2 - 0.02,
                            0.07, GREY, 1.2)
        body = [self.box, self.region, self.hatch]
        self.tokens = VGroup()
        if fill == "full":
            rx0, rx1 = x_l + inset * 2, x_l + rw - inset
            ry0, ry1 = y_box - box_h / 2 + inset * 2, y_box + box_h / 2 - inset * 2
            rows = 2 if box_h > 0.35 else 1
            cols = max(1, int((rx1 - rx0) / 0.1))
            tw, th = (rx1 - rx0) / cols, (ry1 - ry0) / rows
            for a in range(rows):
                for b in range(cols):
                    self.tokens.add(Rectangle(width=tw * 0.78, height=th * 0.72, fill_color=RED_PX, fill_opacity=1,
                                              stroke_width=0).move_to([rx0 + (b + 0.5) * tw, ry0 + (a + 0.5) * th, 0]))
            body.append(self.tokens)
        self.x_l, self.y_bar, self.inner, self.slots = x_l, y_bar, inner, slots
        self.bar = None
        if bar:
            bh = h * 0.15
            track = Line([x_l, y_bar, 0], [x_l + inner, y_bar, 0], stroke_color="#2d333b", stroke_width=bh * 70)
            y_top = y_box - box_h / 2
            guides = VGroup(
                DashedLine([x_l + rw, y_top, 0], [x_l + rw, y_bar - bh / 2 - 0.02, 0], dash_length=0.04,
                           stroke_color=RED_PX, stroke_width=1.4, stroke_opacity=0.8),
                DashedLine([x_l + bw, y_top, 0], [x_l + bw, y_bar - bh / 2 - 0.02, 0], dash_length=0.04,
                           stroke_color=INK_SOFT, stroke_width=1.4, stroke_opacity=0.7))
            self.bar = Rectangle(width=inner, height=bh, fill_color=RED_PX, fill_opacity=0.95, stroke_width=0)
            self.bar.move_to([x_l + inner / 2, y_bar, 0])
            body += [track, guides, self.bar]
            self.set_load(0)
        self.body = Group(*body)
        for m in self.body.get_family():
            m.set_z_index(z + 1)
        if self.bar is not None:
            self.bar.set_z_index(z + 1.5)
        self.add(self.card, self.body)

    def set_load(self, slots: float) -> "ShelfCard":
        """The bar's length: `slots` 4-byte entries on the card's byte scale."""
        w = max(slots * 4 * self.unit, 1e-3)
        self.bar.stretch_to_fit_width(w)
        self.bar.move_to([self.x_l + w / 2, self.y_bar, 0])
        return self

    def region_end(self) -> np.ndarray:
        """Frame point at the queue region's end: where the overflow pours out."""
        return np.array([self.region.get_right()[0], self.region.get_center()[1], 0.0])

    def to_vram(self, pl: Placement, slot: int, sm: int = 0) -> list[np.ndarray]:
        """The overflow's way: the region's end, the leader into SM `sm`,
        then the memory controller to queue slot `slot`."""
        lead = self.card[2]
        return [self.region_end(), lead.get_start(), *vram_path(pl, sm, slot)]


def shelf_gauge(pl: Placement, centre, size, slots: int, shared_bytes: int, peak_slots: int,
                sm: int = 0) -> ShelfCard:
    """D1, r03_giant: the gauge with its load bar (call set_load each ring)."""
    return ShelfCard(pl, centre, size, sm, slots=slots, shared_bytes=shared_bytes, bar=True, peak_slots=peak_slots)


def shelf_card(pl: Placement, centre, size, sm: int, full: bool, slots: int, shared_bytes: int) -> ShelfCard:
    """D1, r04: one SM's card without the bar, its region full or empty. The
    two cards stack, so each leader leaves its card's right edge (with
    SM 0's card below SM 2's, the two leaders never cross)."""
    return ShelfCard(pl, centre, size, sm, slots=slots, shared_bytes=shared_bytes, fill="full" if full else "empty",
                     side="right")


# ---- the VRAM close-up (D2): the global-memory layer ----------------------

ELEM_BYTES = {"image": 3, "visited": 4, "queue": 4, "runs": 20}   # uint8 RGB, int32, int32, five int32
FORMS = {"tags": ("image", "visited", "queue"), "runs": ("image", "runs")}
PIXEL_BG = "#c9d1d9"         # an image pixel off the blobs (the picture's white, toned down)
VISITED_ON = "#adbac7"
WARP_FILL, WARP_EDGE = "#1d3554", "#7fb2ea"     # the SM's lit colours: the warp's 32 threads
BIT_FILL = "#5a6b80"         # an address bit (grey-blue)


@dataclass
class Lane:
    """One array's strip of cells: cell i spans [x0 + i * elem * unit, + elem * unit)."""
    name: str
    x0: float
    y: float
    h: float
    elem: int
    unit: float
    cells: VGroup

    def span(self, i0: int, n: int = 1) -> tuple[float, float]:
        a = self.x0 + i0 * self.elem * self.unit
        return a, a + n * self.elem * self.unit

    def centre(self, i0: int, n: int = 1) -> np.ndarray:
        a, b = self.span(i0, n)
        return np.array([(a + b) / 2, self.y, 0.0])


class VramPanel(Group):
    """D2: the VRAM chips' contents drawn large, a flat dark card (the twin
    of `sm_closeup`). Lanes are horizontal strips of cells, one per array,
    in memory order; a cell is as wide as its bytes, on one scale per
    panel. The only text is the lane names (and "1 warp" in the runs form).

    form "tags" (r08): image (3 B a pixel), visited (4 B), queue (4 B an
    entry, each with its tag stripe; the level's window [front, rear)
    bracketed, new entries land at rear). A warp strip of 32 cells (one
    per entry read) sits on the panel's SM edge, the top, unlabelled.
    form "runs" (r10_runs): one memory row of the image, then the runs
    (five int32 sub-cells an entry); the warp strip, 32 pixels of the row,
    labelled "1 warp".

    Packets: `packet(lane, i0, n)` is a bar over those cells (as long as
    their bytes); a load moves it up to `warp_y`, a store down to a lane.
    Grow the panel out of `vram_point(pl)` and fold it back into it (no
    leader line by default: `leader=True` draws one).
    """

    def __init__(self, pl: Placement, centre, size, form: str = "tags", *, pixels=None, visited=None,
                 window=(4, 36), tags=None, row=None, leader: bool = False, z: float = Z_CALLOUT):
        from manim import Line, Rectangle

        super().__init__()
        if form not in FORMS:
            raise ValueError(f"form must be one of {sorted(FORMS)}, not {form!r}")
        from scenes.chain.overlays import lbl

        w, h = size
        cx, cy = float(centre[0]), float(centre[1])
        self.form, self.z = form, z
        self.card = callout((cx, cy), size, vram_point(pl), z)
        if not leader:          # the panel sits over the floor: growing out of the chips links it, a line would cross the caption
            self.card.remove(*self.card[2:])
        pad = 0.2
        inner = w - 2 * pad
        x_l = cx - inner / 2
        y_t = cy + h / 2
        read = "queue" if form == "tags" else "image"           # what the warp reads
        shown = 160 if form == "tags" else 120                    # bytes across the panel
        self.unit = inner / shown
        self.x_l, self.inner = x_l, inner
        lane_h = 0.26
        names = FORMS[form]
        # top down: (the "1 warp" label), the warp strip, the gap packets cross,
        # each lane under its name, then (tags) the window's bracket and a row
        # for an entry opened into its bits; the gap takes what is left
        warp_h, name_h, step = 0.2, 0.17, 0.14
        lanes_h = len(names) * (name_h + lane_h) + (len(names) - 1) * step
        head = 0.16 + (0.19 if form == "runs" else 0.0)
        foot = (0.2 + 0.27 if form == "tags" else 0.0) + 0.16
        travel = min(max(h - head - warp_h - lanes_h - foot, 0.45), 1.0)
        self.warp_y = y_t - head - warp_h / 2
        y = self.warp_y - warp_h / 2 - travel
        ys = []
        for _ in names:
            ys.append(y - name_h - lane_h / 2)
            y -= name_h + lane_h + step
        self.lanes: dict[str, Lane] = {}
        labels = VGroup()
        gap = 0.012
        row = list(row) if row is not None else None
        for name, y in zip(names, ys):
            e = ELEM_BYTES[name]
            n = int(inner // (e * self.unit))
            cells = VGroup()
            colors = self._colors(name, n, pixels, visited, window, row)
            cw = e * self.unit
            for i, col in enumerate(colors):
                x = x_l + i * cw
                if col == "run":            # one run: five int32 sub-cells, a wider gap between runs
                    sw = (cw - 0.03) / 5
                    cells.add(VGroup(*[Rectangle(width=sw - gap, height=lane_h, fill_color=TEAL, fill_opacity=1,
                                                 stroke_width=0).move_to([x + 0.015 + (k + 0.5) * sw, y, 0])
                                       for k in range(5)]))
                else:
                    cells.add(Rectangle(width=max(cw - gap, 0.004), height=lane_h, fill_color=col, fill_opacity=1,
                                        stroke_width=0).move_to([x + cw / 2, y, 0]))
            self.lanes[name] = Lane(name, x_l, y, lane_h, e, self.unit, cells)
            t = lbl(name, 14, INK_SOFT)
            t.move_to([x_l + t.width / 2, y + lane_h / 2 + 0.06 + t.height / 2, 0])
            labels.add(t)
        parts = [labels, *[ln.cells for ln in self.lanes.values()]]
        self.stripes = VGroup()
        self.bracket = VGroup()
        if form == "tags":
            q = self.lanes["queue"]
            f0, r0 = window
            for k, i in enumerate(range(f0, r0)):
                tag = tags[k % len(tags)] if tags else None
                if tag and i < len(q.cells):
                    a, b = q.span(i)
                    sw = (b - a - gap) * 0.3      # wider than the board's stripe (TAG_D): cells are small here
                    self.stripes.add(Rectangle(width=sw, height=lane_h, fill_color=tag, fill_opacity=1,
                                               stroke_width=0).move_to([b - gap / 2 - sw / 2, q.y, 0]))
            a, _ = q.span(f0)
            _, b = q.span(r0 - 1)
            yb = q.y - lane_h / 2 - 0.07
            self.bracket = VGroup(Line([a, yb, 0], [b, yb, 0]), Line([a, yb, 0], [a, yb + 0.05, 0]),
                                  Line([b, yb, 0], [b, yb + 0.05, 0])).set_stroke(INK_SOFT, 1.6)
            parts += [self.stripes, self.bracket]
        # the warp strip: 32 cells, one per element read, over the elements it reads
        e = ELEM_BYTES[read]
        i0 = window[0] if form == "tags" else 4
        a, _ = self.lanes[read].span(i0)
        self.warp_i0 = i0
        cw = e * self.unit
        self.warp = VGroup(*[Rectangle(width=cw - gap, height=warp_h, fill_color=WARP_FILL, fill_opacity=1,
                                       stroke_color=WARP_EDGE, stroke_width=0.8).move_to([a + (k + 0.5) * cw,
                                                                                          self.warp_y, 0])
                             for k in range(32)])
        parts.append(self.warp)
        if form == "runs":
            t = lbl("1 warp", 14, INK)
            t.next_to(self.warp, np.array([0, 1, 0]), buff=0.08).align_to(self.warp, np.array([-1, 0, 0]))
            self.warp_label = t
            parts.append(t)
        self.body = Group(*parts)
        for m in self.body.get_family():
            m.set_z_index(z + 1)
        for m in self.stripes.get_family():
            m.set_z_index(z + 1.2)
        self.add(self.card, self.body)

    @staticmethod
    def _colors(name: str, n: int, pixels, visited, window, row) -> list:
        if name == "image":
            if row is not None:
                return [RED_PX if row[i % len(row)] else PIXEL_BG for i in range(n)]
            if pixels is not None:
                return [pixels[i % len(pixels)] for i in range(n)]
            pat = (0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0)         # a schematic picture row: blobs and gaps
            return [RED_PX if pat[i % len(pat)] else PIXEL_BG for i in range(n)]
        if name == "visited":
            v = visited if visited is not None else [1] * (window[1] + 2)
            return [VISITED_ON if i < len(v) and v[i] else SLOT for i in range(n)]
        if name == "queue":
            return [RED_PX if window[0] <= i < window[1] else SLOT for i in range(n)]
        if name == "runs":
            spans = len(runs_of(row)) if row is not None else 3
            return ["run" if i < spans else SLOT for i in range(n)]
        raise KeyError(name)

    def packet(self, lane: str, i0: int, n: int, color: str = INK) -> object:
        """A bar over cells [i0, i0 + n) of a lane: its length is their bytes."""
        a, b = self.lanes[lane].span(i0, n)
        bar = packet(b - a, color, unit=1.0, height=0.14, z=self.z + 3)
        bar.move_to([(a + b) / 2, self.lanes[lane].y, 0])
        return bar

    def warp_point(self, k: int = 16) -> np.ndarray:
        """Frame point of warp lane k (0..31); k = 16 is the strip's middle edge."""
        if k >= 32:
            return self.warp.get_center()
        return self.warp[k].get_center()

    def entry_bits(self, i: int, tag: str, y: float | None = None):
        """Queue entry i opened into its 32 bit cells across the panel, under
        the queue lane: 31 grey-blue address bits, the last one the tag.
        Transform the entry's cell into it."""
        from manim import Rectangle

        q = self.lanes["queue"]
        y = q.y - q.h / 2 - 0.34 if y is None else y
        bw = self.inner / 32
        g = VGroup(*[Rectangle(width=bw - 0.012, height=0.2, fill_color=tag if k == 31 else BIT_FILL,
                               fill_opacity=1, stroke_width=0).move_to([self.x_l + (k + 0.5) * bw, y, 0])
                     for k in range(32)])
        for m in g.get_family():
            m.set_z_index(self.z + 2)
        return g


def runs_of(row) -> list[tuple[int, int]]:
    """(start, end) of each red span of a row, end inclusive."""
    out, i, row = [], 0, list(row)
    while i < len(row):
        if row[i]:
            j = i
            while j + 1 < len(row) and row[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def vram_closeup(pl: Placement, centre, height: float, form: str = "tags", width: float | None = None,
                 **kw) -> VramPanel:
    """D2: the VRAM close-up at `centre`, `height` tall (width default
    0.62 x height); see VramPanel for the forms and the keywords."""
    return VramPanel(pl, centre, (width or 0.62 * height, height), form, **kw)
