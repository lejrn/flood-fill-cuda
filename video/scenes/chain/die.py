"""The two chips seen from above, drawn like the vendors' block diagrams.

GPU: the RTX 4060 Laptop GPU (AD107): 3 GPCs x 4 TPCs x 2 SMs = 24 SMs,
a 32 MB L2 (what the device reports), the GigaThread engine that hands
blocks to SMs, four 32-bit memory controllers (a 128-bit bus). The plan
follows NVIDIA's Ada diagrams (GPCs above and below the L2); AD107's real
floorplan is not public, so this is a schematic, as NVIDIA's are.
Each SM is drawn like NVIDIA's SM diagram: four partitions (scheduler,
register file, FP32 cores, tensor core, LD/ST), the L1 / shared memory,
the texture units and the RT core.

CPU: the i9-13900H (Raptor Lake-H): 6 P-cores, 8 E-cores in two
clusters of 4, a 24 MB L3 on the ring, the Xe graphics and the system
agent (memory controller).

Everything is in world units on the chip's top face, x right, y away
from the viewer, origin at the chip's front-left corner. `gpu_raster`
and `cpu_raster` draw the chip at rest; `sm_tile` and `pcore_tile` draw
one unit in a state, laid over the chip by the machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
SS = 2                      # supersampling: draw at 2x, then downsample


@dataclass(frozen=True)
class R:
    """A rectangle on the chip, in world units (front-left corner, size)."""
    x: float
    y: float
    w: float
    d: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.d

    def inset(self, a: float, b: float | None = None) -> "R":
        b = a if b is None else b
        return R(self.x + a, self.y + b, self.w - 2 * a, self.d - 2 * b)

    def shift(self, dx: float, dy: float) -> "R":
        return R(self.x + dx, self.y + dy, self.w, self.d)


# ---- palettes ------------------------------------------------------------
# NVIDIA's colours, darkened so the chip at rest reads as unpowered; amber
# stays reserved for "waiting", so the schedulers are a muted brown.
DIE = "#161b22"
GPC_BG = "#1d232b"
GPC_EDGE = "#38414c"
TPC_EDGE = "#2c343e"
LABEL = "#8a96a3"

SM = {
    #        frame      edge       sched      regs       cores      tensor     ldst       l1         tex        rt
    "off":  ("#222931", "#333c47", "#3d3328", "#1d343b", "#24331f", "#20301c", "#3a2626", "#1c2f3a", "#1f2a40", "#35341f"),
    "idle": ("#28303a", "#4a5562", "#54452f", "#244650", "#33502c", "#2d4727", "#4e3131", "#25404f", "#283a5a", "#4a482a"),
    "lit":  ("#1d3554", "#7fb2ea", "#8a6a3f", "#2f6f7d", "#4c8fd6", "#3a74b4", "#7a4848", "#2f5f78", "#3d5d92", "#6b6835"),
    "wait": ("#3b2f1a", "#d99a2b", "#8a6a3f", "#4a4430", "#9b7124", "#7e5d1f", "#6a4630", "#4b4632", "#4e4632", "#6b6835"),
}
SHELF_LIT = "#2aa198"        # the shared memory in use (one block's queue)

CPU_COL = {
    "die": "#171c23", "edge": "#38414c",
    "pcore": "#262e38", "pcore_l2": "#1e2a36", "ecore": "#252c35", "ecore_l2": "#1d2833",
    "l3": "#1f2c3d", "ring": "#4a5562", "igpu": "#1d2530", "eu": "#26303d", "sa": "#20262e",
}
PCORE = {"off": ("#262e38", "#38414c", "#1e2a36"), "lit": ("#4c8fd6", "#a9cdf5", "#2c5d93"),
         "idle": ("#2f3946", "#4a5562", "#253446")}


# ---- GPU plan ------------------------------------------------------------

M = 0.05                     # margin
MC_W = 0.10                  # memory controller bars, left and right
SM_W, SM_D = 0.40, 0.56
TPC_PAD, SM_GAP = 0.03, 0.04
TPC_W = SM_W + 2 * TPC_PAD
TPC_D = 2 * SM_D + SM_GAP + 2 * TPC_PAD
TPC_GAP = 0.04
RASTER_D = 0.10
GPC_W = 4 * TPC_W + 3 * TPC_GAP + 2 * M
GPC_D = TPC_D + RASTER_D + 3 * M
L2_D = 0.42
GT_D, PCIE_D = 0.12, 0.10
INNER_W = 2 * GPC_W + 0.06
GPU_W = 2 * (M + MC_W + M) + INNER_W
GPU_D = M + GPC_D + M + L2_D + M + GPC_D + M + GT_D + 0.02 + PCIE_D + M


@dataclass(frozen=True)
class GpuPlan:
    gpcs: tuple            # R per GPC (0, 1 at the back, 2 at the front left)
    raster: tuple          # the raster engine bar of each GPC
    tpcs: tuple            # R per TPC (12)
    sms: tuple             # R per SM (24): i = 8 g + 2 t + s, s = 0 the back SM
    media: R               # NVENC / NVDEC / display, front right
    l2: R
    gigathread: R
    pcie: R
    mcs: tuple             # 4 memory controllers


@lru_cache(maxsize=None)
def gpu_plan() -> GpuPlan:
    x0 = M + MC_W + M
    y_front = M
    y_l2 = y_front + GPC_D + M
    y_back = y_l2 + L2_D + M
    y_gt = y_back + GPC_D + M
    gpcs = (R(x0, y_back, GPC_W, GPC_D), R(x0 + GPC_W + 0.06, y_back, GPC_W, GPC_D), R(x0, y_front, GPC_W, GPC_D))
    raster, tpcs, sms = [], [], []
    for g, G in enumerate(gpcs):
        mirrored = g == 2            # the front row faces the L2: raster engine at the front
        if mirrored:
            raster.append(R(G.x + M, G.y + M, G.w - 2 * M, RASTER_D))
            ty = G.y + M + RASTER_D + M
        else:
            raster.append(R(G.x + M, G.y1 - M - RASTER_D, G.w - 2 * M, RASTER_D))
            ty = G.y + M
        for t in range(4):
            T = R(G.x + M + t * (TPC_W + TPC_GAP), ty, TPC_W, TPC_D)
            tpcs.append(T)
            back = R(T.x + TPC_PAD, T.y + TPC_PAD + SM_D + SM_GAP, SM_W, SM_D)
            front = R(T.x + TPC_PAD, T.y + TPC_PAD, SM_W, SM_D)
            sms += [back, front]
    media = R(x0 + GPC_W + 0.06, y_front, GPC_W, GPC_D)
    l2 = R(x0, y_l2, INNER_W, L2_D)
    gt = R(x0, y_gt, INNER_W, GT_D)
    pcie = R(x0, y_gt + GT_D + 0.02, INNER_W, PCIE_D)
    h = (GPU_D - 2 * M - M) / 2
    mcs = (R(M, M, MC_W, h), R(M, M + h + M, MC_W, h),
           R(GPU_W - M - MC_W, M, MC_W, h), R(GPU_W - M - MC_W, M + h + M, MC_W, h))
    return GpuPlan(tuple(gpcs), tuple(raster), tuple(tpcs), tuple(sms), media, l2, gt, pcie, mcs)


# ---- CPU plan ------------------------------------------------------------

CPU_W = 2.10
CPU_D = GPU_D
P_W, P_D = 0.70, 0.52
L3_W = 0.48
E_D = 0.52
SA_D = 0.30
IGPU_D = CPU_D - (M + SA_D + M + E_D + M + 3 * P_D + 2 * M + M + M)     # the graphics take what is left


@dataclass(frozen=True)
class CpuPlan:
    igpu: R
    pcores: tuple          # 6, row-major from the back left
    l3: tuple              # one L3 slice per P-core row, between the two columns
    eclusters: tuple       # 2 clusters
    ecores: tuple          # 8
    sa: R
    ring: R                # the ring bus: drawn as this rectangle's outline


@lru_cache(maxsize=None)
def cpu_plan() -> CpuPlan:
    x_l = M
    x_c = M + P_W + M
    x_r = x_c + L3_W + M
    y = CPU_D - M - IGPU_D
    igpu = R(M, y, CPU_W - 2 * M, IGPU_D)
    pcores, l3 = [], []
    rows_top = y - M
    for row in range(3):
        yy = rows_top - (row + 1) * P_D - row * M
        pcores += [R(x_l, yy, P_W, P_D), R(x_r, yy, P_W, P_D)]
        l3.append(R(x_c, yy, L3_W, P_D))
    y_e = M + SA_D + M
    half = (CPU_W - 3 * M) / 2
    eclusters = (R(M, y_e, half, E_D), R(2 * M + half, y_e, half, E_D))
    ecores = []
    for E in eclusters:
        cw, cd = (E.w - 0.08) / 2 - 0.02, (E.d - 0.06) / 2 - 0.02
        for a in range(2):
            for b in range(2):
                ecores.append(R(E.x + 0.03 + a * (cw + 0.03), E.y + 0.03 + b * (cd + 0.03), cw, cd))
    sa = R(M, M, CPU_W - 2 * M, SA_D)
    ring = R(x_c + 0.04, rows_top - 3 * P_D - 2 * M + 0.04, L3_W - 0.08, 3 * P_D + 2 * M - 0.08)
    return CpuPlan(igpu, tuple(pcores), tuple(l3), eclusters, tuple(ecores), sa, ring)


# ---- drawing -------------------------------------------------------------

class Canvas:
    """A PIL canvas for a W x D world-unit area at `ppw` pixels per unit (x SS)."""

    def __init__(self, W: float, D: float, ppw: float, bg: str | None):
        self.W, self.D, self.k = W, D, ppw * SS
        self.size = (max(1, int(round(W * self.k))), max(1, int(round(D * self.k))))
        self.im = Image.new("RGBA", self.size, (0, 0, 0, 0) if bg is None else bg)
        self.dr = ImageDraw.Draw(self.im)

    def box(self, r: R) -> tuple:
        k = self.k
        return (r.x * k, (self.D - r.y1) * k, r.x1 * k - 1, (self.D - r.y) * k - 1)

    def rect(self, r: R, fill: str, edge: str | None = None, width: float = 0.0) -> None:
        w = max(1, int(round(width * self.k))) if edge else 0
        self.dr.rectangle(self.box(r), fill=fill, outline=edge, width=w)

    def text(self, r: R, s: str, color: str = LABEL, height: float = 0.08, bold: bool = False,
             anchor: str = "mm") -> None:
        size = max(6, int(round(height * self.k)))
        font = ImageFont.truetype(FONT_DIR + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
        x0, y0, x1, y1 = self.box(r)
        pos = {"mm": ((x0 + x1) / 2, (y0 + y1) / 2), "lm": (x0 + 0.03 * self.k, (y0 + y1) / 2),
               "lt": (x0 + 0.03 * self.k, y0 + 0.02 * self.k)}[anchor]
        self.dr.text(pos, s, fill=color, font=font, anchor={"mm": "mm", "lm": "lm", "lt": "lt"}[anchor])

    def done(self) -> np.ndarray:
        w, h = self.size
        return np.asarray(self.im.resize((w // SS, h // SS), Image.LANCZOS)).copy()


def _sm(c: Canvas, r: R, state: str, shelf: bool = False) -> None:
    """One SM in NVIDIA's layout: four partitions over L1, texture units, RT core."""
    frame, edge, sched, regs, cores, tensor, ldst, l1, tex, rt = SM[state]
    c.rect(r, frame, edge, 0.012)
    p = 0.018
    inner = r.inset(p)
    rt_d = inner.d * 0.17
    tex_d = inner.d * 0.05
    l1_d = inner.d * 0.07
    part_d = inner.d - rt_d - tex_d - l1_d - 3 * 0.008
    y = inner.y
    c.rect(R(inner.x, y, inner.w, rt_d), rt)
    y += rt_d + 0.008
    for q in range(4):
        tw = (inner.w - 3 * 0.008) / 4
        c.rect(R(inner.x + q * (tw + 0.008), y, tw, tex_d), tex)
    y += tex_d + 0.008
    c.rect(R(inner.x, y, inner.w, l1_d), SHELF_LIT if shelf else l1)
    y += l1_d + 0.008
    pw, pd = (inner.w - 0.012) / 2, (part_d - 0.012) / 2
    for a in range(2):
        for b in range(2):
            P = R(inner.x + a * (pw + 0.012), y + b * (pd + 0.012), pw, pd)
            sd, rd, ld = pd * 0.13, pd * 0.16, pd * 0.12
            c.rect(R(P.x, P.y1 - sd, P.w, sd), sched)
            c.rect(R(P.x, P.y1 - sd - rd - 0.004, P.w, rd), regs)
            c.rect(R(P.x, P.y, P.w, ld), ldst)
            body = R(P.x, P.y + ld + 0.004, P.w, P.d - sd - rd - ld - 0.012)
            cw = body.w * 0.6
            c.rect(R(body.x, body.y, cw, body.d), cores)
            c.rect(R(body.x + cw + 0.004, body.y, body.w - cw - 0.004, body.d), tensor)
            # the core grid: 2 columns x 4 rows of lanes
            for gx in (1,):
                xx = body.x + gx * cw / 2
                c.dr.line([(xx * c.k, (c.D - body.y1) * c.k), (xx * c.k, (c.D - body.y) * c.k)], fill=frame,
                          width=max(1, int(0.003 * c.k)))
            for gy in (1, 2, 3):
                yy = body.y + gy * body.d / 4
                c.dr.line([(body.x * c.k, (c.D - yy) * c.k), ((body.x + cw) * c.k, (c.D - yy) * c.k)], fill=frame,
                          width=max(1, int(0.003 * c.k)))


@lru_cache(maxsize=8)
def gpu_raster(ppw: float) -> np.ndarray:
    """The GPU at rest, every SM off."""
    pl = gpu_plan()
    c = Canvas(GPU_W, GPU_D, ppw, DIE)
    for mc in pl.mcs:
        c.rect(mc, "#202833", "#2f3a46", 0.008)
    c.rect(pl.pcie, "#1f252d", "#2f3740", 0.006)
    c.text(pl.pcie, "PCI Express 4.0", height=0.065)
    c.rect(pl.gigathread, "#3d3328", "#54452f", 0.006)
    c.text(pl.gigathread, "GigaThread Engine", color="#b9a58a", height=0.075)
    c.rect(pl.l2, "#1f3347", "#2c4a66", 0.01)
    c.text(pl.l2, "L2 cache  32 MB", color="#9fb9d4", height=0.13)
    for g, G in enumerate(pl.gpcs):
        c.rect(G, GPC_BG, GPC_EDGE, 0.012)
        c.rect(pl.raster[g], "#1b3a44", None)
    for T in pl.tpcs:
        c.rect(T, GPC_BG, TPC_EDGE, 0.008)
    for S in pl.sms:
        _sm(c, S, "off")
    md = pl.media.inset(M)
    c.rect(pl.media, GPC_BG, GPC_EDGE, 0.012)
    hw = (md.w - 0.04) / 3
    for j, name in enumerate(("NVENC", "NVDEC", "display")):
        b = R(md.x + j * (hw + 0.02), md.y + md.d * 0.35, hw, md.d * 0.5)
        c.rect(b, "#20283a", "#2f3a50", 0.006)
        c.text(b, name, height=0.065)
    return c.done()


@lru_cache(maxsize=64)
def sm_tile(state: str, ppw: float, shelf: bool = False) -> np.ndarray:
    c = Canvas(SM_W, SM_D, ppw, None)
    _sm(c, R(0, 0, SM_W, SM_D), state, shelf)
    return c.done()


def _pcore(c: Canvas, r: R, state: str) -> None:
    core, edge, l2 = PCORE[state]
    c.rect(r, core, edge, 0.012)
    c.rect(R(r.x + r.w * 0.62, r.y + 0.03, r.w * 0.35, r.d - 0.06), l2)


@lru_cache(maxsize=8)
def cpu_raster(ppw: float) -> np.ndarray:
    """The CPU at rest: every core off."""
    pl = cpu_plan()
    k = CPU_COL
    c = Canvas(CPU_W, CPU_D, ppw, k["die"])
    c.rect(pl.igpu, k["igpu"], k["edge"], 0.01)
    ig = pl.igpu.inset(0.06)
    cols, rows = 8, 4
    ew, ed = ig.w / cols, (ig.d - 0.14) / rows
    for a in range(cols):
        for b in range(rows):
            c.rect(R(ig.x + a * ew + 0.01, ig.y + b * ed + 0.01, ew - 0.02, ed - 0.02), k["eu"])
    c.text(R(ig.x, ig.y1 - 0.13, ig.w, 0.13), "Xe graphics", height=0.075)
    for L in pl.l3:
        c.rect(L, k["l3"], "#2c4a66", 0.008)
    ring = pl.ring
    c.rect(ring, k["l3"], k["ring"], 0.014)
    c.text(R(ring.x, ring.y + ring.d / 2 - 0.12, ring.w, 0.24), "L3", color="#9fb9d4", height=0.11)
    c.text(R(ring.x, ring.y + ring.d / 2 - 0.26, ring.w, 0.12), "24 MB", color="#9fb9d4", height=0.07)
    for P in pl.pcores:
        _pcore(c, P, "off")
    for E in pl.eclusters:
        c.rect(E, k["ecore_l2"], k["edge"], 0.01)
    for e in pl.ecores:
        c.rect(e, k["ecore"], "#333c47", 0.006)
    c.rect(pl.sa, k["sa"], k["edge"], 0.01)
    c.text(pl.sa, "memory controller, I/O", height=0.07)
    return c.done()


@lru_cache(maxsize=16)
def pcore_tile(state: str, ppw: float) -> np.ndarray:
    c = Canvas(P_W, P_D, ppw, None)
    _pcore(c, R(0, 0, P_W, P_D), state)
    return c.done()
