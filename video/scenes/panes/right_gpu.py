"""Right pane: a 2D schematic of the GPU that changes per stage.

Static part (same in every state): title, the card, 24 idle SM tiles.
Dynamic part (per GpuSpec): block chips on the tiles, the block close-up
(warps x 32 lanes), shared-memory lines, the global-memory bar with its
boxes, the ch06 kernel strip, and the mono caption lines.
"""
from __future__ import annotations

import colorsys

import numpy as np
from manim import (
    Dot, Line, Rectangle, RoundedRectangle, Square, VGroup, LEFT, RIGHT,
)

from scenes.style import BLUE, GREEN, GREY, GRID, INK, INK_SOFT, RED_PX, TEAL, label
from scenes.panes.config import GpuSpec
from scenes.panes.geometry import Box

TILE, TILE_PITCH = 0.40, 0.50
TILE_IDLE = dict(fill_color=GREY, fill_opacity=0.28, stroke_color=GREY,
                 stroke_width=1.2, stroke_opacity=0.7)
CARD_W, CARD_H = 3.1, 2.4
INSET_H = 1.2
MEM_H = 1.05
TITLE_FONT, SMALL_FONT, LINE_FONT = 16, 11, 14
GOLDEN_ANGLE = 137.508


def block_hex(b: int) -> str:
    """The hue rule of ch03's wavefront renderer (golden angle, red band
    excluded), at the ramp's mid tone."""
    hue = (30.0 + (b * GOLDEN_ANGLE) % 300.0) / 360.0
    r, g, bl = colorsys.hls_to_rgb(hue, 0.55, 0.62)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(bl * 255))


def chip_color(palette: str, b: int) -> str:
    if palette == "single":
        return BLUE
    if palette == "pair":
        return (BLUE, GREEN)[b % 2]
    if palette == "teal":
        return TEAL
    return block_hex(b)


class GpuPane:
    def __init__(self, box: Box, spec: GpuSpec, sm_count: int = 24):
        self.box, self.spec, self.sm_count = box, spec, sm_count
        self.y_card_top = box.y1 - 0.42
        self.y_inset_top = self.y_card_top - CARD_H - 0.18
        self.y_mem_top = self.y_inset_top - INSET_H - 0.12
        self.y_lines_top = self.y_mem_top - MEM_H - 0.30
        self.tiles = VGroup(*[
            RoundedRectangle(corner_radius=0.07, width=TILE, height=TILE, **TILE_IDLE)
            .move_to(self.tile_center(i)) for i in range(sm_count)])
        self.static = self._static()
        self.dynamic = self._dynamic()

    # ---- geometry
    def tile_center(self, i: int) -> np.ndarray:
        row, col = divmod(i, 6)
        return self.box.at(self.box.cx + (col - 2.5) * TILE_PITCH,
                           self.y_card_top - 0.32 - row * TILE_PITCH)

    # ---- static
    def _static(self) -> VGroup:
        title = label(f"the GPU · {self.sm_count} SMs", size=TITLE_FONT, color=INK_SOFT)
        title.move_to(self.box.at(self.box.x0, self.box.y1 - 0.12), aligned_edge=LEFT)
        card = RoundedRectangle(corner_radius=0.12, width=CARD_W, height=CARD_H,
                                stroke_width=1.5, stroke_color=GRID, fill_opacity=0)
        card.move_to(self.box.at(self.box.cx, self.y_card_top - CARD_H / 2))
        sm = label("one tile = one SM · 1,536 threads resident", size=SMALL_FONT, color=GREY)
        sm.move_to(self.box.at(self.box.cx, self.y_card_top - CARD_H + 0.16))
        return VGroup(title, card, self.tiles, sm)

    # ---- dynamic pieces
    def _chips(self) -> VGroup:
        g = VGroup()
        for i, ids in enumerate(self.spec.sm_blocks[:self.sm_count]):
            c = self.tile_center(i)
            if len(ids) == 1:
                g.add(RoundedRectangle(corner_radius=0.04, width=0.26, height=0.26, stroke_width=0,
                                       fill_color=chip_color(self.spec.palette, ids[0]),
                                       fill_opacity=1.0).move_to(c))
            elif len(ids) >= 2:
                for k, b in enumerate(ids[:2]):
                    g.add(RoundedRectangle(corner_radius=0.03, width=0.11, height=0.26, stroke_width=0,
                                           fill_color=chip_color(self.spec.palette, b),
                                           fill_opacity=1.0)
                          .move_to(c + np.array([(-0.075, 0.075)[k], 0, 0])))
        return g

    def _inset(self) -> VGroup:
        """The block close-up: warps x 32 lanes as dots, plus its label."""
        spec = self.spec
        g = VGroup()
        rows = max(1, spec.tpb // 32)
        color = chip_color(spec.palette, 0)
        pitch = 0.09
        x0 = self.box.cx - 31 * pitch / 2
        y = self.y_inset_top - 0.06
        for r in range(rows):
            for c in range(32):
                g.add(Dot(self.box.at(x0 + c * pitch, y - r * pitch), radius=0.028, color=color))
        y_label = y - rows * pitch - 0.08
        g.add(label(f"1 block = {spec.tpb} threads = {rows} warp{'s' if rows > 1 else ''} × 32 lanes",
                    size=SMALL_FONT, color=INK_SOFT).move_to(self.box.at(self.box.cx, y_label)))
        y_next = y_label - 0.2
        for line in spec.shared:
            g.add(label(line, size=SMALL_FONT, color=BLUE).move_to(self.box.at(self.box.cx, y_next)))
            y_next -= 0.18
        if spec.stencil:
            g.add(self._stencil(self.box.at(self.box.cx - 0.55, y_next - 0.12), spec.stencil))
            g.add(label(f"{spec.stencil} neighbours per pixel", size=SMALL_FONT, color=INK_SOFT)
                  .move_to(self.box.at(self.box.cx - 0.28, y_next - 0.12), aligned_edge=LEFT))
        return g

    def _stencil(self, centre: np.ndarray, conn: int) -> VGroup:
        g = VGroup()
        s = 0.11
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                lit = (dx == 0 and dy == 0) or (abs(dx) + abs(dy) == 1) or (conn == 8)
                col = INK if (dx == 0 and dy == 0) else INK_SOFT
                g.add(Square(s, stroke_width=0.8, stroke_color=GRID,
                             fill_color=col, fill_opacity=(1.0 if lit else 0.0))
                      .move_to(centre + np.array([dx * s, dy * s, 0])))
        return g

    def _kernel_strip(self) -> VGroup:
        g = VGroup()
        bw, bh, gap = 0.70, 0.24, 0.06
        rows = [self.spec.kernels[:4], self.spec.kernels[4:]]
        y = self.y_inset_top - 0.14
        for row in rows:
            n = len(row)
            x0 = self.box.cx - (n * bw + (n - 1) * gap) / 2 + bw / 2
            for i, (name, blocks) in enumerate(row):
                c = self.box.at(x0 + i * (bw + gap), y)
                g.add(RoundedRectangle(corner_radius=0.06, width=bw, height=bh, stroke_width=1.2,
                                       stroke_color=TEAL, fill_color=TEAL, fill_opacity=0.15).move_to(c))
                g.add(label(name, size=SMALL_FONT, color=INK_SOFT).move_to(c))
                g.add(label(blocks, size=10, color=GREY).move_to(c + np.array([0, -0.21, 0])))
            y -= 0.50
        g.add(label("7 plain launches · 256 threads each", size=SMALL_FONT, color=INK_SOFT)
              .move_to(self.box.at(self.box.cx, self.y_inset_top - INSET_H + 0.06)))
        return g

    def _memory(self) -> VGroup:
        spec = self.spec
        g = VGroup()
        bar = RoundedRectangle(corner_radius=0.1, width=CARD_W, height=MEM_H, stroke_width=1.5,
                               stroke_color=GRID, fill_opacity=0)
        bar.move_to(self.box.at(self.box.cx, self.y_mem_top - MEM_H / 2))
        g.add(bar)
        g.add(label(spec.memory_title, size=SMALL_FONT, color=GREY)
              .move_to(self.box.at(self.box.x0 + 0.22, self.y_mem_top - 0.14), aligned_edge=LEFT))
        boxes = []
        for item in spec.memory:
            t = label(item, size=SMALL_FONT, color=INK_SOFT)
            w = t.width + 0.18
            extra = None
            if spec.queue_chips and item.startswith("queue"):
                extra = VGroup(Square(0.09, stroke_width=0, fill_color=BLUE, fill_opacity=1),
                               Square(0.09, stroke_width=0, fill_color=GREEN, fill_opacity=1))
                extra.arrange(RIGHT, buff=0.04)
                w += extra.width + 0.08
            r = RoundedRectangle(corner_radius=0.05, width=w, height=0.26, stroke_width=1.0,
                                 stroke_color=GREY, fill_color=GREY, fill_opacity=0.12)
            boxes.append((r, t, extra, w))
        # greedy wrap into rows of at most CARD_W - 0.3
        rows, cur, cur_w = [], [], 0.0
        for b in boxes:
            if cur and cur_w + 0.1 + b[3] > CARD_W - 0.3:
                rows.append(cur)
                cur, cur_w = [], 0.0
            cur.append(b)
            cur_w += b[3] + (0.1 if len(cur) > 1 else 0)
        if cur:
            rows.append(cur)
        y = self.y_mem_top - 0.44
        for row in rows:
            total = sum(b[3] for b in row) + 0.1 * (len(row) - 1)
            x = self.box.cx - total / 2
            for r, t, extra, w in row:
                c = self.box.at(x + w / 2, y)
                r.move_to(c)
                if extra is not None:
                    t.move_to(c + np.array([-(extra.width + 0.08) / 2, 0, 0]))
                    extra.move_to(c + np.array([(t.width + 0.08) / 2, 0, 0]))
                    g.add(r, t, extra)
                else:
                    t.move_to(c)
                    g.add(r, t)
                x += w + 0.1
            y -= 0.32
        return g

    def _cpu_box(self) -> VGroup:
        g = VGroup()
        c = self.box.at(self.box.cx, self.y_inset_top - INSET_H / 2)
        cpu = RoundedRectangle(corner_radius=0.1, width=1.8, height=0.9, stroke_width=1.5,
                               stroke_color=GRID, fill_opacity=0).move_to(c)
        core = RoundedRectangle(corner_radius=0.04, width=0.34, height=0.34, stroke_width=0,
                                fill_color=INK, fill_opacity=1.0).move_to(c + np.array([-0.45, 0, 0]))
        t1 = label("CPU", size=SMALL_FONT, color=GREY).move_to(cpu.get_corner(np.array([-1, 1, 0])) + np.array([0.28, -0.14, 0]))
        t2 = label("1 core", size=SMALL_FONT, color=INK_SOFT).move_to(c + np.array([0.35, 0.12, 0]))
        t3 = label("1 pixel per step", size=SMALL_FONT, color=INK_SOFT).move_to(c + np.array([0.35, -0.12, 0]))
        g.add(cpu, core, t1, t2, t3)
        return g

    def _lines(self) -> VGroup:
        g = VGroup()
        for i, line in enumerate(self.spec.lines):
            g.add(label(line.format(sm=self.sm_count), size=LINE_FONT, color=INK_SOFT, mono=True)
                  .move_to(self.box.at(self.box.x0 + 0.05, self.y_lines_top - i * 0.28), aligned_edge=LEFT))
        return g

    def _dynamic(self) -> VGroup:
        g = VGroup()
        if self.spec.mode == "cpu":
            g.add(self._cpu_box())
        else:
            g.add(self._chips())
            g.add(self._kernel_strip() if self.spec.kernels else self._inset())
        g.add(self._memory())
        g.add(self._lines())
        return g

    def all(self) -> list:
        return [self.static, self.dynamic]
