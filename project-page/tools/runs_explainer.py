#!/usr/bin/env python
"""Build the chapter 6 "runs" explainer for the project page: one real blob
travels through the four memory layouts of the chapter 6 pipeline (the RGB
image, the 1-bit mask, the run table, the painted image), and the video shows
which bytes move together at every step.

Run from the repo root (the directory that holds src/ and project-page/):

    FFMPEG=/path/to/ffmpeg python project-page/tools/runs_explainer.py

or through project-page/tools/make_runs_explainer.sh.

FFMPEG  ffmpeg with libx264 (default: ffmpeg on PATH)
FONT, FONT_BOLD, FONT_MONO  TTF files (default: DejaVu Sans, DejaVu Sans Bold
        and DejaVu Sans Mono Bold under /usr/share/fonts)

Needs numpy and Pillow (built with WebP support). No GPU. Frames go to ffmpeg
as raw RGB on a pipe, the same pattern as make_race.py. Nothing but the four
outputs is written to the repo; --still writes one PNG, and only outside the
repo's project-page folder.

Outputs:
    project-page/static/videos/runs_explainer.mp4             1280x720, about 45 s
    project-page/static/images/runs_explainer_poster.webp     the 20-chunk paint frame
    project-page/static/videos/carousel/runs.mp4              720x720 loop, 9 s
    project-page/static/images/carousel/runs.webp             its poster

Data: tools/runs_blob.json holds the mask of blob 678 of the 9000 x 9000
benchmark image (24 rows x 64 columns, the two packed words the kernels read),
its 34 runs with their real ids, and the lock-step merge schedule. Everything
else is recomputed here and asserted against the JSON before a frame is drawn:
the runs, the row counts, the 8-connectivity partners, both merge rounds, the
byte counts, and (from the committed benchmark JSON) the 58.51 ms, 2.96 ms,
19.8x, 27,000 B and 243 MB figures.

What is real and what is a drawing device
-----------------------------------------
Real: the blob, its mask words, its runs, the run ids, the 20 chunks of 32
runs the blob falls into, the byte counts, the timings.
Device: the order. The video shows the rows leaving the picture one by one,
the merge in two lock-step rounds and the 20 chunks stepping together. The GPU
runs warps in waves in no fixed order. Notes on screen say so ("one possible
order", "Real order varies. Shown in step.").

Design time and video time: every beat below is authored in "design time"
(the numbers inside the drawing functions). WARP_SEGMENTS maps the final video
clock to design time, which slows a few beats down so that their caption can be
read (and speeds two of them up a little). To change how long a beat stays on
screen, change its entry in WARP_SEGMENTS, not the drawing code.

Axes: the kernels' x is the image column and y is the image row, so the video
turns the picture over its diagonal first (a reflection, not a rotation) and
then keeps the kernel view: one row = one line of memory, y runs to the right.
Bit 0 of a word is the smallest y, so it is drawn on the left.
"""

import argparse
import colorsys
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SPEC_PATH = os.path.join(HERE, "runs_blob.json")
BENCH_PATH = os.path.join(REPO, "src", "flood_fill_cuda", "results", "ch06_gpu_nblob_runs",
                          "benchmark_results", "runs_20260725T161448Z.json")
STATIC = os.path.join(REPO, "project-page", "static")
OUT_VIDEO = os.path.join(STATIC, "videos", "runs_explainer.mp4")
OUT_POSTER = os.path.join(STATIC, "images", "runs_explainer_poster.webp")
OUT_CAROUSEL = os.path.join(STATIC, "videos", "carousel", "runs.mp4")
OUT_CAROUSEL_POSTER = os.path.join(STATIC, "images", "carousel", "runs.webp")

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# ---------------------------------------------------------------- constants
W, H = 1280, 720
FPS = 30
CAR_SIZE = 720
CAR_DURATION = 9.0
POSTER_D = 34.1            # design time of the poster: 20 chunks paint together, half the blob painted
CAR_POSTER_T = 6.4         # half painted in both views

# (design_from, design_to, seconds on screen): the final clock, beat by beat
WARP_SEGMENTS = [
    (0.0, 1.8, 2.4),       # one blob (caption 2.2 s), flipped over its diagonal
    (1.8, 3.0, 2.0),       # each row is one line of memory (caption 2.0 s)
    (3.0, 4.4, 2.15),      # memory is one long line (caption 2.15 s)
    (4.4, 7.9, 2.4),       # the peel and the zoom
    (7.9, 9.0, 2.2),       # 34 runs hold 301 pixels (caption 2.2 s)
    (9.0, 11.0, 2.15),     # pack: 32 pixels (caption 2.15 s)
    (11.0, 13.2, 2.05),    # pack: one word
    (13.2, 15.8, 2.0),     # 24 times fewer bytes
    (15.8, 18.2, 2.5),     # count: one warp takes 32 words
    (18.2, 21.6, 2.5),     # emit: a row's runs side by side
    (21.6, 23.0, 3.0),     # merge: touching runs in the next row join (caption 3 s)
    (23.0, 24.2, 1.5),     # merge: everyone at once, roots 34
    (24.2, 25.2, 1.2),     # roots 3
    (25.2, 26.4, 1.4),     # roots 1, probes and repeats
    (26.4, 27.6, 2.0),     # flatten (caption 1.9 s)
    (27.6, 29.6, 2.15),    # paint: a warp loads 32 runs (caption 2.15 s)
    (29.6, 32.2, 2.4),     # paint: one run at a time
    (32.2, 36.4, 2.6),     # 20 chunks step together
    (36.4, 42.8, 6.4),     # ladder, then the race (1 : 1, the race clock is real time)
]
DESIGN_END = WARP_SEGMENTS[-1][1]
DURATION = sum(seg[2] for seg in WARP_SEGMENTS)
S = 2                      # supersampling: draw at 2x, shrink with Lanczos

WHITE = (255, 255, 255)
INK = (0x1F, 0x23, 0x28)
GREY = (0x6B, 0x6B, 0x6B)
TEAL = (0x1A, 0x7F, 0x72)
AMBER = (0x9A, 0x67, 0x00)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
CYAN_EDGE = (0x00, 0xA6, 0xA6)
FOREIGN = (0xD0, 0xD0, 0xD0)
CELL_BG = (0xF1, 0xF1, 0xF1)
PANEL = (0xF4, 0xF4, 0xF4)
HAIR = (0xB4, 0xB4, 0xB4)
BAR_GREY = (0x8C, 0x8C, 0x8C)


# ------------------------------------------------------------------- fonts
def find_font(env, names):
    path = os.environ.get(env)
    if path:
        return path
    for root in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
                 "/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype"):
        for name in names:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand
    sys.exit("runs_explainer.py: no TTF found, set %s to a font file" % env)


FONT_FILES = {
    "reg": find_font("FONT", ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf"]),
    "bold": find_font("FONT_BOLD", ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf"]),
    "mono": find_font("FONT_MONO", ["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "NotoSansMono-Bold.ttf"]),
}
_font_cache = {}


def get_font(kind, size):
    key = (kind, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_FILES[kind], int(round(size * S)))
    return _font_cache[key]


def text_w(s, size, kind="bold"):
    return get_font(kind, size).getlength(s) / S


# ------------------------------------------------------------------- maths
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ramp(t, t0, t1):
    return clamp01((t - t0) / (t1 - t0))


def ease(p):
    p = clamp01(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def lerp(a, b, p):
    return a + (b - a) * p


def mix(c, a, bg=WHITE):
    """c at opacity a over the flat colour bg."""
    a = clamp01(a)
    return tuple(int(round(bg[i] + (c[i] - bg[i]) * a)) for i in range(3))


def vis(t, t0, t1, fi=0.25, fo=0.25):
    """0 outside [t0, t1], 1 inside, linear ramps of fi and fo at the edges."""
    a = ramp(t, t0, t0 + fi) if fi > 0 else (1.0 if t >= t0 else 0.0)
    b = 1.0 - ramp(t, t1 - fo, t1) if fo > 0 else (1.0 if t < t1 else 0.0)
    return min(a, b)


def keyframes(t, keys):
    """Piecewise eased interpolation of tuples: keys = [(t, (v0, v1, ...)), ...]."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (ta, va), (tb, vb) in zip(keys, keys[1:]):
        if t <= tb:
            p = ease((t - ta) / (tb - ta)) if tb > ta else 1.0
            return tuple(lerp(a, b, p) for a, b in zip(va, vb))
    return keys[-1][1]


def design_time(t):
    """Video seconds -> design seconds (see WARP_SEGMENTS)."""
    v0 = 0.0
    for d0, d1, dur in WARP_SEGMENTS:
        if t < v0 + dur:
            return d0 + (d1 - d0) * (t - v0) / dur
        v0 += dur
    return DESIGN_END + (t - v0)


def video_time(d):
    """Design seconds -> video seconds (the inverse of design_time)."""
    v0 = 0.0
    for d0, d1, dur in WARP_SEGMENTS:
        if d < d1:
            return v0 + dur * (d - d0) / (d1 - d0)
        v0 += dur
    return v0 + (d - DESIGN_END)


# ------------------------------------------------------------------ canvas
class Cv:
    """A drawing surface in design pixels, backed by a 2x image."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * S, h * S), WHITE)
        self.d = ImageDraw.Draw(self.img)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, w=1.0):
        X0, Y0, X1, Y1 = (int(round(v * S)) for v in (x0, y0, x1, y1))
        if X1 <= X0:
            X1 = X0 + 1
        if Y1 <= Y0:
            Y1 = Y0 + 1
        self.d.rectangle((X0, Y0, X1 - 1, Y1 - 1), fill=fill, outline=outline,
                         width=max(1, int(round(w * S))))

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, w=1.0):
        X0, Y0, X1, Y1 = (int(round(v * S)) for v in (x0, y0, x1, y1))
        if X1 - X0 < 2 or Y1 - Y0 < 2:
            return
        self.d.rounded_rectangle((X0, Y0, X1 - 1, Y1 - 1), radius=int(r * S), fill=fill,
                                 outline=outline, width=max(1, int(round(w * S))))

    def line(self, x0, y0, x1, y1, color, w=1.0):
        self.d.line(((x0 * S, y0 * S), (x1 * S, y1 * S)), fill=color, width=max(1, int(round(w * S))))

    def dline(self, x0, y0, x1, y1, color, w=1.0, dash=6.0, gap=5.0):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            self.line(x0 + ux * pos, y0 + uy * pos, x0 + ux * end, y0 + uy * end, color, w)
            pos += dash + gap

    def drect(self, x0, y0, x1, y1, color, w=1.0, dash=6.0, gap=4.0):
        self.dline(x0, y0, x1, y0, color, w, dash, gap)
        self.dline(x1, y0, x1, y1, color, w, dash, gap)
        self.dline(x1, y1, x0, y1, color, w, dash, gap)
        self.dline(x0, y1, x0, y0, color, w, dash, gap)

    def dot(self, x, y, r, color):
        self.d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=color)

    def poly(self, pts, color):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=color)

    def arrow(self, x0, y0, x1, y1, color, w=2.0, head=8.0, both=False):
        self.line(x0, y0, x1, y1, color, w)
        for (ax, ay, bx, by) in ((x0, y0, x1, y1),) + (((x1, y1, x0, y0),) if both else ()):
            ang = math.atan2(by - ay, bx - ax)
            tip = (bx, by)
            l = (bx - head * math.cos(ang - 0.45), by - head * math.sin(ang - 0.45))
            r = (bx - head * math.cos(ang + 0.45), by - head * math.sin(ang + 0.45))
            self.poly([tip, l, r], color)

    def text(self, x, y, s, size, color, kind="bold", anchor="la"):
        self.d.text((x * S, y * S), s, font=get_font(kind, size), fill=color, anchor=anchor)

    def finish(self):
        return self.img.resize((self.w, self.h), Image.LANCZOS)


# -------------------------------------------------------------------- data
class Blob:
    """The blob in kernel view plus everything derived from its mask."""

    def __init__(self, spec, bench):
        fr = spec["frame"]
        rows = spec["mask"]
        self.n_rows = fr["rows"]
        assert len(rows) == self.n_rows == 24 and all(len(r) == fr["cols"] == 64 for r in rows)
        self.mask = np.array([[ch == "#" for ch in r] for r in rows])
        self.y_origin = fr["global_y0"]                  # 2,848: local y 0 of the window
        self.c0, c1 = fr["blob_cols"]
        self.n_cols = c1 - self.c0 + 1
        assert self.n_cols == 28
        assert not self.mask[:, :self.c0].any() and not self.mask[:, c1 + 1:].any()
        self.K = self.mask[:, self.c0:c1 + 1]            # (24, 28), kernel view
        self.n_px = int(self.mask.sum())
        assert self.n_px == 301

        # runs, found the way emit orders them: by row, then by y0
        runs = []
        for i in range(self.n_rows):
            j = 0
            while j < self.n_cols:
                if self.K[i, j]:
                    j0 = j
                    while j + 1 < self.n_cols and self.K[i, j + 1]:
                        j += 1
                    runs.append({"id": len(runs), "row": i, "j0": j0, "j1": j,
                                 "y0": self.c0 + j0, "y1": self.c0 + j, "n": j - j0 + 1})
                j += 1
        self.runs = runs
        assert len(runs) == spec["totals"]["runs"] == 34
        assert sum(r["n"] for r in runs) == self.n_px
        for r, s in zip(runs, spec["runs"]):
            assert (r["id"], r["row"], r["y0"], r["y1"]) == (s["id"], s["x"], s["y0"], s["y1"])
            r["gid"], r["chunk"], r["cross"] = s["global_id"], s["chunk32"], s["crosses_word_boundary"]
            r["off"] = r["gid"] - 32 * r["chunk"]
            assert 0 <= r["off"] < 32
            assert r["cross"] == ((r["y0"] // 32) != (r["y1"] // 32))
        assert sum(r["cross"] for r in runs) == spec["totals"]["runs_crossing_word_boundary"] == 16

        # row counts and the exclusive scan (count and scan kernels)
        self.row_count = [sum(1 for r in runs if r["row"] == i) for i in range(self.n_rows)]
        self.row_off = [0]
        for c in self.row_count:
            self.row_off.append(self.row_off[-1] + c)
        assert self.row_off[-1] == 34 and self.row_off[4] == 3 and self.row_count[5] == 4
        self.group_rows = [i for i in range(self.n_rows) if self.row_count[i]]
        self.group_of = [self.group_rows.index(r["row"]) for r in runs]

        # the two packed words per row: bit b of word w is column 32 w + b (bit 0 = smallest y)
        self.word_val = [[sum(int(self.mask[i, 32 * w + b]) << b for b in range(32)) for w in range(2)]
                         for i in range(self.n_rows)]
        assert self.word_val[4] == [0xE8000000, 0x3F]
        # count: one warp step reads 32 words (lane l takes word base + l); both words of the blob
        # (89 and 90) sit in the same group of 32, words 64-95
        self.word0 = fr["first_global_word"]
        self.group0 = 32 * (self.word0 // 32)
        assert (self.word0, self.group0) == (89, 64) and (self.word0 + 1) // 32 == self.word0 // 32
        r3, r4 = runs[3], runs[4]
        assert (r3["y0"], r3["y1"], r4["y0"], r4["y1"], r4["cross"]) == (27, 27, 29, 37, True)

        # 8-connectivity partners in the next row and the two lock-step merge rounds
        self.partners = [[s["id"] for s in runs if s["row"] == r["row"] + 1
                          and s["y1"] >= r["y0"] - 1 and s["y0"] <= r["y1"] + 1] for r in runs]
        assert sum(len(p) for p in self.partners) == spec["totals"]["links"] == 38
        parent = list(range(len(runs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        self.rounds = []
        for j in range(max(len(p) for p in self.partners)):
            attempts = []
            for r in runs:
                if j < len(self.partners[r["id"]]):
                    p = self.partners[r["id"]][j]
                    a, b = find(r["id"]), find(p)
                    linked = a != b
                    if linked:
                        parent[max(a, b)] = min(a, b)
                    attempts.append((r["id"], p, linked))
            self.rounds.append({"attempts": attempts, "parent": list(parent)})
        assert len(self.rounds) == len(spec["union_rounds"]) == 2
        for mine, theirs in zip(self.rounds, spec["union_rounds"]):
            assert mine["attempts"] == [tuple(a) for a in theirs["attempts"]]
            assert mine["parent"] == theirs["parent_after"]
        self.n_links = sum(1 for rd in self.rounds for a in rd["attempts"] if a[2])
        assert (self.n_links, 38 - self.n_links) == (spec["totals"]["linked"], spec["totals"]["redundant"]) == (33, 5)
        assert [len(rd["attempts"]) for rd in self.rounds] == [31, 7]
        self.roots = [sum(1 for i, p in enumerate(rd["parent"]) if i == p) for rd in self.rounds]
        assert self.roots == [3, 1] and self.rounds[1]["parent"][10] == 8 and self.rounds[1]["parent"][33] == 31

        def root_of(par, i):
            while par[i] != i:
                i = par[i]
            return i

        # root of every run after round 1 and after round 2 (round 2 is the final state)
        self.root_after = [[root_of(rd["parent"], i) for i in range(len(runs))] for rd in self.rounds]
        assert sorted(set(self.root_after[0])) == [0, 19, 31] and set(self.root_after[1]) == {0}
        # flatten stamps the root into every parent cell: only cells 10 and 33 are deeper than 1 before it
        deep = [i for i, p in enumerate(self.rounds[1]["parent"]) if p != self.root_after[1][i]]
        assert deep == [10, 33]
        assert spec["root_run_id"] == 0 and runs[0]["row"] == 2 and runs[0]["y0"] == 34

        # the label of the whole blob: root's x * height + y0, in the real image
        self.label = spec["canonical_label_global"]
        gx, gy = fr["global_x0"] + runs[0]["row"], fr["global_y0"] + runs[0]["y0"]
        assert self.label == gx * bench["height"] + gy == 22070882

        # byte counts of the ladder (algorithmic bytes, not DRAM traffic)
        self.b_rgb = self.n_rows * 64 * 3
        self.b_mask = self.n_rows * 2 * 4
        self.b_runs = len(runs) * 12
        self.b_paint = self.n_px * 3
        fp = spec["footprint"]
        assert (self.b_rgb, self.b_mask, self.b_runs, self.b_paint) == (
            fp["rgb_bytes_under_the_words"], fp["mask_bytes"], fp["descriptor_bytes"], fp["painted_bytes"])
        assert (self.b_rgb, self.b_mask, self.b_runs, self.b_paint) == (4608, 192, 408, 903)
        assert fp["pack_warp_reads"] == 48 and fp["pack_bytes_per_warp"] == 96

        # the 20 chunks of 32 runs the blob lands in, with 606 foreign runs
        self.chunks = sorted({r["chunk"] for r in runs})
        assert len(self.chunks) == fp["distinct_32run_chunks"] == len(spec["run_chunks32"]) == 20
        assert [c["chunk"] for c in spec["run_chunks32"]] == self.chunks
        assert 32 * len(self.chunks) - len(runs) == fp["other_runs_sharing_those_chunks"] == 606
        for r in runs:
            r["crow"] = self.chunks.index(r["chunk"])
        self.rows_runs_total = spec["global_runs_in_blob_rows_total"]    # 1,637, from the PNG (not shown)
        assert self.rows_runs_total == 1637 and round(self.rows_runs_total / self.n_rows) == 68
        # the local ids 0-33 stand for the real ids 147,170 to 148,457 (same order, same roots)
        assert (runs[0]["gid"], runs[-1]["gid"]) == (147170, 148457)
        assert all(a["gid"] < b["gid"] for a, b in zip(runs, runs[1:]))
        # merge example: run 4 reaches runs 6 and 7 of row 5, run 8 starts too far right (y0 40 > 37 + 1)
        assert self.partners[4] == [6, 7] and runs[8]["y0"] > runs[4]["y1"] + 1
        assert [r["id"] for r in runs if r["row"] == 5] == [5, 6, 7, 8]
        # chunk 4604 holds runs 5-8 of row 5 at offsets 17-20 (the paint walk-through)
        self.walk = [r for r in runs if r["chunk"] == 4604]
        assert [r["id"] for r in self.walk] == [5, 6, 7, 8] and [r["off"] for r in self.walk] == [17, 18, 19, 20]
        assert [spec["paint_bytes_per_run"][i] for i in (5, 6, 7, 8)] == [9, 12, 18, 12]

        # whole-image figures from the committed benchmark JSON
        self.height, self.width = bench["height"], bench["width"]
        self.row_bytes = self.height * 3
        self.img_bytes = bench["n_pixels"] * 3
        self.mask_bytes_all = self.width * math.ceil(self.height / 32) * 4
        self.ch5_ms = bench["ch05"]["median_ms"]
        self.ch6_ms = bench["ch06"]["rgb"]["median_ms"]
        self.speedup = bench["ch06"]["rgb"]["speedup_vs_ch05"]
        self.words_per_row = math.ceil(self.height / 32)
        self.count_steps = math.ceil(self.words_per_row / 32)
        assert self.words_per_row == 282 and self.count_steps == 9      # count: 32 words per warp step
        assert self.row_bytes == 27000 and self.img_bytes == 243000000 and self.mask_bytes_all == 10152000
        assert "%.2f" % self.ch5_ms == "58.51" and "%.2f" % self.ch6_ms == "2.96"
        assert abs(self.ch5_ms / self.ch6_ms - self.speedup) < 1e-6 and "%.1f" % self.speedup == "19.8"
        assert bench["n_runs"] == 539207 and bench["n_blobs"] == 2522
        self.runs_per_row = round(bench["n_runs"] / bench["width"])      # 59.9 on average
        assert self.runs_per_row == 60


def load_blob():
    with open(SPEC_PATH) as fh:
        spec = json.load(fh)
    with open(BENCH_PATH) as fh:
        bench_all = json.load(fh)
    scene = bench_all["scenes"][0]
    assert scene["scene"] == "input_blobs"
    return Blob(spec, scene)


# ------------------------------------------------------------ timeline cues
CAPTIONS = [
    (0.2, "One blob. 301 red pixels."),
    (1.8, "Each row is one line of memory."),
    (3.0, "Memory is one long line."),
    (4.4, "A run is a piece of a row."),
    (7.9, "34 runs hold 301 pixels."),
    (9.0, "Read 32 pixels together."),
    (11.0, "Keep one bit each: 1 word."),
    (13.2, "24 times fewer bytes."),
    (15.8, "Count the runs in each row."),
    (18.2, "Each row's runs sit side by side."),
    (21.6, "Touching runs in the next row join."),
    (23.0, "All runs at once."),
    (26.4, "One label for the whole blob."),
    (27.6, "A warp loads 32 runs together."),
    (29.6, "Then paints them one by one."),
    (32.2, "20 chunks step together."),
    (36.4, "Move less. Move in groups."),
    (39.5, "Same job, whole image."),
]
CHIPS = [("pack", AMBER, 9.0, 13.2), ("count scan emit", TEAL, 15.8, 21.6),
         ("merge", TEAL, 21.6, 26.4), ("flatten", TEAL, 26.4, 27.6), ("paint", AMBER, 27.6, 36.4)]
# text, colour, with the "moves together" label, from, to
PILLS = [("32 px = 96 B", AMBER, True, 9.0, 11.0), ("1 word = 4 B", AMBER, True, 11.0, 13.2),
         ("32 words = 128 B", TEAL, True, 15.8, 18.2), ("a row's runs", TEAL, True, 18.2, 21.6),
         ("1 thread per run", TEAL, False, 23.0, 26.4),
         ("32 runs = 4 x 128 B", AMBER, True, 27.6, 29.6), ("1 run = 1 write", AMBER, True, 29.6, 32.2)]
TITLES = [(3.8, "the image, 3 B per pixel"), (11.5, "the mask, 1 bit per pixel"),
          (17.9, "the run table"), (27.8, "run table to image"), (32.2, None),
          (36.6, "one blob, four layouts"), (39.3, None)]
# (t, (x0, y0, x1, y1, alpha)) keyframes of the grey memory panel
PANEL_KEYS = [(3.3, (24, 560, 1256, 708, 0.0)), (4.0, (24, 560, 1256, 708, 1.0)),
              (9.2, (24, 560, 1256, 708, 1.0)), (10.0, (24, 140, 1256, 708, 1.0)),
              (13.2, (24, 140, 1256, 708, 1.0)), (13.7, (24, 560, 1256, 708, 1.0)),
              (16.0, (24, 560, 1256, 708, 1.0)), (16.7, (24, 436, 1256, 712, 1.0)),
              (27.6, (24, 436, 1256, 712, 1.0)), (28.2, (24, 140, 1256, 708, 1.0)),
              (32.2, (24, 140, 1256, 708, 1.0)), (32.9, (36, 140, 444, 624, 1.0)),
              (36.4, (36, 140, 444, 624, 1.0)), (37.1, (24, 140, 1256, 708, 1.0)),
              (39.3, (24, 140, 1256, 708, 1.0)), (39.5, (24, 140, 1256, 708, 0.0))]
# (t, (x, y, cell, alpha)) keyframes of the picture of the blob in kernel view
THUMB_KEYS = [(3.0, (360, 190, 20, 1.0)), (3.8, (48, 150, 16, 1.0)), (9.0, (48, 150, 16, 1.0)),
              (9.35, (48, 150, 16, 0.0)), (13.5, (48, 150, 16, 0.0)), (13.95, (48, 150, 16, 1.0)),
              (15.8, (48, 150, 16, 1.0)), (16.5, (48, 150, 11, 1.0)), (26.4, (48, 150, 11, 1.0)),
              (26.9, (48, 150, 11, 0.3)), (27.6, (48, 150, 11, 0.3)), (28.0, (48, 150, 11, 0.0))]

# the peel: when each row of the picture is copied into the memory lane
FLIGHTS = {0: (4.5, 4.6), 1: (4.6, 4.7), 2: (4.7, 5.0), 3: (5.0, 5.3), 4: (5.3, 5.8)}
for _r in range(5, 24):
    FLIGHTS[_r] = (6.2 + 0.05 * (_r - 5), 6.2 + 0.05 * (_r - 5) + 0.3)
ZOOM = (6.2, 8.0)
LANE_X0, LANE_X1, LANE_TOP = 54, 1226, 602
GAP_U = 0.9                # a fold between two rows, in cells
ROW_U = 28 + GAP_U


# ----------------------------------------------------------- small drawing
def draw_note(cv, x, y, s, a=1.0, size=30, anchor="la", color=GREY, bg=WHITE, kind="bold"):
    if a > 0.01:
        cv.text(x, y, s, size, mix(color, a, bg), kind, anchor)


def draw_chip(cv, x, y, s, color, a, size=36):
    """A rounded 3 px frame around mono bold text. Returns the right edge."""
    w = text_w(s, size, "mono") + 30
    if a > 0.01:
        c = mix(color, a)
        cv.rrect(x, y, x + w, y + 48, 12, fill=WHITE, outline=c, w=3)
        cv.text(x + w / 2, y + 25, s, size, c, "mono", "mm")
    return x + w


def draw_header(cv, B, t):
    # caption: the old line fades out, then the new one fades in
    cur = max((i for i, (ts, _) in enumerate(CAPTIONS) if t >= ts), default=None)
    if cur is not None:
        ts, text = CAPTIONS[cur]
        a = ramp(t, ts + 0.1, ts + 0.25)
        if cur > 0 and t < ts + 0.1:
            cv.text(48, 22, CAPTIONS[cur - 1][1], 44, mix(INK, 1 - ramp(t, ts, ts + 0.08)), "bold")
        if a > 0.01:
            cv.text(48, 22, text, 44, mix(INK, a), "bold")
    # chip, then the "moves together" pill
    chip_end, best = 48, 0.0
    for text, color, t0, t1 in CHIPS:
        a = vis(t, t0, t1, 0.2, 0.15)
        if a > 0:
            end = draw_chip(cv, 48, 88, text, color, a)
            if a >= best:
                best, chip_end = a, end
    for text, color, label, t0, t1 in PILLS:
        a = vis(t, t0, t1, 0.25, 0.15)
        if a <= 0:
            continue
        x = chip_end + 20 if chip_end > 48 else 48       # the pill sits next to the chip, never over it
        if label:
            cv.text(x, 112, "moves together:", 26, mix(GREY, a), "reg", "lm")
            x += text_w("moves together:", 26, "reg") + 12
        w = text_w(text, 40, "mono") + 30
        c = mix(color, a)
        cv.rrect(x, 85, x + w, 137, 14, fill=WHITE, outline=c, w=3)
        cv.text(x + w / 2, 112, text, 40, c, "mono", "mm")
    # the 20-chunk beat names the neighbours
    na = vis(t, 32.4, 36.4, 0.3, 0.3)
    if na > 0.01:
        cv.text(chip_end + 20, 112, "%d other runs share them" % (32 * len(B.chunks) - len(B.runs)), 30,
                mix(GREY, na), "bold", "lm")


def draw_panel(cv, t):
    x0, y0, x1, y1, a = keyframes(t, PANEL_KEYS)
    if a < 0.01:
        return
    cv.rrect(x0, y0, x1, y1, 18, fill=mix(PANEL, a))
    cur = max((i for i, (ts, _) in enumerate(TITLES) if t >= ts), default=None)
    if cur is None:
        return
    ts, text = TITLES[cur]
    pairs = [(text, ramp(t, ts + 0.12, ts + 0.3))]
    if cur > 0 and t < ts + 0.1:
        pairs.append((TITLES[cur - 1][1], 1 - ramp(t, ts, ts + 0.08)))
    # row 4 flies up from the lane to its strip while the panel grows: keep the title out of its way
    away = 1.0 - vis(t, 9.0, 10.4, 0.1, 0.3)
    for s, f in pairs:
        if s and f > 0.01:
            cv.text(x0 + 24, y0 + 6, "GLOBAL MEMORY: " + s, 28, mix(GREY, f * a * away, PANEL), "mono")


def tray(cv, x0, y0, x1, y1, a=1.0):
    if a > 0.01:
        cv.rrect(x0, y0, x1, y1, 8, fill=mix(WHITE, a, PANEL))


def draw_fold_simple(cv, x, y, a, bg=PANEL):
    col = mix(GREY, a, bg)
    for dx in (-2.5, 2.5):
        cv.line(x + dx - 2.5, y + 7, x + dx + 2.5, y - 7, col, 1.5)


# ------------------------------------------------------- the picture (thumb)
def cell_rect(x, y, c, i, j, gap=1.0):
    return x + c * j, y + c * i, x + c * (j + 1) - gap, y + c * (i + 1) - gap


def run_rect(x, y, c, r):
    return x + c * r["j0"], y + c * r["row"], x + c * (r["j1"] + 1), y + c * (r["row"] + 1)


def run_center(x, y, c, r):
    return x + c * (r["j0"] + r["j1"] + 1) / 2.0, y + c * (r["row"] + 0.5)


def draw_blob_cells(cv, B, x, y, c, alpha, bg=WHITE, focus=None, amt=0.0, colors=None):
    """Window cells and red cells in kernel view. Runs outside `focus` are
    drawn paler by `amt`. `colors` (run id -> colour) replaces the red."""
    cell = mix(CELL_BG, alpha, bg)
    for i in range(B.n_rows):
        for j in range(B.n_cols):
            if not B.K[i, j]:
                cv.rect(*cell_rect(x, y, c, i, j), fill=cell)
    for r in B.runs:
        a = alpha * (1 - 0.65 * amt if (focus is not None and r["id"] not in focus) else 1.0)
        col = mix(colors[r["id"]] if colors else RED, a, bg)
        for j in range(r["j0"], r["j1"] + 1):
            cv.rect(*cell_rect(x, y, c, r["row"], j), fill=col)


def outline_run(cv, x, y, c, r, color, w=2.0, pad=1.0):
    x0, y0, x1, y1 = run_rect(x, y, c, r)
    cv.rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad, outline=color, w=w)


def outline_alpha(t):
    if t < 13.2:
        return ramp(t, 8.2, 8.7)
    if t < 18.9:
        return 0.0
    return ramp(t, 18.9, 19.4)


def draw_thumb(cv, B, t, focus=None, amt=0.0, tint=None, row_hl=(), windows=0.0, win_pulse=0.0,
               colors=None):
    """The picture of the blob in kernel view, wherever the keyframes put it."""
    x, y, c, a = keyframes(t, THUMB_KEYS)
    if t < 3.0 or a < 0.01:
        return
    draw_blob_cells(cv, B, x, y, c, a, focus=focus, amt=amt, colors=colors)
    oa = outline_alpha(t) * a * (0.0 if colors else 1.0)      # the colours name the runs while merging
    if oa > 0.02:
        for r in B.runs:
            k = oa * (1 - 0.65 * amt if (focus is not None and r["id"] not in focus) else 1.0)
            outline_run(cv, x, y, c, r, mix(TEAL, k), 1.5 if c < 14 else 2.0)
    if tint:
        for rid, (col, w) in tint.items():
            outline_run(cv, x, y, c, B.runs[rid], mix(col, a), w, 1.5)
    for i, p in row_hl:
        if p > 0.01:
            cv.rect(x - 2, y + c * i - 1, x + c * B.n_cols + 1, y + c * (i + 1) + 1, outline=mix(INK, p * a), w=2)
    if windows > 0.01:
        edge = x + c * (32 - B.c0)
        cv.dline(edge, y - 8, edge, y + c * B.n_rows + 8, mix(GREY, windows * a), 1.5, 5, 4)
        wcol = mix(AMBER, windows * a)
        wpx = 1.5 + 2.5 * win_pulse
        for i in range(B.n_rows):
            cv.rect(x - 2, y + c * i, edge - 3, y + c * (i + 1) - 1, outline=wcol, w=wpx)
            cv.rect(edge + 3, y + c * i, x + c * B.n_cols + 2, y + c * (i + 1) - 1, outline=wcol, w=wpx)


# the merge spotlight (run 4 looks at row 5) and the colour steps of the merge
SPOT = (21.7, 23.0)
MERGE_RED_TO_OWN = (23.05, 23.45)      # red runs take a colour each: roots 34
MERGE_ROUND1 = 24.2                    # 34 roots -> 3
MERGE_ROUND2 = 25.2                    # 3 roots -> 1


def spot_alpha(t):
    return vis(t, SPOT[0], SPOT[1], 0.15, 0.2)


def make_palette(n):
    """One colour per run id; ids 0, 19 and 31 (the three roots after round 1)
    get clearly different colours, and run 0 (the final root) is teal."""
    pal = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((0.07 + i * 0.381966) % 1.0, 0.62, 0.86)
        pal.append((int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)))
    pal[0], pal[19], pal[31] = TEAL, (0xD9, 0x7A, 0x1E), (0x6A, 0x4C, 0xC8)
    return pal


def lerp_rgb(a, b, p):
    return tuple(lerp(a[i], b[i], p) for i in range(3))


def run_colors(B, t):
    """Colour of every run while the merge runs: red, then one colour per run
    (roots 34), then the colour of its root after round 1 (roots 3) and after
    round 2 (roots 1). None before the merge."""
    if t < MERGE_RED_TO_OWN[0]:
        return None
    pal = make_palette(len(B.runs))
    p0 = ease(ramp(t, *MERGE_RED_TO_OWN))
    p1 = ease(ramp(t, MERGE_ROUND1, MERGE_ROUND1 + 0.4))
    p2 = ease(ramp(t, MERGE_ROUND2, MERGE_ROUND2 + 0.4))
    out = {}
    for i in range(len(B.runs)):
        r1 = lerp_rgb(pal[i], pal[B.root_after[0][i]], p1)
        r2 = lerp_rgb(r1, pal[B.root_after[1][i]], p2)
        out[i] = lerp_rgb(RED, r2, p0)
    return out


def thumb_opts(B, t):
    """Per-beat decorations of the picture."""
    o = {}
    if 13.2 <= t < 16.4:
        o["windows"] = ramp(t, 13.95, 14.4) * (1 - ramp(t, 15.8, 16.3))
        o["win_pulse"] = math.sin(math.pi * ramp(t, 14.5, 15.3))
    # the example row stays marked from count to emit
    hl = vis(t, 16.7, 19.4, 0.3, 0.3)
    if hl > 0.01:
        o["row_hl"] = [(4, hl)]
    # while the table fills, the thumbnail row of each group lights up in step
    if 19.3 <= t < 21.4:
        group_rows = [g for g in B.group_rows if g != 4]
        sweep = []
        for oi, row in enumerate(group_rows):
            ts = 19.4 + 0.065 * oi
            pa = ramp(t, ts - 0.03, ts + 0.03) * (1 - ramp(t, ts + 0.06, ts + 0.12))
            if pa > 0.01:
                sweep.append((row, pa))
        if sweep:
            o["row_hl"] = list(o.get("row_hl", [])) + sweep
    if 18.2 <= t < 19.5:
        s1 = vis(t, 18.2, 18.65, 0.15, 0.1)
        o["tint"] = {i: (mix(TEAL, s1), 3.0) for i in range(3)} if s1 > 0.01 else None
    spot = spot_alpha(t)
    if spot > 0.01:
        o["focus"] = {4, 5, 6, 7, 8}
        o["amt"] = spot
    cols = run_colors(B, t)
    if cols:
        o["colors"] = cols
    return o


# --------------------------------------------------------- beat 1: reflection
def beat1(cv, B, t):
    if t >= 3.0 or t < 0.0:
        return
    c = 20
    a = ramp(t, 0.0, 0.5)
    p = ease(ramp(t, 0.7, 1.7))
    png = (400, 140)
    ker = (360, 190)
    # the picture slides and cross-fades into its mirror image over a dashed diagonal
    ox, oy = lerp(png[0], ker[0], p), lerp(png[1], ker[1], p)
    da = ramp(t, 0.5, 0.8) * (1 - ramp(t, 1.5, 1.8))
    if da > 0.01:
        n = max(B.n_rows, B.n_cols) + 2
        cv.dline(png[0] - 14, png[1] - 14, png[0] + c * n, png[1] + c * n, mix(GREY, da), 2.0, 8, 6)
    # a true dissolve: each view is drawn alone on white, then the two layers are
    # blended, so no cell of one view ever paints over the other (no pop, no stray strip)
    fade = ease(ramp(t, 0.68, 1.05))
    layers = []
    for view, va in (("img", 1.0 - fade), ("ker", fade)):
        if va < 0.005:
            layers.append(None)
            continue
        lay = Cv(W, H)
        for want_red in (False, True):
            col = mix(RED if want_red else CELL_BG, a)
            for i in range(B.n_rows):
                for j in range(B.n_cols):
                    if bool(B.K[i, j]) == want_red:
                        if view == "img":
                            x, yy = ox + c * i, oy + c * j       # image view: column i, row j
                        else:
                            x, yy = ox + c * j, oy + c * i       # kernel view: column j, row i
                        lay.rect(x, yy, x + c - 1, yy + c - 1, fill=col)
        layers.append(lay.img)
    if layers[0] is not None and layers[1] is not None:
        both = Image.blend(layers[0], layers[1], fade)
    else:
        both = layers[0] if layers[0] is not None else layers[1]
    if both is not None:
        cv.img = ImageChops.multiply(cv.img, both)       # the canvas is white here, except the dashed diagonal
        cv.d = ImageDraw.Draw(cv.img)
    draw_note(cv, 48, 90, "Flipped over its diagonal to match memory.", vis(t, 0.8, 3.0, 0.3, 0.0))
    aa = vis(t, 1.8, 3.0, 0.3, 0.0)
    if aa > 0.01:
        ink, teal = mix(INK, aa), mix(TEAL, aa)
        y_row4 = ker[1] + c * 5
        cv.line(ker[0], y_row4 + 2, ker[0] + c * B.n_cols, y_row4 + 2, ink, 3)
        cv.text(ker[0] - 18, ker[1] + c * 4.5, "side by side", 36, ink, "mono", "rm")
        ya, yb = ker[1] + c * 7.5, ker[1] + c * 8.5
        xe = ker[0] + c * B.n_cols
        for row in (7, 8):
            cv.rect(ker[0] - 2, ker[1] + c * row - 1, xe + 2, ker[1] + c * (row + 1), outline=teal, w=2)
        cv.dline(xe + 4, ya, 944, ya, mix(GREY, aa), 1.5, 3, 3)
        cv.dline(xe + 4, yb, 944, yb, mix(GREY, aa), 1.5, 3, 3)
        cv.arrow(950, ya, 950, yb, teal, 2.5, 8, both=True)
        cv.text(972, ya - 12, "%s B" % format(B.row_bytes, ","), 40, teal, "mono", "lm")
        cv.text(972, yb + 14, "apart", 40, teal, "mono", "lm")
        for k, line in enumerate(("Real image:", "%s px x 3 B" % format(B.height, ","), "in every row.")):
            draw_note(cv, 972, 452 + 34 * k, line, aa, size=28)


# ---------------------------------------------------- beats 2-4: peel and zoom
def lane_scale(t):
    s_final = (LANE_X1 - LANE_X0) / (24 * 28 + GAP_U * 23)
    return 20.0 * (s_final / 20.0) ** ease(ramp(t, *ZOOM))


def lane_height(t):
    return lerp(44.0, 80.0, ease(ramp(t, *ZOOM)))


def lane_geometry(B, t):
    s = lane_scale(t)
    prog = [ease(ramp(t, *FLIGHTS[r])) for r in range(B.n_rows)]
    return s, lane_height(t), sum(prog) * ROW_U - GAP_U, prog


STREAM_FROM = 5          # rows 0-4 fly down one by one, rows 5-23 stream in from the right


def lane_cell_rects(B, t):
    """{(row, col): (x0, y0, x1, y1, alpha)} of every lane cell that has landed
    or is on its way. Rows 0-4 fly from the picture to the lane tail. Later
    rows stream in at the tail (the zoom is running, so they would only smear
    if they flew). Rows that have not started are left out."""
    s, h, u_end, prog = lane_geometry(B, t)
    gap = 1.0 if s >= 6 else 0.0
    out = {}
    for r in range(B.n_rows):
        p = prog[r]
        if p <= 0:
            continue
        for j in range(B.n_cols):
            lx = LANE_X1 - (u_end - (r * ROW_U + j)) * s
            lx1 = lx + s - gap
            if p >= 1:
                out[(r, j)] = (lx, LANE_TOP, lx1, LANE_TOP + h, 1.0)
            elif r >= STREAM_FROM:
                out[(r, j)] = (lx, LANE_TOP, lx1, LANE_TOP + h, p)
            else:
                tx0, ty0, tx1, ty1 = cell_rect(48, 150, 16, r, j)
                top = lerp(ty0, LANE_TOP, p)
                out[(r, j)] = (lerp(tx0, lx, p), top, lerp(tx1, lx1, p),
                               top + lerp(ty1 - ty0, h, ramp(p, 0.7, 1.0)), 1.0)
    return out


def draw_fold(cv, x, s, top, h, a):
    if a <= 0.01 or x < LANE_X0 - 2 or x > LANE_X1:
        return
    col = mix(GREY, a)
    if s >= 8:
        mid = top + h / 2
        for dx in (0.28, 0.58):
            cv.line(x + dx * s * GAP_U - 3, mid + 8, x + dx * s * GAP_U + 3, mid - 8, col, 2)
    else:
        cv.line(x + s * GAP_U / 2, top, x + s * GAP_U / 2, top + h, col, 1)


def draw_lane(cv, B, t, alpha=1.0):
    """The memory lane of the image (beats 2 - 4). Returns the cell rects."""
    if t < 3.8 or alpha <= 0.01:
        return None
    s, h, u_end, prog = lane_geometry(B, t)
    ea = vis(t, 3.8, 99, 0.4, 0) * alpha
    cv.rrect(LANE_X0, LANE_TOP, LANE_X1, LANE_TOP + h, 6, fill=mix(WHITE, ea, PANEL))
    if t < 4.6:
        cv.drect(LANE_X0, LANE_TOP, LANE_X1, LANE_TOP + h, mix(HAIR, ea, PANEL), 2.0, 8, 5)
    for r in range(B.n_rows - 1):
        pa = prog[r + 1]
        if pa > 0.5:
            draw_fold(cv, LANE_X1 - (u_end - (r * ROW_U + 28)) * s, s, LANE_TOP, h, alpha * ramp(pa, 0.5, 1.0))
    rects = lane_cell_rects(B, t)
    flying = []
    for (r, j), (x0, y0, x1, y1, ra) in rects.items():
        if prog[r] < 1 and r < STREAM_FROM:
            flying.append((r, j, x0, y0, x1, y1))
            continue
        if x1 <= LANE_X0 or x0 >= LANE_X1:
            continue
        col = RED if B.K[r, j] else CELL_BG
        cv.rect(max(x0, LANE_X0), y0, min(x1 + (0.0 if s >= 6 else 0.4), LANE_X1), y1, fill=mix(col, alpha * ra))
    for r, j, x0, y0, x1, y1 in flying:
        # only the red cells travel in full colour, so the copy does not wipe the picture
        k = 1.0 if B.K[r, j] else ramp(prog[r], 0.7, 1.0)
        if k > 0.01:
            cv.rect(x0, y0, x1, y1, fill=mix(RED if B.K[r, j] else CELL_BG, alpha * k))
    return rects


def draw_row_copy_outline(cv, B, t):
    """Dark outline on the picture row that is being copied."""
    for r in range(B.n_rows):
        a, b = FLIGHTS[r]
        if r >= STREAM_FROM:
            lo, hi, fi, fo = a, a + 0.07, 0.02, 0.05
        else:
            lo, hi, fi, fo = a - 0.02, b + 0.1, 0.03, 0.1
        if lo <= t <= hi:
            cv.rect(46, 150 + 16 * r - 1, 48 + 16 * B.n_cols + 1, 150 + 16 * (r + 1), outline=mix(INK, vis(t, lo, hi, fi, fo)), w=2)


def beat3(cv, B, t):
    """Empty lane, the peel and the zoom (3.8 - 9.4)."""
    if t < 3.8 or t > 9.6:
        return
    la = 1.0 - ramp(t, 9.0, 9.35)
    rects = draw_lane(cv, B, t, la)
    if 4.4 <= t < 8.0:
        draw_row_copy_outline(cv, B, t)
    # brackets under run 3 and run 4 while row 4 sits at the tail
    ba = vis(t, 5.7, 6.5, 0.25, 0.3)
    if ba > 0.01 and rects:
        bot = LANE_TOP + lane_height(t) + 5
        for rid, label in ((3, "run 3"), (4, "run 4")):
            r = B.runs[rid]
            xa, xb = rects[(4, r["j0"])][0], rects[(4, r["j1"])][2]
            col = mix(TEAL, ba, PANEL)
            cv.line(xa, bot, xb, bot, col, 3)
            cv.line(xa, bot - 6, xa, bot, col, 3)
            cv.line(xb, bot - 6, xb, bot, col, 3)
            cv.text((xa + xb) / 2, bot + 8, label, 32, col, "mono", "ma")
    # ticks over the 34 runs, left to right
    if t >= 8.0 and rects:
        bot = LANE_TOP + lane_height(t) + 4
        for r in B.runs:
            ta = 8.0 + 0.5 * r["id"] / 33
            a = ramp(t, ta, ta + 0.12) * la
            if a <= 0.01:
                continue
            xa, xb = rects[(r["row"], r["j0"])][0], rects[(r["row"], r["j1"])][2]
            cx = (xa + xb) / 2
            cv.rect(min(xa, cx - 1.5), bot, max(xb, cx + 1.5), bot + 11, fill=mix(TEAL, a, PANEL))
    fn = vis(t, 4.8, 9.2, 0.4, 0.4)
    draw_note(cv, 540, 432, "Fold marks = rest of the row.", fn)
    draw_note(cv, 540, 466, "%s more px in every row." % format(B.height - B.n_cols, ","), fn, size=28)
    ra = vis(t, 8.2, 9.35, 0.3, 0.35)
    if ra > 0.01:
        cv.text(540, 236, "34 runs", 96, mix(TEAL, ra), "bold")


# ------------------------------------------------------- beats 4-5: pack
STRIP_X0, STRIP_PITCH, STRIP_Y0, STRIP_Y1 = 96, 17, 250, 290
BITS_Y0, BITS_Y1 = 392, 428
BOX_Y0, BOX_Y1 = 520, 562
BOXES = [(96, 286), (360, 550)]


def strip_x(k):
    return STRIP_X0 + STRIP_PITCH * k


def beat4(cv, B, t):
    """Row 4 as 64 pixels, the word edge, the warp window (9.0 - 13.4)."""
    if t < 9.0 or t > 13.4:
        return
    out = 1.0 - ramp(t, 13.0, 13.4)
    rects = lane_cell_rects(B, 9.0)
    p4 = ease(ramp(t, 9.0, 10.0))
    row = B.mask[4]
    ta = ramp(t, 9.3, 10.0)
    tray(cv, 90, STRIP_Y0 - 6, 1190, STRIP_Y1 + 6, ta * out)
    cv.text(96, STRIP_Y0 - 14, "row 4", 28, mix(GREY, ta * out, PANEL), "mono", "ls")
    for k in range(64):
        col = RED if row[k] else CELL_BG
        fx0, fx1 = strip_x(k), strip_x(k) + STRIP_PITCH - 1
        if B.c0 <= k < B.c0 + B.n_cols:
            bx0, by0, bx1, by1, _ = rects[(4, k - B.c0)]
            x0, y0 = lerp(bx0, fx0, p4), lerp(by0, STRIP_Y0, p4)
            x1, y1 = lerp(bx1, fx1, p4), lerp(by1, STRIP_Y1, p4)
            a = out
        else:
            edge = strip_x(B.c0) if k < B.c0 else strip_x(B.c0 + B.n_cols)
            x0, x1 = lerp(edge, fx0, p4), lerp(edge, fx1, p4)
            y0, y1 = STRIP_Y0, STRIP_Y1
            a = ta * out
        if a > 0.01 and x1 - x0 > 0.2:
            cv.rect(x0, y0, x1, y1, fill=mix(col, a, WHITE if p4 > 0.9 else PANEL))
    ea = vis(t, 9.9, 13.4, 0.3, 0.4) * out
    if ea > 0.01:
        ex = strip_x(32) - 0.5
        cv.dline(ex, STRIP_Y0 - 18, ex, STRIP_Y1 + 58, mix(GREY, ea, PANEL), 2, 6, 4)
        cv.text(strip_x(16) - 0.5, STRIP_Y1 + 22, "word 89", 28, mix(GREY, ea, PANEL), "reg", "ma")
        cv.text(strip_x(48) - 0.5, STRIP_Y1 + 22, "word 90", 28, mix(GREY, ea, PANEL), "reg", "ma")
    # the amber window drops on word 89 (pulses once), later on word 90
    for w, (t0, pulse_t) in enumerate(((10.1, 10.7), (11.9, None))):
        drop = ease(ramp(t, t0, t0 + 0.5))
        if drop <= 0.01:
            continue
        wpx = 3 + (3 * math.sin(math.pi * ramp(t, pulse_t, pulse_t + 0.5)) if pulse_t else 0)
        yo = (1 - drop) * -26
        cv.rrect(strip_x(32 * w) - 3, STRIP_Y0 - 8 + yo, strip_x(32 * w + 32) - 2, STRIP_Y1 + 8 + yo, 8,
                 outline=mix(AMBER, drop * out, PANEL), w=wpx)
    beat5(cv, B, t, out)


def beat5(cv, B, t, out):
    """One bit per pixel, 32 bits squeezed into one 4 B word (11.0 - 13.2)."""
    if t < 11.0:
        return
    row = B.mask[4]
    for w, (t_in, t_sq, sq_d) in enumerate(((11.0, 11.7, 0.5), (12.1, 12.5, 0.4))):
        a_in = ramp(t, t_in, t_in + 0.3) * out
        if a_in <= 0.01:
            continue
        sq = ease(ramp(t, t_sq, t_sq + sq_d))
        bx0, bx1 = BOXES[w]
        bw = (bx1 - bx0) / 32.0
        for b in range(32):
            k = 32 * w + b
            x0, x1 = lerp(strip_x(k), bx0 + b * bw, sq), lerp(strip_x(k) + STRIP_PITCH - 1, bx0 + (b + 1) * bw, sq)
            y0, y1 = lerp(BITS_Y0, BOX_Y0, sq), lerp(BITS_Y1, BOX_Y1, sq)
            st = ramp(t, t_in + 0.006 * b, t_in + 0.006 * b + 0.25) * out
            if st <= 0.01:
                continue
            if row[k]:
                cv.rect(x0, y0, x1, y1, fill=mix(AMBER, st, PANEL))
            else:
                cv.rect(x0, y0, x1, y1, fill=mix(WHITE, st, PANEL),
                        outline=mix(HAIR, st, PANEL) if sq < 0.5 else None, w=1)
        if sq > 0.9:
            cv.rrect(bx0 - 3, BOX_Y0 - 3, bx1 + 3, BOX_Y1 + 3, 6,
                     outline=mix(AMBER, ramp(sq, 0.9, 1.0) * out, PANEL), w=3)
            ba = ramp(t, t_sq + sq_d, t_sq + sq_d + 0.2) * out
            cv.text((bx0 + bx1) / 2, BOX_Y1 + 14, "4 B", 40, mix(AMBER, ba, PANEL), "mono", "ma")
        if w == 0:
            la = a_in * (1 - ease(ramp(t, t_sq, t_sq + 0.2)))
            cv.text(strip_x(0), BITS_Y1 + 8, "bit 0", 28, mix(GREY, la, PANEL), "reg", "la")
            cv.text(strip_x(32) - 2, BITS_Y1 + 8, "bit 31", 28, mix(GREY, la, PANEL), "reg", "ra")


# --------------------------------------------------------- beat 6: 24 times
BAR_X = 540


def beat6(cv, B, t):
    """The 48 windows, the 24 pairs of words, 4,608 B against 192 B (13.2 - 16.3)."""
    if t < 13.2 or t > 16.4:
        return
    out = 1.0 - ramp(t, 15.8, 16.3)
    la = ramp(t, 13.9, 14.5) * out
    if la > 0.01:
        pair_gap = 14.0
        pair_w = (LANE_X1 - LANE_X0 - 23 * pair_gap) / 24
        box_w = pair_w / 2 - 0.5
        tray(cv, LANE_X0, LANE_TOP, LANE_X1, LANE_TOP + 44, la)
        pulse = vis(t, 14.5, 15.3, 0.25, 0.4)
        for r in range(B.n_rows):
            px = LANE_X0 + r * (pair_w + pair_gap)
            for w in range(2):
                bx = px + w * (box_w + 1)
                cv.rect(bx, LANE_TOP + 6, bx + box_w, LANE_TOP + 38, fill=mix(WHITE, la, PANEL),
                        outline=mix(AMBER, la * (0.45 + 0.55 * pulse), PANEL), w=1.5 + 1.5 * pulse)
                for b in range(32):
                    if B.mask[r, 32 * w + b]:
                        xx = bx + 1 + (box_w - 2) * b / 32.0
                        cv.rect(xx, LANE_TOP + 9, xx + max(0.5, (box_w - 2) / 32.0), LANE_TOP + 35,
                                fill=mix(AMBER, la, PANEL))
            if r < B.n_rows - 1:
                draw_fold_simple(cv, px + pair_w + pair_gap / 2, LANE_TOP + 22, la)
    ba = ramp(t, 14.4, 14.9) * out
    if ba > 0.01:
        full = 640.0
        for name, nbytes, y_lab, color, t_a in (("RGB in", B.b_rgb, 190, BAR_GREY, 14.5),
                                                ("bits", B.b_mask, 330, AMBER, 14.9)):
            grow = ease(ramp(t, t_a, t_a + 0.6))
            if grow <= 0.01:
                continue
            cv.text(BAR_X, y_lab + 14, name, 28, mix(GREY, ba), "mono", "lm")
            cv.text(BAR_X + 130, y_lab + 14, "%s B" % format(nbytes, ","), 40, mix(INK, ba), "mono", "lm")
            cv.rect(BAR_X, y_lab + 44, BAR_X + max(2, full * nbytes / B.b_rgb * grow), y_lab + 82, fill=mix(color, ba))
        xa = ramp(t, 15.2, 15.6) * out
        if xa > 0.01:
            xe = BAR_X + full * B.b_mask / B.b_rgb + 22
            cv.text(xe, 330 + 69, "%dx" % round(B.b_rgb / B.b_mask), 72, mix(AMBER, xa), "bold", "lm")
            cv.text(xe + 160, 330 + 69, "this blob", 30, mix(GREY, xa), "bold", "lm")
        na = ramp(t, 15.0, 15.5) * out
        draw_note(cv, BAR_X, 468, "This blob: 24 rows x 64 px.", na, size=28)
        draw_note(cv, BAR_X, 504, "48 words, 48 warps.", na, size=28)


# ------------------------------------------------- beats 7-12: the run table
STR_X0, STR_PITCH, STR_GAP = 60, 26, 14
STR_LABEL_Y = [478, 542, 606]
STR_Y = [508, 570, 634]
STR_H = 30
STR_NAMES = ["run_y0", "run_y1", "parent"]


def slot_x(B, k):
    return STR_X0 + STR_PITCH * k + STR_GAP * B.group_of[k]


def draw_gap_cells(cv, B, g, y, a):
    """Two narrow grey cells in the gap before group g: other blobs' runs."""
    first = next(k for k in range(len(B.runs)) if B.group_of[k] == g)
    gx = slot_x(B, first) - STR_GAP
    col = mix(FOREIGN, a, PANEL)
    for ix in range(2):
        cv.rect(gx + 1 + ix * 6.5, y, gx + 1 + ix * 6.5 + 5, y + STR_H, fill=col)


def tint_color(B, k, a):
    return mix(TEAL, a * (0.25 if B.group_of[k] % 2 == 0 else 0.45), PANEL)


def parent_change_time(B, k):
    """When the parent cell of run k first differs from k (None if never)."""
    if B.rounds[0]["parent"][k] != k:
        return MERGE_ROUND1
    if B.rounds[1]["parent"][k] != k:
        return MERGE_ROUND2
    return None


FLATTEN_PARENT = 27.0          # flatten writes parent[r] = root: cells 10 and 33 become 0


def draw_table(cv, B, t):
    """The run table in global memory as three of its five arrays (16.2 - 28.1)."""
    if t < 17.95 or t > 28.1:
        return
    base = vis(t, 18.0, 28.1, 0.35, 0.4)
    dim = ramp(t, 21.8, 22.2)                    # run_y0 / run_y1 dim: merge only reads them
    fade_ys = ramp(t, 26.5, 26.9)                # flatten swaps them for run_label
    spot = spot_alpha(t)                         # spotlight on row 5's slots
    labelfill = ramp(t, 26.9, 27.35)
    others = [g for g in B.group_rows if g != 4]

    def fill_a(k):
        row = B.runs[k]["row"]
        if row == 4:
            return ramp(t, 18.9, 19.2)
        o = others.index(row)
        return ramp(t, 19.4 + 0.065 * o, 19.7 + 0.065 * o)

    na = base * (1 - ramp(t, 26.4, 26.8))
    if na > 0.01:
        cv.text(1232, 449, "5 arrays, 3 drawn here", 24, mix(GREY, na, PANEL), "reg", "ra")
    for si, name in enumerate(STR_NAMES):
        sa = base * ((1 - fade_ys) if name == "run_y1" else 1.0)
        if sa <= 0.01:
            continue
        la = sa * ((1 - 0.6 * dim) if name != "parent" else 1.0)
        cv.text(STR_X0, STR_LABEL_Y[si], name, 26, mix(TEAL, la, PANEL), "mono")
        if t >= 21.8:
            tag = "read" if name != "parent" else "read + written"
            cv.text(STR_X0 + 130, STR_LABEL_Y[si] + 2, tag, 26,
                    mix(GREY, sa * ramp(t, 21.9, 22.3) * (1 - ramp(t, 26.1, 26.4)), PANEL), "reg")
        for k in range(len(B.runs)):
            x0 = slot_x(B, k)
            y0 = STR_Y[si]
            fa = fill_a(k)
            if fa < 1:
                cv.rect(x0, y0, x0 + STR_PITCH - 1, y0 + STR_H, fill=mix(WHITE, sa * (1 - fa), PANEL),
                        outline=mix(HAIR, sa * (1 - fa), PANEL), w=1)
            if fa <= 0.01:
                continue
            a = sa * fa
            if name != "parent":
                a *= 1 - 0.6 * dim
                if spot > 0 and B.runs[k]["row"] != 5:
                    a *= 1 - 0.5 * spot
            cv.rect(x0, y0, x0 + STR_PITCH - 1, y0 + STR_H, fill=tint_color(B, k, a))
            if name == "parent":
                v = k if t < MERGE_ROUND1 else (B.rounds[0]["parent"][k] if t < MERGE_ROUND2
                                                else B.rounds[1]["parent"][k])
                if k in (10, 33) and t >= FLATTEN_PARENT:
                    v = B.root_after[1][k]                # flatten: parent[r] = root
                ct = parent_change_time(B, k)
                tcol = INK
                if ct is not None and t >= ct:
                    ca = ramp(t, ct, ct + 0.4)
                    cv.rect(x0, y0, x0 + STR_PITCH - 1, y0 + STR_H, fill=mix(TEAL, a * ca * 0.75, PANEL))
                    tcol = WHITE if ca > 0.5 else INK
                txt = str(v)
            else:
                txt = str(B.runs[k]["y0"] if name == "run_y0" else B.runs[k]["y1"])
                tcol = INK
            cv.text(x0 + STR_PITCH / 2 - 0.5, y0 + STR_H / 2 + 1, txt, 17, mix(tcol, a, PANEL), "mono", "mm")
    # gaps between groups stand for other blobs' runs
    ga = ramp(t, 19.5, 20.5) * base
    if ga > 0.01:
        for g in range(1, len(B.group_rows)):
            for si in range(3):
                a = ga * ((1 - fade_ys) if si == 1 else 1.0)
                if a > 0.01:
                    draw_gap_cells(cv, B, g, STR_Y[si], a)
    # spotlight brackets on row 5's slots in run_y0 and run_y1
    if spot > 0.01:
        s5 = [r["id"] for r in B.runs if r["row"] == 5]
        xa, xb = slot_x(B, s5[0]) - 4, slot_x(B, s5[-1]) + STR_PITCH + 3
        for si in range(2):
            cv.rrect(xa, STR_Y[si] - 4, xb, STR_Y[si] + STR_H + 4, 6, outline=mix(TEAL, spot, PANEL), w=3)
    # flatten: run_label, cyan, in run_y1's place (flatten reads run_x and run_y0, never run_y1)
    if labelfill > 0.01:
        a = labelfill * base
        cv.text(STR_X0, STR_LABEL_Y[1], "run_label", 26, mix(TEAL, a, PANEL), "mono")
        for k in range(len(B.runs)):
            x0 = slot_x(B, k)
            cv.rect(x0, STR_Y[1], x0 + STR_PITCH - 1, STR_Y[1] + STR_H, fill=mix(CYAN, a, PANEL),
                    outline=mix(CYAN_EDGE, a, PANEL), w=2)
    # the two cells that are not flat yet, until flatten writes the root into them
    nf = ramp(t, 26.0, 26.3) * base * (1 - ramp(t, FLATTEN_PARENT - 0.1, FLATTEN_PARENT + 0.1))
    if nf > 0.01:
        for k in (10, 33):
            x0 = slot_x(B, k)
            cv.rect(x0 - 2, STR_Y[2] - 2, x0 + STR_PITCH + 1, STR_Y[2] + STR_H + 2, outline=mix(AMBER, nf, PANEL), w=3)
        cv.text(slot_x(B, 10) + STR_PITCH / 2, STR_Y[2] + STR_H + 6, "not flat yet", 26, mix(GREY, nf, PANEL), "reg", "ma")
        cv.text(slot_x(B, 33) + STR_PITCH, STR_Y[2] + STR_H + 6, "not flat yet", 26, mix(GREY, nf, PANEL), "reg", "ra")
    # flatten: a short teal flash on the two cells that change
    ff = vis(t, FLATTEN_PARENT, FLATTEN_PARENT + 0.5, 0.1, 0.4) * base
    if ff > 0.01:
        for k in (10, 33):
            x0 = slot_x(B, k)
            cv.rect(x0 - 2, STR_Y[2] - 2, x0 + STR_PITCH + 1, STR_Y[2] + STR_H + 2, outline=mix(TEAL, ff, PANEL), w=3)


def draw_count_chips(cv, B, t):
    a = vis(t, 16.9, 19.9, 0.4, 0.4)
    if a <= 0.01:
        return
    for i in range(B.n_rows):
        for k in range(B.row_count[i]):
            x0, y0 = 370 + 16 * k, 150 + 11 * i + 0.5
            cv.rect(x0, y0, x0 + 10, y0 + 10, fill=mix(TEAL, a))


CW_X0, CW_PITCH, CW_Y0, CW_Y1 = 64, 36, 508, 548


def draw_count_words(cv, B, t):
    """Count: one warp, 32 lanes, 32 words of one mask row (16.5 - 18.0). Words
    64-95 of the row; words 89 and 90 hold the blob. The other words are drawn
    without bits: they are not part of this blob and their content is not shown."""
    a = vis(t, 16.5, 18.0, 0.3, 0.25)
    if a <= 0.01:
        return
    for k in range(32):
        w = B.group0 + k
        x0 = CW_X0 + CW_PITCH * k
        ours = w in (B.word0, B.word0 + 1)
        cv.rect(x0, CW_Y0, x0 + CW_PITCH - 3, CW_Y1, fill=mix(WHITE, a, PANEL),
                outline=mix(AMBER if ours else HAIR, a, PANEL), w=3 if ours else 1.5)
        if ours:
            for b in range(32):
                if B.mask[4, 32 * (w - B.word0) + b]:
                    xx = x0 + 2 + (CW_PITCH - 7) * b / 32.0
                    cv.rect(xx, CW_Y0 + 4, xx + max(1.0, (CW_PITCH - 7) / 32.0), CW_Y1 - 4, fill=mix(AMBER, a, PANEL))
            cv.text(x0 + (CW_PITCH - 3) / 2, CW_Y0 - 7, str(w), 24, mix(AMBER, a, PANEL), "mono", "ms")
    cv.text(CW_X0, CW_Y0 - 7, "row 4, words %d-%d" % (B.group0, B.group0 + 31), 24, mix(GREY, a, PANEL), "mono", "ls")
    cv.text(CW_X0 + CW_PITCH * 16 - 2, CW_Y1 + 8, "other words not shown", 22, mix(GREY, a, PANEL), "reg", "ma")
    cv.text(CW_X0, CW_Y1 + 8, "lane 0", 22, mix(GREY, a, PANEL), "mono", "la")
    cv.text(CW_X0 + CW_PITCH * 32 - 3, CW_Y1 + 8, "lane 31", 22, mix(GREY, a, PANEL), "mono", "ra")
    # the teal bracket closes over all 32 words, then names the group
    bp = ease(ramp(t, 16.75, 17.15))
    if bp > 0.01:
        by = CW_Y1 + 44
        xe = CW_X0 + (CW_PITCH * 32 - 3) * bp
        col = mix(TEAL, a, PANEL)
        cv.line(CW_X0, by, xe, by, col, 4)
        cv.line(CW_X0, by - 12, CW_X0, by, col, 4)
        if bp > 0.99:
            cv.line(xe, by - 12, xe, by, col, 4)
        la = ramp(t, 17.05, 17.3) * a
        cv.text((CW_X0 + CW_X0 + CW_PITCH * 32 - 3) / 2, by + 12, "1 warp = 32 words = 128 B", 32,
                mix(TEAL, la, PANEL), "mono", "ma")
        draw_note(cv, CW_X0, by + 62, "A row has %d words, so this warp takes %d steps of 32." % (B.words_per_row, B.count_steps),
                  ramp(t, 17.2, 17.45) * a, size=26, bg=PANEL)


def beat7_8(cv, B, t):
    """Count, scan and emit: the words beside the picture (15.8 - 21.8)."""
    if t < 15.8 or t > 21.8:
        return
    na = vis(t, 16.7, 18.2, 0.3, 0.2)
    draw_note(cv, 470, 150, "count and emit: 1 warp per row.", na)
    draw_note(cv, 470, 192, "scan: one block adds a running total,", na)
    draw_note(cv, 470, 226, "so each row gets its first slot.", na)
    # the chips and the slot numbers are this blob's, not the real table's
    ca = vis(t, 16.8, 19.7, 0.3, 0.3)
    draw_note(cv, 470, 290, "Chips: this blob's runs only.", ca, size=28)
    draw_note(cv, 470, 324, "About %d runs in a real row." % B.runs_per_row, ca, size=28)
    s1 = vis(t, 18.2, 18.65, 0.15, 0.1)
    s2 = vis(t, 18.65, 19.4, 0.15, 0.3)
    if s1 > 0.01:
        cv.text(470, 150, "3 runs above", 40, mix(TEAL, s1), "mono")
    if s2 > 0.01:
        cv.text(470, 150, "row 4 starts at local slot 3", 40, mix(TEAL, s2), "mono")
        draw_note(cv, 470, 204, "run 4 spans 2 words.", s2 * ramp(t, 18.9, 19.2))
    la = vis(t, 18.2, 19.7, 0.3, 0.3)       # the two caveats are on screen for about a second
    draw_note(cv, 470, 372, "Local ids 0-33 (real ids: %s to %s)." % (format(B.runs[0]["gid"], ","),
                                                                      format(B.runs[-1]["gid"], ",")), la, size=28)
    draw_note(cv, 470, 402, "y counts from the window edge (real: %s)." % format(B.y_origin, ","), la, size=28)
    ba = ramp(t, 19.9, 20.4) * (1 - ramp(t, 21.5, 21.8))
    if ba > 0.01:
        cv.text(470, 136, "34 runs", 96, mix(TEAL, ba), "bold")
        cv.text(470, 262, "34 x 12 B = %d B" % B.b_runs, 40, mix(GREY, ramp(t, 20.3, 20.8) * ba), "mono")
        n = ramp(t, 20.6, 21.1) * ba
        draw_note(cv, 470, 306, "12 B per run: x, y0, y1.", n, size=28)
        draw_note(cv, 470, 336, "Gaps = other blobs' runs.", n, size=28)


# ------------------------------------------------------- beats 9-12: merge
def thumb_xyc(t):
    x, y, c, a = keyframes(t, THUMB_KEYS)
    return x, y, c


def connector_xy(B, t, a_id, p_id):
    """Where run a (upper) meets run p (next row) on the picture: the middle of
    the shared columns, or the corner when they only touch diagonally."""
    x, y, c = thumb_xyc(t)
    ra, rp = B.runs[a_id], B.runs[p_id]
    lo, hi = max(ra["j0"], rp["j0"]), min(ra["j1"], rp["j1"])
    if lo <= hi:
        col = (lo + hi + 1) / 2.0
    elif rp["j1"] < ra["j0"]:
        col = float(ra["j0"])
    else:
        col = float(ra["j1"] + 1)
    return x + c * col, y + c * (ra["row"] + 1)


def draw_connector(cv, B, t, a_id, p_id, color, a):
    cx, cy = connector_xy(B, t, a_id, p_id)
    cv.dot(cx, cy, 5.0, mix(WHITE, a))
    cv.dot(cx, cy, 3.4, mix(color, a))


def counter_value(B, t):
    """The roots counter steps 34, 3, 1: no rolling numbers, only the states of the shown schedule."""
    if t < MERGE_ROUND1:
        return len(B.runs), None, 0.0
    if t < MERGE_ROUND2:
        return len(B.runs), B.roots[0], ramp(t, MERGE_ROUND1, MERGE_ROUND1 + 0.3)
    return B.roots[0], B.roots[1], ramp(t, MERGE_ROUND2, MERGE_ROUND2 + 0.3)


INSET_X, INSET_C, INSET_Y = 480, 26, {4: 172, 5: 232}


def draw_inset(cv, B, t, a):
    """Rows 4 and 5 enlarged: run 4 looks one cell wider on each side (corner
    touches count), reaches runs 6 and 7 of the next row, and run 8 is too far.
    The three marks come one after another."""
    x0, c = INSET_X, INSET_C
    for row, y in INSET_Y.items():
        cv.text(x0, y - 6, "row %d" % row, 22, mix(GREY, a), "mono", "ls")
        for j in range(B.n_cols):
            if not B.K[row, j]:
                cv.rect(x0 + c * j, y, x0 + c * j + c - 1, y + c - 1, fill=mix(CELL_BG, a))
        for r in B.runs:
            if r["row"] != row:
                continue
            xa, xb = x0 + c * r["j0"], x0 + c * (r["j1"] + 1) - 1
            cv.rect(xa, y, xb, y + c - 1, fill=mix(RED, a))
            if r["n"] >= 2 or r["id"] in (4, 6, 7, 8):
                cv.text((xa + xb) / 2, y + c / 2, str(r["id"]), 18, mix(WHITE, a, RED), "mono", "mm")
    r4 = B.runs[4]
    # the widened span: one cell more on each side, drawn over row 5
    wa = a * ramp(t, 21.85, 22.0)
    if wa > 0.01:
        wx0, wx1 = x0 + c * (r4["j0"] - 1), x0 + c * (r4["j1"] + 2) - 1
        cv.drect(wx0, INSET_Y[5] - 5, wx1, INSET_Y[5] + c + 4, mix(TEAL, wa), 2.5, 6, 4)
        cv.text((wx0 + wx1) / 2, INSET_Y[5] + c + 8, "one cell wider each side", 22, mix(TEAL, wa), "reg", "ma")
    ax, ay = x0 + c * (r4["j0"] + r4["j1"] + 1) / 2, INSET_Y[4] + c
    for pid, ta in ((6, 22.0), (7, 22.14)):
        g = ramp(t, ta, ta + 0.13)
        if g > 0.01:
            rp = B.runs[pid]
            bx, by = x0 + c * (rp["j0"] + rp["j1"] + 1) / 2, INSET_Y[5]
            cv.arrow(ax, ay, ax + (bx - ax) * g, ay + (by - 1 - ay) * g, mix(TEAL, a * g), 3.0, 9)
    xa_ = a * ramp(t, 22.3, 22.45)
    if xa_ > 0.01:
        r8 = B.runs[8]
        bx, by = x0 + c * (r8["j0"] + r8["j1"] + 1) / 2, INSET_Y[5] - 9
        cv.dline(ax, ay, bx, by, mix(GREY, xa_), 2.0, 5, 4)
        cv.line(bx - 6, by - 6, bx + 6, by + 6, mix(GREY, xa_), 3)
        cv.line(bx - 6, by + 6, bx + 6, by - 6, mix(GREY, xa_), 3)


def beat9_12(cv, B, t):
    """Merge (21.6 - 26.4) and flatten (26.4 - 27.6): the picture and its counters."""
    if t < 21.8 or t > 28.0:
        return
    # 9: one example, run 4 looks down at row 5's slots (shown enlarged beside the picture)
    sp = spot_alpha(t)
    if sp > 0.01:
        draw_inset(cv, B, t, sp)
        cv.text(480, 296, "row 5 = slots 5-8", 40, mix(TEAL, sp), "mono")
        draw_note(cv, 480, 350, "touching corners counts too.", sp * ramp(t, 22.0, 22.2))
        draw_note(cv, 480, 386, "reads the table, no pixels.", sp * ramp(t, 22.0, 22.2))
    # 10-11: everyone at once. Each run takes the colour of its root; a dot marks each probe
    ca = vis(t, 23.2, 26.6, 0.3, 0.3)
    if ca > 0.01:
        da = ramp(t, 23.2, 23.5) * (1 - ramp(t, MERGE_ROUND1 + 0.1, MERGE_ROUND1 + 0.5))
        for (a_id, p_id, linked) in B.rounds[0]["attempts"]:
            draw_connector(cv, B, t, a_id, p_id, TEAL, da)
        d2 = ramp(t, MERGE_ROUND2 - 0.45, MERGE_ROUND2 - 0.2) * (1 - ramp(t, MERGE_ROUND2 + 0.1, MERGE_ROUND2 + 0.6))
        for (a_id, p_id, linked) in B.rounds[1]["attempts"]:
            draw_connector(cv, B, t, a_id, p_id, TEAL if linked else GREY, d2)
        # the five repeats stay marked: already one blob, no new edge
        rp_a = ramp(t, MERGE_ROUND2 + 0.1, MERGE_ROUND2 + 0.5) * (1 - ramp(t, 26.3, 26.6))
        for (a_id, p_id, linked) in B.rounds[1]["attempts"]:
            if not linked:
                draw_connector(cv, B, t, a_id, p_id, GREY, rp_a)
        # the counter
        old, new, p = counter_value(B, t)
        cv.text(470, 136, "roots ", 96, mix(TEAL, ca), "bold")
        nx = 470 + text_w("roots ", 96)
        cv.text(nx, 136, str(old), 96, mix(TEAL, ca * (1 - p)), "bold")
        if new is not None:
            cv.text(nx, 136, str(new), 96, mix(TEAL, ca * p), "bold")
        na_ = ca * vis(t, 23.3, 26.5, 0.3, 0.3)
        draw_note(cv, 470, 250, "roots = separate groups", na_, size=28)
        draw_note(cv, 470, 284, "Same color = same root. One possible order.", na_, size=28)
        pa = ramp(t, MERGE_ROUND2, MERGE_ROUND2 + 0.4) * ca
        if pa > 0.01:
            cv.text(470, 338, "%d probes = %d links + %d repeats" % (
                B.n_links + 5, B.n_links, 5), 34, mix(GREY, pa), "mono")
            cv.dot(486, 410, 5.0, mix(GREY, pa))
            draw_note(cv, 504, 392, "= already the same blob", pa, size=28)
    # 12: flatten
    fa = vis(t, 26.6, 28.0, 0.3, 0.4)
    if fa > 0.01:
        cv.text(470, 190, "label %s" % format(B.label, ","), 40, mix(INK, fa), "mono")
        draw_note(cv, 470, 244, "root = smallest id: run 0", fa)
        draw_note(cv, 470, 290, "reads run_x and run_y0 of the root,", fa, size=28)
        draw_note(cv, 470, 322, "writes run_label (and parent[r] = root).", fa * ramp(t, 26.8, 27.1), size=28)


# ------------------------------------------------------ beats 13-14: paint
SL_X0, SL_PITCH, SL_Y = 248, 28, 208
AR_X0, AR_PITCH = 296, 29
AR_Y = [380, 424, 468, 512]
AR_NAMES = ["run_x", "run_y0", "run_y1", "run_label"]
BURST = (0xF0, 0xA8, 0x10)


def paint_times(B):
    """When the highlight reaches each of the 32 columns of the chunk."""
    dur = [0.37 if 17 <= c <= 20 else 0.04 for c in range(32)]
    start = [29.6 + sum(dur[:c]) for c in range(32)]
    return start, dur


def beat13_14(cv, B, t):
    if t < 27.7 or t > 32.5:
        return
    a = vis(t, 28.0, 32.4, 0.4, 0.35)
    if a <= 0.01:
        return
    start, dur = paint_times(B)
    walk = B.walk
    row5 = [r for r in B.runs if r["row"] == 5]
    assert [r["id"] for r in row5] == [5, 6, 7, 8]
    # row 5 of the image as a 28-pixel slice
    tray(cv, SL_X0 - 6, SL_Y - 4, SL_X0 + SL_PITCH * B.n_cols + 5, SL_Y + SL_PITCH + 3, a)
    cv.text(48, SL_Y + 14, "image row 5", 28, mix(GREY, a, PANEL), "mono", "lm")
    for j in range(B.n_cols):
        if not B.K[5, j]:
            cv.rect(SL_X0 + SL_PITCH * j, SL_Y, SL_X0 + SL_PITCH * (j + 1) - 1, SL_Y + SL_PITCH - 1, fill=mix(CELL_BG, a, WHITE))
    label_slot = {}
    for n, r in enumerate(row5):
        tc = start[17 + n]
        x0 = SL_X0 + SL_PITCH * r["j0"]
        x1 = SL_X0 + SL_PITCH * (r["j1"] + 1) - 1
        if t < tc + 0.10:
            cv.rect(x0, SL_Y, x1, SL_Y + SL_PITCH - 1, fill=mix(RED, a, WHITE))
        elif t < tc + 0.22:
            cv.rect(x0, SL_Y, x1, SL_Y + SL_PITCH - 1, fill=mix(BURST, a, WHITE), outline=mix(AMBER, a, WHITE), w=2)
        else:
            cv.rect(x0, SL_Y, x1, SL_Y + SL_PITCH - 1, fill=mix(CYAN, a, WHITE), outline=mix(CYAN_EDGE, a, WHITE), w=2)
        # teal bracket above the run while it is being handled, run id above the slice
        id_a = a * ramp(t, 28.6, 28.9)
        cv.text((x0 + x1) / 2, SL_Y - 8, "run %d" % r["id"], 22, mix(TEAL, id_a, PANEL), "mono", "ms")
        if tc <= t < tc + 0.22:
            cv.line(x0, SL_Y - 3, x1, SL_Y - 3, mix(TEAL, a, PANEL), 3)
        if t >= tc + 0.25:
            # the bytes this run writes, centred under the run (all in one row)
            la = ramp(t, tc + 0.25, tc + 0.4) * a
            cx = (x0 + x1) / 2
            cv.line(cx, SL_Y + SL_PITCH + 3, cx, SL_Y + SL_PITCH + 14, mix(AMBER, la, PANEL), 2.5)
            cv.text(cx, SL_Y + SL_PITCH + 16, "%d B" % (3 * r["n"]), 40, mix(AMBER, la, PANEL), "mono", "ma")
        # while this run is painted, say how many of the 32 lanes work on it
        lv = vis(t, tc, tc + dur[17 + n], 0.06, 0.08) * a
        if lv > 0.01:
            cv.text(846, 112, "%d px: %d of 32 lanes" % (r["n"], r["n"]), 30, mix(AMBER, lv), "bold", "lm")
    # the chunk of 32 runs: four arrays side by side
    ca = ramp(t, 28.2, 28.7) * a
    if ca > 0.01:
        for ai, name in enumerate(AR_NAMES):
            cv.text(AR_X0 - 10, AR_Y[ai] + 14, name, 28, mix(TEAL, ca, PANEL), "mono", "rm")
            for col in range(32):
                x0 = AR_X0 + AR_PITCH * col
                ours = 17 <= col <= 20
                fill = TEAL if ours else FOREIGN
                if ours and ai == 3:        # run_label: cyan, as flatten drew it
                    cv.rect(x0, AR_Y[ai], x0 + AR_PITCH - 2, AR_Y[ai] + 28, fill=mix(CYAN, ca, PANEL),
                            outline=mix(CYAN_EDGE, ca, PANEL), w=2)
                else:
                    cv.rect(x0, AR_Y[ai], x0 + AR_PITCH - 2, AR_Y[ai] + 28, fill=mix(fill, ca, PANEL))
                if ours and ai == 0:
                    cv.text(x0 + AR_PITCH / 2 - 1, AR_Y[ai] + 15, str(walk[col - 17]["row"]), 18, mix(WHITE, ca, PANEL), "mono", "mm")
        cv.rrect(AR_X0 - 4, AR_Y[0] - 8, AR_X0 + AR_PITCH * 32 + 1, AR_Y[3] + 36, 8,
                 outline=mix(TEAL, ca, PANEL), w=3)
        # which column is which run (before the walk-through starts)
        link = ca * (1 - ramp(t, 29.45, 29.65))
        if link > 0.01:
            for n, r in enumerate(row5):
                sx = (SL_X0 + SL_PITCH * r["j0"] + SL_X0 + SL_PITCH * (r["j1"] + 1)) / 2
                cx = AR_X0 + AR_PITCH * (17 + n) + AR_PITCH / 2 - 1
                cv.dline(sx, SL_Y + SL_PITCH + 8, cx, AR_Y[0] - 10, mix(TEAL, link, PANEL), 2, 5, 4)
        draw_note(cv, AR_X0 - 4, AR_Y[3] + 48, "28 are other blobs' runs.", ca * (1 - ramp(t, 29.6, 29.9)), bg=PANEL)
        draw_note(cv, AR_X0 - 4, AR_Y[3] + 48, "Slowed down for this blob's 4 runs.",
                  ramp(t, 29.9, 30.2) * ca, bg=PANEL)
        draw_note(cv, AR_X0 - 4, AR_Y[3] + 84, "All 4 are in row 5 (local), so run_x is the same in each.",
                  ramp(t, 28.9, 29.2) * ca, size=26, bg=PANEL)
        # the highlight steps along the columns
        if t >= 29.6:
            col = max(c for c in range(32) if t >= start[c])
            ha = ca
            x0 = AR_X0 + AR_PITCH * col
            cv.rect(x0 - 2, AR_Y[0] - 5, x0 + AR_PITCH, AR_Y[3] + 33, outline=mix(AMBER, ha, PANEL), w=3)
        draw_note(cv, 48, 650, "White is never written.", ramp(t, 31.2, 31.6) * ca, bg=PANEL)


# ------------------------------------------------------------- beat 15
CH_X0, CH_Y0, CH_PITCH_X, CH_PITCH_Y = 48, 150, 12, 23
BL_X, BL_Y, BL_C = 520, 150, 20
SWEEP = (32.5, 35.4)


def burst_state(t, tc):
    """0 = not yet, 1 = amber flash, 2 = painted."""
    if t < tc:
        return 0
    return 1 if t < tc + 0.12 else 2


def beat15(cv, B, t):
    if t < 32.2 or t > 36.6:
        return
    a = vis(t, 32.4, 36.6, 0.5, 0.3)
    if a <= 0.01:
        return
    step_t = (SWEEP[1] - SWEEP[0]) / 32.0
    # chunk bars
    for cr in range(len(B.chunks)):
        for col in range(32):
            ours = [r for r in B.runs if r["crow"] == cr and r["off"] == col]
            x0 = CH_X0 + CH_PITCH_X * col
            y0 = CH_Y0 + CH_PITCH_Y * cr
            if not ours:
                cv.rect(x0, y0, x0 + CH_PITCH_X - 1, y0 + 18, fill=mix(FOREIGN, a, PANEL))
                continue
            st = burst_state(t, SWEEP[0] + col * step_t)
            fill = (TEAL, BURST, CYAN)[st]
            cv.rect(x0, y0, x0 + CH_PITCH_X - 1, y0 + 18, fill=mix(fill, a, PANEL),
                    outline=mix(CYAN_EDGE, a, PANEL) if st == 2 else None, w=1)
    # the picture, painted run by run
    bx, by = BL_X, BL_Y
    cell = mix(CELL_BG, a)
    for i in range(B.n_rows):
        for j in range(B.n_cols):
            if not B.K[i, j]:
                cv.rect(*cell_rect(bx, by, BL_C, i, j), fill=cell)
    for r in B.runs:
        st = burst_state(t, SWEEP[0] + r["off"] * step_t)
        if st == 0:
            for j in range(r["j0"], r["j1"] + 1):
                cv.rect(*cell_rect(bx, by, BL_C, r["row"], j), fill=mix(RED, a))
        else:
            x0, y0, x1, y1 = run_rect(bx, by, BL_C, r)
            if st == 1:
                cv.rect(x0, y0, x1 - 1, y1 - 1, fill=mix(BURST, a), outline=mix(AMBER, a), w=2)
            else:
                cv.rect(x0, y0, x1 - 1, y1 - 1, fill=mix(CYAN, a), outline=mix(CYAN_EDGE, a), w=2)
    # the sweep line and the counter
    if SWEEP[0] <= t <= SWEEP[1] + 0.1:
        kf = clamp01((t - SWEEP[0]) / (SWEEP[1] - SWEEP[0])) * 32
        lx = CH_X0 + CH_PITCH_X * min(kf, 31.999)
        cv.line(lx + 0.5, CH_Y0 - 6, lx + 0.5, CH_Y0 + CH_PITCH_Y * len(B.chunks) - 2, mix(AMBER, a), 3)
        k = min(int(kf) + 1, 32)
        cv.text(BL_X + 14 * BL_C, 654, "step %d of 32" % k, 40, mix(AMBER, a), "mono", "ma")
    elif t > SWEEP[1] + 0.1:
        cv.text(BL_X + 14 * BL_C, 654, "step 32 of 32", 40, mix(AMBER, a), "mono", "ma")
    ga = ramp(t, 32.9, 33.0) * a                       # appears at full grey, not pale
    draw_note(cv, 48, 632, "each row = one chunk of 32 runs,", ga, size=24)
    draw_note(cv, 48, 660, "teal = this blob. Real order varies,", ga, size=24)
    draw_note(cv, 48, 688, "shown in step.", ga, size=24)


# ------------------------------------------------------------- beats 16-17
LAD_X0 = 80


LADDER_OUT = (39.3, 39.5)      # the ladder is gone before the race starts


def beat16(cv, B, t):
    """The four layouts of this blob on one scale. The bars come 0.3 s apart,
    then the whole ladder stays for 1.5 s."""
    if t < 36.4 or t > LADDER_OUT[1]:
        return
    out = 1.0 - ramp(t, *LADDER_OUT)
    rows = [("RGB in", B.b_rgb, BAR_GREY, None), ("bits", B.b_mask, AMBER, None),
            ("runs", B.b_runs, TEAL, None), ("painted", B.b_paint, CYAN, CYAN_EDGE)]
    full = 1100.0
    for i, (name, nbytes, color, edge) in enumerate(rows):
        t0 = 36.7 + 0.3 * i
        ba = ramp(t, t0, t0 + 0.2) * out
        if ba <= 0.01:
            continue
        y = 190 + 104 * i
        cv.text(LAD_X0, y + 14, name, 28, mix(GREY, ba, PANEL), "mono", "lm")
        cv.text(LAD_X0 + 175, y + 14, "%s B" % format(nbytes, ","), 40, mix(INK, ba, PANEL), "mono", "lm")
        grow = ease(ramp(t, t0, t0 + 0.4))
        cv.rect(LAD_X0, y + 44, LAD_X0 + max(2, full * nbytes / B.b_rgb * grow), y + 82, fill=mix(color, ba, PANEL),
                outline=mix(edge, ba, PANEL) if edge else None, w=2)
    na = ramp(t, 38.1, 38.4) * out
    draw_note(cv, LAD_X0, 616, "This blob: 24 rows x 64 px. Whole image: %d MB to %.1f MB of bits." % (
        round(B.img_bytes / 1e6), B.mask_bytes_all / 1e6), na, size=28, bg=PANEL)
    draw_note(cv, LAD_X0, 652, "runs = x, y0, y1 only. Counted bytes, not DRAM traffic.", na, size=28, bg=PANEL)


SLOWDOWN = 29                 # 1 ms of real time takes 29 ms of video
MS_TO_VIDEO = SLOWDOWN / 1000.0
RACE_T0 = 40.3                # the shared start, after the ladder has gone and the race has faded in


def beat17(cv, B, t):
    """Chapter 5 against chapter 6 on one clock, played 29x slower than real time."""
    if t < 39.42:
        return
    a = ramp(t, 39.42, 39.72)           # starts while the ladder is still fading out: no blank frame
    x0, full = 80, 1100.0
    real_ms = max(0.0, (t - RACE_T0) / MS_TO_VIDEO)
    for name, total, color, y in (("Chapter 5", B.ch5_ms, AMBER, 262), ("Chapter 6", B.ch6_ms, TEAL, 440)):
        ms = min(real_ms, total)
        ln = full * ms / B.ch5_ms
        cv.text(x0, y - 18, name, 28, mix(GREY, a), "mono", "ls")
        cv.rect(x0, y, x0 + max(ln, 2), y + 60, fill=mix(color, a))
        label = "%.2f ms" % ms
        lw = text_w(label, 40, "mono")
        if total == B.ch5_ms:
            lx = min(max(x0 + 220, x0 + ln - lw), x0 + full - lw)       # above the bar end
            cv.text(lx, y - 14, label, 40, mix(INK, a), "mono", "ls")
        else:
            cv.text(x0 + ln + 18, y + 31, label, 40, mix(INK, a), "mono", "lm")   # right of the bar end
    # the speedup pops next to the short bar's label
    t_end = RACE_T0 + B.ch5_ms * MS_TO_VIDEO          # the long bar is full
    xa = ease(ramp(t, t_end - 0.3, t_end - 0.05))
    if xa > 0.01:
        ex = x0 + full * B.ch6_ms / B.ch5_ms + 18 + text_w("2.96 ms", 40, "mono") + 40
        cv.text(ex, 470, "%.1fx" % B.speedup, 96, mix(TEAL, xa), "bold", "lm")
    draw_note(cv, x0, 570, "From RGB. Bars share one clock,", a)
    draw_note(cv, x0, 606, "played %dx slower than real time." % SLOWDOWN, a)


# ----------------------------------------------------------------- the video
def frame_video(B, t_video):
    t = design_time(t_video)             # every beat below is authored in design time
    cv = Cv(W, H)
    draw_panel(cv, t)
    draw_thumb(cv, B, t, **thumb_opts(B, t))
    beat1(cv, B, t)
    beat3(cv, B, t)
    beat4(cv, B, t)
    beat6(cv, B, t)
    draw_table(cv, B, t)
    draw_count_words(cv, B, t)
    draw_count_chips(cv, B, t)
    beat7_8(cv, B, t)
    beat9_12(cv, B, t)
    beat13_14(cv, B, t)
    beat15(cv, B, t)
    beat16(cv, B, t)
    beat17(cv, B, t)
    draw_header(cv, B, t)
    return cv.finish()


# --------------------------------------------------------------- the carousel
CAR_FOLD_GAP = 2.0           # px between two rows of the folded lane


def car_lane_x(lane, B, r, j):
    """Left edge and width of cell (row r, column j) in the lane. The 24 rows of
    28 cells sit one after the other, with a fold gap (and a tick) between rows."""
    cw = (lane[2] - lane[0] - CAR_FOLD_GAP * (B.n_rows - 1)) / float(B.n_rows * B.n_cols)
    return lane[0] + r * (B.n_cols * cw + CAR_FOLD_GAP) + j * cw, cw


def frame_carousel(B, t):
    """Square 720 x 720 loop: the blob as a picture and as one line of memory
    (rows folded: a tick marks each fold), painted piece by piece, 9 s, first
    and last frame both empty."""
    cv = Cv(CAR_SIZE, CAR_SIZE)
    c, bx, by = 14, 164, 36
    lane = (24, 480, 696, 560)
    # crossfade back to the empty first frame
    fade = 1.0 - ramp(t, 8.6, 8.95)
    pop = ease(ramp(t, 0.0, 0.6))
    a = pop * fade
    if a <= 0.01:
        return cv.finish()
    cv.rrect(12, 430, 708, 636, 16, fill=mix(PANEL, a))
    cv.rect(lane[0], lane[1], lane[2], lane[3], fill=mix(WHITE, a, PANEL))
    # the picture pops in
    sc = lerp(0.88, 1.0, pop)
    cx, cy = bx + c * B.n_cols / 2, by + c * B.n_rows / 2
    pc = c * sc
    px, py = cx - pc * B.n_cols / 2, cy - pc * B.n_rows / 2
    step_t = 0.1
    t_paint = 4.5
    states = {r["id"]: burst_state(t, t_paint + r["off"] * step_t) for r in B.runs}
    cell = mix(CELL_BG, a)
    for i in range(B.n_rows):
        for j in range(B.n_cols):
            if not B.K[i, j]:
                cv.rect(*cell_rect(px, py, pc, i, j), fill=cell)
    outline_a = ramp(t, 3.3, 3.9) * fade
    for r in B.runs:
        st = states[r["id"]]
        if st == 0:
            for j in range(r["j0"], r["j1"] + 1):
                cv.rect(*cell_rect(px, py, pc, r["row"], j), fill=mix(RED, a))
            if outline_a > 0.01:
                outline_run(cv, px, py, pc, r, mix(TEAL, outline_a), 2.0, 1.0)
        else:
            x0, y0, x1, y1 = run_rect(px, py, pc, r)
            if st == 1:
                cv.rect(x0, y0, x1 - 1, y1 - 1, fill=mix(BURST, a), outline=mix(AMBER, a), w=2)
            else:
                cv.rect(x0, y0, x1 - 1, y1 - 1, fill=mix(CYAN, a), outline=mix(CYAN_EDGE, a), w=2)
    # rows peel down into the lane: 28 cells squash from 392 px to about 26 px
    lane_cells = {}
    row_p = {}
    for r in range(B.n_rows):
        t0 = 0.6 + 0.09 * r
        p = ease(ramp(t, t0, t0 + 0.45))
        row_p[r] = p
        if p <= 0:
            continue
        for j in range(B.n_cols):
            sx0, sy0, sx1, sy1 = cell_rect(bx, by, c, r, j, gap=0.0)
            dx0, cw = car_lane_x(lane, B, r, j)
            dx1 = dx0 + cw
            x0, y0 = lerp(sx0, dx0, p), lerp(sy0, lane[1], p)
            x1 = lerp(sx1, dx1, p)
            # the cell only grows to the lane height once it is inside the lane
            y1 = min(lane[3], y0 + lerp(sy1 - sy0, lane[3] - lane[1], ramp(p, 0.92, 1.0)))
            lane_cells[(r, j)] = (x0, y0, x1, y1, p)
    flying = []
    for (r, j), (x0, y0, x1, y1, p) in lane_cells.items():
        if B.K[r, j]:
            col = RED
            rid = next(q["id"] for q in B.runs if q["row"] == r and q["j0"] <= j <= q["j1"])
            st = states[rid]
            if st == 1:
                col = BURST
            elif st == 2:
                col = CYAN
        else:
            col = CELL_BG
        if p >= 1:
            cv.rect(x0, y0, x1 + 0.35, y1, fill=mix(col, a))
        elif B.K[r, j]:
            flying.append((x0, y0, x1, y1, mix(col, a)))      # only the red cells travel
    for x0, y0, x1, y1, col in flying:
        cv.rect(x0, y0, x1, y1, fill=col)
    # a fold tick between two rows, once the later row has landed
    for r in range(1, B.n_rows):
        ka = ramp(row_p[r], 0.9, 1.0) * a
        if ka > 0.01:
            fx = lane[0] + r * (B.n_cols * car_lane_x(lane, B, 0, 1)[1] + CAR_FOLD_GAP) - CAR_FOLD_GAP / 2.0
            cv.line(fx, lane[1] - 3, fx, lane[3] + 3, mix(GREY, ka, PANEL), 1.5)
    # the lane's title sits under it, out of the way of the flying cells
    cv.text(28, 588, "one line of memory", 36, mix(GREY, a, PANEL), "bold")
    # teal ticks and the count
    ta = ramp(t, 3.3, 4.5) * fade
    if ta > 0.01:
        for r in B.runs:
            n0 = 3.3 + 1.2 * r["id"] / 33.0
            k = ramp(t, n0, n0 + 0.15) * a
            if k <= 0.01:
                continue
            xa = car_lane_x(lane, B, r["row"], r["j0"])[0]
            xb = car_lane_x(lane, B, r["row"], r["j1"] + 1)[0]
            cx = (xa + xb) / 2
            cv.rect(min(xa, cx - 1.5), lane[3] + 4, max(xb, cx + 1.5), lane[3] + 16, fill=mix(TEAL, k, PANEL))
        cv.text(CAR_SIZE / 2, 419, "34 runs", 48, mix(TEAL, ta, WHITE), "bold", "ms")
    # the step bar
    ba = ramp(t, 4.3, 4.7) * a
    if ba > 0.01:
        k_now = min(31, max(0, int((t - t_paint) / step_t))) if t >= t_paint else -1
        for k in range(32):
            x0 = 24 + k * (672 / 32.0)
            col = AMBER if k <= k_now else HAIR
            cv.rect(x0, 646, x0 + 672 / 32.0 - 3, 660, fill=mix(col, ba))
        label = "20 chunks, step %d of 32" % max(1, k_now + 1)
        cv.text(CAR_SIZE / 2, 668, label, 36, mix(AMBER, ba), "mono", "ma")
    return cv.finish()


# ------------------------------------------------------------------- output
X264 = ["-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", "tv", "-movflags", "+faststart", "-an"]
TO_YUV = "scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,format=yuv420p"


def encode(make_frame, size, n_frames, out_path, crf, poster_t, poster_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(poster_path), exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % size, "-r", str(FPS), "-i", "-",
           "-vf", TO_YUV, "-r", str(FPS)] + X264 + ["-crf", str(crf), out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    poster_frame = int(round(poster_t * FPS))
    for f in range(n_frames):
        img = make_frame(f / FPS)
        if f == poster_frame:
            img.save(poster_path, lossless=True, quality=100, method=6)
        try:
            proc.stdin.write(img.tobytes())
        except BrokenPipeError:
            break
        if f % 150 == 0:
            print("  frame %d / %d" % (f, n_frames), flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    for p in (out_path, poster_path):
        print("  %9d  %s" % (os.path.getsize(p), os.path.relpath(p, REPO)))


def check_still_path(path):
    """--still writes a PNG for review. Never inside project-page (the repo
    gitignores .png and .jpg there)."""
    inside = os.path.abspath(path).startswith(os.path.join(REPO, "project-page") + os.sep)
    if inside:
        sys.exit("runs_explainer.py: refusing to write %s inside project-page" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=("video", "carousel"), help="build one of the two clips")
    ap.add_argument("--still", type=float, metavar="T", help="write the frame at T seconds as a PNG (needs --out)")
    ap.add_argument("--carousel", action="store_true", help="with --still: the carousel loop instead of the video")
    ap.add_argument("--design", action="store_true", help="with --still: T is design time, not video time")
    ap.add_argument("--out", help="PNG path for --still, outside project-page")
    args = ap.parse_args()

    B = load_blob()
    print("checks passed: runs, row counts, partners, both merge rounds, bytes, benchmark figures")
    print("  video %.1f s, %d frames; poster at %.2f s; carousel %.1f s, poster at %.1f s" % (
        DURATION, round(DURATION * FPS), video_time(POSTER_D), CAR_DURATION, CAR_POSTER_T))

    if args.still is not None:
        if not args.out:
            sys.exit("--still needs --out")
        check_still_path(args.out)
        t_still = video_time(args.still) if args.design and not args.carousel else args.still
        img = frame_carousel(B, t_still) if args.carousel else frame_video(B, t_still)
        img.save(args.out)
        print("wrote", args.out)
        return

    if args.only in (None, "video"):
        print("video ->", os.path.relpath(OUT_VIDEO, REPO))
        encode(lambda t: frame_video(B, t), (W, H), int(round(DURATION * FPS)), OUT_VIDEO, 23,
               video_time(POSTER_D), OUT_POSTER)
    if args.only in (None, "carousel"):
        print("carousel ->", os.path.relpath(OUT_CAROUSEL, REPO))
        encode(lambda t: frame_carousel(B, t), (CAR_SIZE, CAR_SIZE), int(round(CAR_DURATION * FPS)),
               OUT_CAROUSEL, 22, CAR_POSTER_T, OUT_CAROUSEL_POSTER)


if __name__ == "__main__":
    main()
