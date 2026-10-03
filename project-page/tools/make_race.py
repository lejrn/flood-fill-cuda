#!/usr/bin/env python
"""Build the project page's race video: one CPU lane and three GPU lanes
filling the same 8000 x 8000 square on one shared, time-true clock.

Run from the repo root (the directory that holds src/ and project-page/):

    FFMPEG=/path/to/ffmpeg python project-page/tools/make_race.py

FFMPEG  ffmpeg with libx264 (default: ffmpeg on PATH)
FONT, FONT_BOLD  TTF files (default: the first of DejaVu Sans, Liberation
        Sans or Noto Sans that exists under /usr/share/fonts)

Needs numpy and Pillow (built with WebP support). No GPU, nothing but the
two outputs is written to disk; frames go to ffmpeg as raw RGB on a pipe.

Outputs:
    project-page/static/videos/race.mp4          H.264, CRF 20, 30 fps, BT.709
    project-page/static/images/race_poster.webp  a frame where the three GPU
                                                 lanes are done and the CPU
                                                 is about half way

The data (tools/race_spec.json) holds only the measured totals, the scene and
the timeline. Ring sizes and per-level clocks are recomputed here. If the
source benchmark JSON named in the spec is present, every number is checked
against it before a frame is drawn.

What is measured and what is modeled
-------------------------------------
Measured (benchmark JSON, median of 5, one session): each lane's total time.
Modeled: how that total is spread over the 8001 BFS levels. The video says
so on screen ("Finish times are measured. The fill in between is modeled."). Per level k the
lane spends t_k = o + c * n_k, where n_k is the exact ring size (the pixels
at BFS level k), o is a per-level cost taken from the one-pixel-per-level
"serpentine" scene (CPU: 0), and c is chosen so the lane ends exactly at its
measured total. Inside a level, pixels fill linearly with time. The finish
times are exact by construction; the curve shape is approximate.

Which numbers: the session is the one behind the Chapter 3 README tables
(source.note in the spec; a later session gives a different CPU time, so the
speedups are one session's values, not constants). Each "done" card shows the
lane's measured total, its speedup over the CPU (CPU total / lane total, equal
to the JSON's speedup fields), and how far the CPU lane had got at the instant
this lane finished under the constant-rate CPU model ("CPU then: ~6.5%
(model)"). That line is past tense on purpose: the card stays on screen after
the CPU has moved on. GPU totals are kernel time only; the CPU total is the
whole cpu_flood_fill call, which also allocates its arrays. The lanes are
anti-aliased at the fill front so it moves smoothly even when it advances less
than one display pixel per frame.

Timing on screen: a lane is marked done on the frame nearest to its measured
time (round, not ceil), so every lane ends within half a frame (2.8 ms of
benchmark time) of its measured total and the on-screen frame counts keep the
true speedups (367 / 24 frames = 15.3x for 48 blocks).

The video plays at 6x slow motion: every lane advances in real benchmark
milliseconds, the shared clock shows those milliseconds, and one video second
is 1000 / 6 ms of real time. Lanes that were faster in the benchmark finish
sooner on screen, in proportion.
"""

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(HERE, "race_spec.json")
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT_VIDEO = os.path.join(REPO, "project-page", "static", "videos", "race.mp4")
OUT_POSTER = os.path.join(REPO, "project-page", "static", "images",
                          "race_poster.webp")

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# ---------------------------------------------------------------- palette
# Copied from chapters/ch01_gpu_1blob_1block/benchmarks/wavefront.py, so the
# race reads like the chapter GIFs on the same page.
BLUE_ANCHORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
FRONTIER_BLUE = "#cde2fb"
GREY_ANCHORS = ["#b9c0c8", "#939ba5", "#6e7680", "#4c535c", "#2b3037"]
TRAIL_GREY = "#dde1e6"
UNFILLED_RED = (255, 0, 0)          # the scene's red pixels
BACKGROUND = (255, 255, 255)
INK = (11, 15, 20)                  # wavefront.py CURSOR, used as text color
INK_SOFT = (76, 83, 92)             # GREY_ANCHORS[3]
EDGE = (59, 67, 76)
ACCENT = (13, 54, 107)              # BLUE_ANCHORS[4]


def _hex(c):
    return np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)],
                    dtype=np.float64)


def _ramp_lut(anchors, n=256):
    """n RGB rows interpolated through the anchor colors (as in ch02)."""
    pts = np.array([_hex(a) for a in anchors])
    x = np.linspace(0, len(anchors) - 1, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(x, np.arange(len(anchors)),
                               pts[:, ch]).round().astype(np.uint8)
    return lut


LUT_BLUE = _ramp_lut(BLUE_ANCHORS)
LUT_GREY = _ramp_lut(GREY_ANCHORS)
PALE = {"cpu": _hex(TRAIL_GREY).astype(np.uint8),
        "gpu": _hex(FRONTIER_BLUE).astype(np.uint8)}

# ----------------------------------------------------------------- layout
W = 1080
COL_W = W // 2                      # one lane column
SQ = 480                            # displayed square, in video pixels
HEADER_H = 108
LABEL_H = 46
LANE_GAP = 12
FOOT_LINE = 38                      # footer line step
FOOT_LINES = 4
FOOTER_H = 14 + FOOT_LINES * FOOT_LINE + 12
ROW_H = LABEL_H + SQ + LANE_GAP
H = HEADER_H + 2 * ROW_H + FOOTER_H
assert H % 2 == 0 and W % 2 == 0

# Every size is at least 30 px at 1080 wide (10 px at 360), the page's phone
# width. The labels shrink from FONT_LABEL only if a label would not fit.
FONT_CLOCK = 62
FONT_HEAD = 34
HEAD_LINE = 42
FONT_LABEL = 33
FONT_LABEL_MIN = 30
FONT_DONE = 44
FONT_FASTER = 38
FONT_KERNEL = 30
FONT_CPUPCT = 32
FONT_FOOT = 32

BAND_MIN_LEVELS = 90                # pale frontier band, at least this deep
FADE_FRAMES = 6                     # the "done" card fades in over 0.2 s
CARD_PAD = 48                       # card width = widest text line + this


def find_font(env, names):
    path = os.environ.get(env)
    if path:
        return path
    roots = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype"]
    for root in roots:
        for name in names:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand
    sys.exit("make_race.py: no TTF found, set %s to a font file" % env)


FONT_REG = find_font("FONT", ["DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                              "NotoSans-Regular.ttf"])
FONT_BLD = find_font("FONT_BOLD", ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
                                   "NotoSans-Bold.ttf"])


def font(path, size):
    return ImageFont.truetype(path, size)


# ------------------------------------------------------------------- model
def ring_sizes(width, height, seed):
    """Pixels at each BFS level of a full solid rectangle, 4-connected.

    level(x, y) = |x - sx| + |y - sy|, so the ring sizes are the convolution
    of the two 1-D distance histograms (exact, int64, no big arrays)."""
    def axis_hist(n, s):
        h = np.zeros(max(s, n - 1 - s) + 1, dtype=np.int64)
        np.add.at(h, np.abs(np.arange(n) - s), 1)
        return h
    return np.convolve(axis_hist(width, seed[0]), axis_hist(height, seed[1]))


def build_model(spec):
    sc = spec["scene"]
    assert sc["connectivity"] == 4
    ring = ring_sizes(sc["width"], sc["height"], tuple(sc["seed_xy"]))
    levels, pixels = len(ring), int(ring.sum())
    assert levels == sc["levels"] == 8001
    assert pixels == sc["pixels"] == sc["width"] * sc["height"]
    assert int(ring.max()) == sc["peak_ring"] == 15998
    cum_px = np.cumsum(ring)
    lanes = []
    for ln in spec["lanes"]:
        total = float(ln["total_ms"])
        if ln["overhead"] == "meter":
            o = ln["meter_ms"] / ln["meter_levels"]     # ms per level
        else:
            o = 0.0
        c = (total - levels * o) / pixels               # ms per pixel
        cum_ms = np.cumsum(o + c * ring)
        assert abs(cum_ms[-1] - total) < 1e-6, (ln["key"], cum_ms[-1], total)
        lanes.append(dict(ln, o_ms=o, c_ms=c, cum_ms=cum_ms))
    return ring, cum_px, lanes


def verify_against_source(spec, ring, lanes):
    """If the benchmark JSON is on disk, check every spec number against it."""
    src = os.path.join(REPO, spec["source"]["json"])
    if not os.path.isfile(src):
        print("note: source JSON not found, skipping the cross-check:", src)
        return
    with open(src) as fh:
        d = json.load(fh)
    rows = {x["scene"]: x for x in d["scenes"]}
    row = rows[spec["source"]["scene"]]
    meter = rows[spec["source"]["meter_scene"]]
    assert d["sm_count"] == spec["source"]["sm_count"]
    assert row["filled"] == spec["scene"]["pixels"]
    assert row["levels"] == spec["scene"]["levels"]
    assert row["peak_frontier"] == spec["scene"]["peak_ring"]
    assert row["filled_crosscheck"] == "OK"
    assert np.array_equal(ring, np.array(row["level_sizes"], dtype=np.int64)), \
        "computed rings differ from the kernel's own per-level trace"
    for ln in lanes:
        jk = ln["json_key"]
        assert abs(row[jk] - ln["total_ms"]) < 1e-9, jk
        if ln["overhead"] == "meter":
            assert abs(meter[jk] - ln["meter_ms"]) < 1e-9, jk
            assert meter["levels"] == ln["meter_levels"], jk
    cpu, ch01, _, ch03 = lanes
    assert abs(row["multi_speedup_vs_njit"] - cpu["total_ms"] / ch03["total_ms"]) < 1e-9
    assert abs(row["speedup_v2_vs_njit"] - cpu["total_ms"] / ch01["total_ms"]) < 1e-9
    assert row["multi_distinct_sms"] == 24 and row["multi_blocks"] == 48
    ctx = spec["context"]
    assert abs(row["multi_bare_kernel_ms"] - ctx["multi_bare_kernel_ms"]) < 1e-9
    assert abs(row["multi_total_ms"] - ctx["multi_total_ms"]) < 1e-9
    print("cross-check against", spec["source"]["json"], ": OK")
    print("  for the caption (not drawn): 48 blocks bare kernel {:.2f} ms = {:.1f}x "
          "the CPU, end to end {:.1f} ms = {:.2f}x".format(
              ctx["multi_bare_kernel_ms"], cpu["total_ms"] / ctx["multi_bare_kernel_ms"],
              ctx["multi_total_ms"], cpu["total_ms"] / ctx["multi_total_ms"]))


def lane_state(lane, t_ms):
    """(levels as a float, done) at real time t_ms. A pixel at BFS level k is
    filled once the float is above k, so level k fills during [k, k + 1)."""
    cum = lane["cum_ms"]
    n = len(cum)
    if t_ms >= cum[-1]:
        return float(n), True
    m = int(np.searchsorted(cum, t_ms, side="right"))   # levels fully done
    t0 = cum[m - 1] if m else 0.0
    return m + (t_ms - t0) / (cum[m] - t0), False


def pixels_done(cum_px, lvl):
    """Pixels filled when `lvl` levels (a float) are done: level k contributes
    its ring while lvl runs from k to k + 1."""
    return float(np.interp(lvl, np.arange(len(cum_px) + 1),
                           np.concatenate(([0], cum_px))))


# --------------------------------------------------------------- display
def display_grid(spec, ring, cum_px):
    """Per display pixel: its BFS level (float) and the two color indexes.

    The 8000 x 8000 scene is shown at SQ x SQ. A display pixel stands for a
    (8000 / SQ)-pixel block and takes the BFS level of the block's center."""
    sc = spec["scene"]
    sx, sy = sc["seed_xy"]
    step = sc["width"] / SQ
    xc = (np.arange(SQ) + 0.5) * step - 0.5            # scene index of center
    yc = (np.arange(SQ) + 0.5) * (sc["height"] / SQ) - 0.5
    # array [row = y, col = x], the way an image is drawn
    level = np.abs(yc - sy)[:, None] + np.abs(xc - sx)[None, :]
    levels = len(ring)
    blue_idx = np.clip(level / (levels - 1) * 255, 0, 255).astype(np.int64)
    # CPU grey ramp: by visit order, the fraction of pixels visited before
    # this display pixel's level (cum_px[k] counts through level k)
    visit = np.interp(level, np.arange(levels), cum_px) / cum_px[-1]
    grey_idx = np.clip(visit * 255, 0, 255).astype(np.int64)
    return level, blue_idx, grey_idx, step


class LaneRenderer:
    """Draws one lane. A display pixel stands for a block of `step` x `step`
    scene pixels whose BFS levels span about +-step around the block center's
    level, so the fill front is anti-aliased by coverage: the pixel mixes red
    and its fill color linearly while the front crosses that range. Without
    it the front would move in whole display pixels (step levels at a time)
    and stall on some frames of the slow CPU and 1-block lanes."""

    def __init__(self, key, level, idx, lut, step):
        self.level = level
        self.colors = lut[idx]                 # (SQ, SQ, 3) color per pixel
        self.pale = PALE["cpu" if key == "cpu" else "gpu"].astype(np.float32)
        self.step = float(step)
        self.last_lvl = 0.0
        self.final = self.colors.copy()
        self.red = np.empty((SQ, SQ, 3), dtype=np.uint8)
        self.red[:] = UNFILLED_RED

    def render(self, lvl, done):
        if done:
            return self.final
        if lvl <= 0:
            return self.red
        # pale frontier: what this frame added, at least BAND_MIN_LEVELS deep
        depth = max(lvl - self.last_lvl, BAND_MIN_LEVELS)
        self.last_lvl = lvl
        lo = lvl - depth                       # back edge of the pale band
        s = self.step
        # beyond the ramp the answer is exact: fill color behind the band,
        # red ahead of the front
        out = np.where((self.level < lo - s)[:, :, None], self.colors, self.red)
        m = (self.level >= lo - s) & (self.level <= lvl + s)
        lv = self.level[m]
        a = np.clip((lvl - lv) / (2 * s) + 0.5, 0.0, 1.0).astype(np.float32)[:, None]
        b = np.clip((lv - lo) / (2 * s) + 0.5, 0.0, 1.0).astype(np.float32)[:, None]
        fill = self.colors[m].astype(np.float32) * (1 - b) + self.pale * b
        red = np.asarray(UNFILLED_RED, dtype=np.float32)
        out[m] = np.rint(red * (1 - a) + fill * a).astype(np.uint8)
        return out


# ------------------------------------------------------------------ text
def fmt_ms(v):
    return "{:,.0f} ms".format(v)


def fit_label(draw, bold, rest, max_w):
    """Bold name + regular detail on one line, shrunk until it fits."""
    for size in range(FONT_LABEL, FONT_LABEL_MIN - 1, -1):
        fb, fr = font(FONT_BLD, size), font(FONT_REG, size)
        wb = draw.textlength(bold, font=fb)
        wr = draw.textlength("  " + rest, font=fr)
        if wb + wr <= max_w:
            return fb, fr, wb
    sys.exit("label does not fit at %d px: %s %s" % (FONT_LABEL_MIN, bold, rest))


def lane_origin(i):
    col, row = i % 2, i // 2
    x = col * COL_W + (COL_W - SQ) // 2
    y = HEADER_H + row * ROW_H + LABEL_H
    return x, y


HEADER_TEXT = ("played 6x slower than real time", "64 megapixels (8000 x 8000)")
FOOT_TEXT = ("Lighter shade = filled earlier.",
             "Finish times are measured. The fill in between is modeled.",
             "GPU: kernel time only. CPU: whole call, allocation included.")
# color key: a flat swatch for red, a light-to-dark ramp for each fill color
KEY = (("not filled yet", 26, [UNFILLED_RED]),
       ("GPU filled", 64, [tuple(int(v) for v in LUT_BLUE[i]) for i in range(0, 256, 4)]),
       ("CPU filled", 64, [tuple(int(v) for v in LUT_GREY[i]) for i in range(0, 256, 4)]))


def make_background(lanes):
    img = Image.new("RGB", (W, H), BACKGROUND)
    d = ImageDraw.Draw(img)
    f_head = font(FONT_REG, FONT_HEAD)
    f_foot = font(FONT_REG, FONT_FOOT)
    # header, right block (the clock is drawn per frame, on the left)
    for i, text in enumerate(HEADER_TEXT):
        w = d.textlength(text, font=f_head)
        d.text((W - 30 - w, 18 + i * HEAD_LINE), text, font=f_head, fill=INK)
    # lane labels and square frames
    for i, ln in enumerate(lanes):
        x, y = lane_origin(i)
        fb, fr, wb = fit_label(d, ln["label_bold"], ln["label_rest"],
                               min(SQ + 40, W - 30 - x))
        ty = y - LABEL_H + 4
        d.text((x, ty), ln["label_bold"], font=fb, fill=INK)
        d.text((x + wb, ty), "  " + ln["label_rest"], font=fr, fill=INK_SOFT)
        d.rectangle((x - 2, y - 2, x + SQ + 1, y + SQ + 1), outline=EDGE, width=2)
    # footer: color key, what shade means, what is measured, what the
    # GPU and CPU times cover
    y0 = H - FOOTER_H + 14
    widths = [d.textlength(t, font=f_foot) + sw + 12 for t, sw, _ in KEY]
    gap = 40
    x = (W - (sum(widths) + gap * (len(KEY) - 1))) / 2
    for (text, sw, cols), wd in zip(KEY, widths):
        n = len(cols)
        for j, rgb in enumerate(cols):
            x0 = int(round(x + j * sw / n))
            x1 = int(round(x + (j + 1) * sw / n))
            d.rectangle((x0, y0 + 5, x1, y0 + 5 + 26), fill=rgb)
        d.rectangle((x, y0 + 5, x + sw, y0 + 5 + 26), outline=EDGE, width=1)
        d.text((x + sw + 12, y0), text, font=f_foot, fill=INK)
        x += wd + gap
    for i, text in enumerate(FOOT_TEXT):
        w = d.textlength(text, font=f_foot)
        assert w <= W - 40, ("footer line too wide", text, w)
        d.text(((W - w) / 2, y0 + (i + 1) * FOOT_LINE), text, font=f_foot, fill=INK)
    return np.asarray(img).copy()


def fmt_pct(v):
    return "{:.1f}%".format(v) if v < 10 else "{:.0f}%".format(v)


def card_lines(ln, speedup, cpu_pct):
    """The text of a lane's 'done' card: (line1, line2, tail, line3).
    cpu_pct is how far the CPU lane had got (percent of pixels) at the instant
    this lane finished, under the constant-rate CPU model; None for the CPU
    card. line3 is past tense because the card stays up after the CPU has
    moved on, and carries '(model)' because the CPU curve is not measured."""
    cpu = ln["key"] == "cpu"
    line1 = "done: " + fmt_ms(ln["total_ms"])
    line2 = "baseline" if cpu else "{:.1f}x faster".format(speedup)
    tail = "" if cpu else " (kernel)"
    line3 = None if cpu else "CPU then: ~{} (model)".format(fmt_pct(cpu_pct))
    return line1, line2, tail, line3


def card_width(lines):
    """Width of a card for those lines (shared by the cards that sit side by
    side, so all GPU cards look alike)."""
    f1, f2 = font(FONT_BLD, FONT_DONE), font(FONT_BLD, FONT_FASTER)
    fk, f3 = font(FONT_REG, FONT_KERNEL), font(FONT_REG, FONT_CPUPCT)
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    line1, line2, tail, line3 = lines
    w2 = probe.textlength(line2, font=f2) + (probe.textlength(tail, font=fk) if tail else 0)
    widest = max(probe.textlength(line1, font=f1), w2,
                 probe.textlength(line3, font=f3) if line3 else 0)
    w = int(widest) + CARD_PAD
    assert w <= SQ - 16, ("card wider than the square", line1, w)
    return w


def make_card(ln, lines, w):
    """The 'done' card: (rgb, alpha) arrays, w wide, to be centerd on the
    lane's square."""
    f1, f2 = font(FONT_BLD, FONT_DONE), font(FONT_BLD, FONT_FASTER)
    fk, f3 = font(FONT_REG, FONT_KERNEL), font(FONT_REG, FONT_CPUPCT)
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    line1, line2, tail, line3 = lines
    cpu = ln["key"] == "cpu"
    w2 = probe.textlength(line2, font=f2) + (probe.textlength(tail, font=fk) if tail else 0)
    h = 140 if cpu else 184
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=18, fill=(255, 255, 255),
                        outline=INK, width=3)
    d.text((w / 2, 40), line1, font=f1, fill=INK, anchor="mm")
    color = INK_SOFT if cpu else ACCENT
    if tail:
        x2 = (w - w2) / 2
        d.text((x2, 109), line2, font=f2, fill=color, anchor="ls")
        d.text((x2 + probe.textlength(line2, font=f2), 109), tail, font=fk,
               fill=INK_SOFT, anchor="ls")
    else:
        d.text((w / 2, 96), line2, font=f2, fill=color, anchor="mm")
    if line3:
        d.text((w / 2, 144), line3, font=f3, fill=INK, anchor="mm")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=18, fill=255)
    return np.asarray(img), np.asarray(mask).astype(np.float32) / 255.0


class Clock:
    """'t = 1,234 ms' on the left of the header. DejaVu digits are tabular,
    so the text only shifts when the number gains a digit."""

    def __init__(self):
        self.font = font(FONT_BLD, FONT_CLOCK)
        probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        self.widest = 30 + probe.textlength("t = 0,000 ms", font=self.font)

    def draw(self, d, t_ms):
        d.text((30, HEADER_H // 2), "t = {:,.0f} ms".format(t_ms),
               font=self.font, fill=INK, anchor="lm")


# ------------------------------------------------------------------ main
def main():
    with open(SPEC_PATH) as fh:
        spec = json.load(fh)
    ring, cum_px, lanes = build_model(spec)
    verify_against_source(spec, ring, lanes)

    tl = spec["timeline"]
    fps = tl["fps"]
    ms_per_frame = 1000.0 / tl["slowdown_x"] / fps
    lead = int(round(tl["lead_in_s"] * fps))
    cpu = lanes[0]
    assert cpu["key"] == "cpu"
    for ln in lanes:
        ln["speedup"] = cpu["total_ms"] / ln["total_ms"]
        # nearest frame, so a lane ends within half a frame of its measured time
        ln["finish_frame"] = lead + int(round(ln["total_ms"] / ms_per_frame))
    n_frames = (max(ln["finish_frame"] for ln in lanes)
                + int(round(tl["hold_s"] * fps)))
    poster_frame = (lanes[1]["finish_frame"] + FADE_FRAMES)  # CPU about half done

    print("frame {}x{}, {} frames, {:.2f} s, {:.4f} ms of real time per frame".format(
        W, H, n_frames, n_frames / fps, ms_per_frame))
    for ln in lanes:
        print("  {:5s} total {:9.3f} ms  o {:7.4f} us  c {:7.4f} ns/px  "
              "finishes frame {:3d} ({:5.2f} s)  {:5.2f}x CPU".format(
                  ln["key"], ln["total_ms"], ln["o_ms"] * 1e3, ln["c_ms"] * 1e6,
                  ln["finish_frame"], ln["finish_frame"] / fps, ln["speedup"]))
    # on-screen motion frames per lane (from t = 0) against the true speedups,
    # and how far a lane's done frame is from its measured time
    for ln in lanes:
        frames = ln["finish_frame"] - lead
        late = frames * ms_per_frame - ln["total_ms"]
        assert abs(late) <= ms_per_frame / 2 + 1e-9, (ln["key"], late)
        print("  {:5s} {:3d} motion frames, done frame is {:+.1f} ms from measured; "
              "on screen {:5.2f}x, measured {:5.2f}x".format(
                  ln["key"], frames, late, (cpu["finish_frame"] - lead) / frames,
                  ln["speedup"]))
    t_poster = (poster_frame - lead) * ms_per_frame
    lv, _ = lane_state(cpu, t_poster)
    print("poster: frame {} (t = {:.1f} ms), CPU {:.1f}% of pixels".format(
        poster_frame, t_poster, 100 * pixels_done(cum_px, lv) / cum_px[-1]))

    level, blue_idx, grey_idx, step = display_grid(spec, ring, cum_px)
    print("display: 1 pixel = {:.2f} scene pixels".format(step))
    renderers = []
    for ln in lanes:
        if ln["key"] == "cpu":
            renderers.append(LaneRenderer("cpu", level, grey_idx, LUT_GREY, step))
        else:
            renderers.append(LaneRenderer(ln["key"], level, blue_idx, LUT_BLUE, step))

    # how far the CPU lane is when each lane finishes (percent of pixels)
    cpu_pct = []
    for ln in lanes:
        lv, _ = lane_state(cpu, ln["total_ms"])
        pct = 100 * pixels_done(cum_px, lv) / cum_px[-1]
        cpu_pct.append(pct)
        print("  {:5s} done at {:9.3f} ms: CPU at {:5.2f}%".format(ln["key"], ln["total_ms"], pct))

    background = make_background(lanes)
    texts = [card_lines(ln, ln["speedup"], pct) for ln, pct in zip(lanes, cpu_pct)]
    gpu_w = max(card_width(t) for ln, t in zip(lanes, texts) if ln["key"] != "cpu")
    cards = [make_card(ln, t, card_width(t) if ln["key"] == "cpu" else gpu_w)
             for ln, t in zip(lanes, texts)]
    clock = Clock()
    # the clock and the right-hand header block must not collide
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    f_head = font(FONT_REG, FONT_HEAD)
    assert clock.widest < W - 30 - max(probe.textlength(t, font=f_head)
                                       for t in HEADER_TEXT) - 10

    os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_POSTER), exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(fps), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,"
                  "format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-movflags", "+faststart", "-an", OUT_VIDEO]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t_clock_end = cpu["total_ms"]
    for f in range(n_frames):
        t = max(0.0, (f - lead) * ms_per_frame)
        frame = background.copy()
        for i, (ln, rend) in enumerate(zip(lanes, renderers)):
            lvl, _ = lane_state(ln, t)
            done = f >= ln["finish_frame"]          # nearest frame, see above
            x, y = lane_origin(i)
            frame[y:y + SQ, x:x + SQ] = rend.render(lvl, done)
            if done:
                fade = min(1.0, (f - ln["finish_frame"] + 1) / FADE_FRAMES)
                if fade > 0:
                    rgb, a = cards[i]
                    h, w = a.shape
                    cx, cy = x + (SQ - w) // 2, y + (SQ - h) // 2
                    region = frame[cy:cy + h, cx:cx + w].astype(np.float32)
                    a3 = (a * fade)[:, :, None]
                    frame[cy:cy + h, cx:cx + w] = (
                        a3 * rgb + (1 - a3) * region).round().astype(np.uint8)
        img = Image.fromarray(frame)
        # the clock stops on the CPU's measured total, on the CPU's done frame
        clock_ms = t_clock_end if f >= cpu["finish_frame"] else min(t, t_clock_end)
        clock.draw(ImageDraw.Draw(img), clock_ms)
        if f == poster_frame:
            img.save(OUT_POSTER, lossless=True, quality=100, method=6)
        try:
            proc.stdin.write(img.tobytes())
        except BrokenPipeError:
            break
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    for p in (OUT_VIDEO, OUT_POSTER):
        print("  {:9d}  {}".format(os.path.getsize(p), os.path.relpath(p, REPO)))


if __name__ == "__main__":
    main()
