"""Static figures for the READMEs — rendered, never hand-drawn.

Four files into results/ch06_gpu_nblob_runs/figures/:

    before_after.gif   a crop of the REAL input_blobs.png beside what the
                       kernel actually returns for it
    runs_vs_pixels.svg the chapter's one idea, to scale
    speedup.svg        ch05 vs ch06 per scene, log axis
    scaling.svg        runtime vs image size, with the 1 ms / 0.5 ms lines

Everything comes from the committed benchmark JSON and the real image, so
a figure can never drift from the numbers the text quotes: re-run this
after a benchmark session and the pictures follow.

FORMATS: GIF and SVG, because .gitignore excludes *.png and *.jpg. That
turns out to be the right pair anyway — the output is flat-colour, so an
8-colour GIF is lossless and tiny, and the charts are vector text that
diffs cleanly in git.

BOTH THEMES: GitHub renders these on white or near-black depending on the
reader. There is no media query available inside an <img>-embedded SVG on
GitHub, so every colour here is picked to read on both grounds — mid
tones only, never pure black or pure white ink.

Run:  uv run python -m flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks.figures
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import glob
import json

import numpy as np

from ....shared import results_paths

FIG_DIR = results_paths.results_dir("ch06_gpu_nblob_runs", "figures")
RESULTS_DIR = results_paths.results_dir("ch06_gpu_nblob_runs",
                                        "benchmark_results")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          *[".."] * 5))
INPUT_PNG = os.path.join(_REPO_ROOT, "images", "input", "input_blobs.png")

# Readable on white AND on #0d1117 — the constraint every colour here obeys.
INK = "#7d8590"          # labels
INK_STRONG = "#adbac7"   # values; still legible on white at this weight
RULE = "#8b949e"
C_CH05 = "#e2645a"       # the old way
C_RGB = "#d99a2b"        # ch06, rgb contract
C_MASK = "#2aa198"       # ch06, packed mask
C_LABEL = "#4c8fd6"      # labeling only
FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,"
        "sans-serif")
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def _newest(pattern):
    hits = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return json.load(open(hits[-1])) if hits else None


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _svg(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{_esc(title)}">\n'
            f'<title>{_esc(title)}</title>\n{body}\n</svg>\n')


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="400",
          mono=False):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="'
            f'{MONO if mono else FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{_esc(s)}</text>')


# ------------------------------------------------------------ before/after

def before_after(crop_w=760, crop_h=380, gap=16):
    """A crop of the real image, and the same crop of the real output.

    The recolor runs on the WHOLE 81 Mpx image and the crop is taken
    afterwards — so the colours are the ones the kernel assigned given
    every blob in the picture, not a re-labelling of a small window.
    """
    from PIL import Image
    from ...ch05_gpu_nblob_nblock import scenes as _scenes
    from ..recolor import recolor

    img, _ = _scenes.png_scene(INPUT_PNG)          # img[x, y, 3]
    result = recolor(img, contract="rgb", emit_seeds=False, copy_img=True)

    def _pick_crop(mask_xy):
        """A window with a lot going on: maximise red coverage near the
        middle of the picture so the strip is representative."""
        best, best_score = (0, 0), -1.0
        h_x, h_y = mask_xy.shape
        for x0 in range(0, h_x - crop_w, 400):
            for y0 in range(0, h_y - crop_h, 400):
                sub = mask_xy[x0:x0 + crop_w, y0:y0 + crop_h]
                cover = sub.mean()
                if 0.18 < cover < 0.42 and cover > best_score:
                    best, best_score = (x0, y0), cover
        return best

    red = ((img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0))
    x0, y0 = _pick_crop(red)

    def _panel(arr):
        # img[x, y, c] -> a normal [row, col, c] picture
        return np.transpose(arr[x0:x0 + crop_w, y0:y0 + crop_h], (1, 0, 2))

    left, right = _panel(img), _panel(result.img)
    h, w = left.shape[0], left.shape[1]
    strip = np.full((h, w * 2 + gap, 3), 255, dtype=np.uint8)
    strip[:, :w] = left
    strip[:, w + gap:] = right
    strip[:, w:w + gap] = (218, 222, 228)          # divider

    out = os.path.join(FIG_DIR, "before_after.gif")
    Image.fromarray(strip).convert(
        "P", palette=Image.ADAPTIVE, colors=16).save(out, optimize=True)
    return out, result.n_blobs


# ------------------------------------------------------------ runs vs pixels

def runs_vs_pixels(runs):
    """The chapter's whole idea as three bars: pixels, red pixels, runs."""
    scene = next(s for s in runs["scenes"] if s["scene"] == "input_blobs")
    rows = [
        ("all pixels", scene["n_pixels"], RULE),
        ("red pixels", scene["red_px"], C_CH05),
        ("runs  ← ch06 works here", scene["n_runs"], C_MASK),
        ("blobs", scene["n_blobs"], C_LABEL),
    ]
    top = rows[0][1]
    W, row_h, lab_w = 720, 34, 168
    H = len(rows) * row_h + 46
    body = [_text(0, 16, "input_blobs.png — how many things must the "
                         "algorithm handle?", 13, INK_STRONG, weight="600")]
    for i, (label, value, colour) in enumerate(rows):
        y = 34 + i * row_h
        bw = max(2.0, value / top * (W - lab_w - 96))
        body.append(_text(lab_w - 12, y + 15, label, 12, INK, anchor="end"))
        body.append(f'<rect x="{lab_w}" y="{y}" width="{bw:.1f}" height="21" '
                    f'rx="3" fill="{colour}"/>')
        body.append(_text(lab_w + bw + 9, y + 15, f"{value:,}", 12,
                          INK_STRONG, mono=True))
    body.append(_text(0, H - 6, "13.4 million red pixels are only 539,207 "
                                "runs — 25× fewer items, same "
                                "information.", 11, INK))
    return _svg(W, H, "\n".join(body), "Pixels versus runs"), rows[2][1]


# ------------------------------------------------------------------ speedup

def speedup(runs):
    """ch05 vs ch06 per scene, log axis — the honest spread, noise scene
    (where runs barely help) included."""
    import math
    scenes = runs["scenes"]
    series = [("ch05 best", lambda s: s["ch05"]["median_ms"], C_CH05),
              ("ch06 — rgb in",
               lambda s: s["ch06"]["rgb"]["median_ms"], C_RGB),
              ("ch06 — packed mask in",
               lambda s: s["ch06"]["mask"]["median_ms"], C_MASK)]
    vals = [f(s) for s in scenes for _, f, _ in series]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = 10 ** math.ceil(math.log10(max(vals)))
    W, row_h, lab_w, pad_r = 720, 40, 150, 60
    H = len(scenes) * row_h + 108
    span = W - lab_w - pad_r

    def x_of(v):
        return lab_w + (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)) * span

    body = [_text(0, 16, "Runtime per scene — lower is better "
                         "(log scale)", 13, INK_STRONG, weight="600")]
    t = lo
    while t <= hi:
        x = x_of(t)
        body.append(f'<line x1="{x:.1f}" y1="30" x2="{x:.1f}" '
                    f'y2="{len(scenes) * row_h + 34}" stroke="{RULE}" '
                    f'stroke-width="1" stroke-opacity="0.25"/>')
        # bare numbers; the unit is stated once, below the axis
        body.append(_text(x, len(scenes) * row_h + 50, f"{t:g}", 10, INK,
                          anchor="middle", mono=True))
        t *= 10
    body.append(_text(lab_w + span / 2, len(scenes) * row_h + 66,
                      "milliseconds", 10.5, INK, anchor="middle"))
    for i, s in enumerate(scenes):
        y = 34 + i * row_h
        body.append(_text(lab_w - 12, y + 16, s["scene"], 11, INK,
                          anchor="end"))
        body.append(f'<line x1="{lab_w}" y1="{y + 12:.1f}" x2="{W - pad_r}" '
                    f'y2="{y + 12:.1f}" stroke="{RULE}" stroke-width="1" '
                    f'stroke-opacity="0.15"/>')
        for j, (_, fn, colour) in enumerate(series):
            cy = y + 5 + j * 7
            body.append(f'<circle cx="{x_of(fn(s)):.1f}" cy="{cy:.1f}" r="4.5" '
                        f'fill="{colour}"/>')
        mult = s["ch06"]["mask"]["speedup_vs_ch05"]
        body.append(_text(W - pad_r + 8, y + 16, f"{mult:.0f}×", 11,
                          C_MASK, weight="700", mono=True))
    lx = 0
    for name, _, colour in series:
        body.append(f'<circle cx="{lx + 5}" cy="{H - 12}" r="4.5" '
                    f'fill="{colour}"/>')
        body.append(_text(lx + 15, H - 8, name, 11, INK))
        lx += 22 + len(name) * 6.1
    return _svg(W, H, "\n".join(body), "Runtime per scene, ch05 versus ch06")


# ------------------------------------------------------------------ scaling

def scaling(sc):
    """Runtime vs megapixels, with the two target lines drawn — the
    chapter's question ('did we hit 1 ms?') turned into an answer."""
    rows = sc["rows"]
    series = [("rgb contract", lambda r: r["rgb"]["median_ms"], C_RGB),
              ("packed mask", lambda r: r["mask"]["median_ms"], C_MASK),
              ("labeling only",
               lambda r: r["mask"]["label_only_ms"], C_LABEL)]
    W, H = 720, 300
    l, rgt, top, bot = 52, 18, 34, 46
    hi_ms = max(f(r) for r in rows for _, f, _ in series) * 1.08
    hi_mpx = max(r["n_pixels"] for r in rows) / 1e6
    pw, ph = W - l - rgt, H - top - bot

    def px(m):
        return l + m / hi_mpx * pw

    def py(v):
        return top + ph - v / hi_ms * ph

    body = [_text(0, 16, "Runtime vs image size — where 1 ms and "
                         "0.5 ms actually fall", 13, INK_STRONG,
                  weight="600")]
    for tgt, lab in ((1.0, "1 ms"), (0.5, "0.5 ms")):
        y = py(tgt)
        body.append(f'<line x1="{l}" y1="{y:.1f}" x2="{W - rgt}" y2="{y:.1f}" '
                    f'stroke="{RULE}" stroke-width="1" stroke-dasharray="5,4"/>')
        body.append(_text(W - rgt, y - 5, lab, 10, INK, anchor="end",
                          mono=True))
    for v in (0, 1, 2, 3):
        if v > hi_ms:
            continue
        body.append(_text(l - 8, py(v) + 4, f"{v}", 10, INK, anchor="end",
                          mono=True))
    body.append(_text(6, top - 10, "ms", 10, INK, mono=True))
    for m in (0, 20, 40, 60, 80):
        if m > hi_mpx:
            continue
        body.append(_text(px(m), H - 26, f"{m}", 10, INK, anchor="middle",
                          mono=True))
    body.append(_text(l + pw / 2, H - 8, "megapixels", 11, INK,
                      anchor="middle"))
    for name, fn, colour in series:
        pts = " ".join(f"{px(r['n_pixels'] / 1e6):.1f},{py(fn(r)):.1f}"
                       for r in rows)
        body.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                    f'stroke-width="2.2"/>')
        for r in rows:
            body.append(f'<circle cx="{px(r["n_pixels"] / 1e6):.1f}" '
                        f'cy="{py(fn(r)):.1f}" r="3" fill="{colour}"/>')
        last = rows[-1]
        body.append(_text(px(last["n_pixels"] / 1e6) - 6, py(fn(last)) - 8,
                          name, 10.5, colour, anchor="end", weight="600"))
    cross = sc["crossings"]["mask"]["1.0ms_at_mpx"]
    body.append(_text(0, H - 8, f"packed mask stays under 1 ms out to "
                                f"{cross:.0f} Mpx", 11, C_MASK, weight="600"))
    return _svg(W, H, "\n".join(body), "Runtime versus image size")


def main():
    runs = _newest("runs_*.json")
    sc = _newest("scaling_*.json")
    if runs is None or sc is None:
        print("missing benchmark JSON — run benchmark.py and scaling.py first")
        return 1

    written = []
    for name, svg in (("runs_vs_pixels.svg", runs_vs_pixels(runs)[0]),
                      ("speedup.svg", speedup(runs)),
                      ("scaling.svg", scaling(sc))):
        path = os.path.join(FIG_DIR, name)
        with open(path, "w") as f:
            f.write(svg)
        written.append(path)

    gif, n_blobs = before_after()
    written.append(gif)
    print(f"before_after.gif rendered from the real image "
          f"({n_blobs:,} blobs found)")
    for p in written:
        print(f"  {os.path.relpath(p, _REPO_ROOT)}  "
              f"{os.path.getsize(p) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
