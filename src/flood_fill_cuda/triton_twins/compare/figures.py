"""
Static figures for the Triton twins' README, rendered from the committed
comparison JSON, never drawn by hand.

Two files into results/triton_twins/figures/:

    speedup_by_unit.svg  every like-for-like row of every unit, one strip
                         per unit on a log axis, with the unit's geometric
                         mean marked
    grand_table.svg      the grand table, cell by cell: which backend wins
                         each (scene, chapter column), and by how much

Speedup is numba_ms / triton_ms throughout: right of 1x (blue) Triton is
faster, left of it (red) Numba is. Hue says who wins, opacity says by how
much, so a cell that is close to even fades into the page.

BOTH THEMES: like the chapter 6 figures, every ink is a mid tone that
reads on white and on GitHub's #0d1117, and the fills are translucent, so
the page colour itself is the "even" end of the scale. The blue/red pair
passes the dataviz validator on both surfaces (CVD separation 21.0,
normal-vision 29.7, contrast >= 3:1).

Run:  python -m flood_fill_cuda.triton_twins.compare.figures
"""

import json
import math
import os

from flood_fill_cuda.shared.results_paths import results_dir
from flood_fill_cuda.triton_twins.compare.summary import (
    TWINS_ROOT, geomean, newest_per_unit,
)

FIG_DIR = results_dir("triton_twins", "figures")

INK = "#7d8590"          # labels, cell values
INK_STRONG = "#768390"   # titles, means: ~4:1 on white, ~4.4:1 on #0d1117
RULE = "#8b949e"
C_TRITON = "#3987e5"     # Triton faster
C_NUMBA = "#e05a59"      # Numba faster
FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,"
        "sans-serif")
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

# Display order and names; units not listed here are appended by name.
UNITS = [
    ("ch00_cpu_baseline", "ch00 prototype"),
    ("ch01_gpu_1blob_1block", "ch01 one block"),
    ("ch02_gpu_1blob_2block", "ch02 two blocks"),
    ("ch03_gpu_1blob_nblock", "ch03 N blocks"),
    ("ch04_gpu_2blob_nblock", "ch04 two blobs"),
    ("ch05_gpu_nblob_nblock", "ch05 N blobs"),
    ("ch06_gpu_nblob_runs", "ch06 runs"),
    ("scan_multi_blob", "scan experiment"),
    ("overview", "grand table"),
]


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _svg(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{_esc(title)}">\n'
            f'<title>{_esc(title)}</title>\n{body}\n</svg>\n')


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="400",
          mono=False, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="'
            f'{MONO if mono else FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{extra}>'
            f'{_esc(s)}</text>')


def _ratio(s):
    """1.37 -> 'x1.37'; values under 1 keep two significant digits."""
    return f"x{s:.2f}" if s >= 0.995 else f"x{s:.2g}"


def _load_units():
    out = []
    found = newest_per_unit()
    for key, name in UNITS + [(k, k) for k in found
                              if k not in dict(UNITS)]:
        if key in found:
            with open(found[key]) as f:
                out.append((key, name, json.load(f)))
    return out


def _comparable(rows):
    return [r for r in rows if "error" not in r and r.get("comparable", True)
            and r.get("speedup_kernel")]


# ------------------------------------------------------ speedup by unit

def speedup_by_unit(units):
    strips = [(name, _comparable(doc["rows"])) for _, name, doc in units]
    strips = [(n, rows) for n, rows in strips if rows]
    vals = [r["speedup_kernel"] for _, rows in strips for r in rows]
    lo = 2.0 ** math.floor(math.log2(max(min(vals), 1 / 64)))
    hi = 2.0 ** math.ceil(math.log2(min(max(vals), 64)))
    lo, hi = min(lo, 0.5), max(hi, 2.0)

    W, row_h, lab_w, pad_r, top = 760, 34, 170, 70, 44
    H = top + len(strips) * row_h + 74
    span = W - lab_w - pad_r

    def x_of(v):
        v = min(max(v, lo), hi)
        return lab_w + (math.log2(v) - math.log2(lo)) / (
            math.log2(hi) - math.log2(lo)) * span

    body = [_text(0, 16, "Kernel time, Numba / Triton, every like-for-like "
                         "row (log scale)", 13, INK_STRONG, weight="600"),
            _text(0, 33, "right of x1: Triton faster  |  left: Numba "
                         "faster  |  bar: geometric mean", 11, INK)]
    bottom = top + len(strips) * row_h
    t = lo
    while t <= hi * 1.0001:
        x = x_of(t)
        one = abs(t - 1) < 1e-9
        dash = "" if one else ' stroke-dasharray="3 3"'
        body.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" '
                    f'y2="{bottom}" stroke="{RULE}" stroke-width="'
                    f'{1.5 if one else 1}" stroke-opacity="'
                    f'{0.7 if one else 0.22}"{dash}/>')
        body.append(_text(x, bottom + 16, f"x{t:g}", 10, INK,
                          anchor="middle", mono=True))
        t *= 2
    for i, (name, rows) in enumerate(strips):
        y = top + i * row_h
        cy = y + row_h / 2
        body.append(_text(lab_w - 12, cy + 4, name, 11.5, INK_STRONG,
                          anchor="end"))
        body.append(_text(lab_w - 12, cy + 16, f"{len(rows)} rows", 9.5,
                          INK, anchor="end"))
        for j, r in enumerate(sorted(rows, key=lambda r: r["speedup_kernel"])):
            s = r["speedup_kernel"]
            jitter = ((j * 7) % 9 - 4) * 1.6  # deterministic, spreads ties
            body.append(f'<circle cx="{x_of(s):.1f}" cy="{cy + jitter:.1f}" '
                        f'r="3.2" fill="{C_TRITON if s > 1 else C_NUMBA}" '
                        f'fill-opacity="0.55"><title>{_esc(r["experiment"])} '
                        f'{_esc(r["scene"])}: {_ratio(s)}</title></circle>')
        g = geomean(r["speedup_kernel"] for r in rows)
        gx = x_of(g)
        body.append(f'<rect x="{gx - 1.5:.1f}" y="{cy - 11:.1f}" width="3" '
                    f'height="22" rx="1.5" fill="{INK_STRONG}"/>')
        body.append(_text(W - pad_r + 10, cy + 4, _ratio(g), 11.5,
                          INK_STRONG, weight="700", mono=True))
    body.append(_text(lab_w + span / 2, bottom + 34,
                      "numba_ms / triton_ms", 10.5, INK, anchor="middle"))
    lx = lab_w
    for label, colour in (("Triton faster", C_TRITON),
                          ("Numba faster", C_NUMBA)):
        body.append(f'<circle cx="{lx + 5}" cy="{H - 13}" r="4" '
                    f'fill="{colour}" fill-opacity="0.8"/>')
        body.append(_text(lx + 14, H - 9, label, 11, INK))
        lx += 120
    body.append(f'<rect x="{lx + 3}" y="{H - 22}" width="3" height="18" '
                 f'rx="1.5" fill="{INK_STRONG}"/>')
    body.append(_text(lx + 14, H - 9, "geometric mean", 11, INK))
    return _svg(W, H, "\n".join(body),
                "Numba versus Triton kernel time, every unit")


# ------------------------------------------------------------ grand table

def _cell_fill(s):
    """Hue by winner, opacity by |log2 speedup| (full at 4x)."""
    strength = min(abs(math.log2(s)) / 2.0, 1.0)
    return (C_TRITON if s > 1 else C_NUMBA), 0.06 + 0.74 * strength


def grand_table(doc):
    rows = [r for r in doc["rows"] if "error" not in r]
    scenes, cols = [], []
    for r in doc["rows"]:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])
        if r["experiment"] not in cols:
            cols.append(r["experiment"])
    skipped = doc.get("meta", {}).get("skipped_cells", [])
    for s in skipped:
        if s.get("row") not in scenes:
            scenes.append(s["row"])
        if s.get("column") not in cols:
            cols.append(s["column"])
    cell = {(r["scene"], r["experiment"]): r for r in rows}
    errors = {(r["scene"], r["experiment"]) for r in doc["rows"]
              if "error" in r}
    skip = {(s["row"], s["column"]) for s in skipped}

    cw, ch, lab_w, top = 40, 22, 110, 150
    W = lab_w + cw * len(cols) + 20
    H = top + ch * len(scenes) + 70
    body = [_text(0, 16, "The grand table, Numba / Triton kernel time per "
                         "cell", 13, INK_STRONG, weight="600"),
            _text(0, 33, "blue: Triton faster, red: Numba faster, deeper = "
                         "bigger gap (full at 4x)", 11, INK)]
    for j, c in enumerate(cols):
        x = lab_w + j * cw + cw / 2
        body.append(_text(x, top - 8, c, 10, INK, mono=True,
                          extra=f' transform="rotate(-55 {x:.1f} '
                                f'{top - 8:.1f})"'))
    for i, sc in enumerate(scenes):
        y = top + i * ch
        body.append(_text(lab_w - 8, y + ch / 2 + 4, sc, 10.5, INK_STRONG,
                          anchor="end", mono=True))
        for j, c in enumerate(cols):
            x = lab_w + j * cw
            key = (sc, c)
            if key in cell:
                r = cell[key]
                s = r["speedup_kernel"]
                fill, op = _cell_fill(s)
                dash = ("" if r.get("comparable", True)
                        else f' stroke="{RULE}" stroke-dasharray="2 2"')
                est = r.get("est") or r.get("config", {}).get("est")
                body.append(
                    f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" '
                    f'height="{ch - 2}" rx="3" fill="{fill}" '
                    f'fill-opacity="{op:.2f}"{dash}><title>{_esc(sc)} / '
                    f'{_esc(c)}: numba {r["numba"]["kernel_ms"]["median"]:.3g}'
                    f' ms, triton {r["triton"]["kernel_ms"]["median"]:.3g} ms'
                    f'{" (estimated per blob)" if est else ""}'
                    f'{"" if r.get("comparable", True) else " (not like-for-like)"}'
                    f'</title></rect>')
                label = (f"{s:.2f}" if s < 9.95 else f"{s:.0f}")
                body.append(_text(x + cw / 2, y + ch / 2 + 3.5,
                                  label + ("*" if est else ""), 9, INK,
                                  anchor="middle", mono=True))
            elif key in errors:
                body.append(_text(x + cw / 2, y + ch / 2 + 3.5, "err", 9,
                                  C_NUMBA, anchor="middle", mono=True))
            elif key in skip:
                body.append(_text(x + cw / 2, y + ch / 2 + 3.5, "-", 9, INK,
                                  anchor="middle", mono=True))
    yk = top + ch * len(scenes) + 22
    body.append(_text(lab_w, yk, "*  estimated per blob, the same sample on "
                                 "both sides     dashed  not like-for-like "
                                 "(different thread count or grid)     -  "
                                 "skipped on both sides, as the Numba table "
                                 "does", 10, INK))
    body.append(_text(lab_w, yk + 18, "values are numba_ms / triton_ms; "
                                      "above 1 Triton is faster", 10, INK))
    return _svg(W, H, "\n".join(body), "Grand table, Numba versus Triton")


def main():
    units = _load_units()
    figs = [("speedup_by_unit.svg", speedup_by_unit(units))]
    over = [doc for key, _, doc in units if key == "overview"]
    if over:
        figs.append(("grand_table.svg", grand_table(over[0])))
    else:
        print("no overview compare JSON yet: grand_table.svg skipped")
    for name, svg in figs:
        path = os.path.join(FIG_DIR, name)
        with open(path, "w") as f:
            f.write(svg)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
