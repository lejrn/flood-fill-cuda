"""Put the page's code snippets into index.html, straight from the kernels.

A snippet is a list of lines that must appear in its source file, in
order, after an anchor line. The script fails if one of them is missing,
so a snippet on the page cannot drift from the code: when a kernel
changes, the build says which line went.

Lines between the listed ones are left out (accounting, comments). The
snippet keeps the source's own indentation, less the common indent.

Each snippet lands in index.html between a pair of markers:

    <!-- snippet:row_fill:start --> ... <!-- snippet:row_fill:end -->

highlighted by Pygments at build time (the page loads no highlighter),
with a link to the first line in the source.

Run from the repository root:

    .venv/bin/python project-page/tools/build_snippets.py
"""

from __future__ import annotations

import html
import re
import sys
import textwrap
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

PAGE = Path(__file__).resolve().parents[1]
REPO = PAGE.parent
CH = "src/flood_fill_cuda/chapters"
PAGES = [p for p in (PAGE / "index.html", PAGE / "full.html") if p.exists()]

# name -> (source file, anchor line, the lines to show, in order)
SNIPPETS = {
    "bfs_level": (
        f"{CH}/ch01_gpu_1blob_1block/kernels.py",
        "    while front < rear:",
        """
    while front < rear:
        for i in range(front + tid, rear, nthreads):
            pixel = ring[i & RING_MASK]
            x = pixel // height
            y = pixel % height
            depth[x, y] = level
            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        ticket = cuda.atomic.add(s_rear, 0, 1)
                        if ticket - front < RING_CAPACITY:
                            ring[ticket & RING_MASK] = nx * height + ny
        cuda.syncthreads()  # all enqueues + final s_rear for this level visible
        level += 1
""",
    ),
    "union": (
        f"{CH}/ch05_gpu_nblob_nblock/kernels.py",
        "def _union(parent, a, b):",
        """
    while True:
        a = _find(parent, a)
        b = _find(parent, b)
        if a == b:
            return 0
        if a < b:
            a, b = b, a
        old = cuda.atomic.min(parent, a, b)
        if old == a:
            return 1
        a = old
""",
    ),
    "row_fill": (
        f"{CH}/ch07_gpu_1blob_tiles/kernels.py",
        "def _row_fill(f, m):",
        """
    g = f
    p = m
    g = uint32(g | (p & (g << S1)))
    p = uint32(p & (p << S1))
    g = uint32(g | (p & (g << S2)))
    p = uint32(p & (p << S2))
    g = uint32(g | (p & (g << S4)))
    p = uint32(p & (p << S4))
    g = uint32(g | (p & (g << S8)))
    p = uint32(p & (p << S8))
    g = uint32(g | (p & (g << S16)))
""",
    ),
    "prange": (
        f"{CH}/ch09_cpu_nthreads/kernels.py",
        "def pixel_bfs(",
        """
            for c in prange(nch):
                a = n * c // nch
                b = n * (c + 1) // nch
                r = _expand_pixels(img, visited, front, a, b, raw, k * a,
                                   c + 1, conn8)
                counts[c] = r[0]
""",
    ),
}


def locate(name: str) -> tuple[str, int, list[str]]:
    """The snippet's path, the 1-based line of its first line, its lines."""
    path, anchor, wanted = SNIPPETS[name]
    source = (REPO / path).read_text(encoding="utf-8").splitlines()
    try:
        at = next(i for i, line in enumerate(source) if line.startswith(anchor))
    except StopIteration:
        sys.exit(f"snippet {name}: anchor {anchor!r} is not in {path}")
    first = None
    for line in wanted.strip("\n").splitlines():
        try:
            at = next(i for i in range(at, len(source)) if source[i] == line)
        except StopIteration:
            sys.exit(f"snippet {name}: this line is no longer in {path} after the anchor:\n{line}")
        first = at if first is None else first
        at += 1
    return path, first + 1, textwrap.dedent(wanted.strip("\n")).splitlines()


def render(name: str) -> str:
    path, line, lines = locate(name)
    code = highlight("\n".join(lines), PythonLexer(), HtmlFormatter(nowrap=True)).rstrip("\n")
    short = "/".join(path.split("/")[-2:])
    return (f'<figure class="code-block">\n<pre><code>{code}</code></pre>\n'
            f'<figcaption><a href="../{html.escape(path)}" data-repo-path="{html.escape(path)}#L{line}">'
            f'{html.escape(short)}, line {line}</a></figcaption>\n</figure>')


def main() -> None:
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        done = []
        for name in SNIPPETS:
            pattern = re.compile(rf"(<!-- snippet:{name}:start -->)(.*?)(<!-- snippet:{name}:end -->)", re.S)
            if not pattern.search(text):
                continue
            block = render(name)
            text = pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", text)
            done.append(name)
        page.write_text(text, encoding="utf-8")
        print(f"{page.name}: snippets {', '.join(done) or 'none'}")


if __name__ == "__main__":
    for name in SNIPPETS:
        path, line, lines = locate(name)
        print(f"{name}: {path}:{line}, {len(lines)} lines")
    main()
