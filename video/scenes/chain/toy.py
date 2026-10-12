"""The toy pictures of the chain cut, and the real rules run on them.

Every wave, ring count, seed, merge and run the floor shows is computed
here by the same rule the chapter kernels use, on a small grid, so the
picture is never drawn by hand.

Convention: a toy is a bool array `m[r, c]` (screen row, column). The
repo stores images as `img[x, y]` with `lin = x * height + y` and shows
them transposed, so a toy row is a repo `x` and the rules carry over
exactly: the linear index is `r * W + c`, a run is a horizontal span
(ch06: a run inside one memory row), and the ch05 seed rule looks at
the 4 of 8 neighbours with a smaller index, (r-1, c-1), (r-1, c),
(r-1, c+1) and (r, c-1) (`PDX_HOST`, `PDY_HOST` in ch05 kernels.py).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

N4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
N8 = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
PRED = ((-1, -1), (-1, 0), (-1, 1), (0, -1))      # the lex-predecessors


def neighbours(conn: int) -> tuple:
    return N4 if conn == 4 else N8


# ---- shapes -------------------------------------------------------------

def blob(n: int = 24) -> np.ndarray:
    """An organic single blob: two overlapping ellipses and a bump."""
    rr, cc = np.indices((n, n)).astype(float)
    a = ((rr - 11.5) / 8.6) ** 2 + ((cc - 10.0) / 7.8) ** 2 <= 1
    b = ((rr - 13.0) / 6.0) ** 2 + ((cc - 15.5) / 6.4) ** 2 <= 1
    c = ((rr - 4.5) / 3.0) ** 2 + ((cc - 15.0) / 3.2) ** 2 <= 1
    return a | b | c


def square(n: int = 24, side: int = 21) -> np.ndarray:
    m = np.zeros((n, n), bool)
    o = (n - side) // 2
    m[o:o + side, o:o + side] = True
    return m


def snake(n: int = 24, margin: int = 1) -> np.ndarray:
    """A one-pixel-wide serpentine: full rows joined at alternating ends,
    like the benchmark's `serpentine_256` (every level is one pixel)."""
    m = np.zeros((n, n), bool)
    lo, hi = margin, n - 1 - margin
    rows = list(range(lo, hi + 1, 2))
    for i, r in enumerate(rows):
        m[r, lo:hi + 1] = True
        if i + 1 < len(rows):
            m[r + 1, hi if i % 2 == 0 else lo] = True
    return m


def two_blobs(n: int = 24) -> np.ndarray:
    """A big blob and a small one, two pixels apart at least (ch04's asym pair)."""
    rr, cc = np.indices((n, n)).astype(float)
    big = ((rr - 12.0) / 9.5) ** 2 + ((cc - 8.5) / 7.0) ** 2 <= 1
    small = ((rr - 6.0) / 3.6) ** 2 + ((cc - 19.5) / 3.2) ** 2 <= 1
    return big | small


def many_blobs(n: int = 24) -> np.ndarray:
    """A U whose arms start as separate waves, plus a few plain blobs."""
    m = np.zeros((n, n), bool)
    # the U: two arms joined at the bottom
    m[3:16, 2:5] = True
    m[3:16, 10:13] = True
    m[13:16, 2:13] = True
    rr, cc = np.indices((n, n)).astype(float)
    m |= ((rr - 5.0) / 3.2) ** 2 + ((cc - 18.5) / 3.6) ** 2 <= 1
    m |= ((rr - 19.5) / 2.6) ** 2 + ((cc - 7.0) / 4.4) ** 2 <= 1
    m |= ((rr - 15.5) / 4.0) ** 2 + ((cc - 19.0) / 2.8) ** 2 <= 1
    return m


# ---- BFS ----------------------------------------------------------------

@dataclass
class Fill:
    """A level-synchronous BFS: `level[r, c]` (-1 = never reached) and the
    order a single serial worker would pop cells from its queue."""
    level: np.ndarray
    order: list
    queue_len: list = field(default_factory=list)   # queue length after each pop

    @property
    def n_levels(self) -> int:
        return int(self.level.max()) + 1


def bfs(m: np.ndarray, seeds, conn: int = 4) -> Fill:
    level = np.full(m.shape, -1, np.int32)
    q = deque()
    for s in seeds:
        level[s] = 0
        q.append(s)
    order, qlen = [], []
    nb = neighbours(conn)
    H, W = m.shape
    while q:
        r, c = q.popleft()
        order.append((r, c))
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and m[rr, cc] and level[rr, cc] < 0:
                level[rr, cc] = level[r, c] + 1
                q.append((rr, cc))
        qlen.append(len(q))
    return Fill(level, order, qlen)


def centre_seed(m: np.ndarray) -> tuple:
    """The red cell closest to the shape's centroid."""
    rs, cs = np.nonzero(m)
    r0, c0 = rs.mean(), cs.mean()
    i = int(np.argmin((rs - r0) ** 2 + (cs - c0) ** 2))
    return int(rs[i]), int(cs[i])


def end_seed(m: np.ndarray) -> tuple:
    """The first red cell in scan order (the snake's start)."""
    rs, cs = np.nonzero(m)
    return int(rs[0]), int(cs[0])


# ---- ch05: seeds, merges ------------------------------------------------

def candidates(m: np.ndarray) -> np.ndarray:
    """ch05's rule: a red pixel with no red lex-predecessor."""
    H, W = m.shape
    out = np.zeros_like(m)
    for r, c in zip(*np.nonzero(m)):
        if not any(0 <= r + dr < H and 0 <= c + dc < W and m[r + dr, c + dc] for dr, dc in PRED):
            out[r, c] = True
    return out


def lattice(m: np.ndarray, stride: int) -> np.ndarray:
    """Every red pixel on a stride x stride lattice (ch05's lattice twin)."""
    rr, cc = np.indices(m.shape)
    return m & (rr % stride == 0) & (cc % stride == 0)


@dataclass
class Merge:
    """seed_merge on a toy: one wave per seed, label = the seed's linear
    index, colliding waves union by min. `frames[k]` is the provisional
    label map after level k (-1 = unclaimed); `unions[k]` the label pairs
    that met during level k; `final` the canonical labels (each blob's
    minimum linear index, exactly what ch05's flatten produces)."""
    seeds: list
    frames: list
    unions: list
    final: np.ndarray


def merge_fill(m: np.ndarray, seeds, conn: int = 8) -> Merge:
    H, W = m.shape
    lab = np.full(m.shape, -1, np.int64)
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seeds = sorted(set(map(tuple, seeds)), key=lambda s: s[0] * W + s[1])
    front = []
    for r, c in seeds:
        lab[r, c] = r * W + c
        parent[r * W + c] = r * W + c
        front.append((r, c))
    frames, unions = [lab.copy()], [[]]
    nb = neighbours(conn)
    while front:
        nxt, met = [], []
        for r, c in front:
            for dr, dc in nb:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W and m[rr, cc]):
                    continue
                if lab[rr, cc] < 0:
                    lab[rr, cc] = lab[r, c]
                    nxt.append((rr, cc))
                else:
                    a, b = find(int(lab[r, c])), find(int(lab[rr, cc]))
                    if a != b:
                        lo, hi = min(a, b), max(a, b)
                        parent[hi] = lo
                        met.append((hi, lo))
        front = nxt
        if front or met:
            frames.append(lab.copy())
            unions.append(met)
    final = np.full(m.shape, -1, np.int64)
    for r, c in zip(*np.nonzero(m)):
        final[r, c] = find(int(lab[r, c]))
    return Merge(seeds, frames, unions, final)


def canonical(m: np.ndarray, conn: int = 8) -> np.ndarray:
    """Reference labels: each blob's minimum linear index (-1 off the mask)."""
    H, W = m.shape
    out = np.full(m.shape, -1, np.int64)
    for r, c in zip(*np.nonzero(m)):
        if out[r, c] >= 0:
            continue
        f = bfs(m, [(r, c)], conn)
        out[f.level >= 0] = r * W + c       # scan order: (r, c) is the blob's minimum
    return out


# ---- ch06: runs ---------------------------------------------------------

@dataclass(frozen=True)
class Run:
    r: int
    c0: int
    c1: int          # inclusive


def runs(m: np.ndarray) -> list:
    """Maximal red spans of every row, in scan order (ch06's run ids)."""
    out = []
    for r in range(m.shape[0]):
        row = m[r]
        c = 0
        while c < row.size:
            if row[c]:
                c0 = c
                while c + 1 < row.size and row[c + 1]:
                    c += 1
                out.append(Run(r, c0, c))
            c += 1
    return out


def run_labels(m: np.ndarray) -> tuple[list, list]:
    """ch06's merge on runs: two runs in adjacent rows touch when their
    column intervals overlap after widening by one (8-connectivity). Union
    by min run id; the canonical label of a run is its blob's minimum
    linear index, the first pixel of its smallest run. Returns (label per
    run, the touching pairs)."""
    rs = runs(m)
    parent = list(range(len(rs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for i, a in enumerate(rs):
        for j in range(i + 1, len(rs)):
            b = rs[j]
            if b.r > a.r + 1:
                break
            if b.r == a.r + 1 and b.c0 <= a.c1 + 1 and a.c0 <= b.c1 + 1:
                pairs.append((i, j))
                x, y = find(i), find(j)
                if x != y:
                    parent[max(x, y)] = min(x, y)
    W = m.shape[1]
    labels = [rs[find(i)].r * W + rs[find(i)].c0 for i in range(len(rs))]
    return labels, pairs


def paint_runs(m: np.ndarray) -> np.ndarray:
    """The label map that painting every run with its label gives."""
    out = np.full(m.shape, -1, np.int64)
    rs = runs(m)
    labels, _ = run_labels(m)
    for run, lab in zip(rs, labels):
        out[run.r, run.c0:run.c1 + 1] = lab
    return out


def selftest() -> None:
    """The toy rules agree with each other and with the facts the video states."""
    sq = square()
    seed = centre_seed(sq)
    n4, n8 = bfs(sq, [seed], 4).n_levels, bfs(sq, [seed], 8).n_levels
    assert (n4, n8) == (21, 11), (n4, n8)                  # 8-conn halves a square's rings
    sn = snake()
    f = bfs(sn, [end_seed(sn)], 4)
    assert f.n_levels == int(sn.sum()), "every snake level is one pixel"
    assert np.bincount(f.level[f.level >= 0]).max() == 1
    for shape in (blob(), two_blobs(), many_blobs(), sq, sn):
        ref = canonical(shape)
        assert np.array_equal(paint_runs(shape), ref), "runs + min merge == canonical labels"
        seeds = list(zip(*np.nonzero(candidates(shape) | lattice(shape, 4))))
        assert np.array_equal(merge_fill(shape, seeds).final, ref), "seed_merge == canonical labels"
        # every blob's minimum pixel is a candidate (at least one seed per blob)
        H, W = shape.shape
        mins = {int(v) for v in np.unique(ref[ref >= 0])}
        cand = {r * W + c for r, c in zip(*np.nonzero(candidates(shape)))}
        assert mins <= cand
    ta = two_blobs()
    lab = canonical(ta)
    assert len(np.unique(lab[lab >= 0])) == 2
    mb = many_blobs()
    lab = canonical(mb)
    assert len(np.unique(lab[lab >= 0])) == 4, len(np.unique(lab[lab >= 0]))
    m = merge_fill(mb, list(zip(*np.nonzero(candidates(mb)))))
    assert any(m.unions), "the U's arms must meet"


if __name__ == "__main__":
    selftest()
    for name, shape in (("blob", blob()), ("square", square()), ("snake", snake()),
                        ("two_blobs", two_blobs()), ("many_blobs", many_blobs())):
        lab = canonical(shape)
        seeds = candidates(shape)
        print(f"{name:10s} red {int(shape.sum()):4d}  blobs {len(np.unique(lab[lab >= 0]))}  "
              f"4-conn rings from the centre {bfs(shape, [centre_seed(shape)], 4).n_levels:3d}  "
              f"8-conn {bfs(shape, [centre_seed(shape)], 8).n_levels:3d}  "
              f"seeds {int(seeds.sum()):2d}  runs {len(runs(shape)):3d}")
    print("selftest OK")
