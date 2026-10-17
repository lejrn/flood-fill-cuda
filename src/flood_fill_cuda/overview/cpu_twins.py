"""bench_v5's CPU twins: the chapters' @njit oracles on buffers kept
between calls.

The oracles (ch01's cpu_flood_fill, the shared cpu_flood_fill_8, ch05's
cpu_label_components) allocate their outputs and their queue on every
call, so the first touch of every page lands on their clock. On the
64 Mpx square that was about 30% of the 4-conn BFS's time (a scratch
run: 1,547 ms as the oracle, about 1,100 ms on reused buffers). The GPU
sides bench_v5 compares them with allocate their buffers once. A twin
here runs its oracle's loop verbatim on buffers allocated once per
image, and resets them where its GPU side resets its own:

  before the clock   ch01-ch05 and act 2's estimates reset on the device
                     between runs, off their clocks
  on the clock       ch07 and ch08 initialise their state inside the
                     timed pipeline (ch07's init clears its reached bits)

The loops:

  bfs4_into(img, sx, sy, visited, depth, queue)    ch01's 4-conn BFS,
      visited and depth (cpu_flood_fill's loop)
  bfs8_into(img, sx, sy, visited, depth, queue)    the shared 8-conn BFS,
      visited and depth (cpu_flood_fill_8's loop)
  fill8_into(img, sx, sy, visited, queue)          the same 8-conn BFS
      without the depth map: visited only. ch07 keeps no per-pixel depth
  label_into(img, label, queue)                    ch05's labels only:
      canonical labels, 8-conn (cpu_label_components' loop)

Each expects its buffers reset (visited 0, depth -1, label -1) and
returns what its oracle returns besides the arrays: (levels, filled) for
the BFS loops, n_blobs for the labels. Twin binds one loop to its
buffers and its reset scope; check_against_oracle compares a twin's
output with its oracle's, array for array (bench_v5 runs it untimed on
every scene).
"""

import numpy as np
from numba import njit

from ..chapters.ch01_gpu_1blob_1block.cpu_oracle import (
    _DX as _DX4, _DY as _DY4, cpu_flood_fill)
from ..shared.cpu_oracle import _DX as _DX8, _DY as _DY8, cpu_flood_fill_8
from ..chapters.ch05_gpu_nblob_nblock.cpu_oracle import (
    _DX as _DXL, _DY as _DYL, _is_red, cpu_label_components)


@njit(cache=True)
def reset_i32(a, value):
    """a[...] = value, for a C-contiguous int32 array of any shape."""
    flat = a.reshape(a.size)
    for i in range(flat.size):
        flat[i] = value


@njit(cache=True)
def bfs4_into(img, seed_x, seed_y, visited, depth, queue):
    """cpu_flood_fill's loop (ch01, 4-conn) on given buffers."""
    width, height = img.shape[0], img.shape[1]
    visited[seed_x, seed_y] = 1
    queue[0] = seed_x * height + seed_y
    front, rear = 0, 1
    level = 0
    while front < rear:
        level_end = rear
        for i in range(front, level_end):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            depth[x, y] = level
            for d in range(4):
                nx = x + _DX4[d]
                ny = y + _DY4[d]
                if 0 <= nx < width and 0 <= ny < height and visited[nx, ny] == 0:
                    if img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0 and img[nx, ny, 2] == 0:
                        visited[nx, ny] = 1
                        queue[rear] = nx * height + ny
                        rear += 1
        front = level_end
        level += 1
    return level, rear


@njit(cache=True)
def bfs8_into(img, seed_x, seed_y, visited, depth, queue):
    """cpu_flood_fill_8's loop (shared, 8-conn) on given buffers."""
    width, height = img.shape[0], img.shape[1]
    visited[seed_x, seed_y] = 1
    queue[0] = seed_x * height + seed_y
    front, rear = 0, 1
    level = 0
    while front < rear:
        level_end = rear
        for i in range(front, level_end):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            depth[x, y] = level
            for d in range(8):
                nx = x + _DX8[d]
                ny = y + _DY8[d]
                if 0 <= nx < width and 0 <= ny < height and visited[nx, ny] == 0:
                    if img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0 and img[nx, ny, 2] == 0:
                        visited[nx, ny] = 1
                        queue[rear] = nx * height + ny
                        rear += 1
        front = level_end
        level += 1
    return level, rear


@njit(cache=True)
def fill8_into(img, seed_x, seed_y, visited, queue):
    """cpu_flood_fill_8's loop without the depth map: visited only."""
    width, height = img.shape[0], img.shape[1]
    visited[seed_x, seed_y] = 1
    queue[0] = seed_x * height + seed_y
    front, rear = 0, 1
    level = 0
    while front < rear:
        level_end = rear
        for i in range(front, level_end):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            for d in range(8):
                nx = x + _DX8[d]
                ny = y + _DY8[d]
                if 0 <= nx < width and 0 <= ny < height and visited[nx, ny] == 0:
                    if img[nx, ny, 0] == 255 and img[nx, ny, 1] == 0 and img[nx, ny, 2] == 0:
                        visited[nx, ny] = 1
                        queue[rear] = nx * height + ny
                        rear += 1
        front = level_end
        level += 1
    return level, rear


@njit(cache=True)
def label_into(img, label, queue):
    """cpu_label_components' loop (ch05, 8-conn, canonical labels) on
    given buffers; label must be -1 everywhere. Returns n_blobs."""
    width, height = img.shape[0], img.shape[1]
    n_blobs = 0
    for x in range(width):
        for y in range(height):
            if not _is_red(img, x, y) or label[x, y] >= 0:
                continue
            n_blobs += 1
            root = x * height + y
            label[x, y] = root
            queue[0] = root
            front, rear = 0, 1
            while front < rear:
                pixel = queue[front]
                front += 1
                px = pixel // height
                py = pixel % height
                for d in range(8):
                    nx = px + _DXL[d]
                    ny = py + _DYL[d]
                    if (0 <= nx < width and 0 <= ny < height
                            and label[nx, ny] < 0 and _is_red(img, nx, ny)):
                        label[nx, ny] = root
                        queue[rear] = nx * height + ny
                        rear += 1
    return n_blobs


# key -> (loop, outputs, connectivity, oracle)
KINDS = {
    "bfs4": (bfs4_into, ("visited", "depth"), 4, "cpu_flood_fill (ch01)"),
    "bfs8": (bfs8_into, ("visited", "depth"), 8, "cpu_flood_fill_8 (shared)"),
    "fill8": (fill8_into, ("visited",), 8,
              "cpu_flood_fill_8 (shared) without its depth map"),
    "label": (label_into, ("labels",), 8, "cpu_label_components (ch05)"),
}


class Buffers:
    """One image shape's CPU buffers, allocated once (and touched once by
    the untimed check) and shared by every twin of that shape: the twins
    run one at a time, and each resets what it reads before it runs.
    Only the arrays the given kinds use are allocated."""

    def __init__(self, width, height, kinds):
        kinds = set(kinds)
        bfs = kinds & {"bfs4", "bfs8", "fill8"}
        self.visited = np.zeros((width, height), np.int32) if bfs else None
        self.depth = (np.full((width, height), -1, np.int32)
                      if kinds & {"bfs4", "bfs8"} else None)
        self.label = (np.full((width, height), -1, np.int32)
                      if "label" in kinds else None)
        self.queue = np.zeros(width * height, np.int64)


class Twin:
    """One CPU twin on one image. prepare() runs before the clock and
    call() on it; the reset sits in one or the other (reset_on_clock).
    call() returns (levels, filled) for a BFS, n_blobs for the labels."""

    def __init__(self, kind, img, seed, bufs, reset_on_clock):
        if kind not in KINDS:
            raise KeyError(kind)
        self.kind = kind
        self.loop, self.outputs, self.connectivity, self.oracle = KINDS[kind]
        self.img, self.bufs = img, bufs
        self.seed = (None if seed is None
                     else (int(seed[0]), int(seed[1])))
        self.reset_on_clock = bool(reset_on_clock)

    def _reset(self):
        b = self.bufs
        if self.kind == "label":
            reset_i32(b.label, -1)
            return
        reset_i32(b.visited, 0)
        if self.kind != "fill8":
            reset_i32(b.depth, -1)

    def _run(self):
        b, img = self.bufs, self.img
        if self.kind == "label":
            return label_into(img, b.label, b.queue)
        sx, sy = self.seed
        if self.kind == "fill8":
            return fill8_into(img, sx, sy, b.visited, b.queue)
        return self.loop(img, sx, sy, b.visited, b.depth, b.queue)

    def prepare(self):
        if not self.reset_on_clock:
            self._reset()

    def call(self):
        if self.reset_on_clock:
            self._reset()
        return self._run()


def check_against_oracle(twin):
    """Run the twin once (prepare + call) and its oracle once, untimed;
    every array the twin keeps and every count must be equal. Returns
    (ok, the twin's result)."""
    twin.prepare()
    res = twin.call()
    b, img = twin.bufs, twin.img
    if twin.kind == "label":
        lab, nb = cpu_label_components(img)
        ok = bool(np.array_equal(lab, b.label)) and int(nb) == int(res)
        del lab
        return ok, res
    sx, sy = twin.seed
    if twin.kind == "bfs4":
        v, d, lv, f = cpu_flood_fill(img, sx, sy)
    else:
        v, d, lv, f = cpu_flood_fill_8(img, sx, sy)
    ok = (bool(np.array_equal(v, b.visited)) and int(lv) == int(res[0])
          and int(f) == int(res[1]))
    if twin.kind != "fill8":
        ok = ok and bool(np.array_equal(d, b.depth))
    del v, d
    return ok, res
