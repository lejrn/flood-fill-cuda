"""Host driver for ch09: BFS on every CPU core.

Public API:

    PixelBFS(width, height, connectivity=8, threads=None, ...)
    TileBFS(width, height, connectivity=8, threads=None, ...)
        .run(img, seed_x, seed_y, paint=True)  -> stats dict
        .visited()                             -> bool (W, H) (off clock)
    flood_fill(img, seed_x, seed_y, connectivity=8, method="tile",
               threads=None, **engine_kw)      -> FillResult

An engine owns its buffers for one image shape (allocated once, as the
GPU sides allocate theirs) and resets them inside run(), on the clock,
as ch07 and ch08 initialise theirs inside their timed pipelines. run()
paints the image it is given in place; flood_fill() and label() copy
it first.

threads: numba.set_num_threads for the duration of the call (None: the
current setting, by default every logical CPU). A prange splits its
iterations over that many threads.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import numba
import numpy as np

from . import kernels as K

CONNECTIVITIES = (4, 8)

# pixel BFS: a level below SERIAL_BELOW entries runs on the calling
# thread; above it, chunks of at least MIN_CHUNK entries (up to four per
# thread). Untuned defaults, recorded by the benchmark.
PIXEL_SERIAL_BELOW = 1024
PIXEL_MIN_CHUNK = 256
FRONT_START = 1 << 16
# tile BFS: a round below TILE_SERIAL_BELOW tiles runs on the calling
# thread (one tile visit is ~ a microsecond, a fork and barrier more)
TILE_SERIAL_BELOW = 8


@contextmanager
def num_threads(n):
    """numba.set_num_threads(n) inside the block (None: unchanged)."""
    if n is None:
        yield numba.get_num_threads()
        return
    n = int(n)
    if not 1 <= n <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError(
            f"threads must be in [1, {numba.config.NUMBA_NUM_THREADS}], "
            f"got {n}")
    old = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        yield n
    finally:
        numba.set_num_threads(old)


def _check_shape(width, height):
    if width < 1 or height < 1:
        raise ValueError(f"bad shape {width}x{height}")
    if width * height >= 2 ** 31:
        raise ValueError(
            f"{width}x{height} has 2**31 pixels or more: labels and "
            "anchors are int32 (x * height + y), as ch05's are")


def _check_conn(connectivity):
    if connectivity not in CONNECTIVITIES:
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity!r}")


def check_image(img, width=None, height=None):
    if (not isinstance(img, np.ndarray) or img.ndim != 3
            or img.shape[2] != 3 or img.dtype != np.uint8):
        raise ValueError("img must be a (width, height, 3) uint8 array")
    if not img.flags.c_contiguous:
        raise ValueError("img must be C-contiguous (np.ascontiguousarray)")
    if width is not None and img.shape[:2] != (width, height):
        raise ValueError(
            f"img is {img.shape[0]}x{img.shape[1]}, the engine was built "
            f"for {width}x{height}")


def check_seed(img, seed_x, seed_y):
    seed_x, seed_y = int(seed_x), int(seed_y)
    w, h = img.shape[0], img.shape[1]
    if not (0 <= seed_x < w and 0 <= seed_y < h):
        raise ValueError(f"seed ({seed_x}, {seed_y}) outside {w}x{h} image")
    if not (img[seed_x, seed_y, 0] == 255 and img[seed_x, seed_y, 1] == 0
            and img[seed_x, seed_y, 2] == 0):
        raise ValueError(
            f"seed pixel ({seed_x}, {seed_y}) is not red - nothing to fill")
    return seed_x, seed_y


class PixelBFS:
    """ch03's level-synchronous pixel BFS on every core (one blob).

    Buffers: visited (W x H uint8 flags, also the race stamps), the
    frontier and the chunks' discovery regions (grown when a level needs
    more). run() resets visited, fills, and paints if asked."""

    method = "pixel"

    def __init__(self, width, height, connectivity=8, threads=None,
                 serial_below=PIXEL_SERIAL_BELOW, min_chunk=PIXEL_MIN_CHUNK):
        _check_shape(width, height)
        _check_conn(connectivity)
        if serial_below < 1 or min_chunk < 1:
            raise ValueError("serial_below and min_chunk must be >= 1")
        self.width, self.height = int(width), int(height)
        self.connectivity = connectivity
        self.threads = threads
        self.serial_below = int(serial_below)
        self.min_chunk = int(min_chunk)
        self._visited = np.zeros((self.width, self.height), np.uint8)
        n0 = min(FRONT_START, self.width * self.height)
        self._front = np.empty(max(n0, 1), np.int64)
        self._raw = np.empty(max(8 * n0, 8), np.int64)

    def run(self, img, seed_x, seed_y, paint=True):
        check_image(img, self.width, self.height)
        sx, sy = check_seed(img, seed_x, seed_y)
        with num_threads(self.threads) as nt:
            K.reset_u8(self._visited)
            (levels, filled, dups, peak, par, self._front,
             self._raw) = K.pixel_bfs(img, self._visited, sx, sy,
                                      self.connectivity == 8, self._front,
                                      self._raw, self.min_chunk,
                                      self.serial_below, nt)
            if paint:
                K.paint_visited(img, self._visited)
        return {"levels": int(levels), "filled": int(filled),
                "duplicates": int(dups), "peak_frontier": int(peak),
                "parallel_levels": int(par), "threads": int(nt)}

    def visited(self):
        """bool (W, H): the reached set of the last run (off clock)."""
        return self._visited != 0


class TileBFS:
    """ch07's tile BFS on every core (one blob): 64 x 64 tiles of
    uint64 row words, a frontier of tiles, a prange per round.

    run() packs the RGB image to the tile-major mask (rows in parallel),
    resets the per-tile state, runs the rounds and paints the reached
    words (rows in parallel), all on the clock."""

    method = "tile"

    def __init__(self, width, height, connectivity=8, threads=None,
                 serial_below=TILE_SERIAL_BELOW):
        _check_shape(width, height)
        _check_conn(connectivity)
        if serial_below < 1:
            raise ValueError("serial_below must be >= 1")
        self.width, self.height = int(width), int(height)
        self.connectivity = connectivity
        self.threads = threads
        self.serial_below = int(serial_below)
        self.tiles_x = (self.width + K.TILE - 1) // K.TILE
        self.wpr = (self.height + K.TILE - 1) // K.TILE
        nt = self.n_tiles = self.tiles_x * self.wpr
        self.mask = np.zeros((nt, K.TILE), np.uint64)
        self.colm = np.zeros((nt, 2), np.uint64)
        self.reach = np.zeros((nt, K.TILE), np.uint64)
        self.inbox = np.zeros((2, nt, 8), np.uint64)
        self.known = np.zeros((nt, 8), np.uint64)
        self.sent = np.zeros((nt, 8), np.uint64)
        self.tdirty = np.zeros(nt, np.uint64)
        self.tlo = np.zeros(nt, np.int16)
        self.thi = np.zeros(nt, np.int16)
        self.queued = np.zeros(nt, np.int32)
        self.front = np.zeros(nt, np.int64)
        self.out = np.zeros(8 * nt, np.int64)

    def run(self, img, seed_x, seed_y, paint=True):
        check_image(img, self.width, self.height)
        sx, sy = check_seed(img, seed_x, seed_y)
        with num_threads(self.threads) as nt:
            K.pack_tiles(img, self.mask, self.wpr)
            K.tile_columns(self.mask, self.colm)
            K.reset_tile_state(self.reach, self.inbox, self.known, self.sent,
                               self.tdirty, self.tlo, self.thi, self.queued)
            rounds, peak, par, tot = K.tile_bfs(
                self.mask, self.colm, self.reach, self.inbox, self.known,
                self.sent, self.tdirty, self.tlo, self.thi, self.queued,
                self.front, self.out, self.connectivity == 8, self.tiles_x,
                self.wpr, sx, sy, self.serial_below, nt)
            if paint:
                K.paint_tiles(img, self.reach, self.wpr)
        return {"rounds": int(rounds), "peak_round_tiles": int(peak),
                "parallel_rounds": int(par),
                "tile_visits": int(tot[K.TS_VISITS]),
                "idle_visits": int(tot[K.TS_IDLE]),
                "closure_passes": int(tot[K.TS_PASSES]),
                "sends": int(tot[K.TS_SENDS]), "n_tiles": self.n_tiles,
                "threads": int(nt)}

    def filled(self):
        """Reached pixels of the last run (off clock)."""
        return int(K.popcount_words(self.reach))

    def visited(self):
        """bool (W, H): the reached set of the last run (off clock)."""
        out = np.empty((self.width, self.height), np.bool_)
        K.unpack_tiles(self.reach, self.wpr, out)
        return out


@dataclass
class FillResult:
    img: np.ndarray        # (W, H, 3) uint8: a copy, the blob painted blue
    visited: np.ndarray    # (W, H) int32, 1 where reached (ch07's form)
    filled: int
    steps: int             # levels (pixel BFS) or rounds (tile BFS)
    method: str
    connectivity: int
    threads: int
    stats: dict = field(repr=False)


ENGINES = {"pixel": PixelBFS, "tile": TileBFS}


def flood_fill(img, seed_x, seed_y, connectivity=8, method="tile",
               threads=None, **engine_kw):
    """Flood-fill the red blob holding (seed_x, seed_y) on every core.

    img: (width, height, 3) uint8, any memory layout; not modified (a
    painted copy is returned). Raises ValueError for bad inputs,
    including a seed that is not red (like ch03 and ch07)."""
    if method not in ENGINES:
        raise ValueError(
            f"method must be one of {tuple(ENGINES)}, got {method!r}")
    if (not isinstance(img, np.ndarray) or img.ndim != 3
            or img.shape[2] != 3 or img.dtype != np.uint8):
        raise ValueError("img must be a (width, height, 3) uint8 array")
    out = np.ascontiguousarray(img).copy()
    check_seed(out, seed_x, seed_y)
    eng = ENGINES[method](out.shape[0], out.shape[1],
                          connectivity=connectivity, threads=threads,
                          **engine_kw)
    st = eng.run(out, seed_x, seed_y, paint=True)
    visited = eng.visited().astype(np.int32)
    if method == "pixel":
        steps, filled = st["levels"], st["filled"]
    else:
        steps, filled = st["rounds"], eng.filled()
    return FillResult(img=out, visited=visited, filled=filled,
                      steps=steps, method=method,
                      connectivity=connectivity, threads=st["threads"],
                      stats=st)
