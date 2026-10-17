"""Host driver for ch09: BFS on every CPU core.

Public API:

    PixelBFS(width, height, connectivity=8, threads=None, ...)
    TileBFS(width, height, connectivity=8, threads=None, ...)
        .run(img, seed_x, seed_y, paint=True)  -> stats dict
        .visited()                             -> bool (W, H) (off clock)
    TileLabels(width, height, connectivity=8, lattice=8, threads=None,
               ...)
        .run(img, paint=True)                  -> stats dict
        .label_map()                           -> int32 (W, H), -1 off
                                                  blob (off clock)
    flood_fill(img, seed_x, seed_y, connectivity=8, method="tile",
               threads=None, **engine_kw)      -> FillResult
    label(img, connectivity=8, lattice=8, threads=None, **engine_kw)
                                               -> LabelResult

An engine owns its per-pixel and per-tile buffers for one image shape
(allocated once, as the GPU sides allocate theirs) and resets them
inside run(), on the clock, as ch07 and ch08 initialise theirs inside
their timed pipelines. Two exceptions, both on the clock: the pixel
BFS grows its frontier buffers when a level needs more (the engine
keeps them, so only a first run grows them), and the wave rounds of
TileLabels allocate their per-round scratch with np.empty (no zeroing;
ch08 allocates nothing in run()). run() paints the image it is given
in place; flood_fill() and label() copy it first.

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
# thread; above it, chunks of at least MIN_CHUNK entries (at most one
# per thread: kernels.chunks_per_round). Untuned defaults, recorded by
# the benchmark.
PIXEL_SERIAL_BELOW = 1024
PIXEL_MIN_CHUNK = 256
FRONT_START = 1 << 16
# tile BFS: a round below TILE_SERIAL_BELOW tiles runs on the calling
# thread (one tile visit is ~ a microsecond, a fork and barrier more)
TILE_SERIAL_BELOW = 8
# tile labels: a wave round below LABEL_SERIAL_BELOW entries runs on the
# calling thread; the chunks' claim and meeting regions hold this many
# per entry (plus one worst-case expansion each; past that an entry is
# requeued); ch08's seed lattice
LABEL_SERIAL_BELOW = 64
CLAIMS_PER_ENTRY = 8
MEETS_PER_ENTRY = 4
DEFAULT_LATTICE = 8


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
            (levels, filled, dups, peak, par, pch, self._front,
             self._raw) = K.pixel_bfs(img, self._visited, sx, sy,
                                      self.connectivity == 8, self._front,
                                      self._raw, self.min_chunk,
                                      self.serial_below, nt)
            if paint:
                K.paint_visited(img, self._visited)
        return {"levels": int(levels), "filled": int(filled),
                "duplicates": int(dups), "peak_frontier": int(peak),
                "parallel_levels": int(par), "peak_chunks": int(pch),
                "threads": int(nt)}

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
            rounds, peak, par, pch, tot = K.tile_bfs(
                self.mask, self.colm, self.reach, self.inbox, self.known,
                self.sent, self.tdirty, self.tlo, self.thi, self.queued,
                self.front, self.out, self.connectivity == 8, self.tiles_x,
                self.wpr, sx, sy, self.serial_below, nt)
            if paint:
                K.paint_tiles(img, self.reach, self.wpr)
        return {"rounds": int(rounds), "peak_round_tiles": int(peak),
                "parallel_rounds": int(par), "peak_chunks": int(pch),
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


class TileLabels:
    """ch08's tile waves on every core (many blobs, no seeds given):
    canonical labels (the blob's min x * height + y) and the palette
    paint, 4- or 8-connected.

    run() packs, splits every tile into pieces, seeds, runs the waves,
    unions the meetings and paints, all on the clock; the per-pixel
    label map is an off-clock twin of the paint (label_map())."""

    method = "tile_labels"

    def __init__(self, width, height, connectivity=8,
                 lattice=DEFAULT_LATTICE, threads=None,
                 serial_below=LABEL_SERIAL_BELOW,
                 claims_per_entry=CLAIMS_PER_ENTRY,
                 meets_per_entry=MEETS_PER_ENTRY):
        _check_shape(width, height)
        _check_conn(connectivity)
        if lattice < 0:
            raise ValueError(f"lattice must be >= 0, got {lattice}")
        if serial_below < 1 or claims_per_entry < 0 or meets_per_entry < 0:
            raise ValueError("serial_below >= 1, per-entry sizes >= 0")
        self.width, self.height = int(width), int(height)
        self.connectivity = connectivity
        self.lattice = int(lattice)
        self.threads = threads
        self.serial_below = int(serial_below)
        self.claims_per_entry = int(claims_per_entry)
        self.meets_per_entry = int(meets_per_entry)
        self.tiles_x = (self.width + K.TILE - 1) // K.TILE
        self.wpr = (self.height + K.TILE - 1) // K.TILE
        nt = self.n_tiles = self.tiles_x * self.wpr
        self.mask = np.zeros((nt, K.TILE), np.uint64)
        self.colm = np.zeros((nt, 2), np.uint64)
        self.n_ring = np.zeros(nt, np.int32)
        self.n_int = np.zeros(nt, np.int32)
        self.n_seed = np.zeros(nt, np.int32)
        self.seed_off = np.zeros(nt, np.int64)
        slots = nt * K.RING_CAP
        self.anchor = np.zeros(slots, np.int32)
        self.lab = np.full(slots, -1, np.int32)
        self.claimer = np.zeros(slots, np.uint8)
        self.dirs = np.zeros(slots, np.uint8)
        self.fmap = np.full((nt, 4, K.TILE), K.NO_PIECE, np.uint8)
        self._seed_cap = 0
        self._ensure_seeds(1024)
        self.meet_a = np.empty(1024, np.int64)
        self.meet_b = np.empty(1024, np.int64)
        self._dummy_img = np.zeros((1, 1, 3), np.uint8)
        self._dummy_lab = np.zeros((1, 1), np.int32)
        self.n_blobs = 0

    def _ensure_seeds(self, n):
        if n <= self._seed_cap:
            return
        cap = max(n, 2 * self._seed_cap)
        self.seed_slot = np.empty(cap, np.int64)
        self.seed_anchor = np.empty(cap, np.int32)
        self.parent = np.empty(cap, np.int64)
        self.seed_label = np.empty(cap, np.int32)
        self.ent_a = np.empty(cap, np.int64)
        self.ent_w = np.empty(cap, np.int64)
        self._seed_cap = cap

    def run(self, img, paint=True):
        check_image(img, self.width, self.height)
        conn8 = self.connectivity == 8
        with num_threads(self.threads) as nt:
            K.pack_tiles(img, self.mask, self.wpr)
            K.tile_columns(self.mask, self.colm)
            K.label_pieces(self.mask, self.colm, self.wpr, self.tiles_x,
                           self.width, self.height, conn8, self.lattice,
                           self.n_ring, self.n_int, self.n_seed, self.anchor,
                           self.lab, self.claimer, self.dirs, self.fmap, nt)
            n_seeds, n_int = K.seed_offsets(self.n_seed, self.n_int,
                                            self.seed_off)
            self._ensure_seeds(n_seeds)
            K.assign_seeds(self.n_ring, self.seed_off, self.lab, self.anchor,
                           self.seed_slot, self.seed_anchor, self.parent,
                           self.ent_a, self.ent_w)
            (rounds, peak, par, pch, n_meet, tots, self.meet_a,
             self.meet_b) = K.label_waves(
                self.ent_a, self.ent_w, n_seeds, self.lab, self.claimer,
                self.dirs, self.fmap, self.wpr, conn8, self.meet_a,
                self.meet_b, self.claims_per_entry, self.meets_per_entry,
                self.serial_below, nt)
            roots = K.resolve_labels(self.parent, self.seed_anchor,
                                     self.seed_label, n_seeds, self.meet_a,
                                     self.meet_b, n_meet)
            uniform = 0
            if paint:
                uniform = K.paint_labels(img, self._dummy_lab, True, self.mask,
                                         self.wpr, conn8, self.n_ring,
                                         self.n_int, self.lab, self.anchor,
                                         self.seed_label, nt)
        self.n_blobs = int(roots + n_int)
        return {"n_blobs": self.n_blobs, "rounds": int(rounds),
                "peak_round_entries": int(peak), "parallel_rounds": int(par),
                "peak_chunks": int(pch),
                "seeds": int(n_seeds), "ring_pieces": int(self.n_ring.sum()),
                "interior_pieces": int(n_int), "meetings": int(n_meet),
                "expanded": int(tots[K.LS_EXPANDED]),
                "lost": int(tots[K.LS_LOST]),
                "dropped": int(tots[K.LS_DROPPED]),
                "claims": int(tots[K.LS_CLAIMS]),
                "requeued": int(tots[K.LS_REQUEUED]),
                "uniform_tiles": int(uniform), "n_tiles": self.n_tiles,
                "lattice": self.lattice, "threads": int(nt)}

    def label_map(self):
        """int32 (W, H): the last run's labels, -1 off blob (off clock)."""
        out = np.empty((self.width, self.height), np.int32)
        K.fill_i32(out, -1)
        K.paint_labels(self._dummy_img, out, False, self.mask, self.wpr,
                       self.connectivity == 8, self.n_ring, self.n_int,
                       self.lab, self.anchor, self.seed_label,
                       numba.get_num_threads())
        return out


@dataclass
class LabelResult:
    img: np.ndarray        # (W, H, 3) uint8: a copy, palette[label % 6]
    label: np.ndarray      # (W, H) int32: canonical labels, -1 off blob
    n_blobs: int
    rounds: int            # wave rounds
    connectivity: int
    threads: int
    stats: dict = field(repr=False)


def label(img, connectivity=8, lattice=DEFAULT_LATTICE, threads=None,
          **engine_kw):
    """Label and paint every red blob on every core (tile waves).

    img: (width, height, 3) uint8, any memory layout; not modified."""
    if (not isinstance(img, np.ndarray) or img.ndim != 3
            or img.shape[2] != 3 or img.dtype != np.uint8):
        raise ValueError("img must be a (width, height, 3) uint8 array")
    out = np.ascontiguousarray(img).copy()
    eng = TileLabels(out.shape[0], out.shape[1], connectivity=connectivity,
                     lattice=lattice, threads=threads, **engine_kw)
    st = eng.run(out, paint=True)
    return LabelResult(img=out, label=eng.label_map(), n_blobs=st["n_blobs"],
                       rounds=st["rounds"], connectivity=connectivity,
                       threads=st["threads"], stats=st)


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
