"""Host driver for the tile-BFS flood fill.

Public API:
    flood_fill(img, seed_x, seed_y, connectivity=4, threads_per_block=128,
               blocks=None) -> TileFloodFillResult      one-shot, allocates
    TileFill(width, height, connectivity, threads_per_block, blocks)
        .run(img_dev, seed_x, seed_y) -> (names, events)  reusable buffers
    max_blocks(threads_per_block, connectivity)          cooperative capacity
    regs_per_thread(connectivity)                        compiled register count

The pipeline is four launches on one stream, each bracketed by CUDA
events so the phases are attributable individually:

    pack     RGB -> 1 bit/px mask         (ch06's pack_kernel, imported)
    init     reached = seed bit, stamps, queue, counters (device-side)
    rounds   the cooperative tile BFS     (two grid.sync per round)
    paint    reached -> (0, 0, 255)       (one coalesced pass)

The rounds kernel's switches (SWITCHES, kernels._make_rounds_kernel)
change the middle: one_sync (one grid.sync per round), fused_pack (no
pack launch: the rounds read each reached tile's RGB, and paint is
paint_tiles_kernel), skip_full (no exports into full tiles already
queued) and fused_paint (no paint launch; needs fused_pack). With a
fused phase its events are recorded back to back (an empty interval).

`kernel_ms` is the first event to the last: all four phases, no
allocation and no host copies. ch03's `kernel_ms` covers its one launch
and leaves its own buffer initialisation (visited, depth, owner, queue)
to host copies before the clock; ch07 puts its initialisation ON the
clock, so the comparison is conservative for this chapter.

Same input contract as ch03: img is (width, height, 3) uint8, x first; a
seed that is not pure red raises ValueError; the input is not modified
and a recoloured copy is returned. There is no depth map: the BFS runs
over tiles, so per-pixel depth does not exist here.
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import time
from dataclasses import dataclass, field

import numpy as np
from numba import cuda

from ..ch06_gpu_nblob_runs.recolor import PACK_ROW_BLOCKS
from .kernels import (
    pack_kernel, init_kernel, paint_kernel, paint_tiles_kernel,
    get_rounds_kernel, gpu_sleep_kernel,
    TILE, TY_BITS, NUM_COUNTERS, ROUNDS, TILE_VISITS, CLOSURE_ITERS,
    OR_ATOMICS, STAMP_ATOMICS, PEAK_ROUND, OVERFLOW, PACKED_TILES,
    SKIPPED_EXPORTS, FIRST_ENTRY, NOT_PACKED,
)

# Register cap of the rounds kernel unless the caller picks one (see
# kernels.get_rounds_kernel; None = whatever the linker allocates).
# Linked, the 8-conn kernel takes 143 registers/thread (ptxas compiles
# the same PTX standalone to 87; linking the external grid.sync call adds
# the rest), which seats 288 warps. A cap of 80 doubled that to 576 warps
# and won while every closure shuffle took a slow path (see
# kernels._col_fill); with the shuffles fixed, its spills cost more than
# the extra warps buy, so the default is no cap.
DEFAULT_MAX_REGISTERS = None

# 4 warps per block: 72 blocks x 128 seat all 288 warps the registers
# allow. At 256 threads the same registers seat only 24 blocks (192
# warps), so ch03's tpb=256 default is not carried over.
DEFAULT_TPB = 128

PHASES = ("pack", "init", "rounds", "paint")

# The rounds kernel's switches (kernels._make_rounds_kernel), each one an
# ablation the benchmark times against the original kernel:
#   one_sync     one grid.sync per round: three rotating queue counters
#   fused_pack   no pack launch: a tile's first visit reads its own RGB
#   skip_full    drop exports into full tiles already queued this round
#   fused_paint  no paint launch: each visit paints what it added
#                (needs fused_pack)
# flood_fill() and TileFill take None for "this module's default", so the
# tests can run the whole suite under every combination.
SWITCHES = ("one_sync", "fused_pack", "skip_full", "fused_paint")
DEFAULT_SWITCHES = {"one_sync": False, "fused_pack": False,
                    "skip_full": False, "fused_paint": False}

# Plain (non-cooperative) launch shapes for the three streaming kernels.
# pack/paint use ch06's 2D grid: word along x (one warp per word), rows
# strided along y. init is a 1D grid-stride.
PLAIN_TPB = 256
INIT_BLOCKS = 512
# paint_tiles_kernel: 8 warps per block, one tile per warp up to 65,536
# tiles, grid-stride beyond.
PAINT_TILE_TPB = 256
PAINT_TILE_MAX_BLOCKS = 8192
# pack/paint shape (see TileFill.plain_shape): a block is at most one
# row's words wide (32 * min(8, words_per_row) threads), and gridDim.y
# puts about PACK_WARPS_IN_FLIGHT working warps on the image, never fewer
# than ch06's PACK_ROW_BLOCKS rows of blocks. ch06's fixed (wpr*32/256,
# 64) x 256 left 7 of 8 warps idle and only 64 rows in flight on a thin
# image: H = 32 packed and painted in 5.5 ms instead of 0.69.
PACK_WARPS_IN_FLIGHT = 8192

# Image-size limits from the queue-entry format (tx << 16) | ty.
MAX_TILES_X = 1 << (31 - TY_BITS)
MAX_WORDS_PER_ROW = 1 << TY_BITS

_coop_cache = {}
_warmed = set()


def _is_red(img, x, y):
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


def _overload(kernel_fn):
    return next(iter(kernel_fn.overloads.values()))


def resolve_switches(**switches):
    """{name: bool} for every name in SWITCHES; None (or absent) means
    DEFAULT_SWITCHES[name] at call time."""
    unknown = set(switches) - set(SWITCHES)
    if unknown:
        raise TypeError(f"unknown switch(es) {sorted(unknown)}")
    return {k: bool(DEFAULT_SWITCHES[k] if switches.get(k) is None
                    else switches[k]) for k in SWITCHES}


def _sw_key(sw):
    return tuple(sw[k] for k in SWITCHES)


def _warmup(connectivity, bare=False, max_registers=DEFAULT_MAX_REGISTERS,
            **switches):
    """JIT-compile every kernel of the pipeline once on a tiny image."""
    sw = resolve_switches(**switches)
    key = (connectivity, bare, max_registers) + _sw_key(sw)
    if key in _warmed:
        return
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    for h in (8, 7):      # both RGB read paths of the fused pack
        eng = TileFill(8, h, connectivity=connectivity, threads_per_block=32,
                       blocks=1, bare=bare, max_registers=max_registers,
                       _skip_warmup=True, **sw)
        dev = cuda.to_device(np.ascontiguousarray(tiny[:, :h]))
        eng.run(dev, 1, 1)
    gpu_sleep_kernel[1, 32](1)
    cuda.synchronize()
    _warmed.add(key)


def max_blocks(threads_per_block=DEFAULT_TPB, connectivity=4, bare=False,
               max_registers=DEFAULT_MAX_REGISTERS, **switches):
    """Largest cooperative grid of the rounds kernel at this block size
    (what blocks=None resolves to). Queried per twin, never assumed."""
    sw = resolve_switches(**switches)
    _warmup(connectivity, bare, max_registers, **sw)
    key = (connectivity, bare, max_registers, threads_per_block) + _sw_key(sw)
    if key not in _coop_cache:
        kern = get_rounds_kernel(connectivity, bare, max_registers, **sw)
        _coop_cache[key] = _overload(kern).max_cooperative_grid_blocks(
            threads_per_block)
    return _coop_cache[key]


def regs_per_thread(connectivity=4, bare=False,
                    max_registers=DEFAULT_MAX_REGISTERS, **switches):
    """Registers per thread of the compiled (linked) rounds kernel."""
    sw = resolve_switches(**switches)
    _warmup(connectivity, bare, max_registers, **sw)
    regs = get_rounds_kernel(connectivity, bare, max_registers,
                             **sw).get_regs_per_thread()
    return int(next(iter(regs.values()))) if isinstance(regs, dict) \
        else int(regs)


def local_bytes_per_thread(connectivity=4, bare=False,
                           max_registers=DEFAULT_MAX_REGISTERS, **switches):
    """Local memory per thread (register spills) of the rounds kernel."""
    sw = resolve_switches(**switches)
    _warmup(connectivity, bare, max_registers, **sw)
    lm = get_rounds_kernel(connectivity, bare, max_registers,
                           **sw).get_local_mem_per_thread()
    return int(next(iter(lm.values()))) if isinstance(lm, dict) else int(lm)


def plain_shape(width, words_per_row):
    """((gridDim.x, gridDim.y), blockDim.x) for pack_kernel / paint_kernel.

    Both kernels give one warp one word of a row and stride rows along
    gridDim.y; warps with w >= words_per_row return at once. So a block
    is at most one row's words wide, and gridDim.y is chosen for about
    PACK_WARPS_IN_FLIGHT working warps (words_per_row x gridDim.y), with
    ch06's 64 rows as the floor. On squares of 4 to 64 Mpx this is ch06's
    shape (to within 2%); on images with H <= 256 it is 1.6-7.5x faster
    (measured in the chapter's A/B, outputs bit-identical).
    """
    tpb = 32 * min(8, words_per_row)
    gx = (words_per_row * 32 + tpb - 1) // tpb
    gy = max(PACK_ROW_BLOCKS,
             -(-PACK_WARPS_IN_FLIGHT // words_per_row))
    return (gx, min(width, 65535, gy)), tpb


def unpack_bits(words, height):
    """(width, words_per_row) uint32 -> (width, height) int32 0/1, bit b
    of word w being pixel 32*w + b (little-endian bit order)."""
    width = words.shape[0]
    as_bytes = np.ascontiguousarray(words).view(np.uint8).reshape(width, -1)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
    return bits[:, :height].astype(np.int32)


class TileFill:
    """Device buffers for one image shape and one launch configuration,
    reused across runs (the benchmark's engine).

    run() only enqueues: it neither synchronizes nor reads anything back,
    so the caller owns the clock boundaries.
    """

    def __init__(self, width, height, connectivity=4,
                 threads_per_block=DEFAULT_TPB, blocks=None, bare=False,
                 max_registers=DEFAULT_MAX_REGISTERS,
                 _skip_warmup=False, **switches):
        if width < 1 or height < 1:
            raise ValueError(f"bad shape {width}x{height}")
        if connectivity not in (4, 8):
            raise ValueError(
                f"connectivity must be 4 or 8, got {connectivity!r}")
        if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 1024):
            raise ValueError(
                f"threads_per_block must be a multiple of 32 in [32, 1024], "
                f"got {threads_per_block}")
        if blocks is not None:
            if not isinstance(blocks, (int, np.integer)) or isinstance(blocks, bool):
                raise ValueError(f"blocks must be an int or None, got {blocks!r}")
            if blocks < 1:
                raise ValueError(f"blocks must be >= 1, got {blocks}")
        self.width = int(width)
        self.height = int(height)
        self.connectivity = connectivity
        self.words_per_row = (self.height + 31) // 32
        self.tiles_x = (self.width + TILE - 1) // TILE
        if self.tiles_x > MAX_TILES_X or self.words_per_row > MAX_WORDS_PER_ROW:
            raise ValueError(
                f"image too large for the (tx << {TY_BITS}) | ty queue entry: "
                f"{self.tiles_x} tile rows, {self.words_per_row} words per row")
        self.n_tiles = self.tiles_x * self.words_per_row
        self.threads_per_block = int(threads_per_block)

        self.bare = bool(bare)
        self.max_registers = max_registers
        self.switches = resolve_switches(**switches)
        self.one_sync = self.switches["one_sync"]
        self.fused_pack = self.switches["fused_pack"]
        self.skip_full = self.switches["skip_full"]
        self.fused_paint = self.switches["fused_paint"]
        if self.fused_paint and not self.fused_pack:
            raise ValueError("fused_paint needs fused_pack")
        self.kernel_fn = get_rounds_kernel(connectivity, self.bare,
                                           max_registers, **self.switches)
        if not _skip_warmup:
            _warmup(connectivity, self.bare, max_registers, **self.switches)
            coop_max = max_blocks(self.threads_per_block, connectivity,
                                  self.bare, max_registers, **self.switches)
            if coop_max < 1:
                raise RuntimeError(
                    f"this GPU cannot host one cooperative block of the "
                    f"rounds kernel at {threads_per_block} threads")
            if blocks is None:
                self.blocks = coop_max
            elif blocks > coop_max:
                raise RuntimeError(
                    f"blocks={blocks} exceeds this GPU's cooperative-launch "
                    f"capacity of {coop_max} blocks at {threads_per_block} "
                    f"threads (grid.sync would deadlock)")
            else:
                self.blocks = int(blocks)
            self.coop_max_blocks = coop_max
        else:
            self.blocks = 1 if blocks is None else int(blocks)
            self.coop_max_blocks = -1

        self.mask = cuda.device_array((self.width, self.words_per_row),
                                      dtype=np.uint32)
        self.reached = cuda.device_array((self.width, self.words_per_row),
                                         dtype=np.uint32)
        self.stamp = cuda.device_array((self.tiles_x, self.words_per_row),
                                       dtype=np.int32)
        n_queues = 3 if self.one_sync else 2
        self.queues = cuda.device_array(n_queues * self.n_tiles,
                                        dtype=np.int32)
        self.q_state = cuda.device_array(3, dtype=np.int32)
        self.counters = cuda.device_array(NUM_COUNTERS, dtype=np.int64)
        # per-tile info (packed round, full bit) for the fused pack and
        # the skip-full test; a 1-element dummy otherwise
        use_tinfo = self.fused_pack or self.skip_full
        self.tinfo = cuda.device_array(self.n_tiles if use_tinfo else 1,
                                       dtype=np.int32)
        self.tinfo_init = np.int32(NOT_PACKED if self.fused_pack else 0)
        # the fused pack reads RGB as uint32 words: 3 per 4 pixels when
        # every row starts 4-byte aligned (height % 4 == 0), 4 and a
        # funnel shift otherwise
        self.aligned = self.height % 4 == 0
        self.painted = cuda.device_array(
            self.width * self.words_per_row if self.fused_paint else 1,
            dtype=np.uint32)
        self._dummy_u8 = cuda.device_array(1, dtype=np.uint8)
        self._dummy_u32 = cuda.device_array(1, dtype=np.uint32)
        n_tile_warps = -(-self.n_tiles * 32 // PAINT_TILE_TPB)
        self.paint_tile_blocks = max(1, min(PAINT_TILE_MAX_BLOCKS,
                                            n_tile_warps))
        # flat views: the rounds kernel indexes everything with a uint32
        n_words = self.width * self.words_per_row
        self._mask_flat = self.mask.reshape(n_words)
        self._reached_flat = self.reached.reshape(n_words)
        self._stamp_flat = self.stamp.reshape(self.n_tiles)

        self.plain_grid, self.plain_tpb = plain_shape(self.width,
                                                      self.words_per_row)

    def pack(self, img_dev):
        """RGB -> packed mask only (untimed helper, used by the tests)."""
        pack_kernel[self.plain_grid, self.plain_tpb](img_dev, self.mask)

    def seed_args(self, seed_x, seed_y):
        """(seed word, seed bit, seed tile, first queue entry).

        The seed is made a Python int first. Under NumPy 2 (NEP 50) a
        Python int operand does not widen a NumPy scalar, so an np.int16
        seed wrapped `tx << 16` to 0 (round 0 closed the wrong tile) and
        an np.uint8 seed wrapped `1 << (y % 32)` to 0 (no seed bit).
        """
        seed_x, seed_y = int(seed_x), int(seed_y)
        tx = seed_x // TILE
        ty = seed_y // 32
        return (int(seed_x * self.words_per_row + ty),
                np.uint32(1 << (seed_y % 32)),
                int(tx * self.words_per_row + ty),
                int((tx << TY_BITS) | ty))

    def run(self, img_dev, seed_x, seed_y):
        """Enqueue the four phases; returns (phase names, 5 events).

        With fused_pack there is no pack launch (its interval is empty:
        the RGB is read inside `rounds`), and `paint` is
        paint_tiles_kernel."""
        seed_word, seed_bits, seed_tile, seed_entry = self.seed_args(
            seed_x, seed_y)
        if self.fused_pack or self.skip_full:
            seed_entry |= FIRST_ENTRY     # the seed tile's first visit
        if self.fused_pack:
            img8 = img_dev.reshape(self.width * self.height * 3)
            n32 = img8.shape[0] // 4
            img32 = (img8[:4 * n32].view(np.uint32) if n32
                     else self._dummy_u32)
        else:
            img8, img32, n32 = self._dummy_u8, self._dummy_u32, 0
        events = [cuda.event(timing=True) for _ in range(len(PHASES) + 1)]
        events[0].record()
        if not self.fused_pack:
            pack_kernel[self.plain_grid, self.plain_tpb](img_dev, self.mask)
        events[1].record()
        init_kernel[INIT_BLOCKS, PLAIN_TPB](
            self._reached_flat, self._stamp_flat, self.queues, self.q_state,
            self.counters, seed_word, seed_bits, seed_tile, seed_entry,
            self.tinfo, self.tinfo_init, self.painted)
        events[2].record()
        self.kernel_fn[self.blocks, self.threads_per_block](
            self._mask_flat, self._reached_flat, self._stamp_flat,
            self.queues, self.q_state, self.counters,
            np.int32(self.width), np.int32(self.words_per_row),
            np.int32(self.n_tiles), img8, img32, np.int64(n32),
            np.int32(self.height), self.aligned, self.tinfo, self.painted)
        events[3].record()
        if self.fused_paint:
            pass                          # painted inside the rounds
        elif self.fused_pack:
            paint_tiles_kernel[self.paint_tile_blocks, PAINT_TILE_TPB](
                self._reached_flat, self._mask_flat, self.tinfo, img8,
                img32, self.aligned, np.int32(self.width),
                np.int32(self.height), np.int32(self.words_per_row),
                np.int32(self.n_tiles))
        else:
            paint_kernel[self.plain_grid, self.plain_tpb](self.reached,
                                                          img_dev)
        events[4].record()
        return PHASES, events

    def read_counters(self):
        c = self.counters.copy_to_host()
        if c[OVERFLOW]:
            raise RuntimeError(
                "structural tripwire fired - this indicates a kernel bug: "
                "the stamp admits a tile to a round's queue at most once")
        # the bare twins leave everything but rounds/peak at zero
        return {
            "rounds": int(c[ROUNDS]),
            "tile_visits": int(c[TILE_VISITS]),
            "closure_iters": int(c[CLOSURE_ITERS]),
            "or_atomics": int(c[OR_ATOMICS]),
            "stamp_atomics": int(c[STAMP_ATOMICS]),
            "peak_round_tiles": int(c[PEAK_ROUND]),
            "packed_tiles": int(c[PACKED_TILES]),
            "skipped_exports": int(c[SKIPPED_EXPORTS]),
        }

    def reached_words(self):
        """The fill set as packed words (host). With fused_pack, `reached`
        may hold candidate bits that are not red, and the fill set is
        reached & mask (mask words of never-packed tiles are garbage, but
        such tiles hold no reached bit)."""
        words = self.reached.copy_to_host()
        if self.fused_pack:
            words &= self.mask.copy_to_host()
        return words

    def filled(self):
        """Reached pixels (popcount of the fill set)."""
        words = self.reached_words()
        return int(np.unpackbits(words.view(np.uint8)).sum())

    def tiles_touched(self):
        """Tiles holding at least one reached pixel: the visits a perfect
        schedule would need (each tile closed exactly once)."""
        words = self.reached_words()
        pad = self.tiles_x * TILE - self.width
        if pad:
            words = np.concatenate(
                [words, np.zeros((pad, self.words_per_row), words.dtype)])
        per_tile = words.reshape(self.tiles_x, TILE, self.words_per_row)
        return int(np.count_nonzero(per_tile.any(axis=1)))


def phase_times(events):
    """Event list -> (per-phase ms dict, total ms)."""
    ms = {name: cuda.event_elapsed_time(events[i], events[i + 1])
          for i, name in enumerate(PHASES)}
    return ms, cuda.event_elapsed_time(events[0], events[-1])


@dataclass
class TileFloodFillResult:
    img: np.ndarray        # (width, height, 3) uint8, blob recolored blue
    visited: np.ndarray    # (width, height) int32, 1 where reached
    filled: int
    rounds: int            # tile-BFS rounds (the "levels" of this chapter)
    tile_visits: int       # warps that closed a tile, summed over rounds
    closure_iters: int     # closure loop trips, summed over visits
    or_atomics: int        # atomic ORs on the reached bitmap
    stamp_atomics: int     # enqueue requests (atomic.max on the stamp)
    peak_round_tiles: int  # largest round
    n_tiles: int           # tiles in the image (32 rows x 32 y each)
    connectivity: int
    threads_per_block: int
    blocks: int
    coop_max_blocks: int
    bare: bool             # uninstrumented twin: work counters read 0
    one_sync: bool         # one grid.sync per round (three queues)
    fused_pack: bool       # RGB read on a tile's first visit
    skip_full: bool        # exports into queued full tiles dropped
    fused_paint: bool      # each visit paints what it added
    packed_tiles: int      # fused pack: tiles whose RGB was read
    skipped_exports: int   # skip-full: export words dropped
    regs_per_thread: int
    phase_ms: dict = field(repr=False)  # pack / init / rounds / paint
    kernel_ms: float = 0.0  # first event to last: the four phases
    alloc_ms: float = 0.0
    h2d_ms: float = 0.0
    d2h_ms: float = 0.0
    total_ms: float = 0.0


def flood_fill(img_host, seed_x, seed_y, connectivity=4,
               threads_per_block=DEFAULT_TPB, blocks=None, bare=False,
               max_registers=DEFAULT_MAX_REGISTERS, one_sync=None,
               fused_pack=None, skip_full=None, fused_paint=None):
    """Flood-fill the red blob containing (seed_x, seed_y) by tile BFS.

    img_host: (width, height, 3) uint8. Not modified; a recolored copy is
    returned. Raises ValueError for bad inputs (including a seed that is
    not red, exactly like ch03) and RuntimeError if the GPU cannot host
    the requested cooperative launch.
    """
    if img_host.ndim != 3 or img_host.shape[2] != 3 or img_host.dtype != np.uint8:
        raise ValueError("img must be a (width, height, 3) uint8 array")
    width, height = img_host.shape[0], img_host.shape[1]
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError(f"seed ({seed_x}, {seed_y}) outside {width}x{height} image")
    if not _is_red(img_host, seed_x, seed_y):
        raise ValueError(
            f"seed pixel ({seed_x}, {seed_y}) is not red - nothing to fill")

    t0 = time.perf_counter()
    eng = TileFill(width, height, connectivity=connectivity,
                   threads_per_block=threads_per_block, blocks=blocks,
                   bare=bare, max_registers=max_registers,
                   one_sync=one_sync, fused_pack=fused_pack,
                   skip_full=skip_full, fused_paint=fused_paint)
    d_img = cuda.device_array_like(img_host)
    cuda.synchronize()
    t1 = time.perf_counter()
    d_img.copy_to_device(img_host)
    cuda.synchronize()
    t2 = time.perf_counter()
    gpu_sleep_kernel[1, 32](20)  # keep Python's launch cost off the events
    _, events = eng.run(d_img, seed_x, seed_y)
    events[-1].synchronize()
    t3 = time.perf_counter()
    stats = eng.read_counters()
    img_out = d_img.copy_to_host()
    visited = unpack_bits(eng.reached_words(), height)
    t4 = time.perf_counter()
    phase_ms, kernel_ms = phase_times(events)

    return TileFloodFillResult(
        img=img_out,
        visited=visited,
        filled=int(visited.sum()),
        n_tiles=eng.n_tiles,
        connectivity=connectivity,
        threads_per_block=eng.threads_per_block,
        blocks=eng.blocks,
        coop_max_blocks=eng.coop_max_blocks,
        bare=eng.bare,
        one_sync=eng.one_sync,
        fused_pack=eng.fused_pack,
        skip_full=eng.skip_full,
        fused_paint=eng.fused_paint,
        regs_per_thread=regs_per_thread(connectivity, eng.bare,
                                        max_registers, **eng.switches),
        phase_ms=phase_ms,
        kernel_ms=kernel_ms,
        alloc_ms=(t1 - t0) * 1000,
        h2d_ms=(t2 - t1) * 1000,
        d2h_ms=(t4 - t3) * 1000,
        total_ms=(t4 - t0) * 1000,
        **stats,
    )
