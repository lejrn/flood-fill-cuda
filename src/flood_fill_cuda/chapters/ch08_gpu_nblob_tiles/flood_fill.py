"""Host driver for the seeded tile waves (ch08).

Public API (mirrors ch06's recolor.py):
    TileWaves(width, height, connectivity=8, lattice=8, valve_round=16,
              lookup=DEFAULT_LOOKUP, dense=DEFAULT_DENSE, dense_min=4,
              ccl_min=3, paint_ccl_min=3, coll_capacity=None,
              instrumented=False, debug=False, ...)   reusable buffers
        .pack(img_dev)                 RGB -> mask (mask contract, untimed)
        .run(img_dev, contract="rgb")  enqueue only -> (phases, events)
        .emit_label_map(label_dev)     int32 (W, H), -1 off-blob (off clock)
        .emit_seeds()                  canonical seeds, sorted (off clock)
        .read_counters(), .n_blobs()   (off clock)
    recolor(img_host, contract="rgb", emit_label=False, emit_seeds=True,
            copy_img=True, engine=None, **engine_kw) -> TileWavesResult

The pipeline is five launches on one stream with a CUDA event after
each (the mask contract drops K1):

    pack    K1 pack_tiles: RGB -> 1 bit/px, read tile by tile
    init    K0: zero the queue state and the counters
    label   K2: close every tile comp in registers, write ring records
    waves   K3: cooperative rounds of claims, unions, flatten
    paint   K4: palette[label % 6] on red, word-coalesced

With dense=True two more: label_ccl (K2b, the run CCL of the tiles K2
left in the tile list) after label, and paint_ccl (K4b) after paint.
The dindex lookup with waves_compact="auto" launches two K3 twins in
the waves stage; one returns at once.

`kernel_ms` is the first event to the last. No host copy is on the
clock (ch06 lost 1.72 ms to one): init is a device kernel.

Contracts (ch06's three tiers):
    rgb         RGB image on the device, recoloured in place, pack timed
    mask        the packed mask is the input (pack off the clock); the
                paint still writes RGB
    label-only  init + label + waves of the mask contract: every comp
                carries its canonical label, nothing is painted
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import time
from dataclasses import dataclass, field

import numpy as np
from numba import cuda

from . import kernels as k
from .kernels import (
    init_kernel, get_pack_kernel, get_label_kernel, get_waves_kernel,
    get_paint_kernel, get_label_dense_kernel, get_paint_dense_kernel,
    get_waves_dense_kernel,
    gpu_sleep_kernel, fill_i32, pack_shape, Q_DENSE, Q_PDENSE,
    MODE_RGB, MODE_LABEL, MODE_SEEDS, NUM_COUNTERS, COUNTER_NAMES,
    N_QSTATE, N_STAMPS, IDS_PER_TILE, REC, TILE, HDR_RING_MASK,
    HDR_TOTAL_SHIFT, ST_TAIL, ST_UNION, ST_FLAT1, ST_FLAT2, INDEX_BYTES,
    CCL_MAX_WARPS)

CONTRACTS = ("rgb", "mask")
# dindex (dense tiles): the index where K2's run CCL wrote one, the scan
# elsewhere; it needs the dense K2
LOOKUPS = ("scan", "index", "hybrid", "dindex")
# row-major stores of the mask K2-K4 read; get_pack_kernel also builds
# "tilemajor", experiment E1's reference store, which no engine may use
PACK_VARIANTS = ("direct", "smem")
PHASES = ("pack", "init", "label", "waves", "paint")
MASK_PHASES = PHASES[1:]
LABEL_ONLY_PHASES = ("init", "label", "waves")

# Plain launch shapes, grid-stride over every tile. K2 runs 1536 x 128
# with an 80-register cap on the scan twin (experiment E3: label 0.268 ->
# 0.257 ms on input_blobs, 24 warps/SM instead of 20, no spills; also
# faster on input_blocks and random_4000). The index twin's K2 spills 16
# bytes at 80, so it keeps the compiler's 96 (E4b). K4 stays 768 x 128
# (E5: 256 threads tied on input_blobs and lost 6.5% on input_blocks).
# K3 is cooperative at 256 threads per block and the cooperative maximum
# of its compiled twin (E10: 128 threads tied).
LABEL_GRID = (1536, 128)
LABEL_MAX_REGISTERS = {"scan": 80, "index": None, "hybrid": 80,
                       "dindex": 80}
# K3's register cap per lookup (None: the compiler's choice); "auto" in
# the API means this table. The hybrid twin takes 97 registers by
# itself, which allows 2 cooperative blocks of 256 per SM; at 80 it
# keeps 3 (72 blocks), with no more local memory than the scan twin
# (experiment E13).
WAVES_MAX_REGISTERS = {"scan": None, "index": None, "hybrid": 80,
                       "dindex": 80}
# K3's neighbour lookup. E13 (session experiments_20261015T180817Z) chose
# the hybrid at dense_min 4 and 80 registers: input_blobs 0.991 of the
# scan (per-round median), 0.29 on the decoy staircase, 0.42 on
# zz_lattice_4k. The index twin is faster on long shapes but cost 1.5-2.0%
# on input_blobs (its K2 wrote the byte index for every tile). The dense
# path's dindex indexes only the tiles K2b labels, and falls back to the
# hybrid elsewhere.
DEFAULT_LOOKUP = "dindex"
PAINT_GRID = (768, 128)
WAVES_TPB = 256
DEFAULT_LATTICE = 8
# The escalation valve: if round 16 still claims comps, every unclaimed
# ring comp seeds itself. E9 (session experiments_20261015T181414Z, at
# L8 and the hybrid lookup): the main scenes need at most 8 rounds, so
# it never fires there; where it fires, 16 beat 128 on every scene
# (decoy staircase 0.394, diagonal snake 0.620 of the 128 arm).
DEFAULT_VALVE_ROUND = 16
# hybrid lookup: a neighbour tile with more ring comps than this is
# scanned by the whole warp at once (a runtime argument of K3, E13)
DEFAULT_DENSE_MIN = 4
# collision list soft cap per tile; entries past it become inline unions
COLL_PER_TILE = 16
# Dense tiles (the run CCL in K2 and K4). A tile goes to K2's run CCL
# when its estimated comp count (runs touching no red pixel of the row
# above, an upper bound) reaches ccl_min, and to K4's when its header's
# comp count reaches paint_ccl_min. Both are runtime arguments.
DEFAULT_DENSE = True
DEFAULT_CCL_MIN = 12
DEFAULT_PAINT_CCL_MIN = 10
# ... and at most this many runs per (estimated) comp
DEFAULT_CCL_RPC = 12
DEFAULT_PAINT_RPC = 24
# The dindex K3 (its own twin; the other lookups keep the chapter's K3):
# flat_opt, lone seeds skip the find and the rmin is read before the
# atomicMin; compact, a warp runs only the (entry, direction) tasks that
# exist, 32 at a time. Both compile-time flags, dindex only.
DEFAULT_FLAT_OPT = True
# compact: True, False, or "auto": both twins are launched and the first
# round's size picks one (compact iff it holds >= compact_min entries).
# auto picks the better twin on almost every scene, but the cooperative
# launch that returns at once costs 6-10 us (crop_1000 1.152 of the old
# path vs 1.086 for the plain twin alone), so the default is True.
DEFAULT_COMPACT = True
DEFAULT_COMPACT_MIN = 16384
DEFAULT_COOP = True
# fit K2's and K4's grids to the tile count (small images)
DEFAULT_FIT_GRIDS = True
# K2b's register cap (None: the compiler's choice)
LABEL_DENSE_MAX_REGISTERS = None
# The chapter's path before the dense-tile work: the ablation baseline.
OLD_PATH = dict(dense=False, lookup="hybrid", fit_grids=False)

_warmed = set()
_coop = {}


def _overload(kern):
    return next(iter(kern.overloads.values()))


def _one(v):
    return int(next(iter(v.values()))) if isinstance(v, dict) else int(v)


def _fit(grid, n_tiles):
    """A one-warp-per-tile grid cut to the tile count."""
    blocks, tpb = (int(v) for v in grid)
    wpb = tpb // 32
    return (max(1, min(blocks, (n_tiles + wpb - 1) // wpb)), tpb)


def _check_grid(name, grid):
    """A plain launch shape (blocks, threads) for a one-warp-per-tile
    kernel: warp = grid(1) >> 5 and FULL-mask warp calls need whole
    warps, or a logical warp straddles two hardware warps and the tiles
    come out wrong with no error."""
    blocks, tpb = (int(v) for v in grid)
    if blocks < 1 or tpb % 32 or not 32 <= tpb <= 1024:
        raise ValueError(f"{name} must be (blocks >= 1, threads a multiple "
                         f"of 32 in [32, 1024]), got {tuple(grid)}")


@dataclass
class TileWavesResult:
    img: np.ndarray            # painted copy (empty unless copy_img)
    label: np.ndarray          # canonical label map (empty unless emit_label)
    contract: str
    n_blobs: int
    seeds: list                # canonical (x, y) per blob, label order
    counters: dict             # every counter of the run
    rounds: int
    phase_ms: dict = field(repr=False)
    kernel_ms: float = 0.0
    label_only_ms: float = 0.0  # init + label + waves of this run
    alloc_ms: float = 0.0
    d2h_ms: float = 0.0
    total_ms: float = 0.0
    waves_blocks: int = 0
    waves_tpb: int = 0


class TileWaves:
    """Device buffers for one image shape and one configuration, reused
    across runs (the benchmark's engine). run() only enqueues."""

    def __init__(self, width, height, connectivity=8,
                 lattice=DEFAULT_LATTICE, valve_round=DEFAULT_VALVE_ROUND,
                 lookup=DEFAULT_LOOKUP, coll_capacity=None,
                 instrumented=False, debug=False, waves_tpb=WAVES_TPB,
                 waves_blocks=None,
                 label_grid=LABEL_GRID, paint_grid=PAINT_GRID,
                 label_max_registers="auto", paint_fast=True,
                 pack_variant="direct", label_exits=2,
                 dense_min=DEFAULT_DENSE_MIN, waves_max_registers="auto",
                 dense=DEFAULT_DENSE, ccl_min=DEFAULT_CCL_MIN,
                 paint_ccl_min=DEFAULT_PAINT_CCL_MIN,
                 ccl_rpc=DEFAULT_CCL_RPC, paint_rpc=DEFAULT_PAINT_RPC,
                 flat_opt=DEFAULT_FLAT_OPT, fit_grids=DEFAULT_FIT_GRIDS,
                 label_dense_max_registers=LABEL_DENSE_MAX_REGISTERS,
                 waves_compact=DEFAULT_COMPACT, waves_coop=DEFAULT_COOP,
                 compact_min=DEFAULT_COMPACT_MIN, _skip_warmup=False):
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError(f"bad shape {width}x{height}")
        if width * height >= 2 ** 31:
            raise ValueError("image too large: width*height must be < 2**31 "
                             "(labels are int32 linear indices)")
        if connectivity not in (4, 8):
            raise ValueError(
                f"connectivity must be 4 or 8, got {connectivity!r}")
        if lookup not in LOOKUPS:
            raise ValueError(f"lookup must be one of {LOOKUPS}, got "
                             f"{lookup!r}")
        if lattice < 0 or valve_round < 0:
            raise ValueError("lattice and valve_round must be >= 0")
        if waves_tpb % 32 or not 32 <= waves_tpb <= 1024:
            raise ValueError(f"waves_tpb must be a multiple of 32 in "
                             f"[32, 1024], got {waves_tpb}")
        _check_grid("label_grid", label_grid)
        _check_grid("paint_grid", paint_grid)
        if pack_variant not in PACK_VARIANTS:
            raise ValueError(f"pack_variant must be one of {PACK_VARIANTS}, "
                             f"got {pack_variant!r}")
        self.dense = bool(dense)
        if lookup == "dindex" and not self.dense:
            raise ValueError("the dindex lookup reads the index that the "
                             "dense K2 writes: it needs dense=True")
        if self.dense:
            for name, grid in (("label_grid", label_grid),
                               ("paint_grid", paint_grid)):
                if int(grid[1]) > 32 * CCL_MAX_WARPS:
                    raise ValueError(
                        f"{name}: the dense twins take at most "
                        f"{32 * CCL_MAX_WARPS} threads per block, got "
                        f"{tuple(grid)}")
        self.ccl_min = int(ccl_min)
        self.paint_ccl_min = int(paint_ccl_min)
        self.ccl_rpc = int(ccl_rpc)
        self.paint_rpc = int(paint_rpc)
        self.flat_opt = bool(flat_opt) and lookup == "dindex"
        if waves_compact not in (True, False, "auto"):
            raise ValueError("waves_compact must be True, False or 'auto'")
        self.waves_compact = (waves_compact if lookup == "dindex"
                              else False)
        self.compact_min = int(compact_min)
        self.waves_coop = bool(waves_coop) or lookup != "dindex"
        self.fit_grids = bool(fit_grids)
        self.width = width
        self.height = height
        self.connectivity = connectivity
        self.words_per_row = (height + 31) // 32
        self.tiles_x = (width + TILE - 1) // TILE
        self.n_tiles = self.tiles_x * self.words_per_row
        n_ids = IDS_PER_TILE * self.n_tiles
        if REC * n_ids >= 2 ** 31:
            raise ValueError(
                f"image too large: {self.n_tiles} tiles need {REC * n_ids} "
                f"record words, over the int32 index range")
        self.lattice = int(lattice)
        self.valve_round = int(valve_round)
        self.lookup = lookup
        self.index = lookup == "index"
        self.hybrid = lookup == "hybrid"
        self.dindex = lookup == "dindex"
        # the dense K2 writes the index for the lookups that read it
        self.pindex = self.dense and (self.index or self.dindex)
        self.dense_min = int(dense_min)
        if self.dense_min < 0:
            raise ValueError("dense_min must be >= 0")
        if label_max_registers == "auto":
            label_max_registers = LABEL_MAX_REGISTERS[lookup]
        if waves_max_registers == "auto":
            waves_max_registers = WAVES_MAX_REGISTERS[lookup]
        self.waves_max_registers = waves_max_registers
        self.label_exits = int(label_exits)
        self.instrumented = bool(instrumented)
        self.debug = bool(debug)
        self.paint_fast = bool(paint_fast)
        self.pack_variant = pack_variant
        self.label_grid = tuple(label_grid)
        self.paint_grid = tuple(paint_grid)
        if self.fit_grids:
            # one warp per tile at most: blocks past that only exit
            self.label_grid = _fit(self.label_grid, self.n_tiles)
            self.paint_grid = _fit(self.paint_grid, self.n_tiles)
        self.label_wpb = self.label_grid[1] // 32
        self.paint_wpb = self.paint_grid[1] // 32
        self.waves_tpb = int(waves_tpb)
        self.label_max_registers = label_max_registers
        self.label_dense_max_registers = label_dense_max_registers
        self.coll_capacity = int(COLL_PER_TILE * self.n_tiles
                                 if coll_capacity is None else coll_capacity)
        if self.coll_capacity < 0:
            raise ValueError("coll_capacity must be >= 0")

        self.pack_k = get_pack_kernel(pack_variant)
        self.label_k = get_label_kernel(connectivity, self.instrumented,
                                        self.debug, label_max_registers,
                                        self.index, self.label_exits,
                                        self.dense)
        # the dindex K3: one twin, or (auto) the compact and the plain
        # twin, each launched with its own compact_min guard
        self.waves_ks = []
        if self.waves_compact in (True, "auto"):
            self.waves_ks.append(
                (_waves_twin(connectivity, self.instrumented, self.debug,
                             lookup, waves_max_registers, True,
                             self.flat_opt, self.waves_tpb, self.waves_coop),
                 self.compact_min if self.waves_compact == "auto" else 0))
        if self.waves_compact in (False, "auto"):
            self.waves_ks.append(
                (_waves_twin(connectivity, self.instrumented, self.debug,
                             lookup, waves_max_registers, False,
                             self.flat_opt, self.waves_tpb, self.waves_coop),
                 self.compact_min if self.waves_compact == "auto"
                 else 1 << 30))
        self.waves_k = self.waves_ks[0][0]
        self.paint_k = get_paint_kernel(connectivity, MODE_RGB,
                                        self.paint_fast, self.instrumented,
                                        self.debug, self.dense)
        self.lmap_k = get_paint_kernel(connectivity, MODE_LABEL, False,
                                       False, False, self.dense)
        self.seeds_k = get_paint_kernel(connectivity, MODE_SEEDS, False,
                                        False, False, self.dense)
        if self.dense:
            self.label_dense_k = get_label_dense_kernel(
                connectivity, self.instrumented, self.debug,
                label_dense_max_registers, self.label_wpb, self.pindex)
            self.paint_dense_k = get_paint_dense_kernel(
                connectivity, MODE_RGB, self.paint_fast, self.instrumented,
                self.debug, self.paint_wpb)
            self.lmap_dense_k = get_paint_dense_kernel(
                connectivity, MODE_LABEL, False, False, False,
                self.paint_wpb)
            self.seeds_dense_k = get_paint_dense_kernel(
                connectivity, MODE_SEEDS, False, False, False,
                self.paint_wpb)

        self.mask = cuda.device_array((width, self.words_per_row), np.uint32)
        self._mask_flat = self.mask.reshape(width * self.words_per_row)
        self.hdr = cuda.device_array(self.n_tiles, np.int32)
        self.rec = cuda.device_array(REC * n_ids, np.int32)
        self.parent = cuda.device_array(n_ids, np.int32)
        self.qc = cuda.device_array(n_ids, np.int32)
        self.ql = cuda.device_array(n_ids, np.int32)
        self.qd = cuda.device_array(n_ids, np.int32)
        # the cap is passed to K3 as its own argument, so a cap of 0
        # (every collision an inline union) still gets a 1-slot array
        cc = max(1, self.coll_capacity)
        self.ca = cuda.device_array(cc, np.int32)
        self.cb = cuda.device_array(cc, np.int32)
        self.q_state = cuda.device_array(N_QSTATE, np.int32)
        # dense: K2 -> K2b and K4 -> K4b tile lists (one buffer: K2b is
        # done before K4 appends), the K4 count in q_state[Q_PDENSE]; the
        # off-clock twins count in their own zeroed slot
        self.tlist = cuda.device_array(self.n_tiles if self.dense else 1,
                                       np.int32)
        self._pcount = self.q_state[Q_PDENSE:Q_PDENSE + 1]
        self._pcount_off = cuda.device_array(1, np.int32)
        self.counters = cuda.device_array(NUM_COUNTERS, np.int64)
        # per-round rears (instrumented): rounds <= crossing comps + 1,
        # and <= valve_round + 2 with the valve on; a run may switch the
        # valve off, so size for that (rounds past the end are not kept)
        nb = min(n_ids + 2, 1 << 16) if self.instrumented else 1
        self.bounds = cuda.device_array(nb, np.int32)
        self.stamps = cuda.device_array(N_STAMPS if self.instrumented else 1,
                                        np.int64)
        if self.index or self.dindex:
            self.idx = cuda.device_array(INDEX_BYTES * self.n_tiles, np.uint8)
            self.frame = cuda.device_array(4 * self.n_tiles, np.uint32)
        else:
            self.idx = cuda.device_array(1, np.uint8)
            self.frame = cuda.device_array(1, np.uint32)
        self._dummy_img = cuda.device_array((1, 1, 3), np.uint8)
        self._dummy_lmap = cuda.device_array((1, 1), np.int32)
        self._dummy_seeds = cuda.device_array(1, np.int32)
        self._seed_n = cuda.device_array(1, np.int32)
        self.pack_grid, self.pack_tpb = pack_shape(width, self.words_per_row)

        # K2b and K4b grid-stride over a tile list whose length only the
        # device knows: launched at most at their resident capacity
        # (with fit_grids), so an empty list costs one wave of blocks
        self.label_dense_grid = self.label_grid
        self.paint_dense_grid = self.paint_grid
        if _skip_warmup:
            self.waves_blocks = 1 if waves_blocks is None else int(waves_blocks)
            self.coop_max_blocks = -1
            return
        _warmup(connectivity, self.instrumented, self.debug,
                label_max_registers, self.paint_fast, pack_variant,
                lookup, self.label_exits, waves_max_registers, self.dense,
                self.flat_opt, self.label_wpb, self.paint_wpb,
                self.waves_compact, label_dense_max_registers)
        if self.dense and self.fit_grids:
            for name, grid in (("label_dense_grid", self.label_grid),
                               ("paint_dense_grid", self.paint_grid)):
                kern = (self.label_dense_k if name == "label_dense_grid"
                        else self.paint_dense_k)
                res = _overload(kern).max_cooperative_grid_blocks(grid[1])
                setattr(self, name, (max(1, min(grid[0], res)), grid[1]))
        cap = min(max_blocks(self.waves_tpb, connectivity, self.instrumented,
                             self.debug, lookup, waves_max_registers,
                             self.flat_opt, cm, self.waves_coop)
                  for cm in ((True, False) if self.waves_compact == "auto"
                             else (bool(self.waves_compact),)))
        if cap < 1:
            raise RuntimeError("this GPU cannot host one cooperative block "
                               f"of the waves kernel at {waves_tpb} threads")
        if waves_blocks is None:
            self.waves_blocks = cap
        else:
            if waves_blocks < 1 or waves_blocks > cap:
                raise RuntimeError(
                    f"waves_blocks={waves_blocks} is outside [1, {cap}] at "
                    f"{waves_tpb} threads (grid.sync would deadlock)")
            self.waves_blocks = int(waves_blocks)
        self.coop_max_blocks = cap

    # ---------------------------------------------------------- pieces
    def pack(self, img_dev):
        """RGB -> mask (the mask contract's input; untimed helper)."""
        self.pack_k[self.pack_grid, self.pack_tpb](img_dev, self.mask)

    def label_pass(self, lattice=None):
        """K0 + K2 only, from the current mask (unit tests: the records
        before any wave touches them). No sync."""
        lat = self.lattice if lattice is None else int(lattice)
        init_kernel[1, 64](self.q_state, self.counters)
        for _, launch in self._label_launches(np.int32(lat)):
            launch()

    def _label_launches(self, lat):
        """K2 (and K2b when dense) as (phase name, launch) pairs."""
        w, h = np.int32(self.width), np.int32(self.height)
        wpr, nt = np.int32(self.words_per_row), np.int32(self.n_tiles)

        def label():
            self.label_k[self.label_grid[0], self.label_grid[1]](
                self._mask_flat, self.hdr, self.rec, self.parent, self.qc,
                self.ql, self.qd, self.q_state, self.counters, self.idx,
                self.frame, self.tlist, w, h, wpr, nt, lat,
                np.int32(self.ccl_min), np.int32(self.ccl_rpc))

        def label_ccl():
            self.label_dense_k[self.label_dense_grid[0],
                               self.label_dense_grid[1]](
                self._mask_flat, self.hdr, self.rec, self.parent, self.qc,
                self.ql, self.qd, self.q_state, self.counters, self.idx,
                self.frame, self.tlist, w, h, wpr, nt, lat)

        out = [("label", label)]
        if self.dense:
            out.append(("label_ccl", label_ccl))
        return out

    def _paint_launches(self, kern, kern_dense, img, lmap, seeds, seed_n,
                        pcount):
        """K4 (and K4b when dense) of one mode as (name, launch) pairs."""
        w, h = np.int32(self.width), np.int32(self.height)
        wpr, nt = np.int32(self.words_per_row), np.int32(self.n_tiles)

        def paint():
            kern[self.paint_grid[0], self.paint_grid[1]](
                self._mask_flat, self.hdr, self.rec, img, lmap, seeds,
                seed_n, self.counters, self.tlist, pcount, w, h, wpr, nt,
                np.int32(self.paint_ccl_min), np.int32(self.paint_rpc))

        def paint_ccl():
            kern_dense[self.paint_dense_grid[0], self.paint_dense_grid[1]](
                self._mask_flat, self.hdr, self.rec, img, lmap, seeds,
                seed_n, self.counters, self.tlist, pcount, w, h, wpr, nt)

        out = [("paint", paint)]
        if self.dense:
            out.append(("paint_ccl", paint_ccl))
        return out

    def stages(self, img_dev, contract="rgb", lattice=None,
               valve_round=None, paint=True):
        """The pipeline as (phase name, launch) pairs in stream order;
        nothing is enqueued until a launch is called. The benchmark times
        whole pipelines, single phases and the label-only part (paint
        False, mask contract) from this one list.

        lattice and valve_round override the engine's values for this run
        only (both are runtime arguments, so one engine serves every seed
        rule and valve setting; valve_round 0 is "off")."""
        if contract not in CONTRACTS:
            raise ValueError(f"contract must be one of {CONTRACTS}")
        lat = np.int32(self.lattice if lattice is None else lattice)
        valve = np.int32(self.valve_round if valve_round is None
                         else valve_round)
        w, h = np.int32(self.width), np.int32(self.height)
        wpr, nt = np.int32(self.words_per_row), np.int32(self.n_tiles)

        def pack():
            self.pack_k[self.pack_grid, self.pack_tpb](img_dev, self.mask)

        def init():
            init_kernel[1, 64](self.q_state, self.counters)

        def waves():
            args = (self.hdr, self.rec, self.parent, self.qc, self.ql,
                    self.qd, self.ca, self.cb, self.q_state, self.counters,
                    self.bounds, self.stamps, self.idx, self.frame, wpr, nt,
                    valve, np.int32(self.coll_capacity),
                    np.int32(self.dense_min))
            if self.dindex:
                for kern, cmin in self.waves_ks:
                    kern[self.waves_blocks, self.waves_tpb](*args,
                                                            np.int32(cmin))
            else:
                self.waves_k[self.waves_blocks, self.waves_tpb](*args)

        out = [("pack", pack)] if contract == "rgb" else []
        out += [("init", init)] + self._label_launches(lat)
        out.append(("waves", waves))
        if paint:
            out += self._paint_launches(
                self.paint_k, getattr(self, "paint_dense_k", None), img_dev,
                self._dummy_lmap, self._dummy_seeds, self._seed_n,
                self._pcount)
        return out

    def run(self, img_dev, contract="rgb", lattice=None, valve_round=None):
        """Enqueue the pipeline with a CUDA event after each phase;
        returns (phase names, events). No sync."""
        st = self.stages(img_dev, contract, lattice, valve_round)
        events = [cuda.event(timing=True) for _ in range(len(st) + 1)]
        events[0].record()
        for i, (_, launch) in enumerate(st):
            launch()
            events[i + 1].record()
        return [n for n, _ in st], events

    # ------------------------------------------------- off-clock twins
    def emit_label_map(self, label_dev):
        """int32 (W, H) canonical label map, -1 off-blob (off the clock)."""
        flat = label_dev.reshape(self.width * self.height)
        fill_i32[256, 256](flat, np.int32(-1))
        fill_i32[1, 32](self._pcount_off, np.int32(0))
        for _, launch in self._paint_launches(
                self.lmap_k, getattr(self, "lmap_dense_k", None),
                self._dummy_img, label_dev, self._dummy_seeds, self._seed_n,
                self._pcount_off):
            launch()

    def emit_seeds(self, capacity=None):
        """Canonical seeds (x, y), sorted by label: the anchor of every
        comp whose final label equals its own anchor, one per blob."""
        if capacity is None:
            capacity = max(1, self.n_blobs())
        buf = cuda.device_array(capacity, np.int32)
        n_dev = cuda.to_device(np.zeros(1, np.int32))
        fill_i32[1, 32](self._pcount_off, np.int32(0))
        for _, launch in self._paint_launches(
                self.seeds_k, getattr(self, "seeds_dense_k", None),
                self._dummy_img, self._dummy_lmap, buf, n_dev,
                self._pcount_off):
            launch()
        n = int(n_dev.copy_to_host()[0])
        if n > capacity:
            raise RuntimeError(f"{n} seeds found, more than the {capacity} "
                               f"blobs counted: labels are inconsistent")
        lins = np.sort(buf.copy_to_host()[:n].astype(np.int64))
        return [(int(v) // self.height, int(v) % self.height) for v in lins]

    def read_counters(self):
        c = self.counters.copy_to_host()
        out = {name: int(c[i]) for i, name in enumerate(COUNTER_NAMES)}
        out["n_blobs"] = out["roots"] + out["interior_comps"]
        return out

    def n_blobs(self):
        return self.read_counters()["n_blobs"]

    def read_q_state(self):
        return self.q_state.copy_to_host()

    def round_stats(self):
        """Per round (instrumented twin): frontier comps, distinct
        frontier tiles, and %globaltimer splits. Off the clock."""
        if not self.instrumented:
            raise RuntimeError("round_stats needs the instrumented twin")
        rounds = self.read_counters()["rounds"]
        bounds = self.bounds.copy_to_host()
        stamps = self.stamps.copy_to_host()
        # bounds[0] = the seed queue's rear, bounds[r] = the rear after
        # round r; round r processes [bounds[r-2], bounds[r-1]) with
        # bounds[-1] = 0 (the seeds are round 1's frontier)
        nb = min(rounds, bounds.size)
        b = [0] + [int(v) for v in bounds[:nb]]
        rear = b[-1]
        qc = self.qc[:max(rear, 1)].copy_to_host()[:rear]
        comps = []
        tiles = []
        for r in range(1, nb + 1):
            seg = qc[b[r - 1]:b[r]]
            comps.append(int(seg.size))
            tiles.append(int(np.unique(seg >> 6).size))
        # rounds 1 .. ST_TAIL-1 have a stamp each; rounds from ST_TAIL on
        # share one bucket (the slot ST_TAIL holds the last round's end)
        times = []
        ns = min(rounds, ST_TAIL - 1)
        for r in range(1, ns + 1):
            times.append((int(stamps[r]) - int(stamps[r - 1])) / 1000.0)
        tail_us = 0.0
        last = int(stamps[ns]) if ns else int(stamps[0])
        if rounds >= ST_TAIL:
            tail_us = (int(stamps[ST_TAIL]) - last) / 1000.0
            last = int(stamps[ST_TAIL])
        return {
            "rounds": rounds,
            "frontier_comps": comps,
            "frontier_tiles": tiles,
            "tile_visits": int(sum(tiles)),
            "round_us": times,
            "rounds_past_stamps": max(0, rounds - ns),
            "rounds_past_stamps_us": tail_us,
            "union_us": (int(stamps[ST_UNION]) - last) / 1000.0,
            "flatten1_us": (int(stamps[ST_FLAT1])
                            - int(stamps[ST_UNION])) / 1000.0,
            "flatten2_us": (int(stamps[ST_FLAT2])
                            - int(stamps[ST_FLAT1])) / 1000.0,
            "waves_us_total": (int(stamps[ST_FLAT2])
                               - int(stamps[0])) / 1000.0,
        }

    def poison(self, value=0x5A5A5A5A):
        """Fill every reusable buffer with garbage (engine-reuse tests)."""
        v = np.int32(np.uint32(value).view(np.int32))
        for a in (self.hdr, self.rec, self.parent, self.qc, self.ql,
                  self.qd, self.ca, self.cb):
            fill_i32[256, 256](a, v)
        if self.index or self.dindex:
            fill_i32[256, 256](self.idx, np.int32(0x5A))
            fill_i32[256, 256](self.frame, v)


def phase_times(names, events):
    """Event list -> (per-phase ms dict, total ms)."""
    ms = {name: cuda.event_elapsed_time(events[i], events[i + 1])
          for i, name in enumerate(names)}
    return ms, cuda.event_elapsed_time(events[0], events[-1])


def _waves_twin(connectivity, instrumented, debug, lookup,
                waves_max_registers, compact, flat_opt, tpb=WAVES_TPB,
                coop=DEFAULT_COOP):
    if lookup == "dindex":
        return get_waves_dense_kernel(connectivity, instrumented, debug,
                                      waves_max_registers, compact, flat_opt,
                                      tpb // 32, coop)
    return get_waves_kernel(connectivity, instrumented, debug,
                            lookup == "index", lookup == "hybrid",
                            waves_max_registers)


def max_blocks(threads_per_block=WAVES_TPB, connectivity=8,
               instrumented=False, debug=False, lookup=DEFAULT_LOOKUP,
               waves_max_registers="auto", flat_opt=DEFAULT_FLAT_OPT,
               compact=DEFAULT_COMPACT, coop=DEFAULT_COOP):
    """Cooperative capacity of the waves kernel twin, queried after its
    first compile (an uncompiled dispatcher has no overload)."""
    if waves_max_registers == "auto":
        waves_max_registers = WAVES_MAX_REGISTERS[lookup]
    if lookup != "dindex":
        flat_opt = compact = False
        coop = True
    key = (threads_per_block, connectivity, bool(instrumented), bool(debug),
           lookup, waves_max_registers, bool(flat_opt), bool(compact),
           bool(coop))
    if key not in _coop:
        _warmup(connectivity, instrumented, debug, lookup=lookup,
                waves_max_registers=waves_max_registers,
                dense=lookup == "dindex" or DEFAULT_DENSE,
                flat_opt=flat_opt, compact=compact)
        kern = _waves_twin(connectivity, instrumented, debug, lookup,
                           waves_max_registers, compact, flat_opt,
                           threads_per_block, coop)
        if not kern.overloads:
            # the dindex twin of this block size: compile it on a tiny run
            eng = TileWaves(40, 40, connectivity=connectivity,
                            instrumented=instrumented, debug=debug,
                            lookup=lookup, dense=lookup == "dindex",
                            waves_max_registers=waves_max_registers,
                            waves_tpb=threads_per_block, waves_blocks=1,
                            flat_opt=flat_opt, waves_compact=compact,
                            waves_coop=coop, _skip_warmup=True)
            eng.run(cuda.to_device(np.zeros((40, 40, 3), np.uint8)))
            cuda.synchronize()
        _coop[key] = _overload(kern).max_cooperative_grid_blocks(
            threads_per_block)
    return _coop[key]


def kernel_resources(connectivity=8, instrumented=False, debug=False,
                     label_max_registers="auto", paint_fast=True,
                     lookup=DEFAULT_LOOKUP, label_exits=2,
                     waves_max_registers="auto", dense=DEFAULT_DENSE,
                     flat_opt=DEFAULT_FLAT_OPT, compact=DEFAULT_COMPACT,
                     label_dense_max_registers=LABEL_DENSE_MAX_REGISTERS):
    """Registers and local (spill) bytes per thread of each kernel (the
    default grids' block sizes)."""
    if lookup != "dindex":
        flat_opt = compact = False
    if label_max_registers == "auto":
        label_max_registers = LABEL_MAX_REGISTERS[lookup]
    if waves_max_registers == "auto":
        waves_max_registers = WAVES_MAX_REGISTERS[lookup]
    lw, pw = LABEL_GRID[1] // 32, PAINT_GRID[1] // 32
    _warmup(connectivity, instrumented, debug, label_max_registers,
            paint_fast, lookup=lookup, label_exits=label_exits,
            waves_max_registers=waves_max_registers, dense=dense,
            flat_opt=flat_opt, label_wpb=lw, paint_wpb=pw, compact=compact,
            label_dense_max_registers=label_dense_max_registers)
    max_blocks(WAVES_TPB, connectivity, instrumented, debug, lookup,
               waves_max_registers, flat_opt, compact)
    index = lookup == "index"
    pindex = dense and lookup in ("index", "dindex")
    out = {}
    kerns = [
        ("label", get_label_kernel(connectivity, instrumented, debug,
                                   label_max_registers, index, label_exits,
                                   dense)),
        ("waves", _waves_twin(connectivity, instrumented, debug, lookup,
                              waves_max_registers, compact, flat_opt,
                              WAVES_TPB)),
        ("paint", get_paint_kernel(connectivity, MODE_RGB, paint_fast,
                                   instrumented, debug, dense))]
    if dense:
        kerns += [
            ("label_ccl", get_label_dense_kernel(
                connectivity, instrumented, debug, label_dense_max_registers,
                lw, pindex)),
            ("paint_ccl", get_paint_dense_kernel(
                connectivity, MODE_RGB, paint_fast, instrumented, debug,
                pw))]
    for name, kern in kerns:
        out[name] = {"regs": _one(kern.get_regs_per_thread()),
                     "local_bytes": _one(kern.get_local_mem_per_thread())}
    return out


def _warmup(connectivity=8, instrumented=False, debug=False,
            label_max_registers="auto", paint_fast=True,
            pack_variant="direct", lookup=DEFAULT_LOOKUP, label_exits=2,
            waves_max_registers="auto", dense=DEFAULT_DENSE,
            flat_opt=DEFAULT_FLAT_OPT, label_wpb=None, paint_wpb=None,
            compact=DEFAULT_COMPACT,
            label_dense_max_registers=LABEL_DENSE_MAX_REGISTERS):
    """JIT-compile every kernel of a configuration once, on a tiny image
    (with the given block sizes: the dense twins are compiled per warps
    per block, K2b also per register cap)."""
    if label_max_registers == "auto":
        label_max_registers = LABEL_MAX_REGISTERS[lookup]
    if waves_max_registers == "auto":
        waves_max_registers = WAVES_MAX_REGISTERS[lookup]
    lw = LABEL_GRID[1] // 32 if label_wpb is None else int(label_wpb)
    pw = PAINT_GRID[1] // 32 if paint_wpb is None else int(paint_wpb)
    key = (connectivity, bool(instrumented), bool(debug),
           label_max_registers, bool(paint_fast), pack_variant, lookup,
           int(label_exits), waves_max_registers, bool(dense),
           bool(flat_opt) and lookup == "dindex",
           bool(compact) and lookup == "dindex", lw if dense else 0,
           pw if dense else 0,
           label_dense_max_registers if dense else None)
    if key in _warmed:
        return
    tiny = np.full((40, 40, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    tiny[30:34, 30:34] = (255, 0, 0)
    tiny[5:8, 3] = (255, 0, 0)
    eng = TileWaves(40, 40, connectivity=connectivity,
                    instrumented=instrumented, debug=debug,
                    label_max_registers=label_max_registers,
                    paint_fast=paint_fast, pack_variant=pack_variant,
                    lookup=lookup, label_exits=label_exits,
                    waves_max_registers=waves_max_registers,
                    waves_tpb=32, waves_blocks=1, label_grid=(1, 32 * lw),
                    paint_grid=(1, 32 * pw), dense=dense, flat_opt=flat_opt,
                    waves_compact=compact, fit_grids=False,
                    label_dense_max_registers=label_dense_max_registers,
                    _skip_warmup=True)
    dev = cuda.to_device(tiny)
    eng.run(dev, "rgb")
    lab = cuda.device_array((40, 40), np.int32)
    eng.emit_label_map(lab)
    cuda.synchronize()
    eng.emit_seeds(capacity=8)
    gpu_sleep_kernel[1, 32](1)
    cuda.synchronize()
    _warmed.add(key)


def recolor(img_host, contract="rgb", emit_label=False, emit_seeds=True,
            copy_img=True, engine=None, lattice=None, **engine_kw):
    """Discover, label and recolour every red blob by seeded tile waves.

    img_host: (width, height, 3) uint8, not modified (a painted copy is
    returned when copy_img). Blank images are valid (n_blobs = 0).
    contract "rgb" times the pack; "mask" packs off the clock.
    engine: reuse a TileWaves of the same shape; else one is built from
    engine_kw (connectivity, lattice, valve_round, ...).
    """
    if (img_host.ndim != 3 or img_host.shape[2] != 3
            or img_host.dtype != np.uint8):
        raise ValueError("img must be a (width, height, 3) uint8 array")
    if contract not in CONTRACTS:
        raise ValueError(f"contract must be one of {CONTRACTS}")
    width, height = img_host.shape[0], img_host.shape[1]
    t0 = time.perf_counter()
    eng = engine if engine is not None else TileWaves(width, height,
                                                      **engine_kw)
    if (eng.width, eng.height) != (width, height):
        raise ValueError(f"engine is {eng.width}x{eng.height}, image is "
                         f"{width}x{height}")
    img_dev = cuda.to_device(img_host)
    if contract == "mask":
        eng.pack(img_dev)
    cuda.synchronize()
    t1 = time.perf_counter()
    gpu_sleep_kernel[1, 32](20)       # keep Python's launch cost off events
    names, events = eng.run(img_dev, contract, lattice=lattice)
    events[-1].synchronize()
    t2 = time.perf_counter()
    phase_ms, kernel_ms = phase_times(names, events)
    counters = eng.read_counters()
    label_out = np.zeros((0, 0), np.int32)
    if emit_label:
        label_dev = cuda.device_array((width, height), np.int32)
        eng.emit_label_map(label_dev)
        label_out = label_dev.copy_to_host()
    seeds = eng.emit_seeds() if emit_seeds else []
    img_out = img_dev.copy_to_host() if copy_img else np.zeros(
        (0, 0, 3), np.uint8)
    t3 = time.perf_counter()
    return TileWavesResult(
        img=img_out, label=label_out, contract=contract,
        n_blobs=counters["n_blobs"], seeds=seeds, counters=counters,
        rounds=counters["rounds"], phase_ms=phase_ms, kernel_ms=kernel_ms,
        label_only_ms=sum(phase_ms[p] for p in LABEL_ONLY_PHASES),
        alloc_ms=(t1 - t0) * 1000, d2h_ms=(t3 - t2) * 1000,
        total_ms=(t3 - t0) * 1000, waves_blocks=eng.waves_blocks,
        waves_tpb=eng.waves_tpb)
