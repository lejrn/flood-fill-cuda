"""Host driver for the seeded tile waves (ch08).

Public API (mirrors ch06's recolor.py):
    TileWaves(width, height, connectivity=8, lattice=8, valve_round=128,
              lookup="scan", coll_capacity=None, instrumented=False,
              debug=False, ...)                       reusable buffers
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
    get_paint_kernel, gpu_sleep_kernel, fill_i32, pack_shape,
    MODE_RGB, MODE_LABEL, MODE_SEEDS, NUM_COUNTERS, COUNTER_NAMES,
    N_QSTATE, N_STAMPS, IDS_PER_TILE, REC, TILE, HDR_RING_MASK,
    HDR_TOTAL_SHIFT, ST_UNION, ST_FLAT1, ST_FLAT2, INDEX_BYTES)

CONTRACTS = ("rgb", "mask")
LOOKUPS = ("scan", "index")
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
LABEL_MAX_REGISTERS = {"scan": 80, "index": None}
PAINT_GRID = (768, 128)
WAVES_TPB = 256
DEFAULT_LATTICE = 8
DEFAULT_VALVE_ROUND = 128
# collision list soft cap per tile; entries past it become inline unions
COLL_PER_TILE = 16

_warmed = set()
_coop = {}


def _overload(kern):
    return next(iter(kern.overloads.values()))


def _one(v):
    return int(next(iter(v.values()))) if isinstance(v, dict) else int(v)


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
                 lookup="scan", coll_capacity=None, instrumented=False,
                 debug=False, waves_tpb=WAVES_TPB, waves_blocks=None,
                 label_grid=LABEL_GRID, paint_grid=PAINT_GRID,
                 label_max_registers="auto", paint_fast=True,
                 pack_variant="direct", label_exits=2, _skip_warmup=False):
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
        if label_max_registers == "auto":
            label_max_registers = LABEL_MAX_REGISTERS[lookup]
        self.label_exits = int(label_exits)
        self.instrumented = bool(instrumented)
        self.debug = bool(debug)
        self.paint_fast = bool(paint_fast)
        self.pack_variant = pack_variant
        self.label_grid = tuple(label_grid)
        self.paint_grid = tuple(paint_grid)
        self.waves_tpb = int(waves_tpb)
        self.label_max_registers = label_max_registers
        self.coll_capacity = int(COLL_PER_TILE * self.n_tiles
                                 if coll_capacity is None else coll_capacity)
        if self.coll_capacity < 0:
            raise ValueError("coll_capacity must be >= 0")

        self.pack_k = get_pack_kernel(pack_variant)
        self.label_k = get_label_kernel(connectivity, self.instrumented,
                                        self.debug, label_max_registers,
                                        self.index, self.label_exits)
        self.waves_k = get_waves_kernel(connectivity, self.instrumented,
                                        self.debug, self.index)
        self.paint_k = get_paint_kernel(connectivity, MODE_RGB,
                                        self.paint_fast, self.instrumented,
                                        self.debug)
        self.lmap_k = get_paint_kernel(connectivity, MODE_LABEL, False,
                                       False, False)
        self.seeds_k = get_paint_kernel(connectivity, MODE_SEEDS, False,
                                        False, False)

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
        self.counters = cuda.device_array(NUM_COUNTERS, np.int64)
        # per-round rears (instrumented): rounds <= crossing comps + 1,
        # and <= valve_round + 2 with the valve on; a run may switch the
        # valve off, so size for that (rounds past the end are not kept)
        nb = min(n_ids + 2, 1 << 16) if self.instrumented else 1
        self.bounds = cuda.device_array(nb, np.int32)
        self.stamps = cuda.device_array(N_STAMPS if self.instrumented else 1,
                                        np.int64)
        if self.index:
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

        if _skip_warmup:
            self.waves_blocks = 1 if waves_blocks is None else int(waves_blocks)
            self.coop_max_blocks = -1
            return
        _warmup(connectivity, self.instrumented, self.debug,
                label_max_registers, self.paint_fast, pack_variant,
                lookup, self.label_exits)
        cap = max_blocks(self.waves_tpb, connectivity, self.instrumented,
                         self.debug, lookup)
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
        self.label_k[self.label_grid[0], self.label_grid[1]](
            self._mask_flat, self.hdr, self.rec, self.parent, self.qc,
            self.ql, self.qd, self.q_state, self.counters, self.idx,
            self.frame, np.int32(self.width), np.int32(self.height),
            np.int32(self.words_per_row), np.int32(self.n_tiles),
            np.int32(lat))

    def run(self, img_dev, contract="rgb", lattice=None, valve_round=None):
        """Enqueue the pipeline; returns (phase names, events). No sync.

        lattice and valve_round override the engine's values for this run
        only (both are runtime arguments, so one engine serves every seed
        rule and valve setting; valve_round 0 is "off")."""
        if contract not in CONTRACTS:
            raise ValueError(f"contract must be one of {CONTRACTS}")
        lat = self.lattice if lattice is None else int(lattice)
        valve = self.valve_round if valve_round is None else int(valve_round)
        names = PHASES if contract == "rgb" else MASK_PHASES
        events = [cuda.event(timing=True) for _ in range(len(names) + 1)]
        i = 0
        events[i].record()
        if contract == "rgb":
            self.pack_k[self.pack_grid, self.pack_tpb](img_dev, self.mask)
            i += 1
            events[i].record()
        init_kernel[1, 64](self.q_state, self.counters)
        i += 1
        events[i].record()
        self.label_k[self.label_grid[0], self.label_grid[1]](
            self._mask_flat, self.hdr, self.rec, self.parent, self.qc,
            self.ql, self.qd, self.q_state, self.counters, self.idx,
            self.frame, np.int32(self.width), np.int32(self.height),
            np.int32(self.words_per_row), np.int32(self.n_tiles),
            np.int32(lat))
        i += 1
        events[i].record()
        self.waves_k[self.waves_blocks, self.waves_tpb](
            self.hdr, self.rec, self.parent, self.qc, self.ql, self.qd,
            self.ca, self.cb, self.q_state, self.counters, self.bounds,
            self.stamps, self.idx, self.frame, np.int32(self.words_per_row),
            np.int32(self.n_tiles), np.int32(valve),
            np.int32(self.coll_capacity))
        i += 1
        events[i].record()
        self.paint_k[self.paint_grid[0], self.paint_grid[1]](
            self._mask_flat, self.hdr, self.rec, img_dev, self._dummy_lmap,
            self._dummy_seeds, self._seed_n, self.counters,
            np.int32(self.width), np.int32(self.height),
            np.int32(self.words_per_row), np.int32(self.n_tiles))
        i += 1
        events[i].record()
        return names, events

    # ------------------------------------------------- off-clock twins
    def emit_label_map(self, label_dev):
        """int32 (W, H) canonical label map, -1 off-blob (off the clock)."""
        flat = label_dev.reshape(self.width * self.height)
        fill_i32[256, 256](flat, np.int32(-1))
        self.lmap_k[self.paint_grid[0], self.paint_grid[1]](
            self._mask_flat, self.hdr, self.rec, self._dummy_img, label_dev,
            self._dummy_seeds, self._seed_n, self.counters,
            np.int32(self.width), np.int32(self.height),
            np.int32(self.words_per_row), np.int32(self.n_tiles))

    def emit_seeds(self, capacity=None):
        """Canonical seeds (x, y), sorted by label: the anchor of every
        comp whose final label equals its own anchor, one per blob."""
        if capacity is None:
            capacity = max(1, self.n_blobs())
        buf = cuda.device_array(capacity, np.int32)
        n_dev = cuda.to_device(np.zeros(1, np.int32))
        self.seeds_k[self.paint_grid[0], self.paint_grid[1]](
            self._mask_flat, self.hdr, self.rec, self._dummy_img,
            self._dummy_lmap, buf, n_dev, self.counters,
            np.int32(self.width), np.int32(self.height),
            np.int32(self.words_per_row), np.int32(self.n_tiles))
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
        times = []
        ns = min(rounds, N_STAMPS - 4)
        for r in range(1, ns + 1):
            times.append((int(stamps[r]) - int(stamps[r - 1])) / 1000.0)
        last = int(stamps[ns]) if ns else int(stamps[0])
        return {
            "rounds": rounds,
            "frontier_comps": comps,
            "frontier_tiles": tiles,
            "tile_visits": int(sum(tiles)),
            "round_us": times,
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
        if self.index:
            fill_i32[256, 256](self.idx, np.int32(0x5A))
            fill_i32[256, 256](self.frame, v)


def phase_times(names, events):
    """Event list -> (per-phase ms dict, total ms)."""
    ms = {name: cuda.event_elapsed_time(events[i], events[i + 1])
          for i, name in enumerate(names)}
    return ms, cuda.event_elapsed_time(events[0], events[-1])


def max_blocks(threads_per_block=WAVES_TPB, connectivity=8,
               instrumented=False, debug=False, lookup="scan"):
    """Cooperative capacity of the waves kernel twin, queried after its
    first compile (an uncompiled dispatcher has no overload)."""
    key = (threads_per_block, connectivity, bool(instrumented), bool(debug),
           lookup)
    if key not in _coop:
        _warmup(connectivity, instrumented, debug, lookup=lookup)
        kern = get_waves_kernel(connectivity, instrumented, debug,
                                lookup == "index")
        _coop[key] = _overload(kern).max_cooperative_grid_blocks(
            threads_per_block)
    return _coop[key]


def kernel_resources(connectivity=8, instrumented=False, debug=False,
                     label_max_registers="auto", paint_fast=True,
                     lookup="scan", label_exits=2):
    """Registers and spill bytes per thread of each compiled kernel."""
    if label_max_registers == "auto":
        label_max_registers = LABEL_MAX_REGISTERS[lookup]
    _warmup(connectivity, instrumented, debug, label_max_registers,
            paint_fast, lookup=lookup, label_exits=label_exits)
    index = lookup == "index"
    out = {}
    for name, kern in (
            ("label", get_label_kernel(connectivity, instrumented, debug,
                                       label_max_registers, index,
                                       label_exits)),
            ("waves", get_waves_kernel(connectivity, instrumented, debug,
                                       index)),
            ("paint", get_paint_kernel(connectivity, MODE_RGB, paint_fast,
                                       instrumented, debug))):
        out[name] = {"regs": _one(kern.get_regs_per_thread()),
                     "local_bytes": _one(kern.get_local_mem_per_thread())}
    return out


def _warmup(connectivity=8, instrumented=False, debug=False,
            label_max_registers="auto", paint_fast=True,
            pack_variant="direct", lookup="scan", label_exits=2):
    """JIT-compile every kernel of a configuration once, on a tiny image."""
    if label_max_registers == "auto":
        label_max_registers = LABEL_MAX_REGISTERS[lookup]
    key = (connectivity, bool(instrumented), bool(debug),
           label_max_registers, bool(paint_fast), pack_variant, lookup,
           int(label_exits))
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
                    waves_tpb=32, waves_blocks=1, label_grid=(1, 32),
                    paint_grid=(1, 32), _skip_warmup=True)
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
