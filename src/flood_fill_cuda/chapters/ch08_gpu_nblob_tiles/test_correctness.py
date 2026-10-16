"""Chapter 8 correctness: seeded tile waves against the CPU oracles.

The output contract is ch05's and ch06's, bit for bit:

    label map      every red pixel's canonical label (its blob's min lin
                   x*height + y), -1 off-blob
    painted image  palette[label % 6] on red, untouched elsewhere
    n_blobs        the component count
    seeds          one canonical (lex-min) pixel per blob, label order

plus the invariants of the algorithm (design section 11.2), checked on
every end-to-end case against an independent CPU model of the tiles and
the waves (cpu_model.py, built from pixels, not from side words):

    ring + interior == comps == the model's comp count
    claims + seeds + valve seeds == ring comps   (each claimed ONCE)
    links == seeds + valve seeds - roots          (each link retires a root)
    n_blobs == roots + interior comps == the oracle's
    rounds == the model's multi-source BFS rounds, exactly
    tasks == sum of popc(dirs) over the ring comps
    seeds == the model's seeds for the same rule
    debug twin: ordinal overflow, anchor mismatch, unlabelled all 0

and, for the dense-tile path (K2b's run CCL, K4b, the dindex K3),
against the model's per-tile run estimate:

    ccl tiles == the tiles the model sends to the CCL
    indexed tiles == those with 2+ ring comps (index and dindex lookups)
    dense tasks (dindex) == tasks served by the index or the whole warp

The unit tests come first (closure, records, pack), so a failure further
down can be localised.

Slow cases (the 6 benchmark scenes, 16 Mpx strips) run only with
FLOOD_FILL_SLOW=1; the PNG scenes also need FLOOD_FILL_IMAGES=<dir with
input_blobs.png and input_blocks.png>.

Run:
    uv run pytest src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/test_correctness.py -q
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import functools
import tempfile

import numpy as np
import pytest
from numba import cuda, int32, uint32

from . import kernels as k
from . import scenes as S8
from .flood_fill import (TileWaves, recolor, max_blocks, DEFAULT_LOOKUP,
                         DEFAULT_DENSE_MIN, DEFAULT_LATTICE,
                         DEFAULT_VALVE_ROUND, DEFAULT_DENSE, DEFAULT_CCL_MIN,
                         DEFAULT_CCL_RPC, DEFAULT_COOP, OLD_PATH)
from .cpu_oracle import cpu_label, cpu_label_components, red_mask
from .cpu_model import tile_model
from ..ch05_gpu_nblob_nblock import scenes as S5
from ..ch05_gpu_nblob_nblock.kernels import PALETTE_HOST, N_PALETTE
from ..ch06_gpu_nblob_runs.kernels import pack_kernel
from ..ch06_gpu_nblob_runs.recolor import recolor as ch06_recolor
from ..ch07_gpu_1blob_tiles.flood_fill import plain_shape
from ..ch07_gpu_1blob_tiles.benchmarks import worst_cases as W7

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning"),
              pytest.mark.filterwarnings(
                  "ignore::numba.core.errors.NumbaPerformanceWarning")]

SLOW = os.environ.get("FLOOD_FILL_SLOW") == "1"
# the dense path on every touched tile: no comp floor, and up to 512 runs
# (the most a tile can hold) per estimated comp
ALL_CCL = dict(ccl_min=0, ccl_rpc=512, paint_ccl_min=0, paint_rpc=512)
IMAGES = os.environ.get("FLOOD_FILL_IMAGES")
SCRATCH = os.environ.get("FLOOD_FILL_SCRATCH", tempfile.gettempdir())

FULL = np.uint32(0xFFFFFFFF)


# ------------------------------------------------------------ helpers

def _stripes(width, height, step):
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    for y in range(0, height, step):
        img[:, y] = (255, 0, 0)
    return img


def expected_paint(img, label):
    out = img.copy()
    red = red_mask(img)
    out[red] = PALETTE_HOST[(label % N_PALETTE)[red]]
    return out


def expected_seeds(label, height):
    return [(int(v) // height, int(v) % height)
            for v in np.unique(label[label >= 0])]


# ch06's 20-scene battery, plus the ch07 worst cases and ch08's own
SCENES = {
    "two_squares": lambda: S5.two_squares_scene(64, 64, 20, 20, gap=4)[0],
    "two_disks": lambda: S5.two_disks_scene(80, 80, 15, gap=6)[0],
    "asym_squares": lambda: S5.asym_squares_scene(80, 80, 30, 8, gap=4)[0],
    "two_pixels": lambda: S5.two_pixels_scene(32, 32)[0],
    "u_shape": lambda: S5.u_shape_scene(40, 40)[0],
    "comb_80_teeth": lambda: S5.comb_scene(64, 200, teeth=80)[0],
    "serpentine": lambda: S5.serpentine_scene(64, 64)[0],
    "single_pixel": lambda: S5.single_pixel_scene(16, 16)[0],
    "full_red": lambda: S5.full_red_scene(48, 48)[0],
    "blank": lambda: S5.blank_scene(32, 32)[0],
    "blob_grid_100": lambda: S5.blob_grid_scene(200, 200, 10, 10, 12,
                                                gap=4)[0],
    "noise_030": lambda: S5.random_blobs_scene(120, 120, 0.30, 1)[0],
    "noise_045": lambda: S5.random_blobs_scene(100, 100, 0.45, 7)[0],
    "noise_010": lambda: S5.random_blobs_scene(150, 150, 0.10, 3)[0],
    "size_33x33": lambda: S5.random_blobs_scene(33, 33, 0.35, 2)[0],
    "size_31x97": lambda: S5.random_blobs_scene(31, 97, 0.35, 4)[0],
    "size_97x131": lambda: S5.blob_grid_scene(97, 131, 4, 5, 11, gap=5)[0],
    "one_row": lambda: _stripes(1, 64, 3),
    "one_col": lambda: _stripes(64, 1, 1),
    "stripes_1px": lambda: _stripes(40, 97, 2),
    # ch07 worst cases (single blobs that stress the comp graph)
    "zz_serp_1k": lambda: W7.zz_serpentine_scene()[0],
    "zz_lattice_1k": lambda: W7.zz_lattice_scene(1024, 1024)[0],
    "maze_1k": lambda: W7.maze_scene(512)[0],
    # ch08 adversarial
    "corner_main": lambda: S8.corner_joined_scene("main"),
    "corner_anti": lambda: S8.corner_joined_scene("anti"),
    "corner_main_px": lambda: S8.corner_joined_scene("main", solid=False),
    "corner_anti_px": lambda: S8.corner_joined_scene("anti", solid=False),
    "all_eight": S8.all_eight_scene,
    "frame_comb_31": lambda: S8.frame_comb_scene(31),
    "frame_comb_32": lambda: S8.frame_comb_scene(32),
    "frame_comb_33": lambda: S8.frame_comb_scene(33),
    "frame_comb_60": lambda: S8.frame_comb_scene(60),
    "frame_comb_62_c4": lambda: S8.frame_comb_scene(62, connectivity=4),
    "dot_lattice_256": lambda: S8.dot_lattice_scene((2, 3)),
    "checker_512_c4": lambda: S8.checker_tile_scene((2, 2)),
    "rmin_trap": S8.rmin_trap_scene,
    "lexmin_multi": S8.lexmin_multi_scene,
    "spiral": S8.spiral_scene,
    "diag_snake": lambda: S8.diag_snake_scene(13),
    "tile_corner_px": lambda: S8.tile_corners_scene(100, 100),
    "tile_corner_blocks": lambda: S8.tile_corners_scene(100, 100,
                                                        block=True),
    "size_1x1": lambda: S5.full_red_scene(1, 1)[0],
    "size_31x31_full": lambda: S5.full_red_scene(31, 31)[0],
    "size_1x97": lambda: S5.random_blobs_scene(1, 97, 0.5, 5)[0],
    "size_97x1": lambda: S5.random_blobs_scene(97, 1, 0.5, 6)[0],
    "size_63x95": lambda: S5.random_blobs_scene(63, 95, 0.4, 8)[0],
    "size_65x33": lambda: S5.random_blobs_scene(65, 33, 0.4, 9)[0],
    "full_red_100": lambda: S5.full_red_scene(100, 100)[0],
    "stair_decoy_3": lambda: S8.stair_scene(3),
}

# the end-to-end config matrix: one axis at a time from the base config
# (L8, the default valve and lookup, default collision cap, max blocks,
# 8-conn, rgb, fast paint), every case instrumented + debug so the
# invariants can be read
CONFIGS = {
    "base": {},
    "one_block": dict(waves_tpb=32, waves_blocks=1),
    "L0": dict(lattice=0),
    "L1": dict(lattice=1),
    "valve1": dict(valve_round=1),
    "coll0": dict(coll_capacity=0),
    "mask": dict(_contract="mask"),
    "slow_paint": dict(paint_fast=False),
    "c4": dict(connectivity=4),
    "c4_one_block_L0": dict(connectivity=4, lattice=0, waves_tpb=32,
                            waves_blocks=1),
    "scan": dict(lookup="scan"),
    "scan_one_block_L0": dict(lookup="scan", lattice=0, waves_tpb=32,
                              waves_blocks=1),
    "index": dict(lookup="index"),
    "index_one_block_L0": dict(lookup="index", lattice=0, waves_tpb=32,
                               waves_blocks=1),
    "index_c4_valve1": dict(lookup="index", connectivity=4, valve_round=1),
    "hybrid_dense8_uncapped_k3": dict(lookup="hybrid", dense_min=8,
                                      waves_max_registers=None),
    "hybrid_all_dense_one_block_L0": dict(lookup="hybrid", dense_min=0,
                                          lattice=0, waves_tpb=32,
                                          waves_blocks=1),
    "hybrid_c4_coll0_dense1": dict(lookup="hybrid", connectivity=4,
                                   coll_capacity=0, dense_min=1),
    "uncapped_k2": dict(label_max_registers=None, label_grid=(768, 128)),
    "exits_ch07": dict(label_exits=0),
    "exits_frontier": dict(label_exits=1),
    # the bare twins, which the benchmark times (outputs and the counters
    # they keep): every lookup, both connectivities
    "bare": dict(instrumented=False, debug=False),
    "bare_c4_one_block": dict(connectivity=4, instrumented=False,
                              debug=False, waves_tpb=32, waves_blocks=1),
    "bare_scan": dict(lookup="scan", instrumented=False, debug=False),
    "bare_scan_c4": dict(lookup="scan", connectivity=4, instrumented=False,
                         debug=False),
    "bare_index": dict(lookup="index", instrumented=False, debug=False),
    "bare_index_c4_L0": dict(lookup="index", connectivity=4, lattice=0,
                             instrumented=False, debug=False),
    # the chapter's path before the dense-tile work (the ablation)
    "old": dict(OLD_PATH),
    "old_c4_one_block": dict(OLD_PATH, connectivity=4, waves_tpb=32,
                             waves_blocks=1),
    "old_bare": dict(OLD_PATH, instrumented=False, debug=False),
    # the dense-tile path forced on every touched tile (ccl_min 0): K2b's
    # run CCL, K4b, the dindex K3, at both connectivities, every lookup
    "dense_all": dict(ALL_CCL),
    "dense_all_c4": dict(ALL_CCL, connectivity=4),
    "dense_all_one_block_L0": dict(ALL_CCL, lattice=0,
                                   waves_tpb=32, waves_blocks=1),
    "dense_all_valve1_coll0": dict(ALL_CCL, valve_round=1,
                                   coll_capacity=0),
    "dense_all_L1_mask": dict(ALL_CCL, lattice=1,
                              _contract="mask"),
    "dense_all_no_compact_no_flat": dict(ALL_CCL,
                                         waves_compact=False, flat_opt=False),
    "dense_all_no_coop_c4": dict(ALL_CCL,
                                 waves_coop=False, connectivity=4),
    "dense_all_hybrid": dict(ALL_CCL, lookup="hybrid"),
    "dense_all_index_c4": dict(ALL_CCL, lookup="index",
                               connectivity=4),
    "dense_all_scan": dict(ALL_CCL, lookup="scan"),
    "dense_all_slow_paint": dict(ALL_CCL,
                                 paint_fast=False),
    "dense_all_odd_grids": dict(ALL_CCL,
                                label_grid=(3, 96), paint_grid=(5, 64),
                                fit_grids=False, waves_tpb=96),
    "dense_rpc1": dict(ccl_min=2, ccl_rpc=1, paint_ccl_min=2, paint_rpc=1),
    "dense_all_compact_auto": dict(ALL_CCL, waves_compact="auto",
                                   compact_min=64),
    "dense_all_bare": dict(ALL_CCL, instrumented=False,
                           debug=False),
    "dense_all_bare_c4_one_block": dict(ALL_CCL,
                                        connectivity=4, instrumented=False,
                                        debug=False, waves_tpb=32,
                                        waves_blocks=1),
}


@functools.lru_cache(maxsize=None)
def _scene(name):
    img = SCENES[name]()
    img.setflags(write=False)
    return img


@functools.lru_cache(maxsize=None)
def _oracle(name, connectivity):
    img = _scene(name)
    if connectivity == 8:
        return cpu_label_components(img)
    return cpu_label(img, 4)


@functools.lru_cache(maxsize=None)
def _model(name, connectivity):
    return tile_model(_scene(name), connectivity)


def run_and_check(img, label, n_blobs, model, contract="rgb", **kw):
    """Run ch08 (instrumented + debug unless the config says otherwise),
    check the outputs and every invariant the twin can show. The bare
    twin (the timed one) keeps only rounds, roots, interior comps and
    the valve counters, so it is checked on those. Returns the result."""
    kw = dict(kw)
    kw.setdefault("instrumented", True)
    kw.setdefault("debug", True)
    res = recolor(np.ascontiguousarray(img), contract=contract,
                  emit_label=True, **kw)
    c = res.counters
    height = img.shape[1]
    np.testing.assert_array_equal(res.label, label)
    np.testing.assert_array_equal(res.img, expected_paint(img, label))
    assert res.n_blobs == n_blobs
    assert res.seeds == expected_seeds(label, height)
    # ---- invariants (design 11.2)
    lattice = kw.get("lattice", DEFAULT_LATTICE)
    valve = kw.get("valve_round", DEFAULT_VALVE_ROUND)
    wm = model.waves(lattice, valve)
    if not kw["instrumented"]:
        assert c["interior_comps"] == model.n_interior
        assert c["roots"] + c["interior_comps"] == n_blobs
        assert c["rounds"] == wm["rounds"]
        assert c["valve_seeds"] == wm["valve_seeds"]
        assert c["valve_round"] == wm["valve_round"]
        return res
    assert c["ring_comps"] + c["interior_comps"] == c["comps"] \
        == model.n_comps
    assert c["ring_comps"] == model.n_ring
    assert c["claims"] + c["seeds"] + c["valve_seeds"] == c["ring_comps"]
    assert c["union_done"] == c["seeds"] + c["valve_seeds"] - c["roots"]
    assert c["roots"] + c["interior_comps"] == n_blobs
    assert c["rounds"] == wm["rounds"]
    assert c["tasks"] == model.tasks
    lookup = kw.get("lookup", DEFAULT_LOOKUP)
    dense_min = kw.get("dense_min", DEFAULT_DENSE_MIN)
    ccl_min = kw.get("ccl_min", DEFAULT_CCL_MIN)
    ccl_rpc = kw.get("ccl_rpc", DEFAULT_CCL_RPC)
    if lookup == "hybrid":
        assert c["dense_tasks"] == model.dense_tasks(dense_min)
    elif lookup == "dindex":
        assert c["dense_tasks"] == model.dindex_dense_tasks(
            ccl_min, ccl_rpc, dense_min, kw.get("waves_coop", DEFAULT_COOP))
    else:
        assert c["dense_tasks"] == 0
    if kw.get("dense", DEFAULT_DENSE):
        assert c["ccl_tiles"] == int(model.ccl_tiles(ccl_min, ccl_rpc).sum())
        n_ix = int(model.indexed_tiles(ccl_min, ccl_rpc).sum())
        assert c["indexed_tiles"] == (n_ix if lookup in ("index", "dindex")
                                      else 0)
    else:
        assert c["ccl_tiles"] == c["indexed_tiles"] == 0
        assert c["paint_ccl_tiles"] == 0
    assert c["seeds"] == wm["seeds"]
    assert c["seeds_lattice"] == wm["seeds_lattice"]
    assert c["seeds_queued"] == wm["seeds_queued"]
    assert c["claims"] == wm["claims"]
    assert c["valve_seeds"] == wm["valve_seeds"]
    assert c["valve_round"] == wm["valve_round"]
    assert c["union_calls"] == c["collisions"]
    # K4 and K4b: every touched tile is counted once and painted by
    # exactly one path (K4 used to count the tiles it hands to K4b, and
    # K4b counted them again)
    assert c["paint_tiles"] == c["tiles_touched"]
    assert (c["paint_uniform"] + c["paint_single_interior"]
            + c["paint_reclosed_tiles"] + c["paint_ccl_tiles"]) \
        == c["tiles_touched"]
    if kw.get("coll_capacity", None) == 0:
        assert c["coll_inline"] == c["collisions"]
    if kw.get("debug"):
        assert c["ordinal_overflow"] == 0
        assert c["anchor_mismatch"] == 0
        assert c["unlabelled"] == 0
    assert c["max_ring_tile"] <= 62
    return res


# ----------------------------------------------------- unit: closure

def _make_comps_probe(conn8, ch07_loop):
    @cuda.jit
    def probe(masks, out, iters):
        """One warp per 32x32 mask (block = 32 threads): every pixel gets
        the order index of its comp, in K2's comp order."""
        tile = cuda.blockIdx.x
        lane = cuda.threadIdx.x
        m = masks[tile, lane]
        rem = m
        c = 0
        n_it = 0
        while cuda.any_sync(k.FULL, rem != k.U0):
            f, a, bit = k._first_pixel(rem, lane)
            if ch07_loop:
                while True:
                    old = f
                    f = k._row_fill(f, rem)
                    if conn8:
                        f = k._diag_step(f, rem, lane)
                    f = k._col_fill(f, rem, lane)
                    n_it += 1
                    if not cuda.any_sync(k.FULL, f != old):
                        break
            else:
                f, it, last = k._close(f, rem, lane, conn8)
                n_it += it
            rem = uint32(rem & ~f)
            for j in range(32):
                if (f >> uint32(j)) & k.U1:
                    out[tile, lane, j] = c
            c += 1
        if lane == 0:
            iters[tile] = n_it

    return probe


_PROBES = {}


def _probe(conn8, ch07_loop):
    key = (conn8, ch07_loop)
    if key not in _PROBES:
        _PROBES[key] = _make_comps_probe(conn8, ch07_loop)
    return _PROBES[key]


def _cpu_tile_comps(bits, conn8):
    """bits (32, 32) bool, [row, y]: comp order index per pixel, comps in
    (row, bit) order of their first pixel, BFS inside the tile."""
    out = np.full((32, 32), -1, np.int32)
    c = 0
    for x in range(32):
        for y in range(32):
            if not bits[x, y] or out[x, y] >= 0:
                continue
            out[x, y] = c
            stack = [(x, y)]
            while stack:
                px, py = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if (dx, dy) == (0, 0):
                            continue
                        if not conn8 and dx and dy:
                            continue
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < 32 and 0 <= ny < 32 and bits[nx, ny] \
                                and out[nx, ny] < 0:
                            out[nx, ny] = c
                            stack.append((nx, ny))
            c += 1
    return out


def _closure_masks():
    rng = np.random.default_rng(2026)
    masks = []
    for _ in range(500):
        d = rng.uniform(0.05, 0.7)
        masks.append(rng.random((32, 32)) < d)
    xx, yy = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    masks.append(np.ones((32, 32), bool))
    masks.append(np.zeros((32, 32), bool))
    masks.append((xx + yy) % 2 == 0)                 # checkerboard
    masks.append((xx % 2 == 0) & (yy % 2 == 0))      # 256 isolated px
    one_row = np.zeros((32, 32), bool)
    one_row[7] = True
    masks.append(one_row)
    one_col = np.zeros((32, 32), bool)
    one_col[:, 19] = True
    masks.append(one_col)
    spiral = S8.spiral_scene(32)[..., 1] == 0
    masks.append(spiral)
    return masks


def _pack_rows(bits):
    w = np.zeros(32, np.uint32)
    for x in range(32):
        v = 0
        for y in range(32):
            if bits[x, y]:
                v |= 1 << y
        w[x] = v
    return w


@pytest.mark.parametrize("connectivity", [8, 4])
@pytest.mark.parametrize("ch07_loop", [False, True])
def test_closure_comps_match_cpu(connectivity, ch07_loop):
    """K2's comp loop (lowest remaining pixel first, ch07 closure with
    the last-comp and frontier exits) finds exactly the CPU's comps, in
    the same order. The ch07_loop twin (no early exits) agrees too."""
    masks = _closure_masks()
    words = np.stack([_pack_rows(m) for m in masks])
    out = cuda.to_device(np.full((len(masks), 32, 32), -1, np.int32))
    iters = cuda.device_array(len(masks), np.int32)
    _probe(connectivity == 8, ch07_loop)[len(masks), 32](
        cuda.to_device(words), out, iters)
    got = out.copy_to_host()
    for i, m in enumerate(masks):
        exp = _cpu_tile_comps(m, connectivity == 8)
        np.testing.assert_array_equal(got[i], exp, err_msg=f"mask {i}")


def test_closure_exits_save_trips():
    """The two exits never cost more trips than ch07's proving loop."""
    masks = _closure_masks()
    words = cuda.to_device(np.stack([_pack_rows(m) for m in masks]))
    trips = {}
    for loop in (False, True):
        out = cuda.to_device(np.full((len(masks), 32, 32), -1, np.int32))
        it = cuda.device_array(len(masks), np.int32)
        _probe(True, loop)[len(masks), 32](words, out, it)
        trips[loop] = it.copy_to_host()
    assert (trips[False] <= trips[True]).all()
    assert trips[False].sum() < trips[True].sum()


# ----------------------------------------- unit: the run CCL (K2b)

def _make_ccl_probe(conn8):
    @cuda.jit
    def probe(masks, root_out, nroot, nring, sides, anc):
        """One warp per 32x32 mask: K2b's run CCL, its ring bitmap and
        ordinals, and the side words and root slot of each ring comp."""
        P = cuda.shared.array(512, int32)
        t = cuda.blockIdx.x
        lane = cuda.threadIdx.x
        m = masks[t, lane]
        st, nr, nu = k._ccl(m, lane, P, 0, conn8)
        n_total = k._wsum32(nr)
        for j in range(32):
            if (m >> uint32(j)) & k.U1:
                root_out[t, lane, j] = P[lane * 16
                                         + (k._start_le(st, j) >> 1)]
        cuda.syncwarp()
        rn, rs, rw, re, n0, n31 = k._frame_roots(m, st, lane, P, 0)
        cuda.syncwarp()
        R, base, b = k._ring_bitmap(rn, rs, rw, re, n0, n31, lane, P, 0)
        on, _ = k._ordinal(rn, R, base)
        os_, _ = k._ordinal(rs, R, base)
        ow, _ = k._ordinal(rw, R, base)
        oe, _ = k._ordinal(re, R, base)
        for h in range(2):
            o = lane + 32 * h
            if o < b:
                for q in range(4):
                    P[k.CCL_SIDES + 4 * o + q] = 0
        cuda.syncwarp()
        k._group_store(rn, on, lane, P, k.CCL_SIDES, 0, k.CCL_ANC)
        k._group_store(rs, os_, lane, P, k.CCL_SIDES, 1, k.CCL_ANC)
        k._group_store(rw, ow, lane, P, k.CCL_SIDES, 2, k.CCL_ANC)
        k._group_store(re, oe, lane, P, k.CCL_SIDES, 3, k.CCL_ANC)
        cuda.syncwarp()
        for h in range(2):
            o = lane + 32 * h
            if o < b:
                for q in range(4):
                    sides[t, o, q] = uint32(P[k.CCL_SIDES + 4 * o + q])
                anc[t, o] = P[k.CCL_ANC + o]
        if lane == 0:
            nroot[t] = n_total
            nring[t] = b

    return probe


@pytest.mark.parametrize("connectivity", [8, 4])
def test_run_ccl_matches_cpu(connectivity):
    """K2b's union-find over run starts: every pixel's root slot is its
    comp's anchor (the min (row, bit) pixel, slot 16*row + bit//2), the
    comp count matches, and the ring comps come out in anchor order with
    the CPU's side words. Masks: the closure set plus a frame comb and
    every-other-row stripes (many runs, no unions)."""
    conn8 = connectivity == 8
    masks = _closure_masks()
    xx, yy = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    comb = np.zeros((32, 32), bool)
    comb[0, ::2] = comb[31, ::2] = comb[::2, 0] = comb[::2, 31] = True
    masks += [comb, xx % 2 == 0, yy % 2 == 0, (xx + 2 * yy) % 3 == 0]
    n = len(masks)
    words = cuda.to_device(np.stack([_pack_rows(m) for m in masks]))
    root = cuda.to_device(np.full((n, 32, 32), -1, np.int32))
    nroot = cuda.device_array(n, np.int32)
    nring = cuda.device_array(n, np.int32)
    sides = cuda.to_device(np.zeros((n, 64, 4), np.uint32))
    anc = cuda.to_device(np.full((n, 64), -1, np.int32))
    _make_ccl_probe(conn8)[n, 32](words, root, nroot, nring, sides, anc)
    root = root.copy_to_host()
    nroot = nroot.copy_to_host()
    nring = nring.copy_to_host()
    sides = sides.copy_to_host()
    anc = anc.copy_to_host()
    for i, m in enumerate(masks):
        comp = _cpu_tile_comps(m, conn8)
        n_c = int(comp.max()) + 1
        exp = np.full((32, 32), -1, np.int32)
        ring = []
        for c in range(n_c):
            xs, ys = np.nonzero(comp == c)
            a = min(zip(xs.tolist(), ys.tolist()))
            exp[comp == c] = 16 * a[0] + (a[1] >> 1)
            on = lambda sel: int(sum(1 << int(v) for v in sel))
            sn = on(ys[xs == 0])
            ss = on(ys[xs == 31])
            sw = on(xs[ys == 0])
            se = on(xs[ys == 31])
            if sn or ss or sw or se:
                ring.append((16 * a[0] + (a[1] >> 1), sn, ss, sw, se))
        np.testing.assert_array_equal(root[i], exp, err_msg=f"mask {i}")
        assert nroot[i] == n_c
        assert nring[i] == len(ring)
        for o, (slot, sn, ss, sw, se) in enumerate(sorted(ring)):
            assert anc[i, o] == slot
            assert tuple(int(v) for v in sides[i, o]) == (sn, ss, sw, se)


# ------------------------------------------- unit: records after K2

RECORD_SCENES = ["noise_030", "noise_045", "size_31x97", "u_shape",
                 "all_eight", "frame_comb_33", "frame_comb_60",
                 "dot_lattice_256", "corner_main_px", "corner_anti_px",
                 "tile_corner_blocks", "rmin_trap", "spiral", "size_65x33"]


@pytest.mark.parametrize("name", RECORD_SCENES)
@pytest.mark.parametrize("connectivity", [8, 4])
@pytest.mark.parametrize("lattice", [0, 8, 1])
@pytest.mark.parametrize("path", ["loop", "ccl"])
def test_records_match_model(name, connectivity, lattice, path):
    """After K0 + K2 (+ K2b) alone: every tile header, and every ring
    record's side words, anchor, direction bits, seed label and parent,
    equal the CPU model's (computed from pixels). The frame words and
    frame-pixel ordinals too. path "loop": K2's comp loop on every tile
    (the old path, its index twin); "ccl": K2b's run CCL on every
    touched tile, which writes the index for 2+ ring comps and flags the
    header."""
    img = _scene(name)
    tm = _model(name, connectivity)
    kw = (dict(OLD_PATH, lookup="index") if path == "loop" else
          dict(ALL_CCL, lookup="dindex"))
    eng = TileWaves(img.shape[0], img.shape[1], connectivity=connectivity,
                    instrumented=True, debug=True, **kw)
    dev = cuda.to_device(img)
    eng.pack(dev)
    eng.label_pass(lattice)
    cuda.synchronize()
    hdr = eng.hdr.copy_to_host()
    np.testing.assert_array_equal(hdr & 63, tm.n_ring_tile)
    np.testing.assert_array_equal(hdr >> 8, tm.n_total_tile)
    np.testing.assert_array_equal(
        (hdr & k.HDR_INDEXED) != 0,
        (path == "ccl") & (tm.n_ring_tile >= 2))
    rec = eng.rec.copy_to_host().reshape(-1, 8)
    parent = eng.parent.copy_to_host()
    seed, _ = tm.seeds(lattice)
    ring = np.flatnonzero(tm.ordinal >= 0)
    ids = 64 * tm.tile[ring].astype(np.int64) + tm.ordinal[ring]
    r = rec[ids]
    np.testing.assert_array_equal(r[:, 0].view(np.uint32), tm.side_n[ring])
    np.testing.assert_array_equal(r[:, 1].view(np.uint32), tm.side_s[ring])
    np.testing.assert_array_equal(r[:, 2].view(np.uint32), tm.side_w[ring])
    np.testing.assert_array_equal(r[:, 3].view(np.uint32), tm.side_e[ring])
    np.testing.assert_array_equal(r[:, 4], tm.anchor[ring])
    np.testing.assert_array_equal(r[:, 5], tm.dirs[ring])
    exp_label = np.where(seed[ring], ids, -1)
    np.testing.assert_array_equal(r[:, 6], exp_label)
    s_ids = ids[seed[ring]]
    np.testing.assert_array_equal(parent[s_ids], s_ids)
    c = eng.read_counters()
    assert c["seeds"] == int(seed.sum())
    assert c["interior_comps"] == tm.n_interior
    # index twin, tiles with 2+ ring comps: frame words = the tile's red
    # frame pixels, and each red frame pixel's byte = the ordinal of the
    # ring comp that owns it
    frame = eng.frame.copy_to_host().reshape(-1, 4)
    idx = eng.idx.copy_to_host().reshape(-1, 4, 32)
    touched = np.flatnonzero(tm.n_ring_tile >= 2)   # indexed tiles only
    exp_frame = np.zeros((tm.n_ring_tile.size, 4), np.uint32)
    exp_idx = np.full((tm.n_ring_tile.size, 4, 32), -1, np.int32)
    for c_i in ring:
        t = tm.tile[c_i]
        sides = (tm.side_n[c_i], tm.side_s[c_i], tm.side_w[c_i],
                 tm.side_e[c_i])
        for sd, word in enumerate(sides):
            exp_frame[t, sd] |= word
            for j in range(32):
                if (int(word) >> j) & 1:
                    exp_idx[t, sd, j] = tm.ordinal[c_i]
    np.testing.assert_array_equal(frame[touched], exp_frame[touched])
    sel = exp_idx[touched] >= 0
    np.testing.assert_array_equal(idx[touched][sel], exp_idx[touched][sel])


def test_ring_bound_is_62():
    """62 ring comps in one tile at 4-conn and 60 at 8-conn (the frame
    comb): the ordinal bound of the README is reached, never passed."""
    for conn, n, name in ((4, 62, "frame_comb_62_c4"),
                          (8, 60, "frame_comb_60")):
        tm = _model(name, conn)
        assert tm.n_ring_tile.max() == n


# ------------------------------------------------- unit: pack_tiles

PACK_SIZES = [1, 31, 32, 33, 63, 64, 65, 97]


@pytest.mark.parametrize("variant", ["direct", "smem"])
def test_pack_tiles_equals_ch06_pack(variant):
    """K1 writes exactly ch06's mask, padding bits included, on every
    size pair from PACK_SIZES and on a few scenes. The mask is poisoned
    first, so an unwritten word cannot pass."""
    pack = k.get_pack_kernel(variant)
    rng = np.random.default_rng(5)
    imgs = []
    for w in PACK_SIZES:
        for h in PACK_SIZES:
            red = rng.random((w, h)) < 0.5
            img = S8.from_mask(red)
            # non-red colours that share bytes with red must not count
            img[rng.random((w, h)) < 0.1] = (255, 0, 1)
            img[rng.random((w, h)) < 0.1] = (254, 0, 0)
            imgs.append(img)
    imgs += [_scene(n) for n in ("noise_030", "size_31x97", "full_red",
                                 "blank", "one_row", "one_col")]
    for img in imgs:
        w, h = img.shape[:2]
        wpr = (h + 31) // 32
        dev = cuda.to_device(img)
        ref = cuda.to_device(np.full((w, wpr), 0xDEADBEEF, np.uint32))
        got = cuda.to_device(np.full((w, wpr), 0xDEADBEEF, np.uint32))
        g6, t6 = plain_shape(w, wpr)
        pack_kernel[g6, t6](dev, ref)
        grid, tpb = k.pack_shape(w, wpr)
        pack[grid, tpb](dev, got)
        cuda.synchronize()
        np.testing.assert_array_equal(got.copy_to_host(), ref.copy_to_host(),
                                      err_msg=f"{w}x{h}")
        tail = got.copy_to_host()[:, -1]
        if h % 32:
            assert ((tail >> np.uint32(h % 32)) == 0).all()


def test_pack_tiles_thin_shapes():
    """The launch shapes of the 524288 x 32 and 32 x 524288 strips (one
    word per row; one tile row)."""
    pack = k.get_pack_kernel("direct")
    imgs = [S8.thin_strip_scene(1 << 19, 32, axis=0),
            S8.thin_strip_scene(1 << 19, 32, axis=1),
            S8.from_mask(np.random.default_rng(1).random((70000, 33)) < 0.3)]
    for img in imgs:
        w, h = img.shape[:2]
        wpr = (h + 31) // 32
        dev = cuda.to_device(img)
        ref = cuda.to_device(np.full((w, wpr), 0xDEADBEEF, np.uint32))
        got = cuda.to_device(np.full((w, wpr), 0xDEADBEEF, np.uint32))
        g6, t6 = plain_shape(w, wpr)
        pack_kernel[g6, t6](dev, ref)
        grid, tpb = k.pack_shape(w, wpr)
        pack[grid, tpb](dev, got)
        cuda.synchronize()
        np.testing.assert_array_equal(got.copy_to_host(), ref.copy_to_host())


# ------------------------------------------------ unit: the rotation

@pytest.mark.parametrize("blocks", ["one", "max"])
def test_counter_rotation_over_many_rounds(blocks):
    """A scene needing far more than 7 rounds reuses each of the three
    rotating append counters many times; rounds and frontiers must equal
    the model's at 1 block of 32 threads and at the cooperative max."""
    name = "diag_snake"
    img = _scene(name)
    label, n = _oracle(name, 8)
    tm = _model(name, 8)
    kw = dict(waves_tpb=32, waves_blocks=1) if blocks == "one" else {}
    res = run_and_check(img, label, n, tm, lattice=0, **kw)
    assert res.rounds == 14 == tm.waves(0, 128)["rounds"]
    eng = TileWaves(img.shape[0], img.shape[1], lattice=0,
                    instrumented=True, **kw)
    dev = cuda.to_device(img)
    eng.run(dev)
    cuda.synchronize()
    st = eng.round_stats()
    assert st["frontier_comps"] == tm.waves(0, 128)["frontier"]


def test_round_stats_past_the_stamp_slots():
    """A 1-px line through 300 one-row tiles is a 300-comp chain: 300
    rounds at L0 with the valve off. The instrumented twin stamps 251
    rounds one by one and the rest in one bucket, so the union pass's
    split holds no round."""
    img = S8.from_mask(np.ones((1, 32 * 300), bool))
    label, n = cpu_label_components(img)
    run_and_check(img, label, n, tile_model(img, 8), lattice=0,
                  valve_round=0)
    eng = TileWaves(1, 32 * 300, lattice=0, valve_round=0,
                    instrumented=True)
    eng.run(cuda.to_device(img))
    cuda.synchronize()
    st = eng.round_stats()
    assert st["rounds"] == 300
    assert len(st["round_us"]) == 251
    assert st["rounds_past_stamps"] == 49
    assert st["rounds_past_stamps_us"] > st["union_us"] >= 0


# ------------------------------------------------- the 4-conn oracle

@pytest.mark.parametrize("name", sorted(SCENES))
def test_4conn_oracle_port_matches_ch05_at_8conn(name):
    """cpu_label (ported from the scout) equals ch05's oracle at 8-conn,
    so it can be trusted at 4-conn."""
    img = _scene(name)
    a, na = cpu_label(img, 8)
    b, nb = cpu_label_components(img)
    assert na == nb
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------- end to end

@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("name", sorted(SCENES))
def test_end_to_end(name, config):
    """Label map, paint, n_blobs, seeds and every invariant, bit for bit,
    for each scene under each config of the matrix."""
    kw = dict(CONFIGS[config])
    contract = kw.pop("_contract", "rgb")
    conn = kw.get("connectivity", 8)
    img = _scene(name)
    label, n = _oracle(name, conn)
    run_and_check(img, label, n, _model(name, conn), contract=contract, **kw)


@pytest.mark.parametrize("name", ["noise_030", "u_shape", "blob_grid_100",
                                  "comb_80_teeth", "serpentine",
                                  "size_97x131", "rmin_trap", "spiral",
                                  "frame_comb_60", "maze_1k"])
def test_paint_equals_ch06_and_ch05(name):
    """All three chapters paint palette[canonical % 6]: ch08's image
    equals ch06's recolor() and ch05's flood_fill() exactly."""
    from ..ch05_gpu_nblob_nblock.flood_fill import flood_fill as ch05_fill
    img = np.ascontiguousarray(_scene(name))
    a = recolor(img).img
    b = ch06_recolor(img).img
    np.testing.assert_array_equal(a, b)
    c = ch05_fill(img, variant="seed_merge", lattice=8, build="split").img
    np.testing.assert_array_equal(a, c)


@pytest.mark.parametrize("lattice", [0, 2, 4, 8, 16, 32, 1])
def test_every_seed_rule_gives_identical_labels(lattice):
    """The strongest single check of the union protocol and of rmin:
    every seed rule must give the same labels."""
    for name in ("noise_045", "maze_1k", "spiral", "rmin_trap",
                 "zz_lattice_1k", "frame_comb_60", "diag_snake"):
        img = _scene(name)
        label, n = _oracle(name, 8)
        run_and_check(img, label, n, _model(name, 8), lattice=lattice)


@pytest.mark.parametrize("valve", [1, 2, 3, 5, 9])
def test_valve_fires_and_stays_exact(valve):
    """The diagonal snake needs 14 rounds at L0: a valve at round v < 14
    self-seeds the rest, then at most one more round has work."""
    name = "diag_snake"
    img = _scene(name)
    label, n = _oracle(name, 8)
    tm = _model(name, 8)
    res = run_and_check(img, label, n, tm, lattice=0, valve_round=valve)
    assert res.counters["valve_round"] == valve
    assert res.counters["valve_seeds"] > 0
    assert res.rounds <= valve + 1


# ---------------------------------------------------------- fuzzing

FUZZ_SEEDS = {"noise": 200, "rects_and_disks": 60, "walks": 60,
              "border_zigzags": 60, "stitched_tiles": 60}


@pytest.mark.parametrize("blocks", ["one", "max"])
@pytest.mark.parametrize("family", list(FUZZ_SEEDS))
def test_fuzz(family, blocks):
    """Seeded fuzz families: oracle, invariants and model rounds. A
    failure names the seed and saves the image (np.save) to
    FLOOD_FILL_SCRATCH."""
    builder = S8.FUZZ_FAMILIES[family]
    kw = dict(waves_tpb=32, waves_blocks=1) if blocks == "one" else {}
    for seed in range(FUZZ_SEEDS[family]):
        rng = np.random.default_rng(10_000 * (1 + list(FUZZ_SEEDS).index(
            family)) + seed)
        img = builder(rng)
        conn = 4 if seed % 4 == 3 else 8
        lat = (0, 8, 1, 4)[seed % 4] if seed % 5 else 8
        valve = 2 if seed % 7 == 6 else 128
        lookup = ("scan", "index", "hybrid", "dindex")[seed % 4]
        twin = ({} if seed % 6 != 5 else
                dict(instrumented=False, debug=False))     # the bare twin
        # the dense path: forced on every tile, at small thresholds, at
        # the defaults, or off (the old path) - dindex needs it on
        dense = [dict(ALL_CCL),
                 dict(ccl_min=seed % 7, paint_ccl_min=seed % 5,
                      ccl_rpc=1 + seed % 13, paint_rpc=1 + seed % 11),
                 {}, dict(dense=False)][(seed // 4) % 4]
        if lookup == "dindex" and dense.get("dense") is False:
            dense = dict(ALL_CCL, waves_compact=False)
        try:
            label, n = cpu_label(img, conn)
            run_and_check(img, label, n, tile_model(img, conn),
                          connectivity=conn, lattice=lat, valve_round=valve,
                          lookup=lookup, dense_min=seed % 5, **dense, **twin,
                          **kw)
        except AssertionError as exc:
            path = os.path.join(SCRATCH, f"ch08_fuzz_{family}_{seed}.npy")
            np.save(path, img)
            raise AssertionError(
                f"{family} seed {seed} (conn {conn}, L {lat}, valve "
                f"{valve}, {lookup}, dense_min {seed % 5}, {dense}) failed; "
                f"image saved to {path}") from exc


# ------------------------------------------- structural / robustness

@pytest.mark.parametrize("kw", [dict(lookup="index"),
                                dict(ALL_CCL, lookup="dindex"),
                                dict(lookup="dindex", ccl_min=4,
                                     paint_ccl_min=4)])
def test_engine_reuse_with_poisoned_buffers(kw):
    """A, B, A on one engine, every reusable buffer filled with
    0x5A5A5A5A between runs: the outputs must not change. Nothing is
    read before it is written in the same run (design 9.4), the dense
    path's tile list and index included."""
    a = S5.random_blobs_scene(120, 120, 0.35, 21)[0]
    b = S5.random_blobs_scene(120, 120, 0.50, 22)[0]
    eng = TileWaves(120, 120, instrumented=True, debug=True, **kw)
    outs = []
    for img in (a, b, a):
        eng.poison()
        cuda.synchronize()
        res = recolor(img, engine=eng, emit_label=True)
        label, n = cpu_label_components(img)
        np.testing.assert_array_equal(res.label, label)
        np.testing.assert_array_equal(res.img, expected_paint(img, label))
        assert res.n_blobs == n
        outs.append(res)
    np.testing.assert_array_equal(outs[0].img, outs[2].img)


def test_repeatability():
    """20 runs: labels, paint, rounds, seeds, claims and links identical.
    Which wave claims a comp is a race, so collision counts may vary."""
    img = _scene("maze_1k")
    eng = TileWaves(img.shape[0], img.shape[1], instrumented=True)
    first = None
    for _ in range(20):
        res = recolor(img, engine=eng, emit_label=True)
        c = res.counters
        key = (res.rounds, c["seeds"], c["claims"], c["union_done"],
               c["roots"], tuple(res.seeds))
        if first is None:
            first = (key, res.img, res.label)
        else:
            assert key == first[0]
            np.testing.assert_array_equal(res.img, first[1])
            np.testing.assert_array_equal(res.label, first[2])


@pytest.mark.parametrize("name", ["noise_045", "maze_1k", "zz_serp_1k",
                                  "frame_comb_60", "blob_grid_100"])
def test_twins_agree(name):
    """Bare, instrumented and debug twins give identical outputs; the
    bare twin's n_blobs comes from its two kept counters."""
    img = _scene(name)
    label, n = _oracle(name, 8)
    outs = []
    for kw in (dict(), dict(instrumented=True), dict(debug=True),
               dict(instrumented=True, debug=True), dict(lookup="index"),
               dict(lookup="scan"),
               dict(lookup="index", instrumented=True, debug=True),
               dict(lookup="hybrid", dense_min=0),
               dict(lookup="hybrid", dense_min=0, instrumented=True,
                    debug=True),
               dict(OLD_PATH), dict(OLD_PATH, instrumented=True),
               dict(ALL_CCL),
               dict(ALL_CCL, instrumented=True,
                    debug=True),
               dict(ALL_CCL, waves_compact=False)):
        res = recolor(img, emit_label=True, **kw)
        assert res.n_blobs == n
        np.testing.assert_array_equal(res.label, label)
        outs.append(res.img)
        if kw.get("debug"):
            c = res.counters
            assert c["ordinal_overflow"] == c["anchor_mismatch"] \
                == c["unlabelled"] == 0
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


def test_contracts_agree():
    """rgb and mask produce identical images, labels and counters."""
    for name in ("noise_030", "maze_1k", "size_63x95"):
        img = _scene(name)
        a = recolor(img, contract="rgb", emit_label=True, instrumented=True)
        b = recolor(img, contract="mask", emit_label=True, instrumented=True)
        np.testing.assert_array_equal(a.img, b.img)
        np.testing.assert_array_equal(a.label, b.label)
        for key in ("rounds", "seeds", "claims", "roots", "ring_comps",
                    "interior_comps", "tasks"):
            assert a.counters[key] == b.counters[key]


def test_blank_image_is_valid():
    img = S5.blank_scene(64, 64)[0]
    res = recolor(img, emit_label=True)
    assert (res.n_blobs, res.seeds, res.rounds) == (0, [], 0)
    np.testing.assert_array_equal(res.img, img)
    assert (res.label == -1).all()


def test_shape_limits():
    """The shape asserts raise without allocating anything."""
    with pytest.raises(ValueError, match="2\\*\\*31"):
        TileWaves(50000, 50000, _skip_warmup=True)
    with pytest.raises(ValueError, match="record words"):
        TileWaves(1, 1 << 30, _skip_warmup=True)   # 2**25 one-row tiles
    with pytest.raises(ValueError, match="bad shape"):
        TileWaves(0, 5, _skip_warmup=True)
    with pytest.raises(ValueError, match="connectivity"):
        TileWaves(8, 8, connectivity=6, _skip_warmup=True)
    with pytest.raises(ValueError, match="uint8"):
        recolor(np.zeros((8, 8, 3), np.int32))
    with pytest.raises(ValueError, match="contract"):
        recolor(np.zeros((8, 8, 3), np.uint8), contract="nope")


@pytest.mark.parametrize("grid", [(8, 48), (16, 40), (4, 100), (1, 16),
                                  (0, 128), (1, 2048)])
def test_label_grid_is_validated(grid):
    """K2 takes warp = grid(1) >> 5 and runs FULL-mask ballots and
    shuffles, so its block must be a whole number of warps. At 48
    threads a logical warp straddles two hardware warps: tiles are
    closed twice or half, with no error (140 blobs for 133 on a 64x64
    noise scene). The engine must refuse such a grid."""
    with pytest.raises(ValueError, match="label_grid"):
        TileWaves(64, 64, label_grid=grid, _skip_warmup=True)


@pytest.mark.parametrize("grid", [(8, 48), (16, 40), (4, 100), (1, 16),
                                  (0, 128), (1, 2048)])
def test_paint_grid_is_validated(grid):
    """K4 and its label-map and seed twins split warps the same way as
    K2 (warp = grid(1) >> 5, FULL-mask calls). At 48 threads the paint,
    the label map and the seeds come out wrong (139 seeds for 133
    blobs). The engine must refuse such a grid."""
    with pytest.raises(ValueError, match="paint_grid"):
        TileWaves(64, 64, paint_grid=grid, _skip_warmup=True)


@pytest.mark.parametrize("grid_arg", ["label_grid", "paint_grid"])
def test_dense_twins_limit_the_block(grid_arg):
    """The dense twins hold 2 KiB of shared slots per warp in a static
    array: at most 512 threads per block. The old path keeps its range,
    and the dindex lookup needs the dense K2 (only K2b writes its
    index)."""
    with pytest.raises(ValueError, match="dense twins"):
        TileWaves(64, 64, _skip_warmup=True, **{grid_arg: (2, 1024)})
    TileWaves(64, 64, _skip_warmup=True, **{grid_arg: (2, 1024)},
              **OLD_PATH)
    with pytest.raises(ValueError, match="dindex"):
        TileWaves(64, 64, dense=False, lookup="dindex", _skip_warmup=True)


@pytest.mark.parametrize("cap", [64, 96])
def test_k2b_register_cap_builds_and_stays_exact(cap):
    """label_dense_max_registers caps K2b's registers. The warm-up used
    to compile only the uncapped twin, so under the default fit_grids
    the constructor asked a dispatcher with no overload for its capacity
    and raised StopIteration (and with fit_grids off it launched the
    uncapped twin). A capped engine must build and stay exact."""
    img = S8.dot_lattice_scene((2, 3))
    label, n = cpu_label_components(img)
    for kw in (dict(), dict(fit_grids=False), dict(instrumented=True)):
        res = recolor(img, emit_label=True, label_dense_max_registers=cap,
                      **kw)
        assert res.n_blobs == n
        np.testing.assert_array_equal(res.label, label)
        np.testing.assert_array_equal(res.img, expected_paint(img, label))


def test_dense_path_takes_the_dense_tiles():
    """On a dot lattice (256 one-pixel comps per tile) every tile goes to
    K2b and K4b, and every tile with 2+ ring comps is indexed; on a solid
    square none does. The old path never uses them."""
    dots = S8.dot_lattice_scene((4, 3))
    square = S5.full_red_scene(96, 96)[0]
    for img, all_dense in ((dots, True), (square, False)):
        res = recolor(img, emit_label=True, instrumented=True)
        c = res.counters
        n_tiles = ((img.shape[0] + 31) // 32) * ((img.shape[1] + 31) // 32)
        assert c["ccl_tiles"] == (n_tiles if all_dense else 0)
        assert c["paint_ccl_tiles"] == (n_tiles if all_dense else 0)
        assert c["paint_tiles"] == c["tiles_touched"] == n_tiles
        label, n = cpu_label_components(img)
        np.testing.assert_array_equal(res.label, label)
        old = recolor(img, instrumented=True, **OLD_PATH).counters
        assert old["ccl_tiles"] == old["paint_ccl_tiles"] == 0
        assert old["comps"] == c["comps"]


def test_pack_variant_is_validated():
    """'tilemajor' is experiment E1's reference store: a tile-major
    layout with no row guard, which no later kernel reads. Through the
    engine it moved pixel (0, 32) to (32, 0) and stored past the mask.
    The engine and recolor() must refuse it; the kernel itself stays
    available to E1."""
    img = S8.from_mask(np.zeros((33, 33), bool))
    with pytest.raises(ValueError, match="pack_variant"):
        TileWaves(33, 33, pack_variant="tilemajor", _skip_warmup=True)
    with pytest.raises(ValueError, match="pack_variant"):
        recolor(img, pack_variant="tilemajor")
    assert k.get_pack_kernel("tilemajor") is not None


@pytest.mark.parametrize("grid_arg", ["label_grid", "paint_grid"])
def test_odd_but_whole_warp_grids_are_exact(grid_arg):
    """Block sizes that are whole warps stay exact on small grids."""
    img = S5.random_blobs_scene(64, 64, 0.35, 2)[0]
    label, n = cpu_label_components(img)
    for grid in ((3, 96), (8, 64), (1, 32)):
        res = recolor(img, emit_label=True, **{grid_arg: grid})
        assert res.n_blobs == n
        np.testing.assert_array_equal(res.label, label)
        np.testing.assert_array_equal(res.img, expected_paint(img, label))
        assert res.seeds == expected_seeds(label, img.shape[1])


def test_cooperative_grid_is_bounded():
    cap = max_blocks(256, 8, False, False)
    assert cap >= 1
    with pytest.raises(RuntimeError, match="grid.sync"):
        TileWaves(64, 64, waves_blocks=cap + 1)


# ------------------------------------------------------- slow scenes

def _png(name):
    if not IMAGES:
        pytest.skip("set FLOOD_FILL_IMAGES to the folder with the PNGs")
    path = os.path.join(IMAGES, f"{name}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return S5.png_scene(path)[0]


BENCH_SCENES = {
    "input_blobs": lambda: _png("input_blobs"),
    "input_blocks": lambda: _png("input_blocks"),
    "blob_grid_100": lambda: S5.blob_grid_scene(4000, 4000, 10, 10, 360,
                                                gap=40)[0],
    "random_4000": lambda: S5.random_blobs_scene(4000, 4000, 0.30, 0)[0],
    "disk_r2000": lambda: S5.disk_scene(4200, 4200, 2000)[0],
    "serpentine_2048": lambda: S5.serpentine_scene(2048, 2048)[0],
}


@pytest.mark.skipif(not SLOW, reason="set FLOOD_FILL_SLOW=1")
@pytest.mark.parametrize("name", list(BENCH_SCENES))
def test_benchmark_scenes(name):
    """The 6 benchmark scenes: default, L0, L1, valve 1, coll 0, the
    index lookup at the cooperative max, and the mask contract."""
    img = BENCH_SCENES[name]()
    label, n = cpu_label_components(img)
    tm = tile_model(img, 8)
    for contract, kw in (("rgb", {}), ("rgb", dict(lattice=0)),
                         ("rgb", dict(lattice=1)),
                         ("rgb", dict(valve_round=1)),
                         ("rgb", dict(coll_capacity=0)),
                         ("rgb", dict(lookup="index")),
                         ("rgb", dict(lookup="hybrid", dense_min=0)),
                         ("rgb", dict(lookup="scan")),
                         ("rgb", dict(OLD_PATH)),
                         ("rgb", dict(ALL_CCL)),
                         ("rgb", dict(waves_compact=False)),
                         ("mask", {})):
        run_and_check(img, label, n, tm, contract=contract, **kw)


@pytest.mark.skipif(not SLOW, reason="set FLOOD_FILL_SLOW=1")
@pytest.mark.parametrize("axis", [0, 1])
def test_thin_strips(axis):
    """524288 x 32 and 32 x 524288 noise strips (16.8 Mpx each)."""
    img = S8.thin_strip_scene(1 << 19, 32, axis)
    label, n = cpu_label_components(img)
    run_and_check(img, label, n, tile_model(img, 8))
