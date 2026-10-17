"""Act 3 of the chain cut (v5): ch07's floors, recorded from the kernel.

Two exports, one session, both with the chapter's default (one_sync +
tile_paint, 8-conn, 128 threads per block, instrumented):

1. sq_8000_center (8000 x 8000, full bleed, seed (4000, 4000)): the
   round in which each 32x32 tile was first opened (250 x 250), how
   often each tile was visited, the frontier size of every round, the
   rounds and the visits.
2. A warp close-up on input_blobs.png: one tile on a blob edge, its 32
   mask words, its entry bits (the reached words its first visit read),
   and the reached words after every sweep of the closure (row fill,
   diagonal step, column fill) until no lane changes.

WHY A TWIN
The chapter's kernel keeps no per-round record. Its per-tile stamp is
raised with atomic.max, so after a run it holds the LAST round a tile
was queued for, not the first. So this script builds an instrumented
export twin: the chapter's own factory, `kernels._make_rounds_kernel`,
with four short insertions made by an anchored source patch (each
anchor must occur exactly once, or the script stops). Nothing else
changes, and the timed kernels are untouched:

    first_open[t]   set by the warp whose atomic.max on t's stamp
                    returned -1 (exactly one warp sees -1): the round of
                    t's first visit. The host sets the seed tile's to 0.
    round_size[r]   thread 0 writes n_cur, the frontier of round r
    visit_count[t]  lane 0 adds 1 per visit of tile t
    probe[s, v]     for probed tiles only (probe_slot[t] = s >= 0): the
                    32 mask words m, the entry words f0 and the closed
                    words f of visit v, its closure trips and its round

Checks (the script stops on any failure):

- The twin's fill equals the chapter kernel's fill, bit for bit, and
  both equal the @njit oracle's visited set (cpu_flood_fill_8).
- Rounds equal the chapter kernel's on every run (127 on sq_8000).
- sum(round_size) == the TILE_VISITS counter == sum(visit_count), and
  max(round_size) == the PEAK_ROUND counter.
- first_open is set on every tile, never exceeds the stamp of the same
  run, and is the same on every run (on the full square it is a
  property of the shape). It is also compared with the Chebyshev tile
  distance from the seed tile, which the argument in the README of
  video/assets/tiles predicts for a full square.
- The close-up: a one-warp kernel replays the probed visit with the
  chapter's own device functions (_row_fill, _diag_step, _col_fill, the
  loop and the any_sync test of the rounds kernel) and records every
  sweep. Its last words must equal the twin's closed words of that
  visit, its trip count the twin's, and (the chosen tile is visited
  once) the chapter kernel's final reached words for that tile.
  Whether a tile is visited once is a timing effect, so the close-up's
  twin run is repeated (up to CLOSEUP_ATTEMPTS times) until the pinned
  tile is; the JSON records the attempts.

Visits and per-round frontier sizes depend on timing (a neighbour's
export may land after a tile's closure and queue it again), so they are
recorded per run; first-open rounds, rounds and the fill do not move.

Index order is the repo's [x, y] everywhere: x is the PNG column, y the
PNG row, a tile is 32 x-rows (lanes) by one 32-bit word along y.

Saved to video/assets/tiles/ (see its README.md):

- ch07_tiles_<UTC>.npz   the arrays
- ch07_tiles_<UTC>.json  the facts, the checks and the provenance

Memory: about 2 GB host at 64 Mpx (the image, the oracle's visited,
depth and queue); about 0.5 GB device. A minute or two in all.

Run:  uv run python -m flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks.video_tiles --images images/input
      (--quick: a 1024^2 square, every check, no files written)
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import gc
import hashlib
import inspect
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
from numba import cuda

from .. import flood_fill as ff
from .. import kernels as k7
from ..kernels import FULL, _col_fill, _diag_step, _row_fill
from ...ch03_gpu_1blob_nblock import scenes
from ...ch05_gpu_nblob_nblock import scenes as S5
from ...ch08_gpu_nblob_tiles.cpu_oracle import cpu_label
from ....shared.cpu_oracle import cpu_flood_fill_8

TILE = k7.TILE
TPB = 128                          # the chapter's default block (w4)
CONN = 8
SWITCHES = dict(ff.DEFAULT_SWITCHES)
N_RUNS = 5                         # twin runs on the square
PROBE_VISITS = 4                   # visits kept per probed tile
MAX_SWEEPS = 3 * 64                # the close-up replay's sweep buffer
PNG_MD5 = "47639eb37f973b5092ec2c4db86f45a4"   # = ch08's pinned input_blobs
# The close-up: the largest blob (rank 0, 76,090 px, a rough-edged
# clump) and its tile (176, 142), picked by eye among the 63 candidates
# that pass score_probe on the 12 largest blobs: the entry is a partial
# stripe on one border, the edge is concave, and every sweep of the first
# two trips adds pixels (in the survey run: 17, 86, 109, 181, 387, 481,
# 541, 543 reached). The entry bits are this run's, so they may differ.
CLOSEUP_BLOB_RANK = 0
CLOSEUP_TILE = "176,142"
CLOSEUP_ATTEMPTS = 50
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    *[os.pardir] * 5))
OUT_DIR = os.path.join(REPO, "video", "assets", "tiles")
SQUARE = "sq_8000_center"
SESSION = "tiles_20261016T064724Z"   # the chapter's committed session

# ------------------------------------------------------------ the twin

_PATCHES = (
    # the export buffers ride at the end of the argument list
    ("                      aligned, tinfo, painted):\n",
     "                      aligned, tinfo, painted, first_open,\n"
     "                      visit_count, round_size, probe_slot, probe,\n"
     "                      probe_round):\n"),
    # the frontier size of every round
    ("            if n_cur > peak:\n"
     "                peak = n_cur\n",
     "            if n_cur > peak:\n"
     "                peak = n_cur\n"
     "            if tid == 0 and rnd < round_size.shape[0]:   # EXPORT\n"
     "                round_size[rnd] = n_cur\n"),
    # per-tile visits, and the probed tiles' words, after the closure
    ("                if instrumented:\n"
     "                    c_visits = uint32(c_visits + U1)\n"
     "                    c_iters = uint32(c_iters + it)\n",
     "                if instrumented:\n"
     "                    c_visits = uint32(c_visits + U1)\n"
     "                    c_iters = uint32(c_iters + it)\n"
     "                v_old = int32(0)                         # EXPORT\n"
     "                if lane == 0:\n"
     "                    v_old = cuda.atomic.add(visit_count,\n"
     "                                            uint32(t_own), 1)\n"
     "                cuda.syncwarp()\n"
     "                v_old = cuda.shfl_sync(FULL, v_old, 0)\n"
     "                s_pr = probe_slot[uint32(t_own)]\n"
     "                if s_pr >= 0 and v_old < probe.shape[1]:\n"
     "                    probe[s_pr, v_old, lane, 0] = m\n"
     "                    probe[s_pr, v_old, lane, 1] = f0\n"
     "                    probe[s_pr, v_old, lane, 2] = f\n"
     "                    probe[s_pr, v_old, lane, 3] = it\n"
     "                    if lane == 0:\n"
     "                        probe_round[s_pr, v_old] = rnd\n"),
    # the first enqueue ever: the only atomic.max that returns -1
    ("                    old_st = cuda.atomic.max(stamp, uint32(ttx * wpr + tty),\n"
     "                                             next_stamp)\n",
     "                    old_st = cuda.atomic.max(stamp, uint32(ttx * wpr + tty),\n"
     "                                             next_stamp)\n"
     "                    if old_st < 0:                       # EXPORT\n"
     "                        first_open[uint32(ttx * wpr + tty)] = next_stamp\n"),
)

_twin_cache = {}


def export_twin():
    """The chapter's rounds kernel (8-conn, instrumented, default
    switches) with the export insertions: (kernel, cooperative blocks at
    TPB, registers per thread)."""
    if "k" in _twin_cache:
        return _twin_cache["k"]
    src = inspect.getsource(k7._make_rounds_kernel)
    for old, new in _PATCHES:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"export patch anchor found {n} times, not "
                               f"once (the kernel changed?):\n{old}")
        src = src.replace(old, new)
    ns = dict(vars(k7))
    exec(compile(src, "<ch07 export twin>", "exec"), ns)
    kern = ns["_make_rounds_kernel"](
        CONN == 8, True, None, **ff.rounds_switches(SWITCHES))
    # compile on a tiny image (blocks=1), then ask for the grid
    tiny = np.full((8, 8, 3), 255, np.uint8)
    tiny[1, 1] = (255, 0, 0)
    eng = ff.TileFill(8, 8, connectivity=CONN, threads_per_block=32,
                      blocks=1, _skip_warmup=True, **SWITCHES)
    bufs = ExportBuffers(eng, probes=())
    bufs.reset(eng, 1, 1)
    eng.kernel_fn = _TwinLaunch(kern, 1, bufs.args())
    eng.run(cuda.to_device(tiny), 1, 1)
    cuda.synchronize()
    ov = next(iter(kern.overloads.values()))
    blocks = ov.max_cooperative_grid_blocks(TPB)
    regs = kern.get_regs_per_thread()
    regs = int(next(iter(regs.values()))) if isinstance(regs, dict) \
        else int(regs)
    _twin_cache["k"] = (kern, blocks, regs)
    return _twin_cache["k"]


class _TwinLaunch:
    """Stands in for TileFill.kernel_fn, so TileFill.run() launches the
    twin with the chapter's own pack, init and tile paint around it: the
    twin's cooperative grid, the export buffers appended."""

    def __init__(self, kern, blocks, extra):
        self.kern, self.blocks, self.extra = kern, blocks, extra

    def __getitem__(self, cfg):
        launch = self.kern[self.blocks, cfg[1]]
        return lambda *args: launch(*args, *self.extra)


class ExportBuffers:
    def __init__(self, eng, probes, max_rounds=1 << 16):
        n = eng.n_tiles
        self.first_open = cuda.device_array(n, np.int32)
        self.visit_count = cuda.device_array(n, np.int32)
        self.round_size = cuda.device_array(max_rounds, np.int32)
        slot = np.full(n, -1, np.int32)
        for s, t in enumerate(probes):
            slot[t] = s
        self.probe_slot = cuda.to_device(slot)
        k = max(1, len(probes))
        self.probe = cuda.device_array((k, PROBE_VISITS, 32, 4), np.uint32)
        self.probe_round = cuda.device_array((k, PROBE_VISITS), np.int32)
        self.k = len(probes)

    def reset(self, eng, sx, sy):
        """Host-side, off any clock: first_open = -1 but the seed tile's 0."""
        fo = np.full(eng.n_tiles, -1, np.int32)
        fo[eng.seed_args(sx, sy)[2]] = 0
        self.first_open.copy_to_device(fo)
        self.visit_count.copy_to_device(np.zeros(eng.n_tiles, np.int32))
        self.round_size.copy_to_device(
            np.zeros(self.round_size.shape[0], np.int32))
        self.probe.copy_to_device(np.zeros(self.probe.shape, np.uint32))
        self.probe_round.copy_to_device(
            np.full(self.probe_round.shape, -1, np.int32))

    def args(self):
        return (self.first_open, self.visit_count, self.round_size,
                self.probe_slot, self.probe, self.probe_round)


def chapter_engine(width, height):
    """The chapter's default engine, instrumented (flood_fill()'s)."""
    return ff.TileFill(width, height, connectivity=CONN,
                       threads_per_block=TPB, **SWITCHES)


def run_chapter(eng, d_img, sx, sy):
    """The chapter's own kernel: counters, reached words, stamps."""
    eng.run(d_img, sx, sy)
    cuda.synchronize()
    return (eng.read_counters(), eng.reached_words(),
            eng.stamp.copy_to_host())


def run_twin(eng, d_img, sx, sy, bufs):
    kern, blocks, _ = export_twin()
    saved = eng.kernel_fn
    bufs.reset(eng, sx, sy)
    eng.kernel_fn = _TwinLaunch(kern, min(blocks, eng.coop_max_blocks),
                                bufs.args())
    try:
        eng.run(d_img, sx, sy)
        cuda.synchronize()
    finally:
        eng.kernel_fn = saved
    c = eng.read_counters()
    shape = (eng.tiles_x, eng.words_per_row)
    out = {"counters": c, "reached": eng.reached_words(),
           "stamp": eng.stamp.copy_to_host(),
           "first_open": bufs.first_open.copy_to_host().reshape(shape),
           "visit_count": bufs.visit_count.copy_to_host().reshape(shape),
           "round_size": bufs.round_size.copy_to_host()[:c["rounds"]]}
    if bufs.k:
        out["probe"] = bufs.probe.copy_to_host()[:bufs.k]
        out["probe_round"] = bufs.probe_round.copy_to_host()[:bufs.k]
    return out


def check(ok, what, checks):
    checks[what] = bool(ok)
    if not ok:
        raise AssertionError(f"check failed: {what}")


def oracle_words(img, sx, sy, wpr):
    """The @njit oracle's visited set as packed words, and its levels."""
    visited, depth, levels, filled = cpu_flood_fill_8(img, sx, sy)
    del depth
    w, h = visited.shape
    bits = np.zeros((w, wpr * 32), np.uint8)
    bits[:, :h] = visited
    del visited
    words = np.packbits(bits, axis=1, bitorder="little").view(np.uint32)
    return words, int(levels), int(filled)


def chebyshev(tiles_x, wpr, seed_tile_xy):
    tx = np.arange(tiles_x)[:, None]
    ty = np.arange(wpr)[None, :]
    return np.maximum(np.abs(tx - seed_tile_xy[0]),
                      np.abs(ty - seed_tile_xy[1]))


# ------------------------------------------------------------ export 1

def square_export(side, checks):
    img, sx, sy = scenes.square_scene(side, side, side, side)
    eng = chapter_engine(side, side)
    d_img = cuda.to_device(img)
    d_pristine = cuda.to_device(img)
    chap_c, chap_reached, chap_stamp = run_chapter(eng, d_img, sx, sy)
    print(f"  chapter kernel: rounds {chap_c['rounds']}, visits "
          f"{chap_c['tile_visits']:,}, peak {chap_c['peak_round_tiles']:,}",
          flush=True)
    o_words, o_levels, o_filled = oracle_words(img, sx, sy,
                                               eng.words_per_row)
    del img
    gc.collect()
    check(np.array_equal(chap_reached, o_words),
          f"{SQUARE}: chapter fill == @njit oracle visited", checks)
    filled = int(np.unpackbits(chap_reached.view(np.uint8)).sum())
    check(filled == o_filled == side * side,
          f"{SQUARE}: filled == oracle == every pixel", checks)
    seed_tile = (sx // TILE, sy // TILE)
    cheb = chebyshev(eng.tiles_x, eng.words_per_row, seed_tile)
    bufs = ExportBuffers(eng, probes=())
    runs = []
    for i in range(N_RUNS):
        d_img.copy_to_device(d_pristine)
        r = run_twin(eng, d_img, sx, sy, bufs)
        c = r["counters"]
        print(f"  twin run {i}: rounds {c['rounds']}, visits "
              f"{c['tile_visits']:,}, peak {c['peak_round_tiles']:,}",
              flush=True)
        tag = f"{SQUARE} twin run {i}"
        check(np.array_equal(r["reached"], chap_reached),
              f"{tag}: fill == chapter kernel's, bit for bit", checks)
        check(c["rounds"] == chap_c["rounds"],
              f"{tag}: rounds == chapter kernel's", checks)
        check(int(r["round_size"].sum()) == c["tile_visits"]
              == int(r["visit_count"].sum()),
              f"{tag}: sum(round_size) == TILE_VISITS == sum(visit_count)",
              checks)
        check(int(r["round_size"].max()) == c["peak_round_tiles"],
              f"{tag}: max(round_size) == PEAK_ROUND", checks)
        check((r["first_open"] >= 0).all() and (r["visit_count"] >= 1).all(),
              f"{tag}: every tile opened and visited", checks)
        check((r["first_open"] <= r["stamp"]).all(),
              f"{tag}: first_open <= stamp (the last round queued)", checks)
        if runs:
            check(np.array_equal(r["first_open"], runs[0]["first_open"]),
                  f"{tag}: first_open == run 0's", checks)
        runs.append(r)
    fo = runs[0]["first_open"]
    cheb_ok = bool(np.array_equal(fo, cheb))
    checks[f"{SQUARE}: first_open == Chebyshev tile distance"] = cheb_ok
    print(f"  first_open == Chebyshev distance: {cheb_ok}", flush=True)
    opened = np.bincount(fo.ravel(), minlength=chap_c["rounds"])
    check(int(opened.sum()) == eng.n_tiles,
          f"{SQUARE}: tiles opened per round sum to n_tiles", checks)
    pick = 0
    arrays = {
        "sq_first_open": fo.astype(np.int16),
        "sq_visit_count": runs[pick]["visit_count"].astype(np.int16),
        "sq_last_round": runs[pick]["stamp"].astype(np.int16),
        "sq_round_size": runs[pick]["round_size"].astype(np.int32),
        "sq_opened_per_round": opened.astype(np.int32),
        "sq_chapter_last_round": chap_stamp.astype(np.int16),
    }
    facts = {
        "scene": SQUARE, "width": side, "height": side, "seed": [sx, sy],
        "seed_tile": list(seed_tile), "connectivity": CONN,
        "tiles": [eng.tiles_x, eng.words_per_row], "n_tiles": eng.n_tiles,
        "filled": filled, "oracle_levels": o_levels,
        "chapter_kernel": chap_c,
        "twin_runs": [{"rounds": r["counters"]["rounds"],
                       "tile_visits": r["counters"]["tile_visits"],
                       "closure_iters": r["counters"]["closure_iters"],
                       "peak_round_tiles": r["counters"]["peak_round_tiles"],
                       "tiles_visited_more_than_once":
                           int((r["visit_count"] > 1).sum()),
                       "max_visits_per_tile": int(r["visit_count"].max())}
                      for r in runs],
        "picked_run": pick,
        "first_open_max": int(fo.max()),
        "first_open_equals_chebyshev": cheb_ok,
        "opened_per_round": [int(v) for v in opened],
        "round_size_picked_run": [int(v) for v in runs[pick]["round_size"]],
    }
    del eng, d_img, d_pristine, runs, chap_reached, o_words
    gc.collect()
    return arrays, facts


# ------------------------------------------------------------ export 2

@cuda.jit
def closure_trace_kernel(m_in, f_in, out, n_out):
    """One warp replays one visit's closure with the rounds kernel's own
    device functions and loop, recording f after every sweep:
    out[0] the entry words, then (row fill, diagonal, column fill) per
    trip. n_out[0] = sweeps written, n_out[1] = trips."""
    lane = cuda.laneid
    m = m_in[lane]
    f = f_in[lane]
    out[0, lane] = f
    k = 0
    trips = 0
    while True:
        old = f
        f = _row_fill(f, m)
        if k + 1 < out.shape[0]:
            out[k + 1, lane] = f
        f = _diag_step(f, m, lane)
        if k + 2 < out.shape[0]:
            out[k + 2, lane] = f
        f = _col_fill(f, m, lane)
        if k + 3 < out.shape[0]:
            out[k + 3, lane] = f
        k += 3
        trips += 1
        if not cuda.any_sync(FULL, f != old):
            break
    if lane == 0:
        n_out[0] = k + 1
        n_out[1] = trips


def replay(m, f0):
    d_out = cuda.to_device(np.zeros((MAX_SWEEPS + 1, 32), np.uint32))
    d_n = cuda.to_device(np.zeros(2, np.int32))
    m = np.ascontiguousarray(m, np.uint32)
    f0 = np.ascontiguousarray(f0, np.uint32)
    closure_trace_kernel[1, 32](cuda.to_device(m), cuda.to_device(f0),
                                d_out, d_n)
    cuda.synchronize()
    n, trips = (int(v) for v in d_n.copy_to_host())
    if n > MAX_SWEEPS + 1:
        raise RuntimeError(f"closure took {trips} trips, more than the "
                           f"buffer's {MAX_SWEEPS // 3}")
    return d_out.copy_to_host()[:n], trips


def tile_words(words, tx, ty, width):
    """The 32 words of tile (tx, ty) from (width, wpr) words, rows past
    the image as 0 (as the kernel's lanes hold them)."""
    out = np.zeros(32, np.uint32)
    x0 = tx * TILE
    n = min(32, width - x0)
    out[:n] = words[x0:x0 + n, ty]
    return out


def _bits(words):
    return np.unpackbits(np.ascontiguousarray(words).view(np.uint8)
                         .reshape(32, 4), axis=1, bitorder="little")


def score_probe(m, f0, f, trips, visits, chap_tile, other_red):
    """Close-up candidates: one visit closes the tile's whole part of the
    blob, the entry is a border stripe (not the seed), the tile is an
    edge (30-75% red), no other blob's pixels, and the most closure
    trips. Returns a score or None."""
    if visits != 1 or not np.array_equal(f, chap_tile) or other_red:
        return None
    red = int(_bits(m).sum())
    if not 0.30 * 1024 <= red <= 0.75 * 1024:
        return None
    entry = int(_bits(f0).sum())
    if entry < 2:
        return None
    return trips * 10000 + red - entry


def closeup_export(png_path, checks, blob_rank=0, quick=False, tile=None):
    img = S5.png_scene(png_path)[0]
    if quick:
        img = np.ascontiguousarray(img[:1024, :1024])
    width, height = img.shape[:2]
    lab, n_blobs = cpu_label(img, CONN)
    u, cnt = np.unique(lab[lab >= 0], return_counts=True)
    order = np.argsort(-cnt, kind="stable")
    blob = int(u[order[blob_rank]])
    blob_px = int(cnt[order[blob_rank]])
    xs, ys = np.nonzero(lab == blob)
    # seed: the blob pixel nearest its centroid
    cx, cy = xs.mean(), ys.mean()
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    sx, sy = int(xs[i]), int(ys[i])
    eng = chapter_engine(width, height)
    d_img = cuda.to_device(img)
    d_pristine = cuda.to_device(img)
    chap_c, chap_reached, _ = run_chapter(eng, d_img, sx, sy)
    o_words, o_levels, o_filled = oracle_words(img, sx, sy,
                                               eng.words_per_row)
    check(np.array_equal(chap_reached, o_words),
          "close-up blob: chapter fill == @njit oracle visited", checks)
    check(o_filled == blob_px,
          "close-up blob: oracle fill == the blob's pixel count", checks)
    # candidates: tiles holding blob pixels
    tx_all = np.unique(np.stack([xs // TILE, ys // TILE], 1), axis=0)
    probes = [int(a * eng.words_per_row + b) for a, b in tx_all]
    bufs = ExportBuffers(eng, probes=probes)
    mask_words = None
    red_all = (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)
    seed_tile = eng.seed_args(sx, sy)[2]
    want = None if tile is None else int(tile[0] * eng.words_per_row
                                         + tile[1])
    # A pinned tile qualifies only on runs where it is visited once (a
    # timing effect), so the twin run is repeated until it does.
    for attempt in range(1, CLOSEUP_ATTEMPTS + 1):
        d_img.copy_to_device(d_pristine)
        r = run_twin(eng, d_img, sx, sy, bufs)
        check(np.array_equal(r["reached"], chap_reached),
              "close-up blob: twin fill == chapter kernel's", checks)
        check(r["counters"]["rounds"] == chap_c["rounds"],
              "close-up blob: twin rounds == chapter kernel's", checks)
        if mask_words is None:
            mask_words = eng.mask.copy_to_host()
        scored = []
        for s, t in enumerate(probes):
            if t == seed_tile:
                continue
            tx, ty = divmod(t, eng.words_per_row)
            visits = int(r["visit_count"].ravel()[t])
            p = r["probe"][s, 0]
            m, f0, f, trips = p[:, 0], p[:, 1], p[:, 2], int(p[0, 3])
            x0, y0 = tx * TILE, ty * TILE
            sub_lab = lab[x0:x0 + TILE, y0:y0 + TILE]
            other = bool(((sub_lab >= 0) & (sub_lab != blob)).any())
            sc = score_probe(m, f0, f, trips, visits,
                             tile_words(chap_reached, tx, ty, width), other)
            if sc is not None:
                scored.append((sc, s, t, trips, int(_bits(m).sum()),
                               int(_bits(f0).sum())))
        scored.sort(reverse=True)
        hit = [c for c in scored if c[2] == want] if want is not None \
            else scored[:1]
        if hit:
            break
    else:
        raise RuntimeError(f"tile {tile} was no close-up candidate in "
                           f"{CLOSEUP_ATTEMPTS} runs")
    for sc, s, t, trips, red, ent in scored[:12]:
        print(f"    candidate tile {divmod(t, eng.words_per_row)}: {trips} "
              f"trips, {red} red px, {ent} entry px", flush=True)
    s, t = hit[0][1], hit[0][2]
    tx, ty = divmod(t, eng.words_per_row)
    p = r["probe"][s, 0]
    m, f0, f, trips = (p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy(),
                       int(p[0, 3]))
    sweeps, rtrips = replay(m, f0)
    check(np.array_equal(m, tile_words(mask_words, tx, ty, width)),
          "close-up: probed mask words == the packed mask", checks)
    check(np.array_equal(sweeps[-1], f) and rtrips == trips,
          "close-up: replay's last words and trips == the twin's visit",
          checks)
    chap_tile = tile_words(chap_reached, tx, ty, width)
    check(np.array_equal(sweeps[-1], chap_tile),
          "close-up: replay's last words == chapter kernel's reached "
          "words of the tile", checks)
    check(not (f0 & ~m).any() and not (sweeps[-1] & ~m).any(),
          "close-up: entry and fill inside the mask", checks)
    names = ["entry"] + ["row", "diag", "col"] * trips
    # context: 7 x 7 tiles around the close-up, clipped to the image
    cx0, cy0 = max(0, (tx - 3) * TILE), max(0, (ty - 3) * TILE)
    cx1, cy1 = min(width, (tx + 4) * TILE), min(height, (ty + 4) * TILE)
    fill_bits = ff.unpack_bits(chap_reached[cx0:cx1], height)
    ctx_red = red_all[cx0:cx1, cy0:cy1]
    ctx_fill = fill_bits[:, cy0:cy1].astype(bool)
    ctx_blob = lab[cx0:cx1, cy0:cy1] == blob
    check(np.array_equal(ctx_fill, ctx_blob),
          "close-up context: fill == the oracle's blob", checks)
    tfo = r["first_open"]
    arrays = {
        "cu_mask": m, "cu_entry": f0, "cu_sweeps": sweeps,
        "cu_final_chapter": chap_tile,
        "cu_ctx_red": ctx_red, "cu_ctx_fill": ctx_fill,
        "cu_ctx_first_open": tfo[cx0 // TILE:(cx1 + TILE - 1) // TILE,
                                 cy0 // TILE:(cy1 + TILE - 1) // TILE]
        .astype(np.int16),
    }
    entry_bits = _bits(f0)
    facts = {
        "image": os.path.basename(png_path), "png_md5": PNG_MD5,
        "quick_crop": quick, "width": width, "height": height,
        "blobs": n_blobs, "blob_label": blob, "blob_rank_by_size": blob_rank,
        "blob_px": blob_px, "seed": [sx, sy],
        "seed_tile": list(divmod(seed_tile, eng.words_per_row)),
        "blob_rounds": chap_c["rounds"], "blob_oracle_levels": o_levels,
        "tile": [tx, ty], "tile_origin": [tx * TILE, ty * TILE],
        "tile_first_round": int(r["probe_round"][s, 0]),
        "tile_visits": int(r["visit_count"].ravel()[t]),
        "trips": trips, "sweep_names": names,
        "red_px": int(_bits(m).sum()),
        "entry_px": int(entry_bits.sum()),
        "entry_lanes": [int(v) for v in np.nonzero(entry_bits.any(1))[0]],
        "entry_bits": [int(v) for v in np.nonzero(entry_bits.any(0))[0]],
        "reached_px_after_sweep": [int(_bits(w).sum()) for w in sweeps],
        "context_origin": [cx0, cy0], "context_size": [cx1 - cx0, cy1 - cy0],
        "candidates_probed": len(probes) - 1,
        "candidates_kept": len(scored),
        "twin_attempts": attempt,
        "tile_picked_by": "--tile" if tile is not None else "score",
        "chapter_kernel": chap_c,
    }
    print(f"  close-up: blob {blob} ({blob_px:,} px), tile ({tx}, {ty}), "
          f"round {facts['tile_first_round']}, {trips} trips, "
          f"{facts['red_px']} red px, {facts['entry_px']} entry px",
          flush=True)
    return arrays, facts


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", default=os.environ.get("FLOOD_FILL_IMAGES"),
                    help="folder holding input_blobs.png")
    ap.add_argument("--blob-rank", type=int, default=CLOSEUP_BLOB_RANK,
                    help="close-up blob: 0 = the largest")
    ap.add_argument("--tile", default=CLOSEUP_TILE,
                    help="close-up tile 'tx,ty', or 'auto' for the best "
                         "score (--quick always scores)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    if not args.images:
        raise SystemExit("give --images DIR (input_blobs.png)")
    png = os.path.join(args.images, "input_blobs.png")
    md5 = hashlib.md5(open(png, "rb").read()).hexdigest()
    if md5 != PNG_MD5:
        raise SystemExit(f"{png}: md5 {md5}, expected {PNG_MD5}")
    checks = {}
    kern, blocks, regs = export_twin()
    side = 1024 if args.quick else 8000
    print(f"export twin: {regs} registers, {blocks} cooperative blocks of "
          f"{TPB}; chapter default: {ff.regs_per_thread(CONN, **SWITCHES)} "
          f"registers, {ff.max_blocks(TPB, CONN, **SWITCHES)} blocks",
          flush=True)
    print(f"{SQUARE if not args.quick else 'sq_1024'}:", flush=True)
    sq_arrays, sq_facts = square_export(side, checks)
    print("close-up:", flush=True)
    tile = (None if args.quick or args.tile == "auto"
            else tuple(int(v) for v in args.tile.split(",")))
    cu_arrays, cu_facts = closeup_export(png, checks, args.blob_rank,
                                         args.quick, tile)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "what": "ch07 tile BFS, act 3 floors of the chain cut (v5)",
        "script": "src/flood_fill_cuda/chapters/ch07_gpu_1blob_tiles/"
                  "benchmarks/video_tiles.py",
        "commit": git_commit(), "utc": stamp,
        "device": cuda.get_current_device().name.decode()
        if isinstance(cuda.get_current_device().name, bytes)
        else str(cuda.get_current_device().name),
        "chapter_session": SESSION,
        "config": {"connectivity": CONN, "threads_per_block": TPB,
                   "switches": SWITCHES, "instrumented": True},
        "twin": {"registers": regs, "blocks": blocks,
                 "chapter_registers": ff.regs_per_thread(CONN, **SWITCHES),
                 "chapter_blocks": ff.max_blocks(TPB, CONN, **SWITCHES),
                 "patch_marks": "EXPORT"},
        "index_order": "[x, y]: x = PNG column, y = PNG row; tile (tx, ty) "
                       "= x in [32 tx, 32 tx + 32), y in [32 ty, 32 ty + "
                       "32); lane i = x0 + i, bit j = y0 + j",
        "square": sq_facts, "closeup": cu_facts, "checks": checks,
    }
    print(f"{sum(checks.values())} of {len(checks)} checks hold", flush=True)
    if args.quick:
        print("--quick: nothing written")
        return
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"ch07_tiles_{stamp}")
    np.savez_compressed(base + ".npz", **sq_arrays, **cu_arrays)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    print(f"wrote {os.path.basename(base)}.npz and .json")


if __name__ == "__main__":
    main()
