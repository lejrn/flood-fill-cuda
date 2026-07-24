"""The grand table: every approach from every chapter, one common grid.

Rows are shapes x scales (squares, disks, serpentine, comb, two-blob
pairs, random noise, plus any external PNGs present in images/input/);
columns are ALL kernel variants from ch01-ch05 plus the two CPU bars.
Cells are median kernel ms — the same measurement every chapter's own
benchmark reports.

Every cell answers "what would it cost THIS stage to do the WHOLE
job", even beyond its native contract:

- one-blob kernels (ch01-ch03) on a two-blob scene: MEASURED as the
  sum of one call per blob (what using that stage would really cost).
- one/two-blob kernels on an N-blob scene: a full loop is hours at
  755k blobs, so the cell is an ESTIMATE — one call per blob (per pair
  for ch04), median per-call kernel ms over a k-blob sample x the call
  count. Estimated cells are marked, carry their formula, and never
  win a row.
- "—" survives only where the job cannot be expressed at all: ch04's
  kernel takes exactly two blobs in two components, so one-blob scenes
  are outside its input space.
- pure Python runs everything (single run above 300k px) up to a 20M
  red-px cap.

Every shape is the same component set under 4- and 8-connectivity
(solid shapes, gaps >= 8), so cross-connectivity cells compare the same
job; the per-row crosscheck asserts every completed (non-estimated)
cell agrees on the filled pixel count.

Run:  PYTHONUNBUFFERED=1 uv run python -m flood_fill_cuda.overview.bench
Writes overview_<stamp>.json to results/overview/benchmark_results/.
Budget ~20-35 min (17 rows x up to 20 columns; sampling for estimates).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import gc
import json
import statistics
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from numba import cuda

from ..shared import results_paths
from ..shared import scenes as shared_scenes
from ..chapters.ch01_gpu_1blob_1block import flood_fill as ff1
from ..chapters.ch01_gpu_1blob_1block import cpu_oracle as oracle1
from ..chapters.ch02_gpu_1blob_2block import flood_fill as ff2
from ..chapters.ch03_gpu_1blob_nblock import flood_fill as ff3
from ..chapters.ch04_gpu_2blob_nblock import flood_fill as ff4
from ..chapters.ch04_gpu_2blob_nblock import scenes as ch04_scenes
from ..chapters.ch04_gpu_2blob_nblock import cpu_oracle as oracle4
from ..chapters.ch05_gpu_nblob_nblock import flood_fill as ff5
from ..chapters.ch05_gpu_nblob_nblock import scenes as ch05_scenes
from ..chapters.ch05_gpu_nblob_nblock import cpu_oracle as oracle5

RESULTS_DIR = results_paths.results_dir("overview", "benchmark_results")

TPB = 256
GPU_REPEATS = 5
NJIT_REPEATS = 3
PURE_CAP = 20_000_000        # red px above which pure Python is "capped"
PURE_SINGLE = 300_000        # above this: single pure-Python run, not 3
EST_SAMPLE = 16              # blobs sampled per estimated cell
EST_SAMPLE_HUGE = 6          # ... on images past ~20M px


def _lex_min_seed(img):
    """The canonical seed rule from ch05, on the host: the red pixel
    with the smallest (x, y) — for feeding seedless-born scenes (comb)
    to the seed-needing chapters."""
    reds = np.argwhere((img[..., 0] == 255) & (img[..., 1] == 0)
                       & (img[..., 2] == 0))
    x, y = reds[np.lexsort((reds[:, 1], reds[:, 0]))][0]
    return int(x), int(y)


def _component_seeds(img):
    """One seed per blob: the oracle's canonical labels ARE the lex-min
    linear indices, so the unique label values decode straight into one
    red pixel per component."""
    label, _ = oracle5.cpu_label_components(img)
    height = img.shape[1]
    roots = np.unique(label[label >= 0])
    return [(int(v) // height, int(v) % height) for v in roots]


_CONN4 = ((1, 0), (0, 1), (-1, 0), (0, -1))
_CONN8 = _CONN4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def pure_python_bfs(img, seeds, connectivity=4):
    """The ch00 bar: classic sequential BFS on a copy — one shared
    visited array, one BFS per seed (blobs are disjoint). 4-conn on the
    solid shapes (the historical ch00 bar; identical fill there), 8-conn
    on N-blob rows where the components are 8-conn by definition."""
    out = img.copy()
    width, height = out.shape[0], out.shape[1]
    offsets = _CONN4 if connectivity == 4 else _CONN8
    visited = np.zeros((width, height), dtype=np.uint8)
    filled = 0
    for seed_x, seed_y in seeds:
        if visited[seed_x, seed_y]:
            continue
        visited[seed_x, seed_y] = 1
        queue = deque([(seed_x, seed_y)])
        while queue:
            x, y = queue.popleft()
            out[x, y, 0] = 0
            out[x, y, 1] = 0
            out[x, y, 2] = 255
            filled += 1
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height
                        and not visited[nx, ny]):
                    p = out[nx, ny]
                    if p[0] == 255 and p[1] == 0 and p[2] == 0:
                        visited[nx, ny] = 1
                        queue.append((nx, ny))
    return filled


# ------------------------------------------------------------------ rows
def _one(builder):
    def build():
        img, sx, sy = builder()
        return {"kind": "one", "img": img, "sx": sx, "sy": sy}
    return build


def _one_seedless(builder):
    def build():
        img, _ = builder()
        sx, sy = _lex_min_seed(img)
        return {"kind": "one", "img": img, "sx": sx, "sy": sy}
    return build


def _two(builder):
    def build():
        img, seeds = builder()
        return {"kind": "two", "img": img, "seeds": seeds}
    return build


def _n(builder):
    def build():
        img, n_blobs = builder()
        return {"kind": "n", "img": img, "n_blobs": n_blobs}
    return build


# (key, family, note, est_filled, build)
ROWS = [
    ("sq_256", "square", "256², 128² blob, center seed", 16_384,
     _one(lambda: shared_scenes.square_scene(256, 256, 128, 128))),
    ("sq_1024", "square", "1024², 512² blob", 262_144,
     _one(lambda: shared_scenes.square_scene(1024, 1024, 512, 512))),
    ("sq_4000", "square", "4000², 2000² blob (4M px)", 4_000_000,
     _one(lambda: shared_scenes.square_scene(4000, 4000, 2000, 2000))),
    ("disk_256", "disk", "r=120 disk", 45_000,
     _one(lambda: shared_scenes.disk_scene(256, 256, 120))),
    ("disk_1024", "disk", "r=480 disk", 724_000,
     _one(lambda: shared_scenes.disk_scene(1024, 1024, 480))),
    ("disk_4000", "disk", "r=1900 disk (11.3M px)", 11_340_000,
     _one(lambda: shared_scenes.disk_scene(4000, 4000, 1900))),
    ("serp_128", "serpentine", "1-px snake, ~8k levels", 8_200,
     _one(lambda: shared_scenes.serpentine_scene(128, 128))),
    ("serp_256", "serpentine", "1-px snake, ~33k levels", 33_000,
     _one(lambda: shared_scenes.serpentine_scene(256, 256))),
    ("comb_24", "comb", "24 teeth, 96×64", 1_900,
     _one_seedless(lambda: ch05_scenes.comb_scene(
         96, 64, teeth=24, tooth_len=64, spine_w=6))),
    ("comb_2000", "comb", "2000 teeth, 3000×4001 (4.8M px)", 4_830_000,
     _one_seedless(lambda: ch05_scenes.comb_scene(
         3000, 4001, teeth=2000, tooth_len=2400, spine_w=8))),
    ("two_sq_300", "two squares", "2 × 300² squares, 700×400", 180_000,
     _two(lambda: ch04_scenes.two_squares_scene(700, 400, 300, 300,
                                                gap=8))),
    ("two_sq_2800", "two squares", "2 × 7.8M px squares, 6000×3200",
     15_680_000,
     _two(lambda: ch04_scenes.two_squares_scene(6000, 3200, 2800, 2800,
                                                gap=16))),
    ("asym_4000_800", "asymmetric", "16M + 0.6M px pair, 5200×4400",
     16_640_000,
     _two(lambda: ch04_scenes.asym_squares_scene(5200, 4400, 4000, 800,
                                                 gap=16))),
    ("random_1000", "random noise", "density 0.3, ~thousands of blobs",
     300_000,
     _n(lambda: ch05_scenes.random_blobs_scene(1000, 1000, density=0.3,
                                               rng_seed=0))),
    ("random_4000", "random noise", "density 0.3, ~755k blobs", 4_800_000,
     _n(lambda: ch05_scenes.random_blobs_scene(4000, 4000, density=0.3,
                                               rng_seed=0))),
]

# External PNGs (gitignored inputs — rows appear only when present).
_PNG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        *[".."] * 3, "images", "input")
for _key, _fname, _note, _est in (
        ("png_blobs", "input_blobs.png",
         "external PNG, 9000², ~2.5k blobs (13.5M red px)", 13_451_960),
        ("png_blocks", "input_blocks.png",
         "external PNG, 1000², ~21.6k blocks", 387_587)):
    _p = os.path.abspath(os.path.join(_PNG_DIR, _fname))
    if os.path.exists(_p):
        ROWS.append((_key, "external PNG", _note, _est,
                     _n(lambda p=_p: ch05_scenes.png_scene(p))))


# --------------------------------------------------------------- columns
# (key, group, label, kinds it can attempt, runner) — runner(ctx) returns
# the chapter's result object (has .kernel_ms, .filled).
COLS = [
    ("ch01_ring", "ch01 · 1 block", "ring", ("one",),
     lambda c: ff1.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB, variant="ring")),
    ("ch01_spill", "ch01 · 1 block", "spill", ("one",),
     lambda c: ff1.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB, variant="spill")),
    ("ch02_split", "ch02 · 2 blocks", "split", ("one",),
     lambda c: ff2.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB, kernel="split")),
    ("ch02_global", "ch02 · 2 blocks", "global", ("one",),
     lambda c: ff2.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB, kernel="global")),
    ("ch02_dirsplit", "ch02 · 2 blocks", "dirsplit", ("one",),
     lambda c: ff2.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB, kernel="dirsplit")),
    ("ch02_pinned", "ch02 · 2 blocks", "pinned (spread, tpb 768)",
     ("one",),
     lambda c: ff2.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=768, kernel="pinned",
                              placement="spread")),
    ("ch03_conn4", "ch03 · N blocks", "conn4", ("one",),
     lambda c: ff3.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB)),
    ("ch03_conn8", "ch03 · N blocks", "conn8", ("one",),
     lambda c: ff3.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB, connectivity=8)),
    ("ch03_conn8_r2", "ch03 · N blocks", "conn8 r=2", ("one",),
     lambda c: ff3.flood_fill(c["img"], c["sx"], c["sy"],
                              threads_per_block=TPB, connectivity=8,
                              radius=2)),
    ("ch04_seq", "ch04 · 2 blobs", "sequential", ("two",),
     lambda c: ff4.flood_fill(c["img"], c["seeds"], mode="sequential",
                              threads_per_block=TPB)),
    ("ch04_streams", "ch04 · 2 blobs", "streams", ("two",),
     lambda c: ff4.flood_fill(c["img"], c["seeds"], mode="streams",
                              threads_per_block=TPB)),
    ("ch04_multi", "ch04 · 2 blobs", "multisource", ("two",),
     lambda c: ff4.flood_fill(c["img"], c["seeds"], mode="multisource",
                              threads_per_block=TPB)),
    ("ch05_merge", "ch05 · N blobs, no seeds", "seed_merge",
     ("one", "two", "n"),
     lambda c: ff5.flood_fill(c["img"], variant="seed_merge",
                              threads_per_block=TPB)),
    ("ch05_ccl", "ch05 · N blobs, no seeds", "ccl_fill",
     ("one", "two", "n"),
     lambda c: ff5.flood_fill(c["img"], variant="ccl_fill",
                              threads_per_block=TPB)),
    ("ch05_fused_L8", "ch05 · N blobs, no seeds", "fused_L8",
     ("one", "two", "n"),
     lambda c: ff5.flood_fill(c["img"], variant="seed_merge", lattice=8,
                              build="fused", threads_per_block=TPB)),
    ("ch05_r128_L8", "ch05 · N blobs, no seeds", "r128_L8",
     ("one", "two", "n"),
     lambda c: ff5.flood_fill(c["img"], variant="seed_merge", lattice=8,
                              build="r128", threads_per_block=TPB)),
    ("ch05_split_L8", "ch05 · N blobs, no seeds", "split_L8",
     ("one", "two", "n"),
     lambda c: ff5.flood_fill(c["img"], variant="seed_merge", lattice=8,
                              build="split", threads_per_block=TPB)),
    ("ch05_split_I1", "ch05 · N blobs, no seeds", "split_I1 (interior)",
     ("one", "two", "n"),
     lambda c: ff5.flood_fill(c["img"], variant="seed_merge", lattice=1,
                              interior=True, build="split",
                              threads_per_block=TPB)),
]

CPU_COLS = [
    ("pure_python", "CPU", "pure Python BFS", ("one", "two")),
    ("njit", "CPU", "@njit BFS / CCL+fill", ("one", "two", "n")),
]

# Known interaction, reproduced in isolation: ch04's streams mode
# deadlocks when ch03's cooperative kernels have run earlier in the
# SAME process (standalone it completes — each chapter's own benchmark
# never sees this). Until that cross-chapter co-residency puzzle is
# solved, the combined session skips it as "n/s".
STATIC_SKIPS = {"ch04_streams": "unsupported"}


def _stats(vals):
    return {"ms": statistics.median(vals), "ms_min": min(vals),
            "ms_max": max(vals)}


def _cell_gpu(runner, ctx):
    """Warm once for this shape, then median of GPU_REPEATS. Typed skip
    on the known refusals."""
    try:
        r = runner(ctx)
    except RuntimeError as e:
        msg = str(e).lower()
        reason = ("overflow" if any(w in msg for w in
                                    ("overflow", "capacity", "ring"))
                  else f"error:{type(e).__name__}")
        return {"skip": reason}
    except NotImplementedError:
        return {"skip": "unsupported"}
    filled, levels = int(r.filled), int(r.levels)
    del r
    vals = []
    for _ in range(GPU_REPEATS):
        r = runner(ctx)
        vals.append(r.kernel_ms)
        del r
    return {**_stats(vals), "filled": filled, "levels": levels,
            "skip": None}


def _cell_njit(ctx):
    kind = ctx["kind"]
    if kind == "one":
        call = lambda: oracle1.cpu_flood_fill(ctx["img"], ctx["sx"],
                                              ctx["sy"])
    elif kind == "two":
        call = lambda: oracle4.cpu_flood_fill_two(ctx["img"], ctx["seeds"],
                                                  connectivity=4)
    else:
        call = lambda: oracle5.cpu_fill_canonical(ctx["img"])
    call()                                   # compile / cache warm
    vals = []
    for _ in range(NJIT_REPEATS):
        t0 = time.perf_counter()
        call()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return {**_stats(vals), "skip": None}


def _cell_pure(ctx, est_filled):
    if est_filled > PURE_CAP:
        return {"skip": "capped"}
    if ctx["kind"] == "one":
        seeds, conn = [(ctx["sx"], ctx["sy"])], 4
    elif ctx["kind"] == "two":
        seeds, conn = ctx["seeds"], 4
    else:
        seeds, conn = ctx["seeds"], 8      # N-blob components are 8-conn
    repeats = 1 if est_filled > PURE_SINGLE else NJIT_REPEATS
    vals, filled = [], 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        filled = pure_python_bfs(ctx["img"], seeds, connectivity=conn)
        vals.append((time.perf_counter() - t0) * 1000.0)
    return {**_stats(vals), "filled": filled, "skip": None}


def _cell_gpu_loop(runner, ctx):
    """A one-blob kernel doing a multi-blob job, MEASURED: one call per
    blob, kernel times summed per round. This is what really using that
    stage on this scene would cost."""
    seeds = ctx["seeds"]
    subs = [{"kind": "one", "img": ctx["img"], "sx": sx, "sy": sy}
            for sx, sy in seeds]
    filled = 0
    try:
        for sub in subs:                    # probe + shape warmup
            r = runner(sub)
            filled += int(r.filled)
            del r
    except RuntimeError as e:
        msg = str(e).lower()
        return {"skip": ("overflow" if any(w in msg for w in
                                           ("overflow", "capacity", "ring"))
                         else f"error:{type(e).__name__}")}
    except NotImplementedError:
        return {"skip": "unsupported"}
    vals = []
    for _ in range(GPU_REPEATS):
        total = 0.0
        for sub in subs:
            r = runner(sub)
            total += r.kernel_ms
            del r
        vals.append(total)
    return {**_stats(vals), "filled": filled, "calls": len(subs),
            "skip": None}


def _cell_gpu_est(runner, ctx, pair=False):
    """A one/two-blob kernel on an N-blob scene, ESTIMATED: median
    per-call kernel ms over a k-blob sample x the number of calls a
    full loop would need. Marked est; never a row winner."""
    seeds = ctx["seeds"]
    n = len(seeds)
    w, h = ctx["img"].shape[0], ctx["img"].shape[1]
    k = EST_SAMPLE_HUGE if w * h > 20_000_000 else EST_SAMPLE
    if pair:
        n_calls = (n + 1) // 2
        pairs = [(seeds[i], seeds[i + 1]) for i in range(0, n - 1, 2)]
        idx = np.linspace(0, len(pairs) - 1,
                          min(k, len(pairs))).astype(int)
        subs = [{"kind": "two", "img": ctx["img"], "seeds": list(pairs[i])}
                for i in np.unique(idx)]
    else:
        n_calls = n
        idx = np.linspace(0, n - 1, min(k, n)).astype(int)
        subs = [{"kind": "one", "img": ctx["img"],
                 "sx": seeds[i][0], "sy": seeds[i][1]}
                for i in np.unique(idx)]
    vals = []
    try:
        for sub in subs:
            r = runner(sub)
            vals.append(r.kernel_ms)
            del r
    except RuntimeError as e:
        msg = str(e).lower()
        return {"skip": ("overflow" if any(w_ in msg for w_ in
                                           ("overflow", "capacity", "ring"))
                         else f"error:{type(e).__name__}")}
    except NotImplementedError:
        return {"skip": "unsupported"}
    return {"ms": statistics.median(vals) * n_calls, "est": True,
            "sample": len(subs), "calls": n_calls, "skip": None}


def _warmup():
    print("Warming up JITs across all chapters (first run compiles "
          f"~{len(COLS)} kernel variants — expect tens of minutes)...")
    img1, sx, sy = shared_scenes.square_scene(64, 64, 20, 20)
    img2, seeds = ch04_scenes.two_squares_scene(96, 64, 24, 24, gap=8)
    ctx1 = {"kind": "one", "img": img1, "sx": sx, "sy": sy}
    ctx2 = {"kind": "two", "img": img2, "seeds": seeds}
    for key, _, _, kinds, runner in COLS:
        if key in STATIC_SKIPS:
            print(f"  {key}: skipped ({STATIC_SKIPS[key]})")
            continue
        t0 = time.perf_counter()
        ctx = ctx1 if "one" in kinds else ctx2
        try:
            runner(ctx)
            note = "ok"
        except (RuntimeError, NotImplementedError) as e:
            note = type(e).__name__           # cell runs will record it
        print(f"  {key}: {note} ({time.perf_counter() - t0:.0f}s)")


def bench_row(key, family, note, est_filled, build):
    ctx = build()
    img = ctx["img"]
    if ctx["kind"] == "n":
        # one seed per blob, for the CPU bar and the loop/estimate cells
        ctx["seeds"] = _component_seeds(img)
        ctx["n_blobs"] = len(ctx["seeds"])
    cells = {}
    for col_key, _, _, kinds in CPU_COLS:
        if col_key == "pure_python":
            cells[col_key] = _cell_pure(ctx, est_filled)
        else:
            cells[col_key] = _cell_njit(ctx)
    for col_key, _, _, kinds, runner in COLS:
        if col_key in STATIC_SKIPS:
            cells[col_key] = {"skip": STATIC_SKIPS[col_key]}
        elif ctx["kind"] in kinds:
            cells[col_key] = _cell_gpu(runner, ctx)
        elif ctx["kind"] == "two" and kinds == ("one",):
            cells[col_key] = _cell_gpu_loop(runner, ctx)
        elif ctx["kind"] == "n" and kinds == ("one",):
            cells[col_key] = _cell_gpu_est(runner, ctx)
        elif ctx["kind"] == "n" and kinds == ("two",):
            cells[col_key] = _cell_gpu_est(runner, ctx, pair=True)
        else:
            # the one truly impossible family: ch04 needs exactly two
            # blobs in two components — one-blob scenes are outside
            # its input space
            cells[col_key] = {"skip": "na"}

    fills = {c["filled"] for c in cells.values()
             if c.get("skip") is None and "filled" in c}
    crosscheck = "OK" if len(fills) <= 1 else "MISMATCH"
    gpu_ms = {k: c["ms"] for k, c in cells.items()
              if c.get("skip") is None and not c.get("est")
              and k != "pure_python" and k != "njit"}
    best = min(gpu_ms, key=gpu_ms.get) if gpu_ms else None

    row = {"row": key, "family": family, "note": note,
           "width": int(img.shape[0]), "height": int(img.shape[1]),
           "kind": ctx["kind"], "cells": cells, "best": best,
           "crosscheck": crosscheck}
    if ctx["kind"] == "n" and ctx.get("n_blobs") is not None:
        row["n_blobs"] = int(ctx["n_blobs"])

    parts = []
    for col_key in [c[0] for c in CPU_COLS] + [c[0] for c in COLS]:
        c = cells[col_key]
        if c.get("skip") is not None:
            parts.append(f"{col_key}={c['skip']}")
        elif c.get("est"):
            parts.append(f"{col_key}≈{c['ms']:.0f}")
        else:
            parts.append(f"{col_key}={c['ms']:.2f}")
    print(f"\n{key}  ({note})  [{crosscheck}]  best={best}")
    print("  " + "  ".join(parts))

    del ctx, img
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass
    return row


def main():
    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    print(f"Device: {dev_name.strip()} "
          f"({int(device.MULTIPROCESSOR_COUNT)} SMs; tpb={TPB})")
    _warmup()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"overview_{stamp}.json")
    payload = {
        "device": dev_name.strip(),
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "tpb": TPB,
        "gpu_repeats": GPU_REPEATS,
        "columns": ([{"key": k, "group": g, "label": l}
                     for k, g, l, _ in CPU_COLS]
                    + [{"key": k, "group": g, "label": l}
                       for k, g, l, _, _ in COLS]),
        "experiment": (
            "the grand table, full coverage: every chapter's every "
            "variant on one common scene x scale grid, kernel-only "
            "median ms. One-blob kernels on multi-blob scenes run one "
            "call per blob — measured (summed) on two-blob rows, "
            "estimated (k-blob sample x call count, est:true) on "
            "N-blob rows. 'na' survives only for ch04 on one-blob "
            "scenes (its kernel takes exactly two components); "
            "'overflow' = ring capacity; 'capped' = pure Python past "
            "20M red px; 'n/s' = the documented ch04-streams "
            "cross-chapter deadlock"),
        "rows": [],
    }
    for key, family, note, est_filled, build in ROWS:
        payload["rows"].append(bench_row(key, family, note, est_filled,
                                         build))
        with open(json_path, "w") as f:     # crash-safe per row
            json.dump(payload, f, indent=2)

    print(f"\nResults written to {json_path}")


if __name__ == "__main__":
    main()
