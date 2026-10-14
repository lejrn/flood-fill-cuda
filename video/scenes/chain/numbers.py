"""Every number the chain cut (v4, two acts) shows or says, from committed JSON.

Each value names its file and key; files are pinned by name, so a later
benchmark run cannot change the video silently. Sessions:

- OV  overview_20260724T234423Z.json (Jul 24): the CPU baselines, ch01-ch05
      cells and the estimates on the 81 Mpx frame (png_blobs).
- C6  ch06_overview_20260927T013331Z.json (Sep 27): ch06 on the same rows.
- CH01 single_block_shared_20260719T041848Z.json (Jul 19): the shared ring
      (its capacity, its measured peak, what spilled).
- DB2 dual_block_20260719T163033Z.json (Jul 19): two blocks; where the
      scheduler put them, and the pinned same-SM vs spread pair.
- CH03 multi_block_20260720T012243Z.json (Jul 20): act 1 (size), one
      session for 1 block, 2 blocks, 48 blocks, 8 ways and the snake;
      the level sizes, the block placement and the launch-shape sweep.
- R61 runs_20260725T161448Z.json (Jul 25): ch06 on the 81 Mpx frame.
- DR  drones_20261014T023648Z.json (Oct 14): CPU vs GPU on the 240 drone
      frames, one session (the host CPU ran ~3x slower than on Jul 24).
- OWN owner_map_20261014T182637Z.json + .npz (Oct 14): which block painted
      each pixel of the 64 Mpx square (ch02's 2 x 256, ch03's 48 x 256 at 4
      and 8 ways), sampled on the floor's grid (6 x 6 dots per toy cell)
      plus full-resolution crops. Positions are race-dependent; the
      per-block counts are not, and equal CH03's.

Most values are floats; level sizes and SM ids are tuples of ints.

No manim import: `uv run python -m scenes.chain.numbers` prints the table
and exits 1 if a stated fact does not hold.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from scenes.panes import data

R = data.RESULTS
OVD = R / "overview" / "benchmark_results"
OV = OVD / "overview_20260724T234423Z.json"
C6 = OVD / "ch06_overview_20260927T013331Z.json"
DR = OVD / "drones_20261014T023648Z.json"
CH01 = R / "ch01_gpu_1blob_1block" / "benchmark_results" / "single_block_shared_20260719T041848Z.json"
DB2 = R / "ch02_gpu_1blob_2block" / "benchmark_results" / "dual_block_20260719T163033Z.json"
CH03 = R / "ch03_gpu_1blob_nblock" / "benchmark_results" / "multi_block_20260720T012243Z.json"
R61 = R / "ch06_gpu_nblob_runs" / "benchmark_results" / "runs_20260725T161448Z.json"
OWN = R / "ch03_gpu_1blob_nblock" / "owner_map" / "owner_map_20261014T182637Z.json"
OWN_LAUNCHES = {"two": "ch02_global_2x256", "c4": "ch03_multi_48x256_c4", "c8": "ch03_multi_48x256_c8"}
VIDEO = Path(__file__).resolve().parents[2]
DRONE_META = VIDEO / "assets" / "drones_labels" / "meta.json"

# Not in any benchmark JSON: the host CPU's core count, from Intel's spec
# for the model lscpu reports ("13th Gen Intel(R) Core(TM) i9-13900H":
# 6 performance + 8 efficiency cores, 20 threads). WSL exposes 16 of them.
CPU_CORES = 14
CPU_CORES_SOURCE = "Intel spec for the i9-13900H (lscpu model name), not benchmark JSON"

# Not in any benchmark JSON either: two device limits from the device dump
# in .github/copilot-instructions.md (the RTX 4060 Laptop GPU's attributes):
# "Max Shared Memory per Block: 49152 bytes (48.0 KB)", the default static
# limit (the opt-in limit is 99 KB), and "Total Global Memory: 8585216000
# bytes (8.00 GB)". The dump's KB and GB are 1024-based, so are ours: in
# 1000-based units the card has 8.6 GB. ch01 kernels.py:37-38 sizes the
# shared ring under the same 48 KB.
SHARED_BLOCK_BYTES = 49_152
VRAM_BYTES = 8_585_216_000
DEVICE_SOURCE = ".github/copilot-instructions.md device dump, not benchmark JSON"


# r05's grid card: rows are block counts (the last, "max", is each column's
# cooperative cap), columns threads per block; CH03's sweep, 16 Mpx, 4 ways.
# Where a column's cap is already a row (48 x 256, 96 x 128, 192 x 64) its
# "max" cell is None: each measured launch shape is drawn once.
SWEEP_ROWS = (1, 2, 4, 8, 16, 32, 48, 96, 128, 192, "max")
SWEEP_TPB = (32, 64, 128, 256, 512)
# The card's grey bins, by throughput against the best cell (the top bin first).
SWEEP_BINS = (0.97, 0.85, 0.6, 0.3)


def sweep_bin(ms: float, best_ms: float) -> int:
    """0 for the top bin .. len(SWEEP_BINS) for the bottom one."""
    r = best_ms / ms
    return next((i for i, t in enumerate(SWEEP_BINS) if r >= t), len(SWEEP_BINS))


@dataclass(frozen=True)
class Num:
    value: float | tuple
    source: str


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _row(d: dict, name: str) -> dict:
    return next(r for r in d["rows"] if r.get("row", r.get("key")) == name)


def _scene(d: dict, name: str) -> dict:
    return next(s for s in d["scenes"] if s["scene"] == name)


def _place(d: dict, scene: str, config: str) -> dict:
    return next(p for p in d["placement"] if p["scene"] == scene and p["config"] == config)


def busy_blocks(level_size: int, blocks: int, tpb: int) -> int:
    """Blocks with work in a ring of `level_size` entries: entry j goes to
    block (j mod blocks * tpb) // tpb (a grid stride), so blocks 0 .. busy - 1."""
    return min(blocks, math.ceil(level_size / tpb))


def grid_stride_totals(level_sizes, blocks: int, tpb: int) -> list[int]:
    """Entries each block takes over all rings under the same grid stride
    (to compare with the run's measured processed_per_block)."""
    out, span = [0] * blocks, blocks * tpb
    for size in level_sizes:
        full, rem = divmod(size, span)
        for b in range(blocks):
            out[b] += full * tpb + min(tpb, max(0, rem - b * tpb))
    return out


def _two_ring(level_sizes) -> list[int]:
    """The queue's load per ring: the ring being read plus the ring being written."""
    return [a + b for a, b in zip(level_sizes, level_sizes[1:])]


@lru_cache(maxsize=1)
def load() -> dict[str, Num]:
    n: dict[str, Num] = {}
    ov, c6, dr = _load(OV), _load(C6), _load(DR)

    def put(key, value, source):
        n[key] = Num(float(value), source)

    def put_list(key, values, source):
        n[key] = Num(tuple(values), source)

    # the 81 Mpx frame (Jul 24): CPU, estimates, ch05
    png = _row(ov, "png_blobs")
    c = png["cells"]
    put("pure_ms", c["pure_python"]["ms"], f"{OV.name}:png_blobs.pure_python.ms")
    put("njit_ms", c["njit"]["ms"], f"{OV.name}:png_blobs.njit.ms")
    put("n_blobs", png["n_blobs"], f"{OV.name}:png_blobs.n_blobs")
    put("n_pixels", png["width"] * png["height"], f"{OV.name}:png_blobs.width*height")
    for key, col in (("est_ch03_ms", "ch03_conn8"), ("est_ch04_ms", "ch04_multi")):
        assert c[col].get("est") is True, col
        put(key, c[col]["ms"], f"{OV.name}:png_blobs.{col}.ms (est)")
        put(key.replace("_ms", "_calls"), c[col]["calls"], f"{OV.name}:png_blobs.{col}.calls")
    put("ch05_ms", c["ch05_split_L8"]["ms"], f"{OV.name}:png_blobs.ch05_split_L8.ms")
    put("sm", ov["sm_count"], f"{OV.name}:sm_count")
    put("small_tpb", ov["tpb"], f"{OV.name}:tpb")

    # ch06 on the 81 Mpx frame (Jul 25, CUDA events)
    s = _scene(_load(R61), "input_blobs")
    put("ch06_rgb_ms", s["ch06"]["rgb"]["median_ms"], f"{R61.name}:input_blobs.ch06.rgb.median_ms")
    put("ch06_mask_ms", s["ch06"]["mask"]["median_ms"], f"{R61.name}:input_blobs.ch06.mask.median_ms")
    put("red_px", s["red_px"], f"{R61.name}:input_blobs.red_px")
    put("n_runs", s["n_runs"], f"{R61.name}:input_blobs.n_runs")

    # scoreboard row 1: one small blob (sq_256, Jul 24, one session)
    sq = _row(ov, "sq_256")["cells"]
    put("small_cpu_ms", sq["njit"]["ms"], f"{OV.name}:sq_256.njit.ms")
    gpu = {k: v["ms"] for k, v in sq.items()
           if k not in ("njit", "pure_python") and v.get("skip") is None and not v.get("est") and v.get("ms")}
    best = min(gpu, key=gpu.get)
    put("small_gpu_ms", gpu[best], f"{OV.name}:sq_256.{best}.ms (fastest GPU cell)")
    put("small_ch01_ms", sq["ch01_ring"]["ms"], f"{OV.name}:sq_256.ch01_ring.ms")

    # act 1, size: the 64 Mpx square and the snake (Jul 20, one session)
    d3 = _load(CH03)
    q = _scene(d3, "sq_8000_center")
    f = CH03.name + ":sq_8000_center"
    put("big_cpu_ms", q["njit_ms"], f + ".njit_ms")
    put("big_1block_ms", q["v2_kernel_ms"], f + ".v2_kernel_ms")
    put("big_2blocks_ms", q["dual_global_kernel_ms"], f + ".dual_global_kernel_ms")
    put("big_48blocks_ms", q["multi_kernel_ms"], f + ".multi_kernel_ms")
    put("big_8ways_ms", q["conn8_kernel_ms"], f + ".conn8_kernel_ms")
    put("big_blocks", q["multi_blocks"], f + ".multi_blocks")
    put("big_tpb", q["multi_tpb"], f + ".multi_tpb")
    put("big_rings4", q["levels"], f + ".levels")
    put("big_rings8", q["conn8_levels"], f + ".conn8_levels")
    put("big_total_ms", q["multi_total_ms"], f + ".multi_total_ms (with copies)")
    put("big_filled", q["filled"], f + ".filled")
    put_list("big_level_sizes", q["level_sizes"], f + ".level_sizes")
    put_list("big8_level_sizes", q["conn8_level_sizes"], f + ".conn8_level_sizes")
    put_list("big_sm_ids", q["sm_ids"], f + ".sm_ids")
    put_list("big_processed_per_block", q["processed_per_block"], f + ".processed_per_block")
    sn = _scene(d3, "serpentine_256")
    f = CH03.name + ":serpentine_256"
    put("snake_1block_ms", sn["v2_kernel_ms"], f + ".v2_kernel_ms")
    put("snake_2blocks_ms", sn["dual_global_kernel_ms"], f + ".dual_global_kernel_ms")
    put("snake_48blocks_ms", sn["multi_kernel_ms"], f + ".multi_kernel_ms")
    put("snake_rings", sn["levels"], f + ".levels")
    put("snake_util_pct", sn["multi_thread_util_pct"], f + ".multi_thread_util_pct")
    d1 = _load(CH01)
    s1 = _scene(d1, "sq_6000_center")
    f = CH01.name + ":sq_6000_center"
    put("spill_pct", s1["spill_pct"], f + ".spill_pct")

    # act 1, the queue's memory (v4). The shared ring (Jul 19): its slots,
    # and its measured peak on 36 Mpx, which is two adjacent rings.
    put("ring_slots", s1["ring_capacity"], f + ".ring_capacity")
    put("ring_peak_36m", s1["peak_occupancy"], f + ".peak_occupancy")
    put_list("ring_level_sizes_36m", s1["level_sizes"], f + ".level_sizes")
    put("shared_kb", SHARED_BLOCK_BYTES / 1024, DEVICE_SOURCE + " (Max Shared Memory per Block)")
    put("vram_gb", VRAM_BYTES / 2**30, DEVICE_SOURCE + " (Total Global Memory)")

    # r04 "no slower": one block on the 16 Mpx corner square, same session
    # (Jul 20): the shared ring vs the global queue. The ring never spilled
    # there (Jul 19), so the queue's home is the only difference.
    c1 = _scene(d1, "sq_4000_corner")
    f = CH01.name + ":sq_4000_corner"
    put("q1_ring_peak", c1["peak_occupancy"], f + ".peak_occupancy")
    put("q1_spilled_px", c1["spilled_px"], f + ".spilled_px")
    c3 = _scene(d3, "sq_4000_corner")
    put("q1_px", c3["width"] * c3["height"], CH03.name + ":sq_4000_corner.width*height")
    put("q1_shared_ms", c3["v2_kernel_ms"], CH03.name + ":sq_4000_corner.v2_kernel_ms")
    sweep = [r for r in d3["block_tpb_sweep"] if r["scene"] == "sq_4000_corner" and r["connectivity"] == 4]
    timed = [r for r in sweep if r.get("kernel_ms") is not None]
    cell = {(r["blocks"], r["tpb"]): r for r in timed}
    assert len(cell) == len(timed), "two sweep rows for one launch shape"
    f = CH03.name + ":block_tpb_sweep[sq_4000_corner, conn 4"
    put("q1_global_ms", cell[1, q["multi_tpb"]]["kernel_ms"], f + ", 1 x 256].kernel_ms")

    # r05's grid card: the launch-shape sweep on the same square
    ranked = sorted(timed, key=lambda r: r["kernel_ms"])
    best = ranked[0]
    put("sweep_best_ms", best["kernel_ms"], f + f", {best['blocks']} x {best['tpb']}].kernel_ms (the minimum)")
    put("sweep_best_blocks", best["blocks"], f + "].blocks of the minimum")
    put("sweep_best_tpb", best["tpb"], f + "].tpb of the minimum")
    put("sweep_third_ms", ranked[2]["kernel_ms"], f + "].kernel_ms (the third best)")
    put("sweep_default_ms", cell[q["multi_blocks"], q["multi_tpb"]]["kernel_ms"], f + ", 48 x 256].kernel_ms")
    cap = d3["coop_max_by_tpb"]
    put("sweep_cap_ok", all((r.get("kernel_ms") is not None) == (r["blocks"] <= cap[str(r["tpb"])]) for r in sweep),
        f + "]: timed iff within coop_max_by_tpb")
    grid, drawn = [], []
    for b in SWEEP_ROWS:
        for t in SWEEP_TPB:
            shape = (cap[str(t)] if b == "max" else b, t)
            if b == "max" and shape[0] in SWEEP_ROWS:
                grid.append(None)                                 # the cap is a row: drawn there
                continue
            r = cell.get(shape)
            grid.append(r["kernel_ms"] if r is not None else math.nan)
            drawn += [shape] if r is not None else []
    put_list("sweep_grid_ms", grid, f + "].kernel_ms by SWEEP_ROWS x SWEEP_TPB (NaN past the cap, "
             "None where the cap is already a row)")
    put("sweep_grid_once", sorted(drawn) == sorted(cell), f + "]: every timed launch shape is one grid cell")

    # act 1, two blocks (Jul 19): where the scheduler put them, and Q2's
    # pinned pair (ch02's `pinned` kernel, 2 x 768 threads, 40 registers,
    # a hand-rolled pair barrier): the same two blocks on one SM vs on two
    d2 = _load(DB2)
    s2 = _scene(d2, "sq_6000_center")
    put_list("two_sm_ids", s2["global_sm_ids"], f"{DB2.name}:sq_6000_center.global_sm_ids")
    put("two_sm_ids_all", all(s["global_sm_ids"] == s2["global_sm_ids"] for s in d2["scenes"]),
        f"{DB2.name}:scenes[].global_sm_ids (the same on every scene)")
    put("pin_px", s2["width"] * s2["height"], f"{DB2.name}:sq_6000_center.width*height")
    for key, config in (("pin_same", "pinned 2x768 same SM"), ("pin_spread", "pinned 2x768 spread")):
        p = _place(d2, "sq_6000_center", config)
        put(key + "_ms", p["kernel_ms"], f"{DB2.name}:placement[sq_6000_center, {config}].kernel_ms")
        put_list(key + "_sm_ids", p["sm_ids"], f"{DB2.name}:placement[sq_6000_center, {config}].sm_ids")

    # act 2, count: many blobs on one megapixel (row 3; two sessions)
    r1 = _row(ov, "random_1000")
    put("count_blobs", r1["n_blobs"], f"{OV.name}:random_1000.n_blobs")
    put("count_cpu_ms", r1["cells"]["njit"]["ms"], f"{OV.name}:random_1000.njit.ms")
    put("count_gpu_ms", _row(c6, "random_1000")["cells"]["ch06_rgb"]["ms"], f"{C6.name}:random_1000.ch06_rgb.ms")

    # drone frames (Oct 14, one session)
    ds = dr["drone_summary"]
    put("drone_cpu_ms", ds["njit"]["median_ms"], f"{DR.name}:drone_summary.njit.median_ms")
    put("drone_cpu_max_ms", ds["njit"]["max_ms"], f"{DR.name}:drone_summary.njit.max_ms")
    put("drone_gpu_ms", ds["ch06_rgb"]["median_ms"], f"{DR.name}:drone_summary.ch06_rgb.median_ms")
    put("drone_gpu_max_ms", ds["ch06_rgb"]["max_ms"], f"{DR.name}:drone_summary.ch06_rgb.max_ms")
    put("drone_frames", ds["njit"]["n"], f"{DR.name}:drone_summary.njit.n")
    put("drone_ok", all(r["crosscheck"] == "OK" for r in dr["rows"]), f"{DR.name}:rows[].crosscheck")
    meta = _load(DRONE_META)
    put("drone_blobs_median", sorted(meta["blobs_per_frame"])[len(meta["blobs_per_frame"]) // 2],
        "assets/drones_labels/meta.json:blobs_per_frame (median)")
    put("cpu_cores", CPU_CORES, CPU_CORES_SOURCE)

    # act 1, who painted what (Oct 14): the owner-map run, one per launch
    own = _load(OWN)
    for key, launch in OWN_LAUNCHES.items():
        L = own["launches"][launch]
        f = f"{OWN.name}:launches.{launch}"
        put(f"own_{key}_ok", all(v in ("OK", "same") for v in L["checks"].values()) and not L["level_trace_truncated"],
            f + ".checks (all OK) and not .level_trace_truncated")
        put(f"own_{key}_cell_min", L["mixing"]["distinct_owners_per_cell_min"],
            f + ".mixing.distinct_owners_per_cell_min (blocks in a floor cell's 36 dots)")
        put_list(f"own_{key}_block_px", L["block_px"], f + ".block_px")
        put_list(f"own_{key}_sm_ids", L["sm_ids"], f + ".sm_ids")
    return n


@lru_cache(maxsize=1)
def owner_map() -> dict:
    """OWN's arrays per launch ("two", "c4", "c8"): owner_samples and
    depth_samples (126 x 126, index [x, y] = toy [row, col]), the crops
    (crop_seed, crop_ring 64 x 64, crop_seed_wide 192 x 192, block ids)
    and their metadata ("crops", "levels"); "grid" is the floor grid."""
    import numpy as np

    meta = _load(OWN)
    z = np.load(OWN.with_suffix(".npz"))
    out = {"grid": meta["floor_grid"]}
    for key, launch in OWN_LAUNCHES.items():
        L = meta["launches"][launch]
        out[key] = {name: z[f"{launch}.{name}"] for name in
                    ("owner_samples", "depth_samples", "crop_seed", "crop_ring", "crop_seed_wide")}
        out[key]["crops"] = L["crops"]
        out[key]["levels"] = L["levels"]
    return out


def derived() -> dict[str, float]:
    n = {k: v.value for k, v in load().items()}
    levels, blocks, tpb = n["big_level_sizes"], int(n["big_blocks"]), int(n["big_tpb"])
    load2 = _two_ring(levels)
    spill = [L for L, v in enumerate(load2) if v > n["ring_slots"]]   # rings whose two-ring load overflows
    busy = [busy_blocks(s, blocks, tpb) for s in levels]
    every = [L for L, b in enumerate(busy) if b == blocks]
    return {
        "budget_ms": 1000 / 30,
        "small_cpu_x": n["small_gpu_ms"] / n["small_cpu_ms"],          # CPU wins by
        "big_1block_x": n["big_cpu_ms"] / n["big_1block_ms"],          # 2.46
        "big_2v1_x": n["big_1block_ms"] / n["big_2blocks_ms"],         # 2.04
        "big_2blocks_x": n["big_cpu_ms"] / n["big_2blocks_ms"],        # 5.0
        "big_48_x": n["big_cpu_ms"] / n["big_48blocks_ms"],            # 18.75
        "big_8ways_x": n["big_cpu_ms"] / n["big_8ways_ms"],            # 25.8
        "big_total_x": n["big_cpu_ms"] / n["big_total_ms"],            # 3.8 with copies
        "sm_pct_1": 100 / n["sm"],
        "sm_pct_2": 200 / n["sm"],
        "threads": n["big_blocks"] * n["big_tpb"],                     # 12,288
        "snake_stops": 2 * n["snake_rings"],                           # 65,792
        "us_per_stop": 1000 * n["snake_48blocks_ms"] / (2 * n["snake_rings"]),
        "count_x": n["count_cpu_ms"] / n["count_gpu_ms"],              # ~35
        "drone_x": n["drone_cpu_ms"] / n["drone_gpu_ms"],              # ~3.4
        "runs_x": n["red_px"] / n["n_runs"],                           # 24.95
        "frame_x": n["njit_ms"] / n["ch06_rgb_ms"],                    # ~455 (two sessions)
        # v4, the queue's memory (1024-based KB, as the device dump)
        "ring_kb": n["ring_slots"] * 4 / 1024,                         # 32: the gauge's queue region
        "two_ring_36m": max(_two_ring(n["ring_level_sizes_36m"])),     # 23,994 = 11,998 + 11,996
        "two_ring_peak": max(load2),                                   # 31,994 slots
        "two_ring_kb": max(load2) * 4 / 1024,                          # 125 (124.98)
        "first_spill_ring": spill[0],                                  # 1,024: 4,096 + 4,100 slots
        "last_spill_ring": spill[-1],                                  # 6,975
        "spill_rings": len(spill),                                     # 5,952: one band
        "block0_only_rings": next(L for L, b in enumerate(busy) if b > 1),  # 65: rings 0-64
        "busy_all_first": every[0],                                    # 3,009 (first ring > 47 x 256)
        "busy_all_last": every[-1],                                    # 4,991
        "busy_all_rings": len(every),                                  # 1,983 of 8,001
        "pin_x": n["pin_same_ms"] / n["pin_spread_ms"],                # 1.68: one SM vs two
        "q1_x": n["q1_shared_ms"] / n["q1_global_ms"],                 # 1.07: the global queue is not slower
        "sweep_x": n["sweep_default_ms"] / n["sweep_best_ms"],         # 1.155: the best launch shape's gain
    }


def facts() -> list[tuple[str, bool]]:
    """Every spoken sentence that rests on a number."""
    n = {k: v.value for k, v in load().items()}
    d = derived()
    levels, ids, ring = n["big_level_sizes"], n["big_sm_ids"], int(n["ring_slots"])
    blocks, tpb = int(n["big_blocks"]), int(n["big_tpb"])
    run = [ms for ms in n["sweep_grid_ms"] if ms is not None and ms == ms]   # the grid's measured cells
    return [
        ("i0 thirty-three milliseconds", round(d["budget_ms"]) == 33),
        ("i1 on frames this small, compiled CPU code keeps up (drone frames, worst frame)",
         n["drone_cpu_max_ms"] < d["budget_ms"] and n["drone_ok"] == 1),
        ("i1 eighty-one megapixels", round(n["n_pixels"] / 1e6) == 81),
        ("i1 the CPU needs one point three seconds", round(n["njit_ms"] / 1000, 1) == 1.3),
        ("r01 twenty-four SMs", n["sm"] == 24),
        ("r02 two hundred fifty-six threads", n["small_tpb"] == 256),
        ("r02 on one small blob, the CPU still wins", n["small_cpu_ms"] < n["small_gpu_ms"]),
        ("r03 the rings: 8,001 levels, every pixel once",
         len(levels) == n["big_rings4"] and sum(levels) == n["big_filled"]),
        ("r03 the queue holds two rings (the shared ring's measured peak on 36 Mpx = its widest pair)",
         n["ring_peak_36m"] == d["two_ring_36m"]),
        ("r03 forty-eight kilobytes (caption 48 KB, per block, by default)", n["shared_kb"] == 48),
        ("r03 gauge: the queue region is two thirds of the box, the largest power of two that fits",
         3 * d["ring_kb"] == 2 * n["shared_kb"] and ring & (ring - 1) == 0 and 2 * d["ring_kb"] > n["shared_kb"]),
        ("r03 that queue overflows", d["two_ring_peak"] > n["ring_slots"]),
        ("r03 this queue needs over a hundred (caption 125 KB)",
         100 < d["two_ring_kb"] < 128 and round(d["two_ring_kb"]) == 125),
        ("r03 gauge: it spills from ring 1,024 to ring 6,975, in one band",
         d["first_spill_ring"] == 1024 and d["last_spill_ring"] == 6975
         and d["spill_rings"] == d["last_spill_ring"] - d["first_spill_ring"] + 1),
        ("r03 the card's eight gigabytes", round(n["vram_gb"]) == 8),
        ("r03b now one block is over twice the CPU", 2.0 < d["big_1block_x"] < 3.0),
        ("r03b one SM of twenty-four", n["sm"] == 24),
        ("r04 a second SM: block 1 ran on SM 2, in every two-block scene",
         n["two_sm_ids"] == (0, 2) and n["two_sm_ids_all"] == 1),
        ("r04 squeezed onto one SM, two blocks ran one point seven times slower (vs two SMs)",
         len(set(n["pin_same_sm_ids"])) == 1 and len(set(n["pin_spread_sm_ids"])) == 2
         and round(d["pin_x"], 1) == 1.7),
        ("r04 on big blobs, measured, that's no slower (one block, caption 16 Mpx square)",
         n["q1_global_ms"] <= n["q1_shared_ms"] and round(n["q1_px"] / 1e6) == 16),
        ("r04 ... a clean test: the shared ring never spilled there",
         n["q1_ring_peak"] <= n["ring_slots"] and n["q1_spilled_px"] == 0),
        ("r04 twice as fast", 1.9 <= d["big_2v1_x"] < 2.5),
        ("r04 twenty-two SMs stay dark", n["sm"] - 2 == 22),
        ("r05 forty-eight blocks", n["big_blocks"] == 48),
        ("r05 two on each: every SM id 0-23 twice",
         len(ids) == n["big_blocks"] and sorted(ids) == sorted(list(range(24)) * 2)),
        ("r05 the measured order: blocks 0-11 on SMs 0, 2, .., 22, blocks 12-23 on 1, 3, .., 23, then again",
         ids == (tuple(range(0, 24, 2)) + tuple(range(1, 24, 2))) * 2),
        ("r05 eighteen times the CPU", int(d["big_48_x"]) == 18),
        ("r05 busy blocks: the grid-stride rule gives every block's measured total",
         grid_stride_totals(levels, blocks, tpb) == list(n["big_processed_per_block"])),
        ("r05 small rings keep only a few busy: rings 0-64 are block 0's alone", d["block0_only_rings"] == 65),
        ("r05 all 48 busy only on rings 3,009-4,991 (1,983 of 8,001), one band",
         (d["busy_all_first"], d["busy_all_last"], d["busy_all_rings"]) == (3009, 4991, 1983)
         and d["busy_all_rings"] == d["busy_all_last"] - d["busy_all_first"] + 1),
        ("r05 we also tried: the best launch shape is not the default",
         (n["sweep_best_blocks"], n["sweep_best_tpb"]) != (n["big_blocks"], n["big_tpb"])),
        ("r05 on a smaller square, the best gained fifteen percent (caption +15% at best, 16 Mpx)",
         round(100 * (d["sweep_x"] - 1)) == 15 and n["q1_px"] < n["big_filled"]),
        ("r05 grid card: a flat top (the three best cells within 2%)", n["sweep_third_ms"] / n["sweep_best_ms"] < 1.02),
        ("r05 grid card: the empty cells are exactly those past the cooperative cap", n["sweep_cap_ok"] == 1),
        ("r05 grid card: every measured launch shape is drawn, once", n["sweep_grid_once"] == 1),
        ("r05 grid card: the top bin holds exactly the three best cells (a flat top, never one glowing cell)",
         sum(sweep_bin(ms, n["sweep_best_ms"]) == 0 for ms in run) == 3 and min(run) == n["sweep_best_ms"]),
        ("r05 grid card: the default 48 x 256 is a cell of the grid",
         n["sweep_grid_ms"][SWEEP_ROWS.index(48) * len(SWEEP_TPB) + SWEEP_TPB.index(256)] == n["sweep_default_ms"]),
        ("r05 we kept forty-eight by two hundred fifty-six", n["big_blocks"] == 48 and n["big_tpb"] == 256),
        ("D4 owner maps: every check of the run passed, the level traces whole",
         all(n[f"own_{k}_ok"] == 1 for k in OWN_LAUNCHES)),
        ("D4 owner maps: the 48-block counts are the pinned session's (processed_per_block), on its SMs",
         n["own_c4_block_px"] == n["big_processed_per_block"] and n["own_c4_sm_ids"] == ids
         and n["own_c8_sm_ids"] == ids),
        ("D4 owner maps: the 2-block run is DB2's placement, SMs 0 and 2", n["own_two_sm_ids"] == n["two_sm_ids"]),
        ("r04 the 2-block floor: blue and green mixed in every cell, never two halves", n["own_two_cell_min"] == 2),
        ("r05 their pixels mix: every floor cell holds at least four blocks",
         n["own_c4_cell_min"] >= 4 and n["own_c8_cell_min"] >= 4),
        ("r05b on a thin snake, forty-eight blocks lose to one", n["snake_48blocks_ms"] > n["snake_1block_ms"]),
        ("r05b stops twice, two microseconds each", 1.5 <= d["us_per_stop"] < 2.5),
        ("r06 the 8-way rings: 4,001 levels, every pixel once",
         len(n["big8_level_sizes"]) == n["big_rings8"] and sum(n["big8_level_sizes"]) == n["big_filled"]),
        ("r06 on a square, half the rings", abs(n["big_rings8"] / n["big_rings4"] - 0.5) < 0.01),
        ("r06 caption 25.8x", round(d["big_8ways_x"], 1) == 25.8),
        ("r07 twenty-five hundred blobs", 2400 <= n["n_blobs"] <= 2600),
        ("r07 one launch each", n["est_ch03_calls"] == n["n_blobs"]),
        ("r07 slower than the CPU (estimate)", n["est_ch03_ms"] > n["njit_ms"]),
        ("r08 two per launch", n["est_ch04_calls"] * 2 == n["n_blobs"]),
        ("r08 about CPU speed (within 5%)", abs(n["est_ch04_ms"] / n["njit_ms"] - 1) < 0.05),
        ("r09 fifty-one milliseconds", int(n["ch05_ms"]) == 51),
        ("r09 still over one frame", n["ch05_ms"] > d["budget_ms"]),
        ("r10 about three milliseconds (RGB)", 2.5 <= n["ch06_rgb_ms"] < 3.5),
        ("r10 twenty-five times fewer (caption 25x fewer)", round(d["runs_x"]) == 25),
        ("r11 many at once: the GPU (count row)", d["count_x"] > 1),
        ("r11 every drone, every frame (GPU worst frame under budget)", n["drone_gpu_max_ms"] < d["budget_ms"]),
    ]


def _fmt(v) -> str:
    if isinstance(v, tuple):
        return f"{len(v):,} items: " + ", ".join(f"{x:,}" for x in v[:3]) + (", ..." if len(v) > 3 else "")
    return f"{v:>14,.3f}"


def main() -> int:
    for k, v in load().items():
        print(f"{k:24s} {_fmt(v.value)}   {v.source}")
    print()
    for k, v in derived().items():
        print(f"{k:24s} {_fmt(v)}")
    print()
    ok = True
    for text, good in facts():
        ok &= bool(good)
        print(("ok    " if good else "FALSE ") + text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
