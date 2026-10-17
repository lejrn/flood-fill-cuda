"""Every number the chain cut (v5, three acts) shows or says, from committed JSON.

Each value names its file and key; files are pinned by name, so a later
benchmark run cannot change the video silently. Sessions:

- V5  v5_20261017T192640Z.json (Oct 17, night): every CPU-vs-GPU number
      of v5, in one session, like for like: each GPU cell against its CPU
      twin, which takes the same steps, keeps no output its GPU side does
      not, and resets in the same scope, on buffers allocated once (the
      session's rules.cpu_twin, cpu_meta, gpu_outputs). The twins: the
      4-neighbour BFS with depth (cpu_bfs4: one block, two blocks, 48
      blocks), the 8-neighbour BFS with depth (cpu_bfs8: the diagonals),
      the 8-neighbour BFS without depth, reset on the clock (cpu_fill8: the
      tiles), and labels only, reset on the clock (cpu_label: the waves,
      the drones) or before it (cpu_label_pre: act 2). Every twin equals
      its oracle array for array. The oracles as called (fresh arrays on
      every call) and cpu_fill_canonical (v4's frame baseline) are
      reference only. Scenes: the small blob (sq_256, row 1), the huge
      blob (sq_8000_center: act 1 and the tiles, row 2), the 81 Mpx frame
      (input_blobs: act 2's two estimates, re-made here with v4's method
      at 8 neighbours, the seeds and the waves; the meter and row 3) and
      the 240 drone frames. Every ch08 number is ch08's first version
      (OLD_PATH: no dense-tile path); the default engine is timed beside
      it on the frame, reference only. GPU: kernel-only CUDA events; every
      timed block at full clocks (P0 only, memory 8,001 MHz, SM median at
      least 2,040; ch05 on the frame on its second attempt). CPU: @njit on
      one core, after all GPU work and 60 s of idle, 5 interleaved rounds
      behind a quiet-host gate. The evening session
      (v5_20261017T182449Z, the oracles as called, so the CPU paid its
      allocations), the morning one and July's overview session are no
      longer read.
- CH01 single_block_shared_20260719T041848Z.json (Jul 19): the shared ring
      (its capacity, its measured peak, what spilled).
- DB2 dual_block_20260719T163033Z.json (Jul 19): two blocks; where the
      scheduler put them, and the pinned same-SM vs spread pair.
- CH03 multi_block_20260720T012243Z.json (Jul 20): act 1's mechanism
      (level sizes, block placement, the launch-shape sweep) and the snake.
      Only GPU-to-GPU comparisons come from here in v5.
- OWN owner_map_20261014T182637Z.json + .npz (Oct 14): which block painted
      each pixel of the 64 Mpx square (ch02's 2 x 256, ch03's 48 x 256 at 4
      and 8 ways), sampled on the floor's grid (6 x 6 dots per toy cell)
      plus full-resolution crops. Positions are race-dependent; the
      per-block counts are not, and equal CH03's.
- T7  tiles_20261016T064724Z.json (Oct 16): the tile BFS's own session
      (rounds, pixel levels, the default switches) on the 64 Mpx square.
- D8  dense_20261016T054922Z.json (Oct 16): the tile waves' final session
      on the 81 Mpx frame (seeds, pieces, rounds, blobs, the dense-tile
      counters), the default engine and the first version (old_path, the
      engine V5 times for the story): their counters are equal.
- W8  waves_20261015T183401Z.json (Oct 15): the tile waves' first-version
      session; its deterministic counters must equal D8's.
- TL  assets/tiles/ch07_tiles_20261017T160022Z.json + .npz (Oct 17): act
      3's tile floors, recorded from the tile BFS kernel (the 64 Mpx
      square's rounds, and one warp's close-up of one tile).
- WV  assets/tiles/ch08_waves_20261017T160051Z.json + .npz (Oct 17): act
      3's wave floors, recorded from the tile waves kernel (a 384 px crop
      of the frame, run as its own image, and the whole frame's facts).
- PIC assets/ch05_input_blobs/frame_000.png (local, unpacked from the
      parent repo's GIF): the 900 px frame the floors draw; only the
      orientation fact reads it.

v5 drops v4's r10 numbers and their sessions (C6, R61, DR). The drone
card, the scoreboard rows, act 2's estimates and the frame's red-pixel
count come from V5. `floors()` computes the act-3 floor facts from TL's
and WV's arrays; `card_ms` / `card_x` are the on-screen number formats the
facts check (`card_ms_note` for the r14 footnote's unspoken times, rounded
where `card_ms` truncates). `STORY_SHOWN` ties every shown CPU-vs-GPU ratio to V5's own
story entry, so a ratio built from the wrong CPU twin fails.

Most values are floats; level sizes, SM ids and per-round counts are
tuples of ints.

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
V5 = OVD / "v5_20261017T192640Z.json"
CH01 = R / "ch01_gpu_1blob_1block" / "benchmark_results" / "single_block_shared_20260719T041848Z.json"
DB2 = R / "ch02_gpu_1blob_2block" / "benchmark_results" / "dual_block_20260719T163033Z.json"
CH03 = R / "ch03_gpu_1blob_nblock" / "benchmark_results" / "multi_block_20260720T012243Z.json"
OWN = R / "ch03_gpu_1blob_nblock" / "owner_map" / "owner_map_20261014T182637Z.json"
OWN_LAUNCHES = {"two": "ch02_global_2x256", "c4": "ch03_multi_48x256_c4", "c8": "ch03_multi_48x256_c8"}
T7 = R / "ch07_gpu_1blob_tiles" / "benchmark_results" / "tiles_20261016T064724Z.json"
D8 = R / "ch08_gpu_nblob_tiles" / "benchmark_results" / "dense_20261016T054922Z.json"
W8 = R / "ch08_gpu_nblob_tiles" / "benchmark_results" / "waves_20261015T183401Z.json"
VIDEO = Path(__file__).resolve().parents[2]
TILES = VIDEO / "assets" / "tiles"
TL = TILES / "ch07_tiles_20261017T160022Z.json"
WV = TILES / "ch08_waves_20261017T160051Z.json"
DRONE_META = VIDEO / "assets" / "drones_labels" / "meta.json"
PICTURE = VIDEO / "assets" / "ch05_input_blobs" / "frame_000.png"

# Every CPU-vs-GPU ratio the video shows or says, and V5's story entry it
# must equal: (derived key, story key, True when the derived key is
# GPU / CPU, i.e. the CPU wins and the story's x is inverted).
STORY_SHOWN = (
    ("small_cpu_x", "small_pixel_bfs", True),             # r02: CPU 4.3×, row 1
    ("big_1block_x", "big_1block", False),                # r03b: 1.1× ("a little faster")
    ("big_2blocks_x", "big_2blocks", False),              # r04 (not shown; twice as fast is GPU-GPU)
    ("big_48_x", "big_48blocks", False),                  # r05: eight and a half times
    ("big_8ways_x", "big_8ways", False),                  # r06: 11×, row 2
    ("est_ch03_cpu_x", "frame_one_launch_per_blob", True),    # r07: slower than the CPU
    ("est_ch04_cpu_x", "frame_two_per_launch", True),     # r08: still slower than the CPU
    ("count_ch05_x", "frame_seeds", False),               # r09: faster than the CPU, 3.8×, row 3
    ("big_tiles_x", "big_tiles", False),                  # r11: 340×, row 2
    ("count_x", "frame_waves", False),                    # r13: 100×, row 3
    ("small_tiles_x", "small_tiles", False),              # r14: 4.4×, row 1
)

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
WARP_LANES = 32                     # CUDA's warp size: a constant of the hardware, not a measurement


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


def card_ms(ms: float) -> str:
    """An on-screen time: overlays.fmt_ms (1,065 ms / 53.9 ms / 3.12 ms),
    plus three decimals under 1 ms (0.192 ms, 0.044 ms): with two, the
    r14 card's 0.19 / 0.044 would read 4.3x against a true 4.4x."""
    if ms >= 100:
        return f"{ms:,.0f} ms"
    if ms >= 10:
        return f"{int(ms * 10) / 10:.1f} ms"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ms:.3f} ms"


def card_ms_note(ms: float) -> str:
    """An on-screen time nobody speaks (the r14 card's footnote): as
    card_ms, but rounded where card_ms truncates (10-100 ms), so a cost is
    never shown below its measured value."""
    if 10 <= ms < 100:
        return f"{ms:.1f} ms"
    return card_ms(ms)


def card_x(x: float) -> str:
    """An on-screen CPU-vs-GPU factor, two significant figures (4.3×, 14×,
    450×): the CPU's round medians spread by up to about 8%."""
    if x < 10:
        return f"{x:.1f}×"
    if x < 100:
        return f"{x:.0f}×"
    return f"{round(x, -1):.0f}×"


def _shown(s: str) -> float:
    return float(s.split()[0].replace(",", ""))


def card_row_ok(slow_ms: float, fast_ms: float) -> bool:
    """A card row reads true: its two shown times give its shown factor."""
    return card_x(_shown(card_ms(slow_ms)) / _shown(card_ms(fast_ms))) == card_x(slow_ms / fast_ms)


@dataclass(frozen=True)
class Num:
    value: float | tuple
    source: str


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _scene(d: dict, name: str) -> dict:
    return next(s for s in d["scenes"] if s["scene"] == name)


def _place(d: dict, scene: str, config: str) -> dict:
    return next(p for p in d["placement"] if p["scene"] == scene and p["config"] == config)


def _cells_ok(s: dict) -> bool:
    """A V5 scene: every GPU cell's image check and every CPU cell's check passed."""
    gpu = all(c["check"]["ok"] for c in s["cells"].values() if c["side"] == "gpu")
    cpu = all(c["check_ok"] for c in s["cells"].values() if c["side"] == "cpu")
    return bool(s["all_checks_ok"] and gpu and cpu)


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
    v5 = _load(V5)

    def put(key, value, source):
        n[key] = Num(float(value), source)

    def put_list(key, values, source):
        n[key] = Num(tuple(values), source)

    # ================= V5 (Oct 17): every CPU-vs-GPU number, like for like =================
    f5 = V5.name
    put("v5_ok", v5["all_checks_ok"], f"{f5}:all_checks_ok")
    put("v5_script_at_head", v5["script_matches_head"], f"{f5}:script_matches_head")
    put("v5_all_full_clocks", v5["all_blocks_full_clocks"], f"{f5}:all_blocks_full_clocks")
    # every timed block: each scene's blocks (interleaved rounds, ch05, the
    # estimates, the loops) and the drones
    win = {f"{s['scene']}.{b}": blk for s in v5["scenes"] for b, blk in s["blocks"].items()}
    win["drones"] = v5["drones"]
    rule = v5["rules"]["full_clocks"]
    put("v5_blocks", len(win), f"{f5}:scenes[].blocks.*, drones (timed blocks)")
    put("v5_sm_median_rule", rule["sm_median_min_mhz"], f"{f5}:rules.full_clocks.sm_median_min_mhz")
    put("v5_sm_clock_median", min(w["clk_sm_median"] for w in win.values()),
        f"{f5}:scenes[].blocks.*.clk_sm_median, drones.clk_sm_median (the lowest)")
    put("v5_mem_clock_min", min(w["clk_mem_min"] for w in win.values()),
        f"{f5}:scenes[].blocks.*.clk_mem_min, drones.clk_mem_min (the lowest)")
    put("v5_p0_only", all(w["clk_pstates"] == ["P0"] and w["clk_p0_share"] == 1.0 for w in win.values()),
        f"{f5}:every block's clk_pstates == [P0], clk_p0_share == 1")
    put("v5_full_clocks_used", all(w["full_clocks"] and w["attempts"][w["used_attempt"]]["full_clocks"]
                                   and w["used_attempt"] == len(w["attempts"]) - 1 for w in win.values()),
        f"{f5}:every block's used attempt at full clocks, and the last one run")
    put("v5_first_attempt_blocks", sum(w["used_attempt"] == 0 for w in win.values()),
        f"{f5}:blocks at full clocks on their first attempt")
    retried = [k for k, w in win.items() if w["used_attempt"] > 0]
    put("v5_retried_is_ch05", retried == ["input_blobs.ch05"]
        and win["input_blobs.ch05"]["attempts"][0]["clk_pstates"] == ["P3"]
        and "sw_power_cap" in win["input_blobs.ch05"]["attempts"][0]["clk_event_reasons"],
        f"{f5}:input_blobs.blocks.ch05.attempts (the only block retried; attempt 0 in P3, sw_power_cap)")
    put("v5_power_capped_blocks", sum("sw_power_cap" in w["clk_event_reasons"] for w in win.values()),
        f"{f5}:blocks whose used attempt logged sw_power_cap")
    put("sm", v5["sm_count"], f"{f5}:sm_count")

    # like for like: every GPU cell's CPU twin is the session's rule; the
    # twin keeps no output its GPU side does not (outputs), takes the same
    # steps (4 or 8 neighbours; labels are 8-neighbour), resets in the same
    # scope (before the clock or on it), equals its oracle array for array,
    # and its twin.x is the two medians' ratio
    twin_rule = v5["rules"]["cpu_twin"]
    ok = True
    for s in v5["scenes"]:
        for name, cell in s["cells"].items():
            if cell["side"] != "gpu":
                continue
            t = cell["twin"]
            cpu = s["cells"][t["cpu"]]
            ok &= (t["cpu"] == twin_rule[name] and set(cpu["outputs"]) <= set(cell["outputs"])
                   and cpu["connectivity"] == cell["config"]["connectivity"] and cpu["reset"] == cell["reset"]
                   and cpu["check_ok"] and cpu["same_as_oracle"] and t["like_for_like"]
                   and t["cpu_ms"] == cpu["ms"] and math.isclose(t["x"], t["cpu_ms"] / cell["ms"]))
    dr = v5["drones"]
    dm = dr["cpu_twin_meta"]
    ok &= (twin_rule[dr["gpu_cell"]] == dr["twin"] == "cpu_label" and set(dm["outputs"]) <= set(dr["outputs"])
           and dm["connectivity"] == dr["config"]["connectivity"] and dm["reset"] == dr["reset"]
           and dr["like_for_like"] and all(fm["cpu_label"]["same_as_oracle"] for fm in dr["frames"]))
    put("v5_twins_ok", ok, f"{f5}:scenes[].cells[gpu].twin == rules.cpu_twin; the twin's outputs within the GPU "
        "cell's, same connectivity, same reset; the twin equals its oracle; x = cpu_ms / ms; drones alike")
    put("v5_all_like_for_like", v5["all_like_for_like"], f"{f5}:all_like_for_like")
    story = {e["key"]: e for e in v5["story"]}
    for k, e in story.items():
        if "x" in e:
            put(f"story_{k}_x", e["x"], f"{f5}:story[{k}].x ({e['gpu']} vs {e['cpu']})")
    put("story_full_clocks", all(e.get("gpu_block_full_clocks", True) for e in story.values()),
        f"{f5}:story[].gpu_block_full_clocks")

    # the 81 Mpx frame: the CPU (labels only: reset on the clock, the twin of
    # the waves; reset before it, the twin of act 2; the oracle as called and
    # v4's labels-plus-fill baseline for reference), act 2's estimates (v4's
    # method, 8 neighbours), the seeds (ch05 split_L8) and the waves (ch08's
    # first version; the default engine for reference)
    fr = _scene(v5, "input_blobs")
    c, f = fr["cells"], f"{f5}:input_blobs"
    put("njit_ms", c["cpu_label"]["ms"],
        f + ".cpu_label.ms (labels only, reset on the clock: i1's stopwatch, the meter's CPU tick, row 3 from r13)")
    put_list("njit_round_ms", c["cpu_label"]["round_medians"], f + ".cpu_label.round_medians")
    put("count_cpu_pre_ms", c["cpu_label_pre"]["ms"],
        f + ".cpu_label_pre.ms (labels only, reset before the clock: the twin of r07, r08, r09)")
    put_list("count_cpu_pre_round_ms", c["cpu_label_pre"]["round_medians"], f + ".cpu_label_pre.round_medians")
    put("frame_oracle_ms", c["oracle_label"]["ms"],
        f + ".oracle_label.ms (cpu_label_components as called, fresh arrays: reference only)")
    put("frame_cpu_v4_ms", c["cpu_fill_canonical"]["ms"],
        f + ".cpu_fill_canonical.ms (v4's baseline, labels + a fill per blob: reference only)")
    put("n_pixels", fr["width"] * fr["height"], f + ".width*height")
    put("red_px", fr["red_px"], f + ".red_px")
    put("n_blobs", fr["n_blobs"], f + ".n_blobs")
    put("n_blobs_agree", fr["n_blobs"] == c["cpu_label"]["n_blobs"] == c["cpu_label_pre"]["n_blobs"]
        == c["ch08_old"]["info"]["n_blobs"] == c["ch08"]["info"]["n_blobs"],
        f + ".n_blobs == cpu_label(_pre).n_blobs == ch08_old.info.n_blobs == ch08.info.n_blobs")
    put("ch05_ms", c["ch05_split_L8"]["ms"], f + ".ch05_split_L8.ms (its own block)")
    put_list("ch05_x_range", c["ch05_split_L8"]["twin"]["x_range_cpu_rounds"],
             f + ".ch05_split_L8.twin.x_range_cpu_rounds")
    put("frame_ms", c["ch08_old"]["ms"], f + ".ch08_old.ms (kernel only, ch08's first version)")
    put_list("frame_x_range", c["ch08_old"]["twin"]["x_range_cpu_rounds"], f + ".ch08_old.twin.x_range_cpu_rounds")
    put("frame_default_ms", c["ch08"]["ms"], f + ".ch08.ms (the default engine, the dense-tile path: reference only)")
    put("frame_default_over_old_paired", fr["paired"]["ch08/ch08_old"]["median"],
        f + ".paired[ch08/ch08_old].median")
    put("frame_rounds", c["ch08_old"]["info"]["rounds"], f + ".ch08_old.info.rounds")
    put("frame_e2e_ms", c["ch08_old"]["e2e"]["ms"], f + ".ch08_old.e2e.ms (with the copies, context only)")
    put("frame_ch05_over_ch08", fr["ratio_of_medians"]["ch05_split_L8/ch08_old"],
        f + ".ratio_of_medians[ch05_split_L8/ch08_old]")
    put("frame_labels_ok", c["ch08_old"]["check"]["labels_match_cpu"] and c["ch08"]["check"]["labels_match_cpu"],
        f + ".ch08_old.check.labels_match_cpu, ch08.check.labels_match_cpu")
    put("frame_dense_tiles", sum(c[k]["info"]["dense_label_tiles"] + c[k]["info"]["dense_paint_tiles"]
                                 for k in ("ch08_old", "ch08")),
        f + ".ch08_old/ch08.info.dense_label_tiles + dense_paint_tiles")
    put("frame_ok", _cells_ok(fr), f + ": every cell's check")
    cfg8 = c["ch08_old"]["config"]
    put("frame_waves_blocks", cfg8["waves_blocks"], f + ".ch08_old.config.waves_blocks")
    put("frame_waves_tpb", cfg8["waves_tpb"], f + ".ch08_old.config.waves_tpb")
    old = v5["rules"]["ch08_old_path"]
    put("frame_first_version", old == {"dense": False, "lookup": "hybrid", "fit_grids": False}
        and all(cfg8[k] == v for k, v in old.items()) and c["ch08"]["reference_only"]
        and not c["ch08_old"]["reference_only"],
        f + ".ch08_old.config.{dense, lookup, fit_grids} == rules.ch08_old_path (OLD_PATH); ch08 reference only")
    for key, col in (("est_ch03", "est_ch03_conn8"), ("est_ch04", "est_ch04_multi8")):
        e = c[col]
        assert e["est"] is True, col
        put(key + "_ms", e["ms"], f + f".{col}.ms (estimate: median sampled call x calls, v4's method)")
        put(key + "_calls", e["calls"], f + f".{col}.calls")
        put(key + "_mean_ms", e["est_mean_based_ms"], f + f".{col}.est_mean_based_ms")
        put(key + "_host_ms", e["host_clock_est_ms"], f + f".{col}.host_clock_est_ms (v4's clock, same calls)")
        put(key + "_conn", e["config"]["connectivity"], f + f".{col}.config.connectivity")
    put("loop_ch03_ms", c["loop_ch03_conn8"]["ms"], f + ".loop_ch03_conn8.ms (the full loop, reference)")
    put("loop_ch04_ms", c["loop_ch04_multi8"]["ms"], f + ".loop_ch04_multi8.ms (the full loop, reference)")
    put("est_ch03_over_ch04_paired", fr["paired"]["est_ch03_conn8/est_ch04_multi8"]["median"],
        f + ".paired[est_ch03_conn8/est_ch04_multi8].median")

    # row 3, many blobs: the same frame (the meter's). Until r13 its CPU is
    # the labels reset before the clock (count_cpu_pre_ms, against ch05);
    # from r13 the labels reset on it (count_cpu_ms, against the waves).
    # The keys keep v4's names, so score_rows() keeps working.
    for key, same in (("count_cpu_ms", "njit_ms"), ("count_ch05_ms", "ch05_ms"), ("count_gpu_ms", "frame_ms"),
                      ("count_px", "n_pixels"), ("count_blobs", "n_blobs")):
        n[key] = n[same]

    # row 1, one small blob: r02's one block against the 4-neighbour CPU,
    # r14's tiles against the 8-neighbour CPU without depth (ch07 keeps none)
    sq = _scene(v5, "sq_256")
    c, f = sq["cells"], f"{f5}:sq_256"
    put("small_px", sq["red_px"], f + ".red_px")
    put("small_cpu_ms", c["cpu_bfs4"]["ms"], f + ".cpu_bfs4.ms (4 neighbours, depth, reset before: r02's twin)")
    put("small_cpu8_ms", c["cpu_fill8"]["ms"],
        f + ".cpu_fill8.ms (8 neighbours, no depth, reset on the clock: r14's twin)")
    put("small_cpu8_depth_ms", c["cpu_bfs8"]["ms"], f + ".cpu_bfs8.ms (8 neighbours, depth: ch05_fused_L8's twin)")
    put("small_oracle4_ms", c["oracle_bfs4"]["ms"], f + ".oracle_bfs4.ms (as called: reference)")
    g = c["ch01_ring"]
    put("small_gpu_ms", g["ms"], f + ".ch01_ring.ms (r02's one block, its queue a ring in shared memory)")
    put_list("small_gpu_x_range", g["twin"]["x_range_cpu_rounds"], f + ".ch01_ring.twin.x_range_cpu_rounds")
    put("small_tpb", g["config"]["tpb"], f + ".ch01_ring.config.tpb")
    put("small_gpu_blocks", g["config"]["blocks"], f + ".ch01_ring.config.blocks")
    put("small_ch05_ms", c["ch05_fused_L8"]["ms"], f + ".ch05_fused_L8.ms (reference only: act 2's seeds, not shown here)")
    put_list("small_ch05_x_range", c["ch05_fused_L8"]["twin"]["x_range_cpu_rounds"],
             f + ".ch05_fused_L8.twin.x_range_cpu_rounds")
    put("small_tiles_ms", c["ch07"]["ms"], f + ".ch07.ms (kernel only)")
    put_list("small_tiles_x_range", c["ch07"]["twin"]["x_range_cpu_rounds"], f + ".ch07.twin.x_range_cpu_rounds")
    put("small_tiles_e2e_ms", c["ch07"]["e2e"]["ms"], f + ".ch07.e2e.ms (with the copies)")
    put("small_tiles_rounds", c["ch07"]["info"]["rounds"], f + ".ch07.info.rounds")
    put("small_tiles_blocks", c["ch07"]["config"]["blocks"], f + ".ch07.config.blocks")
    put("small_ok", _cells_ok(sq), f + ": every cell's check")

    # row 2, one huge blob: act 1's cells (1 block, 2 blocks, 48 blocks
    # against the 4-neighbour CPU; the diagonals against the 8-neighbour
    # CPU with depth, as ch03 keeps it) and act 3's tiles (the 8-neighbour
    # CPU without depth, reset on the clock), one session
    q5 = _scene(v5, "sq_8000_center")
    c, f = q5["cells"], f"{f5}:sq_8000_center"
    put("big_cpu4_ms", c["cpu_bfs4"]["ms"], f + ".cpu_bfs4.ms (4 neighbours: the twin of r03b, r04, r05)")
    put("big_cpu8_ms", c["cpu_bfs8"]["ms"], f + ".cpu_bfs8.ms (8 neighbours, depth: the twin of r06)")
    put("big_cpu_fill_ms", c["cpu_fill8"]["ms"],
        f + ".cpu_fill8.ms (8 neighbours, no depth, reset on the clock: the twin of r11, the r14 card's row 2)")
    put("big_oracle4_ms", c["oracle_bfs4"]["ms"], f + ".oracle_bfs4.ms (as called, fresh arrays: reference)")
    put("big_oracle8_ms", c["oracle_bfs8"]["ms"], f + ".oracle_bfs8.ms (as called, fresh arrays: reference)")
    n["big_cpu_ms"] = n["big_cpu8_ms"]     # row 2's CPU until r11 (score_rows(beat) moves to big_cpu_fill_ms)
    for key, cfg in (("big_1block_ms", "ch01_spill"), ("big_2blocks_ms", "ch02_global"),
                     ("big_48blocks_ms", "ch03_conn4"), ("big_8ways_ms", "ch03_conn8"),
                     ("big_tuned_ms", "ch03_wc_bare"), ("big_tiles_ms", "ch07")):
        put(key, c[cfg]["ms"], f + f".{cfg}.ms")
    for key, cfg in (("big_1block_x_range", "ch01_spill"), ("big_48_x_range", "ch03_conn4"),
                     ("big_8ways_x_range", "ch03_conn8"), ("big_tiles_x_range", "ch07")):
        put_list(key, c[cfg]["twin"]["x_range_cpu_rounds"], f + f".{cfg}.twin.x_range_cpu_rounds")
    put("big_tiles_e2e_ms", c["ch07"]["e2e"]["ms"], f + ".ch07.e2e.ms (with the copies)")
    put("big_tiles_rounds", c["ch07"]["info"]["rounds"], f + ".ch07.info.rounds")
    put("big_levels4_v5", c["ch03_conn4"]["info"]["levels"], f + ".ch03_conn4.info.levels")
    put("big_levels8_v5", c["ch03_conn8"]["info"]["levels"], f + ".ch03_conn8.info.levels")
    put("big_tiles_one_sync", c["ch07"]["config"]["one_sync"], f + ".ch07.config.one_sync")
    put("big_tiles_blocks", c["ch07"]["config"]["blocks"], f + ".ch07.config.blocks")
    put("big_tiles_tpb", c["ch07"]["config"]["tpb"], f + ".ch07.config.tpb")
    for ph in ("pack", "init", "rounds", "paint"):
        put(f"big_tiles_{ph}_ms", c["ch07"]["phase_ms"][ph], f + f".ch07.phase_ms.{ph}")
    put_list("big_seed", q5["seed"], f + ".seed")
    p = q5["paired"]
    put("big_2v1_paired", p["ch01_spill/ch02_global"]["median"], f + ".paired[ch01_spill/ch02_global].median")
    put("big_8_over_tiles_paired", p["ch03_conn8/ch07"]["median"], f + ".paired[ch03_conn8/ch07].median")
    put("big_tuned_over_tiles_paired", p["ch03_wc_bare/ch07"]["median"], f + ".paired[ch03_wc_bare/ch07].median")
    put("big_ok", _cells_ok(q5), f + ": every cell's check")

    # the 240 drone frames: ch08's first version (no dense-tile path),
    # against the labels-only CPU (reset on the clock, as ch08 initialises)
    s, f = dr["summary"], f"{f5}:drones.summary"
    put("drone_cpu_ms", s["cpu_label"]["median_ms"], f + ".cpu_label.median_ms (labels only)")
    put("drone_cpu_max_ms", s["cpu_label"]["max_ms"], f + ".cpu_label.max_ms")
    put("drone_gpu_ms", s["gpu"]["median_ms"], f + ".gpu.median_ms (per-frame medians)")
    put("drone_gpu_max_ms", s["gpu"]["max_ms"], f + ".gpu.max_ms (the worst frame's median)")
    put("drone_gpu_worst_run_ms", s["gpu_worst_run_ms"], f + ".gpu_worst_run_ms (the worst single run)")
    put("drone_e2e_ms", s["gpu_e2e"]["median_ms"], f + ".gpu_e2e.median_ms (with the copies)")
    put("drone_e2e_max_ms", s["gpu_e2e"]["max_ms"], f + ".gpu_e2e.max_ms (the worst frame's median)")
    put("drone_x_json", s["cpu_over_gpu_median"], f + ".cpu_over_gpu_median")
    put("drone_frames", s["gpu"]["n"], f + ".gpu.n")
    fd = f"{f5}:drones"
    put("drone_first_version", all(dr["config"][k] == v for k, v in old.items()) and dr["gpu_cell"] == "ch08_old",
        fd + ".config.{dense, lookup, fit_grids} == rules.ch08_old_path (OLD_PATH)")
    put("drone_dense_tiles", sum(fm["gpu_info"]["dense_label_tiles"] + fm["gpu_info"]["dense_paint_tiles"]
                                 for fm in dr["frames"]), fd + ".frames[].gpu_info.dense_*_tiles (summed)")
    cen = dr["default_engine_dense_census"]
    put("drone_census_tiles", cen["tiles_touched"], fd + ".default_engine_dense_census.tiles_touched (untimed)")
    put("drone_census_label", cen["ccl_tiles"], fd + ".default_engine_dense_census.ccl_tiles")
    put("drone_census_paint", cen["paint_ccl_tiles"], fd + ".default_engine_dense_census.paint_ccl_tiles")
    put("drone_ok", dr["checks_ok"] and dr["cpu_checks_ok"] and dr["e2e_matches_cpu"] and dr["last_timed_run_matches_cpu"]
        and all(fm["gpu_check"]["ok"] and fm["gpu_check"]["labels_match_cpu"] for fm in dr["frames"])
        and dr["frames_sha1_equal_to_v4"] == len(dr["frames"]),
        fd + ".checks_ok, cpu_checks_ok, e2e_matches_cpu, last_timed_run_matches_cpu, frames[].gpu_check, "
        "frames_sha1_equal_to_v4")
    meta = _load(DRONE_META)
    put("drone_blobs_median", sorted(meta["blobs_per_frame"])[len(meta["blobs_per_frame"]) // 2],
        "assets/drones_labels/meta.json:blobs_per_frame (median)")
    put("cpu_cores", CPU_CORES, CPU_CORES_SOURCE)

    # ================= act 1's mechanism (Jul 19-20): GPU only =================
    d3 = _load(CH03)
    q = _scene(d3, "sq_8000_center")
    f = CH03.name + ":sq_8000_center"
    put("big_blocks", q["multi_blocks"], f + ".multi_blocks")
    put("big_tpb", q["multi_tpb"], f + ".multi_tpb")
    put("big_rings4", q["levels"], f + ".levels")
    put("big_rings8", q["conn8_levels"], f + ".conn8_levels")
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

    # act 1, two blocks (Jul 19): where the scheduler put them, and the
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

    # ================= act 3: the tiles' own sessions (Oct 15-16) =================
    t7 = _load(T7)
    s7 = _scene(t7, "sq_8000_center")
    cfg, f = t7["new_config"], f"{T7.name}:sq_8000_center"
    put("t7_rounds", s7[cfg + "_rounds"], f + f".{cfg}_rounds (the default)")
    put("t7_levels8", s7["levels8"], f + ".levels8")
    put("t7_one_sync", t7["default_switches"]["one_sync"], f"{T7.name}:default_switches.one_sync")
    d8full = _load(D8)
    d8, w8 = _scene(d8full, "input_blobs"), _scene(_load(W8), "input_blobs")
    f = f"{D8.name}:input_blobs.ch08_L8"
    for key in ("rounds", "seeds", "seeds_lattice", "comps", "interior_comps", "n_blobs", "links", "roots",
                "claims", "ccl_tiles", "paint_ccl_tiles", "valve_round"):
        put(f"d8_{key}", d8["ch08_L8_" + key], f + "_" + key)
    put_list("d8_frontier", d8["ch08_L8_frontier_comps"], f + "_frontier_comps")
    same = ("rounds", "seeds", "seeds_lattice", "comps", "interior_comps", "n_blobs", "links", "roots", "claims",
            "frontier_comps")
    put("w8_same_as_d8", all(w8["ch08_L8_" + k] == d8["ch08_L8_" + k] for k in same),
        f"{W8.name}:input_blobs.ch08_L8_{{{', '.join(same)}}} == D8's")
    # D8 timed ch08's first version (its old_path, OLD_PATH) beside the
    # default on the frame: the same counters, so the default's floors
    # (WV) and counters (D8) describe the first version's run, V5's r13
    same_old = same + ("ccl_tiles", "paint_ccl_tiles")
    put("d8_old_same_as_default", all(d8["ch08_old_" + k] == d8["ch08_L8_" + k] for k in same_old),
        f"{D8.name}:input_blobs.ch08_old_{{{', '.join(same_old)}}} == ch08_L8's")
    put("d8_old_is_v5_engine", d8full["old_path"] == v5["rules"]["ch08_old_path"],
        f"{D8.name}:old_path == {V5.name}:rules.ch08_old_path")

    # ================= act 3: the floors, recorded from the kernels (Oct 17) =================
    tl = _load(TL)
    sq8, cu = tl["square"], tl["closeup"]
    f = f"{TL.name}:square"
    put("tile_px", sq8["width"] / sq8["tiles"][0], f + ".width / tiles[0]")
    put("tl_tiles_side", sq8["tiles"][0], f + ".tiles[0]")
    put("tl_n_tiles", sq8["n_tiles"], f + ".n_tiles")
    put("tl_levels", sq8["oracle_levels"], f + ".oracle_levels (the @njit oracle's BFS levels)")
    put("tl_rounds", sq8["chapter_kernel"]["rounds"], f + ".chapter_kernel.rounds")
    put("tl_twin_rounds_same", all(r["rounds"] == sq8["chapter_kernel"]["rounds"] for r in sq8["twin_runs"]),
        f + ".twin_runs[].rounds (all equal the chapter kernel's)")
    put("tl_first_open_max", sq8["first_open_max"], f + ".first_open_max")
    put("tl_chebyshev", sq8["first_open_equals_chebyshev"], f + ".first_open_equals_chebyshev")
    put_list("tl_opened_per_round", sq8["opened_per_round"], f + ".opened_per_round")
    put("tl_seed_tile_centre", sq8["seed_tile"] == [s // 2 for s in sq8["tiles"]], f + ".seed_tile (the centre)")
    f = f"{TL.name}:closeup"
    put("cu_blob_px", cu["blob_px"], f + ".blob_px")
    put("cu_blob_rank", cu["blob_rank_by_size"], f + ".blob_rank_by_size")
    put("cu_red_px", cu["red_px"], f + ".red_px (of the tile's 1,024)")
    put("cu_entry_px", cu["entry_px"], f + ".entry_px")
    put_list("cu_reached", cu["reached_px_after_sweep"], f + ".reached_px_after_sweep")
    put("cu_trips", cu["trips"], f + ".trips")
    put("cu_sweeps_named", cu["sweep_names"] == ["entry"] + ["row", "diag", "col"] * cu["trips"],
        f + ".sweep_names == entry, then (row, diag, col) per trip")
    put("cu_tile_round", cu["tile_first_round"], f + ".tile_first_round")
    put("cu_tile_visits", cu["tile_visits"], f + ".tile_visits")
    put("tl_ok", all(tl["checks"].values()), f"{TL.name}:checks (all true)")

    wv = _load(WV)
    cr, fu = wv["crop"], wv["full"]
    f = f"{WV.name}:crop"
    put("crop_side", cr["side"], f + ".side")
    put("crop_tiles_side", cr["tiles"][0], f + ".tiles[0]")
    put("crop_aligned", all(o % 256 == 0 for o in cr["origin"]), f + ".origin (multiples of 256: the frame's lattice)")
    for key in ("blobs", "pieces", "interior_pieces", "max_pieces_per_tile", "seeds", "seeds_lattice_only",
                "rounds", "claims", "meeting_edges", "collisions_repeated", "links", "roots",
                "blobs_with_two_or_more_seeds", "lattice_tiles"):
        put(f"crop_{key}", cr[key], f + "." + key)
    put_list("crop_claimed_per_round", cr["claimed_per_round"], f + ".claimed_per_round")
    put("crop_dense_tiles", cr["counters"]["ccl_tiles"] + cr["counters"]["paint_ccl_tiles"],
        f + ".counters.ccl_tiles + paint_ccl_tiles (tiles on the dense-tile path)")
    f = f"{WV.name}:full"
    for key in ("blobs", "pieces", "interior_pieces", "max_pieces_per_tile", "seeds", "seeds_lattice_only",
                "rounds", "claims", "links", "roots", "tiles_with_red", "red_px", "blobs_with_two_or_more_seeds"):
        put(f"full_{key}", fu[key], f + "." + key)
    put_list("full_frontier", fu["frontier_pieces_per_round"], f + ".frontier_pieces_per_round")
    put("full_ccl_tiles", fu["counters"]["ccl_tiles"], f + ".counters.ccl_tiles (tiles off the warp's piece loop)")
    put("full_paint_ccl_tiles", fu["counters"]["paint_ccl_tiles"], f + ".counters.paint_ccl_tiles (tiles painted off it)")
    # WV recorded the default engine, which V5 timed for reference (ch08);
    # V5's r13 is the first version (ch08_old), whose counters D8 found
    # equal to the default's (d8_old_same_as_default). Both V5 runs give
    # WV's rounds and blobs
    e, cf = fu["engine"], _scene(v5, "input_blobs")["cells"]
    same = ("dense", "lookup", "lattice", "waves_blocks", "waves_tpb")
    put("wv_engine_is_v5", all(e[k] == cf["ch08"]["config"][k] for k in same)
        and wv["config"]["connectivity"] == cf["ch08"]["config"]["connectivity"]
        and all((fu["rounds"], fu["blobs"]) == (cf[k]["info"]["rounds"], cf[k]["info"]["n_blobs"])
                for k in ("ch08", "ch08_old"))
        and cf["ch08_old"]["config"]["lattice"] == e["lattice"]
        and cf["ch08_old"]["config"]["waves_tpb"] == e["waves_tpb"],
        f"{WV.name}:full.engine.{{{', '.join(same)}}} == V5's input_blobs.ch08 (reference); rounds, blobs == "
        "ch08's and ch08_old's; ch08_old's lattice and waves_tpb the same")
    put("wv_ok", all(wv["checks"].values()), f"{WV.name}:checks (all true)")
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


@lru_cache(maxsize=1)
def act3_arrays() -> dict:
    """TL's and WV's arrays by name (assets/tiles/README.md lists them):
    sq_* (the 64 Mpx square's 250 x 250 tiles), cu_* (the warp close-up),
    crop_* (the 384 px crop: per pixel, per tile, per piece, per meeting)
    and full_* (the frame's 282 x 282 tiles, and claim rounds at 900 px).
    Index order [x, y] everywhere (x = PNG column), drawn as (row, col):
    a lane is a screen row. The 900 px frame PNGs are PNG order, so v5
    transposes them once at the source (state.picture()); `floors()`
    checks that the two then agree."""
    import numpy as np

    out = {}
    for path in (TL, WV):
        z = np.load(path.with_suffix(".npz"))
        out.update({k: z[k] for k in z.files})
    return out


@lru_cache(maxsize=1)
def floors() -> dict:
    """The act-3 floor facts, computed from TL's and WV's arrays (and the
    900 px picture for the orientation): what r10-r13 draw, pinned."""
    import numpy as np

    a, tl, wv = act3_arrays(), _load(TL), _load(WV)
    cu, tp = tl["closeup"], 32
    out: dict = {}

    # r10-r11: the close-up's 7 x 7 tile context; ctx[i, j] = tile (ox + i, oy + j)
    fo = a["cu_ctx_first_open"]
    ox, oy = (o // tp for o in cu["context_origin"])
    ci, cj = cu["tile"][0] - ox, cu["tile"][1] - oy
    k = int(fo[ci, cj])
    out["ctx_seed_tile_round"] = int(fo[cu["seed_tile"][0] - ox, cu["seed_tile"][1] - oy])
    out["ctx_centre_round"] = k
    out["ctx_tiles_in_centre_round"] = int((fo == k).sum())
    out["ctx_tiles_before"] = int(((fo >= 0) & (fo < k)).sum())
    out["ctx_entry_side_round"] = int(fo[ci, cj - 1]) if cu["entry_bits"] == [0] else -9   # bit 0 = y - 1

    # the closed tile's border bits and where they go: lane i = x0 + i (a
    # screen row), bit j = y0 + j; 8-way neighbours outside the tile that are red
    fin = a["cu_final_chapter"].astype(np.uint64)
    reached = ((fin[:, None] >> np.arange(tp, dtype=np.uint64)) & 1).astype(bool)
    red = a["cu_ctx_red"]
    x0, y0 = ci * tp, cj * tp
    out["cu_reached_is_fill"] = bool((reached == (a["cu_ctx_fill"] & red)[x0:x0 + tp, y0:y0 + tp]).all())
    out["cu_border_bits"] = (int(reached[0].sum()), int(reached[:, 0].sum()),
                             int(reached[:, -1].sum()), int(reached[-1].sum()))    # top, left, right, bottom
    R = np.zeros_like(red)
    R[x0:x0 + tp, y0:y0 + tp] = reached
    D = np.zeros_like(red)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            D |= np.roll(np.roll(R, dx, 0), dy, 1)                  # the tile sits mid-context: no wrap
    D &= red
    D[x0:x0 + tp, y0:y0 + tp] = False
    hit: dict = {}
    for p, q in np.argwhere(D):
        hit[(p // tp, q // tp)] = hit.get((p // tp, q // tp), 0) + 1
    new = {t: v for t, v in hit.items() if fo[t] > k}
    out["cu_targets_new"] = len(new)
    out["cu_targets_new_px"] = tuple(sorted(new.values()))
    out["cu_targets_new_rounds"] = tuple(sorted({int(fo[t]) for t in new}))
    out["cu_targets_old_rounds_max"] = max(int(fo[t]) for t in hit if t not in new)

    # r12: the crop, run as its own image
    cr, crop = wv["crop"], {}
    lab, cred = a["crop_label"], a["crop_red"]
    edge = np.zeros_like(cred)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    out["crop_blobs_cut_by_edge"] = len(np.unique(lab[cred & edge]))
    ty_n = cr["tiles"][1]
    cmax = np.full(cr["tiles"], -9)
    cmin = np.full(cr["tiles"], 99)
    seeds = np.zeros(cr["tiles"], int)
    for t, r, s in zip(a["crop_comp_tile"], a["crop_comp_round"], a["crop_comp_seed"]):
        tx, ty = divmod(int(t), ty_n)
        cmax[tx, ty], cmin[tx, ty] = max(cmax[tx, ty], r), min(cmin[tx, ty], r)
        seeds[tx, ty] += int(s)
    gx, gy = (o // tp for o in cr["origin"])
    window = (slice(gx, gx + cr["tiles"][0]), slice(gy, gy + cr["tiles"][1]))
    has = cmax > -9
    fmax, fmin = a["full_tile_round_max"][window].astype(int), a["full_tile_round_min"][window].astype(int)
    out["crop_tiles_with_red"] = int(has.sum())
    out["crop_tiles_other_last_round"] = int((cmax[has] != fmax[has]).sum())
    out["crop_tiles_other_first_round"] = int((cmin[has] != fmin[has]).sum())
    out["crop_last_round"] = int(cmax[has].max())
    out["crop_window_last_round_in_frame"] = int(fmax[has].max())
    out["crop_seeds_are_frame_seeds"] = bool((seeds == a["full_tile_seeds"][window]).all())

    # r13: the frame's claim rounds at 900 px, and the orientation
    r9 = a["full_round_900"]
    out["full900_last_round"] = int(r9.max())
    out["full900_interior"] = int((r9 == -2).sum())
    try:
        from PIL import Image

        pic = np.asarray(Image.open(PICTURE).convert("RGB"))
        fg = (pic != 255).any(axis=2)                            # red (and seeds) on white
        red9 = r9 != -1

        def iou(m):
            return float((m & fg).sum() / (m | fg).sum())

        out["picture_iou_as_is"], out["picture_iou_transposed"] = iou(red9), iou(red9.T)
    except FileNotFoundError:
        out["picture_iou_as_is"] = out["picture_iou_transposed"] = math.nan
    return out


def derived() -> dict[str, float]:
    n = {k: v.value for k, v in load().items()}
    levels, blocks, tpb = n["big_level_sizes"], int(n["big_blocks"]), int(n["big_tpb"])
    load2 = _two_ring(levels)
    spill = [L for L, v in enumerate(load2) if v > n["ring_slots"]]   # rings whose two-ring load overflows
    busy = [busy_blocks(s, blocks, tpb) for s in levels]
    every = [L for L, b in enumerate(busy) if b == blocks]
    reached = n["cu_reached"]
    return {
        "budget_ms": 1000 / 30,
        # i1: the CPU on the 81 Mpx frame (labels only, its reset on the clock) against the budget
        "frame_cpu_over_budget": n["njit_ms"] * 30 / 1000,
        # row 1, one small blob: r02's one block (4 neighbours), r14's tiles (8 neighbours, no depth)
        "small_cpu_x": n["small_gpu_ms"] / n["small_cpu_ms"],          # the CPU wins by (r02)
        "small_tiles_x": n["small_cpu8_ms"] / n["small_tiles_ms"],     # the GPU wins by, kernel time (r14)
        "small_e2e_cpu_x": n["small_tiles_e2e_ms"] / n["small_cpu8_ms"],   # the CPU wins by, with the copy
        "small_ch05_x": n["small_cpu8_depth_ms"] / n["small_ch05_ms"],     # act 2's seeds here (not shown)
        # row 2, one huge blob: act 1 (4 neighbours, then 8 with depth), then the tiles (8, no depth)
        "big_1block_x": n["big_cpu4_ms"] / n["big_1block_ms"],
        "big_2v1_x": n["big_1block_ms"] / n["big_2blocks_ms"],         # paired: big_2v1_paired
        "big_2blocks_x": n["big_cpu4_ms"] / n["big_2blocks_ms"],
        "big_48_x": n["big_cpu4_ms"] / n["big_48blocks_ms"],
        "big_8ways_x": n["big_cpu8_ms"] / n["big_8ways_ms"],
        "big_tiles_x": n["big_cpu_fill_ms"] / n["big_tiles_ms"],
        "big_tiles_e2e_x": n["big_cpu_fill_ms"] / n["big_tiles_e2e_ms"],   # with the copies
        "big_8_over_tiles": n["big_8ways_ms"] / n["big_tiles_ms"],     # GPU against GPU (paired too)
        "big_tuned_over_tiles": n["big_tuned_ms"] / n["big_tiles_ms"],
        "big_stops_8ways": 2 * n["big_levels8_v5"],                    # two gates per ring
        "big_stops_tiles": n["big_tiles_rounds"],                      # one gate per round
        "rings_per_round": n["big_levels8_v5"] / n["big_tiles_rounds"],
        "big_tiles_search_share": n["big_tiles_rounds_ms"] / n["big_tiles_ms"],    # the rounds' share
        # what the per-call allocation cost the oracles (reference: v5's earlier CPU)
        "big_oracle4_over_twin": n["big_oracle4_ms"] / n["big_cpu4_ms"],
        "big_oracle8_over_twin": n["big_oracle8_ms"] / n["big_cpu8_ms"],
        "frame_oracle_over_twin": n["frame_oracle_ms"] / n["njit_ms"],
        "big_depth_share": 1 - n["big_cpu_fill_ms"] / n["big_cpu8_ms"],    # no depth, reset on vs depth, before
        "frame_reset_share": 1 - n["count_cpu_pre_ms"] / n["njit_ms"],     # the label map's reset, on the clock
        # row 3, many blobs: the 81 Mpx frame, labels only: act 2 (reset before), then the waves (reset on)
        "est_ch03_cpu_x": n["est_ch03_ms"] / n["count_cpu_pre_ms"],    # the CPU wins by (r07, estimate)
        "est_ch04_cpu_x": n["est_ch04_ms"] / n["count_cpu_pre_ms"],    # the CPU wins by (r08, estimate)
        "est_ch03_over_ch04": n["est_ch03_ms"] / n["est_ch04_ms"],     # two per launch is faster
        "loop_ch03_over_est": n["loop_ch03_ms"] / n["est_ch03_ms"],    # the estimate runs low
        "loop_ch04_over_est": n["loop_ch04_ms"] / n["est_ch04_ms"],
        "est_host_cpu_x": max(n["est_ch03_host_ms"], n["est_ch04_host_ms"]) / n["count_cpu_pre_ms"],
        "count_ch05_x": n["count_cpu_pre_ms"] / n["count_ch05_ms"],
        "count_x": n["count_cpu_ms"] / n["count_gpu_ms"],
        "count_e2e_x": n["count_cpu_ms"] / n["frame_e2e_ms"],          # with the copies
        "frame_ch05_over_ch08": n["ch05_ms"] / n["frame_ms"],
        "frame_v4_over_label": n["frame_cpu_v4_ms"] / n["njit_ms"],    # v4's CPU baseline did more work
        # the drones
        "drone_x": n["drone_cpu_ms"] / n["drone_gpu_ms"],              # kernel time (not shown)
        "drone_e2e_cpu_x": n["drone_e2e_ms"] / n["drone_cpu_ms"],      # the CPU wins by, with the copies
        # act 1's GPU-only numbers (CH03, DB2)
        "sm_pct_1": 100 / n["sm"],
        "sm_pct_2": 200 / n["sm"],
        "threads": n["big_blocks"] * n["big_tpb"],                     # 12,288
        "snake_stops": 2 * n["snake_rings"],                           # 65,792
        "us_per_stop": 1000 * n["snake_48blocks_ms"] / (2 * n["snake_rings"]),
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
        # act 3, the warp close-up (TL)
        "cu_tile_px": n["tile_px"] ** 2,                               # 1,024
        "cu_last_growth_trip": max(t for t in range(1, int(n["cu_trips"]) + 1)
                                   if reached[3 * t] > reached[3 * t - 3]),   # 3: trip 4 only proves it
    }


def facts() -> list[tuple[str, bool]]:
    """Every spoken sentence (and shown number) that rests on a number."""
    n = {k: v.value for k, v in load().items()}
    d = derived()
    fl = floors()
    levels, ids, ring = n["big_level_sizes"], n["big_sm_ids"], int(n["ring_slots"])
    blocks, tpb = int(n["big_blocks"]), int(n["big_tpb"])
    run = [ms for ms in n["sweep_grid_ms"] if ms is not None and ms == ms]   # the grid's measured cells
    reached, trips = n["cu_reached"], int(n["cu_trips"])
    claimed = n["crop_claimed_per_round"]
    rng = {k: n[k] for k in ("small_gpu_x_range", "small_tiles_x_range", "big_1block_x_range", "big_48_x_range",
                             "big_8ways_x_range", "big_tiles_x_range", "ch05_x_range", "frame_x_range")}
    return [
        # ---- the sessions themselves
        ("V5: every check passed; bench_v5.py and cpu_twins.py were the committed ones",
         n["v5_ok"] == 1 and n["v5_script_at_head"] == 1 and n["frame_ok"] == 1 and n["small_ok"] == 1
         and n["big_ok"] == 1 and n["drone_ok"] == 1),
        ("V5: all 8 timed blocks at full clocks: P0 only, memory 8,001 MHz, SM median 2,070 in every block; 7 on "
         "their first attempt, ch05 on the frame on its second (its first sat in P3 under the power cap)",
         n["v5_all_full_clocks"] == 1 and n["v5_full_clocks_used"] == 1 and n["v5_p0_only"] == 1
         and n["v5_blocks"] == 8 and n["v5_first_attempt_blocks"] == 7 and n["v5_retried_is_ch05"] == 1
         and n["v5_mem_clock_min"] == 8001 and n["v5_sm_clock_median"] >= n["v5_sm_median_rule"] == 2040
         and n["v5_sm_clock_median"] == 2070 and n["story_full_clocks"] == 1),
        ("V5: three blocks logged the software power cap (the huge blob's, the waves', ch05's); all at full clocks",
         n["v5_power_capped_blocks"] == 3),
        ("V5 like for like: every GPU cell's twin is the session's rule, keeps no output its GPU side does not, "
         "takes the same neighbours, resets in the same scope, and equals its oracle; twin.x = CPU / GPU",
         n["v5_twins_ok"] == 1 and n["v5_all_like_for_like"] == 1),
        ("V5 like for like: every shown CPU-vs-GPU ratio is V5's own story entry (STORY_SHOWN)",
         all(math.isclose(d[k], 1 / n[f"story_{s}_x"] if inv else n[f"story_{s}_x"], rel_tol=1e-9)
             for k, s, inv in STORY_SHOWN)),
        ("V5 reference: the oracles as called (fresh arrays every call) took 1.5x and 1.4x their twins "
         "(4 neighbours on the square, labels on the frame): the allocations v5's evening session timed",
         round(d["big_oracle4_over_twin"], 1) == 1.5 and round(d["frame_oracle_over_twin"], 1) == 1.4),
        ("V5 reference: the label map's reset is 10% of the CPU's frame time (on the clock 225 ms, before it 202)",
         round(100 * d["frame_reset_share"]) == 10),
        ("act 1's levels: V5's kernels ran CH03's rings (8,001 at 4 ways, 4,001 at 8 ways)",
         n["big_levels4_v5"] == n["big_rings4"] and n["big_levels8_v5"] == n["big_rings8"]),
        ("V5's frame: 2,522 blobs in the scene, the CPU's labels and both wave engines", n["n_blobs_agree"] == 1),
        # ---- intro
        ("i0 thirty-three milliseconds", round(d["budget_ms"]) == 33),
        ("i1 on frames this small, compiled CPU code keeps up (drone frames, labels only, worst frame 0.88 ms)",
         n["drone_cpu_max_ms"] < d["budget_ms"] and n["drone_ok"] == 1),
        ("i1 drone card: CPU 0.679 ms against the 33 ms budget, the GPU a '?' (as the 81 Mpx card): no GPU drone "
         "time in kernel time anywhere in the cut",
         card_ms(n["drone_cpu_ms"]) == "0.679 ms" and n["drone_cpu_ms"] < d["budget_ms"]),
        ("i1 eighty-one megapixels", round(n["n_pixels"] / 1e6) == 81),
        ("i1 the CPU needs over a fifth of a second (labels only, the reset on the clock: every round 214-238 ms; "
         "stopwatch 225 ms; not a quarter)",
         200 < min(n["njit_round_ms"]) and max(n["njit_round_ms"]) < 250 and card_ms(n["njit_ms"]) == "225 ms"),
        ("i1 ... against the same thirty-three: nearly seven times over (not spoken)",
         6.5 < d["frame_cpu_over_budget"] < 7),
        # ---- act 1
        ("r01 twenty-four SMs", n["sm"] == 24),
        ("r02 two hundred fifty-six workers, called threads, in one block", n["small_tpb"] == 256
         and n["small_gpu_blocks"] == 1),
        ("r02 on one small blob, the CPU still wins: one block against the 4-neighbour CPU, in every CPU round",
         n["small_cpu_ms"] < n["small_gpu_ms"] and max(rng["small_gpu_x_range"]) < 1),
        ("r02 race and scoreboard row 1: CPU 4.3x (0.121 vs 0.520 ms, the card reads true)",
         card_x(d["small_cpu_x"]) == "4.3×" and card_row_ok(n["small_gpu_ms"], n["small_cpu_ms"])
         and card_ms(n["small_cpu_ms"]) == "0.121 ms" and card_ms(n["small_gpu_ms"]) == "0.520 ms"),
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
        ("r03b now one block is a little faster than the 4-neighbour CPU: 1.09x, above 1 in every CPU round "
         "(1.02-1.17), under 1.2 (caption 1.1x)",
         1.0 < min(rng["big_1block_x_range"]) and max(rng["big_1block_x_range"]) < 1.2
         and card_x(d["big_1block_x"]) == "1.1×"),
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
        ("r04 twice as fast (GPU against GPU, caption 2x again; medians and paired rounds agree)",
         1.9 <= d["big_2v1_x"] < 2.5 and 1.9 <= n["big_2v1_paired"] < 2.5 and round(d["big_2v1_x"]) == 2),
        ("r04 twenty-two SMs stay dark", n["sm"] - 2 == 22),
        ("r05 forty-eight blocks", n["big_blocks"] == 48),
        ("r05 two on each: every SM id 0-23 twice",
         len(ids) == n["big_blocks"] and sorted(ids) == sorted(list(range(24)) * 2)),
        ("r05 the measured order: blocks 0-11 on SMs 0, 2, .., 22, blocks 12-23 on 1, 3, .., 23, then again",
         ids == (tuple(range(0, 24, 2)) + tuple(range(1, 24, 2))) * 2),
        ("r05 eight and a half times the 4-neighbour CPU (caption 8.5x; the CPU's rounds give 8.0-9.1, within 8%)",
         card_x(d["big_48_x"]) == "8.5×" and all(abs(x / 8.5 - 1) < 0.08 for x in rng["big_48_x_range"])),
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
        ("r06 caption and scoreboard row 2: GPU 11x against the 8-neighbour CPU with depth (1,126 vs 101 ms, the "
         "card reads true; 9.9-11.7 over the CPU's rounds)",
         card_x(d["big_8ways_x"]) == "11×" and card_row_ok(n["big_cpu8_ms"], n["big_8ways_ms"])
         and card_ms(n["big_cpu8_ms"]) == "1,126 ms" and card_ms(n["big_8ways_ms"]) == "101 ms"
         and n["big_cpu_ms"] == n["big_cpu8_ms"] and min(rng["big_8ways_x_range"]) > 9),
        # ---- act 2 (V5: v4's estimates re-made at 8 neighbours, against the labels-only CPU, reset before the
        # clock as act 2's loops and ch05 reset theirs)
        ("r07 twenty-five hundred blobs", 2400 <= n["n_blobs"] <= 2600),
        ("r07 one launch each, 8 neighbours (as the CPU's labels)",
         n["est_ch03_calls"] == n["n_blobs"] and n["est_ch03_conn"] == 8),
        ("r07 slower than the CPU by every measure: the estimate, its mean-based form, v4's host clock and the "
         "full loop, each over the CPU's slowest round (caption ~1,462 ms; the CPU by 7.2x)",
         min(n["est_ch03_ms"], n["est_ch03_mean_ms"], n["est_ch03_host_ms"], n["loop_ch03_ms"])
         > max(n["count_cpu_pre_round_ms"]) and card_x(d["est_ch03_cpu_x"]) == "7.2×"
         and "~" + card_ms(n["est_ch03_ms"]) == "~1,462 ms"),
        ("r08 two per launch, 8 neighbours", n["est_ch04_calls"] * 2 == n["n_blobs"] and n["est_ch04_conn"] == 8),
        ("r08 faster (than one per launch) by every measure, paired 1.72x",
         n["est_ch04_ms"] < n["est_ch03_ms"] and n["est_ch04_mean_ms"] < n["est_ch03_mean_ms"]
         and n["est_ch04_host_ms"] < n["est_ch03_host_ms"] and n["loop_ch04_ms"] < n["loop_ch03_ms"]
         and n["est_ch03_over_ch04_paired"] > 1.5),
        ("r08 but still slower than the CPU, by every measure, over the CPU's slowest round (caption ~850 ms; the "
         "CPU by 4.2x)",
         min(n["est_ch04_ms"], n["est_ch04_mean_ms"], n["est_ch04_host_ms"], n["loop_ch04_ms"])
         > max(n["count_cpu_pre_round_ms"]) and card_x(d["est_ch04_cpu_x"]) == "4.2×"
         and "~" + card_ms(n["est_ch04_ms"]) == "~850 ms"),
        ("r07-r08 the estimates run low (the full loops take 1.13x and 1.35x): the direction is safe; every form "
         "is 4.2-9.5x the CPU",
         1 < d["loop_ch03_over_est"] < 1.15 and 1.3 < d["loop_ch04_over_est"] < 1.4
         and round(d["est_ch04_cpu_x"], 1) == 4.2 and round(d["est_host_cpu_x"], 1) == 9.5),
        ("r09 about fifty milliseconds (53.9 ms, full clocks on the block's second attempt; caption 53.9 ms)",
         round(n["ch05_ms"], -1) == 50 and card_ms(n["ch05_ms"]) == "53.9 ms"),
        ("r09 faster than the CPU, in every CPU round (3.5-4.2x)", min(rng["ch05_x_range"]) > 1),
        ("r09 but over budget", n["ch05_ms"] > d["budget_ms"]),
        ("r09 the queue's counter: every red pixel of the frame (13,451,960)",
         n["red_px"] == n["full_red_px"] == 13_451_960),
        ("r09 scoreboard row 3 fills: GPU 3.8x against the labels-only CPU, reset before the clock (202 vs 53.9 "
         "ms, the same 81 Mpx frame; the strip shows the factor only)",
         card_x(d["count_ch05_x"]) == "3.8×" and round(n["count_px"] / 1e6) == 81
         and card_ms(n["count_cpu_pre_ms"]) == "202 ms"),
        # ---- act 3: the tile (r10)
        ("r10 squares of thirty-two by thirty-two (on the square and on the crop)",
         n["tile_px"] == 32 and n["crop_side"] / n["crop_tiles_side"] == 32),
        ("r10 one warp, thirty-two threads that run in step, takes a tile; each thread holds one row (32 threads, "
         "32 rows)", WARP_LANES == n["tile_px"]),
        ("r10 three blocks per SM in act 3: both kernels launch 72 cooperative blocks (tiles 72 x 128, waves 72 x "
         "256, the first version's)",
         n["big_tiles_blocks"] == n["small_tiles_blocks"] == n["frame_waves_blocks"] == 3 * n["sm"]
         and n["big_tiles_tpb"] == 128 and n["frame_waves_tpb"] == 256),
        ("r10 the close-up tile is one blob's: the frame's largest, its fill = its red pixels",
         n["cu_blob_rank"] == 0 and n["cu_reached"][-1] == n["cu_red_px"] and n["tl_ok"] == 1
         and fl["cu_reached_is_fill"]),
        ("r10 floor: the blob's seed tile is in the context (round 0); the centre opened in round 4 with 6 other "
         "tiles, 15 tiles before it, its entry side (bit 0) in round 3",
         fl["ctx_seed_tile_round"] == 0 and fl["ctx_centre_round"] == n["cu_tile_round"] == 4
         and fl["ctx_tiles_in_centre_round"] == 7 and fl["ctx_tiles_before"] == 15
         and fl["ctx_entry_side_round"] == 3),
        ("r10 fill along the rows, step diagonally, along the columns, repeat (the recorded sweeps)",
         n["cu_sweeps_named"] == 1 and len(reached) == 1 + 3 * trips),
        ("r10 every sweep of the first two trips adds pixels",
         all(reached[i] > reached[i - 1] for i in range(1, 7))),
        ("r10 until nothing changes: the last trip adds nothing (caption 4 trips / the last changes nothing)",
         trips == 4 and d["cu_last_growth_trip"] == trips - 1 and len(set(reached[-3:])) == 1
         and reached[-3] == reached[-4]),
        ("r10 the tile was visited once, in its round", n["cu_tile_visits"] == 1 and n["cu_tile_round"] >= 1),
        # ---- act 3: the rounds (r11)
        ("r11 only the tile's border bits go out: 31 / 26 / 18 / 2 (top, left, right, bottom)",
         fl["cu_border_bits"] == (31, 26, 18, 2)),
        ("r11 ... into the tiles around it: 4 tiles not yet open (32, 18, 1, 1 px), all opened in round 5; the "
         "others they touch were open by round 4",
         fl["cu_targets_new"] == 4 and fl["cu_targets_new_px"] == (1, 1, 18, 32)
         and fl["cu_targets_new_rounds"] == (n["cu_tile_round"] + 1,)
         and fl["cu_targets_old_rounds_max"] <= n["cu_tile_round"]),
        ("r11 one gate per round (the default, in V5 and in the tiles' own session)",
         n["big_tiles_one_sync"] == 1 and n["t7_one_sync"] == 1),
        ("r11 four thousand rings (8 ways; V5, CH03, the oracle and the tiles' session agree)",
         n["big_levels8_v5"] == n["big_rings8"] == n["tl_levels"] == n["t7_levels8"]
         and round(n["big_levels8_v5"], -3) == 4000),
        ("r11 a hundred twenty-seven rounds (V5, the export, its twin runs, the tiles' session)",
         n["big_tiles_rounds"] == n["tl_rounds"] == n["t7_rounds"] == 127 and n["tl_twin_rounds_same"] == 1),
        ("r11 floor: the seed tile, then 125 rings of tiles (rounds 0-125, a tile's round = its ring); "
         "round 126 opens none",
         n["tl_chebyshev"] == 1 and n["tl_seed_tile_centre"] == 1 and n["tl_first_open_max"] == 125
         and len(n["tl_opened_per_round"]) == n["tl_rounds"] and n["tl_opened_per_round"][-1] == 0
         and sum(n["tl_opened_per_round"]) == n["tl_n_tiles"] and n["tl_n_tiles"] == n["tl_tiles_side"] ** 2),
        ("r11 from a hundred milliseconds to three (r06's kernel, 101 ms, then the tiles, 3.12; paired 32x)",
         round(n["big_8ways_ms"], -1) == 100 and round(n["big_tiles_ms"]) == 3
         and 30 < n["big_8_over_tiles_paired"] < 35 and round(d["big_8_over_tiles"]) == 32),
        ("r11 scoreboard row 2 jumps: GPU 340x against the 8-neighbour CPU without depth, reset on the clock "
         "(1,065 vs 3.12 ms, the card reads true; 311-345 over the CPU's rounds)",
         card_x(d["big_tiles_x"]) == "340×" and card_ms(n["big_tiles_ms"]) == "3.12 ms"
         and card_ms(n["big_cpu_fill_ms"]) == "1,065 ms" and card_row_ok(n["big_cpu_fill_ms"], n["big_tiles_ms"])
         and min(rng["big_tiles_x_range"]) > 300),
        ("r11 but that's one blob, from one seed", len(n["big_seed"]) == 2),
        # ---- act 3: the pieces (r12, the crop)
        ("r12 every tile finds its pieces with its own warp: no tile of the frame or the crop takes the "
         "dense-tile path, in labelling or paint (WV, D8, and V5's timed frame, both engines, and drones)",
         n["full_ccl_tiles"] == n["full_paint_ccl_tiles"] == 0 and n["d8_ccl_tiles"] == n["d8_paint_ccl_tiles"] == 0
         and n["crop_dense_tiles"] == 0 and n["frame_dense_tiles"] == 0 and n["drone_dense_tiles"] == 0),
        ("r12 the crop: 8 blobs, 135 pieces, at most 4 per tile, none interior",
         (n["crop_blobs"], n["crop_pieces"], n["crop_max_pieces_per_tile"], n["crop_interior_pieces"]) == (8, 135, 4, 0)),
        ("r12 some pieces are seeds (caption 36 seeds / 8 blobs; 11 from the lattice)",
         n["crop_seeds"] == 36 and n["crop_seeds"] < n["crop_pieces"] and n["crop_seeds_lattice_only"] == 11),
        ("r12 the crop is its own image: the frame's tile grid and seeds (36 = 36, tile for tile), but 6 of its 8 "
         "blobs are cut by its edge and 7 of its 98 tiles end on another round than in the frame (5 vs 4)",
         n["crop_aligned"] == 1 and fl["crop_seeds_are_frame_seeds"] and fl["crop_blobs_cut_by_edge"] == 6
         and fl["crop_tiles_with_red"] == 98 and fl["crop_tiles_other_last_round"] == 7
         and (fl["crop_last_round"], fl["crop_window_last_round_in_frame"]) == (5, 4)),
        ("r12 a wave goes piece to piece, one round at a time: every piece claimed once, in round = its distance",
         n["wv_ok"] == 1 and n["crop_claims"] + n["crop_seeds"] == n["crop_pieces"]
         and sum(claimed) == n["crop_claims"] and len(claimed) == n["crop_rounds"] == 6 and claimed[-1] == 0),
        ("r12 where two waves meet, the meeting is noted (116 meetings, no repeats; 6 blobs hold 2+ seeds)",
         n["crop_meeting_edges"] == 116 and n["crop_collisions_repeated"] == 0
         and n["crop_blobs_with_two_or_more_seeds"] == 6),
        ("r12 at the end each blob keeps its smallest label (links = seeds - blobs)",
         n["crop_links"] == n["crop_seeds"] - n["crop_roots"] and n["crop_roots"] == n["crop_blobs"]),
        # ---- act 3: the frame (r13)
        ("r13 the frame floors: act 3's [x, y] arrays match the 900 px picture only transposed",
         fl["picture_iou_transposed"] > 0.7 and fl["picture_iou_as_is"] < 0.2),
        ("r13 V5 timed ch08's first version (OLD_PATH, D8's old_path), whose counters D8 found equal to the "
         "default's; WV recorded the default, which V5 timed beside it (2.25 ms, reference)",
         n["frame_first_version"] == 1 and n["d8_old_same_as_default"] == 1 and n["d8_old_is_v5_engine"] == 1
         and n["wv_engine_is_v5"] == 1 and n["drone_first_version"] == 1),
        ("r13 over eleven thousand seeds (the export, the waves' sessions agree)",
         11_000 < n["full_seeds"] < 12_000 and n["full_seeds"] == n["d8_seeds"] and n["w8_same_as_d8"] == 1),
        ("r13 six rounds (V5, the export, the waves' session)",
         n["frame_rounds"] == n["full_rounds"] == n["d8_rounds"] == 6 and len(n["full_frontier"]) == 6
         and n["full_frontier"] == n["d8_frontier"]),
        ("r13 floor: at 900 px only rounds 0-4 show (rounds 5 and 6 claim 6 and 0 pieces); 94 samples interior",
         fl["full900_last_round"] == 4 and fl["full900_interior"] == 94),
        ("r13 every blob counted: 2,522 blobs, labels equal the CPU's",
         n["frame_labels_ok"] == 1 and n["full_blobs"] == n["d8_n_blobs"] == n["n_blobs"] == 2522),
        ("r13 two point two milliseconds (caption 2.21 ms)",
         round(n["frame_ms"], 1) == 2.2 and card_ms(n["frame_ms"]) == "2.21 ms"),
        ("r13 once the frame is on the card: inside the budget (kernel time; with the copies, 54.8 ms, over it)",
         n["frame_ms"] < d["budget_ms"] < n["frame_e2e_ms"] and card_ms_note(n["frame_e2e_ms"]) == "54.8 ms"),
        ("r13 the needle from act 2's 53.9 ms to 2.21 ms: 24x (V5's ratio of medians, 24.3)",
         round(d["frame_ch05_over_ch08"]) == 24 and math.isclose(d["frame_ch05_over_ch08"], n["frame_ch05_over_ch08"])),
        ("r13 scoreboard row 3 jumps: GPU 100x against the labels-only CPU, reset on the clock (225 vs 2.21 ms, the "
         "card reads true; 97-107 over the CPU's rounds)",
         card_x(d["count_x"]) == "100×" and card_row_ok(n["count_cpu_ms"], n["count_gpu_ms"])
         and min(rng["frame_x_range"]) > 90),
        # ---- the finale (r14)
        ("r14 one huge blob, or many at once: the GPU (kernel time, and with the copies too: 24x and 4.1x)",
         d["big_tiles_x"] > 1 and d["count_x"] > 1
         and n["big_tiles_e2e_ms"] < n["big_cpu_fill_ms"] and n["frame_e2e_ms"] < n["count_cpu_ms"]
         and card_x(d["big_tiles_e2e_x"]) == "24×" and card_x(d["count_e2e_x"]) == "4.1×"),
        ("r14 one small blob: in kernel time the GPU now, in every CPU round (row 1 in two steps: the CPU cell to "
         "the 8-neighbour CPU, 0.192 ms, then the tiles' 0.044 ms and GPU 4.4x; the card reads true)",
         n["small_tiles_ms"] < n["small_cpu8_ms"] and min(rng["small_tiles_x_range"]) > 1
         and card_x(d["small_tiles_x"]) == "4.4×" and card_ms(n["small_tiles_ms"]) == "0.044 ms"
         and card_ms(n["small_cpu8_ms"]) == "0.192 ms" and card_row_ok(n["small_cpu8_ms"], n["small_tiles_ms"])),
        ("r14 ... with the copy to the card, still the CPU (0.665 vs 0.192 ms: the CPU by 3.5x)",
         n["small_tiles_e2e_ms"] > n["small_cpu8_ms"] and n["small_tiles_e2e_ms"] > n["small_cpu_ms"]
         and card_ms(n["small_tiles_e2e_ms"]) == "0.665 ms" and card_x(d["small_e2e_cpu_x"]) == "3.5×"),
        ("r14 card footnote, with the copies (rounded, never spoken): small blob 0.665 ms (CPU 0.192), huge blob "
         "44.3 ms, frame 54.8 ms",
         (card_ms_note(n["small_tiles_e2e_ms"]), card_ms_note(n["small_cpu8_ms"]), card_ms_note(n["big_tiles_e2e_ms"]),
          card_ms_note(n["frame_e2e_ms"])) == ("0.665 ms", "0.192 ms", "44.3 ms", "54.8 ms")),
        ("r14 every drone, every frame (caption 3: with copies 1.26 ms, the worst frame's median; far under 33)",
         n["drone_e2e_max_ms"] < d["budget_ms"] and card_ms(n["drone_e2e_max_ms"]) == "1.26 ms"
         and n["drone_e2e_ms"] <= n["drone_e2e_max_ms"] and n["drone_ok"] == 1 and n["drone_first_version"] == 1),
        ("r14 the drones are not called a GPU win: with the copies the CPU's 0.679 ms beats the GPU's 0.977 "
         "(kernel time 0.090 ms, shown nowhere)",
         n["drone_e2e_ms"] > n["drone_cpu_ms"] and math.isclose(d["drone_x"], n["drone_x_json"])),
        ("N2 (not shown): act 2's seeds kernel already beats the 8-neighbour CPU with depth on the small blob, "
         "1.12x, in every CPU round",
         round(d["small_ch05_x"], 2) == 1.12 and min(n["small_ch05_x_range"]) > 1),
    ]


def _fmt(v) -> str:
    if isinstance(v, tuple):
        return f"{len(v):,} items: " + ", ".join(f"{x:,}" if isinstance(x, (int, float)) else str(x)
                                                 for x in v[:3]) + (", ..." if len(v) > 3 else "")
    return f"{v:>14,.3f}"


def main() -> int:
    for k, v in load().items():
        print(f"{k:28s} {_fmt(v.value)}   {v.source}")
    print()
    for k, v in derived().items():
        print(f"{k:28s} {_fmt(v)}")
    print()
    for k, v in floors().items():
        print(f"{k:32s} {v if isinstance(v, (bool, tuple)) else _fmt(v)}")
    print()
    ok = True
    for text, good in facts():
        ok &= bool(good)
        print(("ok    " if good else "FALSE ") + text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
