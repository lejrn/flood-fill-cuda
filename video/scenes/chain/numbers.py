"""Every number the chain cut (v2, two acts) shows or says, from committed JSON.

Each value names its file and key; files are pinned by name, so a later
benchmark run cannot change the video silently. Sessions:

- OV  overview_20260724T234423Z.json (Jul 24): the CPU baselines, ch01-ch05
      cells and the estimates on the 81 Mpx frame (png_blobs).
- C6  ch06_overview_20260927T013331Z.json (Sep 27): ch06 on the same rows.
- CH03 multi_block_20260720T012243Z.json (Jul 20): act 1 (size), one
      session for 1 block, 2 blocks, 48 blocks, 8 ways and the snake.
- R61 runs_20260725T161448Z.json (Jul 25): ch06 on the 81 Mpx frame.
- DR  drones_20261014T023648Z.json (Oct 14): CPU vs GPU on the 240 drone
      frames, one session (the host CPU ran ~3x slower than on Jul 24).

No manim import: `uv run python -m scenes.chain.numbers` prints the table
and exits 1 if a stated fact does not hold.
"""
from __future__ import annotations

import json
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
CH03 = R / "ch03_gpu_1blob_nblock" / "benchmark_results" / "multi_block_20260720T012243Z.json"
R61 = R / "ch06_gpu_nblob_runs" / "benchmark_results" / "runs_20260725T161448Z.json"
VIDEO = Path(__file__).resolve().parents[2]
DRONE_META = VIDEO / "assets" / "drones_labels" / "meta.json"

# Not in any benchmark JSON: the host CPU's core count, from Intel's spec
# for the model lscpu reports ("13th Gen Intel(R) Core(TM) i9-13900H":
# 6 performance + 8 efficiency cores, 20 threads). WSL exposes 16 of them.
CPU_CORES = 14
CPU_CORES_SOURCE = "Intel spec for the i9-13900H (lscpu model name), not benchmark JSON"


@dataclass(frozen=True)
class Num:
    value: float
    source: str


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _row(d: dict, name: str) -> dict:
    return next(r for r in d["rows"] if r.get("row", r.get("key")) == name)


def _scene(d: dict, name: str) -> dict:
    return next(s for s in d["scenes"] if s["scene"] == name)


@lru_cache(maxsize=1)
def load() -> dict[str, Num]:
    n: dict[str, Num] = {}
    ov, c6, dr = _load(OV), _load(C6), _load(DR)

    def put(key, value, source):
        n[key] = Num(float(value), source)

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
    sn = _scene(d3, "serpentine_256")
    f = CH03.name + ":serpentine_256"
    put("snake_1block_ms", sn["v2_kernel_ms"], f + ".v2_kernel_ms")
    put("snake_2blocks_ms", sn["dual_global_kernel_ms"], f + ".dual_global_kernel_ms")
    put("snake_48blocks_ms", sn["multi_kernel_ms"], f + ".multi_kernel_ms")
    put("snake_rings", sn["levels"], f + ".levels")
    put("snake_util_pct", sn["multi_thread_util_pct"], f + ".multi_thread_util_pct")
    s1 = _scene(_load(CH01), "sq_6000_center")
    put("spill_pct", s1["spill_pct"], f"{CH01.name}:sq_6000_center.spill_pct")

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
    return n


def derived() -> dict[str, float]:
    n = {k: v.value for k, v in load().items()}
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
    }


def facts() -> list[tuple[str, bool]]:
    """Every spoken sentence that rests on a number."""
    n = {k: v.value for k, v in load().items()}
    d = derived()
    return [
        ("i0 thirty-three milliseconds", round(d["budget_ms"]) == 33),
        ("i1 on frames this small, compiled CPU code keeps up (drone frames, worst frame)",
         n["drone_cpu_max_ms"] < d["budget_ms"] and n["drone_ok"] == 1),
        ("i1 eighty-one megapixels", round(n["n_pixels"] / 1e6) == 81),
        ("i1 the CPU needs one point three seconds", round(n["njit_ms"] / 1000, 1) == 1.3),
        ("r01 twenty-four SMs", n["sm"] == 24),
        ("r02 two hundred fifty-six threads", n["small_tpb"] == 256),
        ("r02 on one small blob, the CPU still wins", n["small_cpu_ms"] < n["small_gpu_ms"]),
        ("r03 one block is over twice the CPU", 2.0 < d["big_1block_x"] < 3.0),
        ("r03 one SM of twenty-four", n["sm"] == 24),
        ("r04 twice as fast", 1.9 <= d["big_2v1_x"] < 2.5),
        ("r04 twenty-two SMs stay dark", n["sm"] - 2 == 22),
        ("r05 forty-eight blocks", n["big_blocks"] == 48),
        ("r05 eighteen times the CPU", int(d["big_48_x"]) == 18),
        ("r05 on a thin snake they lose to one", n["snake_48blocks_ms"] > n["snake_1block_ms"]),
        ("r05 stops twice, two microseconds each", 1.5 <= d["us_per_stop"] < 2.5),
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
        ("r10 caption 25x fewer runs", round(d["runs_x"]) == 25),
        ("r11 many at once: the GPU (count row)", d["count_x"] > 1),
        ("r11 every drone, every frame (GPU worst frame under budget)", n["drone_gpu_max_ms"] < d["budget_ms"]),
    ]


def main() -> int:
    for k, v in load().items():
        print(f"{k:20s} {v.value:>14,.3f}   {v.source}")
    print()
    for k, v in derived().items():
        print(f"{k:20s} {v:>14,.3f}")
    print()
    ok = True
    for text, good in facts():
        ok &= bool(good)
        print(("ok    " if good else "FALSE ") + text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
