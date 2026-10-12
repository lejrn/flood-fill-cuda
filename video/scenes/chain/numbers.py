"""Every number the chain cut shows or says, read from committed JSON.

Each value names its file and key. Files are pinned by name (not "the
newest"), so a later benchmark run cannot change the video silently.
Beats c03-c05 read ONE ch03 session (`multi_block_20260720T012243Z`): it
holds the 4-connected and 8-connected 48 x 256 runs on the 64 Mpx square
and the snake, so the 8-connectivity gain is not inflated by another
session's CPU time.

No manim import: `uv run python -m scenes.chain.numbers` prints the
table for auditing and exits 1 if a stated fact does not hold.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from scenes.panes import data

R = data.RESULTS
CH01 = R / "ch01_gpu_1blob_1block" / "benchmark_results" / "single_block_shared_20260719T041848Z.json"
CH02 = R / "ch02_gpu_1blob_2block" / "benchmark_results" / "dual_block_20260719T163033Z.json"
CH03 = R / "ch03_gpu_1blob_nblock" / "benchmark_results" / "multi_block_20260720T012243Z.json"
CH04 = R / "ch04_gpu_2blob_nblock" / "benchmark_results" / "dual_blob_20260720T140944Z.json"
OV = R / "overview" / "benchmark_results" / "overview_20260724T234423Z.json"


@dataclass(frozen=True)
class Num:
    value: float
    source: str          # "file:key"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _scene(d: dict, name: str) -> dict:
    return next(s for s in d["scenes"] if s["scene"] == name)


@lru_cache(maxsize=1)
def load() -> dict[str, Num]:
    n: dict[str, Num] = {}
    h = data.headline()
    for key in ("pure_ms", "njit_ms", "ch05_ms", "ch06_mask_ms", "ch06_rgb_ms"):
        n[key] = Num(getattr(h, key), h.sources[key])
    n["n_blobs"] = Num(h.n_blobs, "runs:input_blobs.n_blobs")
    n["red_px"] = Num(h.red_px, "runs:input_blobs.red_px")
    n["n_runs"] = Num(h.n_runs, "runs:input_blobs.n_runs")
    n["n_pixels"] = Num(h.n_pixels, "runs:input_blobs.n_pixels")
    n["sm"] = Num(data.load_bench().sm_count, f"{OV.name}:sm_count")

    s = _scene(_load(CH01), "sq_6000_center")
    f = CH01.name + ":sq_6000_center"
    n["c02_gpu_ms"] = Num(s["spill_kernel_ms"], f + ".spill_kernel_ms")
    n["c02_njit_ms"] = Num(s["njit_ms"], f + ".njit_ms")
    n["c02_ring"] = Num(s["ring_capacity"], f + ".ring_capacity")
    n["c02_mpx"] = Num(36.0, f + " (6000 x 6000)")

    s = _scene(_load(CH02), "sq_6000_center")
    f = CH02.name + ":sq_6000_center"
    n["c03_two_ms"] = Num(s["global_kernel_ms"], f + ".global_kernel_ms")
    n["c03_one_ms"] = Num(s["v2_kernel_ms"], f + ".v2_kernel_ms")

    d3 = _load(CH03)
    s = _scene(d3, "sq_8000_center")
    f = CH03.name + ":sq_8000_center"
    n["c03_blocks"] = Num(s["multi_blocks"], f + ".multi_blocks")
    n["c03_tpb"] = Num(s["multi_tpb"], f + ".multi_tpb")
    n["c03_n_ms"] = Num(s["multi_kernel_ms"], f + ".multi_kernel_ms")
    n["c05_ms"] = Num(s["conn8_kernel_ms"], f + ".conn8_kernel_ms")
    n["c05_blocks"] = Num(s["conn8_blocks"], f + ".conn8_blocks")
    n["c0305_njit_ms"] = Num(s["njit_ms"], f + ".njit_ms")
    n["c05_levels4"] = Num(s["levels"], f + ".levels")
    n["c05_levels8"] = Num(s["conn8_levels"], f + ".conn8_levels")
    n["c05_mpx"] = Num(64.0, f + " (8000 x 8000)")
    s = _scene(d3, "serpentine_256")
    f = CH03.name + ":serpentine_256"
    n["c04_n_ms"] = Num(s["multi_kernel_ms"], f + ".multi_kernel_ms")
    n["c04_one_ms"] = Num(s["v2_kernel_ms"], f + ".v2_kernel_ms")
    n["c04_levels"] = Num(s["levels"], f + ".levels")
    n["c04_util_pct"] = Num(s["multi_thread_util_pct"], f + ".multi_thread_util_pct")
    n["c04_blocks"] = Num(s["multi_blocks"], f + ".multi_blocks")
    n["c04_tpb"] = Num(s["multi_tpb"], f + ".multi_tpb")

    png = next(r for r in _load(OV)["rows"] if r["row"] == "png_blobs")["cells"]
    for key, col in (("c06_ms", "ch03_conn8"), ("c07_ms", "ch04_multi")):
        c = png[col]
        assert c.get("est") is True, f"{col} is expected to be an estimate"
        n[key] = Num(c["ms"], f"{OV.name}:png_blobs.{col}.ms (est, {c['calls']} launches)")
        n[key.replace("_ms", "_calls")] = Num(c["calls"], f"{OV.name}:png_blobs.{col}.calls")

    sc = _load(CH04)["scenes"]
    lo = min(s["speedup_multi_vs_seq"] for s in sc)
    hi = max(s["speedup_multi_vs_seq"] for s in sc)
    n["c07_lo"] = Num(lo, CH04.name + ":scenes[].speedup_multi_vs_seq (min)")
    n["c07_hi"] = Num(hi, CH04.name + ":scenes[].speedup_multi_vs_seq (max)")
    return n


# ---- derived (the only arithmetic, all here) ----------------------------

def derived() -> dict[str, float]:
    n = {k: v.value for k, v in load().items()}
    return {
        "c02_x": n["c02_njit_ms"] / n["c02_gpu_ms"],             # 2.03
        "c02_sm_pct": 100 / n["sm"],                              # 4.17
        "c03_two_x": n["c03_one_ms"] / n["c03_two_ms"],          # 2.10
        "c03_n_x": n["c0305_njit_ms"] / n["c03_n_ms"],           # 18.75
        "c04_threads": n["c04_blocks"] * n["c04_tpb"],           # 12,288
        "c04_gates": 2 * n["c04_levels"],                        # 2 grid.sync per level
        "c04_us_per_gate": 1000 * n["c04_n_ms"] / (2 * n["c04_levels"]),
        "c05_x": n["c0305_njit_ms"] / n["c05_ms"],               # 25.8
        "c05_gain": n["c03_n_ms"] / n["c05_ms"],                 # 1.37
        "c08_x": n["njit_ms"] / n["ch05_ms"],
        "c09_x": n["ch05_ms"] / n["ch06_mask_ms"],               # 40.2
        "c09_runs_x": n["red_px"] / n["n_runs"],                 # 24.95
        "c10_x": n["pure_ms"] / n["ch06_mask_ms"],               # 16,551
    }


def facts() -> list[tuple[str, bool]]:
    """Every sentence of the narration that rests on a number."""
    n = {k: v.value for k, v in load().items()}
    d = derived()
    return [
        ("c00 two and a half thousand blobs", 2400 <= n["n_blobs"] <= 2600),
        ("c00 eighty-one million pixels", round(n["n_pixels"] / 1e6) == 81),
        ("c01 one point three seconds (@njit)", round(n["njit_ms"] / 1000, 1) == 1.3),
        ("c02 twice the CPU", 1.95 <= d["c02_x"] < 2.5),
        ("c02 one room of twenty-four", n["sm"] == 24),
        ("c03 two teams, twice as fast", 1.95 <= d["c03_two_x"] < 2.5),
        ("c03 forty-eight teams", n["c03_blocks"] == 48 and n["c03_tpb"] == 256),
        ("c03 eighteen times the CPU", int(d["c03_n_x"]) == 18),
        ("c03 on the snake forty-eight teams lose to one", n["c04_n_ms"] > n["c04_one_ms"] and n["c04_blocks"] == 48),
        ("c04 twelve thousand wait", 12000 <= d["c04_threads"] - 1 < 13000 and n["c04_util_pct"] < 0.01),
        ("c04 sixty-five thousand times", 65000 <= d["c04_gates"] < 66000),
        ("c04 about 2 us per gate", 1.5 <= d["c04_us_per_gate"] < 2.5),
        ("c05 half the rings (square)", abs(n["c05_levels8"] / n["c05_levels4"] - 0.5) < 0.01),
        ("c05 twenty-five times the CPU", int(d["c05_x"]) == 25),
        ("c06 it loses to the CPU (estimate)", n["c06_ms"] > n["njit_ms"]),
        ("c06 one launch per blob", n["c06_calls"] == n["n_blobs"]),
        ("c07 half the launches, no faster than the CPU", n["c07_calls"] * 2 == n["n_blobs"] and n["c07_ms"] > n["njit_ms"]),
        ("c07 up to 2x", 1.95 <= n["c07_hi"] < 2.5),
        ("c08 fifty-eight milliseconds", int(n["ch05_ms"]) == 58),
        ("c08 still too slow for one frame (33 ms)", n["ch05_ms"] > 1000 / 30),
        ("c09 thirteen million pixels", int(n["red_px"] / 1e6) == 13),
        ("c09 half a million spans", 450_000 <= n["n_runs"] < 600_000),
        ("c09 one and a half milliseconds", round(n["ch06_mask_ms"], 1) == 1.5),
        ("c10 twenty-four seconds", int(n["pure_ms"] / 1000) == 24),
        ("c10 16,000x (floored to two figures)", int(d["c10_x"] / 1000) == 16),
    ]


def main() -> int:
    for k, v in load().items():
        print(f"{k:16s} {v.value:>14,.3f}   {v.source}")
    print()
    for k, v in derived().items():
        print(f"{k:16s} {v:>14,.3f}")
    print()
    ok = True
    for text, good in facts():
        ok &= good
        print(("ok    " if good else "FALSE ") + text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
