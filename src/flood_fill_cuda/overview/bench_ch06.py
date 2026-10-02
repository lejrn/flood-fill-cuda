"""Chapter 6 on the grand table: the run-table recolor on every overview row.

The overview benchmark (overview/bench.py) predates Chapter 6, so its
committed table has no ch06 column. This script fills that column, and
only that column, on the SAME 17 rows, with the ch06 chapter's own
protocol (clock spin-up, interleaved rounds over both contracts, CUDA
events, kernel-only). It never touches the overview file: the numbers
every README and the video already quote stay byte-identical.

It is a separate session from the overview run, so the njit / pure cells
it will be compared with were timed on a different day and at different
clocks. The per-row SM clock is recorded so a reader can judge that;
ch06 vs njit margins are 10-100x on most rows, which no clock swing
closes.

Crosscheck, measured not assumed: after the last round the painted image
is copied back and (a) the painted pixel count must equal the scene's
red count, (b) no red pixel may remain, (c) on one- and two-blob rows the
kernel's blob count must be 1 or 2, and on N-blob rows it must equal the
overview's committed n_blobs for that row.

Output: results/overview/benchmark_results/ch06_overview_<stamp>.json
(+ a flat CSV). The `ch06_` prefix is deliberate: overview/build.py and
ch06's figures.py glob `overview_*.json` and must keep picking the
overview file.

Run:  PYTHONUNBUFFERED=1 uv run python -m flood_fill_cuda.overview.bench_ch06
Budget ~5-8 min (JIT, then 8 s spin + 9 rounds x 2 contracts per row).
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import csv
import gc
import glob
import json
import traceback
from datetime import datetime, timezone

import numpy as np
from numba import cuda

from .bench import ROWS, RESULTS_DIR
from ..chapters.ch06_gpu_nblob_runs.recolor import (
    RunRecolor, CONTRACTS, DEFAULT_GRID, PHASE_BLOCKS, PACK_ROW_BLOCKS,
    _warmup)
from ..chapters.ch06_gpu_nblob_runs.benchmarks.benchmark import (
    _bench_ch06, _spin_up, _clocks, _host_run_count, ROUNDS, SPIN_SECONDS)

COLS = [
    ("ch06_rgb", "ch06 · runs", "rgb (incl. pack)"),
    ("ch06_mask", "ch06 · runs", "packed mask"),
]
EXPECTED_BLOBS = {"one": 1, "two": 2, "n": None}


def _red_count(img):
    return int(np.count_nonzero(
        (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)))


def _dev_name(device):
    name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    return name.rstrip("\x00").strip()


def _overview_rows():
    """The committed overview rows, keyed by row name, for the crosscheck."""
    paths = sorted(p for p in glob.glob(os.path.join(RESULTS_DIR, "overview_*.json"))
                   if "ch06" not in os.path.basename(p))
    if not paths:
        return None, {}
    with open(paths[-1]) as f:
        data = json.load(f)
    return os.path.basename(paths[-1]), {r["row"]: r for r in data["rows"]}


def _skip_reason(exc):
    msg = str(exc).upper()
    if "MEMORY" in msg or isinstance(exc, MemoryError):
        return "oom"
    if "OVERFLOW" in msg:
        return "overflow"
    return f"error:{type(exc).__name__}"


def bench_row(key, family, note, build, ref):
    ctx = build()
    img = ctx["img"]
    width, height = int(img.shape[0]), int(img.shape[1])
    red_px = _red_count(img)
    capacity = max(8192, int(_host_run_count(img) * 1.05))
    row = {"row": key, "family": family, "note": note,
           "width": width, "height": height, "kind": ctx["kind"],
           "red_px": red_px, "run_capacity": capacity}

    engine = pristine = dev = None
    try:
        engine = RunRecolor(width, height, run_capacity=capacity)
        pristine = cuda.to_device(img)
        dev = cuda.to_device(img)
        engine.pack(dev)
        cuda.synchronize()
        _spin_up(engine, dev)
        clocks = _clocks()
        stats = _bench_ch06(engine, dev, pristine)
        info = stats.pop("info")
        out = dev.copy_to_host()
    except Exception as exc:                      # typed skip, never a crash
        reason = _skip_reason(exc)
        if reason.startswith("error:"):
            traceback.print_exc()
        row["cells"] = {c: {"skip": reason} for c, _, _ in COLS}
        row["crosscheck"] = "n/a"
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        painted = int(np.count_nonzero(np.any(out != img, axis=2)))
        still_red = _red_count(out)
        expected = EXPECTED_BLOBS[ctx["kind"]]
        if expected is None:
            expected = (ref or {}).get("n_blobs")
        ok = (painted == red_px and still_red == 0
              and (expected is None or info["n_blobs"] == expected))
        ref_filled = None
        if ref:
            fills = {c["filled"] for c in ref["cells"].values()
                     if c.get("skip") is None and "filled" in c}
            ref_filled = fills.pop() if len(fills) == 1 else None
        if ref_filled is not None and ref_filled != painted:
            ok = False
        row["cells"] = {}
        for col, _, _ in COLS:
            c = col[len("ch06_"):]
            s = stats[c]
            row["cells"][col] = {
                "ms": s["median_ms"], "ms_min": s["min_ms"],
                "ms_max": s["max_ms"], "filled": painted,
                "phase_ms": s["phase_ms"],
                "label_only_ms": s["label_only_ms"], "skip": None}
        row.update({
            "n_runs": info["n_runs"], "n_blobs": info["n_blobs"],
            "union_attempts": info["union_attempts"],
            "still_red": still_red, "expected_blobs": expected,
            "overview_filled": ref_filled,
            "clocks_during_run": clocks,
            "crosscheck": "OK" if ok else "MISMATCH"})
        del out
    finally:
        del engine, pristine, dev, ctx, img
        gc.collect()
        try:
            cuda.current_context().deallocations.clear()
        except AttributeError:
            pass
    return row


def _fmt(cell):
    return cell["skip"] if cell.get("skip") else f"{cell['ms']:.3f}"


def main():
    device = cuda.get_current_device()
    dev_name = _dev_name(device)
    print(f"Device: {dev_name} ({int(device.MULTIPROCESSOR_COUNT)} SMs)")
    print(f"Idle clocks: {_clocks()}")
    print("Warming up the ch06 kernels...")
    _warmup()

    source, ref_rows = _overview_rows()
    missing = [k for k, *_ in ROWS if k not in ref_rows]
    if missing:
        print(f"note: rows without an overview reference: {missing}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"ch06_overview_{stamp}.json")
    payload = {
        "schema": "ch06_overview_column",
        "device": dev_name,
        "sm_count": int(device.MULTIPROCESSOR_COUNT),
        "tpb": int(DEFAULT_GRID[1]),
        "grid": list(DEFAULT_GRID),
        "phase_blocks": PHASE_BLOCKS,
        "pack_row_blocks": PACK_ROW_BLOCKS,
        "rounds": ROUNDS,
        "spin_seconds": SPIN_SECONDS,
        "idle_clocks": _clocks(),
        "overview_source": source,
        "columns": [{"key": k, "group": g, "label": l} for k, g, l in COLS],
        "experiment": (
            "ch06 run-table recolor on the grand table's 17 rows, both "
            "contracts, timed with the ch06 chapter's protocol (8 s clock "
            "spin, 9 interleaved rounds, CUDA events, kernel-only). A "
            "separate session from overview_*.json: compare against its "
            "njit / pure cells knowing the clocks differ; the SM clock "
            "during each row is recorded. filled = painted pixels counted "
            "on the host after the last round; crosscheck also requires "
            "zero red pixels left, n_blobs == 1 / 2 on one- / two-blob "
            "rows and == the overview's n_blobs on N-blob rows, and "
            "filled == the overview's filled for the row."),
        "rows": [],
    }

    for key, family, note, _est, build in ROWS:
        print(f"\n{key}  ({note})", flush=True)
        row = bench_row(key, family, note, build, ref_rows.get(key))
        payload["rows"].append(row)
        with open(json_path, "w") as f:               # crash-safe per row
            json.dump(payload, f, indent=2)
        cells = row["cells"]
        extra = ""
        if row.get("n_runs") is not None:
            extra = (f"  runs={row['n_runs']:,} blobs={row['n_blobs']:,} "
                     f"sm={row['clocks_during_run'].get('sm_mhz', '?')} MHz")
        print(f"   [{row['crosscheck']}] rgb {_fmt(cells['ch06_rgb'])} ms | "
              f"mask {_fmt(cells['ch06_mask'])} ms{extra}", flush=True)

    csv_path = json_path.replace(".json", "_rows.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "kind", "width", "height", "red_px", "n_runs",
                    "n_blobs", "ch06_rgb_ms", "ch06_mask_ms",
                    "ch06_label_ms", "sm_mhz", "crosscheck"])
        for r in payload["rows"]:
            c = r["cells"]
            w.writerow([
                r["row"], r["kind"], r["width"], r["height"], r["red_px"],
                r.get("n_runs", ""), r.get("n_blobs", ""),
                _fmt(c["ch06_rgb"]), _fmt(c["ch06_mask"]),
                (f"{c['ch06_mask']['label_only_ms']:.4f}"
                 if c["ch06_mask"].get("skip") is None else ""),
                r.get("clocks_during_run", {}).get("sm_mhz", ""),
                r["crosscheck"]])

    bad = [r["row"] for r in payload["rows"] if r["crosscheck"] != "OK"]
    print(f"\nResults written to {json_path}")
    print(f"                   {csv_path}")
    if bad:
        print(f"rows not OK: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
