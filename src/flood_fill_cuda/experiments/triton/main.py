#!/usr/bin/env python3
"""GPU flood fill of a single big blob with Triton: pixels/second benchmark.

Fills one connected red blob starting from a seed pixel, turning it blue with
parallel Triton kernels (no torch: Triton runs on CuPy memory through the
shared runtime, flood_fill_cuda.triton_twins.runtime).  Reports converted
pixels/second, verifies the result against OpenCV's CPU floodFill, and saves
before/after PNGs.

Usage (from the repo root), as a module or as a script:

    uv run python -m flood_fill_cuda.experiments.triton.main
    uv run python src/flood_fill_cuda/experiments/triton/main.py --size 8192 --shape spiral
    uv run python src/flood_fill_cuda/experiments/triton/main.py --mode all --repeats 5
"""

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    # Run as a script: make the flood_fill_cuda package importable even
    # without the editable install (src/ is three levels up).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cupy as cp
import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from flood_fill_cuda.experiments.triton.blobs import GENERATORS, make_blob
from flood_fill_cuda.experiments.triton.kernels import (
    BLUE, RED, run_flood_fill, run_flood_fill_scan,
)
from flood_fill_cuda.triton_twins.runtime import install

# background -> white, red -> red, blue -> blue
PALETTE = [255, 255, 255, 220, 60, 54, 38, 110, 210]


def save_png(arr: np.ndarray, path: Path) -> None:
    img = Image.fromarray(arr, mode="P")
    img.putpalette(PALETTE + [0] * (768 - len(PALETTE)))
    img.save(path)


def bench_gpu(pristine, changed, seed, *, mode, repeats, block, num_warps, max_launches):
    """Warm up once (JIT compile), then time `repeats` full fills."""
    times, launches, converged = [], 0, True
    grid_dev = None
    for i in range(repeats + 1):
        grid_dev = pristine.copy()
        grid_dev[seed] = BLUE
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        if mode == "scan":
            launches, converged = run_flood_fill_scan(
                grid_dev, changed, max_rounds=max_launches
            )
        else:
            launches, converged = run_flood_fill(
                grid_dev,
                changed,
                block=block,
                num_warps=num_warps,
                converge=(mode == "tile"),
                max_launches=max_launches,
            )
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        if i > 0:  # run 0 is the compile/warm-up run
            times.append(dt)
    return grid_dev, times, launches, converged


def bench_cpu(initial: np.ndarray, seed, repeats: int):
    """OpenCV floodFill baseline (single-threaded scanline fill)."""
    ref, best = None, float("inf")
    for _ in range(repeats):
        ref = initial.copy()
        t0 = time.perf_counter()
        cv2.floodFill(ref, None, (seed[1], seed[0]), BLUE)
        best = min(best, time.perf_counter() - t0)
    return ref, best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=4096, help="image side in pixels")
    p.add_argument("--shape", choices=sorted(GENERATORS), default="amoeba")
    p.add_argument(
        "--mode",
        choices=["scan", "tile", "naive", "all"],
        default="scan",
        help="scan: fill whole rows/columns per pass via segmented scans; "
        "tile: sweep each tile to convergence per launch; "
        "naive: one BFS level per launch (slow on long blobs)",
    )
    p.add_argument("--block", type=int, default=32, help="tile side (power of 2)")
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3, help="timed runs after warm-up")
    p.add_argument("--max-launches", type=int, default=200_000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--no-verify", action="store_true", help="skip OpenCV comparison")
    p.add_argument("--no-images", action="store_true", help="skip PNG output")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "output")
    args = p.parse_args()

    console = Console()
    install()  # idempotent: importing the runtime already installed it

    initial, seed = make_blob(args.shape, args.size, args.rng_seed)
    blob_px = int(np.count_nonzero(initial == RED))
    total_px = args.size * args.size
    console.print(
        f"[bold]{args.shape}[/bold] blob: {args.size}x{args.size} image "
        f"({total_px / 1e6:.1f} MPx), {blob_px / 1e6:.2f} MPx red, "
        f"seed at {seed}"
    )

    pristine = cp.asarray(initial)
    changed = cp.zeros(1, dtype=cp.int32)

    # CPU reference: correctness oracle + throughput baseline
    ref, cpu_time = (None, None)
    if not args.no_verify:
        ref, cpu_time = bench_cpu(initial, seed, repeats=3)
        console.print(
            f"OpenCV floodFill (CPU): {cpu_time * 1e3:.1f} ms "
            f"-> {blob_px / cpu_time / 1e6:.1f} MPx/s"
        )

    modes = ["scan", "tile", "naive"] if args.mode == "all" else [args.mode]
    if not args.no_images:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        save_png(initial, args.out_dir / f"{args.shape}_{args.size}_initial.png")

    table = Table(title=f"Triton flood fill: {args.shape} {args.size}x{args.size}")
    for col in ("mode", "filled MPx", "launches", "best ms", "mean ms", "MPx/s", "vs CPU"):
        table.add_column(col, justify="right")

    exit_code = 0
    for name in modes:
        result_dev, times, launches, converged = bench_gpu(
            pristine,
            changed,
            seed,
            mode=name,
            repeats=args.repeats,
            block=args.block,
            num_warps=args.num_warps,
            max_launches=args.max_launches,
        )
        result = cp.asnumpy(result_dev)
        filled = int(np.count_nonzero(result == BLUE))
        best, mean = min(times), sum(times) / len(times)
        pps = filled / best

        status = ""
        if not converged:
            status = "[red]hit --max-launches before converging[/red]"
            exit_code = 1
        elif not args.no_verify:
            if np.array_equal(result, ref):
                status = "[green]matches OpenCV floodFill exactly[/green]"
            else:
                status = "[red]MISMATCH vs OpenCV floodFill[/red]"
                exit_code = 1
        console.print(f"{name}: {status}")

        table.add_row(
            name,
            f"{filled / 1e6:.2f}",
            str(launches),
            f"{best * 1e3:.1f}",
            f"{mean * 1e3:.1f}",
            f"{pps / 1e6:,.1f}",
            f"{pps * cpu_time / blob_px:.2f}x" if cpu_time else "-",
        )
        if not args.no_images:
            save_png(result, args.out_dir / f"{args.shape}_{args.size}_filled_{name}.png")

    console.print(table)
    if not args.no_images:
        console.print(f"images written to {args.out_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
