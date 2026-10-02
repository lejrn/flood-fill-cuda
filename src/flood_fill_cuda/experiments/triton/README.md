# Triton GPU Flood Fill (single blob)

Flood fill of one connected red blob, parallelized on the GPU with
[Triton](https://github.com/triton-lang/triton) kernels. No Numba, and no
PyTorch either: the shared runtime in `flood_fill_cuda.triton_twins.runtime`
plugs CuPy's device and stream management into Triton's CUDA driver, so the
kernels launch directly on CuPy-allocated memory. The Triton twins of the
chapters use the same runtime.

This experiment is Triton-only and has no Numba counterpart, so it has no
twin and no Numba-vs-Triton comparison of its own.

## Run

From the repo root, as a module or as a script:

```bash
uv run python -m flood_fill_cuda.experiments.triton.main
uv run python src/flood_fill_cuda/experiments/triton/main.py --size 8192 --shape spiral
uv run python src/flood_fill_cuda/experiments/triton/main.py --mode all --repeats 5
```

The script generates a single big blob (red = 1) on a white background
(0), seeds the fill at one pixel, converts every red pixel 4-connected to it
into blue (2) on the GPU, and reports **converted pixels per second**. The
result is verified pixel-for-pixel against OpenCV's CPU `floodFill`
(4-connectivity), which also serves as the CPU throughput baseline (best of
3). It exits with status 1 on a mismatch or when a mode hits
`--max-launches` before converging. Before/after PNGs land in `output/`
(`*.png` is gitignored).

Timing: one warm-up fill per mode (the Triton compile), then `--repeats`
timed fills, each bracketed by `time.perf_counter()` and a device
synchronize. The table shows the best and the mean; MPx/s uses the best.

## Cell states

The image is a row-major `(H, W)` uint8 grid of states, not the chapters'
`(width, height, 3)` RGB layout.

| value | meaning              | color |
|-------|----------------------|-------|
| 0     | background (wall)    | white |
| 1     | fillable             | red   |
| 2     | filled               | blue  |

## Kernels (`--mode`)

`naive` and `tile` share one host loop (`run_flood_fill`): launch
`flood_fill_step`, read a global `changed` flag back, and relaunch until a
launch converts no pixel. A pixel only ever goes red to blue, so races
between programs are benign, and a launch that changes nothing proves
convergence.

- **`naive`**: textbook level-synchronous parallel BFS. Each launch makes
  one sweep over every 2D tile, so the frontier moves about one pixel per
  launch (a sweep can also see a neighbouring tile's fresh stores through
  L2). One launch per BFS level: the clean reference point.
- **`tile`**: each program owns a 2D tile (`--block` side) and re-sweeps it
  until the tile stops changing, so one launch advances the frontier by a
  whole tile or more (`.cg` loads let programs see sibling tiles' stores
  through L2 mid-launch).
- **`scan`** (default): rows and columns are flooded with a *segmented
  max-scan* (`tl.associative_scan`, walls reset the running max, run
  forward and backward), so one pass fills entire horizontal or vertical
  runs whatever their length. The host loop (`run_flood_fill_scan`)
  alternates a row pass and a column pass. Each pass first compacts its
  dirty lines into a list with `gather_dirty` (a block-wide `tl.cumsum`
  plus one atomic per program), reads the count back, and launches
  `scan_line_step` with one program per dirty line. A line that gains blue
  marks the crossing lines dirty for the other pass. The fill has converged
  when neither pass finds a dirty line. A column pass reads its line with
  stride `W` (one program per column, uncoalesced loads). Convergence takes
  roughly one round per bend in the blob rather than one launch per pixel
  of distance.

## Blob shapes (`--shape`)

- `amoeba` (default): wavy star-shaped blob, about a third of the image red.
- `spiral`: Archimedean spiral arm, compact on screen but with a very long
  geodesic, the worst case for BFS-style propagation.
- `disc`: filled circle covering about 64% of the image, the friendliest
  case.

## Files

- `main.py`: benchmark CLI (throughput, verification, PNGs)
- `kernels.py`: Triton kernels and the host convergence loops
- `blobs.py`: synthetic single-blob generators
- `cupy_bridge.py`: compatibility shim that re-exports the shared runtime's
  bridge (`t`, `install_cupy_driver`, ...), so old imports keep working
