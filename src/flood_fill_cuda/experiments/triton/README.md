# Triton GPU Flood Fill (single blob)

BFS-based flood fill of one connected red blob, parallelized on the GPU with
[Triton](https://github.com/triton-lang/triton) kernels — no Numba, and no
PyTorch either: `cupy_bridge.py` plugs CuPy's device/stream management into
Triton's CUDA driver, so kernels launch directly on CuPy-allocated memory.

## Run

```bash
uv run python src/flood_fill_cuda/experiments/triton/main.py
uv run python src/flood_fill_cuda/experiments/triton/main.py --size 8192 --shape spiral
uv run python src/flood_fill_cuda/experiments/triton/main.py --mode all --repeats 5
```

The script generates a single big blob (red = 1) on a white background
(0), seeds the fill at one pixel, converts every reachable red pixel to blue
(2) on the GPU, and reports **converted pixels per second**. The result is
verified pixel-for-pixel against OpenCV's CPU `floodFill`, which also serves
as the CPU throughput baseline. Before/after PNGs land in `output/`.

## Cell states

| value | meaning              | color |
|-------|----------------------|-------|
| 0     | background (wall)    | white |
| 1     | fillable             | red   |
| 2     | filled               | blue  |

## Kernels (`--mode`)

All three run the same host loop: launch, then relaunch until a launch
converts no pixel (a global `changed` flag). Because a pixel only ever goes
red → blue, races between programs are benign, and a zero-change launch
proves convergence.

- **`naive`** — textbook level-synchronous parallel BFS: each launch advances
  the frontier exactly one pixel. One launch per BFS level; the clean
  reference point.
- **`tile`** — each program owns a 2D tile and re-sweeps it until the tile
  stops changing, so a launch advances the frontier by a whole tile (or more:
  `.cg` loads let programs see sibling tiles' stores through L2 mid-launch).
- **`scan`** (default) — rows and columns are flooded with a *segmented
  max-scan* (`tl.associative_scan` with walls resetting the running max), so
  one pass fills entire horizontal or vertical runs regardless of length.
  Convergence takes roughly one round per bend in the blob rather than one
  launch per pixel of distance. Column passes use a chunked kernel that walks
  down 32 adjacent columns at a time, carrying the scan state between chunks
  to keep loads coalesced.

## Blob shapes (`--shape`)

- `amoeba` (default) — wavy star-shaped blob, ~⅓ of the image red.
- `spiral` — Archimedean spiral arm: compact on screen but with a very long
  geodesic, the worst case for BFS-style propagation.
- `disc` — filled circle, the friendliest case.

## Files

- `main.py` — benchmark CLI (throughput, verification, PNGs)
- `kernels.py` — Triton kernels + host convergence loops
- `blobs.py` — synthetic single-blob generators
- `cupy_bridge.py` — run Triton on CuPy memory without torch
