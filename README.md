# flood-fill-cuda

GPU-accelerated flood fill, built up in stages from a sequential CPU BFS
to a cooperative-launch, N-block, two-blob CUDA kernel — using CUDA and
Numba, benchmarked and pixel-tested against a CPU oracle at every stage.

**The real story lives in
[`src/flood_fill_cuda/chapters/README.md`](src/flood_fill_cuda/chapters/README.md)** —
a living document that narrates the evolution chapter by chapter
(inherited problems → approaches tried → measured results → new problems
exposed), on real hardware, with every kernel proven pixel-exact against
a CPU reference.

## Quickstart

```bash
uv sync

# Run everything (all four chapters' correctness suites)
uv run pytest

# Run one chapter's benchmark and regenerate its dashboard
uv run python -m flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.benchmark
uv run python -m flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.visualize
```

## Layout

```
src/flood_fill_cuda/
  shared/         scene generators, CPU oracles, bandwidth model — shared across chapters
  chapters/       the numbered narrative: ch00_cpu_baseline .. ch04_gpu_2blob_nblock
  experiments/    live side-tracks outside the numbered chain (triton/, scan_multi_blob/)
  tutorials/      standalone Numba/CUDA learning scripts
  results/        generated benchmark JSON/CSV/HTML and wavefront renders, one folder per chapter
graveyard/        superseded code, kept for reference, not part of the live tree
```

Each chapter folder holds its kernel, CPU-matching test suite, and a
`benchmarks/` subfolder with the scripts that measure and visualize it —
see `chapters/README.md` for what each chapter actually found.
