# flood-fill-cuda

GPU-accelerated flood fill, built up in stages from a sequential CPU BFS
to a cooperative-launch, N-block CUDA kernel that discovers, labels and
fills every blob in the image itself — no seeds given — using CUDA and
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

# Run everything (all chapters' correctness suites)
uv run pytest

# Run one chapter's benchmark and regenerate its own dashboard section
uv run python -m flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.benchmark
uv run python -m flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.visualize

# Regenerate the whole-project dashboard (all chapters so far, one page)
uv run python -m flood_fill_cuda.dashboard

# Paint-and-fill web service: paint a blob, the GPU floods it live
uv run python -m flood_fill_cuda.service
```

## Layout

```
src/flood_fill_cuda/
  shared/         scene generators, CPU oracles, bandwidth model, shared HTML/plot core — used across chapters
  chapters/       the numbered narrative: ch00_cpu_baseline .. ch05_gpu_nblob_nblock
  dashboard/      assembles every chapter's renderer into one whole-project dashboard page
  service/        interactive web app — paint a blob, ch03's kernel floods it, browser animates + it falls
  experiments/    live side-tracks outside the numbered chain (triton/, scan_multi_blob/)
  tutorials/      standalone Numba/CUDA learning scripts
  results/        generated benchmark JSON/CSV/HTML and wavefront renders, one folder per chapter (+ dashboard/)
graveyard/        superseded code, kept for reference, not part of the live tree
```

Each chapter folder holds its kernel, CPU-matching test suite, and a
`benchmarks/` subfolder with the scripts that measure it and render its
own dashboard section — see `chapters/README.md` for what each chapter
actually found, and run `uv run python -m flood_fill_cuda.dashboard` for
the combined view.
