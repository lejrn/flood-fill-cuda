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

## Abstract

The same job — find the red blobs in a white image and fill them — is
solved over and over, each chapter answering the previous one's
measured weakness:

```
CPU BFS ──► 1 block ──► 2 blocks ──► N blocks ──► 2 blobs ──► N blobs ──► N runs
 "one core   "one SM is    "2 SMs      "one blob    "who finds     "why move
 is serial"  4% of the     are 8%"     is one BFS"  the seeds?"    pixels at all?"
             GPU"
```

Chapter 1 puts a level-synchronous BFS inside one CUDA block and hits
the shared-memory wall (a ring that overflows, solved by a spill tier).
Chapter 2 adds a second block and buys almost nothing — until the queue
goes global. Chapter 3 goes cooperative: N co-resident blocks with
grid-wide barriers between BFS levels, plus the 8-connectivity and
radius-2 experiments. Chapter 4 fills two blobs at once and shows one
multisource launch beats two sequential ones. Chapter 5 removes the last
crutch — the seeds themselves: the GPU finds every blob's canonical
seed, merges colliding flood waves with an atomicMin union-find, and
discovers-labels-fills 755,000 blobs in ~25 ms, one launch, zero seeds
given. The tuning epilogue densifies seeding on a stride lattice
(S8 optimal on solids), adds an interior seeding rule (3.2× on disks),
and closes with a comic register lesson: the lattice kernel spent 129
registers per thread where the two-blocks-per-SM line is exactly 128 —
one register cost half the GPU, fixed twice over (compiler cap and a
split build), the fixes landing in a dead heat.

Chapter 6 stops optimizing *how* pixels move and asks whether the pixel
is the right unit at all. `images/input/input_blobs.png` has 81,000,000
pixels, 13.4M of them red — and only **539,207 runs** (maximal red spans
within a row). Connectivity, canonical labels and the spans to paint are
all facts about runs, so the whole connected-components problem shrinks
to 539k items and the clock collapses onto the only two things that must
still touch pixels: one read and one write. The same image ch05 recolors
in **53 ms** takes **3.03 ms** (RGB in, recolored in place) or
**1.35 ms** from a packed 1-bit mask — 17.6× and 39.5×, with the
labeling alone at 0.70 ms and blob *shape* no longer mattering (the
serpentine that cost ch03 32,641 BFS levels costs one merge pass). It
also finds the wall: at 243 MB, reading the RGB image costs ~1.3 ms at
the measured read peak, so no algorithm recolors it from RGB in under a
millisecond on this hardware — and the negative results (word stores,
channel-skipping, one-block-per-row) are as load-bearing as the wins.

Every kernel is pixel-exact against a compiled CPU oracle; every number
is a committed benchmark JSON. The one-page overview —
**abstract, glossary, and the grand table (every approach × every shape
× scale)** — lives at [`site/index.html`](site/index.html); rebuild it
with:

```bash
uv run python -m flood_fill_cuda.overview.bench    # ~5-10 min GPU session
uv run python -m flood_fill_cuda.overview.build    # writes site/index.html
```

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
  chapters/       the numbered narrative: ch00_cpu_baseline .. ch06_gpu_nblob_runs
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
