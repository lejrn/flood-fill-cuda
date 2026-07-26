# flood-fill-cuda

Find every red blob in an image and recolor it — on the GPU, in CUDA +
Numba, built up over six chapters from a sequential CPU BFS.

Every kernel is pixel-exact against a CPU oracle. Every number is a
committed benchmark JSON, measured on an RTX 4060 Laptop (24 SMs).

**The real story:
[`chapters/README.md`](src/flood_fill_cuda/chapters/README.md)** — a
living document: inherited problems → approaches tried → measured
results → new problems exposed.

---

## The chain

```
CPU BFS ─► 1 block ─► 2 blocks ─► N blocks ─► 2 blobs ─► N blobs ─► N runs
"one core  "one SM is  "2 SMs      "one blob   "who finds  "why move
is serial" 4% of the   are 8%"     is one BFS" the seeds?" pixels at all?"
           GPU"
```

Each chapter answers the previous one's **measured** weakness.

## The chapters

| # | Question | What it built | Headline |
|---|---|---|---|
| **1** | One core is serial | Level-synchronous BFS in one block; shared-memory ring + global spill tier | 2.03× the CPU at 36 Mpx |
| **2** | One SM is 4% of the GPU | Two blocks, three ways to split the work | 2.10× ch1; the plainest design won |
| **3** | Does it keep scaling? | N co-resident blocks, grid-wide barriers, 8-connectivity | 64 Mpx in 100 ms; plateau found at 512 threads/SM |
| **4** | One blob is one BFS | Two blobs in one launch, labels inside the queue entry | 1.05–2.05× vs two sequential launches |
| **5** | Who finds the seeds? | Seedless discovery: candidate waves + atomicMin union-find | **755,577 blobs in 24.8 ms**, one launch, zero seeds |
| **6** | Why move pixels at all? | Runs, not pixels — run-table connected components | **58 ms → 1.46 ms** on the 81 Mpx image |

---

## Chapter 6 — the current best

Chapters 1–5 optimize *how* pixels move. Chapter 6 changes **what moves**.

`images/input/input_blobs.png` — 9000 × 9000:

| | count |
|---|---|
| pixels | 81,000,000 |
| red pixels | 13,451,960 |
| **runs** (red spans inside one row) | **539,207** |
| blobs | 2,522 |

**25× fewer things to work with.** Connectivity, labels and the paint
spans are all facts about *runs*, so the whole connected-components
problem shrinks to 539k items — and the clock collapses onto the only
two things that still touch pixels: one read, one write.

### Results — same image, same session

| | time | vs ch05 |
|---|---|---|
| ch05 best (`split_L8`) | 58.51 ms | — |
| **ch06, RGB in** (recolored in place) | **2.96 ms** | 19.8× |
| **ch06, packed 1-bit mask in** | **1.46 ms** | 40.2× |
| ch06, labeling only (nothing painted) | 0.78 ms | — |

Shape also stopped mattering — the serpentine that cost ch03 32,641 BFS
levels now costs one merge pass:

| scene | ch05 | ch06 (mask) | speedup |
|---|---|---|---|
| serpentine 2048² | 18.79 ms | 0.47 ms | 39.9× |
| disk r=2000 | 26.43 ms | 1.00 ms | 26.5× |
| 100-blob grid | 26.80 ms | 0.62 ms | 43.5× |
| percolation noise (1.4 px/run) | 24.50 ms | 6.18 ms | 4.0× |

### The wall — and where 1 ms actually is

Reading 243 MB of RGB costs **1.14 ms** at the measured peak. So nothing
recolors this image from RGB in under a millisecond. Not the algorithm —
arithmetic.

Measured on crops of the real image:

| tier | ≤ 1 ms up to | ≤ 0.5 ms up to |
|---|---|---|
| RGB contract | 21.9 Mpx | 10.4 Mpx |
| packed mask | **49.8 Mpx** | 17.9 Mpx |
| labeling only | the whole sweep | 45.6 Mpx |

### Results that were thrown away

Each killed a theory that sounded airtight:

- **Word stores instead of byte stores** — no difference (64 vs 61 GB/s).
  The paint is scatter-bound, not store-bound.
- **Skipping color channels that don't change** — *slower* (0.750 vs
  0.665 ms). The branch costs more than the store it skips.
- **One block per image row** — 2.4× slower. 324,000 tiny blocks;
  dispatch outweighs the memory traffic.

### Two instrument bugs it exposed

Both retroactively explain five chapters of "timing noise":

- `counters.copy_to_device()` is a **synchronous** numba H2D copy. Once
  per run it turned six async launches into six host-blocked round trips
  — 1.72 ms of host time, more than the whole GPU pipeline. *The clock
  was measuring Python.*
- **This GPU idles at ~700–1470 MHz of 3105** and won't spin up for a
  millisecond kernel. Cold: 2.27 ms. Hot: 1.37 ms. Benchmarks now spin
  the clock up first and record what they achieved.

---

## Quickstart

```bash
uv sync
uv run pytest                              # 723 tests, all chapters
uv run python -m flood_fill_cuda.dashboard # whole-project dashboard
uv run python -m flood_fill_cuda.service   # the web app, on :8000
```

One chapter's benchmark + its dashboard section:

```bash
uv run python -m flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks.benchmark
uv run python -m flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks.scaling
```

The one-page overview (abstract, glossary, and the grand table — every
approach × every shape × scale) is [`site/index.html`](site/index.html):

```bash
uv run python -m flood_fill_cuda.overview.bench    # ~5-10 min GPU session
uv run python -m flood_fill_cuda.overview.build
```

> **Never** run a benchmark and the web service at the same time, and
> never `--workers > 1`. Concurrent cooperative launches wedge the GPU
> under WSL2.

---

## The web app

`uv run python -m flood_fill_cuda.service` → http://127.0.0.1:8000/

**`/` — paint page.** Paint a blob, let go, watch it fill.

| tool | what it does | kernel |
|---|---|---|
| CPU / GPU | seeded flood fill from where you released | ch01 / ch03 |
| RUNS | no seed — finds and recolors *every* blob at once | ch06 |
| ✳ SHOOT | fire hundreds to 50,000 drops; scans the whole canvas | ch06 |

**SHOOT** is the clearest demo of chapter 6. The canvas never clears, so
drops pile up and start touching:

| drops fired | blobs found | merged away | GPU | CPU |
|---|---|---|---|---|
| 500 | 311 | 189 | 0.68 ms | 3.9 ms |
| 10,000 | 4,068 | 5,932 | 0.68 ms | 20.9 ms |
| 50,000 | **1,670** | 48,330 | 0.77 ms | 36.4 ms |

Two things to watch: the **GPU time never moves** across a 100× range of
input, and the blob count **peaks then collapses** — percolation, live.

**`/skysurvey.html` — DEEP FIELD.** Guess the star count; the real kernel
counts for you. Stars that touch merge into one blob, so the true count
is below the number placed — the exact mistake a human eye makes.

---

## Layout

```
src/flood_fill_cuda/
  chapters/    the numbered narrative: ch00_cpu_baseline .. ch06_gpu_nblob_runs
  shared/      scene generators, CPU oracles, bandwidth model, plot core
  dashboard/   assembles every chapter's section into one page
  service/     the web app (paint · RUNS · SHOOT · DEEP FIELD)
  overview/    the one-page grand table
  experiments/ live side-tracks (triton/, scan_multi_blob/)
  results/     generated JSON/CSV/HTML + wavefront renders, one folder per chapter
graveyard/     superseded code, kept for reference
```

Each chapter folder holds its kernel, its CPU-matching test suite, and a
`benchmarks/` subfolder that measures it and renders its own dashboard
section.

---

## How to trust the numbers

- **Pixel-exact.** Every kernel is checked against a compiled CPU oracle
  — visited maps, depth maps, label maps, blob counts, and each blob's
  canonical seed. Not just pixel totals.
- **Same session, interleaved.** A/B comparisons re-measure the baseline
  in the same session and alternate the order, because run-to-run spread
  reaches 73% of the median on this laptop.
- **Losses reported.** Where the CPU wins, the tables say so.
- **Committed.** Every figure quoted here has a timestamped JSON in
  `results/`.
