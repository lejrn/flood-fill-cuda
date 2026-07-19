# Multi-Block Flood Fill — 2 blocks → N blocks, one global queue

Chapter 3 of the single-blob evolution (see `../README.md`). The dual-block
stage's verdict was that the **global-memory queue kernel wins**, so this
stage scales exactly that design — one monotonic global queue, grid-stride
work loops, two `grid.sync()` per level — to any cooperative grid size, and
sweeps **blocks × threads-per-block** to find where the scaling stops and
why. It also adds the project's first **bandwidth instrumentation**,
because the stage hypothesis says bandwidth is what the scaling will hit.

## The predictions (stated before measuring — house style)

| # | prediction | verdict |
|---|---|---|
| 0 | The frontier queue's hot window fits L2 even at huge blob sizes (a 10000² blob's frontier ≈ 40K px × 4 B ≈ 160 KB ≪ 32 MB), so queue traffic stays cheap | consistent with everything measured; direct proof needs `ncu` |
| 1 | `grid.sync` costs more across 24 SMs than across 2 | **barely** — ~1.73 µs at 1 block → ~2.2 µs at 192 (serpentine-derived); the cost is mostly fixed |
| 2 | The single global rear becomes an atomic hotspot at 48+ blocks | unproven — a plateau exists but cannot be attributed without `ncu` |
| 3 | The serpentine still loses | **confirmed, emphatically** — it got *worse* (145 ms vs the single-block kernel's 71 ms) |
| 4 | If bandwidth is the real constraint, Mpx/s plateaus while blocks grow | **the plateau is real** — flat from 48 → 96 → 192 blocks; attribution open (see below) |

## What changed from `dual_block` (and what didn't)

The algorithm is byte-for-byte the dual "global" kernel's: claim protocol
(bounds → is-red → `atomic.cas(visited)` → enqueue), warp-aggregated global
enqueue, two `grid.sync()` per level, `depth[x,y]=level` at dequeue. Only
the instrumentation stopped assuming two blocks:

- per-block work counts and %smid observations moved from fixed counter
  slots (`PROCESSED_B0/B1`, `SMID_B0/B1`) to a **`block_stats` int64
  `(blocks, 2)` array**;
- the per-level trace collapsed to **one grid-wide 1D row** — a per-block
  per-level trace at this stage's grid sizes would cost 400 MB (48 blocks)
  to 4.8 GB (576), and the owner map already answers "who filled what";
- `owner` widened **int8 → int16** (cooperative capacity reaches 192
  blocks; a follow-up with leaner registers could exceed 127);
- `blocks=None` launches the **queried cooperative maximum**
  (`max_cooperative_grid_blocks`, never hardcoded); `blocks=1` is legal and
  is the equivalence anchor in the tests.

## Cooperative capacity — registers now cap the *grid*

Measured on the RTX 4060 Laptop GPU (24 SMs, 64 K registers/SM):

| tpb | instrumented max blocks | = grid threads | bare max blocks | = grid threads |
|----:|----:|----:|----:|----:|
| 64  | 192 | 12,288 | 288 | 18,432 |
| 128 |  96 | 12,288 | 144 | 18,432 |
| 256 |  48 | 12,288 |  72 | 18,432 |
| 512 |  24 | 12,288 |  24 | 12,288 |

Every instrumented column multiplies out to the same **12,288 threads =
512 per SM**: the kernel's ~104 registers/thread exhaust the register file
at exactly 512 resident threads regardless of how they are grouped into
blocks. At the dual stage register pressure capped the *block* (tpb ≤ 512);
here it caps the *whole grid*. The leaner bare twin fits 768/SM — living
proof that register dieting buys residency.

## Bandwidth instrumentation (`bandwidth.py`)

1. **Measured peak** — a saturating D2D grid-stride copy (two 256 MB int64
   buffers, CUDA-event timed, median of 10): **192 GB/s** on this machine.
   Every derived figure is expressed against this operational reference,
   never a spec-sheet number.
2. **Derived model** — algorithmic bytes from the kernel's own
   exactly-once counters: `processed × (4 queue-read + 3 img-recolor + 4
   depth [+ 2 owner]) + processed × 4 neighbors × 3 img-read +
   cas_attempts × 8 + (filled−1) × 4` ≈ 61 B/pixel on solid interiors.
   **A lower bound on traffic, not a measurement**: L2 absorbs the queue's
   hot window (deflating real DRAM bytes) while 32 B sector granularity
   inflates them — each scattered 3–4 B access pulls a whole sector.

## Results (median of 5, blocks=None → 48, tpb=256)

| scene | filled | @njit ms | v2 ms | dual ms | **multi ms** | vs v2 | vs @njit | model GB/s (% peak) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sq_256 center | 16K | 0.1 | 1.0 | 1.3 | 1.4 | 0.70× | CPU wins | 0.6 (0.3%) |
| sq_1024 center | 262K | 5.8 | 11.0 | 6.9 | **4.2** | 2.64× | 1.40× | 2.9 (1.5%) |
| sq_2000 center | 1M | 29.8 | 27.9 | 15.1 | **8.5** | 3.28× | 3.51× | 4.9 (2.6%) |
| sq_4000 corner | 16M | 311.7 | 278.0 | 148.9 | **60.0** | 4.63× | 5.20× | 12.3 (6.4%) |
| disk r=950 | 2.8M | 38.4 | 71.3 | 29.6 | **10.8** | 6.58× | 3.54× | 12.2 (6.3%) |
| disk r=1900 | 11.3M | 277.3 | 186.4 | 97.2 | **26.8** | 6.97× | 10.37× | 18.9 (9.8%) |
| serpentine_256 | 33K | 0.3 | 70.8 | 147.5 | 145.1 | 0.49× | 0.00× | ~0 |
| sq_4600 full | 21.2M | 519.6 | 355.7 | 188.5 | **49.7** | 7.15× | 10.45× | 19.2 (10.0%) |
| sq_5000 center | 25M | 674.5 | 432.1 | 213.6 | **54.6** | 7.91× | 12.35× | 20.6 (10.7%) |
| sq_6000 center | 36M | 1012.0 | 592.5 | 294.1 | **74.9** | 7.91× | 13.51× | 21.6 (11.2%) |
| sq_8000 center | 64M | 2026.0 | 1041.9 | 510.8 | **132.2** | 7.88× | **15.33×** | 21.9 (11.4%) |

v2 = single-block spill kernel; dual = dual-block global kernel, both
re-measured fresh in the same session. The 64M px scene is the guarded
stretch run: it completed; **10000² (the hypothesis scale) cannot even be
attempted — its ~3 GB of host arrays exceed this laptop's free RAM.** The
Chapter 1 host-RAM ceiling, not the GPU, is the binding constraint.

## The centerpiece: blocks × tpb sweep (Mpx/s, median of 3)

`*` = that tpb's cooperative maximum; `-` = beyond capacity.

**sq_4000_corner (16M px):**

| blocks | tpb 64 | tpb 128 | tpb 256 | tpb 512 |
|---:|---:|---:|---:|---:|
| 1 | 20.7 | 39.1 | 62.8 | 75.4 |
| 2 | 41.0 | 71.8 | 107.1 | 114.9 |
| 4 | 75.5 | 127.8 | 176.3 | 145.1 |
| 8 | 134.4 | 201.9 | 252.3 | 185.8 |
| 16 | 213.6 | 274.5 | 303.0 | 211.3 |
| 24 | 265.0 | 295.5 | 270.6 | 204.9* |
| 48 | 298.9 | **315.8** | 243.9* | - |
| 96 | - | 263.6* | - | - |
| 192 | 265.4* | - | - | - |

**disk_4001_r1900 (11.3M px):**

| blocks | tpb 64 | tpb 128 | tpb 256 | tpb 512 |
|---:|---:|---:|---:|---:|
| 1 | 19.0 | 36.7 | 61.1 | 89.0 |
| 2 | 38.6 | 69.2 | 104.0 | 153.1 |
| 4 | 76.4 | 131.2 | 191.0 | 242.5 |
| 8 | 138.7 | 216.7 | 256.2 | 331.1 |
| 16 | 235.9 | 331.1 | 361.9 | 400.7 |
| 24 | 307.0 | 387.2 | 458.8 | 415.3* |
| 48 | 400.9 | **467.4** | 452.9* | - |
| 96 | - | 422.2* | - | - |
| 192 | 395.6* | - | - | - |

**serpentine_256** (kernel ms — lower is better): flat ~114–146 ms across
the entire grid, worst at the biggest grids. 65,792 `grid.sync`s at ~2 µs
each *are* the runtime; no configuration of blocks and threads can help a
shape that starves every block between barriers.

### What the sweep says

- **Near-linear scaling to ~8–16 blocks, then a hard plateau** (~316 Mpx/s
  square, ~467 Mpx/s disk). Doubling 48 → 96 → 192 blocks moves nothing
  (slightly negative) — prediction 4's plateau is real.
- **The best cell is 48 × 128 in both big scenes.** At equal thread
  counts, many smaller blocks beat fewer bigger ones, and tpb=512
  anti-scales beyond ~8 blocks — the scheduler juggles small blocks better
  than big ones.
- **Attribution is honestly open.** At the plateau the *model* says only
  ~7–11% of the measured 192 GB/s copy peak. But the model is a lower
  bound: with 32 B sectors, the kernel's scattered 3–4 B accesses can
  inflate real DRAM traffic ~8×, and 11% × 8 ≈ 90% of peak — *consistent
  with* bandwidth saturation, but consistency is not proof. Candidate
  culprits (DRAM sectors, L2 thrash, the single rear atomic) need `ncu`.
- **Balance is structural, not a defect:** grid-stride ownership follows
  thread id, so any level smaller than the grid feeds low block ids first.
  The per-block CV falls from 686% (tiny scenes — most blocks idle) to 42%
  at 64M px. Another instance of the Chapter 2 lesson that aggregate
  metrics need reading against the frontier trace.
- **Instrumentation overhead is below this laptop's clock noise** — the
  bare twin (pinned to the same block count; its leaner registers would
  otherwise resolve `blocks=None` to a bigger grid) swings −36%…+11%
  between runs. The serpentine's +1.9% on a barrier-dominated scene is the
  cleanest signal that the counters cost roughly nothing.

## Wavefront renders (`wavefront.py` → `wavefront/`)

The kernel records `depth[x,y]` (when) and `owner[x,y]` (which block), so
the BFS replays as animation with zero extra GPU work. Color legend: **hue
= block** (golden-angle spacing inside 30°–330° — the red band is excluded
because unfilled scene pixels ARE red), **lightness = fill level**
(light → dark), frontier band = near-white tint of the owner's hue.

| artifact | config | what it shows |
|---|---|---|
| `square256_b8_t32.gif/.png` | 8×32 | all 8 blocks share the wave; per-block coherence near the seed decays into speckle |
| `fullred256_b8_t32.gif/.png` | 8×32, corner seed | blocks come online one by one as the frontier outgrows the threads in front of them |
| `disk512_b8_t64.gif/.png` | 8×64 | the frontier ring sweeping a disk, 8-hue trail behind it |
| `square256_b48_t256.png` | 48×256 — the benchmark's own config | a ~440 px frontier feeds exactly 2 of 48 blocks; 46 idle at every barrier |
| `serpentine128_b8_t32.png` | 8×32 | starvation itself: block 0 owns every single pixel |

**The speckle is a finding, not an artifact.** A block owns contiguous
chunks of the queue window, but queue order is warp-aggregated discovery
order, which spatially scrambles within a few levels — ownership becomes
fine-grained noise (with visible horizontal streaks: individual warp
slabs). That scatter is the picture behind the bandwidth section's
sector-inflation argument: adjacent pixels are touched by unrelated warps,
so small scattered accesses drag whole 32 B sectors. Compare the dual
stage's `split` render (two solid territories) — spatial coherence was the
one thing its losing kernel had that the winner doesn't.

## Run

```bash
# Correctness (51 tests vs the 4-connectivity @njit CPU reference).
uv run pytest src/gpu/single_blob/multi_block/test_correctness.py -v

# Benchmark: measured copy peak, scene suite vs @njit/v2/dual-global,
# blocks x tpb sweep; writes JSON + two CSVs to benchmark_results/.
uv run python src/gpu/single_blob/multi_block/benchmark.py

# Wavefront GIFs + gradient PNGs (block hues) into wavefront/.
uv run python src/gpu/single_blob/multi_block/wavefront.py
```

## Open problems (Chapter 4 candidates)

1. **`ncu` ground truth** — arbitrate the plateau: DRAM sector traffic,
   L2 hit rates, atomic contention on the rear. The derived model got the
   project this far; only a profiler can close the attribution gap.
2. **Register dieting** — the bare twin proves 768 threads/SM fit at ~85
   registers; slimming the instrumented kernel's window state would grow
   the grid ceiling and test whether more resident warps move the plateau.
3. **The serpentine remains unbeaten** — and N blocks made it *worse*.
   Tile-based BFS (iterate inside a block-local tile between global syncs)
   is still the standing candidate.
4. **Multi-blob** — connected-component labeling, per the original
   roadmap's design analysis.
