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
| 1 | `grid.sync` costs more across 24 SMs than across 2 | **mildly** — ~1.7 µs at 1 block → ~2.2 µs at 192 → ~2.6 µs at 384 (serpentine-derived); mostly fixed, but the growth is real and turns the capacity ends into losses |
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
| 32  | 384 | 12,288 | 576 | 18,432 |
| 64  | 192 | 12,288 | 288 | 18,432 |
| 128 |  96 | 12,288 | 144 | 18,432 |
| 256 |  48 | 12,288 |  72 | 18,432 |
| 512 |  24 | 12,288 |  24 | 12,288 |

Every instrumented column multiplies out to the same **12,288 threads =
512 per SM**: the kernel's ~104 registers/thread exhaust the register file
at exactly 512 resident threads regardless of how they are grouped into
blocks — so the smallest legal block size yields the most blocks, and
**384 (tpb=32) is this kernel's absolute block ceiling**. At the dual
stage register pressure capped the *block* (tpb ≤ 512); here it caps the
*whole grid*. The leaner bare twin fits 768/SM, and at tpb=32 its 576
blocks are **Ada's hard architectural cap of 24 blocks per SM** — no
kernel on this GPU can place more; only a register diet gets the
instrumented kernel near it.

## Bandwidth instrumentation (`bandwidth.py`)

1. **Measured peak** — a saturating D2D grid-stride copy (two 256 MB int64
   buffers, CUDA-event timed, median of 10): **187–193 GB/s** across
   sessions on this machine (laptop clocks). Every derived figure is
   expressed against the same-session measurement, never a spec-sheet
   number.
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
| sq_256 center | 16K | 0.1 | 0.9 | 1.1 | 1.3 | 0.66× | CPU wins | 0.6 (0.3%) |
| sq_1024 center | 262K | 4.7 | 6.9 | 5.5 | **5.0** | 1.39× | 0.95× | 2.3 (1.2%) |
| sq_2000 center | 1M | 23.2 | 25.5 | 12.8 | **8.2** | 3.10× | 2.83× | 5.4 (2.9%) |
| sq_4000 corner | 16M | 316.0 | 270.3 | 150.1 | **59.5** | 4.54× | 5.31× | 12.1 (6.5%) |
| disk r=950 | 2.8M | 40.4 | 63.5 | 27.7 | **11.2** | 5.66× | 3.60× | 10.0 (5.4%) |
| disk r=1900 | 11.3M | 284.1 | 191.3 | 97.6 | **27.0** | 7.09× | 10.52× | 18.7 (10.0%) |
| serpentine_256 | 33K | 0.3 | 57.4 | 127.2 | 151.6 | 0.38× | 0.00× | ~0 |
| sq_4600 full | 21.2M | 540.2 | 341.9 | 176.2 | **46.3** | 7.38× | 11.66× | 17.1 (9.1%) |
| sq_5000 center | 25M | 567.2 | 421.2 | 217.7 | **54.4** | 7.75× | 10.43× | 20.9 (11.2%) |
| sq_6000 center | 36M | 885.6 | 589.5 | 302.8 | **85.4** | 6.90× | 10.37× | 18.7 (10.0%) |
| sq_8000 center | 64M | 2041.2 | 1062.6 | 516.9 | **133.7** | 7.95× | **15.27×** | 21.6 (11.6%) |

(Numbers are the newest committed run; medians move a few percent between
sessions with this laptop's clocks — earlier runs' JSON/CSVs remain in
`benchmark_results/` as the durable record.)

v2 = single-block spill kernel; dual = dual-block global kernel, both
re-measured fresh in the same session. The 64M px scene is the guarded
stretch run: it completed; **10000² (the hypothesis scale) cannot even be
attempted — its ~3 GB of host arrays exceed this laptop's free RAM.** The
Chapter 1 host-RAM ceiling, not the GPU, is the binding constraint.

## The centerpiece: blocks × tpb sweep (Mpx/s, median of 3)

`*` = that tpb's cooperative maximum; `-` = beyond capacity.

**sq_4000_corner (16M px), Mpx/s:**

| blocks | tpb 32 | tpb 64 | tpb 128 | tpb 256 | tpb 512 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.3 | 20.5 | 39.2 | 64.2 | 85.7 |
| 2 | 21.0 | 40.6 | 71.7 | 111.0 | 135.1 |
| 4 | 41.7 | 75.4 | 123.7 | 177.8 | 195.4 |
| 8 | 77.1 | 134.2 | 206.3 | 252.8 | 220.0 |
| 16 | 136.7 | 212.6 | 278.2 | 297.7 | 216.6 |
| 24 | - | - | - | - | 202.7* |
| 32 | 214.7 | 285.8 | 308.6 | 287.2 | - |
| 48 | 261.2 | 311.6 | 314.0 | 280.3* | - |
| 96 | 302.7 | 303.6 | 262.5* | - | - |
| 128 | **317.0** | 295.2 | - | - | - |
| 192 | 299.1 | 273.0* | - | - | - |
| 384 | 223.9* | - | - | - | - |

**disk_4001_r1900 (11.3M px), Mpx/s:**

| blocks | tpb 32 | tpb 64 | tpb 128 | tpb 256 | tpb 512 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.0 | 20.8 | 39.3 | 65.3 | 89.2 |
| 2 | 20.4 | 42.0 | 76.2 | 118.4 | 153.3 |
| 4 | 42.8 | 82.5 | 140.4 | 203.7 | 242.3 |
| 8 | 84.3 | 154.3 | 243.3 | 320.6 | 347.9 |
| 16 | 156.8 | 254.3 | 354.2 | 422.0 | 421.9 |
| 24 | - | - | - | - | 416.6* |
| 32 | 260.5 | 370.5 | 452.3 | 462.5 | - |
| 48 | 330.2 | 426.8 | 476.6 | 454.0* | - |
| 96 | 421.8 | 461.1 | 426.9* | - | - |
| 128 | 448.5 | **477.7** | - | - | - |
| 192 | 430.1 | 443.2* | - | - | - |
| 384 | 358.4* | - | - | - | - |

**serpentine_256** (kernel ms — lower is better): ~113 ms at 1 block,
drifting up to **169 ms at 384 blocks**. 65,792 `grid.sync`s *are* the
runtime (1.7 → 2.6 µs each as the barrier population grows); no
configuration of blocks and threads can help a shape that starves every
block between barriers — more blocks only make the barrier costlier.

### What the sweep says

- **Near-linear scaling to ~8–16 blocks, then a plateau — and past it a
  measured decline.** The capacity ends (384×32, 192×64) run 10–30% below
  the best cells: every extra block is another barrier arrival, and past
  ~48 blocks that is all it is. Prediction 4's plateau is real, and the
  block ceiling overshoots it.
- **The best cells obey a rule: the smallest tpb whose grid still covers
  the peak frontier in one stride pass, with blocks maxed.** The square's
  levels peak at 4,000 px → best is 128×32 = 4,096 threads (317 Mpx/s);
  the disk's peak ring is ~7,600 px → best is 128×64 = 8,192 threads
  (478 Mpx/s). Mid-plateau neighbors (48×128, 96×64…) sit within clock
  noise of them.
- **Why smaller blocks win — spreading, observed via %smid.** The
  scheduler places consecutive block ids on *different* SMs (blocks 0–7
  land on SMs 0, 2, 4, …, 14), and grid-stride work goes to the first
  ⌈level/tpb⌉ blocks — so a 2,000-px level runs on 4 SMs at tpb=512 but
  ~16 SMs at tpb=128: same pixels, four times the L1s, load/store units
  and issue slots. At a fixed 12,288 threads the square reads 202.7
  (24×512) → 280.3 (48×256) → 262.5–273.0 (96×128 / 192×64) → 223.9
  (384×32): small blocks beat big blocks until barrier arrivals eat the
  gain — the optimum is interior.
- **Attribution is honestly open.** At the plateau the *model* says only
  ~7–11% of the measured ~190 GB/s copy peak. But the model is a lower
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

## Glossary (the stage's load-bearing terms)

- **Throughput** (תפוקה) — finished work per second: **Mpx/s**, pixels
  recolored per second. **Bandwidth** (רוחב פס) — bytes carried per second
  between the GPU cores and memory: **GB/s**. Throughput is the product
  coming off the line; bandwidth is the trucking capacity feeding it.
  Every pixel needs ~dozens of bytes moved, so saturated trucks cap the
  line even with idle workers.
- **Register** — a thread's private scratch slot, the fastest storage on
  the chip. The compiler counts how many live variables one thread of the
  kernel needs (`x, y, pixel, front, rear, level`, loop state…) — this
  kernel needs **~104 per thread**. Per *thread* because in SIMT every
  thread holds its own copy of every variable.
- **Register file** — each SM's fixed supply of **65,536 registers**,
  shared by all threads seated on it. 65,536 ÷ 104 ≈ 630, rounded down by
  the hardware's coarse allocation granularity to **512 threads per SM**
  (that's threads per SM — the registers per SM are the fixed 65,536).
  ×24 SMs = the ubiquitous **12,288 threads**: every capacity number in
  this stage (384×32, 192×64, 96×128, 48×256, 24×512) is this one wall in
  different clothing.
- **Resident / co-resident** — a thread (or block) is *resident* when its
  seat and registers are physically allocated on an SM right now, rather
  than waiting in the scheduler's queue. Blocks are *co-resident* when all
  of them are seated simultaneously — the precondition for `grid.sync`:
  a block that never gets a seat can never arrive at the barrier, so the
  seated ones would wait forever. That is why cooperative launch refuses
  more blocks than can all be seated at once (the queried max), and why
  exceeding it is deadlock, not slowness.
- **`grid.sync`** — the all-blocks checkpoint: nobody starts level L+1
  until everyone finished level L. Block-local `syncthreads` costs tens of
  cycles; `grid.sync` spans all SMs and costs **~1.7–2.6 µs (~4,000
  cycles)**, growing with the block count. Two per level (publish the new
  queue rear; everyone reads it) × 7,999 levels of the corner square ≈
  **16,000 syncs ≈ 32 ms** — most of that scene's runtime is the ritual.
- **Bare twin** — the same kernel with every measurement counter deleted;
  exists so instrumentation cost is measured, not assumed. Fewer variables
  → fewer registers → 768 threads/SM fit. **Register diet** — capping
  registers at compile time (`max_registers=N`); excess variables spill to
  slow memory. The bare twin proves the diet buys residency; the open
  question is whether the spills cost more than the residency pays.
- **Hard architectural caps** — independent of registers, an Ada SM seats
  at most **24 blocks** and **1,536 threads**; 24 blocks × 24 SMs = 576 is
  a ceiling no kernel can pass.
- **Equal-thread diagonal** — the sweep-table cells with blocks × tpb =
  12,288 (24×512 … 384×32). Total threads identical, only the grouping
  differs, so comparing along it isolates the grouping effect.
- **Plateau** — the flat part of the scaling curve: once the grid covers
  the typical frontier and every SM is fed, extra blocks add barrier
  arrivals and nothing else, so Mpx/s stops rising (and past ~128 blocks,
  falls).

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
