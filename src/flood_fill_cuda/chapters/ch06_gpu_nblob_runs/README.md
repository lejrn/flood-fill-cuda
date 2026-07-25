# Chapter 6 — runs, not pixels (`ch06_gpu_nblob_runs/`)

**Stop moving pixels.** Chapter 5 discovers, labels and fills every blob
in `images/input/input_blobs.png` — 9000×9000, 13.45M red pixels, 2,522
blobs — in **53 ms**. This chapter asks what the same job costs if it is
bounded only by memory, and answers it by changing the *unit of work*
from the pixel to the **run**: a maximal contiguous span of red inside
one row.

The whole argument is in one inventory:

```
81,000,000 px    the grid
13,451,960 px    red
   539,207       maximal runs of red (mean 24.9 px)
     2,522       blobs
```

There are **150× fewer runs than red pixels**, and every fact the job
needs — connectivity, canonical labels, the spans to paint — is a fact
*about runs*. So the connected-components problem shrinks to 539k items
(a rounding error), and the clock collapses onto the only two things
that must still touch pixels: one read and one write.

| | ch05 best | ch06 rgb | ch06 mask | ch06 labeling only |
|---|---|---|---|---|
| `input_blobs.png` | 53.26 ms | **3.03 ms** (17.6×) | **1.35 ms** (39.5×) | **0.70 ms** |

## Inherited problems

1. **Everything so far moves pixels.** A BFS frontier is a list of pixel
   indices; ch05's `ccl_fill` unions once per red *adjacency* (15.7M of
   them on the squares scene); the label map is one int32 per pixel.
   ch05's own open-problem list asks for "BUF/BKE 2×2-block union-find
   to cut ccl's per-adjacency volume" — this chapter answers the
   question the list was circling, but with 1×N blocks, not 2×2.
2. **ch05's clock is a geodesic clock.** The serpentine costs ch03
   32,641 barriers because BFS depth *is* the runtime. Anything shaped
   badly is slow, and no amount of bandwidth fixes it.
3. **Nothing has ever been measured against the machine's actual
   limit.** Five chapters compare against a D2D *copy* peak. A pipeline
   whose two ends are a read and a write needs a measured read peak and
   a measured write peak, or "we are at the floor" is an opinion.

## The design

```
pack     RGB -> 1 bit/px          the ONLY full-resolution read
count    runs per row          \
scan     row -> run offsets     |  over the packed mask:
emit     the run table          |  10.15 MB, not 243 MB
merge    union-find over vertically adjacent runs   (539k items)
flatten  path-compress to roots                     (539k items)
paint    one warp per run, writes its span          (only red px)
```

Six plain kernels, stream-ordered. **No cooperative launch**: chapters
3–5 needed one because BFS levels must interleave with grid-wide
barriers, but this pipeline is a DAG with no loop, so stream order is
the barrier — and plain kernels run at full occupancy instead of the
cooperative residency cap that cost ch05 half its grid over a single
register.

### Why the run table carries the canonical label for free

Runs are emitted in row-major order (x ascending, then y0 ascending), so
run ids are ordered exactly like the linear index of their first pixel:

```
id(r) < id(s)   <=>   lin(r) < lin(s),   lin(r) = x*height + y0
```

ch05's union-by-`atomicMin` over *pixel* indices converges to a blob's
minimum linear index. The identical protocol over *run ids* converges to
the blob's minimum-id run — whose first pixel is exactly the blob's
lex-min pixel. **The canonical label is therefore unchanged**, the same
`cpu_label_components` oracle judges both chapters, and the label map,
painted image, blob count and per-blob canonical seeds all match ch05
bit for bit. Only the carrier changed.

That ordering is the one thing the design pays for: it is why `count`
and `scan` exist at all (a row's base offset is a prefix sum over rows,
so it cannot be known until every row is counted). The re-read costs
10.15 MB — 0.05 ms of the budget, and worth it.

### The bit algebra

`mask` is `(width, ceil(height/32))` uint32; bit *b* of word *(x, w)* is
pixel *(x, w·32+b)*. For a word `w` with predecessor bit `p` and
successor bit `q`:

```
starts = w & ~((w << 1) | p)          a 1 whose predecessor is 0
ends   = w & ~((w >> 1) | q<<31)      a 1 whose successor is 0
```

Within a row the k-th start and the k-th end belong to the same run, so
two independently warp-scanned streams pair up by index — no cross-word
stitching, no serial walk. Bits past `height` are written as zero, which
is what terminates the last run of every row.

### What is given up, deliberately

There is no BFS, so there is **no `depth` map and no level count**.
Chapters 1–5 measure a geodesic clock; this one has no geodesic
structure at all — `merge` is a single data-independent pass over run
adjacencies. That is why the serpentine, which cost ch03 32,641 levels,
costs the same here as a square: **0.49 ms, 37.8× ch05**.

## Results (RTX 4060 Laptop, 24 SMs; measured peaks: 193 GB/s copy, 167 GB/s read, 169 GB/s write)

| scene | px | runs | blobs | mean run | ch05 `split_L8` | ch06 rgb | ch06 mask | label only | ×rgb | ×mask |
|---|---|---|---|---|---|---|---|---|---|---|
| **`input_blobs.png`** | 81M | 539,207 | 2,522 | 24.9 px | 53.26 ms | **3.03 ms** | **1.35 ms** | 0.70 ms | 17.6× | **39.5×** |
| `input_blocks.png` | 1M | 61,479 | 21,618 | 6.3 px | 1.55 ms | 0.48 ms | 0.41 ms | 0.30 ms | 3.2× | 3.8× |
| grid of 100 blobs | 16M | 36,000 | 100 | 360 px | 25.46 ms | 0.78 ms | 0.61 ms | 0.30 ms | 32.5× | 41.4× |
| random noise 4000² | 16M | 3,361,514 | 755,577 | 1.4 px | 24.16 ms | 6.35 ms | 5.94 ms | 2.62 ms | 3.8× | 4.1× |
| disk r=2000 | 17.6M | 4,001 | 1 | 3141 px | 27.46 ms | 1.11 ms | 0.92 ms | 0.24 ms | 24.7× | 29.8× |
| serpentine 2048² | 4M | 2,048 | 1 | 1024 px | 18.68 ms | 0.60 ms | 0.49 ms | 0.31 ms | 31.2× | 37.8× |

ch05 runs the config its own tuning picked for the headline image
everywhere, which is not its per-scene best — read the non-PNG rows as
indicative, not as ch05's ceiling.

**Phase breakdown, `input_blobs.png`, rgb contract (ms):**

| pack | count | scan | emit | merge | flatten | paint |
|---|---|---|---|---|---|---|
| 1.65 | 0.08 | 0.02 | 0.23 | 0.32 | 0.03 | 0.65 |

Two phases are the runtime. `pack` is the only full-resolution read;
`paint` is the only write. **The entire connected-components problem —
count, scan, emit, merge, flatten — is 0.68 ms**, and 0.32 of that is
the union-find.

### Two contracts, and why both are always reported

At 81 Mpx the RGB image is 243 MB and the packed mask is 10.15 MB. At
the measured read peak, *merely reading the RGB* costs ~1.3 ms.

> **No algorithm of any kind recolors this image from RGB in under a
> millisecond on this hardware.** The sub-millisecond numbers belong to
> the packed-mask contract, and they say so every time.

That is not a caveat bolted on — it is the chapter's central finding.
The representation, not the algorithm, is what stands between 3 ms and
1 ms; ch05's 53 ms was never near either wall.

## The measurement lesson (which invalidated a day of numbers)

**This GPU idles at 1470 MHz of a 3105 MHz maximum and does not raise
its clock for short kernels separated by host syncs.** Timed cold, the
pipeline reads 2.27 ms; run back-to-back for a few seconds it reads
1.37 ms and is still falling. Mid-session the same unchanged `merge`
kernel measured 0.38, 0.59 and 0.73 ms.

Every earlier chapter's "the spread is 73% of the median" noise
complaint has this underneath it. The benchmark now spins the GPU for 8
seconds before timing anything and records the achieved clock in the
JSON.

**And worse:** `counters.copy_to_device(zeros)` — a numba H2D copy from
pageable memory — is **synchronous**. Issued once per pipeline run it
turned six asynchronous launches into six host-blocked round trips:
**1.72 ms of pure host enqueue time**, more than the entire GPU
pipeline. The clock was measuring Python. Zeroing the counters inside
`row_scan_kernel` instead dropped host enqueue to 0.33 ms and removed a
launch.

## What was built, measured, and thrown away

The negative results are the load-bearing ones here, because each killed
a theory that sounded airtight:

| idea | theory | measured |
|---|---|---|
| **word stores in `paint`** | three per-channel byte stores each touch the same ~96 bytes of sectors — write aligned 4-byte words instead and cut write traffic 3× | **no difference**: 64 GB/s vs 61 GB/s. Store width is irrelevant; the kernel is scatter-bound, not store-bound. (For *long* runs it does win — the crossover is ~64 px/run — but the mean here is 25.) |
| **skip unchanged channels** | every painted pixel is known to be pure RED, so magenta and orange need one store instead of three (palette mean 1.83/3), and the test is per-color hence warp-uniform | **slower**: 0.750 ms vs 0.665. The branch costs more than the store it skips. |
| **one block per row in `pack`** | the natural 2D mapping, no runtime division | **2.4× slower**: 324,000 blocks of 256 threads, and block dispatch alone outweighs the memory traffic. Capping `gridDim.y` at 64 and striding rows: 3.71 ms → 1.12 ms, i.e. the device's full read peak. |
| **`prev`/`next` words via shfl in `emit`** | replace 3 global loads per word with 1 load + 2 shuffles | **no difference** — those loads were already L1 hits. |

Two that *did* pay:

- **Independent loads in `_is_red`.** `a == 255 and b == 0 and c == 0`
  short-circuits, so it compiles to three *dependent* loads separated by
  branches — the G byte is not requested until the R byte returns. `&`
  on the comparisons keeps all three in flight.
- **Path halving in `_find`.** ch05's find is deliberately read-only
  ("no compression while unions are in flight"). That rule is stronger
  than it needs to be: writing the grandparent is safe under the
  invariant the union protocol already maintains (`parent[i]` is always
  a same-class index strictly below `i`), and the class minimum is never
  written at all because a root returns before any store. **0.367 →
  0.262 ms.**

## New problems and lessons

- **The unit of work is a design decision, not a given.** Five chapters
  optimized *how* pixels move — barriers, occupancy, seeding density,
  one register. Changing *what* moves beat all of it by 17–40× and made
  blob shape irrelevant.
- **Scattered writes run at ~1/3 of streaming bandwidth.** 40 MB of red
  pixels in 75-byte spans separated by ~375-byte gaps costs 0.65 ms
  (62 GB/s) against a 169 GB/s measured streaming write. Every paint
  formulation lands within 5% of that number regardless of store width,
  instruction count or grid. This is the floor of "recolor RGB in
  place", and the only way under it is to stop writing RGB.
- **Run length is the parameter that decides everything.** 25 px/run
  gives 39.5×; 1.4 px/run (percolation noise, 3.4M runs) gives 4.1×;
  3141 px/run gives 29.8× but under-parallelizes (one warp per run means
  4,001 warps for a 17.6 Mpx disk).
- **A pipeline of small kernels is a host-side problem too.** With the
  synchronous copy removed, host enqueue is 0.33 ms against 1.35 ms of
  GPU work — comfortable, but only 4× of headroom, and CUDA graphs are
  the obvious next lever.

## Open problems → Chapter 7 candidates

1. **`ncu`, still owed** — now for a sharper question than ch05's: why
   do scattered 75-byte writes cap at 62 GB/s, and is it sector
   occupancy, DRAM page thrash, or write-allocate?
2. **Long runs under-parallelize.** One warp per run leaves the disk
   scene with 4,001 warps. Splitting long runs across warps needs a
   prefix sum over run lengths — 539k items, essentially free.
3. **The percolation-noise case (1.4 px/run) is where runs lose their
   edge** — 3.4M runs for 4.8M red pixels. A 2×2-block method (BUF/BKE,
   ch05's original suggestion) should win exactly there; the two are
   complementary, and an image statistic could pick.
4. **CUDA graphs** to collapse six launches, and fusing `count` into
   `pack` for the RGB contract.
5. **Sub-millisecond end to end** needs the output to stop being RGB: a
   paletted (1 B/px) or bit-planed output would put the write at 13 MB
   instead of 40.

## Files

```
kernels.py           the six kernels + the bit algebra and union-find
recolor.py           host driver: RunRecolor (reusable buffers), recolor()
test_correctness.py  109 tests — pixel-exact vs ch05's CPU oracle
benchmarks/
  benchmark.py       head-to-head with ch05, clock spin-up, JSON + CSV
  visualize.py       the dashboard's section 4
```

Run:

```bash
uv run pytest src/flood_fill_cuda/chapters/ch06_gpu_nblob_runs/
uv run python -m flood_fill_cuda.chapters.ch06_gpu_nblob_runs.benchmarks.benchmark
uv run python -m flood_fill_cuda.dashboard
```
