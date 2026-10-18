# Chapter 9 - every CPU core, the GPU's ideas (`ch09_cpu_nthreads/`)

**Can BFS run in parallel on the CPU's cores, and how does it compare?**
Chapters 3, 7 and 8 were compared with ONE CPU core. This chapter
asks the fair question.

The same work, the same ideas, on all 16 logical CPUs. Three
variants, each an `@njit(parallel=True)` loop with `prange`.

**Yes, and the tiles make it pay.** All 16 vCPUs fill the 64 Mpx
square in 21.29 ms, 51x faster than one core. They label the 81 Mpx
frame in 21.06 ms, 13x faster than one core.

The GPU with the same idea is still faster on its own clock: 6.24x on
the square, 9.43x on the frame. Counting its copies in and out, the
CPU wins both, by 1.89x and 2.34x.

| scene (8-conn) | ch09 variant vs its GPU twin | one core | ch09, 1 thread | ch09, 16 threads | GPU | GPU with copies | 16 threads vs GPU | vs GPU with copies |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sq_8000_center` (64 Mpx, one blob) | tile BFS vs ch07 | 1,087.4 | 89.33 | **21.29** | **3.411** | 40.31 | GPU 6.24x | **CPU 1.89x** |
| `sq_8000_center` | pixel BFS vs ch03 | 1,087.4 | 986.2 | 259.0 | 101.2 | 137.0 | GPU 2.56x | GPU 1.89x |
| `input_blobs` (81 Mpx, 2,522 blobs) | tile labels vs ch08's first version | 271.5 | 97.84 | **21.06** | **2.232** | 49.30 | GPU 9.43x | **CPU 2.34x** |
| drones (240 frames, 0.59 Mpx) | tile labels vs ch08's first version | 1.009 | 1.039 | 0.792 | 0.0911 | 0.776 | GPU 8.66x | GPU 1.02x (a tie) |
| `sq_256` (16 K px blob) | tile BFS vs ch07 | 0.1995 | 0.0759 | 0.0563 | 0.0451 | 0.503 | GPU 1.25x | **CPU 8.9x** |
| `serpentine_256` (one-tile front) | tile BFS vs ch07 | 0.470 | 0.165 | 0.214 | 2.962 | 3.698 | **CPU 13.8x** | **CPU 17.3x** |

All times are medians in ms from one session,
`ch09_20261017T233237Z` (JSON and CSV in
`results/ch09_cpu_nthreads/benchmark_results/`). Every number below
comes from it unless another session is named.

**One core** is the like-for-like twin: the oracle's loop with its
buffers kept, the reset and the paint on the clock. **GPU** is the
same idea timed by CUDA events, its L2 flushed first.

**With copies** adds a pageable (ניתן להחלפה, not pinned) copy of
the image in and the painted image back, timed on the host. The CPU's
image is already in host memory.

Factors say who is faster, and by how much. The drone row gives
medians over the 240 frames of each frame's median; its factors are
medians of per-frame ratios.

On the two small scenes 16 threads is not the best count. `sq_256` is
fastest at 2 threads (0.0461 ms, a tie with ch07's 0.0451). The
serpentine is fastest at 4 (0.132 ms, 22.4x faster than ch07).

**The short version.**

The method came first. On the square, tiles on ONE thread are already
12.2x faster than one core, and 16 threads add 4.18x. The pixel BFS,
the same work without tiles, gains 1.10x and 3.80x.

For the tile variants, 16 vCPUs are worth about four cores: 4.2-4.6x
on the big scenes, 26-29% per vCPU. From 4 to 8 threads they gain
nothing.

The rest of the gap is the hardware, with the same idea on both
sides: 6.24x on the square, 9.43x on the frame. On the drone frames
the cores give only 1.35x, and the GPU leads 8.66x by events.

The copies change the verdict (ההכרעה). When the image starts in host
memory, the GPU's copies cost more than the CPU's whole run on both
big scenes. The drones become a tie, the small square a CPU win.

One-tile fronts belong to the CPU at any thread count. On the
serpentine even one core's twin is 6.3x faster than ch07.

Timings taken while this chapter was built ran next to other work on
the same cores. None of them is quoted here.

## Inherited problems

1. **The baseline was one core.** The video's v5 session
   (`v5_20261017T192640Z`, on the branch `video-v5`) compares the GPU
   with the same work on one core. `sq_8000` 8-conn fill: 1,065 ms on
   one core, 3.12 ms for ch07 (341x). `input_blobs` labels: 225 ms,
   2.21 ms for ch08's first version (101x). `sq_256`: 0.192 ms, 0.044
   ms for ch07.
2. **The CPU is a hybrid seen through a hypervisor** (מנהל
   וירטואליזציה). The i9-13900H has 6 P-cores with two threads each
   and 8 E-cores: 14 cores, 20 threads. WSL2 shows 16 of them as a
   synthetic 8 cores x 2 threads, and Windows schedules those vCPUs
   over the real cores. So one vCPU is not a sixteenth of the CPU, and
   efficiency is given per vCPU.
3. **Numba's API exposes no CPU atomics.** Every GPU chapter leans on
   atomic (אטומית) operations: CAS claims, atomic OR, atomicMin. A
   dozen lines of `numba.extending.intrinsic` would emit LLVM's
   `cmpxchg` on the CPU too, and it works inside a `prange`. This
   chapter takes the other route: each claim is safe by construction,
   or harmless when it races, and costs no locked instruction.

## The three variants

| variant | the GPU idea it ports | the bet |
|---|---|---|
| **pixel BFS** (`PixelBFS`) | ch03: a level-synchronous BFS, one barrier per level | the frontier (חזית) splits into chunks (נתחים), one per thread; each chunk writes its discoveries into its own buffer, so no shared queue is needed |
| **tile BFS** (`TileBFS`) | ch07: tiles closed with word operations, a frontier of tiles (64 x 64 here, 32 x 32 in ch07) | a round moves the wave a tile, not a pixel: 55-60x fewer barriers (מחסומים) than levels on a solid square; neighbours talk through inboxes with one writer each |
| **tile labels** (`TileLabels`) | ch08: local pieces per tile, seeds, waves from piece to piece, unions at the end | the waves claim pieces, not pixels; a lost race costs one recorded meeting, never a wrong label |

All three keep the contracts of ch03, ch07 and ch08. RGB uint8 `(W,
H, 3)`, x first, red exactly `(255, 0, 0)`.

One blob: painted `(0, 0, 255)`, 4- or 8-connectivity, a non-red seed
raises `ValueError` (like ch03 and ch07). Many blobs: canonical labels
(the blob's min `x * H + y`) and `palette[label % 6]`, ch05's palette.

## One chunk per thread

A `prange` here always runs over **chunks**, at most one per thread.
Numba's OpenMP layer schedules a `prange` statically (סטטית): thread
`k` runs one contiguous block of the iterations. A test checks this.

So more chunks than threads would only run back to back on the same
threads. And 17 chunks on 16 threads would give thread 0 twice the
work of the others.

Inside `@njit(parallel=True)`, `np.zeros` and `.sum()` become parallel
loops too: a fork and a barrier each, on the clock. The kernels use
`np.empty` and serial loops instead. A test checks that every parallel
loop of every kernel is one of its explicit `prange`s.

## No atomics: why each variant is exact

Each chunk owns what it writes: its slice of an output buffer, its
counters, its tiles.

Where two chunks can still touch one place, it only takes whole
aligned values (one byte or one int32). A store of that size is one
instruction on x86-64, so it is never torn.

These are **benign races** (מרוצים תמימים): plain loads and stores
that another thread may overwrite. The hardware side is sound. The
compiler side is an assumption: LLVM's memory model calls a racing
plain load undefined, and the code relies on each access compiling to
one load or store, as it does here.

Relaxed atomic loads and stores through an intrinsic would make this
formally defined at no cost on x86-64. That change is left open.

### Pixel BFS: a stamp in the visited byte

The frontier is an array of entries `(z << 8) | stamp`. A level cuts
it into contiguous chunks. Chunk `c` writes its discoveries into its
own region of a raw buffer, at most 4 or 8 per entry.

After the barrier, a serial prefix sum (סכום קידומת) over the chunks'
counts gives each region its offset. A parallel copy builds the next
frontier.

The `visited` flag is one `uint8` per pixel, written without atomics.
Chunk `c` stores its stamp (חותמת) `c + 1` when it claims a pixel.

Two chunks can both read 0 and both claim one pixel: a race (מרוץ).
After the barrier the byte holds one of their stamps, the last store.
An entry is expanded only if the byte equals its own stamp.

So exactly one copy of the pixel is expanded. The others are dropped
and counted as duplicates. Duplicates never claim anything, so they
cannot grow from level to level.

Why the fill set is exact (the full argument is in `kernels.py`):

1. A pixel is claimed only next to an expanded pixel, and only if red.
   Nothing outside the blob is reached.
2. A claimer appends what it claims in the same iteration, and every
   region is copied after the barrier. A non-zero flag means the pixel
   is in some frontier, so no red neighbour is missed.
3. A pixel is claimed only during the level that first reaches it. The
   levels are the oracle's levels, and so are the counts.

The tests check the level count and the fill count against the oracle
on every scene. How often races happen depends on the load and on the
thread pool's wait policy.

On a quiet machine, a 65 x 65 full square at 16 threads had duplicates
in every one of 1,000 runs. A test runs it 1,000 times, and every
output must equal the oracle's.

### Tile BFS: inboxes with one writer

A tile is 64 rows times one `uint64` word, stored tile-major: 512
contiguous bytes. The closure fills along the word by one carry
(upward) and a Kogge-Stone fill (downward). Rows are joined by passes,
top-down then bottom-up.

A pass runs again only if a grown row can still feed its backward
neighbour. That is checked per row, when the row is final.

A tile's words are touched only by the thread that visits it. The
frontier is deduplicated, so a tile is visited at most once per round.

Neighbours talk only through **inboxes** (תיבות דואר).
`inbox[buf, t, d]` holds the raw border of `t`'s neighbour in
direction `d`. Each `(target, side)` slot has exactly one writer: the
neighbour on that side.

The inboxes are double-buffered by round parity. In round `r` a tile
reads and clears `inbox[r & 1]`, its own slots only. It writes
`inbox[(r + 1) & 1]` of its neighbours.

So in one round a slot is either read by its owner or written by its
one source. The barrier orders the write before the read.

A tile sends its raw border, not the target's bits. Two filters keep
it from bouncing:

- `known[t, d]`: the latest border each neighbour sent `t`. A send
  happens only if some red bit next to it is not known to be reached.
- `sent[t, d]`: what `t` last sent. Nothing is sent twice.

Both are owned by `t`. Neither filter can drop a needed bit: a dropped
bit is reached already, or was delivered already.

The next frontier is built serially from the chunks' target lists,
with a per-tile round stamp. Frontiers of tiles are small.

No round depends on itself, so the tile BFS's counters (rounds,
visits, closure passes, sends) depend on the image and the seed only.
A test checks they are equal at 1, 2 and 16 threads: any inbox race
would move them.

Pack (RGB to bits) and paint run with rows in parallel. Every word
belongs to the one thread that owns its row.

### Tile labels: a claimer byte beside the label

Five steps:

1. **Pieces** (tiles in parallel). Each tile splits its mask into
   pieces (connected inside the tile), lowest remaining pixel first.
   That pixel is the piece's anchor (עוגן), its min `x * H + y`. A
   piece on the tile's frame is a **ring piece**; any other piece is a
   whole blob.
2. **Seeds** (זרעים). A ring piece is a seed if its anchor has no red
   lex-predecessor neighbour (ch05's corner rule, at the anchor), or
   if its tile is on ch08's diagonal lattice (סריג), `(tx + ty) % 8 == 0`.
3. **Waves** (rounds, frontier pieces in parallel). A frontier piece
   claims the unclaimed ring pieces next to it across its borders.
   Where it finds another wave, the pair is recorded: a meeting.
4. **Unions** (serial). The meetings are unioned, keeping the seed
   with the smaller anchor as the root.
5. **Paint** (tiles in parallel). A tile whose pieces give one colour
   is painted from its mask. Others replay step 1 in the same order.

**Why the root is the canonical label.** A blob's min-index pixel has
no red lex-predecessor, and it is its piece's anchor. So every blob's
canonical piece is a seed, and it has the smallest anchor of its
class. ch08 needs a min-anchor flatten over all comps for this; the
corner rule at the anchor makes it free.

**Claims without atomics.** `lab[Q]` (int32, -1 = unclaimed) is
written only while it reads -1. Two chunks can race: both read -1,
both store their wave, both list `Q`.

Each claim also stores the chunk's stamp in `claimer[Q]`, one byte.
After the barrier, `lab[Q]` holds one racer's wave and `claimer[Q]`
one racer's stamp, not always the same racer's.

In the next round each listed entry of `Q`:

- records a meeting `(its wave, lab[Q])` if they differ;
- expands `Q` only if `claimer[Q]` is its own stamp: exactly one
  entry, and it expands with `lab[Q]`, the final wave.

Every adjacency is seen by the one expansion of one side, with that
side's final wave. That alone joins every wave that touched `Q`. So
the lost-race meeting is a safety net: a fuzz without it stayed exact.

Labels, the paint and the blob count are deterministic. Which wave
wins a race, and how many meetings are recorded, are not.

The **duel** scene forces the race. 32 bars cross three tiles A, Q
and B; at 16 threads, A's and B's seeds start together on different
threads and claim Q's pieces with different waves.

A probe saw a lost claim in 999 of 1,000 duel runs. A test runs it
1,000 times, with every output and invariant exact.

**Bounded buffers.** A chunk's claims and meetings go into its own
regions, sized per entry plus one worst-case expansion. An entry that
might not fit is requeued, unchanged, for the next round. The first
entry of a chunk always fits, so every round makes progress.

## Correctness

`test_correctness.py` and `benchmarks/test_benchmark.py`: **627
tests, all passing** (about 40 seconds with `nice` on a quiet machine,
2-3 minutes on a loaded one).

| group | against | what |
|---|---|---|
| bit helpers | plain-Python references, a pixel BFS inside the tile | row fill, popcount, lowest bit, columns, the closure from one seed and with new seeds on flagged rows |
| the schedule | Numba's OpenMP layer, a fresh compile of every kernel | a `prange` is one contiguous block per thread; every engine uses at most one chunk per thread; every parallel loop is an explicit `prange` |
| one blob, both methods | ch01's 4-conn and shared/'s 8-conn `@njit` oracles | 20 scenes x 4- and 8-conn x 1, 2, 16 threads: visited set, painted image, fill count; the pixel BFS's level count too |
| one blob, races | the same | every level and round split into one chunk per thread; engines reused across seeds and thread counts; a fuzz on odd sizes; the 65 x 65 amplifier (1,000 runs); tile BFS counters equal at 1, 2, 16 threads |
| pack | numpy's red mask | colours that share bytes with red, padding rows, mask columns |
| many blobs | ch05's `cpu_label_components` (8-conn), ch08's `cpu_label` (4-conn) | 30 scenes x 4- and 8-conn x 1, 2, 16 threads: label map, palette image, blob count, every ring piece expanded once |
| many blobs, every path | the same | forced prange rounds, zero-size buffers (requeues), lattices 0, 1, 3, 8, ch08's five fuzz families, the duel amplifier (1,000 runs) |
| the frame | ch05's oracle | `input_blobs.png` crops, and the full 81 Mpx frame once: 2,522 blobs |
| determinism | the first run | one engine, nine runs, three thread counts, two images: identical outputs, rounds and seeds |
| the harness | fault injection, fakes | every timed output checked; the gate, the settle and the runtime rules; the GPU pairing; the summaries |

The scenes include serpentines, spirals, mazes, checkerboards, noise
near the percolation thresholds (0.40-0.62), zigzags on the 64-tile
borders, pixels on tile corners, sizes not multiples of 64, 1 x 1,
blank images, thin strips and a non-red seed.

A review fuzzed the three variants for about 1.08 million runs against
the oracles, races included, and found no wrong output. Its deliberate
bugs (mutants, מוטנטים) were all caught, except one that cannot change
an output.

## Work counted while building (counts, not timings)

These are counters, the same on any machine:

| scene | variant | counts |
|---|---|---|
| 2000 x 2000 all red, centre seed, 8-conn | tile BFS | 18 rounds, 1,024 tiles each closed once, 1,023 idle visits (a same-round neighbour's border, no closure) |
| the same, 4-conn | tile BFS | 33 rounds, 1,024 visits, no idle visit |
| `serpentine_256` | tile BFS | 388 rounds (the pixel BFS: 32,641 levels at 8-conn) |
| `maze_1k` | tile BFS | 1,208-1,214 rounds, 5.2-5.3k visits, 5.3-5.4 closure passes per visit |
| ch07's `zz_serp_1k`, 8-conn | tile BFS | 15,556 rounds of one tile: the worst case, as in ch07 |
| `input_blobs`, lattice 8 | tile labels | 16,200 ring pieces, 280 interior, 7,690 seeds, 5 rounds, 20,001 meetings, 9,314 tiles painted from the mask |
| `input_blobs`, lattice 0 | tile labels | 6,442 seeds, the same 5 rounds, 15,289 meetings |

Meeting counts depend on which wave wins each race, so they vary a
little from run to run; these are one run's.

On the frame, 64-pixel tiles make ch08's lattice cost meetings without
saving rounds. The unions are serial here, so the benchmark times
both lattices.


## Results (i9-13900H under WSL2, RTX 4060 Laptop)

### What is compared

`benchmarks/benchmark.py` is one session for every number. Its
docstring is the full protocol. Each step is reported apart, per
variant:

1. one core's twin (the oracle's loop, buffers kept, the reset and the
   paint on the clock);
2. ch09 on one thread: the **method** (`x_method`);
3. ch09 on 16 threads: the **cores** (`x_cores`, and the speedup at
   1, 2, 4, 8 and 16 threads);
4. the GPU with the **same idea**: the **hardware** (`x_gpu`). That is
   ch03 for the pixel BFS, ch07 for the tile BFS, ch08's first version
   for the labels.

`x_method` and `x_cores` are medians of per-round ratios, paired round
by round. `x_gpu` is a ratio of medians: the GPU ran in its own
blocks, before the CPU passes.

**Scenes.** `sq_256`, `sq_8000_center` and `serpentine_256` at 4- and
8-conn; `input_blobs` and the 240 drone frames at 8-conn; two 1 x 1
scenes for the fixed cost of a call (CPU only).

**The GPU cells.** Every factor names its GPU cell. ch03's cell
(`ch03_conn8`, 48 x 256) is instrumented (מצויד במונים): it counted
202.6M CAS attempts on the square.

ch03's bare warp kernel (`ch03_wc_bare`) takes 75.37 ms on the square.
Against it the pixel BFS's hardware step is 3.44x, not 2.56x.

ch07's cell took 3.411 ms on the square (min 3.096). ch07's own
session had 3.107 ms (`tiles_20261016T064724Z`); against that, the
tile BFS's hardware step would be 6.85x (derived).

**The video's factors.** The video uses `v5_20261017T192640Z`, in
`results/overview/benchmark_results/` on the branch `video-v5` (not
merged here). It had 1,064.7 ms for one core and 3.122 ms for ch07 on
the square: 341x.

Its twin, `cpu_fill8`, fills without the paint, like `one_fill_c8`
here: 1,087.3 ms, 2.1% more. With the paint, this session gives
1,087.4 against 3.411 ms: 319x. The lower factor comes from ch07's
cell, 9% slower here (derived).

An earlier v5 session, `182449Z`, timed the oracles as called, with
their allocations on the clock (1,396 ms on the square). The video
does not use it.

### Thread scaling

Each cell: ms, and in brackets the speedup over one thread (the
median of the per-round ratios). 8-conn unless marked.

| scene | variant | 1 thread | 2 | 4 | 8 | 16 | per vCPU at 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| `sq_8000_center` | tile BFS | 89.33 | 47.64 (1.89) | 28.94 (3.13) | 28.73 (3.10) | **21.29 (4.18)** | 0.26 |
| `sq_8000_center`, 4-conn | tile BFS | 88.87 | 47.98 (1.88) | 28.16 (3.12) | 27.45 (3.22) | 20.35 (4.33) | 0.27 |
| `sq_8000_center` | pixel BFS | 986.2 | 722.8 (1.39) | 460.7 (2.12) | 334.8 (2.97) | 259.0 (3.80) | 0.24 |
| `sq_8000_center`, 4-conn | pixel BFS | 930.1 | 637.3 (1.43) | 412.6 (2.26) | 299.0 (3.06) | 259.8 (3.52) | 0.22 |
| `input_blobs` | tile labels | 97.84 | 52.49 (1.89) | 29.83 (3.26) | 28.59 (3.49) | **21.06 (4.65)** | 0.29 |
| drones, per frame | tile labels | 1.039 | 0.918 (1.15) | 0.804 (1.34) | 0.807 (1.32) | 0.792 (1.35) | 0.08 |
| `sq_256` | tile BFS | 0.0759 | **0.0461 (1.66)** | 0.0475 (1.64) | 0.0572 (1.54) | 0.0563 (1.42) | 0.09 |
| `sq_256` | pixel BFS | 0.1999 | 0.2022 (0.99) | 0.2054 (0.98) | 0.2212 (0.86) | 0.2371 (0.82) | 0.05 |
| `serpentine_256` | tile BFS | 0.1650 | 0.1346 (1.23) | **0.1322 (1.29)** | 0.1722 (1.04) | 0.2139 (0.76) | 0.05 |
| `serpentine_256` | pixel BFS | 0.6728 | 0.6914 (0.98) | 0.6924 (0.97) | 0.8216 (0.86) | 0.9788 (0.69) | 0.04 |
| `dispatch_1x1`, in µs | tile BFS | 8.6 | 10.2 | 12.2 | 18.3 | 23.7 | |
| `dispatch_1x1`, in µs | pixel BFS | 5.7 | 6.6 | 7.1 | 8.7 | 12.8 | |
| `dispatch_1x1_labels`, in µs | tile labels | 14.5 | 15.2 | 18.9 | 26.3 | 39.8 | |

The drone row's speedups are derived from the per-frame samples:
paired per round, then the median over frames.

**On the big scenes the tile variants reach 4.2-4.6x**, the pixel BFS
3.5-3.8x. Two threads are nearly perfect for the tiles and the labels
(0.94 per vCPU). At 16 threads each vCPU gives them 26-29% of a
thread's work.

**From 4 to 8 threads the tile variants stall.** On the square the
tile BFS goes from 3.13x to 3.10x; on the frame the labels go from
3.26x to 3.49x. Both gain again at 16. The pixel BFS keeps climbing:
2.12x, 2.97x, 3.80x.

This was not profiled. One reading (inferred, not measured): a static
schedule gives each thread an equal chunk, and every barrier waits for
the slowest one.

At 8 threads some chunks may run on E-cores, or beside a busy
hyperthread sibling. The guest cannot see where Windows places its
vCPUs.

**Small scenes never split a BFS step.** On `sq_256` the pixel BFS's
widest level has 504 entries, under its 1,024 threshold. The tile BFS
has 3 rounds of at most 3 tiles, under its 8.

So the tiles' 1.66x at 2 threads comes from the pack, the reset and
the paint, which run in parallel at any size. On the serpentine every
round has one tile, and more threads only cost.

At 4-conn the small scenes agree. `sq_256` is fastest at 4 threads
(0.0408 ms, 1.21x faster than ch07's 0.0492). The serpentine is
fastest at 2 (0.1329 ms, 23.5x faster than ch07's 3.116).

### Method, cores, hardware

Each row walks from one core's twin to the GPU in three steps. The
method is ch09 on one thread against one core. The cores are 16
threads against one. The hardware is the GPU, same idea, against 16
threads.

| scene | one core | x method | ch09, 1 thread | x cores | ch09, 16 threads | x hardware | GPU, same idea | one core / GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sq_8000_center`, tile BFS | 1,087.4 | **12.2** | 89.33 | 4.18 | 21.29 | 6.24 | ch07: 3.411 | 319 |
| the same, 4-conn | 1,007.8 | 11.5 | 88.87 | 4.33 | 20.35 | 6.31 | ch07: 3.226 | 312 |
| `sq_8000_center`, pixel BFS | 1,087.4 | 1.10 | 986.2 | 3.80 | 259.0 | 2.56 | ch03: 101.2 | 10.7 |
| `input_blobs`, tile labels | 271.5 | 2.75 | 97.84 | 4.65 | 21.06 | **9.43** | ch08 first version: 2.232 | 122 |
| drones, tile labels, per frame | 1.009 | 0.97 | 1.039 | 1.35 | 0.792 | 8.66 | ch08 first version: 0.0911 | 11.1 |

The three steps multiply to the last column within 2%. They are
medians of paired rounds (or of frames), not one chain of ratios. The
drone row's cores step is derived from the per-frame samples.

**On the big blob the method is the largest step.** Tiles on one
thread are 12.2x one core; the 16 threads add 4.18x, the GPU 6.24x.

A round closes a 64 x 64 tile with word operations. The square needs
64 rounds and 31,249 tile visits, where the pixel BFS needs 4,001
levels over 64M pixels.

Without tiles, the pixel BFS gains 1.10x from the method and 3.80x
from the cores. On 16 threads (259.0 ms) it is still 2.9x slower than
the tiles on one (89.33 ms, derived).

**On the frame the hardware is the largest step** (9.43x), then the
cores (4.65x), then the method (2.75x). One core's labels are cheaper
per pixel here: 271.5 ms for 81 Mpx, against 1,087.4 ms for the 64 Mpx
square. So the tiles have less to save.

Without the paint the picture is the same. Against one core's labels
alone (239.9 ms), one thread is 2.69x faster and 16 threads (18.21
ms) 13.5x.

**On the drone frames the hardware is everything.** The tiles on one
thread are a little slower than one core (0.97x), and 16 threads give
1.35x. A frame has 570-1,210 blobs on 0.59 Mpx: little work to split.

### Where the CPU wins

**The serpentine: a one-tile front.** All 388 rounds of the tile BFS
have one tile, so none is split. The CPU runs them back to back in
0.132 ms at 4 threads.

ch07 needs 904 rounds of 32 x 32 tiles and 2.962 ms: the CPU is 22.4x
faster. Even one core's twin (0.470 ms) is 6.3x faster than ch07.
ch03's pixel BFS takes 179.0 ms for its 32,641 levels.

**The small square, once copies count.** By events it is a tie:
ch09's tiles at 2 threads take 0.0461 ms, ch07 0.0451 (1.02x).

With its copies ch07 takes 0.503 ms, so the CPU is 10.9x faster. One
core alone (0.1995 ms) is 2.5x faster (derived).

Even with the image already on the GPU, the host waits 0.305 ms for
ch07, from launch to synchronize. That is 6.6x ch09's best.

**The drones, once copies count: a tie.** With its copies,
ch08's first version takes 0.776 ms per frame. ch09 at 16 threads
takes 0.792 ms.

The median per-frame ratio is 1.02 (p90 1.22) in the GPU's favour. The
CPU is faster on 104 of the 240 frames; at 4 threads, on 102.

v5 found one core faster than the GPU end to end on the drones. Here
one core without the paint (0.797 ms) ties too (1.03). Like for like,
with the paint (1.009 ms), the GPU is 1.29x faster end to end.

**The big scenes, once copies count.** ch07 fills the square in 3.411
ms, but 40.31 ms with the pageable copies: 192 MB in, 192 MB back.
ch09's 16 threads take 21.29 ms, 1.89x faster.

On the frame (243 MB each way) the copies take ch08's first version
to 49.30 ms. ch09's labels take 21.06 ms, 2.34x faster. Already at 4
threads (29.83 ms) they win.

### The GPU with its copies

| scene | GPU cell | events | host wall | end to end | ch09, 16 threads | ch09, best count |
|---|---|---:|---:|---:|---:|---:|
| `sq_8000_center` | `ch07_c8` | 3.411 | 3.251 | 40.31 | 21.29 | 21.29 (16) |
| `sq_8000_center` | `ch03_conn8` | 101.2 | 99.50 | 137.0 | 259.0 (pixel BFS) | 259.0 (16) |
| `input_blobs` | `ch08_old` | 2.232 | 2.642 | 49.30 | 21.06 | 21.06 (16) |
| drones, per frame | `ch08_old` | 0.0911 | 0.281 | 0.776 | 0.792 | 0.792 (16) |
| `sq_256` | `ch07_c8` | 0.0451 | 0.305 | 0.503 | 0.0563 | 0.0461 (2) |
| `serpentine_256` | `ch07_c8` | 2.962 | 3.499 | 3.698 | 0.214 | 0.132 (4) |

**Events**: CUDA events around the pipeline, the image restored and
the L2 flushed off the clock. **Host wall**: launch to synchronize,
the image already on the GPU, no gap before it.

**End to end**: a pageable host-to-GPU copy, the pipeline, the copy
back (bench_v5's e2e), 5 repeats after the timed block. Its output is
checked too.

Where the GPU's data lives decides as much as its kernel. By events
the GPU leads on every scene but the serpentine (`sq_256` is a tie).
Its launch path alone loses it `sq_256`.

With the image in host memory, the GPU leads only on the pixel BFS
and, barely, on the drones.

ch08's default engine (the dense-tile path) is the slower one on these
small frames: 0.1137 against 0.0911 ms by events. On `input_blobs` the
two tie (2.261 against 2.232 ms).

### Fixed costs and cold caches

**Fixed costs.** On a 1 x 1 image a call measures its fixed cost. One
core's twins take 0.5-0.8 µs.

ch09 takes 5.7-14.5 µs on one thread and 12.8-39.8 µs on 16: the
checks, the thread count, the njit dispatches and the forks.

The GPU's fixed cost is larger. On `sq_256` the host waits 0.305 ms
for a 0.0451 ms pipeline: about 0.26 ms of launch path (derived).

**Cold caches.** On the scenes that fit the 24 MB L3, cold twins flush
every cache before each call, as the GPU flushes its L2. On `sq_256`
and the serpentine, on one thread (one core's twin included), they
are 1.1-3.0x slower than the hot cells.

Cold, ch09's tiles on one thread take 0.227 ms on `sq_256`. ch07's
kernel is 5.0x faster, but with its copies still 2.2x slower
(derived).

At 16 threads the cold cells are 1.5-10.5x slower than the hot ones,
and slower than at one thread (`sq_256` tiles: 0.494 against 0.227
ms). The flush also parks the pool, so each cold 16-thread call wakes
15 workers too.

On the drones, cold costs little: 0.846 against 0.792 ms per frame at
16 threads, 1.102 against 1.009 for one core.

### The work, counted

| scene | ch09 | GPU twin |
|---|---|---|
| `sq_8000_center`, pixel BFS | 4,001 levels, 3,873 split into chunks; 212 duplicates in one 16-thread call | ch03: 4,001 levels, 202.6M CAS attempts |
| `sq_8000_center`, tile BFS | 64 rounds (63 split), 15,625 tiles, 31,249 visits (15,624 idle), at most 984 tiles a round | ch07 (32 x 32 tiles): 127 rounds |
| `serpentine_256` | tile BFS: 388 rounds of one tile; pixel BFS: 32,641 levels of at most 2 pixels; nothing split | ch07: 904 rounds; ch03: 32,641 levels |
| `sq_256` | tile BFS: 3 rounds, at most 3 tiles; pixel BFS: 65 levels, at most 504 entries; nothing split | ch07: 4 rounds |
| `input_blobs`, lattice 8 | 7,690 seeds, 16,200 ring pieces (280 interior), 5 rounds (4 split), 20,028 meetings at 16 threads (19,940 at one), 9,314 of 19,881 tiles painted from the mask | ch08 first version: 6 rounds |

**Lattice 0 or 8.** Without lattice seeds the labels at 16 threads
take 0.970 of the lattice-8 time (range 0.806-1.106). Both need 5
rounds; lattice 0 records 15,321 meetings against 20,028. A tie.

### How the session was protected

**The order.** GPU blocks first, then 60 s of idle, then the CPU
passes, scene by scene. A CPU pass between GPU blocks would let the
GPU clock fall.

**The idle gate.** Before each of the 42 CPU rounds, the rest of the
machine had to be at most 3% busy over one second. Every round passed
on its first try, after 1.0 s (once 1.5 s).

No round was run again and none was flagged loaded. Over whole rounds
the rest of the machine was at most 1.7% busy.

**What the gate cannot see.** WSL2 shows no steal time, so load on the
Windows side is invisible to it. A calibration probe at the start of
each round shows that load as a ratio above 1.

The one-thread probe read 0.93-1.29 of the session's first probe,
except one round at 2.17 (`sq_256`, round 4). The all-thread probe
read 0.83-1.96, except one round at 6.18 (`sq_8000_center`, round 2).

Both rounds were kept: the probe is a record, not a redo rule. In the
square's round 2 the tile BFS at 8 and 16 threads had its slowest
round, 6-7% above its median. The medians absorb it.

**The batch meter is coarse.** Each cell's batch also gets a load
reading, but /proc/stat counts in 10 ms ticks (תקתוקים).

A batch's time on the clock is the sum of its timed calls. Batches
under 20 ms read up to 92% (167 of 481 above 3%). Batches of 300 ms or
more read a median of 0.3%, and 2 of 239 above 3%.

The worst long batch (12.3%, the pixel BFS at 2 threads on the
square) left no trace: its round median is 720.9 ms, the cell's median
722.8.

**Parked workers.** libgomp's idle workers spin for about 10 ms after
a parallel call. Before each cell's batch, and before a drone call
with fewer threads than the call before, the harness waited until they
parked.

No batch and no drone call started unsettled.

**P- and E-cores.** The guest sees 8 cores x 2 threads, with no hybrid
PMU and no cpufreq: it cannot tell a P-core from an E-core. Threads
are not bound (a binding pins vCPUs, not cores). So efficiency is
given per vCPU.

**The runtime.** The threading layer was pinned to OpenMP (libgomp
12.3.0). Niceness 0, an affinity of all 16 CPUs, and `OMP_WAIT_POLICY`,
`GOMP_SPINCOUNT`, `OMP_PROC_BIND` and `OMP_PLACES` unset. The run
recorded no notes.

**The GPU.** All five GPU blocks ran at full clocks: P0, an SM median
of 2,070 MHz, memory at 8,001 MHz. The square needed a second attempt;
the first ran in P3 with the memory at 7,001 MHz.

The GPU's gate (10%) saw at most 0.9%.

**Checks.** Every timed CPU call was checked against the oracles, off
the clock: 64,446 calls on the scenes and 12,960 on the drones, beside
every untimed call. None differed.

The GPU cells checked their untimed and last timed runs, and every end
to end output. All matched.

**The cost.** Commit `3fb710e`, a clean tree. About 15 minutes from
the session's start to its files, and a peak RSS of 5,210 MB.

## New problems and lessons

- **The method is worth more than the cores.** On the square the
  tiles on one thread gave 12.2x, and 16 threads 4.18x. The pixel BFS
  on 16 threads is still 2.9x slower than the tiles on one.
- **16 vCPUs are not 16 cores.** The tile variants reach 4.2-4.6x on
  the big scenes, and stall from 4 to 8 threads. With a static
  schedule, every barrier waits for the slowest chunk.
- **A copy can cost more than the whole job.** The square's pageable
  copies took longer than the 16-thread fill. Where the data lives
  decides the winner as much as the kernel does.
- **Small images want few threads.** Pack, reset and paint fork at
  any size, and the small scenes never split a BFS step. `sq_256` is
  fastest at 2 threads, the serpentine at 4.
- **A load meter needs a window much longer than its tick.** Short
  batches read up to 92% busy, most likely from tick counting. The
  gate and the round readings stayed at or under 1.7%.

## What is still open

- **The 4-to-8 stall** is not profiled. Uneven chunks, a dynamic
  schedule or per-thread timings would test the slow-chunk reading.
  Numba's OpenMP layer offers no dynamic schedule here, and its TBB
  layer needs a newer TBB than this system has.
- **One-tile fronts slow down with threads.** On the serpentine no
  step is split, yet the pixel BFS's fill alone takes 0.949 ms at 16
  threads against 0.673 at one.
- A likely cause (inferred, not measured): the serial levels run
  beside 15 workers still spinning after the parallel reset. A size
  threshold on pack, reset and paint, or a passive wait policy, would
  test it.
- **The thresholds are untuned**: `serial_below` (pixel 1,024 entries,
  tiles 8, labels 64), the pixel BFS's 256-entry minimum chunk, and no
  threshold on pack, reset and paint. A thread count chosen by image
  size would take `sq_256` from 0.0563 to 0.0461 ms.
- **The paint is 29% of the square's 16-thread run** (21.29 ms, 15.16
  without it). The pack is not fused, every run resets all tile state,
  and the label waves allocate their per-round scratch (`np.empty`) on
  the clock.
- **32 x 32 tiles of `uint32`** were not compared with 64 x 64 of
  `uint64`. The timed harness can now measure the swap.
- **Relaxed atomics** for the benign races, and a CAS claim variant
  to time against the stamps, were not done.
- **Dense tiles.** The pieces pass closes every piece on its own, as
  ch08's first version did. A tile of 1,000 isolated pixels costs
  1,000 closures; ch08's run CCL for dense tiles is not ported.
- **Pinned memory, and data that stays on the GPU.** The end-to-end
  cells use pageable copies, as bench_v5 did. Pinned copies, or frames
  kept on the GPU, were not timed; they would move the copy verdicts.
- **One session.** The square's one-call cells have 6 samples each,
  and the pixel BFS's 16-thread rounds spread wide (211-362 ms at
  4-conn). A second session would show the run-to-run spread.
- **The budget in `benchmark.py`'s docstring** still gives the
  estimate made before the run (20-35 min, 4.5-5 GB).

## Glossary

- **chunk** (נתח): a contiguous slice of a `prange` loop's work; one
  iteration, so one thread, owns it and everything it writes.
- **stamp** (חותמת): a chunk's id stored beside a claim, so the claim
  that survived a race can be told from the ones that lost.
- **benign race** (מרוץ תמים): a race whose every outcome gives the
  same final answer.
- **inbox** (תיבת דואר): a per-tile, per-side slot where a neighbour
  leaves its border; one writer per slot.
- **ring piece**: a piece that touches its tile's frame, so it may go
  on in a neighbour tile.
- **meeting**: two waves found next to each other; unioned at the end.
- **requeue**: an entry carried to the next round, unchanged, because
  its chunk's buffer might not hold its worst case.
- **oversubscription** (רישום יתר): more runnable threads than CPUs;
  a barrier then waits for threads the scheduler has set aside.
- **parked** (חונה): a pool thread asleep in the kernel, no longer
  spinning.
- **like-for-like twin**: the one-core loop that does the same job as
  the cell it is compared with: buffers kept, the reset and the paint
  on the clock.
- **method, cores, hardware**: the three steps from one core to the
  GPU: ch09 on one thread, ch09 on 16 threads, the GPU with the same
  idea.
- **per-vCPU efficiency**: the speedup over one thread, divided by the
  number of threads.
- **end to end**: the GPU's time with a pageable copy of the image in
  and the painted image back, timed on the host.
- **host wall**: launch to synchronize, with the image already on the
  GPU.
- **cold cell**: a cell whose every call starts with the CPU caches
  flushed and the pool parked.
- **idle gate**: a wait, before every round, until the rest of the
  machine is quiet.

## Files

```
kernels.py           bit helpers and the tile closure; the pixel BFS;
                     pack, paint and the tile BFS; the pieces, seeds,
                     waves, unions and paint of the tile labels
flood_fill.py        PixelBFS, TileBFS, TileLabels (buffers allocated
                     once, reset on the clock), flood_fill(), label(),
                     the threads argument (numba.set_num_threads)
test_correctness.py  596 tests
benchmarks/
  benchmark.py       the one-session benchmark
  test_benchmark.py  31 tests of the harness
```

The session: `results/ch09_cpu_nthreads/benchmark_results/`
`ch09_20261017T233237Z.json` and `.csv`.

## Run

```bash
nice -n 10 uv run pytest src/flood_fill_cuda/chapters/ch09_cpu_nthreads/

# harness check only (small sizes, no gate; its timings mean nothing):
uv run python -m flood_fill_cuda.chapters.ch09_cpu_nthreads.benchmarks.benchmark \
    --smoke --no-save

# the timed run: a quiet machine, NOT under nice (it refuses nice),
# holding the GPU lock if one is used (session ch09_20261017T233237Z:
# about 15 min, peak RSS 5.2 GB):
PYTHONUNBUFFERED=1 uv run python -m \
    flood_fill_cuda.chapters.ch09_cpu_nthreads.benchmarks.benchmark
```

The drone masks are git-ignored assets. From the main checkout the
default finds `video/assets/drones_mask`; elsewhere, pass
`--drones-dir`.
