# Chapter 7 - one blob, a frontier of tiles (`ch07_gpu_1blob_tiles/`)

**Keep the BFS, change the step.** Chapter 3's best kernel spreads one
blob one pixel per level. Every level costs two grid-wide barriers.

This chapter keeps the frontier, the queue and the barriers. But one
step now closes a whole 32x32 tile in registers.

| `sq_8000_center` (64 Mpx, 8-conn) | ch03 `wc` | ch07 | ch06 runs (CCL) |
|---|---:|---:|---:|
| BFS steps (levels / rounds) | 4,001 | **127** | none |
| `grid.sync` calls (2 per step) | 8,002 | **254** | 0 |
| atomics per filled pixel | 3.29 CAS | **0.080** (OR + stamp) | |
| pipeline, bare twins at their own grids | 70.51 ms | **3.493 ms (20.2x)** | |
| pipeline, instrumented, ch03 at its 48 x 256 pin | 83.29 ms | **3.495 ms (23.8x)** | 2.378 ms |

This table and the Results sections measure the chapter's first
kernel, in session `tiles_20261015T114440Z.json`.

Today's default has its own section, [One barrier per round, and a tile
paint](#one-barrier-per-round-and-a-tile-paint), measured in session
`tiles_20261016T064724Z`. Any other session is named where it is quoted.

Numbers marked *derived* are arithmetic on measured values.

**The short version.** Against a pixel BFS, tiles win 20-26x on the
blobs from 11 Mpx up, and the reason is in the first two rows. Against
ch06's run-based labelling (no BFS at all), they win only where ch06's
launch shapes starve. Dropping the frontier is cheaper still.

**Today's default** runs one `grid.sync` per round instead of two, and
paints only the tiles the fill touched. It takes `sq_8000` in 3.107 ms
against the first kernel's 3.457 (1.11x, same session), and wins on all
12 scenes, 1.11-1.34x. Against ch03's best grid that is 22.7x on
`sq_8000`.

## Inherited problems

1. **The barrier ritual is the bill.** ch03's `wc` kernel filled
   `sq_8000` in 88.5 ms over 4,001 levels (session
   `neighbors_20260721T000108Z`). Two `grid.sync` per level is 8,002
   barriers.
2. **Every pixel pays atomics.** The same session counts 3.16
   visited-CAS attempts per filled pixel at 8-conn. Pixels are claimed,
   probed and enqueued one by one.
3. **The serpentine is ch03's worst shape.** It has 32,641 one-pixel
   levels at 8-conn, and every level pays the barrier pair. Fewer blocks
   help a little; only fewer barriers can fix it.
4. **The candidate was named twice.** Chapters 2 and 3 both listed
   "tile-based BFS": iterate inside a tile between global syncs, and
   trade barrier count for redundant tile work.

## The approach

A **tile** is 32 consecutive rows times one mask word (32 consecutive
y). One warp owns one tile per visit. Lane *i* holds row x0+*i* as two
32-bit registers: `m` (the red mask word) and `f` (the reached word).

| piece | the bet |
|---|---|
| **closure in registers** | Kogge-Stone fills close a tile with no memory traffic: 5 doubling steps each way, as shifts along the word and as shuffles across the lanes (20 shuffles per column fill). A solid tile needs 2 passes: one fills, one proves it |
| **border exports** | only border bits leave a tile. Plain loads filter them first (red in the target, not yet reached), so atomics touch only new bits |
| **rounds of tiles** | ch03's skeleton: a global queue, the warp-aggregated enqueue (copied), a cooperative launch, one `grid.sync` per round (two in the first kernel). A round moves the frontier one tile, not one pixel |
| **a per-tile stamp** | `atomic.max` of the round number lets a tile into a round's queue once, so ping-pong queues of `n_tiles` entries cannot overflow |
| **ch06's pack, a tile paint** | RGB becomes the 1-bit mask with ch06's `pack_kernel` (imported). The paint gives one warp to each tile the fill touched and skips the rest; reached pixels become (0, 0, 255), exactly like ch03. The first kernel painted every word in one coalesced pass |

The pipeline is four launches, each bracketed by CUDA events:

```
pack     RGB -> 1 bit/px mask         ch06's pack_kernel
init     reached = seed bit, stamps   on the device, on the clock
rounds   cooperative tile BFS         one grid.sync per round
paint    reached -> (0, 0, 255)       one warp per touched tile
```

The first kernel, which the Results sections measure, ran two
`grid.sync` per round and painted in one coalesced pass over every word.

### The closure

All in registers; every lane runs every step:

```
row fill      gen = f; pro = m
              for k in 1, 2, 4, 8, 16:  gen |= pro & (gen << k); pro &= pro << k
              (and the mirror with >>)
diagonal      8-conn only: f |= m & (up(dil(f)) | down(dil(f)))
              dil(f) = f | f<<1 | f>>1, up/down = shuffle by one lane
column fill   the same Kogge-Stone across lanes, with shfl_up / shfl_down
repeat while any_sync(f changed)
```

Each lane exports to at most three words. Its own row's words ty-1 and
ty+1 take bit 0 and bit 31. Lanes 0-2 and 29-31 take the side rows x0-1
and x0+32, including the 8-conn corner words.

The three export atomics are issued together, and their old values are
tested afterwards. The owner writes its own word back with an atomic OR,
never a store: a neighbour may be OR-ing into the same word.

## Is it still BFS?

Start from what BFS is. A **frontier** (חזית) is the set of places
reached in the last step. One step visits everything the frontier
touches, and that becomes the next frontier.

ch03 runs this on the **pixel graph**. A pixel is a node, its 8
neighbours are its edges, and one level moves the wave one pixel.

ch07 runs the same loop on the **tile graph**. A node is a 32x32 tile,
and its edges go to the 8 tiles around it. The queue holds tiles, one
round is one step, and the stamp admits a tile once per round. So yes:
it is BFS, with a frontier of tiles.

Three things differ from "the same wave, 32 px thick".

**The thickness is not fixed.** Inside a tile the closure runs to a
fixed point (נקודת שבת), so the wave goes as far as the blob goes
within that tile. On solid squares that is about one tile:
`sq_8000` needs 4,001 levels and 127 rounds, 31.5 levels per round.

The serpentine turns only in its end tiles, where one visit takes a
whole U-turn. So a round covers 36.1 levels (32,641 -> 904), more than
a tile's 32. The maze re-enters its tiles, so only 22.4.

A zigzag on the tile borders gains nothing inside a tile: 1.03 levels
per round.

**Tiles come back.** A pixel BFS processes each pixel once. Here a tile
is visited again whenever new bits reach it after its closure.

On solid blobs that costs 1.4-1.6 visits per tile: a neighbour's bits
land after the tile's closure. The maze: 12.4.

The serpentine has 16 red rows per tile. Its 48 interior tiles are
visited once per red row (16 each). Its 16 end tiles take a U-turn,
two rows, per visit: about 8.5 each (*derived*). That averages 14.1.

**The step count is a property of the shape.** "About 30x fewer
barriers" holds on all six of ch03's scenes (27.8x to 36.1x). It fails
when every pixel step crosses a tile border (see the worst cases).

So does a frontier of tiles speed the BFS up? Yes, wherever the blob
lets a tile close in one visit. A step costs the same barrier ritual
whether it moves one pixel or one tile, and `sq_8000` needs 31.5x fewer
steps.

## What is given up, and why the fill is still exact

**Per-pixel BFS depth.** ch03 records each pixel's level: its distance
from the seed, ring by ring. ch07 cannot.

Inside a tile the fills move bits by doubling (1, 2, 4, 8, 16 places),
not ring by ring. A pixel 30 steps away is reached in the same pass as
one 1 step away. A round counts tile hops, not pixel steps.

**Exactly-once work.** Revisits (above) and the closure pass that only
proves nothing changed are redundant work a pixel BFS does not do.

**Why the fill set is still identical.** The set a flood fill paints is
the **connected component** (רכיב קשירות) of the seed. It is a property
of the image, not of the order in which pixels are found.

Two facts pin it down. First, every bit ch07 sets is red and joined to
an already-set bit through red pixels (row fill, column fill, diagonal
step, exports). So nothing outside the component is ever reached.

Second, the rounds stop only when no tile can add a bit. Then every red
neighbour of a reached pixel is reached, and a set closed like that
which holds the seed contains the whole component.

So any order ends on the same set, and the paint depends only on the
set. Every ch07 painted image in this session equals ch03's, for every
ch07 config and scene.

**Size limits.** A queue entry packs a tile as `(tx << 16) | ty`, so
`TileFill` raises `ValueError` for W > 1,048,576 (32,768 tile rows) or
H > 2,097,152 (65,536 words per row). ch03 caps at `W * H < 2**31`.

## Results (RTX 4060 Laptop, 24 SMs)

These sections measure the first kernel: two `grid.sync` per round and
the plain paint, the default until session `tiles_20261016T064724Z`.
Config names such as `w4` and `w4_bare` are that kernel's.

**Method.** One interleaved round-robin per scene: 9 rounds, the order
reversed on odd rounds. Medians; brackets are the range of the 9
per-round ratios.

A speedup is a ratio of times (12.7x). A percentage is time saved or
added: "takes 15% less time" means 0.85 of the time.

Before every run, on every side: the image is restored from a device
copy, a 96 MB buffer is read (no side finds its input in the 32 MB L2),
and a GPU sleep kernel is queued. Every launch is timed with CUDA
events, so Python's launch cost is outside the interval.

ch03 runs its kernels on reusable device buffers. Its device reset is
off its clock (2.606 ms at 64 Mpx, measured; the part its painted image
needs, visited + seed, is 1.052 ms). ch07's `init` is on its clock.

A background `nvidia-smi` sampler logged the clocks. The SM clock was
2,070 MHz in every timed window; the memory clock's median was 8,001
MHz in every scene.

Every ch07 painted image matched ch03's, every ch06 fill set matched ch03's,
and every `@njit` fill count matched.

### ch07 against ch03 (8-conn)

The headline compares the bare twins (no counters), each at its own
default grid: ch03 `wc_bare` at 72 x 256, ch07 `w4_bare` at 72 x 128.

| scene | filled px | ch03 `wc_bare` ms | ch07 `w4_bare` ms | speedup | levels -> rounds |
|---|---:|---:|---:|---|---|
| sq_2000 center | 1,000,000 | 2.83 | **0.222** | **12.7x** [12.7-12.8] | 501 -> 18 |
| sq_4000 corner | 16,000,000 | 28.89 | **1.200** | **24.1x** [22.9-24.1] | 4,000 -> 126 |
| disk r=1900 | 11,340,981 | 17.35 | **0.854** | **20.3x** [17.3-20.5] | 1,901 -> 61 |
| serpentine_256 | 32,896 | 135.48 | **3.935** | **34.4x** [34.4-35.3] | 32,641 -> 904 |
| sq_6000 center | 36,000,000 | 41.87 | **2.040** | **20.5x** [20.5-20.6] | 3,001 -> 96 |
| sq_8000 center | 64,000,000 | 70.51 | **3.493** | **20.2x** [20.0-20.2] | 4,001 -> 127 |

The instrumented pair carries the counters. ch03 `wc` runs at its own
default and twin pin, 48 x 256. The best-grid columns take each side's
fastest instrumented grid on that scene.

| scene | ch03 `wc` ms | ch07 `w4` ms | speedup | ch03 best grid | ch07 best grid | best vs best |
|---|---:|---:|---|---|---|---:|
| sq_2000 center | 2.84 | 0.231 | 12.3x [12.2-12.3] | 2.84 (48 x 256) | 0.224 (`w8`) | 12.7x |
| sq_4000 corner | 30.97 | 1.211 | 25.6x [25.5-25.8] | 30.39 (120 x 128) | 1.211 (`w4`) | 25.1x |
| disk r=1900 | 19.48 | 0.860 | 22.6x [22.4-23.0] | 18.59 (120 x 128) | 0.860 (`w4`) | 21.6x |
| serpentine_256 | 121.69 | 3.916 | 31.1x [18.1-31.2] | 95.81 (1 x 128) | 3.344 (1 x 32) | 28.6x |
| sq_6000 center | 48.98 | 2.049 | 23.9x [23.7-24.0] | 45.64 (120 x 128) | 2.049 (`w4`) | 22.3x |
| sq_8000 center | 83.29 | 3.495 | 23.8x [22.7-24.0] | 77.71 (120 x 128) | 3.495 (`w4`) | 22.2x |

Three framings, one answer: 20-26x on the blobs from 11 Mpx up,
12-13x on the 1 Mpx one, 29-34x on the serpentine.

The bare pair is the lowest on the big blobs. ch03's bare twin drops
its counters and fits 72 blocks where the instrumented one fits 48.

This session does not split the two. At one grid (48 x 256), ch03's own
session `neighbors_20260721T000108Z` puts the counters alone at about
6% of the time on `sq_8000` (88.50 vs 82.88 ms, an older method).

ch03's grids are 48 x 256, 120 x 128, 240 x 64 and (on the serpentine)
1 x 128. ch07's are `w2`, `w4`, `w8`, `w4_cap96` and (on the
serpentine) 1 x 32 and 1 x 64.

### ch07 against ch06's run-based CCL

ch06 never runs a BFS. It packs the mask, lists the maximal red runs of
every row, joins touching runs with union-find (איחוד-חיפוש), and
paints. It labels every blob; on these one-blob scenes it changes the
same pixels, in its own label colour instead of blue.

| scene | ch06 ms | ch07 `w4` ms | ch07 vs ch06 | ch06 core | ch07 core |
|---|---:|---:|---|---:|---:|
| sq_2000 center | 0.411 | **0.231** | **1.77x** [1.65-1.90] | 0.043 | 0.110 |
| sq_4000 corner | **0.798** | 1.211 | 0.66x [0.65-0.80] | 0.075 | 0.711 |
| disk r=1900 | 0.964 | **0.860** | **1.12x** [1.06-1.27] | 0.075 | 0.401 |
| serpentine_256 | **0.135** | 3.916 | 0.03x [0.02-0.06] | 0.111 | 3.901 |
| sq_6000 center | 2.078 | 2.049 | 1.01x [0.94-1.06] | 0.238 | 0.848 |
| sq_8000 center | **2.378** | 3.495 | 0.68x [0.63-0.71] | 0.345 | 1.305 |

"Core" is the connectivity work alone, after the shared `pack`: ch06's
count, scan, emit, merge and flatten against ch07's init and rounds.
**ch06's core is cheaper on all six scenes**, by 2.5x (sq_2000) to 9.5x
(sq_4000) on the blobs and 35x on the serpentine (*derived*).

So ch07's two wins come from the ends, not the middle. ch06's paint
gives each warp a batch of 32 runs, and a solid blob has one long run
per row. On sq_2000 its paint takes 0.279 ms against ch07's 0.043.

The ch06 column here is its instrumented twin. Its bare twin is up to
1.18x faster (the maze); the new section pairs bare with bare.

The pack is the same kernel on both sides: 1.152 ms (ch07) and 1.153
ms (ch06) on sq_8000.

### Where ch07's time goes

| scene | pack | init | rounds | paint | total | pack + paint share | pack GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| sq_2000 center | 0.079 | 0.009 | 0.100 | 0.043 | 0.231 | 53% | 152 |
| sq_4000 corner | 0.287 | 0.011 | 0.699 | 0.211 | 1.211 | 41% | 167 |
| disk r=1900 | 0.286 | 0.011 | 0.390 | 0.173 | 0.860 | 53% | 168 |
| serpentine_256 | 0.008 | 0.005 | 3.896 | 0.006 | 3.916 | 0% | 24 |
| sq_6000 center | 0.639 | 0.015 | 0.833 | 0.559 | 2.049 | 58% | 169 |
| sq_8000 center | 1.152 | 0.022 | 1.283 | 1.035 | 3.495 | 63% | 167 |

On `sq_8000` the BFS is now the smaller part. `pack` and `paint` are a
192 MB read and a 192 MB write, at 167 and 185 GB/s (*derived*: bytes
over measured time).

That pair is the floor ch07 shares with ch06: both read the RGB once
and write it once. Chapter 6's lesson arrives from the other side: once
the algorithm is cheap, the RGB image is the runtime.

### The work, counted

| scene | tiles touched | tile visits | visits per tile | closure iters per visit | peak round (tiles) | ch07 OR per px | ch07 stamp per px | ch03 `wc` CAS per px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sq_2000 center | 1,024 | 1,422 | 1.39 | 1.72 | 161 | 0.080 | 0.0023 | 3.15 |
| sq_4000 corner | 15,625 | 22,896 | 1.47 | 1.68 | 375 | 0.079 | 0.0025 | 3.18 |
| disk r=1900 | 11,293 | 16,894 | 1.50 | 1.67 | 510 | 0.080 | 0.0025 | 3.22 |
| serpentine_256 | 64 | 904 | 14.12 | 2.13 | 1 | 0.062 | 0.0275 | 1.00 |
| sq_6000 center | 35,344 | 54,772 | 1.55 | 1.65 | 1,177 | 0.078 | 0.0026 | 3.26 |
| sq_8000 center | 62,500 | 98,195 | 1.57 | 1.64 | 1,571 | 0.077 | 0.0026 | 3.29 |

- **Atomics fall about 40x.** On `sq_8000`: 210.9M CAS attempts
  against 4.94M ORs plus 0.17M stamps (*derived* ratio: 41.3x). A word
  is OR-ed only when the plain-load filter says it gains bits.
- **The redundancy is real but small on blobs.** 1.4-1.6 visits per
  tile. Revisits depend on timing, so they move between sessions;
  rounds do not.

### The price of a round (probes, same session)

The single-warp probes count `%clock64` cycles inside the kernel, so the
clock state cancels. They ran first after the spin-up, at 2,070 MHz.

| probe | measured |
|---|---|
| barrier ritual (2 `grid.sync` + a read), ch03's 48 x 256 | 2.40 us per step |
| the same, ch07's 72 x 128 | 2.19 us per round |
| the same, 1 x 32 / 1 x 64 / 24 x 256 / 72 x 256 / 240 x 64 | 1.49 / 1.49 / 1.84 / 3.05 / 3.35 us |
| one dependent L2 load, one warp | 304 cycles (155 ns) |
| one dependent atomic OR, one warp | 335 cycles (161 ns) |
| one 8-conn closure iteration, one warp | 599 cycles (280 ns) |

*Derived* from those: on `sq_8000` ch03 spends at least 9.6 ms of 83.3
in barriers (12%). ch07 spends 0.28 ms of its 1.28 ms rounds kernel
(22%).

On the serpentine ch03 spends 78.5 ms of 121.7 in barriers (64%). ch07
spends 1.98 ms of 3.90; the other 1.92 ms are 904 tile visits, 2.12 us
each (*derived*).

A visit is about 4,390 cycles at 2,070 MHz. Its 2.13 closure iterations
take about 1,280; the rest is about 9-10 dependent memory trips of
300-335 cycles (*derived*).

So a ch07 round on the serpentine costs 4.31 us against a ch03 level's
3.73 us. It covers 36 levels instead of one.

### Grid sizes and registers

| config | blocks x tpb | warps | regs/thread | local bytes/thread | sq_8000 total (rounds) | disk total (rounds) | serpentine |
|---|---|---:|---:|---:|---|---|---:|
| `w2` | 144 x 64 | 288 | 143 | 16 | 3.538 (1.314) | 0.871 (0.399) | 4.031 |
| `w4` (then the default) | 72 x 128 | 288 | 143 | 16 | 3.495 (1.283) | 0.860 (0.390) | 3.916 |
| `w8` | 24 x 256 | 192 | 143 | 16 | 3.732 (1.469) | 0.877 (0.408) | 3.688 |
| `w4_cap96` | 120 x 128 | 480 | 96 | 48 | 3.529 (1.300) | 0.900 (0.429) | 4.620 |
| `w4_bare` | 72 x 128 | 288 | 129 | 16 | 3.493 (1.283) | 0.854 (0.381) | 3.935 |
| `1x32` | 1 x 32 | 1 | 143 | 16 | | | 3.344 |
| ch03 `wc` | 48 x 256 | 384 | 96 | 24 | 83.29 | 19.48 | 121.69 |
| ch03 `wc` | 120 x 128 | 480 | 96 | 24 | 77.71 | 18.59 | 133.49 |
| ch03 `wc` | 1 x 128 | 4 | 96 | 24 | | | 95.81 |
| ch03 `wc_bare` | 72 x 256 | 576 | 65 | 16 | 70.51 | 17.35 | 135.48 |

- **One block wins the serpentine, on both sides.** Its round is one
  tile, and one-block grids have the cheapest barrier (1.49 us at
  1 x 32 and 1 x 64). ch07 at 1 x 32 takes 15% less time than at
  72 x 128 (3.344 vs 3.916 ms).
- **A register cap buys warps and costs spills.** `w4_cap96` seats 480
  warps and lost on all ten scenes.
- **The counters cost about 9 us on sq_2000** (9% of its rounds kernel,
  4% of the total). Elsewhere the bare twin is within 2%.

**Where 143 registers come from.** `ptxas` alone compiles the same PTX
to 86 registers with zero spills (8-conn; 82 at 4-conn, 81 bare). The
other ~57 arrive when the external `grid.sync()` call from libcudadevrt
is linked.

### The 4-connectivity twin

ch03's `conn4` runs at 48 x 128. In the fairness audit it beat the
48 x 256 pin on every scene; that run is not a committed session (commit
33e2ffe records it).

| scene | ch03 `conn4` ms | ch07 `c4_w4` ms | speedup (min-vs-min) | levels -> rounds |
|---|---:|---:|---|---|
| sq_2000 center | 6.55 | 0.295 | 22.2x (22.3x) | 1,001 -> 33 |
| sq_4000 corner | 50.97 | 1.672 | 30.5x (30.5x) | 7,999 -> 249 |
| disk r=1900 | 23.49 | 0.897 | 26.2x (26.3x) | 2,688 -> 85 |
| serpentine_256 | 123.41 | 3.748 | 32.9x (32.9x) | 32,896 -> 904 |
| sq_6000 center | 70.50 | 2.261 | 31.2x (31.2x) | 6,001 -> 189 |
| sq_8000 center | 123.34 | 3.736 | 33.0x (33.0x) | 8,001 -> 251 |

For ch07 the two connectivities are close: 4-conn takes 0.96x to 1.38x
the 8-conn time. A closed tile already spans 32 pixels in every
direction.

### Where tiles degrade: the worst cases

Four scenes from the chapter's adversarial review (`worst_cases.py`).
Each is one blob; every red pixel is filled.

| scene | filled px | levels -> rounds | tile visits per round | ch07 us per round | ch03 us per level | ch03 `wc` ms | ch07 `w4` ms | ch07 vs `wc` | `@njit` ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `zz_serp_1k` | 32,580 | 32,550 -> 31,620 | 1.0 | 4.02 | 3.76 | 122.34 | 127.31 | **0.96x** | 1.79 |
| `zz_lattice_4k` | 1,008,126 | 3,939 -> 3,938 | 126.9 | 4.91 | 4.42 | 17.43 | 19.87 | **0.88x** | 165.44 |
| `maze_1k` | 524,287 | 56,976 -> 2,544 | 5.0 | 8.07 | 4.01 | 228.43 | 20.59 | 11.1x | 27.44 |
| `thin_h32_blob` | 1,024 | 17 -> 2 | 1.0 | | | 0.09 | 0.601 | **0.15x** | 68.50 |

**Zigzags on the tile borders.** The path steps diagonally across a
tile border at every pixel: (32k+31, y), (32k+32, y+1), (32k+31, y+2).
Inside a tile it is isolated pixels, so the closure adds nothing and
every level is an export.

Then rounds equal levels, and a round costs more than ch03's level.
`zz_serp_1k` (one tile per round) takes 4% more time. At the best grids
(ch03 1 x 128, ch07 1 x 32) it takes 11% more: 96.36 vs 107.12 ms.

`zz_lattice_4k` puts a zigzag on every border, so ~127 tiles move per
round. A wide front does not rescue it: each tile still advances one
pixel per round, and the round grows from 4.02 to 4.91 us. It takes 14%
more time (bare pair: 5%).

With one `grid.sync` per round the default now ties `zz_serp_1k` and
wins `zz_lattice_4k` against ch03's best grid. But ch03 still pays two
barriers per level (see the new section).

**The maze** is tortuous but realistic: 12.4 visits per tile and 6.53
closure iterations per visit. It still wins 11.1x, at 8.07 us per
round.

**The thin strip** (524288 x 32, a 32x32 blob) is all `pack` and
`paint`: 0.564 of 0.601 ms. ch03's kernel touches only the blob (0.09
ms); its 0.701 ms reset is off its clock.

That strip once cost far more. ch06's pack launch (`(wpr*32/256, 64)`
blocks of 256) leaves 7 of 8 warps idle at one word per row. ch07 now
sizes the block by the row's words and keeps ~8,192 warps busy.

The same pack kernel takes 3.156 ms in ch06's shape and 0.410 ms in
ch07's, on this scene. ch06 still uses its shape (5.47 ms total).

### Against the CPU

The `@njit` oracle is one CPU core, end to end: it allocates its
arrays, computes a depth map and does not paint. "Kernel-only" means the
image is already on the GPU. "End to end" adds the pageable host-to-GPU
copy and the copy of the painted image back.

| scene | `@njit` 8-conn ms | ch07 kernel-only ms | x | ch07 end to end ms | x |
|---|---:|---:|---:|---:|---:|
| sq_2000 center | 43.93 | 0.231 | 190x | 3.56 | 12.3x |
| sq_4000 corner | 613.07 | 1.211 | 506x | 13.96 | 43.9x |
| disk r=1900 | 431.56 | 0.860 | 502x | 11.88 | 36.3x |
| serpentine_256 | 0.94 | 3.916 | 0.24x | 4.48 | 0.21x |
| sq_6000 center | 1521.60 | 2.049 | 743x | 27.01 | 56.3x |
| sq_8000 center | 2700.07 | 3.495 | 773x | 46.18 | 58.5x |
| `zz_lattice_4k` | 165.44 | 19.87 | 8.3x | 30.82 | 5.4x |
| `maze_1k` | 27.44 | 20.59 | 1.33x | 22.08 | 1.24x |
| `thin_h32_blob` | 68.50 | 0.601 | 114x | 12.77 | 5.4x |

**When the fill is the oracle's work, ch07 beats one CPU core only
when a round has many tiles.** The maze (5 tile visits per round) is
near parity.

On one-tile fronts the CPU wins: 4.2x on the serpentine and 71x on
`zz_serp_1k` (1.79 vs 127.31 ms).

The thin strip is a one-tile front too, yet ch07 wins 114x. Its blob is
1,024 px, but the oracle allocates full-image arrays (visited, depth, a
W x H queue) for 16.8 Mpx. That setup, not the fill, decides its time.

`zz_serp_1k` has no end-to-end row: its perf_counter clock read below
its event clock (125.5 vs 127.3 ms).

## The lesson that took the rounds kernel from 4.19 ms to 1.42 ms

The first version was correct and already 13.5x faster than `wc` on
`sq_8000` (session `tiles_20261015T092605Z`). But a one-warp probe
found the 20-shuffle column fill slow.

It takes 6,520 cycles, against 429 with one `cuda.syncwarp()` in
front: **15.2x**, with identical results (this session's probe).

The SASS explains it. Inside a loop, ptxas cannot prove the warp is
converged. So every shuffle gets a runtime test and a slow path, a call
to `__cuda_sm70_shflsync_up_p`.

After the loop's back-edge the warp tests as diverged, and all twenty
shuffles take the slow path. An explicit `syncwarp()` reconverges the
warp, and the fast `SHFL` runs.

The same grid (72 x 128, uncapped) before and after the fix, from
sessions `tiles_20261015T092605Z` (config `w4_nocap` there) and
`tiles_20261015T094117Z`:

| scene | before: total (rounds) | after: total (rounds) | rounds kernel speedup |
|---|---|---|---:|
| sq_2000 center | 0.352 (0.237) | 0.226 (0.111) | 2.14x |
| sq_4000 corner | 2.677 (1.928) | 1.295 (0.740) | 2.60x |
| disk r=1900 | 1.539 (1.034) | 0.965 (0.424) | 2.44x |
| serpentine_256 | 11.047 (11.030) | 4.210 (4.191) | 2.63x |
| sq_6000 center | 3.988 (2.718) | 2.180 (0.913) | 2.98x |
| sq_8000 center | 6.882 (4.187) | 3.956 (1.419) | 2.95x |

The fix also flipped the register question. Before it, an 80-register
cap (576 warps) beat no cap on `sq_8000`: 6.214 ms against 6.882
(session `092605Z`). After it, the cap's spills cost more than its
warps buy.

Two smaller edits followed, each measured in an interleaved A/B with
identical images. The 8-conn side broadcast became one converged
shuffle; the three export atomics are now issued before any result is
used.

Those A/B runs are not committed sessions. Their numbers are in the
bodies of commits 5dce49c and 9fe81ac.

Across sessions (`094117Z` to this one), the rounds kernel on `sq_8000`
went from 1.419 to 1.283 ms, and on the serpentine from 4.191 to 3.896.

A second Numba trap sits under all of this. Its shuffle intrinsic has
no uint32 overload (int32, int64, float32 and float64 only), and uint32
converts safely to int64. A plain uint32 shuffle is two `SHFL`s plus
64-bit arithmetic after it; `kernels._up` shuffles the int32 bit
pattern instead.

## The measurement lessons (from the fairness audit)

The first draft of this page quoted sessions `094117Z` and `094228Z`.
An audit of that benchmark found eight problems; the benchmark and this
page now handle each one.

- **ch03's best was not the reference.** The draft timed only the
  instrumented `wc` at 48 x 256. Its bare twin at its own grid takes 15%
  less time on `sq_8000` (70.51 vs 83.29 ms), so the bare pair is now
  the headline.
- **"The comparison favours ch03" was wrong.** ch03's off-clock reset
  (2.606 ms at 64 Mpx) is smaller than that bare-twin gap.
- **A CPU pass between scenes relaxed the GPU.** The `@njit` oracle ran
  after each scene, and its seconds of CPU work let the clock fall (both
  sessions read 210 MHz right after the last one). The two draft
  sessions disagreed on the disk: 20.6x and 16.3x.
- **5 rounds were not enough.** Their medians also disagreed by 7% on
  `sq_8000` (21.7x vs 23.2x) and 8% on `sq_6000` (23.0x vs 21.2x). Now:
  9 rounds, the oracle last, a clock sampler, and per-round ratio ranges.
- **A warm L2 helped ch07 on small images.** The restore copy left
  sq_2000's 12 MB image in L2, and `pack` read it at 186 GB/s (`094117Z`).
  With the flush it reads at 152 GB/s.
- **Single-warp probes relaxed the clock under themselves.** The draft's
  294 ns L2 load was taken at about half clock: 304 cycles in 294 ns is
  about 1,030 MHz (*derived*). Cycles do not move, so they are recorded.
- **The CPU table mixed scopes.** It divided an end-to-end `@njit` call
  by a kernel-only GPU time. Both GPU scopes are now shown and labelled.
- **"The BFS kernel alone: 60.4x" was not like for like.** It set ch03's
  whole kernel (RGB reads, paint, depth and owner stores) against ch07's
  rounds kernel only. The row is gone.

One more, found in the first full session after the audit (`114021Z`,
not committed; commit 6b52ec5 records it). The untimed correctness pass
is mostly host work at 36-64 Mpx (image copies and compares).

The clock sampler showed the GPU relaxing during it, so the per-scene
clock ramp now runs after it, right before the timed rounds. This
page's session reads 2,070 MHz in every timed window.

## One barrier per round, and a tile paint

Session `tiles_20261016T064724Z`: 12 scenes, 9 interleaved rounds, the
timing policy above. Every painted image equals ch03's.

The scenes are the ten above plus two:

- `tiny_blob_8k`: a 5,025 px disk (radius 40) in a white 8000 x 8000
  image, so `pack` and the paint dominate;
- `sq_2000_corner`: a full red 2000 x 2000 image seeded in a corner, a
  front narrower than the grid.

**What changed.** `flood_fill()` now runs two switches by default:

- `one_sync`: one `grid.sync` per round instead of two (open problem 4
  of the first version);
- `tile_paint`: the paint skips every tile the fill never touched, and
  gives each touched tile one warp.

Two more switches exist and stay off. `fused_pack` reads each tile's
RGB inside the rounds kernel (open problem 5). `skip_full` drops
exports into a full tile that is already queued.

**Why.** The first `ncu` profile (next section) put 50-54% of every
one-tile round in the barrier ritual. On the thin strip, 74% of the
paint's samples waited on loads of `reached` words that were almost all
zero.

### The protocol, in plain words

**One barrier per round.** The old round ended with two barriers.
Between them, every thread read the next round's size, and thread 0
zeroed the counter just used up. The second barrier kept anyone from
appending to that counter before everyone had read it.

Three counters now take turns. In round *r*, counter *r* + 1 (mod 3)
collects the next round's tiles, and every thread reads it right after
the round's one barrier. Before that barrier, thread 0 zeroes counter
*r* + 2.

So a counter is zeroed one full barrier after its last read, and one
full barrier before anyone appends to it again. Every thread reads the
same size, so every thread runs the same number of barriers. The
kernel's docstring follows one counter through its cycle.

**The tile paint.** A bit lands in a tile only through an atomic that
also raises the tile's stamp: that is how a tile enters a round. So a
tile whose stamp is still -1 holds no reached bit, and the paint skips
it after one load.

A touched tile gets one warp. Each lane loads its row's word before
any store, then the warp paints the set rows one by one, 96 contiguous
(רציפים) bytes each. A test checks the premise after a run in every
switch mode: the tiles holding a reached bit are exactly the tiles with
stamp >= 0.

**The fused pack** (opt-in). There is no pack launch. The warp that
first enqueues a tile reads the tile's 32 x 32 RGB pixels and stores its
mask, one round before the tile's first visit.

An export into a tile whose mask is not stored yet cannot be filtered.
It goes out as a **candidate** (מועמד): border bits that may not be red.
The owner ANDs them with the mask at its visit, and so does the paint.

### Before and after

Bare twins at their own grids (72 x 128; the fused pack 48 x 128).
Paired per-round ratio, median [range of 9]; above 1, the new default is
faster.

| scene | first kernel ms | default ms | default vs first | fused pack ms | fused pack vs first |
|---|---:|---:|---|---:|---|
| sq_2000 center | 0.224 | **0.184** | **1.22x** [1.18-1.23] | 0.164 | 1.37x [1.32-1.38] |
| sq_4000 corner | 1.188 | **1.024** | **1.16x** [1.14-1.17] | 1.238 | 0.96x [0.94-0.97] |
| disk r=1900 | 0.871 | **0.768** | **1.12x** [1.03-1.32] | 0.826 | 1.06x [1.03-1.25] |
| serpentine_256 | 3.843 | **2.937** | **1.31x** [1.30-1.31] | 2.988 | 1.29x [1.28-1.29] |
| sq_6000 center | 2.020 | **1.801** | **1.12x** [1.05-1.12] | 1.940 | 1.04x [0.98-1.05] |
| sq_8000 center | 3.457 | **3.107** | **1.11x** [1.05-1.14] | 3.246 | 1.07x [1.01-1.09] |
| `zz_serp_1k` | 125.59 | **93.88** | **1.34x** [1.33-1.34] | 93.40 | 1.35x [1.34-1.35] |
| `zz_lattice_4k` | 19.62 | **15.56** | **1.26x** [1.26-1.35] | 18.48 | 1.07x [1.05-1.13] |
| `maze_1k` | 20.48 | **17.97** | **1.14x** [0.35-1.14] | 18.27 | 1.12x [1.11-1.12] |
| `thin_h32_blob` | 0.612 | **0.481** | **1.28x** [1.24-1.31] | 0.053 | 11.48x [8.94-11.89] |
| `tiny_blob_8k` | 1.650 | **1.272** | **1.30x** [1.29-1.34] | 0.127 | 12.96x [12.26-13.22] |
| sq_2000 corner | 0.445 | **0.374** | **1.19x** [1.19-1.20] | 0.481 | 0.93x [0.92-0.93] |

The default wins on all 12 scenes, 1.11-1.34x. The one-tile fronts
gain most, where the barrier was half of every round.

The maze's 0.35 is one round in which the default's rounds kernel took
57.8 ms instead of 17.9. That kernel is the same binary as `one_sync`
alone, so it was a disturbance, not the switch.

**Where the time went** on `sq_8000` (phase medians, ms):

| phase | first kernel | default | fused pack |
|---|---:|---:|---:|
| pack | 1.154 | 1.155 | 0.002 |
| init | 0.023 | 0.023 | 0.025 |
| rounds | 1.244 | **1.114** | 2.369 |
| paint | 1.035 | **0.808** | 0.845 |
| total (median of totals) | 3.457 | **3.107** | 3.246 |

On the thin strip the paint falls from 0.154 to 0.023 ms, and on
`tiny_blob_8k` from 0.449 to 0.075 ms. Their `pack` (0.430 and 1.154
ms) is now nearly all of the default's time.

### Each switch on its own

The first version of this change adopted `one_sync` and the fused pack
as one bundle (session `tiles_20261016T002856Z`: faster than the first
kernel on 10 of 11 scenes). A fairness audit found that the bundle hid a
loss.

So the chapter's **adoption rule** is now applied to each switch's own
increment (תוספת): on top of the switches adopted before it, a switch
must lose nothing measurable. Median [range of 9], and the rounds of 9
it lost when it lost any:

| scene | `one_sync` on the first kernel | `tile_paint` on `one_sync` | `fused_pack` on both | `skip_full` on both |
|---|---|---|---|---|
| sq_2000 center | 1.09x [1.08-1.10] | 1.12x [1.09-1.13] | 1.13x [1.08-1.14] | 0.96x [0.95-0.98], 9 lost |
| sq_4000 corner | 1.12x [1.10-1.13] | 1.03x [1.02-1.06] | 0.83x [0.81-0.84], 9 lost | 1.01x [1.00-1.04] |
| disk r=1900 | 1.09x [1.07-1.25] | 1.04x [0.96-1.06], 1 lost | 0.93x [0.92-1.04], 8 lost | 1.02x [0.72-1.12], 3 lost |
| serpentine_256 | 1.29x [1.29-1.30] | 1.01x [1.01-1.01] | 0.98x [0.98-0.99], 9 lost | 0.94x [0.94-0.94], 9 lost |
| sq_6000 center | 1.05x [1.03-1.06] | 1.07x [1.00-1.08], 1 lost | 0.93x [0.87-1.00], 8 lost | 1.05x [1.04-1.11] |
| sq_8000 center | 1.04x [1.02-1.05] | 1.07x [1.04-1.09] | 0.96x [0.95-0.96], 9 lost | 1.05x [1.00-1.08] |
| `zz_serp_1k` | 1.34x [1.33-1.34] | 1.00x [1.00-1.00], 4 lost | 1.01x [1.00-1.01] | 0.95x [0.95-0.95], 9 lost |
| `zz_lattice_4k` | 1.26x [1.25-1.35] | 1.00x [1.00-1.00] | 0.84x [0.83-0.85], 9 lost | 0.87x [0.87-0.87], 9 lost |
| `maze_1k` | 1.14x [1.14-1.14] | 1.00x [0.31-1.00], 6 lost | 0.98x [0.97-3.16], 8 lost | 0.93x [0.93-2.97], 8 lost |
| `thin_h32_blob` | 1.00x [0.99-1.04], 2 lost | 1.27x [1.24-1.31] | 8.98x [6.81-9.54] | 1.00x [0.98-1.01], 6 lost |
| `tiny_blob_8k` | 1.00x [0.99-1.01], 2 lost | 1.29x [1.28-1.34] | 10.00x [9.13-10.18] | 1.00x [0.99-1.01], 5 lost |
| sq_2000 corner | 1.17x [1.16-1.17] | 1.02x [1.01-1.03] | 0.78x [0.77-0.78], 9 lost | 0.93x [0.93-0.93], 9 lost |

- **`one_sync`: adopted.** 1.04-1.34x, and a tie on the two sparse
  scenes, where 2-4 rounds leave little to save.
- **`tile_paint`: adopted.** 1.02-1.29x on the eight scenes where the
  paint matters. On the serpentine and the lattice it gains a hair
  (1.010x and 1.002x) and loses no round. On `zz_serp_1k` and the maze
  it ties, with rounds on both sides of 1. Its paint phase is never
  slower, except by 0.001 ms on the serpentine.
- **`fused_pack`: opt-in.** 9.0x on the thin strip, 10.0x on
  `tiny_blob_8k` and 1.13x on `sq_2000_center`: a mostly untouched image
  is never read. But it loses 2-22% on the eight dense scenes, in 8 or 9
  of 9 rounds. Pass `fused_pack=True` when the blob is a small part of a
  large image.
- **`skip_full`: off.** It removes revisits: `sq_8000` takes 79,573
  visits instead of 98,611 (instrumented twins). It wins 1-5% on four
  big blobs, but loses 4-13% on the serpentines, the lattice, the maze
  and both 2000 px squares.

**Most of the fused pack's old "dense-blob win" was the tile paint.**
On `one_sync` alone, without the tile paint, the fused pack still loses
on seven scenes (0.80-0.99x, 8-9 rounds lost each). It wins `sq_8000`
by only 1.03x.

**Why the fused pack loses on dense blobs.** Reading RGB inside the
rounds kernel links it at 247 registers (bare), so only 48 blocks of
128 fit: 192 warps against 288. `ncu` shows 16.8% achieved occupancy
(תפוסה) against 25%, and 32,588 cycles per round on `sq_8000` against
the default's 18,493.

Its rounds phase there takes 2.369 ms: more than the default's separate
pack and rounds together (1.155 + 1.114 ms). On narrow fronts each pack
is also a memory trip on the round's critical path (נתיב קריטי), so
`sq_2000_corner` loses most (0.78x).

**Part of `skip_full`'s loss is its grid.** Its bare twin links at 128
registers, so its own grid is 96 x 128. There the one-sync ritual costs
1.36 us against 1.21 at 72 x 128 (this session's probe). That is about
0.14 of its 0.19 ms loss on the serpentine, and 0.38 of 1.27 ms on the
maze (*derived*).

### Against ch03 and ch06, measured fairly

The audit also found the cross-chapter columns flattering ch07. The
benchmark now takes **ch03's best** per scene over all its warp-coop
grids, bare and instrumented. Its bare kernel also runs at 48 x 256 and,
on the narrow scenes, at 1 x 128.

It also times **ch06 with its counters off**, the twin the bare ch07
configs pair with. The instrumented ch06 was 1.18x slower on the maze
[1.17-1.19] and 1.02-1.07x slower on the big squares.

| scene | ch03 best ms (grid, bare) | default vs ch03 best | ch06 bare ms | default vs ch06 bare |
|---|---|---|---:|---|
| sq_2000 center | 2.81 (48 x 256) | 15.26x [14.84-15.46] | 0.403 | **2.19x** [2.10-2.64] |
| sq_4000 corner | 29.11 (72 x 256) | 28.44x [28.17-30.97] | 0.807 | 0.78x [0.76-0.83] |
| disk r=1900 | 17.77 (72 x 256) | 23.14x [20.77-26.67] | 1.111 | **1.41x** [1.21-1.51] |
| serpentine_256 | 93.50 (1 x 128) | 31.83x [31.77-31.91] | 0.127 | 0.04x [0.04-0.05] |
| sq_6000 center | 42.50 (72 x 256) | 23.59x [22.00-23.72] | 1.946 | **1.08x** [1.01-1.20] |
| sq_8000 center | 70.56 (72 x 256) | 22.72x [21.48-22.83] | 2.252 | 0.73x [0.67-0.79] |
| `zz_serp_1k` | 94.25 (1 x 128) | 1.00x [1.00-1.01] | 0.205 | 0.002x |
| `zz_lattice_4k` | 17.17 (48 x 256) | **1.10x** [1.09-1.25] | 1.830 | 0.12x [0.12-0.12] |
| `maze_1k` | 226.95 (48 x 256) | 12.63x [3.93-12.66] | 0.495 | 0.03x [0.01-0.03] |
| `thin_h32_blob` | 0.077 (48 x 256) | 0.16x [0.15-0.18] | 5.621 | **11.74x** [11.39-19.74] |
| `tiny_blob_8k` | 0.169 (48 x 256) | 0.13x [0.13-0.15] | 1.321 | 1.04x [0.84-1.06] |
| sq_2000 corner | 11.03 (48 x 256) | 29.48x [29.41-29.62] | 0.427 | **1.14x** [1.07-1.21] |

**The zigzags changed sides, with a caveat.** Against ch03's best, the
default ties `zz_serp_1k`. At one block it wins it 1.09x [1.09-1.10]
(86.28 vs 94.25 ms), and it wins `zz_lattice_4k` 1.10x. The first
kernel lost both (0.96x and 0.88x above).

But every ch03 kernel still runs two `grid.sync` per level, and ch03
has no one-sync twin. This session's probes price the gap at ch03's
narrow grid, 1 x 128: 2.50 us per level against 0.89. On `zz_serp_1k`
that is about 52 ms of 94.25 (*derived*). A one-sync ch03 would win the
zigzags back.

**ch06 still wins where connectivity is the work**: `sq_8000` (0.73x),
`sq_4000_corner`, the serpentines, the lattice and the maze. ch07 wins
`sq_2000_center` (2.19x), the disk (1.41x), `sq_2000_corner`, `sq_6000`
and the thin strip (11.7x, where ch06's launch shapes starve).

**At 4-connectivity**, `flood_fill()`'s default, the instrumented new
default beats the first kernel 1.14-1.31x on all 12 scenes. Against
ch03's `conn4` it wins 26-45x on the blobs, the serpentine and the
maze. On the zigzags the 4-conn fill is one pixel, the seed, and ch03
wins (0.34x on `zz_serp_1k`, 0.06x on `zz_lattice_4k`).

On the two sparse scenes the fill is the same at both connectivities
(1,024 and 5,025 px), and ch03 wins at both: 0.41x and 0.28x at
4-conn, 0.16x and 0.13x at 8-conn (table above). There ch07's time is
mostly the whole-image `pack`.

### The price of a round, again

This session's probes, us per round (median [range of 7]):

| grid | two `grid.sync` + read | one `grid.sync` + read |
|---|---|---|
| 72 x 128 (ch07's grid) | 2.19 [2.19-2.21] | 1.21 [1.15-1.26] |
| 96 x 128 (`skip_full`'s bare twin) | 2.53 [2.52-2.54] | 1.36 [1.34-1.37] |
| 48 x 256 | 2.40 [2.39-2.43] | 1.30 [1.30-1.31] |
| 1 x 128 (ch03's narrow best) | 2.50 [2.48-2.52] | 0.89 [0.87-0.93] |
| 1 x 32 | 2.41 [2.40-2.55] | 0.86 [0.84-0.87] |

The scenes agree with the 72 x 128 row. The rounds phase lost 1.01 us
per round on the serpentine and 1.00 on `zz_serp_1k` (*derived*),
against 0.98 between the two probes.

The two-sync probe at one block is not stable between sessions: 1.49
us at 1 x 32 in session `114440Z`, 2.41 here. `ncu` counts 3,048 cycles
for it (1.47 us at 2,070 MHz). Quote the cycles, or the scenes.

**Clocks.** The SM clock's median was 2,070 MHz in every timed window.
Eight windows dipped to 930-1,620 MHz. In each, the last dip fell in
the first 9-33% of the window, about the first one to three rounds.
Every config ran through those rounds, so the paired ratios carry them.

The memory clock's median was 8,001 MHz in eight scenes, 7,001 in three
(`sq_2000_center`, the disk, the lattice) and 5,501 on the thin strip.
Every ch07 median sits within 6.3% of its minimum (the disk; 3.8% for
the bare default, the first kernel and the fused pack).

### What did not work

- **The fused pack as the default** (above): 9-13x on mostly untouched
  images, 2-22% slower on dense ones.
- **Register caps on the fused pack** (kernel docstring, one interleaved
  A/B). 168 registers (288 warps, 32 bytes of spills) tied two blobs,
  and lost 6-7% on the serpentines and 37% on `sq_8000`. 128 (384
  warps) won 9% on `sq_8000` but lost 10-12% on two blobs and 25-90% on
  the serpentines.
- **Packing on a tile's first visit** instead of one round ahead.
  Visits on `sq_8000` rose from 98K to 104K (kernel docstring): the pack
  sat inside the window in which same-round exports re-enqueue a tile.
- **One lane per row in the pack** (the plan's layout). It packed
  `sq_8000` in 1.78 ms, against 0.90 with four rows per load
  instruction (kernel docstring).
- **The fused paint** (each visit paints what it added; session
  `tiles_20261016T002856Z`): 0.69-0.94x on every scene but the thin
  strip (1.31x) and the tiny blob (1.67x).
- **`skip_full`** (above): it does remove the revisits, but its extra
  loads, votes and bigger grid cost more on narrow fronts.

### What the review found

Three bugs, each fixed with a regression test that failed first:

- **F-ordered images.** `flood_fill()` kept an F-ordered input in F
  order on the device; `chw.T` of a channel-first image is one. The
  fused pack reads the raw bytes as C-ordered (x, y, c), so it filled 0
  of a disk's 7,845 px. The upload is now C-ordered, and
  `TileFill.run()` refuses an F-ordered device image for the fused pack.
  The tile paint falls back to the plain paint, which follows strides.
- **Misaligned device views.** `d_big[1:]` can start 2 bytes past a
  word, and the fused pack's word loads then died with
  `CUDA_ERROR_MISALIGNED_ADDRESS`. Such a view is now refused (fused
  pack) or painted by the plain paint (tile paint).
- **A vote behind a race** (מרוץ). Without the fused pack, `skip_full`
  ran `stamp >= r and all_sync(FULL, ...)`. Other warps raise that stamp
  in the same round, so the lanes of one warp could disagree, and only
  some of them reached the full-warp vote.

  Natural timing never showed it. A legal delay of one thread (schedule
  jitter, ריצוד) deadlocked (קיפאון) all four test cases. Every lane
  now votes first. The test runs a jittered build of the kernel in a
  child process with a kill timeout.

## Under `ncu`, before and after

The curated profiles are in `results/ch07_gpu_1blob_tiles/ncu/` (its
README holds the tables, the setup and the commands). Both are compared
in cycles, which do not move with the clock.

- **Before**, the rounds kernel was a latency chain plus the barrier,
  not a throughput (תפוקה) problem. On `sq_8000` the L2 ran at 31.6% of
  its peak and the L2 atomic unit was 12% busy. The top stalls were long
  scoreboard (29.2%), barrier (23.2%) and wait (16.2%). Occupancy was
  25%, register-bound at 3 blocks of 128.
- **One `grid.sync`** cut the ritual from 4,517 to 2,387 cycles per
  round at 72 x 128. A serpentine round fell from 9,015 to 6,578 cycles,
  a `zz_serp_1k` round from 8,378 to 6,038, an `sq_8000` round from
  21,613 to 18,493. The ritual is still 36-40% of a one-tile round.
- **The tile paint is no latency chain.** Its long-scoreboard stall is
  5.8% on `sq_8000`, against 51.5% for the plain paint, and it is now
  issue-bound (74.7% of issue slots). On the thin strip it reads 0.1 MB
  of stamps instead of 2.1 MB of `reached` words.
- **`pack` is unchanged**: ch06's kernel, about 1.5x a pure read of its
  bytes, and now 37% of `sq_8000`'s pipeline. 56% of its samples wait
  right after a load.
- A likely cause (inferred from the source, not checked in SASS): the
  compiler cannot move a row's load above the previous row's store,
  since nothing tells it the arrays do not alias. Then each warp has one
  row in flight.
- **The fused pack** links at 247 registers: 16.8% achieved occupancy
  and 32,588 cycles per round on `sq_8000`.

## New problems and lessons

- **A frontier of tiles beats a frontier of pixels, not the absence of
  a frontier.** ch07 is 20-26x faster than ch03's BFS on the big
  blobs. But ch06's run CCL does the connectivity part 3.6-9.5x cheaper
  on the same blobs and 35x cheaper on the serpentine.
- **Fewer steps is a property of the shape.** Rounds are ~31x fewer than
  levels when paths run through tiles, and equal when every pixel step
  crosses a border. Then the cheaper barrier decides: with one
  `grid.sync` ch07 ties or wins the zigzags, against a ch03 that still
  pays two.
- **The BFS is no longer the runtime on big blobs.** On `sq_8000` the
  rounds kernel is 36% of today's pipeline (1.114 of 3.107 ms). `pack`
  and `paint` are the RGB read and write, as in Chapter 6.
- **A round is a latency chain plus a barrier.** In the first kernel
  half of a serpentine round was the barrier pair. With one `grid.sync`
  it is 36% (`ncu`), and the one visit's dependent memory trips are the
  larger part.
- **Judge each switch by its own increment.** Bundled with `one_sync`,
  the fused pack looked like a win on 10 of 11 scenes. On its own it
  lost 2-22% on every dense scene, and most of its dense-blob gain was
  the tile paint that came with it.
- **Skipping work needs a marker the algorithm already keeps.** The
  stamp already said which tiles the fill touched, so the tile paint
  skips the rest for one load each: 6.8x on the thin strip's paint.
- **A switch that saves work can lose on its grid.** `skip_full`
  removed a fifth of `sq_8000`'s visits, but its registers moved its
  bare twin to 96 blocks, where every barrier costs more.
- **A collective must never sit behind a racy load.** The `skip_full`
  vote was correct only while a warp happened to run in lockstep.
  Natural timing hid it; a legal delay of one thread exposed it.
  Schedule jitter is now part of the test suite.
- **A raw view of a buffer needs the layout it assumes.** The fused
  pack's flat view of the image assumed C order and a 4-byte-aligned
  start. The API never promised either, and the stride-aware kernels
  never needed them.
- **Warp intrinsics in loops need an explicit reconvergence.** The
  compiler's safety net is a slow path that runs silently. Only a
  one-warp micro-benchmark and the SASS exposed it.
- **A launch shape is part of the algorithm.** ch06's pack grid was
  tuned on a square. On a 32-px strip it left 7 of 8 warps idle, and
  the same kernel ran 7.7x slower.
- **Measure the baseline's best, not its default.** The bare twin at its
  own grid moved the headline from 23.8x to 20.2x. And give the
  baseline the same ritual before calling a close race.

## Open problems (Chapter 8 candidates)

1. **Many blobs: labels on `input_blobs.png`.** ch05 and ch06 solved
   the multi-blob job; ch07 fills one. The natural next step merges the
   two ideas: close each tile in registers with local labels, then join
   labels across tile borders with ch06's union-find. *(Chapter 8 took
   it up.)*
2. **A narrow-front mode.** When a round has one or two tiles, one block
   could drain a local queue without `grid.sync`. With one sync the
   ritual is still 36-40% of a one-tile round and 87-89% of its warp
   samples (`ncu`). It would still trail one CPU core.
3. **A shifted tiling against border-aligned paths.** Alternating the
   tile origin by 16 px between rounds puts a zigzag on one tiling's
   border inside the other tiling's tiles. Not measured.
4. **One `grid.sync` per round.** *Done*: the default since session
   `tiles_20261016T064724Z`, 1.04-1.34x on its own. Its twin for ch03
   is problem 9.
5. **Fold `pack` into the rounds.** *Done as an opt-in switch*
   (`fused_pack`): 9-13x on mostly untouched images, 2-22% slower on
   dense ones. Open: bring its 247 registers down to a 72-block grid,
   and take the packs off a narrow front's critical path (idle warps
   could pack ahead).
6. **ch06's launch shapes on thin images.** Its pack (3.156 ms), scan
   (1.077 ms) and emit on the 524288 x 32 strip, and its paint on long
   runs.
7. **`ncu`.** *Done* (section above).
8. **Several rows in flight in `pack`.** It takes 1.5x a pure read of
   its bytes, with one row in flight per warp, and it is now 37% of
   `sq_8000`'s pipeline.
9. **A one-sync ch03.** Every ch03 kernel still pays two barriers per
   level; the probes price that at about 52 ms of `zz_serp_1k`'s 94
   (*derived*). Only with that twin are the zigzags a fair race.
10. **`skip_full` at the default's grid.** Pinned at 72 x 128 its
    narrow-scene loss would shrink to its own extra loads and votes
    (*derived* from the probes); not measured.

## Glossary

- **Tile** (אריח) - 32 rows x 32 columns of pixels, owned by one warp
  for one visit.
- **Frontier** (חזית) - the places reached in the last BFS step; here a
  set of tiles.
- **Round** (סבב) - one BFS step over the tile graph: one `grid.sync`
  (two in the first kernel).
- **Closure** (סגור) - repeating the fills until nothing changes; the
  tile's part of the blob is then complete.
- **Fixed point** (נקודת שבת) - the state a repeated step no longer
  changes.
- **Connected component** (רכיב קשירות) - all pixels reachable from the
  seed through red neighbours; what a flood fill paints.
- **Kogge-Stone fill** - a doubling scheme: steps of 1, 2, 4, 8, 16
  move bits up to 31 places in five steps.
- **Shuffle** (ערבוב בין נתיבים) - a warp instruction that reads
  another lane's register directly, with no memory.
- **Convergence / divergence** (התכנסות / התפצלות) - whether all 32
  lanes of a warp execute together. Shuffles need it.
- **Barrier** (מחסום) - a point every thread must reach before any
  continues; `grid.sync` is the grid-wide one.
- **Union-find** (איחוד-חיפוש) - a structure that joins sets and finds
  a set's representative; ch06's engine.
- **Spill** (שפיכה) - a value moved from a register to slow local
  memory when registers run out.
- **Latency** (השהיה) - the time one operation takes from start to
  result.
- **Throughput** (תפוקה) - work done per unit of time.
- **Occupancy** (תפוסה) - resident warps per SM, against the 48 an SM
  can hold.
- **Interleaving** (שזירה) - timing every config once per round, in
  turn, so clock drift hits all of them alike.
- **Increment** (תוספת) - what one switch adds on top of the switches
  adopted before it.
- **Candidate** (מועמד) - a border bit the fused pack exports into a
  tile whose mask is not known yet; it may not be red.
- **Critical path** (נתיב קריטי) - the chain of dependent steps that
  sets a round's length.
- **Race** (מרוץ) - two threads touching the same memory, where the
  result depends on their timing.
- **Schedule jitter** (ריצוד) - a delay of one thread that the CUDA
  model allows; the tests inject it to expose code that needs lockstep.
- **Deadlock** (קיפאון) - threads that wait for each other forever.
- **Contiguous** (רציף) - stored in one unbroken run of memory.

## Files

```
kernels.py           bit helpers, the rounds kernel factory (its switches,
                     and a tests-only jitter build), init, the paints
flood_fill.py        host driver: flood_fill(), TileFill (reusable buffers),
                     the switches and their defaults, plain_shape()
test_correctness.py  247 tests, run in each of 9 switch modes: bit
                     helpers, oracle scenes, worst cases, seed dtypes,
                     memory layouts, ch03 cross-checks, the switches,
                     schedule jitter (1,959 runs pass; 264 are skipped,
                     a switch-independent test runs in one mode)
benchmarks/
  benchmark.py       interleaved ch07 vs ch03 vs ch06, the switch
                     ablation, clocks, JSON (with per-round samples) + CSV
  probes.py          latency, shuffle, barrier and register probes
  worst_cases.py     the zigzag, lattice, maze and thin-strip scenes
```

Result files are in `results/ch07_gpu_1blob_tiles/benchmark_results/`.
`064724Z` is the session of today's default; `114440Z` is the first
kernel's. `002856Z` timed the first, bundled switch ablation (the fused
paint's numbers come from it). `001509Z` is discarded: its rounds were
bimodal at a steady clock, as if another GPU context shared the GPU.

`0925`/`0926` predate the `syncwarp` fix. `0933` has no probes.
`0935`/`0936` ran their first scene on a relaxed clock. `0937`/`0938`
have no `@njit` column. `0941`/`0942` are the draft's pair, from the
benchmark before the audit.

`results/ch07_gpu_1blob_tiles/ncu/` holds the `ncu` profiles before and
after the switch change (curated tables and their README).

## Run

```bash
uv run pytest src/flood_fill_cuda/chapters/ch07_gpu_1blob_tiles/
uv run python -m flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks.benchmark
```
