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

All numbers on this page come from one session,
`tiles_20261015T114440Z.json`, unless a line names another session.
Numbers marked *derived* are arithmetic on measured values.

**The short version.** Against a pixel BFS, tiles win 20-26x on the
blobs from 11 Mpx up, and the reason is in the first two rows. Against
ch06's run-based labelling (no BFS at all), they win only where ch06's
launch shapes starve. Dropping the frontier is cheaper still.

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
| **closure in registers** | Kogge-Stone fills along the word (5 shifts) and across the lanes (5 shuffles) close a tile with no memory traffic. A solid tile needs 2 passes: one fills, one proves it |
| **border exports** | only border bits leave a tile. Plain loads filter them first (red in the target, not yet reached), so atomics touch only new bits |
| **rounds of tiles** | ch03's skeleton: a global queue, the warp-aggregated enqueue (copied), a cooperative launch, two `grid.sync` per round. A round moves the frontier one tile, not one pixel |
| **a per-tile stamp** | `atomic.max` of the round number lets a tile into a round's queue once, so ping-pong queues of `n_tiles` entries cannot overflow |
| **ch06's pack, a coalesced paint** | RGB becomes the 1-bit mask with ch06's `pack_kernel` (imported). One pass paints reached pixels (0, 0, 255), exactly like ch03 |

The pipeline is four launches, each bracketed by CUDA events:

```
pack     RGB -> 1 bit/px mask         ch06's pack_kernel
init     reached = seed bit, stamps   on the device, on the clock
rounds   cooperative tile BFS         two grid.sync per round
paint    reached -> (0, 0, 255)       one coalesced pass
```

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

The serpentine winds inside each tile, so a round covers 36.1 levels
(32,641 -> 904). The maze re-enters its tiles, so only 22.4. A zigzag on
the tile borders gains nothing inside a tile: 1.03 levels per round.

**Tiles come back.** A pixel BFS processes each pixel once. Here a tile
is visited again whenever new bits reach it after its closure.

On solid blobs that costs 1.4-1.6 visits per tile: a neighbour's bits
land after the tile's closure. The serpentine passes through each tile
once per red row: 14.1 visits per tile. The maze: 12.4.

**The step count is a property of the shape.** "About 30x fewer
barriers" holds on all six of ch03's scenes (27.8x to 36.1x). It fails
when every pixel step crosses a tile border (see the worst cases).

So does a frontier of tiles speed the BFS up? Yes, wherever the blob
lets a tile close in one visit. A step costs one barrier pair whether it
moves one pixel or one tile, and `sq_8000` needs 31.5x fewer steps.

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
set. Every painted image in this session equals ch03's, for every config
and scene.

**Size limits.** A queue entry packs a tile as `(tx << 16) | ty`, so
`TileFill` raises `ValueError` for W > 1,048,576 (32,768 tile rows) or
H > 2,097,152 (65,536 words per row). ch03 caps at `W * H < 2**31`.

## Results (RTX 4060 Laptop, 24 SMs)

**Method.** One interleaved round-robin per scene: 9 rounds, the order
reversed on odd rounds. Medians; brackets are the range of the 9
per-round ratios.

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

Every painted image matched ch03's, every ch06 fill set matched ch03's,
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

The bare pair is the lowest on the big blobs, because ch03's bare twin
fits 72 blocks where the instrumented one fits 48.

ch03's grids are 48 x 256, 120 x 128, 240 x 64 and (on the serpentine)
1 x 128. ch07's are `w2`, `w4`, `w8`, `w4_cap96` and (on the
serpentine) 1 x 32 and 1 x 64.

### ch07 against ch06's run-based CCL

ch06 never runs a BFS. It packs the mask, lists the maximal red runs of
every row, joins touching runs with union-find (איחוד-חיפוש), and
paints. It labels every blob; on these one-blob scenes that is the same
job.

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
| `w4` (default) | 72 x 128 | 288 | 143 | 16 | 3.495 (1.283) | 0.860 (0.390) | 3.916 |
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
  1 x 32 and 1 x 64). ch07 at 1 x 32 is 15% faster than at 72 x 128.
- **A register cap buys warps and costs spills.** `w4_cap96` seats 480
  warps and lost on all ten scenes.
- **The counters cost about 9 us on sq_2000** (9% of its rounds kernel,
  4% of the total). Elsewhere the bare twin is within 2%.

**Where 143 registers come from.** `ptxas` alone compiles the same PTX
to 86 registers with zero spills (8-conn; 82 at 4-conn, 81 bare). The
other ~57 arrive when the external `grid.sync()` call from libcudadevrt
is linked.

### The 4-connectivity twin

ch03's `conn4` runs at 48 x 128, which beat its 48 x 256 pin on every
scene of the fairness audit.

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
`zz_serp_1k` (one tile per round) loses 4%; at the best grids (ch03
1 x 128, ch07 1 x 32) it loses 10%: 96.36 vs 107.12 ms.

`zz_lattice_4k` puts a zigzag on every border, so ~127 tiles move per
round. A wide front does not rescue it: each tile still advances one
pixel per round, and the round grows from 4.02 to 4.91 us. It loses
12% (bare pair: 4%).

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

**ch07 beats one CPU core only when a round has many tiles.** The maze
(5 tile visits per round) is near parity. On one-tile fronts the CPU
wins: 4.2x on the serpentine and 71x on `zz_serp_1k` (1.79 vs 127.31
ms).

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

Across sessions (`094117Z` to this one), the rounds kernel on `sq_8000`
went from 1.419 to 1.283 ms, and on the serpentine from 4.191 to 3.896.

A second Numba trap sits under all of this. Its shuffle intrinsic has
only int32 and int64 overloads, and uint32 converts safely to int64. A
plain uint32 shuffle is two `SHFL`s plus 64-bit arithmetic after it;
`kernels._up` shuffles the int32 bit pattern instead.

## The measurement lessons (from the fairness audit)

The first draft of this page quoted sessions `094117Z` and `094228Z`.
An audit of that benchmark found eight problems; the benchmark and this
page now handle each one.

- **ch03's best was not the reference.** The draft timed only the
  instrumented `wc` at 48 x 256. Its bare twin at its own grid is 15%
  faster on `sq_8000` (70.51 vs 83.29 ms), so the bare pair is now the
  headline.
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

One more, found while running this session: the untimed correctness
pass is mostly host work at 64 Mpx (image copies and compares). The
clock sampler showed the GPU relaxing during it, so the per-scene clock
ramp now runs after it, right before the timed rounds.

## `ncu`: still owed

`ncu` fails with `ERR_NVGPUCTRPERM` on this machine, also after a WSL
restart. The Windows driver must allow non-admin access to the
performance counters (NVIDIA Control Panel, Developer, Manage GPU
Performance Counters), perhaps with a full Windows reboot.

No profile exists, so every bottleneck claim above rests on timings and
probes. The questions are ready: the shuffle slow path in the rounds
kernel, the dependent chain per visit, and ch06's paint occupancy.

## New problems and lessons

- **A frontier of tiles beats a frontier of pixels, not the absence of
  a frontier.** ch07 is 20-26x faster than ch03's BFS on the big
  blobs. But ch06's run CCL does the connectivity part 3.6-9.5x cheaper
  on the same blobs and 35x cheaper on the serpentine.
- **Fewer steps is a property of the shape.** Rounds are ~31x fewer than
  levels when paths run through tiles, and equal when every pixel step
  crosses a border. A zigzag on the tile grid makes ch07 lose to ch03.
- **The BFS is no longer the runtime on big blobs.** On `sq_8000` the
  rounds kernel is 37% of the pipeline. `pack` and `paint` are the RGB
  read and write, as in Chapter 6.
- **A round is a latency chain, not a barrier.** On the serpentine half
  of a round is the barrier pair, half is one visit's ~9-10 dependent
  memory trips. Spreading the queue counter did not help (the audit
  measured it); overlapping the export atomics did.
- **Warp intrinsics in loops need an explicit reconvergence.** The
  compiler's safety net is a slow path that runs silently. Only a
  one-warp micro-benchmark and the SASS exposed it.
- **A launch shape is part of the algorithm.** ch06's pack grid was
  tuned on a square. On a 32-px strip it left 7 of 8 warps idle, and
  the same kernel ran 7.7x slower.
- **Measure the baseline's best, not its default.** The bare twin at its
  own grid moved the headline from 23.8x to 20.2x.

## Open problems (Chapter 8 candidates)

1. **Many blobs: labels on `input_blobs.png`.** ch05 and ch06 solved
   the multi-blob job; ch07 fills one. The natural next step merges the
   two ideas: close each tile in registers with local labels, then join
   labels across tile borders with ch06's union-find. There would be no
   rounds at all, and the serpentine's 904 visits would become one pass.
2. **A narrow-front mode.** When a round has one or two tiles, one block
   could drain a local queue without `grid.sync`. Half of a serpentine
   round is the barrier (*derived*), so it could halve those scenes. It
   would still trail one CPU core.
3. **A shifted tiling against border-aligned paths.** Alternating the
   tile origin by 16 px between rounds puts a zigzag on one tiling's
   border inside the other tiling's tiles. Not measured.
4. **One `grid.sync` per round.** Three rotating queue counters make the
   second barrier unnecessary. It helps ch03 by the same amount per
   step, so it does not change the zigzag verdict.
5. **Fold `pack` into the rounds.** A tile could read its own RGB on its
   first visit. Untouched tiles would never be read: the thin strip
   would cost almost nothing.
6. **ch06's launch shapes on thin images.** Its pack (3.156 ms), scan
   (1.077 ms) and emit on the 524288 x 32 strip, and its paint on long
   runs.
7. **`ncu`**, still owed (above).

## Glossary

- **Tile** (אריח) - 32 rows x 32 columns of pixels, owned by one warp
  for one visit.
- **Frontier** (חזית) - the places reached in the last BFS step; here a
  set of tiles.
- **Round** (סבב) - one BFS step over the tile graph: two `grid.sync`.
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
- **Interleaving** (שזירה) - timing every config once per round, in
  turn, so clock drift hits all of them alike.

## Files

```
kernels.py           bit helpers, the rounds kernel factory, init, paint
flood_fill.py        host driver: flood_fill(), TileFill (reusable buffers),
                     plain_shape() (the pack/paint launch)
test_correctness.py  157 tests: bit helpers, oracle scenes, worst cases,
                     seed dtypes, ch03 cross-checks
benchmarks/
  benchmark.py       interleaved ch07 vs ch03 vs ch06, clocks, JSON + CSV
  probes.py          latency, shuffle, barrier and register probes
  worst_cases.py     the zigzag, lattice, maze and thin-strip scenes
```

Result files are in `results/ch07_gpu_1blob_tiles/benchmark_results/`.
`114440Z` is this page's session. `0925`/`0926` predate the `syncwarp`
fix. `0933` has no probes. `0935`/`0936` ran their first scene on a
relaxed clock. `0937`/`0938` have no `@njit` column. `0941`/`0942` are
the draft's pair, from the benchmark before the audit.

## Run

```bash
uv run pytest src/flood_fill_cuda/chapters/ch07_gpu_1blob_tiles/
uv run python -m flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks.benchmark
```
