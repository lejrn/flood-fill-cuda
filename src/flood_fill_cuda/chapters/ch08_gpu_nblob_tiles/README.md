# Chapter 8 - many blobs, waves over tile components (`ch08_gpu_nblob_tiles/`)

**Can ch07's tiles label many blobs, and how close do they get to ch06?**
Yes, if the BFS node is a tile component, not a tile.

On `input_blobs.png` the tile waves beat ch06's run-based labelling in
all three contracts. They stay BFS-like: labels travel as frontier
(חזית) waves, one barrier per round.

| `input_blobs.png` (81 Mpx, 2,522 blobs) | ch05 `split_L8` | ch06 runs | ch06 + ch08's pack | ch08, first version | **ch08** | ch08 / ch06 |
|---|---:|---:|---:|---:|---:|---:|
| rgb contract (RGB in, recoloured RGB out) | 47.06 | 2.567 | 2.402 | 2.378 | **2.403** | **0.890** [0.874, 0.937] |
| mask contract (packed mask in, RGB painted) | | 1.107 | | 1.086 | **1.122** | **0.954** [0.876, 1.014] |
| labelling only (no pack, no paint) | | 0.505 | | 0.380 | **0.408** | **0.807** [0.800, 0.817] |

All times are medians in ms from the final session,
`dense_20261016T054922Z.json`, unless a line names another session.
Ratios are medians of the 9 per-round ratios, with their range.

"ch08" is the default engine, the dense-tile path. "First version" is
the chapter's path before it (`flood_fill.OLD_PATH`), timed in the
same rounds.

ch06 runs at its fastest merge grid for this image (4096 blocks, swept
in the same session). ch05 is timed by CUDA events, like the GPU
sides; before this session it was its own host clock.

**A memory-clock dip.** In this scene's window the memory clock's
median was 7,501 MHz (minimum 5,501, P3), not 8,001. The totals that
start with ch08's pack, a 243 MB stream, read 0.11-0.17 ms more than in
the first dense session (`dense_20261016T021919Z`: ch08 2.255, first
version 2.209); ch06 with its own pack, 0.02 ms more.

The phase pass ran at 8,001 MHz, and its sums match that session's
within 0.02 ms. The ratios are per round, so both sides share the dip.

**The short version.** ch08 is 0.16 ms faster than ch06 in the rgb
contract (2.403 vs 2.567 ms). Most of that is the faster pack read,
which ch06 can take too.

With ch08's pack, ch06 takes 2.402 ms (0.906 of itself). Against that
ch06, ch08 is 0.982 [0.960, 1.001]: 2% faster.

The labelling is 0.10 ms faster (0.408 vs 0.505 ms). ch08's paint gives
back 0.04 ms (phase pass: K4 and K4b 0.649 ms, ch06's paint 0.606).
ch08 labels 36,863 tile comps, ch06 539,207 runs.

**Since the dense-tile path, tiles also win on noise and dot
lattices:** `dots_4k` 1.484 ms (first version 15.405, ch06 5.934) and
`random_4000` 2.287 (6.652, ch06 5.618). Section "Dense tiles" below
has the whole story.

**Tiles still lose** on long thin shapes along tile borders
(`zz_lattice_4k` 1.546 x ch06, `serpentine_2048` 1.466), on
`input_blocks` (1.459) and on small images: up to about 2000 x 2000
px ch06 is faster or ties.

The tables from here to the worst cases are the first version's
session, `waves_20261015T183401Z`.

## Inherited problems

1. **ch07 filled ONE blob.** Its BFS over 32x32 tiles closed in
   registers (אוגרים) beat ch03's pixel BFS 20x, but ch05 and ch06 label
   MANY blobs with no seeds given.
2. **ch07's own candidate was not BFS.** Its open problem 1 proposed
   local labels per tile joined by ch06's union-find, with no rounds.
   That is block CCL. This chapter keeps the rounds on purpose.
3. **Tiles came back.** In ch07 a tile was opened again whenever new
   bits reached it after an earlier opening: 98,195 openings on 62,500
   tiles in `sq_8000`, 1.57 per tile (ch07's final session
   `tiles_20261015T114440Z`).
4. **Runs beat tiles on the connectivity core.** ch06's union-find did
   the connectivity part 2.5-9.5x cheaper than ch07's rounds on every
   single blob.

## From tiles to comps

**A tile can hold a blob more than once.** A U-shaped blob that leaves a
tile and comes back gives two local components (רכיבים) in that tile.

**ch07 reopened tiles for two reasons.** The first is timing: a
neighbour's bits often landed after the tile's closure (סגור) had run.
That alone gave 1.57 openings per tile on the solid `sq_8000`, with one
comp per tile.

The second is shape: a tile with several comps was opened again when a
wave reached the next one. ch07's serpentine, 16 red rows per tile, took
14.1 visits per tile, and the maze 12.4.

Each opening ran the closure over the whole tile again. That was
repeated work on the same tile, not overlap: no pixel ever got two
different answers.

**So the node of the BFS is the comp.** K2 closes every comp once,
before any wave starts, and a claim takes the whole comp. That removes
both reasons.

Two comps are adjacent when two of their pixels are 8-adjacent across a
tile border.

Every comp is claimed exactly once. The tests assert it on every run:
`CLAIMS + SEEDS + VALVE_SEEDS == RING_COMPS`.

A **ring comp** touches the 124-pixel frame of its tile, so it may have
neighbours in other tiles. An **interior comp** touches no frame pixel:
it is a whole blob by itself, and its label is its own anchor (עוגן).

On `input_blobs`, 28,276 of 79,524 tiles hold red. They hold 36,863
comps: 36,788 ring comps and 75 interior ones, at most 8 per tile.

## The pipeline

One stream, five launches. The mask contract drops K1; the labelling
tier also drops K4.

| # | kernel | what it does | input_blobs (ms) | ch06 for comparison |
|---|---|---|---:|---|
| K1 | `pack_tiles` | RGB to 1 bit/px, read tile by tile (design 3's read), stored row-major, equal to ch06's mask bit for bit | 1.183 | pack 1.464 |
| K0 | `init` | zero the queue state and the counters | 0.005 | |
| K2 | `label` | one warp per tile: close every comp in registers, write a 32-byte record per ring comp, seed it or leave it unclaimed | 0.256 | count 0.048, scan 0.024, emit 0.221 |
| K3 | `waves` | cooperative (שיגור שיתופי): rounds of CAS claims, one `grid.sync` per round, then one union pass and two flatten passes | 0.123 | merge 0.181, flatten 0.032 |
| K4 | `paint` | `palette[label % 6]` on red, 96 contiguous bytes per row word | 0.650 | paint 0.605 |
| | **total** | one event before K1, one after K4 | **2.214** | **2.617** |

The phase times come from a separate pass, each phase in its own run
(see the method below). They sum to 2.217 ms for ch08 and 2.575 for
ch06.

In this pass the pack alone is 0.28 ms faster (1.183 vs 1.464). The
gains on this page come from the totals instead: there a shared pack is
worth 0.33 ms.

### K2: comps, records, seeds

A warp holds the tile as 32 words; lane i is row `x0+i`. The comp loop
takes the lowest remaining pixel and closes its comp with ch07's
closure.

That pixel is the min lin of what remains, so it is the comp's anchor.
K4 replays the same order.

Two exits replace ch07's proving trip. The **last-comp exit** stops when
the fill covers all that remains. The **frontier exit** stops when no
remaining pixel touches the fill (E2: closure trips 110,126 to 76,355).

A ring comp gets record `64*t + b`, where `b` is its ordinal (מספר
סידורי) among the tile's ring comps. The frame has 124 pixels, and two
ring comps need a white frame pixel between them, so `b <= 61`.

The record holds four side words, the anchor, 8 direction bits and the
label. A direction bit is set when a red pixel lies across that border.

**Seed (זרע) rule.** A ring comp with no red neighbour in a lex-smaller
tile is a seed (S2). A diagonal lattice (סריג) adds every comp of the
tiles with `(tx + ty) % 8 == 0`.

The lex-min tile of every blob holds a seed, so every blob gets one.

### K3: waves, collisions, unions

Each round, 8 lanes share one frontier entry, one lane per direction.
A lane tests the ring comps of the neighbour tile against its side word
and claims each adjacent unclaimed comp with one CAS on its label.

**One barrier (מחסום) per round.** Appends go to three rotating
counters, so no second `grid.sync` is needed to reset one.

**The hybrid lookup (new in this chapter's final version).** The scan
tests the neighbour tile's ring comps one step at a time, so a tile
with 60 ring comps costs 60 dependent steps per task.

Above `dense_min = 4` ring comps the whole warp takes the task instead:
lane j tests ring comp j, one coalesced step per 32 comps. The scan
keeps the sparse tasks, and K2 is unchanged (E13).

**Collisions (התנגשויות).** When a wave meets a comp claimed by
another wave, the pair of seed ids is recorded, but only from the SUCC
side of the edge. Every edge has exactly one SUCC side, so every
collision is seen once.

After the last round, one pass unions the pairs with ch06's `_union`
(verbatim). Two flatten passes then give each class the minimum anchor
of its comps: ch05's canonical (קנוני) label, not just an equivalent
partition.

**Escalation valve (שסתום).** If round 16 still claims comps, every
unclaimed ring comp seeds itself. The main scenes and crops need at
most 8 rounds, so it never fires there (E9).

### K4: paint

Most tiles need no closure at all. If every comp of a tile is a ring
comp and all labels give one palette colour, the whole mask word is
painted in that colour.

On `input_blobs`: 27,534 tiles painted uniform, 68 single interior comps,
674 tiles re-closed (3,552 closure iterations). Both paints write the
same 40 MB of scattered bytes: 0.650 vs 0.605 ms.

## Is this still BFS?

Start from what BFS is. A **frontier** is the set of nodes reached in
the last step. One step visits every neighbour of the frontier, and the
new nodes become the next frontier.

Here a node is a tile comp, and an edge joins two comps that touch
across a tile border. One round is one step: every frontier comp tests
its neighbours and claims the unclaimed ones. Those form the next
frontier.

**It is a multi-source BFS.** The waves start from 11,609 seeds at
once, not from one seed per blob. A comp is claimed in round r exactly
when its distance to the nearest seed is r. The tests check the round
count against a CPU BFS on every run, exactly.

On `input_blobs` the frontier holds 11,586, 14,744, 7,834, 2,306, 289
and 6 comps in its 6 rounds.

**Waves of one blob meet, and then they merge.** Two seeds in one blob
send two waves. Where they meet, the pair is recorded and later joined
by union-find (איחוד-חיפוש). ch05's `seed_merge` did the same; it was
still BFS.

The unions touch only the 30,979 edges where two waves met. That is
43% of the 72,127 comp adjacencies (E6's L = 1 row, experiments
`155420Z`), and far from ch06's 681,039 union attempts over runs
(ch06 session `runs_20260725T161448Z`).

**The dial (חוגה) shows how much BFS is left.** The lattice spacing L
moves the method between two ends.

- **L = 0 (S2 seeds only, valve off): pure waves.** 7,936 seeds, 8
  rounds, 28,852 claims. On `input_blobs` it costs the same as the
  default (0.998 of L8). On long shapes it pays one round per comp hop
  from the farthest comp to its nearest seed: 90 rounds on the disk
  (12,513 comps, 1.60x), 64 on the serpentine (64,576 comps, 1.50x).
- **L = 8 (the default): waves plus a seed lattice.** 6 rounds, 25,179
  claims, 9,162 links.
- **L = 1: every ring comp a seed.** Zero claims and one round: block
  union-find CCL over comp adjacencies. That is not BFS, and in E6 it
  was not faster (experiments `155420Z`).

**The valve is an escape hatch, not the method.** At L8 it fires only
on chains that avoid the lattice for 16 rounds: the diagonal snake and
the two staircases of E9 (the decoy one is in the worst cases below).
There it stops the waves and lets the unions finish.

So yes: on every main scene the labels travel as frontier waves over
the comp graph, and union-find only joins waves of one blob that met.

## What it gives up

**Per-pixel BFS depth.** Inside a tile the closure moves bits by
doubling, so a round counts comp hops, not pixel steps. Labels live per
comp, as ch06's live per run. The per-pixel label map is an off-clock
twin.

**Which wave wins is a race.** Labels, paint, rounds, seeds and claims
are deterministic. Which wave claims a comp is not, so the collision
count varies from run to run (30,979 in this session, 31,104 in the
draft's).

**The two labelling tiers end at different points.** ch06's ends with
a canonical label per run, which maps straight to pixels. ch08's ends
with a label per comp; its paint still re-closes the 674 tiles that hold
two or more colours.

On the dense-tile path the gap is wider: K4b rebuilds the run CCL of
every dense tile to paint it, outside the labelling tier. So on dense
scenes the labelling ratio flatters ch08, and the rgb contract is the
fair one (see "The final session" below).

**Memory per tile, not per pixel.** Every tile reserves 64 record ids,
whatever it holds: about 3.2 KB of buffers plus its mask words. That is
253 MiB at 81 Mpx, but 416 MiB (104 B/px) for a 4 Mpx 1-px strip, with
32 px per tile (*derived* from the buffer sizes).

## Results (RTX 4060 Laptop, 24 SMs)

**Method.** One interleaved round-robin per scene: 9 rounds, every
config once per round, the order reversed on odd rounds.

Before every run, on every side: the image is restored from a device
copy, the mask is packed if the contract starts from it (off the
clock), a 96 MB read flushes the L2, and a ~2 ms GPU sleep kernel is
queued.

**Totals have two CUDA events:** one before the first launch, one after
the last. Phase times come from a separate pass of 5 rounds: each phase
in its own run, with its two events around that one launch.

**ch06 runs at its best merge grid**, swept per scene (512 to 8192
blocks, bare, the merge phase alone). `ch06_m512` keeps its shipped
grid as a reference row.

ch05 runs after the GPU rounds, with a 0.5 s GPU-bound respin before
each call, timed by its own host clock in this session (the final
dense session times it by events). The `@njit` oracle runs after every
GPU scene.

Clocks: the SM clock's median was 2,070 MHz in every timed window but
`crop_7000` (2,047.5). Its minimum was 2,025 MHz or more everywhere but
`dots_4k` (see the worst cases).

The memory clock dipped to 7,001 MHz on several scenes (in P0 as well
as P3) and ran at 5,501 on `maze_1k` (P3). Its lowest sample was 405
MHz on `dots_4k`: one trace sample, the same one as that scene's 210
MHz SM minimum.

The ratios are per round, so both sides share a dip.

Every painted image equals ch06's own `RunRecolor.run`, for every
config and scene, and ch05's wherever ch05 ran.

### The six scenes

| scene | ch08 (hybrid) | scan lookup | index lookup | ch06 rgb | ch08 / ch06 | labelling only, ch08 / ch06 | ch05 (host) |
|---|---:|---:|---:|---:|---:|---:|---:|
| input_blobs | **2.214** | 2.201 | 2.265 | 2.617 | **0.862** | **0.381 / 0.500** | 47.45 |
| input_blocks | 0.254 | 0.252 | 0.227 | 0.137 | 1.851 | 0.139 / 0.043 | 1.37 |
| blob_grid_100 | **0.551** | 0.556 | 0.553 | 0.669 | **0.838** | 0.133 / 0.102 | 26.00 |
| random_4000 | 6.893 | 7.879 | 6.918 | 5.892 | 1.178 | 3.113 / 2.068 | 22.57 |
| disk_r2000 | **0.552** | 0.550 | 0.555 | 1.004 | **0.544** | 0.129 / 0.073 | 25.29 |
| serpentine_2048 | 0.558 | 0.601 | 0.421 | 0.242 | 2.309 | 0.461 / 0.037 | 19.05 |

**Where ch08 wins:** few comps per tile and big images. On the disk
and the blob grid ch08 wins the rgb contract through its paint (disk:
0.157 vs 0.615 ms): one word per row, where ch06's runs are long.

ch06 still labels both faster (disk 0.073 vs 0.129 ms). Among the six
scenes, only on `input_blobs` does ch08 win the labelling itself.

Elsewhere it also wins on the crops from 5000 on (below) and on
`strip_1px_x` (3.737 vs 21.259 ms), where each of ch06's 2.1M runs is
one pixel.

**Where it loses:** many comps per tile. `input_blocks` has up to 47
comps per tile and `random_4000` up to 86, closed one after another
(label 2.697 ms on `random_4000`). Paint costs the same there (3.313 vs
3.292 ms).

The serpentine has 16 ring comps per tile in one long chain: 2.03M
signature tests. The index lookup cuts them to 129k and the time to
0.421 ms, still 1.7x ch06.

### The crop sweep

Centred crops of `input_blobs`, ch08 against ch06 (rgb), ch06 with
ch08's pack, and the labelling tiers:

| side | 1000 | 1500 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ch08 rgb (ms) | 0.095 | 0.122 | 0.169 | 0.273 | 0.443 | 0.664 | 0.966 | 1.335 | 1.756 | 2.210 |
| ch06 rgb (ms) | 0.088 | 0.123 | 0.181 | 0.330 | 0.553 | 0.835 | 1.184 | 1.583 | 2.059 | 2.551 |
| ch08 / ch06 | 1.081 | 1.000 | 0.938 | 0.830 | 0.794 | 0.795 | 0.816 | 0.835 | 0.853 | 0.866 |
| ch08 / ch06 + ch08's pack | 1.107 | 1.052 | 1.012 | 0.905 | 0.865 | 0.862 | 0.907 | 0.938 | 0.941 | 0.962 |
| labelling, ch08 / ch06 | 1.556 | 1.348 | 1.400 | 1.143 | 1.008 | 0.900 | 0.852 | 0.820 | 0.785 | 0.762 |

**There is a crossover**, as the design predicted. This first version
ties ch06 at 1500 (2.25 Mpx) and wins from 2000 on. Against the same
pack it wins from 3000 on. The dense-tile default moved it: a tie at
2000 and a win from 3000 (the final session, below).

The labelling itself crosses much later: a tie at 4000 x 4000 (1.008)
and a win from 5000 on. Below that the waves' fixed cost shows: 0.043
ms of waves at 1000 against ch06's merge of 0.015.

### The work, counted (input_blobs, L8)

| | ch08 | |
|---|---:|---|
| seeds | 11,609 | 3,673 from the lattice; 11,586 queued |
| rounds | 6 | frontier comps 11,586 / 14,744 / 7,834 / 2,306 / 289 / 6 |
| frontier tiles per round | 8,579 / 13,837 / 6,744 / 1,700 / 205 / 5 | 31,070 tile visits on 28,276 touched tiles (1.10 each) |
| tasks (comp, direction) | 136,764 | 739 of them dense (the warp-cooperative path) |
| warp steps / signature tests | 23,022 / 170,995 | 1.25 tests per task |
| claims | 25,179 | |
| collisions recorded = union calls | 30,979 | varies run to run |
| links (unions that retired a root) | 9,162 | = seeds - roots |
| roots + interior comps | 2,447 + 75 | = 2,522 blobs |
| closure iterations (K2) | 76,355 | |
| atomics, without the unions' atomicMin | 206,466 | |
| ch06 | 539,207 runs, 681,039 union attempts | ch06 session `runs_20260725T161448Z` |

### Worst cases

| scene | ch08 | scan | index | valve off | ch06 | ch08 / ch06 | labelling, ch08 / ch06 | ch05 (host) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zz_serp_1k | 0.361 | 0.517 | 0.312 | 0.358 | 0.188 | 1.882 | 0.326 / 0.128 | 0.989 |
| maze_1k | **0.297** | 0.344 | 0.258 | 0.297 | 0.460 | **0.646** | 0.258 / 0.169 | 1.896 |
| diag_snake_159 | **0.528** | 0.535 | 0.524 | 0.956 | 0.574 | **0.919** | 0.124 / 0.083 | 35.53 |
| zz_lattice_4k | 6.888 | 16.330 | 6.199 | 7.029 | 1.729 | 3.985 | 6.252 / 0.441 | 7.94 |
| stair_decoy_19 | 0.310 | 0.584 | 0.294 | 0.798 | 0.125 | 2.504 | 0.197 / 0.051 | 0.844 |
| zz_serp_4k | 3.232 | 5.915 | 3.055 | 3.069 | 1.398 | 2.351 | 2.766 / 0.647 | 4.53 |
| dots_4k | 16.080 | 16.071 | 16.256 | 16.065 | 6.244 | 2.577 | 7.093 / 1.526 | 4.95 |
| strip_1px_x | 8.011 | 8.184 | 8.360 | 7.992 | **48.160** | **0.167** | 3.737 / 21.259 | 5.36 |

The last four come from the perf review of this chapter.

- **`zz_lattice_4k`** puts up to 60 ring comps in each of its 16,384
  tiles (59.6 on average). The scan makes 14.6M warp steps (16.3 ms);
  the hybrid 2.1M (6.9 ms).
- **`stair_decoy_19`** (608 x 4288) is a lattice-avoiding staircase:
  one seed and a 133-comp chain. Isolated decoy pixels sit on every
  path tile's frame, up to 59 ring comps per tile, and the scan must
  test each one. The valve fires at round 16 (17 rounds); off, 133.
- **`zz_serp_4k`**: 32 ring comps per tile, 17.6M signature tests.
- **`dots_4k`**: red at even (x, y), 256 one-pixel comps per tile. There
  are no waves at all (0 rounds): K2 closes 4.19M comps one by one
  (6.74 ms) and the paint re-closes them (8.86 ms).
- **`strip_1px_x`** (4194304 x 1): a tile holds 32 px, so 131,072
  tiles for 4 Mpx. ch06 suffers more (its pack takes 24.8 ms).

**ch08 lost to ch05 on two of them**: `dots_4k` (16.08 vs 4.95 ms) and
`strip_1px_x` (8.01 vs 5.36). These ch05 numbers are its host clock.
The hybrid lookup turned the other ch05 losses into wins.

An earlier draft of this page said that the true `dots_4k` gap was
larger, because ch05's clock window read a 510 MHz median. That was
wrong: the window mostly covered ch05's host work. Timed by events at
2,070 MHz (final session), ch05 takes 4.48 ms on `dots_4k`, and the
dense-tile path 1.484.

The `dots_4k` GPU window also logged one sample at 210 MHz SM and 405
MHz memory (P3), with a 2,070 MHz SM median, so its ch08 and ch06 times
may carry a small dip.

## Dense tiles: the run CCL (the default path)

**Where the first version lost: tiles full of tiny comps.** It took
16.08 ms on `dots_4k` (ch06 6.24), 6.89 on `random_4000` (5.89), 6.89
on `zz_lattice_4k` (1.73) and 0.254 on `input_blocks` (0.137).

An `ncu` profile of that version
(`results/ch08_gpu_nblob_tiles/ncu/before/`) found two causes:

- **K2 and K4 are issue-bound (חסומות בהנפקה) on dense tiles.** The
  comp loop pays one whole-warp closure per comp: about 145 warp
  instructions per closure trip, plus 70-230 per comp. A `dots_4k` tile
  of 256 one-pixel comps cost 54,656 in K2.
- **K3 is latency-bound (חסום בהשהיה) on `zz_lattice_4k`.** Its tiles
  of 60 ring comps made 2.15M serial warp-wide steps.

**The switch.** The default engine is now the dense-tile path
(`TileWaves(dense=True, lookup="dindex")`). `flood_fill.OLD_PATH`
(`dense=False, lookup="hybrid", fit_grids=False`) is the first version,
unchanged, and is timed as the ablation (אבלציה, the old path beside
the new). The experiments E1-E13 (section "Experiments") ran on it.

### The protocol, in plain words

Every time in this section comes from one session,
`dense_20261016T054922Z.json` (the final session), unless a line names
another.

- **Same rounds.** Each scene is timed in 9 rounds. Every config runs
  once per round, and odd rounds run the configs in reverse order. So a
  slow moment of the GPU hits every config alike.
- **Same start for every run.** The image is restored from a clean
  device copy, a 96 MB read flushes the L2 cache (מטמון), and a short
  GPU sleep kernel is queued, so all launches are in the queue before
  the clock starts.
- **Two events.** One CUDA event before the first launch and one after
  the last. A separate pass times each phase on its own (5 rounds).
- **Fair baselines.** ch06 runs at its fastest merge grid, swept per
  scene in the same session, and also with ch08's pack (`ch06_tp`).
  ch05 now gets the same treatment: fresh buffers, a GPU-bound respin,
  the flush, the sleep kernel, two events, 9 samples.
- **Checks.** Every painted image equals ch06's own `RunRecolor.run`,
  in every config and scene, ch05's too. An instrumented engine reruns
  each ch08 config for the counters.
- **Ratios** are medians of the per-round ratios, with their range.

**Clocks.** The SM clock read 2,070 MHz (minimum and median) in every
timed window, every phase pass and every ch05 window but one
(`input_blobs`' ch05 window: minimum 1,875). The memory clock dipped
on `input_blobs` and `input_blocks` (P3, minimum 5,501 MHz; see the
top of this page).

The first session of this path, `dense_20261016T021919Z`, timed ch05
by its host clock and recorded an inflated `paint_tiles` counter (the
review's fixes are below). Its new / old ratios agree with this
session's within 0.025 in the rgb and mask contracts, and within 0.038
in the labelling tier.

Its other ratios moved more: the ablations by up to 0.059, rgb against
ch06 by up to 0.061, and labelling against ch06 by up to 0.44
(`zz_serp_4k`).

### The final session

| scene | ch08 new | ch08 old | new / old | ch06 rgb | new / ch06 | ch05 (events) |
|---|---:|---:|---:|---:|---:|---:|
| **dots_4k** | **1.484** | 15.405 | **0.096** [0.096, 0.098] | 5.934 | **0.250** | 4.481 |
| **random_4000** | **2.287** | 6.652 | **0.344** [0.341, 0.346] | 5.618 | **0.407** | 21.905 |
| **zz_lattice_4k** | 2.663 | 6.642 | **0.401** [0.390, 0.421] | 1.721 | 1.546 | 7.328 |
| **input_blocks** | 0.201 | 0.255 | **0.785** [0.773, 0.800] | 0.137 | 1.459 | 0.951 |
| input_blobs | 2.403 | 2.378 | 1.008 [0.981, 1.016] | 2.567 | 0.890 | 47.064 |
| blob_grid_100 | 0.570 | 0.552 | 1.034 [0.986, 1.157] | 0.666 | 0.859 | 25.315 |
| disk_r2000 | 0.560 | 0.543 | 1.028 [0.984, 1.107] | 1.002 | 0.554 | 24.980 |
| serpentine_2048 | 0.354 | 0.558 | 0.635 [0.630, 0.647] | 0.242 | 1.466 | 18.518 |
| crop_1000 | 0.105 | 0.094 | 1.119 [1.085, 1.143] | 0.088 | 1.198 | |
| crop_1500 | 0.132 | 0.120 | 1.111 [1.085, 1.138] | 0.121 | 1.093 | |
| crop_2000 | 0.182 | 0.169 | 1.074 [1.072, 1.123] | 0.179 | 1.017 | |
| crop_3000 | 0.288 | 0.272 | 1.056 [0.959, 1.064] | 0.329 | 0.875 | |
| crop_4000 | 0.451 | 0.433 | 1.040 [0.983, 1.048] | 0.551 | 0.818 | |
| zz_serp_1k | 0.290 | 0.359 | 0.796 [0.741, 0.865] | 0.186 | 1.533 | 0.574 |
| maze_1k | 0.276 | 0.296 | 0.934 [0.924, 0.945] | 0.459 | 0.602 | 1.486 |
| diag_snake_159 | 0.544 | 0.540 | 1.009 [0.980, 1.100] | 0.561 | 0.967 | 34.687 |
| stair_decoy_19 | 0.224 | 0.311 | 0.722 [0.709, 0.747] | 0.127 | 1.766 | 0.329 |
| zz_serp_4k | 1.424 | 3.067 | 0.462 [0.451, 0.490] | 1.294 | 1.079 | 3.994 |
| strip_1px_x | 7.499 | 7.762 | 0.965 [0.959, 0.972] | 46.634 | 0.161 | 4.767 |

**The four targets.** `dots_4k` is 10.4x faster than the first
version and beats ch06 4x. `random_4000` is 2.9x faster and beats ch06
2.5x. `zz_lattice_4k` is 2.5x faster but still 1.55x ch06.
`input_blocks` is 1.27x faster but still 1.46x ch06.

**ch05, timed by events.** ch08 now beats ch05 on every scene where
ch05 ran but `strip_1px_x` (7.499 vs 4.767 ms). On `dots_4k` it takes
0.33 of ch05 (1.484 vs 4.481 ms).

ch05's host clock had added 0.3-0.5 ms per call. In the first
dense session it read 0.83 ms on `stair_decoy_19`, 0.93 on
`zz_serp_1k` and 1.26 on `input_blocks`; events give 0.329, 0.574 and
0.951. On the 4 ms scenes the two clocks agree (`dots_4k` 4.61 vs
4.481, `zz_lattice_4k` 7.32 vs 7.328).

**`input_blobs` is within the 2% bound in the rgb contract:** 1.008
[0.981, 1.016]. The mask contract (1.031) and the labelling tier
(1.073: 0.408 vs 0.380 ms) are above it. No tile of `input_blobs`
takes the dense path, so this is fixed overhead (phase pass): K2's
estimate 0.012 ms, K3's compacting twin 0.011, and two empty launches
of 0.005 each.

**The other contracts**, new / old and new / ch06:

| scene | mask: new, old, ch06 | new / old | new / ch06 | labelling: new, old, ch06 | new / old | new / ch06 |
|---|---:|---:|---:|---:|---:|---:|
| dots_4k | 1.231, 15.156, 5.636 | 0.081 | 0.218 | 0.636, 6.678, 1.527 | 0.095 | 0.416 |
| random_4000 | 2.037, 6.413, 5.333 | 0.318 | 0.382 | 1.183, 3.107, 2.039 | 0.380 | 0.580 |
| zz_lattice_4k | 2.397, 6.383, 1.428 | 0.376 | 1.678 | 2.050, 6.013, 0.443 | 0.341 | 4.619 |
| input_blocks | 0.174, 0.229, 0.111 | 0.766 | 1.574 | 0.121, 0.138, 0.043 | 0.874 | 2.857 |
| input_blobs | 1.122, 1.086, 1.107 | 1.031 | 0.954 | 0.408, 0.380, 0.505 | 1.073 | 0.807 |
| blob_grid_100 | 0.337, 0.314, 0.372 | 1.075 | 0.911 | 0.151, 0.133, 0.094 | 1.131 | 1.587 |
| disk_r2000 | 0.298, 0.282, 0.686 | 1.058 | 0.433 | 0.139, 0.129, 0.073 | 1.087 | 1.915 |
| diag_snake_159 | 0.158, 0.140, 0.105 | 1.132 | 1.495 | 0.138, 0.124, 0.082 | 1.116 | 1.687 |

**The labelling tier flatters the dense path.** It ends at K3, and K4b
then rebuilds each dense tile's run CCL to paint it. On `dots_4k` that
rebuild is 0.569 ms (phase pass): charged to the labelling, the ratio
to ch06 would be (0.636 + 0.569) / 1.527 = 0.79, not 0.416. On
`random_4000` it is 0.831 ms: (1.183 + 0.831) / 2.039 = 0.99, not
0.580. The rgb contract carries no such gap: 0.250 and 0.407.

**Where the time went (phase pass, ms).** K2 + K2b, K3, and K4 + K4b,
new against old:

| scene | labelling | waves | paint |
|---|---:|---:|---:|
| dots_4k | 0.526 vs 6.512 | 0.110 vs 0.163 | 0.600 vs 8.482 |
| random_4000 | 0.922 vs 2.482 | 0.264 vs 0.625 | 0.857 vs 3.299 |
| zz_lattice_4k | 0.481 vs 1.739 | 1.567 vs 4.277 | 0.358 vs 0.359 |
| input_blocks | 0.056 vs 0.068 | 0.065 vs 0.073 | 0.057 vs 0.093 |
| input_blobs | 0.273 vs 0.256 | 0.133 vs 0.122 | 0.649 vs 0.640 |

### What changed

1. **K2 decides, K2b labels.** K2 estimates each tile's comp count
   before its comp loop. A tile whose estimate reaches `ccl_min` (12),
   with at most `ccl_rpc` (12) runs per estimated comp, goes to a list.
   K2b, a new launch, labels the listed tiles by a run CCL.
2. **K4 decides, K4b paints.** A tile that no fast path took, with at
   least `paint_ccl_min` (10) comps and at most 24 runs per comp, goes
   to K4b. K4b rebuilds the CCL and paints from one label per run.
3. **The dindex K3.** Index steps where K2b indexed the neighbour tile,
   task compaction, and a cheaper flatten (below). It is a new twin; the
   first version's K3 is untouched.
4. **Grids.** K2 and K4 are cut to one warp per tile. K2b and K4b run at
   their resident capacity, so an empty list costs one wave of blocks.
   The engine refuses a block that one of its twins cannot launch (the
   review's F1, below).

### The run CCL (K2b)

A run (ריצה) is a maximal stretch of red bits in one row word. Lane i
holds row i, so a tile has at most 16 runs per row and 512 in all.

**Slots.** Every run start gets one int32 slot in shared memory:
`16 * row + (s >> 1)`. Two starts of one row are at least 2 bits
apart, so `s >> 1` is unique, and the slot order is the (row, bit)
order.

**Union by the smaller slot.** ch06's atomicMin union-find, run on the
slots, ends with the minimum run start of each comp as its root. That
start is the comp's anchor: the comp's min (row, bit) pixel starts a
run.

**So K2b writes exactly K2's records.** Ring ordinals are anchor order,
so a prefix count over a 512-bit bitmap of the ring roots gives them.
The side words come from `match_any_sync`: the lanes whose frame pixel
on one side belongs to one comp form a match group, and the group's
lane mask is that comp's side word.

The header, records, seeds and queue entries equal the comp loop's.
The tests check them against the same CPU model.

**Union events.** Row i-1 holds runs a, row i runs b. At
8-connectivity:

- E1: bit j starts b, and row i-1 is red at j-1, j or j+1. Union b with
  the run holding the lowest of those.
- E2: row i is red at j, and a run of row i-1 starts at j+1. Union them
  (skipped where E1 at j already chose j+1).

**Why every adjacent pair gets an event.** Let b span q0 to e.

- If a holds one of q0-1, q0, q0+1: E1 at q0 picks the lowest red one.
  That is a, unless a starts at q0+1 behind a red q0-1 of another run;
  then E2 at q0 unions them.
- Otherwise a starts at some sa in q0+2..e+1 (an earlier run that
  reaches b would hold q0-1..q0+1). E2 at sa-1, a bit of b, unions them.

4-connectivity is the same with "overlaps" for "touches": E1 at q0 if
row i-1 is red at q0, E2 where a run of row i-1 starts inside b past
q0.

**Flatten by pointer jumping (קפיצת מצביעים).** A round sets every
start's slot to its grandparent, so a chain of depth d takes about
log2(d) rounds. A read-only find per run walked the whole chain: 23
steps on a disk 24 rows tall.

**The cost model (measured, per tile at full occupancy).** The CCL
costs about 12 ns plus 0.19 ns per run. The comp loop costs about 2 ns
per comp and closure trip. So on density-0.5 noise (9 comps, 264 runs
per tile) the comp loop should win.

The runs-per-comp cap was meant to keep such tiles on it, but it does
not: the estimate below counts far more comps than there are on noise.
See "Noise around the hand-off" below.

**The estimate.** It counts the runs that touch no red pixel of the row
above. Every comp has one in its first row, so this bounds the comp
count from above. It also counts the same from below, and takes the
smaller.

One side alone counts every bump of a rough blob edge. On `input_blobs`
it sent 10,847 of 28,276 tiles (1.3 comps each) to the CCL at 3. A
carry (נשא) trick counts the untouched runs: `m + con` carries from
each run's lowest contact bit through its last bit.

**Why a separate kernel.** With the CCL inline, K2 needed 146
registers. Capped at 80 it spilled, and the comp loop got 22% slower on
`input_blobs` with no tile taking the CCL.

### K4b

K4 keeps every fast path. It hands a tile to K4b only when none of
them took it, so one-colour tiles (`zz_lattice_4k`, `maze_1k`) never
reach K4b. K4b used to repeat K4's colour-uniform test; with the same
header and labels that test could never fire, and the review removed
it.

K4b rebuilds the CCL, and each run start gets its comp's colour. A ring
comp reads its record through the ring bitmap; an interior comp's label
is its root's anchor. The rows are painted from the slots.

The palette is packed into three 48-bit constants. A lookup with a
per-lane index in constant memory serialises; it was a third of K4b's
instructions.

### The dindex K3

**Index steps.** K2b writes the frame words and one byte per frame
pixel (the ordinal of the comp that owns it) for every tile with 2+
ring comps, and flags the header. On a flagged neighbour tile a task
takes one step per distinct adjacent comp, in parallel over the lanes.
Unflagged tiles keep the hybrid lookup.

**Task compaction (דחיסת משימות).** The first version gives each
queue entry 8 lanes, one per direction. On `zz_lattice_4k` a comp has
1.1 direction bits, so 7 lanes in 8 idle.

Now a warp takes a chunk of E entries, prefix-sums `popc(dirs)` over
the lanes, writes the tasks to shared memory, and runs them 32 at a
time. E is set per round, from 4 to 32, so that a small round still
spreads over every warp. With E fixed at 32, `crop_1000`'s waves took
0.081 ms instead of 0.043.

**Flatten.** A ring comp with no direction bit is a lone seed: its own
root and class. Flatten 1 writes its anchor as the final label, and
flatten 2 skips it. Elsewhere the root's rmin is read before the
atomicMin. On `zz_lattice_4k` 975,867 atomicMins hit one root; now none
is issued.

**The price of compaction: spills.** It takes the twin from 16 to 56
bytes of local memory at the 80-register cap (88 or 96 registers cost
cooperative blocks). On `input_blobs` the twin issues fewer
instructions than the old K3 (6.10M vs 6.82M warp instructions,
`ncu`), but it is 9% slower in the phase pass (0.133 vs 0.122 ms). The
spills are the likely cause (inferred).

### The ablations (rgb, per-round ratio medians)

| scene | no compaction / new | no compaction / old | auto / new | new K3 alone / old | CCL with old K3 / new |
|---|---:|---:|---:|---:|---:|
| dots_4k | 0.993 | 0.096 | 0.999 | 1.002 | 1.018 |
| random_4000 | 1.062 | 0.365 | 1.000 | 1.007 | 1.078 |
| zz_lattice_4k | 1.343 | 0.535 | 0.994 | 0.940 | 1.997 |
| input_blocks | 0.969 | 0.759 | 0.995 | 1.056 | 1.005 |
| input_blobs | 0.997 | 1.006 | 0.999 | 1.014 | 0.995 |
| zz_serp_4k | 1.318 | 0.615 | 1.019 | 0.903 | 1.722 |
| serpentine_2048 | 0.977 | 0.619 | 0.994 | 1.024 | 1.411 |
| maze_1k | 1.022 | 0.955 | 1.044 | 1.048 | 1.060 |
| blob_grid_100 | 0.987 | 1.019 | 0.996 | 1.039 | 0.982 |
| disk_r2000 | 0.983 | 1.009 | 1.002 | 1.028 | 0.998 |
| diag_snake_159 | 0.979 | 0.985 | 0.992 | 1.008 | 0.977 |
| zz_serp_1k | 1.032 | 0.840 | 0.996 | 1.074 | 1.170 |
| stair_decoy_19 | 0.946 | 0.680 | 0.977 | 1.062 | 1.069 |
| strip_1px_x | 0.990 | 0.956 | 1.001 | 0.989 | 1.021 |

- **Compaction** wins where rounds are huge (`zz_lattice_4k` 1.34,
  `zz_serp_4k` 1.32 without it, `random_4000` 1.06). It costs 0.3-5.7%
  where they are small (most on `stair_decoy_19`: 0.946 [0.924, 0.954]
  without it). It stays on.
- **Compaction is only part of the sparse cost.** Without it the path
  is still 1.006-1.019 of the first version on `input_blobs`, the blob
  grid and the disk. The rest is K2's estimate and the two extra
  launches.
- **auto** launches the compact and the plain twin; each returns before
  its first barrier unless the first round's size (16,384 queued seeds)
  is on its side. A cooperative launch that returns at once still
  costs its launch: auto is 0.98 to 1.04 of the default. Off by
  default.
- **The new K3 alone** (no tile reaches the CCL) gains on
  `zz_lattice_4k` and `zz_serp_4k` (0.940, 0.903), and 1% on
  `strip_1px_x` (0.989). It costs up to 7.4% elsewhere: `zz_serp_1k`
  1.074 [1.008, 1.209], slower in every round. The index needs K2b.
- **The CCL with the old K3** loses the index and the compaction: 2.00x
  on `zz_lattice_4k`, 1.72x on `zz_serp_4k`.
- **Grid fitting** is worth 2-6% on the crops (full grids: 1.058,
  1.054, 1.034, 1.021, 1.016 of the new path, crops 1000 to 4000).

### Noise around the hand-off

The final session adds 4000 x 4000 noise of density 0.40 to 0.60
(seed 0) with one more arm: the default with K2's hand-off off
(`ccl_min` never reached; K4b unchanged).

| density | comps per tile | K2b tiles (of 15,625) | new | first version | hand-off off | ch06 | new / first | off / new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 27.5 | 15,625 | 2.719 | 4.679 | 3.649 | 6.908 | 0.581 | 1.344 |
| 0.45 | 16.1 | 15,624 | 2.842 | 3.329 | 3.138 | 7.231 | 0.853 | 1.107 |
| 0.50 | 9.1 | 15,283 | 2.609 | 2.271 | 2.405 | 7.304 | 1.148 | 0.921 |
| 0.55 | 5.2 | 9,591 | 1.998 | 1.562 | 1.606 | 7.213 | 1.282 | 0.804 |
| 0.60 | 3.2 | 1,427 | 1.358 | 1.209 | 1.267 | 6.939 | 1.125 | 0.932 |

**The runs-per-comp cap does not keep these tiles on the comp loop.**
At density 0.50 a tile holds 9.1 comps and 264 runs, yet K2 sends
15,283 of 15,625 tiles to K2b. The estimate counts the runs that touch
nothing above (or below), and a noise comp near the percolation
(חלחול) threshold has many of them.

In the phase pass at 0.50, K2 and K2b take 1.245 ms; K2 alone, with
the hand-off off, 0.703.

**The crossover lies between 0.45 and 0.50.** Below it the hand-off
pays (1.344 and 1.107 without it); above it, turning it off saves
7-20%.

A lower cap cannot find it. The runs per tile are almost equal at
0.45 and 0.50 (260 and 264), while the comps per tile halve (16.1 and
9.1). A fix needs a tighter comp estimate (inferred, not built).

Even with the hand-off off, the path is 1.032-1.060 of the first
version at 0.50-0.60: the compacting K3 and K4's hand-off to K4b cost
the rest. Every noise scene still beats ch06 2.5-5.1x.

### `ncu`, before and after

Both profiles are in `results/ch08_gpu_nblob_tiles/ncu/`: the first
version's (`before/`) and the dense-tile default's (`after/`), on the
same four scenes, each one rgb pipeline in a profiling window, with
the same sections and clock control.

**Instructions, not cycles.** Under `--clock-control base` the first
profile's SM clock wandered from 0.40 to 1.14 GHz; the second held
1.10-1.14. Instruction counts do not depend on the clock, so they carry
the comparison. Cycles divided by 2.07 GHz match the final session's
phase times within 0.96-1.01 for K2b, K4b and K3 on `random_4000`.

| warp instructions (M) | before | after | after / before |
|---|---:|---:|---:|
| `dots_4k`, labelling (K2 + K2b) | 895.4 | 30.0 | 0.034 |
| `dots_4k`, paint (K4 + K4b) | 1,224.0 | 60.9 | 0.050 |
| `random_4000`, labelling | 331.8 | 70.6 | 0.213 |
| `random_4000`, paint | 469.5 | 103.0 | 0.219 |
| `zz_lattice_4k`, waves (K3) | 319.3 | 46.6 | 0.146 |
| `input_blobs`, labelling | 30.5 | 31.9 | 1.046 |
| `input_blobs`, waves | 6.82 | 6.10 | 0.894 |

**Before: issue-bound comp loops.** K2 and K4 ran at 81-87% of the
ALU pipe, about 145 warp instructions per closure trip: 213.5 per comp
in K2 on `dots_4k`, where every comp is one pixel. K3 on
`zz_lattice_4k` was latency-bound: 2.15M serial warp steps and 64.0M
signature tests.

**After: the work went where the profile pointed.** The labelling
costs 7.2 warp instructions per comp on `dots_4k` (ch06: 8.1 per run).
On `zz_lattice_4k` the index makes every signature test a hit: 2.05M
tests, all of them CAS, in 115k warp steps; no rmin atomic is left (it
was 975,867).

**After: the new limits.**

- **K2b and K4b wait on shared memory.** On `dots_4k`, 85.6% and 84.3%
  of their shared-memory wavefronts are bank conflicts (התנגשויות
  בנקים), and short-scoreboard stalls lead. A likely cause (inferred):
  slot rows of 16 words put even rows in one half of the banks and odd
  rows in the other, so the 32 lanes share 2 banks per slot column. A
  stride of 17 would spread them.
- **K2b runs at 25% occupancy:** 130 registers.
- **K3 on `zz_lattice_4k` still waits on global memory:**
  long-scoreboard stalls of 17.5 cycles per issue, and 78.5% of its
  2.05M CAS lose.

**Against ch06 (cycles at the ncu clock).** The labelling went from
4.94x ch06 to 0.46x on `dots_4k`, and from 1.58x to 0.59x on
`random_4000`. The paint went from 2.07x to 0.14x and from 0.99x to
0.25x.


### The review and its fixes

A review of this path (correctness, fuzzing, and the fairness of the
numbers) found three bugs, and five issues of performance or
fairness. Each bug got a failing test first, then the fix.

- **A block a twin could not launch (F1).** The dense twins' block
  bound, 512 threads, counted only shared memory. K2b holds 130
  registers (dindex, bare) or 178 (instrumented), so blocks of 416 or
  288 threads and more passed validation. The constructor's warm-up
  then died with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, before any image.
  Now the engine compiles every label and paint twin it launches for
  its own arguments, and refuses such a block with a `ValueError` that
  names the largest block that fits. The first version's K2 failed the
  same way at 1024 threads, and is refused too.
- **K2b's register cap crashed the engine (F2).** The warm-up ignored
  `label_dense_max_registers`, so the constructor asked an uncompiled
  twin for its capacity: `StopIteration` for any cap. The cap now
  reaches the warm-up.
- **`paint_tiles` counted K4b's tiles twice (F2).** K4 counted every
  touched tile and K4b its own again. The tests now assert that every
  touched tile is counted once and painted by exactly one path. The
  earlier sessions' `paint_tiles` values are inflated by
  `paint_ccl_tiles`.
- **ch05's clock (fairness F1, F2).** See "ch05, timed by events"
  above: the host clock added 0.3-0.5 ms per call, and its clock window
  was mostly host work.
- **Sparse scenes and small crops (P1, P3).** The first dense session
  timed crops only up to 2000 and named only `input_blobs` and
  `crop_1000`. This session adds crops of 3000 and 4000, and the limits
  below list every scene over 2%.
- **The noise crossover (F3)**: the section above.
- **Smaller fixes.** K4b's colour-uniform branch could never fire and
  is gone. The fuzz test drew the lookup and the connectivity from one
  `seed % 4`, so the default lookup never ran at 8-connectivity there;
  now each config is drawn on its own stream, and every pair runs.

### What did not work

- **The runs-per-comp cap as a guard for noise**, above.
- **Task compaction for every image.** It is worth 1.34x on
  `zz_lattice_4k`, but costs up to 5.7% on small rounds, and choosing
  the twin per image (auto) costs a launch of its own.
- **Grid fitting** recovered 2-6% on the crops, not the 4-12% the dense
  path costs there.
- **A size-only switch back to the first version** would not fix small
  images. `input_blocks` has the tile count of `crop_1000` (1,024), and
  the default runs it at 0.785 of the first version.
- **The CCL inside K2.** It needed 146 registers; capped at 80 it
  spilled, and the comp loop got 22% slower on `input_blobs`.

### Limits and deviations

1. **`input_blobs` holds the 2% bound in the rgb contract only:** 1.008
   [0.981, 1.016] (the first dense session 1.019, its draft 1.017).
   The mask contract (1.031) and the labelling tier (1.073) exceed it.
   The cost is fixed overhead: no tile of it takes the dense path.
2. **Every sparse scene pays that overhead, in one contract or more.**
   `blob_grid_100`: rgb 1.034, mask 1.075, labelling 1.131.
   `disk_r2000`: 1.028, 1.058, 1.087. `diag_snake_159`: 1.009, 1.132,
   1.116. The crops: rgb 1.119 at 1000 down to 1.040 at 4000,
   labelling 1.143 to 1.102.
3. **On `blob_grid_100` the phase pass shows where:** K3 0.0645 vs
   0.0553 ms, K2 0.0809 vs 0.0768, and the two empty launches 0.005
   each.
4. **The crossover with ch06 moved.** On the crops the default is
   1.198, 1.093, 1.017, 0.875 and 0.818 of ch06 (1000 to 4000); the
   first version 1.071, 0.991, 0.938, 0.829 and 0.786. So the default
   ties at 2000 and wins from 3000; the first version tied at 1500 and
   won from 2000.
5. **Noise of density 0.50-0.60 is slower than the first version**
   (1.148, 1.282, 1.125), though 2.8-5.1x faster than ch06.
6. **`zz_lattice_4k` and `input_blocks` still lose to ch06** (1.55x,
   1.46x), and so do `serpentine_2048`, `zz_serp_1k` and the decoy
   staircase (1.47-1.77x). `zz_serp_4k` loses by less: 1.08x [0.91,
   1.20] in rgb, 1.19x in mask, 1.97x in labelling.
   zz_lattice's K3 is 1.57 ms of its 2.66: 2 rounds of 976k entries
   and a flatten of 976k comps by 18k threads.
7. **One CCL tile costs about 10-15 us of latency** on a lone warp
   (thousands of dependent shared-memory steps). A sparse image with a
   few dense tiles pays that once per K2b and K4b.
8. **Not built:** a bulk path for isolated pixels alone (the run CCL
   handles them, a run with no union), a per-warp shared-memory
   flatten for K3, a CCL inside K3's cooperative launch, a tighter comp
   estimate.
9. **Earlier sessions of this path:** the draft
   `dense_20261016T013317Z` (code at `97e8243`) and
   `dense_20261016T021919Z`. Both precede the review: host-clock ch05,
   inflated `paint_tiles`, crops only to 2000.

## Experiments

Sessions `experiments_20261015T155420Z` and `155822Z` (E4b) ran E1-E10
with the scan lookup, the valve at 128 and an event after every phase:
ch08 against ch08 only. `180817Z` ran E13, `181414Z` redid E9.

| id | question | result | decision |
|---|---|---|---|
| E1 | pack store | input_blobs: ch06 1.450, direct 1.181 (206 GB/s), smem 1.188, tile-major ref 1.148 | direct |
| E2 | closure exits | label 0.305 (ch07 loop), 0.268 (frontier), 0.268 (both) | both kept |
| E3 | K2 registers, grid | 80 regs at 1536 x 128: label 0.257 vs 0.268, no spill, also faster on blocks and random | adopted |
| E4 | lookup | index: input_blobs +1.1% (per-round), random -12%, serpentine -32%, blocks -7% | scan then (the session JSON says index, see deviation 4); superseded by E13 |
| E5 | paint | 256 threads: input_blobs 0.636 vs 0.638 (tie), input_blocks +6.5%; fast path off +2-10% | 768 x 128, fast path on |
| E6 | seed dial | L = 0 to 1, all within 0.4% on input_blobs; L0 pays 90 rounds on the disk | L8 |
| E7 | union timing | inline: waves 0.131 vs 0.120 ms on input_blobs | deferred |
| E8 | counter cost | instrumented: waves +0.09, label +0.11 ms on input_blobs | bare twins timed |
| E10 | waves launch | 256 vs 128 threads: identical (72 vs 144 blocks) | 256 |
| **E13** | lookup x K3 registers | hybrid, dense_min 4, 80 regs: input_blobs 0.991 of the scan, decoy staircase 0.287, zz_lattice 0.423, zz_serp_1k 0.692, maze 0.871, serpentine 0.929. Index: 1.020 on input_blobs. The headline session differs (below) | **hybrid, dense_min 4, 80 regs** |
| **E9** (redone) | valve round at L8 | where the valve fires, 16 is fastest: decoy staircase 0.394 of the 128 arm, plain staircase 0.308, diagonal snake 0.620. Main scenes need at most 8 rounds | **16** |

**E13's rule** was E4's: input_blobs may move under +1% (per-round
median), and at least 2 other scenes must gain 10%. Among the arms that
pass, the lowest geometric mean over the other scenes wins. The index
lookup fails the first part: its K2 writes a byte index and spills at
80 registers.

**Two of E13's numbers did not hold in the headline session.** There
the hybrid is 1.005 of the scan on input_blobs (2.214 vs 2.201 ms): a
tie, not E13's 0.991. The scan also edges it on input_blocks (0.252 vs
0.254) and the disk (0.550 vs 0.552).

E13's decoy staircase window ran at a 285 MHz SM median (P4, memory 810
MHz), with the valve still at 128 (129 rounds). Its input_blobs window
logged 210 and 405 MHz minima.

At full clocks and with the valve at 16, the decoy staircase's hybrid
is 0.531 of the scan (0.310 vs 0.584 ms), not 0.287. The big gains
hold: `zz_lattice_4k` 6.888 vs 16.330 ms, `zz_serp_4k` 3.232 vs 5.915.

**The 80-register cap matters for a cooperative kernel.** The hybrid
twin takes 97 registers by itself: 2 blocks of 256 per SM, 48 in all.
Capped at 80 it keeps 3 per SM (72) with no extra local memory.

**E9's old rule needed a strict win on maze_1k and zz_serp_1k.** No arm
can fire the valve there, so the arms run the same code, and 128 "won"
maze_1k by 0.001 ms of noise. The rule now looks only at scenes where
some arm fires.

## The measurement lessons (from the fairness review)

The draft of this page quoted session `waves_20261015T160144Z`. A review
of its benchmark found three fairness problems. The benchmark and this
page now handle each one.

- **Phase events stalled ch06 on small scenes.** An event after every
  phase put 16 commands behind the sync for ch06 and 12 for ch08. With
  that many, the driver stalls inside the timed window (the review's
  sweep tied it to the command count, not to a kernel). ch06 read
  0.214 ms on `input_blocks` then and 0.137 now: 1.211 x ch06 became
  1.851. `crop_1000` went from 0.687 to 1.081.
- **"No crossover" was that stall.** The draft said ch08 wins at every
  crop size. It ties at 1500 and loses at 1000. Its "ch06 paints 0.097
  ms at 2000" was mostly stall too: the phase pass gives 0.048.
- **ch06's merge grid was tuned for its instrumented twin.** ch06 chose
  512 blocks with counter atomics in every thread. Bare, this session's
  sweep gives 0.181 ms at 4096 blocks on `input_blobs`, 0.305 at 512.
- **So the core gain was overstated.** The draft's labelling ratio was
  0.607 (0.384 vs 0.630 ms); with ch06 at its best it is 0.760.
- **ch05 in the timed loop dropped the clocks.** Its slow host calls let
  the GPU idle between GPU rounds. It now runs after them.

The headline moved from 0.827 to 0.862 against ch06, and from 0.927 to
0.962 against ch06 with the same pack.

The review of the dense-tile path added two more lessons:

- **A host clock is not a GPU clock.** ch05's `kernel_ms` wraps its
  launches and a synchronize in `perf_counter`, with no sleep kernel
  ahead. On scenes under a millisecond that more than doubled its time
  (`stair_decoy_19`: 0.83 ms host, 0.329 events). ch05 is now timed
  like every other side.
- **A clock window can watch the wrong thing.** ch05's `dots_4k` window
  read a 390-510 MHz median because it mostly covered ch05's host work,
  not its kernels. This page once "scaled" ch05 to 2,070 MHz on that
  basis (about 0.9 ms); events at 2,070 MHz give 4.48 ms.

## Correctness

The proof (design section 7) in five lines:

1. Comps partition the red pixels, and the comp graph's components are
   exactly the blobs.
2. Every multi-comp blob has a seed in its lex-min tile.
3. Every crossing ring comp is processed once with its full dirs.
4. Every edge ends with equal labels or a pair recorded from its SUCC
   side, so the unions join exactly each blob's seeds.
5. Flatten takes the min anchor into the root: the canonical label.

The hybrid lookup keeps step 3: a dense task tests every ring comp of
the neighbour tile, with the same CAS and collision rule. So does the
dindex lookup: an index step names every comp that owns a wanted
frame pixel.

The dense path changes none of the five steps. K2b writes the same
records, headers and seeds as the comp loop (its union events cover
every adjacent run pair; the argument is in the dense-tile section).

Tests (`test_correctness.py`, 2,567, 8 of them behind
`FLOOD_FILL_SLOW=1`):

- the closure against a CPU flood (both exits and ch07's loop)
- K2's records and the index bytes against a CPU model built from pixels
- `pack_tiles` against ch06's pack on odd sizes
- a 46-config end-to-end matrix on 49 scenes, every invariant checked,
  including the bare (timed) twins of every lookup at both connectivities
- paint equal to ch06 and ch05; every seed rule giving identical labels
- the valve, rounds past the 252 stamp slots, five fuzz families at 1
  block and at max blocks (every lookup at both connectivities, about
  one case in six bare)
- poisoned engine reuse, repeatability, twins, limits, launch grids
  that are not whole warps, the E1-only pack store
- the dense path: K2b's run CCL against a CPU flood on 511 masks,
  K2b's records, index and header flag against the model, 19 more
  matrix configs (16 of the dense path, most of them forced on every
  touched tile: every lookup, both connectivities, odd grids, auto;
  3 of the old path), and the model's prediction of the CCL tiles,
  indexed tiles and dense tasks checked on every instrumented run
- the review's fixes: blocks of 512, 384 and 256 threads for every
  lookup, bare and instrumented, run exactly or are refused; the first
  version refuses 1024; K2b's register cap builds and stays exact; and
  every instrumented run counts each touched tile once in `paint_tiles`
  and paints it by exactly one path

ch07's 247 tests (1,959 runs over its 9 switch modes) and ch06's 109
also pass.

## Deviations from the design (and why)

1. **The default lookup is a new hybrid** (E13), not the scan or the
   index. The scan and index twins stay. Every main and worst-case
   scene times all three; the crops time only the hybrid.
2. **The index covers only tiles with 2+ ring comps.** A single ring
   comp is ordinal 0. Indexing every touched tile cost +2-4%.
3. **K2 is capped at 80 registers on a 1536-block grid** (E3), and the
   hybrid K3 at 80 (E13). The index twin's K2 spills at 80 and keeps 96.
4. **E4's recorded decision is not taken.** Sessions `155420Z` and
   `155822Z` recorded `index` (E4b: `index_rauto_g1536`). They read
   the rule with the ratio of medians: 1.006 on input_blobs (E4b 1.007).
   The rule's statistic is the per-round median, 1.011 (E4b 1.013).
   That fails the +1% bound, so the scan stayed until E13.
5. **E5's mechanical winner is not taken.** 256-thread paint won by
   0.3% on input_blobs (a tie) and lost 6.5% on input_blocks.
6. **The valve fires at round 16**, not 128 (E9 redone, with the rule
   fixed).
7. **Timing differs from the design's.** Totals have two events, phases
   come from their own pass, the labelling tier is timed directly, ch06
   is timed bare at its best merge grid, and ch05 runs after the rounds.
8. **The respin is GPU-bound.** A respin of back-to-back runs is
   launch-bound on small scenes and let the clock fall to 585-790 MHz.
9. **Not counted:** `FIND_STEPS` and the unions' atomicMin calls.
   ch06's `_find` and `_union` are imported verbatim.
10. **Not built:** E5's touched-tile list arm, E8's per-thread-atomics
    arm, the optional E11 dense-tile path and E12 minprop twin.
11. **The valve scenes are an anti-diagonal snake and two staircases.**
    A lin-monotone zigzag on tile borders is full of S2 seeds.
12. **ch05 got 3 samples on input_blobs and dots_4k**: their first calls
    took over 2 s (the design's rule). The final dense session times
    ch05 by events, 9 samples on every scene.

## New problems and lessons

- **The node decides the work.** A tile node is opened again whenever
  new bits reach it; a comp node, closed once in K2, is claimed exactly
  once.
- **The pack is the biggest single gain, and it is portable.** 0.33 ms
  of the 0.40 ms comes from reading the RGB tile by tile: ch06 with that
  pack drops from 2.617 to 2.291 ms (0.899 of itself).
- **A lookup has a cost model.** The scan costs one dependent step per
  ring comp of the neighbour tile. Dense tiles need the whole warp, or
  an index.
- **A cooperative kernel's registers set its grid.** 97 to 80 registers
  took the hybrid K3 from 48 to 72 resident blocks.
- **Paint is the shared floor.** Both chapters write 40 MB of scattered
  bytes in about 0.6 ms.
- **On this WDDM driver, the number of commands in the timed window is
  itself a variable.** Count them, or keep the window to two events.
- **Tune the baseline in the same harness.** ch06's merge grid was
  right for the twin it was tuned on, and wrong for the twin timed here.

## Open problems (next steps)

1. **Dense tiles: done, with what remains.** K2b and K4b took
   `dots_4k` from 15.4 to 1.5 ms. `zz_lattice_4k` (K3: two rounds of
   976k entries, a flatten of 976k comps) and `input_blocks` still
   lose to ch06. One CCL tile costs 10-15 us of latency on a lone warp.
2. **An index for dense tiles only: done** (the dindex lookup). Its
   compacting twin spills at 80 registers, and on `input_blobs` it is
   0.011 ms slower than the first version's K3 (phase pass); fewer live
   registers in its task loop could free that (inferred).
3. **Degenerate shapes.** A 1-px strip puts 32 px in a tile: 131,072
   tiles and about 416 MiB of buffers for 4 Mpx. A 1-D tiling, or
   routing thin images to another method, would fix both.
4. **Port `pack_tiles` to ch06.** It is worth 0.33 ms there (2.617 to
   2.291 ms on input_blobs).
5. **Small images and sparse ones.** The default ties ch06 at 2000 x
   2000 px and loses below (`crop_1000` 1.198). The dense path added
   two launches, the estimate and the compacting K3 (`crop_1000` 1.119
   of the first version; `blob_grid_100` 1.131 in the labelling tier).
   CUDA graphs or a narrow-front mode without `grid.sync` are
   candidates. A host-side switch by size is not: `input_blocks` has
   `crop_1000`'s tile count and gains 1.27x from the dense path.
6. **The paint floor.** `ncu` runs now (the GPU counters were enabled
   on 2026-10-16; profiles in `results/ch08_gpu_nblob_tiles/ncu/`). On
   `input_blobs` K4 uses 31% of the ALU pipe and 33% of DRAM's peak,
   with long-scoreboard stalls first. Its time does not follow the SM
   clock (ncu cycles at 2.07 GHz give 0.70 of the phase time), and its
   stores are 1.93M requests of 2.86 sectors each. Why scattered bytes
   cap near 63 GB/s is still unproven.
7. **K2b's and K4b's bank conflicts.** 84-86% of their shared-memory
   wavefronts on `dots_4k` (ncu). A padded slot row is the first thing
   to try.
8. **K2's comp estimate on noise.** At density 0.50-0.60 it sends
   tiles to K2b that the comp loop labels faster (1.148-1.282 of the
   first version). The cap cannot separate them; a tighter estimate
   could.

## Glossary

- **Tile** (אריח) - 32 rows x 32 columns, one warp per tile.
- **Comp** (רכיב מקומי) - a blob restricted to one tile.
- **Ring comp / interior comp** - touches the tile frame / does not.
- **Anchor** (עוגן) - a comp's minimum linear index `x*H + y`.
- **Canonical label** (תווית קנונית) - a blob's minimum anchor.
- **Seed** (זרע) - a comp that starts a wave with its own id.
- **Lattice** (סריג) - the tiles with `(tx + ty) % L == 0`, all seeds.
- **Wave** (גל) - the comps that carry one seed's label.
- **Frontier** (חזית) - the comps claimed in the last round.
- **Collision** (התנגשות) - two waves meeting on one edge.
- **Union-find** (איחוד-חיפוש) - ch06's structure for joining labels.
- **Valve** (שסתום) - the round after which unclaimed comps self-seed.
- **Dial** (חוגה) - the lattice spacing L, from pure waves to pure CCL.
- **Ordinal** (מספר סידורי) - a ring comp's index in its tile.
- **Signature test** - one side word tested against one ring comp.
- **Dense task** - a task whose neighbour tile has more than
  `dense_min` ring comps; the whole warp takes it.
- **Spill** (שפיכה) - a value moved from a register to local memory.
- **Interleaving** (שזירה) - timing every config once per round.
- **Run** (ריצה) - a maximal stretch of red bits in one row word.
- **Run CCL** - K2b's union-find over a tile's run starts, in shared
  memory; K4b rebuilds it to paint.
- **Slot** (תא) - a run start's int32 in shared memory,
  `16 * row + (s >> 1)`.
- **Dense-tile path** - K2b, K4b and the dindex K3; the default.
- **Task compaction** (דחיסת משימות) - a warp running only the
  (entry, direction) tasks that exist, 32 at a time.
- **Bank conflict** (התנגשות בנקים) - lanes of one warp touching
  different words of one shared-memory bank; the access is replayed.

## Files

```
kernels.py           K0-K4, K2b and K4b (the run CCL), the dindex K3,
                     the off-clock twins, factories and caches
flood_fill.py        TileWaves (stages, run), recolor(), max_blocks,
                     OLD_PATH
cpu_oracle.py        ch05's 8-conn oracle, a 4-conn port with the same labels
cpu_model.py         per-tile comps from pixels, the wave model with the valve,
                     the run estimate and the dense path's predictions
scenes.py            adversarial scenes (staircase, snake, frame combs, ...)
                     and five fuzz families
test_correctness.py  2,567 tests (8 behind FLOOD_FILL_SLOW=1)
benchmarks/
  benchmark.py       the interleaved session vs ch06 and ch05;
                     --session dense: old vs new ch08, the ablations,
                     crops to 4000, noise around K2's hand-off
  experiments.py     E1-E10, E4b, E13, R8 and E9, with the decision rules
```

Result files are in `results/ch08_gpu_nblob_tiles/benchmark_results/`.
`waves_20261015T183401Z` is the first version's session.
`dense_20261016T054922Z` is the final session of the dense-tile path;
`dense_20261016T021919Z` and its draft `dense_20261016T013317Z` came
before the review.
`waves_20261015T160144Z` is the draft's, from the benchmark before the
fairness review. The experiment sessions are named above.

The `ncu` profiles before and after the dense-tile path are in
`results/ch08_gpu_nblob_tiles/ncu/`: a summary, curated CSV tables
and the driver (no `.ncu-rep` file and no raw export).

## Run

```bash
uv run pytest src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/
FLOOD_FILL_SLOW=1 FLOOD_FILL_IMAGES=images/input uv run pytest src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.benchmark --images images/input
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.benchmark --images images/input --session dense
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.experiments --images images/input
```
