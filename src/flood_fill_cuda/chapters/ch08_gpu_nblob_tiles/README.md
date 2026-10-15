# Chapter 8 - many blobs, waves over tile components (`ch08_gpu_nblob_tiles/`)

**Can ch07's tiles label many blobs, and how close do they get to ch06?**
Yes, if the BFS node is a tile component, not a tile.

On `input_blobs.png` the tile waves beat ch06's run-based labelling in
all three contracts. They stay BFS-like: labels travel as frontier
(חזית) waves, one barrier per round.

| `input_blobs.png` (81 Mpx, 2,522 blobs) | ch05 `split_L8` | ch06 runs | ch06 + ch08's pack | **ch08** | ch08 / ch06 |
|---|---:|---:|---:|---:|---:|
| rgb contract (RGB in, recoloured RGB out) | 47.45 (host clock) | 2.617 | 2.291 | **2.214** | **0.862** [0.784, 0.960] |
| mask contract (packed mask in, RGB painted) | | 1.103 | | **1.018** | **0.923** [0.661, 0.930] |
| labelling only (no pack, no paint) | | 0.500 | | **0.381** | **0.760** [0.759, 0.772] |

All times are medians in ms from one session,
`waves_20261015T183401Z.json`, unless a line names another session.
Ratios are medians of the 9 per-round ratios, with their range.

ch06 runs at its fastest merge grid for this image (4096 blocks, swept
in the same session). At its shipped 512 blocks it takes 2.675 ms.
ch05 is timed by its own host clock, so "21x" is only indicative.

**The short version.** ch08 is 0.40 ms faster than ch06 in the rgb
contract (2.214 vs 2.617 ms). Most of that is the faster pack read,
which ch06 can take too.

With ch08's pack, ch06 takes 2.291 ms: the pack is worth 0.33 ms end to
end. Against that ch06, ch08 is 0.962 [0.898, 1.071], 0.08 ms faster.

Those 0.08 ms are roughly the labelling, 0.12 ms faster (0.381 vs 0.500
ms), less ch08's paint, 0.045 ms slower. ch08 labels 36,863 tile comps,
ch06 539,207 runs.

Tiles still lose where a tile holds many comps (noise, dot lattices),
on long thin shapes along tile borders, and on small images: below
about 1500 x 1500 px ch06 is faster.

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
each call. The `@njit` oracle runs after every GPU scene.

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

**There is a crossover**, as the design predicted. ch08 ties ch06 at
1500 (2.25 Mpx) and wins from 2000 on. Against the same pack it wins
from 3000 on.

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

**ch08 loses to ch05 on two of them**: `dots_4k` (16.08 vs 4.95 ms) and
`strip_1px_x` (8.01 vs 5.36). ch05's `dots_4k` samples ran at a 510
MHz median SM clock (its host work idles the GPU), so the true gap is
larger. The hybrid lookup turned the other ch05 losses into wins.

The `dots_4k` GPU window also logged one sample at 210 MHz SM and 405
MHz memory (P3), with a 2,070 MHz SM median, so its ch08 and ch06 times
may carry a small dip.

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
the neighbour tile, with the same CAS and collision rule.

Tests (`test_correctness.py`, 1,534, 8 of them behind
`FLOOD_FILL_SLOW=1`):

- the closure against a CPU flood (both exits and ch07's loop)
- K2's records and the index bytes against a CPU model built from pixels
- `pack_tiles` against ch06's pack on odd sizes
- a 27-config end-to-end matrix on 49 scenes, every invariant checked,
  including the bare (timed) twins of every lookup at both connectivities
- paint equal to ch06 and ch05; every seed rule giving identical labels
- the valve, rounds past the 252 stamp slots, five fuzz families at 1
  block and at max blocks (every lookup, every sixth case bare)
- poisoned engine reuse, repeatability, twins, limits, launch grids
  that are not whole warps, the E1-only pack store

ch07's 157 tests and ch06's 109 also pass.

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
    took over 2 s (the design's rule).

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

1. **Dense tiles (E11).** `dots_4k` and `random_4000` are bound by one
   closure per comp, in K2 and again in K4. A bulk path for isolated
   pixels (`m & ~dilate(m)`) or a shared-memory union-find for dense
   tiles would attack both.
2. **An index for dense tiles only.** The index lookup wins on long
   shapes (serpentine 0.421 vs 0.558 ms) but costs on `input_blobs`
   because K2 indexes every tile with 2+ ring comps.
3. **Degenerate shapes.** A 1-px strip puts 32 px in a tile: 131,072
   tiles and about 416 MB of buffers for 4 Mpx. A 1-D tiling, or
   routing thin images to another method, would fix both.
4. **Port `pack_tiles` to ch06.** It is worth 0.33 ms there (2.617 to
   2.291 ms on input_blobs).
5. **Small images.** The waves' fixed cost (about 0.04 ms) loses below
   1500 x 1500 px. CUDA graphs and a narrow-front mode without
   `grid.sync` are candidates.
6. **The paint floor and `ncu`.** Why scattered bytes cap near 63 GB/s
   is unproven. `ncu` needs the NVIDIA Control Panel's "Allow access to
   the GPU performance counters to all users" and a full Windows
   restart (`ERR_NVGPUCTRPERM` until then).

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

## Files

```
kernels.py           K0-K4, the off-clock twins, factories and caches
flood_fill.py        TileWaves (stages, run), recolor(), max_blocks
cpu_oracle.py        ch05's 8-conn oracle, a 4-conn port with the same labels
cpu_model.py         per-tile comps from pixels, the wave model with the valve
scenes.py            adversarial scenes (staircase, snake, frame combs, ...)
                     and five fuzz families
test_correctness.py  1,534 tests (8 behind FLOOD_FILL_SLOW=1)
benchmarks/
  benchmark.py       the interleaved session vs ch06 and ch05
  experiments.py     E1-E10, E4b, E13, R8 and E9, with the decision rules
```

Result files are in `results/ch08_gpu_nblob_tiles/benchmark_results/`.
`waves_20261015T183401Z` is this page's session. `waves_160144Z` is the
draft's, from the benchmark before the fairness review. The experiment
sessions are named above.

## Run

```bash
uv run pytest src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/
FLOOD_FILL_SLOW=1 FLOOD_FILL_IMAGES=images/input uv run pytest src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.benchmark --images images/input
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.experiments --images images/input
```
