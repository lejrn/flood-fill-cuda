# Chapter 8 - many blobs, waves over tile components (`ch08_gpu_nblob_tiles/`)

> **Draft.** The numbers are measured and final for this session; the
> prose, figures and the top-level chapter index are not done yet.

**Can ch07's tiles label many blobs, and how close do they get to ch06?**
Yes, if the BFS node is a tile component, not a tile.

On `input_blobs.png` the tile waves beat ch06's run-based labelling.
They stay BFS-like: labels travel as frontier (חזית) waves.

| `input_blobs.png` (81 Mpx, 2,522 blobs) | ch05 `split_L8` | ch06 runs | ch06 + ch08's pack | **ch08** |
|---|---:|---:|---:|---:|
| rgb contract (RGB in, recoloured RGB out) | 45.17 (host clock) | 2.756 | 2.476 | **2.242** |
| mask contract (packed mask in) | | 1.237 | | **1.072** |
| label-only (no paint, no pack) | | 0.630 | | **0.384** |
| ch08 / baseline, rgb (per-round median [range]) | about 1/20 (different clocks) | **0.827** [0.776, 0.961] | **0.927** [0.856, 1.031] | |

All times are medians in ms from one session,
`waves_20261015T160144Z.json`, unless a line names another session.
Ratios are medians of the per-round ratios, with their range.

**The short version.** About half of the 0.51 ms gain is the faster
pack read (1.217 vs 1.452 ms), which ch06 can take too: the ch06+tp
column. The other half is the connectivity core: 0.384 vs 0.630 ms.

ch08 does that work on 36,863 tile comps. ch06 does it on 539,207 runs.

Tiles still lose where tiles are dense: noise, the 1-px serpentine, and
zigzags on tile borders. The index twin halves some of those losses.

## The question

ch07 filled ONE blob by a BFS over 32x32 tiles closed in registers
(אוגרים). ch05 and ch06 labelled MANY blobs.

The job here is ch05's: find every blob and give it its canonical
(קנוני) label, with no seeds given. The method must stay BFS-like.

## From tiles to comps

**A tile can hold a blob more than once.** A U-shaped blob that leaves a
tile and comes back gives two local components (רכיבים) in that tile.

In ch07 such a tile was opened once per arriving wave. That was the
revisit (ביקור חוזר): 87,840 openings on 62,500 tiles in `sq_8000`.

It was not overlap. A revisit happened only when new pixels arrived,
and those pixels belonged to a comp not yet reached. But each opening
paid for the whole tile.

**So the node of the BFS is the comp.** Two comps are adjacent when two
of their pixels are 8-adjacent across a tile border.

Every comp is claimed exactly once. The tests assert it on every run:
`CLAIMS + SEEDS + VALVE_SEEDS == RING_COMPS`.

A **ring comp** touches the 124-pixel frame of its tile, so it may have
neighbours in other tiles. An **interior comp** touches no frame pixel:
it is a whole blob by itself, and its label is its own anchor (עוגן).

On `input_blobs`, 28,276 tiles are touched. They hold 36,863 comps:
36,788 ring comps and 75 interior ones, at most 8 per tile.

## The pipeline

One stream, five launches, a CUDA event after each. The mask contract
drops K1.

| # | kernel | what it does | input_blobs (ms) |
|---|---|---|---:|
| K1 | `pack_tiles` | RGB to 1 bit/px, read tile by tile (design 3's read), row-major store equal to ch06's mask | 1.217 |
| K0 | `init` | zero the queue state and the counters | 0.004 |
| K2 | `label` | one warp per tile: close every comp in registers, write a 32-byte record per ring comp, seed it or leave it unclaimed | 0.256 |
| K3 | `waves` | cooperative (שיגור שיתופי): rounds of CAS claims, one `grid.sync` per round, then one union pass and two flatten passes | 0.121 |
| K4 | `paint` | `palette[label % 6]` on red, 96 contiguous bytes per row word | 0.643 |
| | **total** | first event to last | **2.242** |

ch06 in the same rounds: pack 1.452, count 0.048, scan 0.024, emit
0.221, merge 0.304, flatten 0.032, paint 0.636.

### K2: comps, records, seeds

A warp holds the tile as 32 words, lane i is row `x0+i`. The comp loop
takes the lowest remaining pixel and closes its comp with ch07's
closure (סגור).

That pixel is the min lin of what remains, so it is the comp's anchor.
The same order is replayed by K4.

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

**Collisions (התנגשויות).** When a wave meets a comp claimed by
another wave, the pair of seed ids is recorded, but only from the SUCC
side of the edge. Every edge has exactly one SUCC side, so every
collision is seen once.

After the last round, one pass unions the pairs with ch06's
`_union` (verbatim). Two flatten passes then give each class the
minimum anchor of its comps: ch05's canonical label, not just an
equivalent partition.

**Escalation valve (שסתום).** If round 128 still claims comps, every
unclaimed ring comp seeds itself. It never fires on the six benchmark
scenes (max 8 rounds at L8).

### K4: paint

Most tiles need no closure at all. If every comp of a tile is a ring
comp and all labels give one palette colour, the whole mask word is
painted in that colour.

On `input_blobs`: 27,534 tiles painted uniform, 68 single interior comps,
674 tiles re-closed. Paint costs what ch06's does (0.643 vs 0.636 ms):
both are scattered byte writes of the same 40 MB.

## Results (RTX 4060 Laptop, 24 SMs)

Session `waves_20261015T160144Z`. Policy (ch07's): restore the image,
96 MB L2 flush read, a ~2 ms GPU sleep kernel, CUDA events, 9
interleaved (משולב לסירוגין) rounds, order reversed on odd rounds.

Clocks: P0, 2070 MHz SM for nearly every sample. On `input_blobs` the
sampler saw a dip to 1875 MHz and 7001 MHz memory (74 samples).

### The six scenes

| scene | ch08 L8 (scan) | ch08 index twin | ch06 rgb | ch08 / ch06 | ch06+tp | ch05 (host) |
|---|---:|---:|---:|---:|---:|---:|
| input_blobs | **2.242** | 2.344 | 2.756 | **0.827** | 2.476 | 45.17 |
| input_blocks | 0.259 | 0.242 | 0.214 | 1.211 | 0.212 | 1.21 |
| blob_grid_100 | **0.556** | 0.564 | 0.683 | **0.827** | 0.633 | 26.84 |
| random_4000 | 7.460 | 6.570 | 5.924 | 1.258 | 5.874 | 21.54 |
| disk_r2000 | **0.550** | 0.556 | 1.029 | **0.535** | 0.959 | 24.80 |
| serpentine_2048 | 0.613 | 0.415 | 0.314 | 1.967 | 0.301 | 18.22 |

Label-only tier (ms, ch08 / ch06): input_blobs 0.384 / 0.630, blocks
0.141 / 0.051, blob_grid 0.137 / 0.102, random 3.903 / 2.343, disk
0.132 / 0.081, serpentine 0.512 / 0.044.

**Where ch08 wins:** few comps per tile. On the disk ch06's run paint is
slow on long runs (0.615 vs 0.155 ms); ch08 paints whole words.

**Where it loses:** many comps per tile. `random_4000` has up to 86
comps per tile, closed one after another (label 2.48 ms); paint costs
3.30 ms for both chapters.

The serpentine has 16 comps per tile in one long chain. Its 2.03M
signature tests are what the index twin removes (129k).

### The crop sweep

Centred crops of `input_blobs`, ch08 L8 vs ch06 (rgb):

| side | 1000 | 1500 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ch08 (ms) | 0.102 | 0.129 | 0.176 | 0.280 | 0.444 | 0.712 | 0.967 | 1.329 | 1.735 | 2.208 |
| ch06 (ms) | 0.147 | 0.185 | 0.249 | 0.342 | 0.562 | 0.888 | 1.242 | 1.657 | 2.173 | 2.681 |
| ratio | 0.687 | 0.691 | 0.700 | 0.817 | 0.790 | 0.795 | 0.781 | 0.803 | 0.799 | 0.824 |

There is no crossover: ch08 wins at every size. The design guessed one
near 2000-4000, from the waves' fixed cost.

The waves do cost more on crops (0.051 ms at 2000 against ch06's
merge 0.020). But ch08's paint is about 2x faster there (0.028 vs 0.097
ms at 2000, 0.067 vs 0.137 at 4000), and the two paints converge on
the full image (0.643 vs 0.636). Why ch06's paint scales that way is
not measured yet.

### The work, counted (input_blobs, L8)

| | ch08 | |
|---|---:|---|
| seeds | 11,609 | 3,673 from the lattice |
| rounds | 6 | frontier comps 11,586 / 14,744 / 7,834 / 2,306 / 289 / 6 |
| frontier tiles per round | 8,579 / 13,837 / 6,744 / 1,700 / 205 / 5 | 31,070 tile visits on 28,276 touched tiles |
| tasks (comp, direction) | 136,764 | |
| signature tests | 170,995 | 1.25 per task |
| claims | 25,179 | |
| collisions recorded | 31,104 | varies run to run (a race) |
| links (unions that retired a root) | 9,162 | = seeds - roots |
| closure iterations | 76,355 | |
| atomics, without the unions' atomicMin | 205,843 | |
| ch06 | 539,207 runs, 681,039 union attempts | |

Per round (instrumented twin): 66.6, 89.1, 44.0, 14.3, 6.1, 6.1 us;
union pass 20.5 us, flatten 46.1 us.

### Seeds and the dial (חוגה), E6

| L | seeds | rounds | claims | union calls | links | input_blobs (ms) | disk | serpentine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (S2 alone) | 7,936 | 8 | 28,852 | 20,594 | 5,489 | 2.217 | 0.845 | 0.902 |
| 32 | 8,849 | 8 | 27,939 | 23,388 | 6,402 | 2.226 | 0.584 | 0.771 |
| 16 | 9,796 | 8 | 26,992 | 25,884 | 7,349 | 2.226 | 0.558 | 0.658 |
| **8** | 11,609 | 6 | 25,179 | 31,059 | 9,162 | 2.223 | 0.550 | 0.609 |
| 4 | 15,167 | 4 | 21,621 | 39,926 | 12,720 | 2.217 | 0.551 | 0.592 |
| 2 | 22,335 | 2 | 14,453 | 54,899 | 19,888 | 2.214 | 0.553 | 0.586 |
| 1 | 36,788 | 1 | 0 | 72,127 | 34,341 | 2.226 | 0.556 | 0.642 |

Session `experiments_20261015T155420Z`. On `input_blobs` the dial does
not matter (all within 0.4%). On long shapes S2 alone pays 90 rounds
(disk) and 64 (serpentine).

L = 1 makes every ring comp a seed: zero claims, one round, pure
union-find over comp adjacencies. That is block CCL, not BFS, and it is
not faster. L8 stays the default.

### Worst cases

| scene | ch08 L8 scan | valve off | ch08 L0 | index twin | ch06 |
|---|---:|---:|---:|---:|---:|
| zz_serp_1k | 0.523 | 0.532 | 0.582 | 0.340 | 0.209 |
| maze_1k | 0.351 | 0.351 | 0.342 | 0.285 | 0.495 |
| diag_snake_159 | 0.863 (129 rounds) | 0.949 (160) | 0.865 | 0.870 | 0.602 |
| zz_lattice_4k | 15.698 | 15.693 | 15.762 | 5.443 | 1.824 |

`diag_snake_159` is a chain of 160 comps through tile corners that no
lattice tile touches: one seed. The valve cuts it at round 128.

`zz_lattice_4k` puts 60 ring comps in each of 16,384 tiles. The scan
does 64M signature tests (13.3 ms of waves); the index does 2.0M
(2.9 ms). 306,695 collisions overflow the soft cap and become inline
unions.

## Experiments E1-E10

Sessions `experiments_20261015T155420Z` and `155822Z` (E4b), same
policy, 9 interleaved rounds each.

| id | question | result | decision |
|---|---|---|---|
| E1 | pack store | input_blobs: ch06 1.450, direct 1.181 (206 GB/s), smem 1.188, tile-major ref 1.148 | direct |
| E2 | closure exits | label 0.305 (ch07 loop), 0.268 (frontier), 0.268 (both) | both kept |
| E3 | K2 registers, grid | 80 regs at 1536 x 128: label 0.257 vs 0.268, no spill, also faster on blocks and random | adopted for scan |
| E4 | lookup | index: input_blobs +1.1% (per-round), random -12%, serpentine -32%, blocks -7% | scan stays (see deviations) |
| E4b | lookup x K2 | index (uncapped K2) vs scan at E3's pick: input_blobs +1.3% | scan stays |
| E5 | paint | 256 threads: input_blobs 0.636 vs 0.638 (tie), input_blocks +6.5%; fast path off +2-10% | 768 x 128, fast path on |
| E6 | seed dial | table above | L8 |
| E7 | union timing | inline: waves 0.131 vs 0.120 ms on input_blobs | deferred |
| E8 | counter cost | instrumented: waves +0.09, label +0.11 ms on input_blobs | bare twins timed |
| E9 | valve round, at L0 | 32: snake 0.582 vs 0.865, serpentine 0.783 vs 0.900; ties on maze_1k and zz_serp_1k | 128 (the coded rule needs a strict win on every worst case; 32 is defensible) |
| E10 | waves launch | 256 vs 128 threads: identical (72 vs 144 blocks) | 256 |

The prototype's ~0.10 ms waves floor on 3-5 round scenes is gone:
`blob_grid_100` waves take 0.054 ms in 5 rounds, `input_blocks` 0.071 ms
in 3. Counters are reduced per warp now, and timed runs are bare.

## Correctness

The proof (design section 7) in five lines:

1. Comps partition the red pixels, and the comp graph's components are
   exactly the blobs.
2. Every multi-comp blob has a seed in its lex-min tile.
3. Every crossing ring comp is processed once with its full dirs.
4. Every edge ends with equal labels or a pair recorded from its SUCC
   side, so the unions join exactly each blob's seeds.
5. Flatten takes the min anchor into the root: the canonical label.

Tests (`test_correctness.py`, 962): the closure against a CPU flood
(both exits and ch07's loop), K2's records and the index bytes against a
CPU model built from pixels, `pack_tiles` against ch06's pack on odd
sizes, a 16-config end-to-end matrix on 48 scenes with every invariant,
paint equal to ch06 and ch05, every seed rule giving identical labels,
the valve, five fuzz families at 1 block and at max blocks, poisoned
engine reuse, repeatability, twins, limits. The six benchmark scenes and
two 16 Mpx strips run with `FLOOD_FILL_SLOW=1`.

## Deviations from the design (and why)

1. **The index twin is built, and scan stays the default.** E4's rule
   ("input_blobs moves under +1%") passed by the ratio of medians
   (1.006) and failed by the per-round ratio median (1.011; E4b 1.013).
   The per-round median is the design's pair statistic (13.4), and pack
   noise (the same kernel in both arms) moved the other one. The phases
   that differ cost +0.033 ms on input_blobs. The session JSON still
   stores the first reading; the code now applies the stricter one.
2. **The index covers only tiles with 2+ ring comps.** A single ring
   comp is ordinal 0, so K3 takes it without a lookup. Indexing every
   touched tile cost +2-4% on input_blobs in a first A/B.
3. **K2 is capped at 80 registers on a 1536-block grid** (E3), scan twin
   only: the index twin's K2 spills 16 bytes at 80.
4. **E5's mechanical winner is not taken.** 256-thread paint won by
   0.3% on input_blobs (a tie) and lost 6.5% on input_blocks.
5. **The respin is GPU-bound.** A respin of back-to-back runs is
   launch-bound on small scenes: a smoke session timed `input_blocks`
   at 585-790 MHz SM and 810 MHz memory. The respin now interleaves 96
   MB flush reads with runs; every scene since ran at P0.
6. **ch06 is timed bare** (`instrumented=False`), as ch08 is.
7. **Not counted:** `FIND_STEPS` and the unions' atomicMin calls.
   ch06's `_find` and `_union` are imported verbatim (risk 8), so the
   "atomics" column excludes them.
8. **Not built:** E5's touched-tile list arm, E8's per-thread-atomics
   arm, the optional E11 dense-tile path and E12 minprop twin.
9. **The valve scene is an anti-diagonal snake.** A lin-monotone zigzag
   on tile borders is full of S2 seeds (the comps on the low side of an
   x-border have only SUCC neighbours), so it never needs the valve.
10. **ch05 got 3 samples on input_blobs**: its first call took 2.2 s
    (the design's rule: over 2 s, 3 samples).

## Where tiles lose

- **Dense noise.** `random_4000`: 907,448 comps, up to 86 per tile,
  closed one at a time (label 2.48 ms, 1.79M closure trips). ch06's
  merge does it in 1.43 ms.
- **Long thin shapes on tile borders.** `zz_serp_1k` and the serpentine
  have 16-32 ring comps per tile; even the index twin trails ch06.
- **Zigzags on every border.** `zz_lattice_4k`: 3.0x behind ch06 with
  the index, 8.6x with the scan.

## Open problems

1. **A dense-tile path (E11).** Label tiles with many corners by a
   run union-find in shared memory instead of sequential closures.
2. **The paint floor.** Both chapters paint at ~63 GB/s of scattered
   bytes; why is unproven without a profiler. `ncu` needs the GPU
   performance counters enabled (NVIDIA Control Panel, then a full
   Windows restart).
3. **A bigger collision buffer on dense scenes**, or a per-round union
   pass, so `zz_lattice` stops paying 300k inline unions.
4. **Port `pack_tiles` to ch06.** It is worth 0.28 ms there (ch06+tp).

## Glossary

- **Tile** (אריח) - 32 rows x 32 columns, one warp per tile.
- **Comp** (רכיב מקומי) - a blob restricted to one tile.
- **Ring comp / interior comp** - touches the tile frame / does not.
- **Anchor** (עוגן) - a comp's minimum linear index `x*H + y`.
- **Canonical label** (תווית קנונית) - a blob's minimum anchor.
- **Seed** (זרע) - a comp that starts a wave with its own id.
- **Lattice** (סריג) - the tiles with `(tx + ty) % L == 0`, all seeds.
- **Wave** (גל) - the comps that carry one seed's label.
- **Collision** (התנגשות) - two waves meeting on one edge.
- **Union-find** (איחוד-חיפוש) - ch06's structure for joining labels.
- **Valve** (שסתום) - the round after which unclaimed comps self-seed.
- **Dial** (חוגה) - the lattice spacing L, from pure waves to pure CCL.
- **Ordinal** (מספר סידורי) - a ring comp's index in its tile.
- **Interleaving** (שזירה) - timing every config once per round.

## Files

```
kernels.py           K0-K4, the off-clock twins, factories and caches
flood_fill.py        TileWaves, recolor(), phase_times, max_blocks
cpu_oracle.py        ch05's 8-conn oracle, a 4-conn port with the same labels
cpu_model.py         per-tile comps from pixels, the wave model with the valve
scenes.py            adversarial scenes and five fuzz families
test_correctness.py  962 tests (8 behind FLOOD_FILL_SLOW=1)
benchmarks/
  benchmark.py       the interleaved session vs ch06 and ch05
  experiments.py     E1-E10 and E4b, with the decision rules
```

Results: `results/ch08_gpu_nblob_tiles/benchmark_results/`.

## Run

```bash
uv run pytest src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/
FLOOD_FILL_SLOW=1 FLOOD_FILL_IMAGES=images/input uv run pytest src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.benchmark --images images/input
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.experiments --images images/input
```
