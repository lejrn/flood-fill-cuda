# Act 3 floors: tiles (ch07) and waves (ch08)

Real data for the chain cut's third act (v5). Every array was recorded
from the chapters' own GPU kernels on the RTX 4060 Laptop, then checked
against the CPU oracles (אורקל) and the chapters' CPU model.

These files hold mechanism data: rounds, tiles, pieces, seeds, labels.
They hold no timings. Every CPU-vs-GPU ratio the video shows comes from
the bench session (`src/flood_fill_cuda/overview/bench_v5.py`), not
from here.

| file | what |
|---|---|
| `ch07_tiles_20261017T160022Z.npz` / `.json` | ch07 on `sq_8000_center`, and one warp close-up on `input_blobs.png` |
| `ch08_waves_20261017T160051Z.npz` / `.json` | ch08 on a 384 px crop of `input_blobs.png`, and the full frame's facts |

## Provenance (מקור)

| | ch07 | ch08 |
|---|---|---|
| script | `src/flood_fill_cuda/chapters/ch07_gpu_1blob_tiles/benchmarks/video_tiles.py` | `src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/benchmarks/video_waves.py` |
| HEAD at the run (`commit` in the JSON) | `43f3c8f` | `ea2b0a4` |
| script at that HEAD | as committed in `43f3c8f` | as committed in `83f7ad3` (`ea2b0a4` changed only `overview/`) |
| chapter sessions it follows | `tiles_20261016T064724Z` | `dense_20261016T054922Z`, `waves_20261015T183401Z` |
| config | default switches (one_sync, tile_paint), 8-conn, 72 x 128, instrumented | `TileWaves` defaults (dense, dindex, lattice 8, valve 16), 8-conn, instrumented |
| input | `input_blobs.png`, md5 `47639eb37f973b5092ec2c4db86f45a4` | the same |

After the runs, one field of each JSON was normalised: `device` lost
the driver's NUL padding (commit `4246bb0` fixed the scripts). Nothing
else was edited.

Each JSON has a `checks` dict: every check the script ran, all `true`.
A failed check stops the script before it writes anything.

To regenerate (new UTC stamps, new files):

```
uv run python -m flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks.video_tiles --images images/input
uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.video_waves --images images/input
```

## Index order: read this first

Every array is indexed like the kernels: `[x, y]`, where x is the PNG
column and y the PNG row. `array.T` is the PNG orientation.

A tile `(tx, ty)` covers x in `[32 tx, 32 tx + 32)` and y in
`[32 ty, 32 ty + 32)`. Its index is `t = tx * tiles_y + ty`.

Inside a tile, lane i of the warp holds x-row `x0 + i` as one 32-bit
word, and bit j of that word is `y0 + j` (little-endian bit order:
`np.unpackbits(words.view(np.uint8).reshape(32, 4), axis=1,
bitorder="little")` gives a 32 x 32 grid `[lane, bit]`).

The chain floors draw `[x, y]` as (row, col) (`scenes/chain/state.py`).
So these arrays drop onto a floor as they are, with lanes as screen rows
and the "row fill" running along a screen row.

The 900 px frame pictures (`assets/ch05_input_blobs/frame_*.png`) are in
PNG orientation. A zoom from them into these crops needs `.T` on one
side.

## ch07: `ch07_tiles_20261017T160022Z`

### How it was recorded: an export twin

The chapter kernel keeps no per-round record. Its per-tile stamp is
raised with `atomic.max`, so after a run it holds the LAST round a tile
was queued for, not the first.

So the script builds an export twin: the chapter's own factory
(`kernels._make_rounds_kernel`) with four insertions, made by an
anchored source patch (each anchor must occur exactly once):

- `first_open[t]`: written by the one warp whose `atomic.max` on t's
  stamp returned -1, the first enqueue ever;
- `round_size[r]`: the frontier size of round r;
- `visit_count[t]`: one per visit;
- `probe`: for chosen tiles, the mask, entry and closed words of each
  visit.

The twin links at 168 registers (the chapter kernel 144). Both seat 72
cooperative blocks of 128, so the twin runs on the chapter's own grid.

This is not a CPU replay. The close-up's sweeps are replayed on the GPU,
by one warp, with the chapter's own device functions (below).

### The square: `sq_*` arrays

`sq_8000_center`: 8000 x 8000, every pixel red, seed (4000, 4000) in
tile (125, 125). 250 x 250 tiles.

| array | shape, dtype | meaning |
|---|---|---|
| `sq_first_open` | (250, 250) int16 | the round each tile was first visited (0 = the seed tile) |
| `sq_opened_per_round` | (127,) int32 | tiles opened for the first time in each round |
| `sq_round_size` | (127,) int32 | the frontier of each round: first visits plus revisits (twin run 0) |
| `sq_visit_count` | (250, 250) int16 | visits per tile, 1 or 2 (twin run 0) |
| `sq_last_round` | (250, 250) int16 | the stamp after twin run 0: the last round each tile was queued for |
| `sq_chapter_last_round` | (250, 250) int16 | the same stamp from the chapter kernel's own run |

Checks, on 5 twin runs: the fill equals the chapter kernel's and the
`@njit` oracle's visited set, bit for bit. Rounds equal the chapter
kernel's (127). `sum(round_size)` equals the visit counter and
`sum(visit_count)`. `first_open` is set on every tile, never exceeds
the stamp, and is the same on every run.

**Why first-open rounds do not move.** On a full square every tile is
full, so its closure fills all of it from any entry bit. Round 0 closes
the seed tile and exports into its 8 neighbours.

By induction, a tile at Chebyshev distance (מרחק צ'בישב) d from the
seed tile is first queued in round d - 1, by a neighbour at distance
d - 1. Nothing nearer could reach it earlier.

So `sq_first_open` is the Chebyshev distance, on every run. The export
confirms it on all 62,500 tiles (`first_open_equals_chebyshev`).

The last round, 126, opens nothing: it only revisits tiles of the last
ring, re-queued by a neighbour's export that landed after their closure.

Visits are timing-dependent. They are a record of these runs, not a
property of the square.

### The warp close-up: `cu_*` arrays

The largest blob of `input_blobs.png` (76,090 px, a rough-edged clump,
canonical label 48,568,496), seeded at the blob pixel nearest its
centroid, (5602, 4476). The tile is (176, 142), origin (5632, 4544), on
the blob's edge.

It was picked by eye among the 63 tiles of the 12 largest blobs that
qualify (visited once, 30-75% red, no other blob's pixels, not the seed
tile). Its entry is a partial stripe on one border, its edge is
concave, and every sweep of the first two trips adds pixels.

| array | shape, dtype | meaning |
|---|---|---|
| `cu_mask` | (32,) uint32 | the 32 mask words (lane i = x-row 5632 + i) |
| `cu_entry` | (32,) uint32 | the reached words the tile's visit read: its entry bits |
| `cu_sweeps` | (13, 32) uint32 | reached words: entry, then (row fill, diagonal, column fill) per trip |
| `cu_final_chapter` | (32,) uint32 | the chapter kernel's final reached words of the tile |
| `cu_ctx_red`, `cu_ctx_fill` | (224, 224) bool | 7 x 7 tiles around it, origin (5536, 4448): red, and the blob's fill |
| `cu_ctx_first_open` | (7, 7) int16 | first-open round of those tiles in the picked run (-1: never opened) |

| sweep | entry | row | diag | col | row | diag | col | row | diag | col | row | diag | col |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reached px | 17 | 86 | 109 | 181 | 387 | 481 | 541 | 543 | 543 | 543 | 543 | 543 | 543 |

543 of the tile's 1,024 pixels are red. The 17 entry bits all sit at
bit 0 (y = 4544, the border to tile (176, 141)), in lanes 13-31 except
19 and 28.

Trip 3 adds 2 pixels, so trip 4 runs, adds nothing, and the `any_sync`
test ends the loop: 4 trips, 12 sweeps. The tile was first visited in
round 4 and visited once.

Checks: the probed mask equals the packed mask. A one-warp kernel
replays the visit with the rounds kernel's own `_row_fill`,
`_diag_step`, `_col_fill`, loop and `any_sync` test. Its last words and
trip count equal the twin's visit, and its last words equal the
chapter kernel's final words for the tile. The context fill equals the
oracle's blob.

On this irregular blob the round count moves with timing: the chapter
kernel gave 9, 10, 9, 10, 9 rounds in five runs, and the picked twin
run took 10. An export that lands before a tile's read lets the wave
run ahead. The fill does not move (273 BFS levels on the CPU).

## ch08: `ch08_waves_20261017T160051Z`

### How it was recorded

No twin is needed. After a run, the instrumented engine's buffers hold
everything:

- `rec[64 t + b]`: anchor, side words, direction bits and (after the
  flatten passes) the canonical label of ring piece b of tile t;
- `qc` / `ql`: the seeds K2 queued, then the pieces claimed in round r
  at `qc[bounds[r-1]:bounds[r]]`, each with the seed id of the wave
  that claimed it;
- `ca` / `cb`: the seed ids of two waves that met;
- the painted RGB, the label map and the canonical seeds.

The GPU keeps no per-pixel piece map. So the per-pixel piece id comes
from the chapter's CPU model (`cpu_model.tile_model`, K2's order:
lowest remaining pixel first). It is checked against every GPU record:
anchor, four side words and direction bits.

### The crop: `crop_*` arrays

A 384 x 384 crop at x = 4352, y = 5632, run as an image of its own: 12 x
12 tiles.

**Why this crop.** Its origin is a multiple of 256 px, so its tiles and
its seed lattice ((tx + ty) % 8 == 0) are the full frame's. It holds 8
blobs; 6 of them have two or more seeds, so their waves meet. 11 of its
36 seeds come from the lattice, a tile holds at most 4 pieces, and it
takes 6 rounds, like the full frame.

It was picked by eye among the 1,249 of the 2,312 aligned 384 and 512
px crops that pass a CPU-model survey (rounds >= 4, >= 2 blobs with two
seeds, <= 4 pieces per tile, <= 24 blobs). 384 px keeps a tile about
50 px wide on a 600 px floor.

Per pixel, (384, 384):

| array | dtype | meaning |
|---|---|---|
| `crop_red` | bool | the red mask |
| `crop_piece` | int8 | the piece id within its tile, K2's order (-1 off red) |
| `crop_comp` | int16 | the piece index k into the per-piece arrays (-1 off red) |
| `crop_label` | int32 | the canonical label: the blob's minimum `x * 384 + y` (-1 off red) |

Per tile, (12, 12): `crop_tile_lattice` (bool, a lattice tile) and
`crop_tile_pieces` (int8, pieces in the tile).

Per piece, (135,), index k in K2's order (tile by tile, then lowest
pixel first):

| array | meaning |
|---|---|
| `crop_comp_tile`, `crop_comp_local` | its tile t and its id within the tile |
| `crop_comp_ordinal`, `crop_comp_rec` | its ring ordinal b and record id `64 t + b` (-1: interior) |
| `crop_comp_anchor`, `crop_comp_size` | its minimum `x * 384 + y`, its pixel count |
| `crop_comp_dirs` | direction bits: 0 up (tx-1), 1 down, 2 left (ty-1), 3 right, 4 UL, 5 UR, 6 DL, 7 DR |
| `crop_comp_seed`, `crop_comp_seed_lattice_only` | a seed; a seed only because of the lattice |
| `crop_comp_round` | 0 for a seed, r if claimed in round r (-1: interior) |
| `crop_comp_wave` | k of the seed whose wave claimed it (itself for a seed) |
| `crop_comp_label` | its final canonical label |

Waves that met, (116,):

| array | meaning |
|---|---|
| `crop_coll_a`, `crop_coll_b` | k of the two seeds, (tester's wave, target's wave), in the order the kernel recorded them |
| `crop_coll_merged` | this record joined two classes in a CPU union-find run in that order (28 do) |
| `crop_meet_tester`, `crop_meet_other` | the meeting edge: two pieces claimed by different waves; the tester is the piece in the lex-smaller tile |
| `crop_meet_round` | the round in which the tester recorded it: the round after the tester was claimed |

The `coll` and `meet` lists are separate. `coll` is the kernel's list
as recorded. `meet` is the deduplicated set of edges where waves met.
In this crop they have the same length (no repeats; see the quirk
below).

### The full frame: `full_*` arrays

`input_blobs.png`, 9000 x 9000: 282 x 282 tiles.

| array | shape, dtype | meaning |
|---|---|---|
| `full_tile_pieces` | (282, 282) int8 | pieces per tile |
| `full_tile_seeds` | (282, 282) int8 | seeds per tile |
| `full_tile_round_min`, `full_tile_round_max` | (282, 282) int8 | the earliest and latest claim round among a tile's ring pieces (-1: none) |
| `full_round_900` | (900, 900) int8 | the claim round at pixel (10 i + 5, 10 j + 5): -1 off red, -2 an interior piece, else the round (0 = seed) |

### Checks (both images)

- The tile headers' piece counts and every ring record (anchor, side
  words, direction bits) equal the CPU model.
- The seeds equal the seed rule; the seed counters agree.
- Every ring piece is seeded or claimed exactly once, the valve never
  fires, and the claim round equals the piece's BFS distance to the
  nearest seed over the piece graph. So rounds and claim rounds are not
  timing-dependent.
- The label map equals the `@njit` oracle (`cpu_label`), pixel for
  pixel; so do every record's label and the canonical seeds. The
  painted RGB is `palette[label % 6]` on red and unchanged elsewhere.
- The collision list holds every meeting edge, as (tester wave, target
  wave), once, or twice where the quirk below applies. A CPU union-find
  over it retires as many roots as the kernel's `union_done`, and gives
  every piece the record's label.

**Which wave claims a piece is a race.** So `crop_comp_wave`, the
meeting edges and the collisions are this run's. Seeds, rounds, claims,
links and labels are not: the checks above pin them.

**A kernel quirk found here.** In the dindex waves kernel, a lane whose
task takes the warp-wide dense path (target tile with more than
`dense_min = 4` ring pieces) still runs the scan test of the k-loop,
which is not gated by `dense`. So targets of ordinal < 4 in such a tile
are tested twice.

Labels are unaffected: a repeated CAS or union changes nothing. But the
collision, pred-skip and CAS counters include the repeats. In this run,
142 of the full frame's 31,198 collisions are repeats; the crop has
none. The ch08 README's "every collision is seen once" does not quite
hold. The kernel was left as it is.

## Numbers worth showing

| number | value | source |
|---|---|---|
| sq_8000 BFS levels (pixel rings) | 4,001 | ch07 JSON `square.oracle_levels` (the `@njit` oracle) |
| sq_8000 rounds (tile rings + 1 revisit round) | 127 | `square.chapter_kernel.rounds`, every twin run |
| rounds that open new tiles | 126 (0-125) | `sq_opened_per_round` |
| tiles | 62,500 (250 x 250) | `square.n_tiles` |
| visits | 98,267 (chapter kernel), 99,303-99,439 (5 twin runs) | `square.chapter_kernel`, `square.twin_runs` |
| tiles visited twice | 36,803-36,939 of 62,500 | `square.twin_runs` |
| close-up: trips, sweeps, red px | 4 trips, 12 sweeps, 543 of 1,024 | `closeup` |
| crop: blobs, pieces, seeds (lattice) | 8, 135, 36 (11) | ch08 JSON `crop` |
| crop: pieces claimed per round | 50, 38, 7, 3, 1, 0 | `crop.claimed_per_round` |
| crop: meeting edges, links | 116 (this run), 28 | `crop` |
| full frame: blobs, pieces, interior | 2,522, 36,863, 75 | `full` |
| full frame: seeds (lattice-only) | 11,609 (3,673) | `full` |
| full frame: rounds, frontier | 6: 11,586 / 14,744 / 7,834 / 2,306 / 289 / 6 pieces | `full.frontier_pieces_per_round` |
| full frame: claims, links, roots | 25,179, 9,162, 2,447 | `full` |

The full frame's deterministic facts equal session
`dense_20261016T054922Z` (pieces, seeds, rounds, frontier, links,
roots, blobs).
