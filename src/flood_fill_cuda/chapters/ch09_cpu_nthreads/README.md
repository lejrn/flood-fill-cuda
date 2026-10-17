# Chapter 9 - every CPU core, the GPU's ideas (`ch09_cpu_nthreads/`)

**Can BFS run in parallel on the CPU's cores, and how does it compare?**
Chapters 3, 7 and 8 were compared with ONE CPU core. This chapter
asks the fair question.

The same work, the same ideas, on all 16 logical CPUs. Three
variants, each an `@njit(parallel=True)` loop with `prange`.

**Results: pending a quiet machine.** The code and the tests are done.
The benchmark is written and was smoke-tested on small sizes only.

Every timing taken while building ran next to a video render on all
cores. None of them is a result, and none is quoted here.

## Inherited problems

1. **The baseline was one core.** The video's v5 session compares the
   GPU with the same work on one core. `sq_8000` 8-conn BFS: 1,396 ms
   on one core, 3.1 ms for ch07. `input_blobs` labels: 274 ms, 2.2 ms
   for ch08. `sq_256`: 0.19 ms, 0.044 ms for ch07.
2. **The CPU has 16 logical CPUs.** An i9-13900H: 8 cores seen as 16
   threads, a mix of P- and E-cores. One core is a sixteenth of it.
3. **Numba has no CPU atomics.** Every GPU chapter leans on atomic
   (אטומית) operations: CAS claims, atomic OR, atomicMin. On the CPU
   each claim must be safe by construction, or harmless when it races.

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

## No atomics: why each variant is exact

A `prange` here always runs over **chunks**. Each chunk owns what it
writes: its slice of an output buffer, its counters, its tiles.

Where two chunks can still touch one place, it only takes whole
aligned values (one byte or one int32). A store of that size is one
instruction on x86-64, so it is never torn.

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
on every scene. Races do happen: forcing every level into the most
chunks gave a few hundred duplicate entries per run on a 1024 x 1024
square, and the output never moved.

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

So every wave that ever touched `Q` is joined with `Q`'s final wave.
Every adjacency is seen by the one expansion of one side, with that
side's final wave.

Labels, the paint and the blob count are deterministic. Which wave
wins a race, and how many meetings are recorded, are not.

**Bounded buffers.** A chunk's claims and meetings go into its own
regions, sized per entry plus one worst-case expansion. An entry that
might not fit is requeued, unchanged, for the next round. The first
entry of a chunk always fits, so every round makes progress.

## Correctness

`test_correctness.py`: **582 tests, all passing** (about 3 minutes
with `nice`).

| group | against | what |
|---|---|---|
| bit helpers | plain-Python references, a pixel BFS inside the tile | row fill, popcount, lowest bit, columns, the closure from one seed and with new seeds on flagged rows |
| one blob, both methods | ch01's 4-conn and shared/'s 8-conn `@njit` oracles | 20 scenes x 4- and 8-conn x 1, 2, 16 threads: visited set, painted image, fill count; the pixel BFS's level count too |
| one blob, forced races | the same | every level and round split into the most chunks; one engine reused across seeds and thread counts; a fuzz of rectangles, disks and noise on odd sizes |
| pack | numpy's red mask | colours that share bytes with red, padding rows, mask columns |
| many blobs | ch05's `cpu_label_components` (8-conn), ch08's `cpu_label` (4-conn) | 30 scenes x 4- and 8-conn x 1, 2, 16 threads: label map, palette image, blob count, every ring piece expanded once |
| many blobs, every path | the same | forced prange rounds, zero-size buffers (requeues), lattices 0, 1, 3, 8, ch08's five fuzz families |
| the frame | ch05's oracle | `input_blobs.png` crops, and the full 81 Mpx frame once: 2,522 blobs |
| determinism | the first run | one engine, nine runs, three thread counts, two images: identical outputs, rounds and seeds |

The scenes include serpentines, spirals, mazes, checkerboards, noise
near the percolation thresholds (0.40-0.62), zigzags on the 64-tile
borders, pixels on tile corners, sizes not multiples of 64, 1 x 1,
blank images, thin strips and a non-red seed.

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

## Results: pending a quiet machine

`benchmarks/benchmark.py` is one session for every number:

- **scaling**: 1, 2, 4, 8, 16 threads for each variant;
- **all cores vs one core**: ch09 against the one-core twins, like for
  like (buffers kept, the reset on the clock, with and without the
  paint);
- **all cores vs the GPU**: ch07 and ch08 timed by CUDA events at
  full clocks, in the same session;
- **scenes**: `sq_256`, `sq_8000_center` and `serpentine_256` at
  4- and 8-conn, `input_blobs`, the 240 drone frames.

An idle gate waits until the rest of the machine is at most 10% busy
before every CPU round and every GPU attempt.

Why the gate matters: a smoke run next to a render (load 11-15) made
the 16-thread cells far slower than the 2-thread ones. Each barrier
waits for its slowest thread, and on a full machine some thread is
always preempted (נדחק מהמעבד). That is oversubscription (רישום יתר),
not the method.

## What did not get done

- **32 x 32 tiles of `uint32`** were not compared with 64 x 64 of
  `uint64`. The task asked for the swap only if measurably better, and
  nothing could be measured.
- **The thresholds are untuned**: `serial_below` (pixel 1,024 entries,
  tiles 8, labels 64) and the pixel BFS's 256-entry minimum chunk.
- **Dense tiles.** The pieces pass closes every piece on its own, as
  ch08's first version did. A tile of 1,000 isolated pixels costs
  1,000 closures; ch08's run CCL for dense tiles is not ported.
- **The pack is not fused**, and every run resets all tile state, not
  only the tiles the last run touched.

## Glossary

- **chunk** (נתח): a contiguous slice of a `prange` loop's work; one
  iteration, so one thread, owns it and everything it writes.
- **stamp** (חותמת): a chunk's id stored beside a claim, so the claim
  that survived a race can be told from the ones that lost.
- **inbox** (תיבת דואר): a per-tile, per-side slot where a neighbour
  leaves its border; one writer per slot.
- **ring piece**: a piece that touches its tile's frame, so it may go
  on in a neighbour tile.
- **meeting**: two waves found next to each other; unioned at the end.
- **requeue**: an entry carried to the next round, unchanged, because
  its chunk's buffer might not hold its worst case.
- **oversubscription** (רישום יתר): more runnable threads than CPUs;
  a barrier then waits for threads the scheduler has set aside.

## Files

```
kernels.py           bit helpers and the tile closure; the pixel BFS;
                     pack, paint and the tile BFS; the pieces, seeds,
                     waves, unions and paint of the tile labels
flood_fill.py        PixelBFS, TileBFS, TileLabels (buffers allocated
                     once, reset on the clock), flood_fill(), label(),
                     the threads argument (numba.set_num_threads)
test_correctness.py  582 tests
benchmarks/
  benchmark.py       the one-session benchmark (not yet run)
```

## Run

```bash
nice -n 10 uv run pytest src/flood_fill_cuda/chapters/ch09_cpu_nthreads/

# on a quiet machine (about 15-25 min, about 3.5 GB of host memory):
PYTHONUNBUFFERED=1 uv run python -m \
    flood_fill_cuda.chapters.ch09_cpu_nthreads.benchmarks.benchmark

# harness check only (small sizes, no gate, nothing written):
uv run python -m flood_fill_cuda.chapters.ch09_cpu_nthreads.benchmarks.benchmark \
    --smoke --no-save
```
