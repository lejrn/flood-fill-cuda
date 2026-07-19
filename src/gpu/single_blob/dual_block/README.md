# Dual-Block BFS Flood Fill — 1 block → 2 blocks

The single-block stage ended at a ceiling: one block = one SM, at most 1,024
threads. This stage adds exactly one thing — a second block — and measures
everything that changes: how two blocks can share one BFS (three
partitionings), what the inter-block barrier costs, where the two blocks
physically run (observed via `%smid`, never assumed), and whether forcing
both onto ONE SM (deeper latency hiding, same execution units) beats letting
the scheduler spread them across two.

## The three partitionings

All three run the same claim protocol (bounds → is-red → `atomic.cas` on
visited → enqueue) as one cooperative launch of 2 blocks, with two
`grid.sync()` per BFS level. They differ only in who owns which frontier
pixel:

| kernel | partition axis | frontier storage | cross-block traffic |
|---|---|---|---|
| `split` | **space**: block b owns half the image | per-block shared ring (8192) + own spill tier | tiny: inbox ≤ height entries, ever |
| `global` | **index**: grid-stride interleave | one global queue | everything (the queue is shared) |
| `dirsplit` | **direction**: right/up claims → queue 0, down/left → queue 1 | one global buffer filled from both ends | everything |

- **`split`** reuses the single-block v2 two-tier machinery per half.
  Cross-seam discoveries (only possible along the column pair
  `width//2 − 1 ↔ width//2` in 4-connectivity) go to the other block's
  **inbox** — a global-memory mailbox, structurally bounded by `height`
  entries because only one column's pixels can ever cross each way, each
  CAS-claimed once. Its weakness: a blob living in one half leaves the
  other block with literally zero work.
- **`global`** is persistent/'s design at 4-connectivity: balanced by
  construction *when frontiers exceed one block's thread count* — the
  grid-stride assigns item `front+tid` to global thread `tid`, so a
  frontier smaller than tpb lands entirely in block 0.
- **`dirsplit`** (proposed by the project owner) is spatially agnostic: a
  wavefront always expands via both direction pairs, so the queues track
  two rotating arcs of the frontier and stay ~50/50 balanced even for
  blobs the split kernel handles 100/0. The double-ended buffer is
  memory-neutral: q0 appends forward, q1 backward at slot `N−1−ticket`,
  and total CAS-claimed appends ≤ width×height means the ends can never
  meet. Its weakness is direction degeneracy — and the serpentine result
  below is a masterclass in why aggregate metrics lie.

### The split kernel's level choreography

```
process fused own window [ring | spill | inbox]     (enqueue: own half → own
        │                                            ring/spill; across the
   syncthreads          own shared atomics final     seam → their inbox)
        │
   thread 0: clamp own ring rear, publish next-level ring/spill COUNTS
        │
   grid.sync #1         enqueues + inbox rears + both pubs visible grid-wide
        │
   every thread reads the six published/inbox counters
        │
   grid.sync #2         reads ordered before the next level's atomics
        │
   advance windows; loop while total work > 0        (grid-uniform predicate:
                                                      both blocks always meet
                                                      the same barriers)
```

The early rear clamp is race-free precisely because — unlike the
single-block v2 kernel — **no thread other than thread 0 ever reads the
shared rear**: every thread takes its next windows from the published
global counts. Depth stays exact BFS distance across the seam because the
level windows are grid-synchronized: a pixel claimed at level L (from
either side) is processed at level L+1, wherever it lands.

## Placement: observed, forced, and priced

**You cannot choose which SM a block runs on.** Blocks are CUDA's unit of
scheduling freedom — they may not depend on each other's progress, so the
hardware may place them anywhere; that is what lets the same kernel scale
from 2-SM to 100-SM GPUs, and it is why there is no cross-block
`syncthreads` in the base model (`grid.sync` exists only under cooperative
launch, which first proves all blocks fit simultaneously). So this package
*records* placement instead: every kernel links a 5-line CUDA-C helper that
reads the `%smid` hardware register, and every result reports which SM each
block actually ran on. Measured here: a natural 2-block launch lands on two
different SMs (the scheduler spreads).

The **`pinned` kernel** corners the scheduler anyway: launch 48 blocks of
768 threads, and since ⌊1536/768⌋ = 2 blocks fit per SM, the only way 48
blocks fit on 24 SMs is exactly 2 per SM — measured: every SM hosts
exactly two. The first block CASes its smid into a global slot; the two
blocks matching it become the workers (provably co-resident on one SM) and
the other 46 exit. Because `grid.sync` would wait for all 48, the pair
synchronizes with a **hand-rolled sense-reversing barrier** (atomic arrival
counter + generation flag + `threadfence`; spins use atomic reads because
plain global reads may be register-cached) — deadlock-free only because
co-residency is guaranteed from launch, which is exactly the guarantee
cooperative launches formalize. It is `grid.sync` built by hand.

Same-SM (2×768 = 1,536 resident threads = 100% of one SM's residency —
beyond any single block's 1,024 cap, same 128 cores) vs spread (768/SM on
two SMs, doubled execution units) vs the single-block baseline: three
configurations, one variable at a time. The benchmark's placement section
holds the measured answer.

## Register pressure: the ceiling this stage discovered

The dual kernels carry more per-thread state than the single-block ones,
and it shows in the register file (64K per SM):

| kernel | regs/thread | consequence |
|---|---|---|
| `global` / `dirsplit` | ~104–106 | 1024-thread blocks impossible (>64K regs); tpb capped at 512 |
| `split` (uncapped) | 159 | even 512 impossible → compiled with `max_registers=120` |
| `pinned` | capped at 40 | 2 blocks × 768 per SM need ≤ 42.67 regs/thread, allocated in granules of 8 |

So `threads_per_block` here is 32–512 (2×512 = 1,024 total threads, matching
the single-block sweep's top config), and the pinned experiment exists at
all only because of an explicit register cap. Coordination costs registers;
registers cost residency.

## Metrics added over the single-block stage

- `processed_b0/b1`, `balance_pct` — per-block work split (and
  `level_sizes_per_block`, the balance *trace*: the serpentine shows 98.5%
  total balance while the per-level trace proves the blocks essentially
  never work at the same time).
- `inbox_to_b0/b1`, `inbox_pct` — cross-seam traffic (split).
- `spilled_b0/b1` — per-half spill tiers (split).
- `sm_id_b0/b1`, `same_sm` — observed placement.
- `bare=True` runs the uninstrumented twin of each kernel (identical BFS,
  counters/traces stripped) so the instrumentation overhead is measured,
  not assumed.

## Results

Run `benchmark.py` to (re)generate; the measured tables live in the
benchmark commit alongside the JSON/CSV in `benchmark_results/`.

## Run

```bash
# Correctness (98 tests vs the 4-connectivity @njit CPU reference).
# Run per-directory: this file shares basenames with the sibling packages.
uv run pytest src/gpu/single_blob/dual_block/test_correctness.py -v

# Benchmark: three partitionings vs single-block v2 vs @njit, plus the
# tpb sweep, the placement experiment, and instrumentation overhead.
uv run python src/gpu/single_blob/dual_block/benchmark.py
```

```python
from flood_fill import flood_fill
flood_fill(img, x, y, kernel="split")                      # or "global"/"dirsplit"
flood_fill(img, x, y, kernel="split", bare=True)           # uninstrumented twin
flood_fill(img, x, y, kernel="pinned",
           threads_per_block=768, placement="same_sm")     # or "spread"
```

## Glossary (הסברים)

- **SM** (Streaming Multiprocessor): one of the GPU's 24 independent
  processors — 128 cores, 4 warp schedulers, a 64K-register file, ~100 KB
  shared memory, room for 1,536 resident threads.
- **Block**: a software container of up to 1,024 threads, placed on one SM
  by the hardware scheduler (never migrates, placement not programmable).
  Co-resident blocks on one SM run *concurrently* (interleaved on the same
  cores for latency hiding), not in parallel on more hardware — and even
  then each gets a hardware-isolated slice of shared memory: same-SM
  placement does NOT let blocks read each other's shared memory.
- **Stride / צעד**: the fixed jump between consecutive items one thread
  handles. In a grid-stride loop with 2×256 threads, thread 7 takes items
  7, 519, 1031… (גודל הצעד = סך כל החוטים; החוטים משתלבים כמו שיני מסרק
  ומכסים את כל הטווח בלי חורים).
- **Inbox**: a global-memory mailbox where the *other* block deposits
  pixels it discovered but doesn't own; the owner reads it at the next
  level.
- **Window, and "advancing" it**: queues only grow; `[front, rear)` marks
  the slice belonging to the current level. At a boundary, front jumps to
  rear and rear extends over the new enqueues — the window slides, nothing
  moves.
- **Fused window**: a block's level work = ring segment + spill segment +
  inbox segment, walked as ONE flat index range so no thread idles because
  a particular segment is empty.
- **Cost ladder** (RTX 4060, rules of thumb): shared memory ~20–30 cycles;
  global memory ~400–600 cycles (L2 hit ~200), stores mostly fire-and-
  forget; `syncthreads()` tens of cycles; `grid.sync()` ~1–3 µs — roughly
  **100× a syncthreads**, which is why per-level barriers dominate
  narrow-frontier scenes.
- **Kernel argument typing**: Numba infers each argument's type at the
  first call and JIT-compiles that specialization (the warmup call absorbs
  this, plus the NVRTC compile of the linked `%smid` helper).

## Roadmap

- Dashboard (deferred by decision): 5-series runtime plot, balance-over-
  time panels, inbox/spill columns, placement comparison chart.
- The natural next stage: N blocks (the `../multi-blocks/` and
  `../persistent/` designs, revisited with this package's rigor).
