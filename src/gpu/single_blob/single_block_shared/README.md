# Single-Block Shared-Memory BFS Flood Fill

One CUDA block runs the whole level-synchronous BFS. The frontier lives in a
**ring buffer in shared memory** (8192 int32 linear indices = 32 KB of the
48 KB static limit); the image, visited mask, depth map, and counters live in
global memory. This is the clean successor to the historical
`../single_block.py` (Stage 3), fixing its non-wrapping queue, silent
overflow drop, CAS-before-red-check bug, debug coloring, and hardcoded
thread count.

Two kernels, selected by `flood_fill(..., variant=...)`:

- **v1 `"ring"`** — pure shared-memory queue. Fastest possible frontier
  access, but scenes whose peak frontier exceeds 8192 slots abort loudly
  (the overflow tripwire).
- **v2 `"spill"`** — two-tier queue: the same shared ring as the fast path
  plus a **global-memory spill tier** that absorbs whatever the ring cannot
  hold, with a **warp-aggregated enqueue** (one shared atomic per warp per
  tier, ported from `../persistent/`). Oversized frontiers become slower,
  not fatal — any blob that fits in memory completes.

## Design in three sentences

Each BFS level occupies the virtual index window `[front, rear)` of a ring
addressed by `index & 8191`; `front` is frozen during a level, so every
enqueue ticket that passes the occupancy check `ticket - front < 8192` maps
to a distinct free slot and can never overwrite a live entry. A pixel is
claimed exactly once via `atomic.cas` on `visited` (bounds → red check →
CAS → enqueue), so double-processing is structurally impossible. Levels are
separated by two `syncthreads()` — the first publishes the level's enqueues
and final `rear`, the second guarantees everyone has read them before the
next level's atomics begin (block-local analogue of `persistent/`'s two
`grid.sync()`).

If an enqueue ever finds the ring full it sets an **overflow tripwire**; the
kernel aborts at the next level boundary and the host raises `RuntimeError`
instead of returning a silently partial fill. (v1 only — v2 has nothing to
trip, see below.)

## The v2 two-tier queue

Enqueues reserve **warp-aggregated ticket slabs**: all lanes of a warp that
won their CAS together aggregate via `activemask`/`popc`/`shfl_sync`; the
lowest lane does one `atomic.add` on the shared rear for the whole group.
Tickets inside the ring window `[front, front + 8192)` go to shared slots —
the v1 distinct-slot safety argument holds unchanged because `front` stays
frozen per level. Lanes whose tickets fall past the window are exactly the
ones active in the else branch, so they aggregate a second time and append
to the **global spill tier**. At the level boundary a third `syncthreads()`
publishes thread 0's rear clamp (tickets that diverted to the spill tier
are retracted so the next level's ring window stays contiguous), and the
next frontier is the fused window: ring entries first, then the spill
slice, one flat index space.

Spill safety is **structural, not tripwired**: the spill array holds
width×height int32 entries, appends are monotonic (slots are never
reused), and every pixel is enqueued at most once (CAS-claimed) with the
seed living in the ring — so total spill appends ≤ filled − 1 < width×height.
The v2 kernel cannot overflow and never aborts.

## Capacity limits (v1 ring; v2 removes them)

Peak ring occupancy spans two adjacent BFS levels:

| scene family                        | peak occupancy | v1 fits 8192? |
|-------------------------------------|----------------|---------------|
| center-seeded solid square, side W  | ~4W            | W ≤ ~2048     |
| corner-seeded solid square, side W  | ~2W            | W ≤ ~4096     |
| center-seeded disk, radius R        | ~5.7R          | R ≤ ~1400     |
| serpentine                          | O(width)       | always        |

Measured proof: the corner-seeded 4000×4000 full-red scene peaks at
**7999** occupied slots — exactly the predicted 2W−1, 193 slots under
capacity. The overflow test (center-seeded 2600² full-bleed) trips the wire
at ~level 1024 as predicted by 8r+4 > 8192. Under v2 those same rules just
predict *spilled pixels* instead of failure: the center-seeded 6000² scene
peaks at 23,994 queued pixels (~4W) and completes with 15.6M pixels routed
through the spill tier.

## Metrics reported (`FloodFillResult`)

- **Timing decomposition**: `alloc_ms / h2d_ms / kernel_ms / d2h_ms /
  total_ms` (perf_counter + `cuda.synchronize` around each phase).
- **BFS shape**: `levels`, `peak_level` (largest frontier),
  `peak_occupancy` (max ring usage), `level_sizes` (per-level frontier
  trace — the "frontier size over time" curve).
- **Utilization**: `thread_util_pct` (avg fraction of block threads with
  work per level), `warp_engagement_pct`, `lane_efficiency_pct`,
  `occupancy_pct` (theoretical, tpb/1536). `sm_utilization_pct` is a
  **constant 1/24 ≈ 4.2% by design** — one block pins one SM; blocks/SM
  become live metrics only at the multi-block stage.
- **Work efficiency**: `processed` (== `filled`: exactly-once processing,
  asserted in tests), `cas_attempts`, `discovery_redundancy`
  (attempts/(filled−1): how many threads raced to claim each pixel; 1.0 =
  zero duplicated discovery), `neighbor_check_efficiency_pct`.
- **Spill tier** (v2; zero under v1 or when the scene fits): `spilled`
  (total pixels routed through the global tier), `peak_spill_window`
  (largest single-level spill count). For v2, `peak_occupancy` counts both
  tiers — compare it against `ring_capacity` to see how far past the ring
  a scene went.

## Results (RTX 4060 Laptop GPU, median of 5, tpb=256)

"trip" = the v1 ring kernel's overflow tripwire fired; those scenes exist
only on v2. Best GPU-vs-@njit speedup per row in bold where the GPU wins.

| scene              | filled | levels | ring ms | spill ms | spilled px | @njit ms | best vs @njit |
|--------------------|--------|--------|---------|----------|------------|----------|---------------|
| sq_256 center      | 16K    | 129    | 0.8     | 0.8      | 0          | 0.14     | 0.18× (CPU wins) |
| sq_1024 center     | 262K   | 513    | 6.5     | 7.1      | 0          | 2.5      | 0.38× |
| sq_2000 center     | 1M     | 1001   | 25.6    | 20.0     | 0          | 14.9     | 0.75× |
| sq_4000 corner     | 16M    | 7999   | 243     | 258      | 0          | 327      | **1.35×** |
| serpentine_256     | 33K    | 32896  | 55.9    | 55.1     | 0          | 0.33     | 0.006× (worst case) |
| disk_1024          | 724K   | 679    | 11.2    | 12.4     | 0          | 9.0      | 0.80× |
| sq_2600 full center| 6.8M   | 2601   | trip    | 113      | 304,702    | 137      | **1.21×** |
| sq_4000 center     | 16M    | 4001   | trip    | 267      | 3,810,302  | 385      | **1.44×** |
| sq_5000 center     | 25M    | 5001   | trip    | 416      | 8,714,302  | 762      | **1.83×** |
| sq_6000 center     | 36M    | 6001   | trip    | 596      | 15,618,302 | 1,212    | **2.03×** |

The honest story this stage tells: one block (1/24 of the GPU) crushes
pure-Python BFS, only beats compiled CPU code once the image is large
enough (~16M px) to amortize per-level sync and launch overhead, and is
catastrophically wrong for serpentines — 32,896 levels of ~1-pixel
frontiers leave 255 of 256 threads idle (thread utilization < 1%) while
pure Python wins on the same scene. The v2 columns extend it: on scenes
that fit the ring, the spill machinery costs roughly nothing (its
warp-aggregated enqueue even wins on sq_2000), and on the center-seeded
giants v1 cannot run at all, the GPU's lead over @njit *grows* with blob
size — 2.03× at 36M px even with 43% of the blob routed through the
global tier. The tpb sweep on sq_2000 shows wide frontiers reward more
threads (64 → 18 Mpx/s, 1024 → 86 Mpx/s). The remaining ceilings — 1 SM,
per-level `syncthreads`, frontier-starved parallelism — are precisely what
the multi-block (`../multi-blocks/`) and persistent cooperative
(`../persistent/`) stages remove.

## Run

```bash
# Correctness (29 tests vs the 4-connectivity @njit CPU reference).
# Run per-directory: this file shares basenames with persistent/'s modules.
uv run pytest src/gpu/single_blob/single_block_shared/test_correctness.py -v

# Benchmark: pure-Python vs @njit vs GPU + tpb sweep; writes JSON/CSV
# (with per-level frontier traces) to benchmark_results/ next to the script.
uv run python src/gpu/single_blob/single_block_shared/benchmark.py
uv run python src/gpu/single_blob/single_block_shared/benchmark.py --slow  # + pure Python on 16M px

# Render the newest benchmark JSON as an interactive HTML dashboard
# (benchmark_results/single_block_benchmark.html — open in a browser).
uv run python src/gpu/single_blob/single_block_shared/visualize.py
```

Note: this package is 4-connected (matching `src/cpu/sequential.py` and its
own `reference.py`); `persistent/` and `multi-blocks/` are 8-connected, so
fill results are not comparable across connectivity.

## Roadmap

Done in v2: spill-to-global overflow (two-tier queue) and the
warp-aggregated enqueue. Remaining ideas:

- ~~Wavefront visualization~~ — done at the dual-block stage:
  `../dual_block/wavefront.py` animates the recorded `depth` map (the BFS
  is identical, so its renders cover this stage's kernels too).
- **ncu profiling guide**: measured (not derived) occupancy, memory
  throughput, and atomic contention for both kernels.
