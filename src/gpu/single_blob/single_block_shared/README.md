# Single-Block Shared-Memory BFS Flood Fill

One CUDA block runs the whole level-synchronous BFS. The frontier lives in a
**ring buffer in shared memory** (8192 int32 linear indices = 32 KB of the
48 KB static limit); the image, visited mask, depth map, and counters live in
global memory. This is the clean successor to the historical
`../single_block.py` (Stage 3), fixing its non-wrapping queue, silent
overflow drop, CAS-before-red-check bug, debug coloring, and hardcoded
thread count.

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
instead of returning a silently partial fill.

## Capacity limits (v1, pure shared memory)

Peak ring occupancy spans two adjacent BFS levels:

| scene family                        | peak occupancy | fits 8192?   |
|-------------------------------------|----------------|--------------|
| center-seeded solid square, side W  | ~4W            | W ≤ ~2048    |
| corner-seeded solid square, side W  | ~2W            | W ≤ ~4096    |
| center-seeded disk, radius R        | ~5.7R          | R ≤ ~1400    |
| serpentine                          | O(width)       | always       |

Measured proof: the corner-seeded 4000×4000 full-red scene peaks at
**7999** occupied slots — exactly the predicted 2W−1, 193 slots under
capacity. The overflow test (center-seeded 2600² full-bleed) trips the wire
at ~level 1024 as predicted by 8r+4 > 8192.

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

## Results (RTX 4060 Laptop GPU, median of 5, tpb=256)

| scene            | filled | levels | GPU kernel | @njit CPU | GPU vs @njit | vs pure Python |
|------------------|--------|--------|-----------|-----------|--------------|----------------|
| sq_256 center    | 16K    | 129    | 0.9 ms    | 0.13 ms   | 0.15× (CPU wins) | 7× |
| sq_1024 center   | 262K   | 513    | 10.2 ms   | 4.3 ms    | 0.42× | 13× |
| sq_2000 center   | 1M     | 1001   | 32.8 ms   | 18.3 ms   | 0.56× | 16× |
| sq_4000 corner   | 16M    | 7999   | 302 ms    | 485 ms    | **1.61× (GPU wins)** | — |
| serpentine_256   | 33K    | 32896  | 74.5 ms   | 0.31 ms   | 0.004× (worst case) | 0.6× |
| disk_1024        | 724K   | 679    | 12.3 ms   | 9.5 ms    | 0.77× | 45× |

The honest story this stage exists to tell: one block (1/24 of the GPU)
crushes pure-Python BFS, only beats compiled CPU code once the image is
large enough (~16M px) to amortize per-level sync and launch overhead, and
is catastrophically wrong for serpentines — 32,896 levels of ~1-pixel
frontiers leave 255 of 256 threads idle (thread utilization < 1%) while
pure Python wins on the same scene. The tpb sweep on sq_2000 shows wide
frontiers reward more threads (64 → 11 Mpx/s, 1024 → 69 Mpx/s). These
ceilings — 1 SM, per-level `syncthreads`, frontier-starved parallelism —
are precisely what the multi-block (`../multi-blocks/`) and persistent
cooperative (`../persistent/`) stages remove.

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

## v2 roadmap (not implemented)

- **Spill-to-global overflow**: two-tier queue (shared ring + global backup)
  removes the 8192-slot cap — oversized frontiers become slower, not fatal.
- **Warp-aggregated enqueue**: port `persistent/`'s `_warp_enqueue`
  (ballot/shuffle, one atomic per warp) to the shared rear counter.
- **Wavefront visualization**: color by the recorded `depth` map / animate
  from `level_sizes`.
