# Parallel BFS Flood Fill on GPU — Design Analysis

> Superseded by [`chapters/README.md`](src/flood_fill_cuda/chapters/README.md)'s
> Chapters 1–4, which re-measure these same questions with this repo's later
> rigor (exact tests, bare twins, interleaved A/B timing). Kept here for the
> general GPU-BFS theory background and the bug catalog, which is still
> accurate for the graveyarded code it describes.

*Date: 2026-07-06. Hardware referenced throughout: RTX 4060 Laptop GPU (Ada, compute capability 8.9, 24 SMs, 8 GB, ~256 GB/s).*

This document analyzes how to parallelize a BFS-based flood fill on a GPU: what the "spawn threads as the frontier grows" intuition maps to, where the fundamental limits are, which algorithm wins for which blob shape and blob count, how each design maps to GPU hardware (threads / warps / blocks / SMs), and the concrete bottlenecks and bugs found in this repo's current implementation.

## TL;DR

The intuition of growing thread counts with the expanding frontier is correct and has a name: **level-synchronous frontier BFS** — and it is what [kernels_fixed.py](graveyard/multi-blocks/kernels_fixed.py) already implements. The "no lock, just check-and-pass" idea for duplicate neighbors is exactly the standard idiom (atomic compare-and-swap, first-writer-wins). The part that doesn't transfer to GPUs is "spawning threads as the frontier grows" — on a GPU you do the opposite: launch a sea of threads up front and let the frontier decide how many have work. The real ceiling isn't threads at all: **parallelism is bounded by frontier width, and total runtime is bounded by blob diameter × per-level sync cost**. This repo's own benchmarks show it — the 8000×8000 run took 2050 iterations at ~0.106 ms each, which is almost pure kernel-launch/sync overhead, not pixel work. And for *many* blobs, the right answer is to abandon BFS entirely for connected-component labeling, which fills everything in ~4 passes regardless of shape.

## The proposal, translated to GPU reality

**"Spawn 3 more threads when 4 pixels are added"** — GPUs can't do this cheaply. The literal mechanism exists (CUDA Dynamic Parallelism, where a kernel launches child kernels), but measured results are damning: Wang & Yalamanchili (IISWC 2014) found CDP versions of BFS-like workloads average a **1.21× slowdown** versus flat kernels, because device-side launch overhead exceeds even host-side launch overhead, and swarms of tiny child grids starve the SMs. It's also unsupported in Numba, so it's off the table for this repo anyway. The idiomatic version of the idea: keep the frontier in an array, and each BFS level, assign one thread per frontier pixel. When the frontier has 4 pixels, 4 threads work and thousands idle; when it has 50,000, the GPU is saturated. Threads are free to over-provision — occupancy is the launch default, not something you grow.

**"Thread 2 catches neighbor X too, queries if it was added, and passes"** — this is exactly right, with one refinement: do the test-and-claim *atomically and before enqueueing*, not after. The pattern is `atomicCAS(visited[nx,ny], 0, 1)` — hardware guarantees exactly one thread wins the race, the winner enqueues, losers pass. No locks (locks are an anti-pattern on GPUs; 32 threads in a warp execute in lockstep, so a spinlock inside a warp can deadlock). The repo's code already does this. It matters more than it looks: Merrill & Garland's scalable-BFS paper (PPoPP 2012) measured that on a 2D lattice — precisely the flood-fill case — skipping duplicate-culling caused **4.2× redundant expansion**, because adjacent frontier pixels keep discovering the same neighbors simultaneously.

## The fundamental limits nobody escapes

For a queue-based BFS, two numbers govern everything:

1. **Max parallelism = frontier width** ≈ blob perimeter at that level. A compact blob of area A has a frontier that peaks around O(√A). For a 2000×2000 image with a 1M-pixel blob, the frontier peaks in the low thousands — barely enough for this GPU (24 SMs × 1536 resident threads = ~36,864 threads at full occupancy on Ada).
2. **Sequential depth = geodesic diameter** of the blob (BFS hop count from seed to farthest pixel). Each level requires a global synchronization — there is no way to start level N+1 before level N finishes without breaking BFS ordering. The 550–2050 iterations in the benchmarks are this number, and each one costs a kernel launch + `cuda.synchronize()` (~5–20 µs) plus a 2×int32 host copy.

Multiply them out for the 8000×8000 run: 217 ms ÷ 2050 iterations ≈ 0.106 ms/iteration. At 24 blocks × 128 threads, the actual pixel work per level is microseconds — almost the entire runtime is per-level orchestration. That's also why 24 small blocks beat bigger grids in the benchmark sweep: the kernel is launch-bound, so extra threads just idle. For calibration, the measured peak of 73.6M px/s uses well under 1% of the 4060's ~256 GB/s memory bandwidth. The algorithm isn't compute- or bandwidth-limited; it's latency-limited.

## The design matrix: what to build for which workload

### Single large compact blob (the current benchmark case)

Frontier BFS is right, but fix the orchestration:

- **Persistent kernel with cooperative groups.** Launch once; inside the kernel, loop over levels with `cg.this_grid().sync()` as the level barrier (~1–3 µs vs ~10–20 µs for launch + host sync + the host round-trip). Numba supports this ([Numba cooperative groups](https://numba.readthedocs.io/en/stable/cuda/cooperative_groups.html)). Hard constraint: all blocks must be co-resident on the SMs simultaneously or the barrier deadlocks — size the grid with the occupancy API, never hardcode. This is the *correct* version of what [multi_blocks.py](graveyard/multi-blocks/multi_blocks.py) tried to do with `cuda.syncthreads()`, which only synchronizes within a block (that file is silently broken; `run_flood_fill_kernel` in kernels_fixed.py documents the same flaw).
- **Two queues (current/next) instead of one queue with front/rear.** Swap pointers each level. This eliminates the rear-inflation race the host snapshot works around, with no host copy at all.
- **Warp-aggregated enqueue.** Every discovered neighbor currently does its own `atomic.add` on one global counter — thousands of threads serializing on one memory location. Instead: `ballot_sync` to find which lanes have an item, lane 0 does **one** atomicAdd for the whole warp, broadcasts the base offset via `shfl_sync`, and lanes write at base+rank. NVIDIA measured ~21× on this pattern. In Numba you write it manually with `cuda.ballot_sync`/`cuda.shfl_sync`.

### Single thin / snake / spiral blob

The killer case for level-synchronous BFS: geodesic diameter is huge, frontier is ~2 pixels wide, so you get thousands of levels each doing trivial work while 36K threads idle. Two escapes:

- **Tile-based BFS** (Stava & Benes, GPU Computing Gems 2011): each block owns a 32×32 tile cached in shared memory and iterates the fill *locally* to convergence using cheap `syncthreads()` (hundreds of local iterations cost less than one grid sync). Only border crossings enter a global frontier *of tiles*. Depth is now bounded by (blob diameter ÷ tile size), a ~32× cut in expensive global syncs, and scattered global reads become coalesced tile loads. This is largely shape-insensitive — the strongest pure-BFS design.
- **Hybrid CPU/GPU switching** (Hong et al., PACT 2011): below a frontier-size threshold (~a few thousand items), a single CPU core beats the GPU on those levels outright. Only worth it if frontiers stay tiny for many consecutive levels, since switching costs transfers.

### Many blobs / label the whole image

**Stop doing BFS.** This is the biggest strategic point. Per-seed BFS scales with blob count, but connected-component labeling (CCL) via GPU union-find labels **all blobs in ~4 fixed passes, with runtime independent of blob shape, diameter, or count**:

- Every pixel starts as its own label (its raster index); each pixel atomically unions with its neighbors (`atomicMin` retry loop on tree roots); a final pass flattens each pixel to its root. No queue, no frontier, no iteration count that depends on content. The "many threads, one queue" tension dissolves because there is no queue.
- State of the art is block-based union-find, **BUF/BKE** (Allegretti, Bolelli, Grana, IEEE TPDS 2020): under 8-connectivity all pixels in a 2×2 block share a label, so the union-find runs at 2×2-block granularity — 4× fewer nodes and atomics. On a 2014-era Quadro K2200 it labels megapixel medical images in ~1.2 ms; an RTX 4060 should do 8000×8000 in roughly 5–15 ms (bandwidth-bound: ~4 sweeps × 64M pixels) versus the current 217 ms for *one* blob. Reference CUDA code lives in [YACCLAB](https://github.com/prittt/YACCLAB). This is the honest benchmark ceiling for the whole project, and the right foundation for the planned multi-blob Stage 5 — [bfs_plus_scan.py](graveyard/bfs_plus_scan.py) (currently a non-runnable sketch) is the file this would replace.
- One caveat: CCL gives you *labels*, not BFS distances. If you want the expanding-wavefront structure (the spatial-gradient coloring suggests the spread visualization matters here), only BFS-style methods give per-level distance for free.

### Distance-style flooding without walls

Jump Flooding (Rong & Tan 2006) reaches the far corner of a 4096² image in ~12 log-step rounds instead of ~8000 wave steps — but it jumps *over* barriers, so it can't do connectivity-respecting fill. Right tool for Voronoi diagrams and distance fields; wrong tool for flood fill. Mentioned so nobody reaches for it by mistake.

## Mapping to the RTX 4060 Laptop (Ada, CC 8.9)

- **Threads/blocks:** Ada SMs hold max 1536 resident threads. A 1024-thread block caps you at one block per SM = 67% occupancy; prefer 128–512 threads/block (e.g. 6 × 256). The benchmark sweep's 24×128 winner reflects launch-bound behavior, not a real compute optimum — after moving to a persistent kernel, re-sweep; the optimum will shift toward more resident threads.
- **Warps:** the unit that actually executes (32 lanes, lockstep). Two things to design around: *divergence* — the 4/8-neighbor `if visited/if red` branches split the warp, mostly unavoidable but cheap; and *coalescing* — the queue reads are already coalesced (good), but frontier pixels scatter across the image, so image/visited accesses are random. Tiles fix this; a bitmask visited array (1 bit/pixel with `atomicOr`, 8× less traffic than bytes) helps too.
- **SMs (24):** for cooperative launch, grid ≤ 24 × (blocks-co-resident-per-SM by the occupancy API). For kernel-per-level, oversubscribe freely (hundreds of blocks) so the scheduler hides latency.
- **Shared memory (~100 KB/SM):** the tile-local fill's workspace; a 32×32 tile with 4-byte labels is 4 KB, so several tiles can be resident per SM.

## Watchouts (several found live in this repo)

1. **`cuda.syncthreads()` is not a grid barrier.** [multi_blocks.py](graveyard/multi-blocks/multi_blocks.py) and [multi_blocks_simple.py](graveyard/multi-blocks/multi_blocks_simple.py) rely on it across blocks — results are silently wrong whenever gridDim > 1. Only cooperative `grid.sync()` or a kernel-launch boundary synchronizes blocks.
2. **Silent queue overflow loses pixels.** In [kernels_fixed.py](graveyard/multi-blocks/kernels_fixed.py), a neighbor past `QUEUE_CAPACITY` is dropped *after* being marked visited — never enqueued, never filled, and nothing reports it. Add an overflow flag the host checks.
3. **The benchmark sweep contains broken runs.** Several configs in the CSV (24×256, 40×128, 48×128 at chunk 16/32) filled only 2,000–6,000 of 1,000,000 pixels — those rows are correctness failures, not slow configs, and the "optimal config" conclusion is contaminated until they're excluded or the bug (likely a work-distribution edge case at those thread counts) is found. Also, `debug_warp_usage` is sized for 2 warps/block everywhere, so warp-utilization numbers are wrong for ≥128-thread blocks.
4. **Debug instrumentation is inflating timings.** Per-pixel `atomic.add` on a debug counter plus host prints in the loop; strip these before believing any number.
5. **Duplicate discovery is the lattice tax.** Keep the atomic-CAS-before-enqueue pattern; without it, redundant expansion multiplies (~4×).
6. **Config drift:** [utils.py](graveyard/multi-blocks/utils.py) says 96×512, [main_optimized.py](graveyard/multi-blocks/main_optimized.py) ships 96×128, the two summary JSONs disagree on the optimum, and `check_work_remaining` (built to avoid host copies) is defined but never called.

## Recommended sequencing

1. Strip debug overhead, exclude the broken CSV rows, re-baseline honestly (there is also no CPU-vs-GPU timing anywhere in the repo yet — worth recording).
2. Two-queue frontier swap → removes the host snapshot round-trip.
3. Persistent cooperative kernel with `grid.sync()` → kills ~2050 launches; this is where the 8000×8000 time should collapse.
4. Warp-aggregated enqueue → removes the last atomic hotspot.
5. Tile-based variant → shape robustness (test on a spiral blob; it will be a bloodbath for the level-sync version and fine for tiles).
6. For Stage 5 multi-blob: implement BUF/BKE union-find CCL instead of per-blob BFS — it's not an optimization of the current approach, it's a different (and for that problem, strictly better) algorithm.

## Sources

- Merrill, Garland, Grimshaw — *Scalable GPU Graph Traversal*, PPoPP 2012. <https://mgarland.org/papers/2012/bfs/>
- NVIDIA — *CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics*. <https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/>
- NVIDIA — *Cooperative Groups*. <https://developer.nvidia.com/blog/cooperative-groups/>
- Numba — *Cooperative Groups (grid sync) support*. <https://numba.readthedocs.io/en/stable/cuda/cooperative_groups.html>
- Wang, Yalamanchili — *Characterization and Analysis of Dynamic Parallelism in Unstructured GPU Applications*, IISWC 2014.
- Hong, Oguntebi, Olukotun — *Efficient Parallel Graph Exploration on Multi-Core CPU and GPU*, PACT 2011.
- Stava, Benes — *Connected Component Labeling in CUDA*, GPU Computing Gems Emerald Edition, 2011. <https://www.cs.purdue.edu/homes/bbenes/papers/Stava2011CCL.pdf>
- Playne, Hawick — *A New Algorithm for Parallel Connected-Component Labelling on GPUs*, IEEE TPDS 29(6), 2018. <https://github.com/DanielPlayne/playne-equivalence-algorithm>
- Allegretti, Bolelli, Grana — *Optimized Block-Based Algorithms to Label Connected Components on GPUs*, IEEE TPDS 31(2), 2020. <https://www.federicobolelli.it/media/publications/pdfs/2019tpds.pdf>
- YACCLAB benchmark suite (reference CUDA CCL implementations). <https://github.com/prittt/YACCLAB>
- Rong, Tan — *Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform*, ACM I3D 2006. <https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf>
