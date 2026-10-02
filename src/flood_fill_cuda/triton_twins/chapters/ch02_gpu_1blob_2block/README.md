# Chapter 2 in Triton: the dual-block BFS twin

This folder rebuilds `chapters/ch02_gpu_1blob_2block` in Triton: the same
four kernels, the same host API, the same tests against the same CPU oracle,
and a Numba-vs-Triton comparison of the chapter's own benchmark.

| file | role |
|---|---|
| `kernels.py` | `global`, `split`, `dirsplit` (each with an `INSTRUMENTED` constexpr: `False` is the bare twin) and `pinned` |
| `flood_fill.py` | the host driver: same `flood_fill(...)` signature, defaults, validation and `DualFloodFillResult` fields as Numba |
| `test_correctness.py` | the Numba test file test for test (same 102 names), plus `test_cross_backend_*` |
| `compare.py` | the chapter benchmark's scenes, tpb sweep and placement experiment, timed on both backends |

Scenes and the CPU oracle are imported from the Numba chapter, never copied.

## What is twinned

| Numba kernel | Triton twin | launch |
|---|---|---|
| `dual_block_global_kernel` | `dual_block_global_kernel[INSTRUMENTED=True]` | 2 programs, cooperative |
| `dual_block_global_bare_kernel` | `dual_block_global_kernel[INSTRUMENTED=False]` | 2 programs, cooperative |
| `dual_block_split_kernel` | `dual_block_split_kernel[INSTRUMENTED=True]`, `maxnreg=120` | 2 programs, cooperative |
| `dual_block_split_bare_kernel` | `dual_block_split_kernel[INSTRUMENTED=False]`, `maxnreg=120` | 2 programs, cooperative |
| `dual_block_dirsplit_kernel` | `dual_block_dirsplit_kernel[INSTRUMENTED=True]` | 2 programs, cooperative |
| `dual_block_dirsplit_bare_kernel` | `dual_block_dirsplit_kernel[INSTRUMENTED=False]` | 2 programs, cooperative |
| `dual_block_pinned_kernel` (48 x 768, `max_registers=40`) | `dual_block_pinned_kernel` (48 x 512, `maxnreg=64`) | cooperative |

The bare specializations compile without any counter, trace, owner or
`%smid` code: the `INSTRUMENTED` branches are constexpr, so they are never
emitted.

## Mapping

One Numba block of T threads is one Triton program of T lanes
(`num_warps = T // 32`); lane i plays thread i, and the grid-stride and
block-stride loops keep Numba's strides, so the item-to-program map of
`global` is the same as Numba's item-to-block map.

| Numba construct | Triton construct | fidelity | note |
|---|---|---|---|
| `cuda.cg.this_grid()` (cooperative launch) | `launch_cooperative_grid=True` | exact | an oversized grid raises, as in Numba |
| `grid.sync()` (2 per level) | `runtime.device.grid_sync(bar, epoch * nprog)` | emulated | same placement; a monotonic counter on a host-zeroed int32 |
| `cuda.syncthreads()` | `cta_sync()` | exact | kept at the same places (split prologue and level end) |
| `_pair_barrier` (pinned) | `_pair_barrier`, same sense-reversing algorithm | close | `threadfence()` becomes acq_rel / release / acquire on the barrier atomics; the reset is an atomic exchange |
| `cuda.atomic.cas(visited, (x, y), 0, 1) == 0` | masked `tl.atomic_xchg(visited, 1, sem="relaxed") == 0` | close | same exactly-once claim on a 0/1 flag; Triton's CAS has no mask |
| `_warp_enqueue_global` (one atomic per warp) | `_enqueue_global` (`tl.cumsum` ranks, one atomic per program) | close | per call site, as in Numba; tickets keep their meaning, contiguity is per program |
| `_warp_enqueue_two_tier` (shared ring, then spill) | `_enqueue_two_tier` | close | same tickets, ring window and spill rule; the second ballot is the closed form `rank - room` |
| `cuda.shared.array(8192)` ring + shared rears | 8192-slot region of a `(2, 8192)` global scratch array per program; rears in registers | emulated | no user shared memory in Triton; the program is the only producer of its ring, so the shared atomics are register adds |
| backward append `arr[cap-1-idx]` | same | exact | dirsplit's double-ended buffer |
| OVERFLOW tripwire | same slot, same host `RuntimeError` | exact | |
| per-thread `my_processed`, `my_cas_attempts` | per-lane `[TPB]` int32 accumulators | exact | summed per program at exit; Numba's per-thread exit atomics become one atomic per program |
| grid-uniform level state (`front`, `rear`, peaks, active sums) | program scalars | exact | same values, same counter slots |
| `level_sizes[bx, level]` trace | same layout `(2, trace_cap)` | exact | |
| `get_smid()` (linked `smid.cu`) | `read_smid` (inline PTX `%smid`) | exact | |
| `max_registers=120` (split) / `40` (pinned) | `maxnreg=120` / `64` | close | see the pinned deviation below |
| `max_cooperative_grid_blocks(tpb)` | `max_coresident_programs(compiled)` | exact | same RuntimeError text when the grid does not fit |
| tpb: any multiple of 32 in [32, 512] | powers of 2 in [32, 512] | close | 96, 160, ... raise ValueError naming the power-of-2 rule |
| `_warmup` keyed on (kernel, bare) | keyed on (kernel, bare, tpb) | close | TPB is a constexpr; every int argument is `do_not_specialize`, so no compile lands in `kernel_ms` |

## Deviations

- **Pinned experiment, 512 instead of 768.** Numba pins two 768-thread
  blocks to one SM: 48 blocks at 40 registers fit exactly 2 per SM, so the
  pair holds 1,536 threads, the SM's full residency. Triton needs a power
  of 2, so the twin uses 512-lane programs capped at 64 registers. The
  compiled kernel uses 46 registers, `programs_per_sm` reports exactly 2,
  and the twin launches `2 x sm_count = 48` programs: every SM hosts exactly
  two, as in Numba. The workers then hold 1,024 threads (67% of the SM), not
  1,536 (100%). The experiment's question (same SM vs spread) is the same;
  the "full residency" premise is not reproducible. Validation and tests
  use the twin's `PINNED_TPB = 512` where Numba hardcodes 768.
- **split's ring is global memory.** The chapter's split kernel bets on
  shared-memory latency; the twin's ring is L1/L2-cached global scratch
  with the same capacity and overflow semantics (spill-free on 2600^2,
  both halves spill on 4600^2). So split vs global means something
  different under Triton: both queues are in global memory.
- **Aggregation granularity.** One atomic per program per call site instead
  of one per warp. Only schedule-dependent outputs change (queue order,
  global's per-pixel owner speckle, split's seam-race inbox/spill counts,
  dirsplit's per-program split).

## Determinism and what the cross-backend tests compare

The 4-connected grid is bipartite: a depth-L pixel sees its depth L-1
neighbors already blue (behind a barrier) and its depth L+1 neighbors still
red, whatever the interleaving. So these are identical across runs, kernels
and backends: `img`, `visited`, `depth`, `levels`, `filled`, `level_sizes`,
`peak_level`, `peak_occupancy`, `processed`, and `cas_attempts` (one attempt
per directed edge p -> q with depth(q) = depth(p) + 1). A `cas_attempts`
mismatch would mean a stale read slipped past a barrier.

Per kernel: split's owner map, per-program counts, per-program trace and
utilization are exact (ownership is spatial); global's per-program counts,
trace and utilization are exact (item-to-program is positional at the same
tpb), its owner census is exact but the per-pixel owner is not. dirsplit's
per-program numbers are race-dependent (only the totals are exact). The
pinned BFS outputs are compared between Numba 768 and Triton 512. `%smid`
values are never compared.

## Measured behaviour (smoke runs, RTX 4060 Laptop)

At small and narrow-frontier scenes the twin matches or beats Numba: on the
serpentines, `grid_sync` is a little cheaper than CG `grid.sync`. On
large frontiers it trails: 36M px take ~300 ms in Numba and ~430-450 ms in
the twin. A Triton program aggregates in lockstep: every aggregated append
is a CTA-wide reduction, a scan and one atomic whose round trip stalls all
of the program's warps, while Numba's warps overlap their own appends.
The twin is also leaner in registers (global: 36 vs 106 per thread). Run
`compare.py` for the full table.

## Running

Every command that touches the GPU must hold the repo's GPU lock (see the
project notes); bare commands:

```
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch02_gpu_1blob_2block/test_correctness.py
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch02_gpu_1blob_2block.compare --quick
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch02_gpu_1blob_2block.compare [--repeats N]
```

The full comparison writes
`results/triton_twins/ch02_gpu_1blob_2block/compare_<UTC>.json`. It does
not use the `dual_block_*.json` names, so the chapter dashboard never picks
it up by mistake. Its `same()` compares the deterministic outputs above;
big arrays are compared by BLAKE2 digest, so no two 36M-pixel results are
alive at once (peak host RSS ~1.45 GB). The benchmark's single-block "v2"
rows belong to chapter 1's comparison and its @njit rows have no GPU
backend, so neither is repeated here.
