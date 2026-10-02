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
| `_pair_barrier` (pinned) | `_pair_barrier`, same sense-reversing algorithm | close | relaxed snapshot as in Numba, acq_rel arrival, relaxed reset then releasing bump on one thread; `threadfence()` after the spin becomes acquire spin reads (a separate trailing acquire costs one more program-wide broadcast per barrier, measured up to 5% slower) |
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
- **Input layout.** Numba's kernels index `img[x, y, c]` through strides;
  the twin's kernels address a flat C-order buffer. So the driver
  allocates the device image in C order and copies the input with
  `np.ascontiguousarray` inside the H2D bracket (a no-op for the
  C-contiguous scenes). Fortran-ordered and strided inputs give the same
  result on both backends. A permuted-axis view, which Numba's
  `copy_to_device` rejects with a ValueError, is accepted by the twin.
- **Seed types.** NumPy integer seeds are converted with `operator.index`
  after validation: Triton's launcher cannot specialize NumPy scalars,
  while Numba types them as kernel arguments.

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

## Measured behaviour (subset runs, RTX 4060 Laptop)

Kernel-time ratios, Numba / Triton (above 1: the twin is faster), from
interleaved runs with equal outputs. Run `compare.py` for the full table.

- **Narrow frontiers** (`serpentine_256`, `seam_serpentine_256`: ~33k
  one-pixel levels, so per-level overhead dominates). `global` is 3-5%
  faster in the twin and `dirsplit` 16-18% faster. `split` is 15-19%
  slower. A likely cause (read from the code, not profiled): each split
  chunk runs 8 program-wide aggregations (4 directions x ring and inbox)
  against 4 in `global`, and `_enqueue_two_tier` runs its reductions even
  when no lane claimed, because scans must stay outside conditionals.
- **Small and medium blobs.** `sq_256_center` is within about 10% either
  way (`dirsplit` ahead, `split` behind). On `sq_2000_center` every kernel
  trails by 7-20%.
- **Large frontiers.** The twin trails: on 36M px, `global` takes ~280 ms
  in Numba and ~420 ms in the twin, `split` ~305 vs ~430 ms. A Triton
  program aggregates in lockstep: every aggregated append is a CTA-wide
  reduction, a scan and one atomic whose round trip stalls all of the
  program's warps, while Numba's warps overlap their own appends.
- **pinned** trails by 25-45%. The matched row (`pinned 2x512 spread`,
  both backends at 512) is the backend comparison: `sq_2000_center`
  9.5 vs 12.9 ms, `serpentine_256` 108 vs 169 ms. In the twin every
  scalar atomic of the pair barrier is broadcast to the whole 16-warp
  program through shared memory, so each barrier step costs more than
  Numba's thread-0 spin.
- **Placement rows are not all like-for-like.** The chapter's own rows
  pit 2 x 768 Numba threads against 2 x 512 Triton lanes, so their ratio
  mixes a config change with the backend change (`comparable=false` in
  the JSON). On `sq_6000_center` spread: Numba 2x768 168 ms, Numba 2x512
  183 ms, twin 2x512 296 ms, so about 9% of that gap is the thread count.

The twin is leaner in registers (global: 36 vs 106 per thread).

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
alive at once (peak host RSS ~1.45 GB).

It mirrors every GPU row of the benchmark:
- the scenes, each with the single-block v2 baseline (chapter 1's `spill`
  kernel on both backends, cross-imported as the benchmark does);
- the tpb sweep;
- the placement experiment: v2 1x1024; v2 1x512 on both backends in
  place of 1x768 (no 768-lane program in Triton); pinned same SM and
  spread as the chapter runs them (Numba 768, twin 512,
  `comparable=false`); and a matched pinned spread row at 2 x 512.
  Same SM has no matched row: Numba's pinned kernel at 512 threads fits 3
  blocks per SM and has no `rank < 2` guard, so it is not a valid same-SM
  experiment.

The @njit rows have no GPU backend and are not repeated. Read
`speedup_kernel`: `total_ms` also compares the host stacks (CuPy's pooled
allocation vs Numba's `cuMemAlloc` on every run), which dominates on small
scenes.
