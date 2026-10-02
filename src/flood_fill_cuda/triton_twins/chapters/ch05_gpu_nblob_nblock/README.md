# ch05 in Triton: N blobs, N blocks, no seeds

The Triton twin of [`chapters/ch05_gpu_nblob_nblock`](../../../chapters/ch05_gpu_nblob_nblock/README.md).
Same two discovery strategies (`seed_merge`, `ccl_fill`), same phases, same
atomicMin union-find, same barriers, same counters, same host API. Read the
Numba chapter for what the algorithm does and why it is correct; this page
only records how each CUDA construct was spelled in Triton, and what that
costs.

| File | Role |
|---|---|
| `kernels.py` | the `@triton.jit` twins; phases are shared helpers |
| `flood_fill.py` | host driver: `flood_fill`, `max_blocks`, `discovery_only`, plus `compiled_kernel` / `kernel_info` / `regs_per_thread` |
| `test_correctness.py` | the Numba suite's GPU tests, same names, plus `test_cross_backend_*` |

Reused unchanged from the Numba chapter (imported, never copied): scenes,
CPU oracles, `SeedDiscoveryResult`, `model_bytes_ch05`, `MODEL_NOTE`,
`VARIANTS`, `BUILDS`, the phase-key table, `PALETTE_HOST` and the counter
slot constants.

## What is twinned

| Numba kernel | Triton twin |
|---|---|
| `ccl_fill_kernel` | `ccl_fill_kernel[INSTR=True]` |
| `ccl_fill_bare_kernel` | `ccl_fill_kernel[INSTR=False]` |
| `seed_merge_kernel` | `seed_merge_kernel[INSTR=True]` |
| `seed_merge_bare_kernel` | `seed_merge_kernel[INSTR=False]` |
| `seed_scan_kernel` (benchmark phase) | `seed_scan_kernel` |
| `ccl_kernel` (benchmark phase) | `ccl_kernel` |

`INSTR=False` compiles the instrumentation out (counters, owner and
prov_label stores, phase stamps, level trace, block_stats, the closing
seed_merge barrier); those pointer arguments are passed as `None`.

Not twinned yet: the lattice family (`seed_merge_lat_kernel` and its bare,
`r128` and `split` builds, `lat_compress_kernel`, `lat_finish_kernel`).
The driver already validates `lattice` / `interior` / `build` with the
Numba messages, and the lattice tests skip until those kernels are
registered in `flood_fill._KERNELS`.

## Mapping

| Numba construct | Triton construct | Fidelity | Note |
|---|---|---|---|
| block of T threads, grid of B blocks | program with `BLOCK = T` lanes, `num_warps = T // 32`; B programs | exact | lane i plays thread i; T must be a power of 2 |
| cooperative launch, `grid.sync()` | `launch_cooperative_grid=True`, `runtime.device.grid_sync` on a monotonic int64 counter | emulated | same placement and count: 2 per level, the fence sandwich, seed_merge's instrumented closing barrier. An oversized grid raises like Numba |
| `max_cooperative_grid_blocks(tpb)` | `max_coresident_programs(compiled)` | close | per compiled twin and block size; its own registers, so its own number (table below) |
| grid-stride loops | `range(start, end, nprog * BLOCK)` + `pid * BLOCK + lanes` | exact | same element to (block, lane) map: `processed_per_block` matches Numba when blocks and tpb are pinned (tested) |
| `_warp_enqueue_global` (activemask, popc, shfl, one atomic per warp) | `_cta_enqueue`: `tl.cumsum` ranks, one atomic per program per direction | close | no warp intrinsics in Triton. Queue order inside a level differs; the level's set does not |
| `atomic.cas(visited, 0, 1) == 0` | masked `tl.atomic_xchg(visited, 1, sem="relaxed") == 0` | close | visited holds only 0/1, so the winner is the same exactly-once claimant; Triton's `atomic_cas` has no mask |
| `atomic.min(parent, a, b)` | masked `tl.atomic_min(..., sem="relaxed")` | exact | |
| `_find` per-thread while loop | lockstep loop over a lane mask, exit checked every 8 hops | emulated | finished lanes run masked no-ops; the slowest of the program's lanes sets the pace |
| `_union` while-True with early returns (a real device call) | lane-mask loop (`go`, `retired`), inlined | emulated | same retry-from-returned-value protocol, same termination argument |
| per-thread `my_*` registers, exit `atomic.add` per thread | per-lane register vectors, one reduction + one atomic per program at exit | exact | totals identical (tested against Numba) |
| `%clock64` around each in-flight `_union`, summed per thread | `%clock64` around the program's lockstep union, added on each colliding lane | emulated | per-thread meaning kept: a colliding lane is busy for the whole lockstep union |
| `smid.cu` (`%smid`, `%globaltimer`) | inline PTX (`read_smid`, `read_globaltimer`) | exact | |
| `cuda.const.array_like` DX8/DY8/PDX/PDY | `tl.constexpr` tuples under `tl.static_range` | exact | same probe order E SE S SW W NW N NE |
| `cuda.const.array_like(PALETTE_HOST)` | 18-byte device array, cached | close | |
| `tid == 0` / `threadIdx.x == 0` stores | `if pid == 0:` scalar stores / per-program stores | exact | |
| device arrays from `cuda.to_device(host)` / `device_array` | `cp.asarray(host)` / `cp.empty` | exact | same host arrays uploaded in the same alloc bracket |
| one `[1, 32]` warm-up launch per kernel | one 1-program launch per (kernel, bare, tpb) | close | `num_warps` is compile-time; every runtime int is `do_not_specialize`, so no scene size recompiles (tested) |
| `regs_per_thread(dispatcher)` | `regs_per_thread(compiled)`, `kernel_info(...)` | close | a Triton kernel has one compiled object per block size |

## Deviations

- `threads_per_block` must be a power of 2 (32 ... 512). Numba accepts any
  multiple of 32; the twin raises `ValueError` naming the power-of-2 rule.
- Two Triton-only buffers per launch: the grid barrier counter (int64,
  zeroed in the alloc bracket) and the palette.
- `blocks=None` resolves to the twin's own capacity. At tpb 256 on the
  RTX 4060 Laptop (24 SMs):

  | kernel | Numba regs / blocks | Triton regs / programs |
  |---|---|---|
  | seed_merge | 114 / 48 | 116 / 48 |
  | seed_merge bare | 64 / 96 | 76 / 72 |
  | ccl_fill | 113 / 48 | 80 / 72 |
  | ccl_fill bare | 66 / 72 | 46 / 120 |

- Schedule-dependent outputs differ between backends as they differ
  between Numba runs: queue order, `owner`, `prov_label` at equidistant
  seams, `seed_merge` `union_attempts`, `ccl_fill` `cas_attempts`, smid,
  timings. Everything else is asserted equal to Numba in the tests.

## The cost of the lockstep

A Triton program has no per-warp control flow, so a loop whose trip count
differs per lane (`_find`, `_union`) runs until the slowest lane of the
whole program is done, with a program-wide reduction for every exit test.
On solid blobs that dominates `ccl_fill`'s union-find phases. Median phase
times, `ccl_fill`, 1500 x 800 two-squares scene, instrumented, capacity
grid:

| tpb | Numba union_merge | Triton union_merge |
|---|---|---|
| 32 | 4.65 ms | 5.51 ms |
| 64 | 3.79 ms | 6.50 ms |
| 128 | 4.55 ms | 36.1 ms |
| 256 | 3.85 ms | 34.5 ms |

At one warp per program the twin is within 1.2x; at 256 lanes it pays
the lockstep. The fill phases take 1.0-1.7x Numba's time on the same
mid-size scenes: every direction step's enqueue needs program-wide
reductions where Numba uses warp votes. Whole seed_merge runs, which union
only where waves collide, take 1.0-1.6x Numba's time. Checking the find
exit every 8 hops instead of every hop halved the union-find phases (blob
grid union_merge 9.9 ms to 4.3 ms) without changing any lane's chase.
These are development measurements; the comparison script is the
reference.

## Run

Tests (needs the GPU; the Numba suite runs separately):

    .venv/bin/python -m pytest -p no:cacheprovider \
        src/flood_fill_cuda/triton_twins/chapters/ch05_gpu_nblob_nblock

`test_cross_backend_*` runs the Numba driver and the twin on the same
images and asserts exact equality of img, visited, depth, label, n_blobs,
seeds, filled, levels, the level trace and the deterministic counters
(candidates, union_done, processed, peak_level, peak_occupancy, seed_merge
cas_attempts, ccl_fill union_attempts), and with pinned blocks and tpb
also `processed_per_block` and `thread_util_pct`.
