# Chapter 3 in Triton: one blob, N cooperative programs

This folder is the Triton twin of
[`chapters/ch03_gpu_1blob_nblock`](../../../chapters/ch03_gpu_1blob_nblock/README.md).
It is the same persistent, level-synchronous BFS. One global int32 queue
holds the frontier. Every program grid-strides each level window, and two
grid barriers close each level.

| file | twin of | what it holds |
|---|---|---|
| `kernels.py` | `kernels.py` | the eight kernels, same names, one shared body |
| `flood_fill.py` | `flood_fill.py` | `flood_fill`, `max_blocks`, `MultiFloodFillResult` |
| `test_correctness.py` | `test_correctness.py` | all 133 Numba tests by name, plus cross-backend, enqueue-mode and twin-only tests |
| `compare.py` | `benchmarks/benchmark.py`, `benchmarks/benchmark_connectivity_and_barrier_work.py` | Numba vs Triton timings, plus the `enqueue` experiment |

Scenes, CPU oracles and the bandwidth model (`model_bytes`, `model_gb_s`)
are imported from the Numba chapter and `shared/`, never copied. The
Triton copy-peak probe is `triton_twins/runtime/bandwidth.py`.

## What is twinned

All eight Numba kernels have a compiled Triton counterpart with the same
name and the same host contract:

| Numba kernel | Triton kernel | constexpr flags of `_bfs` |
|---|---|---|
| `multi_block_global_kernel` | same name | CONN=4, INSTRUMENTED |
| `multi_block_global_bare_kernel` | same name | CONN=4 |
| `multi_block_global8_kernel` | same name | CONN=8, INSTRUMENTED |
| `multi_block_global8_bare_kernel` | same name | CONN=8 |
| `multi_block_global8r2_kernel` | same name | CONN=8, RADIUS2, INSTRUMENTED |
| `multi_block_global8r2_bare_kernel` | same name | CONN=8, RADIUS2 |
| `multi_block_global8wc_kernel` | same name | CONN=8, WARP_COOP, INSTRUMENTED |
| `multi_block_global8wc_bare_kernel` | same name | CONN=8, WARP_COOP |

Each wrapper calls one `@triton.jit` body with fixed constexpr flags. So
every variant compiles to its own binary, and the bare twins contain no
instrumentation code. Every wrapper also takes the `ENQ` constexpr
(`"lane"` by default, or `"program"`), which picks the enqueue; see
[Enqueue](#enqueue-per-lane-default-and-per-program-first-translation). The counters, `block_stats`, the level trace and the
owner map have the same slots, dtypes and meanings, and they are measured
in the kernel, as in Numba.

## Mapping

One Numba block of T threads is one Triton program of T lanes, with
`num_warps = T // 32`. Lane i plays thread i, and a grid of B blocks is B
programs.

| Numba construct | Triton construct | fidelity | note |
|---|---|---|---|
| cooperative launch + `grid.sync()` (2 per level) | `launch_cooperative_grid=True` + `runtime.device.grid_sync` (2 per level) | emulated | The barrier is a monotonic int64 counter, with a CTA barrier, a release arrival and an acquire spin. It sits at the same two points. An oversized grid is refused by the driver, as in Numba. |
| `max_cooperative_grid_blocks(tpb)` | `runtime.occupancy.max_coresident_programs(compiled)` | close | Same occupancy formula. The value differs because the register counts differ (see below). |
| `cuda.grid(1)`, `gridsize(1)`, `blockIdx.x` | `pid * BLOCK + tl.arange(0, BLOCK)`, `num_programs * BLOCK`, `program_id` | exact | Same stride, so each queue index lands on the same thread and block as in Numba. |
| `for i in range(front + tid, rear, stride)` | `for base in range(front + pid * BLOCK, rear, stride)` + lane mask `i < rear` | exact | Programs past a narrow level skip the body, like Numba's idle threads. |
| `cuda.const.array_like(DX...)`, `for d in range(n)` | constexpr tuples, `tl.static_range(n)` | exact | Same direction order. |
| wc: per-lane `dx[d]` with `d = lane & 7` | per-lane offsets built once with `tl.where` from the same table | close | Same values. A register select replaces a divergent constant-memory read. |
| `_is_red`: three byte loads, short-circuit `and` | three masked byte loads, each masked by the previous test | exact | Same loads per lane. Torn reads of a paint in progress read as not red in both. |
| `cuda.atomic.cas(visited, (x, y), 0, 1) == 0` | masked `tl.atomic_xchg(visited, 1, sem="relaxed")`, `old == 0` | close | The same exactly-once claim on a 0/1 flag. `tl.atomic_cas` has no mask in Triton 3.7, and a sink-redirected CAS would add dummy atomics. |
| `_warp_enqueue_global` (activemask, popc, leader `atomic.add`, shfl: one atomic per warp) | `_lane_enqueue_global` (`ENQ="lane"`, the default): masked per-lane `tl.atomic_add(rear, 1, sem="relaxed", scope="gpu")`, the item stored at the returned slot | exact (same SASS) | The address is warp-uniform, so ptxas warp-aggregates the atomic: `VOTEU.ANY`, `FLO` leader, `POPC` count, one leader `ATOMG.E.ADD`, `SR_LTMASK` + `POPC` rank, `SHFL.IDX` broadcast. That is Numba's pattern, with no CTA barrier. Only the leader differs (highest active lane, Numba's `ffs` takes the lowest). The tripwire is the same. |
| (first translation) | `_block_enqueue_global` (`ENQ="program"`): `tl.sum` + `tl.cumsum` over the program, one `atomic_add` per program | emulated | Kept for measurement only. Each enqueue site costs 7 `BAR.SYNC` and about 9 shared-memory accesses, and locks the 8 warps together direction by direction. Numba never had these barriers. Cost: x0.58-1.07 instead of x1.07-1.55 on conn4/conn8. |
| per-thread `my_processed` / `my_cas_attempts` / `my_interior` + exit atomics | per-lane int32 accumulators, `tl.sum` at exit, one atomic per program | exact | Same values. There are fewer exit atomics (one per program, not per thread). |
| `block_stats[bx, 0] += ...`, `block_stats[bx, 1] = get_smid()` | `atomic_add` / store at `bx * 2 + col`, `runtime.device.read_smid` | exact | `%smid` comes from inline PTX instead of the linked `smid.cu`. |
| `tid == 0` writes (trace, final counters) | stores guarded by `bx == 0` (scalar stores) | exact | |
| r2 guard `elif visited[nx, ny] == 0` | masked plain `tl.load(visited)` on in-bounds, not-red lanes | exact | It reads claims from earlier levels, published by the barrier. |
| r2 divergent `if interior:` ring-2 loop | ring 2 masked by `interior`, skipped when no lane of the program is interior | close | The skip is per program rather than per warp. It is a 0/1-trip loop, because Triton 3.7 miscompiles scans inside an `if` (the program enqueue's scan). Both enqueue modes use the same skip. |
| `new_rear = q_state[0]` between the barriers | plain `tl.load` between the two `grid_sync` calls | exact | |
| `cuda.device_array` / `copy_to_device` / `copy_to_host` | `cp.empty` / `.set` / `.get` | close | Same bracket points for alloc, h2d, kernel and d2h. CuPy's pool makes repeated `alloc_ms` cheaper than Numba's. |

## Deviations

- **threads_per_block must be a power of 2** (32, 64, 128, 256 or 512).
  Numba's other multiples of 32 raise `ValueError`, and the message names
  the power-of-2 rule. The Numba message comes first for values Numba
  also rejects. No Numba test uses a non-power-of-2 value.
- **Capacity differs, so `blocks=None` differs.** The register counts
  differ, so the co-resident maximum differs too. Measured at tpb=256 on
  the RTX 4060 Laptop (24 SMs, Triton 3.7.1), for both enqueue modes:

  | variant | Numba regs | Numba cap | lane regs | lane cap | program regs | program cap |
  |---|---|---|---|---|---|---|
  | conn4 | 104 | 48 | 38 | 144 | 38 | 144 |
  | conn4_bare | 74 | 72 | 28 | 144 | 40 | 144 |
  | conn8 | 104 | 48 | 40 | 144 | 48 | 120 |
  | conn8_bare | 74 | 72 | 34 | 144 | 40 | 144 |
  | r2 | 112 | 48 | 80 | 72 | 128 | 48 |
  | r2_bare | 79 | 72 | 80 | 72 | 118 | **48** |
  | wc | 96 | 48 | 36 | 144 | 30 | 144 |
  | wc_bare | 65 | 72 | 24 | 144 | 33 | 144 |

  With the default per-lane enqueue every twin hosts at least Numba's
  grid, most of them 1.5-3x more programs (r2_bare: 72 vs 72 at tpb=256,
  336 vs 288 at tpb=64). The program-wide scan of the first translation
  cost the radius-2 pair 38-48 registers, so its `r2_bare` hosted fewer
  programs than Numba (48 vs 72). Compare backends at an explicit, equal
  `blocks`, as the cross-backend tests and the compare script do.
- **One extra keyword, `enqueue`.** `flood_fill`, `max_blocks` and
  `kernel_info` take `enqueue="lane"` (the default) or `"program"`, and
  the result records it. Numba has no such switch: it only has the
  per-warp enqueue that `"lane"` compiles to. Any other value raises
  `ValueError`.
- **Error-message dashes.** The two messages that contain an em dash in
  Numba ("is not red ...", "structural tripwire fired ...") use "-" here.
  The tests match on the words, not the dash.
- **Grid barrier counter.** Each launch gets one extra int64 buffer. It is
  allocated with the other buffers and zeroed with the H2D copies. It
  counts 2 arrivals per program per level, so an int32 counter would wrap
  on long runs (a 2048x2048 serpentine at 576 programs). int64 cannot
  wrap in practice, like Numba's `grid.sync()`.
- **Non-contiguous inputs are accepted.** A transposed or Fortran-order
  view works, because CuPy's `.set()` copies it contiguous. Numba's
  `copy_to_device` refuses a transposed view with a ValueError. That is a
  transfer limitation, not a validation rule, so the twin does not copy it.

## Enqueue: per lane (default) and per program (first translation)

Numba's `_warp_enqueue_global` aggregates a warp's claims by hand. It
takes the active mask and counts it, the lowest lane does one
`atomic.add` on the rear, a shuffle broadcasts the base, and each lane
adds its rank.

The twin's default, `enqueue="lane"` (`ENQ="lane"` in the kernels), is
plainer. Each claiming lane does one relaxed `tl.atomic_add(rear, 1)` and
stores its pixel at the slot it got back. The address is the same for the
whole warp, so ptxas builds the warp aggregation itself.

The first translation, `enqueue="program"`, aggregated over the whole
program instead: `tl.sum` and `tl.cumsum` across 8 warps, then one atomic
per program. It is kept, behind the same switch, so its cost stays
measurable.

### What the SASS shows

`nvdisasm` (CUDA 12.9) on the cubins at tpb=256. One enqueue site of
`multi_block_global_kernel`, in instruction order:

| step | Numba `_warp_enqueue_global` | Triton `enqueue="lane"` |
|---|---|---|
| active mask | `VOTE.ANY R71` | `VOTEU.ANY UR6` (uniform register) |
| leader | `BREV` + `FLO.U32.SH` (lowest lane) | `FLO.U32` (highest lane) |
| count | `POPC R3, R71` | `POPC R19, UR6` |
| one atomic per warp | `@!P0 ATOMG.E.ADD.STRONG.GPU` | `@P2 ATOMG.E.ADD.STRONG.GPU` |
| rank | `S2R SR_LTMASK`, `POPC` | `S2R SR_LTMASK`, `LOP3`, `POPC` |
| broadcast the base | `SHFL.IDX` (in `__cuda_sm70_shflsync_idx_p`) | `SHFL.IDX` |
| CTA barrier | none | none |

Every 32-bit `ATOMG.E.ADD` of every lane kernel has this shape. There are
4 in conn4, 8 in conn8, 24 in r2 and 1 in wc, one per enqueue site.
`test_twin_lane_enqueue_is_warp_aggregated_in_sass` checks it on every
build.

The barrier count tells the two modes apart:

| kernel | enqueue sites | `BAR.SYNC` lane | `BAR.SYNC` program |
|---|---|---|---|
| conn4 / conn4_bare | 4 | 15 / 10 | 43 / 38 |
| conn8 / conn8_bare | 8 | 15 / 10 | 71 / 66 |
| r2 / r2_bare | 24 | 21 / 13 | 189 / 181 |
| wc / wc_bare | 1 | 15 / 10 | 22 / 17 |

In lane mode no barrier belongs to the enqueue. The two grid barriers
take 5 each, and the rest are the exit reductions of the instrumented
kernels and r2's ring-2 skip (`tl.max`). Program mode adds exactly 7
`BAR.SYNC`, 4 `STS` and 5 `LDS` per enqueue site. So every direction of
every tile made the 8 warps of a program wait for each other, with the
sum and the scan going through shared memory. Numba never had those
barriers.

### What it costs

The `enqueue` experiment of `compare.py` times both modes against the
same Numba kernel, through the same harness. Median of 4 rounds, `x` =
Numba ms / Triton ms (above 1: Triton faster), first translation ->
per lane, at 48 x 256 (the pinned grid):

| variant | sq_2000_center | disk_4001_r1900 | serpentine_256 |
|---|---|---|---|
| conn4 | x0.94 -> x1.24 | x1.00 -> x1.12 | x1.07 -> x1.55 |
| conn4_bare | x0.94 -> x1.21 | x0.99 -> x1.11 | x1.07 -> x1.54 |
| conn8 | x0.82 -> x1.13 | x0.90 -> x1.14 | x0.87 -> x1.40 |
| conn8_bare | x0.79 -> x1.12 | x0.89 -> x1.13 | x0.83 -> x1.39 |
| r2 | x0.70 -> x1.02 | x0.79 -> x1.08 | x0.83 -> x1.37 |
| r2_bare | x0.68 -> x0.99 | x0.78 -> x1.07 | x0.80 -> x1.36 |
| wc | x1.20 -> x1.44 | x1.04 -> x1.33 | x1.57 -> x1.87 |
| wc_bare | x1.23 -> x1.36 | x1.07 -> x1.27 | x1.56 -> x1.90 |

The first run's worst sweep cells, one 512-lane program on
disk_4001_r1900, move from x0.62 to x1.07 (conn4) and from x0.58 to
x1.16 (conn8). Outputs were identical in every row.

The gain is largest where the barriers sit on the critical path. On the
serpentine every level is one pixel, so one warp works and its program
pays 7 barriers per direction on every level. On a one-program grid all
of a level's work goes through one program, tile after tile, and each
tile pays them again.

Before and after on the five worst rows of the first full run, measured
interleaved (Numba, program, lane, rotating order, 6 rounds, median
kernel_ms):

| row (first-run speedup) | Numba | program | lane | x program | x lane |
|---|---|---|---|---|---|
| sweep disk_4001_r1900 conn4 1x512 (x0.571) | 121.85 | 192.01 | 113.53 | 0.63 | 1.07 |
| sweep disk_4001_r1900 conn8 1x512 (x0.578) | 135.13 | 234.83 | 121.92 | 0.58 | 1.11 |
| barrier_work sq_2000_center r2_bare 48x256 (x0.669) | 4.92 | 7.20 | 4.78 | 0.68 | 1.03 |
| barrier_work sq_2000_center r2 48x256 (x0.709) | 5.47 | 7.65 | 5.30 | 0.72 | 1.03 |
| suite sq_1024_center conn8_bare 48x256 (x0.729) | 2.89 | 4.08 | 2.89 | 0.71 | 1.00 |

The program column reproduces the first run's ratios, so the enqueue
accounts for the whole gap on these rows.

## Deterministic outputs (what the cross-backend tests assert)

On the same scene and the same explicit grid, the Numba and Triton results
are equal in these fields:

- `img`, `visited`, `depth`, `levels`, `filled`, `processed`, `interior`;
- `peak_level`, `peak_occupancy`, `level_sizes`;
- `processed_per_block` and the balance and utilization percentages;
- the owner census;
- `cas_attempts` and `model_bytes` for conn4 only. The 4-neighbour graph
  is bipartite, so there is exactly one probe per edge. A test checks the
  edge count on both backends.

With `blocks=1`, the full owner map is equal too. Never compared:
`cas_attempts` for conn8, r2 and wc (same-level 8-neighbours race with
painting), the spatial owner map, `sm_ids` and timings.

All of this holds in both enqueue modes, which the
`test_enqueue_modes_*` tests check against Numba, the CPU oracle and each
other (every variant, five scenes up to a 1M-pixel disk, two grids). The
modes change only the queue order, which is schedule-dependent in both
backends anyway.

## Running

Tests (take the GPU lock when other GPU work may be running):

```
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch03_gpu_1blob_nblock/test_correctness.py
```

Comparison (writes `results/triton_twins/ch03_gpu_1blob_nblock/compare_<UTC>.json`):

```
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch03_gpu_1blob_nblock.compare [--quick] [--repeats N]
```

`compare.py` builds four experiments from the Numba benchmarks' own lists,
plus the `enqueue` experiment:

- `suite`: conn4, conn8 and their bare twins at 256 lanes, pinned to
  min(Numba capacity, Triton capacity).
- `suite_blocks_none`: each backend at its own maximum, with both grids
  in `config["resolved_blocks"]`. The grids differ, so these rows are
  `comparable=false` and stay out of the summary averages.
- `sweep`: TPB_SWEEP x BLOCKS_SWEEP. Its "max" column is the common grid,
  min(both capacities). That is Numba's maximum but usually not Triton's
  (`is_common_max`, and `is_backend_max` per backend).
- `barrier_work`: the seven per-barrier-work configs at one pinned grid.
- `enqueue`: two rows per cell, `config.enqueue` `"lane"` (label
  `per_lane`, the default every other experiment runs) and `"program"`
  (label `first_translation`). All eight variants on sq_2000_center,
  disk_4001_r1900 and serpentine_256 at the pinned grid, plus the first
  run's worst sweep cells (1 x 512 on disk_4001_r1900, conn4 and conn8).
  The `first_translation` rows are `comparable=false`: they measure the
  first translation's cost and stay out of the unit's averages.

Its `meta["caps"]` lists the one reduction: the 64M-pixel scene is
dropped, because holding a Numba and a Triton result at that size would
pass the 2.5 GB host-RAM budget. Every other scene and sweep cell is the
Numba benchmarks' own. The default run (4 rounds) takes about 13 minutes
of GPU time (the `enqueue` experiment adds about 70 s) with a 1.84 GB
peak RSS. `--quick` runs tiny scenes and writes
nothing.

Reading the rows:

- `speedup_kernel` is the Numba-vs-Triton number. `kernel_ms` includes
  each runtime's Python launch path. Numba's is 13-52 us slower per
  launch on this machine, which matters only for the sub-2 ms scenes.
- `speedup_total` mostly compares host memory management: CuPy's caching
  pool against Numba's `cuMemAlloc` per array. It is not a kernel metric.
- Repeats are even (an odd `--repeats` is rounded up), so each backend
  runs first in half the rounds. The second run of a round is slower on
  the large scenes, so an odd count would favour one backend.
