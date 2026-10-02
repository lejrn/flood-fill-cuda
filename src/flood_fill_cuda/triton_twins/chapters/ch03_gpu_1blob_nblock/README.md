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
| `test_correctness.py` | `test_correctness.py` | all 133 Numba tests by name, plus cross-backend and twin-only tests |
| `compare.py` | `benchmarks/benchmark.py`, `benchmarks/benchmark_connectivity_and_barrier_work.py` | Numba vs Triton timings |

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
instrumentation code. The counters, `block_stats`, the level trace and the
owner map have the same slots, dtypes and meanings, and they are measured
in the kernel, as in Numba.

## Mapping

One Numba block of T threads is one Triton program of T lanes, with
`num_warps = T // 32`. Lane i plays thread i, and a grid of B blocks is B
programs.

| Numba construct | Triton construct | fidelity | note |
|---|---|---|---|
| cooperative launch + `grid.sync()` (2 per level) | `launch_cooperative_grid=True` + `runtime.device.grid_sync` (2 per level) | emulated | The barrier is a monotonic counter, with a CTA barrier, a release arrival and an acquire spin. It sits at the same two points. An oversized grid is refused by the driver, as in Numba. |
| `max_cooperative_grid_blocks(tpb)` | `runtime.occupancy.max_coresident_programs(compiled)` | close | Same occupancy formula. The value differs because the register counts differ (see below). |
| `cuda.grid(1)`, `gridsize(1)`, `blockIdx.x` | `pid * BLOCK + tl.arange(0, BLOCK)`, `num_programs * BLOCK`, `program_id` | exact | Same stride, so each queue index lands on the same thread and block as in Numba. |
| `for i in range(front + tid, rear, stride)` | `for base in range(front + pid * BLOCK, rear, stride)` + lane mask `i < rear` | exact | Programs past a narrow level skip the body, like Numba's idle threads. |
| `cuda.const.array_like(DX...)`, `for d in range(n)` | constexpr tuples, `tl.static_range(n)` | exact | Same direction order. |
| wc: per-lane `dx[d]` with `d = lane & 7` | per-lane offsets built once with `tl.where` from the same table | close | Same values. A register select replaces a divergent constant-memory read. |
| `_is_red`: three byte loads, short-circuit `and` | three masked byte loads, each masked by the previous test | exact | Same loads per lane. Torn reads of a paint in progress read as not red in both. |
| `cuda.atomic.cas(visited, (x, y), 0, 1) == 0` | masked `tl.atomic_xchg(visited, 1, sem="relaxed")`, `old == 0` | close | The same exactly-once claim on a 0/1 flag. `tl.atomic_cas` has no mask in Triton 3.7, and a sink-redirected CAS would add dummy atomics. |
| `_warp_enqueue_global` (activemask, popc, shfl, one `atomic.add` per warp) | `_block_enqueue_global` (`tl.sum` + `tl.cumsum`, one `atomic_add` per program) | emulated | Triton has no ballot, popc or shfl. The rank is the claimant's position among the program's claimants. The tripwire is the same. |
| per-thread `my_processed` / `my_cas_attempts` / `my_interior` + exit atomics | per-lane int32 accumulators, `tl.sum` at exit, one atomic per program | exact | Same values. There are fewer exit atomics (one per program, not per thread). |
| `block_stats[bx, 0] += ...`, `block_stats[bx, 1] = get_smid()` | `atomic_add` / store at `bx * 2 + col`, `runtime.device.read_smid` | exact | `%smid` comes from inline PTX instead of the linked `smid.cu`. |
| `tid == 0` writes (trace, final counters) | stores guarded by `bx == 0` (scalar stores) | exact | |
| r2 guard `elif visited[nx, ny] == 0` | masked plain `tl.load(visited)` on in-bounds, not-red lanes | exact | It reads claims from earlier levels, published by the barrier. |
| r2 divergent `if interior:` ring-2 loop | ring 2 masked by `interior`, skipped when no lane of the program is interior | close | The skip is per program rather than per warp. It is a 0/1-trip loop, because Triton 3.7 miscompiles scans inside an `if`. |
| `new_rear = q_state[0]` between the barriers | plain `tl.load` between the two `grid_sync` calls | exact | |
| `cuda.device_array` / `copy_to_device` / `copy_to_host` | `cp.empty` / `.set` / `.get` | close | Same bracket points for alloc, h2d, kernel and d2h. CuPy's pool makes repeated `alloc_ms` cheaper than Numba's. |

## Deviations

- **threads_per_block must be a power of 2** (32, 64, 128, 256 or 512).
  Numba's other multiples of 32 raise `ValueError`, and the message names
  the power-of-2 rule. The Numba message comes first for values Numba
  also rejects. No Numba test uses a non-power-of-2 value.
- **Capacity differs, so `blocks=None` differs.** Triton allocates far
  fewer registers: about 39 for conn4 versus Numba's ~104. So at tpb=256,
  `blocks=None` resolves to 144 programs, where Numba gets 48. The
  radius-2 kernel uses about 123 registers and gets 48. Compare backends
  at an explicit, equal `blocks`, as the cross-backend tests and the
  compare script do.
- **Enqueue aggregation is per program, not per warp.** One atomic per
  program per direction replaces one per warp. That costs a cross-warp
  reduction and scan, done through shared memory with CTA barriers.
  Numba's warp intrinsics stay in registers. This changes queue order
  only, which is schedule-dependent in both backends anyway.
- **Error-message dashes.** The two messages that contain an em dash in
  Numba ("is not red ...", "structural tripwire fired ...") use "-" here.
  The tests match on the words, not the dash.
- **Grid barrier counter.** Each launch gets one extra int32 buffer. It is
  allocated with the other buffers and zeroed with the H2D copies.

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

## Running

Tests (take the GPU lock when other GPU work may be running):

```
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch03_gpu_1blob_nblock/test_correctness.py
```

Comparison (writes `results/triton_twins/ch03_gpu_1blob_nblock/compare_<UTC>.json`):

```
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch03_gpu_1blob_nblock.compare [--quick] [--repeats N]
```

`compare.py` builds four experiments from the Numba benchmarks' own lists:

- `suite`: conn4, conn8 and their bare twins at 256 lanes, pinned to
  min(Numba capacity, Triton capacity).
- `suite_blocks_none`: each backend at its own maximum, with both grids
  recorded.
- `sweep`: TPB_SWEEP x BLOCKS_SWEEP.
- `barrier_work`: the seven per-barrier-work configs at one pinned grid.

Its `meta["caps"]` lists the one reduction: the 64M-pixel scene is
dropped, because holding a Numba and a Triton result at that size would
pass the 2.5 GB host-RAM budget. Every other scene and sweep cell is the
Numba benchmarks' own. The default run takes about 13-14 minutes of GPU
time with a 1.84 GB peak RSS. `--quick` runs tiny scenes once and writes
nothing.
