# Chapter 4 in Triton: two blobs, N cooperative programs

This folder is the Triton twin of
[`chapters/ch04_gpu_2blob_nblock`](../../../chapters/ch04_gpu_2blob_nblock/README.md).
It is ch03's persistent, level-synchronous BFS on a global queue, with two
changes. The seed count is a launch argument, and each int32 queue entry
carries its blob's label. The kernel paints blob 0 blue and blob 1 green.

| file | twin of | what it holds |
|---|---|---|
| `kernels.py` | `kernels.py` | the ten kernels, same names, one shared body |
| `flood_fill.py` | `flood_fill.py` | `flood_fill`, `max_blocks`, `DualBlobResult`, `LaunchStats` |
| `test_correctness.py` | `test_correctness.py` | all 86 GPU tests by name, plus twin-only and cross-backend tests |
| `compare.py` | `benchmarks/benchmark.py`, `benchmarks/benchmark_radius2_barrier_work.py` | Numba vs Triton timings |

Scenes, the CPU oracles and the bandwidth model (`model_bytes`,
`model_gb_s`) are imported from the Numba chapter and `shared/`, never
copied. The packing-tax rows run ch03's kernel, through ch03's Numba
driver and its Triton twin.

## What is twinned

All ten Numba kernels have a compiled Triton counterpart with the same name
and the same arguments, plus the barrier counter and the sizes Numba reads
from array shapes:

| Numba kernel | entry format | constexpr flags of `_dual_blob` |
|---|---|---|
| `dual_blob_lin_kernel` | lin | CONN=4, INSTR |
| `dual_blob_lin_bare_kernel` | lin | CONN=4 |
| `dual_blob_lin8_kernel` | lin | CONN=8, INSTR |
| `dual_blob_lin8_bare_kernel` | lin | CONN=8 |
| `dual_blob_lin8r2_kernel` | lin | CONN=8, RADIUS=2, INSTR |
| `dual_blob_lin8r2_bare_kernel` | lin | CONN=8, RADIUS=2 |
| `dual_blob_xy_kernel` | xy | CONN=4, INSTR |
| `dual_blob_xy_bare_kernel` | xy | CONN=4 |
| `dual_blob_xy8_kernel` | xy | CONN=8, INSTR |
| `dual_blob_xy8_bare_kernel` | xy | CONN=8 |

Each wrapper calls one `@triton.jit` body with fixed constexpr flags, so
each variant compiles to its own binary. A test checks that the bare
binaries contain no `%smid` read, no int16 owner store and no int64
counter atomics. The counters, `block_stats`, the level trace and the
owner map keep Numba's slots, dtypes and meanings, and they are measured
in the kernel.

The three modes are twinned too: `sequential` (two launches),
`multisource` (both seeds in one queue, one launch) and `streams` (two
cooperative launches on two CuPy streams, each with its own barrier
counter).

## Mapping

One Numba block of T threads is one Triton program of T lanes, with
`num_warps = T // 32`. Lane i plays thread i, and a grid of B blocks is B
programs.

| Numba construct | Triton construct | fidelity | note |
|---|---|---|---|
| cooperative launch + `grid.sync()` (2 per level) | `launch_cooperative_grid=True` + `runtime.device.grid_sync` (2 per level) | emulated | A monotonic int64 counter per launch: CTA barrier, release arrival, acquire spin. It sits at the same two points. The driver refuses an oversized grid, as in Numba. |
| `max_cooperative_grid_blocks(tpb)` | `runtime.occupancy.max_coresident_programs(compiled)` | close | Same occupancy formula, per (kernel, tpb). The values differ because the register counts differ (see below). |
| `cuda.grid(1)`, `gridsize(1)`, `blockIdx.x` | `pid * BLOCK + tl.arange(0, BLOCK)`, `num_programs * BLOCK`, `program_id` | exact | Same stride, so each queue index lands on the same thread and block as in Numba. |
| `for i in range(front + tid, rear, stride)` | `for base in range(front + pid * BLOCK, rear, stride)` + lane mask `i < rear` | exact | |
| `rear = n_seeds` (kernel argument) | same, with `do_not_specialize` | exact | Never read from `q_state` at kernel start (the Numba chapter's Finding 1). |
| lin decode `entry & 1`, `pixel // height`, `pixel % height` (int64) | the same, on int64 | exact | 64-bit div/mod on purpose: a 32-bit divide would make Triton's lin look cheaper than Numba's and bias the lin-vs-xy comparison. |
| xy decode/pack (shifts and masks) | the same expressions | exact | |
| `cuda.const.array_like(DX...)`, `for d in range(n)` | constexpr tuples, `tl.static_range(n)` | exact | Same direction order. |
| `palette[lbl, c]` (const 2x3 table) | `tl.where(lbl == 0, PAL0[c], PAL1[c])` on constexpr tuples | exact | Same colors for the chapter's two labels. Numba's table has no rows past label 1 either. |
| `_is_red`: three byte loads, short-circuit `and` | three masked byte loads, each masked by the previous test | exact | |
| `cuda.atomic.cas(visited, (x, y), 0, 1) == 0` | masked `tl.atomic_xchg(visited, 1, sem="relaxed")`, `old == 0` | close | The same exactly-once claim on a 0/1 flag. `tl.atomic_cas` has no mask in Triton 3.7. |
| `_warp_enqueue_global` (activemask, popc, shfl, one `atomic.add` per warp) | `_block_enqueue_global` (`tl.sum` + `tl.cumsum`, one relaxed `atomic_add` per program) | emulated | Triton has no ballot, popc or shfl. The tripwire is the same. |
| per-thread int64 `my_processed` / `my_cas_attempts` / `my_interior` + exit atomics | per-lane int64 tensors + the same per-lane exit atomics | exact | |
| `block_stats[bx, 0] += ...`, `block_stats[bx, 1] = get_smid()` | per-lane `atomic_add` / store at `bx * 2 + col`, `runtime.device.read_smid` | exact | `%smid` comes from inline PTX instead of the linked `smid.cu`. |
| `tid == 0` writes (trace, final counters) | stores guarded by `pid == 0` | exact | |
| r2 guard `elif visited[nx, ny] == 0` | masked plain `tl.load(visited)` on in-bounds, not-red lanes | exact | It reads claims from earlier levels, published by the barrier. |
| r2 divergent `if interior:` ring-2 loop | ring 2 masked by `interior`, skipped when no lane of the program is interior | close | The skip is per program, not per warp. It is a 0/1-trip loop, because Triton 3.7 miscompiles scans inside an `if`. |
| `new_rear = q_state[0]` between the barriers | plain `tl.load` between the two `grid_sync` calls | exact | |
| streams: `cuda.stream()`, `kernel[b, t, stream]`, `cuda.event` | `cp.cuda.Stream()`, launch inside `with stream:`, `cp.cuda.Event` | close | Same issue order (both launches before either sync) and the same overlap ratio. |
| `cuda.device_array` / `copy_to_device` / `copy_to_host` | `cp.empty` / `.set` / `.get` | close | Same bracket points for alloc, h2d, kernel and d2h. CuPy's pool makes repeated `alloc_ms` cheaper than Numba's. |

## Deviations

- **threads_per_block must be a power of 2** (32, 64, 128, 256 or 512).
  Numba's other multiples of 32 raise `ValueError`, and the message names
  the power-of-2 rule. Values Numba also rejects get Numba's message. No
  Numba test uses a non-power-of-2 value.
- **Capacity differs, so `blocks=None` differs.** The register counts
  differ, so the co-resident maximum differs too. Measured at tpb=256 on
  the RTX 4060 Laptop (24 SMs):

  | kernel | Numba regs | Numba cap | Triton regs | Triton cap |
  |---|---|---|---|---|
  | lin | 104 | 48 | 40 | 144 |
  | lin_bare | 79 | 72 | 40 | 144 |
  | lin8 | 104 | 48 | 64 | 96 |
  | lin8_bare | 79 | 72 | 48 | 120 |
  | lin8r2 | 106 | 48 | 106 | 48 |
  | lin8r2_bare | 80 | 72 | 80 | 72 |
  | xy | 104 | 48 | 40 | 144 |
  | xy_bare | 72 | 72 | 40 | 144 |
  | xy8 | 105 | 48 | 61 | 96 |
  | xy8_bare | 74 | 72 | 48 | 120 |

  At tpb 64 and 256 every Triton capacity is at least Numba's (measured),
  so a pinned grid of min(both) is Numba's own default grid. Compare backends at an explicit,
  equal `blocks`, as the cross-backend tests and the compare script do.
- **Enqueue aggregation is per program, not per warp.** One atomic per
  program per direction replaces one per warp. That changes queue order
  only, which is schedule-dependent in both backends anyway.
- **Streams pair rail.** In `streams` mode the twin refuses an explicit
  `blocks` whose pair (2 x blocks) exceeds the cooperative capacity, with
  a `RuntimeError` naming the cooperative-launch capacity. Numba checks
  each launch on its own and can wedge. The `blocks=None` default is
  Numba's: a third of the capacity per launch.
- **Error-message dashes.** The two messages that contain an em dash in
  Numba ("is not red ...", "structural tripwire fired ...") use "-" here.
  The tests match on the words, not the dash.
- **Grid barrier counter.** Each launch gets one extra int64 buffer. It is
  allocated with the other buffers and zeroed with the H2D copies.
- **Warm-up per lane count.** Numba compiles one binary per kernel. A
  Triton program's lane count is a compile-time constant, so the twin
  warms the exact (kernel, tpb) binary each call uses. A test checks that
  no other image size or seed count recompiles it.

## Deterministic outputs (what the cross-backend tests assert)

On the same scene, mode and explicit grid, the Numba and Triton results
are equal in these fields:

- `img`, `visited`, `depth`, `label`;
- `filled`, `filled_a`, `filled_b`, `levels`, `levels_a`, `levels_b`,
  `processed`, `interior`;
- per launch: `filled`, `levels`, `peak_level`, `peak_occupancy`,
  `level_sizes`, `processed_per_block`, `thread_util_pct`;
- the owner census (per-program counts of the owner map);
- `cas_attempts` and `model_bytes` at 4-conn radius 1 only. Each
  component is bipartite, so every edge is probed once, from its nearer
  end. A test checks this count against the oracle's depth map on both
  backends.

With `blocks=1`, the full owner map is equal too. Never compared:
`cas_attempts` at 8-conn and radius 2 (same-level diagonal neighbours race
with painting), the spatial owner map, `sm_ids`, timings and the
streams overlap ratio.

## Running

Tests (take the GPU lock when other GPU work may be running):

```
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch04_gpu_2blob_nblock/test_correctness.py
```

Streams-mode tests are opt-in, as in Numba (`DUAL_BLOB_STREAMS=1`). The
cross-backend streams test compares Triton's streams mode with Numba's
sequential mode, because Numba's own concurrent pair can wedge.

Comparison (writes `results/triton_twins/ch04_gpu_2blob_nblock/compare_<UTC>.json`):

```
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch04_gpu_2blob_nblock.compare [--quick] [--repeats N]
```

`compare.py` runs six experiments on the Numba benchmarks' four scenes,
all at 256 lanes:

- `modes`: benchmark.py's eight configs (seq, seq_half, multi, multi_xy,
  bare, bare_xy, seq8, multi8). Each runs on its own kernel's grid, as in
  the Numba benchmark, pinned to min(Numba capacity, Triton capacity).
- `seq_a`, `seq_b`: the seq config, reporting one launch's kernel time
  (for ideal_max and the packing tax).
- `packing_tax`: ch03's single-blob kernel on blob A, both backends.
- `blocks_none`: multi and multi8 at each backend's own maximum. The
  grids differ, so these rows are `comparable=false`.
- `radius2`: benchmark_radius2_barrier_work.py's four configs at one
  pinned grid.

There are no caps. The default run (warm-up and 6 rounds) takes about 10
minutes of GPU time with a 1.5 GB peak RSS. `--quick` runs tiny scenes
and writes nothing. `speedup_kernel` is the Numba-vs-Triton number.
`speedup_total` mostly compares CuPy's caching pool with Numba's
per-array `cuMemAlloc`. `mode="streams"` and the @njit oracle timings are
not run, as in the Numba benchmark and by scope.
