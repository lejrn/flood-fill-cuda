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
| `test_correctness.py` | `test_correctness.py` | all 86 GPU tests by name, plus twin-only, cross-backend and enqueue-switch tests |
| `compare.py` | `benchmarks/benchmark.py`, `benchmarks/benchmark_radius2_barrier_work.py` | Numba vs Triton timings |
| `sass.py` | (none) | reads the enqueue's machine code back: PTX barrier count, SASS pattern per enqueue atomic |

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
each variant compiles to its own binary. Every wrapper also takes one
twin-only constexpr, `ENQ`, with a default of `"lane"` (see "The enqueue
switch" below). A test checks that the bare
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
| `_warp_enqueue_global` (activemask, popc, leader `atomic.add`, shfl: one atomic per warp) | `_lane_enqueue_global`: one relaxed `tl.atomic_add(rear, 1)` per winning lane, the item stored at the returned slot (`ENQ="lane"`, the default) | exact (same SASS) | ptxas warp-aggregates the uniform-address atomic: `VOTEU.ANY`, `FLO` + `POPC`, one leader `ATOMG.E.ADD`, `SHFL.IDX`. That is the machine code of Numba's hand-written helper, with no CTA barrier. The tripwire is the same. The first translation's program-wide enqueue stays selectable as `ENQ="program"` (see below). |
| per-thread int64 `my_processed` / `my_cas_attempts` / `my_interior` + exit atomics | per-lane int64 tensors + the same per-lane exit atomics | exact | |
| `block_stats[bx, 0] += ...`, `block_stats[bx, 1] = get_smid()` | per-lane `atomic_add` / store at `bx * 2 + col`, `runtime.device.read_smid` | exact | `%smid` comes from inline PTX instead of the linked `smid.cu`. |
| `tid == 0` writes (trace, final counters) | stores guarded by `pid == 0` | exact | |
| r2 guard `elif visited[nx, ny] == 0` | masked plain `tl.load(visited)` on in-bounds, not-red lanes | exact | It reads claims from earlier levels, published by the barrier. |
| r2 divergent `if interior:` ring-2 loop | ring 2 masked by `interior`, skipped when no lane of the program is interior | close | The skip is per program, not per warp: structural, since Triton has no per-warp branch. The gate is a program-wide `tl.max`, which costs 3 `BAR.SYNC` per tile. It is a 0/1-trip loop, because Triton 3.7 miscompiles scans inside an `if` (the program enqueue has one). |
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
  the RTX 4060 Laptop (24 SMs), for both enqueue binaries:

  | kernel | Numba regs | Numba cap | Triton lane regs | lane cap | Triton program regs | program cap |
  |---|---|---|---|---|---|---|
  | lin | 104 | 48 | 40 | 144 | 40 | 144 |
  | lin_bare | 79 | 72 | 35 | 144 | 40 | 144 |
  | lin8 | 104 | 48 | 40 | 144 | 64 | 96 |
  | lin8_bare | 79 | 72 | 40 | 144 | 48 | 120 |
  | lin8r2 | 106 | 48 | 64 | 96 | 106 | 48 |
  | lin8r2_bare | 80 | 72 | 58 | 96 | 80 | 72 |
  | xy | 104 | 48 | 36 | 144 | 40 | 144 |
  | xy_bare | 72 | 72 | 36 | 144 | 40 | 144 |
  | xy8 | 105 | 48 | 40 | 144 | 61 | 96 |
  | xy8_bare | 74 | 72 | 40 | 144 | 48 | 120 |

  The per-lane enqueue needs fewer registers: there is no scan to keep
  alive. At tpb 64 and 256 every Triton capacity is at least Numba's
  (measured, both binaries), so a pinned grid of min(all) is Numba's own
  default grid. Compare backends at an explicit, equal `blocks`, as the
  cross-backend tests and the compare script do.
- **Enqueue order differs, enqueue structure does not.** Under the
  default `ENQ="lane"` each warp takes one ticket slab per direction, as
  in Numba. Only the order of entries inside a level can differ, which
  is schedule-dependent in both backends anyway. The first translation
  (`ENQ="program"`) took one slab per program; see "The enqueue switch".
- **The radius-2 ring-2 skip is per program, not per warp.** In Numba a
  warp with no interior lane skips the 16 ring-2 probes. In the twin the
  skip is decided for the whole program (256 lanes, 8 warps at the
  benchmark's tpb), by a program-wide `tl.max` that costs 3 `BAR.SYNC`
  per tile. When any lane is interior, every warp steps through the 16
  probes. Non-interior lanes are masked off, so they cause no memory
  traffic and the outputs and counters are unchanged. But the
  instructions still run, so the Triton side of the radius-2 rows
  (`seq8r2`, `multi8r2`) includes masked probe work that Numba skips.
  Read `r2_multi_vs_conn8` with that in mind. This is structural (Triton
  has no per-warp branch), so it stays under both enqueue settings.
- **One twin-only keyword, `enqueue`.** `flood_fill`, `max_blocks` and
  `kernel_info` take `enqueue="lane"` (default) or `"program"`, and the
  result records it in `DualBlobResult.enqueue`. Numba has one enqueue,
  so the Numba API has no such keyword.
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

## The enqueue switch (`ENQ`)

Every discovery appends one entry to the global queue. A lane takes a
slot from the rear counter, then writes its entry there.

Numba does this in `_warp_enqueue_global`. The warp's active lanes count
themselves (`activemask`, `popc`). One leader lane takes a slab of slots
with one `atomic.add`, and `shfl` hands the slab's start to every lane.

Triton has no `activemask`, `popc` or `shfl`. So the first translation
aggregated over the whole program: `tl.sum` counts the winners,
`tl.cumsum` ranks them, and one atomic serves the program. That kernel
is still here, as `ENQ="program"`.

The default is now `ENQ="lane"`. Each winning lane calls
`tl.atomic_add(rear, 1)` itself and writes its entry at the returned
slot. The PTX says exactly that: one `atom.global.gpu.relaxed.add.u32`
per lane.

But the address is uniform and the operand is the constant 1. So ptxas
rewrites the atomic into Numba's pattern. This is one enqueue site of
`dual_blob_lin_kernel`, as `nvdisasm` prints it:

```
VOTEU.ANY UR14, UPT, PT ;                           // active lanes (activemask)
FLO.U32 R24, UR14 ;                                 // leader lane
POPC R23, UR14 ;                                    // count (popc)
ISETP.EQ.U32.AND P2, PT, R24, R25, PT ;
@P2 ATOMG.E.ADD.STRONG.GPU PT, R23, [R20.64], R23 ; // one atomic per warp
S2R R25, SR_LTMASK ;
LOP3.LUT R25, R25, UR14, RZ, 0xc0, !PT ;
POPC R25, R25 ;                                     // rank (lanemask_lt)
SHFL.IDX PT, R22, R23, R24, 0x1f ;                  // slab start (shfl)
IMAD.IADD R22, R22, 0x1, R25 ;                      // slot = start + rank
```

Numba's `dual_blob_lin_kernel`, disassembled the same way, runs the same
steps. It uses `VOTE.ANY`, `BREV` + `FLO` (the lowest active lane
leads), `POPC`, one leader `ATOMG.E.ADD` of the count and `SHFL.IDX`,
with no CTA barrier.

ptxas also wraps Numba's leader-only atomic in its own aggregation check.
That adds a second `VOTE.ANY` and a full-warp `SHFL.UP` path that a
single lane never takes, so Numba's sequence is a few instructions
longer.

So the per-lane atomic is the faithful twin: it compiles to the machine
code of Numba's hand-written helper. ptxas picks the highest active lane
as leader, where Numba picks the lowest. That only changes the order of
entries inside a slab.

The program version pays for its scan. Its `tl.sum` and `tl.cumsum`
cross the program's 8 warps through shared memory, at 7 `BAR.SYNC` per
enqueue site. Those barriers also hold the warps in lockstep, direction
by direction.

Counted in the tpb=256 binaries with `sass.py` (the bare twins have the
same counts):

| binaries | enqueue atomics | warp-aggregated by ptxas | BAR.SYNC in the binary | BAR.SYNC between two enqueue sites |
|---|---|---|---|---|
| lin, xy, `lane` | 4 | 4 | 10 | 0 |
| lin, xy, `program` | 4 | 4 | 38 | 7 each |
| lin8, xy8, `lane` | 8 | 8 | 10 | 0 |
| lin8, xy8, `program` | 8 | 8 | 66 | 7 each |
| lin8r2, `lane` | 24 | 24 | 13 | 0, except 3 at the ring-2 gate |
| lin8r2, `program` | 24 | 24 | 181 | 7 each, 10 at the gate |

The 10 other barriers of a lane binary sit outside the tile loop: in the
two grid barriers per level and in the exit code. The PTX `bar.sync`
count is the same number, so `compare.py` records it without a
disassembler.

A test (`test_twin_lane_enqueue_is_warp_aggregated_in_sass`) checks this
pattern on all ten twins, from their cubins.

Nothing else changes. Each winning lane still gets exactly one slot. So
the rear, `filled`, `peak_occupancy`, the level trace and every counter
keep Numba's meaning.

The overflow tripwire is the same per-lane bound check. A test forces it,
with a kernel told `qcap=1`. Under both settings it fires, no slot at or
past `qcap` is written, and the rear overshoots `qcap` without
corrupting anything.

### What the barriers cost

Five of the first comparison's worst rows, measured with Numba, Triton
`program` and Triton `lane` on the same scene and the same pinned grid.
The runs were interleaved: 12 rounds, cycling through all six run
orders, after a GPU spin-up.

Kernel time medians in ms, and numba_ms / triton_ms (above 1: Triton
faster):

| row (first-run grid) | Numba | Triton `program` | Triton `lane` | x program | x lane | program / lane |
|---|---|---|---|---|---|---|
| radius2 two_disks_r1400 seq8r2 (48) | 37.73 | 49.89 | 35.50 | 0.76 | 1.06 | 1.41 |
| modes two_disks_r1400 bare_xy (72) | 26.10 | 36.93 | 26.97 | 0.71 | 0.97 | 1.37 |
| modes two_disks_r1400 seq8 (48) | 45.65 | 43.28 | 32.72 | 1.05 | 1.40 | 1.32 |
| modes two_sq_2800 multi8 (48) | 38.35 | 37.19 | 30.41 | 1.03 | 1.26 | 1.22 |
| radius2 two_sq_300 multi8r2 (48) | 2.07 | 3.16 | 2.06 | 0.66 | 1.01 | 1.53 |

Best run vs best run: `program` x0.66-0.87, `lane` x1.00-1.10, and
program / lane 1.22-1.52. Outputs were identical on all three sides.

The GPU sat at 1350-2025 MHz of its 3105 MHz boost, so single runs
spread widely. The Numba medians of the seq8 and multi8 rows are high
against the first run (32.9 and 27.3 ms there). Read those two Numba
ratios loosely.

The program / lane column compares two Triton binaries in the same
rounds, and it is stable. So the first translation's enqueue was the
cost.

With the per-lane enqueue, the twin runs at about Numba's speed on these
rows, at the same grid: x0.97-1.40 by median, x1.00-1.10 best vs best.
The radius-2 rows keep one structural extra, the per-program ring-2
gate.

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

These hold under both enqueue settings. The `test_enqueue_*` tests run
every twin with `enqueue="lane"` and `"program"`, in both modes and on
five scenes. Both are checked against Numba and the CPU oracle, and at
the benchmark grid also against each other.

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
in-process cross-backend streams test compares Triton's streams mode with
Numba's sequential mode, because Numba's own concurrent pair can wedge
after other launches in the same process.

A second, separately opt-in test (`DUAL_BLOB_STREAMS_NUMBA=1`) compares
the two streams modes directly. It runs Numba's 8+8 streams pair in a
fresh process under a 120 s timeout, which is the Numba README's own
fresh-process probe, and skips with the reason if that pair wedges.

During the port, Numba's 8+8 pair wedged in every fresh-process attempt
on this GPU (tpb 256 and 64, stuck in the stream synchronize), so that
test skipped. The twin's pair completed in every run. So the direct
streams-vs-streams comparison is still unverified on this machine.

Comparison (writes `results/triton_twins/ch04_gpu_2blob_nblock/compare_<UTC>.json`):

```
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch04_gpu_2blob_nblock.compare [--quick] [--repeats N]
```

`compare.py` runs seven experiments on the Numba benchmarks' four scenes,
all at 256 lanes. The Triton side uses the default per-lane enqueue,
except for the program rows of the `enqueue` experiment. Every row's
config records its `enqueue`:

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
- `enqueue`: the cost of the first translation. Four configs, one per
  kernel family (`multi`, `bare_xy`, `multi8`, `multi8r2`), on every
  scene, as two rows each: `enqueue="lane"` (label `per_lane`) and
  `enqueue="program"` (label `first_translation`). Both rows of a pair
  run at one grid, the minimum of the Numba capacity and both Triton
  capacities, which is Numba's own grid. Each row also stores
  `triton_ptx_bar_sync`, the CTA barrier count of the binary it ran, and
  `meta.enqueue` holds registers, capacity and barriers per binary.

There are no caps. A run at the big scenes spends about 0.9 s in the
drivers, so the default (100 cases, warm-up and 6 rounds) takes about 18
minutes of GPU time with a 1.5 GB peak RSS. `--quick` runs tiny scenes and writes
nothing. Its copy peaks are a smoke test of the probes only, with no
spin-up and 2 copies.

`speedup_kernel` (the median ratio) is the Numba-vs-Triton number.
Each row also stores `speedup_kernel_min`, the best-vs-best ratio of the
two backends' fastest runs. This is the Numba benchmark's `(min)`
column. Single runs on this GPU can be 1.4-2.8x slower than the median,
on either backend, so a row's difference is real only when the two
ratios agree. If they fall on opposite sides of 1, the row is noise. The
script prints both ratios for every row at the end.

`speedup_total` mostly compares CuPy's caching pool with Numba's
per-array `cuMemAlloc`. `mode="streams"` and the @njit oracle timings are
not run, as in the Numba benchmark and by scope.
