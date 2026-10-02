# Chapter 6 in Triton: runs, not pixels (`triton_twins/chapters/ch06_gpu_nblob_runs/`)

The Triton twin of [`chapters/ch06_gpu_nblob_runs`](../../../chapters/ch06_gpu_nblob_runs/README.md).
Same pipeline, same run table, same union-find protocol, same counters,
same host API. The chapter README explains the design; this one says
only what changed when it moved to Triton.

The per-thread loops of `merge` and `flatten` (binary search, walk,
`_union` retries, `_find`) have two spellings, because a Triton program
has no per-lane control flow. `lane_schedule="independent"` (the
default) runs them as per-lane state machines, each lane moving on as a
SIMT thread does. `lane_schedule="lockstep"` is the first translation,
kept to measure what it costs. See [Lockstep vs lane-independent merge
and flatten](#lockstep-vs-lane-independent-merge-and-flatten).

## What is twinned

| Numba (`chapters/ch06_gpu_nblob_runs/`) | Triton twin (this folder) |
|---|---|
| `kernels.py`: `pack`, `unpack`, `count`, `row_scan`, `emit`, `merge_rows`, `flatten`, `paint`, `label` | `kernels.py`, same names, same arguments plus explicit sizes |
| `merge_rows` / `flatten` with a runtime `instrumented` flag | `INSTRUMENTED: tl.constexpr` (the bare variant has no counter atomics) and `LANE: tl.constexpr` (the lane schedule): four compiled variants each |
| `recolor.py`: `recolor()`, `RunRecolor` (`pack`, `unpack_to`, `run`, `emit_label_map`), `RunRecolorResult`, `_warmup` | `recolor.py`, same signatures, defaults, errors and result fields |
| `benchmarks/benchmark.py` (six scenes, both contracts, read/write peak probes) | `compare.py` cases `benchmark`, plus Triton twins of the `_read` / `_write` probes |
| `benchmarks/scaling.py` (centered crops of `input_blobs.png`) | `compare.py` cases `scaling` |
| (no Numba counterpart) | `compare.py` cases `lane_schedule`: the twin's two schedules against the same Numba pipeline |
| `test_correctness.py` (109 tests) | `test_correctness.py`: the same 109 test names, plus `test_cross_backend_*`, `test_twin_*` and `test_lane_*` |

Not twinned: `benchmarks/figures.py` and `benchmarks/visualize.py` (no
kernels of their own; the one GPU call in figures is `recolor()`), and
`overview/bench_ch06.py` (the overview phase).

The configuration (`DEFAULT_GRID`, `PHASE_BLOCKS`, `PACK_ROW_BLOCKS`,
capacity heuristic, `model_bytes_ch06`), the counter slots and the
palette are imported from the Numba modules, so the two backends cannot
drift apart on any of them.

## Mapping

One Numba block of T threads is one Triton program of T lanes with
`num_warps = T // 32`, on the same grid. The warp-centric kernels work
on a `[T // 32, 32]` tile (row i = warp i, column l = lane l), so every
`warp_id` and every grid-stride loop is the Numba one. The
thread-per-run kernels use a flat `[T]` tensor.

| Numba construct | Triton construct | Fidelity | Note |
|---|---|---|---|
| 6-7 plain launches, stream order as the barrier | 6-7 plain launches on the null stream | exact | no cooperative launch anywhere in ch06 |
| `kernel[blocks, tpb]` | `kernel[(blocks,)](..., num_warps=tpb // 32)` | exact | same grids per phase; the 2D pack grid is `(words/WPB, min(width, 64))` |
| warp per row / per run, `warp_id = grid(1) // 32` | `[WPB, 32]` tile, `warp_id = pid * WPB + i` | exact | same rows, same strides |
| `cuda.ballot_sync` (pack) | `tl.reduce(red << lane, axis=1, or)` | close | the same word; it compiles to one `redux.sync.or.b32` per warp (sm_80+, checked in the PTX), one warp instruction like the vote |
| `shfl_down` reduction (count) | `tl.sum(axis=1)` | exact | one `redux.sync.add.s32` per warp |
| `shfl_up` warp scan + `shfl_sync(31)` total (emit) | `tl.cumsum(axis=1)` + `tl.sum(axis=1)` | exact | same slot for every run |
| `cuda.popc`, `cuda.ffs` | libdevice `popc` / `ffs` on the bit-cast `uint32` word | exact | words stay `uint32`, so `>>` is logical (Numba widens to int64) |
| per-lane bit walk `while s: ffs; s &= s - 1` (emit) | `for _ in range(tl.max(popc))`, masked by `s != 0` | emulated | trip count = max popc in the program (Numba: in the warp) |
| `cuda.shared.array(1024)` Hillis-Steele scan + `syncthreads` (row_scan) | `tl.cumsum` over a `[1024]` register tensor, one program, 32 warps | close | integer scan, identical offsets; the counter reset stays in this kernel |
| `_find` with path halving, lane-divergent | independent (default): one halving hop per lane per mini-step, inside each lane's state machine; lockstep: `while tl.max(active) > 0` over a lane mask | emulated | same loads, halving stores and root per lane in both. Independent: a lane at its root moves on; lockstep: the slowest of the program's lanes sets the pace, and every trip ends in a program-wide max |
| `_union` retry loop, `cuda.atomic.min` | independent: per-lane registers (a, b, cursor, which find) advanced hop by hop, the `tl.atomic_min(sem="relaxed")` at the next full step, retry from the value it returned; lockstep: lane-mask loop | emulated | same protocol in both: find(a) to the end, then find(b), the larger root takes the smaller, retry from `old`. Numba atomics are relaxed; so are these |
| binary search and forward walk with `continue` (merge) | independent: one probe or one walk test per lane per mini-step, the next run's 5 descriptor loads at the full step; lockstep: `while` loops, `continue` as a lane mask | emulated | same probes, same walk tests, same `continue` on the last row. Triton has no `break` / `continue` |
| grid-stride `for r in range(tid, n_runs, stride)` around them (merge, flatten) | independent: each lane walks its own runs, taking the next at the full step after its current one is done; lockstep: `range(...)` batches of BLOCK runs | close | same run -> (program, lane) map; the counters per lane are the same |
| per-thread `cuda.atomic.add` of attempts / done / roots, `if count:` | per-lane `tl.atomic_add(int64, mask=count > 0, sem="relaxed")` under `INSTRUMENTED` | exact | one atomic per thread with a nonzero count, as in Numba; the bare variants have none |
| `shfl_sync(k)` descriptor replay (paint) | masked sum along the lane axis picks lane k's value | close | one `redux.sync.add.s32` per value, one warp instruction like the shuffle; descriptors still fetched 32 at a time, coalesced; the rejected per-run broadcast loads stay out |
| span loop `for y in range(y0 + lane, y1 + 1, 32)` (paint, label) | `for ci in range(n_chunks)`, masked by `y <= y1` | emulated | trip count = longest span among the program's warps at that step |
| `cuda.event(timing=True)`, `event_elapsed_time` | `cp.cuda.Event()`, `cp.cuda.get_elapsed_time` | exact | same bracket points: before the first launch, after each |
| `cuda.to_device`, `device_array`, `copy_to_host` | `cp.asarray`, `cp.empty`, `.get()` | exact | |
| shapes read from arrays (`img.shape`, `mask.shape[1]`, `run_x.shape[0]`) | explicit int args, all `do_not_specialize` | exact | image offsets in int64 |

## Deviations

- **Block size.** Triton needs `num_warps` and `tl.arange` lengths to be
  powers of 2, so the twin accepts tpb in {32, 64, ..., 1024}. A multiple
  of 32 that Numba accepts (96, 160, ...) raises `ValueError` naming the
  power-of-2 rule. The reverse also exists: Numba's `emit_kernel` uses
  79 registers per thread, so a 1024-thread Numba launch fails with
  `LAUNCH_OUT_OF_RESOURCES`, while the twin compiles each kernel for its
  block size and runs it.
- **Warm-up.** Numba compiles once for every block size; Triton's block
  size, `INSTRUMENTED` and `LANE` are compile-time constants.
  `_warmup(tpb, lane_schedule)` compiles all of them for one block size
  (default 256) and schedule (default `"independent"`), and `recolor()`
  calls it for the block size and schedule it is about to use. Every
  size argument is `do_not_specialize`, so a new image shape never
  recompiles, in either schedule (tested).
- **`lane_schedule`.** One twin-only keyword, keyword-only:
  `RunRecolor(..., *, lane_schedule="independent")` and
  `recolor(..., *, lane_schedule=None)` (None runs the engine's schedule,
  or the default when `recolor` builds the engine; a name that differs
  from a given engine's raises `ValueError`). `"independent"` or
  `"lockstep"`, the merge and flatten kernels' `LANE` constexpr. The two
  give identical outputs and counters (tested against each other and
  against Numba). `LANE_SCHEDULES` and `DEFAULT_LANE_SCHEDULE` are
  exported. Numba's signatures are otherwise unchanged.
- **Device arrays.** CuPy arrays; a Numba device array is accepted and
  viewed zero-copy. They must be C-contiguous (the kernels use raw
  pointers), and a host `numpy` array raises `TypeError` instead of being
  copied in and out implicitly.
- **`RunRecolor.compiled`.** The twin keeps the `CompiledKernel` of each
  kernel's latest launch, for `runtime.kernel_resources` (registers,
  spills, shared bytes). Numba has no equivalent attribute.
- **Divergence.** Where a Numba warp diverges, the twin either runs a
  per-lane state machine (merge and flatten, the default schedule) or
  runs the loop until the slowest lane of the whole program is done (the
  emit bit walk, the paint and label span loops, and merge and flatten in
  the lockstep schedule). The section below measures what each costs.
- **Registers.** At tpb 256: merge 39 (independent) / 35 (lockstep), 33
  / 29 bare; flatten 26 / 22, 24 / 20 bare; Numba 34 and 33. Six
  programs per SM in every variant.

## Lockstep vs lane-independent merge and flatten

Numba's merge and flatten are thread-per-run, grid-stride, with
per-thread `while` loops inside: the binary search for the first run of
the next row that can touch run r, the walk along the runs that do, one
`_union` per walk step (its retries, two halving finds each), and
flatten's halving `_find`. A thread that finishes its loops moves on
while its warp neighbours still loop, and the warps of a block never
wait for each other.

### Why the lockstep translation was slow

The first translation ran each loop while any lane of the 256-lane
program still looped, and ended every trip with a program-wide
`tl.max` (a barrier across 8 warps). GPU-only, `input_blobs.png` in the
mask contract, its merge took 0.60 ms against Numba's 0.31 (x0.51), its
flatten 0.047 against 0.033 (x0.70).

Width: the same 131072 threads (merge and flatten), 9000 x 9000
`input_blobs.png`, GPU-only medians:

| tpb x blocks | 32 x 4096 | 64 x 2048 | 128 x 1024 | 256 x 512 |
|---|---|---|---|---|
| Numba merge | 0.374 ms | 0.312 ms | 0.308 ms | 0.309 ms |
| twin merge, lockstep | 0.251 ms | 0.376 ms | 0.466 ms | 0.595 ms |
| Numba flatten | 0.047 ms | 0.036 ms | 0.032 ms | 0.032 ms |
| twin flatten, lockstep | 0.043 ms | 0.042 ms | 0.043 ms | 0.047 ms |

At one warp per program (tpb 32, Numba's divergence unit) the lockstep
merge beats Numba's; at 256 lanes it takes 2.4x longer. Counts, from
copies of both merges that count every loop trip and parent load
(Numba's warp issues through `activemask`), same scene:

| merge | Numba | twin lockstep, 256 lanes | twin lockstep, 32 lanes |
|---|---|---|---|
| lane-steps (search, walk, union, find) | 6.70 M | 6.94 M | 6.96 M |
| warp-steps issued | 0.82 M | 0.97 M | 0.58 M |
| of which in the finds | 0.32 M | 0.62 M | 0.31 M |
| parent loads | 2.75 M | 2.88 M | 2.87 M |
| forest depth after the merge (mean / max) | 3.2 / 20 | 3.1 / 24 | 2.5 / 27 |

The cost is the lockstep itself: 1.7x the warp-steps of the same loops
at one warp per program (the finds use about 10% of their lane slots),
and a barrier on every trip. Unlike ch05, the forest is not the
problem: path halving keeps it at Numba's depth, with the same parent
loads. Flatten's finds are short (2.1 iterations per run) and the
lockstep wastes less there (0.057 M warp-steps against 0.044 M at 32
lanes).

### The lane-independent schedule

`"independent"`, the default, makes each lane a small state machine with
its own registers: its grid-stride run (the same run to (program, lane)
map as Numba), its place inside that run and its union in flight. Cheap,
frequent work runs in mini-steps, during which a program's warps never
wait for each other; after every 4 mini-steps a full step does the rarer
work and takes the program-wide "any lane left?" max:

| | mini-step | full step |
|---|---|---|
| merge | one halving hop of find(a) or find(b), one binary-search probe, one walk test | the link `atomic_min` of every lane whose roots differ (retry or retire, then its next walk test); the lanes whose run is done fetch their next run (5 descriptor loads) and take its first probe |
| flatten | one halving hop | the lanes at their root store `parent[r]` and the label and start their next run |

Every lane fetches its first merge run before the loop. Per run the
operations are Numba's, in Numba's order (the same probes, walk tests,
finds, links, retries and counters), and so are the outputs.

Two placement choices were measured, not assumed:

- **The next run at the full step, not at once.** Lanes that finish
  inside the same window start their next runs together, on neighbouring
  indices, so the per-run loads and stores stay coalesced. Free-running
  lanes drift apart and scatter them: on `random_4000` (26 runs per lane)
  that flatten took 0.63 ms against 0.33 lockstep, and that merge gained
  nothing over lockstep (x0.78 both).
- **The link at the full step, not in the mini-step.** On scenes whose
  runs chain row after row (`serpentine_2048`, `disk_r2000`: one run per
  row), every union is one hop if all finds read the iota before any link
  lands, which is what lockstep does. A program that reads a neighbour's
  boundary run after the neighbour has linked its rows walks the whole
  chain instead, one hop per mini-step, and its program waits. With the
  link in the mini-step programs lost that race on every run (the first
  lane of each program walked 128-386 hops; merge 0.085-0.16 ms against
  0.007 lockstep; fetching the first run before the loop alone fixed
  `serpentine_2048` but not `disk_r2000`). Linking at the full step puts
  a window of finds before a program's first link. Numba has the same
  race and loses it now and then (`disk_r2000` merge: 3 runs of 30 at
  0.12-0.15 ms, the rest 0.006).

Mini-steps per full step: 4 for both kernels. At 8 the merge was 11-19%
slower on the big and the chain scenes; flatten was flat from 4 to 8 and
worse at 2.

### Before / after

GPU-only phase medians (every launch queued behind a 3 ms device spin),
mask contract, Numba and both twin schedules interleaved in one process,
10-12 rounds, `x` = numba_ms / twin_ms:

| scene | merge: Numba | lockstep | independent | flatten: Numba | lockstep | independent | span: lockstep | independent |
|---|---|---|---|---|---|---|---|---|
| `input_blobs` 9000² | 0.310 ms | 0.604 (x0.51) | 0.275 (x1.13) | 0.0328 ms | 0.0471 (x0.70) | 0.0379 (x0.87) | x0.86 | x1.12 |
| crop 8000² | 0.251 ms | 0.511 (x0.49) | 0.246 (x1.02) | 0.0282 ms | 0.0410 (x0.69) | 0.0328 (x0.86) | x0.84 | x1.08 |
| crop 6000² | 0.220 ms | 0.406 (x0.54) | 0.229 (x0.96) | 0.0225 ms | 0.0317 (x0.71) | 0.0256 (x0.88) | x0.94 | x1.22 |
| crop 4000² | 0.135 ms | 0.191 (x0.71) | 0.139 (x0.97) | 0.0113 ms | 0.0143 (x0.79) | 0.0133 (x0.85) | x1.05 | x1.24 |
| crop 2000² | 0.044 ms | 0.061 (x0.72) | 0.049 (x0.89) | 0.0072 ms | 0.0072 (x1.00) | 0.0077 (x0.94) | x1.14 | x1.28 |
| `random_4000` | 1.506 ms | 1.915 (x0.79) | 1.344 (x1.12) | 0.3005 ms | 0.3236 (x0.93) | 0.3164 (x0.95) | x1.27 | x1.42 |
| `blob_grid_100` | 0.069 ms | 0.048 (x1.45) | 0.100 (x0.69) | 0.0072 ms | 0.0092 (x0.78) | 0.0092 (x0.78) | x1.37 | x1.18 |
| `disk_r2000` | 0.012 ms | 0.011 (x1.09) | 0.011 (x1.09) | 0.0072 ms | 0.0092 (x0.78) | 0.0082 (x0.88) | x1.18 | x1.20 |
| `serpentine_2048` | 0.009 ms | 0.009 (x1.00) | 0.008 (x1.12) | 0.0072 ms | 0.0082 (x0.88) | 0.0072 (x1.00) | x1.10 | x1.09 |
| `input_blocks` | 0.052 ms | 0.056 (x0.94) | 0.053 (x0.98) | 0.0189 ms | 0.0195 (x0.97) | 0.0184 (x1.03) | x1.41 | x1.36 |

Geometric means over the ten scenes: merge x0.78 lockstep, x0.99
independent; flatten x0.82 and x0.90; the GPU-only span x1.10 and x1.21.
On the three largest (`input_blobs` and the 8000 and 6000 crops), where
merge dominates the mask contract: merge x0.51 and x1.03, flatten x0.70
and x0.87, span x0.88 and x1.14. The counters agreed on every scene;
the outputs are compared by the tests and by `compare.py` on every run.

### What it does not recover

- `blob_grid_100` loses its merge: x1.45 lockstep, x0.69 independent.
  Each of its 100 squares is a 360-run chain spread over 14 programs.
  Lockstep links all of them while every find still reads the iota (the
  forest after the merge is a full chain, mean depth 178), so each union
  is one hop. Lane-independent programs still lose the chain race at
  some boundaries, and one lane walks up to 181 hops. The absolute cost
  is 0.05 ms.
- Flatten stays behind Numba's (x0.86-0.88 on the big scenes). Its finds
  are short (2.1 iterations per run), so the lockstep waste was small to
  begin with; the lane-independent flatten still takes a program-wide max
  every 4 hops and holds a finished lane until the window ends, which
  Numba's warps do not.

## Run

Tests (same CPU oracle as the Numba chapter, plus Numba-vs-Triton
equality on every variant; the `test_lane_*` tests run both lane
schedules against each other and against Numba, also with merge and
flatten pinned to 2-3 blocks so every lane walks hundreds of runs, on
the multi-chunk shapes and on many-run scenes):

```
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch06_gpu_nblob_runs
```

Performance comparison (Numba vs Triton on the chapter's own
benchmarks, written to `results/triton_twins/ch06_gpu_nblob_runs/`):

```
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare            # full, ~5-7 min
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare --quick    # smoke test, no JSON
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare --experiments lane_schedule
```

| experiment | what | cases |
|---|---|---|
| `benchmark` | `benchmark.py`'s six scenes, both contracts | 12 |
| `scaling` | `scaling.py`'s crops of `input_blobs.png`, both contracts | 20 |
| `lane_schedule` | the six benchmark scenes, mask contract, both schedules (`config.lane_sched`) | 12 |

The twin runs its default schedule everywhere except the
`lane_schedule` experiment's `lockstep` rows. Those are the first
translation: label `first_translation`, `first_translation=true`,
`comparable=false`, so they stay out of every average and get the
summary's paired first-translation figures (as in ch01-ch05). The
`lane_independent` rows repeat benchmark cells and carry
`duplicate_of="benchmark"` (when that experiment runs).

Each row has both backends' `kernel_ms` (CUDA-event span of the
pipeline, the chapter's number) and `total_ms` (host wall time of
`run()` + synchronize), their per-phase medians, `model_gb_s`,
`speedup_kernel = numba / triton` (above 1: Triton faster), the floor
arithmetic at each backend's own read/write peaks, the Triton kernels'
registers and the Numba kernels' registers. Outputs are compared on the
device after every run. The scaling sweep's 1.0 / 0.5 ms crossings are
in `meta.scaling_crossings`, per backend. `meta.caps` lists what the
full run leaves out of the Numba benchmarks (the ch05 column, the 8 s
spin).

**Small scenes are launch-bound.** The events sit on the stream. When
the GPU finishes a kernel before Python has queued the next one, the
event span includes the host's launch time.

A pipeline run is 6-7 launches, about 25 us each in Triton and about
57 us each in Numba. Below roughly 4 Mpx (`input_blocks`, the
1000-2000 px crops, and the mask rows of the smaller scenes),
`kernel_ms` and `phase_ms` therefore mostly measure Python launch
overhead, for both backends. There `speedup_kernel` is a host-stack
result, not a kernel result. Each row also carries:

- `gpu_only`: the same pipeline, once per call, with every launch queued
  behind a 3 ms device spin, so its event span is GPU work only.
  `speedup_gpu_kernel` and `speedup_gpu_phase` compare the kernels;
- `enqueue_ms`, `gpu_fraction` (GPU-only span / event span) and a
  `launch_bound` flag (`enqueue_ms >= 0.8 * kernel_ms` or
  `gpu_fraction < 0.8`, per backend; the row flag is set when either
  backend's is).

In the quick smoke run, for example, every scene reads about 2x
"faster in Triton" on the event span, but only 0.6-1.3x GPU-only.
