# Chapter 2 in Triton: the dual-block BFS twin

This folder rebuilds `chapters/ch02_gpu_1blob_2block` in Triton: the same
four kernels, the same host API, the same tests against the same CPU oracle,
and a Numba-vs-Triton comparison of the chapter's own benchmark.

| file | role |
|---|---|
| `kernels.py` | `global`, `split`, `dirsplit` (each with an `INSTRUMENTED` constexpr: `False` is the bare twin) and `pinned`; every kernel takes the `ENQ` constexpr (`"lane"` default, `"program"` = the first translation) |
| `flood_fill.py` | the host driver: same `flood_fill(...)` signature, defaults, validation and `DualFloodFillResult` fields as Numba, plus the twin-only `enqueue="lane"` keyword |
| `test_correctness.py` | the Numba test file test for test (same 102 names), plus `test_cross_backend_*`, `test_enqueue_*` (both enqueue modes vs Numba and the CPU oracle) and SASS and PTX checks of the enqueue |
| `compare.py` | the chapter benchmark's scenes, tpb sweep and placement experiment, plus an `enqueue` experiment (both enqueue modes), timed on both backends |

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
| `dual_block_pinned_kernel` (48 x 768, `max_registers=40`) | `dual_block_pinned_kernel` (72 x 512 by default, 48 x 512 with `enqueue="program"`; `maxnreg=64`) | cooperative; same_sm launch size, see the pinned deviation |

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
| `_warp_enqueue_global` (ballot, popc, leader atomic, shfl: one atomic per warp) | `_enqueue_global`, `ENQ="lane"`: a masked per-lane `tl.atomic_add(rear, 1, sem="relaxed")`, each lane writes its item at the returned ticket | exact | ptxas warp-aggregates the per-lane atomic: same SASS pattern as Numba (VOTE.ANY, POPC, one leader `ATOMG.E.ADD`, `SHFL.IDX`, lanemask POPC), no CTA barrier; see "Enqueue" below |
| (first translation) | `_enqueue_global`, `ENQ="program"`: `tl.sum` + `tl.cumsum` over the program, one atomic per program | close | about 7 extra `BAR.SYNC` per append site, all warps of a program in lockstep; measured 1.18-2.08x slower than `"lane"` on every timed row |
| `_warp_enqueue_two_tier` (shared ring, then spill) | `_enqueue_two_tier_lane` (`ENQ="lane"`) | exact | statement for statement: per-lane ticket atomic, `ticket - front < 8192` takes ring slot `ticket & 8191`, the rest take a second per-lane spill atomic (Numba's re-ballot); both warp-aggregated by ptxas |
| (first translation) | `_enqueue_two_tier` (`ENQ="program"`) | close | same tickets, window and spill rule; the second ballot is the closed form `rank - room` |
| `cuda.shared.array(8192)` ring + shared rears | 8192-slot region of a `(2, 8192)` global scratch array per program; rears: two program-private int32 slots of `g_state` (`"lane"`) or registers (`"program"`) | emulated | no user shared memory in Triton, so the ticket atomics are global (L2) atomics where Numba's are shared; the ring rear overshoots by the spilled tickets and is clamped back at the level end, as in Numba |
| backward append `arr[cap-1-idx]` | same | exact | dirsplit's double-ended buffer |
| OVERFLOW tripwire | same slot, same host `RuntimeError` | exact | |
| per-thread `my_processed`, `my_cas_attempts` | per-lane `[TPB]` int32 accumulators | exact | summed per program at exit; Numba's per-thread exit atomics become one atomic per program |
| grid-uniform level state (`front`, `rear`, peaks, active sums) | program scalars | exact | same values, same counter slots |
| `level_sizes[bx, level]` trace | same layout `(2, trace_cap)` | exact | |
| `get_smid()` (linked `smid.cu`) | `read_smid` (inline PTX `%smid`) | exact | |
| `max_registers=120` (split) / `40` (pinned) | `maxnreg=120` / `64` | close | see the pinned deviation below |
| `max_cooperative_grid_blocks(tpb)` | `max_coresident_programs(compiled)` | exact | same RuntimeError text when the grid does not fit |
| tpb: any multiple of 32 in [32, 512] | powers of 2 in [32, 512] | close | 96, 160, ... raise ValueError naming the power-of-2 rule |
| `_warmup` keyed on (kernel, bare) | keyed on (kernel, bare, tpb, enqueue) | close | TPB and ENQ are constexprs; every int argument is `do_not_specialize`, so no compile lands in `kernel_ms` |

## Deviations

- **Pinned experiment, 512 instead of 768.** Numba pins two 768-thread
  blocks to one SM: 48 blocks at 40 registers fit exactly 2 per SM, so the
  pair holds 1,536 threads, the SM's full residency. Triton needs a power
  of 2, so the twin uses 512-lane programs capped at 64 registers. The
  default build (`enqueue="lane"`) uses 30 registers, so `programs_per_sm`
  reports 3 (the 1,536-thread limit) and the twin launches
  `3 x sm_count = 72` programs: every SM hosts exactly three, the first two
  on the chosen SM work and the third leaves at once (the kernel's
  `rank < 2` guard). The first translation (46 registers) fits exactly 2
  and launches 48, as Numba. Either way the two workers share one SM and
  hold 1,024 threads (67% of the SM), not 1,536 (100%). The experiment's
  question (same SM vs spread) is the same; the "full residency" premise
  is not reproducible. Validation and tests use the twin's
  `PINNED_TPB = 512` where Numba hardcodes 768.
- **split's ring is global memory.** The chapter's split kernel bets on
  shared-memory latency; the twin's ring is L1/L2-cached global scratch
  with the same capacity and overflow semantics (spill-free on 2600^2,
  both halves spill on 4600^2). Under `enqueue="lane"` its two rears are
  global int32 slots too, so the ticket atomics go to L2 where Numba's
  stay in shared memory. So split vs global means something different
  under Triton: both queues are in global memory.
- **Aggregation leader.** Under `enqueue="lane"` the aggregation is per
  warp, as in Numba, but ptxas elects the highest claiming lane as leader
  (`FLO`) where Numba's code elects the lowest (`ffs`). Ranks are still
  lanemask popcounts, so tickets inside a warp keep lane order. The first
  translation (`enqueue="program"`) reserves one slab per program per call
  site. Only schedule-dependent outputs can differ (queue order, global's
  per-pixel owner speckle, split's seam-race inbox/spill counts,
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
red, whatever the interleaving. So these are identical across runs,
kernels, backends and enqueue modes: `img`, `visited`, `depth`, `levels`,
`filled`, `level_sizes`, `peak_level`, `peak_occupancy`, `processed`, and
`cas_attempts` (one attempt per directed edge p -> q with
depth(q) = depth(p) + 1). A `cas_attempts` mismatch would mean a stale
read slipped past a barrier.

Per kernel: split's owner map, per-program counts, per-program trace and
utilization are exact (ownership is spatial); global's per-program counts,
trace and utilization are exact (item-to-program is positional at the same
tpb), its owner census is exact but the per-pixel owner is not. dirsplit's
per-program numbers are race-dependent (only the totals are exact). The
pinned BFS outputs are compared between Numba 768 and Triton 512. `%smid`
values are never compared.

## Enqueue: the per-lane atomic and the first translation

Numba's appends are warp-aggregated by hand. `activemask` and `popc`
count the claiming lanes, the lowest lane adds that count with one
atomic, and `shfl_sync` hands every lane the base. Each lane then adds
its lanemask rank.

Triton has no warp intrinsics. So the first translation aggregated over
the whole program (`enqueue="program"`): `tl.sum` and `tl.cumsum` of the
claim mask, then one atomic per program per call site.

Those scans compile to CTA barriers: about 7 `BAR.SYNC` per append site
in `global`/`dirsplit`, 5-6 in `split`. They also make all warps of a
program wait for each other, direction by direction. Numba's appends have
no barrier at all.

The default (`enqueue="lane"`) writes the append the simple way. Every
claiming lane does its own masked
`tl.atomic_add(rear, 1, sem="relaxed", scope="gpu")` and stores its item
at the ticket it got back.

ptxas recognises an add of a constant to a warp-uniform address and
warp-aggregates it. Disassembled (`nvdisasm` on `compiled.asm["cubin"]`,
sm_89, Triton 3.7.1), one append site of the bare `global` kernel is:

```
VOTEU.ANY UR6, UPT, PT                ; mask of the claiming lanes
FLO.U32 R6, UR6                       ; leader = highest claiming lane
POPC R21, UR6                         ; the warp's count
@P5 ATOMG.E.ADD.STRONG.GPU PT, R21, [R4.64], R21   ; leader only
S2R R7, SR_LTMASK
LOP3.LUT R22, R7, UR6, RZ, 0xc0, !PT
POPC R7, R22                          ; lane rank
SHFL.IDX PT, R6, R21, R6, 0x1f        ; leader's base to every lane
IMAD.IADD R7, R6, 0x1, R7             ; ticket = base + rank
```

Numba's `_warp_enqueue_global` compiles to the same steps: `VOTE.ANY`,
`BREV` + `FLO` for the lowest lane, `POPC`, one leader `ATOMG.E.ADD`,
`SHFL.IDX`, `SR_LTMASK` + `POPC`.

In the lane builds every `ATOMG.E.ADD` is aggregated this way. `global`
and `dirsplit` have 4 append sites, `split` has 12 (ring ticket, spill
ticket and inbox, per direction). `pinned` has 7 add atomics: its 4
append sites, plus the worker-rank dispenser and 2 pair-barrier arrivals,
which ptxas wraps the same way.

`BAR.SYNC` counts per kernel, lane vs program: `global` bare 10 vs 38,
`dirsplit` bare 10 vs 38, `split` bare 15 vs 60, `pinned` 18 vs 46. At
tpb 32 (one warp per program) the scans need fewer barriers: `global`
bare 10 vs 18, `split` bare 15 vs 20.

The SASS alone does not tell the two modes apart. ptxas also wraps the
first translation's single slab atomic in the vote/leader idiom, so the
program builds pass the same aggregation check.

The PTX does tell them apart. In a lane build each append site is a
predicated per-lane `atom.global.gpu.relaxed.add.u32`, and its result
goes straight into that lane's `st.global`, with no `bar.sync` or
`st.shared` in between. In a program build every append atomic is a
scalar: a `bar.sync`, the atomic, then a `st.shared` broadcast and
another `bar.sync` before any lane can store.

Two tests check this on every kernel at tpb 256 (512 for `pinned`), and
on `global` and `split` at tpb 32:
- `test_lane_enqueue_is_warp_aggregated_in_sass`: every add atomic is
  aggregated, and the program build has more barriers (20+ more above one
  warp per program).
- `test_lane_enqueue_is_per_lane_in_ptx`: the lane build has exactly the
  per-lane append sites above (plus pinned's one scalar rank dispenser),
  and the program build has none.

So a compiler change that stops the aggregation fails a test.

`split` needs one more step. Its rears were shared-memory atomics in
Numba and registers in the first translation. Under `"lane"` they are two
program-private int32 slots of `g_state`, one 128-byte line per program.

The seed owner stores its ring rear (1) before the prologue barrier. At
each level end, after the `cta_sync`, one thread reads both rears. It
clamps the ring rear to `front + 8192` and writes the clamp back, as
Numba's thread 0 does.

So the tickets that went to the spill tier are retracted, and nothing was
ever written past the window. Spill, peak and processed counters keep
Numba's meaning.

The tests check this on a scene where only one program works and its
ring overflows. There the spill counts and the peak spill window are
deterministic, and they equal Numba's in both modes. The scene runs as
built (program 0 spills) and mirrored along x (program 1 spills, through
its own rear slots), at tpb 32, 256 and 512. The bare split runs through
the same spill tier and matches Numba and the CPU oracle.

Register use, lane vs program: `split` 72 vs 93 (bare 53 vs 75), `global`
39 vs 36 (bare 28 vs 40), `dirsplit` 40 vs 39 (bare 28 vs 40), `pinned`
30 vs 46.

### Measured before and after (RTX 4060 Laptop)

Kernel ms, median of 9 interleaved rounds per row: Numba, lane and
program in rotating order, after an 8 s spin-up. Outputs were identical
in all three.

Speedup is Numba / Triton, so above 1 means the twin is faster. The rows
are the chapter's worst rows in the first full comparison, plus the
narrow and small scenes.

| row | Numba | Triton program (first translation) | Triton lane (default) |
|---|---|---|---|
| dirsplit sq_4000_corner | 180.3 | 323.2 (x0.56) | 158.5 (x1.14) |
| dirsplit bare sq_4000_corner | 173.0 | 310.2 (x0.56) | 148.9 (x1.16) |
| global sq_4600_full_center | 166.5 | 247.1 (x0.67) | 150.1 (x1.11) |
| split sq_4600_full_center (both halves spill) | 180.7 | 249.3 (x0.72) | 153.9 (x1.17) |
| split bare sq_4000_corner | 195.8 | 275.5 (x0.71) | 168.6 (x1.16) |
| split bare offcenter_2000 | 21.4 | 29.7 (x0.72) | 18.0 (x1.19) |
| split serpentine_256 | 115.6 | 136.8 (x0.84) | 89.9 (x1.29) |
| split seam_serpentine_256 | 113.4 | 132.8 (x0.85) | 88.3 (x1.28) |
| global serpentine_256 | 111.0 | 106.7 (x1.04) | 71.1 (x1.56) |
| dirsplit serpentine_256 | 121.1 | 104.5 (x1.16) | 71.8 (x1.69) |
| split sq_256_center | 0.80 | 0.78 (x1.02) | 0.60 (x1.33) |
| split tpb 64 sq_2000_center | 27.9 | 33.9 (x0.82) | 24.2 (x1.15) |
| split tpb 512 sq_2000_center | 17.7 | 25.2 (x0.70) | 14.7 (x1.21) |
| pinned 2x512 spread sq_2000_center (matched) | 10.1 | 17.4 (x0.58) | 11.9 (x0.85) |

A second pass, in another session, re-timed 9 of these rows plus
`global sq_256_center` and `global` tpb 512 `sq_2000_center`. It agreed
within 5% on every ratio except the short pinned row. There it gave lane
x0.92 and program x0.78.

A third session added the four worst rows of the first comparison that
the table above lacks, with the same method (6 interleaved rounds after
an 8 s spin-up, identical outputs in every run):

| row | Numba | Triton program (first translation) | Triton lane (default) |
|---|---|---|---|
| pinned 2x512 spread sq_6000_center (matched), first run x0.625 | 206.9 | 329.7 (x0.63) | 201.3 (x1.03) |
| global bare sq_6000_center, first run x0.645 | 272.2 | 428.6 (x0.64) | 247.6 (x1.10) |
| global sq_6000_center, first run x0.662 | 289.4 | 434.6 (x0.67) | 264.8 (x1.09) |
| global bare sq_5000_center, first run x0.667 | 191.6 | 290.7 (x0.66) | 172.0 (x1.11) |

On the big pinned row lane beat Numba in 5 of 6 rounds (per-round median
x1.05). An independent re-measurement by the reviewer gave x1.06.

The same session re-timed the short pinned row twice. The medians were
lane x1.06 and x1.10, program x0.75 and x0.78. Its 10-25 ms kernels sit
where the laptop GPU drops its clock between runs (915-2070 MHz
recorded). So per-round ratios ranged from x0.79 to x2.48, and lane was
ahead in 13 of 18 rounds.

What this shows:

- **The first translation's loss was the enqueue.** With nothing else
  changed, the lane path is 1.18-2.08x faster than the program path on
  every row of every pass. Every `split`, `global` and `dirsplit` row is
  now ahead of Numba (x1.09-1.70), including all the rows that trailed
  (down to x0.56).
- **`split` is ahead even though its ticket atomics are global.** Numba's
  ring rears are shared-memory atomics, the twin's are L2 atomics, and
  the twin still wins by 16-29%. A likely cause, read from the code and
  the register counts but not profiled: the twin uses 32-bit indices and
  fewer registers (72 vs Numba's capped 120).
- **`pinned` is level with Numba or ahead.** On the large matched row
  (`sq_6000_center`) the lane build leads, x1.03-1.06, up from x0.625.
  The short `sq_2000_center` row moved from x0.58-0.78 to x0.85-1.10
  across four passes. Its kernels run 10-25 ms and are clock-sensitive,
  so it shows no stable gap either way.
- **The pair barrier differs, but it was not measured.** In the twin
  every scalar atomic of the barrier is broadcast to the whole 16-warp
  program through shared memory. That is a per-level cost Numba's
  thread-0 spin does not pay. It was not ablated, and no gap is
  attributed to it. It was left as it is, since it is not an enqueue.
- **Placement rows are not all like-for-like.** The chapter's own rows
  pit 2 x 768 Numba threads against 2 x 512 Triton lanes, so their ratio
  mixes a config change with the backend change (`comparable=false` in
  the JSON). Only the matched row above is a backend ratio.

`compare.py` repeats this in the harness: its `enqueue` experiment runs
each kernel on `sq_2000_center`, `sq_4000_corner`, `serpentine_256` and
`seam_serpentine_256` (and the matched pinned row) twice, once per
enqueue mode.

The lane rows carry the config label `per_lane`, the program rows
`first_translation`, the pinned pair included; its placement and tpb are
in the config. These are the labels the ch01, ch03 and ch04 twins use.

The program rows are `comparable: false` and carry
`first_translation: true`, as in ch01, ch03 and ch04. They time code the
twin no longer runs by default. So they stay out of the like-for-like
averages and extremes of `summary.py` and `figures.py`, and still show in
the experiment's own block and in the JSON.

The lane rows repeat cells that `scenes` and `placement` already time.
They carry `duplicate_of` naming that experiment. A unit-wide average
should skip rows with that tag, so each cell counts once.

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

It mirrors every GPU row of the benchmark (all at the default
`enqueue="lane"`):
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

It adds the `enqueue` experiment above: Numba vs both enqueue modes on
four scenes x three kernels plus the matched pinned row, 26 rows.

The @njit rows have no GPU backend and are not repeated. Read
`speedup_kernel`: `total_ms` also compares the host stacks (CuPy's pooled
allocation vs Numba's `cuMemAlloc` on every run), which dominates on small
scenes.
