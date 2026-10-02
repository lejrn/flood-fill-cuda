# ch05 in Triton: N blobs, N blocks, no seeds

The Triton twin of [`chapters/ch05_gpu_nblob_nblock`](../../../chapters/ch05_gpu_nblob_nblock/README.md).
Same two discovery strategies (`seed_merge`, `ccl_fill`), same phases, same
atomicMin union-find, same barriers, same counters, same host API. Read the
Numba chapter for what the algorithm does and why it is correct; this page
only records how each CUDA construct was spelled in Triton, and what that
costs.

The union-find loops (`_find`, `_union`) have two spellings, because a
Triton program has no per-lane control flow:
`lane_schedule="independent"` (the default) runs them as per-lane state
machines, each lane moving on as a SIMT thread does;
`lane_schedule="lockstep"` is the first translation, kept to measure what
it costs. See [Lockstep vs lane-independent
union-find](#lockstep-vs-lane-independent-union-find).

| File | Role |
|---|---|
| `kernels.py` | the `@triton.jit` twins; phases are shared helpers |
| `flood_fill.py` | host driver: `flood_fill`, `max_blocks`, `discovery_only`, plus `compiled_kernel` / `kernel_info` / `regs_per_thread` |
| `test_correctness.py` | the Numba suite's GPU tests, same names, plus `test_cross_backend_*` |
| `compare.py` | Numba vs Triton timings on the chapter's benchmark, seeding, tuning and PNG experiments |

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
| `seed_merge_lat_kernel` | `seed_merge_lat_kernel[INSTR=True]` |
| `seed_merge_lat_bare_kernel` | `seed_merge_lat_kernel[INSTR=False]` |
| `seed_merge_lat_r128_kernel` (`max_registers=128`) | `seed_merge_lat_kernel[INSTR=True]` launched with `maxnreg=128` |
| `seed_merge_lat_core_kernel` (split build) | `seed_merge_lat_core_kernel` |
| `lat_compress_kernel` (split build, plain) | `lat_compress_kernel` |
| `lat_finish_kernel` (split build, plain) | `lat_finish_kernel` |
| `seed_scan_kernel` (benchmark phase) | `seed_scan_kernel` |
| `ccl_kernel` (benchmark phase) | `ccl_kernel` |

`INSTR=False` compiles the instrumentation out (counters, owner and
prov_label stores, phase stamps, level trace, block_stats, the closing
seed_merge barrier); those pointer arguments are passed as `None`. The
lattice twin reuses seed_merge's fill and flatten helpers and adds a
lattice scan (P1) and a compress pass plus barrier (between fill and
flatten); the split build's two plain kernels are those compress and
flatten helpers with no barrier, on the Numba driver's `_PLAIN_GRID`.

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
| `_find` per-thread while loop | independent (default): one parent hop per lane per mini-step, inside each lane's state machine; lockstep: a loop over a lane mask, exit checked every 8 hops | emulated | independent: a lane at its root moves on to its next union or item; lockstep: finished lanes run masked no-ops and the slowest of the program's lanes sets the pace |
| `_union` while-True with early returns (a real device call) | independent: per-lane registers (a, b, cursor, which find) advanced hop by hop, the atomic_min at the next full step, retry from the value it returned; lockstep: lane-mask loop (`go`, `retired`), inlined | emulated | same protocol and termination argument in both; per item the same finds, links and counters as Numba |
| grid-stride loops that call `_find` / `_union` (ccl merge and flatten, compress) | independent: each lane walks its own grid-stride items and takes the next as soon as its current one is done; lockstep: `range(...)` batches | close | same item -> (program, lane) map. The ccl merge reads an item's 4 lex-predecessor pixels together before its unions (img is constant there, so the pairs and their order are Numba's) |
| seed_merge fill: `_union` at a CAS loss, between two direction probes | per-direction batches; independent: a program switches to lane-independent probing for the rest of a level after a batch with >= 3 colliding probes per lane (if >= 2 slots per lane remain), a lane staying fewer than 16 slots ahead of the program's slowest | close | scheduling only: per item the same probes, unions and counters in Numba's order, the same slot map and barriers |
| per-thread `my_*` registers, exit `atomic.add` per thread | per-lane register vectors, one reduction + one atomic per program at exit | exact | totals identical (tested against Numba) |
| `%clock64` around each in-flight `_union`, summed per thread | per colliding lane: the program's lockstep union (per-direction batches), or the lane's own union from its collision to its end (lane-independent levels) | emulated | per-thread meaning kept: the lane is busy for that time. Both include waits Numba's thread does not have, so `union_thread_ms` reads higher than Numba's by construction: compare it within one backend only |
| `smid.cu` (`%smid`, `%globaltimer`) | inline PTX (`read_smid`, `read_globaltimer`) | exact | |
| `lat_stride > 0 and x % lat_stride == 0 and ...` (short-circuit) | `hit = red & (lat_stride > 0) & (x % max(lat_stride, 1) == 0) & ...` | exact | lanes evaluate every operand, so the modulo never divides by 0 |
| `_is_interior` (8 probes, early `return False`) | 8 probes, each masked by the lanes still interior | exact | a failed probe masks the later ones off: the early return's loads |
| `if not is_cand:` corner probes | 4 probes masked by `red & ~cand` | exact | |
| COMPRESS: `if parent[i] != i: parent[i] = _find(parent, i)` + `grid.sync()` | independent: `_compress_lanes` (a lane writes its root the step it reaches it); lockstep: `_compress` (masked `_find`, masked store); then `_sync` | close | same barrier, in both twins |
| seed_merge P3 flatten: `final = _find(parent, prov)` | `_merge_flatten`: the lockstep `_find` in both schedules | close | neighbouring pixels share a provisional label, so lockstep chases walk the same chain together and their loads broadcast; per-lane chases lose that (comb_2000: 52 ms lockstep, 97 ms per lane, Numba 26) |
| `cuda.jit(max_registers=128)(py_func)` | same `@triton.jit` body, `maxnreg=128` launch option | close | Triton honours the cap but its register counts are its own (table below) |
| split: core, then `lat_compress_kernel[_PLAIN_GRID]`, `lat_finish_kernel[_PLAIN_GRID]` | core, then two plain launches of 256 programs x 256 lanes | exact | stream order is the compress-before-flatten barrier on both backends |
| `cuda.event(timing=True)` around the cleanup kernels | `cp.cuda.Event()` + `cp.cuda.get_elapsed_time` | exact | same three record points, phase keys `compress` / `flatten` |
| `cuda.const.array_like` DX8/DY8/PDX/PDY | `tl.constexpr` tuples under `tl.static_range` | exact | same probe order E SE S SW W NW N NE |
| `cuda.const.array_like(PALETTE_HOST)` | 18-byte device array, cached | close | |
| `tid == 0` / `threadIdx.x == 0` stores | `if pid == 0:` scalar stores / per-program stores | exact | |
| device arrays from `cuda.to_device(host)` / `device_array` | `cp.asarray(host)` / `cp.empty` | exact | same host arrays uploaded in the same alloc bracket |
| one `[1, 32]` warm-up launch per kernel | one 1-program launch per (kernel, bare, tpb) | close | `num_warps` is compile-time; every runtime int (sizes, `lat_stride`, `lat_interior`) is `do_not_specialize`, so no scene size or stride recompiles (tested) |
| `regs_per_thread(dispatcher)` | `regs_per_thread(compiled)`, `kernel_info(...)` | close | a Triton kernel has one compiled object per block size |

## Deviations

- `threads_per_block` must be a power of 2 (32 ... 512). Numba accepts any
  multiple of 32; the twin raises `ValueError` naming the power-of-2 rule.
- Grid-stride indices are int32 in the twin (Numba's `range` loops are
  int64), so the twin also needs `width * height + blocks *
  threads_per_block < 2**31` (the split build counts its 256 x 256 plain
  grid too) and raises `ValueError` otherwise. Numba's own limit is
  `width * height < 2**31`. The gap cannot be reached on an 8 GB GPU:
  such an image needs about 35 GB of device buffers.
- Image layout: the twin applies `cuda.to_device`'s own contiguity check
  (Numba's `sentry_contiguous`, same `ValueError`), so a non-contiguous
  view is refused on both backends. An F-ordered image is accepted by
  both; the twin copies it to C order on the host (its kernels index C
  bytes), where Numba uploads it as is and indexes through its strides.
  The outputs are equal (tested).
- Two Triton-only buffers per launch: the grid barrier counter (int64,
  zeroed in the alloc bracket) and the palette.
- One twin-only keyword, `lane_schedule` (keyword-only, on `flood_fill`,
  `max_blocks`, `discovery_only`, `compiled_kernel`, `kernel_info` and
  `phase_kernel`): `"independent"` (the default) or `"lockstep"`, the
  kernels' `LANE` constexpr. Each is its own compile with its own
  capacity; the outputs are identical (tested against each other and
  against Numba). Numba's signatures are otherwise unchanged.
- `blocks=None` resolves to the twin's own capacity. On the RTX 4060
  Laptop (24 SMs), registers / co-resident grid (twin: lockstep and
  independent builds):

  | kernel | Numba @256 | lockstep @256 | independent @256 | Numba @128 | lockstep @128 | independent @128 |
  |---|---|---|---|---|---|---|
  | seed_merge | 114 / 48 | 116 / 48 | 128 (10 spills) / 48 | 114 / 96 | 96 (6 spills) / 120 | 115 / 96 |
  | seed_merge bare | 64 / 96 | 76 / 72 | 80 / 72 | 64 / 192 | 76 / 144 | 80 / 144 |
  | ccl_fill | 113 / 48 | 80 / 72 | 64 (4 spills) / 96 | 113 / 96 | 64 / 192 | 56 (6 spills) / 216 |
  | ccl_fill bare | 66 / 72 | 46 / 120 | 47 / 120 | 66 / 168 | 48 / 240 | 48 / 240 |
  | lattice fused | 129 / 24 | 116 / 48 | 128 (12 spills) / 48 | 129 / 72 | 96 (6 spills) / 120 | 119 / 96 |
  | lattice fused bare | 64 / 96 | 76 / 72 | 80 (2 spills) / 72 | 64 / 192 | 76 / 144 | 80 / 144 |
  | lattice r128 | 122 / 48 | 118 / 48 | 119 / 48 | 122 / 96 | 123 / 96 | 115 / 96 |
  | lattice split core | 114 / 48 | 116 / 48 | 128 (10 spills) / 48 | 114 / 96 | 96 (6 spills) / 120 | 115 / 96 |

  The split cleanup kernels: Numba 30 / 33 registers, Triton 18 (19
  independent) / 22. The seed_merge builds carry both fill bodies (see
  the mapping table), hence 128 registers at tpb 256; every pinned grid
  of the comparison (min of the Numba and the twin capacity) is the same
  for both schedules at tpb 256.

- The chapter's register story does not carry over. Numba's fused
  lattice kernel needs 129 registers, one over the line that allows two
  256-thread blocks per SM, so it runs 24 cooperative blocks, and r128 /
  split exist to win back 48. Triton compiles the same fused body to 116
  (lockstep) or 128 (independent) registers, so all three builds already
  run 48 programs at tpb 256; r128 changes nothing there, and at tpb 128
  the cap even lowers the lockstep build's count (96 with spills becomes
  123 without, 120 programs become 96). The builds are still separate
  compiles with their own measured capacity, and all three give
  bit-identical output (tested against each other and against Numba).

- Schedule-dependent outputs differ between backends as they differ
  between Numba runs: queue order, `owner`, `prov_label` at equidistant
  seams, `seed_merge` `union_attempts`, `ccl_fill` `cas_attempts`, smid,
  timings. The tests never compare them, nor what is built from them:
  `model_bytes` / `model_gb_s` (they add those two counters; compared
  only on a `seed_merge` scene with no unions), `union_cycles` /
  `union_thread_ms` (emulated, see the mapping table). `blocks`,
  `thread_util_pct` and `processed_per_block` depend on the grid, so
  they are compared only with `blocks` and `threads_per_block` pinned;
  at `blocks=None` each backend runs its own capacity.

## Lockstep vs lane-independent union-find

Numba's `_find` and `_union` are per-thread `while` loops: a thread that
reaches its root moves on to its next union or its next grid-stride item
while its neighbours still climb. A Triton program has no per-lane control
flow, so the twin spells them two ways (`lane_schedule`, the kernels'
`LANE` constexpr).

### Why the lockstep translation was slow

`"lockstep"`, the first translation, runs each loop while any lane of the
program still climbs. Every lane waits for the slowest of the program's
256 at every find and every retry. On the solid scenes this cost 3-13x
Numba's time, in `ccl_fill`'s union-find phases and in the lattice-1
fill. Two experiments located it.

Width: the same 12288 threads, `ccl_fill` `union_merge` on
`two_disks_r1400`. Numba takes 30-36 ms at every block size; the lockstep
twin grows with the program width.

| tpb x programs | 32 x 384 | 64 x 192 | 128 x 96 | 256 x 48 |
|---|---|---|---|---|
| Numba | 33.8 ms | 35.9 ms | 34.1 ms | 30.4 ms |
| twin, lockstep | 104 ms | 159 ms | 347 ms | 556 ms |

Counts: copies of both merge kernels that count every parent load, and
Numba's warp issues through `activemask`. `two_disks_r1400`, 24
programs x 256 (the counting kernels' capacity):

| | Numba | twin, lockstep |
|---|---|---|
| merge: parent loads | 0.34 G | 3.11 G |
| merge: warp-steps issued | 23.3 M | 364 M |
| flatten: warp-steps issued | 3.3 M | 59.0 M |
| forest depth after the merge (mean / max) | 4.1 / 44 | 41 / 147 |

So the lockstep loops cost twice. First, the wasted slots: 2.7x the
warp-steps that a per-warp schedule of the same run issues. Second, a
forest ten times deeper. A program advances at the pace of its slowest
lane, programs drift apart, and pixels link into regions not merged yet,
so every later find is longer (on `two_sq_2800`, depth 191 vs 9.7). At
tpb 32 the slots are a warp's, yet the forest still reaches depth 45 vs
Numba's 7: the depth is the larger cost.

### The lane-independent schedule

`"independent"`, the default, makes each lane a small state machine with
its own registers: its grid-stride item (the same item to (program,
lane) map as Numba), its place inside that item, and its union in flight
(a, b, the cursor, which find). Per item the operations are Numba's: the
same read-only finds, the same atomic_min of the smaller root into the
larger root's slot, the same retry from the value the atomic returned,
the same counters.

A Triton step issues every state's code on every lane, so a step that
does everything costs about 193 SASS instructions. That version was
issue-bound: 85 ms for the `two_disks` merge. The cheap, frequent work
therefore runs in mini-steps: one parent hop per lane, the switch from
find(a) to find(b), the next pair of an item, the scan of the next pixel.
The rare, costly work (the link atomic, the pixel reads of a found item,
the fill's probe and enqueue) waits for a full step after every 8
mini-steps. That brought the merge to 34 ms (Numba 31) and the mean depth
to 3-8.

Where it is used:

- `ccl_fill`'s merge and flatten, and the lattice builds' compress: always.
- `seed_merge`'s fill (unions at CAS losses, between direction probes):
  per program and level. A level starts with per-direction batches, the
  lockstep body. After a batch with at least 3 colliding probes per lane,
  while at least 2 slots per lane remain, the program runs the rest of
  the level lane-independently. Sparse levels gain nothing (lanes mostly
  probe, and a per-lane probe costs more than the static direction loop);
  dense ones (lattice 1: every probe collides, lattice 4: 35%) do. In
  that body a lane takes its next slot only while it is fewer than 16
  slots ahead of the program's slowest lane. Without that window,
  free-running lanes split into a fast group and a stuck tail (38% of
  lane-steps idle; `two_sq_2800` lattice 1: 387 ms, 121 ms with it).
- `seed_merge`'s final flatten keeps the lockstep find in both schedules.
  Neighbouring pixels share a provisional label, so lockstep chases walk
  the same chain together and their loads broadcast; per-lane chases lose
  that (`comb_2000`: 52 ms lockstep, 97 ms per lane, Numba 26).

### Before / after

The `lane_schedule` experiment of `compare.py` through the harness (4
interleaved rounds, the same grid for all three), kernel_ms medians, x =
numba_ms / twin_ms:

| case | blocks | Numba | lockstep | independent |
|---|---|---|---|---|
| ccl `asym_4000_800` | 48 | 83.9 ms | 618 ms (x0.13) | 129 ms (x0.65) |
| ccl `two_disks_r1400` | 48 | 53.9 ms | 493 ms (x0.11) | 67.0 ms (x0.80) |
| ccl `two_sq_2800` | 48 | 106 ms | 622 ms (x0.16) | 112 ms (x0.94) |
| ccl `blob_grid_100` | 48 | 46.8 ms | 216 ms (x0.22) | 65.2 ms (x0.72) |
| cclp `asym_4000_800` | 120 | 41.6 ms | 444 ms (x0.09) | 53.7 ms (x0.78) |
| cclp `two_disks_r1400` | 120 | 21.8 ms | 252 ms (x0.09) | 36.0 ms (x0.61) |
| cclp `two_sq_2800` | 120 | 48.1 ms | 291 ms (x0.17) | 34.2 ms (x1.41) |
| cclp `blob_grid_100` | 120 | 27.8 ms | 140 ms (x0.20) | 30.5 ms (x0.91) |
| S1 `two_disks_r1400` | 24 | 59.4 ms | 248 ms (x0.24) | 101 ms (x0.59) |
| S1 `two_sq_2800` | 24 | 81.1 ms | 287 ms (x0.28) | 122 ms (x0.66) |
| S1 `blob_grid_100` | 24 | 67.3 ms | 233 ms (x0.29) | 116 ms (x0.58) |
| S4 `two_disks_r1400` | 24 | 48.5 ms | 120 ms (x0.41) | 68.4 ms (x0.71) |
| S4 `two_sq_2800` | 24 | 60.4 ms | 161 ms (x0.37) | 81.1 ms (x0.74) |
| S4 `blob_grid_100` | 24 | 49.1 ms | 108 ms (x0.45) | 67.2 ms (x0.73) |

Geometric mean over the 14 cells: x0.20 lockstep, x0.75 independent (the
twin 3.7x faster). Every output was equal on every run. Interleaved
development runs of the two schedules on the other benchmark, seeding
and tuning configurations (the independent schedule against the
lockstep one, or against Numba where stated):

- `ccl_bare` on the solid scenes, against Numba: x0.14-0.21 became
  x0.67-1.10.
- `r128_L1` / `split_L1`: 1.7-2.2x faster on `two_disks` and `blob_grid`,
  1.2-1.3x on `two_sq`.
- The lattice-8 builds: 1.17-1.34x faster on `two_disks` and `two_sq`,
  0.84-1.06x on blob grid, random and comb.
- `seed_merge` v1, its bare twin, lattice 0, 64 and 256: within 10% of the
  lockstep schedule, most within 5%.

### What it does not recover

- `ccl_fill` on `asym_4000_800` stays at x0.74. SIMT warps keep 32
  neighbouring pixels on the same item, so their chases share cache lines
  (Numba's merge sustains 17 G parent loads/s there). Free lanes drift
  apart and reach about 8 G/s. Coupling the lanes (a window on the item
  counter) brings the loads together but deepens the forest again; it was
  slower at every window tried.
- Lattice 16 on `two_sq` and `blob_grid`, and lattice 8 on `blob_grid`
  and `comb`, lose 10-17% against the lockstep schedule. Waves meet in
  dense fronts there, some programs switch, and the rest of their level
  is sparse. Switching back after a fixed span cost lattice 1 and 4 more
  than it saved; a switch on the lockstep unions' round count did no
  better.
- `comb_2000` at lattice 1: thin teeth and short chains leave lockstep
  little to wait for. The lockstep twin takes 11-12 ms, the
  lane-independent one 14 ms; both beat Numba's 18-27 ms.

## Run

Tests (needs the GPU; the Numba suite runs separately):

    .venv/bin/python -m pytest -p no:cacheprovider \
        src/flood_fill_cuda/triton_twins/chapters/ch05_gpu_nblob_nblock

`test_cross_backend_*` runs the Numba driver and the twin on the same
images and asserts exact equality of img, visited, depth, label, n_blobs,
seeds, filled, levels, the level trace and the deterministic counters
(candidates, union_done, processed, peak_level, peak_occupancy, seed_merge
family cas_attempts, ccl_fill union_attempts), and with pinned blocks and
tpb also `blocks`, `processed_per_block` and `thread_util_pct`. It covers
both variants and both bare twins on eight scenes, the discovery-only
phase kernels, and every lattice build on five scenes with a sparse set
of 11 (stride, rule, build) configs, not the full cross product: fused at
strides 0, 1, 5, 32 plain and 8 interior; bare at 8 plain and 1
interior; r128 at 1 plain and 8 interior; split at 16 plain and 1
interior. Each build also runs pinned at stride 8 on three grids, and at
`blocks=None` (each backend's own capacity) with equal outputs. With
`lattice=1` (no interior) even `prov_label` is deterministic, and is
compared exactly.

These tests run the default lane schedule. `test_lane_schedules_*` and
`test_cross_backend_both_lane_schedules` run both schedules on nine
configurations (both variants, both bare twins, lattice 1 and 4, every
build) and five scenes. They assert the same equality between the two
schedules and between each schedule and Numba. Small grids (32 x 2 and
64 x 1) make every lane walk many items, so the lane bodies, the fill's
switch and its window all run.

Comparison (writes `results/triton_twins/ch05_gpu_nblob_nblock/compare_<UTC>.json`;
never commit it):

    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch05_gpu_nblob_nblock.compare
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch05_gpu_nblob_nblock.compare --quick

| experiment | mirrors | cases (default) |
|---|---|---|
| `benchmark` | `benchmark.py`: merge, ccl, both bare twins, the scan / cclp probes, all 7 scenes, each pinned to min(Numba, Triton) capacity | 42 |
| `benchmark_blocks_none` | the same launches at `blocks=None` for the runners whose capacity differs (ccl, merge_bare, ccl_bare) | 18 |
| `seeding` | `seeding.py`: strides 0, 1, 4, 16, 64, 256, fused build, pinned (Numba's 24-block grid) | 36 |
| `tuning` | `tuning.py`: fused L8 / I8, r128 and split L1 / L8 / I8, `blocks=None` | 48 |
| `png` | `png_inputs.py`: v1, ccl and the tuning subset on both input PNGs | 20 |
| `lane_schedule` | the twin's two union-find schedules against the same Numba kernel, two rows per cell (`config.lane_sched`): ccl and cclp on the 4 solid scenes, lattice 1 and 4 on 3 of them, pinned to the min of the Numba and both twin capacities | 28 |

Caps, recorded in the JSON's `meta.caps`: 8 of tuning.py's 53 configs;
`asym_4000_800` (the largest scene) only in `benchmark`;
`input_blobs.png` cropped to its top-left 4500 x 4500 quadrant. Rows
whose `blocks=None` grids differ per backend (all of
`benchmark_blocks_none`, the fused-lattice rows of `tuning` and `png`,
and the ccl rows of `png`) are `comparable=false`. The interior rule's
like-for-like rows are r128 and split L8 vs I8 (same grid on both
backends). `meta.builds_info` is tuning.py's register story on both
backends at tpb 256 and 128. `union_thread_ms` in each row's runs is
not comparable across backends (emulated, see the mapping table). Read
`speedup_kernel`; `speedup_total` also compares the host allocators.
The default run is about 15 minutes of GPU time (4 repeats, even so
each backend goes first equally often) and peaks at about 2 GB of host
RAM, plus about 5 minutes for `lane_schedule`. Its `lockstep` rows are
the first translation: label `first_translation`,
`first_translation=true`, `comparable=false`, so they stay out of every
average and get the summary's paired first-translation figures (as in
ch01-ch04). Its `lane_independent` rows repeat benchmark and seeding
cells and carry `duplicate_of`. `--quick` runs every experiment on tiny
scenes with one repeat and writes nothing. `--experiments a,b` runs a
subset.
