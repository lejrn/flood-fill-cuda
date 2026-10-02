# Chapter 1 in Triton: single-block BFS flood fill

The Triton twin of [`chapters/ch01_gpu_1blob_1block`](../../../chapters/ch01_gpu_1blob_1block/).
One Triton program plays the one CUDA block and runs the whole
level-synchronous, 4-connected BFS. Both kernels are twinned:

- **v1 `"ring"`** (`single_block_bfs_kernel`): the 8192-slot ring with
  never-wrapped virtual tickets, one ticket atomic per winning lane, and the
  overflow tripwire that makes the host raise `RuntimeError`.
- **v2 `"spill"`** (`single_block_bfs_spill_kernel`): the same ring as the
  fast path plus a `width*height` global spill tier, with a per-lane
  two-tier enqueue (warp-aggregated by ptxas, as Numba's is by hand) and
  the rear clamp. It never aborts.

`flood_fill.py` keeps the Numba driver's API: `flood_fill(img, seed_x,
seed_y, threads_per_block=256, variant="ring")`, the same validation
messages, the same alloc / H2D / kernel / D2H / total timing brackets, and
the Numba driver's own `FloodFillResult` (imported, so every field name and
meaning is shared). One Triton-only keyword is added: `enqueue="lane"` (the
default) or `enqueue="program"` (v2's first translation, see
[Enqueue](#enqueue-per-lane-warp-aggregated-by-ptxas)). The counter slots, `RING_CAPACITY` and
`LEVEL_TRACE_CAPACITY` are imported from the Numba chapter too. The CPU
oracle and the scenes are the chapter's own.

## Mapping

One Numba block of T threads = one program with T-lane tensors and
`num_warps = T // 32`; lane i plays thread i.

| Numba construct | Triton construct | Fidelity | Note |
|---|---|---|---|
| `kernel[1, tpb]` | `kernel[(1,)](..., BLOCK=tpb, num_warps=tpb // 32)` | exact | Only for power-of-2 tpb. 96, 768, ... raise `ValueError` naming the power-of-2 rule. That check runs after every check shared with Numba, so a combined bad input raises Numba's error. |
| seeds passed to the kernel as given (Numba types `np.int32`/`np.int64`) | `operator.index(seed)` after validation, then the launch | exact | Triton's launcher cannot specialize NumPy scalars. The seeds already indexed the image, so they are integers. |
| block-stride `for i in range(front + tid, rear, nthreads)` | `for base in range(front, rear, BLOCK)`, `i = base + tl.arange(0, BLOCK)` | exact | Same lane-to-entry assignment. |
| `cuda.shared.array(8192, int32)` ring | `ring`: 8192 int32 of global scratch, allocated per call (alloc phase) | emulated | No user-addressable shared memory in Triton. Same capacity, `ticket & 8191` slots, `ticket - front < 8192` window, tripwire. The ring is served by L1/L2, not shared memory. |
| shared scalars `s_rear`, `s_overflow`, `s_spill_rear` | `state`: int32[4] global scratch | emulated | Updated by the same atomics (at L2), re-read after the barrier with volatile loads. |
| `cuda.syncthreads()` | `cta_sync()` (`bar.sync 0`) | exact | Same places: 1 after init, then 2 per level (v1) or 3 per level (v2). A CTA barrier orders global memory within the block. Both default kernels carry exactly Numba's barriers (3 and 4 `bar.sync`, tested). The first translation's v2 (`enqueue="program"`) had 40. |
| `cuda.const.array_like(DX/DY)`, `for d in range(4)` | constexpr tuples, `tl.static_range(4)` | exact | Same order: right, down, left, up. |
| `_is_red`: `img[x,y,0]==255 and ...` | `_is_red`: chained masked uint8 loads | exact | Short-circuits like the `and` chain. Offsets are `pixel.to(int64) * 3 + c`. |
| `cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0` | masked `tl.atomic_xchg(visited + nidx, 1, sem="relaxed")`, `old == 0` wins | close | `tl.atomic_cas` has no mask in Triton 3.7.1. On a 0/1 flag the exchange is the same exactly-once claim, with no traffic from inactive lanes. |
| v1: `cuda.atomic.add(s_rear, 0, 1)` per winning lane | `tl.atomic_add(state + REAR + offs * 0, 1, mask=won, sem="relaxed")` | close | One atomic per winning lane in the source, as in Numba (deliberately not aggregated). ptxas warp-aggregates both sides (leader `ATOMS` per warp in Numba, leader `ATOMG` per warp in Triton), so the executed atomic counts match. L2 atomics instead of shared-memory atomics. |
| v2: `_warp_enqueue_two_tier` (`activemask`, `popc`, `lanemask_lt`, `ffs`, `shfl_sync`) | `_lane_enqueue_two_tier` (default): masked `tl.atomic_add(rear + offs * 0, 1, mask=won, sem="relaxed", scope="gpu")` per winning lane, then the same on the spill rear for the lanes past the ring window | close (same per-warp SASS idiom) | A per-lane atomic, warp-aggregated by ptxas: `VOTEU.ANY` (active mask), `FLO` (leader), `POPC` (count), one predicated leader `ATOMG.E.ADD`, `SR_LTMASK` + `POPC` (rank), `SHFL.IDX` (base). That is the idiom Numba's hand-written source compiles to (`VOTE.ANY`, `FLO`, `POPC`, leader `ATOMS.ADD`, `SHFL.IDX`), with no CTA barrier (tested). Not identical machine code: Numba aggregates by hand where Triton relies on ptxas, the atomics resolve at L2 (`ATOMG`) instead of shared memory (`ATOMS`), and Triton unrolls the 4 directions into 8 sites where Numba keeps 2 in a loop. The spill atomic is skipped by a warp-uniform branch when no lane spills, like Numba's `else`. |
| (first translation) | `_program_enqueue_two_tier` (`enqueue="program"`): `tl.cumsum` rank, `tl.sum` count, one scalar `tl.atomic_add` per tier | emulated | Aggregated per program instead of per warp, with the same tickets, slots and spill ranks (`rank - k`). The scan, the reduction and the scalar-atomic broadcast cost 9 `bar.sync` per direction and put the 8 warps in lockstep: x0.64-0.70 vs Numba where the lane form is x1.17-1.21 (the chapter's six worst rows). Kept to measure that cost. |
| ring store, `s_overflow[0] = 1`, `spill[gbase + rank2] = item` | masked `tl.store` | exact | A failing v1 enqueue writes nothing. |
| `while front < rear: ... level += 1; if overflowed: break` | `while (front < rear) & (overflowed == 0)`, `level += 1` before the exit | exact | Triton has no `break`. LEVELS matches on abort. |
| `if tid == 0:` scalar writes (seed, trace, rear clamp, counters) | scalar `tl.store`, guarded by scalar `if` | exact | |
| per-thread `my_processed`, `my_cas_attempts`; per-thread `atomic.add` at exit | `[BLOCK]` int64 lane tensors; per-lane `tl.atomic_add` at exit | exact | Per-lane registers, no per-level reductions. |
| uniform per-level scalars (front, rear, peaks, active sums) | program scalars | exact | |
| int64 `counters`, slots 0..10 | int64 `counters`, same slots (imported) | exact | |
| `device_array_like`, `copy_to_device`, `copy_to_host`, `cuda.synchronize` | `cp.empty`, `.set`, `.get`, `runtime.sync()` | close | CuPy's pool serves repeat allocations, and `.set`/`.get` cost less per call than `copy_to_device`/`copy_to_host`. So `alloc_ms`, `h2d_ms`, `d2h_ms`, `total_ms` and `speedup_total` compare host stacks, not backends. |
| `device.MAX_THREADS_PER_MULTIPROCESSOR`, `MULTIPROCESSOR_COUNT` | `runtime.device_info()` | exact | 1536 and 24 on the RTX 4060 Laptop. |
| lazy JIT, `_warmup(variant)` | `do_not_specialize` on every runtime int; `_warmup(variant, tpb, enqueue)` | close | Compiles are per (kernel, BLOCK, num_warps, ENQ); one warm-up per (variant, tpb, enqueue) keeps every compile out of `kernel_ms` (tested for both enqueue forms). |

## Deviations

- **Shared memory.** The ring and its scalars live in global scratch. So
  v1's "shared memory is close" story does not carry over, and the 8192
  capacity is kept only for parity (same trips, same spill counts).
- **Power-of-2 block sizes only.** ch02's benchmark calls the ch01 spill
  baseline at 768 threads; the twin refuses that size.
- **v2 enqueue.** The default is the per-lane form, which ptxas compiles
  to Numba's warp-aggregated machine code. The first translation's
  program-aggregated form stays behind `enqueue="program"` (a constexpr
  `ENQ` in the kernel). v1 has one form only, so `variant="ring"` refuses
  `enqueue="program"`.
- **Overflow message** says `ring overflow:` instead of `shared-memory ring
  overflow:`, since the twin's ring is not in shared memory. The rest of the
  text and the occupancy number are the same.
- **OVERFLOW counter** is stored from the loop variable `overflowed` (the
  value read after the last level's barrier) instead of re-reading the state
  slot. The value is the same. Re-reading it crashes Triton 3.7.1: when the
  while condition is a variable's only use, the canonicalizer drops it from
  the `scf.while` results, and `TritonGPURemoveLayoutConversions` then fails
  with "Result number is out of range".
- `benchmarks/wavefront.py` and `benchmarks/visualize.py` are not twinned.
  The wavefront replays the depth map, which is bit-identical across
  backends (tested). The dashboard is CPU-only.

## Enqueue: per lane, warp-aggregated by ptxas

Numba's v2 enqueue aggregates per warp by hand: `activemask`, `popc`,
`lanemask_lt`, a leader `atomic.add` on the shared rear, `shfl_sync` of
the base. Triton has no warp intrinsics.

The first translation therefore aggregated over the whole program:
`tl.cumsum` for the rank, `tl.sum` for the count, one scalar atomic per
tier. Each of those lowers to a cross-warp exchange through shared memory
with CTA barriers.

The SASS of the compiled kernels (`compiled.asm["cubin"]`, read with
`nvdisasm`, sm_89, Triton 3.7.1's bundled ptxas 12.8) shows the cost:

| kernel | `BAR.SYNC` | enqueue atomics in SASS | cross-warp exchange | regs | shared bytes |
|---|---|---|---|---|---|
| Numba v2 | 4 | `ATOMS.ADD` by the warp leader, per tier (the direction loop is not unrolled) | none | 64 | 32776 (the ring) |
| Triton v2, `enqueue="lane"` | 4 | 8 `ATOMG.E.ADD` (4 directions x 2 tiers), each warp-aggregated by ptxas | none: 0 `STS`/`LDS` | 36 | 0 |
| Triton v2, `enqueue="program"` | 40 | 8 `ATOMG.E.ADD` by one thread of the program | `tl.cumsum` (20 `SHFL.UP`), `tl.sum` (12 `SHFL.BFLY`), 20 `STS` / 24 `LDS` | 40 | 32 |

(Numba's leader branch also contains a `SHFL.UP` scan: ptxas's own
full-warp path for a variable atomic add, never taken because only the
leader lane enters the branch.)

The per-lane form is one masked `tl.atomic_add` per winning lane on the
ring rear and one per spilling lane on the spill rear. ptxas recognizes a
same-address atomic under a lane mask and emits, per warp (the ring-rear
site of the first direction; `[R18.64]` is `state[REAR]`):

```
S2R R21, SR_LANEID ;
VOTEU.ANY UR5, UPT, PT ;                          // active mask
FLO.U32 R20, UR5 ;                                // leader lane
POPC R23, UR5 ;                                   // count
ISETP.EQ.U32.AND P1, PT, R20, R21, PT ;           // lane == leader?
@P1 ATOMG.E.ADD.STRONG.GPU PT, R23, [R18.64], R23 ;  // leader only: rear += count
S2R R21, SR_LTMASK ; LOP3.LUT R22, R21, UR5 ... ; POPC R21, R22 ;  // rank
SHFL.IDX PT, R20, R23, R20, 0x1f ;                // broadcast the base
IMAD.IADD R21, R20, 0x1, R21 ;                    // ticket = base + rank
```

The spill-rear sites (`[R18.64+0x8]`, `state[SPILL_REAR]`) have the same
shape and end in spill slot = base + rank.

That is the sequence Numba's hand-written enqueue compiles to (Numba:
`VOTE.ANY`, `BREV` + `FLO.U32.SH` for `ffs`, `POPC`, leader `ATOMS.ADD`,
`SHFL.IDX`, `SR_LTMASK` + `POPC` for the rank). All 8
enqueue atomics of the default v2 kernel (4 directions x 2 tiers) and all
4 of v1 have this shape, and both default kernels have Numba's barrier
count. `test_lane_enqueue_is_warp_aggregated_like_numba` checks both on
every run.

Semantics are unchanged. Per level, both forms hand out the contiguous
ticket range `[sr, sr + size)`. The first `sf + 8192 - sr` tickets fill the
ring and the rest spill, so `spilled`, `peak_spill_window` and
`peak_occupancy` keep Numba's values. The ring rear overshoots by the
level's spill count exactly as in Numba; those tickets write nothing, and
the level-end clamp retracts them.

Before/after, kernel_ms medians of 6 interleaved rounds (each round runs
Numba, the first translation and the default back to back, in rotating
order), v2 at 256 threads, RTX 4060 Laptop. These are the chapter's six
worst rows of the first comparison run (all v2 at 256 threads; "first
run" is that run's speedup). sq_5000_center and sq_4000_center come from
a second session of the same script.

| scene | first run | spilled | Numba ms | program ms | program vs Numba | lane ms | lane vs Numba | lane vs program |
|---|---|---|---|---|---|---|---|---|
| sq_6000_center | x0.60 | 15,618,302 | 562.8 | 886.2 | x0.64 | 480.3 | x1.17 | 1.84x |
| sq_5000_center | x0.65 | 8,714,302 | 370.6 | 575.4 | x0.64 | 317.9 | x1.17 | 1.81x |
| sq_4000_center | x0.66 | 3,810,302 | 239.9 | 365.2 | x0.66 | 201.6 | x1.19 | 1.81x |
| serpentine_256 | x0.67 | 0 | 52.3 | 82.3 | x0.64 | 43.5 | x1.20 | 1.89x |
| sq_2600_full_center | x0.68 | 304,702 | 106.3 | 157.7 | x0.67 | 87.7 | x1.21 | 1.80x |
| sq_4000_corner | x0.70 | 0 | 242.6 | 346.3 | x0.70 | 208.2 | x1.17 | 1.66x |

Outputs and counters were identical to Numba in every run. The compare's
`enqueue` experiment repeats this in the harness on five scenes, two rows
each (`label: "per_lane"` and `label: "first_translation"`).

## Determinism

Everything the driver returns except the `*_ms` timings is identical across
backends: img, visited, depth, level_sizes, levels, filled, processed,
peak_level, peak_occupancy, spilled, peak_spill_window, the utilization
percentages, and also `cas_attempts`. The 4-connected grid is bipartite, so
during level L a neighbor is either at L-1 (already blue) or at L+1 (still
red): every claim attempt is one edge of the filled component, whatever the
schedule. Spill counts depend only on the BFS layer sizes, since each
level's tickets are the contiguous range `[sr, sr + size)`.

## Run

```bash
# Tests: the Numba chapter's tests (same names) plus test_cross_backend_*
# and test_enqueue_* (both v2 enqueue forms vs the oracle and Numba, also
# at the ring/spill window edge at 32/256/1024 threads; SASS and barrier
# checks)
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch01_gpu_1blob_1block/test_correctness.py -v

# Numba vs Triton on the chapter's benchmark scenes, both kernels, the
# tpb sweep and the enqueue experiment (v2 lane vs program);
# writes results/triton_twins/ch01_gpu_1blob_1block/compare_<UTC>.json
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare --quick  # smoke test
```

The compare runs at full scene size (up to the 36M px `sq_6000_center`).

- **Read `speedup_kernel`.** `total_ms` and `speedup_total` add the host
  libraries' allocation and copy costs (CuPy pool, `.set`/`.get` vs Numba's
  fresh `cuMemAlloc`, `copy_to_device`). On small scenes `speedup_total`
  exceeds `speedup_kernel` and can flip its direction on spill rows. Each
  row's `phases_ms` keeps alloc / H2D / D2H per backend: the cold warm-up
  call and the medians of the timed rounds.
- **`ring_tripwire` rows** (the 4 scenes where v1 trips): both backends
  must raise with the same occupancy. `kernel_ms` is the driver's own kernel
  bracket up to the abort, read from its `perf_counter` stamps, so it means
  the same as in the other rows. `total_ms` runs from the driver's first
  stamp to the `RuntimeError`.
- **`enqueue` rows** run v2 twice per scene: `enqueue="lane"` (label
  `per_lane`, the default) and `enqueue="program"` (label
  `first_translation`). Numba runs its one v2 kernel in both.
  `info.*.bar_sync_in_ptx` records the CTA barriers of each kernel (4, 4
  and 40). The first-translation rows are `comparable=false` and carry
  `first_translation: true`, as in ch02-ch04, so the summary's
  like-for-like averages and extremes leave them out. The `per_lane` rows
  repeat the `scenes` spill rows of the same scenes.
- **`--quick`** uses three small scenes plus the 2600x2600 tripwire scene.
  Overflowing the 8192-slot ring needs a frontier above 8192 pixels, so it
  is the only quick case for the spill tier and the tripwire (about 3 s).
