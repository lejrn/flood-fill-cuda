# Chapter 1 in Triton: single-block BFS flood fill

The Triton twin of [`chapters/ch01_gpu_1blob_1block`](../../../chapters/ch01_gpu_1blob_1block/).
One Triton program plays the one CUDA block and runs the whole
level-synchronous, 4-connected BFS. Both kernels are twinned:

- **v1 `"ring"`** (`single_block_bfs_kernel`): the 8192-slot ring with
  never-wrapped virtual tickets, one ticket atomic per winning lane, and the
  overflow tripwire that makes the host raise `RuntimeError`.
- **v2 `"spill"`** (`single_block_bfs_spill_kernel`): the same ring as the
  fast path plus a `width*height` global spill tier, with an aggregated
  two-tier enqueue and the rear clamp. It never aborts.

`flood_fill.py` keeps the Numba driver's API: `flood_fill(img, seed_x,
seed_y, threads_per_block=256, variant="ring")`, the same validation
messages, the same alloc / H2D / kernel / D2H / total timing brackets, and
the Numba driver's own `FloodFillResult` (imported, so every field name and
meaning is shared). The counter slots, `RING_CAPACITY` and
`LEVEL_TRACE_CAPACITY` are imported from the Numba chapter too. The CPU
oracle and the scenes are the chapter's own.

## Mapping

One Numba block of T threads = one program with T-lane tensors and
`num_warps = T // 32`; lane i plays thread i.

| Numba construct | Triton construct | Fidelity | Note |
|---|---|---|---|
| `kernel[1, tpb]` | `kernel[(1,)](..., BLOCK=tpb, num_warps=tpb // 32)` | exact | Only for power-of-2 tpb. 96, 768, ... raise `ValueError` naming the power-of-2 rule. |
| block-stride `for i in range(front + tid, rear, nthreads)` | `for base in range(front, rear, BLOCK)`, `i = base + tl.arange(0, BLOCK)` | exact | Same lane-to-entry assignment. |
| `cuda.shared.array(8192, int32)` ring | `ring`: 8192 int32 of global scratch, allocated per call (alloc phase) | emulated | No user-addressable shared memory in Triton. Same capacity, `ticket & 8191` slots, `ticket - front < 8192` window, tripwire. The ring is served by L1/L2, not shared memory. |
| shared scalars `s_rear`, `s_overflow`, `s_spill_rear` | `state`: int32[4] global scratch | emulated | Updated by the same atomics (at L2), re-read after the barrier with volatile loads. |
| `cuda.syncthreads()` | `cta_sync()` (`bar.sync 0`) | exact | Same places: 1 after init, then 2 per level (v1) or 3 per level (v2). A CTA barrier orders global memory within the block. v2's `tl.cumsum`/`tl.sum`/scalar atomic add Triton-internal barriers. |
| `cuda.const.array_like(DX/DY)`, `for d in range(4)` | constexpr tuples, `tl.static_range(4)` | exact | Same order: right, down, left, up. |
| `_is_red`: `img[x,y,0]==255 and ...` | `_is_red`: chained masked uint8 loads | exact | Short-circuits like the `and` chain. Offsets are `pixel.to(int64) * 3 + c`. |
| `cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0` | masked `tl.atomic_xchg(visited + nidx, 1, sem="relaxed")`, `old == 0` wins | close | `tl.atomic_cas` has no mask in Triton 3.7.1. On a 0/1 flag the exchange is the same exactly-once claim, with no traffic from inactive lanes. |
| v1: `cuda.atomic.add(s_rear, 0, 1)` per winning lane | `tl.atomic_add(state + REAR + offs * 0, 1, mask=won, sem="relaxed")` | close | One atomic per winning lane, as in Numba (deliberately not aggregated). L2 atomics instead of shared-memory atomics. |
| v2: `_warp_enqueue_two_tier` (`activemask`, `popc`, `lanemask_lt`, `ffs`, `shfl_sync`) | `_program_enqueue_two_tier`: `tl.cumsum` rank, `tl.sum` count, one scalar `tl.atomic_add` per tier | emulated | No warp intrinsics in Triton: aggregated per program (per chunk and direction) instead of per warp. The spill rank is `rank - k` (k = slab tickets that fit the ring), the value Numba's second ballot computes, since the spilled lanes are the top of the slab. |
| ring store, `s_overflow[0] = 1`, `spill[gbase + rank2] = item` | masked `tl.store` | exact | A failing v1 enqueue writes nothing. |
| `while front < rear: ... level += 1; if overflowed: break` | `while (front < rear) & (overflowed == 0)`, `level += 1` before the exit | exact | Triton has no `break`. LEVELS matches on abort. |
| `if tid == 0:` scalar writes (seed, trace, rear clamp, counters) | scalar `tl.store`, guarded by scalar `if` | exact | |
| per-thread `my_processed`, `my_cas_attempts`; per-thread `atomic.add` at exit | `[BLOCK]` int64 lane tensors; per-lane `tl.atomic_add` at exit | exact | Per-lane registers, no per-level reductions. |
| uniform per-level scalars (front, rear, peaks, active sums) | program scalars | exact | |
| int64 `counters`, slots 0..10 | int64 `counters`, same slots (imported) | exact | |
| `device_array_like`, `copy_to_device`, `copy_to_host`, `cuda.synchronize` | `cp.empty`, `.set`, `.get`, `runtime.sync()` | close | CuPy's pool serves repeat allocations, so `alloc_ms` is not comparable with Numba's fresh allocations. |
| `device.MAX_THREADS_PER_MULTIPROCESSOR`, `MULTIPROCESSOR_COUNT` | `runtime.device_info()` | exact | 1536 and 24 on the RTX 4060 Laptop. |
| lazy JIT, `_warmup(variant)` | `do_not_specialize` on every runtime int; `_warmup(variant, tpb)` | close | Compiles are per (kernel, BLOCK, num_warps); one warm-up per pair keeps every compile out of `kernel_ms` (tested). |

## Deviations

- **Shared memory.** The ring and its scalars live in global scratch. So
  v1's "shared memory is close" story does not carry over, and the 8192
  capacity is kept only for parity (same trips, same spill counts).
- **Power-of-2 block sizes only.** ch02's benchmark calls the ch01 spill
  baseline at 768 threads; the twin refuses that size.
- **v2 aggregation granularity** is the program (one atomic per tier per
  chunk and direction), not the warp. The tickets, slots and spill ranks are
  the same values.
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
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch01_gpu_1blob_1block/test_correctness.py -v

# Numba vs Triton on the chapter's benchmark scenes, both kernels, and the
# tpb sweep; writes results/triton_twins/ch01_gpu_1blob_1block/compare_<UTC>.json
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch01_gpu_1blob_1block.compare --quick  # smoke test
```

The compare runs at full scene size (up to the 36M px `sq_6000_center`).
On the 4 scenes where v1 trips, the `ring_tripwire` rows time the whole
call up to the `RuntimeError` on both backends and check that both report
the same occupancy.
