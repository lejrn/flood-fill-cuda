# Chapter 6 in Triton: runs, not pixels (`triton_twins/chapters/ch06_gpu_nblob_runs/`)

The Triton twin of [`chapters/ch06_gpu_nblob_runs`](../../../chapters/ch06_gpu_nblob_runs/README.md).
Same pipeline, same run table, same union-find protocol, same counters,
same host API. The chapter README explains the design; this one says
only what changed when it moved to Triton.

## What is twinned

| Numba (`chapters/ch06_gpu_nblob_runs/`) | Triton twin (this folder) |
|---|---|
| `kernels.py`: `pack`, `unpack`, `count`, `row_scan`, `emit`, `merge_rows`, `flatten`, `paint`, `label` | `kernels.py`, same names, same arguments plus explicit sizes |
| `merge_rows` / `flatten` with a runtime `instrumented` flag | one body, `INSTRUMENTED: tl.constexpr`: two compiled variants, the bare one without the counter atomics |
| `recolor.py`: `recolor()`, `RunRecolor` (`pack`, `unpack_to`, `run`, `emit_label_map`), `RunRecolorResult`, `_warmup` | `recolor.py`, same signatures, defaults, errors and result fields |
| `benchmarks/benchmark.py` (six scenes, both contracts, read/write peak probes) | `compare.py` cases `benchmark`, plus Triton twins of the `_read` / `_write` probes |
| `benchmarks/scaling.py` (centered crops of `input_blobs.png`) | `compare.py` cases `scaling` |
| `test_correctness.py` (109 tests) | `test_correctness.py`: the same 109 test names, plus `test_cross_backend_*` and `test_twin_*` |

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
| `_find` with path halving, lane-divergent | `while tl.max(active) > 0` over a lane mask, masked loads, masked halving store | emulated | same operations per lane; divergence unit is the program, not the warp |
| `_union` retry loop, `cuda.atomic.min` | lockstep loop, `tl.atomic_min(mask=..., sem="relaxed")` | emulated | Numba atomics are relaxed; so are these |
| binary search and forward walk with `continue` (merge) | lockstep `while` loops; `continue` becomes a lane mask | emulated | Triton has no `break` / `continue` |
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
  size and `INSTRUMENTED` are compile-time constants. `_warmup(tpb)`
  compiles all of them for one block size (default 256), and `recolor()`
  calls it for the block size it is about to use. Every size argument is
  `do_not_specialize`, so a new image shape never recompiles (tested).
- **Device arrays.** CuPy arrays; a Numba device array is accepted and
  viewed zero-copy. They must be C-contiguous (the kernels use raw
  pointers), and a host `numpy` array raises `TypeError` instead of being
  copied in and out implicitly.
- **`RunRecolor.compiled`.** The twin keeps the `CompiledKernel` of each
  kernel's latest launch, for `runtime.kernel_resources` (registers,
  spills, shared bytes). Numba has no equivalent attribute.
- **Divergence.** Where a Numba warp diverges (bit walk, binary search,
  find, union, span loops), the twin runs the loop until the slowest lane
  of the whole program is done. That is the main performance difference
  between the backends: in development probes on `input_blobs.png` the
  twin's merge took about twice the Numba merge, while its pack and paint
  were faster. The compare script measures this properly.

## Run

Tests (same CPU oracle as the Numba chapter, plus Numba-vs-Triton
equality on every variant):

```
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch06_gpu_nblob_runs
```

Performance comparison (Numba vs Triton on the chapter's own
benchmarks, written to `results/triton_twins/ch06_gpu_nblob_runs/`):

```
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare            # full, ~4-6 min
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch06_gpu_nblob_runs.compare --quick    # smoke test, no JSON
```

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
