# Scan-based blob discovery in Triton (early attempt)

The Triton twin of [`experiments/scan_multi_blob/scan_only.py`](../../../experiments/scan_multi_blob/scan_only.py).
Both Numba kernels are twinned:

- **`scan_image_small_example`** (launched by the Numba host): one block of
  100 threads, one 10x10 patch per thread, each patch walked along its
  anti-diagonals. Each step takes a tick from a block-wide clock and paints
  the pixel into the RGB `visited` image: the thread's golden-ratio hue at
  the tick's brightness, or magenta for red. The first thread to find red
  wins a CAS on `found_flag[0]`, records `(x, y)` and raises a stop flag;
  every thread stops once it sees it.
- **`simple_scan_kernel`** (defined, never launched by the Numba host): the
  same scan, row by row, no early stop. `found_flag[0]` by atomic max, `x`
  and `y` by plain racy stores.

`scan_only.py` keeps the module's names: the two kernels, the helpers,
`setup_scene_small_example()`, `process_image_small_example(...)` (same
print, same return) and `main_small_example()`. The scene builder is the
Numba one (imported), with an optional `rng_seed` (run under
`random.seed`, global random state restored). `run_scan(..., kernel="scan"
| "simple")` is new: one launch with h2d / kernel / d2h / total timings and
the final tick count, for the tests and the compare.

## What can be tested

The output is nondeterministic by construction: the tick order, the race to
the first find, the stop flag read while it is written. So no pixel-exact
comparison exists, on either backend. The tests check what every correct
run satisfies, on both backends:

- `found_flag[0]` is 1 exactly when the image has red; the found pixel is
  red, it is the first red pixel of its patch in scan order, and the
  finder's scan ends on it;
- each patch's painted pixels are a prefix of its scan order, and the
  brightness never decreases along it;
- every painted pixel is one of the exact colors its owner thread can
  paint: its hue (red pixels: magenta) at the brightness of some tick, with
  the kernels' float64 arithmetic and truncations;
- `simple_scan_kernel` paints every pixel and ticks exactly 10,000 times;
- after the stop no thread keeps iterating to the end of its scan (the
  `COUNT_ITERS` build counts loop iterations: 190 only on the white image).

Across backends, the same scene gives the same found flag, the same found
pixel when only one thread can find it, and the same coverage on full scans.

## Mapping

| Numba construct | Triton construct | Fidelity | Note |
|---|---|---|---|
| `kernel[1, 100]` | `kernel[(1,)](..., N_THREADS=100, BLOCK=128, num_warps=4)` | close | 100 is not a power of 2: 128 lanes, lanes >= 100 masked off. 4 warps, as Numba's 100 threads occupy. |
| `cuda.shared.array(1)` `stop_scanning`, `clock` | `state`: int32[2] global scratch, initialized by the kernel | emulated | No user-addressable shared memory in Triton. |
| `cuda.atomic.add(clock, 0, 1)` per step | per-lane `tl.atomic_add(state + CLOCK, 1, sem="relaxed")` | close | One atomic per active lane per step, as in Numba (L2 instead of shared memory). |
| `if stop_scanning[0] == 1: break` (inner), then again after each diagonal (outer) | scalar volatile load of the flag at the same two places, folded into the conditions of two `while` loops | close | Triton has no `break`. Every thread reads the same flag word (as every Numba thread reads the same shared word), and a thread that has read 1 leaves the inner loop, then the outer one, where Numba's break-then-break leaves it. So the program stops iterating at the stop, as Numba's threads do: about 10 us on the GPU on both backends when the stop comes at the first step. The tests pin this with the `COUNT_ITERS` build (not compiled into the default kernel), which counts each thread's loop iterations. |
| (loop structure) `for sum_idx` / `for i` | `while sum_idx < 19 and stop != 1` / `while i < 10 and stop != 1` | close | Same iteration order and the same steps; only the exit is folded into the condition. |
| `cuda.syncthreads()` (scan: after init and at exit; simple: after init) | `cta_sync()` | exact | Same places. The compiled PTX has exactly 2 `bar.sync` (scan) and 1 (simple), and no shared memory. |
| `cuda.atomic.cas(found_flag, 0, 0, 1)` | masked `tl.atomic_xchg(found_flag, 1, sem="relaxed")`, `old == 0` wins | close | `tl.atomic_cas` has no mask in Triton 3.7.1; same exactly-once claim on a 0/1 flag. |
| `cuda.atomic.max(found_flag, 0, 1)`, racy `found_flag[1:3]` stores | masked `tl.atomic_max`, masked `tl.store` | exact | Racy on both. |
| per-thread hue (`h_i` if/elif chain), brightness, `min(255, int(...))` | float64 lane tensors, `tl.where` chain, `.to(int32)` truncation | exact | Same float64 constants and truncations. |
| `for _ in range(5): pass` delay | (nothing) | exact | Compiles to nothing in Numba too. |
| unchecked `img[x, y]` indexing | accesses masked with `x < width`, `y < height` | close | Never fires on the 100x100 image the host uses; avoids out-of-bounds access on other sizes. |
| `cuda.to_device`, `copy_to_host`, `cuda.synchronize` | `cp.asarray(np.ascontiguousarray(...))`, `.get`, `runtime.sync()` | close | The kernels index raw C-order buffers, so Fortran-ordered or strided inputs are made contiguous first (Numba follows their strides). |
| `./images/...` PNGs, `plt.show()` | `results/triton_twins/scan_multi_blob/` PNGs, `show=False` by default | close | The repo's results layout; the window is opt-in. |

## Deviations

- The Numba folder's README describes a nested launch of `flood_fill[1, 256]`
  from inside a `process_image_kernel`. Neither exists in `scan_only.py`;
  the twin follows the code, not that README.
- The 100-thread block is a 128-lane program with 28 idle lanes, not a
  100-thread block. The warp count (4) is the same.
- The stop flag is read with volatile loads at the two places Numba reads
  it. Numba's reads are plain shared loads, which the compiler may keep or
  cache; either way the outcome is schedule-dependent.

## Run

```bash
# Tests: the invariants on seeded, spot and white scenes, both kernels,
# plus test_cross_backend_* against the Numba kernels
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/experiments/scan_multi_blob/test_correctness.py -v

# Numba vs Triton (timing only, 100x100 image; small_example stops after
# one step, so its kernel_ms is mostly launch overhead: read info.device_us);
# writes results/triton_twins/scan_multi_blob/compare_<UTC>.json
.venv/bin/python -m flood_fill_cuda.triton_twins.experiments.scan_multi_blob.compare
.venv/bin/python -m flood_fill_cuda.triton_twins.experiments.scan_multi_blob.compare --quick  # smoke test

# The experiment itself, on Triton (PNGs under results/triton_twins/scan_multi_blob/)
.venv/bin/python -m flood_fill_cuda.triton_twins.experiments.scan_multi_blob.scan_only
```
