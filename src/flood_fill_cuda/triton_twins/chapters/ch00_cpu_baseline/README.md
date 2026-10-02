# ch00 prototype in Triton: the single-block BFS that Chapter 1 replaced

The Triton twin of [`chapters/ch00_cpu_baseline/single_block.py`](../../../chapters/ch00_cpu_baseline/single_block.py),
the historical one-block GPU prototype. `sequential.py` is pure Python (no
Numba, no GPU) and has no twin.

The twin keeps the prototype's algorithm, flaws included:

- 8-connectivity, in the prototype's DX/DY order (imported);
- a non-wrapping 6000-slot queue: tickets past slot 6000 are dropped, while
  the rear still advances;
- the visited claim runs **before** the red test, so the white ring around
  the blob ends up visited too;
- the debug blue channel `(tid * 4) % 255` on every filled pixel;
- a contiguous `items_per_thread` split of each level, with the rear
  re-read while other threads enqueue (the level-mixing race);
- one block of 64 threads.

`single_block.py` keeps the module's names: the `flood_fill` kernel, the
`is_red` / `is_white` / `is_not_visited` / `is_valid_pixel` helpers,
`setup_scene()` and `profile_kernel(num_runs=100, explore_configs=False)`
with the same printed statistics. `setup_scene(rng_seed=None)` is the Numba
builder itself (imported), run under `random.seed(rng_seed)` with the global
random state restored, so one seed gives one scene on both backends.
`run_flood_fill(*setup_scene(seed))` is new: one scene, one launch, a
`PrototypeRun` with h2d / kernel / d2h / total timings, for the tests and
the compare.

## The deterministic contract

The prototype has no tests and its blue channel depends on the schedule.
While the blob fits the queue (oracle fill <= 6000), its code fixes:

- recolored pixels (R, G = new_color[0], new_color[1]) = the 8-connected
  blob = `shared.cpu_oracle.cpu_flood_fill_8`'s visited mask;
- `visited` = that blob grown by one pixel in all 8 directions, clipped to
  the image (claim before red test). It is **not** the oracle's visited;
- the printed `queue_front` = the oracle's fill count;
- every other pixel is unchanged.

Every scene of `setup_scene` fits: its blobs are 1.5k to 3.2k pixels.

## Mapping

One Numba block of T threads = one program with T-lane tensors and
`num_warps = T // 32`; lane i plays thread i.

| Numba construct | Triton construct | Fidelity | Note |
|---|---|---|---|
| `flood_fill[1, 64]` | `flood_fill[(1,)](..., BLOCK=64, num_warps=2)` | exact | Other power-of-2 sizes in [32, 1024] work too; 96 and other non-power-of-2 sizes raise `ValueError` naming the power-of-2 rule. `blocks_per_grid != 1` raises (the queue belongs to one block). |
| `cuda.shared.array(6000)` `queue_x`, `queue_y` | `queue`: int32[12000] global scratch (x at [0, 6000), y at [6000, 12000)) | emulated | No user-addressable shared memory in Triton. Same capacity, same `pos < 6000` drop rule. Allocated once and reused (the kernel initializes its state each launch). |
| shared `queue_front`, `queue_rear` | `state`: int32 global scratch, volatile loads | emulated | Read at the same places as Numba (loop test, `current_size`, `start_idx`, `end_idx`), so the level-mixing race window is the same. |
| `cuda.syncthreads()` | `cta_sync()` | exact | Same places: once after init, twice per level. The compiled PTX has exactly 3 `bar.sync`. |
| `cuda.const.array_like(DX/DY)`, `for i in range(8)` | constexpr tuples, `tl.static_range(8)` | exact | Same 8-direction order. |
| `if global_tid == 0:` init | scalar stores | exact | One program: global_tid = tid. |
| `queue_front[0] += current_size` by thread 0 | load + store masked to lane 0 | exact | Thread 0's own `current_size`, as in Numba. |
| per-thread `for idx in range(start_idx, end_idx)` | `for k in range(items_per_thread)` with lane mask `idx < end_idx` | exact | Same contiguous chunk per lane. |
| `cuda.atomic.cas(visited, (nx, ny), 0, 1)` | masked `tl.atomic_xchg(visited, 1, sem="relaxed")`, `old == 0` wins | close | `tl.atomic_cas` has no mask in Triton 3.7.1; on a 0/1 flag the exchange is the same exactly-once claim. Still before the red test. |
| `cuda.atomic.add(queue_rear, 0, 1)` per winning thread | per-lane masked `tl.atomic_add(state + REAR, 1, sem="relaxed")` | close | One atomic per winning lane, as in Numba (L2 instead of shared memory). |
| `queue_x[pos] = nx` if `pos < 6000` | masked `tl.store` | exact | A dropped enqueue writes nothing. |
| `img[x, y, 2] = (tid*4) % 255` | `((lane * 4) % 255).to(uint8)` | exact | Schedule-dependent across runs on both backends. |
| `print(queue_front[0])` at exit | `state[PRINTED] = front`, read by the host | close | The device print is dropped; the value is the same. |
| host `new_color` passed to the kernel | `cp.asarray` before, `.get(out=new_color)` after the launch | exact | The twin of Numba's implicit host-array round trip, inside `kernel_ms` on both sides. |
| `cuda.to_device`, `copy_to_host`, `cuda.synchronize` | `cp.asarray`, `.get`, `runtime.sync()` | close | |
| unseeded `random` scene | `setup_scene(rng_seed)` | close | The Numba builder, seeded from outside. `None` keeps the unseeded behaviour. |

## Deviations

- **Queue slots past 6000.** Once the rear passes 6000, Numba's level loop
  reads `queue_x[idx]` for `idx >= 6000`: out of bounds of a shared array,
  undefined behaviour that can kill the CUDA context. The twin masks those
  reads, so it processes exactly the 6000 stored entries and stops. The
  tests and the compare never run Numba on such a scene; the twin's
  overflow behaviour has its own tests.
- **The device print** of `queue_front` is replaced by a store the host
  reads. Numba's `kernel_ms` includes one device printf; the twin's
  read-back happens after its timed brackets.
- **Images** from `python -m ...single_block` go to
  `results/triton_twins/ch00_cpu_baseline/` instead of `./images/results/`.

## Run

```bash
# Tests: the contract on seeded and shared scenes, overflow, block sizes,
# plus test_cross_backend_* against the Numba prototype
.venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch00_cpu_baseline/test_correctness.py -v

# Numba vs Triton, profile_kernel's method (100 fresh seeded scenes) plus a
# fixed scene; writes results/triton_twins/ch00_cpu_baseline/compare_<UTC>.json
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch00_cpu_baseline.compare
.venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch00_cpu_baseline.compare --quick  # smoke test
```
