"""
Triton twin of the ch00 single-block BFS prototype
(chapters/ch00_cpu_baseline/single_block.py).

The prototype is the historical Stage-3 kernel that Chapter 1 replaced. The
twin keeps its algorithm, flaws included, so both backends run the same
thing:

- 8-connectivity, with the prototype's own DX/DY order (imported);
- a non-wrapping 6000-slot queue (queue_x, queue_y): an enqueue whose ticket
  is >= 6000 is silently dropped, while the rear still advances;
- the visited claim (compare-and-swap 0 -> 1) runs BEFORE the red test, so
  every in-bounds 8-neighbor of the blob ends up visited, red or not;
- a filled pixel gets R, G = new_color[0], new_color[1] and a debug blue
  channel, (tid * 4) % 255;
- each level is split into contiguous items_per_thread chunks, and the rear
  is re-read while other threads enqueue (the level-mixing race);
- one block of 64 threads (setup_scene's threads_per_block).

What changes, and why
---------------------
One Triton program plays the one CUDA block: tensors of BLOCK lanes,
num_warps = BLOCK // 32, lane i plays thread i (64 threads = num_warps 2).

Triton has no user-addressable shared memory. The queue and its scalars
live in a small global scratch: `queue` (int32, queue_x in [0, 6000),
queue_y in [6000, 12000)) and `state` (front, rear and the value the Numba
kernel prints). Same capacity, same drop rule. The scalars are re-read with
volatile loads, the twin of a shared-memory read, at the same places.

cuda.syncthreads() becomes cta_sync() (bar.sync 0), at the same places: once
after the init, twice per level.

The visited claim is a masked atomic exchange to 1: Triton 3.7.1's
atomic_cas has no mask, and on a 0/1 flag "old == 0 wins" is the same
exactly-once claim as cas(0, 1).

The enqueue keeps one rear atomic per winning lane, as in Numba (at L2
instead of in shared memory).

The device print of queue_front at exit is dropped. Thread 0 stores the
same value in state[PRINTED] instead, and the host reads it.

Numba reads queue slots past 6000 when the rear has run past the capacity
(out-of-bounds shared memory: undefined behaviour). The twin masks those
reads, so an overflowing scene drops exactly the entries Numba never
stored. Below the capacity nothing differs.

Host side: setup_scene and profile_kernel keep their names, arguments and
printed statistics. setup_scene gets an optional rng_seed (the Numba scene
builder is reused, seeded). run_flood_fill is added for the tests and the
compare: one scene, with an h2d / kernel / d2h / total decomposition whose
kernel bracket is profile_kernel's (launch + synchronize). It also takes a
CuPy new_color, which stays on the device (no implicit round trip). Host
arrays are made C-contiguous before the upload: the kernel indexes raw
C-order buffers, while Numba follows a Fortran-ordered array's strides.
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import random
import time
import timeit
from dataclasses import dataclass

import cupy as cp
import numpy as np
import triton
import triton.language as tl

from flood_fill_cuda.chapters.ch00_cpu_baseline import single_block as _numba_proto
from flood_fill_cuda.triton_twins.runtime import sync, t
from flood_fill_cuda.triton_twins.runtime.device import cta_sync

# The prototype's queue: cuda.shared.array(shape=6000) and "if pos < 6000".
QUEUE_CAPACITY = 6000

# Slots of the per-call `state` scratch, the twin of the shared scalars.
FRONT = 0     # queue_front[0]
REAR = 1      # queue_rear[0]
PRINTED = 2   # the value the Numba kernel prints at exit (queue_front[0])
STATE_SLOTS = 4

# The block size setup_scene hands to the launch (threads_per_block = 64).
THREADS_PER_BLOCK = 64

# 8-connectivity offsets, in the prototype's order.
DX_host = _numba_proto.DX_host
DY_host = _numba_proto.DY_host

# Triton kernels may only read module globals that are constexpr.
_QCAP = tl.constexpr(QUEUE_CAPACITY)
_FRONT = tl.constexpr(FRONT)
_REAR = tl.constexpr(REAR)
_PRINTED = tl.constexpr(PRINTED)
_DX = tl.constexpr(tuple(int(v) for v in DX_host))
_DY = tl.constexpr(tuple(int(v) for v in DY_host))

# Every runtime int that varies between calls: no recompiles per scene.
_RUNTIME_INTS = ["start_x", "start_y", "width", "height"]


# Kernel helper functions
@triton.jit
def is_red(img_ptr, x, y, height, mask):
    """Lanes in mask whose pixel is (255, 0, 0); short-circuits like Numba's
    `and` chain (channel 1 is read only where channel 0 is 255, ...)."""
    base = (x.to(tl.int64) * height + y) * 3
    r = tl.load(img_ptr + base, mask=mask, other=0).to(tl.int32)
    m1 = mask & (r == 255)
    g = tl.load(img_ptr + base + 1, mask=m1, other=1).to(tl.int32)
    m2 = m1 & (g == 0)
    b = tl.load(img_ptr + base + 2, mask=m2, other=1).to(tl.int32)
    return m2 & (b == 0)


@triton.jit
def is_white(img_ptr, x, y, height, mask):
    """Unused by the kernel, as in the prototype."""
    base = (x.to(tl.int64) * height + y) * 3
    r = tl.load(img_ptr + base, mask=mask, other=0).to(tl.int32)
    m1 = mask & (r == 255)
    g = tl.load(img_ptr + base + 1, mask=m1, other=0).to(tl.int32)
    m2 = m1 & (g == 255)
    b = tl.load(img_ptr + base + 2, mask=m2, other=0).to(tl.int32)
    return m2 & (b == 255)


@triton.jit
def is_not_visited(visited_ptr, x, y, height, mask):
    v = tl.load(visited_ptr + x * height + y, mask=mask, other=1)
    return mask & (v == 0)


@triton.jit
def is_valid_pixel(x, y, width, height):
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def flood_fill(img_ptr, visited_ptr, start_x, start_y, width, height,
               new_color_ptr, queue_ptr, state_ptr, BLOCK: tl.constexpr):
    """BFS from (start_x, start_y), recoloring every reached pixel.

    Launch with grid (1,) and num_warps = BLOCK // 32. queue (int32,
    2 * 6000) and state (int32, STATE_SLOTS) are the twin of the kernel's
    shared memory; the kernel initializes them itself, as in Numba.
    """
    tid = tl.arange(0, BLOCK)
    zero = tid * 0          # every lane addresses the same scalar slot
    # One program: global_tid = bid * block_size + tid = tid.

    # Initialization (global_tid == 0)
    tl.store(queue_ptr, start_x)
    tl.store(queue_ptr + _QCAP, start_y)
    tl.store(state_ptr + _FRONT, 0)
    tl.store(state_ptr + _REAR, 1)
    tl.store(visited_ptr + start_x * height + start_y, 1)
    cta_sync()

    color_r = tl.load(new_color_ptr)
    color_g = tl.load(new_color_ptr + 1)
    debug_b = ((tid * 4) % 255).to(tl.uint8)  # img[x, y, 2] = (tid*4) % 255

    # while the queue is not empty
    front = tl.load(state_ptr + _FRONT, volatile=True)
    rear = tl.load(state_ptr + _REAR, volatile=True)
    while front < rear:
        current_size = (tl.load(state_ptr + _REAR, volatile=True)
                        - tl.load(state_ptr + _FRONT, volatile=True))

        # Items per thread: a contiguous chunk per lane.
        items_per_thread = tl.maximum(1, (current_size + BLOCK - 1) // BLOCK)
        start_idx = (tl.load(state_ptr + _FRONT, volatile=True)
                     + tid * items_per_thread)
        end_idx = tl.minimum(start_idx + items_per_thread,
                             tl.load(state_ptr + _REAR, volatile=True))

        for k in range(0, items_per_thread):
            idx = start_idx + k
            # Slots past the capacity were never stored (Numba reads them
            # out of bounds; the twin skips them).
            m = (idx < end_idx) & (idx < _QCAP)
            x = tl.load(queue_ptr + idx, mask=m, other=0)
            y = tl.load(queue_ptr + _QCAP + idx, mask=m, other=0)
            # Mark the pixel with new_color (blue channel: debug value)
            base = (x.to(tl.int64) * height + y) * 3
            tl.store(img_ptr + base, color_r, mask=m)
            tl.store(img_ptr + base + 1, color_g, mask=m)
            tl.store(img_ptr + base + 2, debug_b, mask=m)

            # Process 8-connected neighbors (including diagonals)
            for i in tl.static_range(8):
                nx = x + _DX[i]
                ny = y + _DY[i]
                valid = m & is_valid_pixel(nx, ny, width, height)
                unvisited = is_not_visited(visited_ptr, nx, ny, height, valid)
                # The claim comes before the red test (prototype behaviour).
                old = tl.atomic_xchg(visited_ptr + nx * height + ny, 1,
                                     mask=unvisited, sem="relaxed")
                won = is_red(img_ptr, nx, ny, height, unvisited) & (old == 0)
                # Add to queue: one rear atomic per winning lane.
                pos = tl.atomic_add(state_ptr + _REAR + zero, 1, mask=won,
                                    sem="relaxed")
                fits = won & (pos < _QCAP)  # past the capacity: dropped
                tl.store(queue_ptr + pos, nx, mask=fits)
                tl.store(queue_ptr + _QCAP + pos, ny, mask=fits)

        cta_sync()
        # Update queue front pointer once all threads complete processing
        # (global_tid == 0, with its own current_size).
        f = tl.load(state_ptr + _FRONT + zero, mask=tid == 0, other=0,
                    volatile=True)
        tl.store(state_ptr + _FRONT + zero, f + current_size, mask=tid == 0)
        cta_sync()
        front = tl.load(state_ptr + _FRONT, volatile=True)
        rear = tl.load(state_ptr + _REAR, volatile=True)

    # Numba: if global_tid == 0: print(queue_front[0])
    tl.store(state_ptr + _PRINTED, front)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_scratch_cache = {}
_warmed_up = {}  # threads_per_block -> CompiledKernel of the warm-up launch


def _scratch():
    """The twin of the kernel's shared memory: queue + scalars. The kernel
    initializes it on every launch, so one allocation serves every call."""
    if "queue" not in _scratch_cache:
        _scratch_cache["queue"] = cp.empty(2 * QUEUE_CAPACITY, dtype=cp.int32)
        _scratch_cache["state"] = cp.zeros(STATE_SLOTS, dtype=cp.int32)
    return _scratch_cache["queue"], _scratch_cache["state"]


def _check_launch(threads_per_block, blocks_per_grid):
    if (threads_per_block & (threads_per_block - 1)
            or not 32 <= threads_per_block <= 1024):
        raise ValueError(
            f"threads_per_block must be a power of 2 in [32, 1024] for the "
            f"Triton twin (num_warps = threads_per_block // 32 and tl.arange "
            f"lengths must be powers of 2), got {threads_per_block}")
    if blocks_per_grid != 1:
        raise ValueError(
            f"blocks_per_grid must be 1: the prototype's queue belongs to one "
            f"block, got {blocks_per_grid}")


def _to_device(a):
    """cuda.to_device's twin for the kernel's (x, y[, c]) layout: the kernel
    indexes the raw buffer as C order, so a Fortran-ordered or strided host
    array is made C-contiguous first (Numba follows the strides instead)."""
    return cp.asarray(np.ascontiguousarray(a))


def _launch(d_img, d_visited, start_x, start_y, width, height, new_color,
            threads_per_block=THREADS_PER_BLOCK, blocks_per_grid=1):
    """flood_fill[blocks_per_grid, threads_per_block](...), the Numba call.

    new_color may be a host array, as in the prototype. Numba then copies it
    to the device before the launch and back after it (its implicit
    host-array transfer); the twin does the same steps (cp.asarray, then
    .get(out=new_color)). The steps match, the cost does not: Numba's
    transfer allocates with cuMemAlloc and copies synchronously, CuPy's
    comes from its pool, so the round trip costs Numba about twice as much
    (see the README). A CuPy new_color stays on the device: no round trip.
    """
    _check_launch(threads_per_block, blocks_per_grid)
    d_queue, d_state = _scratch()
    host_color = isinstance(new_color, np.ndarray)
    d_color = cp.asarray(new_color) if host_color else new_color
    compiled = flood_fill[(1,)](
        t(d_img), t(d_visited), int(start_x), int(start_y), int(width),
        int(height), t(d_color), t(d_queue), t(d_state),
        BLOCK=threads_per_block, num_warps=threads_per_block // 32)
    if host_color:
        d_color.get(out=new_color)
    return compiled


def _warmup(threads_per_block=THREADS_PER_BLOCK):
    """Compile on a tiny scene, so no compile lands inside a timed window."""
    if threads_per_block in _warmed_up:
        return _warmed_up[threads_per_block]
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[4, 4] = (255, 0, 0)
    compiled = _launch(cp.asarray(tiny), cp.zeros((8, 8), dtype=cp.int32),
                       4, 4, 8, 8, np.array([0, 0, 255], dtype=np.uint8),
                       threads_per_block)
    sync()
    _warmed_up[threads_per_block] = compiled
    return compiled


def compiled_kernel(threads_per_block=THREADS_PER_BLOCK):
    """The CompiledKernel for this block size, for runtime.kernel_resources()."""
    return _warmup(threads_per_block)


def setup_scene(rng_seed=None):
    """The prototype's scene: 400x400 white, one random-walk red blob.

    This is the Numba module's own setup_scene (imported, not copied). With
    rng_seed it runs under random.seed(rng_seed) and restores the global
    random state afterwards, so the same seed builds the same scene for
    both backends. Returns (img, visited, start_x, start_y, width, height,
    new_color, threads_per_block, blocks_per_grid), as in Numba.
    """
    if rng_seed is None:
        return _numba_proto.setup_scene()
    state = random.getstate()
    random.seed(rng_seed)
    try:
        return _numba_proto.setup_scene()
    finally:
        random.setstate(state)


@dataclass
class PrototypeRun:
    """One prototype launch on one scene (run_flood_fill's result).

    front is the value the Numba kernel prints at exit (queue_front[0]):
    the number of queue tickets taken, dropped ones included.
    """

    img: np.ndarray        # (width, height, 3) uint8
    visited: np.ndarray    # (width, height) int32, 0/1
    front: int
    threads_per_block: int
    h2d_ms: float          # cuda.to_device(img), cuda.to_device(visited)
    kernel_ms: float       # launch (+ host new_color round trip) + synchronize
    d2h_ms: float          # img and visited back to the host
    total_ms: float


def run_flood_fill(img, visited, start_x, start_y, width, height, new_color,
                   threads_per_block=THREADS_PER_BLOCK, blocks_per_grid=1):
    """One launch on one scene; the arguments are setup_scene's tuple, so
    run_flood_fill(*setup_scene(seed)) works. Inputs are not modified.

    kernel_ms is profile_kernel's bracket: the launch and a synchronize.
    A host new_color (setup_scene's) adds its implicit round trip, as in
    Numba; a CuPy new_color is already on the device and adds nothing.
    """
    _check_launch(threads_per_block, blocks_per_grid)
    _warmup(threads_per_block)
    if not isinstance(new_color, cp.ndarray):
        new_color = np.array(new_color, dtype=np.uint8)  # the launch writes it back
    _, d_state = _scratch()

    t0 = time.perf_counter()
    d_img = _to_device(img)
    d_visited = _to_device(visited)
    sync()
    t1 = time.perf_counter()
    _launch(d_img, d_visited, start_x, start_y, width, height, new_color,
            threads_per_block, blocks_per_grid)
    sync()
    t2 = time.perf_counter()
    img_out = d_img.get()
    visited_out = d_visited.get()
    t3 = time.perf_counter()
    front = int(d_state[PRINTED].get())  # the twin of the device print

    return PrototypeRun(
        img=img_out, visited=visited_out, front=front,
        threads_per_block=threads_per_block,
        h2d_ms=(t1 - t0) * 1000, kernel_ms=(t2 - t1) * 1000,
        d2h_ms=(t3 - t2) * 1000, total_ms=(t3 - t0) * 1000)


def profile_kernel(num_runs=100, explore_configs=False, rng_seed=None):
    """
    Profile the flood fill kernel performance.

    Args:
        num_runs: Number of times to run the kernel for averaging
        explore_configs: unused, as in the prototype
        rng_seed: None draws unseeded scenes, as in the prototype; an int
            seeds the warm-up scene with rng_seed and run i with
            rng_seed + 1 + i

    Returns:
        The processed image and visited array from the last run
    """
    def scene(i):
        return setup_scene(None if rng_seed is None else rng_seed + i)

    # First run for warm-up (compilation)
    img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid = scene(0)
    d_img = _to_device(img)
    d_visited = _to_device(visited)
    _launch(d_img, d_visited, start_x, start_y, width, height, new_color,
            threads_per_block, blocks_per_grid)
    sync()

    # For accurate timing, create fresh data for each run
    run_times = []

    for i in range(num_runs):
        # Generate new scene for each run
        img, visited, start_x, start_y, width, height, new_color, threads_per_block, blocks_per_grid = scene(1 + i)
        d_img = _to_device(img)
        d_visited = _to_device(visited)

        # Time this run
        start_time = timeit.default_timer()
        _launch(d_img, d_visited, start_x, start_y, width, height, new_color,
                threads_per_block, blocks_per_grid)
        sync()
        end_time = timeit.default_timer()

        run_times.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate statistics
    avg_time = sum(run_times) / len(run_times)
    min_time = min(run_times)
    max_time = max(run_times)
    std_dev = (sum((t - avg_time) ** 2 for t in run_times) / len(run_times)) ** 0.5

    # Print results
    print(f"Kernel execution time over {num_runs} runs:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Std Dev: {std_dev:.2f} ms")

    # Return results from the last run
    return d_img.get(), d_visited.get()


if __name__ == '__main__':
    from PIL import Image

    from flood_fill_cuda.shared.results_paths import results_dir

    out = results_dir("triton_twins", "ch00_cpu_baseline")
    img_result, visited_result = profile_kernel()
    Image.fromarray(img_result).save(os.path.join(out, "bfs_only_timeit.png"))
    Image.fromarray(visited_result.astype(np.uint8) * 255).save(
        os.path.join(out, "bfs_only_timeit_visited.png"))
    print(f"images written to {out}")
