"""
Triton twin of the scan-based blob discovery prototype
(experiments/scan_multi_blob/scan_only.py).

Both Numba kernels are twinned:

- scan_image_small_example (the one the Numba host launches): one block of
  100 threads, one 10x10 patch per thread, each patch walked along its
  anti-diagonals. Every step takes a tick from a block-wide clock and paints
  the pixel into the RGB `visited` image: the thread's hue (golden-ratio
  HSV) scaled by the tick's brightness, or magenta-ish for a red pixel. The
  first thread to find red wins a CAS on found_flag[0], records (x, y) in
  found_flag[1:3] and raises a stop flag; every thread stops once it sees
  the flag.
- simple_scan_kernel (defined but never launched by the Numba host): the
  same scan, row by row and without the early stop. Red pixels raise
  found_flag[0] with an atomic max and store x and y with plain racy stores.

What changes, and why
---------------------
One Triton program plays the one CUDA block. The block has 100 threads, not
a power of 2, so the program has BLOCK = 128 lanes and lanes >= 100 are
masked off: num_warps = 4, Numba's own warp count for 100 threads. Lane i
plays thread i.

Triton has no user-addressable shared memory. The stop flag and the clock
live in a global scratch `state` (int32: STOP, CLOCK) that the kernel
initializes itself, then cta_sync() (the twin of syncthreads). The clock
tick is a per-lane atomic add on it (one atomic per lane, as in Numba, at L2
instead of in shared memory). The stop flag is read with volatile loads, at
the same two places (after the tick, and after each diagonal).

Triton has no per-lane break. A sticky lane mask `alive` replaces it: a lane
that sees the stop flag is masked off for the rest of the scan, which is
where Numba's break-then-break leaves its thread.

The first-finder CAS on found_flag[0] is a masked atomic exchange to 1
(Triton 3.7.1's atomic_cas has no mask; on a 0/1 flag "old == 0 wins" is the
same exactly-once claim).

The hue and brightness arithmetic runs in float64 with the same constants
and truncations, so a thread paints the same colors for the same ticks.

The delay loop (`for _ in range(5): pass`) compiles to nothing in Numba and
is left out. The kernel also guards its pixel accesses with x < width and
y < height: Numba relies on a 100x100 image for 100 patches and indexes out
of bounds otherwise; on that image the guard never fires.

Host side: setup_scene_small_example, process_image_small_example and
main_small_example keep their names and roles. The scene builder is the
Numba one (imported, not copied), with an optional rng_seed. Images go to
results/triton_twins/scan_multi_blob/ instead of ./images/, and the
matplotlib window is opt-in (show=True). run_scan is added for the tests and
the compare: one launch of either kernel with an h2d / kernel / d2h / total
decomposition.
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

from flood_fill_cuda.experiments.scan_multi_blob import scan_only as _numba_scan
from flood_fill_cuda.triton_twins.runtime import sync, t
from flood_fill_cuda.triton_twins.runtime.device import cta_sync

# The Numba host's launch: scan_image_small_example[1, 100]
THREADS_PER_BLOCK = 100
BLOCK = triton.next_power_of_2(THREADS_PER_BLOCK)   # 128 lanes
NUM_WARPS = BLOCK // 32                             # 4 warps, as in Numba
PATCH_SIZE = 10
MAX_TICKS = 10000   # 100 threads * 100 pixels, the kernels' normalizer

# Slots of the per-call `state` scratch, the twin of the shared arrays.
STOP = 0    # stop_scanning[0]
CLOCK = 1   # clock[0]
STATE_SLOTS = 2

KERNELS = ("scan", "simple")  # scan_image_small_example, simple_scan_kernel

_STOP = tl.constexpr(STOP)
_CLOCK = tl.constexpr(CLOCK)
_PATCH = tl.constexpr(PATCH_SIZE)
_MAX_TICKS = tl.constexpr(float(MAX_TICKS))
_GOLDEN = tl.constexpr(0.618033988749895)

_RUNTIME_INTS = ["width", "height"]


# Kernel Helper functions
@triton.jit
def is_red(img_ptr, x, y, height, mask):
    """Check if a pixel is red (short-circuits like Numba's `and` chain)"""
    base = (x.to(tl.int64) * height + y) * 3
    r = tl.load(img_ptr + base, mask=mask, other=0).to(tl.int32)
    m1 = mask & (r == 255)
    g = tl.load(img_ptr + base + 1, mask=m1, other=1).to(tl.int32)
    m2 = m1 & (g == 0)
    b = tl.load(img_ptr + base + 2, mask=m2, other=1).to(tl.int32)
    return m2 & (b == 0)


@triton.jit
def is_white(img_ptr, x, y, height, mask):
    """Check if a pixel is white (unused by the kernels, as in Numba)"""
    base = (x.to(tl.int64) * height + y) * 3
    r = tl.load(img_ptr + base, mask=mask, other=0).to(tl.int32)
    m1 = mask & (r == 255)
    g = tl.load(img_ptr + base + 1, mask=m1, other=0).to(tl.int32)
    m2 = m1 & (g == 255)
    b = tl.load(img_ptr + base + 2, mask=m2, other=0).to(tl.int32)
    return m2 & (b == 255)


@triton.jit
def is_not_visited(visited_ptr, x, y, height, mask):
    """Check if a pixel has not been visited (all three channels zero)"""
    base = (x.to(tl.int64) * height + y) * 3
    r = tl.load(visited_ptr + base, mask=mask, other=1).to(tl.int32)
    m1 = mask & (r == 0)
    g = tl.load(visited_ptr + base + 1, mask=m1, other=1).to(tl.int32)
    m2 = m1 & (g == 0)
    b = tl.load(visited_ptr + base + 2, mask=m2, other=1).to(tl.int32)
    return m2 & (b == 0)


@triton.jit
def is_valid_pixel(x, y, width, height):
    """Check if a pixel is within the image boundaries"""
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)


@triton.jit
def _thread_base_color(tid):
    """The thread's hue: golden-ratio HSV (s = v = 1) -> RGB, in float64."""
    h = (tid.to(tl.float64) * _GOLDEN) % 1.0
    h_i = (h * 6).to(tl.int32)
    f = h * 6 - h_i
    up = (255 * f).to(tl.int32)          # int(255 * f)
    down = (255 * (1 - f)).to(tl.int32)  # int(255 * (1 - f))
    base_r = tl.where(h_i == 0, 255, tl.where(h_i == 1, down, tl.where(
        h_i == 2, 0, tl.where(h_i == 3, 0, tl.where(h_i == 4, up, 255)))))
    base_g = tl.where(h_i == 0, up, tl.where(h_i == 1, 255, tl.where(
        h_i == 2, 255, tl.where(h_i == 3, down, 0))))
    base_b = tl.where(h_i == 0, 0, tl.where(h_i == 1, 0, tl.where(
        h_i == 2, up, tl.where(h_i == 3, 255, tl.where(h_i == 4, 255, down)))))
    return base_r, base_g, base_b


@triton.jit
def _paint(img_ptr, visited_ptr, x, y, height, tick, base_r, base_g, base_b,
           mask):
    """One scan step on the lanes in mask (pixel not visited yet): paint the
    pixel with the thread's hue at the tick's brightness, or magenta for red.
    Returns the lanes whose pixel is red."""
    # Brightness from the global timeline: dark (20%) to full (100%).
    normalized_tick = tl.minimum(1.0, tick.to(tl.float64) / _MAX_TICKS)
    brightness = 0.2 + 0.8 * normalized_tick
    r_val = tl.minimum(255, (base_r * brightness).to(tl.int32))
    g_val = tl.minimum(255, (base_g * brightness).to(tl.int32))
    b_val = tl.minimum(255, (base_b * brightness).to(tl.int32))
    mag = tl.minimum(255, (255 * brightness).to(tl.int32))

    red = is_red(img_ptr, x, y, height, mask)
    base = (x.to(tl.int64) * height + y) * 3
    tl.store(visited_ptr + base, tl.where(red, mag, r_val).to(tl.uint8),
             mask=mask)
    tl.store(visited_ptr + base + 1, tl.where(red, 0, g_val).to(tl.uint8),
             mask=mask)
    tl.store(visited_ptr + base + 2, tl.where(red, mag, b_val).to(tl.uint8),
             mask=mask)
    return red


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def scan_image_small_example(img_ptr, visited_ptr, width, height,
                             found_flag_ptr, state_ptr,
                             N_THREADS: tl.constexpr, BLOCK: tl.constexpr):
    """
    Each thread scans one 10x10 patch along its anti-diagonals and stops as
    soon as any thread has found a red pixel.

    Launch with grid (1,), num_warps = BLOCK // 32; lanes >= N_THREADS idle.
    state (int32, STATE_SLOTS) is the twin of the shared stop flag and
    clock; the kernel initializes it.
    """
    tid = tl.arange(0, BLOCK)
    zero = tid * 0
    patches_per_row = width // _PATCH
    patch_x = (tid % patches_per_row) * _PATCH
    patch_y = (tid // patches_per_row) * _PATCH

    # Initialize the shared state (tid == 0)
    tl.store(state_ptr + _STOP, 0)
    tl.store(state_ptr + _CLOCK, 0)
    cta_sync()

    base_r, base_g, base_b = _thread_base_color(tid)

    alive = tid < N_THREADS   # Numba's thread still in the scan loops
    for sum_idx in range(2 * _PATCH - 1):  # Process diagonals
        for i in range(_PATCH):
            j = sum_idx - i
            if (j >= 0) & (j < _PATCH):
                x = patch_x + i
                y = patch_y + j
                # Get the current global clock value and increment it
                tick = tl.atomic_add(state_ptr + _CLOCK + zero, 1, mask=alive,
                                     sem="relaxed")
                # Check if we should stop scanning (Numba: break)
                stop = tl.load(state_ptr + _STOP + zero, mask=alive, other=1,
                               volatile=True)
                alive = alive & (stop != 1)

                go = alive & is_valid_pixel(x, y, width, height)
                go = is_not_visited(visited_ptr, x, y, height, go)
                red = _paint(img_ptr, visited_ptr, x, y, height, tick,
                             base_r, base_g, base_b, go)
                # Set found flag and notify other threads to stop
                old = tl.atomic_xchg(found_flag_ptr + zero, 1, mask=red,
                                     sem="relaxed")
                first = red & (old == 0)
                tl.store(found_flag_ptr + 1 + zero, x, mask=first)
                tl.store(found_flag_ptr + 2 + zero, y, mask=first)
                tl.store(state_ptr + _STOP + zero, 1, mask=first)

        # Check again if we should stop scanning (Numba: break)
        stop = tl.load(state_ptr + _STOP + zero, mask=alive, other=1,
                       volatile=True)
        alive = alive & (stop != 1)

    cta_sync()


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def simple_scan_kernel(img_ptr, visited_ptr, width, height, found_flag_ptr,
                       state_ptr, N_THREADS: tl.constexpr,
                       BLOCK: tl.constexpr):
    """
    Each thread scans its 10x10 patch row by row, with no early stop. Same
    launch contract as scan_image_small_example (STOP stays unused).
    """
    tid = tl.arange(0, BLOCK)
    zero = tid * 0
    patches_per_row = width // _PATCH
    patch_x = (tid % patches_per_row) * _PATCH
    patch_y = (tid // patches_per_row) * _PATCH

    # Initialize the shared clock (tid == 0)
    tl.store(state_ptr + _CLOCK, 0)
    cta_sync()

    base_r, base_g, base_b = _thread_base_color(tid)

    on = tid < N_THREADS
    for j in range(_PATCH):
        for i in range(_PATCH):
            x = patch_x + i
            y = patch_y + j
            tick = tl.atomic_add(state_ptr + _CLOCK + zero, 1, mask=on,
                                 sem="relaxed")
            go = on & is_valid_pixel(x, y, width, height)
            go = is_not_visited(visited_ptr, x, y, height, go)
            red = _paint(img_ptr, visited_ptr, x, y, height, tick,
                         base_r, base_g, base_b, go)
            # Set found flag (no early termination): racy x, y stores
            tl.atomic_max(found_flag_ptr + zero, 1, mask=red, sem="relaxed")
            tl.store(found_flag_ptr + 1 + zero, x, mask=red)
            tl.store(found_flag_ptr + 2 + zero, y, mask=red)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_warmed_up = {}  # kernel name -> CompiledKernel of the warm-up launch


def _kernel(name):
    if name not in KERNELS:
        raise ValueError(f'kernel must be "scan" or "simple", got {name!r}')
    return scan_image_small_example if name == "scan" else simple_scan_kernel


def _launch(name, d_img, d_visited, width, height, d_found_flag, d_state):
    """kernel[1, 100](d_img, d_visited, width, height, d_found_flag)."""
    return _kernel(name)[(1,)](
        t(d_img), t(d_visited), int(width), int(height), t(d_found_flag),
        t(d_state), N_THREADS=THREADS_PER_BLOCK, BLOCK=BLOCK,
        num_warps=NUM_WARPS)


def _warmup(name):
    """Compile on a blank scene, so no compile lands inside a timed window."""
    if name in _warmed_up:
        return _warmed_up[name]
    img, visited, width, height, found_flag = blank_scene()
    compiled = _launch(name, cp.asarray(img), cp.asarray(visited), width,
                       height, cp.asarray(found_flag),
                       cp.zeros(STATE_SLOTS, dtype=cp.int32))
    sync()
    _warmed_up[name] = compiled
    return compiled


def blank_scene():
    """setup_scene_small_example's layout with no red square: an all-white
    100x100 image, so the early-stop kernel scans every patch to the end."""
    width, height = 100, 100
    return (np.full((width, height, 3), 255, dtype=np.uint8),
            np.zeros((width, height, 3), dtype=np.uint8), width, height,
            np.zeros(3, dtype=np.int32))


def compiled_kernel(name="scan"):
    """The CompiledKernel of a kernel, for runtime.kernel_resources()."""
    return _warmup(name)


def setup_scene_small_example(rng_seed=None):
    """Set up a smaller 100x100 test scene with a red square.

    This is the Numba module's own builder (imported, not copied). With
    rng_seed it runs under random.seed(rng_seed) and restores the global
    random state afterwards, so the same seed builds the same scene for
    both backends. Returns (img, visited, width, height, found_flag).
    """
    if rng_seed is None:
        return _numba_scan.setup_scene_small_example()
    state = random.getstate()
    random.seed(rng_seed)
    try:
        return _numba_scan.setup_scene_small_example()
    finally:
        random.setstate(state)


def process_image_small_example(img, visited, width, height, found_flag):
    """Process the small example image."""
    _warmup("scan")
    # Copy data to device
    d_img = cp.asarray(img)
    d_visited = cp.asarray(visited)
    d_found_flag = cp.asarray(found_flag)
    d_state = cp.zeros(STATE_SLOTS, dtype=cp.int32)  # the shared arrays' twin

    # Configure thread block - exactly 100 threads for 100 patches (10x10 each)
    threads_per_block = THREADS_PER_BLOCK

    # Launch kernel with a single block
    print(f"Launching with single block of {threads_per_block} threads")
    _launch("scan", d_img, d_visited, width, height, d_found_flag, d_state)
    sync()

    # Get results
    result_visited = d_visited.get()
    result_found = d_found_flag.get()

    return result_visited, result_found


@dataclass
class ScanRun:
    """One launch of either kernel on one scene (run_scan's result).

    clock is the final tick count (the shared clock's twin): the number of
    scan steps all threads took. stop is the final stop flag.
    """

    kernel: str
    visited: np.ndarray     # (width, height, 3) uint8
    found_flag: np.ndarray  # int32 [found, x, y]
    clock: int
    stop: int
    h2d_ms: float           # img, visited, found_flag to the device
    kernel_ms: float        # launch + synchronize
    d2h_ms: float           # visited and found_flag back
    total_ms: float


def run_scan(img, visited, width, height, found_flag, kernel="scan"):
    """One launch of `kernel` ("scan" = scan_image_small_example, "simple" =
    simple_scan_kernel) on one scene, with process_image_small_example's
    steps (without its print) and a timing decomposition. The arguments are
    setup_scene_small_example's tuple. Inputs are not modified."""
    _warmup(kernel)
    d_state = cp.zeros(STATE_SLOTS, dtype=cp.int32)  # the shared arrays' twin
    sync()

    t0 = time.perf_counter()
    d_img = cp.asarray(img)
    d_visited = cp.asarray(visited)
    d_found_flag = cp.asarray(found_flag)
    sync()
    t1 = time.perf_counter()
    _launch(kernel, d_img, d_visited, width, height, d_found_flag, d_state)
    sync()
    t2 = time.perf_counter()
    result_visited = d_visited.get()
    result_found = d_found_flag.get()
    t3 = time.perf_counter()
    state = d_state.get()

    return ScanRun(
        kernel=kernel, visited=result_visited, found_flag=result_found,
        clock=int(state[CLOCK]), stop=int(state[STOP]),
        h2d_ms=(t1 - t0) * 1000, kernel_ms=(t2 - t1) * 1000,
        d2h_ms=(t3 - t2) * 1000, total_ms=(t3 - t0) * 1000)


def main_small_example(rng_seed=None, show=False):
    """Main function for the small example."""
    from PIL import Image

    from flood_fill_cuda.shared.results_paths import results_dir

    out = results_dir("triton_twins", "scan_multi_blob")
    _warmup("scan")  # compile outside the timed window

    # Setup the scene
    img, visited, width, height, found_flag = setup_scene_small_example(rng_seed)

    # Save original image
    Image.fromarray(img).save(os.path.join(out, "scan_small_original.png"))

    # Process on GPU
    print("Processing smaller example image...")
    start_time = timeit.default_timer()
    processed_visited, found_info = process_image_small_example(img, visited, width, height, found_flag)
    end_time = timeit.default_timer()

    elapsed_time = (end_time - start_time) * 1000  # Convert to ms
    print(f"Processing time: {elapsed_time:.2f} ms")

    if found_info[0] == 1:
        print(f"Found red pixel at ({found_info[1]}, {found_info[2]})")
    else:
        print("No red pixels found")

    # Save visited visualization
    Image.fromarray(processed_visited).save(
        os.path.join(out, "scan_small_visited.png"))
    print(f"images written to {out}")

    if show:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img)
        ax1.set_title("Original Image")
        ax2.imshow(processed_visited)
        ax2.set_title("Visited Pixels")
        if found_info[0] == 1:
            ax2.plot(found_info[2], found_info[1], 'ro', markersize=10, markeredgecolor='yellow')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main_small_example()
