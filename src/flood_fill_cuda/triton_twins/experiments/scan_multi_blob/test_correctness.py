"""
Correctness tests: Triton twin of the scan-based blob discovery prototype.

The Numba experiment has no tests, and its output is nondeterministic by
construction: the ticks come from a block-wide atomic clock, the first
finder wins a race, and the stop flag is read while others write it. So
these tests check the invariants every correct run satisfies, whatever the
schedule, on both kernels:

- found_flag[0] is 1 exactly when the image has red;
- the found pixel is red and is the first red pixel of its patch in that
  patch's scan order, and the finder's scan ends on it (it stops at its
  next step);
- each patch's painted pixels are a prefix of its scan order (the
  anti-diagonal order for scan_image_small_example, row-major for
  simple_scan_kernel), and the brightness never decreases along it (each
  thread's ticks increase);
- a painted red pixel is magenta-ish (G == 0, R == B), every other painted
  pixel carries the hue of the thread that owns its patch;
- simple_scan_kernel paints every pixel and ticks exactly 100 * 100 times.

The test_cross_backend_* section runs the Numba kernels on the same seeded
scenes and requires the deterministic parts to be identical (found or not,
coverage on full scans, the red/magenta mask) and both results to satisfy
the same invariants. The brightness of each pixel (its tick) is
schedule-dependent and never compared.

Run:

    .venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/experiments/scan_multi_blob/test_correctness.py -v
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import numpy as np
import pytest

from flood_fill_cuda.experiments.scan_multi_blob import scan_only as numba_scan

from .scan_only import (
    BLOCK, MAX_TICKS, NUM_WARPS, PATCH_SIZE, THREADS_PER_BLOCK, blank_scene,
    compiled_kernel, process_image_small_example, run_scan,
    setup_scene_small_example,
)

SEEDS = [0, 1, 2, 3, 7]
GOLDEN = 0.618033988749895


# ---------------------------------------------------------------------------
# Invariant checkers (shared by both backends' results)
# ---------------------------------------------------------------------------

def base_color(tid):
    """The kernels' per-thread hue, computed on the host in float64."""
    h = (tid * GOLDEN) % 1.0
    h_i = int(h * 6)
    f = h * 6 - h_i
    if h_i == 0:
        return 255, int(255 * f), 0
    if h_i == 1:
        return int(255 * (1 - f)), 255, 0
    if h_i == 2:
        return 0, 255, int(255 * f)
    if h_i == 3:
        return 0, int(255 * (1 - f)), 255
    if h_i == 4:
        return int(255 * f), 0, 255
    return 255, 0, int(255 * (1 - f))


def scan_order(kernel, tid, width=100):
    """The (x, y) pixels thread tid visits, in its order."""
    ppr = width // PATCH_SIZE
    px, py = (tid % ppr) * PATCH_SIZE, (tid // ppr) * PATCH_SIZE
    if kernel == "scan":   # anti-diagonals: sum_idx, then i
        return [(px + i, py + s - i) for s in range(2 * PATCH_SIZE - 1)
                for i in range(PATCH_SIZE) if 0 <= s - i < PATCH_SIZE]
    return [(px + i, py + j) for j in range(PATCH_SIZE)   # rows: j, then i
            for i in range(PATCH_SIZE)]


def red_mask(img):
    return (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)


def brightness_index(pixel, is_red, base):
    """int(255 * brightness), read back from a painted pixel."""
    if is_red:
        return int(pixel[0])
    return int(pixel[base.index(255)])


def check_pixel(pixel, is_red, base):
    """Magenta for red; the thread's hue at some brightness otherwise."""
    r, g, b = (int(c) for c in pixel)
    if is_red:
        assert g == 0 and r == b, f"red pixel painted {pixel}, not magenta"
        assert 51 <= r <= 254, f"red pixel brightness {r} outside [51, 254]"
        return
    v = brightness_index(pixel, False, base)
    assert 51 <= v <= 254, f"pixel {pixel}: brightness {v} outside [51, 254]"
    # brightness in [v / 255, (v + 1) / 255): each channel follows.
    for c, bc in zip((r, g, b), base):
        lo, hi = (bc * v) // 255, (bc * (v + 1)) // 255
        assert lo <= c <= hi, f"pixel {pixel} is not hue {base} at {v}/255"


def check_invariants(kernel, img, visited, found_flag):
    """Every schedule-free property of one run (see the module docstring)."""
    width = img.shape[0]
    reds = red_mask(img)
    painted = visited.any(axis=2)
    found, fx, fy = (int(v) for v in found_flag)
    assert found == int(reds.any())

    for tid in range(THREADS_PER_BLOCK):
        order = scan_order(kernel, tid, width)
        flags = [bool(painted[x, y]) for x, y in order]
        k = sum(flags)
        assert flags == [True] * k + [False] * (len(order) - k), \
            f"thread {tid}: painted pixels are not a prefix of its scan"
        base = base_color(tid)
        last = -1
        for x, y in order[:k]:
            check_pixel(visited[x, y], reds[x, y], base)
            v = brightness_index(visited[x, y], reds[x, y], base)
            assert v >= last, f"thread {tid}: brightness went down at {(x, y)}"
            last = v

    if kernel == "simple":
        assert painted.all()
        if found:
            # Racy x and y stores can mix two finders: still red on a square.
            assert reds[fx, fy]
        return
    if found:
        assert reds[fx, fy]
        tid = (fy // PATCH_SIZE) * (width // PATCH_SIZE) + fx // PATCH_SIZE
        order = scan_order("scan", tid, width)
        first_red = next(p for p in order if reds[p])
        assert (fx, fy) == first_red, "found pixel is not its patch's first red"
        k = sum(bool(painted[p]) for p in order)
        assert order[k - 1] == (fx, fy), "the finder kept painting after the find"
    else:
        assert fx == 0 and fy == 0


def spot_scene(x0, y0, size):
    """blank_scene with a small red square at (x0, y0). The seeded scenes'
    20x20 square always covers a patch corner (the first pixel of a scan),
    so their scans stop after one step; an off-corner spot is found mid-scan
    and the threads stop at different points."""
    img, visited, width, height, found_flag = blank_scene()
    img[x0:x0 + size, y0:y0 + size] = (255, 0, 0)
    return img, visited, width, height, found_flag


SPOTS = [(57, 33, 2), (4, 95, 1), (91, 6, 3)]


def run_numba(kernel, img, visited, width, height, found_flag):
    """The Numba host's launch: kernel[1, 100] (simple_scan_kernel is never
    launched by the Numba host; the test launches it the same way)."""
    if kernel == "scan":
        return process_numba(img, visited, width, height, found_flag)
    from numba import cuda

    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(visited)
    d_found = cuda.to_device(found_flag)
    numba_scan.simple_scan_kernel[1, THREADS_PER_BLOCK](
        d_img, d_visited, width, height, d_found)
    cuda.synchronize()
    return d_visited.copy_to_host(), d_found.copy_to_host()


def process_numba(img, visited, width, height, found_flag):
    return numba_scan.process_image_small_example(
        img, visited, width, height, found_flag)


# ---------------------------------------------------------------------------
# Twin tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_scan_finds_the_square_and_stops(seed):
    img, visited, w, h, found_flag = setup_scene_small_example(seed)
    r = run_scan(img, visited, w, h, found_flag, kernel="scan")
    check_invariants("scan", img, r.visited, r.found_flag)
    assert r.found_flag[0] == 1 and r.stop == 1
    # Ticks are taken before the stop check, so a stopping step ticks too.
    assert r.clock >= int(r.visited.any(axis=2).sum())
    assert r.clock < MAX_TICKS  # the early stop cut the scan short


@pytest.mark.parametrize("spot", SPOTS)
def test_scan_stops_mid_patch(spot):
    img, visited, w, h, found_flag = spot_scene(*spot)
    r = run_scan(img, visited, w, h, found_flag, kernel="scan")
    check_invariants("scan", img, r.visited, r.found_flag)
    assert r.found_flag[0] == 1 and r.stop == 1
    assert 1 < int(r.visited.any(axis=2).sum()) < w * h


def test_scan_blank_image_scans_everything():
    img, visited, w, h, found_flag = blank_scene()
    r = run_scan(img, visited, w, h, found_flag, kernel="scan")
    check_invariants("scan", img, r.visited, r.found_flag)
    assert r.visited.any(axis=2).all()
    assert list(r.found_flag) == [0, 0, 0] and r.stop == 0
    assert r.clock == MAX_TICKS


@pytest.mark.parametrize("seed", SEEDS)
def test_simple_scan_paints_everything(seed):
    img, visited, w, h, found_flag = setup_scene_small_example(seed)
    r = run_scan(img, visited, w, h, found_flag, kernel="simple")
    check_invariants("simple", img, r.visited, r.found_flag)
    assert r.clock == MAX_TICKS
    magenta = (r.visited[..., 1] == 0) & (r.visited[..., 0] == r.visited[..., 2])
    assert np.array_equal(magenta & red_mask(img), red_mask(img))


def test_simple_scan_blank_image():
    img, visited, w, h, found_flag = blank_scene()
    r = run_scan(img, visited, w, h, found_flag, kernel="simple")
    check_invariants("simple", img, r.visited, r.found_flag)
    assert list(r.found_flag) == [0, 0, 0]


def test_seeded_scene_is_reproducible_and_leaves_global_random_alone():
    import random

    random.seed(123)
    expected_next = random.random()
    random.seed(123)
    a = setup_scene_small_example(5)
    b = setup_scene_small_example(5)
    assert random.random() == expected_next
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert red_mask(a[0]).sum() == 20 * 20


def test_process_image_small_example_matches_contract(capsys):
    img, visited, w, h, found_flag = setup_scene_small_example(11)
    out_visited, out_found = process_image_small_example(img, visited, w, h,
                                                         found_flag)
    assert "Launching with single block of 100 threads" in capsys.readouterr().out
    check_invariants("scan", img, out_visited, out_found)
    assert not visited.any() and not found_flag.any()  # inputs untouched


def test_hundred_threads_run_as_128_lanes_in_4_warps():
    from flood_fill_cuda.triton_twins.runtime import kernel_resources

    assert (BLOCK, NUM_WARPS) == (128, 4)
    for kernel in ("scan", "simple"):
        assert kernel_resources(compiled_kernel(kernel))["num_warps"] == 4


def test_unknown_kernel_is_rejected():
    img, visited, w, h, found_flag = blank_scene()
    with pytest.raises(ValueError, match="kernel must be"):
        run_scan(img, visited, w, h, found_flag, kernel="dfs")


# ---------------------------------------------------------------------------
# Cross-backend: the Numba kernels on the same scenes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_cross_backend_scan_same_scene(seed):
    scene = setup_scene_small_example(seed)
    img = scene[0]
    nb_visited, nb_found = run_numba("scan", *scene)
    tri = run_scan(*scene, kernel="scan")
    check_invariants("scan", img, nb_visited, nb_found)
    check_invariants("scan", img, tri.visited, tri.found_flag)
    assert nb_found[0] == tri.found_flag[0] == 1


@pytest.mark.parametrize("spot", SPOTS)
def test_cross_backend_scan_spot_scene(spot):
    """Found mid-scan: where each thread stops depends on the schedule,
    the invariants and the found pixel do not. Each spot lies inside one
    patch, so that patch's thread is the only possible finder and the found
    pixel is its first red pixel on both backends."""
    scene = spot_scene(*spot)
    nb_visited, nb_found = run_numba("scan", *scene)
    tri = run_scan(*scene, kernel="scan")
    check_invariants("scan", scene[0], nb_visited, nb_found)
    check_invariants("scan", scene[0], tri.visited, tri.found_flag)
    assert nb_found[0] == tri.found_flag[0] == 1
    np.testing.assert_array_equal(nb_found, tri.found_flag)


def test_cross_backend_scan_blank_scene():
    scene = blank_scene()
    nb_visited, nb_found = run_numba("scan", *scene)
    tri = run_scan(*scene, kernel="scan")
    check_invariants("scan", scene[0], nb_visited, nb_found)
    check_invariants("scan", scene[0], tri.visited, tri.found_flag)
    np.testing.assert_array_equal(nb_visited.any(axis=2), tri.visited.any(axis=2))
    np.testing.assert_array_equal(nb_found, tri.found_flag)


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_backend_simple_scan_same_scene(seed):
    scene = setup_scene_small_example(seed)
    img = scene[0]
    nb_visited, nb_found = run_numba("simple", *scene)
    tri = run_scan(*scene, kernel="simple")
    check_invariants("simple", img, nb_visited, nb_found)
    check_invariants("simple", img, tri.visited, tri.found_flag)
    np.testing.assert_array_equal(nb_visited.any(axis=2), tri.visited.any(axis=2))
    assert nb_found[0] == tri.found_flag[0]
    reds = red_mask(img)
    for v in (nb_visited, tri.visited):
        magenta = (v[..., 1] == 0) & (v[..., 0] == v[..., 2])
        assert (magenta[reds]).all()


def test_cross_backend_same_seed_same_scene():
    """The twin's seeded builder is the Numba builder: same seed, same
    scene, whichever module is asked."""
    import random

    state = random.getstate()
    random.seed(4)
    try:
        direct = numba_scan.setup_scene_small_example()
    finally:
        random.setstate(state)
    for a, b in zip(direct, setup_scene_small_example(4)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
