"""
Correctness tests: Triton twin of the ch00 single-block BFS prototype.

The Numba prototype has no tests. Its deterministic contract, while the
blob fits the 6000-slot queue, follows from its code:

- the recolored pixels (R, G = new_color[0], new_color[1]) are exactly the
  8-connected blob of the seed: shared.cpu_oracle.cpu_flood_fill_8's
  visited mask;
- visited is that blob grown by one pixel in all 8 directions (clipped to
  the image): the prototype claims a neighbor BEFORE testing it for red, so
  the white ring around the blob is marked visited too;
- the value the kernel prints at exit (queue_front) is the oracle's fill
  count;
- every other pixel keeps its color. The blue channel of a recolored pixel
  is the debug value (tid * 4) % 255 of whichever thread took it: a
  schedule-dependent lane id, checked only for being one of the 64 values.

Past the capacity, enqueues are dropped: exactly 6000 pixels get recolored
(which ones depends on the schedule), visited is the recolored set grown by
one pixel, and the printed front counts every ticket, dropped ones too.

The test_cross_backend_* section runs the Numba kernel on the same seeded
scenes and requires identical visited arrays, identical R and G channels,
an identical recolored mask, and the same printed queue_front (captured
from the Numba device print). Numba is never run past the capacity: it then
reads its shared queue out of bounds.

Run:

    .venv/bin/python -m pytest -p no:cacheprovider src/flood_fill_cuda/triton_twins/chapters/ch00_cpu_baseline/test_correctness.py -v
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import ctypes
import random
import sys
import tempfile

import numpy as np
import pytest

from flood_fill_cuda.chapters.ch00_cpu_baseline import single_block as numba_proto
from flood_fill_cuda.shared import scenes
from flood_fill_cuda.shared.cpu_oracle import cpu_flood_fill_8

from .single_block import (
    QUEUE_CAPACITY, THREADS_PER_BLOCK, compiled_kernel, profile_kernel,
    run_flood_fill, setup_scene,
)

SEEDS = [0, 1, 2, 3, 4, 17, 99]
NEW_COLOR = np.array([0, 0, 255], dtype=np.uint8)
DEBUG_BLUES = {(tid * 4) % 255 for tid in range(1024)}


def dilate8(mask):
    """mask grown by one pixel in all 8 directions, clipped to the image."""
    w, h = mask.shape
    p = np.pad(mask.astype(bool), 1)
    out = np.zeros((w, h), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out |= p[1 + dx:1 + dx + w, 1 + dy:1 + dy + h]
    return out


def as_scene(img, sx, sy, threads_per_block=THREADS_PER_BLOCK):
    """A shared.scenes scene in setup_scene's tuple layout."""
    w, h = img.shape[0], img.shape[1]
    return (img, np.zeros((w, h), dtype=np.int32), sx, sy, w, h,
            NEW_COLOR.copy(), threads_per_block, 1)


def recolored_mask(img_out, new_color):
    return (img_out[..., 0] == new_color[0]) & (img_out[..., 1] == new_color[1])


def assert_contract(scene, img_out, visited_out, front, tpb=THREADS_PER_BLOCK):
    """The prototype's deterministic contract below the queue capacity."""
    img, _, sx, sy, _, _, new_color = scene[:7]
    ref_visited, _, _, ref_filled = cpu_flood_fill_8(img, sx, sy)
    assert ref_filled <= QUEUE_CAPACITY, "scene does not fit the queue"
    blob = ref_visited == 1
    rec = recolored_mask(img_out, new_color)
    np.testing.assert_array_equal(rec, blob)
    np.testing.assert_array_equal(visited_out == 1, dilate8(blob))
    assert set(np.unique(visited_out)) <= {0, 1}
    assert front == ref_filled
    np.testing.assert_array_equal(img_out[~blob], img[~blob])
    blues = set(int(b) for b in np.unique(img_out[blob][:, 2]))
    assert blues <= {(tid * 4) % 255 for tid in range(tpb)}


SCENES = {
    "square_50_in_128": lambda: scenes.square_scene(128, 128, 50, 50),
    "square_at_corner": lambda: scenes.square_scene(96, 96, 40, 40, corner=True),
    "square_nonsquare_img": lambda: scenes.square_scene(150, 70, 60, 40),
    "disk_r30": lambda: scenes.disk_scene(101, 101, 30),
    "serpentine_64": lambda: scenes.serpentine_scene(64, 64),
    "random_supercritical": lambda: scenes.random_scene(100, 100, 0.45, rng_seed=7),
    "random_subcritical": lambda: scenes.random_scene(200, 200, 0.30, rng_seed=7),
    "single_pixel": lambda: scenes.single_pixel_scene(64, 64),
    "full_red_64": lambda: scenes.full_red_scene(64, 64),
}


# ---------------------------------------------------------------------------
# Twin tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_default_scene_meets_contract(seed):
    scene = setup_scene(seed)
    r = run_flood_fill(*scene)
    assert_contract(scene, r.img, r.visited, r.front)


@pytest.mark.parametrize("name", SCENES.keys())
def test_shared_scenes_meet_contract(name):
    scene = as_scene(*SCENES[name]())
    r = run_flood_fill(*scene)
    assert_contract(scene, r.img, r.visited, r.front)


@pytest.mark.parametrize("tpb", [32, 64, 128, 256, 512, 1024])
def test_block_size_invariance(tpb):
    scene = as_scene(*scenes.random_scene(100, 100, 0.5, rng_seed=3),
                     threads_per_block=tpb)
    r = run_flood_fill(*scene)
    assert r.threads_per_block == tpb
    assert_contract(scene, r.img, r.visited, r.front, tpb)


def test_queue_exactly_full():
    """6000 pixels fill tickets 0..5999: nothing is dropped."""
    scene = as_scene(*scenes.full_red_scene(60, 100))
    r = run_flood_fill(*scene)
    assert_contract(scene, r.img, r.visited, r.front)
    assert r.front == QUEUE_CAPACITY


@pytest.mark.parametrize("name,build", [
    ("full_red_100", lambda: scenes.full_red_scene(100, 100)),
    ("square_90_in_128", lambda: scenes.square_scene(128, 128, 90, 90)),
])
def test_queue_overflow_drops_past_capacity(name, build):
    """Past 6000 tickets the enqueue is dropped: exactly the 6000 stored
    entries get processed, the rest of the blob stays red."""
    img, sx, sy = build()
    scene = as_scene(img, sx, sy)
    ref_visited, _, _, ref_filled = cpu_flood_fill_8(img, sx, sy)
    assert ref_filled > QUEUE_CAPACITY
    r = run_flood_fill(*scene)
    rec = recolored_mask(r.img, NEW_COLOR)
    assert int(rec.sum()) == QUEUE_CAPACITY
    assert not (rec & (ref_visited == 0)).any()
    np.testing.assert_array_equal(r.visited == 1, dilate8(rec))
    reds = (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)
    # Every claimed red pixel took one ticket (the seed is ticket 0).
    assert r.front == int(((r.visited == 1) & reds).sum()) > QUEUE_CAPACITY
    np.testing.assert_array_equal(r.img[~rec], img[~rec])


def test_profile_kernel_prints_stats_and_returns_last_run(capsys):
    img_out, visited_out = profile_kernel(num_runs=3, rng_seed=40)
    out = capsys.readouterr().out
    assert "Kernel execution time over 3 runs:" in out
    for label in ("Average", "Min", "Max", "Std Dev"):
        assert f"  {label}: " in out
    last = setup_scene(40 + 3)  # warm-up scene 40, runs 41..43
    ref_visited, _, _, ref_filled = cpu_flood_fill_8(last[0], last[2], last[3])
    np.testing.assert_array_equal(recolored_mask(img_out, last[6]), ref_visited == 1)
    np.testing.assert_array_equal(visited_out == 1, dilate8(ref_visited == 1))


def test_seeded_scene_is_reproducible_and_leaves_global_random_alone():
    random.seed(321)
    expected_next = random.random()
    random.seed(321)
    a = setup_scene(8)
    b = setup_scene(8)
    assert random.random() == expected_next
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    img, _, sx, sy, w, h, new_color, tpb, bpg = a
    assert (w, h, tpb, bpg) == (400, 400, 64, 1)
    assert tuple(img[sx, sy]) == (255, 0, 0)
    assert list(new_color) == [0, 0, 255]


def test_inputs_are_not_modified():
    scene = setup_scene(5)
    copies = [np.array(v, copy=True) for v in scene[:2]] + [scene[6].copy()]
    run_flood_fill(*scene)
    for before, after in zip(copies, (scene[0], scene[1], scene[6])):
        np.testing.assert_array_equal(before, after)


def test_non_power_of_2_block_is_rejected():
    scene = as_scene(*scenes.square_scene(64, 64, 20, 20), threads_per_block=96)
    with pytest.raises(ValueError, match="threads_per_block must be a power of 2"):
        run_flood_fill(*scene)


def test_more_than_one_block_is_rejected():
    scene = list(as_scene(*scenes.square_scene(64, 64, 20, 20)))
    scene[8] = 2
    with pytest.raises(ValueError, match="blocks_per_grid must be 1"):
        run_flood_fill(*scene)


def test_two_warps_for_64_threads():
    from flood_fill_cuda.triton_twins.runtime import kernel_resources

    assert kernel_resources(compiled_kernel(64))["num_warps"] == 2


def test_no_recompile_inside_timing():
    """After the warm-up, other scenes, seeds and image sizes reuse the
    compiled kernel, so no Triton compile lands inside kernel_ms."""
    from flood_fill_cuda.triton_twins.runtime import bridge  # noqa: F401
    from triton.runtime.driver import driver

    from .single_block import flood_fill

    dev = driver.active.get_current_device()
    run_flood_fill(*setup_scene(0))
    before = len(flood_fill.device_caches[dev][0])
    for scene in (setup_scene(1), as_scene(*scenes.square_scene(97, 33, 51, 17)),
                  as_scene(*scenes.full_red_scene(129, 1)),
                  as_scene(*scenes.single_pixel_scene(32, 48))):
        run_flood_fill(*scene)
    assert len(flood_fill.device_caches[dev][0]) == before


# ---------------------------------------------------------------------------
# Cross-backend: the Numba prototype on the same scenes
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL(None)


def run_numba(scene):
    """The prototype's launch, as profile_kernel does it, with its device
    print (queue_front) captured: CUDA printf writes through the C stdout,
    so fd 1 is redirected and the C buffer flushed around the launch.

    Refuses scenes that overflow the queue: Numba then reads its shared
    queue out of bounds, which can kill the CUDA context."""
    from numba import cuda

    img, visited, sx, sy, w, h, new_color, tpb, bpg = scene
    assert cpu_flood_fill_8(img, sx, sy)[3] <= QUEUE_CAPACITY, \
        "scene overflows the 6000-slot queue: not safe to run on Numba"
    d_img = cuda.to_device(img)
    d_visited = cuda.to_device(visited)
    color = np.array(new_color, copy=True)
    sys.stdout.flush()
    _libc.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile() as f:
        os.dup2(f.fileno(), 1)
        try:
            numba_proto.flood_fill[bpg, tpb](d_img, d_visited, sx, sy, w, h, color)
            cuda.synchronize()
            _libc.fflush(None)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        printed = f.read().decode().split()
    assert len(printed) == 1, f"expected one printed value, got {printed}"
    return d_img.copy_to_host(), d_visited.copy_to_host(), int(printed[0])


def assert_same_as_numba(scene, tri, tpb=THREADS_PER_BLOCK):
    nb_img, nb_visited, nb_front = run_numba(scene)
    assert_contract(scene, nb_img, nb_visited, nb_front, tpb)
    assert_contract(scene, tri.img, tri.visited, tri.front, tpb)
    np.testing.assert_array_equal(tri.visited, nb_visited)
    np.testing.assert_array_equal(tri.img[..., :2], nb_img[..., :2])
    new_color = scene[6]
    rec = recolored_mask(nb_img, new_color)
    np.testing.assert_array_equal(recolored_mask(tri.img, new_color), rec)
    np.testing.assert_array_equal(tri.img[~rec], nb_img[~rec])
    assert tri.front == nb_front


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_backend_default_scene(seed):
    scene = setup_scene(seed)
    assert_same_as_numba(scene, run_flood_fill(*scene))


@pytest.mark.parametrize("name", SCENES.keys())
def test_cross_backend_shared_scenes(name):
    scene = as_scene(*SCENES[name]())
    assert_same_as_numba(scene, run_flood_fill(*scene))


@pytest.mark.parametrize("tpb", [32, 128, 1024])
def test_cross_backend_block_sizes(tpb):
    scene = as_scene(*scenes.random_scene(100, 100, 0.5, rng_seed=3),
                     threads_per_block=tpb)
    assert_same_as_numba(scene, run_flood_fill(*scene), tpb)


def test_cross_backend_queue_exactly_full():
    scene = as_scene(*scenes.full_red_scene(60, 100))
    assert_same_as_numba(scene, run_flood_fill(*scene))


def test_cross_backend_96_threads_numba_accepts_triton_rejects():
    """96 threads is a valid Numba block but not a Triton program
    (num_warps must be a power of 2): the twin refuses it by name."""
    scene = as_scene(*scenes.square_scene(64, 64, 20, 20), threads_per_block=96)
    nb_img, nb_visited, nb_front = run_numba(scene)
    assert_contract(scene, nb_img, nb_visited, nb_front, 96)
    with pytest.raises(ValueError, match="threads_per_block must be a power of 2"):
        run_flood_fill(*scene)


def test_cross_backend_same_seed_same_scene():
    """The twin's seeded builder is the Numba builder: same seed, same
    scene, whichever module is asked."""
    state = random.getstate()
    random.seed(6)
    try:
        direct = numba_proto.setup_scene()
    finally:
        random.setstate(state)
    for a, b in zip(direct, setup_scene(6)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
