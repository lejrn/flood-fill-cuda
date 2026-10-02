"""Chapter 6 correctness, Triton twin: the same suite, the same oracle.

Part 1 is chapters/ch06_gpu_nblob_runs/test_correctness.py test for
test (same names, scenes, parameters and assertions), run against the
Triton driver. It judges the twin with ch05's `cpu_label_components`,
unchanged, bit for bit: label map, painted image, n_blobs, seeds, the
run count against numpy, and union_done == n_runs - n_blobs. Only the
four low-level tests change their device plumbing (CuPy instead of
numba.cuda).

Part 2 (test_cross_backend_*) runs the Numba driver beside the twin on
the same input and demands identical deterministic outputs: the result
fields, the device buffers after run() (packed mask, row counts, row
offsets, counters, run table, labels, the root set), unpack and the
label map, overflowed tables, and other block sizes. parent[] for
non-root runs is NOT compared: the path-halving race makes it
schedule-dependent in both backends (only the root set is fixed).

Part 3 (test_twin_*) covers what only the twin has: the power-of-2
block-size rule, the 1024-thread block Numba cannot launch, the
no-recompile guarantee behind its timings, and the Triton twins of the
benchmark's read/write peak probes.
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import dataclasses

import cupy as cp
import numpy as np
import pytest

from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock import scenes as _scenes
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.cpu_oracle import (
    cpu_label_components,
)
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.kernels import (
    PALETTE_HOST, N_PALETTE,
)
from flood_fill_cuda.triton_twins.runtime import sync

from .recolor import recolor, RunRecolor, CONTRACTS

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RED = (255, 0, 0)


def _stripes(width, height, step):
    """Vertical 1-px red lines every `step` columns: every run has
    length 1, the run table's densest legitimate shape."""
    img = np.full((width, height, 3), 255, dtype=np.uint8)
    for y in range(0, height, step):
        img[:, y] = RED
    return img


def _red_mask(img):
    return ((img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0))


def _expected_paint(img, label):
    """ch05's paint contract: palette[canonical label % 6] on red."""
    out = img.copy()
    red = _red_mask(img)
    out[red] = PALETTE_HOST[(label % N_PALETTE)[red]]
    return out


def _numpy_run_count(img):
    """Maximal runs of red along y (the contiguous axis): the run table
    size, computed a completely different way."""
    red = _red_mask(img).astype(np.int8)
    d = np.diff(red, axis=1, prepend=0)
    return int((d == 1).sum())


def _expected_seeds(label, height):
    return [(int(l) // height, int(l) % height)
            for l in np.unique(label[label >= 0])]


SCENES = {
    "two_squares": lambda: _scenes.two_squares_scene(64, 64, 20, 20, gap=4),
    "two_disks": lambda: _scenes.two_disks_scene(80, 80, 15, gap=6),
    "asym_squares": lambda: _scenes.asym_squares_scene(80, 80, 30, 8, gap=4),
    "two_pixels": lambda: _scenes.two_pixels_scene(32, 32),
    "u_shape": lambda: _scenes.u_shape_scene(40, 40),
    "comb_80_teeth": lambda: _scenes.comb_scene(64, 200, teeth=80),
    "serpentine": lambda: _scenes.serpentine_scene(64, 64),
    "single_pixel": lambda: _scenes.single_pixel_scene(16, 16),
    "full_red": lambda: _scenes.full_red_scene(48, 48),
    "blank": lambda: _scenes.blank_scene(32, 32),
    "blob_grid_100": lambda: _scenes.blob_grid_scene(200, 200, 10, 10, 12,
                                                     gap=4),
    "noise_030": lambda: _scenes.random_blobs_scene(120, 120, 0.30, 1),
    "noise_045": lambda: _scenes.random_blobs_scene(100, 100, 0.45, 7),
    "noise_010": lambda: _scenes.random_blobs_scene(150, 150, 0.10, 3),
    # word-size and degenerate-shape edges
    "size_33x33": lambda: _scenes.random_blobs_scene(33, 33, 0.35, 2),
    "size_31x97": lambda: _scenes.random_blobs_scene(31, 97, 0.35, 4),
    "size_97x131": lambda: _scenes.blob_grid_scene(97, 131, 4, 5, 11, gap=5),
    "one_row": lambda: (_stripes(1, 64, 3), None),
    "one_col": lambda: (_stripes(64, 1, 1), None),
    "stripes_1px": lambda: (_stripes(40, 97, 2), None),
}


def _scene(name):
    built = SCENES[name]()
    return built[0] if isinstance(built, tuple) else built


# ============================================================ part 1
# The Numba suite, test for test.

@pytest.mark.parametrize("name", sorted(SCENES))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_matches_cpu_oracle(name, contract):
    """Label map, paint, blob count and seeds, all bit-for-bit."""
    img = _scene(name)
    height = img.shape[1]
    result = recolor(img, contract=contract, emit_label=True)
    label, n_blobs = cpu_label_components(img)

    assert result.n_blobs == n_blobs
    np.testing.assert_array_equal(result.label, label)
    np.testing.assert_array_equal(result.img, _expected_paint(img, label))
    assert result.seeds == _expected_seeds(label, height)


@pytest.mark.parametrize("name", sorted(SCENES))
def test_run_table_is_the_images_runs(name):
    """The run table's size is the image's run count, computed by numpy:
    an independent witness that emit found every run and invented none."""
    img = _scene(name)
    result = recolor(img, emit_seeds=False, copy_img=False)
    assert result.n_runs == _numpy_run_count(img)


@pytest.mark.parametrize("name", sorted(SCENES))
def test_every_link_retires_one_root(name):
    """union_done == n_runs - n_blobs: the merge cannot over-link (that
    would drop blobs) or under-link (that would split one)."""
    img = _scene(name)
    result = recolor(img, emit_seeds=False, copy_img=False)
    assert result.union_done == result.n_runs - result.n_blobs


@pytest.mark.parametrize("name", sorted(SCENES))
def test_contracts_agree(name):
    """rgb and mask must produce identical everything: if the two
    contracts ever differ, pack lost information."""
    img = _scene(name)
    a = recolor(img, contract="rgb", emit_label=True)
    b = recolor(img, contract="mask", emit_label=True)
    np.testing.assert_array_equal(a.label, b.label)
    np.testing.assert_array_equal(a.img, b.img)
    assert (a.n_runs, a.n_blobs, a.seeds) == (b.n_runs, b.n_blobs, b.seeds)


def test_blank_image_is_valid():
    """Zero blobs is a legal answer, not an error (ch05's rule kept)."""
    img, _ = _scenes.blank_scene(32, 32)
    result = recolor(img, emit_label=True)
    assert (result.n_runs, result.n_blobs, result.seeds) == (0, 0, [])
    np.testing.assert_array_equal(result.img, img)
    assert (result.label == -1).all()


def test_pack_is_lossless():
    """unpack(pack(img)) == img for a pure red/white scene: the mask
    contract's whole premise, tested rather than assumed."""
    img, _ = _scenes.random_blobs_scene(97, 131, 0.3, 11)
    engine = RunRecolor(97, 131)
    dev = cp.asarray(img)
    engine.pack(dev)
    out = cp.asarray(np.zeros_like(img))
    engine.unpack_to(out)
    sync()
    np.testing.assert_array_equal(out.get(), img)


def test_padding_bits_are_zero():
    """Bits past `height` in a row's last word must be 0: every
    downstream kernel relies on them terminating the row's last run."""
    height = 100                       # 4 words, 28 padding bits
    img = _scenes.full_red_scene(8, height)[0]
    engine = RunRecolor(8, height)
    dev = cp.asarray(img)
    engine.pack(dev)
    sync()
    mask = engine.mask.get()
    tail = mask[:, -1] >> np.uint32(height % 32)
    assert (tail == 0).all()
    # and the full-red row is exactly one run, not two
    result = recolor(img, emit_seeds=False, copy_img=False)
    assert result.n_runs == 8 and result.n_blobs == 1


def test_run_capacity_overflow_is_reported_not_silent():
    """An undersized run table must raise with the count it needed:
    never a wrong answer, never an out-of-bounds index."""
    img, _ = _scenes.random_blobs_scene(64, 64, 0.4, 5)
    engine = RunRecolor(64, 64, run_capacity=8)
    with pytest.raises(RuntimeError, match="run table overflowed"):
        recolor(img, engine=engine)


def test_one_shot_grows_its_own_run_table():
    """Without a caller-owned engine the same overflow self-heals: the
    tripwire reports the exact count and the driver re-runs."""
    img, _ = _scenes.random_blobs_scene(64, 64, 0.4, 5)
    result = recolor(img, run_capacity=8, emit_label=True)
    label, n_blobs = cpu_label_components(img)
    assert result.n_blobs == n_blobs
    np.testing.assert_array_equal(result.label, label)


def test_engine_reuse_is_repeatable():
    """A reused engine must give the same answer every time: state must
    not leak between runs (counters, row offsets, parent array)."""
    from .kernels import RUN_OVERFLOW, N_BLOBS
    img, _ = _scenes.random_blobs_scene(80, 80, 0.3, 9)
    label, n_blobs = cpu_label_components(img)
    engine = RunRecolor(80, 80, run_capacity=4096)
    for _ in range(3):
        dev = cp.asarray(img)
        engine.run(dev, contract="rgb")
        sync()
        lab = cp.asarray(np.full((80, 80), -1, dtype=np.int32))
        engine.emit_label_map(lab)
        sync()
        np.testing.assert_array_equal(lab.get(), label)
        counters = engine.counters.get()
        assert counters[RUN_OVERFLOW] == 0
        assert int(counters[N_BLOBS]) == n_blobs


def test_low_level_run_flags_overflow_without_corrupting_memory():
    """RunRecolor.run() is the raw API and does NOT raise: it sets the
    tripwire and clamps every downstream loop to the capacity. (An 80x80
    noise scene really does have 1343 runs.)"""
    from .kernels import RUN_OVERFLOW, N_RUNS, N_RUNS_USED
    img, _ = _scenes.random_blobs_scene(80, 80, 0.3, 9)
    engine = RunRecolor(80, 80, run_capacity=256)
    dev = cp.asarray(img)
    engine.run(dev, contract="rgb")
    sync()
    counters = engine.counters.get()
    assert counters[RUN_OVERFLOW] == 1
    assert counters[N_RUNS] > 256
    assert counters[N_RUNS_USED] == 256


@pytest.mark.parametrize("bad,err", [
    (dict(contract="nope"), "contract must be one of"),
])
def test_input_validation(bad, err):
    img, _ = _scenes.blank_scene(8, 8)
    with pytest.raises(ValueError, match=err):
        recolor(img, **bad)


def test_rejects_non_uint8_image():
    with pytest.raises(ValueError, match="uint8"):
        recolor(np.zeros((8, 8, 3), dtype=np.int32))


# ============================================================ part 2
# Numba and Triton side by side: identical deterministic outputs.

# Representative scenes: dense noise (the run table's worst case and
# the merge's heaviest contention), word-size edges, degenerate shapes,
# 1-px runs, the shapes that broke earlier chapters, and one scene big
# enough to keep every program of the merge grid busy.
CROSS_SCENES = {
    "noise_030": SCENES["noise_030"],
    "noise_045": SCENES["noise_045"],
    "size_31x97": SCENES["size_31x97"],
    "one_row": SCENES["one_row"],
    "one_col": SCENES["one_col"],
    "stripes_1px": SCENES["stripes_1px"],
    "comb_80_teeth": SCENES["comb_80_teeth"],
    "blank": SCENES["blank"],
    "noise_1000": lambda: _scenes.random_blobs_scene(1000, 1000, 0.30, 0),
}

_built = {}


def _cross_scene(name):
    if name not in _built:
        built = CROSS_SCENES[name]()
        _built[name] = built[0] if isinstance(built, tuple) else built
    return _built[name]


def _numba():
    from flood_fill_cuda.chapters.ch06_gpu_nblob_runs import recolor as nb
    return nb


DETERMINISTIC_FIELDS = (
    "contract", "threads_per_block", "blocks", "n_runs", "n_blobs", "seeds",
    "union_attempts", "union_done", "red_px", "h2d_ms", "model_bytes",
    "run_capacity",
)


def _assert_results_equal(a, b):
    for f in DETERMINISTIC_FIELDS:
        assert getattr(a, f) == getattr(b, f), f
    np.testing.assert_array_equal(a.img, b.img)
    np.testing.assert_array_equal(a.label, b.label)
    assert list(a.phase_ms) == list(b.phase_ms)


@pytest.mark.parametrize("instrumented", [True, False])
@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("name", sorted(CROSS_SCENES))
def test_cross_backend_recolor_matches_numba(name, contract, instrumented):
    """Every deterministic field of the result, both contracts, both
    instrumentation variants (with instrumented=False, n_blobs comes
    from the root set and the union counters are the reset zeros)."""
    img = _cross_scene(name)
    kw = dict(contract=contract, emit_label=True, instrumented=instrumented)
    _assert_results_equal(_numba().recolor(img, **kw), recolor(img, **kw))


def _numba_run(img, capacity, contract, instrumented, grid=None):
    from numba import cuda
    nb = _numba()
    nb._warmup()
    engine = nb.RunRecolor(img.shape[0], img.shape[1], run_capacity=capacity,
                           **({"grid": grid} if grid else {}))
    dev = cuda.to_device(img)
    if contract == "mask":
        engine.pack(dev)
    engine.run(dev, contract=contract, instrumented=instrumented)
    cuda.synchronize()
    get = lambda a: a.copy_to_host()                       # noqa: E731
    return engine, get(dev), get


def _triton_run(img, capacity, contract, instrumented, grid=None):
    from .recolor import _warmup
    engine = RunRecolor(img.shape[0], img.shape[1], run_capacity=capacity,
                        **({"grid": grid} if grid else {}))
    _warmup(engine.grid[1])
    dev = cp.asarray(img)
    if contract == "mask":
        engine.pack(dev)
    engine.run(dev, contract=contract, instrumented=instrumented)
    sync()
    get = lambda a: a.get()                                # noqa: E731
    return engine, get(dev), get


def _buffers(engine, img_out, get):
    """Every deterministic device-side output of one run(): the whole
    packed mask (padding bits included), row counts and offsets, the
    counters, the run table and labels up to N_RUNS_USED, and the root
    set (each class's root is its minimum run id by the atomicMin
    protocol; non-root parent pointers are schedule-dependent)."""
    from .kernels import N_RUNS_USED
    counters = get(engine.counters)
    n = int(counters[N_RUNS_USED])
    parent = get(engine.parent)[:n]
    return {
        "img": img_out,
        "mask": get(engine.mask),
        "row_count": get(engine.row_count),
        "row_off": get(engine.row_off),
        "counters": counters,
        "run_x": get(engine.run_x)[:n],
        "run_y0": get(engine.run_y0)[:n],
        "run_y1": get(engine.run_y1)[:n],
        "run_label": get(engine.run_label)[:n],
        "roots": np.flatnonzero(parent == np.arange(n)),
    }


def _assert_buffers_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("instrumented", [True, False])
@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("name", ["noise_045", "size_31x97", "one_col",
                                  "noise_1000"])
def test_cross_backend_device_buffers_match_numba(name, contract,
                                                  instrumented):
    img = _cross_scene(name)
    cap = max(8192, _numpy_run_count(img) + 1)
    nb = _numba_run(img, cap, contract, instrumented)
    tr = _triton_run(img, cap, contract, instrumented)
    _assert_buffers_equal(_buffers(*nb), _buffers(*tr))


def test_cross_backend_overflowed_table_is_identical():
    """An overflowed run table is wrong but deterministic: the counters,
    the table prefix up to the capacity and everything computed from it
    (roots, labels, the paint of the truncated table) match exactly."""
    img = _cross_scene("noise_030")
    for contract in CONTRACTS:
        nb = _numba_run(img, 256, contract, True)
        tr = _triton_run(img, 256, contract, True)
        a, b = _buffers(*nb), _buffers(*tr)
        assert a["counters"][0] == 1
        _assert_buffers_equal(a, b)


@pytest.mark.parametrize("tpb", [32, 64, 128, 512])
def test_cross_backend_block_sizes_match_numba(tpb):
    """Other block sizes (one Numba block = one Triton program of tpb
    lanes): same buffers as Numba at the same grid. 1024 is not here:
    Numba's emit_kernel uses 79 registers per thread, so a 1024-thread
    Numba launch fails with LAUNCH_OUT_OF_RESOURCES
    (test_twin_block_size_1024_matches_cpu_oracle covers the twin)."""
    img = _cross_scene("noise_045")
    grid = (64, tpb)
    nb = _numba_run(img, 8192, "rgb", True, grid=grid)
    tr = _triton_run(img, 8192, "rgb", True, grid=grid)
    _assert_buffers_equal(_buffers(*nb), _buffers(*tr))


def test_cross_backend_unpack_and_label_map_match_numba():
    from numba import cuda
    img = _cross_scene("size_31x97")
    w, h = img.shape[:2]
    nb_engine, _, _ = _numba_run(img, 8192, "rgb", True)
    tr_engine, _, _ = _triton_run(img, 8192, "rgb", True)

    nb_lab = cuda.to_device(np.full((w, h), -1, dtype=np.int32))
    nb_engine.emit_label_map(nb_lab)
    nb_img = cuda.to_device(np.zeros_like(img))
    nb_engine.unpack_to(nb_img)
    cuda.synchronize()
    tr_lab = cp.full((w, h), -1, dtype=cp.int32)
    tr_engine.emit_label_map(tr_lab)
    tr_img = cp.zeros(img.shape, dtype=cp.uint8)
    tr_engine.unpack_to(tr_img)
    sync()
    np.testing.assert_array_equal(nb_lab.copy_to_host(), tr_lab.get())
    np.testing.assert_array_equal(nb_img.copy_to_host(), tr_img.get())


# ============================================================ part 3
# What only the twin has.

def test_twin_result_fields_match_numba():
    nb_fields = [f.name for f in dataclasses.fields(_numba().RunRecolorResult)]
    from .recolor import RunRecolorResult
    assert [f.name for f in dataclasses.fields(RunRecolorResult)] == nb_fields


@pytest.mark.parametrize("tpb", [96, 160, 2048])
def test_twin_rejects_non_power_of_2_block_size(tpb):
    """Numba accepts any multiple of 32; Triton's num_warps and
    tl.arange lengths must be powers of 2."""
    with pytest.raises(ValueError, match="power of 2"):
        RunRecolor(8, 8, grid=(2, tpb))
    img, _ = _scenes.blank_scene(8, 8)
    with pytest.raises(ValueError, match="power of 2"):
        recolor(img, grid=(2, tpb))


def test_twin_block_size_1024_matches_cpu_oracle():
    """Triton compiles each kernel for its block size (32 warps caps it
    at 64 registers), so tpb = 1024 runs where the Numba build cannot."""
    img = _cross_scene("noise_045")
    label, n_blobs = cpu_label_components(img)
    for contract in CONTRACTS:
        result = recolor(img, contract=contract, grid=(64, 1024),
                         emit_label=True)
        assert result.n_blobs == n_blobs
        np.testing.assert_array_equal(result.label, label)
        np.testing.assert_array_equal(result.img, _expected_paint(img, label))


def test_twin_rejects_multiple_of_32_rule_like_numba():
    with pytest.raises(ValueError, match="multiple of 32"):
        RunRecolor(8, 8, grid=(2, 48))


def test_twin_read_write_probes_measure_real_bandwidth():
    """The Triton twins of benchmark.py's read/write peak probes report
    a real DRAM bandwidth (64 MiB is past the L2), not a deleted read."""
    from .compare import measure_read_write_peaks_triton
    read_gb_s, write_gb_s = measure_read_write_peaks_triton(64 * 2 ** 20,
                                                            repeats=3)
    assert 20 < read_gb_s < 1000
    assert 20 < write_gb_s < 1000


def test_twin_new_shapes_do_not_recompile():
    """Every size argument is do_not_specialize, so after the warm-up a
    new image shape (1, 16, 33, 97 ... wide or tall) reuses the compiled
    kernels: a compile can never land inside a timed window."""
    from . import kernels as k
    kernels = (k.pack_kernel, k.unpack_kernel, k.count_kernel,
               k.row_scan_kernel, k.emit_kernel, k.merge_rows_kernel,
               k.flatten_kernel, k.paint_kernel, k.label_kernel)

    def n_compiled():
        return sum(len(c[0]) for kern in kernels
                   for c in kern.device_caches.values())

    recolor(_stripes(8, 8, 2), emit_label=True)
    engine = RunRecolor(8, 8)
    engine.unpack_to(cp.zeros((8, 8, 3), dtype=cp.uint8))
    sync()
    before = n_compiled()
    for w, h in [(1, 64), (64, 1), (16, 16), (33, 97), (48, 160), (1, 1)]:
        img, _ = _scenes.random_blobs_scene(w, h, 0.4, 3)
        for contract in CONTRACTS:
            for instrumented in (True, False):
                recolor(img, contract=contract, emit_label=True,
                        instrumented=instrumented)
        RunRecolor(w, h).unpack_to(cp.zeros((w, h, 3), dtype=cp.uint8))
    sync()
    assert n_compiled() == before
