"""Chapter 6 correctness: the run pipeline against the SAME CPU oracle.

The whole point of keeping ch05's canonicalisation rule — a blob's label
is the minimum linear index over its pixels — is that changing the
carrier (pixels to runs) must not change a single answer. So this suite
judges ch06 with ch05's `cpu_label_components` unchanged, and demands
bit-for-bit equality of:

    label map      every pixel's canonical label, -1 off-blob
    painted image  palette[label % 6] on red, untouched elsewhere
    n_blobs        the component count
    seeds          one canonical (lex-min) pixel per blob, label order

plus two structural invariants that are properties of the ALGORITHM and
would catch a merge that silently over- or under-links:

    union_done == n_runs - n_blobs     every successful link retires
                                       exactly one root
    runs == the numpy run count        the run table is the image's runs

Both contracts run on every scene: "rgb" (pack included) and "mask"
(packed input). They must agree exactly — a packed mask is a lossless
re-encoding, and if the two ever disagree the pack is lying.

The scene battery is deliberately hostile to a run-based method: rows
one pixel tall, images one pixel wide, dimensions that are not
multiples of 32 (the word size), 1-px stripes (every run length 1),
dense noise (the run table's worst case), and the shapes that broke
earlier chapters (U, comb, serpentine).
"""

import numpy as np
import pytest

from .recolor import recolor, RunRecolor, CONTRACTS
from ..ch05_gpu_nblob_nblock import scenes as _scenes
from ..ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components
from ..ch05_gpu_nblob_nblock.kernels import PALETTE_HOST, N_PALETTE

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RED = (255, 0, 0)


def _stripes(width, height, step):
    """Vertical 1-px red lines every `step` columns — every run has
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
    """Maximal runs of red along y (the contiguous axis) — the run table
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
    """The run table's size is the image's run count, computed by numpy —
    an independent witness that emit found every run and invented none."""
    img = _scene(name)
    result = recolor(img, emit_seeds=False, copy_img=False)
    assert result.n_runs == _numpy_run_count(img)


@pytest.mark.parametrize("name", sorted(SCENES))
def test_every_link_retires_one_root(name):
    """union_done == n_runs - n_blobs. ch05's structural invariant, over
    runs instead of pixels: the merge cannot over-link (that would drop
    blobs) or under-link (that would split one)."""
    img = _scene(name)
    result = recolor(img, emit_seeds=False, copy_img=False)
    assert result.union_done == result.n_runs - result.n_blobs


@pytest.mark.parametrize("name", sorted(SCENES))
def test_contracts_agree(name):
    """rgb and mask must produce identical everything. The mask is a
    lossless re-encoding of which pixels are red; if the two contracts
    ever differ, pack lost information."""
    img = _scene(name)
    a = recolor(img, contract="rgb", emit_label=True)
    b = recolor(img, contract="mask", emit_label=True)
    np.testing.assert_array_equal(a.label, b.label)
    np.testing.assert_array_equal(a.img, b.img)
    assert (a.n_runs, a.n_blobs, a.seeds) == (b.n_runs, b.n_blobs, b.seeds)


def test_blank_image_is_valid():
    """Zero blobs is a legal answer, not an error — ch05's rule kept."""
    img, _ = _scenes.blank_scene(32, 32)
    result = recolor(img, emit_label=True)
    assert (result.n_runs, result.n_blobs, result.seeds) == (0, 0, [])
    np.testing.assert_array_equal(result.img, img)
    assert (result.label == -1).all()


def test_pack_is_lossless():
    """unpack(pack(img)) == img for a pure red/white scene: the mask
    contract's whole premise, tested rather than assumed."""
    from numba import cuda
    img, _ = _scenes.random_blobs_scene(97, 131, 0.3, 11)
    engine = RunRecolor(97, 131)
    dev = cuda.to_device(img)
    engine.pack(dev)
    out = cuda.to_device(np.zeros_like(img))
    engine.unpack_to(out)
    cuda.synchronize()
    np.testing.assert_array_equal(out.copy_to_host(), img)


def test_padding_bits_are_zero():
    """Bits past `height` in a row's last word must be 0 — every
    downstream kernel relies on them terminating the row's last run."""
    from numba import cuda
    height = 100                       # 4 words, 28 padding bits
    img = _scenes.full_red_scene(8, height)[0]
    engine = RunRecolor(8, height)
    dev = cuda.to_device(img)
    engine.pack(dev)
    cuda.synchronize()
    mask = engine.mask.copy_to_host()
    tail = mask[:, -1] >> np.uint32(height % 32)
    assert (tail == 0).all()
    # and the full-red row is exactly one run, not two
    result = recolor(img, emit_seeds=False, copy_img=False)
    assert result.n_runs == 8 and result.n_blobs == 1


def test_run_capacity_overflow_is_reported_not_silent():
    """An undersized run table must raise with the count it needed —
    never return a wrong answer, and never index out of bounds."""
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
    """A reused engine must give the same answer every time — the
    benchmark measures steady state, so state must not leak between
    runs (counters, row offsets, parent array)."""
    from numba import cuda
    from .kernels import RUN_OVERFLOW, N_BLOBS
    img, _ = _scenes.random_blobs_scene(80, 80, 0.3, 9)
    label, n_blobs = cpu_label_components(img)
    engine = RunRecolor(80, 80, run_capacity=4096)
    for _ in range(3):
        dev = cuda.to_device(img)
        engine.run(dev, contract="rgb")
        cuda.synchronize()
        lab = cuda.to_device(np.full((80, 80), -1, dtype=np.int32))
        engine.emit_label_map(lab)
        cuda.synchronize()
        np.testing.assert_array_equal(lab.copy_to_host(), label)
        counters = engine.counters.copy_to_host()
        assert counters[RUN_OVERFLOW] == 0
        assert int(counters[N_BLOBS]) == n_blobs


def test_low_level_run_flags_overflow_without_corrupting_memory():
    """RunRecolor.run() is the raw API and does NOT raise — it sets the
    tripwire and clamps every downstream loop to the capacity, so an
    undersized table gives a wrong answer but never an illegal access.
    (An 80x80 noise scene really does have 1343 runs.)"""
    from numba import cuda
    from .kernels import RUN_OVERFLOW, N_RUNS, N_RUNS_USED
    img, _ = _scenes.random_blobs_scene(80, 80, 0.3, 9)
    engine = RunRecolor(80, 80, run_capacity=256)
    dev = cuda.to_device(img)
    engine.run(dev, contract="rgb")
    cuda.synchronize()
    counters = engine.counters.copy_to_host()
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
