"""
The CPU oracles against code this project did not write.

Every GPU kernel is tested against one of four CPU oracles, all written
here, all a queue BFS. A mistake shared by an oracle and its kernels (a
wrong neighbour offset, a wrong definition of red) would pass every
chapter's suite. These tests anchor the oracles outside the project:

- components: OpenCV's connectedComponents (a two-pass union-find
  labelling, not a BFS) must give the same partition and the same count;
- canonical labels: rebuilt from OpenCV's partition by the rule alone
  (the minimum linear index x*height + y of each component), they must
  equal the oracle's label map exactly;
- BFS depth: a breadth-first fill written as whole-array dilations (no
  queue, no neighbour table) must give the same level for every pixel.

CPU only. Run:

    uv run pytest src/flood_fill_cuda/shared/test_oracles_vs_opencv.py -v
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from .cpu_oracle import cpu_flood_fill_8
from ..chapters.ch01_gpu_1blob_1block.cpu_oracle import cpu_flood_fill
from ..chapters.ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components
from ..chapters.ch08_gpu_nblob_tiles.cpu_oracle import cpu_label

RED = (255, 0, 0)
INPUTS = Path(__file__).resolve().parents[3] / "images" / "input"
# colours one step away from red, and plain ones: none of them is red
DECOYS = np.array([[255, 255, 255], [0, 0, 0], [254, 0, 0], [255, 1, 0],
                   [255, 0, 1], [0, 0, 255], [0, 255, 0]], np.uint8)


def _image(mask, rng=None):
    """(W, H, 3) uint8: red where mask, white elsewhere (decoy colours
    with rng)."""
    img = np.full(mask.shape + (3,), 255, np.uint8)
    if rng is not None:
        img[...] = DECOYS[rng.integers(0, len(DECOYS), mask.shape)]
    img[mask] = RED
    return img


def _noise(w, h, p, seed, decoys=True):
    rng = np.random.default_rng(seed)
    return _image(rng.random((w, h)) < p, rng if decoys else None)


def _checkerboard(w, h):
    x, y = np.mgrid[0:w, 0:h]
    return _image((x + y) % 2 == 0)


def _diagonals(w, h, period):
    x, y = np.mgrid[0:w, 0:h]
    return _image((x - y) % period == 0)


def _rings(n):
    """Nested one-pixel square rings, two apart: separate components at
    both connectivities."""
    x, y = np.mgrid[0:n, 0:n]
    d = np.minimum(np.minimum(x, y), np.minimum(n - 1 - x, n - 1 - y))
    return _image(d % 3 == 0)


def _spiral(n):
    """One 4-connected path, 1 px wide, from the corner to the centre."""
    mask = np.zeros((n, n), bool)
    x = y = 0
    dx, dy = 0, 1
    mask[0, 0] = True
    for _ in range(n * n):
        nx, ny = x + dx, y + dy
        ax, ay = nx + dx, ny + dy
        blocked = not (0 <= nx < n and 0 <= ny < n) or (
            0 <= ax < n and 0 <= ay < n and mask[ax, ay])
        if blocked:
            dx, dy = dy, -dx            # turn
            nx, ny = x + dx, y + dy
            ax, ay = nx + dx, ny + dy
            if not (0 <= nx < n and 0 <= ny < n) or mask[nx, ny] or (
                    0 <= ax < n and 0 <= ay < n and mask[ax, ay]):
                break
        x, y = nx, ny
        mask[x, y] = True
    return _image(mask)


SCENES = {
    "blank": lambda: _image(np.zeros((33, 65), bool)),
    "full": lambda: _image(np.ones((33, 65), bool)),
    "one_pixel": lambda: _image(np.ones((1, 1), bool)),
    "row_1xN": lambda: _noise(1, 257, 0.5, 1),
    "col_Nx1": lambda: _noise(257, 1, 0.5, 2),
    "checkerboard": lambda: _checkerboard(64, 97),
    "diagonals_3": lambda: _diagonals(96, 131, 3),
    "diagonals_7": lambda: _diagonals(131, 96, 7),
    "rings": lambda: _rings(61),
    "spiral": lambda: _spiral(48),
    "noise_010": lambda: _noise(150, 211, 0.10, 3),
    "noise_030": lambda: _noise(97, 131, 0.30, 4),
    "noise_045_no_decoys": lambda: _noise(513, 300, 0.45, 5, decoys=False),
    # the percolation thresholds of the square lattice: site 0.5927 for
    # 4-connectivity, 0.4073 for 8: the largest, most tangled components
    "noise_perc4": lambda: _noise(400, 400, 0.5927, 6),
    "noise_perc8": lambda: _noise(400, 400, 0.4073, 7),
    "noise_070": lambda: _noise(300, 517, 0.70, 8),
    "noise_big": lambda: _noise(1027, 1500, 0.55, 9),
}


def _png(name):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    path = INPUTS / f"{name}.png"
    if not path.exists():
        pytest.skip(f"{path} is missing")
    return np.ascontiguousarray(np.array(Image.open(path).convert("RGB")))


def _red(img):
    return (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)


def _opencv(img, connectivity):
    """OpenCV's labels (0 background, 1..n) and n."""
    n, lab = cv2.connectedComponents(_red(img).astype(np.uint8),
                                     connectivity=connectivity,
                                     ltype=cv2.CV_32S)
    return lab, n - 1


def _canonical_from(cv_lab, n):
    """The canonical label map by the rule alone, from OpenCV's partition:
    every pixel of a component takes the component's minimum linear index
    x*height + y; -1 off the components."""
    w, h = cv_lab.shape
    lin = np.arange(w * h, dtype=np.int64).reshape(w, h)
    lowest = np.full(n + 1, w * h, np.int64)
    np.minimum.at(lowest, cv_lab.ravel(), lin.ravel())
    out = lowest[cv_lab]
    out[cv_lab == 0] = -1
    return out.astype(np.int32)


def _assert_labels(label, n_blobs, img, connectivity):
    cv_lab, n = _opencv(img, connectivity)
    assert n_blobs == n, f"{n_blobs} blobs, OpenCV counts {n}"
    np.testing.assert_array_equal(label, _canonical_from(cv_lab, n))


def _dilation_depth(img, seed, connectivity):
    """BFS levels from seed by whole-array dilations: level k+1 is every
    unreached red pixel next to level k. No queue, no neighbour table."""
    red = _red(img)
    depth = np.full(red.shape, -1, np.int32)
    front = np.zeros(red.shape, bool)
    front[seed] = True
    depth[seed] = 0
    level = 0
    while front.any():
        grown = np.zeros(red.shape, bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) == (0, 0) or (connectivity == 4 and dx and dy):
                    continue
                src = front[max(0, -dx):red.shape[0] - max(0, dx),
                            max(0, -dy):red.shape[1] - max(0, dy)]
                grown[max(0, dx):red.shape[0] - max(0, -dx),
                      max(0, dy):red.shape[1] - max(0, -dy)] |= src
        front = grown & red & (depth < 0)
        level += 1
        depth[front] = level
    return depth


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("name", sorted(SCENES))
def test_ch05_label_oracle_equals_opencv(name):
    img = SCENES[name]()
    label, n_blobs = cpu_label_components(img)
    _assert_labels(label, n_blobs, img, 8)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("name", sorted(SCENES))
def test_ch08_label_oracle_equals_opencv(name, connectivity):
    img = SCENES[name]()
    label, n_blobs = cpu_label(img, connectivity)
    _assert_labels(label, n_blobs, img, connectivity)


@pytest.mark.parametrize("name,blobs8", [("input_blobs", 2522),
                                         ("input_blocks", 21618)])
def test_label_oracles_equal_opencv_on_the_benchmark_inputs(name, blobs8):
    """The two committed inputs: the counts the READMEs quote, and both
    label oracles against OpenCV at full size (81 Mpx for input_blobs)."""
    img = _png(name)
    label, n_blobs = cpu_label_components(img)
    assert n_blobs == blobs8
    _assert_labels(label, n_blobs, img, 8)
    for connectivity in (4, 8):
        label, n_blobs = cpu_label(img, connectivity)
        _assert_labels(label, n_blobs, img, connectivity)


def test_the_two_connectivities_differ_where_they_should():
    """A guard on the test itself: on the checkerboard 8-connectivity
    sees one component and 4-connectivity one per red pixel, so a
    connectivity mix-up in the comparison could not pass both."""
    img = _checkerboard(64, 97)
    assert cpu_label(img, 8)[1] == 1
    assert cpu_label(img, 4)[1] == int(_red(img).sum())


# ----------------------------------------------------------------- fills

def _seeds(img, k=3):
    """Up to k red pixels spread over the image (first, middle, last)."""
    xs, ys = np.nonzero(_red(img))
    if xs.size == 0:
        return []
    picks = sorted({0, xs.size // 2, xs.size - 1})[:k]
    return [(int(xs[i]), int(ys[i])) for i in picks]


FILL_SCENES = sorted(set(SCENES) - {"blank", "noise_big"})


@pytest.mark.parametrize("oracle,connectivity", [(cpu_flood_fill, 4),
                                                 (cpu_flood_fill_8, 8)])
@pytest.mark.parametrize("name", FILL_SCENES)
def test_fill_oracles_equal_opencv_and_the_dilation_bfs(name, oracle,
                                                        connectivity):
    img = SCENES[name]()
    cv_lab, _ = _opencv(img, connectivity)
    for seed in _seeds(img):
        visited, depth, levels, filled = oracle(img, *seed)
        want = cv_lab == cv_lab[seed]
        np.testing.assert_array_equal(visited.astype(bool), want)
        assert filled == int(want.sum())
        ref = _dilation_depth(img, seed, connectivity)
        np.testing.assert_array_equal(depth, ref)
        assert levels == int(ref.max()) + 1
