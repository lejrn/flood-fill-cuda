"""Synthetic single-blob test images for the flood fill benchmark.

State encoding matches ``kernels``: 0 background, 1 red (fillable).
Every generator returns ``(grid, seed)`` where ``grid`` is a uint8 HxW array
containing exactly one 4-connected red blob and ``seed`` is a (row, col)
red pixel from which the fill starts.
"""

import cv2
import numpy as np

from .kernels import RED


def make_disc(size: int, rng: np.random.Generator):
    """Filled circle covering ~64% of the image. Shortest geodesics."""
    c = size / 2.0
    dy = (np.arange(size, dtype=np.float32) - c)[:, None]
    dx = (np.arange(size, dtype=np.float32) - c)[None, :]
    grid = (dy * dy + dx * dx <= (0.45 * size) ** 2).astype(np.uint8)
    return grid, (size // 2, size // 2)


def make_amoeba(size: int, rng: np.random.Generator):
    """Star-shaped blob with a wavy sinusoidal boundary (always connected)."""
    c = size / 2.0
    dy = (np.arange(size, dtype=np.float32) - c)[:, None]
    dx = (np.arange(size, dtype=np.float32) - c)[None, :]
    r = np.hypot(dy, dx)
    theta = np.arctan2(dy, dx)
    r0 = 0.34 * size
    boundary = np.full_like(theta, r0)
    # amplitudes sum to < 1, so the boundary radius stays positive and the
    # region stays star-shaped around the centre => 4-connected
    for k, amp in ((3, 0.16), (7, 0.10), (13, 0.06)):
        boundary += r0 * amp * np.sin(k * theta + rng.uniform(0.0, 2.0 * np.pi))
    grid = (r <= boundary).astype(np.uint8)
    return grid, (size // 2, size // 2)


def make_spiral(size: int, rng: np.random.Generator):
    """Archimedean spiral arm: one blob with a very long geodesic path.

    Worst case for level-synchronous BFS: the wavefront must crawl along
    the whole arm even though the blob is compact on screen.
    """
    c = size // 2
    turns = 10
    R = 0.47 * size
    theta_max = 2.0 * np.pi * turns
    a = R / theta_max  # r = a * theta
    spacing = R / turns  # radial distance between arm centrelines
    half_t = max(2, int(round(0.22 * spacing)))  # arm half-thickness < gap/2

    # sample uniformly in arc length (s ~ theta^2/2 for an Archimedean spiral)
    arc_len = 0.5 * a * theta_max**2
    n = max(1000, int(3 * arc_len))
    theta = np.sqrt(np.linspace(0.0, theta_max**2, n))
    r = a * theta
    y = np.clip(np.round(c + r * np.sin(theta)).astype(np.int64), 0, size - 1)
    x = np.clip(np.round(c + r * np.cos(theta)).astype(np.int64), 0, size - 1)

    grid = np.zeros((size, size), dtype=np.uint8)
    grid[y, x] = RED
    k = 2 * half_t + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    grid = cv2.dilate(grid, kernel)
    return grid, (c, c)


GENERATORS = {
    "amoeba": make_amoeba,
    "spiral": make_spiral,
    "disc": make_disc,
}


def make_blob(shape: str, size: int, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    grid, seed = GENERATORS[shape](size, rng)
    assert grid[seed] == RED, f"seed {seed} is not on the blob"
    return grid, seed
