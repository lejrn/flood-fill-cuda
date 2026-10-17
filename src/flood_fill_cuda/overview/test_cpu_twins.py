"""
The CPU twins (cpu_twins) must equal their oracles, array for array, on
buffers they keep between calls, with the reset before the clock or on
it. CPU only.

Run:

    uv run pytest src/flood_fill_cuda/overview/test_cpu_twins.py -v
"""

import numpy as np
import pytest

from . import cpu_twins
from ..chapters.ch04_gpu_2blob_nblock import scenes as ch04_scenes
from ..shared import scenes as shared_scenes


def _scenes():
    sq, sx, sy = shared_scenes.square_scene(64, 48, 30, 20)
    two, seeds = ch04_scenes.two_squares_scene(96, 64, 24, 24, gap=8)
    rng = np.random.default_rng(7)
    noise = np.full((70, 50, 3), 255, np.uint8)
    noise[rng.random((70, 50)) < 0.55] = (255, 0, 0)
    noise[10, 10] = (255, 0, 0)
    return [("square", sq, (sx, sy)),
            ("two_squares", two, tuple(int(v) for v in seeds[0])),
            ("noise", noise, (10, 10))]


@pytest.mark.parametrize("kind", sorted(cpu_twins.KINDS))
@pytest.mark.parametrize("on_clock", [False, True])
def test_twin_equals_oracle_every_call(kind, on_clock):
    """Three calls on one buffer set per scene, and one buffer set shared
    by the scenes of one shape: every call equals the oracle (a stale
    buffer would show on the second call)."""
    for name, img, seed in _scenes():
        bufs = cpu_twins.Buffers(img.shape[0], img.shape[1],
                                 tuple(cpu_twins.KINDS))
        tw = cpu_twins.Twin(kind, img, seed, bufs, on_clock)
        for _ in range(3):
            ok, _ = cpu_twins.check_against_oracle(tw)
            assert ok, (name, kind, on_clock)


def test_twins_share_buffers():
    """Twins of one image on one buffer set, interleaved: each still
    equals its oracle (each resets what it reads)."""
    name, img, seed = _scenes()[2]
    bufs = cpu_twins.Buffers(img.shape[0], img.shape[1],
                             tuple(cpu_twins.KINDS))
    twins = [cpu_twins.Twin(k, img, seed, bufs, on)
             for k in sorted(cpu_twins.KINDS) for on in (False, True)]
    for _ in range(2):
        for tw in twins:
            ok, _ = cpu_twins.check_against_oracle(tw)
            assert ok, (tw.kind, tw.reset_on_clock)


def test_reset_scope():
    """prepare() resets only when the reset is off the clock; call()
    resets only when it is on it."""
    name, img, seed = _scenes()[0]
    bufs = cpu_twins.Buffers(img.shape[0], img.shape[1], ("label",))
    on = cpu_twins.Twin("label", img, None, bufs, True)
    off = cpu_twins.Twin("label", img, None, bufs, False)
    bufs.label[...] = 5                  # stale labels: nothing unlabelled
    off.prepare()
    assert (bufs.label == -1).all()
    bufs.label[...] = 5
    on.prepare()                         # no reset before the clock
    assert (bufs.label == 5).all()
    assert on.call() == 1                # the reset is inside the call


def test_fill8_keeps_no_depth():
    assert cpu_twins.KINDS["fill8"][1] == ("visited",)
    assert cpu_twins.KINDS["bfs8"][1] == ("visited", "depth")
    bufs = cpu_twins.Buffers(8, 8, ("fill8",))
    assert bufs.depth is None and bufs.label is None
