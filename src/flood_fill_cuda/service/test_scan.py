"""HTTP-layer tests for /api/scan — the seedless discovery endpoint,
now backed by ch06's run-table connected components.

Same philosophy as test_api.py: real GPU via the lifespan warmup, real
TestClient requests, no mocking; the track map is checked EXACTLY against
the CPU oracle's canonical components. That oracle is unchanged from the
ch05 era on purpose — swapping the kernel underneath must not move a
single label, which is the whole reason ch06 kept ch05's canonicalisation
rule.

Run:

    uv run pytest src/flood_fill_cuda/service/test_scan.py -v
"""

import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from .app import SCAN_HEADER_FMT, SCAN_MAGIC, SCAN_FLAG_AMPLIFIED, app
from ..chapters.ch05_gpu_nblob_nblock.cpu_oracle import cpu_label_components

SCAN_HEADER_SIZE = struct.calcsize(SCAN_HEADER_FMT)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _two_blob_png(size=96):
    """Two separated squares — n_blobs must come back as exactly 2."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[10:34, 10:34] = (200, 30, 30, 255)
    rgba[60:88, 56:90] = (200, 30, 30, 255)
    return rgba


def _decode(content):
    magic, width, height, steps, n_blobs, flags = struct.unpack(
        SCAN_HEADER_FMT, content[:SCAN_HEADER_SIZE])
    assert magic == SCAN_MAGIC
    n = width * height
    off = SCAN_HEADER_SIZE
    sweep = np.frombuffer(content, dtype="<u2", count=n,
                          offset=off).reshape(height, width)
    off += 2 * n
    track = np.frombuffer(content, dtype="<u2", count=n,
                          offset=off).reshape(height, width)
    off += 2 * n
    seeds = np.frombuffer(content, dtype="<u4", count=2 * n_blobs,
                          offset=off).reshape(n_blobs, 2)
    off += 8 * n_blobs
    assert off == len(content)
    return dict(width=width, height=height, steps=steps, flags=flags,
                n_blobs=n_blobs, sweep=sweep, track=track, seeds=seeds)


def test_scan_two_blobs_matches_oracle(client):
    rgba = _two_blob_png()
    r = client.post("/api/scan", content=_png(rgba))
    assert r.status_code == 200
    d = _decode(r.content)
    assert d["n_blobs"] == 2

    # rebuild the kernel-side image and ask the oracle for ground truth
    mask = rgba[:, :, 3] >= 128
    img = np.full((mask.shape[1], mask.shape[0], 3), 255, dtype=np.uint8)
    img[mask.T] = (255, 0, 0)
    label, n = cpu_label_components(img)
    assert n == 2
    roots = np.unique(label[label >= 0])
    expect = np.zeros(label.shape, dtype=np.int64)
    m = label >= 0
    expect[m] = np.searchsorted(roots, label[m]) + 1
    np.testing.assert_array_equal(d["track"], expect.T)

    # seeds: row i belongs to track i+1, decoded from the canonical root,
    # and every seed is a painted pixel
    height = img.shape[1]
    np.testing.assert_array_equal(
        d["seeds"], np.stack([roots // height, roots % height], axis=1))
    for x, y in d["seeds"]:
        assert mask[y, x]

    # the sweep field covers exactly the painted pixels, and nothing else
    np.testing.assert_array_equal(d["sweep"] > 0, mask)
    assert 1 <= d["sweep"].max() <= d["steps"]
    assert int(r.headers["X-Filled"]) == int(mask.sum())
    assert float(r.headers["X-Kernel-Ms"]) > 0
    assert float(r.headers["X-Njit-Ms"]) > 0
    # ch06's own phases, not ch05's
    assert "merge" in r.headers["X-Phase-Ms"]
    assert "paint" in r.headers["X-Phase-Ms"]
    # runs are a real count, and strictly cheaper than the pixels they cover
    runs = int(r.headers["X-Runs"])
    assert 0 < runs < int(mask.sum())
    # every link retires exactly one root
    assert int(r.headers["X-Unions"]) == runs - d["n_blobs"]


def test_scan_empty_canvas_is_valid(client):
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    r = client.post("/api/scan", content=_png(rgba))
    assert r.status_code == 200
    d = _decode(r.content)
    assert d["n_blobs"] == 0
    assert (d["track"] == 0).all()
    assert d["seeds"].shape == (0, 2)


def test_scan_sweep_is_left_to_right(client):
    """The sweep is the kernel's row-major scan order, and the kernel
    image is the canvas transposed — so the bucket a pixel lands in is a
    function of its COLUMN only, and it increases to the right. This is
    what makes the reveal a scan bar rather than an arbitrary shuffle."""
    rgba = _two_blob_png()
    d = _decode(client.post("/api/scan", content=_png(rgba)).content)
    sweep, mask = d["sweep"], rgba[:, :, 3] >= 128
    cols = np.nonzero(mask.any(axis=0))[0]
    per_col = [np.unique(sweep[:, c][mask[:, c]]) for c in cols]
    assert all(len(u) == 1 for u in per_col)          # one bucket per column
    firsts = [int(u[0]) for u in per_col]
    assert firsts == sorted(firsts)                   # and non-decreasing


def test_scan_amp_reports_at_scale_timing(client):
    """?amp=1 adds an upscaled timing run: same shape, ~250x the pixels,
    so a small stroke reports a number that isn't launch-bound."""
    rgba = _two_blob_png()
    plain = client.post("/api/scan", content=_png(rgba))
    amped = client.post("/api/scan?amp=1", content=_png(rgba))
    assert plain.status_code == amped.status_code == 200
    assert "X-Amplified-Filled" not in plain.headers
    assert _decode(amped.content)["flags"] & SCAN_FLAG_AMPLIFIED
    filled = int(amped.headers["X-Filled"])
    assert int(amped.headers["X-Amplified-Filled"]) > filled * 10
    assert int(amped.headers["X-Amplified-Runs"]) > int(amped.headers["X-Runs"])
    assert float(amped.headers["X-Amplified-Kernel-Ms"]) > 0
    # the payload itself is the real-size scan either way
    assert len(plain.content) == len(amped.content)


def test_scan_rejects_oversize_and_garbage(client):
    huge = np.zeros((2100, 2100, 4), dtype=np.uint8)   # 4.41M px > 4M cap
    huge[0, 0] = (200, 30, 30, 255)
    r = client.post("/api/scan", content=_png(huge))
    assert r.status_code == 413
    assert client.post("/api/scan", content=b"").status_code == 400
    assert client.post("/api/scan", content=b"not a png").status_code == 400


def test_scan_reports_a_cold_clock(client):
    """The first launch after an idle GPU is flagged, because this laptop
    drops to ~700 MHz and won't spin up for a millisecond kernel — the
    same effect chapter 6's benchmark spins the clock up to avoid. Two
    scans back to back: the second cannot be cold."""
    png = _png(_two_blob_png())
    client.post("/api/scan", content=png)
    warm = client.post("/api/scan", content=png)
    assert warm.headers["X-Cold"] == "0"
