"""HTTP-layer tests for /api/scan — the SKYWATCH seedless discovery
endpoint. Same philosophy as test_api.py: real GPU via the lifespan
warmup, real TestClient requests, no mocking; the track map is checked
EXACTLY against the ch05 CPU oracle's canonical components.

Run:

    uv run pytest src/flood_fill_cuda/service/test_scan.py -v
"""

import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from .app import SCAN_HEADER_FMT, SCAN_MAGIC, SCAN_FLAG_PROV, app
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
    magic, width, height, levels, n_blobs, flags = struct.unpack(
        SCAN_HEADER_FMT, content[:SCAN_HEADER_SIZE])
    assert magic == SCAN_MAGIC
    n = width * height
    off = SCAN_HEADER_SIZE
    depth = np.frombuffer(content, dtype="<u2", count=n,
                          offset=off).reshape(height, width)
    off += 2 * n
    track = np.frombuffer(content, dtype="<u2", count=n,
                          offset=off).reshape(height, width)
    off += 2 * n
    seeds = np.frombuffer(content, dtype="<u4", count=2 * n_blobs,
                          offset=off).reshape(n_blobs, 2)
    off += 8 * n_blobs
    prov = None
    if flags & SCAN_FLAG_PROV:
        prov = np.frombuffer(content, dtype="<u2", count=n,
                             offset=off).reshape(height, width)
        off += 2 * n
    assert off == len(content)
    return dict(width=width, height=height, levels=levels,
                n_blobs=n_blobs, depth=depth, track=track, seeds=seeds,
                prov=prov)


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

    # depth covers exactly the painted pixels
    assert (d["depth"] > 0).sum() == mask.sum()
    assert int(r.headers["X-Filled"]) == int(mask.sum())
    assert float(r.headers["X-Kernel-Ms"]) > 0
    assert float(r.headers["X-Njit-Ms"]) > 0
    assert "fill" in r.headers["X-Phase-Ms"]


def test_scan_empty_canvas_is_valid(client):
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    r = client.post("/api/scan", content=_png(rgba))
    assert r.status_code == 200
    d = _decode(r.content)
    assert d["n_blobs"] == 0
    assert (d["track"] == 0).all()
    assert d["seeds"].shape == (0, 2)


def test_scan_prov_flag_extends_payload(client):
    rgba = _two_blob_png()
    plain = client.post("/api/scan", content=_png(rgba))
    with_prov = client.post("/api/scan?prov=1", content=_png(rgba))
    assert plain.status_code == with_prov.status_code == 200
    d = _decode(with_prov.content)
    assert d["prov"] is not None
    # provisional labels cover the same pixels as the final tracks
    np.testing.assert_array_equal(d["prov"] > 0, d["track"] > 0)
    assert len(with_prov.content) > len(plain.content)


def test_scan_rejects_oversize_and_garbage(client):
    huge = np.zeros((2100, 2100, 4), dtype=np.uint8)   # 4.41M px > 4M cap
    huge[0, 0] = (200, 30, 30, 255)
    r = client.post("/api/scan", content=_png(huge))
    assert r.status_code == 413
    assert client.post("/api/scan", content=b"").status_code == 400
    assert client.post("/api/scan", content=b"not a png").status_code == 400
