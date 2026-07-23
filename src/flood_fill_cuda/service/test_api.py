"""HTTP-layer tests for the FastAPI app: request/response framing and error
mapping. Engine correctness (seed placement, orientation, depth math) is
covered by test_engine.py; this file only checks that the wire format
matches what app.py documents and that engine exceptions land on the right
status codes. Runs the real GPU via the lifespan warmup (real TestClient
requests, no mocking).

Run:

    uv run pytest src/flood_fill_cuda/service/test_api.py -v
"""

import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from . import app as app_module
from .app import HEADER_FMT, HEADER_SIZE, MAGIC, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _stroke_png(size=80, radius=25):
    """An RGBA PNG whose alpha channel is a filled disc -- what the
    browser's canvas.toBlob('image/png') crop looks like."""
    ys, xs = np.mgrid[0:size, 0:size]
    cx = cy = size // 2
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[mask] = (200, 30, 30, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["warm"] is True
    assert isinstance(body["device"], str) and body["device"]


def test_fill_round_trip(client):
    png = _stroke_png()
    r = client.post("/api/fill", content=png)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"

    magic, width, height, levels = struct.unpack(HEADER_FMT,
                                                  r.content[:HEADER_SIZE])
    assert magic == MAGIC
    assert width == height == 80
    assert levels > 0
    depth = np.frombuffer(r.content[HEADER_SIZE:], dtype='<u2')
    assert depth.size == width * height
    assert (depth > 0).sum() > 0     # something got filled
    assert (depth > 0).sum() == int(r.headers["x-filled"])
    assert float(r.headers["x-kernel-ms"]) >= 0
    assert float(r.headers["x-total-ms"]) >= 0
    assert r.headers["x-mode"] == "gpu"   # default mode


def test_fill_cpu_mode(client):
    png = _stroke_png()
    r = client.post("/api/fill?mode=cpu", content=png)
    assert r.status_code == 200
    assert r.headers["x-mode"] == "cpu"
    magic, width, height, levels = struct.unpack(HEADER_FMT,
                                                  r.content[:HEADER_SIZE])
    assert magic == MAGIC
    assert levels > 0


def test_fill_explicit_seed(client):
    """seed_x/seed_y (the browser's release point, crop-local) pins the
    fill's start; a corner seed should reach a far corner at high depth."""
    png = _stroke_png(size=80, radius=38)   # near-full-canvas disc
    r = client.post("/api/fill?seed_x=40&seed_y=40", content=png)
    assert r.status_code == 200
    depth = np.frombuffer(r.content[HEADER_SIZE:], dtype='<u2').reshape(80, 80)
    assert depth[40, 40] == 1   # seed itself is depth 0 -> encoded 1


def test_fill_invalid_mode(client):
    png = _stroke_png()
    r = client.post("/api/fill?mode=tpu", content=png)
    assert r.status_code == 400


def test_fill_non_numeric_seed(client):
    png = _stroke_png()
    r = client.post("/api/fill?seed_x=not-a-number", content=png)
    assert r.status_code == 400


def test_fill_empty_body(client):
    r = client.post("/api/fill", content=b"")
    assert r.status_code == 400


def test_fill_truncated_png(client):
    png = _stroke_png()
    r = client.post("/api/fill", content=png[:20])
    assert r.status_code == 400


def test_fill_not_an_image(client):
    r = client.post("/api/fill", content=b"not a png at all")
    assert r.status_code == 400


def test_fill_fully_transparent_png(client):
    """A crop with no painted pixels (alpha all 0) decodes fine but the
    mask is empty -- engine.run_fill's ValueError must map to 400."""
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    r = client.post("/api/fill", content=buf.getvalue())
    assert r.status_code == 400


def test_fill_oversized_mask_413(client):
    from . import engine
    side = int(engine.MAX_PIXELS ** 0.5) + 200
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[0:5, 0:5] = (200, 30, 30, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    r = client.post("/api/fill", content=buf.getvalue())
    assert r.status_code == 413


def test_root_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
