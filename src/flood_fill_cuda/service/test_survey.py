"""HTTP + logic tests for DEEP FIELD (the star-count game). Real GPU via
the lifespan warmup; the leaderboard's truth comes from the real kernel,
so we assert the honest invariant (true component count <= stars placed)
and the ranking/persistence behavior.

Run:  uv run pytest src/flood_fill_cuda/service/test_survey.py -v
"""

import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from .app import app, SCAN_MAGIC, SCAN_HEADER_FMT
from . import survey

SCAN_HEADER_SIZE = struct.calcsize(SCAN_HEADER_FMT)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_generator_truth_is_the_kernel(client):
    # roll a fresh field and read its stored challenge
    r = client.post("/api/challenge/new?difficulty=scout")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    cid = r.headers["X-Challenge-Id"]
    w = int(r.headers["X-Width"])
    h = int(r.headers["X-Height"])
    assert (w, h) == (survey.DIFFICULTIES["scout"]["w"],
                      survey.DIFFICULTIES["scout"]["h"])

    lb = client.get("/api/leaderboard").json()
    assert lb["challenge_id"] == cid
    assert lb["players"] == 0            # fresh field, board reset

    # the served PNG scanned by /api/scan must yield the SAME component
    # count the game will score against (truth = the kernel, not placed)
    png = r.content
    scan = client.post("/api/scan", content=png)
    assert scan.status_code == 200
    n_blobs = struct.unpack(SCAN_HEADER_FMT,
                            scan.content[:SCAN_HEADER_SIZE])[4]
    assert n_blobs > 0
    # touching stars merge: true count never exceeds stars placed
    assert n_blobs <= survey.DIFFICULTIES["scout"]["n"]


def test_guess_scores_and_ranks(client):
    client.post("/api/challenge/new?difficulty=scout")
    truth = survey._load()["challenge"]["true_count"]

    far = client.post("/api/guess",
                      json={"name": "FAR", "guess": truth + 500}).json()
    assert far["true_count"] == truth
    assert far["your_error"] == 500
    assert far["your_rank"] == 1        # only player so far

    near = client.post("/api/guess",
                       json={"name": "NEAR", "guess": truth + 1}).json()
    assert near["your_error"] == 1
    assert near["your_rank"] == 1       # closest now leads
    assert near["players"] == 2
    names = [row["name"] for row in near["leaderboard"]]
    assert names[:2] == ["NEAR", "FAR"]


def test_guess_validation_and_name_fallback(client):
    client.post("/api/challenge/new?difficulty=scout")
    assert client.post("/api/guess", json={"guess": -3}).status_code == 400
    assert client.post("/api/guess", json={"name": "x"}).status_code == 400
    blank = client.post("/api/guess", json={"name": "   ", "guess": 10}).json()
    assert blank["leaderboard"][0]["name"] == "ANON"


def test_new_field_resets_board(client):
    client.post("/api/challenge/new?difficulty=scout")
    client.post("/api/guess", json={"name": "A", "guess": 5})
    assert client.get("/api/leaderboard").json()["players"] == 1
    client.post("/api/challenge/new?difficulty=scout")
    assert client.get("/api/leaderboard").json()["players"] == 0


def test_bad_difficulty_rejected(client):
    assert client.post("/api/challenge/new?difficulty=nope").status_code == 400
