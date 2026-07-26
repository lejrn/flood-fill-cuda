"""DEEP FIELD: the star-count guessing game's server state.

A shared challenge — one procedurally-generated star field that every
player guesses — plus a persistent leaderboard of names and guesses.

The field is generated from a seed (so the server can regenerate the
exact bytes any player scans), but the TRUTH is never the generator's
bookkeeping: it is `engine.discover(...).n_blobs`, the real ch06 kernel
run on the field. That matters because stars that TOUCH merge into one
component, so the honest answer differs from "stars placed" — which is
exactly the counting mistake a human eye makes, and the whole game.

State (current challenge + guesses) persists to a small JSON file so the
leaderboard survives restarts; the field PNG is not stored — it is
regenerated from the seed on demand and cached in memory.
"""

import io
import json
import os
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from . import engine
from ..shared import results_paths

STATE_DIR = results_paths.results_dir("service", "state")
STATE_PATH = os.path.join(STATE_DIR, "survey_state.json")

# (stars placed, field width, height). Placed != revealed count: some
# stars overlap and merge, and the denser tiers merge more — part of the
# challenge.
DIFFICULTIES = {
    "scout":     dict(n=120,  w=1100, h=760),
    "surveyor":  dict(n=550,  w=1400, h=900),
    "deepfield": dict(n=2400, w=1700, h=1050),
}
DEFAULT_DIFFICULTY = "surveyor"
MAX_NAME = 18

_lock = threading.RLock()
_png_cache = {}          # challenge id -> PNG bytes (regenerated from seed)


# --------------------------------------------------------------- field gen
def _stamp_disk(mask, cx, cy, r):
    h, w = mask.shape
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r


def _field_mask(seed, n, w, h):
    """(h, w) bool star mask. Radii follow a steep small-favoring curve —
    many faint pinpricks, a few bright discs, like a real exposure."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=bool)
    cx = rng.integers(0, w, n)
    cy = rng.integers(0, h, n)
    radii = (1 + (rng.random(n) ** 3) * 13).astype(np.int64)   # 1..14, skewed
    for i in range(n):
        _stamp_disk(mask, int(cx[i]), int(cy[i]), int(radii[i]))
    return mask


def _mask_to_png(mask):
    """Crisp stars: bright blue-white, alpha 255 on the star bodies, fully
    transparent elsewhere. The mask is exactly alpha>=128, so the client's
    /api/scan rebuilds the identical component set the server measured. The
    glow is added on the client (CSS), never here — nothing decorative may
    perturb the mask."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask] = (206, 224, 255, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _field_png(challenge):
    cid = challenge["id"]
    if cid not in _png_cache:
        mask = _field_mask(challenge["seed"], challenge["n_placed"],
                           challenge["width"], challenge["height"])
        _png_cache[cid] = _mask_to_png(mask)
    return _png_cache[cid]


# --------------------------------------------------------------- persistence
def _load():
    if not os.path.exists(STATE_PATH):
        return {"challenge": None, "guesses": []}
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)          # atomic


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_challenge(difficulty, seed):
    cfg = DIFFICULTIES.get(difficulty, DIFFICULTIES[DEFAULT_DIFFICULTY])
    mask = _field_mask(seed, cfg["n"], cfg["w"], cfg["h"])
    # THE TRUTH: the real kernel on the real field, not the placed count.
    truth = engine.discover(mask).n_blobs
    cid = uuid.uuid4().hex[:12]
    challenge = {
        "id": cid, "seed": int(seed), "difficulty": difficulty,
        "width": cfg["w"], "height": cfg["h"], "n_placed": cfg["n"],
        "true_count": int(truth), "created": _now(),
    }
    _png_cache[cid] = _mask_to_png(mask)
    return challenge


# --------------------------------------------------------------- public API
def ensure_challenge():
    """The current shared challenge, creating one on first use. Runs the
    kernel when it generates — call from the GPU executor."""
    with _lock:
        state = _load()
        if state["challenge"] is None:
            state["challenge"] = _make_challenge(
                DEFAULT_DIFFICULTY, _seed_from_time())
            state["guesses"] = []
            _save(state)
        return state["challenge"]


def new_challenge(difficulty=DEFAULT_DIFFICULTY):
    """Roll a fresh field for everyone; the leaderboard resets to it. Runs
    the kernel — call from the GPU executor."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}")
    with _lock:
        challenge = _make_challenge(difficulty, _seed_from_time())
        _save({"challenge": challenge, "guesses": []})
        return challenge


def _seed_from_time():
    # a fresh, non-repeating field each roll; the seed is stored so the
    # exact bytes are reproducible afterwards
    return uuid.uuid4().int % (2 ** 32)


def field_png():
    """PNG bytes of the current field (regenerated from its seed)."""
    with _lock:
        return _field_png(ensure_challenge())


def _sorted_board(guesses):
    return sorted(guesses, key=lambda g: (g["error"], g["ts"]))


def submit_guess(name, guess):
    """Score a guess against the CURRENT field's true count, store it, and
    return the reveal + the updated board. No GPU needed (truth is already
    computed and stored)."""
    name = (str(name).strip()[:MAX_NAME] or "ANON")
    guess = int(guess)
    if guess < 0:
        raise ValueError("guess must be non-negative")
    with _lock:
        state = _load()
        challenge = state["challenge"]
        if challenge is None:                     # someone guessed pre-init
            challenge = ensure_challenge()
            state = _load()
        truth = challenge["true_count"]
        entry = {"name": name, "guess": guess,
                 "error": abs(guess - truth), "ts": _now()}
        state["guesses"].append(entry)
        _save(state)
        board = _sorted_board(state["guesses"])
        rank = board.index(entry) + 1
        return {
            "challenge_id": challenge["id"],
            "true_count": truth,
            "n_placed": challenge["n_placed"],
            "your_guess": guess,
            "your_error": entry["error"],
            "your_rank": rank,
            "players": len(board),
            "leaderboard": [{"name": g["name"], "guess": g["guess"],
                             "error": g["error"]} for g in board[:12]],
        }


def leaderboard():
    """Current board (names, guesses, errors) + how many have played. The
    truth is derivable from any entry, so the FRONTEND hides errors until a
    viewer has surveyed the field themselves."""
    with _lock:
        state = _load()
        challenge = state["challenge"]
        board = _sorted_board(state["guesses"])
        return {
            "challenge_id": challenge["id"] if challenge else None,
            "difficulty": challenge["difficulty"] if challenge else None,
            "players": len(board),
            "leaderboard": [{"name": g["name"], "guess": g["guess"],
                             "error": g["error"]} for g in board[:12]],
        }
