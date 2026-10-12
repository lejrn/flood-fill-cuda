"""Shared helpers for the TTS scripts: parse a script into beats, write timing and words."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "script.md"

_BEAT_RE = re.compile(r"^## beat (\S+)\s*$", re.M)
_FENCE_RE = re.compile(r"```text\n(.*?)\n```", re.S)


@dataclass
class Beat:
    name: str
    text: str


def load_beats(path: Path = SCRIPT) -> list[Beat]:
    """Return the beats in script order: heading name + first ```text block."""
    src = path.read_text(encoding="utf-8")
    heads = list(_BEAT_RE.finditer(src))
    beats: list[Beat] = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(src)
        body = src[h.end():end]
        m = _FENCE_RE.search(body)
        if not m:
            raise ValueError(f"beat {h.group(1)} has no ```text block")
        beats.append(Beat(h.group(1), " ".join(m.group(1).split())))
    return beats


def trim_bounds(audio: np.ndarray, sr: int, thresh_db: float = -45.0, keep: float = 0.12) -> tuple[int, int]:
    """(lo, hi) sample bounds that cut leading/trailing silence, keeping `keep` s of air."""
    if audio.size == 0:
        return 0, 0
    amp = np.abs(audio)
    floor = amp.max() * (10 ** (thresh_db / 20))
    loud = np.flatnonzero(amp > floor)
    if loud.size == 0:
        return 0, audio.size
    pad = int(sr * keep)
    return max(0, loud[0] - pad), min(audio.size, loud[-1] + pad)


def trim_edges(audio: np.ndarray, sr: int, thresh_db: float = -45.0, keep: float = 0.12) -> np.ndarray:
    """Cut leading/trailing silence, keeping `keep` seconds of air on each side."""
    lo, hi = trim_bounds(audio, sr, thresh_db, keep)
    return audio[lo:hi]


def script_path(arg: str | None) -> Path:
    """--script value -> path; relative paths are taken from narration/."""
    if not arg:
        return SCRIPT
    p = Path(arg)
    return p if p.is_absolute() else HERE / p


def estimate_words(text: str, seconds: float) -> list:
    """[word, start, end] spread over `seconds` by character count: a stand-in
    for voices without timestamps (ElevenLabs via the connector)."""
    words = text.split()
    weights = [len(w) + 1 for w in words]
    total = float(sum(weights)) or 1.0
    rows, t = [], 0.0
    for w, n in zip(words, weights):
        d = seconds * n / total
        rows.append([w, round(t, 3), round(t + d, 3)])
        t += d
    return rows


def write_words(out_dir: Path, words: dict, source: str) -> Path:
    """words.json: per beat, [word, start, end] in seconds from the trimmed wav's start."""
    p = out_dir / "words.json"
    p.write_text(json.dumps({"source": source, "beats": words}, indent=1), encoding="utf-8")
    return p


def read_words(out_dir: Path) -> dict:
    p = out_dir / "words.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))["beats"]


def write_timing(out_dir: Path, voice: str, durations: dict[str, float], gap: float) -> Path:
    """Write timing.json: per-beat start/end on one timeline with `gap` seconds between beats."""
    t = 0.0
    rows = []
    for name, d in durations.items():
        rows.append({"beat": name, "start": round(t, 3), "end": round(t + d, 3), "seconds": round(d, 3)})
        t += d + gap
    payload = {"voice": voice, "gap": gap, "total": round(t - gap, 3), "beats": rows}
    p = out_dir / "timing.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
