"""Shared helpers for the TTS scripts: parse script.md into beats, write timing."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

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
