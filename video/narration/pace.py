"""Slow a narration down between sentences, so the screen can be read.

Usage (from video/):
    uv run narration/pace.py --src kokoro_chain --out kokoro_paced_chain --script script_chain.md

Reads out/<src>/ (beat wavs, words.json, timing.json), and writes
out/<out>/ in the same form with every pause after a sentence end (. ? !)
stretched to --sentence seconds and after a colon or semicolon to
--clause seconds (pauses already that long are left alone; commas are
never touched). Silence is inserted at the quietest point of the
existing pause, and every later word time shifts by what was inserted,
so the scenes' word cues stay on their words.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import HOP, boundaries, energy  # noqa: E402
from common import HERE, load_beats, read_words, script_path, write_timing, write_words  # noqa: E402


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def word_map(script_words: list[str], rows: list[list]) -> dict[int, int]:
    """script word index -> words.json row index, where they match."""
    a = [_norm(w) for w in script_words]
    b = [_norm(r[0]) for r in rows]
    out = {}
    for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            out[blk.a + k] = blk.b + k
    return out


def pause_at(audio: np.ndarray, sr: int, t_a: float, t_b: float, rel_db: float = -36.0) -> tuple[float, float]:
    """(insertion time, current pause length) for the quietest point in [t_a, t_b]."""
    e = energy(audio, sr)
    i0, i1 = max(0, int(t_a / HOP)), min(len(e), max(int(t_a / HOP) + 1, int(t_b / HOP)))
    seg = e[i0:i1]
    if seg.size == 0:
        return t_b, 0.0
    k = i0 + int(np.argmin(seg))
    thr = e.max() * 10 ** (rel_db / 20)
    if e[k] >= thr:
        return k * HOP, 0.0
    a = b = k
    while a > 0 and e[a - 1] < thr:
        a -= 1
    while b < len(e) - 1 and e[b + 1] < thr:
        b += 1
    return (a + b + 1) / 2 * HOP, (b + 1 - a) * HOP


def pace_beat(text: str, audio: np.ndarray, sr: int, rows: list[list], sentence: float, clause: float):
    words = text.split()
    wmap = word_map(words, rows)
    cuts = []                                     # (sample, extra seconds, row index)
    for wi, mark in boundaries(text):
        target = sentence if mark in ".?!" else clause if mark in ":;" else 0.0
        if target <= 0 or wi not in wmap or wi + 1 not in wmap:
            continue
        r = wmap[wi]
        if wmap[wi + 1] != r + 1:
            continue
        w_start, w_end = rows[r][1], rows[r][2]
        nxt = rows[r + 1][1]
        t, have = pause_at(audio, sr, w_start + 0.5 * (w_end - w_start), nxt + 0.05)
        extra = round(max(0.0, target - have), 2)
        if extra > 0:
            cuts.append((int(t * sr), extra, r))
    pieces, last = [], 0
    new_rows = [list(r) for r in rows]
    for s, extra, r in cuts:
        pieces += [audio[last:s], np.zeros(int(extra * sr), dtype=audio.dtype)]
        last = s
        t = s / sr
        for row in new_rows[r + 1:]:
            row[1] = round(row[1] + extra, 3)
            row[2] = round(row[2] + extra, 3)
        if new_rows[r][2] > t:
            new_rows[r][2] = round(new_rows[r][2] + extra, 3)
    pieces.append(audio[last:])
    return np.concatenate(pieces), new_rows, sum(e for _, e, _ in cuts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder under out/ to read")
    ap.add_argument("--out", required=True, help="folder under out/ to write")
    ap.add_argument("--script", default=None, help="narration file under narration/ (default script.md)")
    ap.add_argument("--sentence", type=float, default=0.9, help="pause after . ? ! (seconds)")
    ap.add_argument("--clause", type=float, default=0.45, help="pause after : ; (seconds)")
    args = ap.parse_args()

    src, out = HERE.parent / "out" / args.src, HERE.parent / "out" / args.out
    out.mkdir(parents=True, exist_ok=True)
    timing = json.loads((src / "timing.json").read_text(encoding="utf-8"))
    words = read_words(src)
    texts = {b.name: b.text for b in load_beats(script_path(args.script))}
    durations, new_words, joined, sr = {}, {}, [], None
    for row in timing["beats"]:
        name = row["beat"]
        audio, sr = sf.read(str(src / f"{name}.wav"), dtype="float32")
        paced, rows, added = pace_beat(texts[name], audio, sr, words[name], args.sentence, args.clause)
        sf.write(out / f"{name}.wav", paced, sr)
        durations[name] = len(paced) / sr
        new_words[name] = rows
        joined += [paced, np.zeros(int(sr * timing["gap"]), dtype=np.float32)]
        print(f"{name:18s} {len(audio) / sr:6.2f} s -> {durations[name]:6.2f} s (+{added:.2f})")
    sf.write(out / "narration.wav", np.concatenate(joined[:-1]), sr)
    write_timing(out, f"{timing['voice']}, paced {args.sentence}/{args.clause} s", durations, timing["gap"])
    write_words(out, new_words, f"paced from {args.src}")
    print(f"total {sum(durations.values()):.1f} s of speech -> {out}")


if __name__ == "__main__":
    main()
