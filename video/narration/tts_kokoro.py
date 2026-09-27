"""Render every beat of script.md with Kokoro (local, free) into out/kokoro/.

Usage (from video/):
    uv run narration/tts_kokoro.py [--voice am_onyx] [--gap 0.6]

Writes one wav per beat, a joined narration.wav with `gap` seconds of
silence between beats, and timing.json for the assembly step.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep the model cache inside the project, not in ~/.cache.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from common import HERE, load_beats, write_timing  # noqa: E402

SR = 24_000


def trim_edges(audio: np.ndarray, sr: int, thresh_db: float = -45.0, keep: float = 0.12) -> np.ndarray:
    """Cut leading/trailing silence, keeping `keep` seconds of air on each side."""
    if audio.size == 0:
        return audio
    amp = np.abs(audio)
    floor = amp.max() * (10 ** (thresh_db / 20))
    loud = np.flatnonzero(amp > floor)
    if loud.size == 0:
        return audio
    pad = int(sr * keep)
    lo = max(0, loud[0] - pad)
    hi = min(audio.size, loud[-1] + pad)
    return audio[lo:hi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="am_onyx", help="deep male; af_heart was the first cut")
    ap.add_argument("--gap", type=float, default=0.4, help="silence between beats, seconds")
    ap.add_argument("--speed", type=float, default=1.1, help="Kokoro reads slowly at 1.0")
    ap.add_argument("--out", default="kokoro", help="folder under out/ (the VOICE the scenes read)")
    args = ap.parse_args()

    # This laptop has 6 GB of RAM. Loading the 327 MB checkpoint the normal
    # way peaks at 1.4 GB and gets the process OOM-killed when other work
    # is open, so the checkpoint is memory-mapped (file-backed pages the
    # kernel can drop) and torch stays single-threaded. The CPU is fast
    # enough: the whole script runs in about a minute.
    import torch

    _torch_load = torch.load

    def _mmap_load(*a, **k):
        k.setdefault("mmap", True)
        return _torch_load(*a, **k)

    torch.load = _mmap_load
    torch.set_num_threads(1)
    from kokoro import KPipeline

    out = HERE.parent / "out" / args.out
    out.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code="a", device="cpu")

    durations: dict[str, float] = {}
    joined: list[np.ndarray] = []
    silence = np.zeros(int(SR * args.gap), dtype=np.float32)
    for beat in load_beats():
        audio = np.concatenate([a for _, _, a in pipe(beat.text, voice=args.voice, speed=args.speed)])
        audio = trim_edges(audio.astype(np.float32), SR)
        sf.write(out / f"{beat.name}.wav", audio, SR)
        durations[beat.name] = len(audio) / SR
        joined += [audio, silence]
        print(f"{beat.name:20s} {durations[beat.name]:5.2f} s")

    track = np.concatenate(joined[:-1])
    sf.write(out / "narration.wav", track, SR)
    timing = write_timing(out, f"kokoro:{args.voice}", durations, args.gap)
    print(f"total {len(track) / SR:.2f} s  ->  {out / 'narration.wav'}\n{timing}")


if __name__ == "__main__":
    main()
