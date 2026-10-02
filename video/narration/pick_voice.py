"""Compare Kokoro voices on one sentence: median pitch (deeper = lower) and length.

    CUDA_VISIBLE_DEVICES= .venv/bin/python narration/pick_voice.py am_onyx am_adam ...

Writes out/voices/<voice>.wav for listening. Pitch is a plain autocorrelation
estimate on voiced 40 ms windows; good enough to rank bass against tenor.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf-cache"))

import numpy as np
import soundfile as sf

SR = 24_000
TEXT = ("Labelling one frame on the CPU takes twenty-four seconds. "
        "On the GPU, the same frame takes one and a half milliseconds.")


def median_pitch(audio: np.ndarray, sr: int) -> float:
    win, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 300), int(sr / 60)          # 60-300 Hz
    f0s = []
    for i in range(0, len(audio) - win, hop):
        x = audio[i:i + win]
        if np.sqrt(np.mean(x ** 2)) < 0.03:
            continue
        x = x - x.mean()
        ac = np.correlate(x, x, mode="full")[win - 1:]
        ac /= ac[0] + 1e-9
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else float("nan")


def main() -> int:
    import torch

    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "mmap": True})
    torch.set_num_threads(1)
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a", device="cpu")
    out = Path(__file__).resolve().parents[1] / "out" / "voices"
    out.mkdir(parents=True, exist_ok=True)
    for voice in sys.argv[1:]:
        audio = np.concatenate([a for _, _, a in pipe(TEXT, voice=voice, speed=1.1)])
        sf.write(str(out / f"{voice}.wav"), audio, SR)
        print(f"{voice:12s} pitch {median_pitch(audio, SR):6.1f} Hz   {len(audio) / SR:5.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
