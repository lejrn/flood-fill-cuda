"""Render every beat of script.md with ElevenLabs into out/elevenlabs/.

Usage (from video/):
    uv run narration/tts_elevenlabs.py [--voice-id Ix8C14HEHgIQkJswik2o] [--model eleven_v4]

Reads ELEVENLABS_API_KEY from video/.env (never printed, never logged).
Only the beat text is sent. Each beat is one request, so the free tier
(10k characters/month) covers the whole script a few times over.

Without a key, the same narration can be made with the ElevenLabs
connector (one TTS node per beat) and brought in with import_audio.py.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from common import HERE, load_beats, trim_edges, write_timing

ENV = HERE.parent / ".env"
# "Peter Baker - Deep, Clear and Mature", a British narrator from the voice
# library, chosen by the user. Override with --voice-id.
DEFAULT_VOICE = "Ix8C14HEHgIQkJswik2o"


def load_env_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key and ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("ELEVENLABS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY missing: put it in video/.env (gitignored)")
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-id", default=DEFAULT_VOICE)
    ap.add_argument("--model", default="eleven_v4")
    ap.add_argument("--gap", type=float, default=0.4)
    args = ap.parse_args()

    from elevenlabs.client import ElevenLabs

    headers = {}
    if os.environ.get("PUBLIC_USER_AGENT"):
        headers["User-Agent"] = os.environ["PUBLIC_USER_AGENT"]
    client = ElevenLabs(api_key=load_env_key(), headers=headers or None)

    out = HERE.parent / "out" / "elevenlabs"
    out.mkdir(parents=True, exist_ok=True)

    durations: dict[str, float] = {}
    joined: list[np.ndarray] = []
    sr = None
    for beat in load_beats():
        stream = client.text_to_speech.convert(
            voice_id=args.voice_id,
            text=beat.text,
            model_id=args.model,
            output_format="mp3_44100_128",
        )
        raw = b"".join(stream)
        (out / f"{beat.name}.mp3").write_bytes(raw)
        audio, beat_sr = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sr = sr or beat_sr
        audio = trim_edges(audio, beat_sr)
        sf.write(out / f"{beat.name}.wav", audio, beat_sr)
        durations[beat.name] = len(audio) / beat_sr
        joined += [audio, np.zeros(int(beat_sr * args.gap), dtype=np.float32)]
        print(f"{beat.name:20s} {durations[beat.name]:5.2f} s")

    track = np.concatenate(joined[:-1])
    sf.write(out / "narration.wav", track, sr)
    timing = write_timing(out, f"elevenlabs:{args.voice_id}", durations, args.gap)
    print(f"total {len(track) / sr:.2f} s  ->  {out / 'narration.wav'}\n{timing}")


if __name__ == "__main__":
    main()
