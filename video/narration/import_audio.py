"""Turn a folder of per-beat audio files into a narration folder the scenes read.

Usage (from video/):
    uv run narration/import_audio.py --src out/elevenlabs_raw --out elevenlabs \
        --voice "elevenlabs:Peter Baker (Ix8C14HEHgIQkJswik2o) eleven_v4"

For narration made outside this repo, e.g. with the ElevenLabs connector:
one file per beat, named `<beat>.mp3` or `<beat>.wav` after the `## beat`
headings of script.md. Writes out/<out>/<beat>.wav (edges trimmed like the
Kokoro script), narration.wav and timing.json, so `VOICE=<out>` and
`assemble.py --voice <out>` work exactly as they do for Kokoro.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from common import HERE, load_beats, trim_edges, write_timing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder with <beat>.mp3 or <beat>.wav")
    ap.add_argument("--out", required=True, help="folder under out/ (the VOICE the scenes read)")
    ap.add_argument("--voice", required=True, help="free text recorded in timing.json")
    ap.add_argument("--gap", type=float, default=0.4, help="silence between beats, seconds")
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else HERE.parent / args.src
    out = HERE.parent / "out" / args.out
    out.mkdir(parents=True, exist_ok=True)

    durations: dict[str, float] = {}
    joined: list[np.ndarray] = []
    sr = None
    for beat in load_beats():
        hits = [p for ext in ("wav", "mp3") if (p := src / f"{beat.name}.{ext}").exists()]
        if not hits:
            raise SystemExit(f"missing {src}/{beat.name}.mp3 (or .wav)")
        audio, beat_sr = sf.read(str(hits[0]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr is not None and beat_sr != sr:
            raise SystemExit(f"{hits[0].name}: {beat_sr} Hz, the others are {sr} Hz")
        sr = beat_sr
        audio = trim_edges(audio, sr)
        sf.write(out / f"{beat.name}.wav", audio, sr)
        durations[beat.name] = len(audio) / sr
        joined += [audio, np.zeros(int(sr * args.gap), dtype=np.float32)]
        print(f"{beat.name:20s} {durations[beat.name]:5.2f} s")

    track = np.concatenate(joined[:-1])
    sf.write(out / "narration.wav", track, sr)
    timing = write_timing(out, args.voice, durations, args.gap)
    print(f"total {len(track) / sr:.2f} s  ->  {out / 'narration.wav'}\n{timing}")


if __name__ == "__main__":
    main()
