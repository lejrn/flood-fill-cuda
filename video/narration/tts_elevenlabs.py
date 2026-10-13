"""Render every beat of a narration script with ElevenLabs into out/elevenlabs/.

Usage (from video/):
    uv run narration/tts_elevenlabs.py [--voice-id Ix8C14HEHgIQkJswik2o] [--model eleven_v4]
    uv run narration/tts_elevenlabs.py --script script_chain.md --out elevenlabs_chain
    uv run narration/tts_elevenlabs.py --script script_chain.md --out elevenlabs_chain --only c03_more_teams

Each beat is one timestamped request: the audio comes back with every
character's start and end, so words.json holds the voice's real word
times (not an estimate). The raw mp3 and its alignment are kept in
out/<out>_raw/, and a beat that already has them is never sent again
unless named with --only: a re-run costs no credits.

Reads ELEVENLABS_API_KEY from video/.env (never printed, never logged).
Only the beat text is sent. Each beat is one request, so the free tier
(10k characters/month) covers the whole script a few times over.

Without a key, the same narration can be made with the ElevenLabs
connector (one TTS node per beat) and brought in with import_audio.py.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from common import HERE, load_beats, script_path, trim_bounds, write_timing, write_words

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


def words_from_alignment(chars, starts, ends) -> list:
    """[word, start, end] from per-character times (whitespace splits words)."""
    rows, cur, t0, t1 = [], "", 0.0, 0.0
    for ch, a, b in zip(chars, starts, ends):
        if ch.isspace():
            if cur:
                rows.append([cur, t0, t1])
            cur = ""
            continue
        if not cur:
            t0 = a
        cur += ch
        t1 = b
    if cur:
        rows.append([cur, t0, t1])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-id", default=DEFAULT_VOICE)
    ap.add_argument("--model", default="eleven_v4")
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--script", default=None, help="narration file under narration/ (default script.md)")
    ap.add_argument("--out", default="elevenlabs", help="folder under out/ (the VOICE the scenes read)")
    ap.add_argument("--only", nargs="+", default=None, help="(re)voice these beats even if a take exists")
    args = ap.parse_args()

    out = HERE.parent / "out" / args.out
    raw_dir = HERE.parent / "out" / f"{args.out}_raw"
    out.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    beats = load_beats(script_path(args.script))
    unknown = set(args.only or ()) - {b.name for b in beats}
    if unknown:
        raise SystemExit(f"no such beat: {sorted(unknown)}")

    client = None
    durations: dict[str, float] = {}
    words: dict[str, list] = {}
    joined: list[np.ndarray] = []
    sr = None
    for beat in beats:
        mp3, align = raw_dir / f"{beat.name}.mp3", raw_dir / f"{beat.name}.alignment.json"
        if not mp3.exists() or (args.only and beat.name in args.only):
            if client is None:
                from elevenlabs.client import ElevenLabs

                headers = {}
                if os.environ.get("PUBLIC_USER_AGENT"):
                    headers["User-Agent"] = os.environ["PUBLIC_USER_AGENT"]
                client = ElevenLabs(api_key=load_env_key(), headers=headers or None)
            res = client.text_to_speech.convert_with_timestamps(
                voice_id=args.voice_id, text=beat.text, model_id=args.model, output_format="mp3_44100_128")
            mp3.write_bytes(base64.b64decode(res.audio_base_64))
            a = res.alignment
            align.write_text(json.dumps({
                "characters": list(a.characters),
                "starts": list(a.character_start_times_seconds),
                "ends": list(a.character_end_times_seconds),
            } if a is not None else {}), encoding="utf-8")
        audio, beat_sr = sf.read(io.BytesIO(mp3.read_bytes()), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sr = sr or beat_sr
        lo, hi = trim_bounds(audio, beat_sr)
        audio = audio[lo:hi]
        sf.write(out / f"{beat.name}.wav", audio, beat_sr)
        durations[beat.name] = len(audio) / beat_sr
        al = json.loads(align.read_text(encoding="utf-8")) if align.exists() else {}
        if al:
            shift = lo / beat_sr
            words[beat.name] = [[w, round(a - shift, 3), round(b - shift, 3)]
                                for w, a, b in words_from_alignment(al["characters"], al["starts"], al["ends"])]
        joined += [audio, np.zeros(int(beat_sr * args.gap), dtype=np.float32)]
        print(f"{beat.name:20s} {durations[beat.name]:5.2f} s")

    track = np.concatenate(joined[:-1])
    sf.write(out / "narration.wav", track, sr)
    timing = write_timing(out, f"elevenlabs:{args.voice_id} {args.model}", durations, args.gap)
    write_words(out, words, "elevenlabs character alignment")
    print(f"total {len(track) / sr:.2f} s  ->  {out / 'narration.wav'}\n{timing}")


if __name__ == "__main__":
    main()
