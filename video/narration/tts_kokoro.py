"""Render every beat of a narration script with Kokoro (local, free) into out/kokoro/.

Usage (from video/):
    uv run narration/tts_kokoro.py [--voice am_adam] [--gap 0.6]
    uv run narration/tts_kokoro.py --only 08_triton      # one new beat, keep the rest
    uv run narration/tts_kokoro.py --script script_chain.md --out kokoro_chain

Writes one wav per beat, a joined narration.wav with `gap` seconds of
silence between beats, timing.json for the assembly step, and words.json
(every word's start and end from Kokoro's own duration predictions, in
seconds from the trimmed wav's start) for scenes that cue on a word.
Kokoro is not bit-reproducible, so `--only` voices just the named beats
and reads every other beat from its existing wav.

Each side of a semicolon is voiced as its own piece (SPLIT): Kokoro can
read straight through a semicolon, and then pace.py has no pause to
stretch and cuts its silence into running speech.
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

from common import HERE, load_beats, read_words, script_path, trim_bounds, write_timing, write_words  # noqa: E402

SR = 24_000
# Kokoro's own default splits on newlines, which a beat never has (load_beats
# joins its lines). The chain cut's r14 ("the GPU now; with the copy") had
# no pause at all at its semicolon; voiced in two pieces it gets a real one.
SPLIT = r"(?<=;)\s+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="am_adam", help="the chosen male voice; am_onyx is deeper, af_heart was the first cut")
    ap.add_argument("--gap", type=float, default=0.4, help="silence between beats, seconds")
    ap.add_argument("--speed", type=float, default=1.1, help="Kokoro reads slowly at 1.0")
    ap.add_argument("--out", default="kokoro", help="folder under out/ (the VOICE the scenes read)")
    ap.add_argument("--only", nargs="+", default=None, help="voice only these beats; the rest keep their wavs")
    ap.add_argument("--script", default=None, help="narration file under narration/ (default script.md)")
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
    script = script_path(args.script)
    beats = load_beats(script)
    unknown = set(args.only or ()) - {b.name for b in beats}
    if unknown:
        raise SystemExit(f"no such beat in {script.name}: {sorted(unknown)}")
    old_words = read_words(out)
    words: dict[str, list] = {}
    for beat in beats:
        wav = out / f"{beat.name}.wav"
        if args.only is None or beat.name in args.only:
            chunks, rows, t0 = [], [], 0.0
            for res in pipe(beat.text, voice=args.voice, speed=args.speed, split_pattern=SPLIT):
                a = res.audio.numpy() if hasattr(res.audio, "numpy") else np.asarray(res.audio)
                for tk in res.tokens or ():
                    if tk.start_ts is not None and tk.end_ts is not None and any(c.isalnum() for c in tk.text):
                        rows.append([tk.text, t0 + tk.start_ts, t0 + tk.end_ts])
                chunks.append(a)
                t0 += len(a) / SR
            audio = np.concatenate(chunks).astype(np.float32)
            lo, hi = trim_bounds(audio, SR)
            audio = audio[lo:hi]
            shift = lo / SR
            words[beat.name] = [[w, round(a - shift, 3), round(b - shift, 3)] for w, a, b in rows]
            sf.write(wav, audio, SR)
        else:
            if beat.name in old_words:
                words[beat.name] = old_words[beat.name]
            if not wav.exists():
                raise SystemExit(f"missing {wav}: voice it too, or drop --only")
            audio, sr = sf.read(str(wav), dtype="float32")
            if sr != SR:
                raise SystemExit(f"{wav.name}: {sr} Hz, expected {SR}")
        durations[beat.name] = len(audio) / SR
        joined += [audio, silence]
        print(f"{beat.name:20s} {durations[beat.name]:5.2f} s")

    track = np.concatenate(joined[:-1])
    sf.write(out / "narration.wav", track, SR)
    timing = write_timing(out, f"kokoro:{args.voice}", durations, args.gap)
    write_words(out, words, f"kokoro:{args.voice} duration predictions")
    print(f"total {len(track) / SR:.2f} s  ->  {out / 'narration.wav'}\n{timing}")


if __name__ == "__main__":
    main()
