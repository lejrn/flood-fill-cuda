"""Render every beat, concatenate, and lay the narration over it.

Usage (from video/):
    uv run build/assemble.py --render            # render all scenes at 1080p30, then assemble
    uv run build/assemble.py                     # assemble from existing renders
    uv run build/assemble.py --layout vertical --render
    uv run build/assemble.py --voice elevenlabs

Output: out/final_<layout>_<voice>.mp4

Each scene is exactly `beat + gap` seconds long (see style.BeatScene.finish),
so the narration wavs are placed at the cumulative start of each scene as
measured from the rendered files, not from the plan. That keeps the voice
locked to the picture even if a scene is a frame short or long.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import soundfile as sf

VIDEO = Path(__file__).resolve().parents[1]
SCENES = VIDEO / "scenes"
OUT = VIDEO / "out"
PY = VIDEO / ".venv" / "bin" / "python"
FFMPEG = VIDEO / ".venv" / "bin" / "ffmpeg"

# (scene file stem, class name, beat name) in narrative order.
BEATS = [
    ("s00_hook", "Hook", "00_hook"),
    ("s01_cpu", "Cpu", "01_cpu"),
    ("s02_gpu_waves", "GpuWaves", "02_gpu_waves"),
    ("s03_twist", "Twist", "03_twist"),
    ("s04_blobs_together", "BlobsTogether", "04_blobs_together"),
    ("s05_runs", "Runs", "05_runs"),
    ("s06_outro", "Outro", "06_outro"),
]

SR = 48_000


def media_dir(layout: str) -> Path:
    return VIDEO / "media" / layout


def render(layout: str, voice: str, quality: str) -> None:
    env = dict(os.environ, VOICE=voice, VIDEO_LAYOUT=layout)
    for stem, cls, _ in BEATS:
        cmd = [str(PY), "-m", "manim", "render", f"-q{quality}", "--fps", "30",
               "--media_dir", str(media_dir(layout))]
        if layout == "vertical":
            cmd += ["-r", "1080,1920"]
        cmd += [str(SCENES / f"{stem}.py"), cls]
        print("render", stem, cls, flush=True)
        subprocess.run(cmd, cwd=VIDEO, env=env, check=True)


def newest_render(layout: str, stem: str, cls: str) -> Path:
    hits = sorted((media_dir(layout) / "videos" / stem).glob(f"*/{cls}.mp4"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no render for {stem}/{cls} under {media_dir(layout)}; run with --render")
    return hits[-1]


def duration(path: Path) -> float:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if s.duration is not None and s.time_base is not None:
            return float(s.duration * s.time_base)
        return c.duration / 1e6


def load_wav(path: Path, sr: int) -> np.ndarray:
    audio, in_sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if in_sr != sr:
        # linear resample, good enough for speech
        n = int(round(len(audio) * sr / in_sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["landscape", "vertical"], default="landscape")
    ap.add_argument("--voice", default="kokoro")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--quality", default="h", help="manim quality letter: l, m, h, k")
    ap.add_argument("--music", type=Path, default=None, help="optional background track, mixed at --music-gain")
    ap.add_argument("--music-gain", type=float, default=0.12)
    args = ap.parse_args()

    if args.render:
        render(args.layout, args.voice, args.quality)

    clips = [newest_render(args.layout, stem, cls) for stem, cls, _ in BEATS]
    durs = [duration(p) for p in clips]
    starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
    total = float(sum(durs))
    for (stem, _, beat), d, t0 in zip(BEATS, durs, starts):
        print(f"{beat:20s} start {t0:6.2f}  len {d:5.2f}")
    print(f"total {total:.2f} s")

    OUT.mkdir(exist_ok=True)
    work = OUT / f"work_{args.layout}_{args.voice}"
    work.mkdir(exist_ok=True)

    # 1. video-only concat (re-encode so every clip shares one timebase)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in clips), encoding="utf-8")
    video_only = work / "video_only.mp4"
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
                    "-an", str(video_only)], check=True)

    # 2. narration laid at each scene's measured start
    track = np.zeros(int(np.ceil(total * SR)) + SR, dtype=np.float32)
    for (_, _, beat), t0 in zip(BEATS, starts):
        wav = OUT / args.voice / f"{beat}.wav"
        if not wav.exists():
            raise SystemExit(f"missing narration {wav}; run narration/tts_{args.voice}.py")
        a = load_wav(wav, SR)
        i = int(round(t0 * SR))
        track[i:i + len(a)] += a[: len(track) - i]
    if args.music is not None:
        m = load_wav(args.music, SR)
        reps = int(np.ceil(len(track) / len(m)))
        m = np.tile(m, reps)[: len(track)] * args.music_gain
        fade = SR * 2
        m[-fade:] *= np.linspace(1, 0, fade)
        track += m
    peak = float(np.abs(track).max()) or 1.0
    if peak > 0.98:
        track *= 0.98 / peak
    aligned = work / "narration_aligned.wav"
    sf.write(str(aligned), track[: int(np.ceil(total * SR))], SR)

    # 3. mux
    final = OUT / f"final_{args.layout}_{args.voice}.mp4"
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(aligned),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
                    str(final)], check=True)
    print(f"->  {final}  ({duration(final):.2f} s)")


if __name__ == "__main__":
    sys.exit(main())
