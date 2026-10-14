"""Render every beat, concatenate, and lay the narration over it.

Usage (from video/):
    uv run build/assemble.py --render            # render all scenes at 1080p30, then assemble
    uv run build/assemble.py                     # assemble from existing renders
    uv run build/assemble.py --layout vertical --render
    uv run build/assemble.py --voice elevenlabs
    uv run build/assemble.py --render --only s03_n_blocks   # one clip
    uv run build/assemble.py --no-audio --check              # silent cut + seam check
    uv run build/assemble.py --cut chain --look iso --layout vertical --render --check

Output: out/final_<layout>_<voice>.mp4 for the three-pane cut, and
out/final_chain_<look>_<layout>_<voice>.mp4 for the chain cut (narration
from out/<voice>_chain/, clips in media/chain_<look>/<layout>/).

Each scene is exactly `beat + gap` seconds long (see style.BeatScene.finish),
so the narration wavs are placed at the cumulative start of each scene as
measured from the rendered files, not from the plan. That keeps the voice
locked to the picture even if a scene is a frame short or long.
"""
from __future__ import annotations

import argparse
import json
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
PANES = [
    ("s_intro", "Intro", "intro_problem"),
    ("s_intro", "Intro", "intro_budget"),
    ("s00_cpu", "Cpu", "00_cpu"),
    ("s01_one_block", "OneBlock", "01_one_block"),
    ("s02_two_blocks", "TwoBlocks", "02_two_blocks"),
    ("s03_n_blocks", "NBlocks", "03_n_blocks"),
    ("s04_conn8", "Conn8", "04_conn8"),
    ("s05_two_blobs", "TwoBlobs", "05_two_blobs"),
    ("s06_n_blobs", "NBlobs", "06_n_blobs"),
    ("s07_runs", "Runs", "07_runs"),
    ("s08_triton", "Triton", "08_triton"),
    ("s09_outro", "Outro", "09_outro"),
]
BEATS = PANES

# The problem -> fix chain cut, v2 "size, then count" (narration/script_chain.md,
# scenes/chain/). The intro clip voices two beats, like the three-pane intro.
CHAIN = [
    ("chain/intro", "Intro", "i0_problem"),
    ("chain/intro", "Intro", "i1_compare"),
    ("chain/r00_zoom", "Zoom", "r00_zoom"),
    ("chain/r01_cpu", "Cpu", "r01_cpu"),
    ("chain/r02_one_block", "OneBlock", "r02_one_block"),
    ("chain/r03_giant", "Giant", "r03_giant"),
    ("chain/r04_two_blocks", "TwoBlocks", "r04_two_blocks"),
    ("chain/r05_all_blocks", "AllBlocks", "r05_all_blocks"),
    ("chain/r06_diagonals", "Diagonals", "r06_diagonals"),
    ("chain/r07_real_frame", "RealFrame", "r07_real_frame"),
    ("chain/r08_tags", "Tags", "r08_tags"),
    ("chain/r09_seeds", "Seeds", "r09_seeds"),
    ("chain/r10_runs", "Runs", "r10_runs"),
    ("chain/r11_reveal", "Reveal", "r11_reveal"),
]
CUTS = {"panes": PANES, "chain": CHAIN}
LOOKS = ("flat", "iso")

SR = 48_000
GAP = 0.4          # seconds between beats inside a clip (style.BeatScene.finish tail)


def clip_list(cut: str = "panes") -> list:
    """[(stem, cls, [beats])]: consecutive entries of one scene share a clip."""
    clips = []
    for stem, cls, beat in CUTS[cut]:
        if clips and clips[-1][0] == stem and clips[-1][1] == cls:
            clips[-1][2].append(beat)
        else:
            clips.append((stem, cls, [beat]))
    return clips


def media_dir(layout: str, cut: str = "panes", look: str = "flat") -> Path:
    if cut == "panes":
        return VIDEO / "media" / layout
    return VIDEO / "media" / f"chain_{look}" / layout


def voice_dir(voice: str, cut: str = "panes") -> str:
    """The folder under out/ that a cut's narration lives in (the scenes' VOICE)."""
    return voice if cut == "panes" else f"{voice}_chain"


def render(layout: str, voice: str, quality: str, only: list[str] | None = None,
           cut: str = "panes", look: str = "flat", preview: bool = False) -> None:
    env = dict(os.environ, VOICE=voice_dir(voice, cut), VIDEO_LAYOUT=layout, LOOK=look)
    for stem, cls, _ in clip_list(cut):
        if only and stem not in only and Path(stem).name not in only:
            continue
        # --disable_caching: a cached play skips update_mobjects(0), so end
        # states can differ from an uncached render. Always render fresh.
        cmd = [str(PY), "-m", "manim", "render", "--disable_caching", "--media_dir", str(media_dir(layout, cut, look))]
        if preview:     # 480 px, 15 fps: for checking timing and layout, never for a cut
            cmd += ["-ql", "-r", "480,854" if layout == "vertical" else "854,480"]
        else:
            cmd += [f"-q{quality}", "--fps", "30"]
            if layout == "vertical":
                cmd += ["-r", "1080,1920"]
        cmd += [str(SCENES / f"{stem}.py"), cls]
        print("render", stem, cls, flush=True)
        subprocess.run(cmd, cwd=VIDEO, env=env, check=True)


def newest_render(layout: str, stem: str, cls: str, cut: str = "panes", look: str = "flat",
                  quality: str | None = None) -> Path:
    """The newest clip of a scene (Manim names the folder after the file's stem)."""
    folder = media_dir(layout, cut, look) / "videos" / Path(stem).name
    pattern = f"{quality}/{cls}.mp4" if quality else f"*/{cls}.mp4"
    hits = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no render for {stem}/{cls} under {folder}; run with --render")
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
    ap.add_argument("--only", nargs="*", default=None, help="render only these scene stems")
    ap.add_argument("--no-audio", action="store_true", help="video-only cut, no narration needed")
    ap.add_argument("--check", action="store_true", help="run build/seam_check.py on the clips")
    ap.add_argument("--strict", action="store_true", help="fail outside the 60-120 s target")
    ap.add_argument("--cut", choices=sorted(CUTS), default="panes", help="panes (three-pane) or chain")
    ap.add_argument("--look", choices=LOOKS, default="flat", help="chain cut only: flat or iso (2.5D)")
    ap.add_argument("--preview", action="store_true", help="render at 480 px and 15 fps (checks only)")
    args = ap.parse_args()
    vdir = voice_dir(args.voice, args.cut)
    name = f"{args.layout}_{args.voice}" if args.cut == "panes" else f"chain_{args.look}_{args.layout}_{args.voice}"

    if args.render:
        render(args.layout, args.voice, args.quality, args.only, args.cut, args.look, args.preview)
        if args.only:
            print("rendered", args.only)
            return

    groups = clip_list(args.cut)
    clips = [newest_render(args.layout, stem, cls, args.cut, args.look) for stem, cls, _ in groups]
    durs = [duration(p) for p in clips]
    starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
    total = float(sum(durs))
    timing = OUT / vdir / "timing.json"
    beat_len = {}
    if timing.exists():
        beat_len = {r["beat"]: float(r["seconds"]) for r in json.loads(timing.read_text())["beats"]}
    beat_starts = {}
    for (stem, _, beats), d, t0 in zip(groups, durs, starts):
        off = 0.0
        for beat in beats:
            beat_starts[beat] = t0 + off
            off += beat_len.get(beat, 0.0) + GAP
        note = ""
        if all(b in beat_len for b in beats):
            slot = sum(beat_len[b] + GAP for b in beats)
            note = f"  beats {slot - GAP * len(beats):5.2f} + gaps = {slot:5.2f}" + \
                   ("  OVERRUN" if d > slot + 0.05 else "")
        print(f"{Path(stem).name:18s} start {t0:6.2f}  len {d:5.2f}{note}   [{', '.join(beats)}]")
    print(f"total {total:.2f} s")
    if not 60.0 <= total <= 150.0:
        msg = f"total {total:.1f} s is outside the 60-150 s target"
        if args.strict:
            raise SystemExit(msg)
        print("warning:", msg)
    if args.check:
        subprocess.run([str(PY), str(VIDEO / "build" / "seam_check.py"), "--layout", args.layout,
                        "--cut", args.cut, "--look", args.look, "--intra"], check=True)

    OUT.mkdir(exist_ok=True)
    work = OUT / f"work_{name}"
    work.mkdir(exist_ok=True)

    # 1. video-only concat (re-encode so every clip shares one timebase)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in clips), encoding="utf-8")
    video_only = work / "video_only.mp4"
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
                    "-an", str(video_only)], check=True)

    # 2. narration laid at each scene's measured start
    if args.no_audio:
        silent = args.layout if args.cut == "panes" else f"chain_{args.look}_{args.layout}"
        final = OUT / f"final_{silent}_silent.mp4"
        subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(video_only), "-c:v", "copy",
                        "-movflags", "+faststart", str(final)], check=True)
        print(f"->  {final}  ({duration(final):.2f} s, no audio)")
        return
    track = np.zeros(int(np.ceil(total * SR)) + SR, dtype=np.float32)
    for _, _, beat in CUTS[args.cut]:
        t0 = beat_starts[beat]
        wav = OUT / vdir / f"{beat}.wav"
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
    final = OUT / f"final_{name}.mp4"
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(aligned),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
                    str(final)], check=True)
    print(f"->  {final}  ({duration(final):.2f} s)")


if __name__ == "__main__":
    sys.exit(main())
