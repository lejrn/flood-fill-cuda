"""Render every beat, concatenate, and lay the narration over it.

Usage (from video/):
    uv run build/assemble.py --render            # render all scenes at 1080p30, then assemble
    uv run build/assemble.py                     # assemble from existing renders
    uv run build/assemble.py --layout vertical --render
    uv run build/assemble.py --voice elevenlabs
    uv run build/assemble.py --render --only s03_n_blocks   # one clip
    uv run build/assemble.py --no-audio --check              # silent cut + seam check
    uv run build/assemble.py --cut chain --look iso --layout vertical --render --check
    uv run build/assemble.py --cut chain --look iso --layout vertical --render --jobs 3 \
        --wrap path/to/slot_wrapper.sh                      # clips in parallel, each through the wrapper

Output: out/final_<layout>_<voice>.mp4 for the three-pane cut, and
out/final_chain_<look>_<layout>_<voice>.mp4 for the chain cut (narration
from out/<voice>_chain/, clips in media/chain_<look>/<layout>/).

--jobs N renders N clips at once. Manim's text cache is per media folder,
and two renders that share one race on it, so each job renders into its
own folder (<media>_jobs/<clip>/) and its finished clip (with the snap
sidecar) is then moved to where a sequential render puts it. --check, the
seam check and the mux read the same layout either way. --wrap CMD (or
RENDER_WRAP) runs every render through a command prefix, such as a script
that caps how many renders run at once on a small machine.

Each scene is exactly `beat + gap` seconds long (see style.BeatScene.finish),
so the narration wavs are placed at the cumulative start of each scene as
measured from the rendered files, not from the plan. That keeps the voice
locked to the picture even if a scene is a frame short or long.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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

# The problem -> fix chain cut, v5 "from pixels to tiles" (narration/script_chain.md,
# scenes/chain/): v4's first fourteen beats, then act 3 (the tiles) and the
# finale. The intro clip voices two beats, like the three-pane intro.
# v4's r10_runs and r11_reveal left the chain; their files stay.
CHAIN = [
    ("chain/intro", "Intro", "i0_problem"),
    ("chain/intro", "Intro", "i1_compare"),
    ("chain/r00_zoom", "Zoom", "r00_zoom"),
    ("chain/r01_cpu", "Cpu", "r01_cpu"),
    ("chain/r02_one_block", "OneBlock", "r02_one_block"),
    ("chain/r03_giant", "Giant", "r03_giant"),
    ("chain/r03b_one_sm", "OneSm", "r03b_one_sm"),
    ("chain/r04_two_blocks", "TwoBlocks", "r04_two_blocks"),
    ("chain/r05_all_blocks", "AllBlocks", "r05_all_blocks"),
    ("chain/r05b_snake", "Snake", "r05b_snake"),
    ("chain/r06_diagonals", "Diagonals", "r06_diagonals"),
    ("chain/r07_real_frame", "RealFrame", "r07_real_frame"),
    ("chain/r08_tags", "Tags", "r08_tags"),
    ("chain/r09_seeds", "Seeds", "r09_seeds"),
    ("chain/r10_tiles", "Tiles", "r10_tiles"),
    ("chain/r11_rounds", "Rounds", "r11_rounds"),
    ("chain/r12_pieces", "Pieces", "r12_pieces"),
    ("chain/r13_frame", "BigFrame", "r13_frame"),
    ("chain/r14_reveal", "Reveal", "r14_reveal"),
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


def render_cmd(stem: str, cls: str, media: Path, layout: str, quality: str, preview: bool,
               wrap: list[str]) -> list[str]:
    # --disable_caching: a cached play skips update_mobjects(0), so end
    # states can differ from an uncached render. Always render fresh.
    cmd = [*wrap, str(PY), "-m", "manim", "render", "--disable_caching", "--media_dir", str(media)]
    if preview:     # 480 px, 15 fps: for checking timing and layout, never for a cut
        cmd += ["-ql", "-r", "480,854" if layout == "vertical" else "854,480"]
    else:
        cmd += [f"-q{quality}", "--fps", "30"]
        if layout == "vertical":
            cmd += ["-r", "1080,1920"]
    return cmd + [str(SCENES / f"{stem}.py"), cls]


def _show(p: Path) -> str:
    """A path for the console: relative to video/ when it is under it."""
    try:
        return str(p.relative_to(VIDEO))
    except ValueError:
        return str(p)


def job_dir(layout: str, stem: str, cut: str, look: str) -> Path:
    """A parallel job's own media folder, beside the cut's: <media>_jobs/<clip>."""
    m = media_dir(layout, cut, look)
    return m.parent / f"{m.name}_jobs" / Path(stem).name


def adopt(job: Path, layout: str, stem: str, cls: str, cut: str, look: str) -> Path:
    """Move a job's newest clip of stem/cls (and its .snap.json) to the path a
    sequential render gives it: <media>/videos/<stem>/<quality>/<cls>.mp4.
    A move keeps the render's mtime, so newest_render() still picks it."""
    folder = job / "videos" / Path(stem).name
    hits = sorted(folder.glob(f"*/{cls}.mp4"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise RuntimeError(f"{stem}/{cls}: the job finished but left no clip under {folder}")
    src = hits[-1]
    dst = media_dir(layout, cut, look) / "videos" / Path(stem).name / src.parent.name / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    snap = Path(str(src) + ".snap.json")
    os.replace(src, dst)
    if snap.exists():
        os.replace(snap, Path(str(dst) + ".snap.json"))
    return dst


def render(layout: str, voice: str, quality: str, only: list[str] | None = None,
           cut: str = "panes", look: str = "flat", preview: bool = False, jobs: int = 1,
           wrap: list[str] | None = None) -> None:
    env = dict(os.environ, VOICE=voice_dir(voice, cut), VIDEO_LAYOUT=layout, LOOK=look)
    wrap = wrap or []
    todo = [(stem, cls) for stem, cls, _ in clip_list(cut)
            if not only or stem in only or Path(stem).name in only]
    if jobs <= 1:
        for stem, cls in todo:
            print("render", stem, cls, flush=True)
            subprocess.run(render_cmd(stem, cls, media_dir(layout, cut, look), layout, quality, preview, wrap),
                           cwd=VIDEO, env=env, check=True)
        return

    def one(job: tuple[str, str]) -> tuple[str, str, int, Path]:
        stem, cls = job
        d = job_dir(layout, stem, cut, look)
        d.mkdir(parents=True, exist_ok=True)
        log = d / f"{cls}.log"
        print("render", stem, cls, f"(job, log {_show(log)})", flush=True)
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(render_cmd(stem, cls, d, layout, quality, preview, wrap), cwd=VIDEO, env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            dst = adopt(d, layout, stem, cls, cut, look)
            print(f"done   {stem} {cls} -> {_show(dst)}", flush=True)
        else:
            print(f"FAILED {stem} {cls} (exit {rc}); see {_show(log)}", flush=True)
        return stem, cls, rc, log

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, todo))
    failed = [(s, c, log) for s, c, rc, log in results if rc != 0]
    if failed:
        raise SystemExit("failed renders: " + ", ".join(f"{s}/{c} ({_show(log)})" for s, c, log in failed))


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
    ap.add_argument("--jobs", type=int, default=1,
                    help="render this many clips at once, each in its own media folder, then move each clip "
                         "where a sequential render puts it")
    ap.add_argument("--wrap", default=os.environ.get("RENDER_WRAP", ""),
                    help="a command prefix for every render (default: $RENDER_WRAP), e.g. a render-slot script")
    args = ap.parse_args()
    vdir = voice_dir(args.voice, args.cut)
    name = f"{args.layout}_{args.voice}" if args.cut == "panes" else f"chain_{args.look}_{args.layout}_{args.voice}"
    wrap = shlex.split(args.wrap)
    if args.render and args.jobs > 1 and not wrap:
        print(f"warning: {args.jobs} parallel renders without --wrap: each 1080p render can take over 1 GB",
              flush=True)

    if args.render:
        render(args.layout, args.voice, args.quality, args.only, args.cut, args.look, args.preview, args.jobs, wrap)
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
    hi = 300.0 if args.cut == "chain" else 150.0          # the chain cut may run to five minutes (v4)
    if not 60.0 <= total <= hi:
        msg = f"total {total:.1f} s is outside the 60-{hi:.0f} s target"
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
