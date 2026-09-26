# Handoff: tools and frameworks for the flood-fill explainer video

Status on 2026-09-26: two finished cuts exist, both 58.3 s, both with the
Kokoro voice. `out/final_landscape_kokoro.mp4` (1920x1080) and
`out/final_vertical_kokoro.mp4` (1080x1920). Everything that made them
is on branch `video-explainer`, folder `video/`. Nothing is installed
outside that folder except two apt header packages (see Setup).

## 1. The stack

| Role | Tool | Version | Cost / license | Why this one |
|---|---|---|---|---|
| Animation engine | Manim Community | 0.19.1 | MIT, free | 3Blue1Brown look, Python, deterministic frames, no browser needed |
| Environment | uv | project venv, Python 3.10 | free | One `uv sync` reproduces it; user preference over conda |
| Narration | Kokoro (`kokoro` + `misaki`) | 0.9.4 | Apache-2.0, runs locally on the RTX 4060 | Free, no quota, good English voice (`af_heart`) |
| Narration (alt) | ElevenLabs Python SDK | 2.69 | free tier, needs API key | Higher quality; not run yet, key missing |
| Muxing / concat | ffmpeg static via `imageio-ffmpeg` | 7.0.2 | free | No system install; symlinked into `.venv/bin/ffmpeg` |
| Frame extraction | Pillow | 12.x | free | Unpacks the parent repo's wavefront GIFs into PNG sequences |
| Text shaping | ManimPango + pycairo | built from source | free | Needs `libcairo2-dev libpango1.0-dev` once |

Installed but unused: Remotion agent skills (`.claude/skills/remotion-*`)
for a possible captions pass. Evaluated and dropped: HyperFrames (HTML
video, would have split the look), Excalimate (hand-drawn style clashes
with Manim), micromamba (user prefers uv), Whisper (no human voice to
transcribe).

## 2. Folder layout

```
video/
  pyproject.toml, uv.lock     dependencies, incl. the spaCy model Kokoro needs
  narration/
    script.md                 the 7 beats; each ```text block is one TTS call
    common.py                 parses script.md, writes timing.json
    tts_kokoro.py             -> out/kokoro/<beat>.wav + narration.wav + timing.json
    tts_elevenlabs.py         -> out/elevenlabs/..., reads ELEVENLABS_API_KEY from .env
  scenes/
    style.py                  palette, BeatScene, Stopwatch, FrameSequence, helpers
    BRIEF.md                  the spec every scene was built from (numbers, layout rules)
    s00_hook.py .. s06_outro.py   one Manim scene per beat
  assets/
    extract_gifs.py           GIF -> assets/<name>/frame_NNN.png + meta.json (gitignored output)
  build/
    assemble.py               render all, concat, lay narration by measured clip starts
  out/                        narration tracks and final mp4s (gitignored)
  media/                      Manim renders and review frames (gitignored)
  .env                        ELEVENLABS_API_KEY=... (gitignored, user-owned)
```

## 3. The pipeline

1. `narration/script.md` holds the spoken text per beat. Editing a
   sentence means re-running the TTS script; scenes then pad themselves
   to the new length automatically.
2. `narration/tts_kokoro.py` renders each beat to its own wav, trims
   edge silence, and writes `timing.json` (seconds per beat, 0.4 s gap).
3. Each scene subclasses `BeatScene`, reads its beat length from
   `timing.json` (`VOICE` env var picks the folder), animates, then
   `self.finish()` pads to beat + 0.4 s and fades out.
4. `build/assemble.py --render` renders the seven scenes at 1080p30
   with `--disable_caching`, concatenates them (re-encode, libx264 crf 18),
   measures each clip, places every beat wav at its clip's real start,
   and muxes AAC audio. `--layout vertical` renders 1080x1920.

## 4. Commands

```bash
cd /home/lrn/Repos/flood-fill-cuda/video
uv sync                                        # environment
uv run assets/extract_gifs.py                  # frames from the parent repo's GIFs
uv run narration/tts_kokoro.py                 # voice + timing
uv run build/assemble.py --render              # 16:9 final
uv run build/assemble.py --render --layout vertical
uv run build/assemble.py --voice elevenlabs    # re-mux only, after tts_elevenlabs.py
uv run build/assemble.py --music track.mp3 --music-gain 0.12
```

Single scene preview at low quality:

```bash
VOICE=kokoro uv run python -m manim render -ql scenes/s02_gpu_waves.py GpuWaves
```

## 5. Setup from scratch

- `sudo apt install libcairo2-dev libpango1.0-dev` (ManimPango and
  pycairo have no Linux wheels). gcc, make and Python headers were
  already present.
- `uv sync` installs everything else, including torch (CUDA build) for
  Kokoro and the spaCy `en_core_web_sm` wheel pinned in `pyproject.toml`.
- The Kokoro model downloads on first run into `video/.hf-cache/`
  (`HF_HOME` is set by the script).
- Run Python scripts with the venv on `PATH` or via `uv run`: misaki
  shells out to `python`, and the bare pyenv shim has no global version.

## 6. Conventions the scenes follow

- Colours only from `style.py`: red is "unfilled pixel / slow number",
  teal is "the final answer", grey is idle.
- `Text` only (no LaTeX installed). Numbers in `DejaVu Sans Mono`.
- Two panels per scene, side by side in landscape, stacked in portrait
  (`is_vertical()`); portrait uses an 8-unit-wide frame so text keeps its
  pixel size.
- Every number on screen comes from `scenes/BRIEF.md`, which quotes the
  parent repo's READMEs and benchmark JSON. Estimates carry a `~`.
- Scenes do not call `add_sound`; audio is laid in assembly only.

## 7. Gotchas that cost time (all fixed, keep them fixed)

- Manim muxes scene audio with `shortest=1`, so an in-scene `add_sound`
  truncates the padded tail. Assembly muxes instead.
- The Cairo renderer snapshots the mobject family when a play or wait
  starts. Anything removed mid-play keeps being drawn. `FrameSequence`
  therefore swaps one image's `pixel_array` in place, and
  `Stopwatch.set_ms` blanks the old glyphs.
- Cached partial renders skip `update_mobjects(0)`, so end states can
  differ between cached and fresh renders. Always `--disable_caching`.
- The renderer clock advances by whole frames; scene lengths can be a
  frame off. Assembly measures real clip lengths, so nothing drifts.
- Under `-r 1080,1920` Manim keeps `frame_width` at 14.22 and everything
  shrinks; `style.py` sets it to 8 when `VIDEO_LAYOUT=vertical`.
- Two concurrent Manim renders into the same media folder can corrupt
  partial files. Render sequentially.
- zsh does not word-split unquoted variables; use `${=var}` or
  `${pair%%:*}` patterns in shell loops.

## 8. Numbers worth double-checking before publishing

- The parent README's "150x fewer runs" is runs vs all pixels; vs red
  pixels it is 25x (13,451,960 / 539,207). The video says 25x.
- ch05 baseline: 58.51 ms (same session as ch06) vs 51.03 ms (overview
  session). The video uses 58.51 everywhere.
- ch01 to ch04 bars on the real image are estimates (`est: true` in the
  overview JSON). The video marks them `~`.
- "0.64 ms" for the five middle kernels is the README's figure; the
  per-phase table sums to 0.63.

## 9. Open items

1. ElevenLabs voice: add `video/.env`, run `tts_elevenlabs.py`, re-assemble
   with `--voice elevenlabs`, compare with Kokoro.
2. Optional background music via `--music`.
3. Reviewer feedback on pace (Kokoro at speed 1.15) and on the noisy
   8-block replay in beat 02 (a cleaner alternative is
   `assets/ch03_disk`).
4. Captions pass, if wanted, with the installed Remotion skills or a
   Manim subtitle layer driven by `timing.json`.
