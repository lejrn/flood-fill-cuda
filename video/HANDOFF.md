# The flood-fill-cuda explainer video: summary and handoff

Status on 2026-09-28. Branch `video-explainer`, folder `video/`, commits
2a86ea3 .. cc75266. Final cut: `video/out/final_landscape_kokoro.mp4`,
1920x1080 at 30 fps, AAC narration, 142.9 s, voice Kokoro `am_adam`.
Thirteen alternative voices sit beside it as
`out/final_landscape_kokoro_<voice>.mp4` (rendered before the blob
counter was added). `out/` and `media/` are gitignored; only sources are
tracked.

## 1. What the video shows

| part | length | on screen |
|---|---|---|
| intro, beat `intro_problem` | ~15 s | a drone light show (real footage), then its filter output beside it, then the frame budget: 30 fps, one frame every 33 ms |
| intro, beat `intro_budget` | ~17 s | the CPU stuck on frame 1 with a real-time stopwatch, the GPU labelling every frame with a live blob counter and a 3 s sparkline, then the verdict: budget 33 ms, pure Python 24,083 ms, @njit 1,346 ms, GPU 1.46 ms |
| stages 00-07 | ~90 s | three panes for the rest of the video (below) |
| stage 08, outro | ~7 s | the full strip and matrix, "24,083 ms -> 1.46 ms", "16,000x" |

The three panes, after the intro:

- **Left, the benchmark matrix.** 17 shapes of the overview benchmark as
  rows, one column per chapter. A cell glows blue (teal for ch06) when
  that chapter's fastest measured variant beats the CPU `@njit` time,
  brighter for a bigger win; grey when slower; dashed when the number is
  an estimate (one launch per blob; never glows); hollow when the chapter
  cannot run the row. Every column keeps its ms printed.
- **Middle, the blob.** The chapter's replay, big: the CPU walk, one block,
  two blocks, N blocks, 8-connectivity, two blobs, N blobs (the U merge
  then the real image), runs. Finished stages sweep up into a strip of
  thumbnails with tags.
- **Right, the GPU.** 24 SM tiles with block chips in the replay's hues,
  the block close-up (warps x 32 lanes), shared-memory and global-memory
  boxes, the ch06 kernel strip, and the launch-config lines.

## 2. How it is built

```
benchmarks (JSON, committed)  ->  scenes/panes/data.py  ->  matrix cells, captions
chapter GIFs (committed)      ->  assets/extract_gifs.py -> assets/<name>/frame_NNN.png
drone-show Short (yt-dlp)     ->  assets/make_drone_frames.py -> drones_{sky,mask,labels}/
narration/script.md           ->  narration/tts_kokoro.py -> out/kokoro/<beat>.wav + timing.json
scenes/*.py (Manim, Cairo)    ->  media/landscape/videos/<stem>/1080p30/<Class>.mp4, one clip per scene
build/assemble.py             ->  concat, wavs at measured clip starts, mux, seam check, out/final_*.mp4
```

Stack: Manim Community 0.19.1 (Cairo renderer, `Text` only, no LaTeX),
Python 3.10 in `video/.venv` via uv, Kokoro 0.9.4 for the voice (runs on
the CPU here), a static ffmpeg from `imageio-ffmpeg` symlinked into
`.venv/bin/ffmpeg`, PyAV and Pillow for checks and frames, yt-dlp for the
footage. The GPU work (the ch06 kernel labelling the footage, the ch06
overview benchmark, the ch01 wavefront GIFs) runs in the ROOT venv.

## 3. Folder map

```
video/
  pyproject.toml, uv.lock       dependencies (incl. yt-dlp and the spaCy model Kokoro needs)
  narration/
    script.md                   11 beats; each ```text block is one TTS call
    common.py                   parses script.md, writes timing.json
    tts_kokoro.py               -> out/<--out>/<beat>.wav + timing.json (default voice am_adam)
    pick_voice.py               one sentence in several voices, ranked by median pitch
    tts_elevenlabs.py           alternative voice, needs ELEVENLABS_API_KEY in .env (unused)
  scenes/
    style.py                    palette, BeatScene (whole-frame padding), helpers
    stage.py                    fr(), Replay (lazy frames), State, build_state(), StageScene
    s_intro.py                  the problem: four vertical footage panels, two beats in one clip
    s00_cpu.py .. s08_outro.py  one thin scene per stage (k = stage index)
    snapshot.py                 STAGE=k [LIVE=1] -> one PNG of a pane state with `-s`
    panes/geometry.py           the three pane boxes (5.3 / 4.1 / 3.3 units wide)
    panes/config.py             STAGES: beat, tag, replay set, captions, GpuSpec per stage
    panes/data.py               benchmark JSON -> rows, cells, headline numbers; CLI audit
    panes/left_matrix.py        MatrixPane
    panes/middle_strip.py       strip, big image, captions, the runs row
    panes/right_gpu.py          GpuPane
    BRIEF.md                    the on-screen spec and the rules
  assets/
    extract_gifs.py             chapter GIFs -> assets/<name>/ (gitignored output)
    make_drone_frames.py        footage -> camera / filter / tracked-labels sets + meta.json
    source/                     downloaded footage (gitignored)
  build/
    assemble.py                 --render [--only stem] [--voice name] [--no-audio] [--check] [--strict]
    variants.py                 one full cut per Kokoro male voice
    review.py                   frames at given times + a contact sheet
    seam_check.py               every cut and every hold, on 4x4 box-averaged frames
    audit_numbers.py            every printed number recomputed from the JSON, independently
  out/                          narration folders (kokoro, kokoro_<voice>) and final mp4s
  media/landscape/              per-scene renders; media/review/ contact sheets and seam images
```

## 4. Commands

All from `video/`. Never run the TTS, a render and the GPU benchmark at
the same time: the laptop has 6 GB of RAM (section 7).

```bash
uv sync                                                        # once; needs apt libcairo2-dev libpango1.0-dev
uv run assets/extract_gifs.py                                  # chapter GIFs -> frame folders
.venv/bin/yt-dlp -f "bv*[height<=1080]" -o "assets/source/drones_short.%(ext)s" \
    https://www.youtube.com/shorts/p2cDTfSIwqs                 # video stream only; rename to drones_short.mp4
../.venv/bin/python assets/make_drone_frames.py \
    --source assets/source/drones_short.mp4 --start 200 --frames 240   # root venv, GPU; prints per-stage ms
uv run python scenes/panes/data.py                             # the matrix as text, exit 1 on a mismatch
uv run build/audit_numbers.py                                  # every on-screen number vs the JSON
uv run narration/tts_kokoro.py                                 # default voice into out/kokoro/
uv run build/assemble.py --render --check                      # full cut -> out/final_landscape_kokoro.mp4
uv run build/assemble.py --render --only s_intro && uv run build/assemble.py --check   # one clip, then re-mux
uv run build/variants.py am_onyx bm_lewis                      # other voices, 6-11 min each
uv run build/review.py s03_n_blocks NBlocks --times 0,0.3,0.6,1,1.5,2.6,5,end
STAGE=4 LIVE=1 VOICE=kokoro .venv/bin/python -m manim render -s -qh --disable_caching \
    --media_dir media/landscape scenes/snapshot.py Snapshot     # one PNG for layout work
VOICE=kokoro .venv/bin/python -m manim render -ql --disable_caching \
    --media_dir media/landscape scenes/s03_n_blocks.py NBlocks  # 15 fps preview of one scene
```

Changing a spoken sentence: edit `script.md`, run the TTS, then
`assemble.py --render --check`. Clip lengths follow the narration, so the
whole cut is re-rendered (about 4 minutes).

Changing a stage's picture: edit `panes/config.py` (captions, GPU spec,
replay set) or the pane module, look at a snapshot, preview the scene at
`-ql`, then re-render only that clip with `--only` and re-mux with
`--check`. A change that alters a stage's END state (the still, caption
or column that the next clip starts from) also needs the next clip
re-rendered, or the seam check will say SEAM.

## 5. The state contract (how separate clips become one video)

Clips are rendered separately and concatenated, yet the panes persist.
`build_state(k)` in `scenes/stage.py` is a pure function of the stage
index and the JSON: it draws the picture "after stage k-1". Scene k adds
it in `setup()`, so its first frame equals the previous clip's last
frame. It animates its transition (sweep the previous blob up, bring in
the new replay, GPU config and matrix column), runs its `stage()` hook,
then `snap()`s: everything is removed, `build_state(k+1)` is added, and
the clip holds on it to `beat + 0.4 s`. Only the intro and the outro
fade. Rules that keep this true:

- Everything that survives to the end of a scene is made by a `panes/*`
  builder with the arguments `build_state(k+1)` will use.
- No `ValueTracker` or updater in stage scenes (the intro is exempt; it
  ends on a fade). Text never changes in place: fade out, fade in.
- `ImageMobject`s live in `Group`, never `VGroup`; one resampling
  algorithm (bicubic) for a picture's whole life (replay, still, thumb).
- Add order is fixed (matrix, GPU, tags, caption, strip, big image last).
- Plays use `fr(n)` (= n/15 - 1e-6 s), holds go through `until()`.
  `np.arange` rounds a play UP to whole frames and `int()` rounds a
  frozen wait DOWN, so `-ql` (15 fps) and `--fps 30` agree to the frame.
- Numbers are never typed into a scene: `panes/data.py` and the
  `{placeholders}` in `config.py` carry them; `build/audit_numbers.py`
  and `data.py --help` are the checks.

## 6. Checks to run before publishing

1. `uv run python scenes/panes/data.py` and `uv run build/audit_numbers.py`
   both exit 0.
2. `uv run build/assemble.py --check` (or `--render --check`): every cut
   `OK` (box-averaged mean below 1.5, under 0.05 % of averaged pixels
   above 16 levels), no `JUMP` inside a hold, no `OVERRUN` you did not
   expect, total length printed.
3. Look at contact sheets (`build/review.py`) of any scene you touched,
   and at the final file around the seams and at 20-30 s (the counter).
4. `git status` clean apart from `.worktrees/`; `out/` and `media/` never
   committed.

## 7. Gotchas (all handled; keep them handled)

- **Memory.** 6 GB RAM, 2 GB swap. Kokoro peaks at 1.9 GB even on the
  CPU with the checkpoint memory-mapped; a 1080p Manim render peaks at
  0.8 GB; the footage generator holds 240 frames plus a CUDA context.
  With two Jupyter kernels and VS Code open the TTS was OOM-killed four
  times; after a WSL restart it ran. Check `free -m` (want ~2 GB
  available) before the TTS, and run heavy steps one at a time.
- **Cairo snapshots the mobject family when a play starts.** Swapping a
  submobject mid-play leaves the old one drawn; `Replay` swaps one
  image's `pixel_array`, `LiveText`/`Stopwatch` blank the old glyphs.
  Always `--disable_caching`: a cached play skips `update_mobjects(0)`.
- **`Replay` reads frames from disk on demand.** Holding a 900 px set as
  97 image mobjects is half a gigabyte and got a render killed.
- **The Short is one continuous take from a moving camera.** Phase
  correlation reported "camera motion" that was the swarm moving, and a
  temporal median background produced garbage. The filter is therefore
  spatial: a white top-hat (gray minus its 9x9 opening), threshold 60,
  blobs under 4 px or wider than 26 px dropped. `--filter median` and
  `stabilise()` remain for a static camera. The wording on screen and in
  the narration is "a filter", not "a motion filter".
- **Labels vs tracks.** The ch06 kernel labels each frame on its own, in
  scan order (its labels are sparse run indices; `make_drone_frames.py`
  makes them dense first). Colours come from a centroid tracker with
  constant-velocity prediction (gate 18 px, coast 8 frames). It is a
  simple tracker: crossings inside a dense swarm can swap identities.
- **The ch06 overview column lives in `ch06_overview_*.json`** on purpose:
  `overview/build.py` and ch06's `figures.py` glob `overview_*.json`.
- **Two narration numbers for the real image.** The matrix's ch06 cell is
  1.55 ms (the 2026-09-27 column session); the caption says 1.46 ms
  (ch06's own 2026-07-25 session, which the READMEs quote). Both are
  labelled.
- **Parallel shell commands share one working directory.** A `cd` in one
  command changed the cwd of another running beside it and one patch
  silently wrote nothing. Use absolute paths in scripts and one-off
  commands. zsh also expands a word starting with `=`.
- **yt-dlp cannot merge** without ffmpeg on its PATH; the video-only
  stream (`bv*`) is all the intro needs.
- **Variants take 6-11 minutes each** (TTS + full render), and each
  narration lands in its own `out/kokoro_<voice>/`; `--voice` on
  `assemble.py` picks the folder and names the output.

## 8. Numbers worth knowing

- Every number on screen comes from a committed JSON through
  `scenes/panes/data.py`: the overview benchmark (`overview_20260724T234423Z.json`),
  the ch06 overview column (`ch06_overview_20260927T013331Z.json`), and
  ch06's headline session (`runs_20260725T161448Z.json`).
- 16,000x is 24,083 / 1.455 = 16,551, floored to two figures. The frame
  budget is 1000 / 30 = 33.3 ms, shown as 33 ms. "81 Mpx" is the real
  image's 81,000,000 pixels.
- ch01-ch04 on N-blob rows are estimates (`est: true`, one launch per
  blob); they are dashed and never glow.
- Footage pipeline, per 576 x 1024 frame with 570-1,210 blobs (medians,
  host numpy unless noted; printed by the generator and written to
  `assets/drones_labels/timing.json`): top-hat filter 4.3 ms, ch06 kernel
  1.3 ms (GPU, CUDA events), label map to host 0.7 ms, blob prep 10 ms,
  tracking 5.9 ms (p90 17 ms; it started at 72 ms before gating, vectorised
  greedy rounds and a float32 matrix product for the distances), painting
  2 ms. None of the host stages is optimised.
- Voices by median pitch (`narration/pick_voice.py`): am_onyx 89 Hz,
  bm_lewis 96, am_echo 111, am_adam 121, am_michael 121, bm_daniel 128,
  bm_george 142, am_fenrir 148. `am_adam` was chosen by ear.

## 9. How it got here (one line per step)

1. First cut: seven two-panel scenes (rejected: no persistent story).
2. Three-pane accumulating cut, one scene per stage, pure state builder,
   seam checker; ch06 measured on the 17 overview shapes; two new ch01
   wavefront GIFs (CPU walk, one block) on the ch02/ch03 square.
3. Matrix columns keep their numbers after their stage.
4. Deep male voice (`am_onyx`) and a 30 s intro on simulated footage.
5. Real drone-show footage, top-hat filter, tracked colours per drone,
   one cut per male voice; `am_adam` chosen; tracker 72 -> 6 ms.
6. Live blob counter and sparkline on the GPU panel.

## 10. Open items and ideas

- Vertical 9:16 cut: the panes are box-relative; a stacked
  `pane_geometry` and a two-row intro are the missing pieces.
- Re-render the 12 other voice cuts with the counter if another voice is
  ever wanted (`build/variants.py <voice>`).
- Add the ch06 column to the README grand table (`overview/build.py`).
- Tracking on the GPU: seed the next frame's fill from the previous
  frame's labels (ch05 takes seeds), which would carry identities without
  a host-side tracker; or a Hungarian/Kalman tracker for dense crossings.
- ElevenLabs voice, background music (`--music`), captions.
