# The flood-fill-cuda explainer video: summary and handoff

Status on 2026-10-02. Branch `video-explainer` (PR #4), folder `video/`,
version 0.4.0. Two layouts and two voices make four cuts in `out/`.
`out/` and `media/` are gitignored; only sources are tracked.

| file | frame | voice | length |
|---|---|---|---|
| `final_landscape_kokoro.mp4` | 1920x1080 | Kokoro `am_adam`, local | 134.5 s |
| `final_landscape_elevenlabs.mp4` | 1920x1080 | ElevenLabs "Peter Baker", `eleven_v4` | 185.7 s |
| `final_vertical_kokoro.mp4` | 1080x1920 | Kokoro `am_adam`, local | 134.5 s |
| `final_vertical_elevenlabs.mp4` | 1080x1920 | ElevenLabs "Peter Baker", `eleven_v4` | 185.7 s |

Both voices were chosen by the user by ear. The other 12 Kokoro voice
cuts were deleted on 2026-10-02.

## 1. What the video shows

| part | length (Kokoro / Peter Baker) | on screen |
|---|---|---|
| intro, beat `intro_problem` | ~16 / 21 s | a drone light show (real footage), then its filter output beside it, then the frame budget: 30 fps, one frame every 33 ms |
| intro, beat `intro_budget` | ~17 / 22 s | the CPU stuck on frame 1 with a real-time stopwatch, the GPU labelling every frame with a live blob counter and a 3 s sparkline, then the verdict: budget 33 ms, pure Python 24,083 ms, @njit 1,346 ms, GPU 1.46 ms |
| stages 00-07 | ~94 / 132 s | three panes for the rest of the video (below) |
| stage 08, outro | ~8 / 10 s | the full strip and matrix, "24,083 ms -> 1.46 ms", "16,000x" |

The three panes, after the intro:

- **The benchmark matrix.** 17 shapes of the overview benchmark as rows,
  one column per chapter. A cell glows blue (teal for ch06) when that
  chapter's fastest measured variant beats the CPU `@njit` time.
  Brighter means a bigger win; grey means slower; dashed means an
  estimate (one launch per blob; never glows); hollow means the chapter
  cannot run the row. Every column keeps its ms printed.
- **The blob.** The chapter's replay, big: the CPU walk, one block, two
  blocks, N blocks, 8-connectivity, two blobs, N blobs (the U merge, then
  the real image), runs. Finished stages sweep up into a strip of
  thumbnails with tags.
- **The GPU.** 24 SM tiles with block chips in the replay's hues, the
  block close-up (warps x 32 lanes), shared and global memory boxes, the
  ch06 kernel strip, and the launch-config lines.

In landscape the panes stand side by side (matrix, blob, GPU). In 9:16
the blob and the GPU share the top row and the matrix takes the bottom
row, and the intro's four footage panels become two rows of two.

## 2. How it is built

```
benchmarks (JSON, committed)  ->  scenes/panes/data.py  ->  matrix cells, captions
chapter GIFs (committed)      ->  assets/extract_gifs.py -> assets/<name>/frame_NNN.png
drone-show Short (yt-dlp)     ->  assets/make_drone_frames.py (+ gpu_tracker.py) -> drones_{sky,mask,labels}/
narration/script.md           ->  narration/tts_kokoro.py -> out/kokoro/<beat>.wav + timing.json
narration/script.md           ->  ElevenLabs connector, one TTS node per beat -> out/elevenlabs_raw/<beat>.mp3
out/elevenlabs_raw/           ->  narration/import_audio.py -> out/elevenlabs/<beat>.wav + timing.json
scenes/*.py (Manim, Cairo)    ->  media/<layout>/videos/<stem>/<res>/<Class>.mp4, one clip per scene
build/assemble.py             ->  concat, wavs at measured clip starts, mux, seam check, out/final_<layout>_<voice>.mp4
```

Stack: Manim Community 0.19.1 (Cairo renderer, `Text` only, no LaTeX),
Python 3.10 in `video/.venv` via uv, Kokoro 0.9.4 for the local voice
(runs on the CPU here), a static ffmpeg from `imageio-ffmpeg` symlinked
into `.venv/bin/ffmpeg`, PyAV and Pillow for checks and frames, yt-dlp
for the footage.

The GPU work runs in the ROOT venv: the footage labelling and tracking,
the ch06 overview benchmark, and the ch01 wavefront GIFs.

## 3. Folder map

```
video/
  pyproject.toml, uv.lock       dependencies (incl. yt-dlp and the spaCy model Kokoro needs)
  narration/
    script.md                   11 beats; each ```text block is one TTS call
    common.py                   parses script.md, trims edge silence, writes timing.json
    tts_kokoro.py               -> out/<--out>/<beat>.wav + timing.json (default voice am_adam)
    import_audio.py             a folder of <beat>.mp3/.wav from any TTS -> out/<--out>/ in the same form
    tts_elevenlabs.py           the same narration through the API (Peter Baker, eleven_v4); needs
                                ELEVENLABS_API_KEY in .env, unused so far (the connector made it)
    pick_voice.py               one sentence in several Kokoro voices, ranked by median pitch
  scenes/
    style.py                    palette, BeatScene (whole-frame padding), helpers; VIDEO_LAYOUT=vertical
                                makes the frame 8 x 14.22 units so text keeps its landscape size
    stage.py                    fr(), Replay (lazy frames), State, build_state(), StageScene
    s_intro.py                  the problem: four vertical footage panels, two beats in one clip
    s00_cpu.py .. s08_outro.py  one thin scene per stage (k = stage index)
    snapshot.py                 STAGE=k [LIVE=1] -> one PNG of a pane state with `-s`
    panes/geometry.py           the pane boxes: three columns (landscape) or two rows (vertical)
    panes/config.py             STAGES: beat, tag, replay set, captions, GpuSpec per stage
    panes/data.py               benchmark JSON -> rows, cells, headline numbers; CLI audit
    panes/left_matrix.py        MatrixPane
    panes/middle_strip.py       strip, big image, captions, the runs row
    panes/right_gpu.py          GpuPane
    BRIEF.md                    the on-screen spec and the rules
  assets/
    extract_gifs.py             chapter GIFs -> assets/<name>/ (gitignored output)
    make_drone_frames.py        footage -> camera / filter / tracked-labels sets + meta.json
    gpu_tracker.py              blob prep, centroid tracker and paint on the GPU, from ch06's run table
    source/                     downloaded footage (gitignored)
  build/
    assemble.py                 --render [--only stem] [--layout vertical] [--voice name] [--no-audio] [--check]
    layout_check.py             every pane state measured without rendering; fails on overlap or overflow
    variants.py                 one full cut per Kokoro male voice (only am_adam's is kept)
    review.py                   frames at given times + a contact sheet
    seam_check.py               every cut and every hold, on 4x4 box-averaged frames
    audit_numbers.py            every printed number recomputed from the JSON, independently
  out/                          narration folders (kokoro, elevenlabs, elevenlabs_raw) and final mp4s
  media/landscape, vertical/    per-scene renders; media/review/ contact sheets and seam images
```

## 4. Commands

All from `video/` unless noted. Never run the TTS, a render and the GPU
work at the same time: the laptop has 6 GB of RAM (section 7).

```bash
uv sync                                                        # once; needs apt libcairo2-dev libpango1.0-dev
uv run assets/extract_gifs.py                                  # chapter GIFs -> frame folders
.venv/bin/yt-dlp -f "bv*[height<=1080]" -o "assets/source/drones_short.%(ext)s" \
    https://www.youtube.com/shorts/p2cDTfSIwqs                 # video stream only; rename to drones_short.mp4
../.venv/bin/python assets/make_drone_frames.py --source assets/source/drones_short.mp4 \
    --start 200 --frames 240 [--tracker both]                  # root venv, GPU; `both` checks host == GPU
uv run python scenes/panes/data.py                             # the matrix as text, exit 1 on a mismatch
uv run build/audit_numbers.py                                  # every on-screen number vs the JSON
uv run build/layout_check.py --layout landscape                # then --layout vertical; both must exit 0
uv run narration/tts_kokoro.py                                 # Kokoro am_adam into out/kokoro/
uv run narration/import_audio.py --src out/elevenlabs_raw --out elevenlabs \
    --voice "elevenlabs:Peter Baker (Ix8C14HEHgIQkJswik2o) eleven_v4"
uv run build/assemble.py --render --check                      # landscape, Kokoro
uv run build/assemble.py --render --check --layout vertical --voice elevenlabs
uv run build/assemble.py --render --only s_intro && uv run build/assemble.py --check   # one clip, then re-mux
uv run build/review.py s03_n_blocks NBlocks --times 0,0.3,0.6,1,1.5,2.6,5,end
STAGE=4 LIVE=1 VOICE=kokoro VIDEO_LAYOUT=vertical .venv/bin/python -m manim render -s -ql \
    --disable_caching -r 540,960 --media_dir media/vertical_preview scenes/snapshot.py Snapshot
```

ElevenLabs narration without an API key: one TTS node per beat with the
claude.ai ElevenLabs connector. Use voice `Ix8C14HEHgIQkJswik2o` and
model `eleven_v4`, one take each. Download each result's mp3 to
`out/elevenlabs_raw/<beat>.mp3`, then run `import_audio.py`. The whole
script is about 2,300 credits.

Changing a spoken sentence: edit `script.md` and re-voice that beat in
both voices. Then run `assemble.py --render --check` for each cut you
want. Clip lengths follow the narration, so every cut is a full
re-render: 6-8 minutes each.

Changing a stage's picture: edit `panes/config.py` or the pane module.
Then run `layout_check.py` for both layouts and look at a snapshot. After
that, re-render only that clip with `--only` and re-mux with `--check`.
A change to a stage's END state also needs the next clip re-rendered,
or the seam check will say SEAM.

## 5. The state contract (how separate clips become one video)

Clips are rendered separately and concatenated, yet the panes persist.
`build_state(k)` in `scenes/stage.py` is a pure function of the stage
index and the JSON: it draws the picture "after stage k-1".

Scene k adds that picture in `setup()`, so its first frame equals the
previous clip's last frame. It animates its transition, runs its
`stage()` hook, then `snap()`s. That removes everything, adds
`build_state(k+1)`, and holds on it to `beat + 0.4 s`. Only the intro and
the outro fade. Rules that keep this true:

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
- Positions come from the pane boxes, never from frame coordinates, so
  one scene renders in both layouts. Narration never says where a pane
  is ("the table", not "on the left").

## 6. Checks to run before publishing

1. `uv run python scenes/panes/data.py` and `uv run build/audit_numbers.py`
   both exit 0.
2. `uv run build/layout_check.py` with `--layout landscape` and with
   `--layout vertical` both exit 0.
3. `uv run build/assemble.py --check --layout <layout> --voice <voice>`
   (or with `--render`): every cut `OK`, no `JUMP` inside a hold, no
   `OVERRUN` you did not expect, total length printed. On 2026-10-02 the
   four cuts passed 36 of 36 seams.
4. Look at contact sheets (`build/review.py`) of any scene you touched,
   and at the final file around the seams and at 20-30 s (the counter).
5. After touching the tracker: `make_drone_frames.py ... --tracker both`
   reports 240 of 240 frames identical.
6. `git status` clean; `out/` and `media/` never committed.

## 7. Gotchas (all handled; keep them handled)

- **Memory.** 6 GB RAM, 2 GB swap. Kokoro peaks at 1.8 GB even on the
  CPU with the checkpoint memory-mapped. A 1080p render peaks at 1.1 GB,
  and the footage generator at 0.8 GB. With VS Code, its Python
  extension and other sessions open, free memory fell to 0.5 GB during
  the 2026-10-02 renders. Check `free -m` before each heavy step and run
  them one at a time.
- **Voices share the media folder.** Clips land in `media/<layout>/`
  whatever the voice, and `assemble.py` takes the newest clip. So render
  and assemble one voice completely before starting the next.
- **Cairo snapshots the mobject family when a play starts.** Swapping a
  submobject mid-play leaves the old one drawn. `Replay` swaps one
  image's `pixel_array`; `LiveText`/`Stopwatch` blank the old glyphs.
  Always `--disable_caching`: a cached play skips `update_mobjects(0)`.
- **FadeIn is a Transform.** It pairs glyphs once, at its start. A text
  that gains glyphs during the fade ("936" -> "1,104") draws without the
  extra ones. The blob counter therefore starts at the live frame and
  holds still for its 0.4 s fade.
- **`Replay` reads frames from disk on demand.** Holding a 900 px set as
  97 image mobjects is half a gigabyte and got a render killed.
- **The Short is one continuous take from a moving camera.** Phase
  correlation reported "camera motion" that was the swarm moving, and a
  temporal median background produced garbage. The filter is therefore
  spatial: a white top-hat (gray minus its 9x9 opening), threshold 60,
  blobs under 4 px or wider than 26 px dropped. The wording on screen and
  in the narration is "a filter", not "a motion filter".
- **Labels vs tracks.** The ch06 kernel labels each frame on its own, in
  scan order. Colours come from a centroid tracker with constant-velocity
  prediction (gate 18 px, coast 8 frames), on the GPU by default. It is a
  simple tracker: crossings inside a dense swarm can swap identities.
- **ch06's `parent[]` is not fully compressed after flatten.** Its path
  halving races, so some runs point at an ancestor, not the root.
  `run_label` is always right. `gpu_tracker.py` walks to the root
  (`_root`) instead of trusting `parent[r]`.
- **Exact host == GPU parity.** The GPU does each float step with
  libdevice `_rn` operations, which are never fused into an FMA, and
  breaks ties by index as numpy's stable sort does. The host computes the
  exact `dx^2 + dy^2` for candidates; the old `|a|^2 + |b|^2 - 2a.b` form
  is off by up to ~0.5 px^2 in float32 here and only pre-selects.
- **The cuts' intro colours are the previous tracker's.** The footage was
  re-tracked on 2026-10-02 after the cuts were rendered. Blobs and masks
  are identical, but one near-tie in frame 1 shifts every later track id,
  so 63% of blob pixels change colour. Re-render `s_intro` to pick up the
  new frames; nothing else changes.
- **The ch06 overview column lives in `ch06_overview_*.json`** on purpose:
  `overview/build.py` and ch06's `figures.py` glob `overview_*.json`.
- **Two narration numbers for the real image.** The matrix's ch06 cell is
  1.55 ms (the 2026-09-27 column session); the caption says 1.46 ms
  (ch06's own 2026-07-25 session, which the READMEs quote). Both are
  labelled.
- **Shell.** Parallel commands share one working directory, so use
  absolute paths. zsh expands a word starting with `=`, and it does not
  split `$var` into words (`set -- $spec` gets one argument).
- **yt-dlp cannot merge** without ffmpeg on its PATH; the video-only
  stream (`bv*`) is all the intro needs.
- **Kokoro is not bit-reproducible.** A re-run gives different samples at
  the same lengths (to 0.01 s). Re-render the cut after re-voicing.
- **ElevenLabs download links expire after 2 hours.** Fetch the mp3s
  right after the generations complete. The connector also queues
  generations beyond its concurrency limit and retries them on its own.

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
- Footage pipeline per 576 x 1024 frame with 570-1,210 blobs (medians,
  2026-10-02): top-hat filter 6.3 ms on the host, ch06 kernel 1.2 ms.
  Then, on the GPU: blob prep 0.84 ms, tracking 0.83 ms, paint 0.17 ms,
  3.3 ms wall with launches and the copy back. The host path for the
  same steps is 22.5 ms: label map 0.7, blob prep 11.6, tracking 7.8,
  paint 2.4. 4,380 tracks over 240 frames.
- Why the tracker matches centroids and does not seed from the previous
  frame's labels: the lights are ~2.2 px in radius and move ~3 px a
  frame. Plain overlap carries 74% of blobs (22% in fast pans) and makes
  82,396 tracks against the centroid tracker's 4,396. With a global
  shift and a 2 px window it still makes 12,873.
- Kokoro voices by median pitch (`narration/pick_voice.py`): am_onyx
  89 Hz, bm_lewis 96, am_echo 111, am_adam 121, am_michael 121,
  bm_daniel 128, bm_george 142, am_fenrir 148.
- Peter Baker reads the script 38% slower than Kokoro am_adam at speed
  1.1: 185.3 s of narration against 134.2 s.

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
7. Branch pushed, PR #4; the other 12 voice cuts deleted.
8. ElevenLabs "Peter Baker" on `eleven_v4` as the second voice.
9. The 9:16 layout, the layout check, the counter's fade fixed, and
   "the table" instead of "on the left" in beat 00.
10. Tracking on the GPU, identical to the host tracker on every frame.

## 10. Open items and ideas

- The Peter Baker cuts run 3:06. YouTube Shorts and Instagram Reels cap
  at 3 minutes, so the 9:16 one needs about 6 s less: a slightly faster
  voice setting, or a shorter line somewhere.
- The top-hat filter (6.3 ms on the host) is now the largest stage of the
  footage pipeline. A GPU opening (a 9 x 9 min, then max) would put the
  whole frame near 3 ms.
- A Hungarian or Kalman tracker for crossings in a dense swarm.
- Add the ch06 column to the README grand table (`overview/build.py`).
- Background music (`--music`) and captions.
