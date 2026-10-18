# The flood-fill-cuda explainer video: summary and handoff

Status on 2026-10-18. On `main` (merged from `video-v5`), folder
`video/`, version 0.10.1. Two cuts of the same story: the three-pane cut (sections
1-10, four files below) and the chain cut (section 11, a "But... So"
chain in three acts, size, count, then tiles, 2.5D). `out/` and `media/`
are gitignored; only sources are tracked.

| file | frame | voice | length | Triton beat |
|---|---|---|---|---|
| `final_landscape_kokoro.mp4` | 1920x1080 | Kokoro `am_adam`, local | 134.5 s | not yet |
| `final_landscape_elevenlabs.mp4` | 1920x1080 | ElevenLabs "Peter Baker", `eleven_v4` | 185.7 s | not yet |
| `final_vertical_kokoro.mp4` | 1080x1920 | Kokoro `am_adam`, local | 146.5 s | yes |
| `final_vertical_elevenlabs.mp4` | 1080x1920 | ElevenLabs "Peter Baker", `eleven_v4` | 200.9 s | yes |

The landscape cuts predate the Triton beat (2026-10-06). The scenes
render it in both layouts and the layout check passes for both, so
`assemble.py --render` in landscape picks it up.

Both voices were chosen by the user by ear. The other 12 Kokoro voice
cuts were deleted on 2026-10-02.

## 1. What the video shows

| part | length (Kokoro / Peter Baker) | on screen |
|---|---|---|
| intro, beat `intro_problem` | ~16 / 21 s | a drone light show (real footage), then its filter output beside it, then the frame budget: 30 fps, one frame every 33 ms |
| intro, beat `intro_budget` | ~17 / 22 s | the CPU stuck on frame 1 with a real-time stopwatch, the GPU labelling every frame with a live blob counter and a 3 s sparkline, then the verdict: budget 33 ms, pure Python 24,083 ms, @njit 1,346 ms, GPU 1.46 ms |
| stages 00-07 | ~94 / 132 s | three panes for the rest of the video (below) |
| stage 08, Triton | ~12 / 15 s | the matrix flips to Numba time ÷ Triton time (one session), the GPU turns purple, "Numba → Triton", "1.12×", "faster in 738 of 934 cases", "chapter 5 still slower: 0.87×" |
| stage 09, outro | ~8 / 10 s | the matrix back in ms, the full strip, "24,083 ms -> 1.46 ms", "16,000x" |

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
triton_twins JSON (committed) ->  data.load_twins()    ->  the Triton stage's cells and card
chapter GIFs (committed)      ->  assets/extract_gifs.py -> assets/<name>/frame_NNN.png
drone-show Short (yt-dlp)     ->  assets/make_drone_frames.py (ch08 labels) -> drones_{sky,mask,labels}/
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
    tts_kokoro.py               -> out/<--out>/<beat>.wav + timing.json (default voice am_adam);
                                --only <beat> voices one beat and keeps the other wavs
    import_audio.py             a folder of <beat>.mp3/.wav from any TTS -> out/<--out>/ in the same form
    tts_elevenlabs.py           the same narration through the API (Peter Baker, eleven_v4); needs
                                ELEVENLABS_API_KEY in .env, unused so far (the connector made it)
    pick_voice.py               one sentence in several Kokoro voices, ranked by median pitch
  scenes/
    style.py                    palette, BeatScene (whole-frame padding), helpers; VIDEO_LAYOUT=vertical
                                makes the frame 8 x 14.22 units so text keeps its landscape size
    stage.py                    fr(), Replay (lazy frames), State, build_state(), StageScene
    s_intro.py                  the problem: four vertical footage panels, two beats in one clip
    s00_cpu.py .. s07_runs.py   one thin scene per stage (k = stage index)
    s08_triton.py               the Triton card; the transition flips the matrix to Numba / Triton
    s09_outro.py                flips the matrix back to ms, then the three centre lines
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
    gpu_tracker.py              blob prep, centroid tracker and paint on the GPU, from ch06's run table (--labeller ch06)
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
    --start 200 --frames 240 [--out DIR]                       # root venv, GPU; ch08's first version, host tracker
uv run python scenes/panes/data.py                             # the matrix as text, exit 1 on a mismatch
uv run build/audit_numbers.py                                  # every on-screen number vs the JSON
uv run build/layout_check.py --layout landscape                # then --layout vertical; both must exit 0
uv run narration/tts_kokoro.py                                 # Kokoro am_adam into out/kokoro/
uv run narration/tts_kokoro.py --only 08_triton                # one beat; the others keep their wavs
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
both voices (`tts_kokoro.py --only <beat>`; one ElevenLabs take into
`out/elevenlabs_raw/<beat>.mp3`, then `import_audio.py`, which rebuilds
every beat from its mp3 exactly). Renaming a beat means renaming its
wav and mp3 too: the 2026-10-06 outro went from `08_outro` to `09_outro`. Then run `assemble.py --render --check` for each cut you
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
   four cuts passed 36 of 36 seams; on 2026-10-06 the two vertical cuts
   with the Triton beat passed 22 of 22. A hold "jump" of about 0.65
   with nothing moving is an I-frame landing on a clip's last frame
   (Peter Baker's s05): encoder noise over the whole picture, not content.
4. Look at contact sheets (`build/review.py`) of any scene you touched,
   and at the final file around the seams and at 20-30 s (the counter).
5. After touching the tracker: `make_drone_frames.py ... --labeller ch06
   --tracker both` reports 240 of 240 frames identical. After touching
   the labelling: regenerate with `--out` into scratch, compare the frames
   with the set in use, and run `numbers.py` (its i0 fact checks the
   labeller, the masks' SHA-1 and the counts against V5).
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
- **Labels vs tracks.** ch08's first version (`OLD_PATH`) labels each
  frame on its own. Colours come from a centroid tracker with
  constant-velocity prediction (gate 18 px, coast 8 frames), on the host.
  It is a simple tracker: crossings inside a dense swarm can swap
  identities. Until v5 the labels came from ch06's run kernel and the
  tracker ran on the GPU (`--labeller ch06`); the ch08 set came out byte
  for byte the same (2026-10-17).
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
- **The Triton stage flips the matrix in place.** Its cells must sit
  where the ms cells sit, dashed and hollow included: `data.py` flags any
  cell whose kind differs (`!!`). `flip_matrix` fades the old pieces out
  one by one, never through a wrapper `VGroup`: FadeOut's clean-up
  removes its own mobject, and a wrapper is not in the scene, so the old
  cells would come back at full opacity.
- **Two sessions on screen in the Triton stage.** The ms matrix is the
  July overview; the twin cells are both backends re-timed on
  2026-10-02. A twin ratio is never one of the ms cells divided by
  another.
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

- Triton (2026-10-06, `results/triton_twins/`): a twin cell is
  `numba_ms / triton_ms` of the grand table's kernel medians, each side
  at its own fastest like-for-like variant of the chapter (ch02 pinned
  is the 2 x 512 twin). 74 of the 89 numbered cells are faster in
  Triton; ch05 is the grey column. The card's 1.12× is the report's
  headline (geometric mean of the 9 unit means, 1.1193), "738 of 934"
  its like-for-like rows, 0.87× ch05's unit mean. The narration says
  "twelve percent faster", and the audit checks that wording.

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
11. The Triton beat before the outro (vertical cuts re-rendered; the
    landscape cuts not yet).

## 10. Open items and ideas

- The 9:16 Peter Baker cut runs 3:21 with the Triton beat.
  YouTube Shorts and Instagram Reels cap at 3 minutes, so it needs about
  21 s less: a faster voice setting, shorter lines, or the
  Kokoro cut (2:27) for Shorts.
- Re-render the landscape cuts to add the Triton beat.
- The top-hat filter (6.3 ms on the host) is now the largest stage of the
  footage pipeline. A GPU opening (a 9 x 9 min, then max) would put the
  whole frame near 3 ms.
- A Hungarian or Kalman tracker for crossings in a dense swarm.
- Add the ch06 column to the README grand table (`overview/build.py`).
- Background music (`--music`) and captions.

## 11. The chain cut (v5 on 2026-10-18; v4, v3 and v2 on 2026-10-14; v1 on 2026-10-13)

A second cut beside the three-pane one, told as a chain of "But... So"
rungs in the style of the Infinite Desk Shorts (`reference/infinite-desk/`,
whose README distils 21 of them). v5 is the current version; v4, v3, v2
and v1 are in git history. v4's two renders are kept in `out/chain_v4/`,
v1's four in `out/chain_v1/`.

| file | frame | look | voice | length |
|---|---|---|---|---|
| `final_chain_iso_vertical_kokoro_paced.mp4` | 1080x1920 | 2.5D | Kokoro `am_adam`, paced | 278.4 s |
| `final_chain_iso_landscape_kokoro_paced.mp4` | 1920x1080 | 2.5D | Kokoro `am_adam`, paced | 278.4 s |
| `share/chain_v5_iso_vertical_kokoro_small.mp4` | 1080x1920 | 2.5D | the same, phone copy (22.2 MB) | 278.4 s |
| `share/chain_v5_iso_landscape_kokoro_small.mp4` | 1920x1080 | 2.5D | the same, phone copy (19.7 MB) | 278.4 s |

### v5: tiles, like for like (278.4 s)

The user dropped v4's run-based beat (`r10_runs`): the video is about
the BFS. A third act replaces it. The queue carries 32 x 32 tiles (ch07),
then tiles find their blob pieces and seeds send waves (ch08).

ch06 is out entirely: no "CCL", "runs", "spans" or "union-find", spoken
or shown, and no number from it. No GPU number comes from ch08's
dense-tile path (K2b/K4b, a run union-find inside tiles).

Every CPU-vs-GPU number was re-made like for like, in one session. The
ratios got smaller (r05 18x -> 8.5x, r06 25.8x -> 11x); the user
accepted that.

The user's decisions (2026-10-17), all binding:

1. One session: every CPU-vs-GPU number comes from V5 (below). No July
   number is set against a CPU.
2. Like for like: the CPU does exactly the GPU's work. The same
   neighbours (4 against 4, 8 against 8), the same output (no depth map
   where the GPU keeps none; many blobs: labels only) and the same setup
   (arrays allocated once, reset where the GPU resets its own).
3. The finale says both sides of the small blob aloud: in kernel time
   the GPU wins now; with the copy to the card, the CPU still wins.
4. Title card: "flood fill, from pixels to tiles" over "flood-fill-cuda".
5. Kokoro `am_adam`, paced. Cairo renders in parallel on the CPU; the
   GPU only measures. Act 3 at the end, mechanism and numbers. At most 5
   minutes.

Defaults adopted with them: r13 keeps "once the frame is on the card"
(with the copies the frame takes 54.8 ms; the r14 footnote says so); r11
compares against r06's kernel; act 2's ch05 is timed at full clocks;
every tile-wave number is ch08's first version (`OLD_PATH`, no dense-tile
path at all); the 900 px frame is transposed, so one orientation holds
everywhere. The design, both critics' reviews and the open questions are
in `narration/chain_v5_proposal.md`.

The session (V5): `results/overview/benchmark_results/v5_20261017T192640Z.json`
(and `.csv`), from `overview/bench_v5.py` with `overview/cpu_twins.py`,
both at the committed HEAD (`script_matches_head`).

- GPU: kernel-only CUDA events. All 8 timed blocks at full clocks (P0
  only, memory 8,001 MHz, SM median 2,070 MHz). ch05 on the frame got
  there on its second attempt; its first sat in P3 under the power cap.
- CPU: `@njit` on one core, after all GPU work and 60 s of idle, 5
  interleaved rounds behind a quiet-host gate. The round medians spread
  by up to 19%, so factors show two significant figures, and each one
  is checked over the CPU's rounds.
- The twins (תאומים) run each oracle's loop on buffers kept between
  calls: `cpu_bfs4` (ch01-ch03 at 4 ways), `cpu_bfs8` (ch03 at 8 ways),
  `cpu_fill8` (ch07: no depth, reset on the clock), `cpu_label` (ch08 and
  the drones: `cpu_label_components`' loop, reset on the clock) and
  `cpu_label_pre` (act 2, reset before the clock).
- Two earlier v5 sessions stay committed but are not read: `161122Z`
  predates the twins, and `182449Z` ran the oracles as called, so the
  CPU paid for its allocations (critic 1, F1).
- Mechanism numbers (rounds, seeds, pieces) come from the tiles' own
  sessions (T7, D8, W8) and the act-3 exports in `assets/tiles/`. Act
  1's GPU-only numbers (spill, placement, sweep, snake) still come from
  the July sessions; none of them is set against a CPU.

| beat | start, s | length, s | the link |
|---|---|---|---|
| intro (i0, i1) | 0.00 | 26.73 | drones, 33 ms; drone card CPU 0.679 ms, GPU "?"; 81 Mpx: the CPU needs 225 ms; "size and count" |
| r00 zoom | 26.73 | 5.73 | one drone; "Freeze" is a white outline pulse round the floor, not a flash |
| r01 CPU | 32.47 | 9.73 | one pixel per step, one core, 24 SMs idle (two captions) |
| r02 one block | 42.20 | 15.73 | threads and block defined; on one small blob the CPU wins, 4.3x (0.121 vs 0.520 ms); row 1 |
| r03 giant | 57.93 | 13.70 | 48 KB per block, the queue needs 125 KB: it spills to global memory |
| r03b one SM | 71.63 | 6.53 | one block a little faster than the CPU, 1.1x; 1 of 24 SMs |
| r04 two blocks | 78.17 | 24.93 | the queue moves to global memory; twice as fast (GPU vs GPU); 22 SMs dark |
| r05 all blocks | 103.10 | 23.60 | 48 blocks, 8.5x; their pixels mix; the launch-shape sweep, +15% at best |
| r05b snake | 126.70 | 9.37 | 48 blocks lose to one: the gate, twice per ring, ~2 us (the caption on "each") |
| r06 diagonals | 136.07 | 7.53 | half the rings; 11x against the 8-neighbour CPU; row 2 |
| r07 real frame | 143.60 | 11.33 | 2,522 launches, ~1,462 ms (estimate): the CPU (202 ms) wins by 7.2x |
| r08 tags | 154.93 | 20.40 | two per launch, ~850 ms: faster, but the CPU still wins by 4.2x |
| r09 seeds | 175.33 | 13.00 | 53.9 ms: faster than the CPU (3.8x, row 3), over 33 ms: every red pixel rides the queue |
| r10 tiles | 188.33 | 24.13 | 32 x 32 tiles; one warp per tile, a row per thread in registers; row, diagonal, column fills until nothing changes |
| r11 rounds | 212.47 | 19.53 | border bits join the next round; one gate per round; 4,001 rings -> 127 rounds, 101 -> 3.12 ms; row 2 340x |
| r12 pieces | 232.00 | 17.17 | a 384 px crop as its own image: 135 pieces, 36 seeds, 8 blobs; waves meet, the smallest label wins |
| r13 frame | 249.17 | 12.00 | 11,609 seeds, 6 rounds, 2,522 blobs: 2.21 ms once the frame is on the card; row 3 100x |
| r14 reveal | 261.17 | 17.27 | the card (0.192 / 0.044 ms GPU 4.4x, 1,065 / 3.12 ms 340x, 225 / 2.21 ms 100x), copies in its footnote; from "CPU" a note under the small-blob pill, "with the copy" / "CPU 3.5x", to the end; the drones ("GPU + copies: 1.26 ms"); the title card |

Files:
- Narration: `narration/script_chain.md` (v5, 19 beats, k = 0..18) and
  `narration/chain_v5_proposal.md`. Numbers and claim checks:
  `scenes/chain/numbers.py`.
- Scenes: new `r10_tiles.py`, `r11_rounds.py`, `r12_pieces.py`,
  `r13_frame.py`, `r14_reveal.py`; the intro, r02, r03b, r05, r06, r07,
  r08 and r09 changed. `state.py` holds the 19 beats' states,
  `machine.py` the warp card and 3 blocks per SM. v4's `r10_runs.py` and `r11_reveal.py` stay in the
  folder, out of the cut.
- Act-3 data: `chapters/ch07_gpu_1blob_tiles/benchmarks/video_tiles.py`
  and `chapters/ch08_gpu_nblob_tiles/benchmarks/video_waves.py` record
  the floors into `assets/tiles/` (its README; tracked through a
  `.gitignore` exception). `assets/make_frame_mask.py` writes
  `assets/tiles/input_blobs_red.npz`, every red pixel of the frame, for
  r13's zoom.
- The drone footage is labelled by ch08's first version now (section
  7, "Labels vs tracks").
- Build: `assemble.py --jobs N --wrap CMD` renders clips in parallel,
  each in its own media folder; `seam_check.py --intra` checks the
  intro's snap; `tts_kokoro.py` voices each side of a semicolon as its
  own piece.

How it was rendered (2026-10-18): Kokoro `am_adam`, then `pace.py`;
`assemble.py --cut chain --look iso --layout <layout> --voice
kokoro_paced --render --check` with `--jobs` and a `--wrap` script that
lets at most 3 renders run at once (memory), one layout after the other.

- `--check` stops before the mux when a seam is flagged. After the three
  seams below were reviewed, a second run without `--render` or
  `--check` muxed each layout.
- Phone copies: v4's settings (x264 preset slow, CRF 27, AAC 128k,
  faststart).
- v4's clips in `media/chain_iso/{landscape,vertical}/` and its
  `work_chain_iso_*_kokoro_paced` folders were overwritten.

Checks:
- `numbers.py`: 116 claim checks, all ok. `build/audit_numbers.py` has
  no chain mode (it audits the three-pane cut, and still passes), so
  `numbers.py` is the chain cut's audit. Every on-screen number is a
  `numbers.py` value, and `STORY_SHOWN` ties every shown CPU-vs-GPU
  factor to V5's own story entry.
- Snaps: all 17 pass in both layouts (largest 1.02 in 9:16, 0.82 in
  16:9, limit 1.5). r14 ends on a fade, so the check skips it.
- Seams: 14 of 17 pass. Three are flagged (16:9 / 9:16): r05 -> r05b,
  the 48-block owner mosaic (0.133% / 0.174%, as in v4); r09 -> r10, the
  speckled label frame (0.052% / 0.084%, below v4's); r13 -> r14, new,
  the 81 Mpx labelled frame (0.148% / 0.215%).
- The flagged pixels are 67-258 scattered patches of 1-4 boxes (4x4 px),
  at most 22-30 levels. Both frames show the same picture: a clip ends
  on a P-frame and the next opens on a keyframe. Nothing moved.
- The final files: video and audio both 278.433 s, 8,353 frames. The
  frame 0.5 s into each clip matches that clip's own frame, so the clip
  order holds. Every beat's voice starts 0.12-0.14 s into its slot.
- `assemble.py` prints OVERRUN for every clip, as designed: each chain
  scene holds at least 0.75 s after its beat. Speech plus gaps is 258.8
  s; the proposal's model said 286.9 s.
- Contact sheets of 20 frames per layout; the title card is up at
  277.5 s.

The final check (2026-10-18) found four problems, all in the picture; no
word changed. Only those four clips were re-rendered, in both layouts,
then muxed again. The seams and snaps read as before, and the three
flagged seams flag to the same figure. The phone copies were rebuilt
with the same settings.
- F1, r14: the closing strip read as a GPU win on every row, drones
  included. The CPU's side ("with the copy, still the CPU") lived only
  in the footnote, which leaves with the card. Now, from "CPU", a grey
  note pill hangs under the small-blob pill: "with the copy" / "CPU
  3.5x" (`small_e2e_cpu_x`, checked against V5's own `twin.x_e2e`). It
  stays to the end, where the inset was. Caption 3's sub names whose
  time it is: "GPU + copies: 1.26 ms".
- F2, r01: three captions in a 4.5 s sentence. The SMs' caption landed
  after the line and stood 1.2 s. Now there are two: "1 pixel per step"
  / "1 core of 14", then "24 SMs" on "SMs" (up 2.1 s).
- F3, r05b: the last caption waited a full dwell after the "grid.sync"
  tag, so it came 1.4 s after its line. Now it lands on "each" (up
  2.6 s). The gate comes down with its tag as the race ends, just after.
- F4, r00: "Freeze" was a two-frame white flash over the whole floor
  (12.6% / 17.4% of the frame). Now it is a white outline pulse round
  the floor, so no large area flashes.

Gotchas found in v5:
- The 900 px frame PNGs are in PNG order, act 3's arrays [x, y]:
  `state.picture()` transposes the PNG side once, and `numbers.py`
  checks it (overlap 0.78 transposed, 0.10 as is).
- `card_ms` truncates 10-100 ms for spoken values; the r14 footnote uses
  `card_ms_note`, which rounds, so a cost never reads below its measure.
- A flashing element flashes at most twice per second (the cut's rule
  allows three), and large areas never flash.

Open (2026-10-18): the user has not reviewed the v5 renders yet; the
cut is on `main` (0.10.1). The proposal's open questions stay
open: Q8 (where r11's time goes), N2 (act 2's seeds beat the CPU on the
small blob, unseen), N3 ("as a thirty-two bit word" in r10) and N4
(r03b's thin margin, 1.02-1.17 over the CPU's rounds).

### v4: the memory story

The user liked v3 ("WAY better") and asked for 25% more time between
sentences and seven additions: the one-block start with its queue in
shared memory (and its 48 KB limit), the queue moving to global memory,
neighbours written there, the threads-per-block search, a colour per
block on the huge blob, the global-memory layer (the queue and its tags,
visited), and data moving between VRAM and the SMs (coalescing). The
user allowed up to 5 minutes. Six readers and seven adversarial checkers
held each point against the repo first (`narration/chain_v4_proposal.md`,
"Where v4 departs from the user's wording"): there was never a
blob-per-SM plan; 48 KB is the default per-block limit (the ring used
32 KB, the giant square needs about 125 KB); the queue went global in
two steps (ch01 v2's spill, then ch02); only ch03 swept to a plateau
(+15%, never adopted); blocks share one queue, so their pixels mix; a
tag is one spare bit of a queue word and ch04 takes two seeds;
coalescing was never profiled and ch06's measured win is 25x fewer items.

- 16 beats: two split off (`r03b_one_sm`, `r05b_snake`); r02, r03, r04,
  r05, r08 and r10 say more (`script_chain.md`). Every new number is
  pinned in `numbers.py` with claim checks (59).
- New data: `chapters/ch03_gpu_1blob_nblock/benchmarks/owner_map.py`
  (results `results/ch03_gpu_1blob_nblock/owner_map/`): which block
  painted each pixel of the 64 Mpx square for 2 x 256, 48 x 256 4-way and
  48 x 256 8-way. Its per-block counts equal the pinned session's.
- New visuals (`machine.py`, `die.py`, `overlays.py`): the SM close-up
  with its queue in shared memory; the shared-memory gauge that
  overflows into VRAM; per-SM shelf cards; a global-memory close-up
  (image, visited and queue lanes, a 32-bit entry with its tag bit, one
  dense warp load, a row read as one 96 B packet "coalesced, by design,
  not profiled"); load and store packets between VRAM and the SMs; block
  hues, real owner mosaics and an owner lens; busy and idle cubes per
  ring; the launch-shape sweep card. Explaining panels open large over
  the floor area and fold back into their hardware.
- Pacing: sentence pauses 1.125 s, colon pauses 0.56 s, BREATH 0.75 s,
  MIN_DWELL 1.75 s.
- How v4 was made: a research workflow (6 readers, 7 verifiers, a
  designer, a critic), the user's answers, a data and foundation workflow
  (owner map, script and numbers, shared modules, a reviewer), then a
  scene workflow (a builder and a reviewer per scene, 24 agents).
- Checks at 1080p30 (v4, Kokoro paced, 217.5 s): all 13 snaps pass in
  both layouts (largest 1.03 in 9:16, 0.83 in 16:9) and 11 of 14 seams.
  The three flagged seams sit on high-detail pictures: r05 -> r05b (the
  48-block owner mosaic, 0.174% / 0.133%), r09 -> r10 (the speckled label
  frame, 0.221% / 0.163%) and r10 -> r11 (0.113% / 0.103%). Their
  difference maps are only element outlines (a clip's last P-frame
  against the next clip's keyframe), nothing moved or missing. Phone
  copies: `out/share/chain_v4_iso_{vertical,landscape}_kokoro_small.mp4`
  (17.5 / 15.6 MB).
- Parallel renders that share one `media_dir` can race on Manim's text
  cache (a missing `texts/*.svg`): render the same layout one at a time,
  or give each job its own media folder.

### v3: real hardware, less text, more air (138.6 s)

The user's notes on v2: the CPU sat at the head of the GPU's queue
conveyor, which read as "the CPU is on the memory queue"; the GPU should
look like NVIDIA's Ada diagrams, with the CPU in a parallel diagram of
its own; more time between changes; less text; ElevenLabs "Adam". The
script did not change. The plan with every rule is
`narration/chain_v3_plan.md`.

- The machine is two boards (`scenes/chain/die.py`, `machine.py`): the
  i9-13900H (6 P-cores, 8 E-cores, L3 24 MB, Xe graphics) and the RTX
  4060 Laptop GPU drawn as AD107 (3 GPCs x 4 TPCs x 2 SMs, each SM in
  NVIDIA's layout, L2 32 MB as the device reports, GigaThread, PCIe,
  memory controllers, four VRAM chips), joined by a PCIe link. The
  P-core lights while the CPU code runs; blocks land on SMs as cubes; an
  SM close-up appears when the story first moves onto one SM; the queue
  lives in VRAM; launches and seeds travel over the link; ch06's six
  kernels queue in the launch lane. The algorithm gadgets of v2 (stencil
  badge, tag printer, "64 max" rack, min unit) are gone from the boards.
  AD107's floorplan is not public: the GPC count comes from 3DNews and
  Wccftech's configuration tables, the plan follows NVIDIA's Ada figures.
- Less text: no "SMs busy" gauge, a one-line meter and inset, no strip
  header, at most three captions per beat (a big word and at most a short
  qualifier), small text set at 4x so Pango keeps its spaces.
- More air: `narration/pace.py` stretches the pause after each sentence
  to 0.9 s and after a colon to 0.45 s, shifting the word times with it;
  every beat ends with at least 0.6 s of silence (`BREATH`); a caption
  stays at least 1.4 s (`MIN_DWELL`) before the next replaces it.
- Voices: `kokoro_paced` (`out/kokoro_paced_chain`, from `kokoro_chain`).
  ElevenLabs Adam is premade `pNInz6obpgDQGcFmaJgB` ("Adam - Dominant,
  Firm"), `eleven_v4`, made with the claude.ai connector (flow
  `pEZs8Czt8aDISzS8Ye62`): 11 of 14 takes are in `out/adam_chain_raw/`.
  r01_cpu, r07_real_frame and r10_runs failed on 2026-10-14 because
  ElevenLabs disabled the account's free tier ("unusual activity").
  When the three takes exist: `import_audio.py --src out/adam_chain_raw
  --out adam_unpaced_chain --script script_chain.md --voice "elevenlabs:Adam
  (pNInz6obpgDQGcFmaJgB) eleven_v4"`, then `pace.py --src
  adam_unpaced_chain --out adam_chain --script script_chain.md`, then
  render with `--voice adam`. `import_audio.py` now pins word times on
  the take's own pauses (`narration/align.py`).
- How v3 was made: the shared modules first (two commits), then a
  workflow with one agent per scene and an independent reviewer per
  scene (26 agents), then the reviewers' shared findings fixed by hand.

Checks at 1080p30 (v3, Kokoro paced): all 11 snaps pass in both layouts
(largest 1.00 in 9:16, 0.82 in 16:9) and 11 of 12 seams. The flagged
seam is r10 -> r11 again (0.113% / 0.103% of pixels against 0.05%): its
difference map is only the outlines of every element (the last frame of
one clip against the keyframe that opens the next, on the speckled teal
picture), nothing moved or missing. Compressed copies for phones:
`out/share/chain_v3_iso_{vertical,landscape}_kokoro_small.mp4` (12.6 /
11.4 MB). v2's cuts (Kokoro unpaced, 115.9 s) passed all 11 snaps.
v2 at 1080p30: all 11 snaps (largest 0.97) and 11 of 12 seams. The
flagged seam, r10 -> r11 (0.09% / 0.08% of pixels against 0.05%), is
the speckled teal picture: at most 79 levels, spread over the whole frame,
i.e. keyframe noise, not content (v1's only flag was the same picture).

The story (v2 and v3, "size, then count"): a drone show opens it. On frames that
small the compiled CPU keeps up (measured: 3.1 ms vs the GPU's 0.93 ms per
drone frame), but at 81 Mpx it needs 1.3 s against a 33 ms budget. "Where
does the GPU win? Two places: size and count." The video zooms into one
real drone (119 px), walks it on the CPU, races one GPU block against it
(the CPU still wins on one small blob), then earns the two wins: act 1,
SIZE, on one giant 64 Mpx blob (1 block, 2 blocks, 48 blocks and the gate
on a thin snake, diagonal steps: 25.8x); act 2, COUNT (one launch per blob
loses, tags, seeds found on the GPU, runs: 2.96 ms). A three-row CPU vs
GPU scoreboard fills as the acts are won, and the reveal answers the
opening question on the live drones. The name is on the title card only.

| beat | the link |
|---|---|
| intro (i0, i1) | drones, 33 ms; drone card CPU 3.1 / GPU 0.93 ms; 81 Mpx card CPU 1.3 s; "size and count" |
| r00 zoom | one drone, 119 pixels |
| r01 CPU | one pixel per step, one core, 24 SMs idle |
| r02 one block | a whole ring per step, but on one small blob the CPU wins (0.15 vs 0.37 ms); row 1 |
| r03 giant | one block on 64 Mpx: 2.46x the CPU; 1 of 24 SMs |
| r04 two blocks | twice as fast (5.0x); 22 SMs dark |
| r05 all blocks | 48 blocks, 18x; on the snake they lose to one: the gate, twice per ring, ~2 us |
| r06 diagonals | half the rings, half the gates (on a square): 25.8x; row 2 |
| r07 real frame | size is won, now count: 2,522 launches, ~2,181 ms (estimate), slower than the CPU |
| r08 tags | two per launch, ~1,368 ms (estimate); 64 tags max, seeds handed in |
| r09 seeds | the GPU finds its seeds, waves merge by min: 51.0 ms, still over 33 ms |
| r10 runs | spans, not pixels; no queue, no ring loop, six kernels: 2.96 ms (1.46 from a packed mask); row 3 |
| r11 reveal | the full scoreboard; every drone, every frame |

Files: narration `narration/script_chain.md`; the approved design with
every source `narration/chain_v2_proposal.md`; scenes `scenes/chain/`
(`intro.py`, `r00_zoom.py` ... `r11_reveal.py` on `base.ChainScene`;
`state.py` is the boundary table, `overlays.py` the meter, scoreboard,
inset and act tag; `machine.py` and `die.py` the two boards); numbers and claim checks
`scenes/chain/numbers.py` (28 checks). The drone-frame benchmark is
`src/flood_fill_cuda/overview/bench_drones.py`, results
`results/overview/benchmark_results/drones_20261014T023648Z.json`.

How v2 was made: a design workflow (6 readers, 4 drafts, 3 judges, 3
adversarial verifiers), a revision workflow after the user's answers, a
committed drone benchmark, then a build workflow (a builder, an
adversarial reviewer and a fixer per group of scenes) on a foundation of
states and overlays.

Numbers worth knowing (all in `numbers.py` with file:key):
- Drone frames (Oct 14, one session, 240 frames, all crosschecks OK):
  @njit label + fill 3.14 ms median, 5.13 max; ch06 RGB 0.93 / 1.81 ms;
  pure Python 68 ms median. The host CPU ran about 3x slower than on Jul 24
  (png_blocks @njit 35.8 vs 11.2 ms), so the video never claims that pure
  Python misses the drone budget.
- Scoreboard: one small blob (16 K px) CPU 0.15 vs best GPU 0.37 ms (one
  session); one huge blob (64 Mpx) CPU 2,584 vs GPU 100 ms, 25.8x (one
  session, kernel time); many blobs (1 Mpx, 47,655 blobs) CPU 22.9 vs GPU
  0.65 ms, 35x (two sessions).

Commands (from `video/`):

```bash
uv run narration/tts_kokoro.py --script script_chain.md --out kokoro_chain
uv run narration/tts_elevenlabs.py --script script_chain.md --out elevenlabs_chain   # needs ELEVENLABS_API_KEY in .env
uv run python -m scenes.chain.toy && uv run python -m scenes.chain.numbers
uv run narration/pace.py --src kokoro_chain --out kokoro_paced_chain --script script_chain.md
uv run build/assemble.py --cut chain --look iso --layout vertical --voice kokoro_paced --render --preview   # 480 px, 15 fps
uv run build/assemble.py --cut chain --look iso --layout vertical --voice kokoro_paced --render --check     # 1080p30
uv run build/seam_check.py --cut chain --look iso --layout vertical --intra
uv run build/chain_snapcheck.py r05_all_blocks AllBlocks --layout vertical                            # one clip
STATE=7 LOOK=iso VIDEO_LAYOUT=vertical .venv/bin/python -m manim render -s -qh --disable_caching \
    -r 1080,1920 --media_dir media/chain_preview scenes/chain/snapshot.py Snapshot
```

Gotchas found while building the chain cut:
- An updater must be a plain function, never a bound method of the
  scene: `FadeIn` deep-copies the mobject, and copying a bound method
  copies the scene's thread lock. The render then crashes and hangs.
- Hooks read the renderer's clock: a play's time steps start at t = 0,
  so summing dt loses a frame per play (a one-frame wait never advances).
- A play's last update is never drawn, so `settle()` draws one frozen
  frame after the hooks finish; without it a snap jumps.
- Cairo ignores shear on images: the iso floor warps its raster itself.
- Do not pipe a render into `head`: an early exit can leave Manim
  blocked on its console.
- At 480 px the seam check can flag busy pictures by a hair (encoder
  noise between a clip's last P-frame and the next clip's I-frame).
  Judge seams at 1080p30.
- On 2026-10-13 `assemble.py` without `--render` rebuilt
  `final_vertical_kokoro.mp4` from the newest clips in `media/vertical/`,
  which were Peter Baker's (section 7, "Voices share the media folder").
  It was restored with a full re-render (146.53 s, 10 of 10 seams OK).
