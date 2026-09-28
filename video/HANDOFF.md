# Handoff: the three-pane explainer video

Status on 2026-09-28: the cut is ten Manim clips, about 129 s, in
`video/` on branch `video-explainer`, narrated by Kokoro's deep male
voice `am_adam` by default (chosen by ear on 2026-09-28 from the 13 cuts
`build/variants.py` renders into `out/final_landscape_kokoro_<voice>.mp4`). A 30 s intro states the problem (a defence camera must label every
drone in every frame at 30 fps; real drone-show footage, motion mask,
CPU vs GPU). Then three panes stay on screen and accumulate: the benchmark matrix on the left (17 shapes,
one column per chapter, every column keeps its ms, a cell glows when the
GPU beats the CPU), the
chapter's blob in the middle (finished chapters sweep up into a strip),
and the GPU schematic on the right (which SMs, blocks, threads and memory
each chapter uses). Landscape only. See `scenes/BRIEF.md` for the spec.

## 1. The stack

| Role | Tool | Version | Why this one |
|---|---|---|---|
| Animation engine | Manim Community | 0.19.1 | 3Blue1Brown look, deterministic frames, no browser |
| Environment | uv | Python 3.10 in `.venv` | one `uv sync` reproduces it |
| Narration | Kokoro (`kokoro` + `misaki`) | 0.9.4 | free, local; `am_adam` (male, 121 Hz median; `am_onyx` is the deepest at 89 Hz), `narration/pick_voice.py` ranks voices by pitch |
| Muxing / concat | ffmpeg static via `imageio-ffmpeg` | 7.0.2 | symlinked into `.venv/bin/ffmpeg` |
| Frame extraction, review sheets | Pillow, PyAV | | GIF frames in, review frames out |
| Text shaping | ManimPango + pycairo | built from source | needs `libcairo2-dev libpango1.0-dev` once |

## 2. Folder layout

```
video/
  pyproject.toml, uv.lock       dependencies
  narration/
    script.md                   the 9 beats; each ```text block is one TTS call
    common.py                   parses script.md, writes timing.json
    tts_kokoro.py               -> out/kokoro/<beat>.wav + timing.json (CPU, memory-mapped model)
    pick_voice.py               one sentence in several voices, median pitch per voice
    tts_elevenlabs.py           alternative voice, needs ELEVENLABS_API_KEY in .env
  scenes/
    style.py                    palette, BeatScene (whole-frame finish), FrameSequence, helpers
    stage.py                    fr(), Replay, State, build_state(), StageScene
    panes/geometry.py           the three pane boxes
    panes/config.py             STAGES: beat, tag, replay set, captions, GpuSpec per stage
    panes/data.py               benchmark JSON -> matrix rows, headline numbers; CLI audit
    panes/left_matrix.py        MatrixPane
    panes/middle_strip.py       strip, big image, captions, the runs row
    panes/right_gpu.py          GpuPane
    snapshot.py                 STAGE=k [LIVE=1] -> one PNG with `-s`
    s_intro.py                  the problem: four footage panels, two beats in one clip
    s00_cpu.py .. s08_outro.py  one thin scene per stage
    BRIEF.md                    the spec
  assets/extract_gifs.py        parent-repo GIFs -> assets/<name>/frame_NNN.png (gitignored output)
  assets/make_drone_frames.py   intro footage: camera, motion mask, ch06-labelled frames (root venv);
                                --source for real footage (median background), else a simulation
  assets/source/                downloaded footage (gitignored): drones_short.mp4 = Short p2cDTfSIwqs
  build/
    assemble.py                 render, concat, narration at measured clip starts, mux; --check
    review.py                   frames at given times + a contact sheet
    variants.py                 one full cut per Kokoro male voice
    seam_check.py               cut-to-cut and hold checks
    audit_numbers.py            every printed number vs the JSON, independently
  out/, media/                  narration, renders, review frames (gitignored)
```

## 3. The pipeline

1. Parent repo, once: `uv run python -m
   flood_fill_cuda.chapters.ch01_gpu_1blob_1block.benchmarks.wavefront`
   (the CPU-walk and one-block GIFs) and `uv run python -m
   flood_fill_cuda.overview.bench_ch06` (the ch06 column on the 17
   overview rows, `results/overview/benchmark_results/ch06_overview_*.json`).
   Both outputs are committed.
2. `uv run assets/extract_gifs.py` unpacks the GIFs into `assets/`.
   `.venv/bin/yt-dlp -f "bv*[height<=1080]" -o "assets/source/drones_short.%(ext)s"
   https://www.youtube.com/shorts/p2cDTfSIwqs` fetches the footage (video
   stream only; rename to `drones_short.mp4`), then
   `../.venv/bin/python assets/make_drone_frames.py --source assets/source/drones_short.mp4 --start 1020 --frames 240`
   (root venv, GPU) renders the three intro sets. The window was picked by
   phase correlation: 34-41 s is the only 8 s where the camera holds still.
3. `uv run python scenes/panes/data.py` and `uv run build/audit_numbers.py`
   print and check every number the video shows.
4. `uv run narration/tts_kokoro.py` renders each beat to a wav, trims edge
   silence, and writes `timing.json`. Scenes read their beat length from
   it (`VOICE=kokoro`) and fall back to `Stage.target_s` when the beat is
   missing.
5. `uv run build/assemble.py --render --check`: renders the nine scenes at
   1080p30 with `--disable_caching`, concatenates them, places every beat's
   wav at its clip's measured start, muxes AAC, prints each clip against
   its beat, warns outside 60-120 s, and runs the seam check.

## 4. Commands

```bash
cd /home/lrn/Repos/flood-fill-cuda/video
uv sync
uv run assets/extract_gifs.py
../.venv/bin/python assets/make_drone_frames.py --source assets/source/drones_short.mp4 --start 1020 --frames 240   # intro footage (root venv, GPU)
uv run build/variants.py [voice ...]         # one full cut per male voice
uv run python scenes/panes/data.py            # the matrix as text
uv run build/audit_numbers.py                 # numbers vs JSON
uv run narration/tts_kokoro.py                # voice + timing
uv run build/assemble.py --render --check     # full 16:9 cut -> out/final_landscape_kokoro.mp4
uv run build/assemble.py --render --only s03_n_blocks   # one clip
uv run build/assemble.py --no-audio --check   # silent cut from existing renders
uv run build/review.py s03_n_blocks NBlocks --times 0,0.3,0.6,1,1.5,2.6,5,end
STAGE=4 LIVE=1 VOICE=kokoro .venv/bin/python -m manim render -s -qh --disable_caching --media_dir media/landscape scenes/snapshot.py Snapshot
VOICE=kokoro .venv/bin/python -m manim render -ql --disable_caching --media_dir media/landscape scenes/s03_n_blocks.py NBlocks
```

## 5. How a stage works (the state contract)

Clips are rendered separately and concatenated, yet the panes persist.
`build_state(k)` is a pure function of the stage index and the JSON: it
draws the picture "after stage k-1". Scene k adds it in `setup()`, so its
first frame is exactly the previous clip's last frame; it animates its
transition (sweep, new replay, GPU config, matrix column), runs its
stage, then `snap()`s: everything is removed, `build_state(k+1)` is
added, and the clip holds on it to `beat + 0.4 s`. `build/seam_check.py`
compares each cut on 4x4 box-averaged frames (encoder noise averages out,
a moved element does not) and reports the largest jump inside every hold.

## 6. Gotchas (all handled, keep them handled)

- Play frames are counted with `np.arange` (rounds up); frozen waits with
  `int()` (rounds down). `fr(n) = n/15 - 1e-6` for plays, `n/15 + 1e-6`
  for frozen holds, so 15 fps previews and the 30 fps render agree.
- The Cairo renderer snapshots the mobject family when a play starts:
  never swap submobjects, swap one image's `pixel_array` (`Replay`) and
  never change text in place (fade out, fade in). Removed-mid-play
  mobjects keep being drawn.
- `--disable_caching` always: a cached play skips `update_mobjects(0)`.
- One resampling algorithm for a picture's whole life (replay frame,
  still, thumbnail), or the snap pops.
- Add order is fixed (matrix, GPU, tags, caption, strip, big image last):
  Cairo redraws the family from the first moving mobject onward.
- `Replay` reads frames from disk on demand; holding 97 image mobjects
  of a 900 px set is half a gigabyte and got a render OOM-killed on this
  6 GB laptop.
- The ch06 column file is named `ch06_overview_*.json` on purpose:
  `overview/build.py` and ch06's `figures.py` glob `overview_*.json`.
- zsh expands a word starting with `=`; do not `echo =====`.

## 7. Numbers worth knowing

- Every number is from a committed JSON through `scenes/panes/data.py`;
  `build/audit_numbers.py` is the independent check.
- The matrix's ch06 cell on the real image is 1.55 ms (the 2026-09-27
  overview column session); the headline says 1.46 ms (ch06's own
  2026-07-25 session, the one the READMEs quote). Both are on screen with
  their meaning; they are 7% apart.
- ch01-ch04 on N-blob rows are estimates (`est: true`, one launch per
  blob); they are dashed and never glow.
- 16,000x is 24,083 / 1.455 = 16,551, floored to two figures.
- The intro's frame budget is 1000 / 30 = 33.3 ms, shown as 33 ms; "81 Mpx"
  is the real image's 81,000,000 pixels from the ch06 runs JSON.
- Intro footage pipeline, per 576 x 1024 frame with 570-1,210 blobs
  (`assets/make_drone_frames.py` prints these and writes
  `assets/drones_labels/timing.json`; medians, host numpy unless noted):
  top-hat filter 4.3 ms, ch06 kernel 1.3 ms (GPU, CUDA events), label
  map to host 0.7 ms, blob prep (dense labels, area and bbox filters,
  centroids) 10 ms, tracking 5.9 ms (p90 17 ms), painting 2 ms. The
  tracker started at 72 ms: sorting the full blobs x tracks distance
  matrix; gating the candidates, vectorising the greedy rounds and
  computing the distances as a float32 matrix product brought it to 6 ms.
  None of the host stages is optimised; on the GPU each would be sub-ms.

## 8. Open items

1. Narration: `narration/tts_kokoro.py` needs about 1.5 GB of free RAM
   for the PyTorch CUDA build even on the CPU. With two Jupyter kernels
   and VS Code open this laptop had 0.5-1.2 GB free and the process was
   OOM-killed. Close the kernels (or anything else large), run
   `uv run narration/tts_kokoro.py`, then `uv run build/assemble.py --check`.
   A lighter option is a CPU-only torch environment (`uv venv .venv-tts`,
   `uv pip install --python .venv-tts/bin/python torch --index-url
   https://download.pytorch.org/whl/cpu kokoro soundfile numpy <en_core_web_sm wheel>`),
   then `CUDA_VISIBLE_DEVICES= .venv-tts/bin/python narration/tts_kokoro.py`.
2. Vertical 9:16 cut: the panes are box-relative; a stacked
   `pane_geometry` is the missing piece.
3. Adding the ch06 column to the README grand table (`overview/build.py`).
4. ElevenLabs voice, background music (`--music`), captions.
