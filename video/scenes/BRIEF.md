# Scene brief

One scene per stage in `scenes/`, `s00_cpu.py` ... `s08_outro.py`, each
with one class (`Cpu`, `OneBlock`, `TwoBlocks`, `NBlocks`, `Conn8`,
`TwoBlobs`, `NBlobs`, `Runs`, `Outro`) that subclasses
`scenes.stage.StageScene` and sets `k`. Everything a stage shows is
declared in `scenes/panes/config.py` (`STAGES`); the narration text and
per-beat timing live in `narration/script.md` and `out/kokoro/timing.json`.

## The picture (all stages)

Three panes for the whole video, landscape 1920x1080 at 30 fps:

- **Left, the benchmark matrix.** Rows are the 17 scenes of the overview
  benchmark, in JSON order (three squares, three disks, two snakes, two
  combs, three two-blob pairs, two noise fields, the two PNGs). Column 0
  is the CPU `@njit` time in ms, always printed. Every stage adds one
  column: the chapter's fastest measured variant on that row. A cell
  glows blue (teal for ch06) when it beats the CPU, brighter for a bigger
  win; grey when slower (red tint below 0.1x); dashed when the number is
  an estimate (one launch per blob on N-blob rows; never glows); hollow
  when the chapter cannot run the row. The current column prints its ms
  in a 4-character form (`1.3s`, `269`, `12.7`, `0.87`); older columns
  keep only their glow.
- **Middle, the blob.** The current stage's replay, big. Every finished
  stage is a thumbnail in the strip at the top, with a two-line tag.
  A stage begins by sweeping the previous blob up into the next slot.
- **Right, the GPU.** Title, the card with 24 SM tiles, and a per-stage
  dynamic group: block chips on the tiles (hue = the block's hue in the
  replay), the block close-up (warps x 32 lanes), shared-memory lines,
  the global-memory bar with its boxes, the ch06 kernel strip, and up to
  three mono caption lines. The CPU stage shows a CPU box instead.

Pane boxes come from `scenes/panes/geometry.py` (widths 5.3 / 4.1 / 3.3
units, margins 0.5 x 0.45, gaps 0.26). Nothing is placed in absolute
frame coordinates.

## Rules (all scenes)

- Subclass `StageScene`, set `k`. Do not call `add_sound`.
- Colours only from `style.py`. Red is "unfilled pixel", teal is "the
  final answer" (ch06), blue is "GPU block / faster than the CPU".
- `Text` only (no LaTeX). Numbers in `DejaVu Sans Mono`.
- **No number is typed into a scene.** Matrix cells and the CPU column
  come from `scenes/panes/data.py` (the committed overview JSON and the
  ch06 overview column); caption numbers come from `data.headline()`
  through `{placeholders}` in `config.py`. `uv run python
  scenes/panes/data.py` prints the matrix; `uv run build/audit_numbers.py`
  recomputes every printed number by plain dict lookups and exits 1 on a
  mismatch.
- The state contract (see `scenes/stage.py`): every mobject that survives
  to the end of a scene was made by a `panes/*` builder with the same
  arguments `build_state(k+1)` uses; text never changes in place (fade
  out, fade in); `ImageMobject`s live in `Group`s; one resampling
  algorithm (bicubic) for replay frames, stills and thumbnails.
- Whole-frame timing: plays use `fr(n)`, holds go through `until()`.
  `-ql` (15 fps) and `--fps 30` then agree to the frame.
- Replay sets are PNG folders under `assets/` (`assets/extract_gifs.py`);
  `Replay` reads frames from disk on demand, so a scene holds one frame.

## Stage timeline

Each stage k >= 1: 0.6 s sweep (previous blob to slot k-1, tag in, old
caption / GPU dynamic / column numbers out), 0.6 s new replay + caption +
GPU dynamic in, 1.2 s matrix column reveal, then the replay runs to its
end (97 frames at 16 fps = 6.1 s; `replay_fps` in `STAGES`), then the
`stage()` hook if any, then `snap()` and a frozen hold to
`beat + 0.4 s`. Stage 0 fades the panes in and reveals the CPU column.
Stages with hooks:

- 06: U-shape provisional labels (24 fps) → cross-fade to the merged
  final → the real image (75 frames at 20 fps); the caption swaps when
  the real image arrives.
- 07: a 36-cell pixel row fades in, its red spans collapse into teal
  runs, then the recoloured right half of `before_after` fades in below
  and the caption swaps to the ms line.
- 08: the ch06 still sweeps up (8 thumbs), three centre lines fade in,
  and the scene fades out at the end (the only fade-out).

## Data on screen (all from JSON; see `data.py --help`)

| item | source |
|---|---|
| matrix rows, CPU and ch01-ch05 columns | `results/overview/benchmark_results/overview_20260724T234423Z.json` |
| ch06 column | `results/overview/benchmark_results/ch06_overview_20260927T013331Z.json` |
| pure Python 24,083 ms, @njit 1,346 ms | overview row `png_blobs` |
| ch05 58.51 ms, ch06 1.46 / 2.96 ms, 2,522 blobs, 13,451,960 red px, 539,207 runs | `results/ch06_gpu_nblob_runs/benchmark_results/runs_20260725T161448Z.json`, scene `input_blobs` |
| 16,000x | 24,083 / 1.455, floored to two figures |
| 24 SMs, 1,536 threads per SM, ring 8,192, grids per chapter | chapter code, quoted in `config.py` |

## Verify loop

1. `STAGE=k LIVE=1 VOICE=kokoro .venv/bin/python -m manim render -s -qh
   --disable_caching --media_dir media/landscape scenes/snapshot.py Snapshot`
   gives one PNG of the middle of stage k: iterate on layout here.
2. `VOICE=kokoro .venv/bin/python -m manim render -ql --disable_caching
   --media_dir media/landscape scenes/s03_n_blocks.py NBlocks`, then
   `uv run build/review.py s03_n_blocks NBlocks --times 0,0.3,0.6,1,1.5,2.6,5,end`
   and look at `media/review/s03_n_blocks/sheet.png`.
3. `uv run build/seam_check.py --intra`: every cut and every hold.
4. `uv run build/assemble.py --render --check` for the full cut.
