# Scene brief

One file per beat in `scenes/`, named `s00_hook.py` ... `s06_outro.py`,
each with exactly one class named after the beat in CamelCase
(`Hook`, `Cpu`, `GpuWaves`, `Twist`, `BlobsTogether`, `Runs`, `Outro`).
The narration text and per-beat timing live in `narration/script.md`
and `out/kokoro/timing.json`.

## Rules (all scenes)

- Subclass `style.BeatScene`, set `beat = "<name>"`, end `construct()`
  with `self.finish()`. Never call `add_sound` yourself.
- Look: 3Blue1Brown. Dark background, one idea on screen at a time,
  smooth eased motion, numbers in monospace, generous empty space.
  No decorative boxes, no drop shadows, no emoji.
- Colours only from `style.py`. Red (`RED_PX`) means "unfilled red
  pixel" or "the slow number". Teal (`TEAL`) is the final answer (ch06).
- Text via `style.label` / `style.caption`. No `Tex`/`MathTex` (no LaTeX
  installed). Multiply sign is the unicode `×`.
- Two panels. Build the scene as panel A (picture) and panel B
  (numbers / stopwatch). In landscape arrange them left/right; if
  `is_vertical()` arrange them top/bottom. Scale groups to fit inside
  `self.L["w"] - 2 * margin` by `self.L["h"] - 2 * margin`. Nothing may
  touch the frame edge. Test landscape first.
- Pace: animations follow the narration timing given per beat below;
  total motion must fit inside `beat_seconds(beat)`. `finish()` pads the
  rest and fades out.
- Every number on screen must match the brief exactly. Do not invent
  numbers. Mark estimated numbers with a leading `~`.
- GIF replays: `FrameSequence(name, height)`; call `.start(fps)` then
  wait; `.stop()` before removing. `assets/<name>/meta.json` has frame
  counts. All frame sets have 97 frames unless noted.
- Fonts: default `DejaVu Sans`, numbers `mono=True`.

## Verify loop (mandatory before reporting)

1. `cd video && VOICE=kokoro .venv/bin/python -m manim render -ql scenes/sNN_name.py ClassName`
2. Extract 6 evenly spaced frames with the venv's `ffmpeg`
   (`.venv/bin/ffmpeg -i media/videos/sNN_name/480p15/ClassName.mp4 -vf fps=6/DURATION frames_%02d.png`)
   into `media/review/sNN/` and look at every one of them.
3. Fix anything clipped, overlapping, unreadable, or off-brief. Re-render.
4. Report: final scene length in seconds, the frames you checked, and
   any number you were unsure about.

## Data (from the parent repo, do not change)

| item | value |
|---|---|
| image | 9000 × 9000 = 81,000,000 px, 13,451,960 red px, 2,522 blobs, 539,207 runs |
| pure Python BFS | 24,083 ms |
| Numba `@njit` BFS | 1,346 ms |
| ch01 one block | 2.03× the CPU at 36 Mpx; one SM = 1/24 = 4% |
| ch02 two blocks | 2.10× ch01 |
| ch03 N blocks, 8-conn | 64 Mpx in 100 ms, 20× the CPU |
| on the real image, per-blob launches | ch01 ~1,222 ms, ch02 ~1,916 ms, ch03 ~2,181 ms, ch04 ~1,368 ms (estimates, mark `~`) |
| ch05 seedless | 755,577 blobs in 24.8 ms; real image 58.51 ms |
| ch06 runs | 2.96 ms from RGB, 1.46 ms from packed mask, 0.78 ms labeling only |
| ch06 phases (ms) | pack 1.67, count 0.05, scan 0.02, emit 0.22, merge 0.31, flatten 0.03, paint 0.64 |
| end to end | 24,083 / 1.46 ≈ 16,000× |

## Beat 00 `Hook` (4.81 s)

Panel A: the real image before filling: `assets/ch05_input_blobs/frame_000.png`
as a single `ImageMobject` (red blobs on white), height ≈ 5.5. Slow
scale 0.94 → 1.0 across the beat (Ken Burns).
Panel B: three lines fade in one after another, in sync with the words:
`81,000,000 px` (mono, big), `2,522 red blobs` (RED_PX), then a
`Stopwatch("time", 0)`.
Timing: 0-1.5 s image + first line, 1.5-3 s second line, 3-4.8 s stopwatch.

## Beat 01 `Cpu` (6.14 s)

Panel A: `pixel_grid(12, cell)` with a diamond blob: cells with
Manhattan distance ≤ 4 from the centre are red (41 cells). A small
white square cursor visits the red cells in BFS order
(`manhattan_levels`, ties row-major), one cell per 0.07 s, turning each
visited cell BLUE. Caption under the grid: `one pixel at a time`.
Panel B: two stopwatches stacked: `pure Python` counting up to
`24,083 ms` (shows as `24.1 s`, RED_PX) during the walk, then
`Numba @njit` appearing with `1,346 ms` (INK).
Timing: 0-3.5 s walk + Python counter, 3.5-5 s Numba line, rest hold.

## Beat 02 `GpuWaves` (8.85 s)

Panel A, phase 1 (0-2.5 s): same 12×12 diamond blob, but whole BFS
rings fill together, one ring per 0.25 s (levels 0..4), BLUE with the
newest ring drawn lighter. Caption: `the whole frontier at once`.
Panel A, phase 2 (2.5-8.8 s): replace the grid with
`FrameSequence("ch03_square_conn4", height ≈ 4.5).start(fps=30)`
(8 blocks sharing one wave; hue = block).
Panel B: a 6×4 grid of 24 rounded tiles (GREY, caption `24 SMs`).
At 2.5 s tile 0 turns BLUE and caption becomes `1 block = 1 SM = 4%`.
At 4.5 s tile 1 turns GREEN (`8%`). At 6 s all tiles light in cycling
hues (BLUE, GREEN, PURPLE, TEAL, GOLD) and a mono line appears:
`20× the CPU` with small caption `64 Mpx in 100 ms`.

## Beat 03 `Twist` (6.15 s)

Panel A: the real image again (`assets/ch05_input_blobs/frame_000.png`,
height ≈ 5).
Panel B: two horizontal bars with labels, proportional lengths:
`CPU  @njit   1,346 ms` (INK) and `GPU  ch03   ~2,181 ms` (RED_PX,
longer). The GPU bar grows in at 0.5-2 s so the "lost" is visible.
At 2.5 s a caption: `one blob = one launch`. At 4 s a mono counter
spins from 1 to `2,522 launches` (use a `ValueTracker` + `Integer` or
swap `Text` every few frames; must end exactly on 2,522).

## Beat 04 `BlobsTogether` (10.96 s)

Phase 1 (0-3.6 s) `label rides inside the queue`: Panel A is a queue:
six rounded slots in a row, each holding a mono `x,y` pair with a small
coloured chip attached (BLUE or GREEN) labelled `label`. New entries
slide in from the right carrying their chip. Panel B:
`FrameSequence("ch04_asym_multisource", height ≈ 3.6).start(fps=30)`
(two blobs, one launch, blue and green families).
Phase 2 (3.6-6 s) `colliding waves merge`: replace both panels with one
centred `FrameSequence("ch05_u_prov", height ≈ 5).start(fps=45)`; at
the end cross-fade to the last frame of `ch05_u_final` (one colour).
Caption: `two waves, one label`.
Phase 3 (6-10.9 s): centred `FrameSequence("ch05_input_blobs", height ≈ 6)`
at 30 fps (75 frames = 2.5 s, then hold). Right side, mono lines:
`2,522 blobs · one launch`, then `755,577 blobs · 24.8 ms`, caption
`no seeds given`.

## Beat 05 `Runs` (11.16 s)

Phase 1 (0-2 s): one pixel row of 36 cells (small squares), white/grey
with red spans at columns 2-6, 10-18, 22-24, 27-35. Caption:
`one row of pixels`.
Phase 2 (2-4.5 s): each red span collapses (Transform) into a single
rounded TEAL bar of the same width; caption `a red span = one run`.
Phase 3 (4.5-8.5 s): horizontal bar chart, linear scale, four rows:
`all pixels 81,000,000` (GREY), `red pixels 13,451,960` (RED_PX),
`runs 539,207` (TEAL, visibly tiny), `blobs 2,522` (BLUE). Numbers in
mono at the end of each bar. Then a big mono `25× fewer` beside the
runs row.
Phase 4 (8.5-11.1 s): `Stopwatch("ch05 · pixels", 58.51)` in RED_PX
transforms to `Stopwatch("ch06 · runs", 1.46)` in TEAL. Small caption
under it: `1.46 ms packed mask · 2.96 ms from RGB`. Under everything a
thin strip of seven boxes `pack → count → scan → emit → merge → flatten → paint`,
with the middle five bracketed `0.64 ms`.

## Beat 06 `Outro` (7.52 s)

Phase 1 (0-4.2 s): log-scale horizontal bar chart drawn by hand (no
`BarChart`): axis ticks `1 ms`, `10`, `100`, `1 s`, `10 s`. Rows appear
top to bottom, 0.35 s each:
`pure Python 24,083 ms` (GREY), `@njit 1,346` (GREY),
`ch01 ~1,222` (RED_PX), `ch02 ~1,916`, `ch03 ~2,181`, `ch04 ~1,368`,
`ch05 58.51` (RED_PX), `ch06 RGB 2.96` (GOLD), `ch06 mask 1.46` (TEAL).
Then a mono `16,000×` slides in beside the teal bar.
Phase 2 (4.2-7.5 s): chart shrinks up and dims. Centre text, two lines
fading in one after another: `Not a smarter algorithm.` then
`A better representation.` Bottom-right small: `flood-fill-cuda`
(the repo name only, no URL, no author).
