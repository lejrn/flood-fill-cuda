# The chain cut, v3: real hardware, less text, more air

Status 2026-10-14: the user's notes on v2, in their words (translated):

1. "Why is the CPU sitting on what looks like the memory queue? It isn't
   clear at all." The CPU was a box at the start of the queue conveyor,
   on the GPU board.
2. "The GPU visualization should be more authentic": like NVIDIA's Ada
   diagrams (GPC / TPC / SM, the SM's four partitions; the user attached
   the wccftech Ada article's figures). "Take the CPU you put on the GPU
   board out into a parallel diagram."
3. "Add some delay between transitions, so all the text is easy to follow."
4. "Maybe also reduce the text on screen."
5. Voice: ElevenLabs "Adam" (premade `pNInz6obpgDQGcFmaJgB`, eleven_v4).

The script (`script_chain.md`) does not change.

## What changed in the shared modules (done)

- `die.py` (new): top views of the two chips, drawn like the vendors'
  block diagrams. GPU: RTX 4060 Laptop (AD107), 3 GPCs x 4 TPCs x 2 SMs,
  L2 32 MB (what the device reports), GigaThread engine, PCIe, four
  memory controllers, NVENC / NVDEC / display. Each SM in NVIDIA's
  layout (four partitions: scheduler, register file, FP32, tensor,
  LD/ST; then L1 / shared memory, texture units, RT core), in four
  states: off, idle, lit (blue), wait (amber). CPU: i9-13900H, 6 P-cores,
  8 E-cores (2 clusters), L3 24 MB on the ring, Xe graphics, memory
  controller.
- `machine.py` (rewritten): a CPU board and a GPU board side by side,
  a PCIe link between them. Parts: `boards`, `cpu_die`, `gpu_die`
  (static); `pcore` (P-core 0 lit while our CPU code runs); `room{i}`
  (SM i lit / idle / wait); `teams` (blocks: blue cubes on the SMs);
  `room_fence` (`__syncthreads` around SM 0); `gate` (grid.sync: a low
  rail around the GPU die at rest, a tall amber wall when blocks wait);
  `queue`, `queue_tokens`, `tags` (the global queue lives in VRAM, the
  four chips in front of the GPU die); `launches` (a small counter pill
  over the PCIe link); `kernels`, `seams` (ch06's six kernels, a column
  in the gap between the boards, the launch lane); `labels` (silkscreen
  "CPU i9-13900H", "GPU RTX 4060 Laptop").
  Gone from the machine (algorithm gadgets, not hardware): the stencil
  badge, the tag printer, the "64 max" rack, the min unit, the CPU's
  queue conveyor. Parts are `Group`s (they hold images); every part has
  a key, and `machine_to` keeps a part whose key did not change.
  Helpers: `sm_closeup(pl, centre, height)` (one SM drawn large, flat,
  with a leader line to SM 0), `sm_point`, `link_path(pl)` (CPU P-core
  -> PCIe -> GigaThread, frame points), `slot_point(pl, j)`,
  `gpu_corner(pl, which)`, `launch_chip(text, pl)`.
- `overlays.py`: the room gauge is gone (the lit SMs show it); the goal
  meter is one line ("budget 33 ms") with no "33" tick label; the
  scoreboard strip has no "CPU vs GPU" header and short row names
  ("small blob", "huge blob", "many blobs"); the inset has one line.
- `state.py`: the boundary table uses the new Spec; captions are a big
  word and at most a short qualifier.
- `base.py`: every beat ends with at least BREATH = 0.6 s of silence; a
  caption stays at least MIN_DWELL = 1.4 s before `say` replaces it;
  `gauge()` is gone; parts compare by key.
- Narration: `narration/pace.py` stretches the pause after every
  sentence to 0.9 s and after a colon to 0.45 s (word times shift with
  it, so cues stay on their words). Kokoro paced: `out/kokoro_paced_chain`
  (voice name `kokoro_paced`). Adam: `out/adam_chain_raw/*.mp3` ->
  `import_audio.py` (word times pinned on the take's pauses, `align.py`)
  -> `pace.py` -> `out/adam_chain` (voice name `adam`).

## Rules for every beat scene

1. Start on `chain_state(k)`, end exactly on `chain_state(k + 1)`, with
   `say_last` the last caption (the snap check must pass, both layouts).
2. Text: at most three `say` calls per beat, `say_last` included. A big
   word is a number or at most three words; a sub is at most four words,
   or None, and only for qualifiers the honesty rules need ("estimate",
   "kernel time", a unit). Cut floor labels, side notes and chips that
   repeat the narration. Never put two text changes within 1.4 s.
3. Honesty stays: "~" on estimates, "kernel time" where it applies, the
   final card's footnote.
4. Hardware is hardware: blocks land on SMs, the queue sits in VRAM,
   launches and seeds cross the PCIe link from the CPU.
5. Amber means waiting, only.

## Per beat

- intro (i0, i1): keep the camera / filter panels and the two benchmark
  cards; drop the panel sub-labels and any line the narration says.
- r00_zoom: unchanged but the inset (one line) and the caption (no sub).
- r01_cpu: at "CPU" both boards slide in, P-core 0 lit (the only lit
  unit); the floor crawl; at "twenty-four" the SMs ripple idle and go
  dark. Captions: "1 pixel per step", "1 core of 14", then the carried
  "24 SMs / streaming multiprocessors".
- r02_one_block: at "So move it onto one SM" a block cube drops onto SM
  0 and `sm_closeup` shows the SM for about 2.5 s (away from the split
  floor and the caption), then fades. The split race with step counters
  ("CPU" / "GPU" under the halves, nothing longer). The race in time on
  the 16 K px blob, short. No scoreboard card here: the strip's first
  pill fills ("small blob: CPU 2.4x").
- r03_giant: the CPU core goes dark as the floor swells into the stress
  test; one block's rings with `__syncthreads` (the fence round SM 0
  flashes amber each ring); "2.46x" (sub "vs the CPU"), then the carried
  "1 of 24 SMs".
- r04_two_blocks: a second cube on SM 1, the queue appears in VRAM, the
  gate rail; "2x again"; carried "22 SMs idle".
- r05_all_blocks: 48 cubes land (two per SM); "18x"; the snake; the
  three times (48 / 2 / 1 blocks) kept compact; the gate rises amber
  twice per ring; "48 lose to 1"; carried "65,792 stops / ~2 us each".
- r06_diagonals: a small 3 x 3 stencil glyph beside the floor (4 arrows,
  then 8), transient; "21 -> 11 rings"; carried "25.8x / vs the CPU,
  kernel time"; the strip's second pill and SIZE done.
- r07_real_frame: COUNT; the floor pulls back to the 81 Mpx frame;
  launch pulses run along `link_path` while the counter pill climbs to
  2,522; the needle swings to ~2,181 ms; carried "~2,181 ms / estimate".
- r08_tags: tag caps on the VRAM tokens; two blobs fill at once ("max,
  not sum"); the pill drops to 1,261; at "every seed is handed in" seed
  dots run from the CPU along the link; carried "64 tags max".
- r09_seeds: the tags leave; seed sparks on the floor; "min" at most as
  one short caption; launches "3"; the queue fills (one short counter
  note by VRAM at most); carried "51.0 ms / still over 33 ms".
- r10_runs: one row's spans ("25x fewer"); queue, gate and launches go;
  the six kernels slide into the gap and light one by one; the final
  scoreboard card with its footnote; carried "2.96 ms".
- r11_reveal: the full card, "the CPU" / "the GPU", back to the drones;
  short subs.
