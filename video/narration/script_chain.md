# Narration script: the chain cut ("you build it")

Status 2026-10-13: approved by the user (full chain, both looks, both
layouts, Kokoro first), voiced with Kokoro `am_adam` and built
(`scenes/chain/`). The three-pane cut (`script.md`) stays as it is; this
is a second, separate cut.

## The idea

The same story as the three-pane cut, told the way the turbojet Short
(youtube.com/shorts/dLLCr_N-WPM) builds an engine. "You" are the engineer.
Every beat is one link of a chain:

1. the problem the last stage left, shown as a picture,
2. the fix, which slides onto the machine as a new part,
3. the next problem that the fix exposes.

Mechanics copied from the reference:

- Second person, and three refrains: "So, being clever, you ..." before a
  fix, "Congratulations" after a win, "But ..." before a problem.
- One "the obvious fix did not work" beat (`c03` + `c04`: more teams, then
  the hidden reason). It is the "wrong blade angle" moment.
- One object that grows: every fix stays on screen as a part.
- The last fix removes the crutch: in `c09` the queue, the room fence
  and the gate come OFF the machine. Nothing waits any more.
- The name comes only at the end: "a connected-component labeller", and
  it is not a flood fill any more.

## The picture

Two things on screen for the whole video, plus a caption:

- **The floor** (top): a small toy picture, about 24 x 24 cells, where
  the work happens. Workers are dots on the cells. The toy changes per
  beat (one blob, a thin snake, a square, two blobs, a U). Every wave,
  ring count and seed shown on it is computed by the real rule (BFS
  levels in 4- or 8-connectivity, the ch05 seed rule, the ch06 runs) on
  that toy at render time, not drawn by hand.
- **The machine** (bottom): a GPU chip seen from above, with 24 rooms
  (the SMs). Parts attach to it beat by beat (table below).
- **The caption**: one word or one number, large, centre screen, timed to
  the word in the narration. A small mono label under it says what the
  number is ("x the CPU, one 36 Mpx blob"). A beat's last caption crosses
  the cut and stays until the next beat's first one.
- **The goal chip** (top corner): "goal 33 ms, one camera frame" and the
  real picture's current time beside it. It turns teal only in `c09`.

In 9:16 the floor sits above the machine; in 16:9 they stand side by
side. Narration never says where a thing is on screen.

### The machine's parts

| beat | part slides in | part leaves |
|---|---|---|
| c01 | the CPU worker and its queue, beside a dark chip | |
| c02 | one room lit: a team of 256, its shelf (ring of 8,192 slots), the room fence (syncthreads) | the CPU worker and its queue |
| c03 | all 24 rooms lit, 2 teams each (48 x 256); one long queue along the chip | the shelf |
| c04 | the gate: the same fence, around the whole chip (grid.sync) | |
| c05 | the stencil badge: 4 arrows become 8 | |
| c06 | a launch counter | |
| c07 | the tag printer on the queue (label in the entry) | |
| c08 | seed sparks and the "min" unit (atomicMin) | the tag printer (labels move to a label map) |
| c09 | the run table and a strip of 6 short kernels | the queue, the room fence, the gate, the launch counter |

What is left at the end is the sum of the fixes that survived: all rooms
busy, 8-connectivity (ch06 links runs with it), labels, the min merge,
and runs.

### Colour code (fixed for the whole video)

| colour | means | `style.py` |
|---|---|---|
| red | an unpainted pixel | `RED_PX` |
| blue | a worker that is working; a painted pixel | `BLUE` |
| amber | a worker that is waiting: at the room fence, the gate, between launches | `GOLD` |
| grey | the parts of the machine nobody uses | `GREY` |
| one hue per blob | the blob's tag or label (c07, c08) | `GREEN`, `PURPLE`, ... |
| teal | the final answer (c09, c10) | `TEAL` |

Amber is the problem colour, like the green flow arrows in the
reference: almost every problem in this story is someone waiting. The
last frames have no amber at all.

## Length

365 words. With Kokoro `am_adam` the cut runs 119.0 s (narration 118.6 s
plus the gaps). Peter Baker reads about 38% slower, so about 2:45, under
the 3-minute Shorts cap. Peter Baker costs about 1,900 ElevenLabs
credits for the whole script (2,300 for the 438-word three-pane script).

## Numbers (checked against the JSON on 2026-10-13)

Paths under `src/flood_fill_cuda/results/`.

| beat | on screen / spoken | value | source |
|---|---|---|---|
| c00, c06 | 81 Mpx, 2,522 blobs | 9000 x 9000, 2,522 | `overview/.../overview_20260724T234423Z.json`, `png_blobs` |
| c01 | 1,346 ms | @njit 1,345.9 ms | same, `png_blobs.njit` |
| c02 | "twice the CPU" | ch01 v2 596.5 vs 1,211.5 ms = 2.03x at 36 Mpx (`sq_6000_center`) | `ch01_.../single_block_shared_20260719T041848Z.json` |
| c02 | 4% | 1 of 24 SMs = 4.17% | ch01 `sm_utilization_pct` |
| c03 | "two teams, twice as fast" | 280.3 vs 589.8 ms = 2.10x at 36 Mpx | `ch02_.../dual_block_20260719T163033Z.json` |
| c03 | "forty-eight teams: eighteen times" | 48 x 256, 137.8 vs 2,584.0 ms = 18.75x at 64 Mpx (`sq_8000_center`) | `ch03_.../multi_block_20260720T012243Z.json` |
| c03 | "forty-eight teams lose to one" | snake 48 x 256: 143.4 ms vs ch01's one block 61.7 ms | same, `serpentine_256` |
| c04 | "twelve thousand wait" | 48 x 256 = 12,288 threads, 0.008% busy on the snake | same, `multi_thread_util_pct` |
| c04 | 65,792 gates, about 2 us each | 2 x 32,896 levels; 143.4 ms / 65,792 = 2.2 us (derived) | same |
| c05 | half the rings | squares only: 8,001 to 4,001 levels (disk x0.71); the toy is a square | `multi_block_20260720T012243Z.json` |
| c05 | "twenty-five times the CPU" | 64 Mpx 8-conn 100.3 ms vs the same session's @njit 2,584 ms = 25.8x | same |
| c06 | ~2,181 ms | ch03 8-conn, one launch per blob: an ESTIMATE (`est: true`) | `overview_...json`, `png_blobs` |
| c07 | "up to 2x" | multisource vs sequential, median 1.05-2.05x | `ch04_.../dual_blob_20260720T140944Z.json` |
| c07 | ~1,368 ms | ch04 multisource, one launch per pair: an ESTIMATE | `overview_...json`, `png_blobs` |
| c07 | 64 max | 6 label bits, mask `0x3F` | `chapters/ch04_.../kernels.py` |
| c08 | 58 ms | ch05 `split_L8` 58.51 ms (corner seeds + a stride-8 lattice) | `ch06_.../runs_20260725T161448Z.json`, `input_blobs` |
| c09 | 13,451,960 px, 539,207 runs | 24.95x fewer (not "150x": that is runs vs all 81 Mpx) | same |
| c09 | 1.46 ms, 6 kernels | packed-mask path 1.455 ms, 40.2x vs 58.51 (RGB in: 7 kernels, 2.96 ms) | same |
| c10 | 24,083 ms, 16,000x | pure Python 24,083.0 / 1.455 = 16,551x (two sessions, as in the three-pane cut) | OV + R61 |

Beats c03 to c05 read one ch03 session (`_012243Z`), so the
8-connectivity gain (18.75x to 25.8x, 1.37x) is not inflated by another
session's CPU time. `uv run python -m scenes.chain.numbers` prints every
value with its file and key, and checks each sentence above; no number
is typed into a scene.

## beat c00_job

```text
Eighty-one million pixels, two and a half thousand red blobs. Your job: colour every blob in one camera frame, thirty-three milliseconds.
```

- On screen: the real picture (`images/input/input_blobs.png`) fades in;
  "81 Mpx", "2,522 blobs"; the goal chip appears. One blob lifts out
  and grows into the floor's toy: "start with one blob".
- Caption: "2,522", then "33 ms".

## beat c01_one_worker

```text
Start simple: one worker paints a pixel and queues its red neighbours. Congratulations, a filled blob, and the real picture takes one point three seconds. But the wave's whole edge, one ring of pixels, could be painted at once.
```

- Problem: one core is serial. The worker walks the toy blob one cell at
  a time; the queue fills and drains beside it. Then a freeze: the
  wave's edge lights up, a ring of cells that do not depend on each
  other. The chip below stays dark: "GPU idle".
- Goal chip: "1,346 ms" (@njit). Pure Python (24,083 ms) waits for `c10`.
- Caption: "1 pixel", then "1,346 ms".

## beat c02_team

```text
So, being clever, you hire a GPU team: two hundred fifty-six workers paint the ring in one step, then wait for the slowest. Twice the CPU. But the chip has twenty-four rooms, and your team uses one.
```

- Fix: the worker becomes a team of 256 dots and moves into room 0. On the
  floor the wave now grows one ring per tick (a diamond). After each
  ring the room's fence flashes amber while the team waits for the
  slowest. The gate in `c04` is the same fence around the whole chip.
- Exposes: the other 23 rooms go grey; "1 of 24 = 4%".
- Caption: "team", "2x" (label: "one 36 Mpx blob"), "4%".
- Cut on purpose: the shelf overflowing and the spill tier behind it.
  The 2.03x is the spill-tier kernel, but the spill tier does not lead
  to the next link.

## beat c03_more_teams

```text
Easy: more teams. Two, twice as fast. Forty-eight: eighteen times the CPU. But on a thin snake, forty-eight teams lose to one.
```

- The obvious fix. Rooms light up: 2, then all 24 with two teams each.
  The shelf goes; one long queue runs along the chip. The caption climbs
  "2x", "18x" (label: "one 64 Mpx blob").
- Then it fails: the floor turns into the snake, "48 teams: 143 ms",
  "1 team: 62 ms".
- Caption: "more", "18x", "slower?".
- Cut on purpose: "more teams get slower". At 256 threads per team, 48
  teams is already the most the chip holds; the slower grids (192 x 64,
  384 x 32) use more, smaller teams, which breaks the picture.

## beat c04_gate

```text
Here's why. After every ring, all the teams must meet at one gate across the chip, and wait for the last. A snake's ring is one pixel wide: one worker paints, twelve thousand wait. Sixty-five thousand times.
```

- The hidden reason, the reference's green-arrow moment. A gate closes
  across the whole chip after each ring. On the snake one dot is blue,
  every other dot turns amber and walks to the gate. A counter runs to
  "65,792 gates", "~2 us each".
- Caption: "the gate", "waiting".

## beat c05_diagonals

```text
You can't make the gate cheaper, so pass it less often. Step diagonally too: the diamond becomes a square, half the rings, half the gates. Twenty-five times the CPU.
```

- Fix: the stencil badge slides on, 4 arrows, then 4 more. On the floor
  (a square) the diamond wave becomes a square and the ring counter
  halves: the toy's own BFS levels. The gate counter halves with it.
- Caption: "8", "half", "25x" (label: "one 64 Mpx blob").

## beat c06_real_picture

```text
Congratulations. Now feed it the real picture. It loses to the CPU: two and a half thousand blobs, each its own launch, its own rings, its own gates. You built a machine for one blob.
```

- The twist (the reference's "cut the power"). The floor zooms out from
  the toy to the real picture. The machine runs one blob, stops, starts
  again; the launch counter races to 2,522; between launches every room
  is amber. The goal chip goes the wrong way: "~2,181 ms" (an estimate,
  marked "~").
- Caption: "2,522 launches".

## beat c07_tags

```text
So every queue entry gets a tag: whose blob is this? Two blobs share one launch and finish in the time of the bigger one. But someone still has to hand you every seed.
```

- Fix: the tag printer clips onto the queue; tokens get a coloured tag.
  On the floor two blobs fill at once, blue and green; two ring
  counters: rings = max(a, b), not a + b.
- The goal chip: "~1,368 ms": half the launches, still no faster than
  the CPU.
- Exposes: a hand places each seed. A tag holder reads "64 max" next to
  "2,522".
- Caption: "tags", "max, not sum", "seeds?".

## beat c08_seeds

```text
So the picture finds its own seeds: waves start at every top-left corner and on a grid between. Where two meet, both keep the smaller tag. Every blob, one launch: fifty-eight milliseconds. Still too slow for one frame: every red pixel still rides the queue.
```

- Fix: seed sparks on the floor's toy, first the corners (the ch05 rule:
  a red pixel with no red neighbour before it in scan order), then a
  lattice. Waves start at all of them. On the U two colours meet; the
  "min" unit flashes and one colour wins. Then the real picture, every
  blob coloured.
- Goal chip: "58 ms", still amber: over the 33 ms budget.
- Exposes: the queue conveyor, with a counter to 13,451,960 tokens.
- Caption: "seeds", "min", "58 ms".
- Why not "gate after gate": the 58.51 ms run has only 19 rings on the
  real picture, so the gate is no longer the cost. The pixel count is.

## beat c09_runs

```text
So you look at one row: red pixels come in spans. Thirteen million pixels, half a million spans. Stop moving pixels. Link spans that touch, keep the smaller tag, paint each span in one stroke. No wave, no gate. One and a half milliseconds.
```

- The loop closes (the reference's turbine). Zoom into one row of the
  floor: red cells collapse into teal bars, "25x fewer". The queue, the
  room fence and the gate slide OFF the machine; a strip of 6 kernels
  slides on (count, scan, emit, merge, flatten, paint) and the runs flow
  through it once.
- Goal chip: "1.46 ms", teal. No amber left anywhere.
- Caption: "runs", "no gate", "1.46 ms".

## beat c10_reveal

```text
Twenty-four seconds became one and a half milliseconds. Congratulations: you've built a connected-component labeller. And it isn't a flood fill any more.
```

- The finished machine, the recoloured real picture (`before_after`),
  then the name: "connected-component labelling". Lines "24,083 ms to
  1.46 ms", "16,000x", "flood-fill-cuda".
- Caption: the name.

## Left out on purpose

- The Triton beat: it is a port, not a fix, so it has no place in the
  chain.
- The drone footage intro: the reference opens on the object at once.
  The goal chip carries the "camera frame" idea instead.
- The spill tier, the four ch02 partitioning designs, the seeding
  lattice's tuning, the RGB vs packed-mask contracts.

## How it is built

- `narration/tts_kokoro.py --script script_chain.md --out kokoro_chain`
  voices this file into `out/kokoro_chain/`, with `words.json` (Kokoro's
  word times). `tts_elevenlabs.py --script script_chain.md --out
  elevenlabs_chain` does the same for Peter Baker, with the API's
  character times.
- Scenes in `scenes/chain/`, one per beat, on `base.ChainScene`; the pure
  `state.chain_state(k)` is the picture between beats, so separate clips
  join at the seams. `LOOK=flat|iso` picks the projection.
- `scenes/chain/toy.py` computes the toy waves, seeds, merges and runs
  with the chapters' rules; `numbers.py` holds every number with its
  source and checks each spoken claim.
- `build/assemble.py --cut chain --look <flat|iso> --layout <layout>
  --voice kokoro [--render] [--preview]` writes
  `out/final_chain_<look>_<layout>_<voice>.mp4`. The three-pane cut
  keeps its file names.
