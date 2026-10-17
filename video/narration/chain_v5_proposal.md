# The chain cut, v5: tiles replace the run-based beat

Status: PROPOSAL, revised after the accuracy, story and build review,
2026-10-17. Nothing is built. The script is `script_chain.md`. Every
number below is pinned in `scenes/chain/numbers.py`: 103 claim checks,
all passing (`cd video && uv run python -m scenes.chain.numbers`).

Base: v4 (`chain_v4_proposal.md`, 16 beats, 217.5 s with Kokoro, paced).
Every v4 rule stays: at most three captions per beat, a caption sub of at
most four words, amber only for waiting, every number from committed JSON
through `numbers.py`, "~" on estimates, "kernel time" where it applies,
no "chapter".

New for v5:
- No "CCL", "runs", "spans" or "union-find" (איחוד-חיפוש), spoken or shown.
- On-screen CPU-vs-GPU factors keep two significant figures (1.4×, 17×,
  540×): the CPU's round medians spread by up to 9%.
- A flashing element flashes at most twice per second (r05b's
  photosensitivity (רגישות לאור) rule, made stricter).

---

## What the user asked, and where it lands

| # | The user asked (short) | Where in v5 | Note |
|---|---|---|---|
| 1 | Make the video again; render with the GPU | the build stage | Manim's Cairo renderer draws on the CPU; the GPU did every new measurement. See Q7. |
| 2 | The run-based labelling beat is not relevant: the focus is the BFS | r10_runs is gone, with every number from it | `numbers.py` drops sessions C6, R61, DR, and v4's 1 Mpx count row (its tiles take the dense-tile path) |
| 3 | The tile BFS and the tile waves replace it | act 3: r10_tiles, r11_rounds (tile BFS), r12_pieces, r13_frame (tile waves) | both are BFS: a frontier (חזית) of tiles, then of pieces |
| 4 | Just continue the video with them | i0 to r09 keep their pictures and most of their text | five phrases change their numbers (Q1) |
| 5 | Ask if needed | the open questions at the end | |

---

## What the review changed

Three critics read the draft. Every "must" and almost every "should" is
in; the table lists what moved.

| finding | change |
|---|---|
| Row 3 rested on ch08's dense-tile path (A1, M1: 1,023 of 1,024 tiles on v4's 1 Mpx noise) | Row 3 is now the 81 Mpx frame, the one the meter shows: GPU 22× at r09, GPU 520× at r13. On the frame no tile takes that path (WV, D8). A check pins it. |
| The frame's ch05 block ran at a 7,001 MHz memory clock (A2) | Disclosed; the clock check now covers every block and names the 3 exceptions. r09 says "about fifty milliseconds" (51.7 here, 47.1 at full clocks in D8). |
| r07-r08 compare July GPU estimates with October's CPU (A3) | Labelled as the one cross-session comparison; both claims are checked against both CPUs. |
| The crop is its own image, not a window of the frame (A4) | It is shown with its own border, and nothing of it carries into the frame. 6 of its 8 blobs are cut by its edge. |
| Orientation of the floors (A5, F1) | The 900 px frame is transposed once at the source, so a lane is a screen row everywhere. Checked. |
| "Inside the budget" holds for kernel time only (A6, S3) | r13: "once the frame is on the card: inside the budget"; caption sub "kernel time, no copies"; the card's footnote gives 53 ms. |
| "best GPU" in r02 is false within V5 (A7) | The bar reads "GPU so far". |
| warp, lane and registers undefined (S1, F7) | "One warp, thirty-two threads"; "each thread holds one row"; "in registers, closer than shared memory". The warp card grows out of r02's SM close-up. |
| The finale's flip is kernel time only (S2) | Spoken: "In kernel time, the GPU now; with the copy, still the CPU." |
| r11 overloaded; strobing gate (S4, S5, F2) | No tag flips, no ghost flash; the gate flashes at most twice per second, then holds while the rounds race. |
| "label" in two senses (S6) | "every tile finds the blob pieces". |
| Act-3 pictures that did not match the data (F3, F4, F8, A11) | The context floor starts with rounds 0-3 blue; border bits travel only into the 4 tiles they really opened; the labelled frame is v4's picture, not the kernel's palette. |
| Missing cues, helpers and checks (F5, F6, F12) | Added below; 7 new array-based checks. |
| "0.05 ms" on the card (F10) | Three decimals under 0.1 ms; every card row is checked to read true. |

Not taken, or taken another way:
- **A2, re-time ch05**: not done; the disclosure option was taken. Q9
  offers a re-time.
- **A3, re-estimate r07-r08 in V5**: not done (it needs new benchmark
  code); both CPUs are checked instead.
- **C1, "fifty-two"**: "about fifty" instead, true at both memory clocks.
- **A8, a footnote for the tuned 73 ms kernel**: kept in this document
  and Q6 only; the card's footnote already holds the copies.
- **S6, the optional "a tile can cut through several blobs"**: left out;
  caption 1 says it.
- **F8, the crop's hues taken from the frame's picture**: left out,
  because the crop is its own image (A4).
- **F14, "one round fills a tile the pixel BFS needs 16 rings for"**:
  not used, because the number is wrong. The seed sits on its tile's
  corner, so the pixel BFS needs 32 levels to fill that tile.
- **F15, a per-round frontier note in r13**: left out; r13 is busy enough.
- **F9, regenerate the drone footage from ch08**: planned for the build
  stage (the frames are local, not committed).
- **F3's count**: round 4 opens 7 tiles in the context (the centre and 6
  others), not 6.

---

## The data behind v5

Three kinds of source, each with one job.

**Every CPU-vs-GPU number: one session**, `v5_20261017T161122Z` (V5).
- GPU: kernel-only CUDA events, 9 interleaved rounds per scene.
- CPU: `@njit` on one core, 5 interleaved rounds, each behind a gate that
  waits for a quiet host.
- Every GPU output matched an image built from the CPU's result.

**V5's clocks, block by block.**
- The SM clock's median is 2,070 MHz in every timed block.
- The huge blob's window has 9 of 177 samples under 2,070: 5 at 1,995
  and 4 at 2,055.
- The memory clock is 8,001 MHz in P0 everywhere but three blocks:
  - the frame's ch05 block: P3 at 7,001 MHz in all 5 samples. That is
    act 2's 51.7 ms. D8 timed the same kernel at 47.1 ms with the memory
    mostly at 8,001.
  - the drones: memory median 7,001 (P0 and P3).
  - the small blob's tiles block: P3 in 5 of 6 samples, memory at 8,001.
- The bench notes said "2,070 MHz in every timed window". That holds for
  the SM medians only. `numbers.py` now checks every block.

**The one cross-session comparison: r07 and r08.** Their GPU times are
July estimates (OV: 6 sampled blobs times the call count). V5 has no
re-estimate. So the meter sets July needles against October's CPU tick.
- r07: ~2,181 ms is slower than both CPUs (1,157 in V5, 1,346 in OV).
- r08: ~1,368 ms is 18% slower than V5's CPU and 2% slower than OV's.
  "close to CPU speed" holds for both. v4's "about CPU speed" was OV's 2%.

**The mechanism and its counters: the tiles' own sessions.**
- Tile BFS: `tiles_20261016T064724Z` (T7).
- Tile waves: `dense_20261016T054922Z` (D8) and
  `waves_20261015T183401Z` (W8). Their deterministic counters agree.

**The pictures: recorded from the kernels**, `video/assets/tiles/` (TL, WV).
- TL: the 64 Mpx square's 250 x 250 tiles, round by round, and one warp's
  close-up of one tile with its 7 x 7 tile context.
- WV: a 384 px crop of the frame, run as its own image, piece by piece;
  and the whole frame's claim rounds.
- 47 and 31 checks against the CPU oracles (אורקלים), all true.

**Why one session.** The CPU is faster today than in July: the 4-way BFS
on the square takes 1,666 ms, against 2,584 in v4's session. The GPU
cells did not move (within 0-4%). Mixing a July CPU in act 1 with an
October CPU in act 3 would mix two CPU speeds in one scoreboard (Q1).

**Why row 3 moved to the frame.** v4's row 3 was 1 Mpx of noise
(random_1000). On it, ch08's default sends 1,023 of its 1,024 tiles down
the dense-tile path (K2b and K4b, built on ch06's run labelling), and 77%
of its blobs never ride a wave. That is the mechanism v5 leaves out. On
the 81 Mpx frame no tile takes that path, in labelling or in paint (WV
and D8 both count 0), and V5 timed the very engine WV recorded. So row 3
is now the frame: the same picture as the meter, 2,522 blobs.

---

## The chain at a glance

19 beats (v4 had 16): v4's first 14, four new, and a new finale.

| k | beat | status | rung |
|---|---|---|---|
| 0 | i0_problem | same | |
| 1 | i1_compare | one word ("two") | the CPU needs 1.2 s, not 1.3 |
| 2 | r00_zoom | same | |
| 3 | r01_cpu | same | |
| 4 | r02_one_block | same text, new numbers | CPU still wins the small blob (0.12 vs 0.17 ms) |
| 5 | r03_giant | same | |
| 6 | r03b_one_sm | changed | "beats the CPU" (1.5×) |
| 7 | r04_two_blocks | same | twice as fast: 2.03x |
| 8 | r05_all_blocks | one word ("Twelve") | 12× |
| 9 | r05b_snake | same | |
| 10 | r06_diagonals | same text, new caption | 17×; row 2 |
| 11 | r07_real_frame | same | |
| 12 | r08_tags | two words ("close to") | ~1,368 ms against 1,157: 18% slower |
| 13 | r09_seeds | two words ("about fifty"), row 3 fills | 51.7 ms, GPU 22×. But every red pixel still rides the queue. |
| 14 | r10_tiles | new | So the queue carries tiles; one warp closes a tile in registers. |
| 15 | r11_rounds | new | One gate per round: 4,001 rings become 127 rounds, 100 ms become 3.10. But one blob, one seed. |
| 16 | r12_pieces | new | So every tile finds its pieces; seeds send waves; at the end, the smallest label. |
| 17 | r13_frame | new | The frame: 6 rounds, every blob counted, 2.24 ms once the frame is on the card. |
| 18 | r14_reveal | new text | Huge or many: the GPU. The small blob: the GPU in kernel time, the CPU with the copy. |

Gone: r10_runs and everything only it used: the "runs" form of the VRAM
close-up, the "1 warp" row read and its "coalesced" caption, the six
kernels and their seams on the link, the meter's packed-mask mark, the
teal frame.

---

## Beats that stay unchanged

i0_problem, r00_zoom, r01_cpu, r03_giant, r04_two_blocks, r05b_snake,
r07_real_frame: same text, same takes, same pictures.

Their numbers that come from GPU-only comparisons keep their July
sessions, each within one session: 48 KB and 125 KB (r03), SM 2, 1.7x
and "no slower" (r04), the snake and its gates (r05b), +15% (r05).

r04's "twice as fast" is now read from V5 (1,077 vs 530 ms, paired
2.03x). r07's needle swings past the new CPU tick (1.2 s) to the same
~2,181 ms estimate: still slower (cross-session, see above).

---

## Changed beats (acts 1 and 2)

Every change comes from the one-session rule (Q1), except r09's (the
memory clock, Q9).

### i1_compare: "one point two seconds"

```text
On frames this small, compiled CPU code keeps up. Scale up: eighty-one megapixels, same thirty-three milliseconds. The CPU needs one point two seconds. Where does the GPU win? Two places: size and count.
```

- CPU on the frame: 1,157 ms (`njit_ms`, V5 `input_blobs.njit.ms`, the
  median; its 5 rounds read 1,048-1,242). v4 said 1.3 s (1,346, July).
- The drone card: CPU 1.46 ms, GPU 0.11 ms, budget 33 ms (`drone_cpu_ms`,
  `drone_gpu_ms`, V5 `drones.summary`). v4's GPU number (0.93 ms) was
  the run-based kernel's; this is the tile waves', 12.7x the CPU.
- "keeps up" still holds: the CPU's worst drone frame is 1.88 ms
  (`drone_cpu_max_ms`).

### r02_one_block: same text, new numbers

- The race: CPU 0.12 ms (`small_cpu_ms`) against "GPU so far", 0.17 ms
  (`small_gpu_ms`, V5 `sq_256.ch05_fused_L8`). The bar was "best GPU";
  V5 itself times the tiles at 0.045 ms on this blob, so "best" is false
  (A7).
- Which cell was fastest before the tiles comes from July's host-clock
  ranking of every ch01-ch05 cell (`ov_small_best_is_pin`; ch05_r128_L8
  was within 3% of it). V5 re-timed only that one, by events.
- Scoreboard row 1: "one small blob: CPU 1.4×" (v4: 2.4×).

### r03b_one_sm: "beats the CPU"

```text
Now one block beats the CPU. But that's one SM of twenty-four.
```

- 1,666 / 1,077 = 1.55 (`big_1block_x`), so "over twice" is false now.
- Caption: "1.5×" / "vs CPU, kernel time" (four words, C2), cued at
  "beats" (was "twice").

### r05_all_blocks: "Twelve"

- 1,666 / 137.7 = 12.1 (`big_48_x`). Caption "12×" / "kernel time".

### r06_diagonals: same text, new caption

- 1,666 / 99.96 = 16.7 (`big_8ways_x`). Caption "17×" / "vs CPU, kernel
  time"; scoreboard row 2: "one huge blob: GPU 17×" (v4: 25.8×).

### r08_tags: "close to CPU speed"

- ~1,368 ms (estimate, `est_ch04_ms`, July) against V5's 1,157 ms: 18%
  slower; against OV's own CPU: 2% slower. "close to CPU speed" holds for
  both (the check: 1.0 < ratio < 1.25 for each).

### r09_seeds: "about fifty", row 3 fills

```text
So the GPU finds its own seeds: every blob in one fill launch, about fifty milliseconds. But every red pixel still rides the queue.
```

- "fifty-one" becomes "about fifty". The meter stops at 51.7 ms
  (`ch05_ms`, V5), which ran at a 7,001 MHz memory clock; D8 timed the
  same kernel at 47.1 ms (`d8_ch05_ms`). Both are "about fifty"; v4's
  51.0 was July's host clock.
- New: about 0.6 s after the needle settles, before "But" (C6),
  scoreboard row 3 fills: "many blobs: GPU 22×" (`count_ch05_x`: CPU
  1,157 ms, GPU 51.7 ms, the same 81 Mpx frame).
- Why now: the GPU already beats the CPU on count; it only misses the
  budget. COUNT stays unwon (no tick) until r13. And row 3 needs a value
  to jump from in act 3.
- The P3 window makes the GPU look slower here, so 22× is conservative;
  at full clocks it would read about 25×.

---

## Act 3: the tiles

Four beats. The first two follow the tile BFS from one tile to the giant
square; the last two take the same tiles to the whole frame.

**One orientation for the whole cut (A5, F1).** The kernels index
[x, y] with x the PNG column, and lane i = x0 + i. The act-3 arrays drop
onto the floors as (row, col), so a lane is a screen row. The 900 px frame
PNGs are in PNG order: v5 transposes them once at the source
(`state.picture()` returns `.transpose(1, 0, 2)`; the intro's frame card
the same). Then every floor from i1 to r14 shows one frame, the zooms in
r07-r13 stay continuous, and the warp card's rows are the tile's lanes.
Checked: the frame's claim-round samples match the picture with
intersection over union 0.78 transposed, 0.10 as is.

Shared devices:
- **The warp card** (new helper `warp_closeup`, a sibling of `sm_closeup`
  and `vram_closeup`): it grows out of one "32 lanes" partition of r02's
  SM close-up (`dz.sm_detail`), so the picture defines the warp. 32 rows
  of 32 bit cells: a red cell is a red pixel, a blue cell is reached. One
  `ImageMobject` whose pixels change per sweep (as `Floor.show_raster`),
  plus a frame, a row bracket and a column of 32 vote dots. No title: the
  caption names it.
- **Tile tokens**: in act 3 a queue token carries a tiny 2 x 2 grid (a
  tile); in r12-r13 the tokens are pieces (a tile token with one cell
  lit). `mc.Spec` gets a `token_kind`; tokens are textures on the token
  top.
- **Three blocks per SM**: both act-3 kernels launch 72 cooperative
  blocks (tiles 72 x 128, waves 72 x 256), the most that fit at once: 3
  per SM. The third cube lands visibly at r10's first words (C7).
- **The gate** flashes amber once per round, at most twice per second.
  When rounds race, it holds up at steady amber (r05b's pattern) and
  lowers when they end.
- **No launch counter** in act 3: the count depends on switches and is
  not part of the story.

### r10_tiles (k = 14), new

Rung: So (fixes r09's "every red pixel still rides the queue") the queue
carries tiles, and one warp closes a tile in registers (אוגרים).

```text
So the queue carries tiles, not pixels: squares of thirty-two by thirty-two. One warp, thirty-two threads, takes a tile. Each thread holds one row as a thirty-two bit word, one bit per pixel, in registers, closer than shared memory. Fill along every row at once, step diagonally, fill along the columns, and repeat until nothing changes.
```

Picture:
- "So the queue carries tiles": the labelled frame turns back to red.
  One `machine_to`: the twelve red tokens become tile tokens, the launch
  pill fades out, and each SM's third cube lands (F5).
- "thirty-two by thirty-two": the floor zooms into the frame's largest
  blob (76,090 px). A 7 x 7 grid of tiles lands on it (TL `cu_ctx_red`,
  224 x 224 px). The 15 tiles the BFS opened in rounds 0-3 are already
  blue (`cu_ctx_fill`), the blob's seed tile among them. The 7 tiles of
  round 4 are red; the centre is outlined. No round numbers: on this blob
  the rounds move with timing (F3).
- "One warp, thirty-two threads, takes a tile": SM 0's close-up opens as
  in r02; one partition's "32 lanes" lights, and the warp card unfolds
  out of it with the tile's real mask (`cu_mask`: 543 of 1,024 red) (F7).
- "Each thread holds one row": the rows light top to bottom; one row's
  32 cells get a bracket (one word).
- "in registers, closer than shared memory": the card's frame pulses
  (the new-part convention).
- "Fill along every row": the recorded sweeps (`cu_sweeps`) play, one
  every ~0.4 s. The 17 entry bits sit on the left edge (bit 0, rows
  13-31 but 19 and 28), where the round-3 tile beside it handed them
  over. Then row 86, diagonal 109, column 181, row 387, diagonal 481,
  column 541, row 543. A small glyph under the card shows the direction
  (along, diagonal, down); no words.
- "until nothing changes": trip 4 changes nothing. On "changes" (F17),
  with caption 3: the 32 threads vote (the column of dots dims), the card
  folds back into SM 0, and the 7 round-4 tiles on the floor turn blue
  together: the frontier closes in parallel.

Captions (3):
1. "1 tile, 1 warp" / "32 x 32 pixels", at "warp".
2. "1 bit per pixel" / "a row per thread", at "pixel".
3. Carried: "4 trips" / "then nothing changes", at "changes".

Numbers:
- 32 x 32: `tile_px` (TL `square.width / tiles[0]`), and the crop's 384 /
  12 (WV). One thread per row: `WARP_LANES` = 32, a CUDA constant.
- 3 per SM: `big_tiles_blocks` = `small_tiles_blocks` =
  `frame_waves_blocks` = 72 = 3 x 24.
- 543 of 1,024 red, 17 entry bits: `cu_red_px`, `cu_entry_px` (TL
  `closeup`).
- The context: `floors()` `ctx_seed_tile_round` 0, `ctx_tiles_before`
  15, `ctx_tiles_in_centre_round` 7, `ctx_entry_side_round` 3.
- The sweeps: `cu_reached`: 17, 86, 109, 181, 387, 481, 541, 543, then
  543 five times. 4 trips: `cu_trips`; trip 3 is the last that adds
  pixels (`cu_last_growth_trip`).
- The blob: `cu_blob_px` 76,090, `cu_blob_rank` 0 (the frame's largest).

State at the cut (k = 15): floor `tile_ctx` (rounds 0-4 blue, the rest
red, grid lines faint, no outline); machine: all SMs lit, three cubes
each, queue full (12 tile tokens), gate at rest, no launch pill; the
meter, rows and tag as at k = 14; caption "4 trips".

Length: 56 words, about 20.7 s of speech, 22.1 s with its tail.

### r11_rounds (k = 15), new

Rung: one gate per round: 4,001 rings become 127 rounds, 100 ms become
3.10. But that's one blob, from one seed.

```text
Only its border bits go out, into its neighbours, and those join the next round's queue. One gate per round, not two per ring. On the giant square, four thousand rings become a hundred twenty-seven rounds: from a hundred milliseconds to three. But that's one blob, from one seed.
```

Picture:
- "Only its border bits go out": the closed tile's reached border cells
  light: 31 on the top row (lane 0), 26 on the left column (bit 0), 18
  on the right (bit 31), 2 on the bottom row (lane 31).
- "into its neighbours": bits travel only into the 4 tiles they really
  open: above (32 px), right (18), above-left (1), above-right (1). The
  tiles to the left and lower left are already blue (round 3) and the
  one below closed with the centre (round 4): no arrows go there. No
  arrow goes into the lower-right tile either: it touches no reached
  pixel, and another tile opens it in round 6 (F4, A11).
- "join the next round's queue": the 12 tile tokens fade (consumed) and
  4 drop in (F5).
- "One gate per round": the gate flashes once and the 4 tiles turn blue
  (round 5). During "not two per ring" the context plays on: rounds 6-8,
  one flash each, 0.5 s apart (`cu_ctx_first_open`, no numbers shown).
- "On the giant square": the floor swells into the 64 Mpx square, at
  act 1's footprint (F14). The inset reads "stress test"; the tag stays
  COUNT (S4, F17). The square is drawn from the real rounds: 250 x 250
  tiles, each shaded by the round it first opened (TL `sq_first_open`),
  quantized (לכמת) into about 12 bands, light to dark.
- The rounds play in about 4 s (F2): 2 slow rounds, one flash each; then
  the gate holds up at steady amber while the rest race and a small
  counter beside the floor runs to 127; the gate lowers as the counter
  lands. The queue refills to 12 tile tokens.
- "from a hundred milliseconds to three": the row 2 pill turns from
  "GPU 17×" to "GPU 540×". One change; no caption lands with it.
- "But that's one blob": the floor pulls back to the 81 Mpx frame: the
  largest blob blue, the other 2,521 red; the seed dot pulses. The inset
  returns to "drone frame".

Captions (3, S4):
1. "1 gate" / "per round", at "round," (word 20).
2. "127 rounds" / "was 4,001 rings", at "twenty-seven".
3. Carried: "1 blob" / "from 1 seed", at "seed".

The 3.10 ms is on the r14 card; here the pill carries the jump.

Numbers:
- The border bits and their targets: `floors()` `cu_border_bits` (31,
  26, 18, 2), `cu_targets_new` 4 (32, 18, 1, 1 px), all opened in round
  5; the other tiles they touch were open by round 4.
- One gate per round: `big_tiles_one_sync` (V5) and `t7_one_sync` (T7).
  The kernel calls `grid.sync()` once per round (ch07 `kernels.py`, the
  one_sync branch).
- 4,001 rings: `big_levels8_v5` = `big_rings8` (CH03) = `tl_levels`
  (the CPU oracle) = `t7_levels8`.
- 127 rounds: `big_tiles_rounds` (V5) = `tl_rounds` (the export and its
  5 twin runs) = `t7_rounds`.
- The floor: the seed tile, then 125 rings of tiles (rounds 0-125, a
  tile's round is its ring, `tl_chebyshev`); round 126 only revisits and
  opens none.
- 100 ms: `big_8ways_ms` (V5 `ch03_conn8.ms`, 99.96: r06's kernel). 3.10
  ms: `big_tiles_ms`. Paired per round: 32.2x (`big_8_over_tiles_paired`).
- 540×: `big_tiles_x` = 1,665.9 / 3.10 = 537.
- One seed: `big_seed` (V5 `sq_8000_center.seed`).

State at the cut (k = 16): floor `frame_one_blob` (the 900 px frame,
red, the largest blob blue); machine as k = 15; rows (small CPU 1.4×,
huge GPU 540×, many GPU 22×); tag COUNT; inset as k = 14; needle 51.7 ms;
caption "1 blob".

Length: 49 words, about 18.5 s, 20.1 s with its tail.

### r12_pieces (k = 16), new

Rung: So every tile finds its pieces, seeds send waves piece to piece,
and at the end each blob keeps its smallest label.

```text
So first, every tile finds the blob pieces inside it, in registers, all at once. Some pieces are seeds: each seed sends a wave from piece to piece, one round at a time. Where two waves meet, the meeting is noted, and at the end each blob keeps its smallest label.
```

The picture is WV's crop: 384 x 384 px at x 4352, y 5632, run as its own
image (A4). It shares the frame's tile grid and seed lattice (סריג):
its 36 seeds are the frame's, tile for tile. But it is not a window of
the frame's run:
- 6 of its 8 blobs are cut by its edge, so they are parts of larger
  blobs.
- 7 of its 98 tiles end on another round than in the frame (its last
  claim comes in round 5; the frame's tiles there end by round 4).
- Its labels are its own.

So the picture shows it as a small image of its own, and nothing of it
carries back into the frame.

Picture:
- "So first, every tile": the floor zooms from the frame into the crop's
  window. When it lands, a thin border frames the crop and the rest of
  the frame fades out: 12 x 12 tiles, 8 blobs, red.
- "finds the blob pieces inside it": the SM rooms pulse once. Finding
  the pieces is its own launch, one warp per tile, so the cubes (the
  waves' blocks) stay still (F16). Each piece gets an outline and a dot
  at its first pixel (its anchor): 135 pieces, at most 4 in a tile
  (`crop_piece`, `crop_comp`).
- "Some pieces are seeds": 36 sparks (as r09's); 11 of them are seeds
  only because their tile is on the lattice (`crop_tile_lattice`, faint
  diagonal hatching).
- "each seed sends a wave": the queue's tokens become piece tokens (F5).
  Six rounds, one gate flash each, 0.5 s apart. A claimed piece takes
  its wave's hue in its round (`crop_comp_round`, `crop_comp_wave`): 50,
  38, 7, 3, 1, then none. Each round's meetings appear as faint white
  ticks on the tile borders (`crop_meet_tester`, `crop_meet_other`,
  `crop_meet_round`, 116 in all).
- "Where two waves meet": the ticks brighten together, once.
- "and at the end": the dots, hatching and ticks fade (F5). At "smallest
  label", every piece of a blob turns to one hue at once
  (`crop_comp_label`).
- Hues: the waves need distinct hues where they meet (a colouring of the
  meeting graph from `BLOB_HUES`); the 8 final labels too, at least where
  blobs share tiles.

Captions (3):
1. "pieces" / "blobs cut by tiles", at "pieces".
2. "36 seeds" / "8 blobs", at "seeds".
3. Carried: "min" / "smallest label wins", at "label".

Numbers:
- 12 x 12 tiles, 8 blobs, 135 pieces, at most 4 per tile, none interior:
  `crop_tiles_side`, `crop_blobs`, `crop_pieces`,
  `crop_max_pieces_per_tile`, `crop_interior_pieces` (WV `crop`).
- 36 seeds, 11 lattice-only: `crop_seeds`, `crop_seeds_lattice_only`.
- 6 rounds, claims 50/38/7/3/1/0, each piece claimed once in the round
  equal to its distance from the nearest seed: `crop_rounds`,
  `crop_claimed_per_round`, WV checks.
- 116 meetings, no repeats: `crop_meeting_edges`,
  `crop_collisions_repeated`; 28 links = 36 seeds - 8 blobs: `crop_links`.
- "in registers": no tile of the crop or the frame takes the dense-tile
  path, in labelling or paint (`crop_dense_tiles`, `full_ccl_tiles`,
  `full_paint_ccl_tiles`, `d8_ccl_tiles`, `d8_paint_ccl_tiles`: all 0).
- The crop as its own image: `floors()` `crop_blobs_cut_by_edge` 6,
  `crop_tiles_other_last_round` 7, `crop_seeds_are_frame_seeds`.

State at the cut (k = 17): floor `crop_labels` (the crop inside its
border, one hue per blob, faint tile grid; no dots, hatching or ticks);
machine as k = 16 with piece tokens; rows and needle as k = 16; caption
"min".

Length: 51 words, about 18.3 s, 19.5 s with its tail.

### r13_frame (k = 17), new

Rung: the frame: over 11,000 seeds, 6 rounds, every blob counted, 2.24
ms once the frame is on the card: inside the budget.

```text
On the big frame: over eleven thousand seeds, six rounds, every blob counted. Two point two milliseconds, once the frame is on the card: inside the budget.
```

Picture:
- "On the big frame": the crop's hues blend back to red and its border
  fades (F8); then the floor pulls back to the whole red frame.
- "over eleven thousand seeds": seeds spark over the frame, tile by tile
  (WV `full_tile_seeds`, 282 x 282).
- "six rounds": six gate flashes, 0.5 s apart; the claims spread out
  from the seeds (WV `full_round_900`, the claim round at 900 x 900
  samples). At 900 px only rounds 0-4 show: rounds 5 and 6 claim 6 and
  0 pieces, so the last two flashes change nothing visible. The 94
  interior samples are drawn labelled from the start (F15).
- "every blob counted": the frame turns into the labelled picture (v4's
  `frame_labels`: one pastel hue per blob, the same blobs; not the
  palette ch08 paints) (F8, C5).
- "Two point two milliseconds": with caption 2, the needle slides from
  51.7 ms past the 33 ms line to 2.24 ms and turns teal.
- "inside the budget": about 0.6 s later, the tag turns "COUNT ✓" and
  row 3's pill turns from "GPU 22×" to "GPU 520×".

Captions (2; a third would land under MIN_DWELL after the first):
1. "11,609 seeds" / "6 rounds, 2,522 blobs", at "rounds".
2. Carried: "2.24 ms" / "kernel time, no copies", at "milliseconds".

Numbers:
- 11,609 seeds: `full_seeds` (WV) = `d8_seeds` (D8); W8 agrees.
- 6 rounds: `frame_rounds` (V5) = `full_rounds` = `d8_rounds`; frontier
  11,586 / 14,744 / 7,834 / 2,306 / 289 / 6 pieces (`full_frontier` =
  `d8_frontier`). V5 timed the engine WV recorded (`wv_engine_is_v5`).
- 2,522 blobs, labels equal the CPU's: `n_blobs`, `frame_labels_ok` (V5),
  `full_blobs`, `d8_n_blobs`.
- 2.24 ms: `frame_ms` (V5, kernel only). With the copies 53.1 ms
  (`frame_e2e_ms`), over 33: hence "once the frame is on the card".
- The needle's drop: 51.7 / 2.24 = 23x in V5; 21x against D8's
  full-clock ch05 (47.1 ms).
- 520×: `count_x` = 1,157 / 2.24 = 517.

State at the cut (k = 18): floor `frame_labels`; needle 2.24 ms, teal;
rows (small CPU 1.4×, huge GPU 540×, many GPU 520×); tag COUNT ✓; machine
as k = 17; caption "2.24 ms / kernel time, no copies".

Length: 27 words, about 10.3 s, 12.5 s with its tail (HOLD 1.8 s).

### r14_reveal (k = 18), new text

Rung: the opening question, answered with the scoreboard. Row 1 flips in
kernel time, and the voice says what the copy does.

```text
One huge blob, or many at once: the GPU. And one small blob? In kernel time, the GPU now; with the copy, still the CPU. Every drone, every frame.
```

Picture (v4's r11_reveal, re-cued):
- "One huge blob": the full card opens; rows 2 and 3 framed in teal;
  the machine's blocks land column by column (as v4).
- "the GPU": caption 1.
- "And one small blob?": row 1's frame.
- "In kernel time, the GPU now": row 1's winner cell cross-fades from
  "CPU 1.4×" to "GPU 2.7×" (0.12 vs 0.045 ms); caption 2.
- "with the copy, still the CPU": the footnote's copy line lights; P-core
  0 lights and goes dark again.
- "Every drone": the card closes onto the live drones; caption 3. The
  title card fades in for the last seconds (Q4).

The card (`score_rows(beat)`, every number through `card_ms` / `card_x`):

| row | sub | CPU | GPU | winner | tag |
|---|---|---|---|---|---|
| one small blob | 16 K px test blob | 0.12 ms | 0.045 ms | GPU 2.7× | one session, kernel time |
| one huge blob | 64 Mpx test blob | 1,666 ms | 3.10 ms | GPU 540× | one session, kernel time |
| many blobs | 81 Mpx frame, 2,522 blobs | 1,157 ms | 2.24 ms | GPU 520× | one session, kernel time |

Footnote: "kernel time. With the copies: small blob GPU 0.60 ms (CPU
0.12), frame GPU 53 ms (CPU 1,157)."

Captions (3):
1. "the GPU" / "huge or many blobs", at "GPU".
2. "small blob too" / "in kernel time", at "now".
3. "every drone frame" / "kernel: 0.13 ms max", at "Every" (A13).

Numbers:
- Row 1: CPU 0.121 ms (`small_cpu_ms`), GPU 0.045 ms (`small_tiles_ms`,
  V5 `sq_256.ch07.ms`, 4 rounds): 2.7× (`small_tiles_x`).
- With the copy: 0.60 ms (`small_tiles_e2e_ms`): the CPU wins by 5.0x
  (`small_e2e_cpu_x`). Rows 2 and 3 stay the GPU's with the copies too
  (44.7 and 53.1 ms against 1,666 and 1,157).
- Drones: GPU median 0.115 ms, worst frame 0.130 ms (`drone_gpu_ms`,
  `drone_gpu_max_ms`, each frame's median of 9 runs); the worst single
  run 0.342 ms; with the copies the worst frame is 1.89 ms
  (`drone_e2e_max_ms`): still far under 33.

Length: 29 words, about 12.5 s, 16.4 s with its tail (HOLD 3.5 s).

---

## The scoreboard, before and after

All rows: one session (V5), GPU kernel time, CPU `@njit` on one core,
v4's CPU baseline per scene. Every card row is checked to read true: its
two shown times give its shown factor (`card_row_ok`).

| row | r02 | r06 | r09 | r11_rounds | r13_frame | r14_reveal |
|---|---|---|---|---|---|---|
| one small blob (16 K px) | CPU 1.4× (0.12 vs 0.17 ms) | | | | | **GPU 2.7×** (0.12 vs 0.045 ms) |
| one huge blob (64 Mpx) | ? | GPU 17× (1,666 vs 99.9 ms) | | **GPU 540×** (1,666 vs 3.10 ms) | | |
| many blobs (81 Mpx frame, 2,522 blobs) | ? | ? | GPU 22× (1,157 vs 51.7 ms) | | **GPU 520×** (1,157 vs 2.24 ms) | |

v4 for comparison: CPU 2.4× (July, host clock), GPU 25.8× (July), GPU 35×
(1 Mpx noise, two sessions, the run-based kernel).

How far each shown factor moves with the CPU's 5 round medians (A15):

| shown | CPU round medians | factor range |
|---|---|---|
| row 1: CPU 1.4×, then GPU 2.7× | 0.119-0.127 ms | 1.35-1.43, then 2.65-2.82 |
| r03b 1.5×, r05 12×, row 2: 17×, then 540× | 1,647-1,794 ms | 1.53-1.67, 12.0-13.0, 16.5-17.9, 531-579 |
| row 3: 22×, then 520× | 1,048-1,242 ms | 20.3-24.0, 469-555 |

The like-for-like CPU twins (Q2): small blob 8-way BFS 0.187 ms (GPU
4.1×); huge blob 8-way BFS 1,472 ms (GPU 470×); the frame, labels only,
282 ms (GPU 130×); drones, labels only, 0.64 ms (GPU 5.6×).

---

## Boundary table (`state.py`)

`BEATS` becomes:

```text
i0_problem, i1_compare, r00_zoom, r01_cpu, r02_one_block, r03_giant, r03b_one_sm,
r04_two_blocks, r05_all_blocks, r05b_snake, r06_diagonals, r07_real_frame, r08_tags,
r09_seeds, r10_tiles, r11_rounds, r12_pieces, r13_frame, r14_reveal
```

| k | the state when this beat starts | from v4 |
|---|---|---|
| 2-13 | as v4, with V5's numbers and the transposed picture | `boundary(k)` unchanged in code |
| 14 | the labelled frame, 51.7 ms, rows (CPU 1.4×, GPU 17×, **GPU 22×**), tag COUNT, cap "51.7 ms" | v4's k = 14 + row 3 |
| 15 | `tile_ctx` (rounds 0-4 blue), three cubes per SM, 12 tile tokens, no launch pill, cap "4 trips" | new |
| 16 | `frame_one_blob`, row 2 GPU 540×, tag COUNT, cap "1 blob" | new |
| 17 | `crop_labels` (bordered, hues only), piece tokens, cap "min" | new |
| 18 | `frame_labels`, needle 2.24 ms teal, row 3 GPU 520×, COUNT ✓, cap "2.24 ms" / "kernel time, no copies" | replaces v4's k = 15 |

---

## New fields, rasters and helpers

- `state.picture()` returns the 900 px frame transposed (F1); so do
  `picture_blobs()`, `filled_picture()`, `teal_picture()` and r07's
  zoom target, which build on it.
- `mc.Spec`: `token_kind` ("pixel", "tile", "piece"); `teams` up to 3.
  `mc._boxes` and `place_blocks` draw 3 cubes (about 0.10 wide each).
- A gridless floor raster (F6): an RGB array upscaled nearest to the
  floor's pixels, alpha 255, no keying, with an optional faint tile grid
  every N cells. `fl.toy_raster` draws a line on every cell edge (every
  3.5-3.9 px at 250-282 cells), and `fl.picture_raster` keys light
  colours as white (the square's light end would wash out).
- Raster primitives: piece outlines (from `crop_comp`), anchor dots,
  lattice hatching, meeting ticks (from the pixel pairs of
  `crop_meet_tester` and `crop_meet_other`).
- Token textures: a 2 x 2 grid (tile) or one lit cell (piece) on the
  token top.
- `warp_closeup`: one `ImageMobject` swapped per sweep, a frame, a row
  bracket, 32 vote dots.
- Floors: `tile_ctx`, `square_tiles` (from `sq_first_open`, also played
  round by round in r11), `frame_one_blob`, `crop_red`, `crop_labels`,
  `frame_rounds` (from `full_round_900`).
- `numbers.act3_arrays()` returns every TL and WV array by name;
  `numbers.floors()` the facts computed from them.
- `score_rows(beat)`: the small row's GPU cell is `small_gpu_ms` until
  r14 and `small_tiles_ms` after; the huge row's `big_8ways_ms` until
  r11, then `big_tiles_ms`; the many row's `count_ch05_ms` from r09, then
  `count_gpu_ms` from r13. Sub "81 Mpx frame, 2,522 blobs"; every tag
  "one session, kernel time" (A14).
- `overlays.fmt_ms` gets three decimals under 0.1 ms and `fmt_x` two
  significant figures: both as `numbers.card_ms` / `numbers.card_x`,
  which the checks use. The meter's packed-mask mark goes (it reads the
  removed `ch06_mask_ms`).

About 400-600 new lines in all, every piece on an existing pattern.

Other files that list beats:
- `base.py` HOLD: drop r10_runs and r11_reveal; add r10_tiles 1.0,
  r11_rounds 1.2, r13_frame 1.8, r14_reveal 3.5.
- `build/assemble.py` CHAIN: drop r10_runs and r11_reveal; add the five.
- `build/seam_check.py` FADES_OUT: "chain/r11_reveal" becomes
  "chain/r14_reveal" (F13).
- `HANDOFF.md` section 11.
- `r10_runs.py` and `r11_reveal.py` leave the chain. `r14_reveal.py`
  starts as a copy of `r11_reveal.py`. Deleting the two old files is the
  user's call.

---

## `numbers.py`: what changed

- Sessions: + V5, T7, D8, W8, TL, WV. - C6, R61, DR (every number of the
  run-based kernel, the old drone session).
- Moved to V5: every CPU time and every act-1 GPU time on the square
  (`big_*_ms`), the frame (`njit_ms`, `ch05_ms`, `red_px`, `n_blobs`), the
  small blob, the drones.
- Row 3 is the frame: `count_*` keep their names (so `score_rows()`
  keeps working) and point at V5's `input_blobs`. V5's random_1000 is
  not read.
- Kept from July: act 2's two estimates (now checked against V5's CPU and
  OV's own, `ov_njit_ms`) and act 1's GPU-only facts (levels, placement,
  sweep, snake, shared ring, owner maps).
- Removed keys: `ch06_rgb_ms`, `ch06_mask_ms`, `n_runs`, `runs_x`,
  `big_total_ms`, `small_ch01_ms`, `count_gpu_e2e_ms`, `count_ok`, and the
  `frame_x` family (now `count_x`).
- New: the clock of every timed block with its named exceptions;
  `d8_ch05_ms`; the dense-tile counters of the frame and the crop;
  `wv_engine_is_v5`; `floors()` (the context floor, the border bits and
  their targets, the crop against the frame, the frame's 900 px rounds,
  the orientation); `card_ms`, `card_x`, `card_row_ok`.
- 103 checks (v4 had 59).
- Today `state.py` builds `boundary(0)` to `boundary(14)` with the new
  numbers; `boundary(15)` (v4's r10) fails on the removed `ch06_rgb_ms`,
  as it should until the build stage rewrites it.

---

## Length

Model, calibrated on the real v4 paced takes:
- Speech: words / 3.11 + 0.77 s per inner sentence end + 0.41 s per
  colon or semicolon. It reads v4's 16 takes at +1.8% in all (beat by
  beat within about 12%).
- An unchanged beat uses its real v4 take; a beat with a word or two
  changed uses its take plus the word difference.
- The slot adds the tail: max(BREATH 0.75 s, HOLD) + 0.4 s.

The task's simpler formula (2.6 words per second + 1.125 s per sentence
end + 0.56 s per colon) puts v4 at 270 s against its real 218 s, so it
reads about 23% long; its column is given for reference.

| beat | words | status | speech, s | slot, s | task formula, s |
|---|---|---|---|---|---|
| i0_problem | 22 | same (take) | 8.8 | 9.9 | 11.9 |
| i1_compare | 33 | one word | 16.4 | 17.6 | 19.5 |
| r00_zoom | 9 | same | 3.8 | 5.7 | 6.5 |
| r01_cpu | 24 | same | 8.6 | 9.7 | 12.1 |
| r02_one_block | 35 | same | 12.8 | 14.4 | 18.4 |
| r03_giant | 34 | same | 12.6 | 13.7 | 16.5 |
| r03b_one_sm | 12 | changed | 4.4 | 5.6 | 6.9 |
| r04_two_blocks | 62 | same | 23.8 | 24.9 | 32.3 |
| r05_all_blocks | 60 | one word | 22.2 | 23.3 | 30.4 |
| r05b_snake | 23 | same | 8.2 | 9.4 | 12.2 |
| r06_diagonals | 16 | same | 5.9 | 7.5 | 9.4 |
| r07_real_frame | 25 | same | 10.2 | 11.3 | 14.1 |
| r08_tags | 52 | two words | 18.4 | 19.5 | 26.2 |
| r09_seeds | 24 | two words | 8.9 | 10.0 | 12.1 |
| r10_tiles | 56 | new | 20.7 | 22.1 | 26.9 |
| r11_rounds | 49 | new | 18.5 | 20.1 | 24.4 |
| r12_pieces | 51 | new | 18.3 | 19.5 | 23.6 |
| r13_frame | 27 | new | 10.3 | 12.5 | 14.8 |
| r14_reveal | 29 | new | 12.5 | 16.4 | 19.5 |
| **whole cut** | 643 (v4 496) | | | **273 s (4:33)** | 338 s |

Against v4 (218 s): act 3 adds 74.2 s (r10-r13), r10_runs (23.0 s) goes,
r14 is 3.7 s longer than r11_reveal, and the act 1-2 edits net about 0 s.
So +55 s, under the five-minute cap with 27 s to spare.

Caption spacing: every new caption is cued at least 1.75 s (MIN_DWELL)
after the one before, by word count at 3.11 words per second; r13 keeps
two captions for this reason, and the scenes enforce it in code
(`max(cue, previous + MIN_DWELL)`). Re-check all of them from the new
`words.json`.

---

## What v5 does not claim

- **Kernel time only.** Every GPU time is kernel only, as in v4. With the
  copies, the frame takes 53.1 ms (`frame_e2e_ms`, over 33), the square
  44.7 ms, the small blob 0.60 ms (the CPU's 0.12 wins). The drones fit
  even with the copies (worst 1.89 ms). r13 and r14 say so aloud.
- **v4's CPU baselines.** On many blobs the baseline labels and then
  fills each blob; labels alone are 4.1x faster on the frame. The twins
  are in this document (Q2).
- **"From a hundred milliseconds to three"** compares r06's own 8-way
  kernel: the thread layout with its counters on (`bare=False`, 104
  registers), against the bare tiles. The square's best tuned pixel BFS
  (warp layout, bare, 72 x 256) takes 73.2 ms, 23.3x the tiles, paired
  (`big_tuned_ms`, `big_tuned_over_tiles_paired`). The tiles' own README
  quotes that one (Q6).
- **Where the 3.10 ms go.** The rounds take 1.11 ms; packing the image to
  bits (1.15 ms) and painting it back (0.81 ms) take the rest
  (`big_tiles_*_ms`). Not spoken (Q8).
- **Rounds on irregular blobs move with timing.** On the square they do
  not (127 in every run). The close-up's blob took 9 or 10 rounds in five
  runs, so the video gives that blob no round count. A tile's fill is
  drawn at its first visit; a tile visited twice may have filled part of
  it later.
- **Meetings are joined after the waves**, not at the moment two waves
  touch. The narration says so: "the meeting is noted, and at the end".
- **The collision counter** of the frame holds 142 repeats in 31,198
  (a kernel quirk, labels unaffected). The video shows the crop's 116
  meetings, which have none.
- **The crop is its own image** (6 of 8 blobs cut by its edge, 7 tiles
  on other rounds than the frame's); the picture says so with a border.
- **Act 2's ch05 time** ran at a 7,001 MHz memory clock (V5: 51.7 ms;
  D8 at full clocks: 47.1). Spoken "about fifty".
- **Seeds are not one per blob**: 11,609 seeds for 2,522 blobs (3,673
  only from the lattice).
- **ch06's code is still inside the tiles, unseen** (A12, C3). Nothing
  of it is spoken or shown, but the user should know:
  - ch08's default sends dense tiles down the dense-tile path. On the
    frame and the crop that is 0 tiles; on the drone frames, a critic's
    probe (not committed) found about 0.8% of the tiles in labelling
    and 2.6% in paint. The drone times include that small share (Q10).
  - The merge after the waves reuses ch06's `_union` code, and ch07's
    bit packing is ch06's `pack_kernel`.
  - The drone footage (`assets/drones_labels`) was coloured by ch06's
    run kernel (`make_drone_frames.py`). V5 found ch08's labels equal
    on all 240 frames, so the build stage regenerates the footage from
    ch08 and checks that the frames come out the same (F9).

---

## Open questions for the user

- **Q1. One session for every CPU-vs-GPU number?** Default: yes. Five
  phrases of acts 1-2 change: "one point two seconds", "beats the CPU",
  "Twelve times", "close to CPU speed" and "about fifty milliseconds".
  r07-r08 stay the one cross-session comparison (July GPU estimates). The
  other way keeps v4's lines and July's numbers, but then the scoreboard
  mixes two CPU speeds (the huge row would read 830×).
- **Q2. Which CPU in the scoreboard?** Default: v4's baselines (540×,
  520×, 2.7×) with the twins in this document. Or the twins in the rows
  (470×, 130×, 4.1×).
- **Q3. The finale.** Default: the caveat is spoken ("In kernel time, the
  GPU now; with the copy, still the CPU"). Or the shorter "And now one
  small blob too, in kernel time" (about -3 s), with the copy only in the
  footnote.
- **Q4. The title card.** Default: "flood fill, from pixels to tiles"
  over "flood-fill-cuda". Others: "breadth-first flood fill on the GPU";
  "counting blobs with a BFS".
- **Q5. The copies on the big frame.** Default: "Two point two
  milliseconds, once the frame is on the card: inside the budget", with
  "kernel time, no copies" under the number. Or say the cost: "With the
  copies in and out, fifty-three." (about +2.5 s, ends act 3 on a loss).
- **Q6. "From a hundred milliseconds to three."** Default: r06's own
  kernel (32x), the last one the viewer saw. Or "from seventy" (the tuned
  pixel BFS, 23x), a kernel the video never showed.
- **Q7. Rendering.** The facts:
  - Manim's Cairo renderer draws on the CPU only. The chain code depends
    on Cairo behaviour (pixel arrays swapped under the render cache, PIL
    warps for the floor planes), so the OpenGL renderer would be a port,
    not a switch.
  - The venv's ffmpeg (imageio-ffmpeg, static 7.0.2) has libx264 and
    libx265, but no h264_nvenc. WSL does expose `libnvidia-encode.so.1`.
  - NVENC could only replace `assemble.py`'s one libx264 (crf 18)
    re-encode; Manim's partial movies go through PyAV.

  Proposal: Cairo frames in parallel jobs on the CPU, the GPU for every
  measurement. If you want NVENC, a project-local ffmpeg with nvenc for
  the final encode only (a small gain). Is that what "render with the
  GPU" meant?
- **Q8. Where the time goes.** Add to r11: "Now the search is the small
  part: reading the image and painting it back take most of it." (about
  +4 s)? Default: off.
- **Q9. Act 2's ch05 time.** Default: keep V5's 51.7 ms (memory at
  7,001 MHz), disclosed, spoken "about fifty". Or re-time the frame
  scene (CPU, ch05, tiles) in a second session at full clocks. That
  would likely read about 47 ms and 25×. The CPU would be timed again
  too, and "one point two seconds" could move (its rounds spread
  1.05-1.24 s). The laptop's 40 W power cap may pick P3 again.
- **Q10. The dense-tile share on the drones.** Default: keep ch08's
  default engine (about 1% of the drone tiles), disclosed above. Or
  re-time the drones with that path off, for a strictly piece-loop-only
  number.
- **Q11. The frame is transposed on screen.** To keep one orientation,
  the 900 px frame appears mirrored across its diagonal compared with
  the PNG. Nobody watching can tell; say if it matters (for example
  next to the project page).

---

## Build order (short)

1. The user's answers; then `numbers.py` (done for the defaults) and
   `script_chain.md` (done).
2. `state.picture()` transposed; `overlays.fmt_ms` / `fmt_x` from
   `card_ms` / `card_x`.
3. `machine.py`: `teams` up to 3, `token_kind`, `warp_closeup`; the
   gate's single flash and its held state; the gridless raster and the
   raster primitives.
4. `state.py`: `BEATS`, `boundary(14..18)`, the new floors,
   `score_rows(beat)`; `base.py` HOLD; `build/assemble.py` CHAIN;
   `build/seam_check.py` FADES_OUT.
5. Scenes: r02 ("GPU so far"), r03b (cue "beats", "1.5×"), r06 ("17×"),
   r09 (row 3 fills after the needle), r10_tiles, r11_rounds, r12_pieces,
   r13_frame, r14_reveal (from r11_reveal).
6. The drone footage from ch08 (`make_drone_frames.py` with TileWaves
   and `--tracker host`, under the GPU lock), compared with the old
   frames.
7. TTS: Kokoro for all 19 beats (the unchanged texts give the same
   takes), then `pace.py` at its defaults.
8. Checks: `build/layout_check.py` with the warp card open;
   `build/chain_snapcheck.py` on every snapping clip;
   `build/seam_check.py --cut chain` (17 seams); both layouts.
