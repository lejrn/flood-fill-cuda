# Narration script: the chain cut, v5 ("size, count, then tiles")

Status 2026-10-17 (night): v5 DRAFT on the user's decisions, every
CPU-vs-GPU number re-made like for like in one session, after two
critics' reviews. The full design, with every picture, number and
source, is `narration/chain_v5_proposal.md`; the numbers and the claim
checks are `scenes/chain/numbers.py` (114 checks). v4 (approved
2026-10-14) is in git history.

## v5's decisions

The user, 2026-10-17, all binding:

- The run-based labelling beat (v4's r10_runs) is not relevant: the
  video is about the BFS. It is gone, with every number that came from
  it. No "CCL", "runs", "spans" or "union-find", spoken or shown, and no
  GPU number from ch08's dense-tile path.
- The tile BFS and the tile waves replace it, as a third act after r09:
  first the thicker wave on the huge blob, then the same tiles on the
  whole frame, then the finale. Mechanism and numbers.
- **One session.** Every CPU-vs-GPU number comes from
  `v5_20261017T192640Z`. No July number is set against a CPU.
- **Like for like.** The CPU always does the work of the GPU it is
  compared with: the same neighbours (4-neighbour GPU steps against the
  4-neighbour CPU BFS, 8 against 8), the same output (no depth map where
  the GPU keeps none; many blobs: labels only) and the same setup (its
  arrays allocated once, as the GPU's are, and reset where the GPU resets
  its own). The ratios get smaller; that is accepted.
- **The finale says both sides of the small blob aloud**: in kernel time
  the GPU wins now; with the copy, the CPU still wins.
- **Title card**: "flood fill, from pixels to tiles" over
  "flood-fill-cuda".
- Kokoro `am_adam`, paced. Cairo renders in parallel on the CPU; the GPU
  does the measurements. Act 3 at the end. The cut stays under 5 minutes.

Defaults adopted with them:

- r13 keeps "once the frame is on the card". With the copies the frame
  takes 54.8 ms, over the budget; the r14 card's footnote says so.
- r11's "from a hundred milliseconds to three" compares r06's kernel, the
  last one the viewer saw.
- Act 2's seeds are timed at full clocks (53.9 ms).
- Every tile-wave number (the frame and the drones) is ch08's first
  version, which has no dense-tile path at all; ch08's default engine is
  timed beside it for reference only.
- One orientation for every floor: the 900 px frame is transposed at the
  source, so a thread's row in act 3 is a screen row.
- On-screen CPU-vs-GPU factors keep two significant figures (4.4×, 11×,
  340×): the CPU's round medians spread by up to 19%.

v5 has 19 beats: v4's first 14 (i0 to r09), four new act-3 beats
(r10_tiles, r11_rounds, r12_pieces, r13_frame) and a new finale
(r14_reveal). About 287 s (4:47) with Kokoro, paced.

History: v1 (11 beats, "you build a machine"), v2 (the two-act ladder,
`chain_v2_proposal.md`), v3 (the hardware look, `chain_v3_plan.md`) and
v4 (the memory story, `chain_v4_proposal.md`, 16 beats, 217.5 s) are in
git history.

## The idea

A drone show opens the video. On frames that small the compiled CPU
keeps up, but at 81 Mpx it needs over a fifth of a second (225 ms)
against a 33 ms budget. So where does the GPU win? The video earns each
answer:

- **Size** (act 1): one giant blob, the 64 Mpx test square. One block,
  its queue in shared memory, the spill, the global queue, 48 blocks,
  the gate, diagonal steps.
- **Count** (act 2): thousands of blobs at once. One launch per blob,
  tags, seeds found on the GPU: about 50 ms, faster than the CPU but
  over budget, because every red pixel still rides the queue.
- **Tiles** (act 3): the queue carries 32 x 32 tiles, each closed by one
  warp of 32 threads in registers. One gate per round: 127 rounds on the
  giant square instead of 4,001 rings. Then every tile finds its blob
  pieces, seeds send waves piece to piece, and the whole frame is
  counted in 6 rounds: 2.2 ms once the frame is on the card, inside the
  budget.

A three-row "CPU vs GPU" scoreboard fills in as the acts are won: one
small blob (the CPU, until the finale), one huge blob (the GPU, by
size; it jumps with the tiles), many blobs (the frame: the GPU, by
count; it jumps with the waves). Each row names its CPU: "4 neighbours",
"8 neighbours" or "same labels". The ending answers the opening question
with it.

Rules: every "So" fixes the "But" before it. Every failure is a visible
event. Amber means someone is waiting. At most three captions per beat,
a caption sub of at most four words. Every number comes from committed
JSON through `numbers.py`; "~" marks estimates; "kernel time" where it
applies. No "chapter" anywhere. A flashing element flashes at most twice
per second (the cut's photosensitivity rule allows three); large areas
never flash.

Vocabulary: acts 1 and 2 say ring, gate, queue, SM, block, thread, word,
bit, seed, tag, label, shared memory and global memory. Each new term is
defined where it is first spoken:

- **thread** and **block** (r02): "two hundred fifty-six workers, called
  threads, in one block".
- **warp** (r10): "one warp, thirty-two threads that run in step",
  shown growing out of r02's SM close-up.
- **registers** (r10): "The row sits in registers: the thread's own
  memory, closer than shared memory."

Act 3 also adds tile, round and piece. "Lane" is never spoken or shown
(the SM close-up's partitions read "32", not "32 lanes"). "Label" has
one sense throughout: a blob's own number (r08's tag "picks its colour:
its label"; r12's "smallest label"; the CPU's "same labels").

In act 1 a hue is a block; act 2 goes back to one hue per blob. Act 3
paints one blob in the act-1 blue, and many blobs one hue per label.

## The scoreboard

| row | r02 | r06 | r09 | r11 | r13 | r14 |
|---|---|---|---|---|---|---|
| one small blob | CPU 4.3× (0.121 vs 0.520 ms), CPU: 4 neighbours | | | | | **GPU 4.4×** (0.192 vs 0.044 ms), CPU: 8 neighbours |
| one huge blob | ? | GPU 11× (1,126 vs 101 ms) | | **GPU 340×** (1,065 vs 3.12 ms) | | |
| many blobs | ? | ? | GPU 3.8× (202 vs 53.9 ms) | | **GPU 100×** (225 vs 2.21 ms) | |

The strip shows only the pills; the times appear on the r14 card. Each
factor is against its own twin: row 2's CPU keeps a depth map at r06
(as ch03 does) and none from r11 (as the tiles keep none); row 3's CPU
resets its label map before the clock at r09 (as ch05 resets its
arrays) and on it from r13 (as the waves set up on theirs). Every GPU
time is kernel time.

## beat i0_problem

```text
A camera films a drone show; a filter turns each drone into a white blob. Count them all, every frame: thirty-three milliseconds.
```

- Unchanged (v4 take).

## beat i1_compare

```text
On frames this small, compiled CPU code keeps up. Scale up: eighty-one megapixels, same thirty-three milliseconds. The CPU needs over a fifth of a second. Where does the GPU win? Two places: size and count.
```

- Changed: "one point three seconds" (v4) becomes "over a fifth of a
  second". The CPU now does what the GPU does on the frame, labels only,
  on arrays it keeps: 225 ms, every round 214-238 ms. That is still
  nearly 7× the budget.
- Drone card (240 frames, one session): the CPU's bar, 0.679 ms (labels
  only), against the 33 ms budget, and the GPU as "?", as on the 81 Mpx
  card. Caption "0.59 Mpx" / "240 frames, one session". No GPU drone
  time is shown in the cut: with the copies the CPU wins the drones.
- "keeps up": the CPU's worst drone frame is 0.88 ms.
- Then the 81 Mpx card: the CPU stopwatch runs past 33 ms to 225 ms
  beside a GPU "?". Its stop cue moves from "seconds" to "second".

## beat r00_zoom

```text
Freeze. Zoom into one drone until its pixels show.
```

- Unchanged.

## beat r01_cpu

```text
On the CPU: paint a pixel, queue its red neighbours, repeat. One pixel per step, on one core, while twenty-four GPU SMs sit idle.
```

- Unchanged.

## beat r02_one_block

```text
So move it onto one SM, its queue in the SM's shared memory: two hundred fifty-six workers, called threads, in one block, paint a whole ring at once. But on one small blob, the CPU still wins. So: size first.
```

- Changed: "two hundred fifty-six threads" becomes "two hundred
  fifty-six workers, called threads, in one block" (thread and block,
  defined where first spoken). The "threads" cue stays.
- The race is like for like: the CPU's 4-neighbour BFS, 0.121 ms,
  against this very kernel, one block with its ring in shared memory,
  0.520 ms (CUDA events). The bar's label becomes "1 block" (was "GPU so
  far").
- Scoreboard row 1: "one small blob: CPU 4.3×", tag "CPU: 4 neighbours".

## beat r03_giant

```text
On one giant blob, that queue overflows: a block gets only forty-eight kilobytes of shared memory, and this queue needs over a hundred. So the overflow spills to global memory: the card's eight gigabytes.
```

- Unchanged.

## beat r03b_one_sm

```text
Now one block is a little faster than the CPU. But that's one SM of twenty-four.
```

- "is over twice the CPU" (v4) becomes "is a little faster than the
  CPU": 1.09× against the 4-neighbour CPU (1,195 vs 1,097 ms), above 1 in
  every CPU round (1.02-1.17). Caption "1.1×" / "vs CPU, kernel time",
  at "faster".

## beat r04_two_blocks

```text
So add a second block on a second SM. Squeezed onto one SM, two blocks ran one point seven times slower. But shared memory is private to its block. So the whole queue moves to global memory: every new pixel's address is written there. On big blobs, measured, that's no slower. Two blocks are twice as fast. But twenty-two SMs stay dark.
```

- Unchanged. "Twice as fast" is GPU against GPU: 2.04× (1,097 vs 538
  ms; paired 2.04).

## beat r05_all_blocks

```text
So use every SM: forty-eight blocks, two on each. Eight and a half times the CPU. Each colour is one block: they share one queue, so their pixels mix, and small rings keep only a few busy. We also tried other threads per block and block counts: on a smaller square, the best gained fifteen percent. We kept forty-eight by two hundred fifty-six.
```

- "Eighteen" (v4) becomes "Eight and a half": 8.5× against the
  4-neighbour CPU (1,195 vs 141 ms; 8.0-9.1 over the CPU's rounds).
  Caption "8.5×" / "kernel time".

## beat r05b_snake

```text
But on a thin snake, forty-eight blocks lose to one. Why? Every ring, the chip stops twice at a gate, two microseconds each.
```

- Unchanged.

## beat r06_diagonals

```text
So pass fewer gates: step diagonally too. On a square, half the rings, half the gates.
```

- Text unchanged. Like for like, the CPU steps diagonally too from here:
  the 8-neighbour BFS with its depth map (ch03 keeps one), 1,126 ms,
  against 101 ms.
- Caption "11×" / "CPU: 8 neighbours too". Scoreboard row 2: "one huge
  blob: GPU 11×", tag "CPU: 8 neighbours".

## beat r07_real_frame

```text
Size is won. Now count: twenty-five hundred blobs in the big frame, one launch each. Slower than the CPU: we built for size, not count.
```

- Text unchanged. The estimate is re-made in this session with v4's
  method, at 8 neighbours: ~1,462 ms, against the labels-only CPU's 202
  ms (its label map reset before the clock, as act 2's loops reset
  theirs): the CPU by 7.2×. v4's host clock on the same calls reads 1,919
  ms and the full loop 1,647 ms: slower by every measure.
- The meter's CPU tick sits at the 81 Mpx card's CPU time, 225 ms
  (`njit_ms`, its label map reset on the clock), not at the 202 ms the
  claims in r07-r09 use (reset before the clock). The tick carries no
  number and the scale is log, so this is meant: every needle (~1,462,
  ~850, 53.9 ms) sits on the same side of both. The needle swings past it
  to ~1,462 ms. Caption "~1,462 ms" / "estimate".

## beat r08_tags

```text
So tag each queue entry with its blob: one spare bit of the entry's word, in global memory. Each new pixel inherits the tag, and the tag picks its colour: its label. One bit tells two blobs apart: two per launch, faster. But still slower than the CPU: only two seeds, each handed in.
```

- Changed: "two per launch, about CPU speed. But only two seeds, each
  handed in." becomes "two per launch, faster. But still slower than the
  CPU: only two seeds, each handed in."
- ~850 ms (estimate, 8 neighbours, this session), against the CPU's 202
  ms: the CPU by 4.2×. Faster than one per launch by every measure
  (paired 1.72×; full loops 1,647 and 1,147 ms), slower than the CPU by
  every measure.
- Build: the two blobs fill 8-way now (the estimate's kernel is ch04 at
  8 neighbours); the needle moves at "faster" and settles well right of
  the CPU tick, not beside it; the floor play ends at "faster", not at
  "speed".

## beat r09_seeds

```text
So the GPU finds its own seeds: every blob in one fill launch, about fifty milliseconds. Faster than the CPU. But over budget: every red pixel still rides the queue.
```

- Changed: "fifty-one" (v4) becomes "about fifty" (53.9 ms at full
  clocks), and the beat now says both results aloud: "Faster than the
  CPU" (202 vs 53.9 ms, 3.8×, 3.5-4.2 over the CPU's rounds) and "But
  over budget" (over 33 ms), whose cause is the line act 3 fixes.
- The meter stops at 53.9 ms, over 33. Caption "53.9 ms" / "still over
  33 ms", at "budget".
- At "Faster than the CPU", scoreboard row 3 fills: "many blobs: GPU
  3.8×" (the same 81 Mpx frame, 2,522 blobs), tag "CPU: same labels".
  COUNT is not won yet (no tick).
- The queue counter still climbs to 13,451,960 red pixels. Its last
  line is the bridge into act 3.

## beat r10_tiles

```text
So the queue carries tiles, not pixels: squares of thirty-two by thirty-two. One warp, thirty-two threads that run in step, takes a tile. Each thread holds one row, one bit per pixel. The row sits in registers: the thread's own memory, closer than shared memory. Fill along every row at once, step diagonally, fill along the columns, and repeat until nothing changes.
```

- New. Defines warp ("thirty-two threads that run in step") and
  registers ("the thread's own memory, closer than shared memory"), each
  in its own sentence, so no sentence carries more than one new term.
- The labelled frame goes back to red. The queue's red tokens turn into
  tile tokens and the launch pill fades.
- The floor zooms into the frame's largest blob (76,090 px); a 7 x 7
  grid of real tiles lands on it. The tiles the search reached in its
  first rounds are already blue; the centre tile, on the blob's edge,
  is outlined.
- SM 0's close-up opens as in r02; one "32" partition unfolds into the
  warp card: 32 rows of 32 bit cells, the tile's real mask (543 of 1,024
  red). Its 17 entry bits light on the left edge.
- At "registers" the card's frame pulses (the new-part convention).
- The recorded sweeps play: row 86, diagonal 109, column 181, row 387,
  diagonal 481, column 541, row 543, then one more trip with no change.
  On "changes" the threads vote, the card folds back into SM 0, and the
  centre and the six tiles of its round turn blue together. As the card
  folds back, each SM's third cube lands, quietly (72 blocks: 3 per SM),
  not under the line that introduces tiles.
- Captions: "1 tile, 1 warp" / "32 x 32 pixels"; "1 bit per pixel" /
  "a row per thread"; carried "4 trips" / "the last changes nothing".

## beat r11_rounds

```text
Only the tile's border bits go out, into the tiles around it, and those join the next round's queue. One gate per round, not two per ring. On the giant square, four thousand rings become a hundred twenty-seven rounds: from a hundred milliseconds to three. But that's one blob, from one seed.
```

- New. "The tile's" and "the tiles around it" name what "its" and
  "its neighbours" left open after the beat cut (neighbours were pixels
  until now).
- r06's kernel 101 ms, the tiles 3.12 ms (paired 32×).
- The closed tile's border bits light (31, 26, 18 and 2) and travel
  only into the 4 tiles they really open (above, right, and the two
  upper corners); the others were open already. 4 tile tokens drop into
  the queue. The gate flashes once and the 4 tiles turn blue (round 5);
  the context then holds still until "On the giant square" (no replay
  of later rounds).
- The floor swells into the giant square, drawn from the real tile
  rounds (250 x 250 tiles, about 12 shades): the seed tile, then 125
  rings of tiles, one per round, then one round that only revisits. Two
  slow rounds flash; then the gate holds amber while the counter runs
  to 127. The inset says "stress test"; the tag stays COUNT.
- "from a hundred milliseconds to three": scoreboard row 2 jumps from
  "GPU 11×" to "GPU 340×" (1,065 vs 3.12 ms; the CPU's 8-neighbour BFS,
  without a depth map, as the tiles keep none). One change; no caption
  lands with it.
- "But that's one blob": the floor pulls back to the 81 Mpx frame: one
  blob blue, 2,521 still red.
- Captions: "1 gate" / "per round"; "127 rounds" / "was 4,001 rings";
  carried "1 blob" / "from 1 seed".

## beat r12_pieces

```text
So first, every tile finds the blob pieces inside it, in registers, all at once. Some pieces are seeds: each seed sends a wave from piece to piece, one round at a time. Where two waves meet, the meeting is noted, and at the end each blob keeps its smallest label.
```

- New (text unchanged from the evening draft). A real 384 px piece of
  the frame, run as its own small image and shown inside a border: 12 x
  12 tiles, 8 blobs (6 of them cut by its edge), 135 pieces (at most 4
  per tile). The SMs pulse once; each piece gets an outline and a dot at
  its first pixel; 36 seeds spark (11 only because their tile is on the
  lattice) and keep a mark in their wave's hue. A dashed ring stands
  round each of the 2 tiny blobs (2 and 25 px) at "8 blobs".
- Six rounds, one gate flash each, 0.55 s apart: pieces take their wave's
  hue as they are claimed (50, 38, 7, 3, 1, then 0). Faint ticks mark
  the 116 meetings and brighten once at "meet".
- "at the end": the marks fade but 8 dots, which grow: each blob's
  smallest anchor over all its pieces, its label (in 5 blobs no seed's).
  At "smallest" they grow once more and every blob turns to the hue its
  label piece shows.
- Captions: "pieces" / "blobs cut by tiles"; "36 seeds" / "8 blobs";
  carried "min" / "smallest label wins".

## beat r13_frame

```text
On the big frame: over eleven thousand seeds, six rounds, every blob counted. Two point two milliseconds, once the frame is on the card: inside the budget.
```

- New (text unchanged from the evening draft); it holds: 2.21 ms,
  kernel time, ch08's first version. With the copies the frame takes
  54.8 ms, over 33: hence "once the frame is on the card".
- The crop goes back to red and its border fades; the frame around it
  comes in from its edge, and the floor pulls back to the whole frame,
  drawn from every frame pixel (`assets/tiles/input_blobs_red.npz`, no
  mosaic). The 11,609 seeds fade in as dots over at least 1 s (no white
  rings at this scale); six gate flashes, 0.5 s apart, as the rounds
  sweep out from them (real claim rounds, sampled at 900 px); the frame
  ends labelled.
- At "Two" the needle's old value goes; once the sixth flash is over,
  the needle slides from 53.9 ms past the 33 ms line to 2.21 ms (24×
  down) and turns teal. Its value lands with the carried caption at
  "milliseconds". At "budget", out then in: row 3's old verdict and the
  tag go, then COUNT is won and scoreboard row 3 reads "GPU 100×", up
  from "GPU 3.8×" (225 vs 2.21 ms; the CPU's labels with their reset on
  the clock, as the waves set up on theirs).
- Captions (two: a third would come too soon after the first): r12's
  carried "min" goes with the crop's hues; "11,609 seeds" / "6 rounds,
  2,522 blobs" at "seeds"; carried "2.21 ms" / "kernel time, no
  copies".

## beat r14_reveal

```text
One huge blob, or many at once: the GPU. And one small blob? In kernel time, the GPU now; with the copy to the card, still the CPU. Every drone, every frame.
```

- New text; it says both sides of the small blob (decision 3). "with the
  copy to the card" echoes r13's "once the frame is on the card".
- Like for like, the small blob's CPU is now the 8-neighbour BFS without
  a depth map, as the tiles keep none: in kernel time the GPU wins by
  4.4× (0.192 vs 0.044 ms); with the copy the GPU takes 0.665 ms and the
  CPU wins by 3.5×.
- The full scoreboard opens; rows 2 and 3 are framed at "huge". Row 1
  flips in two steps, each row's tag drawn under its name:
  1. At "And one small blob?": the CPU cell alone changes, from 0.121 to
     0.192 ms, its tag from "CPU: 4 neighbours" to "CPU: 8 neighbours";
     the GPU cell and the winner go to "?".
  2. At "now": the GPU cell lands at 0.044 ms and the winner at "GPU
     4.4×". At "copy" the footnote's copy line lights.
- Card rows (the header says "one session, GPU kernel time"):
  - one small blob: 0.192 / 0.044 ms, GPU 4.4×, CPU: 8 neighbours
  - one huge blob: 1,065 / 3.12 ms, GPU 340×, CPU: 8 neighbours
  - many blobs: 225 / 2.21 ms, GPU 100×, CPU: same labels
- Footnote: "With the copies: small blob 0.665 ms (CPU 0.192), huge blob
  44.3 ms, frame 54.8 ms." The huge blob and the frame stay the GPU's
  with the copies too (24× and 4.1×).
- Captions: "the GPU" / "huge or many blobs"; "small blob too" / "in
  kernel time"; "every drone frame" / "with copies: 1.26 ms" (the worst
  frame, copies included, far under 33).
- Then the live drones, with no GPU-versus-CPU claim: with the copies
  the GPU takes 0.977 ms per drone frame and the CPU's 0.679 ms is
  faster; both are far under the budget.
- The title card: "flood fill, from pixels to tiles" over
  "flood-fill-cuda".
