# Narration script: the chain cut, v5 ("size, count, then tiles")

Status 2026-10-17: v5 DRAFT, revised after review, for the user's
answers. The full design, with every picture, number and source, is
`narration/chain_v5_proposal.md`; the numbers and the claim checks are
`scenes/chain/numbers.py`. v4 (approved 2026-10-14) is in git history.

## v5's decisions

The user, 2026-10-17:

- The run-based labelling beat (v4's r10_runs) is not relevant: the
  video is about the BFS. It is gone, with every number that came from
  it: the drone card's GPU time, the count row, the 2.96 ms, the frame
  ratios. No "CCL", "runs", "spans" or "union-find", spoken or shown.
- The tile BFS and the tile waves replace it, as a third act after
  r09: first the thicker wave on the huge blob, then the same tiles on
  the whole frame with labels, then the finale. Show the mechanism and
  the numbers. The cut may run up to five minutes.
- Voice: Kokoro `am_adam`, paced, as v4.

This draft's defaults, each an open question in the proposal:

- One session for every CPU-vs-GPU number: `v5_20261017T161122Z`. The
  CPU there is faster than in July, so five phrases of acts 1 and 2
  change: "one point three seconds" becomes "one point two", "over twice
  the CPU" becomes "beats the CPU", "Eighteen times" becomes "Twelve
  times", "about CPU speed" becomes "close to CPU speed", and "fifty-one
  milliseconds" becomes "about fifty" (act 2's 51.7 ms ran at a lower
  memory clock; at full clocks the same kernel took 47.1).
- GPU-to-GPU ratios keep their July sessions (each within one session).
  r07 and r08 are the one cross-session comparison: July GPU estimates
  against today's CPU, and both claims also hold against July's CPU.
- Row 3 ("many blobs") is the 81 Mpx frame, the one the meter shows. v4's
  1 Mpx noise scene is dropped: on it, the tile waves send almost every
  tile down the dense-tile path, which v5 leaves out.
- The CPU column keeps v4's baselines (4-way BFS on one blob; labels
  plus a fill per blob on many). The like-for-like twins are in the
  proposal.
- On-screen CPU-vs-GPU factors keep two significant figures (1.5×, 17×,
  540×): the CPU's round medians spread by up to 9%.
- The finale flips the small-blob row to the GPU in kernel time and says
  aloud that with the copy the CPU still wins it.
- One orientation for every floor: the 900 px frame is transposed at the
  source, so a thread's row in act 3 is a screen row.

v5 has 19 beats: v4's first 14 (i0 to r09), four new act-3 beats
(r10_tiles, r11_rounds, r12_pieces, r13_frame) and a new finale
(r14_reveal). About 273 s (4:33) with Kokoro, paced.

History: v1 (11 beats, "you build a machine"), v2 (the two-act ladder,
`chain_v2_proposal.md`), v3 (the hardware look, `chain_v3_plan.md`) and
v4 (the memory story, `chain_v4_proposal.md`, 16 beats, 217.5 s) are in
git history.

## The idea

A drone show opens the video. On frames that small the compiled CPU
keeps up, but at 81 Mpx it needs 1.2 s against a 33 ms budget. So where
does the GPU win? The video earns each answer:

- **Size** (act 1): one giant blob, the 64 Mpx test square. One block,
  its queue in shared memory, the spill, the global queue, 48 blocks,
  the gate, diagonal steps.
- **Count** (act 2): thousands of blobs at once. One launch per blob,
  tags, seeds found on the GPU: about 50 ms, over budget, and every red
  pixel still rides the queue.
- **Tiles** (act 3): the queue carries 32 x 32 tiles, each closed by one
  warp of 32 threads in registers. One gate per round: 127 rounds on the
  giant square instead of 4,001 rings. Then every tile finds its blob
  pieces, seeds send waves piece to piece, and the whole frame is
  counted in 6 rounds: 2.2 ms once the frame is on the card, inside the
  budget.

A three-row "CPU vs GPU" scoreboard fills in as the acts are won: one
small blob (the CPU, until the finale), one huge blob (the GPU, by
size; it jumps with the tiles), many blobs (the frame: the GPU, by
count; it jumps with the waves). The ending answers the opening
question with it.

Rules: every "So" fixes the "But" before it. Every failure is a visible
event. Amber means someone is waiting. At most three captions per beat,
a caption sub of at most four words. Every number comes from committed
JSON through `numbers.py`; "~" marks estimates; "kernel time" where it
applies. No "chapter" anywhere. A flashing element flashes at most twice
per second.

Vocabulary: acts 1 and 2 say ring, gate, queue, SM, block, thread, word,
bit, seed, tag, label, shared memory and global memory. Act 3 adds warp
("one warp, thirty-two threads", shown growing out of r02's SM close-up),
registers ("closer than shared memory"), tile, round and piece. "Lane"
is never spoken.

In act 1 a hue is a block; act 2 goes back to one hue per blob. Act 3
paints one blob in the act-1 blue, and many blobs one hue per label.

## beat i0_problem

```text
A camera films a drone show; a filter turns each drone into a white blob. Count them all, every frame: thirty-three milliseconds.
```

- Unchanged (v4 take).

## beat i1_compare

```text
On frames this small, compiled CPU code keeps up. Scale up: eighty-one megapixels, same thirty-three milliseconds. The CPU needs one point two seconds. Where does the GPU win? Two places: size and count.
```

- One word changes: "three" becomes "two" (CPU 1,157 ms in the v5
  session, 1,346 in July's).
- Drone card (240 frames, one session): CPU 1.46 ms, GPU 0.11 ms,
  budget 33 ms. The GPU time is the tile waves' (v4 showed the run-based
  0.93 ms). Then the 81 Mpx card: the CPU stopwatch runs past 33 ms to
  1.2 s beside a GPU "?".

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
So move it onto one SM, its queue in the SM's shared memory: two hundred fifty-six threads paint a whole ring at once. But on one small blob, the CPU still wins. So: size first.
```

- Text unchanged. The race: CPU 0.12 ms against "GPU so far" 0.17 ms
  (v5 session, CUDA events; the bar was "best GPU", which the tiles
  later beat). Scoreboard row 1: "one small blob: CPU 1.4×".

## beat r03_giant

```text
On one giant blob, that queue overflows: a block gets only forty-eight kilobytes of shared memory, and this queue needs over a hundred. So the overflow spills to global memory: the card's eight gigabytes.
```

- Unchanged.

## beat r03b_one_sm

```text
Now one block beats the CPU. But that's one SM of twenty-four.
```

- Changed: "is over twice the CPU" becomes "beats the CPU" (1.55 in the
  v5 session). Caption "1.5×" / "vs CPU, kernel time", at "beats".

## beat r04_two_blocks

```text
So add a second block on a second SM. Squeezed onto one SM, two blocks ran one point seven times slower. But shared memory is private to its block. So the whole queue moves to global memory: every new pixel's address is written there. On big blobs, measured, that's no slower. Two blocks are twice as fast. But twenty-two SMs stay dark.
```

- Unchanged. "Twice as fast" is 2.03x in the v5 session (1,077 vs 530 ms).

## beat r05_all_blocks

```text
So use every SM: forty-eight blocks, two on each. Twelve times the CPU. Each colour is one block: they share one queue, so their pixels mix, and small rings keep only a few busy. We also tried other threads per block and block counts: on a smaller square, the best gained fifteen percent. We kept forty-eight by two hundred fifty-six.
```

- Changed: "Eighteen" becomes "Twelve" (12.1 in the v5 session).
  Caption "12×" / "kernel time".

## beat r05b_snake

```text
But on a thin snake, forty-eight blocks lose to one. Why? Every ring, the chip stops twice at a gate, two microseconds each.
```

- Unchanged.

## beat r06_diagonals

```text
So pass fewer gates: step diagonally too. On a square, half the rings, half the gates.
```

- Text unchanged. Caption and scoreboard row 2: "one huge blob: GPU
  17×" (CPU 1,666 ms, GPU 99.9 ms, v5 session). Caption sub "vs CPU,
  kernel time".

## beat r07_real_frame

```text
Size is won. Now count: twenty-five hundred blobs in the big frame, one launch each. Slower than the CPU: we built for size, not count.
```

- Unchanged. The meter's needle swings past the CPU tick (now 1.2 s)
  to ~2,181 ms (a July estimate; slower than both CPUs).

## beat r08_tags

```text
So tag each queue entry with its blob: one spare bit of the entry's word, in global memory. Each new pixel inherits the tag, and the tag picks its colour: its label. One bit tells two blobs apart: two per launch, close to CPU speed. But only two seeds, each handed in.
```

- Changed: "about CPU speed" becomes "close to CPU speed" (~1,368 ms,
  a July estimate: 18% slower than today's CPU, 2% slower than July's).

## beat r09_seeds

```text
So the GPU finds its own seeds: every blob in one fill launch, about fifty milliseconds. But every red pixel still rides the queue.
```

- Changed: "fifty-one" becomes "about fifty". The meter stops at 51.7
  ms (v5 session, at a 7,001 MHz memory clock; 47.1 ms at full clocks in
  the tile waves' own session), over 33.
- New picture: about 0.6 s after the needle settles, before "But",
  scoreboard row 3 fills: "many blobs: GPU 22×" (the same 81 Mpx frame,
  2,522 blobs; CPU 1,157 ms, GPU 51.7 ms). The GPU already beats the
  CPU on count; it misses the budget. COUNT is not won yet.
- The queue counter still climbs to 13,451,960 red pixels. Its last
  line is the bridge into act 3.

## beat r10_tiles

```text
So the queue carries tiles, not pixels: squares of thirty-two by thirty-two. One warp, thirty-two threads, takes a tile. Each thread holds one row as a thirty-two bit word, one bit per pixel, in registers, closer than shared memory. Fill along every row at once, step diagonally, fill along the columns, and repeat until nothing changes.
```

- New (act 3). The labelled frame goes back to red. The queue's red
  tokens turn into tile tokens, the launch pill fades, and each SM's
  third cube lands (72 blocks: 3 per SM).
- The floor zooms into the frame's largest blob (76,090 px); a 7 x 7
  grid of real tiles lands on it. The tiles the search reached in its
  first rounds are already blue; the centre tile, on the blob's edge,
  is outlined.
- SM 0's close-up opens as in r02; one "32 lanes" partition unfolds
  into the warp card: 32 rows of 32 bit cells, the tile's real mask
  (543 of 1,024 red). Its 17 entry bits light on the left edge.
- The recorded sweeps play: row 86, diagonal 109, column 181, row 387,
  diagonal 481, column 541, row 543, then one more trip with no change.
  On "changes" the threads vote, the card folds back, and the centre
  and the six tiles of its round turn blue together.
- Captions: "1 tile, 1 warp" / "32 x 32 pixels"; "1 bit per pixel" /
  "a row per thread"; carried "4 trips" / "then nothing changes".

## beat r11_rounds

```text
Only its border bits go out, into its neighbours, and those join the next round's queue. One gate per round, not two per ring. On the giant square, four thousand rings become a hundred twenty-seven rounds: from a hundred milliseconds to three. But that's one blob, from one seed.
```

- New. The closed tile's border bits light (31, 26, 18 and 2) and
  travel only into the 4 tiles they really open (above, right, and the
  two upper corners); the others were open already. 4 tile tokens drop
  into the queue. The gate flashes once per round, at most twice per
  second, as the next rounds close.
- The floor swells into the giant square, drawn from the real tile
  rounds (250 x 250 tiles, about 12 shades): the seed tile, then 125
  rings of tiles, one per round, then one round that only revisits. Two
  slow rounds flash; then the gate holds amber while the counter runs
  to 127. The inset says "stress test"; the tag stays COUNT.
- "from a hundred milliseconds to three": scoreboard row 2 jumps from
  "GPU 17×" to "GPU 540×" (kernel time, v5 session).
- "But that's one blob": the floor pulls back to the 81 Mpx frame: one
  blob blue, 2,521 still red.
- Captions: "1 gate" / "per round"; "127 rounds" / "was 4,001 rings";
  carried "1 blob" / "from 1 seed".

## beat r12_pieces

```text
So first, every tile finds the blob pieces inside it, in registers, all at once. Some pieces are seeds: each seed sends a wave from piece to piece, one round at a time. Where two waves meet, the meeting is noted, and at the end each blob keeps its smallest label.
```

- New. A real 384 px piece of the frame, run as its own small image and
  shown inside a border: 12 x 12 tiles, 8 blobs (6 of them cut by its
  edge), 135 pieces (at most 4 per tile). The SMs pulse once; each piece
  gets an outline and a dot at its first pixel; 36 seeds spark (11 only
  because their tile is on the lattice).
- Six rounds, one gate flash each, 0.5 s apart: pieces take their wave's
  hue as they are claimed (50, 38, 7, 3, 1, then 0). Faint ticks mark
  the 116 meetings and brighten once at "meet".
- "at the end": the marks fade and every blob turns to one hue, its
  smallest label.
- Captions: "pieces" / "blobs cut by tiles"; "36 seeds" / "8 blobs";
  carried "min" / "smallest label wins".

## beat r13_frame

```text
On the big frame: over eleven thousand seeds, six rounds, every blob counted. Two point two milliseconds, once the frame is on the card: inside the budget.
```

- New. The crop goes back to red and its border fades; pull back to the
  whole frame: 11,609 seeds spark; six gate flashes, 0.5 s apart, as the
  rounds sweep out from them (real claim rounds, sampled at 900 px); the
  frame ends labelled.
- At "milliseconds" the needle slides from 51.7 ms past the 33 ms line
  to 2.24 ms and turns teal. At "budget", COUNT is won and scoreboard
  row 3 jumps from "GPU 22×" to "GPU 520×".
- Captions (two: a third would come too soon after the first):
  "11,609 seeds" / "6 rounds, 2,522 blobs"; carried "2.24 ms" /
  "kernel time, no copies".
- With the copies the frame takes 53 ms; the r14 card's footnote says
  so.

## beat r14_reveal

```text
One huge blob, or many at once: the GPU. And one small blob? In kernel time, the GPU now; with the copy, still the CPU. Every drone, every frame.
```

- New text (v4's r11_reveal). The full scoreboard opens; rows 2 and 3
  are framed at "huge". At "now" row 1 flips from "CPU 1.4×" to "GPU
  2.7×" (0.12 vs 0.045 ms); at "copy" the footnote's copy line lights
  (GPU 0.60 ms with the copies, CPU 0.12).
- Card rows: 0.12 / 0.045 ms GPU 2.7×; 1,666 / 3.10 ms GPU 540×; 1,157 /
  2.24 ms GPU 520×; every tag "one session, kernel time".
- Captions: "the GPU" / "huge or many blobs"; "small blob too" / "in
  kernel time"; "every drone frame" / "kernel: 0.13 ms max".
- Then the live drones (GPU 0.11 ms per frame, CPU 1.46 ms, budget 33
  ms). The title card: "flood fill, from pixels to tiles" over
  "flood-fill-cuda" (proposal Q4).
