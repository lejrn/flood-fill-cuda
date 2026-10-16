# Narration script: the chain cut, v4 ("size, then count": the memory story)

Status 2026-10-14: v4 approved by the user. The full design, with every
visual and source, is `narration/chain_v4_proposal.md`; the numbers and
the claim checks are `scenes/chain/numbers.py`.

History: v1 (11 beats, "you build a machine") is in git history and in
`out/chain_v1/`. v2 is the two-act ladder (`chain_v2_proposal.md`). v3
is the hardware look with the same text (`chain_v3_plan.md`). v4 adds
the memory story: shared memory, the spill, the global queue, the tag
bit and the row read. It has 16 beats (v3 had 14): r03b_one_sm and
r05b_snake are split from v3's r03 and r05.

The user's answers to the proposal's questions:

- Q1: run the owner map (real per-pixel block ownership on the 64 Mpx
  square), not the schematic fallback.
- Q2: add the one "1 SM, 2 blocks" measurement: ch02's pinned pair on
  the 36 Mpx square, same SM vs two SMs, placement the only variable
  (308.4 vs 183.8 ms, 1.68x). One short clause in r04; r04 keeps at
  most three captions.
- Q3: r10_runs captions the dense row read "coalesced" / "by design,
  not profiled".
- Q4: the owner-map lens in r05, if the owner crops exist.
- Q5: no "sixty-four tags".
- Q6: the cut may grow to five minutes. The pacing knobs were already
  raised 25% (commit 3f88900).
- Q7: 48 KB and 8 GB are device-dump constants in `numbers.py`, with
  their source.
- Q8: no separate r10_memory beat.
- Q9: no ch01 thread-sweep caption.
- Q10: no image lane before r08.

## The idea

A drone show opens the video. On frames that small the compiled CPU
keeps up, but at 81 Mpx it needs 1.3 s against a 33 ms budget. So where
does the GPU win? Two places, and the video earns each one:

- **Size** (act 1): one giant blob, the 64 Mpx test square. One block
  on one SM with its queue in shared memory; the overflow spills to
  global memory; a second block moves the whole queue there; then all
  48 blocks, then diagonal steps. Each is a "But... So" rung.
- **Count** (act 2): thousands of blobs at once. One launch per blob
  loses to the CPU, then tags (one bit of each queue word), then seeds
  found on the GPU, then runs read row by row.

A three-row "CPU vs GPU" scoreboard fills in as the acts are won: one
small blob (the CPU), one huge blob (the GPU, by size), many blobs (the
GPU, by count). The ending answers the opening question with it.

Rules: every "So" fixes the "But" before it. Every failure is a visible
event. Amber means someone is waiting. At most three captions per beat,
a caption sub of at most four words. Every number comes from committed
JSON through `numbers.py`; "~" marks estimates; "kernel time" where it
applies. The name ("connected-component labelling") is on the title
card, never spoken. No "chapter" anywhere.

In act 1 a hue is a block (block 0 blue, block 1 green, the rest a
golden-angle walk); the floors are real owner maps. Act 2 goes back to
one hue per blob.

## beat i0_problem

```text
A camera films a drone show; a filter turns each drone into a white blob. Count them all, every frame: thirty-three milliseconds.
```

- Reuses the three-pane intro's camera and filter panels; the budget line
  is cued to "thirty-three milliseconds".

## beat i1_compare

```text
On frames this small, compiled CPU code keeps up. Scale up: eighty-one megapixels, same thirty-three milliseconds. The CPU needs one point three seconds. Where does the GPU win? Two places: size and count.
```

- Drone card (measured, one session, 240 frames): CPU 3.1 ms, GPU 0.93 ms,
  budget 33 ms. Then the 81 Mpx card: the CPU stopwatch runs past 33 ms to
  1.3 s beside a GPU "?". The goal chip and the SIZE / COUNT chips appear.

## beat r00_zoom

```text
Freeze. Zoom into one drone until its pixels show.
```

- Frame 216 of the footage, push-in on one drone; its pixels become red
  cells on the iso floor. Caption: the drone's pixel count.

## beat r01_cpu

```text
On the CPU: paint a pixel, queue its red neighbours, repeat. One pixel per step, on one core, while twenty-four GPU SMs sit idle.
```

- One worker crawls the drone, one cell per step; the 24 SMs ripple and
  stay dark. Captions "1 pixel", "1 core" (the i9 has 14), "24 SMs".

## beat r02_one_block

```text
So move it onto one SM, its queue in the SM's shared memory: two hundred fifty-six threads paint a whole ring at once. But on one small blob, the CPU still wins. So: size first.
```

- The SM close-up holds the queue strip in its shared-memory bar: red
  tokens drop in at "queue", the bar's outline pulses at "shared memory".
  It flies back before "paint"; SM 0's shelf stays "full".
- Split floor, then the race on a 16 K px test blob: CPU 0.15 ms, best
  GPU 0.37 ms. Scoreboard row 1: "one small blob: CPU 2.4x".
- Captions (unchanged): "256 threads"; carried "CPU wins / one small blob".

## beat r03_giant

```text
On one giant blob, that queue overflows: a block gets only forty-eight kilobytes of shared memory, and this queue needs over a hundred. So the overflow spills to global memory: the card's eight gigabytes.
```

- The drone swells into the 64 Mpx test square ("stress test"). The
  shelf gauge grows out of SM 0: the box is 48 KB, the queue region two
  thirds of it (8,192 slots), the bar the real two-ring load.
- The sweep runs from "overflows" to "gigabytes". From real ring 1,024
  red tokens pour from the region's end to VRAM chip 0; later rings
  load some back. The four VRAM chips pulse at "eight gigabytes".
- Captions: "48 KB" / "per block, by default"; "125 KB" / "two rings at
  peak"; carried "spill" / "to global memory".

## beat r03b_one_sm

```text
Now one block is over twice the CPU. But that's one SM of twenty-four.
```

- The painted square holds. The 23 other SMs come up idle in a wave and
  go dark at "twenty-four". This one-block run is the shared ring plus
  the global spill, as r03_giant shows. r03_giant's 0.8 s hold moves here.
- Captions: "2.46x" / "vs the CPU, kernel time"; carried "1 of 24 SMs".

## beat r04_two_blocks

```text
So add a second block on a second SM. Squeezed onto one SM, two blocks ran one point seven times slower. But shared memory is private to its block. So the whole queue moves to global memory: every new pixel's address is written there. On big blobs, measured, that's no slower. Two blocks are twice as fast. But twenty-two SMs stay dark.
```

- Block 1's green cube drops onto SM 2 (measured placement, not SM 1).
- "Squeezed onto one SM" (Q2): ch02's pinned pair on the 36 Mpx square,
  308.4 vs 183.8 ms. Spoken only: r04 keeps its three captions.
- The two shelf cards grow out of SM 0 (full) and SM 2 (empty) at
  "private". At "the whole queue moves" SM 0's tokens drain to VRAM,
  the label "queue" and the gate rail appear.
- The refill paints the 2-block owner mosaic: blue and green dots,
  mixed, never two halves. Loads come up from the VRAM queue, stores go
  down to its rear.
- Captions: "no slower" / "16 Mpx square" (cue the second "slower");
  "2x again" / "kernel time"; carried "22 SMs idle".

## beat r05_all_blocks

```text
So use every SM: forty-eight blocks, two on each. Eighteen times the CPU. Each colour is one block: they share one queue, so their pixels mix, and small rings keep only a few busy. We also tried other threads per block and block counts: on a smaller square, the best gained fifteen percent. We kept forty-eight by two hundred fifty-six.
```

- The 22 dark SMs light out from SMs 0 and 2; 48 cubes land in the
  measured order, each in its hue. The square refills with the 48-block
  owner mosaic, then replays slowly over the small rings (few cubes
  bright) and fast through the widest band.
- Optional: a lens over the widest ring with a 64 x 64 crop of the real
  owner map (Q4).
- A small grid card shows the launch-shape sweep: a flat top, the
  default 48 x 256 outlined, no numbers on it.
- Captions: "18x" / "kernel time"; "few busy" / "on small rings";
  carried "+15% at best" / "16 Mpx square".

## beat r05b_snake

```text
But on a thin snake, forty-eight blocks lose to one. Why? Every ring, the chip stops twice at a gate, two microseconds each.
```

- The snake: 48 blocks 143 ms, 2 blocks 128 ms, 1 block (shared queue)
  62 ms. Only block 0 ever has work; the 47 idle cubes are dim, then
  amber from "gate" on. The gate flashes amber twice per ring.
- Captions: "48 lose to 1"; carried "65,792 stops / ~2 µs each".

## beat r06_diagonals

```text
So pass fewer gates: step diagonally too. On a square, half the rings, half the gates.
```

- The stencil goes from 4 to 8 arrows; the diamond becomes a square;
  8,001 -> 4,001 rings. The floor is the 8-way owner mosaic. Caption
  "25.8x". Scoreboard row 2: "one huge blob: GPU 25.8x".

## beat r07_real_frame

```text
Size is won. Now count: twenty-five hundred blobs in the big frame, one launch each. Slower than the CPU: we built for size, not count.
```

- Pull back to the 81 Mpx frame (2,522 blobs); the cubes turn plain
  blue (act 2: one hue per blob). A launch counter races; the meter
  swings past the CPU to ~2,181 ms (estimate).

## beat r08_tags

```text
So tag each queue entry with its blob: one spare bit of the entry's word, in global memory. Each new pixel inherits the tag, and the tag picks its colour: its label. One bit tells two blobs apart: two per launch, about CPU speed. But only two seeds, each handed in.
```

- The VRAM close-up grows out of the four chips: lanes image, visited,
  queue. One queue entry opens into 32 bit cells, the last one the tag.
- One slow step: a dense load of 32 entries, one lane paints its pixel
  palette[tag], claims a neighbour, appends a new entry with the same
  bit. The panel folds back; two blobs fill at once ("max, not sum");
  ~1,368 ms (estimate). Two seeds run in over the PCIe link.
- Captions: "1 spare bit" / "per queue entry"; "max, not sum"; carried
  "2 seeds" / "per launch".

## beat r09_seeds

```text
So the GPU finds its own seeds: every blob in one fill launch, fifty-one milliseconds. But every red pixel still rides the queue.
```

- Seed sparks at corners and on a lattice; colliding waves keep the
  smaller label ("min"). The meter stops at 51 ms, over 33. The queue
  fills with 13,451,960 pixels.

## beat r10_runs

```text
Look at one row: in memory, it's one unbroken line, with its red pixels in spans. So link touching spans, not pixels: twenty-five times fewer. Warps read the image row by row: neighbouring threads, neighbouring bytes. No queue, no ring loop: six plain steps. About three milliseconds.
```

- The VRAM close-up again: the image lane is the marked row (red spans,
  white gaps), then a runs lane with five sub-cells per entry. The warp
  strip, labelled "1 warp", reads 32 neighbouring pixels in one packet.
- The panel folds back. Queue, gate and launches come off; six kernels
  go on. The meter crosses 33 ms at 2.96 ms (1.46 ms from a packed
  mask). Scoreboard row 3: "many blobs: GPU 35x" (1 Mpx, 47,655 blobs;
  two sessions).
- Captions: "25x fewer" / "runs than red pixels"; "coalesced" / "by
  design, not profiled"; carried "2.96 ms" / "from the RGB image".

## beat r11_reveal

```text
One small blob: the CPU. One huge blob, or many at once: the GPU. Every drone, every frame.
```

- The full scoreboard; then back out to the live drones ("drone frame:
  GPU 0.93 ms, CPU 3.1 ms, budget 33 ms"). Title card: "connected-component
  labelling / flood-fill-cuda".
