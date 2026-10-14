# Narration script: the chain cut, v2 ("size, then count")

Status 2026-10-14: approved by the user (the two-act ladder, 2.5D look,
the name on the title card only, "about three milliseconds" at the end,
the count row on the 1 Mpx noise frame, and a committed drone-frame
benchmark in the opening). The full design, with every visual and
source, is `narration/chain_v2_proposal.md`; the numbers and the claim
checks are `scenes/chain/numbers.py`. Version 1 of this cut (11 beats,
"you build a machine") is in git history and in `out/chain_v1/`.

## The idea

A drone show opens the video. On frames that small the compiled CPU
keeps up, but at 81 Mpx it needs 1.3 s against a 33 ms budget. So where
does the GPU win? Two places, and the video earns each one:

- **Size** (act 1): one giant blob. One block, two blocks, all 48, then
  diagonal steps, each a "But... So" rung on the 64 Mpx test square.
- **Count** (act 2): thousands of blobs at once. One launch per blob
  loses to the CPU, then tags, then seeds found on the GPU, then runs.

A three-row "CPU vs GPU" scoreboard fills in as the acts are won: one
small blob (the CPU), one huge blob (the GPU, by size), many blobs (the
GPU, by count). The ending answers the opening question with it.

Rules carried over from v1: every "So" fixes the "But" before it; every
failure is a visible event; amber means someone is waiting; the name
("connected-component labelling") is on the title card, never spoken;
no "chapter" anywhere.

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

- One worker crawls the drone, one cell per step; the 24 rooms ripple and
  stay dark. Captions "1 pixel", "1 core" (the i9 has 14), "24 SMs".

## beat r02_one_block

```text
So move it onto one SM: two hundred fifty-six threads paint a whole ring at once. But on one small blob, the CPU still wins. So: size first.
```

- Split floor: CPU one cell per step vs GPU one ring per step, in steps.
  Then the race in time on a 16 K px test blob: CPU 0.15 ms, best GPU
  0.37 ms. The scoreboard is born: row 1 "one small blob: CPU 2.4x".

## beat r03_giant

```text
On one giant blob, one block is over twice the CPU. But that's one SM of twenty-four.
```

- The drone swells into the 64 Mpx test square ("not a drone: a stress
  test"). One block sweeps its rings: 2.46x the CPU. Gauge "1 / 24 = 4%".

## beat r04_two_blocks

```text
So add a second block on a second SM: twice as fast. But twenty-two SMs stay dark.
```

- A second room lights, one shared queue, the gate at rest. "2x again"
  (5.0x the CPU). Gauge "2 / 24 = 8%".

## beat r05_all_blocks

```text
So use every SM: forty-eight blocks, eighteen times the CPU. But on a thin snake, they lose to one. Why? Every ring, the chip stops twice at a gate, two microseconds each.
```

- All rooms lit (48 x 256). "18x". Then the snake: 48 blocks 143 ms,
  2 blocks 128 ms, 1 block 62 ms. The gate flashes amber twice per ring;
  "65,792 stops".

## beat r06_diagonals

```text
So pass fewer gates: step diagonally too. On a square, half the rings, half the gates.
```

- The stencil goes from 4 to 8 arrows; the diamond becomes a square;
  8,001 -> 4,001 rings. Caption "25.8x". Scoreboard row 2: "one huge
  blob: GPU 25.8x".

## beat r07_real_frame

```text
Size is won. Now count: twenty-five hundred blobs in the big frame, one launch each. Slower than the CPU: we built for size, not count.
```

- Pull back to the 81 Mpx frame (2,522 blobs). A launch counter races;
  the meter swings past the CPU to ~2,181 ms (estimate).

## beat r08_tags

```text
So tag each queue entry with its blob: two per launch, about CPU speed. But tags run out at sixty-four, and every seed is handed in.
```

- Two blobs fill at once ("max, not sum"); ~1,368 ms (estimate). A rack
  "64 max" against 2,522; seeds drop in from outside.

## beat r09_seeds

```text
So the GPU finds its own seeds: every blob in one fill launch, fifty-one milliseconds. But every red pixel still rides the queue.
```

- Seed sparks at corners and on a lattice; colliding waves keep the
  smaller label ("min"). The meter stops at 51 ms, over 33. The queue
  fills with 13,451,960 pixels.

## beat r10_runs

```text
Look at one row: red pixels come in spans. So link touching spans, not pixels. No queue, no ring loop: six plain steps. About three milliseconds.
```

- One row's red cells slide into teal spans ("25x fewer"). Queue, gate
  and launches come off; six kernels go on. The meter crosses 33 ms at
  2.96 ms (1.46 ms from a packed mask). Scoreboard row 3: "many blobs:
  GPU 35x" (1 Mpx, 47,655 blobs; two sessions).

## beat r11_reveal

```text
One small blob: the CPU. One huge blob, or many at once: the GPU. Every drone, every frame.
```

- The full scoreboard; then back out to the live drones ("drone frame:
  GPU 0.93 ms, CPU 3.1 ms, budget 33 ms"). Title card: "connected-component
  labelling / flood-fill-cuda".
