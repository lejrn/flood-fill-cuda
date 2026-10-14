# The chain cut, v4: the memory story

Status: PROPOSAL for the user's approval, 2026-10-14, revised after review. Nothing is built.

Base: v3 (`script_chain.md`, `chain_v3_plan.md`). The two acts stay: size, then count. Every v3 rule stays: at most three captions per beat, a caption sub of at most four words, amber only for waiting, every number from committed JSON through `numbers.py`, "~" on estimates, "kernel time" where it applies, no "chapter".

The pacing request (25% more time between sentences and screen changes) is done. Commit f69fd78 set the sentence pause to 1.125 s, the colon pause to 0.56 s, BREATH to 0.75 s and MIN_DWELL to 1.75 s. Every length below is at that pace.

---

## The seven requests, and where they land

| # | The user asked (short) | Where in v4 | Verdict from the repo |
|---|---|---|---|
| 1 | One block on one SM first, queue in shared memory, which is small (48 KB per block) | r02 sets it up, r03_giant pays it off | Mostly true. The "a blob per SM later" premise is not (see Departures). |
| 2 | So the queue moves to global memory | r03_giant (the overflow spills), r04 (the whole queue) | True, in two steps, not one |
| 3 | Re-implement so new neighbours are written to global memory; works, can improve | r04 | True. It is the neighbour's 4-byte address that is written. |
| 4 | Chapters that changed the thread count until the optimum | r05, one sentence and a small grid card | Partly: one plateau, about 15% better on a smaller square, never adopted |
| 5 | A colour per block on the huge blob: 1 SM, 2 SMs, 24 SMs x 2 blocks | r03_giant, r04, r05, r05b, r06 | The colours mix (speckle); blocks own no region. "1 SM with 2 blocks" never ran on 64 Mpx. |
| 6 | Tags are part of the queue in memory; show the global-memory layer | r08, the new VRAM close-up | True for r08: one spare bit in each queue word |
| 7 | Show data moving between VRAM and the SMs; row scans, coalescing | act-1 packets (r03_giant, r04), r08's slow step, r10_runs | The row access is dense by design. It was never measured as the cause of the win. Rows never go into a queue. |

---

## The new chain at a glance

Sixteen beats (v3 had fourteen). Both new beats are split from a v3 beat. The first draft's separate r10_memory beat is folded into r10_runs (see "Why r10_memory is gone").

| k | beat | status | rung (But -> So) |
|---|---|---|---|
| 0 | i0_problem | same | |
| 1 | i1_compare | same | |
| 2 | r00_zoom | same | |
| 3 | r01_cpu | same | |
| 4 | r02_one_block | changed (point 1) | one SM, queue in shared memory. But the small blob: CPU wins. So: size. |
| 5 | r03_giant | changed (points 1, 2) | But the queue overflows shared memory. So it spills to global memory. |
| 6 | r03b_one_sm | new (v3 r03's tail) | Over twice the CPU. But one SM of 24. |
| 7 | r04_two_blocks | changed (points 2, 3, 5) | So a second block. But shared memory is private. So the whole queue goes global. But 22 SMs dark. |
| 8 | r05_all_blocks | changed (points 4, 5) | So every SM: 18x. Colours: the blocks share one queue. The launch-shape sweep. |
| 9 | r05b_snake | new (v3 r05's tail) | But the snake: 48 lose to 1, the gate. |
| 10 | r06_diagonals | same text | the floor is the 8-way owner mosaic; the cubes keep their block hues |
| 11 | r07_real_frame | same text | at the pull-back the cubes turn plain blue (act 2: one hue per blob) |
| 12 | r08_tags | changed (point 6) | So tags: one bit of each queue word in VRAM. One bit, two blobs. But two seeds, handed in. |
| 13 | r09_seeds | same text | |
| 14 | r10_runs | changed (point 7) | So link spans, not pixels: 25x fewer. The image is read row by row. 3 ms. |
| 15 | r11_reveal | same | |

Scene files: `r03_giant.py` splits into `r03_giant.py` and a new `r03b_one_sm.py`. `r05_all_blocks.py` splits into `r05_all_blocks.py` and a new `r05b_snake.py`.

---

## Shared devices (new or changed)

### Where the new panels sit

The finals use the iso look. Positions are frame units, checked against `mc.placement` for both layouts.

| panel | 9:16 centre, size (w x h) | 16:9 centre, size (w x h) | tied to |
|---|---|---|---|
| D1 gauge (r03_giant) | (-2.70, -2.70), 2.3 x 1.0 | (1.20, 1.85), 2.1 x 1.0 | leader line to SM 0 |
| D1 cards (r04), two stacked | (-2.70, -2.25) and (-2.70, -3.15), 2.3 x 0.8 each | (1.20, 2.25) and (1.20, 1.30), 2.1 x 0.85 each | leader lines to SM 0 and SM 2 |
| D2 close-up (r08, r10_runs) | (-2.33, -4.20), 3.0 x 4.6 | (1.50, 0.35), 2.7 x 4.8 | grows out of the four VRAM chips |

Why there:
- 9:16: the free band between the caption (its bottom near y = -1.65) and the CPU board's back edge, left of the GPU board's back-left corner (x = -1.43). SM 0 sits at (-0.53, -3.81) and VRAM chip 0 at (0.66, -5.38): both stay in view. D2 also covers the CPU board, which is dark there.
- 16:9: above the CPU board, right of the floor box (x = -0.1), left of SM 0 (x = 3.10), under the meter (bottom near y = 3.15) and over the caption (top near y = -3.1).
- The floor stays whole in both layouts, so r03's sweep and r10's marked row stay visible beside the panel.
- The CPU board is dark through act 1. In act 2, P-core 0 lights only after the panel folds (r08's seeds), and the kernels slide onto the link only after it folds (r10).
- Flat look: the same rule puts the panels over the CPU board (dark through act 1).
- Each panel is a raised dark card with a frame and a leader line, so it reads as a call-out from the GPU, not as part of the CPU. Tokens never cross the CPU board: they run back along the leader into the SM first.
- Check: `build/layout_check.py` with each panel open, both layouts, against the caption band and the meter.

### D1. The shelf gauge and the shelf cards (shared memory)

The board's shared-memory strip is 0.07 of an SM's depth: 2.4-2.7 px in the iso finals. Nothing that happens on it can be read. So shared memory gets its own small cards.

The gauge (r03_giant):
- One outlined box: a block's shared memory by default (48 KB). No text on it: caption 1 names it.
- Inside it, the queue region, outlined: two thirds of the box (8,192 slots x 4 B = 32 KB). The last third is hatched. The repo's ring is the largest power of two that fits.
- Under the box, a bar on the same scale: the queue's real load, two adjacent rings, `level_sizes[L] + level_sizes[L+1]` of CH03 `sq_8000_center`. The queue holds the ring being read and the ring being written.
- Toy ring i maps to real level 400 i. The bar moves through the real levels between toy rings, so it never jumps.
- The bar crosses the queue region's end (8,192 slots) exactly at real ring 1,024 (4,096 + 4,100 slots), when the first tokens pour. It peaks at 31,994 slots = 125 KB, 2.6 times the box, past even the 99 KB opt-in.
- Overflow: while the bar is past the region, red tokens pour out of the region's end. They run along the leader into SM 0, then SM 0 -> memory controller -> VRAM chip 0 (`slot_point`). That is real rings 1,024 to 6,975: toy rings 3 to 17.
- Loads too: on each later ring, some tokens come back up from VRAM to SM 0. ch01's v2 reads the ring's shared slots first, then its spill slice.

The cards (r04): the same card without the bar, one per SM, stacked. SM 0's card holds its queue region full of red tokens. SM 2's card shows an empty, outlined region. No text: the narration says "private".

On the board, `sm_tile` gets a third shelf state. The states are off, full (the queue's red) and empty (outlined, no fill). `Spec.shelf: bool` becomes `shelves: tuple`, one state per SM.

### D2. The VRAM close-up (the global-memory layer)

A new helper `vram_closeup(pl, centre, height, form)` in `machine.py`, the twin of `sm_closeup`. It is a flat dark panel. It grows out of the four VRAM chips and folds back into them.

- Lanes are horizontal strips of cells, one per array, in memory order. A cell's width is its bytes: image 3 B per pixel (`uint8` RGB), visited 4 B (`int32`), queue 4 B per entry (`int32`). The level's window [front, rear) is bracketed, new entries land at rear.
- Text: the lane names only. No title (the narration says "global memory" as it grows), no byte counts, no "L2". The die drawing already shows L2 (32 MB, as the device dump reports).
- Depth and owner also live in VRAM. They stay off the panel: they would add text and teach nothing new here.
- Form "tags" (r08): lanes image, visited, queue. A warp strip of 32 cells sits on the panel's SM edge, unlabelled.
- Form "runs" (r10_runs): the image lane redrawn as one memory row, then a runs lane. A runs entry has five `int32` sub-cells, the run table's five arrays (`run_x`, `run_y0`, `run_y1`, `parent`, `run_label`, `recolor.py:178-183`). The warp strip is labelled "1 warp" here, and only here.
- Never draw the queue living in L2. ch03 README:15 calls that a hypothesis, and ncu was never run.

### D3. Packets (loads and stores)

A packet is a bar that travels between memory and an SM: up for a load, down for a store.

- On the board (act 1): between an SM and the VRAM chips, through the memory controller.
- In the close-up: between a lane and the warp strip.
- Its length is the bytes moved. A dense read is one bar, fully lit: 32 queue entries = 128 B, or 32 pixels of one row = 96 B.
- v4 draws no sector packets ("32 B with 3 B used"). That is the derived model in `shared/bandwidth.py`, not a measurement, and it would need its own "a model" label.

### D4. Block hues and the owner map (act 1)

In act 1 the hue is the block. Act 2 goes back to "one hue per blob".

- Block 0 = BLUE (211 deg, the working blue). Block 1 = GREEN (128 deg). ch02's renders use block 0 blues, block 1 greens.
- Blocks 2-47: the golden-angle walk (137.508 deg, as ch03's `benchmarks/wavefront.py`), starting from BLUE's hue. It skips hues within about 12 deg of BLUE and GREEN, the red band (unpainted), the amber band (waiting) and the teal band (the final answer).
- Lightness is the ring: light near the seed, dark at the edge, as `level_colors` does now.
- The cubes on the SMs take their block's hue. They turn plain blue again at r07's pull-back, with the act change.
- One-block floors (r03_giant, the snake) stay the blue ring ramp: block 0's hue, so the rule holds there with no new raster.

The floor must show real ownership. No 64 Mpx owner map is committed, so v4 needs one new run (open question Q1):

- Three kernels on `sq_8000_center`, owner map on: ch02's `global` (2 x 256, `owner` int8), ch03's 4-way kernel (48 x 256) and ch03's 8-way kernel (48 x 256, for r06). The ch03 instrumented `flood_fill` already returns `owner` as int16 (`flood_fill.py:280`).
- Its level trace holds `min(W x H, 2**21)` levels (`flood_fill.py:265`), far above 8,001. The run must still assert `truncated` is false.
- Save point samples on the floor's grid: 21 x 21 toy cells x 6 x 6 samples each, so each drawn dot is one real pixel's owner. Also save two 64 x 64 full-resolution crops (at the seed, on the widest ring) for the optional lens.
- The sample stride is about 63 px, so any finer pattern turns to noise in the mosaic. The lens (Q4) is the only faithful close view.
- Owner positions change from run to run: the order inside a level is warp-aggregated discovery order. The cross-check verifies counts only: per-block counts must equal `processed_per_block` of 012243Z, which follows from `level_sizes` and the grid-stride rule.
- Suggested home: `chapters/ch03_gpu_1blob_nblock/benchmarks/owner_map.py`, writing to `results/ch03_gpu_1blob_nblock/owner_map/`.
- About 1 GB of host RAM and VRAM at 8000^2. Check `free -m` first: TTS (1.8 GB) and a render (1.1 GB) may be running.

Each painted toy cell is then a 6 x 6 mosaic of real owner dots. The toy look stays, and the speckle shows.

Fallback, if no run: the rule (entry j of a level goes to block (j mod 12,288) // 256), with the order inside a level shuffled by a fixed seed. The sub then says "schematic". The repo warns that any made-up order invents a pattern.

### D5. Measured placement

- Two blocks: block 1 lands on SM 2, not SM 1. DB2 `global_sm_ids` is [0, 2] on every ch02 scene. (CH03's `sm_ids` belong to the 48-block run, so they are not cited here.)
- 48 blocks: CH03 `sq_8000_center.sm_ids`. Blocks 0-11 land on SMs 0, 2, ..., 22, blocks 12-23 on SMs 1, 3, ..., 23. Blocks 24-47 repeat that, as each SM's second cube.
- `die.py` numbers SMs i = 8 g + 2 t + s, so even SMs are the back SM of each TPC. The die is a schematic, so this is a faithful reading, not a claim about the silicon.

### D6. Busy blocks per ring, and the gate

A cube is bright while its block has work in the current ring and dim while it has none.

- Rule: busy(L) = min(blocks, ceil(level_sizes[L] / 256)). Blocks 0 .. busy(L) - 1 are the busy ones.
- On the 64 Mpx square: rings 0-64 (256 px or fewer) are block 0's alone. All 48 blocks are busy on rings 3,073-4,927, 1,855 of the 8,001 rings.
- r06 uses CH03 `sq_8000_center.conn8_level_sizes` (4,001 rings).

How dim combines with amber:
- A block with no work in a ring really is waiting at `grid.sync`. Until r05b names the gate, v4 shows that as dim, not amber.
- The gate flash (`team_state="wait"`) turns every cube amber, dim ones included.
- From r05b's "gate" on, idle cubes are amber for the whole ring, not dim. That is the reveal: they were waiting all along. Block 0's cube is its blue while it works and amber at each flash with the others.
- Without this, the flash would show the working block 0 as the one waiting.

`Spec` gets two fields:
- `team_hues: tuple`: per SM, the hue of each resident cube (empty = all BLUE, as now).
- `team_busy: tuple`: per SM, each cube's state: "work", "dim" or "wait".
- `team_state="wait"` keeps its meaning (the gate flash) and overrides every cube.

### D7. The tag is one bit

On the board, the tag is a thin stripe on each queue token, about 15% of the token's depth: readable, without claiming a scale. v3 drew it as a cap over 45% of the slot (`machine.py:239-242`). "One bit of 32" is left to the close-up, where the entry opens into 32 bit cells: 31 for the pixel's address, the last one the tag.

---

## Beats (changed and new only)

Seconds come from the length model in "Length": real Kokoro word rates and the real `pace.py` stretch, plus the beat's tail.

### r02_one_block (k = 4), changed: point 1 set up

Rung: So, one SM, with its queue in the SM's shared memory. But on one small blob, the CPU still wins. So: size first.

```text
So move it onto one SM, its queue in the SM's shared memory: two hundred fifty-six threads paint a whole ring at once. But on one small blob, the CPU still wins. So: size first.
```

Visuals (only the change):
- The SM close-up is built with `shelf=True`. Its shared-memory bar holds the queue strip (red entries).
- At "queue" a few red tokens drop into the strip. At "shared memory" the bar gets the white outline pulse (the new-part convention).
- The close-up still flies back just before "paint", where the race starts. The new clause sits before "paint", so the close-up gains about 1.5 s on screen.
- The close-up leaves SM 0's shelf "full" on the board: the queue lives there from now on.

Captions (unchanged): "256 threads"; carried "CPU wins / one small blob".

State at the cut (k = 5): v3's boundary(5) with SM 0's shelf "full".

Numbers: 256 threads (`OV:tpb`, existing). Nothing new.

Length: 35 words, 15.2 s (v3 12.5 s): +2.7 s.

### r03_giant (k = 5), changed: points 1 and 2 (first step)

Rung: But the queue overflows shared memory. So the overflow spills to global memory.

```text
On one giant blob, that queue overflows: a block gets only forty-eight kilobytes of shared memory, and this queue needs over a hundred. So the overflow spills to global memory: the card's eight gigabytes.
```

Visuals:
- "giant": the swell into the 64 Mpx test square, P-core 0 goes dark, the inset reads "stress test", SIZE tag (all as v3).
- "overflows": the gauge (D1) grows out of SM 0 to its place, on its leader line. The block's sweep starts, ring by ring, with SM 0's fence flashing amber after each ring (as v3). The floor is the blue ring ramp: block 0's hue.
- The sweep is re-anchored to this beat's words: from "overflows" to "gigabytes" (it was "block" to "twenty-four", which moves to r03b).
- From toy ring 3 the bar passes the queue region and red tokens pour toward VRAM. Later rings also bring tokens back up (D1).
- "forty-eight kilobytes": the box outline pulses. Caption 1.
- "over a hundred": caption 2. The bar is near its peak, far past the box.
- "spills": the tokens are landing in VRAM chip 0's slots. Caption 3. No board label for the spill: the caption names it.
- "eight gigabytes": the four VRAM chips get the white outline pulse.
- The sweep ends at about the end of "gigabytes". The gauge folds back into SM 0, which keeps its shelf "full".

Captions (3):
1. "48 KB" / "per block, by default", at "forty-eight".
2. "125 KB" / "two rings at peak", at "hundred".
3. Carried: "spill" / "to global memory", at "spills".

State at the cut (k = 6): floor `square4` painted; SM 0 lit with one cube, shelf "full", `room_fence="rest"`; new `spill=True` (a few tokens in VRAM chip 0's slots, no label); inset "stress test"; tag SIZE.

Numbers:
- 48 KB: device dump, `.github/copilot-instructions.md` ("Max Shared Memory per Block: 49152 bytes (48.0 KB)"); `ch01 kernels.py:37-39`. A constant in `numbers.py` with that source, like `CPU_CORES`.
- 125 KB: the two-ring peak, max(`level_sizes[L] + level_sizes[L+1]`) = 31,994 slots x 4 B / 1,024, CH03 `sq_8000_center`. CH01 `sq_6000_center.peak_occupancy` 23,994 = 11,998 + 11,996 supports the two-ring rule.
- The overflow from real ring 1,024: the first L with a two-ring load above 8,192 (`level_sizes`).
- Queue region 8,192 slots: CH01 `sq_6000_center.ring_capacity`.
- 8 GB: the same device dump ("Total Global Memory: 8585216000 bytes (8.00 GB)"). A constant with that source.

Length: 34 words, 13.7 s (v3 r03 was 7.1 s, now split): +6.6 s.

### r03b_one_sm (k = 6), new: v3 r03's tail

Rung: over twice the CPU. But that's one SM of twenty-four.

```text
Now one block is over twice the CPU. But that's one SM of twenty-four.
```

Visuals: the painted square holds. At "twice", the caption. At "SM", the carried caption. At "twenty-four" the 23 other SMs come up idle in a wave and go dark (moved from v3 r03: `ripple`).

Captions (2): "2.46x" / "vs the CPU, kernel time"; carried "1 of 24 SMs".

State at the cut (k = 7): k = 6 with the caption "1 of 24 SMs".

Numbers: 2.46x = `big_cpu_ms / big_1block_ms` (CH03, existing). This one-block run is ch01's v2 kernel: the shared ring plus the global spill, as r03_giant now shows.

Length: 14 words, 6.4 s, all new as a beat. It replaces 16 words of v3 r03. `base.py`'s 0.8 s HOLD moves here from r03_giant.

### r04_two_blocks (k = 7), changed: points 2, 3, 5

Rung: So a second block. But shared memory is private to its block. So the whole queue moves to global memory. Twice as fast. But 22 SMs stay dark.

```text
So add a second block on a second SM. But shared memory is private to its block. So the whole queue moves to global memory: every new pixel's address is written there. On big blobs, measured, that's no slower. Two blocks are twice as fast. But twenty-two SMs stay dark.
```

Visuals:
- "So": the carried caption leaves, the square goes back to red (as v3).
- "second block": block 1's cube (GREEN) drops onto SM 2 (D5). SM 2 lights at "second SM".
- "private": the two shelf cards (D1) grow out of SM 0 and SM 2, on their leader lines. SM 0's is full of red tokens, SM 2's is empty. SM 2's tile shows its shelf "empty". No text: the narration carries it.
- "the whole queue moves": SM 0's tokens drain down its leader into SM 0, then into VRAM, and join the spill tokens. Both cards fold back. SM 0's shelf goes "off", the fence goes, the board's label "queue" appears under the VRAM chips and the gate rail appears at rest (both as v3).
- "every new pixel's address": the refill, on r04's own plan (below). The floor is the 2-block owner mosaic (D4): blue and green dots, mixed, never two halves.
- Each ring step: the ring's entries come up from the VRAM queue, one packet to SM 0 and one to SM 2 (loads); a blue packet from SM 0 and a green one from SM 2 go down to the queue's rear (stores).
- The first toy ring (real rings 0-64, 256 px or fewer) is block 0's alone: the green cube stays dim until its block has work (D6).
- "slower": caption 1. "twice": caption 2.
- "twenty-two": the 22 dark SMs ripple idle, then go dark at "dark" (as v3).
- v3's `block_dots` (left half / right half) is removed. That was the `split` kernel's picture, while the time shown is the `global` kernel's.

r04's own plan: today `r04_two_blocks.py:26` replays `r03_giant.r03_plan()` at twice the pace, and `r03_plan()` times itself from r03's words "block" to "twenty-four" (`r03_giant.py:35-46`). v4 moves "twenty-four" to r03b, so that plan would silently fall back to 3 s. v4 gives r04 its own `r04_plan()`, timed from its words "address" to "fast", through the same `sweep_plan`.

Captions (3):
1. "no slower" / "16 Mpx square", at "slower".
2. "2x again" / "kernel time", at "twice".
3. Carried: "22 SMs idle".

Spacing: "slower" to "twice" now spans a sentence end (1.125 s) and three words, about 2.4 s at Kokoro's real 3.11 words per second: over MIN_DWELL (1.75 s). Re-check it from the new `words.json`. If it comes out under MIN_DWELL, cue caption 2 at "fast".

State at the cut (k = 8): floor `square_own2` (the 2-block mosaic, painted); SMs 0 and 2 lit, one cube each (blue, green); `queue=True` with tokens; gate at rest; no shelf, no fence, no spill.

Numbers:
- "a second SM" = SM 2: DB2 (`dual_block_20260719T163033Z.json`) `scenes[sq_6000_center].global_sm_ids` = [0, 2], and the same on every scene.
- "On big blobs, measured, that's no slower": one block, same session, the 16 Mpx corner square (CH03): shared ring `sq_4000_corner.v2_kernel_ms` 270.95 ms vs global queue `block_tpb_sweep[sq_4000_corner, conn 4, 1 x 256].kernel_ms` 253.23 ms.
  - The shared ring never spilled there: CH01 `sq_4000_corner.peak_occupancy` 7,999 of 8,192, `spilled_px` 0. So it is a clean shared vs global test.
  - It also holds on the 11 Mpx disk (191.93 vs 176.34 ms), but v2 spills there (its widest level is 7,600 px), so it is not cited.
  - DB2's two-block pair (`split` 314.33 vs `global` 280.26 ms, 36 Mpx) points the same way. `split` also differs in partition, registers and barriers, so it is a note, not the evidence.
  - Why "on big blobs": on the 33 K px snake the one-block global queue takes 119.50 ms and the shared ring 61.71 ms (CH03, same session). The barrier causes that (a grid-wide sync vs the block's own), not where the queue lives.
- "twice as fast": `big_2v1_x` = 1,052.0 / 516.7 = 2.04 (CH03, existing).
- Mosaic: the new owner-map run (D4).

Length: 50 words, 21.5 s (v3 7.3 s): +14.2 s.

### r05_all_blocks (k = 8), changed: points 4 and 5

Rung: So every SM: 18x. Each colour is a block; they share one queue. Aside: the launch-shape sweep, and the default we kept.

```text
So use every SM: forty-eight blocks, two on each. Eighteen times the CPU. Each colour is one block: they share one queue, so their pixels mix, and small rings keep only a few busy. We also tried other threads per block and block counts: on a smaller square, the best gained fifteen percent. We kept forty-eight by two hundred fifty-six.
```

Visuals:
- "every SM": the 22 dark SMs light in a ripple out from SMs 0 and 2 (as v3, new start).
- "forty-eight blocks, two on each": 48 cubes land in the measured order (D5), each in its hue: first one cube on each even SM, then each odd SM, then the second cubes.
- "Eighteen": the square refills in one quick pass with the 48-block mosaic (D4). Caption 1.
- "Each colour is one block": the pass replays from the seed out, in two speeds. Slowly over the first rings, where few blocks have work, until "busy". Then faster through the widest band (all 48 bright) and on, as blocks drop out from the top while the rings shrink toward the corners (D6).
- "busy": caption 2, while the replay still shows the small rings.
- Optional: a round lens over the widest ring shows one 64 x 64 crop of the real owner map at full resolution. It is the evidence for "their pixels mix" (Q4).
- "We also tried": a small card (where v3's snake times panel sits) shows the sweep grid, described below.
- "fifteen percent": caption 3.
- "We kept": the default cell's outline pulses. The card leaves with the beat's end (extra).

The sweep grid card:
- Columns: threads per block 32, 64, 128, 256, 512. Rows: blocks 1, 2, 4, 8, 16, 32, 48, 96, 128, 192, max. 16 Mpx corner square, 4-way, kernel time.
- No tick labels and no numbers: two axis words only, "threads per block" and "blocks". Caption 3 carries the number.
- Cells beyond the cooperative cap (12,288 threads) are empty, not slow.
- A neutral grey ramp in 4-5 bins (blue already means block 0 and "working"). The top three cells (318.8, 316.3, 315.6 Mpx/s) share the top bin: a flat top, never one glowing cell.
- The default 48 x 256 is outlined.

Captions (3):
1. "18x" / "kernel time", at "Eighteen".
2. "few busy" / "on small rings", at "busy".
3. Carried: "+15% at best" / "16 Mpx square", at "fifteen".

State at the cut (k = 9): floor `square_own48` (painted); all SMs lit, two cubes each in block hues, all "work"; queue, gate at rest.

Numbers:
- 48 blocks of 256, two per SM: CH03 `sq_8000_center.multi_blocks`, `.multi_tpb` and `sm_ids` (every SM id appears twice).
- 18x: `big_48_x` (existing).
- Rings 0-64; rings 3,073-4,927: derived from CH03 `sq_8000_center.level_sizes`.
- +15%: CH03 `block_tpb_sweep`, scene `sq_4000_corner`, connectivity 4: best 128 x 32 at 50.19 ms (318.8 Mpx/s) vs the default 48 x 256 at 57.96 ms (276.1), ratio 1.155. On the 4-way disk the best gains 7% (477.1 vs 445.5); on the 8-way disk the default is the best cell.
- "We kept": the ch03-ch05 defaults stay at 256 threads and the cooperative maximum of 48 blocks; the sweep's best shape was never adopted.

Length: 60 words, 24.7 s. v3 r05 was 13.9 s for this text and the snake together; with r05b the pair is +20.9 s.

### r05b_snake (k = 9), new: v3 r05's tail

Rung: But on a thin snake, 48 blocks lose to 1. Why? The gate.

```text
But on a thin snake, forty-eight blocks lose to one. Why? Every ring, the chip stops twice at a gate, two microseconds each.
```

Visuals: v3 r05's second half (the snake, the three times panel, the amber gate twice per ring, the "grid.sync" tag). Changes:
- On a snake every ring is one pixel, so only block 0 ever has work. While the snake fills, block 0's cube on SM 0 stays bright and the 47 others are dim.
- From "gate" on, the 47 idle cubes turn amber and stay amber: they wait at the gate every ring (D6). Block 0's cube turns amber only at each flash, with the rest.
- The times panel's one-block row reads "1, shared queue". That run is ch01's shared-ring kernel with its own SM fence, not the global queue r04 moved to. Without the label, the row contradicts r04.

Captions (2, as v3): "48 lose to 1"; carried "65,792 stops / ~2 µs each".

State at the cut (k = 10): v3's boundary(8), with the cubes in block hues, all "work", the gate at rest.

Numbers: as v3 (`snake_*`, `snake_stops`, `us_per_stop`). "Only block 0 works": ch03 wavefront.py's docstring ("the serpentine's 1-pixel frontier hands block 0 every single pixel"), and D6's rule with ring size 1.

Length: 23 words, 10.1 s (this text was in v3 r05).

### r06_diagonals (k = 10), same text

Only the floor changes. The 8-way refill paints the 8-way owner mosaic `square_own48_8` (D4, from the Q1 run), not the single-blue `level_colors`. Under the act-1 rule a blue floor would read as "block 0 painted everything". The cubes keep their hues and follow D6 with `conn8_level_sizes`.

If Q1 is declined, the same mosaic comes from the schematic fallback, labelled the same way as r04's and r05's.

### r08_tags (k = 12), changed: point 6

Rung: So tag each queue entry with its blob. One bit tells two blobs apart. But only two seeds, each handed in.

```text
So tag each queue entry with its blob: one spare bit of the entry's word, in global memory. Each new pixel inherits the tag, and the tag picks its colour: its label. One bit tells two blobs apart: two per launch, about CPU speed. But only two seeds, each handed in.
```

Visuals:
- "So tag each queue entry": entries land in the VRAM queue, the floor becomes the two blobs (as v3). The tag stripes are thin (D7). The VRAM close-up (D2, form "tags") grows out of the VRAM chips at its place, with three lane names: image, visited, queue.
- "one spare bit": one queue entry opens into its 32 bit cells: 31 grey-blue cells and one blue tag cell. No labels: caption 1 names it.
- "in global memory": the panel's frame pulses.
- "inherits the tag": one step, played slowly, on the panel:
  1. 32 neighbouring entries come up in one packet to the warp strip (a dense load). The other 31 lanes fade to faint lines toward their own pixels, spread along the image lane (schematic positions, no number attached). The camera follows one lane.
  2. It paints its pixel in the image lane with palette[tag], blue (packet down).
  3. A neighbour's visited cell flips 0 -> 1 (the claim).
  4. A new entry with the same tag bit is appended at the queue's rear (packet down).
- "picks its colour: its label": the tag cell and the painted cell pulse together. There is no label array: the paint is the label map.
- The panel folds back into the VRAM chips. Both blobs fill at once (as v3).
- "One bit tells two blobs apart": the tag cell flashes its two values, blue and green.
- "Two per launch": caption 2 ("max, not sum"); the needle settles at ~1,368 ms; the launch pill drops to 1,261 (as v3).
- "But only two seeds, each handed in": seed dots run from P-core 0 over the PCIe link, two per launch (v3 ran six in a row).
- Removed: the "tags run out at sixty-four" flash and the "64 tags max" caption.

Text timing: the lane names land at "So tag", caption 1 at "bit", about 3 s later.

Captions (3):
1. "1 spare bit" / "per queue entry", at "bit".
2. "max, not sum", at "launch" (as v3).
3. Carried: "2 seeds" / "per launch", at "seeds".

State at the cut (k = 13): v3's boundary(11), with the carried caption "2 seeds / per launch" and stripe tags.

Numbers:
- One tag bit: `ch04 kernels.py:29-45` (the `lin` format: entry = (x * height + y) << 1 | label), the default `entry_format` in `flood_fill.py:262-264`. A code fact, not a measurement. The ~1,368 ms estimate ran with this default.
- Two seeds: `ch04 flood_fill.py:290-294` raises unless given exactly two. Also `est_ch04_calls * 2 == n_blobs` (existing fact check).
- ~1,368 ms and 1,261 launches: OV `png_blobs.ch04_multi` (existing, estimate).
- Lane widths: the dtypes in `ch04 flood_fill.py:354-388`.
- The queue read: `ch03 kernels.py:155-157` (thread tid reads queue[front + tid]), so a warp's 32 entries are adjacent. `PARALLEL_BFS_DESIGN_ANALYSIS.md:64`: "the queue reads are already coalesced (good), but frontier pixels scatter across the image".

Length: 51 words, 21.1 s (v3 9.8 s): +11.2 s.

### r09_seeds (k = 13), same text

Only its start state changes (the caption "2 seeds"). "So the GPU finds its own seeds: every blob in one fill launch" now answers r08's new But directly.

### r10_runs (k = 14), changed: point 7

Rung: So link touching spans, not pixels: 25x fewer. The image is read row by row. About three milliseconds.

```text
Look at one row: in memory, it's one unbroken line, with its red pixels in spans. So link touching spans, not pixels: twenty-five times fewer. Warps read the image row by row: neighbouring threads, neighbouring bytes. No queue, no ring loop: six plain steps. About three milliseconds.
```

Visuals:
- "Look at one row": the floor marks one row (as v3). From r07 on the frame is drawn transposed, so a floor row is a memory row. The VRAM close-up (D2, form "runs") grows out of the VRAM chips at its place. Its image lane is that one row: red spans and white gaps.
- "So link touching spans": the row's red cells slide into teal spans (as v3). A runs lane appears: one entry per span, five sub-cells. Caption 1 at "fewer".
- "Warps read the image row by row": the warp strip, now labelled "1 warp" (at the second "row"), reads 32 neighbouring pixels of the row: one packet, 96 B, every byte lit. The spans it finds go down into the runs lane. This merges ch06's pack and emit into one read and one write; the six kernels come next, on the board.
- "neighbouring bytes": caption 2.
- The panel folds back into the VRAM chips. Then as v3: the queue leaves VRAM, the gate and the launch pill go, the six kernels slide in and light one by one, the needle crosses 33 ms to 2.96 ms, the strip's last pill fills.

Captions (3):
1. "25x fewer" / "runs than red pixels", at "fewer" (as v3).
2. "coalesced" / "by design, not profiled", at "bytes".
3. Carried: "2.96 ms" / "from the RGB image", at "milliseconds" (as v3).

Text timing: lane names at "Look", the runs lane at "link", caption 1 at "fewer", "1 warp" at the second "row", caption 2 at "bytes". At Kokoro's rate each lands about 2.5-2.8 s after the one before: over MIN_DWELL. "1 warp" waits for "row" because "Warps" comes only about 1.4 s after "fewer".

State at the cut (k = 15): v3's boundary(13). The close-up is closed.

Numbers:
- 96 B per warp: `ch06 kernels.py:231-266` (pack: one warp per 32 pixels of a row). One warp per row in count and emit: `ch06 README:106-145`.
- The five run arrays: `recolor.py:178-183`.
- 25x: `runs_x` = 13,451,960 / 539,207 = 24.95 (R61, existing).
- 2.96 ms: `ch06_rgb_ms` (R61, existing).

Length: 47 words, 21.7 s (v3 13.9 s): +7.9 s.

### Why r10_memory is gone

The first draft had a beat between r09 and r10_runs: one wide queue fetch, "But the entries point at scattered pixels: thirty-two separate reads", then r10_runs's "So read rows". That made scatter the problem and the row read the fix.

The repo does not support that order. It credits ch06's win to 24.95x fewer items, the 1-bit mask and no levels or barriers. It retracted "the BFS is DRAM-bound" (`chapters/README.md:256-268`) and never measured coalescing.

So v4 keeps the pictures and drops the claim:
- the dense queue load and the scattered pixel trips are in r08's slow step, with no "But" on them;
- the dense row read is in r10_runs, after a So that is the count;
- r09's But ("every red pixel still rides the queue") leads straight into r10_runs's So.

It also saves about 9.6 s. A description-only r10_memory, with no But, is open question Q8.

---

## Boundary table (`state.py`)

`BEATS` becomes:

```text
i0_problem, i1_compare, r00_zoom, r01_cpu, r02_one_block, r03_giant, r03b_one_sm,
r04_two_blocks, r05_all_blocks, r05b_snake, r06_diagonals, r07_real_frame, r08_tags,
r09_seeds, r10_runs, r11_reveal
```

| k | the state when this beat starts | from v3 |
|---|---|---|
| 5 | r02's end, SM 0 shelf "full" | boundary(5) + shelf |
| 6 | `square4` painted, SM 0 shelf "full" + fence, `spill=True`, cap "spill" | new |
| 7 | k = 6, cap "1 of 24 SMs" | boundary(6) + spill |
| 8 | `square_own2`, SMs 0 and 2, cubes blue and green, queue, gate rest, cap "22 SMs idle" | boundary(7), SM 2 and hues |
| 9 | `square_own48`, all SMs, cubes in hues, cap "+15% at best" | new |
| 10 | the snake, cubes in hues (all "work"), cap "65,792 stops" | boundary(8) + hues |
| 11 | `square_own48_8`, row 2, cap "25.8x", cubes in hues | boundary(9), new floor + hues |
| 12 | the frame, launches 2,522, ~2,181 ms; cubes plain blue | boundary(10) |
| 13 | two blobs, cap "2 seeds / per launch", stripe tags | boundary(11), new cap |
| 14 | the labelled frame, 51.0 ms | boundary(12) |
| 15 | the teal frame, 2.96 ms, all three pills | boundary(13) |

New fields:
- `mc.Spec`: `spill: bool`; `shelves: tuple` (replaces `shelf: bool`; per SM "off", "full" or "empty"); `team_hues: tuple` and `team_busy: tuple` (D6); a stripe style for `tag_tokens` (D7).
- No new `Boundary` field: every close-up opens and closes inside its beat.
- `floor_raster`: `square_own2`, `square_own48`, `square_own48_8` (the mosaics, from the owner-map run).

Other files that list beats:
- `base.py`: HOLD keys (r03_giant's 0.8 s hold moves to r03b_one_sm).
- `build/assemble.py`: `CHAIN` gains `("chain/r03b_one_sm", "OneSm", "r03b_one_sm")` and `("chain/r05b_snake", "Snake", "r05b_snake")`.
- `build/chain_snapcheck.py` takes a stem and a class, so its code does not change. The check list does: 13 snapping clips instead of 11, with the two new ones and r04, whose plan changed.
- `HANDOFF.md`, `script_chain.md`.

Each scene's `k` shifts as the table shows.

---

## `numbers.py` additions and fact checks

New sessions:
- DB2 = `ch02_gpu_1blob_2block/benchmark_results/dual_block_20260719T163033Z.json`.
- OWN = the new owner-map result (D4), if approved.

New keys (file and key in each `Num.source`):

| key | value | source |
|---|---|---|
| `ring_slots` | 8,192 | CH01 `sq_6000_center.ring_capacity` |
| `shared_kb` | 48 | constant: device dump in `.github/copilot-instructions.md`; `ch01 kernels.py:37-39` |
| `vram_gb` | 8 | constant: the same device dump |
| `big_level_sizes` | list (8,001) | CH03 `sq_8000_center.level_sizes` |
| `big8_level_sizes` | list (4,001) | CH03 `sq_8000_center.conn8_level_sizes` |
| `big_sm_ids` | list (48) | CH03 `sq_8000_center.sm_ids` |
| `big_blocks`, `big_tpb` | 48, 256 | CH03 `sq_8000_center.multi_blocks`, `.multi_tpb` |
| `two_sm_ids` | [0, 2] | DB2 `scenes[sq_6000_center].global_sm_ids` |
| `q1_shared_ms`, `q1_global_ms` | 270.95, 253.23 | CH03 `sq_4000_corner.v2_kernel_ms`; `block_tpb_sweep` (sq_4000_corner, 4, 1, 256) `.kernel_ms` |
| `q1_ring_peak` | 7,999 | CH01 `sq_4000_corner.peak_occupancy` |
| `sweep_best_ms`, `sweep_default_ms` | 50.19, 57.96 | CH03 `block_tpb_sweep`: min `kernel_ms` on (sq_4000_corner, 4); the (48, 256) cell |

Derived: `two_ring_peak` = 31,994; `two_ring_kb` = 31,994 x 4 / 1,024 = 125.0; `first_spill_ring` = 1,024; `last_spill_ring` = 6,975; `busy_all_rings` = 1,855 (rings 3,073-4,927); `block0_only_rings` = 65; `sweep_x` = 57.96 / 50.19 = 1.155.

New entries in `facts()`:
- r03 "forty-eight kilobytes": `shared_kb == 48`.
- r03 "that queue overflows": `two_ring_peak > ring_slots`.
- r03 "this queue needs over a hundred": `100 < two_ring_kb < 128`.
- r03 gauge: the bar's crossing and the first token are one ring: the first L with `level_sizes[L] + level_sizes[L+1] > ring_slots` is `first_spill_ring`.
- r03 "the card's eight gigabytes": `vram_gb == 8`.
- r04 "a second SM": `two_sm_ids == [0, 2]`.
- r04 "On big blobs, measured, that's no slower": `q1_global_ms <= q1_shared_ms` on scene `sq_4000_corner` (pinned by name), and `q1_ring_peak <= ring_slots` (the shared run never spilled).
- r05 "two on each": every SM id 0-23 appears exactly twice in `big_sm_ids`.
- r05 "small rings keep only a few busy": `block0_only_rings == 65` (`level_sizes[0:65]` all <= 256).
- r05 "fifteen percent": `round(100 * (sweep_x - 1)) in {15, 16}`.
- r05 "We kept forty-eight by two hundred fifty-six": `big_blocks == 48 and big_tpb == 256`.

---

## Length

The first draft's model counted the pauses twice: its 2.6 words per second already held the paced pauses, and it added them again. It put v3 at 178.9 s; the real v3 render is 138.7 s at the old pace. This model is built from the real takes.

Model, per beat:
- Speech: words / 3.11 (Kokoro `am_adam`, unpaced: 316 words in 101.6 s, `out/kokoro_chain`).
- Pauses: + 0.77 s per inner sentence end, + 0.41 s per inner colon. These are the real `pace.py` stretches at 1.125 / 0.56 s, measured on the v3 takes (25.9 s over 24 sentence ends and 18 colons).
- Tail: BREATH 0.75 s or the beat's HOLD, + 0.4 s.
- Check: the same sum over the real v3 takes at the old pace gives 138.6 s, against the render's 138.7 s. At the current pace v3 is 147.0 s.
- Adam (ElevenLabs, 11 of 14 v3 takes in `out/adam_chain_raw`) is slower: 2.24 words per second unpaced, + 0.64 s per sentence end, + 0.15 s per colon. v3 with Adam at the current pace: about 178.7 s (11 takes measured, 3 modelled).

| beat | words | v3 now, s | v4 Kokoro, s | change | v4 Adam, s |
|---|---|---|---|---|---|
| r02_one_block | 35 | 12.5 | 15.2 | +2.7 | 18.8 |
| r03_giant | 34 | 7.1 | 13.7 | +6.6 | 17.3 |
| r03b_one_sm | 14 | | 6.4 | +6.4 | 8.0 |
| r04_two_blocks | 50 | 7.3 | 21.5 | +14.2 | 26.8 |
| r05_all_blocks | 60 | 13.9 | 24.7 | +10.8 | 31.0 |
| r05b_snake | 23 | | 10.1 | +10.1 | 12.7 |
| r08_tags | 51 | 9.8 | 21.1 | +11.2 | 26.3 |
| r10_runs | 47 | 13.9 | 21.7 | +7.9 | 26.1 |
| **whole cut** | 484 (v3: 316) | **147.0** | **217.0** | **+70.0 (+48%)** | **267.3** (v3: 178.7) |

r05b_snake is mostly v3 text in its own beat: r05 and r05b together add +20.9 s.

By request: points 1-2 (r02, r03_giant, r03b) +15.7 s; points 2-3 (r04) +14.2 s; points 4-5 (r05, r05b) +20.9 s; point 6 (r08) +11.2 s; point 7 (r10_runs) +7.9 s. The act-1 packets and r08's dense load cost no words. The two extra beats add about 2.3 s of tails.

Cuts that keep every request on screen:

| cut | saves (Kokoro) |
|---|---|
| r04: drop "On big blobs, measured, that's no slower." Caption 1's slot is freed. | 3.0 s |
| r03_giant: drop "the card's eight gigabytes" | 1.7 s |
| r05: one sweep sentence: "Other threads per block and block counts gained fifteen percent at best, on a smaller square; we kept forty-eight by two hundred fifty-six." | 1.7 s |

All three save about 6.4 s: about 211 s with Kokoro. Folding r03b back into r03_giant saves under 1 s and would need two of its captions dropped, so it is not on the list. Do not cut "every new pixel's address is written there": it is the user's point 3.

No set of cuts that keeps every request gets this cut under three minutes (Q6).

---

## Where v4 departs from the user's wording

1. **"Assume later we can give each SM its own blob."** No era of the repo planned that. The June 2025 roadmap (git `00995f5:README.md`, lines 36-67) goes from one blob on one block to one blob on many blocks with a global queue. CUDA also cannot place a block on a chosen SM ("You cannot choose which SM a block runs on", ch02 README:74). v4 keeps the real next step: more blocks on the same blob ("one SM of twenty-four" -> "add a second block").
2. **"At most 48 KB per block."** 48 KB is the default static limit. The opt-in limit is 99 KB, and an SM has 100 KB (device dump). The ring used 32 KB (8,192 slots, the largest power of two). v4 says "a block gets only forty-eight kilobytes", with the sub "per block, by default", and draws the queue as two thirds of the box. Even 99 KB would not hold the 64 Mpx square: its two-ring load peaks at 31,994 slots, 125 KB.
3. **"It works."** The pure shared-memory kernel refuses centre-seeded squares wider than about 2,048 px. It never ran the 64 Mpx square. v3's "one block, 2.46x" was ch01's v2: shared ring plus global spill. v4 shows that order: overflow, spill, then 2.46x.
4. **"Shared memory is very fast."** The repo never measured this. Its measured lesson is the opposite: "The shared-memory queue didn't pay" (chapters/README.md:144-149). v4 does not say "fast". r04 says "On big blobs, measured, that's no slower", pinned to the 16 Mpx square where the shared ring never spilled (270.95 vs 253.23 ms). On the thin snake the one-block global queue is 1.9x slower (119.50 vs 61.71 ms), because of its grid-wide barrier, not where the queue lives. So the line is scoped to big blobs, and r05b's one-block row says "shared queue".
5. **"Therefore the queue moves to global memory" (one step).** It took two. ch01 v2 spills only the overflow, still with one block. ch02 moves the whole queue, for two reasons other than size: shared memory is private to its block, and the global queue measured fastest. v4 tells both steps (r03_giant, r04).
6. **"Neighbouring pixels are written to global memory."** What is written is the neighbour's 4-byte address (x * height + y). The image, visited and depth lived in global memory from the first GPU kernel. v4 says "every new pixel's address".
7. **"We changed the thread count until we found the optimum."** ch01 and ch02 found no optimum inside the range: more threads were faster up to the cap (1,024 threads; 512 for ch02's registers). Only ch03 found a best cell, on a plateau, about 15% above the default on a 16 Mpx square. It was never adopted, and it never ran on the 64 Mpx square. v4 says "We also tried other threads per block and block counts", "on a smaller square, the best gained fifteen percent" and "We kept forty-eight by two hundred fifty-six". The grid card has a flat top.
8. **"A colour per block shows how 1 SM with 2 blocks, 2 SMs, and 24 SMs x 2 blocks fill the blob."** The blocks share one queue and take 256-entry turns down it, so their pixels mix. No block owns a region. "One SM with two blocks" never ran on the 64 Mpx square. The scheduler spread two blocks onto SMs 0 and 2 every time. The only same-SM pair is ch02's pinned run on 36 Mpx (see Q2). v4 shows the measured ladder: 1 block on 1 SM, 2 blocks on 2 SMs, 48 blocks two per SM, and which blocks have work on each ring.
9. **"Tags are part of the queue in memory."** True for r08 (ch04): one bit of each 32-bit queue word. v4 also drops "tags run out at sixty-four". The kernel takes exactly two seeds, and 64 is only the capacity of the wider `xy` format, never exercised. There is no label array: the paint is the label map.
10. **"Scanning rows of the blobs and storing them in the queue."** ch06 has no queue and no BFS. Runs go into a run table of five `int32` arrays (`recolor.py:178-183`). Also, ch06 does not read only rows instead of pixels: pack reads every pixel (243 MB). v4 says "Warps read the image row by row".
11. **"Memory coalescing was one of the biggest improvements."** Coalescing was never measured: no ncu profile and no coalesced vs uncoalesced test. The repo credits ch06's win to the unit of work (24.95x fewer runs than red pixels), the 1-bit mask (243 MB -> 10.15 MB) and removing levels and barriers. The BFS queue reads were already coalesced; the scatter was in the pixel accesses. The repo also retracted "the BFS was DRAM-bound" after the 8-way test (chapters/README.md:256-263).
    - So r10_runs's So is the count ("twenty-five times fewer"), and the row read comes after it as a design fact, with the caption "coalesced" / "by design, not profiled".
    - No "But" sits on the scattered reads anywhere in v4.
    - "Fewer transfers" is true in ch06 only as fewer items and fewer bytes. Copies between host and device are a different kind of transfer and do not appear here.
12. **Where the word may come from.** The retired `graveyard/multi-blocks/` prototype lists "Coalesced global memory access" (`PR_DESCRIPTION.md:92`). Those kernels used `syncthreads` as a grid barrier and were wrong whenever the grid had more than one block (`PARALLEL_BFS_DESIGN_ANALYSIS.md:70-71`). v4 cites nothing from it.

---

## Problems in v3 that v4 fixes

- r04 painted each ring as block 0 = left half, block 1 = right half (`block_dots`). That is the `split` kernel's picture, while its "2x" is the `global` kernel's time.
- r03 drew the queue on SM 0's shared-memory shelf next to "2.46x", a number from the spill kernel. The overflow to VRAM was missing.
- r08 drew each tag as a cap over 45% of a slot. It is one bit of the word.
- r08's carried caption "64 tags max" stated a format limit as if it had been reached.
- r04 put block 1 on SM 1; it was measured on SM 2.
- `state.py` `BLOB_HUES[3]` "#e0a13a" sits at hue 37 deg, next to amber (GOLD "#d99a2b", 38 deg). v4 replaces it with a hue at least 30 deg from amber and from the other five.

Known caveats kept from v3 (unchanged, noted only):
- "Over twice" and "eighteen times" hold in the pinned session (CPU 2,584 ms). The other two ch03 sessions timed the CPU at 2,026 and 2,041 ms, which gives 1.9x and about 15x.
- r07's estimate is ch03 at 8-way; r08's is ch04 at 4-way.

---

## Open questions for the user

- **Q1. The owner-map run.** Per-block colours need a real owner map of the 64 Mpx square (D4). It is one GPU run of three kernels (2 x 256, 48 x 256 at 4-way and at 8-way), a few seconds each, about 1 GB of RAM. Approve it, or use the labelled schematic fallback?
- **Q2. "One SM with two blocks."** It exists only as ch02's pinned run on 36 Mpx: two blocks on one SM took 308.4 ms, the same two on two SMs 183.8 ms (1.68x). It is a different launch: 2 x 768 threads, a 40-register cap and a pair barrier. r04 already has three captions, and the carried "22 SMs idle" is needed at the cut. So if kept, it replaces "no slower": caption "1.7x" / "36 Mpx, 2 x 768", with a ghost cube on SM 0. Add it?
- **Q3. "Coalesced" on screen.** It is the standard term, and the row access is coalesced by design. The repo never measured it. Keep the word, with the sub "by design, not profiled"?
- **Q4. The lens in r05.** A 64 x 64 crop of the real owner map shows the speckle at full resolution. The mosaic's 63 px stride turns any finer pattern to noise, so only the lens is faithful up close. Worth the extra picture, if Q1 is approved?
- **Q5. Sixty-four tags.** v4 drops "tags run out at sixty-four" (never reached). Keep a short "and even a wider tag stops at sixty-four" in r08 (about +2.5 s)?
- **Q6. Length.** About 217 s with Kokoro (+70 s, +48%) and about 267 s with Adam. Is the vertical cut meant for Shorts (a three-minute cap)? If yes, which requests should get shorter? If not, take all of it, or the cut menu (about 211 s)?
- **Q7. The constants.** 48 KB and 8 GB come from the device dump, not from benchmark JSON. They would be constants in `numbers.py` with that source, like `CPU_CORES`. Acceptable?
- **Q8. A separate r10_memory beat.** A description beat with no But, between r09 and r10_runs: "In memory, a warp reads thirty-two neighbouring queue entries in one fetch. Each entry sends its thread to its own pixel, wherever it lies." About +9.6 s (Kokoro). v4 shows the same pictures in r08 instead. Want it as its own beat?
- **Q9. The one-block thread sweep.** At one block, 64 to 1,024 threads gave 18.0 to 86.2 Mpx/s, rising at every step (CH01). It is the closest match to "we changed the thread count". Add it as one caption in r02 ("more threads, faster" / "one block, CH01"), at the cost of r02's text budget?
- **Q10. "image" in VRAM from r02.** Seeing the image in VRAM from the start would make "only the queue was ever in shared memory" visible. A board label in r02 would land within MIN_DWELL of the SM close-up's labels, and a picture on VRAM chip 0 would collide with the queue slots there. v4 shows the image lane in r08's close-up instead. Add it to r02 anyway?

---

## Build order (short)

1. `numbers.py`: the new keys and fact checks; `uv run python -m scenes.chain.numbers` must pass.
2. The owner-map run (after Q1), committed with its JSON cross-check.
3. `machine.py` and `die.py`: `spill`, `shelves` and the third shelf state, `team_hues` and `team_busy`, stripe tags, the shelf gauge and cards, `vram_closeup`, packets.
4. `state.py`: the new `BEATS` and boundary table; then `base.py` HOLD keys and `build/assemble.py` `CHAIN`.
5. Scenes: r02; r03_giant (sweep re-anchored to "overflows" -> "gigabytes") and r03b_one_sm; r04 (its own `r04_plan()`, no import of `r03_plan`); r05 and r05b_snake; r06 (the 8-way mosaic); r08; r10_runs. Re-cue every changed beat to its new words.
6. `script_chain.md`: the new text. Then TTS for the eight new or changed beats (Kokoro first, then `pace.py` at its defaults, then Adam when the account allows). Re-check r04's caption spacing from the real `words.json`.
7. `build/layout_check.py` with the panels open; `build/chain_snapcheck.py` on all 13 snapping clips; `build/seam_check.py --cut chain` (14 seams); both layouts.
