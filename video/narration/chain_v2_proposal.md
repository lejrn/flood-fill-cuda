# The chain cut, re-edited as a "But... So" ladder in two acts

Status: PROPOSAL v3.1 for the user's approval, 2026-10-14. It revises the approved v2 with the user's four answers (drone scale, opening, look, ending), then applies a review of 22 issues (honesty and story). Nothing is built yet.

## The story in one paragraph

A camera films a drone show, and a filter turns every drone into a white blob.

On small frames, compiled CPU code would keep up: a 1 Mpx test frame with 21,618 blobs takes it 11.2 ms. Scale up to an 81 Mpx frame with the same 33 ms budget, and it needs 1.3 s. So where does the GPU win? We promise two places: **size** and **count**.

We freeze the footage and zoom into one drone until its pixels show. The CPU paints them one at a time, on one core, while 24 SMs sit idle.

One GPU block paints a whole ring per step. But on one small blob the CPU still wins, even against our best GPU code. A three-row scoreboard is born with its first row filled: "one small blob: CPU". So: size first.

**Act 1, SIZE.** We time one giant 64 Mpx blob. One block, then two, then all 48, each step faster. A thin snake shows the gate that every ring pays. On a square, diagonal steps halve the gates: 25.8x the CPU (kernel time). Row 2 fills: "one huge blob: GPU".

**Act 2, COUNT.** "Size is won. Now count." On the 81 Mpx frame (size and count together), 2,522 blobs at one launch each are slower than the CPU: we built for size, not count. Tags put two blobs in a launch. The GPU then finds its own seeds: 51 ms, still over budget. Linking spans of pixels instead of pixels gives about 3 ms. Row 3 fills on the 1 Mpx card from the opening: 21,618 blobs, GPU about 23x. On a frame of the same size, one big blob gives only about 10x, so the lead grows with count.

We pull back to the live drones: many small blobs in every frame. The full scoreboard stays open. "One small blob: the CPU. One huge blob, or many at once: the GPU. Every drone, every frame." The name sits on the title card only.

Base: the approved v2 (its beats, visual language, honesty rules and build notes). The user's answers change four things: two explicit acts, a CPU vs GPU scoreboard built from measured numbers, an honest opening, and an ending on "Every drone, every frame".

---

## Fixed visual language (whole video)

**Look: 2.5D (iso) only.** The floor and the machine are iso stages. The goal chip, the scoreboard, the captions and the inset are flat screen-space overlays on top. No flat variant is built for this cut.

**Six objects, each always in the same place:**
1. **Floor** (top in 9:16, left in 16:9). An iso tile that holds the real pixels of one drone. It changes in three planned ways only:
   - it **splits** into two half tiles for a side by side (r02, r06);
   - it **inflates** once into the giant 64 Mpx test square (r03, at the spoken "one giant blob");
   - it **pulls back** twice: to the 81 Mpx test frame (r07) and to the live drone footage (r11).
   - From r07 on, the 81 Mpx frame is drawn **transposed** (repo orientation [x, y]). A floor row is then a repo row, the same as in the toys and in ch06 (`toy.py` already transposes). In iso, that row runs along one fixed tile axis. The picture is synthetic, so the flip is invisible.
2. **Machine** (bottom or right). The existing 6 x 4 board in iso: 24 rooms = 24 SMs. Exactly one part slides on or off per rung.
3. **Goal chip and frame meter** (top). The chip reads **"81 Mpx frame @ 30 fps: 33 ms"**, with the mono tag "size and count together". It appears in i1 at "same thirty-three milliseconds", never before. It has a thin log track:
   - a teal tick at 33 ms;
   - a grey CPU tick at 1,346 ms, sub "@njit: label + fill, 81 Mpx, Jul 24";
   - an amber needle.
   The needle stays parked and dim through Act 1. It wakes in r07 and crosses 33 ms once, in r10, landing on "2.96 ms" (from the RGB image) and turning teal. A faint second mark shows "1.46 ms packed mask".
4. **Room gauge** on the chip: "1 / 24 = 4%", then 2 / 24, then "24 / 24 SMs" with the mono sub "1/3 of each SM: registers". This is the physical twin of each speed-up. It replaces the SpeedLadder.
5. **Scoreboard "CPU vs GPU"** (top band, under the goal chip). A flat card of three rows. Each row shows only "picture | CPU ms | GPU ms | winner", plus one short tag:
   - row 1 "one small blob" (tag "one session");
   - row 2 "one huge blob", labelled **SIZE** (tag "Act 1 machine; kernel only, ~3.8x with copies");
   - row 3 "thousands of blobs", labelled **COUNT** (tag "final machine; two sessions").
   - The SIZE and COUNT labels are seeded in i1 (two "?" chips at "size and count"). The card is born in r02, with row 1 filled and rows 2-3 showing "?".
   - Most of the time it is **collapsed** to a one-line strip of three winner pips ("CPU | ? | ?"). It **expands for 4 s** only when a row fills (r02, r06, r10) and stays open in r11.
   - Winner chips use the colour code: a CPU win is grey (the CPU tick's colour), a GPU win is teal (the answer).
   - The full sub for each row (picture, session(s), clock, contract) sits in **one footnote line** that is shown only on the r11 full card, where the card is open long enough to read.
   - The act in play shows as a small corner tag, "SIZE" (r03-r06) then "COUNT" (r07-r10). The word "act" is never spoken.
6. **Caption.** 1-3 words timed to the spoken word. A small mono sub-label names what the number was measured on: the blob, the session, "kernel time" or "estimate". The last caption of a beat crosses the cut.

**"You are here" inset** (from r00): the frozen drone frame, small, with a box on the chosen drone. In Act 1 it reads "not a drone: a stress test". In Act 2 it reads "a drone frame: ~900 blobs of ~15 px".

**Colour code (unchanged):**
- red: unpainted
- blue: working or painted
- amber: waiting (only the gate, the room fence and the dots that stand at them)
- grey: idle (and a CPU win on the scoreboard)
- one hue per blob
- teal: the final answer (and a GPU win on the scoreboard)

A new part gets a white outline pulse when it slides on, never a new colour.

**Devices, each used once:**
- zoom in at the start, pull back at the end;
- the promise (i1: "Two places: size and count") and the thesis card (r02: the scoreboard is born with one row filled and two "?");
- failure by absence (dark rooms);
- the enemy drawn as a flow (amber dots streaming to the gate);
- the turn line between the acts ("Size is won. Now count.") and the reframe line ("we built for size, not count");
- the hand leaves the frame;
- one thinking beat (the row dim at runs);
- an ending that answers the opening question ("Where does the GPU win?") with the full scoreboard.

**No "chapter" anywhere,** spoken or shown.

---

## Beats

Estimates come from a punctuation model (it reproduces v2's 328 words, 120.1 s and 174.3 s exactly):
- Kokoro: 0.314 s per word, plus 0.95 s gap and hold per beat, plus silent holds.
- Peter Baker: 0.38 s per word, 0.5 s per full stop, colon, semicolon or "?", 0.2 s per comma, 0.95 s per beat, plus silent holds.
- The reveal is budgeted at 1.9 words/s in Peter Baker.
- Peter Baker figures are predictions only.

### i0_problem: "a camera films a drone show" (intro)
- **DESIGN:** reuse `scenes/s_intro.py` beat 1: the camera panel and the filter panel (white blobs on black).
- **WHY IT WORKS:** it sets the job and the budget in pictures before any machine appears.
- **BUT:** none yet. This is the hook.
- **SO:** none.
- **Result:** 33 ms = 1000/30 (`s_intro.py` FPS_CAMERA = 30). On a 0.59 Mpx drone frame the GPU meets this budget: median about 11 ms per whole frame (see r11).
- **Narration (22 words):**
  > A camera films a drone show; a filter turns each drone into a white blob. Count them all, every frame: thirty-three milliseconds.
- **On screen:**
  - The four-panel layout.
  - The filter cue stays at 3.6 s.
  - The budget line "30 fps = 33 ms per frame" moves from 8.0 s to about 6.0 s, cued to the word.
- **Machine part:** none.
- **Est.:** Kokoro 7.9 s, Peter Baker 11.5 s.

### i1_compare: "the CPU keeps up, until we scale up" (intro)
- **DESIGN:** beat 2 of the intro, re-ordered. The user's side by side is kept: GPU panel live, CPU panel on test cards.
  - The GPU panel plays the live labelled drones with the blob counter from the beat start, tagged "GPU: drone frames, 0.59 Mpx". This replaces the spoken "The GPU keeps pace."
  - **Card 1 (at "On frames this small"):** the CPU panel shows the 1 Mpx test frame (`png_blocks`, red blocks on white), tagged "many blobs, small frame: CPU fits", sub "1 Mpx test frame, 21,618 blobs: not a drone frame". It has its own 33 ms line.
    - The @njit bar stops at 11.2 ms, under the line, with a plain white check.
    - A thin pure Python bar runs past to 814 ms, sub "pure Python, seeds given". It is not spoken.
    - No GPU bar here: its number is held for row 3 (r10).
  - **Card 2 (at "Scale up"):** the card swaps to the 81 Mpx test picture (red blobs on white), tagged "giant frame: CPU misses", sub "81 Mpx test frame, 2,522 blobs: not a drone frame".
    - **At the same moment the GPU drone panel freezes and dims** to 40%, keeping its tag "drone frames, 0.59 Mpx". Its blob counter stops. No live GPU counter runs beside the 81 Mpx stopwatch.
    - The 33 ms line appears on that card at "same thirty-three milliseconds". The goal chip "81 Mpx frame @ 30 fps: 33 ms" slides in at the same moment.
    - The stopwatch runs through the 33 ms line toward 1.3 s.
    - The verdict bars arrive at "one point three seconds", all on the same card: pure Python 24,083 ms (not spoken, sub "seeds given"), @njit 1,346 ms (sub "label + fill"), GPU "?".
  - At "Where does the GPU win?", the "?" bar pulses once. At "Two places: size and count", two small "?" chips pop out of it, labelled SIZE and COUNT. They fly up and park under the goal chip (they become rows 2 and 3 of the scoreboard in r02). Then the 0.5 s fade to black (the seam).
- **Honesty fix:**
  - The old caption "81 Mpx per frame" over the drone footage (`s_intro.py:189`) is removed.
  - The CPU never races the drone footage. It is shown keeping up on a frame that is larger and busier than a drone frame, then missing the budget on a frame 81x larger. The only visible BUT is on one card: the CPU stopwatch passing that card's 33 ms line, with the GPU bar "?" on the same card.
  - "would keep up" is an inference from the 1 Mpx row, not a drone-frame measurement. The drone-frame CPU timing exists only in scratch (see "The CPU on drone frames").
  - The card tags say what kills the CPU here: the giant frame. Count is promised as its own case, and it is shown with the frame size held fixed (row 3).
- **WHY IT WORKS:** the comparison is fair and the viewer can follow it: same CPU code, two named pictures, one budget line on each. The question, and the promise right after it, set up the two acts.
- **BUT (visible event):** on the 81 Mpx card, the CPU stopwatch runs past the 33 ms line toward 1.3 s, beside a GPU bar that reads "?".
- **SO:** "Where does the GPU win? Two places: size and count." The chain starts.
- **Result numbers** (all `overview/benchmark_results/overview_20260724T234423Z.json`, one session):
  - 1 Mpx: @njit 11.17 ms, `rows[png_blocks].cells.njit.ms`, 21,618 blobs (`n_blobs`); pure Python 813.5 ms, `cells.pure_python.ms` (one run, seeds given).
  - 81 Mpx: @njit 1,346 ms, `rows[png_blobs].cells.njit.ms`. This is `cpu_fill_canonical`, a CCL pass plus a multi-source fill (`overview/bench.py`). Pure Python 24,083 ms, `cells.pure_python.ms`, given its seeds.
  - Drone frames: 0.59 Mpx, 570-1,210 blobs (`assets/drones_labels/meta.json`), both smaller than the 1 Mpx row.
- **Narration (34 words):**
  > On frames this small, compiled CPU code would keep up. Scale up: eighty-one megapixels, same thirty-three milliseconds. The CPU needs one point three seconds. Where does the GPU win? Two places: size and count.
- **On screen:** the card tags above. The cues are re-timed to these words (the old ones were CPU at +0, GPU at +2.0 s and verdict at +5.0 s).
- **Machine part:** none.
- **Est.:** Kokoro 11.6 s, Peter Baker 17.8 s.

### r00_zoom: "one drone, magnified" (intro)
- **DESIGN:** from black, frame 216 of the camera footage fades in. Push in on its clean sky drone (box at rows 104-116, cols 64-77).
- **WHY IT WORKS:** a camera move replaces any title card. The unit of work, a pixel, becomes the stage for every rung.
- **BUT / SO:** none. This is the bridge. The white pixels become red floor cells on the iso tile.
- **Result:**
  - The pixel count comes from the mask at render time: new key `drone_px`, from `assets/drones_mask` frame 216.
  - Re-measured: 119 px under both 4- and 8-connectivity, 13 x 14 bounding box. The render-time key decides.
- **Narration (9 words, plus 1.5 s silent push-in):**
  > Freeze. Zoom into one drone until its pixels show.
- **On screen:**
  - At "Freeze": a 2-frame white flash.
  - The mask cross-dissolves over the camera, and a thin ring draws round the drone.
  - A nearest-neighbour push-in goes from 1x to about 40x, then the view tilts into the iso floor.
  - The inset appears, labelled "you are here".
  - Grid lines fade in. White pixels turn RED_PX. The camera crop ghosts under the cells for 0.5 s, so the drone's arms show through.
  - The goal chip and the parked SIZE / COUNT chips stay as set in i1.
  - Caption: "119 pixels", sub "one drone".
- **Machine part:** none yet.
- **Est.:** Kokoro 5.3 s, Peter Baker 6.9 s.

### r01_cpu: "one pixel per step" (rung 1)
- **DESIGN:** CPU BFS from the centre seed: paint a pixel, queue its red neighbours, repeat.
- **WHY IT WORKS:** BFS reaches every touching pixel exactly once. The rings are real: the oracle records `depth`.
- **BUT (visible event):** the side by side, at machine scale.
  - One worker dot crawls the floor, one cell per tick, while the CPU box glows "1 core used".
  - The dark 24-room chip sits beside it. At "twenty-four GPU SMs" a ripple of white outlines runs across all 24 rooms (about 1 s), then they fall back to grey.
  - The worker never stops crawling.
- **SO (carried):** move the work onto the GPU.
- **Result:**
  - 24 SMs: `overview_20260724T234423Z.json:sm_count`.
  - The CPU tick at 1,346 ms lands on the meter (parked, dim), with the sub "@njit: label + fill, 81 Mpx, Jul 24". It never says that this pixel walk took 1.3 s.
  - Step counter "1 ... 119" from `toy.bfs` on the drone mask.
- **Narration (24 words):**
  > On the CPU: paint a pixel, queue its red neighbours, repeat. One pixel per step, on one core, while twenty-four GPU SMs sit idle.
- **On screen:**
  - Each painted cell takes its ring shade, so the rings show even though they are painted one cell at a time.
  - Pace: slow for the first 6 cells, then faster.
  - Captions:
    - "1 pixel" (sub "per step");
    - "1 core" (sub "our CPU code; the laptop has 14 cores");
    - "24 SMs" (sub "streaming multiprocessors").
  - The 14 cores get a `numbers.py` entry with its source: "i9-13900H spec, 14 cores / 20 threads; not in the benchmark JSON".
- **Machine part added:** the CPU box with its queue strip (from c01), beside the dark chip.
- **Est.:** Kokoro 8.5 s, Peter Baker 12.4 s.

### r02_one_block: "fewer steps, still slower; the thesis card" (rung 2)
- **DESIGN:** ch01 v2 on the drone itself. One block of 256 threads (8 warps) in one SM paints one whole ring per step.
- **WHY IT WORKS:** every pixel in a ring is independent, so 256 threads paint the ring together.
- **On screen:**
  1. **Split floor** (the user's CPU-vs-GPU side by side), the same drone on two iso half tiles, reset to red.
     - Left "CPU": one cell per tick.
     - Right "GPU, 1 block": one whole ring per tick.
     - Step counters run on the same clock, with the mono tag "steps, not time". They stay in steps for the whole beat: the drone itself was never timed.
     - The right side finishes at its ring count (render-time key `drone_rings4`; about 12).
  2. **The race in time** (at "But on one small blob"): a small 128 x 128 test square chip pops up between the half tiles, tagged "16 K px test square". Its own two timers run in milliseconds and the CPU one stops first: "CPU 0.15 ms", "1 block 3.0 ms", "best GPU code 0.37 ms". Caption "in time, the CPU wins", sub "16 K px square; one session; kernel time". The drone floor keeps its step counters.
  3. **The thesis card** (at "the CPU still wins"): the scoreboard is born, expanded for 4 s. The two parked SIZE / COUNT "?" chips from i1 drop into rows 2 and 3.
     - Row 1 "one small blob | CPU 0.15 ms | GPU 0.37 ms | **CPU 2.4x**" (grey chip), tag "one session".
     - Row 2 "one huge blob | ? | ? | ?", label **SIZE**. It pulses at the spoken "size first".
     - Row 3 "thousands of blobs | ? | ? | ?", label **COUNT**.
     - Then it collapses to the strip "CPU | ? | ?".
  - Captions: "1 ring" (sub "per step"); "CPU wins" (sub as above).
- **BUT (visible event):** fewer steps on the drone, yet on the 16 K px test square the CPU timer stops first.
- **SO (carried):** the GPU must win another way, as promised. Size first.
- **Result** (`overview_20260724T234423Z.json:rows[sq_256]`, a 256 x 256 frame holding a 128 x 128 blob = 16,384 px, one session):
  - @njit 0.152 ms (`cells.njit.ms`, a seeded BFS with no seed discovery);
  - one block `ch01_ring` 2.997 ms;
  - best GPU of all 15 kernels: `ch05_fused_L8` 0.365 ms (it even finds its own seed). 0.152 / 0.365 = 2.4x for the CPU.
  - The drone itself (119 px) was never timed in committed data on either side. Every GPU kernel we timed takes at least about 0.35 ms even on tiny blobs: ch06 on this same square is 0.349 ms (mask), the floor of its six launches, measured at a low 900 MHz SM clock (`ch06_overview_20260927T013331Z.json:rows[sq_256]`). So a smaller blob would most likely widen the CPU's lead. That is an inference, not a measurement.
- **Narration (28 words, plus 0.4 s hold for the card and 1.0 s for the 4 s expand):**
  > So move it onto one SM: two hundred fifty-six threads paint a whole ring at once. But on one small blob, the CPU still wins. So: size first.
- **Machine part:**
  - Added: room 0 lit with a team (on the drone it needs no shelf spill; that comes in r03).
  - The CPU box stays for this race only.
- **Est.:** Kokoro 11.1 s, Peter Baker 15.7 s.

### r03_giant: "one giant blob" (rung 3, Act 1 SIZE opens)
- **DESIGN:** the same ch01 v2 block on the 64 Mpx test square. It has a shared-memory shelf (8,192 slots) with a spill tier in global memory, and a room fence (`syncthreads`) after every ring.
- **WHY IT WORKS:** a giant blob has giant rings, so the 256 threads always have work.
- **On screen:**
  1. **Inflate** at "one giant blob".
     - The left half tile and the CPU box slide away. The CPU box stays as a faint ghost for the r11 callback.
     - The drone swells into the fine square, labelled "test square 8000 x 8000 = 64 Mpx, drawn small".
     - The inset reads "not a drone: a stress test". The corner tag "SIZE" appears.
  2. **Ring sweep.** The ring is now thousands of cells long.
     - One team, drawn as raster dots (mono "1 dot = many threads"), sweeps it segment by segment, like a progress bar, over several passes.
     - The shelf fills and a small chute pours the overflow into global memory, sub "big rings overflow the shelf (43% of a 36 Mpx blob)".
     - The room fence flashes amber after each ring.
  - Captions:
    - "2x" (sub "2.46x the CPU, 64 Mpx square, kernel time");
    - "1 / 24 = 4%".
- **BUT (visible event):** failure by absence. Room 0 glows, 23 rooms stay dark, and the gauge reads "1 / 24 = 4%".
- **SO (carried):** a second block on a second SM.
- **Result:** one session for r03-r06: `ch03_gpu_1blob_nblock/benchmark_results/multi_block_20260720T012243Z.json:sq_8000_center`.
  - 2.46x = 1,052.0 vs 2,583.96 ms: `v2_kernel_ms`, `njit_ms`, `speedup_v2_vs_njit`. New key replaces `c02_x`.
  - 4.17% = 1/24 (`c02_sm_pct`).
  - Spill 43%: `ch01 single_block_shared_20260719T041848Z.json:sq_6000_center.spill_pct` (sq_8000 spill not measured).
- **Narration (17 words, plus 0.8 s hold for the inflate):**
  > On one giant blob, one block is over twice the CPU. But that's one SM of twenty-four.
- **Machine part:**
  - Added: the shelf with its spill chute, the room fence.
  - Removed: the CPU box and its queue (ghost).
- **Est.:** Kokoro 7.1 s, Peter Baker 9.4 s.

### r04_two_blocks: "two arcs" (rung 4, a fast counter-move)
- **DESIGN:** ch02 `global`. Two blocks on two different SMs share one queue in **global** memory. They meet at a chip-wide barrier (`grid.sync`, two per ring, about 1.9 us each).
  - Each block's shared memory is private to that block, even with two blocks on one SM.
  - ch02 also tried keeping a shelf per block (`split`, with cross-seam inboxes). It was slower (280 / 305 / 314 ms for global / dirsplit / split), because L2 already absorbs the queue's hot window.
  - So the shelf comes off because one shared global queue turned out faster, not because it had to.
- **WHY IT WORKS:** two teams sweep two segments of the ring at once, so each ring needs half the passes.
- **BUT (visible event):** 22 rooms are still dark. The gauge creeps to "2 / 24 = 8%".
- **SO (carried):** use every SM.
- **Result:** same session.
  - 2.04x = 516.7 vs 1,052.0 ms (`sq_8000_center.dual_global_kernel_ms`, `v2_kernel_ms`). That is 5.0x the CPU.
  - 8% = 2/24.
- **Narration (17 words):**
  > So add a second block on a second SM: twice as fast. But twenty-two SMs stay dark.
- **On screen:**
  - Room 0's prop is stamped into a second room (the maglev "repeat the object" move).
  - The shelf slides off with the caption "one shared queue". One long queue strip slides along the chip.
  - A thin grey gate outline appears around the chip, "at rest" and not explained. It is planted here and pays off in r05.
  - Captions: "2x again" (sub "vs one block, same 64 Mpx square, same session, kernel time; 5.0x the CPU"); "2 / 24 = 8%".
- **Machine part:**
  - Added: room 2 lit, the global queue, the gate (at rest).
  - Removed: the shelf.
- **Est.:** Kokoro 6.3 s, Peter Baker 8.9 s.

### r05_all_blocks: "48 blocks, and the gate" (rung 5, the obvious fix backfires)
- **DESIGN:** ch03 on the cooperative maximum grid: 48 blocks x 256 = 12,288 threads, 2 per SM on all 24 SMs.
  - About 104 registers per thread cap the grid at 512 of 1,536 thread slots per SM (occupancy 33.3%).
- **WHY IT WORKS:** on the same 64 Mpx square, the whole ring flashes blue in one pass. No relabel is needed: r03-r06 all use this one square.
- **The square's gates:** a small amber tally band on the square's frame stamps 2 ticks per ring. It reads "16,002 gates = 8,001 rings x 2". This is the tally r06 halves.
- **BUT (visible event):** "the picture fights back".
  - Hard cut on the floor to the thin snake (`serpentine_256`).
  - One blue dot paints at its tip, which is the same picture as the CPU rung.
  - At "Why?" the gate slams amber round the whole chip, **twice per ring**. Every other dot turns amber and drifts to it (12,287 idle, caption only).
  - The mechanism is the **price of each stop**. Each stop prints "~2 us" on a tick. A separate snake counter runs to "65,792 stops = 32,896 rings x 2". A sub reads "~2 us per stop (measured 1.7-2.2 us): the stops are most of the 143 ms".
  - Three timing chips: "1 block 62 ms" (syncthreads), "2 blocks 128 ms", "48 blocks 143 ms" (grid.sync). So the gate type decides it, not the block count.
- **SO (carried):** pass fewer gates.
- **Result:** all from `multi_block_20260720T012243Z.json`.
  - 18.75x = 137.8 vs 2,584 ms: `sq_8000_center` (`c03_n_x`). Sub-label "kernel time; same session".
  - Snake: v2 61.7, dual 127.5, multi 143.4 ms (`serpentine_256`).
  - 12,287 of 12,288 idle: `c04_threads`, util 0.008% (`c04_util_pct`). Caption only.
  - 65,792 = 2 x 32,896 levels (`c04_gates`), on screen only.
  - Price per stop: measured separately at ~1.9 us (ch02 serpentines, `chapters/README.md:129`) and ~1.7 us at 1 block to ~2.2 us at 192 blocks (ch03, `chapters/README.md:216`). At that price, 65,792 stops cost about 112-145 ms of the 143.4 ms; ch03's own note says "65,792 barriers x ~2 us *is* the runtime" (`chapters/README.md:217`). The 143.37 / 65,792 = 2.18 us division is a consistency check, not a measurement.
- **Narration (32 words):**
  > So use every SM: forty-eight blocks, eighteen times the CPU. But on a thin snake, they lose to one. Why? Every ring, the chip stops twice at a gate, two microseconds each.
- **On screen:**
  - A ripple stamps 2 teams into each room. The gauge reads "24 / 24 SMs", sub "1/3 of each SM: registers".
  - Captions:
    - "18x" (sub "x the CPU, 64 Mpx square, kernel time");
    - "48 < 1" (sub "one thin snake, GPU kernels only");
    - "the gate" (sub "grid.sync, 2 per ring");
    - "65,792 stops" (sub "the snake is all gates").
- **Machine part added:** all 24 rooms lit, 2 teams each. The gate turns amber; the square's tally band appears.
- **Est.:** Kokoro 11.0 s, Peter Baker 16.4 s.

### r06_diagonals: "diamond to square; size is won" (rung 6, Act 1 closes)
- **DESIGN:** ch03 conn8. Each pixel also reaches its 4 diagonal neighbours.
- **WHY IT WORKS:** on a square the wave front becomes a square, not a diamond. Two things happen:
  - half the rings, so half the gates;
  - each ring is twice as wide, so it keeps more threads busy (util 61.6% -> 80.8%).
  - The gates alone explain about half of the 1.37x: 2 x 8,001 x ~2.2 us is about 35 ms of 137.8 ms (est.).
- **It does not change the answer here:** the test shapes are the same component set under 4 and 8 ways (`overview/bench.py`), and from ch05 on everything is 8-connected.
- **BUT:** none new. The next failure is the twist (r07).
- **The snake stays open, and the narration says so by scoping the fix:** "On a square" is spoken. 8 ways does not help the snake (32,896 -> 32,641 rings, 178.3 vs 143.4 ms, 0.80x). Its 65,792 counter is hidden here and paid off in r10, where the gate itself goes.
- **SO:** the stencil badge's 4 arrows become 8. Scoreboard row 2 fills.
- **Result:** from `multi_block_20260720T012243Z.json:sq_8000_center`, the same session as r03-r05.
  - Levels 8,001 -> 4,001 (`c05_levels4`, `c05_levels8`); gates 16,002 -> 8,002.
  - 25.8x = 100.3 vs 2,584 ms (`c05_x`, 25.76).
  - With copies to and from the card, the 48-block run is 687.4 ms, 3.8x the CPU (`multi_total_ms`). No total was recorded for conn8, so none is shown for it.
- **Narration (16 words, plus 1.0 s hold for the 4 s expand):**
  > So pass fewer gates: step diagonally too. On a square, half the rings, half the gates.
- **On screen:**
  - Split floor again (two iso half tiles), on the 64 Mpx square. Left "4 ways": a diamond ring. Right "8 ways": a square ring. Caption "diamond -> square".
  - Ring counters come from the toy's own BFS. Mono sub: "64 Mpx: 8,001 vs 4,001 rings".
  - The square's amber tally band shrinks from 16,002 to 8,002.
  - Captions:
    - "8" (sub "same blobs: ours count diagonals anyway");
    - "half" (sub "on a square; twice as wide; the snake still pays every gate");
    - "25.8x" (sub "x the CPU, 64 Mpx square, kernel time"). This caption replaces the cut spoken "twenty-five times the CPU" (cut-order item 3, applied).
  - **Scoreboard row 2** fills at "half the gates": "one huge blob | CPU 2,584 ms | GPU 100 ms | **GPU 25.8x**" (teal chip), tag "Act 1 machine; kernel only, ~3.8x with copies". The card stays open 4 s, then collapses to "CPU | GPU | ?". The SIZE corner tag gets its teal tick.
- **Machine part added:** the stencil badge (4 -> 8 arrows).
- **Est.:** Kokoro 7.0 s, Peter Baker 9.9 s.

### r07_real_frame: "size is won, now count" (the twist, Act 2 COUNT opens)
- **DESIGN:** feed the 81 Mpx test frame through this machine (all blocks, 8 ways), one launch per blob.
- **WHY IT "WORKS":** every blob does get labelled.
- **BUT (visible event):** the story visibly goes backwards.
  - At "Size is won": the SIZE tag already carries its teal tick. At "Now count" it flips to "COUNT".
  - **Pull back:** the giant square shrinks to a dot. The floor becomes the 81 Mpx test picture (drawn transposed) with 2,522 red blobs, tagged "size and count together".
  - **Strobe:** the board lights, paints one blob and drains amber, again and again, while a launch counter races to 2,522.
  - **Meter:** the needle wakes up and swings the wrong way, past the CPU tick at 1,346, to "~2,181 ms".
  - The inset flashes the drone frame with "a drone frame: ~900 blobs of ~15 px". It is not equated with this frame: the drone blobs are about 300x smaller.
- **SO (carried):** put more than one blob in a launch.
- **Result:**
  - 2,522 blobs: `overview_20260724T234423Z.json:rows[png_blobs].n_blobs` (also `runs_20260725T161448Z.json:scenes[input_blobs].n_blobs`).
  - ~2,181 ms, an ESTIMATE (`est: true`, the median of 6 sampled blobs x 2,522 calls): `overview_20260724T234423Z.json:rows[png_blobs].ch03_conn8.ms` (`c06_ms`, `c06_calls`).
  - Drone blobs 570-1,210 per frame, median 938 (`assets/drones_labels/meta.json`); blob size median 11-18 px on 5 sampled frames (scratch).
- **Narration (25 words):**
  > Size is won. Now count: twenty-five hundred blobs in the big frame, one launch each. Slower than the CPU: we built for size, not count.
  - "we built for size" calls back to the promise in i1 and to "So: size first" in r02.
- **On screen:** captions "2,522 launches" (sub "one per blob, 81 Mpx test frame"); "~2,181 ms" (sub "estimate, kernel time only").
- **Machine part added:** the launch counter.
- **Honesty:**
  - "it" means this machine. The one-block ring kernel alone is ~1,222 ms (est.), which edges @njit.
  - Each blob pays a fixed per-call cost inside the kernel (a grid-wide barrier on every ring, the start of the cooperative grid). A mean blob of 5,334 px cannot amortise it.
  - Host launch overhead between calls is not even counted, so ~2,181 ms is a lower bound for a real loop.
  - This frame mixes size and count. Its CPU miss is mostly the pixel volume (13.45 M red px). Count by itself is shown on row 3, with the frame size held fixed.
- **Est.:** Kokoro 8.8 s, Peter Baker 13.3 s.

### r08_tags: "two blobs, one launch" (rung 7)
- **DESIGN:** ch04 multisource. The blob id rides in spare bits of each queue entry, so two seeds share one queue and one launch.
- **WHY IT WORKS:** two blobs fill in max(a, b) rings, not a + b, with half the launches.
- **BUT (visible event):**
  - A tag rack beside the queue reads "64 max", and "2,522" leans against it and overflows.
  - A hand reaches in and drops every seed, one per blob.
- **SO (carried):** let the GPU find its own seeds, with no tag cap.
- **Result:**
  - ~1,368 ms ESTIMATE with 1,261 launches: `overview_20260724T234423Z.json:rows[png_blobs].ch04_multi.ms` (`c07_ms`, `c07_calls`), about 2% slower than @njit 1,346.
  - 64 labels = 6 label bits, mask 0x3F: `chapters/ch04_gpu_2blob_nblock/kernels.py`.
- **Narration (26 words):**
  > So tag each queue entry with its blob: two per launch, about CPU speed. But tags run out at sixty-four, and every seed is handed in.
- **On screen:**
  - Floor: toy `two_blobs` (or two real drone blobs; see open questions) filling in blue and green, captioned "max, not sum".
  - The strobe runs at half the rate. The needle settles on the CPU tick: "~1,368 ms" (sub "estimate").
  - Caption "64 tags" (sub "6 spare bits") at "sixty-four".
- **Machine part added:** the tag printer on the queue, with its rack. The hand is a prop.
- **Est.:** Kokoro 9.1 s, Peter Baker 12.7 s.

### r09_seeds: "the hand leaves the frame" (rung 8)
- **DESIGN:** ch05 `split_L8`.
  - Every red pixel with no red neighbour before it in scan order starts a wave, plus a stride-8 lattice (shown, not spoken).
  - All waves run in **one cooperative fill launch**. Where waves collide, both keep the smaller label (`atomicMin`).
  - Two plain cleanup kernels (compress, finish) follow, so the build is 3 launches in all (`ch05 flood_fill.py:447-456`).
- **WHY IT WORKS:** no hand, no tag cap, one fill launch for every blob.
- **BUT (visible event):**
  - The needle dives and stops at "51 ms", still right of the 33 ms tick and still amber.
  - The camera moves onto the queue conveyor, jammed with tokens, while a counter runs to "13,451,960". Tokens keep their blob hues. Amber stays only on the gate and the fence.
- **SO (carried):** stop moving pixels.
- **Result:**
  - 51.03 ms: `overview_20260724T234423Z.json:rows[png_blobs].cells.ch05_split_L8.ms`. This is the same session as the CPU tick (1,346) and the r07-r08 estimates, so the meter does not mix sessions up to here.
  - 13,451,960 red px: `runs_20260725T161448Z.json:scenes[input_blobs].red_px` (a property of the image, the same in every session).
  - The Jul 25 figure (58.51 ms, `runs...:input_blobs.ch05.median_ms`) is kept only for the same-session ch05 -> ch06 comparison in r10.
- **Narration (23 words):**
  > So the GPU finds its own seeds: every blob in one fill launch, fifty-one milliseconds. But every red pixel still rides the queue.
- **On screen:**
  - The hand lifts out of the frame. Seed sparks pop at the top-left corners, then on the lattice.
  - On the U toy, two colours meet. A 2 s cutaway on black shows a green and a purple label entering the "min" unit, and the smaller one flows back. Caption "where waves meet, the smaller label wins" (this replaces the cut spoken sentence, cut-order item 2, applied).
  - The launch counter drops from 1,261 to "3", sub "1 fill + 2 cleanup".
  - Captions: "seeds", "min", "51 ms" (sub "still over 33 ms: 1.5x the budget; Jul 24, host clock").
- **Machine part:**
  - Added: seed sparks and the min unit.
  - Removed: the tag printer (labels move to a label map), its rack and the hand.
- **Est.:** Kokoro 8.2 s, Peter Baker 11.4 s.

### r10_runs: "stop moving pixels; count is won" (rung 9, the big jump and the only thinking beat)
- **DESIGN:** ch06. Each row's red pixels form runs.
  - Runs that touch (8-connected) are linked by union-find with `atomicMin`, and each run is painted in one stroke.
  - It is a chain of 6 plain kernels: count, scan, emit, merge, flatten, paint (plus `pack` on the RGB contract).
  - It is not barrier-free: stream order is the barrier between kernels ("this pipeline is a DAG with no loop, so stream order is the barrier", `chapters/README.md:469`). The barriers drop from one pair per ring, inside a loop, to 5 seams.
- **WHY IT WORKS:** 24.95x fewer items, and no BFS, so no queue, no ring loop and no cooperative launch.
- **BUT:** none. It answers r09.
- **SO:** the unit of work changes, from pixels to spans. Scoreboard row 3 fills.
- **Result:** from `runs_20260725T161448Z.json:scenes[input_blobs]`, one session (Jul 25).
  - 539,207 runs (`n_runs`). These are rows in repo orientation, so the floor shows the transposed frame.
  - 13,451,960 / 539,207 = 24.95x (`c09_runs_x`).
  - **2.957 ms from the RGB image** (`ch06.rgb.median_ms`, new key `ch06_rgb_ms`), the like-for-like input: the CPU's `cpu_fill_canonical(img)` also starts from the RGB image. 1.455 ms from a packed mask (`ch06.mask.median_ms`, `ch06_mask_ms`), which skips the 243 MB RGB read and the pack step (`chapters/README.md:471, 481`). Both are CUDA events. The second Jul 25 session gives 3.03 / 1.35 ms (`runs_20260725T160000Z.json`), so "about three" holds in both.
  - vs ch05 in the same session: 58.51 / 2.957 = 19.8x (RGB), 58.51 / 1.455 = 40.2x (mask), shown as "20x (40x from a packed mask), same session; ch05 host clock vs ch06 GPU events". On screen only. No launch-overhead figure is added on top: below about 4 Mpx the 0.33 ms host enqueue is already inside the event-timed floor (`chapters/README.md:485`), and on 81 Mpx it overlaps the kernels.
- **Narration (26 words, plus 0.5 s hold and 1.0 s for the 4 s expand):**
  > Look at one row: red pixels come in spans. So link touching spans, not pixels. No queue, no ring loop: six plain steps. About three milliseconds.
- **On screen:**
  - **Thinking beat:** the whole floor dims to 30% except one row (one iso strip). It is a dim, not a blur, because Cairo has no cheap blur.
    - In that row the red cells slide together into teal bars.
    - Caption "13 M pixels -> 0.5 M spans" (sub "25x fewer"). This replaces a spoken sentence.
    - Then the floor collapses row by row.
  - Optional 1 s flash: the r05 snake collapses into a few spans, with its hidden "65,792 stops" counter returning and striking through. Visual only, no time claim, because ch06 was never timed on the snake.
  - Parts slide off one at a time, each with one word: "no queue", "no ring loop". The room fence and the launch counter go too.
  - The amber gate does not just vanish. It shrinks into 5 thin grey seams between the 6 kernel boxes of the strip that slides on. The runs flow through it once.
  - The needle crosses 33 ms, the only crossing in the video, and turns teal at "2.96 ms" (sub "81 Mpx, from the RGB image, GPU events, Jul 25"). A faint mark beside it reads "1.46 ms packed mask".
  - **Scoreboard row 3** fills at "About three milliseconds", on the 1 Mpx card from i1: "thousands of blobs: 1 Mpx, 21,618 blobs | CPU 11.2 ms | GPU 0.48 ms | **GPU ~23x**" (teal chip), tag "final machine; two sessions". Under it, one mono line: "same frame size, one 262 K px blob: CPU 4.5 ms, GPU 0.46 ms, ~10x". So the lead grows with count while the frame size stays fixed. The COUNT tag gets its teal tick. The card stays open 4 s, then collapses to "CPU | GPU | GPU".
  - Captions: "spans" (sub "a red run in one row"); "2.96 ms" (sub "81 Mpx test frame, from the RGB image, GPU events; 1.46 ms from a packed mask").
- **Machine part:**
  - Added: the run table and the 6-kernel strip with its 5 seams.
  - Removed: the queue, the room fence, the gate with both tallies, the launch counter, the seed sparks.
- **Est.:** Kokoro 10.6 s, Peter Baker 15.7 s.

### r11_reveal: "every drone, every frame" (reveal)
- **DESIGN:** the finished board, a short scar map, then the full scoreboard, then the r00 zoom reversed. The teal spans shrink into one teal drone, then out to the drones_labels footage at camera speed with the live blob counter.
- **WHY IT WORKS:** it answers the opening question ("Where does the GPU win?") with the scoreboard the viewer watched fill. The spoken line ties the drones to "many at once", the count case, and closes the loop the intro opened.
- **BUT / SO:** none. No amber anywhere.
- **Result:**
  - The three rows as filled in r02, r06 and r10.
  - **Row 2, second line (new here, issue 15):** the final machine on the largest single blob it was timed on: "final machine, 11 Mpx disk: CPU 269 ms, GPU 1.17 ms, ~230x", tag "two sessions; RGB". Source: `overview_20260724T234423Z.json:rows[disk_4000].cells.njit.ms` 269.48 (a seeded BFS, no discovery) and `ch06_overview_20260927T013331Z.json:rows[disk_4000].cells.ch06_rgb.ms` 1.168 (mask 0.982, ~275x). The 64 Mpx square was never timed with ch06. This line keeps rows 2 and 3 on the same machine, so the viewer does not read the Act 1 machine's 25.8x against the final machine's 23x as "count beats size".
  - **Size and count together (replaces the old "drone scale" bridge line):** "81 Mpx, 2,522 blobs | CPU 1,346 ms | GPU 2.96 ms | ~450x", tag "two sessions; RGB". This answers the "?" bar of i1. Source: `overview...:png_blobs.njit` 1,345.94 (Jul 24) and `runs_20260725T161448Z.json:input_blobs.ch06.rgb.median_ms` 2.957 (Jul 25). From a packed mask it would be 1.455 ms, ~925x; that figure goes only in the footnote.
  - **Footnote line** (mono, one line, r11 only): "Row 1: 16 K px square, one session, kernel time, CPU given its seed. Row 2: 64 Mpx square, one session, kernel only. Row 3 and below: CPU @njit label + fill (Jul 24) vs GPU from the RGB image, GPU events (Jul 25 / Sep 27)."
  - Drone frame, from `video/assets/drones_labels/timing.json` (committed):
    - labelling kernel 1.20 ms (`kernel.median_ms` 1.2045);
    - whole frame about 11 ms of 33 (median): host filter 6.25 ms (`filter.median_ms`) + labelling kernel 1.20 ms (`kernel.median_ms`) + GPU tracker steps 3.29 ms (`gpu_wall.median_ms`) = 10.74 ms. `gpu_wall` starts after the recolor call, so the kernel is not inside it. The recolor call's host wall time (launches, any copy) was never recorded, so 10.7 ms is a lower bound. A sum of medians is not the median of the sum: "about 11 ms" is approximate.
    - The p90s sum to about 16.8 ms (8.14 + 2.47 + 6.14). That is not the p90 of the total, and no per-frame max was recorded. See the build note on recording a per-frame total.
- **Narration (18 words, plus 2.5 s hold on the drones; the title card fades in over the drones for the last 1.5 s):**
  > One small blob: the CPU. One huge blob, or many at once: the GPU. Every drone, every frame.
- **On screen:**
  - Scar callouts, about 2 s and optional:
    - 24 rooms: "1 of 24 = 4%"
    - 8-way badge: "diamond: 8,001 rings -> 4,001"
    - min unit: "two waves met"
    - run strip: "13 M pixels in a queue"
    - ghost of the gate (now the 5 seams): "65,792 stops on a snake"
    - ghost of the seed sparks: "a hand fed every seed"
    - ghost outlines of the removed queue and fence.
  - The scoreboard expands and stays open, with the footnote line.
    - At "One small blob: the CPU", row 1 pulses and the faint CPU worker from r01 reappears in a corner, still painting its one drone.
    - At "One huge blob", row 2 pulses and its final-machine line types in.
    - At "or many at once", row 3 pulses and the "size and count together" line types in under it.
  - At "Every drone, every frame": the reverse zoom to the live drones, captioned "labelling 1.2 ms; whole frame about 11 ms of 33 (median)". No CPU ratio is shown for drone frames.
  - Title card, faded in over the drones: "connected-component labelling / flood-fill-cuda". The name is shown, not spoken.
- **Machine part:** final state, with ghosts.
- **Est.:** Kokoro 9.1 s, Peter Baker 12.9 s (budgeted at 1.9 words/s).

---

## Rung table

| beat | act | But (visible event) | So (one change) | result (source) | scoreboard |
|---|---|---|---|---|---|
| i1 | intro | 1 Mpx card: CPU 11.2 ms under 33; 81 Mpx card: CPU stopwatch passes 33 toward 1.3 s, GPU bar "?" on the same card; drone panel frozen and dimmed | "Where does the GPU win? Two places: size and count." | @njit 11.17 ms (`overview_20260724T234423Z.json:png_blocks.njit`), 1,346 ms (`png_blobs.njit`) | SIZE / COUNT "?" chips seeded |
| r01 CPU | intro | one cell per tick on 1 core; 24 SMs ripple and go dark | move onto one SM | 24 SMs (`sm_count`); step counter 1-119 (`toy.bfs`) | not yet |
| r02 1 block | thesis | fewer steps on the drone, but on a 16 K px test square the CPU timer stops first | size first | 16 K px square: CPU 0.152 vs best GPU 0.365 ms (`overview...:sq_256`) | born; row 1 CPU 2.4x |
| r03 giant | SIZE | 23 rooms dark, gauge 1/24 = 4% | a second block on a second SM | 2.46x the CPU, 64 Mpx, kernel time (`multi_block_20260720T012243Z.json:sq_8000_center.v2_kernel_ms`) | |
| r04 2 blocks | SIZE | 22 rooms dark, gauge 2/24 = 8% | use every SM | 2.04x vs 1 block, 5.0x the CPU (`dual_global_kernel_ms`) | |
| r05 48 blocks | SIZE | snake: 48 blocks 143 ms, 2 blocks 128, 1 block 62; 2 stops per ring at ~2 us; square tally 16,002 | pass fewer gates | 18.75x the CPU (`sq_8000_center`); 65,792 stops (`serpentine_256`) | |
| r06 8 ways | SIZE | (none new; the square's tally halves; "on a square" spoken; the snake stays open) | 8-arrow stencil | 8,001 -> 4,001 rings, 25.8x the CPU, kernel only (same file) | row 2 GPU 25.8x |
| r07 twist | COUNT | "Size is won. Now count."; 2,522 launches; needle swings past the CPU | more than one blob per launch | ~2,181 ms est. (`overview...:png_blobs.ch03_conn8`) | |
| r08 tags | COUNT | rack "64 max" vs 2,522; a hand drops every seed | the GPU finds its own seeds | ~1,368 ms est., 1,261 launches (`png_blobs.ch04_multi`) | |
| r09 seeds | COUNT | needle stops at 51 ms, over 33; conveyor jammed with 13,451,960 tokens | stop moving pixels | 51.03 ms, 1 fill + 2 cleanup (`overview...:png_blobs.ch05_split_L8`, same session as the CPU tick) | |
| r10 runs | COUNT | (answers r09) | spans, not pixels; queue and ring loop removed; gate becomes 5 seams | 539,207 runs (24.95x fewer); 2.957 ms RGB / 1.455 ms mask, GPU events (`runs_20260725T161448Z.json:input_blobs.ch06`) | row 3 GPU ~23x on 1 Mpx, 21,618 blobs (two sessions) |
| r11 reveal | end | no amber left | (none) | row 2 final-machine line ~230x (`disk_4000`, two sessions); size and count together ~450x (`png_blobs`, two sessions); drone frame: labelling 1.2 ms, whole frame about 11 ms median (`drones_labels/timing.json`) | full card + footnote |

## Scoreboard (the CPU vs GPU comparison, measured numbers only)

| row | picture | CPU | GPU | winner | tag on screen | source |
|---|---|---|---|---|---|---|
| 1 one small blob | 16 K px square (`sq_256`) | @njit 0.152 ms (seed given) | best of 15 kernels, `ch05_fused_L8` 0.365 ms | **CPU 2.4x** | one session | `overview_20260724T234423Z.json:rows[sq_256]` |
| 2 one huge blob (SIZE) | 64 Mpx square (`sq_8000_center`) | @njit 2,584 ms (seed given) | 8 ways, 48 blocks, 100.3 ms, kernel only (seed given) | **GPU 25.8x** | Act 1 machine; kernel only, ~3.8x with copies (48 blocks, `multi_total_ms` 687.4) | `multi_block_20260720T012243Z.json:sq_8000_center` |
| 2, r11 line | 11 Mpx disk in a 16 Mpx frame (`disk_4000`, 11,340,981 px) | @njit 269.5 ms (seeded BFS) | ch06 from RGB 1.168 ms (mask 0.982) | **GPU ~230x** | final machine; two sessions; RGB | `overview...:disk_4000.njit`; `ch06_overview_20260927T013331Z.json:disk_4000.ch06_rgb` |
| 3 thousands of blobs (COUNT) | 1 Mpx test frame, 21,618 blobs (`png_blocks`) | @njit label + fill 11.17 ms | ch06 from RGB 0.482 ms (mask 0.490) | **GPU ~23x** | final machine; two sessions | `overview...:png_blocks.njit`; `ch06_overview_20260927T013331Z.json:png_blocks.ch06_rgb` |
| 3, sub line | ~1 Mpx frame, one 262 K px blob (`sq_1024`) | @njit 4.48 ms (seeded BFS) | ch06 from RGB 0.458 ms | ~10x | same frame size; two sessions | `overview...:sq_1024.njit`; `ch06_overview...:sq_1024.ch06_rgb` |
| size and count together (r11) | 81 Mpx test frame, 2,522 blobs (`png_blobs`) | @njit label + fill 1,346 ms | ch06 from RGB 2.957 ms (mask 1.455) | **GPU ~450x** | two sessions; RGB | `overview...:png_blobs.njit`; `runs_20260725T161448Z.json:input_blobs.ch06.rgb` |

Notes on the table:
- **Row 3 vs its sub line.** On the CPU the one-blob row is a seeded BFS, while the 21,618-blob row also labels (finds the seeds). The ch06 `sq_1024` cell ran at 1,410 MHz, against 2,070 MHz on `png_blocks`; both sit near ch06's launch floor (0.35-0.40 ms), so the clock matters little. The same-session ch05 numbers show the same trend: 4.48 / 0.874 = 5.1x on one blob, 11.17 / 1.235 = 9.0x on 21,618 blobs. So "the lead grows with count" holds in one session too.
- **Contracts.** Every ch06 figure leads with the RGB contract, because the CPU baseline also starts from the RGB image. On `png_blocks` RGB is in fact the faster ch06 cell (0.482 vs 0.490 ms).
- **Not on the board: a drone frame (0.59 Mpx).** Its CPU timing was measured only in scratch (see "The CPU on drone frames"). It is **not committed yet: it needs a committed benchmark JSON before it can be shown.**

## Scar map (the final machine, part by part)

- **24 rooms lit, 2 blocks each.** Forced by "one SM of twenty-four" (r03) and "twenty-two SMs stay dark" (r04).
- **Every thread works at once (the GPU itself).** Forced by "one pixel per step, on one core" (r01).
- **The 8-way stencil.** Forced by the square's gates: 8,001 rings x 2 (r05, r06). Callout: "diamond: 8,001 rings -> 4,001". ch06 still links runs with 8-connectivity.
- **Labels in the data (a label map).** Forced by 2,522 launches, slower than the CPU (r07).
- **The min unit.** Forced by two waves meeting (r09); in ch06 it decides where two spans touch. Callout: "two waves met".
- **Run table and the 6-kernel strip, with 5 seams.** Forced by 51 ms still over 33 ms, with every red pixel riding the queue (r09). The seams are what is left of the gate.
- **The scoreboard.** Forced by the opening question "Where does the GPU win?" and its promise "size and count" (i1). Its first row is the one place the CPU stays the winner.

**Ghosts (added, then removed):**
- CPU worker and queue: r01 -> r03 (reappears at r11, beside the scoreboard's CPU row).
- Shelf: r03 -> r04 (each block's shelf is private; one shared global queue was faster).
- Tag printer, rack and the hand: r08 -> r09.
- Seed sparks: r09 -> r10. Callout: "a hand fed every seed".
- Queue, room fence, launch counter: removed in r10.
- The gate with both tallies: r04 -> r10, shrunk to 5 seams. Callout: "65,792 stops on a snake".

## Length

- **Words:** 317 in 14 voiced beats, counted with a script (hyphenated numbers count as one word). Per beat: 22 / 34 / 9 / 24 / 28 / 17 / 17 / 32 / 16 / 25 / 26 / 23 / 26 / 18.
- **Kokoro:** about 121.5 s (2:02).
- **Peter Baker:** about 175.0 s (2:55) at the model's central rates, a prediction. At the budget's pessimistic bounds (1.4 s per beat boundary, 1.9 words/s on the number beats r05, r06, r07, r09, r10) it is about 186 s. With fallback items 1 and 2 below, the pessimistic case comes back to about 179-180 s.
- **Silent holds included:** r00 1.5 s; r02 1.4 s (0.4 card + 1.0 expand); r03 0.8 s; r06 1.0 s (expand); r10 1.5 s (0.5 + 1.0 expand); r11 2.5 s (the drones, with the title card fading in over the last 1.5 s). Each 4 s scoreboard expand overlaps the beat's last words and the next cut by about 3 s, so it adds only 1.0 s of silence. These need a per-beat hold in `assemble.py`.

**Where the room came from** (v3 was 325 words, 175.4 s, with shorter holds):
- Added: the promise "Two places: size and count" (i1, +5 words), "On a square" (r06, +3), "One huge blob, or many at once" in place of "Size and count" (r11, +4), 3 s of scoreboard expand holds, and 1.5 s more on the r11 hold for the title card.
- Cut (cut-order items 2 and 3 of v3, now applied by default): "where waves meet, the smaller label wins" (r09) is a caption over the min cutaway, and "twenty-five times the CPU" (r06) is a caption, with the scoreboard row still filling on the cue.
- Moved: the thesis sentence "The GPU wins two ways: size and count." (r02, -8 words) is replaced by the i1 promise plus "So: size first." (+3); "Size first" leaves r03 (-2).
- Trimmed: "too" (i1), "One and a half milliseconds" -> "About three milliseconds" (r10, -2), "fifty-eight" -> "fifty-one" (r09, same length).

**Cut order if the Peter Baker take still runs long** (seconds saved):
1. Merge r00 into r01 (about 1-2 s). This needs a 1.5 s silent gap inside one voice clip for the push-in, which `assemble.py` cannot do yet (it holds only at beat ends). So it stays a fallback.
2. Last resort: merge r04 into r05 in words only ("Two blocks: twice. Forty-eight: eighteen times the CPU."), while the floor still shows all three steps (about 5 s).
3. Shorten the scoreboard expands from 4 s to 3 s (about 3 s of silence saved, at some cost to readability).

## Honesty checklist (on screen)

- **Opening.** The CPU is never shown stuck on drone footage. `s_intro.py:189` "81 Mpx per frame" is removed.
  - "would keep up" rests on a committed 1 Mpx row (21,618 blobs, @njit 11.2 ms), a frame larger and busier than any drone frame. It is never "the CPU cannot keep up with the drones".
  - Each CPU card is tagged with its picture and what it shows: "many blobs, small frame: CPU fits" and "giant frame: CPU misses", each "not a drone frame".
  - When the 81 Mpx card is up, the GPU drone panel is frozen, dimmed and tagged "drone frames, 0.59 Mpx". No live GPU counter runs beside the 81 Mpx stopwatch; the GPU's answer on that card is "?".
  - Pure Python bars carry "seeds given": the 24 s and 814 ms leave out seed discovery, while @njit includes it.
- **Thesis.** "the CPU still wins" on one small blob is a measured 16 K px square, one session: CPU 0.152 ms vs the best of all 15 GPU kernels, 0.365 ms. The race in milliseconds is shown on a 16 K px test square chip, never on the drone floor, which keeps "steps, not time". The drone blob itself (119 px) was never timed in committed data.
  - "Every GPU kernel we timed takes at least ~0.35 ms on tiny blobs" (ch06's six-launch floor, measured at a low 900 MHz clock), never "every GPU launch costs 0.35 ms". That a smaller blob widens the CPU's lead is an inference.
- **Scoreboard.** Every row names its picture; session(s), clock and contract go in the r11 footnote.
  - Rows 1 and 2 are one session each, kernel time, both sides given the seed (row 1's best GPU kernel even finds its own).
  - Row 2 says "kernel only, ~3.8x with copies" on the card itself.
  - Row 2's 25.8x is the Act 1 machine; row 3 and the r11 lines are the final machine. The r11 final-machine line on row 2 (~230x, 11 Mpx disk) keeps size and count on the same machine.
  - Row 3 holds the frame size fixed (about 1 Mpx) so that count, not pixel volume, is what changes. It never uses the 81 Mpx frame, whose ratio is mostly size.
  - Row 3 and below cross sessions (CPU Jul 24; GPU Jul 25 and Sep 27) and lead with the RGB contract. The packed-mask figures appear only in the footnote and the faint meter mark.
  - No drone-frame row until a CPU drone benchmark is committed. No "drone scale" label on any test frame.
- **Count is never overclaimed.** Count by itself does not make the CPU miss 33 ms (it fits on 21,618 blobs at 1 Mpx). The claim is only that the GPU's lead grows with count at a fixed frame size. The 81 Mpx frame is tagged "size and count together".
- The CPU baseline is "@njit: label + fill" (`cpu_fill_canonical`: a CCL pass, then a multi-source fill). Never imply that the 1.3 s is the pixel walk animated in r01.
- "1 core used / our CPU code; the laptop has 14 cores", never "the CPU has one core". The 14 is from the i9-13900H spec, not the benchmark JSON. A multi-core CPU baseline was never measured.
- "SM = streaming multiprocessor".
- "steps, not time" on the drone race.
- Every single-blob speed-up names its blob (one 64 Mpx square), its session and "kernel time".
  - On the same square, the 48-block run from start to finish (with copies) is 687.4 ms, which is about 3.8x the CPU. This is on the scoreboard's row 2 tag.
  - Two blocks are scored against one block, in the same session.
- **Timing clocks differ:**
  - ch01-ch05 `kernel_ms` is host perf_counter time around launch + `cuda.synchronize()` (it includes the launch path, not the copies).
  - ch06 and the drone timings are CUDA events. Below about 4 Mpx the 0.33 ms host enqueue of the six launches shows up inside the event-timed floor; on 81 Mpx it overlaps the kernels. No enqueue figure is added on top.
  - So any ch05 vs ch06 ratio shows "20x (40x from a packed mask), same session; ch05 host clock vs ch06 GPU events".
- "24 / 24 SMs" carries "1/3 of each SM: registers".
- The gate's cost is its price per stop (~2 us, measured 1.7-2.2 us; two per ring), not the idle threads. 12,287 idle is a caption only. "The stops are most of the 143 ms", never "= the whole 143 ms".
- The snake chips compare GPU kernels only. The @njit CPU does the snake in 0.31 ms. This is not shown.
- "half" is spoken and shown with "on a square". The snake is said to still pay every gate.
- One form of each number: 25.8x on the caption and the scoreboard (25.76); no spoken figure.
- "Size is won" means on the 64 Mpx square (row 2 is on screen as it is said).
- "~" plus "estimate" on 2,181 and 1,368. 2,181 is kernel time only: a lower bound.
- The meter stays in one session (Jul 24) up to r09: CPU tick 1,346, estimates 2,181 and 1,368, ch05 51.0 ms. The r10 needle (2.96 ms, Jul 25) carries "Jul 25" in its sub.
- 51 ms is "1 fill launch + 2 cleanup", never "one launch" in total. It is shown as over budget (1.5x), never "close".
- ch06 is "no queue, no ring loop", with 5 seams, never "no barriers" or "nobody waiting".
- Big rings overflow the r03 shelf (43% of a 36 Mpx blob goes through the spill tier).
- "2.96 ms" carries "from the RGB image, GPU events"; "1.46 ms" carries "packed mask".
- The drone frame says "labelling 1.2 ms; whole frame about 11 ms of 33 (median)", never a bare "1.2 ms of 33" and never "10 ms".
- No 16,000x card: it mixes pure Python (seeds given) with ch06 across two sessions and two contracts. The scoreboard uses @njit throughout.

## Build notes (short)

**Look:** 2.5D (iso) only. Every new floor effect is built once, in iso: the split into two half tiles, the inflate, both pull-backs, the ring sweep raster, the row dim (one iso strip), the reverse zoom. HUD parts (goal chip, scoreboard, captions, inset, the 16 K px test square chip) are flat screen-space overlays.

**Reuse:**
- `ChainScene`, `machine.Spec` parts, `toy.py` rules
- the c04 gate dots, the c06 strobe, the c07 printer, the c08 seeds and min, the c09 kernel strip
- `s_intro.py`, re-ordered (GPU live first, then the 1 Mpx card, then the 81 Mpx card with the drone panel frozen and dimmed, each card with its own 33 ms line), re-timed to word cues, and re-rendered with the current tracker colours

**New:**
- `fl.mask_raster` (white on black)
- a `zoom_crop` fix (it clamps by height only)
- the inset
- `toy.drone` with keys `drone_px`, `drone_rings4`, `drone_rings8`
- the split floor (`geo.Frame.floor_l/r`) as two iso half tiles, used twice
- the 16 K px test square chip with two ms timers (r02)
- the ring sweep as raster dots via `toy_raster(dots=)`, plus the shelf spill chute
- meter ticks, the needle and the faint packed-mask mark on the goal chip
- the i1 drone-panel freeze and dim, and the two SIZE / COUNT "?" chips that fly from the i1 "?" bar to the goal chip
- **the scoreboard**: a flat card with three rows, collapsed and expanded (4 s) states, grey and teal winner chips, one short tag per row, the r11 extra lines (row 2 final machine, size and count together) and the footnote line; the SIZE / COUNT corner tag with its tick
- the i1 1 Mpx card (`png_blocks` picture with its own 33 ms line)
- `Spec.gate_ticks` with 2 ticks per ring and a per-stop "~2 us" label; two tallies (square, snake)
- the room ripple, the hand prop, the tag rack
- the 81 Mpx frame drawn transposed (repo orientation) from r07 on
- the row dim, the gate-to-seams morph, the reverse zoom, the title card faded over the drones
- `state.py` keyed by beat name
- per-entry media and voice dirs plus per-beat holds in `assemble.py`
- **record a per-frame total in `make_drone_frames.py`** (from t0 to the end of `gpu.step`, so the recolor call's host wall is included) plus its median, p90 and max in `timing.json`. This needs a GPU run, so it waits for the user. Until then the r11 caption says "median" and the claim check uses the lower-bound sum.
- `numbers.py`:
  - new keys from `multi_block_20260720T012243Z.json:sq_8000_center` for r03-r06 (`v2_kernel_ms`, `dual_global_kernel_ms`, `multi_kernel_ms`, `multi_total_ms`, `conn8_kernel_ms`, `njit_ms`) and `serpentine_256` (`v2`/`dual_global`/`multi` ms, `conn8_levels`);
  - `ch05_ms` for r09 from `overview_20260724T234423Z.json:rows[png_blobs].cells.ch05_split_L8.ms` (51.03); `ch05_ms_jul25` (58.51) only for the r10 ratio; `ch06_rgb_ms` (2.957) and `ch06_mask_ms` (1.455) from `runs_20260725T161448Z.json`, plus `ch06_rgb_ms_alt` (3.030) from `runs_20260725T160000Z.json`;
  - scoreboard keys: `sb_small_cpu_ms` (`sq_256.njit` 0.1516), `sb_small_gpu_ms` (`sq_256.ch05_fused_L8` 0.3654, asserted to be the minimum of all GPU cells in the row), `sb_small_1block_ms` (`sq_256.ch01_ring` 2.997), `sb_huge_cpu_ms` / `sb_huge_gpu_ms` (= `njit_ms` / `conn8_kernel_ms`), `sb_huge_total_x` (`njit_ms` / `multi_total_ms`), `sb_huge_final_cpu_ms` (`disk_4000.njit` 269.48), `sb_huge_final_gpu_ms` (`ch06_overview...:disk_4000.ch06_rgb` 1.168), `sb_count_cpu_ms` (`png_blocks.njit` 11.17), `sb_count_py_ms` (`png_blocks.pure_python` 813.5), `sb_count_gpu_ms` (`ch06_overview...:png_blocks.ch06_rgb` 0.482), `sb_count_blobs` (21,618), `sb_count_1blob_cpu_ms` (`sq_1024.njit` 4.48), `sb_count_1blob_gpu_ms` (`ch06_overview...:sq_1024.ch06_rgb` 0.458), `sb_both_cpu_ms` (`png_blobs.njit`), `sb_both_gpu_ms` (= `ch06_rgb_ms`);
  - the drone frame total (`filter` + `kernel` + `gpu_wall`, and later the recorded per-frame total);
  - the 14-core entry with its source.
- `numbers.py` claim checks for every sentence:
  - "same thirty-three milliseconds" (the stated budget, 1000/30)
  - "would keep up" (`sb_count_cpu_ms` < 33, and the drone frame's 589,824 px and max 1,210 blobs are below the 1 Mpx row's 1,000,000 px and 21,618 blobs)
  - "one point three seconds" (1,346)
  - "Two places: size and count" (`sb_huge` ratio > 1, `sb_count` ratio > 1, and `sb_count` ratio > `sb_count_1blob` ratio)
  - "on one small blob, the CPU still wins" (`sb_small_cpu_ms` < `sb_small_gpu_ms`, same session)
  - "over twice the CPU" (2.456 > 2)
  - "one SM of twenty-four"
  - "twice as fast" (2.036)
  - "twenty-two SMs stay dark"
  - "they lose to one" (143.4 > 61.7)
  - "stops twice" (2 grid.sync per level)
  - "two microseconds each" (README prices 1.7-2.2 us; the 2.18 us division agrees)
  - "On a square, half the rings" (4,001 / 8,001 on `sq_8000_center`)
  - "Size is won" (25.76 > 1, row 2 filled)
  - "twenty-five hundred blobs" (2,522)
  - "slower than the CPU" (2,181 > 1,346)
  - "about CPU speed" (1,368 vs 1,346, within 5%)
  - "run out at sixty-four" (6 label bits)
  - "every blob in one fill launch" (split_L8: one cooperative fill launch, 2 plain)
  - "fifty-one milliseconds" (51.03 > 33, same session as the CPU tick)
  - "About three milliseconds" (`ch06_rgb_ms` 2.957 and `ch06_rgb_ms_alt` 3.030 both within 2.5-3.5)
  - "One small blob: the CPU. One huge blob, or many at once: the GPU." (rows 1-3 winners, and the row 2 final-machine line)
  - "every frame" (filter + kernel + gpu_wall median 10.7 ms < 33; once the per-frame total is recorded, its max < 33)

---

## Open questions for the user

1. **Row 3 picture and number.** Default: the 1 Mpx `png_blocks` frame (21,618 blobs), ch06 from RGB, ~23x across sessions, so it matches the i1 card. Alternatives: `random_1000` (1 Mpx, 47,655 blobs, ~35x RGB across sessions, 11x same session), or the same-session ch05 figure for `png_blocks` (9x). Which do you want?
2. **Row 2 final-machine line in r11** (11 Mpx disk, ~230x, two sessions). It stops the viewer reading 25.8x vs 23x as "count beats size", but it also shows that size is the GPU's biggest win, by about 10x over count. Keep it, or keep row 2 at the Act 1 machine only, with a "final machine is faster" tag?
3. **"Size and count together" line in r11** (81 Mpx, ~450x RGB, two sessions). Keep the ratio, show times only ("1,346 ms vs 2.96 ms"), or drop the line?
4. **Spoken r10 number.** The default now says "About three milliseconds" (RGB, like the CPU). Revert to "One and a half milliseconds" with a "packed mask" caption?
5. **A drone-frame row.** Rough only, and not same-session: the CPU was timed in scratch on 5 real drone frames today, with clocks about 2.5x slow and one core busy (@njit label + fill 5.1-7.7 ms; label only about 1.2 ms; pure Python 78 ms). Scaled to July clocks, that is an estimate of about 2-3 ms. The GPU kernel (1.20 ms) is a committed number from another session, CUDA events only, on the raw mask before the size filter. Mixed this way, the GPU looks about 2-6x faster, and both fit in 33 ms. Do you want a committed benchmark (a small script that writes a JSON next to `drones_labels/timing.json`, run on a quiet machine, both sides in one session) so a fourth row "one drone frame" can be shown? Until then it stays off screen.
6. **Per-frame drone total.** Re-run `make_drone_frames.py` (GPU, about 0.8 GB) to record a per-frame total and its max, so "every frame" is checked against the max and not a sum of medians?
7. **i1 wording.** Keep "would keep up" (an inference from the 1 Mpx row), or switch to "keeps up" once the drone benchmark is committed?
8. **Act labels.** The promise in i1, the SIZE / COUNT corner tags and the turn line carry the two acts, and the word "act" is never spoken. Clear enough, or add a short silent card between r06 and r07?
9. **Verdict "?" in i1.** Keep the GPU bar as "?" until the needle crosses in r10 (the default), or show "GPU 2.96 ms" up front?
10. **Say "SMs" aloud** (Kokoro's reading is untested and may need "S-Ms"), or "multiprocessors"?
11. **Length.** Peter Baker is predicted at about 175 s central and about 186 s pessimistic. Cut-order items 2 and 3 of v3 are now applied. Do you approve the remaining fallbacks in advance (merge r00 into r01, which first needs a mid-clip silence in `assemble.py`; then merge r04 into r05 in words)? Kokoro runs about 2:02; do you want longer silent holds to bring it to about 2:15?
12. **Thin snake** as the picture of the gate's price (r05), left open at 8 ways with "On a square" spoken in r06, and paid off in r10 by a 1 s visual-only flash. OK?
13. **Gate in ch06:** the amber gate shrinks into 5 thin seams between the 6 kernel boxes. Clear enough, or fade it out with a caption "5 seams, no ring loop"?
14. **Seam from the intro into the zoom:** a fade through black (works today, the default), or a continuous push-in out of the intro's filter panel (needs an edit to `s_intro.py`)?
15. **Tags rung floor:** keep the toy `two_blobs`, or use two real drone blobs from the same mask frame?
16. **Voicing:** re-voice in Kokoro first, and voice Peter Baker (317 words, roughly 1,600+ ElevenLabs credits) only after you approve the Kokoro cut?

## Corrections to the outline

- SM stands for "streaming multiprocessor" (not "semi"). It is like a core, but a much wider one, with 128 lanes and room for 1,536 threads.
- The RTX 4060 Laptop has exactly 24 SMs (not "at least 24").
- One block is not "a single core like the CPU". It is 256 threads that paint a whole ring together, and on a giant 64 Mpx square it is already 2.46x the CPU. On a small blob, though, it is slower than the CPU, and so is every GPU kernel we have: the video says so and makes it row 1 of the scoreboard.
- The CPU is not single-core: this laptop's i9-13900H has 14 cores. Our CPU code just uses one of them, so we say "on one core". A multi-core CPU version was never measured.
- "All the blocks" is 48 blocks, 2 per SM on all 24 SMs. Register use caps the cooperative grid there, so each SM is only one third occupied.
- With 48 blocks the problem is not that threads wait (idle threads cost nothing). It is that each ring ends with two chip-wide gates of about 2 us each. On a thin snake that is 65,792 stops, so 48 blocks (143 ms) and even 2 blocks (128 ms) lose to 1 block (62 ms), whose gate is cheap.
- 8-connectivity halves the rings only on a square (a disk gets about x0.71), and it does not help a thin snake (0.80x, slower). Its real job is fewer gates and wider rings. It does not change the answer here: our blobs are 8-connected anyway.
- The huge blob is 64 Mpx (8000 x 8000), not 62 Mpx. The 2x / 18x / 25.8x results were measured on it in one session, as GPU kernel times without the copies to and from the card. With copies, the 48-block run is about 3.8x the CPU.
- The three-pane intro's CPU times (24 s, 1.3 s) are for the 81 Mpx test frame. The 24 s pure Python figure was given its seeds; the 1.3 s @njit figure includes finding them.
- The 81 Mpx frame's huge GPU lead is mostly size (13.45 M red px), not count: it has the fewest blobs of the many-blob test frames. Count is shown at a fixed 1 Mpx frame size, where the GPU's lead grows from about 10x (one blob) to about 23x (21,618 blobs).
- Not every step improves: on the real test frame, one launch per blob is slower than the CPU (~2,181 ms, an estimate). That becomes the twist between the acts.
- The final kernel chain still has barriers: 5 seams between 6 kernels, instead of two gates per ring inside a loop. So we say "no queue, no ring loop", not "no barriers".

## The CPU on drone frames (checked 2026-10-14)

- **Committed data:** no CPU labeller was ever timed on a drone frame (576 x 1024 = 0.59 Mpx, 570-1,210 blobs). The closest committed picture is `png_blocks` (1000 x 1000, 21,618 blobs, 387,587 red px) in `overview_20260724T234423Z.json`: pure Python 813.5 ms, @njit 11.17 ms. So textbook Python misses a 33 ms frame by about 25x, while compiled CPU code fits. The opening uses exactly this card. It is a test frame, never labelled "drone scale": a drone frame has about 13-18 K red px.
- **Scratch only (not committed, not shown; rough):** @njit label + fill on 5 real drone frames took 5.1-7.7 ms today (label only 1.1-1.3 ms); pure Python on frame 120 took 77.7 ms. A same-day rerun of `png_blocks` gave 27.7 ms against July's 11.2 ms, so today's clocks were about 2.5x slow, with one core busy with another job. Scaled to July, a drone frame is roughly 2-3 ms (an estimate). The CPU cost follows the pixel count, not the blob count: one 15 px blob on the full frame takes 4.4 ms, 936 blobs take 5.1 ms.
- **What this means for the story:** on one drone frame the GPU wins only modestly. Against the committed GPU kernel (1.20 ms, another session, CUDA events, the raw mask before the size filter), that is a rough 2-6x, and both fit in 33 ms. The video therefore never claims a drone-frame race and never shows a CPU ratio on the drones. It shows count at a fixed 1 Mpx frame size (row 3), and says only that the GPU labels every drone in every frame.


---

## What changed from v2

- Row 3 (COUNT) moved off the 81 Mpx frame onto the 1 Mpx png_blocks card from i1 (21,618 blobs): CPU 11.2 ms vs ch06 RGB 0.48 ms, ~23x, two sessions. A sub line shows one 262 K px blob on a frame of the same size (sq_1024) at ~10x, so the lead visibly grows with count at a fixed frame size (issues 2, 14).
- The 81 Mpx frame is now tagged 'size and count together' on the goal chip and the floor. Its result appears in r11 as its own line ('81 Mpx, 2,522 blobs: 1,346 vs 2.96 ms, ~450x, two sessions, RGB'), not as the COUNT row (issues 2, 5, 14).
- The 'at drone scale' bridge line is removed. No test frame is called drone scale, and the drones get only 'labelling 1.2 ms; whole frame about 11 ms of 33 (median)' with no CPU ratio (issue 1).
- Every ch06 figure leads with the RGB contract, like the CPU baseline. The r10 needle lands on 2.96 ms, with a faint 1.46 ms packed-mask mark, and the spoken r10 number becomes 'About three milliseconds' (2.957 / 3.030 in the two Jul 25 sessions) (issue 5).
- In r11, row 2 gets a second line with the final machine on the largest single blob timed with ch06: an 11 Mpx disk (disk_4000), 269.5 vs 1.168 ms, ~230x, two sessions. Each row is also tagged 'Act 1 machine' or 'final machine' (issue 15, with corrected data).
- The row 2 tag now reads 'kernel only, ~3.8x with copies' (48-block multi_total_ms 687.4) (issue 9).
- i1: when the 81 Mpx card comes up, the live GPU drone panel freezes and dims, and is tagged 'drone frames, 0.59 Mpx'. The only BUT is the CPU stopwatch passing 33 ms beside a GPU '?' on the same card. Card tags read 'many blobs, small frame: CPU fits' and 'giant frame: CPU misses' (issues 3, 17, 14).
- i1 now ends with the promise 'Two places: size and count.' (+5 words). Two SIZE / COUNT '?' chips are seeded there. r02 ends 'So: size first.' instead of the thesis sentence, and r03 drops 'Size first' (issue 20).
- i1: 'would keep up too' becomes 'would keep up', since nothing spoken comes before the 'too' (issue 21).
- r02: the race in milliseconds runs on a separate 16 K px test square chip. The drone floor keeps 'steps, not time' (issue 8).
- r02 Result: 'every GPU launch costs 0.35 ms' becomes 'every GPU kernel we timed takes at least ~0.35 ms (ch06 six-launch floor, at a low 900 MHz clock)'. The smaller-blob conclusion is marked as an inference (issue 7).
- r05: the sub becomes '~2 us per stop (measured 1.7-2.2 us): the stops are most of the 143 ms'. The circular '= the whole 143 ms' is dropped, and the 2.18 us division is called a consistency check (issue 12).
- r06 narration scopes the fix: 'So pass fewer gates: step diagonally too. On a square, half the rings, half the gates.' (issue 18).
- r06: one form of the number, 25.8x on both the caption and the scoreboard. 'twenty-five times the CPU' is now a caption (cut-order item 3 applied) (issues 11, 19).
- r09 uses the overview session's ch05 split_L8, 51.0 ms (spoken 'fifty-one milliseconds', 1.5x the budget), so the meter stays in the CPU's session. 58.51 ms is kept only for the r10 same-session ratio (issue 10).
- r09: 'where waves meet, the smaller label wins' is now a caption over the min cutaway (cut-order item 2 applied). The narration says 'every blob in one fill launch' (issue 19).
- r10: '~1.8 ms with launches' and the 32x low end are dropped. The ch05 -> ch06 ratio is shown as '20x (40x from a packed mask), same session; ch05 host clock vs ch06 GPU events'. The checklist now says the 0.33 ms enqueue is inside the event-timed floor (issue 6).
- r11 narration: 'One small blob: the CPU. One huge blob, or many at once: the GPU. Every drone, every frame.' (18 words). It says 'many' rather than 'thousands', because drone frames hold 570-1,210 blobs (issue 16).
- r11 drone frame total corrected: filter + kernel + gpu_wall = 10.74 ms median, a lower bound ('about 11 ms of 33 (median)'). The p90 note now includes the kernel p90 and is marked as not a true p90. A build note records a per-frame total and max in timing.json, and the claim check is updated to match (issue 4).
- The scoreboard expands for 4 s per fill and shows only picture / CPU / GPU / winner plus a short tag. Session, clock and contract details move to one footnote line on the r11 full card. Each expand adds 1.0 s of silent hold and overlaps the rest (issue 22).
- The title card fades in over the drones inside a 2.5 s r11 hold, so it is budgeted explicitly (issue 19).
- The r07 inset now reads 'a drone frame: ~900 blobs of ~15 px'. It is not equated with the 81 Mpx frame, and an honesty note says that frame's CPU miss is mostly pixel volume (issue 2).
- Open question 3 (now 5) labels the drone 2-6x as rough: scratch CPU on ~2.5x-slow clocks plus a July-scaled estimate, against a committed GPU kernel from another session on the raw mask (issue 13).
- Length recounted with a script: 317 words in 14 beats. Kokoro about 121.5 s, Peter Baker about 175.0 s central and about 186 s pessimistic. The cut order is rebuilt: merging r00 into r01 stays a fallback because it needs a mid-clip silence (issue 19).
- numbers.py gets new keys (ch05_ms from the overview, ch06_rgb_ms, ch06_rgb_ms_alt, sb_huge_total_x, sb_huge_final_*, sb_count_* on png_blocks, sb_count_1blob_*, sb_both_*) and new claim checks ('Two places: size and count' including the count-trend check, 'On a square', 'fifty-one', 'About three', the revised reveal, 'every frame').

## Open questions for the user

1. Row 3 picture and number: png_blocks (default, matches the i1 card; ~23x ch06 RGB across sessions, or 9x same-session ch05) or random_1000 (47,655 blobs; ~35x RGB across sessions, 11x same session)?
2. Keep the r11 row 2 final-machine line (11 Mpx disk, ~230x, two sessions)? It keeps size and count on the same machine, but it shows that size is the GPU's biggest win, about 10x more than count.
3. The r11 'size and count together' line for the 81 Mpx frame: keep ~450x (RGB, two sessions), show times only (1,346 vs 2.96 ms), or drop it?
4. Spoken r10 number: keep 'About three milliseconds' (RGB, like-for-like with the CPU) or revert to 'One and a half milliseconds' with a 'packed mask' caption?
5. Commit a same-session CPU vs GPU drone-frame benchmark (a JSON next to drones_labels/timing.json, run on a quiet machine) so a 'one drone frame' row can be shown? The scratch 2-6x is rough and mixes sessions.
6. Re-run make_drone_frames.py (GPU) to record a per-frame total including the recolor host wall, plus its max, so 'every frame' is checked against the max?
7. i1 wording: keep 'would keep up' (an inference) or switch to 'keeps up' after a committed drone benchmark?
8. Are the i1 promise, the SIZE / COUNT corner tags and the turn line enough to mark the acts, or add a short silent card between r06 and r07?
9. i1 verdict: keep the GPU bar as '?' until the r10 needle crosses (default), or show 'GPU 2.96 ms' up front?
10. Say 'SMs' aloud (Kokoro may need 'S-Ms') or 'multiprocessors'?
11. Length: Peter Baker about 175 s central and about 186 s pessimistic. Approve the remaining fallbacks in advance (merge r00 into r01, which needs a mid-clip silence in assemble.py first; then merge r04 into r05 in words)? Add longer Kokoro holds to reach about 2:15?
12. Thin snake as the gate's picture (r05), with 'On a square' spoken in r06 and a 1 s visual-only payoff in r10: OK?
13. Gate in ch06: shrink it into 5 seams (default), or fade it out with the caption '5 seams, no ring loop'?
14. Intro-to-zoom seam: fade through black (default) or a continuous push-in out of the filter panel (needs an s_intro.py edit)?
15. Tags rung floor: toy two_blobs, or two real drone blobs?
16. Voicing: Kokoro first, then Peter Baker (317 words, roughly 1,600+ ElevenLabs credits) only after you approve the Kokoro cut?

## Appendix: the CPU on real drone frames (scratch measurement, 2026-10-14)

Not committed and not on screen: measured once by the revision workflow to
check the opening. `cpu_oracle.cpu_fill_canonical` (the overview's @njit
column: label + fill, seeds found on the CPU), warm-up once, median of 5,
on `assets/drones_mask` frames (576 x 1024):

| frame | blobs | red px | @njit label + fill, ms | @njit label only, ms |
|---|---|---|---|---|
| 0 | 728 | 17,821 | 7.66 | 1.29 |
| 60 | 595 | 13,252 | 6.12 | 1.13 |
| 120 | 936 | 18,350 | 5.11 | 1.31 |
| 180 | 1104 | 15,981 | 6.17 | 1.20 |
| 239 | 925 | 14,672 | 5.09 | 1.18 |

Pure Python on frame 120 (seeds handed in): 78 ms. The machine ran
about 2.5x slower than in July that day (`png_blocks` @njit 27.7 ms vs
11.2 ms in the overview), so July-condition numbers would be lower still.
The GPU labelling kernel on the same frames: 1.20 ms median
(`assets/drones_labels/timing.json`). So on drone frames the compiled CPU
keeps up with 33 ms; the GPU wins as frames and blob counts grow.
