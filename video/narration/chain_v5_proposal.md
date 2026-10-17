# The chain cut, v5: tiles replace the run-based beat

Status: PROPOSAL on the user's decisions of 2026-10-17, every CPU-vs-GPU
number re-made like for like in one session, after two critics' reviews
(answered below). Built and rendered on 2026-10-18 (278.4 s; see
`video/HANDOFF.md` section 11). The script is `script_chain.md`.
Every number below is pinned in `scenes/chain/numbers.py`: 115 claim
checks, all passing (`cd video && uv run python -m scenes.chain.numbers`).

Base: v4 (`chain_v4_proposal.md`, 16 beats, 217.5 s with Kokoro, paced).
Every v4 rule stays: at most three captions per beat, a caption sub of at
most four words, amber only for waiting, every number from committed JSON
through `numbers.py`, "~" on estimates, "kernel time" where it applies,
no "chapter".

New for v5:
- No "CCL", "runs", "spans" or "union-find" (איחוד-חיפוש), spoken or
  shown, and no GPU number from ch08's dense-tile path.
- Like for like: every CPU does the work of the GPU it is compared with,
  with the same setup (arrays kept between calls, reset where the GPU
  resets its own).
- On-screen CPU-vs-GPU factors keep two significant figures (4.4×, 11×,
  340×): the CPU's round medians spread by up to 19%.
- A flashing element flashes at most twice per second (r05b's
  photosensitivity (רגישות לאור) rule, made stricter), and large areas
  never flash.

---

## The user's decisions (2026-10-17)

All binding.

| # | Decision | Where it lands |
|---|---|---|
| 1 | Every CPU-vs-GPU number from one new session; no July numbers | `v5_20261017T192640Z` (V5). Act 2's estimates are re-made in it; `numbers.py` reads no other CPU-vs-GPU session. |
| 2 | The CPU does exactly the GPU's work: the same neighbours, the same output; smaller ratios accepted | Each GPU cell has a CPU twin (below), checked for outputs, neighbours and setup. Spoken lines change for it in i1, r03b, r05, r08 and r09, and every scoreboard factor. |
| 3 | The finale says both sides of the small blob aloud | r14: "In kernel time, the GPU now; with the copy to the card, still the CPU." Both hold (4.4× and 3.5×). |
| 4 | Title card "flood fill, from pixels to tiles" (plus the repo name) | r14's last seconds. |
| 5 | Kokoro `am_adam`, paced; Cairo renders in parallel on the CPU, the GPU measures; act 3 at the end, mechanism and numbers; at most 5 minutes | About 287 s (4:47) planned; 278.4 s (4:38) rendered. |

Defaults adopted with them:
- r13 keeps "once the frame is on the card" (with the copies the frame
  takes 54.8 ms; the r14 footnote says so).
- r11 compares against r06's kernel, the last one the viewer saw.
- Act 2's ch05 is timed at full clocks.
- The drones are timed without ch08's dense-tile path. V5 goes further:
  every tile-wave number (the drones and the frame) is ch08's first
  version, which has no dense-tile path at all (see the critics' F3).
- The frame floors are transposed, so one orientation holds everywhere.

Pitfalls the critics named, and how v5 meets them:
- **Undefined terms.** Thread and block (r02: "two hundred fifty-six
  workers, called threads, in one block"), warp (r10: "thirty-two
  threads that run in step") and registers (r10: "the thread's own
  memory, closer than shared memory") are defined in plain words where
  first spoken.
- **r11 overloaded.** No replay of later rounds after the first gate
  flash; one pill change, no caption with it.
- **Strobe.** No flashing above twice per second; the frame's 11,609
  seeds fade in as dots, with no white rings.
- **"Label" in two senses.** One sense only: a blob's own number. Row
  3's tag reads "CPU: same labels".
- **Orientation.** The 900 px PNGs are PNG order, act 3's arrays [x, y];
  v5 transposes the PNG side once (checked).
- **The crop in r12** is its own image, shown with its own border.

---

## The critics' second review (2026-10-17), and what changed

Two critics read the evening draft (session `v5_20261017T182449Z`). Every
"must" and "should" is applied; so is every "could". None is rejected.
Where a finding offered two fixes, the choice is named.

Critic 1 (the numbers):

| # | Finding | Done |
|---|---|---|
| F1 must | The CPU twins allocated fresh full-image arrays on every call, so first-touch page faults (חריגות דף) were on the CPU's clock; the GPU sides allocate once | The twins (`overview/cpu_twins.py`) run each oracle's loop on arrays kept between calls, reset before the clock or on it as their GPU side resets its own. Re-run as one new session. |
| F2 must | ch07 keeps no depth map, but its twin built one | A new twin, `cpu_fill8`: the 8-neighbour BFS without depth. Every GPU cell and twin now records its outputs and reset scope, and a check requires the twin's outputs within the GPU's. |
| F3 should | The drones' engine was the default with its dense lists empty, not "the pure wave path" | Every tile-wave number is ch08's first version (`OLD_PATH`: no K2b or K4b launch, the hybrid K3), the frame's too. The default is timed beside it on the frame, reference only (2.25 vs 2.21 ms). |
| F4 could | Two more blocks logged the power cap; the ch05 probe is in no JSON | Named below (Clocks), and the probe is marked unpinned. |
| F5 could | `card_ms` truncates 10-100 ms, so the footnote's copy times read low | The footnote uses `card_ms_note`, which rounds (44.3, 54.8 ms). Spoken values keep the truncation. |
| F6 could | ch08's per-pixel label map is an off-clock twin; the CPU builds its own on the clock | Disclosed in "What v5 does not claim". |
| F7 could | "+3 words" for r08; "3-6×" | r08 is +3 words against v4 and +2 against the morning draft; the range now reads 4.2-9.5×. |

Critic 2 (the viewer):

| # | Finding | Done |
|---|---|---|
| F1 should | r09 never says the GPU now beats the CPU, nor that it misses the budget | r09: "Faster than the CPU. But over budget: every red pixel still rides the queue." Row 3's pill lands on "Faster". |
| F2 should | Row 1's flip changes four things at once, and the tag is drawn nowhere | Two steps: at "And one small blob?" the CPU cell and tag change and the GPU cell and winner go to "?"; at "now" the GPU cell and "GPU 4.4×" land. Tags are drawn under the row names. |
| F3 should | "Block" is never put into words | r02: "...workers, called threads, in one block, paint a whole ring at once." |
| F4 should | r10 is the densest beat; the third cube lands under "tiles" | The registers sentence is split off; the third cube lands as the warp card folds back (the first of the two places offered). |
| F5 should | The i1 drone card shows a kernel-time GPU win that r02 then contradicts | The drone card shows the CPU against the budget and the GPU as "?"; r14's caption 3 gives the drones' time with the copies (1.26 ms worst). |
| F6 could | The SM close-up prints "32 lanes" while the voice says threads | The partitions read "32" only (build). |
| F7 could | r11's "its ... its neighbours" after a beat cut | "Only the tile's border bits go out, into the tiles around it". |
| F8 could | r11 replays rounds 6-8 for no reason | The replay is dropped. |
| F9 could | "CPU: labels only" sounds like less work | "CPU: same labels". |
| F10 could | "copy" is spoken without a bridge | "with the copy to the card", echoing r13. |
| F11 could | "4 trips / then nothing changes" reads as five | "4 trips" / "the last changes nothing". |
| F12 could | 11,609 white rings over the whole frame | The seeds fade in as dots over at least 1 s. |

What the re-run moved, against the evening session:

| number | evening (`182449Z`) | V5 (`192640Z`) |
|---|---|---|
| i1, the CPU on the frame | 274 ms ("over a quarter of a second") | 225 ms ("over a fifth of a second") |
| r02 / row 1 | CPU 4.3× | CPU 4.3× |
| r03b | 1.5× ("beats the CPU") | 1.1× ("a little faster than the CPU") |
| r05 | 12× ("Twelve") | 8.5× ("Eight and a half") |
| r06 / row 2 | GPU 14× | GPU 11× |
| r07, r08 | the CPU by 5.3× and 3.1× | the CPU by 7.2× and 4.2× |
| r09 / row 3 | GPU 5.7× | GPU 3.8× |
| r11 / row 2 | GPU 450× | GPU 340× |
| r13 / row 3 | GPU 120× | GPU 100× |
| r14 / row 1 | GPU 4.3× (0.19 vs 0.044 ms) | GPU 4.4× (0.192 vs 0.044 ms) |
| the drones | CPU 0.67 ms, GPU 0.099 ms (default engine, dense lists empty) | CPU 0.679 ms; GPU not shown (0.090 ms, first version) |

---

## The data behind v5

**Every CPU-vs-GPU number: one session**, `v5_20261017T192640Z` (V5),
benchmark `src/flood_fill_cuda/overview/bench_v5.py` with
`overview/cpu_twins.py`, both at the committed HEAD
(`script_matches_head`).
- GPU: kernel-only CUDA events, interleaved rounds per scene.
- CPU: `@njit` on one core, after all GPU work and 60 s of idle, 5
  interleaved rounds, each behind a gate that waits for a quiet host.
- Every GPU output matched the CPU's, in every scene and on all 240
  drone frames, and every twin's arrays equal its oracle's.

**The CPU twins** (`rules.cpu_twin`, `rules.cpu_meta`; checked for every
GPU cell). Each runs its oracle's loop on arrays allocated once per
image:

| CPU twin | what it keeps | reset | GPU cells it is compared with |
|---|---|---|---|
| `cpu_bfs4` | BFS, 4 neighbours: visited + depth | before the clock | ch01 ring and spill (1 block), ch02 (2 blocks), ch03 4-way (48 blocks) |
| `cpu_bfs8` | BFS, 8 neighbours: visited + depth | before the clock | ch03 8-way (diagonals) |
| `cpu_fill8` | BFS, 8 neighbours: visited only | on the clock | ch07 (the tiles) |
| `cpu_label` | labels only, 8 neighbours | on the clock | ch08's first version (the waves), the drones |
| `cpu_label_pre` | labels only, 8 neighbours | before the clock | act 2: the estimates, the loops, ch05 (seeds) |
| `oracle_*`, `cpu_fill_canonical` | the oracles as called (fresh arrays every call); v4's baseline | in the call | none: reference only |

Why the reset differs: ch01-ch05 and act 2's loops reset their arrays on
the device between runs, off their clocks; ch07 clears its reached bits
and ch08 sets up its state inside the timed pipeline. Each twin does the
same. Every GPU cell records its outputs (visited, depth, labels, owner,
paint), and a twin's outputs must be among them (`twin.like_for_like`).

What the kept arrays changed (reference cells, same session): the
oracles as called took 1.49× (4 neighbours, 1,775 vs 1,195 ms on the 64
Mpx square), 1.39× (8 neighbours) and 1.40× (labels, 315 vs 225 ms on
the frame) of their twins. On the small blob the gain is small (0.126 vs
0.121 ms). The label map's reset is 10% of the frame time (225 ms on the
clock, 202 ms before it).

**Clocks.** All 8 timed blocks ran at full clocks: P0 only, memory 8,001
MHz, SM median 2,070 MHz in every block.
- 7 blocks reached full clocks on their first attempt. ch05 on the frame
  did on its second: its first sat in P3 (memory 7,001 MHz) at 41.9 W
  under the software power cap, and the retry put 0.1 s of light load
  between its runs, as the protocol allows.
- Three blocks logged `sw_power_cap` and still met the rule: the huge
  blob's interleaved block (median 21.8 W, peak 39.0 W), the frame's
  waves block (median 39.5 W, SM minimum 2,025 MHz) and ch05's used
  attempt (median 33.4 W).
- ch05 on the frame read 53.9 ms here (50.8-56.1 over its rounds); a
  probe on 2026-10-17 read 48.9-51.3 ms at any spacing. The probe is not
  in a committed JSON (only bench_v5's docstring quotes it), so it is
  context, not a pinned number. "About fifty" holds for both.
- The CPU's round medians spread by up to 19% in this session (the
  labels reset before the clock: 189-226 ms; the 4-neighbour BFS on the
  square: 1,120-1,284 ms). Every shown factor is checked against its
  range over the CPU's rounds.

**Act 2's estimates, re-made in V5.** v4's method (the median of 6
sampled calls, each timed alone by CUDA events, times the call count),
at 8 neighbours, as the CPU's labels:
- One launch per blob: ~1,462 ms (2,522 calls). Two per launch: ~850 ms
  (1,261 calls).
- v4's host clock on the same calls: 1,919 and 1,016 ms.
- The full loops, measured once: 1,647 and 1,147 ms, 1.13× and 1.35× the
  estimates. So the estimates run low, ch04 the most.
- Every form is 4.2-9.5× slower than the CPU's 202 ms: the direction of
  r07 and r08 is safe.

**The mechanism and its counters: the tiles' own sessions.**
- Tile BFS: `tiles_20261016T064724Z` (T7).
- Tile waves: `dense_20261016T054922Z` (D8) and
  `waves_20261015T183401Z` (W8). Their deterministic counters agree. D8
  also timed the first version (its `old_path`, V5's engine): the same
  counters as the default, so the default's floors (WV) describe the
  first version's run.

**The pictures: recorded from the kernels**, `video/assets/tiles/` (TL, WV).
- TL: the 64 Mpx square's 250 x 250 tiles, round by round, and one warp's
  close-up of one tile with its 7 x 7 tile context.
- WV: a 384 px crop of the frame, run as its own image, piece by piece;
  and the whole frame's claim rounds.
- 47 and 31 checks against the CPU oracles (אורקלים), all true.

**Why row 3 is the frame.** v4's row 3 was 1 Mpx of noise
(random_1000). On it, ch08's default sends 1,023 of its 1,024 tiles down
the dense-tile path, which v5 leaves out. On the 81 Mpx frame no tile
takes that path (V5's timed runs, WV and D8 all count 0). So row 3 is
the frame: the same picture as the meter, 2,522 blobs.

---

## The chain at a glance

19 beats (v4 had 16): v4's first 14, four new, and a new finale.

| k | beat | status | rung |
|---|---|---|---|
| 0 | i0_problem | same | |
| 1 | i1_compare | changed | the CPU needs over a fifth of a second (225 ms, labels only) |
| 2 | r00_zoom | same | |
| 3 | r01_cpu | same | |
| 4 | r02_one_block | "workers, called threads, in one block" | the CPU still wins the small blob: 0.121 vs 0.520 ms, CPU 4.3× |
| 5 | r03_giant | same | |
| 6 | r03b_one_sm | changed | "a little faster than the CPU" (1.1×) |
| 7 | r04_two_blocks | same | twice as fast: 2.04× |
| 8 | r05_all_blocks | changed | eight and a half times (8.5×) |
| 9 | r05b_snake | same | |
| 10 | r06_diagonals | same text, new caption | 11× against the 8-neighbour CPU; row 2 |
| 11 | r07_real_frame | same | ~1,462 ms against 202: slower than the CPU |
| 12 | r08_tags | changed | two per launch, faster (~850 ms). But still slower than the CPU: only two seeds, each handed in. |
| 13 | r09_seeds | changed, row 3 fills | 53.9 ms: faster than the CPU (3.8×), but over budget: every red pixel still rides the queue. |
| 14 | r10_tiles | new | So the queue carries tiles; one warp closes a tile in registers. |
| 15 | r11_rounds | new | One gate per round: 4,001 rings become 127 rounds, 101 ms become 3.12. But one blob, one seed. |
| 16 | r12_pieces | new | So every tile finds its pieces; seeds send waves; at the end, the smallest label. |
| 17 | r13_frame | new | The frame: 6 rounds, every blob counted, 2.21 ms once the frame is on the card. |
| 18 | r14_reveal | new text | Huge or many: the GPU. The small blob: the GPU in kernel time, the CPU with the copy to the card. |

Gone: r10_runs and everything only it used: the "runs" form of the VRAM
close-up, the "1 warp" row read and its "coalesced" caption, the six
kernels and their seams on the link, the meter's packed-mask mark, the
teal frame.

---

## Beats that stay unchanged

i0_problem, r00_zoom, r01_cpu, r03_giant, r04_two_blocks, r05b_snake,
r06_diagonals and r07_real_frame: same text, same takes, same pictures.
The final check (2026-10-18) changed three of those pictures, never a
word: r00's "Freeze" is a white outline pulse round the floor, not a
flash of it; r01 has two captions, not three; r05b's last caption lands
on "each".

Their numbers that come from GPU-only comparisons keep their July
sessions, each within one session: 48 KB and 125 KB (r03), SM 2, 1.7x
and "no slower" (r04), the snake and its gates (r05b), +15% (r05). None
of them is set against a CPU.

r04's "twice as fast" is read from V5 (1,097 vs 538 ms, paired 2.04x).
r06's caption and r07's needle move with V5 (below).

---

## Changed beats (acts 1 and 2)

### i1_compare: "over a fifth of a second"

```text
On frames this small, compiled CPU code keeps up. Scale up: eighty-one megapixels, same thirty-three milliseconds. The CPU needs over a fifth of a second. Where does the GPU win? Two places: size and count.
```

- The CPU on the frame, labels only, its label map reset on the clock
  (the whole job): 225 ms (`njit_ms`, V5 `input_blobs.cpu_label.ms`; its
  5 rounds read 214-238). "Over a fifth of a second" holds in every
  round, and "a quarter" would not. It is still nearly 7× the budget.
  v4 said 1.3 s for labels plus a fill per blob (that work now reads
  1,165 ms, reference only).
- The stopwatch on the 81 Mpx card runs to "225 ms"; its stop cue moves
  from "seconds" to "second".
- The drone card: the CPU's bar, 0.679 ms (labels only,
  `drone_cpu_ms`), against the 33 ms budget, and the GPU as "?", as on
  the 81 Mpx card. Caption "0.59 Mpx" / "240 frames, one session".
  "Keeps up" is about the CPU and the budget, so nothing is lost; and no
  kernel-time GPU win on small frames comes before r02's "the CPU still
  wins" (critic 2, F5).
- "keeps up": the CPU's worst drone frame is 0.88 ms
  (`drone_cpu_max_ms`).

### r02_one_block: "workers, called threads, in one block"

```text
So move it onto one SM, its queue in the SM's shared memory: two hundred fifty-six workers, called threads, in one block, paint a whole ring at once. But on one small blob, the CPU still wins. So: size first.
```

- The first spoken "thread" and "block", defined: "workers, called
  threads, in one block" (+5 words against v4; the "threads" cue stays).
  It matches the "1 block" bar and the cube on SM 0.
- The race is like for like: the 4-neighbour CPU BFS, 0.121 ms
  (`small_cpu_ms`), against this very kernel, one block of 256 threads
  with its ring in shared memory, 0.520 ms (`small_gpu_ms`, V5
  `sq_256.ch01_ring`). The CPU wins by 4.3× (4.26-4.44 over its rounds).
  The bar reads "1 block".
- Scoreboard row 1: "one small blob: CPU 4.3×", tag "CPU: 4 neighbours".

### r03b_one_sm: "a little faster than the CPU"

```text
Now one block is a little faster than the CPU. But that's one SM of twenty-four.
```

- 1,195 / 1,097 = 1.09 (`big_1block_x`; 1.02-1.17 over the CPU's
  rounds), against the 4-neighbour CPU on kept arrays. "Beats the CPU"
  (the evening draft, 1.5×) overstated it once the CPU stopped paying
  its allocations; "a little faster" says the margin.
- Caption: "1.1×" / "vs CPU, kernel time", cued at "faster".

### r05_all_blocks: "Eight and a half"

```text
So use every SM: forty-eight blocks, two on each. Eight and a half times the CPU. Each colour is one block: they share one queue, so their pixels mix, and small rings keep only a few busy. We also tried other threads per block and block counts: on a smaller square, the best gained fifteen percent. We kept forty-eight by two hundred fifty-six.
```

- 1,195 / 140.6 = 8.5 (`big_48_x`; 8.0-9.1, within 8% in every round),
  against the 4-neighbour CPU. Caption "8.5×" / "kernel time".

### r06_diagonals: same text, the CPU steps diagonally too

- From here the huge blob's CPU is the 8-neighbour BFS, with its depth
  map (ch03 keeps one): 1,126 / 101 = 11.1 (`big_8ways_x`; 9.9-11.7).
- Caption "11×" / "CPU: 8 neighbours too". Scoreboard row 2: "one huge
  blob: GPU 11×", tag "CPU: 8 neighbours" (v4: 25.8×).

### r07_real_frame: same text

- ~1,462 ms (`est_ch03_ms`, V5 `input_blobs.est_ch03_conn8`) against the
  labels-only CPU's 202 ms (`count_cpu_pre_ms`, its reset before the
  clock, as act 2's loops reset theirs): the CPU by 7.2× (6.5-7.8).
  Slower by every form of the number (host clock 1,919 ms, the full loop
  1,647 ms).
- Caption "~1,462 ms" / "estimate".

### r08_tags: "faster. But still slower than the CPU"

```text
So tag each queue entry with its blob: one spare bit of the entry's word, in global memory. Each new pixel inherits the tag, and the tag picks its colour: its label. One bit tells two blobs apart: two per launch, faster. But still slower than the CPU: only two seeds, each handed in.
```

- ~850 ms (`est_ch04_ms`, V5 `input_blobs.est_ch04_multi8`, 8 neighbours)
  against 202 ms: the CPU by 4.2× (3.8-4.5). "About CPU speed" (v4) is
  false.
- "faster": two per launch beats one per launch by every measure
  (estimates paired 1.72×; host clock 1,016 vs 1,919; full loops 1,147
  vs 1,647).
- "But still slower than the CPU": by every measure, over the CPU's
  slowest round. The cause is said with it: "only two seeds, each handed
  in", which r09's "So the GPU finds its own seeds" fixes.
- Picture: the two blobs now fill 8 ways (the estimate's kernel); the
  needle moves at "faster" and settles well right of the CPU tick, not
  beside it. The floor play ends at "faster" (was "speed").

### r09_seeds: "Faster than the CPU. But over budget", row 3 fills

```text
So the GPU finds its own seeds: every blob in one fill launch, about fifty milliseconds. Faster than the CPU. But over budget: every red pixel still rides the queue.
```

- 53.9 ms at full clocks (`ch05_ms`, V5 `input_blobs.ch05_split_L8`, its
  block's second attempt). "About fifty" holds; v4's 51.0 was July's
  host clock.
- "Faster than the CPU" (critic 2, F1): 202 vs 53.9 ms, 3.8×
  (`count_ch05_x`; 3.5-4.2), in every CPU round. Row 3's pill lands on
  these words: "many blobs: GPU 3.8×", tag "CPU: same labels".
- "But over budget": 53.9 ms against 33. The caption "53.9 ms" / "still
  over 33 ms" lands at "budget". COUNT stays unwon (no tick) until r13,
  and row 3 has a value to jump from in act 3.
- "every red pixel still rides the queue" stays the cause that act 3
  fixes.

---

## Act 3: the tiles

Four beats. The first two follow the tile BFS from one tile to the giant
square; the last two take the same tiles to the whole frame.

**One orientation for the whole cut.** The kernels index [x, y] with x
the PNG column, and lane i = x0 + i. The act-3 arrays drop onto the
floors as (row, col), so a lane is a screen row. The 900 px frame PNGs
are in PNG order: v5 transposes them once at the source
(`state.picture()` returns `.transpose(1, 0, 2)`; the intro's frame card
the same). Then every floor from i1 to r14 shows one frame, the zooms in
r07-r13 stay continuous, and the warp card's rows are the tile's lanes.
Checked: the frame's claim-round samples match the picture with
intersection over union 0.78 transposed, 0.10 as is.

Shared devices:
- **The warp card** (new helper `warp_closeup`, a sibling of `sm_closeup`
  and `vram_closeup`): it grows out of one "32" partition of r02's SM
  close-up (`dz.sm_detail`, whose partitions lose the word "lanes"), so
  the picture defines the warp. 32 rows of 32 bit cells: a red cell is a
  red pixel, a blue cell is reached. One `ImageMobject` whose pixels
  change per sweep (as `Floor.show_raster`), plus a frame, a row bracket
  and a column of 32 vote dots. No title: the caption names it.
- **Tile tokens**: in act 3 a queue token carries a tiny 2 x 2 grid (a
  tile); in r12-r13 the tokens are pieces (a tile token with one cell
  lit). `mc.Spec` gets a `token_kind`; tokens are textures on the token
  top.
- **Three blocks per SM**: both act-3 kernels launch 72 cooperative
  blocks (tiles 72 x 128, the first version's waves 72 x 256), the most
  that fit at once: 3 per SM. The third cube lands quietly as r10's warp
  card folds back into SM 0, not under the line that introduces tiles.
- **The gate** flashes amber once per round, at most twice per second.
  When rounds race, it holds up at steady amber (r05b's pattern) and
  lowers when they end.
- **No launch counter** in act 3: the count depends on switches and is
  not part of the story.

### r10_tiles (k = 14), new

Rung: So (fixes r09's "every red pixel still rides the queue") the queue
carries tiles, and one warp closes a tile in registers (אוגרים).

```text
So the queue carries tiles, not pixels: squares of thirty-two by thirty-two. One warp, thirty-two threads that run in step, takes a tile. Each thread holds one row, one bit per pixel. The row sits in registers: the thread's own memory, closer than shared memory. Fill along every row at once, step diagonally, fill along the columns, and repeat until nothing changes.
```

Changed from the evening draft (critic 2, F4): "Each thread holds one
row, one bit per pixel, in registers: its own memory, closer than shared
memory." packed five ideas into one sentence. It is split: the row, then
where the row sits. +4 words and one sentence end, about +2 s.
"as a thirty-two bit word" stays cut (N3).

Picture:
- "So the queue carries tiles": the labelled frame turns back to red.
  One `machine_to`: the twelve red tokens become tile tokens and the
  launch pill fades out.
- "thirty-two by thirty-two": the floor zooms into the frame's largest
  blob (76,090 px). A 7 x 7 grid of tiles lands on it (TL `cu_ctx_red`,
  224 x 224 px). The 15 tiles the BFS opened in rounds 0-3 are already
  blue (`cu_ctx_fill`), the blob's seed tile among them. The 7 tiles of
  round 4 are red; the centre is outlined. No round numbers: on this blob
  the rounds move with timing.
- "One warp, thirty-two threads that run in step, takes a tile": SM 0's
  close-up opens as in r02; one partition's "32" lights, and the warp
  card unfolds out of it with the tile's real mask (`cu_mask`: 543 of
  1,024 red).
- "Each thread holds one row": the rows light top to bottom; one row's
  32 cells get a bracket.
- "The row sits in registers": the card's frame pulses (the new-part
  convention).
- "Fill along every row": the recorded sweeps (`cu_sweeps`) play, one
  every ~0.4 s. The 17 entry bits sit on the left edge (bit 0, rows
  13-31 but 19 and 28), where the round-3 tile beside it handed them
  over. Then row 86, diagonal 109, column 181, row 387, diagonal 481,
  column 541, row 543. A small two-headed glyph under the card shows
  each sweep's directions (along the rows both ways, both diagonals,
  along the columns both ways: ch07's `_row_fill`, `_diag_step` and
  `_col_fill` all run both ways); no words.
- "until nothing changes": trip 4 changes nothing. On "changes", with
  caption 3: the 32 threads vote (the column of dots dims), the card
  folds back into SM 0, and the 7 round-4 tiles on the floor turn blue
  together: the frontier (חזית) closes in parallel. As the card folds
  back, each SM's third cube lands.

Captions (3):
1. "1 tile, 1 warp" / "32 x 32 pixels", at "warp".
2. "1 bit per pixel" / "a row per thread", at "pixel".
3. Carried: "4 trips" / "the last changes nothing", at "changes".

Numbers:
- 32 x 32: `tile_px` (TL `square.width / tiles[0]`), and the crop's 384 /
  12 (WV). One thread per row: `WARP_LANES` = 32, a CUDA constant.
- 3 per SM: `big_tiles_blocks` = `small_tiles_blocks` =
  `frame_waves_blocks` = 72 = 3 x 24 (V5).
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

Length: 62 words, about 23.8 s of speech, 25.2 s with its tail.

### r11_rounds (k = 15), new

Rung: one gate per round: 4,001 rings become 127 rounds, 101 ms become
3.12. But that's one blob, from one seed.

```text
Only the tile's border bits go out, into the tiles around it, and those join the next round's queue. One gate per round, not two per ring. On the giant square, four thousand rings become a hundred twenty-seven rounds: from a hundred milliseconds to three. But that's one blob, from one seed.
```

Changed from the evening draft (critic 2, F7): "its border bits ... into
its neighbours" becomes "the tile's border bits ... into the tiles
around it". After the beat cut, "its" had to reach back two sentences,
and "neighbours" meant pixels until now. +3 words.

Picture:
- "Only the tile's border bits go out": the closed tile's reached border
  cells light: 31 on the top row (lane 0), 26 on the left column (bit
  0), 18 on the right (bit 31), 2 on the bottom row (lane 31).
- "into the tiles around it": bits travel only into the 4 tiles they
  really open: above (32 px), right (18), above-left (1), above-right
  (1). The tiles to the left and lower left are already blue (round 3)
  and the one below closed with the centre (round 4): no arrows go
  there. No arrow goes into the lower-right tile either: it touches no
  reached pixel, and another tile opens it in round 6.
- "join the next round's queue": the 12 tile tokens fade (consumed) and
  4 drop in.
- "One gate per round": the gate flashes once and the 4 tiles turn blue
  (round 5). Then the context holds still through "not two per ring"
  (critic 2, F8: the replay of rounds 6-8 is gone, three flashes and one
  floor change fewer).
- "On the giant square": the floor swells into the 64 Mpx square, at
  act 1's footprint. The inset reads "stress test"; the tag stays COUNT.
  The square is drawn from the real rounds: 250 x 250 tiles, each shaded
  by the round it first opened (TL `sq_first_open`), quantized (לכמת)
  into about 12 bands, light to dark.
- The rounds play in about 4 s: 2 slow rounds, one flash each; then the
  gate holds up at steady amber while the rest race and a small counter
  beside the floor runs to 127; the gate lowers as the counter lands.
  The queue refills to 12 tile tokens.
- "from a hundred milliseconds to three": the row 2 pill turns from
  "GPU 11×" to "GPU 340×". One change; no caption lands with it.
- "But that's one blob": the floor pulls back to the 81 Mpx frame: the
  largest blob blue, the other 2,521 red; the seed dot pulses. The inset
  returns to "drone frame".

Captions (3):
1. "1 gate" / "per round", at "round," (word 23).
2. "127 rounds" / "was 4,001 rings", at "twenty-seven".
3. Carried: "1 blob" / "from 1 seed", at "seed".

The 3.12 ms is on the r14 card; here the pill carries the jump.

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
- 100 ms: `big_8ways_ms` (V5 `ch03_conn8.ms`, 101: r06's kernel). 3.12
  ms: `big_tiles_ms`. Paired per round: 32.4x (`big_8_over_tiles_paired`).
- 340×: `big_tiles_x` = 1,065 / 3.12 = 341 (311-345), against the
  8-neighbour CPU without a depth map, reset on the clock (`cpu_fill8`,
  `big_cpu_fill_ms`).
- One seed: `big_seed` (V5 `sq_8000_center.seed`).

State at the cut (k = 16): floor `frame_one_blob` (the 900 px frame,
red, the largest blob blue); machine as k = 15; rows (small CPU 4.3×,
huge GPU 340×, many GPU 3.8×); tag COUNT; inset as k = 14; needle 53.9
ms; caption "1 blob".

Length: 52 words, about 19.4 s, 21.0 s with its tail.

### r12_pieces (k = 16), new

Rung: So every tile finds its pieces, seeds send waves piece to piece,
and at the end each blob keeps its smallest label.

```text
So first, every tile finds the blob pieces inside it, in registers, all at once. Some pieces are seeds: each seed sends a wave from piece to piece, one round at a time. Where two waves meet, the meeting is noted, and at the end each blob keeps its smallest label.
```

Unchanged. The picture is WV's crop: 384 x 384 px at x 4352, y 5632,
run as its own image. It shares the frame's tile grid and seed lattice
(סריג): its 36 seeds are the frame's, tile for tile. But it is not a
window of the frame's run:
- 6 of its 8 blobs are cut by its edge, so they are parts of larger
  blobs.
- 7 of its 98 tiles end on another round than in the frame (its last
  claim comes in round 5; the frame's tiles there end by round 4).
- Its labels are its own.

So the picture shows it as a small image of its own, inside its own
border, and nothing of it carries back into the frame.

Picture:
- "So first, every tile": the floor zooms from the frame into the crop's
  window. When it lands, a thin border frames the crop and the rest of
  the frame fades out: 12 x 12 tiles, 8 blobs, red.
- "finds the blob pieces inside it": the SM rooms pulse once. Finding
  the pieces is its own launch, one warp per tile, so the cubes (the
  waves' blocks) stay still. Each piece gets an outline and a dot at its
  first pixel (its anchor): 135 pieces, at most 4 in a tile
  (`crop_piece`, `crop_comp`).
- "Some pieces are seeds": 36 sparks (as r09's); 11 of them are seeds
  only because their tile is on the lattice (`crop_tile_lattice`, faint
  diagonal hatching). Each seed keeps a mark after its spark, its dot,
  larger, in its wave's hue with a white rim: of the 25 seeds not on
  the lattice, 19 are 13 px or less, too small to show a hue.
- "8 blobs": 2 of them are tiny (2 and 25 px, on the crop's edge), so a
  dashed ring stands round each until the rounds start.
- "each seed sends a wave": the queue's tokens become piece tokens.
  Six rounds, one gate flash each, 0.55 s apart. A claimed piece takes
  its wave's hue in its round (`crop_comp_round`, `crop_comp_wave`): 50,
  38, 7, 3, 1, then none. Each round's meetings appear as faint white
  ticks on the tile borders (`crop_meet_tester`, `crop_meet_other`,
  `crop_meet_round`, 116 in all).
- "Where two waves meet": the ticks brighten together, once.
- "and at the end": the hatching, ticks, seed marks and dots fade, all
  but 8 dots, which grow: each blob's label piece, its smallest anchor
  over all its pieces (ch08's flatten passes; `crop_comp_label`). In 5
  of the 8 blobs that piece is no seed. One dot per blob, so the 8 can
  be counted. At "smallest" the 8 dots grow once more and every piece of
  a blob turns at once to the hue its label piece already shows; the
  dots fade after "label".
- Hues: the waves need distinct hues where they meet (a colouring of the
  meeting graph from `BLOB_HUES`); the 8 label pieces' waves too, at
  least where blobs share or neighbour tiles.

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
  `full_paint_ccl_tiles`, `d8_ccl_tiles`, `d8_paint_ccl_tiles`, and V5's
  timed `frame_dense_tiles`, both engines, and `drone_dense_tiles`: all
  0).
- The crop as its own image: `floors()` `crop_blobs_cut_by_edge` 6,
  `crop_tiles_other_last_round` 7, `crop_seeds_are_frame_seeds`.

State at the cut (k = 17): floor `crop_labels` (the crop inside its
border, one hue per blob, faint tile grid; no dots, hatching or ticks);
machine as k = 16 with piece tokens; rows and needle as k = 16; caption
"min".

Length: 51 words, about 18.3 s, 19.5 s with its tail.

### r13_frame (k = 17), new

Rung: the frame: over 11,000 seeds, 6 rounds, every blob counted, 2.21
ms once the frame is on the card: inside the budget.

```text
On the big frame: over eleven thousand seeds, six rounds, every blob counted. Two point two milliseconds, once the frame is on the card: inside the budget.
```

Unchanged; it holds (2.21 ms, ch08's first version).

Picture:
- "On the big frame": the crop's hues blend back to red and its border
  fades; then the floor pulls back to the whole red frame.
- "over eleven thousand seeds": the seeds fade in as small dots over at
  least 1 s, tile by tile (WV `full_tile_seeds`, 282 x 282). No white
  rings at this scale (critic 2, F12): r09's ring stays for r09's toy
  and r12's 36 seeds.
- "six rounds": six gate flashes, 0.5 s apart; the claims spread out
  from the seeds (WV `full_round_900`, the claim round at 900 x 900
  samples). At 900 px only rounds 0-4 show: rounds 5 and 6 claim 6 and
  0 pieces, so the last two flashes change nothing visible. The 94
  interior samples are drawn labelled from the start.
- "every blob counted": the frame turns into the labelled picture (v4's
  `frame_labels`: one pastel hue per blob, the same blobs; not the
  palette ch08 paints).
- "Two point two milliseconds": with caption 2, the needle slides from
  53.9 ms past the 33 ms line to 2.21 ms and turns teal.
- "inside the budget": about 0.6 s later, the tag turns "COUNT ✓" and
  row 3's pill turns from "GPU 3.8×" to "GPU 100×".

Captions (2; a third would land under MIN_DWELL after the first):
1. "11,609 seeds" / "6 rounds, 2,522 blobs", at "rounds".
2. Carried: "2.21 ms" / "kernel time, no copies", at "milliseconds".

Numbers:
- 11,609 seeds: `full_seeds` (WV) = `d8_seeds` (D8); W8 agrees.
- 6 rounds: `frame_rounds` (V5) = `full_rounds` = `d8_rounds`; frontier
  11,586 / 14,744 / 7,834 / 2,306 / 289 / 6 pieces (`full_frontier` =
  `d8_frontier`).
- The engine: V5 times ch08's first version (`frame_first_version`:
  `OLD_PATH`, dense off, the hybrid lookup, no fitted grids), which is
  D8's `old_path` (`d8_old_is_v5_engine`); D8's counters for it equal the
  default's (`d8_old_same_as_default`), and WV recorded the default
  (`wv_engine_is_v5`). The default, timed beside it for reference: 2.25
  ms (paired 1.017).
- 2,522 blobs, labels equal the CPU's: `n_blobs`, `frame_labels_ok` (V5,
  both engines), `full_blobs`, `d8_n_blobs`.
- 2.21 ms: `frame_ms` (V5 `ch08_old`, kernel only), 0 dense tiles. With
  the copies 54.8 ms (`frame_e2e_ms`), over 33: hence "once the frame is
  on the card".
- The needle's drop: 53.9 / 2.21 = 24.3x (V5's ratio of medians).
- 100×: `count_x` = 225 / 2.21 = 101 (97-107), against the labels-only
  CPU with its reset on the clock, as the waves set up on theirs.

State at the cut (k = 18): floor `frame_labels`; needle 2.21 ms, teal;
rows (small CPU 4.3×, huge GPU 340×, many GPU 100×); tag COUNT ✓; machine
as k = 17; caption "2.21 ms / kernel time, no copies".

Length: 27 words, about 10.3 s, 12.5 s with its tail (HOLD 1.8 s).

### r14_reveal (k = 18), new text

Rung: the opening question, answered with the scoreboard. Row 1 flips in
kernel time, and the voice says what the copy does (decision 3).

```text
One huge blob, or many at once: the GPU. And one small blob? In kernel time, the GPU now; with the copy to the card, still the CPU. Every drone, every frame.
```

"with the copy to the card" (critic 2, F10) echoes r13's "once the frame
is on the card", the only bridge to the copy; +3 words.

Picture (v4's r11_reveal, re-cued):
- "One huge blob": the full card opens; rows 2 and 3 framed in teal;
  the machine's blocks land column by column (as v4). Each row's tag is
  drawn under its name.
- "the GPU": caption 1.
- "And one small blob?": row 1's frame, and the first step of its flip
  (critic 2, F2): the CPU cell alone changes, from 0.121 to 0.192 ms,
  and its tag from "CPU: 4 neighbours" to "CPU: 8 neighbours"; the GPU
  cell and the winner go to "?" (the scoreboard's "promised, unknown").
  No ratio of 0.192 against r02's 0.520 is ever on screen.
- "In kernel time, the GPU now": the second step, about 3 s later: the
  GPU cell lands at 0.044 ms and the winner at "GPU 4.4×"; caption 2.
- "with the copy to the card, still the CPU": the footnote's copy line
  lights; P-core 0 lights and goes dark again. At "CPU" a grey note
  pill, "with the copy" / "CPU 3.5×", hangs under the strip's small-blob
  pill and stays to the end, where the inset was: the footnote leaves
  with the card, and without the note the closing strip read as a GPU
  win on every row (final check, F1).
- "Every drone": the card closes onto the live drones; caption 3. The
  title card fades in for the last seconds: "flood fill, from pixels to
  tiles" over "flood-fill-cuda".

The card (`score_rows(beat)`, every number through `card_ms` / `card_x`;
header "one session, GPU kernel time"):

| row | sub | CPU | GPU | winner | tag |
|---|---|---|---|---|---|
| one small blob | 16 K px test blob | 0.192 ms | 0.044 ms | GPU 4.4× | CPU: 8 neighbours |
| one huge blob | 64 Mpx test blob | 1,065 ms | 3.12 ms | GPU 340× | CPU: 8 neighbours |
| many blobs | 81 Mpx frame, 2,522 blobs | 225 ms | 2.21 ms | GPU 100× | CPU: same labels |

Footnote (`card_ms_note`, rounded): "With the copies: small blob 0.665
ms (CPU 0.192), huge blob 44.3 ms, frame 54.8 ms."

Captions (3):
1. "the GPU" / "huge or many blobs", at "GPU".
2. "small blob too" / "in kernel time", at "now".
3. "every drone frame" / "GPU + copies: 1.26 ms", at "Every" (it names
   whose time it is: a time, not a win).

Numbers:
- Row 1: CPU 0.192 ms (`small_cpu8_ms`, the 8-neighbour BFS without a
  depth map, reset on the clock, as ch07 keeps none and clears on its
  clock), GPU 0.044 ms (`small_tiles_ms`, V5 `sq_256.ch07.ms`, 4
  rounds): 4.4× (`small_tiles_x`; 4.27-4.54 over the CPU's rounds).
- With the copy: 0.665 ms (`small_tiles_e2e_ms`): the CPU wins by 3.5x
  (`small_e2e_cpu_x`), and by 5.5x against the 4-neighbour CPU. Rows 2
  and 3 stay the GPU's with the copies too: 44.3 and 54.8 ms against
  1,065 and 225 (24x and 4.1x).
- Caption 3: the drones with the copies, the worst frame's median 1.26
  ms (`drone_e2e_max_ms`; median 0.977 ms), far under 33, so "every
  drone, every frame" holds end to end without naming a winner: the CPU's
  0.679 ms beats the GPU's 0.977 there. The GPU's kernel time on the
  drones (0.090 ms median, first version) is shown nowhere.

Length: 32 words, about 13.4 s, 17.3 s with its tail (HOLD 3.5 s).

---

## The scoreboard, before and after

All rows: one session (V5), GPU kernel time, CPU `@njit` on one core,
each row against its like-for-like CPU twin, named in its tag. Every
card row is checked to read true: its two shown times give its shown
factor (`card_row_ok`). The strip during the acts shows the pills only.

| row | r02 | r06 | r09 | r11_rounds | r13_frame | r14_reveal |
|---|---|---|---|---|---|---|
| one small blob (16 K px) | CPU 4.3× (0.121 vs 0.520 ms), 4 neighbours | | | | | **GPU 4.4×** (0.192 vs 0.044 ms), 8 neighbours |
| one huge blob (64 Mpx) | ? | GPU 11× (1,126 vs 101 ms), 8 neighbours | | **GPU 340×** (1,065 vs 3.12 ms) | | |
| many blobs (81 Mpx frame, 2,522 blobs) | ? | ? | GPU 3.8× (202 vs 53.9 ms), same labels | | **GPU 100×** (225 vs 2.21 ms) | |

v4 for comparison: CPU 2.4× (July, host clock), GPU 25.8× (July), GPU 35×
(1 Mpx noise, two sessions, the run-based kernel).

Within a row the CPU's time moves when its twin does, and only the
r14 card shows times: row 1 from 0.121 (4 neighbours, r02) to 0.192 (8
neighbours, no depth, r14), in its own step; row 2 from 1,126 (depth,
r06) to 1,065 (no depth, r11); row 3 from 202 (reset before the clock,
r09) to 225 (reset on it, r13). The meter's CPU tick and i1's stopwatch
are the 225 ms of the r14 card.

How far each shown factor moves with the CPU's 5 round medians:

| shown | CPU twin, round medians | factor range |
|---|---|---|
| row 1 at r02: CPU 4.3× | 4 neighbours, 0.117-0.122 ms | 4.26-4.44 |
| row 1 at r14: GPU 4.4× | 8 neighbours, no depth, 0.188-0.200 ms | 4.27-4.54 |
| r03b 1.1×, r05 8.5× | 4 neighbours, 1,120-1,284 ms | 1.02-1.17, 7.97-9.13 |
| row 2 at r06: 11× | 8 neighbours, depth, 1,003-1,182 ms | 9.92-11.7 |
| row 2 at r11: 340× | 8 neighbours, no depth, 971-1,077 ms | 311-345 |
| r07 CPU 7.2×, r08 CPU 4.2× | labels, reset before, 189-226 ms | 6.5-7.8, 3.8-4.5 |
| row 3 at r09: 3.8× | labels, reset before, 189-226 ms | 3.51-4.19 |
| row 3 at r13: 100× | labels, reset on, 214-238 ms | 96.8-107 |

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
| 14 | the labelled frame, 53.9 ms, rows (CPU 4.3×, GPU 11×, **GPU 3.8×**), tag COUNT, cap "53.9 ms" | v4's k = 14 + row 3 |
| 15 | `tile_ctx` (rounds 0-4 blue), three cubes per SM, 12 tile tokens, no launch pill, cap "4 trips" | new |
| 16 | `frame_one_blob`, row 2 GPU 340×, tag COUNT, cap "1 blob" | new |
| 17 | `crop_labels` (bordered, hues only), piece tokens, cap "min" | new |
| 18 | `frame_labels`, needle 2.21 ms teal, row 3 GPU 100×, COUNT ✓, cap "2.21 ms" / "kernel time, no copies" | replaces v4's k = 15 |

---

## New fields, rasters and helpers

- `state.picture()` returns the 900 px frame transposed; so do
  `picture_blobs()`, `filled_picture()`, `teal_picture()` and r07's
  zoom target, which build on it.
- `mc.Spec`: `token_kind` ("pixel", "tile", "piece"); `teams` up to 3.
  `mc._boxes` and `place_blocks` draw 3 cubes (about 0.10 wide each).
- A gridless floor raster: an RGB array upscaled nearest to the floor's
  pixels, alpha 255, no keying, with an optional faint tile grid every N
  cells. `fl.toy_raster` draws a line on every cell edge (every 3.5-3.9
  px at 250-282 cells), and `fl.picture_raster` keys light colours as
  white (the square's light end would wash out).
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
- `score_rows(beat)`, like for like:
  - small row: until r14, CPU `small_cpu_ms` (4 neighbours) against
    `small_gpu_ms` (r02's one block), "CPU 4.3×"; at r14, in two steps,
    CPU `small_cpu8_ms` (8 neighbours, no depth) with the GPU and winner
    "?", then `small_tiles_ms` and "GPU 4.4×".
  - huge row: CPU `big_cpu8_ms` (= `big_cpu_ms`) against `big_8ways_ms`
    until r11, then `big_cpu_fill_ms` against `big_tiles_ms`.
  - many row: CPU `count_cpu_pre_ms` against `count_ch05_ms` from r09,
    then `count_cpu_ms` against `count_gpu_ms` from r13.
  - Tags "CPU: 4 neighbours", "CPU: 8 neighbours", "CPU: same labels",
    drawn under the row name by `ov.score_card` (today neither
    `score_strip` nor `score_card` draws `Row.tag`); the card header
    "one session, GPU kernel time". Sub "81 Mpx frame, 2,522 blobs" for
    row 3.
- `overlays.fmt_ms` becomes `numbers.card_ms` (three decimals under 1
  ms: 0.192, 0.044), the footnote's format `numbers.card_ms_note`
  (rounded 10-100 ms), and `fmt_x` two significant figures
  (`numbers.card_x`); the checks use the same functions. Every scene's
  own factor format goes through them too: r03b's `:.2f` would print
  "1.09×", r05's `int()` "8×", and state.py's r06 caption and
  `score_rows()` `:.1f` / `:.0f`. The meter's packed-mask mark goes (it
  reads the removed `ch06_mask_ms`).

About 400-600 new lines in all, every piece on an existing pattern.

Scene edits the decisions and the reviews add (build stage):
- intro: the stopwatch stops at "second" (was "seconds"); the drone
  card shows the CPU's bar against the budget and the GPU as "?",
  caption sub "240 frames, one session".
- r02: the race bar reads "1 block", its value from `small_gpu_ms` (now
  r02's own kernel); `dz.sm_detail`'s partitions read "32", not "32
  lanes" (also in r10).
- r03b: caption "1.1×", cued at "faster".
- r05: caption "8.5×".
- r06: caption "11×", sub "CPU: 8 neighbours too".
- r08: the two blobs fill 8 ways; the floor play ends and the needle
  moves at "faster"; the needle settles right of the CPU tick.
- r09: row 3's pill at "Faster than the CPU"; the caption at "budget".
- r10: the "registers" pulse on the card's frame; the third cube as the
  card folds back; caption 3's sub "the last changes nothing".
- r11: no replay of rounds 6-8.
- r13: the frame's seeds fade in as dots.
- r14: row 1's two-step flip; the tags on the card; caption 3 "with
  copies: 1.26 ms"; the footnote through `card_ms_note`.

Other files that list beats:
- `base.py` HOLD: drop r10_runs and r11_reveal; add r10_tiles 1.0,
  r11_rounds 1.2, r13_frame 1.8, r14_reveal 3.5.
- `build/assemble.py` CHAIN: drop r10_runs and r11_reveal; add the five.
- `build/seam_check.py` FADES_OUT: "chain/r11_reveal" becomes
  "chain/r14_reveal".
- `HANDOFF.md` section 11.
- `r10_runs.py` and `r11_reveal.py` leave the chain. `r14_reveal.py`
  starts as a copy of `r11_reveal.py`. Deleting the two old files is the
  user's call.

---

## `numbers.py`: what changed

- Sessions: V5 is now `v5_20261017T192640Z`. The evening session
  (`182449Z`, the oracles as called), the morning one and July's
  overview session (OV) are no longer read: act 2's estimates, the 24
  SMs and the 256 threads all come from V5. T7, D8, W8, TL, WV and act
  1's GPU-only July sessions (CH01, DB2, CH03, OWN) stay.
- V5's schema: CPU cells `cpu_bfs4`, `cpu_bfs8`, `cpu_fill8`,
  `cpu_label`, `cpu_label_pre`, and the reference cells `oracle_bfs4`,
  `oracle_bfs8`, `oracle_label`, `cpu_fill_canonical`; every CPU twin
  carries `outputs`, `connectivity`, `reset` and `same_as_oracle`; every
  GPU cell carries `outputs`, `reset`, `twin` (with `like_for_like`) and
  `block`; blocks carry `attempts` and `full_clocks`; the frame has
  `ch08_old` (the story's) and `ch08` (reference); the drones carry
  their engine, outputs, twin and `default_engine_dense_census`.
- Keys with a new meaning (the names scenes read stay):
  - `njit_ms`: the CPU on the frame, labels only, its reset on the clock
    (225 ms): i1's stopwatch, the meter's CPU tick, row 3 from r13.
  - `count_cpu_pre_ms` (new): the same, reset before the clock (202
    ms): act 2's twin and row 3 at r09.
  - `small_cpu8_ms`: the 8-neighbour BFS without depth (0.192 ms), r14's
    twin; `small_cpu8_depth_ms` the BFS with depth (reference).
  - `big_cpu8_ms`: the 8-neighbour BFS with depth (r06's twin);
    `big_cpu_fill_ms` (new): without depth, reset on the clock (r11's
    twin, the r14 card's row 2).
  - `frame_ms`, `frame_rounds`, `frame_e2e_ms`: ch08's first version;
    `frame_default_ms` (new) the default engine (reference).
  - `drone_gpu_ms`: the first version (not shown); `drone_e2e_max_ms`:
    r14's caption 3.
- New checks: every twin within its GPU side's outputs, with the same
  neighbours and reset, equal to its oracle (`v5_twins_ok`,
  `v5_all_like_for_like`); every shown factor within its range over the
  CPU's rounds; the clock attempts (7 first, ch05 second); the power-cap
  blocks; the first version's engine against D8's `old_path`; the
  oracles' allocation cost (reference); `card_ms_note` for the footnote.
- A wrong twin fails: giving ch07 on the square the 8-neighbour twin
  with depth (the evening session's pairing) turns `v5_twins_ok` false.
- 114 checks (the evening draft had 108, v4 59).
- `state.py` still builds `boundary(0)` to `boundary(14)` with the new
  numbers; `boundary(15)` (v4's r10) fails on the removed `ch06_rgb_ms`,
  as it should until the build stage rewrites it.

---

## Length

Model, calibrated on the real v4 paced takes:
- Speech: words / 3.11 + 0.77 s per inner sentence end + 0.41 s per
  colon or semicolon. It reads v4's 16 takes at +1.8% in all (beat by
  beat within about 12%).
- An unchanged beat uses its real v4 take; a beat with a few words
  changed uses its take plus the differences.
- The slot adds the tail: max(BREATH 0.75 s, HOLD) + 0.4 s.

The task's simpler formula (2.6 words per second + 1.125 s per sentence
end + 0.56 s per colon) puts v4 at 270 s against its real 218 s, so it
reads about 23% long; its column is given for reference.

| beat | words | status | speech, s | slot, s | task formula, s |
|---|---|---|---|---|---|
| i0_problem | 22 | same (take) | 8.8 | 9.9 | 11.8 |
| i1_compare | 35 | changed (+2 words vs v4) | 17.1 | 18.2 | 20.2 |
| r00_zoom | 9 | same | 3.8 | 5.7 | 5.7 |
| r01_cpu | 24 | same | 8.6 | 9.7 | 12.0 |
| r02_one_block | 40 | +5 words vs v4 | 14.4 | 16.0 | 19.9 |
| r03_giant | 34 | same | 12.6 | 13.7 | 16.4 |
| r03b_one_sm | 16 | changed (+2 words vs v4) | 5.6 | 6.8 | 8.4 |
| r04_two_blocks | 62 | same | 23.8 | 24.9 | 32.3 |
| r05_all_blocks | 63 | changed (+3 words vs v4) | 23.1 | 24.3 | 31.5 |
| r05b_snake | 23 | same | 8.2 | 9.4 | 12.2 |
| r06_diagonals | 16 | same | 6.0 | 7.6 | 9.0 |
| r07_real_frame | 25 | same | 10.2 | 11.3 | 14.1 |
| r08_tags | 54 | changed (+3 words vs v4, +2 vs the morning draft) | 19.4 | 20.6 | 27.5 |
| r09_seeds | 30 | changed (+7 words vs v4) | 12.0 | 13.1 | 16.0 |
| r10_tiles | 62 | new | 23.8 | 25.2 | 30.6 |
| r11_rounds | 52 | new | 19.4 | 21.0 | 25.1 |
| r12_pieces | 51 | new | 18.3 | 19.5 | 23.6 |
| r13_frame | 27 | new | 10.3 | 12.5 | 13.8 |
| r14_reveal | 32 | new | 13.4 | 17.3 | 17.9 |
| **whole cut** | 677 (v4 496) | | | **286.9 s (4:47)** | 348 s |

Against the evening draft (276.5 s): r02 +1.0 s, r03b +1.2 s, r05 +1.0
s, r09 +3.1 s, r10 +2.0 s, r11 +0.9 s, r14 +0.9 s. Against v4 (218 s):
+69 s, under the five-minute cap with 13 s to spare. The task formula,
scaled by its v4 error (218 / 270), gives about 281 s.

Rendered (2026-10-18): 278.4 s (4:38), 258.8 s of it speech plus gaps.
The model read about 3% long.

Caption spacing: every new caption is cued at least 1.75 s (MIN_DWELL)
after the one before, by word count at 3.11 words per second; r13 keeps
two captions for this reason, and the scenes enforce it in code
(`max(cue, previous + MIN_DWELL)`). r14's two flip steps sit about 3 s
apart. Re-check all of them from the new `words.json`.

---

## What v5 does not claim

- **Kernel time only.** Every GPU time is kernel only, as in v4. With
  the copies: the frame takes 54.8 ms (over 33; the GPU still 4.1x), the
  square 44.3 ms (the GPU still 24x), the small blob 0.665 ms (the CPU's
  0.192 wins by 3.5x). The drones take 0.977 ms per frame with the
  copies (worst 1.26), so there the CPU's 0.679 ms wins too; both stay
  far under 33, and nothing in the script says the GPU wins the drones.
  r13 and r14 say the frame's and the small blob's cost aloud.
- **The CPU's arrays are kept between calls.** Like the GPU's, they are
  allocated once per image, and each twin resets them where its GPU side
  resets its own. Called as the oracles are, with fresh arrays (as the
  evening session timed them), the CPU took 1.4-1.5x longer on the big
  images (V5's reference cells).
- **v4's CPU baselines are gone from the screen.** On the frame, v4's
  baseline labelled and then filled each blob from its seed, with depth:
  1,165 ms, 5.2x the labels-only CPU. V5 times it as reference only.
- **The CPU's label map against ch08's labels.** The CPU writes a
  per-pixel int32 label map on its clock; ch08's timed run labels its
  pieces and paints the image, and its per-pixel label map is an
  off-clock twin (`emit_label_map`; ch08 README). Both say which blob
  each pixel is in, in different forms (critic 1, F6).
- **Act 2's loops are handed their seeds.** Their seeds (each blob's
  first pixel) come from the CPU's labels, off the clock, and a loop
  keeps no label map: a blob's identity is its launch (r07) or its tag
  (r08). Their twin finds its own seeds and writes the map. The CPU does
  more there and still wins, so r07 and r08 hold.
- **Row 1 changes its CPU** between r02 (4 neighbours, against r02's
  one block) and r14 (8 neighbours, no depth, against the tiles). The
  tag says so, and the flip shows the CPU's change as its own step.
- **Act 2's seeds kernel on the small blob is not shown.** Like for
  like, it already edges the 8-neighbour CPU there: 0.166 ms against
  0.186 (with depth, as ch05 keeps one), 1.12x (1.10-1.17). Row 1 keeps
  r02's comparison until the finale, so "the GPU now" is the first win
  the viewer sees, not the first that exists. See N2.
- **"From a hundred milliseconds to three"** compares r06's own 8-way
  kernel: the thread layout with its counters on (`bare=False`, 104
  registers), against the bare tiles. The square's best tuned pixel BFS
  (warp layout, bare, 72 x 256) takes 75.2 ms, 23.9x the tiles, paired
  (`big_tuned_ms`, `big_tuned_over_tiles_paired`).
- **Where the 3.12 ms go.** The rounds take 1.11 ms; packing the image to
  bits (1.16 ms) and painting it back (0.82 ms) take the rest
  (`big_tiles_*_ms`). Not spoken (Q8).
- **Act 2's estimates** are v4's method: the median of 6 sampled calls
  times the call count. The full loops, measured once, take 1.13x and
  1.35x the estimates, so the needles sit a little left of the truth.
  Every form is 4.2-9.5x slower than the CPU, so r07 and r08 hold.
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
- **The power cap.** ch05 on the frame draws more than the laptop's 40 W
  cap: its first attempt here sat in P3 and the second, with light load
  between runs, read 53.9 ms. The huge blob's block and the waves' block
  also logged the cap at full clocks. Spoken "about fifty".
- **The CPU's spread.** Its round medians spread by up to 19% here (8%
  in the evening session), with the host quiet and 60 s of idle before
  the pass. Every shown factor is checked over that range, and r03b's
  margin is the thinnest (1.02-1.17; see N4).
- **Seeds are not one per blob**: 11,609 seeds for 2,522 blobs (3,673
  only from the lattice).
- **ch06's code is still inside the tiles, unseen.** Nothing of it is
  spoken or shown, and no ch06 cell is timed, but the user should know:
  - ch07's pack phase is ch06's `pack_kernel`, and the merge after the
    waves reuses ch06's `_union` code.
  - ch08's default sends dense tiles down the dense-tile path. V5 times
    the first version, which has none; on the frame and the crop the
    default sends 0 tiles there anyway. An untimed census of the default
    on the drones found 596 labelling and 2,022 paint tiles of 80,402 on
    that path.
  - The drone footage (`assets/drones_labels`) was coloured by ch06's
    run kernel (`make_drone_frames.py`). V5 found ch08's labels equal
    on all 240 frames, so the build stage regenerates the footage from
    ch08 and checks that the frames come out the same.

---

## Open questions

Answered on 2026-10-17 (see the decisions above): Q1 one session (now
the like-for-like one with kept arrays), Q2 like-for-like CPUs in the
rows, Q3 both sides spoken, Q4 the title, Q5 "once the frame is on the
card", Q6 r06's kernel, Q7 Cairo on the CPU and the GPU for
measurements, Q9 ch05 at full clocks, Q10 the drones without the dense
path (now the first version), Q11 one orientation.

Settled by the reviews: N1 (row 1 read 4.3× on both sides). It now reads
CPU 4.3× at r02 and GPU 4.4× at r14, and the two-step flip puts a "?"
between them.

Still open:
- **Q8. Where the time goes.** Add to r11: "Now the search is the small
  part: reading the image and painting it back take most of it." (about
  +4 s)? Default: off.
- **N2. Act 2's seeds on the small blob.** They beat the 8-neighbour CPU
  by 1.12x, unseen. Default: keep row 1 on r02's comparison and r14's
  "now" (true for the kernels the video shows); disclosed above. Or drop
  "now" ("In kernel time, the GPU; with the copy to the card, the
  CPU."), which loses the callback to r02.
- **N3. "as a thirty-two bit word" in r10.** Cut to keep r10's length
  with the definitions. Default: cut. Restoring it costs about 1.9 s and
  links back to r08's "the entry's word".
- **N4. r03b's thin margin.** One block is 1.09x the CPU (1.02-1.17 over
  its rounds). If a future re-run reads under 1, the line becomes "Now
  one block keeps pace with the CPU." Default: "a little faster", as
  this session measured.

---

## Build order (short)

1. The user's answers (done); `numbers.py` (done) and `script_chain.md`
   (done).
2. `state.picture()` transposed; `overlays.fmt_ms` / `fmt_x` from
   `card_ms` / `card_x` (and `card_ms_note`), and every scene's factor
   format through them.
3. `machine.py`: `teams` up to 3, `token_kind`, `warp_closeup`; the
   gate's single flash and its held state; the gridless raster and the
   raster primitives; `dz.sm_detail` without "lanes".
4. `state.py`: `BEATS`, `boundary(14..18)`, the new floors,
   `score_rows(beat)` with the CPU tags and row 1's two steps;
   `ov.score_card` draws the tags; `base.py` HOLD; `build/assemble.py`
   CHAIN; `build/seam_check.py` FADES_OUT.
5. Scenes: intro (stopwatch cue, the drone card with a GPU "?"), r02
   ("1 block"), r03b (cue "faster", "1.1×"), r05 ("8.5×"), r06 ("11×",
   "CPU: 8 neighbours too"), r08 (8-way fill, cues), r09 (row 3 at
   "Faster", caption at "budget"), r10_tiles, r11_rounds, r12_pieces,
   r13_frame, r14_reveal (from r11_reveal).
6. The drone footage from ch08 (`make_drone_frames.py` with TileWaves
   and `--tracker host`, under the GPU lock), compared with the old
   frames.
7. TTS: Kokoro for all 19 beats (the unchanged texts give the same
   takes), then `pace.py` at its defaults.
8. Checks: `build/layout_check.py` with the warp card open;
   `build/chain_snapcheck.py` on every snapping clip;
   `build/seam_check.py --cut chain` (17 seams); both layouts.
