# The Evolution of a Flood Fill

A living document. Each chapter follows the same loop: **the problems we
inherited → the approaches that attack them → what the measurements said →
the new problems those measurements exposed** — which become the next
chapter's inheritance. Every number below is measured on this repo's RTX
4060 Laptop GPU (24 SMs), every kernel is proven pixel-exact against the
same `@njit` CPU reference, and losses are reported as plainly as wins.

```
CPU BFS ──"one core is serial"──► 1 block ──"one SM is 4% of the GPU"──► 2 blocks ──"2 SMs are 8%"──► N blocks ──"one blob is one BFS"──► 2 blobs ──"who finds the seeds?"──► N blobs ──"why move pixels at all?"──► N runs
                                     │                                      │                            │                                    │                                   │                                    │
                          "the queue doesn't fit"                "how should two blocks share    "does it keep scaling?          "two seeds, one queue:            "no seeds given at all:              "150× fewer runs than red px:
                                     ▼                            one BFS? where do they run?"    what stops it — and is          who owns each pixel?"             candidate waves + atomicMin           connectivity over 539k items,
                                v2 spill tier                     split / global / dirsplit /     it bandwidth?"                  label in the entry;               union vs a CCL prepass;              the clock is one read + one write
                                                                  pinned                          global queue × 48 blocks;      levels = max, not sum             labels leave the entry               (53 ms → 1.35 ms, no BFS left)
                                                                                                  plateau at 512 threads/SM      (1.6–1.8×)                        (755k blobs in 25 ms)
```

Chapters 1–5 all ask *how should pixels move?* Chapter 6 asks whether
they should be the unit at all, and the answer — runs, not pixels — is
worth more than every scheduling result before it combined.

---

## Chapter 0 — the baselines and the broken prototype

**The problem:** flood-filling a blob is BFS — inherently level-by-level,
but *within* a level every pixel is independent. A CPU does them one at a
time; a GPU should do them together.

**The baselines:** pure-Python BFS (the classic algorithm, honest but
~100–1000× slow) and an `@njit`-compiled level-synchronous BFS
(`ch01_gpu_1blob_1block/cpu_oracle.py`) — the bar every kernel must beat
*and* the oracle every kernel must match pixel-for-pixel.

**The prototype's lesson:** the historical `ch00_cpu_baseline/single_block.py` had a
non-wrapping queue that silently dropped pixels, marked non-red pixels
visited (CAS before the color check), and hardcoded 64 threads. Its real
legacy was a checklist of what correctness requires: a queue that can't
overflow silently, a claim protocol that can't double-visit, and tests that
compare *depth maps*, not just pixel counts.

---

## Chapter 1 — one blob, one block (`ch01_gpu_1blob_1block/`)

### Inherited problems
1. Serial CPU fills; the GPU sits idle.
2. The prototype's correctness flaws.
3. A frontier queue needs a home: shared memory is ~20× closer than global
   memory, but a block gets only 48 KB of it.

### Approach v1 — the shared-memory ring
One block runs the whole level-synchronous BFS. The frontier lives in an
8,192-slot **ring buffer in shared memory** (virtual never-wrapped indices;
`front` frozen per level makes every in-window ticket a distinct slot).
Claim protocol: bounds → is-red → `atomic.cas(visited)` → enqueue — exactly
once, structurally. `depth[x,y] = level` recorded per pixel (later the key
to the wavefront animations). Two `syncthreads()` per level. If the ring
ever fills: a loud **tripwire** abort, never a silent partial fill.

### Results (v1)
| finding | number |
|---|---|
| vs pure Python | 7–45× faster |
| vs `@njit`, small scenes | **loses** (0.15–0.8×) — transfers + sync overhead dominate |
| vs `@njit`, 16M px | first GPU win (~1.4–1.6×) |
| serpentine (32,896 one-pixel levels) | **catastrophic** (0.004×): 255 of 256 threads idle |
| threads-per-block sweep | 18 → 86 Mpx/s (64 → 1024 threads): more resident warps hide memory latency |

### New problems v1 exposed
- **Capacity:** center-seeded scenes need ~4W ring slots → the tripwire
  fires for any center-seeded square past W ≈ 2048. Correct-but-refusing.
- **Frontier starvation:** narrow shapes can't feed even one block.
- **Ceilings:** one block caps at 1,024 threads (67% of one SM's 1,536
  residency), and one block = 1 of 24 SMs ≈ 4% of the GPU.

### Approach v2 — the spill tier (solves capacity)
Two-tier queue: the same shared ring as the fast path plus a global-memory
**spill tier** sized width×height. Safety became *structural* instead of
tripwired: every pixel is CAS-claimed at most once, so total spills can
never exceed the tier — nothing left to trip. Enqueues became
**warp-aggregated** (one shared atomic per warp instead of per pixel).

### Results (v2)
| finding | number |
|---|---|
| scenes v1 refused (2600²–6000² center-seeded) | all complete |
| GPU lead vs `@njit` **grows with size** | 1.21× @ 6.8M px → **2.03× @ 36M px** (596 vs 1,211 ms) |
| spill machinery when unused | ≈ free (warp-agg even beat v1 on sq_2000) |
| 36M px scene | 43% of the blob routed through the spill tier — still 2× the CPU |

### Problems carried into Chapter 2
1. Still 1 SM of 24; still ≤ 1,024 threads.
2. Per-level sync cost and serpentine starvation untouched.
3. New ceiling discovered: **host RAM** (5 GB laptop) caps scene size
   before the GPU does.

---

## Chapter 2 — one blob, two blocks (`ch02_gpu_1blob_2block/`)

### Inherited problems
Add one block. But blocks cannot `syncthreads` across each other, cannot
read each other's shared memory (even co-resident on one SM), and cannot
choose which SM they run on — the scheduler owns placement. So: **how do
two blocks share one BFS (partitioning), what does the inter-block barrier
cost, and does placement matter?**

### The approaches
| kernel | partitions by | frontier home | the bet it makes |
|---|---|---|---|
| `split` | **space** (block b owns half the image) | per-block shared ring + spill; cross-seam **inboxes** (≤ height entries, structurally) | keep v2's fast rings alive per block |
| `global` | **index** (grid-stride interleave) | one global queue | simplest possible; balanced by construction |
| `dirsplit` | **direction** (right/up → q0, down/left → q1) | one W×H buffer filled from both ends (the ends can never meet) | balance without spatial assumptions |
| `pinned` | — (placement experiment) | global queue + hand-rolled pair barrier | force 2 blocks onto ONE SM via occupancy (48×768 = exactly 2/SM) + `%smid` filtering |

All barriers: two `grid.sync()` per level (cooperative launch). Placement
is *observed* per run via a linked `%smid` reader, never assumed. **Bare
twins** of each kernel (instrumentation stripped) turn "does measuring slow
it down?" into a measured 0–3% (global/dirsplit) and 1.4–8.7% (split).

### Results
| finding | number |
|---|---|
| the 2nd block pays for itself | from ~262K px; **2.10× vs v2 @ 36M px** (280 vs 590 ms; 3.6× vs `@njit`) |
| ranking on big blobs | **global < dirsplit < split** (280 / 305 / 314 ms) — the plainest design wins |
| `grid.sync` priced by the serpentines | **~1.9 µs** ≈ 100× a `syncthreads`; every dual kernel loses to v2 there |
| placement (the same-SM question) | filling one SM's full 1,536 threads (impossible for any single block): **+12%**; spreading the pair over 2 SMs: **+68% more** — concurrency helps, parallelism wins |
| two rings really double capacity | the 2600² scene that made v2 spill 304,702 px runs **spill-free** on split |
| dirsplit's spatial agnosticism | 83% balance on the off-center blob where split sits at 0% (and runs 1.7× faster there) |
| split's inbox traffic | ≤ 215 px on multi-million-pixel scenes (bound: height) — cross-seam handoff is essentially free |

### New problems and lessons this chapter exposed
- **The barrier is the narrow-frontier killer.** ~4,000 cycles × 2 × tens
  of thousands of levels; no partitioning fixes a shape that can't feed
  the blocks between barriers.
- **Coordination costs registers.** The dual kernels need 104–159
  registers/thread of window state — a 1,024-thread block physically
  cannot be resident (tpb capped at 512), and split needed an explicit
  register cap (spilling 39 values to slow memory) just to reach 512. The
  pinned kernel exists only under a 40-register cap.
- **The shared-memory queue didn't pay.** The queue is a small slice of
  each pixel's global-memory traffic (image, visited, depth are global in
  every kernel), L2 absorbs the queue's hot window, and warps hide the
  latency — so split's rings bought little while its per-level
  publish/clamp choreography and spilled registers cost plenty. *Latency
  numbers mislead; bandwidth and instruction counts decide.*
- **Cycle accounting refined the story:** the two `grid.sync`s are ~95% of
  every kernel's level ritual and identical across kernels. Split's 34 ms
  deficit lives in per-pixel drag (spilled registers, 3-source reads);
  dirsplit's 25 ms is the **waiting tax** — private queues make each level
  last as long as the fuller queue.
- **Aggregate metrics can lie.** dirsplit's serpentine run reports 98.5%
  total balance while the per-level trace shows the blocks essentially
  *never* work simultaneously. Balance-over-time, not balance-in-total.
- **Ownership starvation is real:** a blob living in one half idles the
  split kernel's other block entirely (0% balance).

### Open problems → Chapter 3 candidates
1. **N blocks:** the payoff curve (1→2 blocks ≈ 2×) begs for 4, 8, 24
   blocks — the graveyarded `multi-blocks/` and `persistent/` designs
   revisited with this project's rigor (exact tests, bare twins, smid
   observation, per-block traces). Expected new problems: barrier cost
   grows with grid size; the global queue's single rear counter becomes a
   contention point (warp-aggregation may not be enough). *→ became
   Chapter 3.*
2. **The serpentine remains unbeaten** by everything GPU: candidate
   answer is tile-based BFS (iterate inside a shared-memory tile between
   global syncs) — trade barrier count for redundant tile work.
3. **Register dieting** for the dual kernels: fewer live window variables
   → higher tpb ceilings without spills.
4. **Measured (not derived) hardware truth:** an `ncu` profiling pass —
   real occupancy, DRAM/L2 hit rates, atomic contention.
5. **Multi-blob** (the original roadmap's Stage 5): per the design
   analysis, connected-component labeling beats running BFS per blob.

---

## Chapter 3 — one blob, N blocks (`ch03_gpu_1blob_nblock/`)

### Inherited problems
1. Two blocks are 2 of 24 SMs ≈ 8% of the GPU; the 1→2 payoff curve
   (≈ 2×) begs for N.
2. Chapter 2 crowned the plainest design — the global-memory queue — so
   there is exactly one kernel worth scaling.
3. Feared costs of N: the two-per-level `grid.sync` across a bigger grid,
   and the single global rear atomic under 48-way pressure.
4. The stage hypothesis (the user's inference from Chapter 2): the queue's
   hot window fits L2 outright (even a 10000² blob's frontier ≈ 160 KB ≪
   32 MB), and with enough blocks in flight latency stops mattering — so
   **bandwidth becomes the constraint**. That demanded new instruments:
   bandwidth metrics.

### The approach
One kernel — the dual global kernel generalized to any cooperative grid —
plus the instruments N requires: per-block counters as a `(blocks, 2)`
array instead of fixed slots, the per-level trace collapsed to one
grid-wide row (a per-block trace would cost 0.4–4.8 GB at these grid
sizes — a documented loss), `owner` widened to int16, `blocks=None`
resolving to the queried cooperative maximum. And the repo's first
bandwidth instrumentation: a **measured D2D copy peak** (192 GB/s here) as
the operational reference, plus a **derived bytes-moved model** (~61
B/pixel, honestly labeled a lower bound) on every run. The experiment is a
blocks × tpb sweep: {1…192} × {64…512} on a big square, a big disk, and
the serpentine.

### Results
| finding | number |
|---|---|
| cooperative capacity, instrumented | 384/192/96/48/24 blocks at tpb 32/64/128/256/512 — **every one = 12,288 threads**: ~104 regs/thread cap the whole grid at 512 threads/SM. The bare twin fits 768/SM, and its 576 blocks at tpb=32 are Ada's hard 24-blocks/SM architectural cap |
| the N blocks pay off | **7.9× vs v2 and 3.9× vs dual-global** at 36–64M px; **15.3× vs `@njit`** at 64M px (132 ms vs 2.03 s) |
| scaling shape | near-linear to ~8–16 blocks, then a **plateau** (~317/~478 Mpx/s square/disk) — and past it a decline: the capacity ends (192–384 blocks) run 10–30% slower |
| best configuration | the rule: **smallest tpb whose grid covers the peak frontier in one pass, blocks maxed for SM-spreading** — 128×32 on the square (4,096 ≥ 4,000-px peak), 128×64 on the disk (8,192 ≥ ~7,600); tpb=512 anti-scales past ~8 blocks |
| `grid.sync` re-priced | ~1.7 µs at 1 block → ~2.2 µs at 192 → ~2.6 µs at 384 — mostly fixed, but the growth is what sinks the capacity-end configs |
| serpentine | **worse than ever**: 145 ms vs the single-block kernel's 71 ms — 65,792 barriers × ~2 µs *is* the runtime |
| bandwidth verdict | plateau at ~11% of measured peak *by the lower-bound model*; ×8 sector inflation puts it ≈ 90% — **consistent with saturation, not yet proof** |
| the stretch scene | 64M px (8000²) completes at 132 ms; **10000² cannot even be attempted** — ~3 GB of host arrays exceed this laptop's free RAM |
| instrumentation overhead | below run-to-run clock noise (−36%…+11% swings); serpentine's +1.9% is the cleanest signal |

### New problems and lessons this chapter exposed
- **Registers now cap the grid, not just the block.** Chapter 2's tpb ≤
  512 finding scaled up: every instrumented configuration lands on the
  same 512 threads/SM ceiling. Register dieting graduated from a nicety to
  the lever that could move the plateau.
- **The plateau is real but unattributed.** The derived model can show a
  plateau exists; it cannot say whether DRAM sectors, L2 behavior, or the
  rear atomic causes it. The model earned its keep — and hit its limit.
  Only `ncu` closes the gap.
- **Small blocks spread the frontier across SMs.** The observed %smid map
  shows consecutive block ids placed on *different* SMs, and grid-stride
  work goes to the first ⌈level/tpb⌉ blocks — so the same 2,000-px level
  runs on 4 SMs at tpb=512 but ~16 at tpb=128. That, plus one-pass
  frontier coverage, picks the winners; past ~48 blocks the growing
  barrier-arrival cost takes the gains back (the optimum is interior).
- **Grid-stride balance is structural.** Levels smaller than the grid feed
  low block ids first; per-block CV reads 686% on tiny scenes and 42% at
  64M px, and none of it is a defect. (Chapter 2's "aggregate metrics can
  lie" lesson, third appearance.)
- **The host-RAM ceiling is now binding.** The GPU dispatches 64M px in
  132 ms and could go far bigger; the laptop cannot hold the arrays.

### The 8-direction experiment (an in-stage bet, v0.7.0)
The question (the user's): does probing 8 neighbors instead of 4 slow the
fill (more work per pixel) or speed it (fewer, wider levels)? Twin
kernels, verbatim except the offset table; same measured cooperative
capacity; depth becomes Chebyshev (square waves — see the wavefront
pair). **Verdict: the width bet wins nearly everywhere.** Squares
1.34–1.75× faster, disks 1.14–1.24×, levels exactly halved, utilization
doubled, new records (disk 503.5 Mpx/s at 48×256; 64M px in 100.3 ms —
20× the CPU); the serpentine pays the pure probe tax (0.80×, levels
32,896→32,641), and at small grids 8-conn loses cell-for-cell — the bet
pays only once the grid can eat the wider levels. Best sweep cells
shifted one tpb step up exactly as the coverage rule predicted.

**The failed prediction is the biggest lesson.** The byte model priced
8-conn at +72% traffic and predicted the plateau-bound blobs would slow;
instead they sped up while sustaining **2× the modeled bandwidth** (42
GB/s, 22% of peak). The model counts logical bytes, but the extra probes
land in 32 B sectors the kernel already touches, and doubled level width
doubles the loads in flight (better-hidden latency). Consequence,
correcting this chapter's earlier framing: **the 4-conn plateau was never
a hard DRAM wall** — its cause shifts toward level-width/latency/barrier
structure, sharpening what `ncu` must arbitrate.

### Open problems → Chapter 4 candidates
1. **`ncu` profiling** — arbitrate the plateau (DRAM sector traffic, L2
   hit rates, rear-atomic contention); the 8-direction result proved the
   4-conn plateau was not DRAM-bound, making the attribution question
   sharper, not moot.
2. **Register dieting** — the bare twin proves 768 threads/SM fit; a
   slimmer instrumented kernel tests whether residency moves the plateau.
3. **Tile-based BFS** for the serpentine — N blocks made the worst case
   worse; only fewer barriers can fix it.
4. **Multi-blob** (connected-component labeling) — unchanged from
   Chapter 2's list. *→ became Chapter 4 (the labeling half; discovering
   the components is still open).*

---

## Chapter 4 — two blobs, N blocks (`ch04_gpu_2blob_nblock/`)

### Inherited problems
1. Every kernel so far fills **one** blob from **one** seed; a second blob
   means a second launch, and the clock adds up.
2. Nothing in the pipeline can say *which* blob a pixel belongs to — the
   fill color is a hardcoded blue constant.
3. Chapter 3's plateau is still unattributed, and its levers (barrier
   count, frontier width, occupancy) are exactly what a second blob
   perturbs.

### Approaches — what each bets
| approach | the bet |
|---|---|
| **label in the queue entry** | the blob id fits in spare bits of the int32 the queue already moves, so labeling is free; and it need never be *computed* — inherited at enqueue, it cannot mix across a ≥2 px gap |
| `lin` vs `xy` entry format | `(x·h+y)<<1\|lbl` (pays an integer div/mod to decode) vs `lbl<<26\|x<<13\|y` (shifts only) — is the divide a real cost? |
| `sequential` | two launches, own queue each. The honest baseline: tA + tB |
| `streams` | two cooperative grids on two CUDA streams — bets the driver co-schedules them |
| `multisource` | both seeds in **one** queue, one launch: levels become max(a,b) not a+b, and every level is twice as wide |

### Results (RTX 4060 Laptop, tpb=256, 48 blocks, 185 GB/s measured peak)
| finding | number |
|---|---|
| multisource vs sequential | **1.05–2.05×** (median), 1.20–1.97× (min-vs-min); controlled 11-round A/B on 15.7M px: **1.79× / 1.58×** |
| asymmetric pair vs the max(tA,tB) "ideal" | **0.91** — both blobs finished faster than sequential finished the big one alone |
| best absolute | 15.7M px in **32.35 ms**; 10.1× the `@njit` two-blob oracle |
| `streams` | serialized (overlap 0.67–1.02) when it ran at all — **no win** |
| `xy` vs `lin` entries | **1.03×** — a wash; the divide was never the bottleneck |
| labeling bandwidth cost | **0 bytes** (structural) |
| 8-conn twins, multisource | 1.00–1.65× over 4-conn — Chapter 3's width lesson compounding |
| small scene (180k px) | **0.5×** — the CPU still wins; two 300² blobs cannot fill 12,288 threads |

### New problems and lessons
- **A launch-uniform value cannot be read from mutable global memory
  *without a barrier between the writes and the read*.** Replacing the
  hardcoded `rear = 1` with an unbarriered read of the host-written seed
  count *deadlocks the GPU*: blocks do not start in lockstep, an early
  block enqueues (mutating rear) before a late block's initial read, and
  the two disagree about the level-0 window forever at `grid.sync`. This
  chapter's fix was a kernel parameter — which is retroactively *why*
  every earlier kernel hardcoded it. (*Corrected by Chapter 5*, which
  needs a GPU-produced seed count no host parameter can carry: the rule
  as first written here overshot. A read behind a
  `grid.sync(); read; grid.sync()` fence sandwich — the same pattern
  every level of every live kernel already uses — is safe, and ch05's
  kernels drop the n_seeds parameter entirely.)
- **Concurrent cooperative grids are a placement lottery, not a
  scheduling guarantee.** Fresh-process probes: 48+48 and 80+80 blocks
  ran, 88+88 wedged permanently, 96+96 co-scheduled once and hung in
  another session; even an 8+8 pair hung once mid-suite, so process
  history matters too. `streams` survives as a documented measurement,
  its tests are opt-in, and the benchmark excludes it.
- **Timing methodology is a correctness concern, not a nicety.** A
  single config's min-to-max spread reaches **73% of its median** here,
  so A-then-B timing invented a "28% slower" result that reversed sign
  under interleaving — and understated the chapter's headline win as
  1.30×. Every A/B now runs interleaved with per-round order reversal and
  reports median *and* min-vs-min; disagreement between them means "below
  the noise floor." Two of this chapter's own predictions (packing tax,
  instrumentation overhead) are marked **unmeasurable** on that basis
  rather than quietly reported.
- **The wavefront tells the story better than the table.** The same
  asymmetric pair rendered on the multisource clock (green finishes early,
  stays light) and replayed on the sequential clock (green runs last,
  comes out dark) is max(tA,tB) vs tA+tB in two pictures.

### Open problems → Chapter 5 candidates
1. **Finding the seeds.** This stage is *given* two seeds. Real multi-blob
   work must discover components itself — connected-component labeling,
   where label inheritance stops being a rider and becomes the algorithm.
   *→ became Chapter 5.*
2. **N blobs.** The `xy` format already carries 6 label bits (64 blobs);
   the open question is scheduling N waves in one queue when they are
   *not* disjoint. *→ became Chapter 5 (which retires the entry format
   for a per-pixel label map).*
3. **`ncu`**, still the arbiter — now also for "labels are traffic-free."
4. **Is the cooperative-launch wedge WSL2-specific?** The same probe on
   native Linux or under MPS would say.

---

## Chapter 5 — N blobs, N blocks, zero seeds given (`ch05_gpu_nblob_nblock/`)

### Inherited problems
1. Every stage so far is *told* where to start; real multi-blob work must
   discover its own components. "How many blobs?" is now the kernel's
   question to answer.
2. The in-entry label format caps at 64 blobs, and discovery labels are
   pixel indices — no entry format holds them.
3. The seed count is produced ON the GPU, but Chapter 4's lesson says the
   kernel may not read the initial rear from mutable state. (Resolved:
   the lesson was narrower than written — see below.)

### Approaches — what each bets
| approach | the bet |
|---|---|
| **canonical labels** (both variants) | a blob's label = its minimum linear index; its seed = that pixel. Deterministic, oracle-computable, and exactly what union-by-`atomicMin` converges to |
| **`seed_merge`** | discovery rides *inside* the fill: flood from every candidate (red pixel with no red lex-predecessor — ≥1 per blob, its lex-min among them), colliding waves union their labels in flight, one flatten at the end. Union work scales with *collisions*. Paint deferred — that is what makes collisions detectable |
| **`ccl_fill`** | solve connectivity *first*: one data-independent union-find pass over every red adjacency (Playne-Stephens style), then exactly one seed per blob feeds the ch03 fill unchanged. Union work scales with *area* |
| per-pixel `label_map` | ch04's free in-entry label cannot survive discovery (the CAS loser must ask "who owns this pixel?"); entries revert to plain `lin`, labels get priced |
| fence-sandwich rear read | the GPU-produced seed count is read via `grid.sync(); read; grid.sync()` — no n_seeds parameter at all |

### Results (RTX 4060 Laptop, tpb=256, 48 blocks, 193 GB/s measured peak)
| finding | number |
|---|---|
| headline | **755,577 blobs** discovered, labeled and filled in **24.8 ms** (one launch, no seeds given) — 21.5× the discovery-included `@njit` baseline |
| `seed_merge` vs `ccl_fill` | merge wins 6/7 scenes, 1.28–2.11×; the disks flip it (0.74×) — 500 staircase candidates make collision traffic expensive |
| union volume | merge: 0–2k unions per scene (~400k on the noise); ccl: one per red adjacency — 15.7M on the squares, a 53 ms prepass ≈ a whole fill |
| discovery tax | 2.10–2.15× vs ch04's given-seeds multisource on comparable scenes |
| serpentine | 65 candidates chop 32,641 levels into **257** → seed_merge beats ccl_fill **88×**; `@njit` still beats both (33k px can't feed 12,288 threads) |
| candidate scan alone | 0.9–1.8 ms at every size — discovery is nearly free; it's the *merging* strategies that differ |
| observer overhead | unmeasurable — five scenes read negative (to −26%), ch04's verdict with stronger evidence |
| **seeding density** (lattice twin: corner rule + every red pixel on an S×S lattice, + a parent-compression pass) | S16 wins every big solid — **flips the disks loss to 2.21×** — S4 takes the comb 5.2× (compression alone: 2.6×), S1 wins the serpentine (0.53 ms, **328×** vs ccl); with the right stride the in-flight variant beats the CCL prepass on **all 7 scenes**. Canonical labels provably stride-invariant. Caveats: the lat kernel's coop capacity is 24 vs 48 blocks (register pressure), and the S64 dip awaits `ncu` |
| **tuning cross-product** (builds × rules × strides, 53 configs/scene) | the fused lat kernel sits **one register over the 128 line** (129 vs v1's 114) — that register cost half the grid; both fixes (`max_registers=128` → 122 regs, and a split build → 114) restore 48 blocks, are near-tied, and beat fused ~25–40% on solids. Optimum stride moves **S16 → S8** with occupancy back; the **interior rule** (lattice seeds need all 8 neighbors red) takes the disks outright (27.5 ms at S1, 3.21× vs v1) and reads flat on the serpentine exactly as the coverage theory demands. The "S64 dip" is really a broad **S≥32 hump** — slower than v1 at 15× fewer levels, `ncu`'s clearest target. Best-vs-ccl now 1.18×–373× |

### New problems and lessons
- **The Chapter 4 deadlock lesson, refined:** the rule was never "no
  reads of rear" but "no *unbarriered* reads." All discovery enqueues
  precede sync #1, nothing moves rear before sync #2 — every thread reads
  the same count. Chapter 4's text is corrected in place.
- **Labels stop being free** once they leave the queue entry: claim
  write + dequeue read + flatten sweep, all priced in the model. ch04's
  zero-byte claim was a property of given, few seeds — not of labeling.
- **Union work should scale with collisions, not area** — the whole
  seed_merge margin on solid scenes. BUF-style 2×2-block unions are the
  obvious lever on the ccl side.
- **Multi-source seeding shortens the BFS clock itself** (serpentine:
  127× fewer levels). Where nearest-candidate depth is acceptable,
  merge's clock is strictly cheaper than any single-seed fill — and the
  seeding-density experiment turned this from an observation into a
  DIAL: plant seeds on a stride-S lattice and the clock collapses to
  ~O(S), with the optimum shape-dependent (S16 solids, S1 geodesic
  monsters, off for already-dense noise).
- **Deferred paint keeps probes hot:** merge's claimed pixels stay red
  until the flatten, so every probe of a claimed neighbor pays a CAS
  attempt — the suspected mechanism behind the disks loss. `ncu` owes
  the verdict.

### Open problems → Chapter 6 candidates
1. **BUF/BKE 2×2-block union-find** to cut ccl's per-adjacency volume.
   *→ became Chapter 6, but with 1×N blocks (runs) rather than 2×2 —
   and the answer was bigger than the question.*
2. **`ncu`** — owed label-map traffic, the negative observer overheads,
   and above all the **S≥32 hump** (slower than v1 at 15× fewer
   levels: a memory-system mystery with a clean reproducer).
3. **Promote a default** — fold the register fix into the published
   kernel and pick stride/rule from a cheap image statistic
   (auto-stride: S8 on solids, S1/interior on staircase or thin
   shapes, off for dense noise).
4. **Recoloring past 6 palette rows** (`label % 6` collides hues at
   N=100+; the label map, not the paint, is ground truth).
5. **The cooperative-launch wedge question stands.** *(Moot for
   Chapter 6, which has no cooperative launch at all.)*

---

## Chapter 6 — runs, not pixels (`ch06_gpu_nblob_runs/`)

### Inherited problems
1. **Every chapter so far moves PIXELS.** A BFS frontier is a list of
   pixel indices; ch05's `ccl_fill` unions once per red *adjacency*
   (15.7M on the squares scene); the label map is one int32 per pixel.
   Five chapters optimized *how* pixels move — barriers, occupancy,
   seeding stride, one register — and never asked whether the pixel is
   the right unit.
2. **ch05's clock is a geodesic clock.** BFS depth *is* the runtime, so
   shape decides speed: the serpentine costs ch03 32,641 barriers.
3. **Nothing has been measured against the machine's real limit.** Five
   chapters compare against a D2D *copy* peak. A pipeline whose two ends
   are a read and a write needs a measured *read* peak and a measured
   *write* peak, or "we are at the floor" is an opinion.

### The observation the chapter is built on
`images/input/input_blobs.png` — 81,000,000 px, 13,451,960 red, 2,522
blobs — contains only **539,207 maximal red runs** (mean 24.9 px). That
is 150× fewer items than red pixels, and connectivity, canonical labels
and the spans to paint are all facts *about runs*. Run ids emitted in
row-major order are ordered exactly like the linear index of their first
pixel, so **ch05's union-by-`atomicMin` protocol, applied verbatim to
run ids, converges to the same canonical label** — the same CPU oracle
judges both chapters, bit for bit.

### Approaches — what each bets
| approach | the bet |
|---|---|
| **the run table** | 539k items is a rounding error, so the entire connected-components problem stops mattering and the clock becomes one read plus one write |
| **1-bit packed mask** | the full-resolution data is a *binary* fact; at 1 bit/px the image is 10.15 MB, not 243 MB, and every pass but the first two stops paying for RGB |
| **six plain kernels, no cooperative launch** | this pipeline is a DAG with no loop, so stream order is the barrier — and full occupancy beats the residency cap that cost ch05 half its grid |
| **paired start/end bit scans** | within a row the k-th start and k-th end belong to the same run, so two warp-scanned bitmask streams pair by index: no cross-word stitching, no serial walk |
| **two contracts, always reported together** | "the runtime" is not one number; RGB-in and packed-mask-in differ by the 243 MB the RGB read costs, and hiding that would be the whole result |

### Results (RTX 4060 Laptop, 24 SMs; measured 193 GB/s copy, 167 read, 169 write)
| finding | number |
|---|---|
| **headline** | `input_blobs.png` — 2,522 blobs, 81 Mpx — recolored in **3.03 ms** (RGB contract) / **1.35 ms** (packed mask) against ch05's **53.26 ms**: **17.6× / 39.5×** |
| labeling alone | **0.70 ms** — every blob discovered and canonically labeled, nothing painted |
| the whole CCL problem | count+scan+emit+merge+flatten = **0.68 ms** of the RGB contract's 3.03; `pack` (1.65) and `paint` (0.65) are the runtime |
| shape stopped mattering | serpentine 2048² **0.49 ms** (37.8×), disk r=2000 **0.92 ms** (29.8×), 100-blob grid **0.61 ms** (41.4×) — one data-independent merge pass, no geodesic clock |
| where runs lose | percolation noise (1.4 px/run, 3.4M runs for 755,577 blobs): **4.1×**, the only scene where the run is barely smaller than the pixel |
| **the RGB wall** | 243 MB read = ~1.3 ms at the measured read peak: **no algorithm recolors this image from RGB in under 1 ms on this hardware.** Sub-millisecond belongs to the mask contract, stated as such |
| scattered writes | 40 MB of red pixels in 75-byte spans = **62 GB/s** against a 169 GB/s streaming write — every paint formulation lands within 5% of it |
| path halving | ch05's read-only `_find` rule relaxed (safe: the write is always a same-class index below `i`, and a root returns before any store) — merge **0.367 → 0.262 ms** |

### Built, measured, thrown away — the load-bearing negatives
- **Word stores in `paint`: no difference** (64 vs 61 GB/s). Three
  per-channel byte stores touching the same sectors *looked* like 3×
  waste; the kernel is scatter-bound, not store-bound. Word stores do
  win past ~64 px/run — the mean here is 25.
- **Skipping unchanged channels: slower** (0.750 vs 0.665 ms). Every
  painted pixel is known to be pure RED, so magenta and orange need one
  store not three, and the test is warp-uniform — the branch still
  costs more than the store it skips.
- **One block per row in `pack`: 2.4× slower.** 324,000 blocks of 256
  threads, and block dispatch outweighs the memory traffic. Capping
  `gridDim.y` at 64: 3.71 → 1.12 ms, the device's full read peak.
- **`prev`/`next` via shfl in `emit`: no difference** — those loads were
  already L1 hits.

### New problems and lessons
- **The unit of work is a design decision, not a given.** Changing
  *what* moves beat five chapters of optimizing *how* it moves, by
  17–40×, and made blob shape irrelevant at the same time.
- **The clock was measuring Python.** `counters.copy_to_device(zeros)`
  is a *synchronous* numba H2D copy; once per run it turned six async
  launches into six host-blocked round trips — **1.72 ms of host enqueue
  time, more than the entire GPU pipeline.** Zeroing inside the scan
  kernel instead: 0.33 ms, and one launch fewer.
- **This GPU idles at 1470 MHz of 3105 and will not boost for short
  kernels separated by syncs.** Cold, the pipeline reads 2.27 ms; hot,
  1.37 ms. The same unchanged `merge` measured 0.38, 0.59 and 0.73 ms
  across one session. Every earlier chapter's "spread is 73% of the
  median" complaint has this underneath it — the benchmark now spins the
  clock up for 8 s and records what it achieved.
- **Short-circuit `and` is three dependent loads.** `a == 255 and
  b == 0 and c == 0` will not request the G byte until the R byte
  returns; `&` on the comparisons keeps all three in flight.
- **`depth` is gone, deliberately.** There is no BFS here, so there is
  no geodesic structure to report — the first chapter to give an
  instrument up rather than add one.

### Open problems → Chapter 7 candidates
1. **`ncu`**, owed since Chapter 3, now with a sharper question: why do
   scattered 75-byte writes cap at 62 GB/s — sector occupancy, DRAM page
   thrash, or write-allocate?
2. **Long runs under-parallelize**: one warp per run leaves the disk
   scene with 4,001 warps. Splitting them needs a prefix sum over run
   lengths — 539k items, essentially free.
3. **The 1.4 px/run case is where a 2×2-block method (BUF/BKE) should
   win** — ch05's original suggestion, now with a measured niche and a
   cheap image statistic to choose by.
4. **CUDA graphs** to collapse six launches (host enqueue is 0.33 ms
   against 1.35 ms of GPU work), and fuse `count` into `pack`.
5. **Sub-millisecond end to end needs the OUTPUT to stop being RGB** —
   a paletted 1 B/px output puts the write at 13 MB instead of 40.

---

## The instruments (how every chapter sees)

Each stage ships the same observability kit, and it keeps paying off:
pixel-exact tests against the CPU reference (visited + **depth maps** —
level-mixing races can't hide); benchmarks with JSON/CSV records and
honest losses, written to a centralized `results/<chapter_id>/` tree via
`shared/results_paths.py`; interactive dashboards — each chapter's own
`benchmarks/visualize.py` renders its own section from its own JSON, and
`dashboard/` (`uv run python -m flood_fill_cuda.dashboard`) assembles
every chapter's section into one whole-project page at
`results/dashboard/project_dashboard.html`; **wavefront renders**
(`ch02_gpu_1blob_2block/benchmarks/wavefront.py`,
`ch03_gpu_1blob_nblock/benchmarks/wavefront.py`) that replay the recorded
`depth`/`owner` maps as GIFs — the same BFS, but each partitioning's
territories visibly different (the N-block render's ownership *speckle*
is itself evidence: spatial scatter is the bandwidth chapter's
sector-inflation story made visible); **bare twin kernels** so the
instrumentation itself stays priced; and, since Chapter 3, **bandwidth
instruments** (`shared/bandwidth.py`, promoted from
`ch03_gpu_1blob_nblock/` since ch04 reuses it) — a measured D2D copy peak
as the only reference figures are compared against, and a per-run derived
bytes-moved model that is always labeled the lower bound it is.

## Extending this file

When a stage lands, append a chapter in the same shape: *inherited
problems → approaches (a table: what each one bets) → results (a table:
measured, losses included) → new problems and lessons → open problems*.
Update the chain diagram at the top. If a result contradicts an earlier
chapter's framing, correct the earlier chapter in place and say so — this
document records what we currently believe, not what we once hoped.
