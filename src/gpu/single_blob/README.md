# The Evolution of a Flood Fill

A living document. Each chapter follows the same loop: **the problems we
inherited → the approaches that attack them → what the measurements said →
the new problems those measurements exposed** — which become the next
chapter's inheritance. Every number below is measured on this repo's RTX
4060 Laptop GPU (24 SMs), every kernel is proven pixel-exact against the
same `@njit` CPU reference, and losses are reported as plainly as wins.

```
CPU BFS ──"one core is serial"──► 1 block ──"one SM is 4% of the GPU"──► 2 blocks ──"2 SMs are 8%"──► N blocks ──► …
                                     │                                      │                            │
                          "the queue doesn't fit"                "how should two blocks share    "does it keep scaling?
                                     ▼                            one BFS? where do they run?"    what stops it — and is
                                v2 spill tier                     split / global / dirsplit /     it bandwidth?"
                                                                  pinned                          global queue × 48 blocks;
                                                                                                  plateau at 512 threads/SM
```

---

## Chapter 0 — the baselines and the broken prototype

**The problem:** flood-filling a blob is BFS — inherently level-by-level,
but *within* a level every pixel is independent. A CPU does them one at a
time; a GPU should do them together.

**The baselines:** pure-Python BFS (the classic algorithm, honest but
~100–1000× slow) and an `@njit`-compiled level-synchronous BFS
(`single_block_shared/reference.py`) — the bar every kernel must beat *and*
the oracle every kernel must match pixel-for-pixel.

**The prototype's lesson:** the historical `single_block.py` had a
non-wrapping queue that silently dropped pixels, marked non-red pixels
visited (CAS before the color check), and hardcoded 64 threads. Its real
legacy was a checklist of what correctness requires: a queue that can't
overflow silently, a claim protocol that can't double-visit, and tests that
compare *depth maps*, not just pixel counts.

---

## Chapter 1 — one blob, one block (`single_block_shared/`)

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

## Chapter 2 — one blob, two blocks (`dual_block/`)

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
   blocks — the `../multi-blocks/` and `persistent/` designs revisited
   with this project's rigor (exact tests, bare twins, smid observation,
   per-block traces). Expected new problems: barrier cost grows with grid
   size; the global queue's single rear counter becomes a contention point
   (warp-aggregation may not be enough). *→ became Chapter 3.*
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

## Chapter 3 — one blob, N blocks (`multi_block/`)

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
| cooperative capacity, instrumented | 192/96/48/24 blocks at tpb 64/128/256/512 — **every one = 12,288 threads**: ~104 regs/thread cap the whole grid at 512 threads/SM (the bare twin fits 768) |
| the N blocks pay off | **7.9× vs v2 and 3.9× vs dual-global** at 36–64M px; **15.3× vs `@njit`** at 64M px (132 ms vs 2.03 s) |
| scaling shape | near-linear to ~8–16 blocks, then a **hard plateau** (~316/~467 Mpx/s square/disk); 96 and 192 blocks move nothing |
| best configuration | **48 × 128** in both big scenes — at equal thread counts, many small blocks beat few big ones; tpb=512 anti-scales past ~8 blocks |
| `grid.sync` re-priced | ~1.73 µs at 1 block → ~2.2 µs at 192 — the barrier's cost is mostly *fixed*, prediction "grows with grid" barely materialized |
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
- **The scheduler beats big blocks.** 48×128 outruns 24×256 and every
  tpb=512 configuration at the same thread count — granularity helps the
  hardware hide stalls.
- **Grid-stride balance is structural.** Levels smaller than the grid feed
  low block ids first; per-block CV reads 686% on tiny scenes and 42% at
  64M px, and none of it is a defect. (Chapter 2's "aggregate metrics can
  lie" lesson, third appearance.)
- **The host-RAM ceiling is now binding.** The GPU dispatches 64M px in
  132 ms and could go far bigger; the laptop cannot hold the arrays.

### Open problems → Chapter 4 candidates
1. **`ncu` profiling** — arbitrate the plateau (DRAM sector traffic, L2
   hit rates, rear-atomic contention); the derived model's attribution gap
   is now the project's central open question.
2. **Register dieting** — the bare twin proves 768 threads/SM fit; a
   slimmer instrumented kernel tests whether residency moves the plateau.
3. **Tile-based BFS** for the serpentine — N blocks made the worst case
   worse; only fewer barriers can fix it.
4. **Multi-blob** (connected-component labeling) — unchanged from
   Chapter 2's list.

---

## The instruments (how every chapter sees)

Each stage ships the same observability kit, and it keeps paying off:
pixel-exact tests against the CPU reference (visited + **depth maps** —
level-mixing races can't hide); benchmarks with JSON/CSV records and
honest losses; interactive dashboards (`single_block_shared/visualize.py`,
`dual_block/visualize.py` — the latter renders both stages);
**wavefront renders** (`dual_block/wavefront.py`,
`multi_block/wavefront.py`) that replay the recorded `depth`/`owner` maps
as GIFs — the same BFS, but each partitioning's territories visibly
different (the N-block render's ownership *speckle* is itself evidence:
spatial scatter is the bandwidth chapter's sector-inflation story made
visible); **bare twin kernels** so the
instrumentation itself stays priced; and, since Chapter 3, **bandwidth
instruments** (`multi_block/bandwidth.py`) — a measured D2D copy peak as
the only reference figures are compared against, and a per-run derived
bytes-moved model that is always labeled the lower bound it is.

## Extending this file

When a stage lands, append a chapter in the same shape: *inherited
problems → approaches (a table: what each one bets) → results (a table:
measured, losses included) → new problems and lessons → open problems*.
Update the chain diagram at the top. If a result contradicts an earlier
chapter's framing, correct the earlier chapter in place and say so — this
document records what we currently believe, not what we once hoped.
