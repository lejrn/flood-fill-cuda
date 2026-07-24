# Chapter 5 — N blobs, N blocks, zero seeds given (`ch05_gpu_nblob_nblock/`)

**Finding the seeds.** Every chapter so far was *told* where to start;
this one is handed an image and must answer "how many blobs are there,
where are they, and which pixel belongs to which?" — one canonical seed
and label per blob, any number of blobs, one cooperative launch,
8-connectivity throughout.

## Inherited problems

1. **ch04 was given its two seeds.** Real multi-blob work must discover
   components itself — connected-component labeling, where label
   inheritance stops being a rider and becomes the algorithm.
2. **The in-entry label format caps at 64 blobs** (6 spare bits in the
   `xy` entry). Discovery labels are *pixel indices* — no entry format
   holds them.
3. **The n_seeds kernel parameter.** ch04's deadlock lesson ("never read
   the initial rear from q_state") assumed the host *knows* the seed
   count. Discovery produces it on-GPU — so either the lesson blocks the
   chapter, or the lesson was narrower than written.

## The shared foundation

Both variants agree on one canonicalization: **a blob's label is the
minimum linear index (`x*height + y`) over its pixels; its canonical
seed is that pixel.** Deterministic, a pure function of the image, and
exactly what union-by-`atomicMin` over pixel indices converges to. The
union protocol (`_find` chase + `atomic.min` on the larger root, retry
from the returned value on a lost race) is shared verbatim by both
kernels; parents only ever decrease, so it cannot livelock, and the
class minimum — which no same-class value can overwrite — is the one
guaranteed survivor.

A **candidate** is a red pixel with no red lex-predecessor (none of
`(x-1,y±1)`, `(x-1,y)`, `(x,y-1)` red). Two structural facts carry the
chapter: every blob's lex-min pixel is a candidate (so the union-find
root over candidates IS the canonical label), and no two candidates are
8-adjacent (so the candidate count is scene-shaped, not blob-counted —
a rasterized disk's staircase arc already yields ~250 of them).

## Approaches — what each bets

| approach | the bet |
|---|---|
| **`seed_merge`** | discovery can ride *inside* the fill: flood from every candidate at level 0, let colliding waves union their labels in flight, flatten once at the end. Union work scales with *collisions*, not area. Paint is deferred — that deferral is what makes collisions detectable (claimed pixels stay red, so the CAS loser can look up the winner). |
| **`ccl_fill`** | connectivity should be solved *first*: one data-independent union-find pass over every red adjacency, then exactly one seed per blob feeds the ch03 fill unchanged. Union work scales with *area*, but no mid-flight protocol to defend. |
| **per-pixel `label_map`** (both) | ch04's "label rides free in the entry" cannot survive discovery — the CAS loser must ask *who owns this pixel*, and provisional labels are pixel indices. Entries revert to plain `lin`; the label traffic is priced in the model, not hidden. |
| **fence-sandwich rear read** (both) | the GPU-produced seed count can be read safely inside the kernel: `grid.sync(); rear = q_state[Q_REAR]; grid.sync()` — the pattern every live kernel already uses per level. The kernels take **no n_seeds parameter at all**. |

## Results (RTX 4060 Laptop, tpb=256, 48 blocks both variants, 193 GB/s measured peak)

| scene | filled px | blobs | candidates | `seed_merge` | `ccl_fill` | ccl/merge (min) | @njit |
|---|---|---|---|---|---|---|---|
| two squares 2800² | 15.7M | 2 | 2 | **50.7 ms** | 106.9 ms | 2.11× (1.99×) | 571.6 ms |
| two disks r=1400 | 12.3M | 2 | 500 | 78.8 ms | **58.2 ms** | 0.74× (0.70×) | 325.9 ms |
| asym 4000²+800² | 16.6M | 2 | 2 | **56.8 ms** | 85.9 ms | 1.51× (1.49×) | 592.7 ms |
| grid of 100 blobs | 13.0M | 100 | 100 | **32.7 ms** | 53.0 ms | 1.62× (1.46×) | 419.6 ms |
| random noise 4000² | 4.8M | **755,577** | 1,154,024 | **24.8 ms** | 31.7 ms | 1.28× (1.25×) | 532.5 ms |
| comb, 2000 teeth | 4.8M | 1 | 2,000 | **52.9 ms** | 73.6 ms | 1.39× (2.07×)* | 345.8 ms |
| serpentine 256² | 33k | 1 | 65 | **1.96 ms** | 173.0 ms | **88.4×** (89.5×) | **0.79 ms** |

\* comb's median and min forms disagree wildly (1.39× vs 2.07×) — the
session's drift casualty; direction is consistent, magnitude is not.

| finding | number |
|---|---|
| headline | **755,577 blobs** discovered, labeled and filled in **24.8 ms** — 21.5× the discovery-included `@njit` baseline, ~11,800× past ch04's 64-label cap |
| `seed_merge` vs `ccl_fill` | merge wins 6 of 7 scenes (1.28–2.11×); disks are the exception (0.74×) |
| union volume | merge: 0–2,000 unions per scene, ~400k on the noise — vs ccl: **one union per red adjacency** (15.7M on the squares). cclp phase alone = 53 ms there, ≈ a whole fill |
| discovery tax | 2.10–2.15× vs ch04's given-seeds multisource on the square scenes (4.06× on the disks, where merge also loses the fill) |
| discovery phase alone | candidate scan: 0.9–1.8 ms everywhere; CCL pass: 2.9–54 ms, area-shaped |
| serpentine | 65 candidates chop a 32,641-level geodesic into **257 levels** — an 88× win over the canonical-seed fill, and a *new depth semantics* (nearest candidate), not just a faster clock. `@njit` still beats both (0.79 ms): 33k px cannot feed 12,288 threads |
| observer overhead | **unmeasurable** — five scenes read *negative* (bare slower than instrumented, to −26%); same verdict as ch04, stronger evidence |

## New problems and lessons

- **The ch04 deadlock lesson was narrower than written.** The rule was
  never "a launch-uniform value cannot be read from mutable global
  memory" — it was "cannot be read *without a barrier between the writes
  and the read*." All discovery enqueues structurally precede sync #1,
  nothing moves rear until after sync #2, so every thread reads the same
  count. The n_seeds parameter is gone, and chapter 4's text stands
  corrected in place.
- **Labels stop being free.** The per-pixel label map costs real traffic
  (priced in the model: write at claim, read at dequeue, sweep + find at
  flatten). ch04's zero-byte claim was a property of *given, few* seeds,
  not of labeling.
- **Union work should scale with collisions, not area.** That is the
  whole seed_merge margin on solid scenes: 0–2,000 unions against ccl's
  one-per-adjacency 15.7M. BUF-style 2×2-block unions would cut ccl's
  volume ~4× and is the obvious next lever.
- **Multi-source seeding is a levels lever nobody asked for.** The
  serpentine result says candidate seeding doesn't just discover — it
  *shortens the BFS clock* wherever candidates are spread along the
  geodesic. If only labels (not canonical-seed depths) are needed,
  merge's nearest-candidate depth is strictly cheaper.
- **Deferred paint keeps probes hot.** seed_merge's fill probes never
  benefit from paint pruning (claimed pixels stay red until the flatten),
  so every probe of a claimed neighbor pays a CAS attempt — the likely
  mechanism behind the disks loss (500 staircase waves → long internal
  seams → maximal collision-branch traffic). `ncu` is the arbiter.
  (*Superseded in practice by the seeding experiment below: at stride 16
  the disks flip to a 2.21× seed_merge win.*)

## Seeding density — the stride experiment

Raised while reading the phase table above: the corner rule plants ONE
seed on a solid rectangle, so the fill clock runs O(blob diameter)
levels while the flatten costs ~2%. What if the scan planted more? The
`seed_merge_lat` twin seeds the corner-rule set PLUS every red pixel on
an S×S lattice (S a host parameter; canonical labels provably
stride-invariant, since the corner rule — and with it the lex-min lemma
— stays included), and adds a COMPRESS phase that rewrites every
retired parent slot to its true root in one pass, so the repaint's find
is ≤ 1 hop at any seed count.

Sweep (median ms, interleaved; * = scene's best in-flight config; full
curve in `results/.../seeding_*.json` and the dashboard card):

| scene | v1 | S0 | S1 | S4 | S16 | S64 | S256 | ccl |
|---|---|---|---|---|---|---|---|---|
| two squares 2800² | 55.7 | 57.4 | 90.3 | 67.9 | **49.3*** | 67.1 | 64.7 | 117.4 |
| two disks r=1400 | 90.0 | 107.8 | 69.7 | 55.0 | **40.7*** | 53.3 | 50.0 | 63.2 |
| asym 4000²+800² | 61.2 | 63.0 | 97.4 | 72.5 | **54.1*** | 71.2 | 63.0 | 91.4 |
| grid of 100 blobs | **35.0*** | 36.7 | 77.2 | 56.2 | 40.0 | 50.2 | 40.7 | 59.1 |
| random noise 4000² | **24.8*** | 33.3 | 33.8 | 37.9 | 36.5 | 34.6 | 33.2 | 27.6 |
| comb, 2000 teeth | 67.9 | 25.9 | 26.3 | **13.0*** | 27.8 | 24.4 | 26.6 | 85.1 |
| serpentine 256² | 1.88 | 1.72 | **0.53*** | 1.11 | 1.73 | 1.78 | 1.74 | 174.1 |

**Findings:**

- **The hypothesis holds.** S16 collapses the big solids' clocks
  (2,800–4,000 levels → 16) and wins all three — including **flipping
  the disks**, this chapter's one seed_merge loss, to 2.21× over v1 and
  1.55× over ccl. With the right stride, the in-flight variant now
  beats the CCL prepass on **all seven scenes** (1.11×–328×).
- **The compression pass alone is worth 2.6× on the comb** (v1's 47.6 ms
  chain-walking flatten → 2.1 ms at S0), and S4's lattice additionally
  halves the comb's levels: 13.0 ms total, 5.2× over v1.
- **The optimum is shape-dependent, and the extremes are real
  configurations**: S1 (every red pixel a wave, levels=1) loses on
  solids as predicted — but wins the serpentine outright (0.53 ms,
  328× over ccl's 174 ms geodesic crawl) and even beats v1 on the
  disks. Already-dense scenes (random noise: 1.15M corner candidates)
  gain nothing and keep v1 best — densifying what is already dense
  only pays the handicap below.
- **The handicap: the lat kernel's cooperative capacity is 24 blocks vs
  v1's 48** (register pressure from the extra phase/parameter). Every
  stride pays it — S0, which does the same work as v1 plus one cheap
  sweep, reads ~3-8% slower on that alone. The S16 wins stand DESPITE
  half the grid; a register diet is an obvious lever.
- **Unexplained: the S64 dip.** On the solids S64 is slower than both
  S16 and S256 despite a monotone level count — not a drift artifact
  (consistent across scenes). `ncu` owes the answer.

## The tuning cross-product — builds × rules × strides

The follow-up experiment (user-requested, full cross-product: 53
configs per scene): the register handicap attacked two ways, the
interior seeding rule, and the stride gaps {8, 32, 128} filled in.

**The register story, measured** (65,536 regs/SM ÷ 256 threads ÷ 2
blocks = 128 regs/thread is the two-blocks-per-SM line):

| build | regs/thread | coop blocks @tpb256 |
|---|---|---|
| v1 (corner) | 114 | 48 |
| lat fused | **129** | **24** |
| lat r128 (`max_registers=128`) | 122 | 48 |
| lat split (core P0–P2 + plain cleanup) | 114 | 48 |

The fused kernel sat **one register** over the line — and the cap
landed at 122, *below* its own limit: the compiler had the slack all
along and simply didn't try. No meaningful spill cost was observed
(r128 ≈ split everywhere, within drift).

**Best config per scene** (medians, interleaved; full 53-config tables
in `results/.../tuning_*.json` and the dashboard card):

| scene | best | ms | vs v1 | vs ccl |
|---|---|---|---|---|
| two squares 2800² | split_L8 | 32.7 | 1.61× | 3.49× |
| two disks r=1400 | **split_I1** | 27.5 | 3.21× | 2.42× |
| asym 4000²+800² | r128_L8 | 35.7 | 1.69× | 2.50× |
| grid of 100 blobs | split_L8 | 26.8 | 1.20× | 2.02× |
| random noise 4000² | r128_L1 | 24.9 | 1.02× (wash) | 1.18× |
| comb, 2000 teeth | fused_L8 | 13.4 | 5.00× | 6.28× |
| serpentine 256² | r128_L1 | 0.49 | 4.07× | **373×** |

**Findings:**

- **Occupancy was the bottleneck, not the fix's flavor.** r128 and
  split are near-tied; both beat fused by ~25–40% on the solids. The
  one-register line was worth more than any seeding refinement.
- **With 48 blocks restored, the optimum stride moves S16 → S8** on
  every solid scene — the 24-block build couldn't exploit the finer
  frontier, the 48-block builds can.
- **The interior rule earns its keep exactly where predicted**: the
  disks' best config is interior seeding at S1 — every 8-red-neighbor
  pixel a seed, the noisy staircase boundary seedless (27.5 ms, from
  90 ms at the corner rule). And on the serpentine it measures as
  theory demands: a 1-px snake has no interior pixels, so every
  interior row reads flat corner-only times — the coverage argument,
  benchmarked.
- **The "S64 dip" resolves into a broad S≥32 hump**: on the solids,
  everything from S32 to S256 is *slower than v1* despite 15–90×
  fewer levels (e.g. two_sq r128: S16 36.3 → S32 54.1; S256 58.9 vs
  v1 52.9). Fewer barriers with worse time means the loss is in the
  memory system, not the sync count — `ncu`'s clearest target yet.
- Best-config-vs-ccl now spans **1.18×–373×** across all seven scenes.

## Open problems → Chapter 6 candidates

1. **BUF/BKE block-based union-find** — 2×2-block unions to cut ccl's
   per-adjacency volume; the literature's standard next step.
2. **`ncu`** — owed the label-map traffic claim, the negative observer
   overheads, and above all the **S≥32 hump** (slower than v1 at 15×
   fewer levels — a memory-system mystery with a clean reproducer).
3. **Promote a default** — the cross-product says "a 48-block build at
   S8 (interior on staircase-heavy shapes, S1 on thin ones)"; folding
   the register fix into the published kernel and picking stride/rule
   from a cheap image statistic (**auto-stride**) is the natural
   Chapter 6.
4. **Recoloring past 6 palette rows** — `label % 6` collides adjacent
   hues at N=100+; a host-side dense re-rank (or per-label LUT like the
   wavefront's) would give every blob its own color.
5. **The wedge question stands** — is the cooperative-launch lottery
   WSL2-specific?

## Files

Same kit as every chapter: `kernels.py` (both fused kernels + bare
twins + standalone phase kernels), `flood_fill.py` (driver — note the
missing seeds parameter), `cpu_oracle.py` (canonical CCL + two exact
depth oracles), `scenes.py` (seedless builders incl. the U, the comb
and the noise), `test_correctness.py` (151 tests), `benchmarks/`
(benchmark, wavefront, visualize → dashboard §3).

Money shot: `results/ch05_gpu_nblob_nblock/wavefront/u192_merge_prov.gif`
next to `u192_merge_final.gif` — two candidate waves racing down a U,
colliding at the bridge, and the union erasing the seam.

Run:

    uv run pytest src/flood_fill_cuda/chapters/ch05_gpu_nblob_nblock/test_correctness.py
    uv run python -m flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.benchmarks.benchmark
    uv run python -m flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.benchmarks.wavefront
