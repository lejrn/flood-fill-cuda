# Two blobs, N blocks (`dual_blob/`)

The image levels up: a white background and **two similar, separate red
blobs**. Both must be flooded and recolored — blob 0 **blue**, blob 1
**green**, painted by the kernel itself — and the two floods should
ultimately run *in parallel*. This stage takes the multi_block stage's
proven cooperative global-queue kernel and asks three questions:

1. **How does a label reach the paint site?** (two queue-entry formats,
   benchmarked against each other)
2. **What does "in parallel" actually buy?** (three mechanisms,
   benchmarked against each other)
3. **What breaks on the way?** (two genuine findings: a grid.sync
   deadlock from a racy init read, and nondeterministic wedging of
   concurrent cooperative grids)

Everything load-bearing is inherited unchanged from
[`../../single_blob/multi_block/`](../../single_blob/multi_block/):
the claim protocol (bounds → is_red → `atomic.cas(visited)` →
warp-aggregated enqueue), two `grid.sync()` per level, every counter's
meaning, and the structural no-overflow argument.

## The label rides inside the queue entry

Two blobs in one queue means a thread that dequeues a pixel must know
which blob it belongs to (blue or green?). Nobody computes that — the
label is **inherited like a surname**: the host pre-tags the two seed
entries (blob 0 / blob 1), and every thread stamps its own label onto
each neighbor it enqueues. Since ≥ 2 px of white separates the blobs and
a label only travels by inheritance, mixing is impossible by
construction. The paint site becomes a 2-row constant palette lookup
instead of a hardcoded blue triple.

The label costs **zero extra arrays and zero extra bytes** — it lives in
spare bits of the int32 entry the queue already moves. *How* it is
encoded is itself a measured bet, so two verbatim twin families coexist:

| family | entry encoding | decode cost | constraint |
|---|---|---|---|
| `lin` | `(x*height + y) << 1 \| label` | int div + mod (~20-40 cycles — no hardware integer divide on GPU) | `w*h < 2^30` |
| `xy` | `label << 26 \| x << 13 \| y` | shifts + masks (~3 cycles) | dims ≤ 8192; up to 64 blobs |

The `lin` family is a one-bit diff from the published multi_block kernel,
so labeling cost benchmarks cleanly against the single-blob baseline. The
`xy` family deletes the per-pixel div/mod entirely at equal traffic — the
**decode-tax experiment**: does killing the divide matter, or does warp
parallelism hide its latency anyway?

## Three ways to run "in parallel"

| mode | mechanism | cost model |
|---|---|---|
| `sequential` | two single-seed launches back-to-back, own queue each | tA + tB (baseline) |
| `streams` | the same two launches on two CUDA streams, each grid sized to ⅓ capacity | max(tA', tB') **if** the driver co-schedules |
| `multisource` | both seeds pre-loaded into ONE shared queue, one launch | ~max(tA, tB) + width gains |

The `streams` mode realizes the "each blob gets its own queue" instinct
most literally; `multisource` shows a shared queue can label just as well.

## Finding 1: the initial-rear race (a measured deadlock)

The multi-seed init looked like a one-line fix: replace the hardcoded
`rear = 1` with `rear = q_state[Q_REAR]` (read the seed count the host
loaded). **It deadlocked the GPU on its first non-trivial launch.**
Blocks do not start in lockstep: an early block processed its seed and
enqueued neighbors — mutating `q_state` — *before* a late block performed
its initial read. The threads then disagreed on the level-0 window,
diverged in loop trip count, and hung at `grid.sync` forever (100% GPU,
no progress, process kill required).

The fix: the seed count arrives as a **launch-uniform kernel parameter**
(`n_seeds`), not a read of mutable global memory. This is exactly *why*
the single-blob kernels hardcoded `rear = 1` — the number of initial
entries must be baked in per launch, one way or another.

## Finding 2: concurrent cooperative grids are a placement lottery

A cooperative kernel requires ALL its blocks resident simultaneously.
Two cooperative grids launched on two streams must *share* the device —
and whether they can is decided by a block-placement race, not a rule.
Probed on this GPU (tpb=64, single-grid capacity 192 blocks; each row is
one fresh-process pair launch, 300 s watchdog):

| per-launch blocks | pair total | outcome |
|---|---|---|
| 8   | 16  | ran (overlap ≈ 0.7) |
| 48  | 96  | ran (overlap 0.67) |
| 80  | 160 | ran, serialized (overlap 1.02) |
| 88  | 176 | **WEDGED** — killed at 300 s |
| 92  | 184 | ran, pathologically: 183.8 ms wall for a ~6 ms job (overlap 0.64) |
| 96  | 192 | ran with real co-scheduling once (overlap 1.52) — **and wedged forever in a separate session** |

The same 96+96 config deadlocked in one session and co-scheduled in
another. Near capacity you may get co-scheduling, serialization, thrash,
or a permanent wedge — nondeterministically. Consequently `mode="streams"`
defaults each grid to **⅓ of capacity** (pair ≤ ⅔, well under the
observed cliff), and the correctness tests pin small explicit grids.

The wedge is not theoretical for this repo: it hung the test suite for
82 minutes, and a later run hung again on an **8+8** pair — a
configuration that had run fine standalone — after other fills had
already run in the same process. So the cliff is not a simple block-count
threshold; process history matters too.

**Consequences, applied:**

- `mode="streams"` still exists (it is the experiment) but defaults each
  grid to ⅓ of capacity and carries this warning.
- Its tests are **opt-in**: `DUAL_BLOB_STREAMS=1 uv run pytest …`. A test
  that can hang the runner forever is worse than an unrun test.
- `benchmark.py` **excludes streams entirely.** Measuring it inline would
  risk the whole session for a mechanism that already lost: every pair
  that completed showed overlap 0.67–1.02, i.e. serialized, no benefit.
  The fresh-process probe above *is* its experiment.

## Predictions (written before the benchmark ran)

| bet | prediction | reasoning |
|---|---|---|
| sequential | tA + tB | by construction |
| streams | ≈ sequential-at-⅓-capacity; overlap ≈ 1; **no win** | the probe: co-scheduling is a lottery, serialization is the mode; the safe operating point wastes ⅔ of the device per launch |
| multisource, equal pair | **1.2–1.5× vs sequential** | same total pixels (no throughput win if bandwidth-bound), but HALF the barriers (levels = max, not sum: ~2,800 grid.syncs saved ≈ 10 ms on a ~40 ms scene) plus doubled per-level width → better warp engagement while frontiers are narrow (Chapter 3's spreading + the 8-direction width lesson) |
| multisource, asym pair | multi_vs_ideal ≈ 1.0; speedup vs sequential only ≈ 1.05 | the whole win is bounded by max(tA,tB): the big blob dominates both clocks |
| xy vs lin (decode tax) | wash (±3%) on big solid scenes; xy ahead only if ALU/latency-bound | thousands of resident warps hide a 20–40-cycle divide; traffic is identical |
| packing tax (lin seq-A vs published multi_block) | < 2% | one extra shift+OR per enqueue, one shift+AND per dequeue |
| label bandwidth cost | **zero** by construction | the label occupies bits the queue entry already moved; `bandwidth.py` is a re-export, unchanged |

## Results

*(pending — filled in by `benchmark.py` in the next commit)*

## Wavefront renders

*(pending — `wavefront.py`, next commit: multisource vs sequential-replay
GIFs of the same asymmetric pair — two clocks, same pixels)*

## Files

| file | role |
|---|---|
| `kernels.py` | 8 verbatim-twin kernels: {lin, xy} × {instrumented, bare} × {4-conn, 8-conn} |
| `flood_fill.py` | `flood_fill(img, seeds, mode=..., entry_format=...)` → `DualBlobResult` (+ per-launch `LaunchStats`) |
| `scenes.py` | two-blob builders: `two_squares_scene`, `two_disks_scene`, `asym_squares_scene`, `two_pixels_scene` (all `(img, seeds)`, gap ≥ 2 enforced) |
| `reference.py` | single-seed oracles re-exported + `cpu_flood_fill_two` merged oracle (asserts the blobs are truly disjoint) |
| `bandwidth.py` | re-export of multi_block's model — identical because labeling moves zero extra bytes |
| `test_correctness.py` | ~60 tests: every mode × format × connectivity vs the merged oracle, exact colors, mode equivalence, accounting, validation |
| `benchmark.py` | the modes + decode-tax head-to-head |
| `wavefront.py` | blob-hued timeline GIFs/PNGs |

## Run

```
uv run pytest src/gpu/multi_blob/dual_blob/test_correctness.py -v
uv run python src/gpu/multi_blob/dual_blob/benchmark.py
uv run python src/gpu/multi_blob/dual_blob/wavefront.py

# streams-mode tests are opt-in — they can hang the GPU (see Finding 2)
DUAL_BLOB_STREAMS=1 uv run pytest src/gpu/multi_blob/dual_blob/test_correctness.py -k streams
```

## Open problems → next stages

1. **N blobs.** The xy format already carries 6 label bits (64 blobs);
   the palette and seed API generalize trivially. The real question is
   scheduling N waves in one queue vs frontier interference — none here,
   because blobs are disjoint.
2. **Finding the seeds.** This stage is *given* its two seeds. True
   multi-blob processing (the roadmap's Stage 5) must discover components
   itself — connected-component labeling, where the label-inheritance
   trick becomes the core algorithm instead of a rider.
3. **ncu ground truth.** Still the arbiter for every bandwidth claim,
   now including "labels are traffic-free."
4. **The streams lottery, elsewhere.** Is the wedge WSL2-specific? A
   native-Linux or MPS run of the same probe would say.
