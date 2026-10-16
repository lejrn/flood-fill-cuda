# ch07 under `ncu`, before and after the switch change

Two profiles of Chapter 7 with Nsight Compute.

- **Before**: the old default (two `grid.sync` per round, ch06's pack,
  the plain paint), taken before any switch existed.
- **After**: the new default (`one_sync` + `tile_paint`: one
  `grid.sync` per round, the same pack, the tile paint), on the same
  scenes with the same driver policy. One extra report covers the
  opt-in fused pack, to show why it lost on dense blobs.

Numbers marked *inferred* are arithmetic on measured values. Times in
ms come from the committed session `tiles_20261016T064724Z`, unless a
line says otherwise. Everything else is a counter, a sample count or a
cycle count from the reports.

## Setup

- Nsight Compute 2025.2.1, RTX 4060 Laptop (AD107, 24 SMs), WSL2.
  `--set full` (40 kernel-replay passes), caches flushed before every
  pass, `--clock-control base`.
- The rounds kernel is the bare twin (no counters) at its default grid,
  72 x 128: 129 registers before, 135 after. One more report runs the
  same binary at 1 x 32, where every sample belongs to the one warp.
- The driver builds a `benchmark.SCENES` scene, runs `TileFill` 3 times
  on reusable buffers, and restores the image from a device copy before
  each run. For pack and paint it also reads the benchmark's 96 MB L2
  flush buffer first. The profiled launch is the second real run.
  Every run checks that every red pixel was filled.
- The barrier probe runs `probes._barrier_rounds` (before: two
  `grid.sync` and a read per round) or `probes._barrier_rounds_1`
  (after: one `grid.sync` and a read) at 904 and 3,616 rounds. The
  difference gives the cost per round.

**Clocks.** Before, the base-clock lock did not hold: the SM clock read
605-1,140 MHz across reports. After, every report ran at 1,137-1,140
MHz.

Both profiles are compared in **cycles**, which do not move with the
clock. They match the sessions at 2,070 MHz: cycles per round agree
within 0.3-3.3% before and 1.4-1.8% after.

## The short answers

**1. One `grid.sync` halves the barrier ritual (מחסום).** The probe's
ritual falls from 4,517 to 2,387 cycles per round at 72 x 128 (-47%),
and from 3,048 to 1,747 at 1 x 32 (-43%).

**2. The serpentine rounds lose 27% of their cycles.** A round falls
from 9,015 to 6,578 cycles (`serpentine_256`) and from 8,378 to 6,038
(`zz_serp_1k`).

The ritual's share of the round falls from 50-54% to 36-40%. It is
still the largest part of a one-tile round: 87-89% of the warp samples
on these scenes still sit in `grid.sync` internals.

**3. On `sq_8000` the rounds kernel loses 14% of its cycles** (21,613
to 18,493 per round). The ritual's share falls from 20.9% to 12.9%,
and the warp samples in barrier code from 41.2% to 34.0%.

Nothing else moved. The visit is the same code: 1.77 closure trips,
39.4 `SHFL` and 3.86 atomic instructions per visit, 18.9 sectors per
load request. The rest of each round fell 5-7% (*inferred*: cycles per
round minus the ritual).

**4. The tile paint is no longer a latency (השהיה) chain.** On
`sq_8000` the plain paint's top stall was long scoreboard (לוח
תלויות): 51.5% of samples. The tile paint loads all of a tile's words
before any store, and that stall falls to 5.8%.

Its issue slots are now 74.7% busy, so instruction count is the next
wall (49.4 warp instructions per mask word). The session's paint phase
fell from 1.035 to 0.808 ms.

On the thin strip the tile paint reads 0.1 MB (the 64 KB of stamps)
instead of 2.1 MB of `reached` words. It runs 4.7 warp instructions per
mask word instead of 32.4, and its phase fell from 0.154 to 0.023 ms.

**5. The pack is unchanged**: the same kernel (41 registers, 1.155 ms
on `sq_8000`), still about 1.5x a pure read of its bytes.

**6. Why the fused pack lost on dense blobs (opt-in).** Folding the RGB
read into the rounds kernel links it at 247 registers. So only 48
blocks of 128 fit (192 warps, 16.8% achieved occupancy (תפוסה), against
25%).

On `sq_8000` its rounds kernel takes 32,588 cycles per round against
the default's 18,493, and reads 219.9 MB from DRAM. In the session its
rounds phase took 2.369 ms, against 2.269 ms for the default's separate
pack and rounds.

## Rounds kernel (8-conn, bare)

Cycles per round from ncu. "Ritual" is the barrier probe's cost per
round at the same grid; "rest" is the round minus the ritual
(*inferred*).

| scene | grid | cycles per round, before | after | change | ritual, before | after | ritual share, before | after | rest, before | after |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sq_8000_center | 72 x 128 | 21,613 | 18,493 | -14.4% | 4,517 | 2,387 | 20.9% | 12.9% | 17,096 | 16,106 |
| serpentine_256 | 72 x 128 | 9,015 | 6,578 | -27.0% | 4,517 | 2,387 | 50.1% | 36.3% | 4,498 | 4,192 |
| zz_serp_1k | 72 x 128 | 8,378 | 6,038 | -27.9% | 4,517 | 2,387 | 53.9% | 39.5% | 3,861 | 3,651 |
| serpentine_256 | 1 x 32 | 7,630 | 6,087 | -20.2% | 3,048 | 1,747 | 40.0% | 28.7% | 4,581 | 4,340 |

| metric | sq_8000 before | after | serpentine 72 x 128 before | after | serpentine 1 x 32 before | after |
|---|---:|---:|---:|---:|---:|---:|
| registers per thread | 129 | 135 | 129 | 135 | 129 | 135 |
| achieved occupancy | 25.0% | 25.0% | 25.0% | 25.0% | 2.1% | 2.1% |
| issue slots busy | 25.9% | 28.4% | 9.5% | 8.5% | 13.6% | 15.4% |
| eligible warps per scheduler | 0.35 | 0.38 | 0.11 | 0.11 | 0.14 | 0.15 |
| PC samples in barrier code | 41.2% | 34.0% | 99.6% | 99.4% | 42.5% | 32.1% |
| stall: barrier | 23.2% | 21.9% | 54.3% | 57.2% | 1.0% | 0.6% |
| stall: long scoreboard | 29.2% | 29.1% | 18.7% | 19.8% | 30.7% | 28.7% |
| stall: wait | 16.2% | 16.5% | 9.3% | 8.3% | 28.8% | 30.2% |
| L2 throughput, % of peak | 31.6% | 34.3% | 14.5% | 10.1% | 0.3% | 0.2% |
| L2 atomic unit busy | 12.0% | 14.1% | 0.4% | 0.3% | 0.0% | 0.0% |

"Barrier code" is the `grid.sync` internals (which ncu shows without
SASS), the `cudaCGSynchronize` wrapper, and the round's head and tail.

**Where the one warp's time goes** (serpentine, 1 x 32: shares of PC
samples are shares of time):

| region | before | after |
|---|---:|---:|
| grid-sync internals and wrapper | 37.0% | 23.0% |
| round head and tail (range check, counter reset, `q_state` read) | 5.5% | 9.1% |
| visit setup (range trip count) | 2.8% | 4.2% |
| visit loads (queue entry, tile and filter words) | 12.6% | 15.6% |
| closure (row fill, diagonal, 20 `SHFL`, vote) | 20.7% | 20.8% |
| exports (3 `SHFL`, up to 4 `ATOMG.OR`) | 7.1% | 9.7% |
| enqueue (ballots, stamp `ATOMG.MAX`, queue `ATOMG.ADD`) | 14.2% | 17.6% |

After the change the visit is 68% of the one warp's time. The before
profile found six dependent memory waits in it (queue entry, tile
words, side words, export atomics, stamp, queue add): they are the next
bill.

## Barrier probe

| ritual | grid | cycles per round (ncu) | stall: barrier | stall: long scoreboard | session probe, us (median [range of 7]) |
|---|---|---:|---:|---:|---|
| two `grid.sync` + read (before) | 72 x 128 | 4,517 | 37.7% | 17.5% | 2.19 [2.19-2.21] |
| one `grid.sync` + read (after) | 72 x 128 | 2,387 | 33.7% | 19.0% | 1.21 [1.15-1.26] |
| two `grid.sync` + read (before) | 1 x 32 | 3,048 | 2.4% | 28.6% | 2.41 [2.40-2.55] |
| one `grid.sync` + read (after) | 1 x 32 | 1,747 | 1.6% | 30.4% | 0.86 [0.84-0.87] |

At 2,070 MHz, 2,387 cycles are 1.15 us and 1,747 are 0.84 us
(*inferred*), close to the session's event-timed probe.

The two-sync probe at 1 x 32 does not agree: 3,048 cycles are 1.47
us, and session `tiles_20261015T114440Z` read 1.49 us, but this session
read 2.41. The event-timed one-block probe is not stable between
sessions; the cycle counts are.

The scenes confirm the 72 x 128 saving. The session's rounds phase
falls by 1.01 us per round on the serpentine and 1.00 us on
`zz_serp_1k` (*inferred*), against 0.98 us between the two probes.

## Pack and paint

`pack` is ch06's `pack_kernel` before and after. `paint` is
`paint_kernel` before (one warp per word, rows strided) and
`paint_tiles_kernel` after (one warp per tile, untouched tiles skipped
by their stamp). Times are the session's phase medians at 2,070 MHz.
Every other cell is "before / after".

| scene | kernel | session ms, before / after | grid | registers | achieved occupancy | issue slots busy | stall: long scoreboard | warp instructions per mask word | DRAM read, write MB (ncu) |
|---|---|---|---|---|---|---|---|---|---|
| sq_8000_center | pack | 1.154 / 1.155 | 2048 x 256 / same | 41 / 41 | 78.0% / 77.7% | 69.7% / 69.3% | 54.4% / 55.2% | 64.7 / 64.7 | 192.0, 7.7 / 206.8, 7.8 |
| sq_8000_center | paint | 1.035 / **0.808** | 2048 x 256 / 7813 x 256 | 35 / 63 | 95.0% / 64.8% | 70.0% / 74.7% | 51.5% / 5.8% | 52.2 / 49.4 | 8.0, 176.8 / 8.4, 178.1 |
| thin_h32_blob | pack | 0.430 / 0.430 | 8192 x 32 / same | 41 / 41 | 48.7% / 48.2% | 52.3% / 52.2% | 60.0% / 60.0% | 65.4 / 65.4 | 50.3, 1.8 / 50.4, 1.9 |
| thin_h32_blob | paint | 0.154 / **0.023** | 8192 x 32 / 2048 x 256 | 35 / 63 | 48.1% / 57.3% | 37.2% / 65.2% | 73.3% / 41.2% | 32.4 / 4.7 | 2.1, 0.0 / 0.1, 0.0 |

The before profile's event-timed ceilings (`dram_ceiling_events_before.csv`,
9 interleaved rounds at 2,070 / 8,001 MHz): a pure coalesced (מאוחד)
read of `sq_8000`'s 192 MB takes 0.772 ms (249 GB/s), so `pack` takes
1.50x a pure read of its bytes. That has not changed.

## What the numbers suggest next

1. **A narrow-front mode** (README open problem 2). On the serpentine
   and `zz_serp_1k` the one remaining `grid.sync` is still 36-40% of a
   round, and 87-89% of the warp samples. One block draining a local
   queue with block barriers would leave the ~4,200-cycle visit.
2. **Several rows in flight in `pack`.** It is still 1.5x a pure read
   (56% of its samples wait right after a load). On `sq_8000` it is
   now 37% of the pipeline (1.155 of 3.107 ms).
3. **Fewer instructions in the tile paint.** It is issue-bound now:
   74.7% of issue slots, 49.4 warp instructions per mask word.
4. **The visit's dependent chain.** At one warp the visit is now 68%
   of a round. Its six memory waits are where the cycles go.
5. **The fused pack needs its registers back** before it can be a
   default: at 247 registers it seats 192 warps instead of 288.

## Files

| file | what |
|---|---|
| `ncu_rounds_kernel.csv` | per scene and state: cycles per round, ritual share, occupancy, issue, stalls, `SHFL` and atomics per visit, sectors per request, hit rates, L2 and DRAM traffic; one row for the fused pack |
| `ncu_rounds_regions.csv` | share of PC samples per code region of the rounds kernel, before and after |
| `ncu_barrier_probe.csv` | the barrier probe in cycles (ncu) and in us (the session's 7 repeats), two syncs and one |
| `ncu_pack_paint.csv` | pack and paint, before and after: ncu counters and the session's phase times; the before rows also carry that profile's event-timed GB/s |
| `dram_ceiling_events_before.csv` | the before profile's event-timed pack, paint and pure read and write runs |

The `.ncu-rep` reports and raw exports are not committed: they hold
host and process details.

## Commands

The driver runs a `TileFill` config on a benchmark scene as described
under Setup (`pipeline`), or the barrier probe kernel (`barrier`).
Metrics come from `ncu --import R.ncu-rep --page raw --csv`, SASS-level
samples from `--page source --print-source sass --csv`.

```bash
# rounds kernel, the new default's bare twin: second real run, full set
ncu --target-processes all -k regex:rounds_kernel --launch-skip 3 --launch-count 1 \
    --clock-control base --set full --import-source no -o rounds_<scene>_new \
    uv run python <driver> pipeline --scene <scene> --config w4_bare_new --runs 3
# the same binary at one warp
ncu ... -o rounds_serpentine_256_1x32new \
    uv run python <driver> pipeline --scene serpentine_256 --config 1x32_bare_new --runs 3
# the one-sync barrier probe at 904 and 3,616 rounds (72 x 128 and 1 x 32)
ncu --target-processes all -k regex:_barrier_rounds_1 --launch-skip 1 --launch-count 2 \
    --clock-control base --set full --import-source no -o barrier1_72x128 \
    uv run python <driver> barrier --one --blocks 72 --tpb 128 --rounds 904 3616
# pack and the tile paint, flushed like the benchmark
ncu --target-processes all -k 'regex:pack_kernel|paint_tiles_kernel' --launch-skip 6 \
    --launch-count 2 --clock-control base --set full --import-source no \
    -o packpaint_<scene>_new uv run python <driver> pipeline --scene <scene> \
    --config w4_bare_new --runs 3 --flush
```

The launch skips count the warm-up: `TileFill` compiles each kernel on
two tiny images (heights 8 and 7) before the first real run.

## Glossary

- **Barrier** (מחסום) - a point every thread must reach before any
  continues; `grid.sync` is the grid-wide one.
- **Latency** (השהיה) - the time one operation takes from start to
  result.
- **Occupancy** (תפוסה) - resident warps per SM, against the 48 the SM
  can hold.
- **Long scoreboard** (לוח תלויות) - a warp waiting on a memory result
  (an L1/L2 load or an atomic).
- **Issue slot** (חריץ הנפקה) - one scheduler cycle in which a warp
  instruction can start.
- **Coalesced** (מאוחד) - a warp's 32 accesses served by a few
  consecutive memory sectors.
