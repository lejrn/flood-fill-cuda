# ch03 under ncu: where the warp-coop BFS spends its time

The ch03 README has owed an `ncu` profile since the stage was written.
This is that profile.

Two kernels, both bare twins, both at ch03's default grid (72 x 256):

- `multi_block_global8wc_bare_kernel` ("wc"), the stage's best kernel;
- `multi_block_global8_bare_kernel` ("conn8"), the thread-layout 8-conn
  kernel, for contrast.

Three scenes, built by ch03's own scene code and run through ch03's own
`flood_fill()`: `sq_2000_center` (1 Mpx, 501 levels), `sq_4000_corner`
(16 Mpx, 4,000 levels) and `sq_8000_center` (64 Mpx, 4,001 levels).
Line numbers refer to ch03's `kernels.py` at commit 1ea2888.

## The answer on sq_8000

wc takes **69.57 ms** in a timing session that follows the chapter
policy (9 interleaved rounds, range 69.51-69.74 ms, SM at 2,070 MHz,
memory at 8,001 MHz).

PC sampling (דגימת מונה התוכנית) records, every 1,024 cycles on every SM,
where one resident warp is and why it waits. All 576 warps stay
resident for the whole kernel. So a location's share of the samples is
its share of warp time.

| where the warps are | share | ms of 69.57 | main stall reason (סיבת עיכוב) |
|---|---:|---:|---|
| **grid.sync, total** | **21.2%** | **14.7** | barrier 58% |
| of which: the two-sync ritual (probe, 3.097 µs x 4,001 levels) | 17.8% | 12.4 | |
| of which: waiting for slower warps | 3.4% | 2.3 | |
| neighbor bounds + 3-byte RGB is-red loads (line 720) | 20.5% | 14.3 | long scoreboard 82% |
| enqueue: rear-counter atomicAdd, shfl, store (line 722) | 19.5% | 13.6 | long scoreboard 68% |
| visited atomic CAS (line 721) | 17.4% | 12.1 | long scoreboard 92% |
| loop control, x/y decode, dx/dy table | 9.9% | 6.9 | short scoreboard 29%, branch 22% |
| frontier queue entry load (line 710) | 8.9% | 6.2 | long scoreboard 43% |
| paint (3 byte stores) + depth store | 2.7% | 1.9 | issue (wait, selected) |

The shares come from a PC-sampling profile at 2.06 GHz. The two other
clocks profiled (0.84 and 1.57 GHz) moved no share by more than 4 points.
At lower clocks grid.sync's share is slightly larger.

### The questions ch03 carried

1. **The grid barrier is 21% of the time.** Most of it, 18 points, is
   the bare ritual: two `grid.sync()` and a read cost 3.1 µs per level
   at 72 x 256 (ch07's probe, same session). The other 3 points are
   warps that finished their share of a level early.
2. **The visited CAS is 17%, and it is latency (השהיה), not L2 atomic
   throughput.** The L2 atomic units are busy 7.7% of the time on
   average, 18% on the busiest slice (פרוסה). 95% of CAS requests hit L2.
   A warp still waits 838 cycles (407 ns) for its CAS. That is 2.5x the
   161 ns of one dependent atomic on an idle GPU (ch07's probe).
3. **Load latency (long scoreboard, לוח תלויות) is 52% of all samples.**
   It sits in four dependent round trips (הלוך-חזור) per 4-pixel chunk:
   the queue entry (100 ns), the red byte (307 ns), the CAS (407 ns) and
   the rear atomicAdd (360 ns). Plain loads (RGB + queue) are 29% of the
   time; the two atomics are 37%.
4. **LSU/L1 pressure from the 3-byte loads is about 2%.** lg_throttle is
   1.2% and mio_throttle 0.4% of samples, and most of the lg_throttle
   sits in the grid-sync call frame. The RGB loads cost latency instead.
   The G and B bytes hit L1, but they wait behind the R compare
   (short-circuit `and`): 4.6% of samples.
5. **DRAM bandwidth is not a limit.** 27.4 B per filled pixel reach DRAM,
   1.3x the ~22 B/px compulsory (הכרחי) footprint. At 69.57 ms that is
   25 GB/s, 13% of the ~190 GB/s measured copy peak.

## Every scene, both kernels

Shares from the 2.06 GHz profiles (wc sq_2000: 1.47 GHz), ms from the
timing session. "floor" is the barrier probe x levels; "slower warps" is
the rest of the grid.sync share.

| kernel | scene | ms | grid.sync | floor | slower warps | RGB | CAS | enqueue | queue | loop | paint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wc | sq_2000 | 2.85 | 64.8% | 54.4% | 10.4% | 10.1% | 8.1% | 4.9% | 3.6% | 7.5% | 1.0% |
| wc | sq_4000 corner | 29.01 | 50.1% | 42.7% | 7.4% | 13.9% | 10.9% | 8.6% | 5.6% | 9.1% | 1.7% |
| wc | sq_8000 | 69.57 | 21.2% | 17.8% | 3.4% | 20.5% | 17.4% | 19.5% | 8.9% | 9.9% | 2.7% |
| conn8 | sq_2000 | 5.05 | 92.1% | 30.7% | 61.4% | 2.8% | 1.2% | 0.7% | 0.4% | 2.5% | 0.4% |
| conn8 | sq_4000 corner | 42.05 | 84.4% | 29.5% | 55.0% | 6.9% | 2.2% | 1.0% | 0.7% | 3.7% | 1.0% |
| conn8 | sq_8000 | 96.97 | 45.9% | 12.8% | 33.1% | 28.6% | 7.9% | 4.6% | 1.4% | 6.6% | 5.0% |

conn8 is 1.77x / 1.45x / 1.39x slower than wc on the same grid (median
paired per-round ratios; every range is within 0.02). Its profile shows why. One thread
per queue entry leaves most of the 18,432 threads without work in a
level, so 33-61% of its time is warps waiting at the barrier. Each
active thread runs its 8 probes one after another, with 18 sectors per
load request (wc: 5.4). Its byte stores take 21 sectors per request and
trip lg_throttle (6.4% of samples).

## Stall reasons (share of all PC samples, 2.06 GHz profiles)

| kernel | scene | barrier | long scoreboard | wait | short scoreboard | branch | lg throttle | selected | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wc | sq_2000 | 36.0 | 28.4 | 10.3 | 3.9 | 5.0 | 3.5 | 3.8 | 9.1 |
| wc | sq_4000 corner | 27.8 | 35.0 | 11.4 | 4.8 | 5.0 | 2.6 | 4.5 | 8.9 |
| wc | sq_8000 | 12.2 | 52.4 | 12.0 | 5.5 | 4.5 | 1.2 | 5.3 | 6.9 |
| conn8 | sq_2000 | 66.9 | 14.8 | 4.9 | 1.1 | 2.5 | 3.0 | 1.6 | 5.2 |
| conn8 | sq_4000 corner | 61.1 | 19.2 | 5.4 | 1.2 | 2.6 | 3.3 | 1.9 | 5.3 |
| conn8 | sq_8000 | 34.4 | 44.8 | 4.8 | 2.0 | 2.2 | 6.4 | 2.0 | 3.4 |

The "barrier" reason is the CTA barrier inside the driver's grid-sync
code. ncu cannot list that code, so its samples are counted as total
samples minus the samples on listed instructions. Every barrier sample
falls there; none falls on a ch03 line.

## Memory and occupancy (base-clock full profiles)

| metric | wc 2000 | wc 4000c | wc 8000 | conn8 2000 | conn8 4000c | conn8 8000 |
|---|---:|---:|---:|---:|---:|---:|
| DRAM bytes per filled px | 27.1 | 29.4 | 29.3 | 33.7 | 34.4 | 31.4 |
| DRAM GB/s at the timed ms (% of 190 copy peak) [1] | 10.7 (5.6%) | 15.6 (8.2%) | 25.2 (13.3%) | 7.9 (4.2%) | 10.7 (5.7%) | 18.4 (9.7%) |
| ncu DRAM % of peak, at the profile's clocks | 2.2 | 3.7 | 6.5 | 1.5 | 2.8 | 4.5 |
| L2 hit % (all / reads / atomics) | 98.7 / 88.0 / 95.0 | 98.0 / 91.9 / 95.0 | 96.9 / 92.2 / 95.1 | 99.9 / 92.1 / 95.7 | 98.3 / 95.1 / 95.9 | 97.8 / 96.2 / 96.0 |
| L1 hit % (all / global loads) | 83.8 / 61.2 | 74.2 / 62.2 | 54.3 / 64.2 | 82.2 / 69.9 | 72.8 / 67.2 | 55.4 / 60.9 |
| sectors per request: load / store / atomic | 3.8 / 2.8 / 4.5 | 4.5 / 2.8 / 4.8 | 5.4 / 2.8 / 5.1 | 5.5 / 20.0 / 10.9 | 9.9 / 23.9 / 16.0 | 17.8 / 21.4 / 14.7 |
| L2 atomic unit busy %: average / busiest slice | 3.1 / 8.1 | 4.5 / 10.3 | 7.9 / 18.7 | 2.0 / 6.4 | 3.8 / 8.3 | 6.4 / 8.6 |
| L2 CAS requests per px / atomic adds per px (rear + grid barrier) | 2.12 / 0.31 | 2.13 / 0.27 | 2.21 / 0.25 | 2.58 / 0.14 | 2.88 / 0.08 | 3.02 / 0.08 |
| atomic adds landing on the busiest L2 slice | 75% | 86% | 96% | 49% | 53% | 87% |
| L2 throughput % of peak | 33.6 | 35.5 | 41.0 | 20.7 | 29.7 | 39.0 |
| LSU instruction issue % / L1 data wavefronts % | 21.1 / 16.6 | 21.0 / 20.1 | 20.9 / 27.2 | 11.1 / 10.0 | 11.1 / 15.3 | 7.5 / 22.8 |
| registers per thread | 65 | 65 | 65 | 74 | 74 | 74 |
| occupancy: achieved / theoretical % | 50.0 / 50 | 50.0 / 50 | 50.0 / 50 | 50.0 / 50 | 50.0 / 50 | 49.9 / 50 |
| issue active % / IPC per SM | 24.8 / 0.98 | 28.6 / 1.13 | 36.3 / 1.42 | 10.2 / 0.41 | 11.9 / 0.47 | 11.9 / 0.47 |
| threads per instruction (of 32) / uniform branches % | 25.2 / 86.9 | 23.3 / 82.6 | 20.4 / 76.7 | 27.4 / 97.9 | 28.5 / 98.2 | 27.7 / 93.7 |

[1] Uses the bytes per px of the 1.5-2.1 GHz profiles. Bytes per px
vary 3-7% between profiles on the big scenes, up to 19% on sq_2000.

Occupancy (תפוסה) is capped at 50% by registers in both kernels: 3
blocks of 256 threads per SM, 24 of 48 warps. wc's 65 registers are
allocated as 72; at 64 a fourth block would fit.

## The chain behind each 4-pixel chunk (wc, sq_8000, 2.06 GHz)

Average wait per warp-level execution of the instruction that consumes
each result (samples x 1,024 cycles x 24 warps / executions):

| line | instruction waits for | cycles | ns | share of samples |
|---|---|---:|---:|---:|
| 721 | the visited CAS result | 838 | 407 | 16.1% |
| 722 | the rear-counter atomicAdd result | 742 | 360 | 13.4% |
| 720 | the R byte of the neighbor | 632 | 307 | 12.1% |
| 710 | the queue entry | 206 | 100 | 3.9% |
| 720 | the G byte (L1 hit, after the R compare) | 150 | 73 | 2.9% |
| 719 | `dy[d]` from constant memory, lane-varying index | 138 | 67 | 2.7% |
| 720 | the B byte (L1 hit, after the G compare) | 91 | 44 | 1.7% |

These seven waits add up to 1.36 µs per chunk. Inferred from the timings
(not measured by ncu): sq_4000 corner and sq_8000 have the same level
count, so their difference isolates the per-pixel cost. That gives
`T = 3.87 µs x levels + 0.845 ns x pixels` for wc; it predicts sq_2000
within 2.4%. 0.845 ns per pixel across 576 warps is one 4-pixel chunk
every 1.95 µs per warp: the chain above plus issue and branches. The
3.87 µs per level is the 3.10 µs barrier plus 0.77 µs of work that even
the narrowest level pays.

## What this settles in the ch03 README

- **Prediction 0** (the queue's hot window fits L2): confirmed. L2 hits
  88-96% of reads; DRAM sees 1.3x the compulsory bytes.
- **Prediction 2** (the single rear counter becomes an atomic hotspot):
  hot but not saturated. On sq_8000, 96% of the atomic adds land on the
  rear counter's L2 slice, whose atomic unit is 18% busy. The cost is the 360 ns every chunk
  waits for it, 19.5% of warp time on sq_8000.
- **The plateau and "sector inflation ~8x could mean ~90% of peak"**:
  refuted. Real DRAM traffic (27-29 B/px) is far below the logical
  model (61-105 B/px), because L2 absorbs the repeated probes. At the
  benchmark's speed DRAM runs at 13% of the copy peak. The kernel is
  latency-bound: four dependent round trips per chunk, 24 warps per SM,
  and no unit near its limit.

## What the numbers suggest (inferred, not measured)

Ranked by expected gain on sq_8000:

1. **Stop waiting on the rear counter per chunk.** Each warp could
   buffer its discoveries for the whole level in shared memory and
   reserve global space once per level (or once per block). Today that
   is 0.25 dependent atomics per pixel at ~360 ns each: 19.5% of warp
   time (13.6 ms). It would also coalesce the queue stores (2.8 sectors
   per request today).
2. **Raise occupancy from 50% to 67%.** wc uses 65 registers, one above
   the 64 that fit 4 blocks per SM (a 96 x 256 grid). Nothing is
   saturated, so the per-pixel term (~54 ms) could shrink toward 24/32
   of itself. A bigger grid makes the barrier dearer, so probe 96 x 256
   too. `max_registers=64` is a one-line test.
3. **One grid.sync per level instead of two.** Three rotating per-level
   counters, each reset two levels ahead, make the second barrier
   unnecessary. If one sync costs half of the 3.1 µs pair, this saves
   1.55 µs per level: 6.2 ms on sq_8000 (9%), 6.2 ms on sq_4000 corner
   (21%) and 0.78 ms on sq_2000 (27%).
4. **Test red without short-circuit.** Compare the three bytes with `&`
   so the G and B loads go out with R (4.6% of samples, ~3 ms). A packed
   red mask (the ch06/ch07 pack) goes further: an 8 MB bit array stays
   in L2, and the BFS stops reading RGB.
5. **Compute dx/dy from d.** The constant table with a lane-varying
   index costs 2.7% (short scoreboard on line 719).
6. **Not worth chasing:** DRAM bandwidth, L2 atomic throughput, LSU/L1
   throughput, the coalescing of the RGB loads.

## Method and caveats

- **Clocks.** `--clock-control base` was used for the full profiles, as
  planned, but it did not hold the base clock here: those runs read
  0.83-1.18 GHz SM and 5.8-8.0 GHz memory. Queried afterwards, the
  driver showed "SW Power Cap" and "SW Thermal Slowdown" active. Two
  more profile sets ran without a lock: 1.47-1.60 GHz, and 2.06 GHz with
  a 3 s GPU spin right before the profiled launch. Shares agree within
  4 points across all three. ncu durations are profiler numbers, not
  timings; every ms on this page comes from the timing session, which
  ran at 2,070 / 8,001 MHz.
- **What a share means.** One sample per SM per 1,024 cycles, of one of
  its 24 resident warps. A share is warp time, not wall time. "Waiting
  for slower warps" is the grid.sync share minus the probe floor.
- **Line info.** `lineinfo` was set on ch03's dispatcher in memory. The
  SASS is identical to the plain build apart from the mangled name.
- **Hidden code.** `grid.sync()` calls the driver through
  `cudaCGSynchronize`. ncu lists the wrapper (about 2% of samples) but
  not the driver's barrier code. Those samples (19% of all on wc
  sq_8000 at 2.06 GHz, 44% on conn8) go to grid.sync as described
  above.
- **Timing session** (`timing_session.csv`): ch07's helpers, 8 s ramp,
  barrier probe at 72 x 256, untimed correctness pass (wc and conn8
  paint identical images), 1 s re-ramp, 9 interleaved rounds. Every run
  restores the image, resets on the device, reads a 96 MB buffer, queues
  a sleep kernel, and times one launch with CUDA events. wc matches
  ch07's committed session (`tiles_20261015T114440Z`) within 1.3%.

## Files

| file | contents |
|---|---|
| `ncu_metrics.csv` | headline metrics per kernel, scene and profile run (base, boost) |
| `ncu_stall_reasons.csv` | stall reasons as % of PC samples, all 18 profiles |
| `ncu_stall_location.csv` | the location buckets (add to 100%) and their stall mix |
| `ncu_top_source_lines.csv` | top 8 stalled source lines per kernel and scene |
| `ncu_top_sass.csv` | top 10 stalled SASS instructions, with wait cycles per execution |
| `time_split.csv` | the buckets in ms on the timed kernel, barrier floor, DRAM at timed speed |
| `timing_session.csv` | the policy timing session (both kernels, 3 scenes) |
| `ncu_commands.md` | the ncu and export commands |
| `scripts/` | the driver, the ncu wrapper, the analysis and the timing script |

Raw `.ncu-rep` files and ncu exports stay out of the repository (they
carry host and process details).

## Glossary

- **PC sampling** (דגימת מונה התוכנית): ncu records, at a fixed cycle
  interval, which instruction a warp is at and why it is not issuing.
- **Long scoreboard**: waiting for a global or local memory result (a
  load or an atomic). **Short scoreboard**: waiting for a shorter
  variable-latency unit, such as a constant-cache or shared-memory read.
- **Barrier** (מחסום): waiting for the other warps of the block at a
  CTA barrier; here, always inside `grid.sync()`.
- **lg_throttle / mio_throttle** (ויסות): the memory instruction queue is
  full. This is the sign of LSU pressure, and it is small here.
- **Sector** (מקטע): 32 bytes, the unit of L1/L2/DRAM traffic. Sectors
  per request show how scattered a warp's access is (1-4 is coalesced).
- **Occupancy** (תפוסה): resident warps per SM as a share of the 48 the
  hardware allows.
- **Compulsory traffic** (תעבורה הכרחית): bytes that must reach DRAM
  even with a perfect cache: each array read or written once.
