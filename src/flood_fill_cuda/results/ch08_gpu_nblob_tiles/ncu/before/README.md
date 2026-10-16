# ch08 under ncu - why dense tiles are slow

**Question.** ch08 loses to ch06 where a tile holds many comps:
`dots_4k` 16.08 vs 6.24 ms, `random_4000` 6.89 vs 5.89,
`zz_lattice_4k` 6.89 vs 1.73 (session `waves_20261015T183401Z`).
This page profiles ch08 as it is, before any change.

**Short answer.** K2 and K4 are not slow because their comp loop is
serial. They are slow because each step of it is a whole-warp closure.

Every comp, even a one-pixel one, costs about 145 warp instructions per
closure iteration. On top come 67-97 more per comp in K2 and 147-234 in
K4. The kernels run at 81-87% of the ALU pipe (צינור) peak, so time
follows the instruction count closely.

K3 is a different story. On `zz_lattice_4k` it is latency (השהיה)
bound: dependent global loads and atomics (פעולות אטומיות) in the dense
lookup path, then a serial flatten.

Labels used below: **measured** = an ncu counter or a counter of the
instrumented twin; **derived** = arithmetic on measured numbers;
**inferred** = a reading or an estimate that no run here tested.

## Method

- **Tool.** Nsight Compute 2025.2.1 (the profiler, פרופיילר) on the RTX
  4060 Laptop (AD107, 24 SMs, cc 8.9), driver 610.62, Numba 0.65.1,
  WSL2. Code at HEAD `faac8af` (v0.37.0), nothing changed.
- **Window.** `ncu_driver.py` builds the scene with ch08's own benchmark
  builders. It warms every kernel and runs the pipeline once, untimed,
  on the real scene. Then it restores the image and profiles ONE rgb
  pipeline inside a `cuProfilerStart/Stop` window.
- **Engine.** `TileWaves` defaults: hybrid lookup, `dense_min` 4, L8,
  valve 16, bare twins (80 / 80 / 56 registers, K3 at 72 x 256).
- **Check.** After the window, every run compared the painted image
  with ch06's own `RunRecolor.run`. All ncu runs matched.
- **ch06.** ch06 was profiled the same way, at the session's best merge
  grid per scene (`dots_4k` 2048 blocks, `random_4000` 8192).
- **Replay.** Kernel replay (הרצה חוזרת), the default, with caches
  flushed before every pass (`--cache-control all`, the default). That
  matches the benchmark's L2 flush. The cooperative K3 (`grid.sync`)
  profiled fine under kernel replay; application replay was not needed.
- **Sets.** `--set full` plus extra metrics: thread instructions,
  shuffle/vote count, ld/st/atom/red requests and sectors, L2 atomics,
  divergent branch targets.
- **Lines.** A second pass rebuilt K2, K3 and K4 with
  `lineinfo=True` for source-line stall samples. The twins keep the
  same registers (80 / 80 / 56) and the same cooperative capacity (72).

**Clocks.** `--clock-control base` was used, as the task asked. On this
WSL2 laptop it did NOT hold one clock: the per-kernel average SM clock
ranged from 0.40 to 1.14 GHz (measured, `kernels.csv`).

So durations under ncu are not comparable, and this page compares
**elapsed SM cycles** and **instruction counts**. Instruction counts are
clock-free and repeat exactly for K2 and K4.

**Validation (derived).** Elapsed cycles divided by the benchmark's
2,070 MHz reproduce the benchmarked phase times within 0.93-1.05. That
holds for K2 and K4 on the dense scenes, K3 on `zz_lattice_4k`,
`random_4000` and `input_blobs`, and ch06's emit, merge (`random_4000`)
and paint (`cycles_vs_benchmark.csv`).

| kernel, scene | ncu cycles / 2.07 GHz | benchmark phase |
|---|---:|---:|
| K2 `dots_4k` | 6.59 ms | 6.74 ms |
| K4 `dots_4k` | 8.69 | 8.86 |
| K2 `random_4000` | 2.52 | 2.70 |
| K4 `random_4000` | 3.40 | 3.31 |
| K3 `zz_lattice_4k` | 4.29 | 4.32 |
| ch06 emit `dots_4k` | 1.06 | 1.06 |

Kernels bound by memory or by their small size land between 0.63 and
1.41: both packs, the uniform paint, the flattens, count and scan, K3
on `dots_4k` (flatten only) and K2 on `input_blobs`. That is expected,
because the memory clock does not scale with the SM clock.

A cross-check at `--clock-control none` gave idle clocks (memory at
0.4-0.8 GHz between replay passes), so its durations were not used. Its
pipe percentages agree: K2's ALU pipe at 86.4 / 84.4 / 86.7% there vs
86.4 / 86.0 / 86.8% at base (dots, zz_lattice, random).

## The four scenes in comps (CPU census, measured)

| scene | comps | 1-px comps | comps of 4 px or less | px per comp | comps per touched tile (mean / max) |
|---|---:|---:|---:|---:|---:|
| `input_blobs` | 36,863 | 3,092 (8.4%) | 6,181 | 364.9 | 1.3 / 8 |
| `random_4000` | 907,448 | 344,411 (38.0%) | 642,243 (70.8%) | 5.3 | 58.1 / 86 |
| `dots_4k` | 4,194,304 | all | all | 1.0 | 256 / 256 |
| `zz_lattice_4k` | 975,868 | 943,610 (96.7%) | all | 1.03 | 59.6 / 60 |

So a "dense tile" in these scenes is a tile full of tiny comps. On
`zz_lattice_4k` the comps are one-pixel ring comps on the tile borders.

## ch08's kernels, elapsed Mcycles (measured)

| scene | K1 pack | K2 label | K3 waves | K4 paint |
|---|---:|---:|---:|---:|
| `input_blobs` | 2.234 | 0.606 | 0.255 | 0.996 |
| `random_4000` | 0.453 | **5.215** | 1.271 | **7.045** |
| `dots_4k` | 0.468 | **13.646** | 0.488 | **17.994** |
| `zz_lattice_4k` | 0.462 | 3.592 | **8.877** | 0.506 |

The pack is the same on every 4k scene (0.45-0.47 Mcycles). The dense
losses are K2 and K4 on `dots_4k` and `random_4000`, and K3 on
`zz_lattice_4k`.

## K2 and K4: issue-bound, not divergent, not memory (measured)

| kernel, scene | IPC (of 4) | issue busy | ALU pipe | threads per warp inst | branch eff. | DRAM of peak | top stalls, cycles per issue |
|---|---:|---:|---:|---:|---:|---:|---|
| K2 `dots_4k` | 2.78 | 70.1% | 86.4% | 31.2 | 95.5% | 1.3% | wait 2.24, not selected 1.72, math throttle 1.53 (of 8.35) |
| K2 `random_4000` | 2.71 | 68.4% | 86.8% | 31.5 | 97.0% | 2.3% | wait 2.15, not selected 1.71, math throttle 1.57 (of 8.25) |
| K2 `zz_lattice_4k` | 2.76 | 69.6% | 86.0% | 31.2 | 95.8% | 5.1% | wait 2.26, not selected 1.72, math throttle 1.53 (of 8.32) |
| K4 `dots_4k` | 2.89 | 73.6% | 81.2% | 28.3 | 93.2% | 2.8% | not selected 3.23, math throttle 2.64, wait 2.47 (of 11.33) |
| K4 `random_4000` | 2.89 | 72.8% | 81.6% | 26.8 | 92.4% | 8.7% | not selected 3.10, math throttle 2.56, wait 2.35 (of 11.01) |
| ch06 paint `dots_4k` | 3.07 | 76.7% | 81.6% | 17.3 | 87.6% | 8.6% | not selected 3.37, math throttle 2.77, wait 1.46 (of 10.08) |

**Reading.** The ALU pipe is the limit. On Ada it runs LOP3, SEL,
ISETP, SHF and IADD3 at half rate, so it saturates while the issue
slots are only 70% busy.

The stall (השתהות) reasons confirm it. "Wait", "not selected" and "math
pipe throttle" mean ready warps queue for the pipe. Long-scoreboard
(memory) stalls are under 0.04 cycles per issue.

Warps are full (31 of 32 threads active per instruction), and branch
efficiency is 93-97%. The one divergent (מתפצל) spot is `_is_ring`: its
frame test runs on one lane, 12 warp instructions per comp.

**So more occupancy (תפוסה) or more overlap of comps would not help.**
K2 runs at 50% theoretical occupancy (80 registers). With the ALU pipe
at 86% (K4 at 81%), the most any latency hiding could add is about
1.16x (1.23x for K4, derived). The instruction count per comp is the
lever.

## Instructions per comp (measured, deterministic)

| | `input_blobs` | `random_4000` | `dots_4k` | `zz_lattice_4k` |
|---|---:|---:|---:|---:|
| K2 warp inst per comp | 826.9 | 365.6 | **213.5** | 242.8 |
| K2 closure iterations per comp | 2.07 | 1.97 | 1.00 | 1.00 |
| K2 closure warp inst per iteration | | 144.9 | 145.9 | 145.8 |
| K4 warp inst per comp | 888.3 | 517.4 | **291.8** | 24.4 (fast path) |
| K4 closure warp inst per iteration | | 143.9 | 144.9 | |

`input_blobs` has 1.3 comps per tile, so its per-comp numbers are
mostly per-tile setup; it is the good case.

**The closure costs about 145 warp instructions per iteration, in every
scene and in both kernels.** One iteration of a one-pixel comp is:

- 22 SHFL: 18 in the column fill (the last step's mask shuffle is
  dead code), 2 in the diagonal step, 2 in the frontier exit
- 20 SEL: 18 per-step lane guards of the column fill (`if lane >= k`
  / `if lane + k < 32`) and 2 in the diagonal step
- about 13 ISETP, mostly the same lane tests
- about 52-61 LOP3, plus IMAD/SHF for shifts
- 13 BRA, 2 VOTE (the two exits)

**Hoisted work.** A further 16 instructions per comp sit on the
`anchor` line (663 in K2, 1281 in K4). The anchor itself is 3 of them.
The other 16 are the row fill's mask prefix (`p &= p << k`) for the new
`rem`, hoisted (הועלה) out of the closure loop by the compiler.

**K2's other per-comp work** is 51-81 warp instructions (derived:
non-closure work minus the 16 hoisted ones; 51 on `dots_4k`, mostly
interior comps; 81 on `zz_lattice_4k`, all ring comps):

- `_first_pixel` 17
- `_is_ring` 12 (divergent)
- loop test plus back-edge 12 (the `c_int` counter is int64)
- ring work (side words, ordinal, `_flush`) about 31-45 per ring comp

**K4's other per-comp work** on `dots_4k` is 147 (`regions.csv`):

- `_paint_rgb` 54: 3 single-thread byte stores per pixel plus the row loop
- `lab % N_PALETTE` 24: a 64-bit signed modulo (שארית), 24 SASS
  instructions (measured in the SASS)
- `_is_ring` plus the label shuffle 17
- `_first_pixel` 18, the hoisted masks and anchor 19, the loop 6, the
  tile setup 8.5

**Per tile** (derived): `dots_4k` pays 256 x 213.5 = 54,656 warp
instructions in K2 and 74,700 in K4 for 256 pixels.

## ch08 against ch06 on the same scenes (measured)

| scene | labelling, ch08 (K0+K2+K3) | labelling, ch06 (count..flatten) | ratio | warp inst, ch08 / ch06 | paint, ch08 / ch06 |
|---|---:|---:|---:|---:|---:|
| `random_4000` | 6.489 Mcyc | 4.106 Mcyc | 1.58 | 384.3M / 166.1M | 7.045 / 7.128 = 0.99 |
| `dots_4k` | 14.138 | 2.864 | **4.94** | 896.9M / 34.0M | 17.994 / 8.712 = **2.07** |

The benchmark's labelling ratios were 1.50 and 4.64, so ncu tells the
same story.

**Per unit (derived).** ch08 spends 213.8 warp instructions per comp on
`dots_4k`, and ch06's labelling 8.1 per run. On `random_4000` the
numbers are 423.5 per comp and 49.4 per run.

ch06's emit works one lane per word, all lanes useful. ch08's K2 points
the whole warp at one comp, and for a one-pixel comp one lane of 32
holds the pixel.

ch06's emit is not issue-bound. On `dots_4k` it runs at IPC 0.30, with
long-scoreboard (21.6) and LG-throttle (15.6, the global memory queue)
stalls on uncoalesced (לא מאוחדות) stores: 32 sectors per request. That is why the cycle
ratio is 4.9x, not the 26x of the instruction ratio.

**Paint.** Both paints issue 3 single-thread byte stores per red pixel
on `dots_4k`: 12,582,912 store requests each, 1.0 sector per request.
Both are ALU-bound at about 81%.

The 2.07x gap is ch08's re-closure and per-comp overhead in K4 (145 +
147 warp instructions per comp vs ch06's 148 per run). On
`random_4000` the paints tie.

## K3 on dense tiles: latency, the dense path, the flatten

| | `zz_lattice_4k` | `random_4000` | `input_blobs` |
|---|---:|---:|---:|
| rounds | 2 | 4 | 6 |
| tasks (all dense on the first two) | 1,072,642 | 260,258 | 136,764 |
| warp k-steps | 2,145,154 | 260,258 | 23,005 |
| signature tests | 64,009,006 | 5,488,683 | 170,995 |
| CAS (share of tests) | 2,048,002 (3.2%) | 308,152 (5.6%) | 144,967 |
| claims (share of CAS) | 441,056 (21.5%) | 108,668 (35.3%) | 25,179 |
| collisions = union calls | 568,866 | 44,163 | 31,227 |
| rmin atomicMin / roots | 975,867 / **1** | 151,871 / 175,129 | 34,341 / 2,447 |
| IPC (of 4), ALU pipe | 1.55, 38.0% | 1.76, 42.6% | 1.14, 27.8% |
| cycles per issue: long scoreboard / total | **6.55 / 14.61** | 5.17 / 13.24 | 8.10 / 20.26 (+ barrier 5.82) |
| dense path: share of warp inst / of stall samples | 81.2% / 69.0% | 72.0% / 66.3% | |
| flatten 1 + 2: share of stall samples | 19.0% | 16.2% | |
| warp inst per task | 297.7 | 201.9 | 49.9 |
| L1 atomic requests | 2,161,507 | 409,171 | 43,051 |
| global load requests (sectors per request) | 5,898,029 (12.7) | 950,956 (8.5) | 160,336 (4.6) |
| DRAM read + write per touched tile | 16.5 KB | 2.3 KB | 0.3 KB |

Counts come from the instrumented twin; the collision count varies
from run to run (the README's race). Shares come from the line-info
twin's samples.

**Reading (measured).** K3 is latency-bound on dense scenes: IPC 1.55,
and long-scoreboard waits are 45% of its stall cycles. The top lines
are the dense path's `_claim` (39% of samples on `zz_lattice_4k`) and
its record load (`hit = s2 & wl`, 11.7%, 98% long scoreboard).

Each dense step is a chain of dependent global accesses. The warp
loads 32 records (one sector per lane), compares, does a CAS, then a
warp-aggregated atomic add, a shuffle, and the queue stores.

The step count is the problem: 2.15M serial dense steps on
`zz_lattice_4k` for 441,056 claims. Only 3.2% of the signature tests
hit, and 78% of the CAS lose (the comp was already claimed).

**The flatten (measured).** It is one thread per tile, looping over
the tile's ring comps (`for kk in range(nr)`). Its loads show as 95-100%
long-scoreboard stalls: `_find` is 11.4% of K3's samples on
`zz_lattice_4k`.

On `dots_4k`, with 0 rounds, K3 is only the flatten (0.488 Mcycles),
and its CPI is 159: 126 of it is long scoreboard. On `zz_lattice_4k`
all 975,867 non-root ring comps send their atomicMin to ONE root's
`rmin` word.

**The instrumented split** (relative only: that twin's counters slow
every thread) agrees: rounds 85.4%, union 1.4%, flatten 13.1% of K3 on
`zz_lattice_4k` (`k3_phase_split_instrumented.csv`).

## Memory per tile (measured)

DRAM is not a limiter in K2 or K4 on any dense scene (1.3-8.7% of
peak). K2 moves about 2.4-2.5 KB of DRAM per touched tile on `dots_4k`
and `zz_lattice_4k` (16.2 MB read and 23.4 MB written on `dots_4k`).

**Record stores.** These are written in a stride-8 layout: lane k
writes word j of record `64t + k`. That gives 25.3 sectors per store
request on `dots_4k` and 20.1 on `zz_lattice_4k`, but L2 merges them,
and the stores are 0.02% of K2's instructions.

K3 on `zz_lattice_4k` moves 16.5 KB per touched tile (138 MB read, 132
MB written): records, queues and parents, with an 89.5% L2 hit rate.

## Recommendations, ranked by what the numbers say

1. **A bulk path for one-pixel comps, in K2 and K4.** In a tile,
   `iso = m & ~dilate8(m)` finds every isolated pixel at once (each
   is its own comp).
   - Interior ones need no closure and no record; their label is
     their own `lin`.
   - Ring ones need a record but no closure.
   - **Upper bound (חסם עליון), derived:** one-pixel comps are 100%
     of the comps of `dots_4k`, 96.7% of `zz_lattice_4k` and 38% of
     `random_4000`. At about 213 warp instructions each, they are
     about 100%, 97% and 22% of K2's instructions there, and K2 is
     issue-bound, so time follows. K4 is similar on `dots_4k` (100%)
     and `random_4000` (about 21%).
   - It does not fix `zz_lattice_4k`, whose loss is K3 (item 3).
2. **Stop re-closing interior comps in K4.** An interior comp's label
   is its own anchor, known in K2.
   - K2 could paint them at closure time in the rgb and mask contracts
     (the label-only tier would skip it).
   - Or K4 could close only ring comps, each seeded from its record's
     side words, and skip discovery order.
   - Interior comps are 88% of the comps of `dots_4k` and 64% of
     `random_4000`. K4's closure is 50-55% of its instructions, plus
     about 150 instructions per comp of overhead (inferred gain:
     most of K4 on those scenes).
3. **K3's dense path on one-pixel ring comps.** 2.15M serial dense
   steps, each a chain of dependent L2 round trips, for 3.2% useful
   tests. Candidates (inferred, untested):
   - a cheap prefilter of `want` against the neighbour's frame word
     before the warp-wide step (the index twin's K2 already writes
     frame words)
   - a plain load of the label before the CAS (78% of the CAS lose)
   - one warp-aggregated append instead of two `_warp_slot` calls per
     step
   - Or route such images to ch06 (item 6).
4. **K3's flatten in parallel per ring comp.** One lane per record id
   (`64t + lane`) instead of one thread per tile, so the `_find` and
   label loads of a tile overlap. It is 16-19% of K3's samples on
   dense scenes and almost all of K3 on `dots_4k`. The one-root
   atomicMin hot spot of `zz_lattice_4k` could test before it writes.
5. **Cheap instruction cuts in the shared closure** (inferred, each
   needs a test; the closure is ch07's code, so ch07 must be re-run
   too):
   - Drop the lane guards in `_col_step_up` / `_col_step_down`. At
     the warp edge `shfl_up` / `shfl_down` return the lane's own word,
     and `f | (p & f) = f`, `p & p = p`, so the result is unchanged.
     That removes 18 SEL and up to about 10 ISETP of the 145 per
     iteration (up to about 19% of the closure, on the ALU pipe that
     limits).
   - `lab % N_PALETTE` in uint32: 24 instructions to about 5 (8% of
     K4 on `dots_4k`).
   - `_is_ring` without the one-lane branch, and the counters in
     int32.
6. **Route by density.** K2's header already counts comps per tile.
   On `zz_lattice_4k` and `dots_4k`, ch06 labels in 0.44 and 1.53 ms
   against 6.25 and 7.09 (benchmark). A per-image test on comps per
   touched tile could send such images to ch06 (inferred).

More occupancy, or launching comps in parallel within the same
closure code, is not on the list. K2 and K4 are already at 81-87% of
the ALU pipe (measured), and the ceiling for latency hiding is about
1.16x (derived).

## Files

| file | what |
|---|---|
| `kernels.csv` | every kernel x scene: cycles, ms at the ncu clock, the measured SM clock, instructions, threads per instruction, IPC, issue, pipes, occupancy, branch efficiency, divergent targets, shuffle/vote count, DRAM, L1/L2 hit rates, ld/st/atomic requests and sectors per request, L2 atomics, the top 3 stalls |
| `stalls.csv` | all warp-stall reasons per kernel x scene, in warp cycles per issued instruction |
| `per_unit_ch08.csv` | ch08 counters per scene and instructions per comp, iteration, task and ring comp; DRAM per touched tile |
| `ch08_vs_ch06.csv` | labelling and paint, cycles and instructions, per comp vs per run |
| `opcode_mix.csv` | SASS opcode families of K2, K3, K4 per scene: share of instructions and samples, per comp or task |
| `regions.csv` | K2, K3, K4 split by source region (line-info twins), and the closure cost per iteration |
| `source_lines.csv` | the top source lines per kernel x scene: stall samples, instructions, threads, top stalls |
| `census.csv` | comps, one-pixel comps, comp sizes and comps per tile per scene (CPU, `cpu_model._local_comps`) |
| `cycles_vs_benchmark.csv` | ncu cycles at 2.07 GHz against the benchmark's phase times |
| `k3_phase_split_instrumented.csv` | K3 rounds / union / flatten shares from the instrumented twin's `%globaltimer` stamps |
| `ncu_driver.py` | the driver every ncu run used |

Line numbers refer to `ch08_gpu_nblob_tiles/kernels.py` at `faac8af`.
Numba inlines the device helpers, so ch07's closure helpers are
counted on the ch08 line that calls them.

No `.ncu-rep` file or raw ncu export is kept here: they hold host and
process details. Every table above was written from them by a script.

## Reproduce

One GPU process at a time. From the repository root, with
`FLOOD_FILL_IMAGES` pointing at the input PNGs:

```bash
M=smsp__inst_executed.sum,smsp__thread_inst_executed.sum,smsp__thread_inst_executed_pred_on.sum,sm__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum,sm__sass_inst_executed_op_global_atom.sum,sm__sass_inst_executed_op_global_red.sum,sm__sass_inst_executed_op_global_ld.sum,sm__sass_inst_executed_op_global_st.sum,smsp__inst_executed_op_generic_atom_dot_cas.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_requests_op_atom.sum,lts__t_sectors_op_atom.sum,lts__t_requests_op_red.sum,lts__t_sectors_op_red.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets.sum,smsp__sass_average_branch_targets_threads_uniform.pct

# full set, one pipeline (side ch08 or ch06; scenes input_blobs,
# random_4000, dots_4k, zz_lattice_4k)
ncu --target-processes all --profile-from-start off --clock-control base \
    --set full --metrics $M -f -o ch08_dots_4k \
    uv run python ncu_driver.py --scene dots_4k --side ch08

# line-info twins of K2, K3, K4: source-line stall samples
ncu --target-processes all --profile-from-start off --clock-control base \
    -k regex:'label_kernel|waves_kernel|paint_kernel' \
    --section SourceCounters --section WarpStateStats \
    --section InstructionStats --section LaunchStats --section SpeedOfLight \
    --import-source yes -f -o li_dots_4k \
    uv run python ncu_driver.py --scene dots_4k --side ch08 --lineinfo

# counters of the instrumented twin (no ncu)
uv run python ncu_driver.py --scene dots_4k --counters

# exports the tables were built from
ncu --import ch08_dots_4k.ncu-rep --page raw --csv
ncu --import ch08_dots_4k.ncu-rep -k regex:label_kernel --page source --print-source sass --csv
ncu --import li_dots_4k.ncu-rep -k regex:label_kernel --page source --print-source cuda,sass --csv
```

## Glossary

- **IPC** - warp instructions issued per cycle per SM (at most 4 here).
- **Issue busy** (הנפקה) - the share of cycles a scheduler issued an
  instruction.
- **ALU pipe** (צינור) - the unit for integer and logic instructions;
  on Ada it takes 16 lanes per cycle per scheduler, half rate.
- **Long / short scoreboard** (לוח תלויות) - a warp waits for a memory
  result, or for a shuffle or other MIO result.
- **Wait / math pipe throttle / not selected** - fixed-latency waits,
  a busy pipe, a ready warp that another warp beat to the issue slot.
- **Sectors per request** (מקטעים) - 32-byte pieces one warp memory
  instruction touches; 1-4 is coalesced, 32 is one per lane.
- **Branch efficiency** - the share of branches all active threads
  take the same way.
- **Kernel replay** - ncu runs a kernel several times, restoring its
  memory, to collect all counters.
