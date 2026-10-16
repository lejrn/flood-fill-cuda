# ch08 under ncu - before and after the dense-tile path

**Question.** The first profile (`before/`) asked why ch08 lost on
tiles full of tiny comps. The dense-tile path was built on its answer.
Did the work go where the profile said it would, and what limits the
new kernels?

**Short answer.** Yes, and by more than the cycles show.

- **`dots_4k`:** the labelling (K2 + K2b) issues 30.0M warp
  instructions instead of 895M (7.2 per comp instead of 213.5). The
  paint (K4 + K4b) issues 60.9M instead of 1,224M.
- **`random_4000`:** labelling 332M to 70.6M, paint 470M to 103M.
- **`zz_lattice_4k`:** K3 issues 46.6M instead of 319M. Its warp
  steps fell from 2.15M to 115k, and its 64.0M signature tests to
  2.05M, every one now a CAS.
- **`input_blobs`:** no tile takes the dense path. K2 issues 4.6% more
  (the estimate), K3 10.6% fewer (6.10M vs 6.82M), the paint the same.

**The new limits.** K2b and K4b wait on shared memory: on `dots_4k`
86% and 84% of their shared-memory wavefronts are bank conflicts
(התנגשויות בנקים). K3 on `zz_lattice_4k` is still latency-bound
(השהיה) on global memory, now with 6.8x fewer instructions.

Labels below: **measured** = an ncu counter or a counter of the
instrumented twin; **derived** = arithmetic on measured numbers;
**inferred** = a reading that no run here tested.

## Method

- **Same as before.** Nsight Compute 2025.2.1, RTX 4060 Laptop (AD107,
  24 SMs, cc 8.9), driver 610.62, Numba 0.65.1, WSL2. One rgb pipeline
  inside a `cuProfilerStart/Stop` window, after a warm run on the real
  scene; `--set full` plus the same extra metrics, plus shared-memory
  wavefronts and bank conflicts; kernel replay with caches flushed per
  pass; `--clock-control base`.
- **Before** (`before/`): the chapter at `1ea2888`, the path now called
  `flood_fill.OLD_PATH` (unchanged since). Its own README has the full
  analysis.
- **After** (`after/`): the default engine of the dense-tile path,
  bare twins, exactly the stage list the benchmark times: pack, init,
  K2, K2b, K3 (the dindex twin with task compaction), K4, K4b.
  Registers 80 / 130 / 80 / 56 / 72 (K2, K2b, K3, K4, K4b).
- **Check.** After each window the painted image was compared with
  ch06's own `RunRecolor.run`: all four matched.
- **Counters.** `after/per_unit.csv` takes its counts from the
  instrumented twin of the same engine (`ncu_driver.py --counters`).

**Clocks.** Before, `--clock-control base` did not hold one SM clock:
it ranged from 0.40 to 1.14 GHz per kernel. After, every kernel of
any size ran at 1.10-1.14 GHz (measured, `after/kernels.csv`).

Instruction counts do not depend on the clock, so the comparison
leans on them. Cycles are compared only where they validate.

**Validation (derived).** Elapsed cycles divided by 2.07 GHz, against
the phase times of the final session `dense_20261016T054922Z`
(`after/cycles_vs_benchmark.csv`):

| kernel, scene | ncu cycles / 2.07 GHz (ms) | benchmark phase (ms) |
|---|---:|---:|
| K2b `dots_4k` | 0.471 | 0.466 |
| K4b `dots_4k` | 0.563 | 0.569 |
| K2b `random_4000` | 0.869 | 0.877 |
| K4b `random_4000` | 0.822 | 0.830 |
| K2b `zz_lattice_4k` | 0.421 | 0.421 |
| K3 `random_4000` | 0.268 | 0.264 |
| K3 `zz_lattice_4k` | 1.108 | 1.567 |

The issue- and shared-memory-bound kernels land within 0.96-1.01. K3
on `zz_lattice_4k` lands at 0.71: it waits on DRAM, whose clock does
not scale with the SM's. So its cycle ratio below understates its
time; the benchmark's phase ratio is 0.366 (1.567 vs 4.277 ms).

## Before and after, by stage (measured)

| scene | stage | Mcycles before | after | warp inst before (M) | after (M) | inst after / before |
|---|---|---:|---:|---:|---:|---:|
| `dots_4k` | labelling (K2 + K2b) | 13.646 | 1.091 | 895.4 | 30.0 | **0.034** |
| `dots_4k` | waves (K3) | 0.488 | 0.222 | 1.52 | 1.31 | 0.862 |
| `dots_4k` | paint (K4 + K4b) | 17.994 | 1.258 | 1,224.0 | 60.9 | **0.050** |
| `random_4000` | labelling | 5.215 | 1.885 | 331.8 | 70.6 | **0.213** |
| `random_4000` | waves | 1.271 | 0.554 | 52.5 | 11.1 | **0.211** |
| `random_4000` | paint | 7.045 | 1.769 | 469.5 | 103.0 | **0.219** |
| `zz_lattice_4k` | labelling | 3.592 | 0.986 | 236.9 | 37.2 | 0.157 |
| `zz_lattice_4k` | waves | 8.877 | 2.294 | 319.3 | 46.6 | **0.146** |
| `zz_lattice_4k` | paint | 0.505 | 0.422 | 23.8 | 23.9 | 1.002 |
| `input_blobs` | labelling | 0.606 | 0.548 | 30.5 | 31.9 | 1.046 |
| `input_blobs` | waves | 0.255 | 0.266 | 6.82 | 6.10 | 0.894 |
| `input_blobs` | paint | 0.996 | 0.932 | 32.7 | 32.8 | 1.002 |

The pack (K1) is the same code and issues the same instructions in
every scene. Whole pipeline, sum of kernels: `dots_4k` 32.599 to 3.032
Mcycles, `random_4000` 13.987 to 4.646, `zz_lattice_4k` 13.439 to
4.154, `input_blobs` 4.094 to 3.856 (`before_after.csv`).

**Per unit (derived, `after/per_unit.csv`).**

| | `input_blobs` | `random_4000` | `dots_4k` | `zz_lattice_4k` |
|---|---:|---:|---:|---:|
| labelling warp inst per comp, before | 826.9 | 365.6 | 213.5 | 242.8 |
| labelling warp inst per comp, after | 865.3 | 77.8 | 7.2 | 38.1 |
| paint warp inst per comp, before | 888.3 | 517.4 | 291.8 | 24.4 |
| paint warp inst per comp, after | 890.4 | 113.5 | 14.5 | 24.4 |
| K2b warp inst per CCL tile | - | 4,189.7 | 1,504.5 | 1,939.1 |
| K3 warp inst per task, before / after | 49.9 / 44.6 | 201.9 / 42.6 | - | 297.7 / 43.5 |

On `dots_4k` every comp is one run, so the labelling costs 7.2 warp
instructions per run. ch06's labelling costs 8.1 per run there
(`before/ch08_vs_ch06.csv`).

## Against ch06 (measured cycles at the ncu clock)

ch06's code did not change, so its rows come from the first profile
(`before/kernels.csv`; emit, merge and flatten ran at 1.13-1.14 GHz).

| scene | labelling ch08 / ch06, before | after | paint ch08 / ch06, before | after |
|---|---:|---:|---:|---:|
| `random_4000` | 1.58 | **0.59** | 0.99 | **0.25** |
| `dots_4k` | 4.94 | **0.46** | 2.07 | **0.14** |

The benchmark agrees: in the final session the labelling tier is 0.580
(`random_4000`) and 0.416 (`dots_4k`) of ch06's. On dense scenes that
tier leaves K4b's CCL rebuild to the paint, so the paint column and the
rgb contract (0.407, 0.250) are the fairer view.

## What limits the new kernels (measured)

| kernel, scene | IPC (of 4) | issue busy | ALU pipe | threads per warp inst | branch eff. | occupancy (achieved / theoretical) | shared wavefronts, bank conflicts | top stalls, cycles per issue |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K2b `dots_4k` | 1.10 | 27.6% | 31.0% | 22.6 | 88.4% | 25.0 / 25.0% | 14.9M, 85.6% | short scoreboard 5.61, wait 2.32 |
| K2b `random_4000` | 1.54 | 39.9% | 43.7% | 15.9 | 63.5% | 24.4 / 25.0% | 10.1M, 51.9% | wait 2.52, short scoreboard 1.82 |
| K2b `zz_lattice_4k` | 1.56 | 39.5% | 43.7% | 17.6 | 83.0% | 24.7 / 25.0% | 5.2M, 50.5% | wait 2.27, short scoreboard 2.23 |
| K4b `dots_4k` | 2.09 | 52.3% | 52.7% | 23.2 | 80.5% | 52.8 / 58.3% | 23.9M, 84.3% | short scoreboard 5.17, wait 2.23 |
| K4b `random_4000` | 2.49 | 63.7% | 63.4% | 17.0 | 69.9% | 47.4 / 58.3% | 12.2M, 48.7% | wait 2.62, not selected 1.38 |
| K3 `zz_lattice_4k` | 0.86 | 21.7% | 22.1% | 20.6 | 86.2% | 49.5 / 50.0% | 1.0M, 1.7% | long scoreboard 17.49, barrier 3.92 |
| K3 `random_4000` | 0.87 | 22.3% | 20.9% | 21.9 | 81.9% | 48.7 / 50.0% | 0.3M, 1.0% | long scoreboard 14.68, barrier 5.47 |

**K2b and K4b (reading).** Shared memory is the limit on `dots_4k`.
Short-scoreboard stalls (the wait for a shared-memory result) lead,
and 84-86% of the wavefronts are replays of bank conflicts. K2b also
runs at 25% occupancy (תפוסה): 130 registers allow 3 warps per
scheduler, 12 per SM.

**A likely cause (inferred, not tested).** Lane i owns row i, and its
slots start at `16 * i`. With 32 banks of 4 bytes, slot `16 * i + k`
falls in bank `k` for even rows and `16 + k` for odd rows, so a warp's
lanes that touch the same `k` share two banks: up to 16-way conflicts.
A row stride of 17 words would spread them over all 32 banks.

On `random_4000` the conflicts are about half the wavefronts, and the
union loops diverge (התפצלות): 15.9-17.0 threads per warp instruction,
branch efficiency 63.5-69.9%.

**K3 (reading).** On `zz_lattice_4k` the index made every signature
test a hit: 2.05M tests, all CAS, against 64.0M tests before. The warp
steps fell from 2.15M to 115k and the rmin atomics from 975,867 to 0.
What is left waits on global memory: long-scoreboard stalls of 17.5
cycles per issue, DRAM at 41.6% of peak. 78.5% of the CAS still lose
(441,056 claims of 2,048,002 CAS).

## Recommendations, ranked (inferred)

1. **Pad K2b's and K4b's slot rows to 17 words.** The conflicts are
   most of the shared traffic on dense tiles, and both kernels wait on
   it.
2. **Fewer registers in K2b** (130 now, 25% occupancy): a cap of 128
   would allow 4 warps per scheduler instead of 3, if the spills stay
   small.
3. **A plain label load before K3's CAS**: 78.5% of the CAS lose on
   `zz_lattice_4k`.
4. **A tighter comp estimate in K2** (the chapter README's noise
   section): no profile here covers it.

## Files

| file | what |
|---|---|
| `before_after.csv` | per scene and stage: Mcycles and warp instructions before and after |
| `vs_ch06.csv` | labelling and paint cycles against ch06, before and after |
| `after/kernels.csv` | every kernel x scene of the dense-tile path: cycles, the measured SM clock, instructions, IPC, pipes, occupancy, branch efficiency, DRAM, L2, global and shared-memory traffic with bank conflicts, the top 3 stalls |
| `after/stalls.csv` | all warp-stall reasons per kernel x scene, cycles per issued instruction |
| `after/per_unit.csv` | counters of the instrumented twin and instructions per comp, CCL tile and task |
| `after/cycles_vs_benchmark.csv` | ncu cycles at 2.07 GHz against the final session's phase times |
| `after/ncu_driver.py` | the driver of the after runs (`--side ch08`, `old`, `ch06`, `--counters`) |
| `before/` | the first profile: its README, tables and driver |

No `.ncu-rep` file or raw ncu export is kept here: they hold host and
process details. Every table was written from them by a script.

## Reproduce

One GPU process at a time, from the repository root, with
`FLOOD_FILL_IMAGES` pointing at the input PNGs. `$M` is the metric list
of `before/README.md` plus
`l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`,
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` and the shared
load, store and atomic instruction counts.

```bash
ncu --target-processes all --profile-from-start off --clock-control base \
    --set full --metrics $M -f -o after_dots_4k \
    uv run python after/ncu_driver.py --scene dots_4k --side ch08
uv run python after/ncu_driver.py --scene dots_4k --counters
ncu --import after_dots_4k.ncu-rep --page raw --csv
```
