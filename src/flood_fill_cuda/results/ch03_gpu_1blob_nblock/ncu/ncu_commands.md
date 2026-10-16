# ncu commands behind the ch03 profile

Nsight Compute 2025.2.1 (CUDA 12.9 toolkit), driver 610.62, RTX 4060
Laptop GPU (AD107, 24 SMs), WSL2. Run from the repository root.
`scripts/ncu_driver.py` is the driver (ch03's own `flood_fill()` and
scene builders, default grid). Paths are relative; `<lock>` is the shared
GPU lock file.

## What the driver launches

- `--lineinfo` sets `lineinfo` on ch03's dispatcher in memory before its
  first compile. The SASS was checked identical to the plain build (3,256
  instructions for `wc_bare`, 3,408 for `conn8_bare`; only the mangled
  kernel name differs), so line info changes no code.
- Launch #0 of the selected kernel is `flood_fill()`'s own 1 x 32 JIT
  warm-up, #1 a warm-up fill, #2 the profiled fill. Hence
  `--launch-skip 2 --launch-count 1`.
- Kernel replay works for these cooperative launches (no need for
  application replay). ncu flushes caches before every pass
  (`--cache-control all`, the default).

## The three run types

`KRE` is `regex:multi_block_global8wc_bare_kernel` or
`regex:multi_block_global8_bare_kernel`. `EXTRA` is the metric list in
`scripts/run_ncu.sh` (L2 atomic unit, atomic and CAS requests, sectors per
request, DRAM bytes, hit rates, occupancy, issue, divergence).

**base**: full set at locked base clocks (the main tables).

```bash
flock <lock> timeout -s KILL 3600 \
  ncu --target-processes all --clock-control base --replay-mode kernel \
      -k "$KRE" --launch-skip 2 --launch-count 1 \
      --set full --import-source yes \
      --metrics "regex:smsp__pcsamp_warps_issue_stalled_.*,$EXTRA" \
      -f -o reps/<kernel>_<scene>_base_full \
      uv run python scripts/ncu_driver.py --config <kernel> --scene <scene> --lineinfo
```

**boost**: same, `--clock-control none`, with only SpeedOfLight,
WarpStateStats, SchedulerStats, Occupancy, LaunchStats and
SourceCounters (17 passes instead of 41).

**hot**: PC sampling only (WarpStateStats, SourceCounters, LaunchStats;
11 passes), `--clock-control none`, driver flag `--hot`. The profiled
launch is issued right after a 3 s GPU spin, with ch07's `Ch03Runner`
doing the restore and reset on the device, so the first pass runs at
the boost clock (2.06 GHz on 5 of 6 profiles).

## Exports (raw files stay in scratch, never committed)

```bash
ncu --import X.ncu-rep --page raw    --csv --print-units base     > X_raw.csv
ncu --import X.ncu-rep --page source --csv --print-source sass      > X_sass.csv
ncu --import X.ncu-rep --page source --csv --print-source cuda,sass > X_mixed.csv
```

`scripts/ncu_analyze.py` turns those into the curated CSVs here.

## Timing session (not ncu)

```bash
flock <lock> timeout -s KILL 3600 \
  uv run python scripts/ncu_timing.py <out>.json
```

ch07's method and helpers: 8 s clock ramp, barrier probe at 72 x 256,
per scene an untimed correctness pass, 1 s re-ramp, 9 interleaved
rounds. Every run restores the image, resets on the device, reads a 96 MB
buffer (L2 flush), queues a sleep kernel, and times one launch with CUDA
events.
