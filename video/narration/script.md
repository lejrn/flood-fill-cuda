# Narration script

Concept: "What moves?" Three acts around one question, the unit of work.
One pixel (CPU), pixels in parallel (ch01-ch03), blobs in parallel
(ch04-ch05), runs (ch06).

Target: 45-55 s. English voice. Every number below is quoted from the
parent repo's READMEs and committed benchmark JSON. Session used for
the ch05 vs ch06 pair is `runs_20260725T161448Z.json` (58.51 ms vs
1.46 ms). "16,000x" is 24,083 ms / 1.455 ms.

Each beat is one TTS call, so the assembly step knows where every beat
starts and ends. The `## beat` headings are parsed by the TTS scripts:
the first fenced block under each heading is the spoken text.

## beat 00_hook

```text
Eighty-one million pixels. Two and a half thousand red blobs. How long to paint them all?
```

On screen: crop of `input_blobs.png`, a stopwatch appears at 0.
Visual: `results/ch06_gpu_nblob_runs/figures/before_after.gif` (left half).

## beat 01_cpu

```text
A CPU walks them one pixel at a time. Pure Python: twenty-four seconds. Numba: one point three.
```

On screen: 12x12 grid, one cell lights per tick. Stopwatch: 24,083 ms, then 1,346 ms.
Visual: Manim.

## beat 02_gpu_waves

```text
A GPU floods in waves, the whole frontier at once. One block is one of twenty-four SMs, four percent. All of them: twenty times the CPU.
```

On screen: diamond wave (4-conn BFS) filling a square, one full ring per tick.
Beside it, 24 SM tiles: 1 lit (4%), then 2, then all 24. Caption: "64 Mpx in 100 ms".
Visual: Manim + `ch03/wavefront/square256_b8_t32.gif`.

## beat 03_twist

```text
But on the real image, it lost to the CPU. Every blob was its own launch. Two and a half thousand launches.
```

On screen: back to the real image. Stopwatch jumps to 2,181 ms in red, beside CPU 1,346 ms.
Caption: "one blob = one launch, x 2,522".
Visual: Manim.

## beat 04_blobs_together

```text
So the label rides inside the queue, and every blob floods in one launch. Colliding waves merge in flight. Seven hundred fifty-five thousand blobs, no seeds, twenty-five milliseconds.
```

On screen: two waves racing down a U, colliding at the bridge, the seam erased.
Then all 2,522 blobs flooding at once.
Visual: `ch05/wavefront/u192_merge_prov.gif` then `u192_merge_final.gif`, then `input_blobs_final.gif`.

## beat 05_runs

```text
Then: why move pixels at all? A red span in a row is one run. Thirteen million pixels, five hundred thousand runs. Twenty-five times fewer. Fifty-eight milliseconds becomes one and a half.
```

On screen: one pixel row, red spans collapse into single segments.
Four bars: 81,000,000 / 13,451,960 / 539,207 / 2,522.
Stopwatch: 58.51 ms, then 1.46 ms. Small caption: "1.46 ms packed mask, 2.96 ms from RGB".
Visual: Manim, data from `runs_vs_pixels.svg`.

## beat 06_outro

```text
Twenty-four seconds to one and a half milliseconds. Sixteen thousand times. Not a smarter algorithm. A better representation.
```

On screen: log-scale bar chart, bars appear one by one, pure Python down to ch06.
End card: before/after crop, title, repo name.
Visual: Manim, data from `chain.svg`. ch01-ch04 bars marked with "~" (estimated in the overview JSON).
