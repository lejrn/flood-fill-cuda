# Narration script

Concept: three panes for the whole video. Left, the benchmark matrix
(17 shapes, one column per chapter, a shape glows when the GPU beats the
CPU). Middle, the blob of the current chapter, with every finished chapter
swept up into a strip. Right, the GPU: which SMs, blocks, threads and
memory each chapter uses.

Target: about 2 minutes. English voice, Kokoro `am_onyx` (deep male). Every number below is on screen and
comes from a committed benchmark JSON via `scenes/panes/data.py`
(`uv run python scenes/panes/data.py` prints the whole matrix).

Each beat is one TTS call and one Manim scene (`scenes/sNN_*.py`). The
`## beat` headings are parsed by the TTS scripts: the first fenced block
under each heading is the spoken text.

## beat intro_problem

```text
A defence camera watches a clear sky for drones and missiles. It has to find them in every frame, in real time: thirty frames a second, one frame every thirty-three milliseconds. A motion filter strips the sky away and leaves white blobs on black. Those blobs still have to be labelled.
```

On screen: simulated footage (`assets/make_drone_frames.py`), then its motion mask beside it, then the frame budget line: 30 fps, 33 ms per frame.

## beat intro_budget

```text
Labelling one eighty-one megapixel frame on the CPU takes twenty-four seconds; by then the camera has moved on by seven hundred frames. Even compiled, one point three seconds. On the GPU the same frame takes one and a half milliseconds: real time, with room to spare. This is how we got there.
```

On screen: the CPU panel stuck on frame 1 with a stopwatch running towards 24,083 ms; the GPU panel labelling every frame at 1.46 ms; three bars against the 33 ms budget (pure Python 24,083, @njit 1,346, GPU 1.46).

## beat 00_cpu

```text
A CPU fills a blob one pixel at a time. Pure Python: twenty-four seconds for the real image. Numba: one point three. On the left, the CPU time for seventeen shapes. Every chapter now gets a column, and a shape glows when the GPU wins.
```

On screen: the 256² square filled in CPU visit order (grey), the CPU box on the right, the CPU column of the matrix.

## beat 01_one_block

```text
Chapter one. One block of two hundred fifty-six threads floods the whole frontier at once, its queue in shared memory, a spill tier behind it. One block is one of twenty-four SMs: four percent of the GPU. It beats the CPU only on the biggest shapes.
```

On screen: the same square, one blue block. SM 0 lit, 8 warps x 32 lanes, ring of 8,192 slots, spill tier. Column "1 block": five glows.

## beat 02_two_blocks

```text
Chapter two. Two blocks on two SMs share one global queue, with a grid-wide barrier twice per level. Twice as fast as one block on big blobs. The small shapes still belong to the CPU.
```

On screen: blue and green interleaved. SM 0 and 1 lit, one queue in global memory.

## beat 03_n_blocks

```text
Chapter three. Any number of blocks. Every block takes the next pixels from the same queue, and the hue shows who filled what: ownership speckles. Eight blocks here, forty-eight in the benchmark.
```

On screen: eight hues, eight SMs, one warp of 32 lanes per block, the 4-neighbour stencil.

## beat 04_conn8

```text
Eight neighbours instead of four. The wave becomes a square and needs half the levels, so half the barriers. On the big disk, eleven times the CPU.
```

On screen: the square wave; the 8-neighbour stencil; column "8-conn", disk 4000 at 24.1 ms vs 269 ms.

## beat 05_two_blobs

```text
Chapter four. Two blobs in one launch: the label rides inside the queue entry. But on the real image every blob was still its own launch, two and a half thousand launches, and the GPU lost.
```

On screen: two blobs, blue and green families; the queue box carries label chips; the estimated (dashed) cells on the N-blob rows.

## beat 06_n_blobs

```text
Chapter five. No seeds given. Candidate waves start everywhere and colliding waves merge in flight. Two and a half thousand blobs, one launch, fifty-eight milliseconds. Now the real image glows too.
```

On screen: two waves racing down a U, the seam erased; then all 2,522 blobs of the real image. 48 x 256 cooperative grid, union-find in global memory.

## beat 07_runs

```text
Chapter six. Why move pixels at all? A red span in a row is one run: twenty-five times fewer things to touch. Seven plain launches, no barrier. Fifty-eight milliseconds becomes one and a half.
```

On screen: a pixel row collapses into runs, the recoloured real image; the kernel strip pack to paint; column "runs" in teal.

## beat 08_outro

```text
Twenty-four seconds to one and a half milliseconds. Sixteen thousand times. Not a smarter algorithm. A better representation.
```

On screen: the full strip and the full matrix; centre lines 24,083 ms → 1.46 ms, 16,000×, flood-fill-cuda.
