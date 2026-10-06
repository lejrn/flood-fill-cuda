"""The declarative stage table: one entry per beat, everything a scene needs.

Numbers in captions are placeholders (`{pure}`, `{njit}`, `{ch05}`,
`{mask}`, `{rgb}`, `{n_blobs}`, `{ratio}`, `{sm}`) formatted from
`panes.data` at build time, never typed here. Hardware facts (ring size,
grid shapes, block counts) are quoted from the chapter code:
  ch01 kernels.py RING_CAPACITY = 8192
  ch02 flood_fill.py: 2 blocks, cooperative launch, 2 grid.sync per level
  ch03 wavefront.py: the replay grid is 8 x 32; the benchmark grid 48 x 256
  ch05 flood_fill.py: 48 x 256 cooperative + a plain (256, 256) cleanup grid
  ch06 recolor.py: PHASE_BLOCKS count 256 / emit 512 / merge 512 /
       flatten 512 / paint 1024 at 256 tpb, pack on a 2D grid, scan 1 x 1024
The Triton stage quotes the twins' README (triton_twins/README.md): a
Triton program has no user-addressable shared memory (queues, rings and
block clocks live in global scratch), it launches on CuPy memory, the
grand table pins both backends to Numba's own grid, and every twin passes
the Numba chapter's tests against the same CPU oracles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

N_SM = 24


def sm_blocks(n_blocks: int, per_sm: int = 1) -> tuple:
    """Block ids per SM tile, filling SM 0, 1, 2 ... with `per_sm` blocks each."""
    tiles = [[] for _ in range(N_SM)]
    for b in range(n_blocks):
        tiles[(b // per_sm) % N_SM].append(b)
    return tuple(tuple(t) for t in tiles)


@dataclass(frozen=True)
class GpuSpec:
    mode: str = "gpu"                          # "cpu" | "gpu"
    sm_blocks: tuple = sm_blocks(0)            # 24 tuples of block ids
    tpb: int = 256
    palette: str = "golden"                    # single | pair | golden | teal | purple
    unit: str = "block"                        # the inset's word for a block ("program" in Triton)
    coop: bool = False
    shared: tuple = ()                         # lines under the block inset
    memory: tuple = ("image", "visited / depth")
    memory_title: str = "global memory"
    stencil: int | None = None                 # 4 | 8
    queue_chips: bool = False
    kernels: tuple | None = None               # ((name, blocks), ...)
    lines: tuple = ()                          # mono captions at the bottom


@dataclass(frozen=True)
class Stage:
    k: int
    beat: str
    tag: str
    col: str | None                            # matrix column key
    frames: str | None                         # assets/<name> replay
    replay_fps: float
    thumb: str                                 # final frame, assets-relative
    caption: str
    caption2: str | None
    gpu: GpuSpec
    target_s: float                            # beat length without timing.json
    big_fit: tuple = (4.1, 4.1)                # max (w, h) of the big image
    big_dy: float = 0.0                        # vertical offset of the big image
    extra: str | None = None                   # extra builder for the big area
    matrix: str = "ms"                         # the matrix view: "ms" | "triton" (Numba / Triton)


GPU_CPU = GpuSpec(
    mode="cpu", palette="single",
    memory=("image", "deque of (x, y)"), memory_title="host RAM",
    lines=("1 core, 1 pixel per step", "GPU idle"),
)
GPU_CH01 = GpuSpec(
    sm_blocks=sm_blocks(1), tpb=256, palette="single",
    shared=("shared memory: ring of 8,192 slots",),
    memory=("image", "visited / depth", "spill tier"),
    lines=("grid 1 × 256 threads", "2 syncthreads per level", "1 of {sm} SMs = 4%"),
)
GPU_CH02 = GpuSpec(
    sm_blocks=sm_blocks(2), tpb=256, palette="pair", coop=True,
    memory=("image", "visited / depth", "one queue"),
    lines=("grid 2 × 256 threads", "2 grid.sync per level", "2 of {sm} SMs = 8%"),
)
GPU_CH03_4 = GpuSpec(
    sm_blocks=sm_blocks(8), tpb=32, palette="golden", coop=True, stencil=4,
    memory=("image", "visited / depth", "one queue"),
    lines=("grid 8 × 32 shown here", "benchmark: 48 × 256, 2 per SM", "grid-stride over one queue"),
)
GPU_CH03_8 = GpuSpec(
    sm_blocks=sm_blocks(8), tpb=32, palette="golden", coop=True, stencil=8,
    memory=("image", "visited / depth", "one queue"),
    lines=("grid 8 × 32 shown here", "8 neighbours per pixel", "half the levels, half the barriers"),
)
GPU_CH04 = GpuSpec(
    sm_blocks=sm_blocks(8), tpb=32, palette="golden", coop=True, stencil=None,
    queue_chips=True,
    memory=("image", "visited / depth", "queue: x, y, label"),
    lines=("grid 8 × 32 shown here", "one launch, two seeds", "real image: one launch per blob"),
)
GPU_CH05 = GpuSpec(
    sm_blocks=sm_blocks(48, per_sm=2), tpb=256, palette="golden", coop=True,
    memory=("image", "label map", "union-find (atomicMin)"),
    lines=("grid 48 × 256, cooperative", "+ 256 × 256 cleanup grid", "seeds found on the GPU"),
)
GPU_CH06 = GpuSpec(
    sm_blocks=sm_blocks(48, per_sm=2), tpb=256, palette="teal", coop=False,
    kernels=(("pack", "2D grid"), ("count", "256"), ("scan", "1 × 1024 thr"),
             ("emit", "512"), ("merge", "512"), ("flatten", "512"), ("paint", "1024")),
    memory=("image", "mask, 1 bit / px", "run table", "labels"),
    lines=("plain launches, no barrier", "1 warp per row, per run", "32 lanes = 128 B, coalesced"),
)
GPU_TRITON = GpuSpec(
    sm_blocks=sm_blocks(48, per_sm=2), tpb=256, palette="purple", unit="program",
    shared=("no shared memory: queues live in L2",),
    memory=("image", "labels", "queues and rings"), memory_title="global memory, CuPy arrays",
    lines=("same grids, same shapes", "same tests, same oracles", "outputs identical"),
)

STAGES: tuple = (
    Stage(0, "00_cpu", "CPU", "cpu", "ch00_cpu_square", 16.0,
          "ch00_cpu_square/frame_096.png",
          "one pixel at a time",
          "real image: pure Python {pure} · @njit {njit}",
          GPU_CPU, 10.0),
    Stage(1, "01_one_block", "1 block", "ch01", "ch01_square_1block", 16.0,
          "ch01_square_1block/frame_096.png",
          "one block, one SM",
          "the whole frontier at once, 256 threads",
          GPU_CH01, 11.0),
    Stage(2, "02_two_blocks", "2 blocks", "ch02", "ch02_global_square", 16.0,
          "ch02_global_square/frame_096.png",
          "two blocks, one global queue",
          "blue and green: which block filled the pixel",
          GPU_CH02, 9.0),
    Stage(3, "03_n_blocks", "N blocks", "ch03_conn4", "ch03_square_conn4", 16.0,
          "ch03_square_conn4/frame_096.png",
          "every block takes the next pixels",
          "one hue per block, ownership speckles",
          GPU_CH03_4, 10.0),
    Stage(4, "04_conn8", "8-conn", "ch03_conn8", "ch03_square_conn8", 16.0,
          "ch03_square_conn8/frame_096.png",
          "8 neighbours, half the levels",
          "the wave is a square, not a diamond",
          GPU_CH03_8, 8.0),
    Stage(5, "05_two_blobs", "2 blobs", "ch04", "ch04_asym_multisource", 16.0,
          "ch04_asym_multisource/frame_096.png",
          "two blobs, one launch",
          "the label rides inside the queue entry",
          GPU_CH04, 10.0),
    Stage(6, "06_n_blobs", "N blobs", "ch05", "ch05_u_prov", 24.0,
          "ch05_input_blobs/frame_074.png",
          "{n_blobs} blobs, no seeds, one launch",
          "the real image, 9000², {ch05}",
          GPU_CH05, 12.0),
    Stage(7, "07_runs", "runs", "ch06", None, 0.0,
          "ch06_after_half/frame_000.png",
          "a red span is one run",
          "{mask} from a packed mask · {rgb} from RGB",
          GPU_CH06, 13.0, big_fit=(4.1, 2.4), big_dy=-0.75, extra="runs_row"),
    Stage(8, "08_triton", "Triton", None, None, 0.0,
          "",
          "every chapter, rebuilt in Triton",
          "same tests · outputs identical on every run",
          GPU_TRITON, 12.0, extra="triton", matrix="triton"),
    Stage(9, "09_outro", "", None, None, 0.0,
          "",
          "", None,
          GPU_CH06, 9.0),
)
