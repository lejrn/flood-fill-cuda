"""
Numba vs Triton on the ch00 single-block BFS prototype.

The prototype's only benchmark is profile_kernel(num_runs=100): a fresh
400x400 random-walk scene for every run, the launch + synchronize timed, the
average over 100 runs reported. The cases mirror it:

- profile_kernel: 100 seeded scenes, one per timed round (the warm-up uses
                  one more). Round r runs scene r on both backends, so the
                  median is over 100 different blobs, as in the prototype.
- fixed_scene:    the seed-0 scene, timed for as many rounds, for a
                  per-scene number without the blob-to-blob spread.

Both backends run the prototype's own configuration: one block of 64
threads (Triton: one program, num_warps 2), new_color passed as a host
array (both pay the implicit round trip Numba does for it). kernel_ms is
profile_kernel's bracket (launch + synchronize); total_ms adds the
cuda.to_device of img and visited and the copies back.

same() compares the deterministic outputs (the prototype's contract while
the blob fits its 6000-slot queue, which every scene here does: the blobs
are 1.5k-3.2k pixels): visited, the R and G channels, the recolored mask,
every untouched pixel, and the queue_front the Numba kernel prints (captured
from its device print) against the value the twin stores. The debug blue
channel ((tid * 4) % 255) is schedule-dependent and not compared.

Run (writes results/triton_twins/ch00_cpu_baseline/compare_<UTC>.json):

    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch00_cpu_baseline.compare
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch00_cpu_baseline.compare --quick

--quick is a smoke test (2 scenes, 1 repeat, no JSON). --repeats N sets
the rounds of both cases (default 100, profile_kernel's num_runs).
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import ctypes
import sys
import tempfile
import time

import numpy as np

from flood_fill_cuda.chapters.ch00_cpu_baseline import single_block as numba_proto
from flood_fill_cuda.shared.cpu_oracle import cpu_flood_fill_8
from flood_fill_cuda.triton_twins.compare.harness import (
    Case, arrays_equal, run_cases,
)
from flood_fill_cuda.triton_twins.runtime import kernel_resources

from .single_block import (
    QUEUE_CAPACITY, THREADS_PER_BLOCK, PrototypeRun, compiled_kernel,
    run_flood_fill, setup_scene,
)

CHAPTER = "ch00_cpu_baseline"
NUM_RUNS = 100          # profile_kernel's default num_runs
PROFILE_SEED = 1000     # warm-up scene 1000, round r scene 1001 + r
WIDTH = HEIGHT = 400

_libc = ctypes.CDLL(None)


def numba_run(scene):
    """The prototype's launch with run_flood_fill's brackets, on Numba.

    The kernel prints queue_front at exit. fd 1 is pointed at a temp file
    for the whole call (CUDA printf writes through the C stdout) and the
    value is parsed back, so it can be compared with the twin's. The
    redirect is set up before the first bracket and undone after the last.
    """
    from numba import cuda

    img, visited, sx, sy, w, h, new_color, tpb, bpg = scene
    color = np.array(new_color, copy=True)

    sys.stdout.flush()
    _libc.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile() as f:
        os.dup2(f.fileno(), 1)
        try:
            t0 = time.perf_counter()
            d_img = cuda.to_device(img)
            d_visited = cuda.to_device(visited)
            cuda.synchronize()
            t1 = time.perf_counter()
            numba_proto.flood_fill[bpg, tpb](d_img, d_visited, sx, sy, w, h, color)
            cuda.synchronize()
            t2 = time.perf_counter()
            img_out = d_img.copy_to_host()
            visited_out = d_visited.copy_to_host()
            t3 = time.perf_counter()
            _libc.fflush(None)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        printed = f.read().decode().split()
    return PrototypeRun(
        img=img_out, visited=visited_out,
        front=int(printed[-1]) if printed else -1, threads_per_block=tpb,
        h2d_ms=(t1 - t0) * 1000, kernel_ms=(t2 - t1) * 1000,
        d2h_ms=(t3 - t2) * 1000, total_ms=(t3 - t0) * 1000)


def same_run(nb, tri):
    if nb.front != tri.front:
        return False, f"front: numba printed {nb.front} vs triton {tri.front}"
    nb_rec = (nb.img[..., 0] == 0) & (nb.img[..., 1] == 0)
    tri_rec = (tri.img[..., 0] == 0) & (tri.img[..., 1] == 0)
    return arrays_equal(
        visited=(nb.visited, tri.visited),
        img_rg=(nb.img[..., :2], tri.img[..., :2]),
        recolored=(nb_rec, tri_rec),
        untouched=(nb.img[~nb_rec], tri.img[~nb_rec]))


class SceneSequence:
    """Scene i of a seeded sequence, built once and shared by both runners
    (each runner keeps its own position, and the harness calls each exactly
    once per round, so both always run the same scene)."""

    def __init__(self, first_seed, fixed=False):
        self.first_seed = first_seed
        self.fixed = fixed
        self.cache = {}

    def scene(self, i):
        key = 0 if self.fixed else i
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k >= key - 1}
            scene = setup_scene(self.first_seed + key)
            img, _, sx, sy = scene[:4]
            filled = cpu_flood_fill_8(img, sx, sy)[3]
            if filled > QUEUE_CAPACITY:  # Numba would read out of bounds
                raise RuntimeError(f"scene {self.first_seed + key} has "
                                   f"{filled} px, over the 6000-slot queue")
            self.cache[key] = scene
        return self.cache[key]

    def runner(self, run):
        position = [0]

        def call():
            scene = self.scene(position[0])
            position[0] += 1
            return run(scene)
        return call


def _one(values):
    values = values.values() if isinstance(values, dict) else [values]
    values = sorted(set(int(v) for v in values))
    return values[0] if len(values) == 1 else values


def numba_resources():
    k = numba_proto.flood_fill
    return {"regs_per_thread": _one(k.get_regs_per_thread()),
            "shared_bytes": _one(k.get_shared_mem_per_block()),
            "local_bytes_per_thread": _one(k.get_local_mem_per_thread())}


def info(nb, tri):
    return {
        "front": nb.front,
        "recolored": int(((nb.img[..., 0] == 0) & (nb.img[..., 1] == 0)).sum()),
        "visited": int(nb.visited.sum()),
        "numba": {"grid": [1, THREADS_PER_BLOCK], **numba_resources(),
                  "h2d_ms": nb.h2d_ms, "d2h_ms": nb.d2h_ms},
        "triton": {"grid": [1], "BLOCK": THREADS_PER_BLOCK,
                   "num_warps": THREADS_PER_BLOCK // 32,
                   **kernel_resources(compiled_kernel(THREADS_PER_BLOCK)),
                   "h2d_ms": tri.h2d_ms, "d2h_ms": tri.d2h_ms},
    }


def build_cases():
    config = {"threads_per_block": THREADS_PER_BLOCK, "blocks_per_grid": 1,
              "connectivity": 8, "queue_capacity": QUEUE_CAPACITY}
    seq = SceneSequence(PROFILE_SEED)
    fixed = SceneSequence(0, fixed=True)
    cases = [
        Case(experiment="profile_kernel",
             scene=f"random_walk_400 seeds {PROFILE_SEED}..",
             config=config,
             run_numba=seq.runner(numba_run),
             run_triton=seq.runner(lambda s: run_flood_fill(*s)),
             same=same_run, pixels=WIDTH * HEIGHT, info=info,
             notes=("profile_kernel's method: a fresh seeded 400x400 "
                    "random-walk scene every round (round r = seed "
                    f"{PROFILE_SEED + 1} + r), median over the rounds")),
        Case(experiment="fixed_scene", scene="random_walk_400 seed 0",
             config=config,
             run_numba=fixed.runner(numba_run),
             run_triton=fixed.runner(lambda s: run_flood_fill(*s)),
             same=same_run, pixels=WIDTH * HEIGHT, info=info,
             notes="setup_scene(rng_seed=0), the same blob every round"),
    ]
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 scenes, 1 repeat, no JSON")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per case (default {NUM_RUNS}, "
                         f"profile_kernel's num_runs)")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write the comparison JSON")
    args = ap.parse_args(argv)

    cases = build_cases()
    repeats = args.repeats or (1 if args.quick else NUM_RUNS)
    meta = {
        "source_benchmark": ("chapters/ch00_cpu_baseline/single_block.py "
                             "profile_kernel(num_runs=100)"),
        "quick": args.quick,
        "profile_seeds": [PROFILE_SEED, PROFILE_SEED + repeats],
        "caps": [],  # 400x400 scenes, nothing to cap
        "notes": [
            "Grid is 1 block of 64 threads (Numba [1, 64]) = 1 program "
            "(Triton (1,), num_warps 2) on both sides.",
            "The Numba kernel prints queue_front at exit; that device printf "
            "is inside Numba's kernel_ms. The twin stores the value instead "
            "and the host reads it after the timed brackets.",
            "Numba's 6000-slot queue and scalars are shared memory; the "
            "twin's live in a preallocated global scratch (Triton has no "
            "user-addressable shared memory). Per-lane rear atomics on both "
            "sides: shared-memory atomics in Numba, L2 atomics in Triton.",
            "new_color is a host array on both sides: Numba's implicit "
            "to-device and copy-back are inside kernel_ms, and the twin "
            "does the same round trip.",
            "profile_kernel's own statistic is the mean; the harness reports "
            "the median and min over the rounds.",
        ],
    }
    return run_cases(CHAPTER, cases, repeats=repeats, meta=meta,
                     write=not (args.quick or args.no_write),
                     spin_seconds=0 if args.quick else 8.0)


if __name__ == "__main__":
    main()
