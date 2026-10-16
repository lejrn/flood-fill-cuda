"""ncu driver for ch03's BFS kernels, through ch03's own flood_fill() and scenes.

Runs flood_fill() at its default grid (threads_per_block=256, blocks=None ->
the kernel's cooperative maximum) `--runs` times on one scene. Kernel launch
order for the chosen dispatcher: #0 is flood_fill's own tiny 1 x 32 JIT
warm-up, #1 the warm-up fill, #2 the profiled fill -> ncu --launch-skip 2
--launch-count 1.

--lineinfo sets lineinfo on ch03's dispatcher in memory before its first
compile (no tracked file changes), so SourceCounters can map PCs to lines.
--sass prints register count, local memory and a hash of the SASS opcode
stream (debug directives stripped) so the lineinfo build can be checked
against the plain build.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import gc
import hashlib
import re
import sys

import numpy as np
from numba import cuda

from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import scenes
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import flood_fill as ff

SCENES = {
    # the builders of benchmark_connectivity_and_barrier_work.py
    "sq_2000_center": lambda: scenes.square_scene(2000, 2000, 1000, 1000),
    "sq_4000_corner": lambda: scenes.full_red_scene(4000, 4000),
    "sq_8000_center": lambda: scenes.square_scene(8000, 8000, 8000, 8000),
}

CONFIGS = {
    "wc_bare": dict(connectivity=8, probe_layout="warp", bare=True),
    "conn8_bare": dict(connectivity=8, probe_layout="thread", bare=True),
    "wc": dict(connectivity=8, probe_layout="warp", bare=False),
    "conn8": dict(connectivity=8, probe_layout="thread", bare=False),
}


def _key(cfg):
    return (cfg["bare"], cfg["connectivity"], 1, cfg["probe_layout"])


def sass_report(kern):
    ov = next(iter(kern.overloads.values()))
    regs = ov.get_regs_per_thread() if hasattr(ov, "get_regs_per_thread") else None
    local = ov.get_local_mem_per_thread() if hasattr(ov, "get_local_mem_per_thread") else None
    sass = ov.inspect_sass()
    ops = []
    for line in sass.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?);", line)
        if m:
            ops.append(re.sub(r"\s+", " ", m.group(1)))
    h = hashlib.sha1("\n".join(ops).encode()).hexdigest()[:16]
    calls = sorted({o for o in ops if o.startswith("CALL") or "BAR" in o
                    or "MEMBAR" in o or "ATOM" in o or "RED" in o})
    print(f"regs={regs} local={local} sass_instructions={len(ops)} "
          f"sass_hash={h}")
    print("barrier/atomic/call opcodes:", calls)
    return ops


def hot_runs(a, cfg, kern, img, sx, sy):
    """Launch #0 is flood_fill's JIT warm-up (via max_blocks), #1 a warm-up
    fill, #2 the profiled fill, issued right after a 3 s GPU spin with no
    host work in between (restore + reset happen on the device)."""
    from flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks import benchmark as b7
    blocks = ff.max_blocks(256, **cfg)
    w, h = img.shape[0], img.shape[1]
    d_pristine = cuda.to_device(img)
    d_img = cuda.device_array_like(img)
    r3 = b7.Ch03Runner(w, h, blocks)
    for i in range(2):
        if i == 1:
            b7._spin_up(3.0)
        d_img.copy_to_device(d_pristine)
        r3.reset(sx, sy)
        e0, e1 = r3.run(kern, cfg["bare"], blocks, 256, d_img)
        e1.synchronize()
        c = r3.read()
        print(f"hot run={i} grid={blocks}x256 levels={c['levels']} "
              f"filled={c['filled']} event_ms="
              f"{cuda.event_elapsed_time(e0, e1):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--scene", choices=list(SCENES), required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--lineinfo", action="store_true")
    ap.add_argument("--sass", action="store_true")
    ap.add_argument("--sass-out", default=None)
    ap.add_argument("--hot", action="store_true",
                    help="clock cross-check: ch07's Ch03Runner (device "
                    "buffers, device reset) and a GPU spin right before "
                    "the profiled launch, so it starts at the boost clock")
    a = ap.parse_args()

    cfg = CONFIGS[a.config]
    kern = ff._KERNELS[_key(cfg)]
    if a.lineinfo:
        kern.targetoptions["lineinfo"] = True

    if a.sass:
        ff.max_blocks(256, **cfg)  # compiles via flood_fill's own warm-up
        ops = sass_report(kern)
        if a.sass_out:
            with open(a.sass_out, "w") as f:
                f.write("\n".join(ops) + "\n")
        print("coop max blocks @256:", ff.max_blocks(256, **cfg))
        return

    img, sx, sy = SCENES[a.scene]()
    if a.hot:
        return hot_runs(a, cfg, kern, img, sx, sy)
    for i in range(a.runs):
        r = ff.flood_fill(img, sx, sy, **cfg)
        print(f"run={i} scene={a.scene} config={a.config} "
              f"grid={r.blocks}x{r.threads_per_block} levels={r.levels} "
              f"filled={r.filled} cas_attempts={r.cas_attempts} "
              f"host_kernel_ms={r.kernel_ms:.2f}", flush=True)
        del r
        gc.collect()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
