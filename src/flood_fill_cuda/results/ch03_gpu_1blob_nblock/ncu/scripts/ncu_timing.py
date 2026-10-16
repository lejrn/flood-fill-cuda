"""Policy-compliant timing of ch03's two bare BFS kernels at the default grid,
for converting the ncu location shares into milliseconds on one clock.

Same method as ch07's benchmark (its helpers are imported, not copied):
8 s clock ramp, ch07's barrier probe at the kernels' grid, then per scene
an untimed correctness pass (wc_bare and conn8_bare must paint the same
image), a 1 s re-ramp, and 9 interleaved rounds (order reversed on odd
rounds). Every run: image restored from a device copy, ch03 device reset,
96 MB L2 flush read, GPU sleep kernel queued, CUDA events around the one
kernel launch. Medians, ranges and the per-round paired ratio. A
background nvidia-smi sampler records the clocks of each timed window.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import gc
import json
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks import benchmark as b7
from flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.benchmarks import probes as p7
from flood_fill_cuda.chapters.ch07_gpu_1blob_tiles.kernels import gpu_sleep_kernel
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import flood_fill as ff3
from flood_fill_cuda.chapters.ch03_gpu_1blob_nblock import scenes

warnings.simplefilter("ignore", NumbaPerformanceWarning)

ROUNDS = 9
TPB = 256
SCENES = [
    ("sq_2000_center", lambda: scenes.square_scene(2000, 2000, 1000, 1000)),
    ("sq_4000_corner", lambda: scenes.full_red_scene(4000, 4000)),
    ("sq_8000_center", lambda: scenes.square_scene(8000, 8000, 8000, 8000)),
]
CONFIGS = {"wc_bare": "warp", "conn8_bare": "thread"}


def main(out_path):
    launch = {}
    for n, layout in CONFIGS.items():
        blocks = ff3.max_blocks(TPB, bare=True, connectivity=8,
                                probe_layout=layout)
        launch[n] = {"kernel": ff3._KERNELS[(True, 8, 1, layout)],
                     "blocks": blocks}
    grid_blocks = {n: c["blocks"] for n, c in launch.items()}
    print("default grids:", {n: f"{b}x{TPB}" for n, b in grid_blocks.items()},
          flush=True)

    sampler = b7.ClockSampler()
    res = {"method": __doc__.strip(), "tpb": TPB, "grids": grid_blocks,
           "rounds": ROUNDS, "started_utc": datetime.now(timezone.utc)
           .strftime("%Y%m%dT%H%M%SZ"), "clocks_start": b7._clocks()}
    p7.warm()
    b7._spin_up(b7.SPIN_SECONDS)
    grids = sorted({(b, TPB) for b in grid_blocks.values()})
    t0 = time.time()
    res["barrier_us_per_level"] = p7.barrier_us(grids)
    res["barrier_clocks"] = sampler.window(t0, time.time())
    print("barrier probe:", {k: round(v["median"], 3) for k, v in
                             res["barrier_us_per_level"].items()},
          res["barrier_clocks"], flush=True)

    flush = cuda.to_device(np.zeros(b7.L2_FLUSH_BYTES // 4, np.uint32))
    sink = cuda.device_array(1, np.uint32)
    res["scenes"] = []
    for name, builder in SCENES:
        img, sx, sy = builder()
        w, h = img.shape[0], img.shape[1]
        d_pristine = cuda.to_device(img)
        d_img = cuda.device_array_like(img)
        r3 = b7.Ch03Runner(w, h, max(grid_blocks.values()))

        def run_once(n):
            d_img.copy_to_device(d_pristine)
            r3.reset(sx, sy)
            b7._l2_flush[512, 256](flush, sink)
            cuda.synchronize()
            gpu_sleep_kernel[1, 32](b7.SLEEP_NAPS)
            c = launch[n]
            e0, e1 = r3.run(c["kernel"], True, c["blocks"], TPB, d_img)
            e1.synchronize()
            return cuda.event_elapsed_time(e0, e1)

        meta, paints = {}, {}
        for n in CONFIGS:
            run_once(n)
            meta[n] = r3.read()
            paints[n] = d_img.copy_to_host()
        paint_match = bool(np.array_equal(paints["wc_bare"],
                                          paints["conn8_bare"]))
        del paints
        gc.collect()

        b7._spin_up(b7.SCENE_SPIN_SECONDS)
        names = list(CONFIGS)
        samples = {n: [] for n in names}
        t_a = time.time()
        for i in range(ROUNDS):
            order = names if i % 2 == 0 else list(reversed(names))
            for n in order:
                samples[n].append(run_once(n))
        t_b = time.time()
        ratios = [c / w_ for c, w_ in zip(samples["conn8_bare"],
                                          samples["wc_bare"])]
        row = {"scene": name, "filled": meta["wc_bare"]["filled"],
               "levels": meta["wc_bare"]["levels"],
               "levels_conn8": meta["conn8_bare"]["levels"],
               "filled_conn8": meta["conn8_bare"]["filled"],
               "paint_match": paint_match,
               **sampler.window(t_a, t_b)}
        for n in names:
            k = samples[n]
            row[f"{n}_ms"] = statistics.median(k)
            row[f"{n}_ms_min"] = min(k)
            row[f"{n}_ms_max"] = max(k)
            row[f"{n}_runs"] = k
        row["conn8_over_wc_median"] = statistics.median(ratios)
        row["conn8_over_wc_min"] = min(ratios)
        row["conn8_over_wc_max"] = max(ratios)
        res["scenes"].append(row)
        print(f"{name:15s} wc {row['wc_bare_ms']:8.3f} "
              f"[{row['wc_bare_ms_min']:.3f}-{row['wc_bare_ms_max']:.3f}] "
              f"conn8 {row['conn8_bare_ms']:8.3f} "
              f"ratio {row['conn8_over_wc_median']:.3f} "
              f"[{row['conn8_over_wc_min']:.3f}-{row['conn8_over_wc_max']:.3f}] "
              f"levels {row['levels']} filled {row['filled']} "
              f"paint_match {paint_match} sm {row.get('clk_sm_median')} "
              f"mem {row.get('clk_mem_median')}", flush=True)
        del img, d_pristine, d_img, r3
        gc.collect()

    res["clocks_end"] = b7._clocks()
    sampler.stop()
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", os.path.basename(out_path))


if __name__ == "__main__":
    main(sys.argv[1])
