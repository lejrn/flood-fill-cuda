"""ncu driver for ch08's dense-tile path (the profile after the change).

Builds one scene with ch08's own benchmark builders, warms every kernel,
runs the rgb pipeline once untimed on the real scene, restores the
image, then opens a cuProfilerStart/Stop window around ONE pipeline:

  --side ch08   the engine's defaults, exactly the stage list the
                benchmark times as ch08_L8: pack_tiles, init, label
                (K2), label_ccl (K2b), waves (the dindex K3 with task
                compaction), paint (K4), paint_ccl (K4b); bare twins
  --side old    flood_fill.OLD_PATH (the profile before the change:
                pack_tiles, init, K2, the hybrid K3, K4)
  --side ch06   ch06's run pipeline at the session's best merge grid

ncu runs with --profile-from-start off, so only that window is
profiled. After the window the painted image is checked against ch06's
own RunRecolor.run (outside the window); a mismatch exits 1.

  --counters    no profiling window: run the instrumented twin of the
                side's engine and print its counters as JSON

Run from the repository root with FLOOD_FILL_IMAGES set, for example:

  ncu --target-processes all --profile-from-start off \\
      --clock-control base --set full -f -o after_dots_4k \\
      uv run python ncu_driver.py --scene dots_4k --side ch08
"""
import os
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import json
import sys
import warnings

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda.cudadrv.driver import profile_start, profile_stop

from flood_fill_cuda.chapters.ch08_gpu_nblob_tiles import flood_fill as ff
from flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks import (
    benchmark as B)

warnings.simplefilter("ignore", NumbaPerformanceWarning)

# ch06's best bare merge grid per scene, session waves_20261015T183401Z
CH06_MERGE = {"input_blobs": 4096, "random_4000": 8192, "dots_4k": 2048,
              "zz_lattice_4k": 4096}


def builders():
    pngs = B._resolve_pngs(os.environ.get("FLOOD_FILL_IMAGES"))
    out = {}
    for name, build, *_ in B.main_scenes(pngs):
        out[name] = build
    for name, build, *_ in B.worst_scenes():
        out[name] = build
    return out


def _one(v):
    return int(next(iter(v.values()))) if isinstance(v, dict) else int(v)


def engine(w, h, side, **kw):
    extra = dict(ff.OLD_PATH) if side == "old" else {}
    return ff.TileWaves(w, h, pack_variant="direct", **extra, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--side", choices=("ch08", "old", "ch06"),
                    default="ch08")
    ap.add_argument("--counters", action="store_true")
    args = ap.parse_args()

    img = builders()[args.scene]()
    w, h = img.shape[0], img.shape[1]
    print(f"scene {args.scene} {w}x{h}", flush=True)
    d_pristine = cuda.to_device(img)
    d_img = cuda.device_array_like(img)
    n_runs_host = B._host_run_count(img)
    side6 = B.Ch06Side(w, h, max(8192, int(n_runs_host * 1.05)))
    side6.merge_blocks = CH06_MERGE.get(args.scene, side6.merge_blocks)

    # ch06's own pipeline: the reference paint
    d_img.copy_to_device(d_pristine)
    _, ev = side6.r6.run(d_img, contract="rgb", instrumented=False)
    ev[-1].synchronize()
    ref = d_img.copy_to_host()

    if args.counters:
        engi = engine(w, h, args.side, instrumented=True)
        d_img.copy_to_device(d_pristine)
        engi.run(d_img, "rgb")
        cuda.synchronize()
        c = engi.read_counters()
        ok = bool(np.array_equal(d_img.copy_to_host(), ref))
        print("COUNTERS " + json.dumps({"scene": args.scene, "w": w, "h": h,
                                        "side": args.side,
                                        "n_tiles": engi.n_tiles,
                                        "counters": c,
                                        "paint_matches_ch06": ok,
                                        "ch06_runs": side6.counters()["runs"],
                                        "host_runs": n_runs_host}))
        return

    if args.side in ("ch08", "old"):
        eng = engine(w, h, args.side)
        kerns = {"label": eng.label_k, "waves": eng.waves_k,
                 "paint": eng.paint_k}
        if eng.dense:
            kerns.update(label_ccl=eng.label_dense_k,
                         paint_ccl=eng.paint_dense_k)
        stages = eng.stages(d_img, "rgb")
    else:
        stages = side6.stages(d_img, "ch06", "rgb")

    # untimed warm run on the real scene
    d_img.copy_to_device(d_pristine)
    for _, launch in stages:
        launch()
    cuda.synchronize()
    ok_warm = bool(np.array_equal(d_img.copy_to_host(), ref))
    if args.side in ("ch08", "old"):
        regs = {k: _one(v.get_regs_per_thread()) for k, v in kerns.items()}
        print(f"bare regs {regs}; label grid {eng.label_grid}, "
              f"label_ccl {eng.label_dense_grid}, waves "
              f"{eng.waves_blocks} x {eng.waves_tpb}, paint "
              f"{eng.paint_grid}, paint_ccl {eng.paint_dense_grid}",
              flush=True)

    # the profiled pipeline
    d_img.copy_to_device(d_pristine)
    cuda.synchronize()
    profile_start()
    for _, launch in stages:
        launch()
    cuda.synchronize()
    profile_stop()
    ok = bool(np.array_equal(d_img.copy_to_host(), ref))
    print(f"stages {[n for n, _ in stages]} warm_ok {ok_warm} "
          f"profiled_ok {ok}", flush=True)
    if not (ok and ok_warm):
        sys.exit(1)


if __name__ == "__main__":
    main()
