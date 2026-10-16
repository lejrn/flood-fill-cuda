"""ncu driver for ch08 (and ch06 for contrast), scratch only.

Builds one scene with ch08's own benchmark builders, warms every kernel,
runs the rgb pipeline once untimed on the real scene, restores the
image, then opens a cuProfilerStart/Stop window around ONE pipeline:

  --side ch08   pack_tiles, init, label (K2), waves (K3), paint (K4):
                TileWaves defaults (hybrid lookup, dense_min 4, L8,
                valve 16, bare twins), exactly the stage list the
                benchmark times
  --side ch06   pack, count, scan, emit, merge, flatten, paint: the
                benchmark's Ch06Side at the session's best merge grid

ncu is run with --profile-from-start off, so only that window is
profiled. After the window the painted image is checked against ch06's
own RunRecolor.run (outside the window).

  --lineinfo    ch08 only: K2/K3/K4 rebuilt with lineinfo=True (numba
                says code generation is unaffected; the driver prints
                registers of both twins so that can be checked)
  --counters    no profiling window: run the instrumented twin and print
                its counters and round_stats as JSON
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
from flood_fill_cuda.chapters.ch08_gpu_nblob_tiles import kernels as k8
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


def lineinfo_twins(eng):
    """K2, K3, K4 of the engine's configuration rebuilt with lineinfo."""
    import numba.cuda as nc
    orig = nc.jit

    def jit_li(*a, **kw):
        kw.setdefault("lineinfo", True)
        return orig(*a, **kw)

    nc.jit = jit_li
    try:
        lab = k8._make_label_kernel(
            eng.connectivity == 8, False, False, eng.label_max_registers,
            eng.index, eng.label_exits)
        wav = k8._make_waves_kernel(
            eng.connectivity == 8, False, False, eng.index, eng.hybrid,
            eng.waves_max_registers)
        pai = k8._make_paint_kernel(
            eng.connectivity == 8, k8.MODE_RGB, eng.paint_fast, False, False)
    finally:
        nc.jit = orig
    return lab, wav, pai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--side", choices=("ch08", "ch06"), default="ch08")
    ap.add_argument("--lineinfo", action="store_true")
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
        engi = ff.TileWaves(w, h, instrumented=True, pack_variant="direct")
        d_img.copy_to_device(d_pristine)
        engi.run(d_img, "rgb")
        cuda.synchronize()
        c = engi.read_counters()
        st = engi.round_stats()
        ok = bool(np.array_equal(d_img.copy_to_host(), ref))
        print("COUNTERS " + json.dumps({"scene": args.scene, "w": w, "h": h,
                                        "n_tiles": engi.n_tiles,
                                        "counters": c, "round_stats": st,
                                        "paint_matches_ch06": ok,
                                        "ch06_runs": side6.counters()["runs"],
                                        "host_runs": n_runs_host}))
        return

    if args.side == "ch08":
        eng = ff.TileWaves(w, h, pack_variant="direct")
        res = {"label": _one(eng.label_k.get_regs_per_thread()),
               "waves": _one(eng.waves_k.get_regs_per_thread()),
               "paint": _one(eng.paint_k.get_regs_per_thread())}
        print(f"bare regs {res} waves grid {eng.waves_blocks} x "
              f"{eng.waves_tpb}", flush=True)
        if args.lineinfo:
            lab, wav, pai = lineinfo_twins(eng)
            eng.label_k, eng.waves_k, eng.paint_k = lab, wav, pai
            # compile them on this scene (one untimed run below), then
            # check the cooperative capacity of the new K3
        stages = eng.stages(d_img, "rgb")
    else:
        stages = side6.stages(d_img, "ch06", "rgb")

    # untimed warm run on the real scene (compiles the lineinfo twins)
    d_img.copy_to_device(d_pristine)
    for _, launch in stages:
        launch()
    cuda.synchronize()
    if args.side == "ch08" and args.lineinfo:
        li = {"label": _one(eng.label_k.get_regs_per_thread()),
              "waves": _one(eng.waves_k.get_regs_per_thread()),
              "paint": _one(eng.paint_k.get_regs_per_thread())}
        cap = next(iter(eng.waves_k.overloads.values())) \
            .max_cooperative_grid_blocks(eng.waves_tpb)
        print(f"lineinfo regs {li} coop cap {cap}", flush=True)
        if li != res or cap < eng.waves_blocks:
            print("WARNING: lineinfo twin differs from the bare twin",
                  flush=True)
    ok_warm = bool(np.array_equal(d_img.copy_to_host(), ref))

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
