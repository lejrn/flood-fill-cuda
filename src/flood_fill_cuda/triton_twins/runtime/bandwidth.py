"""
The Triton twin of ``shared.bandwidth.measure_peak_bandwidth``.

Same contract as the Numba probe: two n_bytes int64 buffers, a
grid-stride device-to-device copy at a [1024 programs x 256 lanes] grid,
CUDA-event timing, median GB/s over ``repeats`` copies, read + write =
2 * n_bytes per copy. ``model_bytes`` and ``model_gb_s`` are pure Python
and are reused from ``shared.bandwidth`` unchanged.

Run both probes in one session and the gap between them is how far
Triton's codegen sits from Numba's on a pure streaming kernel. That gap
is the calibration line under every chapter's "% of peak" comparison.
"""

import cupy as cp
import numpy as np
import triton
import triton.language as tl

from . import t


@triton.jit
def _copy_kernel(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n, stride):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=m), mask=m)


def measure_peak_bandwidth(n_bytes=256 * 2 ** 20, repeats=10):
    """Measured D2D copy peak in Triton: median GB/s over ``repeats`` copies."""
    n = n_bytes // 8
    src = cp.empty(n, dtype=cp.int64)  # contents irrelevant
    dst = cp.empty(n, dtype=cp.int64)
    launch = lambda: _copy_kernel[(1024,)](t(dst), t(src), n,
                                           BLOCK=256, num_warps=8)
    launch()  # compile + cache warm-up
    cp.cuda.Device().synchronize()
    runs = []
    for _ in range(repeats):
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        launch()
        e1.record()
        e1.synchronize()
        ms = cp.cuda.get_elapsed_time(e0, e1)
        runs.append(2 * n_bytes / (ms * 1e6))
    del src, dst
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "gb_s": float(np.median(runs)),
        "runs_gb_s": [float(r) for r in runs],
        "n_bytes": int(n_bytes),
    }
