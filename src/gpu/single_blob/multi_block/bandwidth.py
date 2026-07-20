"""Bandwidth instrumentation: a derived bytes-moved model + a measured peak.

The stage hypothesis says that with enough blocks in flight, latency stops
mattering and BANDWIDTH becomes the constraint. To see that in the numbers,
every benchmark row carries two figures:

1. model_gb_s — algorithmic bytes the kernel logically moved (from its own
   counters) divided by kernel time. This is a DERIVED LOWER-BOUND MODEL,
   not measured DRAM traffic: L2 absorbs the queue's hot window (deflating
   real DRAM bytes) while 32 B sector granularity inflates them (each 3 B
   img probe pulls a whole sector). ncu is the ground-truth follow-up.
2. measured peak — a saturating device-to-device grid-stride copy, timed
   with CUDA events, run once per benchmark session. The operational
   reference every model figure is expressed against ("% of peak"). Never
   compare against a spec-sheet number.
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda

MODEL_NOTE = (
    "model_bytes = processed*(4 queue-read + 3 img-recolor + 4 depth-write"
    " [+ 2 owner-write if instrumented]) + probe_reads*3 img-read"
    " (probe_reads defaults to processed*n_dirs, n_dirs = connectivity; the"
    " radius-2 twins pass the exact 8*processed + 16*interior) +"
    " cas_attempts*8 (visited CAS RMW) + (filled-1)*4 (enqueue writes)."
    " Derived lower-bound model of algorithmic traffic: ~61 B/pixel on"
    " solid interiors at 4-conn, ~105 B/pixel at 8-conn (double the probes,"
    " roughly double the interior CAS attempts). NOT modeled: the radius-2"
    " guard's visited loads on non-red ring-1 neighbors (up to 8x4 B per"
    " processed pixel, zero on solid interiors where all ring-1 is red)."
    " L2 caching deflates real DRAM bytes, 32B sectors inflate them."
    " Compare only against the measured copy peak; ncu is ground truth."
)


def model_bytes(processed, cas_attempts, filled, instrumented, n_dirs=4,
                probe_reads=None):
    """Algorithmic bytes moved, from the kernel's own exactly-once counters.

    probe_reads: exact neighbor-probe count when the kernel's probes are not
    a fixed n_dirs per pixel (the radius-2 twins); None means
    processed * n_dirs.
    """
    per_dequeue = 4 + 3 + 4 + (2 if instrumented else 0)
    if probe_reads is None:
        probe_reads = processed * n_dirs
    return (processed * per_dequeue
            + probe_reads * 3             # neighbor is_red probes x 3B read
            + cas_attempts * 8            # visited int32 CAS read+write
            + max(filled - 1, 0) * 4)     # enqueue writes (seed set by host)


def model_gb_s(nbytes, kernel_ms):
    return nbytes / (kernel_ms * 1e6) if kernel_ms > 0 else 0.0


@cuda.jit
def _copy_kernel(dst, src):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for j in range(i, src.shape[0], stride):
        dst[j] = src[j]


def measure_peak_bandwidth(n_bytes=256 * 2 ** 20, repeats=10):
    """Measured D2D copy peak: median GB/s over `repeats` timed copies.

    Two n_bytes buffers of int64 (8 B coalesced accesses), [1024, 256]
    grid-stride launch (far oversubscribed — pure bandwidth), CUDA-event
    timing. GB/s counts read + write = 2*n_bytes per copy. The buffers are
    freed before returning so scene benchmarks get the VRAM back.
    """
    n = n_bytes // 8
    src = cuda.device_array(n, dtype=np.int64)   # contents irrelevant
    dst = cuda.device_array(n, dtype=np.int64)
    _copy_kernel[1024, 256](dst, src)            # JIT + cache warmup
    cuda.synchronize()
    runs = []
    for _ in range(repeats):
        e0 = cuda.event()
        e1 = cuda.event()
        e0.record()
        _copy_kernel[1024, 256](dst, src)
        e1.record()
        e1.synchronize()
        ms = cuda.event_elapsed_time(e0, e1)
        runs.append(2 * n_bytes / (ms * 1e6))
    del src, dst
    return {
        "gb_s": float(np.median(runs)),
        "runs_gb_s": [float(r) for r in runs],
        "n_bytes": int(n_bytes),
    }
