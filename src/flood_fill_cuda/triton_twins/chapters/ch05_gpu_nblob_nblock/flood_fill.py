"""Host driver for the Triton twin of the seed-discovery flood fill.

Public API (the Numba driver's, unchanged):
    flood_fill(img, variant="seed_merge", threads_per_block=256,
               blocks=None, bare=False, lattice=None, interior=False,
               build="fused") -> SeedDiscoveryResult
    max_blocks(variant="seed_merge", threads_per_block=256, bare=False,
               lattice=None, build="fused")
    discovery_only(img, variant="seed_merge", threads_per_block=256,
                   blocks=None) -> (kernel_ms, candidates)

Same defaults, validation messages, result dataclass (imported from the
Numba driver, so the field names cannot drift), model_bytes_ch05 and
timing brackets as chapters/ch05_gpu_nblob_nblock/flood_fill.py: read
its docstring for what the chapter does. What differs is the launch:

- CuPy arrays replace Numba device arrays, uploaded the same way (host
  arrays copied H2D inside the alloc bracket; uninitialized buffers are
  cp.empty), plus two Triton-only buffers: the grid barrier counter and
  the palette (Numba bakes it into a const array).
- threads_per_block must also be a power of 2 (BLOCK lanes, num_warps =
  threads_per_block // 32); Numba accepts any multiple of 32.
- num_warps is compile-time in Triton, so the warm-up compiles once per
  (kernel, bare, threads_per_block) at the real block size, off the
  clock. Every runtime int is do_not_specialize, so no scene size can
  trigger a compile inside kernel_ms.
- blocks=None launches max_coresident_programs of the compiled twin (the
  counterpart of Numba's max_cooperative_grid_blocks): its own register
  count, so its own number, recorded in result.blocks.
- build="r128" launches the fused lattice twin with maxnreg=128 (Numba's
  max_registers=128); build="split" launches the cooperative core, then
  lat_compress_kernel and lat_finish_kernel on _PLAIN_GRID (256 programs
  of 256 lanes), timed with CuPy CUDA events where Numba uses cuda.event.
- The image is checked with cuda.to_device's own contiguity rule (Numba's
  sentry_contiguous, same ValueError), then uploaded in C order: the
  kernels index img[x, y, c] as C-contiguous bytes, so an F-ordered image
  is copied to C order on the host (Numba uploads it as is and indexes
  through its strides; the results are equal).
- Twin-only guard: the grid-stride indices are int32 (Numba's are int64),
  so width*height + blocks*threads_per_block must stay below 2**31 (the
  split build also counts its _PLAIN_GRID). Unreachable on an 8 GB GPU,
  where such an image does not fit in device memory.
"""

import os

# Must be set before numba is imported (the Numba chapter modules below)
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np
# cuda.to_device's host-side contiguity rule and message (no device code)
from numba.cuda.cudadrv.devicearray import sentry_contiguous

from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.flood_fill import (
    BUILDS, LEVEL_TRACE_CAPACITY, MODEL_NOTE, VARIANTS, SeedDiscoveryResult,
    model_bytes_ch05, _PHASE_KEYS, _PLAIN_GRID, _kernel_key, _tiny_scene,
)
from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.kernels import (
    PALETTE_HOST, N_PALETTE, NUM_COUNTERS, N_PHASES,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS,
    CANDIDATES, UNION_ATTEMPTS, UNION_DONE, UNION_CYCLES,
    BS_PROCESSED, BS_SMID,
)
from flood_fill_cuda.shared.bandwidth import model_gb_s as _model_gb_s
from flood_fill_cuda.triton_twins.runtime import (
    kernel_resources, max_coresident_programs, sync, t,
)

from .kernels import (
    ccl_fill_kernel, ccl_kernel, lat_compress_kernel, lat_finish_kernel,
    seed_merge_kernel, seed_merge_lat_core_kernel, seed_merge_lat_kernel,
    seed_scan_kernel,
)

__all__ = [
    "BUILDS", "LEVEL_TRACE_CAPACITY", "MODEL_NOTE", "VARIANTS",
    "SeedDiscoveryResult", "model_bytes_ch05", "flood_fill", "max_blocks",
    "discovery_only", "regs_per_thread", "compiled_kernel", "kernel_info",
]


@dataclass(frozen=True)
class _KernelSpec:
    """One Numba kernel's Triton twin: the @triton.jit body, the
    constexprs that select it (INSTR=False is the bare twin), whether it
    holds grid barriers (cooperative launch), and extra launch options
    (e.g. maxnreg, the twin of Numba's max_registers)."""
    fn: object
    constexprs: dict = field(default_factory=dict)
    cooperative: bool = True
    options: dict = field(default_factory=dict)


# (kernel_key, bare) -> twin. Keys are the Numba driver's _KERNELS keys.
# r128 is the fused lattice body with maxnreg=128, as Numba compiles the
# same py_func with max_registers=128.
_KERNELS = {
    ("seed_merge", False): _KernelSpec(seed_merge_kernel, {"INSTR": True}),
    ("seed_merge", True): _KernelSpec(seed_merge_kernel, {"INSTR": False}),
    ("seed_merge_lat", False): _KernelSpec(seed_merge_lat_kernel,
                                           {"INSTR": True}),
    ("seed_merge_lat", True): _KernelSpec(seed_merge_lat_kernel,
                                          {"INSTR": False}),
    ("seed_merge_lat_r128", False): _KernelSpec(
        seed_merge_lat_kernel, {"INSTR": True}, options={"maxnreg": 128}),
    ("seed_merge_lat_core", False): _KernelSpec(seed_merge_lat_core_kernel),
    ("ccl_fill", False): _KernelSpec(ccl_fill_kernel, {"INSTR": True}),
    ("ccl_fill", True): _KernelSpec(ccl_fill_kernel, {"INSTR": False}),
}

# variant -> discovery-only phase kernel (benchmark attribution)
_PHASE_KERNELS = {
    "seed_merge": _KernelSpec(seed_scan_kernel),
    "ccl_fill": _KernelSpec(ccl_kernel),
}

# kernel_key -> fn(bufs, threads_per_block): extra launches a warm-up must
# compile besides the cooperative kernel (the split build's two plain
# cleanup kernels, registered below _launch_plain).
_EXTRA_WARMUPS = {}

# (kernel_key, bare, tpb) -> CompiledKernel; ("phase", variant, tpb) and
# ("plain", name) (the split build's cleanup kernels) too
_compiled = {}
_coop_cache = {}
_palette_dev = None


def _palette():
    """PALETTE_HOST on the device (the Numba kernels' const array)."""
    global _palette_dev
    if _palette_dev is None:
        _palette_dev = cp.asarray(np.ascontiguousarray(PALETTE_HOST).ravel())
    return _palette_dev


def _check_tpb_pow2(threads_per_block):
    """Triton's extra rule: a program's lane count is a power of 2."""
    tpb = threads_per_block
    if tpb < 32 or tpb & (tpb - 1):
        raise ValueError(
            f"threads_per_block must be a power of 2 (>= 32) for the Triton "
            f"twin: a program has BLOCK = threads_per_block lanes and "
            f"num_warps = threads_per_block // 32, and Triton tensor sizes "
            f"and num_warps are powers of 2; got {threads_per_block}")


def _check_int32_grid_stride(n, grid_threads):
    """Triton's extra rule: every grid-stride loop runs on int32 indices
    (base + pid * BLOCK + lane, and the loop counter itself), which wrap
    past 2**31 - 1. Numba's range loops are int64. So the last stride
    must still fit: n + grid_threads < 2**31."""
    if n + grid_threads >= 2 ** 31:
        raise ValueError(
            f"image too large for the Triton twin: width*height + "
            f"blocks*threads_per_block must be < 2**31 (its grid-stride "
            f"indices are int32), got {n} + {grid_threads}")


def _upload_img(img_host):
    """cuda.to_device(img_host)'s contiguity check (same ValueError), then
    a C-order upload: the kernels index img as C-contiguous bytes."""
    sentry_contiguous(img_host)
    return cp.asarray(np.ascontiguousarray(img_host))


def _spec(kernel_key, bare):
    try:
        return _KERNELS[(kernel_key, bare)]
    except KeyError:
        raise NotImplementedError(
            f"the Triton twin of kernel {kernel_key!r} (bare={bare}) is not "
            "registered in this driver yet") from None


def _launch(spec, grid, args, tpb):
    return spec.fn[(grid,)](
        *args, BLOCK=tpb, num_warps=tpb // 32, num_stages=1,
        launch_cooperative_grid=spec.cooperative, **spec.constexprs,
        **spec.options)


def _launch_plain(fn, args, grid=_PLAIN_GRID[0]):
    """A split-build cleanup kernel on the Numba driver's _PLAIN_GRID:
    (256 blocks, 256 threads) -> 256 programs of 256 lanes, no
    cooperative launch (no barrier inside)."""
    tpb = _PLAIN_GRID[1]
    return fn[(grid,)](*args, BLOCK=tpb, num_warps=tpb // 32, num_stages=1)


def _split_cleanup_args(bufs):
    n = int(bufs["parent"].shape[0])
    return ((t(bufs["parent"]), n),
            (t(bufs["img"]), t(bufs["parent"]), t(bufs["label"]),
             t(bufs["prov"]), t(_palette()), n))


def _warm_split_cleanup(bufs, threads_per_block):
    """The Numba warm-up's lat_compress / lat_finish [1, 32] launches: one
    program each at the real _PLAIN_GRID block size (num_warps is
    compile-time), after the core's warm-up launch on the same stream."""
    compress_args, finish_args = _split_cleanup_args(bufs)
    _compiled[("plain", "lat_compress")] = _launch_plain(
        lat_compress_kernel, compress_args, grid=1)
    _compiled[("plain", "lat_finish")] = _launch_plain(
        lat_finish_kernel, finish_args, grid=1)


_EXTRA_WARMUPS["seed_merge_lat_core"] = _warm_split_cleanup


def _device_buffers(img_host, variant, instrumented, launch_blocks,
                    trace_capacity):
    width, height = img_host.shape[0], img_host.shape[1]
    n = width * height
    bufs = {
        "img": _upload_img(img_host),
        "visited": cp.asarray(np.zeros((width, height), dtype=np.int32)),
        "depth": cp.asarray(np.full((width, height), -1, dtype=np.int32)),
        "label": cp.asarray(np.full((width, height), -1, dtype=np.int32)),
        "parent": cp.empty(n, dtype=np.int32),  # P0 initializes
        "queue": cp.empty(n, dtype=np.int32),
        "q_state": cp.asarray(np.array([0], dtype=np.int32)),
        "counters": cp.asarray(np.zeros(NUM_COUNTERS, dtype=np.int64)),
        # Triton-only: the grid barrier's monotonic arrival counter
        "bar": cp.asarray(np.zeros(1, dtype=np.int64)),
    }
    if instrumented:
        stats_host = np.zeros((launch_blocks, 2), dtype=np.int64)
        stats_host[:, BS_SMID] = -1
        bufs["owner"] = cp.asarray(
            np.full((width, height), -1, dtype=np.int16))
        bufs["stats"] = cp.asarray(stats_host)
        bufs["trace"] = cp.empty(trace_capacity, dtype=np.int32)
        bufs["phase"] = cp.asarray(np.zeros(N_PHASES, dtype=np.int64))
        if variant != "ccl_fill":
            bufs["prov"] = cp.asarray(
                np.full((width, height), -1, dtype=np.int32))
    return bufs


def _ptr(bufs, name):
    """A buffer as a Triton pointer, or None where the twin has none (the
    bare twins' instrumentation slots)."""
    arr = bufs.get(name)
    return None if arr is None else t(arr)


# Pointer arguments per kernel_key, in the Numba instrumented kernel's
# order (prov_label only where the Numba kernel has it: the split core
# has none, lat_finish_kernel writes it)
_SEED_MERGE_POINTERS = ("img", "visited", "depth", "owner", "parent",
                        "label", "prov", "queue", "q_state", "counters",
                        "stats", "trace", "phase")
_POINTERS = {
    "ccl_fill": ("img", "visited", "depth", "owner", "parent", "label",
                 "queue", "q_state", "counters", "stats", "trace", "phase"),
    "seed_merge": _SEED_MERGE_POINTERS,
    "seed_merge_lat": _SEED_MERGE_POINTERS,
    "seed_merge_lat_r128": _SEED_MERGE_POINTERS,
    "seed_merge_lat_core": ("img", "visited", "depth", "owner", "parent",
                            "label", "queue", "q_state", "counters", "stats",
                            "trace", "phase"),
}


def _kernel_args(kernel_key, bare, bufs, lattice=None, interior=False):
    width, height = bufs["img"].shape[0], bufs["img"].shape[1]
    trace = bufs.get("trace")
    # the split core paints nothing (its Numba kernel has no palette)
    palette = () if kernel_key == "seed_merge_lat_core" else (t(_palette()),)
    args = tuple(_ptr(bufs, k) for k in _POINTERS[kernel_key]) + palette + (
        t(bufs["bar"]),
        width, height, width * height,
        int(bufs["queue"].shape[0]),               # queue capacity
        0 if trace is None else int(trace.shape[0]),  # trace capacity
    )
    if kernel_key.startswith("seed_merge_lat"):
        args = args + (int(lattice), 1 if interior else 0)
    return args


def _warmup(kernel_key, bare, threads_per_block):
    """Compile each twin once per block size, off the clock, with a
    one-program launch on _tiny_scene (two isolated red pixels: every
    phase including the fence-sandwich rear read runs, nothing grows)."""
    key = (kernel_key, bare, threads_per_block)
    if key in _compiled:
        return _compiled[key]
    spec = _spec(kernel_key, bare)
    bufs = _device_buffers(_tiny_scene(), kernel_key, not bare, 1, 4)
    compiled = _launch(spec, 1, _kernel_args(kernel_key, bare, bufs,
                                             lattice=2), threads_per_block)
    extra = _EXTRA_WARMUPS.get(kernel_key)
    if extra is not None:
        extra(bufs, threads_per_block)
    sync()
    _compiled[key] = compiled
    return compiled


def _coop_max_blocks(key, compiled):
    if key not in _coop_cache:
        _coop_cache[key] = max_coresident_programs(compiled)
    return _coop_cache[key]


_clock_rate_hz = None


def _cycles_to_ms(cycles):
    """%clock64 cycles -> aggregate thread-milliseconds via the device's
    BASE clock rate (the Numba driver's conversion, same caveats)."""
    global _clock_rate_hz
    if cycles == 0:
        return 0.0
    if _clock_rate_hz is None:
        try:
            _clock_rate_hz = cp.cuda.Device().attributes["ClockRate"] * 1000
        except Exception:
            _clock_rate_hz = 0
    return cycles / _clock_rate_hz * 1000 if _clock_rate_hz else 0.0


def compiled_kernel(variant="seed_merge", threads_per_block=256, bare=False,
                    lattice=None, build="fused"):
    """The warmed CompiledKernel flood_fill launches for this selection."""
    _check_tpb_pow2(threads_per_block)
    return _warmup(_kernel_key(variant, lattice, build), bare,
                   threads_per_block)


def regs_per_thread(compiled):
    """Registers/thread of a compiled twin (the Numba driver's
    regs_per_thread takes a dispatcher; a Triton kernel has one compiled
    object per block size, so this takes that object), or None."""
    try:
        return int(kernel_resources(compiled)["n_regs"])
    except Exception:
        return None


def kernel_info(variant="seed_merge", threads_per_block=256, bare=False,
                lattice=None, build="fused"):
    """Registers, spills, shared bytes, warps and cooperative capacity of
    the selected twin: the facts benchmarks record next to the timings.
    The split build adds its two plain cleanup kernels and their grid."""
    compiled = compiled_kernel(variant, threads_per_block, bare, lattice,
                               build)
    info = kernel_resources(compiled)
    info["coop_max_blocks"] = max_blocks(variant, threads_per_block, bare,
                                         lattice, build)
    if _kernel_key(variant, lattice, build) == "seed_merge_lat_core":
        info["cleanup"] = {
            name: kernel_resources(_compiled[("plain", name)])
            for name in ("lat_compress", "lat_finish")}
        info["cleanup_grid"] = list(_PLAIN_GRID)
    return info


def max_blocks(variant="seed_merge", threads_per_block=256, bare=False,
               lattice=None, build="fused"):
    """The largest cooperative grid this GPU can host at threads_per_block,
    queried per compiled twin (never assumed equal across variants)."""
    _check_tpb_pow2(threads_per_block)
    key = _kernel_key(variant, lattice, build)
    compiled = _warmup(key, bare, threads_per_block)
    return _coop_max_blocks((key, bare, threads_per_block), compiled)


def flood_fill(img_host, variant="seed_merge", threads_per_block=256,
               blocks=None, bare=False, lattice=None, interior=False,
               build="fused"):
    """Discover, label and flood-fill every red blob - no seeds taken.

    The Numba driver's contract (see its docstring for lattice, interior
    and build); raises ValueError for bad inputs and RuntimeError if the
    GPU cannot host the requested cooperative launch (or a structural
    tripwire fires)."""
    if img_host.ndim != 3 or img_host.shape[2] != 3 or img_host.dtype != np.uint8:
        raise ValueError("img must be a (width, height, 3) uint8 array")
    width, height = img_host.shape[0], img_host.shape[1]
    n = width * height
    if n >= 2 ** 31:
        raise ValueError(
            "image too large: width*height must be < 2**31 (labels and "
            "queue entries are int32 linear indices)")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 512):
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 512], "
            f"got {threads_per_block}")
    _check_tpb_pow2(threads_per_block)
    if blocks is not None:
        if not isinstance(blocks, (int, np.integer)) or isinstance(blocks, bool):
            raise ValueError(f"blocks must be an int or None, got {blocks!r}")
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")
    if lattice is not None:
        if variant != "seed_merge":
            raise ValueError(
                "lattice seeding only applies to variant='seed_merge' "
                f"(got variant={variant!r})")
        if (not isinstance(lattice, (int, np.integer))
                or isinstance(lattice, bool) or lattice < 0):
            raise ValueError(
                f"lattice must be an int >= 0 or None, got {lattice!r}")
    if interior and (lattice is None or lattice < 1):
        raise ValueError(
            "interior=True requires lattice >= 1 (the interior test "
            "filters lattice hits; without a lattice there is nothing "
            "to filter)")
    if build not in BUILDS:
        raise ValueError(f"build must be one of {BUILDS}, got {build!r}")
    if build != "fused":
        if lattice is None:
            raise ValueError(
                f"build={build!r} is a lattice-mode experiment "
                "(requires lattice is not None)")
        if bare:
            raise ValueError(
                f"build={build!r} is instrumented-only (no bare twin)")

    kernel_key = _kernel_key(variant, lattice, build)
    spec = _spec(kernel_key, bare)
    compiled = _warmup(kernel_key, bare, threads_per_block)
    coop_max = _coop_max_blocks((kernel_key, bare, threads_per_block),
                                compiled)
    if blocks is None:
        launch_blocks = coop_max
    elif blocks > coop_max:
        raise RuntimeError(
            f"blocks={blocks} exceeds this GPU's cooperative-launch capacity "
            f"of {coop_max} blocks at {threads_per_block} threads "
            f"(grid.sync would deadlock)")
    else:
        launch_blocks = int(blocks)
    widest_stride = launch_blocks * threads_per_block
    if kernel_key == "seed_merge_lat_core":
        widest_stride = max(widest_stride, _PLAIN_GRID[0] * _PLAIN_GRID[1])
    _check_int32_grid_stride(n, widest_stride)

    instrumented = not bare
    trace_capacity = min(n, LEVEL_TRACE_CAPACITY)

    t_total0 = time.perf_counter()
    bufs = _device_buffers(img_host, kernel_key, instrumented, launch_blocks,
                           trace_capacity)
    sync()
    t_kernel0 = time.perf_counter()

    compress_ms = flatten_ms = None
    if kernel_key == "seed_merge_lat_core":
        # split build: cooperative core, then the two plain cleanup
        # kernels - stream order IS the compress-before-flatten barrier;
        # CUDA events attribute their times without extra syncs
        ev = [cp.cuda.Event() for _ in range(3)]
        _launch(spec, launch_blocks,
                _kernel_args(kernel_key, bare, bufs, lattice, interior),
                threads_per_block)
        ev[0].record()
        compress_args, finish_args = _split_cleanup_args(bufs)
        _launch_plain(lat_compress_kernel, compress_args)
        ev[1].record()
        _launch_plain(lat_finish_kernel, finish_args)
        ev[2].record()
        sync()
        compress_ms = cp.cuda.get_elapsed_time(ev[0], ev[1])
        flatten_ms = cp.cuda.get_elapsed_time(ev[1], ev[2])
    else:
        _launch(spec, launch_blocks,
                _kernel_args(kernel_key, bare, bufs, lattice, interior),
                threads_per_block)
        sync()
    t_d2h0 = time.perf_counter()
    kernel_ms = (t_d2h0 - t_kernel0) * 1000

    counters = cp.asnumpy(bufs["counters"])
    if counters[OVERFLOW]:
        raise RuntimeError(
            "structural tripwire fired - this indicates a kernel bug: "
            "the queue cannot legitimately overflow")
    img_out = cp.asnumpy(bufs["img"])
    visited_out = cp.asnumpy(bufs["visited"])
    depth_out = cp.asnumpy(bufs["depth"])
    label_out = cp.asnumpy(bufs["label"])
    levels = int(counters[LEVELS])
    if instrumented:
        owner_out = cp.asnumpy(bufs["owner"])
        stats = cp.asnumpy(bufs["stats"])
        trace = cp.asnumpy(bufs["trace"][:min(levels, trace_capacity)])
        prov_out = (cp.asnumpy(bufs["prov"])
                    if kernel_key != "ccl_fill"
                    else np.zeros((0, 0), dtype=np.int32))
        stamps = cp.asnumpy(bufs["phase"])
        phase_ms = {k: max(int(stamps[i + 1] - stamps[i]), 0) / 1e6
                    for i, k in enumerate(_PHASE_KEYS[kernel_key])}
        if compress_ms is not None:
            # split build: the same five keys as fused, so tables line up
            phase_ms["compress"] = float(compress_ms)
            phase_ms["flatten"] = float(flatten_ms)
    else:
        owner_out = np.zeros((0, 0), dtype=np.int16)
        stats = None
        trace = np.zeros(0, dtype=np.int32)
        prov_out = np.zeros((0, 0), dtype=np.int32)
        phase_ms = {}
    t_end = time.perf_counter()

    # What the GPU discovered, made inspectable: exactly one canonical
    # seed per blob, decoded from the surviving labels.
    unique_labels = np.unique(label_out[visited_out == 1])
    n_blobs = int(unique_labels.size)
    seeds = [(int(l) // height, int(l) % height) for l in unique_labels]

    grid_threads = launch_blocks * threads_per_block
    if instrumented:
        ppb = stats[:, BS_PROCESSED].copy()
        sm_ids = [int(s) for s in stats[:, BS_SMID]]
        distinct = len({s for s in sm_ids if s >= 0})
        util = (100.0 * int(counters[ACTIVE_THREAD_SUM])
                / (levels * grid_threads) if levels else 0.0)
        mbytes = model_bytes_ch05(
            variant, n, int(counters[FILLED]), int(counters[PROCESSED]),
            int(counters[CAS_ATTEMPTS]), int(counters[UNION_ATTEMPTS]),
            int(counters[CANDIDATES]), instrumented=True)
    else:
        ppb = np.zeros(0, dtype=np.int64)
        sm_ids, distinct, util = [], 0, 0.0
        mbytes = 0

    return SeedDiscoveryResult(
        img=img_out,
        visited=visited_out,
        depth=depth_out,
        label=label_out,
        prov_label=prov_out,
        owner=owner_out,
        variant=variant,
        lattice=(int(lattice) if lattice is not None else None),
        interior=bool(interior),
        build=build,
        threads_per_block=threads_per_block,
        blocks=launch_blocks,
        bare=bare,
        n_blobs=n_blobs,
        seeds=seeds,
        candidates=int(counters[CANDIDATES]),
        union_attempts=int(counters[UNION_ATTEMPTS]),
        union_done=int(counters[UNION_DONE]),
        phase_ms=phase_ms,
        union_cycles=int(counters[UNION_CYCLES]),
        union_thread_ms=_cycles_to_ms(int(counters[UNION_CYCLES])),
        filled=int(counters[FILLED]),
        levels=levels,
        peak_level=int(counters[PEAK_LEVEL]),
        peak_occupancy=int(counters[PEAK_OCC]),
        processed=int(counters[PROCESSED]),
        cas_attempts=int(counters[CAS_ATTEMPTS]),
        thread_util_pct=util,
        kernel_ms=kernel_ms,
        alloc_ms=(t_kernel0 - t_total0) * 1000,
        h2d_ms=0.0,  # folded into alloc: buffers upload as they allocate
        d2h_ms=(t_end - t_d2h0) * 1000,
        total_ms=(t_end - t_total0) * 1000,
        model_bytes=mbytes,
        model_gb_s=_model_gb_s(mbytes, kernel_ms),
        processed_per_block=ppb,
        sm_ids=sm_ids,
        distinct_sms=distinct,
        level_sizes=trace,
        level_trace_truncated=(instrumented and levels > trace_capacity),
    )


def discovery_only(img_host, variant="seed_merge", threads_per_block=256,
                   blocks=None):
    """Benchmark probe: run ONLY the discovery phases on fresh buffers -
    seed_merge's candidate scan (P0-P1) or ccl_fill's full CCL (P0-P2).
    Returns (kernel_ms, candidates), like the Numba probe."""
    if img_host.ndim != 3 or img_host.shape[2] != 3 or img_host.dtype != np.uint8:
        raise ValueError("img must be a (width, height, 3) uint8 array")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    _check_tpb_pow2(threads_per_block)
    spec = _PHASE_KERNELS[variant]

    def _bufs(img):
        w, h = img.shape[0], img.shape[1]
        return (_upload_img(img),
                cp.asarray(np.zeros((w, h), dtype=np.int32)),
                cp.asarray(np.full((w, h), -1, dtype=np.int32)),
                cp.empty(w * h, dtype=np.int32),
                cp.empty(w * h, dtype=np.int32),
                cp.asarray(np.array([0], dtype=np.int32)),
                cp.asarray(np.zeros(NUM_COUNTERS, dtype=np.int64)),
                cp.asarray(np.zeros(1, dtype=np.int64)))  # grid barrier

    def _args(bufs):
        w, h = bufs[0].shape[0], bufs[0].shape[1]
        return tuple(t(b) for b in bufs) + (w, h, w * h, w * h)

    key = ("phase", variant, threads_per_block)
    if key not in _compiled:
        _compiled[key] = _launch(spec, 1, _args(_bufs(_tiny_scene())),
                                 threads_per_block)
        sync()

    coop_max = _coop_max_blocks(key, _compiled[key])
    launch_blocks = coop_max if blocks is None else int(blocks)
    if launch_blocks > coop_max:
        raise RuntimeError(
            f"blocks={launch_blocks} exceeds cooperative capacity {coop_max}")
    _check_int32_grid_stride(img_host.shape[0] * img_host.shape[1],
                             launch_blocks * threads_per_block)

    bufs = _bufs(img_host)
    args = _args(bufs)
    sync()
    t0 = time.perf_counter()
    _launch(spec, launch_blocks, args, threads_per_block)
    sync()
    kernel_ms = (time.perf_counter() - t0) * 1000
    candidates = int(cp.asnumpy(bufs[6])[CANDIDATES])
    return kernel_ms, candidates
