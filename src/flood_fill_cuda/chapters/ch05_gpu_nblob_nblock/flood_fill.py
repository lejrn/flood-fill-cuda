"""Host driver for the seed-discovery flood fill.

Public API:
    flood_fill(img, variant="ccl_fill", threads_per_block=256,
               blocks=None, bare=False) -> SeedDiscoveryResult
    max_blocks(variant="ccl_fill", threads_per_block=256, bare=False)

THE headline API change of this chapter: there is no seeds parameter.
The caller hands over an image; the GPU finds every blob, picks each
blob's canonical seed (its minimum-linear-index pixel), fills, labels
and paints them all in ONE cooperative launch. The result reports the
seeds the GPU chose — requirement "one seed per blob" made inspectable.

variant picks the discovery strategy (see kernels.py for the bet each
one makes): "ccl_fill" solves connectivity in a union-find prepass and
fills from one seed per blob; "seed_merge" (the colliding-waves family)
fills from every locally-detectable candidate and merges labels in
flight.

blocks=None launches the maximum cooperative grid (queried per compiled
kernel — never assumed equal across variants/twins); bare=True selects
the uninstrumented twins (timing/filled/levels/label only; zeroed work
metrics).

A blank image is a VALID input (n_blobs=0, filled=0, seeds=[]) — with
no host-side seed there is nothing to validate against, and "how many
blobs are there?" is precisely the question the kernel answers.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ...shared.bandwidth import model_gb_s as _model_gb_s
from .kernels import (
    ccl_fill_kernel, ccl_fill_bare_kernel,
    PALETTE_HOST, N_PALETTE,
    NUM_COUNTERS,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS,
    CANDIDATES, UNION_ATTEMPTS, UNION_DONE,
    BS_PROCESSED, BS_SMID,
)
from numba import cuda

VARIANTS = ("ccl_fill",)

# (variant, bare) -> kernel
_KERNELS = {
    ("ccl_fill", False): ccl_fill_kernel,
    ("ccl_fill", True): ccl_fill_bare_kernel,
}

# Cap on the recorded per-level trace (1D int32 -> 8 MB max)
LEVEL_TRACE_CAPACITY = 2 ** 21

MODEL_NOTE = (
    "ch05 model_bytes = fill traffic [processed*(4 queue-read + 3 recolor"
    " + 4 depth-write [+2 owner]) + processed*8*3 probe reads +"
    " cas_attempts*8 + filled*4 enqueue writes (ALL enqueues are"
    " device-side here, seeds included)] + discovery traffic [n*4 parent"
    " iota + n*3 discovery is_red sweep + filled*4*3 lex-predecessor"
    " probes + union_attempts*8 parent RMW + n*3 flatten re-sweep"
    " (ccl_fill) or candidates*4 pre-visit writes + n*4 visited re-sweep"
    " (seed_merge)] + label traffic [filled*4 label write + filled*4"
    " label read + filled*4 flatten find-start reads] — the bytes ch04's"
    " in-entry labels claimed for free, now priced. Derived lower-bound"
    " model (find-chain reads beyond the first are not modeled; L2"
    " deflates, 32B sectors inflate). Compare only against the measured"
    " copy peak; ncu is ground truth."
)


def model_bytes_ch05(variant, n_pixels, filled, processed, cas_attempts,
                     union_attempts, candidates, instrumented):
    """Algorithmic bytes moved, from the kernel's own counters — the
    shared single-blob fill model plus this chapter's discovery and
    label_map terms (see MODEL_NOTE for the itemization)."""
    per_dequeue = 4 + 3 + 4 + (2 if instrumented else 0)
    fill = (processed * per_dequeue
            + processed * 8 * 3          # neighbor is_red probes x 3B
            + cas_attempts * 8           # visited int32 CAS read+write
            + filled * 4)                # enqueue writes (all device-side)
    discovery = (n_pixels * 4            # parent iota (P0)
                 + n_pixels * 3          # discovery is_red sweep
                 + filled * 4 * 3        # lex-predecessor probes (red only)
                 + union_attempts * 8)   # parent atomic.min RMW
    if variant == "ccl_fill":
        discovery += n_pixels * 3        # P2 flatten re-sweeps is_red
    else:
        discovery += candidates * 4 + n_pixels * 4  # pre-visits; P3 visited sweep
    labels = filled * 4 * 3              # label write + read + find-start read
    return fill + discovery + labels


@dataclass
class SeedDiscoveryResult:
    img: np.ndarray        # (w, h, 3) uint8, painted palette[label % 6]
    visited: np.ndarray    # (w, h) int32, 1 where reached
    depth: np.ndarray      # (w, h) int32; ccl_fill: distance from the
                           # canonical seed; seed_merge: from the nearest
                           # candidate. -1 off-blob.
    label: np.ndarray      # (w, h) int32 canonical label map (min linear
                           # index per blob), -1 off-blob
    prov_label: np.ndarray  # (w, h) int32 pre-merge labels (seed_merge
                            # instrumented only; empty otherwise)
    owner: np.ndarray      # (w, h) int16, block id per pixel; empty for bare
    variant: str
    threads_per_block: int
    blocks: int
    bare: bool
    # What the GPU discovered
    n_blobs: int           # distinct labels on visited pixels
    seeds: list            # the canonical seed (x, y) per blob, sorted by
                           # label — exactly one per blob
    candidates: int        # queue entries after discovery (ccl_fill:
                           # n_blobs; seed_merge: candidate count); 0 bare
    union_attempts: int
    union_done: int        # successful links == initial_roots - n_blobs
    # Fill outcome
    filled: int
    levels: int            # fill-phase level count (the shared clock)
    peak_level: int
    peak_occupancy: int
    processed: int
    cas_attempts: int
    thread_util_pct: float
    # Timing (ms)
    kernel_ms: float
    alloc_ms: float
    h2d_ms: float
    d2h_ms: float
    total_ms: float
    # Derived bandwidth model
    model_bytes: int
    model_gb_s: float
    processed_per_block: np.ndarray = field(repr=False, default=None)
    sm_ids: list = field(repr=False, default=None)
    distinct_sms: int = 0
    level_sizes: np.ndarray = field(repr=False, default=None)
    level_trace_truncated: bool = False


_warmed = set()
_coop_cache = {}


def _tiny_scene():
    """Two isolated red pixels on 8x8 white — two blobs to discover, no
    growth, so a compile launch terminates under any block count and
    exercises every phase including the fence-sandwich rear read."""
    tiny = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny[1, 1] = (255, 0, 0)
    tiny[5, 5] = (255, 0, 0)
    return tiny


def _device_buffers(img_host, instrumented, launch_blocks, trace_capacity):
    width, height = img_host.shape[0], img_host.shape[1]
    n = width * height
    bufs = {
        "img": cuda.to_device(img_host),
        "visited": cuda.to_device(np.zeros((width, height), dtype=np.int32)),
        "depth": cuda.to_device(np.full((width, height), -1, dtype=np.int32)),
        "label": cuda.to_device(np.full((width, height), -1, dtype=np.int32)),
        "parent": cuda.device_array(n, dtype=np.int32),  # P0 initializes
        "queue": cuda.device_array(n, dtype=np.int32),
        "q_state": cuda.to_device(np.array([0], dtype=np.int32)),
        "counters": cuda.to_device(np.zeros(NUM_COUNTERS, dtype=np.int64)),
    }
    if instrumented:
        stats_host = np.zeros((launch_blocks, 2), dtype=np.int64)
        stats_host[:, BS_SMID] = -1
        bufs["owner"] = cuda.to_device(
            np.full((width, height), -1, dtype=np.int16))
        bufs["stats"] = cuda.to_device(stats_host)
        bufs["trace"] = cuda.device_array(trace_capacity, dtype=np.int32)
    return bufs


def _kernel_args(variant, bare, bufs):
    if bare:
        return (bufs["img"], bufs["visited"], bufs["depth"], bufs["parent"],
                bufs["label"], bufs["queue"], bufs["q_state"],
                bufs["counters"])
    return (bufs["img"], bufs["visited"], bufs["depth"], bufs["owner"],
            bufs["parent"], bufs["label"], bufs["queue"], bufs["q_state"],
            bufs["counters"], bufs["stats"], bufs["trace"])


def _warmup(variant, bare):
    """JIT-compile (and NVRTC-link) each kernel once, off the clock."""
    key = (variant, bare)
    if key in _warmed:
        return
    bufs = _device_buffers(_tiny_scene(), not bare, 1, 4)
    _KERNELS[key][1, 32](*_kernel_args(variant, bare, bufs))
    cuda.synchronize()
    _warmed.add(key)


def _coop_max_blocks(kernel_fn, tpb):
    key = (id(kernel_fn), tpb)
    if key not in _coop_cache:
        overload = next(iter(kernel_fn.overloads.values()))
        _coop_cache[key] = overload.max_cooperative_grid_blocks(tpb)
    return _coop_cache[key]


def max_blocks(variant="ccl_fill", threads_per_block=256, bare=False):
    """The largest cooperative grid this GPU can host at threads_per_block.
    Queried per kernel — never assumed equal across variants/twins (the
    union-find loops change register pressure)."""
    _warmup(variant, bare)
    return _coop_max_blocks(_KERNELS[(variant, bare)], threads_per_block)


def flood_fill(img_host, variant="ccl_fill", threads_per_block=256,
               blocks=None, bare=False):
    """Discover, label and flood-fill every red blob — no seeds taken.

    img_host: (width, height, 3) uint8. Not modified; a painted copy is
    returned. Raises ValueError for bad inputs and RuntimeError if the
    GPU cannot host the requested cooperative launch (or a structural
    tripwire fires).
    """
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
    # 512 cap: ~104 regs/thread — a 1024-thread block could never be resident
    if threads_per_block % 32 != 0 or not (32 <= threads_per_block <= 512):
        raise ValueError(
            f"threads_per_block must be a multiple of 32 in [32, 512], "
            f"got {threads_per_block}")
    if blocks is not None:
        if not isinstance(blocks, (int, np.integer)) or isinstance(blocks, bool):
            raise ValueError(f"blocks must be an int or None, got {blocks!r}")
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")

    _warmup(variant, bare)
    kernel_fn = _KERNELS[(variant, bare)]
    coop_max = _coop_max_blocks(kernel_fn, threads_per_block)
    if blocks is None:
        launch_blocks = coop_max
    elif blocks > coop_max:
        raise RuntimeError(
            f"blocks={blocks} exceeds this GPU's cooperative-launch capacity "
            f"of {coop_max} blocks at {threads_per_block} threads "
            f"(grid.sync would deadlock)")
    else:
        launch_blocks = int(blocks)

    instrumented = not bare
    trace_capacity = min(n, LEVEL_TRACE_CAPACITY)

    t_total0 = time.perf_counter()
    bufs = _device_buffers(img_host, instrumented, launch_blocks,
                           trace_capacity)
    cuda.synchronize()
    t_kernel0 = time.perf_counter()

    kernel_fn[launch_blocks, threads_per_block](
        *_kernel_args(variant, bare, bufs))
    cuda.synchronize()
    t_d2h0 = time.perf_counter()
    kernel_ms = (t_d2h0 - t_kernel0) * 1000

    counters = bufs["counters"].copy_to_host()
    if counters[OVERFLOW]:
        raise RuntimeError(
            "structural tripwire fired — this indicates a kernel bug: "
            "the queue cannot legitimately overflow")
    img_out = bufs["img"].copy_to_host()
    visited_out = bufs["visited"].copy_to_host()
    depth_out = bufs["depth"].copy_to_host()
    label_out = bufs["label"].copy_to_host()
    levels = int(counters[LEVELS])
    if instrumented:
        owner_out = bufs["owner"].copy_to_host()
        stats = bufs["stats"].copy_to_host()
        trace = bufs["trace"][:min(levels, trace_capacity)].copy_to_host()
    else:
        owner_out = np.zeros((0, 0), dtype=np.int16)
        stats = None
        trace = np.zeros(0, dtype=np.int32)
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
        prov_label=np.zeros((0, 0), dtype=np.int32),
        owner=owner_out,
        variant=variant,
        threads_per_block=threads_per_block,
        blocks=launch_blocks,
        bare=bare,
        n_blobs=n_blobs,
        seeds=seeds,
        candidates=int(counters[CANDIDATES]),
        union_attempts=int(counters[UNION_ATTEMPTS]),
        union_done=int(counters[UNION_DONE]),
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
