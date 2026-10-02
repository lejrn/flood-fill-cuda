"""
Triton twin of the single-block BFS flood fill (ch01 kernels.py).

One Triton program plays the one CUDA block: its tensors have BLOCK lanes
and num_warps = BLOCK // 32, so lane i plays thread i. The whole
level-synchronous BFS runs inside that one program, exactly as in the Numba
kernels, with the same ring, the same tickets, the same claim protocol
(bounds -> red check -> claim on visited -> enqueue), the same barriers and
the same counters.

What changes, and why
---------------------
Triton has no user-addressable shared memory, so the 8192-slot ring and the
shared scalars (s_rear, s_overflow, s_spill_rear) live in a small global
scratch region the host allocates per call: `ring` (RING_CAPACITY int32)
and `state` (STATE_SLOTS int32). The ring keeps its capacity, its masking
(ticket & RING_MASK), its window check (ticket - front < RING_CAPACITY) and
its overflow tripwire, so the v1 kernel still refuses the same scenes and
the v2 kernel still spills the same pixel counts. Only the latency story
changes: the ring is served by L1/L2 instead of shared memory, and the
ticket atomics resolve at L2.

cuda.syncthreads() becomes cta_sync() (bar.sync 0), at the same places. A
CTA barrier orders global memory among the threads of the block, which is
what makes the global ring correct. The shared scalars are re-read after
the barrier with volatile loads (the twin of a shared-memory read).

The visited claim is a masked atomic exchange to 1. Triton 3.7.1's
atomic_cas has no mask; on a 0/1 flag "old == 0 wins" is the same
exactly-once claim as cas(0, 1), and inactive lanes issue nothing.

v1 keeps one ticket atomic per winning lane (not aggregated), as in Numba.
v2's warp-aggregated two-tier enqueue becomes program-aggregated: one
cumsum gives each winner its rank, one scalar atomic per tier reserves the
slab. The spill ranks need no second scan: the lanes that miss the ring
window are the top of the slab in ticket order, so rank2 = rank - k, the
same value Numba's second ballot/popc computes.

Triton has no break: v1's uniform overflow exit is folded into the while
condition, with level += 1 kept before the exit so LEVELS matches.

Per-thread counters (my_processed, my_cas_attempts) stay per lane in
[BLOCK] int64 tensors and are added to the counters with one atomic per
lane at exit, as in Numba. Per-level uniform scalars (front, rear, peaks,
active sums) are program scalars.
"""

import triton
import triton.language as tl

from flood_fill_cuda.chapters.ch01_gpu_1blob_1block.kernels import (
    RING_CAPACITY, RING_MASK,
    FILLED, LEVELS, OVERFLOW, PEAK_LEVEL, PEAK_OCC,
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, PROCESSED, CAS_ATTEMPTS,
    SPILLED, PEAK_SPILL_WINDOW, NUM_COUNTERS,
)
from flood_fill_cuda.triton_twins.runtime.device import cta_sync

# Slots of the per-call `state` scratch, the twin of the shared scalars.
REAR = 0          # s_rear
OVF = 1           # s_overflow (v1)
SPILL_REAR = 2    # s_spill_rear (v2)
STATE_SLOTS = 4

# Triton kernels may only read module globals that are constexpr.
_CAP = tl.constexpr(RING_CAPACITY)
_MASK = tl.constexpr(RING_MASK)
_REAR = tl.constexpr(REAR)
_OVF = tl.constexpr(OVF)
_SPILL_REAR = tl.constexpr(SPILL_REAR)
_FILLED = tl.constexpr(FILLED)
_LEVELS = tl.constexpr(LEVELS)
_OVERFLOW = tl.constexpr(OVERFLOW)
_PEAK_LEVEL = tl.constexpr(PEAK_LEVEL)
_PEAK_OCC = tl.constexpr(PEAK_OCC)
_ACTIVE_THREAD_SUM = tl.constexpr(ACTIVE_THREAD_SUM)
_ACTIVE_WARP_SUM = tl.constexpr(ACTIVE_WARP_SUM)
_PROCESSED = tl.constexpr(PROCESSED)
_CAS_ATTEMPTS = tl.constexpr(CAS_ATTEMPTS)
_SPILLED = tl.constexpr(SPILLED)
_PEAK_SPILL_WINDOW = tl.constexpr(PEAK_SPILL_WINDOW)

# 4-connectivity neighbor offsets (right, down, left, up), as in Numba
_DX = tl.constexpr((1, 0, -1, 0))
_DY = tl.constexpr((0, 1, 0, -1))

# Every runtime int that varies between calls: no recompiles per scene.
_RUNTIME_INTS = ["seed_x", "seed_y", "width", "height", "trace_cap"]


@triton.jit
def _is_red(img_ptr, pixel, mask):
    """Lanes in mask whose pixel is (255, 0, 0); short-circuits like Numba's
    `and` chain (channel 1 is read only where channel 0 is 255, ...)."""
    base = pixel.to(tl.int64) * 3
    r = tl.load(img_ptr + base, mask=mask, other=0).to(tl.int32)
    m1 = mask & (r == 255)
    g = tl.load(img_ptr + base + 1, mask=m1, other=1).to(tl.int32)
    m2 = m1 & (g == 0)
    b = tl.load(img_ptr + base + 2, mask=m2, other=1).to(tl.int32)
    return m2 & (b == 0)


@triton.jit
def _recolor(img_ptr, depth_ptr, pixel, level, mask):
    """img[x, y] = (0, 0, 255); depth[x, y] = level, on the lanes in mask."""
    base = pixel.to(tl.int64) * 3
    tl.store(img_ptr + base, 0, mask=mask)
    tl.store(img_ptr + base + 1, 0, mask=mask)
    tl.store(img_ptr + base + 2, 255, mask=mask)
    tl.store(depth_ptr + pixel, level, mask=mask)


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def single_block_bfs_kernel(img_ptr, visited_ptr, depth_ptr, seed_x, seed_y,
                            counters_ptr, level_sizes_ptr, ring_ptr, state_ptr,
                            width, height, trace_cap, BLOCK: tl.constexpr):
    """v1 "ring": the whole 4-connected BFS inside this one-program launch.

    Host contract (as in Numba): launch with grid (1,), num_warps =
    BLOCK // 32; visited all zeros, depth all -1, counters all zeros. The
    kernel seeds itself and initializes `state`. level_sizes records each
    level's frontier size while level < trace_cap.
    """
    offs = tl.arange(0, BLOCK)
    zero_offs = offs * 0  # every lane addresses the same scalar slot

    seed = seed_x * height + seed_y
    tl.store(ring_ptr, seed)
    tl.store(state_ptr + _REAR, 1)
    tl.store(state_ptr + _OVF, 0)
    tl.store(visited_ptr + seed, 1)
    cta_sync()

    front = 0
    rear = 1
    level = 0
    overflowed = 0
    peak_level = 1
    peak_occ = 1
    active_thread_sum = tl.full([], 0, tl.int64)
    active_warp_sum = tl.full([], 0, tl.int64)
    my_processed = tl.zeros([BLOCK], tl.int64)
    my_cas_attempts = tl.zeros([BLOCK], tl.int64)

    # Numba: while front < rear: ... if overflowed: break
    while (front < rear) & (overflowed == 0):
        level_size = rear - front
        peak_level = tl.maximum(peak_level, level_size)
        active = tl.minimum(level_size, BLOCK)
        active_thread_sum += active.to(tl.int64)
        active_warp_sum += ((active + 31) // 32).to(tl.int64)
        if level < trace_cap:
            tl.store(level_sizes_ptr + level, level_size)

        # Block-stride partition of [front, rear): lane i takes base + i.
        for base in range(front, rear, BLOCK):
            i = base + offs
            m = i < rear
            pixel = tl.load(ring_ptr + (i & _MASK), mask=m, other=0)
            x = pixel // height
            y = pixel % height
            _recolor(img_ptr, depth_ptr, pixel, level, m)
            my_processed += m.to(tl.int64)

            for d in tl.static_range(4):
                nx = x + _DX[d]
                ny = y + _DY[d]
                inb = m & (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
                nidx = nx * height + ny
                cand = _is_red(img_ptr, nidx, inb)
                my_cas_attempts += cand.to(tl.int64)
                # Exactly-once claim: only the lane that flips 0 -> 1 enqueues.
                old = tl.atomic_xchg(visited_ptr + nidx, 1, mask=cand,
                                     sem="relaxed")
                won = cand & (old == 0)
                # One ticket atomic per winning lane, as in Numba v1.
                ticket = tl.atomic_add(state_ptr + _REAR + zero_offs, 1,
                                       mask=won, sem="relaxed")
                fits = won & (ticket - front < _CAP)
                tl.store(ring_ptr + (ticket & _MASK), nidx, mask=fits)
                tl.store(state_ptr + _OVF + zero_offs, 1,
                         mask=won & (ticket - front >= _CAP))

        cta_sync()  # all enqueues + final rear for this level visible
        new_rear = tl.load(state_ptr + _REAR, volatile=True)
        overflowed = tl.load(state_ptr + _OVF, volatile=True)
        cta_sync()  # everyone has read new_rear; next level's atomics may begin

        level += 1
        if overflowed == 0:  # on overflow the loop exits with front/rear as is
            peak_occ = tl.maximum(peak_occ, new_rear - front)
            front = rear
            rear = new_rear

    tl.atomic_add(counters_ptr + _PROCESSED + zero_offs, my_processed,
                  sem="relaxed")
    tl.atomic_add(counters_ptr + _CAS_ATTEMPTS + zero_offs, my_cas_attempts,
                  sem="relaxed")
    filled = tl.load(state_ptr + _REAR, volatile=True)
    tl.store(counters_ptr + _FILLED, filled.to(tl.int64))
    tl.store(counters_ptr + _LEVELS, level.to(tl.int64))
    # `overflowed` is s_overflow as read after the last level's barrier,
    # nothing writes it later, so it is Numba's exit read of s_overflow[0].
    # Reading the loop variable here (not state) also keeps it live after
    # the loop: when the while condition is its only use, Triton 3.7.1's
    # canonicalizer drops it from the scf.while results and
    # TritonGPURemoveLayoutConversions then crashes (result out of range).
    tl.store(counters_ptr + _OVERFLOW, overflowed.to(tl.int64))
    tl.store(counters_ptr + _PEAK_LEVEL, peak_level.to(tl.int64))
    tl.store(counters_ptr + _PEAK_OCC, peak_occ.to(tl.int64))
    tl.store(counters_ptr + _ACTIVE_THREAD_SUM, active_thread_sum)
    tl.store(counters_ptr + _ACTIVE_WARP_SUM, active_warp_sum)


@triton.jit
def _program_enqueue_two_tier(ring_ptr, spill_ptr, state_ptr, front, item,
                              won):
    """Two-tier enqueue with one atomic per program per tier.

    Twin of Numba's _warp_enqueue_two_tier, aggregated over the program
    instead of the warp. The winners reserve one slab of virtual tickets
    [base, base + count) with a single atomic on the rear; each takes
    base + its rank. Tickets inside the ring window [front, front +
    RING_CAPACITY) go to ring slots (front is frozen per level, so the v1
    distinct-slot argument holds). The rest, the top of the slab, append to
    the global spill tier with a second single atomic; their spill rank is
    rank - k, where k is the number of slab tickets that fit the ring. The
    cumsum stays outside every if (Triton 3.7 miscompiles scans in ifs).
    """
    w = won.to(tl.int32)
    rank = tl.cumsum(w, 0) - w
    count = tl.sum(w, 0)
    base = tl.full([], 0, tl.int32)
    if count > 0:
        base = tl.atomic_add(state_ptr + _REAR, count, sem="relaxed")
    ticket = base + rank
    in_ring = won & (ticket - front < _CAP)
    tl.store(ring_ptr + (ticket & _MASK), item, mask=in_ring)

    # Lanes past the window: ranks [k, count) of this slab.
    k = tl.minimum(tl.maximum(_CAP - (base - front), 0), count)
    to_spill = won & (rank >= k)
    count2 = count - k
    gbase = tl.full([], 0, tl.int32)
    if count2 > 0:
        gbase = tl.atomic_add(state_ptr + _SPILL_REAR, count2, sem="relaxed")
    tl.store(spill_ptr + gbase + (rank - k), item, mask=to_spill)


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def single_block_bfs_spill_kernel(img_ptr, visited_ptr, depth_ptr, seed_x,
                                  seed_y, spill_ptr, counters_ptr,
                                  level_sizes_ptr, ring_ptr, state_ptr,
                                  width, height, trace_cap,
                                  BLOCK: tl.constexpr):
    """v2: two-tier frontier, ring fast path + global spill tier.

    Same host contract as single_block_bfs_kernel, plus `spill`: an int32
    array of width*height entries (every pixel is enqueued at most once and
    the seed lives in the ring, so the tier cannot overflow; no tripwire).

    Each level's frontier is the fused window: n_shared ring entries at
    virtual tickets [sf, sr) followed by the spill slice [gf, gr). Levels
    are separated by three barriers: the v1 pair plus one that publishes
    the rear clamp (tickets past the ring window went to spill, so the rear
    is pulled back to the last ticket the ring stores).
    """
    offs = tl.arange(0, BLOCK)
    zero_offs = offs * 0

    seed = seed_x * height + seed_y
    tl.store(ring_ptr, seed)
    tl.store(state_ptr + _REAR, 1)
    tl.store(state_ptr + _SPILL_REAR, 0)
    tl.store(visited_ptr + seed, 1)
    cta_sync()

    sf = 0   # ring window start (virtual ticket index), frozen per level
    sr = 1   # ring window end
    gf = 0   # spill window start (flat index; the spill tier is append-only)
    gr = 0   # spill window end
    level = 0
    peak_level = 1
    peak_occ = 1
    peak_spill_window = 0
    active_thread_sum = tl.full([], 0, tl.int64)
    active_warp_sum = tl.full([], 0, tl.int64)
    my_processed = tl.zeros([BLOCK], tl.int64)
    my_cas_attempts = tl.zeros([BLOCK], tl.int64)

    while (sr - sf) + (gr - gf) > 0:
        n_shared = sr - sf
        level_size = n_shared + (gr - gf)
        peak_level = tl.maximum(peak_level, level_size)
        peak_spill_window = tl.maximum(peak_spill_window, gr - gf)
        active = tl.minimum(level_size, BLOCK)
        active_thread_sum += active.to(tl.int64)
        active_warp_sum += ((active + 31) // 32).to(tl.int64)
        if level < trace_cap:
            tl.store(level_sizes_ptr + level, level_size)

        # Block-stride over the fused frontier: ring entries first, then the
        # spill slice, one flat index space [0, level_size).
        for base in range(0, level_size, BLOCK):
            i = base + offs
            m = i < level_size
            from_ring = i < n_shared
            p_ring = tl.load(ring_ptr + ((sf + i) & _MASK),
                             mask=m & from_ring, other=0)
            p_spill = tl.load(spill_ptr + (gf + (i - n_shared)),
                              mask=m & (i >= n_shared), other=0)
            pixel = tl.where(from_ring, p_ring, p_spill)
            x = pixel // height
            y = pixel % height
            _recolor(img_ptr, depth_ptr, pixel, level, m)
            my_processed += m.to(tl.int64)

            for d in tl.static_range(4):
                nx = x + _DX[d]
                ny = y + _DY[d]
                inb = m & (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
                nidx = nx * height + ny
                cand = _is_red(img_ptr, nidx, inb)
                my_cas_attempts += cand.to(tl.int64)
                old = tl.atomic_xchg(visited_ptr + nidx, 1, mask=cand,
                                     sem="relaxed")
                won = cand & (old == 0)
                _program_enqueue_two_tier(ring_ptr, spill_ptr, state_ptr, sf,
                                          nidx, won)

        cta_sync()  # all enqueues + final tier counters visible
        sr_raw = tl.load(state_ptr + _REAR, volatile=True)
        gr_new = tl.load(state_ptr + _SPILL_REAR, volatile=True)
        cta_sync()  # everyone has read them
        sr_eff = tl.minimum(sr_raw, sf + _CAP)
        if sr_eff != sr_raw:
            tl.store(state_ptr + _REAR, sr_eff)  # retract the spilled tickets
        cta_sync()  # clamp visible before next level's reservations

        level += 1
        peak_occ = tl.maximum(peak_occ, (sr_eff - sf) + (gr_new - gf))
        sf = sr
        sr = sr_eff
        gf = gr
        gr = gr_new

    tl.atomic_add(counters_ptr + _PROCESSED + zero_offs, my_processed,
                  sem="relaxed")
    tl.atomic_add(counters_ptr + _CAS_ATTEMPTS + zero_offs, my_cas_attempts,
                  sem="relaxed")
    rear_final = tl.load(state_ptr + _REAR, volatile=True)
    spilled = tl.load(state_ptr + _SPILL_REAR, volatile=True)
    tl.store(counters_ptr + _FILLED, (rear_final + spilled).to(tl.int64))
    tl.store(counters_ptr + _LEVELS, level.to(tl.int64))
    tl.store(counters_ptr + _PEAK_LEVEL, peak_level.to(tl.int64))
    tl.store(counters_ptr + _PEAK_OCC, peak_occ.to(tl.int64))
    tl.store(counters_ptr + _ACTIVE_THREAD_SUM, active_thread_sum)
    tl.store(counters_ptr + _ACTIVE_WARP_SUM, active_warp_sum)
    tl.store(counters_ptr + _SPILLED, spilled.to(tl.int64))
    tl.store(counters_ptr + _PEAK_SPILL_WINDOW,
             peak_spill_window.to(tl.int64))


__all__ = [
    "single_block_bfs_kernel", "single_block_bfs_spill_kernel",
    "RING_CAPACITY", "RING_MASK", "NUM_COUNTERS", "STATE_SLOTS",
    "REAR", "OVF", "SPILL_REAR",
    "FILLED", "LEVELS", "OVERFLOW", "PEAK_LEVEL", "PEAK_OCC",
    "ACTIVE_THREAD_SUM", "ACTIVE_WARP_SUM", "PROCESSED", "CAS_ATTEMPTS",
    "SPILLED", "PEAK_SPILL_WINDOW",
]
