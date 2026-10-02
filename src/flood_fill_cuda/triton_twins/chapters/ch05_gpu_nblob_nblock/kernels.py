"""Triton twins of the seed-discovery kernels.

Same two variant families, same phases, same atomicMin union-find
protocol, same barriers and same counters as the Numba kernels.py; read
its docstring for the algorithm and the arguments that make it correct.
This file only changes how each CUDA construct is spelled:

- One Numba block of T threads is one Triton program whose tensors have
  BLOCK = T lanes (num_warps = T // 32): lane i plays thread i. A grid of
  B blocks is B programs, and every grid-stride loop keeps Numba's stride
  (num_programs * BLOCK) and its element -> (block, lane) assignment.
- grid.sync() is runtime.device.grid_sync on a monotonic int64 counter,
  inside a cooperative launch (the driver refuses a grid that cannot be
  co-resident, exactly like Numba). Same barrier placement: two per level,
  the fence sandwich around the discovery rear read, the instrumented-only
  closing barrier after the seed_merge flatten.
- The warp-aggregated enqueue becomes program-aggregated: ranks from one
  exclusive tl.cumsum, ONE atomic on the rear per program instead of one
  per warp. Queue order inside a level differs; the per-level SET (and so
  depth, labels and every level count) does not.
- The visited CAS(0 -> 1) becomes a masked atomic_xchg(1): visited only
  ever holds 0 or 1, so "old == 0 wins" is the same exactly-once claim
  (Triton's atomic_cas takes no mask).
- _find and _union are per-thread while loops in Numba. Here they are
  lockstep loops over a lane mask (while any lane is still active), with
  masked loads and a masked atomic_min: the same per-lane result, but the
  slowest lane of the program sets the iteration count.
- Per-thread counter registers stay per-lane registers (int32 lane
  vectors; the union cycle sum int64), reduced once at exit and added to
  the counters with one atomic per program, so every total is the same
  and the hot loops carry no extra reduction.
- %smid / %globaltimer / %clock64 come from inline PTX (runtime.device)
  instead of the linked smid.cu.
- Each bare twin is the same @triton.jit body compiled with INSTR=False:
  counters, owner/prov stores, stamps, the level trace and block_stats are
  compiled out, exactly the code the Numba *_bare_kernel twins drop (their
  pointer arguments are passed as None).

Phases are @triton.jit helpers (all inlined) so the lattice twins can
reuse the same scan, fill and flatten code, the way the Numba lattice
kernels reuse seed_merge's body.
"""

import triton
import triton.language as tl

from flood_fill_cuda.chapters.ch05_gpu_nblob_nblock.kernels import (
    ACTIVE_THREAD_SUM, ACTIVE_WARP_SUM, BS_PROCESSED, BS_SMID, CANDIDATES,
    CAS_ATTEMPTS, DX8_HOST, DY8_HOST, FILLED, LEVELS, N_PALETTE, OVERFLOW,
    PDX_HOST, PDY_HOST, PEAK_LEVEL, PEAK_OCC, PROCESSED, Q_REAR,
    UNION_ATTEMPTS, UNION_CYCLES, UNION_DONE,
)
from flood_fill_cuda.triton_twins.runtime.device import (
    grid_sync, read_clock64, read_globaltimer, read_smid,
)

# The Numba chapter's tables and slots, as compile-time constants (one
# source of truth: the values come from the Numba kernels module).
_DX8 = tl.constexpr(tuple(int(v) for v in DX8_HOST))   # E SE S SW W NW N NE
_DY8 = tl.constexpr(tuple(int(v) for v in DY8_HOST))
_PDX = tl.constexpr(tuple(int(v) for v in PDX_HOST))   # lex-predecessors
_PDY = tl.constexpr(tuple(int(v) for v in PDY_HOST))
_N_PALETTE = tl.constexpr(N_PALETTE)
_Q_REAR = tl.constexpr(Q_REAR)
_FILLED = tl.constexpr(FILLED)
_LEVELS = tl.constexpr(LEVELS)
_OVERFLOW = tl.constexpr(OVERFLOW)
_PEAK_LEVEL = tl.constexpr(PEAK_LEVEL)
_PEAK_OCC = tl.constexpr(PEAK_OCC)
_ACTIVE_THREAD_SUM = tl.constexpr(ACTIVE_THREAD_SUM)
_ACTIVE_WARP_SUM = tl.constexpr(ACTIVE_WARP_SUM)
_PROCESSED = tl.constexpr(PROCESSED)
_CAS_ATTEMPTS = tl.constexpr(CAS_ATTEMPTS)
_CANDIDATES = tl.constexpr(CANDIDATES)
_UNION_ATTEMPTS = tl.constexpr(UNION_ATTEMPTS)
_UNION_DONE = tl.constexpr(UNION_DONE)
_UNION_CYCLES = tl.constexpr(UNION_CYCLES)
_BS_PROCESSED = tl.constexpr(BS_PROCESSED)
_BS_SMID = tl.constexpr(BS_SMID)

# Runtime ints that change between calls: never specialize on them, or a
# new scene size could trigger a compile inside the timed window.
_RUNTIME_INTS = ["width", "height", "n", "q_cap", "trace_cap"]

# Pointer hops _find takes between two "any lane still climbing?" checks.
# Each check is a program-wide reduction (shared memory + barriers); a lane
# that already reached its root runs the extra hops fully masked (no load,
# no state change), so every lane's chase is unchanged. Measured on the
# 1000^2 blob grid: union_merge 9.9 ms at 1 hop/check, 5.0 at 4, 4.3 at 8.
_FIND_HOPS = tl.constexpr(8)


# ------------------------------------------------------------ device helpers


@triton.jit
def _is_red(img_ptr, lin, m):
    """img[x, y] == (255, 0, 0) on the lanes in m, False elsewhere.

    img is (width, height, 3) uint8, C-contiguous: byte (x*height + y)*3 + c.
    The byte offset is int64, like Numba's index math (lin*3 passes 2**31
    at 716 M pixels)."""
    off = lin.to(tl.int64) * 3
    r = tl.load(img_ptr + off, mask=m, other=0)
    g = tl.load(img_ptr + off + 1, mask=m, other=0)
    b = tl.load(img_ptr + off + 2, mask=m, other=0)
    return m & (r == 255) & (g == 0) & (b == 0)


@triton.jit
def _in_bounds(nx, ny, width, height, m):
    return m & (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)


@triton.jit
def _find(parent_ptr, i, act):
    """Chase to the root on the lanes in act. Read-only, never compresses
    (twin of the Numba _find): the per-thread while loop becomes a
    lockstep loop that runs while any lane of the program still climbs,
    checked every _FIND_HOPS hops."""
    p = tl.load(parent_ptr + i, mask=act, other=0)
    climbing = act & (p != i)
    while tl.max(climbing.to(tl.int32), 0) > 0:
        for _ in tl.static_range(_FIND_HOPS):
            i = tl.where(climbing, p, i)
            p = tl.load(parent_ptr + i, mask=climbing, other=0)
            climbing = climbing & (p != i)
    return i


@triton.jit
def _union(parent_ptr, a, b, act):
    """Link the classes of a and b on the lanes in act; int32 1 on the
    lanes whose atomic_min retired a root, else 0 (twin of the Numba
    _union). Its while-True with two early returns becomes a lane mask: a
    lane leaves when its roots agree (returns 0) or when its atomic_min
    lands on a slot that still held its own index (returns 1); every other
    lane retries from the value the atomic handed back. Same termination
    argument: the working index strictly decreases."""
    retired = tl.zeros_like(a)
    go = act
    while tl.max(go.to(tl.int32), 0) > 0:
        a = _find(parent_ptr, a, go)
        b = _find(parent_ptr, b, go)
        go = go & (a != b)
        hi = tl.maximum(a, b)       # the LARGER root takes the smaller value
        lo = tl.minimum(a, b)
        old = tl.atomic_min(parent_ptr + hi, lo, mask=go, sem="relaxed",
                            scope="gpu")
        retired = tl.where(go & (old == hi), 1, retired)
        go = go & (old != hi)
        a = tl.where(go, old, hi)   # retry from the true current value
        b = lo
    return retired


@triton.jit
def _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, item, won, q_cap):
    """Program-aggregated append on the global rear (twin of
    _warp_enqueue_global): exclusive tl.cumsum ranks, ONE atomic per
    program that has anything to append. The bound check stays the
    structurally unreachable OVERFLOW tripwire."""
    w = won.to(tl.int32)
    cnt = tl.sum(w, 0)
    rank = tl.cumsum(w, 0) - w   # outside the if: Triton 3.7 miscompiles
    base = cnt * 0               # a scan inside a conditional
    if cnt > 0:
        base = tl.atomic_add(q_state_ptr + _Q_REAR, cnt, sem="relaxed",
                             scope="gpu")
    idx = base + rank
    tl.store(queue_ptr + idx, item, mask=won & (idx < q_cap))
    tl.store(counters_ptr + _OVERFLOW + idx * 0, 1, mask=won & (idx >= q_cap))


@triton.jit
def _paint(img_ptr, palette_ptr, lin, lbl, m):
    """img[x, y] = palette[lbl % N_PALETTE] on the lanes in m."""
    c = lbl % _N_PALETTE
    off = lin.to(tl.int64) * 3
    for k in tl.static_range(3):
        tl.store(img_ptr + off + k,
                 tl.load(palette_ptr + c * 3 + k, mask=m, other=0), mask=m)


@triton.jit
def _stamp(phase_ptr, SLOT: tl.constexpr, pid):
    """tid 0 stamps %globaltimer into phase_ns[SLOT]."""
    if pid == 0:
        tl.store(phase_ptr + SLOT, read_globaltimer(pid))


@triton.jit
def _sync(bar_ptr, epoch, nprog):
    """grid.sync(): one more barrier on the monotonic counter."""
    epoch += 1
    grid_sync(bar_ptr, epoch * nprog)
    return epoch


# --------------------------------------------------------------- phases


@triton.jit
def _iota(parent_ptr, n, pid, nprog, BLOCK: tl.constexpr):
    """P0: every pixel its own root."""
    lanes = tl.arange(0, BLOCK)
    for base in range(0, n, nprog * BLOCK):
        i = base + pid * BLOCK + lanes
        tl.store(parent_ptr + i, i, mask=i < n)


@triton.jit
def _corner_scan(img_ptr, visited_ptr, label_ptr, queue_ptr, q_state_ptr,
                 counters_ptr, width, height, n, q_cap, pid, nprog,
                 BLOCK: tl.constexpr):
    """seed_merge P1: a red pixel with no red lex-predecessor is a
    candidate; pre-visit it, label it with its own index, enqueue it.
    One lane per pixel, so the stores need no atomics."""
    lanes = tl.arange(0, BLOCK)
    for base in range(0, n, nprog * BLOCK):
        i = base + pid * BLOCK + lanes
        m = i < n
        x = i // height
        y = i % height
        red = _is_red(img_ptr, i, m)
        found = tl.zeros_like(red)
        for d in tl.static_range(4):
            nx = x + _PDX[d]
            ny = y + _PDY[d]
            found = found | _is_red(img_ptr, nx * height + ny,
                                    _in_bounds(nx, ny, width, height, red))
        cand = red & ~found
        tl.store(visited_ptr + i, 1, mask=cand)
        tl.store(label_ptr + i, i, mask=cand)
        _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, i, cand, q_cap)


@triton.jit
def _ccl_merge(img_ptr, parent_ptr, width, height, n, pid, nprog,
               BLOCK: tl.constexpr, INSTR: tl.constexpr):
    """ccl_fill P1: union along every red lex-predecessor adjacency (each
    8-adjacency once, from its lex-greater endpoint). Returns the per-lane
    (union_attempts, union_done) registers."""
    lanes = tl.arange(0, BLOCK)
    union_attempts = tl.zeros([BLOCK], tl.int32)
    union_done = tl.zeros([BLOCK], tl.int32)
    for base in range(0, n, nprog * BLOCK):
        i = base + pid * BLOCK + lanes
        m = i < n
        x = i // height
        y = i % height
        red = _is_red(img_ptr, i, m)
        for d in tl.static_range(4):
            nx = x + _PDX[d]
            ny = y + _PDY[d]
            nlin = nx * height + ny
            pair = _is_red(img_ptr, nlin, _in_bounds(nx, ny, width, height, red))
            done = _union(parent_ptr, i, nlin, pair)
            if INSTR:
                union_attempts += pair.to(tl.int32)
                union_done += done
    return union_attempts, union_done


@triton.jit
def _ccl_flatten_seed(img_ptr, visited_ptr, label_ptr, parent_ptr, queue_ptr,
                      q_state_ptr, counters_ptr, n, q_cap, pid, nprog,
                      BLOCK: tl.constexpr):
    """ccl_fill P2: flatten every red pixel to its root; the root pixel is
    the blob's canonical seed - pre-visit and enqueue it."""
    lanes = tl.arange(0, BLOCK)
    for base in range(0, n, nprog * BLOCK):
        i = base + pid * BLOCK + lanes
        m = i < n
        red = _is_red(img_ptr, i, m)
        root = _find(parent_ptr, i, red)
        tl.store(label_ptr + i, root, mask=red)
        seed = red & (root == i)
        tl.store(visited_ptr + i, 1, mask=seed)
        _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, i, seed, q_cap)


@triton.jit
def _fence_sandwich(bar_ptr, q_state_ptr, phase_ptr, epoch, pid, nprog,
                    SLOT: tl.constexpr, INSTR: tl.constexpr):
    """grid.sync(); [stamp]; rear = q_state[Q_REAR]; grid.sync() - every
    program reads the same discovery count (no n_seeds argument)."""
    epoch = _sync(bar_ptr, epoch, nprog)
    if INSTR:
        _stamp(phase_ptr, SLOT, pid)
    rear = tl.load(q_state_ptr + _Q_REAR)
    epoch = _sync(bar_ptr, epoch, nprog)
    return rear, epoch


@triton.jit
def _fill_levels(img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                 label_ptr, queue_ptr, q_state_ptr, counters_ptr,
                 level_sizes_ptr, palette_ptr, bar_ptr, width, height, q_cap,
                 trace_cap, rear, epoch, pid, nprog, BLOCK: tl.constexpr,
                 MERGE: tl.constexpr, INSTR: tl.constexpr):
    """The level-synchronous multisource fill, two barriers per level.

    MERGE=False is ccl_fill P3: labels are final, paint at dequeue.
    MERGE=True is seed_merge P2: NO paint (img stays red, so a claimed
    pixel can still be probed), the winner stamps its label BEFORE the
    enqueue, a CAS loser that sees a different visible label unions the
    two (-1 = not yet visible: skip, the other side retries a level
    later). Returns the grid-uniform level bookkeeping and the per-lane
    counter registers (the Numba my_* registers, one per thread)."""
    lanes = tl.arange(0, BLOCK)
    stride = nprog * BLOCK
    front = rear * 0
    level = rear * 0
    peak_level = rear * 0
    peak_occ = rear * 0
    active_thread_sum = tl.full((), 0, tl.int64)
    active_warp_sum = tl.full((), 0, tl.int64)
    processed = tl.zeros([BLOCK], tl.int32)
    cas_attempts = tl.zeros([BLOCK], tl.int32)
    union_attempts = tl.zeros([BLOCK], tl.int32)
    union_done = tl.zeros([BLOCK], tl.int32)
    union_cycles = tl.zeros([BLOCK], tl.int64)

    while front < rear:
        level_size = rear - front
        if INSTR:
            peak_level = tl.maximum(peak_level, level_size)
            active = tl.minimum(level_size, stride)
            active_thread_sum += active.to(tl.int64)
            active_warp_sum += ((active + 31) // 32).to(tl.int64)
            tl.store(level_sizes_ptr + level, level_size,
                     mask=(pid == 0) & (level < trace_cap))

        for base in range(front, rear, stride):
            offs = base + pid * BLOCK + lanes
            m = offs < rear
            pixel = tl.load(queue_ptr + offs, mask=m, other=0)
            x = pixel // height
            y = pixel % height
            lbl = tl.load(label_ptr + pixel, mask=m, other=-1)
            if not MERGE:
                _paint(img_ptr, palette_ptr, pixel, lbl, m)
            tl.store(depth_ptr + pixel, level, mask=m)
            if INSTR:
                tl.store(owner_ptr + pixel, pid.to(tl.int16), mask=m)
                processed += m.to(tl.int32)

            for d in tl.static_range(8):
                nx = x + _DX8[d]
                ny = y + _DY8[d]
                nlin = nx * height + ny
                probe = _is_red(img_ptr, nlin,
                                _in_bounds(nx, ny, width, height, m))
                if INSTR:
                    cas_attempts += probe.to(tl.int32)
                old = tl.atomic_xchg(visited_ptr + nlin, 1, mask=probe,
                                     sem="relaxed", scope="gpu")
                won = probe & (old == 0)
                if MERGE:
                    # win: stamp the inherited label BEFORE the entry
                    # becomes dequeueable
                    tl.store(label_ptr + nlin, lbl, mask=won)
                _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, nlin, won,
                             q_cap)
                if MERGE:
                    lost = probe & (old != 0)
                    other = tl.load(label_ptr + nlin, mask=lost, other=-1)
                    coll = lost & (other >= 0) & (other != lbl)
                    if INSTR:
                        union_attempts += coll.to(tl.int32)
                        t0 = read_clock64(pid)
                        done = _union(parent_ptr, lbl, other, coll)
                        # a colliding lane spends the program's lockstep
                        # union time: its own cycles, as in Numba
                        union_cycles += tl.where(coll, read_clock64(pid) - t0,
                                                 0)
                        union_done += done
                    else:
                        _union(parent_ptr, lbl, other, coll)

        epoch = _sync(bar_ptr, epoch, nprog)   # enqueues + final rear visible
        new_rear = tl.load(q_state_ptr + _Q_REAR)
        epoch = _sync(bar_ptr, epoch, nprog)   # everyone has read new_rear

        level += 1
        if INSTR:
            peak_occ = tl.maximum(peak_occ, new_rear - front)
        front = rear
        rear = new_rear

    return (level, epoch, peak_level, peak_occ, active_thread_sum,
            active_warp_sum, processed, cas_attempts, union_attempts,
            union_done, union_cycles)


@triton.jit
def _merge_flatten(img_ptr, label_ptr, prov_ptr, parent_ptr, palette_ptr, n,
                   pid, nprog, BLOCK: tl.constexpr, INSTR: tl.constexpr):
    """seed_merge P3: prov snapshot (instrumented), resolve the final
    label, relabel, repaint. label_map >= 0 is exactly the filled set."""
    lanes = tl.arange(0, BLOCK)
    for base in range(0, n, nprog * BLOCK):
        i = base + pid * BLOCK + lanes
        m = i < n
        prov = tl.load(label_ptr + i, mask=m, other=-1)
        has = m & (prov >= 0)
        if INSTR:
            tl.store(prov_ptr + i, prov, mask=has)
        final = _find(parent_ptr, prov, has)
        tl.store(label_ptr + i, final, mask=has)
        _paint(img_ptr, palette_ptr, i, final, has)


@triton.jit
def _exit_counters(counters_ptr, block_stats_ptr, q_state_ptr, pid, level,
                   peak_level, peak_occ, active_thread_sum, active_warp_sum,
                   processed, cas_attempts, union_attempts, union_done,
                   union_cycles, n_seeds, INSTR: tl.constexpr,
                   CYCLES: tl.constexpr):
    """Exit: the per-lane counter registers summed and added with one
    atomic per program (Numba: one per thread), the block_stats row, and
    the grid-uniform values written once by tid 0."""
    if INSTR:
        my_processed = tl.sum(processed.to(tl.int64), 0)
        tl.atomic_add(counters_ptr + _PROCESSED, my_processed, sem="relaxed",
                      scope="gpu")
        tl.atomic_add(counters_ptr + _CAS_ATTEMPTS,
                      tl.sum(cas_attempts.to(tl.int64), 0), sem="relaxed",
                      scope="gpu")
        tl.atomic_add(counters_ptr + _UNION_ATTEMPTS,
                      tl.sum(union_attempts.to(tl.int64), 0), sem="relaxed",
                      scope="gpu")
        tl.atomic_add(counters_ptr + _UNION_DONE,
                      tl.sum(union_done.to(tl.int64), 0), sem="relaxed",
                      scope="gpu")
        if CYCLES:
            tl.atomic_add(counters_ptr + _UNION_CYCLES,
                          tl.sum(union_cycles, 0), sem="relaxed", scope="gpu")
        tl.atomic_add(block_stats_ptr + pid * 2 + _BS_PROCESSED, my_processed,
                      sem="relaxed", scope="gpu")
        tl.store(block_stats_ptr + pid * 2 + _BS_SMID,
                 read_smid(pid).to(tl.int64))
    if pid == 0:
        tl.store(counters_ptr + _FILLED,
                 tl.load(q_state_ptr + _Q_REAR).to(tl.int64))
        tl.store(counters_ptr + _LEVELS, level.to(tl.int64))
        if INSTR:
            tl.store(counters_ptr + _PEAK_LEVEL, peak_level.to(tl.int64))
            tl.store(counters_ptr + _PEAK_OCC, peak_occ.to(tl.int64))
            tl.store(counters_ptr + _ACTIVE_THREAD_SUM, active_thread_sum)
            tl.store(counters_ptr + _ACTIVE_WARP_SUM, active_warp_sum)
            tl.store(counters_ptr + _CANDIDATES, n_seeds.to(tl.int64))


# ---------------------------------------------------------------- ccl_fill


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def ccl_fill_kernel(img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                    label_ptr, queue_ptr, q_state_ptr, counters_ptr,
                    block_stats_ptr, level_sizes_ptr, phase_ptr, palette_ptr,
                    bar_ptr, width, height, n, q_cap, trace_cap,
                    BLOCK: tl.constexpr, INSTR: tl.constexpr):
    """Twin of ccl_fill_kernel (INSTR=True) and ccl_fill_bare_kernel
    (INSTR=False; owner/block_stats/level_sizes/phase are None).

    Host contract as the Numba kernel, plus palette (PALETTE_HOST as 18
    uint8, the const array) and bar (int64[1] zeroed: the grid barrier).
    Launch cooperative, [blocks] programs of BLOCK = tpb lanes."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.full((), 0, tl.int64)

    if INSTR:
        _stamp(phase_ptr, 0, pid)
    _iota(parent_ptr, n, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    if INSTR:
        _stamp(phase_ptr, 1, pid)

    union_attempts, union_done = _ccl_merge(img_ptr, parent_ptr, width,
                                            height, n, pid, nprog, BLOCK,
                                            INSTR)
    epoch = _sync(bar_ptr, epoch, nprog)
    if INSTR:
        _stamp(phase_ptr, 2, pid)

    _ccl_flatten_seed(img_ptr, visited_ptr, label_ptr, parent_ptr, queue_ptr,
                      q_state_ptr, counters_ptr, n, q_cap, pid, nprog, BLOCK)
    rear, epoch = _fence_sandwich(bar_ptr, q_state_ptr, phase_ptr, epoch, pid,
                                  nprog, 3, INSTR)
    n_seeds = rear

    (level, epoch, peak_level, peak_occ, active_thread_sum, active_warp_sum,
     processed, cas_attempts, _ua, _ud, _uc) = _fill_levels(
        img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr, label_ptr,
        queue_ptr, q_state_ptr, counters_ptr, level_sizes_ptr, palette_ptr,
        bar_ptr, width, height, q_cap, trace_cap, rear, epoch, pid, nprog,
        BLOCK, False, INSTR)
    # loop exit is post-barrier: this stamp closes the fill for the grid
    if INSTR:
        _stamp(phase_ptr, 4, pid)

    _exit_counters(counters_ptr, block_stats_ptr, q_state_ptr, pid, level,
                   peak_level, peak_occ, active_thread_sum, active_warp_sum,
                   processed, cas_attempts, union_attempts, union_done, _uc,
                   n_seeds, INSTR, False)


# --------------------------------------------------------------- seed_merge


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def seed_merge_kernel(img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                      label_ptr, prov_ptr, queue_ptr, q_state_ptr,
                      counters_ptr, block_stats_ptr, level_sizes_ptr,
                      phase_ptr, palette_ptr, bar_ptr, width, height, n,
                      q_cap, trace_cap, BLOCK: tl.constexpr,
                      INSTR: tl.constexpr):
    """Twin of seed_merge_kernel (INSTR=True) and seed_merge_bare_kernel
    (INSTR=False; owner/prov/block_stats/level_sizes/phase are None).

    Host contract as ccl_fill_kernel plus prov_label filled with -1."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.full((), 0, tl.int64)

    if INSTR:
        _stamp(phase_ptr, 0, pid)
    _iota(parent_ptr, n, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    if INSTR:
        _stamp(phase_ptr, 1, pid)

    _corner_scan(img_ptr, visited_ptr, label_ptr, queue_ptr, q_state_ptr,
                 counters_ptr, width, height, n, q_cap, pid, nprog, BLOCK)
    rear, epoch = _fence_sandwich(bar_ptr, q_state_ptr, phase_ptr, epoch, pid,
                                  nprog, 2, INSTR)
    n_candidates = rear

    (level, epoch, peak_level, peak_occ, active_thread_sum, active_warp_sum,
     processed, cas_attempts, union_attempts, union_done,
     union_cycles) = _fill_levels(
        img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr, label_ptr,
        queue_ptr, q_state_ptr, counters_ptr, level_sizes_ptr, palette_ptr,
        bar_ptr, width, height, q_cap, trace_cap, rear, epoch, pid, nprog,
        BLOCK, True, INSTR)
    if INSTR:
        _stamp(phase_ptr, 3, pid)

    # P3: the loop's closing barrier pair ordered every union before here
    _merge_flatten(img_ptr, label_ptr, prov_ptr, parent_ptr, palette_ptr, n,
                   pid, nprog, BLOCK, INSTR)
    # instrumented-only closing barrier so the flatten stamp covers the
    # whole grid's P3 (the bare twin ends without it)
    if INSTR:
        epoch = _sync(bar_ptr, epoch, nprog)
        _stamp(phase_ptr, 4, pid)

    _exit_counters(counters_ptr, block_stats_ptr, q_state_ptr, pid, level,
                   peak_level, peak_occ, active_thread_sum, active_warp_sum,
                   processed, cas_attempts, union_attempts, union_done,
                   union_cycles, n_candidates, INSTR, True)


# ------------------------------------------------- benchmark phase kernels
# Discovery WITHOUT the fill (benchmark attribution: fill ~= fused - phase).
# They call the same phase helpers as the fused kernels, so the measured
# phase cannot drift from the real one.


@triton.jit(do_not_specialize=["width", "height", "n", "q_cap"])
def seed_scan_kernel(img_ptr, visited_ptr, label_ptr, parent_ptr, queue_ptr,
                     q_state_ptr, counters_ptr, bar_ptr, width, height, n,
                     q_cap, BLOCK: tl.constexpr):
    """Twin of seed_scan_kernel: seed_merge P0-P1, then the rear count."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.full((), 0, tl.int64)
    _iota(parent_ptr, n, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    _corner_scan(img_ptr, visited_ptr, label_ptr, queue_ptr, q_state_ptr,
                 counters_ptr, width, height, n, q_cap, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    if pid == 0:
        tl.store(counters_ptr + _CANDIDATES,
                 tl.load(q_state_ptr + _Q_REAR).to(tl.int64))


@triton.jit(do_not_specialize=["width", "height", "n", "q_cap"])
def ccl_kernel(img_ptr, visited_ptr, label_ptr, parent_ptr, queue_ptr,
               q_state_ptr, counters_ptr, bar_ptr, width, height, n, q_cap,
               BLOCK: tl.constexpr):
    """Twin of ccl_kernel: ccl_fill P0-P2, then the rear count (n_blobs)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.full((), 0, tl.int64)
    _iota(parent_ptr, n, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    _ccl_merge(img_ptr, parent_ptr, width, height, n, pid, nprog, BLOCK, False)
    epoch = _sync(bar_ptr, epoch, nprog)
    _ccl_flatten_seed(img_ptr, visited_ptr, label_ptr, parent_ptr, queue_ptr,
                      q_state_ptr, counters_ptr, n, q_cap, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    if pid == 0:
        tl.store(counters_ptr + _CANDIDATES,
                 tl.load(q_state_ptr + _Q_REAR).to(tl.int64))
