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
- _find and _union are per-thread while loops in Numba. A Triton program
  has no per-lane control flow, so the twin spells them two ways, picked
  by the LANE constexpr:
  LANE=1 (the default): the union-find loops are per-lane state
  machines (section "lane-independent union-find"). Each lane holds its
  own grid-stride item and its place inside that item's finds and
  unions, moves one parent hop per step and takes its next item as soon
  as it is done, as a SIMT thread does. ccl_fill's merge and flatten and
  compress always run this way; seed_merge's fill switches to it per
  program in union-dense levels (_fill_levels); seed_merge's final
  flatten keeps the lockstep find (_merge_flatten).
  LANE=0: lockstep loops over a lane mask (_find, _union): a loop runs
  while any lane of the program is still active, so the slowest of the
  program's lanes sets every lane's pace. Kept to measure what that
  translation costs (compare.py's lane_schedule experiment).
  Both run Numba's per-item operations: the same read-only find, the
  same atomic_min of the smaller root into the larger root's slot, the
  same retry from the value the atomic returned, the same counters.
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

Phases are @triton.jit helpers (all inlined) so the lattice twins reuse
the same fill and flatten code, the way the Numba lattice kernels repeat
seed_merge's body. The r128 build is the same lattice body launched with
maxnreg=128 (Numba: max_registers=128 on the same py_func); the split
build's two plain cleanup kernels are the compress and flatten helpers
with no barrier around them.
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
    (twin of the Numba _find): the LANE=0 spelling, where the per-thread
    while loop becomes a lockstep loop that runs while any lane of the
    program still climbs, checked every _FIND_HOPS hops."""
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
    _union, LANE=0 spelling). Its while-True with two early returns
    becomes a lane mask: a
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
def _fill_batch(img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                label_ptr, queue_ptr, q_state_ptr, counters_ptr, palette_ptr,
                width, height, q_cap, base, rear, level, pid, processed,
                cas_attempts, union_attempts, union_done, union_cycles,
                BLOCK: tl.constexpr, MERGE: tl.constexpr,
                INSTR: tl.constexpr):
    """One grid-stride batch of a level, the per-direction body: BLOCK
    queue slots dequeued together, the 8 directions probed in lockstep,
    each direction's unions with the lockstep _union. Returns the counter
    registers and, per lane, how many of its probes collided (MERGE)."""
    lanes = tl.arange(0, BLOCK)
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
    ncoll = tl.zeros([BLOCK], tl.int32)

    for d in tl.static_range(8):
        nx = x + _DX8[d]
        ny = y + _DY8[d]
        nlin = nx * height + ny
        probe = _is_red(img_ptr, nlin, _in_bounds(nx, ny, width, height, m))
        if INSTR:
            cas_attempts += probe.to(tl.int32)
        old = tl.atomic_xchg(visited_ptr + nlin, 1, mask=probe,
                             sem="relaxed", scope="gpu")
        won = probe & (old == 0)
        if MERGE:
            # win: stamp the inherited label BEFORE the entry becomes
            # dequeueable
            tl.store(label_ptr + nlin, lbl, mask=won)
        _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, nlin, won, q_cap)
        if MERGE:
            lost = probe & (old != 0)
            other = tl.load(label_ptr + nlin, mask=lost, other=-1)
            coll = lost & (other >= 0) & (other != lbl)
            ncoll += coll.to(tl.int32)
            if INSTR:
                union_attempts += coll.to(tl.int32)
                t0 = read_clock64(pid)
                done = _union(parent_ptr, lbl, other, coll)
                # a colliding lane spends the program's lockstep union
                # time: its own cycles, as in Numba
                union_cycles += tl.where(coll, read_clock64(pid) - t0, 0)
                union_done += done
            else:
                _union(parent_ptr, lbl, other, coll)
    return (processed, cas_attempts, union_attempts, union_done,
            union_cycles, ncoll)


@triton.jit
def _fill_levels(img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                 label_ptr, queue_ptr, q_state_ptr, counters_ptr,
                 level_sizes_ptr, palette_ptr, bar_ptr, width, height, q_cap,
                 trace_cap, rear, epoch, pid, nprog, BLOCK: tl.constexpr,
                 MERGE: tl.constexpr, INSTR: tl.constexpr,
                 LANE: tl.constexpr):
    """The level-synchronous multisource fill, two barriers per level.

    MERGE=False is ccl_fill P3: labels are final, paint at dequeue.
    MERGE=True is seed_merge P2: NO paint (img stays red, so a claimed
    pixel can still be probed), the winner stamps its label BEFORE the
    enqueue, a CAS loser that sees a different visible label unions the
    two (-1 = not yet visible: skip, the other side retries a level
    later).

    LANE=0: every batch runs the per-direction body (_fill_batch).
    LANE=1 (MERGE only: ccl_fill's fill has no union): each program
    starts a level with per-direction batches and switches to the
    lane-independent body (_merge_level_lanes) for the rest of the level
    after a batch whose probes collided at least _DENSE_COLL times per
    lane. Where few probes collide (seed_merge v1: about 7% of probes;
    lattice 16: 9%) lanes mostly probe, and a per-lane probe costs more
    than the static direction loop; where most do (lattice 1: all of
    them, lattice 4: 35%) the per-direction lockstep unions are what is
    slow. Programs never meet inside a level, so each decides alone; the
    slot -> (program, lane) map, the barriers and the level bookkeeping
    are the same in every case, and so are the outputs.

    Returns the grid-uniform level bookkeeping and the per-lane counter
    registers (the Numba my_* registers, one per thread)."""
    tl.static_assert(MERGE or not LANE,
                     "LANE applies to seed_merge's in-flight unions only")
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

        if LANE:
            base = front
            dense = level_size * 0
            while (base < rear) & (dense == 0):
                (processed, cas_attempts, union_attempts, union_done,
                 union_cycles, ncoll) = _fill_batch(
                    img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                    label_ptr, queue_ptr, q_state_ptr, counters_ptr,
                    palette_ptr, width, height, q_cap, base, rear, level,
                    pid, processed, cas_attempts, union_attempts, union_done,
                    union_cycles, BLOCK, MERGE, INSTR)
                base += stride
                # dense, and slots left for every lane to move on to
                dense = ((tl.sum(ncoll, 0) >= _DENSE_COLL * BLOCK)
                         & (rear - base >= 2 * stride)).to(tl.int32)
            # the rest of the level (nothing when no batch was dense)
            (processed, cas_attempts, union_attempts, union_done,
             union_cycles) = _merge_level_lanes(
                img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                label_ptr, queue_ptr, q_state_ptr, counters_ptr, width,
                height, q_cap, base, rear, level, pid, nprog, processed,
                cas_attempts, union_attempts, union_done, union_cycles,
                BLOCK, INSTR)
        else:
            for base in range(front, rear, stride):
                (processed, cas_attempts, union_attempts, union_done,
                 union_cycles, _nc) = _fill_batch(
                    img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                    label_ptr, queue_ptr, q_state_ptr, counters_ptr,
                    palette_ptr, width, height, q_cap, base, rear, level,
                    pid, processed, cas_attempts, union_attempts, union_done,
                    union_cycles, BLOCK, MERGE, INSTR)

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
    label, relabel, repaint. label_map >= 0 is exactly the filled set.

    The lockstep _find in both schedules. Neighbouring pixels share a
    provisional label, so a program's lanes chase the same chain side by
    side and lockstep makes their loads broadcasts; lane-independent
    chases drift apart and lose that (seed_merge on comb_2000, where the
    2000 tooth labels chain up: 52 ms lockstep, 97 ms per lane; Numba
    26). After the lattice builds' compress every find is <= 1 hop."""
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


# -------------------------------------------- lane-independent union-find
# LANE=1, the default. Numba runs _find and _union as per-thread while
# loops: a SIMT thread that reaches its root moves on to its next union
# or its next grid-stride item while its neighbours still climb. The
# lockstep spelling above (LANE=0) holds every lane until the slowest of
# the program's BLOCK lanes is done, at every find and every retry. That
# costs twice: the wasted slots (2.7x the warp-steps a per-warp schedule
# of the same run issues), and a deeper forest: programs advance in
# bursts and link into regions not merged yet, so chains grow (ccl_fill
# on two_disks_r1400: mean depth 41 vs Numba's 4, 3.1 G parent loads in
# the merge vs Numba's 0.34 G).
#
# Here each lane is a small state machine with its own registers: its
# grid-stride item (the same item -> (program, lane) map as Numba), its
# place inside that item (the pending directions, or the queue slot and
# direction in the fill) and its union in flight (a, b, the cursor c,
# which of the two finds it is in). A lane that finishes an item takes
# its next one at once; the loop ends when no lane has work left.
# Used by the ccl merge and flatten, compress and the dense levels of the
# seed_merge fill (_fill_levels); the seed_merge flatten keeps the
# lockstep find (_merge_flatten says why).
#
# A Triton step issues every state's code on every lane (masked), so the
# frequent cheap work runs in mini-steps and the rare costly work waits
# for a full step after every _LANE_HOPS mini-steps:
#   mini-step  one parent hop per lane (find a, then find b), the
#              register-only transitions (a's root -> start find b; same
#              root -> next pair of the item), the scan of the next item,
#              and the plain-store epilogue of compress
#   full step  the link atomic_min (retry or retire), the pixel reads of
#              a found item, the fill's dequeue and direction probe with
#              the program's enqueue, the ccl flatten's label + seed
#              append (one _cta_enqueue)
# Per item the operations are Numba's: the same read-only find (the
# first load is parent[start], no compression), the same atomic_min of
# the smaller root into the larger root's slot, the same retry from the
# value the atomic returned, the same counters at the same events. Only
# the ccl merge reads an item's 4 lex-predecessor pixels together before
# its unions (Numba reads each just before its union): img is constant
# in that phase, so the pairs and their order are the same.

# Mini-steps per full step. ccl_fill union_merge on two_disks_r1400 (48
# programs x 256 lanes): one full step per step 85 ms (193 SASS
# instructions per step, issue-bound), 8 mini-steps 34 ms (Numba 31 ms).
_LANE_HOPS = tl.constexpr(8)

# seed_merge fill only. A program switches a level to the lane body after
# a batch with at least _DENSE_COLL colliding probes per lane (of its 8);
# there a lane takes its next slot only while fewer than _LANE_WINDOW
# slots ahead of the program's slowest live lane (see _merge_level_lanes).
# Fill phase / Numba's, 24 programs (one process, interleaved):
#   threshold      1      2      3      4      6   never (= lockstep)
#   two_sq S4    1.41   1.38   1.38   1.69   3.31   2.85
#   two_sq S16   1.37   1.37   1.38   1.44   1.17   1.08
#   comb   S1    1.02   1.10   0.52   0.55   0.52   0.45
# Lattice 4 and 16 overlap in local density, so 3 keeps lattice 1 and 4
# and gives back ~20% at lattice 16 where some programs still switch.
_DENSE_COLL = tl.constexpr(3)
_LANE_WINDOW = tl.constexpr(16)
# DX8 / DY8 + 1 packed two bits per direction: a lane-varying direction d
# reads its offset as ((BITS >> 2d) & 3) - 1
_DX8_BITS = tl.constexpr(sum((int(v) + 1) << (2 * i)
                             for i, v in enumerate(DX8_HOST)))
_DY8_BITS = tl.constexpr(sum((int(v) + 1) << (2 * i)
                             for i, v in enumerate(DY8_HOST)))

# Lane states. HOP is the largest code, so tl.max(st) both tells whether
# any lane has work and whether any lane is in a union.
_ST_DONE = tl.constexpr(0)   # no item left
_ST_SCAN = tl.constexpr(1)   # test the current item / probe the next
                             # direction (fill)
_ST_WAIT = tl.constexpr(2)   # waiting for the full step: a found red
                             # pixel (merge), a finished chase (flatten),
                             # a queue slot to dequeue (fill)
_ST_LINK = tl.constexpr(3)   # both roots known and different: link
_ST_HOP = tl.constexpr(4)    # chasing: cursor c, find a (which=0) or b


@triton.jit
def _step_index(i, x, y, go, stride, sx, sy, height):
    """i += stride on the lanes in go, keeping (x, y) = divmod(i, height)
    with no division: (sx, sy) = divmod(stride, height), and y + sy <
    2 * height, so one carry is enough."""
    y2 = y + sy
    carry = (y2 >= height).to(tl.int32)
    return (tl.where(go, i + stride, i),
            tl.where(go, x + sx + carry, x),
            tl.where(go, y2 - carry * height, y))


@triton.jit
def _pred_pairs(img_ptr, x, y, width, height, red):
    """Bit d set iff lex-predecessor d of the red pixel (x, y) is in
    bounds and red: the ccl merge's union partners, bit order = Numba's
    d order."""
    bits = tl.zeros_like(x)
    for d in tl.static_range(4):
        nx = x + _PDX[d]
        ny = y + _PDY[d]
        pair = _is_red(img_ptr, nx * height + ny,
                       _in_bounds(nx, ny, width, height, red))
        bits = bits | (pair.to(tl.int32) << d)
    return bits


@triton.jit
def _take_pair(go, todo, i, height, a, b, c, which, st):
    """On the lanes in go: start the union of pixel i with its lowest
    pending lex-predecessor (Numba's next d), as _union(i, nlin) starts:
    find(a = i) first. Lanes in go with nothing pending are returned in
    `ended` (their item is done)."""
    start = go & (todo != 0)
    low = todo & (-todo)
    off = tl.zeros_like(todo)
    for d in tl.static_range(4):
        off = tl.where(low == (1 << d), _PDX[d] * height + _PDY[d], off)
    todo = tl.where(start, todo ^ low, todo)
    a = tl.where(start, i, a)
    b = tl.where(start, i + off, b)
    c = tl.where(start, i, c)
    which = tl.where(start, 0, which)
    st = tl.where(start, _ST_HOP, st)
    return todo, a, b, c, which, st, start, go & ~start


@triton.jit
def _union_hop(parent_ptr, st, which, a, b, c):
    """One mini-step of the unions in flight (lanes in _ST_HOP): one
    parent load. At a's root the lane turns to find(b); at b's root it
    leaves the hop state: roots differ -> _ST_LINK, same root -> `same`
    (the _union returns 0; the caller picks the next state)."""
    hop = st == _ST_HOP
    p = tl.load(parent_ptr + c, mask=hop, other=0)
    at_root = hop & (p == c)
    c = tl.where(hop, p, c)
    got_a = at_root & (which == 0)
    got_b = at_root & (which == 1)
    a = tl.where(got_a, c, a)
    c = tl.where(got_a, b, c)
    which = tl.where(got_a, 1, which)
    b = tl.where(got_b, c, b)
    st = tl.where(got_b & (a != b), _ST_LINK, st)
    return st, which, a, b, c, got_b & (a == b)


@triton.jit
def _union_link(parent_ptr, st, which, a, b, c):
    """The link of the lanes in _ST_LINK (full step): atomic_min the
    smaller root into the larger root's slot. `won` lanes retired a root
    (the slot still held its own index; the caller picks their next
    state); the others retry from the value the atomic returned:
    find(a = old) then find(b = the smaller root)."""
    link = st == _ST_LINK
    hi = tl.maximum(a, b)       # the LARGER root takes the smaller value
    lo = tl.minimum(a, b)
    old = tl.atomic_min(parent_ptr + hi, lo, mask=link, sem="relaxed",
                        scope="gpu")
    retry = link & (old != hi)
    a = tl.where(retry, old, a)
    b = tl.where(retry, lo, b)
    c = tl.where(retry, old, c)
    which = tl.where(retry, 0, which)
    st = tl.where(retry, _ST_HOP, st)
    return st, which, a, b, c, link & (old == hi)


@triton.jit
def _ccl_merge_lanes(img_ptr, parent_ptr, width, height, n, pid, nprog,
                     BLOCK: tl.constexpr, INSTR: tl.constexpr):
    """ccl_fill P1, lane-independent (twin of the same Numba loop as
    _ccl_merge): per red pixel, _union with each red lex-predecessor in
    d order. Returns the per-lane (union_attempts, union_done)."""
    lanes = tl.arange(0, BLOCK)
    stride = nprog * BLOCK
    sx = stride // height
    sy = stride % height
    i = pid * BLOCK + lanes
    x = i // height
    y = i % height
    st = tl.where(i < n, _ST_SCAN, _ST_DONE)
    todo = tl.zeros_like(i)
    which = tl.zeros_like(i)
    a = i
    b = i
    c = i
    union_attempts = tl.zeros([BLOCK], tl.int32)
    union_done = tl.zeros([BLOCK], tl.int32)
    while tl.max(st, 0) > 0:
        for _ in tl.static_range(_LANE_HOPS):
            st, which, a, b, c, same = _union_hop(parent_ptr, st, which, a,
                                                  b, c)
            # same root: this pair is done, start the item's next one
            todo, a, b, c, which, st, start, ended = _take_pair(
                same, todo, i, height, a, b, c, which, st)
            if INSTR:
                union_attempts += start.to(tl.int32)
            i, x, y = _step_index(i, x, y, ended, stride, sx, sy, height)
            st = tl.where(ended, tl.where(i < n, _ST_SCAN, _ST_DONE), st)
            # scan: skip a pixel that is not red, park a red one
            scan = st == _ST_SCAN
            red = _is_red(img_ptr, i, scan)
            st = tl.where(red, _ST_WAIT, st)
            skip = scan & ~red
            i, x, y = _step_index(i, x, y, skip, stride, sx, sy, height)
            st = tl.where(skip & (i >= n), _ST_DONE, st)
        # full step: links, the pair masks of the parked red pixels
        st, which, a, b, c, won = _union_link(parent_ptr, st, which, a, b, c)
        if INSTR:
            union_done += won.to(tl.int32)
        found = st == _ST_WAIT
        todo = tl.where(found,
                        _pred_pairs(img_ptr, x, y, width, height, found),
                        todo)
        todo, a, b, c, which, st, start, ended = _take_pair(
            won | found, todo, i, height, a, b, c, which, st)
        if INSTR:
            union_attempts += start.to(tl.int32)
        i, x, y = _step_index(i, x, y, ended, stride, sx, sy, height)
        st = tl.where(ended, tl.where(i < n, _ST_SCAN, _ST_DONE), st)
    return union_attempts, union_done


@triton.jit
def _ccl_flatten_seed_lanes(img_ptr, visited_ptr, label_ptr, parent_ptr,
                            queue_ptr, q_state_ptr, counters_ptr, n, q_cap,
                            pid, nprog, BLOCK: tl.constexpr):
    """ccl_fill P2, lane-independent: per red pixel, find (from the pixel
    itself), store the label; the root pixel is the blob's seed. The
    epilogue waits for the full step, where the program's seeds append
    with one _cta_enqueue (the lockstep body's construct)."""
    lanes = tl.arange(0, BLOCK)
    stride = nprog * BLOCK
    i = pid * BLOCK + lanes
    st = tl.where(i < n, _ST_SCAN, _ST_DONE)
    c = i
    while tl.max(st, 0) > 0:
        for _ in tl.static_range(_LANE_HOPS):
            hop = st == _ST_HOP
            p = tl.load(parent_ptr + c, mask=hop, other=0)
            st = tl.where(hop & (p == c), _ST_WAIT, st)
            c = tl.where(hop, p, c)
            scan = st == _ST_SCAN
            red = _is_red(img_ptr, i, scan)
            st = tl.where(red, _ST_HOP, st)
            c = tl.where(red, i, c)
            skip = scan & ~red
            i = tl.where(skip, i + stride, i)
            st = tl.where(skip & (i >= n), _ST_DONE, st)
        # full step: label, and the root pixel is the blob's seed
        fin = st == _ST_WAIT
        tl.store(label_ptr + i, c, mask=fin)
        seed = fin & (c == i)
        tl.store(visited_ptr + i, 1, mask=seed)
        _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, i, seed, q_cap)
        i = tl.where(fin, i + stride, i)
        st = tl.where(fin, tl.where(i < n, _ST_SCAN, _ST_DONE), st)


@triton.jit
def _compress_lanes(parent_ptr, n, pid, nprog, BLOCK: tl.constexpr):
    """COMPRESS, lane-independent: if parent[i] != i, parent[i] =
    find(i). A lane writes its root the step it reaches it and reads the
    next item's parent slot in the same step."""
    lanes = tl.arange(0, BLOCK)
    stride = nprog * BLOCK
    i = pid * BLOCK + lanes
    st = tl.where(i < n, _ST_SCAN, _ST_DONE)
    c = i
    while tl.max(st, 0) > 0:
        for _ in tl.static_range(_LANE_HOPS):
            hop = st == _ST_HOP
            p = tl.load(parent_ptr + c, mask=hop, other=0)
            fin = hop & (p == c)
            c = tl.where(hop, p, c)
            tl.store(parent_ptr + i, c, mask=fin)
            i = tl.where(fin, i + stride, i)
            st = tl.where(fin, tl.where(i < n, _ST_SCAN, _ST_DONE), st)
            scan = st == _ST_SCAN
            q = tl.load(parent_ptr + i, mask=scan, other=0)
            retired = scan & (q != i)
            st = tl.where(retired, _ST_HOP, st)
            c = tl.where(retired, i, c)
            skip = scan & ~retired
            i = tl.where(skip, i + stride, i)
            st = tl.where(skip & (i >= n), _ST_DONE, st)


@triton.jit
def _merge_level_lanes(img_ptr, visited_ptr, depth_ptr, owner_ptr,
                       parent_ptr, label_ptr, queue_ptr, q_state_ptr,
                       counters_ptr, width, height, q_cap, start, rear, level,
                       pid, nprog, processed, cas_attempts, union_attempts,
                       union_done, union_cycles, BLOCK: tl.constexpr,
                       INSTR: tl.constexpr):
    """The rest of a seed_merge level from slot `start`, lane-independent.
    A lane's items are its grid-stride queue slots (Numba's slot ->
    (block, thread) map, so processed_per_block is unchanged); per item:
    dequeue, then the 8 direction probes in Numba's order, a colliding
    probe running its _union to the end before the next probe.

    Full step: one hop for the lanes in a union, the links, the next
    slot's dequeue, one direction probe per probing lane with the
    program's enqueue (the per-direction _cta_enqueue of the batch
    body). Then _LANE_HOPS hop mini-steps, when at least a quarter of the
    live lanes are in a union (fewer: they climb one hop per full step).
    A lane takes its next slot only while it is fewer than _LANE_WINDOW
    slots ahead of the program's slowest live lane: free-running lanes
    split into a fast group and a stuck tail (on two_sq_2800 at lattice
    1, 38% of the lane-steps were finished lanes waiting), the window
    took that fill from 387 to 121 ms (Numba 83)."""
    lanes = tl.arange(0, BLOCK)
    stride = nprog * BLOCK
    offs = start + pid * BLOCK + lanes
    st = tl.where(offs < rear, _ST_WAIT, _ST_DONE)   # WAIT: dequeue next
    pixel = tl.zeros_like(offs)
    x = pixel
    y = pixel
    lbl = pixel - 1
    d = pixel
    which = pixel
    a = pixel
    b = pixel
    c = pixel
    k = pixel                     # the lane's items taken in this call
    klim = pixel + _LANE_WINDOW
    t0 = tl.zeros([BLOCK], tl.int64)
    live = tl.sum((st > 0).to(tl.int32), 0)
    while live > 0:
        # ---- full step: one hop, then the links
        st, which, a, b, c, same = _union_hop(parent_ptr, st, which, a, b, c)
        st = tl.where(same, _ST_SCAN, st)
        if INSTR:
            union_cycles += tl.where(same, read_clock64(pid) - t0, 0)
        st, which, a, b, c, won = _union_link(parent_ptr, st, which, a, b, c)
        st = tl.where(won, _ST_SCAN, st)
        if INSTR:
            union_done += won.to(tl.int32)
            union_cycles += tl.where(won, read_clock64(pid) - t0, 0)
        # an item with its 8 directions probed: the lane's next slot
        nxt = (st == _ST_SCAN) & (d >= 8) & (k < klim)
        k = tl.where(nxt, k + 1, k)
        offs = tl.where(nxt, offs + stride, offs)
        st = tl.where(nxt, tl.where(offs < rear, _ST_WAIT, _ST_DONE), st)
        # dequeue
        deq = st == _ST_WAIT
        pixel = tl.where(deq, tl.load(queue_ptr + offs, mask=deq, other=0),
                         pixel)
        x = tl.where(deq, pixel // height, x)
        y = tl.where(deq, pixel % height, y)
        lbl = tl.where(deq, tl.load(label_ptr + pixel, mask=deq, other=-1),
                       lbl)
        tl.store(depth_ptr + pixel, level, mask=deq)
        if INSTR:
            tl.store(owner_ptr + pixel, pid.to(tl.int16), mask=deq)
            processed += deq.to(tl.int32)
        d = tl.where(deq, 0, d)
        st = tl.where(deq, _ST_SCAN, st)
        # probe direction d (the batch body's static d loop, per lane):
        # DX8/DY8 + 1 packed two bits per direction
        pr = (st == _ST_SCAN) & (d < 8)
        nx = x + ((_DX8_BITS >> (2 * d)) & 3) - 1
        ny = y + ((_DY8_BITS >> (2 * d)) & 3) - 1
        nlin = nx * height + ny
        probe = _is_red(img_ptr, nlin, _in_bounds(nx, ny, width, height, pr))
        if INSTR:
            cas_attempts += probe.to(tl.int32)
        old = tl.atomic_xchg(visited_ptr + nlin, 1, mask=probe,
                             sem="relaxed", scope="gpu")
        claimed = probe & (old == 0)
        # win: stamp the inherited label BEFORE the entry is dequeueable
        tl.store(label_ptr + nlin, lbl, mask=claimed)
        _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, nlin, claimed,
                     q_cap)
        lost = probe & (old != 0)
        other = tl.load(label_ptr + nlin, mask=lost, other=-1)
        coll = lost & (other >= 0) & (other != lbl)
        d = tl.where(pr, d + 1, d)
        a = tl.where(coll, lbl, a)
        b = tl.where(coll, other, b)
        c = tl.where(coll, lbl, c)
        which = tl.where(coll, 0, which)
        st = tl.where(coll, _ST_HOP, st)
        if INSTR:
            union_attempts += coll.to(tl.int32)
            t0 = tl.where(coll, read_clock64(pid), t0)
        # live lanes and lanes in a union, one reduction (BLOCK < 1024)
        cnt = tl.sum(tl.where(st == _ST_HOP, 1024, 0) + (st > 0).to(tl.int32),
                     0)
        live = cnt % 1024
        klim = (tl.min(tl.where(st > 0, k, 2147483647), 0) + _LANE_WINDOW
                + tl.zeros_like(k))
        # ---- hop mini-steps while a quarter of the live lanes is in one
        if ((cnt // 1024) * 4 >= live) & (live > 0):
            for _ in tl.static_range(_LANE_HOPS):
                st, which, a, b, c, same = _union_hop(parent_ptr, st, which,
                                                      a, b, c)
                st = tl.where(same, _ST_SCAN, st)
                if INSTR:
                    union_cycles += tl.where(same, read_clock64(pid) - t0, 0)
    return processed, cas_attempts, union_attempts, union_done, union_cycles


# The four union-find phases, each in the spelling LANE picks.


@triton.jit
def _ccl_merge_sched(img_ptr, parent_ptr, width, height, n, pid, nprog,
                     BLOCK: tl.constexpr, INSTR: tl.constexpr,
                     LANE: tl.constexpr):
    if LANE:
        ua, ud = _ccl_merge_lanes(img_ptr, parent_ptr, width, height, n, pid,
                                  nprog, BLOCK, INSTR)
    else:
        ua, ud = _ccl_merge(img_ptr, parent_ptr, width, height, n, pid,
                            nprog, BLOCK, INSTR)
    return ua, ud


@triton.jit
def _ccl_flatten_seed_sched(img_ptr, visited_ptr, label_ptr, parent_ptr,
                            queue_ptr, q_state_ptr, counters_ptr, n, q_cap,
                            pid, nprog, BLOCK: tl.constexpr,
                            LANE: tl.constexpr):
    if LANE:
        _ccl_flatten_seed_lanes(img_ptr, visited_ptr, label_ptr, parent_ptr,
                                queue_ptr, q_state_ptr, counters_ptr, n,
                                q_cap, pid, nprog, BLOCK)
    else:
        _ccl_flatten_seed(img_ptr, visited_ptr, label_ptr, parent_ptr,
                          queue_ptr, q_state_ptr, counters_ptr, n, q_cap,
                          pid, nprog, BLOCK)


@triton.jit
def _compress_sched(parent_ptr, n, pid, nprog, BLOCK: tl.constexpr,
                    LANE: tl.constexpr):
    if LANE:
        _compress_lanes(parent_ptr, n, pid, nprog, BLOCK)
    else:
        _compress(parent_ptr, n, pid, nprog, BLOCK)


# ---------------------------------------------------------------- ccl_fill


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def ccl_fill_kernel(img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr,
                    label_ptr, queue_ptr, q_state_ptr, counters_ptr,
                    block_stats_ptr, level_sizes_ptr, phase_ptr, palette_ptr,
                    bar_ptr, width, height, n, q_cap, trace_cap,
                    BLOCK: tl.constexpr, INSTR: tl.constexpr,
                    LANE: tl.constexpr):
    """Twin of ccl_fill_kernel (INSTR=True) and ccl_fill_bare_kernel
    (INSTR=False; owner/block_stats/level_sizes/phase are None). LANE
    picks the union-find spelling of P1 and P2 (module doc).

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

    union_attempts, union_done = _ccl_merge_sched(
        img_ptr, parent_ptr, width, height, n, pid, nprog, BLOCK, INSTR, LANE)
    epoch = _sync(bar_ptr, epoch, nprog)
    if INSTR:
        _stamp(phase_ptr, 2, pid)

    _ccl_flatten_seed_sched(img_ptr, visited_ptr, label_ptr, parent_ptr,
                            queue_ptr, q_state_ptr, counters_ptr, n, q_cap,
                            pid, nprog, BLOCK, LANE)
    rear, epoch = _fence_sandwich(bar_ptr, q_state_ptr, phase_ptr, epoch, pid,
                                  nprog, 3, INSTR)
    n_seeds = rear

    (level, epoch, peak_level, peak_occ, active_thread_sum, active_warp_sum,
     processed, cas_attempts, _ua, _ud, _uc) = _fill_levels(
        img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr, label_ptr,
        queue_ptr, q_state_ptr, counters_ptr, level_sizes_ptr, palette_ptr,
        bar_ptr, width, height, q_cap, trace_cap, rear, epoch, pid, nprog,
        BLOCK, False, INSTR, False)
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
                      INSTR: tl.constexpr, LANE: tl.constexpr):
    """Twin of seed_merge_kernel (INSTR=True) and seed_merge_bare_kernel
    (INSTR=False; owner/prov/block_stats/level_sizes/phase are None). LANE
    picks the union-find spelling of the fill's unions and of P3.

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
        BLOCK, True, INSTR, LANE)
    if INSTR:
        _stamp(phase_ptr, 3, pid)

    # P3: the loop's closing barrier pair ordered every union before here
    # (lockstep find in both schedules: _merge_flatten's docstring)
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


# ----------------------------------------------- seed_merge_lat (v2 twins)
# The seeding-density experiment (see the Numba kernels.py block comment):
# seed_merge with (1) the lattice P1 rule, (2) a verbatim P2, (3) a
# COMPRESS pass + barrier between the fill and the flatten.


@triton.jit
def _lattice_scan(img_ptr, visited_ptr, label_ptr, queue_ptr, q_state_ptr,
                  counters_ptr, width, height, n, q_cap, lat_stride,
                  lat_interior, pid, nprog, BLOCK: tl.constexpr):
    """seed_merge_lat P1: corner rule OR (lattice point [AND interior]).

    Numba short-circuits `lat_stride > 0 and x % lat_stride == 0 and ...`;
    lanes evaluate every operand, so the modulo divides by
    max(lat_stride, 1) and the hit mask carries lat_stride > 0. The
    _is_interior probes run only on lattice hits when lat_interior is set,
    each one narrowing the mask (its early return False); the corner
    probes run only on red pixels that are not candidates yet (Numba's
    `if not is_cand`)."""
    lanes = tl.arange(0, BLOCK)
    s = tl.maximum(lat_stride, 1)
    for base in range(0, n, nprog * BLOCK):
        i = base + pid * BLOCK + lanes
        m = i < n
        x = i // height
        y = i % height
        red = _is_red(img_ptr, i, m)
        hit = red & (lat_stride > 0) & (x % s == 0) & (y % s == 0)
        inner = hit & (lat_interior != 0)
        for d in tl.static_range(8):
            nx = x + _DX8[d]
            ny = y + _DY8[d]
            inner = _is_red(img_ptr, nx * height + ny,
                            _in_bounds(nx, ny, width, height, inner))
        cand = hit & ((lat_interior == 0) | inner)
        rest = red & ~cand
        found = tl.zeros_like(red)
        for d in tl.static_range(4):
            nx = x + _PDX[d]
            ny = y + _PDY[d]
            found = found | _is_red(img_ptr, nx * height + ny,
                                    _in_bounds(nx, ny, width, height, rest))
        cand = cand | (rest & ~found)
        tl.store(visited_ptr + i, 1, mask=cand)
        tl.store(label_ptr + i, i, mask=cand)
        _cta_enqueue(queue_ptr, q_state_ptr, counters_ptr, i, cand, q_cap)


@triton.jit
def _compress(parent_ptr, n, pid, nprog, BLOCK: tl.constexpr):
    """COMPRESS: rewrite every retired parent slot to its true root.
    Post-fill the roots are static; a concurrent chase that reads a fresh
    write only shortcuts, so one pass fully flattens."""
    lanes = tl.arange(0, BLOCK)
    for base in range(0, n, nprog * BLOCK):
        i = base + pid * BLOCK + lanes
        m = i < n
        retired = m & (tl.load(parent_ptr + i, mask=m, other=0) != i)
        root = _find(parent_ptr, i, retired)
        tl.store(parent_ptr + i, root, mask=retired)


_LAT_INTS = _RUNTIME_INTS + ["lat_stride", "lat_interior"]


@triton.jit(do_not_specialize=_LAT_INTS)
def seed_merge_lat_kernel(img_ptr, visited_ptr, depth_ptr, owner_ptr,
                          parent_ptr, label_ptr, prov_ptr, queue_ptr,
                          q_state_ptr, counters_ptr, block_stats_ptr,
                          level_sizes_ptr, phase_ptr, palette_ptr, bar_ptr,
                          width, height, n, q_cap, trace_cap, lat_stride,
                          lat_interior, BLOCK: tl.constexpr,
                          INSTR: tl.constexpr, LANE: tl.constexpr):
    """Twin of seed_merge_lat_kernel (INSTR=True), seed_merge_lat_bare_kernel
    (INSTR=False; owner/prov/block_stats/level_sizes/phase are None) and,
    launched with maxnreg=128, seed_merge_lat_r128_kernel (Numba compiles
    the same py_func with max_registers=128).

    Host contract as seed_merge_kernel plus lat_stride (int >= 0) and
    lat_interior (0/1)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.full((), 0, tl.int64)

    if INSTR:
        _stamp(phase_ptr, 0, pid)
    _iota(parent_ptr, n, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    if INSTR:
        _stamp(phase_ptr, 1, pid)

    _lattice_scan(img_ptr, visited_ptr, label_ptr, queue_ptr, q_state_ptr,
                  counters_ptr, width, height, n, q_cap, lat_stride,
                  lat_interior, pid, nprog, BLOCK)
    rear, epoch = _fence_sandwich(bar_ptr, q_state_ptr, phase_ptr, epoch, pid,
                                  nprog, 2, INSTR)
    n_candidates = rear

    # P2: verbatim seed_merge
    (level, epoch, peak_level, peak_occ, active_thread_sum, active_warp_sum,
     processed, cas_attempts, union_attempts, union_done,
     union_cycles) = _fill_levels(
        img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr, label_ptr,
        queue_ptr, q_state_ptr, counters_ptr, level_sizes_ptr, palette_ptr,
        bar_ptr, width, height, q_cap, trace_cap, rear, epoch, pid, nprog,
        BLOCK, True, INSTR, LANE)
    if INSTR:
        _stamp(phase_ptr, 3, pid)

    # COMPRESS (delta 3), then the barrier the flatten's <= 1-hop find
    # relies on (in both twins)
    _compress_sched(parent_ptr, n, pid, nprog, BLOCK, LANE)
    epoch = _sync(bar_ptr, epoch, nprog)
    if INSTR:
        _stamp(phase_ptr, 4, pid)

    # P3: flatten + relabel + repaint - find is now <= 1 hop
    _merge_flatten(img_ptr, label_ptr, prov_ptr, parent_ptr, palette_ptr, n,
                   pid, nprog, BLOCK, INSTR)
    if INSTR:
        epoch = _sync(bar_ptr, epoch, nprog)
        _stamp(phase_ptr, 5, pid)

    _exit_counters(counters_ptr, block_stats_ptr, q_state_ptr, pid, level,
                   peak_level, peak_occ, active_thread_sum, active_warp_sum,
                   processed, cas_attempts, union_attempts, union_done,
                   union_cycles, n_candidates, INSTR, True)


# ------------------------------------------- register-experiment builds
# r128 is seed_merge_lat_kernel launched with maxnreg=128 (driver). split:
# the cooperative core keeps only what needs grid barriers (P0-P2);
# compress and flatten move to two plain kernels, ordered by the stream.


@triton.jit(do_not_specialize=_LAT_INTS)
def seed_merge_lat_core_kernel(img_ptr, visited_ptr, depth_ptr, owner_ptr,
                               parent_ptr, label_ptr, queue_ptr, q_state_ptr,
                               counters_ptr, block_stats_ptr, level_sizes_ptr,
                               phase_ptr, bar_ptr, width, height, n, q_cap,
                               trace_cap, lat_stride, lat_interior,
                               BLOCK: tl.constexpr, LANE: tl.constexpr):
    """Twin of seed_merge_lat_core_kernel: no prov_label and no palette
    (nothing is painted here). The host must follow this launch with
    lat_compress_kernel then lat_finish_kernel on the same stream."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.full((), 0, tl.int64)

    _stamp(phase_ptr, 0, pid)
    _iota(parent_ptr, n, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    _stamp(phase_ptr, 1, pid)

    _lattice_scan(img_ptr, visited_ptr, label_ptr, queue_ptr, q_state_ptr,
                  counters_ptr, width, height, n, q_cap, lat_stride,
                  lat_interior, pid, nprog, BLOCK)
    rear, epoch = _fence_sandwich(bar_ptr, q_state_ptr, phase_ptr, epoch, pid,
                                  nprog, 2, True)
    n_candidates = rear

    (level, epoch, peak_level, peak_occ, active_thread_sum, active_warp_sum,
     processed, cas_attempts, union_attempts, union_done,
     union_cycles) = _fill_levels(
        img_ptr, visited_ptr, depth_ptr, owner_ptr, parent_ptr, label_ptr,
        queue_ptr, q_state_ptr, counters_ptr, level_sizes_ptr, None,
        bar_ptr, width, height, q_cap, trace_cap, rear, epoch, pid, nprog,
        BLOCK, True, True, LANE)
    _stamp(phase_ptr, 3, pid)

    _exit_counters(counters_ptr, block_stats_ptr, q_state_ptr, pid, level,
                   peak_level, peak_occ, active_thread_sum, active_warp_sum,
                   processed, cas_attempts, union_attempts, union_done,
                   union_cycles, n_candidates, True, True)


@triton.jit(do_not_specialize=["n"])
def lat_compress_kernel(parent_ptr, n, BLOCK: tl.constexpr,
                        LANE: tl.constexpr):
    """Twin of lat_compress_kernel: plain grid-stride compression (no
    barrier; n = parent.shape[0])."""
    _compress_sched(parent_ptr, n, tl.program_id(0), tl.num_programs(0),
                    BLOCK, LANE)


@triton.jit(do_not_specialize=["n"])
def lat_finish_kernel(img_ptr, parent_ptr, label_ptr, prov_ptr, palette_ptr,
                      n, BLOCK: tl.constexpr):
    """Twin of lat_finish_kernel: plain grid-stride prov snapshot, resolve
    (<= 1 hop after lat_compress_kernel), relabel, paint. Must launch
    after lat_compress_kernel on the same stream."""
    _merge_flatten(img_ptr, label_ptr, prov_ptr, parent_ptr, palette_ptr, n,
                   tl.program_id(0), tl.num_programs(0), BLOCK, True)


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
               BLOCK: tl.constexpr, LANE: tl.constexpr):
    """Twin of ccl_kernel: ccl_fill P0-P2, then the rear count (n_blobs)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    epoch = tl.full((), 0, tl.int64)
    _iota(parent_ptr, n, pid, nprog, BLOCK)
    epoch = _sync(bar_ptr, epoch, nprog)
    _ccl_merge_sched(img_ptr, parent_ptr, width, height, n, pid, nprog, BLOCK,
                     False, LANE)
    epoch = _sync(bar_ptr, epoch, nprog)
    _ccl_flatten_seed_sched(img_ptr, visited_ptr, label_ptr, parent_ptr,
                            queue_ptr, q_state_ptr, counters_ptr, n, q_cap,
                            pid, nprog, BLOCK, LANE)
    epoch = _sync(bar_ptr, epoch, nprog)
    if pid == 0:
        tl.store(counters_ptr + _CANDIDATES,
                 tl.load(q_state_ptr + _Q_REAR).to(tl.int64))
