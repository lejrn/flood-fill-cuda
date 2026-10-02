"""Triton twin of chapter 6's kernels: runs, not pixels.

Same pipeline, same data structures and the same counters as
chapters/ch06_gpu_nblob_runs/kernels.py. That module's docstring is the
explanation of WHY the pipeline looks like this (runs as the unit of
work, the run-start/run-end bit algebra, the canonical label carried by
the run order). This one documents HOW each CUDA construct was carried
over to Triton.

    pack     RGB -> 1 bit/px          (the only full-resolution read)
    count    runs per row
    scan     row -> run offsets, plus the counter reset
    emit     the run table, ordered by (x, y0), plus the parent iota
    merge    atomicMin union-find over vertically adjacent runs
    flatten  path-compress, canonical label per run, count the roots
    paint    one warp per run, lanes striding the span
    label    the per-pixel label map (verification only)

MAPPING. One Numba block of T threads is one Triton program of T lanes
with num_warps = T // 32, and every grid is the same grid. The
warp-centric kernels (pack, unpack, count, emit, paint, label) work on a
[T // 32, 32] tile: row i of the tile is warp i of the block and column
l is lane l, so `warp_id = pid * WPB + i` and every grid-stride loop
strides exactly as the Numba one does. The thread-per-run kernels
(merge, flatten) work on a flat [T] tensor, lane i = thread i.

CUDA constructs Triton cannot spell, and what stands in for them:

    ballot_sync           OR-reduction of (red << lane) along the lane axis
    shfl_up warp scan     tl.cumsum / tl.sum along the lane axis
    shfl_down reduction   tl.sum along the lane axis
    shfl_sync broadcast   a masked sum that picks lane k's value
    popc, ffs             libdevice popc / ffs on the bit-cast word
    shared-memory scan    tl.cumsum over a [1024] register tensor
    divergent loops       lockstep loops over a lane mask,
                          `while tl.max(active) > 0`

On sm_80+ an integer OR or sum along the 32-lane axis compiles to ONE
`redux.sync` warp instruction (the PTX of pack, count and paint has no
shuffles for them), so the ballot and the broadcast stay one warp
instruction each.

The last row is the one real difference. A CUDA warp runs a
data-dependent loop until its slowest lane is done, and the warps of a
block never wait for each other. A Triton program has one control flow,
so the loop runs until the slowest lane of the whole PROGRAM is done,
with finished lanes masked off. Every lane performs the same operations
on the same memory as its Numba thread; only the divergence unit is
coarser (a block instead of a warp).

WORDS. The mask is uint32 and stays uint32 in every kernel, so `>>` is a
logical shift. Numba widens each word to int64 for the same reason: an
int32 word would sign-extend `prev >> 31` and corrupt every run start.

SHAPES. Numba kernels read width and height from their arrays. Triton
kernels get raw pointers, so the sizes are explicit int arguments, all
marked do_not_specialize: Triton would otherwise compile a new kernel
for every size that is 1 or divisible by 16, and a compile inside a
timed window is seconds. Image offsets are formed in int64:
(x*height + y)*3 passes 2**31 above ~715 Mpx, which the driver allows.
"""

import os

# Must be set before numba is imported (the counter slots and the
# palette come from the Numba modules, so a divergence is impossible).
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from flood_fill_cuda.chapters.ch06_gpu_nblob_runs.kernels import (  # noqa: F401
    PALETTE_HOST, N_PALETTE, WARP, SCAN_TPB, NUM_COUNTERS,
    RUN_OVERFLOW, N_RUNS, N_BLOBS, UNION_ATTEMPTS, UNION_DONE, N_RUNS_USED,
)

# Triton kernels may only read module globals that are constexpr.
_WARP = tl.constexpr(WARP)
_N_PALETTE = tl.constexpr(N_PALETTE)
_RUN_OVERFLOW = tl.constexpr(RUN_OVERFLOW)
_N_RUNS = tl.constexpr(N_RUNS)
_N_BLOBS = tl.constexpr(N_BLOBS)
_UNION_ATTEMPTS = tl.constexpr(UNION_ATTEMPTS)
_UNION_DONE = tl.constexpr(UNION_DONE)
_N_RUNS_USED = tl.constexpr(N_RUNS_USED)


# --------------------------------------------------------------- helpers

@triton.jit
def _popc(word):
    """cuda.popc on a uint32 word."""
    return libdevice.popc(word.to(tl.int32, bitcast=True))


@triton.jit
def _ffs(word):
    """cuda.ffs on a uint32 word: 1-based index of the lowest set bit."""
    return libdevice.ffs(word.to(tl.int32, bitcast=True))


@triton.jit
def _or(a, b):
    return a | b


@triton.jit
def _is_red(img_ptr, pix, valid):
    """Is the pixel at byte offset `pix` pure red (255, 0, 0)?

    Three INDEPENDENT loads combined with `&`, as in the Numba kernel
    (its docstring has the measurement: a short-circuit `and` serialises
    three memory round trips). A masked-off lane reads R = 0: not red.
    """
    r = tl.load(img_ptr + pix, mask=valid, other=0)
    g = tl.load(img_ptr + pix + 1, mask=valid, other=0)
    b = tl.load(img_ptr + pix + 2, mask=valid, other=0)
    return (r == 255) & (g == 0) & (b == 0)


@triton.jit
def _word(mask_ptr, row, w, words_per_row, valid):
    """mask[x, w] as uint32 (row = x * words_per_row), or 0 outside the
    row or on a masked-off lane."""
    m = valid & (w >= 0) & (w < words_per_row)
    return tl.load(mask_ptr + row + w, mask=m, other=0)


@triton.jit
def _starts_ends(mask_ptr, row, w, words_per_row, valid):
    """(starts, ends) bitmasks for word w of the row: the algebra in the
    Numba module docstring, out-of-row neighbours read as 0."""
    cur = _word(mask_ptr, row, w, words_per_row, valid)
    prev = _word(mask_ptr, row, w - 1, words_per_row, valid)
    nxt = _word(mask_ptr, row, w + 1, words_per_row, valid)
    starts = cur & ~((cur << 1) | (prev >> 31))
    ends = cur & ~((cur >> 1) | ((nxt & 1) << 31))
    return starts, ends


@triton.jit
def _find(parent_ptr, i, act):
    """Root of each active lane's run, HALVING the path on the way
    (parent[i] gets its grandparent): Numba's `_find`, lane for lane.

    The Numba loop is per thread; here a lane drops out of the mask when
    it reaches its root and the loop ends when every lane has. The
    halving store is the same racy plain store, safe for the reason the
    Numba docstring gives (a same-class index below i). Inactive lanes
    return i unchanged.
    """
    root = i
    while tl.max(act.to(tl.int32), axis=0) > 0:
        p = tl.load(parent_ptr + i, mask=act, other=0)
        at_root = act & (p == i)
        step = act & (p != i)
        gp = tl.load(parent_ptr + p, mask=step, other=0)
        at_parent = step & (gp == p)
        halve = step & (gp != p)
        tl.store(parent_ptr + i, gp, mask=halve)
        root = tl.where(at_root, i, tl.where(at_parent, p, root))
        i = tl.where(halve, gp, i)
        act = halve
    return root


@triton.jit
def _union(parent_ptr, a, b, act):
    """ch05's atomicMin union over run ids, Numba's `_union` lane for
    lane: 1 on the lanes whose call retired a root, else 0.

    Per lane: find both roots; equal roots return 0; otherwise the
    larger root is linked under the smaller with one atomicMin, which
    succeeds iff it returns the larger root itself. On failure the lane
    retries from the value the atomic returned (`a = old`), so its
    working index strictly decreases, which is why it terminates.
    """
    linked = tl.zeros_like(a)
    while tl.max(act.to(tl.int32), axis=0) > 0:
        a = _find(parent_ptr, a, act)
        b = _find(parent_ptr, b, act)
        act = act & (a != b)
        hi = tl.maximum(a, b)           # `if a < b: a, b = b, a`
        lo = tl.minimum(a, b)
        old = tl.atomic_min(parent_ptr + hi, lo, mask=act,
                            sem="relaxed", scope="gpu")
        linked += (act & (old == hi)).to(tl.int32)
        act = act & (old != hi)
        a = tl.where(act, old, hi)
        b = lo
    return linked


# ------------------------------------------------------------------ pack
# One warp per 32-pixel word, on the same 2D grid: words along axis 0
# (WPB words per program), rows along axis 1, capped and strided.

@triton.jit(do_not_specialize=["width", "height", "words_per_row"])
def pack_kernel(img_ptr, mask_ptr, width, height, words_per_row,
                WPB: tl.constexpr):
    """img (width, height, 3) uint8 -> mask (width, words_per_row) uint32.

    Bits past `height` in a row's last word are written as 0. The ballot
    is an OR-reduction of each lane's vote shifted to its bit position:
    the same word. It lowers to one `redux.sync.or.b32` per warp (sm_80+,
    checked in the PTX), a single warp instruction like the vote.
    """
    w = tl.program_id(0) * WPB + tl.arange(0, WPB)      # warp i -> word w
    lane = tl.arange(0, _WARP)[None, :]
    y = w[:, None] * _WARP + lane
    valid = (w[:, None] < words_per_row) & (y < height)
    shift = lane.to(tl.uint32)

    for x in range(tl.program_id(1), width, tl.num_programs(1)):
        pix = (x.to(tl.int64) * height + y) * 3
        red = _is_red(img_ptr, pix, valid)
        bits = tl.reduce(red.to(tl.uint32) << shift, 1, _or)
        tl.store(mask_ptr + x * words_per_row + w, bits,
                 mask=w < words_per_row)


@triton.jit(do_not_specialize=["width", "height", "words_per_row"])
def unpack_kernel(mask_ptr, img_ptr, width, height, words_per_row,
                  WPB: tl.constexpr):
    """mask -> a white image with pure-red runs (the inverse of pack)."""
    w = tl.program_id(0) * WPB + tl.arange(0, WPB)
    lane = tl.arange(0, _WARP)[None, :]
    y = w[:, None] * _WARP + lane
    valid = (w[:, None] < words_per_row) & (y < height)
    shift = lane.to(tl.uint32)
    white = tl.full([WPB, _WARP], 255, tl.uint8)

    for x in range(tl.program_id(1), width, tl.num_programs(1)):
        word = tl.load(mask_ptr + x * words_per_row + w,
                       mask=w < words_per_row, other=0)
        bit = (word[:, None] >> shift) & 1
        gb = tl.where(bit != 0, 0, 255).to(tl.uint8)
        pix = (x.to(tl.int64) * height + y) * 3
        tl.store(img_ptr + pix, white, mask=valid)
        tl.store(img_ptr + pix + 1, gb, mask=valid)
        tl.store(img_ptr + pix + 2, gb, mask=valid)


# ----------------------------------------------------------------- count

@triton.jit(do_not_specialize=["width", "words_per_row"])
def count_kernel(mask_ptr, row_count_ptr, width, words_per_row,
                 WPB: tl.constexpr):
    """row_count[x] = number of maximal red runs in row x (counts STARTS,
    reading two words per word). Warp i of the tile owns one row; the
    shfl_down reduction is a sum along the lane axis."""
    wi = tl.arange(0, WPB)
    lane = tl.arange(0, _WARP)[None, :]
    n_warps = tl.num_programs(0) * WPB

    for x0 in range(tl.program_id(0) * WPB, width, n_warps):
        x = x0 + wi
        xv = (x < width)[:, None]
        row = (x * words_per_row)[:, None]
        total = tl.zeros([WPB, _WARP], tl.int32)
        for base in range(0, words_per_row, _WARP):
            w = base + lane
            inw = xv & (w < words_per_row)
            cur = _word(mask_ptr, row, w, words_per_row, inw)
            prev = _word(mask_ptr, row, w - 1, words_per_row, inw)
            starts = cur & ~((cur << 1) | (prev >> 31))
            total += _popc(starts)
        tl.store(row_count_ptr + x, tl.sum(total, axis=1), mask=x < width)


@triton.jit(do_not_specialize=["width", "capacity"])
def row_scan_kernel(row_count_ptr, row_off_ptr, counters_ptr, width,
                    capacity, BLOCK: tl.constexpr):
    """Exclusive prefix sum of row_count into row_off (width+1 entries),
    plus the per-run counter reset. ONE program of SCAN_TPB lanes walks
    the rows in chunks with a carry, as the single Numba block does; the
    Hillis-Steele scan over shared memory is tl.cumsum over the chunk.
    """
    tid = tl.arange(0, BLOCK)
    carry = tl.zeros([], tl.int32)
    for base in range(0, width, BLOCK):
        m = base + tid < width
        v = tl.load(row_count_ptr + base + tid, mask=m, other=0)
        incl = tl.cumsum(v, axis=0)
        tl.store(row_off_ptr + base + tid, carry + incl - v, mask=m)
        carry += tl.sum(v, axis=0)

    tl.store(row_off_ptr + width, carry)
    tl.store(counters_ptr + _N_RUNS, carry.to(tl.int64))
    tl.store(counters_ptr + _N_RUNS_USED,
             tl.minimum(carry, capacity).to(tl.int64))
    tl.store(counters_ptr + _RUN_OVERFLOW, (carry > capacity).to(tl.int64))
    # The reset of the counters the later phases accumulate into, here
    # for the reason the Numba kernel gives: a host-side clear is a
    # synchronous copy, and nothing may touch the host mid-pipeline.
    tl.store(counters_ptr + _N_BLOBS, tl.zeros([], tl.int64))
    tl.store(counters_ptr + _UNION_ATTEMPTS, tl.zeros([], tl.int64))
    tl.store(counters_ptr + _UNION_DONE, tl.zeros([], tl.int64))


# ------------------------------------------------------------------ emit

@triton.jit(do_not_specialize=["width", "words_per_row", "capacity"])
def emit_kernel(mask_ptr, row_off_ptr, run_x_ptr, run_y0_ptr, run_y1_ptr,
                parent_ptr, counters_ptr, width, words_per_row, capacity,
                WPB: tl.constexpr):
    """Fill the run table: run_x/run_y0/run_y1[k] (y1 INCLUSIVE),
    ordered by (x, y0), and parent[k] = k.

    Warp i of the tile owns one row; lane l holds word base + l. The
    warp scans of popc(starts) and popc(ends) are cumsums along the lane
    axis, so slot k of every run is the Numba slot. The bit walk is the
    Numba `ffs` / `s &= s - 1` loop, run for the largest popc in the
    program (a warp runs it for the largest popc in the warp).
    """
    wi = tl.arange(0, WPB)[:, None]
    lane = tl.arange(0, _WARP)[None, :]
    n_warps = tl.num_programs(0) * WPB
    zero = tl.zeros([WPB, _WARP], tl.int32)
    flag = counters_ptr + _RUN_OVERFLOW + zero

    for x0 in range(tl.program_id(0) * WPB, width, n_warps):
        x = x0 + wi                                  # [WPB, 1]
        xv = x < width
        row = x * words_per_row
        xs = x + zero                                # x on every lane
        base_slot = tl.load(row_off_ptr + x, mask=xv, other=0)
        cur_start = tl.zeros([WPB, 1], tl.int32)
        cur_end = tl.zeros([WPB, 1], tl.int32)
        for base in range(0, words_per_row, _WARP):
            w = base + lane
            starts, ends = _starts_ends(mask_ptr, row, w, words_per_row,
                                        xv & (w < words_per_row))
            n_s = _popc(starts)
            n_e = _popc(ends)
            incl_s = tl.cumsum(n_s, axis=1)
            incl_e = tl.cumsum(n_e, axis=1)

            k = base_slot + cur_start + incl_s - n_s
            s = starts
            for _ in range(0, tl.max(n_s)):
                act = s != 0
                ok = act & (k < capacity)
                tl.store(run_x_ptr + k, xs, mask=ok)
                tl.store(run_y0_ptr + k, w * _WARP + _ffs(s) - 1, mask=ok)
                tl.store(parent_ptr + k, k, mask=ok)
                tl.store(flag, 1, mask=act & (k >= capacity))
                k += act.to(tl.int32)
                s = s & (s - 1)

            k = base_slot + cur_end + incl_e - n_e
            e = ends
            for _ in range(0, tl.max(n_e)):
                act = e != 0
                tl.store(run_y1_ptr + k, w * _WARP + _ffs(e) - 1,
                         mask=act & (k < capacity))
                k += act.to(tl.int32)
                e = e & (e - 1)

            cur_start += tl.sum(n_s, axis=1, keep_dims=True)
            cur_end += tl.sum(n_e, axis=1, keep_dims=True)


# ----------------------------------------------------------------- merge

@triton.jit(do_not_specialize=["capacity", "width"])
def merge_rows_kernel(run_x_ptr, run_y0_ptr, run_y1_ptr, row_off_ptr,
                      parent_ptr, counters_ptr, capacity, width,
                      INSTRUMENTED: tl.constexpr, BLOCK: tl.constexpr):
    """Union every run with the runs it touches in the row below.

    Thread-per-run, grid-stride over counters[N_RUNS_USED] read on the
    device. Per lane: the Numba `continue` on the last row is a mask,
    the lower-bound binary search and the forward walk are lockstep
    loops, and each walk step is one `_union` (attempts and successful
    links counted per lane, then one relaxed atomic per lane with a
    nonzero count, as each Numba thread issues its own).
    """
    n_runs = tl.load(counters_ptr + _N_RUNS_USED).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    stride = tl.num_programs(0) * BLOCK
    attempts = tl.zeros([BLOCK], tl.int32)
    done = tl.zeros([BLOCK], tl.int32)

    for r0 in range(tl.program_id(0) * BLOCK, n_runs, stride):
        r = r0 + lane
        valid = r < n_runs
        x = tl.load(run_x_ptr + r, mask=valid, other=0)
        valid = valid & (x + 1 < width)
        y0 = tl.load(run_y0_ptr + r, mask=valid, other=0)
        y1 = tl.load(run_y1_ptr + r, mask=valid, other=0)
        lo = tl.minimum(tl.load(row_off_ptr + x + 1, mask=valid, other=0),
                        capacity)
        hi = tl.minimum(tl.load(row_off_ptr + x + 2, mask=valid, other=0),
                        capacity)
        # first run of the next row whose end reaches y0-1 (8-conn)
        a = lo
        b = hi
        search = valid & (a < b)
        while tl.max(search.to(tl.int32), axis=0) > 0:
            m = (a + b) >> 1
            below = tl.load(run_y1_ptr + m, mask=search, other=0) < y0 - 1
            a = tl.where(search & below, m + 1, a)
            b = tl.where(search & ~below, m, b)
            search = search & (a < b)
        s = a
        walk = valid & (s < hi)
        walk = walk & (tl.load(run_y0_ptr + s, mask=walk, other=0) <= y1 + 1)
        while tl.max(walk.to(tl.int32), axis=0) > 0:
            attempts += walk.to(tl.int32)
            done += _union(parent_ptr, r, s, walk)
            s += walk.to(tl.int32)
            walk = walk & (s < hi)
            walk = walk & (tl.load(run_y0_ptr + s, mask=walk, other=0)
                           <= y1 + 1)

    if INSTRUMENTED:
        # One atomic per THREAD with a nonzero count, as Numba issues
        # them (`if attempts: cuda.atomic.add(...)`): the same-address
        # traffic is part of what the instrumented variant costs.
        slot = tl.zeros([BLOCK], tl.int32)
        tl.atomic_add(counters_ptr + _UNION_ATTEMPTS + slot,
                      attempts.to(tl.int64), mask=attempts > 0,
                      sem="relaxed", scope="gpu")
        tl.atomic_add(counters_ptr + _UNION_DONE + slot, done.to(tl.int64),
                      mask=done > 0, sem="relaxed", scope="gpu")


# --------------------------------------------------------------- flatten

@triton.jit(do_not_specialize=["height"])
def flatten_kernel(parent_ptr, run_x_ptr, run_y0_ptr, run_label_ptr, height,
                   counters_ptr, INSTRUMENTED: tl.constexpr,
                   BLOCK: tl.constexpr):
    """Path-compress every run to its root, resolve its canonical label
    (run_x[root]*height + run_y0[root], the scattered gathers paid here
    and not in paint), and count the survivors (per lane, one atomic per
    lane that found a root, as in Numba)."""
    n_runs = tl.load(counters_ptr + _N_RUNS_USED).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    stride = tl.num_programs(0) * BLOCK
    roots = tl.zeros([BLOCK], tl.int32)

    for r0 in range(tl.program_id(0) * BLOCK, n_runs, stride):
        r = r0 + lane
        valid = r < n_runs
        root = _find(parent_ptr, r, valid)
        tl.store(parent_ptr + r, root, mask=valid)
        rx = tl.load(run_x_ptr + root, mask=valid, other=0)
        ry = tl.load(run_y0_ptr + root, mask=valid, other=0)
        tl.store(run_label_ptr + r, rx * height + ry, mask=valid)
        roots += (valid & (root == r)).to(tl.int32)

    if INSTRUMENTED:
        # One atomic per thread with a nonzero count, as in Numba.
        tl.atomic_add(counters_ptr + _N_BLOBS + tl.zeros([BLOCK], tl.int32),
                      roots.to(tl.int64), mask=roots > 0,
                      sem="relaxed", scope="gpu")


# ----------------------------------------------------------------- paint

@triton.jit(do_not_specialize=["height"])
def paint_kernel(img_ptr, run_x_ptr, run_y0_ptr, run_y1_ptr, run_label_ptr,
                 counters_ptr, palette_ptr, height, WPB: tl.constexpr):
    """Recolor every run's span: one warp per run, lanes striding y.

    Each warp fetches 32 run descriptors cooperatively (lane l takes run
    r0 + l: four coalesced loads) and replays them one at a time, as the
    Numba kernel does with shfl_sync. The broadcast of lane k's values
    is a masked sum along the lane axis, one `redux.sync.add` per value
    (one warp instruction, as the shuffle is). The span loop's trip count for
    replay step k is the largest span among the program's warps at that
    step; it is computed once per 32-run group (one cross-warp max per
    group) and read back per step from a [32] vector.

    The variants the chapter measured and rejected (word stores,
    per-run descriptor loads, skipping unchanged channels) stay out.
    """
    n_runs = tl.load(counters_ptr + _N_RUNS_USED).to(tl.int32)
    wi = tl.arange(0, WPB)[:, None]
    lane1 = tl.arange(0, _WARP)
    lane = lane1[None, :]
    n_warps = tl.num_programs(0) * WPB

    for g in range(tl.program_id(0) * WPB * _WARP, n_runs, n_warps * _WARP):
        r0 = g + wi * _WARP                          # warp i's first run
        r = r0 + lane
        inb = r < n_runs
        my_x = tl.load(run_x_ptr + r, mask=inb, other=0)
        my_y0 = tl.load(run_y0_ptr + r, mask=inb, other=0)
        my_y1 = tl.load(run_y1_ptr + r, mask=inb, other=0)
        my_lab = tl.load(run_label_ptr + r, mask=inb, other=0)
        chunks = tl.where(inb, (my_y1 - my_y0 + _WARP) >> 5, 0)
        step_chunks = tl.max(chunks, axis=0)         # [32], per replay step
        kmax = tl.minimum(n_runs - g, _WARP)         # warp 0 has the most

        for k in range(0, kmax):
            pick = lane == k
            x = tl.sum(tl.where(pick, my_x, 0), axis=1, keep_dims=True)
            y0 = tl.sum(tl.where(pick, my_y0, 0), axis=1, keep_dims=True)
            y1 = tl.sum(tl.where(pick, my_y1, 0), axis=1, keep_dims=True)
            label = tl.sum(tl.where(pick, my_lab, 0), axis=1, keep_dims=True)
            live = r0 + k < n_runs                   # [WPB, 1]
            c = (label % _N_PALETTE) * 3
            p0 = tl.load(palette_ptr + c, mask=live, other=0)
            p1 = tl.load(palette_ptr + c + 1, mask=live, other=0)
            p2 = tl.load(palette_ptr + c + 2, mask=live, other=0)
            p0 = tl.broadcast_to(p0, [WPB, _WARP])
            p1 = tl.broadcast_to(p1, [WPB, _WARP])
            p2 = tl.broadcast_to(p2, [WPB, _WARP])
            row = x.to(tl.int64) * height
            n_chunks = tl.sum(tl.where(lane1 == k, step_chunks, 0), axis=0)
            for ci in range(0, n_chunks):
                y = y0 + ci * _WARP + lane
                m = live & (y <= y1)
                pix = (row + y) * 3
                tl.store(img_ptr + pix, p0, mask=m)
                tl.store(img_ptr + pix + 1, p1, mask=m)
                tl.store(img_ptr + pix + 2, p2, mask=m)


@triton.jit(do_not_specialize=["height"])
def label_kernel(label_ptr, run_x_ptr, run_y0_ptr, run_y1_ptr, run_label_ptr,
                 counters_ptr, height, WPB: tl.constexpr):
    """Materialise the per-pixel int32 canonical label map (verification
    and visualisation only, never on the headline clock). One warp per
    run with per-run descriptor loads, lanes striding y, as in Numba."""
    n_runs = tl.load(counters_ptr + _N_RUNS_USED).to(tl.int32)
    wi = tl.arange(0, WPB)[:, None]
    lane = tl.arange(0, _WARP)[None, :]
    n_warps = tl.num_programs(0) * WPB

    for r0 in range(tl.program_id(0) * WPB, n_runs, n_warps):
        r = r0 + wi
        inb = r < n_runs
        label = tl.load(run_label_ptr + r, mask=inb, other=0)
        x = tl.load(run_x_ptr + r, mask=inb, other=0)
        y0 = tl.load(run_y0_ptr + r, mask=inb, other=0)
        y1 = tl.load(run_y1_ptr + r, mask=inb, other=0)
        labels = tl.broadcast_to(label, [WPB, _WARP])
        row = x.to(tl.int64) * height
        n_chunks = tl.max(tl.where(inb, (y1 - y0 + _WARP) >> 5, 0))
        for ci in range(0, n_chunks):
            y = y0 + ci * _WARP + lane
            tl.store(label_ptr + row + y, labels, mask=inb & (y <= y1))
