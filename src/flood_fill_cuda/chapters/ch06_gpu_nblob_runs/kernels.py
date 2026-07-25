"""Runs, not pixels: connected components at the bandwidth floor.

Every chapter so far moved PIXELS. A BFS frontier is a list of pixel
indices; a union-find prepass unions per pixel adjacency; the label map
is one int32 per pixel. Chapter 5's best config recolors the 81 Mpx
`input_blobs.png` in 53 ms, and the honest question this chapter asks
is: what would the SAME job cost if it were bounded only by memory?

The answer starts with an inventory of the image itself:

    81,000,000 px      the grid
    13,451,960 px      red
       539,207         maximal horizontal runs of red (mean 24.9 px)
         2,522         blobs

A run is a maximal contiguous span of red inside one row (one x, a
y-interval). Runs are the natural unit of a raster blob: there are 150x
fewer of them than red pixels and 25x fewer than the pixel count of any
frontier that would sweep them. And every fact this chapter needs —
connectivity, canonical labels, the paint spans — is a fact ABOUT RUNS.

So the pipeline stops touching pixels except twice, once to read and
once to write:

    pack    RGB -> 1 bit/px       (the ONLY full-resolution read)
    count   runs per row          \\  over the packed mask: 1 bit/px,
    scan    row -> run offsets     >  not 24 -- 10.15 MB, not 243 MB
    emit    the run table         /
    merge   union-find over vertically adjacent runs   (539k items)
    flatten path-compress to roots                     (539k items)
    paint   one warp per run, writes its span          (only red px)

Everything between the read and the write operates on 539k items — a
rounding error. The clock is set by the two ends.

WHY THE RUN TABLE CARRIES THE CANONICAL LABEL FOR FREE
Runs are emitted in row-major order (x ascending, then y0 ascending),
so run ids are ordered exactly like the linear index of their first
pixel: id(r) < id(s)  <=>  lin(r) < lin(s), where lin(r) = x*height+y0.
Chapter 5's union-by-atomicMin over pixel indices converges to a blob's
minimum linear index; the same protocol over RUN IDS converges to the
minimum-id run of the blob, whose first pixel IS the blob's lex-min
pixel. The canonical label is therefore unchanged from ch05 —
`run_x[root]*height + run_y0[root]` — and the same CPU oracle judges
both. Nothing about the label had to be redesigned; only its carrier.

WHAT IS GIVEN UP, DELIBERATELY
There is no BFS here, so there is no `depth` map and no level count.
Chapters 1-5 measure a geodesic clock; this one has no geodesic
structure at all — the merge pass is data-independent (one pass over
run adjacencies, no iteration to convergence), so the serpentine and
the square cost the same. `visited` degenerates to "has a label".

WHY MANY PLAIN KERNELS AND NOT ONE COOPERATIVE LAUNCH
Chapters 3-5 fused everything into one cooperative launch because the
phases had to interleave with grid-wide barriers between BFS levels.
Here the phases are a DAG with six edges and no loop, so stream order
is the barrier — and plain kernels launch at full occupancy instead of
the cooperative residency cap (which cost ch05 half its grid over one
register). The price is host-side: enqueuing the six launches from
Python costs 0.33 ms against 1.35 ms of GPU work. That is affordable,
but it is why nothing here may touch the host mid-pipeline — see
row_scan_kernel, which absorbed the counter clear for exactly that
reason.

BIT LAYOUT
`mask` is (width, words_per_row) uint32, words_per_row =
ceil(height/32); bit b of word (x, w) is pixel (x, w*32 + b). Bits past
`height` in the last word of a row are ZERO — the pack kernel writes
them that way and every downstream kernel relies on it (a run can never
straddle a row boundary because the padding always terminates it).

RUN-START / RUN-END BIT ALGEBRA
For a row word `w` with `p` = the bit that precedes it (bit 31 of the
previous word, 0 at the row start) and `q` = the bit that follows it
(bit 0 of the next word, 0 at the row end):

    starts = w & ~((w << 1) | p)      a 1 whose predecessor is 0
    ends   = w & ~((w >> 1) | q<<31)  a 1 whose successor is 0

Within a row the k-th start and the k-th end belong to the same run
(runs are disjoint and ordered), so the two independently-scanned
streams pair up by index — that is why emit needs no cross-word
stitching and no serial walk.
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda
from numba import int64

# The paint contract is ch05's, unchanged: palette[canonical label % 6].
# Imported rather than copied so a divergence is impossible.
from ..ch05_gpu_nblob_nblock.kernels import PALETTE_HOST, N_PALETTE

FULL_MASK = 0xFFFFFFFF
WARP = 32

# Slots in the device-side int64 counters array
RUN_OVERFLOW = 0        # tripwire: run table capacity exceeded
N_RUNS = 1              # runs the image HAS (== row_off[width])
N_BLOBS = 2             # surviving roots after flatten
UNION_ATTEMPTS = 3      # _union calls (vertical run adjacencies probed)
UNION_DONE = 4          # successful links == n_runs - n_blobs
N_RUNS_USED = 5         # min(N_RUNS, capacity) — the bound every
                        # downstream kernel loops to, so an overflowed
                        # run table produces WRONG answers but never an
                        # out-of-bounds access (the host raises on
                        # RUN_OVERFLOW before anyone reads them)
NUM_COUNTERS = 6

# Threads per block for the single-block row scan (compile-time constant:
# it sizes the shared array). One block scans `width` rows in
# ceil(width/SCAN_TPB) chunks.
SCAN_TPB = 1024


# --------------------------------------------------------------- helpers

@cuda.jit(device=True, inline=True)
def _is_red(img, x, y):
    """Is pixel (x, y) pure red (255, 0, 0)?

    Deliberately NOT ch05's `a == 255 and b == 0 and c == 0`. Python's
    `and` short-circuits, so that spelling compiles to three DEPENDENT
    loads separated by branches: the G byte is not even requested until
    the R byte has come back from memory. In a kernel whose entire job
    is to be bandwidth-bound, that serialises three round trips of
    memory latency per pixel and costs more than everything else this
    chapter does. Bitwise `&` on the three comparisons keeps the loads
    independent, so all three are in flight at once.
    """
    r = img[x, y, 0]
    g = img[x, y, 1]
    b = img[x, y, 2]
    return (r == 255) & (g == 0) & (b == 0)


@cuda.jit(device=True, inline=True)
def _word(mask, x, w, words_per_row):
    """mask[x, w] as a 0..2^32-1 int64, or 0 outside the row."""
    if w < 0 or w >= words_per_row:
        return int64(0)
    return int64(mask[x, w])


@cuda.jit(device=True, inline=True)
def _starts_ends(mask, x, w, words_per_row):
    """(starts, ends) bitmasks for word w of row x — the algebra in the
    module docstring, with the row-edge carries read from the neighbour
    words (out of row => 0, which is what terminates every run)."""
    cur = _word(mask, x, w, words_per_row)
    prev = _word(mask, x, w - 1, words_per_row)
    nxt = _word(mask, x, w + 1, words_per_row)
    starts = cur & ~((cur << 1) | (prev >> 31))
    ends = cur & ~((cur >> 1) | ((nxt & 1) << 31))
    return starts, ends


@cuda.jit(device=True, inline=True)
def _warp_scan(value):
    """Inclusive warp prefix sum; returns (inclusive, warp_total).

    Every lane of the warp must call it (the emit loop is trip-count
    uniform for exactly this reason)."""
    lane = cuda.laneid
    acc = value
    d = 1
    while d < WARP:
        got = cuda.shfl_up_sync(FULL_MASK, acc, d)
        if lane >= d:
            acc += got
        d *= 2
    total = cuda.shfl_sync(FULL_MASK, acc, WARP - 1)
    return acc, total


@cuda.jit(device=True, inline=True)
def _find(parent, i):
    """Chase to the root, HALVING the path on the way (parent[i] gets
    its grandparent).

    ch05's `_find` is deliberately read-only, on the rule "no
    compression while unions are in flight". That rule is stronger than
    it needs to be, and this chapter pays for it: run chains here are as
    deep as a blob is tall, and a read-only find walks all of it. The
    write is safe under the same invariant the union protocol already
    maintains — parent[i] is always a SAME-CLASS index strictly below i
    — because the grandparent is same-class and <= the parent < i. It
    can race with an atomicMin that had already lowered the slot
    further, in which case compression is simply lost for that slot; it
    cannot make the slot leave the class or stop decreasing. And the
    class minimum is never written at all: for the root, parent[i] == i
    returns before any store.

    Measured on the 539k-run image: 0.367 ms -> 0.262 ms of merge.
    """
    while True:
        p = parent[i]
        if p == i:
            return i
        gp = parent[p]
        if gp == p:
            return p
        parent[i] = gp
        i = gp


@cuda.jit(device=True)
def _union(parent, a, b):
    """ch05's atomicMin union, verbatim, over RUN ids instead of pixel
    ids. Returns 1 iff this call retired a root. Terminates for the same
    reason: the working index strictly decreases and is bounded below by
    the class minimum."""
    while True:
        a = _find(parent, a)
        b = _find(parent, b)
        if a == b:
            return 0
        if a < b:
            a, b = b, a
        old = cuda.atomic.min(parent, a, b)
        if old == a:
            return 1
        a = old


# ------------------------------------------------------------------ pack
# The only kernel that reads the full-resolution image. One WARP per
# 32-pixel word: the 32 lanes test their own pixel, one ballot collapses
# the warp's votes into the word, lane 0 stores it. 24 B/px in, 1 bit/px
# out — the 24x that every later pass is not paying.

@cuda.jit
def pack_kernel(img, mask):
    """img (width, height, 3) uint8 -> mask (width, words_per_row) uint32.

    Bits past `height` in a row's last word are written as 0.

    Launched on a 2D grid — row in y, word in x — SPECIFICALLY so that
    no thread ever computes `word_index // words_per_row`. A 64-bit
    integer division by a runtime value has no hardware support on this
    architecture; the flat 1D grid-stride spelling this replaces paid
    one per warp iteration and lost ~1.5 ms of the 3 ms it took, which
    is more than the entire mask-contract pipeline costs.
    """
    width = img.shape[0]
    height = img.shape[1]
    words_per_row = mask.shape[1]

    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    w = t >> 5              # warp-uniform: blockDim.x is a multiple of 32
    lane = t & 31
    if w >= words_per_row:
        return              # warp-uniform, so the ballot below is safe
    y = w * WARP + lane

    for x in range(cuda.blockIdx.y, width, cuda.gridDim.y):
        red = False
        if y < height:
            red = _is_red(img, x, y)
        bits = cuda.ballot_sync(FULL_MASK, red)
        if lane == 0:
            mask[x, w] = bits


@cuda.jit
def unpack_kernel(mask, img, height):
    """mask -> a white image with pure-red runs. The inverse of pack,
    used to build packed-contract inputs and to prove pack lossless."""
    width = mask.shape[0]
    words_per_row = mask.shape[1]
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    w = t >> 5
    lane = t & 31
    if w >= words_per_row:
        return
    y = w * WARP + lane
    if y >= height:
        return
    for x in range(cuda.blockIdx.y, width, cuda.gridDim.y):
        bit = (int64(mask[x, w]) >> lane) & 1
        img[x, y, 0] = 255
        img[x, y, 1] = 0 if bit else 255
        img[x, y, 2] = 0 if bit else 255


# ----------------------------------------------------------------- count
# One warp per ROW, 32 words per iteration. Counting is separate from
# emitting because emit needs each row's base offset, and that offset is
# a prefix sum over rows — which cannot be known until every row is
# counted. The re-read costs 10.15 MB, most of which the 24 MB L2 still
# holds from this very pass.

@cuda.jit
def count_kernel(mask, row_count):
    """row_count[x] = number of maximal red runs in row x."""
    width = mask.shape[0]
    words_per_row = mask.shape[1]
    lane = cuda.laneid
    warp_id = cuda.grid(1) // WARP
    n_warps = cuda.gridsize(1) // WARP

    for x in range(warp_id, width, n_warps):
        total = 0
        base = 0
        while base < words_per_row:
            w = base + lane
            starts = int64(0)
            if w < words_per_row:
                starts, _ends = _starts_ends(mask, x, w, words_per_row)
            total += cuda.popc(starts)
            base += WARP
        # warp-reduce the per-lane totals
        d = WARP // 2
        while d > 0:
            total += cuda.shfl_down_sync(FULL_MASK, total, d)
            d //= 2
        if lane == 0:
            row_count[x] = total


@cuda.jit
def row_scan_kernel(row_count, row_off, counters, capacity):
    """Exclusive prefix sum of row_count into row_off (width+1 entries).

    ONE block of SCAN_TPB threads walking `width` rows in chunks — 9,000
    rows is far too small to be worth a multi-block decoupled scan, and
    a second launch would cost more than the work does."""
    width = row_count.shape[0]
    tid = cuda.threadIdx.x
    ntb = cuda.blockDim.x
    sh = cuda.shared.array(SCAN_TPB, dtype=np.int32)

    carry = 0
    base = 0
    while base < width:
        v = row_count[base + tid] if base + tid < width else 0
        sh[tid] = v
        cuda.syncthreads()
        d = 1
        while d < ntb:
            got = 0
            if tid >= d:
                got = sh[tid - d]
            cuda.syncthreads()
            if tid >= d:
                sh[tid] += got
            cuda.syncthreads()
            d *= 2
        if base + tid < width:
            row_off[base + tid] = carry + sh[tid] - v
        chunk_total = sh[ntb - 1]
        cuda.syncthreads()
        carry += chunk_total
        base += ntb

    if tid == 0:
        row_off[width] = carry
        counters[N_RUNS] = carry
        counters[N_RUNS_USED] = min(carry, capacity)
        counters[RUN_OVERFLOW] = 1 if carry > capacity else 0
        # Reset the counters the LATER phases accumulate into. Every
        # reader of these three runs after this kernel and so does every
        # writer, so this is the right place — and it is the ONLY place,
        # because the obvious alternative is a trap that cost this
        # chapter more than any kernel did: `counters.copy_to_device(
        # zeros)` is a numba H2D copy from pageable memory, which is
        # SYNCHRONOUS. Issued once per pipeline run it turned six
        # asynchronous launches into six host-blocked round trips —
        # 1.72 ms of pure host enqueue time, more than the entire GPU
        # pipeline. The clock was measuring Python. Clearing here costs
        # nothing (this thread is already writing the array), removes a
        # launch, and took host enqueue to 0.33 ms.
        counters[N_BLOBS] = 0
        counters[UNION_ATTEMPTS] = 0
        counters[UNION_DONE] = 0


# ------------------------------------------------------------------ emit
# One warp per row again. Each lane holds one word; a warp prefix sum
# over popc(starts) turns "how many runs does my word open" into "where
# do my runs go", so the run table comes out ordered by (x, y0) without
# any sort — which is exactly the property the canonical label needs.
# The k-th start and k-th end of a row belong to the same run, so the
# two scans write into the same slot from opposite bit algebras.

@cuda.jit
def emit_kernel(mask, row_off, run_x, run_y0, run_y1, parent, counters):
    """Fill the run table. run_x/run_y0/run_y1[k] describe run k;
    y1 is INCLUSIVE. Runs are ordered by (x, y0).

    Also writes parent[k] = k. The union-find iota used to be its own
    launch, on the argument that the merge pass reads parent slots
    belonging to other blocks' runs so the iota must be globally visible
    first. It still must — but emit ALREADY finishes before merge starts
    (stream order), and the thread that writes run k's descriptors is
    exactly the one that would have written parent[k]. Folding it in
    removes a launch and a full pass over the run table.
    """
    width = mask.shape[0]
    words_per_row = mask.shape[1]
    capacity = run_x.shape[0]
    lane = cuda.laneid
    warp_id = cuda.grid(1) // WARP
    n_warps = cuda.gridsize(1) // WARP

    for x in range(warp_id, width, n_warps):
        base_slot = row_off[x]
        cur_start = 0
        cur_end = 0
        base = 0
        while base < words_per_row:          # trip count uniform per warp
            w = base + lane
            starts = int64(0)
            ends = int64(0)
            if w < words_per_row:
                starts, ends = _starts_ends(mask, x, w, words_per_row)
            n_s = cuda.popc(starts)
            n_e = cuda.popc(ends)
            incl_s, tot_s = _warp_scan(n_s)
            incl_e, tot_e = _warp_scan(n_e)

            k = base_slot + cur_start + incl_s - n_s
            s = starts
            while s != 0:
                b = cuda.ffs(s) - 1
                if k < capacity:
                    run_x[k] = x
                    run_y0[k] = w * WARP + b
                    parent[k] = k
                else:
                    counters[RUN_OVERFLOW] = 1
                k += 1
                s &= s - 1

            k = base_slot + cur_end + incl_e - n_e
            e = ends
            while e != 0:
                b = cuda.ffs(e) - 1
                if k < capacity:
                    run_y1[k] = w * WARP + b
                k += 1
                e &= e - 1

            cur_start += tot_s
            cur_end += tot_e
            base += WARP


# ----------------------------------------------------------------- merge
# The whole connectivity problem, over 539k items. One thread per run;
# it binary-searches the NEXT row for the first run that could touch it
# (8-connectivity => the y-interval widens by 1 on each side) and walks
# forward while the overlap holds. Every vertical adjacency is visited
# exactly once, from its upper endpoint — the same "handle each
# adjacency from one side" discipline as ch05's lex-predecessor rule.
# Data-independent: no iteration to convergence, so blob SHAPE costs
# nothing (the serpentine that took ch03 32,641 levels costs one pass).

@cuda.jit
def merge_rows_kernel(run_x, run_y0, run_y1, row_off, parent, counters,
                      instrumented):
    """Union every run with the runs it touches in the row below."""
    n_runs = counters[N_RUNS_USED]
    capacity = run_x.shape[0]
    width = row_off.shape[0] - 1
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    attempts = 0
    done = 0
    for r in range(tid, n_runs, stride):
        x = run_x[r]
        if x + 1 >= width:
            continue
        y0 = run_y0[r]
        y1 = run_y1[r]
        lo = min(row_off[x + 1], capacity)   # clamps are dead code unless
        hi = min(row_off[x + 2], capacity)   # RUN_OVERFLOW already fired
        # first run of the next row whose end reaches y0-1 (8-conn)
        a = lo
        b = hi
        while a < b:
            m = (a + b) >> 1
            if run_y1[m] < y0 - 1:
                a = m + 1
            else:
                b = m
        s = a
        while s < hi and run_y0[s] <= y1 + 1:
            attempts += 1
            done += _union(parent, r, s)
            s += 1

    if instrumented:
        if attempts:
            cuda.atomic.add(counters, UNION_ATTEMPTS, attempts)
        if done:
            cuda.atomic.add(counters, UNION_DONE, done)


# --------------------------------------------------------------- flatten

@cuda.jit
def flatten_kernel(parent, run_x, run_y0, run_label, height, counters,
                   instrumented):
    """Path-compress every run to its root, resolve its canonical label,
    and count the survivors.

    Resolving the label HERE and not in the paint kernel is the whole
    point: `run_x[root]` and `run_y0[root]` are scattered gathers (root
    is an arbitrary earlier run), and a gather chained behind the
    `parent[r]` load is two dependent memory round trips per run. Paying
    them in flatten — which is already a pointer-chasing pass and has
    nothing else to do — leaves the paint kernel reading four
    independent, perfectly sequential arrays.
    """
    n_runs = counters[N_RUNS_USED]
    roots = 0
    for r in range(cuda.grid(1), n_runs, cuda.gridsize(1)):
        root = _find(parent, r)
        parent[r] = root
        run_label[r] = run_x[root] * height + run_y0[root]
        if root == r:
            roots += 1
    if instrumented and roots:
        cuda.atomic.add(counters, N_BLOBS, roots)


# ----------------------------------------------------------------- paint
# The second and last time a pixel is touched. One warp per run, lanes
# striding the span: a run is contiguous in y, and y is the contiguous
# axis of img[x, y, c], so a warp writes 96 consecutive bytes per step.
# Only red pixels are written — the white 83.4% of the image is never
# addressed at all, which is the other half of why this fits in the
# budget.

@cuda.jit
def paint_kernel(img, run_x, run_y0, run_y1, run_label, counters, palette):
    """Recolor every run's span: one warp per run, lanes striding y.

    Two things this kernel does NOT do, both because they were built and
    measured and both were slower (the numbers are in the chapter README):

    - It does not write 4-byte words. A run's span is a contiguous byte
      range, and an aligned-word store looked like the obvious way to
      stop three per-channel byte stores from each touching the same
      ~96 bytes of sectors. Measured on 539k real runs it is 0.62 ms
      against the byte path's 0.34 ms, and the crossover with span
      length is clean: word stores only win past ~64 px per run
      (span 256: 195 GB/s vs 135). The mean run here is 25 px.
    - It does not read its descriptors per run. Read that way each of
      the four loads is a 32-lane broadcast of one int32, and the warp
      pays a full sector for 4 bytes: 0.34 ms of pure metadata traffic.
      The warp instead fetches 32 runs cooperatively (lane l takes run
      r0+l, four coalesced 128 B loads) and replays them through shfl.
    """
    n_runs = counters[N_RUNS_USED]
    lane = cuda.laneid
    warp_id = cuda.grid(1) >> 5
    n_warps = cuda.gridsize(1) >> 5

    for r0 in range(warp_id * WARP, n_runs, n_warps * WARP):
        r = r0 + lane
        inb = r < n_runs
        my_x = run_x[r] if inb else 0
        my_y0 = run_y0[r] if inb else 0
        my_y1 = run_y1[r] if inb else 0
        my_lab = run_label[r] if inb else 0
        kmax = min(WARP, n_runs - r0)

        for k in range(kmax):
            x = cuda.shfl_sync(FULL_MASK, my_x, k)
            y0 = cuda.shfl_sync(FULL_MASK, my_y0, k)
            y1 = cuda.shfl_sync(FULL_MASK, my_y1, k)
            label = cuda.shfl_sync(FULL_MASK, my_lab, k)
            c = label % N_PALETTE
            p0 = palette[c, 0]
            p1 = palette[c, 1]
            p2 = palette[c, 2]
            # A third idea that was built, measured and thrown away:
            # every pixel here is known to be pure RED, so a channel
            # whose new value already equals 255/0/0 need not be stored
            # (magenta and orange differ from red in ONE channel; the
            # palette mean is 1.83 stores instead of 3). The test is
            # per-color, so it is warp-uniform and cannot diverge. It is
            # still SLOWER — 0.750 ms against 0.665 — because the branch
            # per channel costs more than the store it skips. The stores
            # are not what this kernel is short of.
            for y in range(y0 + lane, y1 + 1, WARP):
                img[x, y, 0] = p0
                img[x, y, 1] = p1
                img[x, y, 2] = p2


@cuda.jit
def label_kernel(label_map, run_x, run_y0, run_y1, run_label, counters):
    """Materialise the per-pixel int32 canonical label map — the ch05
    output the fast path deliberately does NOT write (324 MB at 81 Mpx:
    four times the cost of the whole rest of the pipeline). Verification
    and visualisation only; never on the headline clock."""
    n_runs = counters[N_RUNS_USED]
    lane = cuda.laneid
    warp_id = cuda.grid(1) >> 5
    n_warps = cuda.gridsize(1) >> 5

    for r in range(warp_id, n_runs, n_warps):
        label = run_label[r]
        x = run_x[r]
        for y in range(run_y0[r] + lane, run_y1[r] + 1, WARP):
            label_map[x, y] = label
