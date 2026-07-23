"""Seed-discovery BFS flood fill: the GPU finds every blob itself.

Nobody passes a seed anymore. Each kernel here is ONE cooperative launch
that starts from a blank slate — an image, an iota parent array and an
empty queue — and ends with every blob filled, labeled and painted, one
canonical label per blob (its minimum linear index; see cpu_oracle.py).

Two variant families make opposite bets on WHERE connectivity gets
solved, sharing one union-find protocol:

- ccl_fill: solve connectivity FIRST. A union-find pass over red
  lex-predecessor adjacencies collapses each blob to its min-index root
  (Playne-Stephens style: data-independent — one merge pass, one flatten
  pass, no geodesic-diameter iteration), then exactly one seed per blob
  is enqueued and the fill is the ch03 multisource loop unchanged.
- seed_merge: solve connectivity IN FLIGHT. A local scan enqueues every
  candidate (red pixel with no red lex-predecessor — possibly several
  per blob), the waves race outward carrying provisional labels, and
  where two waves of one blob collide the CAS loser unions the two
  labels. A final flatten repaints everything to the surviving root.

Both retire two ch04 load-bearing decisions, deliberately:

1. Labels move OUT of the queue entry into a per-pixel int32 label_map.
   ch04's proudest line — "the label rides inside the entry, zero extra
   bytes" — cannot survive discovery: seed_merge's CAS loser must ask
   "who owns this pixel?", and only a per-pixel map can answer; and
   provisional labels are pixel indices, which no 6-bit field holds.
   Entries revert to the plain ch03 lin format (x*height + y), and the
   label traffic is now PRICED, not free (see the driver's model note).
2. The kernels take NO n_seeds parameter. ch04's deadlock lesson said
   the initial rear must not be read from q_state at kernel start —
   blocks don't start in lockstep, so early enqueues race the read. The
   refined rule: no UNBARRIERED read. Discovery produces the seed count
   on-GPU, so the kernels read it with the same fence-sandwich every
   level of every live kernel already uses: grid.sync() making all scan
   enqueues visible, one uniform read of rear, grid.sync() before any
   next-phase atomic can move it. Every thread reads the same value or
   the barrier semantics of this GPU are broken.

The union protocol (per PARALLEL_BFS_DESIGN_ANALYSIS.md, atomicMin on
tree roots with retry):

    _find:  chase parent[i] until parent[i] == i. Read-only — no path
            compression during concurrent phases; flatten happens once,
            behind a barrier.
    _union: find both roots; atomic.min the SMALLER value into the
            LARGER root's slot; if the CAS-like min lost (the slot no
            longer held a root), retry from the value it returned.

No livelock: parent values only ever decrease, every failed atomic.min
hands back a strictly smaller working index, and the chain is bounded
below by the blob's min index. Stale L1 reads inside _find are repaired
by the atomics themselves: atomic.min returns the true current value,
so a stale "root" costs one retry, never a wrong link. Each successful
link retires exactly one root — UNION_DONE counts them, so
UNION_DONE == initial_roots - n_blobs is a structural test invariant
(initial roots: every red pixel for ccl_fill, every candidate for
seed_merge).
"""

import os

# Must be set before numba is imported - CUDA 12.9 + ctypes bindings segfault
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import numpy as np
from numba import cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
SMID_CU = os.path.join(_HERE, "smid.cu")

# 8-connectivity offsets (E SE S SW W NW N NE — repo convention)
DX8_HOST = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
DY8_HOST = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# Lex-predecessor offsets: the 4 of the 8 neighbors at a SMALLER linear
# index. The candidate rule probes them ("is anything before me red?"),
# and the CCL merge pass unions along them (each 8-adjacency handled
# exactly once, from its lex-greater endpoint).
PDX_HOST = np.array([-1, -1, -1, 0], dtype=np.int32)
PDY_HOST = np.array([-1, 0, 1, -1], dtype=np.int32)

# Per-blob fill colors, indexed by canonical label % 6. None may be RED
# (painted pixels must stop matching _is_red) or WHITE (background).
# Rows 0/1 stay blue/green so two-blob scenes render like ch04; blobs
# past 6 reuse colors — the label_map, not the paint, is ground truth.
PALETTE_HOST = np.array([[0, 0, 255],       # blue
                         [0, 255, 0],       # green
                         [0, 255, 255],     # cyan
                         [255, 0, 255],     # magenta
                         [255, 165, 0],     # orange
                         [128, 0, 255]],    # purple
                        dtype=np.uint8)
N_PALETTE = PALETTE_HOST.shape[0]

# Slots in the device-side int64 counters array (ch03/ch04 layout plus
# the discovery counters)
FILLED = 0
LEVELS = 1
OVERFLOW = 2            # defensive tripwire: structurally unreachable
PEAK_LEVEL = 3          # largest single frontier
PEAK_OCC = 4            # max queue entries alive across two adjacent levels
ACTIVE_THREAD_SUM = 5   # sum over levels of min(level_size, grid threads)
ACTIVE_WARP_SUM = 6     # sum over levels of ceil(min(level_size, grid)/32)
PROCESSED = 7           # pixels dequeued/recolored (== FILLED iff exactly-once)
CAS_ATTEMPTS = 8        # visited-CAS ops tried during the fill
CANDIDATES = 9          # queue entries after discovery (seed_merge:
                        # candidates; ccl_fill: n_blobs)
UNION_ATTEMPTS = 10     # _union calls (collision branches / merge probes)
UNION_DONE = 11         # successful links == initial_roots - n_blobs
NUM_COUNTERS = 12

# Columns of the (blocks, 2) int64 block_stats array
BS_PROCESSED = 0        # pixels this block dequeued -> N-way load balance
BS_SMID = 1             # %smid this block observed itself running on

# q_state slots
Q_REAR = 0

# %smid reader linked from smid.cu
get_smid = cuda.declare_device('get_smid', 'uint32()')


@cuda.jit(device=True, inline=True)
def _is_red(img, x, y):
    """Check if pixel is red (255, 0, 0)."""
    return img[x, y, 0] == 255 and img[x, y, 1] == 0 and img[x, y, 2] == 0


@cuda.jit(device=True, inline=True)
def _warp_enqueue_global(arr, state, rear_slot, item, counters):
    """Warp-aggregated append on a global rear counter (one atomic per warp).

    All lanes arriving together aggregate (activemask/popc/shfl); the lowest
    lane reserves a ticket slab with one global atomic. The bound check is a
    defensive tripwire only (see the structural argument in the module doc).
    """
    mask = cuda.activemask()
    count = cuda.popc(mask)
    rank = cuda.popc(mask & cuda.lanemask_lt())
    leader = cuda.ffs(mask) - 1  # ffs is 1-based; mask always has a bit set
    base = 0
    if cuda.laneid == leader:
        base = cuda.atomic.add(state, rear_slot, count)
    base = cuda.shfl_sync(mask, base, leader)
    idx = base + rank
    if idx < arr.shape[0]:
        arr[idx] = item
    else:
        counters[OVERFLOW] = 1  # unreachable by the structural argument


@cuda.jit(device=True, inline=True)
def _find(parent, i):
    """Chase to the root. Read-only — never compresses (concurrent-phase
    safe); a stale read costs the caller one atomic.min retry at worst."""
    while parent[i] != i:
        i = parent[i]
    return i


@cuda.jit(device=True)
def _union(parent, a, b):
    """Link the classes of a and b; returns 1 iff THIS call retired a root
    (the atomic.min landed on a slot that still held its own index).

    Terminates: the working index strictly decreases on every retry and
    is bounded below by the class minimum. Writing into a slot that has
    meanwhile stopped being a root is harmless — any parent value is a
    same-class node smaller than its index, so chains stay valid and
    strictly decreasing; the class minimum is never written at all
    (no same-class value is smaller), so exactly it survives as root.
    """
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


# ---------------------------------------------------------------- ccl_fill
# Connectivity first: union-find CCL over lex-predecessor adjacencies,
# then exactly one seed per blob feeds the unchanged ch03 fill loop.
# Phases inside ONE cooperative launch, separated by grid.sync():
#   P0 iota parent  P1 merge  P2 flatten + enqueue roots  P3 fill.
# The merge pass is data-INdependent — no geodesic-diameter iteration,
# the serpentine costs the same barriers as a square.


@cuda.jit(link=[SMID_CU])
def ccl_fill_kernel(img, visited, depth, owner, parent, label_map, queue,
                    q_state, counters, block_stats, level_sizes):
    """Host contract: launch [blocks, tpb] cooperative; img untouched red/
    white scene; visited/counters/block_stats zeroed; depth/owner/label_map
    filled with -1; parent uninitialized (P0 writes it); q_state=[0]."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]
    n = width * height

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    pdx = cuda.const.array_like(PDX_HOST)
    pdy = cuda.const.array_like(PDY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    my_union_attempts = 0
    my_union_done = 0

    # P0: every pixel its own root
    for i in range(tid, n, stride):
        parent[i] = i
    grid.sync()

    # P1: merge along red lex-predecessor adjacencies (each of the blob's
    # 8-adjacencies handled exactly once, from its lex-greater endpoint)
    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            for d in range(4):
                nx = x + pdx[d]
                ny = y + pdy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_union_attempts += 1
                    my_union_done += _union(parent, i, nx * height + ny)
    grid.sync()

    # P2: flatten every red pixel to its root; the root pixel itself is
    # the blob's canonical seed — pre-visit and enqueue it
    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            root = _find(parent, i)
            label_map[x, y] = root
            if root == i:
                visited[x, y] = 1
                _warp_enqueue_global(queue, q_state, Q_REAR, i, counters)

    # The fence sandwich: all discovery enqueues precede sync #1, nothing
    # moves rear again until after sync #2 — every thread reads the same
    # seed count (the refined ch04 lesson: barriered reads are fine)
    grid.sync()
    rear = q_state[Q_REAR]
    grid.sync()
    n_seeds = rear

    # P3: multisource fill — the ch03 conn8 loop; labels are already
    # final, so paint returns to the dequeue site
    front = 0
    level = 0
    peak_level = 0
    peak_occ = 0
    active_thread_sum = 0
    active_warp_sum = 0
    my_processed = 0
    my_cas_attempts = 0

    while front < rear:
        level_size = rear - front
        if level_size > peak_level:
            peak_level = level_size
        active = min(level_size, stride)
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if tid == 0 and level < level_sizes.shape[0]:
            level_sizes[level] = level_size

        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            c = label_map[x, y] % N_PALETTE
            img[x, y, 0] = palette[c, 0]
            img[x, y, 1] = palette[c, 1]
            img[x, y, 2] = palette[c, 2]
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map
            my_processed += 1

            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)

        grid.sync()  # enqueues + final rear for this level visible grid-wide
        new_rear = q_state[Q_REAR]
        grid.sync()  # everyone has read new_rear; next level's atomics may begin

        level += 1
        occ = new_rear - front
        if occ > peak_occ:
            peak_occ = occ
        front = rear
        rear = new_rear

    cuda.atomic.add(counters, PROCESSED, my_processed)
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    cuda.atomic.add(counters, UNION_ATTEMPTS, my_union_attempts)
    cuda.atomic.add(counters, UNION_DONE, my_union_done)
    cuda.atomic.add(block_stats, (bx, BS_PROCESSED), my_processed)
    if cuda.threadIdx.x == 0:
        block_stats[bx, BS_SMID] = get_smid()
    if tid == 0:
        # all grid-uniform register values
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum
        counters[CANDIDATES] = n_seeds


# --------------------------------------------------------- ccl_fill bare twin
# Identical phases with all per-level/per-thread instrumentation stripped
# (only exit-time FILLED/LEVELS remain) to measure observer overhead. It
# shares every device helper above so the algorithm cannot drift.


@cuda.jit
def ccl_fill_bare_kernel(img, visited, depth, parent, label_map, queue,
                         q_state, counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]
    n = width * height

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    pdx = cuda.const.array_like(PDX_HOST)
    pdy = cuda.const.array_like(PDY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    for i in range(tid, n, stride):
        parent[i] = i
    grid.sync()

    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            for d in range(4):
                nx = x + pdx[d]
                ny = y + pdy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    _union(parent, i, nx * height + ny)
    grid.sync()

    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            root = _find(parent, i)
            label_map[x, y] = root
            if root == i:
                visited[x, y] = 1
                _warp_enqueue_global(queue, q_state, Q_REAR, i, counters)

    grid.sync()
    rear = q_state[Q_REAR]
    grid.sync()

    front = 0
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            c = label_map[x, y] % N_PALETTE
            img[x, y, 0] = palette[c, 0]
            img[x, y, 1] = palette[c, 1]
            img[x, y, 2] = palette[c, 2]
            depth[x, y] = level
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# --------------------------------------------------------------- seed_merge
# Connectivity in flight: every locally-detectable candidate (red pixel
# with no red lex-predecessor — at least one per blob, its lex-min pixel)
# starts a wave at level 0; where two waves of one blob collide, the CAS
# loser unions the two provisional labels. Paint is DEFERRED to the final
# flatten, and the deferral is load-bearing: a claimed pixel stays red,
# so the later wave still probes it, loses the CAS, and looks up the
# winner's label in label_map. That lookup is safe by a two-sided retry:
# the winner wrote its label BEFORE enqueueing, so by the time the winner
# is dequeued (>= 1 level barrier later) the label is visible grid-wide.
# A CAS loser in the SAME level as the claim may still read the -1 the
# map was initialized to — the guard skips it, and the union is retried
# from the other side one level later (adjacent depths differ by <= 1,
# and the other side's probe finds this pixel still red). The last
# possible union happens in the final level's processing pass, which the
# loop's closing barrier pair orders before the flatten reads.
# Phases: P0 iota parent  P1 candidate scan  P2 racing fill + unions
# P3 flatten + relabel + repaint.


@cuda.jit(link=[SMID_CU])
def seed_merge_kernel(img, visited, depth, owner, parent, label_map,
                      prov_label, queue, q_state, counters, block_stats,
                      level_sizes):
    """Host contract: as ccl_fill_kernel, plus prov_label filled with -1
    (the pre-merge label snapshot the flatten preserves for replay)."""
    grid = cuda.cg.this_grid()
    bx = cuda.blockIdx.x
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]
    n = width * height

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    pdx = cuda.const.array_like(PDX_HOST)
    pdy = cuda.const.array_like(PDY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    my_union_attempts = 0
    my_union_done = 0

    # P0: every pixel its own root
    for i in range(tid, n, stride):
        parent[i] = i
    grid.sync()

    # P1: candidate scan — grid-stride gives one thread per pixel, so the
    # pre-visit and label stores need no atomics
    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            found = False
            for d in range(4):
                nx = x + pdx[d]
                ny = y + pdy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    found = True
            if not found:
                visited[x, y] = 1
                label_map[x, y] = i
                _warp_enqueue_global(queue, q_state, Q_REAR, i, counters)

    # The fence sandwich (see module doc): every thread reads the same
    # candidate count
    grid.sync()
    rear = q_state[Q_REAR]
    grid.sync()
    n_candidates = rear

    # P2: racing multisource fill — labels inherit at claim, collide at
    # CAS loss, merge via union-find. No paint (see block comment above).
    front = 0
    level = 0
    peak_level = 0
    peak_occ = 0
    active_thread_sum = 0
    active_warp_sum = 0
    my_processed = 0
    my_cas_attempts = 0

    while front < rear:
        level_size = rear - front
        if level_size > peak_level:
            peak_level = level_size
        active = min(level_size, stride)
        active_thread_sum += active
        active_warp_sum += (active + 31) // 32
        if tid == 0 and level < level_sizes.shape[0]:
            level_sizes[level] = level_size

        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height

            lbl = label_map[x, y]
            depth[x, y] = level
            owner[x, y] = bx  # per-pixel block-owner map
            my_processed += 1

            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    my_cas_attempts += 1
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        # win: stamp the inherited label BEFORE the entry
                        # becomes dequeueable
                        label_map[nx, ny] = lbl
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)
                    else:
                        # collision: someone owns it — same blob, maybe
                        # another wave. -1 = not yet visible; skip, the
                        # other side retries (block comment above).
                        other = label_map[nx, ny]
                        if other >= 0 and other != lbl:
                            my_union_attempts += 1
                            my_union_done += _union(parent, lbl, other)

        grid.sync()  # enqueues + final rear for this level visible grid-wide
        new_rear = q_state[Q_REAR]
        grid.sync()  # everyone has read new_rear; next level's atomics may begin

        level += 1
        occ = new_rear - front
        if occ > peak_occ:
            peak_occ = occ
        front = rear
        rear = new_rear

    # P3: flatten + relabel + repaint. The loop's closing barrier pair
    # ordered every union before this point; label_map >= 0 is exactly
    # the filled set (labels are stamped at claim time).
    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        prov = label_map[x, y]
        if prov >= 0:
            prov_label[x, y] = prov
            final = _find(parent, prov)
            label_map[x, y] = final
            c = final % N_PALETTE
            img[x, y, 0] = palette[c, 0]
            img[x, y, 1] = palette[c, 1]
            img[x, y, 2] = palette[c, 2]

    cuda.atomic.add(counters, PROCESSED, my_processed)
    cuda.atomic.add(counters, CAS_ATTEMPTS, my_cas_attempts)
    cuda.atomic.add(counters, UNION_ATTEMPTS, my_union_attempts)
    cuda.atomic.add(counters, UNION_DONE, my_union_done)
    cuda.atomic.add(block_stats, (bx, BS_PROCESSED), my_processed)
    if cuda.threadIdx.x == 0:
        block_stats[bx, BS_SMID] = get_smid()
    if tid == 0:
        # all grid-uniform register values
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level
        counters[PEAK_LEVEL] = peak_level
        counters[PEAK_OCC] = peak_occ
        counters[ACTIVE_THREAD_SUM] = active_thread_sum
        counters[ACTIVE_WARP_SUM] = active_warp_sum
        counters[CANDIDATES] = n_candidates


# ------------------------------------------------------- seed_merge bare twin


@cuda.jit
def seed_merge_bare_kernel(img, visited, depth, parent, label_map, queue,
                           q_state, counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]
    n = width * height

    dx = cuda.const.array_like(DX8_HOST)
    dy = cuda.const.array_like(DY8_HOST)
    pdx = cuda.const.array_like(PDX_HOST)
    pdy = cuda.const.array_like(PDY_HOST)
    palette = cuda.const.array_like(PALETTE_HOST)

    for i in range(tid, n, stride):
        parent[i] = i
    grid.sync()

    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            found = False
            for d in range(4):
                nx = x + pdx[d]
                ny = y + pdy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    found = True
            if not found:
                visited[x, y] = 1
                label_map[x, y] = i
                _warp_enqueue_global(queue, q_state, Q_REAR, i, counters)

    grid.sync()
    rear = q_state[Q_REAR]
    grid.sync()

    front = 0
    level = 0

    while front < rear:
        for i in range(front + tid, rear, stride):
            pixel = queue[i]
            x = pixel // height
            y = pixel % height
            lbl = label_map[x, y]
            depth[x, y] = level
            for d in range(8):
                nx = x + dx[d]
                ny = y + dy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    if cuda.atomic.cas(visited, (nx, ny), 0, 1) == 0:
                        label_map[nx, ny] = lbl
                        _warp_enqueue_global(queue, q_state, Q_REAR,
                                             nx * height + ny, counters)
                    else:
                        other = label_map[nx, ny]
                        if other >= 0 and other != lbl:
                            _union(parent, lbl, other)
        grid.sync()
        new_rear = q_state[Q_REAR]
        grid.sync()
        level += 1
        front = rear
        rear = new_rear

    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        prov = label_map[x, y]
        if prov >= 0:
            final = _find(parent, prov)
            label_map[x, y] = final
            c = final % N_PALETTE
            img[x, y, 0] = palette[c, 0]
            img[x, y, 1] = palette[c, 1]
            img[x, y, 2] = palette[c, 2]

    if tid == 0:
        counters[FILLED] = q_state[Q_REAR]
        counters[LEVELS] = level


# ------------------------------------------------- benchmark phase kernels
# Discovery WITHOUT the fill, for benchmark.py's phase attribution
# (fill ~= fused_total - phase). seed_scan_kernel is seed_merge's P0-P1;
# ccl_kernel is ccl_fill's P0-P2. They call the same device functions as
# the fused kernels, so the measured phase cannot drift from the real
# one. Exit stores the discovery queue rear in counters[CANDIDATES].


@cuda.jit
def seed_scan_kernel(img, visited, label_map, parent, queue, q_state,
                     counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]
    n = width * height

    pdx = cuda.const.array_like(PDX_HOST)
    pdy = cuda.const.array_like(PDY_HOST)

    for i in range(tid, n, stride):
        parent[i] = i
    grid.sync()

    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            found = False
            for d in range(4):
                nx = x + pdx[d]
                ny = y + pdy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    found = True
            if not found:
                visited[x, y] = 1
                label_map[x, y] = i
                _warp_enqueue_global(queue, q_state, Q_REAR, i, counters)

    grid.sync()
    if tid == 0:
        counters[CANDIDATES] = q_state[Q_REAR]


@cuda.jit
def ccl_kernel(img, visited, label_map, parent, queue, q_state, counters):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    width = img.shape[0]
    height = img.shape[1]
    n = width * height

    pdx = cuda.const.array_like(PDX_HOST)
    pdy = cuda.const.array_like(PDY_HOST)

    for i in range(tid, n, stride):
        parent[i] = i
    grid.sync()

    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            for d in range(4):
                nx = x + pdx[d]
                ny = y + pdy[d]
                if 0 <= nx < width and 0 <= ny < height and _is_red(img, nx, ny):
                    _union(parent, i, nx * height + ny)
    grid.sync()

    for i in range(tid, n, stride):
        x = i // height
        y = i % height
        if _is_red(img, x, y):
            root = _find(parent, i)
            label_map[x, y] = root
            if root == i:
                visited[x, y] = 1
                _warp_enqueue_global(queue, q_state, Q_REAR, i, counters)

    grid.sync()
    if tid == 0:
        counters[CANDIDATES] = q_state[Q_REAR]
