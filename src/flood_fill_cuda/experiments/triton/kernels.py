"""Triton kernels for BFS-style flood fill of a single connected blob.

The image is a row-major uint8 grid of cell states:

    0 = BACKGROUND (wall)    1 = RED (fillable)    2 = BLUE (filled)

Parallel BFS by wavefront relaxation: on every launch each program owns one
BLOCK_H x BLOCK_W tile and turns red pixels blue when any 4-neighbour is
blue.  With CONVERGE=True a program keeps sweeping its own tile until it
stops changing, so one launch advances the frontier by up to a whole tile
(and often further, because ``.cg`` loads observe sibling programs' stores
through L2 mid-launch).  With CONVERGE=False each launch performs exactly one
lock-step BFS level: the naive baseline.

Races between tiles are benign: a pixel only ever transitions RED -> BLUE,
and a stale read can only *delay* a fill, never corrupt one.  The host
relaunches until a launch converts nothing, which is exact convergence: a
launch that changed nothing observed a fully settled image (kernel launch
boundaries order memory), so no red pixel with a blue neighbour existed.

The scan mode (``run_flood_fill_scan``) works on whole lines instead:
``scan_line_step`` floods every red run that touches blue along a row or a
column with a segmented max-scan, and ``gather_dirty`` compacts the lines
that may still change, so each pass launches one program per dirty line.
"""

import cupy as cp
import triton
import triton.language as tl

# The shared Triton runtime: importing it installs the CuPy driver once
# per process (the Triton twins use the same one).
from flood_fill_cuda.triton_twins.runtime import t

BACKGROUND = 0
RED = 1
BLUE = 2


@triton.jit
def flood_fill_step(
    grid_ptr,  # *uint8, H*W row-major cell states, updated in place
    changed_ptr,  # *int32, set to 1 if this launch converted any pixel
    H,
    W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    RED_V: tl.constexpr,
    BLUE_V: tl.constexpr,
    CONVERGE: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    row = (pid_h * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    col = (pid_w * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    in_bounds = (row < H) & (col < W)
    idx = row * W + col

    center = tl.load(grid_ptr + idx, mask=in_bounds, other=0)

    up_m = in_bounds & (row > 0)
    dn_m = in_bounds & (row < H - 1)
    lf_m = in_bounds & (col > 0)
    rt_m = in_bounds & (col < W - 1)

    # Do-while: enter only if the tile has anything left to fill.
    pending = tl.sum((center == RED_V).to(tl.int32))
    while pending > 0:
        # .cg keeps neighbour reads out of L1 so stores from this and from
        # sibling programs are observed via L2.
        up = tl.load(grid_ptr + idx - W, mask=up_m, other=0, cache_modifier=".cg")
        dn = tl.load(grid_ptr + idx + W, mask=dn_m, other=0, cache_modifier=".cg")
        lf = tl.load(grid_ptr + idx - 1, mask=lf_m, other=0, cache_modifier=".cg")
        rt = tl.load(grid_ptr + idx + 1, mask=rt_m, other=0, cache_modifier=".cg")

        fill = (center == RED_V) & (
            (up == BLUE_V) | (dn == BLUE_V) | (lf == BLUE_V) | (rt == BLUE_V)
        )
        n = tl.sum(fill.to(tl.int32))
        if n > 0:
            center = tl.where(fill, BLUE_V, center).to(grid_ptr.dtype.element_ty)
            tl.store(grid_ptr + idx, center, mask=fill)
            tl.store(changed_ptr, 1)
        # Make this round's stores visible program-wide before the next
        # round of neighbour loads (intra-tile propagation).
        tl.debug_barrier()
        pending = n
        if not CONVERGE:
            pending = tl.zeros_like(n)  # naive mode: one sweep per launch


@triton.jit
def _seg_max_combine(r_a, m_a, r_b, m_b):
    # segmented max: a wall on the right side resets the running max
    r = r_a | r_b
    m = tl.where(r_b != 0, m_b, tl.maximum(m_a, m_b))
    return r, m


@triton.jit
def gather_dirty(
    dirty_ptr,  # *uint8, per-line "needs rescan" flags; cleared as gathered
    list_ptr,  # *int32, output: compacted indices of dirty lines
    count_ptr,  # *int32, output: number of gathered indices (atomic)
    n,  # number of lines
    BLOCK: tl.constexpr,
):
    """Stream-compact the indices of dirty lines into a dense list."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    is_dirty = (tl.load(dirty_ptr + offs, mask=m, other=0) != 0) & m
    flags = is_dirty.to(tl.int32)
    local = tl.cumsum(flags, 0) - flags  # exclusive prefix sum
    base = tl.atomic_add(count_ptr, tl.sum(flags))
    tl.store(list_ptr + base + local, offs, mask=is_dirty)
    tl.store(dirty_ptr + offs, tl.zeros((BLOCK,), tl.uint8), mask=m)


@triton.jit
def scan_line_step(
    grid_ptr,  # *uint8, cell states, updated in place
    line_list_ptr,  # *int32, indices of the (dirty) lines to scan
    cross_dirty_ptr,  # *uint8, per-line flags for the perpendicular orientation
    line_len,  # pixels per line (W for row passes, H for column passes)
    line_stride,  # elements between consecutive lines (W for rows, 1 for cols)
    elem_stride,  # elements between pixels of one line (1 for rows, W for cols)
    L: tl.constexpr,  # next power of two >= line_len
    RED_V: tl.constexpr,
    BLUE_V: tl.constexpr,
):
    """Fill whole horizontal or vertical runs in a single pass.

    One program owns one full line.  A segmented max-scan (walls reset the
    running max) in both directions tells every red pixel whether a blue
    pixel exists anywhere in its contiguous run, so blue floods arbitrarily
    far along the line in one launch instead of one pixel per launch.

    A line can only gain new fills after a perpendicular pass lands a blue
    on it, so each fill marks the crossing line dirty; the host compacts
    dirty lines with ``gather_dirty`` and launches exactly one program per
    dirty line, so settled lines cost nothing.  (The scan must stay outside
    conditionals: Triton 3.7 miscompiles tl.associative_scan inside an if.)
    """
    pid = tl.program_id(0)
    line = tl.load(line_list_ptr + pid)

    offs = tl.arange(0, L)
    mask = offs < line_len
    ptrs = grid_ptr + line * line_stride + offs * elem_stride

    v = tl.load(ptrs, mask=mask, other=0)  # padding = wall
    reset = (v == 0).to(tl.int8)
    _, fwd = tl.associative_scan((reset, v), 0, _seg_max_combine)
    _, bwd = tl.associative_scan((reset, v), 0, _seg_max_combine, reverse=True)

    fill = (v == RED_V) & ((fwd == BLUE_V) | (bwd == BLUE_V))
    if tl.sum(fill.to(tl.int32)) > 0:
        out = tl.where(fill, BLUE_V, v).to(grid_ptr.dtype.element_ty)
        tl.store(ptrs, out, mask=fill)
        ones = tl.full((L,), 1, tl.uint8)
        tl.store(cross_dirty_ptr + offs, ones, mask=fill)


def run_flood_fill_scan(grid_dev, changed_dev, *, max_rounds: int = 100_000):
    """Alternate row and column line-scan passes until nothing changes.

    Each round floods entire horizontal then vertical runs, so convergence
    takes roughly one round per bend in the blob's geodesics rather than one
    launch per tile of distance.  Each pass first compacts its dirty lines
    with ``gather_dirty`` (``changed_dev`` doubles as the count), then
    launches one program per dirty line.  When neither pass finds a dirty
    line, no red pixel can still reach blue: converged.
    Returns ``(kernel_launches, converged)``.
    """
    H, W = grid_dev.shape
    dirty_rows = cp.ones(H, dtype=cp.uint8)
    dirty_cols = cp.ones(W, dtype=cp.uint8)
    line_list = cp.empty(max(H, W), dtype=cp.int32)
    # (own flags, crossing flags, lines, line_len, line_stride, elem_stride, L)
    passes = (
        (dirty_rows, dirty_cols, H, W, W, 1, triton.next_power_of_2(W)),
        (dirty_cols, dirty_rows, W, H, 1, W, triton.next_power_of_2(H)),
    )
    launches = 0
    for _ in range(max_rounds):
        any_dirty = False
        for mine, cross, n_lines, line_len, line_stride, elem_stride, L in passes:
            changed_dev.fill(0)
            gather_dirty[(triton.cdiv(n_lines, 1024),)](
                t(mine), t(line_list), t(changed_dev), n_lines, BLOCK=1024)
            launches += 1
            n_dirty = int(changed_dev[0])  # D2H read; also syncs the stream
            if n_dirty == 0:
                continue
            any_dirty = True
            scan_line_step[(n_dirty,)](
                t(grid_dev),
                t(line_list),
                t(cross),
                line_len,
                line_stride,
                elem_stride,
                L=L,
                RED_V=RED,
                BLUE_V=BLUE,
                num_warps=4,
            )
            launches += 1
        if not any_dirty:
            return launches, True
    return launches, False


def run_flood_fill(
    grid_dev,
    changed_dev,
    *,
    block: int = 32,
    num_warps: int = 4,
    converge: bool = True,
    max_launches: int = 200_000,
):
    """Relaunch ``flood_fill_step`` until a launch converts no pixel.

    ``grid_dev`` (cupy uint8 HxW) is filled in place; the seed pixel must
    already be BLUE.  Returns ``(launches, converged)``.
    """
    H, W = grid_dev.shape
    launch_grid = (triton.cdiv(H, block), triton.cdiv(W, block))
    for launches in range(1, max_launches + 1):
        changed_dev.fill(0)
        flood_fill_step[launch_grid](
            t(grid_dev),
            t(changed_dev),
            H,
            W,
            BLOCK_H=block,
            BLOCK_W=block,
            RED_V=RED,
            BLUE_V=BLUE,
            CONVERGE=converge,
            num_warps=num_warps,
        )
        if int(changed_dev[0]) == 0:  # D2H read; also syncs the stream
            return launches, True
    return max_launches, False
