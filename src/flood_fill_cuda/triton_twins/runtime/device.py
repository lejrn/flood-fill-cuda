"""
Device-side building blocks the Numba chapters get from CUDA and Triton does not.

Each helper is a ``@triton.jit`` function, inlined into the calling kernel.

grid_sync         the twin of Numba's cooperative ``grid.sync()``
cta_sync          the twin of ``cuda.syncthreads()``
load_acquire      an acquire load, for reading another program's published value
atomic_cas_masked ``tl.atomic_cas`` with a lane mask (Triton's CAS has none)
read_smid         the twin of the chapters' linked ``smid.cu`` reader
read_clock64      the twin of a ``%clock64`` cycle stamp
read_globaltimer  the twin of a ``%globaltimer`` nanosecond stamp

Why grid_sync is a hand-rolled barrier: Triton has no ``grid.sync()``. A
spin barrier is only safe when every program of the grid is resident at
once, which is the guarantee a cooperative launch gives. Triton 3.7 can
make that launch: pass ``launch_cooperative_grid=True`` with every kernel
that calls grid_sync. The driver then refuses an oversized grid with
"too many blocks in cooperative launch" (a RuntimeError, context intact),
exactly like Numba, instead of spinning forever. Size the grid with
``occupancy.max_coresident_programs``, the twin of Numba's
``max_cooperative_grid_blocks``.

The counter is monotonic. The host zeroes it before each launch; every
program counts its own barriers in a loop variable ``epoch`` and waits for
``epoch * num_programs`` arrivals:

    epoch += 1
    grid_sync(bar_ptr, epoch * NPROG)

A monotonic counter needs no sense reversal and no reset race. An int32
counter allows 2**31 / num_programs barriers per launch (about 3.7 M
barriers at 576 programs); pass an int64 counter for longer runs.

Visibility: after ``grid_sync`` returns, plain ``tl.load`` (L1-cached)
sees every write made before the barrier by any program. The gpu-scope
acquire invalidates L1; this was stress-tested at 288 programs x 2,000
phases rewriting the same addresses. Data another program publishes
WITHOUT a barrier in between (a flag you spin on, an inbox you poll) must
be read with ``load_acquire``, an atomic, or ``cache_modifier=".cg"``.
"""

import triton
import triton.language as tl


@triton.jit
def cta_sync():
    """Barrier across the threads of this program (``bar.sync 0``)."""
    tl.debug_barrier()


@triton.jit
def load_acquire(ptr):
    """Scalar load with gpu-scope acquire ordering.

    This is a load, not a fence: Triton lowers it to ``ld.acquire.gpu``
    and removes it when the result is unused. Use the value (for example
    to drive a spin loop), or the ordering silently disappears.
    """
    return tl.atomic_add(ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def grid_sync(bar_ptr, target):
    """Wait until ``target`` arrivals have been counted on ``bar_ptr``.

    The leading CTA barrier makes every thread's earlier global writes
    happen-before this program's release arrival; the trailing one keeps any
    thread from running ahead while thread 0 still spins.
    """
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    seen = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while seen < target:
        seen = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def atomic_cas_masked(ptrs, cmp, val, mask, sink_ptrs, sem: tl.constexpr = "relaxed"):
    """Compare-and-swap on the lanes in ``mask``; returns the old values.

    ``tl.atomic_cas`` takes no mask, so inactive lanes are pointed at
    ``sink_ptrs``: one int slot per lane, holding 0 and never written,
    which they CAS with cmp=-1 (never equal to 0). Give each lane its own
    sink slot (for example ``sink + pid * BLOCK + tl.arange(0, BLOCK)``) so
    the dummy atomics never contend. Ignore the result on inactive lanes.

    For a 0/1 visited flag, ``tl.atomic_xchg(ptrs, 1, mask=mask)`` claims
    with the same exactly-once guarantee (old == 0 wins) and issues no
    dummy atomics; use this helper when the swapped value matters.
    """
    p = tl.where(mask, ptrs, sink_ptrs)
    c = tl.where(mask, cmp, -1)
    v = tl.where(mask, val, 0)
    return tl.atomic_cas(p, c, v, sem=sem, scope="gpu")


@triton.jit
def read_smid(dummy):
    """Index of the SM this program is running on (``%smid``).

    ``dummy`` is any int32 value (``tl.program_id(0)`` is the usual one);
    inline PTX needs an operand to take its shape from.
    """
    return tl.inline_asm_elementwise(
        "mov.u32 $0, %smid;", "=r,r", [dummy],
        dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def read_clock64(dummy):
    """This SM's cycle counter (``%clock64``), as int64."""
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;", "=l,r", [dummy],
        dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def read_globaltimer(dummy):
    """The GPU's global nanosecond timer (``%globaltimer``), as int64."""
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [dummy],
        dtype=tl.int64, is_pure=False, pack=1)
