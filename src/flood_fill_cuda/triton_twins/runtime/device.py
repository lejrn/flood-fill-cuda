"""
Device-side building blocks the Numba chapters get from CUDA and Triton does not.

Each helper is a ``@triton.jit`` function, inlined into the calling kernel.

grid_sync    the twin of Numba's cooperative ``grid.sync()``
cta_sync     the twin of ``cuda.syncthreads()``
load_acquire an acquire load, for reading another program's published value
read_smid    the twin of the chapters' linked ``smid.cu`` reader
read_clock64 the twin of a ``%clock64`` cycle stamp

Why grid_sync is a hand-rolled barrier: Triton has no cooperative launch.
A spin barrier is only safe when every program of the grid is resident at
once, which is exactly the guarantee a cooperative launch gives. The host
must therefore size the grid with ``occupancy.max_coresident_programs``,
the same way the Numba chapters size theirs from the cooperative maximum.

The counter is monotonic. The host zeroes it before each launch; every
program counts its own barriers in a loop variable ``epoch`` and waits for
``epoch * num_programs`` arrivals:

    epoch += 1
    grid_sync(bar_ptr, epoch * NPROG)

A monotonic counter needs no sense reversal and no reset race. An int32
counter allows 2**31 / num_programs barriers per launch (about 3.7 M
barriers at 576 programs); pass an int64 counter for longer runs.
"""

import triton
import triton.language as tl


@triton.jit
def cta_sync():
    """Barrier across the threads of this program (``bar.sync 0``)."""
    tl.debug_barrier()


@triton.jit
def load_acquire(ptr):
    """Scalar load with gpu-scope acquire ordering."""
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
