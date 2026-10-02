"""Tests for the Triton twins' shared runtime."""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import cupy as cp
import numpy as np
import pytest
import triton
import triton.language as tl
from numba import cuda

from flood_fill_cuda.triton_twins.runtime import (
    device_info, kernel_resources, max_coresident_programs, sync, t,
)
from flood_fill_cuda.triton_twins.runtime.device import (
    atomic_cas_masked, grid_sync, read_clock64, read_globaltimer, read_smid,
)


@triton.jit
def _add_one(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=m) + 1, mask=m)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.int64])
def test_dtypes_round_trip(dtype):
    n = 1000
    src = cp.arange(n, dtype=dtype) % 100
    dst = cp.zeros_like(src)
    _add_one[(triton.cdiv(n, 256),)](t(src), t(dst), n, BLOCK=256)
    sync()
    np.testing.assert_array_equal(cp.asnumpy(dst), cp.asnumpy(src) + 1)


def test_t_rejects_host_arrays():
    with pytest.raises(TypeError):
        t(np.zeros(4, dtype=np.int32))


@triton.jit
def _phases(buf_ptr, bar_ptr, err_ptr, smid_ptr, clk_ptr, n_phases,
            BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    neighbour = (pid + 1) % nprog
    c0 = read_clock64(pid)
    epoch = 0
    for ph in range(n_phases):
        tl.store(buf_ptr + pid * BLOCK + offs, tl.zeros([BLOCK], tl.int32) + ph)
        epoch += 1
        grid_sync(bar_ptr, epoch * nprog)
        got = tl.load(buf_ptr + neighbour * BLOCK + offs, cache_modifier=".cg")
        tl.atomic_add(err_ptr, tl.sum((got != ph).to(tl.int32)))
        epoch += 1
        grid_sync(bar_ptr, epoch * nprog)
    tl.store(smid_ptr + pid, read_smid(pid))
    tl.store(clk_ptr + pid, read_clock64(pid) - c0)


@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_grid_sync_at_full_residency(num_warps):
    """Every program sees its neighbour's write of the same phase, at the
    largest co-resident grid, so the barrier both orders and never deadlocks."""
    BLOCK = 256

    def run(nprog, phases):
        buf = cp.zeros(nprog * BLOCK, cp.int32)
        bar = cp.zeros(1, cp.int32)
        err = cp.zeros(1, cp.int32)
        smid = cp.full(nprog, -1, cp.int32)
        clk = cp.zeros(nprog, cp.int64)
        k = _phases[(nprog,)](t(buf), t(bar), t(err), t(smid), t(clk), phases,
                              BLOCK=BLOCK, num_warps=num_warps,
                              launch_cooperative_grid=True)
        sync()
        return k, int(bar[0]), int(err[0]), cp.asnumpy(smid), cp.asnumpy(clk)

    k, *_ = run(1, 1)  # compile, and learn the kernel's resources
    nprog = max_coresident_programs(k)
    assert nprog >= device_info().sm_count
    phases = 300
    _, arrivals, errors, smid, clk = run(nprog, phases)
    assert errors == 0
    assert arrivals == 2 * phases * nprog
    assert smid.min() >= 0 and smid.max() < device_info().sm_count
    assert (clk > 0).all()
    assert kernel_resources(k)["num_warps"] == num_warps


def test_numba_and_triton_share_a_buffer():
    """A Numba-allocated buffer, viewed through CuPy, is a valid Triton
    argument: both runtimes bind the same primary context."""

    @cuda.jit
    def numba_fill(a):
        i = cuda.grid(1)
        if i < a.size:
            a[i] = i

    n = 4096
    d = cuda.device_array(n, dtype=np.int32)
    numba_fill[(n + 255) // 256, 256](d)
    cuda.synchronize()
    view = cp.asarray(d)  # zero-copy via __cuda_array_interface__
    out = cp.zeros(n, cp.int32)
    _add_one[(triton.cdiv(n, 256),)](t(view), t(out), n, BLOCK=256)
    sync()
    np.testing.assert_array_equal(cp.asnumpy(out), np.arange(n) + 1)


def test_oversized_cooperative_grid_raises_and_context_survives():
    """The twin of Numba's cooperative-launch refusal: an error, not a hang."""
    BLOCK = 256

    def run(nprog):
        buf = cp.zeros(nprog * BLOCK, cp.int32)
        bar = cp.zeros(1, cp.int32)
        err = cp.zeros(1, cp.int32)
        smid = cp.zeros(nprog, cp.int32)
        clk = cp.zeros(nprog, cp.int64)
        k = _phases[(nprog,)](t(buf), t(bar), t(err), t(smid), t(clk), 2,
                              BLOCK=BLOCK, num_warps=4,
                              launch_cooperative_grid=True)
        sync()
        return k, int(err[0])

    k, _ = run(1)
    nprog = max_coresident_programs(k)
    with pytest.raises(RuntimeError, match="too many blocks"):
        run(nprog + device_info().sm_count)
    assert run(nprog)[1] == 0


@triton.jit
def _claim(flag_ptr, won_ptr, sink_ptr, t_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    m = lanes < n
    # every program races for the same n slots; masked lanes hit the sink
    old = atomic_cas_masked(flag_ptr + lanes, tl.zeros([BLOCK], tl.int32),
                            tl.zeros([BLOCK], tl.int32) + pid + 1, m,
                            sink_ptr + pid * BLOCK + lanes)
    tl.atomic_add(won_ptr, tl.sum(((old == 0) & m).to(tl.int32)))
    tl.store(t_ptr + pid, read_globaltimer(pid))


def test_atomic_cas_masked_claims_exactly_once_and_spares_the_sink():
    BLOCK, nprog, n = 128, 64, 100
    flag = cp.zeros(BLOCK, cp.int32)
    won = cp.zeros(1, cp.int32)
    sink = cp.zeros(nprog * BLOCK, cp.int32)
    stamps = cp.zeros(nprog, cp.int64)
    _claim[(nprog,)](t(flag), t(won), t(sink), t(stamps), n, BLOCK=BLOCK)
    sync()
    assert int(won[0]) == n  # each of the n slots won by exactly one program
    f = cp.asnumpy(flag)
    assert (f[:n] >= 1).all() and (f[:n] <= nprog).all() and (f[n:] == 0).all()
    assert int(sink.sum()) == 0
    assert (cp.asnumpy(stamps) > 0).all()


def test_triton_copy_peak_is_a_real_bandwidth():
    from flood_fill_cuda.triton_twins.runtime.bandwidth import (
        measure_peak_bandwidth,
    )
    peak = measure_peak_bandwidth(n_bytes=64 * 2 ** 20, repeats=3)
    assert 20 < peak["gb_s"] < 1000
