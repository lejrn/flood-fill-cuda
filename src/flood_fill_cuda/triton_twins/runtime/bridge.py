"""
Run Triton kernels on CuPy memory, with no PyTorch in the process.

Triton's CUDA backend finds its device and stream through torch
(``GPUDriver.__init__`` does ``import torch``). Nothing else on the
compile/launch path needs torch: it needs raw device pointers and a CUDA
context. So a small driver subclass backed by CuPy's runtime API, plus a
``data_ptr()`` adapter for arrays, is the whole bridge.

CuPy and Numba both bind the device's primary context, so a Triton kernel
and a Numba kernel can run in one process on the same arrays' memory. The
Numba-vs-Triton comparison harness relies on that.

Usage:
    from flood_fill_cuda.triton_twins.runtime import t
    kernel[(grid,)](t(cupy_array), n, BLOCK=256, num_warps=4)
"""

from __future__ import annotations

import cupy as cp
from triton.backends.nvidia.driver import CudaDriver, CudaLauncher, CudaUtils
from triton.runtime.driver import driver as _driver_config


class TensorArg:
    """The slice of the torch.Tensor surface that Triton's JIT inspects."""

    __slots__ = ("_arr", "dtype")

    def __init__(self, arr: cp.ndarray):
        self._arr = arr
        # str(numpy dtype) ("int32", "uint8", ...) is the key Triton's type
        # table expects, the same spelling torch dtypes reduce to.
        self.dtype = arr.dtype

    def data_ptr(self) -> int:
        return self._arr.data.ptr


def t(arr: cp.ndarray) -> TensorArg:
    """Wrap a CuPy array so it can be passed as a Triton pointer argument."""
    if not isinstance(arr, cp.ndarray):
        raise TypeError(f"expected a cupy.ndarray, got {type(arr).__name__}")
    return TensorArg(arr)


class CupyCudaDriver(CudaDriver):
    def __init__(self):
        # GPUDriver.__init__ is skipped on purpose: it imports torch.
        self.utils = CudaUtils()
        self.launcher_cls = CudaLauncher
        self.get_current_device = cp.cuda.runtime.getDevice
        self.set_current_device = cp.cuda.runtime.setDevice

    @staticmethod
    def get_current_stream(device=None) -> int:
        return cp.cuda.get_current_stream().ptr

    @staticmethod
    def get_device_capability(device=0):
        cc = cp.cuda.Device(device).compute_capability  # e.g. "89"
        return int(cc[:-1]), int(cc[-1])

    @staticmethod
    def is_active() -> bool:
        return True


_installed = False


def install() -> None:
    """Point Triton at CuPy's current device and stream. Idempotent."""
    global _installed
    if _installed:
        return
    cp.cuda.Device(0).use()  # a CUDA context must exist before any launch
    _driver_config.set_active(CupyCudaDriver())
    _installed = True


def sync() -> None:
    """Block until all device work is done (the repo's timing convention)."""
    cp.cuda.Device().synchronize()
