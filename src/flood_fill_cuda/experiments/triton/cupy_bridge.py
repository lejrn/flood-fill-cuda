"""Bridge letting Triton drive CuPy-allocated GPU memory without PyTorch.

Triton >= 3 ships a CUDA backend whose device/stream discovery is hard-wired
to torch (``GPUDriver.__init__`` does ``import torch``).  Everything else in
the compile/launch path only needs raw pointers and a CUDA context, so a thin
driver subclass backed by CuPy's runtime API plus a ``data_ptr()`` adapter for
arrays is enough to run Triton kernels in this otherwise CuPy/Numba project.
"""

from __future__ import annotations

import cupy as cp
from triton.backends.nvidia.driver import CudaDriver, CudaLauncher, CudaUtils
from triton.runtime.driver import driver as _driver_config


class TensorArg:
    """Adapter exposing the torch-tensor surface Triton's JIT inspects."""

    __slots__ = ("_arr", "dtype")

    def __init__(self, arr: cp.ndarray):
        self._arr = arr
        self.dtype = arr.dtype  # str(numpy dtype) maps into Triton's type table

    def data_ptr(self) -> int:
        return self._arr.data.ptr


def t(arr: cp.ndarray) -> TensorArg:
    """Wrap a CuPy array for passing as a Triton kernel argument."""
    return TensorArg(arr)


class CupyCudaDriver(CudaDriver):
    def __init__(self):
        self.utils = CudaUtils()
        self.launcher_cls = CudaLauncher
        # GPUDriver.__init__ is skipped on purpose: it imports torch.
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


def install_cupy_driver() -> None:
    """Make Triton launch on CuPy's current device/stream. Idempotent."""
    global _installed
    if _installed:
        return
    cp.cuda.Device(0).use()  # ensure a CUDA context exists before any launch
    _driver_config.set_active(CupyCudaDriver())
    _installed = True
