"""Compatibility shim: the CuPy bridge now lives in the shared Triton runtime.

This module used to hold its own copy of the bridge that lets Triton launch
on CuPy memory without PyTorch. That copy now lives, tested, in
``flood_fill_cuda.triton_twins.runtime`` (``bridge.py``), which the Triton
twins of the chapters use too. Two copies in one process would each install
their own Triton driver, so this module only re-exports the shared one.

Old imports keep working:

    from flood_fill_cuda.experiments.triton.cupy_bridge import t, install_cupy_driver
"""

from flood_fill_cuda.triton_twins.runtime.bridge import (
    CupyCudaDriver,
    TensorArg,
    install,
    sync,
    t,
)

# The old name of runtime.install(). Idempotent; importing the runtime
# package has already called it.
install_cupy_driver = install

__all__ = ["CupyCudaDriver", "TensorArg", "install", "install_cupy_driver", "sync", "t"]
