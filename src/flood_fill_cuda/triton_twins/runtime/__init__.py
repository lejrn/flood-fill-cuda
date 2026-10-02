"""
Shared runtime for the Triton twins: launch on CuPy memory, plus the device
helpers Triton lacks (grid barrier, %smid, %clock64) and an occupancy
calculator for sizing persistent grids.

Importing this package installs the CuPy driver, so a twin only needs:

    from flood_fill_cuda.triton_twins.runtime import t, sync
"""

from .bridge import TensorArg, install, sync, t
from .occupancy import (
    DeviceInfo, device_info, kernel_resources, max_coresident_programs,
    programs_per_sm,
)

install()

__all__ = [
    "TensorArg", "install", "sync", "t",
    "DeviceInfo", "device_info", "kernel_resources",
    "max_coresident_programs", "programs_per_sm",
]
