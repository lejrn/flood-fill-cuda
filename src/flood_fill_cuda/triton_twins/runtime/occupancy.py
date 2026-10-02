"""
How many programs of a compiled Triton kernel fit on the GPU at once.

This is the Triton counterpart of Numba's
``kernel.max_cooperative_grid_blocks(tpb)``: a persistent kernel that
spins in ``device.grid_sync`` deadlocks if any program has to wait for a
slot, so its grid must not exceed this number.

``compiled`` is the ``CompiledKernel`` a launch returns
(``k = kernel[grid](...)``), or the one ``kernel.warmup(...)`` returns.
Its register count, spills and static shared memory are what the driver's
occupancy calculator needs.
"""

from dataclasses import dataclass
from functools import lru_cache

import cupy as cp


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    sm_count: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int
    shared_per_sm: int


@lru_cache(maxsize=None)
def device_info(device: int = 0) -> DeviceInfo:
    p = cp.cuda.runtime.getDeviceProperties(device)
    name = p["name"]
    return DeviceInfo(
        name=name.decode() if isinstance(name, bytes) else name,
        sm_count=p["multiProcessorCount"],
        max_threads_per_sm=p["maxThreadsPerMultiProcessor"],
        max_blocks_per_sm=p["maxBlocksPerMultiProcessor"],
        regs_per_sm=p["regsPerMultiprocessor"],
        shared_per_sm=p["sharedMemPerMultiprocessor"],
    )


def _loaded(compiled):
    """CompiledKernel loads its module lazily (after ``warmup`` the handle
    and register count may not exist yet)."""
    if getattr(compiled, "function", None) is None and hasattr(compiled, "_init_handles"):
        compiled._init_handles()
    return compiled


def kernel_resources(compiled) -> dict:
    """Registers, spills, shared bytes and warps of a compiled kernel."""
    compiled = _loaded(compiled)
    return {
        "n_regs": int(compiled.n_regs),
        "n_spills": int(compiled.n_spills),
        "shared_bytes": int(compiled.metadata.shared),
        "num_warps": int(compiled.metadata.num_warps),
    }


def programs_per_sm(compiled) -> int:
    """Resident programs per SM, from the driver's occupancy calculator."""
    compiled = _loaded(compiled)
    threads = int(compiled.metadata.num_warps) * 32
    return int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        compiled.function, threads, int(compiled.metadata.shared)))


def max_coresident_programs(compiled) -> int:
    """Largest grid whose programs are all resident at once."""
    return programs_per_sm(compiled) * device_info().sm_count
