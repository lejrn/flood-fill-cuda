"""Micro-probes the chapter's explanations rest on, recorded in the JSON.

Each probe isolates one cost the README argues from:

  barrier      the per-round ritual alone (grid.sync, read the queue
               count, grid.sync) at every grid the benchmark launches -
               what a level costs ch03 and a round costs ch07 before any
               work is done
  column fill  one warp, the closure's 20-shuffle column fill, with and
               without the syncwarp() that kernels._col_fill now starts
               with (the unsynced copy below is the code as first written)
  latency      one warp, a chain of dependent L2 loads and of dependent
               atomics - the unit a tile visit's serial chain is made of
  registers    the rounds kernels' PTX compiled by ptxas alone (nothing
               linked) against the linked kernel's count, to show where
               the registers come from

All timings are CUDA events over a long loop, differenced between two
loop lengths so launch overhead cancels.
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import re
import shutil
import subprocess

import numpy as np
from numba import cuda, int32, uint32

from .. import kernels as k7
from .. import flood_fill as ff


# ------------------------------------------------------------ barrier

@cuda.jit
def _barrier_rounds(q_state, n_rounds, out):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    r = 0
    acc = 0
    while r < n_rounds:
        grid.sync()
        acc += q_state[r & 1]
        if tid == 0:
            q_state[1 - (r & 1)] = 0
        grid.sync()
        r += 1
    if tid == 0:
        out[0] = acc


def _per_iter_us(launch, n0, n1):
    """Event time per loop iteration, differenced between two lengths."""
    launch(10)
    launch(n0)
    cuda.synchronize()
    ts = []
    for n in (n0, n1):
        e0 = cuda.event(timing=True)
        e1 = cuda.event(timing=True)
        e0.record()
        launch(n)
        e1.record()
        e1.synchronize()
        ts.append(cuda.event_elapsed_time(e0, e1))
    return (ts[1] - ts[0]) / (n1 - n0) * 1000.0


def barrier_us(grids):
    """{'BxT': us per round} for each (blocks, tpb)."""
    q = cuda.to_device(np.zeros(2, np.int32))
    out = cuda.to_device(np.zeros(1, np.int64))
    _barrier_rounds[1, 32](q, 1, out)
    cuda.synchronize()
    ov = next(iter(_barrier_rounds.overloads.values()))
    res = {}
    for blocks, tpb in grids:
        if blocks > ov.max_cooperative_grid_blocks(tpb):
            continue
        res[f"{blocks}x{tpb}"] = _per_iter_us(
            lambda n: _barrier_rounds[blocks, tpb](q, n, out), 1000, 4000)
    return res


# -------------------------------------------------------- column fill

@cuda.jit(device=True, inline=True)
def _col_step_up_unsynced(f, p, lane, k):
    fu = k7._up(f, k)
    pu = k7._up(p, k)
    if lane >= k:
        f = uint32(f | (p & fu))
        p = uint32(p & pu)
    return f, p


@cuda.jit(device=True, inline=True)
def _col_step_down_unsynced(f, p, lane, k):
    fd = k7._down(f, k)
    pd = k7._down(p, k)
    if lane + k < 32:
        f = uint32(f | (p & fd))
        p = uint32(p & pd)
    return f, p


@cuda.jit(device=True, inline=True)
def _col_fill_unsynced(f, m, lane):
    """kernels._col_fill as first written: no syncwarp()."""
    p = m
    f, p = _col_step_up_unsynced(f, p, lane, 1)
    f, p = _col_step_up_unsynced(f, p, lane, 2)
    f, p = _col_step_up_unsynced(f, p, lane, 4)
    f, p = _col_step_up_unsynced(f, p, lane, 8)
    f, p = _col_step_up_unsynced(f, p, lane, 16)
    p = m
    f, p = _col_step_down_unsynced(f, p, lane, 1)
    f, p = _col_step_down_unsynced(f, p, lane, 2)
    f, p = _col_step_down_unsynced(f, p, lane, 4)
    f, p = _col_step_down_unsynced(f, p, lane, 8)
    f, p = _col_step_down_unsynced(f, p, lane, 16)
    return f


@cuda.jit
def _loop_col_synced(ms, fs, n, out):
    lane = cuda.laneid
    m = ms[lane]
    f = fs[lane]
    for _ in range(n):
        f = k7._col_fill(f, m, lane)
        f = uint32(f ^ (m >> k7.S1))
    out[lane] = f


@cuda.jit
def _loop_col_unsynced(ms, fs, n, out):
    lane = cuda.laneid
    m = ms[lane]
    f = fs[lane]
    for _ in range(n):
        f = _col_fill_unsynced(f, m, lane)
        f = uint32(f ^ (m >> k7.S1))
    out[lane] = f


@cuda.jit
def _loop_closure8(ms, fs, n, out):
    """One full 8-conn closure iteration per trip, as the rounds kernel
    runs it (row fill, diagonal step, column fill, any_sync)."""
    lane = cuda.laneid
    m = ms[lane]
    f0 = fs[lane]
    acc = k7.U0
    for _ in range(n):
        f = k7._row_fill(f0, m)
        f = k7._diag_step(f, m, lane)
        f = k7._col_fill(f, m, lane)
        if cuda.any_sync(k7.FULL, f != f0):
            acc = uint32(acc ^ f)
        f0 = uint32(f0 ^ (acc & k7.U1))
    out[lane] = acc


def _sass_calls(kern):
    sass = kern.inspect_sass()
    sass = next(iter(sass.values())) if isinstance(sass, dict) else sass
    return sass.count("shflsync")


def column_fill():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 2 ** 32, 32, dtype=np.uint64).astype(np.uint32)
    ms = cuda.to_device(m)
    fs = cuda.to_device(m & np.uint32(0x10001))
    res = {}
    outs = {}
    for name, kern in (("synced", _loop_col_synced),
                       ("unsynced", _loop_col_unsynced),
                       ("closure8_iteration", _loop_closure8)):
        out = cuda.device_array(32, np.uint32)
        res[f"{name}_ns"] = 1000.0 * _per_iter_us(
            lambda n: kern[1, 32](ms, fs, n, out), 2000, 10000)
        kern[1, 32](ms, fs, 777, out)
        outs[name] = out.copy_to_host()
        try:
            # call SITES of the slow path in the SASS (present in both
            # versions; which path RUNS is decided at run time)
            res[f"{name}_sass_slowpath_sites"] = _sass_calls(kern)
        except Exception:
            pass
    res["synced_equals_unsynced"] = bool(
        np.array_equal(outs["synced"], outs["unsynced"]))
    return res


# ------------------------------------------------------------ latency

@cuda.jit
def _chase_load(nxt, n, out):
    i = uint32(cuda.laneid)
    for _ in range(n):
        i = nxt[i]
    out[cuda.laneid] = i


@cuda.jit
def _chase_atomic_or(nxt, n, out):
    i = uint32(cuda.laneid)
    for _ in range(n):
        i = cuda.atomic.or_(nxt, i, uint32(0))
    out[cuda.laneid] = i


def latency():
    perm = np.random.default_rng(0).permutation(1 << 16).astype(np.uint32)
    d = cuda.to_device(perm)
    out = cuda.device_array(32, np.uint32)
    return {
        "dependent_l2_load_ns": 1000.0 * _per_iter_us(
            lambda n: _chase_load[1, 32](d, n, out), 2000, 10000),
        "dependent_atomic_or_ns": 1000.0 * _per_iter_us(
            lambda n: _chase_atomic_or[1, 32](d, n, out), 2000, 10000),
    }


# ---------------------------------------------------------- registers

def _ptxas():
    for cand in (shutil.which("ptxas"), "/usr/local/cuda/bin/ptxas",
                 "/usr/local/cuda-12.9/bin/ptxas"):
        if cand and os.path.exists(cand):
            return cand
    return None


def registers(configs):
    """Linked registers (Numba) vs ptxas -c on the same PTX, per kernel."""
    ptxas = _ptxas()
    res = {}
    for conn, bare in configs:
        kern = k7.get_rounds_kernel(conn, bare, None)
        ff.regs_per_thread(conn, bare, None)   # compiles it
        name = f"conn{conn}{'_bare' if bare else ''}"
        entry = {"linked": ff.regs_per_thread(conn, bare, None)}
        if ptxas:
            ptx = next(iter(kern.inspect_asm().values()))
            cmd = [ptxas, "-arch=sm_89", "-v", "-c", "/dev/stdin",
                   "-o", os.devnull]
            err = subprocess.run(cmd, input=ptx, capture_output=True,
                                 text=True).stderr
            regs = re.findall(r"Used (\d+) registers", err)
            spill = re.findall(r"(\d+) bytes spill stores", err)
            entry["ptxas_standalone"] = int(regs[-1]) if regs else None
            entry["ptxas_standalone_spill_bytes"] = (int(spill[-1])
                                                     if spill else None)
        res[name] = entry
    return res


def run_all(grids):
    return {
        "barrier_us_per_round": barrier_us(grids),
        "column_fill": column_fill(),
        "latency": latency(),
        "registers": registers([(8, False), (8, True), (4, False)]),
    }
