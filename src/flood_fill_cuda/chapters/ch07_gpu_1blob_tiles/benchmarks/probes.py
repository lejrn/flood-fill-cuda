"""Micro-probes the chapter's explanations rest on, recorded in the JSON.

Each probe isolates one cost the README argues from:

  latency      one warp, a chain of dependent L2 loads and of dependent
               atomics - the unit a tile visit's serial chain is made of
  column fill  one warp, the closure's 20-shuffle column fill, with and
               without the syncwarp() that kernels._col_fill now starts
               with (the unsynced copy below is the code as first written),
               and one whole 8-conn closure iteration
  barrier      the per-round ritual alone (grid.sync, read the queue
               count, grid.sync) at every grid the benchmark launches -
               what a level costs ch03 and a round costs ch07 before any
               work is done; and the one_sync ritual (thread 0 resets a
               counter, grid.sync, read the next one: three rotating
               counters)
  registers    the rounds kernels' PTX compiled by ptxas alone (nothing
               linked) against the linked kernel's count, to show where
               the registers come from

Two clocks, because single-warp kernels relax this GPU's clock under
them. Every single-warp probe reports both:
  *_ns      CUDA events over a long loop, differenced between two loop
            lengths so launch overhead cancels (wall time: depends on the
            SM clock the probe happened to run at)
  *_cycles  %clock64 cycles per iteration, read inside the kernel (via
            ch05's linked smid.cu). These do not move with the SM clock;
            cycles / ns is the clock the probe ran at (`*_implied_mhz`).
The single-warp probes run first, right after the clock spin-up, and
warm() compiles them before it, so no JIT gap sits between the two.
The barrier probe keeps the whole grid busy; it is repeated
BARRIER_REPEATS times per grid and the median and the range are
reported. (Three repeats were not stable enough between sessions, the
fairness audit found.)
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import re
import shutil
import statistics
import subprocess

import numpy as np
from numba import cuda, uint32

from .. import kernels as k7
from .. import flood_fill as ff
from ...ch05_gpu_nblob_nblock.kernels import SMID_CU, get_clock64

BARRIER_REPEATS = 7
SHORT, LONG = 2000, 10000      # single-warp loop lengths (differenced)


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


@cuda.jit
def _barrier_rounds_1(q_state, n_rounds, out):
    """The one_sync ritual: reset counter (r+2) % 3, one grid.sync, read
    counter (r+1) % 3 (kernels._make_rounds_kernel, one_sync)."""
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    r = 0
    cur = 0
    acc = 0
    while r < n_rounds:
        nxt = cur + 1
        if nxt == 3:
            nxt = 0
        if tid == 0:
            zap = nxt + 1
            if zap == 3:
                zap = 0
            q_state[zap] = 0
        grid.sync()
        acc += q_state[nxt]
        cur = nxt
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


def barrier_us(grids, kernel=None):
    """{'BxT': {'median': us per round, 'runs': [...]}} per (blocks, tpb),
    for the two-sync ritual (default) or `kernel`."""
    kernel = _barrier_rounds if kernel is None else kernel
    q = cuda.to_device(np.zeros(3, np.int32))
    out = cuda.to_device(np.zeros(1, np.int64))
    kernel[1, 32](q, 1, out)
    cuda.synchronize()
    ov = next(iter(kernel.overloads.values()))
    res = {}
    for blocks, tpb in grids:
        if blocks > ov.max_cooperative_grid_blocks(tpb):
            continue
        runs = [_per_iter_us(
            lambda n: kernel[blocks, tpb](q, n, out), 1000, 4000)
            for _ in range(BARRIER_REPEATS)]
        res[f"{blocks}x{tpb}"] = {"median": statistics.median(runs),
                                  "min": min(runs), "max": max(runs),
                                  "runs": runs}
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


# Every single-warp kernel stores its result BEFORE the second clock
# read, so the read waits for the loop's last value to exist.

@cuda.jit(link=[SMID_CU])
def _loop_col_synced(ms, fs, n, out, cyc):
    lane = cuda.laneid
    m = ms[lane]
    f = fs[lane]
    t0 = get_clock64()
    for _ in range(n):
        f = k7._col_fill(f, m, lane)
        f = uint32(f ^ (m >> k7.S1))
    out[lane] = f
    t1 = get_clock64()
    if lane == 0:
        cyc[0] = t1 - t0


@cuda.jit(link=[SMID_CU])
def _loop_col_unsynced(ms, fs, n, out, cyc):
    lane = cuda.laneid
    m = ms[lane]
    f = fs[lane]
    t0 = get_clock64()
    for _ in range(n):
        f = _col_fill_unsynced(f, m, lane)
        f = uint32(f ^ (m >> k7.S1))
    out[lane] = f
    t1 = get_clock64()
    if lane == 0:
        cyc[0] = t1 - t0


@cuda.jit(link=[SMID_CU])
def _loop_closure8(ms, fs, n, out, cyc):
    """One full 8-conn closure iteration per trip, as the rounds kernel
    runs it (row fill, diagonal step, column fill, any_sync)."""
    lane = cuda.laneid
    m = ms[lane]
    f0 = fs[lane]
    acc = k7.U0
    t0 = get_clock64()
    for _ in range(n):
        f = k7._row_fill(f0, m)
        f = k7._diag_step(f, m, lane)
        f = k7._col_fill(f, m, lane)
        if cuda.any_sync(k7.FULL, f != f0):
            acc = uint32(acc ^ f)
        f0 = uint32(f0 ^ (acc & k7.U1))
    out[lane] = acc
    t1 = get_clock64()
    if lane == 0:
        cyc[0] = t1 - t0


def _sass_calls(kern):
    sass = kern.inspect_sass()
    sass = next(iter(sass.values())) if isinstance(sass, dict) else sass
    return sass.count("shflsync")


def _single_warp(res, name, kern, args, out, cyc):
    """Events (ns, differenced) and in-kernel cycles for one probe."""
    ns = 1000.0 * _per_iter_us(lambda n: kern[1, 32](*args, n, out, cyc),
                               SHORT, LONG)
    kern[1, 32](*args, LONG, out, cyc)
    cycles = int(cyc.copy_to_host()[0]) / LONG
    res[f"{name}_ns"] = ns
    res[f"{name}_cycles"] = cycles
    res[f"{name}_implied_mhz"] = cycles / ns * 1000.0 if ns > 0 else None


def _col_inputs():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 2 ** 32, 32, dtype=np.uint64).astype(np.uint32)
    return cuda.to_device(m), cuda.to_device(m & np.uint32(0x10001))


def column_fill():
    ms, fs = _col_inputs()
    cyc = cuda.device_array(1, np.uint64)
    res = {}
    outs = {}
    for name, kern in (("synced", _loop_col_synced),
                       ("unsynced", _loop_col_unsynced),
                       ("closure8_iteration", _loop_closure8)):
        out = cuda.device_array(32, np.uint32)
        _single_warp(res, name, kern, (ms, fs), out, cyc)
        kern[1, 32](ms, fs, 777, out, cyc)
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

@cuda.jit(link=[SMID_CU])
def _chase_load(nxt, n, out, cyc):
    i = uint32(cuda.laneid)
    t0 = get_clock64()
    for _ in range(n):
        i = nxt[i]
    out[cuda.laneid] = i
    t1 = get_clock64()
    if cuda.laneid == 0:
        cyc[0] = t1 - t0


@cuda.jit(link=[SMID_CU])
def _chase_atomic_or(nxt, n, out, cyc):
    i = uint32(cuda.laneid)
    t0 = get_clock64()
    for _ in range(n):
        i = cuda.atomic.or_(nxt, i, uint32(0))
    out[cuda.laneid] = i
    t1 = get_clock64()
    if cuda.laneid == 0:
        cyc[0] = t1 - t0


def _chase_input():
    perm = np.random.default_rng(0).permutation(1 << 16).astype(np.uint32)
    return cuda.to_device(perm)


def latency():
    d = _chase_input()
    out = cuda.device_array(32, np.uint32)
    cyc = cuda.device_array(1, np.uint64)
    res = {}
    _single_warp(res, "dependent_l2_load", _chase_load, (d,), out, cyc)
    _single_warp(res, "dependent_atomic_or", _chase_atomic_or, (d,), out,
                 cyc)
    return res


# ---------------------------------------------------------- registers

def _ptxas():
    for cand in (shutil.which("ptxas"), "/usr/local/cuda/bin/ptxas",
                 "/usr/local/cuda-12.9/bin/ptxas"):
        if cand and os.path.exists(cand):
            return cand
    return None


# the rounds-kernel switch sets the register probe compiles
# (flood_fill.ROUNDS_SWITCHES): the original kernel, the default's and
# the fused pack's
OLD_SWITCHES = {"one_sync": False, "fused_pack": False, "skip_full": False,
                "fused_paint": False}
DEFAULT_SWITCHES = ff.rounds_switches(ff.DEFAULT_SWITCHES)
FUSED_SWITCHES = {**OLD_SWITCHES, "one_sync": True, "fused_pack": True}


def registers(configs):
    """Linked registers (Numba) vs ptxas -c on the same PTX, per kernel.
    configs: (connectivity, bare[, switches]); switches default to all
    off (the original kernel)."""
    ptxas = _ptxas()
    res = {}
    for conn, bare, *rest in configs:
        sw = rest[0] if rest else OLD_SWITCHES
        kern = k7.get_rounds_kernel(conn, bare, None, **sw)
        ff.regs_per_thread(conn, bare, None, **sw)   # compiles it
        name = f"conn{conn}{'_bare' if bare else ''}"
        if any(sw.values()):
            name += "_" + "_".join(k for k, v in sw.items() if v)
        entry = {"linked": ff.regs_per_thread(conn, bare, None, **sw)}
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


# ---------------------------------------------------------------- run

def warm():
    """Compile (and link) every probe kernel off the clock, so no JIT gap
    sits between the clock spin-up and the probes."""
    ms, fs = _col_inputs()
    d = _chase_input()
    out = cuda.device_array(32, np.uint32)
    cyc = cuda.device_array(1, np.uint64)
    for kern in (_loop_col_synced, _loop_col_unsynced, _loop_closure8):
        kern[1, 32](ms, fs, 1, out, cyc)
    _chase_load[1, 32](d, 1, out, cyc)
    _chase_atomic_or[1, 32](d, 1, out, cyc)
    q = cuda.to_device(np.zeros(3, np.int32))
    acc = cuda.to_device(np.zeros(1, np.int64))
    _barrier_rounds[1, 32](q, 1, acc)
    _barrier_rounds_1[1, 32](q, 1, acc)
    cuda.synchronize()


def run_all(grids):
    # Single-warp probes first, while the clock is still at the spin-up's
    # level (they relax it); then the whole-grid barrier probe.
    lat = latency()
    col = column_fill()
    return {
        "latency": lat,
        "column_fill": col,
        "barrier_us_per_round": barrier_us(grids),
        "barrier_1sync_us_per_round": barrier_us(grids, _barrier_rounds_1),
        "registers": registers([(8, False), (8, True), (4, False),
                                (8, False, DEFAULT_SWITCHES),
                                (8, True, DEFAULT_SWITCHES),
                                (4, False, DEFAULT_SWITCHES),
                                (8, True, FUSED_SWITCHES)]),
    }
