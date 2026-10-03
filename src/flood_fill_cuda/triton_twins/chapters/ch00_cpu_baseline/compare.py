"""
Numba vs Triton on the ch00 single-block BFS prototype.

The prototype's only benchmark is profile_kernel(num_runs=100): a fresh
400x400 random-walk scene for every run, the launch + synchronize timed, the
average over 100 runs reported. The cases mirror it on seeded scenes (the
warm-up uses seed 1000, timed round r seed 1001 + r) and split what
profile_kernel's bracket holds, because two parts of it are not the same
work on both sides:

- Numba's kernel prints queue_front at exit (a device printf, flushed by the
  synchronize). The twin stores the value instead.
- profile_kernel passes new_color as a host array. Numba then allocates it
  with cuMemAlloc, copies it in, launches and copies it back synchronously;
  the twin's cp.asarray / .get(out=) does the same steps through CuPy's
  pool, at a lower cost.

Like-for-like rows (comparable, the primary comparison):
- profile_kernel:  a fresh seeded scene per round.
- fixed_scene:     the seed-0 scene every round (no blob-to-blob spread).
  new_color is uploaded once per case, outside the brackets, on both sides,
  and the Numba kernel is the prototype's own source without its exit print
  (built at import from single_block.py: only the two lines
  `if global_tid == 0: print(queue_front[0])` are removed).

Reference rows (comparable=False, the profile_kernel scenes):
- profile_kernel_with_printf: the Numba kernel as written (with its printf),
  device new_color.
- profile_kernel_as_written: profile_kernel's exact call: the printf, and
  new_color as a host array on both sides.

The two costs themselves (the printf's share of Numba's kernel_ms and each
stack's host new_color round trip) are NOT taken as differences between
rows: the rows run minutes apart, and one block lets the GPU clock down,
so the drift between rows is as large as the costs. After the rows, one
interleaved loop runs six variants back to back on one scene every round
(Numba with / without the printf x host / device new_color, Triton host /
device new_color; the order rotates every round), with the rows' own
runners and brackets. Its medians and within-round differences are logged
and stored in the JSON's meta["decomposition"]; the row differences are
kept there too, labelled approximate.

Every row's info also holds device_us: the GPU-only time of one launch per
backend (CUDA events, with a copy queued ahead so launch latency is hidden),
measured right before the timed rounds on the case's first scene. One block
uses 1 of 24 SMs, so the GPU clocks down during the timed rounds: kernel_ms
and the speedups are clock-sensitive, read them with clocks_after.

Both backends run the prototype's configuration: one block of 64 threads
(Triton: one program, num_warps 2). kernel_ms is profile_kernel's bracket
(launch + synchronize); total_ms adds the cuda.to_device of img and visited
and the copies back.

same() compares the deterministic outputs (the prototype's contract while
the blob fits its 6000-slot queue, which every scene here does: the blobs
are 1.5k-3.2k pixels): visited, the R and G channels, the recolored mask,
every untouched pixel, and, where the Numba kernel prints it, queue_front
(captured from its device print) against the value the twin stores. The
debug blue channel ((tid * 4) % 255) is schedule-dependent and not compared.

Run (writes results/triton_twins/ch00_cpu_baseline/compare_<UTC>.json):

    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch00_cpu_baseline.compare
    .venv/bin/python -m flood_fill_cuda.triton_twins.chapters.ch00_cpu_baseline.compare --quick

--quick is a smoke test (2 rounds per row, one rotation of the
decomposition, no JSON). --repeats N sets the rounds of every case
(default 100, profile_kernel's num_runs); --decomposition-rounds N sets the
interleaved loop's rounds (default 300, 0 skips it).
"""

import os

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import argparse
import contextlib
import ctypes
import inspect
import json
import statistics
import sys
import tempfile
import time
import types

import numpy as np

from flood_fill_cuda.chapters.ch00_cpu_baseline import single_block as numba_proto
from flood_fill_cuda.shared.cpu_oracle import cpu_flood_fill_8
from flood_fill_cuda.triton_twins.compare.harness import (
    Case, arrays_equal, free_device_memory, gpu_clocks, run_cases, spin_up,
)
from flood_fill_cuda.triton_twins.runtime import kernel_resources, sync

from .single_block import (
    QUEUE_CAPACITY, THREADS_PER_BLOCK, PrototypeRun, _launch, _to_device,
    compiled_kernel, run_flood_fill, setup_scene,
)

CHAPTER = "ch00_cpu_baseline"
NUM_RUNS = 100          # profile_kernel's default num_runs
PROFILE_SEED = 1000     # warm-up scene 1000, round r scene 1001 + r
WIDTH = HEIGHT = 400
DEVICE_ROUNDS = 20      # event-timed launches per backend, per case
CASE_SPIN_SECONDS = 2.0
DECOMP_ROUNDS = 300     # interleaved rounds of the cost decomposition
DECOMP_SEED = 0         # its scene: fixed_scene's

_libc = ctypes.CDLL(None)

# The prototype's exit print, removed for the print-less build.
_EXIT_PRINT = "    if global_tid == 0:\n        print(queue_front[0])\n"
_noprint = {}


def numba_noprint_kernel():
    """The Numba prototype's flood_fill, compiled from its own module source
    with the exit print removed and nothing else changed."""
    if "kernel" not in _noprint:
        src = inspect.getsource(numba_proto)
        if src.count(_EXIT_PRINT) != 1:
            raise RuntimeError("single_block.py's exit print changed: "
                               "cannot build the print-less kernel")
        mod = types.ModuleType("ch00_single_block_without_exit_print")
        exec(compile(src.replace(_EXIT_PRINT, ""),
                     "<ch00 single_block.py without its exit print>", "exec"),
             mod.__dict__)
        _noprint["kernel"] = mod.flood_fill
    return _noprint["kernel"]


@contextlib.contextmanager
def c_stdout_captured():
    """Point fd 1 at a temp file (CUDA printf writes through the C stdout);
    the captured text is appended to the yielded list on exit."""
    captured = []
    sys.stdout.flush()
    _libc.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile() as f:
        os.dup2(f.fileno(), 1)
        try:
            yield captured
        finally:
            _libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            f.seek(0)
            captured.append(f.read().decode())


def numba_side(kernel, device_color):
    """A Numba runner with run_flood_fill's brackets. kernel is the prototype
    as written or its print-less build; device_color uploads new_color once
    per case, outside the brackets, instead of passing the host array.

    The printing kernel's queue_front is parsed back from its output (fd 1
    is redirected before the first bracket and restored after the last)."""
    from numba import cuda

    colors = {}

    def run(scene):
        img, visited, sx, sy, w, h, new_color, tpb, bpg = scene
        if device_color:
            key = bytes(np.asarray(new_color, dtype=np.uint8))
            if key not in colors:
                colors[key] = cuda.to_device(np.array(new_color, dtype=np.uint8))
            color = colors[key]
        else:
            color = np.array(new_color, copy=True)
        with c_stdout_captured() as out:
            t0 = time.perf_counter()
            d_img = cuda.to_device(img)
            d_visited = cuda.to_device(visited)
            cuda.synchronize()
            t1 = time.perf_counter()
            kernel[bpg, tpb](d_img, d_visited, sx, sy, w, h, color)
            cuda.synchronize()
            t2 = time.perf_counter()
            img_out = d_img.copy_to_host()
            visited_out = d_visited.copy_to_host()
            t3 = time.perf_counter()
        printed = out[0].split()
        return PrototypeRun(
            img=img_out, visited=visited_out,
            front=int(printed[-1]) if printed else None, threads_per_block=tpb,
            h2d_ms=(t1 - t0) * 1000, kernel_ms=(t2 - t1) * 1000,
            d2h_ms=(t3 - t2) * 1000, total_ms=(t3 - t0) * 1000)
    return run


def triton_side(device_color):
    """run_flood_fill, with new_color either setup_scene's host array (the
    implicit round trip) or a CuPy copy uploaded once per case."""
    import cupy as cp

    colors = {}

    def run(scene):
        if device_color:
            key = bytes(np.asarray(scene[6], dtype=np.uint8))
            if key not in colors:
                colors[key] = cp.asarray(np.array(scene[6], dtype=np.uint8))
            scene = scene[:6] + (colors[key],) + scene[7:]
        return run_flood_fill(*scene)
    return run


def same_run(nb, tri):
    if nb.front is not None and nb.front != tri.front:
        return False, f"front: numba printed {nb.front} vs triton {tri.front}"
    nb_rec = (nb.img[..., 0] == 0) & (nb.img[..., 1] == 0)
    tri_rec = (tri.img[..., 0] == 0) & (tri.img[..., 1] == 0)
    return arrays_equal(
        visited=(nb.visited, tri.visited),
        img_rg=(nb.img[..., :2], tri.img[..., :2]),
        recolored=(nb_rec, tri_rec),
        untouched=(nb.img[~nb_rec], tri.img[~nb_rec]))


class SceneSequence:
    """Scene i of a seeded sequence, built once and shared by both runners
    (each runner keeps its own position, and the harness calls each exactly
    once per round, so both always run the same scene)."""

    def __init__(self, first_seed, fixed=False):
        self.first_seed = first_seed
        self.fixed = fixed
        self.cache = {}

    def scene(self, i):
        key = 0 if self.fixed else i
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k >= key - 1}
            scene = setup_scene(self.first_seed + key)
            img, _, sx, sy = scene[:4]
            filled = cpu_flood_fill_8(img, sx, sy)[3]
            if filled > QUEUE_CAPACITY:  # Numba would read out of bounds
                raise RuntimeError(f"scene {self.first_seed + key} has "
                                   f"{filled} px, over the 6000-slot queue")
            self.cache[key] = scene
        return self.cache[key]

    def runner(self, run):
        position = [0]

        def call():
            scene = self.scene(position[0])
            position[0] += 1
            return run(scene)
        return call


def device_us(scene, numba_kernel, rounds):
    """GPU-only time of one launch per backend, in microseconds: CUDA events
    around the launch, with a 32 MB copy queued ahead of it so the launch
    latency is hidden. new_color is on the device on both sides; the order
    alternates every round. Median and min over the rounds."""
    import cupy as cp
    from numba import cuda

    img, visited, sx, sy, w, h, new_color, tpb, bpg = scene
    color = np.array(new_color, dtype=np.uint8)
    nb_color = cuda.to_device(color)
    tri_color = cp.asarray(color)
    busy_a = cp.zeros(4 * 2 ** 20, dtype=cp.int64)
    busy_b = cp.empty_like(busy_a)
    times = {"numba": [], "triton": []}
    with c_stdout_captured():
        for r in range(rounds):
            for side in (("numba", "triton") if r % 2 == 0
                         else ("triton", "numba")):
                e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                if side == "numba":
                    d_img = cuda.to_device(img)
                    d_visited = cuda.to_device(visited)
                    cuda.synchronize()
                    busy_b[...] = busy_a
                    e0.record()
                    numba_kernel[bpg, tpb](d_img, d_visited, sx, sy, w, h,
                                           nb_color)
                else:
                    c_img = _to_device(img)
                    c_visited = _to_device(visited)
                    sync()
                    busy_b[...] = busy_a
                    e0.record()
                    _launch(c_img, c_visited, sx, sy, w, h, tri_color, tpb, bpg)
                e1.record()
                e1.synchronize()
                times[side].append(cp.cuda.get_elapsed_time(e0, e1) * 1000)
    del busy_a, busy_b
    return {side: {"median": statistics.median(v), "min": min(v)}
            for side, v in times.items()} | {"rounds": rounds}


def _one(values):
    values = values.values() if isinstance(values, dict) else [values]
    values = sorted(set(int(v) for v in values))
    return values[0] if len(values) == 1 else values


def numba_resources(kernel):
    return {"regs_per_thread": _one(kernel.get_regs_per_thread()),
            "shared_bytes": _one(kernel.get_shared_mem_per_block()),
            "local_bytes_per_thread": _one(kernel.get_local_mem_per_thread())}


def info_for(seq, numba_kernel, device_rounds, spin_seconds):
    def info(nb, tri):
        # Called once per case, after the warm-up and before the timed
        # rounds: re-spin the GPU (one block lets it clock down), then
        # measure the GPU-only time on the case's first scene.
        if spin_seconds > 0:
            spin_up(spin_seconds)
        return {
            "front": tri.front,
            "recolored": int(((nb.img[..., 0] == 0) & (nb.img[..., 1] == 0)).sum()),
            "visited": int(nb.visited.sum()),
            "device_us": device_us(seq.scene(0), numba_kernel, device_rounds),
            "numba": {"grid": [1, THREADS_PER_BLOCK],
                      **numba_resources(numba_kernel),
                      "h2d_ms": nb.h2d_ms, "d2h_ms": nb.d2h_ms},
            "triton": {"grid": [1], "BLOCK": THREADS_PER_BLOCK,
                       "num_warps": THREADS_PER_BLOCK // 32,
                       **kernel_resources(compiled_kernel(THREADS_PER_BLOCK)),
                       "h2d_ms": tri.h2d_ms, "d2h_ms": tri.d2h_ms},
        }
    return info


PLAN = [  # (experiment, seed, fixed, numba printf, device new_color, comparable, note)
    ("profile_kernel", PROFILE_SEED, False, False, True, True,
     "profile_kernel's method (a fresh seeded scene every round), like for "
     "like: new_color uploaded once outside the brackets on both sides, the "
     "Numba kernel built without its exit print"),
    ("fixed_scene", 0, True, False, True, True,
     "setup_scene(rng_seed=0) every round, like for like (as profile_kernel)"),
    ("profile_kernel_with_printf", PROFILE_SEED, False, True, True, False,
     "reference: the Numba kernel as written, with its exit printf (its "
     "share is measured interleaved: meta.decomposition)"),
    ("profile_kernel_as_written", PROFILE_SEED, False, True, False, False,
     "reference: profile_kernel's exact call, printf and host new_color "
     "(implicit round trip on both sides; each stack's round trip is "
     "measured interleaved: meta.decomposition)"),
]


def build_cases(quick=False):
    config = {"threads_per_block": THREADS_PER_BLOCK, "blocks_per_grid": 1,
              "connectivity": 8, "queue_capacity": QUEUE_CAPACITY}
    cases = []
    for (experiment, seed, fixed, printf, device_color, comparable,
         note) in PLAN:
        kernel = numba_proto.flood_fill if printf else numba_noprint_kernel()
        seq = SceneSequence(seed, fixed=fixed)
        cases.append(Case(
            experiment=experiment,
            scene=(f"random_walk_400 seed {seed}" if fixed
                   else f"random_walk_400 seeds {seed}.."),
            config={**config, "numba_exit_printf": printf,
                    "new_color": "device" if device_color else "host"},
            run_numba=seq.runner(numba_side(kernel, device_color)),
            run_triton=seq.runner(triton_side(device_color)),
            same=same_run, pixels=WIDTH * HEIGHT,
            info=info_for(seq, kernel, 3 if quick else DEVICE_ROUNDS,
                          0 if quick else CASE_SPIN_SECONDS),
            notes=note, comparable=comparable))
    return cases


DECOMP_VARIANTS = (  # (name, backend, Numba exit printf, device new_color)
    ("numba_noprint_device", "numba", False, True),
    ("numba_printf_device", "numba", True, True),
    ("numba_noprint_host", "numba", False, False),
    ("numba_printf_host", "numba", True, False),
    ("triton_device", "triton", None, True),
    ("triton_host", "triton", None, False),
)

# (key, minuend, subtrahend): each cost is one variant minus another.
DECOMP_COSTS = (
    ("numba_exit_printf_ms", "numba_printf_device", "numba_noprint_device"),
    ("numba_exit_printf_ms_host_color", "numba_printf_host",
     "numba_noprint_host"),
    ("numba_host_new_color_round_trip_ms", "numba_noprint_host",
     "numba_noprint_device"),
    ("numba_host_new_color_round_trip_ms_with_printf", "numba_printf_host",
     "numba_printf_device"),
    ("triton_host_new_color_round_trip_ms", "triton_host", "triton_device"),
)


def measure_decomposition(rounds, spin_seconds, seed=DECOMP_SEED):
    """The printf's share of Numba's kernel_ms and each stack's host
    new_color round trip, measured interleaved on one scene.

    Every round runs the six DECOMP_VARIANTS back to back on
    setup_scene(seed), with the rows' own runners (numba_side,
    triton_side) and so the rows' brackets; the order rotates every round,
    so each variant takes each slot equally often. A cost is one variant
    minus another: reported as the median and quartiles of the within-round
    differences (paired_ms, paired_iqr_ms: the drift between rounds
    cancels) and as the difference of the two medians (diff_of_medians_ms,
    noisier). The costs themselves scale with the clock, so they vary from
    run to run; clocks_after records it. Outputs are checked every round."""
    scene = setup_scene(seed)
    img, _, sx, sy = scene[:4]
    filled = cpu_flood_fill_8(img, sx, sy)[3]
    if filled > QUEUE_CAPACITY:  # Numba would read out of bounds
        raise RuntimeError(f"scene {seed} has {filled} px, over the queue")
    runners = {}
    for name, backend, printf, device_color in DECOMP_VARIANTS:
        if backend == "numba":
            kernel = (numba_proto.flood_fill if printf
                      else numba_noprint_kernel())
            runners[name] = numba_side(kernel, device_color)
        else:
            runners[name] = triton_side(device_color)
    names = [v[0] for v in DECOMP_VARIANTS]
    for name in names:  # compiles and per-case uploads stay off the clock
        runners[name](scene)
    if spin_seconds > 0:
        spin_up(spin_seconds)
    times = {name: [] for name in names}
    mismatches, detail = 0, ""
    for r in range(rounds):
        shift = r % len(names)
        got = {}
        for name in names[shift:] + names[:shift]:
            got[name] = runners[name](scene)
            times[name].append(float(got[name].kernel_ms))
        for name in names:
            if name != "triton_device":
                ok, d = same_run(got[name], got["triton_device"])
                if not ok:
                    mismatches += 1
                    detail = detail or f"{name}: {d}"
        del got
    clocks = gpu_clocks()
    free_device_memory()

    medians = {name: statistics.median(v) for name, v in times.items()}
    costs = {}
    for key, a, b in DECOMP_COSTS:
        diffs = [x - y for x, y in zip(times[a], times[b])]
        quartiles = (statistics.quantiles(diffs, n=4) if len(diffs) > 1
                     else diffs * 3)
        costs[key] = {
            "paired_ms": statistics.median(diffs),
            "paired_iqr_ms": [quartiles[0], quartiles[2]],
            "diff_of_medians_ms": medians[a] - medians[b],
            "variants": [a, b]}
    return {
        "method": ("interleaved: every round runs the six variants back to "
                   "back on one scene, order rotating; costs are within-round "
                   "differences of kernel_ms (median), so clock drift "
                   "cancels; device_us in each row's info gives the GPU-only "
                   "time"),
        "scene": f"random_walk_400 seed {seed}",
        "rounds": rounds,
        "medians_ms": medians,
        **costs,
        "outputs_equal": mismatches == 0,
        "mismatched_checks": mismatches,
        **({"mismatch_detail": detail} if detail else {}),
        "clocks_after": clocks,
    }


def row_differences(rows):
    """The same costs as differences of kernel_ms medians between rows.
    Approximate: the rows run minutes apart at different clocks (one block
    lets the GPU clock down), so the drift is as large as the costs. Kept
    as a cross-check of measure_decomposition."""
    by = {r["experiment"]: r for r in rows}
    need = ("profile_kernel", "profile_kernel_with_printf",
            "profile_kernel_as_written")
    if any(n not in by or "error" in by[n] for n in need):
        return None

    def med(name, side):
        return by[name][side]["kernel_ms"]["median"]

    return {
        "approximate": True,
        "note": ("differences of kernel_ms medians between rows run minutes "
                 "apart: confounded by the clock drift between them; the "
                 "interleaved figures above are the measurement"),
        "numba_exit_printf_ms": (med("profile_kernel_with_printf", "numba")
                                 - med("profile_kernel", "numba")),
        "host_new_color_round_trip_ms": {
            side: (med("profile_kernel_as_written", side)
                   - med("profile_kernel_with_printf", side))
            for side in ("numba", "triton")},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 rounds per row, one rotation of "
                         "the decomposition, no JSON")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"timed rounds per case (default {NUM_RUNS}, "
                         f"profile_kernel's num_runs)")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write the comparison JSON")
    ap.add_argument("--decomposition-rounds", type=int, default=None,
                    help=f"interleaved rounds of the printf / round-trip "
                         f"decomposition (default {DECOMP_ROUNDS}, --quick "
                         f"{len(DECOMP_VARIANTS)}; 0 skips it)")
    args = ap.parse_args(argv)

    cases = build_cases(quick=args.quick)
    repeats = args.repeats or (1 if args.quick else NUM_RUNS)
    repeats += repeats % 2  # the harness rounds up to an even count
    meta = {
        "source_benchmark": ("chapters/ch00_cpu_baseline/single_block.py "
                             "profile_kernel(num_runs=100)"),
        "quick": args.quick,
        "profile_seeds": {"warm_up": PROFILE_SEED,
                          "timed": [PROFILE_SEED + 1, PROFILE_SEED + repeats]},
        "caps": [],  # 400x400 scenes, nothing to cap
        "notes": [
            "Grid is 1 block of 64 threads (Numba [1, 64]) = 1 program "
            "(Triton (1,), num_warps 2) on both sides.",
            "Primary rows (comparable): new_color on the device on both "
            "sides, uploaded outside the brackets, and the Numba kernel "
            "built from single_block.py without its exit print, so both "
            "sides do the same work inside kernel_ms.",
            "Reference rows (comparable=false): profile_kernel_with_printf "
            "(Numba as written: its exit printf) and "
            "profile_kernel_as_written (printf + host new_color on both "
            "sides). Numba's implicit host-array transfer allocates with "
            "cuMemAlloc and copies synchronously; the twin's goes through "
            "CuPy's pool, so the same steps cost Numba more. "
            "meta.decomposition measures the printf's share and each "
            "stack's round trip interleaved (six variants back to back on "
            "one scene every round), not as differences between rows, "
            "which run minutes apart at different clocks.",
            "Numba's 6000-slot queue and scalars are shared memory; the "
            "twin's live in a preallocated global scratch (Triton has no "
            "user-addressable shared memory). Per-lane rear atomics on both "
            "sides: shared-memory atomics in Numba, L2 atomics in Triton.",
            "One block uses 1 of 24 SMs, so the GPU clocks down during the "
            "timed rounds (each case re-spins it for "
            f"{CASE_SPIN_SECONDS:g} s first): kernel_ms and the speedups "
            "are clock-sensitive, read them with clocks_after. "
            "info.device_us is the GPU-only time (CUDA events, queue kept "
            "busy), measured right before the timed rounds.",
            "profile_kernel's own statistic is the mean; the harness reports "
            "the median and min over the rounds.",
        ],
    }
    write = not (args.quick or args.no_write)
    doc = run_cases(CHAPTER, cases, repeats=repeats, meta=meta, write=write,
                    spin_seconds=0 if args.quick else 8.0)
    rounds = args.decomposition_rounds
    if rounds is None:
        rounds = len(DECOMP_VARIANTS) if args.quick else DECOMP_ROUNDS
    if rounds > 0:
        try:
            decomp = measure_decomposition(
                rounds, 0 if args.quick else CASE_SPIN_SECONDS)
        except Exception as exc:  # recorded, not hidden
            decomp = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"decomposition: ERROR {decomp['error']}")
        else:
            print(f"decomposition ({rounds} interleaved rounds, within-round "
                  f"medians): Numba exit printf "
                  f"{decomp['numba_exit_printf_ms']['paired_ms']:+.3f} ms; "
                  f"host new_color round trip numba "
                  f"{decomp['numba_host_new_color_round_trip_ms']['paired_ms']:+.3f}"
                  f" ms, triton "
                  f"{decomp['triton_host_new_color_round_trip_ms']['paired_ms']:+.3f}"
                  f" ms; outputs equal={decomp['outputs_equal']}")
        approx = row_differences(doc["rows"])
        if approx is not None:
            decomp["row_differences"] = approx
        doc["meta"]["decomposition"] = decomp
        if "path" in doc:  # add it to the JSON run_cases wrote
            path = doc.pop("path")
            with open(path, "w") as f:
                json.dump(doc, f, indent=1)
            doc["path"] = path
    for row in doc["rows"]:
        d = row.get("info", {}).get("device_us")
        if d:
            print(f"{row['experiment']}: device_us numba "
                  f"{d['numba']['median']:.1f} | triton "
                  f"{d['triton']['median']:.1f}")
    return doc


if __name__ == "__main__":
    main()
