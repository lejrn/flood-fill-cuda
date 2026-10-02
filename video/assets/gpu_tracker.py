"""The drone tracker on the GPU, fed straight from the ch06 run table.

On the host, tracking a 576 x 1024 frame costs about 19 ms after the
kernel: the label map to the host (0.7), blob prep (10: dense labels,
the size filter, centroids), the centroid tracker (5.9) and painting (2).
Everything those steps need is already on the device once ch06 has run:
every run's column, its y span and its root run. So each step becomes a
handful of small kernels, and the frame crosses to the host once, as the
painted picture.

  stats     one thread per run: area, sum x, sum y and bounding box,
            added into the root run's slot with atomics (the root found
            by walking parent, see _root)
  keep      a root that passes MIN_AREA / MAX_BBOX is a blob
  scan      blob index = rank of its root in run order, which is the
            order of the host's np.unique over the kernel's labels
  centroid  one thread per blob, exact integer sums / area
  bucket    each track's prediction into a grid of GATE + 1 px cells
  pairs     one thread per blob: the tracks within GATE in the 3 x 3
            cells around it
  match     greedy closest-first, in one block: every round each pair
            atomicMins (d2, track) into its blob and (d2, blob) into its
            track; a pair that holds both minima is accepted; rounds
            repeat until none is (the host's vectorised rounds, exactly)
  update    one thread per track: velocity, coasting, miss count
  compact   surviving tracks in order, then one new track per unmatched
            blob in blob order (two scans)
  paint     one thread per run, the blob's track colour

Exactness. Ties break as on the host (by track index for a blob, by blob
index for a track), and each float operation the host does in numpy is
done here with libdevice's `_rn` functions, which the compiler never
fuses into an FMA. So the GPU's centroids and ids equal the host's bit
for bit; `make_drone_frames.py --tracker both` checks every frame.
"""
from __future__ import annotations

import math

import numpy as np
from numba import cuda, float32, int32, int64, uint64
from numba.cuda import libdevice as ld

from flood_fill_cuda.chapters.ch06_gpu_nblob_runs.kernels import N_RUNS_USED

GRID = (128, 256)               # grid-stride launches; every count is read on the device
SCAN_T = 1024                   # single-block scans
MATCH_T = 256                   # the match loop: its 64-bit keys need more registers than 1024 threads get
MAX_U64 = np.uint64(0xFFFFFFFFFFFFFFFF)

# slots of the tracker's counter array
N_BLOBS, N_CAND, N_ALIVE, N_NEW, M, NEXT_ID, N_CELLS, CAND_CAP, N_TRACKED = range(9)
N_SLOTS = 9


# ------------------------------------------------------------------ blobs
@cuda.jit(device=True, inline=True)
def _root(parent, r):
    """Follow parent to its fixed point. After ch06's flatten most runs point
    at their root, but not all: its path compression races (a stale halving
    can overwrite a root pointer with an ancestor), and only run_label is
    guaranteed. The chains left are short, and roots are fixed points."""
    while parent[r] != r:
        r = parent[r]
    return r


@cuda.jit
def stats_init_kernel(counters, area, sx, sy2, bx0, bx1, by0, by1):
    for r in range(cuda.grid(1), counters[N_RUNS_USED], cuda.gridsize(1)):
        area[r] = 0
        sx[r] = 0
        sy2[r] = 0
        bx0[r] = 2 ** 30
        bx1[r] = -1
        by0[r] = 2 ** 30
        by1[r] = -1


@cuda.jit
def stats_kernel(counters, run_x, run_y0, run_y1, parent, rroot, area, sx, sy2, bx0, bx1, by0, by1):
    for r in range(cuda.grid(1), counters[N_RUNS_USED], cuda.gridsize(1)):
        root = _root(parent, r)
        rroot[r] = root
        x, y0, y1 = run_x[r], run_y0[r], run_y1[r]
        n = y1 - y0 + 1
        cuda.atomic.add(area, root, n)
        cuda.atomic.add(sx, root, int64(x) * n)
        cuda.atomic.add(sy2, root, int64(y0 + y1) * n)      # twice the sum of y over the run
        cuda.atomic.min(bx0, root, x)
        cuda.atomic.max(bx1, root, x)
        cuda.atomic.min(by0, root, y0)
        cuda.atomic.max(by1, root, y1)


@cuda.jit
def keep_kernel(counters, parent, area, bx0, bx1, by0, by1, min_area, max_bbox, keep):
    for r in range(cuda.grid(1), counters[N_RUNS_USED], cuda.gridsize(1)):
        ok = (parent[r] == r and area[r] >= min_area
              and bx1[r] - bx0[r] + 1 <= max_bbox and by1[r] - by0[r] + 1 <= max_bbox)
        keep[r] = 1 if ok else 0


@cuda.jit
def scan_kernel(src, n_arr, n_idx, dst, tot_arr, tot_idx):
    """Exclusive prefix sum of src[:n] into dst, total into tot_arr[tot_idx].
    One block: each thread sums a contiguous chunk, the block scans the
    chunk sums, each thread writes its chunk."""
    n = n_arr[n_idx]
    tid = cuda.threadIdx.x
    per = (n + SCAN_T - 1) // SCAN_T
    lo = tid * per
    hi = min(lo + per, n)
    s = 0
    for i in range(lo, hi):
        s += src[i]
    sh = cuda.shared.array(SCAN_T, dtype=int32)
    sh[tid] = s
    cuda.syncthreads()
    off = 1
    while off < SCAN_T:
        v = sh[tid - off] if tid >= off else 0
        cuda.syncthreads()
        sh[tid] += v
        cuda.syncthreads()
        off *= 2
    run = sh[tid] - s
    for i in range(lo, hi):
        dst[i] = run
        run += src[i]
    if tid == SCAN_T - 1:
        tot_arr[tot_idx] = sh[tid]


@cuda.jit
def centroid_kernel(counters, keep, dense, area, sx, sy2, cx, cy, bmatch, newflag):
    for r in range(cuda.grid(1), counters[N_RUNS_USED], cuda.gridsize(1)):
        if keep[r]:
            b = dense[r]
            a = float(area[r])
            cx[b] = ld.ddiv_rn(float(sx[r]), a)
            cy[b] = ld.ddiv_rn(float(sy2[r] // 2), a)
            bmatch[b] = -1
            newflag[b] = 0


# ----------------------------------------------------------------- tracks
@cuda.jit
def frame_init_kernel(cnt, cell_count, cell_fill, hit):
    i0, step = cuda.grid(1), cuda.gridsize(1)
    for c in range(i0, cnt[N_CELLS], step):
        cell_count[c] = 0
        cell_fill[c] = 0
    for t in range(i0, cnt[M], step):
        hit[t] = -1
    if i0 == 0:
        cnt[N_CAND] = 0


@cuda.jit
def predict_kernel(cnt, px, py, vx, vy, qx, qy, tcell, cell_count, gw, gh, cell):
    for t in range(cuda.grid(1), cnt[M], cuda.gridsize(1)):
        x = float32(ld.dadd_rn(px[t], vx[t]))          # host: (pos + vel).astype(float32)
        y = float32(ld.dadd_rn(py[t], vy[t]))
        qx[t] = x
        qy[t] = y
        c = min(max(int(math.floor(x / cell)), 0), gw - 1) + gw * min(max(int(math.floor(y / cell)), 0), gh - 1)
        tcell[t] = c
        cuda.atomic.add(cell_count, c, 1)


@cuda.jit
def fill_kernel(cnt, tcell, cell_start, cell_fill, cell_tracks):
    for t in range(cuda.grid(1), cnt[M], cuda.gridsize(1)):
        c = tcell[t]
        cell_tracks[cell_start[c] + cuda.atomic.add(cell_fill, c, 1)] = t


@cuda.jit
def pairs_kernel(cnt, cx, cy, qx, qy, cell_start, cell_count, cell_tracks, gw, gh, cell, gate2,
                 cb, ct, cd2, calive):
    cap = cnt[CAND_CAP]
    for b in range(cuda.grid(1), cnt[N_BLOBS], cuda.gridsize(1)):
        ax = float32(cx[b])                             # host: centroids.astype(float32)
        ay = float32(cy[b])
        gx = min(max(int(math.floor(ax / cell)), 0), gw - 1)
        gy = min(max(int(math.floor(ay / cell)), 0), gh - 1)
        for yy in range(max(gy - 1, 0), min(gy + 2, gh)):
            for xx in range(max(gx - 1, 0), min(gx + 2, gw)):
                c = xx + gw * yy
                for k in range(cell_start[c], cell_start[c] + cell_count[c]):
                    t = cell_tracks[k]
                    dx = ld.fsub_rn(ax, qx[t])
                    dy = ld.fsub_rn(ay, qy[t])
                    d2 = ld.fadd_rn(ld.fmul_rn(dx, dx), ld.fmul_rn(dy, dy))
                    if d2 <= gate2:
                        slot = cuda.atomic.add(cnt, N_CAND, 1)
                        if slot < cap:
                            cb[slot] = b
                            ct[slot] = t
                            cd2[slot] = d2
                            calive[slot] = 1


@cuda.jit
def match_kernel(cnt, cb, ct, cbits, calive, bmatch, hit, blob_best, track_best):
    """Greedy closest-first matching in one block. A pair is accepted when
    it is both its blob's best (d2, track) and its track's best (d2, blob)
    among the pairs still open; accepted blobs and tracks close their
    other pairs; repeat until a round accepts nothing."""
    tid = cuda.threadIdx.x
    ncand = min(cnt[N_CAND], cnt[CAND_CAP])
    nb, m = cnt[N_BLOBS], cnt[M]
    accepted = cuda.shared.array(1, dtype=int32)
    while True:
        for b in range(tid, nb, MATCH_T):
            blob_best[b] = MAX_U64
        for t in range(tid, m, MATCH_T):
            track_best[t] = MAX_U64
        if tid == 0:
            accepted[0] = 0
        cuda.syncthreads()
        for c in range(tid, ncand, MATCH_T):
            if calive[c]:
                b, t = cb[c], ct[c]
                if bmatch[b] >= 0 or hit[t] >= 0:
                    calive[c] = 0
                else:
                    hi = uint64(cbits[c]) << uint64(32)  # d2 >= 0: its float bits sort like it
                    cuda.atomic.min(blob_best, b, hi | uint64(t))
                    cuda.atomic.min(track_best, t, hi | uint64(b))
        cuda.syncthreads()
        for c in range(tid, ncand, MATCH_T):
            if calive[c]:
                b, t = cb[c], ct[c]
                hi = uint64(cbits[c]) << uint64(32)
                if blob_best[b] == (hi | uint64(t)) and track_best[t] == (hi | uint64(b)):
                    bmatch[b] = t
                    hit[t] = b
                    cuda.atomic.add(accepted, 0, 1)
        cuda.syncthreads()
        done = accepted[0] == 0
        cuda.syncthreads()
        if done:
            break


@cuda.jit
def blob_ids_kernel(cnt, bmatch, tid_old, bid, newflag):
    for b in range(cuda.grid(1), cnt[N_BLOBS], cuda.gridsize(1)):
        t = bmatch[b]
        if t >= 0:
            bid[b] = tid_old[t]
        else:
            newflag[b] = 1


@cuda.jit
def update_kernel(cnt, hit, cx, cy, px, py, vx, vy, miss, alive, miss_limit):
    for t in range(cuda.grid(1), cnt[M], cuda.gridsize(1)):
        h = hit[t]
        if h >= 0:
            # host: vel = 0.6 * (c - pos) + 0.4 * vel; pos = c
            vx[t] = ld.dadd_rn(ld.dmul_rn(0.6, ld.dadd_rn(cx[h], -px[t])), ld.dmul_rn(0.4, vx[t]))
            vy[t] = ld.dadd_rn(ld.dmul_rn(0.6, ld.dadd_rn(cy[h], -py[t])), ld.dmul_rn(0.4, vy[t]))
            px[t] = cx[h]
            py[t] = cy[h]
            miss[t] = 0
        else:
            px[t] = ld.dadd_rn(px[t], vx[t])
            py[t] = ld.dadd_rn(py[t], vy[t])
            miss[t] += 1
        alive[t] = 1 if miss[t] <= miss_limit else 0


@cuda.jit
def compact_kernel(cnt, alive, rank, tid0, px0, py0, vx0, vy0, miss0, tid1, px1, py1, vx1, vy1, miss1):
    for t in range(cuda.grid(1), cnt[M], cuda.gridsize(1)):
        if alive[t]:
            j = rank[t]
            tid1[j] = tid0[t]
            px1[j] = px0[t]
            py1[j] = py0[t]
            vx1[j] = vx0[t]
            vy1[j] = vy0[t]
            miss1[j] = miss0[t]


@cuda.jit
def new_tracks_kernel(cnt, newflag, rank, cx, cy, bid, tid1, px1, py1, vx1, vy1, miss1):
    for b in range(cuda.grid(1), cnt[N_BLOBS], cuda.gridsize(1)):
        if newflag[b]:
            j = cnt[N_ALIVE] + rank[b]
            i = cnt[NEXT_ID] + rank[b]
            bid[b] = i
            tid1[j] = i
            px1[j] = cx[b]
            py1[j] = cy[b]
            vx1[j] = 0.0
            vy1[j] = 0.0
            miss1[j] = 0


@cuda.jit
def finish_kernel(cnt):
    if cuda.grid(1) == 0:
        cnt[NEXT_ID] += cnt[N_NEW]
        cnt[M] = cnt[N_ALIVE] + cnt[N_NEW]


# ------------------------------------------------------------------ paint
@cuda.jit
def clear_kernel(out):
    h, w = out.shape[0], out.shape[1]
    for i in range(cuda.grid(1), h * w, cuda.gridsize(1)):
        y, x = i // w, i % w
        out[y, x, 0] = 0
        out[y, x, 1] = 0
        out[y, x, 2] = 0


@cuda.jit
def paint_kernel(counters, run_x, run_y0, run_y1, rroot, keep, dense, bid, palette, out):
    """out is (H, W, 3) in display orientation: a run is column x, rows y0..y1."""
    n_pal = palette.shape[0]
    for r in range(cuda.grid(1), counters[N_RUNS_USED], cuda.gridsize(1)):
        root = rroot[r]
        if keep[root]:
            k = bid[dense[root]] % n_pal
            x = run_x[r]
            for y in range(run_y0[r], run_y1[r] + 1):
                out[y, x, 0] = palette[k, 0]
                out[y, x, 1] = palette[k, 1]
                out[y, x, 2] = palette[k, 2]


class _Tracks:
    def __init__(self, cap: int):
        self.tid = cuda.device_array(cap, dtype=np.int64)
        self.px = cuda.device_array(cap, dtype=np.float64)
        self.py = cuda.device_array(cap, dtype=np.float64)
        self.vx = cuda.device_array(cap, dtype=np.float64)
        self.vy = cuda.device_array(cap, dtype=np.float64)
        self.miss = cuda.device_array(cap, dtype=np.int32)

    def arrays(self) -> tuple:
        return self.tid, self.px, self.py, self.vx, self.vy, self.miss


class GpuTracker:
    """Blob prep + centroid tracker + paint on the device, after a ch06 run.

    `step()` reads the engine's run table as its last run left it and
    returns the painted (H, W, 3) frame on the host; with a `timing` dict
    it also fills `prep_gpu`, `track_gpu` and `paint_gpu` (ms, CUDA events).
    `host_view()` copies this frame's centroids and track ids, for the
    parity check."""

    def __init__(self, engine, gate: float, miss_limit: int, min_area: int, max_bbox: int,
                 palette: np.ndarray, track_cap: int = 1 << 16, cand_cap: int = 1 << 20):
        self.engine = engine
        w, h = engine.width, engine.height
        self.gate2 = np.float32(gate * gate)
        self.cell = np.float32(gate + 1)                 # a pair within the gate is in adjacent cells
        self.gw, self.gh = math.ceil(w / (gate + 1)), math.ceil(h / (gate + 1))
        self.miss_limit, self.min_area, self.max_bbox = int(miss_limit), int(min_area), int(max_bbox)
        self.track_cap, self.cand_cap = track_cap, cand_cap
        R = engine.run_capacity
        i32 = lambda n: cuda.device_array(n, dtype=np.int32)          # noqa: E731
        self.area, self.bx0, self.bx1, self.by0, self.by1 = (i32(R) for _ in range(5))
        self.sx = cuda.device_array(R, dtype=np.int64)
        self.sy2 = cuda.device_array(R, dtype=np.int64)
        self.keep, self.dense, self.rroot = i32(R), i32(R), i32(R)
        self.cx = cuda.device_array(R, dtype=np.float64)
        self.cy = cuda.device_array(R, dtype=np.float64)
        self.bid = cuda.device_array(R, dtype=np.int64)
        self.bmatch, self.newflag, self.brank = i32(R), i32(R), i32(R)
        self.blob_best = cuda.device_array(R, dtype=np.uint64)
        T = track_cap
        self.cur, self.nxt = _Tracks(T), _Tracks(T)
        self.qx = cuda.device_array(T, dtype=np.float32)
        self.qy = cuda.device_array(T, dtype=np.float32)
        self.tcell, self.hit, self.alive, self.trank = i32(T), i32(T), i32(T), i32(T)
        self.track_best = cuda.device_array(T, dtype=np.uint64)
        ncell = self.gw * self.gh
        self.cell_count, self.cell_start, self.cell_fill = i32(ncell), i32(ncell), i32(ncell)
        self.cell_tracks = i32(T)
        C = cand_cap
        self.cb, self.ct = i32(C), i32(C)
        self.cd2 = cuda.device_array(C, dtype=np.float32)
        self.cbits = self.cd2.view(np.uint32)
        self.calive = cuda.device_array(C, dtype=np.int8)
        init = np.zeros(N_SLOTS, dtype=np.int64)
        init[N_CELLS], init[CAND_CAP] = ncell, C
        self.cnt = cuda.to_device(init)
        self.palette = cuda.to_device(np.ascontiguousarray(palette, dtype=np.uint8))
        self.out = cuda.device_array((h, w, 3), dtype=np.uint8)
        self.n_blobs = 0
        self.next_id = 0

    def step(self, timing: dict | None = None) -> np.ndarray:
        e = self.engine
        g, cnt = GRID, self.cnt
        ev = [cuda.event(timing=True) for _ in range(4)]
        ev[0].record()
        # blobs
        stats_init_kernel[g](e.counters, self.area, self.sx, self.sy2, self.bx0, self.bx1, self.by0, self.by1)
        stats_kernel[g](e.counters, e.run_x, e.run_y0, e.run_y1, e.parent, self.rroot,
                        self.area, self.sx, self.sy2, self.bx0, self.bx1, self.by0, self.by1)
        keep_kernel[g](e.counters, e.parent, self.area, self.bx0, self.bx1, self.by0, self.by1,
                       self.min_area, self.max_bbox, self.keep)
        scan_kernel[1, SCAN_T](self.keep, e.counters, N_RUNS_USED, self.dense, cnt, N_BLOBS)
        centroid_kernel[g](e.counters, self.keep, self.dense, self.area, self.sx, self.sy2,
                           self.cx, self.cy, self.bmatch, self.newflag)
        ev[1].record()
        # tracks
        c0, c1 = self.cur, self.nxt
        frame_init_kernel[g](cnt, self.cell_count, self.cell_fill, self.hit)
        predict_kernel[g](cnt, c0.px, c0.py, c0.vx, c0.vy, self.qx, self.qy, self.tcell, self.cell_count,
                          self.gw, self.gh, self.cell)
        scan_kernel[1, SCAN_T](self.cell_count, cnt, N_CELLS, self.cell_start, cnt, N_TRACKED)
        fill_kernel[g](cnt, self.tcell, self.cell_start, self.cell_fill, self.cell_tracks)
        pairs_kernel[g](cnt, self.cx, self.cy, self.qx, self.qy, self.cell_start, self.cell_count,
                        self.cell_tracks, self.gw, self.gh, self.cell, self.gate2,
                        self.cb, self.ct, self.cd2, self.calive)
        match_kernel[1, MATCH_T](cnt, self.cb, self.ct, self.cbits, self.calive, self.bmatch, self.hit,
                                self.blob_best, self.track_best)
        blob_ids_kernel[g](cnt, self.bmatch, c0.tid, self.bid, self.newflag)
        update_kernel[g](cnt, self.hit, self.cx, self.cy, c0.px, c0.py, c0.vx, c0.vy, c0.miss,
                         self.alive, self.miss_limit)
        scan_kernel[1, SCAN_T](self.alive, cnt, M, self.trank, cnt, N_ALIVE)
        scan_kernel[1, SCAN_T](self.newflag, cnt, N_BLOBS, self.brank, cnt, N_NEW)
        compact_kernel[g](cnt, self.alive, self.trank, *c0.arrays(), *c1.arrays())
        new_tracks_kernel[g](cnt, self.newflag, self.brank, self.cx, self.cy, self.bid, *c1.arrays())
        finish_kernel[1, 1](cnt)
        ev[2].record()
        # paint
        clear_kernel[g](self.out)
        paint_kernel[g](e.counters, e.run_x, e.run_y0, e.run_y1, self.rroot, self.keep, self.dense,
                        self.bid, self.palette, self.out)
        ev[3].record()
        painted = self.out.copy_to_host()
        c = cnt.copy_to_host()
        if c[N_CAND] > self.cand_cap:
            raise RuntimeError(f"{c[N_CAND]} candidate pairs, capacity {self.cand_cap}")
        if c[M] > self.track_cap:
            raise RuntimeError(f"{c[M]} tracks, capacity {self.track_cap}")
        self.cur, self.nxt = c1, c0
        self.n_blobs, self.next_id = int(c[N_BLOBS]), int(c[NEXT_ID])
        if timing is not None:
            timing["prep_gpu"] = cuda.event_elapsed_time(ev[0], ev[1])
            timing["track_gpu"] = cuda.event_elapsed_time(ev[1], ev[2])
            timing["paint_gpu"] = cuda.event_elapsed_time(ev[2], ev[3])
        return painted

    def host_view(self) -> tuple:
        """(centroids (n, 2) as (x, y), track id per blob) of the last frame."""
        n = self.n_blobs
        c = np.stack([self.cx[:n].copy_to_host(), self.cy[:n].copy_to_host()], axis=1)
        return c, self.bid[:n].copy_to_host()
