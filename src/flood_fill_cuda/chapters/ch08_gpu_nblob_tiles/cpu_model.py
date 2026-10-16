"""CPU model of the seeded tile waves (ports the scout's local_ccl and
design 2's wave model).

Per tile it finds the comps in K2's order (lowest remaining pixel
first), splits them into ring and interior comps, numbers the ring comps
(ordinals), and computes each ring comp's side words, anchor and
direction bits. Over the comp graph it computes, for any seed rule
(lattice L) and valve round, the seeds, the multi-source BFS rounds,
the claims and the valve seeds - the numbers the GPU counters must equal
exactly (test invariants 11.2).

Everything is computed from pixels, a different way than the GPU does
it (no side-word algebra): two comps are adjacent iff two of their
pixels are adjacent across a tile border.

For the dense-tile path it also predicts K2's choice per tile (the run
estimate, from per-pixel neighbour tests rather than word arithmetic),
which tiles K2b indexes, and how many K3 tasks the dindex lookup serves
by the index or by the whole warp.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .cpu_oracle import red_mask

TILE = 32
PRED = 0x35


@njit(cache=True)
def _dir_code(dtx, dty):
    if dtx == -1:
        if dty == 0:
            return 0
        return 4 if dty == -1 else 5
    if dtx == 1:
        if dty == 0:
            return 1
        return 6 if dty == -1 else 7
    return 2 if dty == -1 else 3


@njit(cache=True)
def _local_comps(red, conn8):
    """Comps of every tile in K2's order. Returns loc (W, H) and per-comp
    arrays (tile, ordinal or -1, anchor, N, S, Wc, Ec, size) and per-tile
    (n_ring, n_total)."""
    width, height = red.shape
    tiles_x = (width + TILE - 1) // TILE
    wpr = (height + TILE - 1) // TILE
    n_tiles = tiles_x * wpr
    loc = np.full((width, height), -1, np.int32)
    n_red = 0
    for x in range(width):
        for y in range(height):
            if red[x, y]:
                n_red += 1
    cap = max(n_red, 1)
    c_tile = np.empty(cap, np.int32)
    c_ord = np.empty(cap, np.int32)
    c_anchor = np.empty(cap, np.int64)
    c_n = np.zeros(cap, np.uint32)
    c_s = np.zeros(cap, np.uint32)
    c_w = np.zeros(cap, np.uint32)
    c_e = np.zeros(cap, np.uint32)
    c_size = np.zeros(cap, np.int32)
    n_ring = np.zeros(n_tiles, np.int32)
    n_total = np.zeros(n_tiles, np.int32)
    qx = np.empty(TILE * TILE, np.int32)
    qy = np.empty(TILE * TILE, np.int32)
    one = np.uint32(1)
    nc = 0
    for tx in range(tiles_x):
        x0 = tx * TILE
        x1 = min(width, x0 + TILE)
        for ty in range(wpr):
            y0 = ty * TILE
            y1 = min(height, y0 + TILE)
            t = tx * wpr + ty
            b = 0
            k = 0
            for x in range(x0, x1):
                for y in range(y0, y1):
                    if not red[x, y] or loc[x, y] >= 0:
                        continue
                    loc[x, y] = nc
                    qx[0] = x
                    qy[0] = y
                    head = 0
                    tail = 1
                    ring = False
                    sn = np.uint32(0)
                    ss = np.uint32(0)
                    sw = np.uint32(0)
                    se = np.uint32(0)
                    while head < tail:
                        px = qx[head]
                        py = qy[head]
                        head += 1
                        lx = px - x0
                        ly = py - y0
                        if lx == 0:
                            sn |= one << np.uint32(ly)
                            ring = True
                        if lx == TILE - 1:
                            ss |= one << np.uint32(ly)
                            ring = True
                        if ly == 0:
                            sw |= one << np.uint32(lx)
                            ring = True
                        if ly == TILE - 1:
                            se |= one << np.uint32(lx)
                            ring = True
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                if not conn8 and dx != 0 and dy != 0:
                                    continue
                                nx = px + dx
                                ny = py + dy
                                if x0 <= nx < x1 and y0 <= ny < y1:
                                    if red[nx, ny] and loc[nx, ny] < 0:
                                        loc[nx, ny] = nc
                                        qx[tail] = nx
                                        qy[tail] = ny
                                        tail += 1
                    c_tile[nc] = t
                    c_anchor[nc] = x * height + y
                    c_size[nc] = tail
                    if ring:
                        c_ord[nc] = b
                        c_n[nc] = sn
                        c_s[nc] = ss
                        c_w[nc] = sw
                        c_e[nc] = se
                        b += 1
                    else:
                        c_ord[nc] = -1
                    nc += 1
                    k += 1
            n_ring[t] = b
            n_total[t] = k
    return (loc, nc, c_tile[:nc].copy(), c_ord[:nc].copy(),
            c_anchor[:nc].copy(), c_n[:nc].copy(), c_s[:nc].copy(),
            c_w[:nc].copy(), c_e[:nc].copy(), c_size[:nc].copy(), n_ring,
            n_total)


@njit(cache=True)
def _border_pass(red, loc, conn8, dirs, ea, eb, fill):
    """One pass over the cross-tile pixel pairs. fill=False: set the
    direction bits and count the pairs; fill=True: write each pair as
    (min comp, max comp). A pixel pair is seen once, from its lex-smaller
    pixel."""
    width, height = red.shape
    n_e = 0
    for x in range(width):
        tx = x // TILE
        for y in range(height):
            if not red[x, y]:
                continue
            ty = y // TILE
            c = loc[x, y]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    if not conn8 and dx != 0 and dy != 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if not red[nx, ny]:
                        continue
                    ntx = nx // TILE
                    nty = ny // TILE
                    if ntx == tx and nty == ty:
                        continue
                    if not fill:
                        dirs[c] |= 1 << _dir_code(ntx - tx, nty - ty)
                    if nx * height + ny > x * height + y:
                        if fill:
                            c2 = loc[nx, ny]
                            ea[n_e] = min(c, c2)
                            eb[n_e] = max(c, c2)
                        n_e += 1
    return n_e


def _border_edges(red, loc, nc, conn8):
    dirs = np.zeros(max(nc, 1), np.int32)
    dummy = np.zeros(1, np.int32)
    n_e = _border_pass(red, loc, conn8, dirs, dummy, dummy, False)
    ea = np.empty(max(n_e, 1), np.int32)
    eb = np.empty(max(n_e, 1), np.int32)
    _border_pass(red, loc, conn8, dirs, ea, eb, True)
    return dirs[:nc].copy(), ea[:n_e].copy(), eb[:n_e].copy()


@njit(cache=True)
def _waves(off, adj, ring, seed, dirs, valve_round):
    """Multi-source BFS exactly as K3 runs it: round r processes the
    frontier and claims the unclaimed neighbours; at r == valve_round
    (if the round claimed anything) every unclaimed ring comp self-seeds
    and joins the next frontier. Returns (rounds, claims, valve seeds,
    valve round, frontier sizes per round)."""
    n = seed.size
    claimed = seed.copy()
    cur = np.empty(max(n, 1), np.int64)
    nxt = np.empty(max(n, 1), np.int64)
    n_cur = 0
    for i in range(n):
        if seed[i] and dirs[i] != 0:
            cur[n_cur] = i
            n_cur += 1
    front = np.zeros(max(n, 1) + 2, np.int64)
    r = 1
    claims = 0
    valve_seeds = 0
    valve_at = 0
    while n_cur > 0:
        front[r - 1] = n_cur
        n_nxt = 0
        for j in range(n_cur):
            u = cur[j]
            for kk in range(off[u], off[u + 1]):
                v = adj[kk]
                if not claimed[v]:
                    claimed[v] = True
                    nxt[n_nxt] = v
                    n_nxt += 1
                    claims += 1
        if n_nxt > 0 and r == valve_round:
            valve_at = r
            for v in range(n):
                if ring[v] and not claimed[v]:
                    claimed[v] = True
                    valve_seeds += 1
                    if dirs[v] != 0:
                        nxt[n_nxt] = v
                        n_nxt += 1
        tmp = cur
        cur = nxt
        nxt = tmp
        n_cur = n_nxt
        r += 1
    return r - 1, claims, valve_seeds, valve_at, front[:r - 1].copy()


def tile_runs_estimate(red, connectivity=8):
    """Per tile (t = tx*wpr + ty): (estimate, runs) as K2's _dense_tile
    computes them, from pixels. A run is a maximal red stretch of one
    row (x) inside the tile's 32 columns (y); it is "untouched from
    above" if no pixel of it has a red neighbour in row x-1 of the same
    tile (8-conn: y-1, y, y+1; 4-conn: y), likewise from below. The
    estimate is the smaller of the two untouched counts."""
    red = np.asarray(red, bool)
    width, height = red.shape
    tiles_x = (width + TILE - 1) // TILE
    wpr = (height + TILE - 1) // TILE
    pad = np.zeros((tiles_x * TILE, wpr * TILE), bool)
    pad[:width, :height] = red
    t = pad.reshape(tiles_x, TILE, wpr, TILE).transpose(0, 2, 1, 3)
    # t[tx, ty, row, col]
    start = t.copy()
    start[..., 1:] &= ~t[..., :-1]
    runs = start.sum(axis=(2, 3))
    run_id = np.cumsum(start, axis=3, dtype=np.int8)   # run index in a row

    def contact(other):
        if connectivity == 8:
            nb = other.copy()
            nb[..., 1:] |= other[..., :-1]
            nb[..., :-1] |= other[..., 1:]
        else:
            nb = other
        return t & nb

    above = np.zeros_like(t)
    above[:, :, 1:, :] = t[:, :, :-1, :]
    below = np.zeros_like(t)
    below[:, :, :-1, :] = t[:, :, 1:, :]

    def untouched(con):
        # runs (per row) with no contact pixel: runs - distinct run ids hit
        hit = np.where(con, run_id, np.int8(0))
        touched = np.zeros(t.shape[:3], np.int16)
        for k in range(1, TILE // 2 + 1):
            touched += (hit == k).any(axis=3)
        return runs - touched.sum(axis=2, dtype=np.int64)

    est = np.minimum(untouched(contact(above)), untouched(contact(below)))
    return est.reshape(-1).astype(np.int64), runs.reshape(-1).astype(np.int64)


@dataclass
class TileModel:
    width: int
    height: int
    connectivity: int
    loc: np.ndarray          # (W, H) comp index, -1 off-blob
    n_comps: int
    tile: np.ndarray         # per comp: tile index tx*wpr + ty
    ordinal: np.ndarray      # per comp: ring ordinal, -1 for interior
    anchor: np.ndarray       # per comp: min lin
    side_n: np.ndarray
    side_s: np.ndarray
    side_w: np.ndarray
    side_e: np.ndarray
    size: np.ndarray
    dirs: np.ndarray         # per comp: direction bits
    n_ring_tile: np.ndarray  # per tile
    n_total_tile: np.ndarray
    edge_a: np.ndarray       # distinct comp pairs, a < b
    edge_b: np.ndarray
    off: np.ndarray          # CSR adjacency
    adj: np.ndarray
    run_est: tuple = None    # per tile (estimate, runs), tile_runs_estimate

    @property
    def words_per_row(self):
        return (self.height + TILE - 1) // TILE

    @property
    def ring(self):
        return self.ordinal >= 0

    @property
    def n_ring(self):
        return int(np.count_nonzero(self.ordinal >= 0))

    @property
    def n_interior(self):
        return self.n_comps - self.n_ring

    @property
    def tasks(self):
        """Sum of popc(dirs) over ring comps: every crossing ring comp is
        processed exactly once with its full dirs, whatever the seeds."""
        d = np.ascontiguousarray(
            self.dirs[self.ordinal >= 0].astype(np.uint32))
        return int(np.unpackbits(d.view(np.uint8)).sum())

    def dense_tasks(self, dense_min):
        """Tasks whose neighbour tile holds more than dense_min ring
        comps: the hybrid lookup's warp-cooperative tasks."""
        wpr = self.words_per_row
        ring = np.flatnonzero(self.ordinal >= 0)
        n = 0
        for d, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1),
                                      (-1, -1), (-1, 1), (1, -1), (1, 1))):
            has = ring[(self.dirs[ring] >> d) & 1 == 1]
            t2 = self.tile[has].astype(np.int64) + dx * wpr + dy
            n += int(np.count_nonzero(self.n_ring_tile[t2] > dense_min))
        return n

    def ccl_tiles(self, ccl_min, ccl_rpc):
        """Tiles K2 leaves to K2b's run CCL (bool per tile)."""
        est, runs = self.run_est
        return (est >= ccl_min) & (runs <= ccl_rpc * est) & (est > 0)

    def indexed_tiles(self, ccl_min, ccl_rpc):
        """Tiles K2b indexes for the dindex and index lookups."""
        return self.ccl_tiles(ccl_min, ccl_rpc) & (self.n_ring_tile >= 2)

    def dindex_dense_tasks(self, ccl_min, ccl_rpc, dense_min, coop=True):
        """dindex: tasks served by the index (indexed neighbour tile) or,
        with coop, by the whole warp (unindexed, more than dense_min ring
        comps)."""
        wpr = self.words_per_row
        ix = self.indexed_tiles(ccl_min, ccl_rpc)
        ring = np.flatnonzero(self.ordinal >= 0)
        n = 0
        for d, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1),
                                      (-1, -1), (-1, 1), (1, -1), (1, 1))):
            has = ring[(self.dirs[ring] >> d) & 1 == 1]
            t2 = self.tile[has].astype(np.int64) + dx * wpr + dy
            dense = ix[t2]
            if coop:
                dense = dense | (~ix[t2] & (self.n_ring_tile[t2] > dense_min))
            n += int(np.count_nonzero(dense))
        return n

    def seeds(self, lattice):
        wpr = self.words_per_row
        tx = self.tile // wpr
        ty = self.tile % wpr
        pred_free = (self.dirs & PRED) == 0
        lat = (np.zeros(self.n_comps, bool) if lattice <= 0
               else ((tx + ty) % lattice) == 0)
        ring = self.ring
        return ring & (pred_free | lat), ring & lat & ~pred_free

    def waves(self, lattice=8, valve_round=128):
        seed, lat_only = self.seeds(lattice)
        rounds, claims, vseeds, vat, front = _waves(
            self.off, self.adj, self.ring, seed, self.dirs, valve_round)
        return {"seeds": int(seed.sum()), "seeds_lattice": int(lat_only.sum()),
                "seeds_queued": int((seed & (self.dirs != 0)).sum()),
                "rounds": int(rounds), "claims": int(claims),
                "valve_seeds": int(vseeds), "valve_round": int(vat),
                "frontier": [int(v) for v in front]}


def tile_model(img_or_red, connectivity=8):
    """Build the per-tile comp model of an RGB image or a (W, H) bool."""
    red = (img_or_red if img_or_red.ndim == 2 else red_mask(img_or_red))
    red = np.ascontiguousarray(red.astype(np.bool_))
    conn8 = connectivity == 8
    (loc, nc, tile, ordinal, anchor, sn, ss, sw, se, size, n_ring,
     n_total) = _local_comps(red, conn8)
    dirs, ea, eb = _border_edges(red, loc, nc, conn8)
    if ea.size:
        code = np.unique(ea.astype(np.int64) * max(nc, 1) + eb)
        ea = (code // max(nc, 1)).astype(np.int64)
        eb = (code % max(nc, 1)).astype(np.int64)
    else:
        ea = np.zeros(0, np.int64)
        eb = np.zeros(0, np.int64)
    src = np.concatenate([ea, eb])
    dst = np.concatenate([eb, ea])
    order = np.argsort(src, kind="stable")
    adj = dst[order].astype(np.int64)
    off = np.zeros(nc + 1, np.int64)
    np.add.at(off, src + 1, 1)
    off = np.cumsum(off)
    return TileModel(red.shape[0], red.shape[1], connectivity, loc, int(nc),
                     tile, ordinal, anchor, sn, ss, sw, se, size, dirs,
                     n_ring, n_total, ea, eb, off, adj,
                     tile_runs_estimate(red, connectivity))
