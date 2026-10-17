"""Act 3 of the chain cut (v5): ch08's floors, read from the kernel.

The chapter's default engine (TileWaves: dense tiles, dindex lookup,
lattice 8, valve 16, 8-conn), its instrumented twin, on two images:

1. A 384 x 384 crop of input_blobs.png at x = 4352, y = 5632 (the
   repo's [x, y]: x is the PNG column). The crop is run as an image of
   its own. Why this one: its origin is a multiple of 256 px, so its
   12 x 12 tiles and its seed lattice ((tx + ty) % 8 == 0) are the full
   frame's; it holds 8 blobs, 6 of which cross tile borders with two or
   more seeds (waves that meet), 36 seeds of which 11 come from the
   lattice, at most 4 pieces per tile, and 6 rounds like the full frame.
   Picked by eye among the 1,249 of the 2,312 aligned 384 and 512 px
   crops that pass a CPU-model survey (rounds >= 4, >= 2 blobs with
   two seeds, <= 4 pieces per tile, <= 24 blobs).
2. The full frame (9000 x 9000, 2,522 blobs): the counters, the
   frontier per round, and per-tile and 900 x 900 summaries of the
   round each piece was claimed in.

WHAT IS READ FROM THE KERNEL (no twin needed: the instrumented engine
keeps it all in its buffers after a run)
    records   rec[64 t + b]: anchor, side words, direction bits and,
              after the flatten passes, the canonical label of ring
              piece b of tile t
    header    hdr[t]: ring pieces (bits 0-5) and all pieces (bits 8+)
    queue     qc / ql: the seeds K2 queued, then the pieces claimed in
              round r at qc[bounds[r-1]:bounds[r]], each with the seed
              id of the wave that claimed it (ql)
    collisions ca / cb: the seed ids of two waves that met, recorded
              once per meeting edge (from its lex-smaller side)
    paint     the painted RGB, and the label map and the canonical
              seeds from the engine's off-clock twins

WHAT THE CPU ADDS
    The per-pixel piece id (a tile's pieces numbered as K2 numbers them,
    lowest remaining pixel first) comes from the chapter's CPU model
    (cpu_model.tile_model). The GPU keeps no per-pixel piece map; the
    model's numbering is checked against every GPU record instead.

Checks (the script stops on any failure), on both images:
- header counts per tile == the model's; every ring record's anchor,
  four side words and direction bits == the model's piece
- seeds (queued + lone) == the model's seed rule; the counters agree
- every ring piece is a seed or claimed exactly once; the round it was
  claimed in == its BFS distance to the nearest seed over the model's
  piece graph (multi-source BFS), so rounds are not timing-dependent
- the label map == the @njit oracle (cpu_label), pixel for pixel; every
  record's label == the oracle's at its anchor; emit_seeds == the
  oracle's canonical seeds; the painted RGB == palette[label % 6] on
  red and unchanged elsewhere
- the collision list holds every piece-graph edge whose two pieces were
  claimed by different waves (a meeting edge), as (tester wave, target
  wave), recorded once; or twice where the target lies in a tile with
  more than dense_min ring pieces and its ordinal is below dense_min.
  Those targets are tested twice: by the warp-wide dense path, and by
  the scan steps of the k-loop, whose test is not gated by `dense`
  (harmless for the labels; it inflates the collision, pred-skip and
  CAS counters). A CPU union-find over the list, in recorded order,
  retires as many roots as the kernel's union_done and gives every
  piece the record's label
Which wave claims a piece is a race, so owners and collisions are what
this run recorded; seeds, rounds, claims, links and labels are not.

Index order is the repo's [x, y] everywhere (x = PNG column, y = PNG
row): array.T is the PNG orientation.

Saved to video/assets/tiles/ (see its README.md):
- ch08_waves_<UTC>.npz   the arrays
- ch08_waves_<UTC>.json  the facts, the checks and the provenance

Memory: 3.7 GB host at peak, measured (the image, the model, the
oracle's and the GPU's label maps); about 1.2 GB device. About 20 s.

Run:  uv run python -m flood_fill_cuda.chapters.ch08_gpu_nblob_tiles.benchmarks.video_waves --images images/input
      (--quick: the crop only, every check, no files written)
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import gc
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from numba import cuda

from .. import flood_fill as ff
from .. import kernels as k
from ..cpu_model import tile_model
from ..cpu_oracle import cpu_label, red_mask
from ...ch05_gpu_nblob_nblock import scenes as S5

CROP_X, CROP_Y, CROP_SIDE = 4352, 5632, 384
LATTICE = ff.DEFAULT_LATTICE
VALVE = ff.DEFAULT_VALVE_ROUND
CONN = 8
PNG_MD5 = "47639eb37f973b5092ec2c4db86f45a4"   # = the benchmark's pin
SESSIONS = ("dense_20261016T054922Z", "waves_20261015T183401Z")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    *[os.pardir] * 5))
OUT_DIR = os.path.join(REPO, "video", "assets", "tiles")
SAMPLE = 10                       # full frame: 900 x 900 point samples


def check(ok, what, checks):
    checks[what] = bool(ok)
    if not ok:
        raise AssertionError(f"check failed: {what}")


def bfs_dist(tm, seed):
    """Multi-source BFS distance (in piece hops) from the seeds over the
    model's piece graph; -1 where no seed reaches (interior pieces)."""
    dist = np.full(tm.n_comps, -1, np.int32)
    cur = np.flatnonzero(seed)
    dist[cur] = 0
    d = 0
    while cur.size:
        d += 1
        nxt = []
        for u in cur:
            for v in tm.adj[tm.off[u]:tm.off[u + 1]]:
                if dist[v] < 0:
                    dist[v] = d
                    nxt.append(v)
        cur = np.array(nxt, np.int64)
    return dist


def run_engine(img):
    w, h = img.shape[:2]
    eng = ff.TileWaves(w, h, connectivity=CONN, instrumented=True)
    d_img = cuda.to_device(img)
    eng.run(d_img, "rgb")
    cuda.synchronize()
    out = {"counters": eng.read_counters(), "round_stats": eng.round_stats()}
    lab_dev = cuda.device_array((w, h), np.int32)
    eng.emit_label_map(lab_dev)
    cuda.synchronize()
    out["label"] = lab_dev.copy_to_host()
    del lab_dev
    out["seeds_canonical"] = eng.emit_seeds()
    out["painted"] = d_img.copy_to_host()
    out["rec"] = eng.rec.copy_to_host().reshape(-1, k.REC)
    out["hdr"] = eng.hdr.copy_to_host()
    rounds = out["counters"]["rounds"]
    bounds = eng.bounds.copy_to_host()
    if rounds >= bounds.size:
        raise RuntimeError("more rounds than the instrumented bounds keep")
    out["bounds"] = bounds[:rounds + 1].astype(np.int64)
    rear = int(out["bounds"][-1])
    out["qc"] = eng.qc[:max(rear, 1)].copy_to_host()[:rear]
    out["ql"] = eng.ql[:max(rear, 1)].copy_to_host()[:rear]
    n_coll = int(eng.read_q_state()[k.Q_COLL])
    out["n_coll"] = n_coll
    out["ca"] = eng.ca[:max(n_coll, 1)].copy_to_host()[:n_coll]
    out["cb"] = eng.cb[:max(n_coll, 1)].copy_to_host()[:n_coll]
    out["engine"] = {"dense": eng.dense, "lookup": eng.lookup,
                     "lattice": eng.lattice, "valve_round": eng.valve_round,
                     "waves_blocks": eng.waves_blocks,
                     "waves_tpb": eng.waves_tpb,
                     "dense_min": eng.dense_min,
                     "coll_capacity": eng.coll_capacity}
    del eng, d_img
    gc.collect()
    return out


def analyse(img, g, tag, checks):
    """Every check of the module docstring; returns per-piece arrays (the
    model's order) and facts."""
    w, h = img.shape[:2]
    c = g["counters"]
    tm = tile_model(img, CONN)
    wpr = tm.words_per_row
    n_tiles = ((w + 31) // 32) * wpr
    ring = tm.ordinal >= 0
    cid = np.where(ring, 64 * tm.tile.astype(np.int64) + tm.ordinal, -1)
    # ---- K2: headers and records
    hdr = g["hdr"][:n_tiles]
    check(np.array_equal(hdr & k.HDR_RING_MASK, tm.n_ring_tile)
          and np.array_equal(hdr >> k.HDR_TOTAL_SHIFT, tm.n_total_tile),
          f"{tag}: ring and total pieces per tile == CPU model", checks)
    rec = g["rec"]
    rr = rec[cid[ring]]
    sides = rr[:, :4].view(np.uint32)
    check(np.array_equal(rr[:, k.R_ANCHOR], tm.anchor[ring])
          and np.array_equal(sides[:, 0], tm.side_n[ring])
          and np.array_equal(sides[:, 1], tm.side_s[ring])
          and np.array_equal(sides[:, 2], tm.side_w[ring])
          and np.array_equal(sides[:, 3], tm.side_e[ring])
          and np.array_equal(rr[:, k.R_DIRS], tm.dirs[ring]),
          f"{tag}: every ring record (anchor, sides, dirs) == CPU model",
          checks)
    check(c["comps"] == tm.n_comps and c["ring_comps"] == int(ring.sum())
          and c["interior_comps"] == tm.n_interior,
          f"{tag}: piece counters == CPU model", checks)
    # ---- seeds
    seed, lat_only = tm.seeds(LATTICE)
    idx_of = {int(v): i for i, v in enumerate(cid) if v >= 0}
    b = g["bounds"]
    qc, ql = g["qc"], g["ql"]
    queued = qc[:b[0]]
    check(np.array_equal(qc[:b[0]], ql[:b[0]]),
          f"{tag}: a queued seed carries its own id", checks)
    gpu_seed = np.zeros(tm.n_comps, bool)
    gpu_seed[[idx_of[int(v)] for v in queued]] = True
    gpu_seed |= ring & (tm.dirs == 0)            # lone seeds, never queued
    check(np.array_equal(gpu_seed, seed)
          and c["seeds"] == int(seed.sum())
          and c["seeds_lattice"] == int(lat_only.sum())
          and c["seeds_queued"] == int(b[0]),
          f"{tag}: seeds == the seed rule (lattice {LATTICE}) and counters",
          checks)
    # ---- claims: round and wave of every ring piece
    rnd = np.full(tm.n_comps, -1, np.int16)
    wave = np.full(tm.n_comps, -1, np.int64)
    rnd[seed] = 0
    wave[seed] = cid[seed]
    n_claim = 0
    for r in range(1, len(b)):
        for p in range(int(b[r - 1]), int(b[r])):
            i = idx_of[int(qc[p])]
            if rnd[i] >= 0:
                raise AssertionError(f"{tag}: piece {i} claimed twice")
            rnd[i] = r
            wave[i] = int(ql[p])
            n_claim += 1
    check(c["valve_seeds"] == 0 and n_claim == c["claims"]
          and bool((rnd[ring] >= 0).all()),
          f"{tag}: every ring piece seeded or claimed once, no valve",
          checks)
    dist = bfs_dist(tm, seed)
    check(np.array_equal(rnd[ring], dist[ring]),
          f"{tag}: claim round == BFS distance to the nearest seed", checks)
    w_model = tm.waves(LATTICE, VALVE)
    check(w_model["rounds"] == c["rounds"]
          and w_model["frontier"] == g["round_stats"]["frontier_comps"],
          f"{tag}: rounds and frontier sizes == CPU model", checks)
    # ---- labels
    lab, n_blobs = cpu_label(img, CONN)
    check(np.array_equal(g["label"], lab) and c["n_blobs"] == n_blobs,
          f"{tag}: label map == @njit oracle, pixel for pixel", checks)
    flat_lab = lab.reshape(-1)
    check(np.array_equal(rr[:, k.R_LABEL], flat_lab[tm.anchor[ring]]),
          f"{tag}: every record's label == oracle at its anchor", checks)
    canon = np.unique(lab[lab >= 0])
    check([tuple(s) for s in g["seeds_canonical"]]
          == [(int(v) // h, int(v) % h) for v in canon],
          f"{tag}: emit_seeds == the oracle's canonical seeds", checks)
    red = red_mask(img)
    pal = k.PALETTE_HOST[(lab % k.N_PALETTE)[red]]
    check(np.array_equal(g["painted"][red], pal)
          and np.array_equal(g["painted"][~red], img[~red]),
          f"{tag}: painted RGB == palette[label % 6] on red only", checks)
    # ---- collisions and unions
    # meeting edges: two pieces claimed by different waves. The kernel
    # records one from the lex-smaller tile's piece (the tester), in the
    # round after that piece was claimed.
    ea, eb = tm.edge_a, tm.edge_b
    diff = wave[ea] != wave[eb]
    mu, mv = ea[diff], eb[diff]
    swap = tm.tile[mu] > tm.tile[mv]
    tester = np.where(swap, mv, mu)
    other = np.where(swap, mu, mv)
    meet_round = rnd[tester] + 1
    # A target in a tile with more than dense_min ring pieces is tested by
    # the warp-wide path, and (kernel quirk: the k-loop's scan branch is
    # not gated by `dense`) also by its lane's scan steps while the warp
    # runs other lanes' sparse tasks, so pieces of ordinal < dense_min
    # there can be recorded twice. Exactly once elsewhere.
    dense_min = g["engine"]["dense_min"]
    twice_ok = ((tm.n_ring_tile[tm.tile[other]] > dense_min)
                & (tm.ordinal[other] < dense_min))
    key = list(zip(wave[tester].tolist(), wave[other].tolist()))
    lo = Counter(key)
    hi = Counter(kk for kk, t2 in zip(key, twice_ok) if t2)
    rec_pairs = Counter(zip(g["ca"].tolist(), g["cb"].tolist()))
    check(c["coll_inline"] == 0 and g["n_coll"] == c["collisions"]
          and set(rec_pairs) == set(lo)
          and all(lo[p] <= n <= lo[p] + hi[p] for p, n in rec_pairs.items()),
          f"{tag}: collisions == meeting edges (tester wave, target wave), "
          f"repeats only on dense targets", checks)
    n_repeat = g["n_coll"] - int(diff.sum())
    parent = {int(v): int(v) for v in cid[seed]}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    merged = np.zeros(g["n_coll"], bool)
    for j, (a, bb) in enumerate(zip(g["ca"], g["cb"])):
        ra, rb = find(int(a)), find(int(bb))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            merged[j] = True
    check(int(merged.sum()) == c["union_done"]
          == c["seeds"] - c["roots"],
          f"{tag}: CPU union replay links == union_done == seeds - roots",
          checks)
    cls_min = {}
    for i in np.flatnonzero(ring):
        r0 = find(int(wave[i]))
        cls_min[r0] = min(cls_min.get(r0, 1 << 62), int(tm.anchor[i]))
    check(all(cls_min[find(int(wave[i]))] == int(rec[cid[i], k.R_LABEL])
              for i in np.flatnonzero(ring)),
          f"{tag}: union replay's min anchor == every record's label",
          checks)
    per = {
        "tile": tm.tile.astype(np.int32),
        "anchor": tm.anchor.astype(np.int64),
        "ordinal": tm.ordinal.astype(np.int8),
        "rec": cid.astype(np.int32),
        "size": tm.size.astype(np.int32),
        "dirs": tm.dirs.astype(np.uint8),
        "seed": seed, "seed_lattice_only": lat_only,
        "round": rnd, "wave": np.array([idx_of[int(v)] if v >= 0 else -1
                                        for v in wave], np.int32),
        "label": np.where(ring, rec[np.maximum(cid, 0), k.R_LABEL],
                          tm.anchor).astype(np.int64),
        "coll_a": np.array([idx_of[int(v)] for v in g["ca"]], np.int32),
        "coll_b": np.array([idx_of[int(v)] for v in g["cb"]], np.int32),
        "coll_merged": merged,
        "meet_tester": tester.astype(np.int32),
        "meet_other": other.astype(np.int32),
        "meet_round": meet_round.astype(np.int16),
    }
    facts = {
        "width": w, "height": h, "tiles": [(w + 31) // 32, wpr],
        "blobs": n_blobs, "red_px": int(red.sum()),
        "pieces": tm.n_comps, "ring_pieces": int(ring.sum()),
        "interior_pieces": tm.n_interior,
        "max_pieces_per_tile": int(tm.n_total_tile.max()),
        "tiles_with_red": int((tm.n_total_tile > 0).sum()),
        "seeds": c["seeds"], "seeds_lattice_only": c["seeds_lattice"],
        "seeds_queued": c["seeds_queued"],
        "seeds_lone": int((seed & (tm.dirs == 0)).sum()),
        "rounds": c["rounds"], "claims": c["claims"],
        "claimed_per_round": [int((rnd == r).sum())
                              for r in range(1, c["rounds"] + 1)],
        "frontier_pieces_per_round": g["round_stats"]["frontier_comps"],
        "frontier_tiles_per_round": g["round_stats"]["frontier_tiles"],
        "piece_edges": int(ea.size), "meeting_edges": int(diff.sum()),
        "collisions": c["collisions"],
        "collisions_repeated": n_repeat,
        "links": c["union_done"],
        "roots": c["roots"],
        "blobs_with_two_or_more_seeds": int(sum(
            1 for v in Counter(flat_lab[tm.anchor[seed]]).values()
            if v >= 2)),
        "counters": c, "engine": g["engine"],
    }
    return tm, lab, per, facts


def crop_export(full, checks):
    crop = np.ascontiguousarray(full[CROP_X:CROP_X + CROP_SIDE,
                                     CROP_Y:CROP_Y + CROP_SIDE])
    g = run_engine(crop)
    tm, lab, per, facts = analyse(crop, g, "crop", checks)
    first = np.concatenate([[0], np.cumsum(tm.n_total_tile)[:-1]])
    loc = tm.loc
    on = loc >= 0
    piece = np.full(loc.shape, -1, np.int8)
    piece[on] = (loc[on] - first[tm.tile[loc[on]]]).astype(np.int8)
    local = np.arange(tm.n_comps) - first[tm.tile]
    check(int(piece.max()) < int(tm.n_total_tile.max()),
          "crop: piece ids < pieces per tile", checks)
    tiles_x, wpr = facts["tiles"]
    tx, ty = np.meshgrid(np.arange(tiles_x), np.arange(wpr), indexing="ij")
    arrays = {
        "crop_red": red_mask(crop),
        "crop_piece": piece,
        "crop_comp": loc.astype(np.int16),
        "crop_label": lab.astype(np.int32),
        "crop_tile_lattice": ((tx + ty) % LATTICE) == 0,
        "crop_tile_pieces": tm.n_total_tile.reshape(tiles_x, wpr)
        .astype(np.int8),
        "crop_comp_local": local.astype(np.int8),
        **{f"crop_comp_{n}": v for n, v in per.items()
           if not n.startswith(("coll", "meet"))},
        **{f"crop_{n}": v for n, v in per.items()
           if n.startswith(("coll", "meet"))},
    }
    facts.update({"origin": [CROP_X, CROP_Y], "side": CROP_SIDE,
                  "lattice_tiles": int(arrays["crop_tile_lattice"].sum())})
    print(f"  crop: {facts['blobs']} blobs, {facts['pieces']} pieces, "
          f"{facts['seeds']} seeds ({facts['seeds_lattice_only']} lattice), "
          f"{facts['rounds']} rounds, claimed {facts['claimed_per_round']}, "
          f"{facts['collisions']} collisions, {facts['links']} links",
          flush=True)
    return arrays, facts


def full_export(full, checks):
    g = run_engine(full)
    tm, lab, per, facts = analyse(full, g, "full frame", checks)
    del lab
    tiles_x, wpr = facts["tiles"]
    n_tiles = tiles_x * wpr
    ring = per["ordinal"] >= 0
    rmin = np.full(n_tiles, 127, np.int16)
    rmax = np.full(n_tiles, -1, np.int16)
    np.minimum.at(rmin, per["tile"][ring], per["round"][ring])
    np.maximum.at(rmax, per["tile"][ring], per["round"][ring])
    rmin[rmin == 127] = -1
    nseed = np.bincount(per["tile"][per["seed"]], minlength=n_tiles)
    xs = np.arange(SAMPLE // 2, full.shape[0], SAMPLE)
    ys = np.arange(SAMPLE // 2, full.shape[1], SAMPLE)
    loc = tm.loc[np.ix_(xs, ys)]
    rs = np.full(loc.shape, -1, np.int8)              # off-blob
    on = loc >= 0
    rs[on] = np.where(per["ordinal"][loc[on]] >= 0,
                      per["round"][loc[on]], -2)      # -2: interior piece
    arrays = {
        "full_tile_pieces": tm.n_total_tile.reshape(tiles_x, wpr)
        .astype(np.int8),
        "full_tile_round_min": rmin.reshape(tiles_x, wpr).astype(np.int8),
        "full_tile_round_max": rmax.reshape(tiles_x, wpr).astype(np.int8),
        "full_tile_seeds": nseed.reshape(tiles_x, wpr).astype(np.int8),
        "full_round_900": rs,
    }
    facts.update({"sample_stride": SAMPLE, "sample_offset": SAMPLE // 2})
    print(f"  full frame: {facts['blobs']:,} blobs, {facts['pieces']:,} "
          f"pieces, {facts['seeds']:,} seeds, {facts['rounds']} rounds, "
          f"frontier {facts['frontier_pieces_per_round']}, "
          f"{facts['collisions']:,} collisions, {facts['links']:,} links",
          flush=True)
    del tm, g
    gc.collect()
    return arrays, facts


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", default=os.environ.get("FLOOD_FILL_IMAGES"),
                    help="folder holding input_blobs.png")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    if not args.images:
        raise SystemExit("give --images DIR (input_blobs.png)")
    png = os.path.join(args.images, "input_blobs.png")
    md5 = hashlib.md5(open(png, "rb").read()).hexdigest()
    if md5 != PNG_MD5:
        raise SystemExit(f"{png}: md5 {md5}, expected {PNG_MD5}")
    full = S5.png_scene(png)[0]
    checks = {}
    print("crop:", flush=True)
    crop_arrays, crop_facts = crop_export(full, checks)
    full_arrays, full_facts = {}, None
    if not args.quick:
        print("full frame:", flush=True)
        full_arrays, full_facts = full_export(full, checks)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dev = cuda.get_current_device().name
    meta = {
        "what": "ch08 tile waves, act 3 floors of the chain cut (v5)",
        "script": "src/flood_fill_cuda/chapters/ch08_gpu_nblob_tiles/"
                  "benchmarks/video_waves.py",
        "commit": git_commit(), "utc": stamp,
        "device": (dev.decode() if isinstance(dev, bytes) else str(dev))
        .split("\x00")[0].strip(),
        "image": os.path.basename(png), "png_md5": PNG_MD5,
        "chapter_sessions": list(SESSIONS),
        "config": {"connectivity": CONN, "lattice": LATTICE,
                   "valve_round": VALVE, "instrumented": True,
                   "engine": "TileWaves defaults (dense, dindex)"},
        "index_order": "[x, y]: x = PNG column, y = PNG row; tile (tx, ty) "
                       "= x in [32 tx, 32 tx + 32), y in [32 ty, 32 ty + "
                       "32); t = tx * tiles_y + ty; lattice tiles: "
                       "(tx + ty) % 8 == 0",
        "crop": crop_facts, "full": full_facts, "checks": checks,
    }
    print(f"{sum(checks.values())} of {len(checks)} checks hold", flush=True)
    if args.quick:
        print("--quick: nothing written")
        return
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"ch08_waves_{stamp}")
    np.savez_compressed(base + ".npz", **crop_arrays, **full_arrays)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=int)
        fh.write("\n")
    print(f"wrote {os.path.basename(base)}.npz and .json")


if __name__ == "__main__":
    main()
