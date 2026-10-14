"""Owner map: which block painted each pixel of the 64 Mpx square.

No new GPU code. Every instrumented kernel already writes owner[x, y] =
blockIdx.x when it dequeues a pixel; this script records that map on
sq_8000_center (8000 x 8000, full bleed, seed at the centre) for three
launches:

- ch02's global kernel, 2 x 256, 4-conn (owner int8);
- ch03's multi kernel, 48 x 256, 4-conn (owner int16);
- ch03's multi kernel, 48 x 256, 8-conn (owner int16).

What the map is and is not:

- Owner POSITIONS are race-dependent. The order inside a level is
  warp-aggregated discovery order, so two runs paint different speckle.
- Owner COUNTS are not. Entry j of a level goes to grid thread
  j mod (blocks * tpb), so block (j mod G) // tpb, and every block's total
  follows from level_sizes alone. The census is checked three ways:
  against the run's own per-block counters (test_owner_census's
  identity), against that rule applied to the run's level_sizes, and
  against the pinned CH03 session (multi_block_20260720T012243Z.json),
  whose level traces must match this run's exactly and whose
  processed_per_block must equal the 48 x 256 4-conn census.
- Every fill is checked pixel-exact against the @njit oracle, with the
  checks of test_correctness.assert_matches_reference (visited, depth,
  levels, filled, recolour; ch02 and ch03 share them), and the level
  trace must be untruncated.

Saved to results/ch03_gpu_1blob_nblock/owner_map/:

- owner_map_<UTC>.json: per launch the kernel time (one instrumented
  run, not a median), launch shape, sm ids, per-block pixel counts, the
  cross-checks, the crop origins and a mixing summary.
- owner_map_<UTC>.npz: per launch ("<launch>.<array>") the owner and the
  BFS depth at point samples on the video floor's grid, three crops at
  full resolution (64 x 64 at the seed, 64 x 64 on the widest ring,
  192 x 192 at the seed: block 0's whole centre diamond), per-block pixel
  counts, sm ids and kernel time; plus the device name and the sample
  coordinates.

The floor grid: the video's toy square (video/scenes/chain/toy.py,
square()) is 21 x 21 cells at rows/cols 1-21 of a 24 x 24 grid, seeded
at its centre cell (11, 11). Each cell becomes a 6 x 6 mosaic of real
dots: 126 x 126 samples over the 8000 px square, stride 8000 / 126 =
63.5 px, each at its sub-cell's centre pixel. Index order is the repo's
[x, y], which is the toy's [row, col] (toy.py's convention), so dot
(a, b) of toy cell (r, c) is samples[6 * (r - 1) + a, 6 * (c - 1) + b].

Memory at 8000^2: host 3.1 GB peak RSS, measured (the image, one
oracle's visited + depth and one GPU result alive together); device
about 1.1 GB (image, visited, depth, queue, owner). The whole run takes
about 30 s.

Run:  uv run python -m flood_fill_cuda.chapters.ch03_gpu_1blob_nblock.benchmarks.owner_map
      (--quick: a 1024^2 square, every check except the pinned session's,
      no files written)
"""

import os

os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

import argparse
import gc
import json
from datetime import datetime, timezone

import numpy as np

from ..flood_fill import flood_fill
from ..cpu_oracle import cpu_flood_fill, cpu_flood_fill_8
from ...ch02_gpu_1blob_2block.flood_fill import flood_fill as dual_flood_fill
from .. import scenes
from ....shared import results_paths
from numba import cuda

OUT_DIR = os.path.join(results_paths.RESULTS_ROOT, "ch03_gpu_1blob_nblock",
                       "owner_map")
PINNED = os.path.join(results_paths.RESULTS_ROOT, "ch03_gpu_1blob_nblock",
                      "benchmark_results", "multi_block_20260720T012243Z.json")

SCENE = ("sq_8000_center",
         lambda: scenes.square_scene(8000, 8000, 8000, 8000))
QUICK_SCENE = ("sq_1024_full_center",
               lambda: scenes.square_scene(1024, 1024, 1024, 1024))

# (key, chapter, connectivity, blocks, tpb)
LAUNCHES = [
    ("ch02_global_2x256", "ch02", 4, 2, 256),
    ("ch03_multi_48x256_c4", "ch03", 4, 48, 256),
    ("ch03_multi_48x256_c8", "ch03", 8, 48, 256),
]

# The video floor's grid (toy.square(): 21 cells of a 24-cell grid, one
# empty cell of margin), each cell a DOTS x DOTS mosaic of real samples
TOY_N, TOY_SIDE, TOY_OFFSET = 24, 21, 1
DOTS = 6
CROP = 64
CROP_WIDE = 192

BLUE = np.array([0, 0, 255], dtype=np.uint8)
CHUNK = 1 << 22  # pixels per bincount chunk (keeps the int64 cast at 32 MB)


# ---------------------------------------------------------------- checks

def _bincount_shifted(a, length):
    """bincount(a + 1) over a big array in chunks: slot 0 counts the -1s
    (unreached), slot k + 1 counts value k. A whole-array bincount would
    cast 64M entries to int64 at once (512 MB)."""
    flat = a.reshape(-1)
    counts = np.zeros(length + 1, dtype=np.int64)
    for i in range(0, flat.size, CHUNK):
        c = np.bincount(flat[i:i + CHUNK].astype(np.int64) + 1,
                        minlength=length + 1)
        assert c.size == length + 1, "value out of range"
        counts += c
    return counts


def check_pixel_exact(r, img, ref):
    """test_correctness.assert_matches_reference on a result in hand, so
    one oracle run serves both 4-conn launches: visited, depth, levels and
    filled equal the @njit oracle's, every filled pixel is blue and every
    other pixel is untouched."""
    ref_visited, ref_depth, ref_levels, ref_filled = ref
    np.testing.assert_array_equal(r.visited, ref_visited)
    np.testing.assert_array_equal(r.depth, ref_depth)
    assert r.levels == ref_levels, (r.levels, ref_levels)
    assert r.filled == ref_filled, (r.filled, ref_filled)
    filled = r.visited == 1
    assert (r.img[filled] == BLUE).all()
    np.testing.assert_array_equal(r.img[~filled], img[~filled])


def check_level_trace(r):
    """test_frontier_trace_consistency's identities: the trace is whole,
    accounts for every pixel and equals the depth map's ring sizes."""
    sizes = np.asarray(r.level_sizes, dtype=np.int64)
    assert not r.level_trace_truncated, "level trace truncated"
    assert sizes.size == r.levels, (sizes.size, r.levels)
    assert sizes[0] == 1 and sizes.sum() == r.filled
    rings = _bincount_shifted(r.depth, r.levels)
    np.testing.assert_array_equal(rings[1:], sizes)
    return sizes


def owner_census(r, blocks, per_block):
    """test_owner_census: owner in [0, blocks) wherever reached, -1
    elsewhere, and the per-pixel census equals the per-block counters."""
    reached = r.visited == 1
    assert np.array_equal(r.owner >= 0, reached), "owner set != visited set"
    assert int(r.owner.max()) < blocks
    census = _bincount_shifted(r.owner, blocks)[1:]
    np.testing.assert_array_equal(census, per_block)
    return census


def grid_stride_counts(level_sizes, blocks, tpb):
    """Per-block dequeues the grid-stride loop implies: level entry j goes
    to grid thread j mod G (G = blocks * tpb), i.e. to block
    (j mod G) // tpb. Full sweeps of the grid give every block tpb
    entries; the remainder fills blocks 0, 1, ... in order."""
    g = blocks * tpb
    n = np.asarray(level_sizes, dtype=np.int64)
    full, rem = n // g, n % g
    b = np.arange(blocks, dtype=np.int64)
    part = np.clip(rem[:, None] - b[None, :] * tpb, 0, tpb)
    return int(full.sum()) * tpb + part.sum(axis=0)


def block0_radius(r):
    """The largest ring R with every pixel of depth <= R owned by block 0:
    block 0's centre diamond (4-conn) or square (8-conn)."""
    others = np.where(r.owner > 0, r.depth, np.int32(r.levels)).min()
    return int(others) - 1


# ------------------------------------------------------- samples and crops

def floor_samples(width, height):
    """The floor's dot coordinates: TOY_SIDE * DOTS points per axis, each
    at the centre pixel of its 1 / (TOY_SIDE * DOTS) slice of the blob
    (the blob is the whole image here)."""
    n = TOY_SIDE * DOTS
    xs = np.floor((np.arange(n) + 0.5) * width / n).astype(np.int32)
    ys = np.floor((np.arange(n) + 0.5) * height / n).astype(np.int32)
    return xs, ys


def _origin(cx, cy, size, width, height):
    """Top-left of a size x size window centred on (cx, cy), clamped."""
    x0 = min(max(cx - size // 2, 0), width - size)
    y0 = min(max(cy - size // 2, 0), height - size)
    return int(x0), int(y0)


def ring_center(level, sx, sy, connectivity):
    """A pixel of ring `level` in the seed's +x/+y quadrant, at half the
    ring's radius along y. Its depth is `level` under either metric. At
    8-conn the widest ring is the square's outer frame, so its crop
    touches the image edge."""
    if connectivity == 4:
        return sx + level - level // 2, sy + level // 2
    return sx + level, sy + level // 2


def _crop(a, x0, y0, size):
    return a[x0:x0 + size, y0:y0 + size].astype(np.int16)


def _differ_pct(a):
    """% of 4-adjacent pairs (both reached) whose owners differ."""
    pairs = diff = 0
    for u, v in ((a[1:, :], a[:-1, :]), (a[:, 1:], a[:, :-1])):
        both = (u >= 0) & (v >= 0)
        pairs += int(both.sum())
        diff += int(((u != v) & both).sum())
    return 100.0 * diff / pairs if pairs else 0.0


def _shuffle_differ_pct(a):
    """The same % if a's owners were shuffled at random: 1 - sum p_b^2."""
    v = a[a >= 0].astype(np.int64)
    p = np.bincount(v) / v.size
    return 100.0 * (1.0 - float((p ** 2).sum()))


def mixing(samples, census, filled):
    """How mixed the floor's dots are, per toy cell and overall."""
    k = DOTS * DOTS
    cells = (samples.reshape(TOY_SIDE, DOTS, TOY_SIDE, DOTS)
             .swapaxes(1, 2).reshape(TOY_SIDE * TOY_SIDE, k))
    distinct, majority = [], []
    for c in cells:
        c = c[c >= 0].astype(np.int64)
        if c.size:
            distinct.append(len(np.unique(c)))
            majority.append(np.bincount(c).max() / c.size)
    reached = samples[samples >= 0].astype(np.int64)
    share = np.bincount(reached, minlength=census.size) / reached.size
    seed_cell = cells[(TOY_SIDE // 2) * TOY_SIDE + TOY_SIDE // 2]
    return {
        "distinct_owners_per_cell_mean": float(np.mean(distinct)),
        "distinct_owners_per_cell_min": int(np.min(distinct)),
        "distinct_owners_per_cell_max": int(np.max(distinct)),
        "majority_share_per_cell_mean": float(np.mean(majority)),
        "blocks_seen_in_samples": int((share > 0).sum()),
        # a fair draw: each block's share of the dots vs of the pixels
        "samples_vs_census_max_pp": float(
            100.0 * np.abs(share - census / filled).max()),
        "neighbour_differ_pct": _differ_pct(samples),
        "shuffled_differ_pct": _shuffle_differ_pct(samples),
        "seed_cell_block0_dots": int((seed_cell == 0).sum()),
    }


# ------------------------------------------------------------- one launch

def _release():
    gc.collect()
    try:  # hand freed device buffers back before the next 64 Mpx launch
        cuda.current_context().deallocations.clear()
    except AttributeError:
        pass


def _fill(chapter, img, sx, sy, connectivity, blocks, tpb):
    """One instrumented launch with the owner map on, in one shape for
    both chapters: (result, per-block dequeues, sm ids)."""
    if chapter == "ch02":
        r = dual_flood_fill(img, sx, sy, threads_per_block=tpb,
                            kernel="global")
        assert r.blocks == blocks and connectivity == 4
        per_block = np.array([r.processed_b0, r.processed_b1],
                             dtype=np.int64)
        return r, per_block, [r.sm_id_b0, r.sm_id_b1]
    r = flood_fill(img, sx, sy, threads_per_block=tpb, blocks=blocks,
                   connectivity=connectivity)
    assert r.blocks == blocks
    return r, r.processed_per_block.astype(np.int64), list(r.sm_ids)


def run_launch(spec, img, sx, sy, ref, pinned, xs, ys):
    key, chapter, conn, blocks, tpb = spec
    width, height = img.shape[0], img.shape[1]
    r, per_block, sm_ids = _fill(chapter, img, sx, sy, conn, blocks, tpb)

    check_pixel_exact(r, img, ref)
    sizes = check_level_trace(r)
    census = owner_census(r, blocks, per_block)
    rule = grid_stride_counts(sizes, blocks, tpb)
    np.testing.assert_array_equal(census, rule)

    checks = {"pixel_exact_vs_njit_oracle": "OK",
              "level_trace_whole": "OK",
              "census_vs_block_counters": "OK",
              "census_vs_grid_stride_rule": "OK"}
    if pinned is not None:
        trace_key = "level_sizes" if conn == 4 else "conn8_level_sizes"
        np.testing.assert_array_equal(sizes, pinned[trace_key])
        checks[f"level_sizes_vs_pinned.{trace_key}"] = "OK"
        np.testing.assert_array_equal(
            census, grid_stride_counts(pinned[trace_key], blocks, tpb))
        checks["census_vs_rule_on_pinned_trace"] = "OK"
        if conn == 4 and blocks == pinned["multi_blocks"] \
                and tpb == pinned["multi_tpb"]:
            np.testing.assert_array_equal(census,
                                          pinned["processed_per_block"])
            checks["census_vs_pinned.processed_per_block"] = "OK"
        if blocks == len(pinned["sm_ids"]):
            # placement is reported, never asserted
            checks["sm_ids_vs_pinned"] = ("same" if sm_ids == pinned["sm_ids"]
                                          else "differ")

    widest = int(np.argmax(sizes))
    cx, cy = ring_center(widest, sx, sy, conn)
    assert r.depth[cx, cy] == widest, (cx, cy, r.depth[cx, cy], widest)
    origins = {
        "seed": _origin(sx, sy, CROP, width, height),
        "ring": _origin(cx, cy, CROP, width, height),
        "seed_wide": _origin(sx, sy, CROP_WIDE, width, height),
    }
    sizes_px = {"seed": CROP, "ring": CROP, "seed_wide": CROP_WIDE}
    arrays = {
        "owner_samples": r.owner[np.ix_(xs, ys)].astype(np.int16),
        "depth_samples": r.depth[np.ix_(xs, ys)].astype(np.int32),
        "block_px": census,
        "sm_ids": np.array(sm_ids, dtype=np.int32),
        "kernel_ms": np.float64(r.kernel_ms),
    }
    crops = {}
    for name, (x0, y0) in origins.items():
        s = sizes_px[name]
        arrays[f"crop_{name}"] = _crop(r.owner, x0, y0, s)
        win = r.depth[x0:x0 + s, y0:y0 + s]
        crops[name] = {"origin_xy": [x0, y0], "size": s,
                       "depth_min": int(win.min()),
                       "depth_max": int(win.max()),
                       "distinct_owners": int(np.unique(
                           arrays[f"crop_{name}"]).size),
                       "neighbour_differ_pct": _differ_pct(
                           arrays[f"crop_{name}"])}
    crops["ring"].update({"level": widest, "level_size": int(sizes[widest]),
                          "center_xy": [int(cx), int(cy)]})

    row = {
        "chapter": chapter,
        "kernel": "global" if chapter == "ch02" else "multi",
        "connectivity": conn, "blocks": blocks, "tpb": tpb,
        "owner_dtype": str(r.owner.dtype),
        "kernel_ms": r.kernel_ms, "total_ms": r.total_ms,
        "levels": r.levels, "filled": r.filled,
        "peak_level": r.peak_level,
        "level_trace_truncated": bool(r.level_trace_truncated),
        "block_px": census.tolist(),
        "sm_ids": sm_ids,
        "distinct_sms": len({s for s in sm_ids if s >= 0}),
        "block0_radius": block0_radius(r),
        "checks": checks,
        "crops": crops,
        "mixing": mixing(arrays["owner_samples"], census, r.filled),
    }
    if pinned is not None:  # the same launch's median there, for context
        if chapter == "ch02":
            ms_key = "dual_global_kernel_ms"
        else:
            ms_key = "multi_kernel_ms" if conn == 4 else "conn8_kernel_ms"
        row["pinned_kernel_ms_median"] = pinned[ms_key]
    del r
    _release()

    m = row["mixing"]
    print(f"{key:22s} {row['kernel_ms']:8.2f} {row['levels']:>6d} "
          f"{row['filled']:>11,d} {row['block0_radius']:>5d} "
          f"{m['distinct_owners_per_cell_mean']:6.1f} "
          f"{m['neighbour_differ_pct']:6.1f}% {m['shuffled_differ_pct']:6.1f}% "
          f"{' '.join(sorted(set(checks.values())))}")
    return row, arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="a 1024^2 square, no pinned checks, no files")
    args = ap.parse_args(argv)

    device = cuda.get_current_device()
    dev_name = device.name.decode() if isinstance(device.name, bytes) \
        else str(device.name)
    sm_count = int(device.MULTIPROCESSOR_COUNT)
    print(f"Device: {dev_name.strip()} ({sm_count} SMs)")

    print("Warming up JITs (3 kernels + 2 oracles)...")
    warm_img, wx, wy = scenes.square_scene(64, 64, 32, 32)
    dual_flood_fill(warm_img, wx, wy, kernel="global")
    flood_fill(warm_img, wx, wy)
    flood_fill(warm_img, wx, wy, connectivity=8)
    cpu_flood_fill(warm_img, wx, wy)
    cpu_flood_fill_8(warm_img, wx, wy)

    scene_name, builder = QUICK_SCENE if args.quick else SCENE
    pinned = None
    if not args.quick:
        with open(PINNED) as f:
            pinned = next(s for s in json.load(f)["scenes"]
                          if s["scene"] == scene_name)
    img, sx, sy = builder()
    width, height = img.shape[0], img.shape[1]
    xs, ys = floor_samples(width, height)

    print(f"\n{'launch':22s} {'kern ms':>8s} {'levels':>6s} {'filled':>11s} "
          f"{'b0 R':>5s} {'own/c':>6s} {'differ':>7s} {'shuffl':>7s} checks")
    rows, arrays = {}, {}
    for conn, oracle in ((4, cpu_flood_fill), (8, cpu_flood_fill_8)):
        ref = oracle(img, sx, sy)
        for spec in (s for s in LAUNCHES if s[2] == conn):
            row, arr = run_launch(spec, img, sx, sy, ref, pinned, xs, ys)
            rows[spec[0]] = row
            arrays.update({f"{spec[0]}.{k}": v for k, v in arr.items()})
        del ref
        gc.collect()

    print("\nNotes: kernel ms is ONE instrumented run per launch (the owner "
          "map is the product here, not the time; the pinned session's "
          "medians are the times to cite). 'b0 R' = the last ring owned by "
          "block 0 alone; 'own/c' = mean distinct owners among a toy cell's "
          f"{DOTS * DOTS} dots; 'differ' = adjacent dots with different "
          "owners, 'shuffl' = the same for a random shuffle of the dots.")
    if args.quick:
        print("--quick: nothing written.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    npz_path = os.path.join(OUT_DIR, f"owner_map_{stamp}.npz")
    np.savez_compressed(npz_path, device=np.array(dev_name.strip()),
                        sample_x=xs, sample_y=ys,
                        seed_xy=np.array([sx, sy], dtype=np.int32),
                        **arrays)
    payload = {
        "device": dev_name.strip(),
        "sm_count": sm_count,
        "scene": {"name": scene_name, "width": width, "height": height,
                  "seed_xy": [sx, sy], "blob": "full bleed (the image)"},
        "floor_grid": {
            "toy": "video/scenes/chain/toy.py square()",
            "toy_n": TOY_N, "toy_side": TOY_SIDE, "toy_offset": TOY_OFFSET,
            "dots_per_cell": DOTS, "samples_per_axis": TOY_SIDE * DOTS,
            "stride_px": width / (TOY_SIDE * DOTS),
            "rule": "sample_x[s] = floor((s + 0.5) * width / 126); dot "
                    "(a, b) of toy cell (r, c) is "
                    "samples[6 * (r - 1) + a, 6 * (c - 1) + b]",
            "index_order": "[x, y] = toy [row, col]",
        },
        "npz": os.path.basename(npz_path),
        "npz_layout": "<launch>.owner_samples (126, 126) int16, "
                      "<launch>.depth_samples (126, 126) int32, "
                      "<launch>.crop_seed / crop_ring (64, 64) int16, "
                      "<launch>.crop_seed_wide (192, 192) int16, "
                      "<launch>.block_px, <launch>.sm_ids, "
                      "<launch>.kernel_ms; device, sample_x, sample_y, "
                      "seed_xy",
        "pinned_session": os.path.basename(PINNED),
        "notes": (
            "Owner positions are race-dependent (warp-aggregated discovery "
            "order inside a level); owner counts are not (grid-stride "
            "rule on level_sizes) and are cross-checked against the "
            "pinned session. kernel_ms is one instrumented run per launch, "
            "not a median."),
        "launches": rows,
    }
    json_path = os.path.join(OUT_DIR, f"owner_map_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to:\n  {json_path} "
          f"({os.path.getsize(json_path):,} B)\n  {npz_path} "
          f"({os.path.getsize(npz_path):,} B)")


if __name__ == "__main__":
    main()
