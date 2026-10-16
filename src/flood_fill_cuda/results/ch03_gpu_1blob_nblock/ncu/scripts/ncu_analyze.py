"""Turn the ncu reports in reps/ into curated tables.

For every report: export raw / sass / mixed (cuda,sass) CSVs into raw/
(once), then compute
  - headline metrics (duration, clocks, DRAM, L1/L2, sectors per request,
    L2 atomics, occupancy, issue, divergence);
  - the PC-sampling stall breakdown (share of all samples per reason);
  - a location split: every sample is put in one bucket by WHERE the warp
    was stalled (source line text of ch03's kernels.py, or the driver's
    grid-sync code that ncu cannot list), so the buckets add up to 100%;
  - the top stalled source lines and SASS instructions.
Writes CSVs + a JSON dump into the scratch folder. No absolute paths or
host names go into the curated outputs.
"""

import csv
import json
import os
import re
import subprocess
import sys

S = os.path.dirname(os.path.abspath(__file__))
REPS = os.path.join(S, "reps")
RAW = os.path.join(S, "raw")
OUT = os.path.join(S, "curated")
NCU = os.environ.get("NCU", "ncu")

LEVELS = {"sq_2000_center": 501, "sq_4000_corner": 4000,
          "sq_8000_center": 4001}
FILLED = {"sq_2000_center": 1_000_000, "sq_4000_corner": 16_000_000,
          "sq_8000_center": 64_000_000}
SCENE_ORDER = ["sq_2000_center", "sq_4000_corner", "sq_8000_center"]
CFG_ORDER = ["wc_bare", "conn8_bare"]

REASONS = ["barrier", "long_scoreboard", "wait", "short_scoreboard",
           "branch_resolving", "lg_throttle", "mio_throttle", "membar",
           "math_pipe_throttle", "misc", "no_instructions", "not_selected",
           "selected", "dispatch_stall", "drain", "imc_miss", "sleeping",
           "tex_throttle"]
# short names used by the source page
SRC_REASON = {"barrier": "barrier", "long_sb": "long_scoreboard",
              "wait": "wait", "short_sb": "short_scoreboard",
              "branch_resolving": "branch_resolving", "lg": "lg_throttle",
              "mio": "mio_throttle", "membar": "membar",
              "math": "math_pipe_throttle", "misc": "misc",
              "no_inst": "no_instructions", "not_selected": "not_selected",
              "selected": "selected", "dispatch": "dispatch_stall",
              "drain": "drain", "imc": "imc_miss", "sleep": "sleeping",
              "tex": "tex_throttle"}

BUCKETS = ["grid_sync", "rgb_probe", "visited_cas", "enqueue",
           "queue_read", "paint_store", "loop_decode"]
BUCKET_LABEL = {
    "grid_sync": "grid.sync (driver code + call sites + rear read)",
    "rgb_probe": "neighbor bounds + 3-byte RGB is-red loads",
    "visited_cas": "visited atomic CAS",
    "enqueue": "warp-aggregated enqueue (rear atomicAdd, shfl, store)",
    "queue_read": "frontier queue entry load",
    "paint_store": "paint (3 byte stores) + depth store",
    "loop_decode": "loop control, x/y decode (64-bit div), dx/dy",
}


def bucket_of(src):
    s = src.strip()
    if "grid.sync()" in s or s.startswith("new_rear = q_state"):
        return "grid_sync"
    if "_is_red(img, nx, ny)" in s:
        return "rgb_probe"
    if "cuda.atomic.cas" in s:
        return "visited_cas"
    if "_warp_enqueue_global" in s or "nx * height + ny, counters" in s:
        return "enqueue"
    if re.match(r"pixel = queue\[(idx|i)\]", s):
        return "queue_read"
    if s.startswith("img[x, y,") or s.startswith("depth[x, y]"):
        return "paint_store"
    return "loop_decode"


def n(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def export(rep):
    base = os.path.join(RAW, os.path.basename(rep)[:-len(".ncu-rep")])
    jobs = {"raw": ["--page", "raw", "--csv", "--print-units", "base"],
            "sass": ["--page", "source", "--csv", "--print-source", "sass"],
            "mixed": ["--page", "source", "--csv", "--print-source",
                      "cuda,sass"]}
    paths = {}
    for k, args in jobs.items():
        p = f"{base}_{k}.csv"
        if not os.path.exists(p) or os.path.getmtime(p) < os.path.getmtime(rep):
            with open(p, "w") as f:
                subprocess.run([NCU, "--import", rep] + args, stdout=f,
                               check=True)
        paths[k] = p
    return paths


def read_raw(p):
    rows = list(csv.reader(open(p)))
    h, u, v = rows[0], rows[1], rows[2]
    return {h[i]: v[i] for i in range(len(h))}, {h[i]: u[i] for i in range(len(h))}


def read_sass(p):
    """{function: {'samples': int, 'reasons': {...}, 'ins': [...]}}"""
    funcs = {}
    fn = hdr = None
    for r in csv.reader(open(p)):
        if not r:
            continue
        if r[0] == "Kernel Name":
            name = r[1]
            fn = ("kernel" if "cudapy::" in name else name.split("(")[0])
            funcs[fn] = {"samples": 0, "reasons": {}, "ins": []}
            continue
        if r[0] == "Address":
            hdr = r
            continue
        if hdr is None or len(r) < len(hdr):
            continue
        d = dict(zip(hdr, r))
        s = n(d["Warp Stall Sampling (All Samples)"])
        f = funcs[fn]
        f["samples"] += s
        for k in hdr:
            if k.startswith("stall_") and "(Not" not in k:
                f["reasons"][SRC_REASON[k[6:]]] = (
                    f["reasons"].get(SRC_REASON[k[6:]], 0) + n(d[k]))
    return funcs


def read_mixed(p):
    """Per source line: samples, reasons, and its SASS instructions."""
    rows = list(csv.reader(open(p)))
    hdr = next(r for r in rows if r and r[0] == "Line No")
    si = [i for i, k in enumerate(hdr)
          if k.startswith("stall_") and "(Not" not in k]
    i_exec = hdr.index("Instructions Executed")
    lines = []
    cur = None
    for r in rows:
        if len(r) < len(hdr) or r[0] == "Line No":
            continue
        reasons = {SRC_REASON[hdr[i][6:]]: n(r[i]) for i in si if n(r[i])}
        if r[0] != "":
            cur = {"line": int(r[0]), "src": r[1].strip(),
                   "samples": n(r[4]), "reasons": reasons, "ins": []}
            lines.append(cur)
        elif cur is not None:
            cur["ins"].append({"addr": r[2][-5:], "sass": r[3].strip(),
                               "samples": n(r[4]), "exec": n(r[i_exec]),
                               "reasons": reasons})
    return lines


def analyse(rep):
    name = os.path.basename(rep)[:-len(".ncu-rep")]
    cfg, rest = (("wc_bare", name[len("wc_bare_"):]) if name.startswith("wc_bare")
                 else ("conn8_bare", name[len("conn8_bare_"):]))
    scene = next(s for s in SCENE_ORDER if rest.startswith(s))
    clock, sset = rest[len(scene) + 1:].split("_")
    paths = export(rep)
    m, units = read_raw(paths["raw"])
    sass = read_sass(paths["sass"])
    lines = read_mixed(paths["mixed"])

    g = lambda k: n(m.get(k))
    dur_s = g("gpu__time_duration.sum") * 1e-9
    total = g("smsp__pcsamp_sample_count")
    glob = {r: g(f"smsp__pcsamp_warps_issue_stalled_{r}") for r in REASONS}
    # ncu names it no_instructions in pcsamp metrics
    glob_sum = sum(glob.values())

    # --- location split
    loc = {b: 0.0 for b in BUCKETS}
    loc_reasons = {b: {} for b in BUCKETS}
    for ln in lines:
        b = bucket_of(ln["src"])
        loc[b] += ln["samples"]
        for r, v in ln["reasons"].items():
            loc_reasons[b][r] = loc_reasons[b].get(r, 0) + v
    kernel_visible = sass.get("kernel", {}).get("samples", 0)
    line_sum = sum(ln["samples"] for ln in lines)
    wrappers = sum(f["samples"] for k, f in sass.items() if k != "kernel")
    hidden = total - kernel_visible - wrappers
    loc["grid_sync"] += wrappers + hidden
    unmapped = kernel_visible - line_sum   # kernel SASS without a line
    loc["loop_decode"] += unmapped
    # reasons of the hidden (driver) samples = global minus visible
    vis = {}
    for f in sass.values():
        for r, v in f["reasons"].items():
            vis[r] = vis.get(r, 0) + v
    hidden_reasons = {r: glob[r] - vis.get(r, 0) for r in REASONS}
    for k, f in sass.items():
        if k == "kernel":
            continue
        for r, v in f["reasons"].items():
            loc_reasons["grid_sync"][r] = loc_reasons["grid_sync"].get(r, 0) + v
    for r, v in hidden_reasons.items():
        loc_reasons["grid_sync"][r] = loc_reasons["grid_sync"].get(r, 0) + v

    ld_req = g("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    st_req = g("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum")
    at_req = g("l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum")
    dram_r = g("dram__bytes_read.sum")
    dram_w = g("dram__bytes_write.sum")
    filled = FILLED[scene]
    levels = LEVELS[scene]
    # srcunit_tex comes with the full set; aperture_device is in EXTRA
    cas_l2 = (g("lts__t_requests_srcunit_tex_op_atom_dot_cas.sum")
              or g("lts__t_requests_aperture_device_op_atom_dot_cas.sum"))
    alu_l2 = (g("lts__t_requests_srcunit_tex_op_atom_dot_alu.sum")
              or g("lts__t_requests_aperture_device_op_atom_dot_alu.sum"))
    alu_max = g("lts__t_requests_aperture_device_op_atom_dot_alu.max")
    alu_sum = g("lts__t_requests_aperture_device_op_atom_dot_alu.sum")
    row = {
        "kernel": cfg, "scene": scene,
        "run": {"base_full": "base", "none_light": "boost",
                "none_pcs": "hot"}[f"{clock}_{sset}"],
        "clock_control": clock, "set": sset,
        "grid": f"{int(g('launch__grid_size'))}x{int(g('launch__block_size'))}",
        "registers_per_thread": int(g("launch__registers_per_thread")),
        "levels": levels, "filled_px": filled,
        "duration_ms": dur_s * 1e3,
        "us_per_level": dur_s * 1e6 / levels,
        "sm_clock_ghz": g("sm__cycles_elapsed.avg.per_second") / 1e9,
        "dram_clock_ghz": g("dram__cycles_elapsed.avg.per_second") / 1e9,
        "dram_read_mb": dram_r / 1e6, "dram_write_mb": dram_w / 1e6,
        "dram_bytes_per_px": (dram_r + dram_w) / filled,
        "dram_gb_s": (dram_r + dram_w) / dur_s / 1e9,
        "dram_pct_of_peak": g("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        # ncu's DRAM peak at the memory clock it ran at (32 B/cycle on
        # AD107, read from the full-set reports; light sets lack it)
        "dram_peak_gb_s_ncu": (g("dram__bytes.sum.peak_sustained") or 32.0)
        * g("dram__cycles_elapsed.avg.per_second") / 1e9,
        "l2_hit_pct": g("lts__t_sector_hit_rate.pct"),
        "l2_read_hit_pct": g("lts__t_sector_op_read_hit_rate.pct"),
        "l2_atom_hit_pct": g("lts__t_sector_op_atom_hit_rate.pct"),
        "l2_throughput_pct": g("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        "l1_hit_pct": g("l1tex__t_sector_hit_rate.pct"),
        "l1_ld_hit_pct": g("l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"),
        "l1_throughput_pct": g("l1tex__throughput.avg.pct_of_peak_sustained_active"),
        "lsu_data_wavefronts_pct": g("l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_active"),
        "lsu_inst_pct": g("sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"),
        "ld_requests_per_px": ld_req / filled,
        "st_requests_per_px": st_req / filled,
        "atom_requests_per_px": at_req / filled,
        "ld_sectors_per_req": g("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"),
        "st_sectors_per_req": g("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"),
        "atom_sectors_per_req": g("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom.ratio"),
        "l2_cas_per_px": cas_l2 / filled,
        "l2_rear_atomics_per_px": alu_l2 / filled,
        "l2_atom_ops_per_s_g": (cas_l2 + alu_l2) / dur_s / 1e9,
        "l2_atomic_unit_busy_avg_pct": g("lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed"),
        "l2_atomic_unit_busy_max_slice_pct": g("lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed"),
        "rear_hot_slice_share_pct": 100 * alu_max / alu_sum if alu_sum else 0,
        "achieved_occupancy_pct": g("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "theoretical_occupancy_pct": g("sm__maximum_warps_per_active_cycle_pct"),
        "issue_active_pct": g("smsp__issue_active.avg.pct_of_peak_sustained_active"),
        "ipc_per_sm": g("sm__inst_executed.avg.per_cycle_active")
        if m.get("sm__inst_executed.avg.per_cycle_active") else
        g("smsp__inst_executed.sum") / max(g("smsp__cycles_active.avg"), 1) / 24,
        "avg_threads_per_inst": g("smsp__thread_inst_executed_per_inst_executed.ratio"),
        "branch_uniform_pct": g("smsp__sass_average_branch_targets_threads_uniform.pct"),
        "inst_executed_per_px": g("smsp__inst_executed.sum") / filled,
        "pc_samples": total,
        "pc_samples_hidden_driver": hidden,
        "pc_samples_dropped_bytes": g("smsp__pcsamp_dropped_bytes"),
    }
    for r in REASONS:
        row[f"stall_{r}_pct"] = 100 * glob[r] / glob_sum if glob_sum else 0
    for b in BUCKETS:
        row[f"where_{b}_pct"] = 100 * loc[b] / total if total else 0
    top_lines = sorted(lines, key=lambda x: -x["samples"])[:12]
    top_ins = sorted((dict(i, line=ln["line"], src=ln["src"]) for ln in lines
                      for i in ln["ins"]), key=lambda x: -x["samples"])[:15]
    # One PC sample per SM per interval picks one of the SM's resident
    # warps, so a sample stands for interval x warps_per_sm warp-cycles.
    interval = g("smsp__pcsamp_interval_cycles") or 1024.0
    warps_per_sm = (g("launch__grid_size") * g("launch__block_size") / 32
                    / 24)
    clk = g("sm__cycles_elapsed.avg.per_second")
    for t in top_ins:
        t["wait_cycles"] = (t["samples"] * interval * warps_per_sm / t["exec"]
                            if t["exec"] else 0.0)
        t["wait_ns"] = t["wait_cycles"] / clk * 1e9 if clk else 0.0
    detail = {
        "row": row,
        "where_reasons": {b: {r: v for r, v in sorted(
            loc_reasons[b].items(), key=lambda kv: -kv[1]) if v > 0}
            for b in BUCKETS},
        "hidden_reasons": {r: v for r, v in sorted(
            hidden_reasons.items(), key=lambda kv: -kv[1]) if v},
        "functions": {k: f["samples"] for k, f in sass.items()},
        "top_lines": [{"line": t["line"], "src": t["src"], "samples": t["samples"],
                       "pct": 100 * t["samples"] / total,
                       "bucket": bucket_of(t["src"]),
                       "reasons": dict(sorted(t["reasons"].items(),
                                              key=lambda kv: -kv[1])[:4])}
                      for t in top_lines],
        "top_sass": [{"line": t["line"], "sass": t["sass"],
                      "samples": t["samples"], "pct": 100 * t["samples"] / total,
                      "exec": t["exec"], "wait_cycles": t["wait_cycles"],
                      "wait_ns": t["wait_ns"],
                      "reasons": dict(sorted(t["reasons"].items(),
                                             key=lambda kv: -kv[1])[:3])}
                     for t in top_ins],
    }
    return detail


RUN_ORDER = ["base", "boost", "hot"]
META = ["kernel", "scene", "run", "clock_control", "set", "grid",
        "sm_clock_ghz", "dram_clock_ghz", "duration_ms"]


def _fmt(v, nd=4):
    return round(v, nd) if isinstance(v, float) else v


def _write(name, fields, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})


def _reasons_str(d, total):
    return "; ".join(f"{k} {100 * v / total:.0f}%" for k, v in d.items()
                     if total and 100 * v / total >= 5)


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    reps = sorted(os.path.join(REPS, f) for f in os.listdir(REPS)
                  if f.endswith(".ncu-rep"))
    details = [analyse(r) for r in reps]
    key = lambda d: (CFG_ORDER.index(d["row"]["kernel"]),
                     SCENE_ORDER.index(d["row"]["scene"]),
                     RUN_ORDER.index(d["row"]["run"]))
    details.sort(key=key)
    with open(os.path.join(S, "details.json"), "w") as f:
        json.dump(details, f, indent=1)
    rows = [d["row"] for d in details]

    # 1. headline metrics: the full (base) and light (boost) profiles
    mfields = META + [k for k in rows[0] if k not in META
                      and not k.startswith(("stall_", "where_", "pc_samples"))]
    _write("ncu_metrics.csv", mfields,
           [r for r in rows if r["run"] in ("base", "boost")])
    # 2. stall reasons, share of all PC samples
    sfields = META + ["pc_samples"] + [f"stall_{r}_pct" for r in REASONS]
    _write("ncu_stall_reasons.csv", sfields, rows)
    # 3. where the samples sit (buckets add to 100)
    lrows = []
    for d in details:
        r = dict(d["row"])
        for b in BUCKETS:
            tot = sum(d["where_reasons"][b].values())
            r[f"{b}_reasons"] = _reasons_str(d["where_reasons"][b], tot)
        lrows.append(r)
    lfields = (META + ["pc_samples", "pc_samples_hidden_driver"]
               + [f"where_{b}_pct" for b in BUCKETS]
               + [f"{b}_reasons" for b in BUCKETS])
    _write("ncu_stall_location.csv", lfields, lrows)
    # 4. top source lines (base and hot runs)
    trows = []
    for d in details:
        r = d["row"]
        if r["run"] == "boost":
            continue
        for rank, t in enumerate(d["top_lines"][:8], 1):
            trows.append({"kernel": r["kernel"], "scene": r["scene"],
                          "run": r["run"], "rank": rank,
                          "line": t["line"], "source": t["src"],
                          "bucket": t["bucket"], "pct_of_samples": t["pct"],
                          "top_reasons": _reasons_str(t["reasons"], t["samples"])})
    _write("ncu_top_source_lines.csv",
           ["kernel", "scene", "run", "rank", "line", "source", "bucket",
            "pct_of_samples", "top_reasons"], trows)
    # 5. top SASS instructions with the average wait per execution
    arows = []
    for d in details:
        r = d["row"]
        if r["run"] == "boost":
            continue
        for rank, t in enumerate(d["top_sass"][:10], 1):
            arows.append({"kernel": r["kernel"], "scene": r["scene"],
                          "run": r["run"], "sm_clock_ghz": r["sm_clock_ghz"],
                          "rank": rank, "line": t["line"], "sass": t["sass"],
                          "pct_of_samples": t["pct"],
                          "warp_executions": int(t["exec"]),
                          "wait_cycles_per_exec": t["wait_cycles"],
                          "wait_ns_per_exec": t["wait_ns"],
                          "top_reasons": _reasons_str(t["reasons"], t["samples"])})
    _write("ncu_top_sass.csv",
           ["kernel", "scene", "run", "sm_clock_ghz", "rank", "line", "sass",
            "pct_of_samples", "warp_executions", "wait_cycles_per_exec",
            "wait_ns_per_exec", "top_reasons"], arows)

    # 6. policy timing session + 7. the time split on its clock
    tdir = os.path.join(S, "timing")
    tjs = sorted(f for f in os.listdir(tdir) if f.endswith(".json"))
    tj = json.load(open(os.path.join(tdir, tjs[-1])))
    session = tjs[-1][:-len(".json")]
    probe = {k: v["median"] for k, v in tj["barrier_us_per_level"].items()}
    trow = []
    for sc in tj["scenes"]:
        for cfg in CFG_ORDER:
            trow.append({"session": session, "kernel": cfg, "scene": sc["scene"],
                         "grid": f"{tj['grids'][cfg]}x{tj['tpb']}",
                         "levels": sc["levels"], "filled_px": sc["filled"],
                         "kernel_ms_median": sc[f"{cfg}_ms"],
                         "kernel_ms_min": sc[f"{cfg}_ms_min"],
                         "kernel_ms_max": sc[f"{cfg}_ms_max"],
                         "rounds": tj["rounds"],
                         "conn8_over_wc_median": sc["conn8_over_wc_median"],
                         "conn8_over_wc_min": sc["conn8_over_wc_min"],
                         "conn8_over_wc_max": sc["conn8_over_wc_max"],
                         "paint_match": sc["paint_match"],
                         "clk_sm_median_mhz": sc.get("clk_sm_median"),
                         "clk_mem_median_mhz": sc.get("clk_mem_median")})
    _write("timing_session.csv", list(trow[0]), trow)

    hot = {(d["row"]["kernel"], d["row"]["scene"]): d["row"]
           for d in details if d["row"]["run"] == "hot"}
    boost = {(d["row"]["kernel"], d["row"]["scene"]): d["row"]
             for d in details if d["row"]["run"] == "boost"}
    COPY_PEAK_GB_S = 190.0  # ch03 README: measured D2D copy, 187-193 GB/s
    xrows = []
    for t in trow:
        h = hot[(t["kernel"], t["scene"])]
        ms = t["kernel_ms_median"]
        us = probe[t["grid"]]
        floor = t["levels"] * us / 1000
        x = {"kernel": t["kernel"], "scene": t["scene"], "grid": t["grid"],
             "levels": t["levels"], "kernel_ms": ms,
             "barrier_probe_us_per_level": us,
             "hot_profile_sm_clock_ghz": h["sm_clock_ghz"]}
        for b in BUCKETS:
            x[f"{b}_pct"] = h[f"where_{b}_pct"]
            x[f"{b}_ms"] = ms * h[f"where_{b}_pct"] / 100
        x["barrier_floor_ms"] = floor
        x["barrier_floor_pct"] = 100 * floor / ms
        x["barrier_straggler_wait_ms"] = x["grid_sync_ms"] - floor
        x["barrier_straggler_wait_pct"] = x["grid_sync_pct"] - x["barrier_floor_pct"]
        bpp = boost[(t["kernel"], t["scene"])]["dram_bytes_per_px"]
        x["dram_bytes_per_px"] = bpp
        x["dram_gb_s_at_kernel_ms"] = bpp * t["filled_px"] / (ms * 1e-3) / 1e9
        x["dram_pct_of_copy_peak"] = 100 * x["dram_gb_s_at_kernel_ms"] / COPY_PEAK_GB_S
        xrows.append(x)
    _write("time_split.csv", list(xrows[0]), xrows)

    for d in details:
        r = d["row"]
        print(f"{r['kernel']:10s} {r['scene']:15s} {r['run']:5s} "
              f"{r['duration_ms']:7.2f} ms sm={r['sm_clock_ghz']:.2f} "
              f"sync={r['where_grid_sync_pct']:5.1f}% rgb={r['where_rgb_probe_pct']:5.1f}% "
              f"cas={r['where_visited_cas_pct']:5.1f}% enq={r['where_enqueue_pct']:5.1f}% "
              f"q={r['where_queue_read_pct']:4.1f}% paint={r['where_paint_store_pct']:4.1f}% "
              f"loop={r['where_loop_decode_pct']:5.1f}%")
    for x in xrows:
        print({k: _fmt(v, 2) for k, v in x.items()})


if __name__ == "__main__":
    main()
