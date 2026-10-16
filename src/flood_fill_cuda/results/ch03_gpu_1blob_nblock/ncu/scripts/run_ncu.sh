# Run with: bash run_ncu.sh <config> <scene> <base|none> <full|light|pcs>
# Profile one (config, scene) of ch03's bare BFS kernels with ncu.
# Usage: run_ncu.sh <config> <scene> <clock: base|none> [set: full|light]
# Run from the worktree root; the caller wraps it in the GPU flock.
set -euo pipefail
CFG=$1; SCENE=$2; CLK=$3; SET=${4:-full}
S="$(cd "$(dirname "$0")" && pwd)"
NCU=${NCU:-ncu}
case $CFG in
  wc_bare)    KRE='regex:multi_block_global8wc_bare_kernel' ;;
  conn8_bare) KRE='regex:multi_block_global8_bare_kernel' ;;
  *) echo "unknown config $CFG" >&2; exit 2 ;;
esac
EXTRA=lts__d_atomic_input_cycles_active.sum,\
lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed,\
lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed,\
lts__cycles_elapsed.avg,\
lts__t_requests_srcunit_tex_op_atom.sum,\
lts__t_sectors_srcunit_tex_op_atom.sum,\
lts__t_requests_aperture_device_op_atom_dot_cas.sum,\
lts__t_requests_aperture_device_op_atom_dot_alu.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_srcunit_tex_op_write.sum,\
lts__t_sector_hit_rate.pct,\
lts__t_sector_op_read_hit_rate.pct,\
lts__t_sector_op_atom_hit_rate.pct,\
lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom.ratio,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
l1tex__t_sector_hit_rate.pct,\
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum,dram__bytes_write.sum,\
dram__bytes.sum.per_second,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,\
dram__cycles_elapsed.avg.per_second,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
smsp__sass_average_branch_targets_threads_uniform.pct,\
smsp__inst_executed.sum,smsp__cycles_active.avg,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__grid_size,launch__block_size
HOT=()
if [ "$SET" = full ]; then
  SETARGS=(--set full --import-source yes)
elif [ "$SET" = pcs ]; then
  # PC sampling only (few passes), launched hot: the boost-clock cross-check
  SETARGS=(--section WarpStateStats --section SourceCounters --section LaunchStats --import-source yes)
  EXTRA=gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second,launch__registers_per_thread,launch__grid_size,launch__block_size
  HOT=(--hot)
else
  SETARGS=(--section SpeedOfLight --section WarpStateStats --section SchedulerStats --section Occupancy --section LaunchStats --section SourceCounters --import-source yes)
fi
OUT="$S/reps/${CFG}_${SCENE}_${CLK}_${SET}"
echo "+ ncu ${CFG} ${SCENE} clock=${CLK} set=${SET}"
"$NCU" --target-processes all --clock-control "$CLK" --replay-mode kernel \
  -k "$KRE" --launch-skip 2 --launch-count 1 \
  "${SETARGS[@]}" --metrics "regex:smsp__pcsamp_warps_issue_stalled_.*,$EXTRA" \
  -f -o "$OUT" \
  uv run python "$S/ncu_driver.py" --config "$CFG" --scene "$SCENE" --lineinfo "${HOT[@]}"
