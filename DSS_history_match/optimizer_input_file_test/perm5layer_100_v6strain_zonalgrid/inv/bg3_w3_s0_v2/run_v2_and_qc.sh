#!/usr/bin/env bash
set -u
V6=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid
PY=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python
export PYTHONPATH=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/fibeRIS/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig
cd "$V6/inv/bg3_w3_s0_v2" || exit 1
ZONAL_MODE=exact ZONAL_NP=20 "$PY" 107_optimization_runner_zonal_L1.py > zonal_L1.stdout 2>&1
status=$?
echo DONE_EXIT=$status >> zonal_L1.stdout
[ "$status" -eq 0 ] && [ -f optimized_theta_zonal.txt ] || exit 1
"$PY" run_parameter_history_qc.py --history-file parameter_history_L1.csv --row -1 --label zonal_final --np 20 > qc.stdout 2>&1 || exit 1
"$PY" finish_comparison.py
