#!/usr/bin/env bash
set -u
V6=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid
PY=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python
export PYTHONPATH=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/fibeRIS/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig
export PATH=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/real_rfsdss_fixed_zones_14700_14750_400000s || exit 1
case "$1" in unrectified|half_wave_positive) ;; *) exit 2;; esac
mkdir -p "initial_forward_only/$1"
exec 9>"initial_forward_only/$1/run.lock"
flock -n 9 || exit 3
"$PY" -u run_initial_forward_only.py --case "$1" --np 20 > "initial_forward_only/$1/forward.stdout" 2>&1
status=$?
echo DONE_EXIT=$status >> "initial_forward_only/$1/forward.stdout"
exit "$status"
