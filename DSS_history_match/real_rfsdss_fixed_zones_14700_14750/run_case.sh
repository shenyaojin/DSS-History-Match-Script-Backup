#!/usr/bin/env bash
set -u
V6=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid
PY=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python
export PYTHONPATH=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/fibeRIS/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig
export PATH=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/real_rfsdss_fixed_zones_14700_14750
case "$1" in unrectified|half_wave_positive) ;; *) exit 2;; esac
cd "$ROOT" || exit 1
exec 9>"$1/run.lock"
flock -n 9 || exit 3
"$PY" -u run_inversion.py --case "$1" --np 10 > "$1/inversion.stdout" 2>&1
status=$?
echo DONE_EXIT=$status >> "$1/inversion.stdout"
[ "$status" -eq 0 ] || exit "$status"
"$PY" compare_results.py
