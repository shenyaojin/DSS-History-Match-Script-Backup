#!/bin/sh
set -eu
cd /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner
case_dir=scripts/DSS_history_match/real_rfsdss_fixed_zones_14373_14379_400000s_bg21_coupling
export PATH="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin:$PATH"
export PYTHONPATH="$PWD/fibeRIS/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
test -f "$case_dir/inversion/INPUT_CHECK_PASSED"
test ! -e "$case_dir/inversion/RUNNING"
test ! -e "$case_dir/inversion/objective_history.csv"
test ! -e "$case_dir/inversion/driver.log"
exec /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python -u \
    "$case_dir/run_inversion.py" > "$case_dir/inversion/driver.log" 2>&1
