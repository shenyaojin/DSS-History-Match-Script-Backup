#!/bin/sh
set -eu
cd /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner
export PYTHONPATH="$PWD/fibeRIS/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
experiment=scripts/DSS_history_match/real_rfsdss_fixed_zones_14373_14379_400000s_bg21_coupling/background_sensitivity
action=${1:-run}
if test "$action" = run; then
    test ! -e "$experiment/STARTED"
    log_name=driver.log
elif test "$action" = resume; then
    test -f "$experiment/FAILED"
    log_name=resume.log
else
    exit 2
fi
test ! -e "$experiment/$log_name"
exec /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python -u "$experiment/run_experiment.py" "$action" > "$experiment/$log_name" 2>&1
