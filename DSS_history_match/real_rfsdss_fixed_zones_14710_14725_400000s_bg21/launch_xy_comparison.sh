#!/bin/sh
set -u
cd /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner || exit 1
export PYTHONPATH="$PWD/fibeRIS/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig
case_dir=scripts/DSS_history_match/real_rfsdss_fixed_zones_14710_14725_400000s_bg21
python_bin=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python
for mode in fixed_yy free_yy; do
  run_dir="$case_dir/comparison_xx_yy_60/$mode"
  if [ ! -f "$run_dir/INPUT_CHECK_PASSED" ] || [ -e "$run_dir/RUNNING" ] || [ -e "$run_dir/objective_history.csv" ] || [ -e "$run_dir/driver.log" ]; then
    printf 'Refusing unchecked or previously started case: %s\n' "$mode" >&2
    exit 1
  fi
done
# Independent case directories and logs; each worker uses 20 MPI ranks.
"$python_bin" -u "$case_dir/run_xy_comparison.py" fixed_yy > "$case_dir/comparison_xx_yy_60/fixed_yy/driver.log" 2>&1 &
fixed_pid=$!
"$python_bin" -u "$case_dir/run_xy_comparison.py" free_yy > "$case_dir/comparison_xx_yy_60/free_yy/driver.log" 2>&1 &
free_pid=$!
wait "$fixed_pid"
fixed_status=$?
wait "$free_pid"
free_status=$?
"$python_bin" "$case_dir/summarize_xy_comparison.py" > "$case_dir/comparison_xx_yy_60/summary.log" 2>&1
summary_status=$?
printf 'fixed_yy=%s\nfree_yy=%s\nsummary=%s\n' "$fixed_status" "$free_status" "$summary_status" > "$case_dir/comparison_xx_yy_60/LAUNCHER_EXIT_STATUS"
test "$fixed_status" -eq 0 && test "$free_status" -eq 0 && test "$summary_status" -eq 0
