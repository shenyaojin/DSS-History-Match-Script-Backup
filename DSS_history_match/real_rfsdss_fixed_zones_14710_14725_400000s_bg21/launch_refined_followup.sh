#!/bin/sh
set -eu
cd /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner
case_dir=scripts/DSS_history_match/real_rfsdss_fixed_zones_14710_14725_400000s_bg21
export PYTHONPATH="$PWD/fibeRIS/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig
test ! -e "$case_dir/mesh_refinement_followup/STARTED"
mkdir -p "$case_dir/mesh_refinement_followup"
exec /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python -u \
    "$case_dir/continue_refined_inversion.py" \
    > "$case_dir/mesh_refinement_followup/driver.log" 2>&1
