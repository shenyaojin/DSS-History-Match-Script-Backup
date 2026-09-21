#!/bin/sh
set -eu
cd /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner
case_dir=scripts/DSS_history_match/real_rfsdss_fixed_zones_14710_14725_400000s_bg21
export PYTHONPATH="$PWD/fibeRIS/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig
exec /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python -u \
    "$case_dir/run_mesh_refinement.py" run \
    > "$case_dir/mesh_refinement_channel_scale/driver.log" 2>&1
