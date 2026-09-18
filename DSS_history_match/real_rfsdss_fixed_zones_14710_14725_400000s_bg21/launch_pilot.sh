#!/bin/sh
set -eu
cd /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner
exec env PYTHONPATH="$PWD/fibeRIS/src" MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
  /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python -u \
  scripts/DSS_history_match/real_rfsdss_fixed_zones_14710_14725_400000s_bg21/run_pilot.py \
  >> scripts/DSS_history_match/real_rfsdss_fixed_zones_14710_14725_400000s_bg21/pilot_5iter/driver.log 2>&1
