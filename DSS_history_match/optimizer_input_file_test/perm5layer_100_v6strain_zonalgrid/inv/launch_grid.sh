#!/usr/bin/env bash
# Resume-launcher for the 3x3x3 sensitivity grid.
#
# Scans inv/bg*_w*_s* and runs ONLY the cells that have not finished yet
# (i.e. that lack optimized_theta_zonal.txt), so it is safe to re-run after an
# interruption -- completed cells are never redone.
#
# The unfinished list is built into a bash ARRAY inside this script and passed
# with "${TODO[@]}", which avoids the word-splitting/quoting breakage you get
# when trying to pipe a folder list through a setsid/nohup command line.
#
# Meant to be launched fully detached so it survives the parent shell/session:
#   cd inv && setsid nohup bash launch_grid.sh > grid_master.log 2>&1 < /dev/null &
#
# Env: PAR (parallel workers, default 2), ZONAL_NP (MPI procs per cell, default 10)
#      MEASURED: 10 procs/cell is the sweet spot (~27 min/solve). 5 procs is 2.7x
#      slower per solve (~72 min) and 20 procs is slower again (comms overhead),
#      so 2 workers x 10 procs beats both 4x5 and 1x20 on total throughput.
#      PAR * ZONAL_NP should not exceed the physical core count (20 here).

set -uo pipefail

INV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$INV"

TODO=()
DONE=0
for d in "$INV"/bg*_w*_s*; do
  [[ -d "$d" ]] || continue
  b="$(basename "$d")"
  if [[ -f "$d/optimized_theta_zonal.txt" ]]; then
    DONE=$((DONE + 1))
  else
    TODO+=("$b")
  fi
done

echo "grid resume: ${DONE} already finished, ${#TODO[@]} to run"
if [[ ${#TODO[@]} -eq 0 ]]; then
  echo "nothing to do -- all cells finished."
  exit 0
fi
echo "cells: ${TODO[*]}"
echo ""

exec env PAR="${PAR:-2}" ZONAL_NP="${ZONAL_NP:-10}" \
  bash "$INV/run_all_grid.sh" "${TODO[@]}"
