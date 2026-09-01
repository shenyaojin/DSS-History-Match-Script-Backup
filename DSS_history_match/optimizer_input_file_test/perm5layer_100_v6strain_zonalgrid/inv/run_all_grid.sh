#!/usr/bin/env bash
# Run the 2-parameter zonal inversion for each staged grid run folder.
#
# The zone-position offset (factor C) is derived from the folder-name suffix
# _s<N>:  s0 -> ZONAL_MODE=exact ; sN>0 -> ZONAL_MODE=pert, shift = N layers.
# So one runner template serves all 27 cells; only the env differs.
#
# Runs are launched with a bounded number of parallel workers (default 2), each
# using ZONAL_NP MPI processes (default 10) so 2 x 10 = 20 physical cores.
#
# Run this OUTSIDE the Codex sandbox (it needs MPI sockets for MOOSE).
#
# Usage:
#   bash run_all_grid.sh                  # run every staged folder (2 parallel)
#   bash run_all_grid.sh bg1_w1_s0 ...    # run only the named folders
#   bash run_all_grid.sh 'bg1_*'          # shell-glob a subset
#   PAR=1 bash run_all_grid.sh            # serial; PAR=2 default
#   ZONAL_NP=20 PAR=1 bash run_all_grid.sh   # one folder at a time, full 20 cores

set -uo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$INV_DIR/../../../../.." && pwd)}"
PYBIN="$HOME/miniforge/envs/moose/bin/python"
PAR="${PAR:-2}"
export ZONAL_NP="${ZONAL_NP:-10}"

if [[ ! -d "$REPO_ROOT/fibeRIS/src/fiberis" ]]; then
  echo "ERROR: fibeRIS not found at $REPO_ROOT/fibeRIS/src" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  FOLDERS=("$@")
else
  FOLDERS=()
  for d in "$INV_DIR"/bg*_w*_s*; do
    [[ -f "$d/107_optimization_runner_zonal_L1.py" ]] && FOLDERS+=("$(basename "$d")")
  done
fi

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  echo "ERROR: no staged run folders found. Run setup_grid_runs.sh first." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/fibeRIS/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig

run_one() {
  local base="$1"
  local run_dir="$INV_DIR/$base"
  local shift="${base##*_s}"          # trailing number after _s
  local mode="pert"
  [[ "$shift" == "0" ]] && mode="exact"
  echo ">>> $(date +%H:%M:%S) start $base  (ZONAL_MODE=$mode shift=$shift NP=$ZONAL_NP)"
  ( cd "$run_dir" && ZONAL_MODE="$mode" ZONAL_PERT_SHIFT_LAYERS="$shift" \
      "$PYBIN" 107_optimization_runner_zonal_L1.py > zonal_L1.stdout 2>&1
    echo "ZONAL_CASE_DONE_EXIT=$?" >> zonal_L1.stdout )
  echo "<<< $(date +%H:%M:%S) done  $base  theta=[$(tr '\n' ',' < "$run_dir/optimized_theta_zonal.txt" 2>/dev/null)]"
}

echo "Repo root  : $REPO_ROOT"
echo "Parallel   : $PAR workers x $ZONAL_NP procs"
echo "Run folders: ${#FOLDERS[@]}"
echo ""

# Simple bounded-parallel scheduler.
running=0
for base in "${FOLDERS[@]}"; do
  run_one "$base" &
  running=$((running + 1))
  if [[ $running -ge $PAR ]]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait

echo ""
echo "All requested grid runs complete."
echo "Next: python compare_grid_results.py   (sensitivity table + heatmaps)"
