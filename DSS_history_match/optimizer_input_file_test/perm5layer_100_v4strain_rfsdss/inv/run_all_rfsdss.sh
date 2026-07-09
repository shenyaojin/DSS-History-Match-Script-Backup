#!/usr/bin/env bash
# Run the L1 inversion for each staged rfsdss run folder, sequentially.
#
# Covers the realistic RFS-DSS noise family:
#   inv/rfsdss_<tag>/     (per-channel floor + drift + common-mode + spikes)
#
# Each folder reads its own measurement_data.csv (the noisy observation) and
# writes all *_L1.* outputs and inv_output/ inside that same folder, so runs
# are independent.
#
# Run this OUTSIDE the Codex sandbox (it needs MPI sockets for MOOSE).
#
# Usage:
#   bash run_all_rfsdss.sh                  # run every staged folder
#   bash run_all_rfsdss.sh rfsdss_2pct      # run only the named folders
#   bash run_all_rfsdss.sh rfsdss_*         # shell-glob a subset

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root holds fibeRIS/src for PYTHONPATH (derived from this script's location):
#   inv/ -> v4strain_rfsdss/ -> optimizer_input_file_test/ -> DSS_history_match/
#        -> scripts/ -> <repo root>
REPO_ROOT="${REPO_ROOT:-$(cd "$INV_DIR/../../../../.." && pwd)}"
PYBIN="$HOME/miniforge/envs/moose/bin/python"

if [[ ! -d "$REPO_ROOT/fibeRIS/src/fiberis" ]]; then
  echo "ERROR: fibeRIS not found at $REPO_ROOT/fibeRIS/src" >&2
  echo "       (override by exporting REPO_ROOT before calling this script)" >&2
  exit 1
fi

# Resolve the list of run folders (basenames under inv/).
if [[ $# -gt 0 ]]; then
  FOLDERS=("$@")
else
  FOLDERS=()
  for d in "$INV_DIR"/rfsdss_*; do
    [[ -f "$d/106_optimization_runner_L1.py" ]] && FOLDERS+=("$(basename "$d")")
  done
fi

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  echo "ERROR: no staged run folders found. Run setup_rfsdss_runs.sh first." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/fibeRIS/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig

echo "Repo root  : $REPO_ROOT"
echo "Python     : $PYBIN"
echo "Run folders: ${FOLDERS[*]}"
echo ""

for folder in "${FOLDERS[@]}"; do
  run_dir="$INV_DIR/$(basename "$folder")"
  runner="$run_dir/106_optimization_runner_L1.py"
  if [[ ! -f "$runner" ]]; then
    echo "ERROR: $runner not found. Run setup_rfsdss_runs.sh first." >&2
    exit 1
  fi
  log="$run_dir/inversion_L1.stdout"
  echo "==================================================================="
  echo "Running L1 inversion: $(basename "$run_dir")"
  echo "  dir  : $run_dir"
  echo "  log  : $log"
  echo "  start: $(date)"
  echo "==================================================================="
  ( cd "$run_dir" && "$PYBIN" "$runner" ) 2>&1 | tee "$log"
  echo "  done : $(date)"
  echo ""
done

echo "All requested runs complete."
echo "Next: bash run_qc_rfsdss.sh   (forward QC of each final alpha)"
echo "Then: $PYBIN compare_rfsdss_results.py   (comparison table + overlay plot)"
