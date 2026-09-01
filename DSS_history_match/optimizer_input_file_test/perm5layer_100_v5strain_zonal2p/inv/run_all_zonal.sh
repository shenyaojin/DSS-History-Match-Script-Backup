#!/usr/bin/env bash
# Run the 2-parameter zonal inversion for each staged run folder, sequentially.
#
# Covers both zone variants:
#   inv/exact_<case>/   (ZONAL_MODE=exact)
#   inv/pert_<case>/    (ZONAL_MODE=pert)
#
# The zone variant is derived from the folder-name prefix and exported as
# ZONAL_MODE so the single runner template serves both. Each folder reads its
# own measurement_data.csv and writes all *_zonal.* / *_L1.* outputs inside that
# same folder, so runs are independent.
#
# Run this OUTSIDE the Codex sandbox (it needs MPI sockets for MOOSE).
#
# Usage:
#   bash run_all_zonal.sh                       # run every staged folder
#   bash run_all_zonal.sh exact_rfsdss_10pct    # run only the named folders
#   bash run_all_zonal.sh exact_*               # shell-glob a subset
#   ZONAL_PERT_SHIFT_LAYERS=3 bash run_all_zonal.sh pert_*   # override pert shift

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# inv/ -> v5strain_zonal2p/ -> optimizer_input_file_test/ -> DSS_history_match/
#      -> scripts/ -> <repo root>
REPO_ROOT="${REPO_ROOT:-$(cd "$INV_DIR/../../../../.." && pwd)}"
PYBIN="$HOME/miniforge/envs/moose/bin/python"

if [[ ! -d "$REPO_ROOT/fibeRIS/src/fiberis" ]]; then
  echo "ERROR: fibeRIS not found at $REPO_ROOT/fibeRIS/src" >&2
  echo "       (override by exporting REPO_ROOT before calling this script)" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  FOLDERS=("$@")
else
  FOLDERS=()
  for d in "$INV_DIR"/exact_* "$INV_DIR"/pert_*; do
    [[ -f "$d/107_optimization_runner_zonal_L1.py" ]] && FOLDERS+=("$(basename "$d")")
  done
fi

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  echo "ERROR: no staged run folders found. Run setup_zonal_runs.sh first." >&2
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
  base="$(basename "$folder")"
  run_dir="$INV_DIR/$base"
  runner="$run_dir/107_optimization_runner_zonal_L1.py"
  if [[ ! -f "$runner" ]]; then
    echo "ERROR: $runner not found. Run setup_zonal_runs.sh first." >&2
    exit 1
  fi
  variant="${base%%_*}"          # exact | pert
  log="$run_dir/zonal_L1.stdout"
  echo "==================================================================="
  echo "Running zonal 2-param inversion: $base   (ZONAL_MODE=$variant)"
  echo "  dir  : $run_dir"
  echo "  start: $(date)"
  echo "==================================================================="
  ( cd "$run_dir" && ZONAL_MODE="$variant" "$PYBIN" "$runner" ) 2>&1 | tee "$log"
  echo "  done : $(date)"
  echo ""
done

echo "All requested runs complete."
echo "Next: bash run_qc_zonal.sh   (forward QC of each final theta)"
echo "Then: $PYBIN compare_zonal_results.py   (comparison table + overlay plot)"
