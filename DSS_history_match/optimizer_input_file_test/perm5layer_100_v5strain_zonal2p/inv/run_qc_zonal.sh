#!/usr/bin/env bash
# Forward-QC the final zonal solution for each staged run folder.
#
# For each inv/{exact,pert}_<case>/ this runs the MOOSE forward model with the
# last row of parameter_history_L1.csv (the expanded 200-dim alpha of the final
# theta) and produces qc_strain_zonal_final.png. Because the runner writes the
# expanded alpha into parameter_history_L1.csv, the stock run_parameter_history_qc.py
# needs no change.
#
# Run OUTSIDE the Codex sandbox (needs MPI). Run after run_all_zonal.sh.
#
# Usage:
#   bash run_qc_zonal.sh                     # QC every staged folder
#   bash run_qc_zonal.sh exact_rfsdss_10pct
#   NP=20 bash run_qc_zonal.sh               # override MPI process count

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$INV_DIR/../../../../.." && pwd)}"
PYBIN="$HOME/miniforge/envs/moose/bin/python"
NP="${NP:-20}"

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

export PYTHONPATH="$REPO_ROOT/fibeRIS/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig

for folder in "${FOLDERS[@]}"; do
  run_dir="$INV_DIR/$(basename "$folder")"
  hist="$run_dir/parameter_history_L1.csv"
  if [[ ! -f "$hist" ]]; then
    echo "SKIP $(basename "$run_dir"): $hist not found (run the inversion first)." >&2
    continue
  fi
  echo "==================================================================="
  echo "Forward QC: $(basename "$run_dir")"
  echo "==================================================================="
  ( cd "$run_dir" && "$PYBIN" "$run_dir/run_parameter_history_qc.py" \
      --history-file "$hist" \
      --row -1 \
      --label "zonal_final" \
      --np "$NP" )
  echo ""
done

echo "QC complete. See each {exact,pert}_<case>/qc_strain_zonal_final.png"
