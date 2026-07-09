#!/usr/bin/env bash
# Forward-QC the final L1 alpha for each staged rfsdss run folder.
#
# For each inv/rfsdss_<tag>/ this runs the MOOSE forward/adjoint model with the
# last row of parameter_history_L1.csv and produces qc_strain_L1_final.png, the
# visual proof that the inverted alpha reproduces the (noisy) strain.
#
# Run OUTSIDE the Codex sandbox (needs MPI). Run after run_all_rfsdss.sh.
#
# Usage:
#   bash run_qc_rfsdss.sh                    # QC every staged folder
#   bash run_qc_rfsdss.sh rfsdss_2pct
#   NP=20 bash run_qc_rfsdss.sh              # override MPI process count

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root holds fibeRIS/src for PYTHONPATH (derived from this script's location).
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
  for d in "$INV_DIR"/rfsdss_*; do
    [[ -f "$d/106_optimization_runner_L1.py" ]] && FOLDERS+=("$(basename "$d")")
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
      --label "L1_final" \
      --np "$NP" )
  echo ""
done

echo "QC complete. See each rfsdss_<tag>/qc_strain_L1_final.png"
