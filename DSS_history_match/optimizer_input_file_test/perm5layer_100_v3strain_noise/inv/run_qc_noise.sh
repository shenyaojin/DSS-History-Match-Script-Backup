#!/usr/bin/env bash
# Forward-QC the final L1 alpha for each noise level.
#
# For each inv/noise_<tag>/ this runs the MOOSE forward/adjoint model with the
# last row of parameter_history_L1.csv and produces qc_strain_L1_final.png,
# the visual proof that the inverted alpha reproduces the (noisy) strain.
#
# Run OUTSIDE the Codex sandbox (needs MPI). Run after run_all_noise.sh.
#
# Usage:
#   bash run_qc_noise.sh            # QC all four levels
#   bash run_qc_noise.sh 2pct       # QC only the listed levels
#   NP=20 bash run_qc_noise.sh      # override MPI process count

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

ALL_TAGS=("0p5pct" "1pct" "2pct" "5pct")
if [[ $# -gt 0 ]]; then
  TAGS=("$@")
else
  TAGS=("${ALL_TAGS[@]}")
fi

export PYTHONPATH="$REPO_ROOT/fibeRIS/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mplconfig

for tag in "${TAGS[@]}"; do
  run_dir="$INV_DIR/noise_${tag}"
  hist="$run_dir/parameter_history_L1.csv"
  if [[ ! -f "$hist" ]]; then
    echo "SKIP $tag: $hist not found (run the inversion first)." >&2
    continue
  fi
  echo "==================================================================="
  echo "Forward QC for noise level: $tag"
  echo "==================================================================="
  ( cd "$run_dir" && "$PYBIN" "$run_dir/run_parameter_history_qc.py" \
      --history-file "$hist" \
      --row -1 \
      --label "L1_final" \
      --np "$NP" )
  echo ""
done

echo "QC complete. See each noise_<tag>/qc_strain_L1_final.png"
