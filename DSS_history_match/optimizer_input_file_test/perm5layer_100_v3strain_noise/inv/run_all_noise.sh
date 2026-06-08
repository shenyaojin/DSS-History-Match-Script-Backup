#!/usr/bin/env bash
# Run the L1 inversion for each noise level, sequentially.
#
# Each level runs in its own inv/noise_<tag>/ folder, reading that folder's
# measurement_data.csv (the noisy observation) and writing all *_L1.* outputs
# and inv_output/ inside that same folder.
#
# Run this OUTSIDE the Codex sandbox (it needs MPI sockets for MOOSE).
#
# Usage:
#   bash run_all_noise.sh            # run all four levels
#   bash run_all_noise.sh 1pct 5pct  # run only the listed levels

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root holds fibeRIS/src for PYTHONPATH. Derived from this script's location:
#   inv/ -> v3strain_noise/ -> optimizer_input_file_test/ -> DSS_history_match/
#        -> scripts/ -> <repo root>
REPO_ROOT="${REPO_ROOT:-$(cd "$INV_DIR/../../../../.." && pwd)}"
PYBIN="$HOME/miniforge/envs/moose/bin/python"

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

echo "Repo root  : $REPO_ROOT"
echo "Python     : $PYBIN"
echo "Levels     : ${TAGS[*]}"
echo ""

for tag in "${TAGS[@]}"; do
  run_dir="$INV_DIR/noise_${tag}"
  runner="$run_dir/106_optimization_runner_L1.py"
  if [[ ! -f "$runner" ]]; then
    echo "ERROR: $runner not found. Run setup_noise_runs.sh first." >&2
    exit 1
  fi
  log="$run_dir/inversion_L1.stdout"
  echo "==================================================================="
  echo "Running L1 inversion for noise level: $tag"
  echo "  dir : $run_dir"
  echo "  log : $log"
  echo "  start: $(date)"
  echo "==================================================================="
  ( cd "$run_dir" && "$PYBIN" "$runner" ) 2>&1 | tee "$log"
  echo "  done : $(date)"
  echo ""
done

echo "All requested noise levels complete."
echo "Next: bash run_qc_noise.sh   (forward QC of each final alpha)"
echo "Then: $PYBIN compare_noise_results.py   (comparison table + overlay plot)"
