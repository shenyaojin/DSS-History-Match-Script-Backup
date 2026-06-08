#!/usr/bin/env bash
# Stage one self-contained L1 inversion run folder per noise level.
#
# Each inv/noise_<tag>/ gets its own copy of the v2 L1 run set plus the noisy
# observation file renamed to measurement_data.csv (the name optimize.i expects).
# Because the runner uses its own directory as WORKDIR, the four runs are fully
# independent and never overwrite each other's histories/outputs.
#
# Re-running this script is safe: it refreshes the template files and the
# measurement_data.csv but leaves any existing inv_output/ and *_L1.* results
# in place.
#
# Usage:
#   bash setup_noise_runs.sh

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3_DIR="$(dirname "$INV_DIR")"
NOISE_DIR="$V3_DIR/noise_adding"

# Source of the working v2 L1 run set.
V2_INV="$(cd "$V3_DIR/../perm5layer_100_v2strain/inv" && pwd)"

# Files copied verbatim into each per-level run folder.
TEMPLATE_FILES=(
  "106_optimization_runner_L1.py"
  "optimize.i"
  "forward_and_adjoint.i"
  "plot_inversion_qc.py"
  "run_parameter_history_qc.py"
)

TAGS=("0p5pct" "1pct" "2pct" "5pct")

echo "v2 template source : $V2_INV"
echo "noise data source  : $NOISE_DIR"
echo ""

for f in "${TEMPLATE_FILES[@]}"; do
  if [[ ! -f "$V2_INV/$f" ]]; then
    echo "ERROR: missing template file $V2_INV/$f" >&2
    exit 1
  fi
done

for tag in "${TAGS[@]}"; do
  run_dir="$INV_DIR/noise_${tag}"
  noisy_csv="$NOISE_DIR/measurement_data_noise_${tag}.csv"
  if [[ ! -f "$noisy_csv" ]]; then
    echo "ERROR: missing noisy data $noisy_csv (run noise_adding/add_noise.py first)" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  for f in "${TEMPLATE_FILES[@]}"; do
    cp -f "$V2_INV/$f" "$run_dir/$f"
  done
  cp -f "$noisy_csv" "$run_dir/measurement_data.csv"
  cp -f "$NOISE_DIR/measurement_data_noise_${tag}.meta" "$run_dir/measurement_data.meta"
  echo "staged: $run_dir  (measurement_data.csv <- measurement_data_noise_${tag}.csv)"
done

echo ""
echo "Done. Each inv/noise_<tag>/ is ready to run independently."
