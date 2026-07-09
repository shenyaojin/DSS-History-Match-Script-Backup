#!/usr/bin/env bash
# Stage one self-contained L1 inversion run folder per rfsdss noise dataset.
#
# The realistic RFS-DSS noise family (per-channel floor + drift + common-mode +
# strain-driven spikes) is staged into:
#   inv/rfsdss_<tag>/     from measurement_data_rfsdss_<tag>.csv
#
# Each run folder gets its own copy of the v2 L1 run set plus the noisy
# observation renamed to measurement_data.csv (the name optimize.i expects).
# Because the runner uses its own directory as WORKDIR, all runs are fully
# independent and never overwrite each other's histories/outputs.
#
# Re-running this script is safe: it refreshes the template files and the
# measurement_data.csv but leaves any existing inv_output/ and *_L1.* results
# in place.
#
# Usage:
#   bash setup_rfsdss_runs.sh

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V4_DIR="$(dirname "$INV_DIR")"
NOISE_DIR="$V4_DIR/noise_adding"

# Source of the working v2 L1 run set (shared with the v3 noise study).
V2_INV="$(cd "$V4_DIR/../perm5layer_100_v2strain/inv" && pwd)"

# Files copied verbatim into each per-level run folder.
TEMPLATE_FILES=(
  "106_optimization_runner_L1.py"
  "optimize.i"
  "forward_and_adjoint.i"
  "plot_inversion_qc.py"
  "run_parameter_history_qc.py"
)

# run_folder : noisy_csv_basename  (in noise_adding/)
RUNS=(
  "rfsdss_0p5pct:measurement_data_rfsdss_0p5pct"
  "rfsdss_1pct:measurement_data_rfsdss_1pct"
  "rfsdss_2pct:measurement_data_rfsdss_2pct"
  "rfsdss_5pct:measurement_data_rfsdss_5pct"
  "rfsdss_10pct:measurement_data_rfsdss_10pct"
)

echo "v2 template source : $V2_INV"
echo "noise data source  : $NOISE_DIR"
echo ""

for f in "${TEMPLATE_FILES[@]}"; do
  if [[ ! -f "$V2_INV/$f" ]]; then
    echo "ERROR: missing template file $V2_INV/$f" >&2
    exit 1
  fi
done

for entry in "${RUNS[@]}"; do
  folder="${entry%%:*}"
  stem="${entry##*:}"
  run_dir="$INV_DIR/$folder"
  noisy_csv="$NOISE_DIR/${stem}.csv"
  noisy_meta="$NOISE_DIR/${stem}.meta"
  if [[ ! -f "$noisy_csv" ]]; then
    echo "ERROR: missing noisy data $noisy_csv (run noise_adding/add_noise_rfsdss.py first)" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  for f in "${TEMPLATE_FILES[@]}"; do
    cp -f "$V2_INV/$f" "$run_dir/$f"
  done
  cp -f "$noisy_csv" "$run_dir/measurement_data.csv"
  [[ -f "$noisy_meta" ]] && cp -f "$noisy_meta" "$run_dir/measurement_data.meta"
  echo "staged: $folder  (measurement_data.csv <- ${stem}.csv)"
done

echo ""
echo "Done. Each inv/rfsdss_<tag>/ is ready to run independently."
