#!/usr/bin/env bash
# Stage one self-contained L1 inversion run folder per noise dataset.
#
# Two families are staged:
#   peak    -> inv/noise_<tag>/      from measurement_data_noise_<tag>.csv
#   median  -> inv/mednoise_<tag>/   from measurement_data_mednoise_<tag>.csv
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

# run_folder : noisy_csv_basename  (in noise_adding/)
RUNS=(
  "noise_0p5pct:measurement_data_noise_0p5pct"
  "noise_1pct:measurement_data_noise_1pct"
  "noise_2pct:measurement_data_noise_2pct"
  "noise_5pct:measurement_data_noise_5pct"
  "mednoise_1pct:measurement_data_mednoise_1pct"
  "mednoise_2pct:measurement_data_mednoise_2pct"
  "mednoise_5pct:measurement_data_mednoise_5pct"
  "mednoise_10pct:measurement_data_mednoise_10pct"
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
    echo "ERROR: missing noisy data $noisy_csv (run noise_adding/add_noise.py first)" >&2
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
echo "Done. Each inv/{noise,mednoise}_<tag>/ is ready to run independently."
