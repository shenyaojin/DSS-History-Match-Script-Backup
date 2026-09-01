#!/usr/bin/env bash
# Stage the 3x3x3 sensitivity-grid run folders for the 2-parameter zonal
# inversion.
#
# Three factors, 3 levels each (full factorial = 27 runs):
#   A = real DSS background noise level   (bg1 < bg2 < bg3)
#   B = white (instrument) noise level    (w1  < w2  < w3)
#   C = zone-position offset [layers]     (s0=exact, s2=+2 layers, s4=+4 layers)
#
# A and B live in the DATA (9 noisy datasets noise_data/measurement_data_bg<i>_w<j>.csv,
# produced by noise_adding/add_noise_grid.py). C lives in the INVERSION (which
# layers are tied) and is passed at run time by run_all_grid.sh via the folder
# name suffix _s<N> -> ZONAL_MODE / ZONAL_PERT_SHIFT_LAYERS. So the forward model
# is unchanged and every C level reuses the same 9 datasets.
#
# Run folder name: bg<i>_w<j>_s<k>   (i,j in 1..3 ; k in 0,2,4)
#
# Re-running is safe: refreshes templates + measurement_data.csv, leaves any
# existing inv_output/ and *_zonal.* results in place.
#
# Usage:
#   bash setup_grid_runs.sh

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V6_DIR="$(dirname "$INV_DIR")"
NOISE_DIR="$V6_DIR/noise_data"
TEMPLATE_DIR="$INV_DIR/_template"

TEMPLATE_FILES=(
  "107_optimization_runner_zonal_L1.py"
  "optimize.i"
  "forward_and_adjoint.i"
  "plot_inversion_qc.py"
  "run_parameter_history_qc.py"
)

BG_LEVELS=(1 2 3)
W_LEVELS=(1 2 3)
SHIFTS=(0 2 4)

for f in "${TEMPLATE_FILES[@]}"; do
  [[ -f "$TEMPLATE_DIR/$f" ]] || { echo "ERROR: missing template $TEMPLATE_DIR/$f" >&2; exit 1; }
done

echo "template source : $TEMPLATE_DIR"
echo "noise data      : $NOISE_DIR"
echo ""

n=0
for i in "${BG_LEVELS[@]}"; do
  for j in "${W_LEVELS[@]}"; do
    dataset="$NOISE_DIR/measurement_data_bg${i}_w${j}.csv"
    if [[ ! -f "$dataset" ]]; then
      echo "ERROR: missing dataset $dataset (run noise_adding/add_noise_grid.py first)" >&2
      exit 1
    fi
    for k in "${SHIFTS[@]}"; do
      run_dir="$INV_DIR/bg${i}_w${j}_s${k}"
      mkdir -p "$run_dir"
      for f in "${TEMPLATE_FILES[@]}"; do
        cp -f "$TEMPLATE_DIR/$f" "$run_dir/$f"
      done
      cp -f "$dataset" "$run_dir/measurement_data.csv"
      [[ -f "${dataset%.csv}.meta" ]] && cp -f "${dataset%.csv}.meta" "$run_dir/measurement_data.meta"
      n=$((n + 1))
    done
    echo "staged bg${i}_w${j}_s{0,2,4}  (data <- measurement_data_bg${i}_w${j}.csv)"
  done
done

echo ""
echo "Done. Staged $n run folders (should be 27). Each inv/bg<i>_w<j>_s<k>/ is ready."
