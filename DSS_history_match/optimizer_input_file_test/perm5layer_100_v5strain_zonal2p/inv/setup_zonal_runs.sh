#!/usr/bin/env bash
# Stage one self-contained 2-parameter zonal inversion run folder per
# (zone-variant, noise-case).
#
# Two zone variants x 14 noise cases = 28 run folders:
#   inv/exact_<case>/   zones exactly match the synthetic truth
#   inv/pert_<case>/    zone y-windows shifted (mimics DSS zone-read error)
#   <case> in {clean, median_{1,2,5,10}pct, peak_{0p5,1,2,5}pct,
#              rfsdss_{0p5,1,2,5,10}pct}
#
# The forward/adjoint MOOSE model is UNCHANGED; only the outer driver differs
# (107_optimization_runner_zonal_L1.py inverts 2 scalars instead of 100 layers).
# Each run folder gets its own copy of the template set plus the noisy
# observation renamed to measurement_data.csv (the name optimize.i expects).
#
# The exact/pert choice is passed at RUN time by run_all_zonal.sh via the
# ZONAL_MODE env var (derived from the folder-name prefix), so a single runner
# template serves both variants.
#
# Re-running is safe: refreshes templates + measurement_data.csv, leaves any
# existing inv_output/ and *_zonal.* / *_L1.* results in place.
#
# Usage:
#   bash setup_zonal_runs.sh

set -euo pipefail

INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V5_DIR="$(dirname "$INV_DIR")"
NOISE_DIR="$V5_DIR/noise_data"
TEMPLATE_DIR="$INV_DIR/_template"

TEMPLATE_FILES=(
  "107_optimization_runner_zonal_L1.py"
  "optimize.i"
  "forward_and_adjoint.i"
  "plot_inversion_qc.py"
  "run_parameter_history_qc.py"
)

# noise cases: <case> == <family>[_<tag>]; measurement_data_<case>.csv in noise_data/
CASES=(
  "clean"
  "median_1pct" "median_2pct" "median_5pct" "median_10pct"
  "peak_0p5pct" "peak_1pct" "peak_2pct" "peak_5pct"
  "rfsdss_0p5pct" "rfsdss_1pct" "rfsdss_2pct" "rfsdss_5pct" "rfsdss_10pct"
)
VARIANTS=("exact" "pert")

echo "template source : $TEMPLATE_DIR"
echo "noise data      : $NOISE_DIR"
echo ""

for f in "${TEMPLATE_FILES[@]}"; do
  if [[ ! -f "$TEMPLATE_DIR/$f" ]]; then
    echo "ERROR: missing template file $TEMPLATE_DIR/$f" >&2
    exit 1
  fi
done

n=0
for variant in "${VARIANTS[@]}"; do
  for case in "${CASES[@]}"; do
    noisy_csv="$NOISE_DIR/measurement_data_${case}.csv"
    noisy_meta="$NOISE_DIR/measurement_data_${case}.meta"
    if [[ ! -f "$noisy_csv" ]]; then
      echo "ERROR: missing noisy data $noisy_csv" >&2
      exit 1
    fi
    run_dir="$INV_DIR/${variant}_${case}"
    mkdir -p "$run_dir"
    for f in "${TEMPLATE_FILES[@]}"; do
      cp -f "$TEMPLATE_DIR/$f" "$run_dir/$f"
    done
    cp -f "$noisy_csv" "$run_dir/measurement_data.csv"
    [[ -f "$noisy_meta" ]] && cp -f "$noisy_meta" "$run_dir/measurement_data.meta"
    echo "ZONAL_MODE=${variant}" > "$run_dir/zonal.env"
    n=$((n + 1))
    echo "staged: ${variant}_${case}"
  done
done

echo ""
echo "Done. Staged $n run folders (exact_* + pert_*). Each inv/<variant>_<case>/ is ready."
