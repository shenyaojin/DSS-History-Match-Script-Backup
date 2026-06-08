# v3 strain L1 noise-sensitivity workflow

This folder runs the **v2 L1-regularized strain inversion** on noisy copies of
the synthetic observation, at four noise levels (0.5%, 1%, 2%, 5%), to see how
the L1 result degrades with measurement noise.

The inversion code, initial model, zone masks, and L1 settings are copied
verbatim from `perm5layer_100_v2strain/inv`. **Only the observation data
changes** between this study and v2 — nothing in the optimizer is retuned.

## Noise model

For each level `p`, i.i.d. Gaussian noise with a single absolute standard
deviation is added to every point of the clean `measurement_values` column:

```
std_p = p * max_i |d_i|          # max|d| = 3.589e-05 (peak strain)
d_noisy_i = d_i + N(0, std_p^2)
```

| level | std        | peak SNR (max\|d\|/std) | seed     |
|-------|------------|-------------------------|----------|
| 0.5%  | 1.795e-07  | 200                     | 20260608 |
| 1%    | 3.589e-07  | 100                     | 20260609 |
| 2%    | 7.178e-07  | 50                      | 20260610 |
| 5%    | 1.795e-06  | 20                      | 20260611 |

Noise is added to **all** points (including the t=0 baseline), so the
perturbation is independent of the local signal amplitude. Seeds are fixed per
level, so every dataset is reproducible and the realizations are independent
across levels. See `noise_adding/noise_summary.csv` for realized statistics.

## Layout

```
data/                         clean observation (obs_strain_yy.csv), unchanged
noise_adding/
  add_noise.py                generates the noisy datasets + summary
  measurement_data_clean.csv  labeled copy of the clean observation
  measurement_data_noise_<tag>.csv   one per level (tag = 0p5pct,1pct,2pct,5pct)
  noise_summary.csv           requested vs realized noise statistics
inv/
  setup_noise_runs.sh         stages one self-contained run folder per level
  run_all_noise.sh            runs the L1 inversion for each level
  run_qc_noise.sh             forward-QC the final alpha for each level
  compare_noise_results.py    comparison table + alpha overlay across levels
  noise_<tag>/                self-contained L1 run folder (per level)
    106_optimization_runner_L1.py, optimize.i, forward_and_adjoint.i,
    plot_inversion_qc.py, run_parameter_history_qc.py
    measurement_data.csv      <- the noisy observation for this level
    (after running: parameter_history_L1.csv, optimized_alphas_L1.txt,
     objective_history_L1.csv, inv_output/, qc_strain_L1_final.png, ...)
```

Each `noise_<tag>/` is fully independent — the runner uses its own directory as
the working directory, so the four levels never overwrite each other.

## How to run (do this OUTSIDE the Codex sandbox — MOOSE needs MPI sockets)

```bash
cd scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v3strain_noise

# 1. (Re)generate noisy datasets — already done, only needed if you change levels
~/miniforge/envs/moose/bin/python noise_adding/add_noise.py

# 2. (Re)stage the per-level run folders — already done, idempotent
bash inv/setup_noise_runs.sh

# 3. Run the L1 inversions (all four, sequential; long)
bash inv/run_all_noise.sh
#    or a subset:
bash inv/run_all_noise.sh 1pct 5pct

# 4. Forward-QC the final alpha of each level (proves fit to the noisy strain)
bash inv/run_qc_noise.sh

# 5. Build the comparison table + overlay plot
~/miniforge/envs/moose/bin/python inv/compare_noise_results.py
```

## What to look at

- `inv/noise_comparison_summary.csv` — zone means (low SRV / fracture /
  matrix), max alpha error, and relative-L2 errors for every level, with the
  `clean (v2)` baseline and `truth` for reference.
- `inv/noise_alpha_overlay.png` — inverted alpha profiles for all levels on one
  axis vs the synthetic truth.
- `inv/noise_<tag>/qc_strain_L1_final.png` — per-level strain fit QC. The
  "Measured strain_yy" panel here is the **noisy** observation, so the residual
  panel shows how the L1 model fits noisy data without overfitting it.
- `inv/noise_<tag>/objective_history_L1.csv` — note that the raw data misfit can
  no longer reach ~1e-12 as in the clean case; it floors near the noise energy.

### Synthetic truth (for reference)

```
background / matrix : -18
low SRV             : -15
fracture            : log10(3e-15) = -14.522879
```

Clean (v2) L1 baseline: low SRV -15.016, fracture -14.567, matrix -17.999,
rel-L2 (all) 0.001526, rel-L2 (free window) 0.002196.
