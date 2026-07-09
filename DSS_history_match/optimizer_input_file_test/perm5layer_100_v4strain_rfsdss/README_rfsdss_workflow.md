# v4 strain L1 realistic-RFS-DSS-noise workflow

This folder runs the **v2 L1-regularized strain inversion** on copies of the
synthetic observation corrupted with a *realistic* Rayleigh-frequency-shift DSS
noise model, to see how the L1 result degrades under noise that actually looks
like field RFS-DSS data. It mirrors the architecture of
`perm5layer_100_v3strain_noise` (the earlier peak / median white-noise study)
but is fully self-contained: its own `data/`, `noise_adding/`, and `inv/`.

The inversion code, initial model, zone masks, and L1 settings are copied
verbatim from `perm5layer_100_v2strain/inv`. **Only the observation data
changes** — nothing in the optimizer is retuned.

## Why a new noise model

The v3 study added a single spatially/temporally **uniform** i.i.d. Gaussian std
to every point (scaled by the global peak or the median channel peak). Real
RFS-DSS strain noise does not look like that. This model reproduces the four
structural features actually seen in the data:

1. **Per-channel noise floor** — an *absolute* strain floor (nε), different
   channel-to-channel (lognormal spread), that does **not** scale with the local
   signal amplitude.
2. **Per-channel low-frequency drift** — a smooth, time-correlated wander per
   channel (not white).
3. **Common-mode drift** — a single low-frequency series shared by all channels
   (laser / instrument common mode).
4. **Strain-driven spikes** — sparse, heavy-tailed, random-sign spikes whose
   probability is proportional to the local strain magnitude times a spatial
   weight that grows away from the low-strain array centre. So spikes are rare
   while the fibre strain is low (early time, quiet centre) and become frequent
   where/when the fibre is pulled hard (the strain lobes at y≈±18, late time).

## Noise model

```
d_noisy(c,t) = d_clean(c,t)
             + floor_c * N(0,1)      # per-channel white floor (absolute nε)
             + drift_c(t)            # per-channel low-frequency drift
             + common(t)             # shared common-mode drift
             + spike(c,t)            # strain-driven sparse spikes
```

All magnitudes are anchored to `REF = median over receivers of each channel's
peak |strain|` = 2.262e-05 (the same reference as the v3 median family), so the
levels are directly comparable. A single knob `floor_frac = floor mean / REF`
sets each level; the other three components scale with it by fixed ratios
(`drift = 0.75·floor`, `common = 0.5·floor`, `spike scale = 7.5·floor`). The
spike probability pattern `p(t,c)` depends only on the clean signal and geometry,
so it is identical across levels — only the spike amplitude scales.

| level (tag) | floor mean | total noise std | SNR (REF/std) |
|-------------|-----------:|----------------:|--------------:|
| 0.5% (0p5pct) | 113 nε | 277 nε | ~82 |
| 1%   (1pct)   | 226 nε | 555 nε | ~41 |
| 2%   (2pct)   | 452 nε | 1117 nε | ~20 |
| 5%   (5pct)   | 1131 nε | 2684 nε | ~8 |
| 10%  (10pct)  | 2262 nε | 5621 nε | ~4 |

See `noise_adding/rfsdss_noise_summary.csv` for realized statistics and
`noise_adding/rfsdss_noise_qc_<tag>.png` / `rfsdss_noise_levels_summary.png` for
QC figures. All noise parameters live at the top of
`noise_adding/add_noise_rfsdss.py`.

## Layout

```
data/                         clean observation (obs_strain_yy.csv), unchanged
noise_adding/
  add_noise_rfsdss.py         generates the 5 rfsdss levels + QC + summary
  measurement_data_rfsdss_<tag>.csv  noisy observation per level
                                     (tag = 0p5pct,1pct,2pct,5pct,10pct)
  rfsdss_noise_summary.csv    requested vs realized noise statistics
  rfsdss_noise_qc_<tag>.png   per-level 8-panel QC figure
  rfsdss_noise_levels_summary.png   cross-level comparison
inv/
  setup_rfsdss_runs.sh        stages one self-contained run folder per level
  run_all_rfsdss.sh           runs the L1 inversion for each run folder
  run_qc_rfsdss.sh            forward-QC the final alpha for each run folder
  compare_rfsdss_results.py   comparison table + alpha overlay (clean + rfsdss)
  rfsdss_<tag>/               L1 run folder (staged by setup_rfsdss_runs.sh)
    106_optimization_runner_L1.py, optimize.i, forward_and_adjoint.i,
    plot_inversion_qc.py, run_parameter_history_qc.py
    measurement_data.csv      <- the noisy observation for this dataset
    (after running: parameter_history_L1.csv, optimized_alphas_L1.txt,
     objective_history_L1.csv, inv_output/, qc_strain_L1_final.png, ...)
```

Each run folder is fully independent — the runner uses its own directory as the
working directory, so the five runs never overwrite each other. The drivers
auto-discover all `rfsdss_*` folders.

## How to run (do this OUTSIDE the Codex sandbox — MOOSE needs MPI sockets)

```bash
cd scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v4strain_rfsdss

# 1. (re)generate the noisy datasets + QC figures
python noise_adding/add_noise_rfsdss.py

# 2. stage one run folder per level
bash inv/setup_rfsdss_runs.sh

# 3. run the L1 inversion for every level (sequential; long)
bash inv/run_all_rfsdss.sh
#    or a subset:  bash inv/run_all_rfsdss.sh rfsdss_2pct rfsdss_5pct

# 4. forward-QC each final alpha
bash inv/run_qc_rfsdss.sh

# 5. compare against truth (clean v2 + all rfsdss levels)
python inv/compare_rfsdss_results.py
```
