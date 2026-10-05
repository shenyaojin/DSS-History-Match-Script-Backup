# Permeability Inversion from DSS: Theoretical Framework — figure package

Four main figures and one supplementary figure, prepared exclusively from completed numerical results. No MOOSE, forward, adjoint, or optimization runs are launched by the plotting script. English captions and Chinese interpretation notes accompany the figures.

Start with the [five-page figure review PDF](figure_review.pdf), which places each figure above its English caption. Use the individual PDFs below for manuscript layout.

## Figures

| Figure | Purpose | PDF | PNG preview |
|---|---|---|---|
| 1 | Adjoint-gradient verification | [PDF](figures/fig01_adjoint_verification.pdf) | [Preview](figures/fig01_adjoint_verification_preview.png) |
| 2 | Noise-free reconstruction | [PDF](figures/fig02_clean_reconstruction.pdf) | [Preview](figures/fig02_clean_reconstruction_preview.png) |
| 3 | Similar data fits, contrasting model recovery | [PDF](figures/fig03_fit_vs_recovery.pdf) | [Preview](figures/fig03_fit_vs_recovery_preview.png) |
| 4 | Noise and parameterization sensitivity | [PDF](figures/fig04_noise_sensitivity.pdf) | [Preview](figures/fig04_noise_sensitivity_preview.png) |
| S1 | Effect of zone-location error | [PDF](figures/figS01_zone_mislocation.pdf) | [Preview](figures/figS01_zone_mislocation_preview.png) |

Each figure is also available as SVG and a 600 dpi PNG. Figure widths are 180 mm; exact heights and color limits are recorded in `data/figure_metadata.json`. Text and lines remain vector objects in PDF/SVG; time–position heatmaps are rasterized at the export resolution to keep file size manageable. SVG text uses DejaVu Sans; PDF fonts are embedded. Keep the original figure dimensions when assessing text readability. No target journal has yet been specified.

## Text and source data

- [English captions](captions_en.md): panel descriptions, numerical results, metric definitions, and interpretive limits.
- [中文解读与论文叙事建议](interpretation_zh.md): each figure's question, observations, source cases, and appropriate conclusions.
- [Case metrics](data/case_metrics.csv): objectives, model RMSE, zone means, actual noise RMS, and available QC residual metrics.
- [Permeability profiles](data/alpha_profiles.csv): true, initial, and selected final profiles.
- [Validation checks](data/validation.json): agreement between archived final parameters, QC parameters, objectives, and paired observations.
- [Source manifest](data/source_manifest.json): original source paths, roles, file sizes, and SHA256 hashes.
- Compressed forward-response tables in `data/`: reusable plotting data from the selected completed QC solves.
- [Discrete verification record](data/adjoint_console_record.txt): explicit transcription of the previously completed console output; plotted at its printed precision. The plotting script does not rerun that verification.

## Reproduce

From the repository workspace root, using the existing Python environment:

```bash
env MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg \
  /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python \
  scripts/DSS_history_match/paper_theoretical_framework/make_figures.py
```

Requirements: NumPy, pandas, and Matplotlib, already present in the environment used to create these figures. No additional installation is required. Outputs within this package are regenerated; original numerical results are read only. Input assertions stop rendering if essential parameter, observation, or objective consistency checks fail.

## Selection decisions

- The primary noisy-response comparison uses the completed nominal 10% cases because both have complete final forward QC records. The nominal 5% zonal reporter contains an all-zero simulated column, so that case is used only for its saved parameter and objective results in Figure 4.
- Figure 4 is rebuilt directly from per-case final results. The older v5 summary CSV contains only a truth row and is not used.
- All fitting metrics exclude t = 0, retain all later samples, and use equal sample weights. Time coordinates in the heatmaps remain nonuniform.
- Figure 3 uses a documented common 99th-percentile residual display limit. No samples are removed from the numerical metrics. Other heatmaps use untruncated limits.
- Figure S1 shows prediction error against the existing clean synthetic response to expose coherent geometry error. It is explicitly distinguished from the noisy-data fitting residual.
- These are controlled synthetic demonstrations with seeded initial anomalies and fixed background/transverse permeability. Single realizations per noise level do not establish statistical confidence bounds, universal thresholds, global optima, or blind localization performance.
