# Proposed inversion parameters — awaiting QC approval

No inversion or new forward solve has been started. This directory contains only data preparation, QC figures and a proposal.

## Observations

- Requested depth: 14710–14725 ft. Actual native channel centers: 14710.064–14724.751 ft; 45 channels.
- Requested time: 0–400000 s. Actual data: 0–384301.087267 s; 3318 native times. The final unrecorded interval is hatched, not extrapolated.
- DSS start: 2021-08-04 17:07:09.036762. Source: `/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz`.
- Reuses the verified cache of that source: per-time median over 7500 < MD < 15000 ft was subtracted, then microstrain converted to strain (×1e-6). No further detrending, scaling or normalization.
- Positive half-wave: max(strain,0). 37.8012% of native samples become zero. These remain zero-valued observations, not missing or masked observations.
- Rectify native data first, then linearly interpolate to the template solver times. This ordering is explicit; it differs from the earlier full-window preview, which interpolated signed samples before rectifying.
- Inversion array: 45 channels × 130 times = 5850 records; t=0 shown for QC but excluded from transient data misfit (5805 active records). Equal sample weights, no dt weighting.
- QC color limits: signed −5 to +5 με; positive 0 to 3.5 με. All cropped native values fit within these limits.

## Permeability proposal

Crop the observations only. Keep all original zone boundaries and the full physical model. Optimize the three overlapping zones; fix the two outside zones at the confirmed initial values. Boundary-crossing zones retain their full widths.

| Original zone | Full model depth [ft] | Visible overlap [ft] | Initial α = log10(kxx / m²) | Initial kxx [m²] | Action |
|---|---|---|---|---|---|
| 1 | 14701.5–14704 | None | -15.5 | 3.16227766e-16 | Fixed |
| 2 | 14709.5–14712 | 14710–14712 | -15.5 | 3.16227766e-16 | Optimize |
| 3 | 14737.5–14739.5 | None | -15.5 | 3.16227766e-16 | Fixed |
| 4 | 14715–14717 | 14715–14717 | -16.5 | 3.16227766e-17 | Optimize |
| 5 | 14721–14727 | 14721–14725 | -16.5 | 3.16227766e-17 | Optimize |

- Active vector (depth order): θ = [−15.5, −16.5, −16.5]. Bounds: −18 ≤ θj ≤ −12, or 1e−18 ≤ kxx ≤ 1e−12 m².
- Background kxx = 1e−18 m². Positions and widths are fixed; this is not isotropic-permeability fitting. kyy and kzz retain the existing functions:
```text
  [func_kyy]
    type = ParsedFunction
    expression = '1e-18'
  []
  [func_zero]
    type = ParsedFunction
    expression = '1e-18'
  []
```

## Physics and source

- Domain 100 × 100 m; 200 × 200 elements, existing y edges fitted to original zone boundaries. y = 50 + (MD − 14725) × 0.3048 m.
- Injection x = 45 m; observation x = 69.384 m. Separation 80 ft is inherited from the old model, not a surveyed distance.
- E = 35 GPa; Poisson ratio 0.25; Biot coefficient 0.8; porosity 0.01.
- Fluid bulk modulus 2.2 GPa; viscosity 0.001 Pa·s; density 1000 kg/m³; gravity zero.
- Initial pressure perturbation and displacements zero. Pressure boundary uses timestamp-aligned production gauge pressure_g1.npz, minus its value at DSS t=0; same full pressure history as the successful 129-step forward solve.
- Existing boundary conditions retained: injection pressure Dirichlet; zero specified x-traction on left/right and y-traction on top/bottom; natural flow boundaries elsewhere.

## Solver and optimization

- V6 TimeSequenceStepper: 130 timestamps, 129 forward steps; last time adjusted to 384301.087267 s. Hard cap 240 forward steps; abort on solve failure. Adjoint replays the forward schedule.
- Implicit Euler; Newton; LU/MUMPS; l_tol = nl_abs_tol = nl_rel_tol = 0.001; l_max_its = 2000; nl_max_its = 200. 20 MPI ranks, one thread per rank.
- L-BFGS-B with adjoint gradient; maxiter=60, ftol=1e−7, gtol=1e−8, maxls=20. Record objective evaluations and accepted iterations separately.
- Before optimization: compare the three-parameter adjoint directional derivative with central differences (h=0.02 in log10 permeability; two directions, 5% relative tolerance). Stop if validation fails; no automatic tolerance or time-step changes.

## Misfit and L1 regularization

Use the same positive-half-wave observation operator for the simulation: r = max(simulated strain_yy,0) − rectified observation. The gradient has gate 1[simulated strain_yy > 0] (zero at the kink).

`J = S * (0.5 * sum(r²) + beta * sum_layers(sqrt((alpha + 18)² + 0.05²) - 0.05))`

- This is squared data misfit plus smoothed L1 regularization of log-permeability relative to −18; not an L1 data residual.
- beta = 5.049311379655e-15; S = 3.082675825611e+08. Recomputed for this cropped 130-time dataset using the existing rule: beta = 2e−11 × sum(cropped signed sampled data²) / sum(V6 clean data²); S = 1 / sum(cropped signed sampled data²).
- L1 smoothing δ=0.05; active zone layer weights [1,1,3]. The two outside zones retain fixed initial α and contribute only constant regularization. Background deviation is zero.

## Pending approval

Approve the positive-only crop and this proposed three-parameter setup before launching inversion. No solver inputs or launchers for the new inversion have been generated yet. Existing full-window simulations and old inversions remain unchanged.
