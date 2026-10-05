# 14376-ft permeability inversion with global attenuation

User-approved follow-up to `../real_rfsdss_fixed_zones_14373_14379_400000s_bg21`.
The previous two-parameter results are preserved.

## Model and objective

- Same 14373–14379-ft crop, 18 channels; fixed high-permeability zone
  14375.4423828125–14376.7802734375 ft.
- Three independent parameters: alpha_xx, alpha_yy, C. Initial permeability
  amplitudes both -17 (1e-17 m²); permeability bounds [-21, -12].
- C is a single global attenuation, shared by all channels and times.
  Bounds [1e-6, 1]; 1e-6 is a numerical positivity floor for division by C,
  not a measured coupling constraint. Any final bound hit is explicitly reported.
- Bootstrap at C=1, then set initial C to clipped dot(s,d)/dot(s,s), excluding t=0.
  C is subsequently independently optimized, not eliminated analytically.
- s=max(physical simulated strain,0); d is the unchanged observation cache,
  rectified at native times before interpolation. Zeros remain observations.
- J_data = 0.5 sum_{t>0}(C*s-d)^2.
- Original smoothed L1 applies ONLY to alpha_xx and alpha_yy: reference -21,
  delta=0.05, beta=5.049311379654719e-15. No penalty on C.
- Original objective scale 308267582.56113124 multiplies both data and L1 terms.
- Original L-BFGS-B: maxiter=60, ftol=1e-7, gtol=1e-8, maxls=20.

## Reuse of validated adjoint

The MOOSE physical input is byte-identical to the previous case except for
the observation filename. Each evaluation writes d/C to a private
`solver_measurement_data.csv`. The canonical `measurement_data.csv` never changes.
For fixed C, MOOSE returns F=0.5 sum(s-d/C)^2 and its permeability derivatives.
The driver returns:

```
J_data = C² F
dJ/dalpha = C² dF/dalpha
dJ/dC = sum_{t>0} (C*s-d)*s
```

This retains the existing positive-half-wave derivative mask in the adjoint.
No physical strain, pressure source, storage law, material, or boundary condition
is multiplied by C. No MOOSE reporter parameter is added for C.

## Checks and execution

The preparation asserts unchanged physical input, 130 output times / 129 steps,
and the 200×216 mesh. Retain 240 maximum steps, implicit Euler, LU/MUMPS,
20 MPI ranks, all three solver tolerances 0.001, iteration limits 2000/200.

1. `prepare_inversion.py` refuses an existing run directory.
2. `test_coupling_gradient.py` checks the complete evaluation/serialization
   chain against a known nonlinear surrogate, including rectification, t=0
   exclusion, C gradients and permeability-only regularization. It is not a
   substitute for MOOSE finite-difference checks.
3. `run_inversion.py --check-input` checks MOOSE input.
4. `launch_inversion.sh` runs bootstrap, initialized forward/adjoint,
   two permeability central FD checks (h=0.02), the C finite difference, then
   optimization only if all pass. C's check needs no additional PDE solve
   because C is absent from the physical equations.

Original FD acceptance rule retained: absolute scaled derivative difference
<=1e-7 or relative difference <=5%. No automatic relaxation on failure.

## Results

`strain.csv` / `strain_signed.csv` within each evaluation contain C-scaled
predictions with original observations; `strain_physical.csv` and
`strain_physical_signed.csv` preserve unscaled physical simulation values.
Each accepted iteration saves a QC image (waterfalls left, channel fits right),
all three parameters, data/L1 objectives, and RMS in physical strain units.
`coupling_initialization.json`, `gradient_check.json`, `status.json`, and
`driver.log` provide progress. `optimizer_result.json` reports termination and
whether C ended at a bound. Duplicate launches are refused.
