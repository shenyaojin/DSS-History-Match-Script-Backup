# Planned and implemented compatibility contract (2026-09-30)

The implementation is `fiberis.moose.numerical_ddm`; physical parameters are SI.
This contract was established before implementation. Verification status is in RESULTS.md.

| DDMPY surface | Numerical scope |
|---|---|
| Element(length,height,width,S1=0,S2=0), set_global_coors | Same argument meanings and negative stored jumps; mu is Poisson ratio |
| FractureZ.set_fracture_elements, set_monitor_wells, calculate | Same workflow; calculate requires separately configured numerical backend |
| FractureGlobal, set_global_coors(strike,dip,fracx,fracy,fracz) | Same rotation convention; exact 90 degrees by default, explicit legacy option |
| get_ellipse_fracture | Same discrete rectangular strips, no smooth ellipse substitution |
| Well().set_well_path, u1/u2/u3, disp, strain, mds | Same access; corrected simultaneous tangent normalization; endpoint repetition |
| Global fracture well coordinates / displacement components | Numerical wells and u1/u2/u3 stay in the supplied global frame. Original FractureGlobal.monitor_wells stores local coordinates/components; original global_monitor_wells only receives strain. Convert a numerical row-vector displacement with U_local = U_global @ R.T if that original local representation is needed. |
| gauge_length | Small-strain average along arclength, clipped windows; documented difference from original point-count average/chord length |
| DynamicFracture.set_fractures, set_monitor_wells, calculate, gather_strain_data | Independent actual FE snapshots; native fibeRIS Data2D results |
| RectangularFracture.define_LHW / define_asymmetric_grow | Width assignment corrected; scalar or array histories |
| Fracture u1(x,y,z), plotting helpers, WellZ, arbitrary DDMPY mutators | Not promised; use sample(points) / explicit Well |

No production import of DDMPY or analytic displacement formulas is permitted. Analytic results are confined to validation scripts.

Implemented additions after actual baseline runs:

* `from_ddmpy(obj, config, length_scale=1)` imports original static or dynamic objects; `length_scale=.3048` explicitly converts feet to SI, including displacements, coordinates, MD and gauge. Original sample-mean gauge and external MD are preserved. The 0930 inclined-well adapter was checked against the untouched original postprocessor.
* `FractureZ.sample_history(points,times,factors)` solves fixed-geometry, common-factor histories at all requested times; the initial factor must be zero. The optional combined-app ConstantJacobianProblem caches only constant stiffness.
* `NumericalConfig.local_refinements` controls per-axis fine windows; `core_extents` permits distinct x/y/z cores. These are numerical settings, not fracture geometry.
* General changing geometry/independent component histories continue to use genuinely separate `DynamicFracture` static solves.
* `Data2D` field access (`data`, `taxis`, `daxis`, coordinates) is supported. Original JIN_pylib plotting and `take_time_diff()` mutation methods are not promised on the fibeRIS container; use `gather_strain_rate_data()` for the corrected point derivative.

Verification boundaries:

* Actual FE evidence is enumerated in `output/numerical_ddm/RESULTS.md`; a completed result and solver log are required to call a case verified. The optional 30-frame notebook run is listed separately from its three-frame pilots.
* Implemented but without a dedicated FE accuracy study: `Model` superposition of differently oriented fracture groups; `FIRST` interpolation; arbitrary independent mixed-mode histories; strongly overlapping bands; independently varied per-axis core extents. These must not inherit the reported accuracy by assumption.
* Feet-to-SI import and external MD/sample-mean gauge have been checked against original 0930 postprocessing, but no separate field-data inversion is claimed.
* Not implemented: full original plotting/module hierarchy, pressure-to-opening inversion, contact/friction interaction laws, dynamics, poroelasticity, and a smooth-ellipse representation. Near-tip stresses and pointwise traction-jump convergence have not been validated.
