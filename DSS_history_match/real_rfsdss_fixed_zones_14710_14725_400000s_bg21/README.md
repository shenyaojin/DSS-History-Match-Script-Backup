# Isotropic background comparison: alpha = -21

## Current status

The pilot has stopped for user review after **2 accepted iterations and 11
objective evaluations**, by the unchanged L-BFGS-B relative objective reduction
criterion (`ftol=1e-7`). The limit was five accepted iterations; no continuation
has been launched. Final alpha values are approximately
`[-20.98548337, -20.99295243, -20.99125422]`.

Residual RMS fell from 13.8304 to 0.5176 microstrain, but the final prediction
improves the data objective over an all-zero prediction by only 0.71%.
Simulation peak is 0.08275 microstrain versus 2.87353 observed. This is an
underprediction of the field signal, not evidence of a successful history match.
See [pilot results](pilot_5iter/RESULTS.md) and
[the fitting figure](pilot_5iter/qc_after_pilot.png).

Source scripts, configuration snapshots, final QC, and compact result histories
are versioned. Solver working files, evaluation archives, and the cancelled
attempt remain on disk and are ignored by Git. Run markers are local state and
are not versioned. Preparing a new run still requires the original field data,
baseline preparation, and the existing local MOOSE/Fiberis environment.

## Initial comparison and provenance

Background kxx outside the five predefined zones and global kyy/kzz are reduced
from 1e-18 to 1e-21 m², per the user's clarified instruction. Zone anomalies
remain in kxx only. All five zone kxx initial values, data, mesh, pressure source,
mechanics, 129-step schedule, 240-step cap, 0.001 tolerances, and 20 MPI ranks
are retained. The original completed case is unchanged.

This launcher computes the new initial forward/adjoint response only, so its
data gradient is available for a subsequent inversion. It does not run finite
differences or optimize. The earlier background-18 response/gradient is not
reused. In this initial comparison's historical configuration, bounds are
[-18,-12] and the L1 reference is -18; delta remains 0.05 and beta remains
5.049311379654719e-15. Background kxx and the L1 reference are separate settings.

The new driver reads the regularization reference and delta from configuration
instead of the old hard-coded -18/.05. This preserves the current formula while
allowing a separately requested reference change. QC uses the actual background.

The earlier kxx-only comparison was explicitly superseded and cancelled. Its
inputs and partial outputs are preserved in `cancelled_kxx_only/`.

## User decision: five-iteration pilot

The user subsequently set both the active lower bound and L1 reference to -21
and requested at most five accepted L-BFGS-B iterations before reviewing results.
This continuation is isolated in `pilot_5iter/`, with its own config and histories.
The initial comparison files retain their original external regularization
metadata for provenance; only their identical PDE data objective/gradient are
reused. The pilot recomputes the regularization with reference -21. See
`pilot_5iter/README.md`. No continuation beyond this pilot is authorized yet.
