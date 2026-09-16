# Three-parameter field inversion

The current user handover authorizes initial-model verification, two directional
gradient checks, and inversion only if those checks pass. The earlier proposal
and `AWAITING_QC_APPROVAL` marker remain as historical QC records; this note
records the subsequent instruction. Native QC files have not been regenerated.

## Input audit

- Fiberis DSS2D loading and the original 7500 < MD < 15000 ft median subtraction
  reproduce the crop within float32 rounding (maximum difference 4.515e-13
  strain; DSS2D loads into float64, whereas the existing cache used float32).
- Native observations remain 45 × 3318; negatives become zero before linear
  interpolation. The saved 45 × 130 array matches this interpolation exactly.
- The solver schedule matches V6 with only the previously established final
  timestamp adjustment to 384301.087267 s: 129 steps, cap 240, abort on failure.
- `prepare_inversion.py` derives its input from the successful corrected field
  forward run. It retains the physics, pressure function, mesh, geometry,
  tolerances, LU/MUMPS, and implicit Euler (MOOSE's existing default).
- All five zone values and layer boundaries match the authoritative depth map.
  Only A/B/C vary. The existing 200-layer adjoint gradient is summed into those
  three parameters. The two other zones remain -15.5; background remains -18.
- The positive strain auxiliary variable and its adjoint gate `1[s > 0]` are
  inherited from the existing field setup. Zero observations remain in the
  residual, and the initial frame is excluded. Each evaluation checks the
  reporter residuals against the objective and verifies all 130 output times.

## Necessary changes from the earlier proposal

The proposal used a layer-weighted L1 sum, giving A/B/C weights [1,1,3]. The user
handover explicitly specifies one term per optimized alpha. The new driver uses
weights [1,1,1] and omits constant terms from fixed zones. Beta remains the stored
5.049311379654719e-15, reference -18, delta 0.05. The existing objective scale
308267582.56113124 is applied to both data and regularization, and to the gradient.

The five-parameter driver is adapted locally to three parameters and the cropped
QC display. Existing synthetic and full-window scripts are unchanged. The two
central-difference checks retain h=0.02 and the prior 5%/1e-7 scaled absolute
acceptance rule; directions are the three active entries of the old directions.
Every check records its direction, derivatives, and pass status. A failed solve,
reporter check, or gradient check stops the run before L-BFGS-B.

## Run artifacts

`inversion/preflight.json` records inputs and SHA256 checksums. The reused initial
forward data are in `inversion/strain_initial_reused.csv`; the new initial
forward/adjoint result must agree before gradient checks proceed.

Run `prepare_inversion.py`, then `run_inversion.py --check-input`, then
`run_inversion.py` with the existing moose Python environment and Fiberis on
PYTHONPATH. Default MPI size is 20. The driver validates gradients before entering
L-BFGS-B (60 iterations, ftol=1e-7, gtol=1e-8, maxls=20).

Evaluation history and accepted iteration history are separate CSVs. `FAILED`
records an error; only `COMPLETE` indicates successful optimization and final QC.
Existing history prevents accidental overwrite on restart.

## User update: skip repeated validation

On 2026-09-15 the user stated that validation had already been performed and
requested no further validation. The first initial forward/adjoint evaluation
completed successfully; its objective and gradient are reused. The running first
finite-difference perturbation was cancelled by SIGTERM, with its outputs
archived under `inversion/cancelled_gradient_check_output`. The
`SKIP_GRADIENT_CHECK` marker records the instruction. The continuation starts
L-BFGS-B directly and does not claim the skipped checks passed.

## Persistent execution after tmux migration

The first optimization trial was interrupted when the original client exited
during migration to tmux. Its forward result existed, but its adjoint gradient
was incomplete. Artifacts are retained in `inversion/interrupted_20260915_1600`.
The complete initial evaluation is reused; only the interrupted trial is repeated.
The worker now runs in tmux session `rfsdss_14710_14725`, started with
`launch_inversion.sh`. Its persistent console log is `inversion/driver.log`.
No model, observations, solver settings, or optimization settings changed.
