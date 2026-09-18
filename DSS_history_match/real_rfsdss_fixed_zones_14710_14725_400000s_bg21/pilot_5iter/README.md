# Five-iteration pilot: stop for user review

User update: background in all three directions is 1e-21 m²; active-zone lower
bound and L1 reference both become -21. Beta=5.049311379654719e-15 and delta=0.05
remain unchanged. Three active initial alpha values remain [-15.5,-16.5,-16.5].
Both outside zones remain fixed at -15.5. Geometry, source, mechanics, data,
130 output times / 129 forward steps, cap 240, and 0.001 tolerances are unchanged.

The worker first waits for the already-running background-21 initial solve.
It reuses only the identical PDE's data objective and data gradient, then
recalculates the regularization and its gradient with reference -21. It does
not reuse the previous reference-18 total gradient. No finite differences run.

L-BFGS-B: maxiter=5, ftol=1e-7, gtol=1e-8, maxls=20, bounds=[-21,-12].
Five means accepted iterations, excluding iteration zero and line-search
trials. Early convergence stops earlier. Reaching the requested iteration cap
is an intentional pilot stop, not a solver failure or a claim of convergence.

Every evaluation retains its 45 × 130 observation report and gradient under
`evaluations/`. Each accepted iteration has a copy under `iterations/` and a
`qc_iter_NN.png` figure. `accepted_iteration_history.csv` and
`objective_history.csv` separately record accepted steps and all evaluations.
The final pilot plot uses the saved matching evaluation; no extra PDE solve is
needed solely to redraw QC. `PILOT_COMPLETE` and `PAUSED_FOR_REVIEW` mean the
requested pilot is finished. Further optimization requires the user's decision.

Run in dedicated tmux session `rfsdss_bg21_pilot5` using `launch_pilot.sh`.
Console output is persisted in `driver.log`. No inputs/results in the earlier
background-18 case or the initial comparison directory are overwritten.
