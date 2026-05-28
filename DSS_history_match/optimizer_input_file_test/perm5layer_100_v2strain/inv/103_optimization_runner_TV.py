# L-BFGS-B outer driver for the strain_yy permeability inversion --- TV variant.
#
# Differences from 102_optimization_runner.py:
#   - No Tikhonov smoothness Sum(Delta alpha)^2 (it can't distinguish a block
#     from spikes -- both pay nearly the same cost in squared diffs).
#   - No prior anchor Sum(alpha - alpha_prior)^2 (it biases toward sparsity,
#     which prefers the spike pattern; bounds already clamp inactive layers).
#   - Replaced with TOTAL VARIATION:  J_TV(alpha) = beta_TV * Sum |Delta alpha|
#     smoothed with the Huber-like surrogate  |x| ~ sqrt(x^2 + delta^2) - delta
#     so L-BFGS-B has a differentiable gradient everywhere.
#
# Why TV: it pays the same for one big jump as for many small ones, so it
# strongly prefers a block (2 edges of magnitude 3) over spikes (~4 ramps).
#
# Run as:
#   python 103_optimization_runner_TV.py

import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner


# --- Paths --------------------------------------------------------------------
WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
OUTPUT_DIR = os.path.join(WORKDIR, "inv_output")

# --- Objective / gradient scaling for L-BFGS-B --------------------------------
SCALE_FACTOR = 1e6      # maps the O(1e-6) misfit to O(1)
BASELINE_OBJ = 0.0

# --- TV regularization --------------------------------------------------------
# J_TV(alpha) = BETA_TV * Sum sqrt( (Delta alpha)^2 + DELTA_TV^2 )
#
# BETA_TV: TV at the truth block is roughly 2 edges * 3 + 2 edges * 3.5 ~ 13.
# With 1e-7 we put the TV term at ~ 1.3e-6, comparable to the initial data
# misfit (~ 3.26e-6) and well above the converged misfit (~ 1e-8).  Adjust
# downward if the optimizer over-smooths (broad fuzzy bands), upward if spikes
# survive.
#
# DELTA_TV: Huber smoothing scale.  Must be << typical jump magnitude (~3
# in alpha-space here) so the surrogate hugs |x|, but large enough to keep
# gradients well-behaved near small differences.  0.05 is a good default.
BETA_TV = 1e-7
DELTA_TV = 0.05

# --- Adjoint gradient correction ---------------------------------------------
# The C++ VPP PorousFlowOptimizationAnisotropicDiffusionInnerProduct applies
# rho/mu at each qp, so no Python-side scaling is needed.
GRAD_CORRECTION = 1.0

TOTAL_LAYERS = 200

print(f"Working directory : {WORKDIR}")
print(f"MOOSE cwd         : {OUTPUT_DIR}")
print(f"Regularization    : TV only, beta_TV = {BETA_TV}, delta = {DELTA_TV}")

# --- Sanity check -------------------------------------------------------------
with open(INPUT_FILE, "r") as f:
    base_moose_content = f.read()
if "measurement_data.csv" not in base_moose_content:
    raise RuntimeError(f"{INPUT_FILE} does not reference measurement_data.csv.")
fwd_path = os.path.join(WORKDIR, "forward_and_adjoint.i")
with open(fwd_path, "r") as f:
    fwd_content = f.read()
if "variable = 'strain_yy'" not in fwd_content:
    raise RuntimeError(
        f"{fwd_path} does not declare OptimizationData.variable='strain_yy'. "
        f"Refusing to run: this script assumes the strain_yy observation channel."
    )

runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec",
)

# --- History files (separate from the Tikhonov run so we can compare) --------
HISTORY_FILE = os.path.join(WORKDIR, "parameter_history_TV.csv")
GRADIENT_HISTORY_FILE = os.path.join(WORKDIR, "gradient_history_TV.csv")
OBJECTIVE_HISTORY_FILE = os.path.join(WORKDIR, "objective_history_TV.csv")
CHECKPOINT_FILE = os.path.join(WORKDIR, "checkpoint_alpha_TV.npy")
for p in (HISTORY_FILE, GRADIENT_HISTORY_FILE, OBJECTIVE_HISTORY_FILE):
    if os.path.exists(p):
        os.remove(p)
with open(OBJECTIVE_HISTORY_FILE, "w") as f:
    f.write("iter,obj_raw,reg_tv,obj_total,obj_scaled,grad_norm_scaled\n")

iteration_count = 0


def tv_obj_and_grad(x):
    """Smooth-Huber TV: J = BETA_TV * Sum sqrt((Delta alpha)^2 + DELTA_TV^2).
    Returns (J_tv, dJ_tv/dx)."""
    diffs = np.diff(x)                                      # length N-1
    denom = np.sqrt(diffs * diffs + DELTA_TV * DELTA_TV)
    obj = BETA_TV * np.sum(denom)                            # drop -DELTA_TV constant
    g = diffs / denom                                        # smooth sign of each diff
    grad = np.zeros_like(x)
    grad[1:]  += BETA_TV * g                                 # contribution from k = i+1 in Delta_i
    grad[:-1] -= BETA_TV * g                                 # contribution from k = i   in Delta_i
    return obj, grad


def objective_and_gradient(x):
    global iteration_count
    iteration_count += 1

    print(f"\n--- Iter {iteration_count}: evaluating obj+grad for {len(x)} layers ---")

    with open(HISTORY_FILE, "a") as f:
        f.write(",".join(f"{v:.10e}" for v in x) + "\n")

    # Write alpha into the OptimizationReporter block (single match in optimize.i)
    param_str = "; ".join(f"{v:.15e}" for v in x)
    new_moose_content = re.sub(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\1" + param_str + r"\2",
        base_moose_content,
        count=1,
    )

    temp_input_path = os.path.join(WORKDIR, "optimize_temp.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)

    obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out_OptimizationReporter_0001.csv")
    log_path = os.path.join(OUTPUT_DIR, "simulation_opt.log")
    for p in (obj_csv, grad_csv, log_path):
        if os.path.exists(p):
            os.remove(p)

    success, _, _ = runner.run(
        input_file_path=temp_input_path,
        output_directory=OUTPUT_DIR,
        num_processors=20,
        log_file_name="simulation_opt.log",
        stream_output=True,
        clean_output_dir=False,
    )
    if not success:
        print("MOOSE run failed; penalizing this step.")
        return 1e10, np.zeros_like(x)

    try:
        # --- Objective (CSV preferred, log fallback) ---
        obj_val = None
        if os.path.exists(obj_csv):
            obj_df = pd.read_csv(obj_csv)
            if len(obj_df) > 1:
                obj_val = float(obj_df["OptimizationReporter/objective_value"].iloc[-1])
        if obj_val is None or obj_val == 0.0:
            with open(log_path, "r") as lf:
                for line in lf:
                    m = re.search(r"Objective value\s*=\s*([0-9eE.+\-]+)", line)
                    if m:
                        obj_val = float(m.group(1))
        if obj_val is None:
            raise RuntimeError("Could not parse objective value from MOOSE output")

        # --- Raw per-layer gradient ---
        grad_df = pd.read_csv(grad_csv)
        grad_cols = [f"grad_perm_{i+1}" for i in range(TOTAL_LAYERS)]
        grad_array = grad_df[grad_cols].iloc[-1].values.copy()

        # --- TV regularization ---
        reg_obj, reg_grad = tv_obj_and_grad(x)

        # Apply rho/mu factor (currently 1.0; the C++ VPP already includes it)
        grad_array *= GRAD_CORRECTION

        total_obj = float(obj_val) + reg_obj
        total_grad = grad_array + reg_grad

        scaled_obj = (total_obj - BASELINE_OBJ) * SCALE_FACTOR
        scaled_grad = total_grad * SCALE_FACTOR

        print(f"Obj raw: {obj_val:.4e} | Reg TV: {reg_obj:.4e} | Total: {total_obj:.4e}")
        print(f"Obj scaled: {scaled_obj:.4f}")
        print(f"||grad|| raw={np.linalg.norm(total_grad):.4e}  "
              f"scaled={np.linalg.norm(scaled_grad):.4e}")

        with open(GRADIENT_HISTORY_FILE, "a") as f:
            f.write(",".join(f"{v:.10e}" for v in scaled_grad) + "\n")

        with open(OBJECTIVE_HISTORY_FILE, "a") as f:
            f.write(
                f"{iteration_count},{obj_val:.10e},{reg_obj:.10e},"
                f"{total_obj:.10e},{scaled_obj:.10e},"
                f"{np.linalg.norm(scaled_grad):.10e}\n"
            )

        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros_like(x)


if __name__ == "__main__":
    # Initial guess: background caprock value.
    x0 = np.full(TOTAL_LAYERS, -18.0)

    # Bounds: free inside the +/-25 m observation window, fixed at -18 outside.
    bounds = []
    layer_height = 0.5
    for i in range(TOTAL_LAYERS):
        y_center = -50.0 + (i + 0.5) * layer_height
        if -25.0 <= y_center <= 25.0:
            bounds.append((-25.0, -10.0))
        else:
            bounds.append((-18.0, -18.0))

    print(f"Starting L-BFGS-B (TV) with {TOTAL_LAYERS} parameters...")

    def _checkpoint(xk):
        np.save(CHECKPOINT_FILE, xk)

    res = minimize(
        objective_and_gradient,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=_checkpoint,
        options={
            "maxiter": 300,
            "ftol": 1e-12,
            "gtol": 1e-10,
            "iprint": 1,
        },
    )

    print("\n" + "=" * 50)
    print("Optimization Result Summary (TV):")
    print("=" * 50)
    print(res.message)
    print(f"Success       : {res.success}")
    print(f"Final Objective: {res.fun}")

    output_path = os.path.join(WORKDIR, "optimized_alphas_TV.txt")
    np.savetxt(output_path, res.x)
    print(f"Optimized alphas saved to: {output_path}")

    temp_i = os.path.join(WORKDIR, "optimize_temp.i")
    if os.path.exists(temp_i):
        os.remove(temp_i)
