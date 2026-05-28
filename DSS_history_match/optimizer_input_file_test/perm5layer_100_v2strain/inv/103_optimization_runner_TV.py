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
#
# Diagnostic truth probe:
#   TV_RUN_MODE=truth_probe python 103_optimization_runner_TV.py

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
LAYER_HEIGHT = 0.5
BACKGROUND_ALPHA = -18.0

# Run modes:
#   optimize    - start L-BFGS-B from the synthetic truth alpha vector.
#   truth_probe - evaluate obj+grad once at the synthetic truth and exit.
RUN_MODE = os.environ.get("TV_RUN_MODE", "optimize").strip().lower()
TRUE_ALPHA_FILE = os.path.join(WORKDIR, "true_alphas_TV.txt")
TRUE_PROBE_SUMMARY_FILE = os.path.join(WORKDIR, "truth_probe_TV_summary.txt")

print(f"Working directory : {WORKDIR}")
print(f"MOOSE cwd         : {OUTPUT_DIR}")
print(f"Regularization    : TV only, beta_TV = {BETA_TV}, delta = {DELTA_TV}")
print(f"Run mode          : {RUN_MODE}")

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


def build_synthetic_truth_alpha():
    """Return the exact alpha profile used by fwd/output_gt/casing_model_test.i."""
    alpha = np.full(TOTAL_LAYERS, BACKGROUND_ALPHA)
    y_bottom = -50.0 + np.arange(TOTAL_LAYERS) * LAYER_HEIGHT
    y_top = y_bottom + LAYER_HEIGHT

    low_perm_srv = (y_bottom >= -20.0) & (y_top <= -16.0)
    high_perm_srv = (y_bottom >= 14.0) & (y_top <= 20.0)

    alpha[low_perm_srv] = -15.0                 # 1e-15 m^2
    alpha[high_perm_srv] = np.log10(3e-15)      # 3e-15 m^2
    return alpha


def make_bounds():
    """Free layers inside the observation window; fixed background outside."""
    bounds = []
    for i in range(TOTAL_LAYERS):
        y_center = -50.0 + (i + 0.5) * LAYER_HEIGHT
        if -25.0 <= y_center <= 25.0:
            bounds.append((-25.0, -10.0))
        else:
            bounds.append((BACKGROUND_ALPHA, BACKGROUND_ALPHA))
    return bounds


def summarize_alpha(name, alpha):
    active = np.where(np.abs(alpha - BACKGROUND_ALPHA) > 1e-12)[0]
    print(f"{name}: min={alpha.min():.6f}, max={alpha.max():.6f}, active_layers={len(active)}")
    if len(active):
        print(f"{name}: active 1-based layer range {active[0] + 1}..{active[-1] + 1}")


def write_truth_probe_summary():
    if not os.path.exists(OBJECTIVE_HISTORY_FILE):
        return

    df = pd.read_csv(OBJECTIVE_HISTORY_FILE)
    if df.empty:
        return

    row = df.iloc[-1]
    with open(TRUE_PROBE_SUMMARY_FILE, "w") as f:
        f.write("Synthetic truth probe for 103_optimization_runner_TV.py\n")
        f.write(f"run_mode={RUN_MODE}\n")
        f.write(f"beta_tv={BETA_TV:.10e}\n")
        f.write(f"delta_tv={DELTA_TV:.10e}\n")
        f.write(f"obj_raw={row['obj_raw']:.10e}\n")
        f.write(f"reg_tv={row['reg_tv']:.10e}\n")
        f.write(f"obj_total={row['obj_total']:.10e}\n")
        f.write(f"obj_scaled={row['obj_scaled']:.10e}\n")
        f.write(f"grad_norm_scaled={row['grad_norm_scaled']:.10e}\n")


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
    if RUN_MODE not in {"optimize", "truth_probe"}:
        raise RuntimeError("TV_RUN_MODE must be either 'optimize' or 'truth_probe'.")

    # Initial guess: exact synthetic truth from fwd/output_gt/casing_model_test.i.
    x0 = build_synthetic_truth_alpha()
    np.savetxt(TRUE_ALPHA_FILE, x0)
    summarize_alpha("Synthetic truth alpha", x0)
    print(f"Synthetic truth alpha saved to: {TRUE_ALPHA_FILE}")

    bounds = make_bounds()

    if RUN_MODE == "truth_probe":
        print("Evaluating objective/gradient once at the synthetic truth...")
        obj, grad = objective_and_gradient(x0)
        write_truth_probe_summary()
        print(f"Truth probe scaled objective: {obj:.10e}")
        print(f"Truth probe scaled grad norm: {np.linalg.norm(grad):.10e}")
        print(f"Truth probe summary saved to: {TRUE_PROBE_SUMMARY_FILE}")
        temp_i = os.path.join(WORKDIR, "optimize_temp.i")
        if os.path.exists(temp_i):
            os.remove(temp_i)
        raise SystemExit(0)

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
