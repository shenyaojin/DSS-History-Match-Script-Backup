# L-BFGS-B outer driver for the strain_yy permeability inversion.
#
# Ported from perm5layer_100layer/inv/102_optimization_runner.py. Differences
# from the disp_y version:
#   - Observation channel is strain_yy (driven by optimize.i / forward_and_adjoint.i,
#     which already use OptimizationData.variable = 'strain_yy' and a dipole
#     DiracKernel on disp_y_adjoint; see fwd/101_ground_truth.py and
#     fwd/102_observation_extractor.py for the data-side switch).
#   - MOOSE is run out of inv/inv_output/ because optimize.i references
#     '../measurement_data.csv' and '../forward_and_adjoint.i' relative to
#     that sub-directory.
#
# The loop:
#   1. Write the current alpha vector into optimize_temp.i (regex replace of
#      OptimizationReporter/initial_condition).
#   2. Run one MOOSE forward+adjoint; TAO exits at iter 0 because optimize.i
#      hard-codes -tao_gatol=1e50 (used here purely as an obj+grad probe).
#   3. Parse objective + per-layer gradient, apply rho/mu correction, add
#      Tikhonov smoothness + prior-anchor regularization, hand back to SciPy.
#
# Run as:
#   python scripts/.../perm5layer_100_v2strain/inv/102_optimization_runner.py

import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner


# --- Paths --------------------------------------------------------------------
WORKDIR = os.path.dirname(os.path.abspath(__file__))          # .../inv/
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
# MOOSE runs out of inv_output/ so that '../measurement_data.csv' and
# '../forward_and_adjoint.i' inside optimize.i resolve correctly.
OUTPUT_DIR = os.path.join(WORKDIR, "inv_output")

# --- Objective / gradient scaling for L-BFGS-B --------------------------------
SCALE_FACTOR = 1e6      # maps strain_yy objective (~3e-6 at init) to O(1)
BASELINE_OBJ = 0.0

# --- Regularization -----------------------------------------------------------
# Smoothness (Tikhonov on first differences of alpha)
BETA_SMOOTH = 1e-6
# Prior anchor toward background caprock value
ALPHA_PRIOR = -18.0
BETA_PRIOR = 1e-6

# --- Adjoint gradient correction ---------------------------------------------
# ElementOptimization*InnerProduct VPPs compute the perm gradient from
# forward pressure x adjoint pressure, but PorousFlow Darcy carries a rho/mu
# factor that the VPP does not. Multiply the raw gradient by rho/mu to recover
# the correct physical sensitivity. This is a property of the forward PDE
# parameter, not of the observation channel, so it is identical to the disp_y
# case.
FLUID_DENSITY = 1000.0      # kg/m^3 (SimpleFluid density0)
FLUID_VISCOSITY = 1.0e-3    # Pa.s  (SimpleFluid viscosity)
GRAD_CORRECTION = FLUID_DENSITY / FLUID_VISCOSITY   # 1e6

TOTAL_LAYERS = 200

print(f"Working directory : {WORKDIR}")
print(f"MOOSE cwd         : {OUTPUT_DIR}")

# --- Quick sanity check that we really are running the strain_yy pipeline ----
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

# --- History files (reset on every fresh driver invocation) ------------------
HISTORY_FILE = os.path.join(WORKDIR, "parameter_history.csv")
GRADIENT_HISTORY_FILE = os.path.join(WORKDIR, "gradient_history.csv")
for p in (HISTORY_FILE, GRADIENT_HISTORY_FILE):
    if os.path.exists(p):
        os.remove(p)

iteration_count = 0


def objective_and_gradient(x):
    global iteration_count
    iteration_count += 1

    print(f"\n--- Iter {iteration_count}: evaluating obj+grad for {len(x)} layers ---")

    with open(HISTORY_FILE, "a") as f:
        f.write(",".join(f"{v:.10e}" for v in x) + "\n")

    # Write alpha vector into the OptimizationReporter block. optimize.i has
    # exactly one 'initial_condition = '...'' block (inside OptimizationReporter),
    # so count=1 is safe.
    param_str = "; ".join(f"{v:.15e}" for v in x)
    new_moose_content = re.sub(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\1" + param_str + r"\2",
        base_moose_content,
        count=1,
    )

    # Stage a temp input in WORKDIR; MooseRunner copies it into OUTPUT_DIR and
    # runs from there. The '../' paths in optimize.i resolve relative to that.
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
        # --- Objective (CSV preferred, log fallback) -------------------------
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

        # --- Raw per-layer gradient ------------------------------------------
        grad_df = pd.read_csv(grad_csv)
        grad_cols = [f"grad_perm_{i+1}" for i in range(TOTAL_LAYERS)]
        grad_array = grad_df[grad_cols].iloc[-1].values.copy()

        # --- Tikhonov smoothness regularization ------------------------------
        diffs = np.diff(x)
        reg_obj_smooth = BETA_SMOOTH * np.sum(diffs**2)
        reg_grad = np.zeros_like(x)
        reg_grad[1:] += 2 * BETA_SMOOTH * diffs
        reg_grad[:-1] -= 2 * BETA_SMOOTH * diffs

        # --- Prior anchor regularization -------------------------------------
        deviation = x - ALPHA_PRIOR
        reg_obj_prior = BETA_PRIOR * np.sum(deviation**2)
        reg_grad += 2 * BETA_PRIOR * deviation

        reg_obj = reg_obj_smooth + reg_obj_prior

        # Apply rho/mu correction to the raw adjoint gradient.
        grad_array *= GRAD_CORRECTION

        total_obj = float(obj_val) + reg_obj
        total_grad = grad_array + reg_grad

        scaled_obj = (total_obj - BASELINE_OBJ) * SCALE_FACTOR
        scaled_grad = total_grad * SCALE_FACTOR

        print(f"Obj raw: {obj_val:.4e} | Reg smooth: {reg_obj_smooth:.4e} | "
              f"Reg prior: {reg_obj_prior:.4e} | Total: {total_obj:.4e}")
        print(f"Obj scaled: {scaled_obj:.4f}")
        print(f"||grad|| raw={np.linalg.norm(total_grad):.4e}  "
              f"scaled={np.linalg.norm(scaled_grad):.4e}")

        with open(GRADIENT_HISTORY_FILE, "a") as f:
            f.write(",".join(f"{v:.10e}" for v in scaled_grad) + "\n")

        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros_like(x)


if __name__ == "__main__":
    # Initial guess: background caprock value. Ground truth has two thin SRV
    # bands at alpha ~= -14.5 (sandstone) and -15 (shale); the rest is -18.
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

    print(f"Starting L-BFGS-B optimization with {TOTAL_LAYERS} parameters...")

    res = minimize(
        objective_and_gradient,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 300,
            "ftol": 1e-12,
            "gtol": 1e-10,
            "iprint": 1,
        },
    )

    print("\n" + "=" * 50)
    print("Optimization Result Summary:")
    print("=" * 50)
    print(res.message)
    print(f"Success       : {res.success}")
    print(f"Final Objective: {res.fun}")

    output_path = os.path.join(WORKDIR, "optimized_alphas.txt")
    np.savetxt(output_path, res.x)
    print(f"Optimized alphas saved to: {output_path}")

    temp_i = os.path.join(WORKDIR, "optimize_temp.i")
    if os.path.exists(temp_i):
        os.remove(temp_i)
