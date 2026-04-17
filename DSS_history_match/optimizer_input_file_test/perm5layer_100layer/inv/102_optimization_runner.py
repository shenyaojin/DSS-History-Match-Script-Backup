import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner


# Configuration
WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
OUTPUT_DIR = WORKDIR
SCALE_FACTOR = 1e6  # Scale obj/grad so gradient is O(1) for L-BFGS-B
BASELINE_OBJ = 0.0

# --- Regularization (combats ill-posedness: caprock layers being pulled up) ---
# Smoothness (Tikhonov on first differences): penalizes non-smooth alpha
BETA_SMOOTH = 1e-3
# Prior anchoring: penalizes deviation from background caprock value (-18)
ALPHA_PRIOR = -18.0
BETA_PRIOR = 1e-3

# The MOOSE VPP (ElementOptimizationDiffusionCoefFunctionInnerProduct) computes
# ∫ f'(α)·∇p·∇p† dΩ, but PorousFlow Darcy uses (ρ/μ)·K·∇p.
# The VPP is missing the ρ₀/μ factor. Multiply gradient by ρ₀/μ to correct.
FLUID_DENSITY = 1000.0   # kg/m³ (SimpleFluid density0)
FLUID_VISCOSITY = 1.0e-3 # Pa·s  (SimpleFluid viscosity)
GRAD_CORRECTION = FLUID_DENSITY / FLUID_VISCOSITY  # = 1e6

print(f"Working Directory: {WORKDIR}")

# Initialize the MooseRunner
runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)

# Read the base MOOSE file
with open(INPUT_FILE, "r") as f:
    base_moose_content = f.read()

# Number of layers in our model (200 layers at 0.5m each)
TOTAL_LAYERS = 200

# History file to store parameters
HISTORY_FILE = os.path.join(WORKDIR, "parameter_history.csv")
if os.path.exists(HISTORY_FILE):
    os.remove(HISTORY_FILE)

# History file to store gradients
GRADIENT_HISTORY_FILE = os.path.join(WORKDIR, "gradient_history.csv")
if os.path.exists(GRADIENT_HISTORY_FILE):
    os.remove(GRADIENT_HISTORY_FILE)

iteration_count = 0

def objective_and_gradient(x):
    global iteration_count
    iteration_count += 1
    
    print(f"\n--- Evaluating parameters for {len(x)} layers (Iter: {iteration_count}) ---")

    # Save current parameters to history file
    with open(HISTORY_FILE, "a") as f:
        f.write(",".join([f"{val:.10e}" for val in x]) + "\n")

    # Format parameters for MOOSE (semicolon separated for OptimizationReporter)
    param_str = "; ".join([f"{val:.15e}" for val in x])

    # Replace initial_condition in the OptimizationReporter block
    new_moose_content = re.sub(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\1" + param_str + r"\2",
        base_moose_content,
        count=1
    )

    # Save to a temporary input file
    temp_input_path = os.path.join(WORKDIR, "optimize_temp.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)

    # Clean up previous outputs to ensure fresh data
    log_path = os.path.join(WORKDIR, "simulation_opt.log")
    obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out_OptimizationReporter_0001.csv")
    for f_path in [log_path, obj_csv, grad_csv]:
        if os.path.exists(f_path):
            os.remove(f_path)

    # Run MOOSE optimization (one iteration to get obj/grad)
    success, stdout, stderr = runner.run(
        input_file_path=temp_input_path,
        output_directory=OUTPUT_DIR,
        num_processors=20,
        log_file_name="simulation_opt.log",
        stream_output=True,
        clean_output_dir=False
    )

    if not success:
        print("MOOSE run failed! Penalizing this step.")
        return 1e10, np.zeros_like(x)

    try:
        # Read Objective Value — try CSV first, fall back to log parsing
        obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out.csv")
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

        # Read Gradients from the reporter CSV
        # _0001 = after forward+adjoint solve at optimization step 1
        grad_df = pd.read_csv(grad_csv)

        # Extract all gradient columns
        grad_cols = [f"grad_perm_{i+1}" for i in range(TOTAL_LAYERS)]
        grad_array = grad_df[grad_cols].iloc[-1].values.copy()

        # --- Regularization ---
        # (1) Smoothness: Beta_smooth * sum( (alpha_i - alpha_{i-1})^2 )
        diffs = np.diff(x)
        reg_obj_smooth = BETA_SMOOTH * np.sum(diffs**2)

        reg_grad = np.zeros_like(x)
        reg_grad[1:] += 2 * BETA_SMOOTH * diffs
        reg_grad[:-1] -= 2 * BETA_SMOOTH * diffs

        # (2) Prior anchor: Beta_prior * sum( (alpha_i - alpha_prior)^2 )
        deviation = x - ALPHA_PRIOR
        reg_obj_prior = BETA_PRIOR * np.sum(deviation**2)
        reg_grad += 2 * BETA_PRIOR * deviation

        reg_obj = reg_obj_smooth + reg_obj_prior

        # Apply ρ/μ correction to adjoint gradient (see GRAD_CORRECTION above)
        grad_array *= GRAD_CORRECTION

        # Combine Raw Objective/Gradient with Regularization
        total_obj = (float(obj_val) + reg_obj)
        total_grad = grad_array + reg_grad

        # Scale for Optimizer
        scaled_obj = (total_obj - BASELINE_OBJ) * SCALE_FACTOR
        scaled_grad = total_grad * SCALE_FACTOR

        print(f"Objective (Raw): {obj_val:.4e} | Reg(smooth): {reg_obj_smooth:.4e} | Reg(prior): {reg_obj_prior:.4e} | Total: {total_obj:.4e}")
        print(f"Objective (Scaled): {scaled_obj:.4f}")
        print(f"Gradient Norm (Raw): {np.linalg.norm(total_grad):.4e} | (Scaled): {np.linalg.norm(scaled_grad):.4e}")

        # Save per-parameter gradient to history file
        with open(GRADIENT_HISTORY_FILE, "a") as f:
            f.write(",".join([f"{val:.10e}" for val in scaled_grad]) + "\n")

        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros_like(x)


if __name__ == '__main__':
    # 1. Set Initial Guess (Alpha = -18 matches the dominant caprock background)
    # Ground truth: caprock=1e-18 (90% of domain), two thin SRV zones at 1e-15 and 3e-15.
    # Starting from the background value avoids over-permeating the whole domain.
    x0 = np.full(TOTAL_LAYERS, -18.0)

    # 2. Define Bounds (Fixing layers outside center 50m)
    bounds = []
    layer_height = 0.5
    for i in range(TOTAL_LAYERS):
        y_center = -50.0 + (i + 0.5) * layer_height
        if -25.0 <= y_center <= 25.0:
            # Inversion region
            bounds.append((-25.0, -10.0))
        else:
            # Fixed matrix region outside observation
            bounds.append((-18.0, -18.0))

    print(f"Starting L-BFGS-B Optimization with {TOTAL_LAYERS} parameters...")

    # 3. Run Optimization using SciPy
    res = minimize(
        objective_and_gradient,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={
            'maxiter': 300,    # More iterations for 200 parameters
            'ftol': 1e-12,     # High precision
            'gtol': 1e-10,
            'iprint': 1,       # Print convergence info (0=silent, 1=summary)
        }
    )

    print("\n" + "=" * 50)
    print("Optimization Result Summary:")
    print("=" * 50)
    print(res.message)
    print(f"Success: {res.success}")
    print(f"Final Objective: {res.fun}")
    
    # Save optimized parameters
    output_path = os.path.join(WORKDIR, "optimized_alphas.txt")
    np.savetxt(output_path, res.x)
    print(f"Optimized alphas saved to: {output_path}")

    # Clean up temp files
    temp_i = os.path.join(WORKDIR, "optimize_temp.i")
    if os.path.exists(temp_i):
        os.remove(temp_i)
