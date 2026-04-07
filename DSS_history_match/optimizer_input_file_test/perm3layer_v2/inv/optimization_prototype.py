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
SCALE_FACTOR = 1e20  # Scale factor to make tiny misfit values order ~1

# Optional: Set a baseline to subtract from the objective to help L-BFGS-B's relative reduction check
BASELINE_OBJ = 3.9e-13  # Adjust this closer to your actual raw objective values

print(f"Working Directory: {WORKDIR}")
# Initialize the MooseRunner
runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)

# Read the base MOOSE file
with open(INPUT_FILE, "r") as f:
    base_moose_content = f.read()


def objective_and_gradient(x):
    print(f"\n--- Evaluating parameters: {x} ---")

    # FIX 1: Use high precision format so SciPy's line search isn't truncated
    param_str = f"{x[0]:.15e}; {x[1]:.15e}; {x[2]:.15e}"

    # Create new MOOSE content replacing initial_condition
    new_moose_content = re.sub(
        r"initial_condition\s*=\s*'[^']*'",
        f"initial_condition = '{param_str}'",
        base_moose_content
    )

    # Save to a temporary input file
    temp_input_path = os.path.join(WORKDIR, "optimize_temp.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)

    # FIX 2: Manually delete previous CSV outputs to guarantee we don't read stale appended data
    obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out_OptimizationReporter_0001.csv")
    if os.path.exists(obj_csv):
        os.remove(obj_csv)
    if os.path.exists(grad_csv):
        os.remove(grad_csv)

    # Run MOOSE
    success, stdout, stderr = runner.run(
        input_file_path=temp_input_path,
        output_directory=OUTPUT_DIR,
        num_processors=18,
        log_file_name="simulation_opt.log",
        stream_output=True,
        clean_output_dir=False
    )

    if not success:
        print("MOOSE run failed! Penalizing this step.")
        # Return a large scalar and zero gradient to steer optimizer away
        return 1e10, np.zeros(3)

    try:
        # Read Objective
        obj_df = pd.read_csv(obj_csv)
        obj_val = obj_df["OptimizationReporter/objective_value"].iloc[-1]

        # FIX 3: Read Gradient using .iloc[-1] to ensure we get the final state of the run
        grad_df = pd.read_csv(grad_csv)
        g1 = grad_df["grad_perm_1"].iloc[-1]
        g2 = grad_df["grad_perm_2"].iloc[-1]
        g3 = grad_df["grad_perm_3"].iloc[-1]

        grad_array = np.array([g1, g2, g3])

        # FIX 4: Subtract a baseline to keep scaled objective relatively close to 0
        scaled_obj = (float(obj_val) - BASELINE_OBJ) * SCALE_FACTOR
        scaled_grad = grad_array * SCALE_FACTOR

        print(f"Objective (Scaled/Shifted): {scaled_obj}")
        print(f"Gradient (Scaled):  {scaled_grad}")

        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros(3)


if __name__ == '__main__':
    # Starting with a value fully inside the bounds to avoid edge effects
    x0 = np.array([-15.0, -15.0, -15.0])

    # Define bounds as indicated in the file
    bounds = [(-25.0, -10.0), (-25.0, -10.0), (-25.0, -10.0)]

    print(f"Starting optimization loop (Scale Factor: {SCALE_FACTOR})...")

    # FIX 5: Use factr=10.0 (high accuracy) instead of ftol
    res = minimize(
        objective_and_gradient,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={
            'maxiter': 50,  # Increased to give it room to run
            'factr': 10.0,  # factr is the correct tolerance parameter for L-BFGS-B
            'pgtol': 1e-8,  # Gradient tolerance
        }
    )

    print("\n" + "=" * 50)
    print("Optimization Result:")
    print("=" * 50)
    print(res)

    # Clean up temp file
    if os.path.exists(os.path.join(WORKDIR, "optimize_temp.i")):
        os.remove(os.path.join(WORKDIR, "optimize_temp.i"))