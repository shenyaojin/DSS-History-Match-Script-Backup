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
    
    # Format current parameter values
    param_str = f"{x[0]:.6f}; {x[1]:.6f}; {x[2]:.6f}"
    
    # Create new MOOSE content replacing initial_condition
    # We replace the specific string in OptimizationReporter block
    new_moose_content = re.sub(
        r"initial_condition\s*=\s*'[^']*'",
        f"initial_condition = '{param_str}'",
        base_moose_content
    )
    
    # Save to a temporary input file
    temp_input_path = os.path.join(WORKDIR, "optimize_temp.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)
    
    # Run MOOSE with stream_output=False to keep stdout clean
    success, stdout, stderr = runner.run(
        input_file_path=temp_input_path,
        output_directory=OUTPUT_DIR,
        num_processors=18,
        log_file_name="simulation_opt.log",
        stream_output=False,
        clean_output_dir=False
    )
    
    if not success:
        print("MOOSE run failed! Penalizing this step.")
        return 1e10, np.zeros(3)
        
    # The outputs are prefixed with "optimize_temp_out"
    obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, "optimize_temp_out_OptimizationReporter_0001.csv")
    
    try:
        obj_df = pd.read_csv(obj_csv)
        obj_val = obj_df["OptimizationReporter/objective_value"].iloc[-1]
        
        grad_df = pd.read_csv(grad_csv)
        g1 = grad_df["grad_perm_1"].iloc[0]
        g2 = grad_df["grad_perm_2"].iloc[0]
        g3 = grad_df["grad_perm_3"].iloc[0]
        
        grad_array = np.array([g1, g2, g3])
        
        scaled_obj = float(obj_val) * SCALE_FACTOR
        scaled_grad = grad_array * SCALE_FACTOR
        
        print(f"Objective (Scaled): {scaled_obj}")
        print(f"Gradient (Scaled):  {scaled_grad}")
        
        return scaled_obj, scaled_grad
        
    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros(3)

if __name__ == '__main__':
    # Starting with a value fully inside the bounds to avoid edge effects at iteration 0
    x0 = np.array([-15.0, -15.0, -15.0])
    
    # Define bounds as indicated in the file
    bounds = [(-25.0, -10.0), (-25.0, -10.0), (-25.0, -10.0)]
    
    # Run optimization using L-BFGS-B
    print(f"Starting optimization loop (Scale Factor: {SCALE_FACTOR})...")
    res = minimize(
        objective_and_gradient, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={
            'maxiter': 10, 
            'gtol': 1e-30, 
            'ftol': 1e-30
        }
    )
    
    print("\n" + "="*50)
    print("Optimization Result:")
    print("="*50)
    print(res)
    
    # Clean up temp file
    if os.path.exists(os.path.join(WORKDIR, "optimize_temp.i")):
        os.remove(os.path.join(WORKDIR, "optimize_temp.i"))