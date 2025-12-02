# After testing out the misfit function in r3, we can now proceed to optimize the fracture parameters
# using Beyesian optimization.
# Shenyao Jin, 12/05/2025, shenyaojin@mines.edu

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import csv

# Optimization modules
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# fiberis modules
from fiberis.moose.templates.baseline_model_generator import build_baseline_model, post_processor_info_extractor, misfit_calculator
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.runner import MooseRunner

# --- PATH SETUP ---
# Source data paths
DSS_DATAPATH = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
TIMESTEPPER_PROFILE_PATH = "data/fiberis_format/post_processing/timestepper_profile.npz"
INJECTION_PRESSURE_PROFILE_PATH = "data/fiberis_format/post_processing/injection_pressure_full_profile.npz"


# Executable paths
MOOSE_EXECUTABLE = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
MPIEXEC_PATH = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"

# Simulation and logging output paths
PROJECT_NAME = "109r4_optimization"
BASE_OUTPUT_DIR = os.path.join("output", PROJECT_NAME)
LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "optimization_logs")
FIG_PARENT_DIR = os.path.join("figs", PROJECT_NAME)

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_PARENT_DIR, exist_ok=True)

print(f"Project '{PROJECT_NAME}' directories created.")
print(f"Base output: {os.path.abspath(BASE_OUTPUT_DIR)}")
print(f"Log dir: {os.path.abspath(LOG_DIR)}")
print(f"Figure dir: {os.path.abspath(FIG_PARENT_DIR)}")


# --- 1. OBSERVED DATA PREPARATION ---
print("Loading and preprocessing observed DSS data...")
# Load DSS data
DSSdata = DSS2D()
DSSdata.load_npz(DSS_DATAPATH)

# Pre-process DSS data
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
DSSdata.select_depth(14880, 14900) # Focus on the zone of interest

# Scale data: convert from microstrain to strain and apply coupling factor
SCALE_FACTOR = 7.0
DSSdata.data = DSSdata.data * SCALE_FACTOR / 1e6
print("DSS data pre-processing complete.")

# Define fracture center and weight matrix for misfit calculation
CENTER_FRAC_DEPTH_OBS = 14888.97
ind_obs = np.argmin(np.abs(DSSdata.daxis - CENTER_FRAC_DEPTH_OBS))
WEIGHT_MATRIX = np.array([1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1])
print(f"Observed fracture center depth: {CENTER_FRAC_DEPTH_OBS} ft (index: {ind_obs})")


# --- 2. BAYESIAN OPTIMIZATION SETUP ---
# Define the parameter space for the optimization
dimensions = [
    Real(low=1e-15, high=1e-12, prior='log-uniform', name='fracture_perm'),
    Real(low=1e-17, high=1e-14, prior='log-uniform', name='srv_perm')
]

# Prepare the log file
log_file_path = os.path.join(LOG_DIR, "optimization_log.csv")
with open(log_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'misfit'] + [d.name for d in dimensions])
print(f"Optimization log file created at: {log_file_path}")

# Iteration counter
iteration_counter = 0

@use_named_args(dimensions=dimensions)
def objective(**params):
    """
    The objective function for Bayesian optimization.
    It runs a simulation with the given parameters, calculates the misfit,
    saves all intermediate results, and returns the misfit.
    """
    global iteration_counter
    iteration_counter += 1
    instance_id = f"iter_{iteration_counter:03d}"
    
    print(f"\n--- Starting Iteration {iteration_counter} ---")
    param_str = ', '.join([f'{k}: {v:.2e}' for k, v in params.items()])
    print(f"Parameters: {param_str}")

    # Create instance-specific directories
    instance_output_dir = os.path.join(BASE_OUTPUT_DIR, instance_id)
    instance_fig_dir = os.path.join(FIG_PARENT_DIR, instance_id)
    os.makedirs(instance_output_dir, exist_ok=True)
    os.makedirs(instance_fig_dir, exist_ok=True)

    try:
        # --- Simulation ---
        print("1. Building MOOSE model...")
        # Note: We fix srv_height_ft to 5 ft as requested.
        builder = build_baseline_model(
            project_name=f"{PROJECT_NAME}_{instance_id}",
            fracture_perm=params['fracture_perm'],
            srv_perm=params['srv_perm'],
            srv_height_ft=5.0,
            matrix_perm=1e-18
        )

        input_file_path = os.path.join(instance_output_dir, "input.i")
        builder.generate_input_file(output_filepath=input_file_path)

        print("2. Running MOOSE simulation...")
        runner = MooseRunner(moose_executable_path=MOOSE_EXECUTABLE, mpiexec_path=MPIEXEC_PATH)
        success, _, _ = runner.run(
            input_file_path=input_file_path,
            output_directory=instance_output_dir,
            num_processors=20,
            log_file_name="simulation.log",
            stream_output=False
        )
        if not success:
            raise RuntimeError("MOOSE simulation failed.")

        # --- Post-processing and Visualization ---
        print("3. Post-processing simulation results...")
        _, strain_dataframe = post_processor_info_extractor(output_dir=instance_output_dir)

        # Set time axis correctly
        pg_frame = Data1DGauge()
        pg_frame.load_npz(TIMESTEPPER_PROFILE_PATH)
        strain_dataframe.start_time = pg_frame.start_time
        
        # Align time range with observed data
        strain_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())

        # Apply baseline correction (subtract initial strain)
        if strain_dataframe.data.shape[0] > 0:
            strain_dataframe.data -= strain_dataframe.data[:, 0][:, np.newaxis]

        # --- Save Intermediate Results ---
        # a) Save the strain_dataframe object
        strain_df_save_path = os.path.join(instance_output_dir, 'strain_dataframe.npz')
        strain_dataframe.savez(strain_df_save_path)
        print(f"   - Saved strain dataframe to {strain_df_save_path}")

        # b) Save the waterfall plot
        fig, ax = plt.subplots(figsize=(10, 7))
        strain_dataframe.plot(ax=ax, use_timestamp=False, cmap="bwr", method='pcolormesh', clim=(-4e-5, 4e-5))
        plt.colorbar(ax.collections[0], ax=ax, label="Strain (yy)")
        ax.set_title(f"Simulated Strain - Iteration {iteration_counter}\n{param_str}")
        waterfall_path = os.path.join(instance_fig_dir, 'strain_waterfall.png')
        plt.savefig(waterfall_path)
        plt.close(fig)
        print(f"   - Saved waterfall plot to {waterfall_path}")

        # --- Misfit Calculation ---
        print("4. Calculating misfit...")
        ind_sim = len(strain_dataframe.daxis) // 2  # Center of simulated data

        misfit_val = misfit_calculator(
            weight_matrix=WEIGHT_MATRIX,
            sim_fracture_center_ind=ind_sim,
            observed_data_fracture_center_ind=ind_obs,
            simulated_data=strain_dataframe,
            observed_data=DSSdata,
            save_path=instance_fig_dir  # c) This saves the channel comparison plots
        )
        print(f"   - Misfit: {misfit_val:.4e}")
        print(f"   - Saved channel comparison plots to {instance_fig_dir}")

        # --- Logging ---
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration_counter, misfit_val] + list(params.values()))

        return misfit_val

    except Exception as e:
        print(f"!!! ERROR in iteration {iteration_counter}: {e}")
        print("!!! Assigning high misfit value (1e10).")
        # Log failure
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration_counter, 1e10] + list(params.values()))
        return 1e10


# --- 3. RUN OPTIMIZATION ---
print("\n\n=== STARTING BAYESIAN OPTIMIZATION ===")
result = gp_minimize(
    func=objective,
    dimensions=dimensions,
    n_calls=50,  # Total number of evaluations
    n_initial_points=10, # Number of random points to explore before building model
    random_state=1234
)

# --- 4. RESULTS ---
print("\n\n=== OPTIMIZATION COMPLETE ===")
print(f"Best Misfit: {result.fun:.4e}")
best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
print("Best Parameters:")
for key, value in best_params.items():
    print(f"  - {key}: {value:.3e}")

# Save summary of results
summary_file = os.path.join(LOG_DIR, "optimization_summary.txt")
with open(summary_file, 'w') as f:
    f.write("--- Bayesian Optimization Results ---\\n")
    f.write(f"Best Misfit: {result.fun}\\n")
    f.write("Best Parameters:\\n")
    for key, value in best_params.items():
        f.write(f"  - {key}: {value}\\n")
    f.write("\\nFull optimization result object:\\n")
    f.write(str(result))

print(f"\nOptimization summary saved to {summary_file}")