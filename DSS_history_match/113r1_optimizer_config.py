# A version using better frac selection
# The architecture is the same as 112
# Shenyao Jin, 01/27/2026
import io
import json

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import csv
import shutil

# Optimization modules
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# I/O modules from fiberis
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.core1D import Data1D

# Simulation modules from fiberis
from fiberis.moose.templates.baseline_model_generator import build_baseline_model, post_processor_info_extractor, \
    misfit_calculator
from fiberis.moose.runner import MooseRunner

CWD = os.getcwd()
print("working directory:", CWD)
conversion_factor = 0.328084  # meter to feet
DSS_DATA_PATH_ABS = os.path.join(CWD,
                                 "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz")
# --- OPTIMIZATION DEFINITION ---
dimensions = [
    Real(low=1e-17, high=1e-14, prior='log-uniform', name='srv_perm'),
    Real(low=1e-15, high=1e-12, prior='log-uniform', name='fracture_perm'),
    Real(low=0.5, high=5, name='srv_height_ft')
]

# Fixed parameter
matrix_perm = 1e-20
iteration_counter = 0


def objective(frac_config: dict, base_output_dir: str, base_fig_dir: str, summary_log_path:str, **params):
    global iteration_counter
    iteration_counter += 1
    print(f"\n--- Starting Iteration {iteration_counter} ---")

    # --- 1. CONSTRAINTS CHECK ---
    if not (params['fracture_perm'] > params['srv_perm'] > matrix_perm):
        print("Constraint violated: fracture_perm > srv_perm > matrix_perm. Returning high misfit.")
        with open(summary_log_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([iteration_counter, params.get('srv_perm'), params.get('fracture_perm'), matrix_perm,
                             params.get('srv_height_ft'), 1e10])
        return 1e10

    # --- 2. CREATE ITERATION DIRS (LOGS DIR CREATED LATER) ---
    iter_output_dir = os.path.join(base_output_dir, f"iter_{iteration_counter}")
    iter_fig_dir = os.path.join(base_fig_dir, f"iter_{iteration_counter}")
    os.makedirs(iter_output_dir, exist_ok=True)
    os.makedirs(iter_fig_dir, exist_ok=True)
    project_name = f"frac_{frac_config['center_frac_depth_obs']}"
    input_file_path = os.path.join(iter_output_dir, f"{project_name}_input.i")

    try:
        # --- 3. BUILD AND RUN SIMULATION ---
        builder = build_baseline_model(
            project_name=f"{project_name}_iter_{iteration_counter}",
            srv_perm=params['srv_perm'],
            fracture_perm=params['fracture_perm'],
            matrix_perm=matrix_perm,
            srv_height_ft=params['srv_height_ft'],
            ny_per_layer_half=100,
            bias_y=1.08,
            start_offset_y=160,
            end_offset_y=160
        )
        builder.generate_input_file(output_filepath=input_file_path)

        runner = MooseRunner(
            moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
            mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
        )
        success, stdout, stderr = runner.run(
            input_file_path=input_file_path,
            output_directory=iter_output_dir,
            num_processors=20,
            log_file_name="simulation.log",
            stream_output=False
        )

        if not success:
            print(f"!!! Simulation failed for iteration {iteration_counter}. Check logs in {iter_output_dir}")
            return 1e10

        # --- 4. EXTRACT AND SAVE RESULTS ---
        pressure_dataframe, strain_dataframe = post_processor_info_extractor(output_dir=iter_output_dir)

        # Create logs dir *after* simulation run to prevent deletion
        iter_log_dir = os.path.join(iter_output_dir, "logs")
        os.makedirs(iter_log_dir, exist_ok=True)

        # --- 5. DATA PRE-PROCESSING ---
        DSSdata = DSS2D()
        DSSdata.load_npz(DSS_DATA_PATH_ABS)
        start_time = DSSdata.start_time

        pressure_dataframe.start_time = start_time
        strain_dataframe.start_time = start_time

        # Save the dataframes now that start_time is set
        pressure_dataframe.savez(os.path.join(iter_log_dir, "pressure_dataframe.npz"))
        strain_dataframe.savez(os.path.join(iter_log_dir, "strain_dataframe.npz"))

        if pressure_dataframe.taxis.shape[0] > 1:
            pressure_dataframe.data = pressure_dataframe.data[:, 1:]
            pressure_dataframe.taxis = pressure_dataframe.taxis[1:] - pressure_dataframe.taxis[1]
        if strain_dataframe.taxis.shape[0] > 1:
            strain_dataframe.data = strain_dataframe.data[:, 1:]
            strain_dataframe.taxis = strain_dataframe.taxis[1:] - strain_dataframe.taxis[1]

        mds = DSSdata.daxis
        ind = (mds > 7500) & (mds < 15000)
        drift_val = np.median(DSSdata.data[ind, :], axis=0)
        DSSdata.data -= drift_val.reshape((1, -1))
        DSSdata.select_time(0, 400000)
        selection_range = frac_config.get("selection_range_ft", 15)
        DSSdata.select_depth(frac_config['center_frac_depth_obs'] - selection_range,
                               frac_config['center_frac_depth_obs'] + selection_range)

        pressure_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())
        strain_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())

        if strain_dataframe.data.shape[0] > 0 and strain_dataframe.data.shape[1] > 0:
            strain_dataframe.data -= strain_dataframe.data[:, 0][:, np.newaxis]

        # --- 6. MISFIT CALCULATION ---
        DSSdata_for_misfit = DSS2D()
        DSSdata_for_misfit.load_npz(DSS_DATA_PATH_ABS)
        mds_misfit = DSSdata_for_misfit.daxis
        ind_misfit = (mds_misfit > 7500) & (mds_misfit < 15000)
        drift_val_misfit = np.median(DSSdata_for_misfit.data[ind_misfit, :], axis=0)
        DSSdata_for_misfit.data -= drift_val_misfit.reshape((1, -1))
        DSSdata_for_misfit.select_time(DSSdata.start_time, DSSdata.get_end_time())
        DSSdata_for_misfit.select_depth(frac_config['center_frac_depth_obs'] - selection_range,
                                          frac_config['center_frac_depth_obs'] + selection_range)

        SCALE_FACTOR = 6.0
        DSSdata_for_misfit.data = DSSdata_for_misfit.data * SCALE_FACTOR / 1e6

        center_frac_depth_obs = frac_config['center_frac_depth_obs']
        ind_obs = np.argmin(np.abs(DSSdata_for_misfit.daxis - center_frac_depth_obs))
        weight_matrix = np.array(frac_config['weight_matrix'])
        ind_sim = len(strain_dataframe.daxis) // 2

        misfit_val, misfit_per_channel = misfit_calculator(
            weight_matrix=weight_matrix,
            sim_fracture_center_ind=ind_sim,
            observed_data_fracture_center_ind=ind_obs,
            simulated_data=strain_dataframe,
            observed_data=DSSdata_for_misfit,
            save_path=iter_fig_dir,
            return_misfit_per_channel=True
        )
        print(f"--- Misfit Calculated for Iteration {iteration_counter}: {misfit_val:.4e} ---")

        # --- 7. LOGGING ---
        iter_log_path = os.path.join(iter_log_dir, 'misfit_log.txt')
        with open(iter_log_path, 'w') as f:
            f.write(f"Iteration: {iteration_counter}\n{params}\nTotal Misfit: {misfit_val}\n\n")
            np.savetxt(f, misfit_per_channel, fmt='%.4e')

        with open(summary_log_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([iteration_counter, params.get('srv_perm'), params.get('fracture_perm'), matrix_perm,
                             params.get('srv_height_ft'), misfit_val])

        # --- 8. QC PLOTTING ---
        print(f"Generating QC plots for iteration {iteration_counter}...")
        try:
            # QC plot 1: simulated strain data overview + example traces + borehole gauge data
            strain_max = np.max(np.abs(strain_dataframe.data)) if strain_dataframe.data.size > 0 else 1.0
            center_depth_ind_sim = len(strain_dataframe.daxis) // 2
            center_depth_sim = strain_dataframe.daxis[center_depth_ind_sim]

            # Create Data1D objects for example traces
            example_data_upper_data = strain_dataframe.get_value_by_channel(center_depth_ind_sim - 1)
            example_data_lower_data = strain_dataframe.get_value_by_channel(center_depth_ind_sim + 1)
            example_data_center_data = strain_dataframe.get_value_by_channel(center_depth_ind_sim)

            example_data_upper = Data1D()
            example_data_upper.taxis = strain_dataframe.taxis
            example_data_upper.data = example_data_upper_data
            example_data_upper.name = f"Strain at ch:{center_depth_ind_sim - 1}"
            example_data_upper.start_time = strain_dataframe.start_time

            example_data_lower = Data1D()
            example_data_lower.taxis = strain_dataframe.taxis
            example_data_lower.data = example_data_lower_data
            example_data_lower.name = f"Strain at ch:{center_depth_ind_sim + 1}"
            example_data_lower.start_time = strain_dataframe.start_time

            example_data_center = Data1D()
            example_data_center.taxis = strain_dataframe.taxis
            example_data_center.data = example_data_center_data
            example_data_center.name = f"Strain at ch:{center_depth_ind_sim}"
            example_data_center.start_time = strain_dataframe.start_time

            borehole_gauge = Data1D()
            borehole_gauge.load_npz(os.path.join(CWD, "data/fiberis_format/prod/gauges/pressure_g1.npz"))

            plt.figure(figsize=(11, 6))
            ax1 = plt.subplot2grid((6, 8), (0, 0), colspan=5, rowspan=4)
            strain_dataframe.plot(ax=ax1, method='pcolormesh', use_timestamp=False, xaxis_rotation=90,
                                  clim=[-strain_max, strain_max], cmap='bwr')
            ax1.set_title("Simulated Strain Data")
            ax1.set_ylabel("MD (m)")
            ax1.tick_params(labelbottom=False)

            borehole_gauge.select_time(strain_dataframe.start_time, strain_dataframe.get_end_time())
            ax2 = plt.subplot2grid((6, 8), (4, 0), colspan=5, rowspan=3, sharex=ax1)
            borehole_gauge.plot(ax=ax2, use_timestamp=False)
            ax2.set_ylabel("Pressure (psi)")

            ax3 = plt.subplot2grid((6, 8), (0, 5), colspan=3, rowspan=2)
            example_data_upper.plot(ax=ax3, use_timestamp=True)
            ax3.set_title(f"Example Strain Traces: Upper Channel")
            ax3.set_ylabel("Strain")
            ax3.tick_params(labelbottom=False)

            ax4 = plt.subplot2grid((6, 8), (2, 5), colspan=3, rowspan=2, sharex=ax3)
            example_data_center.plot(ax=ax4, use_timestamp=True)
            ax4.set_title(f"Example Strain Traces: Center Channel")
            ax4.set_ylabel("Strain")
            ax4.tick_params(labelbottom=False)

            ax5 = plt.subplot2grid((6, 8), (4, 5), colspan=3, rowspan=2, sharex=ax3)
            example_data_lower.plot(ax=ax5, use_timestamp=True)
            ax5.set_title(f"Example Strain Traces: Lower Channel")
            ax5.set_ylabel("Strain")
            ax5.set_xlabel("Time (absolute)")
            plt.tight_layout()
            qc1_path = os.path.join(iter_fig_dir, "qc_plot_1_strain_overview.png")
            plt.savefig(qc1_path)
            plt.close()
            print(f"Saved QC plot 1 to {qc1_path}")

            # QC plot 2: simulated strain vs DSS-measured strain
            DSSdata_calibrated = DSSdata.copy()
            DSSdata_calibrated.data = DSSdata.data * 6  # Calibration factor
            dss_frac_center = frac_config['center_frac_depth_obs']
            DSSdata_calibrated.select_depth(dss_frac_center - 10, dss_frac_center + 10)

            strain_dataframe_plot_copy = strain_dataframe.copy()
            strain_dataframe_plot_copy.data = strain_dataframe_plot_copy.data * 1e6  # Convert to microstrain
            strain_dataframe_plot_copy.select_depth(center_depth_sim - 10 * conversion_factor,
                                                    center_depth_sim + 10 * conversion_factor)

            plt.figure(figsize=(12, 6))
            ax1 = plt.subplot2grid((6, 8), (0, 0), colspan=4, rowspan=3)
            im1 = DSSdata.plot(ax=ax1, use_timestamp=False, cmap='bwr',
                               clim=0.1 * np.array([-strain_max * 1e6, strain_max * 1e6]))
            ax1.axhline(y=dss_frac_center, color='k', linestyle='--', label='Fracture Center')
            ax1.set_title("DSS-Measured Strain Data (Calibrated * 6)")
            ax1.set_ylabel("Depth (ft)")
            ax1.set_xlabel("Time (absolute)")
            cbar = plt.colorbar(im1, ax=ax1)
            cbar.set_label("Strain (microstrain)")
            ax1.tick_params(labelbottom=False)

            ax2 = plt.subplot2grid((6, 8), (3, 0), colspan=4, rowspan=3)
            im2 = strain_dataframe_plot_copy.plot(ax=ax2, use_timestamp=False, cmap='bwr',
                                                  clim=[-strain_max * 1e6, strain_max * 1e6], method='pcolormesh')
            ax2.axhline(y=center_depth_sim, color='k', linestyle='--', label='Fracture Center')
            ax2.set_title("Simulated Strain Data")
            ax2.set_ylabel("Depth (ft)")
            ax2.set_xlabel("Time (absolute)")
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label("Strain (microstrain)")

            # Example trace comparisons
            center_depth_ind_obs = np.argmin(np.abs(DSSdata_calibrated.daxis - dss_frac_center))
            DSS_center_data = DSSdata_calibrated.get_value_by_channel(center_depth_ind_obs)
            example_DSS_center = Data1D(taxis=DSSdata_calibrated.taxis, data=DSS_center_data,
                                        name=f"DSS Strain at ch:{center_depth_ind_obs}",
                                        start_time=DSSdata_calibrated.start_time)

            DSS_upper_data = DSSdata_calibrated.get_value_by_channel(center_depth_ind_obs - 1)
            example_DSS_upper = Data1D(taxis=DSSdata_calibrated.taxis, data=DSS_upper_data,
                                       name=f"DSS Strain at ch:{center_depth_ind_obs - 1}",
                                       start_time=DSSdata_calibrated.start_time)

            DSS_lower_data = DSSdata_calibrated.get_value_by_channel(center_depth_ind_obs + 1)
            example_DSS_lower = Data1D(taxis=DSSdata_calibrated.taxis, data=DSS_lower_data,
                                        name=f"DSS Strain at ch:{center_depth_ind_obs + 1}",
                                        start_time=DSSdata_calibrated.start_time)

            # Convert simulated for plot
            example_data_upper.data *= 1e6
            example_data_lower.data *= 1e6
            example_data_center.data *= 1e6

            ax3 = plt.subplot2grid((6, 8), (0, 4), colspan=4, rowspan=2)
            example_DSS_upper.plot(ax=ax3, use_timestamp=True, label='DSS Measured')
            example_data_upper.plot(ax=ax3, use_timestamp=True, label='Simulated')
            ax3.set_title(f"Strain Comparison at Upper Channel")
            ax3.set_ylabel("Strain (microstrain)")
            ax3.legend()
            ax3.tick_params(labelbottom=False)

            ax4 = plt.subplot2grid((6, 8), (2, 4), colspan=4, rowspan=2)
            example_DSS_center.plot(ax=ax4, use_timestamp=True, label='DSS Measured')
            example_data_center.plot(ax=ax4, use_timestamp=True, label='Simulated')
            ax4.set_title(f"Strain Comparison at Center Channel")
            ax4.set_ylabel("Strain (microstrain)")
            ax4.legend()
            ax4.tick_params(labelbottom=False)

            ax5 = plt.subplot2grid((6, 8), (4, 4), colspan=4, rowspan=2)
            example_DSS_lower.plot(ax=ax5, use_timestamp=True, label='DSS Measured')
            example_data_lower.plot(ax=ax5, use_timestamp=True, label='Simulated')
            ax5.set_title(f"Strain Comparison at Lower Channel")
            ax5.set_ylabel("Strain (microstrain)")
            ax5.set_xlabel("Time (absolute)")
            ax5.legend()
            plt.tight_layout()
            qc2_path = os.path.join(iter_fig_dir, "qc_plot_2_strain_comparison.png")
            plt.savefig(qc2_path)
            plt.close()
            print(f"Saved QC plot 2 to {qc2_path}")

        except Exception as e:
            print(f"An error occurred during QC plotting for iteration {iteration_counter}: {e}")

        return misfit_val

    except Exception as e:
        print(f"An error occurred during iteration {iteration_counter}: {e}")
        with open(summary_log_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([iteration_counter, params.get('srv_perm'), params.get('fracture_perm'), matrix_perm,
                             params.get('srv_height_ft'), 'ERROR'])
        return 1e10


def run_optimization_for_fracture(frac_config: dict, output_dir: str, fig_dir: str, n_calls: int):
    """
    Wraps the entire optimization process for a single fracture.
    """
    global iteration_counter
    iteration_counter = 0
    # --- GLOBAL OPTIMIZATION LOG ---
    summary_log_path = os.path.join(output_dir, "optimization_summary.txt")
    with open(summary_log_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['iteration', 'srv_perm', 'fracture_perm', 'matrix_perm', 'srv_height_ft', 'misfit'])

    @use_named_args(dimensions=dimensions)
    def objective_wrapper(**params):
        return objective(
            frac_config=frac_config,
            base_output_dir=output_dir,
            base_fig_dir=fig_dir,
            summary_log_path=summary_log_path,
            **params
        )

    result = gp_minimize(
        func=objective_wrapper,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=42
    )
    print("\n--- Optimization Complete ---")
    print(f"Best Misfit: {result.fun:.4e}")
    best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.2e}")

    with open(summary_log_path, 'a') as f:
        f.write("\n--- Best Result ---\n")
        f.write(f"Best Misfit: {result.fun}\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")


def optimizer_multi_frac(configuration_file: io.TextIOBase, project_name: str, max_iter: int):
    """
    Main entry point for running optimization on multiple fractures.
    """
    # 1. Setup main project directory
    main_output_dir = os.path.join(CWD, f"output/{project_name}")
    main_fig_dir = os.path.join(CWD, f"figs/{project_name}")

    if os.path.exists(main_output_dir):
        print(f"Removing existing output directory: {main_output_dir}")
        shutil.rmtree(main_output_dir)
    if os.path.exists(main_fig_dir):
        print(f"Removing existing figure directory: {main_fig_dir}")
        shutil.rmtree(main_fig_dir)

    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(main_fig_dir, exist_ok=True)

    # 2. Parse the configuration
    fracture_configs = json.load(configuration_file)
    print(f"Found {len(fracture_configs)} fractures to optimize.")

    # 3. Loop and run optimization for each fracture
    for frac_config in fracture_configs:
        print(f"\n===== Starting Optimization for Fracture: {frac_config['name']} =====")

        # Create subdirectories for this specific fracture
        frac_output_dir = os.path.join(main_output_dir, frac_config['name'])
        frac_fig_dir = os.path.join(main_fig_dir, frac_config['name'])
        os.makedirs(frac_output_dir, exist_ok=True)
        os.makedirs(frac_fig_dir, exist_ok=True)

        run_optimization_for_fracture(
            frac_config=frac_config,
            output_dir=frac_output_dir,
            fig_dir=frac_fig_dir,
            n_calls=max_iter
        )
        print(f"===== Finished Optimization for Fracture: {frac_config['name']} =====")


if __name__ == "__main__":
    config_file_path = 'scripts/DSS_history_match/config_113r1.json'
    project_name = "113r1_optimizer_test"
    max_iterations = 50
    with open(config_file_path, 'r') as f:
        optimizer_multi_frac(f, project_name, max_iterations)
