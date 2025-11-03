#%% In this script, I will implement a revised version of the 3xFrac history matching
# Modified from 106r_DSS_3xSingleFrac_parameter_scanner.
# Shenyao Jin, 10/27/2025
# Changelog:
# 1. Because I implemented the model_builder editor, the function will be cleaner.
# 2. This script will only focus on single fracture
# 3. Revised pressure profile to make it more realistic
# 4. Add constraint: perm_frac > perm_SRV > perm_matrix
# 5. Calibrate the strain value at the beginning of interference to zero.
# 6. Better data storage structure for optimization logs and figures.

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import datetime
import csv

# I/O modules from fiberis
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader
from fiberis.utils.viz_utils import plot_point_samplers, plot_vector_samplers

# Simulation modules from fiberis
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)

# Optimization modules
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

#%% Set up paths
# Source data paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"

# Simulation and logging output paths
project_name = "1101_DSS_SingleFrac_scanner"
base_output_dir = os.path.join("output", project_name)
moose_output_dir = os.path.join(base_output_dir, "moose_simulations")
log_dir = os.path.join(base_output_dir, "optimization_logs")
fig_parent_dir = os.path.join("figs", project_name)

os.makedirs(moose_output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(fig_parent_dir, exist_ok=True)

#%% 1. Load and preprocess DSS data
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
DSSdata.select_depth(12000, 16360)

DSSdata_copy = DSSdata.copy()
DSSdata_copy.select_depth(14500, 15500)

#%% 2. Load and preprocess gauge data
gauge_data_interference = Data1DGauge()
gauge_data_interference.load_npz(pressure_gauge_g1_path)
gauge_data_interference.select_time(DSSdata.start_time, DSSdata.get_end_time())

injection_gauge_pressure = Data1DGauge()
injection_gauge_pressure.load_npz(injection_gauge_pressure_path)
injection_gauge_pressure.select_time(injection_gauge_pressure.start_time, DSSdata.start_time)
injection_gauge_pressure.remove_abnormal_data(threshold=300, method='mean')

injection_gauge_pressure_copy = injection_gauge_pressure.copy()
gauge_data_interference_copy = gauge_data_interference.copy()
injection_gauge_pressure_copy.adaptive_downsample(300)
gauge_data_interference_copy.adaptive_downsample(600)

# Shift the interference gauge data to align with DSS data (one is wellhead, the other is downhole)
difference_val = injection_gauge_pressure.data[-1] - gauge_data_interference.data[0]
gauge_data_interference_copy.data += difference_val

injection_gauge_pressure_copy.right_merge(gauge_data_interference_copy)
injection_gauge_pressure_copy.rename("injection pressure full profile")

# Save the pressure profile to a npz file

## Plot for checking
# fig, ax = plt.subplots(figsize=(8, 5))
# injection_gauge_pressure_copy.plot(ax=ax, use_timestamp= True)
# ax.axvline(gauge_data_interference_copy.start_time, color='red', linestyle='--', label='Interference Start')
# ax.set_title("Injection Pressure Full Profile")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Pressure (psi)")
# ax.legend()
# plt.show()
#
# # Crop the zoom-in data for better visualization
# injection_gauge_pressure_zoom_in = injection_gauge_pressure_copy.copy()
# injection_gauge_pressure_zoom_in.select_time(gauge_data_interference_copy.start_time - datetime.timedelta(days=3),
#                                         gauge_data_interference_copy.get_end_time(use_timestamp=True))
# fig, ax = plt.subplots(figsize=(8, 5))
# injection_gauge_pressure_zoom_in.plot(ax=ax, use_timestamp=True)
# ax.axvline(gauge_data_interference_copy.start_time, color='red', linestyle='--', label='Interference Start')
# ax.set_title("Injection Pressure Zoom-in Profile")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Pressure (psi)")
# ax.legend()
# plt.show()

#%% 3. Define the model builder function
conversion_factor = 0.3048

def model_builder_parameters_single_frac(**kwargs) -> List[Data1D_MOOSEps]:
    """
    Build a fiberis model for history matching using single hydraulic fracture model.
    It's slightly revised from the previous version, especially in post-processing.
    -- Shenyao Jin, 11/01/2025
    :param kwargs: Those are the parameters for history matching. See implementation for details.
    :return: A list of Data1D objects know as the postprocessors from the simulation. Can be used for misfit calculation.
    """

    time_now = datetime.datetime.now()
    date_time_str_tmp = time_now.strftime("%Y%m%d_%H%M%S")
    instance_id = kwargs.get('instance_id', date_time_str_tmp)

    # Define the output directory for the model, which would be good for reproducibility.
    model_output_dir = os.path.join(moose_output_dir, f"model_instance_{instance_id}")
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    input_file_path = os.path.join(model_output_dir, project_name + "_input.i")
    # Build the model
    builder = ModelBuilder(project_name=project_name)

    # Mesh and geometry
    frac_coords = 0  # Should be in the center of the model
    domain_bounds = (- kwargs.get('model_width', 200.0 * conversion_factor),
                     + kwargs.get('model_width', 200.0 * conversion_factor))

    domain_length = kwargs.get('model_length', 800.0 * conversion_factor)

    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=frac_coords,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=kwargs.get('nx', 200),
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 80),
        bias_y=kwargs.get('bias_y', 1.2)
    )

    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-15)
    srv_perm2 = kwargs.get('srv_perm2', 1e-14)
    fracture_perm = kwargs.get('fracture_perm', 1e-13)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    srv_perm_str2 = f"{srv_perm2} 0 0 0 {srv_perm2} 0 0 0 {srv_perm2}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
    srv_mats2 = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str2)
    fracture_mats = ZoneMaterialProperties(porosity=0.16, permeability=fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft2 = kwargs.get('srv_length_ft', 400)
    srv_height_ft2 = kwargs.get('srv_height_ft', 50)
    srv_length_ft1 = kwargs.get('srv_length_ft1', 250)
    srv_height_ft1 = kwargs.get('srv_height_ft1', 150)
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv_tall", length=srv_length_ft1 * conversion_factor, height=srv_height_ft1 * conversion_factor,
                  center_x=center_x_val, center_y=frac_coords, materials=srv_mats2),
        SRVConfig(name="srv_wide", length=srv_length_ft2 * conversion_factor, height=srv_height_ft2 * conversion_factor,
                  center_x=center_x_val, center_y=frac_coords, materials=srv_mats),
        HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor,
                                height=hf_height_ft * conversion_factor, center_x=center_x_val,
                                center_y=frac_coords, materials=fracture_mats)
    ]

    sorted_geometries = sorted(geometries, key=lambda x: x.height, reverse=True)
    next_block_id = 1
    for geom_config in sorted_geometries:
        if isinstance(geom_config, SRVConfig):
            builder.add_srv_config(geom_config)
            builder.add_srv_zone_2d(geom_config, target_block_id=next_block_id)
        elif isinstance(geom_config, HydraulicFractureConfig):
            builder.add_fracture_config(geom_config)
            builder.add_hydraulic_fracture_2d(geom_config, target_block_id=next_block_id)
        next_block_id += 1

    builder.add_nodeset_by_coord(nodeset_op_name="injection", new_boundary_name="injection",
                                 coordinates=(center_x_val, frac_coords, 0))

    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": kwargs.get('initial_pressure', 5.17E7)}},
        {"name": "disp_x", "params": {"initial_condition": 0}},
        {"name": "disp_y", "params": {"initial_condition": 0}}
    ])

    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    biot_coeff = 0.7
    builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=2E-11
    )

    # Add production data from the injection well
    # Difference from this to 105r is I will use 1. production pressure profile, and 2. interference test data profile.
    # dataframe: gauge_data_interference AND injection_gauge_pressure
    # I will use these two dataframes to configure the time stepper.

    gauge_data_for_moose = injection_gauge_pressure_copy.copy()
    gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa

    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    # Add post-processors
    spacing_dss = (DSSdata.daxis[1] - DSSdata.daxis[0]) * conversion_factor

    # Use 3x locations for history matching: upper, middle, lower channel of DSS data
    # In the future, I will implement more locations for better history matching.
    shift_val = 80
    builder.add_postprocessor(
        PointValueSamplerConfig(name="strain_upper_channel_dss", variable="strain_yy",
                                point=(center_x_val + shift_val * conversion_factor, + spacing_dss, 0))
    )
    builder.add_postprocessor(
        PointValueSamplerConfig(name="strain_middle_channel_dss", variable="strain_yy",
                                point=(center_x_val + shift_val * conversion_factor, 0.0, 0))
    )
    builder.add_postprocessor(
        PointValueSamplerConfig(name="strain_lower_channel_dss", variable="strain_yy",
                                point=(center_x_val + shift_val * conversion_factor, - spacing_dss, 0))
    )

    # Add a vector sampler to mimic the fiber
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="fiber_strain",
            variable="strain_yy",
            start_point=(center_x_val + shift_val * conversion_factor, domain_bounds[0] + 20 * conversion_factor, 0),
            end_point=(center_x_val + shift_val * conversion_factor, domain_bounds[1] - 20 * conversion_factor, 0),
            num_points=200,
            other_params={'sort_by': 'y'}
        )
    )

    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="pressure_monitor_well",
            variable="pp",
            start_point=(center_x_val + shift_val * conversion_factor, domain_bounds[0] + 20 * conversion_factor, 0),
            end_point=(center_x_val + shift_val * conversion_factor, domain_bounds[1] - 20 * conversion_factor, 0),
            num_points=200,
            other_params={'sort_by': 'y'}
        )
    )

    # Define the time stepper
    total_time = gauge_data_for_moose.taxis[-1] - gauge_data_for_moose.taxis[0]
    # Down sample two dataframes to reduce computational cost
    # Logic here can be improved in the future.
    gauge_data_interference_stepper = gauge_data_interference_copy.copy()
    injection_gauge_pressure_stepper = injection_gauge_pressure_copy.copy()
    gauge_data_interference_stepper.adaptive_downsample(120)
    injection_gauge_pressure_stepper.adaptive_downsample(20)
    timestepper_profile = injection_gauge_pressure_stepper.copy()
    timestepper_profile.select_time(timestepper_profile.start_time, gauge_data_interference_stepper.start_time)
    timestepper_profile.right_merge(gauge_data_interference_stepper)

    # Define the time stepper function
    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(timestepper_profile)

    # Define the time stepper block
    builder.add_executioner_block(
        end_time=total_time,
        dt=3600 * 24 * 5,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control_func
    )

    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=False, csv=True)

    # Render the input file
    builder.generate_input_file(input_file_path)
    builder.plot_geometry()

    # Run the model.
    runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
    success, stdout, stderr = runner.run(
        input_file_path=input_file_path,
        output_directory=model_output_dir,
        num_processors=20,
        log_file_name="simulation.log",
        stream_output=True
    )
    channel_result = []
    for i in range(3):
        ps_reader = MOOSEPointSamplerReader()
        ps_reader.read(folder = model_output_dir, variable_index = i + 1)

        simulation_dataframe = Data1D_MOOSEps()
        simulation_dataframe = ps_reader.to_analyzer()
        # CAUTION: the simulation dataframe does not contain a valid start_time! Need to set it here.
        simulation_dataframe.start_time = timestepper_profile.start_time
        channel_result.append(simulation_dataframe)
    return channel_result

# Calculate the misfit between simulation and observation
def calculate_misfit_single_frac(simulation_results: List[Data1D_MOOSEps],
                                 observation_data: DSS2D,
                                 frac_locations: np.ndarray,
                                 params: dict,
                                 instance_id: str,
                                 fig_output_dir: str) -> float:
    """
    Calculate the misfit between simulation results and observation data for single fracture model.
    This function is similar to the one in 105r_broader_param_scanner.py. It calculates the misfit,
    plots the comparison, and saves the figure. -- Shenyao Jin, 10/19/2025

    :param simulation_results: the output from model_builder_with_parameters_single_frac function.
    :param observation_data: the DSS2D data for history matching.
    :param frac_locations: the locations (in md) for history matching.
    :param params: Dictionary of parameters used in the simulation run, for logging and plotting.
    :param instance_id: A unique identifier for the simulation run, for saving figures.
    :param fig_output_dir: Directory to save output figures.
    :return: the calculated misfit value.
    """
    total_misfit = 0.0

    # Create a directory for the figures for this specific run
    os.makedirs(fig_output_dir, exist_ok=True)

    # Format the parameter string for display on the plot
    param_str = '\n'.join([f'{k}: {v:.2e}' for k, v in params.items()])

    for i, sim_result in enumerate(simulation_results):
        depth = frac_locations[i]

        # Get observed data at the specified depth
        obs_data_array = observation_data.get_value_by_depth(depth)

        if obs_data_array is None:
            print(f"Warning: Could not retrieve observation data for depth {depth}. Skipping.")
            continue

        # Create a Data1D object for the observation data to use its methods
        obs_gauge = Data1DGauge()
        obs_gauge.data = obs_data_array * 1e-6 # Convert from microstrain to strain
        obs_gauge.taxis = observation_data.taxis
        obs_gauge.start_time = observation_data.start_time
        obs_gauge.name = f"Observed Strain at {depth} ft"

        # Crop the simulation result to match the observation time range
        sim_result.select_time(obs_gauge.start_time, obs_gauge.get_end_time(use_timestamp=True))

        # Pre-process: Calibrate the strain value at the beginning of interference to zero.
        sim_result.data = sim_result.data - sim_result.data[0]

        # Interpolate observation data to simulation time axis
        obs_gauge.interpolate(sim_result.taxis, sim_result.start_time)

        # Calculate RMSE misfit
        if len(sim_result.data) != len(obs_gauge.data):
            print(
                f"Warning: Length mismatch for depth {depth} after interpolation. Sim: {len(sim_result.data)}, Obs: {len(obs_gauge.data)}. Skipping.")
            continue

        misfit = np.linalg.norm(sim_result.data - obs_gauge.data) / np.sqrt(len(sim_result.data))
        total_misfit += misfit

        # Plotting the comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        sim_result.plot(ax=ax, use_timestamp=True, label=f'Simulated Strain (yy)')
        obs_gauge.plot(ax=ax, use_timestamp=True, label=f'Measured Strain')

        ax.set_title(f'Strain Comparison at Depth {depth:.2f} ft')
        ax.set_ylabel('Strain (yy)')
        ax.set_xlabel('Time')
        ax.legend()

        # Add parameter text to the plot
        plt.text(0.05, 0.95, param_str, transform=ax.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save the figure
        fig_path = os.path.join(fig_output_dir, f"strain_comparison_depth_{int(depth)}_{instance_id}.png")
        plt.savefig(fig_path)
        plt.close(fig)

    return total_misfit


# %% 4. Bayesian Optimization
frac_locs_all = np.array([14887.66, 14972, 14992])
dx_DSSdata = DSSdata.daxis[1] - DSSdata.daxis[0]

dimensions = [
    Real(low=1e-15, high=1e-12, prior='log-uniform', name='fracture_perm'),
    Real(low=1e-17, high=1e-14, prior='log-uniform', name='srv_perm'),
    Real(low=1e-16, high=1e-13, prior='log-uniform', name='srv_perm2'),
    Real(low=10, high=100, name='srv_height_ft'),
    Real(low=50, high=250, name='srv_height_ft1')
]

for frac_loc_center in frac_locs_all:
    print(f"\n--- Starting Optimization for Fracture Location: {frac_loc_center} ft ---")

    log_file_path = os.path.abspath(os.path.join(log_dir, f"optimization_log_frac_{frac_loc_center}.csv"))
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['misfit'] + [d.name for d in dimensions])


    @use_named_args(dimensions=dimensions)
    def objective(**params):
        if params['srv_height_ft'] >= params['srv_height_ft1']:
            return 1e10 # Penalize invalid SRV height hierarchy

        if (params['srv_perm'] <= 1e-18 or params['srv_perm2'] <= 1e-18
                or params['fracture_perm'] <= params['srv_perm2'] or params['fracture_perm'] <= params['srv_perm']):
            return 1e10 # Penalize invalid permeability hierarchy

        instance_id = f"{frac_loc_center}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        try:
            simulation_results = model_builder_parameters_single_frac(instance_id=instance_id, **params)

            current_frac_locs = np.array([frac_loc_center - dx_DSSdata, frac_loc_center, frac_loc_center + dx_DSSdata])

            misfit = calculate_misfit_single_frac(
                simulation_results=simulation_results,
                observation_data=DSSdata_copy,
                frac_locations=current_frac_locs,
                params=params,
                instance_id=instance_id,
                fig_output_dir=os.path.join(fig_parent_dir, f"frac_{frac_loc_center}")
            )
            print(f"Frac: {frac_loc_center}, Misfit: {misfit:.4e}, Params: {params}")

            with open(log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([misfit] + list(params.values()))

            return misfit
        except Exception as e:
            print(f"An error occurred during simulation for frac {frac_loc_center} with params {params}. Error: {e}")
            return 1e10


    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=100,
        random_state=42
    )

    print(f"\n--- Optimization Complete for Fracture Location: {frac_loc_center} ft ---")
    print(f"Best Misfit: {result.fun:.4e}")
    best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.2e}")

    summary_file = os.path.abspath(os.path.join(log_dir, "optimization_summary.txt"))
    with open(summary_file, 'a') as f:
        f.write(f"--- Fracture Location: {frac_loc_center} ---\n")
        f.write(f"Best Misfit: {result.fun}\n")
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
