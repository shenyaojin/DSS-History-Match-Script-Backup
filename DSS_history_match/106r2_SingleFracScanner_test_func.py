#%% In this script, I will implement the first part of the DSS data history match using fiberis.
# Shenyao Jin, shenyaojin@mines.edu, 10/17/2025

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Union

# I/O modules from fiberis
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader
from fiberis.utils.viz_utils import plot_point_samplers, plot_vector_samplers
# Simulation modules from fiberis
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)

#%% Set up paths
# Source data paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
producer_gauge_md = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"


# Simulation output paths
project_name = "1124_DSS_3xSingleFrac_match"
output_dir = os.path.join("output", project_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
import datetime
# Get current date and time for figure output folder
now = datetime.datetime.now()
date_time_str = now.strftime("%Y%m%d_%H%M%S")
fig_output_dir = os.path.join("figs", date_time_str)
if not os.path.exists(fig_output_dir):
    os.makedirs(fig_output_dir)

#%% 1. Load DSS data
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)
mds = DSSdata.daxis
ind = (mds>7500)&(mds<15000)
drift_val = np.median(DSSdata.data[ind,:],axis=0)
DSSdata.data -= drift_val.reshape((1,-1))
DSSdata.select_time(0, 400000)
DSSdata.select_depth(12000, 16360)

DSSdata_copy = DSSdata.copy()
DSSdata_copy.select_depth(14500, 15500) # <- this would be the target zone for history matching

#%% 2. Load gauge data, which will be the source for pressure history matching
# (1/2) Load pressure gauge during interference test
gauge_data_interference = Data1DGauge()
gauge_data_interference.load_npz(pressure_gauge_g1_path) # <- Pressure gauge g1 at the injection well.
gauge_data_interference.select_time(DSSdata.start_time, DSSdata.get_end_time())

# (2/2) Load injection well pressure gauge during production
injection_gauge_pressure = Data1DGauge()
injection_gauge_pressure.load_npz(injection_gauge_pressure_path)
injection_gauge_pressure.select_time(injection_gauge_pressure.start_time, DSSdata.start_time) # <- Left part of the
# injection pressure profile before interference test
injection_gauge_pressure.remove_abnormal_data(threshold=300, method='mean')

# Copy & downsample
injection_gauge_pressure_copy = injection_gauge_pressure.copy()
gauge_data_interference_copy = gauge_data_interference.copy()

injection_gauge_pressure_copy.adaptive_downsample(300)
gauge_data_interference_copy.adaptive_downsample(600)

# Combine the two pressure profiles.
# Process the right merge value.
different_val = injection_gauge_pressure_copy.data[-1] - gauge_data_interference_copy.data[0]
gauge_data_interference_copy.data += different_val

injection_gauge_pressure_copy.right_merge(gauge_data_interference_copy)
injection_gauge_pressure_copy.rename("injection pressure full profile")

injection_gauge_pressure_copy_savepath = "data/fiberis_format/post_processing/injection_pressure_full_profile.npz"
if not os.path.exists(injection_gauge_pressure_copy_savepath):
    injection_gauge_pressure_copy.savez(injection_gauge_pressure_copy_savepath)

#%% 3. Build the model using fiberis
conversion_factor = 0.3048 # 1 ft = 0.3048 m

def model_builder_with_parameters_single_frac(**kwargs) -> List[Data1D_MOOSEps]:
    """
    Build a fiberis model for history matching using single hydraulic fracture model.
    -- Shenyao Jin, 10/17/2025
    :param kwargs: Those are the parameters for history matching.
    1. instance_id: str, the instance id for the model, used for output folder naming.

    :return: A list of Data1D objects known as the postprocessors from the simulation. Can be used for misfit calculation.
    """
    # Define the file paths for the model
    time_now = datetime.datetime.now()
    date_time_str_tmp = time_now.strftime("%Y%m%d_%H%M%S")
    instance_id = kwargs.get('instance_id', date_time_str_tmp)

    # Define the output directory for the model, which would be good for reproducibility.
    model_output_dir = os.path.join(output_dir, f"model_instance_{instance_id}")
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    input_file_path = os.path.join(model_output_dir, project_name + "_input.i")
    # Build the model
    builder = ModelBuilder(project_name=project_name)

    # Mesh and geometry
    frac_coords = 0 # Should be in the center of the model
    domain_bounds = (- kwargs.get('model_width', 200.0 * conversion_factor),
                     + kwargs.get('model_width', 200.0 * conversion_factor))

    domain_length = kwargs.get('model_length', 800.0 * conversion_factor)

    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords = frac_coords,
        domain_bounds = domain_bounds,
        domain_length = domain_length,
        nx = kwargs.get('nx', 200),
        ny_per_layer_half = kwargs.get('ny_per_layer_half', 80),
        bias_y = kwargs.get('bias_y', 1.2)
    )

    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-15)
    fracture_perm = kwargs.get('fracture_perm', 1e-13)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    # Material properties
    matrix_mats = ZoneMaterialProperties(porosity = 0.01, permeability = matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity = 0.1, permeability = srv_perm_str)
    fracture_mats = ZoneMaterialProperties(porosity = 0.16, permeability = fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft = kwargs.get('srv_length_ft', 400)
    srv_height_ft = kwargs.get('srv_height_ft', 50)
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv", length=srv_length_ft * conversion_factor, height=srv_height_ft * conversion_factor,
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

    builder.set_porous_flow_dictator(dictator_name = "dictator", porous_flow_variables = "pp")
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

    builder.add_piecewise_function_from_data1d(name = "injection_pressure_func", source_data1d = gauge_data_for_moose)

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name = "injection",
        injection_pressure_function_name = "injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    # Add postprocessors at locations of interest.
    # The post processors should work because near the fracture, the resolution should be fine enough
    # At least it's finer than RFS-DSS spacing.
    shift_val = 80
    spacing_dss = (DSSdata.daxis[1] - DSSdata.daxis[0]) * conversion_factor # I remember the spacing should be uniform

    # Use one location for history matching, corresponding to the upper channel in the original setup.
    builder.add_postprocessor(
        PointValueSamplerConfig(name="strain_monitor_point", variable = "strain_yy",
                                point = (center_x_val + shift_val * conversion_factor, + spacing_dss, 0))
    )

    y_min_linear_postprocessor = domain_bounds[0] + 20 * conversion_factor
    y_max_linear_postprocessor = domain_bounds[1] - 20 * conversion_factor

    builder.add_postprocessor(
        LineValueSamplerConfig(
            name = "fiber_strain",
            variable = "strain_yy",
            start_point = (center_x_val + shift_val * conversion_factor, y_min_linear_postprocessor, 0),
            end_point = (center_x_val + shift_val * conversion_factor, y_max_linear_postprocessor, 0),
            num_points = 200,
            other_params = {'sort_by': 'y'}
        )
    )

    # Also, pressure sampler at the monitor well for quality control
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name = "pressure_monitor_well",
            variable = "pp",
            start_point = (center_x_val + shift_val * conversion_factor, y_min_linear_postprocessor, 0),
            end_point = (center_x_val + shift_val * conversion_factor, y_max_linear_postprocessor, 0),
            num_points = 200,
            other_params = {'sort_by': 'y'}
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
    # Let's save the timestepper profile for future reference
    timestepper_profile_savepath = "data/fiberis_format/post_processing/timestepper_profile.npz"
    if not os.path.exists(timestepper_profile_savepath):
        timestepper_profile.savez(timestepper_profile_savepath)

    # Define the time stepper function
    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(timestepper_profile)

    # Define the time stepper block
    builder.add_executioner_block(
        end_time = total_time,
        dt=3600 * 24 * 5,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control_func
    )

    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=False, csv=True)

    # Render the input file
    builder.generate_input_file(input_file_path)
    builder.plot_geometry(save_path=os.path.join(model_output_dir, "model_geometry.png"))

    # Run the model.
    runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
    success, stdout, stderr = runner.run(
            input_file_path = input_file_path,
            output_directory = output_dir,
            num_processors = 20,
            log_file_name = "simulation.log",
            stream_output = True
        )

    # After the simulation, output the necessary figures for quality control
    plot_point_samplers(folder = output_dir,
                        output_dir = fig_output_dir)

    plot_vector_samplers(folder = output_dir,
                         output_dir = fig_output_dir)

    # Next, load the postprocessors for history matching
    # Load the single point sampler for strain
    ps_reader = MOOSEPointSamplerReader()
    # There is only one strain point sampler now. It should be the first variable in the output file.
    ps_reader.read(folder = output_dir, variable_index = 1)

    simulation_dataframe = ps_reader.to_analyzer()
    # CAUTION: the simulation dataframe does not contain a valid start_time! Need to set it here.
    simulation_dataframe.start_time = timestepper_profile.start_time
    
    return [simulation_dataframe] # Return as a list to maintain type hint

#%% 4. Input parameters for history matching
frac_loc = 14888

if __name__ == "__main__":
    # Define a default set of parameters for a test run
    default_params = {
        'matrix_perm': 1e-18,
        'srv_perm': 1e-15,
        'fracture_perm': 1e-13,
        'srv_length_ft': 400,
        'srv_height_ft': 50,
        'hf_length_ft': 250,
        'hf_height_ft': 0.2
    }

    # Generate a unique instance ID for this run
    run_instance_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("--- Starting Simulation with Default Parameters ---")
    print(f"Instance ID: {run_instance_id}")
    print("Parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")

    # Run the model builder with the default parameters
    # The simulation_results will be a list containing one Data1D_MOOSEps object.
    simulation_results = model_builder_with_parameters_single_frac(
        instance_id=run_instance_id,
        **default_params
    )

    print("\n--- Simulation Finished. Plotting Comparison ---")

    # Extract the single simulation result
    if not simulation_results:
        print("Error: Simulation did not return any results.")
    else:
        sim_result = simulation_results[0]

        # Get observed data at the specified depth
        obs_data_array = DSSdata_copy.get_value_by_depth(frac_loc)

        if obs_data_array is not None:
            # Create a Data1D object for the observation data to use its methods
            obs_gauge = Data1DGauge()
            obs_gauge.data = obs_data_array * 1e-6 # Convert from microstrain to strain
            obs_gauge.taxis = DSSdata_copy.taxis
            obs_gauge.start_time = DSSdata_copy.start_time
            obs_gauge.name = f"Observed Strain at {frac_loc} ft"

            # Crop the simulation result to match the observation time range
            sim_result.select_time(obs_gauge.start_time, obs_gauge.get_end_time(use_timestamp=True))

            # Interpolate observation data to simulation time axis
            obs_gauge.interpolate(sim_result.taxis, sim_result.start_time)

            # Plotting the comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            sim_result.plot(ax=ax, use_timestamp=True, label=f'Simulated Strain (yy)')
            obs_gauge.plot(ax=ax, use_timestamp=True, label=f'Measured Strain')

            ax.set_title(f'Strain Comparison at Depth {frac_loc:.2f} ft')
            ax.set_ylabel('Strain (yy)')
            ax.set_xlabel('Time')
            ax.legend()

            # Format the parameter string for display on the plot
            param_str = '\n'.join([f'{k}: {v:.2e}' for k, v in default_params.items()])
            plt.text(0.05, 0.95, param_str, transform=ax.transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save the figure
            fig_path = os.path.join(fig_output_dir, f"strain_comparison_depth_{int(frac_loc)}.png")
            plt.savefig(fig_path)
            print(f"Comparison plot saved to: {fig_path}")
            plt.show()
            plt.close(fig)
        else:
            print(f"Warning: Could not retrieve observation data for depth {frac_loc}. Skipping plot.")
