# In this script, I will use a simple degradation version of HMM to run the simulation,
# and compare the results with the original HMM model.

# In this script, I will call 3 models
# 1. the original HMM model, full 2.5 yr + 4 days of simulation
# 2. the degradation HMM model, full 2.5 yr + 4 days of simulation
# 3. the degradation HMM model, only 4 days of simulation [Done in fiberis]

# Then I will compare the results of these 3 models.
# Shenyao Jin, shenyaojin@mines.edu

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.core2D import Data2D
from typing import List
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper, InitialConditionConfig
)
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner

def build_hmm_full(**kwargs):
    # A baseline HMM model with two-way coupling between pressure and stress.
    # Define default parameters
    conversion_factor = 0.3048  # feet to meters

    # "data/fiberis_format/post_processing/injection_pressure_full_profile.npz" <- injection pressure profile
    # Load gauge data for MOOSE, I have already packed the data in fiberis format.
    gauge_data_for_moose = Data1DGauge()
    gauge_data_for_moose.load_npz("data/fiberis_format/post_processing/injection_pressure_full_profile.npz")

    # Load DSS data so that I can crop the time range accordingly.
    DSS_data = Data2D()
    DSS_data.load_npz("data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz")
    gauge_data_for_moose.select_time(DSS_data.start_time, DSS_data.get_end_time())
    gauge_data_for_moose.adaptive_downsample(130)
    gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa

    # Start building the model
    builder = ModelBuilder(project_name=kwargs.get("project_name", "BaselineModel"))
    domain_bounds = (- kwargs.get('model_width', 200.0 * conversion_factor),
                     + kwargs.get('model_width', 200.0 * conversion_factor))

    frac_coords = kwargs.get('fracture_y_coords', [0.0 * conversion_factor])
    # If frac_coords is a list (for multi-fracture mesh), use the first element for this single-fracture model's center.
    frac_y_center = frac_coords[0] if isinstance(frac_coords, list) else frac_coords
    domain_length = kwargs.get('model_length', 800.0 * conversion_factor)
    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=frac_coords,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=kwargs.get('nx', 200),
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 100),
        bias_y=kwargs.get('bias_y', 1.1)
    )

    matrix_perm = kwargs.get('matrix_perm', 1e-20)
    srv_perm = kwargs.get('srv_perm', 1e-16)
    fracture_perm = kwargs.get('fracture_perm', 1e-13)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    # Material properties
    matrix_mats = ZoneMaterialProperties(porosity=0.03, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.032,
                                      permeability=srv_perm_str)  # <- Changed here. Physically more reasonable.
    fracture_mats = ZoneMaterialProperties(porosity=0.16, permeability=fracture_perm_str)

    # Define Initial Conditions
    initial_pressure_val = kwargs.get('initial_pressure', 5.17E7)
    initial_pressure_val_srv = kwargs.get('initial_pressure_srv', gauge_data_for_moose.data[0])
    pressure_ic = InitialConditionConfig(
        name="initial_pressure",
        ic_type="ConstantIC",
        variable="pp",
        params={"value": initial_pressure_val}
    )

    pressure_ic_srv_frac = InitialConditionConfig(
        name="initial_pressure_srv_frac",
        ic_type="ConstantIC",
        variable="pp",
        params={"value": initial_pressure_val_srv}
    )

    builder.set_matrix_config(
        MatrixConfig(name="matrix", materials=matrix_mats, initial_conditions=[pressure_ic_srv_frac]))  # changed here

    center_x_val = domain_length / 2.0
    srv_length_ft = kwargs.get('srv_length_ft', 400)
    srv_height_ft = kwargs.get('srv_height_ft', 5)  # <- Changed here. From 50 to 20
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv", length=srv_length_ft * conversion_factor, height=srv_height_ft * conversion_factor,
                  center_x=center_x_val, center_y=frac_y_center, materials=srv_mats,
                  initial_conditions=[pressure_ic_srv_frac]),
        HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor,
                                height=hf_height_ft * conversion_factor, center_x=center_x_val,
                                center_y=frac_y_center, materials=fracture_mats,
                                initial_conditions=[pressure_ic_srv_frac])
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
                                 coordinates=(center_x_val, frac_y_center, 0))

    builder.add_variables([
        "pp",
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

    is_coupled_thermal = kwargs.get('is_coupled_thermal', False)
    if is_coupled_thermal:
        builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=2E-11
    )

    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    # Add post-processors to model builder
    # This part is from baseline_model_builder.py (v1) which provides better post-processing options.
    shift_val_ft = kwargs.get('monitoring_point_shift_ft', 80)

    # Center point sampler, pressure
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="hf_center_pressure_sampler",
            variable="pp",
            point=(center_x_val, frac_y_center, 0)
        )
    )

    # Center point sampler, strain_yy
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="hf_center_strain_yy_sampler",
            variable="strain_yy",
            point=(center_x_val, frac_y_center, 0)
        )
    )

    # Monitoring point sampler, pressure
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="monitor_point_pressure_sampler",
            variable="pp",
            point=(center_x_val + shift_val_ft * conversion_factor, frac_y_center, 0)
        )
    )

    # Monitoring point sampler, strain_yy
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="monitor_point_strain_yy_sampler",
            variable="strain_yy",
            point=(center_x_val + shift_val_ft * conversion_factor, frac_y_center, 0)
        )
    )

    # Line sampler along the fracture, pressure
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="fiber_pressure_sampler",
            variable="pp",
            start_point=(center_x_val + shift_val_ft * conversion_factor,
                         domain_bounds[0] + kwargs.get("start_offset_y", 20) * conversion_factor, 0),
            end_point=(center_x_val + shift_val_ft * conversion_factor,
                       domain_bounds[1] - kwargs.get("end_offset_y", 20) * conversion_factor, 0),
            num_points=kwargs.get("num_fiber_points", 200),
            other_params={'sort_by': 'y'}
        )
    )

    # Line sampler along the fracture, strain_yy
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="fiber_strain_yy_sampler",
            variable="strain_yy",
            start_point=(center_x_val + shift_val_ft * conversion_factor,
                         domain_bounds[0] + kwargs.get("start_offset_y", 20) * conversion_factor, 0),
            end_point=(center_x_val + shift_val_ft * conversion_factor,
                       domain_bounds[1] - kwargs.get("end_offset_y", 20) * conversion_factor, 0),
            num_points=kwargs.get("num_fiber_points", 200),
            other_params={'sort_by': 'y'}
        )
    )

    # Time sequence stepper
    timestepper_profile = Data1DGauge()
    timestepper_profile.load_npz("data/fiberis_format/post_processing/timestepper_profile.npz")
    total_time = timestepper_profile.taxis[-1]
    # Down sample two dataframes to speed up the simulation.

    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(timestepper_profile)

    # Define the time stepper block
    builder.add_executioner_block(
        end_time=total_time,
        dt=3600 * 24 * 5,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control_func
    )

    builder.add_initial_conditions_from_configs()
    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=False, csv=True, exodus_execute_on='FINAL')

    # Plot gauge data for reference
    fig, ax = plt.subplots()
    gauge_data_for_moose.plot(ax=ax, use_timestamp=True)
    # Set title and labels
    ax.set_title("Injection Pressure Profile for HMM Full Model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Injection Pressure (Pa)")
    plt.show()

    print(timestepper_profile.taxis)
    return builder


if __name__ == "__main__":
    # Build and run the full HMM model

    output_folder = "output/0108_degrade/hmm_full"

    # Run 2 instances
    # coupled expansion term
    output_folder_coupled = os.path.join(output_folder, "coupled")
    os.makedirs(output_folder_coupled, exist_ok=True)
    hmm_full_builder = build_hmm_full(project_name="HMM_Full_Degradation_Test",
                                      is_coupled_thermal=True)
    input_file_path_coupled = os.path.join(output_folder_coupled, "hmm_full_degrade_coupled.i")
    hmm_full_builder.generate_input_file(output_filepath=input_file_path_coupled)
    # runner = MooseRunner(
    #     moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
    #     mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    # )
    # # #
    # success, stdout, stderr = runner.run(
    #     input_file_path=input_file_path_coupled,
    #     output_directory=output_folder_coupled,
    #     num_processors=20,
    #     log_file_name="simulation.log",
    #     stream_output=True
    # )

    # not coupled expansion term
    output_folder_decoupled = os.path.join(output_folder, "not_coupled")
    os.makedirs(output_folder_decoupled, exist_ok=True)
    hmm_full_builder = build_hmm_full(project_name="HMM_Full_Degradation_Test",
                                      is_coupled_thermal=False)
    input_file_path_not_coupled = os.path.join(output_folder_decoupled, "hmm_full_degrade_not_coupled.i")
    hmm_full_builder.generate_input_file(input_file_path_not_coupled)
    # runner = MooseRunner(
    #     moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
    #     mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    # )
    # # #
    # success, stdout, stderr = runner.run(
    #     input_file_path=input_file_path_not_coupled,
    #     output_directory=output_folder_decoupled,
    #     num_processors=20,
    #     log_file_name="simulation.log",
    #     stream_output=True
    # )

    from fiberis.moose.templates.baseline_model_generator import build_baseline_model, post_processor_info_extractor, \
        misfit_calculator

    pressure_dataframe_cp, strain_dataframe_cp = post_processor_info_extractor(output_dir=output_folder_coupled)
    pressure_dataframe_ncp, strain_dataframe_ncp = post_processor_info_extractor(output_dir=output_folder_decoupled)

    # Calculate differences
    pressure_diff_percent = pressure_dataframe_cp.copy()
    pressure_diff_percent.data = (pressure_dataframe_cp.data - pressure_dataframe_ncp.data) / pressure_dataframe_ncp.data * 100.0

    strain_diff_percent = strain_dataframe_cp.copy()
    strain_diff_percent.data = (strain_dataframe_cp.data - strain_dataframe_ncp.data) / strain_dataframe_ncp.data * 100.0

    # Load DSS data for time alignment
    DSS_data = Data2D()
    DSS_data.load_npz("data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz")
    start_time = DSS_data.start_time

    pressure_diff_percent.start_time = start_time
    strain_diff_percent.start_time = start_time

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    pressure_diff_percent.plot(ax=ax[0], useTimeStamp=True, cmap = 'bwr')
    ax[0].set_title("Monitoring Point Pressure - HMM Full Model")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Pressure (Pa)")
    strain_diff_percent.plot(ax=ax[1], useTimeStamp=True, cmap='bwr')
    ax[1].set_title("Monitoring Point Strain_yy - HMM Full Model")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Strain_yy")
    plt.tight_layout()
    plt.show()

    print(np.max(pressure_diff_percent.data[:, 1:-1]))
    print(np.max(strain_diff_percent.data[:, 1:-1]))