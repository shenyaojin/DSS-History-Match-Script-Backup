# This script will implement the discussion on misfit functions and other data processing techniques.
# See my notion. -- Shenyao

# - linear scaling for DSS data. Because DSS is not perfectly coupled with the casing. Let's do 10x
# - remove one SRV (SRV2)
# - change the misfit func, to a manually picked area.
# - sensitivity  test on length

# This script, still will only use 1 fracture.
# Also, this one will not include an optimizer, just to plot the results.

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

print(injection_gauge_pressure_copy.start_time)
#%% 3. Define the model builder function
conversion_factor = 0.3048

def model_builder_parameters_single_frac(**kwargs) -> ModelBuilder:
    """
    Build a fiberis model for history matching using single hydraulic fracture model.
    It's slightly revised from the previous version, especially in post-processing.
    -- Shenyao Jin, 11/01/2025
    :param kwargs: Those are the parameters for history matching. See implementation for details.
    :return: A list of Data1D objects know as the postprocessors from the simulation. Can be used for misfit calculation.
    """
    # Define the output directory for the model, which would be good for reproducibility.

    # Build the model
    builder = ModelBuilder(project_name=project_name)

    # Mesh and geometry
    frac_coords = 0  # Should be in the center of the model
    domain_bounds = (- kwargs.get('model_width', 50.0 * conversion_factor),
                     + kwargs.get('model_width', 50.0 * conversion_factor))

    domain_length = kwargs.get('model_length', 400.0 * conversion_factor)

    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=frac_coords,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=kwargs.get('nx', 200),
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 80),
        bias_y=kwargs.get('bias_y', 1.2)
    )

    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-16)
    fracture_perm = kwargs.get('fracture_perm', 1e-14)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
    fracture_mats = ZoneMaterialProperties(porosity=0.16, permeability=fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft2 = kwargs.get('srv_length_ft', 285)
    srv_height_ft2 = kwargs.get('srv_height_ft', 5)
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
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

    # Add -2 ft -> + 2.5 ft range channels

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

    return builder

builder = model_builder_parameters_single_frac()
builder.plot_geometry()

time_now = datetime.datetime.now()
date_time_str_tmp = time_now.strftime("%Y%m%d_%H%M%S")
instance_id = date_time_str_tmp

model_output_dir = os.path.join(moose_output_dir, f"model_instance_{instance_id}")
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)
input_file_path = os.path.join(model_output_dir, project_name + "_input.i")
builder.generate_input_file(input_file_path)

runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
success, stdout, stderr = runner.run(
    input_file_path=input_file_path,
    output_directory=model_output_dir,
    num_processors=20,
    log_file_name="simulation.log",
    stream_output=True
)