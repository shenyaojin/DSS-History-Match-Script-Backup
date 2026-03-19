# Full simulation using one-way coupled model
# Perform simulation prod+interference
# Shenyao Jin, 03/13/2026, shenyaojin@mines.edu

import numpy as np
import matplotlib.pyplot as plt
import os

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, PointValueSamplerConfig, LineValueSamplerConfig,
    AdaptiveTimeStepperConfig, TimeSequenceStepper, PostprocessorConfig
)

# ==============================================================================
# PARAMETERS
# ==============================================================================
# Global & Unit Conversion
conversion_factor = 0.3048

# Output and Project
output_dir = "output/0313_validation/interf_run"
project_name = "0313_interf_run"
input_file_name = os.path.join(output_dir, f"{project_name}.i")

fracture_coords = 0
domain_bounds_ft = (-200, 200)
domain_length_ft = 800
nx = 200
ny_per_layer_half = 150
bias_y = 1.03

# Hydraulic Properties
matrix_perm = 1e-20
srv_perm = 1e-16
fracture_perm = 1e-13

porosity_matrix = 0.03
porosity_srv = 0.032
porosity_hf = 0.16

# SRV and Fracture Geometry (in feet)
srv_length_ft = 400
srv_height_ft = 5
hf_length_ft = 250
hf_height_ft = 0.2

# Geomechanics
biot_coeff = 0.7
solid_bulk_compliance = 2E-11

# Fluid Properties
bulk_modulus = 2.2E9
viscosity = 1.0E-3
density0 = 1000.0

# Data Processing
gauge_data_path = "scripts/DSS_history_match/validation/data/interference.npz"
time_stepper_path = "scripts/DSS_history_match/validation/data/interference.npz"
downsample_points = 400

# Post-processing / Wellbore Sampler
well_spacing_ft = 80  # Spacing between producer and monitor well
mesh_y_span_ft = 100  # Distance from fracture to sampler ends
sampler_num_points = 200

# Runner path
moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"

# ==============================================================================

# Derived Parameters
domain_bounds = (domain_bounds_ft[0] * conversion_factor, domain_bounds_ft[1] * conversion_factor)
domain_length = domain_length_ft * conversion_factor
fracture_y_coords = fracture_coords
center_x_val = domain_length / 2.0

# Model Builder
builder = ModelBuilder(project_name=project_name)

builder.build_stitched_mesh_for_fractures(
    fracture_y_coords=fracture_y_coords,
    domain_bounds=domain_bounds,
    domain_length=domain_length,
    nx=nx,
    ny_per_layer_half=ny_per_layer_half,
    bias_y=bias_y
)

# Material Property strings
matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

matrix_mats = ZoneMaterialProperties(porosity=porosity_matrix, permeability=matrix_perm_str)
srv_mats = ZoneMaterialProperties(porosity=porosity_srv, permeability=srv_perm_str)
fracture_mats = ZoneMaterialProperties(porosity=porosity_hf, permeability=fracture_perm_str)

builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

geometries = [
    SRVConfig(name="srv", length=srv_length_ft * conversion_factor, height=srv_height_ft * conversion_factor,
              center_x=center_x_val, center_y=fracture_y_coords, materials=srv_mats),
    HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor, height=hf_height_ft * conversion_factor,
                            center_x=center_x_val, center_y=fracture_y_coords, materials=fracture_mats)
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
                             coordinates=(center_x_val, fracture_y_coords, 0))

gauge_data = Data1DGauge()
gauge_data.load_npz(gauge_data_path)

builder.add_variables([
    {"name": "pp", "params": {"initial_condition": gauge_data.data[0]}},
    {"name": "disp_x", "params": {"initial_condition": 0}},
    {"name": "disp_y", "params": {"initial_condition": 0}}
])

builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                         biot_coefficient=biot_coeff)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                         biot_coefficient=biot_coeff)
# builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=bulk_modulus, viscosity=viscosity,
                                             density0=density0)
builder.add_fluid_properties_config(fluid_property)
builder.add_poromechanics_materials(
    fluid_properties_name="water",
    biot_coefficient=biot_coeff,
    solid_bulk_compliance=solid_bulk_compliance
)

gauge_data.data *= 6894.76
gauge_data.adaptive_downsample(downsample_points)

builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data)

builder.set_hydraulic_fracturing_bcs(
    injection_well_boundary_name="injection",
    injection_pressure_function_name="injection_pressure_func",
    confine_disp_x_boundaries="left right",
    confine_disp_y_boundaries="top bottom"
)

builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress",
                                                  "total_strain": "strain"})

y_min_mesh = domain_bounds_ft[0] + 20
y_max_mesh = domain_bounds_ft[1] - 20

builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj", variable="pp", point=(center_x_val, fracture_y_coords, 0)))

builder.add_postprocessor(
    LineValueSamplerConfig(
        name="strain_yy_wellbore",
        variable="strain_yy",
        start_point=(center_x_val + well_spacing_ft * conversion_factor, y_min_mesh * conversion_factor, 0),
        end_point=(center_x_val + well_spacing_ft * conversion_factor, y_max_mesh * conversion_factor, 0),
        num_points=sampler_num_points,
        other_params={'sort_by': 'y'}
    )
)

stepper_data = Data1DGauge()
stepper_data.load_npz(time_stepper_path)
stepper_data.adaptive_downsample(140)

print("Total time step = ", len(stepper_data.taxis))

total_time = stepper_data.taxis[-1] - stepper_data.taxis[0]
dt_control_func = TimeSequenceStepper()
dt_control_func.from_data1d(stepper_data)

builder.add_executioner_block(
    end_time=total_time,
    dt=3600 * 24 * 5,
    time_stepper_type='TimeSequenceStepper',
    stepper_config=dt_control_func
)

builder.add_preconditioning_block(active_preconditioner='mumps')
builder.add_outputs_block(exodus=True, csv=True)

builder.generate_input_file(input_file_name)

builder.plot_geometry()

# # Run the simulation
# runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
# success, stdout, stderr = runner.run(
#     input_file_path=input_file_name,
#     output_directory=output_dir,
#     num_processors=20,
#     log_file_name="simulation.log",
#     stream_output=True
# )