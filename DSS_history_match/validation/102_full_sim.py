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

# global params
conversion_factor = 0.3048

# model builder
output_dir = "output/0313_validation/full_run"
input_file_name = os.path.join(output_dir, "0313_full_run.i")
builder = ModelBuilder(project_name="0313_full_run")

fracture_coords = 0
domain_bounds = (-300 * conversion_factor, 600 * conversion_factor)
domain_length = 600 * conversion_factor
fracture_y_coords = fracture_coords

builder.build_stitched_mesh_for_fractures(
        fracture_y_coords = fracture_y_coords,
        domain_bounds = domain_bounds,
        domain_length = domain_length,
        nx = 200,
        ny_per_layer_half=80,
        bias_y=1.2
)

matrix_perm = 1e-17
srv_perm = 1e-15
srv_perm2 = srv_perm / 5
fracture_perm = 1e-12

center_x_val = domain_length / 2.0
srv_length_ft2 = 400
srv_height_ft2 = 50
srv_length_ft1 = 250
srv_height_ft1 = 150
hf_length_ft = 250
hf_height_ft = 0.2

matrix_perm_str = f"'{matrix_perm} 0 0  0 {matrix_perm} 0  0 0 {matrix_perm}'"
srv_perm_str = f"'{srv_perm} 0 0  0 {srv_perm} 0  0 0 {srv_perm}'"
srv_perm_str2 = f"'{srv_perm2} 0 0  0 {srv_perm2} 0  0 0 {srv_perm2}'"
fracture_perm_str = f"'{fracture_perm} 0 0  0 {fracture_perm} 0  0 0 {fracture_perm}'"

matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
srv_mats2 = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str2)
fracture_mats = ZoneMaterialProperties(porosity=0.14, permeability=fracture_perm_str)

builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

geometries = [
        SRVConfig(name = "srv_tall", length = srv_length_ft1 * conversion_factor, height= srv_height_ft1 * conversion_factor, center_x = center_x_val, center_y = fracture_y_coords, materials = srv_mats2),
        SRVConfig(name = "srv_wide", length = srv_length_ft2 * conversion_factor, height= srv_height_ft2 * conversion_factor, center_x = center_x_val, center_y = fracture_y_coords, materials = srv_mats),
        HydraulicFractureConfig(name="hf", length = hf_length_ft * conversion_factor, height = hf_height_ft * conversion_factor, center_x = center_x_val, center_y = fracture_y_coords, materials = fracture_mats)
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

builder.add_nodeset_by_coord(nodeset_op_name = "injection", new_boundary_name = "injection", coordinates = (center_x_val, fracture_y_coords, 0))

gauge_data = Data1DGauge()
gauge_data.load_npz("scripts/DSS_history_match/validation/data/full.npz")

builder.add_variables([
    {"name": "pp", "params": {"initial_condition": gauge_data.data[0]}},  # Initial pressure ~ 7500 psi in Pa.
    {"name": "disp_x", "params": {"initial_condition": 0}},
    {"name": "disp_y", "params": {"initial_condition": 0}}
])

builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

boit_coeff = 0.7
builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                         biot_coefficient=boit_coeff)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                         biot_coefficient=boit_coeff)
builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
builder.add_fluid_properties_config(fluid_property)
builder.add_poromechanics_materials(
    fluid_properties_name="water",
    biot_coefficient=boit_coeff,
    solid_bulk_compliance=2E-11
)

gauge_data.data *= 6894.76
gauge_data.adaptive_downsample(400)

builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d= gauge_data)

builder.set_hydraulic_fracturing_bcs(
    injection_well_boundary_name="injection",
    injection_pressure_function_name="injection_pressure_func",
    confine_disp_x_boundaries="left right",
    confine_disp_y_boundaries="top bottom"
)

builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress",
                                                  "total_strain": "strain"})

y_min_mesh = fracture_y_coords - 100
y_max_mesh = fracture_y_coords + 100
shift_val = 80 # Spacing between producer and monitor well

builder.add_postprocessor(
    LineValueSamplerConfig(
        name="strain_yy_wellbore",
        variable="strain_yy",
        start_point=(center_x_val + shift_val * conversion_factor, y_min_mesh, 0),
        end_point=(center_x_val + shift_val * conversion_factor, y_max_mesh, 0),
        num_points=200,
        other_params={'sort_by': 'y'}
    )
)  # Strain post processor along the wellbore

