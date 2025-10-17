# This script is to repeat Victor's input generation process.
# Designed by Shenyao using fiberis.
# shenyaojin@mines.edu, 08/08/2025
# Log of changes:
# 08/08/2025: Add mesh refinement step -- Shenyao
# 08/19/2025: change parameters, make it physical? No, just remove the redundant Darcy kernel.

import os
import numpy as np
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, PointValueSamplerConfig, LineValueSamplerConfig,
    AdaptiveTimeStepperConfig, TimeStepperFunctionConfig, PostprocessorConfig
)
from fiberis.analyzer.Data1D.core1D import Data1D
import datetime

#%% Specify loc
output_dir = "output/0818_vf_reproduce_remove_darcy_kernel"
os.makedirs(output_dir, exist_ok=True)
input_file_name = os.path.join(output_dir, "single_simulation.i")

#%% Create the model
builder = ModelBuilder(project_name='vf_reproduce_remove_darcy_kernel')

#%% Generate mesh

# 1. mesh
fracture_y_coords = 0
domain_bounds = (-500, 500)
domain_length = 1000

builder.build_stitched_mesh_for_fractures(
    fracture_y_coords=fracture_y_coords,
    domain_bounds=domain_bounds,
    domain_length=domain_length,
    nx = 200,
    ny_per_layer_half=40,
    bias_y=1.3
)

# 2. materials and properties

# generate data1d files for material properties
synthetic_timestamp = datetime.datetime(2025, 1, 1, 0, 0, 0)
time = np.array([0, 379450, 379452, 461220, 461221])

fracture_diffusivity = np.array([33.5, 33.5, 5, 5, 5]) * 1e6
srv_diffusivity = np.array([0.00082, 0.00082, 0.0023, 0.0023, 0.0023]) * 1e6


fracture_perm = fracture_diffusivity * 1e-12 / 5 # make it ends up in 1e-12, same as Victor's
srv_perm = srv_diffusivity * 1e-17 / 0.0023 # make it ends up in 1e-17, same as Victor's

# generate data1d files for time dependent material properties
print("Generating data1d files for material properties...")
fracture_perm_data1d = Data1D(
    data=fracture_perm,
    taxis=time,
    start_time=synthetic_timestamp,
    name="fracture_perm")


srv_perm_data1d = Data1D(
    data=srv_perm,
    taxis=time,
    start_time=synthetic_timestamp,
    name="srv_perm")

savepath = "scripts/fiberis_moose_generator/repeat_vf/saved_files"

frac_perm_path = os.path.join(savepath, f"fracture_perm_data1d.npz")
srv_perm_path = os.path.join(savepath, f"srv_perm_data1d.npz")

fracture_perm_data1d.savez(frac_perm_path)
srv_perm_data1d.savez(srv_perm_path)

# define material properties.
matrix_mats = ZoneMaterialProperties(
    porosity=0.01,
    permeability="'1E-15 0 0  0 1E-15 0  0 0 1E-15'") # matrix
srv_mats = ZoneMaterialProperties(
    porosity=0.1,
    permeability=srv_perm_path)
fracture_mats = ZoneMaterialProperties(
    porosity=0.1,
    permeability=frac_perm_path)

# Set up the material properties to the model
builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

center_x_val = domain_length / 2

# --- Manual Geometry and Block ID Assignment to replicate example_VF.i ---
# The for-loop logic is too simple for the complex, overlapping geometry
# and non-sequential block IDs in the target .i file.
# We now manually define each geometry and add it to the builder in the
# exact order with the exact block_id from the file we want to replicate.

# 1. Define the config objects for each geometry
srv1_config = SRVConfig(
    name="srv1",
    length=300, height=16, center_x=center_x_val, center_y=fracture_y_coords, materials=srv_mats
)
srv2_config = SRVConfig(
    name="srv2",
    length=100, height=24, center_x=center_x_val, center_y=fracture_y_coords, materials=srv_mats
)
srv3_config = SRVConfig(
    name="srv3",
    length=200, height=20, center_x=center_x_val, center_y=fracture_y_coords, materials=srv_mats
)
hf_config = HydraulicFractureConfig(
    name="hf",
    length=250, height=0.02, center_x=center_x_val, center_y=fracture_y_coords, materials=fracture_mats
)

# Set up the fluid properties
fluid_props = SimpleFluidPropertiesConfig(
    name="water",
    bulk_modulus=2.2E9,
    viscosity=1.0E-3,
    density0=1000.0
)

# 2. Add configs to the builder so it knows about the materials for later steps (like Materials block generation)
builder.add_srv_config(srv1_config)
builder.add_srv_config(srv2_config)
builder.add_srv_config(srv3_config)
builder.add_fracture_config(hf_config)

# 3. Manually add the zones in the correct order with the correct block IDs from example_VF.i
# Operation 1: SRV (our srv1) -> block_id = 1
builder.add_srv_zone_2d(srv1_config, target_block_id=1)
# Operation 2: SRV2 (our srv2) -> block_id = 3
builder.add_srv_zone_2d(srv2_config, target_block_id=3)

# Refine block 3 immediately after it's created to ensure its
# thin leftover regions are populated with elements.
builder.refine_blocks(op_name="refine_srv2", block_ids=[3], refinement_levels=1)

# Operation 3: SRV3 (our srv3) -> block_id = 4
builder.add_srv_zone_2d(srv3_config, target_block_id=4)
# Operation 4: fracture (our hf) -> block_id = 2
builder.add_hydraulic_fracture_2d(hf_config, target_block_id=2)

# Add injection and production wells
builder.add_nodeset_by_coord(nodeset_op_name="injection_well",
                             new_boundary_name="injection_well",
                             coordinates=(center_x_val, fracture_y_coords, 0)) # 500, 0, 0

builder.add_nodeset_by_coord(nodeset_op_name="production_well",
                             new_boundary_name="production_well",
                             coordinates=(center_x_val + 110, fracture_y_coords, 0)) # 610, 0, 0 (See Victor's file)

#%% 3. Add physical fields.
builder.add_variables([
    {"name": "pp", "params": {"initial_condition": 26.4E6}},
    {"name": "disp_x", "params": {"scaling": 1e-10}},
    {"name": "disp_y", "params": {"scaling": 1e-10}}
])

# Physics

# a. dictator
builder.set_porous_flow_dictator(
    dictator_name="dictator",
    porous_flow_variables="pp",
    num_fluid_phases=1,
    num_fluid_components=1
) # dictator for porous flow, how many phase and components in the fluid

# b. global parameters
builder.add_global_params(
    {"PorousFlowDictator": "dictator",
     "displacements": "disp_x disp_y",
     }
)

# c. Add kernels for geophysical processes
biot_coefficient = 0.7

# kernels. there are 7 kernels in total theoretically.
builder.add_time_derivative_kernel(
    variable="pp"
)
builder.add_porous_flow_darcy_base_kernel(
    kernel_name="flux",
    variable="pp"
) # start from flux (see Victor's file)
builder.add_stress_divergence_tensor_kernel(
    kernel_name="grad_stress_x",
    variable="disp_x",
    component=0
)
builder.add_stress_divergence_tensor_kernel(
    kernel_name="grad_stress_y",
    variable="disp_y",
    component=1
)
builder.add_porous_flow_effective_stress_coupling_kernel(
    kernel_name="effective_stress_x",
    variable="disp_x",
    component=0,
    biot_coefficient=biot_coefficient
)
builder.add_porous_flow_mass_volumetric_expansion_kernel(
    kernel_name="vol_strain_rate_water",
    variable="pp",
    fluid_component=0
)

