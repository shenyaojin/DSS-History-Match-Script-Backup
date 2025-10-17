# This script is to repeat Victor's input generation process.
# Designed by Shenyao using fiberis.
# shenyaojin@mines.edu, 08/08/2025
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


#%% Specify the file locations
output_dir = "output/0808_vf_reproduce"
os.makedirs(output_dir, exist_ok=True)

input_file_name = os.path.join(output_dir, "single_frac_simulation_VF.i")

#%% Create the model
builder = ModelBuilder(project_name="single_frac_simulation_VF")

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
    ny_per_layer_half=80,
    bias_y=1.1 # refine it.
)

# 2. materials and properties

# generate data1d files for material properties
synthetic_timestamp = datetime.datetime(2025, 1, 1, 0, 0, 0)
time = np.array([0, 379450, 379452, 461220, 461221])

fracture_diffusivity = np.array([33.5, 33.5, 5, 5, 5])
srv_diffusivity = np.array([0.00082, 0.00082, 0.0023, 0.0023, 0.0023])


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
    permeability="'1E-20 0 0  0 1E-20 0  0 0 1E-20'") # matrix
srv_mats = ZoneMaterialProperties(
    porosity=0.1,
    permeability=srv_perm_path)
fracture_mats = ZoneMaterialProperties(
    porosity=0.1,
    permeability=frac_perm_path)

# Set up the material properties to the model
builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

center_x_val = domain_length / 2

# Set up the hydraulic fracture and SRV geometries
# Order is critical to prevent smaller blocks from being completely overwritten.
# Processing from largest area to smallest area is a robust strategy.
geometries = [
    SRVConfig(
        name="srv1", # Area = 300*16 = 4800
        length=300,
        height=16,
        center_x=center_x_val,
        center_y=fracture_y_coords,
        materials=srv_mats
        ),
    SRVConfig(
        name="srv3", # Area = 200*20 = 4000
        length=200,
        height=20,
        center_x=center_x_val,
        center_y=fracture_y_coords,
        materials=srv_mats
    ),
    SRVConfig(
        name="srv2", # Area = 100*24 = 2400
        length=100,
        height=24,
        center_x=center_x_val,
        center_y=fracture_y_coords,
        materials=srv_mats
        ),
    HydraulicFractureConfig(
        name="hf", # Area = 250*0.02 = 5
        length=250,
        height=0.02,
        center_x=center_x_val,
        center_y=fracture_y_coords,
        materials=fracture_mats
    )
]

# Set up the fluid properties
fluid_props = SimpleFluidPropertiesConfig(
    name="water",
    bulk_modulus=2.2E9,
    viscosity=1.0E-3,
    density0=1000.0
)

# The order of geometry creation is critical. Sorting by size was causing smaller blocks
# to completely overwrite larger blocks, leading to missing block IDs.
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
# Operation 3: SRV3 (our srv3) -> block_id = 4
builder.add_srv_zone_2d(srv3_config, target_block_id=4)
# Operation 4: fracture (our hf) -> block_id = 2
builder.add_hydraulic_fracture_2d(hf_config, target_block_id=2)

# next_block_id = 1
# for geom_config in sorted_geometries:
#     if isinstance(geom_config, SRVConfig):
#         builder.add_srv_config(geom_config)
#         builder.add_srv_zone_2d(geom_config, target_block_id=next_block_id)
#     elif isinstance(geom_config, HydraulicFractureConfig):
#         builder.add_fracture_config(geom_config)
#         builder.add_hydraulic_fracture_2d(geom_config, target_block_id=next_block_id)
#     next_block_id += 1

# Add injection and production wells
builder.add_nodeset_by_coord(nodeset_op_name="injection_well",
                             new_boundary_name="injection_well",
                             coordinates=(center_x_val, fracture_y_coords, 0)) # 500, 0, 0

builder.add_nodeset_by_coord(nodeset_op_name="production_well",
                             new_boundary_name="production_well",
                             coordinates=(center_x_val + 110, fracture_y_coords, 0)) # 610, 0, 0 (See Victor's file)

# 3. Add physical fields

# Variables to be solved in the model
# The variables to be added: borehole pressure (pp), displacement in x (disp_x), and displacement in y (disp_y).
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

# d. fluid.
builder.add_fluid_properties_config(fluid_props)

# e. Fill porous material with fluid(water).
builder.add_poromechanics_materials(
    fluid_properties_name="water",
    biot_coefficient=biot_coefficient,
    solid_bulk_compliance=2E-11
)

# f. Add injection curve from a csv file.
inj_filepath = "scripts/fiberis_moose_generator/repeat_vf/saved_files/t_y_data.csv"
# read csv file and load it into two numpy arrays
import csv
with open(inj_filepath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    t_data, y_data = zip(*reader)

t_data = np.array(t_data[1:], dtype=float)
y_data = np.array(y_data[1:], dtype=float)

pressure_curve_dataframe = Data1D(
    data=y_data,
    taxis=t_data,
    start_time=synthetic_timestamp,
    name="injection_pressure_curve"
)

# Add this pressure curve to [function] block
builder.add_piecewise_function_from_data1d(
    name="injection_pressure_func",
    source_data1d=pressure_curve_dataframe
)


# g. Add boundary conditions (including injection well)
builder.set_hydraulic_fracturing_bcs(
    injection_well_boundary_name="injection_well",
    injection_pressure_function_name="injection_pressure_func",
    confine_disp_x_boundaries="left right",
    confine_disp_y_boundaries="top bottom",
)

# h. Add aux kernels, which are generated from templates in model_builder
builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

# 4. postprocessors
builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_inj",
    variable="pp",
    point=(center_x_val, fracture_y_coords, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_prod",
    variable="pp",
    point=(center_x_val + 110, fracture_y_coords, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_mon1",
    variable="pp",
    point=(435, 5, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_mon2",
    variable="pp",
    point=(545, 2, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_mon3",
    variable="pp",
    point=(545, 5, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_mon4",
    variable="pp",
    point=(545, 10, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_mon5",
    variable="pp",
    point=(545, 50, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_mon6",
    variable="pp",
    point=(545, 200, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="pp_mon7",
    variable="pp",
    point=(435, 2, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="strain_yy_inj",
    variable="strain_yy",
    point=(center_x_val, fracture_y_coords, 0)
))

builder.add_postprocessor(PointValueSamplerConfig(
    name="strain_yy_prod",
    variable="strain_yy",
    point=(center_x_val + 110, fracture_y_coords, 0)
))

# Permeability monitoring
# I'm not sure whether it's working or not.
builder.add_postprocessor(PostprocessorConfig(
    name="perm_srv1",
    pp_type='ElementAverageValue',
    params={'variable': 'scalar_perm_srv1', 'block': 'srv1'}
))

builder.add_postprocessor(PostprocessorConfig(
    name="perm_srv2",
    pp_type='ElementAverageValue',
    params={'variable': 'scalar_perm_srv2', 'block': 'srv2'}
))

builder.add_postprocessor(PostprocessorConfig(
    name="perm_srv3",
    pp_type='ElementAverageValue',
    params={'variable': 'scalar_perm_srv3', 'block': 'srv3'}
))

builder.add_postprocessor(PostprocessorConfig(
    name="perm_hf",
    pp_type='ElementAverageValue',
    params={'variable': 'scalar_perm_hf', 'block': 'hf'}
))

builder.add_postprocessor(LineValueSamplerConfig(
    name="hf_line",
    variable="pp strain_xx strain_yy",
    start_point=(
        center_x_val - 125, fracture_y_coords, 0),  # start point of the line
    end_point=(center_x_val + 125, fracture_y_coords, 0),
    num_points=500,
    other_params={'sort_by': 'x'}
))

builder.add_postprocessor(LineValueSamplerConfig(
    name="line_prod",
    variable="pp strain_xx strain_yy",
    start_point=(610, -500, 0),  # start point of the line
    end_point=(610, 500, 0),
    num_points=1000,
    other_params={'sort_by': 'y'}
))

# 5. Solver and time stepper
# Set up the adaptive time stepper to replicate the settings from example_VF.i.

# Define the four PiecewiseConstant functions for timestep control
constant_step_1 = TimeStepperFunctionConfig(
    name='constant_step_1',
    x_values=[204600, 300000, 310000, 340000, 350000, 353000, 353200, 353400, 353800, 354000, 354400, 354600, 354800, 355000, 355600, 355800, 356000, 356100],
    y_values=[200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100]
)

constant_step_2 = TimeStepperFunctionConfig(
    name='constant_step_2',
    x_values=[356200, 356400, 356800, 357000, 357400, 357600, 357800, 358000, 358200, 358400, 358600, 358800, 359000, 359100, 359200],
    y_values=[100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100, 100]
)

adaptive_step = TimeStepperFunctionConfig(
    name='adaptive_step',
    x_values=[359300, 359400, 360000, 360400, 360600, 360800, 361200, 361600, 362000, 362400, 362800, 363000, 363200, 363400, 363600, 363700, 363800, 363900, 364000],
    y_values=[100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100]
)

adaptive_final = TimeStepperFunctionConfig(
    name='adaptive_final',
    x_values=[364200, 364800, 365200, 365800, 366400, 366800, 367400, 367800, 369000, 370000, 371000, 372000, 373000, 374000, 375000, 376000, 377000, 377200, 377400, 377600, 377800, 378000, 378200, 378300, 378400, 378500, 378600, 378800, 379000, 380000, 380400, 381000, 382000],
    y_values=[200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200]
)

# Package the functions into the main adaptive stepper config
adaptive_config = AdaptiveTimeStepperConfig(
    functions=[constant_step_1, constant_step_2, adaptive_step, adaptive_final],
)

# Pass the config to the executioner, replicating the settings from the .i file
builder.add_executioner_block(
    end_time=461221,
    dt=200,
    adaptive_stepper_config=adaptive_config,
)

builder.add_preconditioning_block(active_preconditioner='mumps')
builder.add_outputs_block(exodus=True, csv=True)

# --- 6. Generate Input File ---
builder.generate_input_file(input_file_name)
print(f"\nSuccessfully generated MOOSE input file at: {input_file_name}")

# --- 7. Run Simulation ---
print("\n--- Starting MOOSE Simulation Runner ---")
try:
    moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
    if not os.path.exists(moose_executable):
        print(f"ERROR: MOOSE executable not found at '{moose_executable}'. Please update the path.")
    else:
        runner = MooseRunner(moose_executable_path=moose_executable)
        success, stdout, stderr = runner.run(
            input_file_path=input_file_name,
            output_directory=output_dir,
            num_processors=2,
            log_file_name="simulation.log"
        )
        if success:
            print("\nSimulation completed successfully!")
        else:
            print("\nSimulation failed.")
            print("--- STDERR from MOOSE ---")
            print(stderr)
except Exception as e:
    print(f"\nAn error occurred during the simulation run: {e}")


if __name__ == "__main__":
    pass


