# scripts/DSS_history_match/112_casing_model_example.py
# This script demonstrates how to build and run a layered casing model using the new CasingConfig workflow.
# It defines three distinct geological layers, builds the corresponding mesh, sets up a
# basic poromechanics simulation, generates the MOOSE input file, and runs the simulation.
# Shenyao Jin, 02/04/2026

import os
import numpy as np
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    CasingConfig, CasingLayerConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig
)
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

def build_casing_model_for_test():
    """
    Builds a 3-layer casing model for MOOSE validity testing.
    This version has all post-processors removed.
    """
    # 1. Initialize the ModelBuilder
    project_name = "CasingModelValidityTest"
    builder = ModelBuilder(project_name=project_name)
    
    # Load and preprocess gauge data for pressure curve and initial conditions
    gauge_data_for_moose = Data1DGauge()
    gauge_data_for_moose.load_npz("data/fiberis_format/prod/gauges/pressure_g1.npz")
    
    # This part is just for getting the time range, you can replace with actual data if needed
    from fiberis.analyzer.Data2D.core2D import Data2D
    DSS_data = Data2D()
    DSS_data.load_npz("data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz")
    gauge_data_for_moose.select_time(DSS_data.start_time, DSS_data.get_end_time())
    
    gauge_data_for_moose.adaptive_downsample(130)
    gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa
    initial_pressure_val = gauge_data_for_moose.data[0]

    # 2. Define Material Properties for Each Layer
    # Permeability must be provided as a 9-component tensor string for 2D/3D simulations.
    caprock_perm = 1e-20
    sandstone_perm = 1e-14
    shale_perm = 1e-18

    caprock_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=f"{caprock_perm} 0 0 0 {caprock_perm} 0 0 0 {caprock_perm}",
        youngs_modulus=4e10, poissons_ratio=0.3
    )
    sandstone_mats = ZoneMaterialProperties(
        porosity=0.15, permeability=f"{sandstone_perm} 0 0 0 {sandstone_perm} 0 0 0 {sandstone_perm}",
        youngs_modulus=2.5e10, poissons_ratio=0.25
    )
    shale_mats = ZoneMaterialProperties(
        porosity=0.03, permeability=f"{shale_perm} 0 0 0 {shale_perm} 0 0 0 {shale_perm}",
        youngs_modulus=3.5e10, poissons_ratio=0.28
    )

    # 3. Define the Layers from Top to Bottom
    layers = [
        CasingLayerConfig(name="caprock", height=50.0, materials=caprock_mats),
        CasingLayerConfig(name="sandstone_reservoir", height=100.0, materials=sandstone_mats),
        CasingLayerConfig(name="shale_basement", height=75.0, materials=shale_mats)
    ]

    # 4. Create the Main Casing Configuration
    casing_config = CasingConfig(
        name="VerticalInjectionSite",
        layers=layers,
        injection_well_name="injection_well",
        injection_well_x_coord=250.0
    )
    builder.add_casing_config(casing_config)

    # 5. Build the Layered Mesh
    domain_length = 500.0
    builder.build_mesh_for_casing_model(
        domain_length=domain_length,
        nx=50,
        ny=30  # Total elements in Y for the whole domain
    )
    
    # 6. Define Primary Variables and Global Parameters
    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": initial_pressure_val}},
        {"name": "disp_x", "params": {"initial_condition": 0}},
        {"name": "disp_y", "params": {"initial_condition": 0}}
    ])
    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    # 7. Define Kernels (Physics)
    biot_coeff = 0.8
    builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0, biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1, biot_coefficient=biot_coeff)

    # 8. Define Fluid and Poromechanics Materials
    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=1e-11 # This is a global fallback
    )

    # 9. Define Boundary Conditions
    total_height = sum(layer.height for layer in layers)
    y_min, y_max = -total_height / 2.0, total_height / 2.0
    
    # Add the pressure curve as a MOOSE function
    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

    # The injection well is now a nodeset, and global boundaries are defined on the base mesh
    builder.add_boundary_condition(
        name="injection_pressure", bc_type="FunctionDirichletBC", variable="pp",
        boundary_name="injection_well", params={"function": "injection_pressure_func"}
    )
    # Set zero-stress (Neumann) boundary conditions for displacement
    builder.add_boundary_condition(
        name="confine_x", bc_type="NeumannBC", variable="disp_x",
        boundary_name="left right", params={"value": 0}
    )
    builder.add_boundary_condition(
        name="confine_y", bc_type="NeumannBC", variable="disp_y",
        boundary_name="top bottom", params={"value": 0}
    )

    # 10. Executioner and Outputs
    builder.add_executioner_block(end_time=gauge_data_for_moose.taxis[-1], time_stepper_type='ConstantDT', dt=3600)
    builder.add_outputs_block(exodus=True, csv=False) # No CSV as there are no samplers

    return builder

if __name__ == "__main__":
    # Create an output directory for the results
    output_dir = os.path.join("output", "112_casing_model_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the model
    model_builder = build_casing_model_for_test()

    # Generate the MOOSE input file
    input_file_path = os.path.join(output_dir, "casing_model_test.i")
    print(f"Generating MOOSE input file at {input_file_path}...")
    model_builder.generate_input_file(input_file_path)

    # Initialize the MOOSE Runner
    runner = MooseRunner(
        moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
        mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    )

    # Run the simulation
    print("\n--- Running MOOSE Simulation ---")
    success, stdout, stderr = runner.run(
        input_file_path=input_file_path,
        output_directory=output_dir,
        num_processors=4, # Using a smaller number of processors for a simple test
        log_file_name="simulation.log",
        stream_output=True
    )

    if success:
        print("\n--- MOOSE Simulation Completed Successfully ---")
    else:
        print("\n--- MOOSE Simulation Failed ---")

    print("Script finished.")