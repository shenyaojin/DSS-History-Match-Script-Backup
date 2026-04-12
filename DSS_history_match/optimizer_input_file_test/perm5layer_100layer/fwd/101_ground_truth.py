# This model builder is creating a 5-layered model, with a structure
# -----------------
# Matrix
# -----------------
# SRV, low perm (Like 1e-15 m^2)
# -----------------
# Matrix
# -----------------
# SRV, high perm (Like 3e-15 m^2)
# -----------------
# Matrix
# -----------------

import os
import numpy as np
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    CasingConfig, CasingLayerConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
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
    gauge_data_for_moose.load_npz("scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100layer/data/interference.npz")

    gauge_data_for_moose.adaptive_downsample(130)
    gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa
    initial_pressure_val = gauge_data_for_moose.data[0]

    # 2. Define Material Properties for Each Layer
    # Permeability must be provided as a 9-component tensor string for 2D/3D simulations.
    caprock_perm = 1e-18
    sandstone_perm = 3e-15
    shale_perm = 1e-15

    matrix_perm = 1e-18
    caprock_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=f"{caprock_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}",
        youngs_modulus=3.5e10, poissons_ratio=0.25
    )
    sandstone_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=f"{sandstone_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}",
        youngs_modulus=3.5e10, poissons_ratio=0.25
    )
    shale_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=f"{shale_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}",
        youngs_modulus=3.5e10, poissons_ratio=0.25
    )

    # 3. Define the Layers from Top to Bottom
    layers = [
        CasingLayerConfig(name="caprock", height=30, materials=caprock_mats),
        CasingLayerConfig(name="low_perm_srv", height=4, materials=shale_mats),
        CasingLayerConfig(name="caprock2", height=30, materials=caprock_mats),
        CasingLayerConfig(name="high_perm_srv", height=6, materials=sandstone_mats),
        CasingLayerConfig(name="caprock3", height=30, materials=caprock_mats)
    ]

    # 4. Create the Main Casing Configuration
    casing_config = CasingConfig(
        name="VerticalInjectionSite",
        layers=layers,
        injection_well_name="injection_well",
        injection_well_x_coord=45
    )
    builder.add_casing_config(casing_config)

    # 5. Build the Layered Mesh
    domain_length = 100
    builder.build_mesh_for_casing_model(
        domain_length=domain_length,
        nx=200,
        ny=200 # Total elements in Y for the whole domain
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
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                             biot_coefficient=biot_coeff)

    # 8. Define Fluid and Poromechanics Materials
    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=1e-11  # This is a global fallback
    )

    # 9. Define Boundary Conditions
    total_height = sum(layer.height for layer in layers)
    y_min, y_max = -total_height / 2.0, total_height / 2.0

    # Add the pressure curve as a MOOSE function
    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

    # Define the injection well as a linear (vertical) boundary condition across the full height
    # This captures the entire wellbore at x=45
    builder.add_linear_pressure_boundary(
        boundary_name="injection_well",
        bottom_left=(45, y_min, 0),
        top_right=(45, y_max, 0),
        pressure_function_name="injection_pressure_func"
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

    builder.add_standard_tensor_aux_vars_and_kernels(
        {"stress": "stress", "total_strain": "strain", "strain_rate": "strain_rate"})

    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="hf_center_pressure_sampler",
            variable="pp",
            point=(60, 0, 0)
        )
    )

    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="observation_disp",
            variable="disp_y",
            start_point = (60, -25, 0),
            end_point = (60, 25, 0),
            num_points = 500, # 0.1 m & 0.3 ft which is the fiber
            other_params={'sort_by': 'y'}
        )
    )

    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="observation_strain",
            variable="strain_yy",
            start_point=(60, -25, 0),
            end_point=(60, 25, 0),
            num_points=500,  # 0.1 m & 0.3 ft which is the fiber
            other_params={'sort_by': 'y'}
        )
    )

    total_time = gauge_data_for_moose.taxis[-1] - gauge_data_for_moose.taxis[0]
    timestepper_profile = gauge_data_for_moose.copy()
    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(timestepper_profile)
    builder.add_executioner_block(
        end_time=total_time,
        dt=3600 * 24 * 5,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control_func
    )

    builder.add_initial_conditions_from_configs()
    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=False, csv=True, exodus_execute_on='FINAL')

    return builder


if __name__ == "__main__":
    # Create an output directory for the results
    output_dir = "scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100layer/fwd/output_gt"
    os.makedirs(output_dir, exist_ok=True)

    # Build the model
    model_builder = build_casing_model_for_test()
    model_builder.plot_permeability_map()

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
        num_processors=18,  # Using a smaller number of processors for a simple test
        log_file_name="simulation.log",
        stream_output=True
    )

    if success:
        print("\n--- MOOSE Simulation Completed Successfully ---")
    else:
        print("\n--- MOOSE Simulation Failed ---")

    print("Script finished.")