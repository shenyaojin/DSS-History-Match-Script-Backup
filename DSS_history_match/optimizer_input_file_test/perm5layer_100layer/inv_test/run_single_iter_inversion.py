"""
Single-iteration inversion test for the 5-layer perm model.

This script:
1. Generates forward_and_adjoint.i and optimize.i using fiberis
2. Points to the correct measurement CSV (../data/measurement.csv)
3. Runs 1 TAO iteration natively in MOOSE (no SciPy wrapper)

The MOOSE TAO solver handles the full optimization internally.
With tao_max_it=1, TAO does:
  - Initial evaluation: 1 forward + 1 adjoint (to get objective & gradient)
  - 1 optimization step: 1 forward (to evaluate the updated point)
So expect 2 forward runs + 1 adjoint run total.

To only get 1 forward + 1 adjoint (evaluation only, no optimization step),
set tao_max_it=0 in petsc_options.
"""

import os
import re
import numpy as np

from fiberis.moose.model_builder import OptimizationLayeredModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    CasingConfig, CasingLayerConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, TimeSequenceStepper
)
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


def build_inversion_model():
    """Build the inversion model using fiberis OptimizationLayeredModelBuilder."""

    base_dir = "scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100layer"
    inv_dir = os.path.join(base_dir, "inv_test")
    os.makedirs(inv_dir, exist_ok=True)

    forward_input_file = "forward_and_adjoint.i"
    forward_full_path = os.path.join(inv_dir, forward_input_file)
    master_input_file = "optimize.i"
    master_full_path = os.path.join(inv_dir, master_input_file)

    # 1. Initialize builder
    builder = OptimizationLayeredModelBuilder(project_name="perm5layer_inversion_test")

    # 2. Load gauge data
    gauge_data = Data1DGauge()
    gauge_data.load_npz(os.path.join(base_dir, "data/interference.npz"))
    gauge_data.adaptive_downsample(130)
    gauge_data.data = 6894.76 * gauge_data.data  # psi -> Pa
    initial_pressure_val = gauge_data.data[0]

    # 3. Material properties (initial guess: uniform 1e-18 m^2 for all layers)
    init_perm = 1e-18
    perm_str = f"{init_perm} 0 0 0 {init_perm} 0 0 0 {init_perm}"

    caprock_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=perm_str,
        youngs_modulus=4e10, poissons_ratio=0.3
    )
    sandstone_mats = ZoneMaterialProperties(
        porosity=0.15, permeability=perm_str,
        youngs_modulus=2.5e10, poissons_ratio=0.25
    )
    shale_mats = ZoneMaterialProperties(
        porosity=0.03, permeability=perm_str,
        youngs_modulus=3.5e10, poissons_ratio=0.28
    )

    # 4. Define 200 layers (0.5m each, 100m total) matching ground truth structure
    total_layers = 200
    layer_height = 0.5
    layers = []
    for i in range(total_layers):
        y_center = -50.0 + (i + 0.5) * layer_height
        if y_center < -20.0:
            mat = caprock_mats
        elif -20.0 <= y_center < -16.0:
            mat = shale_mats
        elif -16.0 <= y_center < 14.0:
            mat = caprock_mats
        elif 14.0 <= y_center < 20.0:
            mat = sandstone_mats
        else:
            mat = caprock_mats
        layers.append(CasingLayerConfig(name=f"layer_{i+1}", height=layer_height, materials=mat))

    # 5. Casing config
    casing_config = CasingConfig(
        name="VerticalInversionWell",
        layers=layers,
        injection_well_name="injection_well",
        injection_well_x_coord=45.0
    )
    builder.set_casing_config(casing_config)

    # 6. Mesh
    builder.build_mesh_for_casing_model(domain_length=100.0, nx=200, ny=200)

    # 7. Variables
    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": initial_pressure_val}},
        {"name": "disp_x", "params": {"initial_condition": 0.0}},
        {"name": "disp_y", "params": {"initial_condition": 0.0}}
    ])
    builder.add_adjoint_variables()

    # 8. Global params
    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    # 9. Kernels
    biot_coeff = 0.8
    builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(
        kernel_name="eff_stress_x", variable="disp_x", component=0, biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(
        kernel_name="eff_stress_y", variable="disp_y", component=1, biot_coefficient=biot_coeff)

    # 10. Optimization forward model setup (perm parameterization)
    builder.setup_optimization_forward_model(perm_y=1e-18, perm_z=1e-18)

    # 11. Fluid and materials
    fluid_props = SimpleFluidPropertiesConfig(
        name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_props)
    builder.add_optimization_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=1e-11,
        displacements=['disp_x', 'disp_y'],
        porepressure_variable='pp'
    )

    # 12. Boundary conditions
    builder.add_piecewise_function_from_data1d("injection_pressure_func", gauge_data)
    y_min, y_max = -50.0, 50.0
    builder.add_linear_pressure_boundary(
        boundary_name="injection_well",
        bottom_left=(45.0, y_min, 0),
        top_right=(45.0, y_max, 0),
        pressure_function_name="injection_pressure_func"
    )
    builder.add_boundary_condition(
        name="confine_x", bc_type="NeumannBC", variable="disp_x",
        boundary_name="left right", params={"value": 0}
    )
    builder.add_boundary_condition(
        name="confine_y", bc_type="NeumannBC", variable="disp_y",
        boundary_name="top bottom", params={"value": 0}
    )

    # 13. Tensor outputs
    builder.add_standard_tensor_aux_vars_and_kernels(
        {"stress": "stress", "total_strain": "strain", "strain_rate": "strain_rate"}
    )

    # 14. Optimization problem and reporters
    builder.add_optimization_problem_block()
    builder.add_optimization_reporters_and_dirac(measurement_variable="disp_y")

    # 15. Executioner
    total_time = gauge_data.taxis[-1] - gauge_data.taxis[0]
    dt_control = TimeSequenceStepper()
    dt_control.from_data1d(gauge_data)
    builder.add_optimization_executioner_block(
        end_time=total_time,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control
    )

    builder.add_outputs_block(exodus=True, console=True)

    # 16. Generate forward/adjoint input file
    builder.generate_input_file(forward_full_path)

    # 17. Generate master optimization input file
    #     KEY FIX: Use correct path to measurement CSV
    measurement_csv_path = "../data/measurement.csv"

    initial_alphas = []
    lower_bounds = []
    upper_bounds = []
    for i in range(total_layers):
        y_center = -50.0 + (i + 0.5) * layer_height
        initial_alphas.append(-18.0)
        if -25.0 <= y_center <= 25.0:
            lower_bounds.append(-25.0)
            upper_bounds.append(-10.0)
        else:
            lower_bounds.append(-18.0)
            upper_bounds.append(-18.0)

    builder.generate_optimization_master_file(
        output_filepath=master_full_path,
        forward_input_file=forward_input_file,
        measurement_csv=measurement_csv_path,
        initial_conds=initial_alphas,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds
    )

    # 18. Post-process optimize.i to add tao_max_it=1
    #     The default model_builder doesn't set tao_max_it, so TAO would run
    #     until convergence. We add tao_max_it=1 for a single iteration test.
    #
    #     NOTE: tao_max_it=1 means 1 optimization step AFTER the initial
    #     evaluation, so you'll see 2 forward runs + 1 adjoint run.
    #     Use tao_max_it=0 if you only want the initial evaluation
    #     (1 forward + 1 adjoint, no optimization step).
    with open(master_full_path, "r") as f:
        content = f.read()

    content = content.replace(
        "petsc_options_iname = '-tao_gatol -tao_grtol -tao_gttol'",
        "petsc_options_iname = '-tao_max_it -tao_gatol -tao_grtol -tao_gttol'"
    )
    content = content.replace(
        "petsc_options_value = '0 0 0'",
        "petsc_options_value = '1 0 0 0'"
    )

    with open(master_full_path, "w") as f:
        f.write(content)

    print(f"\n[+] Generated inversion test files in: {inv_dir}")
    print(f"    - {forward_input_file}")
    print(f"    - {master_input_file}")
    print(f"    - Measurement data: {measurement_csv_path}")

    return inv_dir, master_full_path


def run_inversion(inv_dir, master_input_path):
    """Run the MOOSE optimization directly (no SciPy wrapper)."""
    runner = MooseRunner(
        moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
        mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    )

    print("\n--- Running MOOSE Optimization (1 TAO iteration) ---")
    success, stdout, stderr = runner.run(
        input_file_path=master_input_path,
        output_directory=inv_dir,
        num_processors=20,
        log_file_name="simulation_opt.log",
        stream_output=True,
        clean_output_dir=False
    )

    if success:
        print("\n--- MOOSE Optimization Completed Successfully ---")
    else:
        print("\n--- MOOSE Optimization Failed ---")
        print("Check simulation_opt.log for details.")

    return success


if __name__ == "__main__":
    inv_dir, master_path = build_inversion_model()
    run_inversion(inv_dir, master_path)
