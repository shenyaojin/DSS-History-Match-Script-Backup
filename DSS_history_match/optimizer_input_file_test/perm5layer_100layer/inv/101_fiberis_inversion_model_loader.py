import os
import numpy as np

from fiberis.moose.model_builder import OptimizationLayeredModelBuilder
from fiberis.moose.config import CasingConfig, CasingLayerConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig, TimeSequenceStepper
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

def optimization_model_creater():
    # Define directories
    base_dir = "scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100layer"
    inv_dir = os.path.join(base_dir, "inv")
    os.makedirs(inv_dir, exist_ok=True)

    # Output filenames
    forward_input_file = "forward_and_adjoint.i"
    forward_full_path = os.path.join(inv_dir, forward_input_file)
    master_input_file = "optimize.i"
    master_full_path = os.path.join(inv_dir, master_input_file)

    # 1. Initialize the Optimization Builder
    builder = OptimizationLayeredModelBuilder(project_name="perm5layer_inversion")

    # 2. Load Gauge Data for pressure curve and initial conditions
    gauge_data = Data1DGauge()
    gauge_data.load_npz(os.path.join(base_dir, "data/interference.npz"))
    gauge_data.adaptive_downsample(130)
    gauge_data.data = 6894.76 * gauge_data.data  # Convert psi to Pa
    initial_pressure_val = gauge_data.data[0]

    # 3. Define Material Properties (Base properties matching Ground Truth)
    # Initial permeability guess: 1e-18 m^2 for ALL layers
    init_perm = 1e-18
    perm_str = f"{init_perm} 0 0 0 {init_perm} 0 0 0 {init_perm}"

    caprock_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=perm_str,
        youngs_modulus=3.5e10, poissons_ratio=0.25
    )
    sandstone_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=perm_str,
        youngs_modulus=3.5e10, poissons_ratio=0.25
    )
    shale_mats = ZoneMaterialProperties(
        porosity=0.01, permeability=perm_str,
        youngs_modulus=3.5e10, poissons_ratio=0.25
    )

    # 4. Define 200 layers of 0.5m height each (Total height 100m)
    total_layers = 200
    layer_height = 0.5
    layers = []
    for i in range(total_layers):
        y_center = -50.0 + (i + 0.5) * layer_height
        
        # Map back to ground truth spatial structure to ensure identical twin physics
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

    # 5. Create Casing Configuration
    casing_config = CasingConfig(
        name="VerticalInversionWell",
        layers=layers,
        injection_well_name="injection_well",
        injection_well_x_coord=45.0
    )
    builder.set_casing_config(casing_config)

    # 6. Build the Mesh
    # Total height 100m, domain length 100m. 
    # ny must match total_layers (200) so that every layer is represented by elements.
    builder.build_mesh_for_casing_model(domain_length=100.0, nx=200, ny=200)

    # 7. Define Primary and Adjoint Variables (Removed scaling to match ground truth)
    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": initial_pressure_val}},
        {"name": "disp_x", "params": {"initial_condition": 0.0}},
        {"name": "disp_y", "params": {"initial_condition": 0.0}}
    ])
    builder.add_adjoint_variables()

    # 8. Setup Global Params and Dictator
    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    # 9. Define Kernels (Physics)
    biot_coeff = 0.8
    builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                             biot_coefficient=biot_coeff)

    # 10. Optimization-specific Setup (Variables, Functions, and Gradients)
    # Keep perm_y and perm_z consistent with the initial guess matrix permeability to avoid extreme anisotropy
    builder.setup_optimization_forward_model(perm_y=1e-18, perm_z=1e-18)

    # 11. Define Fluid and Optimization Poromechanics Materials
    fluid_props = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_props)
    builder.add_optimization_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=1e-11,
        displacements=['disp_x', 'disp_y'],
        porepressure_variable='pp'
    )

    # 12. Boundary Conditions and Time Functions
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

    # 13. Tensor Outputs
    builder.add_standard_tensor_aux_vars_and_kernels(
        {"stress": "stress", "total_strain": "strain", "strain_rate": "strain_rate"}
    )

    # 14. Optimization Problem and Reporters
    # Matching disp_y in the center 50m
    builder.add_optimization_problem_block()
    builder.add_optimization_reporters_and_dirac(measurement_variable="disp_y")

    # 15. Executioner Block
    total_time = gauge_data.taxis[-1] - gauge_data.taxis[0]
    dt_control = TimeSequenceStepper()
    dt_control.from_data1d(gauge_data)
    builder.add_optimization_executioner_block(
        end_time=total_time,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control
    )

    builder.add_outputs_block(exodus=True, console=True)
    builder.add_preconditioning_block()

    # 16. Generate Forward/Adjoint Input File
    builder.generate_input_file(forward_full_path)

    # 17. Generate Master Optimization Input File
    # Initial alpha = -18 for 1e-18.
    # Bounds: -25 to -10 for inversion region, fixed at -18 for matrix zones.
    # Observation region is center 50m: y in [-25, 25].
    # Total height 100m, y in [-50, 50]. 
    initial_alphas = []
    lower_bounds = []
    upper_bounds = []
    for i in range(total_layers):
        y_center = -50.0 + (i + 0.5) * layer_height
        initial_alphas.append(-18.0)

        lower_bounds.append(-25.0)
        upper_bounds.append(-10.0)

    builder.generate_optimization_master_file(
        output_filepath=master_full_path,
        forward_input_file=forward_input_file,
        measurement_csv="measurement_data.csv",
        initial_conds=initial_alphas,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds
    )

    # 18. Provide a dummy measurement CSV to ensure the run doesn't crash on load
    csv_path = os.path.join(inv_dir, "measurement_data.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("measurement_xcoord,measurement_ycoord,measurement_zcoord,measurement_time,measurement_values\n")
            # Synthetic observation location at x=60, center y=0, time=600
            f.write("60.0,0.0,0.0,600.0,0.0\n") 

    print(f"\n[+] Generated inversion model files in: {inv_dir}")
    print("\nNote: For the synthetic test, you must convert the ground truth data output")
    print("('fwd/output_gt/observation_disp.csv') into the required measurement format")
    print("('inv/measurement_data.csv') before running the optimization.")

if __name__ == "__main__":
    optimization_model_creater()
