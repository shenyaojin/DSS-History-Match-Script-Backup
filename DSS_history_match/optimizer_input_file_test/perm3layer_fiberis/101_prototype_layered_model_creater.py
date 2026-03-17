import os
import numpy as np

# Adjust imports to your local structure if necessary
from fiberis.moose.model_builder import OptimizationLayeredModelBuilder
from fiberis.moose.config import CasingConfig, CasingLayerConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig
from fiberis.analyzer.Data1D.core1D import Data1D

def create_and_run_optimization_model():
    output_dir = "scripts/DSS_history_match/optimizer_input_file_test/perm3layer_fiberis"
    os.makedirs(output_dir, exist_ok=True)
    
    forward_input_file = "forward_and_adjoint.i"
    forward_full_path = os.path.join(output_dir, forward_input_file)
    master_input_file = "optimize.i"
    master_full_path = os.path.join(output_dir, master_input_file)
    
    # 1. Initialize the Optimization Builder
    builder = OptimizationLayeredModelBuilder(project_name="perm3layer_opt")
    
    # 2. Define materials for the 3 layers
    # Note: Permeability here is a placeholder. The optimization process overrides it using parsed functions.
    bottom_mats = ZoneMaterialProperties(porosity=0.01, permeability=1e-15, youngs_modulus=50e9, poissons_ratio=0.2)
    center_mats = ZoneMaterialProperties(porosity=0.01, permeability=1e-15, youngs_modulus=50e9, poissons_ratio=0.2)
    top_mats = ZoneMaterialProperties(porosity=0.01, permeability=1e-15, youngs_modulus=50e9, poissons_ratio=0.2)
    
    # 3. Setup casing config
    # The builder centers the total height at y=0. Total height = 48 + 4 + 48 = 100.
    # Therefore, the domain spans y = -50 to y = 50.
    # The center layer spans from y = -2 to y = 2.
    # The injection well is placed at y = 0.
    casing_config = CasingConfig(
        name="3layer_casing",
        injection_well_name="injection_well",
        injection_well_x_coord=20.0,
        layers=[
            CasingLayerConfig(name="matrix_bottom", height=48.0, materials=bottom_mats),
            CasingLayerConfig(name="srv", height=4.0, materials=center_mats),
            CasingLayerConfig(name="matrix_top", height=48.0, materials=top_mats)
        ]
    )
    builder.set_casing_config(casing_config)
    
    # 4. Add fluid properties
    fluid_props = SimpleFluidPropertiesConfig(
        name="oil", bulk_modulus=1.2e9, viscosity=0.0012, density0=825.0, 
        thermal_expansion=0.0008, cp=2100.0, cv=2000.0, porepressure_coefficient=1.0
    )
    builder.add_fluid_properties_config(fluid_props)
    
    # 5. Define Primary and Adjoint Variables
    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": 0.0}},
        {"name": "disp_x", "params": {"initial_condition": 0.0}},
        {"name": "disp_y", "params": {"initial_condition": 0.0}}
    ])
    builder.add_adjoint_variables()
    
    # 6. Build the Mesh
    builder.build_mesh_for_casing_model(domain_length=200.0, nx=200, ny=200)
    
    # 7. Add User Objects and Physics Kernels
    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp", num_fluid_phases=1, num_fluid_components=1)
    builder.add_global_params({'PorousFlowDictator': 'dictator'})
    
    builder.add_time_derivative_kernel(variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_custom_kernel("StressDivergenceTensors", "grad_stress_x", variable="disp_x", component=0, displacements="disp_x disp_y")
    builder.add_custom_kernel("StressDivergenceTensors", "grad_stress_y", variable="disp_y", component=1, displacements="disp_x disp_y")
    builder.add_custom_kernel("PorousFlowEffectiveStressCoupling", "eff_stress_x", variable="disp_x", component=0, biot_coefficient=0.7, displacements="disp_x disp_y")
    builder.add_custom_kernel("PorousFlowEffectiveStressCoupling", "eff_stress_y", variable="disp_y", component=1, biot_coefficient=0.7, displacements="disp_x disp_y")
    
    # 8. Optimization-specific Setup (Variables, Functions, and Gradients)
    builder.setup_optimization_forward_model(perm_y=1e-20, perm_z=0.0)
    
    # 9. Materials Block Generation
    builder.add_optimization_poromechanics_materials(
        fluid_properties_name="oil",
        biot_coefficient=0.7,
        solid_bulk_compliance=2e-11,
        displacements=['disp_x', 'disp_y'],
        porepressure_variable='pp'
    )
    
    # 10. Boundary Conditions and Time Functions
    t_vals = np.array([0.0, 725.0, 2023.0, 3446.0, 4010.0, 4486.0, 6280.0, 7401.0, 8565.0, 9324.0, 10004.0, 28800.0])
    p_vals = np.array([12691246.67, 12696027.23, 12685664.88, 12701609.02, 12685981.35, 12682156.91, 12700747.17, 12685880.34, 12707413.0, 12689475.85, 12685940.94, 14000000.0])
    
    pressure_data = Data1D(taxis=t_vals, data=p_vals)
    builder.add_piecewise_function_from_data1d("injection_pressure_func", pressure_data)
    
    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection_well",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )
    
    # 11. Tensor Outputs
    tensor_map = {"stress": "stress", "total_strain": "strain"}
    builder.add_standard_tensor_aux_vars_and_kernels(tensor_map)
    
    # 12. Executioner, Reporters, and Master Files
    builder.add_optimization_problem_block()
    builder.add_optimization_reporters_and_dirac(measurement_variable="disp_y")
    builder.add_optimization_executioner_block(end_time=28800, dt=600)
    builder.add_outputs_block(exodus=True, console=True)
    
    builder.generate_input_file(forward_full_path)
    
    builder.generate_optimization_master_file(
        output_filepath=master_full_path,
        forward_input_file=forward_input_file,
        measurement_csv="measurement_data.csv",
        initial_conds=[-25, -10, -25],
        lower_bounds=[-25, -25, -25],
        upper_bounds=[-10, -10, -10]
    )

    # 13. Provide a dummy measurement CSV to ensure the run doesn't crash on load
    csv_path = os.path.join(output_dir, "measurement_data.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("measurement_xcoord,measurement_ycoord,measurement_zcoord,measurement_time,measurement_values\n")
            f.write("20.0,0.0,0.0,600.0,0.001\n") 
            
    print(f"\n[+] Generated optimization models in: {output_dir}\n")
    print("To run the optimization, use the following code:")
    
    runner_code = f"""
from fiberis.moose.runner import MooseRunner

if __name__ == '__main__':
    runner = MooseRunner(
        moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
        mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    )

    input_file_path = "{master_full_path}"
    output_directory = "{output_dir}"

    success, stdout, stderr = runner.run(
        input_file_path=input_file_path,
        output_directory=output_directory,
        num_processors=18,
        log_file_name="simulation.log",
        stream_output=True,
        clean_output_dir=False
    )
    """
    
    runner_path = os.path.join(output_dir, "run_opt.py")
    with open(runner_path, 'w') as f:
        f.write(runner_code)
    print(f"Runner script saved to: {runner_path}")
    print(f"Run it using: python {runner_path}")

if __name__ == "__main__":
    create_and_run_optimization_model()
