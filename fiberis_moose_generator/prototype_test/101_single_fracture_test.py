import os
import numpy as np
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, PointValueSamplerConfig, LineValueSamplerConfig
)
from fiberis.analyzer.Data1D import core1D


def run_single_fracture_analysis():
    """
    构建并运行一个单裂缝模拟，注入点在中心，观测点在裂缝尖端，
    以分析和验证物理信号的传播。
    """
    # --- 1. 基本设置 ---
    output_dir = "moose_single_frac_analysis"
    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(output_dir, "single_frac_analysis.i")

    builder = ModelBuilder(project_name="SingleFracAnalysis")

    # --- 2. 网格生成 ---
    fracture_y_coords = [0.0]
    domain_bounds = (-250.0, 250.0)
    domain_length = 1000.0

    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=fracture_y_coords,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=100,
        ny_per_layer_half=40,
        bias_y=1.5
    )

    # --- 3. 定义几何与材料配置 ---
    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability="'1E-20 0 0  0 1E-20 0  0 0 1E-20'",
                                         youngs_modulus=5.0E10, poissons_ratio=0.2)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1E-17 0 0  0 1E-17 0  0 0 1E-17'")
    fracture_mats = ZoneMaterialProperties(porosity=0.5, permeability="'1E-12 0 0  0 1E-12 0  0 0 1E-12'")

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    hf_length = 500
    hf_center_x = 500
    hf_tip_x = hf_center_x + hf_length / 2.0

    geometries = [
        SRVConfig(name="srv", length=hf_length + 100, height=50, center_x=hf_center_x, center_y=0, materials=srv_mats),
        HydraulicFractureConfig(name="hf", length=hf_length, height=0.2, center_x=hf_center_x, center_y=0,
                                materials=fracture_mats),
    ]

    for geom in geometries:
        if isinstance(geom, SRVConfig):
            builder.add_srv_config(geom)
        elif isinstance(geom, HydraulicFractureConfig):
            builder.add_fracture_config(geom)

    sorted_geometries = sorted(geometries, key=lambda x: x.height, reverse=True)
    next_block_id = 1
    for geom_config in sorted_geometries:
        if isinstance(geom_config, SRVConfig):
            builder.add_srv_zone_2d(geom_config, target_block_id=next_block_id)
        elif isinstance(geom_config, HydraulicFractureConfig):
            builder.add_hydraulic_fracture_2d(geom_config, target_block_id=next_block_id)
        next_block_id += 1

    builder.add_nodeset_by_coord(nodeset_op_name="injection_well", new_boundary_name="injection_well",
                                 coordinates=(hf_center_x, 0))
    builder.add_nodeset_by_coord(nodeset_op_name="observation_well", new_boundary_name="observation_well",
                                 coordinates=(hf_tip_x, 0))

    # --- 4. 添加完整的物理场 ---

    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": 2.64E7}},
        {"name": "disp_x"},
        {"name": "disp_y"}
    ])

    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    # *** FIX IS HERE: Explicitly pass the required parameters ***
    builder.set_porous_flow_dictator(
        dictator_name="dictator",
        porous_flow_variables='pp',
        num_fluid_phases=1,
        num_fluid_components=1
    )

    builder.add_time_derivative_kernel(variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="poro_x", variable="disp_x", component=0,
                                                             biot_coefficient=0.7)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="poro_y", variable="disp_y", component=1,
                                                             biot_coefficient=0.7)
    builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="vol_strain_rate", variable="pp")

    fluid_props = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_props)
    builder.add_poromechanics_materials(fluid_properties_name="water", biot_coefficient=0.7,
                                        solid_bulk_compliance=2E-11)

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "strain": "strain"})

    time_points = np.array([0, 1000, 1100, 2500, 2600, 3600])
    pressure_points = np.array([2.7E7, 3.5E7, 3.2E7, 4.0E7, 3.8E7, 3.5E7])
    pressure_curve = core1D.Data1D(taxis=time_points, data=pressure_points)
    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=pressure_curve)

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection_well",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries=["left", "right"],
        confine_disp_y_boundaries=["top", "bottom"]
    )

    builder.add_postprocessor(
        PointValueSamplerConfig(name="pressure_at_injection", variable="pp", point=(hf_center_x, 0, 0)))
    builder.add_postprocessor(
        PointValueSamplerConfig(name="pressure_at_observation", variable="pp", point=(hf_tip_x, 0, 0)))
    builder.add_postprocessor(
        PointValueSamplerConfig(name="vertical_stress_at_observation", variable="stress_yy", point=(hf_tip_x, 0, 0)))
    builder.add_postprocessor(
        PointValueSamplerConfig(name="vertical_strain_at_observation", variable="strain_yy", point=(hf_tip_x, 0, 0)))

    # --- 5. 求解器和输出 ---
    builder.add_executioner_block(type="Transient", solve_type="Newton", end_time=3600, dt=100, time_stepper_type='ConstantDT')
    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=True, csv=True)

    # --- 6. 生成并运行 ---
    builder.generate_input_file(input_file)
    print(f"\nSuccessfully generated single fracture simulation file at: {input_file}")

    print("\n--- Starting MOOSE Simulation Runner ---")
    try:
        moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
        mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
        runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
        success, stdout, stderr = runner.run(
            input_file_path=input_file,
            output_directory=output_dir,
            num_processors=2,
            log_file_name="simulation.log"
        )
        if success:
            print("\nSimulation completed successfully!")
            print(
                f"To see the signal at the observation well, check the file: {os.path.join(output_dir, 'single_frac_analysis_out.csv')}")
        else:
            print("\nSimulation failed.")
            print("--- STDERR ---")
            print(stderr)
    except Exception as e:
        print(f"\nAn error occurred during the simulation run: {e}")


if __name__ == "__main__":
    run_single_fracture_analysis()

