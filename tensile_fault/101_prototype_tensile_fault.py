#%% This script contains a baseline prototype for simulating tensile faulting using fiberis.
# The model itself is fully synthetic until I get approved to use real data by Fervo Energy.
# Shenyao Jin, 11/10/2025

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import csv

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.utils.viz_utils import plot_vector_samplers, plot_point_samplers

# Define model parameters
moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"

def model_builder_tensile_fault(**kwargs) -> ModelBuilder:
    """
    This function will create a ModelBuilder object for simulating tensile faulting.
    -- Shenyao Jin, 11/10/2025

    :param kwargs: it will take in various parameters for the model building.
    :return: it will return a ModelBuilder object, so that we can render it and run the simulation.
    """

    conversion_factor = kwargs.get('conversion_factor', 0.3048)

    time_now = datetime.datetime.now()
    date_time_str = time_now.strftime("%Y%m%d_%H%M%S")
    instance_name = kwargs.get('instance_id', date_time_str)

    project_name = f"project_{instance_name}"
    builder = ModelBuilder(project_name=project_name)

    fault_coords = 0 # Make the fault at x = 0 plane
    domain_bounds = (- kwargs.get('model_width', 200.0 * conversion_factor),
                     + kwargs.get('model_width', 200.0 * conversion_factor))

    domain_length = kwargs.get('model_length', 400 * conversion_factor)

    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords = fault_coords,
        domain_bounds = domain_bounds,
        domain_length = domain_length,
        nx = kwargs.get('nx', 200),
        ny_per_layer_half = kwargs.get('ny_per_layer_half', 80),
        bias_y = kwargs.get('bias_y', 1.1)
    )

    # In this case, SRV is not needed.
    matrix_perm = kwargs.get('matrix_perm', 1e-18)  # in m^2
    fault_perm_x = kwargs.get('fault_perm_x', 1e-15)  # in m^2
    fault_perm_y = kwargs.get('fault_perm_y', 1e-17)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    fault_perm = f"{fault_perm_x} 0 0 0 {fault_perm_y} 0 0 0 {matrix_perm}"

    matrix_mats = ZoneMaterialProperties(
        porosity = 0.01, permeability = matrix_perm_str
    )
    fault_mat = ZoneMaterialProperties(
        porosity = 0.15, permeability = fault_perm
    )

    builder.set_matrix_config(
        MatrixConfig(name='matrix', materials = matrix_mats)
    )

    center_x_val = domain_length / 2.0
    fault_height_ft = kwargs.get('fault_height_ft', 2)
    fault_length_ft = kwargs.get('fault_length_ft', 200.0)

    geometries = [
        HydraulicFractureConfig(name="fault",
                                length=fault_length_ft * conversion_factor,
                                height=fault_height_ft * conversion_factor,
                                center_x=center_x_val,
                                center_y=fault_coords,
                                materials=fault_mat)
    ]

    next_block_id = 1
    builder.add_fracture_config(geometries[0])
    builder.add_hydraulic_fracture_2d(
        geometries[0],
        target_block_id = next_block_id
    )
    next_block_id += 1

    builder.add_nodeset_by_coord(nodeset_op_name="injection", new_boundary_name="injection",
                                 coordinates=(center_x_val, fault_coords, 0))

    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": kwargs.get('initial_pressure', 0)}}, # In Pa,
        # Change initial condition as needed after obtaining real data
        {"name": "disp_x", "params": {"initial_condition": 0}},
        {"name": "disp_y", "params": {"initial_condition": 0}}
    ])

    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    biot_coeff = 0.7
    builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=2E-11
    )

    source_pressure = Data1DGauge()
    # Set source pressure to be constant for now. Time range = 5 days. Pressure  == 5e7 Pa
    time_sequence = np.linspace(0, 400, 401)  # 0 to 400 seconds, including 400
    pressure_sequence = np.zeros_like(time_sequence, dtype=float)

    # Set pressure to 5e7 Pa for specified intervals
    pressure_sequence[(time_sequence >= 50) & (time_sequence <= 125)] = 5e7
    pressure_sequence[(time_sequence >= 185) & (time_sequence <= 265)] = 5e7
    pressure_sequence[(time_sequence >= 300) & (time_sequence <= 350)] = 5e7

    source_pressure.taxis = time_sequence
    source_pressure.data = pressure_sequence
    # The reason to set start time is to easily align with the real world date time if needed.
    source_pressure.start_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
    source_pressure.history.add_record("Initial pressure set up", level='INFO')

    fig, ax = plt.subplots(figsize=(8, 4))
    source_pressure.plot(ax=ax)
    plt.show()


    builder.add_piecewise_function_from_data1d(
        name = "source_pressure_func",
        source_data1d = source_pressure
    )

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="source_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    # Add post processors to output fiber data
    # Add some of them within the fault, some of them outside the fault
    fiber_shift_list = np.array([
        fault_length_ft * conversion_factor * 0.75,
        fault_length_ft * conversion_factor * 1.1,
        fault_length_ft * conversion_factor * 1.5
    ])

    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="fault_center_pressure",
            variable="pp",
            point=(center_x_val, fault_coords, 0)
        )
    )

    for shift in fiber_shift_list:
        position_val = np.array([-10, -5, 0, 5, 10]) * conversion_factor
        for pos in position_val:
            builder.add_postprocessor(
                PointValueSamplerConfig(
                    name=f"pressure_point_{shift/conversion_factor/fault_length_ft:.2f}ft_{pos/conversion_factor:.1f}ft_pressure",
                    variable="pp",
                    point=(center_x_val + shift / 2, pos, 0)
                )
            )

            builder.add_postprocessor(
                PointValueSamplerConfig(
                    name=f"pressure_point_{shift / conversion_factor / fault_length_ft:.2f}ft_{pos / conversion_factor:.1f}ft_strain",
                    variable="pp",
                    point=(center_x_val + shift / 2, pos, 0)
                )
            )

        builder.add_postprocessor(
            LineValueSamplerConfig(
                name = f"fiber_sampler_{shift/conversion_factor/fault_length_ft:.2f}ft_strain",
                variable="strain_yy", # what fiber has been measuring
                start_point = (center_x_val + shift / 2, domain_bounds[0] + 20 * conversion_factor, 0),
                end_point = (center_x_val + shift / 2, domain_bounds[1] - 20 * conversion_factor, 0),
                num_points = 200,
                other_params = {"sort_by": "y"}
            )
        )

        builder.add_postprocessor(
            LineValueSamplerConfig(
                name=f"fiber_sampler_{shift / conversion_factor / fault_length_ft:.2f}ft_pressure",
                variable="pp",  # virtual pressure gauge along the fiber
                start_point=(center_x_val + shift / 2, domain_bounds[0] + 20 * conversion_factor, 0),
                end_point=(center_x_val + shift / 2, domain_bounds[1] - 20 * conversion_factor, 0),
                num_points=200,
                other_params={"sort_by": "y"}
            )
        )

    # Define the time stepper
    timestepper_frame = source_pressure.copy()
    timestepper_frame.down_sample(4)  # Down sample to reduce the number of time steps
    timestepper_frame.select_time(source_pressure.start_time, source_pressure.get_end_time(use_timestamp=True))

    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(
        timestepper_frame
    )

    builder.add_executioner_block(
        end_time = timestepper_frame.taxis[-1] - timestepper_frame.taxis[0],
        dt = 3600,
        time_stepper_type = 'TimeSequenceStepper',
        stepper_config = dt_control_func
    )

    builder.add_preconditioning_block(
        active_preconditioner = 'mumps'
    )
    builder.add_outputs_block(
        exodus = False,
        csv = True
    )

    return builder

builder_baseline = model_builder_tensile_fault()
# Render the model to generate the input files
builder_baseline.generate_input_file("output/1111_test_tensile_fault/input.i")
print(builder_baseline)
builder_baseline.plot_geometry(hide_legend=True)

 # Now run the simulation
runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
success, stdout, stderr = runner.run(
    input_file_path="output/1111_test_tensile_fault/input.i",
    output_directory="output/1111_test_tensile_fault",
    num_processors=20,
    log_file_name="simulation.log",
    stream_output=True
)

# Plotting the vector postprocessor data
plot_output_dir = os.path.join("output/1111_test_tensile_fault", "plots")
plot_vector_samplers(folder="output/1111_test_tensile_fault", output_dir=plot_output_dir)
plot_point_samplers(folder="output/1111_test_tensile_fault", output_dir=plot_output_dir)