#%% Enhanced version of 104_const_dt.py
# Shenyao Jin, 10092025, shenyaojin@mines.edu

# Change log:
# 1. Use better spacing relationship to simulate monitor wells <- DON'T. For we are investigating the workflow of parameter study
# 2. Change the fracture to be a single fracture <- DONE.
# 3. Refine the input file builder to make it friendly to change parameters <- DONE. I've wrapped it into a function that takes parameters as input

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader # The module to read point sampler output

# fiberis moose modules
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, PointValueSamplerConfig, LineValueSamplerConfig,
    AdaptiveTimeStepperConfig, TimeSequenceStepper, PostprocessorConfig
)

#%% Decide the path
# Use the old HF-2 as the base case
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)
mds = DSSdata.daxis
ind = (mds>7500)&(mds<15000)
drift_val = np.median(DSSdata.data[ind,:],axis=0)
DSSdata.data -= drift_val.reshape((1,-1))
DSSdata.select_time(0, 400000)
DSSdata.select_depth(12000, 16360)

DSSdata_copy = DSSdata.copy()
DSSdata_copy.select_depth(14500, 15500)

hf_md = 14972

chan_val = DSSdata.get_value_by_depth(hf_md)

pressure_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
gauge_data = Data1DGauge()
gauge_data.load_npz(pressure_g1_path)

DSSdata_crop = DSSdata.copy()
range_val = 200
DSSdata_crop.select_depth(hf_md - range_val, hf_md + range_val)
DSSdata_crop.select_time(0, 400000)

producer_gauge_md = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
dataframe_gauge_md = G3DMeasuredDepth()
dataframe_gauge_md.load_npz(producer_gauge_md)
ind = (dataframe_gauge_md.data > hf_md - range_val) & (dataframe_gauge_md.data < hf_md + range_val)
gauge_mds = dataframe_gauge_md.data[ind]

print("Gauge mds:", gauge_mds)

mesh_y_range = DSSdata_crop.daxis[-1] - DSSdata_crop.daxis[0]
hf_loc = hf_md - DSSdata_crop.daxis[0]
gauge_locs = (gauge_mds - DSSdata_crop.daxis[0])
print("Gauge locs:", gauge_locs)

injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
injection_gauge_pressure_dataframe = Data1DGauge()
injection_gauge_pressure_dataframe.load_npz(injection_gauge_pressure_path)

injection_gauge_pressure_dataframe.remove_abnormal_data(threshold=300, method='mean')

#%% Build the model
conversion_factor = 0.3048

def model_builder_parameter(**kwargs):
    """

    :param kwargs: ny_per_layer_half, matrix_perm, srv_perm, fracture_perm,
    :return:
    """
    output_dir = "output/1009_loop_para"
    os.makedirs(output_dir, exist_ok=True)
    input_file_name = os.path.join(output_dir, "1009_loop_para.i")
    builder = ModelBuilder(project_name="1009_loop_para")

    fracture_coords = hf_loc * conversion_factor

    domain_bounds = (-300 * conversion_factor, 600 * conversion_factor)
    domain_length = 600 * conversion_factor

    fracture_y_coords = fracture_coords
    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords = fracture_y_coords,
        domain_bounds = domain_bounds,
        domain_length = domain_length,
        nx = 200,
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 80),
        bias_y=1.2
    )
    matrix_perm = kwargs.get('matrix_perm', 1e-17)
    srv_perm = kwargs.get('srv_perm', 1e-15)
    src_perm2= srv_perm / 5
    fracture_perm = kwargs.get('fracture_perm', 1e-12)

    matrix_perm_str = f"'{matrix_perm} 0 0  0 {matrix_perm} 0  0 0 {matrix_perm}'"
    srv_perm_str = f"'{srv_perm} 0 0  0 {srv_perm} 0  0 0 {srv_perm}'"
    srv_perm_str2 = f"'{src_perm2} 0 0  0 {src_perm2} 0  0 0 {src_perm2}'"
    fracture_perm_str = f"'{fracture_perm} 0 0  0 {fracture_perm} 0  0 0 {fracture_perm}'"

    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
    srv_mats2 = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str2)
    fracture_mats = ZoneMaterialProperties(porosity=0.14, permeability=fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft2 = kwargs.get('srv_length_ft', 400)
    srv_height_ft2 = kwargs.get('srv_height_ft', 50)
    srv_length_ft1 = kwargs.get('srv_length_ft1', 250)
    srv_height_ft1 = kwargs.get('srv_height_ft1', 150)
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

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

    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": 5.17E7}},  # Initial pressure ~ 7500 psi in Pa.
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
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0, biot_coefficient=boit_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1, biot_coefficient=boit_coeff)
    builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=boit_coeff,
        solid_bulk_compliance=2E-11
    )

    # Add production data from the producer well
    gauge_data_for_moose = injection_gauge_pressure_dataframe.copy()
    gauge_data_for_moose.data = gauge_data_for_moose.data * 6894.76  # convert psi to Pa
    gauge_data_for_moose.adaptive_downsample(400)

    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj", variable="pp", point=(center_x_val, fracture_y_coords, 0)))
    # Add post processors for gauges
    shift_val = 80 # Spacing between producer and monitor well
    gauge_locs = (gauge_mds - DSSdata_crop.daxis[0]) * conversion_factor
    for i, loc in enumerate(gauge_locs):
        builder.add_postprocessor(PointValueSamplerConfig(name=f"pp_gauge_{i+1}", variable="pp", point=(center_x_val + shift_val * conversion_factor, loc, 0)))
        builder.add_postprocessor(PointValueSamplerConfig(name=f"total_strain_yy_gauge_{i+1}", variable="strain_yy", point=(center_x_val + shift_val * conversion_factor, loc, 0)))

    y_min_mesh = fracture_y_coords - 100
    y_max_mesh = fracture_y_coords + 100
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="pressure_wellbore",
            variable="pp",
            start_point=(center_x_val + shift_val * conversion_factor, y_min_mesh, 0),
            end_point=(center_x_val + shift_val * conversion_factor, y_max_mesh, 0),
            num_points=200,
            other_params={'sort_by': 'y'}
        )
    ) # Pressure post processor along the wellbore

    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="strain_yy_wellbore",
            variable="strain_yy",
            start_point=(center_x_val + shift_val * conversion_factor, y_min_mesh, 0),
            end_point=(center_x_val + shift_val * conversion_factor, y_max_mesh, 0),
            num_points=200,
            other_params={'sort_by': 'y'}
            )
    ) # Strain post processor along the wellbore

    # Define time solver,
    total_time = gauge_data_for_moose.taxis[-1] - gauge_data_for_moose.taxis[0]
    gauge_data_for_moose.adaptive_downsample(100)
    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(gauge_data_for_moose)

    builder.add_executioner_block(
        end_time = total_time,
        dt = 3600 * 24 * 5,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control_func
    )

    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=True, csv=True)

    # Render the input file
    builder.generate_input_file(input_file_name)

    # Run the model
    moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
    mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
    success, stdout, stderr = runner.run(
        input_file_path=input_file_name,
        output_directory=output_dir,
        num_processors=20,
        log_file_name="simulation.log",
        stream_output=True
    )

    # Post process the result. We need the simulated gauge data to compare with the real data
    ps_reader = MOOSEPointSamplerReader()
    ps_reader.read(folder=output_dir, variable_index=1)

    simulation_dataframe = Data1DGauge()
    simulation_dataframe.data = ps_reader.data
    simulation_dataframe.taxis = ps_reader.taxis
    simulation_dataframe.start_time = gauge_data_for_moose.start_time
    simulation_dataframe.data[0] = gauge_data_for_moose.data[0] # Make sure the first point matches

    gauge_path = "data/fiberis_format/s_well/gauges/gauge7_data_swell.npz"
    gauge_data_field = Data1DGauge()
    gauge_data_field.load_npz(gauge_path)
    gauge_data_field.data = gauge_data_field.data * 6894.76  # convert psi to Pa

    # Calculate the misfit
    # interpolate the gauge data to the simulation time axis
    gauge_data_field.interpolate(simulation_dataframe.taxis, simulation_dataframe.start_time)
    misfit1 = np.linalg.norm(simulation_dataframe.data - gauge_data_field.data) / np.sqrt(len(simulation_dataframe.data))

    ps_reader = MOOSEPointSamplerReader()
    ps_reader.read(folder=output_dir, variable_index=2)

    simulation_dataframe = Data1DGauge()
    simulation_dataframe.data = ps_reader.data
    simulation_dataframe.taxis = ps_reader.taxis
    simulation_dataframe.start_time = gauge_data_for_moose.start_time
    simulation_dataframe.data[0] = gauge_data_for_moose.data[0]  # Make sure the first point matches

    gauge_path = "data/fiberis_format/s_well/gauges/gauge8_data_swell.npz"
    gauge_data_field = Data1DGauge()
    gauge_data_field.load_npz(gauge_path)
    gauge_data_field.data = gauge_data_field.data * 6894.76  # convert psi to Pa
    # Calculate the misfit
    # interpolate the gauge data to the simulation time axis
    gauge_data_field.interpolate(simulation_dataframe.taxis, simulation_dataframe.start_time)
    misfit2 = np.linalg.norm(simulation_dataframe.data - gauge_data_field.data) / np.sqrt(len(simulation_dataframe.data))
    fig, ax = plt.subplots(figsize=(8, 4))
    simulation_dataframe.plot(ax=ax, use_timestamp=True, use_legend=True, label="simulation")
    gauge_data_field.plot(ax=ax, use_timestamp=True, use_legend=True, label="field")
    plt.tight_layout()
    plt.show()
    return misfit1, misfit2

#%% Run the parameter sweep
base_params = {
    'ny_per_layer_half': 80,
    'matrix_perm': 5e-19,
    'srv_perm': 1e-15,
    'fracture_perm': 1e-12,
    'srv_length_ft': 400,
    'srv_height_ft': 50,
    'srv_length_ft1': 250,
    'srv_height_ft1': 150,
    'hf_length_ft': 250,
    'hf_height_ft': 0.2
}

# Define parameter ranges
srv_perm_range = np.linspace(5e-16, 5e-13, 10)
fracture_perm_range = np.linspace(5e-15, 5e-13, 10)

results = []

for srv_perm in srv_perm_range:
    for fracture_perm in fracture_perm_range:
        print(f"Running with srv_perm: {srv_perm}, fracture_perm: {fracture_perm}")
        params = base_params.copy()
        params['srv_perm'] = srv_perm
        params['fracture_perm'] = fracture_perm
        
        try:
            misfit1, misfit2 = model_builder_parameter(**params)
            results.append({
                'srv_perm': srv_perm,
                'fracture_perm': fracture_perm,
                'misfit1': misfit1,
                'misfit2': misfit2
            })
        except Exception as e:
            print(f"An error occurred for srv_perm={srv_perm}, fracture_perm={fracture_perm}: {e}")
            results.append({
                'srv_perm': srv_perm,
                'fracture_perm': fracture_perm,
                'misfit1': np.nan,
                'misfit2': np.nan
            })

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
output_csv_path = "output/1009_loop_para/misfit_results.csv"
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
results_df.to_csv(output_csv_path, index=False)

print(f"Parameter sweep complete. Results saved to {output_csv_path}")
print(results_df)