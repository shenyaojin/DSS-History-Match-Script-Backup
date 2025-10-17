#%% Enhanced version of 105_single_fracture_full_test.py
# See my notes for more details
# Shenyao Jin, 10/12/2025, shenyaojin@mines.edu

# Change Log:
# 1. Add SRV size as a parameter, remove SRV perm for it’s not
# 2. Make the mesh larger.
# 3. Use literature value to calibrate Bakken formation’s perm

import numpy as np
import os
import matplotlib.pyplot as plt


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
    AdaptiveTimeStepperConfig, TimeSequenceStepper, TimeStepperFunctionConfig
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

#%% Build the model using fiberis
conversion_factor = 0.3048 # 1 ft = 0.3048 m

def model_builder_parameter(**kwargs):
    """
    Build the model using fiberis. Parameters are passed using kwargs to make it flexible.

    :param kwargs: Parameters to build the model.
    :return: model: The misfit.
    """

    output_dir = "output/1012_parameter_scan_test"
    os.makedirs(output_dir, exist_ok=True)
    input_file_name = os.path.join(output_dir, "1012_parameter_scan_test.i")
    builder = ModelBuilder(project_name="1012_parameter_scan_test")

    # Mesh and Geometry
    fracture_coords = hf_loc * conversion_factor
    domain_bounds = (- 600 * conversion_factor, 900 * conversion_factor) # Make the mesh larger,
    # only for not touching the boundary
    domain_length = 800 * conversion_factor

    fracture_y_coords = fracture_coords
    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords = fracture_y_coords,
        domain_bounds = domain_bounds,
        domain_length = domain_length,
        nx = 300,
        ny_per_layer_half = kwargs.get('ny_per_layer_half', 160), # Total ny = 240
        bias_y = 1.2
    )

    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-15)
    srv_perm2= kwargs.get('srv_perm2', 1e-14)
    fracture_perm = kwargs.get('fracture_perm', 1e-13)

    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    srv_perm_str2 = f"{srv_perm2} 0 0 0 {srv_perm2} 0 0 0 {srv_perm2}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"
    # Material Properties
    matrix_mats = ZoneMaterialProperties(porosity = 0.01, permeability = matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity = 0.1, permeability = srv_perm_str)
    srv_mats2 = ZoneMaterialProperties(porosity = 0.1, permeability = srv_perm_str2)
    fracture_mats = ZoneMaterialProperties(porosity = 0.16, permeability = fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft2 = kwargs.get('srv_length_ft', 400)
    srv_height_ft2 = kwargs.get('srv_height_ft', 50)
    srv_length_ft1 = kwargs.get('srv_length_ft1', 250)
    srv_height_ft1 = kwargs.get('srv_height_ft1', 150)
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv_tall", length=srv_length_ft1 * conversion_factor, height=srv_height_ft1 * conversion_factor,
                  center_x=center_x_val, center_y=fracture_y_coords, materials=srv_mats2),
        SRVConfig(name="srv_wide", length=srv_length_ft2 * conversion_factor, height=srv_height_ft2 * conversion_factor,
                  center_x=center_x_val, center_y=fracture_y_coords, materials=srv_mats),
        HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor,
                                height=hf_height_ft * conversion_factor, center_x=center_x_val,
                                center_y=fracture_y_coords, materials=fracture_mats)
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

    builder.add_nodeset_by_coord(nodeset_op_name="injection", new_boundary_name="injection",
                                 coordinates=(center_x_val, fracture_y_coords, 0))

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
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                             biot_coefficient=boit_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                             biot_coefficient=boit_coeff)
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

    builder.add_postprocessor(
        PointValueSamplerConfig(name="pp_inj", variable="pp", point=(center_x_val, fracture_y_coords, 0)))
    # Add post processors for gauges
    shift_val = 80  # Spacing between producer and monitor well
    gauge_locs = (gauge_mds - DSSdata_crop.daxis[0]) * conversion_factor
    for i, loc in enumerate(gauge_locs):
        builder.add_postprocessor(PointValueSamplerConfig(name=f"pp_gauge_{i + 1}", variable="pp",
                                                          point=(center_x_val + shift_val * conversion_factor, loc, 0)))
        builder.add_postprocessor(PointValueSamplerConfig(name=f"total_strain_yy_gauge_{i + 1}", variable="strain_yy",
                                                          point=(center_x_val + shift_val * conversion_factor, loc, 0)))

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
    )  # Pressure post processor along the wellbore

    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="strain_yy_wellbore",
            variable="strain_yy",
            start_point=(center_x_val + shift_val * conversion_factor, y_min_mesh, 0),
            end_point=(center_x_val + shift_val * conversion_factor, y_max_mesh, 0),
            num_points=200,
            other_params={'sort_by': 'y'}
        )
    )  # Strain post processor along the wellbore

    # Define time solver,
    total_time = gauge_data_for_moose.taxis[-1] - gauge_data_for_moose.taxis[0]
    gauge_data_for_moose.adaptive_downsample(150)
    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(gauge_data_for_moose)

    # dt_control_func = TimeStepperFunctionConfig(
    #     name='dt_control_func',
    #     x_values=[0, total_time / 3, total_time / 2, total_time],
    #     y_values=[720000, 360000, 720000]
    # )
    #
    # adaptive_config = AdaptiveTimeStepperConfig(functions=[dt_control_func])

    builder.add_executioner_block(
        end_time=total_time,
        dt=3600 * 24 * 5,
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

    # After the run is complete, process the results into Data1DGauge dataframes
    gauge_results = []
    for i in range(len(gauge_locs)):
        ps_reader = MOOSEPointSamplerReader()
        ps_reader.read(folder=output_dir, variable_index=i+1)  # +1 because 0 is injection well

        simulation_dataframe = Data1DGauge()
        simulation_dataframe.data = ps_reader.data
        simulation_dataframe.taxis = ps_reader.taxis
        simulation_dataframe.start_time = gauge_data_for_moose.start_time
        simulation_dataframe.data[0] = gauge_data_for_moose.data[0] # Match the first point, for post processor will set to 0 at t=0

        gauge_results.append(simulation_dataframe)

    return gauge_results

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

#%% Misfit calculation and optimization
def calculate_misfit(simulated_gauges, params):
    """
    Calculates the misfit between simulated gauge data and field data,
    plots the comparison, and saves the figure. Misfit is calculated in PSI.
    """
    field_gauge_paths = [
        "data/fiberis_format/s_well/gauges/gauge7_data_swell.npz",
        "data/fiberis_format/s_well/gauges/gauge8_data_swell.npz"
    ]
    
    # Conversion factor from Pa to psi
    pa_to_psi = 0.000145038

    # Create a directory for the figures
    fig_dir = "figs/1012/figures"
    os.makedirs(fig_dir, exist_ok=True)

    total_misfit = 0
    # Format the parameter string for display on the plot
    param_str = '\n'.join([f'{k}: {v:.2e}' for k, v in params.items()])

    for i, sim_gauge_pa in enumerate(simulated_gauges):
        # Load field data (assumed to be in PSI from the file)
        field_gauge_psi = Data1DGauge()
        field_gauge_psi.load_npz(field_gauge_paths[i])
        
        # Convert simulation data from Pa to PSI for comparison
        sim_gauge_psi = sim_gauge_pa.copy()
        sim_gauge_psi.data = sim_gauge_pa.data * pa_to_psi

        # Interpolate field data to simulation time axis. 
        field_gauge_psi.interpolate(sim_gauge_psi.taxis, sim_gauge_psi.start_time)

        # Calculate RMSE using data in PSI
        misfit = np.linalg.norm(sim_gauge_psi.data - field_gauge_psi.data) / np.sqrt(len(sim_gauge_psi.data))
        total_misfit += misfit

        # Plotting the comparison in PSI
        fig, ax = plt.subplots(figsize=(12, 6))
        sim_gauge_psi.plot(ax=ax, use_timestamp=True, use_legend=True, label='Simulated pressure')
        field_gauge_psi.plot(ax=ax, use_timestamp=True, use_legend=True, label='Measured pressure (field)')
        
        gauge_number = i + 7 # Gauge numbers are 7 and 8
        ax.set_title(f'Gauge {gauge_number} Pressure Comparison')
        ax.set_ylabel('Pressure (psi)')
        ax.set_xlabel('Time')
        
        # Add parameter text to the plot
        plt.text(0.05, 0.95, param_str, transform=ax.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()

        # Save the figure with a unique name based on parameters
        run_id = "_".join([f"{v:.1e}" for v in params.values()])
        fig_path = os.path.join(fig_dir, f"gauge_{gauge_number}_run_{run_id}.png")
        plt.savefig(fig_path)
        plt.close(fig) # Close the figure to free up memory

    return total_misfit

# Define the search space for the parameters
dimensions = [
    Real(low=100, high=500, name='srv_length_ft'),
    Real(low=1e-14, high=1e-11, prior='log-uniform', name='fracture_perm'),
    Real(low=1e-16, high=1e-13, prior='log-uniform', name='srv_perm'),
    Real(low=1e-19, high=1e-16, prior='log-uniform', name='matrix_perm')
]

# Objective function for skopt
@use_named_args(dimensions=dimensions)
def objective(srv_length_ft, fracture_perm, srv_perm, matrix_perm):
    """
    Objective function for skopt to minimize.
    """
    print(f"Running with parameters: srv_length_ft={srv_length_ft}, fracture_perm={fracture_perm}, srv_perm={srv_perm}, matrix_perm={matrix_perm}")
    
    params = {
        'srv_length_ft': srv_length_ft,
        'fracture_perm': fracture_perm,
        'srv_perm': srv_perm,
        'matrix_perm': matrix_perm
    }

    try:
        simulated_gauges = model_builder_parameter(**params)
        misfit = calculate_misfit(simulated_gauges, params)
        print(f"Misfit: {misfit}")
        return misfit
    except Exception as e:
        print(f"An error occurred with parameters: {params}. Error: {e}")
        return 1e10 # Return a large value if the simulation fails

# Run the optimization
result = gp_minimize(
    func=objective,
    dimensions=dimensions,
    n_calls=50,
    random_state=123
)

# Print the results
print("Best parameters found:")
print(f"  srv_length_ft: {result.x[0]}")
print(f"  fracture_perm: {result.x[1]}")
print(f"  srv_perm: {result.x[2]}")
print(f"  matrix_perm: {result.x[3]}")
print(f"Best misfit: {result.fun}")

