#%% I will try to fix the bug in const DT: what is causing this?
# This version will not include visualization, will start from rendering input file.
# Modify from 103_timestep_bug_investigation.py
# Shenyao Jin, shenyaojin@mines.edu
# Last modified on 10/04/2025
# Modified from 104_const_dt.py. Make it a baseline and the objective is to make time_sequence converge.
# Also make the questioin to a single fracture, not 3 fractures in next script.

# Change the geometry into correct one.

import numpy as np
import os
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth

from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, PointValueSamplerConfig, LineValueSamplerConfig,
    AdaptiveTimeStepperConfig, TimeSequenceStepper, PostprocessorConfig
)

#%% Data handling
datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"

DSSdata = DSS2D()
DSSdata.load_npz(datapath)

mds = DSSdata.daxis
ind = (mds>7500)&(mds<15000)
drift_val = np.median(DSSdata.data[ind,:],axis=0)
DSSdata.data -= drift_val.reshape((1,-1))

DSSdata.select_time(0, 400000)
DSSdata.select_depth(12000, 16360)

DSSdata_copy = DSSdata.copy()
DSSdata_copy.select_depth(14500, 15500)

hf1_md = 14888
hf2_md = 14972
hf3_md = 14992
time_point = 160000

chan1 = DSSdata.get_value_by_depth(hf1_md)
chan2 = DSSdata.get_value_by_depth(hf2_md)
chan3 = DSSdata.get_value_by_depth(hf3_md)

time_slice, _ = DSSdata_copy.get_value_by_time(time_point)
datapath = "data/fiberis_format/prod/gauges/pressure_g1.npz"

gauge_data = Data1DGauge()
gauge_data.load_npz(datapath)

DSSdata_crop = DSSdata.copy()
DSSdata_crop.select_depth(min(hf1_md, hf2_md)-100, max(hf1_md, hf2_md)+100)
DSSdata_crop.select_time(0, 400000)

gauge_md_path = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
dataframe_gauge_md = G3DMeasuredDepth()
dataframe_gauge_md.load_npz(gauge_md_path)
print(dataframe_gauge_md)
ind = (dataframe_gauge_md.data > min(hf1_md, hf2_md)-100) & (dataframe_gauge_md.data < max(hf1_md, hf2_md)+100)
gauge_mds = dataframe_gauge_md.data[ind]
print(gauge_mds)

mesh_y_range = DSSdata_crop.daxis[-1] - DSSdata_crop.daxis[0]
hf_1_loc = (hf1_md - DSSdata_crop.daxis[0])
hf_2_loc = (hf2_md - DSSdata_crop.daxis[0])
hf_3_loc = (hf3_md - DSSdata_crop.daxis[0])
print(f"HF1 location in mesh: {hf_1_loc:.3f} ft")
print(f"HF2 location in mesh: {hf_2_loc:.3f} ft")
print(f"HF3 location in mesh: {hf_3_loc:.3f} ft")

gauge_locs = (gauge_mds - DSSdata_crop.daxis[0])
print("Gauge locations in mesh (ft):", gauge_locs)

inj_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
from fiberis.analyzer.Data1D.core1D import Data1D
injection_pressure_dataframe = Data1D()
injection_pressure_dataframe.load_npz(inj_gauge_pressure_path)
print(injection_pressure_dataframe)

injection_pressure_dataframe.remove_abnormal_data(threshold=300, method='mean')

#%% Render MOOSE input file

# 1. basic parameters
output_dir = "output/1004_constDT_bug_investigation"
os.makedirs(output_dir, exist_ok=True)
input_file_name = os.path.join(output_dir, "DSS_constDT_bug_investigation.i")
builder = ModelBuilder(project_name="DSS_constDT_bug_investigation")

# 2. mesh, HF, SRV and matrix
conversion_factor = 0.3048  # ft to m
shift = - 0.831 # To make coords shift to int
fracture_y1_coords = (hf_1_loc + shift) * conversion_factor
fracture_y2_coords = (hf_2_loc + shift) * conversion_factor
fracture_y3_coords = (hf_3_loc + shift) * conversion_factor

domain_bounds = (-100 * conversion_factor, 400 * conversion_factor)
domain_length = 500 * conversion_factor

fracture_y_coords = [fracture_y1_coords, fracture_y2_coords, fracture_y3_coords]
builder.build_stitched_mesh_for_fractures(
    fracture_y_coords=fracture_y_coords,
    domain_bounds=domain_bounds,
    domain_length=domain_length,
    nx = 200,
    ny_per_layer_half= 25,
    bias_y=1.2
)

matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability="'1E-17 0 0  0 1E-17 0  0 0 1E-17'")
srv_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1E-15 0 0  0 1E-15 0  0 0 1E-15'")
fracture_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1E-12 0 0  0 1E-12 0  0 0 1E-12'")

builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

center_x_val = domain_length / 2.0
geometries = [
    SRVConfig(name="srv_1", length=300 * conversion_factor, height=50 * conversion_factor, center_x=center_x_val, center_y=fracture_y1_coords, materials=srv_mats),
    HydraulicFractureConfig(name="hf_1", length=250 * conversion_factor, height=0.2 * conversion_factor, center_x=center_x_val, center_y=fracture_y1_coords, materials=fracture_mats),
    SRVConfig(name="srv_2", length=300 * conversion_factor, height=50 * conversion_factor, center_x=center_x_val, center_y=fracture_y2_coords, materials=srv_mats),
    HydraulicFractureConfig(name="hf_2", length=250 * conversion_factor, height=0.2 * conversion_factor, center_x=center_x_val, center_y=fracture_y2_coords, materials=fracture_mats),
    SRVConfig(name="srv_3", length=300 * conversion_factor, height=50 * conversion_factor,  center_x=center_x_val, center_y=fracture_y3_coords, materials=srv_mats),
    HydraulicFractureConfig(name="hf_3", length=250 * conversion_factor, height=0.2 * conversion_factor, center_x=center_x_val, center_y=fracture_y3_coords, materials=fracture_mats)
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

builder.add_nodeset_by_coord(nodeset_op_name="injection_1", new_boundary_name="injection_1", coordinates=(center_x_val, fracture_y1_coords, 0))
builder.add_nodeset_by_coord(nodeset_op_name="injection_2", new_boundary_name="injection_2", coordinates=(center_x_val, fracture_y2_coords, 0))
builder.add_nodeset_by_coord(nodeset_op_name="injection_3", new_boundary_name="injection_3", coordinates=(center_x_val, fracture_y3_coords, 0))

# 3. Add geophysical fields
builder.add_variables([
    {"name": "pp", "params": {"initial_condition": 5.17E7}}, # Initial pressure ~ 7500 psi in Pa.
    {"name": "disp_x", "params": {"initial_condition": 0}},
    {"name": "disp_y", "params": {"initial_condition": 0}}
])

builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

biot_coeff_val = 0.7
builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0, biot_coefficient=biot_coeff_val)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1, biot_coefficient=biot_coeff_val)
builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

fluid_props = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
# Viscosity: 1e-3 pa*s
builder.add_fluid_properties_config(fluid_props)
builder.add_poromechanics_materials(
    fluid_properties_name="water",
    biot_coefficient=biot_coeff_val,
    solid_bulk_compliance=2E-11
)

# For the pressure in HFs, let's use the production data from injection well.
gauge_data_for_moose = injection_pressure_dataframe.copy() # Use real injection data
gauge_data_for_moose.data = gauge_data_for_moose.data * 6894.76 # Convert psi to Pa
gauge_data_for_moose.adaptive_downsample(400) # Downsample to 400 points

builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

builder.set_hydraulic_fracturing_bcs(
    injection_well_boundary_name="injection_1 injection_2 injection_3",
    injection_pressure_function_name="injection_pressure_func",
    confine_disp_x_boundaries="left right",
    confine_disp_y_boundaries="top bottom"
)

builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

# Add postprocessors for injection points
builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj_1", variable="pp", point=(center_x_val, fracture_y1_coords, 0)))
builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj_2", variable="pp", point=(center_x_val, fracture_y2_coords, 0)))
builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj_3", variable="pp", point=(center_x_val, fracture_y3_coords, 0)))

# Add postprocessors for gauge locations (assuming wellbore at x=0)
gauge_locs = gauge_locs * conversion_factor # Transfer to ft
for i, gauge_y in enumerate(gauge_locs):
    builder.add_postprocessor(PointValueSamplerConfig(name=f"pp_gauge_{i}", variable="pp", point=(center_x_val - 90 * conversion_factor, gauge_y, 0)))
    builder.add_postprocessor(PointValueSamplerConfig(name=f"strain_gauge_{i}", variable="strain_yy", point=(center_x_val - 90 * conversion_factor, gauge_y, 0)))

y_min_mesh = min(fracture_y_coords) - 50
y_max_mesh = max(fracture_y_coords) + 50
builder.add_postprocessor(
    LineValueSamplerConfig(name="pressure_wellbore", variable="pp", start_point=(center_x_val - 90 * conversion_factor, y_min_mesh, 0),
                           end_point=(center_x_val - 90 * conversion_factor, y_max_mesh, 0), num_points=200,
                           other_params={'sort_by': 'y'}))
builder.add_postprocessor(
    LineValueSamplerConfig(name="strain_wellbore", variable="strain_yy", start_point=(center_x_val - 90 * conversion_factor, y_min_mesh, 0),
                           end_point=(center_x_val - 90 * conversion_factor, y_max_mesh, 0), num_points=200,
                           other_params={'sort_by': 'y'}))

# Define the solver, const DT
total_time = gauge_data_for_moose.taxis[-1] - gauge_data_for_moose.taxis[0]
gauge_data_for_moose.adaptive_downsample(110)
dt_control_func = TimeSequenceStepper()
dt_control_func.from_data1d(gauge_data_for_moose)
print("dt_control_func", dt_control_func.time_sequence)

builder.add_executioner_block(
    end_time=total_time,
    dt=3600*24*5,
    time_stepper_type='TimeSequenceStepper',
    stepper_config=dt_control_func
)

builder.add_preconditioning_block(active_preconditioner='mumps')
builder.add_outputs_block(exodus=False, csv=True)

builder.generate_input_file(input_file_name)
print(builder.postprocessor_info)
builder.plot_geometry()
print(f"\nSuccessfully generated MOOSE input file at: {input_file_name}")

#%% --- 7. Run Simulation ---
print("\n--- Starting MOOSE Simulation Runner ---")
try:
    moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
    mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    if not os.path.exists(moose_executable):
        print(f"ERROR: MOOSE executable not found at '{moose_executable}'. Please update the path.")
    else:
        runner = MooseRunner(moose_executable_path=moose_executable, mpiexec_path=mpiexec_path)
        success, stdout, stderr = runner.run(
            input_file_path=input_file_name,
            output_directory=output_dir,
            num_processors=20,
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

#%% Visualize the results
from fiberis.utils import viz_utils
viz_utils.plot_point_samplers(folder=output_dir, output_dir="figs/10042025")
viz_utils.plot_vector_samplers(folder=output_dir, output_dir="figs/10042025")

#%% Compare the gauge data with simulation results
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

ps_reader = MOOSEPointSamplerReader()
ps_reader.read(folder=output_dir, variable_index=1)

simulation_dataframe = Data1D()
simulation_dataframe.data = ps_reader.data
simulation_dataframe.taxis = ps_reader.taxis
simulation_dataframe.start_time = gauge_data_for_moose.start_time
simulation_dataframe.data[0] = gauge_data_for_moose.data[0] # Match the first point, initial pressure in Pa

#%% find which gauge to compare
np.argmin(np.abs(dataframe_gauge_md.data - (gauge_mds[0]))) # np.int64(7) <-- Choose gauge 8, for index start from 0

gauge_path = "data/fiberis_format/s_well/gauges/gauge8_data_swell.npz"
gauge_data_field = Data1D()
gauge_data_field.load_npz(gauge_path)

#%% Plot the data
simulation_dataframe.data = simulation_dataframe.data / 6894.76 # Pa to psi

fig, ax = plt.subplots(figsize=(8,4))
simulation_dataframe.plot(ax=ax, use_timestamp=True, use_legend=True, label='Simulated pressure')
gauge_data_field.plot(ax=ax, use_timestamp=True, use_legend=True, label='Measured pressure (field)')
ax.set_title('Gauge 8 Pressure Comparison')
ax.set_ylabel('Pressure (psi)')
plt.tight_layout()
plt.show()

#%% Compare the pressure profile vs the injection curve
fig, ax = plt.subplots(figsize=(8,4))
gauge_data_for_moose.data = gauge_data_for_moose.data / 6894.76 # Pa to psi
simulation_dataframe.plot(ax=ax, use_timestamp=True, use_legend=True, label='Simulated pressure')
gauge_data_for_moose.plot(ax=ax, use_timestamp=True, use_legend=True, label='Injection pressure')
ax.set_title('Gauge 8 Pressure Comparison, simulation vs injection data')
# set y label
ax.set_ylabel('Pressure (psi)')
plt.tight_layout()
plt.show()

#%% Compare the injection pressure with the gauge data
plt, ax = plt.subplots(figsize=(8,4))
gauge_data_for_moose.plot(ax=ax, use_timestamp=True, use_legend=True, label='Injection pressure')
gauge_data_field.plot(ax=ax, use_timestamp=True, use_legend=True, label='Measured pressure (field)')
ax.set_title('Injection Pressure Comparison')
ax.set_ylabel('Pressure (psi)')
plt.tight_layout()
plt.show()
