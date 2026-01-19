# Clean-up version of 'scripts/DSS_history_match/109r2_test_builder_function_v3.py'
# Shenyao Jin, 01/2026
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.moose.templates.baseline_model_generator import build_baseline_model
from fiberis.moose.templates.baseline_model_generator import post_processor_info_extractor
from fiberis.moose.runner import MooseRunner
import os

print("working directory:", os.getcwd())

project_name = "0114_degradation_test"
builder = build_baseline_model(project_name=project_name,
                               srv_perm=2.87e-16,
                               fracture_perm=1.09e-13,
                               matrix_perm=1e-20,
                               ny_per_layer_half=100,
                               bias_y=1.08
                               ) # This parameter set is stable

output_dir = f"output/{project_name}"
os.makedirs(output_dir, exist_ok=True)
input_file_path = os.path.join(output_dir, f"{project_name}_input.i")

builder.generate_input_file(output_filepath=input_file_path)

# builder.plot_geometry()
#
# runner = MooseRunner(
#     moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
#     mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
# )
# success, stdout, stderr = runner.run(
#     input_file_path=input_file_path,
#     output_directory=output_dir,
#     num_processors=20,
#     log_file_name="simulation.log",
#     stream_output=True
# )

pressure_dataframe, strain_dataframe = post_processor_info_extractor(output_dir=output_dir)

DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)

start_time = DSSdata.start_time
end_time = DSSdata.get_end_time()

pressure_dataframe.start_time = start_time
strain_dataframe.start_time = start_time

pressure_dataframe.data = pressure_dataframe.data[:, 1:]
strain_dataframe.data = strain_dataframe.data[:, 1:]
pressure_dataframe.taxis = pressure_dataframe.taxis[1:] - pressure_dataframe.taxis[1]
strain_dataframe.taxis = strain_dataframe.taxis[1:] - strain_dataframe.taxis[1]

pressure_dataframe.select_time(start_time, end_time)
strain_dataframe.select_time(start_time, end_time)

mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)

DSSdata.select_depth(14880, 14900)

pressure_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())
strain_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())

if strain_dataframe.data.shape[0] > 0:
    strain_dataframe.data -= strain_dataframe.data[:, 0][:, np.newaxis]
    print("Applied baseline correction to each channel in the strain data.")
    print(np.min(strain_dataframe.data), np.max(strain_dataframe.data))

# Plot comparison
# There will be totally 2 plots. One is strain QC plot, and another one
# is the comparison between simulated strain and DSS-measured strain.

#%% QC plot 1: simulated strain data overview + example traces + borehole gauge data
# strain_max = np.max(np.abs(strain_dataframe.data))
# center_depth_ind = len(strain_dataframe.daxis) // 2
# center_depth = strain_dataframe.daxis[center_depth_ind]
# example_data_upper_data = strain_dataframe.get_value_by_depth(center_depth - 1) # ~ 3ft above center
# example_data_upper2_data = strain_dataframe.get_value_by_depth(center_depth - 2) # ~ 6ft above center
#
# example_data_upper = Data1D()
# example_data_upper.taxis = strain_dataframe.taxis
# example_data_upper.data = example_data_upper_data
# example_data_upper.name = f"Strain at {center_depth - 1} ft"
# example_data_upper.start_time = strain_dataframe.start_time
#
# example_data_upper2 = Data1D()
# example_data_upper2.taxis = strain_dataframe.taxis
# example_data_upper2.data = example_data_upper2_data
# example_data_upper2.name = f"Strain at {center_depth - 2} ft"
# example_data_upper2.start_time = strain_dataframe.start_time
#
# example_data_center = Data1D()
# example_data_center.taxis = strain_dataframe.taxis
# example_data_center.data = strain_dataframe.get_value_by_depth(center_depth)
# example_data_center.name = f"Strain at {center_depth} ft"
# example_data_center.start_time = strain_dataframe.start_time
#
# borehole_gauge = Data1D()
# borehole_gauge.load_npz("data/fiberis_format/prod/gauges/pressure_g1.npz")
#
# # 1. plot the strain data from simulation.
# plt.figure(figsize=(11, 6))
# # AX1: the simulated strain plot
# ax1 = plt.subplot2grid((6,8), (0,0), colspan=5, rowspan=4)
# strain_dataframe.plot(ax=ax1, method='pcolormesh', use_timestamp=False, xaxis_rotation=90,
#                       clim=[-strain_max, strain_max], cmap='bwr')
# ax1.set_title("Simulated Strain Data")
# ax1.set_ylabel("Time (absolute)")
# ax1.tick_params(labelbottom=False)
#
# # AX2: the borehole gauge data plot
# borehole_gauge.select_time(strain_dataframe.start_time, strain_dataframe.get_end_time())
#
# ax2 = plt.subplot2grid((6,8), (4,0), colspan=5, rowspan=3, sharex=ax1)
# borehole_gauge.plot(ax=ax2, use_timestamp=False)
# ax2.set_ylabel("Pressure (psi)")
#
# # AX3: example traces
# ax3 = plt.subplot2grid((6,8), (0,5), colspan=3, rowspan=2)
# example_data_upper.plot(ax=ax3, use_timestamp=True)
# ax3.set_title("Example Strain Traces: Upper 1ft")
# ax3.set_ylabel("Strain")
# ax3.tick_params(labelbottom=False)
#
# ax4 = plt.subplot2grid((6,8), (2,5), colspan=3, rowspan=2, sharex=ax3)
# example_data_upper2.plot(ax=ax4, use_timestamp=True)
# ax4.set_title("Example Strain Traces: Upper 2ft")
# ax4.set_ylabel("Strain")
# ax4.tick_params(labelbottom=False)
#
# ax5 = plt.subplot2grid((6,8), (4,5), colspan=3, rowspan=2, sharex=ax3)
# example_data_center.plot(ax=ax5, use_timestamp=True)
# ax5.set_title("Example Strain Traces: Center")
# ax5.set_ylabel("Strain")
# ax5.set_xlabel("Time (absolute)")
# plt.tight_layout()
# plt.show()

#%% QC plot 2: simulated strain vs DSS-measured strain at the same depth
# The info of depth are from 108p_DSS_area_select.py
plt.figure(figsize=(12, 6))
