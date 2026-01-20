# Clean-up version of 'scripts/DSS_history_match/109r2_test_builder_function_v3.py'
# Shenyao Jin, 01/2026
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.moose.templates.baseline_model_generator import build_baseline_model, post_processor_info_extractor, misfit_calculator
from fiberis.moose.runner import MooseRunner
import os

print("working directory:", os.getcwd())
conversion_factor = 0.328084  # meter to feet

project_name = "0114_degradation_test"
builder = build_baseline_model(project_name=project_name,
                               srv_perm=2.87e-16,
                               fracture_perm=1.09e-13,
                               matrix_perm=1e-20,
                               srv_height_ft=2,
                               ny_per_layer_half=100,
                               bias_y=1.08,
                               start_offset_y=160,
                               end_offset_y=160
                               ) # This parameter set is stable

output_dir = f"output/{project_name}"
fig_dir = f"figs/{project_name}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
print(f"Figures will be saved to: {os.path.abspath(fig_dir)}")
input_file_path = os.path.join(output_dir, f"{project_name}_input.i")

builder.generate_input_file(output_filepath=input_file_path)

builder.plot_geometry()

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

# --- MISFIT CALCULATION ---
print("\n--- Calculating Misfit ---")
# Create a new DSS data object for misfit to avoid conflicts with plotting data objects
DSSdata_for_misfit = DSS2D()
DSSdata_for_misfit.load_npz(DSS_datapath)

# Pre-process DSS data for misfit (same as reference script)
mds_misfit = DSSdata_for_misfit.daxis
ind_misfit = (mds_misfit > 7500) & (mds_misfit < 15000)
drift_val_misfit = np.median(DSSdata_for_misfit.data[ind_misfit, :], axis=0)
DSSdata_for_misfit.data -= drift_val_misfit.reshape((1, -1))
DSSdata_for_misfit.select_time(DSSdata.start_time, DSSdata.get_end_time())
DSSdata_for_misfit.select_depth(14880, 14900)

# Scale data: convert from microstrain to strain and apply coupling factor
SCALE_FACTOR = 6.0
DSSdata_for_misfit.data = DSSdata_for_misfit.data * SCALE_FACTOR / 1e6
print("Misfit DSS data pre-processing complete.")

# Define fracture center and weight matrix for misfit calculation
CENTER_FRAC_DEPTH_OBS = 14888.97
ind_obs = np.argmin(np.abs(DSSdata_for_misfit.daxis - CENTER_FRAC_DEPTH_OBS))
WEIGHT_MATRIX = np.array([1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1])
print(f"Observed fracture center depth for misfit: {CENTER_FRAC_DEPTH_OBS} ft (index: {ind_obs})")

# Find center of simulated data
ind_sim = len(strain_dataframe.daxis) // 2

misfit_val = misfit_calculator(
    weight_matrix=WEIGHT_MATRIX,
    sim_fracture_center_ind=ind_sim,
    observed_data_fracture_center_ind=ind_obs,
    simulated_data=strain_dataframe, # Using the raw strain data before it's scaled for plotting
    observed_data=DSSdata_for_misfit,
    save_path=fig_dir
)
print(f"--- Misfit Calculated: {misfit_val:.4e} ---")


# Plot comparison
# There will be totally 2 plots. One is strain QC plot, and another one
# is the comparison between simulated strain and DSS-measured strain.

#%% QC plot 1: simulated strain data overview + example traces + borehole gauge data
strain_max = np.max(np.abs(strain_dataframe.data))
center_depth_ind = len(strain_dataframe.daxis) // 2
center_depth = strain_dataframe.daxis[center_depth_ind]
example_data_upper_data = strain_dataframe.get_value_by_depth(center_depth - 1) # ~ 3ft above center
example_data_upper2_data = strain_dataframe.get_value_by_depth(center_depth - 2) # ~ 6ft above center

example_data_upper = Data1D()
example_data_upper.taxis = strain_dataframe.taxis
example_data_upper.data = example_data_upper_data
example_data_upper.name = f"Strain at {round(center_depth - 3)} ft"
example_data_upper.start_time = strain_dataframe.start_time

example_data_upper2 = Data1D()
example_data_upper2.taxis = strain_dataframe.taxis
example_data_upper2.data = example_data_upper2_data
example_data_upper2.name = f"Strain at {round(center_depth - 6)} ft"
example_data_upper2.start_time = strain_dataframe.start_time

example_data_center = Data1D()
example_data_center.taxis = strain_dataframe.taxis
example_data_center.data = strain_dataframe.get_value_by_depth(center_depth)
example_data_center.name = f"Strain at {round(center_depth)} ft"
example_data_center.start_time = strain_dataframe.start_time

borehole_gauge = Data1D()
borehole_gauge.load_npz("data/fiberis_format/prod/gauges/pressure_g1.npz")

# 1. plot the strain data from simulation.
plt.figure(figsize=(11, 6))
# AX1: the simulated strain plot
ax1 = plt.subplot2grid((6,8), (0,0), colspan=5, rowspan=4)
strain_dataframe.plot(ax=ax1, method='pcolormesh', use_timestamp=False, xaxis_rotation=90,
                      clim=[-strain_max, strain_max], cmap='bwr')
ax1.set_title("Simulated Strain Data")
ax1.set_ylabel("MD (m)")
ax1.tick_params(labelbottom=False)

# AX2: the borehole gauge data plot
borehole_gauge.select_time(strain_dataframe.start_time, strain_dataframe.get_end_time())

ax2 = plt.subplot2grid((6,8), (4,0), colspan=5, rowspan=3, sharex=ax1)
borehole_gauge.plot(ax=ax2, use_timestamp=False)
ax2.set_ylabel("Pressure (psi)")

# AX3: example traces
ax3 = plt.subplot2grid((6,8), (0,5), colspan=3, rowspan=2)
example_data_upper.plot(ax=ax3, use_timestamp=True)
ax3.set_title(f"Example Strain Traces: Upper {round(center_depth - 3)} ft")
ax3.set_ylabel("Strain")
ax3.tick_params(labelbottom=False)

ax4 = plt.subplot2grid((6,8), (2,5), colspan=3, rowspan=2, sharex=ax3)
example_data_upper2.plot(ax=ax4, use_timestamp=True)
ax4.set_title(f"Example Strain Traces: Upper {round(center_depth - 6)} ft")
ax4.set_ylabel("Strain")
ax4.tick_params(labelbottom=False)

ax5 = plt.subplot2grid((6,8), (4,5), colspan=3, rowspan=2, sharex=ax3)
example_data_center.plot(ax=ax5, use_timestamp=True)
ax5.set_title(f"Example Strain Traces: Center {round(center_depth)} ft")
ax5.set_ylabel("Strain")
ax5.set_xlabel("Time (absolute)")
plt.tight_layout()
qc1_path = os.path.join(fig_dir, "qc_plot_1_strain_overview.png")
plt.savefig(qc1_path)
plt.close()
print(f"Saved QC plot 1 to {qc1_path}")

#%% QC plot 1.5: simulated pressure data overview + example traces
# Use the same center depth as in the strain plot for direct comparison
example_pressure_upper_data = pressure_dataframe.get_value_by_depth(center_depth - 1) # ~ 3ft above center
example_pressure_upper2_data = pressure_dataframe.get_value_by_depth(center_depth - 2) # ~ 6ft above center
example_pressure_center_data = pressure_dataframe.get_value_by_depth(center_depth)

example_pressure_upper = Data1D()
example_pressure_upper.taxis = pressure_dataframe.taxis
example_pressure_upper.data = example_pressure_upper_data
example_pressure_upper.name = f"Pressure at {round(center_depth - 3)} ft"
example_pressure_upper.start_time = pressure_dataframe.start_time

example_pressure_upper2 = Data1D()
example_pressure_upper2.taxis = pressure_dataframe.taxis
example_pressure_upper2.data = example_pressure_upper2_data
example_pressure_upper2.name = f"Pressure at {round(center_depth - 6)} ft"
example_pressure_upper2.start_time = pressure_dataframe.start_time

example_pressure_center = Data1D()
example_pressure_center.taxis = pressure_dataframe.taxis
example_pressure_center.data = example_pressure_center_data
example_pressure_center.name = f"Pressure at {round(center_depth)} ft"
example_pressure_center.start_time = pressure_dataframe.start_time

# 1. plot the pressure data from simulation.
plt.figure(figsize=(11, 6))
# AX1: the simulated pressure plot
ax1 = plt.subplot2grid((6,8), (0,0), colspan=5, rowspan=4)
pressure_dataframe.plot(ax=ax1, method='pcolormesh', use_timestamp=False, xaxis_rotation=90, cmap='viridis')
ax1.set_title("Simulated Pressure Data")
ax1.set_ylabel("MD (m)")
ax1.tick_params(labelbottom=False)

# AX2: the borehole gauge data plot
ax2 = plt.subplot2grid((6,8), (4,0), colspan=5, rowspan=3, sharex=ax1)
borehole_gauge.plot(ax=ax2, use_timestamp=False)
ax2.set_ylabel("Pressure (psi)")

# AX3: example traces
ax3 = plt.subplot2grid((6,8), (0,5), colspan=3, rowspan=2)
example_pressure_upper.plot(ax=ax3, use_timestamp=True)
ax3.set_title(f"Example Pressure Traces: Upper {round(center_depth - 3)} ft")
ax3.set_ylabel("Pressure")
ax3.tick_params(labelbottom=False)

ax4 = plt.subplot2grid((6,8), (2,5), colspan=3, rowspan=2, sharex=ax3)
example_pressure_upper2.plot(ax=ax4, use_timestamp=True)
ax4.set_title(f"Example Pressure Traces: Upper {round(center_depth - 6)} ft")
ax4.set_ylabel("Pressure")
ax4.tick_params(labelbottom=False)

ax5 = plt.subplot2grid((6,8), (4,5), colspan=3, rowspan=2, sharex=ax3)
example_pressure_center.plot(ax=ax5, use_timestamp=True)
ax5.set_title(f"Example Pressure Traces: Center {round(center_depth)} ft")
ax5.set_ylabel("Pressure")
ax5.set_xlabel("Time (absolute)")
plt.tight_layout()
qc1_5_path = os.path.join(fig_dir, "qc_plot_1.5_pressure_overview.png")
plt.savefig(qc1_5_path)
plt.close()
print(f"Saved QC plot 1.5 to {qc1_5_path}")

#%% QC plot 2: simulated strain vs DSS-measured strain at the same depth
# The info of depth are from 108p_DSS_area_select.py

# Add a magnitude calibration factor to the DSS data
DSSdata_calibrated = DSSdata.copy()
DSSdata_calibrated.data = DSSdata.data * 6  # Calibration factor
DSS_frac_center = 14888

DSSdata_calibrated.select_depth(DSS_frac_center -10, DSS_frac_center +10)

# Strain dataframe convert to mu strain
strain_dataframe.data = strain_dataframe.data * 1e6  # Convert to microstrain
# Select depth, since the strain_dataframe is in m, convert to ft
strain_dataframe.select_depth(center_depth - 10 * conversion_factor, center_depth + 10 * conversion_factor)

plt.figure(figsize=(12, 6))
# AX1: DSS-measured strain plot
ax1 = plt.subplot2grid((6, 8), (0,0), colspan=4, rowspan=3)
im1 = DSSdata.plot(ax=ax1, use_timestamp=False, cmap='bwr', clim=0.1 * np.array([-strain_max * 1e6, strain_max * 1e6]))
ax1.axhline(y = DSS_frac_center, color='k', linestyle='--', label='Fracture Center')
ax1.set_title("DSS-Measured Strain Data (Calibrated * 6)")
ax1.set_ylabel("Depth (ft)")
ax1.set_xlabel("Time (absolute)")
# Add colorbar
cbar = plt.colorbar(im1, ax=ax1)
cbar.set_label("Strain (microstrain)")
ax1.tick_params(labelbottom=False)

# AX2: Simulated strain plot
ax2 = plt.subplot2grid((6, 8), (3, 0), colspan=4, rowspan=3)
im2 = strain_dataframe.plot(ax=ax2, use_timestamp=False, cmap='bwr', clim=[-strain_max * 1e6, strain_max * 1e6], method='pcolormesh')
ax2.axhline(y = center_depth, color='k', linestyle='--', label='Fracture Center')
ax2.set_title("Simulated Strain Data")
ax2.set_ylabel("Depth (ft)")
ax2.set_xlabel("Time (absolute)")
# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label("Strain (microstrain)")

# AX3: Example trace comparison at fracture center
# get center value of DSS data
DSS_center_data = DSSdata_calibrated.get_value_by_depth(DSS_frac_center)
example_DSS_center = Data1D()
example_DSS_center.taxis = DSSdata_calibrated.taxis
example_DSS_center.data = DSS_center_data
example_DSS_center.name = f"DSS Strain at {DSS_frac_center} ft"
example_DSS_center.start_time = DSSdata_calibrated.start_time

# get upper value of DSS data, 3 ft above center
DSS_upper_data = DSSdata_calibrated.get_value_by_depth(DSS_frac_center + 3)
example_DSS_upper = Data1D()
example_DSS_upper.taxis = DSSdata_calibrated.taxis
example_DSS_upper.data = DSS_upper_data
example_DSS_upper.name = f"DSS Strain at {DSS_frac_center -3} ft"
example_DSS_upper.start_time = DSSdata_calibrated.start_time
# get upper2 value of DSS data, 6 ft above center
DSS_upper2_data = DSSdata_calibrated.get_value_by_depth(DSS_frac_center + 6)
example_DSS_upper2 = Data1D()
example_DSS_upper2.taxis = DSSdata_calibrated.taxis
example_DSS_upper2.data = DSS_upper2_data
example_DSS_upper2.name = f"DSS Strain at {DSS_frac_center -6} ft"
example_DSS_upper2.start_time = DSSdata_calibrated.start_time

# Convert simulated strain data to microstrain for comparison
example_data_upper.data = example_data_upper.data * 1e6
example_data_upper2.data = example_data_upper2.data * 1e6
example_data_center.data = example_data_center.data * 1e6

# Plot comparison traces, the simulated strain traces are already defined above
ax3 = plt.subplot2grid((6, 8), (0,4), colspan=4, rowspan=2)
example_DSS_upper.plot(ax=ax3, use_timestamp=True, label='DSS Measured')
example_data_upper.plot(ax=ax3, use_timestamp=True, label='Simulated')
ax3.set_title(f"Strain Comparison at {DSS_frac_center -3} ft")
ax3.set_ylabel("Strain (microstrain)")
ax3.legend()
ax3.tick_params(labelbottom=False)

ax4 = plt.subplot2grid((6, 8), (2,4), colspan=4, rowspan=2)
example_DSS_upper2.plot(ax=ax4, use_timestamp=True, label='DSS Measured')
example_data_upper2.plot(ax=ax4, use_timestamp=True, label='Simulated')
ax4.set_title(f"Strain Comparison at {DSS_frac_center -6} ft")
ax4.set_ylabel("Strain (microstrain)")
ax4.legend()
ax4.tick_params(labelbottom=False)

ax5 = plt.subplot2grid((6, 8), (4,4), colspan=4, rowspan=2)
example_DSS_center.plot(ax=ax5, use_timestamp=True, label='DSS Measured')
example_data_center.plot(ax=ax5, use_timestamp=True, label='Simulated')
ax5.set_title(f"Strain Comparison at {DSS_frac_center} ft")
ax5.set_ylabel("Strain (microstrain)")
ax5.set_xlabel("Time (absolute)")
ax5.legend()
plt.tight_layout()
qc2_path = os.path.join(fig_dir, "qc_plot_2_strain_comparison.png")
plt.savefig(qc2_path)
plt.close()
print(f"Saved QC plot 2 to {qc2_path}")
