# QC 109 series full code
# I'm interested in what the pressure/strain looks like for the full QC 109 series
# Basically it's the combination of 109r2 and 109r3 with extra enhancements with visualizations
# Shenyao Jin, shenyaojin@mines.edu, 12/05/2025

import numpy as np
import matplotlib.pyplot as plt
import os

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.templates.baseline_model_generator import build_baseline_model, post_processor_info_extractor, misfit_calculator
from fiberis.moose.runner import MooseRunner

print("working directory:", os.getcwd())

# Run single parameter set simulation

project_name = "1206_qc_single_param_full"
builder = build_baseline_model(project_name=project_name,
                               srv_perm=2.87e-18,
                               fracture_perm=1.09e-15,
                               matrix_perm=1e-21,
                               ny_per_layer_half=100,
                               bias_y=1.15
                               ) # This parameter set is stable

builder.plot_geometry()
# Output the model
output_dir = f"output/{project_name}"
os.makedirs(output_dir, exist_ok=True)
input_file_path = os.path.join(output_dir, f"{project_name}_input.i")

builder.generate_input_file(output_filepath=input_file_path)
runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)
# #
success, stdout, stderr = runner.run(
    input_file_path=input_file_path,
    output_directory=output_dir,
    num_processors=20,
    log_file_name="simulation.log",
    stream_output=True
)

# 1. Post-process the simulation results
pressure_dataframe, strain_dataframe = post_processor_info_extractor(output_dir=output_dir)

# 2. Load and pre-process observed DSS data
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)
scale_factor = 7 # Compensate for imperfect coupling

mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
DSSdata.select_depth(14880, 14900) # <- Select depth range of interest.
DSSdata.data = DSSdata.data * scale_factor / 1e6 # Convert to microstrain

start_time = DSSdata.start_time
pressure_dataframe.start_time = start_time
strain_dataframe.start_time = start_time

# 3. Synchronize simulated data with observed data time range
pressure_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())
strain_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())
# strain_dataframe.data[:, 0] = 0  # <- Set first column to zero to avoid NaN issues
strain_dataframe.data = strain_dataframe.data[:, 1:]
strain_dataframe.taxis = strain_dataframe.taxis[1:] # <- Adjust time axis accordingly
print(strain_dataframe)

# Calibrate simulated strain data by removing the initial value for each channel
if strain_dataframe.data.shape[0] > 0:
    strain_dataframe.data -= strain_dataframe.data[:, 0][:, np.newaxis]
    print("Applied baseline correction to each channel in the strain data.")
    print(np.min(strain_dataframe.data), np.max(strain_dataframe.data))

#%% 4. Misfit Calculation
# Define fracture center indices for observed and simulated data
center_frac_depth_obs = 14888.97
ind_obs = np.argmin(np.abs(DSSdata.daxis - center_frac_depth_obs))
ind_sim = len(strain_dataframe.daxis) // 2  # <- select center depth for simulated data

# Define weight matrix for misfit calculation (from 109r3)
weight_matrix = np.array([1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1])

misfit_val = misfit_calculator(weight_matrix=weight_matrix,
                               sim_fracture_center_ind=ind_sim,
                               observed_data_fracture_center_ind=ind_obs,
                               simulated_data=strain_dataframe,
                               observed_data=DSSdata,
                               save_path=f"figs/{project_name}_misfit_results")

print(f"Calculated Misfit Value: {misfit_val}")


#%% 5. Plotting and Visualization

# Plot 1: Comparison of Simulated Pressure and Strain (New Requirement)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot simulated pressure
pressure_dataframe.plot(ax=ax1, use_timestamp=False, cmap="bwr", method='pcolormesh',
                        colorbar=True, clabel="Pressure (Pa)", vmin=1.25e7, vmax=1.29e7)
ax1.set_title("Simulated Pressure Profile")
ax1.set_ylabel("Depth (m)")

# Plot simulated strain
if strain_dataframe.data is not None and strain_dataframe.data.size > 0:
    max_abs_strain = np.max(np.abs(strain_dataframe.data))
    strain_dataframe.plot(ax=ax2, use_timestamp=False, cmap="bwr", method='pcolormesh',
                          vmin=-max_abs_strain, vmax=max_abs_strain, colorbar=True, clabel="Strain")
else:
    strain_dataframe.plot(ax=ax2, use_timestamp=False, cmap="bwr", method='pcolormesh',
                          colorbar=True, clabel="Strain")
ax2.set_title("Simulated Strain Profile")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Depth (ft)")

plt.tight_layout()
plt.suptitle("Comparison of Simulated Pressure and Strain", y=1.02)
plt.show()


# Plot 2: Observed DSS Data with Fracture Bounds (from 109r3)
fig_obs, ax_obs = plt.subplots(figsize=(10, 6))
upper_bound = 14886.5
lower_bound = 14891.4
DSSdata.plot(ax=ax_obs, use_timestamp=False, cmap='bwr', vmin=-1e-5, vmax=1e-5)
ax_obs.axhline(y=center_frac_depth_obs, color='k', linestyle='--', label=f'Center: {center_frac_depth_obs:.2f} ft')
ax_obs.axhline(y=upper_bound, color='k', linestyle='-', label=f'Upper Bound: {upper_bound:.2f} ft')
ax_obs.axhline(y=lower_bound, color='k', linestyle='-', label=f'Lower Bound: {lower_bound:.2f} ft')
ax_obs.set_title("Observed DSS Data at POW-S with Fracture Bounds")
ax_obs.set_ylabel("Depth (ft)")
ax_obs.legend()
plt.tight_layout()
plt.show()

# Plot 3: Get the simulated strain and compare with DSS data at center depth
chan_data_sim = strain_dataframe.get_value_by_depth(0.5 * (strain_dataframe.daxis[0] + strain_dataframe.daxis[-1]))
chan_data_obs = DSSdata.get_value_by_depth(center_frac_depth_obs)

plt.figure(figsize=(10, 6))
plt.plot(strain_dataframe.taxis, chan_data_sim, label="Simulated Strain (Center)")
plt.plot(DSSdata.taxis, chan_data_obs, label="Observed DSS Strain (Center)")
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Simulated vs. Observed Strain at Fracture Center")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot 4 & 5: Combined Time Slices at time = 320000
time_slice_strain, actual_time = strain_dataframe.get_value_by_time(320000)
time_slice_pressure, actual_time_pressure = pressure_dataframe.get_value_by_time(320000)

fig_ts, (ax_strain, ax_pressure) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

ax_strain.plot(time_slice_strain, strain_dataframe.daxis, '-o', label='Strain')
ax_strain.set_xlabel("Strain")
ax_strain.set_ylabel("Depth (m)")
ax_strain.set_title(f"Strain Profile at {actual_time:.0f} s")
ax_strain.grid(True)
ax_strain.legend()

ax_pressure.plot(time_slice_pressure, pressure_dataframe.daxis * 3.28084, '-o', color='red', label='Pressure')
ax_pressure.set_xlabel("Pressure (Pa)")
ax_pressure.set_title(f"Pressure Profile at {actual_time_pressure:.0f} s")
ax_pressure.grid(True)
ax_pressure.legend()

plt.suptitle(f"Simulated Profiles at Time = {actual_time:.0f} s", y=1.02)
plt.tight_layout()
plt.show()

#%% 6. New plot using the co-plot function
from fiberis.utils.viz_utils import plot_dss_and_gauge_co_plot
from fiberis.analyzer.Data1D.core1D import Data1D

# Extract center channel from the simulated pressure data to create a 1D object
center_depth_sim = pressure_dataframe.daxis[len(pressure_dataframe.daxis) // 2]
center_pressure_data = pressure_dataframe.get_value_by_depth(center_depth_sim)

center_channel_pressure = Data1D(
    data=center_pressure_data,
    taxis=pressure_dataframe.taxis,
    start_time=pressure_dataframe.start_time,
    name=f"Pressure at Center Depth ({center_depth_sim:.2f} m)"
)

# Use the new plotting function
plot_dss_and_gauge_co_plot(
    data2d=pressure_dataframe,
    data1d=center_channel_pressure,
    d2_plot_args={'title': "Simulated Pressure with Center Channel Trace",
                  'clabel': "Pressure (Pa)",
                  'clim': (1.25e7, 1.29e7)}
)
plt.show()
