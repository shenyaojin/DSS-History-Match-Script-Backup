#%% Import
import datetime

import numpy as np
import matplotlib.pyplot as plt
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# Define folder
full_sim_folder = "output/0313_validation/full_run"
interf_sim_folder = "output/0313_validation/interf_run"

# We need the start time info.
full_bgn_time_folder = "scripts/DSS_history_match/validation/data/timestepper_profile_full.npz"
interf_bgn_time_folder = "scripts/DSS_history_match/validation/data/interference.npz"

#%% Load data (full)
full_folder_loader = MOOSEVectorPostProcessorReader()
full_folder_loader.read(full_sim_folder)

full_dataframe = full_folder_loader.to_analyzer()
# Add start time info
full_gauge_dataframe = Data1DGauge()
full_gauge_dataframe.load_npz(full_bgn_time_folder)

full_dataframe.start_time = full_gauge_dataframe.start_time
print("the full simulation end time: ", full_gauge_dataframe.get_end_time())

#%% Load date (interf)
interf_folder_loader = MOOSEVectorPostProcessorReader()
interf_folder_loader.read(interf_sim_folder)

interf_dataframe = interf_folder_loader.to_analyzer()
# Add start time info
interf_gauge_dataframe = Data1DGauge()
interf_gauge_dataframe.load_npz(interf_bgn_time_folder)
interf_dataframe.start_time = interf_gauge_dataframe.start_time

# TECH
interf_dataframe.select_time(50000, float(interf_dataframe.taxis[-1]))

#%% Post-processing
full_dataframe.select_time(interf_dataframe.start_time, interf_gauge_dataframe.get_end_time(use_timestamp=True))

full_dataframe_calibrated =full_dataframe.copy()

# minus the data t=0 to calibrate the data
for iter_t in range(len(full_dataframe.taxis)):
    full_dataframe_calibrated.data[:, iter_t] = full_dataframe.data[:, iter_t] - full_dataframe.data[:, 0]

#%% Print time
print(full_dataframe_calibrated.start_time)
print(interf_dataframe.start_time)

full_dataframe_calibrated.select_depth(40, 80)
interf_dataframe.select_depth(40, 80)

#%% Plot the strain

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot interference result
im1 = interf_dataframe.plot(ax=ax1, use_timestamp=True, cmap='bwr', vmin=-1e-5, vmax=1e-5, method='pcolormesh')
fig.colorbar(im1, ax=ax1, label='Strain')
ax1.set_title("Simulated Strain (INTERF)")
ax1.set_ylabel("Relative Distance (m)")
ax1.set_xlabel("Time (s)")

# Plot FULL Simulated Data
im2 = full_dataframe_calibrated.plot(ax=ax2, use_timestamp=True, cmap='bwr', vmin=-1e-5, vmax=1e-5, method='pcolormesh')
fig.colorbar(im2, ax=ax2, label='Strain')
ax2.set_title("Simulated Strain (FULL)")
ax2.set_ylabel("Relative Distance (m)")
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

#%% Diagnose:
chan_55_full = full_dataframe_calibrated.get_value_by_depth(55) * 1e6
chan_56_full = full_dataframe_calibrated.get_value_by_depth(56) * 1e6

chan_55_interf = interf_dataframe.get_value_by_depth(55) * 1e6
chan_56_interf = interf_dataframe.get_value_by_depth(56) * 1e6

# Plot comparison per depth
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Depth 55m Comparison
ax1.plot(full_dataframe_calibrated.taxis, chan_55_full, label='FULL (55m)')
ax1.plot(interf_dataframe.taxis, chan_55_interf, label='INTERF (55m)', linestyle='--')
ax1.set_title("Comparison at Depth 55m")
ax1.set_ylabel(r"Microstrain ($\mu\epsilon$)")
ax1.grid(True)
ax1.legend()

# Depth 56m Comparison
ax2.plot(full_dataframe_calibrated.taxis, chan_56_full, label='FULL (56m)')
ax2.plot(interf_dataframe.taxis, chan_56_interf, label='INTERF (56m)', linestyle='--')
ax2.set_title("Comparison at Depth 56m")
ax2.set_ylabel(r"Microstrain ($\mu\epsilon$)")
ax2.set_xlabel("Time (s)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Slice along time axis at 24 hours after start
slice_time = full_dataframe_calibrated.start_time + datetime.timedelta(hours=72)

time_slice_full, _ = full_dataframe_calibrated.get_value_by_time(slice_time)
time_slice_interf, _ = interf_dataframe.get_value_by_time(slice_time)

# Scaling to microstrain
time_slice_full *= 1e6
time_slice_interf *= 1e6

# Plot the depth slices
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(full_dataframe_calibrated.daxis, time_slice_full, label='FULL (72h)')
ax.plot(interf_dataframe.daxis, time_slice_interf, label='INTERF (72h)', linestyle='--')

ax.set_title(f"Strain vs Depth at {slice_time}")
ax.set_xlabel("Depth / Relative Distance (m)")
ax.set_ylabel(r"Microstrain ($\mu\epsilon$)")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
