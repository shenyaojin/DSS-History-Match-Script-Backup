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

#%% Plot the strain

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Real DSS Data
interf_dataframe.plot(ax=ax1, use_timestamp=False, cmap='bwr', vmin=-1e-5, vmax=1e-5, method='pcolormesh')
ax1.set_title("Simulated Strain (INTERF)")
ax1.set_ylabel("Relative Distance (m)")
ax1.set_xlabel("Time (s)")

# Plot Simulated Data
# Note: The simulation output is strain, while DSS is microstrain.
# We will plot them on different color scales for now, but a scaling factor might be needed for direct misfit calculation.
full_dataframe_calibrated.plot(ax=ax2, use_timestamp=False, cmap='bwr', vmin=-1e-5, vmax=1e-5, method='pcolormesh')
ax2.set_title("Simulated Strain (FULL)")
ax2.set_ylabel("Relative Distance (m)")
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

#%% Diagnose:
center_chan_calibrated = full_dataframe_calibrated.get_value_by_depth(
    full_dataframe_calibrated.daxis[len(full_dataframe_calibrated.daxis)//2]
)
center_chan_interf = interf_dataframe.get_value_by_depth(
    interf_dataframe.daxis[len(interf_dataframe.daxis)//2]
)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(full_dataframe_calibrated.taxis, center_chan_calibrated)
ax2.plot(interf_dataframe.taxis, center_chan_interf)

plt.show()