# In this script, I will visualize the output from the 108_misfit_func.py script
# and compare it with the real DSS data.
# Shenyao Jin, 11/12/2025
import datetime

import numpy as np
import os
import matplotlib.pyplot as plt

# I/O modules from fiberis
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

# --- 1. Load and Preprocess Real DSS Data ---
# Using the exact same code from 108p_DSS_area_select.py to ensure consistency.

# Define file paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"

# Load DSS data
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)

# Pre-process DSS data
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
# Select the specific depth range of interest as identified in 108p
DSSdata.select_depth(14885, 14894)


# --- 2. Load Simulated Vector Sampler Data ---
sim_data_path = "output/1101_DSS_SingleFrac_scanner/moose_simulations/model_instance_20251112_174228"

# Use the MOOSEVectorPostProcessorReader to load the data
vpp_reader = MOOSEVectorPostProcessorReader()
# Read the 'fiber_strain' vector postprocessor (usually the first one, id=0)
# and the 'strain_yy' variable (the first variable, index=1)
vpp_reader.read(directory=sim_data_path, post_processor_id=0, variable_index=1)

# Convert to a Data2D object for analysis
sim_data = vpp_reader.to_analyzer()
sim_data.start_time = datetime.datetime(2020, 2, 19, 3, 20, 00)
# --- 3. Process Simulated Data ---
# The daxis is in meters, so we convert feet to meters for selection
conversion_factor = 0.3048
center_depth_m = sim_data.daxis[len(sim_data.daxis) // 2]
half_range_m = (20 / 2) * conversion_factor # 20 ft range

# Select the center ~20 ft of the simulated data
sim_data.select_depth(center_depth_m - half_range_m, center_depth_m + half_range_m)
sim_data.select_time(DSSdata.start_time, DSSdata.get_end_time())

# --- 4. Plot Comparison ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Real DSS Data
DSSdata.plot(ax=ax1, use_timestamp=False, cmap='bwr', vmin=-1, vmax=1)
ax1.set_title("Real DSS Data (Selected Area)")
ax1.set_ylabel("Depth (ft)")

# Plot Simulated Data
# Note: The simulation output is strain, while DSS is microstrain.
# We will plot them on different color scales for now, but a scaling factor might be needed for direct misfit calculation.
sim_data.plot(ax=ax2, use_timestamp=False, cmap='bwr', vmin=-1e-5, vmax=1e-5)
ax2.set_title("Simulated Strain (Center ~20 ft)")
ax2.set_ylabel("Relative Distance (m)")
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()