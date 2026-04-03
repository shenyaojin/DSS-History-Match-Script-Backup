#%% Import
import datetime
import numpy as np
import matplotlib.pyplot as plt
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# Define folders
simplified_sim_folder = "output/0313_validation/full_run"
all_kernel_sim_folder = "output/0313_validation/full_run_all_kernel"

# Time extraction profiles
full_bgn_time_folder = "scripts/DSS_history_match/validation/data/timestepper_profile_full.npz"
interf_bgn_time_folder = "scripts/DSS_history_match/validation/data/interference.npz"

#%% Load simplified data
simplified_loader = MOOSEVectorPostProcessorReader()
simplified_loader.read(simplified_sim_folder)
simplified_df = simplified_loader.to_analyzer()

# Add start time info
gauge_df = Data1DGauge()
gauge_df.load_npz(full_bgn_time_folder)
simplified_df.start_time = gauge_df.start_time

#%% Load all_kernel data
all_kernel_loader = MOOSEVectorPostProcessorReader()
all_kernel_loader.read(all_kernel_sim_folder)
all_kernel_df = all_kernel_loader.to_analyzer()
all_kernel_df.start_time = gauge_df.start_time

#%% Define Range (Following 104_output_extractor.py logic)
# Load interference gauge to get the specific time window used in 104
interf_gauge_df = Data1DGauge()
interf_gauge_df.load_npz(interf_bgn_time_folder)

# In 104: interf_dataframe starts at start_time + 50000s
t_start = interf_gauge_df.start_time + datetime.timedelta(seconds=50000)
t_end = interf_gauge_df.get_end_time(use_timestamp=True)

# Apply Time Slicing
simplified_df.select_time(t_start, t_end)
all_kernel_df.select_time(t_start, t_end)

#%% Calibration (Subtracting initial strain of the sliced window)
# In 104, calibration happens after time slicing, using the first slice of the new window
simplified_calibrated = simplified_df.copy()
all_kernel_calibrated = all_kernel_df.copy()

# Subtract the data at index 0 (start of sliced window)
for iter_t in range(len(simplified_df.taxis)):
    simplified_calibrated.data[:, iter_t] = simplified_df.data[:, iter_t] - simplified_df.data[:, 0]

for iter_t in range(len(all_kernel_df.taxis)):
    all_kernel_calibrated.data[:, iter_t] = all_kernel_df.data[:, iter_t] - all_kernel_df.data[:, 0]

#%% Apply Depth Slicing (40m - 80m as in 104)
depth_min, depth_max = 40, 80
simplified_calibrated.select_depth(depth_min, depth_max)
all_kernel_calibrated.select_depth(depth_min, depth_max)

#%% Accuracy Analysis: Residuals
residual_data = all_kernel_calibrated.data - simplified_calibrated.data
residual_df = all_kernel_calibrated.copy()
residual_df.data = residual_data

# Print basic stats
rmse = np.sqrt(np.nanmean(residual_data**2)) * 1e6
print(f"Overall RMSE (Microstrain) in sliced range: {rmse:.4f}")

#%% Plotting Heatmaps
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Color limits from 104
v_min, v_max = -1e-5, 1e-5

# Simplified (Full Run)
im1 = simplified_calibrated.plot(ax=ax1, use_timestamp=True, cmap='bwr', vmin=v_min, vmax=v_max, method='pcolormesh')
fig.colorbar(im1, ax=ax1, label='Strain')
ax1.set_title("Simplified Model (FULL RUN - Sliced)")
ax1.set_ylabel("Relative Distance (m)")

# All Kernel
im2 = all_kernel_calibrated.plot(ax=ax2, use_timestamp=True, cmap='bwr', vmin=v_min, vmax=v_max, method='pcolormesh')
fig.colorbar(im2, ax=ax2, label='Strain')
ax2.set_title("All Kernel Model (FULL RUN ALL KERNEL - Sliced)")
ax2.set_ylabel("Relative Distance (m)")

# Residual (Accuracy Loss)
v_resid = 1e-6 
im3 = residual_df.plot(ax=ax3, use_timestamp=True, cmap='seismic', vmin=-v_resid, vmax=v_resid, method='pcolormesh')
fig.colorbar(im3, ax=ax3, label='Strain Diff')
ax3.set_title("Residual (All Kernel - Simplified)")
ax3.set_ylabel("Relative Distance (m)")
ax3.set_xlabel("Time")

plt.tight_layout()
plt.savefig("scripts/DSS_history_match/validation/106_comparison_heatmap_sliced.png")
plt.show()

#%% 1D Comparison at Depth 55m
depth_to_plot = 55
chan_simple = simplified_calibrated.get_value_by_depth(depth_to_plot) * 1e6
chan_all = all_kernel_calibrated.get_value_by_depth(depth_to_plot) * 1e6

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(simplified_calibrated.taxis, chan_simple, label='Simplified')
ax.plot(all_kernel_calibrated.taxis, chan_all, label='All Kernel', linestyle='--')
ax.set_title(f"Comparison at Depth {depth_to_plot}m (Sliced)")
ax.set_ylabel(r"Microstrain ($\mu\epsilon$)")
ax.set_xlabel("Time")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("scripts/DSS_history_match/validation/106_comparison_depth_55m_sliced.png")
plt.show()

#%% Slice at 72 hours (if within range)
slice_time = simplified_calibrated.start_time + datetime.timedelta(hours=72)
try:
    time_slice_simple, _ = simplified_calibrated.get_value_by_time(slice_time)
    time_slice_all, _ = all_kernel_calibrated.get_value_by_time(slice_time)

    # Scaling
    time_slice_simple *= 1e6
    time_slice_all *= 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(simplified_calibrated.daxis, time_slice_simple, label='Simplified (72h)')
    ax.plot(all_kernel_calibrated.daxis, time_slice_all, label='All Kernel (72h)', linestyle='--')
    ax.set_title(f"Strain vs Depth at {slice_time}")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel(r"Microstrain ($\mu\epsilon$)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not plot 72h slice: {e}")
