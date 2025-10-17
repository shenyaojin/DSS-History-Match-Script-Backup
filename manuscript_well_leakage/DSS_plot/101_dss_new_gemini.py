# %% Load lib
import numpy as np
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Data1D import Data1D_Gauge
import matplotlib.pyplot as plt
import datetime

# %% Load data.
datapath = "data/legacy/s_well/dss_data.npz"

DSS_data = Data2D_XT_DSS.DSS2D()
dataframe_tmp = np.load(datapath, allow_pickle=True)

# %% Try to load data to Data2D_XT_DSS
DSS_data.data = dataframe_tmp['value']
DSS_data.daxis = dataframe_tmp['daxis']

taxis = dataframe_tmp['taxis']

# %% Convert this datetime array to seconds
taxis_sec = [(t - taxis[0]).total_seconds() for t in taxis]
DSS_data.taxis = np.array(taxis_sec)
DSS_data.start_time = taxis[0]

# %% Do processing
DSS_data.select_depth(11500, 16700)
DSS_data.select_time(0, 400000)
extent = [DSS_data.taxis[0], DSS_data.taxis[-1], DSS_data.daxis[-1], DSS_data.daxis[0]]
cx = np.array([-1, 1])



# %% Plot all
calibration_value = 97.8
x1, x2, x3 = 13705, 15069, 16276
depths = np.array([x1, x2, x3]) - calibration_value

# Get channels for these depths
channels = []
for depth in depths:
    chan_val = DSS_data.get_value_by_depth(depth)
    channels.append(chan_val)

# Create the figure for the plot layout
fig = plt.figure(figsize=(12, 6))

# Define the overall grid shape. A (7, 10) grid allows for the specified spans.
grid_shape = (7, 10)

# --- Define all axes first for clarity and control ---

# Top-left axis for the waterfall plot
ax0 = plt.subplot2grid(grid_shape, (0, 0), rowspan=5, colspan=5)

# Bottom-left axis for the new line plot (this will be our main reference x-axis)
ax_new = plt.subplot2grid(grid_shape, (5, 0), rowspan=1, colspan=5)

# A list to hold the three axes on the right
axes_right = []
for i in range(3):
    ax = plt.subplot2grid(grid_shape, (i * 2, 5), rowspan=2, colspan=5)
    axes_right.append(ax)


# --- Plot data and set properties ---

# Plot the waterfall (ax0)
im = DSS_data.plot(ax=ax0, aspect='auto', useTimeStamp=False, cmap='bwr')
ax0.set_ylabel("MD (ft)")
ax0.set_xlabel("")
im.set_clim(-2, 2)
# Plot horizontal lines for each channel
for depth in depths:
    ax0.axhline(depth, color='k', linestyle='--', linewidth=1.5)

# Plot the new line plot (ax_new)
ax_new.plot(DSS_data.taxis, channels[0])
ax_new.set_ylabel("DSS strain")
ax_new.grid(True)

# Plot the three right-hand plots
for i, (ax, y) in enumerate(zip(axes_right, channels)):
    ax.plot(DSS_data.taxis, y)
    ax.set_ylabel("DSS strain")
    ax.grid(True)


# --- Align Axes and Manage Labels (The Fix) ---

# Share the x-axis of all plots with ax_new to ensure they are aligned
ax0.sharex(ax_new)
for ax in axes_right:
    ax.sharex(ax_new)

# After sharing, explicitly hide tick labels on plots that are not on the bottom row.
# This ensures a clean look where the time axis is not repeated.
plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(axes_right[0].get_xticklabels(), visible=False)
plt.setp(axes_right[1].get_xticklabels(), visible=False)

# Set the x-axis label only for the two bottom plots
ax_new.set_xlabel("Time (s)")
axes_right[2].set_xlabel("Time (s)")

# --- Final Layout Adjustment ---
# Use tight_layout to automatically adjust subplot params for a nice fit.
plt.tight_layout()
# Adjust vertical spacing between plots to prevent overlap
plt.subplots_adjust(hspace=0.5)
plt.show()

