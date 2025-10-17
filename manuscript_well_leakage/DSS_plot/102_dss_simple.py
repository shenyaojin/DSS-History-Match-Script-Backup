# %% Load lib
import numpy as np
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Data1D import Data1D_Gauge
import matplotlib.pyplot as plt
import datetime

# %% Load data.
# This script now assumes the 'fiberis' library is installed and the data path is correct.
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

# %% Define depths and get data channels
calibration_value = 97.8
x1 = 13705
# Note: x2 and x3 are no longer used for plotting but are kept for context.
x2, x3 = 15069, 16276
# The depths for the horizontal lines on the waterfall plot
depths_for_lines = np.array([x1, x2, x3]) - calibration_value
# The depth for the line plot below
depth_for_plot = x1 - calibration_value

# Get the specific channel for the line plot
channel_data = DSS_data.get_value_by_depth(depth_for_plot)

#%% Updated code for borehole pressure data
borehole_data_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"

borehole_pressure_dataframe = Data1D_Gauge.Data1DGauge()
borehole_pressure_dataframe.load_npz(borehole_data_path)

# %% Crop the borehole pressure data to match the DSS data time range
borehole_pressure_dataframe.crop(DSS_data.start_time, DSS_data.start_time + datetime.timedelta(seconds=DSS_data.taxis[-1]))

# %% Create the figure and axes
calibration_value = 97.8
x1, x2, x3 = 13705, 15069, 16276
depths = np.array([x1, x2, x3]) - calibration_value

channels = []
for depth in depths:
    chan_val = DSS_data.get_value_by_depth(depth)
    channels.append(chan_val)

# plot the waterfall
fig = plt.figure(figsize=(6, 6))
grid_shape = (4, 3)

ax0 = plt.subplot2grid(grid_shape, (0, 0), colspan=3, rowspan=3)
image0 = DSS_data.plot(ax=ax0, aspect='auto', use_timestamp=False, cmap='bwr')
ax0.set_ylabel("MD (ft)")
ax0.set_yticks([])
ax0.set_xlabel("")
ax0.set_xticks([])
ax0.tick_params(axis='x', length=0, color='none')
image0.set_clim(-2, 2)

for depth in depths:
    ax0.axhline(depth, color='k', linestyle='--', linewidth=1.5)

ax1 = plt.subplot2grid(grid_shape, (3, 0), colspan=3, rowspan=1)
ax1.plot(borehole_pressure_dataframe.taxis, borehole_pressure_dataframe.data, label="borehole pressure", color='blue')
ax1.set_ylabel("DSS strain")
# Set x lim to the same as DSS_data.taxis
ax1.set_xlim(DSS_data.taxis[0], DSS_data.taxis[-1])
# show legend
ax1.legend(loc='lower right')

ax1.set_ylabel("Pressure (psi)")
ax1.set_xlabel("Time (s)")
ax1.set_yticks([])

plt.tight_layout()
plt.show()