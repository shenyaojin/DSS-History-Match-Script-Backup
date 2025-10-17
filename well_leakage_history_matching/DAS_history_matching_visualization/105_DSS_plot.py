#%% Load lib
import numpy as np
from fiberis.analyzer.Data2D import Data2D_XT_DSS
import matplotlib.pyplot as plt
import datetime

#%% Load data.
datapath = "data/legacy/s_well/dss_data.npz"

DSS_data = Data2D_XT_DSS.DSS2D()
dataframe_tmp = np.load(datapath, allow_pickle=True)

#%% Try to load data to Data2D_XT_DSS
DSS_data.data = dataframe_tmp['value']
DSS_data.daxis = dataframe_tmp['daxis']

taxis = dataframe_tmp['taxis']

#%% Convert this datetime array to seconds
taxis_sec = [(t-taxis[0]).total_seconds() for t in taxis]
DSS_data.taxis = np.array(taxis_sec)
DSS_data.start_time = taxis[0]

#%% Do processing
DSS_data.select_depth(11500, 16700)
DSS_data.select_time(0, 400000)
extent = [DSS_data.taxis[0], DSS_data.taxis[-1], DSS_data.daxis[-1], DSS_data.daxis[0]]
cx = np.array([-1, 1])

# #%% Plot
# fig, ax = plt.subplots(figsize=(6, 14))
# im = DSS_data.plot(ax=ax, aspect='auto', useTimeStamp=False, cmap='bwr')
# # Set clim
# im.set_clim(cx * 3)
# plt.show()

#%% Plot all
calibration_value = 97.8
x1, x2, x3 = 13705, 15069, 16276
depths = np.array([x1, x2, x3]) - calibration_value

# Get channels for these depths
channels = []
for depth in depths:
    chan_val = DSS_data.get_value_by_depth(depth)
    channels.append(chan_val)

fig = plt.figure(figsize=(10, 5))

# Waterfall plot (3x3 on the left)
ax0 = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=3)
im = DSS_data.plot(ax=ax0, aspect='auto', useTimeStamp=False, cmap='bwr')
# Plot axhline
for depth in depths:
    ax0.axhline(depth, color='k', linestyle='--', linewidth=2)

im.set_clim(-2, 2)
ax0.set_ylabel("MD (ft)")
ax0.set_xlabel("Time (s)")

# Line plots (1 row each on the right)
for i, (depth, y) in enumerate(zip(depths, channels)):
    ax = plt.subplot2grid((3, 6), (i, 3), colspan=3)
    ax.plot(DSS_data.taxis, y)
    ax.set_ylabel(f"DSS strain")
    if i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)")
    ax.grid(True)

plt.tight_layout()
plt.show()