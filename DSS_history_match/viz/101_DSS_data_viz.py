#%% DSS data visualization example
# Shenyao, 09/07/2025
import datetime

import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

#%% Load data
datapath = "data/fiberis_format/prod/dss_data/Mariner 14x-36-D - RFS strain change.npz" # we do strain change this time
# datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"

#%% Initialize DSS2D dataframe
DSSdata = DSS2D()
DSSdata.load_npz(datapath)

#%% Crop the depth range, but first print the daxis
DSSdata.print_info()
print(DSSdata.daxis[-10:]) # print the last 10 depth points

#%% Remove the drift first
ind = (DSSdata.daxis > 11400) & (DSSdata.daxis < 11500)
# ind = (DSSdata.daxis > 7500) & (DSSdata.daxis < 15000) # for POW-S well
drift_val_swell = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data = DSSdata.data - drift_val_swell.reshape(1, -1)

#%% Crop depth
DSSdata.select_depth(3000, 16360) # crop to 14000-16360 ft, see my notes. -> now do 3000

#%% Plot the cropped data
fig, ax = plt.subplots(figsize=(10, 6))
cx = np.array([-1, 1]) * 6
im = DSSdata.plot(ax=ax, use_timestamp=False, cmap="bwr")
im.set_clim(cx)
plt.show()

#%% Plot the data along depth.
timeslice, _ = DSSdata.get_value_by_time(100000)
# Plot the timeslice
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(DSSdata.daxis, timeslice, color='blue')
ax.set_xlabel('Depth (ft)')
ax.set_ylabel('Strain Change (με)')
ax.set_title('Strain Change vs Depth at 100000s')
# Set x-axis limits
ax.set_xlim(10000, 13000)
ax.set_ylim(-100, 100)
ax.grid(True)
plt.show()

#%% Load the gauge data
import datetime
gauge_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
gauge = Data1DGauge()
gauge.load_npz(gauge_path)
# crop the gauge data to match the DSS time range
gauge.select_time(DSSdata.start_time, DSSdata.get_end_time('datetime'))

# # Change to relative temperature
# DSSdata_initial,_ = DSSdata.get_value_by_time(0)
# DSSdata.data = DSSdata.data - DSSdata_initial.reshape(-1, 1)

#%% Plot two in subplot2grid
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot2grid((4, 3), (0, 0), rowspan=3, colspan=3)
ax2 = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3, sharex=ax1)

# Plot DSS data
cx = np.array([-1, 1]) * 6
im = DSSdata.plot(ax=ax1, use_timestamp=False, cmap="bwr")
im.set_clim(cx)
ax1.set_title('RFS DSS Data, producer')
# Plot gauge data
gauge.plot(ax=ax2, use_timestamp=False)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Pressure (psi)')
plt.show()