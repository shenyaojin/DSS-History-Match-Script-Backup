# Before we do the history matching on 108,
# Let's first select the area of interest using DSS data, so that I can design the misfit function coorrectly.
# Shenyao Jin, shenyaojin@mines.edu, 11/09/2025

#%% Import libs
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# I/O modules from fiberis
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# Define file paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"

#%% Load DSS data
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)

# Load pressure gauge data
pg_dataframe = Data1DGauge()
pg_dataframe.load_npz(pressure_gauge_g1_path)

#%% Pre=process DSS data
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
# DSSdata.select_depth(14820, 14920) # <- Select depth range of interest.
# DSSdata.select_depth(14980, 15010)

DSSdata.select_depth(14865, 14880) # <- Select depth range of interest.
frac_center = 14871
# Might need to change based on well location

slice_data = DSSdata.get_value_by_depth(frac_center)
slice_dataframe = Data1DGauge()
slice_dataframe.data = slice_data
slice_dataframe.taxis = DSSdata.taxis
slice_dataframe.start_time = DSSdata.start_time

#%% Pre-process pressure gauge data
pg_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())
pg_dataframe.remove_abnormal_data(threshold=300, method='mean')

#%% Co-plot DSS and pressure gauge data, use subplot2grid
fig = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4) # <- DSS plot
ax2 = plt.subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4, sharex=ax1) # <- Pressure gauge plot

# Plot DSS data
im1 = DSSdata.plot(ax=ax1, use_timestamp=False, cmap='bwr', vmin=-1, vmax=1)
ax1.axhline(y=frac_center, color='k', linestyle='--')
ax1.set_title("DSS Data at POW-S")
ax1.set_ylabel("Depth (ft)")
clim = np.array([-1, 1])

# Hide x-axis ticks
ax1.tick_params(labelbottom=False)
# Plot pressure gauge data
im2 = slice_dataframe.plot(ax=ax2, use_timestamp=False)
ax2.set_ylabel("Pressure (psi)")

plt.tight_layout()
plt.show()