#%% Load libraries
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

#%% Load DTS data
filepath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - temperature.npz"
dts = DSS2D()
dts.load_npz(filepath)

#%% Print data info
dts.data[:5, :5]

#%% Plot data
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
im = dts.plot(ax=ax, aspect='auto', cmap='jet')
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('DTS Data from Mariner 14x-36-POW-S')
im.set_clim(20, 160)
plt.show()

#%% Pre processing: differential temperature with baseline
base_line_temp_array = np.mean(dts.data[:, :10], axis=1)

dts_preprocessed = dts.copy()

dts_preprocessed.data = dts.data - base_line_temp_array[:, np.newaxis]

#%% Plot data
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
im = dts_preprocessed.plot(ax=ax, aspect='auto', cmap='bwr', use_timestamp=True)
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('DTS Data from Mariner 14x-36-POW-S')
im.set_clim(-1, 1)
# Do not show x tick labels
ax.set_xticklabels([])
plt.show()

#%% Plot the temperature with the borehole pressure data.
# Use subplot2grid to create a grid of subplots
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot2grid((4, 3), (0, 0), rowspan=3, colspan=3)
ax2 = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3, sharex=ax1)

im = dts_preprocessed.plot(ax=ax1, aspect='auto', cmap='bwr', use_timestamp=True)
ax1.set_title('DTS Data from Mariner 14x-36-POW-S')
im.set_clim(-1, 1)
ax1.set_xticklabels([])

# Load borehole pressure data
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
gauge_dataframe = Data1DGauge()
gauge_filepath = "data/fiberis_format/prod/gauges/pressure_g1.npz"
gauge_dataframe.load_npz(gauge_filepath)

im2 = gauge_dataframe.plot(ax=ax2, use_timestamp=True)
ax2.set_ylabel('Pressure (psi)')
ax2.set_xlabel('Time')
ax2.set_title('Borehole Pressure Data from Gauge G1')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.tight_layout()
plt.show()

#%% Pre processing 2, use before shut-in data as baseline
# Find the shut-in time
gauge_dataframe.data = np.diff(gauge_dataframe.data)
# Insert a zero at the beginning to keep the same length
gauge_dataframe.data = np.insert(gauge_dataframe.data, 0, 0)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
im = gauge_dataframe.plot(ax=ax, use_timestamp=True)
ax.set_title('Borehole Pressure Data from Gauge G1')
plt.show()

#%% Find the index of the maximum pressure drop
import datetime
shut_in_index = np.argmax(gauge_dataframe.data) + 1
shut_in_timestamp = gauge_dataframe.start_time + datetime.timedelta(seconds=gauge_dataframe.taxis[shut_in_index])

print(f'Shut-in time: {shut_in_timestamp}')

#%% Crop the DTS data to only include data before shut-in
shut_in_dts = dts.copy()
shut_in_dts.select_time(dts.start_time, shut_in_timestamp)

#%% Calculate the baseline temperature before shut-in
base_line_temp_array_2 = np.mean(shut_in_dts.data, axis=1)
dts_preprocessed_2 = dts.copy()
dts_preprocessed_2.data = dts.data - base_line_temp_array_2[:, np.newaxis]

#%% Plot data
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
im = dts_preprocessed_2.plot(ax=ax, aspect='auto', cmap='bwr', use_timestamp=True)
fig.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('DTS Data from Mariner 14x-36-POW-S')
im.set_clim(-1, 1)
# Do not show x tick labels
ax.set_xticklabels([])
plt.show()

#%% We only care about the temperature in vertical section of the well.

