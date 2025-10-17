#%% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from fiberis.simulator.core import pds
from fiberis.utils import mesh_utils
from fiberis.analyzer.Data1D import Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
import datetime

#%% Set global parameters
range_min = 15000 # Minimum range in ft, for stage 1
range_max = 16750 # Maximum range in ft

start_time = datetime.datetime(2020, 3, 16, 11, 24, 0)
end_time = datetime.datetime(2020, 3, 16, 11, 45, 0)
E = 30e9 # Young's modulus in Pa

#%% Load the DAS data
DASdata_path = "data/fiberis_format/s_well/DAS/LFDASdata_stg1_swell.npz"
DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz(DASdata_path)

DASdata.select_depth(range_min, range_max)
DASdata.select_time(start_time, end_time)

#%% Viz the DAS data
fig, ax = plt.subplots(figsize=(4, 6))
cx = np.array([-1, 1])
img1 = DASdata.plot(ax=ax, useTimeStamp=False, cmap='bwr', aspect='auto')
img1.set_clim(cx * 1000)
# Set title
ax.set_title("LF-DAS data for history matching")
plt.show()

#%% Process the DAS data, convert to strain.
filtered_DAS_data = DASdata.copy()
filtered_DAS_data.data = np.where(DASdata.data > 0, DASdata.data, 0)

DASdata_strain = filtered_DAS_data.copy()
DASdata_strain.data = DASdata_strain.data / 10430.4 * 2.799e-8
DASdata_strain.data = np.cumsum(DASdata_strain.data, axis=1)

#%% Get the pressure at LF-DAS location
from fiberis.analyzer.Geometry3D import DataG3D_md
gauge_md_datapath = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
gauge_md_data = DataG3D_md.G3DMeasuredDepth()
gauge_md_data.load_npz(gauge_md_datapath)
ind = (gauge_md_data.data >= range_min) & (gauge_md_data.data <= range_max)
ind_copy = np.array(ind, copy=True)
print(ind)
# Extract the DAS data channel value
DASchan_all = []
DASpressure_all = []
for val_iter in gauge_md_data.data[ind]:
    chan_val = DASdata_strain.get_value_by_depth(val_iter)
    DASchan_all.append(chan_val)
    DASpressure_all.append(chan_val)
DASchan_all = np.array(DASchan_all)
np.shape(DASchan_all)

#%% Load the pressure gauge data.
gauge_num = np.arange(0, len(gauge_md_data.data))[ind] + 1
gauge_data_all = []
pressure_val_all = []
for gauge_iter in gauge_num:
    gauge_datapath = f"data/fiberis_format/s_well/gauges/gauge{gauge_iter}_data_swell.npz"
    gauge_data_tmp = Data1D_Gauge.Data1DGauge()
    gauge_data_tmp.load_npz(gauge_datapath)

    # Select the time range
    gauge_data_tmp.crop(start_time, end_time)
    gauge_data_tmp.data = np.interp(DASdata.taxis, gauge_data_tmp.taxis, gauge_data_tmp.data)
    gauge_data_tmp.taxis = DASdata.taxis
    gauge_data_tmp.data = gauge_data_tmp.data
    gauge_data_all.append(gauge_data_tmp)
    pressure_val_all.append(gauge_data_tmp.data)

#%% Do linear regression on DAS data
from scipy.stats import linregress
gauge_pressure_flat = np.array(pressure_val_all).flatten()
DASdata_strain_flat = DASchan_all.flatten()

slope, intercept, _, _, _ = linregress(DASdata_strain_flat, gauge_pressure_flat)

#%% Convert the DAS data to pressure
DASdata_pressure = DASdata_strain.copy()
DASdata_pressure.data = DASdata_pressure.data * slope + intercept

#%% QC the DAS data
fig, ax = plt.subplots(figsize=(4, 6))
cx = np.array([-1, 1])
img1 = DASdata_pressure.plot(ax=ax, useTimeStamp=False, cmap='bwr', aspect='auto')
img1.set_clim([8000, 9000])
# Set title
ax.set_title("LF-DAS data for history matching")
plt.show()

#%% History matching

sim1D = pds.PDS1D_SingleSource()

# get mesh.
mesh = DASdata.daxis - np.min(DASdata.daxis)
mesh_negative = DASdata.daxis - np.max(DASdata.daxis)
mesh = np.concatenate([mesh_negative[:-1], mesh])
sim1D.set_mesh(mesh)

sim1D.set_t0(0)
sim1D.set_bcs('Neumann', 'Neumann')
sim1D.set_initial(np.zeros_like(mesh))

idx, _ = mesh_utils.locate(mesh, 0)
idx_tmp, _ = mesh_utils.locate(mesh, -50)
sim1D.set_sourceidx(idx_tmp)

synthetic_source = Data1D_Gauge.Data1DGauge()
synthetic_source = gauge_data_all[0].copy()
synthetic_source.data -= synthetic_source.data[0]
# synthetic_source.data = DASdata_pressure.data[-1]
# jump_t = 220
# synthetic_source.taxis= DASdata.taxis
# ind = synthetic_source.taxis < jump_t
# synthetic_source.data[ind] = 0
# synthetic_source.data[~ind] = synthetic_source.data[-1]
#
# synthetic_source.start_time = DASdata.start_time
# synthetic_source.data -= synthetic_source.data[0]

sim1D.set_source(synthetic_source)
# set the diffusivity array
diffusivity_array = np.zeros_like(mesh)
# use linspace to get the diffusivity. The center is largest, then decrease to the edge.
d_max = 840
d_min = d_max / 6
diffusivity_array[len(mesh)//2+1:] = np.linspace(d_max, d_min, len(mesh)//2)
diffusivity_array[:len(mesh)//2] = np.linspace(d_min, d_max, len(mesh)//2)
diffusivity_array[len(mesh)//2] = d_max
sim1D.set_diffusivity(diffusivity_array)

# sim1D.set_diffusivity(430)
sim1D.self_check()

#%% Run the simulation
sim1D.solve(optimizer=False, dt=1, t_total=synthetic_source.taxis[-1], print_progress=True)

#%% Crop the synthetic data and plot.
synthetic_data = sim1D.snapshot[1:, :idx].T + 8454
synthetic_DASdata = Data2D_XT_DSS.DSS2D()
synthetic_DASdata.data =synthetic_data
synthetic_DASdata.taxis = DASdata.taxis
synthetic_DASdata.daxis = DASdata.daxis
synthetic_DASdata.start_time = DASdata.start_time

#%% Plot the synthetic data
# fig, ax = plt.subplots(figsize=(4, 6))
# img1 = synthetic_DASdata.plot(ax=ax, useTimeStamp=False, cmap='bwr', aspect='auto')
# img1.set_clim(8200, 8800)
# # Set title
# ax.set_title("LF-DAS data, synthetic")
# plt.show()
#
# fig, ax = plt.subplots(figsize=(4, 6))
# img1 = DASdata_pressure.plot(ax=ax, useTimeStamp=False, cmap='bwr', aspect='auto')
# img1.set_clim(8200, 8800)
# # Set title
# ax.set_title("LF-DAS data, synthetic")
# plt.show()

#%% Get the index of those pressure gauge locations.
idx_all = []
for idx_iter in gauge_md_data.data[ind_copy]:
    idx_tmp, _ = mesh_utils.locate(synthetic_DASdata.daxis, idx_iter)
    idx_all.append(idx_tmp)

# Post-processing on data.
DASdata_pressure.daxis -= np.min(DASdata_pressure.daxis)   # Policy from ExxonMobil
synthetic_DASdata.daxis -= np.min(synthetic_DASdata.daxis)

#%% Plot the DAS data and synthetic data
plt.figure(figsize = (6, 8))
ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=4)
extent_field = (
    DASdata_pressure.taxis[0], DASdata_pressure.taxis[-1],
    DASdata_pressure.daxis[-1], DASdata_pressure.daxis[0]
)
img1 = ax1.imshow(DASdata_pressure.data, aspect='auto', cmap='bwr', extent=extent_field)
img1.set_clim(8200, 8800)

ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=4)
img2 = ax2.imshow(synthetic_DASdata.data, aspect='auto', cmap='bwr', extent=extent_field)
img2.set_clim(8200, 8800)


# Plot the gauge data on both images
flag = 0
for idx_iter in idx_all: # temp fix
    # First plot the axhline
    ax1.axhline(y=DASdata_pressure.daxis[idx_iter], color='k', linestyle='--', linewidth=2)
    ax2.axhline(y=synthetic_DASdata.daxis[idx_iter], color='k', linestyle='--', linewidth=2)
    # Then plot the gauge data (or synthetic data)
    ax1.plot(DASdata_pressure.taxis,
             (pressure_val_all[flag] - pressure_val_all[flag][0]) * -0.1 + DASdata_pressure.daxis[idx_iter], c='k')

    # get the synthetic data
    chan_val = synthetic_DASdata.get_value_by_depth(synthetic_DASdata.daxis[idx_iter])
    ax2.plot(synthetic_DASdata.taxis,
             (chan_val - chan_val[0]) * -0.1 + synthetic_DASdata.daxis[idx_iter], c='k')
    flag += 1

# Add label
ax1.set_xlabel("Time/s")
ax2.set_xlabel("Time/s")
ax1.set_ylabel("relative MD/ft")
# Remove the y axis tick for image 2
ax2.set_yticks([])
plt.savefig()
plt.show()