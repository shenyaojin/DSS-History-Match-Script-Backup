import os
import datetime

import numpy as np
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Data1D import Data1D_Gauge

#%% Load the data
data = "output/0324_forward_simulator/"

start_time = datetime.datetime(2020,4,1)
end_time = datetime.datetime(2020,7,1)

dataframe_full = []
for file_iter in os.listdir(data):
    datapath = data + file_iter
    # Load the data
    dataframe_tmp = Data2D_XT_DSS.DSS2D()
    dataframe_tmp.load_npz(datapath)
    dataframe_tmp.select_time(start_time, end_time)
    dataframe_full.append(dataframe_tmp)

# Load the pressure gauge MD and pressure gauge data.
from DSS_analyzer_Mariner import Data3D_geometry
gauge_md_datapath = "data/legacy/s_well/geometry/gauge_md_swell.npz"
gauge_md_dataframe = Data3D_geometry.Data3D_geometry(gauge_md_datapath)
# Extract the data
gauge_md = gauge_md_dataframe.data

# Load the gauge data.
datapath = "data/fiberis_format/"
pg_data_folder = datapath + "s_well/gauges/"
gauge_data_all = []

for iter_filename in os.listdir(pg_data_folder):
    datapath_iter = os.path.join(pg_data_folder, iter_filename)
    gauge_dataframe_iter = Data1D_Gauge.Data1DGauge()
    gauge_dataframe_iter.load_npz(datapath_iter)
    gauge_dataframe_iter.crop(start_time, end_time)
    gauge_data_all.append(gauge_dataframe_iter)

#%% Calculate the pressure drop down at pressure gauge locations .
# Pressure drop down in pressure gauges (field data)
pressure_drop_down = []
pressure_drop_down_label = []

# Calculate the actual drop down
pressure_drop_down_tmp = []
for gauge_iter in gauge_data_all:
    pressure_final = gauge_iter.data[-1]
    pressure_initial = gauge_iter.data[0]

    pressure_drop_down_tmp.append(pressure_initial - pressure_final)

pressure_drop_down.append(pressure_drop_down_tmp)
pressure_drop_down_label.append("Field data")

from fiberis.utils import mesh_utils

gauge_idx = []
for gauge_iter in gauge_md:
    idx_tmp, _ = mesh_utils.locate(dataframe_full[0].daxis, gauge_iter)
    gauge_idx.append(idx_tmp)

#%% Calculate the dropdown in the simulation
flag = 0
for data_iter in dataframe_full[:-1]:
    data_chan = data_iter.data.T[:, gauge_idx]
    pressure_drop_down_tmp = []
    pressure_drop_down_tmp = data_chan[1] - data_chan[-1]
    pressure_drop_down.append(pressure_drop_down_tmp)
    filename = os.listdir(data)[flag]
    pressure_drop_down_label.append(f"{filename}")
    flag += 1

#%% Plot the pressure drop down
import matplotlib.pyplot as plt
plt.figure()
axis = np.arange(1, len(pressure_drop_down[0])+1, 1)
for idx in [0, 3]:# range(len(pressure_drop_down)):
    plt.plot(axis, pressure_drop_down[idx], label=pressure_drop_down_label[idx], marker='o', linestyle='-')
    plt.legend()
    plt.xlabel('Gauge Number')
    plt.ylabel('Pressure Drop Down (psi)')
plt.show()