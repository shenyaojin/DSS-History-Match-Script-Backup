#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fiberis.analyzer.Data1D import Data1D_PumpingCurve, Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Geometry3D import DataG3D_md

#%% Set the stage number
stage1 = 8
stage2 = stage1 + 1

#%% Load data
# 1. DAS, 2. PG, 3. pumping curve
datapath = "data/fiberis_format/"

pc_data_folder_stg7 = datapath + f"prod/pumping_data/stage{stage1}/"
pc_stg7_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg7_prop.load_npz(pc_data_folder_stg7 + "Proppant Concentration.npz")
pc_stg7_slurry_rate.load_npz(pc_data_folder_stg7 + "Slurry Rate.npz")
pc_stg7_pressure.load_npz(pc_data_folder_stg7 + "Treating Pressure.npz")

pc_data_folder_stg8 = datapath + f"prod/pumping_data/stage{stage2}/"
pc_stg8_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg8_prop.load_npz(pc_data_folder_stg8 + "Proppant Concentration.npz")
pc_stg8_slurry_rate.load_npz(pc_data_folder_stg8 + "Slurry Rate.npz")
pc_stg8_pressure.load_npz(pc_data_folder_stg8 + "Treating Pressure.npz")

#%% Get the time range of the stages
stg7_bgtime = pc_stg7_slurry_rate.get_start_time()
stg7_edtime = pc_stg7_slurry_rate.get_end_time()

stg8_bgtime = pc_stg8_slurry_rate.get_start_time()
stg8_edtime = pc_stg8_slurry_rate.get_end_time()

#%% Load DAS data
DASdata_stg7_path = datapath + f"s_well/DAS/LFDASdata_stg{stage1}_swell.npz"
DASdata_stg7interval_path = datapath + f"s_well/DAS/LFDASdata_stg{stage1}_interval_swell.npz"
DASdata_stg8_path= datapath + f"s_well/DAS/LFDASdata_stg{stage2}_swell.npz"

DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz(DASdata_stg7_path)

DASdata_tmp = Data2D_XT_DSS.DSS2D()
DASdata_tmp.load_npz(DASdata_stg7interval_path)

DASdata.right_merge(DASdata_tmp)

DASdata_tmp.load_npz(DASdata_stg8_path)
DASdata.right_merge(DASdata_tmp)

#%% Load frac hit location data
frac_hit_datapath_stg7 = datapath + f"s_well/geometry/frac_hit/frac_hit_stage_{stage1}_swell.npz"
frac_hit_datapath_stg8 = datapath + f"s_well/geometry/frac_hit/frac_hit_stage_{stage2}_swell.npz"

frachit_stg7_dataframe = DataG3D_md.G3DMeasuredDepth()
frachit_stg8_dataframe = DataG3D_md.G3DMeasuredDepth()

frachit_stg7_dataframe.load_npz(frac_hit_datapath_stg7)
frachit_stg8_dataframe.load_npz(frac_hit_datapath_stg8)

#%% Load pressure gauge data
pg_md_dataframe = DataG3D_md.G3DMeasuredDepth()
pg_md_dataframe.load_npz("data/fiberis_format/s_well/geometry/gauge_md_swell.npz")

#%% Filter out those gauges that are in my interested areas
depth_range_min = np.min(frachit_stg8_dataframe.data) - 500
depth_range_max = np.max(frachit_stg7_dataframe.data) + 500

ind = np.array(np.where(np.logical_and(
    pg_md_dataframe.data > depth_range_min, pg_md_dataframe.data < depth_range_max
))).flatten()

gauge_dataframe_all = []
for gauge_iter in tqdm(ind):
    datapath = f'data/fiberis_format/s_well/gauges/gauge{gauge_iter+1}_data_swell.npz'
    gauge_dataframe = Data1D_Gauge.Data1DGauge()
    gauge_dataframe.load_npz(datapath)
    gauge_dataframe.crop(stg7_bgtime, stg8_edtime)
    gauge_dataframe_all.append(gauge_dataframe)

DASdata.select_depth(depth_range_min, depth_range_max)

#%% Get DAS channel data and pressure gauge data at the same depth.
selected_depth_array = pg_md_dataframe.data[ind]
DAS_chan = []
for depth_iter in selected_depth_array:
    DAS_chan_tmp = DASdata.get_value_by_depth(depth_iter)
    DAS_chan.append(DAS_chan_tmp)

#%% Get the pressure gradient data; plot it with DAS data.
from fiberis.utils import signal_utils
pg_grad_array = []
for gauge_iter in gauge_dataframe_all:
    #pg_grad_array.append(signal_utils.lpfilter(np.gradient(gauge_iter.data, gauge_iter.taxis),
      #                   gauge_iter.taxis[1] - gauge_iter.taxis[0], 0.6, order=5))
    pg_grad_array.append(np.gradient(gauge_iter.data, gauge_iter.taxis))

#%% Add low pass filter to DAS data

DAS_chan_filtered = []
for DAS_chan_iter in DAS_chan:
    DAS_chan_filtered.append(signal_utils.lpfilter(DAS_chan_iter, DASdata.taxis[1] - DASdata.taxis[0],
                                                   0.01, order=5))

#%% Co plot the DAS data and pressure gradient data.
# First try one gauge.
idx = 1
cx = np.array([-1, 1])
plt.figure(figsize=(5,3))
plt.plot(DASdata.taxis, DAS_chan_filtered[idx]-np.mean(DAS_chan_filtered[idx]), color='b')
plt.ylim(cx * 1e3)
plt.twinx()
plt.scatter(gauge_dataframe_all[idx].taxis, pg_grad_array[idx], color='r', marker='o', alpha=0.2, s=1)
plt.ylim(cx * 1)
plt.xlabel("Time/s")
plt.ylabel("Value")
plt.show()

#%% Interpolate the DAS data to the same length as the pressure gauge data
# for I need to calculate the Pearson correlation coefficient.
DAS_chan_interp = []
flag = 0
for DAS_chan_iter in DAS_chan_filtered:
    # DAS_chan_interp.append(np.interp(DASdata.taxis, gauge_dataframe_all[flag].taxis, pg_grad_array[flag]))
    DAS_chan_interp.append(np.interp(gauge_dataframe_all[flag].taxis, DASdata.taxis, DAS_chan_iter))
    flag += 1

#%% Calculate the Pearson correlation coefficient
from scipy.stats import spearmanr, zscore
corr_array = []
for i in range(len(DAS_chan_interp)):
    print(len(pg_grad_array[i]))
    print(len(DAS_chan_interp[i]))
    corr_array.append(spearmanr(zscore(DAS_chan_interp[i]), zscore(pg_grad_array[i]))[0])
print(corr_array)

#%% Plot all the DAS vs PG grad data
# 2*2 grid
fig, axs = plt.subplots(2, 1, figsize=(8, 4))

# Collect legend handles for all plots
handles = []
labels = []

flag = 0
for i in [4, 3]: # latest stage, earliest stage
    ax1 = axs[flag]

    # Prepare time and signal data for each case
    if i==0:
        das_signal = DAS_chan_filtered[i] - 50
    else:
        das_signal = DAS_chan_filtered[i] - np.mean(DAS_chan_filtered[i])
    das_time = DASdata.taxis
    pg_time = gauge_dataframe_all[i].taxis
    pg_grad = pg_grad_array[i]

    n = len(das_time)
    m = len(pg_time)

    if flag == 0:
        # Slice last half
        das_time = das_time[n//2:] # Get the last half of the DAS data.
        das_signal = das_signal[n//2:]
        pg_time = pg_time[m//2:]
        pg_grad = pg_grad[m//2:]
    else:  # i == 3
        # Slice first half
        das_time = das_time[:n//2]
        das_signal = das_signal[:n//2] - np.mean(das_signal[:n//2])
        pg_time = pg_time[:m//2]
        pg_grad = pg_grad[:m//2]

    # Plot the LF-DAS data
    h1, = ax1.plot(das_time, das_signal, color='b', label='LF-DAS')


    # Plot the ∇P data on the twin axis
    ax2 = ax1.twinx()
    h2 = ax2.scatter(pg_time, pg_grad, color='r', marker='o', alpha=0.4, s=1, label=r'$\nabla P$')
    if flag == 0:
        ax2.set_ylim(cx * 0.5)
        ax1.set_ylim(cx * 500)
    else:
        ax1.set_ylim(cx * 1000)
        ax2.set_ylim(cx * 1)

    if flag == 0:
        handles.extend([h1, h2])
        labels.extend(['LF-DAS', r'$\nabla P$'])

    flag += 1

# Create a common legend
fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.3, 0.1))
plt.tight_layout()
plt.savefig("figs/sponsor_meeting_report_2025/pressure_strain_coupling/scalar.png")
plt.show()

#%% Plot all the DAS vs PG grad data
# 2*2 grid
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))
#
# # Collect legend handles for all plots
# handles = []
# labels = []
#
# flag = 0
# for i in [0, 3]:
#     ax1 = axs[flag]
#     # Plot the LF-DAS data
#     h1, = ax1.plot(DASdata.taxis, DAS_chan_filtered[i]-np.mean(DAS_chan_filtered[i]), color='b', label='LF-DAS')
#     ax1.set_ylim(cx * 500)
#
#     # Plot the ∇P data on the twin axis
#     ax2 = ax1.twinx()
#     h2 = ax2.scatter(gauge_dataframe_all[i].taxis, pg_grad_array[i], color='r', marker='o', alpha=0.2, s=1, label=r'$\nabla P$')
#     ax2.set_ylim(cx * 1)
#     # plt.setp(ax2.get_yticklabels(), visible=False)
#
#     # Only collect handles from first subplot (to avoid duplicates)
#     if i == 0:
#         handles.extend([h1, h2])
#         labels.extend(['LF-DAS', r'$\nabla P$'])
#
#     flag += 1
#
# # Remove x and y ticks
# # plt.setp(axs, xticks=[], yticks=[])
#
# # Create a common legend outside the loop. Change the loc manually.
# fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.06))
#
# plt.tight_layout()
# plt.show()
