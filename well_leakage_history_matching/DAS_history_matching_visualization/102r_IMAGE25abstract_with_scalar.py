# Modified from 101p_fiberis_matching_postprocessing.py
# IMAGE'25 abstract figure; I need to show the scale so that I can plot the scale
# on 102's figure.
# Shenyao Jin, shenyaojin@mines.edu
# Script created on 02/12/2025

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fiberis.analyzer.Data1D import Data1D_PumpingCurve, Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.utils import mesh_utils
import datetime

#%% Load the data
datapath_phase1 = "output/0211_simulation_MULTIstage/phase1.npz"
phase1_pf_dataframe = Data2D_XT_DSS.DSS2D()
phase1_pf_dataframe.load_npz(datapath_phase1)

datapath_phase2 = "output/0211_simulation_MULTIstage/phase2.npz"
phase2_pf_dataframe = Data2D_XT_DSS.DSS2D()
phase2_pf_dataframe.load_npz(datapath_phase2)

#%% rotate the data
phase1_pf_dataframe.data = phase1_pf_dataframe.data.T
phase2_pf_dataframe.data = phase2_pf_dataframe.data.T

#%%
phase2_pf_dataframe.select_time(30, phase2_pf_dataframe.get_end_time())
print(phase2_pf_dataframe.start_time)
#%% Concatenate the data
phase1_pf_dataframe.right_merge(phase2_pf_dataframe)

#%% Get phase3
datapath_phase3 = "output/0211_simulation_MULTIstage/phase3_test.npz"
phase3_pf_dataframe = Data2D_XT_DSS.DSS2D()
phase3_pf_dataframe.load_npz(datapath_phase3)
phase3_pf_dataframe.data = phase3_pf_dataframe.data.T

#%% select te data
phase3_pf_dataframe.select_time(30, phase3_pf_dataframe.get_end_time())

#%% Concatenate the data
phase1_pf_dataframe.right_merge(phase3_pf_dataframe)


#%% Load the gauge data
# Legacy dataformat

frac_hit_datapath = f"data/legacy/s_well/geometry/frac_hit/"
frac_hit_dataframe_stage7 = np.load(frac_hit_datapath + f"frac_hit_stage_7_swell.npz")
frac_hit_dataframe_stage8 = np.load(frac_hit_datapath + f"frac_hit_stage_8_swell.npz")

# extract data
frac_hit_stg7 = frac_hit_dataframe_stage7['data']
frac_hit_stg8 = frac_hit_dataframe_stage8['data']

gauge_md_datapath = f"data/legacy/s_well/geometry/gauge_md_swell.npz"
gauge_md_datapath = f"data/legacy/s_well/geometry/gauge_md_swell.npz"
gauge_md = np.load(gauge_md_datapath)
gauge_md = gauge_md['data']

ind = np.array(np.where(np.logical_and(gauge_md <=  np.max(frac_hit_stg7) + 500, gauge_md >= np.min(frac_hit_stg8) - 500))).flatten()
gauge_md = gauge_md[ind]
pf_dataframe = phase1_pf_dataframe.copy()
pf_dataframe.select_depth(np.min(frac_hit_stg8) - 500, np.max(frac_hit_stg7) + 500)
phase1_pf_dataframe = pf_dataframe.copy()

datapath = "data/fiberis_format/"
#%% Load pumping data (stage 7 and stage 8)
pc_data_folder_stg7 = datapath + "prod/pumping_data/stage7/"
pc_stg7_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg7_prop.load_npz(pc_data_folder_stg7 + "Proppant Concentration.npz")
pc_stg7_slurry_rate.load_npz(pc_data_folder_stg7 + "Slurry Rate.npz")
pc_stg7_pressure.load_npz(pc_data_folder_stg7 + "Treating Pressure.npz")

pc_data_folder_stg8 = datapath + "prod/pumping_data/stage8/"
pc_stg8_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg8_prop.load_npz(pc_data_folder_stg8 + "Proppant Concentration.npz")
pc_stg8_slurry_rate.load_npz(pc_data_folder_stg8 + "Slurry Rate.npz")
pc_stg8_pressure.load_npz(pc_data_folder_stg8 + "Treating Pressure.npz")

#%% Get the time range of stages
stg7_bgtime = pc_stg7_slurry_rate.get_start_time()
stg7_edtime = pc_stg7_slurry_rate.get_end_time()

stg8_bgtime = pc_stg8_slurry_rate.get_start_time()
stg8_edtime = pc_stg8_slurry_rate.get_end_time()

#%% Load DAS data
DASdata_stg7_path = datapath + "s_well/DAS/LFDASdata_stg7_swell.npz"
DASdata_stg7interval_path = datapath + "s_well/DAS/LFDASdata_stg7_interval_swell.npz"
DASdata_stg8_path= datapath + "s_well/DAS/LFDASdata_stg8_swell.npz"

DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz(DASdata_stg7_path)

DASdata_tmp = Data2D_XT_DSS.DSS2D()
DASdata_tmp.load_npz(DASdata_stg7interval_path)

DASdata.right_merge(DASdata_tmp)

DASdata_tmp.load_npz(DASdata_stg8_path)
DASdata.right_merge(DASdata_tmp)

DASdata.select_depth(np.min(frac_hit_stg8) - 500, np.max(frac_hit_stg7) + 500)

#%% Load the gauge data based on the MD
gauge_dataframe_all = []
for iter in tqdm(ind):
    datapath = f'data/fiberis_format/s_well/gauges/gauge{iter+1}_data_swell.npz'
    gauge_dataframe = Data1D_Gauge.Data1DGauge()
    gauge_dataframe.load_npz(datapath)
    gauge_dataframe.crop(stg7_bgtime, stg8_edtime)
    gauge_dataframe_all.append(gauge_dataframe)

#%% Process the data

pf_dataframe.data = pf_dataframe.data * 6894.76 / 30e9
tmp_strain = np.zeros_like(pf_dataframe.data)
for i in range(pf_dataframe.data.shape[0]):
    tmp_strain[i, :] = np.gradient(pf_dataframe.data[i, :], axis=0) / np.gradient(pf_dataframe.taxis)

scalar_taxis = np.repeat(gauge_dataframe_all[0].start_time + datetime.timedelta(minutes=30), 2)
scalar_value = 500
coeff = 0.2 # Amplitude of the pressure gauge data
scalar_tmp_value = np.array([+ gauge_md[0] + 140, (scalar_value) * - coeff + gauge_md[0] + 140])

#%% Plot the figure
selected_gauge_num = 3
plt.figure(figsize = (14, 8))

cx = np.array([-1, 1])

ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=3, rowspan=3)
for i in range(len(gauge_md)):
    ax1.axhline(y=gauge_md[i], color='black', linestyle='--')
    datetime_taxis = gauge_dataframe_all[i].calculate_time()
    if i==selected_gauge_num:
        ax1.plot(datetime_taxis, (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * -
        coeff + gauge_md[i], color='green', linewidth=2)
    else:
        ax1.plot(datetime_taxis, (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * -
        coeff + gauge_md[i], color='cyan', linewidth=2)
ax1.plot(scalar_taxis, scalar_tmp_value, color='black', linewidth=5)
ax1.text(scalar_taxis[0] + datetime.timedelta(minutes=8),
         scalar_tmp_value[0] - scalar_value/7, f"{scalar_value} psi", fontsize=12, color='black')

img1 = DASdata.plot(ax=ax1, useTimeStamp=True, cmap='bwr', aspect='auto')

img1.set_clim(cx * 3e2)

ax2 = plt.subplot2grid((4, 6), (3, 0), colspan=3, rowspan=1, sharex = ax1)

color = 'black'
pc_stg8_pressure.plot(ax=ax2, useTimeStamp=True, title=None, color=color)
# ax2.set_ylabel(fr"Prop. Conc./lb$\cdot$gal$^{-1}$", color=color)
ax2.set_ylabel(fr"Pressure/psi", color=color)
# set xlim
ax2.set_xlim(stg7_bgtime, stg8_edtime)

ax22 = ax2.twinx()
color = 'black'
pc_stg7_pressure.plot(ax=ax22, useTimeStamp=True, title=None, color=color)
ax22.set_ylabel("Treating Pressure/psi", color=color)
ax22.tick_params(axis='y', labelcolor=color)
ax22.yaxis.set_visible(False)

# Plot the synthetic data in the right panel.
ax3 = plt.subplot2grid((4, 6), (0, 3), colspan=3, rowspan=3, sharey = ax1)
# Plot the synthetic data from the history matching result.
# Load synthetic gauge data
real_time_taxis = pf_dataframe.calculate_time()
for i in range(len(gauge_md)):
    ax3.axhline(y=gauge_md[i], color='black', linestyle='--')
    gauge_idx, _ =  mesh_utils.locate(phase1_pf_dataframe.daxis, gauge_md[i])
    ax3.plot(real_time_taxis, (phase1_pf_dataframe.data[gauge_idx, :]
                                      - phase1_pf_dataframe.data[gauge_idx, 0]) * -0.15
             + gauge_md[i], color='cyan', linewidth=2)


img3 = ax3.pcolormesh(real_time_taxis, pf_dataframe.daxis, tmp_strain, cmap='bwr')

img3.set_clim(cx * 1e-7)

# Create ax4, which would be the comparison between the LF-DAS data and gauge data.
ax4 = plt.subplot2grid((4, 6), (3, 3), colspan=3, rowspan=1) # No need to share the xaxis though the time is the same.
cx_ax4 = np.array([-1, 3])
# Plot the gauge data
# rename the label
gauge_for_ax4 = gauge_dataframe_all[selected_gauge_num].copy()
# Get the gradient
gauge_for_ax4.data = np.gradient(gauge_for_ax4.data, axis=0) / np.gradient(gauge_for_ax4.taxis)
gauge_for_ax4.plot(ax=ax4, useTimeStamp=True, title=None, color='black')
ax4.set_ylim(cx_ax4 * 0.5)

# Extract the LF-DAS data
DASdata_idx = mesh_utils.locate(DASdata.daxis, gauge_md[selected_gauge_num])[0]
# Plot the LF-DAS channel
ax41 = ax4.twinx()
# I forgot to implement the function in DSS2D.
extracted_LFDAS_chan = DASdata.data[DASdata_idx, :]
real_time_taxis_DAS = DASdata.calculate_time()

# Apply low pass filter to extracted_LFDAS_chan
from DSS_analyzer_Mariner.gjsignal import lpfilter
extracted_LFDAS_chan_filtered = lpfilter(extracted_LFDAS_chan, 1, 0.05)
ax41.plot(real_time_taxis_DAS, extracted_LFDAS_chan_filtered, color='red', label='LF-DAS data', alpha=0.5)
ax41.xaxis.set_visible(False)
ax41.yaxis.set_visible(False)
ax41.set_ylim(cx_ax4 * 3e2)

ax4.set_xlim(stg7_bgtime, stg8_edtime)


plt.suptitle(f"LF-DAS data coplot with history matching result. \nDiffusivity drop to 1e-6 times than before")
# plt.tight_layout()
plt.savefig("figs/02132025_IMAGE25figure/coplotV1_scalar.png")
plt.show()