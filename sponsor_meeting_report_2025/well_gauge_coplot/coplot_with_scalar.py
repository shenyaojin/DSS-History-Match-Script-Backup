#%% Import fiberis
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fiberis.analyzer.Data1D import Data1D_PumpingCurve, Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Geometry3D import DataG3D_md
import datetime

# This script is designed for the sponsor_meeting_report_2025
# "Low-Frequency DAS for Cement Quality Monitoring in Horizontal Wells"

#%% Parameters
# Multi-stage co plot with pressure gauge response
stage1 = 8
stage2 = stage1 + 1

coeff = 0.14 # The amplitude of the PG response

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
depth_range_max = np.max(frachit_stg7_dataframe.data) + 700

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


cx = np.array([-1, 1])
scalar_taxis = np.repeat(stg7_bgtime + datetime.timedelta(minutes=30), 2)
scalar_value = 500
scalar_tmp_value = np.array([+ pg_md_dataframe.data[ind][0] + 140, (scalar_value) * - coeff + pg_md_dataframe.data[ind][0] + 140])

DASdata.select_depth(depth_range_min, depth_range_max)

#%% Concatenate the data (pumping)
pc_stg7_prop.right_merge(pc_stg8_prop)
pc_stg7_prop.rename("Proppant Concentration")

pc_stg7_slurry_rate.right_merge(pc_stg8_slurry_rate)
pc_stg7_slurry_rate.rename("Slurry Rate")

#%% Plot the data
plt.figure(figsize = (7, 5))

ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=4)
for i in range(len(pg_md_dataframe.data[ind])):
    ax1.axhline(y=pg_md_dataframe.data[ind][i], color='black', linestyle='--')
    datetime_taxis = gauge_dataframe_all[i].calculate_time()
    ax1.plot(datetime_taxis, (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * - coeff
             + pg_md_dataframe.data[ind][i], color='cyan', linewidth=2)
ax1.plot(scalar_taxis, scalar_tmp_value, color='cyan', linewidth=5)
ax1.text(scalar_taxis[0] + datetime.timedelta(minutes=8), scalar_tmp_value[0] - scalar_value/20,
         f"{scalar_value} psi", fontsize=12, color='black')

img1 = DASdata.plot(ax=ax1, useTimeStamp=True, cmap='bwr', aspect='auto')
img1.set_clim(cx * 3e2)

ax2 = plt.subplot2grid((6, 4), (4, 0), colspan=4, rowspan=2, sharex = ax1)

color = 'blue'
pc_stg7_prop.rename("Prop. Conc.")
pc_stg7_prop.plot(ax=ax2, useTimeStamp=True, title=None, color=color)
ax2.set_ylabel(fr"Prop. Conc./lb$\cdot$gal$^{-1}$", color=color)
ax2.tick_params(axis='y', labelcolor=color)
# set xlim
ax2.set_xlim(stg7_bgtime, stg8_edtime)

ax21 = ax2.twinx()
color = 'green'
pc_stg7_slurry_rate.plot(ax=ax21, useTimeStamp=True, title=None, color=color)
ax21.tick_params(axis='y', labelcolor=color)

ax22 = ax2.twinx()
color = 'red'
pc_stg7_pressure.rename("Treating Pressure")
pc_stg7_pressure.plot(ax=ax22, useTimeStamp=True, title=None, color=color)
ax22.tick_params(axis='y', labelcolor=color)

ax23 = ax2.twinx()
pc_stg8_pressure.plot(ax=ax23, useTimeStamp=True, title=None, color=color)
# Remove yaxis
ax23.yaxis.set_visible(False)

handles2, labels2 = ax2.get_legend_handles_labels()
handles21, labels21 = ax21.get_legend_handles_labels()
handles22, labels22 = ax22.get_legend_handles_labels()

# Combine the handles and labels
combined_handles = handles2 + handles21 + handles22
combined_labels = labels2 + labels21 + labels22

# Create a combined legend on ax2 (or any preferred axis)
ax2.legend(combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.52, 1.05))

# plt.suptitle(f"LF-DAS data with gauge in S well/stage {stage1} and {stage2}")
plt.tight_layout()
plt.savefig(f"figs/manuscript/well_gauge_coplot/Swell_scalar_stage{stage1}.png")
plt.show()