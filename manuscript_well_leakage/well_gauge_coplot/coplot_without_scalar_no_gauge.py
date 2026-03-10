#%% Import fiberis
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D import Data1D_PumpingCurve
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
# 1. DAS, 2. pumping curve
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

#%% Filter out those gauges that are in my interested areas
depth_range_min = np.min(frachit_stg8_dataframe.data) - 500
depth_range_max = np.max(frachit_stg7_dataframe.data) + 700

cx = np.array([-1, 1])

DASdata.select_depth(depth_range_min, depth_range_max)

# Filter the DASdata
DASdata.apply_lowpass_filter(0.01)

#%% Plot the data
plt.figure(figsize = (7, 5))

ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=4)

img1 = DASdata.plot(ax=ax1, use_timestamp=True, cmap='bwr', aspect='auto')
img1.set_clim(cx * 3e2)

# Add frac hit location, using scatter plot
frac_hit_stg7_taxis = np.repeat(stg7_bgtime + datetime.timedelta(minutes=30), len(frachit_stg7_dataframe.data))
ax1.scatter(frac_hit_stg7_taxis, frachit_stg7_dataframe.data, color='lightgray', marker='x', s=40, label='Frac Hit')

frac_hit_stg8_taxis = np.repeat(stg8_bgtime + datetime.timedelta(minutes=30), len(frachit_stg8_dataframe.data))
ax1.scatter(frac_hit_stg8_taxis, frachit_stg8_dataframe.data, color='lightgray', marker='x', s=40)

# Removed gauge plots and ax1 legend as requested

ax2 = plt.subplot2grid((6, 4), (4, 0), colspan=4, rowspan=2, sharex = ax1)

color = 'blue'
pc_stg7_prop.rename("Prop. Conc.")
pc_stg7_prop.plot(ax=ax2, use_timestamp=True, title=None, color=color)
ax2.set_ylabel(fr"Prop. Conc./lb$\cdot$gal$^{-1}$", color=color)
ax2.tick_params(axis='y', labelcolor=color)
# set xlim
ax2.set_xlim(stg7_bgtime, stg8_edtime)

ax21 = ax2.twinx()
color = 'green'
pc_stg7_slurry_rate.rename("Slurry Rate")
pc_stg7_slurry_rate.plot(ax=ax21, use_timestamp=True, title=None, color=color)
ax21.tick_params(axis='y', labelcolor=color)

ax22 = ax2.twinx()
color = 'red'
pc_stg7_pressure.rename("Treating Pressure")
pc_stg7_pressure.plot(ax=ax22, use_timestamp=True, title=None, color=color)
ax22.tick_params(axis='y', labelcolor=color)

ax23 = ax2.twinx()
pc_stg8_pressure.plot(ax=ax23, use_timestamp=True, title=None, color=color)
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

# Add plots after the legend. Stage 8 data
pc_stg8_prop.rename("Prop. Conc. Stage 8")
color = 'blue'
pc_stg8_prop.plot(ax=ax2, use_timestamp=True, title=None, color=color)
color = 'green'
pc_stg8_slurry_rate.plot(ax=ax21, use_timestamp=True, title=None, color=color)
pc_stg8_pressure.rename("Treating Pressure Stage 8")
color = 'red'
pc_stg8_pressure.plot(ax=ax22, use_timestamp=True, title=None, color=color)

# Remove ticks and labels as requested
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel("")

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylabel("")

ax21.set_yticks([])
ax21.set_ylabel("")

ax22.set_yticks([])
ax22.set_ylabel("")

ax23.set_yticks([])

# plt.suptitle(f"LF-DAS data with gauge in S well/stage {stage1} and {stage2}")
plt.tight_layout()
plt.savefig("figs/manuscript/DAS_gauge_coplot/Swell_no_gauge.png")
plt.show()
