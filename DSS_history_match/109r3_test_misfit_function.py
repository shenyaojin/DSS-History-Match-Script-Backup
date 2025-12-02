# I finished the misfit calculator for the DSS history matching project.
# Now this script tests the misfit function with field data
# Shenyao Jin, 12/01/2025, shenyaojin@mines.edu

import numpy as np
import matplotlib.pyplot as plt
import os

from fiberis.moose.templates.baseline_model_generator import misfit_calculator, post_processor_info_extractor
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# 1. Observed data post-processing
# Define file paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
scale_factor = 7 # Compensate for imperfect coupling

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

DSSdata.select_depth(14880, 14900) # <- Select depth range of interest.
DSSdata.data = DSSdata.data * scale_factor / 1e6 # Convert to microstrain

# Might need to change based on well location

#%% Pre-process pressure gauge data
pg_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())
pg_dataframe.remove_abnormal_data(threshold=300, method='mean')

#%% Co-plot DSS and pressure gauge data, use subplot2grid
fig = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4) # <- DSS plot
ax2 = plt.subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4, sharex=ax1) # <- Pressure gauge plot

# Plot DSS data
upper_bound = 14886.5
lower_bound = 14891.4
center_frac = 14888.97

im1 = DSSdata.plot(ax=ax1, use_timestamp=False, cmap='bwr', vmin=-1, vmax=1)
line2 = ax1.axhline(y=center_frac, color='k', linestyle='--') # <- center line
line1 = ax1.axhline(y=upper_bound, color='k', linestyle='-') # <- upper bound
line3 = ax1.axhline(y=lower_bound, color='k', linestyle='-') # <- lower bound

# anotate the lines
# Line 1, add annotation
ax1.text(0.15, upper_bound + 0.9, 'Upper Bound', color='k', fontsize=8, rotation=0, va='bottom', ha='left')
# Line 2, add annotation
ax1.text(0.15, center_frac + 0.9, 'Center Line', color='k', fontsize=8, rotation=0, va='bottom', ha='left')
# Line 3, add annotation
ax1.text(0.15, lower_bound + 0.9, 'Lower Bound', color='k', fontsize=8, rotation=0, va='bottom', ha='left')

ax1.set_title("DSS Data at POW-S")
ax1.set_ylabel("Depth (ft)")
clim = np.array([-1, 1])

# Hide x-axis ticks
ax1.tick_params(labelbottom=False)
# Plot pressure gauge data
im2 = pg_dataframe.plot(ax=ax2, use_timestamp=False)
ax2.set_ylabel("Pressure (psi)")

plt.tight_layout()
plt.show()

# output all the channel depths between the upper and lower bound
ind = (DSSdata.daxis >= upper_bound) & (DSSdata.daxis <= lower_bound)
selected_depths = DSSdata.daxis[ind]
print("Selected depths between upper and lower bound:")
for depth in selected_depths:
    print(f"{depth:.2f} ft")

# We need to ignore
# 14888.30 ft
# 14888.64 ft

# Get the ind of the center depth
ind_obs = np.argmin(np.abs(DSSdata.daxis - center_frac))
# Also, add scale factor for the DSS data, for fiber is not perfectly coupled

#%% 2. Simulation data loading
sim_data_outptu_dir = "output/1124_misfit_func"
pressure_dataframe, strain_dataframe = post_processor_info_extractor(output_dir=sim_data_outptu_dir)
pg_frame = Data1DGauge()
pg_frame.load_npz("data/fiberis_format/post_processing/timestepper_profile.npz")
start_time = pg_frame.start_time
pressure_dataframe.start_time = start_time
strain_dataframe.start_time = start_time

ind_sim = len(strain_dataframe.daxis) // 2  # <- select center depth for simulated data

#%% 3. Misfit calculation
weight_matrix = np.array([1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1])

misfit_val = misfit_calculator(weight_matrix=weight_matrix,
                               sim_fracture_center_ind=ind_sim,
                               observed_data_fracture_center_ind=ind_obs,
                               simulated_data=strain_dataframe,
                               observed_data=DSSdata,
                               save_path="figs/1201_misfit_results")

print(misfit_val)