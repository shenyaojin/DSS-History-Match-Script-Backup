# I'm now creating a smooth injection curve so that the simulation can have a better numerical stability.
# Shenyao Jin, shenyaojin@mines.edu
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# Load pressure gauge data from Mariner dataset for injection and time stepping
# This section is adapted from scripts/DSS_history_match/108_misfit_func.py for better convergence.
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge3_data_prod.npz"

# Load and preprocess gauge data
gauge_data_interference = Data1DGauge()
gauge_data_interference.load_npz(pressure_gauge_g1_path)

injection_gauge_pressure = Data1DGauge()
injection_gauge_pressure.load_npz(injection_gauge_pressure_path)
# Select the production data up to the point where the interference data begins.
injection_gauge_pressure.select_time(injection_gauge_pressure.start_time, gauge_data_interference.start_time)
injection_gauge_pressure.remove_abnormal_data(threshold=300, method='mean')

# Create copies for processing
injection_gauge_pressure_copy = injection_gauge_pressure.copy()
gauge_data_interference_copy = gauge_data_interference.copy()
injection_gauge_pressure_copy.adaptive_downsample(300)
gauge_data_interference_copy.adaptive_downsample(600)

# Shift the interference gauge data to align with DSS data (one is wellhead, the other is downhole)
if len(injection_gauge_pressure.data) > 0:
    difference_val = injection_gauge_pressure.data[-1] - gauge_data_interference.data[0]
    gauge_data_interference_copy.data += difference_val

# Merge the two profiles
injection_gauge_pressure_copy.select_time(datetime.datetime(2020, 4, 1, 0, 0, 0), injection_gauge_pressure_copy.get_end_time())
injection_gauge_pressure_copy.right_merge(gauge_data_interference_copy)
injection_gauge_pressure_copy.rename("injection pressure full profile")
# Save the data if not exists
savepath = "data/fiberis_format/post_processing/history_matching_pressure_profile_full.npz"
if not os.path.exists(savepath):
    injection_gauge_pressure_copy.save_npz(savepath)

# Use this processed data for the injection pressure function
gauge_data_for_moose = injection_gauge_pressure_copy.copy()

# Quick fix: remove abnormal high pressures
gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa

# The three Data1DGauge objects are now available as:
# gauge_data_interference (original interference data)
# injection_gauge_pressure (original injection data, pre-selected time)
# gauge_data_for_moose (merged and processed data for MOOSE)

#%% Post processing: Plot the injection profile
fig, ax = plt.subplots()
injection_gauge_pressure.plot(ax=ax, use_timestamp=True)
plt.show()

fig, ax = plt.subplots()
gauge_data_for_moose.plot(ax=ax, use_timestamp=True)
plt.show()