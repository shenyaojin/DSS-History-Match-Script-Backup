# This script is to show the full pressure profile vduring the production and interference test.
# To figure out why the simulated strain data shows abnormal behavior at the beginning of the interference test.
# Modified from "scripts/DSS_history_match/106r_DSS_3xSingleFrac_parameter_scanner.py"
# Shenyao Jin, 10/24/2025

#%% Imports
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

#%% Those data paths:
gauge_path_production = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
interference_test_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"

# Load the production gauge data
gauge_prod = Data1DGauge()
gauge_prod.load_npz(gauge_path_production)

gauge_interference = Data1DGauge()
gauge_interference.load_npz(interference_test_path)

# Crop the data and combine them
gauge_full_data = Data1DGauge()
gauge_full_data = gauge_prod.copy()

gauge_full_data.select_time(gauge_prod.start_time, gauge_interference.start_time)
gauge_full_data.right_merge(gauge_interference)

gauge_zoom_in_data = Data1DGauge()
gauge_zoom_in_data = gauge_full_data.copy()
gauge_zoom_in_data.select_time(gauge_interference.start_time - datetime.timedelta(hours=1),
                               gauge_interference.start_time + datetime.timedelta(hours=1))

#%% To plot the full pressure profile, also, plot the beginning part of the interference test to see the behavior
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
gauge_full_data.plot(ax=axs[0], use_timestamp=True)
gauge_zoom_in_data.plot(ax=axs[1], use_timestamp=True)
# Mark the start of the interference test
axs[0].axvline(gauge_interference.start_time, color='r', linestyle='--', label='Start of Interference Test')
axs[1].axvline(gauge_interference.start_time, color='r', linestyle='--', label='Start of Interference Test')
axs[0].legend()
axs[0].set_title("Full Pressure Profile During Production and Interference Test")
axs[1].set_title("Zoom-in: Beginning of Interference Test")
plt.tight_layout()
plt.show()