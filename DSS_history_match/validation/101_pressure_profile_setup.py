#%% Pre-processing the pressure profile
# We basically need two: one is prod+interference; another one is interference test
# Shenyao Jin, 03/13/2026

import numpy as np
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
import matplotlib.pyplot as plt
import os
import datetime

# Print path
print("working directory: ", os.getcwd())

#%% Read in the full profile
# full_profile_path = "data/fiberis_format/post_processing/injection_pressure_full_profile.npz"
full_profile_path = "data/fiberis_format/post_processing/timestepper_profile.npz"

gauge_dataframe = Data1DGauge()
gauge_dataframe.load_npz(full_profile_path)
gauge_dataframe.remove_abnormal_data(threshold=300)
gauge_dataframe.select_time(datetime.datetime(2020, 4, 1),
                            gauge_dataframe.get_end_time(use_timestamp=True))

fig, ax = plt.subplots()
gauge_dataframe.plot(ax=ax, use_timestamp=True)
plt.show()

#%% Print end time
print(gauge_dataframe.get_end_time())

# #%% Save this profile to validation
# gauge_dataframe.savez("scripts/DSS_history_match/validation/data/full.npz")

#%% Read the profile during interference test
full_profile_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
gauge_dataframe.load_npz(full_profile_path)

DSS_profile_path = "data/fiberis_format/prod/dss_data/Mariner 14x-36-D - RFS strain change.npz"
DSS_data = DSS2D()
DSS_data.load_npz(DSS_profile_path)

# processing
gauge_dataframe.remove_abnormal_data(threshold=300)
gauge_dataframe.select_time(DSS_data.start_time, DSS_data.get_end_time())

#%% Set initial pressure to zero to mimic injection
gauge_dataframe.data -= gauge_dataframe.data[0]

fig, ax = plt.subplots()
gauge_dataframe.plot(ax=ax, use_timestamp=True)
plt.show()

#%% Save to npz
gauge_dataframe.savez("scripts/DSS_history_match/validation/data/interference.npz")

#%% End time
