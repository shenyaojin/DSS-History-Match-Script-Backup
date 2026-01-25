# To visualize the source data and generate a time stepper profile for MOOSE simulations
# Shenyao Jin

import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
gauge_data_for_moose = Data1DGauge()
gauge_data_for_moose.load_npz("data_fervo/fiberis_format/pressure_data/Bearskin3PA_Stage_28.npz")

fig, ax = plt.subplots(figsize=(10, 6))
gauge_data_for_moose.plot(ax=ax, use_timestamp=False)
plt.show()

# Extract time stepper profile
#
time_stepper_profile = gauge_data_for_moose.copy()
time_stepper_profile.adaptive_downsample(140)

# Plot time stepper profile
fig, ax = plt.subplots(figsize=(10, 6))
time_stepper_profile.plot(ax=ax, use_timestamp=True)
ax.set_title("Time Stepper Profile for MOOSE Simulation")
plt.show()

# Save time stepper profile
time_stepper_profile.savez("data_fervo/fiberis_format/post_processing/Bearskin3PA_Stage_28_timestep_profile.npz")
