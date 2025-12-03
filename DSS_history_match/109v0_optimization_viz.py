# Simple visualization of optimized result.
# Shenyao Jin
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


print(os.getcwd())
best_result_path = "output/109r4_optimization/iter_026/strain_dataframe.npz"
INJECTION_PRESSURE_PROFILE_PATH = "data/fiberis_format/post_processing/injection_pressure_full_profile.npz"

DSSdata = DSS2D()
DSSdata.load_npz(best_result_path)

# Select depth range
DSSdata.daxis = DSSdata.daxis / 0.3048  # Convert to feet
DSSdata.history.add_record("Convert depth to feet", level='INFO')
# center_x = DSSdata.daxis[len(DSSdata.daxis)//2]
# DSSdata.select_depth(center_x - 10, center_x + 10)

pg_dataframe = Data1DGauge()
pg_dataframe.load_npz(INJECTION_PRESSURE_PROFILE_PATH)

pg_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())

# Plotting
fig = plt.figure(figsize=(7, 6))
ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4) # <- DSS plot
ax2 = plt.subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4, sharex=ax1) # <- Pressure gauge plot

# DSS plot
# Plot DSS data
im1 = DSSdata.plot(ax=ax1, use_timestamp=False, cmap='bwr', vmin=-2e-5, vmax=2e-5, method='pcolormesh')
ax1.set_title("DSS Data (simulated)")
ax1.set_ylabel("Depth (ft)")


# Hide x-axis ticks
ax1.tick_params(labelbottom=False)
# Plot pressure gauge data
im2 = pg_dataframe.plot(ax=ax2, use_timestamp=False)
ax2.set_ylabel("Pressure (psi)")

# add colorbar
divider = make_axes_locatable(ax1)

# Append axes to the right of ax1, with 2% width and 0.05 padding
cax = divider.append_axes("right", size="2%", pad=0.05)

# Add colorbar to the newly created cax
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label("Strain")

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="2%", pad=0.05)
cax2.axis('off')

plt.tight_layout()
plt.show()