# Simple visualization of optimized result using new viz_utils function.
# Shenyao Jin

import numpy as np
import matplotlib.pyplot as plt
import os

from fiberis.analyzer.Data2D.core2D import Data2D
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.utils.viz_utils import plot_dss_and_gauge_co_plot


print(os.getcwd())
best_result_path = "output/109r4_optimization/iter_025/strain_dataframe.npz"
INJECTION_PRESSURE_PROFILE_PATH = "data/fiberis_format/post_processing/injection_pressure_full_profile.npz"

dss_data = Data2D()
dss_data.load_npz(best_result_path)

# Select depth range
dss_data.daxis = dss_data.daxis / 0.3048  # Convert to feet
dss_data.history.add_record("Convert depth to feet", level='INFO')

pg_data = Data1D()
pg_data.load_npz(INJECTION_PRESSURE_PROFILE_PATH)

# Define plotting arguments
d2_plot_args = {
    'cmap': 'bwr',
    'clim': (-2e-5, 2e-5),
    'method': 'pcolormesh',
    'title': "DSS Data (simulated)",
    'ylabel': "Depth (ft)",
    'clabel': "Strain"
}

d1_plot_args = {
    'ylabel': "Pressure (psi)"
}

# Plot using the new function
fig, (ax1, ax2) = plot_dss_and_gauge_co_plot(dss_data, pg_data,
                                            # d2_plot_args=d2_plot_args,
                                            # d1_plot_args=d1_plot_args
                                             )

plt.show()
