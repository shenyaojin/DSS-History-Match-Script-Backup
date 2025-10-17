#%% Load data
from fiberis.analyzer.Data1D.core1D import Data1D
import numpy as np
import matplotlib.pyplot as plt

gauge_path = "data/fiberis_format/s_well/gauges/gauge8_data_swell.npz"
gauge_data_field = Data1D()
gauge_data_field.load_npz(gauge_path)

#%% Downsample it
gauge_data_field_downsampled = gauge_data_field.copy()
gauge_data_field_downsampled.adaptive_downsample(110)

#%% Plot the data
fig, ax = plt.subplots(figsize=(8,4))
gauge_data_field.plot(ax=ax, use_timestamp=True, use_legend=True, label='Measured pressure (field)')
gauge_data_field_downsampled.plot(ax=ax, use_timestamp=True, use_legend=True, label='Downsampled pressure (field)')
ax.set_title('Gauge 8 Pressure Comparison, field vs downsampled')
ax.set_ylabel('Pressure (psi)')
plt.tight_layout()
plt.show()

#%% Print the size of the data
print(f"Original size: {len(gauge_data_field.data)}")
print(f"Downsampled size: {len(gauge_data_field_downsampled.data)}")