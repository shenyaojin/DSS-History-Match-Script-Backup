#%% Visualization of multi-frac
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data3D import core3D

#%% Load the data
file_loc = "output/moose_full_sim_output_fixed_v14"
dataframe = core3D.Data3D()

dataframe.load_from_csv_series(directory=file_loc, variable_index=1, post_processor_id=0)

#%% Plotting the data
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# Use imshow to visualize the data
extent = (dataframe.taxis[1], dataframe.taxis[-1], dataframe.xaxis[0], dataframe.xaxis[-1])
im = ax.imshow(dataframe.data[:, 1:], aspect='auto', extent=extent, origin='lower', cmap='viridis')
# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Pressure (Pa)')
# Set labels and title
ax.set_ylabel('Distance (m)')
ax.set_xlabel('Time (s)')
plt.show()

#%% Load point sampler
from fiberis.analyzer.Data1D import Data1D_MOOSEps

dataframe = Data1D_MOOSEps.Data1D_MOOSEps()
dataframe.read_csv(file_loc, variable_index=1)

#%% Plotting the point sampler data
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
dataframe.plot(ax=ax, use_timestamp=False)
plt.show()