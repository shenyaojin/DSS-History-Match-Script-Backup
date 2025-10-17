#%% Load libraries
import matplotlib.pyplot as plt
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
import numpy as np

#%% folder
vector_sampler = MOOSEVectorPostProcessorReader()
folder = "output/0808_vf_reproduce_refined"
# Get the max variable index
max_ind = vector_sampler.get_max_indices(folder)
print(max_ind)

max_ind = np.array(max_ind)

#%% Test read method
vector_sampler.read(folder, 0, 1)
print(vector_sampler.taxis[:10])
print(vector_sampler.xaxis[:10])
print(vector_sampler.yaxis[:10])

#%% Read the data recursively
for i in range(max_ind[0] + 1): # Load all post processors
    for j in range(1, max_ind[1] + 1):
        vector_sampler.read(folder, i, j)
        # Calculate the distance
        distance = np.sqrt(
            (vector_sampler.xaxis - vector_sampler.xaxis[0]) ** 2 +
            (vector_sampler.yaxis - vector_sampler.yaxis[0]) ** 2
        )

        clim = (
            np.min(vector_sampler.data[:, 1:]),
            np.max(vector_sampler.data[:, 1:])
        )

        # Plot the data in pcolormesh
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(
            vector_sampler.taxis[1:],
            distance,
            vector_sampler.data[:, 1:],
            cmap='bwr',
            clim=clim,
            shading='auto'
        )
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (m)')
        plt.title(f"{vector_sampler.variable_name} at index ({i}, {j})")
        plt.tight_layout()
        plt.show()