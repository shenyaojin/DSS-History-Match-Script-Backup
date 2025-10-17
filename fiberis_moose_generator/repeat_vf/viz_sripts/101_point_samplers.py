#%% Load libraries
import matplotlib.pyplot as plt
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

folder = "output/0808_vf_reproduce_refined"

#%%  Load point sampler data
point_samplers = MOOSEPointSamplerReader()
# Get the max variable index
max_ind = point_samplers.get_max_index(folder=folder)


#%% Recursive reading of point sampler data
output_dir = "output/0812_moose_viz"
for i in range(1, max_ind + 1):
    point_samplers.read(folder=folder, variable_index=i)
    plt.figure(figsize=[6,2])
    plt.plot(point_samplers.taxis[1:], point_samplers.data[1:], label=point_samplers.variable_name)

    plt.xlabel("Time [s]")
    plt.ylabel(point_samplers.variable_name)
    plt.title(f"Point Sampler Data - Variable {i}")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{point_samplers.variable_name}.png")
    plt.close()  # Close the figure to save memory
