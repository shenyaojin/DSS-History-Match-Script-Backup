import os

import numpy as np
import datetime
from fiberis.simulator.core import pds
from fiberis.utils import mesh_utils
from fiberis.analyzer.Data1D import Data1D_Gauge
import matplotlib.pyplot as plt
import itertools

#%% Load data / Crop the gauge data
datapath = "data/fiberis_format/"

start_time = datetime.datetime(2020,4,1)
end_time = datetime.datetime(2021,7,1)

#%% Load gauge data
pg_data_folder = datapath + "prod/gauges/"
gauge_data_all = []
for iter_filename in os.listdir(pg_data_folder):
    datapath_iter = os.path.join(pg_data_folder, iter_filename)
    gauge_dataframe_iter = Data1D_Gauge.Data1DGauge()
    gauge_dataframe_iter.load_npz(datapath_iter)
    gauge_dataframe_iter.crop(start_time, end_time)
    gauge_data_all.append(gauge_dataframe_iter)

#%% Get the Measured depth and frac hit info. Use legacy library first.
from DSS_analyzer_Mariner import Data3D_geometry
frac_hit_datapath = "data/legacy/s_well/geometry/frac_hit/"
frac_hit_dataframe_stage7 = Data3D_geometry.Data3D_geometry(frac_hit_datapath + "frac_hit_stage_7_swell.npz")
frac_hit_dataframe_stage8 = Data3D_geometry.Data3D_geometry(frac_hit_datapath + "frac_hit_stage_8_swell.npz")

#%% Load all the frac hit data
frac_hit_dataframe_all = []
frac_hit_data = []
for iter_frachit in os.listdir(frac_hit_datapath):
    full_path_tmp = frac_hit_datapath + iter_frachit
    frac_hit_dataframe = Data3D_geometry.Data3D_geometry(full_path_tmp)
    frac_hit_dataframe_all.append(frac_hit_dataframe)
    frac_hit_data.append(frac_hit_dataframe.data)
# Combine the elements in the list
frac_hit_data = list(itertools.chain.from_iterable(
    item.data if isinstance(item.data, (list, np.ndarray)) else [item.data]
    for item in frac_hit_dataframe_all
))

gauge_md_datapath = "data/legacy/s_well/geometry/gauge_md_swell.npz"
gauge_md_dataframe = Data3D_geometry.Data3D_geometry(gauge_md_datapath)
# Extract the data
frac_hit_md_stg7 = frac_hit_dataframe_stage7.data
frac_hit_md_stg8 = frac_hit_dataframe_stage8.data
gauge_md = gauge_md_dataframe.data

# Set up the mesh
mesh = np.arange(11890, 16890, 10)
gauge_idx = 2 # gauge 3 (center at the producer)
dt = (gauge_data_all[0].taxis[1] - gauge_data_all[0].taxis[0]) * 100

# get the idx of frac hit.
frac_hit_data_list = list(itertools.chain.from_iterable(
    item.data if isinstance(item.data, (list, np.ndarray)) else [item.data]
    for item in frac_hit_dataframe_all
))

frac_hit_data_list = np.array(frac_hit_data_list)

# refine the mesh
for idx_iter in range(len(frac_hit_data_list)):
    mesh = mesh_utils.refine_mesh(mesh, [frac_hit_data_list[idx_iter] - 1, frac_hit_data_list[idx_iter] + 1]
                                  , 20)

frac_hit_idx_list = []
for frac_hit_idx_iter in frac_hit_data_list:
    frac_hit_idx_iter, _ = mesh_utils.locate(mesh, frac_hit_idx_iter)
    frac_hit_idx_list.append(frac_hit_idx_iter)

# for frac_hit_idx_iter in frac_hit_idx_list:

# Set up the source idx.
source_idx_list = []
source_idx, _ = mesh_utils.locate(mesh, (mesh[0] + mesh[-1]) / 2)
source_idx_list.append(source_idx)
# for source_iter in frac_hit_md_stg7:
#     source_idx, _ = mesh_utils.locate(mesh, source_iter)
#     source_idx_list.append(source_idx)

# Set up the diffusivity array
diffusivity_array_all = []
# diffusivity_drop_ratio = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
diffusivity_drop_ratio = np.array([1, 1e-1])
for diff_drop_iter in diffusivity_drop_ratio:
    diffusivity_array_tmp = np.ones_like(mesh) * 140 # original diffusivity
    for item in frac_hit_idx_list:
        diffusivity_array_tmp[item]  = 140 * diff_drop_iter # original diffusivity * drop ratio
    diffusivity_array_all.append(diffusivity_array_tmp)

    plt.figure()
    plt.plot(mesh, diffusivity_array_tmp, label='Diffusivity',
             marker='o', linestyle='-', linewidth=0.5, markersize=0.5)
    plt.xlabel('Measured Depth/ft')
    plt.ylabel('Diffusivity')
    plt.xlim([np.min(frac_hit_data) - 200, np.max(frac_hit_data) + 200])
    plt.show()

#%% Set up the loop for the simulator1d.
flag = 0
for diff_drop_iter in diffusivity_array_all:
    # Set up the simulator1d
    simulator = pds.PDS1D_MultiSource()
    simulator.set_t0(0)
    simulator.set_bcs('Neumann', 'Neumann')
    simulator.set_mesh(mesh)
    simulator.set_diffusivity(diff_drop_iter)

    initial_snapshot = np.zeros_like(mesh)
    initial_snapshot[:] = gauge_data_all[gauge_idx].data[0]
    simulator.set_initial(initial_snapshot)

    # Set up the source
    simulator.set_sourceidx(source_idx_list)
    source_list = []
    for iter_source in range(len(source_idx_list)):
        source_list.append(gauge_data_all[gauge_idx])
    simulator.set_source(source_list)

    # self check
    simulator.self_check()
    simulator.solve(optimizer=False, dt=dt,
                    t_total= gauge_data_all[0].taxis[-1], print_progress=True)

    plt.figure()
    # Use pcolor to plot the result
    plt.pcolormesh(simulator.taxis, simulator.mesh, simulator.snapshot.T,
                   cmap='bwr')
    # Invert the y-axis
    plt.clim(2000, 8000)
    plt.gca().invert_yaxis()
    plt.xlabel("Time/s")
    plt.ylabel("Distance/ft")
    plt.title("Diffusivity Drop Ratio = " + str(diffusivity_drop_ratio[flag]))
    plt.savefig(f"figs/03252025/"+ str(diffusivity_drop_ratio[flag]) + ".png")
    plt.show()
    # Pack the result
    simulator.pack_result(filename=f"output/0324_forward_simulator/{diffusivity_drop_ratio[flag]}_gauge{gauge_idx+1}.npz")

    flag += 1

#%% Setup the pds
# simulator1d = pds.PDS1D_SingleSource()
# simulator1d.set_t0(0)
# simulator1d.set_bcs('Neumann', 'Neumann')
#
# mesh = np.arange(np.min(gauge_md) - 200, np.max(gauge_md) + 200, 1)
# taxis = gauge_data_all[0].taxis
#
# gauge_idx = 5 # gauge 6
#
# simulator1d.set_mesh(mesh)
# simulator1d.set_source(gauge_data_all[gauge_idx]) # Gauge 6
#
# initial_snapshot = np.zeros_like(mesh)
# initial_snapshot[:] = gauge_data_all[gauge_idx].data[0]
# simulator1d.set_initial(initial_snapshot)
#
# diffusivity_array = np.ones_like(mesh) * 1.40
# # idx, _ = mesh_utils.locate(mesh, gauge_md[gauge_idx])
# idx, _ = mesh_utils.locate(mesh, frac_hit_md_stg7[1])
# diffusivity_array[idx] = 1.40 * 0.001 # ft^2/s
#
# simulator1d.set_diffusivity(diffusivity_array)
#
# # set up the source idx
# source_idx, _ = mesh_utils.locate(mesh, gauge_md[gauge_idx])
# simulator1d.set_sourceidx(source_idx)
#
# #%% self check
# simulator1d.self_check()
#
# #%% Run the simulation
# simulator1d.solve(optimizer= False, dt = (taxis[1] - taxis[0]) * 100,
#                 t_total= taxis[-1], print_progress=True)
# #%% Plot the result
# simulator1d.plot_solution()
#
# #%% Co plot the simulation and field data
# for gauge_iter in range(len(gauge_data_all)):
#     # Create a figure and axis
#     fig, ax = plt.subplots()
#
#     # Plot the field data
#     gauge_data_all[gauge_iter].plot(ax=ax)
#
#     # Locate the relevant simulator1d index
#     simulator_idx, _ = mesh_utils.locate(mesh, gauge_md[gauge_iter])
#
#     # Construct the time axis for synthetic data
#     dt = (taxis[1] - taxis[0]) * 100  # Time step
#     # or: synthetic_taxis = np.arange(0, len(simulator1d.snapshot[:, simulator_idx]) * dt, dt)
#     synthetic_taxis = np.arange(0, len(simulator1d.snapshot[:, simulator_idx])) * dt
#
#     # Plot the synthetic data
#     ax.plot(synthetic_taxis, simulator1d.snapshot[:, simulator_idx], label='Synthetic Data')
#
#     # Add legend
#     ax.legend()
#
#     # Save and show figure
#     # plt.savefig(f'figs/02172025/gauge{gauge_iter+1}_new.png')
#     plt.show()
#     # This is not we want to see