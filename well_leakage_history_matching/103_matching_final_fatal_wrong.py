# I hope it will be the last version of simultion script.
# I don't know why this script has such a large difference with the previous one. The result seems to be wrong.
import datetime
from tqdm import tqdm
from fiberis.simulator.core import pds
from fiberis.utils import mesh_utils

#%% Load the pressure gauge data in producer
import numpy as np
from fiberis.analyzer.Data1D import Data1D_Gauge

dataframe = Data1D_Gauge.Data1DGauge()
dataframe.load_npz("data/fiberis_format/prod/gauges/gauge4_data_prod.npz")

start_time = datetime.datetime(2020, 4, 1)
end_time = datetime.datetime(2021, 9, 1)

dataframe.crop(start_time, end_time)
gauge_dataframe = dataframe.copy()

#%% Create the plot to check the data of the dataframe.
import matplotlib.pyplot as plt
plt.figure()
gauge_dataframe.plot(useTimeStamp=True)
plt.show()

#%% Remove the bad data
gauge_dataframe.remove_abnormal_data(method='mean')

#%% QC for clean data
plt.figure()
gauge_dataframe.plot(useTimeStamp=True, title="QC for clean data")
plt.show()

#%% Old Script

from glob import glob
# Gauge md
gauge_md_data = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
# Frac Hit md
frac_hit_md = glob("data/fiberis_format/s_well/geometry/frac_hit/*.npz")
# sort the file
frac_hit_md = sorted(frac_hit_md)

from fiberis.analyzer.Geometry3D import DataG3D_md
gauge_md_dataframe = DataG3D_md.G3DMeasuredDepth()
gauge_md_dataframe.load_npz(gauge_md_data)
gauge_md = gauge_md_dataframe.data

#%% Simulation

from fiberis.analyzer.Data2D import Data2D_XT_DSS
DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz("data/fiberis_format/s_well/DAS/LFDASdata_stg1_interval_swell.npz")
# Set up the grid. 6000 ft with Neumann BC; finer grid at frac hit location
mesh = np.linspace(DASdata.daxis[-1] -5500, DASdata.daxis[-1], 5500)
# Combine the frac hit location
frac_hit = []
for file in tqdm(frac_hit_md):
    frac_hit_dataframe = DataG3D_md.G3DMeasuredDepth()
    frac_hit_dataframe.load_npz(file)
    frac_hit.append(frac_hit_dataframe.data)

# Convert frac hit to 1D array
frac_hit_sep = frac_hit.copy()
frac_hit = np.concatenate(frac_hit)
mesh = np.linspace(DASdata.daxis[-1] -4900, DASdata.daxis[-1], 4900)
# refine the mesh
for frac_hit_iter in frac_hit:
    mesh = mesh_utils.refine_mesh(mesh, [frac_hit_iter - 1, frac_hit_iter + 1], 10)
nx = len(mesh)

dt = (gauge_dataframe.taxis[-1] - gauge_dataframe.taxis[0]) / 50


#%% Create the diffusivity mesh
ratio_arr = np.array([0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6])
diffusivity_array_all = []
for ratio in ratio_arr:
    drop_array = np.linspace(1, ratio, 5)
    drop_array_rev = drop_array[::-1][1:]
    drop_array = np.concatenate([drop_array, drop_array_rev])
    diffusivity_array = np.ones_like(mesh) * 1.40
    for frac_hit_iter in frac_hit:
        idx, _ = mesh_utils.locate(mesh, frac_hit_iter)
        diffusivity_array[idx - 4: idx + 5] = drop_array * 1.40
        diffusivity_array_all.append(diffusivity_array)

    # plt.figure()
    # plt.plot(mesh, diffusivity_array, label='Diffusivity',
    #          marker='o', linestyle='-', linewidth=0.5, markersize=0.5)
    # plt.xlabel('Measured Depth/ft')
    # plt.ylabel('Diffusivity')
    # plt.xlim([mesh[0]-100, mesh[-1]+100])
    # plt.show()

#%% Set up the simulator
flag = 0
for diffusivity_array in diffusivity_array_all:
    simulator1d = pds.PDS1D_SingleSource()

    simulator1d.set_mesh(mesh)
    simulator1d.set_bcs('Neumann', 'Neumann')
    simulator1d.set_t0(0)
    simulator1d.set_initial(np.ones_like(mesh) * gauge_dataframe.data[0])
    simulator1d.set_diffusivity(diffusivity_array)

    # Source idx
    center_val = (mesh[0] + mesh[-1]) / 2
    idx, _ = mesh_utils.locate(mesh, center_val)

    # Set up the source
    simulator1d.set_sourceidx(idx)
    simulator1d.set_source(gauge_dataframe)

    simulator1d.self_check()
    simulator1d.solve(optimizer=False, dt=dt, t_total=gauge_dataframe.taxis[-1],
                      print_progress=True)

    plt.figure()
    # Use pcolormesh to plot the result
    plt.pcolormesh(simulator1d.taxis, simulator1d.mesh, simulator1d.snapshot.T,
                   cmap='bwr')
    # Invert the y-axis
    plt.clim(2000, 8000)
    plt.gca().invert_yaxis()
    plt.xlabel("Time/s")
    plt.ylabel("Distance/ft")
    plt.title("Diffusivity Drop Ratio = " + str(ratio_arr[flag]))
    plt.show()

    # Pack the result
    simulator1d.pack_result(
        filename=f"output/0324_forward_simulator/{ratio_arr[flag]}.npz")

    flag += 1
