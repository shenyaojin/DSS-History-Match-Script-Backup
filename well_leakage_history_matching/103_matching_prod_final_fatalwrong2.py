#%% Import libraries
import numpy as np
import datetime
from glob import glob
from fiberis.simulator.core import pds
from fiberis.utils import mesh_utils
from fiberis.analyzer.Data1D import Data1D_Gauge # 1D simulator source
from fiberis.analyzer.Geometry3D import DataG3D_md # gauge md info; well geometry
import matplotlib.pyplot as plt

#%% Load the data
datapath = "data/fiberis_format/"

start_time = datetime.datetime(2020,4,1)
end_time = datetime.datetime(2021,7,1)

#%% Load gauge data. In the production period, use gauge in producer
pg_data_path = "data/fiberis_format/prod/gauges/gauge3_data_prod.npz"
pg_dataframe = Data1D_Gauge.Data1DGauge()
pg_dataframe.load_npz(pg_data_path)
pg_dataframe.crop(start_time, end_time)

#%% Load the geometry
frac_hit_md = glob("data/fiberis_format/s_well/geometry/frac_hit/*.npz")
frac_hit_md = sorted(frac_hit_md)

gauge_md_data = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
gauge_md_dataframe = DataG3D_md.G3DMeasuredDepth()
gauge_md_dataframe.load_npz(gauge_md_data)

from tqdm import tqdm
frac_hit = []
for file in tqdm(frac_hit_md):
    frac_hit_dataframe = DataG3D_md.G3DMeasuredDepth()
    frac_hit_dataframe.load_npz(file)
    frac_hit.append(frac_hit_dataframe.data)

frac_hit_sep = frac_hit.copy()
frac_hit = np.concatenate(frac_hit)

#%% Set up the simulator
from fiberis.analyzer.Data2D import Data2D_XT_DSS
DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz("data/fiberis_format/s_well/DAS/LFDASdata_stg1_interval_swell.npz")

mesh = np.linspace(DASdata.daxis[-1] -5000, DASdata.daxis[-1] + 5000, 1000)
# refine the mesh
for frac_hit_iter in frac_hit:
    mesh = mesh_utils.refine_mesh(mesh, [frac_hit_iter - 1, frac_hit_iter + 1], 10)
nx = len(mesh)
dt = (pg_dataframe.taxis[1] - pg_dataframe.taxis[0]) * 100

diffusivity_array_all = []
diffusivity_drop_ratio = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

#%% DO the simulation
for ratio in diffusivity_drop_ratio:
    drop_array = np.linspace(1, ratio, 5)
    drop_array_rev = drop_array[::-1][1:]
    drop_array = np.concatenate([drop_array, drop_array_rev])
    diffusivity_array = np.ones_like(mesh) * 140
    for frac_hit_iter in frac_hit:
        idx, _ = mesh_utils.locate(mesh, frac_hit_iter)
        diffusivity_array[idx - 4: idx + 5] = drop_array * 140
        diffusivity_array_all.append(diffusivity_array)

    # plt.figure()
    # plt.plot(mesh, diffusivity_array, label='Diffusivity',
    #          marker='o', linestyle='-', linewidth=0.5, markersize=0.5)
    # plt.xlabel('Measured Depth/ft')
    # plt.ylabel('Diffusivity')
    # plt.xlim([mesh[0]-100, mesh[-1]+100])
    # plt.show()

    flag = 0
    for diffusivity_drop_iter in diffusivity_array_all:
        pds_simulator = pds.PDS1D_MultiSource()
        pds_simulator.set_t0(0)
        pds_simulator.set_bcs('Neumann', 'Neumann')
        pds_simulator.set_mesh(mesh)
        pds_simulator.set_diffusivity(diffusivity_drop_iter)

        init_snapshot = np.zeros_like(mesh)
        init_snapshot[:] = pg_dataframe.data[0]
        pds_simulator.set_initial(init_snapshot)

        source_idx, _ = mesh_utils.locate(mesh, (mesh[0] + mesh[-1]) / 2)
        pds_simulator.set_sourceidx([source_idx])
        pds_simulator.set_source([pg_dataframe])

        pds_simulator.self_check()
        pds_simulator.solve(optimizer=False, dt=dt,
                        t_total=pg_dataframe.taxis[-1], print_progress=True)

        plt.figure()
        # Use pcolormesh to plot the result
        plt.pcolormesh(pds_simulator.taxis, pds_simulator.mesh, pds_simulator.snapshot.T,
                       cmap='bwr')
        # Invert the y-axis
        plt.clim(2000, 8000)
        plt.gca().invert_yaxis()
        plt.xlabel("Time/s")
        plt.ylabel("Distance/ft")
        plt.title("Diffusivity Drop Ratio = " + str(diffusivity_drop_ratio[flag]))
        plt.show()
        # Pack the result
        pds_simulator.pack_result(filename=f"output/0324_forward_simulator/{diffusivity_drop_ratio[flag]}.npz")

        flag += 1