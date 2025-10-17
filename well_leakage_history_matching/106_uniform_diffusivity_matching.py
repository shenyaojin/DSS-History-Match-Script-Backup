import numpy as np
from fiberis.simulator.core import pds
from fiberis.utils import mesh_utils
from fiberis.analyzer.Data1D import Data1D_Gauge, Data1D_PumpingCurve
import matplotlib.pyplot as plt

#%% Load data
datapath = "data/fiberis_format/"

#%% Load gauge data
pg_data_folder = datapath + "s_well/gauges/"
gauge_dataframe_prev = Data1D_Gauge.Data1DGauge()
gauge_dataframe_prev.load_npz(pg_data_folder + "gauge6_data_swell.npz")

gauge_dataframe_next = Data1D_Gauge.Data1DGauge()
gauge_dataframe_next.load_npz(pg_data_folder + "gauge7_data_swell.npz")

#%% Load pumping data (stage 7 and stage 8)
pc_data_folder_stg7 = datapath + "prod/pumping_data/stage7/"
pc_stg7_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg7_prop.load_npz(pc_data_folder_stg7 + "Proppant Concentration.npz")
pc_stg7_slurry_rate.load_npz(pc_data_folder_stg7 + "Slurry Rate.npz")
pc_stg7_pressure.load_npz(pc_data_folder_stg7 + "Treating Pressure.npz")

# pc_stg7_pressure.plot()
# pc_stg7_prop.plot()
# pc_stg7_slurry_rate.plot()

pc_data_folder_stg8 = datapath + "prod/pumping_data/stage8/"
pc_stg8_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg8_prop.load_npz(pc_data_folder_stg8 + "Proppant Concentration.npz")
pc_stg8_slurry_rate.load_npz(pc_data_folder_stg8 + "Slurry Rate.npz")
pc_stg8_pressure.load_npz(pc_data_folder_stg8 + "Treating Pressure.npz")

#%% Get the time range of stages
stg7_bgtime = pc_stg7_slurry_rate.get_start_time()
stg7_edtime = pc_stg7_slurry_rate.get_end_time()

stg8_bgtime = pc_stg8_slurry_rate.get_start_time()
stg8_edtime = pc_stg8_slurry_rate.get_end_time()

#%% Crop the gauge data
phase1_pg_dataframe = gauge_dataframe_prev.copy()
phase1_pg_dataframe.crop(stg7_bgtime, stg7_edtime)
phase1_pg_dataframe.rename("Phase 1 Source")

phase2_pg_dataframe = gauge_dataframe_prev.copy()
phase2_pg_dataframe.crop(stg7_edtime, stg8_bgtime)
phase2_pg_dataframe.rename("Phase 2 Source")

phase3_pg_dataframe = gauge_dataframe_next.copy()
phase3_pg_dataframe.crop(stg8_bgtime, stg8_edtime)
phase3_pg_dataframe.rename("Phase 3 Source")

#%% Get the Measured depth and frac hit info. Use legacy library first.
from DSS_analyzer_Mariner import Data3D_geometry
frac_hit_datapath = "data/legacy/s_well/geometry/frac_hit/"
frac_hit_dataframe_stage7 = Data3D_geometry.Data3D_geometry(frac_hit_datapath + "frac_hit_stage_7_swell.npz")
frac_hit_dataframe_stage8 = Data3D_geometry.Data3D_geometry(frac_hit_datapath + "frac_hit_stage_8_swell.npz")

gauge_md_datapath = "data/legacy/s_well/geometry/gauge_md_swell.npz"
gauge_md_dataframe = Data3D_geometry.Data3D_geometry(gauge_md_datapath)
# Extract the data
frac_hit_md_stg7 = frac_hit_dataframe_stage7.data
frac_hit_md_stg8 = frac_hit_dataframe_stage8.data
gauge_md = gauge_md_dataframe.data[4:10]

#%% define the mesh
dx = 1
nx = 5500
x = np.arange(12500, 12500 + nx * dx, dx)
# refine the mesh round the frac hits
# stage 7, refine the mesh around frac hits.
for frac_hit_iter in np.round(frac_hit_md_stg7):
    x = mesh_utils.refine_mesh(x, [frac_hit_iter - 1, frac_hit_iter + 1], 5)
# stage 8, refine the mesh around frac hits.
for frac_hit_iter in np.round(frac_hit_md_stg8):
    x = mesh_utils.refine_mesh(x, [frac_hit_iter - 1, frac_hit_iter + 1], 5)
nx = len(x)

#%% Update the frac hit index and gauge index
frac_hit_idx_stg7 = [mesh_utils.locate(x, frac_hit_iter)[0] for frac_hit_iter in frac_hit_md_stg7]
frac_hit_idx_stg8 = [mesh_utils.locate(x, frac_hit_iter)[0] for frac_hit_iter in frac_hit_md_stg8]
gauge_idx = [mesh_utils.locate(x, gauge_iter)[0] for gauge_iter in gauge_md]

#%% Set other parameters
d = 140 # Diffusivity coefficient
d_array = np.ones_like(x) * d

t0 = 0

#%% Plot the location
# Plot the location of the pressure gauge and frac hit
plt.figure(figsize=(12, 2))
plt.plot(x, np.zeros_like(x), 'k-', label='Mesh')
for gauge_md_iter in gauge_md:
    # no legend
    plt.axvline(x=gauge_md_iter, color='r', linestyle='--')
plt.plot(frac_hit_md_stg7, np.zeros_like(frac_hit_md_stg7), 'bo', label='Frac Hit Stage 7')
plt.plot(frac_hit_md_stg8, np.zeros_like(frac_hit_md_stg8), 'go', label='Frac Hit Stage 8')
plt.xlabel('Measured Depth/MD (ft)')
plt.title('Pressure Gauge and Frac Hit Locations(multi source)')
plt.legend()
plt.show()


#%% Phase 1 simulation
pds_frame_phase1 = pds.PDS1D_MultiSource()

pds_frame_phase1.set_mesh(x)
pds_frame_phase1.set_diffusivity(d_array)
pds_frame_phase1.set_bcs('Neumann', 'Neumann')
pds_frame_phase1.set_t0(t0) # Phase 1 starts at 0

# Set the source idx
pds_frame_phase1.set_sourceidx(frac_hit_idx_stg7)

# Create the source term array
source_phase1_list = []
for source_iter in range(len(frac_hit_idx_stg7)):
    source_phase1_list.append(phase1_pg_dataframe)

pds_frame_phase1.set_source(source_phase1_list)

u_initial = np.zeros(nx)
u_initial += phase1_pg_dataframe.data[0]
pds_frame_phase1.set_initial(u_initial)

pds_frame_phase1.solve(optimizer= True, dt_init = 2, print_progress=True,
                  max_dt=30, min_dt=1e-4, tol=1e-3, safety_factor=0.9, p=2)

pds_frame_phase1.pack_result(filename="output/0211_simulation_MULTIstage/phase1.npz")

#%% Phase 2 Simulation
# Calculate the total_time for phase 2
phase2_total_time = (stg8_bgtime - stg7_edtime).total_seconds()

#%% Set up the simulation
pds_frame_phase2 = pds.PDS1D_MultiSource()
pds_frame_phase2.set_mesh(x)
pds_frame_phase2.set_diffusivity(d_array)
pds_frame_phase2.set_bcs('Neumann', 'Neumann')
pds_frame_phase2.set_t0(t0) # Considering the gauge dataframe, the t0 will still be 0
pds_frame_phase2.set_sourceidx(frac_hit_idx_stg7)

# Load last phase's result
# from fiberis.analyzer.Data2D import Data2D_XT_DSS
# prev_result = Data2D_XT_DSS.DSS2D()
prev_result = np.load("output/0211_simulation_MULTIstage/phase1.npz", allow_pickle=True)
initial = prev_result['data'][-1, :]
pds_frame_phase2.set_initial(initial)

#%% set up the source term
source_phase2_list = []
for source_iter in range(len(frac_hit_idx_stg7)):
    source_phase2_list.append(phase2_pg_dataframe)

pds_frame_phase2.set_source(source_phase2_list)

#%% Solve the problem
pds_frame_phase2.solve(optimizer= True, dt_init = 2, print_progress=True,
                  max_dt=30, min_dt=1e-4, tol=1e-3, safety_factor=0.9, p=2)

#%% Pack the result
pds_frame_phase2.pack_result(filename="output/0211_simulation_MULTIstage/phase2.npz")
pds_frame_phase2.plot_solution()

#%% Phase 3 Simulation
# Before we start the simulation, set the diffusivity change ratio
# 0.1, 0.01, <- finished
diffusivity_change_ratio = np.array([ 0.001, 0.0001, 0.00001, 0.000001])
phase3_total_time = phase3_pg_dataframe.taxis[-1]

#%% Set up other parameters for the simulation which are the same as phase 1 and phase 2
pds_frame_phase3 = pds.PDS1D_MultiSource()
pds_frame_phase3.set_mesh(x)
pds_frame_phase3.set_bcs('Neumann', 'Neumann')
pds_frame_phase3.set_sourceidx(frac_hit_idx_stg8)

#%% Update the diffusivity
d_pahse3 = np.ones_like(x) * d
# change the diffusivity at the frac hit location at stage 7
for frac_hit_idx_iter in frac_hit_idx_stg7:
    d_pahse3[frac_hit_idx_iter] = d * diffusivity_change_ratio[-1]
pds_frame_phase3.set_diffusivity(d_pahse3)

#%% Load the initial condition.
# from fiberis.analyzer.Data2D import Data2D_XT_DSS
# prev_result = Data2D_XT_DSS.DSS2D()
# prev_result.load_npz("output/0211_simulation_MULTIstage/phase2.npz")
# initial = prev_result.data[-1]
prev_result = np.load("output/0211_simulation_MULTIstage/phase2.npz", allow_pickle=True)
initial = prev_result['data'][-1, :]
pds_frame_phase3.set_initial(initial)

#%% Set up the source term
source_phase3_list = []
for source_iter in range(len(frac_hit_idx_stg8)):
    source_phase3_list.append(phase3_pg_dataframe)

pds_frame_phase3.set_source(source_phase3_list)

#%% Solve the problem
pds_frame_phase3.solve(optimizer= True, dt_init = 2, print_progress=True,
                  max_dt=30, min_dt=1e-4, tol=1e-3, safety_factor=0.9, p=2)

#%% Pack the result
pds_frame_phase3.pack_result(filename="output/0211_simulation_MULTIstage/phase3_test.npz")
pds_frame_phase3.plot_solution()

#%% Phase 3, full loop

for diffusivity_iter in diffusivity_change_ratio[:-1]:

    pds_frame_phase3 = pds.PDS1D_MultiSource()
    pds_frame_phase3.set_mesh(x)
    pds_frame_phase3.set_bcs('Neumann', 'Neumann')
    pds_frame_phase3.set_sourceidx(frac_hit_idx_stg8)

    d_pahse3 = np.ones_like(x) * d
    # change the diffusivity at the frac hit location at stage 7
    for frac_hit_idx_iter in frac_hit_idx_stg7:
        d_pahse3[frac_hit_idx_iter] = d * diffusivity_iter
    pds_frame_phase3.set_diffusivity(d_pahse3)

    prev_result = np.load("output/0211_simulation_MULTIstage/phase2.npz", allow_pickle=True)
    initial = prev_result['data'][-1, :]
    pds_frame_phase3.set_initial(initial)

    # Set up the source term
    source_phase3_list = []
    for source_iter in range(len(frac_hit_idx_stg8)):
        source_phase3_list.append(phase3_pg_dataframe)

    pds_frame_phase3.set_source(source_phase3_list)

    # Solve the problem
    pds_frame_phase3.solve(optimizer=True, dt_init=2, print_progress=True,
                           max_dt=30, min_dt=1e-4, tol=1e-3, safety_factor=0.9, p=2)

    # Pack the result
    pds_frame_phase3.pack_result(filename=f"output/0211_simulation_MULTIstage/phase3_{diffusivity_iter}.npz")
