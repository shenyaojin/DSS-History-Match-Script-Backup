# This script will perform a full workflow for the degradation version of HMM modeling.
# Including data loading, preprocessing, simulation and visualization.
# Shenyao Jin, 2026-01

import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.templates.baseline_model_generator import build_baseline_model, post_processor_info_extractor
from fiberis.moose.runner import MooseRunner
import os

print("working directory:", os.getcwd())

project_name = "0113_test_degradation"
builder = build_baseline_model(project_name=project_name,
                               srv_perm=2.87e-17,
                               fracture_perm=1.09e-15,
                               matrix_perm=1e-19,
                               ny_per_layer_half=100,
                               bias_y=1.15
                               ) # This parameter set is stable


# Output the model
output_dir = f"output/{project_name}"
os.makedirs(output_dir, exist_ok=True)

# Generate the model
input_file_path = os.path.join(output_dir, f"{project_name}_input.i")

builder.generate_input_file(output_filepath=input_file_path)

# Run the simulation
runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)
success, stdout, stderr = runner.run(
    input_file_path=input_file_path,
    output_directory=output_dir,
    num_processors=20,
    log_file_name="simulation.log",
    stream_output=True
)

# Post-process the results
pressure_dataframe, strain_dataframe = post_processor_info_extractor(output_dir=output_dir)
pg_frame = Data1DGauge()
pg_frame.load_npz("data/fiberis_format/post_processing/timestepper_profile.npz")

# Crop the data
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)

start_time = DSSdata.start_time
pressure_dataframe.start_time = start_time
strain_dataframe.start_time = start_time

#%% Pre=process DSS data
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
# DSSdata.select_depth(14820, 14920) # <- Select depth range of interest.
# DSSdata.select_depth(14980, 15010)

DSSdata.select_depth(14880, 14900) # <- Select depth range of interest.

pressure_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())
strain_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())

# Calibrate strain data by add first value of each channel
if strain_dataframe.data.shape[0] > 0:
    # Subtract the first value of each channel (row) from that entire channel
    strain_dataframe.data -= strain_dataframe.data[:, 0][:, np.newaxis]
    print("Applied baseline correction to each channel in the strain data.")
    print(np.min(strain_dataframe.data), np.max(strain_dataframe.data))

# Plot pressure data
fig, ax = plt.subplots(figsize=(10, 6))
pressure_dataframe.plot(ax=ax, use_timestamp=False, cmap="viridis", method='pcolormesh',
                        colorbar=True, clabel="Pressure (Pa)")
ax.set_title("Simulated Pressure")
plt.show()

# Plot to verify strain data
fig, ax = plt.subplots(figsize=(10, 6))
if strain_dataframe.data is not None and strain_dataframe.data.size > 0:
    max_abs_strain = np.max(np.abs(strain_dataframe.data))
    strain_dataframe.plot(ax=ax, use_timestamp=False, cmap="bwr", method='pcolormesh',
                          vmin=-max_abs_strain, vmax=max_abs_strain, colorbar=True, clabel="Strain")
else:
    strain_dataframe.plot(ax=ax, use_timestamp=False, cmap="bwr", method='pcolormesh',
                          colorbar=True, clabel="Strain")
ax.set_title("Simulated Strain")
plt.show()

# Plot original DSS data
fig, ax = plt.subplots()
DSSdata.plot(ax=ax, use_timestamp=False, cmap="bwr")
plt.show()

# Get the simulated strain and compare with DSS data. Let's select the center depth
chan_data = strain_dataframe.get_value_by_depth(0.5 * (strain_dataframe.daxis[0] + strain_dataframe.daxis[-1]))
plt.figure()
plt.plot(strain_dataframe.taxis, chan_data, label="Simulated Strain")
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.title("Simulated Strain at Center")
plt.legend()
plt.show()

# Get the simulated pressure at the center location
chan_data = pressure_dataframe.get_value_by_depth(0.5 * (pressure_dataframe.daxis[0] + pressure_dataframe.daxis[-1]))
plt.figure()
plt.plot(pressure_dataframe.taxis, chan_data, label="Simulated Pressure/Pa")
plt.xlabel("Time (s)")
plt.ylabel("Delta Pressure (Pa)")
plt.title("Simulated Pressure at Center")
plt.legend()
plt.show()
