# I'll test the baseline model builder function so that I can continue to
# Design the misfit functions (v2)
# At least the model works. In next step, I need to design the misfit function and wrapper
# Shenyao Jin, shenyaojin@mines.edu
import datetime

import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model, post_processor_info_extractor
from fiberis.moose.runner import MooseRunner
import os

# In this case, let's just render this baseline model to see if it works
project_name = "1124_misfit_func"
shift_list_ft = np.array([60.0, 80.0, 100.0, 120.0, 140.0, 160.0])
builder = build_baseline_model(project_name=project_name,
                               shift_list_ft=shift_list_ft,
                               angle=30,
                               )
print(f"Configured fiber line postprocessors at offsets ft: {shift_list_ft}")

# Output the model
output_dir = f"output/{project_name}"
os.makedirs(output_dir, exist_ok=True)
input_file_path = os.path.join(output_dir, f"{project_name}_input.i")

builder.generate_input_file(output_filepath=input_file_path)
#
# Run the model
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
pressure_dataframe, strain_xx_dataframe, strain_yy_dataframe, strain_xy_dataframe = post_processor_info_extractor(
    output_dir=output_dir)
pg_frame = Data1DGauge()
pg_frame.load_npz("data_fervo/fiberis_format/post_processing/synthetic_data_simulation.npz")

start_time = pg_frame.start_time
pressure_dataframe.start_time = start_time
strain_xx_dataframe.start_time = start_time
strain_yy_dataframe.start_time = start_time
strain_xy_dataframe.start_time = start_time

# Crop the data
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)

# %% Pre=process DSS data
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
DSSdata.select_depth(14880, 14900)  # <- Select depth range of interest.

sim_start = start_time
sim_end = start_time + datetime.timedelta(seconds=float(pressure_dataframe.taxis[-1]))
dss_start = DSSdata.start_time
dss_end = DSSdata.get_end_time()
overlap_start = max(sim_start, dss_start)
overlap_end = min(sim_end, dss_end)

print(f"Simulation time window: {sim_start} to {sim_end}")
print(f"DSS time window:        {dss_start} to {dss_end}")

if overlap_start <= overlap_end:
    print(f"Cropping simulated output to overlap window: {overlap_start} to {overlap_end}")
    pressure_dataframe.select_time(overlap_start, overlap_end)
    strain_xx_dataframe.select_time(overlap_start, overlap_end)
    strain_yy_dataframe.select_time(overlap_start, overlap_end)
    strain_xy_dataframe.select_time(overlap_start, overlap_end)
else:
    print("No absolute-time overlap between simulation and DSS data; keeping simulated output un-cropped.")

# Calibrate strain data by add first value of each channel
for strain_df in [strain_xx_dataframe, strain_yy_dataframe, strain_xy_dataframe]:
    if strain_df.data.shape[0] > 0 and strain_df.data.shape[1] > 0:
        # Subtract the first finite value of each channel (row) from that channel.
        finite_mask = np.isfinite(strain_df.data)
        valid_rows = finite_mask.any(axis=1)
        baseline = np.full(strain_df.data.shape[0], np.nan)
        first_finite_cols = np.argmax(finite_mask[valid_rows], axis=1)
        baseline[valid_rows] = strain_df.data[valid_rows, first_finite_cols]
        strain_df.data[valid_rows, :] -= baseline[valid_rows, np.newaxis]
        print(f"Applied baseline correction to each channel in {strain_df.name} data.")
        print(np.nanmin(strain_df.data), np.nanmax(strain_df.data))
    else:
        print(f"Skipping baseline correction for {strain_df.name}: empty data shape {strain_df.data.shape}.")

# Plot to verify
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, strain_df, title in zip(
        axes,
        [strain_xx_dataframe, strain_yy_dataframe, strain_xy_dataframe],
        ["Strain XX", "Strain YY", "Strain XY"]):
    artist = strain_df.plot(ax=ax, use_timestamp=False, cmap="bwr", method='pcolormesh', clim=(-4e-5, 4e-5))
    ax.set_title(title)
    if artist is not None:
        plt.colorbar(artist, ax=ax)
plt.show()

# Plot original DSS data
fig, ax = plt.subplots()
DSSdata.plot(ax=ax, use_timestamp=False, cmap="bwr")
plt.show()

# --- New Tensor Processing Section ---
from fiberis.io.reader_moose_tensor_from_data2d import MOOSETensorFromData2D

tensor_reader = MOOSETensorFromData2D()
if all(strain_df.data.shape[0] > 0 and strain_df.data.shape[1] > 0
       for strain_df in [strain_xx_dataframe, strain_yy_dataframe, strain_xy_dataframe]):
    tensor_reader.read(strain_xx_dataframe, strain_yy_dataframe, strain_xy_dataframe)
    tensor_list = tensor_reader.to_analyzer()

    # Get the tensor for the center channel
    center_channel_index = len(tensor_list) // 2
    center_tensor = tensor_list[center_channel_index]

    finite_time_mask = np.all(np.isfinite(center_tensor.data), axis=(0, 1))
    if not np.any(finite_time_mask):
        print("Skipping tensor analysis because the center tensor has no finite timesteps.")
    else:
        tensor_data = center_tensor.data[:, :, finite_time_mask]
        tensor_taxis = center_tensor.taxis[finite_time_mask]

        # Calculate principal strains and orientation
        principal_strains = np.zeros((2, len(tensor_taxis)))
        orientations = np.zeros(len(tensor_taxis))

        for i in range(len(tensor_taxis)):
            eigenvalues, eigenvectors = np.linalg.eig(tensor_data[:, :, i])
            principal_strains[:, i] = eigenvalues
            # Calculate orientation of the first principal eigenvector
            orientations[i] = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Plot principal strains and orientation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(tensor_taxis, principal_strains[0, :], label="Principal Strain 1")
        ax1.plot(tensor_taxis, principal_strains[1, :], label="Principal Strain 2")
        ax1.set_ylabel("Principal Strain")
        ax1.set_title(f"Principal Strains at Center Channel (daxis={center_tensor.name.split('=')[-1]})")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(tensor_taxis, orientations, label="Orientation of Principal Strain 1", color='g')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Orientation (degrees)")
        ax2.legend()
        ax2.grid(True)

        plt.suptitle("Tensor Analysis of Simulated Strain")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
else:
    print("Skipping tensor analysis because at least one strain component is empty.")
