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
from fiberis.moose.templates.baseline_model_generator import build_baseline_model, post_processor_info_extractor
from fiberis.moose.runner import MooseRunner
import os

print("working directory:", os.getcwd())

# In this case, let's just render this baseline model to see if it works
project_name = "1124_misfit_func"
builder = build_baseline_model(project_name=project_name,
                               )

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
pressure_dataframe, strain_dataframe = post_processor_info_extractor(output_dir=output_dir)
pg_frame = Data1DGauge()
pg_frame.load_npz("data/fiberis_format/post_processing/timestepper_profile.npz")

start_time = pg_frame.start_time
pressure_dataframe.start_time = start_time
strain_dataframe.start_time = start_time

# Crop the data
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)

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

# Plot to verify
fig, ax = plt.subplots()
strain_dataframe.plot(ax=ax, use_timestamp=False, cmap="bwr", method='pcolormesh', clim=(-4e-5, 4e-5))
# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax)
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
plt.title("Simulated Strain at Center Depth")
plt.legend()
plt.show()

# Next step, to test the misfit function, we need the ind number of fracture center in DSS data and simulated data
