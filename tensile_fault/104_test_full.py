# I'll test the baseline model builder function with a rotated monitor well
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
project_name = "1203_rotated_monitor_well"
hf_length_ft = 250
builder = build_baseline_model(project_name=project_name,
                               hf_length_ft= hf_length_ft,
                               srv_height_ft=0.25,
                               shift_list_ft=np.array([0.8 * hf_length_ft/2, 0.9 * hf_length_ft/2, 1.1 * hf_length_ft/2, 1.3 * hf_length_ft/2]),
                               angle=30,
                               srv_perm=1e-15,
                               fracture_perm=1e-13
                               )

# Output the model
output_dir = f"output/{project_name}"
builder.plot_geometry(hide_legend=True, equal_aspect=True)
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