import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# prefix for python path
moose_python_path = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/python'
sys.path.append(moose_python_path)

from fiberis.moose.runner import MooseRunner
from fiberis.utils.viz_utils import plot_point_samplers, plot_vector_samplers
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
import datetime

#%% --- Configuration ---
# Define file paths for the simulation.

base_input_file = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/saved_files/example_VF.i'
# Define output directory for the raw simulation files.
output_dir_run = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/single_run_output'
# Define a separate directory for visualizations.
viz_output_dir = '/scripts/fiberis_moose_generator/repeat_vf/viz_results/single_run_viz'
# Define output for NPZ files
npz_output_dir = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/saved_npz/with_kernel'

moose_executable = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt'

vpr = MOOSEVectorPostProcessorReader()
max_ind = vpr.get_max_indices("scripts/fiberis_moose_generator/repeat_vf/single_run_output")
print(max_ind)

for processor_id in range(1, max_ind[0]+ 1):
    for var_id in range(1, max_ind[1]+ 1):
        vpr.read("scripts/fiberis_moose_generator/repeat_vf/single_run_output", processor_id, var_id)
        vpr.write(os.path.join("scripts/fiberis_moose_generator/repeat_vf/saved_npz/with_kernel",
                               f"{vpr.sampler_name}_{vpr.variable_name}"))