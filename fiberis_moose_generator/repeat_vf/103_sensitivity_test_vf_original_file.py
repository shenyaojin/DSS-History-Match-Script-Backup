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
input_file_typical_diffusion = 'scripts/fiberis_moose_generator/repeat_vf/saved_files/example_VF_single_kernel_typical_diffusion.i'
input_file_porous_flow = 'scripts/fiberis_moose_generator/repeat_vf/saved_files/example_VF_single_kernel_porous_flow.i'

# Define output directory for the raw simulation files.
output_dir_run = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/single_run_output'
output_dir_run_typical_diffusion = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/single_run_output_typical_diffusion'
output_dir_run_porous_flow = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/single_run_output_porous_flow'

# Define a separate directory for visualizations.
viz_output_dir = 'scripts/fiberis_moose_generator/repeat_vf/viz_results/single_run_viz'
viz_output_dir_typical_diffusion = 'scripts/fiberis_moose_generator/repeat_vf/viz_results/single_run_viz_typical_diffusion'
viz_output_dir_porous_flow = 'scripts/fiberis_moose_generator/repeat_vf/viz_results/single_run_viz_porous_flow'


# Define output for NPZ files
npz_output_dir = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/saved_npz/with_kernel'
npz_output_dir_typical_diffusion = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/saved_npz/single_diffusion'
npz_output_dir_porous_flow = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/fiberis_moose_generator/repeat_vf/saved_npz/porous_kernel'

moose_executable = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt'

# Check for MOOSE executable
if not os.path.exists(moose_executable):
    raise FileNotFoundError(f"ERROR: MOOSE executable not found at '{moose_executable}'. Please update the path.")

#%% --- Run Simulation ---
# Run the MOOSE simulation on the specified input file.

print(f"--- Running Simulation for {base_input_file} ---")
os.makedirs(output_dir_run, exist_ok=True)

print("Starting MOOSE simulation...")
runner = MooseRunner(moose_executable_path=moose_executable)
success, _, _ = runner.run(
    input_file_path=base_input_file,
    output_directory=output_dir_run,
    num_processors=2,
    log_file_name="simulation.log"
)

if not success:
    print(f"ERROR: Simulation failed.")
    exit()

print("Simulation completed successfully.")

#%% --- Generate Detailed Plots Using viz_utils ---
print("\n--- Generating detailed plots for the run ---")
os.makedirs(viz_output_dir, exist_ok=True)

# --- Point Sampler Plots ---
point_sampler_plot_dir = os.path.join(viz_output_dir, 'point_sampler_plots')
print(f"Saving point sampler plots to: {point_sampler_plot_dir}")
try:
    plot_point_samplers(folder=output_dir_run, output_dir=point_sampler_plot_dir)
except Exception as e:
    print(f"Could not generate point sampler plots: {e}")

# --- Vector Sampler Plots ---
vector_sampler_plot_dir = os.path.join(viz_output_dir, 'vector_sampler_plots')
print(f"Saving vector sampler plots to: {vector_sampler_plot_dir}")
try:
    plot_vector_samplers(folder=output_dir_run, output_dir=vector_sampler_plot_dir)
except Exception as e:
    print(f"Could not generate vector sampler plots: {e}")

#%% --- Save Postprocessors to NPZ Format ---
psr = MOOSEPointSamplerReader()
max_ind = psr.get_max_index("scripts/fiberis_moose_generator/repeat_vf/single_run_output")
for ind in range(1, max_ind + 1):
    psr.read("scripts/fiberis_moose_generator/repeat_vf/single_run_output", ind)
    psr.write(os.path.join("scripts/fiberis_moose_generator/repeat_vf/saved_npz/with_kernel", f"{psr.variable_name}"))

#%% --- Save Vector Postprocessors to NPZ Format ---
vpr = MOOSEVectorPostProcessorReader()
max_ind = vpr.get_max_indices("scripts/fiberis_moose_generator/repeat_vf/single_run_output")
print(max_ind)

for processor_id in range(1, max_ind[0]+ 1):
    for var_id in range(1, max_ind[1]+ 1):
        vpr.read("scripts/fiberis_moose_generator/repeat_vf/single_run_output", processor_id, var_id)
        vpr.write(os.path.join("scripts/fiberis_moose_generator/repeat_vf/saved_npz/with_kernel",
                               f"{vpr.sampler_name}_{vpr.variable_name}"))

        vpr.log_system(f"Saved VPP NPZ: {vpr.sampler_name}_{vpr.variable_name}")

#%% --- Repeat for Typical Diffusion Kernel ---
print(f"\n--- Running Simulation for {input_file_typical_diffusion} ---")
os.makedirs(output_dir_run_typical_diffusion, exist_ok=True)
print("Starting MOOSE simulation for typical diffusion kernel...")
runner = MooseRunner(moose_executable_path=moose_executable)
success, _, _ = runner.run(
    input_file_path=input_file_typical_diffusion,
    output_directory=output_dir_run_typical_diffusion,
    num_processors=2,
    log_file_name="simulation_typical_diffusion.log"
)
if not success:
    print(f"ERROR: Simulation for typical diffusion kernel failed.")
    exit()
print("Simulation for typical diffusion kernel completed successfully.")

#%% --- Repeat for Porous Flow Kernel ---
print(f"\n--- Running Simulation for {input_file_porous_flow} ---")
os.makedirs(output_dir_run_porous_flow, exist_ok=True)
print("Starting MOOSE simulation for porous flow kernel...")
runner = MooseRunner(moose_executable_path=moose_executable)
success, _, _ = runner.run(
    input_file_path=input_file_porous_flow,
    output_directory=output_dir_run_porous_flow,
    num_processors=2,
    log_file_name="simulation_porous_flow.log"
)
if not success:
    print(f"ERROR: Simulation for porous flow kernel failed.")
    exit()
print("Simulation for porous flow kernel completed successfully.")

#%% --- Viz for Typical Diffusion Kernel ---
print("\n--- Generating detailed plots for the typical diffusion kernel run ---")
os.makedirs(viz_output_dir_typical_diffusion, exist_ok=True)
# --- Point Sampler Plots ---
point_sampler_plot_dir_typical = os.path.join(viz_output_dir_typical_diffusion,
                                                'point_sampler_plots')
print(f"Saving point sampler plots to: {point_sampler_plot_dir_typical}")
try:
    plot_point_samplers(folder=output_dir_run_typical_diffusion,
                        output_dir=point_sampler_plot_dir_typical)
except Exception as e:
    print(f"Could not generate point sampler plots: {e}")
# --- Vector Sampler Plots ---
vector_sampler_plot_dir_typical = os.path.join(viz_output_dir_typical_diffusion,
                                                    'vector_sampler_plots')
print(f"Saving vector sampler plots to: {vector_sampler_plot_dir_typical}")
try:
    plot_vector_samplers(folder=output_dir_run_typical_diffusion,
                         output_dir=vector_sampler_plot_dir_typical)
except Exception as e:
    print(f"Could not generate vector sampler plots: {e}")
#%% --- Viz for Porous Flow Kernel ---
print("\n--- Generating detailed plots for the porous flow kernel run ---")
os.makedirs(viz_output_dir_porous_flow, exist_ok=True)
# --- Point Sampler Plots ---
point_sampler_plot_dir_porous = os.path.join(viz_output_dir_porous_flow,
                                                'point_sampler_plots')
print(f"Saving point sampler plots to: {point_sampler_plot_dir_porous}")
try:
    plot_point_samplers(folder=output_dir_run_porous_flow,
                        output_dir=point_sampler_plot_dir_porous)
except Exception as e:
    print(f"Could not generate point sampler plots: {e}")
# --- Vector Sampler Plots ---
vector_sampler_plot_dir_porous = os.path.join(viz_output_dir_porous_flow,
                                                    'vector_sampler_plots')
print(f"Saving vector sampler plots to: {vector_sampler_plot_dir_porous}")
try:
    plot_vector_samplers(folder=output_dir_run_porous_flow,
                         output_dir=vector_sampler_plot_dir_porous)
except Exception as e:
    print(f"Could not generate vector sampler plots: {e}")
#%% --- Save NPZ for Typical Diffusion Kernel ---
psr = MOOSEPointSamplerReader()
max_ind = psr.get_max_index("scripts/fiberis_moose_generator/repeat_vf/single_run_output_typical_diffusion")
for ind in range(1, max_ind + 1):
    psr.read("scripts/fiberis_moose_generator/repeat_vf/single_run_output_typical_diffusion", ind)
    psr.write(os.path.join("scripts/fiberis_moose_generator/repeat_vf/saved_npz/single_diffusion",
                           f"{psr.variable_name}"))

vpr = MOOSEVectorPostProcessorReader()
max_ind = vpr.get_max_indices("scripts/fiberis_moose_generator/repeat_vf/single_run_output_typical_diffusion")
print(max_ind)
for processor_id in range(1, max_ind[0]+ 1):
    for var_id in range(1, max_ind[1]+ 1):
        vpr.read("scripts/fiberis_moose_generator/repeat_vf/single_run_output_typical_diffusion",
                 processor_id, var_id)
        vpr.write(os.path.join("scripts/fiberis_moose_generator/repeat_vf/saved_npz/single_diffusion",
                               f"{vpr.sampler_name}_{vpr.variable_name}"))

#%% --- Save NPZ for Porous Flow Kernel ---
psr = MOOSEPointSamplerReader()
max_ind = psr.get_max_index("scripts/fiberis_moose_generator/repeat_vf/single_run_output_porous_flow")
for ind in range(1, max_ind + 1):
    psr.read("scripts/fiberis_moose_generator/repeat_vf/single_run_output_porous_flow", ind)
    psr.write(os.path.join("scripts/fiberis_moose_generator/repeat_vf/saved_npz/porous_kernel",
                           f"{psr.variable_name}"))
vpr = MOOSEVectorPostProcessorReader()
max_ind = vpr.get_max_indices("scripts/fiberis_moose_generator/repeat_vf/single_run_output_porous_flow")
print(max_ind)
for processor_id in range(1, max_ind[0]+ 1):
    for var_id in range(1, max_ind[1]+ 1):
        vpr.read("scripts/fiberis_moose_generator/repeat_vf/single_run_output_porous_flow",
                 processor_id, var_id)
        vpr.write(os.path.join("scripts/fiberis_moose_generator/repeat_vf/saved_npz/porous_kernel",
                               f"{vpr.sampler_name}_{vpr.variable_name}"))

        vpr.log_system.add_record(f"Saved VPP NPZ: {vpr.sampler_name}_{vpr.variable_name}")

#%% --- Finished ---
print("\nAll simulations, visualizations, and NPZ savings completed successfully.")