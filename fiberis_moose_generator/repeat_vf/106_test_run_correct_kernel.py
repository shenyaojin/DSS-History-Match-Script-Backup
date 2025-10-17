import os
import sys
from fiberis.moose.runner import MooseRunner
from fiberis.utils.viz_utils import plot_point_samplers, plot_vector_samplers
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

# This script runs two MOOSE simulations for the "improved kernel" scenarios,
# generates visualization plots, and saves the results in NPZ format.
# It is based on the workflow from 103_sensitivity_test_vf_original_file.py.

# --- Configuration ---
moose_python_path = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/python'
if moose_python_path not in sys.path:
    sys.path.append(moose_python_path)

# Define input files for the two scenarios
input_files = {
    "improved_kernel": 'scripts/fiberis_moose_generator/repeat_vf/saved_files/example_VF_improved_kernel.i',
    "improved_kernel_porou_DT": 'scripts/fiberis_moose_generator/repeat_vf/saved_files/example_VF_improved_kernel_porou_DT.i'
}

# Define output directories
base_output_path = 'scripts/fiberis_moose_generator/repeat_vf'
run_output_base = os.path.join(base_output_path, 'run_output')
viz_output_base = os.path.join(base_output_path, 'viz_results')
npz_output_base = os.path.join(base_output_path, 'saved_npz')

# MOOSE executable path
moose_executable = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt'

if not os.path.exists(moose_executable):
    raise FileNotFoundError(f"ERROR: MOOSE executable not found at '{moose_executable}'. Please update the path.")

def run_and_process_scenario(scenario_name, input_file):
    """
    Runs a full simulation and post-processing for a given scenario.
    """
    print(f"--- Processing Scenario: {scenario_name} ---")

    # Define scenario-specific paths
    output_dir_run = os.path.join(run_output_base, f"106_{scenario_name}")
    viz_output_dir = os.path.join(viz_output_base, f"106_{scenario_name}")
    npz_output_dir = os.path.join(npz_output_base, f"106_{scenario_name}")

    # Create directories
    os.makedirs(output_dir_run, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)
    os.makedirs(npz_output_dir, exist_ok=True)

    # --- 1. Run Simulation ---
    print(f"Starting MOOSE simulation for {scenario_name}...")
    runner = MooseRunner(moose_executable_path=moose_executable)
    success, _, _ = runner.run(
        input_file_path=input_file,
        output_directory=output_dir_run,
        num_processors=4,  # Using 4 processors as an example
        log_file_name=f"simulation_{scenario_name}.log"
    )
    if not success:
        print(f"ERROR: Simulation failed for {scenario_name}.")
        return

    print("Simulation completed successfully.")

    # --- 2. Generate Detailed Plots ---
    print(f"\nGenerating detailed plots for {scenario_name}...")
    try:
        plot_point_samplers(folder=output_dir_run, output_dir=os.path.join(viz_output_dir, 'point_sampler_plots'))
        plot_vector_samplers(folder=output_dir_run, output_dir=os.path.join(viz_output_dir, 'vector_sampler_plots'))
        print(f"Visualization plots saved in {viz_output_dir}")
    except Exception as e:
        print(f"Could not generate plots: {e}")

    # --- 3. Save Postprocessors to NPZ ---
    print(f"\nSaving postprocessors to NPZ format for {scenario_name}...")
    # Point samplers
    try:
        psr = MOOSEPointSamplerReader()
        max_ind = psr.get_max_index(output_dir_run)
        for ind in range(1, max_ind + 1):
            psr.read(output_dir_run, ind)
            psr.write(os.path.join(npz_output_dir, f"{psr.variable_name}"))
        print("Point samplers saved to NPZ.")
    except Exception as e:
        print(f"Could not save point samplers to NPZ: {e}")

    # Vector postprocessors
    try:
        vpr = MOOSEVectorPostProcessorReader()
        max_ind_vpp = vpr.get_max_indices(output_dir_run)
        if max_ind_vpp and len(max_ind_vpp) > 1:
            for p_id in range(1, max_ind_vpp[0] + 1):
                for v_id in range(1, max_ind_vpp[1] + 1):
                    vpr.read(output_dir_run, p_id, v_id)
                    vpr.write(os.path.join(npz_output_dir, f"{vpr.sampler_name}_{vpr.variable_name}"))
            print("Vector postprocessors saved to NPZ.")
    except Exception as e:
        print(f"Could not save vector postprocessors to NPZ: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    for name, file_path in input_files.items():
        run_and_process_scenario(name, file_path)
        print('-' * 50)
    print("\nAll scenarios processed successfully.")