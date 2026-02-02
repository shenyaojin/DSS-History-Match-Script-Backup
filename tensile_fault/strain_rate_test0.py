# To test out the new format of .i file for strain rate calculation
# Shenyao Jin, 01/27/2026

from fiberis.moose.runner import MooseRunner
import os

print(os.getcwd())

runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)
# #
success, stdout, stderr = runner.run(
    input_file_path="scripts/tensile_fault/test_folder/strain_rate/1203_rotated_monitor_well_input.i",
    output_directory="scripts/tensile_fault/test_folder/strain_rate",
    num_processors=20,
    log_file_name="simulation.log",
    stream_output=True
)


