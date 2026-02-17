from fiberis.moose.runner import MooseRunner

runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)

input_file_path = "scripts/DSS_history_match/optimizer_input_file_test/perm_3layer/forward_gt.i"
output_dir="scripts/DSS_history_match/optimizer_input_file_test/perm_3layer"

# success, stdout, stderr = runner.run(
#     input_file_path=input_file_path,
#     output_directory=output_dir,
#     num_processors=20,
#     log_file_name="simulation.log",
#     stream_output=True,
#     clean_output_dir=False
# )

from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

reader = MOOSEVectorPostProcessorReader()
reader.read(directory=output_dir)

pressure_dataframe = reader.to_analyzer()

pressure_dataframe.daxis += 30
pressure_dataframe.data[:, 0] = 0

print(pressure_dataframe.to_moose_reporter_str(coord_x=22, coord_z=0, precision=2))

import matplotlib.pyplot as plt
