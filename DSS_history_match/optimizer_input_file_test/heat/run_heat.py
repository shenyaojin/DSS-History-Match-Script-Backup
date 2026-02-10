from fiberis.moose.runner import MooseRunner

runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/optimization/optimization-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)

input_file_path = "scripts/DSS_history_match/optimizer_input_file_test/heat/main.i"
output_dir="scripts/DSS_history_match/optimizer_input_file_test/heat/"

success, stdout, stderr = runner.run(
    input_file_path=input_file_path,
    output_directory=output_dir,
    num_processors=20,
    log_file_name="simulation.log",
    stream_output=True,
    clean_output_dir=False
)