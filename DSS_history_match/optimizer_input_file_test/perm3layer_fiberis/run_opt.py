
from fiberis.moose.runner import MooseRunner

if __name__ == '__main__':
    runner = MooseRunner(
        moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
        mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
    )

    input_file_path = "scripts/DSS_history_match/optimizer_input_file_test/perm3layer_fiberis/optimize.i"
    output_directory = "scripts/DSS_history_match/optimizer_input_file_test/perm3layer_fiberis"

    success, stdout, stderr = runner.run(
        input_file_path=input_file_path,
        output_directory=output_directory,
        num_processors=18,
        log_file_name="simulation.log",
        stream_output=True,
        clean_output_dir=False
    )
    