from fiberis.moose.runner import MooseRunner

runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)

input_file_path = "scripts/DSS_history_match/optimizer_input_file_test/perm3layer_v2/forward_gt.i"
output_dir="scripts/DSS_history_match/optimizer_input_file_test/perm3layer_v2"

success, stdout, stderr = runner.run(
    input_file_path=input_file_path,
    output_directory=output_dir,
    num_processors=20,
    log_file_name="simulation.log",
    stream_output=True,
    clean_output_dir=False
)

from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

reader = MOOSEVectorPostProcessorReader()
reader.read(directory=output_dir, variable_index=0)

disp_y_dataframe = reader.to_analyzer()

disp_y_dataframe.daxis += 30 # Make it same as simulation. 
disp_y_dataframe.data[:, 0] = disp_y_dataframe.data[:, 1] # padding

import pandas as pd
import numpy as np

# Flatten the Data2D object for MOOSE inversion CSV format
# Data shape is (n_depth, n_time)
n_depth, n_time = disp_y_dataframe.data.shape

# Repeat time for each depth point
times = np.repeat(disp_y_dataframe.taxis, n_depth)
# Tile depth (y-coord) for each time step
y_coords = np.tile(disp_y_dataframe.daxis, n_time)
# Fixed coordinates
x_coords = np.full_like(times, 22.0)
z_coords = np.zeros_like(times)

# Flatten data in column-major (Fortran) order to align with repeated/tiled axes
values = disp_y_dataframe.data.flatten(order='F')

df = pd.DataFrame({
    'measurement_time': times,
    'measurement_values': values,
    'measurement_xcoord': x_coords,
    'measurement_ycoord': y_coords,
    'measurement_zcoord': z_coords,
    'misfit_values': values,       # Ground truth: misfit and simulation set to measurement
    'simulation_values': values
})

output_csv = "scripts/DSS_history_match/optimizer_input_file_test/perm3layer_v2/inv/measurement_data.csv"
df.to_csv(output_csv, index=False)
print(f"Inversion CSV saved to: {output_csv}")
