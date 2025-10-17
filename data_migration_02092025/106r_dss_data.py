# Move all needed DSS data to npz files for further processing
# %% Import all needed packages
import numpy as np
import matplotlib.pyplot as plt
import os
from fiberis.io.reader_mariner_dssh5 import MarinerDSS2D
import glob

# %% Define the input and output directories
input_dir = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/data/legacy/DSSdata"
output_dir = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/data/fiberis_format/dss"

# %% Get a list of all .h5 files in the input directory
file_list = glob.glob(os.path.join(input_dir, "*.h5"))

# %% Loop through each file
for file_path in file_list:
    # Create a MarinerDSS2D object
    reader = MarinerDSS2D()

    # Read the .h5 file
    reader.read(file_path)

    # Construct the output file path
    file_name = os.path.basename(file_path)
    output_file_name = os.path.splitext(file_name)[0] + ".npz"
    output_file_path = os.path.join(output_dir, output_file_name)

    # Write the data to a .npz file
    reader.write(output_file_path)
    print(f"Converted {file_path} to {output_file_path}")

#%% Test read for the packed files
filepath = "data/fiberis_format/h_well/dss_data/Mariner 14x-36-POW-H - RFS strain change.npz"
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

dss_dataframe = DSS2D()
dss_dataframe.load_npz(filepath)

#%% Plot info
dss_dataframe.print_info()