import os
import numpy as np
from DSS_analyzer_Mariner import Data2D_XT_DSS
from tqdm import tqdm

filepath_test = "data/legacy/s_well/DASdata/LFDASdata_stg7_swell.npz"
dataframe_DSS = Data2D_XT_DSS.Data2D()
dataframe_DSS.loadnpz(filepath_test)

print(dataframe_DSS.taxis[1])

# S well data
datafolder =  "data/legacy/s_well/DASdata/"
filelist = os.listdir(datafolder)

for file in tqdm(filelist):
    filepath = datafolder + file
    # Load the data
    dataframe_tmp = Data2D_XT_DSS.Data2D()
    dataframe_tmp.loadnpz(filepath)

    # Save the data using new packing format
    start_time = dataframe_tmp.start_time
    taxis = dataframe_tmp.taxis
    daxis = dataframe_tmp.daxis
    data = dataframe_tmp.data
    print("Timestamp:", start_time)

    # save the data
    destination = "data/fiberis_format/s_well/DAS/"
    dest_filepath = destination + file

    np.savez(dest_filepath, taxis=taxis, daxis=daxis, data=data, start_time=start_time)