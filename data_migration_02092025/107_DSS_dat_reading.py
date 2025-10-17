#%% Test read for Mariner DSS *.dat data
# Shenyao Jin, shenyaojin@mines.edu

import numpy as np
import matplotlib.pyplot as plt
from fiberis.io.reader_mariner_fiberdata_production_dat2d import MarinerDSSdat2D

reader = MarinerDSSdat2D()
base_file = "data/legacy/DSSdata_dat/POW-S_Strain"

base_file_dat = base_file + "_data.dat"
#%% read data
reader.read(base_file)
# Read first line of basefile
# line1 = np.loadtxt(base_file_dat, max_rows=1)
# line2 = np.loadtxt(base_file_dat, max_rows=1, skiprows=1)
#
# print(np.shape(line1))
# print(np.shape(line2))

#%% Take a look at the data
dataframe = reader.to_analyzer()

#%% plot info
print(dataframe)