#%% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

#%% Define filepath
filepath = "data/fiberis_format/prod/dss_data/Mariner 14x-36-D - RFS strain change.npz"
# Define dataframe
DSS_dataframe = DSS2D()
DSS_dataframe.load_npz(filepath)

#%% Print info of DSS
print(DSS_dataframe)

#%% Pre-processing. Remove drift noise?