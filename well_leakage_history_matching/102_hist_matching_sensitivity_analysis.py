# To analyze the sensitivity of diffusivity change in **S** well.

#%% Load the data.
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D import Data2D_XT_DSS

# Load the data
datapath = 'output/0211_simulation_MULTIstage/'
# Load the phase 1 data
phase1_datapath = datapath + "phase1.npz"
phase2_datapath = datapath + "phase2.npz"

phase3_filest = ["phase3_0.1.npz", "phase3_0.01.npz", "phase3_0.001.npz", "phase3_0.0001.npz", "phase3_1e-05.npz", "phase3_test.npz"]
phase3_datapath = [datapath + f for f in phase3_filest]



# Load the data
phase1_pf_dataframe = Data2D_XT_DSS.DSS2D()
phase1_pf_dataframe.load_npz(phase1_datapath)

phase2_pf_dataframe = Data2D_XT_DSS.DSS2D()
phase2_pf_dataframe.load_npz(phase2_datapath)

phase1_pf_dataframe.data = phase1_pf_dataframe.data.T
phase2_pf_dataframe.data = phase2_pf_dataframe.data.T

# Merge the data. Sicne the phase 2 data includes the initial snapshot
# I need to remove some part of the data.
phase2_pf_dataframe.select_time(30, phase2_pf_dataframe.get_end_time())
print(phase2_pf_dataframe.start_time)

#%% Concatenate the data
phase12_pf_dataframe = phase1_pf_dataframe.copy()
phase12_pf_dataframe.right_merge(phase2_pf_dataframe)

# Load the phase 3 data.
phase3_pf_dataframe_all = []
