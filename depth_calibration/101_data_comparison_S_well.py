# two parts:
# 1. Compare *.dat files with *.h5 files
# 2. Compare DSS data with DAS data
# Shenyao Jin, 09-10-2025

#%% Import packages
import os
import numpy as np
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.io.reader_mariner_fiberdata_production_dat2d import MarinerDSSdat2D

# Load h5 source data
filepath_dss_h5 = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
filepath_dss_dat = "data/legacy/DSSdata_dat/POW-S_Strain"

#%% Part 1: Compare *.dat files with *.h5 files
# Load h5 data
dss_h5 = DSS2D()
dss_h5.load_npz(filepath_dss_h5)

# Load dat data
dss_dat = DSS2D()
dss_dat_io = MarinerDSSdat2D()
dss_dat_io.read(filepath_dss_dat)
dss_dat = dss_dat_io.to_analyzer()

#%% Compare data
print(np.allclose(dss_dat.daxis, dss_h5.daxis))  # False
print(np.allclose(dss_dat.taxis, dss_h5.taxis))  # True
print(np.allclose(dss_dat.data, dss_h5.data))    # False

#%% Check where the differences are
print(np.max(np.abs(dss_dat.daxis - dss_h5.daxis)))
print(np.max(np.abs(dss_dat.data - dss_h5.data)))

#%% Load DAS data
filepath_das = "data/fiberis_format/s_well/DAS/LFDASdata_stg7_swell.npz"
DAS_dataframe = DSS2D()
DAS_dataframe.load_npz(filepath_das)

#%% Part 2: Compare DSS data with DAS data
# Examine the EOF positions

DAS_dataframe_eof = DAS_dataframe.copy()
DSS_dataframe_eof = dss_h5.copy()

DAS_dataframe_eof.select_depth(16700, 16800)
DSS_dataframe_eof.select_depth(16750, 16850)

#%% Plot part of the data to see the differences
cx = np.array([-1, 1])
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
im1 = DAS_dataframe_eof.plot(ax=axs[0], cmap='bwr')
im1.set_clim(cx * 1e3)
# set title
axs[0].set_title('DAS data', fontsize=14)

im2 = DSS_dataframe_eof.plot(ax=axs[1], cmap='bwr')
im2.set_clim(cx * 2)
axs[1].set_title('DSS data', fontsize=14)

plt.suptitle('DAS and DSS data, EOF plot', fontsize=16)
plt.tight_layout()
plt.show()
