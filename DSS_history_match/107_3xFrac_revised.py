#%% In this script, I will implement a revised version of the 3xFrac history matching
# Modified from 106r_DSS_3xSingleFrac_parameter_scanner.
# Shenyao Jin, 10/27/2025
# Changelog:
# 1. Because I implemented the model_builder editor, the function will be cleaner.
# 2. This script will only focus on single fracture
# 3. Revised pressure profile to make it more realistic
# 4. Add constraint: perm_frac > perm_SRV > perm_matrix

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import datetime

# I/O modules from fiberis
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader
from fiberis.utils.viz_utils import plot_point_samplers, plot_vector_samplers

# Simulation modules from fiberis
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)

# Optimization modules
from skopt import gp_minimize
from skopt.space import Real

#%% Set up paths
# Source data paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"

# Simulation and logging output paths
project_name = "107_DSS_SingleFrac_scanner"
base_output_dir = os.path.join("output", project_name)
moose_output_dir = os.path.join(base_output_dir, "moose_simulations")
log_dir = os.path.join(base_output_dir, "optimization_logs")
fig_parent_dir = os.path.join("figs", project_name)

os.makedirs(moose_output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(fig_parent_dir, exist_ok=True)

#%% 1. Load and preprocess DSS data
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
DSSdata.select_depth(12000, 16360)

DSSdata_copy = DSSdata.copy()
DSSdata_copy.select_depth(14500, 15500)

#%% 2. Load and preprocess gauge data
gauge_data_interference = Data1DGauge()
gauge_data_interference.load_npz(pressure_gauge_g1_path)
gauge_data_interference.select_time(DSSdata.start_time, DSSdata.get_end_time())

injection_gauge_pressure = Data1DGauge()
injection_gauge_pressure.load_npz(injection_gauge_pressure_path)
injection_gauge_pressure.select_time(injection_gauge_pressure.start_time, DSSdata.start_time)
injection_gauge_pressure.remove_abnormal_data(threshold=300, method='mean')

injection_gauge_pressure_copy = injection_gauge_pressure.copy()
gauge_data_interference_copy = gauge_data_interference.copy()
injection_gauge_pressure_copy.adaptive_downsample(300)
gauge_data_interference_copy.adaptive_downsample(600)

# Shift the interference gauge data to align with DSS data (one is wellhead, the other is downhole)
difference_val = injection_gauge_pressure.data[-1] - gauge_data_interference.data[0]
gauge_data_interference_copy.data += difference_val

injection_gauge_pressure_copy.right_merge(gauge_data_interference_copy)
injection_gauge_pressure_copy.rename("injection pressure full profile")

## Plot for checking
# fig, ax = plt.subplots(figsize=(8, 5))
# injection_gauge_pressure_copy.plot(ax=ax, use_timestamp= True)
# ax.axvline(gauge_data_interference_copy.start_time, color='red', linestyle='--', label='Interference Start')
# ax.set_title("Injection Pressure Full Profile")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Pressure (psi)")
# ax.legend()
# plt.show()
#
# # Crop the zoom-in data for better visualization
# injection_gauge_pressure_zoom_in = injection_gauge_pressure_copy.copy()
# injection_gauge_pressure_zoom_in.select_time(gauge_data_interference_copy.start_time - datetime.timedelta(days=3),
#                                         gauge_data_interference_copy.get_end_time(use_timestamp=True))
# fig, ax = plt.subplots(figsize=(8, 5))
# injection_gauge_pressure_zoom_in.plot(ax=ax, use_timestamp=True)
# ax.axvline(gauge_data_interference_copy.start_time, color='red', linestyle='--', label='Interference Start')
# ax.set_title("Injection Pressure Zoom-in Profile")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Pressure (psi)")
# ax.legend()
# plt.show()