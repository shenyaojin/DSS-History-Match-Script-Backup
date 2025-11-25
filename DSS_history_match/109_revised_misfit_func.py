# A revised misfit function for history matching in DSS
# Was trying to implement a misfit function.
# A <refactor> version of 108_misfit_func.py
# I hope this version will separate the model builder and original model

# Shenyao Jin, shenyaojin@mines.edu

#%% Import libs
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import csv

# I/O modules from fiberis
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

# MOOSE simulator
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)

# Optimization module
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

#%% Define super parameters
project_name = "1114_revised_misfit_func"

#%% Set up file paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
pressure_profile_full_path = "data/fiberis_format/post_processing/history_matching_pressure_profile_full.npz"

#%% Load DSS data
DSSdata = DSS2D()
DSSdata.load_npz(DSS_datapath)
# Load gauge data
pg_full_dataframe = Data1DGauge() # The full pressure gauge data profile contains all information we need
pg_full_dataframe.load_npz(pressure_profile_full_path)

#%% Pre-process data
# DSS
mds = DSSdata.daxis
ind = (mds > 7500) & (mds < 15000)
drift_val = np.median(DSSdata.data[ind, :], axis=0)
DSSdata.data -= drift_val.reshape((1, -1))
DSSdata.select_time(0, 400000)
# DSSdata.select_depth(14820, 14920) # <- Select depth range of interest.
DSSdata.select_depth(14885, 14894)
# Might need to change based on well location