# This script will implement the discussion on misfit functions and other data processing techniques.
# See my notion. -- Shenyao

# - linear scaling for DSS data. Because DSS is not perfectly coupled with the casing. Let's do 10x
# - remove one SRV (SRV2)
# - change the misfit func, to a manually picked area.
# - sensitivity  test on length

# This script, still will only use 1 fracture.

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import datetime
import csv

from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)

# Optimization modules
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

#%% Set up paths
# Source data paths
DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
mpiexec_path = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"

# Simulation and logging output paths
project_name = "1109_DSS_SingleFrac_MisfitFunc"
base_output_dir = os.path.join("output", project_name)
moose_output_dir = os.path.join(base_output_dir, "moose_simulations")
log_dir = os.path.join(base_output_dir, "optimization_logs")
fig_parent_dir = os.path.join("figs", project_name)

os.makedirs(moose_output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(fig_parent_dir, exist_ok=True)

