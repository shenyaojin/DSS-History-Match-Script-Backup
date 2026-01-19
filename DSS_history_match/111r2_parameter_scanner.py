# The optimizer for the degradation version of HMM model
# Shenyao Jin, 01/16/2025

import numpy as np
import matplotlib.pyplot as plt

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.templates.baseline_model_generator import build_baseline_model
from fiberis.moose.templates.baseline_model_generator import post_processor_info_extractor
from fiberis.moose.runner import MooseRunner
import os