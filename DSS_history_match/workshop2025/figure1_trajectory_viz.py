import numpy as np
import matplotlib.pyplot as plt

from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.templates.baseline_model_generator import build_baseline_model

builder = build_baseline_model()
builder.plot_geometry()