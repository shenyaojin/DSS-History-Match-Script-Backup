#%% Import libs
from fiberis.io import reader_mariner_pressureg1
import numpy as np

filepath = "data/legacy/prod/pressure_g1.csv"

#%% Load data
data_io = reader_mariner_pressureg1.MarinerPressureG1()
data_io.read(filepath)
data_io.write("data/fiberis_format/prod/gauges/pressure_g1.npz")

#%% Read data using data1d
from fiberis.analyzer.Data1D import core1D
dataframe = core1D.Data1D()
dataframe.load_npz("data/fiberis_format/prod/gauges/pressure_g1.npz")

#%% show data
dataframe.start_time