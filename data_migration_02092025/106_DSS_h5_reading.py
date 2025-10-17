#%% Test read for Mariner DSS file.
# Shenyao Jin, shenyaojin@mines.edu

import numpy as np
import matplotlib.pyplot as plt
from fiberis.utils import io_utils

#%% file path
data_path = "data/legacy/DSSdata/Mariner 14x-36-POW-H - RFS strain change rate.h5"

# #%% Read data
# raw_data, raw_depth, raw_timestamp, start_time= io_utils.read_h5(data_path)
#
# #%% Print raw data (some)
# print(("Raw data sample: ", raw_data[:5, :5]))
# print(("Raw depth sample: ", raw_depth[:5]))
#
# print(raw_timestamp[:2])
# print(start_time)

# #%% Convert timestamp to seconds
# import datetime
# time_stamp_test = datetime.datetime.fromtimestamp(raw_timestamp[0]/1e6)
# print(time_stamp_test)
# # Add 8 hours, considering the notes in h5 file! For the start time.

#%% Test the library function to read the DSS file
from fiberis.io.reader_mariner_dssh5 import MarinerDSS2D

reader = MarinerDSS2D()
reader.read(data_path)

#%% Print raw data (some)
reader.start_time