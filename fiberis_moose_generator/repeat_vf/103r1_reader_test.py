#%% To test the reading data are correct or not.
import numpy as np
import matplotlib.pyplot as plt

from fiberis.analyzer.Data1D import core1D
from fiberis.analyzer.Data2D import core2D

data1d_test_filepath = "scripts/fiberis_moose_generator/repeat_vf/saved_npz/with_kernel/pp_inj.npz"
data2d_test_filepath = "scripts/fiberis_moose_generator/repeat_vf/saved_npz/with_kernel/example_VF_csv_LineSampler_inj_pp.npz"

#%% Load data 1D
data1d_frame = core1D.Data1D()
data1d_frame.load_npz(data1d_test_filepath)

data1d_frame.history.print_records()

data1d_frame.print_info()

fig, ax = plt.subplots()
data1d_frame.plot(ax=ax)
plt.show()

#%% Load data 2D
data2d_frame = core2D.Data2D()
data2d_frame.load_npz(data2d_test_filepath)

data2d_frame.print_info()