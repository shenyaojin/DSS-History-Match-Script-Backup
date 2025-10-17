import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Geometry3D import DataG3D_md, coreG3D

#%% Load geometry data
s_well_dataframe = DataG3D_md.G3DMeasuredDepth()
h_well_dataframe = DataG3D_md.G3DMeasuredDepth()
prod_dataframe = DataG3D_md.G3DMeasuredDepth()

s_well_dataframe.load_npz('data/fiberis_format/s_well/geometry/swell_geometry.npz')
h_well_dataframe.load_npz('data/fiberis_format/h_well/geometry/hwell_geometry.npz')
prod_dataframe.load_npz('data/fiberis_format/prod/geometry/prod_geometry.npz')

#%% Create map view:
fig, ax = plt.subplots()
ax.plot(s_well_dataframe.xaxis, s_well_dataframe.yaxis, label='S Well')
ax.plot(h_well_dataframe.xaxis, h_well_dataframe.yaxis, label='H Well')
ax.plot(prod_dataframe.xaxis, prod_dataframe.yaxis, label='Producer')
# Set label
ax.set_xlabel('Easting (ft)')
ax.set_ylabel('Northing (ft)')
plt.show()