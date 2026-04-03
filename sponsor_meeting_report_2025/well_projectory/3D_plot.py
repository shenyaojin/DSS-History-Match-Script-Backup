import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Geometry3D import DataG3D_md

#%% Set GRB for three wells -> Purple, orange and dark green
s_well_color = (128/255, 0, 128/255)
h_well_color = (255/255, 165/255, 0) # Orange
prod_color = (0, 100/255, 0)

#%% Load geometry data
s_well_dataframe = DataG3D_md.G3DMeasuredDepth()
h_well_dataframe = DataG3D_md.G3DMeasuredDepth()
prod_dataframe = DataG3D_md.G3DMeasuredDepth()

s_well_dataframe.load_npz('data/fiberis_format/s_well/geometry/swell_geometry.npz')
h_well_dataframe.load_npz('data/fiberis_format/h_well/geometry/hwell_geometry.npz')
prod_dataframe.load_npz('data/fiberis_format/prod/geometry/prod_geometry.npz')

#%% Load the pressure gauge data
s_well_pg_dataframe = DataG3D_md.G3DMeasuredDepth()
h_well_pg_dataframe = DataG3D_md.G3DMeasuredDepth()
prod_pg_dataframe = DataG3D_md.G3DMeasuredDepth()

s_well_pg_dataframe.load_npz("data/fiberis_format/s_well/geometry/gauge_md_swell.npz")
h_well_pg_dataframe.load_npz("data/fiberis_format/h_well/geometry/gauge_md_hwell.npz")
prod_pg_dataframe.load_npz("data/fiberis_format/prod/geometry/gauge_md_prod.npz")

#%% Process the pressure gauge md. I want to mark the guage location on well trajectory.
s_well_gauge_x = np.interp(s_well_pg_dataframe.data, s_well_dataframe.data, s_well_dataframe.xaxis)
s_well_gauge_y = np.interp(s_well_pg_dataframe.data, s_well_dataframe.data, s_well_dataframe.yaxis)
s_well_gauge_z = np.interp(s_well_pg_dataframe.data, s_well_dataframe.data, s_well_dataframe.zaxis)

h_well_gauge_x = np.interp(h_well_pg_dataframe.data, h_well_dataframe.data, h_well_dataframe.xaxis)
h_well_gauge_y = np.interp(h_well_pg_dataframe.data, h_well_dataframe.data, h_well_dataframe.yaxis)
h_well_gauge_z = np.interp(h_well_pg_dataframe.data, h_well_dataframe.data, h_well_dataframe.zaxis)

prod_gauge_x = np.interp(prod_pg_dataframe.data, prod_dataframe.data, prod_dataframe.xaxis)
prod_gauge_y = np.interp(prod_pg_dataframe.data, prod_dataframe.data, prod_dataframe.yaxis)
prod_gauge_z = np.interp(prod_pg_dataframe.data, prod_dataframe.data, prod_dataframe.zaxis)


# Do processing based on security policy
z_min, z_max = 10900, 11200  # Adjust based on your data range
s_well_dataframe.zaxis -= z_min
h_well_dataframe.zaxis -= z_min
prod_dataframe.zaxis -= z_min

s_well_gauge_z -= z_min
h_well_gauge_z -= z_min
prod_gauge_z -= z_min

#%% Create 3D plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()  # Increase the width
ax = fig.add_subplot(111, projection='3d')
ax.plot(s_well_dataframe.xaxis, s_well_dataframe.yaxis, s_well_dataframe.zaxis
        , linewidth = 2, label = 'S Well', color = s_well_color)
ax.plot(h_well_dataframe.xaxis, h_well_dataframe.yaxis, h_well_dataframe.zaxis,
        linewidth = 2, label = 'H Well', color = h_well_color)
ax.plot(prod_dataframe.xaxis, prod_dataframe.yaxis, prod_dataframe.zaxis
        , linewidth = 2, label = 'Producer', color = prod_color)

ax.scatter(s_well_gauge_x, s_well_gauge_y, s_well_gauge_z, color='cyan', label='Pressure Gauge', marker='^')
ax.scatter(h_well_gauge_x, h_well_gauge_y, h_well_gauge_z, color='cyan', marker='^')
ax.scatter(prod_gauge_x, prod_gauge_y, prod_gauge_z, color='cyan', marker='^')

ax.set_yticks(np.linspace(100, -1900, 5))  # Adjust number of ticks
# Remove y ticks
# ax.set_yticks([])
# Set y label location


# Maintain original Z-axis limits but ensure the first tick starts at 0
ax.set_zlim(0, z_max - z_min)

# Invert z axis
ax.invert_zaxis()

ax.view_init(elev=30, azim=290)

ax.set_xlabel('Easting / ft', labelpad = 20)
ax.set_ylabel('Northing / ft', labelpad = 5)
ax.set_zlabel('TVD / ft')
ax.set_title('Well Geometry')

ax.set_box_aspect([4, 1, 1])  # (x, y, z) aspect ratio

ax.xaxis.pane.fill = False  # Remove x-axis background
ax.yaxis.pane.fill = False  # Remove y-axis background
ax.zaxis.pane.fill = False  # Remove z-axis background

ax.xaxis.line.set_color((1,1,1,0))  # Hide x-axis line
ax.yaxis.line.set_color((1,1,1,0))  # Hide y-axis line
ax.zaxis.line.set_color((1,1,1,0))  # Hide z-axis line

ax.legend()
plt.savefig("figs/03242025/well_trajectory2.png")
plt.show()
