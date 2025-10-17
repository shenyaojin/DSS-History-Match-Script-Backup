# Combine the plot in other files. Finally, I got the well trajectory plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot2grid

from fiberis.analyzer.Geometry3D import DataG3D_md, coreG3D

#%% Set up the RGB color
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

#%% Load pressure gauge MD data.
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

#%% Do processing based on security policy
z_min, z_max = 10900, 11200  # Adjust based on your data range
s_well_dataframe.zaxis -= z_min
h_well_dataframe.zaxis -= z_min
prod_dataframe.zaxis -= z_min

s_well_gauge_z -= z_min
h_well_gauge_z -= z_min
prod_gauge_z -= z_min

#%% Plot the well trajectory
fig = plt.figure(figsize=(6, 8))

# Define ax1 in a 4x4 grid starting from (0,0)
ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=4)
ax1.plot(s_well_dataframe.xaxis, s_well_dataframe.yaxis, label='S Well')
ax1.plot(h_well_dataframe.xaxis, h_well_dataframe.yaxis, label='H Well')
ax1.plot(prod_dataframe.xaxis, prod_dataframe.yaxis, label='Producer')
ax1.scatter(s_well_gauge_x, s_well_gauge_y, label='Pressure Gauge', color='cyan', marker='^')
ax1.scatter(h_well_gauge_x, h_well_gauge_y, color='cyan', marker='^')
# Set label
ax1.set_ylabel('Northing (ft)')
# Make xticks invisible. Don't use set_xticks because I'm sharing x with ax2
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Define ax2 in the same 4x4 grid, directly below ax1
ax2 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=4, sharex=ax1)
ax2.plot(s_well_dataframe.xaxis, s_well_dataframe.zaxis, label='S Well', color=s_well_color)
ax2.plot(h_well_dataframe.xaxis, h_well_dataframe.zaxis, label='H Well', color=h_well_color)
ax2.plot(prod_dataframe.xaxis, prod_dataframe.zaxis, label='Producer', color=prod_color)
ax2.scatter(s_well_gauge_x, s_well_gauge_z, color='cyan', marker='^')
ax2.scatter(h_well_gauge_x, h_well_gauge_z, color='cyan', marker='^')

# Set label
ax2.set_xlabel('Easting (ft)')
ax2.set_ylabel('TVD (ft)')
ax2.set_ylim(0, z_max - z_min)
ax2.invert_yaxis()

plt.savefig("figs/03242025/well_trajectory1.png")
plt.show()

#%% Old code: Create subplot2grid
# fig = plt.figure()
# ax1 = subplot2grid((4, 8), (0, 0),
#                    colspan=4, rowspan=4, projection='3d')
# ax1.plot(s_well_dataframe.xaxis, s_well_dataframe.yaxis, s_well_dataframe.zaxis
#         , linewidth = 2, label = 'S Well', color = s_well_color)
# ax1.plot(h_well_dataframe.xaxis, h_well_dataframe.yaxis, h_well_dataframe.zaxis,
#         linewidth = 2, label = 'H Well', color = h_well_color)
# ax1.plot(prod_dataframe.xaxis, prod_dataframe.yaxis, prod_dataframe.zaxis
#         , linewidth = 2, label = 'Producer', color = prod_color)
#
# ax1.set_yticks(np.linspace(100, -1900, 5))  # Adjust number of ticks
# # Remove y ticks
# # ax1.set_yticks([])
# # Set y label location
#
#
# # Maintain original Z-axis limits but ensure the first tick starts at 0
# ax1.set_zlim(0, z_max - z_min)
#
# # Invert z axis
# ax1.invert_zaxis()
#
# ax1.view_init(elev=30, azim=290)
#
# ax1.set_xlabel('Easting / ft', labelpad = 20)
# ax1.set_ylabel('Northing / ft', labelpad = 5)
# ax1.set_zlabel('TVD / ft')
# ax1.set_title('Well Geometry')
#
# ax1.set_box_aspect([4, 1, 1])  # (x, y, z) aspect ratio
#
# ax1.xaxis.pane.fill = False  # Remove x-axis background
# ax1.yaxis.pane.fill = False  # Remove y-axis background
# ax1.zaxis.pane.fill = False  # Remove z-axis background
#
# ax1.xaxis.line.set_color((1,1,1,0))  # Hide x-axis line
# ax1.yaxis.line.set_color((1,1,1,0))  # Hide y-axis line
# ax1.zaxis.line.set_color((1,1,1,0))  # Hide z-axis line
# ax1.legend()
#
# ax2 = subplot2grid((4, 8), (0, 4)
#                    , rowspan=2, colspan=4)
# ax2.plot(s_well_dataframe.xaxis, s_well_dataframe.yaxis, label='S Well')
# ax2.plot(h_well_dataframe.xaxis, h_well_dataframe.yaxis, label='H Well')
# ax2.plot(prod_dataframe.xaxis, prod_dataframe.yaxis, label='Producer')
# # Set label
# ax2.set_ylabel('Northing (ft)')
# # Make xticks invisible. Don't use set_xticks because I'm sharing x with ax3
# ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#
# ax3 = subplot2grid((4, 8), (2, 4)
#                    , rowspan=2, colspan=4, sharex=ax2)
# ax3.plot(s_well_dataframe.xaxis, s_well_dataframe.zaxis, label='S Well')
# ax3.plot(h_well_dataframe.xaxis, h_well_dataframe.zaxis, label='H Well')
# ax3.plot(prod_dataframe.xaxis, prod_dataframe.zaxis, label='Producer')
# # Set label
# ax3.set_xlabel('Easting (ft)')
# ax3.set_ylabel('TVD (ft)')
# ax3.set_ylim(0, z_max - z_min)
# ax3.invert_yaxis()
#
# plt.show()
