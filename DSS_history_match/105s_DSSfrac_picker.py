# In this script, I aim to create lots of figures helping me to identify the location of
# Conductive HFs.
# The report of Neubrex is meaningless. In their Excel form there is NO any valid information TBH.
# Shenyao Jin, shenyaojin@mines.edu
# Created on 10/06/2025

#%% Import libraries
import numpy as np

import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth
from typing import Union, List, Tuple, Optional

#%% Load necessary data
# DSS data
DSSdata_path = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"

DSSdata = DSS2D()
DSSdata.load_npz(DSSdata_path)

mds = DSSdata.daxis
ind = (mds>7500)&(mds<15000)
drift_val = np.median(DSSdata.data[ind,:],axis=0)
DSSdata.data -= drift_val.reshape((1,-1))

DSSdata.select_time(0, 400000)
DSSdata.select_depth(11500, 16700)

# PG data
PG_datapath = "data/fiberis_format/prod/gauges/pressure_g1.npz"

gauge_dataframe = Data1DGauge()
gauge_dataframe.load_npz(PG_datapath)
gauge_dataframe.select_time(DSSdata.start_time, DSSdata.get_end_time())

# Frac-hit location data
from glob import glob
frac_hit_loc_datalist = glob("data/fiberis_format/s_well/geometry/frac_hit/*.npz")
frac_hit_md = np.array([])

for file in frac_hit_loc_datalist:
    frac_hit_dataframe = G3DMeasuredDepth()
    frac_hit_dataframe.load_npz(file)
    frac_hit_md = np.append(frac_hit_md, frac_hit_dataframe.data)

frac_hit_md.ravel()
frac_hit_md = np.sort(frac_hit_md)

#%% Define the function plotting the info
def plot_HF(data2d: DSS2D, data1d: Data1DGauge, data3d: np.ndarray, selected_md: Union[float, int], output_path: Optional[str]) -> None:
    """
    Co plot the HF data with fiber optic sensing data.

    :param data2d: Fiber optic sensing data
    :param data1d: Gauge data for reference
    :param data3d: Fracture hit location data to indicate how close the selected MD is to HF during stimulation.
    :param output_path: Where I want the figure to output.
    :return: Nothing. The figure will be created and
    """

    # Get the closest frac hit location.
    closest_index = np.argmin(np.abs(frac_hit_md - selected_md))
    closest_md = frac_hit_md[closest_index]

    # Get the depth range
    data2d_copy = data2d.copy()
    data2d_copy.select_depth(selected_md-100, selected_md+100)

    # select channels
    chan1 = data2d.get_value_by_depth(selected_md) # Center
    chan2 = data2d.get_value_by_depth(selected_md-1) # Upper
    chan3 = data2d.get_value_by_depth(selected_md+1) # Lower

    # Use subplot2grid to plot the data
    plt.subplots(figsize=(16, 8))
    ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3, rowspan=1)
    ax3 = plt.subplot2grid((3, 6), (1, 3), colspan=3, rowspan=1)
    ax4 = plt.subplot2grid((3, 6), (2, 3), colspan=3, rowspan=1)

    # Plot the 2D data
    im = data2d_copy.plot(ax=ax1, cmap='bwr', use_timestamp=True)
    ax1.axhline(closest_md, color='k', ls='--')
    ax1.axhline(selected_md, color='g', ls='--')
    ax1.set_title(f"DSS Data around {selected_md} ft MD\nClosest HF at {closest_md} ft MD")

    # Plot the 1D data
    data1d.plot(ax=ax2, use_timestamp=True, use_legend=False)
    ax2.set_title("Gauge Data")
    ax2.set_ylabel("Pressure")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(data2d_copy.calculate_time(), chan1, color='g', label=f'Center MD {selected_md} ft')
    # add legend
    ax2_twin.legend()
    ax2_twin.set_ylabel("Strain Change (με)")

    data1d.plot(ax=ax3, use_timestamp=True, use_legend=False)
    ax3.set_ylabel("Pressure")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(data2d_copy.calculate_time(), chan2, color='b', label=f'Upper MD {selected_md-1} ft')
    # add legend
    ax3_twin.legend()
    ax3_twin.set_ylabel("Strain Change (με)")

    data1d.plot(ax=ax4, use_timestamp=True, use_legend=False)
    ax4.set_ylabel("Pressure")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(data2d_copy.calculate_time(), chan3, color='r', label=f'Lower MD {selected_md+1} ft')
    # add legend
    ax4_twin.legend()
    ax4_twin.set_ylabel("Strain Change (με)")

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

#%% Test the function
# Delete the existing figure
# filepath = "figs/10072025/test.png"
# import os
# if os.path.exists(filepath):
#     os.remove(filepath)
# else:
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)
# plot_HF(DSSdata, gauge_dataframe, frac_hit_md, 14600, "figs/10072025/test.png")

#%% Loop through all the frac hit locations and create figures
output_dir = "figs/10072025/"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create an array round +- 100 ft around each frac hit location
md_array = np.array([])
for md in frac_hit_md:
    md_array = np.append(md_array, np.arange(md-100, md+101, 10))
md_array = np.unique(md_array)
md_array = md_array[(md_array>11500)&(md_array<16700)]
md_array = np.sort(md_array)

for md in md_array:
    output_path = os.path.join(output_dir, f"HF_{int(md)}ft.png")
    if not os.path.exists(output_path):
        print(f"Creating figure for {md} ft MD...")
        plot_HF(DSSdata, gauge_dataframe, frac_hit_md, md, output_path)
    else:
        print(f"Figure for {md} ft MD already exists. Skipping...")