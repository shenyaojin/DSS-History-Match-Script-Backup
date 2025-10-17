# DSS data crossplot, different boreholes
# Shenyao, Sept 2025
#%% import modules
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Geometry3D.coreG3D import DataG3D # Load the geometry data

#%% Load data
# DSS data
prod_DSS_path = "data/fiberis_format/prod/dss_data/Mariner 14x-36-D - RFS strain change.npz"
swell_DSS_path = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
# swell_DSS_path = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change rate.npz"

# Geometry data, path
from glob import glob
swell_geo_path = glob("data/fiberis_format/s_well/geometry/frac_hit/*.npz")
prod_geo_path = glob("data/fiberis_format/prod/geometry/perf/*.npz")

#%% Load data
# DSS data
DSSdata_prod = DSS2D()
DSSdata_prod.load_npz(prod_DSS_path)

DSSdata_swell = DSS2D()
DSSdata_swell.load_npz(swell_DSS_path)

# Geometry data
geo_prod = []
for path in prod_geo_path:
    geo = DataG3D()
    geo.load_npz(path)
    geo_prod.append(geo)

geo_swell = []
for path in swell_geo_path:
    geo = DataG3D()
    geo.load_npz(path)
    geo_swell.append(geo)

#%% Pre-process the data
# Remove the drift in DSS data
ind = (DSSdata_swell.daxis > 7500) & (DSSdata_swell.daxis < 15000)
drift_val_swell = np.median(DSSdata_swell.data[ind, :], axis=0)
DSSdata_swell.data = DSSdata_swell.data - drift_val_swell.reshape((1, -1))

ind = (DSSdata_prod.daxis > 11000) & (DSSdata_prod.daxis < 11950)
drift_val_prod = np.mean(DSSdata_prod.data[ind, :], axis=0)
DSSdata_prod.data = DSSdata_prod.data - drift_val_prod.reshape((1, -1))

#%% Crop DSS data
# Crop S well
DSSdata_swell.select_depth(11500, 16761)
DSSdata_swell.select_time(0, 400000)
# Crop producer
DSSdata_prod.select_depth(11000, 16360)

#%% Convert the geometry data into list
perf_location_list = np.array([])
for geo in geo_prod:
    perf_iter = geo.data
    perf_location_list = np.concatenate((perf_location_list, perf_iter))

frac_location_list = np.array([])
for geo in geo_swell:
    frac_iter = geo.data
    frac_location_list = np.concatenate((frac_location_list, frac_iter))

#%% Sort the location list
perf_location_list = np.sort(perf_location_list)
frac_location_list = np.sort(frac_location_list)

#%% Plot the comparison data (Producer and S well)
# Use plt.subplot2grid to plot the data
fig = plt.figure(figsize=(10, 6))
ax1 = plt.subplot2grid((1, 2), (0, 0)) # For producer
ax2 = plt.subplot2grid((1, 2), (0, 1)) # For S well

cx = np.array([-1, 1])
# Plot producer data
im1 = DSSdata_prod.plot(ax=ax1, cmap='bwr')
im1.set_clim(cx * 2)
# Plot the perf locations, horizontal lines
for loc in perf_location_list:
    ax1.axhline(loc, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
ax1.set_title("Producer well (14x-36-D)")
ax1.set_ylabel("Depth (ft)")
ax1.set_xlabel("Time (s)")
plt.colorbar(im1, ax=ax1, label='Strain change (με)')
# Plot S well data
im2 = DSSdata_swell.plot(ax=ax2, cmap='bwr')
im2.set_clim(cx * 2)
# Plot the frac locations, horizontal lines
for loc in frac_location_list:
    ax2.axhline(loc, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.set_title("S well (14x-36-POW-S)")
ax2.set_ylabel("Depth (ft)")
ax2.set_xlabel("Time (s)")
plt.colorbar(im2, ax=ax2, label='Strain change (με)')
plt.suptitle("DSS data comparison between producer and S well")
plt.tight_layout()
plt.show()

