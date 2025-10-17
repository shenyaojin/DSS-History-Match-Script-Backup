#%% Import lib
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D import Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Geometry3D import DataG3D_md
import datetime

#%% Global parameters list
DASdata_path = "data/fiberis_format/s_well/DAS/LFDASdata_stg1_swell.npz"
gauge_path = "data/fiberis_format/s_well/gauges/gauge1_data_swell.npz"
gauge_md_path = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
frac_md_path = "data/fiberis_format/s_well/geometry/frac_hit/frac_hit_stage_1_swell.npz"

#%% Load data
DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz(DASdata_path)

gauge_data = Data1D_Gauge.Data1DGauge()
gauge_data.load_npz(gauge_path)

gauge_md_data = DataG3D_md.G3DMeasuredDepth()
gauge_md_data.load_npz(gauge_md_path)

frac_md_data = DataG3D_md.G3DMeasuredDepth()
frac_md_data.load_npz(frac_md_path)

#%% Do preprocessing
start_time = datetime.datetime(2020, 3, 16, 10, 36, 0)
end_time = datetime.datetime(2020, 3, 16, 10, 52, 0)
print("Start Time: ", DASdata.start_time)
DASdata.select_time(start_time, end_time)
DASdata.select_depth(15000, 16750)
gauge_data.select_time(start_time, end_time)

#%% Get the DAS data channel
DASchan = DASdata.get_value_by_depth(gauge_md_data.data[0])
DAStaxis = DASdata.taxis
DAStaxis -= DAStaxis[0] # subtract the first time
# Interp gauge to DAS
gauge_data.interpolate(DAStaxis, gauge_data.start_time)

#%% Get the length
print("Length of gauge data", len(gauge_data.data))
print("Length of DAS chan data", len(DAStaxis))

#%% Post-processing
# 1. DAS data
conversion_factor = 1.55e-6 / (4 * 3.14 * 4.09 * 0.79)
unit_factor = 10430.4

DASdata_strain = DASdata.copy()
DASdata_strain.data = DASdata.data * conversion_factor / unit_factor

# 2. Gauge data, convert to temporal pressure gradient
gauge_data_temporal_grad = gauge_data.copy()
gauge_data_temporal_grad = np.gradient(gauge_data.data, gauge_data.taxis)

# 3. Single channel DAS data
DASchan_strain = DASchan * conversion_factor / unit_factor

#%%  Plot the DAS data
fig, ax = plt.subplots()
img = DASdata_strain.plot(ax=ax, cmap='bwr')
# Set clim
cx = np.array([-1, 1])
img.set_clim(cx * 5e-8)
# Set color bar
cbar = fig.colorbar(img, ax=ax)
cbar.set_label(r"strain rate")
plt.savefig("figs/04142025/fig1.png")
plt.show()

#%% Co plot DAS chan and gauge data
fig, ax = plt.subplots()
line1, = ax.plot(DAStaxis, DASchan_strain, label="DAS")
ax.set_ylabel(r"Strain rate")
ax.set_ylim(-0.2e-7, 1.2e-7)

ax1 = ax.twinx()
line2, = ax1.plot(DAStaxis, gauge_data_temporal_grad, label="Gauge", color="red", linestyle='--')
ax1.set_ylabel(r"Pressure gradient (psi/s)")
ax1.set_ylim(-2, 25)

ax.set_xlabel("Time (s)")

# Combine legends
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc="upper right")

plt.title("Strain rate vs pressure gradient")
plt.savefig("figs/04142025/fig2.png")
plt.show()

#%% Plot the scatter plot to get the ratio
import numpy as np
import matplotlib.pyplot as plt

# Prepare data
x = np.array(gauge_data_temporal_grad)
y = np.array(DASchan_strain)

# Linear regression using numpy
slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept

# Plot
plt.figure()
plt.scatter(x, y, label="Data", alpha=0.5)
plt.plot(x, y_fit, color='red', label=f"Fit: y = {slope:.2e}x + {intercept:.2e}")
plt.xlabel("Pressure gradient (psi/s)")
plt.ylabel("Strain rate")
plt.legend()
plt.grid(True)
plt.title("Linear Regression: DAS vs Gauge Gradient")
plt.savefig("figs/04142025/fig3.png")
plt.show()

#%% Do more dedicated analysis
# Do the weighted linear regression.
X = np.vstack([x, np.ones_like(x)]).T
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
residuals = np.abs(y - X @ beta_ols)
weights = 1 / (residuals + 1e-9)
W = np.diag(weights)
beta_wls = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
y_fit = X @ beta_wls

#%% Plot the weighted linear regression.
plt.scatter(x, y, label="Data")
plt.plot(x, y_fit, color="red", label=f"WLS Fit: y = {beta_wls[0]:.2e}x + {beta_wls[1]:.2e}")
plt.xlabel("Pressure gradient (psi/s)")
plt.ylabel("Strain rate")
plt.legend()
plt.title("Weighted Linear Regression")
plt.show()

#%% Co plot the DAS chan and gauge data.