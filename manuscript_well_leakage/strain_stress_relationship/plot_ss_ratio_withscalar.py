# The plot script of scripts/well_leakage_history_matching/strain_pressure_coupling/101_ratio_estimation.py
# Need further better visualization
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D import Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Geometry3D import DataG3D_md
import datetime

#%% Global parameters list
DASdata_path = "data/fiberis_format/s_well/DAS/LFDASdata_stg1_swell.npz"
gauge_path = "data/fiberis_format/s_well/gauges/gauge1_data_swell.npz"
gauge_path2 = "data/fiberis_format/s_well/gauges/gauge2_data_swell.npz"
gauge_md_path = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
frac_md_path = "data/fiberis_format/s_well/geometry/frac_hit/frac_hit_stage_1_swell.npz"

#%% Load data
DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz(DASdata_path)

gauge_data1 = Data1D_Gauge.Data1DGauge()
gauge_data1.load_npz(gauge_path)

gauge_data2 = Data1D_Gauge.Data1DGauge()
gauge_data2.load_npz(gauge_path2)

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
gauge_data1.select_time(start_time, end_time)
gauge_data2.select_time(start_time, end_time)

#%% Get the DAS data channel
DASchan = []
DASchan.append(DASdata.get_value_by_depth(gauge_md_data.data[0]))
DASchan.append(DASdata.get_value_by_depth(gauge_md_data.data[1]))
DAStaxis = DASdata.taxis
DAStaxis -= DAStaxis[0]

DASchan = np.array(DASchan)

gauge_data1.interpolate(DAStaxis, gauge_data1.start_time)
gauge_data2.interpolate(DAStaxis, gauge_data2.start_time)

#%% Post processing
# 1. DAS data
conversion_factor = 1.55e-6 / (4 * 3.14 * 4.09 * 0.79)
unit_factor = 10430.4

DASdata_strain = DASdata.copy()
DASdata_strain.data = DASdata.data * conversion_factor / unit_factor

# 2. Gauge data, temporal gradient
gauge_data1_grad = gauge_data1.copy()
gauge_data1_grad = np.gradient(gauge_data1.data, gauge_data1.taxis)

gauge_data2_grad = gauge_data2.copy()
gauge_data2_grad = np.gradient(gauge_data2.data, gauge_data2.taxis)

# 3. Single channel DAS data
DASchan_strain = DASchan * conversion_factor / unit_factor

#%% Plot the first channel

fig, ax = plt.subplots()
line1, = ax.plot(DAStaxis, DASchan_strain[0], label="DAS")
ax.set_ylabel(r"Strain rate")
ax.set_ylim(-0.2e-7, 1.2e-7)

ax1 = ax.twinx()
line2, = ax1.plot(DAStaxis, gauge_data1_grad, label="Gauge", color="red", linestyle='--')
ax1.set_ylabel(r"Pressure gradient (psi/s)")
ax1.set_ylim(-2, 25)

ax.set_xlabel("Time (s)")

# Combine legends
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc="upper right")

plt.title("Strain rate vs pressure gradient (1st gauge)")
plt.show()

#%% Plot the second channel
fig, ax = plt.subplots()
line1, = ax.plot(DAStaxis, DASchan_strain[1], label="DAS")
ax.set_ylabel(r"Strain rate")
ax.set_ylim(-0.2e-7, 1.2e-7)

ax1 = ax.twinx()
line2, = ax1.plot(DAStaxis, gauge_data2_grad, label="Gauge", color="red", linestyle='--')
ax1.set_ylabel(r"Pressure gradient (psi/s)")
ax1.set_ylim(-2, 25)

ax.set_xlabel("Time (s)")

# Combine legends
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc="upper right")

plt.title("Strain rate vs pressure gradient (2nd gauge)")
plt.show()

#%% Plot all the data in scatter
# 1. Combine the data
gauge_data1_grad = gauge_data1_grad.flatten()
gauge_data2_grad = gauge_data2_grad.flatten()
DASchan_strain_flatten = DASchan_strain.flatten()
# Combine gauge data
gauge_data_grad_all = np.concatenate([gauge_data1_grad, gauge_data2_grad])

# 2. Plot the data
fig, ax = plt.subplots()
img = ax.scatter(gauge_data_grad_all, DASchan_strain_flatten, s=1)
cbar = fig.colorbar(img, ax=ax)
cbar.set_label('Time (s)')
ax.set_xlabel("Pressure gradient (psi/s)")
ax.set_ylabel("Strain rate")
ax.set_title("Strain rate vs pressure gradient (2 gauge locations)")

plt.show()

#%% Post processing: test: z score
from scipy.stats import zscore
x = np.array(gauge_data_grad_all)
y = np.array(DASchan_strain_flatten)

z_x = zscore(x)
z_y = zscore(y)
threshold = 3
mask = (np.abs(z_x) < threshold) & (np.abs(z_y) < threshold)

# plot
fig, ax = plt.subplots()
img = ax.scatter(x[mask], y[mask], s=1, c=np.linspace(0, 1, np.sum(mask)), cmap='viridis')
cbar = fig.colorbar(img, ax=ax)
cbar.set_label('Time (s)')
ax.set_xlabel("Pressure gradient (psi/s)")
ax.set_ylabel("Strain rate")
ax.set_title("Strain rate vs pressure gradient (2 gauge locations, filtered)")

plt.show()

#%% Post processing: test: z score, with WLS
# Prepare data
x = np.array(gauge_data_grad_all)
y = np.array(DASchan_strain_flatten)

# Remove outliers using z-score filtering
z_x = zscore(x)
z_y = zscore(y)
mask = (np.abs(z_x) < 3) & (np.abs(z_y) < 3)
x_filtered = x[mask]
y_filtered = y[mask]

# Design matrix for linear regression
X = np.vstack([x_filtered, np.ones_like(x_filtered)]).T

# Ordinary Least Squares (OLS) regression to compute residuals
beta_ols = np.linalg.lstsq(X, y_filtered, rcond=None)[0]
residuals = np.abs(y_filtered - X @ beta_ols)

# Compute weights inversely proportional to residuals
weights = 1 / (residuals + 1e-9)
W = np.diag(weights)

# Weighted Least Squares (WLS) regression
beta_wls = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y_filtered)
y_fit = X @ beta_wls

# Sort for clean plotting
sort_idx = np.argsort(x_filtered)
x_sorted = x_filtered[sort_idx]
y_sorted = y_filtered[sort_idx]
y_fit_sorted = y_fit[sort_idx]

# Plot
fig, ax = plt.subplots()
ax.scatter(x_sorted, y_sorted, s=5, alpha=0.6, label="Filtered Data")
ax.plot(x_sorted, y_fit_sorted, color="red", linewidth=2,
        label=f"WLS Fit: y = {beta_wls[0]:.2e}x + {beta_wls[1]:.2e}")
ax.set_xlabel("Pressure gradient (psi/s)")
ax.set_ylabel("Strain rate")
ax.set_title("Weighted Linear Regression on Filtered DAS vs Pressure Gradient")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#%% Back up: WLS on original data
# Original data (without outlier filtering)
x_orig = np.array(gauge_data_grad_all)
y_orig = np.array(DASchan_strain_flatten)

# Design matrix
X_orig = np.vstack([x_orig, np.ones_like(x_orig)]).T

# OLS to get initial residuals
beta_ols_orig = np.linalg.lstsq(X_orig, y_orig, rcond=None)[0]
residuals_orig = np.abs(y_orig - X_orig @ beta_ols_orig)

# Compute weights
weights_orig = 1 / (residuals_orig + 1e-9)
W_orig = np.diag(weights_orig)

# Weighted Least Squares
beta_wls_orig = np.linalg.inv(X_orig.T @ W_orig @ X_orig) @ (X_orig.T @ W_orig @ y_orig)
y_fit_orig = X_orig @ beta_wls_orig

# Sort for plotting
sort_idx_orig = np.argsort(x_orig)
x_sorted_orig = x_orig[sort_idx_orig]
y_sorted_orig = y_orig[sort_idx_orig]
y_fit_sorted_orig = y_fit_orig[sort_idx_orig]

# Plot
fig, ax = plt.subplots()
ax.scatter(x_sorted_orig, y_sorted_orig, s=5, alpha=0.6, label="Original Data")
ax.plot(x_sorted_orig, y_fit_sorted_orig, color="darkorange", linewidth=2,
        label=f"WLS Fit: y = {beta_wls_orig[0]:.2e}x + {beta_wls_orig[1]:.2e}")
ax.set_xlabel("Pressure gradient (psi/s)")
ax.set_ylabel("Strain rate")
ax.set_title("Weighted Linear Regression on Raw DAS vs Pressure Gradient")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()