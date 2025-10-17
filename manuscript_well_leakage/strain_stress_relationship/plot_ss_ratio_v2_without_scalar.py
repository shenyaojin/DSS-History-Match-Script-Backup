# The plot script of scripts/well_leakage_history_matching/strain_pressure_coupling/101_ratio_estimation.py
# Enhanced visualization with complex figure layout (scalar-free version for publication)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # For complex layouts
from fiberis.analyzer.Data1D import Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Geometry3D import DataG3D_md
import datetime
from scipy.stats import zscore

#%% Global parameters list
DASdata_path = "data/fiberis_format/s_well/DAS/LFDASdata_stg1_swell.npz"
gauge_path = "data/fiberis_format/s_well/gauges/gauge1_data_swell.npz"
gauge_path2 = "data/fiberis_format/s_well/gauges/gauge2_data_swell.npz"
gauge_md_path = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
frac_md_path = "data/fiberis_format/s_well/geometry/frac_hit/frac_hit_stage_1_swell.npz"

#%% Load data
print("Loading data...")
DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz(DASdata_path)

gauge_data1 = Data1D_Gauge.Data1DGauge()
gauge_data1.load_npz(gauge_path)

gauge_data2 = Data1D_Gauge.Data1DGauge()
gauge_data2.load_npz(gauge_path2)

gauge_md_data = DataG3D_md.G3DMeasuredDepth()
gauge_md_data.load_npz(gauge_md_path)

# frac_md_data = DataG3D_md.G3DMeasuredDepth()
# frac_md_data.load_npz(frac_md_path)

#%% Do preprocessing
print("Preprocessing data...")
start_time = datetime.datetime(2020, 3, 16, 10, 36, 0)
end_time = datetime.datetime(2020, 3, 16, 10, 52, 0)
print(f"Original DAS Start Time (from data object): {getattr(DASdata, 'start_time', 'N/A')}")
DASdata.select_time(start_time, end_time)
DASdata.select_depth(15000, 16750)
gauge_data1.select_time(start_time, end_time)
gauge_data2.select_time(start_time, end_time)

#%% Get the DAS data channel
print("Extracting DAS channels and common time axis...")
DASchan = []
if gauge_md_data.data is not None and len(gauge_md_data.data) >= 2:
    DASchan.append(DASdata.get_value_by_depth(gauge_md_data.data[0]))
    DASchan.append(DASdata.get_value_by_depth(gauge_md_data.data[1]))
else:
    print("Warning: gauge_md_data might not be as expected. Using fallback DAS channels.")
    DASchan.append(DASdata.data[0, :] if DASdata.data.ndim == 2 and DASdata.data.shape[0] > 0 else (DASdata.data if DASdata.data.ndim == 1 else np.array([])))
    DASchan.append(DASdata.data[1, :] if DASdata.data.ndim == 2 and DASdata.data.shape[0] > 1 else (DASdata.data if DASdata.data.ndim == 1 else np.array([])))

DAStaxis = np.copy(DASdata.taxis)
if len(DAStaxis) > 0:
    DAStaxis -= DAStaxis[0]
else:
    print("Warning: DAStaxis is empty after DASdata.select_time.")
    if DASchan and hasattr(DASchan[0], 'shape') and len(DASchan[0]) > 0:
        DAStaxis = np.arange(len(DASchan[0]))


DASchan = np.array(DASchan)
if DASchan.ndim == 1 and DASchan.size > 0 :
    print("Warning: Only one DAS channel seems to be processed into DASchan. Reshaping for consistency.")
    if len(DASchan) == len(DAStaxis):
        DASchan = DASchan.reshape(1, -1)
        if DASchan.shape[0] < 2:
            print("Duplicating single DAS channel data for placeholder second channel.")
            DASchan = np.vstack([DASchan, DASchan])
    else:
        min_len = min(DASchan.size, len(DAStaxis))
        DASchan = DASchan[:min_len].reshape(1,-1)
        DAStaxis = DAStaxis[:min_len]
        if DASchan.shape[0] < 2:
             print("Duplicating single DAS channel data for placeholder second channel after trim.")
             DASchan = np.vstack([DASchan, DASchan])

elif DASchan.ndim == 2 and DASchan.shape[1] != len(DAStaxis) and len(DAStaxis) > 0:
    print(f"Warning: DAS channel length ({DASchan.shape[1]}) and DAStaxis length ({len(DAStaxis)}) mismatch. Attempting to trim/match.")
    min_len = min(DASchan.shape[1], len(DAStaxis))
    DASchan = DASchan[:, :min_len]
    DAStaxis = DAStaxis[:min_len]

if DASchan.ndim < 2 or DASchan.shape[0] < 2:
    print("Error or Warning: DASchan does not have two distinct channels after processing. Plotting for channel 2 might be redundant or fail.")
    if DASchan.ndim == 1: DASchan = DASchan.reshape(1,-1)
    while DASchan.shape[0] < 2 :
        if DASchan.size == 0 :
             DASchan = np.full((2, len(DAStaxis) if len(DAStaxis) > 0 else 1), np.nan)
             print("DASchan was empty, created two NaN channels.")
        else:
             DASchan = np.vstack([DASchan, np.full_like(DASchan[0], np.nan)])
             print("Added a NaN placeholder for the second DAS channel.")


print("Interpolating gauge data to DAS time axis...")
gauge_data1.interpolate(DAStaxis, getattr(gauge_data1, 'start_time', None))
gauge_data2.interpolate(DAStaxis, getattr(gauge_data2, 'start_time', None))

#%% Post processing
print("Post processing data (gradients and strain conversion)...")
conversion_factor = 1.55e-6 / (4 * np.pi * 4.09 * 0.79)
unit_factor = 10430.4
DASchan_strain = DASchan * conversion_factor / unit_factor

if len(gauge_data1.taxis) > 1 and len(gauge_data1.data) == len(gauge_data1.taxis):
    gauge_data1_grad = np.gradient(gauge_data1.data, gauge_data1.taxis)
else:
    print("Warning: Gauge 1 data or taxis problematic for gradient. Using zeros.")
    gauge_data1_grad = np.zeros_like(DAStaxis) if len(DAStaxis) > 0 else np.array([])

if len(gauge_data2.taxis) > 1 and len(gauge_data2.data) == len(gauge_data2.taxis):
    gauge_data2_grad = np.gradient(gauge_data2.data, gauge_data2.taxis)
else:
    print("Warning: Gauge 2 data or taxis problematic for gradient. Using zeros.")
    gauge_data2_grad = np.zeros_like(DAStaxis) if len(DAStaxis) > 0 else np.array([])

#%% Prepare data for combined scatter plots and WLS
print("Preparing combined data for scatter plots...")
x_actual = np.concatenate([gauge_data1_grad.flatten(), gauge_data2_grad.flatten()])
y_actual = DASchan_strain.flatten()

#%% WLS Helper Function
def calculate_wls(x_data, y_data):
    if x_data is None or y_data is None or len(x_data) < 2 or len(y_data) < 2 or len(x_data) != len(y_data):
        return np.array([np.nan, np.nan])
    X_matrix = np.vstack([x_data, np.ones_like(x_data)]).T
    try:
        beta_ols = np.linalg.lstsq(X_matrix, y_data, rcond=None)[0]
        residuals = np.abs(y_data - (X_matrix @ beta_ols))
        weights = 1.0 / (np.maximum(residuals, 1e-9))
        W_diag = np.diag(weights)
        XTWX = X_matrix.T @ W_diag @ X_matrix
        XTWy = X_matrix.T @ W_diag @ y_data
        if np.linalg.cond(XTWX) < 1.0 / np.finfo(float).eps:
            beta_wls = np.linalg.solve(XTWX, XTWy)
        else:
            beta_wls = beta_ols
        return beta_wls
    except np.linalg.LinAlgError:
        beta_ols = np.linalg.lstsq(X_matrix, y_data, rcond=None)[0]
        return beta_ols

#%% Calculate WLS for necessary datasets
print("Calculating WLS fits...")
beta_wls_g1_raw = calculate_wls(gauge_data1_grad, DASchan_strain[0])
beta_wls_g2_raw = calculate_wls(gauge_data2_grad, DASchan_strain[1])

predicted_strain_rate_g1_orig_units = np.full_like(gauge_data1_grad, np.nan)
if not np.isnan(beta_wls_g1_raw).any():
    predicted_strain_rate_g1_orig_units = beta_wls_g1_raw[0] * gauge_data1_grad + beta_wls_g1_raw[1]

predicted_strain_rate_g2_orig_units = np.full_like(gauge_data2_grad, np.nan)
if not np.isnan(beta_wls_g2_raw).any():
    predicted_strain_rate_g2_orig_units = beta_wls_g2_raw[0] * gauge_data2_grad + beta_wls_g2_raw[1]

if len(x_actual) > 0 and len(y_actual) > 0:
    z_x_actual = zscore(x_actual)
    z_y_actual = zscore(y_actual)
    threshold_actual = 3.0
    mask_actual = (np.abs(z_x_actual) < threshold_actual) & (np.abs(z_y_actual) < threshold_actual)
    x_filtered_actual = x_actual[mask_actual]
    y_filtered_actual = y_actual[mask_actual]
else:
    x_filtered_actual, y_filtered_actual = np.array([]), np.array([])
beta_wls_filt_actual = calculate_wls(x_filtered_actual, y_filtered_actual)

#%% Plotting
print("Generating scalar-free complex figure layout for publication...")
micro_scale_factor = 1e6

plt.style.use('seaborn-v0_8-whitegrid')
common_rc_params = {
    'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 9, # Labels will be empty
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
}
plt.rcParams.update(common_rc_params)

fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.15) # Reduced spacing as labels are gone

# --- Left Side: Time-Series Co-plots (Actual DAS vs. Predicted DAS on SAME Y-AXIS, SCALAR-FREE) ---
# Top-Left: Time-Series for Gauge 1 / Channel 1
ax_ts1 = fig.add_subplot(gs[0, 0:2])
das_ch1_micro = DASchan_strain[0] * micro_scale_factor
predicted_strain_g1_micro = predicted_strain_rate_g1_orig_units * micro_scale_factor

line_ts1_das, = ax_ts1.plot(DAStaxis, das_ch1_micro, color='dodgerblue', lw=1.5, label="LF-DAS")
line_ts1_pred_gauge, = ax_ts1.plot(DAStaxis, predicted_strain_g1_micro, color='red', linestyle='--', lw=1.5,
                                    label="Pressure Gauge")
ax_ts1.set_xlabel("")
ax_ts1.set_ylabel("")
ax_ts1.grid(True, linestyle=':', alpha=0.6)
ax_ts1.set_xticklabels([])
ax_ts1.set_yticklabels([])
ax_ts1.tick_params(axis='x', bottom=False, top=False)
ax_ts1.tick_params(axis='y', left=False, right=False)

if np.any(np.isfinite(das_ch1_micro)):
    y_min_actual_das1 = np.nanmin(das_ch1_micro)
    y_max_actual_das1 = np.nanmax(das_ch1_micro)
    y_range1 = y_max_actual_das1 - y_min_actual_das1
    y_margin1 = y_range1 * 0.1 if y_range1 > 0 else (abs(y_min_actual_das1 * 0.1) if y_min_actual_das1 != 0 else 0.1)
    ax_ts1.set_ylim(y_min_actual_das1 - y_margin1, y_max_actual_das1 + y_margin1)
ax_ts1.legend(loc='upper right')

# Bottom-Left: Time-Series for Gauge 2 / Channel 2
ax_ts2 = fig.add_subplot(gs[1, 0:2])
das_ch2_micro = DASchan_strain[1] * micro_scale_factor
predicted_strain_g2_micro = predicted_strain_rate_g2_orig_units * micro_scale_factor

line_ts2_das, = ax_ts2.plot(DAStaxis, das_ch2_micro, color='dodgerblue', lw=1.5, label="LF-DAS")
line_ts2_pred_gauge, = ax_ts2.plot(DAStaxis, predicted_strain_g2_micro, color='red', linestyle='--', lw=1.5,
                                    label="Pressure Gauge")
ax_ts2.set_xlabel("")
ax_ts2.set_ylabel("")
ax_ts2.grid(True, linestyle=':', alpha=0.6)
ax_ts2.set_xticklabels([])
ax_ts2.set_yticklabels([])
ax_ts2.tick_params(axis='x', bottom=False, top=False)
ax_ts2.tick_params(axis='y', left=False, right=False)

if np.any(np.isfinite(das_ch2_micro)):
    y_min_actual_das2 = np.nanmin(das_ch2_micro)
    y_max_actual_das2 = np.nanmax(das_ch2_micro)
    y_range2 = y_max_actual_das2 - y_min_actual_das2
    y_margin2 = y_range2 * 0.1 if y_range2 > 0 else (abs(y_min_actual_das2 * 0.1) if y_min_actual_das2 != 0 else 0.1)
    ax_ts2.set_ylim(y_min_actual_das2 - y_margin2, y_max_actual_das2 + y_margin2)
# Legend removed as per request


# --- Right Side: Single Scatter Plot (SCALAR-FREE) ---
def plot_scatter_scalar_free(ax, x_data, y_data_orig_scale, beta_wls_coeffs, point_color='gray', point_label="Data Points"):
    y_data_micro = y_data_orig_scale * micro_scale_factor

    if len(x_data) > 0 and len(y_data_micro) == len(x_data):
        ax.scatter(x_data, y_data_micro, s=5, alpha=1.0, edgecolors='k', linewidths=0.2, color=point_color, label=point_label) # s=5, alpha=1.0
    elif len(x_data) > 0:
        # Optionally indicate data issue if needed, or just plot nothing
        pass
    else:
        # Optionally indicate no data
        pass

    if beta_wls_coeffs is not None and not np.isnan(beta_wls_coeffs).any() and len(x_data) > 1 :
        x_min, x_max = np.min(x_data), np.max(x_data)
        if x_min == x_max:
            x_range_for_line = np.array([x_min - 0.5, x_max + 0.5]) if x_min == 0 else np.array([x_min * 0.9, x_max * 1.1])
        else:
            x_range_for_line = np.array([x_min, x_max])
        y_fit_on_line_micro = (beta_wls_coeffs[0] * x_range_for_line + beta_wls_coeffs[1]) * micro_scale_factor
        ax.plot(x_range_for_line, y_fit_on_line_micro, color='black', linestyle='-', lw=2, label="WLS Fit")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', bottom=False, top=False)
    ax.tick_params(axis='y', left=False, right=False)
    ax.grid(True, linestyle=':', alpha=0.5)
    if len(x_data)>0 : ax.legend(loc='best')


ax_sc_combined_filtered = fig.add_subplot(gs[:, 2:])
plot_scatter_scalar_free(ax_sc_combined_filtered, x_filtered_actual, y_filtered_actual, beta_wls_filt_actual,
                         point_color='mediumblue', point_label="Filtered Data")

# No suptitle
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98, hspace=0.15, wspace=0.15) # Adjusted for no labels/titles

plt.savefig("figs/manuscript/pressure_strain_coupling/no_scalar.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nScript finished generating scalar-free plots.")