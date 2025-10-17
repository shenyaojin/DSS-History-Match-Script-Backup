# The plot script of scripts/well_leakage_history_matching/strain_pressure_coupling/101_ratio_estimation.py
# Enhanced visualization with complex figure layout (further revisions)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # For complex layouts
from fiberis.analyzer.Data1D import Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.analyzer.Geometry3D import DataG3D_md
import datetime
from scipy.stats import zscore

# %% Global parameters list
DASdata_path = "data/fiberis_format/s_well/DAS/LFDASdata_stg1_swell.npz"
gauge_path = "data/fiberis_format/s_well/gauges/gauge1_data_swell.npz"
gauge_path2 = "data/fiberis_format/s_well/gauges/gauge2_data_swell.npz"
gauge_md_path = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
frac_md_path = "data/fiberis_format/s_well/geometry/frac_hit/frac_hit_stage_1_swell.npz"

# %% Load data
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

# %% Do preprocessing
print("Preprocessing data...")
start_time = datetime.datetime(2020, 3, 16, 10, 36, 0)
end_time = datetime.datetime(2020, 3, 16, 10, 52, 0)
print(f"Original DAS Start Time (from data object): {getattr(DASdata, 'start_time', 'N/A')}")
DASdata.select_time(start_time, end_time)
DASdata.select_depth(15000, 16750)
gauge_data1.select_time(start_time, end_time)
gauge_data2.select_time(start_time, end_time)

# %% Get the DAS data channel
print("Extracting DAS channels and common time axis...")
DASchan = []
if gauge_md_data.data is not None and len(gauge_md_data.data) >= 2:
    DASchan.append(DASdata.get_value_by_depth(gauge_md_data.data[0]))
    DASchan.append(DASdata.get_value_by_depth(gauge_md_data.data[1]))
else:
    print("Warning: gauge_md_data might not be as expected. Using fallback DAS channels.")
    DASchan.append(DASdata.data[0, :] if DASdata.data.ndim == 2 and DASdata.data.shape[0] > 0 else (
        DASdata.data if DASdata.data.ndim == 1 else np.array([])))
    DASchan.append(DASdata.data[1, :] if DASdata.data.ndim == 2 and DASdata.data.shape[0] > 1 else (
        DASdata.data if DASdata.data.ndim == 1 else np.array([])))

DAStaxis = np.copy(DASdata.taxis)
if len(DAStaxis) > 0:
    DAStaxis -= DAStaxis[0]
else:
    print("Warning: DAStaxis is empty after DASdata.select_time.")
    if DASchan and hasattr(DASchan[0], 'shape') and len(DASchan[0]) > 0:  # Check if DASchan is populated
        DAStaxis = np.arange(len(DASchan[0]))

DASchan = np.array(DASchan)
# Ensure DASchan is 2D and handle potential length mismatches carefully
if DASchan.ndim == 1 and DASchan.size > 0:
    print("Warning: Only one DAS channel seems to be processed into DASchan. Reshaping for consistency.")
    if len(DASchan) == len(DAStaxis):
        DASchan = DASchan.reshape(1, -1)
        if DASchan.shape[0] < 2:
            print("Duplicating single DAS channel data for placeholder second channel.")
            DASchan = np.vstack([DASchan, DASchan])  # Make it 2-channel
    else:
        min_len = min(DASchan.size, len(DAStaxis))
        DASchan = DASchan[:min_len].reshape(1, -1)
        DAStaxis = DAStaxis[:min_len]
        if DASchan.shape[0] < 2:
            print("Duplicating single DAS channel data for placeholder second channel after trim.")
            DASchan = np.vstack([DASchan, DASchan])  # Make it 2-channel

elif DASchan.ndim == 2 and DASchan.shape[1] != len(DAStaxis) and len(DAStaxis) > 0:
    print(
        f"Warning: DAS channel length ({DASchan.shape[1]}) and DAStaxis length ({len(DAStaxis)}) mismatch. Attempting to trim/match.")
    min_len = min(DASchan.shape[1], len(DAStaxis))
    DASchan = DASchan[:, :min_len]
    DAStaxis = DAStaxis[:min_len]

# Ensure DASchan has 2 channels for subsequent indexing, even if they are identical due to fallback
if DASchan.ndim < 2 or DASchan.shape[0] < 2:
    print(
        "Error or Warning: DASchan does not have two distinct channels after processing. Plotting for channel 2 might be redundant or fail.")
    if DASchan.ndim == 1: DASchan = DASchan.reshape(1, -1)  # Reshape if it's 1D
    while DASchan.shape[0] < 2:  # If still less than 2 channels (e.g. was empty or 1 channel)
        if DASchan.size == 0:  # Was empty
            DASchan = np.full((2, len(DAStaxis) if len(DAStaxis) > 0 else 1), np.nan)  # Create two NaN channels
            print("DASchan was empty, created two NaN channels.")
        else:  # Was 1 channel
            DASchan = np.vstack([DASchan, np.full_like(DASchan[0], np.nan)])  # Add a NaN second channel
            print("Added a NaN placeholder for the second DAS channel.")

print("Interpolating gauge data to DAS time axis...")
gauge_data1.interpolate(DAStaxis, getattr(gauge_data1, 'start_time', None))
gauge_data2.interpolate(DAStaxis, getattr(gauge_data2, 'start_time', None))

# %% Post processing
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

# %% Prepare data for combined scatter plots and WLS
print("Preparing combined data for scatter plots...")
x_actual = np.concatenate([gauge_data1_grad.flatten(), gauge_data2_grad.flatten()])
y_actual = DASchan_strain.flatten()


# %% WLS Helper Function
def calculate_wls(x_data, y_data):
    """
    Calculates Weighted Least Squares (WLS) coefficients.
    Handles NaN values and potential LinAlgErrors gracefully.
    """
    # Initial check for invalid or insufficient input data
    if x_data is None or y_data is None or len(x_data) < 2 or len(y_data) < 2 or len(x_data) != len(y_data):
        print("Warning: Insufficient or mismatched data for WLS calculation. Returning NaN coefficients.")
        return np.array([np.nan, np.nan])

    # Filter out NaN values from both x_data and y_data
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_data_clean = x_data[valid_indices]
    y_data_clean = y_data[valid_indices]

    # Check if enough valid data remains after cleaning
    if len(x_data_clean) < 2:
        print("Warning: Not enough valid data points after NaN filtering for WLS. Returning NaN coefficients.")
        return np.array([np.nan, np.nan])

    # Construct the design matrix for OLS (Ordinary Least Squares)
    # X_matrix will have two columns: x_data_clean and a column of ones for the intercept
    X_matrix = np.vstack([x_data_clean, np.ones_like(x_data_clean)]).T

    try:
        # Step 1: Perform OLS to get initial coefficients and residuals
        # rcond=None ensures behavior consistent with recent NumPy versions
        beta_ols = np.linalg.lstsq(X_matrix, y_data_clean, rcond=None)[0]

        # Calculate residuals from OLS fit
        residuals = np.abs(y_data_clean - (X_matrix @ beta_ols))

        # Calculate weights for WLS: inversely proportional to absolute residuals
        # Add a small epsilon (1e-9) to avoid division by zero if a residual is exactly zero
        weights = 1.0 / (np.maximum(residuals, 1e-9))
        W_diag = np.diag(weights) # Create a diagonal weight matrix

        # Step 2: Calculate WLS coefficients
        # XTWX = X_transpose * W * X
        XTWX = X_matrix.T @ W_diag @ X_matrix
        # XTWy = X_transpose * W * y
        XTWy = X_matrix.T @ W_diag @ y_data_clean

        # Check the condition number of XTWX before solving
        # A very large condition number indicates an ill-conditioned matrix,
        # which can lead to unstable solutions or LinAlgErrors.
        # np.finfo(float).eps is machine epsilon, a very small number.
        if np.linalg.cond(XTWX) < 1.0 / np.finfo(float).eps:
            # If XTWX is well-conditioned, solve for WLS coefficients
            beta_wls = np.linalg.solve(XTWX, XTWy)
        else:
            # If XTWX is ill-conditioned, fall back to the OLS result
            print("Warning: XTWX matrix for WLS is ill-conditioned. Falling back to OLS result.")
            beta_wls = beta_ols
        return beta_wls

    except np.linalg.LinAlgError as e:
        # Catch specific Linear Algebra errors (e.g., SVD non-convergence, singular matrix)
        print(f"LinAlgError encountered during WLS calculation: {e}. Returning NaN coefficients.")
        return np.array([np.nan, np.nan])
    except Exception as e:
        # Catch any other unexpected errors during calculation
        print(f"An unexpected error occurred during WLS calculation: {e}. Returning NaN coefficients.")
        return np.array([np.nan, np.nan])


# %% Calculate WLS for necessary datasets
print("Calculating WLS fits...")

# --- Debugging WLS Inputs ---
# These print statements help inspect the data before it's passed to calculate_wls
print("\n--- Debugging WLS Inputs ---")
print(f"Shape of gauge_data1_grad: {gauge_data1_grad.shape}")
print(f"NaNs in gauge_data1_grad: {np.sum(np.isnan(gauge_data1_grad))}")
print(f"Infs in gauge_data1_grad: {np.sum(np.isinf(gauge_data1_grad))}")
print(f"All zeros in gauge_data1_grad: {np.all(gauge_data1_grad == 0)}")
print(f"Unique values in gauge_data1_grad: {np.unique(gauge_data1_grad).size}")

# Ensure DASchan_strain has at least two channels before trying to access index 0 and 1
if DASchan_strain.shape[0] > 0:
    print(f"Shape of DASchan_strain[0]: {DASchan_strain[0].shape}")
    print(f"NaNs in DASchan_strain[0]: {np.sum(np.isnan(DASchan_strain[0]))}")
    print(f"Infs in DASchan_strain[0]: {np.sum(np.isinf(DASchan_strain[0]))}")
    print(f"All zeros in DASchan_strain[0]: {np.all(DASchan_strain[0] == 0)}")
    print(f"Unique values in DASchan_strain[0]: {np.unique(DASchan_strain[0]).size}")
else:
    print("DASchan_strain has no channels.")

if DASchan_strain.shape[0] > 1:
    print(f"Shape of gauge_data2_grad: {gauge_data2_grad.shape}")
    print(f"NaNs in gauge_data2_grad: {np.sum(np.isnan(gauge_data2_grad))}")
    print(f"Infs in gauge_data2_grad: {np.sum(np.isinf(gauge_data2_grad))}")
    print(f"All zeros in gauge_data2_grad: {np.all(gauge_data2_grad == 0)}")
    print(f"Unique values in gauge_data2_grad: {np.unique(gauge_data2_grad).size}")

    print(f"Shape of DASchan_strain[1]: {DASchan_strain[1].shape}")
    print(f"NaNs in DASchan_strain[1]: {np.sum(np.isnan(DASchan_strain[1]))}")
    print(f"Infs in DASchan_strain[1]: {np.sum(np.isinf(DASchan_strain[1]))}")
    print(f"All zeros in DASchan_strain[1]: {np.all(DASchan_strain[1] == 0)}")
    print(f"Unique values in DASchan_strain[1]: {np.unique(DASchan_strain[1]).size}")
else:
    print("DASchan_strain has only one channel or is empty, skipping debug for DASchan_strain[1].")
print("---------------------------\n")
# --- End Debugging WLS Inputs ---


# WLS for individual gauges (for left panel predicted strain)
beta_wls_g1_raw = calculate_wls(gauge_data1_grad, DASchan_strain[0])
# Ensure DASchan_strain has at least two channels before trying to access index 1
beta_wls_g2_raw = calculate_wls(gauge_data2_grad, DASchan_strain[1] if DASchan_strain.shape[0] > 1 else np.array([]))


# Calculate predicted strain rates using individual gauge models
predicted_strain_rate_g1_orig_units = np.full_like(gauge_data1_grad, np.nan)
if not np.isnan(beta_wls_g1_raw).any() and gauge_data1_grad.size > 0:
    # Ensure gauge_data1_grad is not empty before multiplication
    predicted_strain_rate_g1_orig_units = beta_wls_g1_raw[0] * gauge_data1_grad + beta_wls_g1_raw[1]

predicted_strain_rate_g2_orig_units = np.full_like(gauge_data2_grad, np.nan)
if not np.isnan(beta_wls_g2_raw).any() and gauge_data2_grad.size > 0:
    # Ensure gauge_data2_grad is not empty before multiplication
    predicted_strain_rate_g2_orig_units = beta_wls_g2_raw[0] * gauge_data2_grad + beta_wls_g2_raw[1]

# Z-score filtering for the right-panel scatter plot
if len(x_actual) > 0 and len(y_actual) > 0:
    # Filter out NaNs from x_actual and y_actual before zscore calculation
    valid_actual_indices = ~np.isnan(x_actual) & ~np.isnan(y_actual)
    x_actual_clean_for_zscore = x_actual[valid_actual_indices]
    y_actual_clean_for_zscore = y_actual[valid_actual_indices]

    if len(x_actual_clean_for_zscore) > 1: # zscore needs at least 2 non-NaN values
        z_x_actual = zscore(x_actual_clean_for_zscore)
        z_y_actual = zscore(y_actual_clean_for_zscore)
        threshold_actual = 3.0
        mask_actual = (np.abs(z_x_actual) < threshold_actual) & (np.abs(z_y_actual) < threshold_actual)
        x_filtered_actual = x_actual_clean_for_zscore[mask_actual]
        y_filtered_actual = y_actual_clean_for_zscore[mask_actual]
    else:
        x_filtered_actual, y_filtered_actual = np.array([]), np.array([])
else:
    x_filtered_actual, y_filtered_actual = np.array([]), np.array([])

beta_wls_filt_actual = calculate_wls(x_filtered_actual, y_filtered_actual)  # For filtered combined data

# %% Plotting
print("Generating complex figure layout (revised right panel: filtered data, no time color, no coeffs)...")
micro_scale_factor = 1e6

plt.style.use('seaborn-v0_8-whitegrid')
common_rc_params = {
    'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
}
plt.rcParams.update(common_rc_params)

fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.45)

# --- Left Side: Time-Series Co-plots (Actual DAS vs. Predicted DAS on SAME Y-AXIS) ---
# Top-Left: Time-Series for Gauge 1 / Channel 1
ax_ts1 = fig.add_subplot(gs[0, 0:2])
das_ch1_micro = DASchan_strain[0] * micro_scale_factor
predicted_strain_g1_micro = predicted_strain_rate_g1_orig_units * micro_scale_factor

line_ts1_das, = ax_ts1.plot(DAStaxis, das_ch1_micro, color='dodgerblue', lw=1.5, label="Actual DAS Ch1")
line_ts1_pred_gauge, = ax_ts1.plot(DAStaxis, predicted_strain_g1_micro, color='red', linestyle='--', lw=1.5,
                                   label="Predicted DAS (G1)")

# ax_ts1.set_xlabel("Time (s)") # Removed as per request
ax_ts1.set_ylabel("Strain Rate (µε/s)")
ax_ts1.grid(True, linestyle=':', alpha=0.6)
ax_ts1.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3), useMathText=True)
# ax_ts1.set_title("Ch1: Actual vs. Predicted DAS Strain Rate") # Removed title
ax_ts1.set_xticklabels([])  # Remove x-tick labels
ax_ts1.tick_params(axis='x', bottom=False)  # Remove x-ticks

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

line_ts2_das, = ax_ts2.plot(DAStaxis, das_ch2_micro, color='dodgerblue', lw=1.5,
                             label="Actual DAS Ch2")  # Changed color for consistency
line_ts2_pred_gauge, = ax_ts2.plot(DAStaxis, predicted_strain_g2_micro, color='red', linestyle='--', lw=1.5,
                                   # Changed color
                                   label="Predicted DAS (G2)")

ax_ts2.set_xlabel("Time (s)")
ax_ts2.set_ylabel("Strain Rate (µε/s)")
ax_ts2.grid(True, linestyle=':', alpha=0.6)
ax_ts2.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3), useMathText=True)
# ax_ts2.set_title("Ch2: Actual vs. Predicted DAS Strain Rate") # Removed title

if np.any(np.isfinite(das_ch2_micro)):
    y_min_actual_das2 = np.nanmin(das_ch2_micro)
    y_max_actual_das2 = np.nanmax(das_ch2_micro)
    y_range2 = y_max_actual_das2 - y_min_actual_das2
    y_margin2 = y_range2 * 0.1 if y_range2 > 0 else (abs(y_min_actual_das2 * 0.1) if y_min_actual_das2 != 0 else 0.1)
    ax_ts2.set_ylim(y_min_actual_das2 - y_margin2, y_max_actual_das2 + y_margin2)


# ax_ts2.legend(loc='upper right') # Removed legend


# --- Right Side: Single Scatter Plot for Combined Z-score Filtered Data ---
def plot_scatter_wls_filtered_custom(ax, x_data, y_data_orig_scale, beta_wls_coeffs, point_color='gray'):
    """
    Plots a scatter plot with WLS fit line for filtered data.
    """
    y_data_micro = y_data_orig_scale * micro_scale_factor

    if len(x_data) > 0 and len(y_data_micro) == len(x_data):
        ax.scatter(x_data, y_data_micro, s=5, alpha=1.0, edgecolors='k', linewidths=0.2, color=point_color,
                   label="Filtered Data Points")  # s=5, alpha=1.0
    elif len(x_data) > 0:
        ax.text(0.5, 0.5, "Data length mismatch", ha='center', va='center', transform=ax.transAxes, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data after filtering", ha='center', va='center', transform=ax.transAxes, fontsize=8)

    if beta_wls_coeffs is not None and not np.isnan(beta_wls_coeffs).any() and len(x_data) > 1:
        x_min, x_max = np.min(x_data), np.max(x_data)
        if x_min == x_max:  # Handle case where all x values are the same
            # Create a small range around x_min for plotting the line
            x_range_for_line = np.array([x_min - 0.5, x_max + 0.5]) if x_min == 0 else np.array(
                [x_min * 0.9, x_max * 1.1])
        else:
            x_range_for_line = np.array([x_min, x_max])

        y_fit_on_line_micro = (beta_wls_coeffs[0] * x_range_for_line + beta_wls_coeffs[1]) * micro_scale_factor
        ax.plot(x_range_for_line, y_fit_on_line_micro, color='black', linestyle='-', lw=2, label="WLS Fit")

    ax.set_xlabel("Pressure Gradient (psi/s)")
    ax.set_ylabel("Strain Rate (µε/s)")
    # ax.set_title(title, fontsize=10) # Title removed
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3), useMathText=True)
    if len(x_data) > 0: ax.legend(loc='best')


ax_sc_combined_filtered = fig.add_subplot(gs[:, 2:])  # Spans both rows and the last 2 columns
plot_scatter_wls_filtered_custom(ax_sc_combined_filtered, x_filtered_actual, y_filtered_actual, beta_wls_filt_actual,
                                 point_color='mediumblue')  # Changed point color for distinction

# plt.suptitle("DAS Strain Rate and Pressure Data Analysis", fontsize=14, y=0.98) # Removed suptitle
plt.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.95, hspace=0.5,
                    wspace=0.45)  # Adjusted top due to suptitle removal

plt.savefig("figs/manuscript/pressure_strain_coupling/scalar_fixed.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nScript finished generating plots.")

#%% Calculate the ratio of LFDAS to gauge data

print("coefficients strain vs gauge pressure", beta_wls_filt_actual[0])