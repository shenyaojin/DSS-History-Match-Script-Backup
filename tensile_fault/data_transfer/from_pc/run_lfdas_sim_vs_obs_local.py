"""Local-adapted run of LFDAS_NPY_Vis_07-13-2026.ipynb.

Reproduces the two matplotlib-only cells that DO NOT need the raw DAS NPY / DASCore:

  * "Simulation within V1 mds range"   (notebook cell 19)
  * "Observation versus Simulation"    (notebook cell 21, simulated-background figure)

Why adapted
-----------
The original notebook (cells 0-17) loads raw LF-DAS NPY strain-rate files from a
Windows OneDrive path and integrates them with DASCore.  On this machine:
  * only Jan/Feb NPY are present (NPY_G4-PB_MM_UTC_202503 is missing), so the
    observed T1->T3 waterfall cannot be rebuilt from scratch, and
  * DASCore is not importable (missing typing_extensions and other deps).
But the observed 4-hour mean profiles were already exported to data_fervo/legacy/
and the two-fault DDM simulation exports live in data_fervo/legacy/07152026/.
Cells 19 and 21 are pure matplotlib, so we point them at those local CSVs.

Data used
---------
Simulated (data_fervo/legacy/07152026/):
  two_fault_rate_waterfall_*         -> simulated strain-rate waterfall (nanostrain/s)
  two_fault_direct_strain_waterfall_*-> simulated strain waterfall (millistrain, T1-ref)
  two_fault_rate_4h_*                -> simulated 4-h mean strain-rate profiles
  two_fault_direct_strain_4h_*       -> simulated 4-h mean strain profiles (T1-ref)
Observed (data_fervo/legacy/):
  strain_rate_4h_mean_profiles_*_1500_*_T1_ref -> observed 4-h mean strain-rate profiles
  strain_4h_mean_profiles_*_1500_*_T1_ref      -> observed 4-h mean strain profiles (T1-ref)

Profile "metadata" (window centre times) are taken from the CSV column headers,
so no separate *_metadata.csv is required.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- paths -------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[4]
SIM_DIR = REPO / "data_fervo" / "legacy" / "07152026"
OBS_DIR = REPO / "data_fervo" / "legacy"
OUT_DIR = REPO / "figs" / "tensile_fault_qc" / "lfdas_sim_vs_obs_07152026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- display convention (matches the notebook) -------------------------------
PROFILE_SCALE_MULTIPLIER = 10.0
RATE_OVERLAY_HALF_WIDTH_HOURS = 28.0
STRAIN_OVERLAY_HALF_WIDTH_HOURS = 28.0
RATE_COLOR_SCALE = (-0.3, 0.3)     # nanostrain/s
STRAIN_COLOR_SCALE = (-0.1, 0.1)   # millistrain
PLOT_START = pd.to_datetime("2025-02-24 11:00")   # T1
PLOT_END = pd.to_datetime("2025-03-03 22:00")     # T3
MD_TOP, MD_BOTTOM = 10200, 10500
MARK_TIMES = pd.to_datetime(["2025-02-24 11:00", "2025-02-28 00:00", "2025-03-03 22:00"])
MARK_LABELS = ["T1", "T2", "T3"]
TICK_TIMES = pd.date_range(PLOT_START, PLOT_END, freq="24h")


# --- file discovery ----------------------------------------------------------
def one(folder, pattern, exclude_metadata=True):
    files = [p for p in folder.glob(pattern)
             if not (exclude_metadata and p.name.endswith("_metadata.csv"))]
    if not files:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}")
    return max(files, key=lambda p: p.stat().st_mtime)


sim_rate_waterfall_csv = one(SIM_DIR, "two_fault_rate_waterfall_*.csv")
sim_strain_waterfall_csv = one(SIM_DIR, "two_fault_direct_strain_waterfall_*.csv")
sim_rate_profile_csv = one(SIM_DIR, "two_fault_rate_4h_*.csv")
sim_strain_profile_csv = one(SIM_DIR, "two_fault_direct_strain_4h_*.csv")
obs_rate_profile_csv = one(OBS_DIR, "strain_rate_4h_mean_profiles_*_1500_*_T1_ref.csv")
obs_strain_profile_csv = one(OBS_DIR, "strain_4h_mean_profiles_*_1500_*_T1_ref.csv")


# --- readers -----------------------------------------------------------------
def read_depth_time_csv(csv_path):
    df = pd.read_csv(csv_path)
    depth = df["measured_depth_ft"].to_numpy(float)
    time = pd.DatetimeIndex(pd.to_datetime(df.columns[1:]))
    data = df.iloc[:, 1:].to_numpy(float)
    return data, depth, time


def read_profile_csv(csv_path):
    """Profile matrix (depth x window) + centre times from the column headers."""
    df = pd.read_csv(csv_path)
    depth = df["measured_depth_ft"].to_numpy(float)
    centre_times = pd.DatetimeIndex(pd.to_datetime(df.columns[1:]))
    profiles = df.iloc[:, 1:].to_numpy(float)
    return profiles, centre_times, depth


def combined_profile_abs_scale(*matrices, multiplier=1.0):
    vals = []
    for m in matrices:
        if m.size:
            fv = np.abs(m[np.isfinite(m)])
            if fv.size:
                vals.append(fv)
    if not vals:
        return np.nan, np.nan
    combined = np.concatenate(vals)
    p95 = np.nanpercentile(combined, 95)
    if not np.isfinite(p95) or p95 == 0:
        p95 = np.nanmax(combined)
    return p95, p95 * multiplier


# --- plotting helpers --------------------------------------------------------
def overlay_profiles(ax, matrix, centre_times, depth, half_width_hours,
                     scale_ref, color, linestyle, label, linewidth=0.9, alpha=0.95):
    if matrix.size == 0 or not np.isfinite(scale_ref) or scale_ref == 0:
        return
    sec_per_unit = half_width_hours * 3600.0 / scale_ref
    for col_idx, centre in enumerate(centre_times):
        profile = matrix[:, col_idx]
        finite = np.isfinite(profile)
        if not finite.any():
            continue
        x = pd.Timestamp(centre) + pd.to_timedelta(profile[finite] * sec_per_unit, unit="s")
        ax.plot(x, depth[finite], color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, zorder=5)
    ax.plot([], [], color=color, linestyle=linestyle, linewidth=1.8, label=label)


def plot_waterfall(ax, data, time, depth, cmap_name, color_scale, cbar_label, title):
    x0 = mdates.date2num(pd.Timestamp(time[0]).to_pydatetime())
    x1 = mdates.date2num(pd.Timestamp(time[-1]).to_pydatetime())
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("0.82")
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=color_scale[0], vmax=color_scale[1],
                   extent=(x0, x1, depth[-1], depth[0]), interpolation="none")
    ax.xaxis_date()
    ax.set_ylabel("Measured Depth (ft)")
    ax.set_ylim(MD_BOTTOM, MD_TOP)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    return im


def format_axis(ax, legend=True):
    for label, t in zip(MARK_LABELS, MARK_TIMES):
        ax.axvline(t, color="gold", linestyle="--", linewidth=1.4, alpha=0.95, zorder=4)
        ax.text(t, 1.01, label, color="gold", fontsize=11, fontweight="bold",
                ha="center", va="bottom", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlim(PLOT_START, PLOT_END)
    ax.set_xticks(TICK_TIMES)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax.grid(color="white", alpha=0.18, linewidth=0.4)
    if legend:
        ax.legend(loc="lower right")


# --- load --------------------------------------------------------------------
SimRate, SimRateDepth, SimRateTime = read_depth_time_csv(sim_rate_waterfall_csv)
SimStrain, SimStrainDepth, SimStrainTime = read_depth_time_csv(sim_strain_waterfall_csv)
SimRateProf, SimRateProfT, SimRateProfDepth = read_profile_csv(sim_rate_profile_csv)
SimStrainProf, SimStrainProfT, SimStrainProfDepth = read_profile_csv(sim_strain_profile_csv)
ObsRateProf, ObsRateProfT, ObsRateProfDepth = read_profile_csv(obs_rate_profile_csv)
ObsStrainProf, ObsStrainProfT, ObsStrainProfDepth = read_profile_csv(obs_strain_profile_csv)

rate_p95, rate_scale = combined_profile_abs_scale(ObsRateProf, SimRateProf,
                                                  multiplier=PROFILE_SCALE_MULTIPLIER)
strain_p95, strain_scale = combined_profile_abs_scale(ObsStrainProf, SimStrainProf,
                                                      multiplier=PROFILE_SCALE_MULTIPLIER)


# --- Figure A: simulation only (notebook cell 19) ----------------------------
figA, (axA1, axA2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
plot_waterfall(axA1, SimRate, SimRateTime, SimRateDepth, "bwr", RATE_COLOR_SCALE,
               "Strain rate (nanostrain/s)",
               "Simulated strain-rate waterfall with 4-hour mean profiles")
_, sim_rate_scale = combined_profile_abs_scale(SimRateProf, multiplier=PROFILE_SCALE_MULTIPLIER)
overlay_profiles(axA1, SimRateProf, SimRateProfT, SimRateProfDepth,
                 RATE_OVERLAY_HALF_WIDTH_HOURS, sim_rate_scale, "black", "-", "Simulated rate")
plot_waterfall(axA2, SimStrain, SimStrainTime, SimStrainDepth, "seismic", STRAIN_COLOR_SCALE,
               "Strain (millistrain)",
               "Simulated strain waterfall with 4-hour mean profiles (T1 referenced)")
_, sim_strain_scale = combined_profile_abs_scale(SimStrainProf, multiplier=PROFILE_SCALE_MULTIPLIER)
overlay_profiles(axA2, SimStrainProf, SimStrainProfT, SimStrainProfDepth,
                 STRAIN_OVERLAY_HALF_WIDTH_HOURS, sim_strain_scale, "black", "-", "Simulated strain")
for ax in (axA1, axA2):
    format_axis(ax)
axA2.set_xlabel("Time [UTC-7]")
figA.savefig(OUT_DIR / "sim_only_waterfall_profiles.png", dpi=130)
print("Saved:", OUT_DIR / "sim_only_waterfall_profiles.png")


# --- Figure B: observation vs simulation on simulated background (cell 21) ----
figB, (axB1, axB2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
plot_waterfall(axB1, SimRate, SimRateTime, SimRateDepth, "bwr", RATE_COLOR_SCALE,
               "Strain rate (nanostrain/s)",
               "Simulated strain-rate waterfall with observed and simulated 4-hour profiles")
overlay_profiles(axB1, ObsRateProf, ObsRateProfT, ObsRateProfDepth,
                 RATE_OVERLAY_HALF_WIDTH_HOURS, rate_scale, "black", "-", "Observed rate profiles")
overlay_profiles(axB1, SimRateProf, SimRateProfT, SimRateProfDepth,
                 RATE_OVERLAY_HALF_WIDTH_HOURS, rate_scale, "green", "--", "Simulated rate profiles")
plot_waterfall(axB2, SimStrain, SimStrainTime, SimStrainDepth, "seismic", STRAIN_COLOR_SCALE,
               "Strain (millistrain)",
               "Simulated T1-referenced strain waterfall with observed and simulated 4-hour profiles")
overlay_profiles(axB2, ObsStrainProf, ObsStrainProfT, ObsStrainProfDepth,
                 STRAIN_OVERLAY_HALF_WIDTH_HOURS, strain_scale, "black", "-", "Observed strain profiles")
overlay_profiles(axB2, SimStrainProf, SimStrainProfT, SimStrainProfDepth,
                 STRAIN_OVERLAY_HALF_WIDTH_HOURS, strain_scale, "green", "--", "Simulated strain profiles")
for ax in (axB1, axB2):
    format_axis(ax)
axB2.set_xlabel("Time [UTC-7]")
figB.savefig(OUT_DIR / "obs_vs_sim_on_sim_background.png", dpi=130)
print("Saved:", OUT_DIR / "obs_vs_sim_on_sim_background.png")


# --- Figure C (bonus): direct obs-vs-sim profile line comparison near T3 ------
def profile_at(matrix, times, target):
    idx = int(np.argmin(np.abs(times - pd.Timestamp(target))))
    return matrix[:, idx], times[idx]

figC, (axC1, axC2) = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
target = pd.to_datetime("2025-03-03 00:00")
op, ot = profile_at(ObsRateProf, ObsRateProfT, target)
sp, st = profile_at(SimRateProf, SimRateProfT, target)
axC1.plot(op, ObsRateProfDepth, "k-", lw=1.6, label=f"Observed ({ot:%m-%d %H:%M})")
axC1.plot(sp, SimRateProfDepth, "g--", lw=1.6, label=f"Simulated ({st:%m-%d %H:%M})")
axC1.set(xlabel="Strain rate (nanostrain/s)", ylabel="Measured Depth (ft)",
         title="Strain-rate profile near 03-03")
axC1.set_ylim(MD_BOTTOM, MD_TOP); axC1.grid(alpha=0.3); axC1.legend()
op, ot = profile_at(ObsStrainProf, ObsStrainProfT, target)
sp, st = profile_at(SimStrainProf, SimStrainProfT, target)
axC2.plot(op, ObsStrainProfDepth, "k-", lw=1.6, label=f"Observed ({ot:%m-%d %H:%M})")
axC2.plot(sp, SimStrainProfDepth, "g--", lw=1.6, label=f"Simulated ({st:%m-%d %H:%M})")
axC2.set(xlabel="Strain (millistrain)", title="Strain profile near 03-03")
axC2.set_ylim(MD_BOTTOM, MD_TOP); axC2.grid(alpha=0.3); axC2.legend()
figC.savefig(OUT_DIR / "obs_vs_sim_profile_lines_near_T3.png", dpi=130)
print("Saved:", OUT_DIR / "obs_vs_sim_profile_lines_near_T3.png")


# --- report ------------------------------------------------------------------
print("\n=== Observation / simulation comparison complete ===")
print(f"Sim rate waterfall : {sim_rate_waterfall_csv.name}  {SimRate.shape}")
print(f"Sim strain waterfall: {sim_strain_waterfall_csv.name}  {SimStrain.shape}")
print(f"Obs rate profiles  : {obs_rate_profile_csv.name}  {ObsRateProf.shape}")
print(f"Obs strain profiles: {obs_strain_profile_csv.name}  {ObsStrainProf.shape}")
print(f"Shared rate scale  : +/-{rate_scale:.4g} nanostrain/s over +/-{RATE_OVERLAY_HALF_WIDTH_HOURS:g} h")
print(f"Shared strain scale: +/-{strain_scale:.4g} millistrain over +/-{STRAIN_OVERLAY_HALF_WIDTH_HOURS:g} h")
print(f"Figures written to : {OUT_DIR}")
