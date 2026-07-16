"""Export the ISOLATED DDM shear component (fault2) for the V1 MOOSE+DDM workflow.

The 07152026 package only contains the TOTAL two-fault DDM strain (fault1 tensile +
fault2 shear). The residual workflow needs the shear-only profile so that
    tensile_target = observed - DDM_shear.
This re-runs the modeling notebook's two-fault cell (cell 15) HEADLESS, using the
local well geometry, and exports fault1 (tensile), fault2 (shear), and total
separately -- 4h mean profiles + full waterfalls, in the SAME format/units/reference
as the 07152026 exports (nanostrain/s, millistrain, T1-referenced, MD 10200-10500).

Fault centre: the notebook pulls it from pumping_windows_stage_with_coordinates_xyz.csv
row 5, which is not in this repo; per the notebook comment it sits at well MD ~10340,
so we place it at the well point interpolated at MD 10340. We VALIDATE this by
reproducing the TOTAL and comparing against the existing 07152026 total export.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "tensile_fault" / "data_transfer" / "from_pc"))
from DDMpy_log import Well, DynamicFracture, StrainRateUtil  # noqa: E402

WELL_CSV = REPO / "data_fervo" / "legacy" / "Gold_4_PB_Well_Geometry.csv"
REF_DIR = REPO / "data_fervo" / "legacy" / "07152026"          # existing TOTAL export (validation)
OUT_DIR = REPO / "data_fervo" / "legacy" / "07152026_decomposed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- geometry / history (verbatim from modeling notebook cell 15) ------------
MD_CENTER = 10340.0
fault1_strike, fault2_strike, dip = -0.8, 0.6, 90
fault1_L, fault2_L, fracture_height = 3840, 3800, 4000
W_max, S1_max = 0.008, -0.03

T0 = datetime(2025, 2, 24, 0, 0)
T1 = datetime(2025, 2, 24, 11, 0)
T2 = datetime(2025, 2, 28, 0, 0)
T3 = datetime(2025, 3, 3, 22, 0)
mins = lambda dt: (dt - T0).total_seconds() / 60.0
t1, t2, t3 = mins(T1), mins(T2), mins(T3)

N = 144
delta_t = t3 / N
N1 = int(t1 // delta_t)
N2 = int((t2 - t1) // delta_t + 1)
N3 = int((t3 - t2 + 1) // delta_t + 1)
taxis = np.concatenate([np.linspace(0, t1 - delta_t, N1),
                        np.linspace(t1, t2 - delta_t, N2),
                        np.linspace(t2, t3, N3)])
fault1_length = np.full_like(taxis, fault1_L)
fault2_length = np.full_like(taxis, fault2_L)
height = np.full_like(taxis, fracture_height)

# fault 1: tensile only, T1->T3   |   fault 2: shear only, T2->T3
fault1_width = np.concatenate([np.zeros(N1),
                               W_max * (np.linspace(0, 1, N2) ** 2),
                               np.linspace(W_max, W_max * 3, N3 + 1)[1:]])
fault1_shear = np.zeros_like(taxis)
fault2_width = np.zeros_like(taxis)
fault2_shear = np.concatenate([np.zeros(N1 + N2), np.linspace(0, S1_max, N3)])

# ---- well + fault centre -----------------------------------------------------
df = pd.read_csv(WELL_CSV).sort_values("MD")
control_points = df[["x_gold", "y_gold", "z_gold"]].values
well = Well.set_well_by_points(control_points, N=len(control_points) * 10, smooth=31)
well.gauge_length = 30.48  # ft

md_csv = df["MD"].to_numpy(float)
x = float(np.interp(MD_CENTER, md_csv, df["x_gold"].to_numpy(float)))
y = float(np.interp(MD_CENTER, md_csv, df["y_gold"].to_numpy(float)))
z = float(np.interp(MD_CENTER, md_csv, df["z_gold"].to_numpy(float)))
print(f"Fault centre @ MD {MD_CENTER:.0f}: (x,y,z)=({x:.1f},{y:.1f},{z:.1f}) ft")


def run_fault(strike, length_hist, width_hist, shear_hist):
    fr = DynamicFracture.GlobalRectangularFracture()
    fr.set_global_coors(strike, dip, x, y, z)
    fr.define_LHW(taxis=taxis, length=length_hist, height=height,
                  width=width_hist, S1=shear_hist, S2=None)
    fr.set_monitor_wells(well)
    fr.calculate()
    return fr.gather_strain_data()[0]


sd1 = run_fault(fault1_strike, fault1_length, fault1_width, fault1_shear)   # tensile
sd2 = run_fault(fault2_strike, fault2_length, fault2_width, fault2_shear)   # shear
print("DDM done. strain array shape (MD x time):", np.asarray(sd1.data).shape)

taxis_sec = np.asarray(sd1.taxis, dtype=float) * 60.0
all_times = pd.DatetimeIndex([T0 + timedelta(minutes=float(v)) for v in np.asarray(sd1.taxis)])
all_mds = np.asarray(sd1.daxis, dtype=float)

PLOT_START, PLOT_END = pd.Timestamp(T1), pd.Timestamp(T3)
MD_TOP, MD_BOT = 10200, 10500
tmask = (all_times >= PLOT_START) & (all_times <= PLOT_END)
dmask = (all_mds >= MD_TOP) & (all_mds <= MD_BOT)
Time_plot = all_times[tmask]
Depth_plot = all_mds[dmask]
print(f"Depth axis in window: {Depth_plot.min():.1f}-{Depth_plot.max():.1f} ft, {Depth_plot.size} pts")


def strain_and_rate(sd):
    """Return (direct_strain_mstrain[MD,t], rate_nstrain_s[MD,t]) cropped to window."""
    data = np.asarray(sd.data, dtype=float)
    rate = StrainRateUtil.centered_time_diff(data, taxis_sec)
    strain_ms = data[dmask, :][:, tmask] * 1e3
    rate_ns = rate[dmask, :][:, tmask] * 1e9
    return strain_ms, rate_ns


def mean_profiles(mat, times, interval="4h"):
    times = pd.DatetimeIndex(times)
    profs, centers = [], []
    for ws in pd.date_range(PLOT_START, PLOT_END, freq=interval):
        we = min(ws + pd.Timedelta(interval), PLOT_END)
        if ws >= PLOT_END:
            continue
        m = (times >= ws) & (times < we)
        if not m.any():
            continue
        profs.append(np.nanmean(mat[:, m], axis=1))
        centers.append(ws)
    return (np.column_stack(profs) if profs else np.empty((mat.shape[0], 0))), pd.DatetimeIndex(centers)


def t1_reference(mat_or_prof):
    """Subtract the first 4h-window column so each component is 0 at T1 (same as the total export)."""
    if mat_or_prof.shape[1] == 0:
        return mat_or_prof
    return mat_or_prof - mat_or_prof[:, [0]]


def export_depth_time(mat, depths, times, path):
    cols = {"measured_depth_ft": np.asarray(depths, float)}
    cols.update({pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"): mat[:, i]
                 for i, t in enumerate(pd.DatetimeIndex(times))})
    pd.DataFrame(cols).to_csv(path, index=False)


def export_profiles(mat, centers, depths, path, meta_path, units, source, p95_mult=10.0):
    cols = {"measured_depth_ft": np.asarray(depths, float)}
    cols.update({pd.Timestamp(c).strftime("%Y-%m-%d %H:%M:%S"): mat[:, i]
                 for i, c in enumerate(centers)})
    pd.DataFrame(cols).to_csv(path, index=False)
    p95 = np.nanpercentile(np.abs(mat), 95) if mat.size else np.nan
    pd.DataFrame({
        "window_start": centers, "profile_center_time": centers,
        "profile_units": units, "profile_source": source,
        "overlay_half_width_hours": 28.0, "overlay_scale_multiplier": p95_mult,
        "profile_p95_abs": p95, "overlay_scale_reference_abs": p95 * p95_mult,
    }).to_csv(meta_path, index=False)


SUFFIX = "20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref"
components = {
    "SHEAR": (sd2, "two-fault fault2 shear-only DDM"),
    "TENSILE": (sd1, "two-fault fault1 tensile-only DDM"),
}
# total = tensile + shear
total_strain_ms = strain_and_rate(sd1)[0] + strain_and_rate(sd2)[0]
total_rate_ns = strain_and_rate(sd1)[1] + strain_and_rate(sd2)[1]

for tag, (sd, source) in components.items():
    strain_ms, rate_ns = strain_and_rate(sd)
    strain_ref = t1_reference(strain_ms)
    # waterfalls
    export_depth_time(strain_ref, Depth_plot, Time_plot,
                      OUT_DIR / f"two_fault_{tag}_direct_strain_waterfall_{SUFFIX}.csv")
    export_depth_time(rate_ns, Depth_plot, Time_plot,
                      OUT_DIR / f"two_fault_{tag}_rate_waterfall_{SUFFIX}.csv")
    # 4h profiles
    sp, sc = mean_profiles(strain_ref, Time_plot)
    rp, rc = mean_profiles(rate_ns, Time_plot)
    export_profiles(sp, sc, Depth_plot,
                    OUT_DIR / f"two_fault_{tag}_direct_strain_4h_{SUFFIX}.csv",
                    OUT_DIR / f"two_fault_{tag}_direct_strain_4h_{SUFFIX}_metadata.csv",
                    "millistrain", source)
    export_profiles(rp, rc, Depth_plot,
                    OUT_DIR / f"two_fault_{tag}_rate_4h_{SUFFIX}.csv",
                    OUT_DIR / f"two_fault_{tag}_rate_4h_{SUFFIX}_metadata.csv",
                    "nanostrain/s", source.replace("DDM", "strain-rate"))
    print(f"  exported {tag}: strain peak {np.nanmax(np.abs(sp)):+.4f} mε, rate peak {np.nanmax(np.abs(rp)):.4f} nε/s")

# ---- VALIDATION: reproduced TOTAL vs existing 07152026 total -----------------
print("\n=== VALIDATION vs existing 07152026 total export ===")
my_total_strain_prof, mc = mean_profiles(t1_reference(total_strain_ms), Time_plot)
ref = pd.read_csv(REF_DIR / f"two_fault_direct_strain_4h_{SUFFIX}.csv")
ref_depth = ref["measured_depth_ft"].to_numpy(float)
ref_mat = ref.iloc[:, 1:].to_numpy(float)
# interpolate mine onto the reference depth grid, compare last snapshot & peak
mine_interp = np.array([np.interp(ref_depth, Depth_plot, my_total_strain_prof[:, j])
                        for j in range(my_total_strain_prof.shape[1])]).T
n = min(mine_interp.shape[1], ref_mat.shape[1])
diff = mine_interp[:, :n] - ref_mat[:, :n]
denom = np.nanmax(np.abs(ref_mat[:, :n]))
print(f"reference peak |strain| = {denom:.4f} mε; my reproduced peak = {np.nanmax(np.abs(mine_interp)):.4f} mε")
print(f"max |my_total - ref_total| = {np.nanmax(np.abs(diff)):.4f} mε "
      f"({100*np.nanmax(np.abs(diff))/denom:.1f}% of peak)")
# peak-depth check on the last snapshot
j = n - 1
print(f"last-snapshot peak MD: mine={ref_depth[np.nanargmax(np.abs(mine_interp[:, j]))]:.0f} ft, "
      f"ref={ref_depth[np.nanargmax(np.abs(ref_mat[:, j]))]:.0f} ft")
print(f"\nSaved decomposed exports to: {OUT_DIR}")
