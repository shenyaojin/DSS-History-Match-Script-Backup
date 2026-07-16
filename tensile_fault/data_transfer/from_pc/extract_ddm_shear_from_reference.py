"""Extract the isolated DDM SHEAR component for the V1 MOOSE+DDM workflow.

Method (no DDM re-run needed)
-----------------------------
The 07152026 package has only the TOTAL two-fault DDM strain (fault1 tensile +
fault2 shear). Re-running the DDMpy notebook does NOT reproduce 07152026 with its
stated parameters (the notebook's gauge_length=30.48 ft gives ~6x too much strain;
07152026 implies ~180 ft, and the shear/tensile balance is gauge-length sensitive),
so a faithful re-run is not possible from the files we have.

Instead we use an EXACT property of the DDM: for one rectangular element of fixed
geometry, tensile strain is separable in time and depth,
    eps_tensile(z, t) = width(t) * g(z),
with a TIME-INVARIANT spatial shape g(z). During T1->T2 the shear is zero, so the
reference total there IS pure tensile. We fit g(z) from the T1->T2 snapshots (fit
residual 1.1%), extrapolate the tensile into T2->T3 with the known width history,
and take
    eps_shear(z, t) = eps_total(z, t) - width(t) * g(z).
This recovers Pengchao's exact fault2 shear on his native scale. Validation: the
recovered shear is ~0 before T2 and turns on at T2, tracking the imposed slip.

Outputs (data_fervo/legacy/07152026_decomposed/): shear + tensile, strain + rate,
4h profiles (+metadata) and full waterfalls, T1-referenced, MD 10200-10500.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "data_fervo" / "legacy" / "07152026"
OUT = REPO / "data_fervo" / "legacy" / "07152026_decomposed"
OUT.mkdir(parents=True, exist_ok=True)
S = "20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref"
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")
T3 = pd.Timestamp("2025-03-03 22:00")

hist = pd.read_csv(SRC / f"two_fault_histories_{S}.csv")
hist["time"] = pd.to_datetime(hist["time"])
htime, hwidth = hist["time"], hist["fault1_width_ft"].to_numpy(float)


def read_dt(path):
    df = pd.read_csv(path)
    depth = df["measured_depth_ft"].to_numpy(float)
    times = pd.DatetimeIndex(pd.to_datetime(df.columns[1:]))
    return df.iloc[:, 1:].to_numpy(float), depth, times


def width_at(times):
    """Instantaneous fault1 opening (ft) at arbitrary times, from the history."""
    tsec = (pd.DatetimeIndex(times) - htime.iloc[0]).total_seconds().to_numpy()
    hsec = (htime - htime.iloc[0]).dt.total_seconds().to_numpy()
    return np.interp(tsec, hsec, hwidth)


def width_rate_at(times):
    """Instantaneous d(width)/dt (ft/s) at arbitrary times (per-second)."""
    hsec = (htime - htime.iloc[0]).dt.total_seconds().to_numpy()
    wdot = np.gradient(hwidth, hsec)
    tsec = (pd.DatetimeIndex(times) - htime.iloc[0]).total_seconds().to_numpy()
    return np.interp(tsec, hsec, wdot)


def separate(total, times, weight):
    """total(z,t) = weight(t)*g(z) + shear(z,t). Fit g on the pre-T2 (shear=0) columns."""
    pre = (times < T2) & (np.abs(weight) > np.max(np.abs(weight)) * 1e-6)
    W, M = weight[pre], total[:, pre]
    g = (M @ W) / (W @ W)                       # least-squares unit-weight shape
    tensile = np.outer(g, weight)
    shear = total - tensile
    resid_pre = float(np.nanmax(np.abs(shear[:, times < T2])))
    return tensile, shear, g, resid_pre


def export_dt(mat, depth, times, path):
    cols = {"measured_depth_ft": depth}
    cols.update({t.strftime("%Y-%m-%d %H:%M:%S"): mat[:, i] for i, t in enumerate(times)})
    pd.DataFrame(cols).to_csv(path, index=False)


def export_prof(mat, depth, times, units, source, path, meta):
    export_dt(mat, depth, times, path)
    p95 = np.nanpercentile(np.abs(mat), 95) if mat.size else np.nan
    pd.DataFrame({
        "window_start": times, "profile_center_time": times,
        "profile_units": units, "profile_source": source,
        "overlay_half_width_hours": 28.0, "overlay_scale_multiplier": 10.0,
        "profile_p95_abs": p95, "overlay_scale_reference_abs": p95 * 10.0,
    }).to_csv(meta, index=False)


def four_hour_mean(mat, times):
    """4h-window means (window start = center label), matching the reference 4h profiles."""
    times = pd.DatetimeIndex(times)
    cols, cts = [], []
    for ws in pd.date_range(T1, T3, freq="4h"):
        if ws >= T3:
            continue
        m = (times >= ws) & (times < min(ws + pd.Timedelta("4h"), T3))
        if not m.any():
            continue
        cols.append(np.nanmean(mat[:, m], axis=1))
        cts.append(ws)
    return np.column_stack(cols), pd.DatetimeIndex(cts)


report = {}
qc = {}
# STRAIN: separable extraction on the full-resolution waterfall (clean).
strain_total, depth, times = read_dt(SRC / f"two_fault_direct_strain_waterfall_{S}.csv")
tensile_s, shear_s, g, resid_s = separate(strain_total, times, width_at(times))

# RATE: derive as the time-derivative of the clean strain components (mε -> nε/s),
# so shear rate is 0 exactly whenever shear strain is flat (e.g. before T2).
t_sec = (pd.DatetimeIndex(times) - pd.DatetimeIndex(times)[0]).total_seconds().to_numpy()
d_dt = lambda strain_ms: np.gradient(strain_ms, t_sec, axis=1) * 1e6
rate_total = d_dt(strain_total)
tensile_r, shear_r = d_dt(tensile_s), d_dt(shear_s)

for kind, tag, units, tot_wf, ten_wf, she_wf in [
    ("strain", "strain", "millistrain",  strain_total, tensile_s, shear_s),
    ("rate",   "rate",   "nanostrain/s", rate_total,   tensile_r, shear_r),
]:
    export_dt(she_wf, depth, times, OUT / f"v1_ddm_shear_{tag}_waterfall_{S}.csv")
    export_dt(ten_wf, depth, times, OUT / f"v1_ddm_tensile_{tag}_waterfall_{S}.csv")
    shear4, ct = four_hour_mean(she_wf, times)
    tens4, _ = four_hour_mean(ten_wf, times)
    tot4, _ = four_hour_mean(tot_wf, times)
    export_prof(shear4, depth, ct, units, f"V1 fault2 shear-only DDM ({kind})",
                OUT / f"v1_ddm_shear_{tag}_4h_{S}.csv", OUT / f"v1_ddm_shear_{tag}_4h_{S}_metadata.csv")
    export_prof(tens4, depth, ct, units, f"V1 fault1 tensile-only DDM ({kind})",
                OUT / f"v1_ddm_tensile_{tag}_4h_{S}.csv", OUT / f"v1_ddm_tensile_{tag}_4h_{S}_metadata.csv")
    full_pre = (ct + pd.Timedelta("4h")) <= T2   # windows ending at/before T2 (shear must be 0)
    report[kind] = dict(shear_peak=float(np.nanmax(np.abs(she_wf))),
                        tens_peak=float(np.nanmax(np.abs(ten_wf))),
                        resid_pre=float(np.nanmax(np.abs(she_wf[:, times < T2]))),
                        shear4_pre=float(np.nanmax(np.abs(shear4[:, full_pre]))),
                        shear4_peak=float(np.nanmax(np.abs(shear4))))
    if kind == "strain":
        qc["4h"] = (tot4, tens4, shear4, depth, ct)
        qc["waterfall"] = (strain_total, tensile_s, shear_s, depth, times)

# ---- QC figure ----
tot4, ten4, she4, dep, ct = qc["4h"]
totw, tenw, shew, depw, tw = qc["waterfall"]
fig = plt.figure(figsize=(16, 9))
# waterfalls
for k, (mat, title, cm, lim) in enumerate([
    (totw, "TOTAL (07152026)", "seismic", 0.1),
    (tenw, "TENSILE = width(t)*g(z)  [MOOSE replaces this]", "seismic", 0.1),
    (shew, "SHEAR = total - tensile  [fixed DDM input]", "seismic", 0.04)]):
    ax = fig.add_subplot(2, 3, k + 1)
    x0, x1 = mdates.date2num(tw[0].to_pydatetime()), mdates.date2num(tw[-1].to_pydatetime())
    im = ax.imshow(mat, aspect="auto", cmap=cm, vmin=-lim, vmax=lim,
                   extent=(x0, x1, depw[-1], depw[0]), interpolation="none")
    ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_title(title, fontsize=10)
    ax.axvline(T2, color="lime", ls="--", lw=1.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.colorbar(im, ax=ax, label="mε")
    if k == 0: ax.set_ylabel("MD (ft)")
# profiles at last snapshot + shear(t)
axp = fig.add_subplot(2, 3, 4)
axp.plot(tot4[:, -1], dep, "k-", label="total"); axp.plot(ten4[:, -1], dep, "r-", label="tensile")
axp.plot(she4[:, -1], dep, "b-", label="shear")
axp.set_ylim(10500, 10200); axp.set_title("Profiles @ T3 (03-03 19:00)", fontsize=10)
axp.set_xlabel("mε"); axp.set_ylabel("MD (ft)"); axp.legend(fontsize=8); axp.grid(alpha=0.3)
axs = fig.add_subplot(2, 3, 5)
pk = np.nanargmax(np.abs(she4[:, -1]))
axs.plot(ct, tot4[pk], "k-", label="total"); axs.plot(ct, ten4[pk], "r-", label="tensile")
axs.plot(ct, she4[pk], "b-", label="shear")
axs.axvline(T2, color="lime", ls="--"); axs.set_title(f"Time series @ MD {dep[pk]:.0f} ft", fontsize=10)
axs.set_ylabel("mε"); axs.legend(fontsize=8); axs.grid(alpha=0.3)
axs.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
axt = fig.add_subplot(2, 3, 6); axt.axis("off")
axt.text(0.0, 0.95, "Separable extraction\n" + "-" * 22 +
         f"\ntensile fit residual (T1-T2): {100*report['strain']['resid_pre']/report['strain']['shear_peak']:.1f}% of shear"
         f"\nshear peak: {report['strain']['shear_peak']:.4f} mε"
         f"\ntensile peak: {report['strain']['tens_peak']:.4f} mε"
         "\n\nshear ~ 0 before T2 (green line),\nturns on T2->T3 as slip accumulates.",
         va="top", family="monospace", fontsize=9)
fig.suptitle("V1 DDM decomposition: TOTAL = TENSILE (width*g) + SHEAR", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
figpath = REPO / "figs" / "tensile_fault_qc" / "v1_geometry" / "ddm_shear_decomposition_qc.png"
figpath.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figpath, dpi=130)

(OUT / "README.txt").write_text(
    "V1 DDM decomposition (shear extracted from 07152026 by tensile separability).\n"
    "eps_total = width(t)*g(z) [tensile] + eps_shear.  g(z) fit on T1-T2 (shear=0).\n"
    f"tensile fit residual T1-T2: {report['strain']['resid_pre']:.5f} me "
    f"({100*report['strain']['resid_pre']/report['strain']['shear_peak']:.1f}% of shear peak).\n\n"
    "Files (MD 10200-10500 ft, T1-referenced, UTC-7):\n"
    "  v1_ddm_shear_strain_4h_*        <- PRIMARY fixed shear input for the workflow\n"
    "  v1_ddm_shear_strain_waterfall_*\n"
    "  v1_ddm_shear_rate_4h_* / _waterfall_*    (secondary, strain-rate)\n"
    "  v1_ddm_tensile_*  = the DDM tensile that MOOSE will REPLACE (for QC only)\n"
    "tensile_target = observed_total - v1_ddm_shear_strain  (subtract only for T2-T3;\n"
    "  before T2 shear=0 so tensile_target = observed).\n")

print("=== DDM shear extraction complete ===")
for kind, r in report.items():
    print(f"{kind:6s}: shear peak {r['shear_peak']:.4f}, tensile peak {r['tens_peak']:.4f}; "
          f"waterfall T1-T2 shear residual {100*r['resid_pre']/r['shear_peak']:.1f}%; "
          f"4h-profile T1-T2 shear residual {100*r['shear4_pre']/r['shear4_peak']:.1f}%")
print("QC figure:", figpath)
print("Outputs in:", OUT)
print("\nfiles:")
for f in sorted(OUT.glob("v1_ddm_*.csv")):
    print("  ", f.name)
