"""Deliverable figure (image.png style): observed vs model strain waterfalls with
overlaid 4-hour mean profiles, for the V1 MOOSE-tensile + DDM-shear match (option a).

Top  = observed strain (T1-referenced) waterfall + observed 4h profiles (black wiggles)
Bottom = model = (s * MOOSE tensile + DDM shear) waterfall + model 4h profiles
Style mirrors data_fervo/legacy/image.png: MD 10200-10500 ft, time T1->T3, T2 green
dashed, seismic colormap +/-0.1 me, black 4h profile overlays, yellow-star peak.
"""
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
PROJ = "v1_tensile_srv_das_refined"
OUT = REPO / "output" / PROJ
FIGP = REPO / "figs" / "tensile_fault_qc" / PROJ / "deliverable_obs_vs_model_waterfall.png"
FT = 0.3048
STAR_DEPTH_FT = 10373.4
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")
OBS_CSV = REPO / "data_fervo" / "legacy" / (
    "strain_4h_mean_profiles_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
SHEAR_CSV = REPO / "data_fervo" / "legacy" / "07152026_decomposed" / (
    "v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
PROFILE_HOURS = 2.6          # a p95-magnitude profile maps to +/-2.6 h (wiggle width)


def read_prof(csv):
    df = pd.read_csv(csv)
    return (df["measured_depth_ft"].to_numpy(float), pd.to_datetime(df.columns[1:]),
            df.iloc[:, 1:].to_numpy(float))


# --- MOOSE fiber strain_yy(MD, t) --------------------------------------------
tdf = pd.read_csv(OUT / f"{PROJ}_input_csv.csv")
taxis_s = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(OUT / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis_s)); taxis_s = taxis_s[:n]; vpp = vpp[:n]
yv = next(pd.read_csv(f).sort_values("y")["y"].to_numpy(float) for f in vpp if len(pd.read_csv(f)))
cols = [pd.read_csv(vpp[i]).sort_values("y")["strain_yy"].to_numpy(float) if len(pd.read_csv(vpp[i]))
        else np.zeros_like(yv) for i in range(n)]
m_strain = (np.column_stack(cols) - np.column_stack(cols)[:, [0]]) * 1e3
m_md = STAR_DEPTH_FT + yv / FT
oo = np.argsort(m_md); m_md, m_strain = m_md[oo], m_strain[oo]

# --- observed + shear on the 11:00 grid --------------------------------------
o_md, o_ct, O = read_prof(OBS_CSV)
s_md, s_ct, s_mat = read_prof(SHEAR_CSV)
common = [t for t in o_ct if t in set(s_ct)]
oi = [list(o_ct).index(t) for t in common]; si = [list(s_ct).index(t) for t in common]
O = O[:, oi]
SH = np.array([np.interp(o_md, s_md, s_mat[:, j]) for j in si]).T
tc_s = np.array([(t - T1).total_seconds() for t in common])
M_t = np.array([np.interp(tc_s, taxis_s, m_strain[k]) for k in range(m_strain.shape[0])])  # (m_md, n_common)
M = np.array([np.interp(o_md, m_md, M_t[:, j]) for j in range(len(common))]).T             # (o_md, n_common)
v = np.isfinite(M) & np.isfinite(O - SH)
s_best = float(np.dot(M[v], (O - SH)[v]) / np.dot(M[v], M[v]))
MODEL = s_best * M + SH
print(f"s={s_best:.3f}; obs peak {np.nanmax(O):.4f}, model peak {np.nanmax(MODEL):.4f} mε")

ct = pd.DatetimeIndex(common)


def wiggle(ax, prof, depths, times, color="k"):
    p95 = np.nanpercentile(np.abs(prof), 95) or np.nanmax(np.abs(prof))
    sec = PROFILE_HOURS * 3600.0 / p95
    for j, t in enumerate(times):
        f = np.isfinite(prof[:, j])
        ax.plot(t + pd.to_timedelta(prof[f, j] * sec, unit="s"), depths[f],
                color=color, lw=0.7, alpha=0.9, zorder=5)


def waterfall(ax, prof4h, depths, times, title, cbar_label, lim=0.1):
    tf = pd.date_range(times[0], times[-1], periods=300)
    tfs = (tf - times[0]).total_seconds().to_numpy()
    tcs = (times - times[0]).total_seconds().to_numpy()
    wf = np.array([np.interp(tfs, tcs, prof4h[k, :]) for k in range(prof4h.shape[0])])
    im = ax.imshow(wf, aspect="auto", cmap="seismic", vmin=-lim, vmax=lim,
                   extent=[mdates.date2num(tf[0]), mdates.date2num(tf[-1]), depths[-1], depths[0]],
                   interpolation="bilinear")
    ax.xaxis_date()
    wiggle(ax, prof4h, depths, times)
    for tt, lab in zip([times[0], T2, times[-1]], ["T1", "T2", "T3"]):
        ax.axvline(tt, color="green", ls="--", lw=1.6, alpha=0.9)
        ax.text(tt, 1.01, lab, color="green", fontweight="bold", ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
    # yellow star at the peak of the last profile
    jlast = prof4h.shape[1] - 1
    ip = int(np.nanargmax(np.abs(prof4h[:, jlast])))
    p95 = np.nanpercentile(np.abs(prof4h), 95) or 1
    xstar = times[jlast] + pd.to_timedelta(prof4h[ip, jlast] * PROFILE_HOURS * 3600.0 / p95, unit="s")
    ax.plot(xstar, depths[ip], "*", color="yellow", mec="k", ms=15, zorder=10)
    ax.annotate(f"{prof4h[ip, jlast]:+.4f} millistrain", (xstar, depths[ip]),
                textcoords="offset points", xytext=(-8, 6), fontsize=8,
                bbox=dict(fc="white", ec="k", alpha=0.85))
    ax.set_ylim(10500, 10200); ax.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax); cb.set_label(cbar_label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
waterfall(ax1, O, o_md, ct, "Observed DAS Strain Waterfall with 4-hour Mean Profiles (T1 Referenced)",
          "Strain (millistrain)")
waterfall(ax2, MODEL, o_md, ct,
          f"Model = {s_best:.2f}×MOOSE tensile + DDM shear  (within-fracture pressure history)",
          "Strain (millistrain)")
ax2.set_xlabel("Time [UTC-7]")
fig.suptitle("V1 fault: observed vs MOOSE+DDM strain (option a deliverable)", fontweight="bold", fontsize=13)
fig.savefig(FIGP, dpi=150)
print("Saved:", FIGP)
