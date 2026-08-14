"""Observed vs MOOSE+DDM model: LF-DAS waterfalls with 4-hour profiles overlaid.

Two figures, rendered identically so they can be read against each other:

  --background observed   observed strain-rate and T1-referenced strain waterfalls
  --background model      a(t)*MOOSE_tensile + b(t)*DDM_shear, the inverted model

The overlay on BOTH is the same pair of wiggles, one per 4-hour window:
  black solid   observed 4-hour mean profile
  green dashed  MOOSE + DDM model 4-hour mean profile      <- the inverted model, NOT the
                                                              raw two-fault DDM baseline

Consistency of rendering matters here. The observed waterfall is native LF-DAS (~11 s
sampling) while the model is built on the DDM export grid (~79 min), so drawn raw they look
nothing alike even where they agree. Both are therefore binned onto the SAME common time grid
before plotting, and share colour limits, colour maps and interpolation.

Model waterfall construction, on the common grid:
    strain(z,t) = a(t) * MOOSE_strain(z,t) + b(t) * DDM_shear(z,t)
    rate(z,t)   = d/dt strain(z,t) * 1e6            millistrain/s -> nanostrain/s
a(t) and b(t) are the per-window inversion coefficients, interpolated onto the grid; the DDM
shear waterfall is the uploaded fault-2 pure-shear export, zero before T2.
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_geometry as fg

REPO = Path(__file__).resolve().parents[3]
LEGACY = REPO / "data_fervo" / "legacy"
DDM = LEGACY / "07152026"
DEC = LEGACY / "07152026_decomposed"
OUT = REPO / "output" / "inversion"
S = fg.SUFFIX_1200
S_T2 = "20250228_0000_to_20250304_0000_10200_10500ft_4h_mean_T2_ref"

_ap = argparse.ArgumentParser(description="Observed vs MOOSE+DDM waterfalls")
_ap.add_argument("--background", choices=["observed", "model"], default="observed")
_ap.add_argument("--project", default="v10_final")
_ap.add_argument("--label", default=None)
_ap.add_argument("--obs-npz", default="observation_wf_1200.npz")
_ap.add_argument("--native", action="store_true",
                 help="draw the observed background at its own LF-DAS sampling "
                      "(~10 s) instead of binning it onto the model grid")
args = _ap.parse_args()
PROJ = args.project
LABEL = args.label or PROJ.split("_")[0].upper()

FIG = REPO / "figs" / "tensile_fault_qc" / PROJ
FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048
STAR = fg.STAR_1200
T1, T2, T3 = (pd.Timestamp(fg.T1_1200), pd.Timestamp(fg.T2_1200), pd.Timestamp(fg.T3_1200))
MD_LO, MD_HI = 10200.0, 10500.0
RATE_LIM, STRAIN_LIM = 0.3, 0.1


def read_dt(path):
    df = pd.read_csv(path)
    md = df["measured_depth_ft"].to_numpy(float)
    tt = pd.DatetimeIndex(pd.to_datetime(df.columns[1:]))
    return md, tt, df.iloc[:, 1:].to_numpy(float)


# ---------------------------------------------------------------- observation
z = np.load(OUT / args.obs_npz, allow_pickle=True)
obs_md = np.asarray(z["md_ft"], float)
obs_t = pd.DatetimeIndex([pd.Timestamp(str(s)) for s in z["times"]])
OBS_R_WF = np.asarray(z["rate_wf"], float)
OBS_S_WF = np.asarray(z["strain_wf"], float)

o_md_s, o_win_s, O_S = read_dt(LEGACY / f"strain_4h_mean_profiles_{S}.csv")
o_md_r, o_win_r, O_R = read_dt(LEGACY / f"strain_rate_4h_mean_profiles_{S}.csv")

# ---------------------------------------------------------------- DDM shear basis
b_md, b_win, B_prof = read_dt(DEC / f"fault2_shear_strain_4h_{S}.csv")
sw_md, sw_t, SW = read_dt(DEC / f"fault2_shear_strain_waterfall_{S_T2}.csv")

# ---------------------------------------------------------------- MOOSE tensile
d = REPO / "output" / PROJ
tax = pd.read_csv(d / f"{PROJ}_input_csv.csv")["time"].to_numpy(float)
vpp = sorted(glob.glob(str(d / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(tax))
y = None
cols = []
for f in vpp[:n]:
    dd = pd.read_csv(f)
    if len(dd) and y is None:
        y = dd.sort_values("y")["y"].to_numpy(float)
    cols.append(dd.sort_values("y")["strain_yy"].to_numpy(float) if len(dd) else None)
cols = [c if c is not None else np.zeros_like(y) for c in cols]
ms = np.column_stack(cols)
ms = (ms - ms[:, [0]]) * 1e3
m_md = STAR + y / FT
order = np.argsort(m_md)
m_md, ms = m_md[order], ms[order]
tax = tax[:n]

# ---------------------------------------------------------------- inversion a(t), b(t)
common = pd.DatetimeIndex([t for t in o_win_s if t in set(b_win)])
nt = len(common)
tc = np.array([(t - T1).total_seconds() for t in common])
O = O_S[:, [list(o_win_s).index(t) for t in common]]
B = np.array([np.interp(o_md_s, b_md, B_prof[:, list(b_win).index(t)]) for t in common]).T
M_t = np.array([np.interp(tc, tax, ms[k, :]) for k in range(ms.shape[0])])
M = np.array([np.interp(o_md_s, m_md, M_t[:, j]) for j in range(nt)]).T
a = np.zeros(nt)
b = np.zeros(nt)
b_unc = np.zeros(nt)
bscale = float(np.nanmax(np.abs(B)))
for j in range(nt):
    dj, mj, bj = O[:, j], M[:, j], B[:, j]
    v = np.isfinite(dj) & np.isfinite(mj) & np.isfinite(bj)
    if v.sum() < 3:
        continue
    A = np.column_stack([mj[v], bj[v]])
    if float(np.nanmax(np.abs(bj[v]))) > 1e-3 * bscale:
        a[j], b[j] = nnls(A, dj[v])[0]
        b_unc[j] = np.linalg.lstsq(A, dj[v], rcond=None)[0][1]
    else:
        den = float(np.dot(mj[v], mj[v]))
        a[j] = max(0.0, float(np.dot(mj[v], dj[v]) / den)) if den > 0 else 0.0
MODEL_PROF_S = a * M + b * B
res = O - MODEL_PROF_S
vm = np.isfinite(O) & np.isfinite(MODEL_PROF_S)
vr = 100 * (1 - np.nanmean(res[vm] ** 2) / np.nanmean(O[vm] ** 2))
print(f"{LABEL}: VR {vr:.2f}%   a final {a[-1]:.3f}   b final {b[-1]:.3f}   "
      f"({int((b_unc < 0).sum())} windows have b clipped to 0 by non-negativity)")

# ---------------------------------------------------------------- common time grid
grid = pd.DatetimeIndex(pd.date_range(T1, T3, periods=200))
gsec = np.array([(t - T1).total_seconds() for t in grid])


def to_grid_wf(md_src, t_src, Mx):
    """Bin/interpolate a [md, t] waterfall onto (obs_md, grid)."""
    ts = np.array([(t - T1).total_seconds() for t in pd.DatetimeIndex(t_src)])
    tmp = np.array([np.interp(gsec, ts, Mx[k, :], left=np.nan, right=np.nan)
                    for k in range(Mx.shape[0])])
    return np.array([np.interp(o_md_s, md_src, tmp[:, j]) for j in range(len(grid))]).T


def bin_obs(Mx):
    """Average the native observed waterfall into the common grid bins."""
    ts = np.array([(t - T1).total_seconds() for t in obs_t])
    edges = np.concatenate([[gsec[0]], 0.5 * (gsec[1:] + gsec[:-1]), [gsec[-1]]])
    idx = np.clip(np.digitize(ts, edges) - 1, 0, len(grid) - 1)
    outm = np.full((Mx.shape[0], len(grid)), np.nan)
    for j in range(len(grid)):
        sel = idx == j
        if sel.any():
            with np.errstate(invalid="ignore"):
                outm[:, j] = np.nanmean(Mx[:, sel], axis=1)
    return np.array([np.interp(o_md_s, obs_md, outm[:, j]) for j in range(len(grid))]).T


if args.background == "observed":
    if args.native:
        # straight from the LF-DAS: no re-gridding, no binning
        BG_R, BG_S = OBS_R_WF, OBS_S_WF
        bg_t_plot, bg_md_plot = obs_t, obs_md
        WHO, TAG = "Observed (native LF-DAS)", "observed_native"
    else:
        BG_R, BG_S = bin_obs(OBS_R_WF), bin_obs(OBS_S_WF)
        bg_t_plot, bg_md_plot = grid, o_md_s
        WHO, TAG = "Observed", "observed"
else:
    MO = np.array([np.interp(gsec, tax, ms[k, :]) for k in range(ms.shape[0])])
    MOOSE_G = np.array([np.interp(o_md_s, m_md, MO[:, j]) for j in range(len(grid))]).T
    SHEAR_G = np.nan_to_num(to_grid_wf(sw_md, sw_t, SW))
    SHEAR_G[:, grid < T2] = 0.0
    a_g = np.interp(gsec, tc, a)
    b_g = np.interp(gsec, tc, b, left=0.0)
    BG_S = a_g * MOOSE_G + b_g * SHEAR_G
    BG_R = np.gradient(BG_S, gsec, axis=1) * 1e6            # millistrain/s -> nanostrain/s
    bg_t_plot, bg_md_plot = grid, o_md_s
    WHO, TAG = f"{LABEL} model  (a·MOOSE + b·DDM)", "model"
print(f"{WHO} background: rate {BG_R.shape}, strain {BG_S.shape}")

# ---------------------------------------------------------------- model 4h profiles
MODEL_PROF_R = np.zeros_like(O_R)
_bgr_model = None
MO = np.array([np.interp(gsec, tax, ms[k, :]) for k in range(ms.shape[0])])
MOOSE_G = np.array([np.interp(o_md_s, m_md, MO[:, j]) for j in range(len(grid))]).T
SHEAR_G = np.nan_to_num(to_grid_wf(sw_md, sw_t, SW))
SHEAR_G[:, grid < T2] = 0.0
MODEL_S_G = np.interp(gsec, tc, a) * MOOSE_G + np.interp(gsec, tc, b, left=0.0) * SHEAR_G
MODEL_R_G = np.gradient(MODEL_S_G, gsec, axis=1) * 1e6
for j, ws in enumerate(o_win_r):
    sel = (grid >= ws) & (grid < ws + pd.Timedelta(hours=4))
    if sel.any():
        MODEL_PROF_R[:, j] = np.interp(o_md_r, o_md_s, np.nanmean(MODEL_R_G[:, sel], axis=1))


def combined_scale(*mats, multiplier=10.0, half_width_hours=28.0):
    vals = [np.abs(m[np.isfinite(m)]) for m in mats if m.size]
    c = np.concatenate([v for v in vals if v.size])
    p95 = np.nanpercentile(c, 95)
    if not np.isfinite(p95) or p95 == 0:
        p95 = np.nanmax(c)
    return half_width_hours * 3600.0 / (p95 * multiplier), p95


SEC_S, P95_S = combined_scale(O_S, MODEL_PROF_S)
SEC_R, P95_R = combined_scale(O_R, MODEL_PROF_R)
print(f"combined obs+model p95: strain {P95_S:.5g} me, rate {P95_R:.5g} ns/s")

x0 = mdates.date2num(bg_t_plot[0].to_pydatetime())
x1 = mdates.date2num(bg_t_plot[-1].to_pydatetime())

# ---- measured upper zero-crossing of the observed strain (the white red/blue boundary) ----
zc = np.full(len(o_win_s), np.nan)
for j in range(len(o_win_s)):
    pr = O_S[:, j]
    ipk = int(np.nanargmax(pr))
    seg, smd = pr[:ipk + 1], o_md_s[:ipk + 1]
    idx = np.where(np.diff(np.sign(seg)) > 0)[0]
    if len(idx):
        i = idx[-1]
        zc[j] = smd[i] + (0 - seg[i]) * (smd[i + 1] - smd[i]) / (seg[i + 1] - seg[i])
_late = zc[o_win_s >= pd.Timestamp("2025-03-01")]
SHEAR_MD = fg.SHEAR_MD_1200
print(f"measured upper zero-crossing: all {np.nanmean(zc):.2f} +/- {np.nanstd(zc):.2f} ft, "
      f"after 03-01 {np.nanmean(_late):.2f} +/- {np.nanstd(_late):.2f} ft, "
      f"final window {zc[-1]:.2f} ft")
print(f"  shear plane {SHEAR_MD:.3f} ft -> final-window offset {zc[-1] - SHEAR_MD:+.2f} ft "
      f"(observed MD spacing {np.diff(o_md_s).mean():.2f} ft)")
DAY = 86400.0


def draw(ax, BG, lim, cmap, o_win, O_, M_, o_md, m_md_, sec, cbl, title):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.82")
    im = ax.imshow(BG, aspect="auto", cmap=cm, vmin=-lim, vmax=lim,
                   extent=[x0, x1, bg_md_plot[-1], bg_md_plot[0]], interpolation="nearest")
    # measured zero-crossing, and the DDM shear plane it should sit on
    ok = np.isfinite(zc)
    ax.plot([mdates.date2num(t.to_pydatetime()) for t in o_win_s[ok]], zc[ok],
            color="#0b3d0b", lw=2.4, marker="o", ms=3.5, zorder=8,
            label=f"MEASURED zero-crossing (final {zc[-1]:.1f} ft)")
    ax.axhline(SHEAR_MD, color="#8e44ad", ls="--", lw=2.0, zorder=8,
               label=f"DDM shear plane MD {SHEAR_MD:.1f}")
    for win, P, md, colour, ls, lw, lab in [
            (o_win, O_, o_md, "black", "-", 1.1, "Observed"),
            (o_win, M_, m_md_, "#1a7f5a", "--", 1.2, f"{LABEL} model  a·MOOSE + b·DDM")]:
        for j, t in enumerate(win):
            if not (T1 <= t <= T3):
                continue
            xs = mdates.date2num(t.to_pydatetime()) + P[:, j] * sec / DAY
            ax.plot(xs, md, color=colour, ls=ls, lw=lw,
                    label=lab if j == 0 else None, zorder=5)
    for tt, st in [(T1, "-"), (T2, "--"), (T3, "-")]:
        ax.axvline(mdates.date2num(tt.to_pydatetime()), color="gold", ls=st, lw=1.8, zorder=6)
    for tt, lab, ha in [(T1, "T1", "left"), (T3, "T3", "right")]:
        ax.text(mdates.date2num(tt.to_pydatetime()), 1.012, lab,
                transform=ax.get_xaxis_transform(), color="goldenrod", ha=ha,
                fontweight="bold", fontsize=11)
    ax.set_xlim(x0, x1)
    ax.set_ylim(MD_HI, MD_LO)
    ax.set_ylabel("Measured Depth (ft)")
    ax.set_title(title, fontsize=12)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax.legend(loc="lower right", fontsize=9, framealpha=.9).set_zorder(7)
    plt.colorbar(im, ax=ax, pad=0.012).set_label(cbl)


fig, (ax_r, ax_s) = plt.subplots(2, 1, figsize=(19, 11), constrained_layout=True, sharex=True)
draw(ax_r, BG_R, RATE_LIM, "bwr", o_win_r, O_R, MODEL_PROF_R, o_md_r, o_md_r, SEC_R,
     "Strain rate (nanostrain/s)",
     f"{WHO} strain-rate waterfall — observed vs MOOSE+DDM 4-hour profiles")
draw(ax_s, BG_S, STRAIN_LIM, "seismic", o_win_s, O_S, MODEL_PROF_S, o_md_s, o_md_s, SEC_S,
     "Strain (millistrain)",
     f"{WHO} T1-referenced strain waterfall — observed vs MOOSE+DDM 4-hour profiles "
     f"(VR {vr:.1f}%)")
ax_s.set_xlabel("Time [UTC-7]")

out = FIG / f"lfdas_waterfall_{TAG}_vs_moose_ddm.png"
fig.savefig(out, dpi=140)
print("saved", out)
