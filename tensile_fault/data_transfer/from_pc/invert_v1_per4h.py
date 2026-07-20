"""Per-4h-snapshot inversion for V1: two free coefficients per snapshot.

At each 4h snapshot t_i, least-squares fit the observed strain profile with
    eps_obs(z,t_i) = s_p(t_i) * eps_MOOSE_tensile(z,t_i) + s_sh(t_i) * eps_DDM_shear(z,t_i)
  - T1->T2: shear = 0, so 1 free coeff s_p (tensile pressure scale).
  - T2->T3: 2 free coeffs (s_p tensile pressure, s_sh shear magnitude).
The DDM shear plane is shifted to the SRV<->matrix boundary (MD_centre - SRV_width/2),
per Pengchao (the red/blue shear boundary sits at the SRV edge, not the fault centre).

Outputs: s_p(t) + s_sh(t) histories, inferred within-fracture pressure, obs/model/residual
waterfall.  Usage:  python invert_v1_per4h.py --project v1_srv_w75_p03 --srv_width_ft 75
"""
import argparse, glob, os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[4]
ap = argparse.ArgumentParser()
ap.add_argument("--project", default="v1_srv_w75_p03")
ap.add_argument("--srv_width_ft", type=float, default=75.0)
args = ap.parse_args()
PROJ = args.project
W = args.srv_width_ft
OUT = REPO / "output" / PROJ
FIG = REPO / "figs" / "tensile_fault_qc" / PROJ; FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048; STAR = 10373.4
T1 = pd.Timestamp("2025-02-24 15:00"); T2 = pd.Timestamp("2025-02-28 00:00"); T3 = pd.Timestamp("2025-03-03 22:00")
SHEAR_PLANE_MD = STAR - W / 2.0     # DDM shear plane at the (upper) SRV-matrix boundary
OBS = REPO / "data_fervo/legacy/strain_4h_mean_profiles_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv"
SHEAR = REPO / "data_fervo/legacy/07152026_decomposed/v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv"
PZ = OUT / "das_pressure_prepended.npz"


def rp(c):
    df = pd.read_csv(c); return df["measured_depth_ft"].to_numpy(float), pd.to_datetime(df.columns[1:]), df.iloc[:, 1:].to_numpy(float)


def interp_time(mat, src, grid):
    ss = (src - T1).total_seconds().to_numpy(); gs = (grid - T1).total_seconds().to_numpy()
    return np.array([np.interp(gs, ss, mat[k]) for k in range(mat.shape[0])])


# MOOSE tensile strain_yy(MD,t)
tdf = pd.read_csv(OUT / f"{PROJ}_input_csv.csv"); taxis = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(OUT / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis)); taxis = taxis[:n]; vpp = vpp[:n]
yv = next(pd.read_csv(f).sort_values("y")["y"].to_numpy(float) for f in vpp if len(pd.read_csv(f)))
cols = [pd.read_csv(vpp[i]).sort_values("y")["strain_yy"].to_numpy(float) if len(pd.read_csv(vpp[i])) else np.zeros_like(yv) for i in range(n)]
m_all = (np.column_stack(cols) - np.column_stack(cols)[:, [0]]) * 1e3
m_md = STAR + yv / FT; oo = np.argsort(m_md); m_md, m_all = m_md[oo], m_all[oo]
m_t = T1 + pd.to_timedelta(taxis, unit="s")

grid = pd.date_range(T1, T3, freq="4h")
o_md, o_ct, o_mat = rp(OBS)
s_md, s_ct, s_mat = rp(SHEAR)
# shear on obs MD grid, SHIFTED so its plane sits at the SRV-matrix boundary
shift = SHEAR_PLANE_MD - STAR                        # native shear crossing ~ STAR
s_on = np.array([np.interp(o_md - shift, s_md, s_mat[:, j]) for j in range(s_mat.shape[1])]).T
O = interp_time(o_mat, o_ct, grid); O -= O[:, [0]]
SH = interp_time(s_on, s_ct, grid); SH -= SH[:, [0]]
M = interp_time(np.array([np.interp(o_md, m_md, m_all[:, j]) for j in range(m_all.shape[1])]).T, m_t, grid); M -= M[:, [0]]

# per-snapshot least squares
ng = len(grid); s_p = np.full(ng, np.nan); s_sh = np.zeros(ng)
for j in range(ng):
    o = O[:, j]; mt = M[:, j]; sh = SH[:, j]; f = np.isfinite(o) & np.isfinite(mt)
    if not f.any() or np.dot(mt[f], mt[f]) == 0:
        s_p[j] = 0; continue
    if grid[j] < T2:                                 # tensile only
        s_p[j] = np.dot(mt[f], o[f]) / np.dot(mt[f], mt[f])
    else:                                            # tensile + shear (2 params)
        A = np.column_stack([mt[f], sh[f]]); coef, *_ = np.linalg.lstsq(A, o[f], rcond=None)
        s_p[j], s_sh[j] = coef
MODEL = M * s_p[None, :] + SH * s_sh[None, :]
vv = np.isfinite(O) & np.isfinite(MODEL)
rms0 = np.sqrt(np.nanmean(O[vv]**2)); rmsr = np.sqrt(np.nanmean((O - MODEL)[vv]**2))
print(f"[{PROJ}] per-4h inversion: RMS {rms0:.4f}->{rmsr:.4f} mε ({100*(1-rmsr/rms0):.0f}% var red)")
print(f"  s_p (tensile pressure scale): {np.nanmin(s_p):.2f}..{np.nanmax(s_p):.2f}")
print(f"  s_sh (shear scale, T2-T3): {np.nanmin(s_sh[grid>=T2]):.2f}..{np.nanmax(s_sh[grid>=T2]):.2f}")
print(f"  shear plane placed at MD {SHEAR_PLANE_MD:.0f} ft (SRV-matrix boundary, W={W:.0f} ft)")

# inferred pressure (per-snapshot) = IC + s_p * DAS Δp
pz = np.load(PZ, allow_pickle=True); pdas = np.asarray(pz["data"], float); pic = float(pdas[0])
pt = T1 + pd.to_timedelta(np.asarray(pz["taxis"], float), unit="s")
das_on_grid = np.interp((grid - T1).total_seconds(), (pt - T1).total_seconds(), pdas)
fracp = pic + s_p * (das_on_grid - pic)
pd.DataFrame({"time": grid, "tensile_pressure_scale": s_p, "shear_scale": s_sh,
              "das_pressure_psi": das_on_grid, "fracture_pressure_psi": fracp,
              "fracture_dp_psi": fracp - pic}).to_csv(OUT / "inversion_per4h.csv", index=False)

# ---- figures ----
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True, sharey=True)
ext = [mdates.date2num(grid[0]), mdates.date2num(grid[-1]), o_md[-1], o_md[0]]
for a, (mat, ttl, lim) in zip(axs, [(O, "observed", 0.1), (MODEL, "model (s_p·tensile + s_sh·shear)", 0.1), (O - MODEL, "residual", 0.05)]):
    im = a.imshow(mat, aspect="auto", cmap="seismic" if "resid" not in ttl else "bwr", vmin=-lim, vmax=lim, extent=ext, interpolation="bilinear")
    a.xaxis_date(); a.set_title(ttl); a.set_ylim(10500, 10200); a.axvline(mdates.date2num(T2.to_pydatetime()), color="lime", ls="--")
    a.axhline(SHEAR_PLANE_MD, color="0.3", ls=":", lw=1); a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); plt.colorbar(im, ax=a, label="mε")
axs[0].set_ylabel("MD (ft)")
fig.suptitle(f"V1 per-4h inversion — SRV width {W:.0f} ft — {100*(1-rmsr/rms0):.0f}% var. reduction", fontweight="bold")
fig.savefig(FIG / "per4h_waterfall.png", dpi=140); print("Saved:", FIG / "per4h_waterfall.png")

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
ax[0].plot(grid, s_p, "o-", color="#b30000", label="s_p  (tensile pressure scale)")
ax[0].plot(grid[grid >= T2], s_sh[grid >= T2], "s--", color="tab:blue", label="s_sh  (shear scale, T2→T3)")
ax[0].axvline(T2, color="0.5", ls=":"); ax[0].set_ylabel("per-snapshot coefficient"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("Per-4h inversion coefficients")
ax[1].plot(pt, pdas, "0.6", lw=1, label="DAS pressure (s=1)")
ax[1].plot(grid, fracp, "r-o", ms=3, lw=1.8, label="inferred within-fracture pressure (per-4h)")
ax[1].axhline(pic, color="k", ls=":", label=f"IC {pic:.0f} psi"); ax[1].axvline(T2, color="0.5", ls=":")
ax[1].set_ylabel("pressure [psi]"); ax[1].set_xlabel("Time [UTC-7]"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title("Inferred within-fracture pressure history")
for a in ax:
    for tt, lab in zip([T1, T2, T3], ["T1", "T2", "T3"]):
        a.text(tt, 1.01, lab, transform=a.get_xaxis_transform(), color="green", fontweight="bold", ha="center", va="bottom")
    a.set_xlim(T1, T3); a.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.savefig(FIG / "per4h_coefficients.png", dpi=140); print("Saved:", FIG / "per4h_coefficients.png")

# ---- deliverable-style waterfalls with 4h profile overlays (image.png style) ----
PROFILE_HOURS = 2.6
p95_shared = np.nanpercentile(np.abs(O), 95) or 1.0


def _wig(ax, prof, color, lw=0.8):
    sec = PROFILE_HOURS * 3600.0 / p95_shared
    for j, t in enumerate(grid):
        f = np.isfinite(prof[:, j])
        ax.plot(t + pd.to_timedelta(prof[f, j] * sec, unit="s"), o_md[f], color=color, lw=lw, alpha=0.9, zorder=5)


def _wf(ax, prof, title, lim=0.1):
    tf = pd.date_range(grid[0], grid[-1], periods=300)
    bgm = np.array([np.interp((tf - grid[0]).total_seconds(), (grid - grid[0]).total_seconds(), prof[k])
                    for k in range(prof.shape[0])])
    im = ax.imshow(bgm, aspect="auto", cmap="seismic", vmin=-lim, vmax=lim,
                   extent=[mdates.date2num(tf[0]), mdates.date2num(tf[-1]), o_md[-1], o_md[0]], interpolation="bilinear")
    ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_ylabel("Gold 4-PB Measured Depth [ft]"); ax.set_title(title)
    for tt, lab in zip([grid[0], T2, grid[-1]], ["T1", "T2", "T3"]):
        ax.axvline(tt, color="green", ls="--", lw=1.5)
        ax.text(tt, 1.01, lab, transform=ax.get_xaxis_transform(), color="green", fontweight="bold", ha="center", va="bottom")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    plt.colorbar(im, ax=ax).set_label("Strain (millistrain)")


# Figure A: 2-panel — observed (top) vs model (bottom), each with its own 4h profiles
figA, (aA1, aA2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
_wf(aA1, O, "Observed DAS strain waterfall with 4-hour mean profiles (T1 referenced)"); _wig(aA1, O, "black")
_wf(aA2, MODEL, f"Model = per-4h (s_p·MOOSE tensile + s_sh·DDM shear) with 4-hour mean profiles"); _wig(aA2, MODEL, "black")
aA2.set_xlabel("Time [UTC-7]")
figA.suptitle(f"V1 observed vs MOOSE+DDM strain — SRV {W:.0f} ft — {100*(1-rmsr/rms0):.0f}% var. reduction", fontweight="bold", fontsize=14)
figA.savefig(FIG / "deliverable_obs_vs_model_waterfall.png", dpi=150); print("Saved:", FIG / "deliverable_obs_vs_model_waterfall.png")

# Figure B: single model-background waterfall + BOTH overlays (observed black, model red)
figB, axB = plt.subplots(figsize=(16, 6.6), constrained_layout=True)
_wf(axB, MODEL, "Model strain waterfall (background) with overlaid OBSERVED (black) and SIMULATED (red) 4-hour profiles")
_wig(axB, O, "black"); _wig(axB, MODEL, "red")
axB.plot([], [], "k-", lw=2, label="observed 4h profiles"); axB.plot([], [], "r-", lw=2, label="model 4h profiles")
axB.legend(loc="lower right", fontsize=11); axB.set_xlabel("Time [UTC-7]")
figB.savefig(FIG / "model_bg_obs_model_overlay.png", dpi=150); print("Saved:", FIG / "model_bg_obs_model_overlay.png")
