"""V1 tensile-SRV (smooth DAS pressure) + DDM shear vs observed, full T1->T3.

Finds the pressure scale s minimizing  || eps_obs - (s*eps_MOOSE_tensile + eps_DDM_shear) ||
over the whole T1->T3 window (shear=0 before T2, so this reduces to obs vs s*MOOSE there).
Reports best s, the implied within-fracture pressure = IC + s*(das_dp), and match figures.
"""
import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
_ap = argparse.ArgumentParser()
_ap.add_argument("--project", default="v1_tensile_srv_das")
PROJ = _ap.parse_args().project
OUT = REPO / "output" / PROJ
FIG = REPO / "figs" / "tensile_fault_qc" / PROJ
FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048
STAR_DEPTH_FT = 10373.4
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")

OBS_CSV = REPO / "data_fervo" / "legacy" / (
    "strain_4h_mean_profiles_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
SHEAR_CSV = REPO / "data_fervo" / "legacy" / "07152026_decomposed" / (
    "v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
PREPPED_P = OUT / "das_pressure_T1_prepended.npz"


def read_prof(csv):
    df = pd.read_csv(csv)
    md = df["measured_depth_ft"].to_numpy(float)
    ct = pd.to_datetime(df.columns[1:])
    return md, ct, df.iloc[:, 1:].to_numpy(float)


# ---- MOOSE fiber strain_yy(MD, t) -------------------------------------------
tdf = pd.read_csv(OUT / f"{PROJ}_input_csv.csv")
taxis_s = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(OUT / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis_s))
taxis_s = taxis_s[:n]
vpp = vpp[:n]
y = None
for f in vpp:
    dd = pd.read_csv(f)
    if len(dd):
        y = dd.sort_values("y")["y"].to_numpy(float); break
cols = [pd.read_csv(vpp[i]).sort_values("y")["strain_yy"].to_numpy(float) if len(pd.read_csv(vpp[i]))
        else np.zeros_like(y) for i in range(n)]
m_strain = (np.column_stack(cols) - np.column_stack(cols)[:, [0]]) * 1e3   # mε, T1-ref
m_md = STAR_DEPTH_FT + (y - 0.0) / FT
o = np.argsort(m_md); m_md, m_strain = m_md[o], m_strain[o]
print(f"MOOSE: {m_strain.shape} peak |eps|={np.nanmax(np.abs(m_strain)):.4f} mε over {len(taxis_s)} times")

# ---- observed + DDM shear (both on the 11:00 grid) --------------------------
o_md, o_ct, o_mat = read_prof(OBS_CSV)
s_md, s_ct, s_mat = read_prof(SHEAR_CSV)
common = [t for t in o_ct if t in set(s_ct)]
oi = [list(o_ct).index(t) for t in common]
si = [list(s_ct).index(t) for t in common]
O = o_mat[:, oi]                                             # (o_md, nt)
SH = np.array([np.interp(o_md, s_md, s_mat[:, j]) for j in si]).T   # shear on obs MD grid
# MOOSE: interpolate in time to the common centers, then in MD to obs grid
tc_s = np.array([(t - T1).total_seconds() for t in common])
M_t = np.array([np.interp(tc_s, taxis_s, m_strain[k, :]) for k in range(m_strain.shape[0])])  # (m_md, nt)
M = np.array([np.interp(o_md, m_md, M_t[:, j]) for j in range(len(common))]).T                # (o_md, nt)
print(f"aligned {len(common)} windows (T1..{common[-1]}) on {len(o_md)} obs depths")

# ---- linear pressure-scale inversion on the residual tensile target ----------
tgt = O - SH                                                # tensile target = obs - shear
v = np.isfinite(tgt) & np.isfinite(M)
s_best = float(np.dot(M[v], tgt[v]) / np.dot(M[v], M[v]))
model = s_best * M + SH
rms0 = np.sqrt(np.nanmean(O[v] ** 2))
rms_r = np.sqrt(np.nanmean((O - model)[v] ** 2))
svals = np.linspace(0, max(2.0, 2 * s_best), 200)
misfit = [np.sqrt(np.nanmean((tgt[v] - sv * M[v]) ** 2)) for sv in svals]
# per-snapshot scale
s_snap = np.array([float(np.dot(M[:, j][np.isfinite(M[:, j])], tgt[:, j][np.isfinite(tgt[:, j])]) /
                          np.dot(M[:, j][np.isfinite(M[:, j])], M[:, j][np.isfinite(M[:, j])]))
                   for j in range(len(common))])
print(f"BEST pressure scale s={s_best:.3f}; RMS {rms0:.4f}->{rms_r:.4f} mε "
      f"({100*(1-rms_r/rms0):.0f}% var reduction); per-snap s {np.nanmin(s_snap):.2f}..{np.nanmax(s_snap):.2f}")

# ---- implied within-fracture pressure ---------------------------------------
pz = np.load(PREPPED_P, allow_pickle=True)
p_das = np.asarray(pz["data"], float); p_ic = float(p_das[0])
p_t = T1 + pd.to_timedelta(np.asarray(pz["taxis"], float), unit="s")
frac_p = p_ic + s_best * (p_das - p_ic)
print(f"das Δp peak={p_das.max()-p_ic:.0f} psi; implied fracture Δp peak (s={s_best:.2f})="
      f"{s_best*(p_das.max()-p_ic):.0f} psi; abs peak={frac_p.max():.0f} psi")

# ---- figures -----------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
ax[0,0].plot(svals, misfit, "b-"); ax[0,0].axvline(s_best, color="r", ls="--", label=f"best s={s_best:.2f}")
ax[0,0].set(xlabel="pressure scale s", ylabel="RMS misfit (mε)", title="(a) misfit vs pressure scale (T1→T3)")
ax[0,0].grid(alpha=0.3); ax[0,0].legend()
j = len(common) - 1
ax[0,1].plot(O[:, j], o_md, "k-", lw=2, label=f"observed ({common[j]:%m-%d %H:%M})")
ax[0,1].plot(model[:, j], o_md, "r--", lw=2, label=f"MOOSE×{s_best:.2f}+shear")
ax[0,1].plot(s_best*M[:, j], o_md, "tab:orange", lw=1, label="MOOSE tensile×s")
ax[0,1].plot(SH[:, j], o_md, "b", lw=1, label="DDM shear")
ax[0,1].set(xlabel="strain (mε)", ylabel="MD (ft)", title="(b) profile near T3")
ax[0,1].set_ylim(10500, 10200); ax[0,1].grid(alpha=0.3); ax[0,1].legend(fontsize=8)
ax[1,0].plot(common, s_snap, "o-", color="tab:green"); ax[1,0].axhline(s_best, color="r", ls="--", label=f"global {s_best:.2f}")
ax[1,0].axvline(T2, color="0.5", ls=":"); ax[1,0].set(xlabel="time", ylabel="per-snapshot s", title="(c) per-snapshot pressure scale")
ax[1,0].grid(alpha=0.3); ax[1,0].legend(); ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax[1,1].plot(p_t, p_das, "0.6", lw=1.2, label="DAS pressure (s=1)")
ax[1,1].plot(p_t, frac_p, "r-", lw=1.8, label=f"implied fracture p (s={s_best:.2f})")
ax[1,1].axhline(p_ic, color="k", ls=":", lw=1, label=f"IC {p_ic:.0f} psi")
ax[1,1].set(xlabel="time", ylabel="pressure (psi)", title="(d) implied within-fracture pressure")
ax[1,1].grid(alpha=0.3); ax[1,1].legend(fontsize=8); ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig.suptitle(f"V1 tensile-SRV (smooth DAS pressure) + DDM shear vs observed (T1→T3)\n"
             f"best s={s_best:.2f}, RMS {rms0:.3f}→{rms_r:.3f} mε ({100*(1-rms_r/rms0):.0f}% var. reduction)",
             fontweight="bold")
fig.savefig(FIG / "das_moose_ddm_match.png", dpi=140); print("Saved:", FIG / "das_moose_ddm_match.png")

fig2, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True, sharey=True)
ext = [mdates.date2num(common[0].to_pydatetime()), mdates.date2num(common[-1].to_pydatetime()), o_md[-1], o_md[0]]
for a, (mat, ttl, lim) in zip(axs, [(O, "observed", 0.1), (model, f"MOOSE×{s_best:.2f}+shear", 0.1), (O-model, "residual", 0.05)]):
    im = a.imshow(mat, aspect="auto", cmap="seismic" if "resid" not in ttl else "bwr",
                  vmin=-lim, vmax=lim, extent=ext, interpolation="none")
    a.xaxis_date(); a.set_title(ttl); a.set_ylim(10500, 10200); a.axvline(mdates.date2num(T2.to_pydatetime()), color="lime", ls="--")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); plt.colorbar(im, ax=a, label="mε")
axs[0].set_ylabel("MD (ft)")
fig2.suptitle("V1 T1→T3: observed vs (scaled MOOSE tensile + DDM shear) vs residual", fontweight="bold")
fig2.savefig(FIG / "das_moose_ddm_waterfall.png", dpi=140); print("Saved:", FIG / "das_moose_ddm_waterfall.png")
