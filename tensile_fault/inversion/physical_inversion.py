"""Physical 2-basis inversion (per-4h-window): pressure & slip histories.

Model (fully physical, 2 params per window):
    d(z, t_j) = s_j * MOOSE_tensile(z, t_j)  +  q_j * DDM_shear(z, t_j)
      * MOOSE_tensile : poroelastic strain from the within-fracture pressure (physical
        shape; s_j scales the pressure amplitude in window j).
      * DDM_shear     : strike-slip dislocation projected onto the REAL Gold 4-PB well
        (Pengchao's DDM; q_j scales the slip amplitude in window j). Zero before T2, so
        q_j is only solved for after T2.

This is the honest physical counterpart to the 19-band phenomenological reproduction:
only 2 physical DOF per window (pressure, slip) instead of 19 free eigenstrain bands.
The variance reduction it reaches (vs the phenomenological 92%) measures how much of the
DAS signal the assumed physical SHAPES explain -- the remainder is fracture heterogeneity
the two fixed shapes cannot represent.

Reuses the exact grid alignment of postproc_v1_das_moose_ddm.py.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[3]
PROJ = "v1_tensile_srv_das"
OUT = REPO / "output" / PROJ
FIG = REPO / "figs" / "tensile_fault_qc" / "inversion"
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


# ---- MOOSE fiber tensile strain_yy(MD, t) -----------------------------------
tdf = pd.read_csv(OUT / f"{PROJ}_input_csv.csv")
taxis_s = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(OUT / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis_s))
taxis_s, vpp = taxis_s[:n], vpp[:n]
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

# ---- align observed + DDM shear + MOOSE tensile to a common grid ------------
o_md, o_ct, o_mat = read_prof(OBS_CSV)
s_md, s_ct, s_mat = read_prof(SHEAR_CSV)
common = [t for t in o_ct if t in set(s_ct)]
oi = [list(o_ct).index(t) for t in common]
si = [list(s_ct).index(t) for t in common]
O = o_mat[:, oi]                                                    # (o_md, nt) observed
SH = np.array([np.interp(o_md, s_md, s_mat[:, j]) for j in si]).T   # DDM shear on obs grid
tc_s = np.array([(t - T1).total_seconds() for t in common])
M_t = np.array([np.interp(tc_s, taxis_s, m_strain[k, :]) for k in range(m_strain.shape[0])])
M = np.array([np.interp(o_md, m_md, M_t[:, j]) for j in range(len(common))]).T   # MOOSE tensile
common = pd.DatetimeIndex(common)
nt = len(common)
print(f"aligned {nt} windows on {len(o_md)} depths; obs peak {np.nanmax(np.abs(O)):.4f} mε")

# ---- per-window 2-param physical inversion  [s_j (pressure), q_j (slip)] -----
s_win = np.full(nt, np.nan)
q_win = np.full(nt, np.nan)
model = np.zeros_like(O)
for j in range(nt):
    d = O[:, j]; mt = M[:, j]; sh = SH[:, j]
    v = np.isfinite(d) & np.isfinite(mt) & np.isfinite(sh)
    shear_active = common[j] >= T2 and np.nanmax(np.abs(sh)) > 1e-6
    if shear_active:
        # non-negative least squares: pressure & slip amplitudes are physically >= 0
        A = np.column_stack([mt[v], sh[v]])
        coef, _ = nnls(A, d[v])
        s_j, q_j = float(coef[0]), float(coef[1])
    else:                                                          # pre-T2: tensile only
        s_j = max(0.0, float(np.dot(mt[v], d[v]) / np.dot(mt[v], mt[v])))
        q_j = 0.0
    s_win[j], q_win[j] = s_j, q_j
    model[:, j] = s_j * mt + q_j * sh

v = np.isfinite(O) & np.isfinite(model)
rms0 = np.sqrt(np.nanmean(O[v] ** 2))
rms_r = np.sqrt(np.nanmean((O - model)[v] ** 2))
vr = 100 * (1 - (rms_r / rms0) ** 2)                              # variance reduction
print(f"PER-WINDOW physical: RMS {rms0:.4f}->{rms_r:.4f} mε | variance reduction {vr:.0f}%")
print(f"  pressure scale s_j: {np.nanmin(s_win):.2f}..{np.nanmax(s_win):.2f} (median {np.nanmedian(s_win):.2f})")
qa = q_win[common >= T2]
print(f"  slip scale q_j (post-T2): {np.nanmin(qa):.2f}..{np.nanmax(qa):.2f} (median {np.nanmedian(qa):.2f})")

# ---- implied within-fracture pressure history -------------------------------
pz = np.load(PREPPED_P, allow_pickle=True)
p_das = np.asarray(pz["data"], float); p_ic = float(p_das[0])
p_t = T1 + pd.to_timedelta(np.asarray(pz["taxis"], float), unit="s")
# map each 4h window's s_j onto the pressure curve: fracture Δp(t_j) = s_j * das_Δp(t_j)
das_dp_win = np.interp([(t - T1).total_seconds() for t in common],
                       np.asarray(pz["taxis"], float), p_das) - p_ic
frac_p_win = p_ic + s_win * das_dp_win
print(f"  fracture pressure: IC {p_ic:.0f} -> peak {np.nanmax(frac_p_win):.0f} psi "
      f"(Δp peak {np.nanmax(frac_p_win)-p_ic:.0f} psi)")

# ---- figures -----------------------------------------------------------------
# (1) waterfall obs / model / residual
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True, sharey=True)
ext = [mdates.date2num(common[0].to_pydatetime()), mdates.date2num(common[-1].to_pydatetime()),
       o_md[-1], o_md[0]]
for a, (mat, ttl, lim, cmap) in zip(axs, [
        (O, "observed", 0.1, "seismic"),
        (model, "physical model (s·MOOSE + q·DDM)", 0.1, "seismic"),
        (O - model, "residual", 0.05, "bwr")]):
    im = a.imshow(mat, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, extent=ext, interpolation="none")
    a.xaxis_date(); a.set_title(ttl); a.set_ylim(10500, 10200)
    a.axvline(mdates.date2num(T2.to_pydatetime()), color="lime", ls="--", lw=1)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); plt.colorbar(im, ax=a, label="mε")
axs[0].set_ylabel("MD (ft)")
fig.suptitle(f"Physical per-window inversion  |  variance reduction {vr:.0f}%  "
             f"(2 physical DOF/window: pressure, slip)", fontweight="bold")
fig.savefig(FIG / "physical_inversion_waterfall.png", dpi=140)

# (2) histories: pressure scale / slip scale / fracture pressure
fig2, ax = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True, sharex=True)
ax[0].plot(common, s_win, "o-", color="tab:orange"); ax[0].axvline(T2, color="0.5", ls=":")
ax[0].set(ylabel="pressure scale s_j", title="(a) per-window pressure scale"); ax[0].grid(alpha=0.3)
ax[1].plot(common, q_win, "s-", color="tab:blue"); ax[1].axvline(T2, color="0.5", ls=":")
ax[1].set(ylabel="slip scale q_j", title="(b) per-window slip scale (0 before T2)"); ax[1].grid(alpha=0.3)
ax[2].plot(common, frac_p_win, "o-", color="tab:red"); ax[2].axhline(p_ic, color="k", ls=":", label=f"IC {p_ic:.0f} psi")
ax[2].axvline(T2, color="0.5", ls=":"); ax[2].set(ylabel="pressure (psi)", xlabel="time",
             title="(c) implied within-fracture pressure"); ax[2].grid(alpha=0.3); ax[2].legend()
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig2.suptitle("Physical inversion: recovered pressure & slip histories", fontweight="bold")
fig2.savefig(FIG / "physical_inversion_histories.png", dpi=140)
print("saved:", FIG / "physical_inversion_waterfall.png", "and _histories.png")

# ---- (3) 107-style 4-hour-profile comparison figures (physical model) --------
T2P = pd.Timestamp("2025-02-28 00:00")
OVERLAY_HALF_WIDTH_H, SCALE_MULT, SEC_PER_DAY = 28.0, 10.0, 86400.0
cnum = mdates.date2num([c.to_pydatetime() for c in common])


def _sref(*sets):
    p95 = max(np.nanpercentile(np.abs(p), 95) for p in sets)
    return (p95 or 1.0) * SCALE_MULT


def _overlay(ax, profiles, color, sref):
    sec = OVERLAY_HALF_WIDTH_H * 3600.0 / sref
    for j in range(len(common)):
        prof = profiles[:, j]; fin = np.isfinite(prof)
        if fin.any():
            ax.plot(cnum[j] + prof[fin] * sec / SEC_PER_DAY, o_md[fin],
                    color=color, lw=0.8, alpha=0.9, zorder=5)


def _wf(ax, dat, title, lim):
    ext = [cnum[0], cnum[-1], o_md[-1], o_md[0]]
    im = ax.imshow(dat, aspect="auto", cmap="bwr", vmin=-lim, vmax=lim, extent=ext,
                   interpolation="bilinear")
    ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_ylabel("Gold 4-PB MD [ft]"); ax.set_title(title)
    for mt, lb in [(cnum[0], "T1"), (mdates.date2num(T2P.to_pydatetime()), "T2"), (cnum[-1], "T3")]:
        ax.axvline(mt, color="green", ls="--", lw=1.6, zorder=4)
        ax.text(mt, 1.01, lb, transform=ax.get_xaxis_transform(), color="green",
                fontweight="bold", ha="center")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="strain [mε]")


sref = _sref(O, model)
lim = float(np.nanpercentile(np.abs(O), 98))

# Fig A: observed (top) vs physical model (bottom), 4-hour profile overlays
figA, (a1, a2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
_wf(a1, O, "Observed DAS strain — 4-hour mean profiles", lim); _overlay(a1, O, "black", sref)
_wf(a2, model, "Physical model (s·MOOSE tensile + q·DDM shear) — 4-hour mean profiles", lim)
_overlay(a2, model, "black", sref); a2.set_xlabel("Time [UTC-7]")
figA.suptitle(f"Observed vs physical model DAS strain ({vr:.0f}% variance reduction)",
              fontweight="bold", fontsize=14)
figA.savefig(FIG / "physical_compare_obs_vs_model.png", dpi=150)

# Fig B: model background + both profile families + fault-centre pressure panel (synced x)
figB, (bx, px) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]},
                              sharex=True, constrained_layout=True)
_wf(bx, model, "Physical model (background) + 4-hour profiles: observed (black) vs model (red)", lim)
_overlay(bx, O, "black", sref); _overlay(bx, model, "red", sref)
bx.plot([], [], "k-", lw=2, label="observed 4h profiles"); bx.plot([], [], "r-", lw=2, label="model 4h profiles")
bx.legend(loc="lower right")
px.plot(cnum, frac_p_win, "o-", color="tab:red", lw=1.9, ms=4, label="within-fracture pressure (physical)")
px.axhline(p_ic, color="k", ls=":", lw=1, label=f"IC {p_ic:.0f} psi")
px.axvline(mdates.date2num(T2P.to_pydatetime()), color="green", ls="--", lw=1.4)
px.set_ylabel("pressure [psi]"); px.set_xlabel("Time [UTC-7]"); px.grid(alpha=0.3)
px.legend(loc="upper left", fontsize=8); px.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
figB.suptitle(f"Physical reproduction + within-fracture pressure (4,056→~{np.nanmax(frac_p_win):.0f} psi, synced x)",
              fontweight="bold")
figB.savefig(FIG / "physical_compare_with_pressure.png", dpi=150)
print("saved: physical_compare_obs_vs_model.png and physical_compare_with_pressure.png")

# save the recovered histories
np.savez(REPO / "scripts" / "tensile_fault" / "inversion" / "physical_inversion_result.npz",
         times=np.array([t.value for t in common]), s_win=s_win, q_win=q_win,
         frac_p_win=frac_p_win, vr=vr, rms0=rms0, rms_r=rms_r)
