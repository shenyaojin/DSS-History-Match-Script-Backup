"""Rebuild the distributed data reproduction + all figures after a time (T1/T2/T3) update.
No new MOOSE runs are needed: the distributed reproduction uses the STATIC eigenstrain footprint
(repro_tensile), shifted along MD, so only the observation + least squares + figures are recomputed.
White-block fix: the waterfall imshow extent and the x-limits both span [T1, T3] exactly (last window
stretched to T3), so there is no empty axis on the right (profiles are clipped at T3, as in script 107).
"""
import glob
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
FIG = REPO / "figs/tensile_fault_qc/inversion"
DELIV = REPO / "output/inversion_deliverable/figures"
FT = 0.3048; STAR = 10373.4; GL = 30.48
T1 = pd.Timestamp("2025-02-24 12:00"); T2 = pd.Timestamp("2025-02-28 00:00"); T3 = pd.Timestamp("2025-03-04 00:00")
OHW = 28.0; SM = 10.0; SPD = 86400.0

OBS = np.load(REPO / "output/inversion/observation.npz", allow_pickle=True)
md = OBS["md_ft"]; obs = OBS["strain_4h"]; centers = pd.to_datetime(OBS["window_starts"])

# ---- distributed reproduction: shift the static eigenstrain footprint, per-window ridge LS ----
f = sorted(glob.glob(str(REPO / "output/inversion/repro_tensile/*fiber*.csv")))[-1]
d = pd.read_csv(f).sort_values("y"); mdg = STAR + d["y"].to_numpy(float) / FT; s = d["strain_yy"].to_numpy(float) * 1e3
o = np.argsort(mdg); mdg = mdg[o]; s = s[o]
dm = np.median(np.diff(mdg)); w = max(1, int(round(GL / dm)))
g0 = np.convolve(s, np.ones(w) / w, mode="same")                         # GL-averaged footprint at MD 10373


def basis_at(mc):
    return np.interp(md - (mc - STAR), mdg, g0, left=0, right=0)


band_mds = np.arange(10215, 10486, 15.0)
G = np.column_stack([basis_at(m) for m in band_mds])
lam = 1e-6 * np.trace(G.T @ G) / G.shape[1]
amp = np.zeros((len(band_mds), len(centers))); model = np.zeros_like(obs)
for j in range(len(centers)):
    ov = obs[:, j]; v = np.isfinite(ov); Gv = G[v]
    a = np.linalg.solve(Gv.T @ Gv + lam * np.eye(G.shape[1]), Gv.T @ ov[v])
    amp[:, j] = a; model[:, j] = G @ a
vv = np.isfinite(obs); rms0 = np.sqrt(np.nanmean(obs[vv] ** 2)); rmsr = np.sqrt(np.nanmean((obs - model)[vv] ** 2))
vr = 100 * (1 - rmsr / rms0)
print(f"distributed reproduce: {vr:.0f}% var. reduction ({len(band_mds)} bands, {len(centers)} windows)")
np.savez(REPO / "output/inversion/reproduce_distributed.npz", amp=amp, band_mds=band_mds, model=model,
         obs=obs, md=md, centers=[str(c) for c in centers])

# ---- shared plotting (extent + xlim = [T1, T3]) ----
sref = max(np.nanpercentile(np.abs(obs), 95), np.nanpercentile(np.abs(model), 95)) * SM
lim = np.nanpercentile(np.abs(obs), 98)
EXT = [mdates.date2num(T1), mdates.date2num(T3), md[-1], md[0]]
XL = (mdates.date2num(T1), mdates.date2num(T3))


def overlay(ax, prof, color):
    sec = OHW * 3600.0 / sref
    for j, c in enumerate(centers):
        p = prof[:, j]; f2 = np.isfinite(p)
        if f2.any():
            ax.plot(mdates.date2num(c) + p[f2] * sec / SPD, md[f2], color=color, lw=0.8, alpha=0.9, zorder=5, clip_on=True)


def star(ax, prof, color="yellow"):
    sec = OHW * 3600.0 / sref
    j = int(np.argmin(np.abs(centers - pd.Timestamp("2025-03-03 00:00"))))
    p = prof[:, j]; fin = np.flatnonzero(np.isfinite(p)); pi = fin[int(np.nanargmax(np.abs(p[fin])))]
    ax.scatter([mdates.date2num(centers[j]) + p[pi] * sec / SPD], [md[pi]], marker="*", color=color, edgecolor="k", lw=0.8, s=150, zorder=10)


def wf(ax, dat, title):
    im = ax.imshow(dat, aspect="auto", cmap="bwr", vmin=-lim, vmax=lim, extent=EXT, interpolation="bilinear")
    ax.set_xlim(*XL); ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_ylabel("Gold 4-PB MD [ft]"); ax.set_title(title)
    for mt, lb in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
        ax.axvline(mdates.date2num(mt), color="green", ls="--", lw=1.8, zorder=4)
        ax.text(mdates.date2num(mt), 1.01, lb, transform=ax.get_xaxis_transform(), color="green", fontweight="bold", ha="center")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="strain [mε]")
    return im


# Fig A: observed vs reproduced
fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
wf(a1, obs, "Observed DAS strain — 4-hour mean profiles"); overlay(a1, obs, "black"); star(a1, obs)
wf(a2, model, "Reproduced (distributed eigenstrain inversion) — 4-hour mean profiles"); overlay(a2, model, "black"); star(a2, model)
a2.set_xlabel("Time [UTC-7]"); fig.suptitle(f"Observed vs Reproduced DAS strain ({vr:.0f}% var. reduction)", fontweight="bold", fontsize=14)
fig.savefig(FIG / "compare_obs_vs_model.png", dpi=150)

# Fig B: model bg + both profiles + pressure panel
DAS = REPO / "data_fervo/fiberis_format/post_processing/das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
dp = np.load(DAS, allow_pickle=True); p = np.asarray(dp["data"], float); tsec = np.asarray(dp["taxis"], float)
t0 = pd.Timestamp(str(dp["start_time"].item() if hasattr(dp["start_time"], "item") else dp["start_time"])); tt = t0 + pd.to_timedelta(tsec, unit="s")
pfrac = p[0] + 0.699 * (p - p[0])
fig2, (b1, b2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True, constrained_layout=True)
wf(b1, model, "Reproduced strain (background) + 4-hour profiles: observed (black) vs model (red)")
overlay(b1, obs, "black"); overlay(b1, model, "red")
b1.plot([], [], "k-", lw=2, label="observed"); b1.plot([], [], "r-", lw=2, label="model"); b1.legend(loc="lower right")
b2.plot(tt, pfrac, color="#b30000", lw=2.4, label="within-fracture pressure @ fault centre (p*=0.70)")
b2.plot(tt, p, color="tab:orange", lw=1.4, ls="--", label="DAS injection pressure")
b2.axhline(p[0], color="k", ls=":", lw=1)
for mt in (T1, T2, T3):
    b2.axvline(mdates.date2num(mt), color="green", ls="--", lw=1.6)
b2.set_xlim(*XL); b2.set_ylabel("pressure [psi]"); b2.set_xlabel("Time [UTC-7]"); b2.legend(loc="upper left", fontsize=9); b2.grid(alpha=0.3)
b2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig2.suptitle(f"Reproduced DAS strain + fault-centre pressure ({vr:.0f}% var. reduction)", fontweight="bold", fontsize=14)
fig2.savefig(FIG / "compare_model_bg_with_pressure.png", dpi=150)

# Fig C: 3-panel obs/model/residual
fig3, axs = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
for ax, dat, ti in [(axs[0], obs, "observed"), (axs[1], model, "reproduced (distributed eigenstrain)"), (axs[2], obs - model, "residual")]:
    im = ax.imshow(dat, aspect="auto", cmap="seismic", vmin=-0.09, vmax=0.09, extent=EXT, interpolation="bilinear")
    ax.set_xlim(*XL); ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_title(ti); ax.set_xlabel("time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="mε")
axs[0].set_ylabel("MD [ft]")
fig3.suptitle(f"Reproduce real DAS via distributed eigenstrain — {vr:.0f}% var. reduction", fontweight="bold")
fig3.savefig(FIG / "reproduce_distributed.png", dpi=140)

# Fig D: recovered source
fig4, ax4 = plt.subplots(figsize=(13, 6), constrained_layout=True)
im = ax4.imshow(amp, aspect="auto", cmap="seismic", vmin=-np.abs(amp).max(), vmax=np.abs(amp).max(),
                extent=[mdates.date2num(T1), mdates.date2num(T3), band_mds[-1], band_mds[0]], interpolation="bilinear")
ax4.set_xlim(*XL); ax4.xaxis_date(); ax4.set_ylim(10500, 10200); ax4.set_ylabel("band MD [ft]"); ax4.set_xlabel("time")
ax4.axvline(mdates.date2num(T2), color="green", ls="--")
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax4, label="eigenstrain amplitude")
ax4.set_title("Recovered distributed source (opening + / closing −) along MD over time")
fig4.savefig(FIG / "reproduce_source.png", dpi=140)

# Fig E: tensile/shear histories
tens = np.where(amp > 0, amp, 0).sum(0); shear = -np.where(amp < 0, amp, 0).sum(0)
fig5, ax5 = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
ax5.plot(centers, tens / tens.max(), "r-o", ms=3, lw=2, label="tensile (opening) content  ~ within-fracture pressure")
ax5.plot(centers, shear / tens.max(), "b-o", ms=3, lw=2, label="shear content  ~ fault slip")
ax5.axvline(T2, color="green", ls="--", lw=1.8); ax5.text(T2, 1.02, "T2 (shear onset)", transform=ax5.get_xaxis_transform(), color="green", ha="center", fontweight="bold")
ax5.set_xlim(T1, T3); ax5.set_xlabel("Time [UTC-7]"); ax5.set_ylabel("recovered amplitude (normalized)"); ax5.legend(loc="upper left"); ax5.grid(alpha=0.3)
ax5.set_title("Recovered mechanism histories: tensile grows T1→T3, shear turns on at T2")
fig5.savefig(FIG / "reproduce_histories.png", dpi=150)

# Fig F: pressure curve (standalone)
figp, axp = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
axp.plot(tt, p, color="tab:orange", lw=1.6, label="DAS-derived injection pressure (P0 + C·S, C=1.63e7)")
axp.plot(tt, pfrac, color="#b30000", lw=2.4, label="inferred within-fracture pressure (scale p*=0.699)")
axp.axhline(p[0], color="k", ls=":", lw=1, label=f"IC {p[0]:.0f} psi")
for mt, lb in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
    axp.axvline(mt, color="green", ls="--", lw=1.4); axp.text(mdates.date2num(mt), 1.01, lb, transform=axp.get_xaxis_transform(), color="green", fontweight="bold", ha="center")
axp.set_xlim(T1, T3); axp.set_ylabel("pressure [psi]"); axp.set_xlabel("Time [UTC-7]"); axp.legend(loc="upper left"); axp.grid(alpha=0.3)
axp.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
axp.set_title(f"Pressure used: DAS-derived injection pressure {p[0]:.0f}→{p.max():.0f} psi; inferred fracture pressure ×0.70")
figp.savefig(FIG / "pressure_curve.png", dpi=150)

# refresh deliverable gallery copies
for src, dst in [("observation_strain_waterfall.png", "04_observed_strain.png"),
                 ("compare_obs_vs_model.png", "05_reproduction_obs_vs_model.png"),
                 ("compare_model_bg_with_pressure.png", "06_reproduction_with_pressure.png"),
                 ("reproduce_distributed.png", "07_reproduction_3panel.png"),
                 ("reproduce_source.png", "08_recovered_source.png"),
                 ("pressure_curve.png", "09_pressure_curve.png"),
                 ("reproduce_histories.png", "10_tensile_shear_histories.png")]:
    shutil.copy(FIG / src, DELIV / dst)
print("rebuilt all figures + refreshed deliverable gallery")
