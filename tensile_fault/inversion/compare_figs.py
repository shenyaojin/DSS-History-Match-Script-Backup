"""Comparison figures in the 107 waterfall + 4-hour-profile-overlay style.

Fig 1 (2 panels): TOP = observed DAS strain, BOTTOM = our reproduced strain (distributed eigenstrain
                  inversion), both with the black 4-hour mean-profile overlays.
Fig 2: the reproduced strain waterfall (background) + BOTH profile sets overlaid -- observed (black)
       and model (red) -- so the two 4-hour profile families can be compared directly.
Also: the pressure curve used (DAS-derived injection pressure and the inferred within-fracture pressure).
"""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
FIG = REPO / "figs/tensile_fault_qc/inversion"
FT = 0.3048
T1 = pd.Timestamp("2025-02-24 15:00"); T2 = pd.Timestamp("2025-02-28 00:00"); T3 = pd.Timestamp("2025-03-03 22:00")
OVERLAY_HALF_WIDTH_H = 28.0
SCALE_MULT = 10.0
SEC_PER_DAY = 86400.0

OBS = np.load(REPO / "output/inversion/observation.npz", allow_pickle=True)
REP = np.load(REPO / "output/inversion/reproduce_distributed.npz", allow_pickle=True)
md = OBS["md_ft"]
obs = OBS["strain_4h"]                                   # [md, win] observed 4h strain
model = REP["model"]                                     # [md, win] reproduced 4h strain
centers = pd.to_datetime(OBS["window_starts"])


def scale_ref(*profile_sets):
    p95 = max(np.nanpercentile(np.abs(p), 95) for p in profile_sets)
    return (p95 or 1.0) * SCALE_MULT


def overlay(ax, profiles, color, sref):
    sec = OVERLAY_HALF_WIDTH_H * 3600.0 / sref
    for j, c in enumerate(centers):
        prof = profiles[:, j]; fin = np.isfinite(prof)
        if fin.any():
            xn = mdates.date2num(c) + prof[fin] * sec / SEC_PER_DAY
            ax.plot(xn, md[fin], color=color, lw=0.8, alpha=0.9, zorder=5)


def star(ax, profiles, sref, color="yellow"):
    sec = OVERLAY_HALF_WIDTH_H * 3600.0 / sref
    j = int(np.argmin(np.abs(centers - pd.Timestamp("2025-03-03 00:00"))))
    prof = profiles[:, j]; fin = np.flatnonzero(np.isfinite(prof))
    pi = fin[int(np.nanargmax(np.abs(prof[fin])))]
    xn = mdates.date2num(centers[j]) + prof[pi] * sec / SEC_PER_DAY
    ax.scatter([xn], [md[pi]], marker="*", color=color, edgecolor="k", lw=0.8, s=150, zorder=10)


def wf(ax, dat, title, lim):
    ext = [mdates.date2num(centers[0]), mdates.date2num(centers[-1]), md[-1], md[0]]
    im = ax.imshow(dat, aspect="auto", cmap="bwr", vmin=-lim, vmax=lim, extent=ext, interpolation="bilinear")
    ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_ylabel("Gold 4-PB MD [ft]"); ax.set_title(title)
    for mt, lb in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
        ax.axvline(mdates.date2num(mt), color="green", ls="--", lw=1.8, zorder=4)
        ax.text(mdates.date2num(mt), 1.01, lb, transform=ax.get_xaxis_transform(), color="green", fontweight="bold", ha="center")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="strain [mε]")
    return im


sref = scale_ref(obs, model)
lim = np.nanpercentile(np.abs(obs), 98)

# ---- Fig 1: observed (top) vs reproduced (bottom), 107 style ----
fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
wf(a1, obs, "Observed DAS strain — 4-hour mean profiles", lim); overlay(a1, obs, "black", sref); star(a1, obs, sref)
wf(a2, model, "Reproduced (distributed eigenstrain inversion) — 4-hour mean profiles", lim)
overlay(a2, model, "black", sref); star(a2, model, sref)
a2.set_xlabel("Time [UTC-7]")
fig.suptitle("Observed vs Reproduced DAS strain (92% var. reduction)", fontweight="bold", fontsize=14)
fig.savefig(FIG / "compare_obs_vs_model.png", dpi=150); print("saved", FIG / "compare_obs_vs_model.png")

# ---- Fig 2: model waterfall + BOTH profile families (observed black, model red) ----
fig2, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
wf(ax, model, "Reproduced strain (background) + overlaid 4-hour profiles: observed (black) vs model (red)", lim)
overlay(ax, obs, "black", sref); overlay(ax, model, "red", sref)
ax.plot([], [], "k-", lw=2, label="observed 4h profiles"); ax.plot([], [], "r-", lw=2, label="model 4h profiles")
ax.legend(loc="lower right"); ax.set_xlabel("Time [UTC-7]")
fig2.savefig(FIG / "compare_model_bg_both_profiles.png", dpi=150); print("saved", FIG / "compare_model_bg_both_profiles.png")

# ---- Pressure curve used ----
DAS = REPO / "data_fervo/fiberis_format/post_processing/das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
d = np.load(DAS, allow_pickle=True); p = np.asarray(d["data"], float); ts = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
tt = t0 + pd.to_timedelta(ts, unit="s"); PSTAR = 0.699
figp, axp = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
axp.plot(tt, p, color="tab:orange", lw=1.6, label="DAS-derived injection pressure (P0 + C·S, C=1.63e7)")
axp.plot(tt, p[0] + PSTAR * (p - p[0]), color="#b30000", lw=2.4, label=f"inferred within-fracture pressure (scale p*={PSTAR})")
axp.axhline(p[0], color="k", ls=":", lw=1, label=f"IC {p[0]:.0f} psi")
for mt, lb in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
    axp.axvline(mt, color="green", ls="--", lw=1.4); axp.text(mt, 1.01, lb, transform=axp.get_xaxis_transform(), color="green", fontweight="bold", ha="center")
axp.set_xlim(T1, T3); axp.set_ylabel("pressure [psi]"); axp.set_xlabel("Time [UTC-7]"); axp.legend(loc="upper left"); axp.grid(alpha=0.3)
axp.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
axp.set_title(f"Pressure used: DAS-derived injection pressure, {p[0]:.0f}→{p.max():.0f} psi; inferred fracture pressure ×{PSTAR}")
figp.savefig(FIG / "pressure_curve.png", dpi=150); print("saved", FIG / "pressure_curve.png")
print(f"DAS pressure: IC {p[0]:.0f} psi, peak {p.max():.0f} psi, Δp_max {p.max()-p[0]:.0f} psi; inferred Δp ×{PSTAR} = {PSTAR*(p.max()-p[0]):.0f} psi")
